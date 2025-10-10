# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torchvision import transforms

from imaginaire.utils import log
from rcm.utils.a2a_cp import MinimalA2AAttnOp
from rcm.utils.selective_activation_checkpoint import CheckpointMode, SACConfig
from rcm.utils.context_parallel import split_inputs_cp
from rcm.utils.jvp_helper import (
    JVP,
    MinimalA2AAttnOpWithT,
    TensorWithT,
    naive_attention_op,
    torch_attention_op,
)

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2

from collections import namedtuple

VideoSize = namedtuple("VideoSize", ["T", "H", "W"])


class VideoPositionEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self._cp_group = None

    def enable_context_parallel(self, process_group: ProcessGroup):
        self._cp_group = process_group

    def disable_context_parallel(self):
        self._cp_group = None

    @property
    def seq_dim(self):
        return 1

    def forward(self, x_B_T_H_W_C: torch.Tensor) -> torch.Tensor:
        """
        With CP, the function assume that the input tensor is already split.
        It delegates the embedding generation to generate_embeddings function.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        if self._cp_group is not None:
            cp_ranks = get_process_group_ranks(self._cp_group)
            cp_size = len(cp_ranks)
            B, T, H, W, C = B_T_H_W_C
            B_T_H_W_C = (B, T * cp_size, H, W, C)
        embeddings = self.generate_embeddings(B_T_H_W_C)

        return self._split_for_context_parallel(embeddings)

    def generate_embeddings(self, B_T_H_W_C: torch.Size):
        raise NotImplementedError

    def _split_for_context_parallel(self, embeddings):
        if self._cp_group is not None:
            embeddings = split_inputs_cp(x=embeddings, seq_dim=self.seq_dim, cp_group=self._cp_group)
        return embeddings


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
    ):
        super().__init__()
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

        self._is_initialized = False

    def cache_parameters(self) -> None:
        if self._is_initialized:
            return

        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().cuda()
        self.dim_spatial_range = torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().cuda() / dim_h
        self.dim_temporal_range = torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().cuda() / dim_t
        self._is_initialized = True

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor.

        Returns:
            Not specified in the original code snippet.
        """
        self.cache_parameters()

        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        B, T, H, W, _ = B_T_H_W_C
        assert (
            H <= self.max_h and W <= self.max_w
        ), f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        freqs_h = torch.outer(self.seq[:H], h_spatial_freqs)
        freqs_w = torch.outer(self.seq[:W], w_spatial_freqs)

        freqs_t = torch.outer(self.seq[:T], temporal_freqs)

        freqs_T_H_W_D = torch.cat(
            [
                repeat(freqs_t, "t d -> t h w d", h=H, w=W),
                repeat(freqs_h, "h d -> t h w d", t=T, w=W),
                repeat(freqs_w, "w d -> t h w d", t=T, h=H),
            ],
            dim=-1,
        )

        return rearrange(freqs_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self):
        return 0


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
        dim=-1,
    )


def rope_apply(x, video_size: VideoSize, freqs, fused):
    """
    Optimized version of rope_apply using flash_attention's rotary embedding implementation.
    This version processes the entire batch at once for efficiency.

    Args:
        x (Tensor): Input tensor with shape [batch_size, seq_len, n_heads, head_dim]
        video_size (VideoSize): Video dimensions with shape [T, H, W]
        freqs (Tensor): Complex frequencies with shape [max_seq_len, head_dim // 2]

    Returns:
        Tensor: Rotary-embedded tensor with same shape as input
    """
    batch_size, seq_len, n_heads, head_dim = x.shape

    # Since all items in the batch share the same grid dimensions, we can use the first item
    T, H, W = video_size
    curr_seq_len = T * H * W

    # Make sure the sequence length matches the grid size
    assert seq_len == curr_seq_len, "Sequence length must be equal to T*H*W"

    freqs = freqs.view(seq_len, head_dim // 2)
    cos = torch.cos(freqs).to(torch.float32)
    sin = torch.sin(freqs).to(torch.float32)

    # Apply the rotation
    if fused:
        rotated = flash_apply_rotary_emb(x.to(torch.float32), cos, sin, interleaved=True, inplace=False)
    else:
        rotated = apply_rotary_emb_torch(x.to(torch.float32), cos, sin, interleaved=True)

    return rotated.to(x.dtype)


class WanRMSNorm(JVP):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        self.weight.data.fill_(1.0)

    def _forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _forward_jvp(self, x: TensorWithT) -> TensorWithT:
        x_withT = x
        x, t_x = x_withT
        out, t_out = torch.func.jvp(self._forward, (x,), (t_x,))
        return (out, t_out.detach())

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(JVP, nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        nn.LayerNorm.__init__(self, dim, eps=eps, elementwise_affine=elementwise_affine)

    def _forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        with amp.autocast("cuda", dtype=torch.float32):
            return nn.LayerNorm.forward(self, x.float()).type_as(x)

    def _forward_jvp(self, x: TensorWithT) -> TensorWithT:
        x_withT = x
        x, t_x = x_withT
        out, t_out = torch.func.jvp(self._forward, (x,), (t_x,))
        return (out, t_out.detach())


class WanSelfAttention(JVP):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, naive_attn=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.qk_norm = qk_norm

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_op = MinimalA2AAttnOp(local_attn=naive_attention_op) if naive_attn else MinimalA2AAttnOp(local_attn=torch_attention_op)
        self.attn_op_withT = MinimalA2AAttnOpWithT()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.q.weight, std=std)
        torch.nn.init.trunc_normal_(self.k.weight, std=std)
        torch.nn.init.trunc_normal_(self.v.weight, std=std)
        torch.nn.init.trunc_normal_(self.o.weight, std=std)
        # zero out bias
        self.q.bias.data.zero_()
        self.k.bias.data.zero_()
        self.v.bias.data.zero_()
        self.o.bias.data.zero_()
        # reset norm weights
        if self.qk_norm:
            self.norm_q.reset_parameters()
            self.norm_k.reset_parameters()

    def _forward(self, x: torch.Tensor, seq_lens, video_size: VideoSize, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            video_size(VideoSize): Shape [T, H, W]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = self.attn_op(rope_apply(q, video_size, freqs, fused=False), rope_apply(k, video_size, freqs, fused=False), v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def _forward_jvp(self, x: TensorWithT, seq_lens, video_size: VideoSize, freqs):
        x_withT = x
        x, t_x = x_withT
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            q = rope_apply(q, video_size, freqs, fused=False)
            k = rope_apply(k, video_size, freqs, fused=False)
            return q, k, v

        (q, k, v), (t_q, t_k, t_v) = torch.func.jvp(qkv_fn, (x,), (t_x,))

        q_withT, k_withT, v_withT = (q, t_q.detach()), (k, t_k.detach()), (v, t_v.detach())

        x_withT = self.attn_op_withT(q_withT, k_withT, v_withT)
        x, t_x = x_withT

        def _fn(x):
            x = x.flatten(2)
            x = self.o(x)
            return x

        # output
        x, t_x = torch.func.jvp(_fn, (x,), (t_x,))
        return (x, t_x.detach())

    def set_context_parallel_group(self, process_group, ranks, stream):
        self.attn_op.set_context_parallel_group(process_group, ranks, stream)


class WanT2VCrossAttention(WanSelfAttention):
    def _forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = self.attn_op(q, k, v)
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def _forward_jvp(self, x: TensorWithT, context, context_lens):
        x_withT = x
        x, t_x = x_withT
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, -1, n, d)
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            return q, k, v

        (q, k, v), (t_q, t_k, t_v) = torch.func.jvp(qkv_fn, (x,), (t_x,))

        q_withT, k_withT, v_withT = (q, t_q.detach()), (k, t_k.detach()), (v, t_v.detach())

        x_withT = self.attn_op_withT(q_withT, k_withT, v_withT)
        x, t_x = x_withT

        def _fn(x):
            x = x.flatten(2)
            x = self.o(x)
            return x

        # output
        x, t_x = torch.func.jvp(_fn, (x,), (t_x,))
        return (x, t_x.detach())


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, naive_attn=False):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, naive_attn)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_op_image = MinimalA2AAttnOp(local_attn=naive_attention_op) if naive_attn else MinimalA2AAttnOp(local_attn=torch_attention_op)
        self.attn_op_image_withT = MinimalA2AAttnOpWithT()

    def init_weights(self):
        super().init_weights()
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.k_img.weight, std=std)
        torch.nn.init.trunc_normal_(self.v_img.weight, std=std)
        # zero out bias
        self.k_img.bias.data.zero_()
        self.v_img.bias.data.zero_()
        # reset norm weights
        if self.qk_norm:
            self.norm_k_img.reset_parameters()

    def _forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = self.attn_op_image(q, k_img, v_img)
        # compute attention
        x = self.attn_op(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x

    def _forward_jvp(self, x: TensorWithT, context, context_lens):
        x_withT = x
        x, t_x = x_withT
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, -1, n, d)
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
            v_img = self.v_img(context_img).view(b, -1, n, d)
            return q, k, v, k_img, v_img

        (q, k, v, k_img, v_img), (t_q, t_k, t_v, t_k_img, t_v_img) = torch.func.jvp(qkv_fn, (x,), (t_x,))

        q_withT, k_withT, v_withT, k_img_withT, v_img_withT = (
            (q, t_q.detach()),
            (k, t_k.detach()),
            (v, t_v.detach()),
            (k_img, t_k_img.detach()),
            (v_img, t_v_img.detach()),
        )

        img_x_withT = self.attn_op_image_withT(q_withT, k_img_withT, v_img_withT)
        x_withT = self.attn_op_withT(q_withT, k_withT, v_withT)
        img_x, t_img_x = img_x_withT
        x, t_x = x_withT

        def _fn(x, img_x):
            x = x.flatten(2)
            img_x = img_x.flatten(2)
            x = x + img_x
            x = self.o(x)
            return x

        # output
        x, t_x = torch.func.jvp(_fn, (x, img_x), (t_x, t_img_x))
        return (x, t_x.detach())


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(JVP):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        naive_attn=False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, naive_attn)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps, naive_attn)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def init_weights(self):
        self.self_attn.init_weights()
        self.cross_attn.init_weights()

        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()

        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.modulation, std=std)

    def _forward(
        self,
        x,
        e,
        seq_lens,
        video_size: VideoSize,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            video_size(VideoSize): Shape [T, H, W]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn((self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x), seq_lens, video_size, freqs)
        with amp.autocast("cuda", dtype=torch.float32):
            x = (x + y * e[2]).type_as(x)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).type_as(x))
            with amp.autocast("cuda", dtype=torch.float32):
                x = (x + y * e[5]).type_as(x)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x

    def _forward_jvp(
        self,
        x: TensorWithT,
        e: TensorWithT,
        seq_lens,
        video_size: VideoSize,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            video_size(VideoSize): Shape [T, H, W]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        x_withT, e_withT = x, e
        x, t_x = x_withT
        e, t_e = e_withT
        assert e.dtype == torch.float32

        def pre_self_attn_fn(x, e):
            e = (self.modulation + e).chunk(6, dim=1)
            z = (self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x)
            return z, e

        with amp.autocast("cuda", dtype=torch.float32):
            (z, e), (t_z, t_e) = torch.func.jvp(pre_self_attn_fn, (x, e), (t_x, t_e))

        assert e[0].dtype == torch.float32
        z_withT, e_withT = (z, t_z.detach()), (e, tuple([_.detach() for _ in t_e]))

        # self-attention
        y_withT = self.self_attn(z_withT, seq_lens, video_size, freqs, withT=True)
        y, t_y = y_withT

        def pre_cross_attn_fn(x, e2, y):
            x = (x + y * e2).type_as(x)
            return x

        with amp.autocast("cuda", dtype=torch.float32):
            x, t_x = torch.func.jvp(pre_cross_attn_fn, (x, e[2], y), (t_x, t_e[2], t_y))
        t_x = t_x.detach()

        z, t_z = torch.func.jvp(self.norm3, (x,), (t_x,))
        z_withT = (z, t_z.detach())
        z_withT = self.cross_attn(z_withT, context, context_lens, withT=True)
        z, t_z = z_withT

        def post_cross_attn_fn(x, z, e3, e4, e5):
            x = x + z
            y = self.ffn((self.norm2(x).float() * (1 + e4) + e3).type_as(x))
            x = (x + y * e5).type_as(x)
            return x

        with amp.autocast("cuda", dtype=torch.float32):
            x, t_x = torch.func.jvp(post_cross_attn_fn, (x, z, e[3], e[4], e[5]), (t_x, t_z, t_e[3], t_e[4], t_e[5]))
        return (x, t_x.detach())


class Head(JVP):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def init_weights(self):
        self.norm.reset_parameters()

        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.modulation, std=std)
        torch.nn.init.trunc_normal_(self.head.weight, std=std)
        self.head.bias.data.zero_()

    def _forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x

    def _forward_jvp(self, x: TensorWithT, e: TensorWithT):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        x_withT, e_withT = x, e
        x, t_x = x_withT
        e, t_e = e_withT
        assert e.dtype == torch.float32

        def fn(x, e):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
            return x

        with amp.autocast("cuda", dtype=torch.float32):
            x, t_x = torch.func.jvp(fn, (x, e), (t_x, t_e))
        return (x, t_x.detach())


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            WanLayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            WanLayerNorm(out_dim),
        )
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def init_weights(self):
        self.proj[0].reset_parameters()
        self.proj[1].reset_parameters()
        self.proj[3].reset_parameters()
        self.proj[4].reset_parameters()

        if hasattr(self, "emb_pos"):
            self.emb_pos.data.zero_()

    def forward(self, image_embeds):
        if hasattr(self, "emb_pos"):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel_JVP(JVP):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        concat_padding_mask: bool = False,
        sac_config: SACConfig = SACConfig(),
        naive_attn: bool = False,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            concat_padding_mask (`bool`, *optional*, defaults to False):
                Enable concat padding mask
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "flf2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.concat_padding_mask = concat_padding_mask
        self.use_crossattn_projection = False

        # embeddings
        in_dim = in_dim + 1 if self.concat_padding_mask else in_dim
        self.patch_embedding = nn.Linear(in_dim * patch_size[0] * patch_size[1] * patch_size[2], dim)

        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    naive_attn,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0

        d = dim // num_heads

        self.rope_position_embedding = VideoRopePosition3DEmb(
            head_dim=d,
            len_h=128,
            len_w=128,
            len_t=32,
        )

        if model_type == "i2v" or model_type == "flf2v":
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == "flf2v")

        # initialize weights
        self.init_weights()

        self.enable_selective_checkpoint(sac_config)

    def _forward(
        self,
        x_B_C_T_H_W,
        timesteps_B_T,
        crossattn_emb,
        seq_len=None,
        frame_cond_crossattn_emb_B_L_D=None,
        y_B_C_T_H_W=None,
        padding_mask: Optional[torch.Tensor] = None,
        is_uncond=False,
        slg_layers=None,
        **kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x_B_C_T_H_W (Tensor):
                Input video tensor with shape [B, C_in, T, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            frame_cond_crossattn_emb_B_L_D (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y_B_C_T_H_W (Tensor, *optional*):
                Conditional video inputs for image-to-video mode, shape [B, C_in, T, H, W]

        Returns:
            Tensor:
                Denoised video tensor with shape [B, C_out, T, H / 8, W / 8]
        """

        assert timesteps_B_T.shape[1] == 1
        t_B = timesteps_B_T[:, 0]
        del kwargs
        if self.model_type == "i2v" or self.model_type == "flf2v":
            assert frame_cond_crossattn_emb_B_L_D is not None and y_B_C_T_H_W is not None

        if y_B_C_T_H_W is not None:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, y_B_C_T_H_W], dim=1)

        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1)

        # embeddings
        x_B_T_H_W_D = rearrange(
            x_B_C_T_H_W,
            "b c (t kt) (h kh) (w kw) -> b t h w (c kt kh kw)",
            kt=self.patch_size[0],
            kh=self.patch_size[1],
            kw=self.patch_size[2],
        )

        x_B_T_H_W_D = self.patch_embedding(x_B_T_H_W_D)

        video_size = VideoSize(T=x_B_T_H_W_D.shape[1], H=x_B_T_H_W_D.shape[2], W=x_B_T_H_W_D.shape[3])
        x_B_L_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
        seq_lens = torch.tensor([u.size(0) for u in x_B_L_D], dtype=torch.long)
        seq_len = seq_lens.max().item()
        assert seq_lens.max() == seq_len

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            e_B_D = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_B).float())
            e0_B_6_D = self.time_projection(e_B_D).unflatten(1, (6, self.dim))
            assert e_B_D.dtype == torch.float32 and e0_B_6_D.dtype == torch.float32

        # context
        context_lens = None
        context_B_L_D = self.text_embedding(crossattn_emb)

        if frame_cond_crossattn_emb_B_L_D is not None:
            context_clip = self.img_emb(frame_cond_crossattn_emb_B_L_D)  # bs x 257 (x2) x dim
            context_B_L_D = torch.concat([context_clip, context_B_L_D], dim=1)

        # arguments
        kwargs = dict(
            e=e0_B_6_D,
            seq_lens=seq_lens,
            video_size=video_size,
            freqs=self.rope_position_embedding(x_B_T_H_W_D),
            context=context_B_L_D,
            context_lens=context_lens,
        )

        for block_idx, block in enumerate(self.blocks):
            if slg_layers is not None and block_idx in slg_layers and is_uncond:
                continue
            x_B_L_D = block(x_B_L_D, **kwargs)

        # head
        x_B_L_D = self.head(x_B_L_D, e_B_D)

        # unpatchify
        t, h, w = video_size
        x_B_C_T_H_W = rearrange(
            x_B_L_D,
            "b (t h w) (nt nh nw d) -> b d (t nt) (h nh) (w nw)",
            nt=self.patch_size[0],
            nh=self.patch_size[1],
            nw=self.patch_size[2],
            t=t,
            h=h,
            w=w,
            d=self.out_dim,
        )

        return x_B_C_T_H_W

    def _forward_jvp(
        self,
        x_B_C_T_H_W: TensorWithT,
        timesteps_B_T: TensorWithT,
        crossattn_emb,
        seq_len=None,
        frame_cond_crossattn_emb_B_L_D=None,
        y_B_C_T_H_W=None,
        padding_mask: Optional[torch.Tensor] = None,
        is_uncond=False,
        slg_layers=None,
        **kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x_B_C_T_H_W (Tensor):
                Input video tensor with shape [B, C_in, T, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            frame_cond_crossattn_emb_B_L_D (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y_B_C_T_H_W (Tensor, *optional*):
                Conditional video inputs for image-to-video mode, shape [B, C_in, T, H, W]

        Returns:
            Tensor:
                Denoised video tensor with shape [B, C_out, T, H / 8, W / 8]
        """
        x_B_C_T_H_W_withT, timesteps_B_T_withT = x_B_C_T_H_W, timesteps_B_T
        x_B_C_T_H_W, t_x_B_C_T_H_W = x_B_C_T_H_W_withT
        timesteps_B_T, t_timesteps_B_T = timesteps_B_T_withT

        assert timesteps_B_T.shape[1] == 1
        t_B = timesteps_B_T[:, 0]
        t_t_B = t_timesteps_B_T[:, 0]
        del kwargs
        if self.model_type == "i2v" or self.model_type == "flf2v":
            assert frame_cond_crossattn_emb_B_L_D is not None and y_B_C_T_H_W is not None

        if y_B_C_T_H_W is not None:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, y_B_C_T_H_W], dim=1)
            t_x_B_C_T_H_W = torch.cat([t_x_B_C_T_H_W, torch.zeros_like(y_B_C_T_H_W)], dim=1)

        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1)
            t_x_B_C_T_H_W = torch.cat(
                [t_x_B_C_T_H_W, torch.zeros_like(padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1))],
                dim=1,
            )

        # embeddings
        x_B_T_H_W_D = rearrange(
            x_B_C_T_H_W,
            "b c (t kt) (h kh) (w kw) -> b t h w (c kt kh kw)",
            kt=self.patch_size[0],
            kh=self.patch_size[1],
            kw=self.patch_size[2],
        )
        t_x_B_T_H_W_D = rearrange(
            t_x_B_C_T_H_W,
            "b c (t kt) (h kh) (w kw) -> b t h w (c kt kh kw)",
            kt=self.patch_size[0],
            kh=self.patch_size[1],
            kw=self.patch_size[2],
        )

        x_B_T_H_W_D, t_x_B_T_H_W_D = torch.func.jvp(self.patch_embedding, (x_B_T_H_W_D,), (t_x_B_T_H_W_D,))

        video_size = VideoSize(T=x_B_T_H_W_D.shape[1], H=x_B_T_H_W_D.shape[2], W=x_B_T_H_W_D.shape[3])
        x_B_L_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")
        t_x_B_L_D = rearrange(t_x_B_T_H_W_D, "b t h w d -> b (t h w) d")
        seq_lens = torch.tensor([u.size(0) for u in x_B_L_D], dtype=torch.long)
        seq_len = seq_lens.max().item()
        assert seq_lens.max() == seq_len

        def time_embed_fn(t):
            e_B_D = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0_B_6_D = self.time_projection(e_B_D).unflatten(1, (6, self.dim))
            return e_B_D, e0_B_6_D

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            (e_B_D, e0_B_6_D), (t_e_B_D, t_e0_B_6_D) = torch.func.jvp(time_embed_fn, (t_B,), (t_t_B,))

        assert e_B_D.dtype == torch.float32 and e0_B_6_D.dtype == torch.float32

        x_B_L_D_withT, e_B_D_withT, e0_B_6_D_withT = (
            (x_B_L_D, t_x_B_L_D.detach()),
            (e_B_D, t_e_B_D.detach()),
            (e0_B_6_D, t_e0_B_6_D.detach()),
        )

        # context
        context_lens = None
        context_B_L_D = self.text_embedding(crossattn_emb)

        if frame_cond_crossattn_emb_B_L_D is not None:
            context_clip = self.img_emb(frame_cond_crossattn_emb_B_L_D)  # bs x 257 (x2) x dim
            context_B_L_D = torch.concat([context_clip, context_B_L_D], dim=1)

        # arguments
        kwargs = dict(
            e=e0_B_6_D_withT,
            seq_lens=seq_lens,
            video_size=video_size,
            freqs=self.rope_position_embedding(x_B_T_H_W_D),
            context=context_B_L_D,
            context_lens=context_lens,
        )

        for block_idx, block in enumerate(self.blocks):
            if slg_layers is not None and block_idx in slg_layers and is_uncond:
                continue
            x_B_L_D_withT = block(x_B_L_D_withT, **kwargs, withT=True)

        # head
        x_B_L_D_withT = self.head(x_B_L_D_withT, e_B_D_withT, withT=True)
        x_B_L_D, t_x_B_L_D = x_B_L_D_withT

        # unpatchify
        t, h, w = video_size
        x_B_C_T_H_W = rearrange(
            x_B_L_D,
            "b (t h w) (nt nh nw d) -> b d (t nt) (h nh) (w nw)",
            nt=self.patch_size[0],
            nh=self.patch_size[1],
            nw=self.patch_size[2],
            t=t,
            h=h,
            w=w,
            d=self.out_dim,
        )
        t_x_B_C_T_H_W = rearrange(
            t_x_B_L_D,
            "b (t h w) (nt nh nw d) -> b d (t nt) (h nh) (w nw)",
            nt=self.patch_size[0],
            nh=self.patch_size[1],
            nw=self.patch_size[2],
            t=t,
            h=h,
            w=w,
            d=self.out_dim,
        )

        return (x_B_C_T_H_W, t_x_B_C_T_H_W.detach())

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for block in self.blocks:
            block.init_weights()
        self.head.init_weights()

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.zeros_(self.patch_embedding.bias)

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.time_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        if self.head.head.bias is not None:
            nn.init.zeros_(self.head.head.bias)

    def fully_shard(self, mesh):
        for i, block in enumerate(self.blocks):
            fully_shard(block, mesh=mesh, reshard_after_forward=True)
        fully_shard(self.head, mesh=mesh, reshard_after_forward=False)
        fully_shard(self.text_embedding, mesh=mesh, reshard_after_forward=True)
        # fully_shard(self.time_embedding, mesh=mesh, reshard_after_forward=True)
        # fully_shard(self.patch_embedding, mesh=mesh, reshard_after_forward=True)

    def disable_context_parallel(self):
        # pos_embedder
        self.rope_position_embedding.disable_context_parallel()
        # attention
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        # pos_embedder
        self.rope_position_embedding.enable_context_parallel(process_group=process_group)
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.self_attn.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = True

    @property
    def is_context_parallel_enabled(self):
        return self._is_context_parallel_enabled

    def enable_selective_checkpoint(self, sac_config: SACConfig):
        if sac_config.mode == CheckpointMode.NONE:
            return self

        log.info(f"Enable selective checkpoint with mm_only, for every {sac_config.every_n_blocks} blocks. Total blocks: {len(self.blocks)}")
        _context_fn = sac_config.get_context_fn()
        for block_id, block in self.blocks.named_children():
            if int(block_id) % sac_config.every_n_blocks == 0:
                log.info(f"Enable selective checkpoint for block {block_id}")
                block = ptd_checkpoint_wrapper(
                    block,
                    context_fn=_context_fn,
                    preserve_rng_state=False,
                )
                self.blocks.register_module(block_id, block)
        self.register_module(
            "head",
            ptd_checkpoint_wrapper(
                self.head,
                context_fn=_context_fn,
                preserve_rng_state=False,
            ),
        )

        return self
