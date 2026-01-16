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

"""
Flash Attention v2 with JVP
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Taken from https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py

Modified 2025/03; Author: Kaiwen Zheng (zkwthu@gmail.com)

(1) Simplified version, combining Triton forward and official backward

(2) Support Jacobian-vector-product (JVP) computation in the forward pass

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

Citation:

@article{zheng2025rcm,
  title={Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency},
  author={Zheng, Kaiwen and Wang, Yuji and Ma, Qianli and Chen, Huayu and Zhang, Jintao and Balaji, Yogesh and Chen, Jianfei and Liu, Ming-Yu and Zhu, Jun and Zhang, Qinsheng},
  journal={arXiv preprint arXiv:2510.08431},
  year={2025}
}
"""

import torch
import triton
import triton.language as tl
from einops import rearrange
from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_varlen_backward

import inspect


def _get_param_names(fn):
    # Works for normal Python functions; for some C++/pybind builtins,
    # inspect.signature might fail, then we fall back to __text_signature__.
    try:
        return set(inspect.signature(fn).parameters.keys())
    except Exception:
        ts = getattr(fn, "__text_signature__", "") or ""
        # crude but usually good enough
        for ch in "(),*:":  # strip common punctuation
            ts = ts.replace(ch, " ")
        return set(ts.split())


def _make_flash_bwd_caller(flash_bwd_fn):
    params = _get_param_names(flash_bwd_fn)

    def call(
        *pos_args,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        **extra_kwargs,
    ):
        ws_left, ws_right = window_size

        kw = dict(
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            window_size_left=ws_left,
            window_size_right=ws_right,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )
        kw.update(extra_kwargs)

        kw = {k: v for k, v in kw.items() if k in params}

        return flash_bwd_fn(*pos_args, **kw)

    return call


_flash_bwd = _make_flash_bwd_caller(_flash_attn_backward)
_flash_varlen_bwd = _make_flash_bwd_caller(_flash_attn_varlen_backward)

DEVICE = "cuda"

configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [16, 32, 64]
    for s in [3, 4, 7]
    for w in [4, 8]
]


@triton.autotune(configs, key=["SEQ_LEN_Q", "SEQ_LEN_KV", "HEAD_DIM_QK", "HEAD_DIM_V"])
@triton.jit
def _attn_fwd(
    Q, K, V,
    tQ, tK, tV,
    sm_scale,                 # fp32 scalar
    LSE,                      # [B, H, SEQ_LEN_Q] fp32 (natural log)
    O, tO,                    # [B, H, SEQ_LEN_Q, HEAD_DIM_V]
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B: tl.constexpr, H: tl.constexpr,
    SEQ_LEN_Q: tl.constexpr, SEQ_LEN_KV: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr, HEAD_DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m  = tl.program_id(0).to(tl.int64)   # query block id
    pid_bh = tl.program_id(1).to(tl.int64)   # fused batch-head

    off_b = pid_bh // H
    off_h = pid_bh %  H

    # offsets
    offs_m  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)         # [BM]
    offs_n  = tl.arange(0, BLOCK_N)                           # [BN]
    offs_dq = tl.arange(0, HEAD_DIM_QK)                       # [Dq]
    offs_dv = tl.arange(0, HEAD_DIM_V)                        # [Dv]

    # base pointers for this (b, h)
    q_base = off_b * stride_qb + off_h * stride_qh
    k_base = off_b * stride_kb + off_h * stride_kh
    v_base = off_b * stride_vb + off_h * stride_vh
    o_base = off_b * stride_ob + off_h * stride_oh

    # Q / tQ: [BM, Dq]
    Q_ptrs  = Q  + q_base + offs_m[:, None] * stride_qm + offs_dq[None, :] * stride_qd
    tQ_ptrs = tQ + q_base + offs_m[:, None] * stride_qm + offs_dq[None, :] * stride_qd
    m_mask = offs_m < SEQ_LEN_Q
    q  = tl.load(Q_ptrs,  mask=m_mask[:, None], other=0.0)
    tq = tl.load(tQ_ptrs, mask=m_mask[:, None], other=0.0)

    # streaming softmax stats (base-2)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)  # row max in log2 domain
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # row sum of exp2
    # JVP-specific accumulators (Algorithm 2: O, A, r, B)
    o_i = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)    # O~ (unnormalized)
    A_i = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)    # A~ = P~ tV
    B_i = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)    # B~ = H~ V
    r_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # r~ = rowsum(H~)

    qk_scale_log2 = sm_scale * 1.4426950408889634  # scale for exp2

    # loop over K/V blocks
    for start_n in range(0, SEQ_LEN_KV, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        n_mask = n_idx < SEQ_LEN_KV

        # K / tK in transposed tile layout: [Dq, BN]
        K_ptrs  = K  + k_base + n_idx[None, :] * stride_kn + offs_dq[:, None] * stride_kd
        tK_ptrs = tK + k_base + n_idx[None, :] * stride_kn + offs_dq[:, None] * stride_kd
        k  = tl.load(K_ptrs,  mask=n_mask[None, :], other=0.0)
        tk = tl.load(tK_ptrs, mask=n_mask[None, :], other=0.0)

        # scores in log2 domain: qk = (QK^T) * sm_scale / ln2
        qk = tl.dot(q, k).to(tl.float32) * qk_scale_log2
        qk = tl.where(n_mask[None, :], qk, -float("inf"))

        # update streaming max + exp2 sums (FlashAttention-2 core)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])                   # P~ (unnormalized)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)                       # rescale factor

        l_i = l_i * alpha + l_ij
        o_i = o_i * alpha[:, None]
        A_i = A_i * alpha[:, None]
        B_i = B_i * alpha[:, None]
        r_i = r_i * alpha

        # V / tV: [BN, Dv]
        V_ptrs  = V  + v_base + n_idx[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        tV_ptrs = tV + v_base + n_idx[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        v  = tl.load(V_ptrs,  mask=n_mask[:, None], other=0.0)
        tv = tl.load(tV_ptrs, mask=n_mask[:, None], other=0.0)

        # ---- JVP pieces ----
        # tS = (tQ K^T + Q tK^T) * sm_scale
        tS = (tl.dot(tq, k).to(tl.float32) + tl.dot(q, tk).to(tl.float32)) * sm_scale
        tS = tl.where(n_mask[None, :], tS, 0.0)

        # H~ = P~ ⊙ tS  (use unnormalized P~)
        Htilde = p * tS
        r_i += tl.sum(Htilde, axis=1)

        # accumulate O~, A~, B~ (Algorithm 2)
        p = p.to(v.dtype)
        Htilde = Htilde.to(v.dtype)
        o_i = tl.dot(p, v, o_i)
        A_i = tl.dot(p, tv, A_i)
        B_i = tl.dot(Htilde, v, B_i)

        m_i = m_ij

    # ---- epilogue ----
    # O = O~ / l,  LSE = (m + log2(l)) * ln2
    # tO = (A~ + B~ - diag(r) O) / l

    inv_l_i = 1.0 / l_i
    O_i  = o_i * inv_l_i[:, None]
    A_i  = A_i * inv_l_i[:, None]
    B_i  = B_i * inv_l_i[:, None]
    mu_i = r_i * inv_l_i
    tO_i = A_i + B_i - mu_i[:, None] * O_i

    lse = (m_i + tl.math.log2(l_i)) * 0.6931471805599453

    # store
    O_ptrs  = O  + o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od
    tO_ptrs = tO + o_base + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od
    LSE_ptrs = LSE + pid_bh * SEQ_LEN_Q + offs_m

    tl.store(O_ptrs,  O_i.to(O.type.element_ty),   mask=m_mask[:, None])
    tl.store(tO_ptrs, tO_i.to(tO.type.element_ty), mask=m_mask[:, None])
    tl.store(LSE_ptrs, lse, mask=m_mask)


def generate_qkv(q, k, v):
    """
    Arguments:
        q: (batch_size, nheads, seqlen_q, d)
        k: (batch_size, nheads_k, seqlen_k, d)
        v: (batch_size, nheads_k, seqlen_k, d)
    """
    batch_size, _, seqlen_q, d = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    assert k.shape == (batch_size, nheads_k, seqlen_k, d)
    assert v.shape == (batch_size, nheads_k, seqlen_k, d)

    def unpad_fn(x):
        return rearrange(x, "b h s d -> (b s) h d")

    def lse_unpad_fn(x):
        return rearrange(x, "b h s -> (b s) h")

    def pad_fn(x):
        return rearrange(x, "(b s) h d -> b h s d", b=batch_size)

    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
    max_seqlen_q = seqlen_q

    cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=q.device)
    max_seqlen_k = seqlen_k

    return (
        unpad_fn,
        lse_unpad_fn,
        pad_fn,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )


class _attention(torch.autograd.Function):
    """
    Arguments:
        q, tq: (batch_size, nheads, seqlen_q, d_qk)
        k, tk: (batch_size, nheads, seqlen_kv, d_qk)
        v, tv: (batch_size, nheads, seqlen_kv, d_v)
    Returns:
        o, to: (batch_size, nheads, seqlen_q, d_v)

    Backward is only supported when d_qk=d_v.
    """

    @staticmethod
    def forward(ctx, q, k, v, tq, tk, tv, sm_scale=None):
        is_grad = any(x.requires_grad for x in [q, k, v])
        # shape constraints
        assert q.shape[:-2] == k.shape[:-2] and k.shape[:-2] == v.shape[:-2]
        assert k.shape[-2] == v.shape[-2] and q.shape[-1] == k.shape[-1]
        B, H = q.shape[:-2]
        SEQ_LEN_Q, SEQ_LEN_KV = q.shape[-2], k.shape[-2]
        HEAD_DIM_QK, HEAD_DIM_V = q.shape[-1], v.shape[-1]
        assert HEAD_DIM_QK in {16, 32, 64, 128, 256}
        assert HEAD_DIM_V in {16, 32, 64, 128, 256}
        assert tq.shape == q.shape and tk.shape == k.shape and tv.shape == v.shape
        assert tq.stride() == q.stride() and tk.stride() == k.stride() and tv.stride() == v.stride()
        if sm_scale is None:
            sm_scale = HEAD_DIM_QK ** (-0.5)
        o = torch.empty((B, H, SEQ_LEN_Q, HEAD_DIM_V), device=q.device, dtype=q.dtype)
        to = torch.empty_like(o)

        M = torch.empty((B, H, SEQ_LEN_Q), device=q.device, dtype=torch.float32)

        def grid(args):
            return (triton.cdiv(SEQ_LEN_Q, args["BLOCK_M"]), B * H, 1)

        _attn_fwd[grid](
            q, k, v,
            tq, tk, tv,
            sm_scale,
            M,
            o, to,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H,
            SEQ_LEN_Q, SEQ_LEN_KV,
            HEAD_DIM_QK, HEAD_DIM_V,
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, o, M)
            ctx.sm_scale = sm_scale
        return o, to

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        assert q.shape[-1] == k.shape[-1] and k.shape[-1] == v.shape[-1], "Backward not supported with different headdim."
        # flash_attn uses the shape (batch_size, seqlen, nheads, headdim)
        # torch.nn.functional.scaled_dot_product_attention and this implementation use (batch_size, nheads, seqlen, headdim)
        if q.shape[-2] == k.shape[-2]:
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            _flash_bwd(
                dout.transpose(1, 2),
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                out.transpose(1, 2),
                softmax_lse,
                dq.transpose(1, 2),
                dk.transpose(1, 2),
                dv.transpose(1, 2),
                dropout_p=0.0,
                softmax_scale=ctx.sm_scale,
                causal=False,
            )
        else:
            unpad_fn, lse_unpad_fn, pad_fn, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = generate_qkv(q, k, v)
            q_unpad, k_unpad, v_unpad = unpad_fn(q), unpad_fn(k), unpad_fn(v)
            dq, dk, dv = torch.empty_like(q_unpad), torch.empty_like(k_unpad), torch.empty_like(v_unpad)
            _flash_varlen_bwd(
                unpad_fn(dout),
                q_unpad,
                k_unpad,
                v_unpad,
                unpad_fn(out),
                lse_unpad_fn(softmax_lse),
                dq,
                dk,
                dv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=ctx.sm_scale,
                causal=False,
            )
            dq, dk, dv = pad_fn(dq), pad_fn(dk), pad_fn(dv)
        return dq, dk, dv, None, None, None, None, None


attention = _attention.apply


def _test_fwd_bwd(B, H, SEQ_LEN, HEAD_DIM, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((B, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((B, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((B, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    tq = torch.zeros_like(q)
    tk = torch.zeros_like(k)
    tv = torch.zeros_like(v)
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    # if causal:
    #     p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = attention(q, k, v, tq, tk, tv, sm_scale)[0].to(dtype)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    rtol = 2e-2 if dtype == torch.bfloat16 else 0
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    torch.testing.assert_close(ref_dq, tri_dq, atol=1e-2, rtol=rtol / 2)
    torch.testing.assert_close(ref_dk, tri_dk, atol=1e-2, rtol=rtol / 2)
    torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=rtol)


def test_fwd_bwd():
    for shape in [(1, 2, 1024, 64), (1, 2, 999, 64)]:
        for dtype in [torch.float16, torch.bfloat16]:
            _test_fwd_bwd(*shape, dtype)
            print(f"Shape={shape}, Dtype={dtype} Passed (SA fwd/bwd).")


def _test_jvp(B, H, SEQ_LEN, HEAD_DIM, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((B, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((B, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((B, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    tq = torch.empty((B, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    tk = torch.empty((B, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    tv = torch.empty((B, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    sm_scale = 0.5

    def naive_attention(q, k, v):
        # reference implementation
        M = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=DEVICE))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        # if causal:
        #     p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).to(dtype)
        ref_out = torch.matmul(p, v)
        return ref_out

    _, ref_tout = torch.func.jvp(naive_attention, (q, k, v), (tq, tk, tv))
    # triton implementation
    tri_tout = attention(q, k, v, tq, tk, tv, sm_scale)[1].to(dtype)
    # compare
    torch.testing.assert_close(ref_tout, tri_tout, atol=1e-2, rtol=1e-2)


def test_jvp():
    for shape in [(1, 2, 1024, 64), (1, 2, 999, 64)]:
        for dtype in [torch.float16, torch.bfloat16]:
            _test_jvp(*shape, dtype)
            print(f"Shape={shape}, Dtype={dtype} Passed (SA JVP).")



BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["SEQ_LEN"],
            x_vals=[2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=["triton-fp16", "flash"],
            line_names=["Triton [FP16]", "FlashAttn-2"],
            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
            ylabel="TFLOPS",
            plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "mode": mode,
            },
        )
    )


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, SEQ_LEN, HEAD_DIM, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        tq = torch.zeros_like(q)
        tk = torch.zeros_like(k)
        tv = torch.zeros_like(v)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, tq, tk, tv, sm_scale)[0]
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == "flash":
        from flash_attn.flash_attn_interface import flash_attn_func

        q = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=False)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    # there are 2 matmuls in the forward pass
    flops_per_matmul = 2.0 * BATCH * H * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if mode == "bwd":
        # there are 5 matmuls in the backward pass
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    elif "triton" in provider:
        # there are 6 matmuls in the forward pass with JVP computation
        total_flops *= 3
    return total_flops * 1e-12 / (ms * 1e-3)


def _test_fwd_bwd_ca(B, H, SEQ_LEN_Q, SEQ_LEN_KV, HEAD_DIM, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((B, H, SEQ_LEN_Q, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((B, H, SEQ_LEN_KV, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((B, H, SEQ_LEN_KV, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    tq = torch.zeros_like(q)
    tk = torch.zeros_like(k)
    tv = torch.zeros_like(v)
    sm_scale = 0.5
    dout = torch.randn((B, H, SEQ_LEN_Q, HEAD_DIM), device=q.device, dtype=q.dtype)
    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = attention(q, k, v, tq, tk, tv, sm_scale)[0].to(dtype)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = 2e-2 if dtype == torch.bfloat16 else 0
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    torch.testing.assert_close(ref_dq, tri_dq, atol=atol, rtol=rtol / 2)
    torch.testing.assert_close(ref_dk, tri_dk, atol=atol, rtol=rtol / 2)
    torch.testing.assert_close(ref_dv, tri_dv, atol=atol, rtol=rtol)


def test_fwd_bwd_ca():
    for shape in [(1, 2, 256, 1024, 128), (1, 2, 1024, 256, 128), (1, 2, 1024, 512, 64), (1, 2, 1000, 515, 64)]:
        for dtype in [torch.float16, torch.bfloat16]:
            _test_fwd_bwd_ca(*shape, dtype)
            print(f"Shape={shape}, Dtype={dtype} Passed (CA fwd/bwd with the same headdim).")


def _test_jvp_ca(B, H, SEQ_LEN_Q, SEQ_LEN_KV, HEAD_DIM_QK, HEAD_DIM_V, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.empty((B, H, SEQ_LEN_Q, HEAD_DIM_QK), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    k = torch.empty((B, H, SEQ_LEN_KV, HEAD_DIM_QK), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    v = torch.empty((B, H, SEQ_LEN_KV, HEAD_DIM_V), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_()
    tq = torch.empty((B, H, SEQ_LEN_Q, HEAD_DIM_QK), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    tk = torch.empty((B, H, SEQ_LEN_KV, HEAD_DIM_QK), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    tv = torch.empty((B, H, SEQ_LEN_KV, HEAD_DIM_V), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    sm_scale = 0.5

    def naive_attention(q, k, v):
        # reference implementation
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        p = torch.softmax(p.float(), dim=-1).to(dtype)
        ref_out = torch.matmul(p, v)
        return ref_out

    ref_out, ref_tout = torch.func.jvp(naive_attention, (q, k, v), (tq, tk, tv))
    # triton implementation
    tri_out, tri_tout = attention(q, k, v, tq, tk, tv, sm_scale)
    # compare
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    torch.testing.assert_close(ref_tout, tri_tout, atol=atol, rtol=1e-2)


def test_jvp_ca():
    for shape in [
        (1, 2, 256, 1024, 64, 128),
        (1, 2, 1000, 15, 128, 32),
        (1, 2, 512, 512, 16, 32),
        (1, 2, 515, 999, 16, 32),
    ]:
        for dtype in [torch.float16, torch.bfloat16]:
            _test_jvp_ca(*shape, dtype)
            print(f"Shape={shape}, Dtype={dtype} Passed (CA fwd/JVP with different headdim).")


if __name__ == "__main__":
    test_fwd_bwd()
    test_jvp()
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
    test_fwd_bwd_ca()
    test_jvp_ca()
