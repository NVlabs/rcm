# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import functools

import pytest
import torch
from einops import repeat

from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import instantiate
from rcm.networks.wan2pt1 import WanModel
from rcm.networks.wan2pt1_jvp import CheckpointMode, SACConfig, WanModel_JVP

"""
Usage:
    pytest -s rcm/networks/wan2pt1_jvp_test.py
"""

N_HEADS = 8
HEAD_DIM = 32
mini_net = L(WanModel)(
    model_type="t2v",
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,
    dim=2048,
    ffn_dim=8192,
    freq_dim=256,
    text_dim=1024,
    num_layers=4,
)
mini_net_jvp = L(WanModel_JVP)(
    model_type="t2v",
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,
    dim=2048,
    ffn_dim=8192,
    freq_dim=256,
    text_dim=1024,
    num_layers=4,
)
mini_net_jvp_naive = L(WanModel_JVP)(
    model_type="t2v",
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,
    dim=2048,
    ffn_dim=8192,
    freq_dim=256,
    text_dim=1024,
    num_layers=4,
    naive_attn=True,
)


@pytest.mark.L1
def test_equivalent_forward_raw_vs_jvp():
    dtype = torch.float16
    net = instantiate(mini_net).cuda().to(dtype=dtype)
    net.eval()
    net_jvp = instantiate(mini_net_jvp).cuda().to(dtype=dtype)
    net_jvp.eval()

    net_jvp.load_state_dict(net.state_dict(), strict=False)

    batch_size = 2
    t = 8
    x_B_C_T_H_W = torch.randn(batch_size, 16, t, 40, 40).cuda().to(dtype=dtype)
    noise_labels_B = torch.randn(batch_size).cuda().to(dtype=dtype)
    noise_labels_BT = repeat(noise_labels_B, "b -> b 1")
    crossattn_emb_B_T_D = torch.randn(batch_size, 512, 1024).cuda().to(dtype=dtype)
    padding_mask_B_T_H_W = torch.zeros(batch_size, 1, 40, 40).cuda().to(dtype=dtype)

    output_BT = net(x_B_C_T_H_W, noise_labels_BT, crossattn_emb_B_T_D, padding_mask=padding_mask_B_T_H_W)
    output_BT_jvp = net_jvp(x_B_C_T_H_W, noise_labels_BT, crossattn_emb_B_T_D, padding_mask=padding_mask_B_T_H_W)
    torch.testing.assert_close(output_BT, output_BT_jvp, rtol=1e-3, atol=1e-3)


@pytest.mark.L1
def test_equivalent_jvp_naive_vs_flash():
    dtype = torch.float16
    net_jvp = instantiate(mini_net_jvp, sac_config=SACConfig(mode=CheckpointMode.NONE)).cuda().to(dtype=dtype)
    net_jvp.eval()
    net_jvp_naive = instantiate(mini_net_jvp_naive, sac_config=SACConfig(mode=CheckpointMode.NONE)).cuda().to(dtype=dtype)
    net_jvp_naive.eval()
    net_jvp_naive.load_state_dict(net_jvp.state_dict(), strict=False)

    batch_size = 2
    t = 8
    x_B_C_T_H_W = torch.randn(batch_size, 16, t, 40, 40).cuda().to(dtype=dtype)
    t_x_B_C_T_H_W = torch.randn_like(x_B_C_T_H_W)
    noise_labels_B = torch.randn(batch_size).cuda().to(dtype=dtype)
    t_noise_labels_B = torch.randn_like(noise_labels_B)
    noise_labels_BT = repeat(noise_labels_B, "b -> b 1")
    t_noise_labels_BT = repeat(t_noise_labels_B, "b -> b 1")
    crossattn_emb_B_T_D = torch.randn(batch_size, 512, 1024).cuda().to(dtype=dtype)
    fps_B = torch.randint(size=(1,), low=2, high=30).cuda().float().repeat(batch_size)
    padding_mask_B_T_H_W = torch.zeros(batch_size, 1, 40, 40).cuda().to(dtype=dtype)

    output_BT_withoutT = net_jvp(x_B_C_T_H_W, noise_labels_BT, crossattn_emb_B_T_D, fps=fps_B, padding_mask=padding_mask_B_T_H_W)
    output_BT, t_output_BT = net_jvp(
        (x_B_C_T_H_W, t_x_B_C_T_H_W),
        (noise_labels_BT, t_noise_labels_BT),
        crossattn_emb_B_T_D,
        fps=fps_B,
        padding_mask=padding_mask_B_T_H_W,
        withT=True,
    )

    fn_naive = functools.partial(net_jvp_naive.forward, crossattn_emb=crossattn_emb_B_T_D, fps=fps_B, padding_mask=padding_mask_B_T_H_W)

    output_BT_naive, t_output_BT_naive = torch.func.jvp(fn_naive, (x_B_C_T_H_W, noise_labels_BT), (t_x_B_C_T_H_W, t_noise_labels_BT))

    torch.testing.assert_close(output_BT_withoutT, output_BT_naive, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(output_BT_withoutT, output_BT, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(t_output_BT, t_output_BT_naive, rtol=1e-3, atol=1e-3)
