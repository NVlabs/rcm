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

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict

from rcm.networks.wan2pt1 import WanModel
from rcm.networks.wan2pt1_jvp import WanModel_JVP

wan2pt1_1pt3B_net_args = dict(
    dim=1536,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=16,
    num_heads=12,
    num_layers=30,
    out_dim=16,
    text_len=512,
)

wan2pt1_14B_net_args = dict(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=16,
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
)

WAN2PT1_1PT3B_T2V: LazyDict = L(WanModel)(**wan2pt1_1pt3B_net_args, model_type="t2v")

WAN2PT1_14B_T2V: LazyDict = L(WanModel)(**wan2pt1_14B_net_args, model_type="t2v")

WAN2PT1_1PT3B_T2V_JVP: LazyDict = L(WanModel_JVP)(**wan2pt1_1pt3B_net_args, model_type="t2v")

WAN2PT1_14B_T2V_JVP: LazyDict = L(WanModel_JVP)(**wan2pt1_14B_net_args, model_type="t2v")


def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="wan2pt1_1pt3B_t2v", node=WAN2PT1_1PT3B_T2V)
    cs.store(group="net", package="model.config.net", name="wan2pt1_14B_t2v", node=WAN2PT1_14B_T2V)
    cs.store(group="net", package="model.config.net", name="wan2pt1_1pt3B_t2v_jvp", node=WAN2PT1_1PT3B_T2V_JVP)
    cs.store(group="net", package="model.config.net", name="wan2pt1_14B_t2v_jvp", node=WAN2PT1_14B_T2V_JVP)


def register_net_fake_score():
    cs = ConfigStore.instance()
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt1_1pt3B_t2v", node=WAN2PT1_1PT3B_T2V)
    cs.store(group="net_fake_score", package="model.config.net_fake_score", name="wan2pt1_14B_t2v", node=WAN2PT1_14B_T2V)


def register_net_teacher():
    cs = ConfigStore.instance()
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt1_1pt3B_t2v", node=WAN2PT1_1PT3B_T2V)
    cs.store(group="net_teacher", package="model.config.net_teacher", name="wan2pt1_14B_t2v", node=WAN2PT1_14B_T2V)
