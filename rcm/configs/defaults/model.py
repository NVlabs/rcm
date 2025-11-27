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
from rcm.models.t2v_model_distill_rcm import T2VDistillConfig_rCM, T2VDistillModel_rCM

FSDP_CONFIG_T2V_DISTILL_RCM = dict(
    trainer=dict(distributed_parallelism="fsdp"),
    model=L(T2VDistillModel_rCM)(config=T2VDistillConfig_rCM(fsdp_shard_size=8), _recursive_=False),
)


def register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="fsdp_t2v_distill_rcm", node=FSDP_CONFIG_T2V_DISTILL_RCM)
