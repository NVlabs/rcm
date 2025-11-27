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

import torch

from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import distributed
from imaginaire.utils.callback import Callback
from imaginaire.config import Config
from imaginaire.utils.misc import get_local_tensor_if_DTensor


def update_master_weights(optimizer: torch.optim.Optimizer):
    if getattr(optimizer, "master_weights", False) and optimizer.param_groups_master is not None:
        params, master_params = [], []
        for group, group_master in zip(optimizer.param_groups, optimizer.param_groups_master):
            for p, p_master in zip(group["params"], group_master["params"]):
                params.append(get_local_tensor_if_DTensor(p.data))
                master_params.append(p_master.data)
        torch._foreach_copy_(params, master_params)


class LowPrecisionCallback(Callback):
    """The callback class handling low precision training

    Config with non-primitive type makes it difficult to override the option.
    The callback gets precision from model.precision instead.
    It also auto disabled when using fp32.
    """

    def __init__(self, config: Config, trainer: ImaginaireTrainer, update_iter: int):
        self.update_iter = update_iter

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        assert model.precision in [
            torch.bfloat16,
            torch.float16,
            torch.half,
        ], "LowPrecisionCallback must use a low precision dtype."
        self.precision_type = model.precision

    def on_training_step_start(self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data[k]):
                data[k] = v.to(dtype=self.precision_type)

    def on_validation_step_start(self, model: ImaginaireModel, data: dict[str, torch.Tensor], iteration: int = 0) -> None:
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data[k]):
                data[k] = v.to(dtype=self.precision_type)

    def on_before_zero_grad(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        if iteration % self.update_iter == 0:
            update_master_weights(optimizer)
