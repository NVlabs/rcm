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


class RectifiedFlow_TrigFlowWrapper:
    def __init__(self, sigma_data: float = 1.0, t_scaling_factor: float = 1.0):
        assert abs(sigma_data - 1.0) < 1e-6, "sigma_data must be 1.0 for RectifiedFlowScaling"
        self.t_scaling_factor = t_scaling_factor

    def __call__(self, trigflow_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trigflow_t = trigflow_t.to(torch.float64)
        c_skip = 1 / (torch.cos(trigflow_t) + torch.sin(trigflow_t))
        c_out = -1 * torch.sin(trigflow_t) / (torch.cos(trigflow_t) + torch.sin(trigflow_t))
        c_in = 1 / (torch.cos(trigflow_t) + torch.sin(trigflow_t))
        c_noise = (torch.sin(trigflow_t) / (torch.cos(trigflow_t) + torch.sin(trigflow_t))) * self.t_scaling_factor
        return c_skip, c_out, c_in, c_noise
