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

import argparse

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcp_checkpoint_dir", type=str, default="checkpoints/iter_000010000/model")
    parser.add_argument("--save_path", type=str, default="saved_model.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    storage_reader = FileSystemReader(args.dcp_checkpoint_dir)

    sd = {}
    _load_state_dict(sd, storage_reader=storage_reader, planner=_EmptyStateDictLoadPlanner(), no_dist=True)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("net_ema."):
            new_key = k.replace("net_ema.", "net.")
            # Save in bf16 precision
            if v.is_floating_point():
                new_sd[new_key] = v.to(torch.bfloat16)
            else:
                new_sd[new_key] = v
    torch.save(new_sd, args.save_path)
