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

from imaginaire.lazy_config import PLACEHOLDER
from imaginaire.lazy_config import LazyCall as L
from imaginaire.callbacks.manual_gc import ManualGarbageCollection
from imaginaire.callbacks.low_precision import LowPrecisionCallback
from rcm.callbacks.compile_tokenizer import CompileTokenizer
from rcm.callbacks.dataloading_monitor import DetailedDataLoadingSpeedMonitor
from rcm.callbacks.device_monitor import DeviceMonitor
from rcm.callbacks.grad_clip import GradClip
from rcm.callbacks.heart_beat import HeartBeat
from rcm.callbacks.iter_speed import IterSpeed
from rcm.callbacks.wandb_log import WandbCallback
from rcm.callbacks.every_n_draw_distill import EveryNDrawSample_Distill

BASIC_CALLBACKS = dict(
    grad_clip=L(GradClip)(),
    low_prec=L(LowPrecisionCallback)(config=PLACEHOLDER, trainer=PLACEHOLDER, update_iter=1),
    iter_speed=L(IterSpeed)(
        every_n="${trainer.logging_iter}",
        save_s3_every_log_n=10,
    ),
    heart_beat=L(HeartBeat)(
        every_n=10,
        update_interval_in_minute=20,
    ),
    device_monitor=L(DeviceMonitor)(
        every_n="${trainer.logging_iter}",
        upload_every_n_mul=10,
    ),
    manual_gc=L(ManualGarbageCollection)(every_n=5),
    compile_tokenizer=L(CompileTokenizer)(
        enabled=False,
        compile_after_iterations=0,
        dynamic=False,  # If there are issues with constant recompilations you may set this value to None or True
    ),
)

SPEED_CALLBACKS = dict(
    dataloader_speed=L(DetailedDataLoadingSpeedMonitor)(
        every_n="${trainer.logging_iter}",
    ),
)

WANDB_CALLBACK = dict(
    wandb=L(WandbCallback)(
        logging_iter_multipler=1,
        save_logging_iter_multipler=10,
    ),
    wandb_10x=L(WandbCallback)(
        logging_iter_multipler=10,
        save_logging_iter_multipler=1,
    ),
)

VIZ_ONLINE_SAMPLING_DISTILL_CALLBACKS = dict(
    every_n_sample_reg=L(EveryNDrawSample_Distill)(
        every_n=5000,
    ),
    every_n_sample_ema=L(EveryNDrawSample_Distill)(
        every_n=5000,
        is_ema=True,
    ),
)


def register_callbacks():
    cs = ConfigStore.instance()
    cs.store(group="callbacks", package="trainer.callbacks", name="basic", node=BASIC_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="dataloading_speed", node=SPEED_CALLBACKS)
    cs.store(group="callbacks", package="trainer.callbacks", name="wandb", node=WANDB_CALLBACK)
    cs.store(group="callbacks", package="trainer.callbacks", name="viz_online_sampling_distill", node=VIZ_ONLINE_SAMPLING_DISTILL_CALLBACKS)
