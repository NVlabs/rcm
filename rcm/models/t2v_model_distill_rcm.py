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

from __future__ import annotations

import collections
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Literal

import attrs
import numpy as np
import torch
import torch._dynamo
import torch.distributed.checkpoint as dcp
from einops import rearrange, repeat
from megatron.core import parallel_state
from torch import Tensor
from torch.distributed._composable.fsdp import FSDPModule, fully_shard, MixedPrecisionPolicy
from torch.distributed._tensor.api import DTensor
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.nn.modules.module import _IncompatibleKeys
from imaginaire.config import ObjectStoreConfig
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
from imaginaire.lazy_config import instantiate as lazy_instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log, misc
from imaginaire.utils.easy_io import easy_io
from imaginaire.utils.ema import FastEmaModelUpdater
from imaginaire.callbacks.low_precision import update_master_weights
from rcm.conditioner import DataType, TextCondition, concat_condition
from rcm.utils.optim_instantiate_dtensor import get_base_scheduler
from rcm.utils.lognormal import LogNormal
from rcm.utils.checkpointer import non_strict_load_model
from rcm.utils.context_parallel import broadcast, broadcast_split_tensor, cat_outputs_cp
from rcm.utils.dtensor_helper import DTensorFastEmaModelUpdater, broadcast_dtensor_model_states
from rcm.utils.fsdp_helper import hsdp_device_mesh
from rcm.utils.jvp_helper import TensorWithT
from rcm.utils.misc import count_params
from rcm.utils.torch_future import clip_grad_norm_
from rcm.modules.denoiser_scaling import RectifiedFlow_TrigFlowWrapper
from rcm.configs.defaults.ema import EMAConfig
from rcm.samplers.euler import FlowEulerSampler
from rcm.samplers.unipc import FlowUniPCMultistepSampler

torch._dynamo.config.suppress_errors = True

IS_PREPROCESSED_KEY = "is_preprocessed"
IS_PROCESSED_KEY = "is_processed"


@dataclass
class DenoisePrediction:
    x0: torch.Tensor  # clean data prediction
    F: torch.Tensor = None  # F prediction in TrigFlow


@attrs.define(slots=False)
class T2VDistillConfig_rCM:

    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    net_teacher: LazyDict = None
    net_fake_score: LazyDict = None
    optimizer_fake_score: LazyDict = None
    teacher_ckpt: str = ""
    tangent_warmup: int = 1
    teacher_guidance: float = 5.0
    grad_clip: bool = False
    sigma_max: float = 80

    ema: EMAConfig = EMAConfig()
    checkpoint: ObjectStoreConfig = ObjectStoreConfig()
    p_G: LazyDict = L(LogNormal)(
        p_mean=-0.8,
        p_std=1.6,
    )
    p_D: LazyDict = L(LogNormal)(
        p_mean=0.0,
        p_std=1.6,
    )
    student_update_freq: int = 5
    fsdp_shard_size: int = 1
    sigma_data: float = 1.0
    precision: str = "bfloat16"
    input_data_key: str = "videos"
    input_latent_key: str = "latents"
    input_caption_key: str = "prompts"
    loss_scale: float = 100.0
    loss_scale_dmd: float = 1.0
    loss_scale_fake_score: float = 1.0
    fd_type: int = 0  # finite difference type
    fd_size: float = 1e-4
    max_simulation_steps_fake: int = 4
    neg_embed_path: str = ""
    timestep_shift: float = 5

    adjust_video_noise: bool = True  # whether or not adjust video noise accroding to the video length

    state_ch: int = 16
    state_t: int = 21  # Number of latent frames
    resolution: str = "480p"
    rectified_flow_t_scaling_factor: float = 1000.0

    text_encoder_class: str = "umT5"
    text_encoder_path: str = ""


class T2VDistillModel_rCM(ImaginaireModel):

    def __init__(self, config: T2VDistillConfig_rCM):
        super().__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        log.warning(f"DiffusionModel: precision {self.precision}")

        # 1. set data keys and data information
        self.sigma_data = config.sigma_data
        self.setup_data_key()

        # 2. setup up diffusion processing and scaling~(pre-condition), sampler
        self.p_G = lazy_instantiate(config.p_G)
        self.p_D = lazy_instantiate(config.p_D)
        self.scaling = RectifiedFlow_TrigFlowWrapper(self.sigma_data, config.rectified_flow_t_scaling_factor)
        self.grad_clip = False
        self.tangent_warmup = config.tangent_warmup
        self.teacher_guidance = config.teacher_guidance
        self.student_update_freq = config.student_update_freq
        self.loss_scale = config.loss_scale
        self.loss_scale_dmd = config.loss_scale_dmd
        self.loss_scale_fake_score = config.loss_scale_fake_score
        self.fd_type = config.fd_type
        self.fd_size = config.fd_size
        self.max_simulation_steps_fake = config.max_simulation_steps_fake
        if config.neg_embed_path:
            self.neg_embed = easy_io.load(config.neg_embed_path)
        else:
            self.neg_embed = None
        self.timestep_shift = config.timestep_shift

        # 3. tokenizer
        with misc.timer("DiffusionModel: set_up_tokenizer"):
            self.tokenizer = lazy_instantiate(config.tokenizer)
            assert self.tokenizer.latent_ch == self.config.state_ch, f"latent_ch {self.tokenizer.latent_ch} != state_shape {self.config.state_ch}"

        # 4. Set up loss options, including loss masking, loss reduce and loss scaling
        if self.config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(self.config.state_t)
        else:
            self.video_noise_multiplier = 1.0

        # 5. create fsdp mesh if needed
        if config.fsdp_shard_size > 1:
            log.info(f"FSDP size: {config.fsdp_shard_size}")
            self.fsdp_device_mesh = hsdp_device_mesh(sharding_group_size=config.fsdp_shard_size)
        else:
            self.fsdp_device_mesh = None

        # 6. diffusion neural networks part
        self.set_up_model()

        # 7. training states
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

    def setup_data_key(self) -> None:
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model
        self.input_latent_key = self.config.input_latent_key
        self.input_caption_key = self.config.input_caption_key

    def build_net(self, net_dict: LazyDict):
        init_device = "meta"
        with misc.timer("Creating PyTorch model"):
            with torch.device(init_device):
                net = lazy_instantiate(net_dict)

            if self.fsdp_device_mesh:
                net.fully_shard(mesh=self.fsdp_device_mesh, mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32))
                net = fully_shard(
                    net, mesh=self.fsdp_device_mesh, mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32), reshard_after_forward=True
                )

            with misc.timer("meta to cuda and broadcast model states"):
                net.to_empty(device="cuda")
                net.init_weights()

            if self.fsdp_device_mesh:
                broadcast_dtensor_model_states(net, self.fsdp_device_mesh)
                for name, param in net.named_parameters():
                    assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"
        return net

    def load_ckpt_to_net(self, net, ckpt_path, prefix="net"):
        storage_reader = FileSystemReader(ckpt_path)
        _state_dict = get_model_state_dict(net)

        metadata = storage_reader.read_metadata()
        checkpoint_keys = metadata.state_dict_metadata.keys()

        model_keys = set(_state_dict.keys())

        # Add the prefix to the model keys for comparison
        prefixed_model_keys = {f"{prefix}.{k}" for k in model_keys}

        missing_keys = prefixed_model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - prefixed_model_keys

        if missing_keys:
            log.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        if not missing_keys and not unexpected_keys:
            log.info("All keys matched successfully.")

        _new_state_dict = collections.OrderedDict()
        for k in _state_dict.keys():
            if "_extra_state" in k:
                log.warning(k)
            _new_state_dict[f"{prefix}.{k}"] = _state_dict[k]
        dcp.load(_new_state_dict, storage_reader=storage_reader, planner=DefaultLoadPlanner(allow_partial_load=True))
        for k in _state_dict.keys():
            _state_dict[k] = _new_state_dict[f"{prefix}.{k}"]

        log.info(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=False)))
        del _state_dict, _new_state_dict

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self):
        config = self.config
        with misc.timer("Creating PyTorch model and ema if enabled"):
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, "conditioner should not have learnable parameters"
            self.net, self.net_teacher = self.build_net(config.net), self.build_net(config.net_teacher)
            self.net_fake_score = self.build_net(config.net_fake_score) if config.net_fake_score else None
            if config.net_fake_score:
                assert self.loss_scale_dmd > 0
            if config.teacher_ckpt:
                # load teacher checkpoint
                self.load_ckpt_to_net(self.net_teacher, config.teacher_ckpt)
                self.net.load_state_dict(self.net_teacher.state_dict(), strict=False)
                if self.net_fake_score:
                    self.net_fake_score.load_state_dict(self.net_teacher.state_dict())
            self.net_teacher.requires_grad_(False)
            self._param_count = count_params(self.net, verbose=False)

            if config.ema.enabled:
                self.net_ema = self.build_net(config.net)
                self.net_ema.requires_grad_(False)

                if self.fsdp_device_mesh:
                    self.net_ema_worker = DTensorFastEmaModelUpdater()
                else:
                    self.net_ema_worker = FastEmaModelUpdater()

                s = config.ema.rate
                self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)
        torch.cuda.empty_cache()

    def init_optimizer_scheduler(self, optimizer_config: LazyDict, scheduler_config: LazyDict):
        """Creates the optimizer and scheduler for the model."""
        # instantiate the net optimizer
        net_optimizer = lazy_instantiate(optimizer_config, model=self.net)
        self.optimizer_dict = {"net": net_optimizer}

        # instantiate the net scheduler
        net_scheduler = get_base_scheduler(net_optimizer, self, scheduler_config)
        self.scheduler_dict = {"net": net_scheduler}

        if self.net_fake_score:
            # instantiate the optimizer and lr scheduler for fake_score
            fake_score_optimizer = lazy_instantiate(self.config.optimizer_fake_score, model=self.net_fake_score)
            fake_score_scheduler = get_base_scheduler(fake_score_optimizer, self, scheduler_config)
            self.optimizer_dict["fake_score"] = fake_score_optimizer
            self.scheduler_dict["fake_score"] = fake_score_scheduler

    def is_student_phase(self, iteration: int):
        return self.net_fake_score is None or iteration < self.tangent_warmup or iteration % self.config.student_update_freq == 0

    def get_effective_iteration(self, iteration: int):
        return (
            iteration
            if self.net_fake_score is None or iteration < self.tangent_warmup
            else self.tangent_warmup + (iteration - self.tangent_warmup) // self.config.student_update_freq
        )

    def get_effective_iteration_fake(self, iteration: int):
        return iteration - self.get_effective_iteration(iteration) - 1

    # ------------------------ training hooks ------------------------
    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int) -> None:
        """
        update the net_ema
        """

        del scheduler, optimizer

        # the net and its master weights are handled by the low precision callback
        # manually update the fake score if needed
        if not self.is_student_phase(iteration):
            if self.net_fake_score:
                optimizer = self.optimizer_dict["fake_score"]
                update_master_weights(optimizer)

        if self.net_fake_score:
            scheduler, optimizer = self.optimizer_dict["fake_score"], self.scheduler_dict["fake_score"]
            del scheduler, optimizer

        if not self.is_student_phase(iteration):
            return

        if self.config.ema.enabled:
            # calculate beta for EMA update
            ema_beta = self.ema_beta(self.get_effective_iteration(iteration))
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()
        self.net = self.net.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.net_teacher:
            self.net_teacher = self.net_teacher.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.net_fake_score:
            self.net_fake_score = self.net_fake_score.to(memory_format=memory_format, **self.tensor_kwargs)

    # ------------------------ training ------------------------

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        """
        Get the optimizers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if self.is_student_phase(iteration):
            return [self.optimizer_dict["net"]]
        else:
            return [self.optimizer_dict["fake_score"]]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """
        Get the lr schedulers for the current iteration
        Args:
            iteration (int): The current training iteration

        """
        if self.is_student_phase(iteration):
            return [self.scheduler_dict["net"]]
        else:
            return [self.scheduler_dict["fake_score"]]

    def optimizers_zero_grad(self, iteration: int) -> None:
        """
        Zero the gradients of the optimizers based on the iteration
        """
        for optimizer in self.get_optimizers(iteration):
            optimizer.zero_grad()

    def optimizers_schedulers_step(self, grad_scaler: torch.cuda.amp.GradScaler, iteration: int) -> None:
        """
        Step the optimizer and scheduler step based on the iteration,
        and gradient scaler is also updated
        """
        for optimizer in self.get_optimizers(iteration):
            grad_scaler.step(optimizer)
            grad_scaler.update()

        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    def draw_training_time_G(self, x0_size: int, condition: Any) -> torch.Tensor:
        batch_size = x0_size[0]
        sigma_B = self.p_G(batch_size).to(device="cuda")
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO
        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        time_B_1 = torch.arctan(sigma_B_1)
        return time_B_1.double()

    def draw_training_time_D(self, x0_size: int, condition: Any) -> torch.Tensor:
        batch_size = x0_size[0]
        if self.timestep_shift > 0:
            sigma_B = torch.rand(batch_size).to(device="cuda").double()
            sigma_B = self.timestep_shift * sigma_B / (1 + (self.timestep_shift - 1) * sigma_B)
            sigma_B_1 = rearrange(sigma_B, "b -> b 1")
            time_B_1 = torch.arctan(sigma_B_1 / (1 - sigma_B_1))
            return time_B_1
        sigma_B = self.p_D(batch_size).to(device="cuda")
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma
        is_video_batch = condition.data_type == DataType.VIDEO
        multiplier = self.video_noise_multiplier if is_video_batch else 1
        sigma_B_1 = sigma_B_1 * multiplier
        time_B_1 = torch.arctan(sigma_B_1)
        return time_B_1.double()

    def denoise(
        self,
        xt_B_C_T_H_W: torch.Tensor,
        time: torch.Tensor,
        condition: TextCondition,
        net_type: Literal["teacher", "fake_score", "student"] = "teacher",
    ) -> DenoisePrediction:
        """
        Network forward to denoise the input noised data given noise level, and condition.

        Assumes EDM-scaling parameterization.

        Compared to base class denoise function, this function supports different net types:
        - fake_score: the fake score net on student generator's outputs
        - student: the student net (few-step generator)

        Args:
            xt (torch.Tensor): The input noise data.
            time (torch.Tensor): The noise level under TrigFlow parameterization.
            condition (TextCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """
        if time.ndim == 1:
            time_B_T = repeat(time, "b -> b 1")
        elif time.ndim == 2:
            time_B_T = time
        else:
            raise ValueError(f"time shape {time.shape} is not supported")
        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")

        # convert noise level time to EDM-formulation coefficients
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(trigflow_t=time_B_1_T_1_1)

        net = {"student": self.net, "teacher": self.net_teacher, "fake_score": self.net_fake_score}[net_type]

        net_output_B_C_T_H_W = net(
            x_B_C_T_H_W=(xt_B_C_T_H_W * c_in_B_1_T_1_1).to(**self.tensor_kwargs),
            timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
            **condition.to_dict(),
        ).float()

        # EDM reconstruction of x0
        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W

        F_pred_B_C_T_H_W = (torch.cos(time_B_1_T_1_1) * xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / torch.sin(time_B_1_T_1_1)
        return DenoisePrediction(x0=x0_pred_B_C_T_H_W, F=F_pred_B_C_T_H_W)

    def student_F_withT(self, xt_B_C_T_H_W: TensorWithT, time: TensorWithT, condition: TextCondition) -> TensorWithT:
        xt_B_C_T_H_W_withT, time_withT = xt_B_C_T_H_W, time
        xt_B_C_T_H_W, t_xt_B_C_T_H_W = xt_B_C_T_H_W_withT
        time, t_time = time_withT
        if time.ndim == 1:
            time_B_T = rearrange(time, "b -> b 1")
            t_time_B_T = rearrange(t_time, "b -> b 1")
        elif time.ndim == 2:
            time_B_T = time
            t_time_B_T = t_time
        else:
            raise ValueError(f"sigma shape {time.shape} is not supported")
        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")
        t_time_B_1_T_1_1 = rearrange(t_time_B_T, "b t -> b 1 t 1 1")
        # get precondition for the network
        (c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1), (
            t_c_skip_B_1_T_1_1,
            t_c_out_B_1_T_1_1,
            t_c_in_B_1_T_1_1,
            t_c_noise_B_1_T_1_1,
        ) = torch.func.jvp(self.scaling, (time_B_1_T_1_1,), (t_time_B_1_T_1_1,))

        def _process_input(xt_B_C_T_H_W, c_in_B_1_T_1_1):
            return xt_B_C_T_H_W * c_in_B_1_T_1_1

        x_B_C_T_H_W, t_x_B_C_T_H_W = torch.func.jvp(_process_input, (xt_B_C_T_H_W, c_in_B_1_T_1_1), (t_xt_B_C_T_H_W, t_c_in_B_1_T_1_1))

        # forward pass through the network
        net_output_B_C_T_H_W, t_net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=(
                x_B_C_T_H_W.to(**self.tensor_kwargs),
                t_x_B_C_T_H_W.to(**self.tensor_kwargs),
            ),
            timesteps_B_T=(
                c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
                t_c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
            ),
            **condition.to_dict(),
            withT=True,
        )

        net_output_B_C_T_H_W, t_net_output_B_C_T_H_W = net_output_B_C_T_H_W.float(), t_net_output_B_C_T_H_W.float()

        def _process_output(xt_B_C_T_H_W, net_output_B_C_T_H_W, c_skip_B_1_T_1_1, c_out_B_1_T_1_1, time_B_1_T_1_1):
            x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
            F_pred_B_C_T_H_W = (torch.cos(time_B_1_T_1_1) * xt_B_C_T_H_W - x0_pred_B_C_T_H_W) / torch.sin(time_B_1_T_1_1)
            return F_pred_B_C_T_H_W

        F_pred_B_C_T_H_W, t_F_pred_B_C_T_H_W = torch.func.jvp(
            _process_output,
            (xt_B_C_T_H_W, net_output_B_C_T_H_W, c_skip_B_1_T_1_1, c_out_B_1_T_1_1, time_B_1_T_1_1),
            (t_xt_B_C_T_H_W, t_net_output_B_C_T_H_W, t_c_skip_B_1_T_1_1, t_c_out_B_1_T_1_1, t_time_B_1_T_1_1),
        )

        return (F_pred_B_C_T_H_W, t_F_pred_B_C_T_H_W.detach())

    def training_step_generator(self, x0_B_C_T_H_W: torch.Tensor, condition: TextCondition, uncondition: TextCondition, iteration: int):
        log.debug(f"Student update {iteration}")
        time_B_T = self.draw_training_time_G(x0_B_C_T_H_W.size(), condition)
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")

        # Broadcast and split the input data and condition for model parallelism
        (
            x0_B_C_T_H_W,
            condition,
            uncondition,
            epsilon_B_C_T_H_W,
            time_B_T,
        ) = self.broadcast_split_for_model_parallelsim(x0_B_C_T_H_W, condition, uncondition, epsilon_B_C_T_H_W, time_B_T)

        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")
        cost_B_1_T_1_1, sint_B_1_T_1_1 = torch.cos(time_B_1_T_1_1), torch.sin(time_B_1_T_1_1)
        # Generate noisy observations
        xt_B_C_T_H_W = x0_B_C_T_H_W * cost_B_1_T_1_1 + epsilon_B_C_T_H_W * sint_B_1_T_1_1
        with torch.no_grad():
            F_teacher_B_C_T_H_W = self.denoise(xt_B_C_T_H_W, time_B_T, condition, net_type="teacher").F
            if self.teacher_guidance > 0.0:
                F_teacher_B_C_T_H_W_uncond = self.denoise(xt_B_C_T_H_W, time_B_T, uncondition, net_type="teacher").F
                F_teacher_B_C_T_H_W = F_teacher_B_C_T_H_W + self.teacher_guidance * (F_teacher_B_C_T_H_W - F_teacher_B_C_T_H_W_uncond)

        # see Section 5.1 JVP rearrangement discussion https://arxiv.org/pdf/2410.11081
        t_xt_B_C_T_H_W = cost_B_1_T_1_1 * sint_B_1_T_1_1 * F_teacher_B_C_T_H_W
        t_time_B_T = (cost_B_1_T_1_1 * sint_B_1_T_1_1).squeeze(dim=[1, 3, 4])

        with torch.no_grad():
            if self.fd_type == 1:  # semi-continuous
                _, t_F_theta_B_C_T_H_W = self.student_F_withT((xt_B_C_T_H_W, t_xt_B_C_T_H_W), (time_B_T, 0 * t_time_B_T), condition)
                h = self.fd_size
                F_theta_B_C_T_H_W_n1 = self.denoise(xt_B_C_T_H_W, time_B_T - h, condition, net_type="student").F
                pF_pt_B_C_T_H_W = (np.cos(h) * _ - F_theta_B_C_T_H_W_n1) / np.sin(h)
                t_F_theta_B_C_T_H_W += cost_B_1_T_1_1 * sint_B_1_T_1_1 * pF_pt_B_C_T_H_W
            elif self.fd_type == 2:  # discrete
                h = self.fd_size
                _ = self.denoise(xt_B_C_T_H_W, time_B_T, condition, net_type="student").F
                xt2_B_C_T_H_W = np.cos(h) * xt_B_C_T_H_W - np.sin(h) * F_teacher_B_C_T_H_W
                _2 = self.denoise(xt2_B_C_T_H_W, time_B_T - h, condition, net_type="student").F
                dF_pt_B_C_T_H_W = (np.cos(h) * _ - _2) / np.sin(h)
                t_F_theta_B_C_T_H_W = cost_B_1_T_1_1 * sint_B_1_T_1_1 * dF_pt_B_C_T_H_W
            else:
                _, t_F_theta_B_C_T_H_W = self.student_F_withT((xt_B_C_T_H_W, t_xt_B_C_T_H_W), (time_B_T, t_time_B_T), condition)

        if self.net_fake_score and iteration > self.tangent_warmup:
            G_time_B_T = math.pi / 2 * torch.ones_like(time_B_T)
            G_time_B_1_T_1_1 = rearrange(G_time_B_T, "b t -> b 1 t 1 1")
            G_cost_B_1_T_1_1, G_sint_B_1_T_1_1 = torch.cos(G_time_B_1_T_1_1), torch.sin(G_time_B_1_T_1_1)
            G_xt_B_C_T_H_W = x0_B_C_T_H_W * G_cost_B_1_T_1_1 + torch.randn_like(epsilon_B_C_T_H_W) * G_sint_B_1_T_1_1
            num_simulation_steps_fake = self.get_effective_iteration(iteration) % self.max_simulation_steps_fake
            for _ in range(num_simulation_steps_fake):
                with torch.no_grad():
                    G_x0_B_C_T_H_W = self.denoise(G_xt_B_C_T_H_W, G_time_B_T, condition, net_type="student").x0
                G_time_B_T = torch.minimum(self.draw_training_time_D(x0_B_C_T_H_W.size(), condition), G_time_B_T)
                G_time_B_T = self.sync(G_time_B_T, condition)
                G_time_B_1_T_1_1 = rearrange(G_time_B_T, "b t -> b 1 t 1 1")
                G_cost_B_1_T_1_1, G_sint_B_1_T_1_1 = torch.cos(G_time_B_1_T_1_1), torch.sin(G_time_B_1_T_1_1)
                G_xt_B_C_T_H_W = G_x0_B_C_T_H_W * G_cost_B_1_T_1_1 + torch.randn_like(epsilon_B_C_T_H_W) * G_sint_B_1_T_1_1
            all_xt_B_C_T_H_W = torch.cat([xt_B_C_T_H_W, G_xt_B_C_T_H_W], dim=0)
            all_time_B_T = torch.cat([time_B_T, G_time_B_T], dim=0)
            all_condition = concat_condition(condition, condition)
            all_theta_B_C_T_H_W = self.denoise(all_xt_B_C_T_H_W, all_time_B_T, all_condition, net_type="student")
            F_theta_B_C_T_H_W, _ = torch.chunk(all_theta_B_C_T_H_W.F, 2)
            _, G_x0_theta_B_C_T_H_W = torch.chunk(all_theta_B_C_T_H_W.x0, 2)
        else:
            F_theta_B_C_T_H_W = self.denoise(xt_B_C_T_H_W, time_B_T, condition, net_type="student").F
        F_theta_B_C_T_H_W_sg = F_theta_B_C_T_H_W.clone().detach()

        warmup_ratio = min(1.0, iteration / self.tangent_warmup)

        g_B_C_T_H_W = -cost_B_1_T_1_1 * torch.sqrt(1 - warmup_ratio**2 * sint_B_1_T_1_1**2) * (
            F_theta_B_C_T_H_W_sg - F_teacher_B_C_T_H_W
        ) - warmup_ratio * (cost_B_1_T_1_1 * sint_B_1_T_1_1 * xt_B_C_T_H_W + t_F_theta_B_C_T_H_W)

        with torch.no_grad():
            df_dt = -cost_B_1_T_1_1 * (F_theta_B_C_T_H_W_sg - F_teacher_B_C_T_H_W) - (sint_B_1_T_1_1 * xt_B_C_T_H_W + t_F_theta_B_C_T_H_W)
            nan_mask_g = torch.isnan(g_B_C_T_H_W).flatten(start_dim=1).any(dim=1).view(*g_B_C_T_H_W.shape[:1], 1, 1, 1, 1).expand_as(g_B_C_T_H_W)
            nan_mask_F_theta = (
                torch.isnan(F_theta_B_C_T_H_W)
                .flatten(start_dim=1)
                .any(dim=1)
                .view(*F_theta_B_C_T_H_W.shape[:1], 1, 1, 1, 1)
                .expand_as(F_theta_B_C_T_H_W)
            )

        nan_mask = nan_mask_g | nan_mask_F_theta

        g_B_C_T_H_W[nan_mask] = 0
        F_theta_B_C_T_H_W = torch.where(nan_mask, torch.tensor(0.0, device=F_theta_B_C_T_H_W.device), F_theta_B_C_T_H_W)
        F_theta_B_C_T_H_W_sg[nan_mask] = 0

        g_B_C_T_H_W = g_B_C_T_H_W.double() / (g_B_C_T_H_W.double().norm(p=2, dim=(1, 2, 3, 4), keepdim=True) + 0.1)

        loss_scm = ((F_theta_B_C_T_H_W - F_theta_B_C_T_H_W_sg - g_B_C_T_H_W) ** 2).sum(dim=(1, 2, 3, 4))
        kendall_loss = self.loss_scale * loss_scm

        x0_teacher_B_C_T_H_W = cost_B_1_T_1_1 * xt_B_C_T_H_W - sint_B_1_T_1_1 * F_teacher_B_C_T_H_W
        x0_theta_B_C_T_H_W = cost_B_1_T_1_1 * xt_B_C_T_H_W - sint_B_1_T_1_1 * F_theta_B_C_T_H_W
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "time": time_B_T,
            "condition": condition,
            "df_dt": df_dt,
            "nan_mask_g": nan_mask_g,
            "nan_mask_F_theta": nan_mask_F_theta,
            "teacher_pred": DenoisePrediction(x0_teacher_B_C_T_H_W, F_teacher_B_C_T_H_W),
            "model_pred": DenoisePrediction(x0_theta_B_C_T_H_W, F_theta_B_C_T_H_W),
        }

        if self.net_fake_score and iteration > self.tangent_warmup:
            D_time_B_T = self.draw_training_time_D(x0_B_C_T_H_W.size(), condition)
            D_time_B_T = self.sync(D_time_B_T, condition)
            D_time_B_1_T_1_1 = rearrange(D_time_B_T, "b t -> b 1 t 1 1")
            D_xt_theta_B_C_T_H_W = G_x0_theta_B_C_T_H_W * torch.cos(D_time_B_1_T_1_1) + torch.randn_like(x0_B_C_T_H_W) * torch.sin(D_time_B_1_T_1_1)

            with torch.no_grad():
                x0_theta_fake_B_C_T_H_W = self.denoise(D_xt_theta_B_C_T_H_W, D_time_B_T, condition, net_type="fake_score").x0

            with torch.no_grad():
                x0_theta_teacher_B_C_T_H_W = self.denoise(D_xt_theta_B_C_T_H_W, D_time_B_T, condition, net_type="teacher").x0
                if self.teacher_guidance > 0.0:
                    x0_theta_teacher_B_C_T_H_W_uncond = self.denoise(D_xt_theta_B_C_T_H_W, D_time_B_T, uncondition, net_type="teacher").x0
                    x0_theta_teacher_B_C_T_H_W = x0_theta_teacher_B_C_T_H_W + self.teacher_guidance * (
                        x0_theta_teacher_B_C_T_H_W - x0_theta_teacher_B_C_T_H_W_uncond
                    )
            with torch.no_grad():
                weight_factor = (
                    torch.abs(G_x0_theta_B_C_T_H_W.double() - x0_theta_teacher_B_C_T_H_W.double())
                    .mean(dim=[1, 2, 3, 4], keepdim=True)
                    .clip(min=0.00001)
                )
            grad_B_C_T_H_W = (x0_theta_fake_B_C_T_H_W.double() - x0_theta_teacher_B_C_T_H_W.double()) / weight_factor
            loss_dmd = (G_x0_theta_B_C_T_H_W.double() - (G_x0_theta_B_C_T_H_W.double() - grad_B_C_T_H_W).detach()) ** 2
            loss_dmd[torch.isnan(loss_dmd).flatten(start_dim=1).any(dim=1)] = 0
            loss_dmd = loss_dmd.sum(dim=(1, 2, 3, 4))
            kendall_loss += self.loss_scale_dmd * loss_dmd
        return output_batch, kendall_loss

    def training_step_critic(
        self, x0_B_C_T_H_W: torch.Tensor, condition: TextCondition, uncondition: TextCondition, iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        log.debug(f"Critic update {iteration}")
        # Sample pertubation noise levels and N(0, 1) noises
        time_B_T = self.draw_training_time_G(x0_B_C_T_H_W.size(), condition)
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")

        # Broadcast and split the input data and condition for model parallelism
        (
            x0_B_C_T_H_W,
            condition,
            uncondition,
            epsilon_B_C_T_H_W,
            time_B_T,
        ) = self.broadcast_split_for_model_parallelsim(x0_B_C_T_H_W, condition, uncondition, epsilon_B_C_T_H_W, time_B_T)
        G_time_B_T = math.pi / 2 * torch.ones_like(time_B_T)
        G_time_B_1_T_1_1 = rearrange(G_time_B_T, "b t -> b 1 t 1 1")
        G_cost_B_1_T_1_1, G_sint_B_1_T_1_1 = torch.cos(G_time_B_1_T_1_1), torch.sin(G_time_B_1_T_1_1)
        G_xt_B_C_T_H_W = x0_B_C_T_H_W * G_cost_B_1_T_1_1 + epsilon_B_C_T_H_W * G_sint_B_1_T_1_1

        num_simulation_steps_fake = self.get_effective_iteration_fake(iteration) % self.max_simulation_steps_fake
        for _ in range(num_simulation_steps_fake):
            with torch.no_grad():
                G_x0_B_C_T_H_W = self.denoise(G_xt_B_C_T_H_W, G_time_B_T, condition, net_type="student").x0
            G_time_B_T = torch.minimum(self.draw_training_time_D(x0_B_C_T_H_W.size(), condition), G_time_B_T)
            G_time_B_T = self.sync(G_time_B_T, condition)
            G_time_B_1_T_1_1 = rearrange(G_time_B_T, "b t -> b 1 t 1 1")
            G_cost_B_1_T_1_1, G_sint_B_1_T_1_1 = torch.cos(G_time_B_1_T_1_1), torch.sin(G_time_B_1_T_1_1)
            G_xt_B_C_T_H_W = G_x0_B_C_T_H_W * G_cost_B_1_T_1_1 + torch.randn_like(epsilon_B_C_T_H_W) * G_sint_B_1_T_1_1

        with torch.no_grad():
            G_x0_theta_B_C_T_H_W = self.denoise(G_xt_B_C_T_H_W, G_time_B_T, condition, net_type="student").x0

        D_time_B_T = self.draw_training_time_D(x0_B_C_T_H_W.size(), condition)
        D_epsilon_B_C_T_H_W = torch.randn_like(x0_B_C_T_H_W)
        D_time_B_T = self.sync(D_time_B_T, condition)
        D_time_B_1_T_1_1 = rearrange(D_time_B_T, "b t -> b 1 t 1 1")
        D_cost_B_1_T_1_1, D_sint_B_1_T_1_1 = torch.cos(D_time_B_1_T_1_1), torch.sin(D_time_B_1_T_1_1)
        D_xt_theta_B_C_T_H_W = G_x0_theta_B_C_T_H_W * D_cost_B_1_T_1_1 + D_epsilon_B_C_T_H_W * D_sint_B_1_T_1_1
        x0_theta_fake_B_C_T_H_W = self.denoise(D_xt_theta_B_C_T_H_W, D_time_B_T, condition, net_type="fake_score").x0
        kendall_loss = self.loss_scale_fake_score * ((G_x0_theta_B_C_T_H_W - x0_theta_fake_B_C_T_H_W) ** 2 / D_sint_B_1_T_1_1**2).sum(
            dim=(1, 2, 3, 4)
        )
        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": G_xt_B_C_T_H_W,
            "x0_pred": G_x0_theta_B_C_T_H_W,
        }
        return output_batch, kendall_loss

    def training_step(self, data_batch: dict[str, torch.Tensor], iteration: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)

        if self.is_student_phase(iteration):
            # update the student
            self.net.train().requires_grad_(True)
            if self.net_fake_score:
                self.net_fake_score.eval().requires_grad_(False)

            output_batch, kendall_loss = self.training_step_generator(x0_B_C_T_H_W, condition, uncondition, iteration)

        else:
            # update the fake_score
            self.net.eval().requires_grad_(False)
            if self.net_fake_score:
                self.net_fake_score.train().requires_grad_(True)

            output_batch, kendall_loss = self.training_step_critic(x0_B_C_T_H_W, condition, uncondition, iteration)

        kendall_loss = kendall_loss.mean()

        return output_batch, kendall_loss

    @torch.no_grad()
    def forward(self, xt, t, condition: TextCondition):
        """
        Performs denoising on the input noise data, noise level, and condition

        Args:
            xt (torch.Tensor): The input noise data.
            sigma (torch.Tensor): The noise level.
            condition (TextCondition): conditional information, generated from self.conditioner

        Returns:
            DenoisePrediction: The denoised prediction, it includes clean data predicton (x0), \
                noise prediction (eps_pred).
        """
        return self.denoise(xt, t, condition)

    # ------------------------ Sampling ------------------------

    def get_x0_fn_from_batch(self, data_batch: Dict) -> Callable:

        _, _, condition, uncondition = self.get_data_and_condition(data_batch)
        _, condition, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(None, condition, uncondition, None, None)

        # For inference, check if parallel_state is initialized
        if not parallel_state.is_initialized():
            assert not self.net.is_context_parallel_enabled, "parallel_state is not initialized, context parallel should be turned off."

        @torch.no_grad()
        def x0_fn(noise_x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            raw_x0 = self.denoise(noise_x, time, condition, net_type="student").x0
            return raw_x0

        return x0_fn

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        num_steps: int = 4,
        init_noise: torch.Tensor = None,
        mid_t: List[float] | None = None,
    ) -> torch.Tensor:
        input_key = self.input_data_key

        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        x0_fn = self.get_x0_fn_from_batch(data_batch)

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)

        if init_noise is None:
            init_noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )

        if self.net.is_context_parallel_enabled:
            init_noise = broadcast_split_tensor(init_noise, seq_dim=2, process_group=self.get_context_parallel_group())

        if mid_t is None:
            mid_t = [1.3, 1.0, 0.6][: num_steps - 1]

        t_steps = torch.tensor(
            [math.atan(self.config.sigma_max)] + list(mid_t),
            dtype=torch.float64,
            device=init_noise.device,
        )

        # t_0 = T, t_N = 0
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        # Sampling steps
        x = init_noise.to(torch.float64)
        ones = torch.ones(x.size(0), device=x.device, dtype=x.dtype)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x = x0_fn(x.float(), t_cur.float() * ones).to(torch.float64)
            noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )
            if self.net.is_context_parallel_enabled:
                noise = broadcast_split_tensor(noise, seq_dim=2, process_group=self.get_context_parallel_group())
            x = torch.cos(t_next) * x + torch.sin(t_next) * noise
        samples = x.float()
        if self.net.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())
        return torch.nan_to_num(samples)

    @torch.no_grad()
    def generate_samples_from_batch_teacher(
        self,
        data_batch: Dict,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        init_noise: torch.Tensor = None,
        num_steps: int = 50,
        sampler="UniPC",
        timestep_shift=5.0,
    ) -> torch.Tensor:
        """
        Generate samples from the batch. Based on given batch, it will automatically determine whether to generate image or video samples.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            num_steps (int): number of steps for the diffusion process
        """
        _, _, condition, uncondition = self.get_data_and_condition(data_batch)
        _, condition, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(None, condition, uncondition, None, None)

        input_key = self.input_data_key

        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)

        if init_noise is None:
            init_noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )

        if self.net_teacher.is_context_parallel_enabled:
            init_noise = broadcast_split_tensor(init_noise, seq_dim=2, process_group=self.get_context_parallel_group())

        x = init_noise.to(torch.float64)

        sigma_max = self.config.sigma_max / (self.config.sigma_max + 1)
        unshifted_sigma_max = sigma_max / (timestep_shift - (timestep_shift - 1) * sigma_max)

        samplers = {"Euler": FlowEulerSampler, "UniPC": FlowUniPCMultistepSampler}
        sampler = samplers[sampler](num_train_timesteps=1000, sigma_max=unshifted_sigma_max, sigma_min=0.0)
        sampler.set_timesteps(num_inference_steps=num_steps, device=self.tensor_kwargs["device"], shift=timestep_shift)

        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        for _, t in enumerate(sampler.timesteps):
            timesteps = t * ones

            with torch.no_grad():
                v_cond = self.net_teacher(
                    x_B_C_T_H_W=x.to(**self.tensor_kwargs), timesteps_B_T=timesteps.to(**self.tensor_kwargs), **condition.to_dict()
                ).float()
                v_uncond = self.net_teacher(
                    x_B_C_T_H_W=x.to(**self.tensor_kwargs), timesteps_B_T=timesteps.to(**self.tensor_kwargs), **uncondition.to_dict()
                ).float()

            v_pred = v_uncond + self.teacher_guidance * (v_cond - v_uncond)

            x = sampler.step(v_pred, t, x)

        samples = x.float()
        if self.net_teacher.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

        return torch.nan_to_num(samples)

    @torch.no_grad()
    def validation_step(self, data: dict[str, torch.Tensor], iteration: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Current code does nothing.
        """
        pass

    # ------------------------ Distributed Parallel ------------------------

    @staticmethod
    def get_context_parallel_group():
        if parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    def sync(self, tensor, condition):
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            tensor = broadcast(tensor, cp_group)
        return tensor

    def broadcast_split_for_model_parallelsim(self, x0_B_C_T_H_W, condition, uncondition, epsilon_B_C_T_H_W, sigma_B_T):
        """
        Broadcast and split the input data and condition for model parallelism.
        Currently, we only support context parallelism.
        """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            x0_B_C_T_H_W = broadcast_split_tensor(x0_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            epsilon_B_C_T_H_W = broadcast_split_tensor(epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] == 1:
                    sigma_B_T = broadcast(sigma_B_T, cp_group)
                else:
                    sigma_B_T = broadcast_split_tensor(sigma_B_T, seq_dim=1, process_group=cp_group)
            condition = condition.broadcast(cp_group)
            uncondition = uncondition.broadcast(cp_group)
            self.net.enable_context_parallel(cp_group)
            if self.net_teacher:
                self.net_teacher.enable_context_parallel(cp_group)
            if self.net_fake_score:
                self.net_fake_score.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()
            if self.net_teacher:
                self.net_teacher.disable_context_parallel()
            if self.net_fake_score:
                self.net_fake_score.disable_context_parallel()

        return x0_B_C_T_H_W, condition, uncondition, epsilon_B_C_T_H_W, sigma_B_T

    # ------------------ Data Preprocessing ------------------

    def _normalize_video_inplace(self, data_batch: dict[str, Tensor]) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key
        # only handle video batch
        # Check if the data has already been normalized and avoid re-normalizing
        if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
            assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
            assert torch.all(
                (data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)
            ), f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
        else:
            assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
            data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
            data_batch[IS_PREPROCESSED_KEY] = True

        from torchvision.transforms.v2 import UniformTemporalSubsample

        expected_length = self.tokenizer.get_pixel_num_frames(self.config.state_t)
        original_length = data_batch[input_key].shape[2]
        if original_length != expected_length:
            video = rearrange(data_batch[input_key], "b c t h w -> b t c h w")
            video = UniformTemporalSubsample(expected_length)(video)
            data_batch[input_key] = rearrange(video, "b t c h w -> b c t h w")

    def _normalize_latent_inplace(self, data_batch: dict[str, Tensor]) -> None:
        latents = data_batch[self.input_latent_key]
        assert latents.shape[2] >= self.config.state_t
        data_batch[self.input_latent_key] = latents[:, :, : self.config.state_t, :, :]

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, TextCondition]:
        if IS_PROCESSED_KEY not in data_batch or not data_batch[IS_PROCESSED_KEY]:
            if self.input_latent_key in data_batch:
                self._normalize_latent_inplace(data_batch)
                data_batch[self.input_data_key] = self.decode(data_batch[self.input_latent_key]).contiguous().float().clamp(-1, 1)
                data_batch[IS_PREPROCESSED_KEY] = True

            self._normalize_video_inplace(data_batch)
            data_batch[self.input_latent_key] = self.encode(data_batch[self.input_data_key]).contiguous().float()
            data_batch[IS_PROCESSED_KEY] = True

        raw_state = data_batch[self.input_data_key]
        latent_state = data_batch[self.input_latent_key]
        # Condition
        if self.neg_embed is not None:
            data_batch["neg_t5_text_embeddings"] = repeat(
                self.neg_embed.to(**self.tensor_kwargs), "l d -> b l d", b=data_batch["t5_text_embeddings"].shape[0]
            )
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        condition = condition.edit_data_type(DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.VIDEO)
        return raw_state, latent_state, condition, uncondition

    # ------------------ Checkpointing ------------------

    def model_dict(self) -> Dict[str, Any]:
        model_dict = {"net": self.net}
        if self.net_fake_score:
            model_dict["fake_score"] = self.net_fake_score
        return model_dict

    def state_dict(self) -> Dict[str, Any]:
        net_state_dict = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled:
            ema_state_dict = self.net_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)
        if self.net_fake_score:
            fake_score_state_dict = self.net_fake_score.state_dict(prefix="net_fake_score.")
            net_state_dict.update(fake_score_state_dict)
        return net_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """
        Loads a state dictionary into the model and optionally its EMA counterpart.
        Different from torch strict=False mode, the method will not raise error for unmatched state shape while raise warning.

        Parameters:e
            state_dict (Mapping[str, Any]): A dictionary containing separate state dictionaries for the model and
                                            potentially for an EMA version of the model under the keys 'model' and 'ema', respectively.
            strict (bool, optional): If True, the method will enforce that the keys in the state dict match exactly
                                    those in the model and EMA model (if applicable). Defaults to True.
            assign (bool, optional): If True and in strict mode, will assign the state dictionary directly rather than
                                    matching keys one-by-one. This is typically used when loading parts of state dicts
                                    or using customized loading procedures. Defaults to False.
        """
        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        _fake_score_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v
            elif k.startswith("net_fake_score."):
                _fake_score_state_dict[k.replace("net_fake_score.", "")] = v

        state_dict = _reg_state_dict

        if strict:
            reg_results: _IncompatibleKeys = self.net.load_state_dict(_reg_state_dict, strict=strict, assign=assign)

            if self.config.ema.enabled:
                ema_results: _IncompatibleKeys = self.net_ema.load_state_dict(_ema_state_dict, strict=strict, assign=assign)
            if self.net_fake_score:
                fake_score_results: _IncompatibleKeys = self.net_fake_score.load_state_dict(_fake_score_state_dict, strict=strict, assign=assign)

            return _IncompatibleKeys(
                missing_keys=reg_results.missing_keys
                + (ema_results.missing_keys if self.config.ema.enabled else [])
                + (fake_score_results.missing_keys if self.net_fake_score else []),
                unexpected_keys=reg_results.unexpected_keys
                + (ema_results.unexpected_keys if self.config.ema.enabled else [])
                + (fake_score_results.unexpected_keys if self.net_fake_score else []),
            )
        else:
            log.critical("load model in non-strict mode")
            log.critical(non_strict_load_model(self.net, _reg_state_dict), rank0_only=False)
            if self.config.ema.enabled:
                log.critical("load ema model in non-strict mode")
                log.critical(non_strict_load_model(self.net_ema, _ema_state_dict), rank0_only=False)
            if self.net_fake_score:
                log.critical("load fake score model in non-strict mode")
                log.critical(non_strict_load_model(self.net_fake_score, _fake_score_state_dict), rank0_only=False)

    # ------------------ public methods ------------------
    def ema_beta(self, iteration: int) -> float:
        """
        Calculate the beta value for EMA update.
        weights = weights * beta + (1 - beta) * new_weights

        Args:
            iteration (int): Current iteration number.

        Returns:
            float: The calculated beta value.
        """
        iteration = iteration + self.config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    def model_param_stats(self) -> Dict[str, int]:
        return {"total_learnable_param_num": self._param_count}

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        return False

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state) * self.sigma_data

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent / self.sigma_data)

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t

    @property
    def text_encoder_class(self) -> str:
        return self.config.text_encoder_class

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        if self.config.ema.enabled:
            # https://github.com/pytorch/pytorch/issues/144289
            for module in self.net.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.net_ema_worker.cache(self.net.parameters(), is_cpu=is_cpu)
            self.net_ema_worker.copy_to(src_model=self.net_ema, tgt_model=self.net)
            if context is not None:
                log.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled:
                for module in self.net.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
                self.net_ema_worker.restore(self.net.parameters())
                if context is not None:
                    log.info(f"{context}: Restored training weights")

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ):
        if not self.grad_clip:
            max_norm = 1e12
        if self.net_fake_score:
            for param in self.net_fake_score.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            clip_grad_norm_(
                self.net_fake_score.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach,
            )
        for param in self.net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        return clip_grad_norm_(
            self.net.parameters(),
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        ).cpu()
