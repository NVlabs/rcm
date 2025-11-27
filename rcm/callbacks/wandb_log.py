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

from dataclasses import dataclass
from typing import Dict

import os
import torch
import torch.distributed as dist
import torch.utils.data
import wandb
from einops import rearrange

from typing import TYPE_CHECKING, Any

from imaginaire.model import ImaginaireModel
from imaginaire.utils import distributed, log, misc, wandb_util
from imaginaire.utils.easy_io import easy_io
from imaginaire.utils.callback import Callback

if TYPE_CHECKING:
    from imaginaire.model import ImaginaireModel


@dataclass
class _LossRecord:
    """Records and tracks various loss metrics during training."""

    # Initialize metrics with default values
    loss: torch.Tensor = 0
    iteration_count: int = 0  # More descriptive name for iter_count

    def reset(self) -> None:
        """Reset all metrics to their default values."""
        self.loss = 0
        self.iteration_count = 0

    def update(self, loss: torch.Tensor) -> None:
        """Update the loss record with new values.

        Args:
            loss: The loss value to add
        """
        self.loss += loss.detach().float()
        self.iteration_count += 1

    def get_stats(self) -> Dict[str, float]:
        """Calculate and return statistics across all metrics.

        Returns:
            Dictionary containing averaged metrics
        """
        stats = {}

        if self.iteration_count > 0:
            # Calculate average for standard metrics
            metrics = {"loss": self.loss}

            # Process each metric
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    avg_value = value / self.iteration_count
                    # Distribute across processes
                    dist.all_reduce(avg_value, op=dist.ReduceOp.AVG)
                    stats[name] = avg_value.item()
                else:
                    stats[name] = value / self.iteration_count
        else:
            # Default values if no iterations
            stats = {"loss": 0.0}

        # Reset after collecting stats
        self.reset()
        return stats


class WandbCallback(Callback):
    def __init__(self, logging_iter_multipler: int = 1, save_logging_iter_multipler: int = 1, save_s3: bool = False) -> None:
        super().__init__()
        self.train_image_log = _LossRecord()
        self.train_video_log = _LossRecord()
        self.final_loss_log = _LossRecord()
        self.img_unstable_count = torch.zeros(1, device="cuda")
        self.video_unstable_count = torch.zeros(1, device="cuda")
        self.logging_iter_multiplier = logging_iter_multipler
        self.save_logging_iter_multiplier = save_logging_iter_multipler
        assert self.logging_iter_multiplier > 0, "logging_iter_multiplier should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = f"@{self.logging_iter_multiplier}" if self.logging_iter_multiplier > 1 else ""
        self.name = "wandb_loss_log" + self.wandb_extra_tag

    @distributed.rank0_only
    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        wandb_util.init_wandb(self.config, model=model)
        config = self.config
        job_local_path = config.job.path_local
        # read optional job_env saved by `log_reproducible_setup`
        if os.path.exists(f"{job_local_path}/job_env.yaml"):
            job_info = easy_io.load(f"{job_local_path}/job_env.yaml")
            if wandb.run:
                wandb.run.config.update({f"JOB_INFO/{k}": v for k, v in job_info.items()}, allow_val_change=True)

        if os.path.exists(f"{config.job.path_local}/config.yaml") and "SLURM_LOG_DIR" in os.environ:
            easy_io.copyfile(
                f"{config.job.path_local}/config.yaml",
                os.path.join(os.environ["SLURM_LOG_DIR"], "config.yaml"),
            )

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:  # Log the curent learning rate.
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0():
            info = {}
            info["sample_counter"] = getattr(self.trainer, "sample_counter", iteration)

            for i, param_group in enumerate(optimizer.param_groups):
                info[f"optim/lr_{i}"] = param_group["lr"]
                info[f"optim/weight_decay_{i}"] = param_group["weight_decay"]

            wandb.log(info, step=iteration)

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        skip_update_due_to_unstable_loss = False
        if torch.isnan(loss) or torch.isinf(loss):
            skip_update_due_to_unstable_loss = True
            log.critical(
                f"Unstable loss {loss} at iteration {iteration} with is_image_batch: {model.is_image_batch(data_batch)}",
                rank0_only=False,
            )

        if not skip_update_due_to_unstable_loss:
            # Update the appropriate log based on batch type
            if model.is_image_batch(data_batch):
                self.train_image_log.update(loss.detach().float())
            else:
                self.train_video_log.update(loss.detach().float())

            # Always update the final loss log
            self.final_loss_log.update(loss.detach().float())
        else:
            # Track unstable losses
            if model.is_image_batch(data_batch):
                self.img_unstable_count += 1
            else:
                self.video_unstable_count += 1

        # Log at specified intervals
        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multiplier) == 0:
            if self.logging_iter_multiplier > 1:
                timer_results = {}
            else:
                timer_results = self.trainer.training_timer.compute_average_results()

            # Get statistics from each loss record
            image_stats = self.train_image_log.get_stats()
            video_stats = self.train_video_log.get_stats()
            final_stats = self.final_loss_log.get_stats()

            # Reduce unstable counts across all processes
            dist.all_reduce(self.img_unstable_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.video_unstable_count, op=dist.ReduceOp.SUM)

            if distributed.is_rank0():
                # Create info dictionary for logging
                info = {f"timer/{key}": value for key, value in timer_results.items()}

                # Add image statistics
                for key, value in image_stats.items():
                    info[f"train{self.wandb_extra_tag}/image_{key}"] = value

                # Add video statistics
                for key, value in video_stats.items():
                    info[f"train{self.wandb_extra_tag}/video_{key}"] = value

                # Add final statistics
                for key, value in final_stats.items():
                    info[f"train{self.wandb_extra_tag}/{key}"] = value

                # Add unstable counts
                info.update(
                    {
                        f"train{self.wandb_extra_tag}/img_unstable_count": self.img_unstable_count.item(),
                        f"train{self.wandb_extra_tag}/video_unstable_count": self.video_unstable_count.item(),
                        "iteration": iteration,
                        "sample_counter": getattr(self.trainer, "sample_counter", iteration),
                    }
                )

                # Save to S3 if enabled
                if self.save_s3:
                    save_interval = self.config.trainer.logging_iter * self.logging_iter_multiplier * self.save_logging_iter_multiplier
                    if iteration % save_interval == 0:
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Train_Iter{iteration:09d}.json",
                        )

                if wandb:
                    wandb.log(info, step=iteration)
                    # self.log_to_wandb(model, output_batch["x0"].detach().clone(), f"train{self.wandb_extra_tag}/x0", iteration)
                    # self.log_to_wandb(model, output_batch["xt"].detach().clone(), f"train{self.wandb_extra_tag}/xt", iteration)
                    # self.log_to_wandb(
                    #     model, output_batch["teacher_pred"].x0.detach().clone(), f"train{self.wandb_extra_tag}/teacher_pred", iteration
                    # )
                    # self.log_to_wandb(
                    #     model, output_batch["model_pred"].x0.detach().clone(), f"train{self.wandb_extra_tag}/student_pred", iteration
                    # )
            if self.logging_iter_multiplier == 1:
                self.trainer.training_timer.reset()

            # reset unstable count
            self.img_unstable_count.zero_()
            self.video_unstable_count.zero_()

    def on_validation_start(self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0) -> None:
        # Cache for collecting data/output batches.
        self._val_cache: dict[str, Any] = dict(
            data_batches=[],
            output_batches=[],
            loss=torch.tensor(0.0, device="cuda"),
            sample_size=torch.tensor(0, device="cuda"),
        )

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Collect the validation batch and aggregate the overall loss.
        # Collect the validation batch and aggregate the overall loss.
        batch_size = misc.get_data_batch_size(data_batch)
        self._val_cache["loss"] += loss * batch_size
        self._val_cache["sample_size"] += batch_size

    def on_validation_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        # Compute the average validation loss across all devices.
        dist.all_reduce(self._val_cache["loss"], op=dist.ReduceOp.SUM)
        dist.all_reduce(self._val_cache["sample_size"], op=dist.ReduceOp.SUM)
        loss = self._val_cache["loss"].item() / self._val_cache["sample_size"]
        # Log data/stats of validation set to W&B.
        if distributed.is_rank0():
            log.info(f"Validation loss (iteration {iteration}): {loss}")
            wandb.log({"val/loss": loss}, step=iteration)

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        wandb.finish()

    @torch.no_grad
    def log_to_wandb(
        self,
        model,
        data_tensor: torch.Tensor,
        wandb_key: str,
        iteration: int,
        n_viz_sample: int = 3,
        fps: int = 8,
        caption: str = None,
    ):
        """
        Logs image or video data to wandb from a [b, c, t, h, w] tensor.

        It normalizes the tensor, selects the first n_viz_sample from the batch (b)
        dimension, and arranges them into a single row grid (n rows, n_viz_sample columns).
        Logs as wandb.Image if t=1, otherwise logs as wandb.Video.

        Args:
            data_tensor (torch.Tensor): Input tensor of shape [b, c, t, h, w].
                                        Values are expected to be in the range [-1, 1].
            wandb_key (str): The key (name) for the log entry in wandb.
            n_viz_sample (int): Max number of samples from the batch dimension (b)
                                to visualize side-by-side. Defaults to 3.
            fps (int): Frames per second to use when logging video. Defaults to 8.
            caption (str, optional): Caption for the logged image/video in wandb. Defaults to None.
        """
        if hasattr(model, "decode"):
            data_tensor = model.decode(data_tensor)
        # Move tensor to CPU and detach from graph (important for logging)
        data_tensor = data_tensor.cpu().float()  # Ensure float for normalization

        _b, _c, _t, _h, _w = data_tensor.shape

        # Clamp and normalize tensor values from [-1, 1] to [0, 1]
        # wandb.Image/Video expect data in [0, 1] range for float or [0, 255] for uint8
        normalized_tensor = (1.0 + data_tensor.clamp(-1, 1)) / 2.0

        actual_n_viz_sample = min(n_viz_sample, _b)

        to_show = normalized_tensor[:actual_n_viz_sample]  # Shape: [actual_n_viz_sample, c, t, h, w]

        is_single_frame = _t == 1

        log_data = {}
        if is_single_frame:
            grid_tensor = rearrange(to_show.squeeze(2), "b c h w -> c h (b w)")
            log_data[wandb_key] = wandb.Image(grid_tensor, caption=caption)
            print(f"Prepared image grid for wandb key '{wandb_key}' with shape {grid_tensor.shape}")

        else:
            # wandb.Video expects time dimension first.
            grid_tensor = rearrange(to_show, "b c t h w -> t c h (b w)")

            # Optional: Convert to uint8 [0, 255] if preferred or if float causes issues
            # grid_tensor = (grid_tensor * 255).to(torch.uint8)

            log_data[wandb_key] = wandb.Video(grid_tensor, fps=fps, caption=caption)
            print(f"Prepared video grid for wandb key '{wandb_key}' with shape {grid_tensor.shape}")

        wandb.log(log_data, step=iteration)
        print(f"Successfully logged to wandb key: {wandb_key}")
