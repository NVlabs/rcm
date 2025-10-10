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

import os
from typing import IO, Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image as PILImage
from torch import Tensor

from imaginaire.utils import log
from imaginaire.utils.easy_io import easy_io


def save_image_or_video_multiview(
    tensor: Tensor, save_path: str | IO[Any], fps: int = 24, quality=None, ffmpeg_params=None, n_views: int = 1
) -> None:
    """
    Split the tensor into n_views, stack them along the width dimension, and save as a video
        Args:
        tensor (Tensor): Input tensor with shape (B, C, T, H, W) or (C, T, H, W) in [-1, 1] or [0, 1] range.
            If in [-1, 1] range, it will be automatically converted to [0, 1] range.
        save_path (Union[str, IO[Any]]): File path (with or without extension) or file-like object.
        fps (int): Frames per second for video. Default is 24.
        quality: Optional quality parameter for images (passed to easy_io).
        ffmpeg_params: Optional ffmpeg parameters for videos (passed to easy_io).
    """
    # Handle batch dimension if present
    if tensor.ndim == 5:
        tensor = tensor[0]  # Take the first item from the batch

    assert tensor.ndim == 4, "Tensor must have shape (C, T, H, W) or (B, C, T, H, W)"
    assert isinstance(save_path, str) or hasattr(save_path, "write"), "save_path must be a string or file-like object"

    # Normalize to [0, 1] range
    if torch.is_floating_point(tensor):
        # Check if tensor is in [-1, 1] range (approximately)
        if tensor.min() < -0.5:
            # Convert from [-1, 1] to [0, 1]
            tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0, 1)
    else:
        assert tensor.dtype == torch.uint8, "Only support uint8 tensor"
        tensor = tensor.float().div(255)

    kwargs = {}
    if quality is not None:
        kwargs["quality"] = quality
    if ffmpeg_params is not None:
        kwargs["ffmpeg_params"] = ffmpeg_params

    save_obj = (rearrange((tensor.cpu().float().numpy() * 255), "c (v t) h w -> t (v h) w c", v=n_views) + 0.5).astype(
        np.uint8
    )
    if isinstance(save_path, str):
        # Check if path already has an extension
        base, ext = os.path.splitext(save_path)
        if not ext:
            save_path = f"{base}.mp4"
    log.info(f"Saving video to {save_path} with fps {fps} and result shape {save_obj.shape}")
    easy_io.dump(save_obj, save_path, file_format="mp4", format="mp4", fps=fps, **kwargs)


def save_image_or_video(
    tensor: Tensor, save_path: str | IO[Any], fps: int = 24, quality=None, ffmpeg_params=None
) -> None:
    """
    Save a tensor as an image or video file based on shape

    Args:
        tensor (Tensor): Input tensor with shape (B, C, T, H, W) or (C, T, H, W) in [-1, 1] or [0, 1] range.
            If in [-1, 1] range, it will be automatically converted to [0, 1] range.
        save_path (Union[str, IO[Any]]): File path (with or without extension) or file-like object.
        fps (int): Frames per second for video. Default is 24.
        quality: Optional quality parameter for images (passed to easy_io).
        ffmpeg_params: Optional ffmpeg parameters for videos (passed to easy_io).
    """
    # Handle batch dimension if present
    if tensor.ndim == 5:
        tensor = tensor[0]  # Take the first item from the batch

    assert tensor.ndim == 4, "Tensor must have shape (C, T, H, W) or (B, C, T, H, W)"
    assert isinstance(save_path, str) or hasattr(save_path, "write"), "save_path must be a string or file-like object"

    # Normalize to [0, 1] range
    if torch.is_floating_point(tensor):
        # Check if tensor is in [-1, 1] range (approximately)
        if tensor.min() < -0.5:
            # Convert from [-1, 1] to [0, 1]
            tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0, 1)
    else:
        assert tensor.dtype == torch.uint8, "Only support uint8 tensor"
        tensor = tensor.float().div(255)

    kwargs = {}
    if quality is not None:
        kwargs["quality"] = quality
    if ffmpeg_params is not None:
        kwargs["ffmpeg_params"] = ffmpeg_params

    if tensor.shape[1] == 1:
        save_obj = PILImage.fromarray(
            (rearrange((tensor.cpu().float().numpy() * 255), "c 1 h w -> h w c") + 0.5).astype(np.uint8),
            mode="RGB",
        )
        if isinstance(save_path, str):
            # Check if path already has an extension
            base, ext = os.path.splitext(save_path)
            if not ext:
                save_path = f"{base}.jpg"
        easy_io.dump(save_obj, save_path, file_format="jpg", format="JPEG", quality=85, **kwargs)
    else:
        save_obj = (rearrange((tensor.cpu().float().numpy() * 255), "c t h w -> t h w c") + 0.5).astype(np.uint8)
        if isinstance(save_path, str):
            # Check if path already has an extension
            base, ext = os.path.splitext(save_path)
            if not ext:
                save_path = f"{base}.mp4"
        easy_io.dump(save_obj, save_path, file_format="mp4", format="mp4", fps=fps, **kwargs)


def save_text_prompts(prompts: dict[str | list], save_path: str | IO[Any]) -> None:
    """
    Save text prompts to a file.

    Args:
        prompts (dict[str]): Dictionary of text prompts to save. Expected keys: "prompt", "negative_prompt", "refined_prompt".
        save_path (Union[str, IO[Any]]): File path (with or without extension) or file-like object.
    """
    if isinstance(save_path, str):
        base, ext = os.path.splitext(save_path)
        if not ext:
            save_path = f"{base}.txt"
    with open(save_path, "w") as f:
        f.write(f"[Prompt]\n{prompts['prompt']}\n")
        if prompts.get("negative_prompt"):
            f.write(f"[Negative Prompt]\n{prompts['negative_prompt']}\n")

        if prompts.get("refined_prompt"):
            if isinstance(prompts["refined_prompt"], str):
                f.write(f"[Refined Prompt]\n{prompts['refined_prompt']}\n")
            elif isinstance(prompts["refined_prompt"], list):
                for chunk_id, refined_prompt in enumerate(prompts["refined_prompt"]):
                    f.write(f"[Refined Prompt for chunk {chunk_id}]\n{refined_prompt}\n")
            else:
                raise ValueError("refined_prompt must be a string or a list of strings")
