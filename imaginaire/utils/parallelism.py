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

import io
import pickle
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh

try:
    from torch.distributed.tensor import Replicate, distribute_tensor

except ImportError:
    print("torch.distributed.tensor is not available. DeepSeek model will not work.")


def broadcast(tensor: torch.Tensor, cp_or_tp_mesh: DeviceMesh) -> torch.Tensor:
    tensor = tensor.to("cuda")
    if cp_or_tp_mesh.size() > 1:
        tensor = distribute_tensor(tensor, cp_or_tp_mesh, [Replicate()]).to_local()
    return tensor


def broadcast_object(
    obj: Any,
    src_rank: int,
    group: object = dist.group.WORLD,
    device: torch.device = torch.device("cpu"),  # noqa: B008
) -> Any:
    r"""
    Broadcasts an object to the given group.

    It will be sending the object if called from the source rank and receiving
    the object otherwise.

    Arguments:
        obj: object to broadcast; only used if called on the source rank.
        src_rank (int): source rank.
        group (``ProcessGroup``, optional): group used for the broadcast
            (default: ``dist.group.WORLD``).
        device (``torch.device``, optional): device to send from or receive
            to (default: ``torch.device("cpu")``).

    Returns:
        The broadcasted object.
    """
    if dist.get_rank() == src_rank:
        # Send the object
        buffer = io.BytesIO()
        torch.save(obj, buffer, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.LongTensor([len(data)]).to(device)
        data_send_tensor = torch.ByteTensor(data).to(device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        dist.broadcast(data_send_tensor, src=src_rank, group=group, async_op=False)
    else:
        # Receive the object
        length_tensor = torch.LongTensor([0]).to(device)
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        data_recv_tensor = torch.empty([int(length_tensor.item())], dtype=torch.uint8, device=device)
        dist.broadcast(data_recv_tensor, src=src_rank, group=group, async_op=False)
        buffer = io.BytesIO(data_recv_tensor.cpu().numpy())
        obj = torch.load(buffer, map_location=device, weights_only=False)
    return obj


def broadcast_with_shape_check(tensor: torch.Tensor, cp_or_tp_mesh: DeviceMesh) -> torch.Tensor:
    """Broadcast a tensor and check if the shape is the same across CP/TP ranks.
    If not, create a new tensor matching rank 0 and broadcast it.

    Args:
        tensor (torch.Tensor): The tensor to broadcast.
        cp_or_tp_mesh (DeviceMesh): The device mesh used to broadcast.

    Returns:
        torch.Tensor: The broadcasted tensor.
    """
    # create a tensor with the original value of the shape
    original_shape = torch.tensor(tensor.shape).cuda()

    # create a tensor that tracks the shape from rank 0.
    final_shape = torch.tensor(tensor.shape).cuda()
    final_shape = broadcast(final_shape, cp_or_tp_mesh)

    # if final shape is different from current shape, create a new tensor
    if final_shape.ne(original_shape).any():
        tensor = torch.zeros(final_shape.tolist(), dtype=tensor.dtype, device=tensor.device)

    tensor = broadcast(tensor, cp_or_tp_mesh)
    return tensor


def broadcast_to_cp_or_tp_ranks(data_batch: dict[str, torch.Tensor], cp_or_tp_mesh: DeviceMesh) -> bool:
    """Copies tensors in data_batch to the GPU and broadcasts across CP or TP ranks.

    The contents of data_batch are updated with the copied and broadcasted
    tensors. The inputs are replicated across CP ranks. The output logits
    and loss calculations are also replicated across CP ranks.

    Args:
        data_batch: Inputs (tokens, token_mask, images) needed for training.
        cp_or_tp_mesh: The DeviceMesh for context parallelism or tensor parallelism.
    """

    tokens = data_batch.get("tokens")
    data_batch["tokens"] = broadcast_with_shape_check(tokens, cp_or_tp_mesh)

    if "attention_mask" in data_batch:
        attention_mask = data_batch.get("attention_mask")
        data_batch["attention_mask"] = broadcast_with_shape_check(attention_mask, cp_or_tp_mesh)

    # Token Mask (Note: this is not attention mask)
    token_mask = data_batch.get("token_mask", None)
    if token_mask is None:
        token_mask = torch.ones_like(tokens, dtype=torch.bool)
    data_batch["token_mask"] = broadcast_with_shape_check(token_mask, cp_or_tp_mesh)

    if "padding_mask" in data_batch:
        padding_mask = data_batch["padding_mask"]
        data_batch["padding_mask"] = broadcast_with_shape_check(padding_mask, cp_or_tp_mesh)

    # Some rank may not have images, e.g. text data, remove images from all ranks in the group if first rank in the group doesn't have it, otherwise, create it
    has_images = (
        torch.ones(1, dtype=torch.bool).to(device=tokens.device)
        if "images" in data_batch
        else torch.zeros(1, dtype=torch.bool).to(device=tokens.device)
    )
    has_images = broadcast_with_shape_check(has_images, cp_or_tp_mesh)
    if not has_images and "images" in data_batch:
        del data_batch["images"]
    elif has_images and "images" not in data_batch:
        data_batch["images"] = torch.zeros(1, 1, 3, 448, 448).to(
            device=tokens.device
        )  # randomly init a zero tensor, the shape will be aligned later in broadcast_with_shape_check

    images = data_batch.get("images", None)
    if images is not None:
        data_batch["images"] = broadcast_with_shape_check(images, cp_or_tp_mesh)

    image_grid_thw = data_batch.get("image_grid_thw", None)
    if image_grid_thw is not None:
        data_batch["image_grid_thw"] = broadcast(image_grid_thw, cp_or_tp_mesh)  # NOTE : no need to check shape

    # TODO : This will NOT work if one batch has video and one batch has image.
    videos = data_batch.get("videos", None)
    if videos is not None:
        data_batch["videos"] = broadcast_with_shape_check(videos, cp_or_tp_mesh)

    video_grid_thw = data_batch.get("video_grid_thw", None)
    if video_grid_thw is not None:
        data_batch["video_grid_thw"] = broadcast(video_grid_thw, cp_or_tp_mesh)  # NOTE : no need to check shape

    # broadcast the string to all ranks
    for key in ["__url__", "dialog_str", "__key__"]:
        if key not in data_batch:
            data_batch[key] = [f"placeholder_{key}"]
        data_batch[key] = broadcast_object(data_batch[key], cp_or_tp_mesh)
    if "dataset_name" not in data_batch:
        data_batch["dataset_name"] = "default"
    data_batch["dataset_name"] = broadcast_object(data_batch["dataset_name"], cp_or_tp_mesh)
    return


class ModelWrapper(Stateful):
    """Wrapper for model state dict handling"""

    def __init__(self, model_parts: list[nn.Module]):
        self.model_parts = model_parts

    def state_dict(self) -> dict[str, Any]:
        sd = {}
        for model in self.model_parts:
            sd.update(get_model_state_dict(model))
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for model in self.model_parts:
            set_model_state_dict(
                model,
                model_state_dict=state_dict,
                options=StateDictOptions(strict=False),
            )
