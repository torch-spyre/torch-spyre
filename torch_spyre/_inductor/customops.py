# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import torch

from .stickify import SpyreDCI, spyre_reduction_result_shape, StickFormat
from . import Unsupported


@torch.library.custom_op("spyre::compact", mutates_args=())
def compact(input: torch.Tensor) -> torch.Tensor:
    if len(input.size()) != 1:
        raise Unsupported("compact not implemented for 1-D tensors")
    return input.clone()


@compact.register_fake
def _(input):
    if len(input.size()) != 1:
        raise Unsupported("compact only implemented for 1-D tensors")
    output = input.new_empty(input.size())
    output.spyre_dci = SpyreDCI([0], format=StickFormat.DENSE)
    return output


@torch.library.custom_op("spyre::swap", mutates_args=(), device_types="spyre")
def swap(input: torch.Tensor) -> torch.Tensor:
    if len(input.size()) != 1:
        raise Unsupported("swap only implemented for 1-D tensors")
    output = input.new_empty_strided(input.size(), [64])
    output.spyre_dci = SpyreDCI([0], format=StickFormat.SPARSE)
    return output


@swap.register_fake
def _(input):
    if len(input.size()) != 1:
        raise Unsupported("swap only implemented for 1-D tensors")
    output = input.new_empty_strided(input.size(), [64])
    output.spyre_dci = SpyreDCI([0], format=StickFormat.SPARSE)
    return output


@torch.library.custom_op("spyre::slice", mutates_args=(), device_types="spyre")
def slice(input: torch.Tensor) -> torch.Tensor:
    if len(input.size()) != 1:
        raise Unsupported("slice only implemented for 1-D tensors")
    output = input.new_empty(input.size())
    output.spyre_dci = SpyreDCI([0], format=StickFormat.DENSE)
    return output


@slice.register_fake
def _(input):
    if len(input.size()) != 1:
        raise Unsupported("slice only implemented for 1-D tensors")
    output = input.new_empty(input.size())
    output.spyre_dci = SpyreDCI([0], format=StickFormat.DENSE)
    return output


@torch.library.custom_op("spyre::layer_norm", mutates_args=())
def layer_norm(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre.layernorm: unsupported reduction shape {normalized_shape}"
        )
    return torch.native_layer_norm(x, normalized_shape, weight, bias, eps)[0]


@layer_norm.register_fake
def _(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    res = x.new_empty(x.size())
    res.spyre_dci = x.get_dci()
    return res


@torch.library.custom_op("spyre::exx2", mutates_args=(), device_types="spyre")
def exx2(x: torch.Tensor, exx2Scale: float, useZeroMean: bool) -> torch.Tensor:  # type: ignore[empty-body]
    pass


@exx2.register_fake
def _(x: torch.Tensor, exx2Scale: float, useZeroMean: bool):
    res_size, res_dci = spyre_reduction_result_shape(x, [x.ndim - 1], False)
    res_dci = SpyreDCI(res_dci.dim_order, format=StickFormat.SPARSE_MULTI)
    res = x.new_empty(res_size)
    res.spyre_dci = res_dci
    return res


@torch.library.custom_op("spyre::layernormscale", mutates_args=(), device_types="spyre")
def layernormscale(x: torch.Tensor, eps: float) -> torch.Tensor:  # type: ignore[empty-body]
    pass


@layernormscale.register_fake
def _(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_dci = x.get_dci()
    if x_dci.format != StickFormat.SPARSE_MULTI:
        raise Unsupported(f"layernormscale: Unexpected format {x_dci.format}")
    res_dci = SpyreDCI(x_dci.dim_order, format=StickFormat.SPARSE)
    res_size = list(x.size())
    res = x.new_empty(res_size)
    res.spyre_dci = res_dci
    return res


@torch.library.custom_op("spyre::layernormnorm", mutates_args=(), device_types="spyre")
def layernormnorm(  # type: ignore[empty-body]
    x: torch.Tensor,
    mean: torch.Tensor,
    norm_mean: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    pass


@layernormnorm.register_fake
def _(
    x: torch.Tensor,
    mean: torch.Tensor,
    norm_mean: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    res = x.new_empty(x.size())
    res.spyre_dci = x.get_dci()
    return res
