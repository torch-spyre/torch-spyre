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

from typing import Optional, Sequence
import torch
from torch_spyre.fallbacks import warn_fallback

from .errors import Unsupported


@torch.library.custom_op("spyre::compact", mutates_args=())
def compact(input: torch.Tensor) -> torch.Tensor:
    if len(input.size()) != 1:
        raise Unsupported("compact not implemented for 1-D tensors")
    return input.clone()


@compact.register_fake
def _(input):
    if len(input.size()) != 1:
        raise Unsupported("compact only implemented for 1-D tensors")
    return input.new_empty(input.size())


@torch.library.custom_op("spyre::swap", mutates_args=(), device_types="spyre")
def swap(input: torch.Tensor) -> torch.Tensor:
    if len(input.size()) != 1:
        raise Unsupported("swap only implemented for 1-D tensors")
    return input.new_empty_strided(input.size(), [64])


@swap.register_fake
def _(input):
    if len(input.size()) != 1:
        raise Unsupported("swap only implemented for 1-D tensors")
    return input.new_empty_strided(input.size(), [64])


@torch.library.custom_op("spyre::slice", mutates_args=(), device_types="spyre")
def slice(input: torch.Tensor) -> torch.Tensor:
    if len(input.size()) != 1:
        raise Unsupported("slice only implemented for 1-D tensors")
    output = input.new_empty(input.size())
    return output


@slice.register_fake
def _(input):
    if len(input.size()) != 1:
        raise Unsupported("slice only implemented for 1-D tensors")
    return input.new_empty(input.size())


@torch.library.custom_op("spyre::softplus", mutates_args=(), device_types="spyre")
def softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    pass


@softplus.register_fake
def _(input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0):
    return input.new_empty(input.size())


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
    return torch.native_layer_norm(x, normalized_shape, weight, bias, eps)[0].clone()


@layer_norm.register_fake
def _(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
):
    return x.new_empty(x.size())


@torch.library.custom_op("spyre::exx2", mutates_args=(), device_types="spyre")
def exx2(x: torch.Tensor, exx2Scale: float, useZeroMean: bool) -> torch.Tensor:  # type: ignore[empty-body]
    pass


@exx2.register_fake
def _(x: torch.Tensor, exx2Scale: float, useZeroMean: bool):
    return x.new_empty(x.size()[:-1])


@torch.library.custom_op("spyre::layernormscale", mutates_args=(), device_types="spyre")
def layernormscale(x: torch.Tensor, eps: float) -> torch.Tensor:  # type: ignore[empty-body]
    pass


@layernormscale.register_fake
def _(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x.new_empty(x.size())


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
    return x.new_empty(x.size())


@torch.library.custom_op("spyre::rms_norm", mutates_args=())
def rms_norm(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if len(normalized_shape) != 1:
        raise Unsupported(
            f"spyre.layernorm: unsupported reduction shape {normalized_shape}"
        )
    return torch.rms_norm(x, normalized_shape, weight, eps)


@rms_norm.register_fake
def _(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    return x.new_empty(x.size())


@torch.library.custom_op("spyre::gelu", mutates_args=(), device_types="spyre")
def gelu(
    input: torch.Tensor,
    approximate: str = "none",
) -> torch.Tensor:
    pass


@gelu.register_fake
def _(input: torch.Tensor, approximate: str = "none"):
    return input.new_empty(input.size())


@torch.library.custom_op("spyre::clamp", mutates_args=(), device_types="spyre")
def clamp(
    input: torch.Tensor,
    min: Optional[torch.types.Number] = None,
    max: Optional[torch.types.Number] = None,
) -> torch.Tensor:
    pass


@clamp.register_fake
def _(
    input: torch.Tensor,
    min: Optional[torch.types.Number] = None,
    max: Optional[torch.types.Number] = None,
):
    return input.new_empty(input.size())


@torch.library.custom_op("spyre::full", mutates_args=(), device_types="spyre")
def spyre_full(
    size: Sequence[int],
    fill_value: torch.types.Number,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    # Fall back to CPU.
    warn_fallback("torch.ops.spyre.full")
    tmp = torch.full(size, fill_value, dtype=dtype, device="cpu")
    return tmp.to(device)


@spyre_full.register_fake
def _(
    size: Sequence[int],
    fill_value: torch.types.Number,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
):
    return torch.empty(size, dtype=dtype, device="spyre")
