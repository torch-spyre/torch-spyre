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
from torch._inductor.fx_passes.reinplace import inplaceable_ops, InplaceableOp
from torch_spyre.ops.fallbacks import warn_fallback
from torch_spyre.ops.eager import compile_once

from .errors import Unsupported


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
    return torch.compile(torch.ops.spyre.rms_norm)(x, normalized_shape, weight, eps)


@rms_norm.register_fake
def _(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    return x.new_empty(x.size())


@torch.library.custom_op("spyre::topkvalue", mutates_args=(), device_types="spyre")
def topkvalue(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    if len(x.size()) != 2:
        raise Unsupported("topk only implemented for 2-D tensors")
    pass


@topkvalue.register_fake
def _(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    if len(x.size()) != 2:
        raise Unsupported("topk only implemented for 2-D tensors")
    norm_dim = dim % len(x.size())
    out_size = list(x.size())
    out_size[norm_dim] = k
    return x.new_empty(out_size)


@torch.library.custom_op("spyre::topkindex", mutates_args=(), device_types="spyre")
def topkindex(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    if len(x.size()) != 2:
        raise Unsupported("topk only implemented for 2-D tensors")
    pass


@topkindex.register_fake
def _(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    if len(x.size()) != 2:
        raise Unsupported("topk only implemented for 2-D tensors")
    norm_dim = dim % len(x.size())
    out_size = list(x.size())
    out_size[norm_dim] = k
    return x.new_empty(out_size, dtype=torch.int64)


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


@torch.library.custom_op("spyre::logical_not", mutates_args=(), device_types="spyre")
def logical_not(input: torch.Tensor) -> torch.Tensor:
    pass


@logical_not.register_fake
def _(input: torch.Tensor):
    return input.new_empty(input.size())


@torch.library.custom_op("spyre::ones_scalar", mutates_args=(), device_types="spyre")
def spyre_ones_scalar(
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return a 1-element tensor containing 1 on Spyre. Used for ones via identity broadcast."""
    warn_fallback("torch.ops.spyre.ones_scalar")
    out = torch.empty(1, dtype=dtype, device=device)
    out.fill_(1)
    return out


@spyre_ones_scalar.register_fake
def _ones_scalar_fake(
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
):
    return torch.empty(1, dtype=dtype, device="spyre")


@torch.library.custom_op(
    "spyre::copy_from_d2d", mutates_args=("dst",), device_types="spyre"
)
@compile_once("spyre.copy_from_d2d")
def copy_from_d2d(
    src: torch.Tensor,
    dst: torch.Tensor,
    compiled,
) -> None:
    return compiled(src, dst)


@copy_from_d2d.register_fake
def _(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> None:
    pass


# Copy input into output starting at offsets along dimensions dims and
# return the updated output.
@torch.library.custom_op(
    "spyre::overwrite", mutates_args=("output",), device_types="spyre"
)
@compile_once("spyre.overwrite")
def overwrite(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
    compiled,
) -> None:
    return compiled(input, output, dims, offsets)


@overwrite.register_fake
def _(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
) -> None:
    return None


@torch.library.register_kernel("spyre::overwrite", ["cpu"])
def overwrite_cpu(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
) -> None:
    sliced_t = output
    for i, dim in enumerate(dims):
        sliced_t = torch.narrow(sliced_t, dim, offsets[i], 1)
    sliced_t.copy_(input)


@torch.library.custom_op("spyre::overwrite_f", mutates_args=(), device_types="spyre")
def overwrite_f(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
) -> torch.Tensor:
    result = output.clone()
    torch.ops.spyre.overwrite(input, result, dims, offsets)
    return result


@overwrite_f.register_fake
def _(
    input: torch.Tensor,
    output: torch.Tensor,
    dims: Sequence[int],
    offsets: Sequence[int],
) -> torch.Tensor:
    return output.clone()


inplaceable_ops[torch.ops.spyre.overwrite_f.default] = InplaceableOp(
    torch.ops.spyre.overwrite.default, 1
)


@torch.library.custom_op("spyre::restickify", mutates_args=(), device_types="spyre")
def restickify(  # type: ignore[empty-body]
    x: torch.Tensor,
) -> torch.Tensor:
    pass


@torch.library.custom_op("spyre::max_dim_int64_fallback", mutates_args=())
def max_dim_int64_fallback(
    input: torch.Tensor, dim: int, keepdim: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU fallback for torch.max(input, dim) when input is int64.
    This custom op will be registered with a CPU fallback in fallbacks.py.
    Returns a tuple (values, indices) as expected by torch.max.
    """
    # This should never be called directly; the fallback in fallbacks.py handles it
    raise RuntimeError(
        "spyre::max_dim_int64_fallback should be handled by CPU fallback registration"
    )


@max_dim_int64_fallback.register_fake
def _(input: torch.Tensor, dim: int, keepdim: bool = False):
    """
    Fake implementation for shape inference.
    Returns the expected output shapes for torch.max(input, dim, keepdim).
    """
    # Compute output shape based on dim and keepdim
    if keepdim:
        output_shape = list(input.size())
        output_shape[dim] = 1
    else:
        output_shape = list(input.size())
        output_shape.pop(dim)

    # Return tuple of (values, indices) with the computed shape
    values = input.new_empty(output_shape)
    indices = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    return (values, indices)


## TODO (imaihal): This needs scalar tensor support from Spyre to CPU. issues #1172
#
# @torch.library.custom_op("spyre::max_default_int64_fallback", mutates_args=())
# def max_default_int64_fallback(input: torch.Tensor) -> torch.Tensor:
#    """
#    CPU fallback for torch.max(input) when input is int64.
#    This custom op will be registered with a CPU fallback in fallbacks.py.
#    Returns a 1D tensor with shape [1] containing the maximum value.
#    """
#    # This should never be called directly; the fallback in fallbacks.py handles it
#    raise RuntimeError(
#        "spyre::max_default_int64_fallback should be handled by CPU fallback registration"
#    )
#
#
# @max_default_int64_fallback.register_fake
# def _(input: torch.Tensor):
#    """
#    Fake implementation for shape inference.
#    Returns a scalar (0D) tensor matching the input dtype.
#    """
#    return input.new_empty([])


@torch.library.custom_op("spyre::constant", mutates_args=(), device_types="spyre")
def spyre_constant(
    fill_value: torch.types.Number, dtype: torch.dtype, device: torch.device
) -> torch.types.Number:
    # This custom operator marks scalar constant in the FX graph.
    # Returning the scalar constant to avoid change in the operator schema which
    # consume the scalar constant as input.
    # This node will have a special handling at lowering to convert the scalar
    # constant to tensor.
    return fill_value


@spyre_constant.register_fake
def _constant(
    fill_value: torch.types.Number, dtype: torch.dtype, device: torch.device
) -> torch.types.Number:
    return fill_value
