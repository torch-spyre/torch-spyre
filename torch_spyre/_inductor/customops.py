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
import torch._dynamo
from torch._inductor.fx_passes.reinplace import inplaceable_ops, InplaceableOp
from torch_spyre.ops.eager import compile_once
from torch_spyre.ops.fallbacks import warn_fallback

from .errors import Unsupported

aten = torch.ops.aten


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


@torch.library.custom_op("spyre::silu", mutates_args=(), device_types="spyre")
def silu(
    input: torch.Tensor,
) -> torch.Tensor:
    return aten.div.Tensor(input, 1 + aten.exp.default(-input))


@silu.register_fake
def _(input: torch.Tensor):
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


@torch.library.custom_op("spyre::empty", mutates_args=(), device_types="spyre")
def spyre_empty(
    size: Sequence[int],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    # Eager-mode simulation: allocate on CPU and move to the Spyre device.
    # This is not a compute fallback — on hardware the compiled kernel receives
    # a device allocation from SpyreAllocator with no host-side initialisation.
    tmp = torch.empty(size, dtype=dtype, device="cpu")
    return tmp.to(device)


@spyre_empty.register_fake
def _(
    size: Sequence[int],
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


# Copy src into dst in-place (the mutating primitive).
# No @compile_once needed: the body calls aten::copy_ which dispatches
# through spyre__copy_from → copy_from_d2d / copy_tensor — no cycle back
# to spyre::copy_ itself.
@torch.library.custom_op("spyre::copy_", mutates_args=("dst",), device_types="spyre")
def copy_(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    dst.copy_(src)
    return dst


@copy_.register_fake
def _(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return dst


@torch.library.register_kernel("spyre::copy_", ["cpu"])
def copy__cpu(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    dst.copy_(src)
    return dst


# Functional (out-of-place) wrapper for spyre::copy_.
# Returns a new tensor equal to dst with src written into it.
# The graph sees a pure value-producing node; Inductor's reinplacer will
# rewrite copy_f → copy_ (in-place) wherever it is safe to do so.
@torch.library.custom_op("spyre::copy_f", mutates_args=(), device_types="spyre")
def copy_f(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    result = dst.clone()
    torch.ops.spyre.copy_(src, result)
    return result


@copy_f.register_fake
def _(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return dst.clone()


inplaceable_ops[torch.ops.spyre.copy_f.default] = InplaceableOp(
    torch.ops.spyre.copy_.default, 1
)


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
    # specialize_int=True installs int-equality guards on the int-list
    # args so each unique (dims, offsets) triggers a fresh trace and a
    # fresh SDSC binary; without this dynamo's default specialize_int=
    # False reuses one baked binary across all values and scatters all
    # writes to the first call's offset (see test_overwrite.py).
    # Patch is call-scoped to leave process-wide dynamo behavior alone.
    # Note: this gives one compiled binary per unique (input shape, dims,
    # offsets) tuple. dynamo's cache_size_limit is bumped to 1024 in
    # torch_spyre/__init__.py — long-running workloads that scatter into
    # many distinct slots can blow past that. Symbolic offsets (one
    # binary, any value) are tracked in issues #220 / #1371-3.
    with torch._dynamo.config.patch(specialize_int=True):
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
        sliced_t = torch.narrow(sliced_t, dim, offsets[i], input.size(dim))
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


@torch.library.custom_op("spyre::unfold", mutates_args=(), device_types="spyre")
def spyre_unfold(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Optional[Sequence[int]] = None,
    padding: Optional[Sequence[int]] = None,
    stride: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Im2col unfold operation via torch.nn.functional.unfold.
    Converts (N, C, H, W) input to (N, C*K_h*K_w, L) where L = H_out * W_out.
    Uses CPU fallback for the unfold operation.
    """

    dilation = dilation or (1, 1)
    padding = padding or (0, 0)
    stride = stride or (1, 1)

    warn_fallback("torch.ops.spyre.unfold")
    # Move to CPU, perform unfold, move back to Spyre
    input_cpu = input.to("cpu")
    result_cpu = torch.nn.functional.unfold(
        input_cpu,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    return result_cpu.to(input.device)


@spyre_unfold.register_fake
def _(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    dilation: Optional[Sequence[int]] = None,
    padding: Optional[Sequence[int]] = None,
    stride: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    dilation = dilation or (1, 1)
    padding = padding or (0, 0)
    stride = stride or (1, 1)

    N, C, H_in, W_in = input.shape
    K_h, K_w = kernel_size
    dil_h, dil_w = dilation
    pad_h, pad_w = padding
    stride_h, stride_w = stride

    H_out = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1

    return input.new_empty((N, C * K_h * K_w, H_out * W_out))


@torch.library.custom_op(
    "spyre::reshape_via_cpu", mutates_args=(), device_types="spyre"
)
def spyre_reshape_via_cpu(
    input: torch.Tensor,
    shape: Sequence[int],
) -> torch.Tensor:
    """
    Reshape operation that executes on CPU to avoid stick-alignment issues.

    When reshaping produces a shape with innermost dimension that doesn't align
    with stick boundaries (64 elements for fp16), the Inductor coordinate
    computation fails. This op moves to CPU, reshapes, then moves back to Spyre.

    This is similar to unfold, which also uses CPU fallback for correct layouts.
    """
    warn_fallback("torch.ops.spyre.reshape_via_cpu")
    input_cpu = input.to("cpu")
    result_cpu = input_cpu.reshape(shape)
    return result_cpu.to(input.device)


@spyre_reshape_via_cpu.register_fake
def _(
    input: torch.Tensor,
    shape: Sequence[int],
) -> torch.Tensor:
    return input.new_empty(shape)


@torch.library.custom_op("spyre::min_dim_int64_fallback", mutates_args=())
def min_dim_int64_fallback(
    input: torch.Tensor, dim: int, keepdim: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU fallback for torch.min(input, dim) when input is int64.
    This custom op will be registered with a CPU fallback in fallbacks.py.
    Returns a tuple (values, indices) as expected by torch.min.
    """
    raise RuntimeError(
        "spyre::min_dim_int64_fallback should be handled by CPU fallback registration"
    )


@min_dim_int64_fallback.register_fake
def _(input: torch.Tensor, dim: int, keepdim: bool = False):
    """
    Fake implementation for shape inference.
    Returns the expected output shapes for torch.min(input, dim, keepdim).
    """
    if keepdim:
        output_shape = list(input.size())
        output_shape[dim] = 1
    else:
        output_shape = list(input.size())
        output_shape.pop(dim)

    values = input.new_empty(output_shape)
    indices = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    return (values, indices)


@torch.library.custom_op("spyre::max_default_int64_fallback", mutates_args=())
def max_default_int64_fallback(input: torch.Tensor) -> torch.Tensor:
    """
    CPU fallback for torch.max(input) when input is int64.
    This custom op will be registered with a CPU fallback in fallbacks.py.
    Returns a 1D tensor with shape [1] containing the maximum value.
    """
    # This should never be called directly; the fallback in fallbacks.py handles it
    raise RuntimeError(
        "spyre::max_default_int64_fallback should be handled by CPU fallback registration"
    )


@max_default_int64_fallback.register_fake
def _(input: torch.Tensor):
    """
    Fake implementation for shape inference.
    Returns a scalar (0D) tensor matching the input dtype.
    """
    return input.new_empty([])


@torch.library.custom_op("spyre::batched_matmul", mutates_args=(), device_types="spyre")
def batched_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[empty-body]
    pass


@batched_matmul.register_fake
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output_shape = list(x.shape[:-1]) + [y.shape[-1]]
    return x.new_empty(output_shape)


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


@torch.library.custom_op("spyre::to_dtype_cpu", mutates_args=(), device_types="spyre")
def to_dtype_cpu(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    warn_fallback(f"conversion from {input.dtype} to {dtype}")
    return input.cpu().to(dtype=dtype).to(input.device)


@to_dtype_cpu.register_fake
def _(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty_like(input, dtype=dtype)


@torch.library.custom_op("spyre::qfp8ch", mutates_args=(), device_types="spyre")
def qfp8ch(input: torch.Tensor) -> torch.Tensor:
    """
    Channel-wise FP8 format conversion (pointwise, optimized for matmul).

    Converts input tensor to FP8 E4M3 format with channel-wise semantics.
    This operation ONLY performs format conversion - scaling must be done separately.

    Args:
        input: Input tensor (FP16/BF16/FP32) to convert to FP8
               Should already be scaled and clamped

    Returns:
        FP8 E4M3 tensor (same shape as input)

    Maps to: deeptools Qfp8ch operation
    """
    pass


@qfp8ch.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    # Output is FP8 with same shape as input
    return torch.empty(input.size(), dtype=torch.float8_e4m3fn, device=input.device)


@torch.library.custom_op(
    "spyre::quantize_fp8_with_scale", mutates_args=(), device_types="spyre"
)
def quantize_fp8_with_scale(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Quantize FP16 tensor to FP8 using pre-computed scale.

    Performs four steps:
    1. Compute inverse scale: inv_scale = 1 / scale (reciprocal, POINTWISE on sfp unit)
    2. Scale the input: x_scaled = x * inv_scale (POINTWISE)
    3. Clamp to FP8 E4M3 range: x_clamped = clamp(x_scaled, -448, 448) (POINTWISE)
    4. Convert to FP8 format: x_fp8 = qfp8ch(x_clamped) (POINTWISE format conversion)

    Args:
        input: Input tensor (FP16) to quantize, shape [batch, seq, hidden]
        scale: Quantization scale (FP16), shape [batch, seq, 1]

    Returns:
        FP8 E4M3 tensor (same shape as input)

    Example:
        >>> x = torch.randn(2, 4, 8, dtype=torch.float16, device='spyre')
        >>> x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale)

    Note:
        - Uses reciprocal operation (hardware sfp unit) for 1/scale computation
    """
    pass


@quantize_fp8_with_scale.register_fake
def _(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Output is FP8 with same shape as input
    return torch.empty(input.size(), dtype=torch.float8_e4m3fn, device=input.device)


@torch.library.custom_op(
    "spyre::dequantize_fp8_with_scale", mutates_args=(), device_types="spyre"
)
def dequantize_fp8_with_scale(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:  # type: ignore[empty-body]
    """
    Dequantize FP8 tensor to FP16 using pre-computed scale.
    Performs two steps:
    1. Convert FP8 to FP16: x_fp16 = fp8todl16(x) (dtype conversion)
    2. Scale the output: x_scaled = x_fp16 * scale (POINTWISE)
    Args:
        input: Input tensor (FP8) to dequantize, shape [batch, seq, hidden]
        scale: Dequantization scale (FP16), shape [batch, seq, 1]
    Returns:
        FP16 tensor (same shape as input)
    Example:
        >>> @torch.compile(backend='inductor')
        >>> def dequant(x_fp8, scale):
        >>>     return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)
    Note:
        - MUST use torch.compile(backend='inductor') - does not work in eager mode
        - Uses fp8todl16 operation for FP8→FP16 conversion
        - Scale must be FP16, NOT FP32
    """
    pass


@dequantize_fp8_with_scale.register_fake
def _(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Output is FP16 with same shape as input
    return torch.empty(input.size(), dtype=torch.float16, device=input.device)


@torch.library.custom_op("spyre::causal_mask", mutates_args=(), device_types="spyre")
def causal_mask(
    seqlen_q: int,
    seqlen_kv: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a causal additive mask on CPU and transfer to the target device.

    Shape: [1, 1, seqlen_q, seqlen_kv] in natural orientation: query i attends
    to keys 0..i.  Entries are 0.0 (keep) or -inf (masked).  The kept diagonal
    guarantees no fully-masked row (no 0/0 NaN denominator).

    Built entirely on CPU (tril + masked_fill_) so those in-place ops are
    opaque to torch.compile — assert_functional_graph is satisfied and the
    compiled graph only sees the resulting device tensor.  No device_types
    restriction is set because there are no tensor arguments to dispatch on;
    the device is an explicit parameter.
    """
    # Causal boolean lower-triangular pattern: True = attend, False = mask
    causal_cpu = torch.tril(
        torch.ones(seqlen_q, seqlen_kv, dtype=torch.bool, device="cpu")
    )
    mask_cpu = torch.zeros(1, 1, seqlen_q, seqlen_kv, dtype=dtype, device="cpu")
    mask_cpu.masked_fill_(~causal_cpu, float("-inf"))
    return mask_cpu.to(device=device)


@causal_mask.register_fake
def _(
    seqlen_q: int,
    seqlen_kv: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(1, 1, seqlen_q, seqlen_kv, dtype=dtype, device=device)


@torch.library.custom_op("spyre::prod_dim_int", mutates_args=(), device_types="spyre")
def prod_dim_int(input: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    pass


@prod_dim_int.register_fake
def _(input: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    if dim < 0:
        dim += input.ndim
    out_shape = list(input.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape = out_shape[:dim] + out_shape[dim + 1 :]
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)
