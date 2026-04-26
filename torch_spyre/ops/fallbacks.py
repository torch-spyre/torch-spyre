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


# How to add a CPU fallback operator:
#
# Step 1. Check if the target operator has a default decomposition.
#    - If yes, verify whether the decomposition expands into sub-ops that:
#        * Can be compiled, OR
#        * Correctly fall back to CPU.
#      If both conditions hold, no further action is needed.
#
#    - If some sub-ops cannot compile or fall back to CPU:
#        * Option A: Proceed to Step 2.
#        * Option B: Repeat Step 1 for each unsupported sub-op.
#
#    Example:
#      aten.arange decomposes into prims.iota, which only supports integer
#      dtypes. This requires int-to-float conversion, which Spyre does not
#      fully support yet. In this case, disable the default decomposition.
#      See: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_refs/__init__.py#L5124-L5222
#
# Step 2. Define an eager CPU fallback in: torch_spyre/fallbacks.py
#
#    Example:
#    @register_fallback([aten.sin.default, aten.sin.out])
#    def spyre__sin(input, **kwargs):
#        return torch.sin(input, **kwargs)
#
#    Note: You can identify the ATen operator name (e.g., aten.sin.default) by:
#      * torch.ops.aten.sin.overloads() # lists overloads like ['default', 'out']
#      * torch.ops.aten.sin.default     # OpOverload object for 'default'
#      * torch.ops.aten.sin._schema     # shows the dispatcher schema


import functools
import os
import warnings

import torch

aten = torch._ops.ops.aten

fallback_ops = list()


class FallbackWarning(UserWarning):
    """
    Warning issued when an operator runs on a fallback device (e.g., CPU)
    instead of Spyre
    """


warnings.simplefilter("once", FallbackWarning)
_warn_skips = (
    os.path.dirname(__file__),
    os.path.dirname(torch.__file__),
    torch._inductor.runtime.cache_dir_utils.cache_dir(),
)


def warn_fallback(op, fallback_device="cpu"):
    warnings.warn(
        f"{op} is falling back to {fallback_device}",
        category=FallbackWarning,
        skip_file_prefixes=_warn_skips,
    )


def register_fallback(ops, device="cpu"):
    """
    Decorator to register a CPU-fallback kernel for each op.

    - Moves all tensor inputs to the fallback device (default: CPU) before calling
      the wrapped function.
    - Executes the function on the fallback device, then returns the result on the
      target Spyre device.
    - If `out=` is provided, allocates a buffer tensor on the fallback device and
      copies the result into the `out` tensor.

    Target Spyre device resolution:
      - If `device=` is specified: treat it as the target device
      - Otherwise infer from tensor inputs; if none exist, use `torch.get_default_device()`

    Example:
        @register_fallback(["aten::op1", "aten::op1.out"]):
        def spyre_op1(input1, input2, **kwargs):
            return torch.op1(input1, input2, **kwargs)
    """

    fallback_device = torch.device(device)

    def _is_tensor(x):
        return isinstance(x, torch.Tensor)

    def _ensure_device(args, kwargs):
        # If `device=` was explicitly specified, use it as the target Spyre device
        spyre_device = kwargs.get("device")
        if spyre_device is not None:
            kwargs["device"] = fallback_device
            return torch.device(spyre_device)

        # Infer the target Spyre device from tensor inputs
        devices = {a.device for a in (*args, *kwargs.values()) if _is_tensor(a)}

        if not devices:
            # No tensor inputs and no 'device=' provided
            kwargs["device"] = fallback_device
            return torch.get_default_device()

        if len(devices) > 1:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but found: {devices}"
            )

        return devices.pop()

    def _move_tensors(args, kwargs):
        # Cache moved tensors to preserve aliasing and avoid redundant moves
        memo = {}

        def _move(t):
            key = id(t)
            moved = memo.get(key)
            if moved is None:
                # Preserve dtype when moving to fallback device
                moved = t.to(device=fallback_device, dtype=t.dtype)
                memo[key] = moved
            return moved

        for i, v in enumerate(args):
            if _is_tensor(v):
                args[i] = _move(v)

        for k, v in kwargs.items():
            if k != "out" and _is_tensor(v):
                kwargs[k] = _move(v)

        # Prepare `out` buffer on the fallback device; reuse alias if already moved
        out = kwargs.get("out")
        if out is not None:
            moved = memo.get(id(out))
            if moved is None:
                moved = torch.empty_like(out, device=fallback_device)
            kwargs["out"] = moved

    def _fallback(fn, *args, **kwargs):
        # Make args mutable
        args = list(args)

        # Validate 'out='
        out = kwargs.get("out")
        if out is not None and not _is_tensor(out):
            raise TypeError(f"argument 'out' must be Tensor, not {type(out)}")

        # Resolve the target Spyre device, and update 'device=' if necessary
        spyre_device = _ensure_device(args, kwargs)

        # Move input tensors to the fallback device
        _move_tensors(args, kwargs)

        # Compute on the fallback device
        fallback_result = fn(*args, **kwargs)

        # If 'out=' was specified, copy result into it
        if out is not None:
            out.copy_(fallback_result)
            return out

        # Otherwise, return result moved to the target Spyre device
        # Handle both single tensor and tuple/list of tensors
        def _move_to_spyre(result):
            if isinstance(result, (tuple, list)):
                # Handle tuple/list of tensors (e.g., torch.max returns (values, indices))
                moved = [_move_to_spyre(item) for item in result]
                # Preserve the original type (tuple or list)
                return type(result)(moved)

            return result.to(spyre_device)

        return _move_to_spyre(fallback_result)

    def _decorator(fn):
        for op in ops:

            @functools.wraps(fn)
            def _wrapped(*args, **kwargs):
                warn_fallback(op, fallback_device)
                return _fallback(fn, *args, **kwargs)

            fallback_ops.append(op)

            torch.library.register_kernel(op, ["spyre"])(_wrapped)
        return fn

    return _decorator


#  CPU-fallback eager operators


@register_fallback([aten.arange.default, aten.arange.start, aten.arange.start_step])
def spyre__arange(*args, **kwargs):
    return torch.arange(*args, **kwargs)


@register_fallback([aten.arange.out, aten.arange.start_out])
def spyre__arange_out(*args, out, **kwargs):
    kwargs.update({"device": "cpu", "dtype": out.dtype, "layout": out.layout})
    return torch.arange(*args, **kwargs)


@register_fallback([aten.sin.default, aten.sin.out])
def spyre__sin(input, **kwargs):
    return torch.sin(input, **kwargs)


@register_fallback([aten.cos.default, aten.cos.out])
def spyre__cos(input, **kwargs):
    return torch.cos(input, **kwargs)


# Manually append to fallback_ops: register_fallback cannot be used here because
# normal_ is an in-place op — register_fallback is designed for out-of-place ops
# and would leave the original Spyre tensor unfilled.
# The kernel itself is registered in ops.py (and therefore codegen_ops.py).
fallback_ops.append(aten.normal_.default)


@register_fallback([aten.embedding.default])
def spyre__embedding(
    weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False
):
    """
    Fallback for torch.nn.functional.embedding.

    Embedding requires indirect indexing (weight[indices]), which is not
    supported by Spyre's current pointwise operation framework.
    """
    # TODO: Remove this fallback once we enable gather/scatter ops on spyre
    return aten.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)


@register_fallback(
    [
        aten.isin.Tensor_Tensor,
        aten.isin.Tensor_Tensor_out,
        aten.isin.Tensor_Scalar,
        aten.isin.Tensor_Scalar_out,
        aten.isin.Scalar_Tensor,
        aten.isin.Scalar_Tensor_out,
    ]
)
def spyre__isin(
    elements, test_elements, *, assume_unique=False, invert=False, **kwargs
):
    """
    Fallback for torch.isin on Spyre.

    """
    return torch.isin(
        elements, test_elements, assume_unique=assume_unique, invert=invert, **kwargs
    )


@register_fallback([aten.tril.default, aten.tril.out])
def spyre__tril(input, diagonal=0, **kwargs):
    return torch.tril(input, diagonal, **kwargs)


@register_fallback([aten.triu.default, aten.triu.out])
def spyre__triu(input, diagonal=0, **kwargs):
    return torch.triu(input, diagonal, **kwargs)


@register_fallback([aten.bitwise_xor.Tensor, aten.bitwise_xor.Tensor_out])
def spyre__bitwise_xor(input1, input2, **kwargs):
    return torch.bitwise_xor(input1, input2, **kwargs)


@register_fallback([aten.bitwise_or.Tensor, aten.bitwise_or.Tensor_out])
def spyre__bitwise_or(input1, input2, **kwargs):
    return torch.bitwise_or(input1, input2, **kwargs)


@register_fallback([aten.argmax.default])
def spyre__argmax(*args, **kwargs):
    return torch.argmax(*args, **kwargs)


@register_fallback(["spyre::max_dim_int64_fallback"])
def spyre__max_dim_int64_fallback(input, dim, keepdim=False, **kwargs):
    """
    CPU fallback for torch.max(input, dim) when input is int64.
    """
    return torch.max(input, dim=dim, keepdim=keepdim, **kwargs)


@register_fallback(["spyre::max_default_int64_fallback"])
def spyre__max_default_int64_fallback(input, **kwargs):
    """
    CPU fallback for torch.max(input) when input is int64.

    Returns a scalar (0D) tensor containing the maximum value.
    This avoids recursive decomposition by directly calling torch.max on CPU.
    """
    return torch.max(input, **kwargs)
