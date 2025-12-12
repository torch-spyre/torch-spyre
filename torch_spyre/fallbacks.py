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


import functools
import os
import torch
import warnings


class FallbackWarning(UserWarning):
    """
    Warning issued when an operator runs on a fallback device (e.g., CPU)
    instead of Spyre
    """


warnings.simplefilter("once", FallbackWarning)
_warn_skips = (os.path.dirname(__file__), os.path.dirname(torch.__file__))


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
                moved = t.to(fallback_device)
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
        return fallback_result.to(spyre_device)

    def _decorator(fn):
        for op in ops:

            @functools.wraps(fn)
            def _wrapped(*args, **kwargs):
                warnings.warn(
                    f"{op} is falling back to {fallback_device}",
                    category=FallbackWarning,
                    skip_file_prefixes=_warn_skips,
                )
                return _fallback(fn, *args, **kwargs)

            torch.library.register_kernel(op, ["spyre"])(_wrapped)
        return fn

    return _decorator


#  CPU-fallback eager operators


@register_fallback(["aten::arange", "aten::arange.start", "aten::arange.start_step"])
def spyre__arange(*args, **kwargs):
    return torch.arange(*args, **kwargs)


@register_fallback(["aten::arange.out", "aten::arange.start_out"])
def spyre__arange_out(*args, out, **kwargs):
    kwargs.update({"device": "cpu", "dtype": out.dtype, "layout": out.layout})
    return torch.arange(*args, **kwargs)


@register_fallback(["aten::sin", "aten::sin.out"])
def spyre__sin(input, **kwargs):
    return torch.sin(input, **kwargs)


@register_fallback(["aten::cos", "aten::cos.out"])
def spyre__cos(input, **kwargs):
    return torch.cos(input, **kwargs)
