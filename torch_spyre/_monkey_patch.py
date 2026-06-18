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


from torch_spyre.constants import DEVICE_NAME

from typing import Optional, TYPE_CHECKING

from torch._dynamo.guards import GuardBuilder

if TYPE_CHECKING:
    from torch_spyre._C import SpyreTensorLayout


def _patch_tensor_for_spyre():
    import torch

    if getattr(torch.Tensor, "_spyre_tensor_patched", False):
        return

    from torch.utils._device import _device_constructors

    _device_constructors()  # warm the cache with the original torch.empty

    orig_repr = torch.Tensor.__repr__
    orig_to = torch.Tensor.to
    orig_empty = torch.empty

    def spyre_aware_repr(self):
        dev = getattr(self, "device", None)
        if dev is not None and dev.type == DEVICE_NAME:
            try:
                s = orig_repr(self.to("cpu"))
            except Exception:
                # Fallback if .to("cpu") fails for some weird reason
                return (
                    f"SpyreTensor(shape={tuple(self.shape)}, "
                    f"dtype={self.dtype}, device={self.device})"
                )
            if "device=" in s:
                return s.replace("device='cpu'", f"device='{self.device}'")
            if s.endswith(")"):
                s = s[:-1] + f", device='{self.device}')"
            else:
                # Odd case: just append device info
                s = s + f" (device='{self.device}')"
            return s

        # Non-spyre tensors use normal behavior
        return orig_repr(self)

    def device_tensor_layout(self: torch.Tensor) -> Optional["SpyreTensorLayout"]:
        if self.device is not None and self.device.type == DEVICE_NAME:
            if isinstance(self, torch._subclasses.FakeTensor):
                return None  # catch FakeTensor BEFORE calling device_tensor_layout()
            from torch_spyre._C import get_spyre_tensor_layout

            return get_spyre_tensor_layout(self)
        else:
            return None

    def spyre_to(self, *args, device_layout=None, **kwargs):
        if device_layout is None:
            # Support D2H and H2D dtype casting via DCI (DataConversionInfo) in spyre_mem.cpp.
            # For D2D data casting, split it into a D2H copy and a H2D dtype conversion.
            _device = kwargs.get("device", None)
            if (
                _device is None
                and len(args) > 0
                and isinstance(args[0], (str, torch.device))
            ):
                _device = args[0]
            _dtype = kwargs.get("dtype", None)
            if _dtype is None and len(args) > 1 and isinstance(args[1], torch.dtype):
                _dtype = args[1]

            if (
                _device is not None
                and _dtype is not None
                and self.device.type == DEVICE_NAME
                and torch.device(_device).type == DEVICE_NAME
            ):
                import warnings

                warnings.warn(
                    "D2D dtype conversion on Spyre is not directly supported. "
                    "Using CPU as an intermediate for the cast.",
                    stacklevel=2,
                )
                # Step 1: plain D2H copy (no dtype change)
                tmp = orig_to(self, "cpu")
                # Step 2: cast dtype via H2D
                return orig_to(tmp, _device, dtype=_dtype)
            return orig_to(self, *args, **kwargs)
        else:
            # Check if copy kwarg is explicitly set
            copy = kwargs.get("copy")

            # Determine dtype from various possible sources
            dtype = None
            if len(args) > 0:
                # If args[0] is a dtype instance, use it
                if isinstance(args[0], torch.dtype):
                    dtype = args[0]
                # If args[0] is a Tensor, use its dtype
                elif isinstance(args[0], torch.Tensor):
                    dtype = args[0].dtype

            # Check for dtype in kwargs
            if dtype is None and "dtype" in kwargs:
                dtype = kwargs["dtype"]

            # Check for tensor kwarg
            if dtype is None and "tensor" in kwargs:
                tensor_arg = kwargs["tensor"]
                if isinstance(tensor_arg, torch.Tensor):
                    dtype = tensor_arg.dtype

            # Fall back to self.dtype if no dtype was specified
            if dtype is None:
                dtype = self.dtype

            from torch_spyre._C import spyre_empty_with_layout

            dst = spyre_empty_with_layout(
                self.size(), self.stride(), dtype, device_layout
            )

            if self.device.type == "cpu":
                from torch_spyre._C import copy_tensor

                copy_tensor(self, dst, non_blocking=False)
                return dst
            else:  # device to device copy
                # If device_layout is the same as self and copy is not True, return self
                current_layout = device_tensor_layout(self)
                if (
                    not copy
                    and current_layout is not None
                    and current_layout == device_layout
                ):
                    return self
                else:
                    return torch.ops.spyre.copy_from_d2d(self, dst)

    def spyre_empty(
        *args,
        device_layout=None,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
        memory_format=torch.contiguous_format,
    ):
        if (
            device_layout is None
        ):  # use original implementation if no layout is provided
            kwargs = dict(
                out=out,
                dtype=dtype,
                layout=layout,
                requires_grad=requires_grad,
                pin_memory=pin_memory,
                memory_format=memory_format,
            )
            if device is not None:
                kwargs["device"] = device
            return orig_empty(*args, **kwargs)
        else:
            # layout_opt is omitted; c10::Layout has no pybind11 type caster,
            # so py_empty_with_layout drops that parameter and always uses
            # the default (Strided).
            from torch_spyre._C import empty_with_layout

            return empty_with_layout(
                *args, device_layout, dtype, device, pin_memory, memory_format
            )

    torch.Tensor.__repr__ = spyre_aware_repr
    torch.Tensor.device_tensor_layout = device_tensor_layout
    torch.Tensor._spyre_tensor_patched = True
    torch.Tensor.to = spyre_to
    torch.empty = spyre_empty

    # ── Optimal weight loading (issue #1339) ──────────────
    # Patch dim_order=[1,0] transfer + nn.Module.to override (issue #1339).
    try:
        from torch_spyre.model_utils import patch_module_to_for_spyre

        patch_module_to_for_spyre()
    except Exception as e:  # pragma: no cover - defensive
        import warnings

        warnings.warn(f"Failed to install optimal weight layout patches: {e}")

    # ── SpyreTensorLayout Guard Extension ────────────
    # Extends TENSOR_MATCH to guard on SpyreTensorLayout
    # preventing wrong compiled graph reuse when layout
    # changes.
    # ─────────────────────────────────────────────────

    _original_TENSOR_MATCH = GuardBuilder.TENSOR_MATCH

    def _spyre_TENSOR_MATCH(self, guard, value=None):
        # run original TENSOR_MATCH
        _original_TENSOR_MATCH(self, guard, value=value)
        # get tensor value
        if value is None:
            value = self.get(guard)
        ## dereference WeakRef if needed
        if isinstance(value, torch.utils.weak.TensorWeakRef):
            value = value()

        if value is None:
            return

        # not a Spyre tensor → skip
        if value.device.type != DEVICE_NAME:
            return

        # get layout safely
        expected_layout = value.device_tensor_layout()
        if expected_layout is None:
            return

        # add lambda guard on tensor's child manager
        # same node as TENSOR_MATCH!
        tensor_guard_manager = self.get_guard_manager(guard)
        tensor_guard_manager.add_lambda_guard(
            lambda x: (
                x.device.type != DEVICE_NAME
                or x.device_tensor_layout() == expected_layout
            ),
            [f"SpyreTensorLayout({guard.name}) == {expected_layout}"],
            guard.user_stack,
        )

    GuardBuilder.TENSOR_MATCH = _spyre_TENSOR_MATCH
    # ───────────────────FxGraph Cache Key Extension ───────────────────
    # Extends FxGraphHashDetails to include SpyreTensorLayout in the cache key
    # preventing incorrect disk cache hits across process boundaries.
    # ──────────────────────────────────────────────────────────────────────────
    _patch_fx_graph_hash()


def _patch_fx_graph_hash():
    """
    Extends FxGraphHashDetails to include SpyreTensorLayout in the cache key.
    """
    import torch
    from torch._inductor.codecache import FxGraphHashDetails
    from torch._inductor.virtualized import V

    if getattr(FxGraphHashDetails, "_spyre_hash_patched", False):
        return

    original_init = FxGraphHashDetails.__init__

    def _spyre_init(self, gm, example_inputs, fx_kwargs, inputs_to_check):
        # run original first — populates all standard hash fields
        original_init(self, gm, example_inputs, fx_kwargs, inputs_to_check)

        # V.get_real_inputs() returns real Spyre tensors with SpyreTensorLayout
        # before they become FakeTensors (which have no layout by design)

        try:
            real_inputs = V.get_real_inputs()
        except RuntimeError:
            return

        # extract layout from real tensors, fallback to example_inputs
        spyre_layouts = []
        # Use real_inputs only if it's a valid list/tuple, otherwise use example_inputs
        inputs_to_use = (
            real_inputs if isinstance(real_inputs, (list, tuple)) else example_inputs
        )

        for inp in inputs_to_use:
            if isinstance(inp, torch.Tensor):
                layout = inp.device_tensor_layout()
                spyre_layouts.append(layout)
            else:
                spyre_layouts.append(None)

        # self.spyre_layouts added as field on FxGraphHashDetails
        # PyTorch pickles ALL fields → spyre_layouts automatically in hash
        self.spyre_layouts = spyre_layouts

    FxGraphHashDetails.__init__ = _spyre_init
    FxGraphHashDetails._spyre_hash_patched = True
