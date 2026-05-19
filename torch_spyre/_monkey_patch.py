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

from typing import Optional
from torch_spyre._C import (
    get_spyre_tensor_layout,
    empty_with_layout,
    spyre_empty_with_layout,
    copy_tensor,
)

from torch._dynamo.guards import GuardBuilder
from torch_spyre._C import SpyreTensorLayout


def _patch_tensor_for_spyre():
    import torch

    if getattr(torch.Tensor, "_spyre_tensor_patched", False):
        return

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

    def device_tensor_layout(self: torch.Tensor) -> Optional[SpyreTensorLayout]:
        if self.device is not None and self.device.type == DEVICE_NAME:
            if isinstance(self, torch._subclasses.FakeTensor):
                return None  # catch FakeTensor BEFORE calling device_tensor_layout()
            return get_spyre_tensor_layout(self)
        else:
            return None

    def spyre_to(self, *args, device_layout=None, **kwargs):
        if (
            device_layout is None
        ):  # use original implementation if no layout is provided
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

            dst = spyre_empty_with_layout(
                self.size(), self.stride(), dtype, device_layout
            )

            if self.device.type == "cpu":
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
            return orig_empty(
                *args,
                out=out,
                dtype=dtype,
                layout=layout,
                device=device,
                requires_grad=requires_grad,
                pin_memory=pin_memory,
                memory_format=memory_format,
            )
        else:
            # layout_opt is omitted; c10::Layout has no pybind11 type caster,
            # so py_empty_with_layout drops that parameter and always uses
            # the default (Strided).
            return empty_with_layout(
                *args, device_layout, dtype, device, pin_memory, memory_format
            )

    torch.Tensor.__repr__ = spyre_aware_repr
    torch.Tensor.device_tensor_layout = device_tensor_layout
    torch.Tensor._spyre_tensor_patched = True
    torch.Tensor.to = spyre_to
    torch.empty = spyre_empty

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
        )

    GuardBuilder.TENSOR_MATCH = _spyre_TENSOR_MATCH
