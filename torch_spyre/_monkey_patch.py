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
from torch_spyre._C import get_spyre_tensor_layout, to_with_layout, empty_with_layout
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
        if dev is not None and dev.type == "spyre":
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
        if self.device is not None and self.device.type == "spyre":
            return get_spyre_tensor_layout(self)
        else:
            return None

    def spyre_to(self, *args, device_layout=None, **kwargs):
        if (
            device_layout is None
        ):  # use original implementation if no layout is provided
            return orig_to(self, *args, **kwargs)
        else:
            return to_with_layout(self, device_layout)

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
            return empty_with_layout(
                *args, device_layout, dtype, None, device, pin_memory, memory_format
            )

    torch.Tensor.__repr__ = spyre_aware_repr
    torch.Tensor.device_tensor_layout = device_tensor_layout
    torch.Tensor._spyre_tensor_patched = True
    torch.Tensor.to = spyre_to
    torch.empty = spyre_empty
