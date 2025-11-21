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
from torch_spyre._C import get_spyre_tensor_layout
from torch_spyre._C import SpyreTensorLayout


def _patch_tensor_for_spyre():
    import torch

    if getattr(torch.Tensor, "_spyre_tensor_patched", False):
        return

    orig_repr = torch.Tensor.__repr__

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

    torch.Tensor.__repr__ = spyre_aware_repr
    torch.Tensor.device_tensor_layout = device_tensor_layout
    torch.Tensor._spyre_tensor_patched = True
