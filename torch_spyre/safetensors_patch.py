"""
Simple monkey-patch for safetensors to support Spyre as a valid device by handling the cpu to spyre transfer.
TODO: load safetensor diectly on spyre
"""

import safetensors

from safetensors import safe_open as _original_safe_open
import torch


def patch_safetensors():
    def patched_safe_open(filename, framework="pt", device=None):
        if device is not None and torch.device(device).type == "spyre":
            return _SpyreSafetensorsFile(filename, device)
        return _original_safe_open(filename, framework, device)

    safetensors.safe_open = patched_safe_open


class _SpyreSafetensorsFile:
    def __init__(self, filename, device):
        self._file = _original_safe_open(filename, framework="pt", device="cpu")
        self._device = device

    def __enter__(self):
        self._file.__enter__()
        return self

    def __exit__(self, *args):
        return self._file.__exit__(*args)

    def get_tensor(self, name):
        return self._file.get_tensor(name).to(device=self._device)
