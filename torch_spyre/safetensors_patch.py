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

"""Monkey-patch safetensors to support Spyre as a valid device.

Resolves issue #400: safetensors does not natively support device="spyre".

The Rust backend of safetensors only recognizes "cpu" and "cuda" as device
strings. This module patches ``safe_open``, ``load_file``, and ``load_model``
so that ``device="spyre"`` loads tensors to CPU first, then transfers them
to Spyre.

The transfer uses a pluggable ``_tensor_to_spyre`` function. By default
this does a plain ``.to("spyre")``. The ``model_utils`` module (issue #1339)
can replace this with an optimized transfer that uses dim_order=[1,0] for
Linear weights.

Based on commits 3dbc7bf and 0866fb1 by pavi2707, enhanced with:
  * Full safe_open proxy (keys, metadata, get_slice, get_tensor)
  * Patches on both ``safetensors`` and ``safetensors.torch`` modules
  * load_file and load_model wrappers
  * Pluggable transfer function for layout optimization

Usage::

    from torch_spyre.safetensors_patch import patch_safetensors
    patch_safetensors()

    # Now these all work:
    safe_open(file, framework="pt", device="spyre")
    load_file(file, device="spyre")
    load_model(model, file, device="spyre")
"""

import logging

import torch
import torch.nn as nn

from torch_spyre.constants import DEVICE_NAME

logger = logging.getLogger(__name__)

_patched = False


# --- Pluggable transfer function ------------------------------------
#
# Default: plain .to("spyre"). model_utils.py can replace this with
# an optimized version that uses dim_order=[1,0] for Linear weights.

def _default_tensor_to_spyre(tensor: torch.Tensor, key: str = "") -> torch.Tensor:
    """Default transfer: plain .to("spyre"), no layout optimization."""
    return tensor.to(DEVICE_NAME)


# The active transfer function. model_utils.py can override this.
_tensor_to_spyre = _default_tensor_to_spyre


def set_tensor_to_spyre_fn(fn) -> None:
    """Replace the tensor-to-Spyre transfer function.

    Called by model_utils.py to plug in the dim_order=[1,0] optimized
    transfer. The function signature must be:
        fn(tensor: Tensor, key: str) -> Tensor
    """
    global _tensor_to_spyre
    _tensor_to_spyre = fn
    logger.debug("safetensors_patch: transfer function replaced by %s", fn)


def _assign_one(model, name, tensor, is_param, requires_grad=False):
    """Assign a single tensor to a model parameter or buffer."""
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        parent = model.get_submodule(parts[0])
        attr_name = parts[1]
    else:
        parent = model
        attr_name = parts[0]
    if is_param:
        parent._parameters[attr_name] = nn.Parameter(
            tensor, requires_grad=requires_grad
        )
    else:
        parent._buffers[attr_name] = tensor


# --- Patch -----------------------------------------------------------


def patch_safetensors() -> None:
    """Patch safetensors to support device="spyre". Idempotent."""
    try:
        import safetensors
        import safetensors.torch as st_torch
    except ImportError:
        logger.warning("safetensors not installed; patch has no effect")
        return

    if getattr(st_torch.load_file, "_spyre_patched", False):
        return

    orig_safe_open = st_torch.safe_open
    orig_load_file = st_torch.load_file
    orig_load_model = st_torch.load_model

    # --- safe_open wrapper -------------------------------------------

    class _SpyreSafeOpen:
        """Wrapper around safetensors.safe_open for Spyre device support.

        For device="spyre": opens the file with device="cpu" and
        transfers tensors to Spyre on get_tensor().

        For all other devices: delegates entirely to the original
        safe_open (zero overhead).
        """

        def __init__(self, filename, framework="pt", device="cpu"):
            self._target_is_spyre = (
                device is not None and torch.device(device).type == DEVICE_NAME
            )
            if self._target_is_spyre:
                self._inner = orig_safe_open(
                    filename, framework=framework, device="cpu"
                )
            else:
                self._inner = orig_safe_open(
                    filename, framework=framework, device=device
                )

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *args):
            return self._inner.__exit__(*args)

        def keys(self):
            return self._inner.keys()

        def metadata(self):
            return self._inner.metadata()

        def offset_keys(self):
            return self._inner.offset_keys()

        def get_tensor(self, name):
            tensor = self._inner.get_tensor(name)
            if self._target_is_spyre:
                return _tensor_to_spyre(tensor, name)
            return tensor

        def get_slice(self, name):
            return self._inner.get_slice(name)

    # --- load_file wrapper -------------------------------------------

    def _spyre_load_file(filename, device="cpu"):
        if device is None or torch.device(device).type != DEVICE_NAME:
            return orig_load_file(filename, device=device)
        cpu_tensors = orig_load_file(filename, device="cpu")
        return {k: _tensor_to_spyre(v, k) for k, v in cpu_tensors.items()}

    # --- load_model wrapper ------------------------------------------

    def _spyre_load_model(model, filename, strict=True, device="cpu"):
        if device is None or torch.device(device).type != DEVICE_NAME:
            return orig_load_model(
                model, filename, strict=strict, device=device
            )

        cpu_tensors = orig_load_file(filename, device="cpu")
        spyre_tensors = {
            k: _tensor_to_spyre(v, k) for k, v in cpu_tensors.items()
        }

        missing = []
        unexpected = set(spyre_tensors.keys())
        for name, param in model.named_parameters():
            if name in spyre_tensors:
                unexpected.discard(name)
                _assign_one(model, name, spyre_tensors[name],
                            is_param=True,
                            requires_grad=param.requires_grad)
            else:
                missing.append(name)
        for name, buf in model.named_buffers():
            if name not in spyre_tensors:
                missing.append(name)
                continue
            unexpected.discard(name)
            _assign_one(
                model, name, spyre_tensors[name], is_param=False
            )

        unexpected = sorted(unexpected)
        if strict and (missing or unexpected):
            error = (
                f"Error(s) in loading state_dict for "
                f"{model.__class__.__name__}:"
            )
            if missing:
                error += f"\n    Missing: {sorted(missing)}"
            if unexpected:
                error += f"\n    Unexpected: {sorted(unexpected)}"
            raise RuntimeError(error)
        return set(missing), unexpected

    # --- Install patches ---------------------------------------------

    st_torch.safe_open = _SpyreSafeOpen
    # safe_open is one object reachable under two module paths; callers
    # import from either, so rebind both names
    safetensors.safe_open = _SpyreSafeOpen
    _spyre_load_file._spyre_patched = True
    _spyre_load_model._spyre_patched = True
    st_torch.load_file = _spyre_load_file
    st_torch.load_model = _spyre_load_model
    logger.info(
        "Patched safetensors (safe_open, load_file, load_model) "
        "for Spyre device support (issue #400)"
    )
