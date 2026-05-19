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

"""Optimal weight layout utilities for loading models onto Spyre.

Transfers ``nn.Linear`` weights to Spyre with a device layout where the
``out_features`` dimension is stickified (the optimal layout for Spyre
matmul where both operands need their rows in the stick).

This is achieved using ``dim_order=[1, 0]`` in ``SpyreTensorLayout``,
which tells the DMA engine to stickify along host dim-0 (out_features)
instead of the default last dim (in_features). No CPU transpose or
intermediate copy is required.

Critically, the tensor's PyTorch shape stays ``(out, in)`` -- only the
*device* layout changes. This means:

  * ``nn.Linear.forward`` works unmodified (no class swap, no forward
    patch, no flag tracking).
  * ``F.linear`` / ``aten.linear`` works unmodified. The Spyre
    decomposition still does ``weight.transpose(-1, -2)`` (a metadata-
    only op), and the Spyre layout propagation engine recognizes the
    stickification matches the matmul's needs -- no restickify cost.
  * Models loaded with this utility are drop-in compatible with all
    existing inference paths.

Resolves:
  * Issue #1339 (optimal weight layout for Spyre)

Usage::

    # Explicit:
    from torch_spyre.model_utils import load_model_to_spyre
    load_model_to_spyre(model)

    # Transparent for any code that uses .to("spyre"):
    from torch_spyre.model_utils import patch_module_to_for_spyre
    patch_module_to_for_spyre()
    model.to("spyre")

    # For safetensors-based loaders (HF transformers, vLLM):
    from torch_spyre.model_utils import patch_safetensors_for_spyre
    patch_safetensors_for_spyre()
    # safetensors.torch.load_file/load_model with device="spyre" will
    # now use the optimal layout transparently.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from torch_spyre._C import (
    SpyreTensorLayout,
    copy_tensor,
    spyre_empty_with_layout,
)
from torch_spyre.constants import DEVICE_NAME

logger = logging.getLogger(__name__)


# --- DMA helpers -----------------------------------------------------


def _dma_to_spyre_default(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """Transfer a CPU tensor to Spyre with the default layout.

    Used for non-Linear-weight tensors (biases, embeddings, layer norm
    parameters, buffers). Stickifies along the last dimension.
    """
    # Ensure tensor is FP16 (Spyre doesn't support dtype conversion during copy)
    if cpu_tensor.dtype != torch.float16:
        cpu_tensor = cpu_tensor.to(dtype=torch.float16)
    
    if not cpu_tensor.is_contiguous():
        cpu_tensor = cpu_tensor.contiguous()
    layout = SpyreTensorLayout(list(cpu_tensor.shape), cpu_tensor.dtype)
    dst = spyre_empty_with_layout(
        cpu_tensor.size(), cpu_tensor.stride(), cpu_tensor.dtype, layout
    )
    copy_tensor(cpu_tensor, dst, non_blocking=False)
    return dst


def _dma_to_spyre_dim_order_swapped(weight: torch.Tensor) -> torch.Tensor:
    """Transfer a 2D Linear weight to Spyre with dim_order=[1, 0].

    The host tensor shape ``(out_features, in_features)`` is preserved
    on the device, but the data is stickified along ``out_features``
    (dim 0) rather than the default ``in_features`` (dim 1). This
    matches the layout Spyre needs for efficient matmul and avoids
    both a CPU transpose and a device-side restickify.

    Caller must ensure ``weight.ndim == 2``.
    """
    assert weight.ndim == 2, "dim_order=[1,0] path is for 2D weights only"
    
    # Ensure tensor is FP16 (Spyre doesn't support dtype conversion during copy)
    if weight.dtype != torch.float16:
        weight = weight.to(dtype=torch.float16)
    
    if not weight.is_contiguous():
        weight = weight.contiguous()
    layout = SpyreTensorLayout(
        list(weight.shape),       # host_size: (out, in)
        list(weight.stride()),    # host_strides: row-major
        weight.dtype,
        [1, 0],                   # dim_order: stick on dim-0 = out_features
    )
    dst = spyre_empty_with_layout(
        weight.size(), weight.stride(), weight.dtype, layout
    )
    copy_tensor(weight, dst, non_blocking=False)
    return dst


# --- Model loading ---------------------------------------------------


def load_model_to_spyre(
    model: nn.Module,
    dtype: Optional[torch.dtype] = None,
    device_index: int = 0,
) -> nn.Module:
    """Transfer model to Spyre with optimal weight layout.

    For each ``nn.Linear``, the weight is transferred using
    ``dim_order=[1, 0]`` so that ``out_features`` is stickified
    (optimal for Spyre matmul). Tensor shapes are preserved, so the
    model works unmodified with the existing inference path.

    All other parameters and buffers use the default Spyre layout.

    Idempotent: parameters already on Spyre are skipped.
    """
    linear_count = 0
    other_param_count = 0
    buffer_count = 0

    for name, module in model.named_modules():
        is_linear = isinstance(module, nn.Linear)

        for param_name, param in list(module._parameters.items()):
            if param is None:
                continue
            if param.device.type == DEVICE_NAME:
                continue

            p = param.data
            if dtype is not None:
                p = p.to(dtype=dtype)

            # 2D Linear weight -> optimal stickified layout via dim_order.
            # Everything else (bias, embeddings, norms, ...) -> default layout.
            if is_linear and param_name == "weight" and p.ndim == 2:
                logger.debug(
                    "  %s.%s: shape=%s -> Spyre dim_order=[1, 0]",
                    name, param_name, list(p.shape),
                )
                dev = _dma_to_spyre_dim_order_swapped(p)
                linear_count += 1
            else:
                logger.debug(
                    "  %s.%s: shape=%s -> Spyre default layout",
                    name, param_name, list(p.shape),
                )
                dev = _dma_to_spyre_default(p)
                other_param_count += 1

            module._parameters[param_name] = nn.Parameter(
                dev, requires_grad=param.requires_grad
            )

        for buf_name, buf in list(module._buffers.items()):
            if buf is None or buf.device.type == DEVICE_NAME:
                continue
            b = buf
            if dtype is not None:
                b = b.to(dtype=dtype)
            module._buffers[buf_name] = _dma_to_spyre_default(b)
            buffer_count += 1

    logger.info(
        "load_model_to_spyre: %d Linear weights optimized "
        "(dim_order=[1,0]), %d other params and %d buffers "
        "transferred with default layout",
        linear_count, other_param_count, buffer_count,
    )
    return model


# --- nn.Module.to() monkeypatch --------------------------------------

_module_to_patched = False


def patch_module_to_for_spyre() -> None:
    """Monkeypatch ``nn.Module.to`` for automatic optimal Spyre loading.

    After patching, ``model.to("spyre")`` will use the optimal weight
    layout for every ``nn.Linear`` in the model. Non-Spyre destinations
    fall through to the original ``nn.Module.to``.

    Idempotent.
    """
    global _module_to_patched
    if _module_to_patched:
        return

    orig_module_to = nn.Module.to

    def _spyre_module_to(self, *args, **kwargs):
        target_is_spyre = False

        for arg in args:
            if isinstance(arg, str) and arg.startswith(DEVICE_NAME):
                target_is_spyre = True
                break
            if isinstance(arg, torch.device) and arg.type == DEVICE_NAME:
                target_is_spyre = True
                break

        device_kwarg = kwargs.get("device")
        if device_kwarg is not None:
            if isinstance(device_kwarg, str) and device_kwarg.startswith(
                DEVICE_NAME
            ):
                target_is_spyre = True
            elif (
                isinstance(device_kwarg, torch.device)
                and device_kwarg.type == DEVICE_NAME
            ):
                target_is_spyre = True

        if not target_is_spyre:
            return orig_module_to(self, *args, **kwargs)

        dtype = kwargs.get("dtype")
        for arg in args:
            if isinstance(arg, torch.dtype):
                dtype = arg

        return load_model_to_spyre(self, dtype=dtype)

    nn.Module.to = _spyre_module_to
    _module_to_patched = True
    logger.info(
        "Patched nn.Module.to() for automatic Spyre weight layout "
        "optimization"
    )


# --- safetensors monkeypatch -----------------------------------------

_safetensors_patched = False


def _is_spyre_device(device) -> bool:
    if isinstance(device, str):
        return device.startswith(DEVICE_NAME)
    if isinstance(device, torch.device):
        return device.type == DEVICE_NAME
    return False


def _transfer_state_dict_model_aware(
    state_dict,
    model: nn.Module,
):
    """Transfer state dict to Spyre using model structure for accuracy.

    Walks the model to identify which keys correspond to ``nn.Linear``
    weights, then uses the dim_order layout for those and the default
    layout for everything else.
    """
    linear_weight_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key = f"{name}.weight" if name else "weight"
            linear_weight_keys.add(key)

    result = {}
    for key, tensor in state_dict.items():
        if key in linear_weight_keys and tensor.ndim == 2:
            result[key] = _dma_to_spyre_dim_order_swapped(tensor)
        else:
            result[key] = _dma_to_spyre_default(tensor)
    return result


def _transfer_tensors_heuristic(tensors):
    """Transfer state dict to Spyre using a name-based heuristic.

    Used when no model object is available (``safetensors.torch.load_file``).
    2D tensors with "weight" in the key are assumed to be Linear weights
    unless the name suggests an embedding or normalization layer.
    """
    result = {}
    for key, tensor in tensors.items():
        is_likely_linear_weight = (
            tensor.ndim == 2
            and "weight" in key
            and "embed" not in key
            and "norm" not in key
            and "layernorm" not in key
            and "ln_" not in key
        )
        if is_likely_linear_weight:
            result[key] = _dma_to_spyre_dim_order_swapped(tensor)
        else:
            result[key] = _dma_to_spyre_default(tensor)
    return result


def patch_safetensors_for_spyre() -> None:
    """Monkeypatch ``safetensors.torch`` for optimal Spyre layout.

    After patching, ``safetensors.torch.load_file(file, device="spyre")``
    and ``safetensors.torch.load_model(model, file, device="spyre")``
    will transfer weights with the optimal layout.

    Non-Spyre destinations fall through to the original safetensors
    functions.
    """
    global _safetensors_patched
    if _safetensors_patched:
        return

    try:
        import safetensors.torch as st_torch
    except ImportError:
        logger.warning(
            "safetensors not installed; patch_safetensors_for_spyre "
            "has no effect"
        )
        return

    orig_load_file = st_torch.load_file
    orig_load_model = st_torch.load_model

    def _spyre_load_file(filename, device="cpu"):
        if not _is_spyre_device(device):
            return orig_load_file(filename, device=device)
        cpu_tensors = orig_load_file(filename, device="cpu")
        return _transfer_tensors_heuristic(cpu_tensors)

    def _spyre_load_model(model, filename, strict=True, device="cpu"):
        if not _is_spyre_device(device):
            return orig_load_model(
                model, filename, strict=strict, device=device
            )

        cpu_tensors = orig_load_file(filename, device="cpu")
        spyre_tensors = _transfer_state_dict_model_aware(cpu_tensors, model)

        # Direct parameter assignment. Shapes match the model exactly
        # (dim_order is a device-layout concern only), so we could in
        # principle use load_state_dict -- but assigning directly avoids
        # an extra device-side copy and lets us preserve requires_grad.
        missing = []
        unexpected = list(spyre_tensors.keys())
        for name, param in model.named_parameters():
            if name in spyre_tensors:
                unexpected.remove(name)
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = model.get_submodule(parts[0])
                    attr_name = parts[1]
                else:
                    parent = model
                    attr_name = parts[0]
                parent._parameters[attr_name] = nn.Parameter(
                    spyre_tensors[name], requires_grad=param.requires_grad
                )
            else:
                missing.append(name)
        for name, buf in model.named_buffers():
            if name in spyre_tensors:
                if name in unexpected:
                    unexpected.remove(name)
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = model.get_submodule(parts[0])
                    attr_name = parts[1]
                else:
                    parent = model
                    attr_name = parts[0]
                parent._buffers[attr_name] = spyre_tensors[name]
            else:
                missing.append(name)
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

    st_torch.load_file = _spyre_load_file
    st_torch.load_model = _spyre_load_model
    _safetensors_patched = True
    logger.info(
        "Patched safetensors.torch for optimal Spyre weight layout "
    )

# Made with Bob
