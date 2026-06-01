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

  * ``nn.Linear.forward`` works unmodified 
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
    # safetensors_patch.py handles the patching; model_utils plugs in
    # the optimal transfer via the hook in patch_module_to_for_spyre().

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
    # TODO: remove explicit fp16 conversion once D2H/H2D dtype
    # conversion support lands — copy_tensor will handle dtype during DMA.
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
    # TODO: remove explicit fp16 conversion once D2H/H2D dtype
    # conversion support lands — copy_tensor will handle dtype during DMA.
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


def load_model_to_spyre(model: nn.Module) -> nn.Module:
    """Transfer model to Spyre with optimal weight layout.

    For each ``nn.Linear``, the weight is transferred using
    ``dim_order=[1, 0]`` so that ``out_features`` is stickified
    (optimal for Spyre matmul). Tensor shapes are preserved, so the
    model works unmodified with the existing inference path.

    All other parameters and buffers use the default Spyre layout.
    The ``_dma_*`` helpers handle dtype conversion to fp16 internally.

    Idempotent: parameters already on Spyre are skipped.
    """
    # Ensure Spyre runtime is initialized before using _C functions
    torch.empty(0, dtype=torch.float16, device=DEVICE_NAME)
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


def patch_module_to_for_spyre() -> None:
    """Monkeypatch ``nn.Module.to`` for automatic optimal Spyre loading.

    After patching, ``model.to("spyre")`` will use the optimal weight
    layout for every ``nn.Linear`` in the model. Non-Spyre destinations
    fall through to the original ``nn.Module.to``.
    # Robust idempotency: check the live attribute on the patched callable
    # rather than a module-level flag.
    """
    if getattr(nn.Module.to, "_spyre_patched", False):
        return
    orig_module_to = nn.Module.to

    def _spyre_module_to(self, *args, **kwargs):
        def _is_spyre(d):
            return d is not None and torch.device(d).type == DEVICE_NAME

        target_is_spyre = any(
            _is_spyre(a) for a in args
            if isinstance(a, (str, torch.device))
        ) or _is_spyre(kwargs.get("device"))

        if not target_is_spyre:
            return orig_module_to(self, *args, **kwargs)


        return load_model_to_spyre(self)

    _spyre_module_to._spyre_patched = True
    nn.Module.to = _spyre_module_to
    logger.info(
        "Patched nn.Module.to() for automatic Spyre weight layout "
        "optimization"
    )

    # Plug optimized transfer into safetensors_patch (issue #400)
    # so safe_open(device="spyre") also gets dim_order=[1,0] layout.
    try:
        from torch_spyre.safetensors_patch import set_tensor_to_spyre_fn

        set_tensor_to_spyre_fn(_transfer_tensor_for_spyre)
        logger.info(
            "Plugged optimal dim_order layout into safetensors_patch"
        )
    except ImportError:
        pass  # safetensors_patch not available; safe_open not patched


# --- safetensors monkeypatch -----------------------------------------


def _transfer_tensor_for_spyre(
    tensor: torch.Tensor,
    key: str = "",
) -> torch.Tensor:
    """Transfer a single tensor to Spyre, choosing the right layout.

    Uses the name-based heuristic: 2D tensors with "weight" in the key
    (excluding embeddings, norms) get dim_order=[1,0]. Everything else
    gets the default layout.

    This function is plugged into safetensors_patch.py so that
    safe_open(device="spyre") also gets the optimal layout.
    """
    is_likely_linear_weight = (
        tensor.ndim == 2
        and "weight" in key
        and "embed" not in key
        and "norm" not in key
        and "layernorm" not in key
        and "ln_" not in key
    )
    if is_likely_linear_weight:
        return _dma_to_spyre_dim_order_swapped(tensor)
    return _dma_to_spyre_default(tensor)

# Made with Bob
