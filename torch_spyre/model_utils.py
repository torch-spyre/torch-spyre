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
"""

from torch_spyre._inductor.logging_utils import get_inductor_logger


import torch
import torch.nn as nn

from torch_spyre._C import (
    DataFormats,
    SpyreTensorLayout,
    copy_tensor,
    get_device_dtype,
    spyre_empty_with_layout,
)
from torch_spyre.constants import DEVICE_NAME

logger = get_inductor_logger("model_utils")


def _ensure_spyre_runtime() -> None:
    """Ensure Spyre runtime is up before calling DMA helpers from _C."""
    spyre = getattr(torch, DEVICE_NAME)
    if spyre.is_initialized():
        return
    torch.empty(0, dtype=torch.float16, device=DEVICE_NAME)


def _validate_target_dtype(dtype: torch.dtype) -> None:
    """Raise early if ``dtype`` has no Spyre device representation."""
    if get_device_dtype(dtype) == DataFormats.INVALID:
        raise ValueError(
            f"dtype {dtype} has no Spyre device representation. "
            f"See torch_spyre._C.DataFormats for the list of supported "
            f"formats, or torch_spyre._inductor.dtype_ops.DtypeOpTable "
            f"for the conversion pairs."
        )


# --- DMA helpers -----------------------------------------------------


def _dma_to_spyre_default(
    cpu_tensor: torch.Tensor,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Transfer a CPU tensor to Spyre with the default layout.

    Used for non-Linear-weight tensors (biases, embeddings, layer norm
    parameters, buffers). Stickifies along the last dimension.
    """
    if not cpu_tensor.is_contiguous():
        cpu_tensor = cpu_tensor.contiguous()
    dev_dtype = target_dtype if target_dtype is not None else cpu_tensor.dtype
    layout = SpyreTensorLayout(list(cpu_tensor.shape), dev_dtype)
    dst = spyre_empty_with_layout(
        cpu_tensor.size(), cpu_tensor.stride(), dev_dtype, layout
    )
    copy_tensor(cpu_tensor, dst, non_blocking=False)
    return dst


def _dma_to_spyre_dim_order_swapped(
    weight: torch.Tensor,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Transfer a 2D Linear weight to Spyre with dim_order=[1, 0].

    The host tensor shape ``(out_features, in_features)`` is preserved
    on the device, but the data is stickified along ``out_features``
    (dim 0) rather than the default ``in_features`` (dim 1). This
    matches the layout Spyre needs for efficient matmul and avoids
    both a CPU transpose and a device-side restickify.

    Caller must ensure ``weight.ndim == 2``.
    """
    assert weight.ndim == 2, "dim_order=[1,0] path is for 2D weights only"

    if not weight.is_contiguous():
        weight = weight.contiguous()
    dev_dtype = target_dtype if target_dtype is not None else weight.dtype
    layout = SpyreTensorLayout(
        list(weight.shape),  # host_size: (out, in)
        list(weight.stride()),  # host_strides: row-major
        dev_dtype,
        [1, 0],  # dim_order: stick on dim-0 = out_features
    )
    dst = spyre_empty_with_layout(weight.size(), weight.stride(), dev_dtype, layout)
    copy_tensor(weight, dst, non_blocking=False)
    return dst


# --- Model loading ---------------------------------------------------


def load_model_to_spyre(
    model: nn.Module,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Transfer model to Spyre with optimal weight layout.

    For each ``nn.Linear``, the weight is transferred using
    ``dim_order=[1, 0]`` so that ``out_features`` is stickified
    (optimal for Spyre matmul). Tensor shapes are preserved, so the
    model works unmodified with the existing inference path.

    All other parameters and buffers use the default Spyre layout.

    Idempotent: parameters already on Spyre are skipped.
    """
    if dtype is not None:
        _validate_target_dtype(dtype)
    # Ensure Spyre runtime is initialized before using _C functions
    _ensure_spyre_runtime()

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
                    name,
                    param_name,
                    list(p.shape),
                )
                dev = _dma_to_spyre_dim_order_swapped(p, target_dtype=dtype)
                linear_count += 1
            else:
                logger.debug(
                    "  %s.%s: shape=%s -> Spyre default layout",
                    name,
                    param_name,
                    list(p.shape),
                )
                dev = _dma_to_spyre_default(p, target_dtype=dtype)
                other_param_count += 1

            module._parameters[param_name] = nn.Parameter(
                dev, requires_grad=param.requires_grad
            )

        for buf_name, buf in list(module._buffers.items()):
            if buf is None or buf.device.type == DEVICE_NAME:
                continue
            module._buffers[buf_name] = _dma_to_spyre_default(buf, target_dtype=dtype)
            buffer_count += 1

    logger.info(
        "load_model_to_spyre: %d Linear weights optimized "
        "(dim_order=[1,0]), %d other params and %d buffers "
        "transferred with default layout",
        linear_count,
        other_param_count,
        buffer_count,
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
            _is_spyre(a) for a in args if isinstance(a, (str, torch.device))
        ) or _is_spyre(kwargs.get("device"))

        if not target_is_spyre:
            return orig_module_to(self, *args, **kwargs)

        dtype = kwargs.get("dtype")
        if dtype is None:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        return load_model_to_spyre(self, dtype=dtype)

    _spyre_module_to._spyre_patched = True  # type: ignore[attr-defined]
    nn.Module.to = _spyre_module_to  # type: ignore[method-assign]
    logger.info("Patched nn.Module.to() for automatic Spyre weight layout optimization")
