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

``nn.Embedding`` tables are instead read as a gather (indexed by token id
along the vocab/leading dim), so they get a gather-optimal "indirect
access" layout -- vocab dim outermost, hidden dim split into sticks --
rather than the matmul or default layout.

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

import warnings

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


def _dma_to_spyre_indirect_access(
    weight: torch.Tensor,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Transfer a 2D ``nn.Embedding`` table to Spyre with a gather-optimal layout.

    An embedding table is read as a gather (indexed by token id along the
    vocab/leading dim), not a matmul, so it wants a different device layout
    than the row-major matmul weights: the vocab dim outermost and the hidden
    dim split into stick-sized blocks, i.e. device dims
    ``[rows, D // eps, eps]`` where ``eps`` is the elements-per-stick for the
    device dtype. This is the "indirect access" layout the gather source
    needs (indexed dim outermost); see the tensors-and-layouts docs.

    Uses the 3-arg device-dims ``SpyreTensorLayout`` overload with the *device*
    dtype (``get_device_dtype``), not the host ``torch.dtype``.

    Requires ``D % eps == 0``; otherwise the sticks can't tile the hidden dim,
    so we warn and return ``None`` to signal the caller to fall back to the
    default layout, which still loads and runs, just without the gather
    optimization.

    Caller must ensure ``weight.ndim == 2``.
    """
    assert weight.ndim == 2, "indirect-access path is for 2D embedding tables only"

    if not weight.is_contiguous():
        weight = weight.contiguous()
    dev_dtype = target_dtype if target_dtype is not None else weight.dtype

    rows, d = weight.shape
    # elems_per_stick is dtype-aware (64 at fp16/bf16, 32 at fp32), so query it
    # rather than hardcoding a stick size.
    eps = SpyreTensorLayout(list(weight.shape), dev_dtype).elems_per_stick()
    if d % eps != 0:
        warnings.warn(
            f"Embedding hidden dim {d} is not a multiple of the Spyre stick "
            f"size {eps} for dtype {dev_dtype}; falling back to the default "
            "layout (no gather optimization) for this embedding table.",
            stacklevel=2,
        )
        return None

    layout = SpyreTensorLayout(
        [rows, d // eps, eps],  # device_size: vocab dim outermost
        [d, eps, 1],  # stride_map
        get_device_dtype(dev_dtype),
    )
    dst = spyre_empty_with_layout(weight.size(), weight.stride(), dev_dtype, layout)
    copy_tensor(weight, dst, non_blocking=False)
    return dst


# --- Model loading ---------------------------------------------------


def _module_overrides_apply(module: nn.Module) -> bool:
    """True if ``module`` customizes ``_apply`` and should govern its own subtree."""
    apply = module._apply
    return getattr(apply, "__func__", apply) is not nn.Module._apply


def _transfer_module(
    module: nn.Module,
    dtype: torch.dtype | None,
    counts: dict[str, int],
    prefix: str = "",
) -> None:
    """Recursively move ``module``'s params/buffers to Spyre, honoring ``_apply``.

    Mirrors ``nn.Module._apply``'s virtual recursion: a submodule that overrides
    ``_apply`` is delegated to and pruned from the walk. Normal modules get the
    optimal ``dim_order=[1, 0]`` layout for 2D ``nn.Linear`` weights, the
    gather-optimal indirect-access layout for 2D ``nn.Embedding`` tables, and
    the default layout for everything else. Tensors already on Spyre are skipped
    (idempotent). ``counts`` accumulates transferred-tensor tallies for logging;
    ``prefix`` is the module's dotted path (as in ``named_modules``) for logs.
    """
    if _module_overrides_apply(module):
        module._apply(
            lambda t: _dma_to_spyre_default(t, target_dtype=dtype)
            if t is not None and t.device.type != DEVICE_NAME
            else t
        )
        return

    for child_name, child in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        _transfer_module(child, dtype, counts, child_prefix)

    is_linear = isinstance(module, nn.Linear)
    is_embedding = isinstance(module, nn.Embedding)
    for name, param in list(module._parameters.items()):
        if param is None or param.device.type == DEVICE_NAME:
            continue
        p = param.data
        # 2D Linear weight -> optimal stickified matmul layout; 2D Embedding
        # table -> gather-optimal indirect-access layout; everything else
        # (bias, norms, ...) -> default layout.
        dev = None
        if is_linear and name == "weight" and p.ndim == 2:
            logger.debug(
                "  %s.%s: shape=%s -> Spyre dim_order=[1, 0]",
                prefix,
                name,
                list(p.shape),
            )
            dev = _dma_to_spyre_dim_order_swapped(p, target_dtype=dtype)
            counts["linear"] += 1
        elif is_embedding and name == "weight" and p.ndim == 2:
            dev = _dma_to_spyre_indirect_access(p, target_dtype=dtype)
            # dev is None if the hidden dim doesn't tile into sticks; the helper
            # has already warned, so fall through to the default layout below.
            if dev is not None:
                logger.debug(
                    "  %s.%s: shape=%s -> Spyre indirect-access (gather) layout",
                    prefix,
                    name,
                    list(p.shape),
                )
                counts["embedding"] += 1
        if dev is None:
            logger.debug(
                "  %s.%s: shape=%s -> Spyre default layout",
                prefix,
                name,
                list(p.shape),
            )
            dev = _dma_to_spyre_default(p, target_dtype=dtype)
            counts["other"] += 1
        module._parameters[name] = nn.Parameter(dev, requires_grad=param.requires_grad)

    for name, buf in list(module._buffers.items()):
        if buf is None or buf.device.type == DEVICE_NAME:
            continue
        module._buffers[name] = _dma_to_spyre_default(buf, target_dtype=dtype)
        counts["buffer"] += 1


def load_model_to_spyre(
    model: nn.Module,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Transfer model to Spyre with optimal weight layout.

    For each ``nn.Linear``, the weight is transferred using
    ``dim_order=[1, 0]`` so that ``out_features`` is stickified
    (optimal for Spyre matmul). Tensor shapes are preserved, so the
    model works unmodified with the existing inference path.

    For each ``nn.Embedding``, the table is transferred with a
    gather-optimal indirect-access layout (vocab dim outermost, hidden
    dim split into sticks) so the token-id gather runs efficiently. If
    the hidden dim doesn't tile into sticks, it falls back to the
    default layout with a warning.

    All other parameters and buffers use the default Spyre layout.

    Submodules that override ``_apply`` are honored, matching ``nn.Module.to`` semantics.
    Idempotent: parameters already on Spyre are skipped.
    """
    if dtype is not None:
        _validate_target_dtype(dtype)
    # Ensure Spyre runtime is initialized before using _C functions
    _ensure_spyre_runtime()

    counts = {"linear": 0, "embedding": 0, "other": 0, "buffer": 0}
    _transfer_module(model, dtype, counts)
    logger.info(
        "load_model_to_spyre: %d Linear weights optimized (dim_order=[1,0]), "
        "%d Embedding tables optimized (indirect-access layout), %d other "
        "params and %d buffers transferred with default layout",
        counts["linear"],
        counts["embedding"],
        counts["other"],
        counts["buffer"],
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
