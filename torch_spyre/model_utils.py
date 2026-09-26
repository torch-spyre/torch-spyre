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

Transfers ``nn.Linear`` weights to Spyre with an optimal device layout:

* **FP16/BF16 weights** use ``dim_order=[1, 0]`` in ``SpyreTensorLayout``,
  which tells the DMA engine to stickify along host dim-0 (out_features)
  instead of the default last dim (in_features).  Because ``aten.linear``
  decomposes to ``matmul(input, weight.T)``, stickifying out_features at
  DMA time means layout propagation sees a zero-cost match after the
  transpose — no ``ReStickify`` required.

* **Pre-quantized FP8 weights** (``torch.float8_e4m3fn``) use
  ``dim_order=[0, 1]`` (identity order) with ``ElementArrangement.QFP8WT``
  and a 2D stick ``[2, 64]`` (KERNEL layout).  The ``_scaled_mm`` path does
  not transpose the weight, so the compiler's ``_qfp8wt_stl`` propagation
  function generates the same identity ``dim_order=[0, 1]``.  Using ``[0, 1]``
  here ensures the DMA-produced ``stride_map`` matches what the compiler
  expects — no ``ReStickify`` required.

``nn.Embedding`` tables are instead read as a gather (indexed by token id
along the vocab/leading dim), so they get a gather-optimal "indirect
access" layout -- vocab dim outermost, hidden dim split into sticks --
rather than the matmul or default layout.

Pre-quantized FP8 ``nn.Linear`` weights (``torch.float8_e4m3fn``) can be
loaded directly into a KERNEL layout with 2D stick ``[2, 64]`` and
``ElementArrangement.QFP8WT``, bypassing any runtime ``qfp8wt`` quantization
step and feeding ``_scaled_mm`` directly.

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

    # Explicit (FP16 model):
    from torch_spyre.model_utils import load_model_to_spyre
    load_model_to_spyre(model)

    # Pre-quantized FP8 model:
    from torch_spyre.model_utils import load_fp8_model_to_spyre
    load_fp8_model_to_spyre(model)

    # Transparent for any code that uses .to("spyre"):
    from torch_spyre.model_utils import patch_module_to_for_spyre
    patch_module_to_for_spyre()
    model.to("spyre")
"""

import warnings

import torch
from torch import nn

from torch_spyre._C import (
    DataFormats,
    ElementArrangement,
    SpyreTensorLayout,
    copy_tensor,
    get_device_dtype,
    spyre_empty_with_layout,
)
from torch_spyre._inductor.logging_utils import get_inductor_logger
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


def _normalize_spyre_device(
    device: torch.device | str | int | None,
) -> torch.device | None:
    """Initialize Spyre and normalize a DMA helper's destination device."""
    _ensure_spyre_runtime()
    if device is None:
        return None
    target = (
        torch.device(DEVICE_NAME, device)
        if isinstance(device, int)
        else torch.device(device)
    )
    if target.type != DEVICE_NAME:
        raise ValueError(f"Expected a Spyre destination, got {target}")
    return target


# --- DMA helpers -----------------------------------------------------


def _dma_to_spyre_default(
    cpu_tensor: torch.Tensor,
    target_dtype: torch.dtype | None = None,
    *,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Transfer a CPU tensor to Spyre with the default layout.

    Used for non-Linear-weight tensors (biases, embeddings, layer norm
    parameters, buffers). Stickifies along the last dimension.
    """
    device = _normalize_spyre_device(device)
    if not cpu_tensor.is_contiguous():
        cpu_tensor = cpu_tensor.contiguous()
    dev_dtype = target_dtype if target_dtype is not None else cpu_tensor.dtype
    layout = SpyreTensorLayout(list(cpu_tensor.shape), dev_dtype)
    dst = spyre_empty_with_layout(
        cpu_tensor.size(), cpu_tensor.stride(), dev_dtype, layout, device=device
    )
    copy_tensor(cpu_tensor, dst, non_blocking=False)
    return dst


def _dma_to_spyre_dim_order_swapped(
    weight: torch.Tensor,
    target_dtype: torch.dtype | None = None,
    *,
    device: torch.device | str | int | None = None,
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
    device = _normalize_spyre_device(device)

    if not weight.is_contiguous():
        weight = weight.contiguous()
    dev_dtype = target_dtype if target_dtype is not None else weight.dtype
    layout = SpyreTensorLayout(
        list(weight.shape),  # host_size: (out, in)
        list(weight.stride()),  # host_strides: row-major
        dev_dtype,
        [1, 0],  # dim_order: stick on dim-0 = out_features
    )
    dst = spyre_empty_with_layout(
        weight.size(), weight.stride(), dev_dtype, layout, device=device
    )
    copy_tensor(weight, dst, non_blocking=False)
    return dst


def _dma_to_spyre_indirect_access(
    weight: torch.Tensor,
    target_dtype: torch.dtype | None = None,
    *,
    device: torch.device | str | int | None = None,
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
    device = _normalize_spyre_device(device)

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
    dst = spyre_empty_with_layout(
        weight.size(), weight.stride(), dev_dtype, layout, device=device
    )
    copy_tensor(weight, dst, non_blocking=False)
    return dst


def _dma_to_spyre_fp8_kernel(
    weight: torch.Tensor,
) -> torch.Tensor:
    """Transfer a pre-quantized FP8 weight to Spyre with KERNEL layout.

    Creates a KERNEL tensor with 2D stick layout ``[2, 64]`` and
    ``ElementArrangement.QFP8WT`` for use with ``_scaled_mm``.

    This is for pre-quantized ``torch.float8_e4m3fn`` weights loaded from
    model checkpoints (e.g. ``granite-3.3-8b-instruct-fp8``).  It creates
    the optimal device layout for FP8 matrix multiplication without requiring
    any runtime ``qfp8wt`` quantization overhead.

    Caller must ensure ``weight.ndim == 2`` and
    ``weight.dtype == torch.float8_e4m3fn``.
    """
    assert weight.ndim == 2, "FP8 KERNEL layout is for 2D weights only"
    assert weight.dtype == torch.float8_e4m3fn, (
        f"Weight must be torch.float8_e4m3fn, got {weight.dtype}"
    )
    assert weight.shape[0] % 2 == 0, (
        f"FP8 KERNEL K={weight.shape[0]} must be divisible by 2 (si=2)"
    )
    assert weight.shape[1] % 64 == 0, (
        f"FP8 KERNEL N={weight.shape[1]} must be divisible by 64 (so=64)"
    )

    _ensure_spyre_runtime()

    # Note: contiguity is NOT required here. The QFP8WT branch in generate_dci
    # (spyre_mem.cpp) derives host strides analytically from K and N — it does
    # not use the CPU tensor's actual strides. This means a non-contiguous view
    # (e.g. p.t()) is safe to pass directly, avoiding a CPU memcpy.
    layout = SpyreTensorLayout(
        list(weight.shape),  # host_size: [K, N] = [in_features, out_features]
        list(weight.stride()),  # host_strides: row-major [N, 1]
        torch.float8_e4m3fn,
        [0, 1],  # dim_order: identity, matches _qfp8wt_stl in propagate_layouts
        ElementArrangement.QFP8WT,  # 2D stick [2, 64] for KERNEL tensor
    )
    dst = spyre_empty_with_layout(
        weight.size(), weight.stride(), torch.float8_e4m3fn, layout
    )
    copy_tensor(weight, dst, non_blocking=False)
    return dst


def dma_moe_expert_weight_to_spyre(
    weight: torch.Tensor,
    target_dtype: torch.dtype | None = None,
    *,
    device: torch.device | str | int | None = None,
) -> torch.Tensor | None:
    """Transfer ``[E, C, F]`` weights in a gather- and matmul-friendly layout.

    The device layout is ``[E, C, F // eps, eps]``. Returns ``None`` when
    ``F`` does not span complete sticks.
    """
    assert weight.ndim == 3, "MoE expert-weight path is for rank-3 [E,C,F] only"
    device = _normalize_spyre_device(device)

    if not weight.is_contiguous():
        weight = weight.contiguous()
    dev_dtype = target_dtype if target_dtype is not None else weight.dtype

    experts, contract, free = weight.shape
    eps = SpyreTensorLayout(list(weight.shape), dev_dtype).elems_per_stick()
    if free % eps != 0:
        warnings.warn(
            f"MoE expert-weight free dim {free} is not a multiple of the Spyre "
            f"stick size {eps} for dtype {dev_dtype}; falling back to the "
            "default layout (no shared-layout optimization) for this weight.",
            stacklevel=2,
        )
        return None

    layout = SpyreTensorLayout(
        [experts, contract, free // eps, eps],
        [contract * free, free, eps, 1],
        get_device_dtype(dev_dtype),
    )
    dst = spyre_empty_with_layout(
        weight.size(), weight.stride(), dev_dtype, layout, device=device
    )
    copy_tensor(weight, dst, non_blocking=False)
    return dst


def dma_moe_per_expert_scale_to_spyre(
    scale: torch.Tensor,
    target_dtype: torch.dtype | None = None,
    *,
    device: torch.device | str | int | None = None,
) -> torch.Tensor | None:
    """Transfer ``[E]`` scales as a gather-ready ``[E, eps]`` tensor.

    Each scale fills one stick. Widening on the host avoids an unsupported
    in-graph rank expansion.
    """
    assert scale.ndim == 1, "per-expert-scale path is for 1D [E] tensors only"
    device = _normalize_spyre_device(device)

    if not scale.is_contiguous():
        scale = scale.contiguous()
    dev_dtype = target_dtype if target_dtype is not None else scale.dtype

    experts = scale.shape[0]
    eps = SpyreTensorLayout([experts, 1], dev_dtype).elems_per_stick()

    widened = scale[:, None].expand(-1, eps).contiguous()

    layout = SpyreTensorLayout(
        [experts, 1, eps],
        [eps, eps, 1],
        get_device_dtype(dev_dtype),
    )
    dst = spyre_empty_with_layout(
        widened.size(), widened.stride(), dev_dtype, layout, device=device
    )
    copy_tensor(widened, dst, non_blocking=False)
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
    use_fp8_weights: bool = False,
) -> None:
    """Recursively move ``module``'s params/buffers to Spyre, honoring ``_apply``.

    Mirrors ``nn.Module._apply``'s virtual recursion: a submodule that overrides
    ``_apply`` is delegated to and pruned from the walk. Normal modules get the
    optimal ``dim_order=[1, 0]`` layout for 2D ``nn.Linear`` weights, the
    gather-optimal indirect-access layout for 2D ``nn.Embedding`` tables, and
    the default layout for everything else. When ``use_fp8_weights=True``,
    ``torch.float8_e4m3fn`` Linear weights are loaded with KERNEL layout and
    ``ElementArrangement.QFP8WT`` for direct use with ``_scaled_mm``.
    Tensors already on Spyre are skipped (idempotent). ``counts`` accumulates
    transferred-tensor tallies for logging; ``prefix`` is the module's dotted
    path (as in ``named_modules``) for logs.
    """
    if _module_overrides_apply(module):
        module._apply(
            lambda t: (
                _dma_to_spyre_default(t, target_dtype=dtype)
                if t is not None and t.device.type != DEVICE_NAME
                else t
            )
        )
        return

    for child_name, child in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        _transfer_module(child, dtype, counts, child_prefix, use_fp8_weights)

    is_linear = isinstance(module, nn.Linear)
    is_embedding = isinstance(module, nn.Embedding)
    for name, param in list(module._parameters.items()):
        if param is None or param.device.type == DEVICE_NAME:
            continue
        p = param.data
        # Priority order:
        #   1. FP8 pre-quantized Linear weight  -> KERNEL 2D-stick layout (QFP8WT)
        #   2. FP16/BF16 2D Linear weight       -> dim_order=[1, 0] matmul layout
        #   3. 2D Embedding table               -> gather indirect-access layout
        #   4. Everything else                  -> default layout
        dev = None
        if (
            use_fp8_weights
            and is_linear
            and name == "weight"
            and p.ndim == 2
            and p.dtype == torch.float8_e4m3fn
        ):
            logger.debug(
                "  %s.%s: shape=%s dtype=%s -> Spyre FP8 KERNEL layout (2D stick) [transposed to K,N]",
                prefix,
                name,
                list(p.shape),
                p.dtype,
            )
            # nn.Linear stores weight as [out_features, in_features] = [N, K].
            # The fp8_linear_kernel expects [K, N] on Spyre (see _kernel_weight_splits).
            # Pass p.t() directly — no .contiguous() needed since the QFP8WT DMA
            # path ignores the CPU tensor's actual strides (see _dma_to_spyre_fp8_kernel).
            dev = _dma_to_spyre_fp8_kernel(p.t())
            counts["fp8_kernel"] += 1
        elif is_linear and name == "weight" and p.ndim == 2:
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
    use_fp8_weights: bool = False,
) -> nn.Module:
    """Transfer model to Spyre with optimal weight layout.

    For each ``nn.Linear``, the weight is transferred using the optimal layout:

    - **FP8 pre-quantized weights** (``torch.float8_e4m3fn``, when
      ``use_fp8_weights=True``): KERNEL layout with 2D stick ``[2, 64]`` and
      ``ElementArrangement.QFP8WT`` — feeds ``_scaled_mm`` directly with no
      runtime quantization overhead.
    - **FP16/BF16 weights**: ``dim_order=[1, 0]`` so ``out_features`` is
      stickified (optimal for Spyre matmul).

    For each ``nn.Embedding``, the table is transferred with a
    gather-optimal indirect-access layout (vocab dim outermost, hidden
    dim split into sticks) so the token-id gather runs efficiently. If
    the hidden dim doesn't tile into sticks, it falls back to the
    default layout with a warning.

    All other parameters and buffers use the default Spyre layout.

    Submodules that override ``_apply`` are honored, matching ``nn.Module.to``
    semantics.  Idempotent: parameters already on Spyre are skipped.

    Args:
        model: Model to transfer to Spyre device.
        dtype: Target dtype for non-FP8 weight conversion (optional). If
               ``None``, preserves the original dtype. Ignored for FP8 weights
               when ``use_fp8_weights=True``.
        use_fp8_weights: If ``True``, ``torch.float8_e4m3fn`` Linear weights are
                         loaded with KERNEL layout and ``QFP8WT`` arrangement.
                         Use this for pre-quantized FP8 model checkpoints.
    """
    if dtype is not None:
        _validate_target_dtype(dtype)
    # Ensure Spyre runtime is initialized before using _C functions
    _ensure_spyre_runtime()

    counts = {"linear": 0, "fp8_kernel": 0, "embedding": 0, "other": 0, "buffer": 0}
    _transfer_module(model, dtype, counts, use_fp8_weights=use_fp8_weights)
    logger.info(
        "load_model_to_spyre: %d Linear weights (dim_order=[1,0]), "
        "%d FP8 KERNEL weights, "
        "%d Embedding tables (indirect-access layout), %d other "
        "params and %d buffers transferred with default layout",
        counts["linear"],
        counts["fp8_kernel"],
        counts["embedding"],
        counts["other"],
        counts["buffer"],
    )
    return model


def load_fp8_model_to_spyre(model: nn.Module) -> nn.Module:
    """Load a pre-quantized FP8 model to Spyre with KERNEL layouts.

    Convenience wrapper around :func:`load_model_to_spyre` for models whose
    ``nn.Linear`` weights are already quantized to ``torch.float8_e4m3fn``
    (e.g. ``granite-3.3-8b-instruct-fp8``, ``gemma-4-26B-A4B-it-FP8-dynamic``).

    Each FP8 weight is DMA'd directly into the KERNEL layout with 2D stick
    ``[2, 64]`` and ``ElementArrangement.QFP8WT``, so ``_scaled_mm`` can
    consume it without any runtime ``qfp8wt`` quantization step.  This
    significantly reduces model startup time (8-10× faster than quantizing
    at runtime).

    Non-FP8 parameters (biases, layer norms, embeddings) are transferred
    with their normal optimal layouts.

    Example::

        from transformers import AutoModelForCausalLM
        from torch_spyre.model_utils import load_fp8_model_to_spyre

        model = AutoModelForCausalLM.from_pretrained(
            "ibm-granite/granite-3.3-8b-instruct-fp8"
        )
        model = load_fp8_model_to_spyre(model)
        # Ready for inference — no runtime quantization needed.

    See also:
        :func:`load_model_to_spyre` — general model loading with ``use_fp8_weights`` flag.
    """
    return load_model_to_spyre(model, use_fp8_weights=True)


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
