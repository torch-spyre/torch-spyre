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


from typing import TYPE_CHECKING, Dict, List, Optional, Union

from torch._dynamo.guards import GuardBuilder

from torch_spyre.constants import DEVICE_NAME

if TYPE_CHECKING:
    import torch

    from torch_spyre._C import SpyreTensorLayout


# Sentinel distinguishing "caller didn't pass target_dtype" (-> resolves to
# torch.float16, the Spyre DMA hardware requirement) from an explicit
# ``target_dtype=None`` (-> disable dtype coercion, keep the checkpoint's own
# dtype). Defined without importing torch at module scope, matching this
# file's lazy-import convention (torch is only guaranteed importable once
# this module's patch functions actually run).
_UNSET = object()


def _add_ea(src_tensor, res_tensor) -> None:
    """Update the EA tag after an eager transfer handled by ``orig_to``.

    Same-device dtype-changing casts return through ``to_dtype_d2d`` before
    this helper is reached; their EA is propagated by the compiled graph.
    """
    if res_tensor.dtype == src_tensor.dtype:
        return

    import torch
    from torch_spyre._inductor.dtype_ops import DtypeOpTable
    from torch_spyre._inductor.constants import STAGGERED_EAS
    from torch_spyre._inductor.pass_utils import rescale_stl_for_dtype

    # Skip FakeTensor tracing contexts during torch.compile
    if (
        torch.compiler.is_compiling()
        or isinstance(src_tensor, torch._subclasses.FakeTensor)
        or isinstance(res_tensor, torch._subclasses.FakeTensor)
    ):
        return

    from torch_spyre._C import (
        get_spyre_tensor_layout,
        set_spyre_tensor_layout,
    )

    # TODO EA torch.bool as it can be fp16 or fp32

    try:
        src_layout = get_spyre_tensor_layout(src_tensor)
    except RuntimeError:
        return

    if src_layout is None:
        return

    input_ea = src_layout.element_arrangement
    fmt = DtypeOpTable.ea_map(src_tensor.dtype, res_tensor.dtype, input_ea)

    try:
        res_layout = get_spyre_tensor_layout(res_tensor)
    except RuntimeError:
        return

    if res_layout is None:
        return

    stl = res_layout.with_element_arrangement(fmt)
    is_staggered_ea = fmt in STAGGERED_EAS or input_ea in STAGGERED_EAS
    if src_tensor.dtype != torch.float32 and is_staggered_ea:
        stl = rescale_stl_for_dtype(src_layout, res_tensor.dtype, fmt)

    set_spyre_tensor_layout(res_tensor, stl)


def _patch_tensor_for_spyre():
    import torch

    if getattr(torch.Tensor, "_spyre_tensor_patched", False):
        return

    from torch.utils._device import _device_constructors

    _device_constructors()  # warm the cache with the original torch.empty

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

    def device_tensor_layout(self: torch.Tensor) -> Optional["SpyreTensorLayout"]:
        if self.device is not None and self.device.type == DEVICE_NAME:
            if isinstance(self, torch._subclasses.FakeTensor):
                return None  # catch FakeTensor BEFORE calling device_tensor_layout()
            from torch_spyre._C import get_spyre_tensor_layout

            return get_spyre_tensor_layout(self)
        else:
            return None

    def spyre_to(self, *args, device_layout=None, **kwargs):
        if device_layout is None:
            # During Dynamo tracing this wrapper is an allow_in_graph leaf: keep
            # the operation device-local so Inductor sees and lowers the dtype
            # conversion. The host-staged path below is only for real eager
            # tensors; introducing CPU copies while tracing would put
            # DeviceCopy nodes into the compiled graph.
            if (
                torch.compiler.is_compiling()
                or isinstance(self, torch._subclasses.FakeTensor)
                or torch._is_functional_tensor(self)
            ):
                return orig_to(self, *args, **kwargs)

            # Support D2H and H2D dtype casting via DCI (DataConversionInfo) in
            # spyre_mem.cpp. Same-device casting is routed through the standalone
            # compiled to_dtype_d2d path, whose lowering converts on device.
            # Unsupported conversion pairs are still accepted here: the compiled
            # lowering checks DtypeOpTable and uses to_dtype_cpu, preserving the
            # previous host-roundtrip fallback and warning.
            _device = kwargs.get("device", None)
            _dtype = kwargs.get("dtype", None)
            if args:
                first = args[0]
                if isinstance(first, torch.Tensor):
                    # ``self.to(other)`` adopts both properties from ``other``.
                    _device = first.device
                    _dtype = first.dtype
                elif isinstance(first, torch.dtype):
                    # ``self.to(dtype)`` keeps the current device.
                    _dtype = first
                elif isinstance(first, (str, torch.device)):
                    _device = first
                    if len(args) > 1 and isinstance(args[1], torch.dtype):
                        _dtype = args[1]

            target_device = self.device if _device is None else torch.device(_device)

            if (
                self.device.type == DEVICE_NAME
                and target_device.type == DEVICE_NAME
                and _dtype is not None
                and _dtype != self.dtype
            ):
                # device_layout is necessarily None in this branch (guarded at
                # function entry), so this dtype-only op drops no layout request.
                return torch.ops.spyre.to_dtype_d2d(self, _dtype, self.storage_offset())

            res = orig_to(self, *args, **kwargs)
            if res.device.type == DEVICE_NAME:
                _add_ea(self, res)
            return res
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

            from torch_spyre._C import spyre_empty_with_layout

            dst = spyre_empty_with_layout(
                self.size(), self.stride(), dtype, device_layout
            )

            if self.device.type == "cpu":
                from torch_spyre._C import copy_tensor

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
                    # Pass storage_offsets explicitly: a graph input's
                    # storage_offset is dropped by Inductor, so the lowering
                    # must re-introduce it in-graph (see copy_from_d2d in
                    # customops.py and lower_spyre_from_d2d).
                    return torch.ops.spyre.copy_from_d2d(
                        self, dst, self.storage_offset(), dst.storage_offset()
                    )

    def spyre_empty(
        *args,
        size=None,
        device_layout=None,
        out=None,
        dtype=None,
        layout=torch.strided,
        device=None,
        requires_grad=False,
        pin_memory=False,
        memory_format=torch.contiguous_format,
    ):
        # torch.empty supports size as either a positional arg or keyword arg.
        # Normalise so downstream always receives it as positional.
        if size is not None:
            if args:
                raise TypeError(
                    "empty() received an invalid combination of arguments - got (tuple, size=tuple)"
                )
            args = (size,)

        if (
            device_layout is None
        ):  # use original implementation if no layout is provided
            kwargs = dict(
                out=out,
                dtype=dtype,
                layout=layout,
                requires_grad=requires_grad,
                pin_memory=pin_memory,
                memory_format=memory_format,
            )
            if device is not None:
                kwargs["device"] = device
            return orig_empty(*args, **kwargs)
        else:
            # layout_opt is omitted; c10::Layout has no pybind11 type caster,
            # so py_empty_with_layout drops that parameter and always uses
            # the default (Strided).
            from torch_spyre._C import empty_with_layout

            return empty_with_layout(
                *args, device_layout, dtype, device, pin_memory, memory_format
            )

    torch.Tensor.__repr__ = spyre_aware_repr
    torch.Tensor.device_tensor_layout = device_tensor_layout
    torch.Tensor._spyre_tensor_patched = True
    torch.Tensor.to = spyre_to
    # Dynamo cannot trace INTO the Python ``spyre_to``: it inlines the wrapper,
    # hits the C++ ``orig_to`` call, and graph-breaks — forcing the whole region
    # to run eager, where D2D dtype casts (e.g. fp16<->bf16) are wrong. Mark
    # ``.to`` allow_in_graph so Dynamo treats it as a leaf and traces its tensor
    # semantics (-> prims.convert_element_type) directly, keeping the region
    # compiled. (An ``is_compiling()`` guard inside spyre_to does NOT help — the
    # break fires on ``orig_to`` regardless of the branch taken.)
    #
    # Scope note: this is a process-global registration affecting every user of
    # ``torch.Tensor.to``, not just Spyre. That is acceptable here because
    # torch-spyre already monkey-patches ``torch.Tensor.to`` globally (line
    # above), so this backend already owns ``.to``'s behavior in-process;
    # marking it allow_in_graph only changes how Dynamo traces it (as a leaf),
    # which is harmless for cpu/other-backend tensors (spyre_to falls through to
    # ``orig_to`` semantics for them).
    torch._dynamo.allow_in_graph(torch.Tensor.to)
    torch.empty = spyre_empty

    # ── Optimal weight loading (issue #1339) ──────────────
    # Patch dim_order=[1,0] transfer + nn.Module.to override (issue #1339).
    try:
        from torch_spyre.model_utils import patch_module_to_for_spyre

        patch_module_to_for_spyre()
    except Exception as e:  # pragma: no cover - defensive
        import warnings

        warnings.warn(f"Failed to install optimal weight layout patches: {e}")

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
            guard.user_stack,
        )

    # ── invoke_subgraph reuse support ────────────────────────────────────
    # Because we replace GuardBuilder.TENSOR_MATCH, guards it builds report
    # their type (via Guard.create_fn_name(), i.e. create_fn.__name__) as
    # "_spyre_TENSOR_MATCH" rather than "TENSOR_MATCH". torch's
    # invoke_subgraph subgraph-reuse path (torch._dynamo.variables.
    # invoke_subgraph) looks each guard's type up in GUARD_VALUE_DISPATCH to
    # re-evaluate it mid-trace; an unknown type there is a hard error
    # ("subgraph_reuse: unsupported guard type ..."). So any use of
    # torch.compiler.nested_compile_region would abort once this patch is
    # installed.
    #
    # Register a spec under our name that mirrors stock TENSOR_MATCH's
    # metadata check AND additionally compares SpyreTensorLayout, matching
    # what the runtime lambda guard above actually enforces — so a subgraph
    # is only reused when both the standard tensor metadata and the device
    # layout still match. Guarded behind availability so older torch without
    # the reuse machinery is unaffected.
    try:
        from torch._dynamo.guards import (
            GUARD_VALUE_DISPATCH,
            GuardCheckSpec,
            extract_tensor_metadata,
        )
    except ImportError:
        # torch predates invoke_subgraph reuse — nothing to register.
        pass
    else:

        def _spyre_tensor_reuse_metadata(guard, value):
            # Standard tensor metadata (shape/stride/dtype/device/
            # requires_grad), plus the device layout for Spyre tensors
            # (None otherwise). Mirrors extract_tensor_metadata so the
            # comparison is identical to stock TENSOR_MATCH on the metadata
            # axis.
            layout = None
            if getattr(value, "device", None) is not None and (
                value.device.type == DEVICE_NAME
            ):
                layout = value.device_tensor_layout()
            return (extract_tensor_metadata(value), layout)

        def _spyre_tensor_reuse_eval(value, metadata):
            base_metadata, expected_layout = metadata
            if not isinstance(value, torch.Tensor):
                return False
            if extract_tensor_metadata(value) != base_metadata:
                return False
            # Layout only constrains Spyre tensors; mirror the runtime
            # lambda guard: non-Spyre value OR layout matches.
            if value.device.type != DEVICE_NAME:
                return expected_layout is None
            return value.device_tensor_layout() == expected_layout

        _spyre_reuse_spec = GuardCheckSpec(
            get_metadata_fn=_spyre_tensor_reuse_metadata,
            eval_fn=_spyre_tensor_reuse_eval,
        )
        # Attach for the auto-dispatch scan, and register directly under the
        # name Guard.create_fn_name() produces for guards this builder makes.
        # GUARD_VALUE_DISPATCH is built once (at torch import, before this
        # patch runs), so a direct insert is required — the scan does not
        # re-run.
        _spyre_TENSOR_MATCH.guard_check_spec = _spyre_reuse_spec
        GUARD_VALUE_DISPATCH["_spyre_TENSOR_MATCH"] = _spyre_reuse_spec

    GuardBuilder.TENSOR_MATCH = _spyre_TENSOR_MATCH
    # ───────────────────FxGraph Cache Key Extension ───────────────────
    # Extends FxGraphHashDetails to include SpyreTensorLayout in the cache key
    # preventing incorrect disk cache hits across process boundaries.
    # ──────────────────────────────────────────────────────────────────────────
    _patch_fx_graph_hash()
    # ─────────────── invoke_subgraph subgraph decompositions ───────────────
    # Threads the Spyre decomposition table into the re-trace of every
    # nested_compile_region / invoke_subgraph subgraph body, so ops that must
    # be decomposed on Spyre (notably SDPA → online-softmax) are decomposed
    # inside the HOP body — not just in the top-level graph.
    # ──────────────────────────────────────────────────────────────────────────
    _patch_invoke_subgraph_decompositions()


def _patch_invoke_subgraph_decompositions():
    """Thread the Spyre decomp table into invoke_subgraph subgraph re-traces.

    torch-spyre installs its decomposition table only on the patched top-level
    ``compile_fx``/``compile_fx_inner`` (see ``torch_spyre/_inductor``). But
    ``torch.compiler.nested_compile_region`` bodies (the ``invoke_subgraph``
    HOP) are RE-TRACED separately, via
    ``reenter_make_fx(subgraph, subgraph_decomp_table=_extract_nested_region_config(subgraph))``.
    ``_extract_nested_region_config`` reads
    ``gm.meta["nested_region_config"].decompositions`` which is ``None`` unless
    the user passed an explicit ``NestedCompileRegionOptions(decompositions=...)``.
    With ``None``, the subgraph body is re-traced with NO decomposition table —
    so e.g. ``aten.scaled_dot_product_attention`` survives in the subgraph and
    torch-spyre lowers it incorrectly (Blocker 6: correct when a single call is
    inlined by Inductor, wrong once ≥2 calls keep it as a shared HOP body).

    This patch wraps ``_extract_nested_region_config`` so that when it returns
    ``None`` (the region inherits its parent's decompositions) AND we are inside
    a Spyre ``compile_fx`` call, it returns ``get_spyre_decomp_table()`` instead.
    An explicit user-provided table is respected unchanged, and — because the
    gate is the ``in_spyre_compile()`` thread-local set by the patched
    ``compile_fx`` wrapper — a nested_compile_region compiled outside a Spyre
    compile (pure-CPU) is left alone.

    Why the thread-local (not device inspection): at HOP re-trace time the
    subgraph body is traced on fake tensors whose device is not ``spyre`` and
    whose weights are lifted as inputs, so scanning the subgraph GraphModule's
    tensor devices always reports "not Spyre" (B6DIAG3, device-proven). The
    reliable signal that this re-trace belongs to a Spyre compile is that a
    Spyre ``compile_fx`` is on the stack — which ``_wrapper`` records.

    Guarded behind availability so a torch without the invoke_subgraph reenter
    machinery is unaffected. Idempotent.
    """
    import sys

    mod = sys.modules.get("torch._higher_order_ops.invoke_subgraph")
    if mod is None:
        try:
            import torch._higher_order_ops.invoke_subgraph as mod  # noqa: F811
        except ImportError:
            # torch predates the invoke_subgraph reenter path — nothing to do.
            return

    original = getattr(mod, "_extract_nested_region_config", None)
    if original is None or getattr(original, "_spyre_decomp_patched", False):
        return

    def _spyre_extract_nested_region_config(fn):
        # Respect an explicit user-provided table; otherwise, if this HOP
        # re-trace is happening inside a Spyre compile, thread the Spyre decomp
        # table so ops that must be decomposed on Spyre (notably SDPA →
        # online-softmax) are decomposed inside the region body.
        table = original(fn)
        if table is not None:
            return table
        from torch_spyre._inductor import in_spyre_compile

        if not in_spyre_compile():
            return None
        from torch_spyre._inductor.decompositions import get_spyre_decomp_table

        return get_spyre_decomp_table()

    _spyre_extract_nested_region_config._spyre_decomp_patched = True
    mod._extract_nested_region_config = _spyre_extract_nested_region_config

    # ── Safetensors Spyre-aware loading (monkey-patch) ──────────────────────────
    # Monkey-patches the three public Python entry points to support device="spyre".
    # ─────────────────────────────────────────────────────────────────────────────
    try:
        _patch_safetensors_for_spyre()
    except Exception as e:  # pragma: no cover - safetensors may not be installed
        import warnings

        warnings.warn(f"Failed to install safetensors Spyre patches: {e}")


# ── Safetensors hook + monkey-patch ──────────────────────────────────────────
#
# Hook signature:
#
#   hook(cpu_tensor: torch.Tensor, name: str, device) -> torch.Tensor
#
# Where:
#   cpu_tensor — tensor freshly loaded from disk, on CPU (may be a
#                memoryview-backed zero-copy view from mmap).
#   name       — tensor key in the safetensors file, e.g.
#                "model.layers.0.self_attn.q_proj.weight". Used to select the
#                optimal Spyre DMA layout.
#   device     — original device argument (str or torch.device). Passed
#                through for potential device-index handling ("spyre:1").
#
# Tensor-type heuristics from key names
# ──────────────────────────────────────
# In a flat safetensors dict we have no nn.Module to call isinstance() on, so
# we recover the tensor kind from its key name — a convention stable across
# virtually all HuggingFace and standard PyTorch checkpoints:
#
#   Embedding table  – 2D ``.weight`` tensor with an embedding module name as
#                      a dot-delimited key segment.
#                      Shape: [vocab_size, hidden_dim] or [max_pos, hidden_dim].
#                      → _dma_to_spyre_indirect_access (gather-optimal layout).
#                        Falls back to default if hidden_dim % eps != 0.
#
#   Linear weight    – 2D tensor whose key ends with ".weight" and is NOT
#                      classified as an embedding. Covers q/k/v/o projections,
#                      MLP up/gate/down, free lm_head, etc.
#                      → _dma_to_spyre_dim_order_swapped (dim_order=[1,0],
#                        matmul-optimal).
#
#   lm_head          – load_file / safe_open have no model context, so an
#                      lm_head is treated as a Linear weight. This is always
#                      correct, but may be sub-optimal when the model later ties
#                      it to an embedding table.
#
#   Everything else  – bias, layer-norm params, 1-D buffers, etc.
#                      → _dma_to_spyre_default (stickify last dim).
# ─────────────────────────────────────────────────────────────────────────────

_EMBEDDING_MODULE_NAMES: tuple = (
    "embed",
    "embedding",
    "embeddings",
    "embed_tokens",
    "wte",  # GPT-2 word-token embeddings
    "wpe",  # GPT-2 word-position embeddings
    "tok_embed",
    "word_embed",
    "token_embed",
    "pos_embed",
    "patch_embed",
)


def _classify_safetensors_key(name: str, ndim: int) -> str:
    """Return ``'embedding'``, ``'linear'``, or ``'other'``.

    Pure name-convention + ndim heuristic; no model object required.
    """
    if ndim != 2:
        return "other"
    parts = name.lower().split(".")
    if parts[-1] != "weight":
        return "other"
    if any(part in _EMBEDDING_MODULE_NAMES for part in parts[:-1]):
        return "embedding"
    if len(parts) > 1:
        return "linear"
    return "other"


def _spyre_tensor_from_safetensors(
    cpu_tensor: "torch.Tensor",
    name: str,
    device,  # str | torch.device — honoured by forward-compat; index unused for now
    target_dtype=_UNSET,
) -> "torch.Tensor":
    """Spyre device-transfer hook for safetensors.

    Receives a CPU tensor loaded from a safetensors file and returns a tensor
    on the Spyre device with the optimal layout for the tensor's role.

    Layout selection (mirrors ``_transfer_module`` in model_utils.py):
      - 2D embedding-named tensor  → indirect-access layout (gather-optimal).
        Falls back to default if ``hidden_dim % elems_per_stick != 0``.
      - 2D Linear weight           → ``dim_order=[1, 0]`` (matmul-optimal).
      - Everything else            → default Spyre layout (stickify last dim).

    ``target_dtype`` selects the on-device float dtype for floating-point
    tensors, threaded through to the ``_dma_to_spyre_*`` helpers in
    model_utils.py (same ``target_dtype`` parameter they already expose).
    Left unspecified, it resolves to ``torch.float16`` — the Spyre DMA
    hardware requirement. Pass ``target_dtype=None`` explicitly to disable
    coercion and keep the checkpoint's own dtype (e.g. for CPU-stub tests
    with no hardware DMA constraint). Non-floating-point tensors (e.g. int
    buffers) are never coerced, regardless of ``target_dtype``.
    """
    from torch_spyre.model_utils import (
        _dma_to_spyre_default,
        _dma_to_spyre_dim_order_swapped,
        _dma_to_spyre_indirect_access,
    )
    from torch_spyre._inductor.logging_utils import get_inductor_logger

    logger = get_inductor_logger("model_utils")

    if target_dtype is _UNSET:
        import torch

        target_dtype = torch.float16

    # Only floating-point tensors are eligible for coercion — an int buffer
    # (e.g. position ids) must never be cast to a float dtype.
    dma_target_dtype = target_dtype if cpu_tensor.dtype.is_floating_point else None

    # Cast on CPU *before* the DMA helpers see this tensor, using the same
    # dma_target_dtype that's about to be threaded into them
    # copy_tensor's DMA path is measurably faster on a same-dtype
    # transfer than on a cross-dtype one (e.g. bf16 checkpoint -> fp16
    # device), so casting here with PyTorch's own vectorized `.to()` and
    # handing the DMA helpers an already-matching dtype avoids that cost.
    # Passing target_dtype=None disables this entirely, same as before.
    if dma_target_dtype is not None and cpu_tensor.dtype != dma_target_dtype:
        cpu_tensor = cpu_tensor.to(dtype=dma_target_dtype)

    # The mmap fast-path in safetensors gives us a memoryview-backed
    # frombuffer view. Make it contiguous before any Spyre DMA call.
    if not cpu_tensor.is_contiguous():
        cpu_tensor = cpu_tensor.contiguous()

    kind = _classify_safetensors_key(name, cpu_tensor.ndim)

    if kind == "embedding":
        result = _dma_to_spyre_indirect_access(
            cpu_tensor, target_dtype=dma_target_dtype
        )
        if result is not None:
            logger.debug(
                "safetensors: %s shape=%s -> Spyre indirect-access (gather) layout",
                name,
                list(cpu_tensor.shape),
            )
            return result
        # _dma_to_spyre_indirect_access warned already; fall through to default.
        logger.debug(
            "safetensors: %s shape=%s -> Spyre default layout (embedding fallback)",
            name,
            list(cpu_tensor.shape),
        )
        return _dma_to_spyre_default(cpu_tensor, target_dtype=dma_target_dtype)

    if kind == "linear":
        logger.debug(
            "safetensors: %s shape=%s -> Spyre dim_order=[1, 0]",
            name,
            list(cpu_tensor.shape),
        )
        return _dma_to_spyre_dim_order_swapped(
            cpu_tensor, target_dtype=dma_target_dtype
        )

    # "other": bias, norms, 1-D buffers, etc.
    logger.debug(
        "safetensors: %s shape=%s -> Spyre default layout",
        name,
        list(cpu_tensor.shape),
    )
    return _dma_to_spyre_default(cpu_tensor, target_dtype=dma_target_dtype)


class _SpyreSafeOpen:
    """Context-manager wrapper around ``safetensors.safe_open`` for Spyre.

    Opened with ``device="cpu"`` internally; each ``get_tensor`` / ``get_tensors``
    / slice call transfers the result to Spyre via ``_spyre_tensor_from_safetensors``.

    All non-tensor methods (``keys``, ``metadata``, ``get_slice``,
    ``offset_keys``) are forwarded verbatim so callers that iterate keys or
    read metadata work without changes.

    ``get_slice`` returns a ``_SpyreSafeSlice`` wrapper that applies the hook
    on ``__getitem__``, matching the slice-dispatch behaviour of the Rust core.

    ``target_dtype`` (default: unset, which resolves to ``torch.float16`` —
    the Spyre DMA hardware requirement) is stored and threaded through to
    every tensor access. Pass ``target_dtype=None`` to disable coercion.

    The stored ``_target_dtype`` attribute may be ``_UNSET`` (the sentinel
    meaning "resolve to fp16 at DMA time") rather than an actual
    ``torch.dtype``.  Inspect ``_target_dtype is _UNSET`` rather than
    comparing it to a dtype in any code that reads this attribute.
    """

    def __init__(
        self,
        filename,
        framework: str,
        backend: str = "mmap",
        target_dtype=_UNSET,
    ):
        import safetensors as _st_mod

        # Open on CPU so the Rust core handles mmap / pread normally.
        self._handle = _st_mod._orig_safe_open(
            filename, framework=framework, device="cpu", backend=backend
        )
        self._target_dtype = target_dtype

    def __enter__(self) -> "_SpyreSafeOpen":
        self._handle.__enter__()
        return self

    def __exit__(self, *args):
        return self._handle.__exit__(*args)

    # ── forwarded metadata ───────────────────────────────────────────────────

    def keys(self) -> List[str]:
        return self._handle.keys()

    def offset_keys(self) -> List[str]:
        """Return keys in file-offset order for sequential-read locality."""
        return self._handle.offset_keys()

    def metadata(self) -> Optional[Dict[str, str]]:
        return self._handle.metadata()

    # ── Spyre-aware tensor accessors ─────────────────────────────────────────

    def get_tensor(self, key: str) -> "torch.Tensor":
        """Load ``key`` on CPU then DMA to Spyre with the optimal layout."""
        cpu_tensor = self._handle.get_tensor(key)
        return _spyre_tensor_from_safetensors(
            cpu_tensor, key, DEVICE_NAME, target_dtype=self._target_dtype
        )

    def get_tensors(self) -> Dict[str, "torch.Tensor"]:
        """Load all tensors and DMA each to Spyre with the optimal layout."""
        result: Dict[str, "torch.Tensor"] = {}
        for key in self._handle.offset_keys():
            result[key] = self.get_tensor(key)
        return result

    def get_slice(self, key: str) -> "_SpyreSafeSlice":
        """Return a ``_SpyreSafeSlice`` that applies the Spyre hook on indexing."""
        return _SpyreSafeSlice(
            self._handle.get_slice(key), key, target_dtype=self._target_dtype
        )

    def __getattr__(self, name):
        """Forward methods added by future safetensors releases."""
        return getattr(self._handle, name)


class _SpyreSafeSlice:
    """Slice wrapper: applies ``_spyre_tensor_from_safetensors`` on ``__getitem__``.

    Gathers bytes on CPU first, then invokes the hook. This avoids the
    double-dispatch problem that would arise if we applied the hook inside
    the Rust slice path (Spyre → Spyre copy).
    """

    def __init__(self, cpu_slice, name: str, target_dtype=_UNSET):
        self._cpu_slice = cpu_slice
        self._name = name
        self._target_dtype = target_dtype

    def __getitem__(self, slices) -> "torch.Tensor":
        # The underlying CPU slice gives us a contiguous CPU tensor.
        cpu_tensor = self._cpu_slice[slices]
        return _spyre_tensor_from_safetensors(
            cpu_tensor, self._name, DEVICE_NAME, target_dtype=self._target_dtype
        )


def _patch_safetensors_for_spyre() -> None:
    """Monkey-patch safetensors for Spyre-aware loading.

    Patches three public entry points so that ``device="spyre"`` transparently
    applies optimal Spyre DMA layout per tensor via
    ``_spyre_tensor_from_safetensors``:

    1. ``safetensors.safe_open(path, framework="pt", device="spyre")``
       Returns ``_SpyreSafeOpen`` (opens on CPU internally, dispatches each
       ``get_tensor`` / ``get_tensors`` / slice through the hook).

    2. ``safetensors.torch.load_file(filename, device="spyre")``
       Delegates to the patched ``safe_open`` so ``f.get_tensors()`` goes
       through the hook path.

    3. ``safetensors.torch.load_model(model, filename, device="spyre")``
       Loads the state dict directly to Spyre via the patched load_file, then
       assigns each tensor into the model with direct _parameters / _buffers
       replacement, avoiding the copy_() no-op that breaks meta-device models.

    All patches are idempotent (``_spyre_patched`` sentinel). Non-Spyre
    device paths fall through to the originals.

    """
    try:
        import safetensors as _st_mod
        import safetensors.torch as _st_torch
    except ImportError:
        return  # safetensors not installed

    # ── 1. Wrap safetensors.safe_open ────────────────────────────────────────
    # Stash the original Rust class under _orig_safe_open so _SpyreSafeOpen
    # can always reach it regardless of further patching.
    if not hasattr(_st_mod, "_orig_safe_open"):
        _st_mod._orig_safe_open = _st_mod.safe_open

    if not getattr(_st_mod.safe_open, "_spyre_patched", False):
        _orig = _st_mod._orig_safe_open

        class _SpyreSafeOpenDispatch:
            """Route ``device="spyre"`` to ``_SpyreSafeOpen``; pass everything else through."""

            _spyre_patched = True

            def __new__(
                cls,
                filename,
                framework: str,
                device=None,
                backend: str = "mmap",
                target_dtype=_UNSET,
            ):
                if (
                    device is not None
                    and str(device).split(":")[0] == DEVICE_NAME
                    and framework == "pt"
                ):
                    return _SpyreSafeOpen(
                        filename,
                        framework=framework,
                        backend=backend,
                        target_dtype=target_dtype,
                    )
                # Non-Spyre device: the original safe_open has no
                # target_dtype concept, so it is intentionally not forwarded.
                return _orig(
                    filename, framework=framework, device=device, backend=backend
                )

        _st_mod.safe_open = _SpyreSafeOpenDispatch
        # Also update the name imported by safetensors.torch so that
        # ``from safetensors.torch import safe_open`` resolves correctly.
        if hasattr(_st_torch, "safe_open"):
            _st_torch.safe_open = _SpyreSafeOpenDispatch

    # ── 2. Patch safetensors.torch.load_file ─────────────────────────────────
    if not getattr(_st_torch.load_file, "_spyre_patched", False):
        _unpatched_load_file = _st_torch.load_file

        def _spyre_load_file(
            filename,
            device: Union[str, int] = "cpu",
            *,
            backend: str = "mmap",
            target_dtype=_UNSET,
        ) -> Dict[str, "torch.Tensor"]:
            if str(device).split(":")[0] != DEVICE_NAME:
                return _unpatched_load_file(filename, device=device, backend=backend)
            # Route through the patched safe_open so get_tensors() applies
            # the Spyre hook per tensor.
            with _st_mod.safe_open(
                filename,
                framework="pt",
                device=DEVICE_NAME,
                backend=backend,
                target_dtype=target_dtype,
            ) as f:
                return f.get_tensors()

        _spyre_load_file._spyre_patched = True  # type: ignore[attr-defined]
        _spyre_load_file._orig = _unpatched_load_file  # type: ignore[attr-defined]
        _st_torch.load_file = _spyre_load_file

    # ── 3. Patch safetensors.torch.load_model ────────────────────────────────
    # The key issue with meta-device models:
    #   model.load_state_dict(cpu_tensors) calls copy_() into each meta parameter,
    #   which is a no-op — the parameter stays on meta. A subsequent
    #   load_model_to_spyre() then tries to DMA from meta → Spyre and crashes.
    #
    # Correct flow:
    #   1. load_file(device="spyre") — each tensor is ALREADY on Spyre with
    #      the optimal layout via _spyre_tensor_from_safetensors.
    #   2. _assign_tensors_to_model() — directly replaces model._parameters[name]
    #      with the Spyre tensor; no copy_() involved, works on meta models.
    #   3. No load_model_to_spyre() needed — tensors are already on Spyre.
    if not getattr(_st_torch.load_model, "_spyre_patched", False):
        _orig_load_model = _st_torch.load_model

        def _spyre_load_model(
            model: "torch.nn.Module",
            filename,
            strict: bool = True,
            device: Union[str, int] = "cpu",
            *,
            assign: Optional[bool] = None,
            backend: str = "mmap",
            target_dtype=_UNSET,
        ):
            if str(device).split(":")[0] != DEVICE_NAME:
                return _orig_load_model(
                    model, filename, strict=strict, device=device, backend=backend
                )

            if target_dtype is not None and target_dtype is not _UNSET:
                from torch_spyre.model_utils import _validate_target_dtype

                _validate_target_dtype(target_dtype)

            # Load state dict directly to Spyre — each tensor gets the optimal
            # layout for its role (embedding/linear/other) via the hook.
            # Use _st_torch.load_file (our patched version) rather than the
            # local _spyre_load_file name, which may not be bound if block 2
            # was skipped due to idempotency.
            state_dict = _st_torch.load_file(
                filename, device=DEVICE_NAME, backend=backend, target_dtype=target_dtype
            )

            import torch.nn as nn
            from torch_spyre._inductor.logging_utils import get_inductor_logger

            logger = get_inductor_logger("model_utils")
            # Coercion is only "off" when the caller explicitly passed
            # target_dtype=None. If unspecified (_UNSET), the hook above
            # already resolved it to torch.float16 per-tensor — a dtype
            # mismatch against the model's own dtype is then expected and
            # intentional (the Spyre DMA hardware requirement), not an error.
            coercion_disabled = target_dtype is None

            # Group aliases before assignment. named_parameters() normally
            # deduplicates tied parameters; replacing only the first name would
            # silently sever the tie and leave the other alias unloaded.
            parameter_groups: Dict[int, List[tuple]] = {}
            for name, param in model.named_parameters(remove_duplicate=False):
                parameter_groups.setdefault(id(param), []).append((name, param))

            missing_keys: List[str] = []
            consumed_keys = set()
            for aliases in parameter_groups.values():
                checkpoint_names = [name for name, _ in aliases if name in state_dict]
                if not checkpoint_names:
                    missing_keys.append(aliases[0][0])
                    continue

                checkpoint_name = checkpoint_names[0]
                spyre_tensor = state_dict[checkpoint_name]
                param = aliases[0][1]
                if spyre_tensor.shape != param.shape:
                    raise RuntimeError(
                        f"size mismatch for {checkpoint_name}: copying a param with shape "
                        f"{spyre_tensor.shape} from checkpoint, the shape in "
                        f"current model is {param.shape}."
                    )
                if spyre_tensor.dtype != param.dtype:
                    if coercion_disabled:
                        raise RuntimeError(
                            f"dtype mismatch for {checkpoint_name}: checkpoint has "
                            f"{spyre_tensor.dtype}, model expects {param.dtype}."
                        )
                    # target_dtype coerced this tensor on the way in (e.g. the
                    # Spyre fp16 DMA requirement) — the model's post-load
                    # dtype for this param is now spyre_tensor.dtype by
                    # construction, since this is a Parameter replacement,
                    # not a copy_(). Not an error, but worth a trace.
                    logger.info(
                        "%s: model dtype %s coerced to %s on Spyre load "
                        "(target_dtype=%s)",
                        checkpoint_name,
                        param.dtype,
                        spyre_tensor.dtype,
                        target_dtype,
                    )

                replacement = nn.Parameter(
                    spyre_tensor, requires_grad=param.requires_grad
                )
                for name, _ in aliases:
                    parts = name.rsplit(".", 1)
                    parent = model.get_submodule(parts[0]) if len(parts) == 2 else model
                    parent._parameters[parts[-1]] = replacement
                consumed_keys.update(checkpoint_names)

            for name, buf in model.named_buffers(remove_duplicate=False):
                if name not in state_dict:
                    missing_keys.append(name)
                    continue
                spyre_tensor = state_dict[name]
                if spyre_tensor.shape != buf.shape:
                    raise RuntimeError(
                        f"size mismatch for {name}: copying a buffer with shape "
                        f"{spyre_tensor.shape} from checkpoint, the shape in "
                        f"current model is {buf.shape}."
                    )
                if spyre_tensor.dtype != buf.dtype:
                    if coercion_disabled:
                        raise RuntimeError(
                            f"dtype mismatch for {name}: checkpoint has "
                            f"{spyre_tensor.dtype}, model expects {buf.dtype}."
                        )
                    logger.info(
                        "%s: model dtype %s coerced to %s on Spyre load "
                        "(target_dtype=%s)",
                        name,
                        buf.dtype,
                        spyre_tensor.dtype,
                        target_dtype,
                    )
                parts = name.rsplit(".", 1)
                parent = model.get_submodule(parts[0]) if len(parts) == 2 else model
                parent._buffers[parts[-1]] = spyre_tensor
                consumed_keys.add(name)

            unexpected_keys = [name for name in state_dict if name not in consumed_keys]

            if strict and (missing_keys or unexpected_keys):
                missing_str = ", ".join(f'"{k}"' for k in missing_keys)
                unexpected_str = ", ".join(f'"{k}"' for k in unexpected_keys)
                error = (
                    f"Error(s) in loading state_dict for {model.__class__.__name__}:"
                )
                if missing_keys:
                    error += f"\n    Missing key(s) in state_dict: {missing_str}"
                if unexpected_keys:
                    error += f"\n    Unexpected key(s) in state_dict: {unexpected_str}"
                raise RuntimeError(error)

            return missing_keys, unexpected_keys

        _spyre_load_model._spyre_patched = True  # type: ignore[attr-defined]
        _st_torch.load_model = _spyre_load_model


def _patch_fx_graph_hash():
    """
    Extends FxGraphHashDetails to include SpyreTensorLayout in the cache key.
    """
    import torch
    from torch._inductor.codecache import FxGraphHashDetails
    from torch._inductor.virtualized import V

    if getattr(FxGraphHashDetails, "_spyre_hash_patched", False):
        return

    original_init = FxGraphHashDetails.__init__

    def _spyre_init(self, gm, example_inputs, fx_kwargs, inputs_to_check):
        # run original first — populates all standard hash fields
        original_init(self, gm, example_inputs, fx_kwargs, inputs_to_check)

        # V.get_real_inputs() returns real Spyre tensors with SpyreTensorLayout
        # before they become FakeTensors (which have no layout by design)

        try:
            real_inputs = V.get_real_inputs()
        except RuntimeError:
            return

        # extract layout from real tensors, fallback to example_inputs
        spyre_layouts = []
        # Use real_inputs only if it's a valid list/tuple, otherwise use example_inputs
        inputs_to_use = (
            real_inputs if isinstance(real_inputs, (list, tuple)) else example_inputs
        )

        for inp in inputs_to_use:
            if isinstance(inp, torch.Tensor):
                layout = inp.device_tensor_layout()
                spyre_layouts.append(layout)
            else:
                spyre_layouts.append(None)

        # self.spyre_layouts added as field on FxGraphHashDetails
        # PyTorch pickles ALL fields → spyre_layouts automatically in hash
        self.spyre_layouts = spyre_layouts

    FxGraphHashDetails.__init__ = _spyre_init
    FxGraphHashDetails._spyre_hash_patched = True
