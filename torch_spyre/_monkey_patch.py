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


from typing import TYPE_CHECKING, Optional

from torch._dynamo.guards import GuardBuilder

from torch_spyre.constants import DEVICE_NAME

if TYPE_CHECKING:
    from torch_spyre._C import SpyreTensorLayout


def _add_ea(src_tensor, res_tensor) -> None:
    """Update ElementArrangement (EA) tag on output SpyreTensorLayout

    For to_dtype op in eager mode.
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
        ElementArrangement,
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

    # FP32 -> FP16 runtime type conversion is not yet supported.
    if (
        src_tensor.dtype == torch.float32
        and res_tensor.dtype in DtypeOpTable.fp16_types()
    ):
        fmt = ElementArrangement.STANDARD

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
            # Support D2H and H2D dtype casting via DCI (DataConversionInfo) in spyre_mem.cpp.
            # For D2D data casting, split it into a D2H copy and a H2D dtype conversion.
            _device = kwargs.get("device", None)
            if (
                _device is None
                and len(args) > 0
                and isinstance(args[0], (str, torch.device))
            ):
                _device = args[0]
            _dtype = kwargs.get("dtype", None)
            if _dtype is None:
                if len(args) > 0 and isinstance(args[0], torch.dtype):
                    _dtype = args[0]
                elif len(args) > 1 and isinstance(args[1], torch.dtype):
                    _dtype = args[1]

            target_device_type = (
                torch.device(_device).type if _device is not None else None
            )

            if (
                target_device_type == DEVICE_NAME
                and _dtype is not None
                and self.device.type == DEVICE_NAME
            ):
                import warnings

                warnings.warn(
                    "D2D dtype conversion on Spyre is not directly supported. "
                    "Using CPU as an intermediate for the cast.",
                    stacklevel=2,
                )
                # Step 1: plain D2H copy (no dtype change)
                tmp = orig_to(self, "cpu")
                # Step 2: cast dtype via H2D
                return orig_to(tmp, _device, dtype=_dtype)

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
