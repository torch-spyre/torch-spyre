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
"""
pipeline/captures.py

In-process provenance capture across the compilation pipeline for issue #2574.

Every measurement reads the *actual* compilation object (no torch.export, no
heuristics). Each stage is observed by class-level monkey-patches installed for
the duration of one ``torch.compile``:

    Stage 2 pre-grad FX   CustomPreGradPasses.__call__(graph)        [passes.py]
    Stage 2 post-grad FX  CustomPostPasses.__call__(graph)          [passes.py]
    Stage 3 Inductor pass CustomPreSchedulingPasses.__call__(graph) [passes.py]
    Stage 4 LoopLevelIR     (read graph.operations after stage-3 passes)
    Stage 5 OpSpec        SpyreKernel.create_op_spec                [spyre_kernel.py]
    Stage 6 SuperDSC      SuperDSCScheduling.define_kernel +        [scheduler.py]
                          async_compile.get_output_dir (exact dirs) [async_compile.py]

DESIGN NOTE — this is Phase A (capture only). Each hook is wrapped in its own
try/except and records whether it fired and any error, so a signature mismatch
on a given torch build degrades to missing data (visible in the dump) rather
than crashing the whole audit. Field locations / signatures here are derived
from source reading and MUST be confirmed against the pinned torch build via
the raw dump before the report layer (Phase B) is built on top.
"""

from __future__ import annotations

import contextlib
from typing import Any

# FX node.meta provenance fields (issue #2574).
FX_META_FIELDS = [
    "stack_trace",
    "nn_module_stack",
    "source_fn_stack",
    "original_aten",
    "from_node",
]

# Inductor IR / LoopLevelIR provenance attributes (on ComputedBuffer / loops).
IR_ATTR_FIELDS = ["origins", "origin_node", "traceback"]

# Provenance fields a future OpSpec could carry. Phase 1: none are populated
# (the dataclass declares no provenance field); Phase 2 (#2575) adds
# debug_handle. Captured per OpSpec instance so the OpSpec column can show
# ◐ x/n once a field is declared but not populated on every op.
OPSPEC_PROVENANCE_FIELDS = IR_ATTR_FIELDS + ["get_stack_traces", "debug_handle"]

_MAX = 200  # repr truncation


def _safe(obj: Any) -> str:
    try:
        return str(obj)[:_MAX]
    except Exception as e:  # pragma: no cover - defensive
        return f"<unrepr-able {type(obj).__name__}: {e}>"


def _is_populated(val: Any) -> bool:
    """A provenance value counts as carried only if non-empty: ``0`` is
    populated (a valid handle); ``None`` / empty collection / ``""`` are not.
    This is the shared population rule used at every stage."""
    if val is None:
        return False
    if isinstance(val, (list, tuple, set, dict, str)) and len(val) == 0:
        return False
    return True


def _first_source_line(stack_trace: Any) -> str | None:
    if not isinstance(stack_trace, str):
        return None
    lines = [ln.strip() for ln in stack_trace.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _node_name(n: Any) -> str:
    return getattr(n, "name", None) or _safe(n)


def _summarize_fx_meta(node: Any) -> dict:
    """Per-field summary of an FX node's provenance meta.

    For each field: None if absent, else {"type", "repr"} so the dump records
    both the data structure (issue asks for "data structure and format") and a
    readable value.
    """
    meta = getattr(node, "meta", {}) or {}
    out: dict[str, Any] = {}
    for field in FX_META_FIELDS:
        if field not in meta:
            out[field] = None
            continue
        val = meta[field]
        rec = {
            "type": type(val).__name__,
            "repr": _safe(val),
            "nonempty": _is_populated(val),
        }
        if field == "stack_trace":
            rec["source_line"] = _first_source_line(val)
        out[field] = rec
    return {
        "name": _node_name(node),
        "op": getattr(node, "op", None),
        "target": _safe(getattr(node, "target", "")),
        "all_meta_keys": sorted(meta.keys()),
        "fields": out,
    }


def _summarize_ir_attrs(op: Any) -> dict:
    """Per-field summary of an IR operation's provenance attributes."""
    out: dict[str, Any] = {}
    origins = getattr(op, "origins", None)
    if origins is None:
        out["origins"] = None
    else:
        out["origins"] = [
            {"name": _node_name(n), "target": _safe(getattr(n, "target", ""))}
            for n in origins
        ]
    origin_node = getattr(op, "origin_node", None)
    out["origin_node"] = None if origin_node is None else _node_name(origin_node)
    tb = getattr(op, "traceback", None)
    out["traceback"] = None if tb is None else _safe(tb)
    # Spyre/Inductor expose a derived stack-trace accessor on some IR nodes.
    if hasattr(op, "get_stack_traces"):
        try:
            st = op.get_stack_traces()
            out["get_stack_traces"] = {"nonempty": bool(st), "repr": _safe(st)}
        except Exception as e:
            out["get_stack_traces"] = {"nonempty": False, "repr": f"<error: {e}>"}
    name = None
    for attr in ("get_operation_name", "get_name"):
        if hasattr(op, attr):
            try:
                name = getattr(op, attr)()
                break
            except Exception:
                pass
    # Record whether each provenance attribute *exists* on the object, separate
    # from its value. This distinguishes "the slot exists but is empty" (a real
    # absence / unused channel -> ✗) from "not an attribute of this object at
    # all" (not applicable -> ➖). getattr-with-default can't tell these apart.
    attr_exists = {
        a: hasattr(op, a)
        for a in ("origins", "origin_node", "traceback", "get_stack_traces")
    }

    return {
        "name": name or _safe(op),
        "type": type(op).__name__,
        "fields": out,
        "attr_exists": attr_exists,
    }


def _opspec_field_present(op_spec: Any, name: str) -> bool:
    """Is provenance field ``name`` declared AND populated on this OpSpec
    instance? (Existence + the shared non-empty rule.)"""
    return hasattr(op_spec, name) and _is_populated(getattr(op_spec, name))


def _graph_nodes(graph: Any):
    """Yield fx nodes from an fx.Graph or GraphModule, defensively."""
    g = getattr(graph, "graph", graph)  # GraphModule -> .graph; else assume Graph
    return list(getattr(g, "nodes", []))


@contextlib.contextmanager
def capture():
    """Install all stage hooks for the duration of the with-block.

    Yields a results dict, fully populated once the block exits. Top-level keys:
    stage2_pre_grad, stage2_post_grad, stage3_passes, stage4_looplevel,
    stage5_opspec, stage6_kernels, plus "_hooks" (fired/error per hook).
    """
    results: dict[str, Any] = {
        "stage2_pre_grad": {"fired": False, "nodes": []},
        "stage2_post_grad": {"fired": False, "nodes": []},
        "stage3_passes": {"fired": False, "before": [], "after": []},
        "stage4_looplevel": {"fired": False, "operations": []},
        "stage5_opspec": {"fired": False, "ops": [], "opspec_fields": None},
        "stage6_kernels": {"fired": False, "kernels": [], "output_dirs": {}},
        "_hooks": {},
    }

    # Imports are local so this module is importable without torch_spyre present.
    from torch_spyre._inductor import passes as _passes
    from torch_spyre._inductor import scheduler as _sched
    from torch_spyre._inductor import spyre_kernel as _sk
    from torch_spyre.execution import async_compile as _ac
    from torch_spyre._inductor.op_spec import OpSpec
    import dataclasses

    # Record OpSpec's declared fields once (structural: expect no provenance).
    try:
        results["stage5_opspec"]["opspec_fields"] = [
            f.name for f in dataclasses.fields(OpSpec)
        ]
    except Exception as e:
        results["stage5_opspec"]["opspec_fields_error"] = str(e)

    saved: list[tuple[Any, str, Any]] = []  # (obj, attr, original)

    def _patch(obj, attr, factory):
        original = getattr(obj, attr)
        setattr(obj, attr, factory(original))
        saved.append((obj, attr, original))

    # ---- Stage 2 pre-grad ------------------------------------------------
    def _pre_grad(original):
        def wrapper(self, *args, **kwargs):
            try:
                graph = args[0]
                results["stage2_pre_grad"]["fired"] = True
                results["stage2_pre_grad"]["nodes"] = [
                    _summarize_fx_meta(n)
                    for n in _graph_nodes(graph)
                    if getattr(n, "op", None) not in ("placeholder", "output")
                ]
            except Exception as e:
                results["_hooks"]["stage2_pre_grad"] = f"error: {e}"
            return original(self, *args, **kwargs)

        return wrapper

    # ---- Stage 2 post-grad -----------------------------------------------
    def _post_grad(original):
        def wrapper(self, *args, **kwargs):
            try:
                graph = args[0]
                results["stage2_post_grad"]["fired"] = True
                results["stage2_post_grad"]["nodes"] = [
                    _summarize_fx_meta(n)
                    for n in _graph_nodes(graph)
                    if getattr(n, "op", None) not in ("placeholder", "output")
                ]
            except Exception as e:
                results["_hooks"]["stage2_post_grad"] = f"error: {e}"
            return original(self, *args, **kwargs)

        return wrapper

    # ---- Stage 3 passes + Stage 4 LoopLevelIR ----------------------------
    def _pre_sched(original):
        def wrapper(self, *args, **kwargs):
            graph = args[0] if args else None
            try:
                results["stage3_passes"]["fired"] = True
                ops = list(getattr(graph, "operations", []) or [])
                results["stage3_passes"]["before"] = [
                    _summarize_ir_attrs(o) for o in ops
                ]
            except Exception as e:
                results["_hooks"]["stage3_before"] = f"error: {e}"
            ret = original(self, *args, **kwargs)
            try:
                ops = list(getattr(graph, "operations", []) or [])
                after = [_summarize_ir_attrs(o) for o in ops]
                results["stage3_passes"]["after"] = after
                # Stage 4 LoopLevelIR == operations state after pre-scheduling.
                results["stage4_looplevel"]["fired"] = True
                results["stage4_looplevel"]["operations"] = after
            except Exception as e:
                results["_hooks"]["stage3_after"] = f"error: {e}"
            return ret

        return wrapper

    # ---- Stage 5 OpSpec ---------------------------------------------------
    def _create_op_spec(original):
        def wrapper(self, *args, **kwargs):
            op_spec = original(self, *args, **kwargs)
            try:
                op, is_reduction, op_args, op_info = args[:4]
                results["stage5_opspec"]["fired"] = True
                ir_node = getattr(getattr(self, "current_node", None), "node", None)
                rec = {"op": op, "is_reduction": is_reduction}
                rec.update(_summarize_ir_attrs(ir_node) if ir_node is not None else {})
                # Per-instance: is each provenance field populated ON the OpSpec
                # object itself? Phase 1: all False (no such field). Phase 2+
                # this lets the OpSpec column show ◐ x/n if a field is declared
                # but not populated on every op (end-to-end guard, #2581).
                rec["opspec_present"] = {
                    name: _opspec_field_present(op_spec, name)
                    for name in OPSPEC_PROVENANCE_FIELDS
                }
                results["stage5_opspec"]["ops"].append(rec)
            except Exception as e:
                results["_hooks"]["stage5_opspec"] = f"error: {e}"
            return op_spec

        return wrapper

    # ---- Stage 6 define_kernel -------------------------------------------
    def _define_kernel(original):
        def wrapper(self, *args, **kwargs):
            kernel_name = original(self, *args, **kwargs)
            try:
                src_code, node_schedule, kernel = args[:3]
                results["stage6_kernels"]["fired"] = True
                node_origins = []
                for sn in node_schedule:
                    irn = getattr(sn, "node", None)
                    if irn is not None:
                        node_origins.append(_summarize_ir_attrs(irn))
                km = None
                try:
                    from torch._inductor.utils import get_kernel_metadata
                    from torch._inductor.virtualized import V

                    md, detailed = get_kernel_metadata(
                        node_schedule, V.graph.wrapper_code
                    )
                    # Keep these full (not _safe-truncated): the detailed
                    # source mapping is primary evidence for the analysis.
                    km = {"metadata": str(md), "detailed": str(detailed)}
                except Exception as e:
                    km = {"error": str(e)}
                results["stage6_kernels"]["kernels"].append(
                    {
                        "kernel_name": kernel_name,
                        "node_origins": node_origins,
                        "kernel_metadata": km,
                    }
                )
            except Exception as e:
                results["_hooks"]["stage6_define_kernel"] = f"error: {e}"
            return kernel_name

        return wrapper

    # ---- Stage 6 exact output dirs ---------------------------------------
    def _get_output_dir(original):
        def wrapper(*args, **kwargs):
            out = original(*args, **kwargs)
            try:
                kernel_name = args[0] if args else kwargs.get("kernel_name")
                results["stage6_kernels"]["output_dirs"][kernel_name] = out
            except Exception as e:
                results["_hooks"]["stage6_output_dir"] = f"error: {e}"
            return out

        return wrapper

    # Install all patches, each guarded so a missing symbol doesn't abort.
    hook_specs = [
        (_passes, "CustomPreGradPasses.__call__", _pre_grad, "stage2_pre_grad"),
        (_passes, "CustomPostPasses.__call__", _post_grad, "stage2_post_grad"),
        (_passes, "CustomPreSchedulingPasses.__call__", _pre_sched, "stage3_passes"),
        (_sk, "SpyreKernel.create_op_spec", _create_op_spec, "stage5_opspec"),
        (
            _sched,
            "SuperDSCScheduling.define_kernel",
            _define_kernel,
            "stage6_define_kernel",
        ),
        (_ac, "get_output_dir", _get_output_dir, "stage6_output_dir"),
    ]
    for module, dotted, factory, label in hook_specs:
        try:
            obj = module
            *parents, attr = dotted.split(".")
            for p in parents:
                obj = getattr(obj, p)
            _patch(obj, attr, factory)
            results["_hooks"].setdefault(label, "installed")
        except Exception as e:
            results["_hooks"][label] = f"install-failed: {e}"

    try:
        yield results
    finally:
        for obj, attr, original in reversed(saved):
            try:
                setattr(obj, attr, original)
            except Exception:
                pass
