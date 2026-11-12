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

"""Element arrangement passes.

``validate_ops`` validates the operands of a multi-arg op are compatible and
raises Unsupported when not compatible. It runs after
``propagate_spyre_tensor_layouts`` pass.

``insert_ea_shuffle`` runs after ``finalize_layouts`` commits the layouts and
after ``insert_restickify``. It finds the ops that mix STANDARD and
staggered operands and not broadcast (layout propagation case 3.3) and inserts a
shuffle op for each EA mismatched operand. The committed output EA is retained,
so layouts are not re-propagated. It first marks all such ops, then inserts the
shuffles.
"""

import logging

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    Scatter,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V
from torch_spyre._C import ElementArrangement, SpyreTensorLayout

from .constants import STAGGERED_EAS, is_ea_compatible
from .errors import Unsupported
from .ir import FixedTiledLayout, SpyreConstantFallback
from .logging_utils import get_inductor_logger
from .pass_utils import (
    indirect_info_from_op,
    origin_in_graph,
    redirect_computed_buffer_reads,
)
from .provenance import decompose_provenance

logger = get_inductor_logger("element_arrangement")

EA = ElementArrangement

_PASS_NAME = "insert_ea_shuffle"
_ALL_EAS = frozenset({EA.STANDARD, *STAGGERED_EAS})
_EXX2_OPS = frozenset({"layernormnorm", "layernormscale"})


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
def _memory_reads(op) -> list[MemoryDep]:
    return [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]


def _has_indirect_access(op, indirect_names) -> bool:
    """True if ``op`` is an indirect scatter/gather (index read or index write)."""
    if isinstance(op.data, Scatter) or indirect_names:
        return True
    read_writes = op.get_read_writes()
    return any(
        isinstance(d, MemoryDep)
        and any(str(s).startswith(("indirect", "tmp")) for s in d.index.free_symbols)
        for d in (*read_writes.reads, *read_writes.writes)
    )


def _committed_layout(graph: GraphLowering, name: str) -> FixedTiledLayout | None:
    layout = graph.get_buffer(name).get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        layout = layout.real_layout()
    return layout if isinstance(layout, FixedTiledLayout) else None


def _committed_stl(graph: GraphLowering, name: str) -> SpyreTensorLayout | None:
    layout = _committed_layout(graph, name)
    return None if layout is None else layout.device_layout


def _operand_stls(
    graph: GraphLowering, op, indirect_names
) -> dict[str, SpyreTensorLayout]:
    """Committed layout of each non-indirect operand of ``op``, in read order."""
    stls: dict[str, SpyreTensorLayout] = {}
    for dep in _memory_reads(op):
        if dep.name not in indirect_names and dep.name not in stls:
            stl = _committed_stl(graph, dep.name)
            if stl is not None:
                stls[dep.name] = stl
    return stls


def _is_broadcast(stl: SpyreTensorLayout) -> bool:
    return stl.stride_map[-1] == -1


def _with_ea(layout: FixedTiledLayout, ea: ElementArrangement) -> FixedTiledLayout:
    """``layout`` with only its element arrangement changed to ``ea``."""
    stl = layout.device_layout
    return FixedTiledLayout(
        layout.device,
        layout.dtype,
        list(layout.size),
        list(layout.stride),
        SpyreTensorLayout(stl.device_size, stl.stride_map, stl.device_dtype, ea),
    )


def _insert_shuffle_op(
    graph: GraphLowering,
    src_name: str,
    target_ea: ElementArrangement,
    before_op,
) -> ComputedBuffer:
    """Insert, just before ``before_op``, a copy of ``src_name`` in ``target_ea``.

    The copy is an ``aten.clone`` lowered from the buffer itself (one FX node
    can lower to several buffers, so the FX node is not a reliable source). Its
    layout is the committed input layout with the EA changed, because a shuffle
    only reorders elements within sticks. Codegen emits a ``shuffle`` because
    input and output arrangements differ. The copy is tagged ``ea_shuffle``.
    """
    from .lowering import clone as lower_clone

    def _fail(why: str):
        return Unsupported(f"Cannot shuffle {src_name} to {target_ea.name}: {why}")

    src_layout = _committed_layout(graph, src_name)
    if src_layout is None:
        raise _fail("it has no device layout")
    if src_name in graph.graph_inputs:
        src_tb = graph.graph_inputs[src_name]
    else:
        buf = graph.get_buffer(src_name)
        if not isinstance(buf, ComputedBuffer):
            raise _fail(f"it is a {type(buf).__name__}, not a computed buffer")
        src_tb = TensorBox(StorageBox(buf))
    src = src_tb.data.data

    clone_tb = lower_clone(src_tb)
    clone_tb.realize()  # make it a named ComputedBuffer in graph.operations
    shuffle_buf = clone_tb.data.data
    assert isinstance(shuffle_buf, ComputedBuffer), type(shuffle_buf).__name__
    # A buffer with no reads of its own is inlined by make_loader()
    if src_name not in shuffle_buf.get_read_names():
        raise _fail("the copy does not read it")
    shuffle_buf.layout = _with_ea(src_layout, target_ea)
    shuffle_buf.ea_shuffle = True  # type: ignore[attr-defined]
    if isinstance(src, ComputedBuffer):
        decompose_provenance(
            src,
            (shuffle_buf,),
            pass_name=_PASS_NAME,
            reason="insert EA shuffle",
            inherit_origins=False,
        )
    # Move the copy to just before ``before_op``.
    graph.operations.remove(shuffle_buf)
    graph.operations.insert(graph.operations.index(before_op), shuffle_buf)
    return shuffle_buf


# ---------------------------------------------------------------------------
# validate_ops
# ---------------------------------------------------------------------------
def _get_op_name(op) -> str:
    """Extract the operation name from a node for validation."""
    if not hasattr(op.data, "origins") or not op.data.origins:
        return ""

    # Prefer the origin in the current lowering graph (subgraph-safe); fall back
    # to an arbitrary origin when no graph context is available. That fallback
    # may yield a parent-graph node for a subgraph buffer, which is safe *here*
    # because this function only extracts a name string and never inserts into
    # or mutates a graph. Do not gate graph-mutating logic on the result.
    gl = V.graph
    origin_node = None
    if hasattr(gl, "graph"):
        origin_node = origin_in_graph(op.data.origins, gl.graph)
    if origin_node is None:
        origin_node = next(iter(op.data.origins))

    target = getattr(origin_node, "target", None)
    if target and hasattr(target, "name"):
        full_name = target.name()
        if "::" in full_name:
            # Extract just the operation name (e.g., "add" from "aten::add.Tensor")
            return full_name.split("::")[1].split(".")[0]
        return full_name
    return str(target) if target else str(origin_node)


def validate_ops(graph: GraphLowering) -> None:
    """Validate inputs to ops have compatible ElementArrangements.

    This pass needs to be run after propagate_spyre_tensor_layouts so that it
    has all the required SpyreTensorLayouts. It only reads the graph. A mixed
    STANDARD/staggered pointwise op is accepted here; ``insert_ea_shuffle``
    later shuffles the operands that broadcast cannot reconcile.
    """
    for op in graph.operations:
        if not isinstance(getattr(op, "data", None), Pointwise):
            continue

        inputs = _memory_reads(op)
        if len(inputs) <= 1:
            continue

        names, layouts = [], []
        for inp in inputs:
            # Skip buffers without layouts
            for layout in getattr(V.graph.get_buffer(inp.name), "layouts", ()):
                if isinstance(layout, SpyreTensorLayout):
                    names.append(inp.name)
                    layouts.append(layout)
        if len(layouts) <= 1:
            continue

        eas = [layout.element_arrangement for layout in layouts]
        op_name = _get_op_name(op)

        # layernormnorm/scale legitimately mix in an EXX2 operand
        if op_name in _EXX2_OPS and EA.EXX2 in eas:
            continue

        # Valid patterns (see is_ea_compatible): all operands share one EA, or
        # one non-STANDARD EA is mixed with STANDARD operands (broadcast).
        if not is_ea_compatible(eas):
            args = ", ".join(f'"{n}": {ea}' for n, ea in zip(names, eas))
            raise Unsupported(
                f"Incompatible ElementArrangement in multi-arg op. "
                f"Valid patterns: all inputs share one EA, or one non-STANDARD EA "
                f"broadcast against STANDARD inputs. "
                f"op: {op_name}, args: {args}"
            )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _format_op(
    graph: GraphLowering,
    op,
    stls: dict[str, SpyreTensorLayout],
    targets: dict[str, ElementArrangement] | None = None,
) -> str:
    """``buf7_<ea>: [buf4_<ea>, buf5_<target>[insert], ...]``.

    The op is shown with its committed output EA and its operands with their
    committed EAs; an operand in ``targets`` is shown in its target EA.
    """
    targets = targets or {}
    out_stl = _committed_stl(graph, op.get_name())
    out_ea = "?" if out_stl is None else out_stl.element_arrangement.name
    operands = ", ".join(
        f"{n}_{targets[n].name}[insert]"
        if n in targets
        else f"{n}_{s.element_arrangement.name}"
        for n, s in stls.items()
    )
    return f"{op.get_name()}_{out_ea}: [{operands}]"


def _log_op(graph: GraphLowering, op, stls, targets) -> None:
    """DEBUG: the op before and after the shuffles. INFO: the shuffles chosen."""
    committed = _format_op(graph, op, stls)
    if not targets:
        logger.debug("EA shuffle skip %s", committed)
        return
    planned = _format_op(graph, op, stls, targets)
    logger.debug("EA shuffle %s\n  path0: %s", committed, planned)
    logger.info(
        "EA shuffle select %s: shuffles=%d path: %s",
        op.get_name(),
        len(targets),
        planned,
    )


def _log_ea_graph(graph: GraphLowering, marked: list) -> None:
    """DEBUG: every pointwise op with the shuffles marked on it."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    targets = {op.get_name(): t for op, t in marked}
    lines = ["EA shuffle graph:"]
    for op in graph.operations:
        if isinstance(op, ComputedBuffer) and isinstance(op.data, Pointwise):
            stls = _operand_stls(graph, op, indirect_info_from_op(op)[0])
            if stls:
                name = op.get_name()
                lines.append("  " + _format_op(graph, op, stls, targets.get(name)))
    logger.debug("\n".join(lines))


# ---------------------------------------------------------------------------
# mark the ops that need a shuffle
# ---------------------------------------------------------------------------
def _check_staggered_eas(op, stls: dict[str, SpyreTensorLayout]) -> None:
    staggered = {s.element_arrangement for s in stls.values()} & STAGGERED_EAS
    if len(staggered) > 1:
        raise Unsupported(
            f"Cannot shuffle {op.get_name()}: multi-arg pointwise with multiple "
            f"staggered EAs not supported: {sorted(e.name for e in staggered)}"
        )


def _check_indirect_op(op, stls: dict[str, SpyreTensorLayout]) -> None:
    distinct = {s.element_arrangement for s in stls.values() if not _is_broadcast(s)}
    if len(distinct) > 1:
        raise Unsupported(
            f"Cannot shuffle {op.get_name()}: mixed element arrangements "
            f"({sorted(e.name for e in distinct)}) in an indirect access op"
        )


def _shuffle_targets(
    stls: dict[str, SpyreTensorLayout], out_ea: ElementArrangement
) -> dict[str, ElementArrangement]:
    """Operands to shuffle, mapped to the EA to shuffle them to (``out_ea``)."""
    eas = {s.element_arrangement for s in stls.values()}
    if EA.STANDARD not in eas or not eas & STAGGERED_EAS:
        return {}
    if not {out_ea, *eas} <= _ALL_EAS:
        return {}
    return {
        n: out_ea
        for n, s in stls.items()
        if s.element_arrangement != out_ea and not _is_broadcast(s)
    }


def _without_constants(
    graph: GraphLowering, stls: dict[str, SpyreTensorLayout]
) -> dict[str, SpyreTensorLayout]:
    return {
        n: s
        for n, s in stls.items()
        if not isinstance(graph.try_get_buffer(n), SpyreConstantFallback)
    }


def _mark_mismatched_ea(
    graph: GraphLowering,
) -> list[tuple[ComputedBuffer, dict[str, ElementArrangement]]]:
    """Mark ops with operands whose EA must be shuffled to the output EA.

    Returns ``(op, {operand name: target EA})`` for each such op.
    """
    marked = []
    for op in graph.operations:
        if not (isinstance(op, ComputedBuffer) and isinstance(op.data, Pointwise)):
            continue
        out_stl = _committed_stl(graph, op.get_name())
        indirect_names, _, _ = indirect_info_from_op(op)
        stls = _operand_stls(graph, op, indirect_names)
        if out_stl is None or len(stls) < 2:
            continue
        if _has_indirect_access(op, indirect_names):
            _check_indirect_op(op, stls)
            continue
        _check_staggered_eas(op, stls)
        targets = _shuffle_targets(
            _without_constants(graph, stls), out_stl.element_arrangement
        )
        _log_op(graph, op, stls, targets)
        if not targets:
            continue
        if getattr(op, "loop_info", None) is not None:
            raise Unsupported(
                f"EA shuffle for {op.get_name()} inside a coarse-tiled loop "
                f"is not supported"
            )
        marked.append((op, targets))
    return marked


# ---------------------------------------------------------------------------
# insert the shuffles at the marked locations
# ---------------------------------------------------------------------------
def insert_ea_shuffle(graph: GraphLowering) -> None:
    """Insert a shuffle before every op marked by ``_mark_mismatched_ea``.

    A (buffer, EA) pair is shuffled once and the shuffle is shared by all the
    ops that read it.
    """
    marked = _mark_mismatched_ea(graph)
    _log_ea_graph(graph, marked)
    shuffles: dict[tuple[str, ElementArrangement], str] = {}
    for op, targets in marked:
        for src, ea in targets.items():
            if (src, ea) not in shuffles:
                shuffles[(src, ea)] = _insert_shuffle_op(graph, src, ea, op).get_name()
        redirect_computed_buffer_reads(
            op,
            {src: shuffles[(src, ea)] for src, ea in targets.items()},
            graph.operations,
            pass_name=_PASS_NAME,
            reason="read EA-shuffled operands",
        )
    logger.info("EA shuffle plan: ops=%d shuffles=%d", len(marked), len(shuffles))
