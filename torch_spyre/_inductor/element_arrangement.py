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

"""Passes that check and act on the element arrangement (EA) of device tensors.

``validate_ops`` is a read-only check that validates the operands of a
multi-arg op are compatible and raises Unsupported. It runs after
``propagate_spyre_tensor_layouts`` pass.

``insert_ea_shuffle`` finds the optimized locations in the graph & insert the
shuffle op in those locations. It runs after ``insert_restickify`` pass.

Optimization stages:

1. ``find_ea_paths``: Find all possible paths to insert shuffe op.
2. ``optimize_ea_path``:  Rules that drop or minimze shuffle ops in the graph.
3. ``select_ea_path``: Selects the minimum cost path to insert shuffle graph.
"""

import logging
from dataclasses import dataclass

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Scatter,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V
from torch_spyre._C import (
    ElementArrangement,
    SpyreTensorLayout,
    get_device_size_in_bytes,
)

from .constants import STAGGERED_EAS, is_ea_compatible
from .dtype_ops import DtypeOpTable
from .errors import Unsupported
from .ir import FixedTiledLayout
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

# Attribute set on an op by stage 3 and consumed by stage 4. It is deliberately
# not ``ea_shuffle``, which already means "this buffer is a shuffle copy".
EA_SHUFFLE_PATH_ATTR = "ea_shuffle_path"

SHUFFLE_COST = 8.0
_STEP_COST = {
    "noop": 0.0,  # already in the target EA, or a broadcast stick
    "cancel": 0.0,  # undoes an earlier shuffle: read its source instead
    "reuse": 0.0,  # an identical shuffle already exists
    "sink": SHUFFLE_COST / 2,  # shuffle the narrow fp16 input of an upcast
    "collapse": SHUFFLE_COST,  # A -> B -> C becomes A -> C
    "insert": SHUFFLE_COST,
}
_SHUFFLEABLE_EAS = frozenset({EA.STANDARD, *STAGGERED_EAS})
_EXX2_OPS = frozenset({"layernormnorm", "layernormscale"})
# IR node kinds planned by ``_find_optimized_locations``.
_PLANNED_OP_TYPES = (Pointwise,)


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

    # Try to get the target and its name
    target = getattr(origin_node, "target", None)
    if target and hasattr(target, "name"):
        full_name = target.name()
        if "::" in full_name:
            # Extract just the operation name (e.g., "add" from "aten::add.Tensor")
            return full_name.split("::")[1].split(".")[0]
        return full_name

    # If we got here, just return the target as string
    return str(target) if target else str(origin_node)


def validate_ops(graph: GraphLowering) -> None:
    """Validate inputs to ops have same ElementArrangement.

    This pass need to be run after propagate_spyre_tensor_layouts so that it
    has all the required SpyreTensorLayouts. It only reads the graph: a mixed
    STANDARD/staggered pointwise op is accepted here and gets its staggered
    operands shuffled by ``insert_ea_shuffle``, which runs once layouts are
    committed.
    """
    for op in graph.operations:
        if not hasattr(op, "data"):
            continue
        if not isinstance(op.data, Pointwise):
            continue

        read_writes = op.get_read_writes()
        inputs = [r for r in read_writes.reads if isinstance(r, MemoryDep)]

        # Exclude single input ops
        if len(inputs) <= 1:
            continue

        layouts = []
        input_names = []
        for inp in inputs:
            buf = V.graph.get_buffer(inp.name)
            # Skip buffers without layouts
            if not hasattr(buf, "layouts"):
                continue
            for layout in buf.layouts:
                if isinstance(layout, SpyreTensorLayout):
                    layouts.append(layout)
                    input_names.append(inp.name)

        if len(layouts) <= 1:
            continue

        op_name = _get_op_name(op)

        # Check ElementArrangement compatibility
        stl_eas = [layout.element_arrangement for layout in layouts]

        # Skip ops with special ElementArrangement e.g. layernormnorm/scale with ElementArrangement.EXX2
        skip_ops = {"layernormnorm", "layernormscale"}
        skip_eas = {ElementArrangement.EXX2}
        if op_name in skip_ops and any(ea in skip_eas for ea in stl_eas):
            continue

        # Valid EA patterns (see is_ea_compatible):
        # 1. All operands share one EA.
        # 2. Exactly one distinct non-STANDARD EA (not EXX2) mixed with STANDARD
        #    operands (the broadcast pattern).
        if not is_ea_compatible(stl_eas):
            args_str = ", ".join(
                f'"{name}": {ea}' for name, ea in zip(input_names, stl_eas)
            )
            raise Unsupported(
                f"Incompatible ElementArrangement in multi-arg op. "
                f"Valid patterns: all inputs share one EA, or one non-STANDARD EA "
                f"broadcast against STANDARD inputs. "
                f"op: {op_name}, args: {args_str}"
            )


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
def _memory_reads(op) -> list[MemoryDep]:
    return [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]


def _has_indirect_access(op, read_writes, indirect_names) -> bool:
    """True if ``op`` is an indirect scatter/gather (index read or index write)."""
    if isinstance(op.data, Scatter) or indirect_names:
        return True
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


def _source_buffer(graph: GraphLowering, name: str):
    """The IR buffer called ``name``, looking through graph inputs."""
    if name in graph.graph_inputs:
        return graph.graph_inputs[name].data.data
    return graph.get_buffer(name)


def _same_geometry(a: SpyreTensorLayout, b: SpyreTensorLayout) -> bool:
    """True if ``a`` and ``b`` differ at most in their element arrangement."""
    return (a.device_size, a.stride_map, a.device_dtype) == (
        b.device_size,
        b.stride_map,
        b.device_dtype,
    )


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


def _reader_ops(graph: GraphLowering, name: str) -> list:
    """Operations that read buffer ``name`` through a ``MemoryDep``."""
    readers = []
    for op in graph.operations:
        if op.get_name() == name:
            continue
        try:
            reads = _memory_reads(op)
        except Exception:  # ops without read/write metadata cannot read ``name``
            continue
        if any(d.name == name for d in reads):
            readers.append(op)
    return readers


def _shuffle_source(op) -> str | None:
    """Name of the buffer a shuffle copy reads, or None if it is not a plain copy."""
    reads = _memory_reads(op)
    return reads[0].name if len(reads) == 1 else None


def _is_shuffle(op) -> bool:
    return bool(getattr(op, "ea_shuffle", False)) and isinstance(op, ComputedBuffer)


def _remove_op(graph: GraphLowering, op) -> None:
    """Drop ``op`` from the graph the way ``deadcode_elimination`` does."""
    graph.removed_buffers.add(op.get_name())
    graph.operations.remove(op)


def _place_before(graph: GraphLowering, op, before_op) -> None:
    """Move ``op`` to just before ``before_op`` (or to the end)."""
    graph.operations.remove(op)
    if before_op is not None and before_op in graph.operations:
        graph.operations.insert(graph.operations.index(before_op), op)
    else:
        graph.operations.append(op)


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
    from .lowering import (
        clone as lower_clone,
    )

    src_layout = _committed_layout(graph, src_name)
    if src_layout is None:
        raise Unsupported(
            f"Cannot shuffle {src_name} to {target_ea.name}: it has no device layout"
        )
    src = _source_buffer(graph, src_name)
    if src_name in graph.graph_inputs:
        src_tb = graph.graph_inputs[src_name]
    elif isinstance(src, ComputedBuffer):
        src_tb = TensorBox(StorageBox(src))
    else:
        raise Unsupported(
            f"Cannot shuffle {src_name} to {target_ea.name}: it is a "
            f"{type(src).__name__}, not a computed buffer"
        )

    clone_tb = lower_clone(src_tb)
    clone_tb.realize()  # make it a named ComputedBuffer in graph.operations
    shuffle_buf = clone_tb.data.data
    assert isinstance(shuffle_buf, ComputedBuffer), type(shuffle_buf).__name__
    # A buffer with no reads of its own is inlined by make_loader()
    if src_name not in shuffle_buf.get_read_names():
        raise Unsupported(
            f"Cannot shuffle {src_name} to {target_ea.name}: the copy does not read it"
        )
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
    _place_before(graph, shuffle_buf, before_op)
    return shuffle_buf


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ShuffleStep:
    """One shuffle of buffer ``src`` from ``from_ea`` into ``target_ea``.

    ``kind`` is the technique stage 2 chose for it (see ``_STEP_COST``);
    ``source`` is the buffer to read or shuffle instead of ``src`` after a
    chain was cancelled or collapsed.
    """

    src: str
    from_ea: ElementArrangement
    target_ea: ElementArrangement
    kind: str = "insert"
    cost: float = SHUFFLE_COST
    source: str | None = None

    @property
    def read_from(self) -> str:
        return self.source or self.src


@dataclass
class EAPath:
    """One way to execute ``op``: all operands in ``target_ea``.

    ``input_steps`` shuffles the operands that are not in ``target_ea``;
    ``output_step`` (set when ``target_ea`` differs from the committed output EA)
    shuffles the result back to the committed arrangement.
    """

    op: Operation | None
    target_ea: ElementArrangement
    input_steps: dict[str, ShuffleStep]
    output_step: ShuffleStep | None = None

    @property
    def steps(self) -> list[ShuffleStep]:
        out = [self.output_step] if self.output_step else []
        return [*self.input_steps.values(), *out]

    @property
    def work_steps(self) -> list[ShuffleStep]:
        """Steps that still change the graph (everything but no-ops)."""
        return [s for s in self.steps if s.kind != "noop"]

    @property
    def cost(self) -> float:
        return sum(s.cost for s in self.steps)


# ---------------------------------------------------------------------------
# Stage 1: find all paths
# ---------------------------------------------------------------------------
def ea_decision_table(
    operand_eas: dict[str, ElementArrangement],
    output_ea: ElementArrangement,
    output_name: str = "out",
) -> list[EAPath]:
    """The decision table of an op: one unoptimized path per candidate target EA.

    Candidates are STANDARD, every operand EA and the committed output EA. For
    target ``T`` the path shuffles each operand whose EA is not ``T`` into ``T``
    and, if ``T`` is not ``output_ea``, shuffles the result from ``T`` back to
    ``output_ea``. Every step is a full-cost "insert"; stage 2 makes them
    cheaper. Pure function of the arrangements, so it needs no graph.
    """
    candidates = sorted(
        {EA.STANDARD, output_ea, *operand_eas.values()}, key=lambda e: e.name
    )
    return [
        EAPath(
            None,
            target,
            {
                name: ShuffleStep(name, ea, target)
                for name, ea in operand_eas.items()
                if ea != target
            },
            ShuffleStep(output_name, target, output_ea)
            if target != output_ea
            else None,
        )
        for target in candidates
    ]


def _operand_stls(
    graph: GraphLowering, read_writes, indirect_names
) -> dict[str, SpyreTensorLayout]:
    """Committed layouts of the non-indirect inputs, by buffer name."""
    stls: dict[str, SpyreTensorLayout] = {}
    for dep in read_writes.reads:
        if (
            isinstance(dep, MemoryDep)
            and dep.name not in indirect_names
            and dep.name not in stls
        ):
            stl = _committed_stl(graph, dep.name)
            if stl is not None:
                stls[dep.name] = stl
    return stls


def find_ea_paths(
    graph: GraphLowering,
    op: Operation,
    outputs: set[str] | frozenset[str] = frozenset(),
) -> list[EAPath]:
    """Every path ``op`` can take, from the committed layouts of its buffers.

    Works for any op with two or more non-indirect operands. Returns [] when
    there is nothing to reconcile or an EA cannot be shuffled (EXX2). Only
    paths that keep the committed output EA are returned: changing it would
    need the layouts re-propagated, which is expensive and circular. An
    indirect (scatter or gather) op cannot be shuffled, so one that mixes
    operand EAs raises ``Unsupported`` rather than compile to wrong values.
    """
    op_name = op.get_name()
    out_stl = _committed_stl(graph, op_name)
    read_writes = op.get_read_writes()
    indirect_names, _, _ = indirect_info_from_op(op)
    operands = _operand_stls(graph, read_writes, indirect_names)
    if out_stl is None or len(operands) < 2:
        return []
    operand_eas = {n: s.element_arrangement for n, s in operands.items()}
    out_ea = out_stl.element_arrangement
    if not {out_ea, *operand_eas.values()} <= _SHUFFLEABLE_EAS:
        return []

    if _has_indirect_access(op, read_writes, indirect_names):
        distinct = {
            s.element_arrangement for s in operands.values() if s.stride_map[-1] != -1
        }
        if len(distinct) > 1:
            raise Unsupported(
                f"Cannot shuffle {op_name}: mixed element arrangements "
                f"({sorted(e.name for e in distinct)}) in an indirect access op"
            )
        return []

    paths = ea_decision_table(operand_eas, out_ea, op_name)
    for path in paths:
        path.op = op
    return [p for p in paths if p.output_step is None]


# ---------------------------------------------------------------------------
# Stage 2: rules that remove or minimize shuffles
# ---------------------------------------------------------------------------
def shuffle_is_noop(
    graph: GraphLowering, src_name: str, target_ea: ElementArrangement
) -> bool:
    """True if shuffling ``src_name`` into ``target_ea`` changes nothing.

    That holds when the buffer is already in ``target_ea``, or when its stick is
    a broadcast (``stride_map[-1] == -1``): every load reads the same element,
    so the element order within the stick cannot matter.
    """
    stl = _committed_stl(graph, src_name)
    if stl is None:
        return False
    return stl.element_arrangement == target_ea or stl.stride_map[-1] == -1


def find_chain_source(
    graph: GraphLowering, src_name: str, target_ea: ElementArrangement
) -> tuple[str, bool] | None:
    """If ``src_name`` is itself a shuffle ``A -> src_name``, look through it.

    Returns ``(A, cancels)`` when ``A`` and ``src_name`` share a geometry.
    ``cancels`` is True if ``A`` is already in ``target_ea`` (the two shuffles
    undo each other: read ``A``); otherwise the chain collapses to a single
    shuffle ``A -> target_ea``. Returns None when ``src_name`` is no shuffle.
    """
    if src_name in graph.graph_inputs:
        return None
    src = graph.get_buffer(src_name)
    a_name = _shuffle_source(src) if _is_shuffle(src) else None
    if a_name is None:
        return None
    a_stl, src_stl = _committed_stl(graph, a_name), _committed_stl(graph, src_name)
    if a_stl is None or src_stl is None or not _same_geometry(a_stl, src_stl):
        return None
    return a_name, a_stl.element_arrangement == target_ea


def find_existing_shuffle(
    graph: GraphLowering,
    src_name: str,
    target_ea: ElementArrangement,
    before_op: Operation | None = None,
) -> ComputedBuffer | None:
    """An existing shuffle of ``src_name`` into ``target_ea`` to share, or None.

    Only shuffles placed before ``before_op`` qualify, so the reader can use it.
    """
    src_stl = _committed_stl(graph, src_name)
    if src_stl is None:
        return None
    ops = graph.operations
    limit = ops.index(before_op) if before_op in ops else len(ops)
    for op in ops[:limit]:
        if not _is_shuffle(op) or _shuffle_source(op) != src_name:
            continue
        out_stl = _committed_stl(graph, op.get_name())
        if (
            out_stl is not None
            and out_stl.element_arrangement == target_ea
            and _same_geometry(out_stl, src_stl)
        ):
            return op
    return None


def _produces_standard(src_dtype, dst_dtype, ea: ElementArrangement) -> bool:
    """True if a ``src_dtype -> dst_dtype`` cast of an ``ea`` input yields STANDARD."""
    try:
        return DtypeOpTable.ea_map(src_dtype, dst_dtype, ea) == EA.STANDARD
    except Unsupported:
        return False


def find_cast_to_sink_into(
    graph: GraphLowering,
    src_name: str,
    consumer: Operation,
    outputs: set[str] | frozenset[str] = frozenset(),
) -> tuple[ComputedBuffer, str, ElementArrangement] | None:
    """If ``src_name`` is an upcast that can produce STANDARD by itself, say how.

    Shuffling the wide fp32 result of ``x(fp16) -> cast`` moves 4 B/element.
    Shuffling the narrow fp16 input ``x`` into an arrangement ``e_in`` for which
    the cast emits STANDARD moves half the bytes and needs no second pass.

    The cast is rewritten in place, so it must be a whole-tensor, single-input
    pointwise conversion read only by ``consumer`` and not a graph output.
    Returns ``(cast, x_name, e_in)`` or None.
    """
    if src_name in outputs or src_name in graph.graph_inputs:
        return None
    cast = graph.get_buffer(src_name)
    if (
        not isinstance(cast, ComputedBuffer)
        or not isinstance(cast.data, Pointwise)
        or getattr(cast, "loop_info", None) is not None
        or getattr(cast, "_read_copy_elision_record", None) is not None
        or _reader_ops(graph, src_name) != [consumer]
    ):
        return None
    cast_stl = _committed_stl(graph, src_name)
    if cast_stl is None or cast_stl.element_arrangement not in STAGGERED_EAS:
        return None

    reads = _memory_reads(cast)
    if (
        len(reads) != 1
        or reads[0].index != next(iter(cast.get_read_writes().writes)).index
    ):
        return None
    x_name = reads[0].name
    x_stl = _committed_stl(graph, x_name)
    x_dtype = _source_buffer(graph, x_name).get_dtype()
    if (
        x_stl is None
        or x_stl.element_arrangement != EA.STANDARD
        or x_dtype == cast.get_dtype()
        or get_device_size_in_bytes(x_stl) >= get_device_size_in_bytes(cast_stl)
    ):
        return None
    e_in = next(
        (e for e in STAGGERED_EAS if _produces_standard(x_dtype, cast.get_dtype(), e)),
        None,
    )
    return None if e_in is None else (cast, x_name, e_in)


def find_unread_shuffles(
    graph: GraphLowering, outputs: set[str] | None = None
) -> list[ComputedBuffer]:
    """Shuffle copies that nothing reads and that are not graph outputs.

    They are left behind when a chain is cancelled or collapsed.
    """
    if outputs is None:
        outputs = set(graph.get_output_names())
    return [
        op
        for op in graph.operations
        if _is_shuffle(op)
        and op.get_name() not in outputs
        and not _reader_ops(graph, op.get_name())
    ]


def optimize_ea_step(
    graph: GraphLowering,
    step: ShuffleStep,
    consumer: Operation,
    outputs: set[str] | frozenset[str] = frozenset(),
) -> ShuffleStep:
    """Apply the rules to one input step, setting its ``kind``, ``cost`` and ``source``.

    Rule order: no-op, sink through an upcast (STANDARD target only), look
    through a chained shuffle, reuse an identical shuffle, else insert one.
    """
    src, ea = step.src, step.target_ea
    kind = "insert"
    if shuffle_is_noop(graph, src, ea):
        kind = "noop"
    elif ea == EA.STANDARD and find_cast_to_sink_into(graph, src, consumer, outputs):
        kind = "sink"
    else:
        chain = find_chain_source(graph, src, ea)
        if chain is not None:
            step.source, cancels = chain
            kind = "cancel" if cancels else "collapse"
        if kind != "cancel" and find_existing_shuffle(
            graph, step.read_from, ea, before_op=consumer
        ):
            kind = "reuse"
    step.kind, step.cost = kind, _STEP_COST[kind]
    return step


def optimize_ea_path(
    graph: GraphLowering,
    path: EAPath,
    outputs: set[str] | frozenset[str] = frozenset(),
) -> EAPath:
    """Apply ``optimize_ea_step`` to every input step of ``path``."""
    for step in path.input_steps.values():
        optimize_ea_step(graph, step, path.op, outputs)
    return path


# ---------------------------------------------------------------------------
# Stage 3: select and mark
# ---------------------------------------------------------------------------
def _path_rank(path: EAPath) -> tuple:
    """Cheapest first. Ties keep the op's committed output layout (no output
    shuffle, i.e. flip the operands), then prefer fewer shuffles, then STANDARD.

    The tie matters: an op with one staggered and two STANDARD operands and a
    staggered output costs 2 shuffles either way (shuffle the staggered operand
    and the result, or flip both STANDARD operands). Flipping leaves the op's
    layout untouched, so it is the safer choice.
    """
    return (
        path.cost,
        path.output_step is not None,
        len(path.work_steps),
        path.target_ea != EA.STANDARD,
        path.target_ea.name,
    )


def select_ea_path(paths: list[EAPath]) -> EAPath | None:
    """The cheapest path, or None if there are none or it needs no graph change."""
    best = min(paths, key=_path_rank, default=None)
    return best if best is not None and best.work_steps else None


def _operand_eas(graph: GraphLowering, op: Operation) -> dict[str, ElementArrangement]:
    """Committed EA of each non-indirect operand of ``op``, in read order."""
    indirect_names, _, _ = indirect_info_from_op(op)
    stls = _operand_stls(graph, op.get_read_writes(), indirect_names)
    return {n: s.element_arrangement for n, s in stls.items()}


def _format_committed(graph: GraphLowering, op: Operation) -> str:
    """``buf19_<ea>: [buf16_<ea>, ...]`` with the committed EA of every buffer."""
    out_stl = _committed_stl(graph, op.get_name())
    out_ea = "?" if out_stl is None else out_stl.element_arrangement.name
    operands = ", ".join(f"{n}_{ea.name}" for n, ea in _operand_eas(graph, op).items())
    return f"{op.get_name()}_{out_ea}: [{operands}]"


def format_ea_path(path: EAPath, operand_eas: dict[str, ElementArrangement]) -> str:
    """``buf19_<ea>: [buf16_<ea>[insert], buf14_<ea>]``: the op and its operands
    all in ``path.target_ea``; a shuffled operand is tagged with its technique.
    """
    ea = path.target_ea.name
    operands = ", ".join(
        f"{n}_{ea}[{path.input_steps[n].kind}]"
        if n in path.input_steps
        else f"{n}_{ea}"
        for n in operand_eas
    )
    return f"{path.op.get_name()}_{ea}: [{operands}]"  # type: ignore[union-attr]



def log_ea_paths(
    graph: GraphLowering,
    op: Operation,
    paths: list[EAPath],
    selected: EAPath | None,
) -> None:
    """Log the paths of ``op``.

    No shuffle needed: one DEBUG line. Otherwise DEBUG: the op IR and every
    candidate path, cheapest first; INFO: the selected path.
    """
    if not paths:
        return
    operand_eas = _operand_eas(graph, op)
    if selected is None:
        logger.debug("EA shuffle skip %s", _format_committed(graph, op))
        return
    if logger.isEnabledFor(logging.DEBUG):
        lines = [
            f"EA shuffle {_format_committed(graph, op)}",
        ]
        for i, p in enumerate(sorted(paths, key=_path_rank)):
            mark = " <- selected" if p is selected else ""
            lines.append(
                f"  path{i} cost={p.cost}{mark}: {format_ea_path(p, operand_eas)}"
            )
        logger.debug("\n".join(lines))
    logger.info(
        "EA shuffle select %s: target=%s cost=%s path: %s",
        op.get_name(),
        selected.target_ea.name,
        selected.cost,
        format_ea_path(selected, operand_eas),
    )


def log_ea_graph(graph: GraphLowering, planned: list[EAPath]) -> None:
    """DEBUG: every planned op with the shuffles marked on it.

    INFO: one line with the number of shuffles and their total cost.
    """
    if logger.isEnabledFor(logging.DEBUG):
        lines = ["EA shuffle graph:"]
        for op in graph.operations:
            if not (
                isinstance(op, ComputedBuffer)
                and isinstance(op.data, _PLANNED_OP_TYPES)
                and _committed_stl(graph, op.get_name()) is not None
            ):
                continue
            operand_eas = _operand_eas(graph, op)
            if not operand_eas:
                continue
            path = getattr(op, EA_SHUFFLE_PATH_ATTR, None)
            lines.append(
                "  "
                + (
                    format_ea_path(path, operand_eas)
                    if isinstance(path, EAPath)
                    else _format_committed(graph, op)
                )
            )
        logger.debug("\n".join(lines))
    logger.info(
        "EA shuffle plan: ops=%d shuffles=%d cost=%s",
        len(planned),
        sum(len(p.work_steps) for p in planned),
        sum(p.cost for p in planned),
    )


def mark_ea_shuffle_path(op: Operation, path: EAPath | None) -> None:
    """Mark ``op`` with its selected ``path`` for stage 4 (no-op if ``path`` is None).

    A coarse-tiled op cannot take new shuffles: a path that costs something
    raises ``Unsupported``; a free one (only cancels/reuses) is left unmarked.
    """
    if path is None:
        return
    if getattr(op, "loop_info", None) is not None:
        if path.cost > 0:
            raise Unsupported(
                f"EA shuffle for {op.get_name()} inside a coarse-tiled loop "
                f"is not supported"
            )
        return
    setattr(op, EA_SHUFFLE_PATH_ATTR, path)


# ---------------------------------------------------------------------------
# Stage 4: insert at the marked locations
# ---------------------------------------------------------------------------
def _sink_cast(
    graph: GraphLowering, name: str, consumer: Operation, outputs: set[str]
) -> bool:
    """Make the upcast ``name`` produce STANDARD by shuffling its narrow input."""
    sink = find_cast_to_sink_into(graph, name, consumer, outputs)
    if sink is None:
        return False
    cast, x_name, e_in = sink
    x_shuffle = find_existing_shuffle(
        graph, x_name, e_in, before_op=cast
    ) or _insert_shuffle_op(graph, x_name, e_in, cast)
    # The redirect rebuilds the buffer: set the layout on the replacement, since
    # the old object is no longer in the graph.
    new_cast = redirect_computed_buffer_reads(
        cast,
        {x_name: x_shuffle.get_name()},
        graph.operations,
        pass_name=_PASS_NAME,
        reason="read the shuffled narrow input",
    )
    new_cast.layout = _with_ea(new_cast.layout, EA.STANDARD)
    logger.info("Sunk EA shuffle through cast %s (%s in %s)", name, x_name, e_in.name)
    return True


def _realize_input_step(
    graph: GraphLowering, step: ShuffleStep, op: Operation, outputs: set[str]
) -> str:
    """Do ``step`` in the graph and return the buffer ``op`` should read."""
    if step.kind == "noop":
        return step.src
    if step.kind == "cancel":
        return step.read_from
    if step.kind == "sink" and _sink_cast(graph, step.src, op, outputs):
        return step.src
    source = step.read_from
    shuffle = find_existing_shuffle(
        graph, source, step.target_ea, before_op=op
    ) or _insert_shuffle_op(graph, source, step.target_ea, op)
    return shuffle.get_name()


def _shuffle_output(graph: GraphLowering, op: Operation, path: EAPath) -> None:
    """Run ``op`` in ``path.target_ea`` and shuffle its result to the committed EA."""
    assert path.output_step is not None
    name = op.get_name()
    op.layout = _with_ea(op.layout, path.target_ea)
    idx = graph.operations.index(op)
    after = graph.operations[idx + 1] if idx + 1 < len(graph.operations) else None
    out_shuffle = _insert_shuffle_op(graph, name, path.output_step.target_ea, after)
    for reader in list(graph.operations):
        if (
            reader is not op
            and reader is not out_shuffle
            and isinstance(reader, ComputedBuffer)
            and name in reader.get_read_names()
        ):
            redirect_computed_buffer_reads(
                reader,
                {name: out_shuffle.get_name()},
                graph.operations,
                pass_name=_PASS_NAME,
                reason="read EA-shuffled op output",
            )


def apply_ea_path(graph: GraphLowering, path: EAPath, outputs: set[str]) -> None:
    """Edit the graph so ``path.op`` follows ``path``."""
    op = path.op
    assert op is not None
    name_map = {
        name: new
        for name, step in path.input_steps.items()
        if (new := _realize_input_step(graph, step, op, outputs)) != name
    }
    if name_map:
        # Keep the replacement: ``op`` is no longer in the graph after this.
        op = redirect_computed_buffer_reads(
            op,
            name_map,
            graph.operations,
            pass_name=_PASS_NAME,
            reason="read EA-shuffled operands",
        )
    if path.output_step is not None:
        _shuffle_output(graph, op, path)


def _find_optimized_locations(graph: GraphLowering) -> None:
    outputs = set(graph.get_output_names())
    planned: list[EAPath] = []
    for op in graph.operations:
        if not (
            isinstance(op, ComputedBuffer) and isinstance(op.data, _PLANNED_OP_TYPES)
        ):
            continue
        paths = [
            optimize_ea_path(graph, p, outputs)
            for p in find_ea_paths(graph, op, outputs)
        ]
        selected = select_ea_path(paths)
        log_ea_paths(graph, op, paths, selected)
        mark_ea_shuffle_path(op, selected)
        marked = getattr(op, EA_SHUFFLE_PATH_ATTR, None)
        if isinstance(marked, EAPath):
            planned.append(marked)
    log_ea_graph(graph, planned)


def insert_ea_shuffle(graph: GraphLowering) -> None:
    """Insert EA shuffle after finding optimized & correct locations in the graph."""
    _find_optimized_locations(graph)
    outputs = set(graph.get_output_names())
    for op in list(graph.operations):
        path = getattr(op, EA_SHUFFLE_PATH_ATTR, None)
        if isinstance(path, EAPath):
            apply_ea_path(graph, path, outputs)
            setattr(op, EA_SHUFFLE_PATH_ATTR, None)
    for dead in find_unread_shuffles(graph, outputs):
        _remove_op(graph, dead)
