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

``validate_ops`` is a read-only check, run right after layout propagation, that
the operands of a multi-arg pointwise op have compatible arrangements.
``insert_ea_shuffle`` makes the operands of a mixed STANDARD/staggered pointwise
op agree by shuffling the staggered ones to STANDARD; ``optimize_shuffling``
then removes or cheapens the shuffles it inserted. Those two run after
``finalize_layouts``/``insert_restickify``, so they decide from the committed
layout of every buffer.
"""

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

_INSERT_PASS_NAME = "insert_ea_shuffle"
_OPTIMIZE_PASS_NAME = "optimize_shuffling"


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
# Helpers
# ---------------------------------------------------------------------------
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
    """The committed layout of buffer ``name``, or None if it has no device layout."""
    layout = graph.get_buffer(name).get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        layout = layout.real_layout()
    return layout if isinstance(layout, FixedTiledLayout) else None


def _committed_stl(graph: GraphLowering, name: str) -> SpyreTensorLayout | None:
    layout = _committed_layout(graph, name)
    return None if layout is None else layout.device_layout


def _needs_shuffle(in_layout: FixedTiledLayout, out_ea: ElementArrangement) -> bool:
    """True if an operand with ``in_layout`` must be shuffled to ``out_ea``.

    Only a staggered operand feeding an op that produces STANDARD needs it, and
    only if its stick is not a broadcast: with ``stride_map[-1] == -1`` every
    load addresses the same element of the stick, so the order cannot matter.
    """
    stl = in_layout.device_layout
    return (
        out_ea == ElementArrangement.STANDARD
        and stl.element_arrangement in STAGGERED_EAS
        and stl.stride_map[-1] != -1
    )


def _insert_shuffle_op(
    graph: GraphLowering,
    src_name: str,
    target_ea: ElementArrangement,
    before_op,
) -> ComputedBuffer:
    """Insert, just before ``before_op``, a copy of ``src_name`` in ``target_ea``.

    The copy is an ``aten.clone`` lowered from the buffer itself, not from the
    FX node that happens to share its origin: one FX node can lower to several
    buffers, and its value is only the last of them. Its output layout is the
    committed input layout with the EA changed to ``target_ea``, because a
    shuffle only reorders elements within sticks. Codegen emits a ``shuffle``
    because the input and output arrangements differ. The copy is tagged
    ``ea_shuffle`` so ``optimize_shuffling`` can find it.
    """
    from .lowering import (
        clone as lower_clone,
    )  # deferred: lowering.py imports insert_restickify at module level

    src_layout = _committed_layout(graph, src_name)
    if src_layout is None:
        raise Unsupported(
            f"Cannot shuffle {src_name} to {target_ea.name}: it has no device layout"
        )
    src_stl = src_layout.device_layout

    if src_name in graph.graph_inputs:
        src_tb = graph.graph_inputs[src_name]
        src = src_tb.data.data
    else:
        src = graph.get_buffer(src_name)
        if not isinstance(src, ComputedBuffer):
            raise Unsupported(
                f"Cannot shuffle {src_name} to {target_ea.name}: it is a "
                f"{type(src).__name__}, not a computed buffer"
            )
        src_tb = TensorBox(StorageBox(src))

    # clone attaches a synthetic aten.clone origin to the result; realize it so
    # it is a named ComputedBuffer in graph.operations.
    clone_tb = lower_clone(src_tb)
    clone_tb.realize()
    shuffle_buf = clone_tb.data.data
    assert isinstance(shuffle_buf, ComputedBuffer), (
        f"Expected ComputedBuffer, got {type(shuffle_buf).__name__}"
    )
    # A buffer with no reads of its own is inlined by make_loader(), which would
    # leave a copy that does not read ``src_name`` at all.
    if src_name not in shuffle_buf.get_read_names():
        raise Unsupported(
            f"Cannot shuffle {src_name} to {target_ea.name}: the copy does not read it"
        )

    shuffle_buf.layout = FixedTiledLayout(
        src_layout.device,
        src_layout.dtype,
        list(src_layout.size),
        list(src_layout.stride),
        SpyreTensorLayout(
            src_stl.device_size,
            src_stl.stride_map,
            src_stl.device_dtype,
            target_ea,
        ),
    )
    shuffle_buf.ea_shuffle = True  # type: ignore[attr-defined]
    if isinstance(src, ComputedBuffer):
        decompose_provenance(
            src,
            (shuffle_buf,),
            pass_name=_INSERT_PASS_NAME,
            reason="insert EA shuffle",
            inherit_origins=False,
        )

    # lowering appended it to the end; place it just before its consumer.
    graph.operations.remove(shuffle_buf)
    graph.operations.insert(graph.operations.index(before_op), shuffle_buf)
    return shuffle_buf


# ---------------------------------------------------------------------------
# insert_ea_shuffle
# ---------------------------------------------------------------------------
def insert_ea_shuffle(graph: GraphLowering) -> None:
    """Shuffle staggered operands of a pointwise op that produces STANDARD.

    A pointwise op can mix a STANDARD and a staggered operand (for example an
    fp32 value that is the upcast of an fp16 one) and still be laid out as a
    STANDARD op; it then has to read every operand in STANDARD order. This pass
    inserts a shuffle in front of each staggered operand whose stick is not a
    broadcast and redirects the op to read the shuffled copy.

    Runs after ``finalize_layouts`` and ``insert_restickify``, so the decision is
    taken from the committed layout of each operand and of the op's output, not
    from layout candidates, and the shuffle's geometry is the operand's final
    one. Read-only validation of the arrangements stays in ``validate_ops``.

    Indirect scatter/gather is skipped: the index operand has no arrangement to
    match, and shuffling the data operand would un-stagger it while the output
    kept its staggered tag.
    """
    for op in list(graph.operations):
        if not isinstance(op, ComputedBuffer) or not isinstance(op.data, Pointwise):
            continue
        out_layout = op.layout
        if not isinstance(out_layout, FixedTiledLayout):
            continue
        out_ea = out_layout.device_layout.element_arrangement
        if out_ea != ElementArrangement.STANDARD:
            continue

        read_writes = op.get_read_writes()
        indirect_names, _, _ = indirect_info_from_op(op)
        if _has_indirect_access(op, read_writes, indirect_names):
            continue

        names = dict.fromkeys(
            r.name
            for r in read_writes.reads
            if isinstance(r, MemoryDep) and r.name not in indirect_names
        )
        # A single operand is a conversion or a copy: it consumes a staggered
        # operand on purpose and produces STANDARD.
        if len(names) <= 1:
            continue

        to_shuffle = []
        for name in names:
            in_layout = _committed_layout(graph, name)
            if in_layout is not None and _needs_shuffle(in_layout, out_ea):
                to_shuffle.append(name)
        if not to_shuffle:
            continue

        # The shuffle would sit inside the op's loop group and would have to
        # take over the advance of the read it copies; that is not implemented.
        if getattr(op, "loop_info", None) is not None:
            raise Unsupported(
                f"EA shuffle of {to_shuffle} for {op.get_name()} inside a "
                f"coarse-tiled loop is not supported"
            )

        name_map = {}
        for name in to_shuffle:
            shuffle_buf = _insert_shuffle_op(
                graph, name, ElementArrangement.STANDARD, op
            )
            name_map[name] = shuffle_buf.get_name()
            logger.info(
                "Inserted EA shuffle %s -> %s for %s",
                name,
                shuffle_buf.get_name(),
                op.get_name(),
            )

        redirect_computed_buffer_reads(
            op,
            name_map,
            graph.operations,
            pass_name=_INSERT_PASS_NAME,
            reason="read EA-shuffled operands",
        )


# ---------------------------------------------------------------------------
# optimize_shuffling
# ---------------------------------------------------------------------------
def _reader_ops(graph: GraphLowering, name: str) -> list:
    """Operations that read buffer ``name`` through a ``MemoryDep``."""
    readers = []
    for op in graph.operations:
        if op.get_name() == name:
            continue
        try:
            reads = op.get_read_writes().reads
        except Exception:  # ops without read/write metadata cannot read ``name``
            continue
        if any(isinstance(d, MemoryDep) and d.name == name for d in reads):
            readers.append(op)
    return readers


def _can_redirect_readers(readers: list, new: str) -> bool:
    """True if every reader can be re-pointed to buffer ``new`` in place.

    A reader is redirectable when it is a plain pointwise/reduction
    ``ComputedBuffer`` whose reads can be renamed one-for-one. Mutation
    consumers, scatters, readers inside a coarse-tiled loop (their per-read
    advance metadata is positional), readers carrying a saved direct-read form,
    and readers that already read ``new`` (renaming would merge two reads) are
    left alone, and the shuffle stays.
    """
    for op in readers:
        if not isinstance(op, ComputedBuffer):
            return False
        if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            return False
        if isinstance(op.data, Scatter):
            return False
        if getattr(op, "loop_info", None) is not None:
            return False
        if getattr(op, "_read_copy_elision_record", None) is not None:
            return False
        if new in op.get_read_names():
            return False
    return True


def _redirect_readers(graph: GraphLowering, readers: list, old: str, new: str) -> None:
    """Make every op in ``readers`` read ``new`` instead of ``old``.

    ``new`` must hold the same value as ``old`` (callers check this), so the
    readers' layouts are unchanged. The input is rewired by wrapping
    ``inner_fn`` (never by rebuilding indices).
    """
    for op in readers:
        redirect_computed_buffer_reads(
            op,
            {old: new},
            graph.operations,
            pass_name=_OPTIMIZE_PASS_NAME,
            reason="redirect reads past a removed EA shuffle",
        )


def _remove_op(graph: GraphLowering, op) -> None:
    """Drop ``op`` from the graph the way ``deadcode_elimination`` does."""
    graph.removed_buffers.add(op.get_name())
    graph.operations.remove(op)


def _shuffle_source(op) -> str | None:
    """Name of the buffer a shuffle copy reads, or None if it is not a plain copy."""
    reads = [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]
    return reads[0].name if len(reads) == 1 else None


def _try_drop_shuffle(graph: GraphLowering, op, outputs: set[str]) -> bool:
    """Techniques 1-3: remove a dead, no-op or collapsible shuffle."""
    name = op.get_name()
    src_name = _shuffle_source(op)
    out_stl = _committed_stl(graph, name)
    if src_name is None or out_stl is None:
        return False
    src = graph.get_buffer(src_name)
    src_stl = _committed_stl(graph, src_name)
    readers = _reader_ops(graph, name)

    # 1. Dead shuffle: nothing reads it and it is not a graph output.
    if not readers and name not in outputs:
        _remove_op(graph, op)
        return True

    # 2. No-op shuffle: the source already has the target arrangement and the same
    #    device geometry, so the copy moves nothing.
    if (
        name not in outputs
        and src_stl is not None
        and src_stl == out_stl
        and _can_redirect_readers(readers, src_name)
    ):
        _redirect_readers(graph, readers, name, src_name)
        _remove_op(graph, op)
        return True

    # 3. Shuffle of a shuffle: ``A -S1-> B -S2-> C`` becomes ``A -S2-> C``. If A
    #    already has C's arrangement the pair cancels and C's readers read A.
    if getattr(src, "ea_shuffle", False) and src_name not in outputs:
        a_name = _shuffle_source(src)
        a_stl = _committed_stl(graph, a_name) if a_name else None
        if a_name is not None and a_stl is not None:
            if (
                a_stl == out_stl
                and name not in outputs
                and _can_redirect_readers(readers, a_name)
            ):
                _redirect_readers(graph, readers, name, a_name)
                _remove_op(graph, op)
                return True
            if (
                a_stl.device_size == out_stl.device_size
                and a_stl.stride_map == out_stl.stride_map
                and a_stl.device_dtype == out_stl.device_dtype
                and _can_redirect_readers([op], a_name)
            ):
                _redirect_readers(graph, [op], src_name, a_name)
                return True
    return False


def _try_merge_duplicates(graph: GraphLowering, shuffles: list, outputs: set[str]):
    """Technique 4: keep one shuffle per (source, target layout)."""
    seen: dict = {}
    for op in shuffles:
        src_name, out_stl = _shuffle_source(op), _committed_stl(graph, op.get_name())
        if src_name is None or out_stl is None:
            continue
        key = (
            src_name,
            out_stl.element_arrangement,
            tuple(out_stl.device_size),
            tuple(out_stl.stride_map),
        )
        keep = seen.setdefault(key, op)
        if keep is op or op.get_name() in outputs:
            continue
        readers = _reader_ops(graph, op.get_name())
        if _can_redirect_readers(readers, keep.get_name()):
            _redirect_readers(graph, readers, op.get_name(), keep.get_name())
            _remove_op(graph, op)
            return True
    return False


def _try_sink_shuffle_through_cast(graph: GraphLowering, op, outputs: set[str]) -> bool:
    """Technique 5: shuffle the narrow side of a cast instead of the wide one.

    Rewrites ``x -upcast-> s (staggered) -shuffle-> t (STANDARD)`` into
    ``x -shuffle-> x' -upcast-> t``. The upcast reads ``x'`` in the arrangement
    for which the hardware conversion map already yields STANDARD (see
    ``DtypeOpTable.ea_map``), so the staggering is undone by the cast that is
    executed anyway. The shuffle now moves fp16 (2 B/element) instead of fp32
    (4 B/element), halving its traffic; the cast and its output layout are
    unchanged, so ``t``'s readers need no layout update.
    """
    name = op.get_name()
    s_name = _shuffle_source(op)
    t_layout = _committed_layout(graph, name)
    if s_name is None or t_layout is None or name in outputs:
        return False
    t_stl = t_layout.device_layout
    if t_stl.element_arrangement != ElementArrangement.STANDARD:
        return False
    cast = graph.get_buffer(s_name)
    if (
        not isinstance(cast, ComputedBuffer)
        or not isinstance(cast.data, Pointwise)
        or s_name in outputs
        or getattr(cast, "loop_info", None) is not None
        or getattr(cast, "_read_copy_elision_record", None) is not None
    ):
        return False
    s_stl = _committed_stl(graph, s_name)
    if s_stl is None or s_stl.element_arrangement not in STAGGERED_EAS:
        return False

    # ``cast`` must be a whole-tensor, single-input conversion with no other user.
    rw = cast.get_read_writes()
    reads = [d for d in rw.reads if isinstance(d, MemoryDep)]
    if len(reads) != 1 or reads[0].index != next(iter(rw.writes)).index:
        return False
    if [r.get_name() for r in _reader_ops(graph, s_name)] != [name]:
        return False
    x_name = reads[0].name
    x = graph.get_buffer(x_name)
    x_stl = _committed_stl(graph, x_name)
    if x_stl is None or x_stl.element_arrangement != ElementArrangement.STANDARD:
        return False
    if x.get_dtype() == cast.get_dtype():
        return False

    # Only the fp16 -> fp32 direction narrows the shuffled data.
    if get_device_size_in_bytes(x_stl) >= get_device_size_in_bytes(s_stl):
        return False

    # Find the input arrangement for which the cast itself produces STANDARD.
    e_in = None
    for candidate in STAGGERED_EAS:
        try:
            produced = DtypeOpTable.ea_map(x.get_dtype(), cast.get_dtype(), candidate)
        except Unsupported:
            continue
        if produced == ElementArrangement.STANDARD:
            e_in = candidate
            break
    if e_in is None:
        return False

    readers = _reader_ops(graph, name)
    if not _can_redirect_readers(readers, s_name):
        return False

    # Insert ``x' = shuffle(x, e_in)`` in front of the cast and re-point the cast.
    x_shuffle = _insert_shuffle_op(graph, x_name, e_in, cast)
    new_cast = redirect_computed_buffer_reads(
        cast,
        {x_name: x_shuffle.get_name()},
        graph.operations,
        pass_name=_OPTIMIZE_PASS_NAME,
        reason="read the shuffled narrow input",
    )
    # The cast now yields STANDARD, which is exactly what ``op`` used to hold.
    new_cast.layout = t_layout.unallocated_copy()
    _redirect_readers(graph, readers, name, new_cast.get_name())
    _remove_op(graph, op)
    return True


def optimize_shuffling(graph: GraphLowering) -> None:
    """Minimize the element-arrangement (EA) shuffles inserted by ``insert_ea_shuffle``.

    A shuffle is a full read-and-write copy of a tensor through the device, so it
    is among the most expensive things the pre-scheduling passes can add, and
    ``insert_ea_shuffle`` inserts one greedily for every mixed STANDARD/staggered
    operand pair. This pass looks at all of them together (``_insert_shuffle_op``
    tags each with ``ea_shuffle``) and removes or cheapens what it can. It runs
    right after ``insert_ea_shuffle``, on committed layouts, and repeats until
    nothing changes.

    Implemented techniques, in the order they are tried:

    1. **Dead shuffle.** A shuffle nobody reads and that is not a graph output
       is deleted. Earlier rewrites below can orphan shuffles.
    2. **No-op shuffle.** If the source already has the target arrangement and
       the same device geometry, the copy moves nothing; its readers read the
       source directly.
    3. **Shuffle chains.** ``A -> B -> C`` through two shuffles collapses to one
       shuffle ``A -> C``; if ``A`` already has ``C``'s layout the pair cancels
       and ``C``'s readers read ``A``. This is how a shuffle in front of and
       behind the same value (``STANDARD -> staggered -> STANDARD``) disappears.
    4. **Duplicate shuffles.** Several ops that need the same value in the same
       arrangement (e.g. a weight shuffled once per consumer) share one shuffle.
    5. **Cast sinking (the "bracket" conversion).** Spyre converts between fp16
       and fp32 by staggering the elements, and the conversion map
       (``DtypeOpTable.ea_map``) is a bracket: an fp16 value stored with
       ``FP32_TO_DL16`` up-casts to a STANDARD fp32 value, and a staggered fp32
       down-casts to STANDARD fp16. A cast therefore already undoes a stagger for
       free. ``upcast`` followed by a shuffle back to STANDARD is rewritten as
       "shuffle the fp16 input, then up-cast", which moves half the bytes and
       removes the shuffle from the wide (fp32) tensor. This is the case where a
       shuffle is not needed at all if the surrounding conversions are chosen
       as a bracket.

    A shuffle is only rewritten when all of its readers are plain ``ComputedBuffer``
    ops that can be re-pointed one-for-one (see ``_can_redirect_readers``) and
    when the value is not a graph output. Anything else is left exactly as
    ``insert_ea_shuffle`` produced it.

    Not implemented yet (the rewrites need layout re-propagation or codegen
    support that this pass cannot provide on its own):

    * **Operand flipping for a bracket.** In ``(a_staggered, b_STANDARD) -> op ->
      downcast`` the shuffle can be placed on ``b`` (to the staggered EA) instead
      of ``a`` (to STANDARD). The op's output then stays staggered and the
      closing conversion restores STANDARD, so the bracket is kept and only one
      narrow operand is shuffled. It also changes the EA of every value up to the
      closing cast, whose layouts would have to be recomputed.
    * **Restickify instead of cast plus shuffle.** When an edge needs both a
      stick change and an EA change, a single ``ReStickifyOpHBM`` that also
      writes the target arrangement replaces a restickify followed by a shuffle,
      saving a full pass over the tensor. It needs the restickify codegen to
      encode the EA on its output tensor.
    """
    outputs = set(graph.get_output_names())
    # Every successful rewrite removes at least one operation or one shuffle
    # candidate, so this loop is bounded.
    for _ in range(len(graph.operations) + 1):
        shuffles = [op for op in graph.operations if getattr(op, "ea_shuffle", False)]
        if not shuffles:
            return
        changed = False
        for op in shuffles:
            if _try_drop_shuffle(graph, op, outputs) or _try_sink_shuffle_through_cast(
                graph, op, outputs
            ):
                changed = True
                break
        if not changed:
            changed = _try_merge_duplicates(graph, shuffles, outputs)
        if not changed:
            return
