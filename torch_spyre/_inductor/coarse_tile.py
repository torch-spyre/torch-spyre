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

"""Coarse-tiling IR pass: stamp loop_group_id / loop_count on ir.Operation objects.

Each group of operations is wrapped in one or more nested counted loops.  For
every operation in the group the iteration ranges divided by each loop's trip
count are scaled down by that factor; the resulting (smaller) per-iteration
ranges are what the downstream scheduler and work-division passes will see.

A ``loop_group_id`` tuple encodes the nesting path:
  - ``(g,)``       — outermost loop group with index ``g``
  - ``(g, h)``     — inner loop group ``h`` nested inside outer group ``g``
  - etc.

``loop_count`` is a *list* of trip counts, one per nesting level from outermost
to innermost.  For a single-level group this is a 1-element list ``[K]``.
``loop_tiled_dims`` is a *list of lists*, one sub-list per nesting level.

Entry point::

    groups = hints_to_coarse_tile_groups(operations)
    coarse_tile(operations, groups)

``groups`` is a list of ``(ops, levels)`` tuples where ``levels`` is a list of
``(hint_id, count, is_reduction_level)`` triples, outermost first.  Each op
resolves its own tiled dimension from its ``loop_var`` in ``dim_hints``.

Each ``ops`` list must be a contiguous sub-sequence of ``operations``.

After stamping, ``coarse_tile`` calls ``insert_tiling_propagation`` to allocate
full-sized output buffers and insert copy/mutation ops for Pointwise operations
whose results are consumed outside the loop.
"""

from __future__ import annotations


import logging
import sympy
from sympy import Expr

import torch
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    Layout,
    Loops,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .constants import BATCH_MATMUL_OP
from .logging_utils import get_inductor_logger
from .loop_info import CoarseTileInfo
from .propagate_hints import get_op_hints
from .pass_utils import op_out_coords

logger = get_inductor_logger("coarse_tile")
hints_logger = get_inductor_logger("assign_dim_hints")


# ---------------------------------------------------------------------------
# Hint-driven group construction
# ---------------------------------------------------------------------------


def _loop_var_to_ranges_pos(out_coords: list, sym: sympy.Symbol) -> int | None:
    """Return the position of loop variable sym in op.data.ranges, or None.

    Looks up sym in the op's output coordinates — the only reliable mapping
    from a loop variable symbol to its data.ranges position, since dep var
    numbering skips size-1 dims while data.ranges does not.
    """
    for i, coord in enumerate(out_coords):
        if len(coord.free_symbols) == 1 and next(iter(coord.free_symbols)) == sym:
            return i
    return None


def _hints_levels(ops: list[Operation]) -> list[tuple]:
    """Build (hint_id, K, is_reduction) level triples from the first hinted op.

    All ops in the group share the same hint IDs and split counts.  Any op
    with a non-None loop_var is representative.  Each op reads its own
    loop_var from dim_hints in _stamp_group.

    Returns a list of (hint_id, count, is_reduction_level) triples, outermost
    first.  Previously this skipped is_reduction hints; it now includes them so
    that _stamp_group can divide reduction_ranges for reduction-dim tiling.
    """
    for op in ops:
        levels = [
            (h.hint_id, sympy.Integer(h.split_count), h.is_reduction)
            for h in getattr(op, "dim_hints", [])
            if h.loop_var is not None
        ]
        if levels:
            return levels
    return []


def _hint_key(op: Operation) -> frozenset | None:
    """Return the frozenset of hint_ids on op, or None if op has no hints."""
    if not isinstance(op, ComputedBuffer):
        return None
    hints = getattr(op, "dim_hints", [])
    return frozenset(h.hint_id for h in hints) if hints else None


def _written_names(op: ComputedBuffer) -> set[str]:
    """Return all buffer names written by op: its output plus any mutation targets."""
    return {op.get_name()} | set(op.get_mutation_names())


def _no_dep_conflict(op: ComputedBuffer, others: list[Operation]) -> bool:
    """Return True if moving op past every op in others introduces no data-flow hazard.

    A conflict exists if any op in others reads or mutates a buffer written by op,
    or if op reads or mutates a buffer written by any op in others.
    """
    op_written = _written_names(op)
    op_needs = op.get_read_names() | set(op.get_mutation_names())
    for other in others:
        if not isinstance(other, ComputedBuffer):
            continue
        if op_written & other.get_read_names():
            return False
        if _written_names(other) & op_needs:
            return False
    return True


def _can_move_before(
    op: Operation,
    ops: list[Operation],
    start: int,
    end: int,
) -> bool:
    """Return True if op (at ops[end]) can move to just before ops[start].

    Legal iff no data-flow conflict exists between op and ops[start..end-1].
    """
    if not isinstance(op, ComputedBuffer):
        return False
    return _no_dep_conflict(op, ops[start:end])


def _can_move_after(
    op: Operation,
    ops: list[Operation],
    start: int,
    end: int,
) -> bool:
    """Return True if op (at ops[start]) can move to just after ops[end-1].

    Legal iff no data-flow conflict exists between op and ops[start+1..end-1].
    """
    if not isinstance(op, ComputedBuffer):
        return False
    return _no_dep_conflict(op, ops[start + 1 : end])


def reorder_unhinted_interlopers(graph: GraphLowering) -> None:
    """Move unhinted ComputedBuffer ops that interrupt hint-group runs.

    ``hints_to_coarse_tile_groups`` breaks a run whenever it encounters a
    ComputedBuffer with no hints (or a different hint key).  If such an op
    could legally be repositioned — either just before the current run's
    start or just after the last same-key op in the remainder — move it so
    the run remains unbroken.

    A move is legal when it introduces no new data-flow violation:
    no op in the skipped range reads or mutates the moved op's written
    buffers, and the moved op reads or mutates no buffer written in the
    skipped range.

    When both directions are legal the op is moved before the run (closer
    to its original position).

    Raises ``RuntimeError`` if an interloper cannot be moved in either
    direction (data-flow dependencies anchor it between hinted ops that share
    the same hint key).
    """
    ops = graph.operations
    i = 0
    while i < len(ops):
        op = ops[i]
        key = _hint_key(op)
        if key is None:
            i += 1
            continue

        # Collect the maximal run of ComputedBuffer ops sharing key,
        # attempting to reorder unhinted interlopers out of the way.
        run_start = i
        j = i + 1
        while j < len(ops):
            candidate = ops[j]
            ckey = _hint_key(candidate)
            if ckey == key:
                j += 1
                continue
            # candidate does not belong to the run.
            if not isinstance(candidate, ComputedBuffer) or ckey is not None:
                # Non-ComputedBuffer or differently-hinted op — cannot reorder.
                break
            # candidate is an unhinted ComputedBuffer interloper.
            # Find the last same-key op anywhere after j (the full move-after
            # target span).  run_end is one past that op so _can_move_after
            # validates the entire range candidate would skip over.
            run_end = None
            for k in range(len(ops) - 1, j, -1):
                if _hint_key(ops[k]) == key:
                    run_end = k + 1
                    break
            # If no hinted op with this key exists anywhere after j the
            # candidate is a trailing consumer, not a true interloper — end
            # the run silently.
            if run_end is None:
                break
            # Hinted ops follow: candidate genuinely splits the run.  Try to
            # move it before the run start first (preferred: stays closer to
            # its original position).
            if _can_move_before(candidate, ops, run_start, j):
                ops.insert(run_start, ops.pop(j))
                run_start += 1
                # j is unchanged: after the rotate, ops[j] is the next op.
                continue
            if _can_move_after(candidate, ops, j, run_end):
                # pop(j) shifts everything after j left by one, so the last
                # same-key op (formerly run_end-1) is now at run_end-2.
                # Insert at run_end-1 to land just after that last hinted op.
                ops.insert(run_end - 1, ops.pop(j))
                # j is unchanged: ops[j] is now the next unexamined op.
                continue
            # The interloper cannot be moved in either direction.
            run_ops = [ops[k].get_name() for k in range(run_start, j)]
            raise RuntimeError(
                f"Cannot reorder unhinted op '{candidate.get_name()}': "
                f"data-flow deps prevent moving it before or after the "
                f"hint-group run [{', '.join(run_ops)}] "
                f"(hint_ids={sorted(key)})"
            )
        i = j


def hints_to_coarse_tile_groups(graph: GraphLowering) -> list[tuple]:
    """Build coarse_tile() groups from op.dim_hints (set by assign_dim_hints).

    coarse_tile() requires ops to be grouped: all ops in a group share the same
    tiling spec and are tiled together inside the same loop nest.  We walk
    operations in topological order and collect consecutive ops that carry
    identical hints into one group, breaking whenever the hint changes or an
    op has no hint at all.
    """

    def _flush(groups, current_ops, current_key):
        if current_ops and current_key is not None:
            levels = _hints_levels(current_ops)
            if levels:
                groups.append((current_ops, levels))
            else:
                hints_logger.warning(
                    "spyre_hint on [%s]: no op iterates over the hinted dimension "
                    "— hint ignored",
                    ", ".join(o.get_name() for o in current_ops),
                )

    groups: list[tuple] = []
    current_ops: list[Operation] = []
    current_key = None

    operations = graph.operations
    for op in operations:
        key = _hint_key(op)

        if key is not None and key == current_key:
            current_ops.append(op)
        else:
            _flush(groups, current_ops, current_key)
            current_ops = [op] if key is not None else []
            current_key = key

    _flush(groups, current_ops, current_key)

    if hints_logger.isEnabledFor(logging.INFO):
        # Build an interleaved view: walk operations in order, emit group boundaries
        # and ungrouped ops so the reader can see what breaks each consecutive run.
        grouped_to_group_idx = {id(o): i for i, g in enumerate(groups) for o in g[0]}
        # Pre-compute hint descriptions per group — get_op_hints is called once per
        # group rather than once per op in the group.
        group_hint_descs: dict[int, str] = {}
        for g_idx, (group_ops, group_levels) in enumerate(groups):
            spec_op = group_ops[0]
            op_hints = get_op_hints(spec_op)
            descs = [
                f"hint_{hint_id}={op_hints[hint_id]}"
                for hint_id, *_ in group_levels
                if hint_id in op_hints
            ]
            group_hint_descs[g_idx] = ", ".join(descs)

        summary_lines = [f"coarse_tile_groups: {len(groups)} group(s) formed"]
        pending_ungrouped: list[str] = []
        last_group_idx: int | None = None
        for o in operations:
            if not isinstance(o, ComputedBuffer):
                continue
            op_group_idx = grouped_to_group_idx.get(id(o))
            if op_group_idx is None:
                hints = getattr(o, "dim_hints", [])
                if hints:
                    ids = sorted({h.hint_id for h in hints})
                    reason = f"hint_ids={ids}"
                else:
                    reason = "no hints"
                pending_ungrouped.append(f"{o.get_name()}({reason})")
            else:
                if op_group_idx != last_group_idx:
                    if pending_ungrouped:
                        summary_lines.append(
                            f"  ungrouped: [{', '.join(pending_ungrouped)}]"
                        )
                        pending_ungrouped = []
                    summary_lines.append(
                        f"  group {op_group_idx} scopes=[{group_hint_descs[op_group_idx]}]:"
                    )
                    last_group_idx = op_group_idx
                # Per-op tiling info.
                tiling_dims = [
                    f"{h.dim_names[0] if h.dim_names else '?'}x{h.split_count}"
                    for h in getattr(o, "dim_hints", [])
                    if h.loop_var is not None and not h.is_reduction
                ]
                aten_ops = [
                    str(n.target)
                    for n in getattr(o, "origins", [])
                    if hasattr(n, "target")
                ]
                summary_lines.append(
                    f"      {o.get_name()}  aten={aten_ops}"
                    + (f"  tiles={tiling_dims}" if tiling_dims else "  (no tiled dims)")
                )
        if pending_ungrouped:
            summary_lines.append(f"  ungrouped: [{', '.join(pending_ungrouped)}]")
        hints_logger.info("%s", "\n".join(summary_lines))

    return groups


def _cache_key(cached_method: object) -> str:
    """Return the cache attribute name used by a cache_on_self / cache_on_self_and_args method.

    cache_on_self uses key ``f"__{fn.__name__}_cache"``; cache_on_self_and_args uses
    ``f"__{class_name}_{fn.__name__}_cache"``.  Both patterns are captured as the
    ``key`` free variable in the method's ``.clear_cache`` closure — extract it once
    at module load so misspellings or upstream renames fail loudly on import.
    """
    clear_fn = getattr(cached_method, "clear_cache")  # AttributeError if absent
    for i, name in enumerate(clear_fn.__code__.co_freevars):
        if name == "key":
            return clear_fn.__closure__[i].cell_contents
    raise AttributeError(
        f"Cannot find 'key' in clear_cache closure of {cached_method!r}"
    )


# Resolve cache keys once at import time — any rename in upstream IR will raise
# AttributeError here rather than silently no-oping at runtime.
_LOOPS_FREE_SYMS_KEY = _cache_key(Loops.get_free_symbol_uses)
_LOOPS_INNER_FN_STR_KEY = _cache_key(Loops.inner_fn_str)
_LOOPS_INNER_FN_OPCOUNT_KEY = _cache_key(Loops.inner_fn_opcount)
_REDUCTION_FREE_SYMS_KEY = _cache_key(Reduction.get_free_symbol_uses)
_LAYOUT_FREE_SYMS_KEY = _cache_key(Layout.get_free_symbol_uses)
_COMPUTED_BUF_FREE_SYMS_KEY = _cache_key(ComputedBuffer.get_free_symbol_uses)
_COMPUTED_BUF_SIZES_KEY = _cache_key(ComputedBuffer.get_default_sizes_body)


def _clear_cache(obj: object, key: str) -> None:
    # cache_on_self/cache_on_self_and_args store results via object.__setattr__ to
    # bypass frozen-dataclass guards (Loops, Reduction, Layout); clearing must also
    # use object.__delattr__ — plain delattr() raises FrozenInstanceError.
    if hasattr(obj, key):
        object.__delattr__(obj, key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def coarse_tile(
    graph: GraphLowering,
    groups: list[tuple],
) -> None:
    """Stamp loop_group_id / loop_count on operations and scale their ranges.

    Parameters
    ----------
    operations:
        The full ordered list of IR operations (as seen by
        CustomPreSchedulingPasses).  Modified in-place when
        insert_tiling_propagation inserts new buffer/copy ops.
    groups:
        Sequence of ``(ops, levels)`` tuples produced by
        ``hints_to_coarse_tile_groups``.  ``levels`` is a list of
        ``(hint_id, count, is_reduction_level)`` triples, outermost first.
    """
    operations = graph.operations
    op_to_position: dict[str, int] = {
        op.get_operation_name(): i for i, op in enumerate(operations)
    }

    for group_idx, (group_ops, levels) in enumerate(groups):
        group_id: tuple[int, ...] = (group_idx,)
        _stamp_group(group_ops, group_id, levels, op_to_position)

    insert_tiling_propagation(operations, groups)


# ---------------------------------------------------------------------------
# Buffer propagation pass
# ---------------------------------------------------------------------------


def insert_tiling_propagation(
    operations: list[Operation],
    groups: list[tuple],
) -> None:
    """Insert full-sized buffers and copy/mutation ops for tiled ops.

    Handles Pointwise and Reduction ComputedBuffers.  For Reductions, tiled
    dims that fall in the reduction_ranges index range raise RuntimeError.

    For each eligible ComputedBuffer in a tiling group, if its result is
    consumed by any operation outside the loop (different loop_group_id or
    absent) or is a graph output, this pass ensures the outside consumer sees
    the complete result by one of two strategies:

    Case 1 — output used both inside and outside the loop:
        Allocate a full-sized buffer.  Insert a copy op (same loop_group_id,
        same loop_tiled_dims) that writes each tile into the correct slice of
        the full buffer.  Patch outside consumers to read the full buffer.

    Case 2 — output used only outside the loop:
        Allocate a full-sized buffer.  Rewire the tiled op to write directly
        into the full buffer via MutationLayoutSHOULDREMOVE.  Patch outside
        consumers to read the full buffer.

    In both cases the existing tiled_symbols / affine.apply machinery in
    SpyreKernel and bundle.py handles the per-iteration address offset.
    """
    for group_ops, _ in groups:
        for op in group_ops:
            if not isinstance(op, ComputedBuffer):
                continue
            if not isinstance(op.data, (Pointwise, Reduction)):
                continue
            _propagate_tiled_op(op, operations)


def _reduction_tiling_is_on_stick_dim(op: ComputedBuffer, red_dim_idx: int) -> bool:
    """Return True if red_dim_idx in reduction_ranges corresponds to the stick dim.

    Uses device_coordinates to find the within-stick coordinate for the primary
    input, then checks whether the reduction symbol for red_dim_idx appears in
    that coordinate's free symbols — the same technique used in propagate_layouts.
    """
    from .ir import FixedTiledLayout
    from .pass_utils import device_coordinates

    data = op.data
    assert isinstance(data, Reduction)
    try:
        rw = op.get_read_writes()
        out_dep = next(iter(rw.writes))
    except (StopIteration, AttributeError, TypeError):
        # StopIteration: mocked ops in unit tests have empty rw.writes.
        # AttributeError/TypeError: guard against partially constructed mocks.
        return False
    out_syms = set(out_dep.index.free_symbols)
    in_dep = next((d for d in rw.reads if hasattr(d, "index")), None)
    if in_dep is None:
        return False
    # reduction_syms: symbols in in_dep.ranges that are absent from the output index,
    # in dep.ranges order (which matches reduction_ranges order).
    reduction_syms = [s for s in in_dep.ranges if s not in out_syms]
    if red_dim_idx >= len(reduction_syms):
        return False
    red_sym = reduction_syms[red_dim_idx]

    in_buf = V.graph.get_buffer(in_dep.name)
    if in_buf is None or not isinstance(in_buf.layout, FixedTiledLayout):
        return False
    # device_coordinates[-1] is the within-stick coordinate expression.
    # If red_sym appears in its free symbols, the reduction is on the stick dim.
    stick_coord = device_coordinates(in_buf.layout.device_layout, in_dep, None)[-1]
    return red_sym in stick_coord.free_symbols


def _validate_reduction_tiling(op: ComputedBuffer) -> None:
    """Raise RuntimeError for Reduction tiling configurations not yet implemented.

    Supported:
      - A single level that tiles only a non-stick reduction dim.
      - A single level that tiles the K (reduction) dim of a BATCH_MATMUL_OP.
        K is the stick dim for operand x, but each tile's output is a full
        [M, N] matrix so no partial-stick sparsity occurs.
      - Multiple nesting levels where outer level(s) tile output dims and the
        innermost level tiles a reduction dim (e.g. outer M + inner K for mm).

    Deferred (raises RuntimeError):
      - Reduction tiling on the stick dimension (except BATCH_MATMUL_OP above).
      - Mixed output+reduction tiling at the same nesting level.
      - Multiple reduction range indices tiled at one level.
    """
    data = op.data
    assert isinstance(data, Reduction)
    loop_info = getattr(op, "loop_info", None)
    if loop_info is None:
        return

    tiled_dims = loop_info.loop_tiled_dims
    tiled_rdims = getattr(loop_info, "loop_tiled_reduction_dims", [])

    # Pad both lists to the same length so zip covers all levels.
    n = max(len(tiled_dims), len(tiled_rdims))
    tiled_dims_padded = tiled_dims + [[]] * (n - len(tiled_dims))
    tiled_rdims_padded = tiled_rdims + [[]] * (n - len(tiled_rdims))

    for i, (out_dims, red_dims) in enumerate(
        zip(tiled_dims_padded, tiled_rdims_padded)
    ):
        if out_dims and red_dims:
            raise RuntimeError(
                f"coarse_tile: op {op.get_name()!r} level {i} tiles both "
                f"output dim(s) {out_dims} and reduction dim(s) {red_dims} "
                "simultaneously (mixed output+reduction tiling at one level "
                "is not yet implemented — Stage 2)."
            )
        if len(red_dims) > 1:
            raise RuntimeError(
                f"coarse_tile: op {op.get_name()!r} level {i} tiles multiple "
                f"reduction dims {red_dims} (tiling more than one reduction "
                "dim per level is not yet implemented — Stage 2)."
            )
        for red_dim_idx in red_dims:
            if (
                data.reduction_type != BATCH_MATMUL_OP
                and _reduction_tiling_is_on_stick_dim(op, red_dim_idx)
            ):
                raise RuntimeError(
                    f"coarse_tile: op {op.get_name()!r} level {i} tiles "
                    f"reduction dim {red_dim_idx} which is the stick dimension "
                    "of the primary input (stick-dim reduction tiling is not "
                    "yet implemented — Stage 2)."
                )


def _propagate_tiled_op(
    op: ComputedBuffer,
    operations: list[Operation],
) -> None:
    """Handle buffer propagation for a single tiled Pointwise or Reduction op."""
    loop_info = getattr(op, "loop_info", None)
    if isinstance(op.data, Reduction):
        _validate_reduction_tiling(op)
        has_tiled_reduction = loop_info is not None and any(
            dims for dims in getattr(loop_info, "loop_tiled_reduction_dims", [])
        )
        if has_tiled_reduction:
            _propagate_tiled_reduction_op(op, operations)
            return

    if loop_info is None:
        return
    loop_group_id = loop_info.loop_group_id

    buf_name = op.get_name()
    outside_consumers, is_graph_output = _find_outside_consumers(
        buf_name, loop_group_id, operations
    )

    # If no dims were tiled (loop_tiled_dims all empty), the op is loop-invariant —
    # mark per_tile_fixed so the unroller reuses the same address each tile.
    if all(not dims for dims in loop_info.loop_tiled_dims):
        from .ir import FixedTiledLayout

        if isinstance(op.layout, FixedTiledLayout):
            op.layout.per_tile_fixed = True
        return

    if not outside_consumers and not is_graph_output:
        # Loop-internal: the buffer is a per-tile scratch region reused every
        # iteration.  Mark it so the unroller does not advance its base address.
        from .ir import FixedTiledLayout

        if isinstance(op.layout, FixedTiledLayout):
            op.layout.per_tile_fixed = True
        # Non-FixedTiledLayout buffers (e.g. MutationLayoutSHOULDREMOVE from a
        # prior pass) are intentionally left unmarked — their addressing is
        # handled by the layout type itself, not by the unroller.
        return

    # Reconstruct the original (pre-division) ranges.
    full_ranges = _compute_full_ranges(op)

    # Insert the full buffer before the first op in the same outermost loop group
    # so it doesn't split the group's contiguous run in the operations list.
    outer_key = loop_group_id[0]
    group_start_idx = next(
        i
        for i, o in enumerate(operations)
        if isinstance(o, ComputedBuffer)
        and getattr(getattr(o, "loop_info", None), "loop_group_id", (None,))[0]
        == outer_key
    )
    full_buf = _allocate_full_buffer(op, full_ranges, operations, group_start_idx)

    has_inside = _has_inside_consumers(buf_name, loop_group_id, operations)

    if has_inside:
        # Case 1: keep tiled op writing to small buffer; insert copy op.
        _insert_copy_op(op, full_buf, operations)
    else:
        # Case 2: tiled op has no inside consumers — rewire it to write directly
        # into the full-size buffer.  Note: MutationLayoutSHOULDREMOVE is
        # incompatible with lx_planning (scratchpad); do not combine the two.
        op.layout = MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(full_buf)))

    # Patch outside consumers and graph outputs to read full_buf.
    full_name = full_buf.get_name()
    _patch_consumers(outside_consumers, buf_name, full_name, operations)
    if is_graph_output:
        _patch_graph_outputs(buf_name, full_buf)

    logger.debug(
        "coarse_tile: propagated %s → %s (case %s)",
        buf_name,
        full_name,
        "1 (copy)" if has_inside else "2 (mutation)",
    )


# ---------------------------------------------------------------------------
# Consumer analysis
# ---------------------------------------------------------------------------


def _reads_buffer(op: ComputedBuffer, buf_name: str) -> bool:
    """Return True if op reads buf_name."""
    try:
        rw = op.get_read_writes()
    except Exception as e:
        logger.debug(
            "_reads_buffer: get_read_writes() raised for %s: %s", op.get_name(), e
        )
        return False
    return any(getattr(dep, "name", None) == buf_name for dep in rw.reads)


def _find_outside_consumers(
    buf_name: str,
    group_loop_id: tuple,
    operations: list[Operation],
) -> tuple[list[ComputedBuffer], bool]:
    """Return (consumer_ops, is_graph_output).

    consumer_ops: ComputedBuffers in operations that read buf_name and are
                  NOT in the same outermost loop group (loop_group_id[0]
                  differs or is absent).
    is_graph_output: True if buf_name appears in graph output names.
    """
    outer_key = group_loop_id[0]
    consumers: list[ComputedBuffer] = []
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        if not _reads_buffer(op, buf_name):
            continue
        li = getattr(op, "loop_info", None)
        if li is None or li.loop_group_id[0] != outer_key:
            consumers.append(op)

    is_graph_output = buf_name in _graph_output_names()
    return consumers, is_graph_output


def _has_inside_consumers(
    buf_name: str,
    group_loop_id: tuple,
    operations: list[Operation],
) -> bool:
    """Return True if any op inside the same outermost loop group reads buf_name."""
    outer_key = group_loop_id[0]
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        li = getattr(op, "loop_info", None)
        if li is None or li.loop_group_id[0] != outer_key:
            continue
        if _reads_buffer(op, buf_name):
            return True
    return False


def _graph_output_names() -> set[str]:
    """Return the set of buffer names that appear in V.graph graph outputs."""
    try:
        return set(V.graph.get_output_names())
    except Exception as e:
        logger.debug("_graph_output_names: V.graph.get_output_names() raised: %s", e)
        return set()


# ---------------------------------------------------------------------------
# Full-buffer allocation
# ---------------------------------------------------------------------------


def _compute_full_ranges(op: ComputedBuffer) -> list[Expr]:
    """Compute the original (pre-division) iteration ranges of op.

    op.data.ranges holds the already-divided ranges.  Reconstruct the full
    ranges by multiplying each tiled dimension back by its loop_count.
    """
    full_ranges = list(op.data.ranges)
    loop_count: list[Expr] = op.loop_info.loop_count
    loop_tiled_dims: list[list[int]] = op.loop_info.loop_tiled_dims
    for count, dims in zip(loop_count, loop_tiled_dims):
        for d in dims:
            if 0 <= d < len(full_ranges):
                full_ranges[d] = sympy.simplify(full_ranges[d] * count)
    return full_ranges


def _allocate_full_buffer(
    tiled_op: ComputedBuffer,
    full_ranges: list[Expr],
    operations: list[Operation],
    insert_at_idx: int,
) -> ComputedBuffer:
    """Allocate a full-sized HBM buffer for the tiled op's original shape.

    Creates a spyre.empty FX node, lowers it via V.graph.run_node(), assigns
    a FixedTiledLayout matching tiled_op's layout, splices it into operations
    at insert_at_idx, and returns the new ComputedBuffer.
    """
    from .propagate_layouts import generic_layout  # deferred: avoids circular import
    from .ir import (
        FixedTiledLayout,
        SpyreEmptyFallback,
    )  # deferred: avoids circular import

    graph_lowering = V.graph
    fx_graph = graph_lowering.graph
    device = tiled_op.get_device()
    dtype = tiled_op.get_dtype()

    # Evaluate full_ranges to concrete ints (they should be integer expressions).
    size = [int(r) for r in full_ranges]

    first_compute = next(n for n in fx_graph.nodes if n.op != "placeholder")
    with fx_graph.inserting_before(first_compute):
        empty_fx = fx_graph.create_node(
            "call_function",
            torch.ops.spyre.empty.default,
            args=(size, device, dtype),
        )
        empty_fx.meta["val"] = torch.empty(size, dtype=dtype, device="cpu")

    empty_tb = graph_lowering.run_node(empty_fx)
    graph_lowering.env[empty_fx] = empty_tb

    full_buf = empty_tb.data.data  # TensorBox → StorageBox → SpyreEmptyFallback
    assert isinstance(full_buf, SpyreEmptyFallback), (
        f"Expected SpyreEmptyFallback, got {type(full_buf).__name__}"
    )
    full_buf.origins = OrderedSet([empty_fx])

    # Assign a FixedTiledLayout with the full size.
    orig_layout = tiled_op.layout
    # Recompute strides for the full size (contiguous row-major).
    strides: list[Expr] = []
    stride: Expr = sympy.Integer(1)
    for s in reversed(full_ranges):
        strides.insert(0, stride)
        stride = stride * s

    if isinstance(orig_layout, FixedTiledLayout):
        # Rebuild SpyreTensorLayout for the full size, preserving the
        # within-stick dimension from the original per-tile layout.
        orig_stl = orig_layout.device_layout
        sm_last = int(list(orig_stl.stride_map)[-1])
        full_strides_ints = [int(s) for s in strides]
        full_size_ints = [int(s) for s in full_ranges]
        within_stick_dim = next(
            (i for i, s in enumerate(full_strides_ints) if s == sm_last), None
        )
        if within_stick_dim is None:
            within_stick_dim = len(full_size_ints) - 1
        ndim = len(full_size_ints)
        dim_order = [i for i in range(ndim) if i != within_stick_dim] + [
            within_stick_dim
        ]
        from torch_spyre._C import SpyreTensorLayout

        device_layout = SpyreTensorLayout(
            full_size_ints, full_strides_ints, dtype, dim_order
        )
    else:
        device_layout = generic_layout(full_buf)

    layout = FixedTiledLayout(
        device,
        dtype,
        list(full_ranges),
        strides,
        device_layout,
    )
    full_buf.layout = layout

    # Splice into operations at the correct position.
    operations.remove(full_buf)
    operations.insert(insert_at_idx, full_buf)

    return full_buf


# ---------------------------------------------------------------------------
# Case 1: copy op insertion
# ---------------------------------------------------------------------------


def _insert_copy_op(
    tiled_op: ComputedBuffer,
    full_buf: ComputedBuffer,
    operations: list[Operation],
) -> None:
    """Insert a copy op after tiled_op that writes each tile into full_buf.

    The copy op carries the same loop metadata as tiled_op so it executes
    inside the same loop body.  Its layout is MutationLayoutSHOULDREMOVE
    pointing at full_buf so store_output writes into full_buf.  Because
    loop_tiled_dims is set, SpyreKernel stamps tiled_symbols on the OpSpec
    and bundle.mlir emits affine.apply for the per-iteration output address.
    """
    copy_data = Pointwise(
        device=tiled_op.get_device(),
        dtype=tiled_op.get_dtype(),
        inner_fn=tiled_op.make_loader(),
        ranges=list(tiled_op.data.ranges),
    )

    copy_name = V.graph.qualify_name(f"coarse_tile_copy_{tiled_op.get_name()}")
    copy_buf = ComputedBuffer(
        name=copy_name,
        layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(full_buf))),
        data=copy_data,
    )
    copy_buf.origins = tiled_op.origins
    copy_buf.operation_name = copy_name

    # Stamp with the same loop metadata so this op is inside the same loop.
    copy_buf.loop_info = tiled_op.loop_info  # type: ignore[attr-defined]

    V.graph.name_to_buffer[copy_name] = copy_buf

    tiled_idx = operations.index(tiled_op)
    operations.insert(tiled_idx + 1, copy_buf)


# ---------------------------------------------------------------------------
# Case: reduction-dim tiling — combine op insertion
# ---------------------------------------------------------------------------


def _insert_combine_op(
    tiled_op: ComputedBuffer,
    accum_buf: ComputedBuffer,
    operations: list[Operation],
) -> None:
    """Insert a pointwise combine op that accumulates tiled_op into accum_buf.

    The combine op reads both the partial result (tiled_op) and the current
    accumulation buffer and writes the combined value back into accum_buf via
    MutationLayoutSHOULDREMOVE.  It carries the same loop_info as tiled_op
    so the scheduler places it inside the same CountedLoopSchedulerNode.
    """
    from torch._inductor.virtualized import ops as vops

    reduction_type = tiled_op.data.reduction_type
    partial_loader = tiled_op.make_loader()
    accum_loader = accum_buf.make_loader()

    def combine_inner_fn(index):
        partial = partial_loader(index)
        accum = accum_loader(index)
        if reduction_type in ("sum", BATCH_MATMUL_OP):
            return vops.add(accum, partial)
        if reduction_type == "xor_sum":
            return vops.bitwise_xor(accum, partial)
        if reduction_type == "prod":
            return vops.mul(accum, partial)
        if reduction_type == "max":
            return vops.maximum(accum, partial)
        if reduction_type == "min":
            return vops.minimum(accum, partial)
        if reduction_type == "any":
            # TODO: add vops.logical_or to SpyreOpFuncs before enabling
            # hardware-level 'any' support — it is currently absent.
            return vops.logical_or(accum, partial)
        raise RuntimeError(
            f"coarse_tile: _insert_combine_op: unsupported reduction_type "
            f"{reduction_type!r}"
        )

    combine_data = Pointwise(
        device=tiled_op.get_device(),
        dtype=tiled_op.get_dtype(),
        inner_fn=combine_inner_fn,
        ranges=list(tiled_op.data.ranges),
    )
    combine_name = V.graph.qualify_name(f"coarse_tile_combine_{tiled_op.get_name()}")
    combine_buf = ComputedBuffer(
        name=combine_name,
        layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(accum_buf))),
        data=combine_data,
    )
    combine_buf.origins = tiled_op.origins
    combine_buf.operation_name = combine_name
    combine_buf.loop_info = tiled_op.loop_info  # type: ignore[attr-defined]
    V.graph.name_to_buffer[combine_name] = combine_buf

    tiled_idx = operations.index(tiled_op)
    operations.insert(tiled_idx + 1, combine_buf)


def _insert_reduction_copy_op(
    tiled_op: ComputedBuffer,
    accum_tile: ComputedBuffer,
    accum_full: ComputedBuffer,
    outer_loop_info: "CoarseTileInfo",
    operations: list[Operation],
) -> None:
    """Insert a copy op that writes accum_tile → accum_full at the outer loop level.

    Reads accum_tile (per_tile_fixed=True, never advances) and writes into
    accum_full via MutationLayoutSHOULDREMOVE.  Carries outer_loop_info so
    the unroller advances accum_full per outer output-dim tile.
    """
    copy_data = Pointwise(
        device=tiled_op.get_device(),
        dtype=tiled_op.get_dtype(),
        inner_fn=accum_tile.make_loader(),
        ranges=list(tiled_op.data.ranges),
    )
    copy_name = V.graph.qualify_name(f"coarse_tile_reduce_copy_{tiled_op.get_name()}")
    copy_buf = ComputedBuffer(
        name=copy_name,
        layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(accum_full))),
        data=copy_data,
    )
    copy_buf.origins = tiled_op.origins
    copy_buf.operation_name = copy_name
    copy_buf.loop_info = outer_loop_info  # type: ignore[attr-defined]
    V.graph.name_to_buffer[copy_name] = copy_buf

    combine_name = V.graph.qualify_name(f"coarse_tile_combine_{tiled_op.get_name()}")
    combine_buf = V.graph.name_to_buffer.get(combine_name)
    if combine_buf is not None and combine_buf in operations:
        insert_idx = operations.index(combine_buf) + 1
    else:
        insert_idx = operations.index(tiled_op) + 1
    operations.insert(insert_idx, copy_buf)


def _compute_fill_loop_info(op: ComputedBuffer) -> "CoarseTileInfo | None":
    """Return the loop_info to stamp on the fill op for a nested tiled reduction.

    For a flat (pure reduction) tiling the fill has no loop_info — it runs
    once before all loops.  Returns None.

    For a nested tiling where outer level(s) tile output dims and the inner
    level tiles a reduction dim, the fill must run inside the outer loop (once
    per outer tile) so the accumulator is per-outer-tile sized.  Returns a
    CoarseTileInfo covering only the outer output-dim levels.
    """
    loop_info = op.loop_info
    tiled_rdims = getattr(loop_info, "loop_tiled_reduction_dims", [])

    outer_counts: list[sympy.Expr] = []
    outer_tiled_dims: list[list[int]] = []
    outer_tiled_rdims: list[list[int]] = []
    for dims, rdims, count in zip(
        loop_info.loop_tiled_dims, tiled_rdims, loop_info.loop_count
    ):
        if dims:  # non-empty output-dim list → this is an output-dim level
            outer_counts.append(count)
            outer_tiled_dims.append(dims)
            outer_tiled_rdims.append([])

    if not outer_counts:
        return None  # flat: fill runs before all loops

    outer_gid = loop_info.loop_group_id[: len(outer_counts)]
    return CoarseTileInfo(
        loop_group_id=outer_gid,
        loop_count=outer_counts,
        loop_tiled_dims=outer_tiled_dims,
        loop_tiled_reduction_dims=outer_tiled_rdims,
    )


def _propagate_tiled_reduction_op(
    op: ComputedBuffer,
    operations: list[Operation],
) -> None:
    """Handle buffer propagation for a Reduction op tiled over a reduction dim.

    Strategy: fill-initialize + per-tile combine.
      1. Allocate a HBM accumulation buffer sized to the full
         (pre-outer-division) output shape (_compute_full_ranges), so that
         address advancement across outer tiles writes each tile into the
         correct slice.  For flat (reduction-only) tiling this equals
         op.data.ranges.
      2. Insert a fill op that writes the reduction's identity value into the
         accumulation buffer.  For flat reduction tiling the fill has no
         loop_info and runs before all loops.  For nested tiling (outer
         output-dim loop + inner reduction loop) the fill carries the outer
         loop's loop_info so it runs inside the outer loop — once per outer
         tile — keeping the accumulator sized to the per-tile output shape.
      3. Insert a combine op (inside the inner loop, same loop_info as the
         tiled reduction op) that merges each tile's partial result into the
         accumulation buffer using the reduction's combining fn.
      4. Mark the tiled reduction op's output as per_tile_fixed (inner-loop
         scratch, not advanced between inner iterations).
      5. Patch outside consumers and graph outputs to read the accumulation
         buffer.
    """
    loop_info = op.loop_info
    loop_group_id = loop_info.loop_group_id
    reduction_type = op.data.reduction_type
    identity = _reduction_identity_value(reduction_type, op.get_dtype())

    # Per-outer-tile output shape (ranges after any outer tiling divided them).
    per_tile_ranges = list(op.data.ranges)

    # Accumulation buffer uses the full (pre-outer-division) output shape so
    # that address advancement across outer output-dim tiles writes each tile's
    # result into the correct slice.  For reduction-dim-only tiling there is no
    # outer division, so full == per-tile.
    full_output_ranges = _compute_full_ranges(op)

    # Insert HBM buffer before the first op in the loop group.
    outer_key = loop_group_id[0]
    group_start_idx = next(
        i
        for i, o in enumerate(operations)
        if isinstance(o, ComputedBuffer)
        and getattr(getattr(o, "loop_info", None), "loop_group_id", (None,))[0]
        == outer_key
    )

    fill_loop_info = _compute_fill_loop_info(op)
    is_nested = fill_loop_info is not None

    if is_nested:
        # Nested case: allocate separate tile-sized and full-sized buffers.
        # accum_tile (per_tile_fixed=True) stays inside the inner K-loop;
        # accum_full accumulates across outer B-tiles via a copy op.
        accum_full = _allocate_full_buffer(
            op, full_output_ranges, operations, group_start_idx
        )
        group_start_idx_after_full = operations.index(accum_full) + 1
        accum_tile = _allocate_full_buffer(
            op, per_tile_ranges, operations, group_start_idx_after_full
        )
        from .ir import FixedTiledLayout

        if isinstance(accum_tile.layout, FixedTiledLayout):
            accum_tile.layout.per_tile_fixed = True
        fill_target = accum_tile
        combine_target = accum_tile
    else:
        # Flat case: single full-sized buffer (unchanged behaviour).
        accum_full = _allocate_full_buffer(
            op, full_output_ranges, operations, group_start_idx
        )
        fill_target = accum_full
        combine_target = accum_full

    # Insert fill op immediately after the fill target buffer allocation
    # (outside the loop for flat, inside the outer loop for nested).
    # Use a SpyreConstantFallback scalar as the fill source so that Spyre's
    # kernel codegen can express this as an IDENTITY_OP broadcast.  We must
    # assign a FixedTiledLayout manually here because finalize_layouts has
    # already run when this pass executes.
    dtype = op.get_dtype()
    device = op.get_device()
    from .ir import (
        FixedTiledLayout,
        SpyreConstantFallback,
    )  # deferred: avoids circular import
    from torch_spyre._C import SpyreTensorLayout  # deferred: avoids circular import

    scalar_op = SpyreConstantFallback(
        torch.ops.spyre.constant.default, float(identity), dtype, device
    )
    # SpyreTensorLayout([], dtype) yields device_size=[1, 64], stride_map=[-1, -1]
    # — a 0-d broadcast scalar in Spyre's device coordinate system.
    scalar_stl = SpyreTensorLayout([], dtype)
    scalar_op.layout = FixedTiledLayout(device, dtype, [], [], scalar_stl)
    scalar_loader = TensorBox.create(scalar_op).make_loader()

    fill_data = Pointwise(
        device=device,
        dtype=dtype,
        inner_fn=lambda index, _loader=scalar_loader: _loader([]),
        ranges=per_tile_ranges,
    )
    fill_name = V.graph.qualify_name(f"coarse_tile_fill_{op.get_name()}")
    fill_buf = ComputedBuffer(
        name=fill_name,
        layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(fill_target))),
        data=fill_data,
    )
    fill_buf.origins = op.origins
    fill_buf.operation_name = fill_name
    if fill_loop_info is not None:
        fill_buf.loop_info = fill_loop_info  # type: ignore[attr-defined]
    # else: no loop_info — fill runs once before all loops (flat reduction case).
    V.graph.name_to_buffer[fill_name] = fill_buf
    fill_target_idx = operations.index(fill_target)
    # scalar_op was appended to graph.operations by register_operation(); move it
    # to just after fill_target, then insert fill_buf after scalar_op.
    operations.remove(scalar_op)
    operations.insert(fill_target_idx + 1, scalar_op)
    operations.insert(fill_target_idx + 2, fill_buf)

    # Insert combine op after the tiled reduction op (inside the loop).
    _insert_combine_op(op, combine_target, operations)

    # For nested case, insert a copy op at the outer loop level that writes
    # accum_tile → accum_full, advancing accum_full across outer output tiles.
    if is_nested:
        assert fill_loop_info is not None  # guaranteed by is_nested == True
        _insert_reduction_copy_op(
            op, accum_tile, accum_full, fill_loop_info, operations
        )

    # Mark tiled op's output as per-tile scratch (no address advance).
    if not isinstance(op.layout, FixedTiledLayout):
        raise RuntimeError(
            f"coarse_tile: tiled reduction op {op.get_name()!r} has layout "
            f"{type(op.layout).__name__}, expected FixedTiledLayout; "
            "cannot mark per_tile_fixed"
        )
    op.layout.per_tile_fixed = True

    # Patch consumers to read accum_full (the fully-assembled output).
    buf_name = op.get_name()
    outside_consumers, is_graph_output = _find_outside_consumers(
        buf_name, loop_group_id, operations
    )
    accum_name = accum_full.get_name()
    _patch_consumers(outside_consumers, buf_name, accum_name, operations)
    if is_graph_output:
        _patch_graph_outputs(buf_name, accum_full)

    logger.debug(
        "coarse_tile: tiled reduction %s → accum_full %s (fill=%s, identity=%s, "
        "nested=%s)",
        buf_name,
        accum_name,
        fill_name,
        identity,
        is_nested,
    )


# ---------------------------------------------------------------------------
# Consumer / graph-output patching
# ---------------------------------------------------------------------------


def _patch_consumers(
    consumers: list[ComputedBuffer],
    old_name: str,
    new_name: str,
    operations: list[Operation],
) -> None:
    """Redirect outside consumers from old_name to new_name.

    Patches each consumer's inner_fn via NameSwapHandler and reconstructs
    the ComputedBuffer to invalidate the sizes cache.
    """
    if not consumers or old_name == new_name:
        return

    from .insert_restickify import NameSwapHandler
    from .pass_utils import replace_computed_buffer_body

    name_map = {old_name: new_name}
    for consumer in consumers:
        orig_inner = consumer.data.inner_fn

        def new_inner_fn(*args, _map=name_map, _orig=orig_inner):
            with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
                return _orig(*args)

        object.__setattr__(consumer.data, "inner_fn", new_inner_fn)
        replace_computed_buffer_body(consumer, consumer.data, operations)
        V.graph.name_to_buffer[consumer.get_name()] = operations[
            next(
                i
                for i, op in enumerate(operations)
                if isinstance(op, ComputedBuffer)
                and op.get_name() == consumer.get_name()
            )
        ]


def _patch_graph_outputs(old_name: str, new_buf: ComputedBuffer) -> None:
    """Replace references to old_name in V.graph.graph_outputs with new_buf."""
    try:
        outputs = V.graph.graph_outputs
    except Exception:
        return

    new_tb = TensorBox(StorageBox(new_buf))
    for i, out in enumerate(outputs):
        # Unwrap StorageBox layers to reach ComputedBuffer without going into
        # the ComputedBuffer's inner data (Pointwise / Reduction).
        candidate = out
        while isinstance(candidate, StorageBox):
            candidate = candidate.data
        if isinstance(candidate, ComputedBuffer) and candidate.get_name() == old_name:
            outputs[i] = new_tb


# ---------------------------------------------------------------------------
# Original stamping helpers (unchanged)
# ---------------------------------------------------------------------------


def _stamp_group(
    ops: list[Operation],
    group_id: tuple[int, ...],
    levels: list[tuple],
    op_to_position: dict[str, int],
) -> None:
    """Stamp loop_group_id / loop_count / loop_tiled_dims and divide ranges.

    ``levels`` is a list of ``(hint_id, count, is_reduction_level)`` triples,
    outermost first.  Each op resolves its own tiled dimension from its
    loop_var in dim_hints.  Ops that have no matching dim for a level are
    loop-invariant at that level.

    For reduction-dim levels (``is_reduction_level=True``), the resolved dim
    index populates ``loop_tiled_reduction_dims`` and ``_divide_reduction_ranges``
    is called instead of ``_divide_ranges``.  End-to-end correctness of this
    path is covered by ``TestCoarseTileReductionDim0E2E`` in
    ``tests/inductor/test_coarse_tile_e2e.py``.
    """
    if not ops:
        return

    _validate_contiguous(ops, op_to_position, group_id)

    nested_group_id: tuple[int, ...] = group_id + (0,) * (len(levels) - 1)
    counts = [count for _, count, _ in levels]

    for op in ops:
        if not isinstance(op, ComputedBuffer):
            logger.debug(
                "coarse_tile: skipping non-ComputedBuffer op %s (%s)",
                op.get_operation_name(),
                type(op).__name__,
            )
            continue

        op_out = op_out_coords(op)

        # Build lookup: hint_id → output-ranges position (non-reduction dims).
        hint_id_to_ranges_pos: dict[int, int] = {
            h.hint_id: pos
            for h in getattr(op, "dim_hints", [])
            if h.loop_var is not None and not h.is_reduction
            if (pos := _loop_var_to_ranges_pos(op_out, h.loop_var)) is not None
        }

        # Build lookup: hint_id → reduction_ranges position (reduction dims).
        hint_id_to_reduction_ranges_pos: dict[int, int] = {}
        if isinstance(op.data, Reduction):
            hint_id_to_reduction_ranges_pos = {
                h.hint_id: pos
                for h in getattr(op, "dim_hints", [])
                if h.loop_var is not None and h.is_reduction
                if (pos := _loop_var_to_reduction_ranges_pos(op, h.loop_var))
                is not None
            }

        op_tiled_dims: list[list[int]] = []
        op_tiled_reduction_dims: list[list[int]] = []
        for hint_id, count, is_reduction_level in levels:
            if is_reduction_level:
                rpos = hint_id_to_reduction_ranges_pos.get(hint_id)
                op_tiled_dims.append([])
                op_tiled_reduction_dims.append([rpos] if rpos is not None else [])
                if isinstance(op.data, Reduction):
                    # NOTE: _divide_reduction_ranges mutates data.reduction_ranges
                    # before _validate_reduction_tiling runs in the later
                    # insert_tiling_propagation pass.  If validation raises (e.g.
                    # stick-dim tiling, Stage 2), the mutated ranges are never
                    # observed: the RuntimeError propagates uncaught through the
                    # pass runner and aborts compilation.
                    _divide_reduction_ranges(
                        op, count, [rpos] if rpos is not None else []
                    )
            else:
                opos = hint_id_to_ranges_pos.get(hint_id)
                op_tiled_dims.append([opos] if opos is not None else [])
                op_tiled_reduction_dims.append([])
                _divide_ranges(op, count, [opos] if opos is not None else [])

        op.loop_info = CoarseTileInfo(  # type: ignore[attr-defined]
            loop_group_id=nested_group_id,
            loop_count=counts,
            loop_tiled_dims=op_tiled_dims,
            loop_tiled_reduction_dims=op_tiled_reduction_dims,
        )

        logger.debug(
            "coarse_tile: stamped %s loop_group_id=%s loop_count=%s "
            "loop_tiled_dims=%s loop_tiled_reduction_dims=%s",
            op.get_operation_name(),
            nested_group_id,
            counts,
            op_tiled_dims,
            op_tiled_reduction_dims,
        )


def _divide_ranges(
    op: ComputedBuffer,
    loop_count: Expr,
    tiled_dims: list[int],
) -> None:
    """Divide the specified iteration ranges of op by loop_count.

    For a ``Pointwise`` the full ranges are op.data.ranges.
    For a ``Reduction`` the non-reduction (outer) ranges are op.data.ranges;
    op.data.reduction_ranges are left untouched.

    ``tiled_dims`` is a list of positional indices into ``data.ranges``.
    All indices must be valid; an out-of-bounds index is a caller bug.

    Also updates ``op.layout.size``, ``op.layout.stride``, and
    ``op.layout.device_layout`` so the layout describes the smaller per-tile
    buffer, not the full tensor.  Contiguous host strides are recomputed from
    the new size; the ``SpyreTensorLayout`` is rebuilt from the new host size
    and strides, preserving the within-stick dimension from the original layout.
    """
    data = op.data
    if not isinstance(data, (Pointwise, Reduction)):
        return

    ranges = list(data.ranges)
    if not ranges:
        return

    for i in tiled_dims:
        assert 0 <= i < len(ranges), (
            f"coarse_tile: op {op.get_name()!r} tiled dim {i} out of bounds "
            f"(ranges has {len(ranges)} entries)"
        )
        r = ranges[i]
        if isinstance(r, (int, sympy.Integer)) and isinstance(
            loop_count, (int, sympy.Integer)
        ):
            if int(r) % int(loop_count) != 0:
                raise RuntimeError(
                    f"coarse_tile: op {op.get_name()!r} loop var d{i} range {r} "
                    f"is not divisible by loop_count {loop_count}.  All tiled "
                    f"dimensions must be evenly divisible by the loop trip count."
                )
            ranges[i] = sympy.Integer(int(r) // int(loop_count))
        else:
            ranges[i] = sympy.sympify(r) / sympy.sympify(loop_count)

    # Loops is a frozen dataclass; use object.__setattr__ to mutate it.
    object.__setattr__(data, "ranges", ranges)

    # Invalidate Loops-level caches that read ranges.
    _clear_cache(data, _LOOPS_FREE_SYMS_KEY)
    _clear_cache(data, _LOOPS_INNER_FN_STR_KEY)
    _clear_cache(data, _LOOPS_INNER_FN_OPCOUNT_KEY)
    if isinstance(data, Reduction):
        _clear_cache(data, _REDUCTION_FREE_SYMS_KEY)

    # Invalidate ComputedBuffer-level caches derived from data.ranges.
    _clear_cache(op, _COMPUTED_BUF_SIZES_KEY)
    _clear_cache(op, _COMPUTED_BUF_FREE_SYMS_KEY)

    # Sync layout.size, layout.stride, and layout.device_layout with the new ranges.
    from torch._inductor.ir import FixedLayout, FlexibleLayout

    from .ir import FixedTiledLayout

    layout = getattr(op, "layout", None)
    if not (isinstance(layout, FixedLayout) and len(layout.size) == len(ranges)):
        return

    new_size = list(layout.size)
    for i in tiled_dims:
        new_size[i] = ranges[i]
    layout.size = new_size

    # Recompute contiguous strides for the smaller buffer.
    layout.stride = list(FlexibleLayout.contiguous_strides(new_size))

    # Invalidate Layout- and ComputedBuffer-level caches that read size/stride.
    _clear_cache(layout, _LAYOUT_FREE_SYMS_KEY)
    _clear_cache(op, _COMPUTED_BUF_FREE_SYMS_KEY)

    # Rebuild SpyreTensorLayout for the new host size, preserving the
    # within-stick dimension.  stride_map[-1] is the element stride of the
    # within-stick host dimension in the original layout; match it against the
    # new contiguous strides to identify which host dim remains the stick dim.
    if not isinstance(layout, FixedTiledLayout):
        return
    orig_stl = layout.device_layout
    sm_last = int(list(orig_stl.stride_map)[-1])
    new_strides_ints = [int(s) for s in layout.stride]
    new_size_ints = [int(s) for s in new_size]
    within_stick_dim = next(
        (i for i, s in enumerate(new_strides_ints) if s == sm_last), None
    )
    if within_stick_dim is None:
        # Fall back to last dim (covers the common contiguous fp16 case where
        # sm_last == 1 and the last stride is also 1).
        within_stick_dim = len(new_size_ints) - 1
    ndim = len(new_size_ints)
    dim_order = [i for i in range(ndim) if i != within_stick_dim] + [within_stick_dim]
    from torch_spyre._C import SpyreTensorLayout

    layout.device_layout = SpyreTensorLayout(
        new_size_ints, new_strides_ints, layout.dtype, dim_order
    )


def _loop_var_to_reduction_ranges_pos(
    op: ComputedBuffer, sym: sympy.Symbol
) -> int | None:
    """Return position of loop variable sym in op.data.reduction_ranges, or None.

    Uses dep-tracking symbols (d0, d1, ...) rather than SymT.R0_INDEX symbols
    (r0_0, r0_1, ...) which are a different namespace.  Finds reduction symbols
    by set-subtracting output index symbols from input index symbols, in
    dep.ranges order (which matches reduction_ranges order).
    """
    assert isinstance(op.data, Reduction)
    rw = op.get_read_writes()
    out_dep = next(iter(rw.writes))
    out_syms = out_dep.index.free_symbols
    in_dep = next(d for d in rw.reads if hasattr(d, "index"))
    reduction_syms = [s for s in in_dep.ranges if s not in out_syms]
    try:
        return reduction_syms.index(sym)
    except ValueError:
        return None


def _divide_reduction_ranges(
    op: ComputedBuffer,
    loop_count: Expr,
    tiled_dims: list[int],
) -> None:
    """Divide the specified reduction_ranges entries of op by loop_count.

    Unlike _divide_ranges, does NOT update op.layout.size/stride — the
    output buffer shape is determined by data.ranges (non-reduction dims)
    and is unchanged by reduction-dim tiling.
    """
    data = op.data
    assert isinstance(data, Reduction)
    if not tiled_dims:
        return
    reduction_ranges = list(data.reduction_ranges)
    for i in tiled_dims:
        assert 0 <= i < len(reduction_ranges), (
            f"coarse_tile: op {op.get_name()!r} tiled reduction dim {i} out of bounds "
            f"(reduction_ranges has {len(reduction_ranges)} entries)"
        )
        r = reduction_ranges[i]
        if isinstance(r, (int, sympy.Integer)) and isinstance(
            loop_count, (int, sympy.Integer)
        ):
            if int(r) % int(loop_count) != 0:
                raise RuntimeError(
                    f"coarse_tile: op {op.get_name()!r} reduction dim {i} range {r} "
                    f"is not divisible by loop_count {loop_count}.  All tiled "
                    f"reduction dimensions must be evenly divisible by the loop trip count."
                )
            reduction_ranges[i] = sympy.Integer(int(r) // int(loop_count))
        else:
            reduction_ranges[i] = sympy.sympify(r) / sympy.sympify(loop_count)
    # Reduction is a frozen dataclass; use object.__setattr__ to mutate it.
    object.__setattr__(data, "reduction_ranges", reduction_ranges)


def _reduction_identity_value(
    reduction_type: str, dtype: "torch.dtype"
) -> "float | int":
    """Return the monoid identity value for the given reduction type.

    Used to initialize the accumulation buffer before a tiled reduction loop.
    """
    if reduction_type in ("sum", "xor_sum", "any", BATCH_MATMUL_OP):
        return 0
    if reduction_type == "prod":
        return 1
    if reduction_type == "max":
        return float("-inf")
    if reduction_type == "min":
        return float("inf")
    raise RuntimeError(
        f"coarse_tile: unsupported reduction_type {reduction_type!r} for tiled "
        "reduction — no identity value is defined for this reduction type."
    )


def _validate_contiguous(
    ops: list[Operation],
    op_to_position: dict[str, int],
    group_id: tuple[int, ...],
) -> None:
    """Assert that ops form a contiguous slice of the operation list.

    A gap indicates a data-flow dependency that crosses the group boundary,
    which would violate the coarse-tiling model.
    """
    positions = []
    for op in ops:
        name = op.get_operation_name()
        if name not in op_to_position:
            raise RuntimeError(
                f"coarse_tile: operation {name!r} (group {group_id}) "
                "is not in the operations list"
            )
        positions.append(op_to_position[name])

    if not positions:
        return

    lo, hi = min(positions), max(positions)
    if hi - lo + 1 != len(ops):
        raise RuntimeError(
            f"coarse_tile: group {group_id} operations are not contiguous "
            f"in the operation list (positions {sorted(positions)}). "
            "A data-flow dependency crosses the group boundary."
        )
