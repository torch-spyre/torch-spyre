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

"""Hint-driven coarse-tile group construction.

Builds coarse-tile groups from explicit ``dim_hints`` set by
``propagate_hints.py``, for consumption by ``coarse_tile_pre_stickify()``
in ``coarse_tile.py``. Runs PRE-stickification (see ``passes.py``'s
``_maybe_coarse_tile_hints``).
"""

from __future__ import annotations

import logging

import sympy
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation

from ..ir import SpyreConstantFallback
from ..logging_utils import get_inductor_logger
from ..propagate_hints import DimHint

hints_logger = get_inductor_logger("assign_dim_hints")


def _hints_levels(ops: list[Operation]) -> list[tuple]:
    """Build (hint_id, K) level pairs by unioning across all ops.

    All ops in the group share the same hint IDs and split counts.  For each
    hint_id, pick the best DimHint across all ops: one with loop_var is not None
    beats one with loop_var=None.  Hints that are broadcast at every op
    (loop_var=None everywhere) are dropped.  Hints with split_count==1 are
    dropped (tiling by 1 is a no-op).  Returns pairs sorted by hint_id
    ascending (outermost-first).

    is_reduction is intentionally absent from the returned pairs: it is a
    per-op, per-dimension property consulted directly from each op's own
    DimHint by plan_coarse_tile_groups, not a group-level concept.
    """
    best: dict[int, DimHint] = {}
    for op in ops:
        for h in getattr(op, "dim_hints", []):
            prev = best.get(h.hint_id)
            if (
                prev is None
                or prev.loop_var is None
                or (prev.split_count == 1 and h.split_count > 1)
            ):
                best[h.hint_id] = h

    levels = []
    for h in sorted(best.values(), key=lambda x: x.hint_id):
        if h.loop_var is None:
            continue
        if h.split_count == 1:
            hints_logger.debug(
                "spyre_hint on [%s]: hint_id=%d dims=%s split_count=1"
                " — tiling by 1 is a no-op, dropping",
                ", ".join(o.get_name() for o in ops),
                h.hint_id,
                h.dim_names,
            )
            continue
        levels.append((h.hint_id, sympy.Integer(h.split_count)))
    return levels


def _hint_key(op: Operation) -> frozenset | None:
    """Return the frozenset of hint_ids on op, or None if op has no hints."""
    if not isinstance(op, ComputedBuffer):
        return None
    hints = getattr(op, "dim_hints", [])
    return frozenset(h.hint_id for h in hints) if hints else None


def _is_movable_interloper(op: Operation) -> bool:
    """True if op is eligible to be relocated by reorder_unhinted_interlopers.

    Either an unhinted ComputedBuffer, or a seed-allocator fallback
    (SpyreConstantFallback/SpyreEmptyFallback) — a one-time scalar/buffer
    materialization with no per-iteration significance. A general
    FallbackKernel is deliberately excluded: it may carry real data-flow
    side effects (see reorder_unhinted_interlopers's docstring), so it is
    not safe to relocate on the strength of get_read_names()/
    get_mutation_names() alone.
    """
    from ..ir import SpyreEmptyFallback  # deferred: avoids circular import

    return isinstance(op, (ComputedBuffer, SpyreConstantFallback, SpyreEmptyFallback))


def _written_names(op: Operation) -> set[str]:
    """Return all buffer names written by op: its output plus any mutation targets."""
    return {op.get_name()} | set(op.get_mutation_names())


def _no_dep_conflict(op: Operation, others: list[Operation]) -> bool:
    """Return True if moving op past every op in others introduces no data-flow hazard.

    A conflict exists if any op in others reads or mutates a buffer written by op,
    or if op reads or mutates a buffer written by any op in others.

    op_needs intentionally includes op.get_mutation_names() alongside read names.
    This covers both RAW (op reads a buffer that other writes) and WAW (op mutates
    a buffer that other also writes) hazards.  The WAW case is conservative: two
    ops mutating the same buffer cannot be reordered safely regardless of direction,
    so conflating them here is deliberate.
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
    # Defensive: the sole caller (reorder_unhinted_interlopers) already
    # filters for this, but guard here in case of a future context.
    if not _is_movable_interloper(op):
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
    # Defensive: same rationale as _can_move_before.
    if not _is_movable_interloper(op):
        return False
    return _no_dep_conflict(op, ops[start + 1 : end])


def reorder_unhinted_interlopers(graph: GraphLowering) -> None:
    """Move unhinted ComputedBuffer ops that interrupt hint-group runs.

    ``hints_to_coarse_tile_groups`` treats unhinted ops as run-breakers.
    This pass attempts to move each such op either just before the run it
    splits or just after the last same-key op in the remainder, so the run
    becomes contiguous.

    Algorithm — two-cursor scan over ops:

    Outer cursor i: start of the next candidate run.  Advances to j when
    the inner loop exits.

    Inner cursor j: walks forward from i+1 building the run.  For each
    op at ops[j]:
      - Same hint key → absorb into run; j += 1.
      - Unhinted ComputedBuffer, or a seed-allocator fallback
        (SpyreConstantFallback/SpyreEmptyFallback — one-time scalar/buffer
        materialization with no per-iteration significance, e.g. the
        torch.zeros(...) that seeds an online-softmax recurrence) →
        interloper; try to relocate (see below).
      - Any other non-ComputedBuffer op (e.g. a general FallbackKernel,
        which may carry real data-flow side effects) or differently-hinted
        op → hard stop; break.
      - Interloper → one of three outcomes:
          (a) Move before: insert at run_start, run_start += 1, j stays
              (the rotate shifts subsequent ops left so ops[j] is fresh).
          (b) Move after: pop(j), insert at run_end-1, j stays.
          (c) Neither legal → RuntimeError.
        run_end is the index one past the *last* same-key op in ops[j+1:],
        found by scanning backward.  Using the last op (not just the next)
        ensures the move-after target span covers the full remaining run,
        which matters when interlopers further right would otherwise still
        split the run.

    When the inner loop exits, j points to the first op that could not be
    absorbed — a hard-stop or end-of-list.  Advancing i to j (not i+1)
    is correct because everything before j has already been processed.

    A move is legal when it introduces no new data-flow violation:
    no op in the skipped range reads or mutates the moved op's written
    buffers, and the moved op reads or mutates no buffer written in the
    skipped range.

    When both directions are legal the op is moved before the run (closer
    to its original position).

    Raises RuntimeError if an interloper cannot be moved in either
    direction (data-flow dependencies anchor it between hinted ops that
    share the same hint key).
    """
    ops = graph.operations
    i = 0
    while i < len(ops):
        op = ops[i]
        key = _hint_key(op)
        if key is None:
            i += 1
            continue

        run_start = i
        j = i + 1
        while j < len(ops):
            candidate = ops[j]
            ckey = _hint_key(candidate)
            if ckey == key:
                j += 1
                continue
            if not _is_movable_interloper(candidate) or ckey is not None:
                break
            # candidate is an unhinted ComputedBuffer, or a seed-allocator
            # fallback, interrupting the run.
            # Scan backward for the last same-key op; run_end is one past it.
            # O(n) per interloper → O(n²) overall; acceptable for small graphs.
            run_end = None
            for k in range(len(ops) - 1, j, -1):
                if _hint_key(ops[k]) == key:
                    run_end = k + 1
                    break
            # No same-key op exists after j: trailing consumer, not an
            # interloper — end the run silently.
            if run_end is None:
                break
            if _can_move_before(candidate, ops, run_start, j):
                ops.insert(run_start, ops.pop(j))
                run_start += 1  # skip past the op we just inserted before the run
                continue
            if _can_move_after(candidate, ops, j, run_end):
                # pop(j) shifts everything after j left by one, so the last
                # same-key op (formerly run_end-1) is now at run_end-2.
                # Insert at run_end-1 to land just after that last hinted op.
                ops.insert(run_end - 1, ops.pop(j))
                continue
            run_ops = [ops[k].get_name() for k in range(run_start, j)]
            raise RuntimeError(
                f"Cannot reorder unhinted op '{candidate.get_name()}': "
                f"data-flow deps prevent moving it before or after the "
                f"hint-group run [{', '.join(run_ops)}] "
                f"(hint_ids={sorted(key)})"
            )
        i = j


def hints_to_coarse_tile_groups(graph: GraphLowering) -> list[tuple]:
    """Build coarse_tile_pre_stickify() groups from op.dim_hints (set by
    assign_dim_hints).

    coarse_tile_pre_stickify() requires ops to be grouped: all ops in a
    group share the same tiling spec and are tiled together inside the
    same loop nest.  We walk
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
        for g_idx, (group_ops, _group_levels) in enumerate(groups):
            # Collect all DimHints across the group, keyed by hint_id.
            # Prefer a hint whose loop_var is not None (op actually iterates
            # that dim) over a broadcast hint (loop_var=None), so that the
            # representative name/count reflects a real iteration.
            best: dict[int, "DimHint"] = {}
            for gop in group_ops:
                for h in getattr(gop, "dim_hints", []):
                    if h.hint_id not in best or best[h.hint_id].loop_var is None:
                        best[h.hint_id] = h
            descs = [
                f"hint_{h.hint_id}={{'tiles': {{"
                + ", ".join(f"'{n}': {h.split_count}" for n in h.dim_names)
                + "}}"
                for h in sorted(best.values(), key=lambda x: x.hint_id)
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
