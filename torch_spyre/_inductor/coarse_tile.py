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
every operation in the group the iteration ranges that are divided by each
loop's trip count are scaled down by that factor; the resulting (smaller)
per-iteration ranges are what the downstream scheduler and work-division passes
will see.

A ``loop_group_id`` tuple encodes the nesting path:
  - ``(g,)``       — outermost loop group with index ``g``
  - ``(g, h)``     — inner loop group ``h`` nested inside outer group ``g``
  - etc.

``loop_count`` is a *list* of trip counts, one per nesting level from outermost
to innermost.  For a flat (depth-1) group this is a 1-element list ``[K]``.
``loop_tiled_dims`` is a *list of lists*, one sub-list per nesting level.

Usage — flat (single loop)::

    coarse_tile(
        operations,
        groups=[
            ([op_a, op_b], K),            # group 0: tile dim 0 by K (default)
            ([op_c], K2, [0, 1]),          # group 1: tile dims 0 and 1 by K2
        ],
    )

Usage — nested (two independent loops on one op)::

    coarse_tile(
        operations,
        groups=[
            ([op_a], [(K1, [0]), (K2, [1])]),  # outer K1 on dim 0; inner K2 on dim 1
        ],
    )

``groups`` is a list of ``(ops, spec[, tiled_dims])`` tuples where ``spec`` is
either:
  - a scalar ``loop_count`` (optionally with a third ``tiled_dims`` element), or
  - a list of ``(loop_count, tiled_dims)`` pairs for nested loops.

Each ``ops`` list must be a contiguous sub-sequence of ``operations``.
"""

from __future__ import annotations


import sympy
from sympy import Expr

from torch._inductor.ir import ComputedBuffer, Operation, Pointwise, Reduction

from .logging_utils import get_inductor_logger

logger = get_inductor_logger("coarse_tile")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def coarse_tile(
    operations: list[Operation],
    groups: list[tuple],
    *,
    tiled_dims: list[int] | None = None,
) -> None:
    """Stamp loop_group_id / loop_count on operations and scale their ranges.

    Parameters
    ----------
    operations:
        The full ordered list of IR operations (as seen by
        CustomPreSchedulingPasses).  Not modified; used only for validation.
    groups:
        Sequence of ``(ops, spec[, tiled_dims])`` tuples.  ``spec`` is either:

        * A scalar ``loop_count`` (with optional third element ``tiled_dims``)
          for a flat single-level loop — tile all ops in ``ops`` by that count.
        * A list of ``(loop_count, tiled_dims)`` pairs for nested loops —
          the outermost pair is first, the innermost last.  The ops end up in
          the innermost loop body; each level's count and dims are stamped on
          the op and the corresponding iteration ranges are divided.
    tiled_dims:
        Default ``tiled_dims`` for flat groups that do not supply their own.
        ``None`` means tile only dimension 0.  Ignored for nested-spec groups.
    """
    op_to_position: dict[str, int] = {
        op.get_operation_name(): i for i, op in enumerate(operations)
    }

    for group_idx, group in enumerate(groups):
        group_ops = group[0]
        spec = group[1]
        group_id: tuple[int, ...] = (group_idx,)

        if isinstance(spec, list):
            # Nested spec: list of (loop_count, tiled_dims) pairs.
            levels: list[tuple[Expr, list[int]]] = spec
        else:
            # Flat spec: scalar loop_count with optional tiled_dims override.
            flat_tiled = group[2] if len(group) > 2 else tiled_dims
            effective_dims: list[int] = [0] if flat_tiled is None else flat_tiled
            levels = [(spec, effective_dims)]

        _stamp_group(group_ops, group_id, levels, op_to_position)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stamp_group(
    ops: list[Operation],
    group_id: tuple[int, ...],
    levels: list[tuple[Expr, list[int]]],
    op_to_position: dict[str, int],
) -> None:
    """Stamp loop_group_id / loop_count / loop_tiled_dims and divide ranges.

    ``levels`` is a list of ``(loop_count, tiled_dims)`` pairs, outermost
    first.  The op receives a ``loop_group_id`` whose length equals
    ``len(levels)`` (one path element per nesting level).  ``loop_count`` and
    ``loop_tiled_dims`` are stamped as lists parallel to ``levels``.

    Iteration ranges are divided in outermost-first order: outer count applied
    to outer dims, then inner count applied to inner dims.
    """
    if not ops:
        return

    _validate_contiguous(ops, op_to_position, group_id)

    # Build full nested group_id: (group_idx, 0, 0, ...) with len == len(levels).
    # The inner path elements are always 0 because each level contains exactly
    # the ops from this group (no siblings at inner depths).
    nested_group_id: tuple[int, ...] = group_id + (0,) * (len(levels) - 1)

    counts = [lvl[0] for lvl in levels]
    dims_per_level = [lvl[1] for lvl in levels]

    for op in ops:
        if not isinstance(op, ComputedBuffer):
            logger.debug(
                "coarse_tile: skipping non-ComputedBuffer op %s (%s)",
                op.get_operation_name(),
                type(op).__name__,
            )
            continue

        for count, dims in levels:
            _divide_ranges(op, count, dims)

        op.loop_group_id = nested_group_id  # type: ignore[attr-defined]
        op.loop_count = counts  # type: ignore[attr-defined]
        op.loop_tiled_dims = dims_per_level  # type: ignore[attr-defined]

        logger.debug(
            "coarse_tile: stamped %s loop_group_id=%s loop_count=%s loop_tiled_dims=%s",
            op.get_operation_name(),
            nested_group_id,
            counts,
            dims_per_level,
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
    Out-of-bounds indices are silently skipped.
    """
    data = op.data
    if not isinstance(data, (Pointwise, Reduction)):
        return

    ranges = list(data.ranges)
    if not ranges:
        return

    for i in tiled_dims:
        if i >= len(ranges):
            continue
        ranges[i] = sympy.Rational(1, 1) * ranges[i] / loop_count
        # Simplify: keep as integer expression when divisible.
        ranges[i] = sympy.simplify(ranges[i])

    # Loops is a frozen dataclass; use object.__setattr__ to mutate it.
    object.__setattr__(data, "ranges", ranges)


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
