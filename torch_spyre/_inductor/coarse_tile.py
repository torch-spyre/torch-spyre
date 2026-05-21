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

Each group of operations is wrapped in a counted outer loop whose trip count is
``loop_count``.  For every operation in the group the iteration ranges that are
divided by ``loop_count`` are scaled down by that factor; the resulting (smaller)
per-iteration ranges are what the downstream scheduler and work-division passes
will see.

A ``loop_group_id`` tuple encodes the nesting path:
  - ``(g,)``       — outermost loop group with index ``g``
  - ``(g, h)``     — inner loop group ``h`` nested inside outer group ``g``
  - etc.

``loop_count`` is always the trip count of the *innermost* loop that directly
contains the operation, i.e. the loop whose body the operation lives in.

Usage::

    coarse_tile(
        operations,
        groups=[
            ([op_a, op_b], count_k),         # group 0: tile outermost dim (default)
            ([op_c], count_m, 1),             # group 1: tile dim 1 specifically
        ],
    )

``groups`` is a list of ``(ops, loop_count[, tiled_dims])`` tuples.  Each
``ops`` list must be a contiguous sub-sequence of ``operations`` (a gap would
indicate a data-flow dependency crossing the group boundary).  The optional
third element ``tiled_dims`` overrides the ``tiled_dims`` keyword argument for
that group, allowing different groups to tile different iteration-space
dimensions.  Nested groups are not yet exposed by this API but the stamped
attributes support them.
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
    tiled_dims: int | None = None,
) -> None:
    """Stamp loop_group_id / loop_count on operations and scale their ranges.

    Parameters
    ----------
    operations:
        The full ordered list of IR operations (as seen by
        CustomPreSchedulingPasses).  Not modified; used only for validation.
    groups:
        Sequence of ``(ops, loop_count[, tiled_dims])`` tuples.  Each ``ops``
        is a list of ``ComputedBuffer`` operations that belong to the same
        outermost loop group.  ``loop_count`` is the trip count (may be
        symbolic).  The optional third element overrides the ``tiled_dims``
        keyword argument for that specific group, allowing different groups to
        tile different iteration-space dimensions.
    tiled_dims:
        Default number of leading iteration-space dimensions to divide by
        ``loop_count``.  ``None`` means tile only the single outermost
        dimension.  Overridden per-group by a third tuple element.
    """
    op_to_position: dict[str, int] = {
        op.get_operation_name(): i for i, op in enumerate(operations)
    }

    for group_idx, group in enumerate(groups):
        group_ops, loop_count = group[0], group[1]
        group_tiled_dims = group[2] if len(group) > 2 else tiled_dims
        group_id: tuple[int, ...] = (group_idx,)
        _stamp_group(group_ops, group_id, loop_count, group_tiled_dims, op_to_position)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stamp_group(
    ops: list[Operation],
    group_id: tuple[int, ...],
    loop_count: Expr,
    tiled_dims: int | None,
    op_to_position: dict[str, int],
) -> None:
    """Stamp loop_group_id / loop_count and divide ranges for a single group."""
    if not ops:
        return

    _validate_contiguous(ops, op_to_position, group_id)

    for op in ops:
        if not isinstance(op, ComputedBuffer):
            logger.debug(
                "coarse_tile: skipping non-ComputedBuffer op %s (%s)",
                op.get_operation_name(),
                type(op).__name__,
            )
            continue

        _divide_ranges(op, loop_count, tiled_dims)

        op.loop_group_id = group_id  # type: ignore[attr-defined]
        op.loop_count = loop_count  # type: ignore[attr-defined]

        logger.debug(
            "coarse_tile: stamped %s loop_group_id=%s loop_count=%s",
            op.get_operation_name(),
            group_id,
            loop_count,
        )


def _divide_ranges(
    op: ComputedBuffer,
    loop_count: Expr,
    tiled_dims: int | None,
) -> None:
    """Divide the outermost ``tiled_dims`` iteration ranges of op by loop_count.

    For a ``Pointwise`` the full ranges are op.data.ranges.
    For a ``Reduction`` the non-reduction (outer) ranges are op.data.ranges;
    op.data.reduction_ranges are left untouched.

    When ``tiled_dims`` is None, only the single outermost dimension is divided.
    When ``tiled_dims`` is 0 or the op has no ranges, nothing is modified.
    """
    data = op.data
    if not isinstance(data, (Pointwise, Reduction)):
        return

    ranges = list(data.ranges)
    if not ranges:
        return

    n_dims = 1 if tiled_dims is None else min(tiled_dims, len(ranges))
    if n_dims == 0:
        return

    for i in range(n_dims):
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
