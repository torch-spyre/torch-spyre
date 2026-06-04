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

After stamping, ``coarse_tile`` calls ``insert_tiling_propagation`` to allocate
full-sized output buffers and insert copy/mutation ops for Pointwise operations
whose results are consumed outside the loop.
"""

from __future__ import annotations


import sympy
from sympy import Expr

import torch
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from .logging_utils import get_inductor_logger

logger = get_inductor_logger("coarse_tile")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def coarse_tile(
    operations: list[Operation],
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
        Sequence of ``(ops, spec)`` tuples produced by
        ``hints_to_coarse_tile_groups``.  ``spec`` is a list of
        ``(hint_id, loop_count, tiled_dims)`` triples for nested loops —
        outermost first.  The ops end up in the innermost loop body; each
        level's count and dims are stamped on the op and the corresponding
        iteration ranges are divided.
    """
    op_to_position: dict[str, int] = {
        op.get_operation_name(): i for i, op in enumerate(operations)
    }

    for group_idx, group in enumerate(groups):
        group_ops = group[0]
        levels: list[tuple[int, Expr, list[int]]] = group[1]
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
    for group in groups:
        group_ops: list[Operation] = group[0]
        for op in group_ops:
            if not isinstance(op, ComputedBuffer):
                continue
            if not isinstance(op.data, (Pointwise, Reduction)):
                continue
            _propagate_tiled_op(op, operations)


def _check_reduction_tiling_safety(op: ComputedBuffer) -> None:
    """Raise RuntimeError for unsupported Reduction-in-loop configurations.

    Rejects any tiled dim that falls in the reduction_ranges index range — the
    accumulation-buffer logic for a tiled reduction dim is not yet implemented.
    """
    data = op.data
    assert isinstance(data, Reduction)

    n_output_dims = len(data.ranges)
    loop_tiled_dims: list[list[int]] = getattr(op, "loop_tiled_dims", [])
    for dims_list in loop_tiled_dims:
        for d in dims_list:
            if d >= n_output_dims:
                raise RuntimeError(
                    f"coarse_tile: reduction op {op.get_name()!r} has "
                    f"tiled_dim={d} which falls in the reduction dimension "
                    "(tiled reduction dims are not yet supported)."
                )


def _propagate_tiled_op(
    op: ComputedBuffer,
    operations: list[Operation],
) -> None:
    """Handle buffer propagation for a single tiled Pointwise or Reduction op."""
    if isinstance(op.data, Reduction):
        _check_reduction_tiling_safety(op)

    loop_group_id = getattr(op, "loop_group_id", None)
    if loop_group_id is None:
        return

    buf_name = op.get_name()
    outside_consumers, is_graph_output = _find_outside_consumers(
        buf_name, loop_group_id, operations
    )

    # If no dims were tiled (loop_tiled_dims all empty), the op is loop-invariant —
    # mark per_tile_fixed so the unroller reuses the same address each tile.
    if all(not dims for dims in getattr(op, "loop_tiled_dims", [[]])):
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
        and getattr(o, "loop_group_id", (None,))[0] == outer_key
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
        gid = getattr(op, "loop_group_id", None)
        if gid is None or gid[0] != outer_key:
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
        gid = getattr(op, "loop_group_id", None)
        if gid is None or gid[0] != outer_key:
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
    loop_count: list[Expr] = op.loop_count
    loop_tiled_dims: list[list[int]] = op.loop_tiled_dims
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

    full_buf.layout = FixedTiledLayout(
        device,
        dtype,
        list(full_ranges),
        strides,
        device_layout,
    )

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
    copy_buf.loop_group_id = tiled_op.loop_group_id  # type: ignore[attr-defined]
    copy_buf.loop_count = tiled_op.loop_count  # type: ignore[attr-defined]
    copy_buf.loop_tiled_dims = tiled_op.loop_tiled_dims  # type: ignore[attr-defined]

    V.graph.name_to_buffer[copy_name] = copy_buf

    tiled_idx = operations.index(tiled_op)
    operations.insert(tiled_idx + 1, copy_buf)


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
    levels: list[tuple[int, Expr, list[int]]],
    op_to_position: dict[str, int],
) -> None:
    """Stamp loop_group_id / loop_count / loop_tiled_dims and divide ranges.

    ``levels`` is a list of ``(hint_id, loop_count, tiled_dims)`` triples,
    outermost first.  Each op's ranges are divided using its own dim_hints,
    matched to the correct level by hint_id so that op-specific dim_index
    values are used rather than the spec op's indices.
    """
    if not ops:
        return

    _validate_contiguous(ops, op_to_position, group_id)

    nested_group_id: tuple[int, ...] = group_id + (0,) * (len(levels) - 1)
    counts = [lvl[1] for lvl in levels]

    for op in ops:
        if not isinstance(op, ComputedBuffer):
            logger.debug(
                "coarse_tile: skipping non-ComputedBuffer op %s (%s)",
                op.get_operation_name(),
                type(op).__name__,
            )
            continue

        # Each op carries its own dim_hints with per-op dim_index values (ops in
        # the same group can have different iteration spaces, e.g. a broadcast op
        # may lack the tiled dimension entirely).  Use the op's own dim_index;
        # fall back to empty when the op has no matching dim for a level.
        op_hints = getattr(op, "dim_hints", [])
        hint_id_to_dim_index: dict[int, int] = {
            h.hint_id: h.dim_index
            for h in op_hints
            if h.dim_index is not None and not h.is_reduction
        }
        op_tiled_dims: list[list[int]] = []
        for hint_id, count, spec_dims in levels:
            dim_index = hint_id_to_dim_index.get(hint_id)
            if dim_index is not None:
                effective = [dim_index]  # use this op's own dim_index
            else:
                effective = []  # op has no dim for this level — loop-invariant
            op_tiled_dims.append(effective)
            _divide_ranges(op, count, effective)

        op.loop_group_id = nested_group_id  # type: ignore[attr-defined]
        op.loop_count = counts  # type: ignore[attr-defined]
        op.loop_tiled_dims = op_tiled_dims  # type: ignore[attr-defined]

        logger.debug(
            "coarse_tile: stamped %s loop_group_id=%s loop_count=%s loop_tiled_dims=%s",
            op.get_operation_name(),
            nested_group_id,
            counts,
            op_tiled_dims,
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
        if i < 0 or i >= len(ranges):
            continue
        r = ranges[i]
        if isinstance(r, (int, sympy.Integer)) and isinstance(
            loop_count, (int, sympy.Integer)
        ):
            if int(r) % int(loop_count) != 0:
                raise RuntimeError(
                    f"coarse_tile: op {op.get_name()!r} dimension {i} range {r} "
                    f"is not divisible by loop_count {loop_count}.  All tiled "
                    f"dimensions must be evenly divisible by the loop trip count."
                )
            ranges[i] = sympy.Integer(int(r) // int(loop_count))
        else:
            ranges[i] = sympy.sympify(r) / sympy.sympify(loop_count)

    # Loops is a frozen dataclass; use object.__setattr__ to mutate it.
    object.__setattr__(data, "ranges", ranges)

    # Sync layout.size, layout.stride, and layout.device_layout with the new ranges.
    from torch._inductor.ir import FixedLayout, FlexibleLayout

    from .ir import FixedTiledLayout

    layout = getattr(op, "layout", None)
    if not (isinstance(layout, FixedLayout) and len(layout.size) == len(ranges)):
        return

    new_size = list(layout.size)
    for i in tiled_dims:
        if 0 <= i < len(new_size):
            new_size[i] = ranges[i]
    layout.size = new_size

    # Recompute contiguous strides for the smaller buffer.
    layout.stride = list(FlexibleLayout.contiguous_strides(new_size))

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
