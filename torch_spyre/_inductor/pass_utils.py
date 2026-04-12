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

from typing import NamedTuple


import sympy
from torch._inductor.ir import ComputedBuffer, FixedLayout, Pointwise, Reduction
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.dependencies import MemoryDep, ReadWrites
from torch._inductor.virtualized import V
from torch_spyre._inductor.errors import Unsupported

from .ir import FixedTiledLayout
from .views import compute_coordinates


class SchedNodeArg(NamedTuple):
    dep: MemoryDep
    layout: FixedTiledLayout


def get_mem_deps(n: SchedulerNode) -> list[SchedNodeArg]:
    res: list[SchedNodeArg] = []
    for arg in n.read_writes.reads:
        if isinstance(arg, MemoryDep):
            buf = V.graph.get_buffer(arg.name)
            layout = buf.get_layout()
            if not isinstance(layout, FixedTiledLayout):
                raise RuntimeError(f"{buf} does not have FixedTiledLayout")
            res.append(SchedNodeArg(arg, layout))
    return res


def get_mem_deps_from_rw(read_writes: ReadWrites) -> list[SchedNodeArg]:
    res: list[SchedNodeArg] = []
    for arg in read_writes.reads:
        if isinstance(arg, MemoryDep):
            buf = V.graph.get_buffer(arg.name)
            layout = buf.get_layout()
            if not isinstance(layout, FixedTiledLayout):
                raise RuntimeError(f"{buf} does not have FixedTiledLayout")
            res.append(SchedNodeArg(arg, layout))
    return res


def host_coordinates(layout: FixedLayout, dep: MemoryDep) -> list[sympy.Expr]:
    return compute_coordinates(layout.size, layout.stride, dep.ranges, dep.index)


def device_coordinates(layout: FixedTiledLayout, dep: MemoryDep) -> list[sympy.Expr]:
    return compute_coordinates(
        layout.device_layout.device_size,
        layout.device_layout.stride_map,
        dep.ranges,
        dep.index,
    )


def iteration_space(n: SchedulerNode) -> dict[sympy.Symbol, sympy.Expr]:
    if isinstance(n.node.data, Pointwise):
        # The iteration space of a Pointwise is that of its output
        return next(iter(n.read_writes.writes)).ranges.copy()
    elif isinstance(n.node.data, Reduction):
        # The iteration space of a Reduction is that of its input
        return next(iter(n.read_writes.reads)).ranges.copy()
    else:
        raise Unsupported("Unexpected node type")


def iteration_space_from_op(op: ComputedBuffer) -> dict[sympy.Symbol, sympy.Expr]:
    """Pre-scheduler version of iteration_space: uses op.get_read_writes() instead
    of SchedulerNode.read_writes."""
    rw = op.get_read_writes()
    if isinstance(op.data, Pointwise):
        return next(iter(rw.writes)).ranges.copy()
    elif isinstance(op.data, Reduction):
        return next(iter(rw.reads)).ranges.copy()
    else:
        raise Unsupported("Unexpected node type")


def map_ir_splits_to_scheduler(
    ir_sizes: list[int],
    ir_splits: list[int],
    sched_it_space: dict[sympy.Symbol, sympy.Expr],
) -> dict[sympy.Symbol, int]:
    """Map positional IR-level core-division splits to scheduler-level symbol keys.

    At pre-scheduler time, core_division stores splits as a list[int] parallel to
    the IR-level iteration-space dimensions (in natural/declaration order).  At
    codegen time the Scheduler has renamed and possibly reordered or merged those
    dimensions via ``simplify_and_reorder``.

    The scheduler sorts dimensions by decreasing stride.  For contiguous row-major
    tensors this equals natural order, but broadcast operands (stride=0) can cause
    the scheduler to reorder non-broadcast dims ahead of others.

    Strategy: sort BOTH the IR (size, split) pairs AND the scheduler (size, symbol)
    pairs by decreasing size using a stable sort.  After sorting, IR and scheduler
    dims align positionally:

    - Distinct sizes: sort uniquely resolves the mapping and handles reordering.
    - Duplicate sizes: Python's stable sort preserves relative order among equal
      elements, which matches the scheduler's tie-breaking by original position.
    - Fewer scheduler dims (merging occurred): after sort-aligning, greedily consume
      consecutive IR dims whose sizes multiply to each scheduler dim's size.
    """
    sched_syms = list(sched_it_space.keys())
    sched_sizes = [int(v) for v in sched_it_space.values()]

    # Sort IR dims by decreasing size (stable: preserves relative order for ties).
    sorted_ir = sorted(zip(ir_sizes, ir_splits), key=lambda p: p[0], reverse=True)
    sorted_ir_sizes = [p[0] for p in sorted_ir]
    sorted_ir_splits = [p[1] for p in sorted_ir]

    # Sort scheduler dims by decreasing size (stable: same tie-breaking rule).
    sorted_sched = sorted(
        zip(sched_sizes, sched_syms), key=lambda p: p[0], reverse=True
    )
    sorted_sched_sizes = [p[0] for p in sorted_sched]
    sorted_sched_syms = [p[1] for p in sorted_sched]

    if len(sorted_ir_sizes) == len(sorted_sched_sizes):
        # Common case: no merging. 1-to-1 positional match on sorted dims.
        return {sym: split for sym, split in zip(sorted_sched_syms, sorted_ir_splits)}

    # Dimension merging occurred.  After size-sorting, the scheduler's merged dims
    # correspond to consecutive runs of IR dims with matching size-products.
    result: dict[sympy.Symbol, int] = {}
    ir_idx = 0
    for sym, sched_size in zip(sorted_sched_syms, sorted_sched_sizes):
        product_size = 1
        product_split = 1
        while ir_idx < len(sorted_ir_sizes) and product_size < sched_size:
            product_size *= sorted_ir_sizes[ir_idx]
            product_split *= sorted_ir_splits[ir_idx]
            ir_idx += 1
        assert product_size == sched_size, (
            f"Cannot map IR dims to scheduler dims: "
            f"ir_sizes={ir_sizes}, sched_sizes={sched_sizes}"
        )
        result[sym] = product_split

    assert ir_idx == len(sorted_ir_sizes), "Not all IR dimensions consumed"
    return result
