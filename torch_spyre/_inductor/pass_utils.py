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
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    Pointwise,
    Reduction,
)
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


def _sym_to_device_dim(
    device_coords: list[sympy.Expr],
) -> dict[sympy.Symbol, int]:
    """Return a mapping from each iteration-space symbol to its outermost device
    dimension index.

    ``device_coords[d]`` is the expression for device dimension ``d`` in terms of
    the iteration-space symbols.  For each symbol ``s`` we find the smallest ``d``
    such that ``s`` appears in ``device_coords[d]``.  This mapping is stable across
    the pre-scheduling / codegen boundary because the device layout does not change.
    """
    result: dict[sympy.Symbol, int] = {}
    for d, coord in enumerate(device_coords):
        for sym in coord.free_symbols:
            if sym not in result:
                result[sym] = d
    return result


def splits_by_device_dim(
    splits: dict[sympy.Symbol, int],
    device_coords: list[sympy.Expr],
) -> dict[int, int]:
    """Convert a symbol→split dict to a device_dim→split dict.

    Only non-unity splits are stored; the caller uses 1 as the default.
    When a scheduler symbol maps to the same device dimension as an IR symbol
    (guaranteed because the device layout is fixed), the two dicts share the
    same integer keys and the mapping is unambiguous.
    """
    sym_to_dim = _sym_to_device_dim(device_coords)
    result: dict[int, int] = {}
    for sym, split in splits.items():
        if split > 1:
            dim = sym_to_dim.get(sym)
            if dim is not None:
                result[dim] = split
    return result


def apply_splits_from_device_dim(
    dim_splits: dict[int, int],
    device_coords: list[sympy.Expr],
    sched_it_space: dict[sympy.Symbol, sympy.Expr],
) -> dict[sympy.Symbol, int]:
    """Reconstruct a scheduler-symbol→split dict from a device_dim→split dict.

    At codegen time the scheduler's iteration-space symbols are different from
    the pre-scheduler symbols, but ``device_coordinates`` evaluated against the
    scheduler write dep produces the same device-dim-to-symbol mapping.
    ``dim_splits`` was produced by ``splits_by_device_dim`` at pre-scheduler time;
    we invert ``_sym_to_device_dim`` on the scheduler side to recover the
    correct symbol for each split.
    """
    sched_sym_to_dim = _sym_to_device_dim(device_coords)
    dim_to_sched_sym = {d: sym for sym, d in sched_sym_to_dim.items()}
    result: dict[sympy.Symbol, int] = {sym: 1 for sym in sched_it_space}
    for dim, split in dim_splits.items():
        sym = dim_to_sched_sym.get(dim)
        if sym is not None and sym in result:
            result[sym] = split
    return result
