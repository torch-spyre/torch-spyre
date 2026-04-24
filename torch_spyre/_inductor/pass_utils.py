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

from typing import Callable, NamedTuple, TypeVar, Union


import sympy
from sympy import Expr
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


def concretize_expr(expr: Union[Expr, int]) -> int:
    """Concretize a sympy expression to a Python int.

    Used at boundaries where concrete values are required (e.g. C++
    constructors that only accept ``int``, comparison operators inside
    algorithms such as core-division and coordinate computation).

    Key invariant: only structural parameters (sizes, strides, split
    counts) are concretized.  Symbolic loop variables inside coordinate
    output expressions are never touched, so the generated coordinate
    expressions remain symbolic and will carry through to the SDSC when
    symbolic SDSC generation is implemented.
    """
    if isinstance(expr, int):
        return expr
    if isinstance(expr, sympy.Integer):
        return int(expr)
    if hasattr(expr, "free_symbols") and expr.free_symbols:
        return V.graph.sizevars.size_hint(expr)
    return int(expr)


def concretize_index(index: sympy.Expr, loop_vars: set) -> sympy.Expr:
    """Replace non-loop symbolic variables in an index expression with concrete values.

    With ``dynamic=True``, the host index may contain symbolic strides. When
    ``normalize_coordinates`` isolates each loop variable's contribution
    by substituting 0 for all other free symbols, the size symbol ``s1``
    is also zeroed.  This function replaces size symbols with their concrete
    hints so that coordinate expressions are structurally identical to static-shape
    compilation while loop variable symbols are preserved.
    """
    size_syms = index.free_symbols - loop_vars
    if not size_syms:
        return index
    subs = {s: V.graph.sizevars.size_hint(s) for s in size_syms}
    result = index.subs(subs)
    return result


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
    # Concretize size/stride so compute_coordinates can use plain ``<``/``>``
    # comparisons.  var_ranges and index stay symbolic so the *output*
    # coordinate expressions remain symbolic.
    # TODO(issue#1373): remove concretization once compute_coordinates handles
    #              symbolic comparisons natively.
    concrete_size = [concretize_expr(s) for s in layout.size]
    concrete_stride = [concretize_expr(s) for s in layout.stride]
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    return compute_coordinates(concrete_size, concrete_stride, dep.ranges, index)


def device_coordinates(layout: FixedTiledLayout, dep: MemoryDep) -> list[sympy.Expr]:
    # device_size and stride_map come from the C++ SpyreTensorLayout and are
    # already concrete, so no concretization is needed here.
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    return compute_coordinates(
        layout.device_layout.device_size,
        layout.device_layout.stride_map,
        dep.ranges,
        index,
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


_V = TypeVar("_V")

# Type alias for the two-namespace split storage: (output_splits, reduction_splits).
# output_splits is keyed by the symbol's coefficient in the write dep's index.
# reduction_splits is keyed by the symbol's coefficient in the first read dep's index.
# The two dicts use different reference indices so their keys never collide.
ItSpaceSplits = tuple[dict[sympy.Expr, int], dict[sympy.Expr, int]]


def _coeff_splits_from_index(
    splits: dict[sympy.Symbol, _V],
    index: sympy.Expr,
    *,
    skip: "Callable[[_V], bool] | None" = None,
) -> dict[sympy.Expr, _V]:
    """Return a coeff→value dict for symbols with a non-zero coefficient in index.

    The coefficient of a symbol in a flat tensor index expression is stable
    across the pre-scheduling / codegen boundary (same layout strides on both
    sides), so it serves as a symbol-identity key that survives the scheduler's
    renaming.  Symbols absent from index (coeff=0) are not included.

    Entries for which ``skip(value)`` returns True are omitted.
    """
    result: dict[sympy.Expr, _V] = {}
    for sym, value in splits.items():
        if skip is not None and skip(value):
            continue
        coeff = index.coeff(sym)
        if coeff != 0:
            result[coeff] = value
    return result


def splits_by_index_coeff(
    splits: dict[sympy.Symbol, int],
    write_index: sympy.Expr,
    read_index: sympy.Expr,
) -> ItSpaceSplits:
    """Encode a symbol→split dict as a pair of coeff-keyed dicts.

    Output dims (those present in write_index) are encoded using their
    coefficient in write_index.  Reduction dims (absent from write_index) are
    encoded using their coefficient in read_index.  The two dicts form separate
    namespaces so their keys never collide, even when output and reduction dims
    happen to share the same stride value in different tensors.

    Only non-unity splits are stored; 1 is the default on the apply side.
    """
    skip = lambda v: v <= 1  # noqa: E731
    output_splits = _coeff_splits_from_index(splits, write_index, skip=skip)
    # Reduction splits: symbols with coeff==0 in write_index but coeff!=0 in read_index
    reduction_only = {
        sym: val for sym, val in splits.items() if write_index.coeff(sym) == 0
    }
    reduction_splits = _coeff_splits_from_index(reduction_only, read_index, skip=skip)
    return output_splits, reduction_splits


def apply_splits_from_index_coeff(
    coeff_splits: ItSpaceSplits,
    write_index: sympy.Expr,
    read_index: sympy.Expr,
    sched_it_space: dict[sympy.Symbol, sympy.Expr],
) -> dict[sympy.Symbol, int]:
    """Reconstruct a scheduler-symbol→split dict from an ItSpaceSplits pair.

    Output dims (non-zero coeff in write_index) are looked up in
    coeff_splits[0]; reduction dims (zero coeff in write_index) are looked up
    in coeff_splits[1] via their coefficient in read_index.  Symbols not found
    in either dict default to 1.
    """
    output_coeff_splits, reduction_coeff_splits = coeff_splits
    result: dict[sympy.Symbol, int] = {sym: 1 for sym in sched_it_space}
    for sym, size in sched_it_space.items():
        # Skip iteration vars with trivial range.  For symbolic ranges we
        # cannot statically determine triviality (and a symbolic size
        # carries no compile-time guarantee that it is 1), so we assume
        # they are non-trivial — consistent with views.compute_coordinates.
        # TODO(issue#1373): replace with a sympy-aware predicate.
        if isinstance(size, (int, sympy.Integer)) and int(size) <= 1:
            continue
        wc = write_index.coeff(sym)
        if wc != 0:
            if wc in output_coeff_splits:
                result[sym] = output_coeff_splits[wc]
        else:
            rc = read_index.coeff(sym)
            if rc != 0 and rc in reduction_coeff_splits:
                result[sym] = reduction_coeff_splits[rc]
    return result
