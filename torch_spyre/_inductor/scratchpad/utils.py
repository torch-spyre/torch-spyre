# Copyright 2026 The Torch-Spyre Authors.
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


import math
from typing import Any, Optional
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    Operation,
    IRNode,
    Pointwise,
)
from torch._inductor.virtualized import V
from torch._inductor.ops_handler import WrapperHandler

import sympy

from torch_spyre._inductor import config
from torch_spyre._inductor.pass_utils import (
    _per_core_view_on_buf,
    concretize_expr,
    op_read_writes,
    device_coordinates,
)
from torch._inductor.ir import MutationLayoutSHOULDREMOVE, ComputedBuffer
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer

# Op outputs eligible for LX-pinning. `amax` is the lowered form of
# `max`; both names are listed to match whichever the IR shows.
OP_OUTPUT_GOOD_FOR_LX_REUSE = frozenset(
    {
        "max",
        "amax",
        "maximum",
        "sum",
        "clone",
        "exp",
        "sub",
        "mul",
        "mean",
        "add",
        "rsqrt",
        "neg",
        "mm",
        "bmm",
        "batched_matmul",
        "div",
        "realdiv",
        "expand",
        "silu",
    }
)


def clone_at_graph_boundaries() -> bool:
    """True when clone ops are eligible for LX, enabling clone insertion at graph
    input/output boundaries so those buffers can also be LX-pinned.

    Gated by listing "clone" in OP_OUTPUT_GOOD_FOR_LX_REUSE. It intentionally
    does NOT consult ``allow_all_ops_in_lx_planning``: that flag widens
    intermediate-output eligibility and is set broadly (e.g. the LX-planning
    op suite), so coupling it here would silently turn on the boundary clone
    path in contexts that don't intend to exercise it."""
    return "clone" in OP_OUTPUT_GOOD_FOR_LX_REUSE


class GraphView:
    """
    Simple wrapper which allows filtering of returned operations
    without mutating the underlying graph.
    """

    def __init__(self, graph, predicate):
        self.graph = graph
        self.operations = predicate(graph)

    def __getattr__(self, name):
        return getattr(self.graph, name)


def calculate_liveness(graph: GraphLowering) -> dict[str, list[int]]:
    """Return a dict mapping each buffer name to the sorted list of operation indices
    at which that buffer is accessed (read or written).  Graph inputs are seeded with
    an empty list; unused inputs remain empty.

    Note: previously, unused graph inputs did not appear in the returned dict at
    all.  Now they appear with an empty list.  Callers that skip buffers with
    ``len(uses) <= 1`` (e.g. ``_build_bound_buffers``) will still skip unused inputs
    correctly, since ``len([]) == 0 <= 1``."""
    liveness: dict[str, list[int]] = {}
    for input_name in graph.graph_input_names:
        liveness[input_name] = []
    for i, op in enumerate(graph.operations):
        rw = op_read_writes(op)
        for mem_dep in rw.reads | rw.writes:
            buf_name = mem_dep.name
            if buf_name not in liveness:
                liveness[buf_name] = []
            liveness[buf_name].append(i)
    return liveness


def mem_usage_by_buf(
    graph: GraphLowering | GraphView,
    cache: Optional[dict] = None,
) -> dict:
    """
    Get a summary of memory usage of each operation.
    Includes detailed info of individual buf, e.g. mem_usage[<buf_name>],
    which has "size_per_core", "size", "core_div_mismatch", "op_inputs" fields
    NOTE:
    if a buf is not in core_div_mismatch => it has no users => graph output
    """
    num_cores_per_op = get_ncores_for_buffers(graph, cache)
    mem_usage: dict = {}

    for op in graph.operations:
        buf_name = op.name
        buf = graph.get_buffer(buf_name)
        num_cores = num_cores_per_op.get(buf_name, -1)
        rw = op_read_writes(op)
        layout = buf.layout
        if isinstance(layout, MutationLayoutSHOULDREMOVE) or not isinstance(
            op, ComputedBuffer
        ):
            mem_usage[buf_name] = {
                "size": -1,
                # Unsized sentinel (mirrors "size"); the core_div_mismatch flag
                # below carries validity, so no arithmetic on num_cores here.
                "size_per_core": -1,
                "core_div_mismatch": num_cores < 0,
                "op_inputs": [dep.name for dep in rw.reads],
            }
            continue
        dev_layout = layout.device_layout
        dev_size = (
            math.prod(dev_layout.device_size[:-1]) * 128
        )  # num_sticks * bytes_per_stick
        mem_usage[buf_name] = {
            "size": dev_size,
            "size_per_core": dev_size // num_cores,
            "core_div_mismatch": num_cores < 0,
            "op_inputs": [dep.name for dep in rw.reads],
        }

    return mem_usage


def buffer_not_read_in_full(graph: GraphLowering | GraphView, buf_name: str) -> bool:
    """True if any consumer reads less than the whole ``buf_name`` (a sliced,
    partial, or multi-offset read), or if the footprint can't be proven to
    cover the full buffer.

    An LX-pinned buffer is addressed by a single base (in SDSC codegen the
    ``start_address`` is ``layout.allocation["lx"]``); unlike the HBM path, a
    per-access slice offset is *not* folded into that base, and strided
    partial reads of a multi-dim buffer mis-address. Both failure modes read
    less than the full buffer per access:

    - multi-offset: ``x[:, 0:512] + x[:, 512:1024]`` — two half reads that
      both resolve to the LX base, yielding ``x0 + x0``;
    - partial slice: ``x[:, :, 0:64]`` — a sub-extent read that mis-addresses
      the 3D LX buffer.

    Only buffers every consumer reads in full (e.g. ``exp(x) + x``) are safe
    to LX-pin. We are deliberately conservative: an unprovable (symbolic)
    footprint is treated as unsafe, costing a missed optimization but never
    correctness.

    Why a guard and not a codegen fix: the root cause is that the SDSC LX
    address path (compute_ops._start_addr_data) uses only ``start_address``,
    dropping the per-access view offset that the HBM path folds in via
    ``core_idx_to_slice_offset``. It is a codegen gap, not a hardware limit.
    But folding ``sum(offsets)`` into the LX base only fixes part of it: the
    view offset interacts with per-core work-slicing (at multi-core the split
    changes which coordinate is constant vs per-core), so a correct fix must
    reconcile the view offset with the per-core LX work-slice geometry rather
    than add a single constant. Until that lands, the guard keeps such buffers
    in HBM (correct, just unpinned).
    """
    layout = getattr(graph.get_buffer(buf_name), "layout", None)
    # No layout, or a layout without a concrete size (e.g. MultiOutputLayout,
    # NoneLayout): we cannot prove a full read, so treat as unsafe to pin.
    size = getattr(layout, "size", None)
    if size is None:
        return True
    try:
        full = math.prod(int(concretize_expr(s)) for s in size)
    except (TypeError, ValueError):
        return True
    for op in graph.operations:
        for dep in op_read_writes(op).reads:
            if dep.name != buf_name:
                continue
            try:
                if int(dep.get_numel()) < full:
                    return True
            except (TypeError, ValueError, AttributeError):
                return True
    return False


def _writes_at_constant_offset(op: Operation) -> bool:
    """True if ``op`` writes any buffer at a non-zero *constant* offset -- a
    sliced in-place mutation into a sub-region (e.g. ``x[:, 32:96] = ...``,
    whose write ``MemoryDep`` index is ``256*d0 + d1 + 32`` with
    ``get_offset() == 32``).

    Coverage-aware: only a *constant* non-zero offset counts. Per-core /
    coarse-tile writes carry their per-core shift as a symbol in the offset
    (``free_symbols`` non-empty), so those are NOT flagged -- avoiding the
    coarse-tile over-guard a flat-numel test would trigger.
    """
    for dep in op_read_writes(op).writes:
        try:
            off = dep.get_offset()
        except (TypeError, ValueError, AttributeError):
            continue
        if off != 0 and not getattr(off, "free_symbols", frozenset()):
            return True
    return False


def ops_in_offset_mutation_component(
    graph: GraphLowering | GraphView,
) -> set[str]:
    """Names of ops data-connected to a sliced in-place mutation that writes at
    a constant non-zero offset (e.g. ``x[:, 32:96] = ...``).

    Such a mutation and everything fused with it land in one SDSC. The offset
    write's codegen assumes the target buffer keeps the slicing the eager path
    chose; if the co-optimizing allocator re-slices any op in that fused kernel
    (a different core division), the deeptools scheduler can no longer place the
    offset write and aborts the compile (``DtException: "There must be at least
    one valid candidate"``, ``L3DlOpsScheduler.cpp:1196``). This is the root
    cause of the ``slice_stick_mutation_*`` co-optimizing-allocator failures --
    the division change, *not* LX residency (the abort reproduces with pinning
    fully disabled).

    The caller pins every op in this set to its upstream (fixed) division, so
    the offset-write SDSC keeps the schedulable slicing the greedy /
    placement-only path uses. Fusion boundaries are unknown at planning time, so
    the SDSC is over-approximated by the undirected data-dependency component
    containing the offset write: producer chain (the value written), the
    mutation target it aliases, and the consumers of that target. Over-approxi-
    mation only forgoes a division optimization (correct, never a new failure --
    a fixed division is exactly what greedy uses).

    Coverage-aware via :func:`_writes_at_constant_offset`: symbolic per-core
    offsets (coarse tiling) are not offset writes, so no component is seeded and
    coarse tiling is not constrained.
    """
    # Undirected adjacency over buffer names (op.name == its output buffer,
    # Inductor convention). Edges: producer<->operand (read deps) and a
    # MutationLayout op <-> its aliased target buffer.
    adj: dict[str, set[str]] = {}

    def link(a: str, b: str) -> None:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    seeds: list[str] = []
    for op in graph.operations:
        for dep in op_read_writes(op).reads:
            name = getattr(dep, "name", None)
            if name:
                link(op.name, name)
        layout = getattr(op, "layout", None)
        if isinstance(layout, MutationLayoutSHOULDREMOVE):
            try:
                link(op.name, layout.target.get_name())
            except (AttributeError, TypeError):
                pass
        if _writes_at_constant_offset(op):
            seeds.append(op.name)

    op_names = {op.name for op in graph.operations}
    component: set[str] = set()
    stack = list(seeds)
    while stack:
        node = stack.pop()
        if node in component:
            continue
        component.add(node)
        stack.extend(adj.get(node, ()))
    return component & op_names


def get_buffer_users(graph: GraphLowering | GraphView) -> dict[str, list[Operation]]:
    buf_users_read_and_write: dict[str, list[Operation]] = {}
    for op in graph.operations:
        rw = op_read_writes(op)
        for dep in rw.reads | rw.writes:  # union of the OrderedSets
            buf = dep.name  # buffer name, i.e. a str
            buf_users_read_and_write[buf] = buf_users_read_and_write.get(buf, []) + [op]
    return buf_users_read_and_write


def _get_buffer_user_deps(
    graph: GraphLowering | GraphView,
) -> dict[str, list[tuple[Operation, MemoryDep]]]:
    """Like get_buffer_users but pairs each op with the specific dep it uses.

    In-place ops (same op reads & writes the same buf) get two entries:
    one per dep. If their per-core views diverge — read at one index,
    write at another — the buffer is correctly rejected for LX, since
    that's a within-core data hazard, not just cross-op disagreement.
    """
    buf_user_deps: dict[str, list[tuple[Operation, MemoryDep]]] = {}
    for op in graph.operations:
        rw = op_read_writes(op)
        for dep in rw.reads | rw.writes:
            buf_user_deps.setdefault(dep.name, []).append((op, dep))
    return buf_user_deps


def _op_num_cores(op: Operation) -> int:
    """Cores implied by op.op_it_space_splits (defaults to 1 when unset).

    `op_it_space_splits` is set conditionally by span_reduction_pass /
    work_distribution; ops that don't get split (e.g. trivial pointwise
    on a small output) leave the attribute unset. Match the existing
    convention (pass_utils.py, work_division.py) and treat missing as
    no-split → 1 core.
    """
    splits: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
    return math.prod([s for p in splits for s in p.values()])


def get_ncores_for_buffers(
    graph: GraphLowering | GraphView,
    cache: Optional[dict] = None,
    reject_reasons_out: Optional[dict[str, str]] = None,
) -> dict[str, int]:
    """
    Return a dictionary mapping buffer names to the number of cores
    used by all the operations that uses the buffer.
    If there is a core division mismatch return -1 instead of the
    number of cores.

    Pass an optional `cache` dict to memoize `_per_core_view_on_buf`
    results across calls (e.g. across co-opt search leaves). Safe to
    share only within a single graph, since the cache key includes the
    op name and `dep` (which carries the buffer name).

    Pass an optional `reject_reasons_out` dict to receive detailed
    reasons for core division mismatches (keyed by buffer name).
    """
    result: dict[str, int] = {}
    using_multicore = config.sencores > 1
    buf_user_deps = _get_buffer_user_deps(graph)
    for buf_name, users in buf_user_deps.items():
        # this dict includes graph input and output
        if using_multicore and len(users) > 1:
            # A K-split-reduction writer leaves partial sums on most cores (only
            # k-last cores hold the final value), so it's unsafe on LX even if
            # geometry matches — the `flag` gate applies to write-deps only.
            ref_view = None
            ref_op_name = None
            mismatch_reason = None
            writer_cores = None
            for op, dep in users:
                view, flag, _ = _per_core_view_on_buf(op, dep, buf_name, cache)
                if ref_view is None:
                    ref_view = view
                op_rw = op_read_writes(op)
                if dep in op_rw.writes:
                    # Size by the writer's core count (the writer sets per-core
                    # footprint size/writer_cores; readers touch only their slice),
                    # not max() over users. One writer per buffer (it's named after
                    # its producing op; an in-place op recurs as a reader, not a
                    # second writer). _op_num_cores folds in K-split factors, an
                    # unfaithful output divisor — but a K-split sets `flag` and is
                    # rejected below, so writer_cores divides only for output splits.
                    writer_cores = _op_num_cores(op)
                    if flag:
                        mismatch_reason = f"K-split writer '{op.get_name()}'"
                        break
                else:
                    # Broadcast-read guard. `view` is how this consumer slices the
                    # buffer; its core count is the product of the split factors.
                    # When a consumer splits an iteration axis the buffer does not
                    # have (e.g. a GEMM's free/N dim over a shared activation, or
                    # its M dim over a shared weight), that split contracts out of
                    # the view, so the view covers fewer cores than the op runs.
                    # An LX (per-core scratchpad) buffer would then live on
                    # view_cores cores but be read by op_cores; the cores without
                    # a local copy read stale scratchpad -> wrong results. There is
                    # no single-base LX broadcast, so treat it as a core-division
                    # mismatch and keep the buffer in HBM (correct, just unpinned).
                    # This is not writer-relative: it catches broadcast reads even
                    # when the buffer has no in-graph writer (a graph input cloned
                    # into LX) or when a producer's view happens to match the
                    # broadcast footprint -- cases the `view != ref_view` check
                    # below cannot see.
                    # work_slice_dims entries are (device-dim, split factor);
                    # the per-dim core count is the split factor.
                    view_cores = math.prod(f for _, f in view.work_slice_dims)
                    if view_cores != _op_num_cores(op):
                        mismatch_reason = (
                            f"broadcast read on '{op.get_name()}': view covers "
                            f"{view_cores} cores but op runs {_op_num_cores(op)}"
                        )
                        break
                if view != ref_view:
                    mismatch_reason = (
                        f"op '{ref_op_name}' ref {ref_view} != '{op.get_name()}' {view}"
                    )
                    break
            if mismatch_reason is not None:
                num_cores = -1
                if reject_reasons_out is not None:
                    reject_reasons_out[buf_name] = mismatch_reason
            elif writer_cores is not None:
                num_cores = writer_cores
            else:
                # No writer (graph input, produced outside the graph): fall back
                # to the users' (matching) max count.
                num_cores = max(_op_num_cores(op) for op, _ in users)
        elif using_multicore:
            num_cores = _op_num_cores(users[0][0])
        else:
            num_cores = 1
        result[buf_name] = num_cores
    return result


class _GetLoadStoreIndices(WrapperHandler):
    def __init__(self, inner):
        super().__init__(inner)
        self._load_map = {}
        self._store_map = {}

    def load(self, name: str, index: sympy.Expr):
        self._load_map[name] = index
        return super().load(name, index)

    def store(self, name: str, index: sympy.Expr, value: Any, mode: Any = None):
        self._store_map[name] = index
        return super().store(name, index, value, mode)


def get_load_and_store_indices(
    pointwise: Pointwise,
) -> tuple[dict[str, sympy.Expr], dict[str, sympy.Expr]]:
    handler = _GetLoadStoreIndices(V.MockHandler())
    index = [sympy.Symbol(f"index{i}") for i in range(len(pointwise.ranges))]
    with V.set_ops_handler(handler):
        pointwise.inner_fn(index)
    return handler._load_map, handler._store_map


def get_op_pointwise_inputs(node: IRNode) -> list[str]:
    if not isinstance(node, Pointwise):
        return []
    loads, stores = get_load_and_store_indices(node)

    return [
        inp
        for inp, load_index in loads.items()
        if all(store_index == load_index for store_index in stores.values())
    ]


def _would_produce_lx_back_gap(
    graph: GraphLowering,
    buf_name: str,
    uses: list[int],
) -> bool:
    """Check if pinning a buffer to LX would produce a backGapCore_.

    A backGap fires when device_size[d] > it_dim_size for any device dimension d.
    The backend supports backGap for HBM but not for LX, so buffers triggering
    this condition must stay in HBM.
    """
    buf = graph.get_buffer(buf_name)
    stl = buf.layout.device_layout
    device_size = stl.device_size

    for use_idx in uses:
        op = graph.operations[use_idx]
        rw = op_read_writes(op)
        for dep in rw.reads | rw.writes:
            if dep.name != buf_name:
                continue
            try:
                coords = device_coordinates(stl, dep, None)
            except Exception:
                continue
            for d, coord_expr in enumerate(coords[:-1]):
                syms = coord_expr.free_symbols
                if not syms:
                    if device_size[d] > 1:
                        return True
                    continue
                sym = next(iter(syms))
                it_dim_size = int(dep.ranges[sym])
                if device_size[d] > it_dim_size:
                    return True
    return False


def plot_buffers(buffers: list[LifetimeBoundBuffer], max_height: int):
    """Visualize a scratchpad allocation layout.

    Allocated buffers are shown in blue; buffers that exceed the capacity
    limit are shown in gray.  In-place parent/child pairs that share an
    address are highlighted: a dark overlay spans the combined lifetime and
    a green marker indicates the handoff tick.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    name_to_index = {b.name: i for i, b in enumerate(buffers)}

    fig, ax = plt.subplots()

    for buffer in buffers:
        addr = buffer.address
        if addr is None:
            continue
        color = "b" if addr + buffer.size <= max_height else "lightgray"
        rect = patches.Rectangle(
            xy=(buffer.start_time, addr),
            width=buffer.end_time - buffer.start_time,
            height=buffer.size,
            linewidth=0.3,
            edgecolor="r",
            facecolor=color,
            fill=True,
        )
        ax.add_patch(rect)

    for buffer in buffers:
        addr = buffer.address
        if addr is None:
            continue
        for p in buffer.in_place_parents:
            pj = name_to_index.get(p)
            if pj is None:
                continue
            parent = buffers[pj]
            if parent.address is None:
                continue
            if addr == parent.address:
                ax.add_patch(
                    patches.Rectangle(
                        xy=(parent.start_time, addr),
                        width=buffer.end_time - parent.start_time,
                        height=buffer.size,
                        linewidth=0.3,
                        edgecolor="r",
                        facecolor="k",
                        fill=True,
                        alpha=0.25,
                    )
                )
                ax.add_patch(
                    patches.Rectangle(
                        xy=(buffer.start_time, addr),
                        width=1,
                        height=buffer.size,
                        linewidth=0.3,
                        edgecolor="r",
                        facecolor="g",
                        fill=True,
                    )
                )

    max_time = max((b.end_time for b in buffers), default=0)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, max_height)
    return fig


def quality_plot(
    quality_logs: list[list[int]], temperature_logs: Optional[list[float]] = None
):
    """Plot quality (buffers allocated) over annealing steps.

    Each run is drawn as a thin blue line; their smoothed average is drawn
    in red.  When temperature data is available (typically recorded by
    a method from an annealing schedule), the first run's temperature schedule is
    overlaid on a log-scale right axis in green.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax1 = plt.subplots()
    for log in quality_logs:
        ax1.plot(log, "b", lw=1, alpha=0.1)

    if quality_logs:
        average = np.array(quality_logs).mean(axis=0)
        n_points = len(average)
        if n_points >= 20:
            n_smoothing = min(n_points // 10, 10)
            smoothed = np.convolve(average, np.ones(n_smoothing) / n_smoothing, "valid")
            ax1.plot(
                [x + n_smoothing / 2 for x in range(len(smoothed))],
                smoothed,
                "r",
                lw=3,
            )
        else:
            ax1.plot(average, "r", lw=3)

    if temperature_logs:
        ax2 = ax1.twinx()
        ax2.set_yscale("log")
        ax2.plot(temperature_logs, "g", lw=1)

    return fig
