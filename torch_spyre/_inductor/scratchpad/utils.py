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
from torch_spyre._inductor.pass_utils import _per_core_view_on_buf, concretize_expr

# Op outputs eligible for LX-pinning. `amax` is the lowered form of
# `max`; both names are listed to match whichever the IR shows.
OP_OUTPUT_GOOD_FOR_LX_REUSE = frozenset(
    {
        "max",
        "amax",
        "maximum",
        "sum",
        # "clone",
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

    Gated by the dedicated ``lx_boundary_clones`` flag (or, legacy, by listing
    "clone" in OP_OUTPUT_GOOD_FOR_LX_REUSE). It intentionally does NOT consult
    ``allow_all_ops_in_lx_planning``: that flag widens intermediate-output
    eligibility and is set broadly (e.g. the LX-planning op suite), so coupling
    it here would silently turn on the not-yet-correct boundary clone path."""
    return config.lx_boundary_clones or "clone" in OP_OUTPUT_GOOD_FOR_LX_REUSE


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

    Note: previously, unused graph inputs did not appear in the returned dict at all.
    Now they appear with an empty list.  Callers that skip buffers with ``len(uses) <= 1``
    (e.g. ``_build_bound_buffers``) will still skip unused inputs correctly, since
    ``len([]) == 0 <= 1``."""
    liveness: dict[str, list[int]] = {}
    for input_name in graph.graph_input_names:
        liveness[input_name] = []
    for i, op in enumerate(graph.operations):
        rw = op.get_read_writes()
        for mem_dep in rw.reads | rw.writes:
            buf_name = mem_dep.name
            if buf_name not in liveness:
                liveness[buf_name] = []
            liveness[buf_name].append(i)
    return liveness


def mem_usage_by_buf(
    graph: GraphLowering | GraphView,
    cache: Optional[dict] = None,
    rw_cache: Optional[dict] = None,
) -> dict:
    """
    Get a summary of memory usage of each operation.
    Includes detailed info of individual buf, e.g. mem_usage[<buf_name>],
    which has "size_per_core", "size", "core_div_mismatch", "op_inputs" fields
    NOTE:
    if a buf is not in core_div_mismatch => it has no users => graph output

    `rw_cache` ({op name: ReadWrites}) memoizes get_read_writes() across
    co-opt search leaves; None recomputes it.
    """
    num_cores_per_op = get_ncores_for_buffers(graph, cache, rw_cache)
    mem_usage: dict = {}

    buf_names = {op.name for op in graph.operations}
    for op in graph.operations:
        buf_name = op.name
        buf = graph.get_buffer(buf_name)
        num_cores = num_cores_per_op.get(buf_name, -1)
        dev_layout = buf.get_layout().device_layout
        dev_size = (
            math.prod(dev_layout.device_size[:-1]) * 128
        )  # num_sticks * bytes_per_stick
        rw = rw_cache[op.get_name()] if rw_cache is not None else op.get_read_writes()
        mem_usage[buf_name] = {
            "size": dev_size,
            "size_per_core": dev_size // num_cores,
            "core_div_mismatch": num_cores < 0,
            "op_inputs": [dep.name for dep in rw.reads if dep.name in buf_names],
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
        for dep in op.get_read_writes().reads:
            if dep.name != buf_name:
                continue
            try:
                if int(dep.get_numel()) < full:
                    return True
            except (TypeError, ValueError, AttributeError):
                return True
    return False


def get_buffer_users(graph: GraphLowering | GraphView) -> dict[str, list[Operation]]:
    buf_users_read_and_write: dict[str, list[Operation]] = {}
    for op in graph.operations:
        rw = op.get_read_writes()
        for dep in rw.reads | rw.writes:  # union of the OrderedSets
            buf = dep.name  # buffer name, i.e. a str
            buf_users_read_and_write[buf] = buf_users_read_and_write.get(buf, []) + [op]
    return buf_users_read_and_write


def _get_buffer_user_deps(
    graph: GraphLowering | GraphView,
    rw_cache: Optional[dict] = None,
) -> dict[str, list[tuple[Operation, MemoryDep]]]:
    """Like get_buffer_users but pairs each op with the specific dep it uses.

    In-place ops (same op reads & writes the same buf) get two entries:
    one per dep. If their per-core views diverge — read at one index,
    write at another — the buffer is correctly rejected for LX, since
    that's a within-core data hazard, not just cross-op disagreement.

    `rw_cache` ({op name: ReadWrites}) memoizes the split-invariant
    get_read_writes() across co-opt search leaves; None recomputes it.
    """
    buf_user_deps: dict[str, list[tuple[Operation, MemoryDep]]] = {}
    for op in graph.operations:
        rw = rw_cache[op.get_name()] if rw_cache is not None else op.get_read_writes()
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
    rw_cache: Optional[dict] = None,
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
    op name and `dep` (which carries the buffer name). `rw_cache`
    ({op name: ReadWrites}) likewise memoizes get_read_writes().

    Pass an optional `reject_reasons_out` dict to receive detailed
    reasons for core division mismatches (keyed by buffer name).
    """
    result: dict[str, int] = {}
    using_multicore = config.sencores > 1
    buf_user_deps = _get_buffer_user_deps(graph, rw_cache)
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
                view, flag = _per_core_view_on_buf(op, dep, buf_name, cache)
                if ref_view is None:
                    ref_view = view
                    ref_op_name = op.get_name()
                op_rw = (
                    rw_cache[op.get_name()]
                    if rw_cache is not None
                    else op.get_read_writes()
                )
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
                    # work_slice_dims entries are (host-dim, (work_div, tile));
                    # the per-dim core count is the work_div factor.
                    view_cores = math.prod(
                        work_div for _, (work_div, _tile) in view.work_slice_dims
                    )
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
