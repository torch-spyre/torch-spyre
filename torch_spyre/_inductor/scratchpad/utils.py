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
from typing import Any

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import Operation, IRNode, Pointwise
from torch._inductor.virtualized import V
from torch._inductor.ops_handler import WrapperHandler

import sympy

from torch_spyre._inductor import config
from torch_spyre._inductor.pass_utils import _per_core_view_on_buf

# Op outputs eligible for LX-pinning. `amax` is the lowered form of
# `max`; both names are listed to match whichever the IR shows.
OP_OUTPUT_GOOD_FOR_LX_REUSE = frozenset(
    {
        "max",
        "amax",
        "sum",
        # "clone",
        "exp",
        "sub",
        "mul",
        "mean",
        "add",
        "rsqrt",
    }
)

OP_GOOD_FOR_LX_INPLACE = frozenset(
    {
        "exp",
        "sub",
        "add",
        "rsqrt",
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


def mem_usage_by_buf(graph: GraphLowering | GraphView) -> dict:
    """
    Get a summary of memory usage of each operation.
    Includes detailed info of individual buf, e.g. mem_usage[<buf_name>],
    which has "size_per_core", "size", "core_div_mismatch", "op_inputs" fields
    NOTE:
    if a buf is not in core_div_mismatch => it has no users => graph output
    """
    num_cores_per_op = get_ncores_for_buffers(graph)
    mem_usage: dict = {}

    buf_names = {op.name for op in graph.operations}
    for op in graph.operations:
        buf_name = op.name
        buf = graph.get_buffer(buf_name)
        num_cores = num_cores_per_op.get(buf_name, -1)
        dev_layout = buf.layout.device_layout
        dev_size = (
            math.prod(dev_layout.device_size[:-1]) * 128
        )  # num_sticks * bytes_per_stick
        rw = op.get_read_writes()
        mem_usage[buf_name] = {
            "size": dev_size,
            "size_per_core": dev_size // num_cores,
            "core_div_mismatch": num_cores < 0,
            "op_inputs": [dep.name for dep in rw.reads if dep.name in buf_names],
        }

    return mem_usage


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
) -> dict[str, list[tuple[Operation, MemoryDep]]]:
    """Like get_buffer_users but pairs each op with the specific dep it uses.

    In-place ops (same op reads & writes the same buf) get two entries:
    one per dep. If their per-core views diverge — read at one index,
    write at another — the buffer is correctly rejected for LX, since
    that's a within-core data hazard, not just cross-op disagreement.
    """
    buf_user_deps: dict[str, list[tuple[Operation, MemoryDep]]] = {}
    for op in graph.operations:
        rw = op.get_read_writes()
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


def get_ncores_for_buffers(graph: GraphLowering | GraphView) -> dict[str, int]:
    """
    Return a dictionary mapping buffer names to the number of cores
    used by all the operations that uses the buffer.
    If there is a core division mismatch return -1 instead of the
    number of cores.
    """
    result: dict[str, int] = {}
    using_multicore = config.sencores > 1
    buf_user_deps = _get_buffer_user_deps(graph)
    for buf_name, users in buf_user_deps.items():
        # this dict includes graph input and output
        if using_multicore and len(users) > 1:
            # K-split-reduction producers leave partial sums on most cores;
            # only k-last cores hold the final value. Without a broadcast
            # codepath the buffer is not safe on LX, even if work-slice
            # geometry happens to match. The flag is meaningful only for
            # write-deps — a consumer reading a K-split input still gets
            # its own valid work slice.
            ref_view = None
            mismatch = False
            for op, dep in users:
                view, flag = _per_core_view_on_buf(op, dep, buf_name)
                if ref_view is None:
                    ref_view = view
                if (flag and dep in op.get_read_writes().writes) or (view != ref_view):
                    mismatch = True
                    break
            num_cores = -1 if mismatch else max(_op_num_cores(op) for op, _ in users)
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
