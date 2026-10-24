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


import abc
import itertools
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import sympy

from .logging_utils import get_inductor_logger

from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    InputBuffer,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V
from torch_spyre._C import SpyreTensorLayout
from . import config
from .pass_utils import (
    compute_restickify_needed,
    device_coordinates,
    host_coordinates,
)

INF = math.inf

logger = get_inductor_logger("optimize_restickify")

try:
    from ortools.sat.python import cp_model as _cp_model
except ImportError:  # pragma: no cover
    _cp_model = None  # type: ignore[assignment]


class EdgeCostMap:
    """Lazy cost table mapping (in_layout, target_layout) -> restick cost for one op input.

    Entries are computed on demand by compute_restickify_needed. `dep` is the
    MemoryDep for this input; it is not used locally but is forwarded to
    compute_restickify_needed in pass_utils.
    """

    def __init__(
        self,
        dep: "MemoryDep",
        in_layouts: list,
        target_layouts: list,
        target_dep: "MemoryDep",
        op,
        forbidden_stick_sym: "sympy.Symbol | None" = None,
    ):
        self.dep = dep
        self._op = op
        self._in_layouts = in_layouts
        self._target_layouts = target_layouts
        self._target_dep = target_dep
        self._dep_layout = V.graph.get_buffer(dep.name).get_layout()
        self._target_dep_layout = V.graph.get_buffer(target_dep.name).get_layout()
        self._forbidden_stick_sym = forbidden_stick_sym

        # _cost and _layout are parallel maps.
        # _cost stores the cost for a given in/target layout pair
        # _layout stores the target STL for the restickify, or None if no restickify is needed
        self._cost: defaultdict[SpyreTensorLayout, dict[SpyreTensorLayout, float]] = (
            defaultdict(dict)
        )
        self._layout: defaultdict[SpyreTensorLayout, dict[SpyreTensorLayout, Any]] = (
            defaultdict(dict)
        )

    # Sentinel stored in _layout when a restickify is needed but infeasible.
    # Distinct from None ("compatible, no restickify needed") so finalize_layouts
    # can raise instead of silently skipping.
    INFEASIBLE: "SpyreTensorLayout" = object()  # type: ignore[assignment]

    def _compute_and_cache_cost(
        self, in_stl: "SpyreTensorLayout", target_stl: "SpyreTensorLayout"
    ) -> None:
        """Populate _cost and _layout for (in_stl, target_stl).

        Cost is 0 if stick-compatible, the input element count if restickifiable, or INF if infeasible.
        _layout stores:
          None               — compatible, no restickify needed
          INFEASIBLE         — restickify needed but compute_restickify_target_layout returned None
          SpyreTensorLayout  — feasible restickify target layout
        """
        needed, tgt = compute_restickify_needed(
            in_stl, self._dep_layout, self.dep, target_stl, self._target_dep, self._op
        )
        if not needed and self._forbidden_stick_sym is not None:
            stick_expr = device_coordinates(in_stl, self.dep, None)[-1]
            if self._forbidden_stick_sym in stick_expr.free_symbols:
                # Input stick contains a forbidden symbol (e.g. topk reduction
                # var): this pairing is invalid even though stick_compatible
                # accepted it. Mark as infeasible so the optimizer is forced to
                # pick a different input candidate or output STL.
                needed, tgt = True, None
        if not needed:
            cost = 0.0
            self._layout[in_stl][target_stl] = None
        elif tgt is None:
            cost = INF  # infeasible restickify
            self._layout[in_stl][target_stl] = EdgeCostMap.INFEASIBLE
        else:
            cost = float(math.prod(in_stl.device_size))
            self._layout[in_stl][target_stl] = tgt
        self._cost[in_stl][target_stl] = cost

    def cost(
        self, in_stl: "SpyreTensorLayout", target_stl: "SpyreTensorLayout"
    ) -> float:
        """Return the restick cost for (in_stl, target_stl), computing it on first access."""
        if target_stl not in self._cost[in_stl]:
            self._compute_and_cache_cost(in_stl, target_stl)
        return self._cost[in_stl][target_stl]

    def layout(
        self, in_stl: "SpyreTensorLayout", target_stl: "SpyreTensorLayout"
    ) -> "SpyreTensorLayout | None":
        """Return target STL for restickifying in_stl to be compatible with target_stl.

        Returns:
          None               — compatible, no restickify needed
          INFEASIBLE         — restickify needed but infeasible (caller must handle)
          SpyreTensorLayout  — the target layout for the restickify op
        """
        if target_stl not in self._cost[in_stl]:
            self._compute_and_cache_cost(in_stl, target_stl)
        return self._layout[in_stl][target_stl]


class RestickNodeCost(abc.ABC):
    """Abstract base for per-op restick cost functions.

    Subclasses encode the stick-compatibility rules for a specific op type and
    compute the total restick cost given each input's committed layout and a
    candidate output layout key.
    """

    def __init__(self, edge_costs):
        self.edge_costs = edge_costs

    @abc.abstractmethod
    def cost(
        self, in_layouts: "list[SpyreTensorLayout]", out_stl: "SpyreTensorLayout"
    ) -> float: ...

    @abc.abstractmethod
    def required_input_stls(
        self, out_stl: "SpyreTensorLayout"
    ) -> "list[tuple[EdgeCostMap, SpyreTensorLayout]]":
        """Return (edge_cost, required_input_stl) pairs for finalize_layouts to schedule restickifies."""
        ...

    @abc.abstractmethod
    def min_input_cost(
        self, dep_name: str, in_stl: "SpyreTensorLayout", out_stl: "SpyreTensorLayout"
    ) -> float:
        """Cost contribution from input dep_name (with in_stl) toward output candidate out_stl.

        Used by the backward DP, which knows one input's STL at a time but not the others.
        Returns INF if out_stl is infeasible — either this input can't be restickified,
        or some other input has no feasible STL for out_stl.
        """
        ...

    def first_blocking_edge(self, out_stl: "SpyreTensorLayout") -> "EdgeCostMap | None":
        """Return the first EdgeCostMap that has at least one input STL with infinite cost against out_stl.

        Only the first blocking edge is returned. For ops with multiple inputs, additional
        blocking edges are not reported.
        """
        for ec in self.edge_costs:
            if any(ec.cost(in_stl, out_stl) == INF for in_stl in ec._in_layouts):
                return ec
        return None


class AllSameNode(RestickNodeCost):
    """Cost node for ops that require all inputs and outputs to share the same stick layout.

    Accepts multiple output deps via out_deps (e.g. mutation ops where two ops
    write the same buffer). Restickify may only be inserted on input edges;
    co-output edges enforce layout equality but never trigger restickify insertion.
    """

    @classmethod
    def from_args(
        cls,
        args,
        out_layouts,
        out_deps,
        op,
        forbidden_stick_sym: "sympy.Symbol | None" = None,
    ):
        """Build an AllSameNode from input PropArgs and output dep(s).

        out_deps is either a single MemoryDep (normal ops) or a list whose first
        entry is the primary output dep and whose remaining entries are co-output
        MemoryDeps (e.g. the shared mutation buffer in copy_forced). Co-output deps
        must agree on the same layout but are not eligible for restickify insertion.

        forbidden_stick_sym: if set, any input whose stick expression contains this
        symbol is treated as incompatible regardless of stick_compatible() — the
        optimizer is forced to restickify or pick a different layout.
        """
        assert out_layouts, "AllSameNode.from_args: out_layouts is empty"
        if not isinstance(out_deps, list):
            out_deps = [out_deps]
        out_dep = out_deps[0]  # reference output dep for stick-compatibility checks
        co_output_deps = out_deps[1:]
        input_edge_costs = [
            EdgeCostMap(
                arg.dep, arg.layouts, out_layouts, out_dep, op, forbidden_stick_sym
            )
            for arg in args
        ]
        output_edge_costs = [
            EdgeCostMap(
                dep,
                # Always non-empty: SpyreEmptyFallback.layouts is set by the
                # SpyreEmptyFallback branch in propagate_layouts before any
                # mutation writer is processed (topo order guarantee). An empty
                # list here would cause min_input_cost to return INF for all
                # beam states with no useful error message.
                getattr(V.graph.get_buffer(dep.name), "layouts", []),
                out_layouts,
                out_dep,
                op,
            )
            for dep in co_output_deps
        ]
        return cls(input_edge_costs, output_edge_costs)

    def __init__(self, input_edge_costs: list, output_edge_costs: "list | None" = None):
        super().__init__(input_edge_costs + (output_edge_costs or []))
        self._input_edge_costs = input_edge_costs
        self._output_edge_costs = output_edge_costs or []

    def cost(
        self, in_layouts: "list[SpyreTensorLayout]", out_stl: "SpyreTensorLayout"
    ) -> float:
        input_cost = sum(
            ec.cost(lk, out_stl)
            for ec, lk in zip(
                self._input_edge_costs, in_layouts[: len(self._input_edge_costs)]
            )
        )
        if input_cost >= INF:
            return INF
        # Co-output edges: the shared buffer must have the exact same STL as this op's
        # output. Any mismatch means two mutation ops write the same buffer with different
        # layouts, which is always wrong — cost INF, not a restickify.
        co_offset = len(self._input_edge_costs)
        for ec, lk in zip(self._output_edge_costs, in_layouts[co_offset:]):
            if lk is not None and lk != out_stl:
                return INF
        return input_cost

    def required_input_stls(self, out_stl):
        return [(ec, out_stl) for ec in self._input_edge_costs]

    def min_input_cost(self, dep_name, in_stl, out_stl):
        # next() takes the first match; if dep_name appears twice (x+x), both edges
        # are identical so the cost is the same either way.
        ec = next(e for e in self.edge_costs if e.dep.name == dep_name)
        edge_c = ec.cost(in_stl, out_stl)
        if edge_c == INF:
            return INF
        other_ok = all(
            any(e.cost(other_c, out_stl) < INF for other_c in e._in_layouts)
            for e in self.edge_costs
            if e.dep.name != dep_name
        )
        return edge_c if other_ok else INF


class FixedInOutNode(RestickNodeCost):
    """Cost node for ops whose input and output stick compatibility is fixed by the op (eg, matmul)."""

    def __init__(
        self,
        edge_costs,
        required_out_stl: "SpyreTensorLayout",
        required_in_stls: "list[SpyreTensorLayout]",
        out_dep: "MemoryDep",
    ):
        super().__init__(edge_costs)
        self.required_out_stl = required_out_stl
        # Parallel to edge_costs by construction in from_args (both built from the
        # same zip over args/req_stls). strict=True in min_input_cost asserts this.
        self.required_in_stls = required_in_stls
        self._out_dep = out_dep
        self._out_host = V.graph.get_buffer(out_dep.name).get_layout()

    @classmethod
    def from_args(cls, args, out_stl, req_stls, op, out_dep):
        assert req_stls, "FixedInOutNode.from_args: req_stls is empty"
        edge_costs = [
            EdgeCostMap(arg.dep, arg.layouts, [req], arg.dep, op)
            for arg, req in zip(args, req_stls)
        ]
        return cls(
            edge_costs,
            required_out_stl=out_stl,
            required_in_stls=req_stls,
            out_dep=out_dep,
        )

    def _out_stick_compatible(self, out_stl: "SpyreTensorLayout") -> bool:
        """True if out_stl is stick-compatible with required_out_stl."""
        if out_stl == self.required_out_stl:
            return True
        needed, _ = compute_restickify_needed(
            self.required_out_stl,
            self._out_host,
            self._out_dep,
            out_stl,
            self._out_dep,
            None,
        )
        return not needed

    def cost(
        self, in_layouts: "list[SpyreTensorLayout]", out_stl: "SpyreTensorLayout"
    ) -> float:
        if not self._out_stick_compatible(out_stl):
            return INF
        return sum(
            ec.cost(lk, rk)
            for ec, lk, rk in zip(self.edge_costs, in_layouts, self.required_in_stls)
        )

    def required_input_stls(self, out_stl):
        return list(zip(self.edge_costs, self.required_in_stls))

    def min_input_cost(self, dep_name, in_stl, out_stl):
        if not self._out_stick_compatible(out_stl):
            return INF
        matching = [
            (ec, req)
            for ec, req in zip(self.edge_costs, self.required_in_stls, strict=True)
            if ec.dep.name == dep_name
        ]
        if not matching:
            return INF
        costs = [ec.cost(in_stl, req) for ec, req in matching]
        if any(c == INF for c in costs):
            return INF
        other_ok = all(
            any(e.cost(other_c, req) < INF for other_c in e._in_layouts)
            for e, req in zip(self.edge_costs, self.required_in_stls, strict=True)
            if e.dep.name != dep_name
        )
        return sum(costs) if other_ok else INF


class AnyInNode(RestickNodeCost):
    """Cost node for ops that accept any input layout and produce a fixed output layout.

    Eg, aten.clone.default: the clone become a restickify when sticks are incompatible
    so no restickify is ever needed before it.
    """

    @classmethod
    def from_args(cls):
        return cls(edge_costs=[])

    def cost(
        self, in_layouts: "list[SpyreTensorLayout]", out_stl: "SpyreTensorLayout"
    ) -> float:
        return 0.0

    def required_input_stls(self, out_stl):
        return []

    def min_input_cost(self, dep_name, in_stl, out_stl):
        return 0.0


def _stick_incompatibility_reason(
    in_stick: "sympy.Expr",
    out_stick: "sympy.Expr",
) -> "str | None":
    """Return a human-readable reason why two tensors are stick-incompatible, or None."""
    in_zero = in_stick == sympy.S.Zero
    out_zero = out_stick == sympy.S.Zero
    if in_zero and not out_zero:
        return "No mechanism to gather elements from multiple sticks into single stick"
    if out_zero and not in_zero:
        return "No mechanism to scatter elements from one stick to multiple sticks"
    return None


def _fmt_buf(layout: Any, dep: "MemoryDep") -> str:
    h_coords = host_coordinates(layout, dep, None)
    return (
        f"size={list(layout.size)}  stride={list(layout.stride)}  h_coords={h_coords}"
    )


def _fmt_stl(d_coords: Any, stl: "SpyreTensorLayout") -> str:
    return (
        f"device_size={list(stl.device_size)}  stride_map={list(stl.stride_map)}"
        f"  dtype={stl.device_dtype}  d_coords={d_coords}"
    )


def _no_feasible_layout_error(op) -> NotImplementedError:
    """Build and return a NotImplementedError describing why no output layout was feasible."""
    node_type = type(getattr(op, "data", op)).__name__
    out_layout = op.get_layout()
    out_dep = next(iter(op.get_read_writes().writes))
    edge_costs = op.restick_cost_fn.edge_costs

    lines = [
        f"{op.get_name()} ({node_type}): no mechanism to resolve stick incompatibility",
        "  Inputs:",
        "",
    ]
    for ec in edge_costs:
        host_layout = V.graph.get_buffer(ec.dep.name).get_layout()
        lines.append(f"    {ec.dep.name}:  {_fmt_buf(host_layout, ec.dep)}")
        for j, stl in enumerate(ec._in_layouts):
            lines.append(
                f"      STL {j}:  {_fmt_stl(device_coordinates(stl, ec.dep, None), stl)}"
            )
        lines.append("")

    lines.append(f"  Output:  {_fmt_buf(out_layout, out_dep)}")
    for i, stl in enumerate(op.layouts):
        lines.append(
            f"    STL {i}:  {_fmt_stl(device_coordinates(stl, out_dep, None), stl)}"
        )

    analysis = []
    for i, candidate_stl in enumerate(op.layouts):
        blocking_ec = op.restick_cost_fn.first_blocking_edge(candidate_stl)
        if blocking_ec is None:
            analysis.append(f"    STL {i}: no blocking input identified")
        else:
            out_stick = device_coordinates(candidate_stl, out_dep, None)[-1]
            for j, in_stl in enumerate(blocking_ec._in_layouts):
                if blocking_ec.cost(in_stl, candidate_stl) == INF:
                    in_stick = device_coordinates(in_stl, blocking_ec.dep, None)[-1]
                    reason = _stick_incompatibility_reason(in_stick, out_stick)
                    reason_str = f": {reason}" if reason else ""
                    analysis.append(
                        f"    {blocking_ec.dep.name} STL {j} --> Out STL {i}{reason_str}"
                    )
    lines += ["", "  Problem:"]
    lines += analysis if analysis else ["    No automated triage available"]
    return NotImplementedError("\n".join(lines))


# Global Stick Optimizer
#
# The global optimizer is a simple forward-propagation algorithm that tracks a frontier of possible
# "states" and their corresponding cost. A state is a combination of concrete restickify decisions
# that have been made so far. The cost is a proxy for the runtime cost of executing those restickify
# decisions.
#
# The number of states can grow exponentially. To prevent this blow-up the number of states is bounded
# by a "beam width". When beam width is exceeded, the highest cost states are trimmed. Optimal cost is
# only achieved if the optimal state always remains in the beam.
#
# A*-style backward pass: before the forward beam runs, compute_future_min_cost does a backward DP
# over the op graph to estimate the minimum remaining cost achievable from each (op, candidate_stl)
# pair onward. The forward beam then trims by lower_bound = cost_so_far + future_min_cost instead of
# cost_so_far alone. This is admissible: the backward DP independently minimizes each downstream
# consumer's cost (treating other inputs optimistically), so it underestimates true remaining cost.
# States with a high future cost (e.g. buf30=cand1 which leads to INF at a join) are de-prioritized
# relative to states with low future cost, even if their cost_so_far is currently lower.
#
# Liveness merge: after expanding states for each op, states whose live buffer slots are identical
# are equivalent (their futures are identical). Only the lowest-lower_bound one is kept per live key.
# A slot is live if its last downstream consumer has not yet been committed.


@dataclass
class BeamState:
    """One hypothesis in the beam: a partial assignment of STLs to ops, with accumulated cost.

    assignments is a tuple parallel to a shared buf_names list — index i holds the
    chosen SpyreTensorLayout for buf_names[i], or None for passthrough ops.
    lower_bound = cost + future_min_cost for the last assigned op's candidate.
    """

    assignments: tuple  # tuple[SpyreTensorLayout | None, ...]
    cost: float
    lower_bound: float = 0.0


BEAM_WIDTH = 200
MAX_BEAM_STATES_LOGGED = 10


class Frontier:
    """Beam search frontier: shared buf_names index plus a list of BeamStates."""

    def __init__(self, K: int):
        self.K = K
        self.buf_names: list[str] = []  # parallel index for BeamState.assignments
        self._buf_idx: dict[str, int] = {}  # name -> index into buf_names
        self.states: list[BeamState] = [BeamState(assignments=(), cost=0.0)]

    def add_buf(self, name: str) -> None:
        self._buf_idx[name] = len(self.buf_names)
        self.buf_names.append(name)

    def input_stl(self, state: BeamState, name: str) -> "SpyreTensorLayout | None":
        """Return the hypothesized STL for an input buffer in this state."""
        idx = self._buf_idx[name]
        return state.assignments[idx]

    def best(self) -> BeamState:
        # At end of search, future costs are 0; use actual cost for final selection.
        return min(self.states, key=lambda s: s.cost)

    def trim(self) -> None:
        self.states.sort(key=lambda s: s.lower_bound)
        before = len(self.states)
        self.states = self.states[: self.K]
        if len(self.states) < before:
            logger.debug(
                "beam trimmed: %d -> %d states (beam_width=%d)",
                before,
                len(self.states),
                self.K,
            )


def _reorder_any_in_nodes(operations: list) -> list:
    """Move AnyInNode ops to just before their first consumer.

    AnyInNode ops (e.g. SpyreEmptyFallback) have no inputs and impose no
    upstream constraints. Committing their layout early causes speculative
    branching that persists until their consumer is reached — potentially
    across many beam steps, blowing up the state count. Moving them to just
    before their first consumer means the branch is immediately resolved by
    the consumer's cost function, eliminating the blowup.
    """
    # For each AnyInNode op, find the position of its first consumer.
    to_move: dict[int, int] = {}  # old_pos -> insert_before_pos
    for i, op in enumerate(operations):
        if not hasattr(op, "layouts"):
            continue
        if not isinstance(op.restick_cost_fn, AnyInNode):
            continue
        name = op.get_name()
        first_consumer_pos = None
        for j, other in enumerate(operations):
            if j <= i:
                continue
            if not hasattr(other, "layouts"):
                continue
            # NOTE: AnyInNode.edge_costs is always [], so an AnyInNode op can
            # never appear as a consumer here. In practice SpyreEmptyFallback
            # (the only current AnyInNode user) has no inputs and cannot consume
            # another SpyreEmptyFallback, so chained AnyInNode ops cannot occur.
            # If new AnyInNode users are added, generalize this into a dedicated
            # reorder pass that handles chained AnyInNode ops.
            # NOTE: This is O(k·n) where k is the number of AnyInNode ops.
            # Since SpyreEmptyFallback buffers are rare, this is effectively O(n).
            if any(ec.dep.name == name for ec in other.restick_cost_fn.edge_costs):
                first_consumer_pos = j
                break
        if first_consumer_pos is not None and first_consumer_pos > i + 1:
            to_move[i] = first_consumer_pos

    if not to_move:
        return operations

    # Build reordered list: skip moved ops in original positions, insert at target.
    moved_ops = {i: operations[i] for i in to_move}
    result = []
    for i, op in enumerate(operations):
        if i in to_move:
            continue
        # Insert any ops whose target position is here (i.e. just before this op).
        for old_pos, insert_before in sorted(to_move.items()):
            if insert_before == i:
                result.append(moved_ops[old_pos])
        result.append(op)
    # Handle any ops targeted past the end.
    for old_pos, insert_before in sorted(to_move.items()):
        if insert_before >= len(operations):
            result.append(moved_ops[old_pos])

    return result


def compute_future_min_cost(
    operations: list,
) -> dict:
    """Backward DP: returns future_min_cost[op_name][stl] = admissible lower bound on
    remaining restickify cost from this op onward, if this op commits to stl.

    Processes ops in reverse topological order. For each op and each of its output
    candidates, sums over all downstream consumers the minimum edge cost achievable
    from that candidate to any of the consumer's output candidates (recursively).

    The bound is admissible (never overestimates) because each downstream consumer's
    cost is minimized independently — ignoring cross-consumer pairwise constraints.
    """
    downstream: dict[str, list] = defaultdict(list)
    downstream_seen: dict[str, set] = defaultdict(set)
    for op in operations:
        if not hasattr(op, "layouts"):
            continue
        for ec in op.restick_cost_fn.edge_costs:
            dep = ec.dep
            if op.get_name() not in downstream_seen[dep.name]:
                downstream[dep.name].append(op)
                downstream_seen[dep.name].add(op.get_name())

    future: dict[str, dict] = {}  # op_name -> {stl -> float}

    for op in reversed(operations):
        if not hasattr(op, "layouts"):
            continue
        name = op.get_name()
        future[name] = {}

        for candidate in op.layouts:
            total_future = 0.0
            for d_op in downstream.get(name, []):
                if not hasattr(d_op, "layouts"):
                    continue
                cost_fn = d_op.restick_cost_fn
                if not any(e.dep.name == name for e in cost_fn.edge_costs):
                    continue
                best = INF
                for d_cand in d_op.layouts:
                    edge_c = cost_fn.min_input_cost(name, candidate, d_cand)
                    if edge_c == INF:
                        continue
                    # d_op is always in future (reverse topo order); .get fallback is defensive.
                    tail = future.get(d_op.get_name(), {}).get(d_cand, 0.0)
                    best = min(best, edge_c + tail)
                # Cap INF at a large finite value so it de-prioritizes but doesn't hard-block.
                if best == INF:
                    best = 1e18  # large but finite: larger than any real cost, smaller than math.inf
                total_future += best

            future[name][candidate] = total_future

    return future


def _compute_last_use(operations: list, step_of: "dict[str, int]") -> "dict[str, int]":
    """Return last_use[op_name] = step of its last consumer. Graph outputs are absent;
    callers use .get(name, -1) so absent entries never satisfy > current_step."""
    last_use: dict[str, int] = {}
    for op in operations:
        if not hasattr(op, "layouts"):
            continue
        for ec in op.restick_cost_fn.edge_costs:
            if ec.dep.name in step_of:
                consumer_step = step_of[op.get_name()]
                if last_use.get(ec.dep.name, -1) < consumer_step:
                    last_use[ec.dep.name] = consumer_step
    return last_use


def beam_global_min_cost(operations: list) -> None:
    """Global beam search layout selection.

    Processes ops in topological order. For each op with a restick_cost_fn,
    expands every current state by branching over candidate output STLs and
    accumulating cost. After each op the beam is pruned to K best states
    sorted by lower_bound = cost_so_far + future_min_cost (A*-style).

    After expansion, states whose live assignments are identical are merged
    (keeping only the lowest lower_bound one). A slot is live if its last
    downstream consumer has not yet been committed.

    At the end, the best state's assignments are committed to the ops.
    """
    operations = _reorder_any_in_nodes(operations)
    # NOTE: compute_future_min_cost and _compute_last_use both iterate over
    # op.restick_cost_fn.edge_costs, which on AllSameNode includes co-output
    # EdgeCostMap entries (the SpyreEmptyFallback buffer dep). This means each
    # mutation writer appears as a "downstream consumer" of the fallback in the
    # future-cost and last-use maps. This is safe in practice: co-output edges
    # have zero cost when STLs match, and SpyreEmptyFallback has all valid STLs
    # as candidates, so the future-min-cost estimate is never pessimistic in a
    # way that causes incorrect pruning. Last-use liveness is also correct — the
    # fallback should stay live until its last writer. A cleaner fix would
    # exclude co-output deps from the base class edge_costs; cleanup is coming.
    future_min_cost = compute_future_min_cost(operations)

    step_of: dict[str, int] = {}
    step_counter = 0
    for op in operations:
        if hasattr(op, "layouts"):
            step_of[op.get_name()] = step_counter
            step_counter += 1
    last_use = _compute_last_use(operations, step_of)

    frontier = Frontier(BEAM_WIDTH)
    # Commit graph inputs and seed into the frontier so input_stl() works uniformly for all deps.
    total_inp_future = 0.0
    for name in V.graph.graph_input_names:
        tb = V.graph.graph_inputs[name]
        if (
            isinstance(tb, TensorBox)
            and isinstance(tb.data, StorageBox)
            and isinstance(tb.data.data, InputBuffer)
            and hasattr(tb, "layouts")
        ):
            stl = next(iter(tb.layouts))
            tb.data.data.committed_stl = stl
            frontier.add_buf(name)
            total_inp_future += future_min_cost.get(name, {}).get(stl, 0.0)
            frontier.states = [
                BeamState(
                    assignments=state.assignments + (stl,),
                    cost=state.cost,
                    lower_bound=state.cost,
                )
                for state in frontier.states
            ]
    frontier.states = [
        BeamState(
            assignments=state.assignments,
            cost=state.cost,
            lower_bound=state.cost + total_inp_future,
        )
        for state in frontier.states
    ]

    max_states = 1
    merged_total = 0

    for op in operations:
        if not hasattr(op, "layouts"):
            continue

        current_step = step_of[op.get_name()]
        frontier.add_buf(op.get_name())

        assert hasattr(op, "restick_cost_fn"), (
            f"op {op.get_name()} has layouts but no restick_cost_fn"
        )
        cost_fn = op.restick_cost_fn
        deps = [ec.dep for ec in op.restick_cost_fn.edge_costs]

        op_future = future_min_cost.get(op.get_name(), {})
        next_states = []
        for state in frontier.states:
            in_layouts = [frontier.input_stl(state, dep.name) for dep in deps]

            for candidate_stl in op.layouts:
                extra_cost = cost_fn.cost(in_layouts, candidate_stl)
                if extra_cost < INF:
                    new_cost = state.cost + extra_cost
                    lb = new_cost + op_future.get(candidate_stl, 0.0)
                    next_states.append(
                        BeamState(
                            assignments=state.assignments + (candidate_stl,),
                            cost=new_cost,
                            lower_bound=lb,
                        )
                    )

        # Liveness merge: keep only the lowest-lower_bound state per live-slot key.
        # Absent from last_use = graph output; treated as dead (cost already sunk).
        # Co-output slots written by this op are kept live in the key so that states
        # where those slots differ are not incorrectly merged: each expansion of this
        # op writes co_output[i] = committed_stl, and we must preserve those distinct
        # values across states until the beam can prune the dominated ones.
        live_indices = frozenset(
            i
            for i, name in enumerate(frontier.buf_names)
            if last_use.get(name, -1) > current_step
        )
        before_merge = len(next_states)
        canon: dict[tuple, BeamState] = {}
        for s in next_states:
            key = tuple(
                s.assignments[i] if i in live_indices else None
                for i in range(len(s.assignments))
            )
            if key not in canon or s.lower_bound < canon[key].lower_bound:
                canon[key] = s
        next_states = list(canon.values())
        merged = before_merge - len(next_states)
        merged_total += merged
        if merged > 0:
            logger.debug(
                "liveness merge after %s: %d -> %d states (%d merged, %d live slots / %d total)",
                op.get_name(),
                before_merge,
                len(next_states),
                merged,
                len(live_indices),
                len(frontier.buf_names),
            )

        frontier.states = next_states
        frontier.trim()
        if not frontier.states:
            raise _no_feasible_layout_error(op)
        max_states = max(max_states, len(frontier.states))
        if logger.isEnabledFor(logging.DEBUG):
            lines = [f"beam after {op.get_name()} [{len(frontier.states)} states]:"]
            for i, s in enumerate(frontier.states[:MAX_BEAM_STATES_LOGGED]):
                lines.append(f"  state {i} (cost={s.cost}):")
                for name, stl in zip(frontier.buf_names, s.assignments):
                    lines.append(f"    {name}: stride_map={list(stl.stride_map)}")
            extra = len(frontier.states) - MAX_BEAM_STATES_LOGGED
            if extra > 0:
                lines.append(f"    ... {extra} additional states not logged")
            logger.debug("\n".join(lines))

    logger.info(
        "beam search done: max states = %d, best cost = %s, total liveness-merged = %d",
        max_states,
        frontier.best().cost,
        merged_total,
    )

    # Commit the best state's assignments to all ops.
    best = frontier.best()
    for name, stl in zip(frontier.buf_names, best.assignments):
        op = V.graph.get_buffer(name)
        op.committed_stl = stl


def _cur_layouts(ec: "EdgeCostMap") -> list:
    """Return the dep buffer's current .layouts list, falling back to the snapshot."""
    dep_buf = V.graph.get_buffer(ec.dep.name)
    cur = getattr(dep_buf, "layouts", None)
    return list(cur) if cur else list(ec._in_layouts)


def _build_node_tables(operations: list) -> list:
    """Build feasible-assignment tables for CP-SAT model construction.

    Returns node_tables: list of (op, table) pairs for each op (with layouts
    and a non-AnyInNode cost_fn), where table is a list of tuples
    (i1, i2, ..., j, cost):
      - i1..iN are indices into each input buffer's CURRENT .layouts list
      - j is an index into the op's own .layouts list (output candidate index)
      - cost is the total restickify cost the op pays for this combination
      - INF combinations are excluded

    Uses the dep buffer's *current* .layouts (not ec._in_layouts snapshot) so
    that layouts added by later propagation passes are included.  This prevents
    CP-SAT from seeing a stale, under-constrained table that is missing valid
    low-cost assignments discovered after snapshot time.

    A list is used instead of a dict because ComputedBuffer is not hashable.

    Raises NotImplementedError (via _no_feasible_layout_error) if any op has an
    empty table -- the graph is infeasible before the solver even runs.
    """
    node_tables: list = []

    for op in operations:
        if not hasattr(op, "layouts"):
            continue
        cost_fn = op.restick_cost_fn
        if isinstance(cost_fn, AnyInNode):
            continue  # zero cost, no constraint
        if isinstance(cost_fn, FixedInOutNode):
            input_stl_lists = [_cur_layouts(ec) for ec in cost_fn.edge_costs]
            table = []
            for in_indices in itertools.product(
                *[range(len(stls)) for stls in input_stl_lists]
            ):
                in_stls = [
                    input_stl_lists[k][in_indices[k]] for k in range(len(in_indices))
                ]
                cost = cost_fn.cost(in_stls, cost_fn.required_out_stl)
                if cost < INF:
                    table.append(in_indices + (0, cost))
            if not table:
                raise _no_feasible_layout_error(op)
            node_tables.append((op, table))
            continue

        # AllSameNode: enumerate all (in_combo, out_candidate) pairs.
        # Use only the real input edges (_input_edge_costs), not co-output edges.
        # Co-output equality is enforced via separate constraints in the CP-SAT model
        # (add_allowed_assignments on the co-output var and op output var pairs), so
        # indices here must correspond to each dep's current .layouts list, not the
        # snapshot, to stay consistent with the CP-SAT layout variables.
        if isinstance(cost_fn, AllSameNode) and cost_fn._output_edge_costs:
            input_stl_lists = [_cur_layouts(ec) for ec in cost_fn._input_edge_costs]
            table = []
            for in_indices in itertools.product(
                *[range(len(stls)) for stls in input_stl_lists]
            ):
                in_stls = [
                    input_stl_lists[k][in_indices[k]] for k in range(len(in_indices))
                ]
                # Pass empty co-output in_layouts so cost() skips co-output checks.
                all_in_stls = in_stls + [None] * len(cost_fn._output_edge_costs)
                for j, out_stl in enumerate(op.layouts):
                    cost = cost_fn.cost(all_in_stls, out_stl)
                    if cost < INF:
                        table.append(in_indices + (j, cost))
        else:
            input_stl_lists = [_cur_layouts(ec) for ec in cost_fn.edge_costs]
            table = []
            for in_indices in itertools.product(
                *[range(len(stls)) for stls in input_stl_lists]
            ):
                in_stls = [
                    input_stl_lists[k][in_indices[k]] for k in range(len(in_indices))
                ]
                for j, out_stl in enumerate(op.layouts):
                    cost = cost_fn.cost(in_stls, out_stl)
                    if cost < INF:
                        table.append(in_indices + (j, cost))

        if not table:
            raise _no_feasible_layout_error(op)
        node_tables.append((op, table))

    return node_tables


def cp_sat_stick_optimizer(operations: list) -> None:
    """CP-SAT layout optimizer — exact global minimum within a time limit.

    Builds a CP-SAT model over all buffers with layout choices, then solves for
    the globally optimal STL assignment that minimizes total restickify cost.

    Variables: one integer variable per buffer indexed into buf.layouts.
    Constraints: add_allowed_assignments per op (feasibility table).
    Objective: sum of per-op cost vars, each tied via add_element to the
               flattened input/output index combination.
    """
    if _cp_model is None:
        raise ImportError(
            "OR-Tools is required for cp_sat_stick_optimizer. "
            "Install with: pip install ortools"
        )

    operations = _reorder_any_in_nodes(operations)

    # Step 1: Build node tables (feasible assignment tuples per op).
    node_tables = _build_node_tables(operations)

    # Step 2: Build CP-SAT model.
    model = _cp_model.CpModel()

    # layout_var[buf_name] = IntVar with domain [0, len(buf.layouts)-1]
    # All consumers of the same buffer share this variable — multi-consumer invariant.
    layout_var: dict = {}

    # Commit graph inputs first (single candidate each).
    # Also set committed_stl on the TensorBox itself so downstream passes that call
    # V.graph.get_buffer(name) (which returns TensorBox, not InputBuffer) can find it.
    # Beam does this via its generic "op.committed_stl = stl" loop; CP-SAT must be
    # explicit here because graph inputs are not in the operations list.
    for name in V.graph.graph_input_names:
        tb = V.graph.graph_inputs[name]
        if (
            isinstance(tb, TensorBox)
            and isinstance(tb.data, StorageBox)
            and isinstance(tb.data.data, InputBuffer)
            and hasattr(tb, "layouts")
        ):
            stl = next(iter(tb.layouts))
            tb.data.data.committed_stl = stl
            tb.committed_stl = (
                stl  # mirrors beam's V.graph.get_buffer(name).committed_stl = stl
            )
            layout_var[name] = model.new_constant(0)

    # Create one variable per op output buffer.
    for op in operations:
        if not hasattr(op, "layouts"):
            continue
        name = op.get_name()
        cost_fn = op.restick_cost_fn
        n = len(op.layouts)
        assert n > 0, (
            f"{name}: op.layouts is empty — propagate_layouts did not set candidates"
        )
        if isinstance(cost_fn, FixedInOutNode):
            # FixedInOutNode always has exactly one output layout (required_out_stl).
            layout_var[name] = model.new_constant(0)
        elif isinstance(cost_fn, AnyInNode):
            # AnyInNode ops accept any layout at zero cost — no constraints are
            # added for them (they are skipped in _build_node_tables), so the solver
            # chooses freely.  Use a free integer variable so the domain matches
            # ec._in_layouts when this buffer appears as a dep of a downstream op.
            layout_var[name] = model.new_int_var(0, n - 1, f"layout_{name}")
        else:
            layout_var[name] = model.new_int_var(0, n - 1, f"layout_{name}")

    # Guard: every dep buffer referenced by any op's edge_costs must be in layout_var.
    # Buffers without layouts (constants, scalars) are not optimized and should not
    # appear as deps of ops that have layouts. Assert rather than silently skip.
    for op in operations:
        if not hasattr(op, "layouts"):
            continue
        for ec in op.restick_cost_fn.edge_costs:
            assert ec.dep.name in layout_var, (
                f"{op.get_name()}: dep '{ec.dep.name}' not in layout_var — "
                f"it has no layouts and was not registered as a graph input with layouts"
            )

    # Step 3: Add feasibility constraints and build objective cost vars.
    cost_vars = []

    for op, table in node_tables:
        cost_fn = op.restick_cost_fn
        op_name = op.get_name()

        if isinstance(cost_fn, FixedInOutNode):
            # Input layout vars + constant output var.
            dep_vars = [layout_var[ec.dep.name] for ec in cost_fn.edge_costs]
            out_var = layout_var[op_name]
            all_vars = dep_vars + [out_var]
            tuples_no_cost = [t[:-1] for t in table]
            costs_list = [int(t[-1]) for t in table]
        elif isinstance(cost_fn, AllSameNode) and cost_fn._output_edge_costs:
            # AllSameNode with co-output deps: use only real input edge_costs in the
            # table constraint.  Co-output equality is handled via a separate
            # add_allowed_assignments below so that index mismatches between
            # ec._in_layouts (captured at propagation time) and the dep buffer's
            # current .layouts list (used to size layout_var) don't corrupt the table.
            dep_vars = [layout_var[ec.dep.name] for ec in cost_fn._input_edge_costs]
            out_var = layout_var[op_name]
            all_vars = dep_vars + [out_var]
            tuples_no_cost = [t[:-1] for t in table]
            costs_list = [int(t[-1]) for t in table]
        else:
            # AllSameNode without co-output deps: input layout vars + output layout var.
            dep_vars = [layout_var[ec.dep.name] for ec in cost_fn.edge_costs]
            out_var = layout_var[op_name]
            all_vars = dep_vars + [out_var]
            tuples_no_cost = [t[:-1] for t in table]
            costs_list = [int(t[-1]) for t in table]

        if not all_vars:
            continue

        # Feasibility constraint: only allow combinations in the table.
        model.add_allowed_assignments(all_vars, tuples_no_cost)

        # Co-output equality constraints: for each co-output dep, the dep's layout
        # and this op's output must be the same STL (from their respective .layouts
        # lists, which may be ordered differently).  Use pairwise allowed_assignments
        # to enforce this without assuming index alignment.
        if isinstance(cost_fn, AllSameNode) and cost_fn._output_edge_costs:
            for co_ec in cost_fn._output_edge_costs:
                co_dep_name = co_ec.dep.name
                if co_dep_name not in layout_var:
                    continue
                co_var = layout_var[co_dep_name]
                co_dep_buf = V.graph.get_buffer(co_dep_name)
                co_dep_layouts = getattr(co_dep_buf, "layouts", [])
                if not co_dep_layouts:
                    continue
                op_layouts = op.layouts
                # Build pairs (j_co, j_op) where co_dep_layouts[j_co] == op_layouts[j_op].
                stl_to_co_idx = {stl: j for j, stl in enumerate(co_dep_layouts)}
                valid_pairs = [
                    (stl_to_co_idx[stl], j_op)
                    for j_op, stl in enumerate(op_layouts)
                    if stl in stl_to_co_idx
                ]
                if valid_pairs:
                    model.add_allowed_assignments([co_var, out_var], valid_pairs)

        # Cost objective via flattened index + add_element.
        # flat_idx = i1*stride1 + i2*stride2 + ... + j  (row-major)
        if not costs_list:
            continue

        # Compute strides for the flattened index.
        # Use v.domain.max() to get the upper bound — Proto().domain[-1] is
        # not a valid API in the installed OR-Tools version and will crash.
        dims = []
        for v in dep_vars:
            dims.append(v.domain.max() + 1)
        dims.append(out_var.domain.max() + 1)

        strides: list[int] = []
        stride = 1
        for d in reversed(dims):
            strides.insert(0, stride)
            stride *= d

        flat_idx_var = model.new_int_var(0, stride - 1, f"flat_{op_name}")
        flat_expr = sum(v * s for v, s in zip(all_vars, strides))
        model.add(flat_idx_var == flat_expr)

        max_cost = max(costs_list)
        cost_var = model.new_int_var(0, max_cost, f"cost_{op_name}")
        # Build a cost lookup table indexed by flat_idx.
        # The table only covers feasible entries (tuples_no_cost is the allowed set),
        # so we need a lookup array of size = product(dims) with 0 for infeasible.
        lookup = [0] * stride
        for t, c in zip(tuples_no_cost, costs_list):
            flat = sum(t[k] * strides[k] for k in range(len(t)))
            lookup[flat] = c
        model.add_element(flat_idx_var, lookup, cost_var)
        cost_vars.append(cost_var)

    # Step 4: Set objective.
    if cost_vars:
        model.minimize(sum(cost_vars))

    # Step 5: Solve.
    solver = _cp_model.CpSolver()
    status = solver.Solve(model)

    if status == _cp_model.MODEL_INVALID:
        raise NotImplementedError(
            f"cp_sat_stick_optimizer: CP-SAT model is invalid — likely an empty "
            f"layouts list on some op (new_int_var(0, -1) is illegal). "
            f"Validation: {solver.SolveLog()}"
        )
    if status == _cp_model.INFEASIBLE:
        raise NotImplementedError(
            f"cp_sat_stick_optimizer: CP-SAT returned INFEASIBLE "
            f"(wall time {solver.WallTime():.3f}s)"
        )
    if status == _cp_model.UNKNOWN:
        # Raise for now so failures surface clearly during testing.
        # TODO: fall back to beam_global_min_cost once UNKNOWN is understood.
        raise NotImplementedError(
            f"cp_sat_stick_optimizer: CP-SAT returned UNKNOWN "
            f"(search exhausted without proof; wall time {solver.WallTime():.3f}s)"
        )

    total_cost = solver.ObjectiveValue() if cost_vars else 0.0
    status_name = "OPTIMAL" if status == _cp_model.OPTIMAL else "FEASIBLE"
    logger.info(
        "cp_sat_stick_optimizer done: status=%s, cost=%.0f, wall_time=%.3fs",
        status_name,
        total_cost,
        solver.WallTime(),
    )

    # Step 6: Commit STLs.
    for op in operations:
        if not hasattr(op, "layouts"):
            continue
        name = op.get_name()
        if name not in layout_var:
            continue
        cost_fn = op.restick_cost_fn

        if isinstance(cost_fn, FixedInOutNode):
            stl = cost_fn.required_out_stl
        else:
            idx = solver.Value(layout_var[name])
            assert 0 <= idx < len(op.layouts), (
                f"{name}: layout index {idx} out of range [0, {len(op.layouts)})"
            )
            stl = op.layouts[idx]

        buf = V.graph.get_buffer(name)
        buf.committed_stl = stl


def optimize_restickify_locations(graph: GraphLowering) -> None:
    """Select restickify locations for all ops, minimizing total restickify cost."""
    operations = graph.operations
    optimizer = config.restickify_optimizer
    if optimizer == "cpsat":
        logger.info("optimizer: cp_sat_stick_optimizer")
        cp_sat_stick_optimizer(operations)
    else:
        if optimizer != "beam":
            logger.warning(
                "Unknown SPYRE_RESTICKIFY_OPTIMIZER=%r, falling back to beam", optimizer
            )
        logger.info("optimizer: beam (global)")
        beam_global_min_cost(operations)
