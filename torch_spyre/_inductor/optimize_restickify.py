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
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import sympy

from . import config
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
from .pass_utils import compute_restickify_needed, device_coordinates, host_coordinates

INF = math.inf

logger = get_inductor_logger("optimize_restickify")


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
    ):
        self.dep = dep
        self._op = op
        self._in_layouts = in_layouts
        self._target_layouts = target_layouts
        self._target_dep = target_dep
        self._dep_layout = V.graph.get_buffer(dep.name).get_layout()
        self._target_dep_layout = V.graph.get_buffer(target_dep.name).get_layout()

        # _cost and _layout are parallel maps.
        # _cost stores the cost for a given in/target layout pair
        # _layout stores the target STL for the restickify, or None if no restickify is needed
        self._cost: defaultdict[SpyreTensorLayout, dict[SpyreTensorLayout, float]] = (
            defaultdict(dict)
        )
        self._layout: defaultdict[SpyreTensorLayout, dict[SpyreTensorLayout, Any]] = (
            defaultdict(dict)
        )

    def _compute_and_cache_cost(
        self, in_stl: "SpyreTensorLayout", target_stl: "SpyreTensorLayout"
    ) -> None:
        """Populate _cost and _layout for (in_stl, target_stl).

        Cost is 0 if stick-compatible, the input element count if restickifiable, or INF if infeasible.
        """
        needed, tgt = compute_restickify_needed(
            in_stl, self._dep_layout, self.dep, target_stl, self._target_dep, self._op
        )
        if not needed:
            cost = 0.0
        elif tgt is None:
            cost = INF  # infeasible restickify
        else:
            cost = float(math.prod(in_stl.device_size))
        self._cost[in_stl][target_stl] = cost
        self._layout[in_stl][target_stl] = tgt

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
        """Return target STL for restickifying in_stl to be compatible with target_stl, or None if no restickify needed."""
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
    """Cost node for ops that require all inputs and the output to be stick compatible (eg pointwise ops)."""

    @classmethod
    def from_args(cls, args, out_layouts, out_dep, op):
        assert out_layouts, "AllSameNode.from_args: out_layouts is empty"
        edge_costs = [
            EdgeCostMap(arg.dep, arg.layouts, out_layouts, out_dep, op) for arg in args
        ]
        return cls(edge_costs)

    def cost(
        self, in_layouts: "list[SpyreTensorLayout]", out_stl: "SpyreTensorLayout"
    ) -> float:
        return sum(ec.cost(lk, out_stl) for ec, lk in zip(self.edge_costs, in_layouts))

    def required_input_stls(self, out_stl):
        return [(ec, out_stl) for ec in self.edge_costs]

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
    ):
        super().__init__(edge_costs)
        self.required_out_stl = required_out_stl  # output layout currently assigned
        self.required_in_stls = (
            required_in_stls  # each input must be stick-compatible with this layout
        )

    @classmethod
    def from_args(cls, args, out_stl, req_stls, op):
        assert req_stls, "FixedInOutNode.from_args: req_stls is empty"
        edge_costs = [
            EdgeCostMap(arg.dep, arg.layouts, [req], arg.dep, op)
            for arg, req in zip(args, req_stls)
        ]
        return cls(edge_costs, required_out_stl=out_stl, required_in_stls=req_stls)

    def cost(
        self, in_layouts: "list[SpyreTensorLayout]", out_stl: "SpyreTensorLayout"
    ) -> float:
        if out_stl != self.required_out_stl:
            return INF
        return sum(
            ec.cost(lk, rk)
            for ec, lk, rk in zip(self.edge_costs, in_layouts, self.required_in_stls)
        )

    def required_input_stls(self, out_stl):
        return list(zip(self.edge_costs, self.required_in_stls))

    def min_input_cost(self, dep_name, in_stl, out_stl):
        if out_stl != self.required_out_stl:
            return INF
        # Returns on first match. If dep_name appears twice (e.g. matmul(x, x)),
        # the two positions may have different required_in_stls — this would return
        # the wrong cost. All current FixedInOutNode ops require the same STL for
        # both positions of a self-matmul, so this is safe today.
        for ec, req in zip(self.edge_costs, self.required_in_stls):
            if ec.dep.name == dep_name:
                edge_c = ec.cost(in_stl, req)
                if edge_c == INF:
                    return INF
                other_ok = all(
                    any(e.cost(other_c, r) < INF for other_c in e._in_layouts)
                    for e, r in zip(self.edge_costs, self.required_in_stls)
                    if e.dep.name != dep_name
                )
                return edge_c if other_ok else INF
        return INF


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


def greedy_local_min_cost(operations: list) -> None:
    """Greedy layout selection: process ops in topological order, picking the output layout with minimum local restick cost.

    On cost ties, the first candidate layout (leftmost arg's stick) is chosen. Each op's chosen
    layout is committed immediately so downstream ops can read it.
    """

    # Process graph inputs first so all upstreams have committed_stl.
    # For now inputs are always a set of size 1, since we use it as it
    # was transferred to device
    for name in V.graph.graph_input_names:
        tb = V.graph.graph_inputs[name]
        if (
            isinstance(tb, TensorBox)
            and isinstance(tb.data, StorageBox)
            and isinstance(tb.data.data, InputBuffer)
            and hasattr(tb, "layouts")
        ):
            if not tb.layouts:
                raise AssertionError(f"graph input {name} has empty layouts set")
            stl = next(iter(tb.layouts))
            tb.data.data.committed_stl = stl
            tb.committed_stl = stl

    for op in operations:
        if not hasattr(op, "layouts"):
            continue  # FallbackKernel and other unhandled op types

        assert hasattr(op, "restick_cost_fn"), (
            f"op {op.get_name()} has layouts but no restick_cost_fn"
        )
        cost_fn = op.restick_cost_fn

        # Collect each input arg's committed layout (finalized by earlier topo iterations).
        in_layouts = []
        for dep in op.get_read_writes().reads:
            if isinstance(dep, MemoryDep):
                buf = V.graph.get_buffer(dep.name)
                assert hasattr(buf, "committed_stl"), (
                    f"buffer {dep.name} has no committed_stl — "
                    "topological order violated or input not committed"
                )
                in_layouts.append(buf.committed_stl)

        assert op.layouts, (
            f"op {op.get_name()} has restick_cost_fn but no candidate output layouts"
        )
        out_stl = None
        best_cost = float("inf")
        for candidate_stl in op.layouts:
            out_layout_cost = cost_fn.cost(in_layouts, candidate_stl)
            if out_layout_cost < best_cost:
                best_cost = out_layout_cost
                out_stl = candidate_stl

        if out_stl is None:
            raise _no_feasible_layout_error(op)

        op.committed_stl = out_stl


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
        for dep in op.get_read_writes().reads:
            if (
                isinstance(dep, MemoryDep)
                and op.get_name() not in downstream_seen[dep.name]
            ):
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
        for dep in op.get_read_writes().reads:
            if isinstance(dep, MemoryDep) and dep.name in step_of:
                consumer_step = step_of[op.get_name()]
                if last_use.get(dep.name, -1) < consumer_step:
                    last_use[dep.name] = consumer_step
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
        deps = [dep for dep in op.get_read_writes().reads if isinstance(dep, MemoryDep)]

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


def optimize_restickify_locations(graph: GraphLowering) -> None:
    """Select restickify locations for all ops, minimizing total restickify cost."""
    operations = graph.operations
    if config.global_stick_optimizer:
        logger.info("optimizer: beam (global)")
        beam_global_min_cost(operations)
    else:
        logger.info("optimizer: greedy (local)")
        greedy_local_min_cost(operations)
