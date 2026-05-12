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

from . import config
from .logging_utils import get_inductor_logger

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V
from torch_spyre._C import SpyreTensorLayout
from .pass_utils import compute_restickify_needed

INF = math.inf

logger = get_inductor_logger("optimize_restickify")


@dataclass(frozen=True)
class LayoutKey:
    """Hashable Python surrogate for SpyreTensorLayout, used as a dict/set key.

    Will be removed once PR to make SpyreTensorLayout hashable is merged.
    """

    device_size: tuple[int, ...]
    stride_map: tuple[int, ...]

    @staticmethod
    def from_stl(stl: SpyreTensorLayout) -> "LayoutKey":
        return LayoutKey(tuple(stl.device_size), tuple(stl.stride_map))


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
    ):
        self.dep = dep
        self._in_layouts = in_layouts
        self._target_layouts = target_layouts
        self._target_dep = target_dep
        self._dep_layout = V.graph.get_buffer(dep.name).get_layout()
        self._target_dep_layout = V.graph.get_buffer(target_dep.name).get_layout()

        # _cost and _layout are parallel maps.
        # _cost stores the cost for a given in/target layout pair
        # _layout stores the target STL for the restickify, or None if no restickify is needed
        self._cost: defaultdict[LayoutKey, dict[LayoutKey, float]] = defaultdict(dict)
        self._layout: defaultdict[LayoutKey, dict[LayoutKey, Any]] = defaultdict(dict)

    def _compute_and_cache_cost(
        self, in_key: "LayoutKey", target_key: "LayoutKey"
    ) -> None:
        """Populate _cost and _layout for (in_key, target_key).

        Cost is 0 if stick-compatible, the input element count if restickifiable, or INF if infeasible.
        """
        in_stl = next(
            (stl for stl in self._in_layouts if LayoutKey.from_stl(stl) == in_key),
            None,
        )
        target_stl = next(
            (
                stl
                for stl in self._target_layouts
                if LayoutKey.from_stl(stl) == target_key
            ),
            None,
        )
        assert in_stl is not None, f"in_key {in_key} not found in in_layouts"
        assert target_stl is not None, (
            f"target_key {target_key} not found in target_layouts"
        )
        needed, tgt = compute_restickify_needed(
            in_stl, self._dep_layout, self.dep, target_stl, self._target_dep
        )
        if not needed:
            cost = 0.0
        elif tgt is None:
            cost = INF  # infeasible restickify
        else:
            cost = float(math.prod(in_stl.device_size))
        self._cost[in_key][target_key] = cost
        self._layout[in_key][target_key] = tgt

    def cost(
        self, in_stl: "SpyreTensorLayout", target_stl: "SpyreTensorLayout"
    ) -> float:
        """Return the restick cost for (in_stl, target_stl), computing it on first access."""

        # Remove conversions once STL is hashable
        in_key = LayoutKey.from_stl(in_stl)
        target_key = LayoutKey.from_stl(target_stl)

        if target_key not in self._cost[in_key]:
            self._compute_and_cache_cost(in_key, target_key)
        return self._cost[in_key][target_key]

    def layout(
        self, in_stl: "SpyreTensorLayout", target_stl: "SpyreTensorLayout"
    ) -> "SpyreTensorLayout | None":
        """Return target STL for restickifying in_stl to be compatible with target_stl, or None if no restickify needed."""
        in_key = LayoutKey.from_stl(in_stl)
        target_key = LayoutKey.from_stl(target_stl)
        if target_key not in self._cost[in_key]:
            self._compute_and_cache_cost(in_key, target_key)
        return self._layout[in_key][target_key]


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


class AllSameNode(RestickNodeCost):
    """Cost node for ops that require all inputs and the output to be stick compatible (eg pointwise ops)."""

    @classmethod
    def from_args(cls, args, out_layouts, out_dep):
        assert out_layouts, "AllSameNode.from_args: out_layouts is empty"
        edge_costs = [
            EdgeCostMap(arg.dep, arg.layouts, out_layouts, out_dep) for arg in args
        ]
        return cls(edge_costs)

    def cost(
        self, in_layouts: "list[SpyreTensorLayout]", out_stl: "SpyreTensorLayout"
    ) -> float:
        return sum(ec.cost(lk, out_stl) for ec, lk in zip(self.edge_costs, in_layouts))

    def required_input_stls(self, out_stl):
        return [(ec, out_stl) for ec in self.edge_costs]


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
    def from_args(cls, args, out_stl, req_stls):
        assert req_stls, "FixedInOutNode.from_args: req_stls is empty"
        edge_costs = [
            EdgeCostMap(arg.dep, arg.layouts, [req], arg.dep)
            for arg, req in zip(args, req_stls)
        ]
        return cls(edge_costs, required_out_stl=out_stl, required_in_stls=req_stls)

    def cost(
        self, in_layouts: "list[SpyreTensorLayout]", out_stl: "SpyreTensorLayout"
    ) -> float:
        if LayoutKey.from_stl(out_stl) != LayoutKey.from_stl(self.required_out_stl):
            return INF
        return sum(
            ec.cost(lk, rk)
            for ec, lk, rk in zip(self.edge_costs, in_layouts, self.required_in_stls)
        )

    def required_input_stls(self, out_stl):
        return list(zip(self.edge_costs, self.required_in_stls))


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

        assert out_stl is not None, (
            f"({op.get_name()}): all stick possibilities had infinite cost. Cannot proceed"
        )

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
# Future improvements include (a) using live node analysis to prune dead states and (b) back-propagating
# a "min_cost" to avoid dropping states that become important later. These will be added only once
# we see evidence it matters in the models we are targeting.


@dataclass
class BeamState:
    """One hypothesis in the beam: a partial assignment of STLs to ops, with accumulated cost.

    assignments is a tuple parallel to a shared buf_names list — index i holds the
    chosen SpyreTensorLayout for buf_names[i], or None for passthrough ops.
    """

    assignments: tuple  # tuple[SpyreTensorLayout | None, ...]
    cost: float


BEAM_WIDTH = 64


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
        return self.states[0]

    def trim(self) -> None:
        self.states.sort(key=lambda s: s.cost)
        before = len(self.states)
        self.states = self.states[: self.K]
        if len(self.states) < before:
            logger.debug(
                "beam trimmed: %d -> %d states (beam_width=%d)",
                before,
                len(self.states),
                self.K,
            )


def beam_global_min_cost(operations: list) -> None:
    """Global beam search layout selection.

    Processes ops in topological order. For each op with a restick_cost_fn,
    expands every current state by branching over candidate output STLs and
    accumulating cost. After each op the beam is pruned to K best states.
    At the end, the best state's assignments are committed to the ops.
    """
    frontier = Frontier(BEAM_WIDTH)
    # Commit graph inputs and seed into the frontier so input_stl() works uniformly for all deps.
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
            frontier.states = [
                BeamState(assignments=state.assignments + (stl,), cost=state.cost)
                for state in frontier.states
            ]

    max_states = 1

    for op in operations:
        if not hasattr(op, "layouts"):
            continue

        frontier.add_buf(op.get_name())

        assert hasattr(op, "restick_cost_fn"), (
            f"op {op.get_name()} has layouts but no restick_cost_fn"
        )
        cost_fn = op.restick_cost_fn
        deps = [dep for dep in op.get_read_writes().reads if isinstance(dep, MemoryDep)]

        next_states = []
        for state in frontier.states:
            in_layouts = [frontier.input_stl(state, dep.name) for dep in deps]

            for candidate_stl in op.layouts:
                extra_cost = cost_fn.cost(in_layouts, candidate_stl)
                if extra_cost < INF:
                    next_states.append(
                        BeamState(
                            assignments=state.assignments + (candidate_stl,),
                            cost=state.cost + extra_cost,
                        )
                    )

        frontier.states = next_states
        frontier.trim()
        if not frontier.states:
            raise RuntimeError(
                f"beam search: no feasible layout combination found after op {op.get_name()}"
            )
        max_states = max(max_states, len(frontier.states))
        if logger.isEnabledFor(logging.DEBUG):
            lines = [f"beam after {op.get_name()} [{len(frontier.states)} states]:"]
            for i, s in enumerate(frontier.states):
                lines.append(f"  state {i} (cost={s.cost}):")
                for name, stl in zip(frontier.buf_names, s.assignments):
                    lines.append(f"    {name}: stride_map={list(stl.stride_map)}")
            logger.debug("\n".join(lines))

    logger.info(
        "beam search done: max states = %d, best cost = %s",
        max_states,
        frontier.best().cost,
    )

    # Commit the best state's assignments to all ops.
    best = frontier.best()
    for name, stl in zip(frontier.buf_names, best.assignments):
        op = V.graph.get_buffer(name)
        if not isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            op.committed_stl = stl


def optimize_restickify_locations(operations: list) -> None:
    """Select restickify locations for all ops, minimizing total restickify cost."""
    if config.global_stick_optimizer:
        logger.info("optimizer: beam (global)")
        beam_global_min_cost(operations)
    else:
        logger.info("optimizer: greedy (local)")
        greedy_local_min_cost(operations)
