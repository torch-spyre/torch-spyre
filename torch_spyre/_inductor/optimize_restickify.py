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
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class LayoutKey:
    """Hashable Python surrogate for SpyreTensorLayout, used as a dict/set key.

    SpyreTensorLayout is not hashable and includes dtype, which is irrelevant
    for stick-compatibility comparisons.
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
        self._cost: defaultdict[LayoutKey, dict[LayoutKey, float]] = defaultdict(dict)
        self._restick_target: defaultdict[LayoutKey, dict[LayoutKey, Any]] = (
            defaultdict(dict)
        )

    def _compute_and_cache_cost(
        self, in_key: "LayoutKey", target_key: "LayoutKey"
    ) -> None:
        """Populate _cost and _restick_target for (in_key, target_key).

        Cost is 0 if stick-compatible, the input element count if restickifiable, or INF if infeasible.
        """
        in_layout = next(
            (
                layout
                for layout in self._in_layouts
                if LayoutKey.from_stl(layout.device_layout) == in_key
            ),
            None,
        )
        target_layout = next(
            (
                layout
                for layout in self._target_layouts
                if LayoutKey.from_stl(layout.device_layout) == target_key
            ),
            None,
        )
        assert in_layout is not None, f"in_key {in_key} not found in in_layouts"
        assert target_layout is not None, (
            f"target_key {target_key} not found in target_layouts"
        )
        needed, tgt = compute_restickify_needed(
            in_layout, self.dep, target_layout, self._target_dep
        )
        if not needed:
            cost = 0.0
        elif tgt is None:
            cost = INF  # infeasible restickify
        else:
            cost = float(math.prod(s for s in in_layout.size))
        self._cost[in_key][target_key] = cost
        self._restick_target[in_key][target_key] = tgt

    def cost(self, in_key: "LayoutKey", target_key: "LayoutKey") -> float:
        """Return the restick cost for (in_key, target_key), computing it on first access."""
        if target_key not in self._cost[in_key]:
            self._compute_and_cache_cost(in_key, target_key)
        return self._cost[in_key][target_key]

    def restick_target(self, in_key: "LayoutKey", target_key: "LayoutKey"):
        """Restickified layout to insert, or None if no restickify needed."""
        if target_key not in self._cost[in_key]:
            self._compute_and_cache_cost(in_key, target_key)
        return self._restick_target[in_key][target_key]


class RestickNodeCost(abc.ABC):
    """Abstract base for per-op restick cost functions.

    Subclasses encode the stick-compatibility rules for a specific op type and
    compute the total restick cost given each input's committed layout and a
    candidate output layout key.
    """

    def __init__(self, edge_costs):
        self.edge_costs = edge_costs

    @abc.abstractmethod
    def cost(self, in_layouts: "list[LayoutKey]", out_key: "LayoutKey") -> float: ...


class AllSameNode(RestickNodeCost):
    """Cost node for ops that require all inputs and the output to be stick compatible (eg pointwise ops)."""

    @classmethod
    def from_args(cls, args, out_layouts, out_dep):
        assert out_layouts, "AllSameNode.from_args: out_layouts is empty"
        edge_costs = [
            EdgeCostMap(arg.dep, arg.layouts, out_layouts, out_dep) for arg in args
        ]
        return cls(edge_costs)

    def cost(self, in_layouts: "list[LayoutKey]", out_key: "LayoutKey") -> float:
        total = 0.0
        for edge_cost, layout_key in zip(self.edge_costs, in_layouts):
            c = edge_cost.cost(layout_key, out_key)
            if c == INF:
                return INF
            total += c
        return total


class FixedInOutNode(RestickNodeCost):
    """Cost node for ops whose input and output stick compatibility is fixed by the op (eg, matmul)."""

    def __init__(
        self,
        edge_costs,
        required_out_key: "LayoutKey",
        required_in_keys: "list[LayoutKey]",
    ):
        super().__init__(edge_costs)
        self.required_out_key = required_out_key  # output layout currently assigned
        self.required_in_keys = (
            required_in_keys  # each input must be stick-compatible with this layout
        )

    @classmethod
    def from_args(cls, args, out_stl, req_layouts):
        assert req_layouts, "FixedInOutNode.from_args: req_layouts is empty"
        required_out_key = LayoutKey.from_stl(out_stl)
        edge_costs = [
            EdgeCostMap(arg.dep, arg.layouts, [req], arg.dep)
            for arg, req in zip(args, req_layouts)
        ]
        req_keys = [LayoutKey.from_stl(req.device_layout) for req in req_layouts]
        return cls(
            edge_costs, required_out_key=required_out_key, required_in_keys=req_keys
        )

    def cost(self, in_layouts: "list[LayoutKey]", out_key: "LayoutKey") -> float:
        if out_key != self.required_out_key:
            return INF
        total = 0.0
        for edge_cost, layout_key, req_key in zip(
            self.edge_costs, in_layouts, self.required_in_keys
        ):
            c = edge_cost.cost(layout_key, req_key)
            if c == INF:
                return INF
            total += c
        return total


class AnyInNode(RestickNodeCost):
    """Cost node for ops that accept any input layout and produce a fixed output layout.

    Used for aten.clone.default: the clone materializes a new buffer in the output
    layout regardless of input stick, so no restickify is ever needed before it.
    """

    @classmethod
    def from_args(cls):
        return cls(edge_costs=[])

    def cost(self, in_layouts: "list[LayoutKey]", out_key: "LayoutKey") -> float:
        return 0.0


def record_stick_decisions(
    op, out_key: LayoutKey, in_layouts: "list[LayoutKey]"
) -> None:
    """Store the optimizer's chosen output layout and input layouts on op for the restickify insertion pass."""
    op.stick_decisions = {
        "out_key": out_key,
        "arg_in_layouts": in_layouts,
    }


def optimize_restickify_locations(operations: list) -> None:
    """Select restickify locations for all ops, minimizing total restickify cost.

    Currently uses a greedy local algorithm; intended to be replaced with a global optimizer.
    """
    greedy_local_min_cost(operations)


def _print_op_layouts(operations: list, label: str) -> None:
    """Debug helper: print op layout candidates before/after greedy selection."""
    print(f"\n=== op layouts [{label}] ===")
    for op in operations:
        layout = op.layout
        layouts = getattr(op, "layouts", None)
        print(
            f"  {op.get_name()}: layout={type(layout).__name__}({layout})"
            + (f" | layouts={[type(lo).__name__ for lo in layouts]}" if layouts else "")
        )


def greedy_local_min_cost(operations: list) -> None:
    """Greedy layout selection: process ops in topological order, picking the output layout with minimum local restick cost.

    On cost ties, the first candidate layout (leftmost arg's stick) is chosen. Each op's chosen
    layout is committed immediately so downstream ops can read it.
    """

    print()
    print("=== In greedy_local_min_cost ===")
    _print_op_layouts(operations, "before")

    print()
    print("-- Running greedy algorithm --")
    # Process graph inputs first so all upstreams have committed_layout.
    # For now inputs are always a set of size 1, since we use it as it
    # was tranferred to device
    for name in V.graph.graph_input_names:
        tb = V.graph.graph_inputs[name]
        if (
            isinstance(tb, TensorBox)
            and isinstance(tb.data, StorageBox)
            and isinstance(tb.data.data, InputBuffer)
            and hasattr(tb, "layouts")
        ):
            print(
                f"MRA input layouts: {name} -> {[list(LayoutKey.from_stl(lo.device_layout).stride_map) for lo in tb.layouts]}"
            )
            if not tb.layouts:
                raise AssertionError(f"graph input {name} has empty layouts set")
            layout = next(iter(tb.layouts))
            tb.data.data.layout = layout
            tb.data.data.committed_layout = LayoutKey.from_stl(layout.device_layout)
            tb.committed_layout = tb.data.data.committed_layout
            print(
                f"MRA input committed: {name} -> {list(tb.committed_layout.stride_map)}"
            )
            del tb.layouts

    for op in operations:
        print("Processing node:", op.get_name())
        if not hasattr(op, "layouts"):
            continue  # FallbackKernel and other unhandled op types

        if not hasattr(op, "restick_cost_fn"):
            if not isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                # Must set the layout for Mutation Ops
                # TODO should this be done in propagate_layouts?
                op.layout = op.layouts[0]
                op.committed_layout = LayoutKey.from_stl(op.layouts[0].device_layout)

            # nothing to do here.
            continue

        cost_fn = op.restick_cost_fn

        # Collect each input arg's committed layout (finalized by earlier topo iterations).
        in_layouts = []
        for dep in op.get_read_writes().reads:
            if isinstance(dep, MemoryDep):
                buf = V.graph.get_buffer(dep.name)
                assert hasattr(buf, "committed_layout"), (
                    f"buffer {dep.name} has no committed_layout — "
                    "topological order violated or input not committed"
                )
                in_layouts.append(buf.committed_layout)

        # TODO: Don't out layout keys every time sigh
        out_layout_keys = [LayoutKey.from_stl(ol.device_layout) for ol in op.layouts]
        assert out_layout_keys, (
            f"op {op.get_name()} has restick_cost_fn but no candidate output layouts"
        )
        layout = None
        out_key = None
        best_cost = float("inf")
        for out_layout, out_layout_key in zip(op.layouts, out_layout_keys):
            out_layout_cost = cost_fn.cost(in_layouts, out_layout_key)
            print(
                f"MRA candidate ({op.get_name()}): "
                f"stick={list(out_layout_key.stride_map)} cost={out_layout_cost}"
            )
            if out_layout_cost < best_cost:
                best_cost = out_layout_cost
                layout = out_layout
                out_key = out_layout_key

        assert out_key is not None, (
            f"({op.get_name()}): all stick possibilities had infinite cost. Cannot proceed"
        )

        print(
            f"MRA select_restickify_locations ({op.get_name()}): "
            f"stick={list(out_key.stride_map)} cost={best_cost} "
            f"in_layouts={[list(lk.stride_map) for lk in in_layouts]}"
        )
        if not isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            op.layout = layout
        op.committed_layout = out_key
        record_stick_decisions(op, out_key, in_layouts)

    _print_op_layouts(operations, "after")
