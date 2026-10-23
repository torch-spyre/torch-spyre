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

"""A fired relayout copy describes the divisions the solve chose. Device-free."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("ortools")

import sympy  # noqa: E402

from torch_spyre._inductor.pass_utils import PerCoreView  # noqa: E402
from torch_spyre._inductor.scratchpad.buffer_graph import (  # noqa: E402
    SolveSettings,
    SpyreOptimizationGraph,
    WorkSplit,
)
from torch_spyre._inductor.scratchpad.division_space import (  # noqa: E402
    DivisionOption,
    DivisionSpace,
)
from torch_spyre._inductor.scratchpad.encoding_strategy import (  # noqa: E402
    EnumeratedStrategy,
)
from torch_spyre._inductor.scratchpad.lazy_strategy import LazyStrategy  # noqa: E402
from torch_spyre._inductor.scratchpad.plan_solver import (  # noqa: E402
    CoreDivision,
    TileSpec,
)

X, Y = sympy.symbols("x y")
CORE = sympy.Symbol("core_id")


class _Context:
    """A two-axis op whose y axis is blocked: (x, y) points with y = 2 are in
    the space but illegal, so its point menu and legal menu index apart."""

    axes = [X, Y]
    reduction_vars: list = []
    constraints = SimpleNamespace(blocked=[Y], allowed_splits={})
    span_min_splits: dict = {}
    tensor_deps: list = []

    def factor_domain(self, axis):
        return {X: [1, 2, 4], Y: [1, 2]}[axis]

    def is_legal(self, splits):
        return splits.get(Y, 1) == 1


class _Space(DivisionSpace):
    def division(self, option, splits):
        return CoreDivision(
            splits={a: v for a, v in splits.items() if v > 1}, tiling=TileSpec()
        )


def _view(dim: int, cd: CoreDivision) -> PerCoreView:
    """The producer's write splits dim 0 by x, the consumer's read dim 1."""
    x = cd.splits.get(X, 1)
    if x == 1:
        return PerCoreView((), (), num_cores=cd.cores_used)
    return PerCoreView(((dim, x),), ((dim, sympy.Mod(CORE, x)),), num_cores=x)


class _RelayoutEdge:
    def source_views(self, divs, cache):
        return [_view(0, cd) for cd in divs]

    def consumer_views(self, divs, cache):
        return [_view(1, cd) for cd in divs]

    def price(self, pv, cv, ncores, dst_cores):
        return None if ncores != dst_cores else (10.0, 128 * 64, 128 * 64)


def _graph(point_menu: bool):
    """P -> C with no direct match, so a resident P is read through a copy."""
    g = SpyreOptimizationGraph(
        alignment=128, capacity=1 << 14, settings=SolveSettings()
    )
    nodes = []
    for name, uses in (("P", [0, 1]), ("C", [1, 2])):
        space = _Space(
            op=None, options=[DivisionOption(TileSpec(), (), _Context())], max_cores=8
        )
        points = [(k, s) for k, s in space.points()]
        menu = [
            space.division(k, s)
            for k, s in points
            if point_menu or space.options[k].context.is_legal(s)
        ]
        nodes.append(
            g.add_ir_node(
                name,
                nbytes=128 * 256,
                uses=uses,
                menu=menu,
                space=space,
                point_menu=point_menu,
            )
        )
    p, c = nodes
    g.add_reads(c, ["P"])
    (edge,) = c.inputs
    edge.relayout_edge = _RelayoutEdge()
    g.link_execution_order()
    return g, p, c, edge


@pytest.mark.parametrize("point_menu", [True, False], ids=["point_menu", "legal_menu"])
@pytest.mark.parametrize(
    "strategy", [LazyStrategy, EnumeratedStrategy], ids=["lazy", "enumerated"]
)
def test_fired_relayout_reads_the_solved_divisions(strategy, point_menu):
    g, p, c, edge = _graph(point_menu)
    g.solve(cost_expr=1000 * (1 - p.sym_is_lx), strategy=strategy())

    assert p.resident and edge.relayout is not None
    fired = edge.relayout
    for node, index in ((p, fired.source_division), (c, fired.consumer_division)):
        assert WorkSplit.of(node.menu[index]) == node.work_split, node.name
    solved_p = p.menu[fired.source_division]
    solved_c = c.menu[fired.consumer_division]
    assert fired.source_view == _view(0, solved_p)
    assert fired.destination_view == _view(1, solved_c)
