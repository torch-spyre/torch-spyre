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

"""The CP-SAT solve's graph: nodes, dependency and execution edges, and the
builder that makes them from the allocator's buffers. Device-free."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path

import pytest

pytest.importorskip("ortools")

sys.path.insert(0, str(Path(__file__).parent))

from cooptimization_capture_loader import load_captures  # noqa: E402
from synthetic_cooptimization_graphs import synthetic_graphs  # noqa: E402
from torch_spyre._inductor.scratchpad.buffer_graph import (  # noqa: E402
    DependencyEdge,
    ExecutionEdge,
    ResourceNode,
    SolveSettings,
    SpyreOptimizationGraph,
    WorkSplit,
)
from torch_spyre._inductor.scratchpad.plan_solver import (  # noqa: E402
    BufferType,
    CoreDivision,
)

from buffer_spec import (  # noqa: E402
    CoreDivisionBuffer,
    LifetimeBoundBuffer,
    add_node,
    build_graph,
    strategy_of,
)

_SETTINGS = SolveSettings(time_limit_seconds=None, bottom_justify=True, record=None)


def _graph() -> SpyreOptimizationGraph:
    return SpyreOptimizationGraph(alignment=128, capacity=16, settings=_SETTINGS)


def _cd(name, uses, parents=(), menu=None, **kw) -> CoreDivisionBuffer:
    return CoreDivisionBuffer(
        name=name,
        size=1024,
        uses=list(uses),
        core_divisions=list(menu if menu is not None else [CoreDivision()]),
        parents=list(parents),
        **kw,
    )


def _corpus() -> list[tuple[str, list]]:
    graphs = [("synthetic", synthetic_graphs()), ("captures", load_captures())]
    return [
        (f"{group}/{name}[{i}]", g.buffers)
        for group, cases in graphs
        for name, gs in cases.items()
        for i, g in enumerate(gs)
    ]


def test_add_node_keeps_insertion_order():
    graph = _graph()
    for name in ("c", "a", "b"):
        add_node(graph, _cd(name, [0, 1]))
    assert [n.name for n in graph.get_nodes()] == ["c", "a", "b"]


def test_add_node_converts_to_alignment_units_and_reads_the_buffer():
    graph = _graph()
    node = add_node(
        graph,
        _cd("x", [3, 5, 9], boundary=BufferType.Output, lifetime_end_override=12),
        residency_reason="barred",
    )
    assert node.size == 8  # 1024 B / 128 B
    assert (node.start_time, node.end_time) == (3, 12)
    assert node.read_count == 2
    assert node.produced_at == 3
    assert node.boundary == BufferType.Output
    assert node.residency_reason == "barred"
    assert node.work_split is None  # open: it has candidates


def test_add_node_commits_a_buffer_without_candidates():
    graph = _graph()
    plain = add_node(graph, LifetimeBoundBuffer("p", 256, [0, 4]))
    empty = add_node(graph, _cd("e", [1, 2], menu=[]))
    graph_input = add_node(
        graph, LifetimeBoundBuffer("i", 256, [2, 3], first_use_is_read=True)
    )
    assert plain.work_split is WorkSplit.COMMITTED and plain.committed
    assert empty.work_split is WorkSplit.COMMITTED
    assert plain.boundary == BufferType.Intermediate
    assert graph_input.boundary == BufferType.Input
    assert graph_input.produced_at is None
    assert not plain.sym_core_divs and not empty.sym_tile_counts


def test_add_child_shares_one_edge_object():
    graph = _graph()
    a = add_node(graph, _cd("a", [0, 1]))
    b = add_node(graph, _cd("b", [1, 2]))
    edge = a.add_child(b, in_place=True)
    assert isinstance(edge, DependencyEdge)
    assert a.outputs == [edge] and b.inputs == [edge]
    assert a.inputs == [] and b.outputs == []
    assert edge.in_place and edge.transform is None
    with pytest.raises(AssertionError):
        a.add_child(b)


def test_nodes_and_edges_carry_no_candidate_data():
    for cls in (ResourceNode, DependencyEdge, ExecutionEdge):
        names = {f.name for f in fields(cls)}
        assert not names & {"core_divisions", "cd_parent_matches", "candidates"}


def test_add_successor_links_both_ends_and_derives_contiguity():
    graph = _graph()
    a, b, c = (add_node(graph, _cd(n, [t, t + 1])) for n, t in zip("abc", (0, 1, 3)))
    ab = a.add_successor(b)
    bc = b.add_successor(c)
    assert a.next is ab and b.prev is ab and b.next is bc and c.prev is bc
    assert ab.contiguous and not bc.contiguous
    assert a.prev is None and c.next is None
    with pytest.raises(AssertionError):
        a.add_successor(c)
    with pytest.raises(AssertionError):
        c.add_successor(b)


def test_build_graph_chains_op_outputs_in_produced_order():
    buffers = [
        _cd("late", [5, 6]),
        LifetimeBoundBuffer("input", 128, [0, 5], first_use_is_read=True),
        _cd("early", [2, 5]),
        _cd("mid", [3, 4], parents=["early"], cd_parent_matches={"early": [(0, 0)]}),
    ]
    graph = build_graph(buffers, 128, 16, _SETTINGS)
    assert graph.first is graph.nodes["early"]
    chain = [(e.before.name, e.after.name) for e in graph.execution_chain()]
    assert chain == [("early", "mid"), ("mid", "late")]
    assert [e.contiguous for e in graph.execution_chain()] == [True, False]
    assert graph.nodes["input"].prev is None and graph.nodes["input"].next is None


def test_build_graph_puts_in_place_inputs_first():
    buffers = [
        _cd("a", [0, 2]),
        _cd("b", [1, 2]),
        _cd("c", [2, 3], parents=["a", "b"], in_place_parents=["b"]),
    ]
    graph = build_graph(buffers, 128, 16, _SETTINGS)
    c = graph.nodes["c"]
    assert [(e.src.name, e.in_place) for e in c.inputs] == [("b", True), ("a", False)]
    assert all(e.reads for e in c.inputs)


def test_a_parent_read_twice_is_one_edge_gated_twice():
    """An op may read one buffer twice; each read is a relation, as before."""
    from ortools.sat.python import cp_model

    def read_rows(parents):
        menu = [CoreDivision(), CoreDivision(splits={1: 2})]
        buffers = [
            _cd("a", [0, 1], menu=menu),
            _cd("b", [1, 2], parents, menu=menu, cd_parent_matches={"a": [(0, 0)]}),
        ]
        graph = _graph_of(buffers)
        strategy = strategy_of(buffers)
        graph.solve(cp_model.CpModel(), None, strategy)
        (edge,) = graph.nodes["b"].inputs
        rows = [p for p in strategy.provenance if p.source == "encode_read"]
        return edge.reads, sum(len(p.constraints) for p in rows)

    once, twice = read_rows(["a"]), read_rows(["a", "a"])
    assert once[0] == 1 and twice[0] == 2
    assert twice[1] == 2 * once[1] > 0


def test_build_graph_keeps_an_unread_in_place_parent():
    """The merge still needs the edge; the read gate and cuts must not see it."""
    buffers = [_cd("a", [0, 1]), _cd("c", [1, 3], in_place_parents=["a"])]
    graph = build_graph(buffers, 128, 16, _SETTINGS)
    (edge,) = graph.nodes["c"].inputs
    assert edge.in_place and not edge.reads


@pytest.mark.parametrize("case,buffers", _corpus(), ids=lambda v: v)
def test_build_graph_matches_the_allocator_buffers(case, buffers):
    graph = build_graph(buffers, 128, 1 << 14, _SETTINGS)
    names = {b.name for b in buffers}
    assert set(graph.nodes) == names
    for b in buffers:
        node = graph.nodes[b.name]
        assert [e.src.name for e in node.inputs if e.in_place] == b.in_place_parents
        reads = [p for p in getattr(b, "parents", ()) if p in names]
        assert [e.src.name for e in node.inputs if e.reads] == [
            *(p for p in b.in_place_parents if p in reads),
            *(p for p in dict.fromkeys(reads) if p not in b.in_place_parents),
        ]
        assert {e.src.name: e.reads for e in node.inputs if e.reads} == Counter(reads)
        assert {e.src.name for e in node.inputs} == {*reads, *b.in_place_parents}
        for e in node.inputs:
            assert e in e.src.outputs
    # The buffer form carries no relayout edges, so the strategy adds no copies.
    strategy = strategy_of(buffers)
    strategy.prepare(graph)
    assert strategy.transforms == ()


def _graph_of(buffers, capacity=16) -> SpyreOptimizationGraph:
    settings = SolveSettings(
        time_limit_seconds=None,
        bottom_justify=True,
        record=lambda solver, model, objective=False: solver.Solve(model),
    )
    return build_graph(buffers, 128, capacity, settings)


def _solve_graph(buffers, pin=lambda graph: None):
    from ortools.sat.python import cp_model

    graph = _graph_of(buffers)
    pin(graph)
    nodes = graph.solve(cp_model.CpModel(), None, strategy_of(buffers))
    return graph, nodes


def _chain3() -> list[CoreDivisionBuffer]:
    menu = [CoreDivision(), CoreDivision(splits={1: 2})]
    every = [(i, j) for i in range(2) for j in range(2)]
    return [
        _cd("a", [0, 1], menu=menu),
        _cd("b", [1, 2], ["a"], menu=menu, cd_parent_matches={"a": every}),
        _cd("c", [2, 3], ["b"], menu=menu, cd_parent_matches={"b": every}),
    ]


def test_solve_fills_every_open_decision_and_returns_every_node():
    graph, nodes = _solve_graph(_chain3())
    assert nodes == list(graph.get_nodes())
    for node in graph.get_nodes():
        assert node.work_split is not None and not node.committed
        assert isinstance(node.resident, bool)
        assert (node.offset is not None) == node.resident
        assert (node.spill_reason is not None) != node.resident


def test_solve_honours_pinned_decisions():
    def pin(graph):
        graph.nodes["a"].resident = False
        graph.nodes["b"].work_split = WorkSplit.of(CoreDivision(splits={1: 2}))
        graph.nodes["c"].resident = True
        graph.nodes["c"].offset = 3

    graph, _ = _solve_graph(_chain3(), pin=pin)
    a, b, c = (graph.nodes[n] for n in "abc")
    assert a.resident is False and a.offset is None
    assert b.work_split == WorkSplit.of(CoreDivision(splits={1: 2}))
    assert c.resident is True and c.offset == 3


def test_an_infeasible_pin_raises_and_writes_nothing():
    from ortools.sat.python import cp_model

    from torch_spyre._inductor.scratchpad.plan_solver import SolveError

    buffers = _chain3()
    graph = _graph_of(buffers)
    graph.nodes["b"].residency_reason = "barred"
    graph.nodes["b"].resident = True
    with pytest.raises(SolveError):
        graph.solve(cp_model.CpModel(), None, strategy_of(buffers))
    for node in graph.get_nodes():
        assert node.work_split is None and node.offset is None
        assert node.spill_reason is None
    assert graph.nodes["a"].resident is None and graph.nodes["c"].resident is None
