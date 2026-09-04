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

"""The CP-SAT solver's relayout decision, driven without a graph.

Handcrafted CoreDivisionBuffers drive ``CpSatLayoutSolver`` directly - no
graph, no compile - to pin the decision semantics:

- relayout fires when its fitted cost beats the spill it avoids, and only then;
- the destination rectangle occupies real LX space (capacity can veto);
- no relayout decision is ever made under the fallback objective;
- the gate's old behavior (no match, no relayout table -> spilled) survives.

The producer P (uses [0, 1]) feeds consumer C (uses [1, 2]). P and C each
carry one 4-way division; ``cd_parent_matches`` is EMPTY on the edge, so
residency for P is possible only through the relayout table.
"""

import pytest
import sympy

pytest.importorskip("ortools")

from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import CpSatLayoutSolver
from torch_spyre._inductor.scratchpad.lx_relayout import (
    ChosenRelayout,
    RelayoutCandidate,
    RelayoutSegment,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    relayout_symbol,
)

_SPILL_NS = 20000.0  # what spilling P charges in the crafted objective
_CORE = sympy.Symbol("core_id")


def _view(slot: int) -> PerCoreView:
    """A 4-way per-core view of device dim 1; ``slot`` rotates the ownership so
    distinct slots are distinct (relayout-compatible) views."""
    return PerCoreView(((1, 4),), ((1, sympy.Mod(_CORE + slot, 4)),), 4)


def _candidate(consumer, i, cost_ns, group=0, j=0) -> RelayoutCandidate:
    """The priced candidate the allocator would enumerate for P -> consumer under
    source division ``i`` / consumer division ``j``, landing on destination
    view ``group``. The solver only carries the views, so their exact geometry
    is immaterial here; they must merely differ from the source and be
    distinct per group."""
    return RelayoutCandidate(
        parent="P",
        consumer=consumer,
        source_division=i,
        consumer_division=j,
        group=group,
        source_view=_view(0),
        destination_view=_view(group + 1),
        num_cores=4,
        cost_ns=cost_ns,
    )


def _solve_with_model(buffers, size, expr):
    """Run the layout solve and also return the solved CP-SAT model and the
    solver's relayout group registry, so tests can read the bridge literals,
    interval ends and offsets the solver actually chose (not just the tuples
    extraction derived from them)."""
    from ortools.sat.python import cp_model

    captured = []
    real_solve = cp_model.CpSolver.Solve

    def spy(self, model, *a, **k):
        status = real_solve(self, model, *a, **k)
        captured.append(self)
        return status

    cp_model.CpSolver.Solve = spy
    try:
        layout = CpSatLayoutSolver(buffers, size, alignment=1)
        result = {b.name: b for b in layout.plan_layout_and_core_divisions(expr)}
    finally:
        cp_model.CpSolver.Solve = real_solve
    return result, captured[-1], layout._relayout_groups


def _buffers(relayout_cost_ns):
    p = CoreDivisionBuffer(
        "P",
        64,
        [0, 1],
        core_divisions=[CoreDivision(output_splits={1: 4})],
    )
    c = CoreDivisionBuffer(
        "C",
        64,
        [1, 2],
        core_divisions=[CoreDivision(output_splits={1: 4})],
        parents=["P"],
        cd_parent_matches={"P": []},
        cd_parent_relayouts={"P": [_candidate("C", 0, relayout_cost_ns, 0)]},
    )
    return p, c


def _objective(p):
    # Spilling P costs _SPILL_NS; relaying out costs the edge's table price.
    # The solver picks whichever is cheaper - or spills if relayout cannot fit.
    return (1 - p.sym_is_lx) * _SPILL_NS + relayout_symbol("P", 0)


def _solve(relayout_cost_ns, size=64, cost_expr="default"):
    p, c = _buffers(relayout_cost_ns)
    expr = _objective(p) if cost_expr == "default" else cost_expr
    solver = CpSatLayoutSolver([p, c], size, alignment=1)
    result = {b.name: b for b in solver.plan_layout_and_core_divisions(expr)}
    return result["P"], result["C"]


def test_relayout_fires_when_cheaper_than_the_spill():
    p, c = _solve(relayout_cost_ns=5000.0)
    assert p.address is not None, "P must reside: relayout beats the spill"
    assert "P" in c.chosen_relayouts
    chosen = c.chosen_relayouts["P"]
    assert chosen.candidate == _candidate("C", 0, 5000.0)
    assert chosen.run_head == "C"
    dest_address = chosen.destination_address
    # The destination is real LX space, disjoint from P per-core footprint
    # (both are alive at C's tick): [addr, addr+16) each, 64-byte capacity.
    per_core = 16
    assert 0 <= dest_address <= 64 - per_core
    assert not (
        dest_address < p.address + per_core and p.address < dest_address + per_core
    ), "destination overlaps its source"


def test_relayout_declines_when_dearer_than_the_spill():
    p, c = _solve(relayout_cost_ns=50000.0)
    assert p.address is None, "spilling is cheaper: P must not reside"
    assert c.chosen_relayouts == {}


def test_capacity_vetoes_a_profitable_relayout():
    # 16-byte LX: P's per-core footprint fills it, so the destination cannot
    # coexist with P at C's tick. The edge stays off no matter how profitable,
    # and the group is never partially placed.
    p, c = _solve(relayout_cost_ns=5000.0, size=16)
    assert c.chosen_relayouts == {}
    # P without a serving edge cannot reside (gate: match or relayout).
    assert p.address is None


def test_fallback_objective_never_decides_a_relayout():
    # Under the HBM-bytes fallback (cost_expr=None) a shuffle is unpriced and
    # would look free; the edge must be pinned off, restoring the old gate:
    # no match pair -> P spilled.
    p, c = _solve(relayout_cost_ns=5000.0, cost_expr=None)
    assert p.address is None
    assert c.chosen_relayouts == {}


def test_gate_without_a_relayout_table_is_unchanged():
    # No match pairs and no relayout table: in_buffer is forced off exactly
    # as before (regression guard for the constrain_residency rewrite).
    p = CoreDivisionBuffer(
        "P", 64, [0, 1], core_divisions=[CoreDivision(output_splits={1: 4})]
    )
    c = CoreDivisionBuffer(
        "C",
        64,
        [1, 2],
        core_divisions=[CoreDivision(output_splits={1: 4})],
        parents=["P"],
        cd_parent_matches={"P": []},
    )
    solver = CpSatLayoutSolver([p, c], 64, alignment=1)
    # No relayout table -> no group cost symbol in the objective (mirrors the
    # allocator, which appends a symbol only per relayout group; an unbound
    # symbol is a KeyError in the name-keyed printer).
    expr = (1 - p.sym_is_lx) * _SPILL_NS
    result = {b.name: b for b in solver.plan_layout_and_core_divisions(expr)}
    assert result["P"].address is None
    assert result["C"].chosen_relayouts == {}


def _fanout_buffers(relayout_cost_ns, same_view: bool):
    """P feeds C1 (uses [1, 2]) and C2 (uses [2, 3]); both edges have an empty
    match table and one priced relayout candidate. With ``same_view`` both
    candidates land on the same destination view of P (group 0); otherwise
    they are distinct views (groups 0 and 1)."""
    p = CoreDivisionBuffer(
        "P", 64, [0, 3], core_divisions=[CoreDivision(output_splits={1: 4})]
    )
    c1 = CoreDivisionBuffer(
        "C1",
        64,
        [1, 2],
        core_divisions=[CoreDivision(output_splits={1: 4})],
        parents=["P"],
        cd_parent_matches={"P": []},
        cd_parent_relayouts={"P": [_candidate("C1", 0, relayout_cost_ns, 0)]},
    )
    c2 = CoreDivisionBuffer(
        "C2",
        64,
        [2, 3],
        core_divisions=[CoreDivision(output_splits={1: 4})],
        parents=["P"],
        cd_parent_matches={"P": []},
        cd_parent_relayouts={
            "P": [_candidate("C2", 0, relayout_cost_ns, 0 if same_view else 1)]
        },
    )
    return p, c1, c2


def test_two_consumers_on_one_view_share_one_relayout():
    """Fan-out to two consumers wanting the SAME destination view: one group,
    one destination rectangle spanning both consumers' ticks, one charge.
    Spilling costs 20000; the shuffle 12000. Charged per edge it would be
    24000 and the solver would spill; charged per group it fires."""
    p, c1, c2 = _fanout_buffers(12000.0, same_view=True)
    expr = (1 - p.sym_is_lx) * _SPILL_NS + relayout_symbol("P", 0)
    solver = CpSatLayoutSolver([p, c1, c2], 64, alignment=1)
    r = {b.name: b for b in solver.plan_layout_and_core_divisions(expr)}
    assert r["P"].address is not None, "one shared shuffle beats the spill"
    a1 = r["C1"].chosen_relayouts["P"]
    a2 = r["C2"].chosen_relayouts["P"]
    assert a1.candidate.group == a2.candidate.group == 0
    assert a1.run_head == a2.run_head == "C1", (
        "both consumers read the segment headed by C1"
    )
    assert a1.destination_address == a2.destination_address, (
        "segment members must share the destination address"
    )
    per_core = 16
    dest = a1.destination_address
    assert not (dest < r["P"].address + per_core and r["P"].address < dest + per_core)


def test_two_consumers_on_different_views_get_two_relayouts():
    """Distinct destination views cannot share: two groups, two charges. At
    12000 each the pair (24000) loses to the 20000 spill, so P is spilled;
    at 8000 each (16000) both fire, with disjoint destinations alive at their
    own ticks."""
    p, c1, c2 = _fanout_buffers(12000.0, same_view=False)
    expr = (
        (1 - p.sym_is_lx) * _SPILL_NS
        + relayout_symbol("P", 0)
        + relayout_symbol("P", 1)
    )
    r = {
        b.name: b
        for b in CpSatLayoutSolver(
            [p, c1, c2], 64, alignment=1
        ).plan_layout_and_core_divisions(expr)
    }
    assert (
        r["P"].address is None
        and r["C1"].chosen_relayouts == {} == r["C2"].chosen_relayouts
    )
    p, c1, c2 = _fanout_buffers(8000.0, same_view=False)
    expr = (
        (1 - p.sym_is_lx) * _SPILL_NS
        + relayout_symbol("P", 0)
        + relayout_symbol("P", 1)
    )
    r = {
        b.name: b
        for b in CpSatLayoutSolver(
            [p, c1, c2], 64, alignment=1
        ).plan_layout_and_core_divisions(expr)
    }
    assert r["P"].address is not None
    assert (
        r["C1"].chosen_relayouts["P"].candidate.group == 0
        and r["C2"].chosen_relayouts["P"].candidate.group == 1
    )


def _distant_buffers(relayout_cost_ns, blocker_size):
    """P (uses [0, 10]) feeds C1 at tick 1 and C2 at tick 8, same destination
    view. B (uses [3, 7], no parents) needs ``blocker_size`` bytes of LX in
    the gap. With a 64-byte LX, P's slice (16) plus a held destination (16)
    leave 32: a 40-byte B fits only if the destination is released between
    C1 and C2, i.e. if the solver chooses TWO segments (two shuffles) instead
    of bridging."""
    p = CoreDivisionBuffer(
        "P", 64, [0, 10], core_divisions=[CoreDivision(output_splits={1: 4})]
    )
    c1 = CoreDivisionBuffer(
        "C1",
        64,
        [1, 2],
        core_divisions=[CoreDivision(output_splits={1: 4})],
        parents=["P"],
        cd_parent_matches={"P": []},
        cd_parent_relayouts={"P": [_candidate("C1", 0, relayout_cost_ns, 0)]},
    )
    c2 = CoreDivisionBuffer(
        "C2",
        64,
        [8, 9],
        core_divisions=[CoreDivision(output_splits={1: 4})],
        parents=["P"],
        cd_parent_matches={"P": []},
        cd_parent_relayouts={"P": [_candidate("C2", 0, relayout_cost_ns, 0)]},
    )
    b = CoreDivisionBuffer(
        "B",
        blocker_size,
        [3, 7],
        core_divisions=[CoreDivision(output_splits={1: 1})],
    )
    return p, c1, c2, b


def test_distant_consumers_release_the_copy_when_holding_it_would_spill():
    """Bridging C1 -> C2 saves one 5000 ns shuffle but holds 16 bytes through
    B's lifetime, spilling B at 30000 ns. Two segments are cheaper: both
    consumers fire, with DIFFERENT segment heads, and B stays resident."""
    p, c1, c2, b = _distant_buffers(5000.0, blocker_size=40)
    expr = (
        (1 - p.sym_is_lx) * _SPILL_NS
        + (1 - b.sym_is_lx) * 30000.0
        + relayout_symbol("P", 0)
    )
    r, cp, groups = _solve_with_model([p, c1, c2, b], 64, expr)
    assert r["P"].address is not None and r["B"].address is not None
    a1, a2 = r["C1"].chosen_relayouts["P"], r["C2"].chosen_relayouts["P"]
    assert a1.run_head == "C1" and a2.run_head == "C2", (
        "two segments: each consumer heads its own"
    )
    # The solved model itself: bridge off, C1's copy lives exactly its tick,
    # both consumers start a segment.
    g = groups[("P", 0)]
    assert cp.BooleanValue(g.bridges[("C1", "C2")]) is False
    assert cp.Value(g.consumers["C1"].end) == 2
    assert cp.Value(g.consumers["C2"].end) == 9
    assert all(
        cp.BooleanValue(z) for c in g.consumers.values() for z in c.pays.values()
    )


def test_distant_consumers_hold_the_copy_when_the_blocker_is_cheap_to_spill():
    """Same geometry, but spilling B costs only 1000 ns: bridging (one shuffle
    plus the 1000 ns spill) beats two shuffles, so one segment headed by C1
    serves both and B is spilled."""
    p, c1, c2, b = _distant_buffers(5000.0, blocker_size=40)
    expr = (
        (1 - p.sym_is_lx) * _SPILL_NS
        + (1 - b.sym_is_lx) * 1000.0
        + relayout_symbol("P", 0)
    )
    r, cp, groups = _solve_with_model([p, c1, c2, b], 64, expr)
    assert r["P"].address is not None and r["B"].address is None
    a1, a2 = r["C1"].chosen_relayouts["P"], r["C2"].chosen_relayouts["P"]
    assert a1.run_head == a2.run_head == "C1"
    assert a1.destination_address == a2.destination_address
    # The solved model itself: bridge on, C1's copy extends to C2's tick at the
    # same offset, and only C1 starts a segment.
    g = groups[("P", 0)]
    assert cp.BooleanValue(g.bridges[("C1", "C2")]) is True
    assert cp.Value(g.consumers["C1"].end) == 8
    assert cp.Value(g.consumers["C1"].offset) == cp.Value(g.consumers["C2"].offset)
    assert cp.BooleanValue(g.consumers["C1"].pays[0]) is True
    assert cp.BooleanValue(g.consumers["C2"].pays[0]) is False


def _consumer(name, start, end, relayout_cost_ns, *, matches=(), group=0, divs=1):
    return CoreDivisionBuffer(
        name,
        64,
        [start, end],
        core_divisions=[CoreDivision(output_splits={1: 4})],
        parents=["P"],
        cd_parent_matches={"P": list(matches)},
        cd_parent_relayouts={
            "P": [_candidate(name, i, relayout_cost_ns, group) for i in range(divs)]
        },
    )


def test_three_consumers_mixed_segmentation():
    """C1 (tick 1) and C2 (tick 2) are adjacent; C3 (tick 9) sits behind a
    40-byte blocker alive [3, 8] whose spill costs 30000. Expected: C1 -> C2
    bridged (one tick of occupancy saves a 5000 ns shuffle), C2 -> C3 not
    (holding through the blocker would spill it). Two segments, headed by C1
    and C3; run heads propagate through the chain of one bridge."""
    p = CoreDivisionBuffer(
        "P", 64, [0, 11], core_divisions=[CoreDivision(output_splits={1: 4})]
    )
    c1, c2, c3 = (
        _consumer("C1", 1, 2, 5000.0),
        _consumer("C2", 2, 3, 5000.0),
        _consumer("C3", 9, 10, 5000.0),
    )
    b = CoreDivisionBuffer(
        "B", 40, [3, 8], core_divisions=[CoreDivision(output_splits={1: 1})]
    )
    expr = (
        (1 - p.sym_is_lx) * _SPILL_NS
        + (1 - b.sym_is_lx) * 30000.0
        + relayout_symbol("P", 0)
    )
    r, cp, groups = _solve_with_model([p, c1, c2, c3, b], 64, expr)
    g = groups[("P", 0)]
    assert r["P"].address is not None and r["B"].address is not None
    heads = {n: r[n].chosen_relayouts["P"].run_head for n in ("C1", "C2", "C3")}
    assert heads == {"C1": "C1", "C2": "C1", "C3": "C3"}, heads
    assert cp.BooleanValue(g.bridges[("C1", "C2")]) is True
    assert cp.BooleanValue(g.bridges[("C2", "C3")]) is False
    assert cp.BooleanValue(g.bridges[("C1", "C3")]) is False
    assert cp.Value(g.consumers["C1"].end) == 2 and cp.Value(g.consumers["C2"].end) == 3
    # exactly two segment starts -> two shuffles charged
    assert (
        sum(cp.BooleanValue(z) for c in g.consumers.values() for z in c.pays.values())
        == 2
    )


def test_bridge_spans_a_consumer_outside_the_group():
    """C2 (tick 4) reads P through a free slicing MATCH and offers no relayout
    candidate, so it is not a member of the group at all; C1 (tick 1) and C3
    (tick 7) must relayout. With nothing else contending for LX, the bridge
    C1 -> C3 spans over C2 in time: one segment headed by C1 serves both, C2
    keeps reading the source directly, and the copy's interval covers C2's
    tick without involving it."""
    p = CoreDivisionBuffer(
        "P", 64, [0, 9], core_divisions=[CoreDivision(output_splits={1: 4})]
    )
    c1 = _consumer("C1", 1, 2, 5000.0)
    c2 = CoreDivisionBuffer(
        "C2",
        64,
        [4, 5],
        core_divisions=[CoreDivision(output_splits={1: 4})],
        parents=["P"],
        cd_parent_matches={"P": [(0, 0)]},
    )
    c3 = _consumer("C3", 7, 8, 5000.0)
    expr = (1 - p.sym_is_lx) * _SPILL_NS + relayout_symbol("P", 0)
    r, cp, groups = _solve_with_model([p, c1, c2, c3], 64, expr)
    g = groups[("P", 0)]
    assert r["P"].address is not None
    assert r["C2"].chosen_relayouts == {} and "C2" not in g.consumers
    assert (
        r["C1"].chosen_relayouts["P"].run_head
        == r["C3"].chosen_relayouts["P"].run_head
        == "C1"
    )
    assert cp.BooleanValue(g.bridges[("C1", "C3")]) is True
    assert cp.Value(g.consumers["C1"].end) == 7
    assert (
        sum(cp.BooleanValue(z) for c in g.consumers.values() for z in c.pays.values())
        == 1
    )


def test_segment_charge_follows_the_chosen_source_division():
    """P offers two divisions whose shuffles into the same destination view
    price differently (5000 vs 3000 ns). The solver picks the cheaper source
    division and the segment is charged at that price: the per-(consumer,
    division) segment-start literals, not a flat per-group constant, carry
    the cost."""
    p = CoreDivisionBuffer(
        "P",
        64,
        [0, 3],
        core_divisions=[
            CoreDivision(output_splits={1: 4}),
            CoreDivision(output_splits={0: 4}),
        ],
    )
    c = CoreDivisionBuffer(
        "C",
        64,
        [1, 2],
        core_divisions=[CoreDivision(output_splits={1: 4})],
        parents=["P"],
        cd_parent_matches={"P": []},
        cd_parent_relayouts={
            "P": [_candidate("C", 0, 5000.0, 0), _candidate("C", 1, 3000.0, 0)]
        },
    )
    expr = (1 - p.sym_is_lx) * _SPILL_NS + relayout_symbol("P", 0)
    r, cp, groups = _solve_with_model([p, c], 64, expr)
    g = groups[("P", 0)]
    assert r["P"].address is not None
    chosen = r["C"].chosen_relayouts["P"]
    assert chosen.candidate == _candidate("C", 1, 3000.0), "cheaper division wins"
    assert chosen.run_head == "C"
    pays = {d: cp.BooleanValue(z) for d, z in g.consumers["C"].pays.items()}
    assert pays == {0: False, 1: True}, pays


def test_same_tick_consumers_get_no_bridge():
    """Two consumers scheduled in the same tick cannot hand a copy across
    time, so no bridge is minted between them and each fires on its own."""
    p = CoreDivisionBuffer(
        "P", 64, [0, 3], core_divisions=[CoreDivision(output_splits={1: 4})]
    )
    c1 = _consumer("C1", 1, 2, 5000.0)
    c2 = _consumer("C2", 1, 2, 5000.0)
    expr = (1 - p.sym_is_lx) * _SPILL_NS + relayout_symbol("P", 0)
    r, cp, groups = _solve_with_model([p, c1, c2], 64, expr)
    g = groups[("P", 0)]
    assert g.bridges == {}, "same-tick consumers must not be bridge candidates"
    assert r["C1"].chosen_relayouts["P"].run_head == "C1"
    assert r["C2"].chosen_relayouts["P"].run_head == "C2"
    assert cp.Value(g.consumers["C1"].end) == 2 and cp.Value(g.consumers["C2"].end) == 2


def test_segments_regroup_fired_edges_by_run_head():
    """The commit path's view of a solve: fired edges regroup into one segment
    per (parent, view, head), members sorted by consumer, and the plan carries
    the candidate's views and the segment's shared address. Members that
    disagree on placement are a solver invariant violation, not a plan."""
    c1 = ChosenRelayout(_candidate("C1", 0, 5000.0), 16, "C1")
    c2 = ChosenRelayout(_candidate("C2", 0, 5000.0), 16, "C1")
    c3 = ChosenRelayout(_candidate("C3", 0, 5000.0), 32, "C3")
    other = ChosenRelayout(_candidate("C3", 0, 7000.0, group=1), 48, "C3")
    segments = RelayoutSegment.from_chosen([c3, other, c2, c1])
    assert [(s.group, s.run_head, s.consumer_names) for s in segments] == [
        (0, "C1", ("C1", "C2")),
        (0, "C3", ("C3",)),
        (1, "C3", ("C3",)),
    ]
    plan = segments[0].plan(source_address=0)
    assert (plan.source_name, plan.consumer_names) == ("P", ("C1", "C2"))
    assert (plan.source_view, plan.destination_view) == (_view(0), _view(1))
    assert (plan.num_cores, plan.source_address, plan.destination_address) == (
        4,
        0,
        16,
    )
    assert c1.scaled(128).destination_address == 16 * 128
    with pytest.raises(AssertionError, match="disagree"):
        RelayoutSegment.from_chosen(
            [c1, ChosenRelayout(_candidate("C2", 0, 5000.0), 64, "C1")]
        )
    with pytest.raises(ValueError, match="equal views"):
        RelayoutCandidate("P", "C", 0, 0, 0, _view(0), _view(0), 4, 1.0)
