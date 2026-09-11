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

"""The relayout decision, driven without a graph.

A relayout group's destination is a ``RelayoutCopyBuffer``: an ordinary buffer
whose residency IS the decision to shuffle, placed by the same no-overlap as
everything else and priced by a plain term of the shared sympy objective
(``cost_term``). Handcrafted buffers drive the pieces directly - no graph, no
compile:

- the price term is solver-agnostic: evaluated by ``lambdify`` it charges the
  source's chosen division exactly when the copy is resident;
- under ``CpSatLayoutSolver`` a relayout fires when its fitted cost beats the
  spill it avoids, and only then; the copy occupies real LX (capacity can
  veto); no relayout is decided under the fallback objective; the gate's old
  behavior (no match, no table -> spilled) survives;
- one copy serves every consumer of a group (one charge), distinct views are
  distinct copies, the charge follows the chosen source division, and a copy
  that cannot be held across its consumers spills the source rather than
  re-shuffling;
- the commit path regroups fired edges by (source, destination view).

The producer P (uses [0, 1]) feeds consumer C (uses [1, 2]). P and C each
carry one 4-way division; ``cd_parent_matches`` is EMPTY on the edge, so
residency for P is possible only through the relayout copy.
"""

import pytest
import sympy

pytest.importorskip("ortools")

from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator
from torch_spyre._inductor.scratchpad.ilp_solver_ortools import CpSatLayoutSolver
from torch_spyre._inductor.scratchpad.lx_relayout import (
    ChosenRelayout,
    FiredRelayoutGroup,
    RelayoutCandidate,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    RelayoutCopyBuffer,
    relayout_copy_name,
)

_SPILL_NS = 20000.0  # what spilling P charges in the crafted objective
_CORE = sympy.Symbol("core_id")
_PER_CORE = 16  # P is 64 bytes sliced 4 ways


def _view(slot: int) -> PerCoreView:
    """A 4-way per-core view of device dim 1; ``slot`` rotates the ownership so
    distinct slots are distinct (relayout-compatible) views."""
    return PerCoreView(((1, 4),), ((1, sympy.Mod(_CORE + slot, 4)),), 4)


def _candidate(
    consumer, i, cost_ns, group=0, j=0, destination_span=16
) -> RelayoutCandidate:
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
        cost_ns=cost_ns,
        # P is 64 bytes over 4 cores; a permutation of an outer split keeps
        # the equal share as its per-core span on both sides.
        source_footprint_bytes=16,
        destination_footprint_bytes=destination_span,
    )


def _producer(uses, divisions=1) -> CoreDivisionBuffer:
    """P with ``divisions`` 4-way divisions (index 0 splits dim 1, index 1 dim 0)."""
    return CoreDivisionBuffer(
        "P",
        64,
        uses,
        core_divisions=[
            CoreDivision(output_splits={1 - d: 4}) for d in range(divisions)
        ],
    )


def _consumer(name, start, end, candidates, *, matches=()) -> CoreDivisionBuffer:
    return CoreDivisionBuffer(
        name,
        64,
        [start, end],
        core_divisions=[CoreDivision(output_splits={1: 4})],
        parents=["P"],
        cd_parent_matches={"P": list(matches)},
        cd_parent_relayouts={"P": list(candidates)} if candidates else {},
    )


def _with_copies(*buffers) -> list[CoreDivisionBuffer]:
    """The buffer set the allocator hands the solver: the graph buffers plus
    one copy per relayout group, built by the allocator's own builder."""
    return [*buffers, *CoOptimizingAllocator._relayout_copy_buffers(list(buffers))]


def _objective(buffers, spill_ns=None) -> sympy.Expr:
    """Spilling a source costs its ``spill_ns`` entry (P: _SPILL_NS); every
    copy adds its own price term, exactly as the allocator composes it."""
    spill_ns = {"P": _SPILL_NS, **(spill_ns or {})}
    expr = sympy.Integer(0)
    for b in buffers:
        if isinstance(b, RelayoutCopyBuffer):
            expr = expr + b.cost_term()
        elif b.name in spill_ns:
            expr = expr + (1 - b.sym_is_lx) * spill_ns[b.name]
    return expr


def _solve(buffers, size=64, expr="default", **spill_ns):
    if expr == "default":
        expr = _objective(buffers, spill_ns)
    solver = CpSatLayoutSolver(buffers, size, alignment=1)
    return {b.name: b for b in solver.plan_layout_and_core_divisions(expr)}


def _copy(result, group=0) -> RelayoutCopyBuffer:
    return result[relayout_copy_name("P", group)]


def _disjoint(a_addr, b_addr, footprint=_PER_CORE) -> bool:
    return not (a_addr < b_addr + footprint and b_addr < a_addr + footprint)


# ---------------------------------------------------------------------------
# The copy buffer and its generic price term
# ---------------------------------------------------------------------------


def test_copy_buffer_is_the_destination_the_allocator_would_build():
    p = _producer([0, 3])
    c1 = _consumer("C1", 1, 2, [_candidate("C1", 0, 5000.0)])
    c2 = _consumer("C2", 2, 3, [_candidate("C2", 0, 5000.0)])
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c1, c2])
    assert copy.name == relayout_copy_name("P", 0)
    assert copy.name.startswith("__spyre_lx_relayout__:"), (
        "must share the synthetic prefix so nothing tries to push or commit it"
    )
    assert copy.relayout_parent == "P" and copy.group == 0
    assert copy.consumers == ("C1", "C2")
    # Live from the first consumer's tick to the last's; the per-core footprint
    # is the destination view's measured span, sliced num_cores ways.
    assert (copy.start_time, copy.end_time) == (1, 3)
    assert copy.size == 64 and copy.num_cores == 4 and copy.min_footprint == 16
    assert copy.parents == [] and copy.cd_parent_matches == {}
    assert copy.cost_by_source_division == {0: 5000.0}


def test_copy_is_sized_by_the_destination_span_not_the_source_share():
    """A view that splits an inner device dim spans more LX per core than its
    equal share (#3440 reserves the span for the committed destination). The
    copy must reserve the same, or the solver packs a neighbour into bytes the
    shuffle will write."""
    p = _producer([0, 3])
    c1 = _consumer("C1", 1, 2, [_candidate("C1", 0, 5000.0, destination_span=48)])
    c2 = _consumer("C2", 2, 3, [_candidate("C2", 0, 5000.0, destination_span=48)])
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c1, c2])
    assert copy.size == 48 * 4 and copy.min_footprint == 48, (
        "per-core footprint must be the destination span, not size / num_cores"
    )
    # Members of one group land on one view, so they were measured alike; a
    # disagreement is an enumeration error, never averaged or maxed away.
    c3 = _consumer("C3", 2, 3, [_candidate("C3", 0, 5000.0, destination_span=32)])
    with pytest.raises(AssertionError, match="mixes destination spans"):
        CoOptimizingAllocator._relayout_copy_buffers([p, c1, c3])


def test_plan_carries_the_measured_spans():
    """The plan hands the allocator the spans the enumeration measured, so the
    committed side sizes source and destination exactly as the solver did."""
    fired = ChosenRelayout(_candidate("C1", 0, 5000.0, destination_span=48), 16)
    (group,) = FiredRelayoutGroup.from_chosen([fired])
    plan = group.plan(source_address=0)
    assert (plan.source_footprint_bytes, plan.destination_footprint_bytes) == (16, 48)
    with pytest.raises(AssertionError, match="disagree"):
        FiredRelayoutGroup.from_chosen(
            [
                fired,
                ChosenRelayout(_candidate("C2", 0, 5000.0, destination_span=32), 16),
            ]
        )


def test_price_term_charges_the_chosen_source_division_while_resident():
    """Evaluated the way the annealer evaluates the objective: ``lambdify`` over
    (is_lx, division) values. The term is the solver-agnostic contract, so pin
    it here independently of any engine."""
    p = _producer([0, 2], divisions=2)
    c = _consumer("C", 1, 2, [_candidate("C", 0, 5000.0), _candidate("C", 1, 3000.0)])
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c])
    term = copy.cost_term()
    assert term.free_symbols == {copy.sym_is_lx, p.sym_division}
    f = sympy.lambdify([copy.sym_is_lx, p.sym_division], term, modules="math")
    assert (f(1, 0), f(1, 1)) == (5000.0, 3000.0)
    assert f(0, 0) == f(0, 1) == 0.0, "no charge while the copy is not resident"


def test_price_disagreement_within_a_group_is_an_error():
    p = _producer([0, 2])
    c1 = _consumer("C1", 1, 2, [_candidate("C1", 0, 5000.0)])
    c2 = _consumer("C2", 1, 2, [_candidate("C2", 0, 7000.0)])
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c1, c2])
    with pytest.raises(AssertionError, match="disagree on the price"):
        copy.cost_by_source_division


def test_group_without_its_source_gets_no_copy():
    c = _consumer("C", 1, 2, [_candidate("C", 0, 5000.0)])
    assert CoOptimizingAllocator._relayout_copy_buffers([c]) == []


# ---------------------------------------------------------------------------
# The CP-SAT decision
# ---------------------------------------------------------------------------


def _pc(relayout_cost_ns):
    return _with_copies(
        _producer([0, 1]), _consumer("C", 1, 2, [_candidate("C", 0, relayout_cost_ns)])
    )


def test_relayout_fires_when_cheaper_than_the_spill():
    r = _solve(_pc(5000.0))
    p, c, copy = r["P"], r["C"], _copy(r)
    assert p.address is not None, "P must reside: relayout beats the spill"
    assert copy.address is not None, "the copy IS the decision"
    chosen = c.chosen_relayouts["P"]
    assert chosen.candidate == _candidate("C", 0, 5000.0)
    assert chosen.destination_address == copy.address
    # The copy is real LX space, disjoint from P's per-core footprint (both
    # are alive at C's tick): [addr, addr+16) each, 64-byte capacity.
    assert 0 <= copy.address <= 64 - _PER_CORE
    assert _disjoint(copy.address, p.address), "copy overlaps its source"


def test_relayout_declines_when_dearer_than_the_spill():
    r = _solve(_pc(50000.0))
    assert r["P"].address is None, "spilling is cheaper: P must not reside"
    assert _copy(r).address is None and r["C"].chosen_relayouts == {}


def test_capacity_vetoes_a_profitable_relayout():
    # 16-byte LX: P's per-core footprint fills it, so the copy cannot coexist
    # with P at C's tick. The copy stays out no matter how profitable, and P
    # without a serving copy cannot reside (gate: match or served).
    r = _solve(_pc(5000.0), size=16)
    assert _copy(r).address is None and r["C"].chosen_relayouts == {}
    assert r["P"].address is None


def test_fallback_objective_never_decides_a_relayout():
    # Under the HBM-bytes fallback (cost_expr=None) a shuffle is unpriced and
    # would look free; every copy must be pinned out, restoring the old gate:
    # no match pair -> P spilled.
    r = _solve(_pc(5000.0), expr=None)
    assert r["P"].address is None
    assert _copy(r).address is None and r["C"].chosen_relayouts == {}


def test_copy_never_resides_without_its_source():
    # P is barred from LX by the allocator (residency_reason). Its copy has a
    # profitable price and room to spare, but a copy needs its source
    # resident: it must stay out.
    p = _producer([0, 1])
    p.residency_reason = "barred for the test"
    bufs = _with_copies(p, _consumer("C", 1, 2, [_candidate("C", 0, 1.0)]))
    r = _solve(bufs)
    assert r["P"].address is None
    assert _copy(r).address is None and r["C"].chosen_relayouts == {}


def test_gate_without_a_relayout_table_is_unchanged():
    # No match pairs and no relayout table: in_buffer is forced off exactly
    # as before (regression guard for the constrain_residency rewrite). No
    # copy exists, so no price term either.
    bufs = _with_copies(_producer([0, 1]), _consumer("C", 1, 2, []))
    assert not any(isinstance(b, RelayoutCopyBuffer) for b in bufs)
    r = _solve(bufs)
    assert r["P"].address is None and r["C"].chosen_relayouts == {}


def _fanout(relayout_cost_ns, same_view: bool):
    """P feeds C1 (uses [1, 2]) and C2 (uses [2, 3]); both edges have an empty
    match table and one priced relayout candidate. With ``same_view`` both
    candidates land on the same destination view of P (group 0); otherwise
    they are distinct views (groups 0 and 1)."""
    return _with_copies(
        _producer([0, 3]),
        _consumer("C1", 1, 2, [_candidate("C1", 0, relayout_cost_ns, 0)]),
        _consumer(
            "C2", 2, 3, [_candidate("C2", 0, relayout_cost_ns, 0 if same_view else 1)]
        ),
    )


def test_two_consumers_on_one_view_share_one_copy_and_one_charge():
    """Fan-out to two consumers wanting the SAME destination view: one copy
    spanning both consumers' ticks, one charge. Spilling costs 20000; the
    shuffle 12000. Charged per edge it would be 24000 and the solver would
    spill; charged per copy it fires."""
    bufs = _fanout(12000.0, same_view=True)
    assert sum(isinstance(b, RelayoutCopyBuffer) for b in bufs) == 1
    r = _solve(bufs)
    copy = _copy(r)
    assert r["P"].address is not None, "one shared shuffle beats the spill"
    assert (copy.start_time, copy.end_time) == (1, 3) and copy.address is not None
    a1, a2 = r["C1"].chosen_relayouts["P"], r["C2"].chosen_relayouts["P"]
    assert a1.candidate.group == a2.candidate.group == 0
    assert a1.destination_address == a2.destination_address == copy.address
    assert _disjoint(copy.address, r["P"].address)


def test_two_consumers_on_different_views_get_two_copies():
    """Distinct destination views cannot share: two copies, two charges. At
    12000 each the pair (24000) loses to the 20000 spill, so P is spilled;
    at 8000 each (16000) both fire, each copy alive at its own tick."""
    r = _solve(_fanout(12000.0, same_view=False))
    assert r["P"].address is None
    assert _copy(r, 0).address is None and _copy(r, 1).address is None
    assert r["C1"].chosen_relayouts == {} == r["C2"].chosen_relayouts

    r = _solve(_fanout(8000.0, same_view=False))
    assert r["P"].address is not None
    assert _copy(r, 0).address is not None and _copy(r, 1).address is not None
    assert (
        r["C1"].chosen_relayouts["P"].candidate.group == 0
        and r["C2"].chosen_relayouts["P"].candidate.group == 1
    )
    assert r["C1"].chosen_relayouts["P"].destination_address == _copy(r, 0).address
    assert r["C2"].chosen_relayouts["P"].destination_address == _copy(r, 1).address


def _distant(relayout_cost_ns, blocker_size):
    """P (uses [0, 10]) feeds C1 at tick 1 and C2 at tick 8, same destination
    view, so the copy lives [1, 9). B (uses [3, 7], no parents) needs
    ``blocker_size`` bytes of LX in the gap. With a 64-byte LX, P's slice (16)
    plus the held copy (16) leave 32: a 40-byte B fits only if the copy is
    not held."""
    return _with_copies(
        _producer([0, 10]),
        _consumer("C1", 1, 2, [_candidate("C1", 0, relayout_cost_ns)]),
        _consumer("C2", 8, 9, [_candidate("C2", 0, relayout_cost_ns)]),
        CoreDivisionBuffer(
            "B",
            blocker_size,
            [3, 7],
            core_divisions=[CoreDivision(output_splits={1: 1})],
        ),
    )


def test_one_copy_per_group_spans_the_gap_or_the_source_spills():
    """The model holds ONE copy across a group's consumers and never
    re-shuffles a released view. Holding it through B spills B; the solver
    weighs that against spilling P. Dear B (30000): P spills (20000) and B
    stays. Cheap B (1000): the copy is held, both consumers read it, B
    spills."""
    r = _solve(_distant(5000.0, blocker_size=40), B=30000.0)
    assert r["P"].address is None and r["B"].address is not None
    assert _copy(r).address is None
    assert r["C1"].chosen_relayouts == {} == r["C2"].chosen_relayouts

    r = _solve(_distant(5000.0, blocker_size=40), B=1000.0)
    copy = _copy(r)
    assert r["P"].address is not None and r["B"].address is None
    assert (copy.start_time, copy.end_time) == (1, 9) and copy.address is not None
    a1, a2 = r["C1"].chosen_relayouts["P"], r["C2"].chosen_relayouts["P"]
    assert a1.destination_address == a2.destination_address == copy.address


def test_copy_spans_a_consumer_outside_the_group():
    """C2 (tick 4) reads P through a free slicing MATCH and offers no relayout
    candidate, so it is not a consumer of the copy at all; C1 (tick 1) and C3
    (tick 7) must relayout. One copy spans over C2 in time and serves both,
    while C2 keeps reading the source directly."""
    bufs = _with_copies(
        _producer([0, 9]),
        _consumer("C1", 1, 2, [_candidate("C1", 0, 5000.0)]),
        _consumer("C2", 4, 5, [], matches=[(0, 0)]),
        _consumer("C3", 7, 8, [_candidate("C3", 0, 5000.0)]),
    )
    r = _solve(bufs)
    copy = _copy(r)
    assert copy.consumers == ("C1", "C3")
    assert r["P"].address is not None and copy.address is not None
    assert r["C2"].chosen_relayouts == {}
    assert (
        r["C1"].chosen_relayouts["P"].destination_address
        == r["C3"].chosen_relayouts["P"].destination_address
        == copy.address
    )


def test_charge_follows_the_chosen_source_division():
    """P offers two divisions whose shuffles into the same destination view
    price differently (5000 vs 3000 ns). The solver picks the cheaper source
    division, and the KroneckerDelta term charges that price: with the
    spill at 4000 only the cheaper division makes the relayout worth it."""
    bufs = _with_copies(
        _producer([0, 3], divisions=2),
        _consumer("C", 1, 2, [_candidate("C", 0, 5000.0), _candidate("C", 1, 3000.0)]),
    )
    r = _solve(bufs, P=4000.0)
    assert r["P"].address is not None and r["P"].chosen_division == 1
    chosen = r["C"].chosen_relayouts["P"]
    assert chosen.candidate == _candidate("C", 1, 3000.0), "cheaper division wins"
    assert chosen.destination_address == _copy(r).address


def test_a_consumer_reads_the_copy_only_under_a_priced_pair():
    """P has two divisions but only division 0 is priced into C's view, and
    that shuffle (50000) is dearer than the spill. Division 1 is unpriced, so
    the served literal cannot pin it: P must spill rather than read the copy
    under a pair the candidates never listed."""
    bufs = _with_copies(
        _producer([0, 3], divisions=2),
        _consumer("C", 1, 2, [_candidate("C", 0, 50000.0)]),
    )
    r = _solve(bufs)
    assert r["P"].address is None and r["C"].chosen_relayouts == {}
    assert _copy(r).address is None


# ---------------------------------------------------------------------------
# The commit path's view
# ---------------------------------------------------------------------------


def test_fired_groups_regroup_edges_by_source_and_view():
    """The commit path's view of a solve: fired edges regroup into one group
    per (parent, view), members sorted by consumer, and the plan carries the
    candidate's views and the copy's address. Members that disagree on
    placement are a solver invariant violation, not a plan."""
    c1 = ChosenRelayout(_candidate("C1", 0, 5000.0), 16)
    c2 = ChosenRelayout(_candidate("C2", 0, 5000.0), 16)
    other = ChosenRelayout(_candidate("C3", 0, 7000.0, group=1), 48)
    groups = FiredRelayoutGroup.from_chosen([other, c2, c1])
    assert [(g.group, g.consumer_names) for g in groups] == [
        (0, ("C1", "C2")),
        (1, ("C3",)),
    ]
    plan = groups[0].plan(source_address=0)
    assert (plan.source_name, plan.consumer_names) == ("P", ("C1", "C2"))
    assert (plan.source_view, plan.destination_view) == (_view(0), _view(1))
    assert (plan.num_cores, plan.source_address, plan.destination_address) == (
        4,
        0,
        16,
    )
    assert c1.scaled(128).destination_address == 16 * 128
    with pytest.raises(AssertionError, match="disagree"):
        FiredRelayoutGroup.from_chosen(
            [c1, ChosenRelayout(_candidate("C2", 0, 5000.0), 64)]
        )
    with pytest.raises(ValueError, match="equal views"):
        RelayoutCandidate("P", "C", 0, 0, 0, _view(0), _view(0), 1.0, 16, 16)
    # The core counts are the views': a view without one cannot be a candidate.
    bare = PerCoreView(((1, 4),), ((1, sympy.Mod(_CORE + 1, 4)),))
    with pytest.raises(ValueError, match="no physical core count"):
        RelayoutCandidate("P", "C", 0, 0, 0, _view(0), bare, 1.0, 16, 16)
