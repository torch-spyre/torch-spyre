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

pytest.importorskip("ortools")

from torch_spyre._inductor.scratchpad.ilp_solver_ortools import CpSatLayoutSolver
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    relayout_symbol,
)

_SPILL_NS = 20000.0  # what spilling P charges in the crafted objective


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
        cd_parent_relayouts={"P": [(0, 0, relayout_cost_ns)]},
    )
    return p, c


def _objective(p):
    # Spilling P costs _SPILL_NS; relaying out costs the edge's table price.
    # The solver picks whichever is cheaper - or spills if relayout cannot fit.
    return (1 - p.sym_is_lx) * _SPILL_NS + relayout_symbol("C", "P")


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
    i, j, dest_address = c.chosen_relayouts["P"]
    assert (i, j) == (0, 0)
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
    # No relayout table -> no edge cost symbol in the objective (mirrors the
    # allocator, which appends a symbol only per cd_parent_relayouts edge; an
    # unbound symbol is a KeyError in the name-keyed printer).
    expr = (1 - p.sym_is_lx) * _SPILL_NS
    result = {b.name: b for b in solver.plan_layout_and_core_divisions(expr)}
    assert result["P"].address is None
    assert result["C"].chosen_relayouts == {}
