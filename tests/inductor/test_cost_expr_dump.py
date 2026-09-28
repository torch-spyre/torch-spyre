# Copyright 2025 IBM Corporation
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
"""The cost-expression dump (``SPYRE_DUMP_COST_EXPR_FILE``): the objective's
per-bundle terms, the solved symbol bindings and the evaluated prices, written
as one JSON record per co-optimized graph. No device, no graph: the record is
built from handcrafted buffers and terms."""

import json

import pytest
import sympy

from torch_spyre._inductor.cost_model import CostParams
from torch_spyre._inductor.dump_common import emit_json_line
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator
from torch_spyre._inductor.scratchpad.lx_relayout import RelayoutCandidate
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    RelayoutCharge,
    cost_expr_record,
    solved_bindings,
)

_CORE = sympy.Symbol("core_id")


def _view(slot: int) -> PerCoreView:
    return PerCoreView(((1, 4),), ((1, sympy.Mod(_CORE + slot, 4)),), 4)


def _buffers():
    p = CoreDivisionBuffer(
        "P",
        64,
        [0, 2],
        core_divisions=[CoreDivision(splits={1: 4}), CoreDivision(splits={0: 4})],
    )
    c = CoreDivisionBuffer(
        "C",
        64,
        [1, 2],
        core_divisions=[CoreDivision(splits={1: 4})],
        parents=["P"],
        cd_parent_relayouts={
            "P": [
                RelayoutCandidate(
                    parent="P",
                    consumer="C",
                    source_division=1,
                    consumer_division=0,
                    group=0,
                    source_view=_view(0),
                    destination_view=_view(1),
                    cost_ns=3000.0,
                    source_footprint_bytes=16,
                    destination_footprint_bytes=16,
                )
            ]
        },
    )
    return p, c


def test_solved_bindings_read_the_plan_like_the_annealer():
    p, c = _buffers()
    p.address, p.chosen_division = 0, 1
    c.address, c.chosen_division = None, 0
    b = solved_bindings([p, c])
    assert b[p.sym_is_lx] == 1 and b[c.sym_is_lx] == 0
    assert b[p.sym_division] == 1 and b[c.sym_division] == 0
    # division 1 of P splits axis 0 four ways and leaves axis 1 whole.
    splits = {str(k): b[sym] for k, sym in p.sym_core_divs.items()}
    assert set(splits.values()) == {4, 1}


def test_record_evaluates_every_term_under_the_solved_plan():
    p, c = _buffers()
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c])
    p.address, p.chosen_division = 0, 1
    c.address, c.chosen_division = 16, 0
    copy.address = 32
    spill = 4000 * (1 - p.sym_is_lx) + 2000 * (1 - c.sym_is_lx)
    bundle_terms = [
        (["P"], 4000 * (1 - p.sym_is_lx)),
        (["C"], 2000 * (1 - c.sym_is_lx)),
    ]
    cost_expr = spill + copy.cost_term()
    rec = cost_expr_record(cost_expr, bundle_terms, [p, c, copy], CostParams())
    assert rec["buffers"] == ["P", "C"]
    assert [b["value_ns"] for b in rec["bundles"]] == [0.0, 0.0]
    (rt,) = rec["relayout_terms"]
    assert rt["source"] == "P" and rt["resident"] is True
    assert rt["value_ns"] == 3000.0, "P chose division 1, priced 3000 ns"
    assert rec["objective_ns"] == 3000.0
    assert rec["params"]["bw_peak_gbps"] == CostParams().bw_peak_gbps
    # srepr round-trips, including the RelayoutCharge node.
    back = sympy.parse_expr(rt["expr"], local_dict={"RelayoutCharge": RelayoutCharge})
    assert back.free_symbols == {copy.sym_is_lx, p.sym_division}


def test_a_bundle_names_its_buffers_after_the_boundary_rewrite(monkeypatch):
    """The dump's join key survives ``charge_boundary_reads_once``.

    That pass rebuilds any op whose graph-input read an earlier bundle already
    paid for (``dataclasses.replace``), which gives the copy a new ``id()``.
    Naming the ops by identity against the caller's features then missed and
    fell back to ``OpFeatures.name`` -- the op KIND ("sub"), not the buffer --
    for exactly the shape the dump exists to explain: several bundles reading
    one graph input. The prices were right; only the names a reader joins on
    were not.

    ``estimate_bundles`` is stubbed because the shape needs TWO bundles sharing
    a graph input, and the real scheduler is what decides that.
    """
    from torch_spyre._inductor import fusion
    from torch_spyre._inductor.cost_model import ArgTraffic, OpFeatures, predict_bundles

    def reader(out, op_kind):
        return OpFeatures(
            name=op_kind,
            is_reduction=False,
            out_elems=64,
            cores=32,
            dtype_bytes=2,
            args=[
                ArgTraffic(out, "output", False, 64, is_boundary=False),
                ArgTraffic("arg0_1", "input", False, 64, is_boundary=True),
            ],
        )

    class _Op:
        def __init__(self, name):
            self.name = name

    operations = [_Op("buf0"), _Op("buf1")]
    features = {"buf0": reader("buf0", "amax"), "buf1": reader("buf1", "sub")}
    monkeypatch.setattr(fusion, "estimate_bundles", lambda ops: [[o] for o in ops])

    named = [names for names, _ in predict_bundles(operations, features)]
    assert named == [["buf0"], ["buf1"]], (
        "the second bundle's op was renamed to its op kind by the rewrite"
    )


def test_the_record_carries_the_context_the_buffers_cannot_show():
    """Graph identity, environment and solve stats ride along with the plan.

    Occupancy means nothing without the LX budget it is measured against, and
    the kernel's name and directory hash do not exist yet at solve time, so the
    graph's output names are the identity a reader joins on.
    """
    p, c = _buffers()
    p.address, p.chosen_division = 0, 1
    c.address, c.chosen_division = 16, 0
    context = {
        "op_names": {"C": "amax_1"},
        "op_ids": {"C": "op1"},
        "env": {"sencores": 8, "lx_capacity": 1234, "solver": "CpSatLayoutSolver"},
        "solve": {"status": "OPTIMAL", "solve_s": 0.4},
    }
    rec = cost_expr_record(sympy.Integer(0), [], [p, c], CostParams(), context=context)
    assert rec["env"]["lx_capacity"] == 1234 and rec["env"]["sencores"] == 8
    assert rec["solve"]["status"] == "OPTIMAL"
    assert rec["op_names"]["C"] == "amax_1", "the readable name"
    assert rec["op_ids"]["C"] == "op1", "the key the numeric dump joins on"
    # And a record built without context is unchanged, so old readers still work.
    plain = cost_expr_record(sympy.Integer(0), [], [p, c], CostParams())
    assert "env" not in plain and "solve" not in plain


def test_context_may_not_redefine_the_record():
    """Context describes the record; it does not get to replace it. Without the
    guard a caller key named ``bundles`` would silently drop the terms."""
    p, c = _buffers()
    p.address, p.chosen_division = 0, 1
    c.address, c.chosen_division = 16, 0
    with pytest.raises(AssertionError, match="bundles"):
        cost_expr_record(
            sympy.Integer(0), [], [p, c], CostParams(), context={"bundles": "oops"}
        )


def test_a_buffer_carries_the_reason_it_never_reached_the_solver():
    """Three outcomes a reader must not conflate: excluded before the solve,
    weighed and declined, resident. ``reason`` separates the first."""
    p, c = _buffers()
    p.residency_reason = "op not allowed"
    p.address, p.chosen_division = None, 1
    c.address, c.chosen_division = 16, 0
    divs = cost_expr_record(sympy.Integer(0), [], [p, c], CostParams())["divisions"]
    assert divs["P"]["reason"] == "op not allowed"
    assert divs["C"]["reason"] is None, "C reached the solver; bindings say the rest"


def test_priced_relayouts_record_the_copies_that_were_not_taken():
    """``relayout_terms`` holds what fired. Without the priced candidates a
    reader cannot tell "relayout was never on the table" from "it was available
    and lost"."""
    p, c = _buffers()
    p.address, p.chosen_division = 0, 1
    c.address, c.chosen_division = 16, 0
    rec = cost_expr_record(sympy.Integer(0), [], [p, c], CostParams())
    priced = rec["priced_relayouts"]
    assert list(priced) == ["C"], "keyed by the consumer, as divisions are"
    (cand,) = priced["C"]["P"]
    assert cand["cost_ns"] == 3000.0
    assert (cand["source_division"], cand["consumer_division"]) == (1, 0)


def test_emit_json_line_appends_one_record_per_call(tmp_path):
    path = tmp_path / "dump.jsonl"
    emit_json_line(str(path), {"a": 1})
    emit_json_line(str(path), {"b": sympy.Integer(2)})
    lines = path.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": "2"}]


def test_record_carries_the_divisions_the_choice_was_made_over():
    p, c = _buffers()
    # The gate admits only P's axis-1 split as a per-core match for C.
    c.cd_parent_matches = {"P": [(0, 0)]}
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c])
    p.address, p.chosen_division = 0, 1
    c.address, c.chosen_division = 16, 0
    copy.address = 32
    rec = cost_expr_record(sympy.Integer(0), [], [p, c, copy], CostParams())
    divs = rec["divisions"]
    # Relayout copies carry a single division each and are left out.
    assert list(divs) == ["P", "C"]
    assert divs["P"]["cores"] == [4, 4] and divs["P"]["chosen"] == 1
    assert divs["P"]["labels"] == ["s1/4", "s0/4"]
    # P's chosen division (1, splitting axis 0) is not one the gate admits,
    # so the record shows the residency C lost and the alternative it had.
    assert divs["C"]["matches"] == {"P": [[0, 0]]}
    assert divs["C"]["chosen"] == 0
    # ``parents`` separates "the gate weighed this edge and admitted nothing"
    # from "no edge was built at all", which have the same empty ``matches``.
    assert divs["C"]["parents"] == ["P"] and divs["P"]["parents"] == []


def test_a_parent_the_gate_refused_outright_is_still_listed():
    """The louder of the two silences, and the one issue #4655 turned on.

    ``build_residency_edge`` returning None drops the producer from
    ``cd_parent_matches`` entirely -- not an empty pair list, no key at all --
    so without ``parents`` the record cannot tell "this edge was weighed and
    admitted nothing" from "this source was refused before any division pair
    was considered". Only the second means a residency reason on the producer
    kept it out of the running.
    """
    p, c = _buffers()
    c.cd_parent_matches = {}  # the gate built no edge for P at all
    (copy,) = CoOptimizingAllocator._relayout_copy_buffers([p, c])
    p.address, p.chosen_division = 0, 1
    c.address, c.chosen_division = 16, 0
    copy.address = 32
    divs = cost_expr_record(sympy.Integer(0), [], [p, c, copy], CostParams())[
        "divisions"
    ]
    assert divs["C"]["parents"] == ["P"], "the edge exists in the graph"
    assert "P" not in divs["C"]["matches"], "and the gate refused it outright"
    # The weighed-but-empty case is the other one, and must stay distinguishable.
    c.cd_parent_matches = {"P": []}
    divs = cost_expr_record(sympy.Integer(0), [], [p, c, copy], CostParams())[
        "divisions"
    ]
    assert divs["C"]["matches"] == {"P": []}
