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

"""Graph-boundary traffic in the cost model (issue #4271).

Pinning a graph input or a graph output into LX does not remove its HBM transfer: the
scratchpad planner pins it by inserting a CLONE, which still performs that one load (or
store). Before this was modelled, the objective freed every load of a resident input and
charged nobody for the clone-in -- a credit that was 65% of the predicted cost on
softmax, and steered the anneal into plans 5-6% worse on flash.

No Spyre device or backend compiler is required; features are built directly.
"""

from types import SimpleNamespace

import pytest
import sympy

import torch_spyre._inductor.dump_cost_model as dcm
from torch_spyre._inductor import cost_model
from torch_spyre._inductor.cost_model import (
    ArgTraffic,
    OpFeatures,
    _fused_hbm_bytes,
    charge_boundary_reads_once,
)

ELEMS, DTYPE = 1024, 2
BYTES = ELEMS * DTYPE


def _reader(name, out, *, input_name="arg0_1", resident=(), resident_expr=None):
    """A pointwise op reading the graph input ``input_name`` and writing ``out``.

    ``resident_expr`` puts the input's residency under a solver decision variable
    instead of a bool, so a test can inspect the objective's slope.
    """
    in_is_lx = resident_expr if resident_expr is not None else input_name in resident
    return OpFeatures(
        name=name,
        is_reduction=False,
        out_elems=ELEMS,
        cores=1,
        dtype_bytes=DTYPE,
        args=[
            ArgTraffic(
                name=out,
                role="output",
                is_lx=out in resident,
                elems=ELEMS,
                is_boundary=False,
            ),
            ArgTraffic(
                name=input_name,
                role="input",
                is_lx=in_is_lx,
                elems=ELEMS,
                is_boundary=True,
            ),
        ],
    )


def _writer(out, *, is_boundary, resident=False):
    """An op whose only traffic is its own write."""
    return OpFeatures(
        name="producer",
        is_reduction=False,
        out_elems=ELEMS,
        cores=1,
        dtype_bytes=DTYPE,
        args=[
            ArgTraffic(
                name=out,
                role="output",
                is_lx=resident,
                elems=ELEMS,
                is_boundary=is_boundary,
            )
        ],
    )


def _read_bytes(bundles):
    return sum(_fused_hbm_bytes(b)[0] for b in charge_boundary_reads_once(bundles))


# --------------------------------------------------------------- input side


def test_a_resident_graph_input_read_is_still_charged():
    # All readers in ONE bundle: the clone-in performs the single load the bundle
    # would have performed itself, so residency saves NOTHING. This is the exact
    # case in the issue's corpus -- every clone-eligible input, every graph.
    hbm = [[_reader("C", "buf1"), _reader("D", "buf2")]]
    lx = [
        [
            _reader("C", "buf1", resident={"arg0_1"}),
            _reader("D", "buf2", resident={"arg0_1"}),
        ]
    ]
    assert _read_bytes(lx) == _read_bytes(hbm)


def test_the_clone_in_load_is_charged_once_across_bundles():
    # Readers in TWO bundles: without residency each bundle loads the input; with
    # residency ONE clone loads it and the second bundle is served from LX. The
    # saving is exactly one load -- not two (the bug) and not zero (charging every
    # bundle would be the opposite error).
    hbm = [[_reader("C", "buf1")], [_reader("D", "buf2")]]
    lx = [
        [_reader("C", "buf1", resident={"arg0_1"})],
        [_reader("D", "buf2", resident={"arg0_1"})],
    ]
    assert _read_bytes(hbm) - _read_bytes(lx) == BYTES
    assert _read_bytes(lx) == BYTES


def test_charging_the_clone_in_once_does_not_disturb_a_non_resident_input():
    # The rewrite only redistributes the boundary charge; an input in HBM is loaded
    # by every bundle that reads it either way.
    bundles = [[_reader("C", "buf1")], [_reader("D", "buf2")]]
    assert _read_bytes(bundles) == 2 * BYTES


def test_a_later_bundle_with_several_readers_still_loads_the_input_once():
    # The regression the once-rule is easy to write wrong: clearing the boundary STAMP
    # (rather than only the charge) would also clear the key ``_fused_hbm_bytes``
    # de-duplicates external reads on, so a fused kernel reading the input in three ops
    # would be charged three loads instead of one. Bundle 2 is the softmax shape --
    # ``amax`` and ``sub`` reading the same input in one kernel.
    bundles = [
        [_reader("C", "buf1")],
        [_reader("D", "buf2"), _reader("E", "buf3"), _reader("F", "buf4")],
    ]
    assert [_fused_hbm_bytes(b)[0] for b in charge_boundary_reads_once(bundles)] == [
        BYTES,
        BYTES,
    ]


def test_residency_frees_a_later_multi_reader_bundle_entirely():
    # Same shape, resident: the clone pays one load in bundle 1 and every reader in
    # bundle 2 is served from LX. The saving must be one load, not one per reader.
    resident = {"arg0_1"}
    bundles = [
        [_reader("C", "buf1", resident=resident)],
        [
            _reader("D", "buf2", resident=resident),
            _reader("E", "buf3", resident=resident),
        ],
    ]
    assert [_fused_hbm_bytes(b)[0] for b in charge_boundary_reads_once(bundles)] == [
        BYTES,
        0,
    ]


def test_a_later_bundles_readers_stay_linear_in_symbolic_residency():
    # The slope, not just the constant, has to be right: an over-counted later bundle
    # over-rewards pinning the input by (readers - 1)x in the solver's objective.
    is_lx = sympy.Symbol("is_lx")
    bundles = [
        [_reader("C", "buf1", resident_expr=is_lx)],
        [
            _reader("D", "buf2", resident_expr=is_lx),
            _reader("E", "buf3", resident_expr=is_lx),
        ],
    ]
    rewritten = charge_boundary_reads_once(bundles)
    assert (
        sympy.simplify(_fused_hbm_bytes(rewritten[1])[0] - (BYTES - BYTES * is_lx)) == 0
    )


def test_the_once_rule_is_idempotent():
    # ``is_boundary`` survives the rewrite, so the second pass recomputes the same
    # "already seen" set and changes nothing.
    bundles = [[_reader("C", "buf1")], [_reader("D", "buf2"), _reader("E", "buf3")]]
    once = charge_boundary_reads_once(bundles)
    twice = charge_boundary_reads_once(once)
    assert [_fused_hbm_bytes(b)[0] for b in once] == [
        _fused_hbm_bytes(b)[0] for b in twice
    ]


# --------------------------------------------------------------- output side


def test_a_resident_graph_output_write_is_still_charged():
    # Mirror image (#4261): the planner gives the LX address to the buffer and makes
    # a clone the graph output, so the write-out still happens.
    assert _writer("buf9", is_boundary=True, resident=True).write_bytes() == BYTES


def test_a_resident_intermediate_is_still_free():
    # The guard against over-charging: only BOUNDARY traffic survives residency.
    assert _writer("buf9", is_boundary=False, resident=True).write_bytes() == 0
    interior = OpFeatures(
        name="reader",
        is_reduction=False,
        out_elems=ELEMS,
        cores=1,
        dtype_bytes=DTYPE,
        args=[
            ArgTraffic("buf2", "output", False, ELEMS, is_boundary=False),
            ArgTraffic("buf1", "input", True, ELEMS, is_boundary=False),
        ],
    )
    assert interior.read_bytes() == 0


# --------------------------------------------------------------- plumbing


def test_predict_by_bundle_applies_the_once_rule(monkeypatch):
    """The graph-level rule has to be reached through the real entry point."""
    bundles = [
        [_reader("C", "buf1", resident={"arg0_1"})],
        [_reader("D", "buf2", resident={"arg0_1"})],
    ]
    monkeypatch.setattr(
        cost_model, "group_features_by_bundle", lambda ops, feats: bundles
    )
    charged_once = sum(
        cost_model.predict_ops(b) for b in charge_boundary_reads_once(bundles)
    )
    charged_twice = sum(cost_model.predict_ops(b) for b in bundles)
    assert cost_model.predict_by_bundle([], {}) == pytest.approx(charged_once)
    assert charged_once < charged_twice


def test_a_boundary_arg_stays_constant_under_symbolic_residency():
    """The solver objective must stay linear in ``sym_is_lx``: a boundary arg's bytes
    no longer depend on the residency variable at all."""
    sym = sympy.Symbol("is_lx_arg0", integer=True)
    boundary = ArgTraffic("arg0_1", "input", sym, ELEMS, is_boundary=True)
    interior = ArgTraffic("buf1", "input", sym, ELEMS, is_boundary=False)
    assert boundary.hbm_elems() == ELEMS
    assert sym in sympy.sympify(interior.hbm_elems()).free_symbols


def test_a_legacy_record_falls_back_to_the_arg_naming_convention():
    """Records captured before the field existed keep the name heuristic
    ``_fused_hbm_bytes`` already used to de-duplicate external reads."""
    op = _reader("C", "buf1", resident={"arg0_1"})
    d = cost_model.op_to_dict(op)
    for a in d["args"]:
        a.pop("is_boundary")
    back = cost_model.op_from_dict(d)
    assert [a.is_graph_boundary for a in back.args] == [False, True]
    assert back.read_bytes() == BYTES


def test_an_explicit_stamp_survives_the_round_trip():
    op = _reader("C", "buf1", resident={"arg0_1"})
    # An arg named like a graph input but stamped interior must NOT be charged: the
    # stamp is the answer, the name only the fallback.
    op.args[1].is_boundary = False
    back = cost_model.op_from_dict(cost_model.op_to_dict(op))
    assert back.args[1].is_boundary is False
    assert back.read_bytes() == 0


# --------------------------------------------------------------- stamping


class _FakeMutationLayout:
    def __init__(self, target):
        self._target = target

    def get_buffer(self):
        return SimpleNamespace(get_name=lambda: self._target)


def _op(name, layout):
    return SimpleNamespace(get_name=lambda: name, get_layout=lambda: layout)


def test_writes_graph_output_follows_the_mutation_target(monkeypatch):
    """A ``MutationLayoutSHOULDREMOVE`` op writes into ANOTHER buffer, and it is that
    target the graph returns -- ``x.add_(1); return x`` has op ``buf2`` writing
    ``arg0_1``. Keyed on the op's own name, the write would go uncharged."""
    monkeypatch.setattr(dcm, "MutationLayoutSHOULDREMOVE", _FakeMutationLayout)
    outputs = {"arg0_1"}
    assert dcm._writes_graph_output(_op("buf2", _FakeMutationLayout("arg0_1")), outputs)
    assert not dcm._writes_graph_output(
        _op("buf2", _FakeMutationLayout("buf1")), outputs
    )


def test_a_returned_input_has_no_output_side_write_to_charge():
    """``return x.t(), x*2`` puts ``arg0_1`` in BOTH name sets, but no op writes it --
    it reaches the model only as a read. The op that merely READS it must not be
    charged for a write it does not perform."""
    outputs = {"arg0_1", "buf0"}
    assert dcm._writes_graph_output(_op("buf0", object()), outputs)  # its own write
    assert not dcm._writes_graph_output(_op("buf1", object()), outputs)


def test_boundary_names_are_unavailable_without_a_graph():
    """The extractor also runs from offline tooling; a missing ``V.graph`` must leave
    args unstamped rather than raise."""
    assert dcm._graph_boundary_names() is None


def test_no_graph_leaves_args_unstamped_rather_than_stamping_them_false():
    """``None`` and ``False`` are NOT interchangeable here. ``False`` is authoritative,
    so it would suppress the naming-convention fallback -- and with it the external-read
    de-duplication in ``_fused_hbm_bytes``, which keys on the same predicate."""
    unstamped = ArgTraffic(
        name="arg0_1", role="input", is_lx=False, elems=ELEMS, is_boundary=None
    )
    stamped_false = ArgTraffic(
        name="arg0_1", role="input", is_lx=False, elems=ELEMS, is_boundary=False
    )
    assert unstamped.is_graph_boundary
    assert not stamped_false.is_graph_boundary


# ------------------------------------------------- stamping, through the extractor


class _StubGraph:
    """Minimal stand-in for ``GraphLowering``. ``extract_op_features`` asks a graph for
    the two boundary name sets and for buffers it may not resolve; everything else it
    reaches for is on the op."""

    def __init__(self, inputs, outputs):
        self.graph_input_names = list(inputs)
        self._outputs = list(outputs)

    def get_output_names(self):
        return list(self._outputs)

    def get_buffer(self, name):
        return None


def _extractable_op(name, reads):
    """An op the real extractor can walk: one HBM write and one read per name."""
    layout = SimpleNamespace(allocation=None, device_layout=None)
    return SimpleNamespace(
        name=name,
        data=None,
        get_name=lambda: name,
        get_operation_name=lambda: f"op_{name}",
        get_layout=lambda: layout,
        get_dtype=lambda: SimpleNamespace(itemsize=2),
        get_size=lambda: [64],
        get_read_writes=lambda: SimpleNamespace(
            reads=[SimpleNamespace(name=r, index=None) for r in reads],
            writes=[],
        ),
    )


def _stamps(op, graph):
    """{(role, name): is_boundary} as the real extractor stamps them under ``graph``."""
    from torch._inductor.virtualized import V

    with V.set_graph_handler(graph):
        feats = dcm.extract_op_features(op)
    return {(a.role, a.name): a.is_boundary for a in feats.args}


def test_the_extractor_stamps_reads_of_graph_inputs():
    """Covers the wiring itself: without this, the stamping line could be deleted and
    every other test in this file would still pass."""
    graph = _StubGraph(inputs=["arg0_1"], outputs=["buf9"])
    stamps = _stamps(_extractable_op("buf1", ["arg0_1", "buf0"]), graph)
    assert stamps[("input", "arg0_1")] is True
    assert stamps[("input", "buf0")] is False


def test_the_extractor_stamps_the_write_of_a_graph_output():
    graph = _StubGraph(inputs=["arg0_1"], outputs=["buf1"])
    assert _stamps(_extractable_op("buf1", ["arg0_1"]), graph)[("output", "op_buf1")]
    graph = _StubGraph(inputs=["arg0_1"], outputs=["buf9"])
    assert not _stamps(_extractable_op("buf1", ["arg0_1"]), graph)[
        ("output", "op_buf1")
    ]


def test_a_buffer_that_is_both_input_and_output_is_stamped_per_role():
    """``x.add_(1); return x``: the read of ``arg0_1`` and the write that returns it are
    two distinct transfers, and resolving the stamp per (arg, role) is what keeps them
    from colliding."""
    graph = _StubGraph(inputs=["arg0_1"], outputs=["arg0_1", "buf1"])
    stamps = _stamps(_extractable_op("buf1", ["arg0_1"]), graph)
    assert stamps[("input", "arg0_1")] is True
    assert stamps[("output", "op_buf1")] is True


def test_the_per_arg_io_breakdown_sums_to_its_own_total():
    """``LAST_IO`` feeds ``profile_ops.py``, whose printed lines ``parse_sweep_logs.py``
    reads back. Its per-arg ``hbm_counted`` must use the same accounting as the total it
    is printed beside -- a resident boundary arg is the case where the two can diverge."""
    feats = [_reader("C", "buf1", resident={"arg0_1"})]
    dcm._record_last_io(feats)
    counted = sum(a["hbm_counted"] for o in dcm.LAST_IO["ops"] for a in o["args"])
    # The clone-in load of the resident input, plus the write of the HBM output.
    assert counted == dcm.LAST_IO["hbm_bytes"] == 2 * BYTES
