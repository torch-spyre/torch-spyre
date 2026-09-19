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

The clone-in is its own read-only pass: a resident input's readers are served from LX,
and the clone's one load is priced separately, outside the readers' bundle turnaround.

No Spyre device or backend compiler is required; features are built directly.
"""

import logging
from types import SimpleNamespace

import pytest
import sympy

import torch_spyre._inductor.dump_cost_model as dcm
from torch_spyre._inductor import cost_model
from torch_spyre._inductor.cost_model import (
    ArgTraffic,
    CostParams,
    OpFeatures,
    _clone_in_bytes,
    _fused_hbm_bytes,
    charge_boundary_reads_once,
)

ELEMS, DTYPE = 1024, 2
BYTES = ELEMS * DTYPE


@pytest.mark.parametrize("shared_weight", [False, True])
@pytest.mark.parametrize("k_split", [1, 2])
def test_matmul_time_does_not_charge_unused_available_cores(shared_weight, k_split):
    from torch_spyre._inductor.work_division import (
        _matmul_execution_cost,
        _matmul_split_cost,
    )

    axes = ((16, 8), (64, 1), (128, 1), (256, k_split))
    used = 8 * k_split
    options = dict(shared_weight=shared_weight, include_hbm=False)
    estimate = _matmul_execution_cost(*axes, used, **options)
    assert estimate > 0
    assert _matmul_execution_cost(*axes, 32, **options) == pytest.approx(estimate)
    split = sympy.Symbol("k_split", integer=True, positive=True)
    symbolic = _matmul_execution_cost(*axes[:3], (256, split), 32, **options)
    assert float(symbolic.subs(split, k_split)) == pytest.approx(estimate)
    assert _matmul_split_cost(*axes, 32, **options) > _matmul_split_cost(
        *axes, used, **options
    )


@pytest.mark.parametrize("shared_weight", [False, True])
def test_split_sum_matmul_prices_one_corelet(shared_weight):
    from torch_spyre._inductor import work_division as wd

    split = sympy.Symbol("k_split", integer=True, positive=True)
    axes = ((2, 2), (128, 2), (256, 2), (1024, split))
    price = wd._matmul_execution_cost(
        *axes, 32, shared_weight=shared_weight, include_hbm=False
    )
    coefficient = (
        wd._PSUM_PER_CORE_ELEM_US if shared_weight else wd._BMM_PSUM_PER_CORE_ELEM_US
    )
    for k in (1, 2, 4):
        compute = (2 * 128 * 256 * 1024) / (8 * k) / wd._PEAK_MACS_US_CORE
        expected = compute * (2 if k > 1 else 1) + (k - 1) * 8192 * coefficient
        assert float(price.subs(split, k)) == pytest.approx(expected)
        assert wd._matmul_execution_cost(
            *axes[:3], (1024, k), 32, shared_weight=shared_weight, include_hbm=False
        ) == pytest.approx(expected)


def test_joint_matmul_price_is_independent_of_standalone_preferences(monkeypatch):
    from torch_spyre._inductor import work_division as wd

    op = OpFeatures(
        name="bmm",
        is_reduction=True,
        dtype_bytes=2,
        args=[],
        is_matmul=True,
        out_elems=16 * 64 * 128,
        cores=16,
        matmul_macs=16 * 64 * 128 * 256,
        matmul_rows_per_core=64,
        matmul_cols_per_core=128,
        matmul_a_bytes=64 * 256 * 2,
        matmul_b_bytes=256 * 128 * 2,
    )
    params = cost_model.CostParams(use_bundled_cost_model=False)
    before = cost_model.predict_ops([op], params)
    axes = ((16, 8), (64, 1), (128, 1), (256, 1))
    standalone = wd._matmul_split_cost(*axes, 32)
    for name in (
        "_CORE_UNDERUSE_PENALTY_US",
        "_M_TILE_UNDERFILL_PENALTY_US",
        "_M_LANE_UNDERUSE_PENALTY_US",
        "_BMM_BATCH_SPLIT_PENALTY_US",
        "_WIDE_N_TILE_PENALTY_US",
        "_LARGE_M_TILE_SHAPE_PENALTY_US",
        "_SHARED_DOWN_N_SPLIT_PENALTY_US",
    ):
        monkeypatch.setattr(wd, name, getattr(wd, name) * 2)
    assert cost_model.predict_ops([op], params) == pytest.approx(before)
    assert wd._matmul_split_cost(*axes, 32) > standalone


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


def _loads(bundles):
    """(reader bytes, clone-in bytes) per bundle after the once-rule."""
    return [
        (_fused_hbm_bytes(b)[0], _clone_in_bytes(b))
        for b in charge_boundary_reads_once(bundles)
    ]


def _read_bytes(bundles):
    """Every HBM byte loaded for inputs: the readers' own and the clones'."""
    return sum(r + c for r, c in _loads(bundles))


# --------------------------------------------------------------- input side


def test_a_resident_graph_inputs_load_moves_from_its_readers_to_the_clone():
    # All readers in ONE bundle: the clone-in performs the single load the bundle
    # would have performed itself, so residency saves no BYTES -- they only move from
    # the readers, now served from LX, to the clone.
    hbm = [[_reader("C", "buf1"), _reader("D", "buf2")]]
    lx = [
        [
            _reader("C", "buf1", resident={"arg0_1"}),
            _reader("D", "buf2", resident={"arg0_1"}),
        ]
    ]
    assert _read_bytes(lx) == _read_bytes(hbm) == BYTES
    assert _loads(hbm) == [(BYTES, 0)]
    assert _loads(lx) == [(0, BYTES)]


def test_the_clone_in_saves_the_turnaround_its_readers_no_longer_pay():
    # The same bytes are not the same time. The clone is its own read-only pass, so
    # its load is priced alone; the readers' bundle keeps only its writes and pays no
    # read/write turnaround. Measured on device (x*2 + x*3, x = 16 MiB): 417 us
    # without the clone, 222 us with it -- one read plus one write at the peak rate.
    p = CostParams()
    hbm = [_reader("C", "buf1"), _reader("D", "buf2")]
    lx = [
        _reader("C", "buf1", resident={"arg0_1"}),
        _reader("D", "buf2", resident={"arg0_1"}),
    ]
    # Each reader writes BYTES to HBM, so min(R, W) is the one load of the input.
    saved = p.rw_turnaround_ns_per_byte * BYTES
    assert cost_model.predict_ops(hbm, p) - cost_model.predict_ops(
        lx, p
    ) == pytest.approx(saved)


def test_the_clone_in_load_is_charged_once_across_bundles():
    # Readers in TWO bundles: without residency each bundle loads the input; with
    # residency ONE clone loads it and every reader is served from LX. The saving is
    # exactly one load -- not two (the bug) and not zero (charging every bundle would
    # be the opposite error).
    hbm = [[_reader("C", "buf1")], [_reader("D", "buf2")]]
    lx = [
        [_reader("C", "buf1", resident={"arg0_1"})],
        [_reader("D", "buf2", resident={"arg0_1"})],
    ]
    assert _read_bytes(hbm) - _read_bytes(lx) == BYTES
    assert _loads(lx) == [(0, BYTES), (0, 0)]


def test_charging_the_clone_in_once_does_not_disturb_a_non_resident_input():
    # The rewrite only redistributes the clone-in charge; an input in HBM is loaded
    # by every bundle that reads it either way, and there is no clone.
    bundles = [[_reader("C", "buf1")], [_reader("D", "buf2")]]
    assert _loads(bundles) == [(BYTES, 0), (BYTES, 0)]


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


def test_residency_frees_every_reader_and_charges_one_clone_in():
    # Same shape, resident: every reader in both bundles is served from LX, and the
    # one clone-in lands in bundle 1 however many readers bundle 1 has.
    resident = {"arg0_1"}
    bundles = [
        [
            _reader("C", "buf1", resident=resident),
            _reader("G", "buf5", resident=resident),
        ],
        [
            _reader("D", "buf2", resident=resident),
            _reader("E", "buf3", resident=resident),
        ],
    ]
    assert _loads(bundles) == [(0, BYTES), (0, 0)]


def test_readers_and_clone_in_stay_linear_in_symbolic_residency():
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
    expected = [(BYTES - BYTES * is_lx, BYTES * is_lx), (BYTES - BYTES * is_lx, 0)]
    for (reader, clone), (want_reader, want_clone) in zip(_loads(bundles), expected):
        assert sympy.simplify(reader - want_reader) == 0
        assert sympy.simplify(clone - want_clone) == 0


def test_the_once_rule_is_idempotent():
    # ``is_boundary`` survives the rewrite, so the second pass recomputes the same
    # "already seen" set and changes nothing.
    bundles = [
        [_reader("C", "buf1", resident={"arg0_1"})],
        [
            _reader("D", "buf2", resident={"arg0_1"}),
            _reader("E", "buf3", resident={"arg0_1"}),
        ],
    ]
    once = charge_boundary_reads_once(bundles)
    assert _loads(once) == _loads(bundles)


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


def test_a_boundary_reads_bytes_are_conserved_under_symbolic_residency():
    """The solver objective must stay linear in ``sym_is_lx``: residency moves a
    boundary read's bytes between the reader and the clone without changing their sum,
    so the variable's weight comes from where they are priced, not from how many."""
    sym = sympy.Symbol("is_lx_arg0", integer=True)
    boundary = ArgTraffic("arg0_1", "input", sym, ELEMS, is_boundary=True)
    interior = ArgTraffic("buf1", "input", sym, ELEMS, is_boundary=False)
    assert sym in sympy.sympify(boundary.hbm_elems()).free_symbols
    assert sympy.simplify(boundary.hbm_elems() + boundary.clone_in_elems()) == ELEMS
    assert sym in sympy.sympify(interior.hbm_elems()).free_symbols
    assert interior.clone_in_elems() == 0


def test_a_legacy_record_falls_back_to_the_arg_naming_convention():
    """Records captured before the field existed keep the name heuristic
    ``_fused_hbm_bytes`` already used to de-duplicate external reads."""
    op = _reader("C", "buf1", resident={"arg0_1"})
    d = cost_model.op_to_dict(op)
    for a in d["args"]:
        a.pop("is_boundary")
    back = cost_model.op_from_dict(d)
    assert [a.is_graph_boundary for a in back.args] == [False, True]
    assert (back.read_bytes(), back.clone_in_bytes()) == (0, BYTES)


def test_an_explicit_stamp_survives_the_round_trip():
    op = _reader("C", "buf1", resident={"arg0_1"})
    # An arg named like a graph input but stamped interior must NOT be charged: the
    # stamp is the answer, the name only the fallback.
    op.args[1].is_boundary = False
    back = cost_model.op_from_dict(cost_model.op_to_dict(op))
    assert back.args[1].is_boundary is False
    assert (back.read_bytes(), back.clone_in_bytes()) == (0, 0)


# --------------------------------------------------------------- stamping


class _FakeMutationLayout:
    def __init__(self, target):
        self._target = target

    def get_buffer(self):
        return SimpleNamespace(get_name=lambda: self._target)


def _op(name, layout):
    return SimpleNamespace(name=name, get_name=lambda: name, get_layout=lambda: layout)


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


def test_an_unreadable_op_is_unknown_not_authoritatively_interior(caplog):
    """An op the stamp cannot be computed for yields ``None``, and says so.

    ``False`` here is authoritative "interior write", and the output side has no
    naming-convention fallback to recover from a wrong one -- residency would free a
    store the graph boundary still performs, silently reinstating #4271. The only
    remaining signal is the log line, so both are pinned."""

    def _raises():
        raise RuntimeError("layout is gone")

    op = SimpleNamespace(
        name="buf_unreadable", get_name=lambda: "buf7", get_layout=_raises
    )
    with caplog.at_level(logging.WARNING, logger="spyre.inductor.cost_model"):
        assert dcm._writes_graph_output(op, {"arg0_1"}) is None
    assert "buf_unreadable" in caplog.text


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


def test_the_extractor_leaves_an_unreadable_write_unstamped(monkeypatch):
    """Through the real extractor: a mutation layout whose target cannot be resolved
    (the shape of failure that keeps ``op.get_layout()`` itself usable) must reach
    ``ArgTraffic`` as ``None``, not ``False``."""

    class _BrokenMutationLayout:
        allocation = None
        device_layout = None

        def get_buffer(self):
            raise RuntimeError("target buffer is gone")

    op = _extractable_op("buf_broken_target", ["arg0_1"])
    op.get_layout = lambda: _BrokenMutationLayout()
    graph = _StubGraph(inputs=["arg0_1"], outputs=["buf9"])
    monkeypatch.setattr(dcm, "MutationLayoutSHOULDREMOVE", _BrokenMutationLayout)
    assert _stamps(op, graph)[("output", "op_buf_broken_target")] is None


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
