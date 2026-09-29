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

"""Per-core reads of replicated and partitioned matmul operands.

When a matmul's core split lies on a dim an operand does not index, every core of
that split loads its own full copy of the operand's slice from HBM. The grouped
LX-relayout sweep (2026-09-09) measured that consumer at f times its one-load bytes;
the once-per-input count under-predicted the demote penalty 10-40x. These tests pin
the rule at the three places it lives: the arg's byte function, the extractor's
stamp, and the fused-bundle de-duplication that has to tolerate the symbolic result.

Partitioned, unreused operands instead stream one slice per core. Their tests
cover the per-core delivery limit, scope guards, and symbolic solver agreement.
"""

import dataclasses
import types

import pytest
import sympy
import torch

import torch_spyre  # noqa: F401
import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
from torch_spyre._inductor import config, cost_model, spyre_hint
from torch_spyre._inductor import cost_model_pass as cmp
from torch_spyre._inductor.cost_model import (
    ArgTraffic,
    CostParams,
    OpFeatures,
    _fused_hbm_bytes,
    _partitioned_operand_read_excess,
    _replicated_operand_reads,
    explain,
    predict_ops,
)
from torch_spyre._inductor.dump_cost_model import _replication
from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS

ELEMS = 4096
BYTES = ELEMS * 2


def _arg(replication, *, resident, boundary=False):
    return ArgTraffic(
        name="arg0_1" if boundary else "buf0",
        role="input",
        is_lx=resident,
        elems=ELEMS,
        is_boundary=boundary,
        replication=replication,
    )


def _op(*args):
    return OpFeatures(
        name="bmm",
        is_reduction=True,
        out_elems=ELEMS,
        cores=8,
        dtype_bytes=2,
        args=list(args),
        is_matmul=True,
    )


# ------------------------------------------------------------- byte function


def test_a_replicated_operand_is_loaded_once_per_replica():
    assert _arg(4, resident=False).hbm_elems() == 4 * ELEMS
    assert _arg(1, resident=False).hbm_elems() == ELEMS


def test_residency_removes_every_replica_load():
    assert _arg(4, resident=True).hbm_elems() == 0


def test_a_resident_boundary_input_adds_exactly_one_clone_in_load():
    # Pinning a graph input inserts one clone-in; without residency each replica
    # core loads it. Resident, the reader is served from LX and the clone loads it
    # once; not resident, the reader pays f loads and there is no clone.
    resident = _arg(4, resident=True, boundary=True)
    assert (resident.hbm_elems(), resident.clone_in_elems()) == (0, ELEMS)
    spilled = _arg(4, resident=False, boundary=True)
    assert (spilled.hbm_elems(), spilled.clone_in_elems()) == (4 * ELEMS, 0)


def test_the_replicated_charge_stays_linear_in_symbolic_residency():
    is_lx = sympy.Symbol("is_lx")
    plain = _arg(4, resident=is_lx).hbm_elems()
    boundary = _arg(4, resident=is_lx, boundary=True)
    assert sympy.simplify(plain - 4 * ELEMS * (1 - is_lx)) == 0
    assert sympy.simplify(boundary.hbm_elems() - 4 * ELEMS * (1 - is_lx)) == 0
    assert sympy.simplify(boundary.clone_in_elems() - ELEMS * is_lx) == 0
    # replication itself may be a solver split symbol
    f = sympy.Symbol("split_n")
    assert sympy.simplify(_arg(f, resident=0).hbm_elems() - f * ELEMS) == 0


def test_replication_defaults_to_one_on_legacy_records():
    legacy = {"name": "buf0", "role": "input", "is_lx": False, "elems": ELEMS}
    op = cost_model.op_from_dict({**cost_model.op_to_dict(_op()), "args": [legacy]})
    assert op.args[0].replication == 1
    back = cost_model.op_from_dict(cost_model.op_to_dict(_op(_arg(4, resident=False))))
    assert back.args[0].replication == 4


def test_shared_load_keeps_consumer_degree_without_reloading_each_copy():
    from torch_spyre._inductor.cost_model import _shared_operand_read_excess
    from torch_spyre._inductor.work_division import _matmul_multicast_penalty

    degree, resident = sympy.symbols("degree resident", integer=True)
    arg = _arg(degree, resident=resident, boundary=True)
    arg.broadcast = True
    op, params = _op(arg), CostParams()
    assert arg.replication == degree
    assert arg.hbm_elems() == ELEMS * (1 - resident)
    assert arg.replicated_hbm_elems() == 0
    excess = _shared_operand_read_excess([op, op], params)
    for count in (1, 8, 12, 16, 32):
        expected = BYTES * (_matmul_multicast_penalty(count) - 1) / params.bw_peak_gbps
        assert float(excess.subs({degree: count, resident: 0})) == pytest.approx(
            expected
        )
        assert excess.subs({degree: count, resident: 1}) == 0
    arg.broadcast = False
    assert _shared_operand_read_excess([op], params) == 0
    assert arg.hbm_elems() == ELEMS * degree * (1 - resident)


# ------------------------------------------------------------------ the rule


def test_replication_is_the_product_of_splits_the_read_does_not_index():
    m, n, k = sympy.symbols("m n k")
    slices = {m: 2, n: 4, k: 1}
    assert _replication(m * 64 + k, slices) == 4  # B[K, N] under an M split
    assert _replication(k * 64 + n, slices) == 2  # A[M, K] under an N split
    assert _replication(m * 64 + n, slices) == 1  # the output: every split indexed
    assert _replication(None, slices) == 1
    assert _replication(m, {}) == 1


def test_replication_carries_solver_split_symbols():
    m, n = sympy.symbols("m n")
    s_m, s_n = sympy.symbols("s_m s_n")
    assert _replication(m, {m: s_m, n: s_n}) == s_n
    assert _replication(m * 64 + n, {m: s_m, n: s_n}) == 1


def test_fused_dedup_tolerates_two_symbolic_readers_of_one_input():
    # Two matmuls in one bundle reading the same external input under different
    # symbolic splits: the de-duplication must not compare symbolic byte counts
    # with ``>`` (which raises), or the whole bundle drops out of the objective.
    f1, f2 = sympy.symbols("f1 f2")
    a = ArgTraffic("arg0_1", "input", False, ELEMS, is_boundary=True, replication=f1)
    b = ArgTraffic("arg0_1", "input", False, ELEMS, is_boundary=True, replication=f2)
    r, _w = _fused_hbm_bytes([_op(a), _op(b)])
    assert r == sympy.Max(f1 * BYTES, f2 * BYTES)
    r, _w = _fused_hbm_bytes(
        [_op(_arg(2, resident=False)), _op(_arg(3, resident=False))]
    )
    assert r == (2 + 3) * BYTES  # interior buffers are not de-duplicated


# ------------------------------------------------------- per-core read rate


def _matmul(*inputs, out_elems=64, cores=32):
    """A bmm with a tiny output so the read/write turnaround term is identical
    across the variants compared below (min(R, W) stays W)."""
    return OpFeatures(
        name="bmm",
        is_reduction=True,
        out_elems=out_elems,
        cores=cores,
        dtype_bytes=2,
        args=[ArgTraffic("buf9", "output", False, out_elems), *inputs],
        is_matmul=True,
    )


def test_replicated_operand_reads_are_priced_at_the_per_core_ceiling():
    p = CostParams()
    rep = _matmul(_arg(1, resident=False), _arg(8, resident=False))
    flat = _matmul(_arg(1, resident=False), _arg(1, resident=False))
    # The replicated operand leaves the shared pool (one load at the peak) and is
    # charged as f loads at the per-core rate, each core reading its own copy.
    expected = (
        8 * BYTES / 32 / p.mm_replicated_read_gbps_per_core - BYTES / p.bw_peak_gbps
    )
    assert predict_ops([rep], p) - predict_ops([flat], p) == pytest.approx(
        expected, rel=1e-6
    )


def test_the_ladder_law_reproduces_a_measured_rung():
    # Rows-per-core ladder 2026-09-10: f=8, 256 KiB operand, 32 cores, 64 KiB per
    # core -> 29.7 / 29.9 / 29.8 / 29.4 / 28.8 us at 1 / 2 / 4 / 8 / 16 rows per core.
    p = CostParams()
    tensor_bytes, f = 256 * 1024, 8
    op = _matmul(
        _arg(1, resident=False),
        ArgTraffic("buf0", "input", False, tensor_bytes // 2, replication=f),
    )
    rep_bytes, ns = _replicated_operand_reads([op], p)
    assert rep_bytes == f * tensor_bytes
    assert ns / 1000 == pytest.approx(29.7, rel=0.1)


def test_a_resident_operand_has_no_per_core_reads():
    assert _arg(8, resident=True).replicated_hbm_elems() == 0
    # A resident boundary operand adds its one clone-in load, which is an aggregate
    # transfer priced on its own, not a per-core read.
    a = _arg(8, resident=True, boundary=True)
    assert (a.replicated_hbm_elems(), a.hbm_elems(), a.clone_in_elems()) == (
        0,
        0,
        ELEMS,
    )
    p = CostParams()
    assert _replicated_operand_reads([_matmul(a)], p) == (0, 0)


def test_per_core_pricing_is_symbolic_and_matches_the_numeric_path():
    """The co-optimizer scores this expression with the solver's split and
    residency symbols; at the chosen point it must equal the committed-path
    number, and ``replication / cores`` must cancel to the inverse of the split
    the operand indexes, the form the CP-SAT printer lowers (``inv_`` symbols)."""
    p = CostParams()
    s_m, s_n, is_lx = sympy.symbols("s_m s_n is_lx")
    sym = _matmul(
        _arg(1, resident=False),
        ArgTraffic("buf0", "input", is_lx, ELEMS, replication=s_n),
        cores=s_m * s_n,
    )
    num = _matmul(
        _arg(1, resident=False),
        ArgTraffic("buf0", "input", False, ELEMS, replication=8),
        cores=32,
    )
    _bytes, ns = _replicated_operand_reads([sym], p)
    expected = sympy.Piecewise(
        (0, sympy.Eq(s_n, 1)),
        (BYTES * (1 - is_lx) / (s_m * p.mm_replicated_read_gbps_per_core), True),
    )
    assert sympy.simplify(ns - expected) == 0
    expr = predict_ops([sym], p)
    at_point = sympy.lambdify([s_m, s_n, is_lx], expr, modules="math")
    assert at_point(4, 8, 0) == pytest.approx(predict_ops([num], p), rel=1e-9)
    # resident: the operand costs nothing, whatever the split
    resident = _matmul(
        _arg(1, resident=False),
        ArgTraffic("buf0", "input", True, ELEMS, replication=8),
        cores=32,
    )
    assert at_point(4, 8, 1) == pytest.approx(predict_ops([resident], p), rel=1e-9)

    # The same symbolic menu also contains unreplicated choices. Neither those
    # choices nor an already-substituted SymPy Integer may pay the replica rate.
    for factor in (1, sympy.Integer(1), 2):
        for resident in (False, True):
            concrete = _matmul(
                _arg(1, resident=False),
                ArgTraffic("buf0", "input", resident, ELEMS, replication=factor),
                cores=4 * factor,
            )
            assert at_point(4, factor, int(resident)) == pytest.approx(
                float(predict_ops([concrete], p)), rel=1e-9
            )


# ----------------------------------------------------- partitioned read rate

# One TP4 Granite 3.3 8B MLP gate projection at decode: x[1, 4096] @ W[4096, 3200].
K, N = 4096, 3200
W_ELEMS = K * N
W_BYTES = 2 * W_ELEMS


def _projection(n_split, k_split, *, weight_lx=False, macs=W_ELEMS, **fields):
    """The projection as the extractor records it: the weight is the graph input
    indexed by both split dims (replication 1); the activation is indexed by K
    only, so every output split replicates it."""
    return OpFeatures(
        name="bmm",
        is_reduction=True,
        out_elems=N,
        cores=n_split * k_split,
        dtype_bytes=2,
        args=[
            ArgTraffic("buf43", "output", False, N),
            ArgTraffic("buf42", "input", False, K, broadcast=True, replication=n_split),
            ArgTraffic(
                "arg12_1",
                "input",
                weight_lx,
                W_ELEMS,
                broadcast=True,
                is_boundary=True,
            ),
        ],
        reduction_cores=k_split,
        is_matmul=True,
        matmul_macs=macs,
        matmul_rows_per_core=N / n_split,
        matmul_cols_per_core=1.0,
        matmul_m_split=n_split,
        matmul_a_bytes=W_BYTES,
        matmul_b_bytes=2 * K,
        **fields,
    )


def _excess_ns(cores, p=_COST_PARAMS):
    return W_BYTES * (1 / (cores * p.mm_partitioned_read_gbps_per_core) - 1 / 150)


def test_the_planner_prices_partitioned_reads_by_default():
    assert _COST_PARAMS.mm_partitioned_read_gbps_per_core > 0
    assert (
        _COST_PARAMS.mm_partitioned_read_gbps_per_core
        == CostParams().mm_partitioned_read_gbps_per_core
    )


def test_fewer_reading_cores_pay_the_slower_delivery_and_all_cores_pay_nothing():
    p = _COST_PARAMS
    assert _partitioned_operand_read_excess([_projection(25, 1)], p) == pytest.approx(
        _excess_ns(25)
    )
    assert _partitioned_operand_read_excess([_projection(2, 16)], p) == 0
    assert _partitioned_operand_read_excess([_projection(1, 32)], p) == 0
    off = dataclasses.replace(p, mm_partitioned_read_gbps_per_core=0.0)
    for splits in ((25, 1), (2, 16)):
        op = _projection(*splits)
        added = predict_ops([op], p) - predict_ops([op], off)
        assert added == pytest.approx(_partitioned_operand_read_excess([op], p))


def test_the_decode_gate_ranking_follows_the_cores_that_stream_the_weight():
    # The measured reversal: the planner ranked 25 cores x 1 above 2 x 16 (all 32
    # cores), while the device ran the 32-core split faster. Pricing the slower
    # 25-core delivery puts the 32-core split ahead; nothing about K is guessed.
    off = dataclasses.replace(_COST_PARAMS, mm_partitioned_read_gbps_per_core=0.0)

    def price(p, splits):
        return predict_ops([_projection(*splits)], p)

    assert price(off, (25, 1)) < price(off, (2, 16))
    assert price(_COST_PARAMS, (2, 16)) < price(_COST_PARAMS, (25, 1))


def test_a_bundle_with_any_looped_op_keeps_its_price():
    looped = [
        {"loop_trip": 4},
        {"tiles_output_dim": True},
        {"tiles_reduction_dim": True},
    ]
    for fields in looped:
        bundle = [_projection(25, 1), _projection(25, 1, **fields)]
        assert _partitioned_operand_read_excess(bundle, _COST_PARAMS) == 0
    advancing = _projection(25, 1)
    advancing.args[0] = dataclasses.replace(advancing.args[0], loop_factor=4)
    assert _partitioned_operand_read_excess([advancing], _COST_PARAMS) == 0


def test_reused_replicated_resident_or_unknown_operands_keep_their_price():
    p = _COST_PARAMS
    # Row reuse: prefill rows, or a GQA group sharing one KV head, feed each
    # element to several MACs -- a regime this rate was not measured in.
    assert (
        _partitioned_operand_read_excess([_projection(25, 1, macs=4 * W_ELEMS)], p) == 0
    )
    # Unknown MAC count (legacy records carry 0).
    assert _partitioned_operand_read_excess([_projection(25, 1, macs=0)], p) == 0
    # Resident weight: nothing to deliver from HBM.
    assert (
        _partitioned_operand_read_excess([_projection(25, 1, weight_lx=True)], p) == 0
    )
    # Replicated operand: priced by _replicated_operand_reads instead.
    replicated = _projection(25, 1)
    replicated.args[2] = dataclasses.replace(replicated.args[2], replication=2)
    assert _partitioned_operand_read_excess([replicated], p) == 0


def test_one_graph_input_read_twice_in_a_bundle_is_charged_once():
    both = [_projection(25, 1), _projection(25, 1)]
    assert _partitioned_operand_read_excess(both, _COST_PARAMS) == pytest.approx(
        _excess_ns(25)
    )


def test_tp1_gqa_decode_attention_reads_are_reused_and_keep_their_price():
    # TP1 Granite decode QK: 32 query heads over 8 KV heads, 512 cached positions.
    # Each cached K element feeds the 4 query heads of its group.
    kv_elems, macs = 8 * 128 * 512, 32 * 128 * 512
    for cores in (1, 8, 16, 32):
        qk = OpFeatures(
            name="bmm",
            is_reduction=True,
            out_elems=32 * 512,
            cores=cores,
            dtype_bytes=2,
            args=[
                ArgTraffic("buf20", "output", False, 32 * 512),
                ArgTraffic("buf18", "input", False, 32 * 128),
                ArgTraffic("arg30_1", "input", False, kv_elems, is_boundary=True),
            ],
            is_matmul=True,
            matmul_macs=macs,
        )
        assert _partitioned_operand_read_excess([qk], _COST_PARAMS) == 0


def test_the_symbolic_price_equals_the_committed_price_at_every_candidate():
    """The co-optimizer scores this term over its split and residency symbols;
    at each candidate it must equal the committed-path number -- including the
    activation, whose symbolic replication is 1 exactly when N is unsplit."""
    n, k = sympy.symbols("split_n split_k", integer=True, positive=True)
    is_lx = sympy.Symbol("is_lx")
    sym = _projection(n, k, weight_lx=is_lx)
    sym.cores = n * k
    expr = _partitioned_operand_read_excess([sym], _COST_PARAMS)
    at = sympy.lambdify([n, k, is_lx], expr, modules="math")
    for n_i, k_i, lx in ((25, 1, 0), (2, 16, 0), (1, 32, 0), (1, 16, 0), (25, 1, 1)):
        concrete = _partitioned_operand_read_excess(
            [_projection(n_i, k_i, weight_lx=bool(lx))], _COST_PARAMS
        )
        assert at(n_i, k_i, lx) == pytest.approx(concrete, rel=1e-9, abs=1e-6)


def _solve_pinned(expr, menu, candidate, residency=None):
    """Lower ``expr`` as the joint planner does -- split symbols wired to one
    buffer's candidate divisions, residency a solver variable -- and solve it
    with the division (and residency) pinned. Returns the objective value."""
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import _SympyExprToCpSat

    model = cp_model.CpModel()
    division = model.new_int_var(0, len(menu) - 1, "div")
    model.add(division == candidate)
    buf = types.SimpleNamespace(division=division)
    sym_map, buffer_map = {}, {}
    for d, name in enumerate(("split_n", "split_k", "split_m")[: len(menu[0])]):
        raw = [c[d] for c in menu]
        sym_map[name] = model.new_int_var(min(raw), max(raw), name)
        model.add_element(division, raw, sym_map[name])
        buffer_map[name] = (buf, raw)
    if residency is not None:
        sym_map["is_lx_w"] = model.new_bool_var("is_lx_w")
        model.add(sym_map["is_lx_w"] == residency)
    model.minimize(_SympyExprToCpSat(model, sym_map, buffer_map).convert(expr))
    solver = cp_model.CpSolver()
    assert solver.Solve(model) == cp_model.OPTIMAL
    return solver.ObjectiveValue()


def test_cp_sat_keeps_the_term_at_every_candidate():
    """Lower the symbolic term exactly as the joint planner does and solve it
    with each candidate division pinned: the integerized objective must carry
    the priced excess, not round it away."""
    n, k = sympy.symbols("split_n split_k", integer=True, positive=True)
    sym = _projection(n, k)
    sym.cores = n * k
    expr = _partitioned_operand_read_excess([sym], _COST_PARAMS)
    menu = [(25, 1), (2, 16), (1, 32), (10, 2), (5, 4)]
    for i, (n_i, k_i) in enumerate(menu):
        exact = _excess_ns(n_i * k_i) if n_i * k_i < 32 else 0.0
        got = _solve_pinned(expr, menu, i)
        assert got == pytest.approx(exact, rel=1e-3, abs=1.0)


def test_cp_sat_follows_symbolic_weight_replication_and_residency():
    """The weight's replication and residency are both solver symbols here: a
    third split the weight does not index replicates it (priced elsewhere), and
    residency removes its HBM read. The lowered term must price exactly the
    partitioned, non-resident candidates."""
    n, k, m = sympy.symbols("split_n split_k split_m", integer=True, positive=True)
    is_lx = sympy.Symbol("is_lx_w", integer=True, nonnegative=True)
    sym = _projection(n, k, weight_lx=is_lx)
    sym.cores = n * k * m
    sym.args[2] = dataclasses.replace(sym.args[2], replication=m)
    expr = _partitioned_operand_read_excess([sym], _COST_PARAMS)
    menu = [(25, 1, 1), (10, 2, 1), (10, 1, 2), (5, 2, 2), (2, 16, 1)]
    for i, (n_i, k_i, m_i) in enumerate(menu):
        for resident in (0, 1):
            cores = n_i * k_i * m_i
            partitioned = m_i == 1 and not resident
            exact = _excess_ns(cores) if partitioned and cores < 32 else 0.0
            got = _solve_pinned(expr, menu, i, residency=resident)
            assert got == pytest.approx(exact, rel=1e-3, abs=1.0), (menu[i], resident)


def test_explain_reports_the_limit():
    text = explain([_projection(25, 1)], _COST_PARAMS)
    assert "partitioned-read core limit" in text
    assert "partitioned-read core limit" not in explain(
        [_projection(2, 16)], _COST_PARAMS
    )


# -------------------------------------------------------- extractor, on device


def test_extractor_stamps_the_matmul_operand_the_core_split_replicates(monkeypatch):
    """The relayout sweep's gather fixture at 8 cores: neg(value) under {H, Lk:4}
    feeds bmm(attention, hidden) under {H, Lq:4}. ``hidden`` is not indexed by Lq,
    so each of the 4 Lq cores per head loads it: replication 4. ``attention`` is
    indexed by every split dim: 1. The pointwise producer's operands stay 1
    whatever their indexing (the rung-G once-per-kernel rule)."""
    captured: dict = {}
    real = cmp.extract_op_features

    def spy(op):
        feats = real(op)
        captured[feats.name] = feats
        return feats

    monkeypatch.setattr(cmp, "extract_op_features", spy)
    batch, key, query, width = 2, 128, 4, 64
    for name, size in (("H", batch), ("Lk", key), ("Lq", query), ("D", width)):
        _pnd.declare_tensor_dim(name, size)
    torch.manual_seed(0)
    value = torch.randn(batch, key, width, dtype=torch.float16)
    attention = torch.randn(batch, query, key, dtype=torch.float16)
    bias = torch.randn(width, dtype=torch.float16)

    def fn(v, a, b):
        with spyre_hint(work_div={"H": batch, "Lk": 4}):
            hidden = torch.neg(v) * b  # pointwise; ``b`` lacks H and Lk
        with spyre_hint(work_div={"H": batch, "Lq": query}):
            return torch.bmm(a, hidden)

    args = (
        _pnd.name_tensor_dims(value.to("spyre"), ["H", "Lk", "D"]),
        _pnd.name_tensor_dims(attention.to("spyre"), ["H", "Lq", "Lk"]),
        bias.to("spyre"),
    )
    torch._inductor.codecache.FxGraphCache.clear()
    torch._dynamo.reset()
    with config.patch(
        {
            "sencores": 8,
            "lx_planning": False,
            "cost_model": "1",
        }
    ):
        compiled = torch.compile(fn, dynamic=False)
        for name, size in (("H", batch), ("Lk", key), ("Lq", query), ("D", width)):
            _pnd.declare_tensor_dim(name, size)
        out = compiled(*args)
    ref = torch.bmm(attention.float(), (value.float().neg() * bias.float()))
    assert torch.allclose(out.cpu().float(), ref, rtol=2e-2, atol=2e-1)

    matmuls = [f for f in captured.values() if f.is_matmul]
    assert len(matmuls) == 1, sorted(captured)
    (bmm,) = matmuls
    by_replication = sorted(a.replication for a in bmm.args if a.role == "input")
    assert by_replication == [1, 4], [(a.name, a.replication) for a in bmm.args]
    replicated = next(a for a in bmm.args if a.replication == 4)
    assert not replicated.name.startswith("arg"), "the replicated operand is hidden"
    for f in captured.values():
        if not f.is_matmul:
            assert all(a.replication == 1 for a in f.args), f.name
