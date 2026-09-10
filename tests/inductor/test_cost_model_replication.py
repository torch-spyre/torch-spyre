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

"""Per-core re-reads of a replicated matmul operand (``ArgTraffic.replication``).

When a matmul's core split lies on a dim an operand does not index, every core of
that split loads its own full copy of the operand's slice from HBM. The grouped
LX-relayout sweep (2026-09-09) measured that consumer at f times its one-load bytes;
the once-per-input count under-predicted the demote penalty 10-40x. These tests pin
the rule at the three places it lives: the arg's byte function, the extractor's
stamp, and the fused-bundle de-duplication that has to tolerate the symbolic result.
"""

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
    _replicated_operand_reads,
    predict_ops,
)
from torch_spyre._inductor.dump_cost_model import _replication

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


def test_a_resident_boundary_input_keeps_exactly_the_clone_in_load():
    # Pinning a graph input inserts one clone-in; without residency each replica
    # core loads it. The charge is one load resident, f loads not.
    assert _arg(4, resident=True, boundary=True).hbm_elems() == ELEMS
    assert _arg(4, resident=False, boundary=True).hbm_elems() == 4 * ELEMS


def test_the_replicated_charge_stays_linear_in_symbolic_residency():
    is_lx = sympy.Symbol("is_lx")
    plain = _arg(4, resident=is_lx).hbm_elems()
    boundary = _arg(4, resident=is_lx, boundary=True).hbm_elems()
    assert sympy.simplify(plain - 4 * ELEMS * (1 - is_lx)) == 0
    assert sympy.simplify(boundary - ELEMS * (is_lx + 4 * (1 - is_lx))) == 0
    # replication itself may be a solver split symbol
    f = sympy.Symbol("split_n")
    assert sympy.simplify(_arg(f, resident=0).hbm_elems() - f * ELEMS) == 0


def test_replication_defaults_to_one_on_legacy_records():
    legacy = {"name": "buf0", "role": "input", "is_lx": False, "elems": ELEMS}
    op = cost_model.op_from_dict({**cost_model.op_to_dict(_op()), "args": [legacy]})
    assert op.args[0].replication == 1
    back = cost_model.op_from_dict(cost_model.op_to_dict(_op(_arg(4, resident=False))))
    assert back.args[0].replication == 4


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
    # A resident boundary operand keeps its one clone-in load, which is an aggregate
    # transfer and stays with the ordinary bytes, not the per-core term.
    a = _arg(8, resident=True, boundary=True)
    assert (a.replicated_hbm_elems(), a.hbm_elems()) == (0, ELEMS)
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
    assert (
        sympy.simplify(
            ns - BYTES * (1 - is_lx) / (s_m * p.mm_replicated_read_gbps_per_core)
        )
        == 0
    )
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
