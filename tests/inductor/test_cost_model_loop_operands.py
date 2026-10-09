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

"""Pricing of operands a loop reads again on every iteration.

Forced core divisions of an SDPA K/V scan (B2/B4 x H16, D64) ranked by measured
time against the co-optimizer's objective agreed on 17 of 48 pairs. Two terms
were inverted: a K/V block replicated by a query split was charged per core,
although those splits measured fastest, and an online-softmax carry left in HBM
was charged at the peak rate, although keeping it resident halved the time.
"""

import dataclasses

import pytest
import sympy
import torch
import torch.nn.functional as F

import torch_spyre  # noqa: F401
import torch_spyre._inductor.decompositions as decompositions
from torch_spyre._inductor import config
from torch_spyre._inductor import cost_model_pass as cmp
from torch_spyre._inductor.cost_model import (
    ArgTraffic,
    CostParams,
    OpFeatures,
    _loop_repeated_read_excess_ns,
    _read_burst_excess_ns,
    explain,
    predict_ops,
)

ELEMS = 4096
TRIPS = 8


def _carry_reader(*, resident, broadcast=False, loop_factor=TRIPS, matmul=False):
    carry = ArgTraffic(
        name="buf16",
        role="input",
        is_lx=resident,
        elems=ELEMS,
        broadcast=broadcast,
        loop_factor=loop_factor,
    )
    out = ArgTraffic(
        name="buf21", role="output", is_lx=True, elems=ELEMS, loop_factor=TRIPS
    )
    return OpFeatures(
        name="mul",
        is_reduction=False,
        out_elems=ELEMS,
        cores=32,
        dtype_bytes=2,
        args=[out, carry],
        is_matmul=matmul,
        loop_trip=TRIPS,
    )


def test_an_hbm_carry_pays_the_repeated_pass_rate():
    p = CostParams()
    per_byte = 1 / p.loop_reread_gbps - 1 / p.bw_peak_gbps
    excess = _loop_repeated_read_excess_ns([_carry_reader(resident=False)], p)
    assert excess == ELEMS * (TRIPS - 1) * 2 * per_byte


def test_resident_broadcast_single_pass_and_matmul_operands_pay_nothing():
    p = CostParams()
    for op in (
        _carry_reader(resident=True),
        _carry_reader(resident=False, broadcast=True),
        _carry_reader(resident=False, loop_factor=1),
        _carry_reader(resident=False, matmul=True),
    ):
        assert _loop_repeated_read_excess_ns([op], p) == 0


def test_the_excess_is_linear_in_symbolic_residency_and_can_be_disabled():
    is_lx = sympy.Symbol("is_lx_buf16", integer=True)
    p = CostParams()
    excess = _loop_repeated_read_excess_ns([_carry_reader(resident=is_lx)], p)
    spilled = _loop_repeated_read_excess_ns([_carry_reader(resident=False)], p)
    assert sympy.simplify(excess - spilled * (1 - is_lx)) == 0
    off = dataclasses.replace(p, loop_reread_gbps=0.0)
    assert _loop_repeated_read_excess_ns([_carry_reader(resident=False)], off) == 0


def test_predict_and_explain_include_the_excess():
    p = CostParams()
    spilled, resident = _carry_reader(resident=False), _carry_reader(resident=True)
    gap = predict_ops([spilled], p) - predict_ops([resident], p)
    assert gap >= _loop_repeated_read_excess_ns([spilled], p)
    assert "loop-repeated reads" in explain([spilled], p)


def test_extractor_shares_a_loop_repeated_replicated_matmul_operand(monkeypatch):
    """A multi-block SDPA scan: the QK^T and P@V bmms run once per K/V block, and
    a split of their query rows replicates the K/V operand. Every such operand is
    stamped as one shared load; replicated operands outside a loop keep their
    per-core price (``test_cost_model_replication``)."""
    captured: dict = {}
    real = cmp.extract_op_features

    def spy(op):
        feats = real(op)
        captured[op.get_name()] = feats
        return feats

    monkeypatch.setattr(cmp, "extract_op_features", spy)
    select = decompositions._select_sdpa_tiling

    def four_blocks(**kwargs):
        c = select(**kwargs)
        return dataclasses.replace(
            c,
            strategy="work_divided_tiled",
            num_batch_tiles=1,
            num_head_tiles=1,
            num_group_tiles=1,
            num_q_tiles=1,
            q_tile_size=kwargs["max_seqlen_q"],
            num_kv_blocks=4,
            kv_block_size=kwargs["max_seqlen_kv"] // 4,
            kv_blocks_per_loop_group=4,
        )

    monkeypatch.setattr(decompositions, "_select_sdpa_tiling", four_blocks)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 256, 64, dtype=torch.float16) for _ in range(3))
    torch._inductor.codecache.FxGraphCache.clear()
    torch._dynamo.reset()
    with config.patch({"lx_planning": False, "cost_model": "1"}):
        out = torch.compile(F.scaled_dot_product_attention, dynamic=False)(
            q.to("spyre"), k.to("spyre"), v.to("spyre")
        )
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    assert torch.allclose(out.cpu().float(), ref, rtol=2e-2, atol=2e-2)

    looped = [
        a
        for f in captured.values()
        if f.is_matmul
        for a in f.args
        if a.role == "input" and a.loop_factor > 1 and a.replication != 1
    ]
    assert looped, {n: f.is_matmul for n, f in captured.items()}
    assert all(a.broadcast for a in looped), [(a.name, a.replication) for a in looped]


# ------------------------------------------------------------ burst pricing


def _streamed(run_bytes, *, resident=False):
    arg = ArgTraffic(
        name="buf0",
        role="input",
        is_lx=resident,
        elems=ELEMS * 64,
        read_run_bytes=run_bytes,
    )
    out = ArgTraffic(name="buf1", role="output", is_lx=True, elems=ELEMS * 64)
    return OpFeatures(
        name="add",
        is_reduction=False,
        out_elems=ELEMS * 64,
        cores=32,
        dtype_bytes=2,
        args=[out, arg],
    )


def test_short_bursts_cost_requests_and_32_stick_bursts_do_not():
    p = CostParams()
    stick = p.transport_dma_word_bytes
    full = _read_burst_excess_ns([_streamed(32 * stick)], p)
    one = _read_burst_excess_ns([_streamed(stick)], p)
    assert full == 0
    assert one > 0
    # A longer run never costs more requests.
    costs = [_read_burst_excess_ns([_streamed(n * stick)], p) for n in (1, 2, 4, 8, 32)]
    assert costs == sorted(costs, reverse=True)


def test_resident_or_unproven_reads_pay_no_burst_excess():
    p = CostParams()
    stick = p.transport_dma_word_bytes
    assert _read_burst_excess_ns([_streamed(stick, resident=True)], p) == 0
    assert _read_burst_excess_ns([_streamed(None)], p) == 0


def test_burst_excess_follows_a_symbolic_split():
    split = sympy.Symbol("split_buf0_d0", integer=True, positive=True)
    p = CostParams()
    stick = p.transport_dma_word_bytes
    expr = _read_burst_excess_ns([_streamed(64 * stick / split)], p)
    assert float(expr.subs(split, 32)) > float(expr.subs(split, 1))


@pytest.mark.parametrize("broadcast", [False, True])
def test_burst_price_agrees_before_and_after_replication_is_chosen(broadcast):
    replication, split = sympy.symbols(
        "split_bmm_m split_bmm_n", integer=True, positive=True
    )
    resident = sympy.Symbol("is_lx_buf0", integer=True)
    p = CostParams()
    op = _streamed(8 * p.transport_dma_word_bytes / split, resident=resident)
    arg = dataclasses.replace(
        op.args[1], replication=replication, broadcast=broadcast, loop_factor=TRIPS
    )
    op = dataclasses.replace(op, args=[op.args[0], arg])
    expr = sympy.sympify(_read_burst_excess_ns([op], p))
    for replicas in (1, 2, 8):
        for divisor in (1, 8, 16):
            for is_lx in (False, True):
                concrete_arg = dataclasses.replace(
                    arg,
                    replication=replicas,
                    read_run_bytes=8 * p.transport_dma_word_bytes / divisor,
                    is_lx=is_lx,
                )
                concrete = dataclasses.replace(op, args=[op.args[0], concrete_arg])
                expected = float(_read_burst_excess_ns([concrete], p))
                actual = float(
                    expr.subs(
                        {replication: replicas, split: divisor, resident: int(is_lx)}
                    )
                )
                assert actual == pytest.approx(expected)
                if is_lx or (not broadcast and replicas > 1):
                    assert expected == 0
                elif divisor >= 8:
                    assert expected > 0


def test_cpsat_keeps_the_burst_price_when_replication_resolves_to_one():
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import _SympyExprToCpSat

    replication, split = sympy.symbols(
        "split_bmm_m split_bmm_n", integer=True, positive=True
    )
    resident = sympy.Symbol("is_lx_buf0", integer=True)
    p = CostParams()
    op = _streamed(8 * p.transport_dma_word_bytes / split, resident=resident)
    arg = dataclasses.replace(op.args[1], replication=replication)
    op = dataclasses.replace(op, args=[op.args[0], arg])
    expr = sympy.sympify(_read_burst_excess_ns([op], p))
    model = cp_model.CpModel()
    variables = {
        replication.name: model.new_int_var(1, 2, replication.name),
        split.name: model.new_int_var(1, 8, split.name),
        resident.name: model.new_bool_var(resident.name),
    }
    model.minimize(_SympyExprToCpSat(model, variables, {}).convert(expr))
    solver = cp_model.CpSolver()
    for replicas, divisor, is_lx in ((1, 1, 0), (1, 8, 0), (2, 8, 0), (1, 8, 1)):
        fixed = model.clone()
        for symbol, value in (
            (replication, replicas),
            (split, divisor),
            (resident, is_lx),
        ):
            fixed.add(variables[symbol.name] == value)
        concrete_arg = dataclasses.replace(
            arg,
            replication=replicas,
            read_run_bytes=8 * p.transport_dma_word_bytes / divisor,
            is_lx=bool(is_lx),
        )
        concrete = dataclasses.replace(op, args=[op.args[0], concrete_arg])
        expected = float(_read_burst_excess_ns([concrete], p))
        assert solver.solve(fixed) == cp_model.OPTIMAL
        assert solver.objective_value == pytest.approx(expected, abs=1)


# ------------------------------------------------- stick-plane run geometry


def test_stick_plane_geometry_measures_the_source_burst():
    from torch_spyre._inductor import dump_cost_model as dcm

    b, h, s, d = sympy.symbols("b h s d", integer=True, nonnegative=True)
    # Granite's interleaved query [B, S, H, D] as the device stores it:
    # [H, S, D/64 planes, B, 64 lanes].
    coords = [h, s, sympy.floor(d / 64), b, sympy.Mod(d, 64)]
    dims = [32, 512, 2, 2, 64]
    space = {b: 2, h: 32, s: 512, d: 128}

    def run(slices):
        return dcm._contiguous_device_run(
            coords, dims, space, slices, stick_planes=True
        )

    assert run({}) == 2 * 32 * 512 * 128
    assert run({h: 16}) == 2 * 2 * 512 * 128
    # B sits inside the stick plane, so splitting it leaves one-stick bursts.
    assert run({b: 2}) == 64
    # The transport term keeps its default: no stick-plane walk.
    assert dcm._contiguous_device_run(coords, dims, space, {}) is None


# ---------------------------------------------------- batched matmul splits


def _bmm(monkeypatch, batch, batch_split, m_extent, m_split, loop_trip=8):
    from torch_spyre._inductor import cost_model as cm

    op = OpFeatures(
        name="bmm",
        is_reduction=True,
        out_elems=ELEMS,
        cores=32,
        dtype_bytes=2,
        args=[],
        is_matmul=True,
        loop_trip=loop_trip,
    )
    monkeypatch.setattr(
        cm,
        "_matmul_axes_for_split_cost",
        lambda o: (
            (batch, batch_split),
            (m_extent, m_split),
            (512, 1),
            (128, 1),
            True,
        ),
    )
    return cm, op


def test_batch_split_charges_the_m_split_it_gives_up(monkeypatch):
    p = CostParams()
    rate = p.mm_batch_split_ns_per_step
    # 1024 rows allow an 8-way M split at 128 rows per core; batch 4 x M 2
    # forgoes two steps of M.
    cm, op = _bmm(monkeypatch, 16, 4, 1024, 2)
    assert cm._matmul_batch_split_ns([op], p) == rate * 8 * 2
    # All 8 useful M ways taken: the batch split gives nothing up.
    cm, op = _bmm(monkeypatch, 16, 4, 1024, 8)
    assert cm._matmul_batch_split_ns([op], p) == 0
    # No batch split, nothing to charge however little M is split.
    cm, op = _bmm(monkeypatch, 16, 1, 1024, 1)
    assert cm._matmul_batch_split_ns([op], p) == 0


def test_short_m_caps_the_charge_at_the_row_bound(monkeypatch):
    p = CostParams()
    # 256 rows: only a 2-way M split keeps 128 rows, so an 8-way batch split
    # forgoes one step, not three.
    cm, op = _bmm(monkeypatch, 16, 8, 256, 1)
    assert cm._matmul_batch_split_ns([op], p) == p.mm_batch_split_ns_per_step * 8
    # Lq 128: no M split is worth taking, so a head split costs nothing.
    cm, op = _bmm(monkeypatch, 16, 8, 128, 1)
    assert cm._matmul_batch_split_ns([op], p) == 0


def test_unbatched_or_disabled_batch_split_costs_nothing(monkeypatch):
    cm, op = _bmm(monkeypatch, 1, 1, 1024, 1)
    assert cm._matmul_batch_split_ns([op], CostParams()) == 0
    cm, op = _bmm(monkeypatch, 16, 8, 1024, 1)
    off = dataclasses.replace(CostParams(), mm_batch_split_ns_per_step=0.0)
    assert cm._matmul_batch_split_ns([op], off) == 0


def test_batch_split_cost_scales_with_a_prospective_tile_count(monkeypatch):
    # The co-optimizer prices a tileable op with its undecided tile count.
    tiles = sympy.Symbol("tiles", integer=True, positive=True)
    p = CostParams()
    cm, op = _bmm(monkeypatch, 16, 4, 1024, 2, loop_trip=tiles)
    assert (
        cm._matmul_batch_split_ns([op], p) == p.mm_batch_split_ns_per_step * tiles * 2
    )


def test_batch_split_cost_follows_symbolic_splits(monkeypatch):
    s0, s1, sm = sympy.symbols(
        "split_bmm_d0 split_bmm_d1 split_bmm_d2", integer=True, positive=True
    )
    cm, op = _bmm(monkeypatch, 16, s0 * s1, 1024, sm)
    expr = cm._matmul_batch_split_ns([op], CostParams())
    rate = CostParams().mm_batch_split_ns_per_step

    def at(b0, b1, m):
        return float(expr.subs({s0: b0, s1: b1, sm: m}))

    assert at(1, 1, 32) == pytest.approx(0, abs=1e-6)
    assert at(4, 2, 2) == pytest.approx(rate * 8 * 2)
    assert at(2, 1, 8) == pytest.approx(0, abs=1e-6)


def test_batch_split_latency_changes_the_cost_and_is_explained():
    # Equal work on 32 cores: 16 batch x 2 M versus 4 batch x 8 M. Keep HBM
    # traffic out of this example so the compute-side correction is visible.
    def bmm(m_split):
        return OpFeatures(
            name="bmm",
            is_reduction=True,
            is_matmul=True,
            out_elems=16 * 1024 * 128,
            cores=32,
            dtype_bytes=2,
            args=[],
            matmul_macs=8 * 16 * 1024 * 128 * 512,
            matmul_rows_per_core=1024 / m_split,
            matmul_cols_per_core=128,
            matmul_a_bytes=1024 * 512 * 2,
            matmul_b_bytes=512 * 128 * 2,
            matmul_m_split=m_split,
            loop_trip=8,
        )

    p = CostParams(use_bundled_cost_model=False)
    disabled = dataclasses.replace(p, mm_batch_split_ns_per_step=0)
    batch, rows = bmm(2), bmm(8)
    assert predict_ops([batch], disabled) == pytest.approx(
        predict_ops([rows], disabled)
    )
    assert predict_ops([batch], p) > predict_ops([rows], p)
    assert "batched-matmul batch splits: +64.00 us" in explain([batch], p)
    assert "batched-matmul batch splits" not in explain([rows], p)
    assert "batched-matmul batch splits" not in explain([batch], disabled)
    assert "batched-matmul batch splits" not in explain(
        [batch], dataclasses.replace(p, use_bundled_cost_model=True)
    )


def test_co_optimizer_prices_batch_splits_of_an_sdpa_scan(monkeypatch, tmp_path):
    """The co-optimizing solve lowers the log2 batch-split term and still
    produces a correct four-block scan."""
    import json

    select = decompositions._select_sdpa_tiling

    def four_blocks(**kwargs):
        c = select(**kwargs)
        return dataclasses.replace(
            c,
            strategy="work_divided_tiled",
            num_batch_tiles=1,
            num_head_tiles=1,
            num_group_tiles=1,
            num_q_tiles=1,
            q_tile_size=kwargs["max_seqlen_q"],
            num_kv_blocks=4,
            kv_block_size=kwargs["max_seqlen_kv"] // 4,
            kv_blocks_per_loop_group=4,
        )

    monkeypatch.setattr(decompositions, "_select_sdpa_tiling", four_blocks)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 256, 64, dtype=torch.float16) for _ in range(3))
    dump = tmp_path / "cost.jsonl"
    torch._inductor.codecache.FxGraphCache.clear()
    torch._dynamo.reset()
    with config.patch({"dump_cost_expr_file": str(dump)}):
        out = torch.compile(F.scaled_dot_product_attention, dynamic=False)(
            q.to("spyre"), k.to("spyre"), v.to("spyre")
        )
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    assert torch.allclose(out.cpu().float(), ref, rtol=2e-2, atol=2e-2)
    records = [json.loads(line) for line in dump.read_text().splitlines()]
    assert records and all(r["solve"]["status"] == "OPTIMAL" for r in records)
    assert any("log" in b["expr"] for r in records for b in r["bundles"])
