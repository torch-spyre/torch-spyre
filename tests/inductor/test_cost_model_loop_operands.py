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


# ------------------------------------------------------- CP-SAT tie-break


def test_tie_break_prefers_m_splits_that_keep_64_rows_per_core(monkeypatch):
    from torch_spyre._inductor.scratchpad import allocator
    from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision

    m, n = sympy.symbols("m n", integer=True, nonnegative=True)
    monkeypatch.setattr(allocator, "_is_matmul_op", lambda op: True)
    for rows, splits, expected in (
        # 1024 rows: M splits count up to 16, n splits never do.
        (1024, ({}, {m: 4}, {m: 16}, {m: 32}, {n: 32}), [0, 2, 4, 4, 0]),
        # 128 rows: past 2 ways a core would hold fewer than 64 rows.
        (128, ({}, {m: 2}, {m: 8}, {m: 32}), [0, 1, 1, 1]),
    ):
        monkeypatch.setattr(
            allocator,
            "_matmul_axis_parse",
            lambda op, rows=rows: {"M": (m, rows, 1), "N": (n, 512, 1)},
        )
        divisions = [CoreDivision(splits=dict(s)) for s in splits]
        assert allocator._division_tie_break_scores(object(), divisions) == expected
    monkeypatch.setattr(allocator, "_is_matmul_op", lambda op: False)
    assert allocator._division_tie_break_scores(object(), divisions) == []


def test_tie_break_keeps_the_optimal_cost(monkeypatch, tmp_path):
    """The re-solve may only choose among plans at the solved cost. Holding
    the cost as a constraint let linearized Max/product terms go slack and the
    true cost rise; keeping it in the objective must reproduce the optimum."""
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
    objectives = {}
    for tie_break in (False, True):
        dump = tmp_path / f"cost_{tie_break}.jsonl"
        torch._inductor.codecache.FxGraphCache.clear()
        torch._dynamo.reset()
        with config.patch(
            {"cpsat_division_tie_break": tie_break, "dump_cost_expr_file": str(dump)}
        ):
            out = torch.compile(F.scaled_dot_product_attention, dynamic=False)(
                q.to("spyre"), k.to("spyre"), v.to("spyre")
            )
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        assert torch.allclose(out.cpu().float(), ref, rtol=2e-2, atol=2e-2)
        records = [json.loads(line) for line in dump.read_text().splitlines()]
        assert records, "the co-optimizer did not solve a priced plan"
        objectives[tie_break] = [round(r["objective_ns"]) for r in records]
    assert objectives[True] == objectives[False]
