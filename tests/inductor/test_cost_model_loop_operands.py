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


def test_cpsat_tabulates_a_gated_burst_price_over_the_op_divisions():
    """A symbolic burst price is one ResidencyGatedPrice node, which CP-SAT
    lowers to a table over the op's candidate divisions: no branch literals,
    and the objective equals the concrete price for every division and
    residency."""
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    from torch_spyre._inductor.cost_model import ResidencyGatedPrice
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
        _CoreDivisionBufferWithCpVars,
        _SympyExprToCpSat,
    )
    from torch_spyre._inductor.scratchpad.plan_solver import (
        CoreDivision,
        CoreDivisionBuffer,
    )

    d0, d1 = sympy.symbols("d0 d1")
    shapes = ((1, 1), (2, 1), (8, 1), (1, 8), (8, 4), (32, 1))
    buffer = CoreDivisionBuffer(
        "buf1",
        ELEMS,
        [0, 1],
        core_divisions=[CoreDivision(splits={d0: a, d1: b}) for a, b in shapes],
    )
    model = cp_model.CpModel()
    wrapper = _CoreDivisionBufferWithCpVars(
        buffer=buffer, model=model, capacity_units=ELEMS
    )
    split = buffer.sym_core_divs
    resident = sympy.Symbol("is_lx_buf0", integer=True, nonnegative=True)
    p = CostParams()
    op = dataclasses.replace(
        _streamed(64 * p.transport_dma_word_bytes / split[d0], resident=resident),
        cores=split[d0] * split[d1],
    )
    expr = sympy.sympify(TRIPS * _read_burst_excess_ns([op], p))
    assert expr.atoms(ResidencyGatedPrice)

    is_lx = model.new_bool_var(resident.name)
    sym_map = {resident.name: is_lx}
    buffer_map = {}
    for key, symbol in split.items():
        sym_map[symbol.name] = wrapper.cp_core_divs[key]
        buffer_map[symbol.name] = (wrapper, wrapper.cp_core_divs_raw[key])
    before = len(model.proto.variables)
    model.minimize(_SympyExprToCpSat(model, sym_map, buffer_map).convert(expr))
    added = [v.name for v in model.proto.variables][before:]
    assert not [n for n in added if n.startswith(("cond_", "piecewise_", "_product"))]

    solver = cp_model.CpSolver()
    for index, (a, b) in enumerate(shapes):
        for lx in (0, 1):
            fixed = model.clone()
            fixed.add(wrapper.division == index)
            fixed.add(is_lx == lx)
            expected = float(
                expr.xreplace({split[d0]: a, split[d1]: b, resident: sympy.Integer(lx)})
            )
            assert solver.solve(fixed) == cp_model.OPTIMAL
            assert solver.objective_value == pytest.approx(expected, abs=1e-6)
            if lx:
                assert expected == 0
    # The table is not flat: the 32-way split's one-stick run pays requests.
    assert float(expr.xreplace({split[d0]: 32, split[d1]: 1, resident: 0})) > 0


def _pointwise(cores, *, resident=False, out_resident=False, elems=ELEMS * 256):
    arg = ArgTraffic(name="buf0", role="input", is_lx=resident, elems=elems)
    out = ArgTraffic(name="buf1", role="output", is_lx=out_resident, elems=elems)
    return OpFeatures(
        name="mul",
        is_reduction=False,
        out_elems=elems,
        cores=cores,
        dtype_bytes=2,
        args=[out, arg],
    )


def test_a_pointwise_op_on_few_cores_pays_its_per_core_rate():
    from torch_spyre._inductor.cost_model import _pointwise_core_excess_ns

    p = CostParams()
    nbytes = ELEMS * 256 * 2
    one = _pointwise_core_excess_ns([_pointwise(1)], p)
    expected = 2 * nbytes * (1 / p.pointwise_gbps_per_core - 1 / p.bw_peak_gbps)
    assert one == pytest.approx(expected)
    # Enough cores to reach the bus peak, or LX-resident traffic, pay nothing.
    assert _pointwise_core_excess_ns([_pointwise(8)], p) == 0
    assert (
        _pointwise_core_excess_ns([_pointwise(1, resident=True, out_resident=True)], p)
        == 0
    )
    costs = [_pointwise_core_excess_ns([_pointwise(c)], p) for c in (1, 2, 4, 8, 32)]
    assert costs == sorted(costs, reverse=True)
    assert "low-core pointwise traffic" in explain([_pointwise(1)], p)
    off = dataclasses.replace(p, pointwise_gbps_per_core=0.0)
    assert _pointwise_core_excess_ns([_pointwise(1)], off) == 0


def test_a_one_input_arithmetic_op_keeps_the_low_core_price():
    """A unary arithmetic op with proven read geometry is priced by the
    transport request law (transport_compute_read), which charges short runs
    only; it still pays the per-core byte rate. A plain copy does not."""
    from torch_spyre._inductor.cost_model import (
        _pointwise_core_excess_ns,
        transport_dma_cost_available,
    )

    p = CostParams()

    def unary(compute_read):
        op = _pointwise(1)
        op.transport_read_run_bytes = 1 << 16
        op.transport_tile_elems = ELEMS * 64
        op.transport_compute_read = compute_read
        assert transport_dma_cost_available(op, p)
        return op

    plain = _pointwise_core_excess_ns([_pointwise(1)], p)
    assert plain > 0
    assert _pointwise_core_excess_ns([unary(True)], p) == pytest.approx(plain)
    assert _pointwise_core_excess_ns([unary(False)], p) == 0


def test_small_pointwise_args_keep_a_division_invariant_price():
    """Below pointwise_core_min_bytes nothing depends on the core count: a
    few-stick tensor's ns-scale excess must not decide its division (a (68,)
    round trip split across cores returned wrong values)."""
    from torch_spyre._inductor.cost_model import _pointwise_core_excess_ns

    p = CostParams()
    small = p.pointwise_core_min_bytes // 2 // 2 - 1
    assert {
        _pointwise_core_excess_ns([_pointwise(c, elems=small)], p) for c in (1, 2, 32)
    } == {0}
    assert {predict_ops([_pointwise(c, elems=68)], p) for c in (1, 2, 4, 32)} == {
        predict_ops([_pointwise(1, elems=68)], p)
    }


def test_low_core_pointwise_price_is_symbolic_in_the_split():
    from torch_spyre._inductor.cost_model import (
        ResidencyGatedPrice,
        _pointwise_core_excess_ns,
    )

    split = sympy.Symbol("split_buf1_d0", integer=True, positive=True)
    resident = sympy.Symbol("is_lx_buf0", integer=True, nonnegative=True)
    p = CostParams()
    expr = sympy.sympify(
        _pointwise_core_excess_ns([_pointwise(split, resident=resident)], p)
    )
    assert expr.atoms(ResidencyGatedPrice)
    for cores in (1, 2, 4, 32):
        for lx in (0, 1):
            concrete = _pointwise_core_excess_ns(
                [_pointwise(cores, resident=bool(lx))], p
            )
            actual = float(expr.xreplace({split: cores, resident: sympy.Integer(lx)}))
            assert actual == pytest.approx(float(concrete))


def test_uncalibrated_core_counts_take_the_next_lower_burst_rate():
    """A 12- or 24-core division is priced like 8 or 16 cores, not for free."""
    p = CostParams()
    stick = p.transport_dma_word_bytes

    def at(cores):
        return _read_burst_excess_ns(
            [dataclasses.replace(_streamed(stick), cores=cores)], p
        )

    assert at(12) == at(8) > 0
    assert at(24) == at(16) > 0
    assert at(3) == at(2) > 0
    split = sympy.Symbol("split_buf1_d0", integer=True, positive=True)
    expr = sympy.sympify(
        _read_burst_excess_ns([dataclasses.replace(_streamed(stick), cores=split)], p)
    )
    for cores in (3, 6, 12, 24, 32):
        assert float(expr.xreplace({split: cores})) == pytest.approx(float(at(cores)))


def _decode_pv(cores, replication, *, reuse=2, loop_factor=1, run=256):
    """A decode GQA P@V: the value cache, reused by ``reuse`` query rows, read
    as ``run``-byte rows of one head (a cache-position-first cache)."""
    v = ArgTraffic(
        name="arg10_1",
        role="input",
        is_lx=False,
        elems=ELEMS * 128,
        broadcast=True,
        replication=replication,
        loop_factor=loop_factor,
        batch_run_bytes=run,
    )
    out = ArgTraffic(name="buf42", role="output", is_lx=True, elems=ELEMS)
    return OpFeatures(
        name="bmm",
        is_reduction=False,
        out_elems=ELEMS,
        cores=cores,
        dtype_bytes=2,
        args=[out, v],
        is_matmul=True,
        matmul_macs=reuse * ELEMS * 128,
    )


def _stream_ns(p, per_core, run, ns_per_request):
    return per_core / p.mm_stream_gbps_per_core + per_core / run * ns_per_request


def test_few_cores_stream_a_reused_matmul_operand_at_their_own_rate():
    from torch_spyre._inductor.cost_model import _reused_operand_stream_excess_ns

    p = CostParams()
    nbytes = ELEMS * 128 * 2
    peak = nbytes / p.bw_peak_gbps
    unicast, multicast = (
        p.mm_stream_unicast_ns_per_request,
        p.mm_stream_multicast_ns_per_request,
    )
    one = _reused_operand_stream_excess_ns([_decode_pv(1, 1)], p)
    assert one == pytest.approx(_stream_ns(p, nbytes, 256, unicast) - peak)
    # A split of the query rows multicasts each core's slice: every request
    # pays the broadcast's longer turnaround.
    shared = _reused_operand_stream_excess_ns([_decode_pv(2, 2)], p)
    assert shared == pytest.approx(_stream_ns(p, nbytes, 256, multicast) - peak)
    assert shared > 2 * one
    # Enough cores, a resident operand or a looped read cost nothing extra.
    assert _reused_operand_stream_excess_ns([_decode_pv(32, 1)], p) == 0
    resident = _decode_pv(1, 1)
    resident.args[1].is_lx = True
    assert _reused_operand_stream_excess_ns([resident], p) == 0
    assert _reused_operand_stream_excess_ns([_decode_pv(1, 1, loop_factor=4)], p) == 0
    # Neither is an operand fed to one multiply-accumulate per element.
    assert _reused_operand_stream_excess_ns([_decode_pv(1, 1, reuse=1)], p) == 0
    assert "reused matmul operand streaming" in explain([_decode_pv(2, 2)], p)


def test_long_runs_stream_a_reused_operand_at_the_byte_rate():
    """A head-first value cache, or a GEMM weight, is read in full bursts: a
    fraction of the per-row request cost, whatever the reuse."""
    from torch_spyre._inductor.cost_model import _reused_operand_stream_excess_ns

    p = CostParams()
    burst = p.transport_dma_word_bytes * p.transport_dma_max_burst_words
    rows = _reused_operand_stream_excess_ns([_decode_pv(2, 2)], p)
    full = _reused_operand_stream_excess_ns([_decode_pv(2, 2, run=1 << 17)], p)
    nbytes = ELEMS * 128 * 2
    expected = _stream_ns(p, nbytes, burst, p.mm_stream_multicast_ns_per_request)
    assert full == pytest.approx(expected - nbytes / p.bw_peak_gbps)
    assert 0 < full < rows / 3
    # An unproven run counts as a full burst; prefill-sized reuse is priced too.
    unproven = _reused_operand_stream_excess_ns([_decode_pv(2, 2, run=None)], p)
    assert unproven == pytest.approx(full)
    prefill = _reused_operand_stream_excess_ns(
        [_decode_pv(2, 2, run=1 << 17, reuse=64)], p
    )
    assert prefill == pytest.approx(full)
    # Past decode reuse a short run is priced as full bursts: prefill
    # attention's K/V rows measured no slower for it.
    rows_prefill = _reused_operand_stream_excess_ns([_decode_pv(2, 2, reuse=64)], p)
    assert rows_prefill == pytest.approx(full)
    assert _reused_operand_stream_excess_ns([_decode_pv(2, 2, reuse=8)], p) == (
        pytest.approx(rows)
    )


def _gemm(m_split, n_split, *, rows=64, k=768, n=768):
    """x[rows, k] @ w[k, n] on an M x N split: the weight is replicated by the
    M split, the activation by the N split, both one broadcast load."""
    cores = m_split * n_split
    x = ArgTraffic(
        name="buf30",
        role="input",
        is_lx=True,
        elems=rows * k,
        broadcast=True,
        replication=n_split,
        read_run_bytes=rows * k * 2 // m_split,
    )
    w = ArgTraffic(
        name="arg5_1",
        role="input",
        is_lx=False,
        elems=k * n,
        broadcast=True,
        replication=m_split,
        read_run_bytes=k * n * 2 // n_split,
        is_boundary=True,
    )
    out = ArgTraffic(name="buf14", role="output", is_lx=True, elems=rows * n)
    return OpFeatures(
        name="mm",
        is_reduction=False,
        out_elems=rows * n,
        cores=cores,
        dtype_bytes=2,
        args=[out, x, w],
        is_matmul=True,
        matmul_macs=rows * k * n,
        matmul_m_split=m_split,
        matmul_n_split=n_split,
    )


def test_an_m_only_split_pays_for_every_core_streaming_the_whole_weight():
    """granite-embedding-125m's attention output projection, [64, 768] @
    [768, 768]: an 8-way M split measured 25.0 us, the 8 x 4 split 12.6 us.
    Each M-split core streams the whole 1.2 MB weight through the multicast;
    the 32-core split streams a quarter of it, under the bus charge."""
    from torch_spyre._inductor.cost_model import _reused_operand_stream_excess_ns

    p = CostParams()
    m_only = _reused_operand_stream_excess_ns([_gemm(8, 1)], p)
    both = _reused_operand_stream_excess_ns([_gemm(8, 4)], p)
    assert m_only == pytest.approx(
        _stream_ns(p, 768 * 768 * 2, 4096, p.mm_stream_multicast_ns_per_request)
        - 768 * 768 * 2 / p.bw_peak_gbps
    )
    assert m_only > 10_000
    assert both == 0
    assert predict_ops([_gemm(8, 1)], p) > predict_ops([_gemm(8, 4)], p) + 10_000
    # Past the cohort limit the multicast penalty already charges part of it.
    wide = _reused_operand_stream_excess_ns([_gemm(16, 1)], p)
    assert 0 < wide < m_only


def test_reused_operand_stream_price_is_symbolic_in_the_split():
    from torch_spyre._inductor.cost_model import (
        ResidencyGatedPrice,
        _reused_operand_stream_excess_ns,
    )

    heads, rows = sympy.symbols("split_buf42_d0 split_buf42_d1", integer=True)
    p = CostParams()
    for run in (256, 1 << 17):
        expr = sympy.sympify(
            _reused_operand_stream_excess_ns(
                [_decode_pv(heads * rows, rows, run=run)], p
            )
        )
        assert expr.atoms(ResidencyGatedPrice)
        for h, r in ((1, 1), (1, 2), (4, 1), (4, 2), (8, 2), (1, 16)):
            concrete = _reused_operand_stream_excess_ns(
                [_decode_pv(h * r, r, run=run)], p
            )
            actual = float(expr.xreplace({heads: h, rows: r}))
            assert actual == pytest.approx(float(concrete))


def test_the_stream_price_subtracts_the_burst_excess_it_overlaps():
    """A head split shortens the read run, which the short-burst term already
    charges; the stream term adds only what is left beyond it."""
    from torch_spyre._inductor.cost_model import _reused_operand_stream_excess_ns

    p = CostParams()
    op = _decode_pv(1, 1)
    op.args[1].read_run_bytes = 256
    bursts = _read_burst_excess_ns([op], p)
    assert bursts > 0
    alone = _reused_operand_stream_excess_ns([_decode_pv(1, 1)], p)
    assert _reused_operand_stream_excess_ns([op], p) == pytest.approx(
        max(0.0, alone - bursts)
    )


def test_cpsat_tabulates_the_reused_stream_price_over_the_op_divisions():
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    from torch_spyre._inductor.cost_model import (
        ResidencyGatedPrice,
        _reused_operand_stream_excess_ns,
    )
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import (
        _CoreDivisionBufferWithCpVars,
        _SympyExprToCpSat,
    )
    from torch_spyre._inductor.scratchpad.plan_solver import (
        CoreDivision,
        CoreDivisionBuffer,
    )

    d0, d1 = sympy.symbols("d0 d1")
    shapes = ((1, 1), (2, 1), (8, 1), (1, 8), (8, 4), (32, 1))
    buffer = CoreDivisionBuffer(
        "buf42",
        ELEMS,
        [0, 1],
        core_divisions=[CoreDivision(splits={d0: a, d1: b}) for a, b in shapes],
    )
    model = cp_model.CpModel()
    wrapper = _CoreDivisionBufferWithCpVars(
        buffer=buffer, model=model, capacity_units=ELEMS
    )
    split = buffer.sym_core_divs
    resident = sympy.Symbol("is_lx_arg10_1", integer=True, nonnegative=True)
    p = CostParams()
    op = _decode_pv(split[d0] * split[d1], split[d0])
    op.args[1].is_lx = resident
    op.args[1].read_run_bytes = 2048 / split[d1]
    expr = sympy.sympify(_reused_operand_stream_excess_ns([op], p))
    assert expr.atoms(ResidencyGatedPrice)

    is_lx = model.new_bool_var(resident.name)
    sym_map = {resident.name: is_lx}
    buffer_map = {}
    for key, symbol in split.items():
        sym_map[symbol.name] = wrapper.cp_core_divs[key]
        buffer_map[symbol.name] = (wrapper, wrapper.cp_core_divs_raw[key])
    before = len(model.proto.variables)
    model.minimize(_SympyExprToCpSat(model, sym_map, buffer_map).convert(expr))
    added = [v.name for v in model.proto.variables][before:]
    assert not [n for n in added if n.startswith(("cond_", "piecewise_", "_product"))]

    solver = cp_model.CpSolver()
    for index, (a, b) in enumerate(shapes):
        for lx in (0, 1):
            fixed = model.clone()
            fixed.add(wrapper.division == index)
            fixed.add(is_lx == lx)
            expected = float(
                expr.xreplace({split[d0]: a, split[d1]: b, resident: sympy.Integer(lx)})
            )
            assert solver.solve(fixed) == cp_model.OPTIMAL
            assert solver.objective_value == pytest.approx(expected, abs=1e-3)
    assert float(expr.xreplace({split[d0]: 2, split[d1]: 1, resident: 0})) > 0


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


def test_a_batched_operand_runs_one_batch_element_at_a_time():
    """A cache-position-first value cache [S, H, D] is one contiguous run over
    every head, but a batched matmul loads one head's [S, D] block at a time:
    rows of D elements."""
    from types import SimpleNamespace

    from torch_spyre._inductor import dump_cost_model as dcm

    b, h, m, s, d = sympy.symbols("b h m s d", integer=True, nonnegative=True)
    coords = [s, h, sympy.floor(d / 64), b, sympy.Mod(d, 64)]
    dims = [512, 8, 2, 1, 64]
    space = {b: 1, h: 8, m: 2, s: 512, d: 128}

    def run(slices):
        return dcm._contiguous_device_run(
            coords, dims, space, slices, stick_planes=True
        )

    assert run({}) == 512 * 8 * 128
    assert run({h: 8}) == 128

    def dep(index):
        return SimpleNamespace(index=index)

    write = dep(h * 256 + m * 128 + d)
    p_read, v_read = dep(h * 1024 + m * 512 + s), dep(s * 1024 + h * 128 + d)
    op = SimpleNamespace()
    bmm = SimpleNamespace(reads=[p_read, v_read], writes=[write])
    assert dcm._batch_symbols(op, bmm, space) == {h}
    # A plain matmul, or a 3d-2d projection whose weight has no batch dim.
    mm = SimpleNamespace(reads=[dep(m * 512 + s), dep(s * 128 + d)], writes=[write])
    assert dcm._batch_symbols(op, mm, space) == set()


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
