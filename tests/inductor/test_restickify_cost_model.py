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

"""Physical source runs, DMA request costs and solver decisions for DL16 swaps."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.virtualized import V
from torch_spyre._C import DataFormats, SpyreTensorLayout
from torch_spyre._inductor import cost_model as cm
from torch_spyre._inductor import dump_cost_model as dcm


B, X, N = sympy.symbols("b x n", integer=True, nonnegative=True)


def run_bytes(sb, sx, *, b=8, x=128, n=1024):
    return 2 * dcm._contiguous_device_run(
        [N, B, sympy.floor(X / 64), sympy.Mod(X, 64)],
        [33280, b, x // 64, 64],
        {B: b, X: x, N: n},
        {B: sb, X: sx, N: 1},
    )


def restickify(sb=8, sx=1, *, b=8, x=128, n=1024, trips=1):
    elems = b * x * n
    return cm.OpFeatures(
        name="transpose",
        is_reduction=False,
        out_elems=elems,
        cores=sb * sx,
        dtype_bytes=2,
        loop_trip=trips,
        hbm_pattern="restickify",
        transport_read_run_bytes=run_bytes(sb, sx, b=b, x=x, n=n),
        transport_tile_elems=elems,
        args=[
            cm.ArgTraffic("input", "input", False, elems, loop_factor=trips),
            cm.ArgTraffic("output", "output", False, elems, loop_factor=trips),
        ],
    )


@pytest.mark.parametrize(
    "sb,sx,expected",
    [(8, 1, 256), (8, 2, 128), (4, 1, 512), (4, 2, 128), (2, 1, 1024), (1, 1, 2097152)],
)
def test_source_run_matches_measured_dma_burst(sb, sx, expected):
    assert run_bytes(sb, sx) == expected


def test_geometry_keeps_symbolic_splits_and_physical_gaps():
    sb, sx = sympy.symbols("sb sx", integer=True, positive=True)
    expr = run_bytes(sb, sx)
    for b in (1, 2, 4, 8):
        for x in (1, 2):
            assert expr.subs({sb: b, sx: x}) == run_bytes(b, x)
    # Padding in X prevents coalescing across B, even though B is unsplit.
    assert (
        dcm._contiguous_device_run(
            [N, B, X], [1024, 8, 256], {N: 1024, B: 8, X: 128}, {}
        )
        == 128
    )
    # A genuine non-affine stick transpose cannot be called a contiguous read.
    assert (
        dcm._contiguous_device_run(
            [sympy.floor(X / 64), N, sympy.Mod(X, 64)],
            [2, 1024, 64],
            {X: 128, N: 1024},
            {},
        )
        is None
    )


@pytest.mark.parametrize(
    "sb,sx,observed_us",
    [
        (1, 1, 54.03),
        (2, 1, 39.14),
        (4, 1, 51.03),
        (8, 1, 81.21),
        (2, 2, 142.94),
        (4, 2, 142.56),
        (8, 2, 142.03),
    ],
)
def test_independent_device_measurements(sb, sx, observed_us):
    op = restickify(sb, sx)
    p = cm.CostParams()
    predicted = 2 * op.transport_tile_elems * 2 / p.bw_restickify_gbps
    predicted += cm._transport_dma_excess_ns([op], p)
    assert float(predicted / 1000) == pytest.approx(observed_us, rel=0.12)


@pytest.mark.parametrize("n,trips", [(512, 1), (512, 128), (1024, 32), (2048, 128)])
def test_tile_size_and_repetition_scale_work_not_applicability(n, trips):
    p = cm.CostParams()
    anchor = cm._transport_dma_excess_ns([restickify(8, 2)], p)
    value = cm._transport_dma_excess_ns([restickify(8, 2, n=n, trips=trips)], p)
    assert float(value) == pytest.approx(float(anchor) * n / 1024 * trips)


def test_wider_stick_axis_changes_the_ranking_without_size_lookup():
    p = cm.CostParams()
    # Same payload, wider X: 8x4 has 32 request streams, unlike the 16 of 8x2.
    op = restickify(8, 4, x=256, n=512)
    baseline = 2 * op.transport_tile_elems * 2 / p.bw_restickify_gbps
    estimate = baseline + cm._transport_dma_excess_ns([op], p)
    assert float(estimate / 1000) == pytest.approx(76.81, rel=0.12)
    assert cm.predict_ops([restickify(4, 1)]) < cm.predict_ops([restickify(8, 1)])
    assert cm.predict_ops([restickify(8, 1)]) < cm.predict_ops([restickify(4, 2)])


def test_residency_dtype_and_unknown_geometry():
    p = cm.CostParams()
    op = restickify(8, 2)
    for candidate in (
        replace(op, dtype_bytes=4),
        replace(op, transport_read_run_bytes=None),
        replace(op, transport_tile_elems=None),
        replace(op, transport_read_run_bytes=0),
        replace(op, transport_tile_elems=0),
        replace(op, cores=3),
    ):
        assert cm._transport_dma_excess_ns([candidate], p) == 0
    args = [replace(op.args[0], is_lx=True), op.args[1]]
    assert cm._transport_dma_excess_ns([replace(op, args=args)], p) == 0
    is_lx = sympy.Symbol("is_lx", integer=True)
    symbolic = replace(op, args=[replace(op.args[0], is_lx=is_lx), op.args[1]])
    expr = cm._transport_dma_excess_ns([symbolic], p)
    assert expr.subs(is_lx, 1) == 0
    assert expr.subs(is_lx, 0) == cm._transport_dma_excess_ns([op], p)
    assert (
        cm._transport_dma_excess_ns([op], replace(p, transport_dma_ns_per_request={}))
        == 0
    )


def test_staging_copy_keeps_the_source_request_cost():
    p = cm.CostParams()
    direct = replace(restickify(8, 2), hbm_pattern="")
    staging = replace(
        direct, args=[direct.args[0], replace(direct.args[1], is_lx=True)]
    )
    consumer = replace(
        restickify(8, 2), args=[replace(direct.args[0], is_lx=True), direct.args[1]]
    )
    expected = cm._transport_dma_excess_ns([direct], p)
    assert expected > 0
    assert cm._transport_dma_excess_ns([staging, consumer], p) == expected
    is_lx = sympy.Symbol("output_is_lx", integer=True)
    symbolic = replace(
        direct, args=[direct.args[0], replace(direct.args[1], is_lx=is_lx)]
    )
    expr = sympy.sympify(cm._transport_dma_excess_ns([symbolic], p))
    assert expr.subs(is_lx, 0) == expected
    assert expr.subs(is_lx, 1) == expected


@pytest.mark.parametrize(
    "sb,sx,observed_us", [(4, 1, 49.11), (8, 1, 82.32), (4, 2, 153.81), (8, 2, 155.28)]
)
def test_copy_only_control(sb, sx, observed_us):
    # Compare the split-dependent increment: the pre-existing balanced-copy
    # bandwidth/turnaround estimate is not recalibrated by this change.
    op = replace(restickify(sb, sx), hbm_pattern="")
    anchor = replace(restickify(2, 1), hbm_pattern="")
    predicted_delta = cm.predict_ops([op]) - cm.predict_ops([anchor])
    assert float(predicted_delta / 1000) == pytest.approx(observed_us - 34.30, rel=0.15)


def test_long_contiguous_read_has_no_request_surcharge():
    op = replace(restickify(), transport_read_run_bytes=32768)
    assert cm._transport_dma_excess_ns([op], cm.CostParams()) == 0


def test_request_excess_is_not_spill_derated(monkeypatch):
    op = restickify(8, 2, trips=128)
    p = cm.CostParams()
    disabled = replace(p, transport_dma_ns_per_request={})
    extra = float(cm._transport_dma_excess_ns([op], p))
    monkeypatch.setattr(cm, "_lx_spill_bw_derate", lambda *args: 0.5)
    assert float(
        cm.predict_ops([op], p) - cm.predict_ops([op], disabled)
    ) == pytest.approx(extra)


def test_invalid_or_element_strided_geometry_is_unknown():
    assert dcm._contiguous_device_run([X], [128], {X: 0}, {}) is None
    assert dcm._contiguous_device_run([2 * X], [256], {X: 128}, {}) is None


def test_serialization_defaults_and_roundtrip():
    record = cm.op_to_dict(restickify())
    assert cm.op_from_dict(record) == restickify()
    record.pop("transport_read_run_bytes")
    record.pop("transport_tile_elems")
    assert cm._transport_dma_excess_ns([cm.op_from_dict(record)], cm.CostParams()) == 0


@pytest.mark.parametrize("n,trips", [(1024, 1), (1024, 128), (16384, 1)])
def test_symbolic_cpsat_picks_measured_fast_division(n, trips):
    cp_model = pytest.importorskip("ortools.sat.python.cp_model")
    from torch_spyre._inductor.scratchpad.ilp_solver_ortools import _SympyExprToCpSat

    sb, sx = sympy.symbols("split_b split_x", integer=True, positive=True)
    expr = sympy.sympify(cm.predict_ops([restickify(sb, sx, n=n, trips=trips)]))
    choices = [(8, 1), (8, 2), (4, 1), (4, 2), (2, 1), (1, 1)]
    model = cp_model.CpModel()
    division = model.new_int_var(0, len(choices) - 1, "division")
    variables = {}
    tables = {}
    for i, symbol in enumerate((sb, sx)):
        values = [c[i] for c in choices]
        variable = model.new_int_var_from_domain(
            cp_model.Domain.FromValues(values), symbol.name
        )
        model.add_element(division, values, variable)
        variables[symbol.name] = variable
        tables[symbol.name] = (None, values)
    model.minimize(_SympyExprToCpSat(model, variables, tables).convert(expr))
    solver = cp_model.CpSolver()
    assert solver.solve(model) == cp_model.OPTIMAL
    assert choices[solver.value(division)] == (2, 1)
    for i, (b, x) in enumerate(choices):
        expected = float(cm.predict_ops([restickify(b, x, n=n, trips=trips)]))
        assert float(expr.subs({sb: b, sx: x})) == pytest.approx(expected)
        fixed = model.clone()
        fixed.add(division == i)
        assert solver.solve(fixed) == cp_model.OPTIMAL
        assert solver.objective_value == pytest.approx(expected, abs=0.1)


def test_real_layout_extraction_uses_device_order_and_invocation_extent(monkeypatch):
    src = SpyreTensorLayout(
        device_size=[33280, 8, 2, 64],
        stride_map=[1024, 128, 64, 1],
        device_dtype=DataFormats.SEN169_FP16,
    )
    dst = SpyreTensorLayout(
        device_size=[8, 16, 128, 64],
        stride_map=[131072, 64, 1024, 1],
        device_dtype=DataFormats.SEN169_FP16,
    )
    read = MemoryDep("input", N * 1024 + B * 128 + X, (B, X, N), (8, 128, 1024))
    write = MemoryDep("output", B * 131072 + X * 1024 + N, (B, X, N), (8, 128, 1024))
    op = SimpleNamespace(
        get_read_writes=lambda: SimpleNamespace(reads=[read], writes=[write]),
        get_layout=lambda: SimpleNamespace(device_layout=dst),
        get_dtype=lambda: torch.float16,
    )
    graph = SimpleNamespace(
        get_buffer=lambda _: SimpleNamespace(
            get_layout=lambda: SimpleNamespace(device_layout=src)
        )
    )
    monkeypatch.setattr(
        dcm, "iteration_space_from_op", lambda _: {B: 8, X: 128, N: 1024}
    )
    with V.set_graph_handler(graph):
        assert dcm._transport_read_geometry(op, {B: 8, X: 2, N: 1}) == (128, 1048576)
        assert dcm._transport_read_geometry(op) == (None, None)


@pytest.fixture
def staged_transport_graph():
    from torch import fx
    from torch._inductor.graph import GraphLowering
    from torch._inductor.ir import ComputedBuffer, InputBuffer, Pointwise
    from torch._inductor.virtualized import ops
    from torch_spyre._inductor.ir import FixedTiledLayout
    from torch_spyre._inductor.loop_info import CoarseTileInfo, ReadCopyElisionRecord
    from torch_spyre._inductor.pass_utils import commit_iteration_space_ownership

    graph = GraphLowering(fx.symbolic_trace(lambda: None))
    device = torch.device("spyre")

    def layout(shape, strides, device_size, stride_map):
        return FixedTiledLayout(
            device,
            torch.float16,
            list(map(sympy.Integer, shape)),
            list(map(sympy.Integer, strides)),
            SpyreTensorLayout(
                device_size=device_size,
                stride_map=stride_map,
                device_dtype=DataFormats.SEN169_FP16,
            ),
        )

    with V.set_graph_handler(graph):
        source = InputBuffer(
            name="input",
            layout=layout(
                [32768, 8, 128], [1024, 128, 1], [32768, 8, 2, 64], [1024, 128, 64, 1]
            ),
        )
        stage = ComputedBuffer(
            name="stage",
            layout=layout(
                [1024, 8, 128], [1024, 128, 1], [8, 2, 1024, 64], [128, 64, 1024, 1]
            ),
            data=Pointwise(
                device=device,
                dtype=torch.float16,
                inner_fn=lambda i: ops.load("input", 1024 * i[0] + 128 * i[1] + i[2]),
                ranges=list(map(sympy.Integer, [1024, 8, 128])),
            ),
        )
        consumer = ComputedBuffer(
            name="consumer",
            layout=layout(
                [8, 128, 1024],
                [131072, 1024, 1],
                [8, 16, 128, 64],
                [131072, 64, 1024, 1],
            ),
            data=Pointwise(
                device=device,
                dtype=torch.float16,
                inner_fn=lambda i: ops.load("stage", 1024 * i[2] + 128 * i[0] + i[1]),
                ranges=list(map(sympy.Integer, [8, 128, 1024])),
            ),
        )
        for op in (source, stage, consumer):
            graph.name_to_buffer[op.get_name()] = op
        graph.graph_inputs = {"input": source}
        graph.graph_input_names = ["input"]
        graph.operations = [stage, consumer]
        for op in (stage, consumer):
            op.operation_name = op.get_name()
            commit_iteration_space_ownership(op, {})
        stage.loop_info = CoarseTileInfo(
            (0,),
            [sympy.Integer(32)],
            [[]],
            tiled_dims_per_read=[[[(0, sympy.Integer(1024))]]],
        )
        consumer.loop_info = CoarseTileInfo(
            (0,), [sympy.Integer(32)], [[]], tiled_dims_per_read=[[[]]]
        )
        consumer._read_copy_elision_record = ReadCopyElisionRecord(
            consumer_name="consumer",
            copy_name="stage",
            source_name="input",
            direct_inner_fn=lambda i: ops.load(
                "input", 1024 * i[2] + 128 * i[0] + i[1]
            ),
            direct_tiled_dims_per_level=(((2, sympy.Integer(1024)),),),
            direct_squeezed_advance_per_level=((),),
        )
        yield graph, stage, consumer


def test_proven_direct_read_is_priced_with_consumer_splits(staged_transport_graph):
    from torch_spyre._inductor.read_copy_elision import project_transport_read_copies
    from torch_spyre._inductor.pass_utils import iteration_space_from_op

    graph, stage, consumer = staged_transport_graph
    b, x, n = iteration_space_from_op(consumer)
    splits = [
        {b: sb, x: sx, n: 1} for sb, sx in [(1, 1), (2, 1), (4, 1), (8, 1), (8, 2)]
    ]
    original_ownership = consumer.iteration_space_ownership
    original_body = consumer.data
    priced = project_transport_read_copies(graph, {"consumer": splits})
    assert len(priced) == 1
    assert priced[0] is not consumer
    assert graph.operations == [stage, consumer]
    assert consumer.iteration_space_ownership is original_ownership
    assert consumer.data is original_body
    assert next(iter(consumer.get_read_writes().reads)).name == "stage"
    assert next(iter(priced[0].get_read_writes().reads)).name == "input"
    for split in splits:
        feats = dcm.extract_op_features(priced[0], split)
        assert feats.transport_tile_elems == 1048576
        assert feats.transport_read_run_bytes == run_bytes(split[b], split[x])
    sb, sx = sympy.symbols("sb sx", positive=True, integer=True)
    feats = dcm.extract_op_features(priced[0], {b: sb, x: sx, n: 1})
    cost = cm._transport_dma_excess_ns([feats], cm.CostParams())
    assert float(cost.subs({sb: 2, sx: 1})) < float(cost.subs({sb: 8, sx: 2}))


@pytest.mark.parametrize(
    "failure",
    [
        "one_candidate",
        "validation",
        "disabled",
        "shared",
        "source_placement",
        "relayout",
    ],
)
def test_projection_declines_without_a_universal_proof(
    staged_transport_graph, monkeypatch, failure
):
    from torch_spyre._inductor import config, read_copy_elision as rce
    from torch_spyre._inductor.pass_utils import iteration_space_from_op

    graph, stage, consumer = staged_transport_graph
    b, x, n = iteration_space_from_op(consumer)
    splits = [{b: 2, x: 1, n: 1}, {b: 8, x: 2, n: 1}]
    divisions = {"consumer": splits}
    relayout_sources = ()
    proof = rce._prove_matmul_direct_read
    if failure == "one_candidate":

        def prove(op, *args):
            if op.iteration_space_ownership.work_slices[x] == 2:
                return None, "unsupported candidate"
            return proof(op, *args)

        monkeypatch.setattr(rce, "_prove_matmul_direct_read", prove)
    elif failure == "validation":
        monkeypatch.setattr(rce, "_validate_proposal", lambda *args: "invalid loop")
    elif failure == "disabled":
        monkeypatch.setattr(config, "read_copy_elision", False)
    elif failure == "source_placement":
        divisions["input"] = [{}]
    elif failure == "relayout":
        relayout_sources = ("stage",)
    else:
        monkeypatch.setattr(rce, "_copy_readers", lambda *args: [consumer, consumer])
    assert rce.project_transport_read_copies(
        graph, divisions, relayout_sources=relayout_sources
    ) == [
        stage,
        consumer,
    ]
