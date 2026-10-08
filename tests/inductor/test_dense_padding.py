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

"""Focused CPU proof tests; native DMA/mutation tests run separately on Spyre."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import sympy
import torch
from torch._inductor.dependencies import MemoryDep, ReadWrites, StarDep
from torch._inductor.ir import (
    ComputedBuffer,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    Reduction,
    ReductionHint,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor import dense_padding as dp
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.pass_utils import (
    alignment_coordinates,
    iteration_space_from_op,
    logical_iteration_space_from_op,
)


class Layout:
    """Explicit device geometry for CPU tests, without native allocation."""

    def __init__(self, size, stride, dtype=None, arrangement=None):
        self.device_size = list(size)
        self.stride_map = list(stride)
        self.device_dtype = dtype or DataFormats.SEN169_FP16
        self.element_arrangement = arrangement or ElementArrangement.STANDARD
        self.zero_padding_valid_size = []

    def elems_per_stick(self):
        return 64


def _layout(m, n, *, weight=False, row_first=False, leading=False, pad=False):
    if weight:
        stl = Layout([m // 64, n, 64], [64 * n, 1, n])
        if pad:
            stl.zero_padding_valid_size = list(stl.device_size)
            stl.device_size = [(m + 255) // 256 * 4, (n + 255) // 256 * 256, 64]
    else:
        stl = (
            Layout([m, n // 64, 64], [n, 64, 1])
            if row_first
            else Layout([n // 64, m, 64], [64, n, 1])
        )
    size = [1, m, n] if leading else [m, n]
    stride = [m * n, n, 1] if leading else [n, 1]
    return FixedTiledLayout(torch.device("cpu"), torch.float16, size, stride, stl)


def _graph(m=512, n=3200, k=4096, leading=True):
    syms = sympy.symbols("d0 d1 d2", integer=True, nonnegative=True)
    a, b, r = syms
    buffers = {}
    operations = []

    def input_(name, rows, cols, weight=False):
        op = InputBuffer(
            name=name,
            layout=_layout(
                rows, cols, weight=weight, pad=weight, leading=leading and not weight
            ),
        )
        buffers[name] = op
        return op

    input_("x", m, k)
    input_("wg", n, k, True)
    input_("wu", n, k, True)
    input_("wd", k, n, True)

    def rw(op, reads, width, red=None):
        write = MemoryDep(op.name, width * a + b, (a, b), (m, width))
        read_deps = [
            MemoryDep(
                name,
                index,
                (a, b, r) if red else (a, b),
                (m, width, red) if red else (m, width),
            )
            for name, index in reads
        ]
        result = ReadWrites(OrderedSet(read_deps), OrderedSet([write]), OrderedSet())
        op.get_read_writes = lambda: result
        op.operation_name = op.name
        buffers[op.name] = op
        operations.append(op)

    def mm(name, x, w, width, red):
        ranges = [1, m, width] if leading else [m, width]
        data = Reduction(
            device=torch.device("cpu"),
            dtype=torch.float16,
            src_dtype=torch.float16,
            inner_fn=lambda idx, ridx: (
                V.ops.load(x, red * idx[-2] + ridx[0]),
                V.ops.load(w, red * idx[-1] + ridx[0]),
            ),
            ranges=list(map(sympy.Integer, ranges)),
            reduction_ranges=[sympy.Integer(red)],
            reduction_type=BATCH_MATMUL_OP,
            reduction_hint=ReductionHint.DEFAULT,
        )
        op = ComputedBuffer(
            name=name,
            layout=_layout(m, width, row_first=True, leading=leading),
            data=data,
        )
        # Intentionally reverse the operand set order: live roles decide.
        rw(op, [(w, red * b + r), (x, red * a + r)], width, red)

    def pw(name, inputs, function, row_first=True):
        ranges = [1, m, n] if leading else [m, n]
        data = Pointwise(
            device=torch.device("cpu"),
            dtype=torch.float16,
            inner_fn=lambda idx: function(
                *[V.ops.load(src, n * idx[-2] + idx[-1]) for src in inputs]
            ),
            ranges=list(map(sympy.Integer, ranges)),
        )
        op = ComputedBuffer(
            name=name,
            layout=_layout(m, n, row_first=row_first, leading=leading),
            data=data,
        )
        rw(op, [(src, n * a + b) for src in inputs], n)

    mm("gate", "x", "wg", n, k)
    pw("silu", ["gate"], lambda x: V.ops.silu(x))
    mm("up", "x", "wu", n, k)
    pw("mul", ["silu", "up"], lambda x, y: V.ops.mul(x, y), False)
    pw("copy", ["mul"], lambda x: x, False)
    mm("down", "copy", "wd", k, n)
    graph = SimpleNamespace(
        operations=operations,
        graph_input_names=["x", "wg", "wu", "wd"],
        graph_inputs={name: buffers[name] for name in ["x", "wg", "wu", "wd"]},
        get_buffer=buffers.__getitem__,
        get_output_names=lambda: ["down"],
        mutated_inputs=set(),
        buffers=buffers,
        sizevars=SimpleNamespace(optimization_hint=int, simplify=sympy.simplify),
    )
    return graph, syms


@pytest.fixture(autouse=True)
def geometry_only(monkeypatch):
    monkeypatch.setattr(dp, "SpyreTensorLayout", Layout)
    # Real native enums already provide this. The disclosed local enum stub
    # needs an explicit value for the real Python work-division calculations.
    if type(DataFormats.SEN169_FP16).__name__ == "_Member":
        monkeypatch.setattr(DataFormats.SEN169_FP16, "elems_per_stick", lambda: 64)


def test_load_policy_preserves_logical_strides_and_requires_whole_sticks():
    gate = Layout([50, 4096, 64], [262144, 1, 4096])
    padded = dp.padded_linear_layout(gate, (3200, 4096), torch.float16)
    assert padded.device_size == [52, 4096, 64]
    assert padded.stride_map == gate.stride_map
    assert padded.zero_padding_valid_size == []
    down = Layout([64, 3200, 64], [204800, 1, 3200])
    assert dp.padded_linear_layout(down, (4096, 3200), torch.float16).device_size == [
        64,
        3328,
        64,
    ]
    assert dp.padded_linear_layout(gate, (3199, 4096), torch.float16) is None
    assert dp.padded_linear_layout(gate, (3200, 4096), torch.float32) is None


def test_retained_rank3_dense_roles_and_both_physical_axis_orders():
    graph, _ = _graph()
    with V.set_graph_handler(graph):
        mm = dp._matmul(graph.buffers["gate"])
        assert (mm.m, mm.n, mm.k, mm.weight_name) == (512, 3200, 4096, "wg")
        for name, expected in (("gate", [512, 52, 64]), ("mul", [52, 512, 64])):
            op = graph.buffers[name]
            size, stride, body = (
                list(op.layout.size),
                list(op.layout.stride),
                op.data.inner_fn,
            )
            dp._grow_output(op, 3200, 3328)
            assert op.layout.device_layout.device_size == expected
            assert op.layout.size == size and op.layout.stride == stride
            assert op.data.inner_fn is body


def test_physical_domains_preserve_logical_address_decomposition_and_clones():
    graph, (m, n, _) = _graph()
    gate = graph.buffers["gate"]
    with V.set_graph_handler(graph):
        dp._grow_output(gate, 3200, 3328)
        logical = logical_iteration_space_from_op(gate)
        physical = iteration_space_from_op(gate)
        assert logical[n] == 3200 and physical[n] == 3328
        coords = alignment_coordinates(
            gate.layout.device_layout, 3200 * m + n, logical, {}
        )
        assert coords == [m, sympy.floor(n / 64), sympy.Mod(n, 64)]
        cloned = _layout(512, 3200, row_first=True, leading=True)
        dp.copy_padding_layout(gate.layout, cloned)
        assert cloned.dense_padding == (3200, 3328)
        with pytest.raises(Unsupported, match="axis lost"):
            dp.physical_iteration_space(gate, gate.get_read_writes(), {n: 512 * 3200})


@pytest.mark.parametrize("op", [SimpleNamespace(), MagicMock(spec=ComputedBuffer)])
def test_unannotated_operations_keep_logical_ranges_without_layout_queries(op):
    op.get_layout = MagicMock(side_effect=NotImplementedError("MultiOutputLayout"))
    logical = {sympy.Symbol("d0"): sympy.Integer(64)}
    assert not dp.has_dense_padding(op)
    assert dp.physical_iteration_space(op, None, logical) == logical
    op.get_layout.assert_not_called()


def test_synthesized_zero_mask_attribute_does_not_enable_padding():
    assert dp.zero_mask_for_op(MagicMock(), None, {}) == {}


def test_unrelated_extern_operation_does_not_block_dense_chain_selection():
    graph, _ = _graph()
    extern = SimpleNamespace(
        layout=object(),
        get_layout=MagicMock(side_effect=NotImplementedError("MultiOutputLayout")),
        get_read_writes=lambda: ReadWrites(OrderedSet(), OrderedSet(), OrderedSet()),
    )
    graph.operations.append(extern)
    with (
        V.set_graph_handler(graph),
        dp.config.patch({"compiler_dense_padding": True, "ktir_emitter": False}),
    ):
        dp.select_dense_padding(graph)
        assert graph.buffers["down"].dense_reduction_padding == (3200, 3328)
    extern.get_layout.assert_not_called()


@pytest.mark.parametrize("mutation", [False, True])
def test_extern_read_or_mutation_declines_dense_chain(mutation):
    graph, _ = _graph()
    graph.operations.append(
        SimpleNamespace(
            layout=object(),
            get_name=lambda: "extern",
            get_mutation_names=lambda: ["silu"] if mutation else [],
            get_read_writes=lambda: ReadWrites(
                OrderedSet() if mutation else OrderedSet([StarDep("silu")]),
                OrderedSet(),
                OrderedSet(),
            ),
        )
    )
    with (
        V.set_graph_handler(graph),
        dp.config.patch({"compiler_dense_padding": True, "ktir_emitter": False}),
    ):
        dp.select_dense_padding(graph)
        assert graph.buffers["gate"].layout.dense_padding is None
        assert getattr(graph.buffers["down"], "dense_reduction_padding", None) is None


def test_mutation_alias_preserves_selected_physical_domain():
    graph, (_, n, _) = _graph()
    source = graph.buffers["gate"]
    dp._grow_output(source, 3200, 3328)
    graph.mark_buffer_mutated = MagicMock()
    with V.set_graph_handler(graph):
        alias = SimpleNamespace(layout=MutationLayoutSHOULDREMOVE(source))
        logical = logical_iteration_space_from_op(source)
    graph.mark_buffer_mutated.assert_called_once_with("gate")
    assert dp.has_dense_padding(alias)
    assert (
        dp.physical_iteration_space(alias, source.get_read_writes(), logical)[n] == 3328
    )


def test_padded_ownership_rejects_a_changed_symbol_context():
    from torch_spyre._inductor.work_division import TensorDep
    from torch_spyre._inductor.work_division_constraints import (
        aligned_ownership_split_domains,
    )

    graph, _ = _graph()
    gate = graph.buffers["gate"]
    dp._grow_output(gate, 3200, 3328)
    rw = gate.get_read_writes()
    ctx = SimpleNamespace(
        op=gate,
        input_tds=[],
        output_td=TensorDep(next(iter(rw.writes)), gate.layout),
        it_space={sympy.Symbol("renamed"): 3328},
    )
    with (
        V.set_graph_handler(graph),
        pytest.raises(Unsupported, match="cannot prove dense padding ownership"),
    ):
        aligned_ownership_split_domains(ctx)


def test_selected_physical_domains_still_reject_coarse_tiling():
    from torch_spyre._inductor.wsr.coarse_tile import (
        _divide_ranges,
        _divide_reduction_ranges,
    )

    graph, _ = _graph()
    gate, down = graph.buffers["gate"], graph.buffers["down"]
    dp._grow_output(gate, 3200, 3328)
    down.dense_reduction_padding = (3200, 3328)
    assert dp.has_dense_padding(gate) and dp.has_dense_padding(down)
    with pytest.raises(Unsupported, match="coarse tiling a selected"):
        _divide_ranges(gate, sympy.Integer(2), [1])
    with pytest.raises(Unsupported, match="coarse tiling a selected"):
        _divide_reduction_ranges(down, sympy.Integer(2), [0])


def test_planner_selects_one_closed_chain_and_declines_unsafe_consumers():
    graph, syms = _graph()

    # Isolate legality/propagation from calibration. The real cost-model test
    # below separately exercises the production split-domain machinery.
    def price(op, mm):
        return (
            1.0
            if (op.layout.dense_padding or getattr(op, "dense_reduction_padding", None))
            else 2.0
        )

    with (
        V.set_graph_handler(graph),
        patch.object(dp, "_best_cost", price),
        dp.config.patch({"compiler_dense_padding": True, "ktir_emitter": False}),
    ):
        dp.select_dense_padding(graph)
        assert graph.buffers["down"].dense_reduction_padding == (3200, 3328)
        assert not graph.buffers["gate"].dense_padding_zero_mask
        assert graph.buffers["silu"].dense_padding_zero_mask
        assert graph.buffers["mul"].dense_padding_zero_mask
        assert not graph.buffers["copy"].dense_padding_zero_mask
        assert iteration_space_from_op(graph.buffers["down"])[syms[2]] == 3328
    graph, _ = _graph()
    graph.get_output_names = lambda: ["down", "gate"]
    with (
        V.set_graph_handler(graph),
        patch.object(dp, "_best_cost", price),
        dp.config.patch({"compiler_dense_padding": True, "ktir_emitter": False}),
    ):
        dp.select_dense_padding(graph)
        assert graph.buffers["gate"].layout.dense_padding is None


def test_real_work_division_admits_four_way_padded_columns_without_lx_gap():
    from torch_spyre._inductor.work_division import (
        enumerate_work_division_candidates,
        work_division_context_for_op,
    )
    from torch_spyre._inductor.scratchpad.utils import _would_produce_lx_back_gap

    graph, (m, n, k) = _graph()
    gate = graph.buffers["gate"]
    with V.set_graph_handler(graph):
        assert 4 not in work_division_context_for_op(gate, 32).factor_domain(n)
        assert _would_produce_lx_back_gap(graph, "wg", [0])
        dp._grow_output(gate, 3200, 3328)
        assert 4 in work_division_context_for_op(gate, 32).factor_domain(n)
        assert work_division_context_for_op(gate, 32).is_legal({m: 8, n: 4, k: 1})
        for name in ("silu", "mul"):
            op = graph.buffers[name]
            dp._grow_output(op, 3200, 3328)
            op.dense_padding_zero_mask = True
            ctx = work_division_context_for_op(op, 32)
            assert not ctx.is_legal({m: 8, n: 4})
            assert ctx.is_legal({m: 32, n: 1})
            candidates = enumerate_work_division_candidates(op, 32)
            assert candidates and all(splits[n] == 1 for splits in candidates)
        assert not _would_produce_lx_back_gap(graph, "gate", [0, 1])
        assert not _would_produce_lx_back_gap(graph, "wg", [0])


def test_absent_certificate_and_decode_keep_logical_execution():
    graph, _ = _graph()
    graph.buffers["wg"].layout.device_layout.zero_padding_valid_size = []
    with (
        V.set_graph_handler(graph),
        dp.config.patch({"compiler_dense_padding": True, "ktir_emitter": False}),
    ):
        dp.select_dense_padding(graph)
        assert graph.buffers["gate"].layout.dense_padding is None
    decode, _ = _graph(m=1)
    with (
        V.set_graph_handler(decode),
        dp.config.patch({"compiler_dense_padding": True, "ktir_emitter": False}),
    ):
        dp.select_dense_padding(decode)
        assert decode.buffers["gate"].layout.dense_padding is None


@pytest.mark.parametrize("fixed", [False, True])
def test_certified_input_padding_is_not_lost_in_an_lx_clone(monkeypatch, fixed):
    from torch_spyre._inductor.scratchpad import allocator as alloc

    graph, _ = _graph()
    graph.try_get_buffer = graph.buffers.get
    planner = alloc.ScratchpadAllocator(alloc.FirstFitLayoutSolver, 1 << 20)
    monkeypatch.setattr(alloc, "clone_at_graph_boundaries", lambda: True)
    # Exercise the shared verdict used by both placement and joint planning.
    # A normal input passes these unrelated admission checks.
    monkeypatch.setattr(
        planner, "_is_index_or_indirectly_accessed", lambda *args: False
    )
    monkeypatch.setattr(
        alloc.GraphEditor, "all_uses_are_rewritable", lambda *args: True
    )
    monkeypatch.setattr(alloc, "buffer_not_read_in_full", lambda *args: False)
    monkeypatch.setattr(planner, "_restickify_barrier", lambda *args: None)
    monkeypatch.setattr(alloc, "_would_produce_lx_back_gap", lambda *args: False)
    with dp.config.patch({"enable_lx_context_switching": True}):
        reason = planner._input_residency_reason(
            graph, "wd", [0, 1], ncores={"wd": 4}, division_is_fixed=fixed
        )
        assert reason == "certified padding requires a physical input clone"
        graph.buffers["wd"].layout.device_layout.zero_padding_valid_size = []
        assert (
            planner._input_residency_reason(
                graph, "wd", [0, 1], ncores={"wd": 4}, division_is_fixed=fixed
            )
            is None
        )


def test_existing_cost_model_selects_padded_prefill_candidate():
    graph, _ = _graph()
    with (
        V.set_graph_handler(graph),
        dp.config.patch({"compiler_dense_padding": True, "ktir_emitter": False}),
    ):
        dp.select_dense_padding(graph)
        assert graph.buffers["down"].dense_reduction_padding == (3200, 3328)


def test_sdsc_emits_zero_mask_and_logical_decode_hbm_gap():
    from torch_spyre._inductor.codegen.superdsc import parse_op_spec
    from torch_spyre._inductor.op_spec import OpSpec, TensorArg

    m, n, k = sympy.symbols("c0 c1 c2", integer=True, nonnegative=True)
    dtype = DataFormats.SEN169_FP16

    def arg(input_, dims, coords, allocation):
        return TensorArg(input_, 0 if input_ else -1, dtype, dims, coords, allocation)

    coords = [m, sympy.floor(n / 64), sympy.Mod(n, 64)]
    spec = OpSpec(
        "silu",
        False,
        {m: (512, 32), n: (3328, 1)},
        [
            arg(True, [512, 52, 64], coords, {"lx": 0}),
            arg(False, [512, 52, 64], coords, {"lx": 200000}),
        ],
        {dp.ZERO_MASK_INFO_KEY: {"logical": 3200, "physical": 3328}},
    )
    core = sympy.Symbol("core_id")
    spec.core_id_to_work_slice = {m: core, n: sympy.S.Zero}
    sdsc, mapping = parse_op_spec(spec)
    assert sdsc.coordinate_masking == {mapping[n]: [[3200, 128]]}
    assert sdsc.constants["samv-maskvalue"] == 0
    assert all(not a.backGap for a in sdsc.args)
    spec.iteration_space = {m: (512, 8), n: (3328, 4)}
    spec.core_id_to_work_slice = {m: sympy.Mod(core, 8), n: sympy.floor(core / 8)}
    with pytest.raises(ValueError, match="cannot be split across cores"):
        parse_op_spec(spec)

    spec = OpSpec(
        "batchmatmul",
        True,
        {n: (4096, 2), k: (3200, 16)},
        [
            arg(
                True,
                [50, 1, 64],
                [sympy.floor(k / 64), sympy.S.Zero, sympy.Mod(k, 64)],
                {"hbm": 0},
            ),
            arg(
                True,
                [64, 3328, 64],
                [sympy.floor(n / 64), k, sympy.Mod(n, 64)],
                {"hbm": 1},
            ),
            arg(
                False,
                [64, 1, 64],
                [sympy.floor(n / 64), sympy.S.Zero, sympy.Mod(n, 64)],
                {"hbm": 2},
            ),
        ],
        {},
    )
    spec.core_id_to_work_slice = {n: sympy.Mod(core, 2), k: sympy.floor(core / 2)}
    sdsc, mapping = parse_op_spec(spec)
    assert sdsc.iteration_space[mapping[k]] == 3200
    assert sdsc.args[1].backGap[mapping[k]] == 128
