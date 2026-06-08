# Copyright 2025 The Torch-Spyre Authors.
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

"""Unit tests for the coarse-tiling loop IR infrastructure.

Covers six areas, each in its own class group:
  1. LoopSpec data structure and codegen_kernel serialization (TestLoopSpec*,
     TestIterOpSpecs, TestCodegenOpSpecListRoundtrip)
  2. coarse_tile IR pass: range rewriting, attribute stamping, nested groups
     (TestDivideRanges, TestCoarseTile, TestCoarseTileNested)
  3. CountedLoopSchedulerNode, build_loop_scheduler_nodes, and
     _tiled_syms_for_sched_node_at_depth
     (TestHelpers, TestBuildLoopSchedulerNodes, TestTiledSymsForSchedNode)
  4. generate_sdsc and compile_op_spec symbol/affine-stride paths
     (TestTiledByteStride, TestGenerateSdscTiledSymbols,
      TestCompileOpSpecTwoTiledSymbols, TestCompileOpSpecSymbolMapping)
  5. generate_bundle MLIR output: loop structure, affine maps, symbol constants
     (TestGenerateBundleMlir, TestFindUnimplemented,
      TestGenerateBundleMlirSnapshot, TestGenerateBundleMlirWithAffineStrides,
      TestGenerateBundleNestedTiling, TestGenerateBundleUnrollPath)
  6. Buffer propagation: consumer analysis helpers for insert_tiling_propagation
     (TestCoarseTileBufferPropagation)

No Spyre device or backend compiler is required.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sympy import Integer, Mod, Symbol, floor, simplify, sympify  # noqa: F401

import torch
from torch import fx
from torch._inductor.utils import IndentedBuffer
from torch.utils._ordered_set import OrderedSet

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.codegen.compute_ops import (
    _tiled_byte_stride,
    generate_sdsc,
)
from torch_spyre._inductor.codegen.superdsc import (
    SDSCArgs,
    SDSCSpec,
    compile_op_spec,
    parse_op_spec,
)
from torch_spyre._inductor.constants import (
    SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY,
    SHARED_WEIGHT_UNIT_BMM_INFO_KEY,
)
from torch_spyre._inductor.loop_info import CoarseTileInfo
from torch_spyre._inductor.coarse_tile import coarse_tile, _divide_ranges
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp
from torch_spyre._inductor.scheduler import (
    CountedLoopSchedulerNode,
    _loop_count,
    _loop_group_id,
    build_loop_scheduler_nodes,
)
from torch_spyre._inductor.spyre_kernel import (
    _codegen_op_spec_list,
    _iter_op_specs,
    _preserve_shared_weight_unit_bmm_dim,
    _shared_weight_unit_bmm_info_from_sizes,
)
from torch_spyre._inductor.temp_passes import bmm_unflatten_pass

_FP16 = DataFormats.SEN169_FP16


# ===========================================================================
# Shared helpers
# ===========================================================================

# Eval namespace for LoopSpec/OpSpec round-trip tests.
_EVAL_NS = {
    "LoopSpec": LoopSpec,
    "OpSpec": OpSpec,
    "TensorArg": TensorArg,
    "UnimplementedOp": UnimplementedOp,
    "DataFormats": DataFormats,
    "sympify": sympify,
}


def _make_tensor_arg(arg_index: int = 0, is_input: bool = True) -> TensorArg:
    x = Symbol("x0")
    return TensorArg(
        is_input=is_input,
        arg_index=arg_index,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[4, 64],
        device_coordinates=[x, Integer(0)],
        allocation=None,
    )


def _make_op_spec(op: str = "add", arg_index: int = 0) -> OpSpec:
    """Full OpSpec with tensor args — used by LoopSpec round-trip tests."""
    x0 = Symbol("x0")
    return OpSpec(
        op=op,
        is_reduction=False,
        iteration_space={x0: (Integer(128), 1)},
        args=[
            _make_tensor_arg(arg_index=arg_index, is_input=True),
            _make_tensor_arg(arg_index=arg_index + 1, is_input=False),
        ],
        op_info={},
    )


def _make_minimal_op_spec(name: str) -> OpSpec:
    """Minimal OpSpec with empty args — used by bundle.mlir tests."""
    return OpSpec(op=name, is_reduction=False, iteration_space={}, args=[], op_info={})


def _roundtrip(specs):
    """Serialize specs to Python source and eval back."""

    def sympy_str(x):
        return "sympify('" + str(x) + "')"

    buf = IndentedBuffer()
    buf.writeline("[")
    with buf.indent():
        _codegen_op_spec_list(specs, buf, sympy_str)
    buf.writeline("]")
    return eval(buf.getvalue(), _EVAL_NS)  # noqa: S307


# ---------------------------------------------------------------------------
# coarse_tile pass helpers
# ---------------------------------------------------------------------------


def _make_pointwise(ranges):
    """Return a fake Pointwise with the given ranges."""
    from torch._inductor.ir import Pointwise

    pw = MagicMock(spec=Pointwise)
    pw.ranges = list(ranges)
    return pw


def _make_reduction(ranges, reduction_ranges):
    """Return a fake Reduction with the given ranges and reduction_ranges."""
    from torch._inductor.ir import Reduction

    red = MagicMock(spec=Reduction)
    red.ranges = list(ranges)
    red.reduction_ranges = list(reduction_ranges)
    return red


def _make_op(data, name="op0"):
    """Return a fake ComputedBuffer wrapping data."""
    from torch._inductor.ir import ComputedBuffer

    op = MagicMock(spec=ComputedBuffer)
    op.data = data
    op.layout = MagicMock()
    op.get_operation_name.return_value = name
    op.get_name.return_value = name
    del op.loop_info
    return op


def _make_hinted_op(data, name="op0", hints=((0, 0),)):
    """Return a fake ComputedBuffer with DimHints for use with coarse_tile().

    ``hints`` is a sequence of ``(hint_id, dim_index)`` pairs, one per tiling
    level.  Each pair produces a DimHint whose ``loop_var`` is the symbol
    ``c{dim_index}``, matching the mock output coords built by this helper
    (``coords[i] = c{i}``).  This convention is valid for mock ops where no
    size-1 dims precede the tiled dimension.
    """
    import sympy
    from torch_spyre._inductor.propagate_hints import DimHint

    op = _make_op(data, name)

    # Build loop_var symbols. coords[i] = cI so _loop_var_to_ranges_pos
    # resolves correctly for mock ops (no size-1 dims in test data).
    n_ranges = len(data.ranges)
    op._test_out_coords = [sympy.Symbol(f"c{i}") for i in range(n_ranges)]

    op.dim_hints = [
        DimHint(
            dim_names=[f"dim{dim_index}"],
            split_count=1,
            loop_var=sympy.Symbol(f"c{dim_index}"),
            is_reduction=False,
            hint_id=hint_id,
        )
        for hint_id, dim_index in hints
    ]
    return op


def _make_non_computed_op(name="extern0"):
    """Return a fake non-ComputedBuffer operation."""
    from torch._inductor.ir import Operation

    op = MagicMock(spec=Operation)
    op.get_operation_name.return_value = name
    return op


def _graph(operations):
    """Wrap an ops list as the GraphLowering-like object coarse_tile() expects.

    coarse_tile() only reads ``graph.operations`` and mutates that list in
    place, so a namespace over the same list reproduces the real GraphLowering
    behavior for these unit tests.
    """
    return SimpleNamespace(operations=operations)


# ---------------------------------------------------------------------------
# Scheduler node helpers
# ---------------------------------------------------------------------------


def _make_scheduler():
    """Return a minimal fake Scheduler."""
    sched = MagicMock()
    sched.name_to_fused_node = {}
    sched.removed_ops = set()
    return sched


def _make_ir_op(loop_group_id=None, loop_count=None, name="op"):
    """Return a fake ir.Operation optionally stamped with loop_info.

    loop_count must be a list of trip counts (one per nesting level), matching
    the contract stamped by coarse_tile().  A bare Expr is accepted as a
    convenience shorthand and is wrapped in a 1-element list.
    """
    op = MagicMock()
    op.name = name
    if loop_group_id is not None:
        counts = loop_count if isinstance(loop_count, list) else [loop_count]
        op.loop_info = CoarseTileInfo(
            loop_group_id=loop_group_id,
            loop_count=counts,
            loop_tiled_dims=[],
        )
    else:
        del op.loop_info
    return op


def _make_snode(scheduler, ir_op, name="buf0"):
    """Return a fake SchedulerNode wrapping ir_op."""
    from torch._inductor.scheduler import SchedulerNode

    snode = MagicMock(spec=SchedulerNode)
    snode.scheduler = scheduler
    snode.node = ir_op
    snode.get_name.return_value = name
    snode.get_nodes.return_value = [snode]
    snode.ancestors = set()
    snode.min_order = 0
    snode.max_order = 0
    snode.read_writes = MagicMock()
    snode.read_writes.reads_and_writes.return_value = []
    snode.outputs_by_name = {}
    return snode


# ---------------------------------------------------------------------------
# SDSC helpers
# ---------------------------------------------------------------------------


def _make_sdsc_spec(
    s: Symbol,
    *,
    iter_range: int = 64,
    device_stride: int = 128,
    start_address: int = 0x1000,
    allocation: dict | None = None,
    num_cores: int = 1,
) -> SDSCSpec:
    """Build a minimal SDSCSpec with one HBM tensor and one iteration-space symbol."""
    if allocation is None:
        allocation = {"hbm": start_address}
    tensor = SDSCArgs(
        layout="A",
        dim_order=[s],
        data_format=_FP16,
        scales={s: 1},
        strides={s: device_stride},
        offsets={s: 0},
        max_dim_sizes={s: -1},
        allocation=allocation,
        start_address=start_address,
        backGap={},
    )
    return SDSCSpec(
        opfunc="add",
        execution_unit="sfp",
        data_format=_FP16,
        num_inputs=1,
        iteration_space={s: iter_range},
        num_cores=num_cores,
        work_slices={s: 1},
        core_id_to_work_slice={s: Integer(0)},
        padding={},
        layouts={
            "A": {
                "dim_order": [s],
                "stick_dim_order": s,
                "stick_size": 64,
            }
        },
        args=[tensor],
        constants={},
        coordinate_masking={},
    )


def _make_tiled_op_spec() -> OpSpec:
    """Minimal OpSpec with tiled_symbols that compile_op_spec can process."""
    c0 = Symbol("c0")
    fp16 = _FP16
    tensor_in = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=fp16,
        device_size=[2, 64],
        device_coordinates=[Integer(0), c0],
        allocation={"hbm": 0x1000},
    )
    tensor_out = TensorArg(
        is_input=False,
        arg_index=1,
        device_dtype=fp16,
        device_size=[2, 64],
        device_coordinates=[Integer(0), c0],
        allocation={"hbm": 0x2000},
    )
    return OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={c0: (Integer(128), 1)},
        args=[tensor_in, tensor_out],
        op_info={},
        tiled_symbols=[c0],
    )


# ---------------------------------------------------------------------------
# bundle.mlir test helpers
# ---------------------------------------------------------------------------


def _fake_compile_op_spec(
    idx: int,
    op_spec: OpSpec,
    symbols: list,
    symbol_id_offset: int = 0,
    use_symbols: bool = False,
):
    """Stub that returns (json, [], [], []) — no real SDSC compilation."""
    return {f"{idx}_{op_spec.op}": {"op": op_spec.op}}, [], [], []


def _read_mlir(output_dir: str) -> str:
    with open(os.path.join(output_dir, "bundle.mlir")) as f:
        return f.read()


def _make_tiled_json(idx: int, sym_id: int) -> dict:
    """Return a minimal SDSC JSON with one HBM tensor whose symbol ID is sym_id."""
    return {
        f"{idx}_add": {
            "numCoresUsed_": 1,
            "dscs_": [
                {
                    "add": {
                        "scheduleTree_": [
                            {
                                "component_": "hbm",
                                "startAddressCoreCorelet_": {
                                    "data_": {"[0, 0, 0]": str(sym_id)}
                                },
                            }
                        ]
                    }
                }
            ],
        }
    }


# ===========================================================================
# 0. CoarseTileInfo dataclass
# ===========================================================================


class TestCoarseTileInfo(unittest.TestCase):
    def test_fields(self):
        info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(4)],
            loop_tiled_dims=[[0]],
        )
        self.assertEqual(info.loop_group_id, (0,))
        self.assertEqual(info.loop_count, [Integer(4)])
        self.assertEqual(info.loop_tiled_dims, [[0]])

    def test_nested(self):
        info = CoarseTileInfo(
            loop_group_id=(0, 0),
            loop_count=[Integer(4), Integer(2)],
            loop_tiled_dims=[[0], [1]],
        )
        self.assertEqual(info.loop_group_id, (0, 0))
        self.assertEqual(info.loop_count, [Integer(4), Integer(2)])
        self.assertEqual(info.loop_tiled_dims, [[0], [1]])


# ===========================================================================
# 1. LoopSpec data structure and codegen serialization
# ===========================================================================


class TestLoopSpecDataclass(unittest.TestCase):
    def test_flat_body(self):
        op = _make_op_spec()
        loop = LoopSpec(count=Integer(4), body=[op])
        self.assertEqual(loop.count, Integer(4))
        self.assertEqual(len(loop.body), 1)
        self.assertIs(loop.body[0], op)

    def test_nested_body(self):
        inner = LoopSpec(count=Integer(2), body=[_make_op_spec("mul")])
        outer = LoopSpec(count=Integer(4), body=[_make_op_spec("add"), inner])
        self.assertEqual(len(outer.body), 2)
        self.assertIsInstance(outer.body[1], LoopSpec)

    def test_empty_body(self):
        loop = LoopSpec(count=Integer(8), body=[])
        self.assertEqual(loop.body, [])


class TestIterOpSpecs(unittest.TestCase):
    def test_flat_list(self):
        specs = [_make_op_spec("add"), _make_op_spec("mul")]
        result = list(_iter_op_specs(specs))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].op, "add")
        self.assertEqual(result[1].op, "mul")

    def test_skips_unimplemented(self):
        specs = [UnimplementedOp(op="foo"), _make_op_spec("add")]
        result = list(_iter_op_specs(specs))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].op, "add")

    def test_single_level_loop(self):
        inner = [_make_op_spec("add"), _make_op_spec("mul")]
        specs = [LoopSpec(count=Integer(4), body=inner)]
        result = list(_iter_op_specs(specs))
        self.assertEqual([s.op for s in result], ["add", "mul"])

    def test_nested_loop_depth_first(self):
        innermost = [_make_op_spec("c")]
        middle = [_make_op_spec("b"), LoopSpec(count=Integer(2), body=innermost)]
        specs = [_make_op_spec("a"), LoopSpec(count=Integer(4), body=middle)]
        result = list(_iter_op_specs(specs))
        self.assertEqual([s.op for s in result], ["a", "b", "c"])

    def test_empty(self):
        self.assertEqual(list(_iter_op_specs([])), [])


class TestCodegenOpSpecListRoundtrip(unittest.TestCase):
    def test_flat_op_spec(self):
        original = [_make_op_spec("add")]
        result = _roundtrip(original)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], OpSpec)
        self.assertEqual(result[0].op, "add")

    def test_unimplemented_op(self):
        original = [UnimplementedOp(op="unknown")]
        result = _roundtrip(original)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], UnimplementedOp)
        self.assertEqual(result[0].op, "unknown")

    def test_single_loop_wrapping_two_ops(self):
        body = [_make_op_spec("add"), _make_op_spec("mul")]
        original = [LoopSpec(count=Integer(4), body=body)]
        result = _roundtrip(original)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], LoopSpec)
        self.assertEqual(result[0].count, Integer(4))
        self.assertEqual(len(result[0].body), 2)
        self.assertEqual(result[0].body[0].op, "add")
        self.assertEqual(result[0].body[1].op, "mul")

    def test_nested_loop(self):
        inner_loop = LoopSpec(count=Integer(2), body=[_make_op_spec("inner")])
        original = [
            LoopSpec(count=Integer(8), body=[_make_op_spec("outer"), inner_loop])
        ]
        result = _roundtrip(original)
        outer = result[0]
        self.assertIsInstance(outer, LoopSpec)
        self.assertEqual(outer.count, Integer(8))
        self.assertEqual(outer.body[0].op, "outer")
        inner = outer.body[1]
        self.assertIsInstance(inner, LoopSpec)
        self.assertEqual(inner.count, Integer(2))
        self.assertEqual(inner.body[0].op, "inner")

    def test_symbolic_count(self):
        s = Symbol("s0")
        original = [LoopSpec(count=s, body=[_make_op_spec("add")])]
        result = _roundtrip(original)
        self.assertIsInstance(result[0], LoopSpec)
        self.assertEqual(result[0].count, s)

    def test_mixed_flat_and_loop(self):
        original = [
            _make_op_spec("before"),
            LoopSpec(count=Integer(4), body=[_make_op_spec("body")]),
            _make_op_spec("after"),
        ]
        result = _roundtrip(original)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], OpSpec)
        self.assertIsInstance(result[1], LoopSpec)
        self.assertIsInstance(result[2], OpSpec)

    def test_arg_index_preserved(self):
        arg = _make_tensor_arg(arg_index=3)
        op = OpSpec(
            op="relu",
            is_reduction=False,
            iteration_space={Symbol("x0"): (Integer(64), 1)},
            args=[arg],
            op_info={},
        )
        original = [LoopSpec(count=Integer(2), body=[op])]
        result = _roundtrip(original)
        self.assertEqual(result[0].body[0].args[0].arg_index, 3)


# ===========================================================================
# 2. coarse_tile IR pass
# ===========================================================================


class TestDivideRanges(unittest.TestCase):
    def test_pointwise_single_dim_divided(self):
        data = _make_pointwise([Integer(64)])
        op = _make_op(data)
        _divide_ranges(op, Integer(4), tiled_dims=[0])
        self.assertEqual(data.ranges[0], Integer(16))

    def test_pointwise_symbolic_count(self):
        k = Symbol("K", positive=True)
        n = Symbol("N", positive=True)
        data = _make_pointwise([n])
        op = _make_op(data)
        _divide_ranges(op, k, tiled_dims=[0])
        self.assertEqual(simplify(data.ranges[0] - n / k), 0)

    def test_pointwise_multidim_default_tiles_outermost_only(self):
        data = _make_pointwise([Integer(32), Integer(8)])
        op = _make_op(data)
        _divide_ranges(op, Integer(4), tiled_dims=[0])
        self.assertEqual(data.ranges[0], Integer(8))
        self.assertEqual(data.ranges[1], Integer(8))

    def test_tiled_dims_indices_0_1(self):
        data = _make_pointwise([Integer(32), Integer(16), Integer(4)])
        op = _make_op(data)
        _divide_ranges(op, Integer(4), tiled_dims=[0, 1])
        self.assertEqual(data.ranges[0], Integer(8))
        self.assertEqual(data.ranges[1], Integer(4))
        self.assertEqual(data.ranges[2], Integer(4))

    def test_tiled_dims_empty_no_change(self):
        data = _make_pointwise([Integer(32)])
        op = _make_op(data)
        original = list(data.ranges)
        _divide_ranges(op, Integer(4), tiled_dims=[])
        self.assertEqual(data.ranges, original)

    def test_empty_ranges_no_change(self):
        data = _make_pointwise([])
        op = _make_op(data)
        _divide_ranges(op, Integer(4), tiled_dims=[0])
        self.assertEqual(data.ranges, [])

    def test_reduction_outer_dims_divided_inner_untouched(self):
        data = _make_reduction([Integer(64)], [Integer(128)])
        op = _make_op(data)
        _divide_ranges(op, Integer(4), tiled_dims=[0])
        self.assertEqual(data.ranges[0], Integer(16))
        self.assertEqual(data.reduction_ranges[0], Integer(128))

    def test_non_loops_type_skipped(self):
        from torch._inductor.ir import Operation

        op = _make_op(MagicMock(spec=Operation))
        _divide_ranges(op, Integer(4), tiled_dims=[0])


def _mock_op_out_coords(op):
    """Return pre-built coords stored on op by _make_hinted_op, or empty list."""
    return getattr(op, "_test_out_coords", [])


class TestCoarseTile(unittest.TestCase):
    def setUp(self):
        self._patch = patch(
            "torch_spyre._inductor.coarse_tile.op_out_coords",
            side_effect=_mock_op_out_coords,
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def _run(self, all_ops, groups, **kwargs):
        coarse_tile(_graph(all_ops), groups, **kwargs)

    def test_empty_groups_list_is_noop(self):
        data = _make_pointwise([Integer(32)])
        op = _make_op(data, "op0")
        original = list(data.ranges)
        coarse_tile(_graph([op]), [])
        self.assertFalse(hasattr(op, "loop_info") and op.loop_info != MagicMock())
        self.assertEqual(data.ranges, original)

    def test_non_computed_buffer_skipped(self):
        op_extern = _make_non_computed_op("extern0")
        data = _make_pointwise([Integer(16)])
        op_computed = _make_hinted_op(data, "op0", hints=((0, 0),))
        coarse_tile(
            _graph([op_extern, op_computed]),
            [([op_extern, op_computed], [(0, Integer(2))])],
        )
        self.assertEqual(op_computed.loop_info.loop_group_id, (0,))
        self.assertEqual(data.ranges[0], Integer(8))

    def test_symbolic_count(self):
        k = Symbol("K", positive=True)
        n = Symbol("N", positive=True)
        data = _make_pointwise([n])
        op = _make_hinted_op(data, "op0", hints=((0, 0),))
        coarse_tile(_graph([op]), [([op], [(0, k)])])
        self.assertEqual(op.loop_info.loop_count, [k])
        self.assertEqual(simplify(data.ranges[0] - n / k), 0)

    def test_non_contiguous_group_raises(self):
        d0 = _make_pointwise([Integer(32)])
        d1 = _make_pointwise([Integer(32)])
        d2 = _make_pointwise([Integer(32)])
        op0 = _make_hinted_op(d0, "op0", hints=((0, 0),))
        op1 = _make_hinted_op(d1, "op1", hints=((0, 0),))
        op2 = _make_hinted_op(d2, "op2", hints=((0, 0),))
        with self.assertRaises(RuntimeError):
            coarse_tile(_graph([op0, op1, op2]), [([op0, op2], [(0, Integer(4))])])

    def test_op_not_in_operations_raises(self):
        data = _make_pointwise([Integer(32)])
        op_known = _make_hinted_op(data, "op0", hints=((0, 0),))
        op_unknown = _make_hinted_op(
            _make_pointwise([Integer(8)]), "unknown", hints=((0, 0),)
        )
        with self.assertRaises(RuntimeError):
            coarse_tile(_graph([op_known]), [([op_unknown], [(0, Integer(2))])])


class TestCoarseTileNested(unittest.TestCase):
    """Verify that the nested group format [(hint_id, K1), ...] works."""

    def setUp(self):
        self._patch = patch(
            "torch_spyre._inductor.coarse_tile.op_out_coords",
            side_effect=_mock_op_out_coords,
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_nested_spec_stamps_list_attributes(self):
        data = _make_pointwise([Integer(256), Integer(128)])
        op = _make_hinted_op(data, "op0", hints=((1, 0), (2, 1)))
        coarse_tile(_graph([op]), [([op], [(1, Integer(4)), (2, Integer(2))])])
        self.assertEqual(op.loop_info.loop_group_id, (0, 0))
        self.assertEqual(op.loop_info.loop_count, [Integer(4), Integer(2)])
        self.assertEqual(op.loop_info.loop_tiled_dims, [[0], [1]])

    def test_nested_spec_divides_ranges_both_levels(self):
        data = _make_pointwise([Integer(256), Integer(128)])
        op = _make_hinted_op(data, "op0", hints=((1, 0), (2, 1)))
        coarse_tile(_graph([op]), [([op], [(1, Integer(4)), (2, Integer(2))])])
        self.assertEqual(data.ranges[0], Integer(64))
        self.assertEqual(data.ranges[1], Integer(64))

    def test_nested_spec_outer_only_divides_outer_dim(self):
        data = _make_pointwise([Integer(32), Integer(64), Integer(16)])
        op = _make_hinted_op(data, "op0", hints=((1, 0), (2, 1)))
        coarse_tile(_graph([op]), [([op], [(1, Integer(4)), (2, Integer(8))])])
        self.assertEqual(data.ranges[0], Integer(8))
        self.assertEqual(data.ranges[1], Integer(8))
        self.assertEqual(data.ranges[2], Integer(16))

    def test_single_and_nested_groups_coexist(self):
        """Group 0: single-level spec tiling dim 0.  Group 1: two-level nested spec."""
        d0 = _make_pointwise([Integer(64), Integer(32)])
        d1 = _make_pointwise([Integer(128), Integer(64)])
        op0 = _make_hinted_op(d0, "op0", hints=((1, 0),))
        op1 = _make_hinted_op(d1, "op1", hints=((2, 0), (3, 1)))
        coarse_tile(
            _graph([op0, op1]),
            [
                ([op0], [(1, Integer(4))]),
                ([op1], [(2, Integer(4)), (3, Integer(2))]),
            ],
        )
        self.assertEqual(op0.loop_info.loop_group_id, (0,))
        self.assertEqual(op0.loop_info.loop_count, [Integer(4)])
        self.assertEqual(op0.loop_info.loop_tiled_dims, [[0]])
        self.assertEqual(d0.ranges[0], Integer(16))
        self.assertEqual(d0.ranges[1], Integer(32))
        self.assertEqual(op1.loop_info.loop_group_id, (1, 0))
        self.assertEqual(op1.loop_info.loop_count, [Integer(4), Integer(2)])
        self.assertEqual(op1.loop_info.loop_tiled_dims, [[0], [1]])
        self.assertEqual(d1.ranges[0], Integer(32))
        self.assertEqual(d1.ranges[1], Integer(32))

    def test_nested_same_dim_different_counts(self):
        data = _make_pointwise([Integer(256)])
        op = _make_hinted_op(data, "op0", hints=((1, 0), (2, 0)))
        coarse_tile(_graph([op]), [([op], [(1, Integer(4)), (2, Integer(2))])])
        self.assertEqual(data.ranges[0], Integer(32))
        self.assertEqual(op.loop_info.loop_count, [Integer(4), Integer(2)])
        self.assertEqual(op.loop_info.loop_tiled_dims, [[0], [0]])


# ===========================================================================
# 3. CountedLoopSchedulerNode and build_loop_scheduler_nodes
# ===========================================================================


class TestHelpers(unittest.TestCase):
    def test_loop_group_id_present(self):
        sched = _make_scheduler()
        op = _make_ir_op(loop_group_id=(0,), loop_count=Integer(4))
        snode = _make_snode(sched, op)
        self.assertEqual(_loop_group_id(snode), (0,))

    def test_loop_group_id_absent(self):
        sched = _make_scheduler()
        op = _make_ir_op()
        snode = _make_snode(sched, op)
        self.assertIsNone(_loop_group_id(snode))

    def test_loop_count(self):
        sched = _make_scheduler()
        op = _make_ir_op(loop_group_id=(0,), loop_count=Integer(8))
        snode = _make_snode(sched, op)
        self.assertEqual(_loop_count(snode, depth=0), Integer(8))

    def test_loop_count_symbolic(self):
        sched = _make_scheduler()
        s = Symbol("s0")
        op = _make_ir_op(loop_group_id=(0,), loop_count=s)
        snode = _make_snode(sched, op)
        self.assertEqual(_loop_count(snode, depth=0), s)


class TestBuildLoopSchedulerNodes(unittest.TestCase):
    def _run(self, nodes):
        created = []

        def fake_create(snodes, loop_count):
            node = MagicMock(spec=CountedLoopSchedulerNode)
            node.snodes = snodes
            node.loop_count = loop_count
            node.get_nodes.return_value = snodes
            node.get_name.return_value = "_".join(n.get_name() for n in snodes)
            node.scheduler = snodes[0].scheduler
            created.append(node)
            return node

        with patch.object(
            CountedLoopSchedulerNode, "create", staticmethod(fake_create)
        ):
            result = build_loop_scheduler_nodes(nodes)
        return result, created

    def test_passthrough_no_loop_group(self):
        sched = _make_scheduler()
        nodes = [
            _make_snode(sched, _make_ir_op(), "a"),
            _make_snode(sched, _make_ir_op(), "b"),
        ]
        result, created = self._run(nodes)
        self.assertEqual(result, nodes)
        self.assertEqual(created, [])

    def test_single_group_two_nodes(self):
        sched = _make_scheduler()
        n1 = _make_snode(sched, _make_ir_op((0,), Integer(4)), "a")
        n2 = _make_snode(sched, _make_ir_op((0,), Integer(4)), "b")
        result, created = self._run([n1, n2])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].loop_count, Integer(4))
        self.assertIn(n1, created[0].snodes)
        self.assertIn(n2, created[0].snodes)

    def test_non_group_nodes_pass_through_around_group(self):
        sched = _make_scheduler()
        before = _make_snode(sched, _make_ir_op(), "before")
        g1 = _make_snode(sched, _make_ir_op((0,), Integer(2)), "g1")
        g2 = _make_snode(sched, _make_ir_op((0,), Integer(2)), "g2")
        after = _make_snode(sched, _make_ir_op(), "after")
        result, created = self._run([before, g1, g2, after])
        self.assertEqual(len(result), 3)
        self.assertIs(result[0], before)
        self.assertIsInstance(result[1], MagicMock)
        self.assertIs(result[2], after)
        self.assertEqual(created[0].loop_count, Integer(2))

    def test_two_separate_groups(self):
        sched = _make_scheduler()
        g0a = _make_snode(sched, _make_ir_op((0,), Integer(4)), "g0a")
        g0b = _make_snode(sched, _make_ir_op((0,), Integer(4)), "g0b")
        g1a = _make_snode(sched, _make_ir_op((1,), Integer(8)), "g1a")
        g1b = _make_snode(sched, _make_ir_op((1,), Integer(8)), "g1b")
        result, created = self._run([g0a, g0b, g1a, g1b])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(created), 2)
        self.assertEqual(created[0].loop_count, Integer(4))
        self.assertEqual(created[1].loop_count, Integer(8))

    def test_nested_group(self):
        sched = _make_scheduler()
        outer = _make_snode(sched, _make_ir_op((0,), Integer(4)), "outer")
        inner1 = _make_snode(
            sched, _make_ir_op((0, 0), [Integer(4), Integer(2)]), "inner1"
        )
        inner2 = _make_snode(
            sched, _make_ir_op((0, 0), [Integer(4), Integer(2)]), "inner2"
        )
        result, created = self._run([outer, inner1, inner2])
        self.assertEqual(len(result), 1)
        outer_loop = result[0]
        self.assertEqual(len(outer_loop.snodes), 2)
        inner_loop = outer_loop.snodes[1]
        self.assertEqual(inner_loop.loop_count, Integer(2))
        self.assertIn(inner1, inner_loop.snodes)
        self.assertIn(inner2, inner_loop.snodes)

    def test_inconsistent_loop_count_raises(self):
        sched = _make_scheduler()
        n1 = _make_snode(sched, _make_ir_op((0,), Integer(4)), "a")
        n2 = _make_snode(sched, _make_ir_op((0,), Integer(8)), "b")
        with self.assertRaises(AssertionError):
            self._run([n1, n2])

    def test_empty_list(self):
        result, created = self._run([])
        self.assertEqual(result, [])
        self.assertEqual(created, [])

    def test_symbolic_loop_count(self):
        sched = _make_scheduler()
        s = Symbol("K")
        n1 = _make_snode(sched, _make_ir_op((0,), s), "a")
        n2 = _make_snode(sched, _make_ir_op((0,), s), "b")
        result, created = self._run([n1, n2])
        self.assertEqual(len(result), 1)
        self.assertEqual(created[0].loop_count, s)


# ===========================================================================
# 4. generate_sdsc and compile_op_spec — symbol/affine-stride paths
# ===========================================================================


class TestTiledByteStride(unittest.TestCase):
    def test_fp16_one_core(self):
        s = Symbol("s")
        tensor = SDSCArgs(
            layout="A",
            dim_order=[s],
            data_format=_FP16,
            scales={s: 1},
            strides={s: 128},
            offsets={s: 0},
            max_dim_sizes={s: -1},
            allocation={"hbm": 0},
            start_address=0,
            backGap={},
        )
        stride = _tiled_byte_stride(tensor, s, {s: 64})
        self.assertEqual(stride, 64 * 128 * 2)

    def test_fp16_larger_stride(self):
        s = Symbol("s")
        tensor = SDSCArgs(
            layout="A",
            dim_order=[s],
            data_format=_FP16,
            scales={s: 1},
            strides={s: 512},
            offsets={s: 0},
            max_dim_sizes={s: -1},
            allocation={"hbm": 0},
            start_address=0,
            backGap={},
        )
        stride = _tiled_byte_stride(tensor, s, {s: 32})
        self.assertEqual(stride, 32 * 512 * 2)

    def test_stride_one(self):
        s = Symbol("s")
        tensor = SDSCArgs(
            layout="A",
            dim_order=[s],
            data_format=_FP16,
            scales={s: 1},
            strides={s: 1},
            offsets={s: 0},
            max_dim_sizes={s: -1},
            allocation={"hbm": 0},
            start_address=0,
            backGap={},
        )
        stride = _tiled_byte_stride(tensor, s, {s: 16})
        self.assertEqual(stride, 16 * 1 * 2)


class TestGenerateSdscTiledSymbols(unittest.TestCase):
    def test_tiled_tensor_affine_strides_correct(self):
        s = Symbol("s")
        sdsc_spec = _make_sdsc_spec(s, iter_range=64, device_stride=128)
        symbols: list[int] = []
        _, _, affine_strides, _ = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[s],
            use_symbols=True,
        )
        self.assertEqual(len(affine_strides), 1)
        self.assertIn(s, affine_strides[0])
        self.assertEqual(affine_strides[0][s], 64 * 128 * 2)

    def test_tiled_tensor_base_address_registered(self):
        s = Symbol("s")
        start = 0x2000
        sdsc_spec = _make_sdsc_spec(s, start_address=start)
        symbols: list[int] = []
        generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[s],
            use_symbols=True,
        )
        self.assertEqual(len(symbols), 1)
        self.assertEqual(symbols[0], start)

    def test_tiled_tensor_json_stores_symbol_id(self):
        s = Symbol("s")
        sdsc_spec = _make_sdsc_spec(s)
        symbols: list[int] = []
        sdsc_json, _, _, _ = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[s],
            use_symbols=True,
        )
        top_val = next(iter(sdsc_json.values()))
        node = top_val["dscs_"][0]["add"]["scheduleTree_"][0]
        data = node["startAddressCoreCorelet_"]["data_"]
        for v in data.values():
            self.assertLess(int(v), 0, f"Expected negative symbol ID, got {v!r}")

    def test_non_tiled_tensor_empty_affine_strides(self):
        s = Symbol("s")
        sdsc_spec = _make_sdsc_spec(s)
        symbols: list[int] = []
        _, _, affine_strides, _ = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[],
        )
        self.assertEqual(affine_strides, [{}])

    def test_lx_tensor_not_in_symbols(self):
        s = Symbol("s")
        lx_addr = 0xABC0
        sdsc_spec = _make_sdsc_spec(
            s, start_address=lx_addr, allocation={"lx": lx_addr}
        )
        symbols: list[int] = []
        _, local_sym_values, affine_strides, _ = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[s],
        )
        self.assertEqual(symbols, [])
        self.assertEqual(local_sym_values, [])
        self.assertEqual(affine_strides, [{}])

    def test_symbol_id_offset_applied(self):
        s = Symbol("s")
        sdsc_spec = _make_sdsc_spec(s)
        symbols: list[int] = []
        sdsc_json, local_sym_values, _, _ = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=5,
            tiled_symbols=[s],
            use_symbols=True,
        )
        top_val = next(iter(sdsc_json.values()))
        node = top_val["dscs_"][0]["add"]["scheduleTree_"][0]
        data = node["startAddressCoreCorelet_"]["data_"]
        ids = [int(v) for v in data.values()]
        self.assertTrue(all(i <= -6 for i in ids), f"Expected ids ≤ -6, got {ids}")

    def test_multi_core_tiled_per_core_symbols(self):
        s = Symbol("s")
        core_id = Symbol("core_id")
        tensor = SDSCArgs(
            layout="A",
            dim_order=[s],
            data_format=_FP16,
            scales={s: 1},
            strides={s: 128},
            offsets={s: 0},
            max_dim_sizes={s: -1},
            allocation={"hbm": 0x1000},
            start_address=0x1000,
            backGap={},
        )
        sdsc_spec = SDSCSpec(
            opfunc="add",
            execution_unit="sfp",
            data_format=_FP16,
            num_inputs=1,
            iteration_space={s: 32},
            num_cores=2,
            work_slices={s: 2},
            core_id_to_work_slice={s: core_id},
            padding={},
            layouts={"A": {"dim_order": [s], "stick_dim_order": s, "stick_size": 64}},
            args=[tensor],
            constants={},
            coordinate_masking={},
        )
        symbols: list[int] = []
        _, local_sym_values, affine_strides, _ = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[s],
            use_symbols=True,
        )
        self.assertEqual(len(symbols), 2)
        self.assertEqual(symbols[0], 0x1000)
        self.assertEqual(symbols[1], 0x1000 + 128)
        self.assertIn(s, affine_strides[0])


class TestCompileOpSpecTwoTiledSymbols(unittest.TestCase):
    def _make_3d_op_spec(self) -> OpSpec:
        c0 = Symbol("c0")
        c1 = Symbol("c1")
        c2 = Symbol("c2")
        fp16 = _FP16
        tensor_in = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=fp16,
            device_size=[2, 4, 64],
            device_coordinates=[c0, c1, c2],
            allocation={"hbm": 0x1000},
        )
        tensor_out = TensorArg(
            is_input=False,
            arg_index=1,
            device_dtype=fp16,
            device_size=[2, 4, 64],
            device_coordinates=[c0, c1, c2],
            allocation={"hbm": 0x2000},
        )
        return OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={
                c0: (Integer(2), 1),
                c1: (Integer(4), 1),
                c2: (Integer(64), 1),
            },
            args=[tensor_in, tensor_out],
            op_info={},
            tiled_symbols=[c0, c1],
        )

    def test_two_tiled_symbols_produce_two_stride_entries(self):
        op_spec = self._make_3d_op_spec()
        symbols: list[int] = []
        _, _, affine_strides, _ = compile_op_spec(0, op_spec, symbols, use_symbols=True)
        hbm_strides = [d for d in affine_strides if len(d) > 0]
        self.assertGreater(len(hbm_strides), 0)
        for tensor_strides in hbm_strides:
            self.assertEqual(len(tensor_strides), 2)

    def test_two_tiled_symbols_strides_are_positive(self):
        op_spec = self._make_3d_op_spec()
        symbols: list[int] = []
        _, _, affine_strides, _ = compile_op_spec(0, op_spec, symbols, use_symbols=True)
        for tensor_strides in affine_strides:
            for sym, stride in tensor_strides.items():
                self.assertGreater(stride, 0)


class TestCompileOpSpecSymbolMapping(unittest.TestCase):
    def test_affine_strides_non_empty_for_tiled_op(self):
        op_spec = _make_tiled_op_spec()
        symbols: list[int] = []
        _, _, affine_strides, _ = compile_op_spec(0, op_spec, symbols, use_symbols=True)
        has_strides = any(len(d) > 0 for d in affine_strides)
        self.assertTrue(
            has_strides,
            f"Expected non-empty affine_strides; got {affine_strides}.",
        )

    def test_generate_bundle_emits_affine_apply_for_tiled_loop(self):
        op_spec = _make_tiled_op_spec()
        loop = LoopSpec(count=Integer(4), body=[op_spec])
        tmpdir = tempfile.mkdtemp()
        generate_bundle(
            "test_kernel", tmpdir, [loop], unroll_loops=False, symbolic_args=True
        )

        with open(os.path.join(tmpdir, "bundle.mlir")) as f:
            mlir = f.read()

        self.assertIn("affine.apply", mlir)
        self.assertIn("affine_map", mlir)
        self.assertIn("scf.for", mlir)


class TestSharedWeightUnitBmmLayout(unittest.TestCase):
    def _static_bmm_custom_meta(self, x_shape, y_shape, out_shape):
        graph = fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = SimpleNamespace(shape=x_shape)
        y = graph.placeholder("y")
        y.meta["val"] = SimpleNamespace(shape=y_shape)
        bmm = graph.call_function(torch.ops.aten.bmm.default, args=(x, y))
        bmm.meta["val"] = SimpleNamespace(shape=out_shape)
        graph.output(bmm)

        bmm_unflatten_pass.apply(fx.GraphModule({}, graph).graph)
        graph.lint()
        return bmm.meta.get("custom") or {}

    def test_marked_squeezed_unit_bmm_recovers_sendnn_like_unit_layout(self):
        c0 = Symbol("c0")
        c1 = Symbol("c1")
        c2 = Symbol("c2")
        input_arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=_FP16,
            device_size=[512, 64, 1, 64],
            device_coordinates=[c0, floor(c2 / 64), Integer(0), Mod(c2, 64)],
            allocation={"hbm": 0},
            stride_map=[4096, 64, -1, 1],
        )
        kernel_arg = TensorArg(
            is_input=True,
            arg_index=1,
            device_dtype=_FP16,
            device_size=[200, 4096, 64],
            device_coordinates=[floor(c1 / 64), c2, Mod(c1, 64)],
            allocation={"hbm": 0x400000000},
            stride_map=[64, 12800, 1],
        )
        output_arg = TensorArg(
            is_input=False,
            arg_index=2,
            device_dtype=_FP16,
            device_size=[512, 200, 1, 64],
            device_coordinates=[c0, floor(c1 / 64), Integer(0), Mod(c1, 64)],
            allocation={"hbm": 0x800000000},
            stride_map=[12800, 64, -1, 1],
        )
        for arg in (input_arg, output_arg):
            del arg.device_size[-2]
            del arg.device_coordinates[-2]
            del arg.stride_map[-2]
        iteration_space = {
            c0: (Integer(512), 4),
            c1: (Integer(12800), 8),
            c2: (Integer(4096), 1),
        }
        args = [input_arg, kernel_arg, output_arg]
        op_info = {SHARED_WEIGHT_UNIT_BMM_INFO_KEY: {"batch_dim": 0}}

        iteration_space = _preserve_shared_weight_unit_bmm_dim(
            "batchmatmul", iteration_space, args, op_info
        )
        sdsc_spec, _ = parse_op_spec(
            OpSpec(
                op="batchmatmul",
                is_reduction=True,
                iteration_space=iteration_space,
                args=args,
                op_info=op_info,
            )
        )

        self.assertEqual(
            [str(dim) for dim in sdsc_spec.iteration_space],
            ["x", "mb", "out", "in"],
        )
        input_layout = sdsc_spec.layouts[sdsc_spec.args[0].layout]
        output_layout = sdsc_spec.layouts[sdsc_spec.args[-1].layout]
        self.assertEqual(
            [str(dim) for dim in input_layout["dim_order"]],
            ["mb", "in", "x"],
        )
        self.assertEqual(
            [str(dim) for dim in output_layout["dim_order"]],
            ["mb", "out", "x"],
        )

    def test_shared_weight_marker_requires_stick_aligned_dims(self):
        self.assertEqual(
            self._static_bmm_custom_meta(
                (1, 512, 4096), (1, 4096, 12800), (1, 512, 12800)
            )[SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY],
            {"batch_dim": 0},
        )
        self.assertNotIn(
            SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY,
            self._static_bmm_custom_meta(
                (4, 512, 4096), (4, 4096, 12800), (4, 512, 12800)
            ),
        )
        self.assertIsNone(
            _shared_weight_unit_bmm_info_from_sizes(
                [4, 512, 4096], [4, 4096, 12800], [4, 512, 12800]
            )
        )
        self.assertNotIn(
            SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY,
            self._static_bmm_custom_meta((1, 55, 2), (1, 2, 99), (1, 55, 99)),
        )
        self.assertIsNone(
            _shared_weight_unit_bmm_info_from_sizes([1, 55, 2], [2, 99], [1, 55, 99])
        )


# ===========================================================================
# 5. generate_bundle MLIR output
# ===========================================================================


class TestGenerateBundleMlir(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patch = patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=_fake_compile_op_spec,
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def _bundle(self, specs):
        generate_bundle("test_kernel", self.tmpdir, specs, unroll_loops=False)
        return _read_mlir(self.tmpdir)

    def test_flat_ops_no_loop(self):
        a, b = _make_minimal_op_spec("a"), _make_minimal_op_spec("b")
        mlir = self._bundle([a, b])
        self.assertIn("sdscbundle.sdsc_execute", mlir)
        self.assertNotIn("scf.for", mlir)
        self.assertNotIn("arith.constant", mlir)
        self.assertEqual(mlir.count("sdsc_execute"), 2)

    def test_single_loop_emits_scf_for(self):
        a, b = _make_minimal_op_spec("a"), _make_minimal_op_spec("b")
        loop = LoopSpec(count=Integer(4), body=[a, b])
        mlir = self._bundle([loop])
        self.assertIn("scf.for", mlir)
        self.assertIn("arith.constant 4 : index", mlir)
        self.assertIn("%c0", mlir)
        self.assertIn("%c1", mlir)
        self.assertEqual(mlir.count("sdsc_execute"), 2)

    def test_single_loop_structure(self):
        a = _make_minimal_op_spec("a")
        loop = LoopSpec(count=Integer(3), body=[a])
        mlir = self._bundle([loop])
        for_pos = mlir.index("scf.for")
        exec_pos = mlir.index("sdsc_execute")
        close_pos = mlir.rindex("}")
        self.assertLess(for_pos, exec_pos)
        self.assertLess(exec_pos, close_pos)

    def test_flat_op_before_and_after_loop(self):
        before = _make_minimal_op_spec("before")
        body = _make_minimal_op_spec("body")
        after = _make_minimal_op_spec("after")
        loop = LoopSpec(count=Integer(2), body=[body])
        mlir = self._bundle([before, loop, after])
        self.assertIn("scf.for", mlir)
        self.assertEqual(mlir.count("sdsc_execute"), 3)

    def test_nested_loops(self):
        a = _make_minimal_op_spec("a")
        b = _make_minimal_op_spec("b")
        inner = LoopSpec(count=Integer(2), body=[b])
        outer = LoopSpec(count=Integer(4), body=[a, inner])
        mlir = self._bundle([outer])
        self.assertEqual(mlir.count("scf.for"), 2)
        self.assertIn("arith.constant 4 : index", mlir)
        self.assertIn("arith.constant 2 : index", mlir)
        self.assertEqual(mlir.count("sdsc_execute"), 2)
        outer_pos = mlir.index("scf.for")
        inner_pos = mlir.index("scf.for", outer_pos + 1)
        self.assertLess(outer_pos, inner_pos)

    def test_sdsc_json_files_written_depth_first(self):
        a = _make_minimal_op_spec("a")
        b = _make_minimal_op_spec("b")
        loop = LoopSpec(count=Integer(2), body=[a, b])
        generate_bundle("test_kernel", self.tmpdir, [loop], unroll_loops=False)
        written = sorted(f for f in os.listdir(self.tmpdir) if f.endswith(".json"))
        self.assertEqual(len(written), 2)

    def test_empty_specs_writes_minimal_bundle(self):
        mlir = self._bundle([])
        self.assertIn("func.func @sdsc_bundle", mlir)
        self.assertIn("return", mlir)
        self.assertNotIn("sdsc_execute", mlir)
        self.assertNotIn("scf.for", mlir)

    def test_symbolic_count_raises(self):
        k = Symbol("K")
        a = _make_minimal_op_spec("a")
        loop = LoopSpec(count=k, body=[a])
        with self.assertRaises(NotImplementedError):
            self._bundle([loop])


class TestFindUnimplemented(unittest.TestCase):
    def test_no_unimplemented(self):
        from torch_spyre._inductor.op_spec import find_unimplemented

        a = _make_minimal_op_spec("a")
        self.assertIsNone(find_unimplemented([a]))

    def test_flat_unimplemented(self):
        from torch_spyre._inductor.op_spec import find_unimplemented

        unimp = UnimplementedOp(op="missing")
        a = _make_minimal_op_spec("a")
        result = find_unimplemented([a, unimp])
        self.assertIs(result, unimp)

    def test_unimplemented_inside_loop(self):
        from torch_spyre._inductor.op_spec import find_unimplemented

        unimp = UnimplementedOp(op="missing")
        loop = LoopSpec(count=Integer(4), body=[unimp])
        result = find_unimplemented([loop])
        self.assertIs(result, unimp)

    def test_unimplemented_in_nested_loop(self):
        from torch_spyre._inductor.op_spec import find_unimplemented

        unimp = UnimplementedOp(op="missing")
        inner = LoopSpec(count=Integer(2), body=[unimp])
        outer = LoopSpec(count=Integer(4), body=[inner])
        result = find_unimplemented([outer])
        self.assertIs(result, unimp)

    def test_returns_first_found(self):
        from torch_spyre._inductor.op_spec import find_unimplemented

        u1 = UnimplementedOp(op="first")
        u2 = UnimplementedOp(op="second")
        result = find_unimplemented([u1, u2])
        self.assertIs(result, u1)


class TestGenerateBundleMlirSnapshot(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.patch = patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=_fake_compile_op_spec,
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def _bundle(self, specs):
        generate_bundle("test_kernel", self.tmpdir, specs, unroll_loops=False)
        return _read_mlir(self.tmpdir)

    def test_single_loop_snapshot(self):
        a = _make_minimal_op_spec("a")
        loop = LoopSpec(count=Integer(8), body=[a])
        mlir = self._bundle([loop])
        expected = (
            "module {\n"
            "\tfunc.func @sdsc_bundle() {\n"
            "\t\t%c0 = arith.constant 0 : index\n"
            "\t\t%c1 = arith.constant 1 : index\n"
            "\t\t%loop_bound_0 = arith.constant 8 : index\n"
            "\t\tscf.for %i_0 = %c0 to %loop_bound_0 step %c1 {\n"
            '\t\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json"}\n'
            "\t\t}\n"
            "\t\treturn\n"
            "\t}\n"
            "}\n"
        )
        self.assertEqual(mlir, expected)

    def test_flat_snapshot(self):
        a = _make_minimal_op_spec("a")
        mlir = self._bundle([a])
        expected = (
            "module {\n"
            "\tfunc.func @sdsc_bundle() {\n"
            '\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json"}\n'
            "\t\treturn\n"
            "\t}\n"
            "}\n"
        )
        self.assertEqual(mlir, expected)


class TestGenerateBundleMlirWithAffineStrides(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._s = Symbol("s")

    def _bundle(self, specs, fake_compile):
        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=fake_compile,
        ):
            generate_bundle(
                "test_kernel",
                self.tmpdir,
                specs,
                unroll_loops=False,
                symbolic_args=True,
            )
        return _read_mlir(self.tmpdir)

    def test_tiled_tensor_emits_affine_apply(self):
        s = self._s
        stride = 16384

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x1000)
            return _make_tiled_json(idx, sym_id), [0x1000], [{s: stride}], []

        op = _make_minimal_op_spec("a")
        op.tiled_symbols = [s]
        loop = LoopSpec(count=Integer(4), body=[op])
        mlir = self._bundle([loop], fake_compile)

        self.assertIn("affine_map", mlir)
        self.assertIn(str(stride), mlir)
        self.assertIn("affine.apply", mlir)
        self.assertIn("%addr_0", mlir)
        self.assertIn(
            'sdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_0.json"', mlir
        )
        self.assertIn('"symbol_ids"=[-1]', mlir)

    def test_non_tiled_tensor_in_loop_no_affine_apply(self):
        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x2000)
            return _make_tiled_json(idx, sym_id), [0x2000], [{}], []

        op = _make_minimal_op_spec("b")
        loop = LoopSpec(count=Integer(2), body=[op])
        mlir = self._bundle([loop], fake_compile)

        self.assertNotIn("affine.apply", mlir)
        self.assertNotIn("affine_map", mlir)
        self.assertIn("%sym_1", mlir)
        self.assertIn("sdscbundle.sdsc_execute (%sym_1)", mlir)

    def test_affine_map_stride_at_module_level(self):
        s = self._s
        stride = 8192

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x3000)
            return _make_tiled_json(idx, sym_id), [0x3000], [{s: stride}], []

        op = _make_minimal_op_spec("c")
        op.tiled_symbols = [s]
        loop = LoopSpec(count=Integer(4), body=[op])
        mlir = self._bundle([loop], fake_compile)

        map_pos = mlir.index("affine_map")
        module_pos = mlir.index("module {")
        self.assertLess(map_pos, module_pos)

    def test_affine_apply_inside_scf_for(self):
        s = self._s

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x4000)
            return _make_tiled_json(idx, sym_id), [0x4000], [{s: 512}], []

        op = _make_minimal_op_spec("d")
        op.tiled_symbols = [s]
        loop = LoopSpec(count=Integer(4), body=[op])
        mlir = self._bundle([loop], fake_compile)

        for_pos = mlir.index("scf.for")
        apply_pos = mlir.index("affine.apply")
        execute_pos = mlir.index("sdsc_execute")
        self.assertLess(for_pos, apply_pos)
        self.assertLess(apply_pos, execute_pos)

    def test_tiled_snapshot(self):
        s = self._s

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x1000)
            return _make_tiled_json(idx, sym_id), [0x1000], [{s: 256}], []

        op = _make_minimal_op_spec("a")
        op.tiled_symbols = [s]
        loop = LoopSpec(count=Integer(4), body=[op])
        mlir = self._bundle([loop], fake_compile)

        expected = (
            "#map_0 = affine_map<(d0)[s0] -> (s0 + 256*d0)>\n"
            "module {\n"
            "\tfunc.func @sdsc_bundle() {\n"
            "\t\t%c0 = arith.constant 0 : index\n"
            "\t\t%c1 = arith.constant 1 : index\n"
            "\t\t%loop_bound_0 = arith.constant 4 : index\n"
            "\t\t%sym_1 = arith.constant 4096 : index\n"
            "\t\tscf.for %i_0 = %c0 to %loop_bound_0 step %c1 {\n"
            "\t\t\t%addr_0 = affine.apply #map_0(%i_0)[%sym_1]\n"
            '\t\t\tsdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_0.json",'
            ' "symbol_ids"=[-1]}\n'
            "\t\t}\n"
            "\t\treturn\n"
            "\t}\n"
            "}\n"
        )
        self.assertEqual(mlir, expected)


class TestGenerateBundleNestedTiling(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.s0 = Symbol("s0")
        self.s1 = Symbol("s1")

    def _bundle(self, specs, fake_compile):
        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=fake_compile,
        ):
            generate_bundle(
                "test_kernel",
                self.tmpdir,
                specs,
                unroll_loops=False,
                symbolic_args=True,
            )
        return _read_mlir(self.tmpdir)

    def _fake_compile_two_strides(self, outer_stride, inner_stride):
        s0, s1 = self.s0, self.s1

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x1000)
            return (
                _make_tiled_json(idx, sym_id),
                [0x1000],
                [{s0: outer_stride, s1: inner_stride}],
                [],
            )

        return fake_compile

    def test_nested_loop_emits_two_scf_for(self):
        op = _make_minimal_op_spec("add")
        inner = LoopSpec(count=Integer(2), body=[op])
        outer = LoopSpec(count=Integer(4), body=[inner])
        mlir = self._bundle(
            [outer], self._fake_compile_two_strides(outer_stride=512, inner_stride=64)
        )
        self.assertEqual(mlir.count("scf.for"), 2)

    def test_nested_tiling_emits_2d_affine_map(self):
        op = _make_minimal_op_spec("add")
        inner = LoopSpec(count=Integer(2), body=[op])
        outer = LoopSpec(count=Integer(4), body=[inner])
        mlir = self._bundle(
            [outer], self._fake_compile_two_strides(outer_stride=512, inner_stride=64)
        )
        self.assertIn("affine_map<(d0, d1)[s0]", mlir)
        self.assertIn("512*d0", mlir)
        self.assertIn("64*d1", mlir)

    def test_nested_tiling_affine_apply_uses_both_loop_vars(self):
        op = _make_minimal_op_spec("add")
        inner = LoopSpec(count=Integer(2), body=[op])
        outer = LoopSpec(count=Integer(4), body=[inner])
        mlir = self._bundle(
            [outer], self._fake_compile_two_strides(outer_stride=512, inner_stride=64)
        )
        self.assertIn("affine.apply", mlir)
        apply_line = next(line for line in mlir.splitlines() if "affine.apply" in line)
        self.assertIn("%i_0", apply_line)
        self.assertIn("%i_1", apply_line)

    def test_nested_tiling_snapshot(self):
        op = _make_minimal_op_spec("add")
        inner = LoopSpec(count=Integer(2), body=[op])
        outer = LoopSpec(count=Integer(4), body=[inner])
        mlir = self._bundle(
            [outer], self._fake_compile_two_strides(outer_stride=512, inner_stride=64)
        )
        expected = (
            "#map_0 = affine_map<(d0, d1)[s0] -> (s0 + 512*d0 + 64*d1)>\n"
            "module {\n"
            "\tfunc.func @sdsc_bundle() {\n"
            "\t\t%c0 = arith.constant 0 : index\n"
            "\t\t%c1 = arith.constant 1 : index\n"
            "\t\t%loop_bound_0 = arith.constant 4 : index\n"
            "\t\t%loop_bound_1 = arith.constant 2 : index\n"
            "\t\t%sym_1 = arith.constant 4096 : index\n"
            "\t\tscf.for %i_0 = %c0 to %loop_bound_0 step %c1 {\n"
            "\t\t\tscf.for %i_1 = %c0 to %loop_bound_1 step %c1 {\n"
            "\t\t\t\t%addr_0 = affine.apply #map_0(%i_0, %i_1)[%sym_1]\n"
            '\t\t\t\tsdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_0.json",'
            ' "symbol_ids"=[-1]}\n'
            "\t\t\t}\n"
            "\t\t}\n"
            "\t\treturn\n"
            "\t}\n"
            "}\n"
        )
        self.assertEqual(mlir, expected)


# ===========================================================================
# 6. coarse_tile buffer propagation pass
# ===========================================================================


def _make_rw_with_reads(*names):
    """Return a fake ReadWrites whose reads set contains MemoryDep mocks for names."""
    from torch._inductor.dependencies import MemoryDep

    reads = []
    for name in names:
        dep = MagicMock(spec=MemoryDep)
        dep.name = name
        reads.append(dep)
    rw = MagicMock()
    rw.reads = reads
    return rw


def _make_tiled_op(name, ranges, loop_group_id, loop_count, loop_tiled_dims):
    """Return a ComputedBuffer mock that looks like a stamped tiled Pointwise op."""
    from torch._inductor.ir import ComputedBuffer, Pointwise

    data = MagicMock(spec=Pointwise)
    data.ranges = list(ranges)

    op = MagicMock(spec=ComputedBuffer)
    op.data = data
    op.get_operation_name.return_value = name
    op.get_name.return_value = name
    op.loop_info = CoarseTileInfo(
        loop_group_id=loop_group_id,
        loop_count=list(loop_count),
        loop_tiled_dims=[list(d) for d in loop_tiled_dims],
    )
    op.get_read_writes.return_value = _make_rw_with_reads()
    op.origins = OrderedSet()
    return op


def _make_consumer_op(name, reads_buf):
    """Return a ComputedBuffer mock that reads reads_buf, with no loop_group_id."""
    from torch._inductor.ir import ComputedBuffer, Pointwise

    data = MagicMock(spec=Pointwise)
    data.ranges = [Integer(64)]
    data.inner_fn = MagicMock()

    op = MagicMock(spec=ComputedBuffer)
    op.data = data
    op.get_operation_name.return_value = name
    op.get_name.return_value = name
    del op.loop_info
    op.get_read_writes.return_value = _make_rw_with_reads(reads_buf)
    op.origins = OrderedSet()
    return op


def _make_inside_consumer_op(name, reads_buf, loop_group_id):
    """Return a ComputedBuffer mock inside the same loop group that reads reads_buf."""
    from torch._inductor.ir import ComputedBuffer, Pointwise

    data = MagicMock(spec=Pointwise)
    data.ranges = [Integer(16)]
    data.inner_fn = MagicMock()

    op = MagicMock(spec=ComputedBuffer)
    op.data = data
    op.get_operation_name.return_value = name
    op.get_name.return_value = name
    op.loop_info = CoarseTileInfo(
        loop_group_id=loop_group_id,
        loop_count=[Integer(4)],
        loop_tiled_dims=[[0]],
    )
    op.get_read_writes.return_value = _make_rw_with_reads(reads_buf)
    op.origins = OrderedSet()
    return op


class TestCoarseTileBufferPropagation(unittest.TestCase):
    """Tests for insert_tiling_propagation — consumer analysis helpers."""

    # ------------------------------------------------------------------
    # Tests for _find_outside_consumers and _has_inside_consumers
    # (these helpers don't call V.graph, so no mocking needed)
    # ------------------------------------------------------------------

    def test_no_consumers_returns_empty(self):
        from torch_spyre._inductor.coarse_tile import _find_outside_consumers

        op = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        with patch(
            "torch_spyre._inductor.coarse_tile._graph_output_names",
            return_value=set(),
        ):
            consumers, is_out = _find_outside_consumers("op0", (0,), [op])
        self.assertEqual(consumers, [])
        self.assertFalse(is_out)

    def test_outside_consumer_detected(self):
        from torch_spyre._inductor.coarse_tile import _find_outside_consumers

        tiled = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        consumer = _make_consumer_op("out0", "op0")  # no loop_group_id → outside
        with patch(
            "torch_spyre._inductor.coarse_tile._graph_output_names",
            return_value=set(),
        ):
            consumers, is_out = _find_outside_consumers("op0", (0,), [tiled, consumer])
        self.assertEqual(consumers, [consumer])
        self.assertFalse(is_out)

    def test_graph_output_detected(self):
        from torch_spyre._inductor.coarse_tile import _find_outside_consumers

        tiled = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        with patch(
            "torch_spyre._inductor.coarse_tile._graph_output_names",
            return_value={"op0"},
        ):
            consumers, is_out = _find_outside_consumers("op0", (0,), [tiled])
        self.assertEqual(consumers, [])
        self.assertTrue(is_out)

    def test_inside_consumer_not_counted_as_outside(self):
        from torch_spyre._inductor.coarse_tile import _find_outside_consumers

        tiled = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        inside = _make_inside_consumer_op("op1", "op0", (0,))
        with patch(
            "torch_spyre._inductor.coarse_tile._graph_output_names",
            return_value=set(),
        ):
            consumers, is_out = _find_outside_consumers("op0", (0,), [tiled, inside])
        self.assertEqual(consumers, [])
        self.assertFalse(is_out)

    def test_has_inside_consumer_true(self):
        from torch_spyre._inductor.coarse_tile import _has_inside_consumers

        tiled = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        inside = _make_inside_consumer_op("op1", "op0", (0,))
        result = _has_inside_consumers("op0", (0,), [tiled, inside])
        self.assertTrue(result)

    def test_has_inside_consumer_false_when_only_outside(self):
        from torch_spyre._inductor.coarse_tile import _has_inside_consumers

        tiled = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        outside = _make_consumer_op("out0", "op0")
        result = _has_inside_consumers("op0", (0,), [tiled, outside])
        self.assertFalse(result)

    def test_compute_full_ranges_flat(self):
        from torch_spyre._inductor.coarse_tile import _compute_full_ranges

        op = _make_tiled_op("op0", [Integer(16), Integer(8)], (0,), [Integer(4)], [[0]])
        full = _compute_full_ranges(op)
        # dim 0: 16 * 4 = 64; dim 1 untiled: 8
        self.assertEqual(full[0], Integer(64))
        self.assertEqual(full[1], Integer(8))

    def test_compute_full_ranges_nested(self):
        from torch_spyre._inductor.coarse_tile import _compute_full_ranges

        # Nested: outer K=4 tiles dim 0, inner K=2 tiles dim 1
        op = _make_tiled_op(
            "op0",
            [Integer(16), Integer(32)],
            (0, 0),
            [Integer(4), Integer(2)],
            [[0], [1]],
        )
        full = _compute_full_ranges(op)
        # dim 0: 16 * 4 = 64; dim 1: 32 * 2 = 64
        self.assertEqual(full[0], Integer(64))
        self.assertEqual(full[1], Integer(64))

    def test_different_loop_group_id_is_outside(self):
        """Op in loop group 1 should be seen as outside consumer of group 0."""
        from torch_spyre._inductor.coarse_tile import _find_outside_consumers

        tiled = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        other_group = _make_tiled_op("op1", [Integer(16)], (1,), [Integer(4)], [[0]])
        # Make op1 read op0
        other_group.get_read_writes.return_value = _make_rw_with_reads("op0")
        with patch(
            "torch_spyre._inductor.coarse_tile._graph_output_names",
            return_value=set(),
        ):
            consumers, _ = _find_outside_consumers("op0", (0,), [tiled, other_group])
        self.assertEqual(consumers, [other_group])


def _make_tiled_reduction_op(
    name,
    ranges,
    reduction_ranges,
    reduction_type,
    loop_group_id,
    loop_count,
    loop_tiled_dims,
):
    """Return a ComputedBuffer mock that looks like a stamped tiled Reduction op."""
    from torch._inductor.ir import ComputedBuffer, Reduction

    data = MagicMock(spec=Reduction)
    data.ranges = list(ranges)
    data.reduction_ranges = list(reduction_ranges)
    data.reduction_type = reduction_type

    op = MagicMock(spec=ComputedBuffer)
    op.data = data
    op.get_operation_name.return_value = name
    op.get_name.return_value = name
    op.loop_info = CoarseTileInfo(
        loop_group_id=loop_group_id,
        loop_count=list(loop_count),
        loop_tiled_dims=[list(d) for d in loop_tiled_dims],
    )
    op.get_read_writes.return_value = _make_rw_with_reads()
    op.origins = OrderedSet()
    return op


class TestCoarseTileReductionPropagation(unittest.TestCase):
    """Tests for insert_tiling_propagation Reduction support."""

    def test_reduction_tiled_reduction_dim_raises(self):
        from torch_spyre._inductor.coarse_tile import _check_reduction_tiling_safety

        # ranges=[M], reduction_ranges=[K]; tiled_dim=1 is >= len(ranges)=1 → reduction dim
        op = _make_tiled_reduction_op(
            "red0",
            ranges=[Integer(128)],
            reduction_ranges=[Integer(256)],
            reduction_type="sum",
            loop_group_id=(0,),
            loop_count=[Integer(4)],
            loop_tiled_dims=[[1]],
        )
        with self.assertRaises(RuntimeError, msg="tiled reduction dim should raise"):
            _check_reduction_tiling_safety(op)

    def test_reduction_output_dim_tiled_ok(self):
        from torch_spyre._inductor.coarse_tile import _check_reduction_tiling_safety

        # ranges=[M], reduction_ranges=[K]; tiled_dim=0 is an output dim → no error
        op = _make_tiled_reduction_op(
            "red0",
            ranges=[Integer(128)],
            reduction_ranges=[Integer(64)],
            reduction_type="sum",
            loop_group_id=(0,),
            loop_count=[Integer(4)],
            loop_tiled_dims=[[0]],
        )
        # Should not raise
        _check_reduction_tiling_safety(op)


class TestTiledSymsForSchedNode(unittest.TestCase):
    """Regression test for _tiled_syms_for_sched_node_at_depth.

    loop_tiled_dims stores host-range indices (e.g. 1 for H in [B=1,H,Lq,D])
    but the iteration space skips unit-size dims (B=1 dropped), so H is at
    iteration-space index 0.  The function must map between the two.
    """

    def test_unit_batch_dim_skipped(self):
        """[B=1,H=8,Lq=256,D=64] with loop_tiled_dims=[[1]] must return H (c0).

        Without the fix, index 1 is used directly and returns c1 (Lq) instead.
        """
        from torch_spyre._inductor.scheduler import _tiled_syms_for_sched_node_at_depth
        from torch._inductor.scheduler import SchedulerNode

        host_ranges = [1, 8, 256, 64]
        non_unit = [r for r in host_ranges if r != 1]
        it_syms = [Symbol(f"c{i}") for i in range(len(non_unit))]
        it_space = {s: Integer(r) for s, r in zip(it_syms, non_unit)}

        ir_op = MagicMock()
        ir_op.data.ranges = [Integer(r) for r in host_ranges]
        ir_op.loop_info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(4)],
            loop_tiled_dims=[[1]],
        )

        snode = MagicMock(spec=SchedulerNode)
        snode.node = ir_op

        with patch(
            "torch_spyre._inductor.scheduler.iteration_space",
            return_value=it_space,
        ):
            result = _tiled_syms_for_sched_node_at_depth(snode, 0)

        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0]), "c0")  # H, not c1 (Lq)


class TestGenerateBundleMlirSymbolicArgs(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _bundle(self, specs, symbolic_args=False, fake_compile=None):
        if fake_compile is None:
            fake_compile = _fake_compile_op_spec
        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=fake_compile,
        ):
            generate_bundle(
                "test_kernel",
                self.tmpdir,
                specs,
                unroll_loops=False,
                symbolic_args=symbolic_args,
            )
        return _read_mlir(self.tmpdir)

    def _make_op_spec_with_hbm_args(self, name: str, arg_indices: list) -> OpSpec:
        """Minimal OpSpec whose TensorArgs have the given arg_indices and hbm allocation."""
        c0 = Symbol("c0")
        args = [
            TensorArg(
                is_input=(i == 0),
                arg_index=idx,
                device_dtype=_FP16,
                device_size=[2, 64],
                device_coordinates=[Integer(0), c0],
                allocation={"hbm": 0x400000000 * (idx + 1)},
            )
            for i, idx in enumerate(arg_indices)
        ]
        return OpSpec(
            op=name,
            is_reduction=False,
            iteration_space={c0: (Integer(128), 1)},
            args=args,
            op_info={},
        )

    def test_signature_accepts_symbolic_args_param(self):
        a = _make_minimal_op_spec("a")
        mlir = self._bundle([a], symbolic_args=False)
        self.assertIn("sdsc_execute", mlir)

    def test_func_signature_has_params_for_tensor_args(self):
        a = self._make_op_spec_with_hbm_args("a", [0, 1])

        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            for i, arg in enumerate(op_spec.args):
                symbols.append(arg.allocation["hbm"])
            ids = [-(symbol_id_offset + i + 1) for i in range(len(op_spec.args))]
            json_out = {
                f"{idx}_{op_spec.op}": {
                    "numCoresUsed_": 1,
                    "dscs_": [
                        {
                            "op": {
                                "scheduleTree_": [
                                    {
                                        "component_": "hbm",
                                        "startAddressCoreCorelet_": {
                                            "data_": {"[0, 0, 0]": str(ids[j])}
                                        },
                                    }
                                    for j in range(len(op_spec.args))
                                ]
                            }
                        }
                    ],
                }
            }
            return (
                json_out,
                [arg.allocation["hbm"] for arg in op_spec.args],
                [{} for _ in op_spec.args],
                [SymbolKind.kernel(arg.arg_index) for arg in op_spec.args],
            )

        mlir = self._bundle([a], symbolic_args=True, fake_compile=fake)

        self.assertIn(
            "func.func @sdsc_bundle("
            "%arg_0_base_addr: !sdscbundle.input_arg<index>,"
            " %arg_1_base_addr: !sdscbundle.input_arg<index>)",
            mlir,
        )
        self.assertIn(
            "%arg_0 = sdscbundle.input_arg_extract value from"
            " %arg_0_base_addr : !sdscbundle.input_arg<index> -> index",
            mlir,
        )
        self.assertIn(
            "%arg_1 = sdscbundle.input_arg_extract value from"
            " %arg_1_base_addr : !sdscbundle.input_arg<index> -> index",
            mlir,
        )
        self.assertNotIn("arith.constant 17179869184", mlir)
        self.assertNotIn("arith.constant 34359738368", mlir)

    def test_sdsc_execute_uses_extracted_names(self):
        a = self._make_op_spec_with_hbm_args("a", [0])

        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(op_spec.args[0].allocation["hbm"])
            return (
                _make_tiled_json(idx, sym_id),
                [op_spec.args[0].allocation["hbm"]],
                [{}],
                [SymbolKind.kernel(0)],
            )

        mlir = self._bundle([a], symbolic_args=True, fake_compile=fake)

        self.assertIn("sdscbundle.sdsc_execute (%arg_0)", mlir)
        self.assertNotIn("sdsc_execute (%sym_0_1)", mlir)
        self.assertNotIn("sdsc_execute (%sym_1)", mlir)

    def test_non_tensor_arg_symbols_remain_as_constants(self):
        c0 = Symbol("c0")
        op_a = self._make_op_spec_with_hbm_args("a", [0])
        # op_b: arg_index=-1, pool-allocated (fake returns "pool" kind)
        op_b = OpSpec(
            op="b",
            is_reduction=False,
            iteration_space={c0: (Integer(128), 1)},
            args=[
                TensorArg(
                    is_input=True,
                    arg_index=-1,
                    device_dtype=_FP16,
                    device_size=[2, 64],
                    device_coordinates=[Integer(0), c0],
                    allocation={"hbm": 0x0},
                )
            ],
            op_info={},
        )
        call_count = [0]
        values = [0x400000000, 0x0]

        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            i = call_count[0]
            call_count[0] += 1
            sym_id = -(symbol_id_offset + 1)
            symbols.append(values[i])
            kind = (
                SymbolKind.kernel(0) if i == 0 else SymbolKind.pool()
            )  # op_b has pool allocation
            return _make_tiled_json(idx, sym_id), [values[i]], [{}], [kind]

        mlir = self._bundle([op_a, op_b], symbolic_args=True, fake_compile=fake)

        # First sym → parameter (kernel tensor arg)
        self.assertIn("%arg_0_base_addr: !sdscbundle.input_arg<index>", mlir)
        self.assertNotIn("arith.constant 17179869184", mlir)
        # Second sym → pool: arith.addi %pool, <offset>
        self.assertIn("%pool_base_addr: !sdscbundle.input_arg<index>", mlir)
        self.assertIn("%pool_addr_0 = arith.addi %pool", mlir)

    def test_symbolic_args_false_no_params(self):
        a = self._make_op_spec_with_hbm_args("a", [0])
        # When symbolic_args=False, use_symbols=False: no symbols registered,
        # sdsc_execute has no operands.
        mlir = self._bundle([a], symbolic_args=False)
        self.assertIn("func.func @sdsc_bundle()", mlir)
        self.assertNotIn("input_arg", mlir)
        self.assertNotIn("%sym_", mlir)
        self.assertIn("sdsc_execute () {sdsc_filename=", mlir)

    def test_multi_sdsc_two_tensor_args_snapshot(self):
        """Two tensor args on first op; remaining ops use arith.constant symbols."""
        op0 = self._make_op_spec_with_hbm_args("op0", [0, 1])
        ops_rest = [_make_minimal_op_spec(f"op{i}") for i in range(1, 5)]
        call_count = [0]
        # sym values: first two are tensor args, rest are intermediates
        sym_values = [
            0x400000000,
            0x800000000,  # op0: tensor args
            0x0,
            0x400000000,
            0x800000000,  # op1
            0x800000000,
            0xC00000000,  # op2
            0xC00000000,
            0x1000000000,  # op3
            0xC00000000,
            0x1000000000,
            0x1400000000,  # op4
        ]
        sym_counts = [2, 3, 2, 2, 3]

        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            i = call_count[0]
            call_count[0] += 1
            n = sym_counts[i]
            start = sum(sym_counts[:i])
            local_ids = [-(symbol_id_offset + j + 1) for j in range(n)]
            for v in sym_values[start : start + n]:
                symbols.append(v)
            json_out = {
                f"{idx}_{op_spec.op}": {
                    "numCoresUsed_": 1,
                    "dscs_": [
                        {
                            "op": {
                                "scheduleTree_": [
                                    {
                                        "component_": "hbm",
                                        "startAddressCoreCorelet_": {
                                            "data_": {"[0, 0, 0]": str(local_ids[j])}
                                        },
                                    }
                                    for j in range(n)
                                ]
                            }
                        }
                    ],
                }
            }
            # All symbols are kernel args; use the running symbol index as arg_index
            # so each unique value produces a distinct input_arg param.
            sym_start = sum(sym_counts[:i])
            symbol_kind_flags = [SymbolKind.kernel(sym_start + j) for j in range(n)]
            return (
                json_out,
                sym_values[start : start + n],
                [{} for _ in range(n)],
                symbol_kind_flags,
            )

        mlir = self._bundle([op0] + ops_rest, symbolic_args=True, fake_compile=fake)

        # 12 symbols with 6 unique values → 6 unique params
        # Param names derive from arg_index (= symbol position in sym_values list)
        self.assertIn("%arg_0_base_addr: !sdscbundle.input_arg<index>", mlir)
        self.assertIn("%arg_1_base_addr: !sdscbundle.input_arg<index>", mlir)
        # There are exactly 6 input_arg params (each appears twice: param + extract)
        self.assertEqual(mlir.count("!sdscbundle.input_arg<index>"), 6 * 2)
        # First sdsc_execute uses first two extracted names
        self.assertIn("sdscbundle.sdsc_execute (%arg_0, %arg_1)", mlir)
        # Duplicate addresses reuse existing extracted SSA names
        self.assertNotIn("arith.constant", mlir)
        self.assertNotIn("%pool:", mlir)

    def test_same_kernel_arg_across_sdsc_deduped(self):
        """The same kernel arg address appearing in two SDSCs maps to one input_arg param."""
        # Simulates softmax: arg_index=0 appears in both op0 and op1.
        a = _make_minimal_op_spec("a")
        b = _make_minimal_op_spec("b")
        base = 0x400000000  # SEGMENT_OFFSETS[1], arg_index=0
        call_count = [0]

        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            call_count[0] += 1
            sym_id = -(symbol_id_offset + 1)
            symbols.append(base)
            return _make_tiled_json(idx, sym_id), [base], [{}], [SymbolKind.kernel(0)]

        mlir = self._bundle([a, b], symbolic_args=True, fake_compile=fake)

        # Only one input_arg param (deduped cross-SDSC)
        self.assertIn("%arg_0_base_addr: !sdscbundle.input_arg<index>", mlir)
        self.assertNotIn("%sym_0_2:", mlir)
        # Both sdsc_execute ops reference the same extracted name
        execute_lines = [ln for ln in mlir.splitlines() if "sdsc_execute" in ln]
        self.assertEqual(execute_lines[0].split("(")[1].split(")")[0], "%arg_0")
        self.assertEqual(execute_lines[1].split("(")[1].split(")")[0], "%arg_0")

    def test_pool_offset_constants_deduped(self):
        """Pool symbols with the same offset share one arith.addi SSA variable."""
        # Three pool symbols: offsets 0, 2048, 0.
        # Expected: 2 arith.constant + 2 arith.addi; sdsc_execute for op[2] reuses %sym_1.
        a = _make_minimal_op_spec("a")
        b = _make_minimal_op_spec("b")
        c = _make_minimal_op_spec("c")
        call_count = [0]
        pool_values = [0, 2048, 0]

        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            i = call_count[0]
            call_count[0] += 1
            sym_id = -(symbol_id_offset + 1)
            symbols.append(pool_values[i])
            return (
                _make_tiled_json(idx, sym_id),
                [pool_values[i]],
                [{}],
                [SymbolKind.pool()],
            )

        mlir = self._bundle([a, b, c], symbolic_args=True, fake_compile=fake)

        # Exactly two arith.constant / arith.addi pairs (offsets 0 and 2048)
        self.assertEqual(mlir.count("arith.constant 0 : index"), 1)
        self.assertEqual(mlir.count("arith.constant 2048 : index"), 1)
        self.assertEqual(mlir.count("arith.addi %pool"), 2)
        # op[0] and op[2] both use %pool_addr_0; op[1] uses %pool_addr_2048
        self.assertIn("sdscbundle.sdsc_execute (%pool_addr_0)", mlir)
        self.assertIn("sdscbundle.sdsc_execute (%pool_addr_2048)", mlir)
        execute_lines = [ln for ln in mlir.splitlines() if "sdsc_execute" in ln]
        self.assertEqual(execute_lines[0].split("(")[1].split(")")[0], "%pool_addr_0")
        self.assertEqual(
            execute_lines[1].split("(")[1].split(")")[0], "%pool_addr_2048"
        )
        self.assertEqual(execute_lines[2].split("(")[1].split(")")[0], "%pool_addr_0")


class TestSymbolKind(unittest.TestCase):
    """Unit tests for the SymbolKind dataclass."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_import(self):
        _ = SymbolKind  # importable as a top-level name

    def test_kernel_base_kind(self):
        sk = SymbolKind.kernel(0)
        self.assertEqual(sk.kind, "kernel")
        self.assertFalse(sk.is_derived)
        self.assertFalse(sk.is_pool)

    def test_kernel_derived_kind_carries_base_index_and_offset(self):
        sk = SymbolKind.kernel_derived(base_sym_idx=3, offset=512, arg_index=0)
        self.assertEqual(sk.kind, "kernel_derived")
        self.assertEqual(sk.base_sym_idx, 3)
        self.assertEqual(sk.offset, 512)
        self.assertTrue(sk.is_derived)
        self.assertFalse(sk.is_pool)

    def test_pool_kind(self):
        sk = SymbolKind.pool()
        self.assertEqual(sk.kind, "pool")
        self.assertFalse(sk.is_derived)
        self.assertTrue(sk.is_pool)

    def test_generate_sdsc_two_cores_emits_kernel_derived_with_base_idx(self):
        """With num_cores=2, the second per-core tiled symbol should be kernel_derived
        and carry the index of the first (kernel base) symbol."""

        s = Symbol("s")
        core_id = Symbol("core_id")
        from sympy import Mod

        # Mirror the existing TestGenerateSdscTiledSymbols multi-core test but
        # with arg_index=0 to exercise the kernel/kernel_derived kind path.
        tensor = SDSCArgs(
            layout="A",
            dim_order=[s],
            data_format=_FP16,
            scales={s: 1},
            strides={s: 128},
            offsets={s: 0},
            max_dim_sizes={s: -1},
            allocation={"hbm": 0x1000},
            start_address=0x1000,
            backGap={},
            arg_index=0,  # kernel arg → kinds should be kernel + kernel_derived
        )
        sdsc_spec = SDSCSpec(
            opfunc="add",
            execution_unit="sfp",
            data_format=_FP16,
            num_inputs=1,
            iteration_space={s: 32},
            num_cores=2,
            work_slices={s: 2},
            core_id_to_work_slice={s: Mod(core_id, 2)},
            padding={},
            layouts={"A": {"dim_order": [s], "stick_dim_order": s, "stick_size": 64}},
            args=[tensor],
            constants={},
            coordinate_masking={},
        )
        symbols: list[int] = []
        _, _, _, kinds = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[s],
            use_symbols=True,
        )
        self.assertEqual(len(kinds), 2)
        self.assertIsInstance(kinds[0], SymbolKind)
        self.assertEqual(kinds[0].kind, "kernel")
        self.assertIsInstance(kinds[1], SymbolKind)
        self.assertEqual(kinds[1].kind, "kernel_derived")
        self.assertEqual(kinds[1].base_sym_idx, 0)  # base is symbols[0]
        self.assertEqual(kinds[1].offset, symbols[1] - symbols[0])

    def test_bundle_kernel_derived_no_backward_scan(self):
        """bundle.py uses SymbolKind.base_sym_idx directly — no backward scan needed.
        Two ops, same kernel arg but different per-core offsets share one param."""
        a = _make_minimal_op_spec("a")
        b = _make_minimal_op_spec("b")

        call_count = [0]

        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            i = call_count[0]
            call_count[0] += 1
            base = 0x400000000
            off = 1024
            if i == 0:
                # op0: sym 0 = kernel base, sym 1 = kernel_derived +1024
                symbols.append(base)
                symbols.append(base + off)
                kinds = [SymbolKind.kernel(0), SymbolKind.kernel_derived(0, off, 0)]
                json0 = _make_tiled_json(idx, -(symbol_id_offset + 1))
                return json0, [base, base + off], [{}, {}], kinds
            else:
                # op1: reuses same derived offset — sym 2
                symbols.append(base + off)
                kinds = [SymbolKind.kernel_derived(0, off, 0)]
                json1 = _make_tiled_json(idx, -(symbol_id_offset + 1))
                return json1, [base + off], [{}], kinds

        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=fake,
        ):
            generate_bundle(
                "test_kernel",
                self.tmpdir,
                [a, b],
                unroll_loops=False,
                symbolic_args=True,
            )
        mlir = _read_mlir(self.tmpdir)

        # Only one input_arg param (the kernel base)
        self.assertIn("%arg_0_base_addr: !sdscbundle.input_arg<index>", mlir)
        self.assertNotIn("%sym_0_2:", mlir)
        # Derived address emitted once as arith.addi (deduped across both ops)
        self.assertEqual(mlir.count("arith.constant 1024"), 1)
        self.assertEqual(mlir.count("arith.addi %arg_0"), 1)
        # op0's execute has the kernel base; op1's execute has the derived %sym_N
        # Both refer to the same canonical derived SSA — no second arith.addi for op1
        self.assertIn("sdscbundle.sdsc_execute (%arg_0)", mlir)
        # op1 operand is the canonical derived var (%arg_0_core_1024), not a new addi
        execute_lines = [ln for ln in mlir.splitlines() if "sdsc_execute" in ln]
        op1_operand = execute_lines[1].split("(")[1].split(")")[0].strip()
        self.assertIn("arg_0_core", op1_operand)  # derived from arg_0 with offset
        self.assertNotIn("input_arg_extract", op1_operand)


if __name__ == "__main__":
    unittest.main()
