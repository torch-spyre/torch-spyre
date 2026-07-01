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
  3. CountedLoopSchedulerNode, build_loop_scheduler_nodes,
     _tiled_syms_for_sched_node_at_depth, and spyre_fuse_nodes budget check
     (TestHelpers, TestBuildLoopSchedulerNodes, TestTiledSymsForSchedNode,
      TestSpyreFuseNodesLoopBudget)
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

import sympy
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
from torch_spyre._inductor.coarse_tile import (
    _LOOPS_FREE_SYMS_KEY,
    _REDUCTION_FREE_SYMS_KEY,
    _divide_ranges,
    coarse_tile,
)
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp
from torch_spyre._inductor.scheduler import (
    CountedLoopSchedulerNode,
    _loop_count,
    _loop_group_id,
    build_loop_scheduler_nodes,
)
from torch_spyre._inductor.fusion import spyre_fuse_nodes
from torch_spyre._inductor.spyre_kernel import (
    _codegen_op_spec_list,
    _iter_op_specs,
    _preserve_shared_weight_unit_bmm_dim,
)
from torch_spyre._inductor.temp_passes import (
    _mark_static_unit_batch_bmm,
    mark_direct_unit_bmm_pass,
)

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
        tiled_symbols=[[c0]],
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

    def test_cache_invalidated_after_divide_pointwise(self):
        from torch._inductor.ir import ComputedBuffer, FixedLayout, Pointwise

        N = sympy.Symbol("N", positive=True)
        pw = Pointwise(
            device=torch.device("cpu"),
            dtype=torch.float16,
            inner_fn=lambda index: sympy.Integer(1),
            ranges=[N, Integer(32)],
        )
        layout = FixedLayout(torch.device("cpu"), torch.float16, [N, Integer(32)])
        op = ComputedBuffer(name="buf0", layout=layout, data=pw)

        pw.get_free_symbol_uses()  # prime the cache
        self.assertTrue(hasattr(pw, _LOOPS_FREE_SYMS_KEY))

        _divide_ranges(op, Integer(4), tiled_dims=[0])

        self.assertFalse(hasattr(pw, _LOOPS_FREE_SYMS_KEY))

    def test_cache_invalidated_after_divide_reduction(self):
        from torch._inductor.ir import (
            ComputedBuffer,
            FixedLayout,
            Reduction,
            ReductionHint,
        )

        N = sympy.Symbol("N", positive=True)
        red = Reduction(
            device=torch.device("cpu"),
            dtype=torch.float16,
            inner_fn=lambda index, rindex: sympy.Integer(1),
            ranges=[N],
            reduction_ranges=[Integer(128)],
            reduction_type="sum",
            src_dtype=torch.float16,
            reduction_hint=ReductionHint.DEFAULT,
        )
        layout = FixedLayout(torch.device("cpu"), torch.float16, [N])
        op = ComputedBuffer(name="buf0", layout=layout, data=red)

        red.get_free_symbol_uses()  # prime both Loops and Reduction cache entries
        self.assertTrue(hasattr(red, _LOOPS_FREE_SYMS_KEY))
        self.assertTrue(hasattr(red, _REDUCTION_FREE_SYMS_KEY))

        _divide_ranges(op, Integer(4), tiled_dims=[0])

        self.assertFalse(hasattr(red, _LOOPS_FREE_SYMS_KEY))
        self.assertFalse(hasattr(red, _REDUCTION_FREE_SYMS_KEY))

    # ------------------------------------------------------------------
    # Device-layout reconstruction tests (FixedTiledLayout path)
    # ------------------------------------------------------------------

    def _make_ftl_op(self, host_size, dim_order, dtype=torch.float16, elem_arr=None):
        """Build a ComputedBuffer with a FixedTiledLayout for testing _divide_ranges.

        Returns (op, layout) where layout.device_layout is a SpyreTensorLayout
        constructed from (host_size, contiguous_strides, dtype, dim_order, elem_arr).
        """
        from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise

        from torch_spyre._C import ElementArrangement, SpyreTensorLayout
        from torch_spyre._inductor.ir import FixedTiledLayout

        if elem_arr is None:
            elem_arr = ElementArrangement.STANDARD

        strides = [int(s) for s in FlexibleLayout.contiguous_strides(host_size)]
        device_layout = SpyreTensorLayout(
            host_size, strides, dtype, dim_order, elem_arr
        )
        layout = FixedTiledLayout(
            torch.device("cpu"),
            dtype,
            [Integer(s) for s in host_size],
            [Integer(s) for s in strides],
            device_layout,
        )
        pw = Pointwise(
            device=torch.device("cpu"),
            dtype=dtype,
            inner_fn=lambda index: sympy.Integer(1),
            ranges=[Integer(s) for s in host_size],
        )
        op = ComputedBuffer(name="buf0", layout=layout, data=pw)
        return op, layout

    def test_divide_ranges_transposed_stick_preserved(self):
        """Tiling a non-stick dim of a transposed-stick layout rebuilds
        device_layout correctly (headline regression from code review)."""
        from torch._inductor.ir import FlexibleLayout

        from torch_spyre._C import SpyreTensorLayout

        # [256, 128] with stick on dim0: dim_order=[1, 0].  This is the layout
        # produced for a transposed Linear weight (model_utils.py restickify).
        op, layout = self._make_ftl_op([256, 128], dim_order=[1, 0])

        # Tile non-stick dim1 by 2: [256, 128] -> [256, 64].
        _divide_ranges(op, Integer(2), tiled_dims=[1])

        # Expected: from-scratch SpyreTensorLayout([256, 64], ..., [1, 0]).
        expected_strides = [
            int(s) for s in FlexibleLayout.contiguous_strides([256, 64])
        ]
        expected = SpyreTensorLayout([256, 64], expected_strides, torch.float16, [1, 0])

        self.assertEqual(layout.device_layout, expected)

        # Also assert it differs from the buggy heuristic result.
        buggy = SpyreTensorLayout(
            [1, 256, 64],
            [64, 64, 1],
            expected.device_dtype,
            expected.element_arrangement,
        )
        self.assertNotEqual(layout.device_layout, buggy)

    def test_divide_ranges_preserves_element_arrangement(self):
        """element_arrangement is copied verbatim — not silently reset to STANDARD."""
        from torch._inductor.ir import FlexibleLayout

        from torch_spyre._C import ElementArrangement, SpyreTensorLayout

        op, layout = self._make_ftl_op(
            [256, 128], dim_order=[1, 0], elem_arr=ElementArrangement.EXX2
        )

        _divide_ranges(op, Integer(2), tiled_dims=[1])

        self.assertEqual(
            layout.device_layout.element_arrangement, ElementArrangement.EXX2
        )

        # Confirm the rebuilt layout also has the right shape.
        expected_strides = [
            int(s) for s in FlexibleLayout.contiguous_strides([256, 64])
        ]
        expected = SpyreTensorLayout(
            [256, 64], expected_strides, torch.float16, [1, 0], ElementArrangement.EXX2
        )
        self.assertEqual(layout.device_layout, expected)

    def test_divide_ranges_stride_collision(self):
        """Tiling an outer dim when stride_map has two entries with the same
        value (device_size tiebreak case) produces the correct device_layout."""
        from torch._inductor.ir import FlexibleLayout

        from torch_spyre._C import SpyreTensorLayout

        # [2, 2, 2, 16] contiguous, stick on dim3 (last).  host_stride[0]=64
        # equals 64*host_stride[3], so the stick tile-count and a non-stick dim
        # share a stride_map value; stride check must break the tie.
        op, layout = self._make_ftl_op([2, 2, 2, 16], dim_order=[0, 1, 2, 3])

        # Tile dim0: [2,2,2,16] -> [1,2,2,16].
        _divide_ranges(op, Integer(2), tiled_dims=[0])

        expected_strides = [
            int(s) for s in FlexibleLayout.contiguous_strides([1, 2, 2, 16])
        ]
        expected = SpyreTensorLayout(
            [1, 2, 2, 16], expected_strides, torch.float16, [0, 1, 2, 3]
        )
        self.assertEqual(layout.device_layout, expected)

    def test_divide_ranges_tile_count_size_collision(self):
        """Tile-count device_size equals a non-stick host dim size — the stride
        check (not size alone) must classify it correctly.

        [2, 128] with stick on dim1: tile-count device_size = ceil(128/64) = 2,
        which equals old_host_size[0] = 2.  Without the stride check, Pass 1
        misclassifies the tile-count dim as non-stick and never updates it."""
        from torch._inductor.ir import FlexibleLayout

        from torch_spyre._C import SpyreTensorLayout

        op, layout = self._make_ftl_op([2, 128], dim_order=[0, 1])

        # Tile dim0: [2, 128] -> [1, 128].
        _divide_ranges(op, Integer(2), tiled_dims=[0])

        expected_strides = [int(s) for s in FlexibleLayout.contiguous_strides([1, 128])]
        expected = SpyreTensorLayout([1, 128], expected_strides, torch.float16, [0, 1])
        self.assertEqual(layout.device_layout, expected)

    def test_resize_device_layout_grow_from_singleton(self):
        """_allocate_full_buffer grow path: a device dim tiled to size 1
        (stride_map != -1) must be grown back on the full-buffer allocation.

        [1, 128] grow dim0 -> [4, 128]: the size-1 non-stick device dim must
        update to device_size=4, not remain frozen at 1."""
        from torch_spyre._C import SpyreTensorLayout
        from torch_spyre._inductor.coarse_tile import _resize_device_layout

        # Per-tile buffer is [1, 128] — dim0 was tiled to extent 1.
        # device_size=[2, 1, 64], stride_map=[64, -1, 1].
        stl = SpyreTensorLayout([1, 128], [128, 1], torch.float16, [0, 1])
        result = _resize_device_layout(stl, [1, 128], [4, 128])

        expected = SpyreTensorLayout([4, 128], [128, 1], torch.float16, [0, 1])
        self.assertEqual(result, expected)

    def test_resize_device_layout_raises_on_unsupported(self):
        """_resize_device_layout raises RuntimeError when the stick host dim
        cannot be uniquely identified from stride_map[-1].

        This guards against unsupported layouts (e.g. future multi-host-dim
        sticks) rather than silently producing a wrong result.
        """
        from torch_spyre._C import SpyreTensorLayout
        from torch_spyre._inductor.coarse_tile import _resize_device_layout

        # Build a real [2, 2] STL (stick on dim1, stride_map[-1] == 1).
        # Then call the helper with a synthetic old_host_size=[1, 1] whose
        # contiguous strides are both 1 — two dims share stride_map[-1], so
        # p* cannot be identified uniquely.
        stl = SpyreTensorLayout([2, 2], [2, 1], torch.float16, [0, 1])
        with self.assertRaises(RuntimeError):
            _resize_device_layout(stl, [1, 1], [1, 1])

    def test_resize_device_layout_reduction_output(self):
        """Reduction output: stick host dim has been eliminated, so old_host_size
        has no unmatched dim.  _resize_device_layout must handle this gracefully
        by leaving the tile-count and inner-stick entries frozen."""
        from torch_spyre._C import SpyreTensorLayout
        from torch_spyre._inductor.coarse_tile import _resize_device_layout

        # [128] reduction output: SpyreTensorLayout([128], [1], fp16, [0]).
        # device_size=[1, 128, 64], stride_map=[-1, 1, -1] — tile-count dim is
        # frozen at 1 (stick collapsed), inner stick frozen at -1.
        stl = SpyreTensorLayout([128], [1], torch.float16, [0])
        # Tile the non-stick dim: [128] -> [64].
        result = _resize_device_layout(stl, [128], [64])

        # Non-stick device dim (j=1, size 128) updates to size 64, stride 1.
        # Tile-count (j=0, size 1) and inner stick (j=2, size 64) are frozen.
        expected = SpyreTensorLayout([64], [1], torch.float16, [0])
        self.assertEqual(result, expected)


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
            [([op_extern, op_computed], [(0, Integer(2), False)])],
        )
        self.assertEqual(op_computed.loop_info.loop_group_id, (0,))
        self.assertEqual(data.ranges[0], Integer(8))

    def test_symbolic_count(self):
        k = Symbol("K", positive=True)
        n = Symbol("N", positive=True)
        data = _make_pointwise([n])
        op = _make_hinted_op(data, "op0", hints=((0, 0),))
        coarse_tile(_graph([op]), [([op], [(0, k, False)])])
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
            coarse_tile(
                _graph([op0, op1, op2]), [([op0, op2], [(0, Integer(4), False)])]
            )

    def test_op_not_in_operations_raises(self):
        data = _make_pointwise([Integer(32)])
        op_known = _make_hinted_op(data, "op0", hints=((0, 0),))
        op_unknown = _make_hinted_op(
            _make_pointwise([Integer(8)]), "unknown", hints=((0, 0),)
        )
        with self.assertRaises(RuntimeError):
            coarse_tile(_graph([op_known]), [([op_unknown], [(0, Integer(2), False)])])


class TestCoarseTileNested(unittest.TestCase):
    """Verify that the nested group format [(hint_id, K1, is_reduction), ...] works."""

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
        coarse_tile(
            _graph([op]), [([op], [(1, Integer(4), False), (2, Integer(2), False)])]
        )
        self.assertEqual(op.loop_info.loop_group_id, (0, 0))
        self.assertEqual(op.loop_info.loop_count, [Integer(4), Integer(2)])
        self.assertEqual(op.loop_info.loop_tiled_dims, [[0], [1]])

    def test_nested_spec_divides_ranges_both_levels(self):
        data = _make_pointwise([Integer(256), Integer(128)])
        op = _make_hinted_op(data, "op0", hints=((1, 0), (2, 1)))
        coarse_tile(
            _graph([op]), [([op], [(1, Integer(4), False), (2, Integer(2), False)])]
        )
        self.assertEqual(data.ranges[0], Integer(64))
        self.assertEqual(data.ranges[1], Integer(64))

    def test_nested_spec_outer_only_divides_outer_dim(self):
        data = _make_pointwise([Integer(32), Integer(64), Integer(16)])
        op = _make_hinted_op(data, "op0", hints=((1, 0), (2, 1)))
        coarse_tile(
            _graph([op]), [([op], [(1, Integer(4), False), (2, Integer(8), False)])]
        )
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
                ([op0], [(1, Integer(4), False)]),
                ([op1], [(2, Integer(4), False), (3, Integer(2), False)]),
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
        coarse_tile(
            _graph([op]), [([op], [(1, Integer(4), False), (2, Integer(2), False)])]
        )
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
# 3b. spyre_fuse_nodes CountedLoopSchedulerNode budget check
# ===========================================================================


def _make_snode_with_tensors(scheduler, ir_op, name, tensor_names):
    """Like _make_snode but read_writes returns named dependency mocks."""
    snode = _make_snode(scheduler, ir_op, name)
    deps = []
    for tname in tensor_names:
        dep = MagicMock()
        dep.name = tname
        deps.append(dep)
    snode.read_writes.reads_and_writes.return_value = deps
    return snode


def _make_counted_loop_node(scheduler, tensor_names, name="loop0"):
    """Return a fake CountedLoopSchedulerNode with the given tensor names."""
    node = MagicMock(spec=CountedLoopSchedulerNode)
    node.scheduler = scheduler
    node.get_name.return_value = name
    node.get_nodes.return_value = [node]
    deps = []
    for tname in tensor_names:
        dep = MagicMock()
        dep.name = tname
        deps.append(dep)
    node.read_writes = MagicMock()
    node.read_writes.reads_and_writes.return_value = deps
    return node


class TestSpyreFuseNodesLoopBudget(unittest.TestCase):
    def setUp(self):
        # All tensor names count as non-intermediate.
        self._patcher = patch(
            "torch_spyre._inductor.fusion._is_non_intermediate",
            side_effect=lambda name: True,
        )
        self._patcher.start()
        # Cap the bundle at 2 tensors so overflow is easy to trigger.
        self._max_patcher = patch(
            "torch_spyre._inductor.fusion._max_bundle_tensors",
            return_value=2,
        )
        self._max_patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._max_patcher.stop()

    def test_loop_node_within_budget_no_error(self):
        """A loop node referencing <= max_tensors: passes through without error."""
        sched = _make_scheduler()
        loop = _make_counted_loop_node(sched, ["t1", "t2"], "loop0")
        result = spyre_fuse_nodes([loop])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], loop)

    def test_loop_node_exceeds_budget_raises(self):
        """A loop node referencing > max_tensors must raise RuntimeError."""
        sched = _make_scheduler()
        loop = _make_counted_loop_node(sched, ["t1", "t2", "t3"], "loop0")
        with self.assertRaises(RuntimeError) as ctx:
            spyre_fuse_nodes([loop])
        self.assertIn("loop0", str(ctx.exception))
        self.assertIn("3", str(ctx.exception))
        self.assertIn("2", str(ctx.exception))

    def test_loop_node_empty_tensors_no_error(self):
        """A loop node with no tensors (all intermediates) is fine."""
        sched = _make_scheduler()
        # Override: nothing is non-intermediate for this test.
        with patch(
            "torch_spyre._inductor.fusion._is_non_intermediate",
            side_effect=lambda name: False,
        ):
            loop = _make_counted_loop_node(sched, ["t1", "t2", "t3"], "loop0")
            result = spyre_fuse_nodes([loop])
        self.assertEqual(len(result), 1)

    def test_plain_scheduler_node_split_unaffected(self):
        """Plain SchedulerNode tensor-budget splits still work (no regression)."""
        sched = _make_scheduler()
        a = _make_snode_with_tensors(sched, _make_ir_op(), "a", ["t1", "t2"])
        b = _make_snode_with_tensors(sched, _make_ir_op(), "b", ["t3"])
        # a fills the 2-tensor budget; b starts a new bundle.
        result = spyre_fuse_nodes([a, b])
        self.assertEqual(len(result), 2)

    def test_loop_node_preceded_by_scheduler_nodes(self):
        """Loop node after plain nodes: budget check on loop node itself."""
        sched = _make_scheduler()
        a = _make_snode_with_tensors(sched, _make_ir_op(), "a", ["t1"])
        loop = _make_counted_loop_node(sched, ["t2", "t3", "t4"], "loop0")
        with self.assertRaises(RuntimeError):
            spyre_fuse_nodes([a, loop])

    def test_fallback_node_exceeds_budget_no_error(self):
        """Non-CountedLoop nodes (e.g. FallbackKernel) bypass the budget check."""
        from torch._inductor.scheduler import FusedSchedulerNode

        sched = _make_scheduler()
        # Build a plain FusedSchedulerNode mock (not CountedLoopSchedulerNode).
        node = MagicMock(spec=FusedSchedulerNode)
        node.scheduler = sched
        node.get_name.return_value = "fallback0"
        deps = []
        for tname in ["t1", "t2", "t3", "t4"]:
            dep = MagicMock()
            dep.name = tname
            deps.append(dep)
        node.read_writes = MagicMock()
        node.read_writes.reads_and_writes.return_value = deps
        # 4 tensors > max_tensors(2), but this is not a CountedLoopSchedulerNode.
        result = spyre_fuse_nodes([node])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], node)

    def test_nested_loop_node_tensor_union_checked(self):
        """Outer CountedLoopSchedulerNode's read_writes covers inner tensors.

        _build_loop_group wraps inner loops first, then the outer loop.
        FusedSchedulerNode.__init__ calls ReadWrites.merge_list on its
        snodes, so the outer node's read_writes is already the full union
        of all inner tensors.  This test confirms that the budget check
        on the outer node sees all tensors from nested inner nodes.
        """
        sched = _make_scheduler()
        # Simulate an outer CountedLoopSchedulerNode whose read_writes
        # already aggregates tensors from two inner loop nodes (t1..t4).
        # In production this aggregation is done by ReadWrites.merge_list
        # during FusedSchedulerNode construction.
        outer = _make_counted_loop_node(sched, ["t1", "t2", "t3", "t4"], "outer_loop")
        # 4 unique non-intermediate tensors > max_tensors(2): must raise.
        with self.assertRaises(RuntimeError) as ctx:
            spyre_fuse_nodes([outer])
        self.assertIn("outer_loop", str(ctx.exception))
        self.assertIn("4", str(ctx.exception))


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
            tiled_symbols=[[s]],
            use_symbols=True,
        )
        # affine_strides[tensor_idx] is list[dict] (per level, outermost first).
        # With one tiling level, affine_strides[0] = [{s: stride}].
        self.assertEqual(len(affine_strides), 1)
        self.assertIn(s, affine_strides[0][0])
        self.assertEqual(affine_strides[0][0][s], 64 * 128 * 2)

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
            tiled_symbols=[[s]],
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
            tiled_symbols=[[s]],
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
        self.assertEqual(affine_strides, [[]])

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
            tiled_symbols=[[s]],
        )
        self.assertEqual(symbols, [])
        self.assertEqual(local_sym_values, [])
        # lx tensor: one level of tiled_symbols, but lx allocation is always non-tiled.
        self.assertEqual(affine_strides, [[{}]])

    def test_symbol_id_offset_applied(self):
        s = Symbol("s")
        sdsc_spec = _make_sdsc_spec(s)
        symbols: list[int] = []
        sdsc_json, local_sym_values, _, _ = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=5,
            tiled_symbols=[[s]],
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
            tiled_symbols=[[s]],
            use_symbols=True,
        )
        self.assertEqual(len(symbols), 2)
        self.assertEqual(symbols[0], 0x1000)
        self.assertEqual(symbols[1], 0x1000 + 128)
        # affine_strides[0] = [{s: stride}] (one level, one tensor)
        self.assertIn(s, affine_strides[0][0])


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
            tiled_symbols=[[c0, c1]],
        )

    def test_two_tiled_symbols_produce_two_stride_entries(self):
        op_spec = self._make_3d_op_spec()
        symbols: list[int] = []
        _, _, affine_strides, _ = compile_op_spec(0, op_spec, symbols, use_symbols=True)
        # affine_strides[tensor_idx] = list[dict] (per level, outermost first).
        # Both tensors have one tiling level with two symbols.
        # Find tensors with non-empty strides at any level.
        hbm_strides = [
            per_level
            for per_level in affine_strides
            if any(len(d) > 0 for d in per_level)
        ]
        self.assertGreater(len(hbm_strides), 0)
        for per_level in hbm_strides:
            total_strides = sum(len(d) for d in per_level)
            self.assertEqual(total_strides, 2)

    def test_two_tiled_symbols_strides_are_positive(self):
        op_spec = self._make_3d_op_spec()
        symbols: list[int] = []
        _, _, affine_strides, _ = compile_op_spec(0, op_spec, symbols, use_symbols=True)
        for per_level in affine_strides:
            for level_strides in per_level:
                for sym, stride in level_strides.items():
                    self.assertGreater(stride, 0)


class TestCompileOpSpecSymbolMapping(unittest.TestCase):
    def test_affine_strides_non_empty_for_tiled_op(self):
        op_spec = _make_tiled_op_spec()
        symbols: list[int] = []
        _, _, affine_strides, _ = compile_op_spec(0, op_spec, symbols, use_symbols=True)
        # affine_strides[tensor_idx] = list[dict] (per level, outermost first).
        has_strides = any(
            any(len(level_d) > 0 for level_d in per_level)
            for per_level in affine_strides
        )
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

        _mark_static_unit_batch_bmm(bmm, x, y)
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
        )
        kernel_arg = TensorArg(
            is_input=True,
            arg_index=1,
            device_dtype=_FP16,
            device_size=[200, 4096, 64],
            device_coordinates=[floor(c1 / 64), c2, Mod(c1, 64)],
            allocation={"hbm": 0x400000000},
        )
        output_arg = TensorArg(
            is_input=False,
            arg_index=2,
            device_dtype=_FP16,
            device_size=[512, 200, 1, 64],
            device_coordinates=[c0, floor(c1 / 64), Integer(0), Mod(c1, 64)],
            allocation={"hbm": 0x800000000},
        )
        for arg in (input_arg, output_arg):
            del arg.device_size[-2]
            del arg.device_coordinates[-2]
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
        m, k, n = 2, 128, 64
        self.assertEqual(
            self._static_bmm_custom_meta((1, m, k), (1, k, n), (1, m, n))[
                SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY
            ],
            {"batch_dim": 0},
        )
        self.assertNotIn(
            SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY,
            self._static_bmm_custom_meta((4, m, k), (4, k, n), (4, m, n)),
        )
        self.assertNotIn(
            SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY,
            self._static_bmm_custom_meta((1, m, 2), (1, 2, n), (1, m, n)),
        )

    def test_mark_direct_unit_bmm_pass_does_not_mark_reshape_inputs(self):
        m, k, n = 2, 64, 128
        graph = fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        x_view = graph.call_function(
            torch.ops.aten.reshape.default, args=(x, (1, m, k))
        )
        y_view = graph.call_function(
            torch.ops.aten.reshape.default, args=(y, (1, k, n))
        )
        bmm = graph.call_function(torch.ops.aten.bmm.default, args=(x_view, y_view))
        graph.output(bmm)

        mark_direct_unit_bmm_pass(graph)
        graph.lint()
        self.assertNotIn(
            SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY,
            bmm.meta.get("custom") or {},
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
            '\t\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json", "symbol_ids"=[]}\n'
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
            '\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json", "symbol_ids"=[]}\n'
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
            # affine_strides: list[list[dict]] — one tensor, one level, one stride.
            return _make_tiled_json(idx, sym_id), [0x1000], [[{s: stride}]], []

        op = _make_minimal_op_spec("a")
        op.tiled_symbols = [[s]]
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
            return _make_tiled_json(idx, sym_id), [0x2000], [[{}]], []

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
            return _make_tiled_json(idx, sym_id), [0x3000], [[{s: stride}]], []

        op = _make_minimal_op_spec("c")
        op.tiled_symbols = [[s]]
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
            return _make_tiled_json(idx, sym_id), [0x4000], [[{s: 512}]], []

        op = _make_minimal_op_spec("d")
        op.tiled_symbols = [[s]]
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
            return _make_tiled_json(idx, sym_id), [0x1000], [[{s: 256}]], []

        op = _make_minimal_op_spec("a")
        op.tiled_symbols = [[s]]
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
            # per_level_strides: outermost-first. Level 0 (outer) has s0 stride,
            # level 1 (inner) has s1 stride.  One tensor, two levels.
            return (
                _make_tiled_json(idx, sym_id),
                [0x1000],
                [[{s0: outer_stride}, {s1: inner_stride}]],
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


class TestGenerateBundleUnrollPath(unittest.TestCase):
    """Verify affine-map correctness for the unroll_loops=False path.

    One test group per scenario covered by test_unroll_loop_specs.py:
      Group 1 — flat row-tiling         (mirrors TestUnrollLoopSpecs)
      Group 2 — nested outer-B/inner-K reduction  (mirrors TestNestedReductionUnroll)
      Group 3 — tile-accum copy pattern (mirrors TestNestedReductionTileAccum)

    Key invariants:
      - ops tiled only by the inner loop var emit affine.apply with that var only
      - ops not tiled (per_tile_fixed or fixed address) emit no affine.apply
      - the copy op (outer-B tiled) emits affine.apply with the outer loop var
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._s = Symbol("s")
        self._c_k = Symbol("c_k")
        self._c_b = Symbol("c_b")

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

    # --- Group 1: flat row-tiling ---

    def test_flat_loop_tiled_tensor_emits_affine_apply(self):
        s = self._s

        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x1000)
            # One tensor, one level (the enclosing loop), one stride.
            return _make_tiled_json(idx, sym_id), [0x1000], [[{s: 256}]], []

        op = _make_minimal_op_spec("a")
        loop = LoopSpec(count=Integer(4), body=[op])
        mlir = self._bundle([loop], fake)

        self.assertIn("affine_map", mlir)
        self.assertIn("affine.apply", mlir)
        self.assertIn("256", mlir)

    def test_flat_loop_non_tiled_tensor_no_affine_apply(self):
        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x2000)
            return _make_tiled_json(idx, sym_id), [0x2000], [[{}]], []

        op = _make_minimal_op_spec("b")
        loop = LoopSpec(count=Integer(4), body=[op])
        mlir = self._bundle([loop], fake)

        self.assertNotIn("affine_map", mlir)
        self.assertNotIn("affine.apply", mlir)
        self.assertIn("%sym_1", mlir)

    def test_flat_loop_snapshot(self):
        s = self._s

        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x1000)
            return _make_tiled_json(idx, sym_id), [0x1000], [[{s: 256}]], []

        op = _make_minimal_op_spec("a")
        loop = LoopSpec(count=Integer(4), body=[op])
        mlir = self._bundle([loop], fake)

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

    # --- Group 2: nested outer-B + inner-K reduction ---
    #
    # Strides match TestNestedReductionUnroll in test_unroll_loop_specs.py:
    #   k_input: device_size=[2,64,64]; device_stride[0]=prod([64,64])=4096
    #     128 K-elems/tile → 2 sticks; byte_stride = (128//64)*4096*2 = 16384
    #   accum_buf: device_size=[1,2,64]; device_stride[1]=prod([64])=64
    #     2 batches/tile; byte_stride = 2*64*2 = 256
    # Only K_STRIDE appears in the affine map (accum_buf not tiled on K).

    _GRP2_K_STRIDE = 16384  # (128//64) * prod([64,64]) * 2

    def _fake_nested_reduction(self, k_stride):
        c_k = self._c_k
        call_count = [0]

        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            i = call_count[0]
            call_count[0] += 1
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x1000 * (i + 1))
            if i == 0:
                # bmm: tiled on inner K var only (level 1 in outer>inner nesting).
                # per_level_strides: outermost first — outer has no K stride, inner has it.
                return (
                    _make_tiled_json(idx, sym_id),
                    [0x1000],
                    [[{}, {c_k: k_stride}]],
                    [],
                )
            else:
                # combine: accum_buf not tiled on K at any level.
                return _make_tiled_json(idx, sym_id), [0x2000], [[{}, {}]], []

        return fake

    def _make_nested_reduction_specs(self):
        bmm = _make_minimal_op_spec("batchmatmul")
        combine = _make_minimal_op_spec("add")
        inner = LoopSpec(count=Integer(4), body=[bmm, combine])
        outer = LoopSpec(count=Integer(2), body=[inner])
        return [outer]

    def test_nested_reduction_bmm_emits_affine_apply(self):
        mlir = self._bundle(
            self._make_nested_reduction_specs(),
            self._fake_nested_reduction(self._GRP2_K_STRIDE),
        )
        self.assertIn("affine.apply", mlir)
        self.assertIn(str(self._GRP2_K_STRIDE), mlir)

    def test_nested_reduction_combine_no_affine_apply(self):
        """combine's accum_buf (not tiled on K) must not get an affine.apply."""
        mlir = self._bundle(
            self._make_nested_reduction_specs(),
            self._fake_nested_reduction(self._GRP2_K_STRIDE),
        )
        # Only one affine.apply (for the bmm); the combine uses %sym_2 directly.
        self.assertEqual(mlir.count("affine.apply"), 1)
        execute_lines = [ln for ln in mlir.splitlines() if "sdsc_execute" in ln]
        combine_line = execute_lines[1]
        self.assertIn("%sym_2", combine_line)
        self.assertNotIn("addr", combine_line)

    def test_nested_reduction_loop_structure(self):
        mlir = self._bundle(
            self._make_nested_reduction_specs(),
            self._fake_nested_reduction(self._GRP2_K_STRIDE),
        )
        self.assertEqual(mlir.count("scf.for"), 2)

    def test_nested_reduction_snapshot(self):
        mlir = self._bundle(
            self._make_nested_reduction_specs(),
            self._fake_nested_reduction(self._GRP2_K_STRIDE),
        )
        expected = (
            f"#map_0 = affine_map<(d0)[s0] -> (s0 + {self._GRP2_K_STRIDE}*d0)>\n"
            "module {\n"
            "\tfunc.func @sdsc_bundle() {\n"
            "\t\t%c0 = arith.constant 0 : index\n"
            "\t\t%c1 = arith.constant 1 : index\n"
            "\t\t%loop_bound_0 = arith.constant 2 : index\n"
            "\t\t%loop_bound_1 = arith.constant 4 : index\n"
            "\t\t%sym_1 = arith.constant 4096 : index\n"
            "\t\t%sym_2 = arith.constant 8192 : index\n"
            "\t\tscf.for %i_0 = %c0 to %loop_bound_0 step %c1 {\n"
            "\t\t\tscf.for %i_1 = %c0 to %loop_bound_1 step %c1 {\n"
            "\t\t\t\t%addr_0 = affine.apply #map_0(%i_1)[%sym_1]\n"
            '\t\t\t\tsdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_0.json",'
            ' "symbol_ids"=[-1]}\n'
            '\t\t\t\tsdscbundle.sdsc_execute (%sym_2) {sdsc_filename="sdsc_1.json",'
            ' "symbol_ids"=[-2]}\n'
            "\t\t\t}\n"
            "\t\t}\n"
            "\t\treturn\n"
            "\t}\n"
            "}\n"
        )
        self.assertEqual(mlir, expected)

    # --- Group 3: tile-accum copy pattern ---
    #
    # Strides match TestNestedReductionTileAccum in test_unroll_loop_specs.py:
    #   bmm K-input: same geometry as Group 2 → K_STRIDE = 16384
    #   accum_full (copy output): device_size=[1,128,32]
    #     device_stride[0]=prod([128,32])=4096; 1 tile advances c_b by 1
    #     byte_stride = 1 * 4096 * 2 = 8192  (_OUTER_TILE_STRIDE_BYTES)

    _GRP3_K_STRIDE = 16384  # (128//64) * prod([64,64]) * 2
    _GRP3_B_STRIDE = 8192  # 1 * prod([128,32]) * 2

    def _fake_tile_accum(self, k_stride, b_stride):
        c_k, c_b = self._c_k, self._c_b
        call_count = [0]

        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            i = call_count[0]
            call_count[0] += 1
            sym_id = -(symbol_id_offset + 1)
            symbols.append(0x1000 * (i + 1))
            if i == 0:
                # fill: inside outer loop only, not tiled.
                return _make_tiled_json(idx, sym_id), [0x1000], [[{}]], []
            elif i == 1:
                # bmm: inside outer>inner; K-input tiled at inner level (level 1).
                return (
                    _make_tiled_json(idx, sym_id),
                    [0x2000],
                    [[{}, {c_k: k_stride}]],
                    [],
                )
            elif i == 2:
                # combine: inside outer>inner, per_tile_fixed accum_tile, not tiled.
                return _make_tiled_json(idx, sym_id), [0x3000], [[{}, {}]], []
            else:
                # copy: inside outer loop only; accum_full tiled at outer level (level 0).
                return _make_tiled_json(idx, sym_id), [0x4000], [[{c_b: b_stride}]], []

        return fake

    def _make_tile_accum_specs(self):
        fill = _make_minimal_op_spec("fill")
        bmm = _make_minimal_op_spec("batchmatmul")
        combine = _make_minimal_op_spec("add")
        copy = _make_minimal_op_spec("copy")
        inner = LoopSpec(count=Integer(4), body=[bmm, combine])
        outer = LoopSpec(count=Integer(2), body=[fill, inner, copy])
        return [outer]

    def test_tile_accum_copy_advances_per_outer_tile(self):
        """copy op (tiled on outer B) emits affine.apply with outer loop var %i_0."""
        mlir = self._bundle(
            self._make_tile_accum_specs(),
            self._fake_tile_accum(self._GRP3_K_STRIDE, self._GRP3_B_STRIDE),
        )
        apply_lines = [ln for ln in mlir.splitlines() if "affine.apply" in ln]
        # bmm uses %i_1 (inner K); copy uses %i_0 (outer B)
        self.assertTrue(
            any("%i_1" in ln for ln in apply_lines),
            "Expected bmm affine.apply to use inner loop var %i_1",
        )
        self.assertTrue(
            any("%i_0" in ln and "%i_1" not in ln for ln in apply_lines),
            "Expected copy affine.apply to use only outer loop var %i_0",
        )

    def test_tile_accum_fill_no_affine_apply(self):
        """fill op (per_tile_fixed output) must not get an affine.apply."""
        mlir = self._bundle(
            self._make_tile_accum_specs(),
            self._fake_tile_accum(self._GRP3_K_STRIDE, self._GRP3_B_STRIDE),
        )
        execute_lines = [ln for ln in mlir.splitlines() if "sdsc_execute" in ln]
        # fill is the first sdsc_execute inside the outer loop
        fill_line = execute_lines[0]
        self.assertIn("%sym_1", fill_line)
        self.assertNotIn("addr", fill_line)

    def test_tile_accum_snapshot(self):
        mlir = self._bundle(
            self._make_tile_accum_specs(),
            self._fake_tile_accum(self._GRP3_K_STRIDE, self._GRP3_B_STRIDE),
        )
        expected = (
            f"#map_0 = affine_map<(d0)[s0] -> (s0 + {self._GRP3_K_STRIDE}*d0)>\n"
            f"#map_1 = affine_map<(d0)[s0] -> (s0 + {self._GRP3_B_STRIDE}*d0)>\n"
            "module {\n"
            "\tfunc.func @sdsc_bundle() {\n"
            "\t\t%c0 = arith.constant 0 : index\n"
            "\t\t%c1 = arith.constant 1 : index\n"
            "\t\t%loop_bound_0 = arith.constant 2 : index\n"
            "\t\t%loop_bound_1 = arith.constant 4 : index\n"
            "\t\t%sym_1 = arith.constant 4096 : index\n"
            "\t\t%sym_2 = arith.constant 8192 : index\n"
            "\t\t%sym_3 = arith.constant 12288 : index\n"
            "\t\t%sym_4 = arith.constant 16384 : index\n"
            "\t\tscf.for %i_0 = %c0 to %loop_bound_0 step %c1 {\n"
            '\t\t\tsdscbundle.sdsc_execute (%sym_1) {sdsc_filename="sdsc_0.json",'
            ' "symbol_ids"=[-1]}\n'
            "\t\t\tscf.for %i_1 = %c0 to %loop_bound_1 step %c1 {\n"
            "\t\t\t\t%addr_0 = affine.apply #map_0(%i_1)[%sym_2]\n"
            '\t\t\t\tsdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_1.json",'
            ' "symbol_ids"=[-2]}\n'
            '\t\t\t\tsdscbundle.sdsc_execute (%sym_3) {sdsc_filename="sdsc_2.json",'
            ' "symbol_ids"=[-3]}\n'
            "\t\t\t}\n"
            "\t\t\t%addr_1 = affine.apply #map_1(%i_0)[%sym_4]\n"
            '\t\t\tsdscbundle.sdsc_execute (%addr_1) {sdsc_filename="sdsc_3.json",'
            ' "symbol_ids"=[-4]}\n'
            "\t\t}\n"
            "\t\treturn\n"
            "\t}\n"
            "}\n"
        )
        self.assertEqual(mlir, expected)

    # --- Group 4: two-tensor op — one tiled, one not ---
    #
    # Directly exercises per_tensor_lv_indices[tensor_idx] for both tensor_idx=0
    # (tiled, non-empty index list) and tensor_idx=1 (non-tiled, empty list).
    # Uses a single flat loop so the setup stays minimal.

    def test_two_tensor_op_only_tiled_tensor_gets_affine_apply(self):
        """Op with two tensors: first tiled (affine.apply), second not (sym direct)."""
        s = self._s

        def _make_two_tensor_json(idx, sym_id0, sym_id1):
            return {
                f"{idx}_mm": {
                    "numCoresUsed_": 1,
                    "dscs_": [
                        {
                            "mm": {
                                "scheduleTree_": [
                                    {
                                        "component_": "hbm",
                                        "startAddressCoreCorelet_": {
                                            "data_": {"[0, 0, 0]": str(sym_id0)}
                                        },
                                    },
                                    {
                                        "component_": "hbm",
                                        "startAddressCoreCorelet_": {
                                            "data_": {"[0, 0, 0]": str(sym_id1)}
                                        },
                                    },
                                ]
                            }
                        }
                    ],
                }
            }

        def fake(idx, op_spec, symbols, symbol_id_offset=0, use_symbols=False):
            sid0 = -(symbol_id_offset + 1)
            sid1 = -(symbol_id_offset + 2)
            symbols.append(0x1000)
            symbols.append(0x2000)
            # tensor 0: tiled at the enclosing loop level (level 0).
            # tensor 1: not tiled.
            return (
                _make_two_tensor_json(idx, sid0, sid1),
                [0x1000, 0x2000],
                [[{s: 256}], [{}]],
                [],
            )

        op = _make_minimal_op_spec("mm")
        loop = LoopSpec(count=Integer(4), body=[op])
        mlir = self._bundle([loop], fake)

        # Exactly one affine.apply (for tensor 0 only)
        self.assertEqual(mlir.count("affine.apply"), 1)
        apply_line = next(ln for ln in mlir.splitlines() if "affine.apply" in ln)
        self.assertIn("%i_0", apply_line)

        # tensor 1 (sym_2) appears directly in sdsc_execute, not via an %addr_N
        execute_line = next(ln for ln in mlir.splitlines() if "sdsc_execute" in ln)
        self.assertIn("%sym_2", execute_line)


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

    def test_reduction_tiled_reduction_dim_nested_ok(self):
        from torch_spyre._inductor.coarse_tile import _validate_reduction_tiling

        # Nested: outer tiles output dim, inner tiles reduction dim — now supported
        op = _make_tiled_reduction_op(
            "red0",
            ranges=[Integer(128)],
            reduction_ranges=[Integer(256)],
            reduction_type="sum",
            loop_group_id=(0, 0),
            loop_count=[Integer(2), Integer(4)],
            loop_tiled_dims=[[0], []],
        )
        op.loop_info.loop_tiled_reduction_dims = [[], [0]]
        _validate_reduction_tiling(op)  # must not raise

    def test_reduction_output_dim_tiled_ok(self):
        from torch_spyre._inductor.coarse_tile import _validate_reduction_tiling

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
        # output-dim-only tiling should not raise
        _validate_reduction_tiling(op)

    def test_nested_fill_gets_outer_loop_info(self):
        """Fill op gets outer-level loop_info for nested output+reduction tiling."""
        from torch_spyre._inductor.coarse_tile import _compute_fill_loop_info

        op = _make_tiled_reduction_op(
            "red0",
            ranges=[Integer(64)],
            reduction_ranges=[Integer(256)],
            reduction_type="sum",
            loop_group_id=(0, 0),
            loop_count=[Integer(2), Integer(4)],
            loop_tiled_dims=[[0], []],
        )
        op.loop_info.loop_tiled_reduction_dims = [[], [0]]
        fill_info = _compute_fill_loop_info(op)
        self.assertIsNotNone(fill_info)
        self.assertEqual(fill_info.loop_group_id, (0,))
        self.assertEqual(fill_info.loop_count, [Integer(2)])
        self.assertEqual(fill_info.loop_tiled_dims, [[0]])

    def test_flat_fill_has_no_loop_info(self):
        """Fill op gets no loop_info for flat (pure) reduction tiling."""
        from torch_spyre._inductor.coarse_tile import _compute_fill_loop_info

        op = _make_tiled_reduction_op(
            "red0",
            ranges=[Integer(128)],
            reduction_ranges=[Integer(256)],
            reduction_type="sum",
            loop_group_id=(0,),
            loop_count=[Integer(4)],
            loop_tiled_dims=[[]],
        )
        op.loop_info.loop_tiled_reduction_dims = [[0]]
        fill_info = _compute_fill_loop_info(op)
        self.assertIsNone(fill_info)


class TestComputeFillLoopInfo(unittest.TestCase):
    """_compute_fill_loop_info returns trimmed CoarseTileInfo for the fill op."""

    def test_flat_reduction_returns_none(self):
        """Pure reduction tiling (no output-dim level) → None (fill before all loops)."""
        from torch_spyre._inductor.coarse_tile import _compute_fill_loop_info

        op = _make_tiled_reduction_op(
            "red0",
            ranges=[Integer(128)],
            reduction_ranges=[Integer(256)],
            reduction_type="sum",
            loop_group_id=(0,),
            loop_count=[Integer(4)],
            loop_tiled_dims=[[]],
        )
        op.loop_info.loop_tiled_reduction_dims = [[0]]
        result = _compute_fill_loop_info(op)
        self.assertIsNone(result)

    def test_nested_outer_output_inner_reduction(self):
        """Outer tiles dim 0 (output), inner tiles reduction dim 0 → fill gets outer loop_info."""
        from torch_spyre._inductor.coarse_tile import _compute_fill_loop_info

        op = _make_tiled_reduction_op(
            "red0",
            ranges=[Integer(64)],
            reduction_ranges=[Integer(256)],
            reduction_type="sum",
            loop_group_id=(0, 0),
            loop_count=[Integer(2), Integer(4)],
            loop_tiled_dims=[[0], []],
        )
        op.loop_info.loop_tiled_reduction_dims = [[], [0]]
        result = _compute_fill_loop_info(op)
        self.assertIsNotNone(result)
        self.assertEqual(result.loop_group_id, (0,))
        self.assertEqual(result.loop_count, [Integer(2)])
        self.assertEqual(result.loop_tiled_dims, [[0]])
        self.assertEqual(result.loop_tiled_reduction_dims, [[]])


class TestValidateReductionTiling(unittest.TestCase):
    """Tests for _validate_reduction_tiling: raising on unsupported cases,
    passing on supported ones."""

    def _make_op(self, loop_tiled_dims, loop_tiled_reduction_dims):
        from torch._inductor.ir import ComputedBuffer, Reduction

        data = MagicMock(spec=Reduction)
        data.ranges = [Integer(128)]
        data.reduction_ranges = [Integer(256)]
        data.reduction_type = "sum"
        op = MagicMock(spec=ComputedBuffer)
        op.data = data
        op.get_name.return_value = "test_op"
        op.loop_info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(4)],
            loop_tiled_dims=loop_tiled_dims,
            loop_tiled_reduction_dims=loop_tiled_reduction_dims,
        )
        return op

    def test_pure_reduction_tile_ok(self):
        """Single level, only reduction dim tiled — Stage 1 supported case."""
        from torch_spyre._inductor.coarse_tile import _validate_reduction_tiling

        op = self._make_op(loop_tiled_dims=[[]], loop_tiled_reduction_dims=[[0]])
        _validate_reduction_tiling(op)  # must not raise

    def test_pure_output_tile_ok(self):
        """Single level, only output dim tiled — existing supported case."""
        from torch_spyre._inductor.coarse_tile import _validate_reduction_tiling

        op = self._make_op(loop_tiled_dims=[[0]], loop_tiled_reduction_dims=[[]])
        _validate_reduction_tiling(op)  # must not raise

    def test_no_loop_info_ok(self):
        """Op with no loop_info is not tiled — no error."""
        from torch._inductor.ir import ComputedBuffer, Reduction
        from torch_spyre._inductor.coarse_tile import _validate_reduction_tiling

        data = MagicMock(spec=Reduction)
        data.ranges = [Integer(128)]
        data.reduction_ranges = [Integer(256)]
        op = MagicMock(spec=ComputedBuffer)
        op.data = data
        op.loop_info = None
        _validate_reduction_tiling(op)  # must not raise

    def test_mixed_same_level_raises(self):
        """Both output and reduction dim tiled at the same level — Stage 2, raises."""
        from torch_spyre._inductor.coarse_tile import _validate_reduction_tiling

        op = self._make_op(loop_tiled_dims=[[0]], loop_tiled_reduction_dims=[[0]])
        with self.assertRaises(RuntimeError, msg="mixed same-level should raise"):
            _validate_reduction_tiling(op)

    def test_mixed_different_levels_allowed(self):
        """Outer output-dim tiling + inner reduction-dim tiling — now supported."""
        from torch._inductor.ir import ComputedBuffer, Reduction
        from torch_spyre._inductor.coarse_tile import _validate_reduction_tiling

        data = MagicMock(spec=Reduction)
        data.ranges = [Integer(128)]
        data.reduction_ranges = [Integer(256)]
        data.reduction_type = "sum"
        op = MagicMock(spec=ComputedBuffer)
        op.data = data
        op.get_name.return_value = "test_op"
        op.loop_info = CoarseTileInfo(
            loop_group_id=(0, 0),
            loop_count=[Integer(2), Integer(4)],
            loop_tiled_dims=[[0], []],
            loop_tiled_reduction_dims=[[], [0]],
        )
        # Must not raise: outer output-dim + inner reduction-dim is now supported.
        _validate_reduction_tiling(op)

    def test_multiple_reduction_dims_same_level_raises(self):
        """Multiple reduction dims tiled at one level — Stage 2, raises."""
        from torch._inductor.ir import ComputedBuffer, Reduction
        from torch_spyre._inductor.coarse_tile import _validate_reduction_tiling

        data = MagicMock(spec=Reduction)
        data.ranges = [Integer(128)]
        data.reduction_ranges = [Integer(64), Integer(64)]
        op = MagicMock(spec=ComputedBuffer)
        op.data = data
        op.get_name.return_value = "test_op"
        op.loop_info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(4)],
            loop_tiled_dims=[[]],
            loop_tiled_reduction_dims=[[0, 1]],
        )
        with self.assertRaises(
            RuntimeError, msg="multiple reduction dims should raise"
        ):
            _validate_reduction_tiling(op)

    def test_batchmatmul_k_tiling_allowed(self):
        """BATCH_MATMUL_OP tiling on the stick (K) dim is allowed — no Stage 2 error."""
        from torch._inductor.ir import ComputedBuffer, Reduction
        from torch_spyre._inductor.coarse_tile import _validate_reduction_tiling
        from torch_spyre._inductor.constants import BATCH_MATMUL_OP

        data = MagicMock(spec=Reduction)
        data.ranges = [Integer(64), Integer(32)]  # [M, N]
        data.reduction_ranges = [Integer(512)]  # [K]
        data.reduction_type = BATCH_MATMUL_OP
        op = MagicMock(spec=ComputedBuffer)
        op.data = data
        op.get_name.return_value = "test_matmul"
        op.loop_info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(4)],
            loop_tiled_dims=[[]],
            loop_tiled_reduction_dims=[[0]],
        )
        # Must not raise: BATCH_MATMUL_OP is exempt from the stick-dim guard.
        _validate_reduction_tiling(op)


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
            tiled_symbols=[[s]],
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


class TestCoarseTileInfoReductionField(unittest.TestCase):
    """CoarseTileInfo carries loop_tiled_reduction_dims parallel to loop_tiled_dims."""

    def test_field_present_and_defaults_to_empty(self):
        from torch_spyre._inductor.loop_info import CoarseTileInfo

        info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(4)],
            loop_tiled_dims=[[0]],
        )
        self.assertEqual(info.loop_tiled_reduction_dims, [])

    def test_field_can_be_set(self):
        from torch_spyre._inductor.loop_info import CoarseTileInfo

        info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(4)],
            loop_tiled_dims=[[]],
            loop_tiled_reduction_dims=[[0]],
        )
        self.assertEqual(info.loop_tiled_reduction_dims, [[0]])

    def test_nested_parallel_shape(self):
        """For a two-level nest, both fields have two sub-lists."""
        from torch_spyre._inductor.loop_info import CoarseTileInfo

        info = CoarseTileInfo(
            loop_group_id=(0, 0),
            loop_count=[Integer(2), Integer(4)],
            loop_tiled_dims=[[0], []],
            loop_tiled_reduction_dims=[[], [0]],
        )
        self.assertEqual(len(info.loop_tiled_dims), 2)
        self.assertEqual(len(info.loop_tiled_reduction_dims), 2)
        self.assertEqual(info.loop_tiled_reduction_dims[0], [])
        self.assertEqual(info.loop_tiled_reduction_dims[1], [0])


class TestDivideReductionRanges(unittest.TestCase):
    """_divide_reduction_ranges divides reduction_ranges, leaves ranges intact."""

    def _make_reduction_op(self, ranges, reduction_ranges, reduction_type="sum"):
        from torch._inductor.ir import ComputedBuffer, Reduction, ReductionHint
        import torch

        data = Reduction(
            device=torch.device("cpu"),
            dtype=torch.float16,
            inner_fn=lambda idx, ridx: None,
            ranges=list(ranges),
            reduction_ranges=list(reduction_ranges),
            reduction_type=reduction_type,
            src_dtype=torch.float16,
            reduction_hint=ReductionHint.DEFAULT,
        )
        op = MagicMock(spec=ComputedBuffer)
        op.data = data
        op.get_name.return_value = "test_op"
        return op

    def test_basic_halves_reduction_range(self):
        from torch_spyre._inductor.coarse_tile import _divide_reduction_ranges

        op = self._make_reduction_op(
            ranges=[Integer(128)], reduction_ranges=[Integer(256)]
        )
        _divide_reduction_ranges(op, Integer(2), [0])
        self.assertEqual(op.data.reduction_ranges[0], Integer(128))
        self.assertEqual(op.data.ranges[0], Integer(128))  # output ranges untouched

    def test_empty_tiled_dims_is_noop(self):
        from torch_spyre._inductor.coarse_tile import _divide_reduction_ranges

        op = self._make_reduction_op(
            ranges=[Integer(128)], reduction_ranges=[Integer(64)]
        )
        _divide_reduction_ranges(op, Integer(4), [])
        self.assertEqual(op.data.reduction_ranges[0], Integer(64))  # unchanged

    def test_not_divisible_raises(self):
        from torch_spyre._inductor.coarse_tile import _divide_reduction_ranges

        op = self._make_reduction_op(
            ranges=[Integer(128)], reduction_ranges=[Integer(100)]
        )
        with self.assertRaises(RuntimeError, msg="not divisible should raise"):
            _divide_reduction_ranges(op, Integer(3), [0])

    def test_divides_second_reduction_dim(self):
        from torch_spyre._inductor.coarse_tile import _divide_reduction_ranges

        op = self._make_reduction_op(
            ranges=[Integer(32)], reduction_ranges=[Integer(64), Integer(128)]
        )
        _divide_reduction_ranges(op, Integer(4), [1])
        self.assertEqual(op.data.reduction_ranges[0], Integer(64))  # untouched
        self.assertEqual(op.data.reduction_ranges[1], Integer(32))  # divided


class TestLoopVarToReductionRangesPos(unittest.TestCase):
    """_loop_var_to_reduction_ranges_pos finds the position of a symbol in reduction_ranges."""

    def _make_op_with_rw(self, out_syms, red_syms):
        """Return a mock ComputedBuffer whose get_read_writes() reflects the given symbols.

        out_syms: list of sympy.Symbol appearing in both the input and output index
        red_syms: list of sympy.Symbol appearing only in the input index (reduction dims)
        """
        from torch._inductor.ir import ComputedBuffer, Reduction
        from torch._inductor.dependencies import MemoryDep

        data = MagicMock(spec=Reduction)
        data.reduction_ranges = [Integer(64)] * len(red_syms)

        op = MagicMock(spec=ComputedBuffer)
        op.data = data
        op.get_name.return_value = "test_op"

        # Output dep: index contains only out_syms
        out_dep = MagicMock(spec=MemoryDep)
        out_dep.index = (
            sympy.Add(*out_syms)
            if len(out_syms) > 1
            else (out_syms[0] if out_syms else sympy.Integer(0))
        )
        out_dep.index = sympy.sympify(out_dep.index)

        # Input dep: index contains out_syms + red_syms; ranges preserves insertion order
        in_dep = MagicMock(spec=MemoryDep)
        all_syms = out_syms + red_syms
        in_dep.index = sympy.Add(*all_syms) if len(all_syms) > 1 else all_syms[0]
        in_dep.index = sympy.sympify(in_dep.index)
        # dict preserves insertion order in Python 3.7+ — out dims first, then red dims
        in_dep.ranges = {s: Integer(64) for s in all_syms}

        rw = MagicMock()
        rw.reads = [in_dep]
        rw.writes = iter([out_dep])
        # Make iter(rw.writes) work for next()
        out_dep_list = [out_dep]
        rw.writes = out_dep_list
        op.get_read_writes.return_value = rw
        return op, red_syms

    def test_finds_reduction_symbol(self):
        from torch_spyre._inductor.coarse_tile import _loop_var_to_reduction_ranges_pos

        i0 = sympy.Symbol("i0")
        r0 = sympy.Symbol("r0")
        op, red_syms = self._make_op_with_rw(out_syms=[i0], red_syms=[r0])
        result = _loop_var_to_reduction_ranges_pos(op, r0)
        self.assertEqual(result, 0)

    def test_returns_none_for_output_symbol(self):
        from torch_spyre._inductor.coarse_tile import _loop_var_to_reduction_ranges_pos

        i0 = sympy.Symbol("i0")
        r0 = sympy.Symbol("r0")
        op, _ = self._make_op_with_rw(out_syms=[i0], red_syms=[r0])
        result = _loop_var_to_reduction_ranges_pos(op, i0)
        self.assertIsNone(result)


class TestReductionIdentityValues(unittest.TestCase):
    """_reduction_identity_value returns the correct monoid identity per reduction type."""

    def _identity(self, reduction_type):
        from torch_spyre._inductor.coarse_tile import _reduction_identity_value
        import torch

        return _reduction_identity_value(reduction_type, torch.float16)

    def test_sum(self):
        self.assertEqual(self._identity("sum"), 0)

    def test_xor_sum(self):
        self.assertEqual(self._identity("xor_sum"), 0)

    def test_any(self):
        self.assertEqual(self._identity("any"), 0)

    def test_prod(self):
        self.assertEqual(self._identity("prod"), 1)

    def test_max(self):
        self.assertEqual(self._identity("max"), float("-inf"))

    def test_min(self):
        self.assertEqual(self._identity("min"), float("inf"))

    def test_unknown_raises(self):
        from torch_spyre._inductor.coarse_tile import _reduction_identity_value
        import torch

        with self.assertRaises(RuntimeError):
            _reduction_identity_value("welford_reduce", torch.float16)

    def test_batchmatmul(self):
        """BATCH_MATMUL_OP identity value is 0 — partial products are summed."""
        from torch_spyre._inductor.constants import BATCH_MATMUL_OP

        self.assertEqual(self._identity(BATCH_MATMUL_OP), 0)


# ===========================================================================
# TestReorderUnhintedInterlopers
# ===========================================================================


def _make_rui_op(name, reads=(), hint_ids=(), mutates=()):
    """Return a fake ComputedBuffer for reorder_unhinted_interlopers tests.

    ``reads`` is an iterable of buffer names this op reads.
    ``hint_ids`` is an iterable of hint-id integers; empty means unhinted.
    ``mutates`` is an iterable of buffer names this op mutates in-place.
    """
    from torch._inductor.ir import ComputedBuffer
    from torch_spyre._inductor.propagate_hints import DimHint

    op = MagicMock(spec=ComputedBuffer)
    op.get_name.return_value = name
    op.get_read_names.return_value = OrderedSet(reads)
    op.get_mutation_names.return_value = list(mutates)
    if hint_ids:
        op.dim_hints = [
            DimHint(
                dim_names=["d0"],
                split_count=1,
                loop_var=None,
                is_reduction=False,
                hint_id=hid,
            )
            for hid in hint_ids
        ]
    else:
        op.dim_hints = []
    return op


def _make_rui_non_computed(name):
    """Return a fake non-ComputedBuffer operation.

    Uses an unspec'd MagicMock so isinstance(..., ComputedBuffer) is False,
    which causes reorder_unhinted_interlopers to treat it as an immovable
    boundary that breaks any hint-group run.
    """
    op = MagicMock()
    op.get_name.return_value = name
    return op


class TestReorderUnhintedInterlopers(unittest.TestCase):
    """reorder_unhinted_interlopers moves unhinted ops out of hint-group runs."""

    def _run(self, ops):
        from torch_spyre._inductor.coarse_tile import reorder_unhinted_interlopers

        graph = SimpleNamespace(operations=list(ops))
        reorder_unhinted_interlopers(graph)
        return [op.get_name() for op in graph.operations]

    def test_no_ops(self):
        self.assertEqual(self._run([]), [])

    def test_all_hinted_unchanged(self):
        a = _make_rui_op("a", hint_ids=(0,))
        b = _make_rui_op("b", hint_ids=(0,))
        self.assertEqual(self._run([a, b]), ["a", "b"])

    def test_all_unhinted_unchanged(self):
        a = _make_rui_op("a")
        b = _make_rui_op("b")
        self.assertEqual(self._run([a, b]), ["a", "b"])

    def test_interloper_moved_before_run(self):
        # [hinted, unhinted, hinted] → [unhinted, hinted, hinted]
        # unhinted has no data deps; move before is preferred.
        a = _make_rui_op("a", hint_ids=(0,))
        x = _make_rui_op("x")  # interloper
        b = _make_rui_op("b", hint_ids=(0,))
        self.assertEqual(self._run([a, x, b]), ["x", "a", "b"])

    def test_interloper_blocked_move_before_reads_hinted(self):
        # x reads a's output → cannot move before a; try move-after.
        a = _make_rui_op("a", hint_ids=(0,))
        x = _make_rui_op("x", reads=("a",))
        b = _make_rui_op("b", hint_ids=(0,))
        self.assertEqual(self._run([a, x, b]), ["a", "b", "x"])

    def test_interloper_move_after_blocked_by_hinted_reader(self):
        # x reads a (blocks move-before) AND b reads x (blocks move-after) → error.
        a = _make_rui_op("a", hint_ids=(0,))
        x = _make_rui_op("x", reads=("a",))
        b = _make_rui_op("b", reads=("x",), hint_ids=(0,))
        c = _make_rui_op("c", hint_ids=(0,))
        with self.assertRaises(RuntimeError):
            self._run([a, x, b, c])

    def test_interloper_blocked_both_directions(self):
        # x reads a (blocks move-before) AND b reads x (blocks move-after) → error.
        a = _make_rui_op("a", hint_ids=(0,))
        x = _make_rui_op("x_out", reads=("a",))
        b = _make_rui_op("b", reads=("x_out",), hint_ids=(0,))
        with self.assertRaises(RuntimeError):
            self._run([a, x, b])

    def test_non_computed_buffer_breaks_run(self):
        # A non-ComputedBuffer between two hinted ops cannot be reordered.
        a = _make_rui_op("a", hint_ids=(0,))
        extern = _make_rui_non_computed("extern")
        b = _make_rui_op("b", hint_ids=(0,))
        self.assertEqual(self._run([a, extern, b]), ["a", "extern", "b"])

    def test_differently_hinted_breaks_run(self):
        # An op with a different hint_id is not a candidate for reordering.
        a = _make_rui_op("a", hint_ids=(0,))
        c = _make_rui_op("c", hint_ids=(1,))
        b = _make_rui_op("b", hint_ids=(0,))
        self.assertEqual(self._run([a, c, b]), ["a", "c", "b"])

    def test_multiple_interlopers_all_moveable_before(self):
        # [H, U1, U2, H] with no deps → both move before.
        a = _make_rui_op("a", hint_ids=(0,))
        x = _make_rui_op("x")
        y = _make_rui_op("y")
        b = _make_rui_op("b", hint_ids=(0,))
        self.assertEqual(self._run([a, x, y, b]), ["x", "y", "a", "b"])

    def test_multiple_interlopers_second_depends_on_first(self):
        # y reads x → x can move before, but then y reads x which is now
        # before the run start → y can also move before (after x).
        a = _make_rui_op("a", hint_ids=(0,))
        x = _make_rui_op("x")
        y = _make_rui_op("y", reads=("x",))
        b = _make_rui_op("b", hint_ids=(0,))
        result = self._run([a, x, y, b])
        # x moves before, then y (reads x, which is now before run_start)
        # — y's reads are not produced by any op in run_start..j-1 after x moved.
        self.assertEqual(result, ["x", "y", "a", "b"])

    def test_trailing_consumer_not_error(self):
        # Unhinted op after the run that reads run outputs — trailing consumer,
        # not an interloper.  No hinted ops follow it so it should not raise.
        a = _make_rui_op("a", hint_ids=(0,))
        b = _make_rui_op("b", hint_ids=(0,))
        x = _make_rui_op("x", reads=("a", "b"))
        self.assertEqual(self._run([a, b, x]), ["a", "b", "x"])

    def test_interloper_at_start_of_list(self):
        # Unhinted op before any hinted op — no run started yet, nothing to do.
        x = _make_rui_op("x")
        a = _make_rui_op("a", hint_ids=(0,))
        self.assertEqual(self._run([x, a]), ["x", "a"])

    def test_move_after_multiple_trailing_hinted(self):
        # [H, U, H, H] where U reads nothing and no one reads U:
        # move-before is legal and preferred over move-after.
        a = _make_rui_op("a", hint_ids=(0,))
        x = _make_rui_op("x")
        b = _make_rui_op("b", hint_ids=(0,))
        c = _make_rui_op("c", hint_ids=(0,))
        self.assertEqual(self._run([a, x, b, c]), ["x", "a", "b", "c"])

    def test_move_after_op_follows_run(self):
        # [H, U(reads H), H, H, V(reads U)] — move-before blocked (x reads a);
        # move-after should land x just after c, before d.
        # Catches the pop-then-insert off-by-one: insert must be at run_end-1
        # not run_end after the pop shifts indices.
        a = _make_rui_op("a", hint_ids=(0,))
        x = _make_rui_op("x", reads=("a",))
        b = _make_rui_op("b", hint_ids=(0,))
        c = _make_rui_op("c", hint_ids=(0,))
        d = _make_rui_op("d", reads=("x",))  # unhinted, reads x — after run
        self.assertEqual(self._run([a, x, b, c, d]), ["a", "b", "c", "x", "d"])

    def test_interloper_with_unhinted_gap_before_next_hinted(self):
        # [H, U(reads H), V(unhinted), H2] — run_end must span past V to H2.
        # Without this fix, run_end collapses to j+1=V and move-after is a
        # silent no-op that leaves U in place with no error.
        a = _make_rui_op("a", hint_ids=(0,))
        x = _make_rui_op("x", reads=("a",))  # blocked from moving before
        v = _make_rui_op("v")  # another unhinted op (no deps)
        a2 = _make_rui_op("a2", hint_ids=(0,))
        # v has no deps and moves before the run; x (reads a) cannot move before
        # but can move after a2 (run_end spans past v to a2).
        self.assertEqual(self._run([a, x, v, a2]), ["v", "a", "a2", "x"])

    def test_non_contiguous_run_multiple_interlopers(self):
        # [H, U1(reads H), H2, U2, H3] — U1 cannot move before (reads a);
        # move-after must span to H3 (the last same-key op), not just H2.
        # Without the fix U1 moves to between H2 and U2, still splitting [H3].
        # With the fix: u1 moves after c (run_end spans to c); u2 then moves
        # before the run; result is one contiguous hinted block [a, b, c].
        a = _make_rui_op("a", hint_ids=(0,))
        u1 = _make_rui_op("u1", reads=("a",))
        b = _make_rui_op("b", hint_ids=(0,))
        u2 = _make_rui_op("u2")
        c = _make_rui_op("c", hint_ids=(0,))
        self.assertEqual(self._run([a, u1, b, u2, c]), ["u2", "a", "b", "c", "u1"])

    def test_two_interlopers_both_move_after(self):
        # [H(a), U1(reads a), U2(reads a), H(b), H(c)]
        # U1 and U2 both read 'a' so neither can move before the run.
        # Both have no dependents in the remaining hinted ops so both can
        # move after.  After U1 moves after c, U2 is encountered next; it
        # also reads a (blocked from moving before) and can move after c.
        # Verifies the chained move-after path for consecutive interlopers.
        a = _make_rui_op("a", hint_ids=(0,))
        u1 = _make_rui_op("u1", reads=("a",))
        u2 = _make_rui_op("u2", reads=("a",))
        b = _make_rui_op("b", hint_ids=(0,))
        c = _make_rui_op("c", hint_ids=(0,))
        # u1 is processed first and moves after c; u2 is processed next and
        # also moves after c (now at index 3), landing between c and u1.
        self.assertEqual(self._run([a, u1, u2, b, c]), ["a", "b", "c", "u2", "u1"])

    def test_mutating_interloper_blocked(self):
        # x mutates buffer 'a' produced by a hinted op; x cannot legally move
        # before the run (would run before 'a' is produced) and b reads x so
        # x cannot move after — should raise RuntimeError.
        a = _make_rui_op("a", hint_ids=(0,))
        x = _make_rui_op("x", mutates=("a",))  # mutation dep on a
        b = _make_rui_op("b", reads=("x",), hint_ids=(0,))
        c = _make_rui_op("c", hint_ids=(0,))
        with self.assertRaises(RuntimeError):
            self._run([a, x, b, c])


# ===========================================================================
# TestHintsLevels
# ===========================================================================


class TestHintsLevels(unittest.TestCase):
    """_hints_levels must drop size-1 split_count hints as no-ops."""

    def _make_op(self, hints):
        """Return a fake ComputedBuffer with the given DimHint list.

        hints: list of (hint_id, split_count, loop_var) tuples.
        """
        from torch._inductor.ir import ComputedBuffer
        from torch_spyre._inductor.propagate_hints import DimHint

        op = MagicMock(spec=ComputedBuffer)
        op.get_name.return_value = "buf0"
        op.dim_hints = [
            DimHint(
                dim_names=[f"dim{i}"],
                split_count=sc,
                loop_var=lv,
                is_reduction=False,
                hint_id=hid,
            )
            for i, (hid, sc, lv) in enumerate(hints)
        ]
        return op

    def test_size1_hint_dropped(self):
        """A single hint with split_count=1 produces an empty levels list."""
        import sympy
        from torch_spyre._inductor.coarse_tile import _hints_levels

        op = self._make_op([(0, 1, sympy.Symbol("c0"))])
        self.assertEqual(_hints_levels([op]), [])

    def test_size1_hint_dropped_with_debug_log(self):
        """A size-1 hint emits a debug log message when dropped."""
        import logging
        import logging.handlers
        import sympy
        import torch_spyre._inductor.coarse_tile as ct_mod
        from torch_spyre._inductor.coarse_tile import _hints_levels

        op = self._make_op([(7, 1, sympy.Symbol("c0"))])

        original_level = ct_mod.hints_logger.level
        ct_mod.hints_logger.setLevel(logging.DEBUG)
        handler = logging.handlers.MemoryHandler(
            capacity=100, flushLevel=logging.CRITICAL
        )
        ct_mod.hints_logger.addHandler(handler)
        try:
            result = _hints_levels([op])
            handler.flush()
            messages = [r.getMessage() for r in handler.buffer]
        finally:
            ct_mod.hints_logger.removeHandler(handler)
            ct_mod.hints_logger.setLevel(original_level)

        self.assertEqual(result, [])
        self.assertTrue(
            any("split_count=1" in m and "no-op" in m for m in messages),
            f"Expected a 'split_count=1 … no-op' debug message; got: {messages}",
        )

    def test_nonunit_hint_kept(self):
        """A hint with split_count > 1 is retained normally."""
        import sympy
        from torch_spyre._inductor.coarse_tile import _hints_levels

        c0 = sympy.Symbol("c0")
        op = self._make_op([(3, 4, c0)])
        levels = _hints_levels([op])
        self.assertEqual(len(levels), 1)
        hint_id, count, is_reduction = levels[0]
        self.assertEqual(hint_id, 3)
        self.assertEqual(count, sympy.Integer(4))
        self.assertFalse(is_reduction)

    def test_mixed_hints_drops_only_size1(self):
        """When one hint is size-1 and another is size>1, only the size>1 survives."""
        import sympy
        from torch_spyre._inductor.coarse_tile import _hints_levels

        c0, c1 = sympy.Symbol("c0"), sympy.Symbol("c1")
        op = self._make_op([(0, 1, c0), (1, 8, c1)])
        levels = _hints_levels([op])
        self.assertEqual(len(levels), 1)
        hint_id, count, _ = levels[0]
        self.assertEqual(hint_id, 1)
        self.assertEqual(count, sympy.Integer(8))

    def test_all_size1_hints_dropped_falls_through_to_next_op(self):
        """If every hint on op0 is size-1, _hints_levels tries op1 next."""
        import sympy
        from torch_spyre._inductor.coarse_tile import _hints_levels

        c0 = sympy.Symbol("c0")
        op0 = self._make_op([(0, 1, c0)])
        op1 = self._make_op([(0, 4, c0)])
        levels = _hints_levels([op0, op1])
        self.assertEqual(len(levels), 1)
        _, count, _ = levels[0]
        self.assertEqual(count, sympy.Integer(4))


# ===========================================================================
# TestHintsToCoarseTileGroupsLogging
# ===========================================================================


def _make_htctg_op(name, hints):
    """Return a fake ComputedBuffer for hints_to_coarse_tile_groups logging tests.

    hints: list of (hint_id, dim_names, split_count, loop_var) tuples.
    loop_var may be None to simulate an op that is broadcast on that dim.
    """
    from torch._inductor.ir import ComputedBuffer
    from torch_spyre._inductor.propagate_hints import DimHint

    op = MagicMock(spec=ComputedBuffer)
    op.get_name.return_value = name
    op.get_operation_name.return_value = name
    op.origins = []
    op.dim_hints = [
        DimHint(
            dim_names=dim_names,
            split_count=split_count,
            loop_var=loop_var,
            is_reduction=False,
            hint_id=hint_id,
        )
        for hint_id, dim_names, split_count, loop_var in hints
    ]
    return op


def _run_htctg_and_capture_log(ops):
    """Run hints_to_coarse_tile_groups with INFO logging and return the log text."""
    import logging
    import logging.handlers
    from types import SimpleNamespace
    from torch_spyre._inductor.coarse_tile import hints_to_coarse_tile_groups
    import torch_spyre._inductor.coarse_tile as coarse_tile_mod

    graph = SimpleNamespace(operations=list(ops))

    # Temporarily force the module-level hints_logger to INFO so the logging
    # block inside hints_to_coarse_tile_groups actually runs.
    original_level = coarse_tile_mod.hints_logger.level
    coarse_tile_mod.hints_logger.setLevel(logging.INFO)

    handler = logging.handlers.MemoryHandler(capacity=1000, flushLevel=logging.CRITICAL)
    coarse_tile_mod.hints_logger.addHandler(handler)
    try:
        hints_to_coarse_tile_groups(graph)
        handler.flush()
        return "\n".join(r.getMessage() for r in handler.buffer)
    finally:
        coarse_tile_mod.hints_logger.removeHandler(handler)
        coarse_tile_mod.hints_logger.setLevel(original_level)


class TestHintsToCoarseTileGroupsLogging(unittest.TestCase):
    """The scopes= log line must list all hint dims, not just those with
    loop_var set on the first op in the group.

    Regression test for a bug where group_ops[0] had loop_var=None for a hint
    (e.g. a restickify op that doesn't iterate over Lq), causing that hint to
    be absent from group_levels and therefore omitted from the scopes= line.
    """

    def test_scopes_includes_all_hints_when_first_op_is_broadcast_on_second_hint(self):
        """When group_ops[0] has loop_var=None for hint 2 (Lq), the scopes= line
        must still include Lq — not just H."""
        import sympy

        h_sym = sympy.Symbol("c0")
        lq_sym = sympy.Symbol("c1")

        # op0: iterates over H only — loop_var=None for Lq (broadcast, like restickify)
        op0 = _make_htctg_op(
            "op0",
            [
                (1, ["H"], 8, h_sym),  # hint_id=1, H, has loop_var
                (2, ["Lq"], 4, None),  # hint_id=2, Lq, loop_var=None → broadcast
            ],
        )
        # op1: iterates over both H and Lq
        op1 = _make_htctg_op(
            "op1",
            [
                (1, ["H"], 8, h_sym),
                (2, ["Lq"], 4, lq_sym),
            ],
        )

        log_output = _run_htctg_and_capture_log([op0, op1])

        # Find the scopes= line specifically
        scopes_line = next(
            (ln for ln in log_output.splitlines() if "scopes=" in ln), ""
        )
        self.assertIn("H", scopes_line, f"scopes= must mention H; got: {scopes_line!r}")
        self.assertIn(
            "Lq",
            scopes_line,
            f"scopes= must mention Lq even though op0 is broadcast on Lq "
            f"(loop_var=None for hint_id=2 on group_ops[0]); "
            f"got: {scopes_line!r}",
        )


if __name__ == "__main__":
    unittest.main()
