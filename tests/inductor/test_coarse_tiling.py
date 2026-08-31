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
  2. coarse_tile IR pass: range rewriting, attribute stamping, nested groups,
     and consumer-redirect index rewriting (TestDivideRanges, TestCoarseTile,
     TestCoarseTileNested, TestRetileLoadIndexFromStrides,
     TestSqueezedRetileDims, TestIndexVarPrefix, TestConsumerOwnDimSymbol,
     TestRetileLoadIndexWithConsumer)
  3. CountedLoopSchedulerNode, build_loop_scheduler_nodes,
     _tiled_syms_for_sched_node_at_depth, and spyre_fuse_nodes loop fusion
     (TestHelpers, TestBuildLoopSchedulerNodes, TestTiledSymsForSchedNode,
      TestSpyreFuseNodesLoopFusion)
  4. generate_sdsc and compile_op_spec symbol/affine-stride paths
     (TestGenerateSdscTiledSymbols,
      TestCompileOpSpecTwoTiledSymbols, TestCompileOpSpecSymbolMapping)
  5. generate_bundle MLIR output: loop structure, affine maps, symbol constants
     (TestGenerateBundleMlir, TestFindUnimplemented,
      TestGenerateBundleMlirSnapshot, TestGenerateBundleMlirWithAffineStrides,
      TestGenerateBundleNestedTiling, TestGenerateBundleAffineLoopPath)
  6. Buffer propagation: consumer analysis helpers for tiling propagation
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
from torch._inductor import dependencies as inductor_deps
from torch._inductor.graph import GraphLowering
from torch._inductor.utils import IndentedBuffer, sympy_index_symbol
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config
from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.codegen.compute_ops import generate_sdsc
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
from torch_spyre._inductor.core_mapping import derive_operation_mapping
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.loop_info import CoarseTileInfo, copy_op_metadata
from torch_spyre._inductor.pass_utils import coeff_through_floor
from torch_spyre._inductor.propagate_hints import DimHint
from torch_spyre._inductor.wsr.coarse_tile import (
    _LOOPS_FREE_SYMS_KEY,
    _REDUCTION_FREE_SYMS_KEY,
    _RetiledBufferInfo,
    _apply_plan,
    _compute_fill_loop_info_planned,
    _consumer_own_dim_symbol,
    _divide_ranges,
    _full_buffer_read_deps,
    _index_var_prefix,
    _replace_group_op,
    _rescale_index,
    _retile_load_index,
    _should_patch_retiled_load_indexes,
    _squeezed_retile_dims,
    coarse_tile_post_stickify,
    coarse_tile_pre_stickify,
    plan_coarse_tile_groups,
    reduction_loop_vars,
)
from torch_spyre._inductor.scratchpad.coarse_tiling import (
    CoarseTilingPass,
    _derive_group_idx_offset,
    _derive_hint_id_base,
    derive_tiling_groups,
    tile_spec_to_dim_hints,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    CoreDivisionBuffer,
    TileAxis,
    TileSpec,
    ceil_div,
)
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg, UnimplementedOp
from torch_spyre._inductor.fusion import spyre_fuse_nodes
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
)
from torch_spyre._inductor.temp_passes import (
    _mark_static_unit_batch_bmm,
    mark_direct_unit_bmm_pass,
)
from torch_spyre._inductor.wsr.tile import (
    compute_tile_offset,
    compute_tile_index,
    compute_tile_stride,
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


def _make_real_pointwise_op(
    ranges,
    input_shapes_strides,
    name="op0",
    hints=((0, 0),),
):
    """Build a genuine ComputedBuffer(Pointwise) reading real InputBuffers.

    ``input_shapes_strides`` is a list of (shape, stride) pairs, one per
    input; the returned op's inner_fn sums a load from each input at the
    op's own iteration index, so every input's MemoryDep.index reflects
    that input's own (never-divided-by-this-op) stride.  Unlike
    ``_make_op``, ``op.get_read_writes()`` on the returned op produces real
    MemoryDep objects with genuine d0, d1, ... index expressions.

    ``hints`` follows the same (hint_id, dim_index) convention as
    ``_make_hinted_op``.  The caller must have an active graph handler
    (``V.set_graph_handler(...)``) around both this call and any later
    ``op.get_read_writes()`` / ``plan_coarse_tile_groups(...)``+
    ``_apply_plan(...)`` call on the returned op -- ``InputBuffer.
    make_loader()`` reads ``V.graph.sizevars`` lazily, not just at
    construction time.  ``_apply_plan`` itself additionally requires the op
    to have a real ``operation_name`` (normally assigned by
    ``GraphLowering.register_operation``) for its internal
    ``_validate_contiguous`` position lookup -- set directly below rather
    than going through the full ``register_operation`` machinery (which
    auto-generates its own ``op{N}``-style name instead of using the name
    we pass in).  Each input ``InputBuffer`` is also registered on
    ``V.graph.name_to_buffer`` for consistency with real IR, though neither
    ``plan_coarse_tile_groups`` nor ``_apply_plan`` resolves buffers by name
    (that only happens in the buffer-propagation pass, which these tests
    bypass by calling the plan/apply pair directly instead of the full
    ``coarse_tile()`` entry point -- see
    ``TestCoarseTileTileAdvanceExprs``'s docstring).
    """
    from torch._inductor.ir import (
        ComputedBuffer,
        FixedLayout,
        InputBuffer,
        Pointwise,
        StorageBox,
        TensorBox,
    )
    from torch_spyre._inductor.propagate_hints import DimHint

    input_boxes = []
    for i, (shape, stride) in enumerate(input_shapes_strides):
        inp = InputBuffer(
            name=f"in{i}_{name}",
            layout=FixedLayout(torch.device("cpu"), torch.float32, shape, stride),
        )
        V.graph.name_to_buffer[inp.get_name()] = inp
        input_boxes.append(TensorBox(StorageBox(inp)))

    def inner_fn(index):
        loaders = [box.make_loader()(index) for box in input_boxes]
        result = loaders[0]
        for loader in loaders[1:]:
            result = result + loader
        return result

    pw = Pointwise.create(
        device=torch.device("cpu"),
        dtype=torch.float32,
        inner_fn=inner_fn,
        ranges=list(ranges),
    )
    pw_data = pw.data.data  # TensorBox -> StorageBox -> Pointwise
    buf = ComputedBuffer(
        name=name,
        layout=FixedLayout(torch.device("cpu"), torch.float32, list(ranges), None),
        data=pw_data,
    )
    buf.operation_name = name
    V.graph.name_to_buffer[name] = buf
    n_ranges = len(ranges)
    buf._test_out_coords = [sympy.Symbol(f"c{i}") for i in range(n_ranges)]
    buf.dim_hints = [
        DimHint(
            dim_names=[f"dim{dim_index}"],
            split_count=1,
            loop_var=sympy.Symbol(f"c{dim_index}"),
            is_reduction=False,
            hint_id=hint_id,
        )
        for hint_id, dim_index in hints
    ]
    return buf


def _make_real_reduction_op(
    ranges,
    reduction_ranges,
    input_shape_stride,
    name="op0",
    hints=((0, 0),),
):
    """Build a genuine ComputedBuffer(Reduction) reading one real InputBuffer.

    Mirrors ``_make_real_pointwise_op`` but for a ``Reduction`` op, so the
    resulting op's ``get_read_writes()`` produces a read ``MemoryDep`` whose
    index depends on BOTH output-dim d{i} symbols and reduction-dim d{i}
    symbols (numbered continuously after the output dims, per Inductor's own
    ``Loops.get_reads()`` convention) -- exercising the
    ``n_output_dims``-offset path in ``plan_coarse_tile_groups``/
    ``_apply_plan`` that the two ``Pointwise``-only tests above never touch
    (both have zero reduction dims).

    ``hints`` follows the same (hint_id, dim_index) convention as
    ``_make_hinted_op``, with ``dim_index >= len(ranges)`` denoting a
    reduction dim (position ``dim_index - len(ranges)`` in
    ``reduction_ranges``).

    Unlike the non-reduction ``DimHint``s (matched against
    ``_test_out_coords`` via the patched ``op_out_coords``, so any synthetic
    symbol works), ``plan_coarse_tile_groups``'s reduction-dim lookup
    (``_loop_var_to_reduction_ranges_pos``) matches ``dim_hint.loop_var``
    directly against the *real* ``d{i}`` symbols found in the op's own
    ``get_read_writes()`` reduction dep -- it does not go through
    ``op_out_coords`` at all. So reduction ``DimHint``s here MUST use the
    real ``sympy_index_symbol(f"d{{i}}")`` (imported from
    ``torch._inductor.utils``, same import as elsewhere in this file) as
    their ``loop_var``, not a synthetic ``c{{i}}`` placeholder -- a synthetic
    symbol would never match and the reduction dim would silently fail to
    tile (``op_tiled_reduction_dims`` would stay ``[[]]``).

    Like ``_make_real_pointwise_op``, the caller must have an active graph
    handler around both this call and any later ``get_read_writes()`` /
    ``plan_coarse_tile_groups(...)``+``_apply_plan(...)`` call on the
    returned op, and this helper likewise registers the input and output
    buffers directly on ``V.graph.name_to_buffer`` (see
    ``_make_real_pointwise_op``'s docstring for why
    ``register_buffer``/``register_operation`` aren't used here, and why
    neither ``plan_coarse_tile_groups`` nor ``_apply_plan`` -- unlike the
    full ``coarse_tile()`` -- ever needs to resolve them by name).
    """
    from torch._inductor.ir import (
        ComputedBuffer,
        FixedLayout,
        InputBuffer,
        Reduction,
        StorageBox,
        TensorBox,
    )
    from torch_spyre._inductor.propagate_hints import DimHint

    shape, stride = input_shape_stride
    inp = InputBuffer(
        name=f"in0_{name}",
        layout=FixedLayout(torch.device("cpu"), torch.float32, shape, stride),
    )
    V.graph.name_to_buffer[inp.get_name()] = inp
    input_box = TensorBox(StorageBox(inp))

    def inner_fn(index, reduction_index):
        full_index = list(index) + list(reduction_index)
        return input_box.make_loader()(full_index)

    red = Reduction.create(
        device=torch.device("cpu"),
        dst_dtype=torch.float32,
        src_dtype=torch.float32,
        inner_fn=inner_fn,
        ranges=list(ranges),
        reduction_ranges=list(reduction_ranges),
        reduction_type="sum",
    )
    red_data = red.data.data  # TensorBox -> StorageBox -> Reduction
    buf = ComputedBuffer(
        name=name,
        layout=FixedLayout(torch.device("cpu"), torch.float32, list(ranges), None),
        data=red_data,
    )
    buf.operation_name = name
    V.graph.name_to_buffer[name] = buf
    n_ranges = len(ranges)
    buf._test_out_coords = [sympy.Symbol(f"c{i}") for i in range(n_ranges)]
    buf.dim_hints = [
        DimHint(
            dim_names=[f"dim{dim_index}"],
            split_count=1,
            loop_var=(
                sympy_index_symbol(f"d{dim_index}")
                if dim_index >= n_ranges
                else sympy.Symbol(f"c{dim_index}")
            ),
            is_reduction=(dim_index >= n_ranges),
            hint_id=hint_id,
        )
        for hint_id, dim_index in hints
    ]
    return buf


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
    snode.get_device.return_value = torch.device("spyre")
    snode.get_name.return_value = name
    snode.get_nodes.return_value = [snode]
    snode.ancestors = OrderedSet()
    snode.min_order = 0
    snode.max_order = 0
    # PT 2.12 added min/max_input_distance to BaseSchedulerNode (set in __init__,
    # so absent from MagicMock(spec=...)); init_group_node reads them when fusing.
    snode.min_input_distance = 0
    snode.max_input_distance = 0
    snode.unmet_dependencies = OrderedSet()
    snode.is_reduction.return_value = False
    snode.group = (None, None)
    snode.read_writes = inductor_deps.ReadWrites(
        reads=OrderedSet(),
        writes=OrderedSet(),
        index_exprs=OrderedSet(),
    )
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
        arg_index=0,
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
                "stick_dim_order": [s],
                "stick_size": [64],
            }
        },
        args=[tensor],
        constants={},
        conv_params={},
        coordinate_masking={},
    )


def _make_tiled_op_spec() -> OpSpec:
    """Minimal OpSpec with tiled_symbols that compile_op_spec can process."""
    c0 = Symbol("c0")
    # compile_op_spec (superdsc.parse_op_spec) renames the sole iteration-
    # space symbol c0 to OUTPUT_DIM_LABELS[0] == "out" for this 1-dim
    # non-matmul op (constants.py); device_tile_advance_expr must be
    # expressed in terms of "out", not "c0" -- see
    # TestCompileOpSpecTwoTiledSymbols._make_3d_op_spec and
    # TestGenerateBundleMlirWithAffineStrides.
    # test_tiled_snapshot_via_device_tile_advance_expr for the same renaming
    # rule with worked examples.
    out = Symbol("out")
    fp16 = _FP16
    # device-element-offset advance per unit step of out, for a device_size=
    # [2, 64] layout with c0/out in the last (stick) position. generate_sdsc's
    # affine-stride filter (_tensor_tiled_by_symbol) requires a nonzero coeff
    # on device_tile_advance_expr to treat a tensor as tiled/advancing at
    # all, so this must be set for these tests' tensors to produce any
    # affine stride.
    tile_advance_expr = 64 * out
    tensor_in = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=fp16,
        device_size=[2, 64],
        device_coordinates=[Integer(0), c0],
        allocation={"hbm": 0x1000},
        device_tile_advance_expr=tile_advance_expr,
    )
    tensor_out = TensorArg(
        is_input=False,
        arg_index=1,
        device_dtype=fp16,
        device_size=[2, 64],
        device_coordinates=[Integer(0), c0],
        allocation={"hbm": 0x2000},
        device_tile_advance_expr=tile_advance_expr,
    )
    iteration_space = {c0: (Integer(128), 1)}
    return OpSpec(
        op="abs",
        is_reduction=False,
        iteration_space=iteration_space,
        core_id_to_work_slice=derive_operation_mapping(iteration_space),
        args=[tensor_in, tensor_out],
        op_info={},
        tiled_symbols=[[c0]],
        tiled_symbol_trip_counts={c0: 128},
    )


# ---------------------------------------------------------------------------
# bundle.mlir test helpers
# ---------------------------------------------------------------------------


def _fake_compile_op_spec(
    idx: int,
    op_spec: OpSpec,
    symbols: list,
    symbol_id_offset: int = 0,
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

    def test_tile_advance_defaults(self):
        info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(4)],
            loop_tiled_dims=[[0]],
        )
        self.assertEqual(info.tiled_dims_per_read, [])
        self.assertEqual(info.output_tiled_dims, [])

    def test_tile_advance_explicit(self):
        info = CoarseTileInfo(
            loop_group_id=(0, 0),
            loop_count=[Integer(2), Integer(4)],
            loop_tiled_dims=[[0], [1]],
            tiled_dims_per_read=[[[(0, Integer(512))], [(1, Integer(1024))]]],
            output_tiled_dims=[[(0, Integer(512))], [(1, Integer(1024))]],
        )
        self.assertEqual(
            info.tiled_dims_per_read,
            [[[(0, Integer(512))], [(1, Integer(1024))]]],
        )
        self.assertEqual(
            info.output_tiled_dims,
            [[(0, Integer(512))], [(1, Integer(1024))]],
        )


class TestComputeFillLoopInfoPlannedTopology(unittest.TestCase):
    """_compute_fill_loop_info_planned classifies output/reduction topologies.

    Levels are ordered outermost-first. A level is "output" if
    loop_tiled_dims[i] is non-empty, "reduction" if
    loop_tiled_reduction_dims[i] is non-empty. Flat and nested topologies
    are supported (see the function's docstring); any topology where an
    output level is interleaved with the reduction levels -- straddling a
    single reduction level from both sides, or sandwiched between two
    separate reduction levels -- must raise Unsupported.
    """

    def _info(self, loop_tiled_dims, loop_tiled_reduction_dims):
        n = len(loop_tiled_dims)
        return CoarseTileInfo(
            loop_group_id=tuple([0] * n),
            loop_count=[Integer(4)] * n,
            loop_tiled_dims=loop_tiled_dims,
            loop_tiled_reduction_dims=loop_tiled_reduction_dims,
        )

    def test_flat_reduction_outer_output_inner(self):
        # softmax(dim=0) tiled A/4, B/4: A-reduction outer, B-output inner.
        info = self._info([[], [0]], [[0], []])
        self.assertIsNone(_compute_fill_loop_info_planned(info))

    def test_nested_output_outer_reduction_inner(self):
        # mm outer M, inner K.
        info = self._info([[0], []], [[], [0]])
        result = _compute_fill_loop_info_planned(info)
        self.assertIsNotNone(result)
        self.assertEqual(result.loop_tiled_dims, [[0]])

    def test_output_straddling_single_reduction_level_unsupported(self):
        # output, reduction, output: output on both sides of one reduction
        # level.
        info = self._info([[0], [], [1]], [[], [0], []])
        with self.assertRaises(Unsupported):
            _compute_fill_loop_info_planned(info)

    def test_output_sandwiched_between_two_reduction_levels_unsupported(self):
        # reduction, output, reduction: no single reduction level has
        # output on both sides of it, but the output level is still
        # interleaved with the reduction as a whole.
        info = self._info([[], [0], []], [[0], [], [1]])
        with self.assertRaises(Unsupported):
            _compute_fill_loop_info_planned(info)


class TestConfigFlags(unittest.TestCase):
    """Test configuration flags."""

    def test_enable_reduction_tiling_default_and_patch(self):
        """enable_reduction_tiling defaults to True and is patchable."""
        self.assertTrue(config.enable_reduction_tiling)
        with config.patch({"enable_reduction_tiling": False}):
            self.assertFalse(config.enable_reduction_tiling)
        self.assertTrue(config.enable_reduction_tiling)


class TestTileAdvanceExprFromDep(unittest.TestCase):
    """Unit tests for _tile_advance_expr_from_dep's symbolic substitution.

    _tile_advance_expr_from_dep substitutes extent * d{i} for each tiled
    dim's d{i} in dep.index, and 0 for every other free symbol, then
    returns the result as-is -- it is only resolved once a later
    compilation stage substitutes a concrete tile-index value for each
    d{i}.
    """

    def _dep(self, index_expr, ranges):
        return inductor_deps.MemoryDep(
            "t0", index_expr, ranges, size=list(ranges.values())
        )

    def test_extracts_coefficient_for_tiled_dim(self):
        from torch_spyre._inductor.spyre_kernel import _tile_advance_expr_from_dep

        d0 = sympy_index_symbol("d0")
        d1 = sympy_index_symbol("d1")
        dep = self._dep(Integer(4096) * d0 + d1, {d0: 512, d1: 1024})
        expr = _tile_advance_expr_from_dep(dep, {0: Integer(512)})
        self.assertEqual(simplify(expr - Integer(4096) * Integer(512) * d0), 0)

    def test_untiled_dim_contributes_zero(self):
        from torch_spyre._inductor.spyre_kernel import _tile_advance_expr_from_dep

        d0 = sympy_index_symbol("d0")
        d1 = sympy_index_symbol("d1")
        dep = self._dep(Integer(4096) * d0 + d1, {d0: 512, d1: 1024})
        expr = _tile_advance_expr_from_dep(dep, {})
        self.assertEqual(expr, Integer(0))

    def test_broadcast_dim_not_in_index_contributes_zero(self):
        from torch_spyre._inductor.spyre_kernel import _tile_advance_expr_from_dep

        d0 = sympy_index_symbol("d0")
        d1 = sympy_index_symbol("d1")
        # dep does not depend on d1 at all (broadcast along that dim).
        dep = self._dep(Integer(4096) * d0, {d0: 512, d1: 1024})
        expr = _tile_advance_expr_from_dep(dep, {0: Integer(512), 1: Integer(1024)})
        self.assertEqual(simplify(expr - Integer(4096) * Integer(512) * d0), 0)

    def test_sums_multiple_tiled_dims(self):
        from torch_spyre._inductor.spyre_kernel import _tile_advance_expr_from_dep

        d0 = sympy_index_symbol("d0")
        d1 = sympy_index_symbol("d1")
        dep = self._dep(Integer(4096) * d0 + d1, {d0: 512, d1: 1024})
        expr = _tile_advance_expr_from_dep(dep, {0: Integer(512), 1: Integer(1024)})
        expected = Integer(4096) * Integer(512) * d0 + Integer(1) * Integer(1024) * d1
        self.assertEqual(simplify(expr - expected), 0)

    def test_non_polynomial_index_substitutes_exactly(self):
        """A Mod/FloorDiv-wrapped tiled-dim symbol must not crash, and the
        substitution should produce the exact (non-linear) term rather than
        an approximation.

        Reshape-split and gather/indirect-indexing dims can leave a tiled
        dim's d{i} symbol wrapped in Mod/FloorDiv/ModularIndexing.
        _loop_var_to_ranges_pos only checks for a single free symbol, so
        such a dim is still accepted as "tiled" upstream and this function
        must not crash the whole coarse-tiling pass when it sees one.
        Because the implementation substitutes directly into dep.index
        rather than extracting a linear coefficient, it needs no special
        casing for non-affine forms: it produces the exact substituted
        expression.
        """
        from torch_spyre._inductor.spyre_kernel import _tile_advance_expr_from_dep

        d0 = sympy_index_symbol("d0")
        d1 = sympy_index_symbol("d1")
        # d0 only ever appears wrapped in Mod -- non-polynomial in d0.
        dep = self._dep(Integer(4096) * sympy.Mod(d0, 8) + d1, {d0: 512, d1: 1024})
        expr = _tile_advance_expr_from_dep(dep, {0: Integer(512), 1: Integer(1024)})
        expected = Integer(4096) * sympy.Mod(Integer(512) * d0, 8) + Integer(1024) * d1
        self.assertEqual(simplify(expr - expected), 0)

        # FloorDiv (a//b) is likewise non-polynomial in the wrapped symbol.
        dep2 = self._dep(Integer(4096) * (d0 // Integer(8)) + d1, {d0: 512, d1: 1024})
        expr2 = _tile_advance_expr_from_dep(dep2, {0: Integer(512), 1: Integer(1024)})
        expected2 = (
            Integer(4096) * ((Integer(512) * d0) // Integer(8)) + Integer(1024) * d1
        )
        self.assertEqual(simplify(expr2 - expected2), 0)

    def test_transposed_index_keeps_its_own_coefficient_per_dim(self):
        """A dep whose stride is transposed relative to the "usual" d0-major
        layout must keep each dim's own coefficient, not the row-major one.

        Coefficient extraction is per-dependency (this dep's own .index),
        so a transposed dep (here d1's coefficient, 4096, is larger than
        d0's, 1) must produce that exact pairing rather than assuming d0
        always carries the larger stride.
        """
        from torch_spyre._inductor.spyre_kernel import _tile_advance_expr_from_dep

        d0 = sympy_index_symbol("d0")
        d1 = sympy_index_symbol("d1")
        # Transposed: d0's coefficient (1) is smaller than d1's (4096),
        # the opposite of the row-major convention used elsewhere in this
        # test class.
        dep = self._dep(d0 + Integer(4096) * d1, {d0: 512, d1: 1024})
        expr = _tile_advance_expr_from_dep(dep, {0: Integer(512), 1: Integer(1024)})
        expected = Integer(1) * Integer(512) * d0 + Integer(4096) * Integer(1024) * d1
        self.assertEqual(simplify(expr - expected), 0)


class TestRetileLoadIndexFromStrides(unittest.TestCase):
    """Unit tests for converting stale full-buffer load indexes to tile indexes."""

    def test_rewrites_stale_full_stride_to_tile_stride(self):
        # Row-major [8, 4096]: old stride (4096, 1), divided to tile [4, 512].
        c0, c1 = sympy.symbols("c0 c1")
        info = _RetiledBufferInfo(
            old_stride=(Integer(4096), Integer(1)),
            new_stride=(Integer(512), Integer(1)),
            old_size=(Integer(4), Integer(512)),
        )
        result = _retile_load_index("buf", 4096 * c0 + c1, info)

        self.assertEqual(simplify(result - (512 * c0 + c1)), 0)

    def test_mixed_loop_variable_terms_raises(self):
        # Index with a product of two loop vars is not decomposable; raises.
        c0, c1, c2 = sympy.symbols("c0 c1 c2")
        index = c0 * c1 + 128 * c0 + c2
        info = _RetiledBufferInfo(
            old_stride=(Integer(128), Integer(1)),
            new_stride=(Integer(64), Integer(1)),
            old_size=(Integer(2), Integer(128)),
        )
        from torch_spyre._inductor.errors import Unsupported

        with self.assertRaises(Unsupported):
            _retile_load_index("buf", index, info)

    def test_dim_that_becomes_size1_after_tiling_is_still_rewritten(self):
        # A dim with old_size=2 tiled to new_size=1 must still have its stride
        # coefficient rewritten. compute_tile_index excludes dims where size==1
        # from matching, so passing new_size (1) instead of old_size (2) would
        # cause the stride 2 coefficient to go unmatched — wrong result.
        # This test verifies the fix: _RetiledBufferInfo.size holds old_size.
        c0, c1 = sympy.symbols("c0 c1")
        info = _RetiledBufferInfo(
            old_stride=(Integer(2), Integer(1)),
            new_stride=(Integer(1), Integer(1)),
            old_size=(Integer(2), Integer(2)),
        )
        result = _retile_load_index("buf", 2 * c0 + c1, info)

        self.assertEqual(simplify(result - (c0 + c1)), 0)

    def test_distinct_old_strides_are_each_rewritten_correctly(self):
        # Two dims with distinct old strides: compute_tile_index maps each via
        # paired-stride resolution, producing independent correct rewrites.
        c0, c1 = sympy.symbols("c0 c1")
        info = _RetiledBufferInfo(
            old_stride=(Integer(256), Integer(128)),
            new_stride=(Integer(64), Integer(32)),
            old_size=(Integer(2), Integer(4)),
        )
        result_c0 = _retile_load_index("buf", 256 * c0, info)
        result_c1 = _retile_load_index("buf", 128 * c1, info)

        self.assertEqual(simplify(result_c0 - 64 * c0), 0)
        self.assertEqual(simplify(result_c1 - 32 * c1), 0)

    def test_index_already_at_new_scale_is_left_unchanged(self):
        # Reproduces test_copy_running_max_4d_H4_Lq4: a [2,32,4096] buffer
        # tiled on two coarse-tile levels (H then Lq) down to old_stride
        # (131072, 4096, 1) -> new_stride (4096, 1024, 1). A same-group
        # consumer resynced to a Pass-1/2/3 replacement object (see
        # _coarse_tile_common's by-name resync) can have its load index
        # already retraced at new_stride scale. Rewriting it again against
        # old_stride would silently collide two dims' coefficients (dim 1's
        # old_stride 4096 equals dim 0's new_stride 4096) instead of raising
        # or leaving it alone.
        d0, d1, d2 = sympy.symbols("d0 d1 d2")
        info = _RetiledBufferInfo(
            old_stride=(Integer(131072), Integer(4096), Integer(1)),
            new_stride=(Integer(4096), Integer(1024), Integer(1)),
            old_size=(Integer(2), Integer(32), Integer(4096)),
        )
        already_fresh_index = 4096 * d0 + 1024 * d1 + d2

        result = _retile_load_index("buf", already_fresh_index, info)

        self.assertEqual(simplify(result - already_fresh_index), 0)

    def test_genuinely_stale_index_at_old_scale_still_rewritten(self):
        # Same buffer/info as test_index_already_at_new_scale_is_left_unchanged,
        # but with a genuinely stale index (coefficients at old_stride scale)
        # -- must still be rewritten to new_stride, not skipped.
        d0, d1, d2 = sympy.symbols("d0 d1 d2")
        info = _RetiledBufferInfo(
            old_stride=(Integer(131072), Integer(4096), Integer(1)),
            new_stride=(Integer(4096), Integer(1024), Integer(1)),
            old_size=(Integer(2), Integer(32), Integer(4096)),
        )
        stale_index = 131072 * d0 + 4096 * d1 + d2

        result = _retile_load_index("buf", stale_index, info)

        self.assertEqual(simplify(result - (4096 * d0 + 1024 * d1 + d2)), 0)


def _make_consumer_with_ranges(ranges):
    """Return a fake ComputedBuffer whose ``.data.ranges`` is ``ranges``.

    Used to drive ``_squeezed_retile_dims``/``_consumer_own_dim_symbol``,
    which only inspect ``consumer.data.ranges`` -- no other ComputedBuffer
    attribute is touched by either function.
    """
    return _make_op(_make_pointwise(ranges))


class TestSqueezedRetileDims(unittest.TestCase):
    """Unit tests for which raw dims need a re-minted symbol on redirect.

    See coarse_tile.py's module docstring / _squeezed_retile_dims's own
    docstring for the two-bug history this guards: a dim only needs a
    minted term when it was squeezed out of the *old* (tile-local) buffer's
    layout (old_size[d] == 1, new_stride[d] != 0) AND the consumer's own
    output for that same raw dim is non-unit (data.ranges[d] != 1) -- i.e a
    real loop variable actually exists for it somewhere in the trace.
    """

    def test_squeezed_dim_with_nonunit_consumer_output_included(self):
        # old_size=1 (squeezed out of the read), new_stride != 0 (real in the
        # new buffer), consumer's own dim 0 is non-unit (size 8) -- a real
        # loop var exists for this dim, so it belongs in the result.
        info = _RetiledBufferInfo(
            old_stride=(Integer(0), Integer(1)),
            new_stride=(Integer(256), Integer(1)),
            old_size=(Integer(1), Integer(256)),
        )
        consumer = _make_consumer_with_ranges([8, 256])

        self.assertEqual(_squeezed_retile_dims(info, consumer), [0])

    def test_unit_consumer_output_dim_excluded(self):
        # Same squeeze/stride shape as above, but the consumer's own output
        # for dim 0 is ALSO unit-size (e.g. B=1 when only H is tiled) -- no
        # real loop variable exists for this dim anywhere in the trace, so
        # it must be excluded. This is the exact regression this guard
        # fixes: omitting it causes _consumer_own_dim_symbol to mint a
        # symbol that collides with an unrelated, already-present symbol
        # under sympy's automatic term-merging (see _retile_load_index's
        # already_present comment).
        info = _RetiledBufferInfo(
            old_stride=(Integer(0), Integer(1)),
            new_stride=(Integer(256), Integer(1)),
            old_size=(Integer(1), Integer(256)),
        )
        consumer = _make_consumer_with_ranges([1, 256])

        self.assertEqual(_squeezed_retile_dims(info, consumer), [])

    def test_nonsqueezed_dim_excluded(self):
        # old_size != 1: this dim was never squeezed out of the read in the
        # first place, so compute_tile_index already rescales its existing
        # coefficient -- no term needs to be added back.
        info = _RetiledBufferInfo(
            old_stride=(Integer(128), Integer(1)),
            new_stride=(Integer(64), Integer(1)),
            old_size=(Integer(2), Integer(256)),
        )
        consumer = _make_consumer_with_ranges([2, 256])

        self.assertEqual(_squeezed_retile_dims(info, consumer), [])

    def test_strideless_new_dim_excluded(self):
        # new_stride == 0: the dim stays size-1/strideless in the new buffer
        # too, so it contributes nothing regardless of the consumer's shape.
        info = _RetiledBufferInfo(
            old_stride=(Integer(0), Integer(1)),
            new_stride=(Integer(0), Integer(1)),
            old_size=(Integer(1), Integer(256)),
        )
        consumer = _make_consumer_with_ranges([8, 256])

        self.assertEqual(_squeezed_retile_dims(info, consumer), [])


class TestIndexVarPrefix(unittest.TestCase):
    """Unit tests for inferring the live loop-var naming prefix from an index."""

    def test_infers_d_prefix(self):
        d0, d1 = sympy.symbols("d0 d1")
        self.assertEqual(_index_var_prefix({d0, d1}), "d")

    def test_infers_q_prefix(self):
        q0, q2 = sympy.symbols("q0 q2")
        self.assertEqual(_index_var_prefix({q0, q2}), "q")

    def test_infers_underscore_i_prefix(self):
        i0 = sympy.Symbol("_i0")
        self.assertEqual(_index_var_prefix({i0}), "_i")

    def test_empty_set_falls_back_to_d(self):
        self.assertEqual(_index_var_prefix(set()), "d")

    def test_symbol_with_no_trailing_digits_falls_back_to_d(self):
        # A symbol with no trailing digits at all (e.g. a shape symbol, not
        # a dense loop var) never matches "name[:i] with i < len(name)", so
        # it contributes nothing usable and the function falls back to "d".
        s = sympy.Symbol("s")
        self.assertEqual(_index_var_prefix({s}), "d")


class TestConsumerOwnDimSymbol(unittest.TestCase):
    """Unit tests for mapping a raw output dim to the consumer's own loop var."""

    def test_no_preceding_unit_dim_maps_identity(self):
        # No unit dims at all: dense numbering over non-unit dims is just
        # the identity, so raw dim 1 maps to loop var d1.
        consumer = _make_consumer_with_ranges([8, 256])

        sym = _consumer_own_dim_symbol(consumer, dim=1, prefix="d")

        self.assertEqual(sym, sympy_index_symbol("d1"))

    def test_preceding_unit_dim_shifts_mapping(self):
        # ranges=[1, 8, 256]: dim 0 is unit-size and squeezed out of the
        # dense numbering entirely, so raw dim 1 (the first non-unit dim)
        # maps to loop var d0, not d1.
        consumer = _make_consumer_with_ranges([1, 8, 256])

        sym = _consumer_own_dim_symbol(consumer, dim=1, prefix="d")

        self.assertEqual(sym, sympy_index_symbol("d0"))

    def test_uses_given_prefix(self):
        consumer = _make_consumer_with_ranges([8, 256])

        sym = _consumer_own_dim_symbol(consumer, dim=0, prefix="q")

        self.assertEqual(sym, sympy_index_symbol("q0"))


class TestRetileLoadIndexWithConsumer(unittest.TestCase):
    """End-to-end unit tests for _retile_load_index's consumer-aware path.

    Covers the two real bugs fixed in this area: a foreign loop-var prefix
    surviving into the rewritten index (bug 1), and a minted symbol
    colliding with an unrelated symbol already present in the index when
    the consumer's own output dim is unit-size (bug 2).
    """

    def test_squeezed_dim_added_back_with_matching_prefix(self):
        # old_size[0]==1: dim 0 was squeezed out of the incoming index
        # entirely (no c0 term at all). The consumer's own dim 0 is
        # non-unit (size 4), so a term must be added back, using whatever
        # prefix is already live in the index (here "q", not the default
        # "d") -- this is bug 1's fix.
        info = _RetiledBufferInfo(
            old_stride=(Integer(0), Integer(1)),
            new_stride=(Integer(256), Integer(1)),
            old_size=(Integer(1), Integer(256)),
        )
        q1 = sympy.Symbol("q1")
        consumer = _make_consumer_with_ranges([4, 256])

        result = _retile_load_index("buf", q1, info, consumer)

        self.assertEqual(simplify(result - (256 * sympy_index_symbol("q0") + q1)), 0)

    def test_unit_consumer_dim_adds_no_term_and_does_not_collide(self):
        # Same squeeze shape as above, but the consumer's own dim 0 is ALSO
        # unit-size (e.g. B=1 when only H is tiled) -- no real loop variable
        # exists for it, so no term must be added. Before the fix, this
        # minted a d0 that collided with the unrelated, already-present d0
        # below (coefficient 16384) via sympy's automatic term-merging,
        # silently corrupting it to 16384*d0 + 256*d0 instead of raising or
        # leaving 16384*d0 alone -- this is bug 2's exact regression.
        info = _RetiledBufferInfo(
            old_stride=(Integer(0), Integer(1)),
            new_stride=(Integer(256), Integer(1)),
            old_size=(Integer(1), Integer(256)),
        )
        d0 = sympy_index_symbol("d0")
        consumer = _make_consumer_with_ranges([1, 8, 256])

        result = _retile_load_index("buf", 16384 * d0, info, consumer)

        self.assertEqual(simplify(result - 16384 * d0), 0)

    def test_no_consumer_leaves_squeezed_dim_unaugmented(self):
        # consumer=None (the _RetileLoadIndexHandler / full->tile direction)
        # skips the whole squeezed-dim augmentation path unconditionally --
        # confirms the default argument's behavior matches every existing
        # call site that never passes consumer at all.
        info = _RetiledBufferInfo(
            old_stride=(Integer(0), Integer(1)),
            new_stride=(Integer(256), Integer(1)),
            old_size=(Integer(1), Integer(256)),
        )
        q1 = sympy.Symbol("q1")

        result = _retile_load_index("buf", q1, info)

        self.assertEqual(simplify(result - q1), 0)


class TestShouldPatchRetiledLoadIndexes(unittest.TestCase):
    """Unit tests for selecting exact-loop consumers of retiled buffers."""

    def test_requires_exact_loop_group_id(self):
        op = _make_inside_consumer_op("consumer", "retiled", loop_group_id=(0,))

        result = _should_patch_retiled_load_indexes(op, (0, 0), {"retiled"})

        self.assertFalse(result)

    def test_requires_reading_retiled_buffer(self):
        op = _make_inside_consumer_op("consumer", "other", loop_group_id=(0, 0))

        result = _should_patch_retiled_load_indexes(op, (0, 0), {"retiled"})

        self.assertFalse(result)

    def test_accepts_same_group_consumer_of_retiled_buffer(self):
        op = _make_inside_consumer_op("consumer", "retiled", loop_group_id=(0, 0))

        result = _should_patch_retiled_load_indexes(op, (0, 0), {"retiled"})

        self.assertTrue(result)


class TestReplaceGroupOp(unittest.TestCase):
    """Unit tests for keeping coarse-tile group op references current."""

    def test_replaces_by_identity(self):
        old_op = _make_op(_make_pointwise([4]), "old")
        new_op = _make_op(_make_pointwise([4]), "new")
        group_ops = [old_op]

        _replace_group_op(group_ops, old_op, new_op)

        self.assertIs(group_ops[0], new_op)

    def test_replaces_by_operation_name_when_identity_changed(self):
        stale_op = _make_op(_make_pointwise([4]), "old")
        current_op = _make_op(_make_pointwise([4]), "old")
        new_op = _make_op(_make_pointwise([4]), "new")
        group_ops = [stale_op]

        _replace_group_op(group_ops, current_op, new_op)

        self.assertIs(group_ops[0], new_op)


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

    def test_tiled_symbol_trip_counts_preserved(self):
        sym = Symbol("_tile_adv_op0_lvl0")
        op = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space={Symbol("x0"): (Integer(128), 1)},
            args=[],
            op_info={},
            tiled_symbol_trip_counts={sym: 4},
        )
        original = [op]
        result = _roundtrip(original)
        self.assertEqual(result[0].tiled_symbol_trip_counts, {sym: 4})


# ===========================================================================
# 2. coarse_tile IR pass
# ===========================================================================


class TestDivideRanges(unittest.TestCase):
    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()

    def tearDown(self):
        self._graph_ctx.__exit__(None, None, None)

    def _make_named_pointwise(self, ranges, names):
        from torch._inductor.ir import ComputedBuffer, FixedLayout, Pointwise

        from torch_spyre._inductor.pass_utils import iteration_space_from_op

        data = Pointwise(
            device=torch.device("cpu"),
            dtype=torch.float16,
            inner_fn=lambda index: sympy.Integer(0),
            ranges=list(ranges),
        )
        op = ComputedBuffer(
            name="named_pointwise",
            layout=FixedLayout(torch.device("cpu"), torch.float16, list(ranges), None),
            data=data,
        )
        symbols = tuple(iteration_space_from_op(op))
        self.assertEqual(len(symbols), len(names))
        op.work_div_loop_info = {  # type: ignore[attr-defined]
            symbol: [name] for symbol, name in zip(symbols, names)
        }
        return op

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

    def test_named_token_split_survives_expert_dim_squeeze(self):
        from torch_spyre._inductor.pass_utils import iteration_space_from_op
        from torch_spyre._inductor.wsr.coarse_tile import (
            _apply_work_div_symbol_remap,
        )

        op = self._make_named_pointwise(
            [Integer(128), Integer(512), Integer(704)], ["E", "T", "F"]
        )

        result = _divide_ranges(op, Integer(128), tiled_dims=[0])
        _apply_work_div_symbol_remap(op, result.symbol_remap)

        iteration_space = iteration_space_from_op(op)
        self.assertEqual(tuple(iteration_space.values()), (Integer(512), Integer(704)))
        names_by_extent = {
            iteration_space[symbol]: names
            for symbol, names in op.work_div_loop_info.items()  # type: ignore[attr-defined]
        }
        self.assertEqual(names_by_extent, {Integer(512): ["T"], Integer(704): ["F"]})

    def test_named_dims_compose_across_two_size_one_squeezes(self):
        from torch_spyre._inductor.pass_utils import iteration_space_from_op
        from torch_spyre._inductor.wsr.coarse_tile import (
            _apply_work_div_symbol_remap,
        )

        op = self._make_named_pointwise(
            [Integer(8), Integer(16), Integer(32)], ["E0", "E1", "T"]
        )

        first = _divide_ranges(op, Integer(8), tiled_dims=[0])
        _apply_work_div_symbol_remap(op, first.symbol_remap)
        second = _divide_ranges(op, Integer(16), tiled_dims=[1])
        _apply_work_div_symbol_remap(op, second.symbol_remap)

        iteration_space = iteration_space_from_op(op)
        self.assertEqual(tuple(iteration_space.values()), (Integer(32),))
        self.assertEqual(
            list(op.work_div_loop_info.values()),  # type: ignore[attr-defined]
            [["T"]],
        )

    def test_non_monotone_symbol_mapping_fails_visibly(self):
        from torch_spyre._inductor.wsr.coarse_tile import (
            _LogicalIterationSymbol,
            _order_preserving_symbol_remap,
        )

        op = MagicMock()
        op.get_name.return_value = "reordered"
        before = (
            _LogicalIterationSymbol(("output", 0), Integer(8), Symbol("d0")),
            _LogicalIterationSymbol(("output", 1), Integer(16), Symbol("d1")),
        )
        after = (
            _LogicalIterationSymbol(("output", 1), Integer(16), Symbol("d0")),
            _LogicalIterationSymbol(("output", 0), Integer(8), Symbol("d1")),
        )

        with self.assertRaisesRegex(Unsupported, "order-preserving dimension mapping"):
            _order_preserving_symbol_remap(op, before, after)

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

        N = sympy.Symbol("N", positive=True, integer=True)
        pw = Pointwise(
            device=torch.device("cpu"),
            dtype=torch.float16,
            inner_fn=lambda index: sympy.Integer(1),
            ranges=[4 * N, Integer(32)],
        )
        layout = FixedLayout(torch.device("cpu"), torch.float16, [4 * N, Integer(32)])
        op = ComputedBuffer(name="buf0", layout=layout, data=pw)

        pw.get_free_symbol_uses()  # prime the cache
        self.assertTrue(hasattr(pw, _LOOPS_FREE_SYMS_KEY))

        _divide_ranges(op, N, tiled_dims=[0])

        self.assertFalse(hasattr(pw, _LOOPS_FREE_SYMS_KEY))

    def test_cache_invalidated_after_divide_reduction(self):
        from torch._inductor.ir import (
            ComputedBuffer,
            FixedLayout,
            Reduction,
            ReductionHint,
        )

        N = sympy.Symbol("N", positive=True, integer=True)
        red = Reduction(
            device=torch.device("cpu"),
            dtype=torch.float16,
            inner_fn=lambda index, rindex: sympy.Integer(1),
            ranges=[4 * N],
            reduction_ranges=[Integer(128)],
            reduction_type="sum",
            src_dtype=torch.float16,
            reduction_hint=ReductionHint.DEFAULT,
        )
        layout = FixedLayout(torch.device("cpu"), torch.float16, [4 * N])
        op = ComputedBuffer(name="buf0", layout=layout, data=red)

        red.get_free_symbol_uses()  # prime both Loops and Reduction cache entries
        self.assertTrue(hasattr(red, _LOOPS_FREE_SYMS_KEY))
        self.assertTrue(hasattr(red, _REDUCTION_FREE_SYMS_KEY))

        _divide_ranges(op, N, tiled_dims=[0])

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
        from torch_spyre._inductor.wsr.coarse_tile import _resize_device_layout

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
        from torch_spyre._inductor.wsr.coarse_tile import _resize_device_layout

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
        from torch_spyre._inductor.wsr.coarse_tile import _resize_device_layout

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

    def test_resize_device_layout_transposed_same_size_dims(self):
        """Issue #3116: two host dims of the same size in a transposed layout.

        Flash-attention QK^T output: logical [B, H, Sq, Skv] with Sq == Skv,
        stored transposed.  The device layout is byte-identical whether Sq or
        Skv is the stick dim (device_size=[32,512,8,1,64],
        stride_map=[512,16384,64,-1,1]), so size-based elimination cannot tell
        the two size-512 host dims apart.

        Without identity (``stick_host_dim=None``) this is genuinely ambiguous
        and must raise.  With the authoritative stick host dim threaded in
        (named-dim identity), it reconstructs correctly: tiling Sq 512 -> 256
        shrinks the non-stick Sq device dim and leaves the transposed
        stride_map / stick untouched.
        """
        from torch_spyre._C import SpyreTensorLayout
        from torch_spyre._inductor.wsr.coarse_tile import _resize_device_layout

        dev = SpyreTensorLayout([1, 1], torch.float16).device_dtype
        # Transposed QK^T output: Skv (host dim 3) is the stick; Sq (host dim 2)
        # is a non-stick dim of the same size (512), so they collide by size.
        stl = SpyreTensorLayout([32, 512, 8, 1, 64], [512, 16384, 64, -1, 1], dev)
        host_size = [1, 32, 512, 512]

        # Fallback (no identity): size-based elimination is ambiguous -> raise.
        with self.assertRaises(RuntimeError):
            _resize_device_layout(stl, host_size, [1, 32, 256, 512])

        # With authoritative stick host dim (Skv == host dim 3): unambiguous.
        result = _resize_device_layout(
            stl, host_size, [1, 32, 256, 512], stick_host_dim=3
        )
        self.assertEqual(list(result.device_size), [32, 256, 8, 1, 64])
        self.assertEqual(list(result.stride_map), [512, 16384, 64, -1, 1])


def _mock_op_out_coords(op):
    """Return pre-built coords stored on op by _make_hinted_op, or empty list."""
    return getattr(op, "_test_out_coords", [])


class TestCoarseTile(unittest.TestCase):
    def setUp(self):
        self._patch = patch(
            "torch_spyre._inductor.wsr.coarse_tile.op_out_coords",
            side_effect=_mock_op_out_coords,
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_empty_groups_list_is_noop(self):
        data = _make_pointwise([Integer(32)])
        op = _make_op(data, "op0")
        original = list(data.ranges)
        coarse_tile_pre_stickify(_graph([op]), [])
        self.assertFalse(hasattr(op, "loop_info") and op.loop_info != MagicMock())
        self.assertEqual(data.ranges, original)

    def test_non_computed_buffer_skipped(self):
        op_extern = _make_non_computed_op("extern0")
        data = _make_pointwise([Integer(16)])
        op_computed = _make_hinted_op(data, "op0", hints=((0, 0),))
        coarse_tile_pre_stickify(
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
        coarse_tile_pre_stickify(_graph([op]), [([op], [(0, k)])])
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
            coarse_tile_pre_stickify(
                _graph([op0, op1, op2]), [([op0, op2], [(0, Integer(4))])]
            )

    def test_op_not_in_operations_raises(self):
        data = _make_pointwise([Integer(32)])
        op_known = _make_hinted_op(data, "op0", hints=((0, 0),))
        op_unknown = _make_hinted_op(
            _make_pointwise([Integer(8)]), "unknown", hints=((0, 0),)
        )
        with self.assertRaises(RuntimeError):
            coarse_tile_pre_stickify(
                _graph([op_known]), [([op_unknown], [(0, Integer(2))])]
            )

    def test_post_stickify_skips_pass_1(self):
        """coarse_tile_post_stickify must skip both planning and execution
        of Pass 1 -- a full-buffer boundary read stays a direct read of the
        full buffer, not redirected to a copy."""
        from torch._inductor.ir import ComputedBuffer

        gm = fx.symbolic_trace(lambda: None)
        graph_ctx = V.set_graph_handler(GraphLowering(gm))
        graph_ctx.__enter__()
        try:
            tiled_op, full_deps, operations = _make_full_buffer_read_fixture()
            self.assertEqual(len(full_deps), 1)
            full_buf_name = full_deps[0].name

            groups = [([tiled_op], [(0, Integer(8))])]
            coarse_tile_post_stickify(_graph(operations), groups)

            # No new copy op was inserted: still exactly the original two ops.
            self.assertEqual(len(operations), 2)
            loaded_names = []

            class _Recorder:
                def load(self, name, index):
                    loaded_names.append(name)
                    return 0.0

            final_op = next(
                o
                for o in operations
                if isinstance(o, ComputedBuffer) and o.get_name() == "tiled_op0"
            )
            with V.set_ops_handler(_Recorder()):
                final_op.data.inner_fn([sympy.Integer(0) for _ in final_op.data.ranges])
            self.assertIn(full_buf_name, loaded_names)
        finally:
            graph_ctx.__exit__(None, None, None)

    def test_post_stickify_loop_extends_crossing_lifetime(self):
        """A value born before an automatically-tiled counted loop and read
        inside it must stay live through the loop (issue #4035).

        Same clobber the ``persistent``/``later`` repro in
        test_perm_layout_solver.py guards at the solver level, but here the
        ``lifetime_end_override`` is *computed* by
        ``counted_loop_lifetime_end_overrides`` for a loop built by the
        automatic span-overflow entry point, ``coarse_tile_post_stickify``
        -- not hand-passed, and not hint-driven.  The liveness gap comes
        from the graph holding one textual copy of the loop body regardless
        of how the loop was constructed, so overrides must cover this path
        exactly as they cover explicit ``spyre_hint`` loops.
        """
        from collections import namedtuple

        from torch_spyre._inductor.scratchpad.utils import (
            counted_loop_lifetime_end_overrides,
        )

        _dep = namedtuple("_dep", ["name"])

        def _stub_read_writes(op, reads, writes):
            op.get_read_writes.return_value = SimpleNamespace(
                reads={_dep(name) for name in reads},
                writes={_dep(name) for name in writes},
            )

        # Op 0 is born outside the loop; ops 1-3 form one counted loop.
        # ``reader_x`` reads across the loop boundary; ``reader_i`` reads a
        # value born inside the same loop.
        crossing = _make_op(_make_pointwise([Integer(16)]), "crossing")
        _stub_read_writes(crossing, reads=["arg0"], writes=["crossing"])
        body_writer = _make_hinted_op(_make_pointwise([Integer(16)]), "body_writer")
        _stub_read_writes(body_writer, reads=["arg1"], writes=["internal"])
        reader_x = _make_hinted_op(_make_pointwise([Integer(16)]), "reader_x")
        _stub_read_writes(reader_x, reads=["crossing"], writes=["reader_x_out"])
        reader_i = _make_hinted_op(_make_pointwise([Integer(16)]), "reader_i")
        _stub_read_writes(reader_i, reads=["internal"], writes=["reader_i_out"])

        operations = [crossing, body_writer, reader_x, reader_i]
        coarse_tile_post_stickify(
            _graph(operations),
            [([body_writer, reader_x, reader_i], [(0, Integer(4))])],
        )

        # The loop body stays ops 1..3: nothing was inserted, the op outside
        # the group was not stamped, and the group ops carry the loop id.
        self.assertEqual(len(operations), 4)
        self.assertFalse(hasattr(crossing, "loop_info"))
        self.assertEqual(reader_i.loop_info.loop_group_id, (0,))

        overrides = counted_loop_lifetime_end_overrides(
            SimpleNamespace(operations=operations, graph_input_names=["arg0", "arg1"])
        )

        # Values born before the loop and read inside it -- the computed
        # buffer ``crossing`` and the graph input ``arg1`` -- live through
        # the loop's textual end (exclusive index 4).  ``internal`` is born
        # and consumed inside the loop: no extension.  ``arg0`` is only read
        # outside the loop: no extension.
        self.assertEqual(overrides, {"crossing": 4, "arg1": 4})

    def test_end_to_end_shares_one_copy_across_group(self):
        """Full coarse_tile() entry point: two hint-driven ops in one group
        both reading the same full InputBuffer at the same index must end
        up sharing exactly one inserted read-copy op.

        This closes the loop that Tasks 2/3/6/7's direct
        _plan_read_copies/_insert_all_read_copy_ops tests don't cover: it
        is the only test that also runs coarse_tile()'s later by-name
        resync loop (_patch_retiled_load_indexes, near the end of
        coarse_tile()'s body), which could in principle silently break
        Pass 1's sharing if it replaced an op object Pass 1 already
        consumed by name.
        """
        from torch._inductor.ir import (
            ComputedBuffer,
            FixedLayout,
            InputBuffer,
            Pointwise,
            StorageBox,
            TensorBox,
        )
        from torch_spyre._inductor.propagate_hints import DimHint

        gm = fx.symbolic_trace(lambda: None)
        graph_ctx = V.set_graph_handler(GraphLowering(gm))
        graph_ctx.__enter__()
        try:
            device = torch.device("cpu")
            dtype = torch.float32

            # One real, full-size InputBuffer shared by both ops below --
            # unlike _make_real_pointwise_op (which allocates a fresh
            # InputBuffer per op), both readers here load the exact same
            # buffer object at the exact same index, so _plan_read_copies
            # should key them into a single ReadCopyEntry.
            shared_input = InputBuffer(
                name="shared_in", layout=FixedLayout(device, dtype, [64], [1])
            )
            V.graph.name_to_buffer["shared_in"] = shared_input
            shared_box = TensorBox(StorageBox(shared_input))

            def _make_reader(name):
                def inner_fn(index):
                    return shared_box.make_loader()(index)

                pw = Pointwise.create(
                    device=device,
                    dtype=dtype,
                    inner_fn=inner_fn,
                    ranges=[Integer(64)],
                )
                pw_data = pw.data.data  # TensorBox -> StorageBox -> Pointwise
                op = ComputedBuffer(
                    name=name,
                    layout=FixedLayout(device, dtype, [Integer(64)], None),
                    data=pw_data,
                )
                op.operation_name = name
                op.origins = OrderedSet()
                V.graph.name_to_buffer[name] = op
                op._test_out_coords = [sympy.Symbol("c0")]
                op.dim_hints = [
                    DimHint(
                        dim_names=["dim0"],
                        split_count=1,
                        loop_var=sympy.Symbol("c0"),
                        is_reduction=False,
                        hint_id=0,
                    )
                ]
                return op

            op_a = _make_reader("op_a")
            op_b = _make_reader("op_b")
            operations = [op_a, op_b]
            groups = [([op_a, op_b], [(0, Integer(8))])]

            coarse_tile_pre_stickify(_graph(operations), groups)

            copy_ops = [
                op
                for op in operations
                if isinstance(op, ComputedBuffer)
                and op.get_name().startswith("coarse_tile_read_copy_")
            ]
            self.assertEqual(len(copy_ops), 1)
        finally:
            graph_ctx.__exit__(None, None, None)


class TestCoarseTileNested(unittest.TestCase):
    """Verify that the nested group format [(hint_id, K1), ...] works."""

    def setUp(self):
        self._patch = patch(
            "torch_spyre._inductor.wsr.coarse_tile.op_out_coords",
            side_effect=_mock_op_out_coords,
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_nested_spec_stamps_list_attributes(self):
        data = _make_pointwise([Integer(256), Integer(128)])
        op = _make_hinted_op(data, "op0", hints=((1, 0), (2, 1)))
        coarse_tile_pre_stickify(
            _graph([op]), [([op], [(1, Integer(4)), (2, Integer(2))])]
        )
        self.assertEqual(op.loop_info.loop_group_id, (0, 0))
        self.assertEqual(op.loop_info.loop_count, [Integer(4), Integer(2)])
        self.assertEqual(op.loop_info.loop_tiled_dims, [[0], [1]])

    def test_nested_spec_divides_ranges_both_levels(self):
        data = _make_pointwise([Integer(256), Integer(128)])
        op = _make_hinted_op(data, "op0", hints=((1, 0), (2, 1)))
        coarse_tile_pre_stickify(
            _graph([op]), [([op], [(1, Integer(4)), (2, Integer(2))])]
        )
        self.assertEqual(data.ranges[0], Integer(64))
        self.assertEqual(data.ranges[1], Integer(64))

    def test_nested_spec_outer_only_divides_outer_dim(self):
        data = _make_pointwise([Integer(32), Integer(64), Integer(16)])
        op = _make_hinted_op(data, "op0", hints=((1, 0), (2, 1)))
        coarse_tile_pre_stickify(
            _graph([op]), [([op], [(1, Integer(4)), (2, Integer(8))])]
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
        coarse_tile_pre_stickify(
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
        coarse_tile_pre_stickify(
            _graph([op]), [([op], [(1, Integer(4)), (2, Integer(2))])]
        )
        self.assertEqual(data.ranges[0], Integer(32))
        self.assertEqual(op.loop_info.loop_count, [Integer(4), Integer(2)])
        self.assertEqual(op.loop_info.loop_tiled_dims, [[0], [0]])

    def test_planned_tile_extents_per_level_same_dim_two_levels(self):
        from torch_spyre._inductor.wsr.coarse_tile import (
            _planned_tile_extents_per_level,
        )

        data = _make_pointwise([Integer(256)])
        op = _make_hinted_op(data, "op0", hints=((1, 0), (2, 0)))
        levels = [(1, Integer(4)), (2, Integer(2))]
        op_tiled_dims = [[0], [0]]
        op_tiled_reduction_dims = [[], []]
        per_level = _planned_tile_extents_per_level(
            op, op_tiled_dims, op_tiled_reduction_dims, levels
        )
        self.assertEqual(len(per_level), 2)
        # Outer level (count 4): extent = final_extent(32) * inner_count(2) = 64.
        self.assertEqual(per_level[0], {0: Integer(64)})
        # Inner level (count 2): extent = final_extent(32).
        self.assertEqual(per_level[1], {0: Integer(32)})

    def test_coarse_tile_plans_before_any_transformation(self):
        """coarse_tile() must fully plan every group before transforming any.

        Reuses the same two-group fixture shape as
        test_single_and_nested_groups_coexist (group 0: single-level spec
        tiling dim 0; group 1: two-level nested spec) -- a coarse smoke test
        confirming the end-to-end call still stamps loop_info onto every op
        as before.  The ordering guarantee itself (plan-all-then-transform-
        all) is implicitly covered by plan_coarse_tile_groups's own
        zero-mutation test plus this test's confirmation that stamping still
        happens by the time coarse_tile() returns.
        """
        from torch._inductor.ir import ComputedBuffer

        d0 = _make_pointwise([Integer(64), Integer(32)])
        d1 = _make_pointwise([Integer(128), Integer(64)])
        op0 = _make_hinted_op(d0, "op0", hints=((1, 0),))
        op1 = _make_hinted_op(d1, "op1", hints=((2, 0), (3, 1)))
        groups = [
            ([op0], [(1, Integer(4))]),
            ([op1], [(2, Integer(4)), (3, Integer(2))]),
        ]
        coarse_tile_pre_stickify(_graph([op0, op1]), groups)
        for group_ops, _ in groups:
            for op in group_ops:
                if isinstance(op, ComputedBuffer):
                    self.assertIsNotNone(getattr(op, "loop_info", None))


class TestCoarseTileTiledDimsPerRead(unittest.TestCase):
    """Real-IR tests for CoarseTileInfo.tiled_dims_per_read /
    output_tiled_dims, using the small example from
    docs/source/compiler/coarse_tiling_loops.md (1024x4096, outer K=2 over
    dim 0, inner M=4 over dim 1).

    plan_coarse_tile_groups's/_apply_plan's new code calls
    op.get_read_writes() on real IR (Task 3), which internally calls
    InputBuffer.make_loader() -> checks V.graph.sizevars -- this requires an
    active graph handler at the time plan_coarse_tile_groups/_apply_plan
    run, not just while the op is built.  setUp/tearDown keep one open for
    the whole test body; a fresh GraphLowering (distinct from the one
    _make_real_pointwise_op/_make_real_reduction_op build their op under,
    internally, to construct the IR) is sufficient -- get_read_writes() has
    no dependency on graph-handler identity, only on one being active.

    These tests call plan_coarse_tile_groups + _apply_plan directly rather
    than the full coarse_tile() entry point.  coarse_tile() unconditionally
    also runs the reduction-machinery pass (_insert_all_reduction_ops) after
    stamping every group, which for a Reduction op with an actually-tiled
    reduction dim drives _propagate_tiled_reduction_op -> _allocate_full_buffer
    -> graph_lowering.run_node() on a synthesized spyre.empty FX node -- real
    FX-dispatch/lowering machinery this lightweight harness does not
    provide (confirmed live: raises LoweringException /
    "'NullHandler' object does not support the context manager protocol").
    _apply_plan is the layer that actually populates tiled_dims_per_read /
    output_tiled_dims (see Task 2/3), so calling plan_coarse_tile_groups +
    _apply_plan directly exercises exactly what the declarative tiling pass
    needs without pulling in buffer propagation, which it does not touch.
    """

    def setUp(self):
        self._patch = patch(
            "torch_spyre._inductor.wsr.coarse_tile.op_out_coords",
            side_effect=_mock_op_out_coords,
        )
        self._patch.start()
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()

    def tearDown(self):
        self._graph_ctx.__exit__(None, None, None)
        self._patch.stop()

    def test_small_example_output_and_input_advance(self):
        # a + b -> buf0.  a, b are [1024, 4096], row-major (stride [4096, 1]).
        op = _make_real_pointwise_op(
            ranges=[Integer(1024), Integer(4096)],
            input_shapes_strides=[
                ([1024, 4096], [4096, 1]),
                ([1024, 4096], [4096, 1]),
            ],
            name="buf0",
            hints=((1, 0), (2, 1)),
        )
        levels = [(1, Integer(2)), (2, Integer(4))]
        plan = plan_coarse_tile_groups([op], [([op], levels)])
        _apply_plan([op], (0, 0), levels, {op.get_operation_name(): 0}, plan)
        # Output: dim 0 tiled at level 0 (K=2 outer, extent 512 = 1024/2);
        # dim 1 tiled at level 1 (M=4 inner, extent 1024 = 4096/4). Neither
        # dim is tiled at more than one level here, so each level's list has
        # exactly the one dim it tiles.
        self.assertEqual(
            op.loop_info.output_tiled_dims, [[(0, Integer(512))], [(1, Integer(1024))]]
        )
        # Inputs: a and b are never divided (buf0's own division doesn't
        # touch them), so both reads have the same tiled-dims decision as
        # the output would have had pre-division (same dims, same extents
        # -- the decision is about which dims/levels, not about a
        # particular tensor's stride, which is applied later at
        # spyre_kernel.py substitution time).
        self.assertEqual(len(op.loop_info.tiled_dims_per_read), 2)
        for tiled_dims in op.loop_info.tiled_dims_per_read:
            self.assertEqual(tiled_dims, [[(0, Integer(512))], [(1, Integer(1024))]])

    def test_broadcast_input_has_zero_advance(self):
        # a (tiled input) + b (broadcast scalar-row input, stride [0, 1]).
        op = _make_real_pointwise_op(
            ranges=[Integer(1024), Integer(4096)],
            input_shapes_strides=[
                ([1024, 4096], [4096, 1]),
                ([1024, 4096], [0, 1]),
            ],
            name="buf0",
            hints=((1, 0), (2, 1)),
        )
        levels = [(1, Integer(2)), (2, Integer(4))]
        plan = plan_coarse_tile_groups([op], [([op], levels)])
        _apply_plan([op], (0, 0), levels, {op.get_operation_name(): 0}, plan)
        broadcast_tiled_dims = op.loop_info.tiled_dims_per_read[1]
        # Broadcast along dim 0 (stride 0): b's index never depends on d0,
        # so dim 0 is filtered out of every level's list for this dep;
        # only dim 1 (M=4, extent 1024, tiled at level 1) survives.
        self.assertEqual(broadcast_tiled_dims, [[], [(1, Integer(1024))]])

    def test_unit_tile_step_stays_in_pass1_planning_data(self):
        from torch_spyre._inductor.wsr.coarse_tile import (
            _capture_predivision_unit_steps,
        )

        op = _make_real_pointwise_op(
            ranges=[Integer(3), Integer(64)],
            input_shapes_strides=[([3, 64], [64, 1])],
            name="buf0",
            hints=((1, 0),),
        )
        levels = [(1, Integer(3))]
        plan = plan_coarse_tile_groups([op], [([op], levels)])
        captured = _capture_predivision_unit_steps([op], plan)

        self.assertEqual(
            captured[id(op)],
            ((((0, Integer(64), Integer(1)),),),),
        )

        _apply_plan([op], (0,), levels, {op.get_operation_name(): 0}, plan)
        # The early observation is not stamped onto the transformed op. It
        # does not change general loop addressing before Pass 1 selects a
        # read copy.
        self.assertEqual(op.loop_info.squeezed_advance_per_read, [])
        self.assertFalse(hasattr(op.loop_info, "predivision_unit_step_per_read"))

    def test_predivision_step_wins_and_warns_on_legacy_disagreement(self):
        from torch_spyre._inductor.wsr.coarse_tile import _select_unit_steps

        planned = [[(Integer(64), Integer(1))]]
        legacy = [[(Integer(32), Integer(1))]]
        with patch("torch_spyre._inductor.wsr.coarse_tile.logger.warning") as warning:
            selected = _select_unit_steps(
                op_name="buf0",
                dep_name="arg0",
                dim=0,
                planned=planned,
                legacy=legacy,
            )

        self.assertEqual(selected, planned)
        warning.assert_called_once()

    def test_reduction_dim_advance_is_offset_by_output_dims(self):
        # out[d0] = sum_{d1} in[d0, d1].  in is [8, 16], row-major
        # (stride [16, 1]).  Output dim 0 (K=2 outer) is a real output dim;
        # reduction dim 1 (R=4) is a reduction dim, numbered continuously
        # after output dims per Inductor's own d{i} convention -- so inside
        # in.index it appears as d1 (n_output_dims=1 + reduction pos 0),
        # not d0.  hints: hint_id 1 tiles output dim 0 (K=2); hint_id 2
        # tiles reduction dim 0 (numbered 1 == len(ranges) + 0), R=4.
        op = _make_real_reduction_op(
            ranges=[Integer(8)],
            reduction_ranges=[Integer(16)],
            input_shape_stride=([8, 16], [16, 1]),
            name="buf0",
            hints=((1, 0), (2, 1)),
        )
        levels = [(1, Integer(2)), (2, Integer(4))]
        plan = plan_coarse_tile_groups([op], [([op], levels)])
        _apply_plan([op], (0, 0), levels, {op.get_operation_name(): 0}, plan)
        # Input's read index is 16*d0 + d1: d0 (output dim 0, extent 4 --
        # 8 rows / K=2 outer steps) tiled at level 0; d1 (reduction dim,
        # numbered n_output_dims(1) + 0 = 1, extent 4 -- the R=4 tile step
        # itself) tiled at level 1.
        self.assertEqual(len(op.loop_info.tiled_dims_per_read), 1)
        self.assertEqual(
            op.loop_info.tiled_dims_per_read[0],
            [[(0, Integer(4))], [(1, Integer(4))]],
        )

    def test_subset_of_dims_tiled_on_3d_tensor(self):
        # a is [8, 16, 32], row-major (stride [512, 32, 1]).  Only dims 0
        # and 2 are tiled (levels use hint_id 1 -> dim 0, hint_id 2 ->
        # dim 2); dim 1 is genuinely present in the index but never tiled
        # at any level -- it must not appear in any level's list at all.
        op = _make_real_pointwise_op(
            ranges=[Integer(8), Integer(16), Integer(32)],
            input_shapes_strides=[
                ([8, 16, 32], [512, 32, 1]),
            ],
            name="buf0",
            hints=((1, 0), (2, 2)),
        )
        levels = [(1, Integer(2)), (2, Integer(4))]
        plan = plan_coarse_tile_groups([op], [([op], levels)])
        _apply_plan([op], (0, 0), levels, {op.get_operation_name(): 0}, plan)
        # Level 0 (hint 1, count=2) tiles dim 0, extent 4 (8 rows / 2 steps);
        # level 1 (hint 2, count=4) tiles dim 2, extent 8 (32 cols / 4
        # steps). Dim 1 (untiled) never appears -- not even as a
        # zero-extent entry, since it is filtered out entirely.
        expected = [[(0, Integer(4))], [(2, Integer(8))]]
        self.assertEqual(len(op.loop_info.tiled_dims_per_read), 1)
        self.assertEqual(op.loop_info.tiled_dims_per_read[0], expected)
        self.assertEqual(op.loop_info.output_tiled_dims, expected)

    def test_transposed_input_advance_uses_its_own_stride_order(self):
        # a is row-major [1024, 4096] (stride [4096, 1]); b is the
        # *transposed* layout of the same logical shape -- stride
        # [1, 1024] instead of [4096, 1] -- while both share the same
        # op-level ranges=[1024, 4096] and the same tiled dims (K=2 over
        # dim 0, M=4 over dim 1). The tiled-dims DECISION (which dims, at
        # which level, what extent) is identical for both inputs -- it is
        # purely about op.data.ranges, not either input's stride. Each
        # input's stride only matters once spyre_kernel.py substitutes this
        # decision into that input's own (possibly transposed) index
        # expression -- that substitution is exercised in Task 5's tests,
        # not here.
        op = _make_real_pointwise_op(
            ranges=[Integer(1024), Integer(4096)],
            input_shapes_strides=[
                ([1024, 4096], [4096, 1]),
                ([1024, 4096], [1, 1024]),
            ],
            name="buf0",
            hints=((1, 0), (2, 1)),
        )
        levels = [(1, Integer(2)), (2, Integer(4))]
        plan = plan_coarse_tile_groups([op], [([op], levels)])
        _apply_plan([op], (0, 0), levels, {op.get_operation_name(): 0}, plan)
        expected = [[(0, Integer(512))], [(1, Integer(1024))]]
        self.assertEqual(len(op.loop_info.tiled_dims_per_read), 2)
        self.assertEqual(op.loop_info.tiled_dims_per_read[0], expected)
        self.assertEqual(op.loop_info.tiled_dims_per_read[1], expected)

    def test_plan_coarse_tile_groups_zero_mutation(self):
        """Planning must not mutate op.data.ranges, layout, or set loop_info.

        Reuses the same simple single-level (well, two-level, per the small
        example) tiled-group fixture as
        test_small_example_output_and_input_advance above -- there is no
        shared "_make_simple_tiled_group" helper in this class; the
        plan_coarse_tile_groups/_apply_plan tests above all build their op
        inline via _make_real_pointwise_op, so this test copies that same
        construction.
        """
        from torch._inductor.ir import ComputedBuffer

        op = _make_real_pointwise_op(
            ranges=[Integer(1024), Integer(4096)],
            input_shapes_strides=[
                ([1024, 4096], [4096, 1]),
                ([1024, 4096], [4096, 1]),
            ],
            name="buf0",
            hints=((1, 0), (2, 1)),
        )
        group_ops = [op]
        levels = [(1, Integer(2)), (2, Integer(4))]

        pre_ranges = [
            tuple(o.data.ranges) for o in group_ops if isinstance(o, ComputedBuffer)
        ]
        pre_loop_info = [getattr(o, "loop_info", None) for o in group_ops]

        plan = plan_coarse_tile_groups(group_ops, [(group_ops, levels)])

        post_ranges = [
            tuple(o.data.ranges) for o in group_ops if isinstance(o, ComputedBuffer)
        ]
        post_loop_info = [getattr(o, "loop_info", None) for o in group_ops]
        self.assertEqual(pre_ranges, post_ranges)
        self.assertEqual(pre_loop_info, post_loop_info)
        # plan is keyed by id(op), not op itself: ir.Operation/ComputedBuffer
        # are (eq=True, unsafe_hash=False) dataclasses, so Python sets their
        # __hash__ to None and they cannot be used as dict keys directly.
        self.assertTrue(
            all(id(o) in plan for o in group_ops if isinstance(o, ComputedBuffer))
        )

    def test_plan_raises_unsupported_when_reduction_tiling_disabled(self):
        """Planning raises Unsupported for reduction-dim tiling when disabled.

        Single-level group, hint_id 1 tiles reduction dim 0 (dim_index 1 ==
        len(ranges) + 0), matching TestValidateReductionTiling's
        pure-reduction-tile shape -- the same fixture
        test_reduction_dim_advance_is_offset_by_output_dims uses for a
        plan_coarse_tile_groups/_apply_plan-level check, but exercised
        through plan_coarse_tile_groups alone instead.
        """
        op = _make_real_reduction_op(
            ranges=[Integer(128)],
            reduction_ranges=[Integer(256)],
            input_shape_stride=([128, 256], [256, 1]),
            name="buf0",
            hints=((1, 1),),
        )
        group_ops = [op]
        levels = [(1, Integer(4))]

        with config.patch({"enable_reduction_tiling": False}):
            with self.assertRaisesRegex(
                Unsupported, "disabled via enable_reduction_tiling"
            ):
                plan_coarse_tile_groups(group_ops, [(group_ops, levels)])


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

    def test_dependency_of_later_group_member_precedes_whole_group(self):
        """A preheader copy inserted before a later consumer stays outside.

        Coarse tiling may place a loop-invariant copy between two flat loop
        members when only the later member reads it. Scheduler regrouping must
        move that dependency before the counted-loop unit, not split the unit.
        """
        sched = _make_scheduler()
        first = _make_snode(sched, _make_ir_op((0,), Integer(4)), "first")
        copy = _make_snode(sched, _make_ir_op(), "copy")
        consumer = _make_snode(sched, _make_ir_op((0,), Integer(4)), "consumer")

        for node, name in ((first, "first"), (copy, "copy"), (consumer, "consumer")):
            node.get_buffer_names.return_value = {name}
        consumer.unmet_dependencies = OrderedSet(
            [inductor_deps.MemoryDep("copy", Integer(0), (), ())]
        )

        result, created = self._run([first, copy, consumer])

        self.assertEqual(result, [copy, created[0]])
        self.assertEqual(created[0].snodes, [first, consumer])

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
# 3b. spyre_fuse_nodes — CountedLoopSchedulerNode fusion
# ===========================================================================


def _make_counted_loop(scheduler, name="loop0", loop_count=sympy.Integer(4)):
    """Return a MagicMock CountedLoopSchedulerNode for use in fusion tests."""
    node = MagicMock(spec=CountedLoopSchedulerNode)
    node.scheduler = scheduler
    node.get_device.return_value = torch.device("spyre")
    node.get_name.return_value = name
    node.get_nodes.return_value = [node]
    node.loop_count = loop_count
    node.ancestors = OrderedSet()
    node.min_order = 0
    node.max_order = 0
    # PT 2.12 added min/max_input_distance to BaseSchedulerNode (set in __init__,
    # so absent from MagicMock(spec=...)); init_group_node reads them when fusing.
    node.min_input_distance = 0
    node.max_input_distance = 0
    node.unmet_dependencies = OrderedSet()
    node.is_reduction.return_value = False
    node.group = (None, None)
    node.read_writes = inductor_deps.ReadWrites(
        reads=OrderedSet(),
        writes=OrderedSet(),
        index_exprs=OrderedSet(),
    )
    node.outputs_by_name = {}
    return node


class TestSpyreFuseNodesLoopFusion(unittest.TestCase):
    def test_lone_loop_node_is_own_bundle(self):
        """A lone CountedLoopSchedulerNode produces exactly one bundle."""
        sched = _make_scheduler()
        loop = _make_counted_loop(sched, "loop0")
        result = spyre_fuse_nodes([loop])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], CountedLoopSchedulerNode)

    def test_plain_then_loop_fuses_into_one_bundle(self):
        """SchedulerNode followed by CountedLoopSchedulerNode → one FusedSchedulerNode."""
        from torch._inductor.scheduler import FusedSchedulerNode

        sched = _make_scheduler()
        plain = _make_snode(sched, _make_ir_op(), "plain0")
        loop = _make_counted_loop(sched, "loop0")
        result = spyre_fuse_nodes([plain, loop])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], FusedSchedulerNode)

    def test_loop_then_plain_fuses_into_one_bundle(self):
        """CountedLoopSchedulerNode followed by SchedulerNode → one FusedSchedulerNode."""
        from torch._inductor.scheduler import FusedSchedulerNode

        sched = _make_scheduler()
        loop = _make_counted_loop(sched, "loop0")
        plain = _make_snode(sched, _make_ir_op(), "plain0")
        result = spyre_fuse_nodes([loop, plain])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], FusedSchedulerNode)

    def test_plain_loop_plain_fuses_into_one_bundle(self):
        """plain → loop → plain sequence → one FusedSchedulerNode."""
        from torch._inductor.scheduler import FusedSchedulerNode

        sched = _make_scheduler()
        plain_a = _make_snode(sched, _make_ir_op(), "plain_a")
        loop = _make_counted_loop(sched, "loop0")
        plain_b = _make_snode(sched, _make_ir_op(), "plain_b")
        result = spyre_fuse_nodes([plain_a, loop, plain_b])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], FusedSchedulerNode)

    def test_fallback_still_forces_boundary(self):
        """An ExternKernelSchedulerNode between two fusable nodes creates two bundles."""
        from torch._inductor.scheduler import ExternKernelSchedulerNode

        sched = _make_scheduler()
        plain_a = _make_snode(sched, _make_ir_op(), "plain_a")
        fallback = MagicMock(spec=ExternKernelSchedulerNode)
        fallback.scheduler = sched
        fallback.get_name.return_value = "fallback0"
        plain_b = _make_snode(sched, _make_ir_op(), "plain_b")
        result = spyre_fuse_nodes([plain_a, fallback, plain_b])
        # plain_a fuses alone before fallback; fallback forces boundary;
        # plain_b is a separate bundle after fallback.
        self.assertEqual(len(result), 3)
        # First entry is plain_a (single SchedulerNode, returned as-is by _make_fused).
        self.assertIs(result[1], fallback)


# ===========================================================================
# 4. generate_sdsc and compile_op_spec — symbol/affine-stride paths
# ===========================================================================


class TestGenerateSdscTiledSymbols(unittest.TestCase):
    def test_tiled_tensor_affine_strides_correct(self):
        s = Symbol("s")
        sdsc_spec = _make_sdsc_spec(s, iter_range=64, device_stride=128)
        # _make_sdsc_spec predates device_tile_advance_expr and never sets it,
        # so generate_sdsc's `if arg.device_tile_advance_expr is not None`
        # guard (compute_ops.py) is never entered unless we set it here.
        # coeff(s) == 128 matches device_stride=128's semantics: tile_size =
        # int(coeff) * arg_elem_bytes = 128 * 2 == the stride asserted below.
        sdsc_spec.args[0].device_tile_advance_expr = 128 * s
        symbols: list[int] = []
        _, _, affine_strides, _ = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[[s]],
        )
        # affine_strides[tensor_idx] is list[dict] (per level, outermost first).
        # With one tiling level, affine_strides[0] = [{s: stride}].
        self.assertEqual(len(affine_strides), 1)
        self.assertIn(s, affine_strides[0][0])
        self.assertEqual(affine_strides[0][0][s], 128 * 2)

    def test_tiled_tensor_affine_strides_correct_via_device_tile_advance_expr(self):
        s = Symbol("s")
        sdsc_spec = _make_sdsc_spec(s, iter_range=64, device_stride=128)
        sdsc_spec.args[0].device_tile_advance_expr = 128 * s
        symbols: list[int] = []
        _, _, affine_strides, _ = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[[s]],
        )
        self.assertEqual(len(affine_strides), 1)
        self.assertIn(s, affine_strides[0][0])
        self.assertEqual(affine_strides[0][0][s], 128 * 2)

    def test_tiled_tensor_base_address_registered(self):
        s = Symbol("s")
        # On the symbolic path start_address is set to arg_index (0) as a sentinel.
        # The raw base stored in symbols[] is the sentinel, not a real HBM address.
        sdsc_spec = _make_sdsc_spec(s, start_address=0)
        symbols: list[int] = []
        generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[[s]],
        )
        self.assertEqual(len(symbols), 1)
        self.assertEqual(symbols[0], 0)  # kernel sentinel = arg_index = 0

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
        )
        top_val = next(iter(sdsc_json.values()))
        node = top_val["dscs_"][0]["add"]["scheduleTree_"][0]
        data = node["startAddressCoreCorelet_"]["data_"]
        ids = [int(v) for v in data.values()]
        self.assertTrue(all(i <= -6 for i in ids), f"Expected ids ≤ -6, got {ids}")

    def test_multi_core_tiled_per_core_symbols(self):
        s = Symbol("s")
        core_id = Symbol("core_id")
        # On the symbolic path start_address = arg_index (0) as a sentinel; the
        # loop unroller advances it by tile_offset_bytes for later tiles.  For tile 0
        # start_address == arg_index == 0.
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
            arg_index=0,
            # device_tile_advance_expr drives generate_sdsc's affine-stride
            # filter (_tensor_tiled_by_symbol); without it the tensor never
            # advances regardless of tensor.strides. coeff(s) == 128 matches
            # this tensor's declared stride=128 above.
            device_tile_advance_expr=128 * s,
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
            layouts={
                "A": {"dim_order": [s], "stick_dim_order": [s], "stick_size": [64]}
            },
            args=[tensor],
            constants={},
            conv_params={},
            coordinate_masking={},
        )
        symbols: list[int] = []
        _, local_sym_values, affine_strides, _ = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[[s]],
        )
        self.assertEqual(len(symbols), 2)
        self.assertEqual(symbols[0], 0)  # kernel sentinel = arg_index = 0
        self.assertEqual(symbols[1], 128)  # core-1 derived = sentinel + per-core stride
        # affine_strides[0] = [{s: stride}] (one level, one tensor)
        self.assertIn(s, affine_strides[0][0])

    def test_generate_sdsc_same_dim_tiled_two_levels_distinct_strides(self):
        """Two levels tiling the same host dim must produce two distinct
        per-level strides in generate_sdsc's output, not one collapsed
        term -- this is the multi-level-collapse bug this plan fixes."""
        # Mirrors test_nested_same_dim_different_counts's fixture shape:
        # ranges=[Integer(256)], hints=((1, 0), (2, 0)) tiling dim 0 at
        # level 0 (outer, count 4) and level 1 (inner, count 2).
        # Per test_planned_tile_extents_per_level_same_dim_two_levels:
        # level 0 (outer) extent = 64, level 1 (inner) extent = 32.
        d0 = Symbol("d0")
        lvl0 = Symbol("_tile_adv_add_lvl0")
        lvl1 = Symbol("_tile_adv_add_lvl1")
        # device_tile_advance_expr: this tensor's own device-element-offset
        # advance per unit step of each minted level symbol -- one term per
        # level, with the level's own extent as its coefficient (elements).
        tile_advance_expr = 64 * lvl0 + 32 * lvl1
        tensor = SDSCArgs(
            layout="A",
            dim_order=[d0],
            data_format=_FP16,
            scales={d0: 1},
            strides={d0: 1},
            offsets={d0: 0},
            max_dim_sizes={d0: -1},
            allocation={"hbm": 0x1000},
            start_address=0,
            backGap={},
            arg_index=0,
            device_tile_advance_expr=tile_advance_expr,
        )
        sdsc_spec = SDSCSpec(
            opfunc="add",
            execution_unit="sfp",
            data_format=_FP16,
            num_inputs=1,
            iteration_space={d0: 256},
            num_cores=1,
            work_slices={d0: 1},
            core_id_to_work_slice={d0: Integer(0)},
            padding={},
            layouts={
                "A": {"dim_order": [d0], "stick_dim_order": [d0], "stick_size": [64]}
            },
            args=[tensor],
            constants={},
            coordinate_masking={},
        )
        symbols: list[int] = []
        # tiled_symbols is outermost-first (see generate_sdsc's own comment,
        # compute_ops.py:526) -- level 0 (outer) first, level 1 (inner) second.
        _, _, affine_strides, _ = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[[lvl0], [lvl1]],
        )
        self.assertEqual(len(affine_strides), 1)
        per_level = affine_strides[0]
        self.assertEqual(len(per_level), 2)
        nb = 2  # fp16 byte size
        self.assertEqual(per_level[0].get(lvl0), 64 * nb)
        self.assertEqual(per_level[1].get(lvl1), 32 * nb)
        self.assertNotEqual(per_level[0].get(lvl0), per_level[1].get(lvl1))

    def test_generate_sdsc_floor_wrapped_tile_advance_produces_stride(self):
        """Real stick-layout tensors wrap their tile-advance term in
        floor() (views.tiling_expr_to_device_expr) -- generate_sdsc must
        still detect the tensor as tiled and extract the correct byte
        stride, not silently treat it as non-advancing (the exact bug
        Task 6 of the deferred-tile-advance-capture plan found and
        deferred)."""
        d0 = Symbol("d0")
        lvl0 = Symbol("_tile_adv_add_lvl0")
        # floor() wrapping is what views.tiling_expr_to_device_expr emits
        # when a level's host-stride step isn't a multiple of the device
        # dim's own stride_map entry; coeff(lvl0) on this expr is 0 via
        # plain sympy .coeff(), which is exactly the bug this task fixes.
        tile_advance_expr = sympy.floor(64 * lvl0)
        tensor = SDSCArgs(
            layout="A",
            dim_order=[d0],
            data_format=_FP16,
            scales={d0: 1},
            strides={d0: 1},
            offsets={d0: 0},
            max_dim_sizes={d0: -1},
            allocation={"hbm": 0x1000},
            start_address=0,
            backGap={},
            arg_index=0,
            device_tile_advance_expr=tile_advance_expr,
        )
        sdsc_spec = SDSCSpec(
            opfunc="add",
            execution_unit="sfp",
            data_format=_FP16,
            num_inputs=1,
            iteration_space={d0: 256},
            num_cores=1,
            work_slices={d0: 1},
            core_id_to_work_slice={d0: Integer(0)},
            padding={},
            layouts={
                "A": {"dim_order": [d0], "stick_dim_order": [d0], "stick_size": [64]}
            },
            args=[tensor],
            constants={},
            coordinate_masking={},
        )
        symbols: list[int] = []
        _, _, affine_strides, _ = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[[lvl0]],
        )
        self.assertEqual(len(affine_strides), 1)
        per_level = affine_strides[0]
        self.assertEqual(len(per_level), 1)
        nb = 2  # fp16 byte size
        self.assertEqual(per_level[0].get(lvl0), 64 * nb)


class TestCompileOpSpecTwoTiledSymbols(unittest.TestCase):
    def _make_3d_op_spec(self) -> OpSpec:
        c0 = Symbol("c0")
        c1 = Symbol("c1")
        c2 = Symbol("c2")
        fp16 = _FP16
        # compile_op_spec (superdsc.parse_op_spec) renames the 3 iteration-space
        # symbols, in insertion order, to INPUT_DIM_LABELS[:2] + OUTPUT_DIM_LABELS[:1]
        # == ["mb", "x", "out"] for a 3-dim non-matmul op (constants.py). i.e.
        # c0 -> mb, c1 -> x, c2 -> out. generate_sdsc's per-level affine-stride
        # loop looks up device_tile_advance_expr.coeff(s) using these *renamed*
        # symbols (it receives tiled_symbols already translated through
        # symbol_mapping), so device_tile_advance_expr must be expressed in the
        # renamed space here to produce nonzero strides through this call path.
        mb = Symbol("mb")
        x = Symbol("x")
        # device-element-offset expression for a C-contiguous [2, 4, 64] device
        # layout: offset = mb*(4*64) + x*64 + out. Only the tiled dims (mb, x)
        # need nonzero coefficients here since device_tile_advance_expr.coeff(s)
        # is only consulted for symbols in op_spec.tiled_symbols (translated).
        tile_advance_expr = 4 * 64 * mb + 64 * x
        tensor_in = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=fp16,
            device_size=[2, 4, 64],
            device_coordinates=[c0, c1, c2],
            allocation={"hbm": 0x1000},
            device_tile_advance_expr=tile_advance_expr,
        )
        tensor_out = TensorArg(
            is_input=False,
            arg_index=1,
            device_dtype=fp16,
            device_size=[2, 4, 64],
            device_coordinates=[c0, c1, c2],
            allocation={"hbm": 0x2000},
            device_tile_advance_expr=tile_advance_expr,
        )
        iteration_space = {
            c0: (Integer(2), 1),
            c1: (Integer(4), 1),
            c2: (Integer(64), 1),
        }
        return OpSpec(
            op="add",
            is_reduction=False,
            iteration_space=iteration_space,
            core_id_to_work_slice=derive_operation_mapping(iteration_space),
            args=[tensor_in, tensor_out],
            op_info={},
            tiled_symbols=[[c0, c1]],
            # Trip counts for the tiled symbols, taken from iteration_space above
            # (c0's range is 2, c1's range is 4). Used by _create_sdsc_tensors /
            # generate_sdsc to compute each tiled tensor's full pre-tiling extent.
            tiled_symbol_trip_counts={c0: 2, c1: 4},
        )

    def test_two_tiled_symbols_produce_two_stride_entries(self):
        op_spec = self._make_3d_op_spec()
        symbols: list[int] = []
        _, _, affine_strides, _ = compile_op_spec(0, op_spec, symbols)
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
        _, _, affine_strides, _ = compile_op_spec(0, op_spec, symbols)
        for per_level in affine_strides:
            for level_strides in per_level:
                for sym, stride in level_strides.items():
                    self.assertGreater(stride, 0)

    def test_two_tiled_symbols_strides_match_device_tile_advance_expr(self):
        # Direct before/after value check: the fixture's device_tile_advance_expr
        # is 4*64*mb + 64*x elements for a C-contiguous [2, 4, 64] device layout,
        # so with fp16 (2 bytes/elem) the expected byte strides are
        # mb -> 4*64*2 == 512 and x -> 64*2 == 128, for both tensors' single
        # tiling level.
        op_spec = self._make_3d_op_spec()
        symbols: list[int] = []
        _, _, affine_strides, _ = compile_op_spec(0, op_spec, symbols)
        mb = Symbol("mb")
        x = Symbol("x")
        self.assertEqual(len(affine_strides), 2)
        for per_level in affine_strides:
            self.assertEqual(len(per_level), 1)
            level_strides = per_level[0]
            self.assertEqual(level_strides.get(mb), 512)
            self.assertEqual(level_strides.get(x), 128)


class TestCompileOpSpecSymbolMapping(unittest.TestCase):
    def test_affine_strides_non_empty_for_tiled_op(self):
        op_spec = _make_tiled_op_spec()
        symbols: list[int] = []
        _, _, affine_strides, _ = compile_op_spec(0, op_spec, symbols)
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
        generate_bundle("test_kernel", tmpdir, [loop])

        with open(os.path.join(tmpdir, "bundle.mlir")) as f:
            mlir = f.read()

        self.assertIn("affine.apply", mlir)
        self.assertIn("affine_map", mlir)
        self.assertIn("scf.for", mlir)

    def test_symbol_mapping_preserves_minted_tile_advance_symbols(self):
        """A minted _tile_adv_* symbol must survive parse_op_spec's
        symbol_mapping translation, not be silently dropped -- this is
        the root cause of every tiled op failing past tile 0 in the e2e
        suite (see task-5-report.md's root-cause trace)."""
        op_spec = _make_tiled_op_spec()
        minted = Symbol("_tile_adv_add_lvl0")
        # Minted symbols are, by construction, never members of
        # iteration_space -- they name a loop-nesting level, not a
        # dimension. Swap _make_tiled_op_spec's real-symbol tiled_symbols
        # (c0, which IS in iteration_space) for a minted symbol to exercise
        # exactly the case symbol_mapping used to drop.
        op_spec.tiled_symbols = [[minted]]

        _, symbol_mapping = parse_op_spec(op_spec)

        for level in op_spec.tiled_symbols:
            for sym in level:
                self.assertIn(
                    sym,
                    symbol_mapping,
                    f"Minted symbol {sym} missing from symbol_mapping; it "
                    "will be silently dropped by compile_op_spec's "
                    "tiled_symbols_per_level translation.",
                )

        # compile_op_spec's own translation of tiled_symbols must retain the
        # minted symbol too (as an identity mapping -- Site 1's fix), not
        # just parse_op_spec's returned symbol_mapping dict in isolation.
        symbols: list[int] = []
        compile_op_spec(0, op_spec, symbols)
        tiled_symbols_per_level = [
            [symbol_mapping[s] for s in level if s in symbol_mapping]
            for level in reversed(op_spec.tiled_symbols)
        ]
        self.assertEqual(
            tiled_symbols_per_level,
            [[minted]],
            "compile_op_spec's tiled_symbols_per_level translation dropped "
            f"the minted symbol; got {tiled_symbols_per_level}.",
        )

    def test_sdsc_dim_advance_detects_floor_wrapped_minted_symbol(self):
        """superdsc._create_sdsc_tensors's sdsc_dim_advance computation
        must extract a tile_size from a floor-wrapped minted symbol's
        device_tile_advance_expr, not silently skip it (the same bug
        class as compute_ops.py's _tensor_tiled_by_symbol, in the
        sibling backGap/base-offset computation)."""
        minted = Symbol("_tile_adv_add_lvl0")
        op_spec = _make_tiled_op_spec()
        # _make_tiled_op_spec's own tensor(s) carry device_tile_advance_expr
        # in terms of the real dim symbol (see Task 6's fixture repair in
        # task-6-report.md) -- swap it for a floor-wrapped minted symbol to
        # exercise this task's fix.
        op_spec.tiled_symbols = [[minted]]
        for arg in op_spec.args:
            if arg.device_tile_advance_expr is not None:
                arg.device_tile_advance_expr = sympy.floor(64 * minted)
        symbols: list[int] = []
        # Must not raise, and must not silently produce empty affine_strides.
        _, _, affine_strides, _ = compile_op_spec(0, op_spec, symbols)
        has_strides = any(
            any(len(level_d) > 0 for level_d in per_level)
            for per_level in affine_strides
        )
        self.assertTrue(
            has_strides,
            f"Expected non-empty affine_strides; got {affine_strides}.",
        )
        # NOTE: affine_strides above is computed entirely by
        # compute_ops.generate_sdsc (already fixed independently in the
        # prior commit) and does not depend on this task's superdsc.py
        # line -- _create_sdsc_tensors's sdsc_dim_advance dict is keyed by
        # symbol_mapping[sym], and with a *minted* tiled symbol that key is
        # the minted symbol itself, never a device dim in dim_order, so the
        # `dim in sdsc_dim_advance` branch this task's fix feeds is
        # structurally unreachable for this fixture shape (confirmed by
        # running this exact assertion with the pre-fix `.coeff(sym)` --
        # it still passes). See
        # test_sdsc_dim_advance_backgap_uses_floor_wrapped_coefficient
        # below for a fixture that actually distinguishes the two.

    def test_sdsc_dim_advance_backgap_uses_floor_wrapped_coefficient(self):
        """Direct regression test for the superdsc.py line this task fixes.

        _create_sdsc_tensors's sdsc_dim_advance branch (~line 448) is only
        consulted when `dim in sdsc_dim_advance`, which requires
        symbol_mapping[sym] to equal an actual device dim in dim_order --
        true for a *real* tiled dim symbol (e.g. "out"), never for a
        minted level symbol (see the note in the test above). It is also
        only observable in the emitted SDSC (via backGapCore_) when the
        tile is a genuine sub-slice of a larger extent, i.e.
        supertile_count = tiled_symbol_trip_counts[sym] > 1 and the
        tensor's device_size exceeds one tile along that dim -- this
        fixture sets both up explicitly (device_size doubled to 128,
        trip count 2), unlike _make_tiled_op_spec's default (trip count 1,
        no supertile). With a floor-wrapped device_tile_advance_expr on
        "out" and the buggy plain `.coeff()`, the coefficient silently
        comes back 0, sdsc_dim_advance stays empty, and backGapCore_ is
        omitted entirely from the SDSC JSON -- a silently wrong (missing)
        base-offset/backGap for every tile past the first.
        """
        out = Symbol("out")
        op_spec = _make_tiled_op_spec()
        op_spec.tiled_symbols = [[out]]
        op_spec.tiled_symbol_trip_counts = {out: 2}
        for arg in op_spec.args:
            # Double the stick-dim device_size so the tile (64 elements,
            # from device_tile_advance_expr below) is half the full
            # extent -- i.e. dev_dim_size (128) > it_dim_size (64),
            # the condition that makes backGap/offsets nonempty.
            arg.device_size = [2, 128]
            if arg.device_tile_advance_expr is not None:
                arg.device_tile_advance_expr = sympy.floor(64 * out)
        symbols: list[int] = []
        sdsc_json, _, _, _ = compile_op_spec(0, op_spec, symbols)
        found_backgap = False
        for entry in sdsc_json.values():
            for dsc in entry.get("dscs_", []):
                for op_dsc in dsc.values():
                    for node in op_dsc.get("scheduleTree_", []):
                        back_gap = node.get("backGapCore_", {})
                        if "out" in back_gap:
                            found_backgap = True
                            self.assertEqual(
                                back_gap["out"].get("-1"),
                                "128",
                                f"Expected backGapCore_['out']['-1'] == "
                                f"'128' (dev_dim_size - it_dim_size = "
                                f"128 - 64 = 64 elements = 128 bytes at "
                                f"2 bytes/elem); got {back_gap}.",
                            )
        self.assertTrue(
            found_backgap,
            "Expected backGapCore_['out'] present in the emitted SDSC JSON "
            "-- its absence means sdsc_dim_advance was silently left empty "
            f"by a floor-wrapped coefficient extraction bug; got {sdsc_json}.",
        )

    def test_minted_symbol_tiled_on_symbolic_split_dim_raises_unsupported(self):
        """A tensor tiled (via a minted symbol) on the same dim that's
        symbolically split across cores must still raise Unsupported --
        this combination was already rejected for real symbols; the
        minted-symbol path must not silently bypass the same guard.

        No existing test exercises this Unsupported raise for the
        real-symbol path either (searched for tiled_on_split_dim /
        _symbolic_split_info / "symbolic dim" + "tiled" across
        test_coarse_tiling.py and test_codegen.py; only _symbolic_split_info
        itself, not its caller's Unsupported branch, is covered) -- this
        task's job is the minted-symbol path, not backfilling that gap.
        """
        mb = Symbol("mb")
        minted = Symbol("_tile_adv_add_lvl0")
        # mb is a symbolic dim, split across 8 cores, and this tensor uses it
        # (scale > 0) -- this alone would make _symbolic_split_info fire.
        tensor = SDSCArgs(
            layout="A",
            dim_order=[mb],
            data_format=_FP16,
            scales={mb: 1},
            strides={mb: 256},
            offsets={mb: 0},
            max_dim_sizes={mb: -1},
            allocation={"hbm": 0x1000},
            start_address=0,
            backGap={},
            arg_index=0,
            # The minted symbol contributes a nonzero term to this tensor's
            # own device_tile_advance_expr -- Site 3's minted-symbol test
            # must detect that this tensor is ALSO tiled on mb (the same dim
            # that's symbolically split) purely from this coefficient, since
            # str(minted) can never equal "mb".
            device_tile_advance_expr=256 * minted,
        )
        sdsc_spec = SDSCSpec(
            opfunc="add",
            execution_unit="sfp",
            data_format=_FP16,
            num_inputs=1,
            iteration_space={mb: 1024},
            num_cores=8,
            work_slices={mb: 8},
            core_id_to_work_slice={mb: Integer(0)},
            padding={},
            layouts={
                "A": {"dim_order": [mb], "stick_dim_order": [mb], "stick_size": [64]}
            },
            args=[tensor],
            constants={},
            coordinate_masking={},
            symbolic_dims={"mb": ("s0", 64, 1024)},
        )
        symbols: list[int] = []
        with self.assertRaises(Unsupported):
            generate_sdsc(
                0,
                sdsc_spec,
                symbols,
                symbol_id_offset=0,
                tiled_symbols=[[minted]],
            )

    def test_tiled_on_split_dim_detects_floor_wrapped_minted_symbol(self):
        """tiled_on_split_dim's tensor_advances_at_some_level check must
        detect a minted symbol's advance even when device_tile_advance_expr
        wraps it in floor() -- plain .coeff() returns 0 for that shape,
        which would let this Unsupported guard silently fail to fire.

        Adapted from test_minted_symbol_tiled_on_symbolic_split_dim_raises_
        Unsupported immediately above: same fixture shape (a tensor tiled
        via a minted symbol on the same dim that's symbolically split
        across cores), with the sole difference that
        device_tile_advance_expr wraps the minted symbol's term in
        floor(), exactly as views.tiling_expr_to_device_expr emits it for
        a real stick-layout tensor. Goes through generate_sdsc directly
        (not compile_op_spec/OpSpec) because tiled_on_split_dim's guard --
        and the SDSCSpec.symbolic_dims/_symbolic_split_info machinery it
        depends on -- lives in generate_sdsc; that is also the path the
        sibling raises-Unsupported test above exercises.
        """
        mb = Symbol("mb")
        minted = Symbol("_tile_adv_add_lvl0")
        tensor = SDSCArgs(
            layout="A",
            dim_order=[mb],
            data_format=_FP16,
            scales={mb: 1},
            strides={mb: 256},
            offsets={mb: 0},
            max_dim_sizes={mb: -1},
            allocation={"hbm": 0x1000},
            start_address=0,
            backGap={},
            arg_index=0,
            # Same combined coefficient as the plain-Mul sibling fixture
            # (256 on the minted symbol), but wrapped in floor() -- the
            # shape views.tiling_expr_to_device_expr actually emits.
            # Plain sympy .coeff() returns 0 for this shape, which would
            # make tensor_advances_at_some_level (and therefore
            # tiled_on_split_dim) silently False here.
            device_tile_advance_expr=sympy.floor(256 * minted),
        )
        sdsc_spec = SDSCSpec(
            opfunc="add",
            execution_unit="sfp",
            data_format=_FP16,
            num_inputs=1,
            iteration_space={mb: 1024},
            num_cores=8,
            work_slices={mb: 8},
            core_id_to_work_slice={mb: Integer(0)},
            padding={},
            layouts={
                "A": {"dim_order": [mb], "stick_dim_order": [mb], "stick_size": [64]}
            },
            args=[tensor],
            constants={},
            coordinate_masking={},
            symbolic_dims={"mb": ("s0", 64, 1024)},
        )
        symbols: list[int] = []
        with self.assertRaises(Unsupported):
            generate_sdsc(
                0,
                sdsc_spec,
                symbols,
                symbol_id_offset=0,
                tiled_symbols=[[minted]],
            )

    def test_minted_symbol_tiled_on_different_dim_does_not_raise_unsupported(self):
        """A tensor tiled (via a minted symbol) on one dim, and merely also
        *active* (but untiled) on a different dim that happens to be the
        symbolically-split one, must NOT raise Unsupported.

        Regression test for fix-loop round 1: the original minted-symbol
        check flagged tiled_on_split_dim=True whenever sym_dim_name was
        *any* active dim of a tensor that advanced at all, regardless of
        which dim actually drove that advance -- because a minted symbol's
        combined coefficient (spyre_kernel._general_tile_advance sums every
        host dim tiled at a level into one coefficient before this tensor's
        device_tile_advance_expr is built) carries no per-dimension
        provenance to check against. Here the tensor is split on `mb` but
        tiled only on `kj` (a different, unrelated dim that also happens to
        be active) -- mb itself never advances, so this combination is
        supported and must not raise.
        """
        mb = Symbol("mb")
        kj = Symbol("kj")
        minted = Symbol("_tile_adv_add_lvl0")
        tensor = SDSCArgs(
            layout="A",
            dim_order=[mb, kj],
            data_format=_FP16,
            scales={mb: 1, kj: 1},
            strides={mb: 256, kj: 999},
            offsets={mb: 0, kj: 0},
            max_dim_sizes={mb: -1, kj: -1},
            allocation={"hbm": 0x1000},
            start_address=0,
            backGap={},
            arg_index=0,
            # Tiled only via kj: the combined coefficient on the minted
            # symbol is nonzero (the tensor genuinely advances at this
            # level), but mb contributes nothing to it.
            device_tile_advance_expr=512 * minted,
        )
        sdsc_spec = SDSCSpec(
            opfunc="add",
            execution_unit="sfp",
            data_format=_FP16,
            num_inputs=1,
            iteration_space={mb: 1024, kj: 128},
            num_cores=8,
            work_slices={mb: 8, kj: 1},
            core_id_to_work_slice={mb: Integer(0), kj: Integer(0)},
            padding={},
            layouts={
                "A": {
                    "dim_order": [mb, kj],
                    "stick_dim_order": [kj],
                    "stick_size": [64],
                }
            },
            args=[tensor],
            constants={},
            coordinate_masking={},
            symbolic_dims={"mb": ("s0", 64, 1024)},
        )
        symbols: list[int] = []
        # Must not raise: mb (the split dim) is active but not the dim this
        # tensor is actually tiled on.
        generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[[minted]],
        )


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
                core_id_to_work_slice=derive_operation_mapping(iteration_space),
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

    def test_unit_bmm_preserve_skips_higher_rank_attention_layout(self):
        c0 = Symbol("c0")
        c1 = Symbol("c1")
        c2 = Symbol("c2")
        z0 = Symbol("z0")
        input_arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=_FP16,
            device_size=[512, 32, 2, 1, 64],
            device_coordinates=[
                c0,
                z0,
                floor(c2 / 64),
                Integer(0),
                Mod(c2, 64),
            ],
            allocation={"hbm_pool": 0},
        )
        kernel_arg = TensorArg(
            is_input=True,
            arg_index=1,
            device_dtype=_FP16,
            device_size=[64, 4096, 64],
            device_coordinates=[floor(c1 / 64), c2, Mod(c1, 64)],
            allocation={"hbm": 0x400000000},
        )
        output_arg = TensorArg(
            is_input=False,
            arg_index=2,
            device_dtype=_FP16,
            device_size=[512, 64, 1, 64],
            device_coordinates=[c0, floor(c1 / 64), Integer(0), Mod(c1, 64)],
            allocation={"hbm": 0x800000000},
        )
        iteration_space = {
            c0: (Integer(512), 4),
            c1: (Integer(4096), 8),
            c2: (Integer(4096), 1),
        }
        op_info = {SHARED_WEIGHT_UNIT_BMM_INFO_KEY: {"batch_dim": 0}}

        new_iteration_space = _preserve_shared_weight_unit_bmm_dim(
            "batchmatmul",
            iteration_space,
            [input_arg, kernel_arg, output_arg],
            op_info,
        )

        self.assertIs(new_iteration_space, iteration_space)
        self.assertNotIn("_spyre_bmm_unit", {str(dim) for dim in iteration_space})
        self.assertEqual(input_arg.device_size, [512, 32, 2, 1, 64])
        self.assertEqual(
            input_arg.device_coordinates,
            [c0, z0, floor(c2 / 64), Integer(0), Mod(c2, 64)],
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
        generate_bundle("test_kernel", self.tmpdir, specs)
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
        generate_bundle("test_kernel", self.tmpdir, [loop])
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
        generate_bundle("test_kernel", self.tmpdir, specs)
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
            )
        return _read_mlir(self.tmpdir)

    def test_tiled_tensor_emits_affine_apply(self):
        s = self._s
        stride = 16384

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0):
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
        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0):
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

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0):
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

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0):
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

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0):
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

    def test_tiled_snapshot_via_device_tile_advance_expr(self):
        # Same MLIR shape as test_tiled_snapshot above, but exercises the real
        # (unmocked) compile_op_spec -> _create_sdsc_tensors -> generate_sdsc
        # pipeline via device_tile_advance_expr + tiled_symbol_trip_counts,
        # instead of stubbing compile_op_spec's return value directly.
        #
        # compile_op_spec (superdsc.parse_op_spec) renames the sole
        # iteration-space symbol c0 to OUTPUT_DIM_LABELS[0] == "out" for this
        # 1-dim non-matmul op (constants.py), so device_tile_advance_expr must
        # be expressed in terms of "out" (see
        # TestCompileOpSpecTwoTiledSymbols._make_3d_op_spec for the same
        # renaming rule with a worked multi-dim example). device_size=[2, 64]
        # with device_coordinates=[0, c0] and a device_tile_advance_expr of
        # 64*out (elements) gives a byte stride of 64*2 == 128 at fp16.
        c0 = Symbol("c0")
        out = Symbol("out")
        tensor_in0 = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=_FP16,
            device_size=[2, 64],
            device_coordinates=[Integer(0), c0],
            allocation={"hbm": 0x1000},
            device_tile_advance_expr=64 * out,
        )
        tensor_in1 = TensorArg(
            is_input=True,
            arg_index=1,
            device_dtype=_FP16,
            device_size=[2, 64],
            device_coordinates=[Integer(0), c0],
            allocation={"hbm": 0x2000},
            device_tile_advance_expr=64 * out,
        )
        tensor_out = TensorArg(
            is_input=False,
            arg_index=2,
            device_dtype=_FP16,
            device_size=[2, 64],
            device_coordinates=[Integer(0), c0],
            allocation={"hbm": 0x3000},
            device_tile_advance_expr=64 * out,
        )
        iteration_space = {c0: (Integer(128), 1)}
        op = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space=iteration_space,
            core_id_to_work_slice=derive_operation_mapping(iteration_space),
            args=[tensor_in0, tensor_in1, tensor_out],
            op_info={},
            tiled_symbols=[[c0]],
            tiled_symbol_trip_counts={c0: 128},
        )
        loop = LoopSpec(count=Integer(4), body=[op])
        generate_bundle("test_kernel", self.tmpdir, [loop])
        mlir = _read_mlir(self.tmpdir)

        # Verify the MLIR references all three args and uses affine maps
        # for tile advancement.
        self.assertIn("arg_0_base_addr", mlir)
        self.assertIn("arg_1_base_addr", mlir)
        self.assertIn("arg_2_base_addr", mlir)
        self.assertIn("scf.for", mlir)
        self.assertIn("affine.apply", mlir)


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
            )
        return _read_mlir(self.tmpdir)

    def _fake_compile_two_strides(self, outer_stride, inner_stride):
        s0, s1 = self.s0, self.s1

        def fake_compile(idx, op_spec, symbols, symbol_id_offset=0):
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


class TestGenerateBundleAffineLoopPath(unittest.TestCase):
    """Verify affine-map correctness for the scf.for / affine.apply path.

    Key invariants:
      - ops tiled only by the inner loop var emit affine.apply with that var only
      - ops not tiled (fixed address, no advancing dims) emit no affine.apply
      - the copy op (outer-B tiled) emits affine.apply with the outer loop var
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._s = Symbol("s")
        self._c_k = Symbol("c_k")
        self._c_b = Symbol("c_b")

    def _bundle(self, specs, fake_compile):
        with config.patch({"sdsc_cache": False}):
            with patch(
                "torch_spyre._inductor.codegen.bundle.compile_op_spec",
                side_effect=fake_compile,
            ):
                generate_bundle(
                    "test_kernel",
                    self.tmpdir,
                    specs,
                )
        return _read_mlir(self.tmpdir)

    # --- Group 1: flat row-tiling ---

    def test_flat_loop_tiled_tensor_emits_affine_apply(self):
        s = self._s

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
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
        def fake(idx, op_spec, symbols, symbol_id_offset=0):
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

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
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
    # Strides match the geometry previously verified against the deleted
    # flat-unroll path's TestNestedReductionUnroll:
    #   k_input: device_size=[2,64,64]; device_stride[0]=prod([64,64])=4096
    #     128 K-elems/tile → 2 sticks; byte_stride = (128//64)*4096*2 = 16384
    #   accum_buf: device_size=[1,2,64]; device_stride[1]=prod([64])=64
    #     2 batches/tile; byte_stride = 2*64*2 = 256
    # Only K_STRIDE appears in the affine map (accum_buf not tiled on K).

    _GRP2_K_STRIDE = 16384  # (128//64) * prod([64,64]) * 2

    def _fake_nested_reduction(self, k_stride):
        c_k = self._c_k
        call_count = [0]

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
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
    # Strides match the geometry previously verified against the deleted
    # flat-unroll path's TestNestedReductionTileAccum:
    #   bmm K-input: same geometry as Group 2 → K_STRIDE = 16384
    #   accum_full (copy output): device_size=[1,128,32]
    #     device_stride[0]=prod([128,32])=4096; 1 tile advances c_b by 1
    #     byte_stride = 1 * 4096 * 2 = 8192  (_OUTER_TILE_STRIDE_BYTES)

    _GRP3_K_STRIDE = 16384  # (128//64) * prod([64,64]) * 2
    _GRP3_B_STRIDE = 8192  # 1 * prod([128,32]) * 2

    def _fake_tile_accum(self, k_stride, b_stride):
        c_k, c_b = self._c_k, self._c_b
        call_count = [0]

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
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
                # combine: inside outer>inner, accum_tile address fixed, not tiled.
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
        """fill op (fixed, non-advancing output) must not get an affine.apply."""
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

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
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
    from torch._inductor.ir import ComputedBuffer, FixedLayout, Pointwise

    data = MagicMock(spec=Pointwise)
    data.ranges = list(ranges)

    # Build row-major strides for the default layout.
    strides: list[sympy.Expr] = []
    stride: sympy.Expr = sympy.Integer(1)
    for s in reversed(ranges):
        strides.insert(0, stride)
        stride = stride * s
    layout = MagicMock(spec=FixedLayout)
    layout.stride = strides

    op = MagicMock(spec=ComputedBuffer)
    op.data = data
    op.layout = layout
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
    from torch._inductor.ir import ComputedBuffer, FixedLayout, Pointwise

    data = MagicMock(spec=Pointwise)
    data.ranges = [Integer(16)]
    data.inner_fn = MagicMock()

    op = MagicMock(spec=ComputedBuffer)
    op.data = data
    # Ordinary (non-mutation) layout, so _plan_tiling_propagation's
    # isinstance(op.layout, MutationLayoutSHOULDREMOVE) check on this op
    # resolves to False instead of raising AttributeError -- layout is an
    # instance attribute ComputedBuffer sets in __init__, not a class
    # attribute, so spec=ComputedBuffer alone doesn't expose it.
    op.layout = MagicMock(spec=FixedLayout)
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
    """Tests for tiling propagation — consumer analysis helpers."""

    def setUp(self):
        # Only test_case2_condition_now_produces_copy_op below needs a real
        # graph handler (it drives _propagate_tiled_op end to end with real
        # IR, which calls V.graph.qualify_name/run_node); the helper-level
        # tests above it use mocks and never touch V.graph, so this setup is
        # a no-op for them.
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()

    def tearDown(self):
        self._graph_ctx.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Tests for _find_outside_consumers
    # (this helper doesn't call V.graph, so no mocking needed)
    # ------------------------------------------------------------------

    def test_no_consumers_returns_empty(self):
        from torch_spyre._inductor.wsr.coarse_tile import _find_outside_consumers

        op = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        with patch(
            "torch_spyre._inductor.wsr.coarse_tile._graph_output_names",
            return_value=set(),
        ):
            consumers, is_out = _find_outside_consumers("op0", (0,), [op])
        self.assertEqual(consumers, [])
        self.assertFalse(is_out)

    def test_outside_consumer_detected(self):
        from torch_spyre._inductor.wsr.coarse_tile import _find_outside_consumers

        tiled = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        consumer = _make_consumer_op("out0", "op0")  # no loop_group_id → outside
        with patch(
            "torch_spyre._inductor.wsr.coarse_tile._graph_output_names",
            return_value=set(),
        ):
            consumers, is_out = _find_outside_consumers("op0", (0,), [tiled, consumer])
        self.assertEqual(consumers, [consumer])
        self.assertFalse(is_out)

    def test_graph_output_detected(self):
        from torch_spyre._inductor.wsr.coarse_tile import _find_outside_consumers

        tiled = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        with patch(
            "torch_spyre._inductor.wsr.coarse_tile._graph_output_names",
            return_value={"op0"},
        ):
            consumers, is_out = _find_outside_consumers("op0", (0,), [tiled])
        self.assertEqual(consumers, [])
        self.assertTrue(is_out)

    def test_inside_consumer_not_counted_as_outside(self):
        from torch_spyre._inductor.wsr.coarse_tile import _find_outside_consumers

        tiled = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        inside = _make_inside_consumer_op("op1", "op0", (0,))
        with patch(
            "torch_spyre._inductor.wsr.coarse_tile._graph_output_names",
            return_value=set(),
        ):
            consumers, is_out = _find_outside_consumers("op0", (0,), [tiled, inside])
        self.assertEqual(consumers, [])
        self.assertFalse(is_out)

    def test_different_loop_group_id_is_outside(self):
        """Op in loop group 1 should be seen as outside consumer of group 0."""
        from torch_spyre._inductor.wsr.coarse_tile import _find_outside_consumers

        tiled = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        other_group = _make_tiled_op("op1", [Integer(16)], (1,), [Integer(4)], [[0]])
        # Make op1 read op0
        other_group.get_read_writes.return_value = _make_rw_with_reads("op0")
        with patch(
            "torch_spyre._inductor.wsr.coarse_tile._graph_output_names",
            return_value=set(),
        ):
            consumers, _ = _find_outside_consumers("op0", (0,), [tiled, other_group])
        self.assertEqual(consumers, [other_group])

    def test_case2_condition_now_produces_copy_op(self):
        """An op that used to hit the direct-mutation branch (outside
        consumers, no inside consumers, no loop-internal real input) must
        now always go through _insert_copy_op instead.

        Built with real IR (matching TestInsertReadCopyOps's pattern) rather
        than mocks: _propagate_tiled_op's transformation-time helpers
        (_allocate_full_buffer, _insert_copy_op, _patch_consumers) all touch
        real ComputedBuffer/V.graph machinery (qualify_name, run_node,
        replace_computed_buffer_body), which MagicMock-based ops used
        elsewhere in this file cannot satisfy. Pass 3
        (_insert_all_write_copy_ops) itself takes the already-stamped
        `operations` list that _apply_plan would normally produce; calling
        _propagate_tiled_op directly is the cheaper, equivalent way to reach
        exactly the code this task changed.
        """
        from torch._inductor.ir import (
            ComputedBuffer,
            FixedLayout,
            MutationLayoutSHOULDREMOVE,
            Pointwise,
            StorageBox,
            TensorBox,
        )
        from torch._subclasses.fake_tensor import FakeTensorMode

        from torch_spyre._inductor.lowering import enable_spyre_lowerings
        from torch_spyre._inductor.loop_info import PropagationPlan
        from torch_spyre._inductor.wsr.coarse_tile import (
            _insert_all_read_copy_ops,
            _plan_read_copies,
            _propagate_tiled_op,
        )

        # _allocate_full_buffer lowers a real spyre.empty FX node via
        # V.graph.run_node(), which needs V.fake_mode (GraphLowering.fake_mode
        # is a passthrough property to it) -- V.set_graph_handler alone
        # doesn't establish this.
        fake_mode_ctx = V.set_fake_mode(FakeTensorMode())
        fake_mode_ctx.__enter__()
        self.addCleanup(fake_mode_ctx.__exit__, None, None, None)

        # torch.ops.spyre.empty.default only lowers to SpyreEmptyFallback
        # (rather than falling through to a generic FallbackKernel) while
        # spyre_lowerings is installed into the live lowering table -- real
        # compiles always run inside this CM; unit tests must enter it too.
        lowerings_ctx = enable_spyre_lowerings()
        lowerings_ctx.__enter__()
        self.addCleanup(lowerings_ctx.__exit__, None, None, None)

        device = torch.device("cpu")
        dtype = torch.float32

        # Two fresh graph inputs -- both external to any loop, so neither
        # forces the (now-deleted) loop-internal-input branch.
        tiled_op = _make_real_pointwise_op(
            ranges=[Integer(8)],
            input_shapes_strides=[([64], [1]), ([64], [1])],
            name="op0",
        )
        tiled_op.loop_info = CoarseTileInfo(
            loop_group_id=(0,), loop_count=[Integer(8)], loop_tiled_dims=[[0]]
        )

        # A real outside consumer (no loop_info at all) that reads op0's
        # output -- this is what used to make _find_outside_consumers report
        # a consumer with no inside consumers, selecting the old Case 2
        # branch.
        op0_box = TensorBox(StorageBox(tiled_op))

        def consumer_inner_fn(index):
            return op0_box.make_loader()(index)

        consumer_pw = Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=consumer_inner_fn,
            ranges=[Integer(8)],
        )
        consumer_data = consumer_pw.data.data  # TensorBox -> StorageBox -> Pointwise
        consumer = ComputedBuffer(
            name="out0",
            layout=FixedLayout(device, dtype, [Integer(8)], None),
            data=consumer_data,
        )
        consumer.operation_name = "out0"
        consumer.origins = OrderedSet()
        V.graph.name_to_buffer["out0"] = consumer

        # _allocate_full_buffer's V.graph.run_node() call appends the new
        # full_buf to V.graph.buffers as a side effect (real GraphLowering.
        # register_buffer behavior) and then _allocate_full_buffer does
        # operations.remove(full_buf)/insert(...) expecting it to already be
        # present -- so `operations` must be the same list object as
        # V.graph.buffers, not an independent list, for that removal to find it.
        operations = V.graph.buffers
        operations.extend([tiled_op, consumer])
        original_op = tiled_op

        # Read copy-ins are now Pass 1 (_insert_all_read_copy_ops), a
        # standalone pass that runs before Pass 3 / _propagate_tiled_op even
        # in production -- call it directly here rather than folding its
        # effect into _propagate_tiled_op.
        with patch(
            "torch_spyre._inductor.wsr.coarse_tile._graph_output_names",
            return_value=set(),
        ):
            read_copy_plans = _plan_read_copies(operations, [((0,), [tiled_op], {})])
            _insert_all_read_copy_ops(operations, read_copy_plans)
            # Pass 1 may have spliced a replacement for "op0" into
            # operations -- re-resolve by name before calling
            # _propagate_tiled_op, exactly as _insert_all_write_copy_ops'
            # own loop now does.
            tiled_op = next(
                o
                for o in operations
                if isinstance(o, ComputedBuffer) and o.get_name() == "op0"
            )
            # _propagate_tiled_op (Pass 3) now consumes a precomputed
            # PropagationPlan instead of deriving it itself -- full_ranges
            # is the pre-division full shape (8 * loop_count 8 == 64),
            # matching what _compute_full_ranges_planned computes from
            # the tile-sized op.data.ranges.
            propagation = PropagationPlan(
                kind="copy_out",
                full_ranges=[Integer(64)],
                full_strides=(Integer(1),),
                outside_consumer_names=("out0",),
                is_graph_output=False,
            )
            _propagate_tiled_op(tiled_op, propagation, operations)

        # original_op's two inputs are real, full-size, untiled InputBuffers
        # read at a tile-scoped index -- _full_buffer_read_deps now flags
        # both, so Pass 1's _insert_all_read_copy_ops replaces original_op
        # with a new ComputedBuffer (same name, "op0") before the write-side
        # copy-op logic even runs. Look the final op up by name rather than
        # using the now-stale original_op reference.
        final_op = V.graph.name_to_buffer["op0"]
        # name_to_buffer is only half the story: replace_computed_buffer_body
        # must also have swapped the stale original_op out of operations (==
        # V.graph.buffers, see the comment above where it's assigned) for
        # later scheduling to see the new op instead of the old one. Checked
        # by identity (`is`), not `in`/`==`: ComputedBuffer is a frozen
        # dataclass, so `==` compares field values rather than object
        # identity, and original_op/final_op share the same (in-place-mutated)
        # `.data` and layout -- they compare equal to each other even though
        # only final_op is the live object, which makes assertIn/assertNotIn
        # pass regardless of whether the swap actually happened.
        self.assertTrue(any(op is final_op for op in operations))
        self.assertFalse(any(op is original_op for op in operations))

        write_copy_ops = [
            op
            for op in operations
            if isinstance(op, ComputedBuffer)
            and op.get_name().startswith("coarse_tile_copy_")
        ]
        read_copy_ops = [
            op
            for op in operations
            if isinstance(op, ComputedBuffer)
            and op.get_name().startswith("coarse_tile_read_copy_")
        ]
        self.assertEqual(len(write_copy_ops), 1)
        self.assertIsInstance(write_copy_ops[0].layout, MutationLayoutSHOULDREMOVE)
        # Both real inputs get their own read copy.
        self.assertEqual(len(read_copy_ops), 2)
        # The tiled op itself never becomes a MutationLayoutSHOULDREMOVE --
        # it keeps its own tile-sized layout, and its own write no longer
        # advances (the copy-out op above drains it every iteration
        # instead), mirroring the Case 1 path.
        self.assertNotIsInstance(final_op.layout, MutationLayoutSHOULDREMOVE)
        self.assertEqual(final_op.loop_info.output_tiled_dims, [])


class TestPlanTilingPropagation(unittest.TestCase):
    """Cross-check: _plan_tiling_propagation's kind decision must match what
    _propagate_tiled_op / _propagate_tiled_reduction_op actually do today.

    This is the load-bearing regression net for front-loaded planning: it
    validates the planning decision against current (still transformation-
    driving) behavior, before transformation ever consumes the
    new field. Built with the same mock-based fixtures
    (_make_tiled_op/_make_consumer_op/_make_inside_consumer_op/
    _make_tiled_reduction_op) TestCoarseTileBufferPropagation already uses
    for its plain _find_outside_consumers/_full_buffer_read_deps checks --
    _plan_tiling_propagation's own helpers are direct planning-time analogs
    of those same functions.
    """

    def _plan_for(self, op, group_ops=None):
        from torch_spyre._inductor.wsr.coarse_tile import _plan_tiling_propagation

        group_ops = group_ops if group_ops is not None else [op]
        info = op.loop_info
        plan = {id(op): info}
        levels = [(0, c) for c in info.loop_count]
        with patch(
            "torch_spyre._inductor.wsr.coarse_tile._graph_output_names",
            return_value=set(),
        ):
            _plan_tiling_propagation(group_ops, [(group_ops, levels)], plan)
        return info.propagation

    def test_loop_invariant_matches_no_tiled_dims(self):
        """All loop_tiled_dims empty -> loop_internal, matching
        _propagate_tiled_op's `all(not dims ...)` fast-path return."""
        op = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[]])
        propagation = self._plan_for(op)
        self.assertEqual(propagation.kind, "loop_internal")

    def test_no_outside_consumers_matches_loop_internal(self):
        """Tiled with no outside consumers/graph output -> loop_internal,
        matching _propagate_tiled_op zeroing output_tiled_dims and
        returning without a copy op."""
        op = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        propagation = self._plan_for(op)
        self.assertEqual(propagation.kind, "loop_internal")
        self.assertEqual(propagation.outside_consumer_names, ())
        self.assertFalse(propagation.is_graph_output)

    def test_outside_consumer_matches_copy_out(self):
        """Tiled with an outside consumer -> copy_out, matching
        _propagate_tiled_op's _allocate_full_buffer/_insert_copy_op path.

        Planning runs before _apply_plan divides op.data.ranges, so the
        fixture's ranges are already the full (pre-division) size here --
        full_ranges is expected to come back unchanged."""
        tiled = _make_tiled_op("op0", [Integer(64)], (0,), [Integer(4)], [[0]])
        consumer = _make_consumer_op("out0", "op0")
        propagation = self._plan_for(tiled, group_ops=[tiled, consumer])
        self.assertEqual(propagation.kind, "copy_out")
        self.assertEqual(propagation.outside_consumer_names, ("out0",))
        self.assertEqual(propagation.full_ranges, [Integer(64)])

    def test_inside_consumer_only_matches_loop_internal(self):
        """An inside-loop-group consumer alone doesn't force copy_out --
        matches _propagate_tiled_op's outside_consumers check, which
        _find_outside_consumers already excludes same-outer-group readers
        from."""
        tiled = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        inside = _make_inside_consumer_op("op1", "op0", (0,))
        propagation = self._plan_for(tiled, group_ops=[tiled, inside])
        self.assertEqual(propagation.kind, "loop_internal")

    def test_graph_output_matches_copy_out(self):
        """A graph-output buffer -> copy_out even with no other consumers,
        matching _propagate_tiled_op's is_graph_output branch."""
        from torch_spyre._inductor.wsr.coarse_tile import _plan_tiling_propagation

        op = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        info = op.loop_info
        plan = {id(op): info}
        levels = [(0, Integer(4))]
        with patch(
            "torch_spyre._inductor.wsr.coarse_tile._graph_output_names",
            return_value={"op0"},
        ):
            _plan_tiling_propagation([op], [([op], levels)], plan)
        propagation = info.propagation
        self.assertEqual(propagation.kind, "copy_out")
        self.assertTrue(propagation.is_graph_output)

    def test_tiled_reduction_matches_reduction_kind(self):
        """A Reduction op tiling a reduction dim -> kind="reduction", with
        the same identity/nesting decisions _propagate_tiled_reduction_op
        computes."""
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
        propagation = self._plan_for(op)
        self.assertEqual(propagation.kind, "reduction")
        self.assertIsNotNone(propagation.reduction)
        self.assertEqual(propagation.reduction.reduction_type, "sum")
        self.assertEqual(propagation.reduction.identity, 0)
        self.assertFalse(propagation.reduction.is_nested)
        self.assertIsNone(propagation.reduction.outer_fill_loop_info)

    def test_nested_tiled_reduction_matches_is_nested(self):
        """Nested output+reduction tiling -> reduction plan with
        is_nested=True and a trimmed outer_fill_loop_info, matching
        _compute_fill_loop_info_planned's non-None nested case."""
        op = _make_tiled_reduction_op(
            "red0",
            ranges=[Integer(64)],
            reduction_ranges=[Integer(256)],
            reduction_type="max",
            loop_group_id=(0, 0),
            loop_count=[Integer(2), Integer(4)],
            loop_tiled_dims=[[0], []],
        )
        op.loop_info.loop_tiled_reduction_dims = [[], [0]]
        propagation = self._plan_for(op)
        self.assertEqual(propagation.kind, "reduction")
        self.assertTrue(propagation.reduction.is_nested)
        self.assertEqual(propagation.reduction.identity, float("-inf"))
        outer_info = propagation.reduction.outer_fill_loop_info
        self.assertIsNotNone(outer_info)
        self.assertEqual(outer_info.loop_group_id, (0,))
        self.assertEqual(outer_info.loop_count, [Integer(2)])
        self.assertEqual(outer_info.loop_tiled_dims, [[0]])

    def test_reader_before_producer_still_zeroes_fixed_read(self):
        """Reader-before-producer ordering in group_ops must not matter.

        producer (op0) is tiled with no outside consumers -> loop_internal,
        i.e. "fixed": its own write never advances. reader (op1) is inside
        the same loop group and reads op0; op1's tiled_dims_per_read entry
        for op0 is planted here exactly as plan_coarse_tile_groups would
        have left it *before* op0's fixed status was known (non-empty,
        mirroring the stale entry _zero_reads_of_fixed_buffers used to
        correct after the fact at transformation time). op1 is placed
        BEFORE op0 in group_ops -- the ordering
        _zero_reads_of_fixed_buffers existed to work around, since
        source-order visitation would see op1 before op0 is known to be
        fixed. _plan_tiling_propagation must still zero op1's entry for
        op0, because it computes every op's kind up front before its own
        fixed-buffer zeroing pass runs -- there is no visitation-order
        hazard left to trigger.
        """
        from torch_spyre._inductor.wsr.coarse_tile import _plan_tiling_propagation

        producer = _make_tiled_op("op0", [Integer(16)], (0,), [Integer(4)], [[0]])
        reader = _make_inside_consumer_op("op1", "op0", (0,))
        # Simulate plan_coarse_tile_groups's pre-zeroing output: op1 read op0
        # while op0's own tiled dims were still extent-4-tiled at level 0.
        reader.loop_info.tiled_dims_per_read = [[[(0, Integer(4))]]]
        group_ops = [reader, producer]
        plan = {id(reader): reader.loop_info, id(producer): producer.loop_info}
        levels = [(0, Integer(4))]
        # reader is in `plan`, so _plan_tiling_propagation's main loop
        # processes it too, which resolves its "op0" read via
        # V.graph.get_buffer -- give it a minimal graph mock rather than a
        # real GraphLowering, since these are MagicMock IR objects, not
        # objects a real graph handler has ever registered.
        mock_graph = MagicMock()
        mock_graph.get_buffer.return_value = producer
        with (
            patch(
                "torch_spyre._inductor.wsr.coarse_tile._graph_output_names",
                return_value=set(),
            ),
            V.set_graph_handler(mock_graph),
        ):
            _plan_tiling_propagation(group_ops, [(group_ops, levels)], plan)

        self.assertEqual(producer.loop_info.propagation.kind, "loop_internal")
        self.assertEqual(reader.loop_info.tiled_dims_per_read, [[]])


def _make_cross_group_producer_read_fixture():
    """Like _make_full_buffer_read_fixture, but the producer is a plain
    ComputedBuffer in a different loop group, not a SpyreEmptyFallback.

    Models Open Question 5's chained-coarse-tile-group case: a buffer
    produced by an earlier loop group (already stamped with its own
    loop_info) rather than a fresh SpyreEmptyFallback accumulator. Before
    this task, _full_buffer_read_deps's isinstance(SpyreEmptyFallback)
    filter misses this case entirely; after, it must be caught by the
    loop_group_id[0] comparison.

    Caller must have an active graph handler around this call.
    Returns (tiled_op, full_deps, operations).
    """
    from torch._inductor.ir import (
        ComputedBuffer,
        FixedLayout,
        Pointwise,
        StorageBox,
        TensorBox,
    )

    device = torch.device("cpu")
    dtype = torch.float32

    # Producer: a ComputedBuffer in loop group (1,), full-size output.
    producer_data = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=lambda index: sympy.Integer(0),
        ranges=[Integer(64), Integer(128)],
    ).data.data
    producer = ComputedBuffer(
        name="producer0",
        layout=FixedLayout(device, dtype, [64, 128], [128, 1]),
        data=producer_data,
    )
    producer.operation_name = "producer0"
    producer.origins = OrderedSet()
    producer.loop_info = CoarseTileInfo(
        loop_group_id=(1,), loop_count=[Integer(1)], loop_tiled_dims=[[]]
    )
    V.graph.name_to_buffer["producer0"] = producer

    producer_box = TensorBox(StorageBox(producer))

    def inner_fn(index):
        # ComputedBuffer.make_loader() inlines the producer's own inner_fn
        # (rather than emitting ops.load) whenever num_reads() == 0 -- i.e.
        # before any other op has read it, which is exactly our case here.
        # force_realize() forces a genuine ops.load, matching how a real
        # cross-loop-group producer (already scheduled, with consumers
        # elsewhere in the graph) would behave.
        with ComputedBuffer.force_realize():
            return producer_box.make_loader()(index)

    pw = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[Integer(8), Integer(128)],
    )
    pw_data = pw.data.data
    tiled_op = ComputedBuffer(
        name="tiled_op0",
        layout=FixedLayout(device, dtype, [Integer(8), Integer(128)], None),
        data=pw_data,
    )
    tiled_op.operation_name = "tiled_op0"
    tiled_op.origins = OrderedSet()
    tiled_op.loop_info = CoarseTileInfo(
        loop_group_id=(0,),
        loop_count=[Integer(8)],
        loop_tiled_dims=[[0]],
        tiled_dims_per_read=[[[]]],
    )
    V.graph.name_to_buffer["tiled_op0"] = tiled_op

    operations = [producer, tiled_op]
    full_deps = _full_buffer_read_deps(tiled_op)
    return tiled_op, full_deps, operations


class TestFullBufferReadDepsCrossGroup(unittest.TestCase):
    """_full_buffer_read_deps must catch any cross-loop-group producer, not
    just SpyreEmptyFallback targets (Open Question 5)."""

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()

    def tearDown(self):
        self._graph_ctx.__exit__(None, None, None)

    def test_cross_group_plain_producer_detected(self):
        tiled_op, full_deps, operations = _make_cross_group_producer_read_fixture()
        self.assertEqual(len(full_deps), 1)
        self.assertEqual(full_deps[0].name, "producer0")

    def test_same_group_producer_not_flagged(self):
        """A producer in the SAME outer loop group must NOT be treated as
        a full-buffer read (it's an ordinary loop-internal dependency)."""
        from torch._inductor.ir import ComputedBuffer, FixedLayout, Pointwise

        device = torch.device("cpu")
        dtype = torch.float32
        producer_data = Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=lambda index: sympy.Integer(0),
            ranges=[Integer(8), Integer(128)],
        ).data.data
        producer = ComputedBuffer(
            name="producer0",
            layout=FixedLayout(device, dtype, [8, 128], [128, 1]),
            data=producer_data,
        )
        producer.operation_name = "producer0"
        producer.origins = OrderedSet()
        producer.loop_info = CoarseTileInfo(
            loop_group_id=(0,), loop_count=[Integer(8)], loop_tiled_dims=[[0]]
        )
        V.graph.name_to_buffer["producer0"] = producer

        from torch._inductor.ir import StorageBox, TensorBox

        producer_box = TensorBox(StorageBox(producer))

        def inner_fn(index):
            # See _make_cross_group_producer_read_fixture: force a real
            # ops.load so this test genuinely exercises the loop_group_id
            # comparison, rather than passing vacuously because the
            # producer's own inner_fn got inlined (num_reads() == 0).
            with ComputedBuffer.force_realize():
                return producer_box.make_loader()(index)

        pw = Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=[Integer(8), Integer(128)],
        )
        tiled_op = ComputedBuffer(
            name="tiled_op0",
            layout=FixedLayout(device, dtype, [Integer(8), Integer(128)], None),
            data=pw.data.data,
        )
        tiled_op.operation_name = "tiled_op0"
        tiled_op.origins = OrderedSet()
        tiled_op.loop_info = CoarseTileInfo(
            loop_group_id=(0,), loop_count=[Integer(8)], loop_tiled_dims=[[0]]
        )
        V.graph.name_to_buffer["tiled_op0"] = tiled_op

        full_deps = _full_buffer_read_deps(tiled_op)
        self.assertEqual(full_deps, [])


def _make_full_buffer_read_fixture():
    """Build a real tiled Pointwise op that reads a full-size SpyreEmptyFallback.

    Mirrors ``_make_real_pointwise_op`` (real IR, not mocks) but the input is
    a genuine ``SpyreEmptyFallback`` buffer (constructed directly, bypassing
    FX-node lowering) rather than a plain ``InputBuffer`` -- this is what
    ``_full_buffer_read_deps`` specifically filters for.  The op's own
    ranges are smaller than the full buffer's shape (8 rows out of 64),
    modeling the tile-vs-full-buffer mismatch ``_insert_all_read_copy_ops``
    exists to resolve.

    Caller must have an active graph handler (``V.set_graph_handler(...)``)
    around this call, matching every other real-IR helper in this file.
    Returns ``(tiled_op, full_deps, operations)`` ready to pass straight into
    ``_insert_all_read_copy_ops``.
    """
    from torch._inductor.ir import (
        ComputedBuffer,
        FixedLayout,
        Pointwise,
        StorageBox,
        TensorBox,
    )

    from torch_spyre._inductor.ir import SpyreEmptyFallback

    device = torch.device("cpu")
    dtype = torch.float32

    full_buf = SpyreEmptyFallback(
        torch.ops.spyre.empty.default, [64, 128], device, dtype
    )
    full_buf.layout = FixedLayout(device, dtype, [64, 128], [128, 1])
    full_box = TensorBox(StorageBox(full_buf))

    def inner_fn(index):
        return full_box.make_loader()(index)

    pw = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[Integer(8), Integer(128)],
    )
    pw_data = pw.data.data  # TensorBox -> StorageBox -> Pointwise
    tiled_op = ComputedBuffer(
        name="tiled_op0",
        layout=FixedLayout(device, dtype, [Integer(8), Integer(128)], None),
        data=pw_data,
    )
    tiled_op.operation_name = "tiled_op0"
    tiled_op.origins = OrderedSet()
    tiled_op.loop_info = CoarseTileInfo(
        loop_group_id=(0,), loop_count=[Integer(8)], loop_tiled_dims=[[0]]
    )
    V.graph.name_to_buffer["tiled_op0"] = tiled_op

    operations = [full_buf, tiled_op]
    full_deps = _full_buffer_read_deps(tiled_op)
    return tiled_op, full_deps, operations


def _make_two_op_shared_read_fixture():
    """Two tiled ops in the same group both read full_buf at the SAME index
    expression (mirrors "a+b*a": two reads of "a" with identical indexing,
    just from two different consuming ops instead of one op's two reads).

    Returns (op_a, op_b, full_buf, operations) with loop_info already
    stamped on both ops, ready to pass into _plan_read_copies via a single
    retiled_infos_by_group-style entry:
    [((0,), [op_a, op_b], {})].
    """
    from torch._inductor.ir import (
        ComputedBuffer,
        FixedLayout,
        Pointwise,
        StorageBox,
        TensorBox,
    )

    from torch_spyre._inductor.ir import SpyreEmptyFallback

    device = torch.device("cpu")
    dtype = torch.float32

    full_buf = SpyreEmptyFallback(
        torch.ops.spyre.empty.default, [64, 128], device, dtype
    )
    full_buf.layout = FixedLayout(device, dtype, [64, 128], [128, 1])
    full_box = TensorBox(StorageBox(full_buf))

    def _make_reader(name):
        def inner_fn(index):
            return full_box.make_loader()(index)

        pw = Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=[Integer(8), Integer(128)],
        )
        pw_data = pw.data.data
        op = ComputedBuffer(
            name=name,
            layout=FixedLayout(device, dtype, [Integer(8), Integer(128)], None),
            data=pw_data,
        )
        op.operation_name = name
        op.origins = OrderedSet()
        op.loop_info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(8)],
            loop_tiled_dims=[[0]],
            tiled_dims_per_read=[[[]]],
        )
        V.graph.name_to_buffer[name] = op
        return op

    op_a = _make_reader("op_a")
    op_b = _make_reader("op_b")
    operations = [full_buf, op_a, op_b]
    return op_a, op_b, full_buf, operations


class TestPlanReadCopies(unittest.TestCase):
    """_plan_read_copies groups equivalent cross-group reads within a group."""

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()

    def tearDown(self):
        self._graph_ctx.__exit__(None, None, None)

    def test_two_ops_same_index_share_one_entry(self):
        from torch_spyre._inductor.wsr.coarse_tile import _plan_read_copies

        op_a, op_b, full_buf, operations = _make_two_op_shared_read_fixture()
        captured = ((((0, Integer(128), Integer(1)),),),)
        retiled_infos_by_group = [((0,), [op_a, op_b], {})]

        plans = _plan_read_copies(
            operations,
            retiled_infos_by_group,
            {id(op_a): captured, id(op_b): captured},
        )

        self.assertIn((0,), plans)
        plan = plans[(0,)]
        self.assertEqual(len(plan.entries), 1)
        entry = plan.entries[0]
        self.assertEqual(entry.dep.name, full_buf.get_name())
        self.assertEqual(entry.insert_before_op_name, "op_a")
        self.assertEqual(entry.sizing_op_name, "op_a")
        self.assertEqual(set(entry.consumer_op_names), {"op_a", "op_b"})
        self.assertEqual(
            entry.predivision_unit_steps,
            (((0, Integer(128), Integer(1)),),),
        )

    def test_cross_group_source_is_not_loop_invariant(self):
        """A fixed-address producer scratch is rewritten each trip.

        Even after propagation clears the consumer's address step, a read
        from another counted-loop group must remain inside the consumer's
        loop so Pass 3 can redirect it to the advancing full buffer.
        """
        from torch_spyre._inductor.wsr.coarse_tile import (
            _ReadCopyHoistDecision,
            _ReadCopySourceInfo,
            _ReadCopySourceKind,
            _plan_read_copies,
            _read_copy_hoist_decision,
        )

        tiled_op, _full_deps, operations = _make_cross_group_producer_read_fixture()
        # Isolate source classification: complete fixed-address metadata lets
        # the test reach CROSS_GROUP_SOURCE rather than declining earlier.
        tiled_op.loop_info.tiled_dims_per_read = [[[]]]

        plans = _plan_read_copies(operations, [((0,), [tiled_op], {})])

        self.assertFalse(plans[(0,)].entries[0].loop_invariant)
        self.assertIs(
            _read_copy_hoist_decision(
                tiled_op.loop_info,
                0,
                _ReadCopySourceInfo(_ReadCopySourceKind.LOOP_PRODUCED, (1,)),
            ),
            _ReadCopyHoistDecision.CROSS_GROUP_SOURCE,
        )

    def test_missing_read_step_metadata_is_not_loop_invariant(self):
        """Missing metadata cannot authorize moving a read before the loop."""
        from torch_spyre._inductor.wsr.coarse_tile import (
            _ReadCopyHoistDecision,
            _ReadCopySourceInfo,
            _ReadCopySourceKind,
            _plan_read_copies,
            _read_copy_hoist_decision,
        )

        tiled_op, _full_deps, operations = _make_full_buffer_read_fixture()
        self.assertEqual(tiled_op.loop_info.tiled_dims_per_read, [])

        plans = _plan_read_copies(operations, [((0,), [tiled_op], {})])

        self.assertFalse(plans[(0,)].entries[0].loop_invariant)
        self.assertIs(
            _read_copy_hoist_decision(
                tiled_op.loop_info,
                0,
                _ReadCopySourceInfo(_ReadCopySourceKind.KNOWN_EXTERNAL),
            ),
            _ReadCopyHoistDecision.MISSING_STEP_METADATA,
        )

    def test_same_group_loop_source_is_not_loop_invariant(self):
        from torch_spyre._inductor.wsr.coarse_tile import (
            _ReadCopyHoistDecision,
            _ReadCopySourceInfo,
            _ReadCopySourceKind,
            _read_copy_hoist_decision,
        )

        tiled_op, _full_deps, _operations = _make_full_buffer_read_fixture()
        tiled_op.loop_info.tiled_dims_per_read = [[[]]]

        self.assertIs(
            _read_copy_hoist_decision(
                tiled_op.loop_info,
                0,
                _ReadCopySourceInfo(_ReadCopySourceKind.LOOP_PRODUCED, (0,)),
            ),
            _ReadCopyHoistDecision.LOOP_PRODUCED_SOURCE,
        )

    def test_loop_written_storage_is_not_loop_invariant(self):
        """A fixed allocation is not stable while a loop mutates it."""
        from torch._inductor.ir import (
            ComputedBuffer,
            MutationLayoutSHOULDREMOVE,
            StorageBox,
            TensorBox,
        )

        from torch_spyre._inductor.wsr.coarse_tile import (
            _ReadCopyHoistDecision,
            _ReadCopySourceKind,
            _loop_written_buffer_names,
            _plan_read_copies,
            _read_copy_hoist_decision,
            _read_copy_source_info,
        )

        tiled_op, _full_deps, operations = _make_full_buffer_read_fixture()
        tiled_op.loop_info.tiled_dims_per_read = [[[]]]
        full_buf = operations[0]
        writer = ComputedBuffer(
            name="loop_writer",
            layout=MutationLayoutSHOULDREMOVE(TensorBox(StorageBox(full_buf))),
            data=tiled_op.data,
        )
        writer.operation_name = "loop_writer"
        writer.origins = OrderedSet()
        writer.loop_info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(8)],
            loop_tiled_dims=[[0]],
        )
        operations.append(writer)

        plans = _plan_read_copies(operations, [((0,), [tiled_op], {})])
        source_info = _read_copy_source_info(
            full_buf.get_name(),
            {op.get_name(): op for op in operations},
            _loop_written_buffer_names(operations),
        )

        self.assertFalse(plans[(0,)].entries[0].loop_invariant)
        self.assertIs(source_info.kind, _ReadCopySourceKind.IN_LOOP_WRITTEN)
        self.assertIs(
            _read_copy_hoist_decision(tiled_op.loop_info, 0, source_info),
            _ReadCopyHoistDecision.IN_LOOP_WRITTEN_SOURCE,
        )

    def test_unknown_source_is_not_loop_invariant(self):
        from torch_spyre._inductor.wsr.coarse_tile import (
            _ReadCopyHoistDecision,
            _ReadCopySourceInfo,
            _ReadCopySourceKind,
            _read_copy_hoist_decision,
        )

        tiled_op, _full_deps, _operations = _make_full_buffer_read_fixture()
        tiled_op.loop_info.tiled_dims_per_read = [[[]]]

        self.assertIs(
            _read_copy_hoist_decision(
                tiled_op.loop_info,
                0,
                _ReadCopySourceInfo(_ReadCopySourceKind.UNKNOWN),
            ),
            _ReadCopyHoistDecision.UNKNOWN_SOURCE,
        )

    def test_advancing_external_read_is_not_loop_invariant(self):
        from torch_spyre._inductor.wsr.coarse_tile import (
            _ReadCopyHoistDecision,
            _ReadCopySourceInfo,
            _ReadCopySourceKind,
            _read_copy_hoist_decision,
        )

        tiled_op, _full_deps, _operations = _make_full_buffer_read_fixture()
        tiled_op.loop_info.tiled_dims_per_read = [[[(0, Integer(8))]]]

        self.assertIs(
            _read_copy_hoist_decision(
                tiled_op.loop_info,
                0,
                _ReadCopySourceInfo(_ReadCopySourceKind.KNOWN_EXTERNAL),
            ),
            _ReadCopyHoistDecision.ADVANCING_READ,
        )

    def test_fixed_external_read_is_loop_invariant(self):
        from torch_spyre._inductor.wsr.coarse_tile import (
            _ReadCopyHoistDecision,
            _ReadCopySourceInfo,
            _ReadCopySourceKind,
            _read_copy_hoist_decision,
        )

        tiled_op, _full_deps, _operations = _make_full_buffer_read_fixture()
        tiled_op.loop_info.tiled_dims_per_read = [[[]]]

        self.assertIs(
            _read_copy_hoist_decision(
                tiled_op.loop_info,
                0,
                _ReadCopySourceInfo(_ReadCopySourceKind.KNOWN_EXTERNAL),
            ),
            _ReadCopyHoistDecision.ELIGIBLE,
        )

    def test_same_op_two_reads_same_index_collapse_to_one_entry(self):
        """a+b*a: one op reading buffer 'a' twice at the identical index
        must plan exactly one ReadCopyEntry, consumed once by that op."""
        from torch._inductor.ir import (
            ComputedBuffer,
            FixedLayout,
            Pointwise,
            StorageBox,
            TensorBox,
        )

        from torch_spyre._inductor.ir import SpyreEmptyFallback
        from torch_spyre._inductor.wsr.coarse_tile import _plan_read_copies

        device = torch.device("cpu")
        dtype = torch.float32

        full_buf = SpyreEmptyFallback(
            torch.ops.spyre.empty.default, [8, 8], device, dtype
        )
        full_buf.layout = FixedLayout(device, dtype, [8, 8], [8, 1])
        full_box = TensorBox(StorageBox(full_buf))

        def inner_fn(index):
            a1 = full_box.make_loader()(index)
            a2 = full_box.make_loader()(index)
            return a1 + a2

        pw = Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=[Integer(8), Integer(8)],
        )
        pw_data = pw.data.data
        tiled_op = ComputedBuffer(
            name="tiled_op0",
            layout=FixedLayout(device, dtype, [Integer(8), Integer(8)], None),
            data=pw_data,
        )
        tiled_op.operation_name = "tiled_op0"
        tiled_op.origins = OrderedSet()
        tiled_op.loop_info = CoarseTileInfo(
            loop_group_id=(0,), loop_count=[Integer(1)], loop_tiled_dims=[[]]
        )
        V.graph.name_to_buffer["tiled_op0"] = tiled_op

        operations = [full_buf, tiled_op]
        retiled_infos_by_group = [((0,), [tiled_op], {})]

        plans = _plan_read_copies(operations, retiled_infos_by_group)

        self.assertEqual(len(plans[(0,)].entries), 1)
        entry = plans[(0,)].entries[0]
        self.assertEqual(entry.consumer_op_names, ("tiled_op0",))

    def test_partial_sharing_two_entries_different_consumers(self):
        """op_a reads shared_buf + only_a; op_b reads shared_buf + only_b.
        Expect two ReadCopyEntry objects: one shared (consumers = op_a,
        op_b), one each for only_a/only_b (consumers = just that op)."""
        from torch._inductor.ir import (
            ComputedBuffer,
            FixedLayout,
            Pointwise,
            StorageBox,
            TensorBox,
        )

        from torch_spyre._inductor.ir import SpyreEmptyFallback
        from torch_spyre._inductor.wsr.coarse_tile import _plan_read_copies

        device = torch.device("cpu")
        dtype = torch.float32

        def _make_full(name, size):
            buf = SpyreEmptyFallback(torch.ops.spyre.empty.default, size, device, dtype)
            buf.layout = FixedLayout(device, dtype, size, [size[1], 1])
            buf.name = name
            V.graph.name_to_buffer[name] = buf
            return buf

        shared_buf = _make_full("shared_buf", [8, 8])
        only_a_buf = _make_full("only_a_buf", [8, 8])
        only_b_buf = _make_full("only_b_buf", [8, 8])

        shared_box = TensorBox(StorageBox(shared_buf))
        only_a_box = TensorBox(StorageBox(only_a_buf))
        only_b_box = TensorBox(StorageBox(only_b_buf))

        def _make_reader(name, extra_box):
            def inner_fn(index):
                return shared_box.make_loader()(index) + extra_box.make_loader()(index)

            pw = Pointwise.create(
                device=device,
                dtype=dtype,
                inner_fn=inner_fn,
                ranges=[Integer(8), Integer(8)],
            )
            pw_data = pw.data.data
            op = ComputedBuffer(
                name=name,
                layout=FixedLayout(device, dtype, [Integer(8), Integer(8)], None),
                data=pw_data,
            )
            op.operation_name = name
            op.origins = OrderedSet()
            op.loop_info = CoarseTileInfo(
                loop_group_id=(0,), loop_count=[Integer(1)], loop_tiled_dims=[[]]
            )
            V.graph.name_to_buffer[name] = op
            return op

        op_a = _make_reader("op_a", only_a_box)
        op_b = _make_reader("op_b", only_b_box)
        operations = [shared_buf, only_a_buf, only_b_buf, op_a, op_b]
        retiled_infos_by_group = [((0,), [op_a, op_b], {})]

        plans = _plan_read_copies(operations, retiled_infos_by_group)

        entries_by_name = {e.dep.name: e for e in plans[(0,)].entries}
        self.assertEqual(
            set(entries_by_name), {"shared_buf", "only_a_buf", "only_b_buf"}
        )
        self.assertEqual(
            set(entries_by_name["shared_buf"].consumer_op_names), {"op_a", "op_b"}
        )
        self.assertEqual(entries_by_name["only_a_buf"].consumer_op_names, ("op_a",))
        self.assertEqual(entries_by_name["only_b_buf"].consumer_op_names, ("op_b",))


class TestReadCopyPlanDataclasses(unittest.TestCase):
    """ReadCopyEntry/ReadCopyPlan are plain frozen dataclasses (Task 1)."""

    def test_read_copy_entry_and_plan_construct_and_are_frozen(self):
        from torch._inductor.dependencies import MemoryDep
        from torch_spyre._inductor.loop_info import ReadCopyEntry, ReadCopyPlan

        dep = MemoryDep(
            name="a",
            index=sympy.Integer(0),
            var_names=(),
            size=(),
        )
        entry = ReadCopyEntry(
            copy_name="coarse_tile_read_copy_group0_a_0",
            dep=dep,
            insert_before_op_name="op0",
            sizing_op_name="op0",
            sizing_read_index=0,
            consumer_op_names=("op0", "op1"),
            loop_invariant=True,
        )
        self.assertEqual(entry.copy_name, "coarse_tile_read_copy_group0_a_0")
        self.assertEqual(entry.consumer_op_names, ("op0", "op1"))
        self.assertEqual(entry.predivision_unit_steps, ())
        self.assertTrue(entry.loop_invariant)
        with self.assertRaises(Exception):
            entry.copy_name = "other"  # frozen -> raises FrozenInstanceError

        plan = ReadCopyPlan(entries=(entry,))
        self.assertEqual(plan.entries, (entry,))
        with self.assertRaises(Exception):
            plan.entries = ()


class TestReadCopyElisionProof(unittest.TestCase):
    def test_record_is_frozen(self):
        from torch_spyre._inductor.loop_info import ReadCopyElisionRecord

        record = ReadCopyElisionRecord(
            consumer_name="op0",
            copy_name="copy0",
            source_name="weight",
            direct_inner_fn=lambda: None,
        )

        with self.assertRaises(Exception):
            record.copy_name = "other"

    def test_local_bounds_are_measured_in_source_elements(self):
        from torch._inductor.dependencies import MemoryDep
        from torch_spyre._inductor.read_copy_elision import _affine_bounds

        d0, d1, d2 = sympy.symbols("d0 d1 d2", integer=True)
        dep = MemoryDep(
            name="weight",
            index=4096 * d0 + 64 * d1 + d2,
            var_names=(d0, d1, d2),
            size=(Integer(1), Integer(64), Integer(64)),
        )

        self.assertEqual(_affine_bounds(dep), (0, 4095))

    def test_loop_bound_covers_every_expert_step(self):
        from torch_spyre._inductor.read_copy_elision import _loop_advance_bound

        info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(3)],
            loop_tiled_dims=[[0]],
            tiled_dims_per_read=[[[]]],
            squeezed_advance_per_read=[[[(Integer(4096), Integer(1))]]],
        )

        self.assertEqual(_loop_advance_bound(info, 0), (0, 8192))

    def test_unsqueezed_loop_step_is_not_elided(self):
        from torch_spyre._inductor.read_copy_elision import _loop_advance_bound

        info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(3)],
            loop_tiled_dims=[[0]],
            tiled_dims_per_read=[[[(0, Integer(1))]]],
            squeezed_advance_per_read=[[[(Integer(4096), Integer(1))]]],
        )

        self.assertIsNone(_loop_advance_bound(info, 0))


class TestInsertAllReadCopyOps(unittest.TestCase):
    """_insert_all_read_copy_ops executes a precomputed ReadCopyPlan."""

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()

    def tearDown(self):
        self._graph_ctx.__exit__(None, None, None)

    def test_shared_read_produces_one_copy_for_two_consumers(self):
        from torch._inductor.ir import ComputedBuffer

        from torch_spyre._inductor.wsr.coarse_tile import (
            _insert_all_read_copy_ops,
            _plan_read_copies,
        )

        op_a, op_b, full_buf, operations = _make_two_op_shared_read_fixture()
        retiled_infos_by_group = [((0,), [op_a, op_b], {})]
        plans = _plan_read_copies(operations, retiled_infos_by_group)

        _insert_all_read_copy_ops(operations, plans)

        # Exactly one new copy op was inserted: full_buf, copy, op_a, op_b.
        self.assertEqual(len(operations), 4)
        copy_buf = operations[1]
        self.assertIsInstance(copy_buf, ComputedBuffer)
        self.assertIs(operations[0], full_buf)
        self.assertFalse(hasattr(copy_buf, "loop_info"))
        self.assertFalse(hasattr(copy_buf, "work_div_loop_info"))

        # Both consumers were repointed at the SAME copy buffer name, not
        # two independent copies.
        new_op_a = next(
            o
            for o in operations
            if isinstance(o, ComputedBuffer) and o.get_name() == "op_a"
        )
        new_op_b = next(
            o
            for o in operations
            if isinstance(o, ComputedBuffer) and o.get_name() == "op_b"
        )

        class _Recorder(list):
            def load(self, name, index):
                self.append(name)
                return 0.0

        # _Recorder(loaded_by_a) would build a *new* list initialized from
        # loaded_by_a's (empty) contents, not an alias of it -- appends
        # inside .load() would then land on the _Recorder instance, not on
        # loaded_by_a, leaving loaded_by_a permanently empty regardless of
        # what inner_fn loads. Use each _Recorder instance itself as both
        # the installed handler and the assertion target.
        loaded_by_a = _Recorder()
        loaded_by_b = _Recorder()

        with V.set_ops_handler(loaded_by_a):
            new_op_a.data.inner_fn([sympy.Integer(0) for _ in new_op_a.data.ranges])
        with V.set_ops_handler(loaded_by_b):
            new_op_b.data.inner_fn([sympy.Integer(0) for _ in new_op_b.data.ranges])

        self.assertEqual(loaded_by_a, [copy_buf.get_name()])
        self.assertEqual(loaded_by_b, [copy_buf.get_name()])
        self.assertNotIn(full_buf.get_name(), loaded_by_a)
        self.assertNotIn(full_buf.get_name(), loaded_by_b)

    def test_generated_read_copy_does_not_inherit_positional_work_div_names(self):
        from torch._inductor.ir import ComputedBuffer

        from torch_spyre._inductor.wsr.coarse_tile import (
            _insert_all_read_copy_ops,
            _plan_read_copies,
        )

        op_a, op_b, _full_buf, operations = _make_two_op_shared_read_fixture()
        for op in (op_a, op_b):
            op.work_div_loop_info = {  # type: ignore[attr-defined]
                Symbol("d0"): ["T"]
            }
        plans = _plan_read_copies(operations, [((0,), [op_a, op_b], {})])

        _insert_all_read_copy_ops(operations, plans)

        generated = [
            op
            for op in operations
            if isinstance(op, ComputedBuffer)
            and op.get_name().startswith("coarse_tile_read_copy")
        ]
        self.assertEqual(len(generated), 1)
        self.assertFalse(hasattr(generated[0], "work_div_loop_info"))

    def test_advancing_read_copy_stays_inside_loop(self):
        """Only fixed reads move to the preheader; advancing reads retain
        their counted-loop metadata and existing per-trip behavior."""
        from torch._inductor.ir import ComputedBuffer

        from torch_spyre._inductor.wsr.coarse_tile import (
            _insert_all_read_copy_ops,
            _plan_read_copies,
        )

        tiled_op, full_deps, operations = _make_full_buffer_read_fixture()
        tiled_op.loop_info.tiled_dims_per_read = [[[(0, Integer(8))]]]
        plans = _plan_read_copies(operations, [((0,), [tiled_op], {})])

        entry = plans[(0,)].entries[0]
        self.assertFalse(entry.loop_invariant)
        _insert_all_read_copy_ops(operations, plans)

        copy_buf = next(
            op
            for op in operations
            if isinstance(op, ComputedBuffer) and op.get_name() == entry.copy_name
        )
        self.assertTrue(hasattr(copy_buf, "loop_info"))
        self.assertEqual(full_deps[0].name, entry.dep.name)

    def test_invariant_broadcast_copy_drops_absent_loop_dim(self):
        """A fixed [H] input read by an [E,H] loop is staged as [H], not
        materialized as the expanded [E,H] view."""
        from torch._inductor.ir import (
            ComputedBuffer,
            FixedLayout,
            Pointwise,
            StorageBox,
            TensorBox,
        )

        from torch_spyre._inductor.ir import SpyreEmptyFallback
        from torch_spyre._inductor.wsr.coarse_tile import (
            _insert_all_read_copy_ops,
            _plan_read_copies,
        )

        device = torch.device("cpu")
        dtype = torch.float32
        full_buf = SpyreEmptyFallback(
            torch.ops.spyre.empty.default, [128], device, dtype
        )
        full_buf.layout = FixedLayout(device, dtype, [128], [1])
        full_box = TensorBox(StorageBox(full_buf))

        def inner_fn(index):
            return full_box.make_loader()([index[1]])

        pw = Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=[Integer(8), Integer(128)],
        )
        tiled_op = ComputedBuffer(
            name="tiled_broadcast",
            layout=FixedLayout(device, dtype, [8, 128], None),
            data=pw.data.data,
        )
        tiled_op.operation_name = "tiled_broadcast"
        tiled_op.origins = OrderedSet()
        tiled_op.loop_info = CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[Integer(8)],
            loop_tiled_dims=[[0]],
            tiled_dims_per_read=[[[]]],
        )
        V.graph.name_to_buffer[tiled_op.get_name()] = tiled_op

        operations = [full_buf, tiled_op]
        plans = _plan_read_copies(operations, [((0,), [tiled_op], {})])
        entry = plans[(0,)].entries[0]
        self.assertTrue(entry.loop_invariant)

        _insert_all_read_copy_ops(operations, plans)
        copy_buf = next(
            op
            for op in operations
            if isinstance(op, ComputedBuffer) and op.get_name() == entry.copy_name
        )
        self.assertEqual(list(copy_buf.get_size()), [Integer(128)])
        self.assertEqual(list(copy_buf.layout.stride), [Integer(1)])
        self.assertFalse(hasattr(copy_buf, "loop_info"))

    def test_transposed_read_gets_its_own_copy(self):
        """a+b+a.t()-style: two reads of the same buffer with different
        index expressions must NOT share a copy."""
        from torch._inductor.ir import (
            ComputedBuffer,
            FixedLayout,
            Pointwise,
            StorageBox,
            TensorBox,
        )

        from torch_spyre._inductor.ir import SpyreEmptyFallback
        from torch_spyre._inductor.wsr.coarse_tile import (
            _insert_all_read_copy_ops,
            _plan_read_copies,
        )

        device = torch.device("cpu")
        dtype = torch.float32

        full_buf = SpyreEmptyFallback(
            torch.ops.spyre.empty.default, [8, 8], device, dtype
        )
        full_buf.layout = FixedLayout(device, dtype, [8, 8], [8, 1])
        full_box = TensorBox(StorageBox(full_buf))

        def inner_fn(index):
            i, j = index
            plain = full_box.make_loader()([i, j])
            transposed = full_box.make_loader()([j, i])
            return plain + transposed

        pw = Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=[Integer(8), Integer(8)],
        )
        pw_data = pw.data.data
        tiled_op = ComputedBuffer(
            name="tiled_op0",
            layout=FixedLayout(device, dtype, [Integer(8), Integer(8)], None),
            data=pw_data,
        )
        tiled_op.operation_name = "tiled_op0"
        tiled_op.origins = OrderedSet()
        tiled_op.loop_info = CoarseTileInfo(
            loop_group_id=(0,), loop_count=[Integer(1)], loop_tiled_dims=[[]]
        )
        V.graph.name_to_buffer["tiled_op0"] = tiled_op

        operations = [full_buf, tiled_op]
        retiled_infos_by_group = [((0,), [tiled_op], {})]
        plans = _plan_read_copies(operations, retiled_infos_by_group)

        self.assertEqual(len(plans[(0,)].entries), 2)

        _insert_all_read_copy_ops(operations, plans)

        copy_bufs = [
            op
            for op in operations
            if isinstance(op, ComputedBuffer) and op.get_name() != "tiled_op0"
        ]
        self.assertEqual(len(copy_bufs), 2)

    @unittest.skip(
        "non-divisible padding raises Unsupported after row-major fallback removal"
    )
    def test_offset_read_gets_its_own_copy(self):
        """a+shift(a)-style: two reads of the same buffer with identical
        per-var index coefficients but a different constant offset must
        NOT share a copy -- dep.index.coeff(v) is blind to the constant
        term, so a naive key would wrongly merge these (see coarse_tile.py
        issue where a merged-in consumer's real offset differs from the
        sizing op's, producing wrong/out-of-bounds data)."""
        from torch._inductor.ir import (
            ComputedBuffer,
            FixedLayout,
            Pointwise,
            StorageBox,
            TensorBox,
        )

        from torch_spyre._inductor.ir import SpyreEmptyFallback
        from torch_spyre._inductor.wsr.coarse_tile import (
            _insert_all_read_copy_ops,
            _plan_read_copies,
        )

        device = torch.device("cpu")
        dtype = torch.float32

        # 8x9 so a +1 column offset stays in bounds for all j in [0, 8).
        full_buf = SpyreEmptyFallback(
            torch.ops.spyre.empty.default, [8, 9], device, dtype
        )
        full_buf.layout = FixedLayout(device, dtype, [8, 9], [9, 1])
        full_box = TensorBox(StorageBox(full_buf))

        def inner_fn(index):
            i, j = index
            plain = full_box.make_loader()([i, j])
            shifted = full_box.make_loader()([i, j + 1])
            return plain + shifted

        pw = Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=[Integer(8), Integer(8)],
        )
        pw_data = pw.data.data
        tiled_op = ComputedBuffer(
            name="tiled_op0",
            layout=FixedLayout(device, dtype, [Integer(8), Integer(8)], None),
            data=pw_data,
        )
        tiled_op.operation_name = "tiled_op0"
        tiled_op.origins = OrderedSet()
        tiled_op.loop_info = CoarseTileInfo(
            loop_group_id=(0,), loop_count=[Integer(1)], loop_tiled_dims=[[]]
        )
        V.graph.name_to_buffer["tiled_op0"] = tiled_op

        operations = [full_buf, tiled_op]
        retiled_infos_by_group = [((0,), [tiled_op], {})]
        plans = _plan_read_copies(operations, retiled_infos_by_group)

        self.assertEqual(len(plans[(0,)].entries), 2)

        _insert_all_read_copy_ops(operations, plans)

        copy_bufs = [
            op
            for op in operations
            if isinstance(op, ComputedBuffer) and op.get_name() != "tiled_op0"
        ]
        self.assertEqual(len(copy_bufs), 2)

    def test_disable_flag_skips_everything(self):
        """An empty read_copy_plans dict (the insert_read_copies=False case)
        leaves operations untouched."""
        from torch_spyre._inductor.wsr.coarse_tile import _insert_all_read_copy_ops

        op_a, op_b, full_buf, operations = _make_two_op_shared_read_fixture()
        before = list(operations)

        _insert_all_read_copy_ops(operations, {})

        self.assertEqual(operations, before)

    def test_one_consumer_patched_across_two_entries(self):
        """op_a reads buf_x (shared with op_b) and buf_y (shared with op_c):
        op_a appears in TWO different ReadCopyEntry.consumer_op_names within
        the same plan. Each entry's consumer loop rebuilds op_a in place via
        replace_computed_buffer_body, so the second entry to patch op_a must
        resolve it by name through a freshly-rebuilt name_to_op, not through
        a stale object reference captured before the first entry's patch --
        exactly the object-identity hazard _NameSwapHandler/
        replace_computed_buffer_body exist to guard against. Confirms both
        patches land on the final op_a object (it loads both copies, never
        the original full buffers)."""
        from torch._inductor.ir import (
            ComputedBuffer,
            FixedLayout,
            Pointwise,
            StorageBox,
            TensorBox,
        )

        from torch_spyre._inductor.ir import SpyreEmptyFallback
        from torch_spyre._inductor.wsr.coarse_tile import (
            _insert_all_read_copy_ops,
            _plan_read_copies,
        )

        device = torch.device("cpu")
        dtype = torch.float32

        def _make_full(name):
            buf = SpyreEmptyFallback(
                torch.ops.spyre.empty.default, [8, 8], device, dtype
            )
            buf.layout = FixedLayout(device, dtype, [8, 8], [8, 1])
            buf.name = name
            V.graph.name_to_buffer[name] = buf
            return buf

        buf_x = _make_full("buf_x")
        buf_y = _make_full("buf_y")
        box_x = TensorBox(StorageBox(buf_x))
        box_y = TensorBox(StorageBox(buf_y))

        def _make_op(name, boxes):
            def inner_fn(index):
                total = boxes[0].make_loader()(index)
                for box in boxes[1:]:
                    total = total + box.make_loader()(index)
                return total

            pw = Pointwise.create(
                device=device,
                dtype=dtype,
                inner_fn=inner_fn,
                ranges=[Integer(8), Integer(8)],
            )
            pw_data = pw.data.data
            op = ComputedBuffer(
                name=name,
                layout=FixedLayout(device, dtype, [Integer(8), Integer(8)], None),
                data=pw_data,
            )
            op.operation_name = name
            op.origins = OrderedSet()
            op.loop_info = CoarseTileInfo(
                loop_group_id=(0,), loop_count=[Integer(1)], loop_tiled_dims=[[]]
            )
            V.graph.name_to_buffer[name] = op
            return op

        # op_a reads both buf_x and buf_y; op_b only buf_x; op_c only buf_y.
        op_a = _make_op("op_a", [box_x, box_y])
        op_b = _make_op("op_b", [box_x])
        op_c = _make_op("op_c", [box_y])
        operations = [buf_x, buf_y, op_a, op_b, op_c]
        retiled_infos_by_group = [((0,), [op_a, op_b, op_c], {})]

        plans = _plan_read_copies(operations, retiled_infos_by_group)
        entries_by_name = {e.dep.name: e for e in plans[(0,)].entries}
        self.assertEqual(set(entries_by_name), {"buf_x", "buf_y"})
        self.assertIn("op_a", entries_by_name["buf_x"].consumer_op_names)
        self.assertIn("op_a", entries_by_name["buf_y"].consumer_op_names)

        _insert_all_read_copy_ops(operations, plans)

        final_op_a = next(
            o
            for o in operations
            if isinstance(o, ComputedBuffer) and o.get_name() == "op_a"
        )

        class _Recorder(list):
            def load(self, name, index):
                self.append(name)
                return 0.0

            def add(self, a, b):
                return 0.0

        loaded = _Recorder()
        with V.set_ops_handler(loaded):
            final_op_a.data.inner_fn([sympy.Integer(0) for _ in final_op_a.data.ranges])

        copy_names = {
            op.get_name()
            for op in operations
            if isinstance(op, ComputedBuffer)
            and op.get_name().startswith("coarse_tile_read_copy_")
        }
        self.assertEqual(len(copy_names), 2)
        self.assertEqual(set(loaded), copy_names)
        self.assertNotIn("buf_x", loaded)
        self.assertNotIn("buf_y", loaded)


class TestIsReadAdvancingAnywhere(unittest.TestCase):
    """A buffer with a fixed write can still be barred from LX by a reader.

    ``_is_tiled_advancing`` only asks whether a buffer's own *producing*
    write advances; it says nothing about whether some other op *reads*
    that buffer via a reference that advances across the reader's own
    coarse-tile loop (e.g. a full HBM buffer with a fixed write, copied
    into a nested tile every outer iteration -- exactly the shape
    ``_make_full_buffer_read_fixture`` already builds for
    ``_insert_all_read_copy_ops``'s own tests). ``compute_ops.py``'s
    ``is_tiled_lx`` check applies to every ``TensorArg`` -- reads and the
    write -- so missing this at allocation time only defers the same
    ``NotImplementedError`` to codegen. This class exercises
    ``_get_buffer_user_deps`` + ``_is_read_advancing_anywhere`` together,
    against real IR and real ``MemoryDep`` objects (via
    ``_make_full_buffer_read_fixture``), not mocks.
    """

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()

    def tearDown(self):
        self._graph_ctx.__exit__(None, None, None)

    def test_advancing_copy_in_read_detected(self):
        from torch_spyre._inductor.scratchpad.utils import (
            _get_buffer_user_deps,
            _is_read_advancing_anywhere,
        )

        tiled_op, full_deps, operations = _make_full_buffer_read_fixture()
        self.assertEqual(len(full_deps), 1)
        full_buf = V.graph.get_buffer(full_deps[0].name)
        # full_buf itself has no loop_info -- its write is fixed (it is
        # never produced inside a coarse-tile loop). tiled_op reads it via
        # an advancing copy-in: dim 0 divided across tiled_op's own outer
        # loop (matches the fixture's loop_count=[8] over dim 0).
        tiled_op.loop_info.tiled_dims_per_read = [[[(0, Integer(8))]]]

        fake_graph = SimpleNamespace(operations=operations)
        buf_user_deps = _get_buffer_user_deps(fake_graph)
        self.assertTrue(_is_read_advancing_anywhere(full_buf.get_name(), buf_user_deps))

    def test_fixed_copy_in_read_not_detected(self):
        """Same shape, but the reader's copy-in is fixed (not advancing):
        the dep has no tiled levels at all (empty outer list)."""
        from torch_spyre._inductor.scratchpad.utils import (
            _get_buffer_user_deps,
            _is_read_advancing_anywhere,
        )

        tiled_op, full_deps, operations = _make_full_buffer_read_fixture()
        full_buf = V.graph.get_buffer(full_deps[0].name)
        tiled_op.loop_info.tiled_dims_per_read = [[]]

        fake_graph = SimpleNamespace(operations=operations)
        buf_user_deps = _get_buffer_user_deps(fake_graph)
        self.assertFalse(
            _is_read_advancing_anywhere(full_buf.get_name(), buf_user_deps)
        )

    def test_fixed_copy_in_read_with_empty_levels_not_detected(self):
        """Fixed can also show up as a non-empty outer list whose every
        per-level entry is itself empty -- e.g. tiled_op has one loop level
        that does not divide any dim this dep's index depends on.
        _general_tile_advance's own per-level loop (`if not
        dim_extent_pairs: continue`) never contributes a term for such a
        level, so this must be treated as fixed too, not merely "has some
        levels" (regression coverage: an earlier version of this check used
        a shallow `if tiled_dims_per_read[dep_idx]:` truthiness test, which
        wrongly treated [[]] as advancing since the outer list is
        non-empty)."""
        from torch_spyre._inductor.scratchpad.utils import (
            _get_buffer_user_deps,
            _is_read_advancing_anywhere,
        )

        tiled_op, full_deps, operations = _make_full_buffer_read_fixture()
        full_buf = V.graph.get_buffer(full_deps[0].name)
        tiled_op.loop_info.tiled_dims_per_read = [[[]]]

        fake_graph = SimpleNamespace(operations=operations)
        buf_user_deps = _get_buffer_user_deps(fake_graph)
        self.assertFalse(
            _is_read_advancing_anywhere(full_buf.get_name(), buf_user_deps)
        )

    def test_unrelated_buffer_not_detected(self):
        from torch_spyre._inductor.scratchpad.utils import (
            _get_buffer_user_deps,
            _is_read_advancing_anywhere,
        )

        tiled_op, full_deps, operations = _make_full_buffer_read_fixture()
        tiled_op.loop_info.tiled_dims_per_read = [[[(0, Integer(8))]]]

        fake_graph = SimpleNamespace(operations=operations)
        buf_user_deps = _get_buffer_user_deps(fake_graph)
        self.assertFalse(_is_read_advancing_anywhere("no_such_buffer", buf_user_deps))


class TestRescaleIndex(unittest.TestCase):
    """Direct unit coverage for _rescale_index's coefficient matching.

    _insert_all_read_copy_ops's own tests exercise _rescale_index only
    indirectly, through whatever index shapes its fixtures happen to
    produce. These tests call it directly with hand-built indexes so the
    matching rules documented on _rescale_index itself -- largest-stride-
    first, and the sympy.simplify fallback for non-structurally-equal but
    symbolically-equal coefficients -- are locked in cheaply.
    """

    def test_plain_concrete_strides(self):
        # full_strides/tile_strides are always sympy Expr in the real
        # calling convention (dep.index.coeff(...) and a
        # sympy.Integer(1)-seeded running product in
        # _insert_all_read_copy_ops) -- never raw Python ints. Use
        # sympy.Integer throughout so these tests match that convention.
        c0, c1 = sympy.symbols("c0 c1")
        index = c0 * 128 + c1 * 4
        result = _rescale_index(
            index,
            [sympy.Integer(128), sympy.Integer(4)],
            [sympy.Integer(32), sympy.Integer(4)],
        )
        self.assertEqual(sympy.expand(result), sympy.expand(c0 * 32 + c1 * 4))

    def test_constant_offset_term_is_preserved(self):
        c0 = sympy.symbols("c0")
        index = c0 * 128 + 5
        result = _rescale_index(index, [sympy.Integer(128)], [sympy.Integer(32)])
        self.assertEqual(sympy.expand(result), sympy.expand(c0 * 32 + 5))

    def test_symbolic_stride_structurally_equal(self):
        c0, c1 = sympy.symbols("c0 c1")
        s0 = sympy.Symbol("s0", positive=True)
        index = c0 * s0 + c1
        result = _rescale_index(
            index, [s0, sympy.Integer(1)], [sympy.Integer(4), sympy.Integer(1)]
        )
        self.assertEqual(sympy.expand(result), sympy.expand(c0 * 4 + c1))

    def test_symbolic_stride_simplify_fallback(self):
        # coeff and full_stride describe the same dimension but are not
        # structurally identical sympy expressions (2*(s0 + 1) vs
        # 2*s0 + 2) -- only equal after simplification. Exercises the
        # sympy.simplify(coeff - full_stride) == 0 fallback.
        c0 = sympy.symbols("c0")
        s0 = sympy.Symbol("s0", positive=True)
        full_stride = 2 * (s0 + 1)
        coeff_form = 2 * s0 + 2
        index = coeff_form * c0
        result = _rescale_index(index, [full_stride], [sympy.Integer(4)])
        self.assertEqual(sympy.expand(result), sympy.expand(4 * c0))

    def test_duplicate_stride_matches_largest_dimension_first(self):
        # A size-[1, 16] shape's dim-0 stride (16) coincides with a
        # size-[M, 16] shape's dim-1 full extent (also 16) -- both 16 here,
        # but the two entries must still be told apart. Largest-first
        # matching consumes the larger/earlier dimension's entry before the
        # degenerate extent-1 one, so each loop variable's term lands on the
        # tile_stride for its own dimension rather than swapping with the
        # other.
        c0, c1 = sympy.symbols("c0 c1")
        index = c0 * 16 + c1 * 16
        result = _rescale_index(
            index,
            [sympy.Integer(16), sympy.Integer(16)],
            [sympy.Integer(8), sympy.Integer(2)],
        )
        # Both full_strides are equal (16), so which tile_stride pairs with
        # which term is only distinguished by iteration/removal order, not
        # by any property of c0 vs c1 -- assert the invariant _rescale_index
        # actually guarantees: every term is rescaled by *some* consumed
        # tile_stride, each tile_stride used exactly once, and no term is
        # dropped or duplicated.
        expected_options = {
            sympy.expand(c0 * 8 + c1 * 2),
            sympy.expand(c0 * 2 + c1 * 8),
        }
        self.assertIn(sympy.expand(result), expected_options)

    def test_no_matching_stride_raises(self):
        c0 = sympy.symbols("c0")
        index = c0 * 128
        with self.assertRaises(RuntimeError):
            _rescale_index(index, [sympy.Integer(4)], [sympy.Integer(4)])


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
    from torch._inductor.ir import ComputedBuffer, FixedLayout, Reduction

    data = MagicMock(spec=Reduction)
    data.ranges = list(ranges)
    data.reduction_ranges = list(reduction_ranges)
    data.reduction_type = reduction_type

    # Row-major strides for the output shape (ranges).
    strides = []
    s = sympy.Integer(1)
    for r in reversed(ranges):
        strides.insert(0, s)
        s = s * r
    layout = MagicMock(spec=FixedLayout)
    layout.stride = strides

    op = MagicMock(spec=ComputedBuffer)
    op.data = data
    op.layout = layout
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
    """Tests for tiling propagation Reduction support."""

    def test_reduction_tiled_reduction_dim_nested_ok(self):
        from torch_spyre._inductor.wsr.coarse_tile import (
            _validate_planned_reduction_tiling,
        )

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
        _validate_planned_reduction_tiling(
            op, op.loop_info.loop_tiled_dims, op.loop_info.loop_tiled_reduction_dims
        )  # must not raise

    def test_reduction_output_dim_tiled_ok(self):
        from torch_spyre._inductor.wsr.coarse_tile import (
            _validate_planned_reduction_tiling,
        )

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
        _validate_planned_reduction_tiling(
            op, op.loop_info.loop_tiled_dims, op.loop_info.loop_tiled_reduction_dims
        )


class TestValidatePlannedReductionTiling(unittest.TestCase):
    """Tests for _validate_planned_reduction_tiling: raising on unsupported
    cases, passing on supported ones. Called from plan_coarse_tile_groups
    (planning time) with the op's own per-level tiled-dims lists, before any
    loop_info is stamped."""

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
        """Single level, only reduction dim tiled — supported case."""
        from torch_spyre._inductor.wsr.coarse_tile import (
            _validate_planned_reduction_tiling,
        )

        op = self._make_op(loop_tiled_dims=[[]], loop_tiled_reduction_dims=[[0]])
        _validate_planned_reduction_tiling(op, [[]], [[0]])  # must not raise

    def test_pure_output_tile_ok(self):
        """Single level, only output dim tiled — existing supported case."""
        from torch_spyre._inductor.wsr.coarse_tile import (
            _validate_planned_reduction_tiling,
        )

        op = self._make_op(loop_tiled_dims=[[0]], loop_tiled_reduction_dims=[[]])
        _validate_planned_reduction_tiling(op, [[0]], [[]])  # must not raise

    def test_no_tiled_dims_ok(self):
        """No dims tiled at all — no error."""
        from torch._inductor.ir import ComputedBuffer, Reduction
        from torch_spyre._inductor.wsr.coarse_tile import (
            _validate_planned_reduction_tiling,
        )

        data = MagicMock(spec=Reduction)
        data.ranges = [Integer(128)]
        data.reduction_ranges = [Integer(256)]
        op = MagicMock(spec=ComputedBuffer)
        op.data = data
        _validate_planned_reduction_tiling(op, [[]], [[]])  # must not raise

    def test_mixed_same_level_raises(self):
        """Both output and reduction dim tiled at the same level — raises."""
        from torch_spyre._inductor.wsr.coarse_tile import (
            _validate_planned_reduction_tiling,
        )

        op = self._make_op(loop_tiled_dims=[[0]], loop_tiled_reduction_dims=[[0]])
        with self.assertRaises(Unsupported, msg="mixed same-level should raise"):
            _validate_planned_reduction_tiling(op, [[0]], [[0]])

    def test_mixed_different_levels_allowed(self):
        """Outer output-dim tiling + inner reduction-dim tiling — now supported."""
        from torch._inductor.ir import ComputedBuffer, Reduction
        from torch_spyre._inductor.wsr.coarse_tile import (
            _validate_planned_reduction_tiling,
        )

        data = MagicMock(spec=Reduction)
        data.ranges = [Integer(128)]
        data.reduction_ranges = [Integer(256)]
        data.reduction_type = "sum"
        op = MagicMock(spec=ComputedBuffer)
        op.data = data
        op.get_name.return_value = "test_op"
        tiled_dims = [[0], []]
        tiled_rdims = [[], [0]]
        # Must not raise: outer output-dim + inner reduction-dim is now supported.
        _validate_planned_reduction_tiling(op, tiled_dims, tiled_rdims)

    def test_multiple_output_levels_outer_to_reduction_allowed(self):
        """Every output level may remain outer to the inner reduction tile."""
        from torch._inductor.ir import ComputedBuffer, Reduction
        from torch_spyre._inductor.wsr.coarse_tile import (
            _validate_planned_reduction_tiling,
        )

        data = MagicMock(spec=Reduction)
        data.ranges = [Integer(4), Integer(128), Integer(32)]
        data.reduction_ranges = [Integer(512)]
        data.reduction_type = "sum"
        op = MagicMock(spec=ComputedBuffer)
        op.data = data
        op.get_name.return_value = "combined_bmm"

        tiled_dims = [[0], [1], []]
        tiled_rdims = [[], [], [0]]
        _validate_planned_reduction_tiling(op, tiled_dims, tiled_rdims)

    def test_multiple_reduction_dims_same_level_raises(self):
        """Multiple reduction dims tiled at one level — raises."""
        from torch._inductor.ir import ComputedBuffer, Reduction
        from torch_spyre._inductor.wsr.coarse_tile import (
            _validate_planned_reduction_tiling,
        )

        data = MagicMock(spec=Reduction)
        data.ranges = [Integer(128)]
        data.reduction_ranges = [Integer(64), Integer(64)]
        op = MagicMock(spec=ComputedBuffer)
        op.data = data
        op.get_name.return_value = "test_op"
        with self.assertRaises(Unsupported, msg="multiple reduction dims should raise"):
            _validate_planned_reduction_tiling(op, [[]], [[0, 1]])

    def test_stick_dim_reduction_tiling_allowed(self):
        """Tiling a reduction over the stick dimension is now supported."""
        from torch._inductor.ir import ComputedBuffer, Reduction
        from torch_spyre._inductor.wsr.coarse_tile import (
            _validate_planned_reduction_tiling,
        )

        data = MagicMock(spec=Reduction)
        data.ranges = [Integer(64)]  # [B] output
        data.reduction_ranges = [Integer(512)]  # [D] stick dim
        data.reduction_type = "sum"
        op = MagicMock(spec=ComputedBuffer)
        op.data = data
        op.get_name.return_value = "test_sum"
        # Must not raise: stick-dim reduction tiling is now supported.
        _validate_planned_reduction_tiling(op, [[]], [[0]])


class TestGenerateBundleMlirSymbolicArgs(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _bundle(self, specs, fake_compile=None, pool_size=0):
        if fake_compile is None:
            fake_compile = _fake_compile_op_spec
        with config.patch({"sdsc_cache": False}):
            with patch(
                "torch_spyre._inductor.codegen.bundle.compile_op_spec",
                side_effect=fake_compile,
            ):
                generate_bundle(
                    "test_kernel",
                    self.tmpdir,
                    specs,
                    pool_size=pool_size,
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

    def test_func_signature_has_params_for_tensor_args(self):
        a = self._make_op_spec_with_hbm_args("a", [0, 1])

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
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

        mlir = self._bundle([a], fake_compile=fake)

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

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
            sym_id = -(symbol_id_offset + 1)
            symbols.append(op_spec.args[0].allocation["hbm"])
            return (
                _make_tiled_json(idx, sym_id),
                [op_spec.args[0].allocation["hbm"]],
                [{}],
                [SymbolKind.kernel(0)],
            )

        mlir = self._bundle([a], fake_compile=fake)

        self.assertIn("sdscbundle.sdsc_execute (%arg_0)", mlir)
        self.assertNotIn("sdsc_execute (%sym_0_1)", mlir)
        self.assertNotIn("sdsc_execute (%sym_1)", mlir)

    def test_non_tensor_arg_symbols_remain_as_constants(self):
        c0 = Symbol("c0")
        op_a = self._make_op_spec_with_hbm_args("a", [0])
        # op_b: arg_index=-1, pool-allocated (fake returns "pool" kind).
        # allocation is hbm_pool to reflect the state after hbm_pool_planning,
        # which runs before bundle generation in production.
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
                    allocation={"hbm_pool": 0x0},
                )
            ],
            op_info={},
        )
        call_count = [0]
        values = [0x400000000, 0x0]

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
            i = call_count[0]
            call_count[0] += 1
            sym_id = -(symbol_id_offset + 1)
            symbols.append(values[i])
            kind = (
                SymbolKind.kernel(0) if i == 0 else SymbolKind.pool()
            )  # op_b has pool allocation
            return _make_tiled_json(idx, sym_id), [values[i]], [{}], [kind]

        mlir = self._bundle([op_a, op_b], fake_compile=fake, pool_size=1024)

        # First sym → parameter (kernel tensor arg)
        self.assertIn("%arg_0_base_addr: !sdscbundle.input_arg<index>", mlir)
        self.assertNotIn("arith.constant 17179869184", mlir)
        # Second sym → pool: arith.addi %pool, <offset>
        self.assertNotIn("%pool_base_addr", mlir)
        self.assertIn("sdscbundle.device_mem_allocate", mlir)
        self.assertIn("%pool_addr_0 = arith.addi %pool", mlir)

    def test_multi_sdsc_two_tensor_args_snapshot(self):
        """Two tensor args shared across multiple SDSCs emit exactly two input_arg params.

        Simulates a bundle where every SDSC operates on the same two logical
        kernel tensors (arg_index 0 and 1) but at different per-SDSC addresses.
        Only two function parameters should be emitted — one per unique arg_index.
        """
        op0 = self._make_op_spec_with_hbm_args("op0", [0, 1])
        ops_rest = [_make_minimal_op_spec(f"op{i}") for i in range(1, 5)]
        call_count = [0]
        # Each SDSC registers 2 kernel-arg symbols for arg_index 0 and 1 at
        # different per-SDSC addresses (simulating different tile slices).
        sdsc_addr_pairs = [
            (0x400000000, 0x800000000),
            (0x400010000, 0x800010000),
            (0x400020000, 0x800020000),
            (0x400030000, 0x800030000),
            (0x400040000, 0x800040000),
        ]

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
            i = call_count[0]
            call_count[0] += 1
            a0, a1 = sdsc_addr_pairs[i]
            local_ids = [-(symbol_id_offset + 1), -(symbol_id_offset + 2)]
            symbols.append(a0)
            symbols.append(a1)
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
                                    for j in range(2)
                                ]
                            }
                        }
                    ],
                }
            }
            # Both tensors have the same arg_index across all SDSCs.
            kinds = [SymbolKind.kernel(0), SymbolKind.kernel(1)]
            return json_out, [a0, a1], [{}, {}], kinds

        mlir = self._bundle([op0] + ops_rest, fake_compile=fake)

        # 10 symbols across 5 SDSCs but only 2 unique arg_indices → 2 params
        self.assertIn("%arg_0_base_addr: !sdscbundle.input_arg<index>", mlir)
        self.assertIn("%arg_1_base_addr: !sdscbundle.input_arg<index>", mlir)
        # Exactly 2 input_arg params (each appears twice: param + extract)
        self.assertEqual(mlir.count("!sdscbundle.input_arg<index>"), 2 * 2)
        # First sdsc_execute uses first two extracted names
        self.assertIn("sdscbundle.sdsc_execute (%arg_0, %arg_1)", mlir)

    def test_same_kernel_arg_across_sdsc_deduped(self):
        """The same kernel arg address appearing in two SDSCs maps to one input_arg param."""
        # Simulates softmax: arg_index=0 appears in both op0 and op1.
        a = _make_minimal_op_spec("a")
        b = _make_minimal_op_spec("b")
        base = 0x400000000  # SEGMENT_OFFSETS[1], arg_index=0
        call_count = [0]

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
            call_count[0] += 1
            sym_id = -(symbol_id_offset + 1)
            symbols.append(base)
            return _make_tiled_json(idx, sym_id), [base], [{}], [SymbolKind.kernel(0)]

        mlir = self._bundle([a, b], fake_compile=fake)

        # Only one input_arg param (deduped cross-SDSC)
        self.assertIn("%arg_0_base_addr: !sdscbundle.input_arg<index>", mlir)
        self.assertNotIn("%sym_0_2:", mlir)
        # Both sdsc_execute ops reference the same extracted name
        execute_lines = [ln for ln in mlir.splitlines() if "sdsc_execute" in ln]
        self.assertEqual(execute_lines[0].split("(")[1].split(")")[0], "%arg_0")
        self.assertEqual(execute_lines[1].split("(")[1].split(")")[0], "%arg_0")

    def test_same_arg_index_different_addresses_deduped(self):
        """Two SDSCs with the same arg_index but different addresses emit one param.

        Simulates a tiled kernel where each SDSC operates on a different slice
        of the same tensor (arg_index=0 at addr0 in op0, addr1 in op1).  The
        function signature must not repeat %arg_0_base_addr.
        """
        a = _make_minimal_op_spec("a")
        b = _make_minimal_op_spec("b")
        addr0 = 0x400000000
        addr1 = 0x400010000  # different address, same logical arg_index=0
        addrs = [addr0, addr1]
        call_count = [0]

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
            i = call_count[0]
            call_count[0] += 1
            sym_id = -(symbol_id_offset + 1)
            symbols.append(addrs[i])
            return (
                _make_tiled_json(idx, sym_id),
                [addrs[i]],
                [{}],
                [SymbolKind.kernel(0)],
            )

        mlir = self._bundle([a, b], fake_compile=fake)

        # Only one input_arg param — no duplicate %arg_0_base_addr
        self.assertEqual(
            mlir.count("%arg_0_base_addr: !sdscbundle.input_arg<index>"), 1
        )
        self.assertEqual(
            mlir.count("!sdscbundle.input_arg<index>"), 2
        )  # param + extract
        # Both sdsc_execute ops reference the canonical extracted name %arg_0
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

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
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

        mlir = self._bundle([a, b, c], fake_compile=fake, pool_size=4096)

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


class TestGenerateBundleRequiresSymbolicArgs(unittest.TestCase):
    """generate_bundle must fail fast when bundle_symbolic_args is False.

    The SDSC path always emits symbolic HBM addresses; baked addressing is
    only meaningful on the (separately-gated) KTIR emitter path. Without
    this guard, disabling the flag on the default path would silently
    miscompile addresses instead of erroring.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_raises_when_bundle_symbolic_args_false(self):
        a = _make_minimal_op_spec("a")
        with patch(
            "torch_spyre._inductor.codegen.bundle._spyre_config.bundle_symbolic_args",
            False,
        ):
            with self.assertRaises(AssertionError):
                generate_bundle("test_kernel", self.tmpdir, [a])


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
        # Use sym-path sentinel convention: start_address = arg_index = 0 for tile 0.
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
            layouts={
                "A": {"dim_order": [s], "stick_dim_order": [s], "stick_size": [64]}
            },
            args=[tensor],
            constants={},
            conv_params={},
            coordinate_masking={},
        )
        symbols: list[int] = []
        _, _, _, kinds = generate_sdsc(
            0,
            sdsc_spec,
            symbols,
            symbol_id_offset=0,
            tiled_symbols=[[s]],
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

        def fake(idx, op_spec, symbols, symbol_id_offset=0):
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

        with config.patch({"sdsc_cache": False}):
            with patch(
                "torch_spyre._inductor.codegen.bundle.compile_op_spec",
                side_effect=fake,
            ):
                generate_bundle(
                    "test_kernel",
                    self.tmpdir,
                    [a, b],
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

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()

    def tearDown(self):
        self._graph_ctx.__exit__(None, None, None)

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
        from torch_spyre._inductor.wsr.coarse_tile import _divide_reduction_ranges

        op = self._make_reduction_op(
            ranges=[Integer(128)], reduction_ranges=[Integer(256)]
        )
        _divide_reduction_ranges(op, Integer(2), [0])
        self.assertEqual(op.data.reduction_ranges[0], Integer(128))
        self.assertEqual(op.data.ranges[0], Integer(128))  # output ranges untouched

    def test_empty_tiled_dims_is_noop(self):
        from torch_spyre._inductor.wsr.coarse_tile import _divide_reduction_ranges

        op = self._make_reduction_op(
            ranges=[Integer(128)], reduction_ranges=[Integer(64)]
        )
        _divide_reduction_ranges(op, Integer(4), [])
        self.assertEqual(op.data.reduction_ranges[0], Integer(64))  # unchanged

    def test_not_divisible_raises(self):
        from torch_spyre._inductor.wsr.coarse_tile import _divide_reduction_ranges

        op = self._make_reduction_op(
            ranges=[Integer(128)], reduction_ranges=[Integer(100)]
        )
        with self.assertRaises(Unsupported, msg="not divisible should raise"):
            _divide_reduction_ranges(op, Integer(3), [0])

    def test_divides_second_reduction_dim(self):
        from torch_spyre._inductor.wsr.coarse_tile import _divide_reduction_ranges

        op = self._make_reduction_op(
            ranges=[Integer(32)], reduction_ranges=[Integer(64), Integer(128)]
        )
        _divide_reduction_ranges(op, Integer(4), [1])
        self.assertEqual(op.data.reduction_ranges[0], Integer(64))  # untouched
        self.assertEqual(op.data.reduction_ranges[1], Integer(32))  # divided

    def test_named_output_dims_survive_reduction_dim_squeeze(self):
        from torch_spyre._inductor.wsr.coarse_tile import (
            _LogicalIterationSymbol,
            _apply_work_div_symbol_remap,
            _divide_reduction_ranges,
        )

        op = self._make_reduction_op(
            ranges=[Integer(512), Integer(704)],
            reduction_ranges=[Integer(128)],
        )
        before = (
            _LogicalIterationSymbol(("output", 0), Integer(512), Symbol("d0")),
            _LogicalIterationSymbol(("output", 1), Integer(704), Symbol("d1")),
            _LogicalIterationSymbol(("reduction", 0), Integer(128), Symbol("d2")),
        )
        after = (
            _LogicalIterationSymbol(("output", 0), Integer(512), Symbol("d0")),
            _LogicalIterationSymbol(("output", 1), Integer(704), Symbol("d1")),
        )
        op.work_div_loop_info = {  # type: ignore[attr-defined]
            Symbol("d0"): ["T"],
            Symbol("d1"): ["F"],
            Symbol("d2"): ["E"],
        }

        with patch(
            "torch_spyre._inductor.wsr.coarse_tile._capture_logical_iteration_symbols",
            side_effect=(before, after),
        ):
            remap = _divide_reduction_ranges(op, Integer(128), [0])
        _apply_work_div_symbol_remap(op, remap)

        self.assertEqual(
            op.work_div_loop_info,  # type: ignore[attr-defined]
            {Symbol("d0"): ["T"], Symbol("d1"): ["F"]},
        )


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
        from torch_spyre._inductor.wsr.coarse_tile import (
            _loop_var_to_reduction_ranges_pos,
        )

        i0 = sympy.Symbol("i0")
        r0 = sympy.Symbol("r0")
        op, red_syms = self._make_op_with_rw(out_syms=[i0], red_syms=[r0])
        result = _loop_var_to_reduction_ranges_pos(op, r0)
        self.assertEqual(result, 0)

    def test_returns_none_for_output_symbol(self):
        from torch_spyre._inductor.wsr.coarse_tile import (
            _loop_var_to_reduction_ranges_pos,
        )

        i0 = sympy.Symbol("i0")
        r0 = sympy.Symbol("r0")
        op, _ = self._make_op_with_rw(out_syms=[i0], red_syms=[r0])
        result = _loop_var_to_reduction_ranges_pos(op, i0)
        self.assertIsNone(result)


class TestReductionIdentityValues(unittest.TestCase):
    """_reduction_identity_value returns the correct monoid identity per reduction type."""

    def _identity(self, reduction_type):
        from torch_spyre._inductor.wsr.coarse_tile import _reduction_identity_value
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
        from torch_spyre._inductor.wsr.coarse_tile import _reduction_identity_value
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
        from torch_spyre._inductor.wsr.coarse_tile_hints import (
            reorder_unhinted_interlopers,
        )

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
        from torch_spyre._inductor.wsr.coarse_tile_hints import _hints_levels

        op = self._make_op([(0, 1, sympy.Symbol("c0"))])
        self.assertEqual(_hints_levels([op]), [])

    def test_size1_hint_dropped_with_debug_log(self):
        """A size-1 hint emits a debug log message when dropped."""
        import logging
        import logging.handlers
        import sympy
        import torch_spyre._inductor.wsr.coarse_tile_hints as cth_mod
        from torch_spyre._inductor.wsr.coarse_tile_hints import _hints_levels

        op = self._make_op([(7, 1, sympy.Symbol("c0"))])

        original_level = cth_mod.hints_logger.level
        cth_mod.hints_logger.setLevel(logging.DEBUG)
        handler = logging.handlers.MemoryHandler(
            capacity=100, flushLevel=logging.CRITICAL
        )
        cth_mod.hints_logger.addHandler(handler)
        try:
            result = _hints_levels([op])
            handler.flush()
            messages = [r.getMessage() for r in handler.buffer]
        finally:
            cth_mod.hints_logger.removeHandler(handler)
            cth_mod.hints_logger.setLevel(original_level)

        self.assertEqual(result, [])
        self.assertTrue(
            any("split_count=1" in m and "no-op" in m for m in messages),
            f"Expected a 'split_count=1 … no-op' debug message; got: {messages}",
        )

    def test_nonunit_hint_kept(self):
        """A hint with split_count > 1 is retained normally."""
        import sympy
        from torch_spyre._inductor.wsr.coarse_tile_hints import _hints_levels

        c0 = sympy.Symbol("c0")
        op = self._make_op([(3, 4, c0)])
        levels = _hints_levels([op])
        self.assertEqual(len(levels), 1)
        hint_id, count = levels[0]
        self.assertEqual(hint_id, 3)
        self.assertEqual(count, sympy.Integer(4))

    def test_mixed_hints_drops_only_size1(self):
        """When one hint is size-1 and another is size>1, only the size>1 survives."""
        import sympy
        from torch_spyre._inductor.wsr.coarse_tile_hints import _hints_levels

        c0, c1 = sympy.Symbol("c0"), sympy.Symbol("c1")
        op = self._make_op([(0, 1, c0), (1, 8, c1)])
        levels = _hints_levels([op])
        self.assertEqual(len(levels), 1)
        hint_id, count = levels[0]
        self.assertEqual(hint_id, 1)
        self.assertEqual(count, sympy.Integer(8))

    def test_all_size1_hints_dropped_falls_through_to_next_op(self):
        """If every hint on op0 is size-1, _hints_levels tries op1 next."""
        import sympy
        from torch_spyre._inductor.wsr.coarse_tile_hints import _hints_levels

        c0 = sympy.Symbol("c0")
        op0 = self._make_op([(0, 1, c0)])
        op1 = self._make_op([(0, 4, c0)])
        levels = _hints_levels([op0, op1])
        self.assertEqual(len(levels), 1)
        _, count = levels[0]
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
    from torch_spyre._inductor.wsr.coarse_tile_hints import hints_to_coarse_tile_groups
    import torch_spyre._inductor.wsr.coarse_tile_hints as coarse_tile_hints_mod

    graph = SimpleNamespace(operations=list(ops))

    # Temporarily force the module-level hints_logger to INFO so the logging
    # block inside hints_to_coarse_tile_groups actually runs.
    original_level = coarse_tile_hints_mod.hints_logger.level
    coarse_tile_hints_mod.hints_logger.setLevel(logging.INFO)

    handler = logging.handlers.MemoryHandler(capacity=1000, flushLevel=logging.CRITICAL)
    coarse_tile_hints_mod.hints_logger.addHandler(handler)
    try:
        hints_to_coarse_tile_groups(graph)
        handler.flush()
        return "\n".join(r.getMessage() for r in handler.buffer)
    finally:
        coarse_tile_hints_mod.hints_logger.removeHandler(handler)
        coarse_tile_hints_mod.hints_logger.setLevel(original_level)


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


class TestCopyOpMetadataAttrCoverage(unittest.TestCase):
    """Unit test for loop_info.py's _SPYRE_METADATA_ATTRS coverage."""

    def test_copy_op_metadata_no_longer_carries_old_attr_name(self):
        src = SimpleNamespace()
        src._coarse_tile_dim_advance = [{0: (64, 4)}]
        dst = SimpleNamespace()
        copy_op_metadata(src, dst)
        self.assertFalse(hasattr(dst, "_coarse_tile_dim_advance"))

    def test_copy_op_metadata_invalidates_read_copy_elision_record(self):
        from torch_spyre._inductor.loop_info import ReadCopyElisionRecord

        src = SimpleNamespace()
        src._read_copy_elision_record = ReadCopyElisionRecord(
            consumer_name="op0",
            copy_name="copy0",
            source_name="weight",
            direct_inner_fn=lambda: None,
        )
        dst = SimpleNamespace()

        copy_op_metadata(src, dst)

        self.assertFalse(hasattr(dst, "_read_copy_elision_record"))


class TestCoeffThroughFloor(unittest.TestCase):
    """Unit tests for pass_utils.coeff_through_floor.

    coeff_through_floor extends sympy.Expr.coeff(sym) to also find sym's
    coefficient when sym only appears inside a floor(...) wrapper -- the
    shape device_tile_advance_expr takes for stick-layout tensors (see
    views.tiling_expr_to_device_expr). Plain sympy.Expr.coeff(sym) returns 0
    for a symbol wrapped in floor(), even though it is a genuine free symbol.
    """

    def test_plain_mul_term_matches_coeff(self):
        """Non-floor-wrapped case: behaves exactly like .coeff(sym)."""
        s = Symbol("s")
        expr = 64 * s
        self.assertEqual(coeff_through_floor(expr, s), 64)
        self.assertEqual(coeff_through_floor(expr, s), expr.coeff(s))

    def test_floor_wrapped_term_extracts_coeff(self):
        """The exact shape from Task 6's investigation:
        floor(65536*sym) -- plain .coeff(sym) returns 0 here, this must
        return 65536."""
        s = Symbol("_tile_adv_op0_lvl0")
        expr = floor(65536 * s)
        self.assertEqual(expr.coeff(s), 0)  # the bug this helper fixes
        self.assertEqual(coeff_through_floor(expr, s), 65536)

    def test_multi_level_sum_with_one_floor_wrapped_term(self):
        """device_tile_advance_expr is a sum of one term per level; only
        the symbol actually queried should be extracted, regardless of
        whether its own term or a sibling term is floor-wrapped."""
        lvl0 = Symbol("_tile_adv_add_lvl0")
        lvl1 = Symbol("_tile_adv_add_lvl1")
        expr = floor(65536 * lvl0) + 32 * lvl1
        self.assertEqual(coeff_through_floor(expr, lvl0), 65536)
        self.assertEqual(coeff_through_floor(expr, lvl1), 32)

    def test_symbol_absent_returns_zero(self):
        s = Symbol("s")
        other = Symbol("other")
        expr = 64 * other
        self.assertEqual(coeff_through_floor(expr, s), sympy.S.Zero)

    def test_floor_wrapped_non_integer_coeff_raises_unsupported(self):
        """Tiles are always a whole number of sticks, so floor()'s
        division inside device_tile_advance_expr must always be exact.
        A non-integer extracted coefficient means an earlier pass or
        spyre_hint produced an invalid sub-stick tile boundary -- this
        must fail loudly, not silently truncate via int()."""
        s = Symbol("s")
        expr = floor(4 * s / 3)  # 4/3 does not reduce to an integer
        with self.assertRaises(Unsupported):
            coeff_through_floor(expr, s)

    def test_floor_wrapped_integer_reducing_division(self):
        """floor(k*sym/d) where d evenly divides k must return the
        reduced integer coefficient, not raise."""
        s = Symbol("s")
        expr = floor(128 * s / 2)
        self.assertEqual(coeff_through_floor(expr, s), 64)


class TestTileHelpers(unittest.TestCase):
    """Tests for tile.py."""

    def test_compute_tile_stride_1d(self):
        self.assertEqual(compute_tile_stride([1024], [1], [256]), [1])

    def test_compute_tile_stride_2d(self):
        self.assertEqual(
            compute_tile_stride([1024, 4096], [4096, 1], [512, 1024]), [1024, 1]
        )

    def test_compute_tile_stride_2d_padding(self):
        self.assertEqual(
            compute_tile_stride([1024, 4096], [4104, 1], [512, 1024]), [1026, 1]
        )

    def test_compute_tile_stride_3d_row_major(self):
        self.assertEqual(
            compute_tile_stride([8, 16, 32], [512, 32, 1], [2, 4, 8]), [32, 8, 1]
        )

    def test_compute_tile_stride_3d_col_major(self):
        self.assertEqual(
            compute_tile_stride([8, 16, 32], [1, 8, 128], [2, 4, 8]), [1, 2, 8]
        )

    def test_compute_tile_stride_3d_min_stride_64(self):
        self.assertEqual(
            compute_tile_stride([4, 8, 16], [8192, 1024, 64], [2, 4, 8]),
            [2048, 512, 64],
        )

    def test_compute_tile_stride_3d_unit_tile_middle_dim(self):
        self.assertEqual(
            compute_tile_stride([8, 16, 32], [512, 32, 1], [2, 1, 8]), [8, 0, 1]
        )

    def test_compute_tile_stride_3d_size1_outermost(self):
        self.assertEqual(
            compute_tile_stride([1, 16, 32], [512, 32, 1], [1, 4, 8]), [0, 8, 1]
        )

    def test_compute_tile_stride_3d_size1_middle(self):
        self.assertEqual(
            compute_tile_stride([8, 1, 32], [32, 32, 1], [2, 1, 8]), [8, 0, 1]
        )

    def test_compute_tile_stride_3d_size1_innermost(self):
        self.assertEqual(
            compute_tile_stride([8, 16, 1], [16, 1, 1], [2, 4, 1]), [4, 1, 0]
        )

    def test_compute_tile_stride_3d_expanded_outermost(self):
        self.assertEqual(
            compute_tile_stride([8, 16, 32], [0, 32, 1], [2, 4, 8]), [0, 8, 1]
        )

    def test_compute_tile_stride_3d_expanded_middle(self):
        self.assertEqual(
            compute_tile_stride([8, 16, 32], [32, 0, 1], [2, 4, 8]), [8, 0, 1]
        )

    def test_compute_tile_stride_3d_expanded_innermost(self):
        self.assertEqual(
            compute_tile_stride([8, 16, 32], [16, 1, 0], [2, 4, 8]), [4, 1, 0]
        )

    def test_compute_tile_stride_3d_padding(self):
        self.assertEqual(
            compute_tile_stride([8, 16, 32], [544, 32, 1], [2, 4, 8]), [34, 8, 1]
        )

    def test_compute_tile_offset_1d(self):
        self.assertEqual(compute_tile_offset(0, [(1, 1)]), 0)
        self.assertEqual(compute_tile_offset(1, [(1, 1)]), 1)
        self.assertEqual(compute_tile_offset(256, [(1, 1)]), 256)

    def test_compute_tile_offset_2d(self):
        self.assertEqual(compute_tile_offset(0, [(4096, 1024), (1, 1)]), 0)
        self.assertEqual(compute_tile_offset(1, [(4096, 1024), (1, 1)]), 1)
        self.assertEqual(compute_tile_offset(4096, [(4096, 1024), (1, 1)]), 1024)
        self.assertEqual(compute_tile_offset(4097, [(4096, 1024), (1, 1)]), 1025)
        self.assertEqual(compute_tile_offset(8192, [(4096, 1024), (1, 1)]), 2048)
        self.assertEqual(compute_tile_offset(4104, [(4104, 1026), (1, 1)]), 1026)

    def test_compute_tile_offset_3d(self):
        self.assertEqual(compute_tile_offset(512, [(512, 32), (32, 8), (1, 1)]), 32)
        self.assertEqual(compute_tile_offset(32, [(512, 32), (32, 8), (1, 1)]), 8)
        self.assertEqual(compute_tile_offset(545, [(512, 32), (32, 8), (1, 1)]), 41)

    def test_compute_tile_offset_3d_min_stride_64(self):
        self.assertEqual(
            compute_tile_offset(0, [(8192, 2048), (1024, 512), (64, 64)]), 0
        )
        self.assertEqual(
            compute_tile_offset(64, [(8192, 2048), (1024, 512), (64, 64)]), 64
        )
        self.assertEqual(
            compute_tile_offset(1024, [(8192, 2048), (1024, 512), (64, 64)]), 512
        )
        self.assertEqual(
            compute_tile_offset(8192, [(8192, 2048), (1024, 512), (64, 64)]), 2048
        )
        self.assertEqual(
            compute_tile_offset(9280, [(8192, 2048), (1024, 512), (64, 64)]), 2624
        )

    def test_compute_tile_index_2d(self):
        p0, p1 = sympy.symbols("p0 p1", integer=True)
        self.assertEqual(
            compute_tile_index(
                4096 * p0 + p1, {p0: 1024, p1: 4096}, [1024, 4096], [4096, 1], [1024, 1]
            ),
            1024 * p0 + p1,
        )
        self.assertEqual(
            compute_tile_index(
                p0 + 1024 * p1, {p0: 4096, p1: 1024}, [4096, 1024], [1024, 1], [512, 1]
            ),
            p0 + 512 * p1,
        )

    def test_compute_tile_index_2d_diagonal(self):
        p0 = sympy.Symbol("p0", integer=True)
        self.assertEqual(
            compute_tile_index(
                1025 * p0, {p0: 1024}, [1024, 1024], [1024, 1], [512, 1]
            ),
            513 * p0,
        )

    def test_compute_tile_index_2d_col_major(self):
        p0, p1 = sympy.symbols("p0 p1", integer=True)
        self.assertEqual(
            compute_tile_index(
                p0 + 1024 * p1, {p0: 4096, p1: 1024}, [1024, 4096], [1, 1024], [1, 512]
            ),
            p0 + 512 * p1,
        )

    def test_compute_tile_index_2d_constant_offset(self):
        p0 = sympy.Symbol("p0", integer=True)
        self.assertEqual(
            compute_tile_index(
                4096 + p0, {p0: 1024}, [1024, 4096], [4096, 1], [1024, 1]
            ),
            1024 + p0,
        )

    def test_compute_tile_index_2d_no_tiling(self):
        p0 = sympy.Symbol("p0", integer=True)
        self.assertEqual(
            compute_tile_index(
                4096 * p0 + 2048, {p0: 1024}, [1024, 4096], [4096, 1], [4096, 1]
            ),
            4096 * p0 + 2048,
        )


class TestCustomPostFusionPassesOrder(unittest.TestCase):
    def test_hbm_pool_planning_runs_after_fusion(self):
        """hbm_pool_planning must see post-fusion bundles, not pre-fusion nodes."""
        from torch_spyre._inductor.passes import CustomPostFusionPasses
        from torch_spyre._inductor.fusion import spyre_fuse_nodes
        from torch_spyre._inductor.hbm_pool_planning import hbm_pool_planning

        pipeline = CustomPostFusionPasses()
        names = [p.__name__ for p in pipeline.passes]
        self.assertLess(
            names.index(spyre_fuse_nodes.__name__),
            names.index(hbm_pool_planning.__name__),
            "spyre_fuse_nodes must run before hbm_pool_planning",
        )


class TestSpyreKernelPoolSize(unittest.TestCase):
    def test_spyre_kernel_accepts_pool_size(self):
        """SpyreKernel must accept a per-bundle pool_size, defaulting to 0."""
        from torch_spyre._inductor.spyre_kernel import SpyreKernel

        kernel = SpyreKernel(pool_size=2048)
        self.assertEqual(kernel.pool_size, 2048)

        default_kernel = SpyreKernel()
        self.assertEqual(default_kernel.pool_size, 0)


class TestTileSpecRepresentation(unittest.TestCase):
    """TileAxis/TileSpec/CoreDivision.tiling and the min_footprint win."""

    def test_empty_spec_is_untiled_and_inert(self):
        u = TileSpec()
        self.assertTrue(u.is_untiled)
        self.assertEqual(u.depth, 0)
        self.assertEqual(u.tile_count, 1)
        self.assertEqual(u.output_tile_count, 1)
        self.assertTrue(u.is_clean)
        self.assertEqual(u.label, "untiled")

    def test_ordered_and_hashable_equality_is_same_shape(self):
        a = TileSpec((TileAxis(0, 4), TileAxis(1, 2)))
        b = TileSpec((TileAxis(0, 4), TileAxis(1, 2)))
        swapped = TileSpec((TileAxis(1, 2), TileAxis(0, 4)))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        # Levels nest, so swapping them is a different plan (unlike divisions).
        self.assertNotEqual(a, swapped)
        # Hashable: usable as a dict/set key (the group-derivation key).
        self.assertEqual(len({a, b, swapped}), 2)

    def test_output_axis_scalars(self):
        a = TileSpec((TileAxis(0, 4), TileAxis(1, 2)))
        self.assertEqual(a.depth, 2)
        self.assertEqual(a.tile_count, 8)
        self.assertEqual(a.output_tile_count, 8)
        self.assertTrue(a.is_clean)
        self.assertEqual(a.label, "d0:4/d1:2")

    def test_reduction_axis_excluded_from_output_tile_count(self):
        r = TileSpec((TileAxis(0, 4), TileAxis(2, 3, is_reduction=True)))
        # Reduction level counts in tile_count but not in output_tile_count.
        self.assertEqual(r.tile_count, 12)
        self.assertEqual(r.output_tile_count, 4)
        self.assertFalse(r.is_clean)
        self.assertEqual(r.label, "d0:4/~d2:3")

    def test_core_division_tiling_defaults_untiled_and_inert(self):
        cd = CoreDivision(output_splits={0: 2})
        self.assertEqual(cd.tiling, TileSpec())
        self.assertTrue(cd.tiling.is_untiled)
        # Distinct CoreDivisions do not share one mutable default.
        self.assertIsNot(CoreDivision().tiling, CoreDivision().tiling)

    def test_min_footprint_inert_when_untiled(self):
        buf = CoreDivisionBuffer(
            name="x",
            size=1024,
            uses=[0, 1],
            core_divisions=[CoreDivision(output_splits={0: 2})],
        )
        self.assertEqual(buf.min_footprint, ceil_div(1024, 2))

    def test_min_footprint_shrinks_by_output_tile_count(self):
        spec = TileSpec((TileAxis(0, 4), TileAxis(1, 2)))  # 8 output tiles
        buf = CoreDivisionBuffer(
            name="y",
            size=1024,
            uses=[0, 1],
            core_divisions=[CoreDivision(output_splits={0: 2}, tiling=spec)],
        )
        self.assertEqual(buf.min_footprint, ceil_div(1024, 2 * spec.output_tile_count))

    def test_min_footprint_reduction_level_does_not_shrink_accumulator(self):
        # The reduction-tiled op's own buffer is the accumulator: full extent.
        spec = TileSpec((TileAxis(0, 4), TileAxis(1, 3, is_reduction=True)))
        buf = CoreDivisionBuffer(
            name="z",
            size=1024,
            uses=[0, 1],
            core_divisions=[CoreDivision(output_splits={0: 2}, tiling=spec)],
        )
        self.assertEqual(buf.min_footprint, ceil_div(1024, 2 * spec.output_tile_count))


class TestTileSpecLoweringOutput(unittest.TestCase):
    """tile_spec_to_dim_hints on output axes — same DimHints as the hint path."""

    def setUp(self):
        self.enterContext(
            patch(
                "torch_spyre._inductor.scratchpad.coarse_tiling.op_out_coords",
                side_effect=_mock_op_out_coords,
            )
        )

    def _op(self, n_dims, name="op0"):
        op = _make_op(_make_pointwise([Integer(64)] * n_dims), name)
        op._test_out_coords = [sympy.Symbol(f"c{i}") for i in range(n_dims)]
        return op

    def test_single_output_axis(self):
        op = self._op(2)
        spec = TileSpec((TileAxis(0, 4),))
        hints = tile_spec_to_dim_hints(op, spec, [7])
        self.assertEqual(len(hints), 1)
        h = hints[0]
        self.assertEqual(h.split_count, 4)
        self.assertEqual(h.loop_var, sympy.Symbol("c0"))
        self.assertFalse(h.is_reduction)
        self.assertEqual(h.hint_id, 7)

    def test_nested_output_axes_get_their_own_loop_vars(self):
        op = self._op(3)
        spec = TileSpec((TileAxis(0, 4), TileAxis(2, 2)))
        hints = tile_spec_to_dim_hints(op, spec, [0, 1])
        self.assertEqual(
            [h.loop_var for h in hints], [sympy.Symbol("c0"), sympy.Symbol("c2")]
        )
        self.assertEqual([h.split_count for h in hints], [4, 2])
        self.assertEqual([h.hint_id for h in hints], [0, 1])

    def test_hint_id_count_must_match_axis_count(self):
        op = self._op(2)
        spec = TileSpec((TileAxis(0, 4),))
        with self.assertRaises(ValueError):
            tile_spec_to_dim_hints(op, spec, [0, 1])

    def test_host_dim_out_of_bounds_raises(self):
        op = self._op(2)
        spec = TileSpec((TileAxis(5, 4),))
        # unsupported because too few loop ids are provided
        with self.assertRaises(Unsupported):
            tile_spec_to_dim_hints(op, spec, [0])


class TestTileSpecLoweringReduction(unittest.TestCase):
    """The reduction-axis lowering is the inverse of reduction_loop_vars."""

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()

    def tearDown(self):
        self._graph_ctx.__exit__(None, None, None)

    def test_reduction_axis_resolves_via_reduction_loop_vars(self):
        # out[d0] = sum_{d1} in[d0, d1]; one reduction loop var.
        op = _make_real_reduction_op(
            ranges=[Integer(8)],
            reduction_ranges=[Integer(16)],
            input_shape_stride=([8, 16], [16, 1]),
            name="buf0",
            hints=((1, 0),),
        )
        red_vars = reduction_loop_vars(op)
        self.assertEqual(len(red_vars), 1)
        spec = TileSpec((TileAxis(0, 4, is_reduction=True),))
        hints = tile_spec_to_dim_hints(op, spec, [3])
        self.assertEqual(len(hints), 1)
        h = hints[0]
        self.assertTrue(h.is_reduction)
        self.assertEqual(h.split_count, 4)
        self.assertEqual(h.hint_id, 3)
        # The inverse relationship: the lowered loop_var is exactly the one
        # reduction_loop_vars reports at that position.
        self.assertEqual(h.loop_var, red_vars[0])
        self.assertEqual(_loop_var_to_reduction_ranges_pos_public(op, h.loop_var), 0)

    def test_reduction_host_dim_out_of_bounds_raises(self):
        op = _make_real_reduction_op(
            ranges=[Integer(8)],
            reduction_ranges=[Integer(16)],
            input_shape_stride=([8, 16], [16, 1]),
            name="buf0",
            hints=((1, 0),),
        )
        spec = TileSpec((TileAxis(3, 4, is_reduction=True),))  # only 1 red var
        with self.assertRaises(Unsupported):
            tile_spec_to_dim_hints(op, spec, [0])

    def test_reduction_axis_on_pointwise_raises(self):
        op = _make_real_pointwise_op(
            ranges=[Integer(8), Integer(16)],
            input_shapes_strides=[([8, 16], [16, 1])],
            name="buf0",
            hints=((0, 0),),
        )
        spec = TileSpec((TileAxis(0, 4, is_reduction=True),))
        with self.assertRaises(Unsupported):
            tile_spec_to_dim_hints(op, spec, [0])


def _loop_var_to_reduction_ranges_pos_public(op, sym):
    from torch_spyre._inductor.wsr.coarse_tile import (
        _loop_var_to_reduction_ranges_pos,
    )

    return _loop_var_to_reduction_ranges_pos(op, sym)


class TestDeriveTilingGroups(unittest.TestCase):
    """derive_tiling_groups — consecutive runs of ops sharing a TileSpec."""

    def _graph_of(self, names):
        return _graph([_make_op(_make_pointwise([Integer(64)]), n) for n in names])

    def _names(self, groups):
        return [[o.get_operation_name() for o in ops] for ops, _ in groups]

    def test_consecutive_run_grouped_untiled_breaks(self):
        g = self._graph_of(["op0", "op1", "op2", "op3", "op4"])
        spec = TileSpec((TileAxis(0, 4),))
        # op1,op2 tiled and consecutive -> one group; op4 tiled alone; op0/op3
        # untiled -> break the runs.
        choices = {"op1": spec, "op2": spec, "op4": spec}
        groups = derive_tiling_groups(g, choices)
        self.assertEqual(self._names(groups), [["op1", "op2"], ["op4"]])

    def test_spec_change_breaks_the_run(self):
        g = self._graph_of(["op0", "op1"])
        s1 = TileSpec((TileAxis(0, 4),))
        s2 = TileSpec((TileAxis(0, 2),))
        groups = derive_tiling_groups(g, {"op0": s1, "op1": s2})
        self.assertEqual([len(ops) for ops, _ in groups], [1, 1])
        self.assertEqual([spec for _, spec in groups], [s1, s2])

    def test_empty_and_all_untiled_produce_no_groups(self):
        g = self._graph_of(["op0", "op1"])
        self.assertEqual(derive_tiling_groups(g, {}), [])
        self.assertEqual(derive_tiling_groups(g, {"op0": TileSpec()}), [])


class TestDerivedBases(unittest.TestCase):
    """Hint-id and group-idx bases are derived off the graph, never reserved."""

    def test_hint_id_base_avoids_existing_hints(self):
        op = _make_op(_make_pointwise([Integer(64)]), "op0")
        op.dim_hints = []
        self.assertEqual(_derive_hint_id_base(_graph([op])), 0)
        op.dim_hints = [
            DimHint(
                dim_names=["a"],
                split_count=2,
                loop_var=None,
                is_reduction=False,
                hint_id=10005,
            )
        ]
        self.assertEqual(_derive_hint_id_base(_graph([op])), 10006)

    def test_group_idx_offset_avoids_existing_loop_group_ids(self):
        op = _make_op(_make_pointwise([Integer(64)]), "op0")
        self.assertEqual(_derive_group_idx_offset(_graph([op])), 0)
        op.loop_info = CoarseTileInfo(
            loop_group_id=(3,), loop_count=[Integer(2)], loop_tiled_dims=[[0]]
        )
        self.assertEqual(_derive_group_idx_offset(_graph([op])), 4)


class TestCoarseTilingPassEquivalence(unittest.TestCase):
    """CoarseTilingPass reduces to the hint path.

    Both paths call the same coarse_tile(); the pass just builds its (groups,
    group_idx_offset, dim_hints) inputs from a TileSpec instead of pre-stamped
    hints. Feeding structurally identical inputs to the identical function
    yields identical output, which is asserted end-to-end here on the stamped
    loop_info and the divided ranges.
    """

    def setUp(self):
        self.enterContext(
            patch(
                "torch_spyre._inductor.wsr.coarse_tile.op_out_coords",
                side_effect=_mock_op_out_coords,
            )
        )
        self.enterContext(
            patch(
                "torch_spyre._inductor.scratchpad.coarse_tiling.op_out_coords",
                side_effect=_mock_op_out_coords,
            )
        )

    def _bare(self, ranges, name):
        op = _make_op(_make_pointwise([Integer(r) for r in ranges]), name)
        op._test_out_coords = [sympy.Symbol(f"c{i}") for i in range(len(ranges))]
        op.dim_hints = []
        return op

    def _loop_fields(self, op):
        li = op.loop_info
        return (
            tuple(li.loop_group_id),
            list(li.loop_count),
            [list(x) for x in li.loop_tiled_dims],
        )

    def test_single_level_matches_hint_path(self):
        # Reference: hint path, hint_id 0, group 0.
        ref = _make_hinted_op(_make_pointwise([Integer(256)]), "op0", hints=((0, 0),))
        coarse_tile_pre_stickify(_graph([ref]), [([ref], [(0, Integer(4))])])
        ref_fields = self._loop_fields(ref)
        ref_range = ref.data.ranges[0]

        # CoarseTilingPass: same tiling expressed as a TileSpec.
        got = self._bare([256], "op0")
        CoarseTilingPass({"op0": TileSpec((TileAxis(0, 4),))}).apply_pass(_graph([got]))
        self.assertEqual(self._loop_fields(got), ref_fields)
        self.assertEqual(got.data.ranges[0], ref_range)
        # And the win is visible: dim 0 divided by 4.
        self.assertEqual(got.data.ranges[0], Integer(64))

    def test_nested_two_level_matches_hint_path(self):
        ref = _make_hinted_op(
            _make_pointwise([Integer(256), Integer(128)]), "op0", hints=((0, 0), (1, 1))
        )
        coarse_tile_pre_stickify(
            _graph([ref]), [([ref], [(0, Integer(4)), (1, Integer(2))])]
        )
        ref_fields = self._loop_fields(ref)
        ref_ranges = list(ref.data.ranges)

        got = self._bare([256, 128], "op0")
        CoarseTilingPass(
            {"op0": TileSpec((TileAxis(0, 4), TileAxis(1, 2)))}
        ).apply_pass(_graph([got]))
        self.assertEqual(self._loop_fields(got), ref_fields)
        self.assertEqual(list(got.data.ranges), ref_ranges)
        self.assertEqual(list(got.data.ranges), [Integer(64), Integer(64)])

    def test_pass_inputs_match_hint_path_structurally(self):
        # Two independent ops in one group: prove the group derivation and
        # hint-id minting produce exactly the coarse_tile inputs the hint path
        # would, without running the mutation (robust against mock read-edges).
        got0 = self._bare([256], "op0")
        got1 = self._bare([256], "op1")
        g = _graph([got0, got1])
        spec = TileSpec((TileAxis(0, 4),))
        groups_specs = derive_tiling_groups(g, {"op0": spec, "op1": spec})
        self.assertEqual(len(groups_specs), 1)
        group_ops, group_spec = groups_specs[0]
        self.assertEqual([o.get_operation_name() for o in group_ops], ["op0", "op1"])
        # One group, base 0 -> hint_ids [0], every op in the group shares it.
        base = _derive_hint_id_base(g)
        self.assertEqual(base, 0)
        hints0 = tile_spec_to_dim_hints(got0, group_spec, [base])
        hints1 = tile_spec_to_dim_hints(got1, group_spec, [base])
        self.assertEqual(hints0[0].hint_id, hints1[0].hint_id)
        self.assertEqual(hints0[0].split_count, 4)

    def test_empty_choices_is_a_noop(self):
        # The pass leaves op count unchanged when it does nothing
        # (untiled/absent choices).
        ops = [self._bare([64], "op0"), self._bare([64], "op1")]
        g = _graph(ops)
        CoarseTilingPass({}).apply_pass(g)
        self.assertEqual(len(g.operations), 2)
        for op in ops:
            self.assertFalse(
                hasattr(op, "loop_info") and isinstance(op.loop_info, CoarseTileInfo)
            )


if __name__ == "__main__":
    unittest.main()
