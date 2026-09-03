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

"""IR-level pass to pad y's K (row) dimension to a stick boundary for
BATCH_MATMUL_OP operations.  Runs in CustomPreSchedulingPasses before
stickification, while every buffer still carries a plain host FixedLayout.

Padding before stickification means the padded buffer is laid out and, if
needed, restickified like any user-written F.pad: propagate_spyre_tensor_layouts
chooses its device layout and finalize_layouts plans a restickify when the
matmul reads it through a view whose stick dim does not match (a transposed
nn.Linear weight, issue #4208).  Padding after insert_restickify would instead
have to grow a restickify output, whose device layout and index expressions
cannot be reconciled with a larger host extent.

Only y is padded; x is left untouched.

For y, the following IR sequence is emitted:
  1. ComputedBuffer - output buffer allocation (FixedLayout)
  2. SpyreConstantFallback - fill constant (FixedLayout)
  3. ComputedBuffer - fill padding region (MutationLayoutSHOULDREMOVE)
  4. ComputedBuffer - copy input data (MutationLayoutSHOULDREMOVE)

y's padded buffer is built at the full K_padded host size by lower_pad_sequence.
reduction_ranges stays at K; the K→K_padded extension happens at SDSC codegen
time: _extend_matmul_k_to_padded in superdsc.py rounds sdsc_iteration_space[K]
up to the stick boundary before _create_sdsc_tensors runs.

x is left physically untouched.  The hardware masks within-stick elements of x
beyond the true K to zero, so extending the SDSC iteration to K_padded does not
introduce numerical error from x.

Deduplication of identical constants across multiple pad calls happens later
at the IR level via dedup_and_promote_constants.

x and y are identified via identify_matmul_inputs() using the BatchMatmul
generated_dim definition: y is the input whose index contains a symbol
present in the output but absent from x (N).  This handles M==K==N and
M=1 (decode phase) correctly.
"""

from typing import Optional

import sympy
import torch
from sympy import Expr
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    FixedLayout,
    Operation,
    Pointwise,
    Reduction,
    TensorBox,
)
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.virtualized import V

from .constants import BATCH_MATMUL_FP8_OP, BATCH_MATMUL_OP
from .errors import Unsupported
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .pass_utils import (
    concretize_expr,
    concretize_index,
    find_reduction_var,
    host_coordinates,
    identify_matmul_inputs,
    is_restickify_coords,
    lower_pad_sequence,
    redirect_computed_buffer_reads,
    replace_computed_buffer_body,
)
from .views import compute_coordinates
from torch_spyre._C import SpyreTensorLayout, get_elem_in_stick

logger = get_inductor_logger("padding")


def compute_padding(cur_size: int, dtype: torch.dtype) -> int:
    stick_size = get_elem_in_stick(dtype)
    pad = (stick_size - (cur_size % stick_size)) % stick_size
    return pad


def round_up_to_stick(cur_size: int, dtype: torch.dtype) -> int:
    """Round ``cur_size`` up to the next whole stick for ``dtype`` (no-op when
    already a stick multiple)."""
    return cur_size + compute_padding(cur_size, dtype)


def _patch_env(graph_lowering) -> None:
    """Add view nodes (ReinterpretView) to env from name_to_users."""
    env: dict = {}
    for tbs in graph_lowering.name_to_users.values():
        for tb in tbs:
            if not tb.data.origins:
                continue
            tb_fx_node = list(tb.data.origins)[0]
            env[tb_fx_node] = tb
    graph_lowering.env.update(env)


def _move_ops_before(
    operations: list[Operation], new_ops: list[Operation], anchor: Operation
) -> None:
    """Relocate *new_ops* to sit immediately before *anchor* in *operations*.

    ``run_node`` appends each newly lowered op at the end of ``operations``;
    this helper moves them to just before the op that consumes them so
    topological order is preserved.
    """
    for o in new_ops:
        operations.remove(o)
    idx = operations.index(anchor)
    for i, o in enumerate(new_ops):
        operations.insert(idx + i, o)


def _find_arg_fx_node(arg_name: str) -> Optional[torch.fx.Node]:
    """Return the FX node whose lowered TensorBox has the given buffer name.

    Buffer names are unique, but a single buffer can be reached through
    multiple FX nodes that present it at different sizes.  For example,
    mm_to_bmm_pass inserts an unsqueeze/reshape so the matmul inner_fn
    indexes x as 3D [1, M, K] even though the underlying buffer is 2D
    [M, K].  Both FX nodes lower to a TensorBox whose get_name() returns
    the same buffer name, but with different get_size() results.

    Returns the first candidate (the base buffer, with no view applied), or
    None if no candidate exists -- e.g. a coarse_tile read-copy buffer (see
    coarse_tile.py's _insert_one_read_copy), which is synthesized purely at
    the IR level after FX lowering completed and so has no FX-graph
    counterpart at all.
    """
    graph_lowering = V.graph
    _patch_env(graph_lowering)
    candidates = [
        fx_node
        for fx_node, tb in graph_lowering.env.items()
        if isinstance(fx_node, torch.fx.Node)
        and isinstance(tb, TensorBox)
        and tb.get_name() == arg_name
    ]
    return candidates[0] if candidates else None


class _PaddedLoadHandler(WrapperHandler):
    """Redirect every ``load`` of ``orig_name`` to ``padded_name``, re-striding
    the index.

    The matmul's inner_fn loads y through whatever view the graph applied (a
    plain ``[K, N]`` buffer, a transposed ``nn.Linear`` weight, an unsqueezed
    batch dim, ...), so the flat index it produces is expressed against y's own
    host strides.  The padded buffer holds the same logical coordinates at the
    strides of the padded shape.  Each load's index is therefore decomposed
    into host coordinates against y's layout and re-composed with the padded
    buffer's indexer.

    This wraps the live index (see CLAUDE.md "Compiler Pass Conventions")
    rather than re-deriving y's coordinates from the matmul's dim order, so
    any view stays correct by construction.
    """

    def __init__(
        self,
        inner,
        orig_name: str,
        orig_size: list[int],
        orig_stride: list[int],
        padded_name: str,
        padded_indexer,
        var_ranges: dict,
    ):
        super().__init__(inner)
        self._orig_name = orig_name
        self._orig_size = orig_size
        self._orig_stride = orig_stride
        self._padded_name = padded_name
        self._padded_indexer = padded_indexer
        self._var_ranges = var_ranges

    def load(self, name, index):
        if name != self._orig_name:
            return super().load(name, index)
        index = concretize_index(sympy.sympify(index), set(self._var_ranges))
        coords = compute_coordinates(
            self._orig_size, self._orig_stride, self._var_ranges, index
        )
        return super().load(self._padded_name, self._padded_indexer(coords))


def _rebuild_matmul(
    op: ComputedBuffer,
    y_buf: Buffer,
    y_padded_buf: Buffer,
    operations: list[Operation],
) -> ComputedBuffer:
    """Rebuild the matmul ComputedBuffer so y's loader reads from the padded buffer.

    Wraps the original inner_fn with ``_PaddedLoadHandler`` so x's loading (and
    any view applied to y) is preserved unchanged; only the buffer y's loads
    resolve to, and the strides of their index, change.  reduction_ranges stays
    at K; the K→K_padded extension happens at SDSC codegen time via
    _extend_matmul_k_to_padded in superdsc.py.

    y's host dims are [.., K, N] in the common case, but need not be: a
    transposed ``nn.Linear`` weight is [N, K] (issue #4208), and a coarse-tile
    read-copy buffer for a loop-invariant y can be transposed and column-major
    (see test_bmm_read_copy_y_unaligned_k_pads).  Re-striding the live load
    index covers all of these without knowing which position K occupies.
    """
    reduction = op.data
    assert isinstance(reduction, Reduction)

    orig_inner_fn = reduction.inner_fn
    y_name = y_buf.get_name()
    y_layout = y_buf.get_layout()
    y_size = [concretize_expr(s) for s in y_layout.size]
    y_stride = [concretize_expr(s) for s in y_layout.stride]
    padded_name = y_padded_buf.get_name()
    padded_indexer = y_padded_buf.get_layout().make_indexer()

    def new_inner_fn(
        index,
        reduction_index,
        _orig_inner_fn=orig_inner_fn,
        _reduction=reduction,
    ):
        # Loop-variable ranges are needed to decompose a flat index into
        # coordinates.  Read them from the reduction at call time so later
        # passes that re-range the op (coarse tiling) stay consistent.
        var_ranges = {
            sym: rng
            for sym, rng in [
                *zip(index, _reduction.ranges),
                *zip(reduction_index, _reduction.reduction_ranges),
            ]
            if isinstance(sym, sympy.Symbol)
        }
        handler = _PaddedLoadHandler(
            V.ops,
            y_name,
            y_size,
            y_stride,
            padded_name,
            padded_indexer,
            var_ranges,
        )
        with V.set_ops_handler(handler):
            return _orig_inner_fn(index, reduction_index)

    object.__setattr__(reduction, "inner_fn", new_inner_fn)
    # reduction_ranges stays at K; no extension here.

    return replace_computed_buffer_body(
        op,
        reduction,
        operations,
        pass_name="insert_bmm_padding",
        reason="redirect matmul to padded input",
    )


def insert_bmm_padding(graph: GraphLowering) -> None:
    """
    Pad y's K (row) dimension for each BATCH_MATMUL_OP to a stick boundary.

    Mutates ``operations`` in place.  New buffers for y are inserted immediately
    before the matmul that consumes them to preserve topological order.

    x is left entirely untouched.  y's padded buffer is built at K_padded host
    size by lower_pad_sequence; reduction_ranges stays at K so the IR iteration
    space is unchanged.  The K→K_padded widening happens at SDSC codegen time.

    x and y are identified via identify_matmul_inputs() using the BatchMatmul
    generated_dim definition: y is the input whose index contains a symbol
    present in the output but absent from x (N).  This handles M==K==N and
    M=1 (decode phase) correctly.

    Deduplication of identical constants across multiple pad calls happens later
    at the IR level via dedup_and_promote_constants.
    """
    operations = graph.operations
    for op in list(operations):
        if not isinstance(op, ComputedBuffer):
            continue
        reduction = op.data
        if not isinstance(reduction, Reduction):
            continue
        if reduction.reduction_type not in [BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP]:
            continue

        rw = op.get_read_writes()
        reads = [r for r in rw.reads if hasattr(r, "name")]
        if len(reads) != 2:  # noqa: PLR2004
            continue

        # Skip aligned-K matmuls early before any x/y identification.
        # Aligned-K matmuls need no padding regardless of input layout, and
        # skipping here avoids a spurious warning for e.g. decode-phase SDPA
        # attention where constant-folded dimensions cause identify_matmul_inputs
        # to fail.
        k_val = concretize_expr(reduction.reduction_ranges[0])
        first_buf = next(
            (graph.get_buffer(d.name) for d in reads if graph.get_buffer(d.name)),
            None,
        )
        assert first_buf is not None, (
            f"insert_bmm_padding: no input buffer found for matmul {op.get_name()}"
        )
        dtype = first_buf.get_dtype()
        if compute_padding(k_val, dtype) == 0:
            continue

        write_dep = next(iter(rw.writes))
        x_dep, y_dep = identify_matmul_inputs(reads, write_dep)

        reduction_var = find_reduction_var((x_dep,), write_dep)

        # y's K host dim: the dim whose host coordinate contains reduction_var.
        y_buf_tmp = graph.get_buffer(y_dep.name)
        y_host_k_dim: int | None = None
        if y_buf_tmp is not None and isinstance(y_buf_tmp.get_layout(), FixedLayout):
            y_h_coords = host_coordinates(y_buf_tmp.get_layout(), y_dep, None)
            y_host_k_dim = next(
                (
                    i
                    for i, c in enumerate(y_h_coords)
                    if reduction_var in c.free_symbols
                ),
                None,
            )

        x_name = x_dep.name
        y_name = y_dep.name
        x_buf = graph.get_buffer(x_name)
        y_buf = graph.get_buffer(y_name)
        if x_buf is None or y_buf is None:
            continue

        device = x_buf.get_device()
        pad = compute_padding(k_val, dtype)

        k_padded = k_val + pad

        logger.debug(
            "insert_bmm_padding: padding %s K=%d -> K=%d (pad=%d)",
            op.get_name(),
            k_val,
            k_padded,
            pad,
        )

        # The FX node for the matmul is used as the insertion anchor so padding
        # nodes are placed immediately before the matmul in the FX graph,
        # minimising their live range.
        matmul_fx_node = next(iter(op.origins))

        # --- Pad y only ---
        # y's K dimension is y's row (mb) dimension.  Padding it to K_padded
        # ensures rows K..K_padded-1 of y are zero-filled so the hardware
        # accumulates no contribution from those rows.
        # lower_pad_sequence builds the padded buffer at K_padded host size;
        # reduction_ranges is NOT changed.  superdsc._extend_matmul_k_to_padded
        # widens sdsc_iteration_space[K] to K_padded at SDSC codegen time.
        y_size = [concretize_expr(s) for s in y_buf.get_size()]
        if y_host_k_dim is None:
            y_k_dim = len(y_size) - 2
        else:
            y_k_dim = y_host_k_dim
        y_padded_size = list(y_size)
        y_padded_size[y_k_dim] = k_padded
        y_fx_node = _find_arg_fx_node(y_name)

        # No orig_stl: this pass runs before stickification, so the new ops keep
        # host FixedLayouts and propagate_spyre_tensor_layouts lays them out.
        y_padded_buf, y_new_ops = lower_pad_sequence(
            y_fx_node,
            padded_size=y_padded_size,
            device=device,
            dtype=dtype,
            dim=y_k_dim,
            insert_before=matmul_fx_node,
            arg_buf=y_buf if y_fx_node is None else None,
        )

        # --- Relocate new ops before the matmul ---
        # run_node appended them at the end of operations; move before op.
        _move_ops_before(operations, y_new_ops, op)

        # When coarse-tiling runs pre-stickification, the consuming matmul
        # already carries loop_info.  The padding ops must inherit it so they
        # remain contiguous in the loop group for build_loop_scheduler_nodes.
        if hasattr(op, "loop_info"):
            for new_op in y_new_ops:
                if isinstance(new_op, ComputedBuffer):
                    new_op.loop_info = op.loop_info  # type: ignore[attr-defined]

        # --- Rebuild matmul inner_fn to load y from the padded buffer ---
        # x is left entirely untouched: the original inner_fn's x loader is
        # preserved as-is.  Only y's loads are redirected to the padded buffer.
        _rebuild_matmul(op, y_buf, y_padded_buf, operations)


# --------------------------------------------------------------------------- #
# insert_restickify_padding                                                   #
# --------------------------------------------------------------------------- #


def _device_coords(stl: SpyreTensorLayout, dep) -> list[Expr]:
    """Return device-space coordinate expressions for ``dep`` against ``stl``."""
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    return compute_coordinates(stl.device_size, stl.stride_map, dep.ranges, index)


def _write_dep(op):
    """Return op's write dependency on the buffer it produces."""
    writes = [d for d in op.get_read_writes().writes if hasattr(d, "name")]
    assert len(writes) == 1, (
        f"_write_dep: {op.get_name()} has {len(writes)} named writes, want exactly 1"
    )
    return writes[0]


def _restickify_input(op, graph: GraphLowering):
    """Return ``(in_dep, in_buf, in_layout)`` for a restickify's single input, or
    ``(None, None, None)`` if ``op`` cannot be one: it must have exactly one
    named read whose buffer has a FixedTiledLayout.
    """
    # Callers that already know op is a confirmed restickify can assume this
    # succeeds.
    reads = [r for r in op.get_read_writes().reads if hasattr(r, "name")]
    if len(reads) != 1:
        return None, None, None
    in_dep = reads[0]
    in_buf = graph.get_buffer(in_dep.name)
    if in_buf is None:
        return None, None, None
    in_layout = in_buf.get_layout()
    if not isinstance(in_layout, FixedTiledLayout):
        return None, None, None
    return in_dep, in_buf, in_layout


def _host_dim_carrying_sym(host_coords: list[Expr], sym) -> int | None:
    """Return the outermost host dim whose coordinate carries ``sym``, or None."""
    # A symbol whose range spans more than one tile can appear split across
    # several coordinates (e.g. v // 64 in one, v % 64 in another); the
    # outermost (lowest-index) one is the governing dim.
    for dim, coord in enumerate(host_coords):
        if sym in coord.free_symbols:
            return dim
    return None


def _device_dim_carrying_sym(stl: SpyreTensorLayout, write_dep, sym) -> int | None:
    """Return the outermost non-within-stick ``device_size`` dim whose device
    coordinate carries ``sym`` (the free symbol of some host dim), or None.
    """
    # Two host dims can share a dim size, so only the symbol unambiguously
    # identifies the dim.  A symbol whose range spans more than one tile can
    # appear split across several coordinates (e.g. v // 64 in one, v % 64 in
    # another); the outermost (lowest-index) one is the governing dim.
    device_coords = _device_coords(stl, write_dep)
    for dim in range(len(device_coords) - 1):
        if sym in device_coords[dim].free_symbols:
            return dim
    return None


def _stick_symbol(stl: SpyreTensorLayout, dep) -> object | None:
    """Return dep's within-stick device coordinate's free symbol against stl,
    or None if it has none (e.g. a symbol-free size-1 stick, or a scalar /
    zero-dim layout with no coords at all).
    """
    device_coords = _device_coords(stl, dep)
    if not device_coords:
        return None
    # A coefficient or offset on the coordinate doesn't add free symbols, so
    # e.g. 2*(Mod(var, 32)) + 1 and Mod(var, 32) both resolve to {var}.
    syms = device_coords[-1].free_symbols
    if not syms:
        return None
    assert len(syms) == 1, (
        f"_stick_symbol: within-stick coordinate {device_coords[-1]} carries "
        f"{len(syms)} free symbols, want exactly 1"
    )
    return next(iter(syms))


def is_restickify_op(op: Operation, graph: GraphLowering) -> bool:
    """Return whether ``op`` is a restickify: a single-input pointwise copy
    between two FixedTiledLayouts that lands a *different* host dim within the
    stick.
    """
    if not isinstance(op, ComputedBuffer):
        return False
    out_layout = op.get_layout()
    if not isinstance(out_layout, FixedTiledLayout):
        return False
    if not isinstance(op.data, Pointwise):
        return False

    in_dep, _in_buf, in_layout = _restickify_input(op, graph)
    if in_dep is None:
        return False

    in_coords = _device_coords(in_layout.device_layout, in_dep)
    out_coords = _device_coords(out_layout.device_layout, _write_dep(op))
    # A scalar / zero-dim layout has no coordinates, so it can't be a restickify.
    if not in_coords or not out_coords:
        return False
    # is_restickify_coords is the single source of truth for this check, shared
    # with codegen's store side, so the two can't disagree on which ops restickify.
    return is_restickify_coords(in_coords, out_coords)


def _pad_device_dim(
    layout: FixedTiledLayout,
    device_dim: int,
    new_dim_size,
) -> FixedTiledLayout:
    """Pad the device dim ``device_dim`` up to ``new_dim_size``.

    This handles a restickify whose transposed dim survives as a device dim larger
    than one; when that dim has been elided to size one, use ``_pad_elided_dim``
    instead.

    The result is a copy of ``layout`` with only that dim's size grown.  The host
    size is left alone, and since ``stride_map`` stores host strides rather than
    sizes, it does not need to change either.
    """
    stl = layout.device_layout
    new_device_size = list(stl.device_size)
    new_device_size[device_dim] = new_dim_size
    padded_stl = SpyreTensorLayout(
        new_device_size, list(stl.stride_map), stl.device_dtype, stl.element_arrangement
    )
    host_size = [concretize_expr(s) for s in layout.size]
    host_stride = [concretize_expr(s) for s in layout.stride]
    return FixedTiledLayout(
        layout.device, layout.dtype, host_size, host_stride, padded_stl
    )


def _pad_elided_dim(buf: ComputedBuffer) -> None:
    """Pad ``buf``'s allocation by prepending an outermost size-64 gap dim, for a
    restickify whose transposed dim was elided to a size-1 device dim.

    The transposed dim is a non-stick dim here -- the new stick on the input side,
    the old stick on the output side -- that pairs with the stick on the other
    operand.  When it has host size 1, Inductor elides it, so the STL carries it as
    a size-1 dim with ``stride_map == -1``, and its allocation holds one plane, but
    the paired stick needs 64.

    Prepending a size-64 gap dim makes the allocation cover all 64 planes.  This
    only bumps the dim size; a later pass, ``_restickify_restore_elided_dim``,
    creates the shared iteration variable that pairs this dim with the stick on the
    other operand.
    """
    layout = buf.get_layout()
    assert isinstance(layout, FixedTiledLayout)
    stl = layout.device_layout
    stick = stl.elems_per_stick()
    grown_stl = SpyreTensorLayout(
        [stick, *stl.device_size],
        [-1, *stl.stride_map],
        stl.device_dtype,
        stl.element_arrangement,
    )
    host_size = [concretize_expr(s) for s in layout.size]
    host_stride = [concretize_expr(s) for s in layout.stride]
    buf.layout = FixedTiledLayout(
        layout.device, layout.dtype, host_size, host_stride, grown_stl
    )


def _pad_restickify_output(op: Operation, graph: GraphLowering) -> None:
    """Pad the output dim carrying the input's old stick to a stick boundary, so
    the second+ stick block and every batch plane land at the correct offset.

    The old stick is a non-stick dim on the output.  Bump its device dim to a stick
    multiple; only the device layout changes, and ``_create_sdsc_tensors``'s backGap
    path covers the tail rows, which are never read back.  Does nothing when its
    device dim is already stick-aligned.

    When the old stick was elided to a size-1 device dim, there is no device dim to
    bump; delegate to ``_pad_elided_dim`` instead.

    Raises ``Unsupported`` for a sliced output: it writes only part of the
    old-stick device dim of a base tensor whose layout we do not own, so there is
    no room to widen the dim (see the TODO below for how it could be supported).
    """
    assert isinstance(op, ComputedBuffer)
    in_dep, _in_buf, in_layout = _restickify_input(op, graph)
    assert in_dep is not None  # op is a confirmed restickify
    # The old stick is the INPUT's own stick symbol (None when it collapsed to a
    # size-1 device dim).
    old_sym = _stick_symbol(in_layout.device_layout, in_dep)
    out_layout = op.get_layout()
    write_dep = _write_dep(op)
    stl = out_layout.device_layout

    if old_sym is None:
        _pad_elided_dim(op)
        logger.debug(
            "insert_restickify_padding: grew size-1 output %s allocation",
            op.get_name(),
        )
        return

    device_dim = _device_dim_carrying_sym(stl, write_dep, old_sym)
    assert device_dim is not None
    unpadded_dim_size = stl.device_size[device_dim]

    # A sliced output writes only part of the old-stick device dim: the rows it
    # writes are the interior of a base tensor whose layout we do not own, and the
    # rows around them belong to other slice positions.  We cannot pad such an output
    # -- growing the dim would overwrite a neighbour's rows.
    #
    # TODO: support this by restickifying into a fresh temp buffer sized to a stick
    # multiple, then cloning that buffer into the slice.
    host_dim_size = concretize_expr(write_dep.ranges[old_sym])
    if host_dim_size < unpadded_dim_size:
        raise Unsupported(
            f"insert_restickify_padding: sliced output on {op.get_name()} "
            f"(written size {host_dim_size} < device dim size {unpadded_dim_size}) "
            f"cannot be padded in place"
        )

    # Already a stick multiple: stick blocks land aligned, no padding needed.
    pad = compute_padding(unpadded_dim_size, out_layout.dtype)
    if pad == 0:
        return

    padded_dim_size = unpadded_dim_size + pad
    op.layout = _pad_device_dim(out_layout, device_dim, padded_dim_size)

    logger.debug(
        "insert_restickify_padding: padded output %s device dim %d %d -> %d",
        op.get_name(),
        device_dim,
        unpadded_dim_size,
        padded_dim_size,
    )


def _assert_input_paddable(op: ComputedBuffer, in_dep, in_layout) -> None:
    """Raise ``Unsupported`` for restickify inputs outside what the stick-boundary
    bump supports, classifying each input dim's read by its coordinate.

    Outside the supported set:

    - **Strided** read of any dim (coord ``k*var``, k not in {0, 1}: step > 1 or
      reversed), e.g. ``x[::2].transpose(1, 2).clone()``.

    TODO: the strided case is liftable by double-restickifying -- a re-base copy
    that reads the strided source into a fresh stick-aligned buffer, then
    restickifies that.  With that in place, these expressions take that path and
    do not reach this guard.
    """
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    for i, coord in enumerate(in_host_coords):
        syms = coord.free_symbols
        if not syms:  # degenerate size-1 host dim, nothing to slice
            continue
        assert len(syms) == 1, (
            f"insert_restickify_padding: host dim {i} of {op.get_name()} "
            f"(coord {coord}) carries multiple free symbols -- an interleaved "
            f"index this pass's per-dim strided/sliced classification cannot "
            f"read; a restickify input must not reach this shape"
        )
        sym = next(iter(syms))
        # Strided (k not in {0, 1}), on any dim: codegen carries only a contiguous
        # tail.  A broadcast (coeff 0) is read from the device layout, not this
        # coefficient, so it is not a stride and is left to flow through.
        if concretize_expr(coord.coeff(sym)) not in (0, 1):
            raise Unsupported(
                f"insert_restickify_padding: strided input on host dim "
                f"{i} of {op.get_name()} (coord {coord}) is not supported"
            )


def lower_identity_clone(
    arg_fx_node: torch.fx.Node,
    host_size: list[int],
    host_stride: list[int],
    device: torch.device,
    dtype: torch.dtype,
    orig_stl: SpyreTensorLayout,
    insert_before: torch.fx.Node,
) -> tuple[ComputedBuffer, list[Operation]]:
    """Lower an identity ``aten.clone`` of ``arg_fx_node`` as a layout-preserving
    copy: allocated at ``host_size`` / ``host_stride`` (not contiguous) with a
    ``SpyreTensorLayout`` copied from ``orig_stl``, so it replicates the graph
    input's layout. The caller then bumps ``device_size`` on the stick dim.

    Returns ``(clone_buf, new_ops)`` -- the single new ComputedBuffer and the
    (length-1) list of new IR operations.
    """
    graph_lowering = V.graph
    fx_graph = graph_lowering.graph

    ops_before = len(graph_lowering.operations)

    with fx_graph.inserting_before(insert_before):
        clone_fx = fx_graph.create_node(
            "call_function", torch.ops.aten.clone.default, args=(arg_fx_node,)
        )
        clone_fx.meta["val"] = torch.empty_strided(
            host_size, host_stride, dtype=dtype, device=device
        )

    clone_tb = graph_lowering.run_node(clone_fx)
    graph_lowering.env[clone_fx] = clone_tb

    # aten.clone lowers to an identity Pointwise; force realization so it becomes
    # a named ComputedBuffer rather than inlining into the consumer.
    clone_tb.data.realize()
    new_ops = graph_lowering.operations[ops_before:]
    assert len(new_ops) == 1 and isinstance(new_ops[0], ComputedBuffer), (
        f"lower_identity_clone: expected exactly one ComputedBuffer, got "
        f"{[type(o).__name__ for o in new_ops]}"
    )
    clone_buf = new_ops[0]

    host_layout = clone_buf.layout
    clone_stl = SpyreTensorLayout(
        list(orig_stl.device_size),
        list(orig_stl.stride_map),
        orig_stl.device_dtype,
        orig_stl.element_arrangement,
    )
    # insert_restickify_padding runs after propagate_spyre_tensor_layouts, so
    # run_node left this op with a FlexibleLayout; assign a FixedTiledLayout
    # here so the clone carries a real device layout.
    clone_buf.layout = FixedTiledLayout(
        host_layout.device,
        host_layout.dtype,
        host_layout.size,
        host_stride,
        clone_stl,
    )

    assert clone_buf.origins, "lower_identity_clone: clone buffer has no origins"

    return clone_buf, new_ops


def _pad_restickify_input(op: Operation, graph: GraphLowering) -> None:
    """Pad a restickify input's non-stick dim to cover codegen's stick-boundary
    read window.

    Decide whether and how to pad first, then apply it.  In general we pad the
    producer's output buffer in place; a graph input has no producer to pad, so
    we clone it and pad the clone.  Deciding before cloning keeps the graph
    unmutated for an input we end up skipping.  The padding differs by whether
    the new-stick dim collapsed to size 1:

    - **Size-1 new-stick dim:** the input carries the restickify's INTACT operand
      (``_restickify_restore_elided_dim`` restores the stick on it), so codegen
      sweeps a full 64-plane stick while the input under-allocates 64x.  Prepend
      an outermost size-64 gap dim (``_pad_elided_dim``); restore reuses that dim
      as the restored stick.
    - **Size>1 new-stick dim:** codegen reads a whole-stick window from the slice
      base, so the allocation must extend past a narrowing or end-of-dim slice.
      Bump the device dim carrying the new-stick symbol so the allocation covers
      that window.

    Does nothing when the read window already fits the allocation, or the input
    has no device.
    """
    assert isinstance(op, ComputedBuffer)
    in_dep, in_buf, in_layout = _restickify_input(op, graph)
    assert in_dep is not None  # op is a confirmed restickify

    # --- Gates: decide whether (and how) to pad BEFORE mutating the graph. ---
    out_stick_sym = _stick_symbol(op.get_layout().device_layout, _write_dep(op))
    size1 = out_stick_sym is None
    new_stick_dim = None
    padded_dim_size = None
    if not size1:
        _assert_input_paddable(op, in_dep, in_layout)
        in_host_coords = host_coordinates(in_layout, in_dep, None)
        new_stick_dim = _host_dim_carrying_sym(in_host_coords, out_stick_sym)
        assert new_stick_dim is not None, (
            f"restickify padding: no input host dim carries new-stick symbol "
            f"{out_stick_sym} for {op.get_name()} (layout invariant broken)"
        )
        dtype = in_layout.dtype
        coord = in_host_coords[new_stick_dim]
        syms = coord.free_symbols
        assert len(syms) == 1
        sym = next(iter(syms))
        slice_offset = concretize_expr(coord.as_coeff_Add()[0])
        slice_size = concretize_expr(in_dep.ranges[sym])
        device_dim = _device_dim_carrying_sym(in_layout.device_layout, in_dep, sym)
        assert device_dim is not None
        dim_size = concretize_expr(in_layout.device_layout.device_size[device_dim])
        padded_dim_size = slice_offset + round_up_to_stick(slice_size, dtype)
        if padded_dim_size <= dim_size:
            return

    # --- Get a buffer we can pad: the producer's output, or a graph-input clone. ---
    # Run AFTER the gate: cloning mutates the graph (insert + redirect), so
    # cloning for an input we then skip would strand a redundant clone.
    if not isinstance(in_buf, ComputedBuffer):
        # A graph input has no producer output to pad: insert an identity clone
        # ahead of the restickify, move it into place, redirect the read to it,
        # then pad the clone.
        device = in_buf.get_device()
        if device is None:
            return
        in_fx_node = _find_arg_fx_node(in_dep.name)
        if in_fx_node is None:
            raise RuntimeError(f"no FX node found for buffer {in_dep.name!r}")
        clone_buf, new_ops = lower_identity_clone(
            in_fx_node,
            host_size=[concretize_expr(s) for s in in_layout.size],
            host_stride=[concretize_expr(s) for s in in_layout.stride],
            device=device,
            dtype=in_layout.dtype,
            orig_stl=in_layout.device_layout,
            insert_before=next(iter(op.origins)),
        )
        _move_ops_before(graph.operations, new_ops, op)
        redirect_computed_buffer_reads(
            op,
            {in_dep.name: clone_buf.get_name()},
            graph.operations,
            pass_name="insert_restickify_padding",
            reason="redirect consumer to padded input",
        )
        in_buf = clone_buf

        logger.debug(
            "insert_restickify_padding: inserted clone %s for input of %s",
            clone_buf.get_name(),
            op.get_name(),
        )

    # --- Pad the input. ---
    if size1:
        # Prepend an outermost size-64 gap dim so the elided size-1 dim's
        # allocation covers the full stick the restickify sweeps.
        _pad_elided_dim(in_buf)
    else:
        assert new_stick_dim is not None  # set in the size>1 gate above
        assert padded_dim_size is not None  # computed alongside it
        # Bump device_size on the dim carrying new_stick_dim so the read window
        # lands inside in_buf's own allocation.
        layout = in_buf.get_layout()
        write_dep = _write_dep(in_buf)
        host_coords = host_coordinates(layout, write_dep, None)
        syms = host_coords[new_stick_dim].free_symbols
        assert len(syms) == 1
        device_dim = _device_dim_carrying_sym(
            layout.device_layout, write_dep, next(iter(syms))
        )
        assert device_dim is not None, (
            f"restickify padding: {in_buf.get_name()} exposed no paddable device "
            f"dim for new stick dim {new_stick_dim}"
        )
        in_buf.layout = _pad_device_dim(layout, device_dim, padded_dim_size)

    logger.debug(
        "insert_restickify_padding: padded input %s to device_size %s%s",
        in_buf.get_name(),
        in_buf.get_layout().device_layout.device_size,
        "" if size1 else f" (new stick host dim {new_stick_dim})",
    )


def insert_restickify_padding(graph: GraphLowering) -> None:
    """Pad a restickify's buffers so both are stick-aligned.

    A restickify re-tiles a tensor so a different host dim lands within the
    stick.  Codegen widens both its read and its write to stick boundaries, so
    both buffers need dims that cover the widened access.  The write side and the
    read side are independent -- either can need padding without the other -- so
    this pass handles them separately:

    - Write side (``_pad_restickify_output``): keeps the output's old-stick host
      dim at the correct physical offset.
    - Read side (``_pad_restickify_input``): keeps the read within the allocation.

    Padding touches only the device layout, never host tensor dim sizes, so later
    passes that depend on host sizes (e.g. ``propagate_named_dims``) see the
    tensor unchanged.  Codegen's backGap path covers the resulting host/device
    gap.

    A size-1 (elided) stick dim needs the same two adjustments, but its size and
    its coordinate are set by different passes:

    - Size, here: ``_pad_elided_dim`` prepends an outermost size-64 gap dim.
    - Coordinate, later: ``_restickify_restore_elided_dim`` (spyre_kernel.py)
      mints a fresh iteration symbol just before ``align_tensors``.  It waits
      because both operands must share that symbol for align to match them, and
      align only sees a symbol that already exists when it runs.

    The restore asserts the prepended gap dim is present before binding the
    symbol, so swapping the pass order fails loudly rather than silently.
    """
    for op in list(graph.operations):
        if is_restickify_op(op, graph):
            _pad_restickify_output(op, graph)
            _pad_restickify_input(op, graph)
