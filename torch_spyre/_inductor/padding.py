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

from typing import Any, Union
import dataclasses
import functools
import itertools
import struct
import sys
import math
import operator

import sympy
from sympy import FiniteSet, Interval, S, Set, oo, EmptySet

import torch
from sympy import Expr
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    IRNode,
    Layout,
    Buffer,
    ComputedBuffer,
    InputBuffer,
    FixedLayout,
    Operation,
    Pointwise,
    Reduction,
    InputsKernel,
    MutableBox,
    MutationLayoutSHOULDREMOVE,
    ReinterpretView,
    MultiOutput,
    FallbackKernel,
    TorchBindObject,
)
from torch._inductor.ops_handler import DefaultHandler, WrapperHandler
from torch._inductor.utils import sympy_product
from torch._inductor.virtualized import V

from .constants import (
    BATCH_MATMUL_FP8_OP,
    BATCH_MATMUL_OP,
    DEVICE_NAME,
    FP16_MAX,
    FP16_MIN,
)
from .errors import Unsupported
from .ir import (
    FixedTiledLayout,
    SpyreConstantFallback,
    SpyreEmptyFallback,
    SpyreCollectiveKernel,
    WaitKernel,
)
from .logging_utils import get_inductor_logger
from .pass_utils import (
    concretize_expr,
    concretize_index,
    device_coordinates,
    find_reduction_var,
    find_fx_node,
    host_coordinates,
    identify_matmul_inputs,
    is_restickify_coords,
    loop_induction_symbols,
    lower_pad_sequence,
    op_read_writes,
    patch_env,
    redirect_computed_buffer_reads,
    replace_computed_buffer_body,
)
from .views import compute_coordinates
from torch_spyre._C import (
    DataFormats,
    ElementArrangement,
    SpyreTensorLayout,
    get_elem_in_stick,
)
from .config import padding_strict_mode


logger = get_inductor_logger("padding")


def compute_padding(cur_size: int, dtype: torch.dtype) -> int:
    stick_size = get_elem_in_stick(dtype)
    pad = (stick_size - (cur_size % stick_size)) % stick_size
    return pad


def round_up_to_stick(cur_size: int, dtype: torch.dtype) -> int:
    """Round ``cur_size`` up to the next whole stick for ``dtype`` (no-op when
    already a stick multiple)."""
    return cur_size + compute_padding(cur_size, dtype)


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


def restride_loop_step(
    step: Expr,
    size: list[int],
    stride: list[int],
    padded_stride: list[int],
    buffer_name: str,
) -> Expr:
    """The per-trip step of a loop that walks ``buffer_name``, in the padded buffer.

    ``step`` is the loop variable's coefficient in a read of the unpadded buffer,
    so it is a whole number of positions along one of that buffer's dims:
    ``step = count * stride[d]``. Padding grows some dims, so the stride of every
    dim outside the padded one can change (the per-expert stride of a tiled weight
    goes from 67 * 128 to 128 * 128). The step in the padded buffer is
    ``count * padded_stride[d]``.

    ``count`` is 1 for one position and above 1 for a tile of several, and the
    count must stay inside the dim. Dims of size one are never stepped, so they
    cannot own the step. If two dims could own it the loop cannot be re-strided,
    and this raises ``Unsupported`` rather than guess or leave a step that walks
    the padded buffer by the unpadded stride (#5573). A step that no dim owns is
    returned unchanged.
    """
    dims = [d for d in range(len(stride)) if size[d] > 1 and stride[d] > 0]
    # Every dim that could own the step: a whole number (1 or more) of positions
    # along it, staying inside it. In a contiguous layout this is at most one dim
    # (stride[d] = stride[d + 1] * size[d + 1], so a multiple of an inner stride
    # that fits inside the inner dim is smaller than the outer stride). A layout
    # that is not contiguous (a transposed or column-major read-copy buffer) can
    # give two readings of one step, and re-striding the wrong one would read the
    # wrong place after the first trip, so it is rejected, not resolved by
    # preferring an exact match.
    owners = [d for d in dims if step % stride[d] == 0 and step // stride[d] < size[d]]
    if len(owners) > 1:
        raise Unsupported(
            f"insert_bmm_padding: cannot tell which dim of {buffer_name} the loop "
            f"steps by {step} (strides {stride}, sizes {size}); the padded "
            f"buffer's per-trip step would be wrong"
        )
    if not owners:
        return step
    d = owners[0]
    return sympy.Integer(step // stride[d] * padded_stride[d])


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

    This wraps the live index (see AGENTS.md "Compiler Pass Conventions")
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
        padded_stride: list[int],
    ):
        super().__init__(inner)
        self._orig_name = orig_name
        self._orig_size = orig_size
        self._orig_stride = orig_stride
        self._padded_name = padded_name
        self._padded_indexer = padded_indexer
        self._var_ranges = var_ranges
        self._padded_stride = padded_stride

    def load(self, name, index):
        if name != self._orig_name:
            return super().load(name, index)
        index = concretize_index(sympy.sympify(index), set(self._var_ranges))
        # A loop induction variable is not an iteration variable of this op, so
        # ``compute_coordinates`` cannot place it and would drop the term; the
        # padded read would then read tile 0 on every trip. Take the term out,
        # decompose the rest, and add it back at the padded buffer's stride.
        loop_term = sympy.S.Zero
        for symbol in loop_induction_symbols(index, self._var_ranges):
            step = index.coeff(symbol)
            if step != 0:
                loop_term += symbol * restride_loop_step(
                    step,
                    self._orig_size,
                    self._orig_stride,
                    self._padded_stride,
                    self._orig_name,
                )
                index = index - step * symbol
        coords = compute_coordinates(
            self._orig_size, self._orig_stride, self._var_ranges, index
        )
        return super().load(self._padded_name, self._padded_indexer(coords) + loop_term)


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
    (see test_bmm_invariant_y_unaligned_k_pads_for_each_tile).  Re-striding the live load
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
    y_padded_stride = [concretize_expr(s) for s in y_padded_buf.get_layout().stride]

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
            y_padded_stride,
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


def _refresh_loop_advance_for_padded_read(
    op: ComputedBuffer, y_buf: Buffer, y_padded_buf: Buffer
) -> None:
    """Re-stride the per-trip step of the read that now targets the padded y.

    ``splice_while_loops`` stamps ``squeezed_advance_per_read`` before padding,
    as the loop variable's coefficient in the read of y, which is a stride of
    y's own layout.  Padding retargets that read to a buffer whose K is larger,
    so the stride of every dim outside K grows (for example the per-expert stride
    of a tiled weight, 67 * 128 to 128 * 128).  The stamped step would walk the
    padded buffer by the unpadded stride, and every trip after the first would
    read the wrong place (#5573).

    A step that is the stride of one dim of y becomes that dim's stride in the
    padded buffer.  A step that is no dim's stride is left alone.  A step shared by
    several dims of size above one cannot be mapped, and raises ``Unsupported``
    rather than leave a step that would silently read the wrong place.
    """
    info = getattr(op, "loop_info", None)
    if info is None or not info.squeezed_advance_per_read:
        return
    old_size = [concretize_expr(s) for s in y_buf.get_layout().size]
    old_stride = [concretize_expr(s) for s in y_buf.get_layout().stride]
    new_stride = [concretize_expr(s) for s in y_padded_buf.get_layout().stride]
    if len(old_stride) != len(new_stride):
        return
    reads = [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]
    squeezed = [
        [list(level) for level in per_read]
        for per_read in info.squeezed_advance_per_read
    ]
    changed = False
    for idx, dep in enumerate(reads):
        if dep.name != y_padded_buf.get_name() or idx >= len(squeezed):
            continue
        for level in squeezed[idx]:
            for pos, (step, extent) in enumerate(level):
                # The same rule the index uses (_PaddedLoadHandler), so the two
                # sources of the advance cannot disagree until the record is gone.
                new_step = restride_loop_step(
                    step, old_size, old_stride, new_stride, y_buf.get_name()
                )
                if new_step != step:
                    level[pos] = (new_step, extent)
                    changed = True
    if changed:
        op.loop_info = dataclasses.replace(info, squeezed_advance_per_read=squeezed)


def _loop_info_without_advance(info, new_op):
    """Loop group membership of ``info`` with no per-trip advance on ``new_op``.

    ``tiled_dims_per_read`` and ``squeezed_advance_per_read`` are positional in the
    op's own reads.  A padding op reads different buffers than the matmul it pads
    for, so the matmul's records would land on the wrong reads (for example the
    tiled ``x``'s advance on the loop-invariant ``y`` being padded).  The padding
    op re-runs identical work every trip, so it advances nothing.
    """
    n_reads = len([d for d in new_op.get_read_writes().reads if hasattr(d, "name")])
    levels = len(info.loop_count)
    return dataclasses.replace(
        info,
        tiled_dims_per_read=[[[] for _ in range(levels)] for _ in range(n_reads)],
        output_tiled_dims=[[] for _ in range(levels)],
        squeezed_advance_per_read=[[[] for _ in range(levels)] for _ in range(n_reads)],
        squeezed_advance_output=[[] for _ in range(levels)],
        propagation=None,
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
        patch_env(V.graph)
        y_fx_node = find_fx_node(y_name, V.graph)

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
                    new_op.loop_info = _loop_info_without_advance(  # type: ignore[attr-defined]
                        op.loop_info, new_op
                    )

        # --- Rebuild matmul inner_fn to load y from the padded buffer ---
        # x is left entirely untouched: the original inner_fn's x loader is
        # preserved as-is.  Only y's loads are redirected to the padded buffer.
        new_op = _rebuild_matmul(op, y_buf, y_padded_buf, operations)
        _refresh_loop_advance_for_padded_read(new_op, y_buf, y_padded_buf)


# --------------------------------------------------------------------------- #
# insert_restickify_padding                                                   #
# --------------------------------------------------------------------------- #


def _write_dep(op):
    """Return op's write dependency on the buffer it produces."""
    writes = [d for d in op_read_writes(op).writes if hasattr(d, "name")]
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
    reads = [r for r in op_read_writes(op).reads if hasattr(r, "name")]
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
    device_coords = device_coordinates(stl, write_dep)
    for dim in range(len(device_coords) - 1):
        if sym in device_coords[dim].free_symbols:
            return dim
    return None


def _written_extent(stl: SpyreTensorLayout, write_dep, device_dim: int, sym) -> int:
    """Number of positions the write covers along ``device_dim``.

    Normally that dim's coordinate is ``sym`` alone, so its extent is the range of
    ``sym``.  A collapsed factorized dim (for example the contraction dim K of a
    ``[N, heads, D]`` weight, coordinate ``128*heads + D``) is written by several
    loop variables together; its extent is the highest coordinate written, plus
    one.  Those variables' ranges must be used, or the write looks like a slice of
    the dim.
    """
    coord = device_coordinates(stl, write_dep)[device_dim]
    if len(coord.free_symbols) <= 1:
        return concretize_expr(write_dep.ranges[sym])
    last = coord.xreplace(
        {v: concretize_expr(write_dep.ranges[v]) - 1 for v in coord.free_symbols}
    )
    return int(last) + 1


def _stick_symbol(stl: SpyreTensorLayout, dep) -> object | None:
    """Return dep's within-stick device coordinate's free symbol against stl,
    or None if it has none (e.g. a symbol-free size-1 stick, or a scalar /
    zero-dim layout with no coords at all).
    """
    device_coords = device_coordinates(stl, dep)
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

    in_coords = device_coordinates(in_layout.device_layout, in_dep)
    out_coords = device_coordinates(out_layout.device_layout, _write_dep(op))
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
    host_dim_size = _written_extent(stl, write_dep, device_dim, old_sym)
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


def _compute_digits(coord, ranges) -> list[tuple[int, sympy.Symbol]]:
    syms = coord.free_symbols
    if len(syms) < 2:
        return []

    residual = sympy.expand(coord)
    digits: list[tuple[int, sympy.Symbol]] = []
    for sym in syms:
        if sym not in ranges:
            return []
        coeff_expr = residual.coeff(sym)
        if coeff_expr.free_symbols or coeff_expr.is_integer is not True:
            return []
        coeff = concretize_expr(coeff_expr)
        if coeff <= 0:
            return []
        digits.append((coeff, sym))
        residual -= coeff_expr * sym

    # A numeric constant is a slice offset and does not change contiguity.
    if residual.free_symbols or not residual.is_number:
        return []

    return sorted(digits, key=lambda item: item[0])


def _is_dense_flattened_coordinate(digits, ranges) -> bool:
    """Return whether ``coord`` densely flattens two or more loop dimensions.

    ``host_coordinates`` normally returns one loop symbol per host dimension.
    A view may instead split one physical host dimension into several logical
    dimensions, for example ``128 * head + feature`` for a ``[H, 128]`` view of
    one ``H * 128`` projection dimension.  That is still a unit-stride,
    contiguous traversal of the host dimension: ordering the symbols from the
    innermost coefficient outwards must produce the mixed-radix strides
    ``1, size(inner), size(inner) * size(next), ...``.

    Reject nonlinear expressions and gapped or overlapping flattenings.  They need
    the same re-base copy as an ordinary strided input.
    """
    if not digits:
        return False
    expected = 1
    for coeff, sym in digits:
        if coeff != expected:
            return False
        expected *= concretize_expr(ranges[sym])
    return True


def _is_nonoverlapping_aligned_stick_coordinate(
    digits, coord, ranges, stick_sym, dtype
) -> bool:
    """Return whether ``coord`` gives each outer index a disjoint stick window.

    Coarse tiling can narrow an inner range without changing its outer physical
    stride, for example ``512 * batch + tile_sequence`` with a 256-element tile.
    That is not dense, but it is safe for a restickify when the inner symbol is
    unit-stride and every outer coefficient leaves enough room for the padded
    span of all lower-order symbols.
    """
    if not digits:
        return False
    syms = coord.free_symbols
    if stick_sym not in syms:
        return False

    stick_extent = concretize_expr(ranges[stick_sym])
    if compute_padding(stick_extent, dtype) != 0:
        return False

    if digits[0] != (1, stick_sym):
        return False

    span = stick_extent
    for coeff, sym in digits[1:]:
        if coeff < span:
            return False
        span += coeff * (concretize_expr(ranges[sym]) - 1)
    return True


def _restickify_input_required_extent(coord, ranges, stick_sym, dtype) -> int:
    """Return the allocation extent needed for a restickify read window.

    A view can split one physical input dimension into several logical loop
    dimensions, so the physical coordinate may be a dense mixed-radix sum such
    as ``64 * batch + sequence``.  The restickify reads a whole stick starting
    at each outer digit's base.  Account for the last such base, then round only
    the innermost (unit-stride) stick digit's extent.
    """
    syms = coord.free_symbols
    if stick_sym not in syms:
        raise Unsupported(
            f"insert_restickify_padding: new-stick symbol {stick_sym} is absent "
            f"from input coordinate {coord}"
        )
    if len(syms) == 1:
        # Preserve the existing one-symbol behavior, including coordinates
        # split across host dimensions with FloorDiv/Mod expressions.
        slice_offset = concretize_expr(coord.as_coeff_Add()[0])
        return slice_offset + round_up_to_stick(
            concretize_expr(ranges[stick_sym]), dtype
        )

    digits = _compute_digits(coord, ranges)
    is_dense = _is_dense_flattened_coordinate(digits, ranges)
    has_disjoint_stick_windows = _is_nonoverlapping_aligned_stick_coordinate(
        digits, coord, ranges, stick_sym, dtype
    )
    if not is_dense and not has_disjoint_stick_windows:
        raise Unsupported(
            f"insert_restickify_padding: input coordinate {coord} is not a "
            "dense flattening or a set of non-overlapping stick windows"
        )

    stick_extent = concretize_expr(ranges[stick_sym])
    if not has_disjoint_stick_windows:
        raise Unsupported(
            f"insert_restickify_padding: new-stick extent {stick_extent} in "
            f"dense flattened input coordinate {coord} requires a re-base copy"
        )

    max_slice_start = coord.subs(stick_sym, 0)
    for sym in syms - {stick_sym}:
        max_slice_start = max_slice_start.subs(sym, concretize_expr(ranges[sym]) - 1)
    return concretize_expr(max_slice_start) + round_up_to_stick(stick_extent, dtype)


def _assert_input_paddable(
    op: ComputedBuffer, in_dep, in_layout, out_stick_sym
) -> list[sympy.Expr]:
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
        if len(syms) > 1:
            digits = _compute_digits(coord, in_dep.ranges)
            if _is_dense_flattened_coordinate(
                digits, in_dep.ranges
            ) or _is_nonoverlapping_aligned_stick_coordinate(
                digits, coord, in_dep.ranges, out_stick_sym, in_layout.dtype
            ):
                continue
            raise Unsupported(
                f"insert_restickify_padding: host dim {i} of {op.get_name()} "
                f"(coord {coord}) is neither dense nor a set of "
                "non-overlapping stick windows"
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
    return in_host_coords


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
        in_host_coords = _assert_input_paddable(op, in_dep, in_layout, out_stick_sym)
        new_stick_dim = _host_dim_carrying_sym(in_host_coords, out_stick_sym)
        assert new_stick_dim is not None, (
            f"restickify padding: no input host dim carries new-stick symbol "
            f"{out_stick_sym} for {op.get_name()} (layout invariant broken)"
        )
        dtype = in_layout.dtype
        coord = in_host_coords[new_stick_dim]
        device_dim = _device_dim_carrying_sym(
            in_layout.device_layout, in_dep, out_stick_sym
        )
        assert device_dim is not None
        dim_size = concretize_expr(in_layout.device_layout.device_size[device_dim])
        padded_dim_size = _restickify_input_required_extent(
            coord, in_dep.ranges, out_stick_sym, dtype
        )
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
        patch_env(V.graph)
        in_fx_node = find_fx_node(in_dep.name, V.graph)
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


class InvalidDomain(Exception):
    def __init__(self, domain: Set, func: str, arg_index: int = 0):
        self.domain = domain
        self.func = func
        self.arg_index = arg_index


def _sigmoid(x: float) -> float:
    # Written so neither branch overflows: math.exp(-x) overflows for x < -709.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def _log(x: float) -> float:
    # log(0) is -inf and log(inf) is inf, as on the device; math.log raises.
    if x == 0:
        return -math.inf
    return math.log(x)


def _reciprocal(x: float) -> float:
    # 1/(+-0) is +-inf and 1/(+-inf) is 0, as on the device.
    if x == 0:
        return math.copysign(math.inf, x)
    return 1.0 / x


def _evaluate(op, *args) -> Expr:
    try:
        value = op(*(float(a) for a in args))
    except OverflowError:
        return oo
    except (ValueError, ZeroDivisionError):
        return _NAN
    if math.isnan(value):
        return _NAN
    if math.isinf(value):
        return oo if value > 0 else -oo
    return sympy.Float(value)


# The sign of a padding value: the piece of the extended reals it lies in.
# sympy intervals exclude +-oo, so the infinities are pieces of their own.
_SIGN_PIECES: dict[str, Set] = {
    "-oo": FiniteSet(-oo),
    "-": Interval.open(-oo, 0),
    "0": FiniteSet(0),
    "+": Interval.open(0, oo),
    "+oo": FiniteSet(oo),
}

# Values standing for each sign: the extremes and 1. Every sign an op can
# give on a piece is given on one of these, since the op is monotonic on each
# piece and a result past them underflows or overflows. The signs are the
# same in every float dtype, so float64's extremes serve for all.
_TINY = 5e-324  # smallest float64 subnormal
_HUGE = sys.float_info.max
_SIGN_SAMPLES: dict[str, tuple[float, ...]] = {
    "-oo": (-math.inf,),
    "-": (-_HUGE, -1.0, -_TINY),
    "0": (0.0,),
    "+": (_TINY, 1.0, _HUGE),
    "+oo": (math.inf,),
}


def _signs(s: Set) -> list[str]:
    """The signs of the values in s. An undecidable piece counts (safe)."""
    return [
        k
        for k, piece in _SIGN_PIECES.items()
        if s.intersect(piece).is_empty is not True
    ]


def _sign_of(value: float) -> str | None:
    """The sign of value, or None for NaN."""
    if math.isnan(value):
        return None
    if math.isinf(value):
        return "+oo" if value > 0 else "-oo"
    return "0" if value == 0 else ("+" if value > 0 else "-")


def _sign_image(op, *sets: Set) -> Set:
    """The values op can give on sets, known only up to their signs.

    op is evaluated on the samples of every combination of the inputs' signs,
    and the result is the union of the pieces of the signs it gives. A NaN
    result (0 * oo) adds NaN.
    """
    signs = set()
    for combination in itertools.product(*(_signs(s) for s in sets)):
        for values in itertools.product(*(_SIGN_SAMPLES[k] for k in combination)):
            try:
                signs.add(_sign_of(op(*values)))
            except OverflowError:
                signs.add("+oo")
            except (ValueError, ZeroDivisionError):
                signs.add(None)
    pieces = [_SIGN_PIECES[k] for k in signs if k is not None]
    if None in signs:
        pieces.append(FiniteSet(_NAN))
    return sympy.Union(*pieces)


# Placeholder returned by the tracers' indirect_indexing. A load whose index
# contains it is an indirect (gather) load.
_INDIRECT_INDEX_SYM = sympy.Symbol("indirect_index")


# NaN as a set element. sympy's S.NaN cannot be in a set with numbers: sets
# sort their elements, and NaN is not comparable. A non-real symbol can, and
# it is in no domain, so every domain check rejects it.
_NAN = sympy.Symbol("NaN", real=False)


def _without_nan(s: Any) -> tuple[Any, bool]:
    """s without _NAN, and whether s held it. Non-sets pass through."""
    if not isinstance(s, Set) or not s.has(_NAN):
        return s, False
    if isinstance(s, FiniteSet):
        return FiniteSet(*(e for e in s if e != _NAN)), True
    if isinstance(s, sympy.Union):
        return sympy.Union(*(_without_nan(a)[0] for a in s.args)), True
    raise TypeError(f"Cannot split NaN from {s}")


def _nan_args(args, kwargs):
    """args and kwargs without _NAN, and whether any of them held it."""
    split_args = [_without_nan(a) for a in args]
    split_kwargs = {k: _without_nan(v) for k, v in kwargs.items()}
    had_nan = any(n for _, n in split_args) or any(n for _, n in split_kwargs.values())
    real_args = [a for a, _ in split_args]
    real_kwargs = {k: v for k, (v, _) in split_kwargs.items()}
    return real_args, real_kwargs, had_nan


def _propagates_nan(op):
    """op(..., NaN, ...) is NaN: the IEEE-754 rule for arithmetic, and on
    Spyre also for max and min (see _FP16_MAX). op sees only the real values,
    and NaN is added to its result when an argument held it."""

    @functools.wraps(op)
    def wrapper(self, *args, **kwargs):
        real_args, real_kwargs, had_nan = _nan_args(args, kwargs)
        # A NaN-only argument has no real values, and neither has the result.
        if any(a is S.EmptySet for a in real_args):
            result = S.EmptySet
        else:
            result = op(self, *real_args, **real_kwargs)
        return result.union(FiniteSet(_NAN)) if had_nan else result

    return wrapper


def _nan_gives_any_boolean(op):
    """A comparison or logical op with a NaN argument may give either value
    (IEEE-754 gives False except for ne, but the device is not guaranteed)."""

    @functools.wraps(op)
    def wrapper(self, *args, **kwargs):
        real_args, real_kwargs, had_nan = _nan_args(args, kwargs)
        if had_nan:
            return self.booleans
        return op(self, *real_args, **real_kwargs)

    return wrapper


def _user_values():
    values = S.Reals
    if padding_strict_mode:
        values = values.union(FiniteSet(_NAN))
    return values


class DomainRangeSolver(DefaultHandler):
    """
    Validates input set and propagate to image.

    Each function receives as input a sympy Set for each argument, which stands
    for the possible values that can be in the tensor's padding, if any. This input
    set has to be a subset of the Domain or else an InvalidDomain exception is
    raised. This exception is a signal that we must set a specific padding value.

    The functions are free to return the codomain of the function, e.g. the real numbers,
    but we want to narrow it down at the very least to the range of the function, e.g.
    positive real numbers. Where possible, for example for some unary or binary
    operations on FiniteSets or Intervals, we might be able to return the image.
    For example FiniteSet(0) represents the default zero-padding and for most
    functions it's trivial to return the image of {0}.
    """

    def __init__(self):
        self.non_zero_reals = S.Reals - FiniteSet(0)
        self.positive_numbers = Interval(0, sympy.oo)
        self.booleans = FiniteSet(0, 1)
        self.strictly_positive_numbers = Interval(0, sympy.oo, left_open=True)
        # -oo and oo are always the open-ended limits of an interval in sympy
        self.extended_non_zero_reals = self.non_zero_reals.union(FiniteSet(-oo, oo))
        self.extended_strictly_positive_numbers = self.strictly_positive_numbers.union(
            FiniteSet(oo)
        )
        self.extended_positive_numbers = self.positive_numbers.union(FiniteSet(oo))

    def constant(self, value: Union[bool, float, int], dtype: torch.dtype) -> Set:
        return FiniteSet(value)

    def indirect_indexing(
        self,
        x: Any,
        size: Any,
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        zero = FiniteSet(0)
        # Due to the risk of invalid memory access we only admit 0 as padding
        if not zero == x:
            raise InvalidDomain(zero, "indirect_indexing")
        return _INDIRECT_INDEX_SYM

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type,
        value: Set | tuple[Set, ...],
    ) -> Set:
        if not isinstance(value, tuple):
            value = (value,)
        required_padding = FiniteSet(0)
        if reduction_type.startswith("batchmatmul"):
            assert isinstance(value, tuple) and len(value) == 2

            # at least one of the paddings has to be zero
            if not any(v.is_subset(required_padding) for v in value):
                # TODO: verify which choice is optimal.
                raise InvalidDomain(required_padding, reduction_type, arg_index=0)
            return FiniteSet(0)
        else:
            if (pad := get_pad_value(reduction_type, dtype)) is not None:
                required_padding = FiniteSet(pad)
            for arg_index, v in enumerate(value):
                if not v.is_subset(required_padding):
                    raise InvalidDomain(required_padding, reduction_type, arg_index)
            # other reductions can leave some garbage in padding
            return S.Reals

    def _unary_op(self, op, x0, image) -> Set:
        if isinstance(x0, FiniteSet):
            return FiniteSet(*(_evaluate(op, e) for e in x0))
        return _sign_image(op, x0).intersect(image)

    def _binary_op(self, op, x0, x1, image=None) -> Set:
        if isinstance(x0, FiniteSet) and isinstance(x1, FiniteSet):
            return FiniteSet(*(_evaluate(op, a, b) for a in x0 for b in x1))
        result = _sign_image(op, x0, x1)
        return result if image is None else result.intersect(image)

    def _compare(self, op, x0, x1) -> Set:
        if isinstance(x0, FiniteSet) and isinstance(x1, FiniteSet):
            return FiniteSet(*(int(bool(op(a, b))) for a in x0 for b in x1))
        return self.booleans

    @_propagates_nan
    def abs(self, x0):
        return self._unary_op(abs, x0, self.positive_numbers)

    @_propagates_nan
    def add(self, x0, x1):
        return self._binary_op(operator.add, x0, x1)

    @_propagates_nan
    def clamp(self, x, min=None, max=None):
        if min is not None:
            x = self.maximum(x, FiniteSet(min))
        if max is not None:
            x = self.minimum(x, FiniteSet(max))
        return x

    @_nan_gives_any_boolean
    def eq(self, x0, x1):
        return self._compare(operator.eq, x0, x1)

    @_propagates_nan
    def exp(self, x0):
        return self._unary_op(math.exp, x0, self.extended_positive_numbers)

    def exx2(self, x, scale, useZeroMean):
        return self.positive_numbers

    @_propagates_nan
    def floor(self, x0):
        return x0.intersection(S.Integers)

    @_nan_gives_any_boolean
    def ge(self, x0, x1):
        return self._compare(operator.ge, x0, x1)

    @_propagates_nan
    def gelu(self, x):
        return Interval(-0.17, sympy.oo)

    @_nan_gives_any_boolean
    def gt(self, x0, x1):
        return self._compare(operator.gt, x0, x1)

    def keep_by_index(self, x, indices):
        return x

    def layernormnorm(self, x, mean, norm_mean, weight, bias):
        return S.Reals

    def layernormscale(self, x, eps):
        return self.positive_numbers

    @_propagates_nan
    def log(self, x0):
        if not x0.is_subset(self.extended_strictly_positive_numbers):
            raise InvalidDomain(self.extended_strictly_positive_numbers, "log")
        return self._unary_op(_log, x0, S.Reals)

    @_nan_gives_any_boolean
    def le(self, x0, x1):
        return self._compare(operator.le, x0, x1)

    @_nan_gives_any_boolean
    def logical_and(self, x0, x1):
        return self._compare(lambda a, b: a and b, x0, x1)

    @_nan_gives_any_boolean
    def logical_or(self, x0, x1):
        return self._compare(lambda a, b: a or b, x0, x1)

    @_nan_gives_any_boolean
    def lt(self, x0, x1):
        return self._compare(operator.lt, x0, x1)

    # max and min are constant where one argument dominates, so a bound from
    # an open corner can still be attained (max(x, 0) = 0 for every x <= 0).
    # Their exact image is set algebra instead: max(a, b) = a exactly when
    # some b <= a, so max(x0, x1) = (x0 & {v >= some b}) | (x1 & {v >= some a}).

    @staticmethod
    def _at_or_above(s: Set) -> Set:
        """The values, including oo, at or above some element of s."""
        attained = s.contains(s.inf) is S.true
        # sympy intervals exclude +-oo, so s.inf (if it is -oo) and oo, which
        # is above every element, are added as points.
        bound = FiniteSet(s.inf) if attained else S.EmptySet
        return sympy.Union(
            Interval(s.inf, oo, left_open=not attained), bound, FiniteSet(oo)
        )

    @staticmethod
    def _at_or_below(s: Set) -> Set:
        """The values, including -oo, at or below some element of s."""
        attained = s.contains(s.sup) is S.true
        bound = FiniteSet(s.sup) if attained else S.EmptySet
        return sympy.Union(
            Interval(-oo, s.sup, right_open=not attained), bound, FiniteSet(-oo)
        )

    @_propagates_nan
    def maximum(self, x0, x1):
        if x0 is S.EmptySet or x1 is S.EmptySet:
            return S.EmptySet
        return x0.intersect(self._at_or_above(x1)).union(
            x1.intersect(self._at_or_above(x0))
        )

    @_propagates_nan
    def minimum(self, x0, x1):
        if x0 is S.EmptySet or x1 is S.EmptySet:
            return S.EmptySet
        return x0.intersect(self._at_or_below(x1)).union(
            x1.intersect(self._at_or_below(x0))
        )

    @_propagates_nan
    def mul(self, x0, x1):
        return self._binary_op(operator.mul, x0, x1)

    @_nan_gives_any_boolean
    def ne(self, x0, x1):
        return self._compare(operator.ne, x0, x1)

    @_propagates_nan
    def neg(self, x0: Set):
        return self._unary_op(operator.neg, x0, S.Reals)

    @_propagates_nan
    def reciprocal(self, x0):
        if not x0.is_subset(self.extended_non_zero_reals):
            raise InvalidDomain(self.extended_non_zero_reals, "reciprocal")
        return self._unary_op(_reciprocal, x0, self.extended_non_zero_reals)

    def qfp8ch(self, x):
        return x

    def qfp8wt(self, x):
        return x

    def quantscalepertokenfp8(self, x, scale_ub):
        return self.positive_numbers

    @_propagates_nan
    def relu(self, x0):
        # relu(x) = max(x, 0) is constant for x <= 0, so 0 is attained even
        # when x0 is open below it; maximum gets that right.
        return self.maximum(x0, FiniteSet(0))

    @_propagates_nan
    def rsqrt(self, x0):
        if not x0.is_subset(self.extended_strictly_positive_numbers):
            raise InvalidDomain(self.extended_strictly_positive_numbers, "rsqrt")
        return self._unary_op(
            lambda x: _reciprocal(math.sqrt(x)),
            x0,
            self.extended_strictly_positive_numbers,
        )

    @_propagates_nan
    def sigmoid(self, x0):
        # Saturates to exactly 0 and 1 on the device (sigmoid(-20) == 0).
        return self._unary_op(_sigmoid, x0, Interval(0, 1))

    @_propagates_nan
    def softplus(self, x, beta, threshold):
        def sp(v):
            return v if beta * v > threshold else math.log1p(math.exp(beta * v)) / beta

        return self._unary_op(sp, x, self.positive_numbers)

    @_propagates_nan
    def sqrt(self, x0):
        if not x0.is_subset(self.positive_numbers):
            raise InvalidDomain(self.positive_numbers, "rsqrt")
        return self._unary_op(math.sqrt, x0, self.positive_numbers)

    @_propagates_nan
    def square(self, x0):
        return self._unary_op(lambda x: x * x, x0, self.extended_positive_numbers)

    @_propagates_nan
    def sub(self, x0, x1):
        return self._binary_op(operator.sub, x0, x1)

    @_propagates_nan
    def tanh(self, x0):
        # Saturates to exactly -1 and 1 on the device.
        return self._unary_op(math.tanh, x0, Interval(-1, 1))

    def to_dtype(self, x, dtype, src_dtype=None, use_compute_types=False):
        return x

    @_propagates_nan
    def truediv(self, x0, x1):
        if not x1.is_subset(self.extended_non_zero_reals):
            # realdiv is the correct key in _PADDING_MASK_VALUES
            raise InvalidDomain(self.extended_non_zero_reals, "realdiv", arg_index=1)
        return self._binary_op(operator.truediv, x0, x1)

    @_propagates_nan
    def silu(self, x0):
        # silu's minimum is ~-0.2785 (-0.27856 in fp16).
        return self._unary_op(lambda x: x * _sigmoid(x), x0, Interval(-0.279, oo))

    @_propagates_nan
    def where(self, condition, input, other):
        return input.union(other)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        logger.warning(
            "Operation %s has not been enabled in padding propagation. "
            "No inferences can be made about its padding state.",
            name,
        )
        return _user_values()


@dataclasses.dataclass(frozen=True)
class PaddingDecision:
    # The value of the output padding, of the tensor has padding
    output_padding: float | None
    # True if the op supports masking and the input needs fixing
    do_mask: bool
    # True if the op doesn't support masking and the input needs fixing
    do_overwrite: bool
    # The value to set the input padding to with masking or overwriting
    input_vals: dict[str, float]


class DomainRangeTracer(DomainRangeSolver):
    """
    Traces through a graph to infer and correct the state of padding.

    Each graph node has to be fed to the tracer in topological order.
    For each node, it will validate the operands, verify if they
    need padding correction and compute output padding if any.
    """

    name = "DomainRangeTracer"

    def __init__(self, graph: GraphLowering):
        super().__init__()
        self.current_op: ComputedBuffer | None = None
        # buffer name -> output set
        self.buffer_range = dict[str, Set]()
        # Same as buffer_range, but reset for each op to account
        # for slicing
        self.current_reads = dict[str, Set]()
        # op name -> (arg name -> adjusted )
        self.buffer_input = dict[str, dict[str, Set]]()
        self.graph = graph

    def start_op(self, op):
        self.current_op = op
        self.current_reads = {}

    def load(self, name: str, index: sympy.Expr) -> Set:
        assert self.current_op is not None

        padding_range = None
        if (overwrites := self.buffer_input.get(self.current_op.name)) is not None:
            if (val := overwrites.get(name)) is not None:
                padding_range = val
        else:
            self.buffer_input[self.current_op.name] = {}

        if padding_range is None:
            padding_range = self.current_reads.get(name)
            if padding_range is not None and _INDIRECT_INDEX_SYM in getattr(
                index, "free_symbols", ()
            ):
                # An indirect load does not read the source's padding: in the
                # output's padding lanes it reads source[i], where i is the
                # index's padding, which indirect_indexing restricts to 0. So
                # the lanes hold valid source elements, i.e. user values.
                padding_range = _user_values()
        if padding_range is None:
            current_op_name = (
                self.current_op.name if self.current_op is not None else None
            )
            raise ValueError(
                f"Expected range of buffer {name} to be known. Current op is {current_op_name}"
            )
        return padding_range

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: Set,
        mode=None,
    ) -> None:
        self.buffer_range[name] = value

    def store_reduction(self, name: str, index: sympy.Expr, value: Set) -> None:
        self.buffer_range[name] = value

    def run_node(self, op: Operation) -> PaddingDecision:
        self.start_op(op)
        padded = is_padded(self.graph, op)

        input_buffers, any_input_padded = self.handle_operands(op, padded)

        if isinstance(op, InputsKernel):
            self.handle_graph_input(op, padded)
            return PaddingDecision(None, False, False, {})

        if _is_allocation(op):
            # An empty() allocation writes nothing: its memory, padding
            # included, holds whatever was there. Tracing it would load
            # Inductor's zero-element placeholder, a NaN constant.
            self.buffer_range[op.name] = _user_values() if padded else EmptySet
            return PaddingDecision(None, False, False, {})

        if padded or any_input_padded:
            # Even if the output of this op is not padded, we need to run
            # the input validation if there are padded inputs
            do_coord_mask, do_pad = self.trace_op(op, input_buffers)
        else:
            do_coord_mask = do_pad = False

        if not padded:
            self.buffer_range[op.name] = EmptySet
        elif is_restickify_op(op, self.graph):
            # The output's stick is a non-stick dim of the input, so its padding
            # lanes come from input rows past that dim's valid extent, not from
            # the input's padding. insert_restickify_padding (which runs later)
            # grows the input allocation to cover them, and nothing writes the
            # new rows, so their content is unknown. Mark that on the output and
            # leave the input's range alone.
            self.buffer_range[op.name] = self.buffer_range[op.name].union(
                _user_values()
            )

        overwrites = {}
        output_padding = None

        if do_coord_mask or do_pad:
            if padded:
                # Record the output padding only when it is proven to be one
                # value. Later graphs trust a recorded value and skip masking,
                # so picking an element of a wider set (e.g. the Reals a
                # masked reduction leaves) would claim a padding that is
                # not there.
                output_padding = _single_padding_value(self.buffer_range[op.name])

            for arg_name, pad_set in self.buffer_input[op.name].items():
                overwrites[arg_name] = float(pick_element(pad_set))

        return PaddingDecision(output_padding, do_coord_mask, do_pad, overwrites)

    def handle_graph_input(self, op, padded):
        if not padded:
            self.buffer_range[op.name] = EmptySet
            return
        if isinstance(op, SpyreConstantFallback):
            self.buffer_range[op.name] = FiniteSet(op.constant_args[0])
            return

        for _class, pad in _ASSUMED_PADDING.items():
            if isinstance(op, _class):
                self.buffer_range[op.name] = pad
                break
        else:
            raise ValueError(f"Couldn't find entry for {op} in _ASSUMED_PADDING")

    def handle_operands(
        self, op: Operation, padded: bool
    ) -> tuple[dict[str, Buffer], bool]:
        reads = op.get_read_writes().reads
        input_buffers = dict[str, Buffer]()
        any_input_padded = False
        for dep in reads:
            buf = self.graph.get_buffer(dep.name)
            while isinstance(buf, MutableBox):
                buf = buf.data
            if isinstance(buf, TorchBindObject):
                # An opaque script object (e.g. a process group passed to a
                # collective) holds no tensor data, so it has no padding.
                continue
            stl = _get_stl(self.graph, buf)
            buf_is_padded = is_padded(self.graph, buf) if stl is not None else False
            if isinstance(buf, InputBuffer):
                if dep.name not in self.buffer_range:
                    if is_spyre_layout(buf.layout):
                        assert stl is not None
                        # Seed from the padding the input held before the graph,
                        # once: after a mutation the range is what that left.
                        self.buffer_range[dep.name] = (
                            _input_padding_range(stl) if buf_is_padded else EmptySet
                        )
                    else:
                        # other device buffers are not padded
                        self.buffer_range[dep.name] = EmptySet
            else:
                assert dep.name in self.buffer_range, (
                    f"Nodes seen out of order, expected {dep.name} "
                    f"({type(buf).__name__}) to have been visited"
                )
            self.current_reads[dep.name] = self.buffer_range[dep.name]
            input_buffers[dep.name] = buf
            if stl is None:
                continue

            # The read is a slice when it touches fewer elements than the
            # buffer holds. Compare host numels, both in the graph's size
            # symbols (valid_elements is concrete even when the sizes are
            # dynamic, so it cannot be compared with a symbolic read). A
            # read not proven to cover the buffer counts as a slice.
            is_sliced = not V.graph.sizevars.statically_known_geq(
                dep.get_numel(), V.graph.get_numel(dep.name)
            )

            # If the output is padded and there is slicing on the input it means
            # that the if there is slicing along the stiock dim, valid values
            # from the input can end up in the output padding
            # TODO: refine analysis to exclude non-stick slicing cases
            # An input with no padding cannot supply the output's padding lanes
            # from its own padding: they read valid input elements instead
            # (a broadcast, a transpose, or a restickify that also reads
            # another buffer and so is not recognised by is_restickify_op).
            reads_valid_into_padding = is_sliced or (
                self.buffer_range[dep.name] is EmptySet
            )
            if padded and reads_valid_into_padding:
                self.current_reads[dep.name] = self.current_reads[dep.name].union(
                    _user_values()
                )
                buf_is_padded = True
            any_input_padded = any_input_padded or buf_is_padded
        return input_buffers, any_input_padded

    def trace_op(self, op, input_buffers: dict[str, Buffer]) -> tuple[bool, bool]:
        index2arg = inx2arg(op)
        do_coord_mask = False
        do_pad = False
        # Fixed point loop until we can either allow the inputs to pass unmodified or
        # find a suitable padding. So the max iteration count is 1 for the
        # first try and 1 retry for each argument
        for i in range(1 + len(input_buffers)):
            try:
                _call_innner(op)
                break
            except InvalidDomain as inv_domain:
                name = index2arg[inv_domain.arg_index]
                buf = input_buffers[name]
                if can_coord_mask(inv_domain.func, buf.layout.dtype):
                    do_coord_mask = True
                    # For now this is true for all operations that are padded
                    if op.layout.dtype in (torch.float16, torch.bfloat16):
                        self.buffer_range[op.name] = FiniteSet(0)
                    else:
                        self.buffer_range[op.name] = Interval(0, 1e-2)
                else:
                    do_pad = True

                try:
                    pad_value = get_or_pick_pad_value(inv_domain, buf.layout.dtype)
                    self.buffer_input[op.name][name] = FiniteSet(pad_value)
                except ValueError:
                    raise ValueError(
                        f"Could not find legal pad for input {name} of {op.name}:{inv_domain.func}"
                    )
        else:
            raise ValueError(f"Could not find an input padding that op {op.name}")
        return do_coord_mask, do_pad

    def handle_mutation_output(self, op):
        if isinstance(op, ComputedBuffer) and isinstance(
            op.layout, MutationLayoutSHOULDREMOVE
        ):
            target = op.layout.target
            while isinstance(target, (MutableBox, ReinterpretView)):
                target = target.data
            target_name = target.maybe_get_name()
            if target_name is not None:
                target_is_padded = is_padded(self.graph, target)
                # Seed a mutated input with its padding from before the graph,
                # unless an earlier mutation already set what it holds now.
                if (
                    isinstance(target, InputBuffer)
                    and target_name not in self.buffer_range
                ):
                    self.buffer_range[target_name] = (
                        _input_padding_range(_get_stl(self.graph, target))
                        if target_is_padded
                        else EmptySet
                    )

                assert target_name in self.buffer_range, (
                    f"Mutation buffer {op} seen before the buffer it mutates {target_name}"
                )

                if not target_is_padded:
                    self.buffer_range[target_name] = EmptySet
                    return

                # If we're here it means that the target is padded.
                # Without deeper analysis we don't know if the values written into the
                # destination padding come from source padding or valid elements.
                # Set the resulting padding to all possible inputs united with
                # what could be in the mutation source buffer padding.
                user_data = _user_values()
                pad_set = self.buffer_range[op.name]
                self.buffer_range[target_name] = user_data.union(pad_set)


@dataclasses.dataclass(frozen=True)
class SetDtype:
    set: Set
    dtype: torch.dtype | None


class SetDtypeTracer(DefaultHandler):
    """Run DomainRangeTracer and DtypePropagationOpsHandler side by side.

    Some operations like exp() can have special limitations depending on
    the Dtype. This handler wraps the DomainRangeTracer which operates on
    a purely mathematical level of abstraction.
    """

    name = "SetDtypeTracer"

    def __init__(self, tracer: DomainRangeTracer):
        super().__init__()
        self.tracer = tracer
        self.dtype_rules = DtypePropagationOpsHandler()

    @staticmethod
    def _sets(x):
        if isinstance(x, SetDtype):
            return x.set
        if isinstance(x, (tuple, list)):
            return type(x)(SetDtypeTracer._sets(e) for e in x)
        return x

    def _dtype(self, name: str, args, kwargs) -> Any:
        values = [*args, *kwargs.values()]
        if any(isinstance(v, SetDtype) and v.dtype is None for v in values):
            return None
        rule = getattr(self.dtype_rules, name, None)
        if rule is None:
            return None
        try:
            return rule(*args, **kwargs)
        except (AssertionError, AttributeError, TypeError):
            # A rule that cannot handle these args (e.g. a list of indices
            # passed to promote_types) leaves the dtype unknown; only
            # dtype-dependent domains need it.
            return None

    @staticmethod
    def _wrap(r, dtype):
        if isinstance(r, Set):
            return SetDtype(r, dtype)
        if isinstance(r, tuple):
            dtypes = dtype if isinstance(dtype, tuple) else (dtype,) * len(r)
            return tuple(SetDtypeTracer._wrap(e, d) for e, d in zip(r, dtypes))
        return r

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]):
        r = getattr(self.tracer, name)(
            *self._sets(args), **{k: self._sets(v) for k, v in kwargs.items()}
        )
        if not isinstance(r, (Set, tuple)):
            return r
        return self._wrap(r, self._dtype(name, args, kwargs))

    def load(self, name: str, index: sympy.Expr) -> SetDtype:
        return SetDtype(self.tracer.load(name, index), V.graph.get_dtype(name))

    def constant(self, value: Union[bool, float, int], dtype: torch.dtype) -> Set:
        return self._wrap(self.tracer.constant(value, dtype), dtype)

    def exp(self, x0) -> SetDtype:
        # exp overflows above log(max finite value): ~11.09 for fp16, ~88.72
        # for fp32. An unknown dtype gets the narrower fp16 domain.
        upper = 88.7 if x0.dtype == torch.float32 else 11.05
        domain = Interval(-sympy.oo, upper)
        if not x0.set.is_subset(domain):
            raise InvalidDomain(domain, "exp")
        return self._default("exp", args=(x0,), kwargs={})

    def to_dtype(
        self,
        x: SetDtype,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> SetDtype:
        return SetDtype(
            self.tracer.to_dtype(x.set, dtype, src_dtype, use_compute_types), dtype
        )


@dataclasses.dataclass(frozen=True)
class ArgName:
    name: str


class ArgNameToIndexTracer(DefaultHandler):
    """Helper to map argument names to operand indices

    For an operation like x * y, op.read_wites().reads will contain
    2 dependencies which means that we should be able to match the argument
    index to the argument name. But for example, x * x has only one read
    dependency. This tracer is meant to build this mapping.
    """

    def __init__(self):
        super().__init__()
        self.name2inx = dict[str, list[int]]()
        self.constant_counter = 0

    def get_inx2arg(self) -> list[str]:
        for name, indices in self.name2inx.items():
            assert len(indices) > 0, f"Could not resolve index of arg {name}"

        index_entries = [
            (index, name)
            for name, indices in self.name2inx.items()
            for index in indices
        ]
        index_entries.sort(key=lambda pair: pair[0])

        inx2arg = []

        for i, (index, name) in enumerate(index_entries):
            if i > 1:
                assert index_entries[i - 1][0] in [index, index - 1], "Skipped argument"
            inx2arg.append(name)

        return inx2arg

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if name in ("store", "store_reduction"):
            return
        for argname, arg in kwargs.items():
            assert not isinstance(arg, ArgName), (
                f"Can't determine order of kwargs. Found kwarg {argname}"
            )

        counter = 0
        for arg in args:
            if isinstance(arg, ArgName):
                self.name2inx.setdefault(arg.name, [])
                self.name2inx[arg.name].append(counter)
                counter = counter + 1

        if name == "indirect_indexing":
            return _INDIRECT_INDEX_SYM

    def load(self, name: str, index: sympy.Expr) -> ArgName:
        return ArgName(name)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type,
        value: Set | tuple[Set, ...],
    ) -> None:
        if not isinstance(value, tuple):
            value = (value,)
        for i, v in enumerate(value):
            assert isinstance(v, ArgName)
            self.name2inx[v.name] = [i]

    def constant(self, value: Union[bool, float, int], dtype: torch.dtype):
        name = f"__const_{self.constant_counter}"
        self.constant_counter += 1
        return ArgName(name)


def inx2arg(op: ComputedBuffer):
    arg_pos_tracer = ArgNameToIndexTracer()
    with V.set_ops_handler(arg_pos_tracer):
        _call_innner(op)
    return arg_pos_tracer.get_inx2arg()


# NOTE: the mask value is a large finite negative, not -inf. encode_constant()
# (module.cpp -> deeptools::FloatToFp16Bin) mis-encodes IEEE +-inf as a NaN bit
# pattern instead of the fp16 infinity encoding (confirmed empirically: -inf
# round-trips to NaN, not 0xFC00), so exp(-inf) never reaches the runtime as
# exp(-inf) -- it reaches it as exp(NaN), which poisons the output instead of
# masking it to 0. -1e4 is finite (encodes correctly) and still underflows
# exp() to exactly 0 in fp16 (fp16's smallest positive value is ~6e-8; anything
# below about -12 already underflows), so it produces the same masking effect
# without going through the broken infinity path.
#
# The max/min reduction identities below have the same encode_constant bug:
# _get_mask_value("max") fed float("-inf") through the same broken path, so a
# max-reduction's padded lanes were seeded with NaN instead of -inf. Unlike
# exp's poisoned-but-locally-contained NaN, max(x, NaN) == NaN -- the identity
# failure propagates through the whole reduction result, not just the padded
# lanes, which would poison any block_max = torch.amax(scores, dim=-1) whose
# padded lanes get reduced over. _FP16_MAX/_FP16_MIN are the most extreme
# finite fp16 values, so they lose (max)/win (min) against any real fp16 score
# while still encoding correctly.
_FP16_MAX = FP16_MAX
_FP16_MIN = FP16_MIN

# 32-bit coordinate masking uses a repeating 16-bit pattern for the mask
# value. 0.0 (used for sum/mean) already satisfies this requirement.
# For max and min, the largest finite values with matching upper and lower
# 16-bit halves are constructed directly from their bit patterns.
_FP32_MIN = struct.unpack("<f", struct.pack("<I", 0xFF7FFF7F))[0]  # -3.40279751e38
_FP32_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7F7F7F))[0]  # 3.39617752e38
_FP32_ALMOST_1 = struct.unpack("<f", struct.pack("<I", 0x3F803F80))[
    0
]  # 1.0019378662109375

_PADDING_MASK_VALUES: dict[str, dict[DataFormats, float]] = {
    "max": {
        DataFormats.SEN169_FP16: _FP16_MIN,
        DataFormats.IEEE_FP32: _FP32_MIN,
    },
    "min": {
        DataFormats.SEN169_FP16: _FP16_MAX,
        DataFormats.IEEE_FP32: _FP32_MAX,
    },
    "exp": {
        DataFormats.SEN169_FP16: -1e4,  # exp(-1e4) underflows to 0 in fp16; see NOTE above.
        DataFormats.IEEE_FP32: _FP32_MIN,
    },
    "reciprocal": {
        DataFormats.SEN169_FP16: float("inf"),
        DataFormats.IEEE_FP32: _FP32_MAX,
    },
    "rsqrt": {
        DataFormats.SEN169_FP16: float("inf"),
        DataFormats.IEEE_FP32: _FP32_MAX,
    },
    "log": {
        DataFormats.SEN169_FP16: 1,
        DataFormats.IEEE_FP32: _FP32_ALMOST_1,
    },
    "realdiv": {
        DataFormats.SEN169_FP16: float("inf"),
        DataFormats.IEEE_FP32: _FP32_MAX,
    },
    "sum": {
        DataFormats.SEN169_FP16: 0,
        DataFormats.IEEE_FP32: 0,
    },
    "mean": {
        DataFormats.SEN169_FP16: 0,
        DataFormats.IEEE_FP32: 0,
    },
}

_POINTWISE_PADDING_OPS: set[str] = {"exp", "reciprocal", "rsqrt", "log", "realdiv"}


def pick_element(s: Set):
    """Return a valid real element of s, or raise ValueError.

    NaN is never picked: the element is a padding value to write.
    """
    s, _ = _without_nan(s)

    # Prefer zero, then one.
    for x in (0, 1):
        if s.contains(x) is S.true:
            return x

    if isinstance(s, FiniteSet):
        if not s:
            raise ValueError("Empty set")
        return min(s)

    if isinstance(s, Interval):
        a, b = s.start, s.end

        if not s.left_open:
            x = a
        elif a == -oo:
            x = b - 1
        elif b == oo:
            x = a + 1
        else:
            x = (a + b) / 2

        if s.contains(x):
            return x

    raise ValueError(f"Cannot select an element from {s}")


def get_pad_value(func, dtype) -> float | None:
    if func.startswith("batchmatmul"):
        return 0
    if func == "indirect_indexing":
        return 0
    # bfloat16 is stored on device as SEN169_FP16, like float16.
    if dtype in (torch.float16, torch.bfloat16):
        data_format = DataFormats.SEN169_FP16
    elif dtype == torch.float32:
        data_format = DataFormats.IEEE_FP32
    else:
        return None
    return _PADDING_MASK_VALUES.get(func, {}).get(data_format, None)


def can_coord_mask(func_name, dtype) -> bool:
    if func_name in _PADDING_MASK_VALUES:
        return get_pad_value(func_name, dtype) is not None
    return False


def _index_syms(n: int) -> tuple[sympy.Symbol, ...]:
    """The index symbols _trace_inner_fn passes to a traced inner_fn."""
    return tuple(sympy.Symbol(f"_i{k}") for k in range(n))


def _is_allocation(op: Operation) -> bool:
    """Whether op is an empty() allocation, which writes nothing.

    Inductor's empty_strided realizes a zero-filled Pointwise and then sets its
    ranges to zeros, so the scheduler emits no kernel for it, and make_loader
    swaps its body for nop_loader_fn: a NaN constant for floating dtypes.
    """
    return (
        isinstance(op, ComputedBuffer)
        and isinstance(op.data, Pointwise)
        and op.data.is_zero_elements()
        and not isinstance(op.layout, MutationLayoutSHOULDREMOVE)
    )


def _call_innner(op):
    ranges = op.data.ranges
    syms = _index_syms(len(ranges))
    store_fn = op.get_store_function()
    if isinstance(op.data, Reduction):
        r_ranges = op.data.reduction_ranges
        r_syms = tuple(sympy.Symbol(f"_r{k}") for k in range(len(r_ranges)))
        store_fn(syms, r_syms)
    else:
        store_fn(syms)


def is_spyre_layout(layout: Layout):
    return layout.device.type == DEVICE_NAME


def _get_stl(
    graph: GraphLowering, node: IRNode | Layout | SpyreTensorLayout
) -> SpyreTensorLayout | None:
    if isinstance(node, ComputedBuffer) and isinstance(
        node.layout, MutationLayoutSHOULDREMOVE
    ):
        target = node.layout.target
        while isinstance(target, ReinterpretView):
            target = target.data
        target_name = target.get_name()
        node = graph.get_buffer(target_name)
    if isinstance(node, IRNode):
        layout = node.maybe_get_layout()
        if layout is None:
            # FallbackKernels and others
            return None
        node = layout
    if isinstance(node, Layout):
        assert isinstance(node, FixedLayout), f"Expected finalized layout, got {node}"
        if not isinstance(node, FixedTiledLayout):
            assert not is_spyre_layout(node)
            return None
        node = node.device_layout
    return node


def is_padded(graph: GraphLowering, node: IRNode | Layout | SpyreTensorLayout) -> bool:
    stl = _get_stl(graph, node)
    if stl is None:
        return False

    total_size = math.prod(stl.device_size)
    if not stl.valid_elements:
        # TODO: the constructor that creates empty valid_elements is
        # being deprecated. Remove this if afterwards.
        numel = _host_numel(node)
        return numel is None or numel < total_size

    total_valid_elems = math.prod(stl.valid_elements.values())
    return total_valid_elems < total_size


def _host_numel(node: IRNode | Layout) -> int | None:
    size = node.size if isinstance(node, Layout) else node.get_size()
    numel = sympy_product(size)
    return int(numel) if numel.is_Integer else None


def _input_padding_range(layout: SpyreTensorLayout) -> Set:
    """Possible values in the padding of a padded graph input.

    A graph input may have been DMA'd from the host or written by an earlier
    graph or eager kernel, so the padding is only known when the runtime
    recorded a constant padding value in the tensor's layout. The layout,
    padding state included, is part of the dynamo guard, so a kernel compiled
    for a known padding value is not reused for a tensor without one.
    """
    if (value := layout.padding_value) is not None:
        return FiniteSet(value)
    return _user_values()


_ASSUMED_PADDING: dict[type, Set] = {
    # Graph inputs (InputBuffer) are handled by _input_padding_range.
    # Each collective behaves a bit differently wrt. to padding,
    # but the fact is that we don't know what is being read from
    # other processes, so for now it's better to be conservative.
    SpyreCollectiveKernel: _user_values(),
    WaitKernel: _user_values(),
    SpyreEmptyFallback: _user_values(),
    MultiOutput: _user_values(),
    FallbackKernel: _user_values(),
}


def _single_padding_value(pad_set: Set) -> float | None:
    """The padding value pad_set proves, or None if it is not one finite value."""
    if isinstance(pad_set, FiniteSet) and len(pad_set) == 1:
        (value,) = pad_set
        if value.is_finite and value.is_real:
            return float(value)
    return None


def get_or_pick_pad_value(inv_domain: InvalidDomain, dtype: torch.dtype):
    pad_value = get_pad_value(inv_domain.func, dtype)
    if pad_value is None or not inv_domain.domain.contains(pad_value):
        pad_value = pick_element(inv_domain.domain)
    return pad_value


def track_padding_value(graph: GraphLowering) -> None:
    operations = graph.operations
    tracer = DomainRangeTracer(graph)

    with V.set_ops_handler(SetDtypeTracer(tracer)):
        for op in list(operations):
            decision = tracer.run_node(op)
            if decision.do_mask:
                if decision.output_padding is not None:
                    # set the propagated output padding
                    op.layout.device_layout.padding_value = decision.output_padding
                op.input_mask_values = decision.input_vals
            elif decision.do_overwrite:
                # TODO: if coord masking is not available, the padding must be overwritten.
                # For now there are no passes to do that, so we log a warning
                logger.warning(
                    "Operation input buffer may contain incompatible values in the padding. "
                    "op=%s",
                    op.get_operation_name(),
                )

            tracer.handle_mutation_output(op)

    _record_mutated_input_padding(graph, tracer)


def _can_record_padding(layout: SpyreTensorLayout) -> bool:
    """Whether layout can record a constant padding (set_constant_padding)."""
    return layout.element_arrangement == ElementArrangement.STANDARD and (
        layout.device_dtype in (DataFormats.SEN169_FP16, DataFormats.IEEE_FP32)
    )


def _record_mutated_input_padding(
    graph: GraphLowering, tracer: DomainRangeTracer
) -> None:
    """Record the padding each graph input is left with by its mutations.

    A graph input is a runtime tensor whose layout carries a recorded padding
    value that later graphs trust (see _input_padding_range). A graph that
    writes an input must update that record, or it keeps claiming the value
    from before the write.

    The wrapper decides which inputs to update: every Spyre input in
    graph.mutated_inputs, after its last use (see
    _SpyreWrapperCodegenMixin._input_padding_update). This only supplies the
    value, so it lists just the inputs whose final padding is proven to be one
    value; the wrapper clears any other mutated input (unknown padding, a
    fallback-kernel mutation, or a layout that cannot record a value).
    """
    padding: dict[str, float] = {}
    for name in graph.graph_input_names:
        pad_set = tracer.buffer_range.get(name)
        stl = _get_stl(graph, graph.get_buffer(name))
        if pad_set is None or stl is None or not _can_record_padding(stl):
            continue
        if (value := _single_padding_value(pad_set)) is not None:
            padding[name] = value
    graph._spyre_input_padding = padding  # type: ignore[attr-defined]
