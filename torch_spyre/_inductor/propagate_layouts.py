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


from collections import Counter
from typing import NamedTuple

import logging
import math

import sympy
import torch
from .logging_utils import get_inductor_logger
from torch._inductor.ir import (
    ComputedBuffer,
    DeviceCopy,
    ExternKernel,
    FallbackKernel,
    FixedLayout,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    MultiOutput,
    ReinterpretView,
    Operation,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.virtualized import V

from . import config
from torch_spyre._C import (
    DataFormats,
    ElementArrangement,
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
)
from .dtype_ops import (
    bool_equivalent_dtype,
    bool_layout_dtype,
    resolve_output_formats,
)
from .errors import Unsupported
from .constants import (
    BATCH_MATMUL_OP,
    BATCH_MATMUL_FP8_OP,
    CONV2D_FWD_OP,
    COPY_BACK_CANDIDATE_ATTR,
    DEVICE_NAME,
    ELIDED_COPY_BACK_ATTR,
    REDUCTIONS_NON_STICK_DIM_ONLY,
    STAGGERED_EAS,
)
from .ir import (
    AllGatherAsyncFallback,
    AllReduceAsyncFallback,
    FixedTiledLayout,
    SpyreConstantFallback,
    SpyreEmptyFallback,
    BroadcastAsyncFallback,
    WaitWorkFallback,
)
from .pass_utils import (
    compute_restickify_target_layout,
    concretize_expr,
    find_matmul_generated_var,
    find_reduction_var,
    get_matmul_n_size,
    identify_matmul_inputs,
    host_coordinates,
    device_coordinates,
    try_device_coordinates,
    indirect_info_from_op,
    is_keep_by_index,
    is_stick_expr_offset_free,
    is_topk,
    iter_var_id,
)
from .optimize_restickify import AllSameNode, AnyInNode, FixedInOutNode
from .views import compute_coordinates, matching_dim

# ---------------------------------------------------------------------------
# TODO(issue#1371): once SpyreTensorLayout is migrated to c10::SymInt, all
# concretize_expr calls in this file can be removed.
# ---------------------------------------------------------------------------

logger = get_inductor_logger("propagate_layouts")


prims = torch.ops.prims
aten = torch.ops.aten
spyreop = torch.ops.spyre


class PropArg(NamedTuple):
    """Input arg during layout propagation.

    layout is the host FixedLayout (may not be FixedTiledLayout until finalize_layouts).
    layouts is the set of candidate device layouts being propagated.
    """

    dep: MemoryDep
    layout: FixedLayout
    layouts: list[SpyreTensorLayout]


def _get_prop_args(reads) -> list[PropArg]:
    # Local to this pass — the FixedLayout/FixedTiledLayout ambiguity only exists
    # during propagation and should not infect downstream passes.
    res: list[PropArg] = []
    for arg in reads:
        if isinstance(arg, MemoryDep):
            buf = V.graph.get_buffer(arg.name)
            layout = buf.get_layout()
            # Skip 0-d scalar constants — they have no meaningful STL to propagate.
            if isinstance(buf, SpyreConstantFallback) and not layout.size:
                continue
            # SpyreEmptyFallback has no device layout until its mutation writers
            # have run. If it already has .layouts (assigned by the SpyreEmptyFallback
            # branch below), include it as a normal input so downstream consumers
            # (e.g. mul reading the mutation target buffer) see the stick constraint.
            if isinstance(buf, SpyreEmptyFallback) and not hasattr(buf, "layouts"):
                continue
            if hasattr(buf, "layouts"):
                res.append(PropArg(arg, layout, list(buf.layouts)))
            else:
                if not isinstance(layout, FixedTiledLayout):
                    raise RuntimeError(f"{buf} does not have FixedTiledLayout")
                res.append(PropArg(arg, layout, [layout.device_layout]))
    return res


def same_device_size(t1: torch.dtype, t2: torch.dtype) -> bool:
    return get_elem_in_stick(t1) == get_elem_in_stick(t2)


def infer_bool_device_dtype(args: list[PropArg]) -> DataFormats:
    """Infer the on-device format for a torch.bool pointwise output.

    Spyre has no native bool: a bool tensor reuses the physical format of its
    producing operands. fp16 operands produce SEN169_FP16 bool (64 elems/stick);
    fp32 operands produce IEEE_FP32 bool (32 elems/stick). This must match the
    operand format because multi-arg pointwise ops require all args to share the
    same stick element size.

    Type promotion guarantees that all pointwise operands share one format, so
    any operand's device layout reveals the output format.
    """
    formats = {stl.device_dtype for arg in args for stl in arg.layouts}
    if len(formats) != 1 or bool_equivalent_dtype(next(iter(formats))) is None:
        raise Unsupported(
            f"torch.bool result from operands with device format(s) {formats}"
        )
    (device_dtype,) = formats
    return device_dtype


def _compute_dim_order(stick_dim, size, coords):
    """Order dimensions with stick_dim last, placing size-one dimensions to the right to avoid tiling."""
    dim_order = [d for d in range(len(size)) if d != stick_dim and coords[d] != 0]
    dim_order += [d for d in range(len(size)) if d != stick_dim and coords[d] == 0]
    dim_order += [stick_dim]
    return dim_order


def _pick_stick_dim(stick_expr, out_coords) -> int:
    """Map a stick expression to an output dimension index, or -1 if it doesn't survive."""
    maybe = matching_dim(out_coords, stick_expr)
    return -1 if maybe is None else maybe


def _output_stl_from_stick_expr(
    stick_expr, output, output_dep, c_size, c_stride, dtype=None
) -> SpyreTensorLayout | None:
    """If stick_expr is offset-free, build an output STL with it mapped to the right dim.

    Returns None if stick_expr has an offset (caller should fall back to scanning).
    """
    dtype = output.dtype if dtype is None else dtype
    stick_size = get_elem_in_stick(dtype)
    if not is_stick_expr_offset_free(stick_expr, stick_size):
        return None
    out_coords = host_coordinates(output, output_dep, None)
    out_stick_dim = _pick_stick_dim(stick_expr, out_coords)
    return _make_output_stl(output, output_dep, c_size, c_stride, out_stick_dim, dtype)


def _dims_by_alignment(dims, sizes, stick_size: int) -> tuple[list[int], list[int]]:
    """Split ``dims`` into (aligned, unaligned) by their extent in ``sizes``.

    A caller scans the aligned dims first (no padding needed) and only falls
    back to the unaligned dims when the aligned group yields nothing; the
    unaligned dim it then picks is padded up to a stick boundary later by
    ``insert_restickify_padding`` (See #1756).
    """
    aligned, unaligned = [], []
    for dim in dims:
        if concretize_expr(sizes[dim]) % stick_size == 0:
            aligned.append(dim)
        else:
            unaligned.append(dim)
    return aligned, unaligned


def _make_output_stl(
    output, output_dep, c_size, c_stride, stick_dim, dtype=None
) -> SpyreTensorLayout | None:
    """Build a candidate output STL with stick_dim last and verify the resulting stick is offset-free.

    Returns None if the resulting stick expression has an offset.
    """
    dtype = output.dtype if dtype is None else dtype
    stick_size = get_elem_in_stick(dtype)
    if stick_dim >= 0 and c_size[stick_dim] == 1:
        return None
    out_coords = host_coordinates(output, output_dep, None)
    dim_order = _compute_dim_order(stick_dim, c_size, out_coords)
    stl = SpyreTensorLayout(c_size, c_stride, dtype, dim_order)
    coords = device_coordinates(stl, output_dep, None)
    if is_stick_expr_offset_free(coords[-1], stick_size):
        return stl
    return None


def _candidate_output_stls(
    output: FixedLayout,
    output_dep: MemoryDep,
    c_size: list,
    c_stride: list,
    skip_stick_expr: sympy.Expr,
    dtype=None,
) -> list[SpyreTensorLayout]:
    """Enumerate candidate output STLs by trying each dim as the stick.

    Skip the dim that already produces an unsupported stick.
    """
    out_coords = host_coordinates(output, output_dep, None)
    skip_dim = _pick_stick_dim(skip_stick_expr, out_coords)

    dtype = output.dtype if dtype is None else dtype
    stick_size = get_elem_in_stick(dtype)
    # Prefer stick-aligned dims; fall back to unaligned dims (padded later by
    # insert_restickify_padding) only when no aligned dim yields a candidate.
    all_dims = [d for d in range(len(output.size)) if d != skip_dim]
    aligned_dims, unaligned_dims = _dims_by_alignment(all_dims, output.size, stick_size)
    stls: list[SpyreTensorLayout] = []
    for dims in (aligned_dims, unaligned_dims):
        for d in dims:
            stl = _make_output_stl(output, output_dep, c_size, c_stride, d, dtype)
            if stl is not None:
                stls.append(stl)
        if stls:
            break
    return stls


def _check_supported_input_sticks(args: list[PropArg], op_label: str) -> None:
    """Reject fixed-layout ops when any input has a stick expression with a constant offset.

    These ops require a fixed input layout.  An offset stick would need two
    restickify ops — one to remove the offset before the op, and one to restore
    the layout after — which is not yet implemented.
    """
    for i, arg in enumerate(args):
        representable = 0
        for stl in arg.layouts:
            coords = try_device_coordinates(stl, arg.dep, None)
            if coords is None:
                # This candidate layout has a stick expression the backend
                # cannot represent (e.g. floor(var/N) from a cross-stick
                # access). It is not a usable candidate, so skip it rather than
                # aborting — another candidate for this input may be valid.
                continue
            representable += 1
            stick_expr = coords[-1]
            if not is_stick_expr_offset_free(stick_expr, stl.elems_per_stick()):
                raise Unsupported(
                    f"{op_label}: input arg{i} has stick expression with offset "
                    f"{stick_expr!r} (likely from slicing the stick dimension); "
                    f"this op requires a fixed input layout and double-restickify is not yet supported"
                )
        if arg.layouts and representable == 0:
            # Every candidate layout for this input was unrepresentable. This
            # is not fatal here, but the downstream layout selection will fail
            # (find_stick_compatible_input_layout raises "cannot restickify any
            # input layout"). Log the more specific cause so that error is
            # easier to diagnose.
            logger.warning(
                "%s: all %d candidate layout(s) of input arg%d have "
                "unrepresentable stick expressions; downstream layout "
                "selection is expected to fail for this op.",
                op_label,
                len(arg.layouts),
                i,
            )


def _rescale_stl_for_dtype(
    stl: SpyreTensorLayout,
    out_dtype: torch.dtype,
    ea: ElementArrangement,
) -> SpyreTensorLayout:
    """Propagate a device layout across a same-shape, differing-stick-depth dtype conversion.

    Copies the input STL's ``device_size``/``stride_map`` and rescales the stick
    depth (the last device dim) plus, when present, the one non-stick dim whose
    stride equals the input stick depth. This preserves any non-canonical layout
    or padding present in the input STL instead of reconstructing a dense layout
    from the logical size/stride.

    The input elements-per-stick is read from ``stl.device_size[-1]`` (the stick
    dimension is always full, so it equals ``get_elem_in_stick(in_dtype)``); the
    output count comes from ``out_dtype``.

    Args:
        stl: Input device layout to rescale.
        out_dtype: Torch dtype of the conversion output.
        ea: ElementArrangement to stamp on the returned layout.
    """
    in_eps = stl.device_size[-1]
    out_eps = get_elem_in_stick(out_dtype)
    out_device_size = list(stl.device_size)
    out_stride_map = list(stl.stride_map)
    out_device_size[-1] = out_eps
    # Rescale the first non-stick dim that indexes whole sticks (stride == the
    # input stick depth) by the stick-depth ratio. A staggered/sparse layout
    # (e.g. the DL16_TO_FP32 restoration operand, whose stride_map carries
    # sentinel -1 entries rather than a linear num-sticks stride) has no such
    # dim; there only the stick depth changes, so a no-match is expected and
    # left as-is.
    for i, s in enumerate(stl.stride_map):
        if s == in_eps:
            out_device_size[i] = stl.device_size[i] * in_eps // out_eps
            out_stride_map[i] = out_eps
            break
    return SpyreTensorLayout(
        out_device_size,
        out_stride_map,
        get_device_dtype(out_dtype),
        ea,
    )


def _qfp8wt_stl(
    output: FixedLayout,
    in_layout: FixedLayout,
) -> SpyreTensorLayout:
    """Propagate special FP8 tensor with 2D stick [2, 64]

    The QFP8WT weight (KERNEL) tensor requires stick dimensions of [2, 64].  This
    function propagates the output layout for both aligned and unaligned input tensors.

    Outer dimensions and strides are preserved from output layout, with the innermost
    stride always set to 1. For unaligned inputs where input stick size is not a
    multiple of input stick size, the last dimension is padded to the stick size.  For
    aligned inputs, it retains the concrete output size.

    Args:
        output: Inductor ``FixedLayout`` for the op's output tensor.
        in_layout: Inductor ``FixedLayout`` for the op's input tensor.
    """
    in_eps = get_elem_in_stick(in_layout.dtype)
    stick_dim_size = in_layout.size[-1]
    unaligned = stick_dim_size % in_eps
    outer_sizes = [concretize_expr(s) for s in output.size[:-1]]
    outer_strides = [concretize_expr(s) for s in output.stride[:-1]]
    last_dim = in_eps if unaligned > 0 else concretize_expr(output.size[-1])
    c_size = outer_sizes + [last_dim]
    c_stride = outer_strides + [1]
    return SpyreTensorLayout(
        c_size,
        c_stride,
        output.dtype,
        list(range(len(c_size))),
        ElementArrangement.QFP8WT,
    )


def _single_arg_op_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    dep: MemoryDep,
    in_layout: FixedLayout,
    stl: SpyreTensorLayout,
) -> list[SpyreTensorLayout]:
    """Compute output STL(s) for one candidate input STL of a single-arg op.

    Called once per candidate input STL to produce corresponding output STL(s).
    An empty result means this candidate cannot produce a supported output layout;
    callers continue trying the other input STLs and fail only if none are usable.
    """
    data = op.data
    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]

    if isinstance(data, Reduction):
        # Bool physical format resolution: see bool_layout_dtype's docstring.
        out_dtype_for_layout = (
            bool_layout_dtype(stl.device_dtype)
            if output.dtype == torch.bool
            else output.dtype
        )
        stick_size = get_elem_in_stick(out_dtype_for_layout)

        x_dev_coords = try_device_coordinates(stl, dep, None)
        if x_dev_coords is None:
            return []
        x_stick_expr = x_dev_coords[-1]
        reduction_var = next(
            iter(dep.index.free_symbols - output_dep.index.free_symbols), None
        )

        # Do not preserve the input layout for reduction ops listed in
        # REDUCTIONS_NON_STICK_DIM_ONLY when reducing along the stick
        # dimension. Those reductions are currently unsupported in the backend.
        # See backend issue #4409.
        if not (
            data.reduction_type in REDUCTIONS_NON_STICK_DIM_ONLY
            and reduction_var in x_stick_expr.free_symbols
        ):
            # Try to preserve input layout
            out_stl = _output_stl_from_stick_expr(
                x_stick_expr, output, output_dep, c_size, c_stride, out_dtype_for_layout
            )
            if out_stl is not None:
                return [out_stl]

        # Try alternative layouts when input layout is not supported.
        # Skip the dim already known to produce an unsupported stick.
        in_coords = host_coordinates(in_layout, dep, None)
        out_coords = host_coordinates(output, output_dep, None)
        skip_in_dim = matching_dim(in_coords, x_stick_expr)

        # Prefer stick-aligned input dims; fall back to unaligned dims (padded
        # later by insert_restickify_padding) only when no aligned dim maps to a
        # supported output stick.
        all_dims = [d for d in range(len(in_layout.size)) if d != skip_in_dim]
        aligned_dims, unaligned_dims = _dims_by_alignment(
            all_dims, in_layout.size, stick_size
        )
        layouts: list[SpyreTensorLayout] = []
        for dims in (aligned_dims, unaligned_dims):
            for in_dim in dims:
                in_coord = in_coords[in_dim]
                # Map input dim to output dim. If input dim carries reduction
                # var, it's collapsed
                if reduction_var is not None and reduction_var in in_coord.free_symbols:
                    out_stick_dim = -1
                else:
                    out_stick_dim = _pick_stick_dim(in_coord, out_coords)
                    if out_stick_dim < 0:
                        continue
                out_stl = _make_output_stl(
                    output,
                    output_dep,
                    c_size,
                    c_stride,
                    out_stick_dim,
                    out_dtype_for_layout,
                )
                if out_stl is not None:
                    layouts.append(out_stl)
            if layouts:
                break
        return layouts

    # Single-arg pointwise
    assert isinstance(data, Pointwise)
    origin_node = next(iter(data.origins))
    aten_op = origin_node.target
    match aten_op:
        case prims.convert_element_type.default | aten.copy.default if (
            output.dtype != torch.bool
            and stl.elems_per_stick() != get_elem_in_stick(output.dtype)
        ):
            # Type conversion may require padding when input has padding due to stick
            # alignment. For example, 4x16 FP16 has 48 elements of padding (64 total),
            # which becomes 64 FP32 elements when converted. We need to reflect this
            # in the output host size so the constructor creates the correct device layout.
            try:
                in_stick_expr = device_coordinates(stl, dep, None)[-1]
            except Unsupported:
                # Staggered-EA candidate whose physical stick depth differs from
                # elems_per_stick — not a valid input for this conversion path.
                return []
            if not is_stick_expr_offset_free(in_stick_expr, stl.elems_per_stick()):
                return []

            input_ea = stl.element_arrangement

            # Determine output EA based on conversion direction and input EA
            if (
                in_layout.dtype in (torch.float16, torch.bfloat16)
                and output.dtype == torch.float32
            ):
                # FP16/BF16 → FP32 conversion. Both share SEN169_FP16 physical
                # storage and the DL16TOFP32 hardware op.
                if input_ea == ElementArrangement.STANDARD:
                    # Case 1: STANDARD → DL16_TO_FP32 (creates staggered layout)
                    fmt = ElementArrangement.DL16_TO_FP32
                elif input_ea == ElementArrangement.FP32_TO_DL16:
                    # Case 2: FP32_TO_DL16 → STANDARD (restoration)
                    fmt = ElementArrangement.STANDARD
                else:
                    # Unexpected input EA for FP16→FP32
                    raise Unsupported(
                        f"FP16→FP32 conversion with unsupported input EA: {input_ea}"
                    )
            elif in_layout.dtype == torch.float32 and output.dtype == torch.float16:
                # FP32 → FP16 conversion
                if input_ea == ElementArrangement.STANDARD:
                    # Case 3: STANDARD → FP32_TO_DL16 (creates staggered layout)
                    fmt = ElementArrangement.FP32_TO_DL16
                elif input_ea == ElementArrangement.DL16_TO_FP32:
                    # Case 4: DL16_TO_FP32 → STANDARD (restoration)
                    fmt = ElementArrangement.STANDARD
                else:
                    # Unexpected input EA for FP32→FP16
                    raise Unsupported(
                        f"FP32→FP16 conversion with unsupported input EA: {input_ea}"
                    )
            else:
                # Other type conversions default to STANDARD
                fmt = ElementArrangement.STANDARD

            # Two strategies, chosen by whether a staggered EA is involved:
            #
            # 1. Staggered conversions (RMSNorm up/down-cast and their
            #    restoration: STANDARD<->DL16_TO_FP32 / FP32_TO_DL16). The
            #    staggered element ordering only exists on the physical device
            #    layout, so we must propagate the input's device_size/stride_map
            #    and rescale just the stick depth via _rescale_stl_for_dtype.
            #    Reconstructing from the logical host size would lose it.
            #
            # 2. Plain conversions (e.g. fp8->fp16 after qfp8ch). Here the input
            #    device layout can be degenerate — qfp8ch rescales a size-1
            #    num-sticks dim to 0 (1*64//128), leaving a size-0 dim — and
            #    _rescale_stl_for_dtype would faithfully propagate that garbage,
            #    changing the layout rank and downstream graph partitioning.
            #    Rebuild a clean dense layout from the output host size instead,
            #    as the general (non-EA) convert path does.
            if fmt in STAGGERED_EAS or input_ea in STAGGERED_EAS:
                return [_rescale_stl_for_dtype(stl, output.dtype, fmt)]

            # Dense reconstruction from the output host size. When the input
            # stick dim is unaligned, force a full input-stick depth so stick
            # padding is reflected in the device layout (see #1756 example above).
            in_elems_per_stick = get_elem_in_stick(in_layout.dtype)
            if concretize_expr(in_layout.size[-1] % in_elems_per_stick) > 0:
                c_size = [concretize_expr(s) for s in output.size[:-1]] + [
                    in_elems_per_stick
                ]
                c_stride = [concretize_expr(s) for s in output.stride[:-1]] + [1]
            return [
                SpyreTensorLayout(
                    c_size, c_stride, output.dtype, list(range(len(c_size))), fmt
                )
            ]

        case spyreop.qfp8ch.default:
            # fp16 (64 elems/stick) -> fp8 (128 elems/stick) quantization.
            # Propagate the input device layout and rescale for the dtype change,
            # preserving any padding present in the input STL.
            return [
                _rescale_stl_for_dtype(stl, output.dtype, ElementArrangement.QFP8CH)
            ]

        case spyreop.qfp8wt.default:
            # fp16 -> fp8 weight quantization with 2D-stick layout [2, 64].
            return [_qfp8wt_stl(output, in_layout)]

    # For a bool output this op is dtype-preserving data movement or a
    # same-width reinterpret (reshape/transpose/etc); reuse the input STL's
    # physical format directly.
    out_device_dtype, out_dtype_for_layout = resolve_output_formats(
        output.dtype, stl.device_dtype
    )

    in_coords = host_coordinates(in_layout, dep, None)
    out_coords = host_coordinates(output, output_dep, None)
    if (
        in_coords == out_coords
        and in_layout.size == output.size
        and dep.index == output_dep.index
        and same_device_size(in_layout.dtype, out_dtype_for_layout)
    ):
        # Input and output tensors are being accessed identically and elem size is the same.
        # We can simply propagate the device_layout including ElementArrangement.
        stl = SpyreTensorLayout(
            stl.device_size,
            stl.stride_map,
            out_device_dtype,
            stl.element_arrangement,
        )
        return [stl]

    in_device_coords = try_device_coordinates(stl, dep, None)
    if in_device_coords is None:
        return []
    stick_expr = in_device_coords[-1]

    # Try to preserve input layout, fall back to scanning all output dims
    out_stl = _output_stl_from_stick_expr(
        stick_expr, output, output_dep, c_size, c_stride, out_dtype_for_layout
    )
    if out_stl is not None:
        return [out_stl]
    return _candidate_output_stls(
        output,
        output_dep,
        c_size,
        c_stride,
        stick_expr,
        out_dtype_for_layout,
    )


def _clone_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[PropArg],
) -> list[SpyreTensorLayout]:
    """
    Clone is generated by an explicit `contiguous()`; on Spyre that means use the default row-major tiling.

    Case 1: Input has supported stick expression
      - No restickify insertion needed
      - Clone op becomes identity if input is already row-major, otherwise becomes restickify

    Case 2: Input has unsupported stick expression (due to offset)
      - Insert restickify before clone to swap stick with non-stick dimension
      - Clone op also becomes restickify op to swap dimensions back
      - The second restickify handles the tensor with offset
    """
    data = op.data

    assert isinstance(data, Pointwise)
    origin_node = next(iter(data.origins))
    aten_op = origin_node.target
    assert aten_op == aten.clone.default

    in_dep = args[0].dep
    in_stl = next(iter(args[0].layouts))
    in_device_coords = device_coordinates(in_stl, in_dep, None)
    stick_expr = in_device_coords[-1]

    # Clone preserves physical format, so a bool output must match its
    # input's -- substitute the equivalent logical dtype since
    # get_device_dtype(torch.bool) can't express that.
    if output.dtype == torch.bool:
        dtype_for_layout = bool_layout_dtype(in_stl.device_dtype, "clone")
    else:
        dtype_for_layout = output.dtype
    stick_size = get_elem_in_stick(dtype_for_layout)

    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]
    out_stl = SpyreTensorLayout(
        c_size, c_stride, dtype_for_layout, list(range(len(output.size)))
    )

    if is_stick_expr_offset_free(stick_expr, stick_size):
        # Case 1: No restickify insertion needed.
        # Use AnyInNode to produce the fixed output layout.
        op.restick_cost_fn = AnyInNode.from_args()
        return [out_stl]

    # Case 2: Find alternative dimension to swap with the current stick dimension.
    # TODO: FixedInOutNode only supports a single required STL, so we select
    # a layout where restickify is feasible to avoid optimizer rejection.
    # Consider implementing a cost node that supports multiple required STLs.
    out_coords = host_coordinates(output, output_dep, None)
    in_layout = args[0].layout
    in_host_coords = host_coordinates(in_layout, in_dep, None)
    required_in_stl = None
    for candidate in _candidate_output_stls(
        output, output_dep, c_size, c_stride, stick_expr, dtype_for_layout
    ):
        target_stick = device_coordinates(candidate, output_dep, None)[-1]
        target_stl = compute_restickify_target_layout(
            in_stl, in_layout, target_stick, in_host_coords, in_device_coords
        )
        if target_stl is not None:
            required_in_stl = target_stl
            break

    if not required_in_stl:
        raise Unsupported(
            f"No supported layout found for stick expression {stick_expr!r}. "
            f"Cannot find alternative layout with size={output.size} and coordinates={out_coords}"
        )

    op.restick_cost_fn = FixedInOutNode.from_args(args, out_stl, [required_in_stl], op)
    return [out_stl]


def _exx2_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[PropArg],
) -> list[SpyreTensorLayout]:
    """exx2 requires its input stick on the reduction dim (= last logical dim).
    Use FixedInOutNode to schedule a restickify if the input stick is elsewhere.
    """
    _check_supported_input_sticks(args, "exx2")
    x = args[0]
    out_dim_order = list(range(len(output.size))) + [-1]
    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]
    out_stl = SpyreTensorLayout(
        c_size, c_stride, output.dtype, out_dim_order, ElementArrangement.EXX2
    )
    reduction_var = find_reduction_var((x.dep,), output_dep)
    req_in_stl = find_stick_compatible_input_layout(x, reduction_var, "exx2", "x")
    op.restick_cost_fn = FixedInOutNode.from_args(args, out_stl, [req_in_stl], op)
    return [out_stl]


def _layernormnorm_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[PropArg],
) -> list[SpyreTensorLayout]:
    """layernormnorm requires x's stick to match mean/norm_mean (= last logical dim).
    Use FixedInOutNode to schedule a restickify if x's stick is elsewhere.
    """
    _check_supported_input_sticks(args, "layernormnorm")
    x = args[0]
    out_dim_order = list(range(len(output.size)))
    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]
    out_stl = SpyreTensorLayout(c_size, c_stride, output.dtype, out_dim_order)
    reduction_var = find_reduction_var((x.dep,), output_dep)
    req_in_stl = find_stick_compatible_input_layout(
        x, reduction_var, "layernormnorm", "x"
    )
    op.restick_cost_fn = FixedInOutNode.from_args(args[:1], out_stl, [req_in_stl], op)
    return [out_stl]


def _dev_coord_for_var(dev_coords, arg_host_coords, var):
    """Return the first device coord that carries var and is resolvable via matching_dim."""
    for c in dev_coords:
        if var in c.free_symbols and matching_dim(arg_host_coords, c) is not None:
            return c
    return None


def _find_layout_avoiding_var_on_stick(
    arg: PropArg,
    avoid_var: sympy.Symbol,
    label: str,
) -> SpyreTensorLayout:
    """Find required STL for ``arg`` whose stick does not carry ``avoid_var``.

    1. Return the first layout whose stick already excludes avoid_var (zero cost).
    2. Else return the first layout that can be restickified to move avoid_var
       off the stick, onto a surviving coordinate.
    3. Else raise Unsupported.
    """
    constant_stick = None
    for stl in arg.layouts:
        dev_coords = device_coordinates(stl, arg.dep, None)
        if dev_coords is None:
            continue
        stick_vars = dev_coords[-1].free_symbols
        if avoid_var not in stick_vars:
            if stick_vars:
                return stl
            constant_stick = stl

    if constant_stick is not None:
        return constant_stick

    arg_host_coords = host_coordinates(arg.layout, arg.dep, None)
    surviving_vars = set()
    for coord in arg_host_coords:
        if len(coord.free_symbols) > 0 and avoid_var not in coord.free_symbols:
            surviving_vars.update(coord.free_symbols)

    if not surviving_vars:
        raise Unsupported(
            f"{label}: no surviving coordinates after removing {avoid_var}"
        )
    surviving_var = next(iter(surviving_vars))

    for stl in arg.layouts:
        dev_coords = device_coordinates(stl, arg.dep, None)
        if dev_coords is None:
            continue
        target_stick_expr = _dev_coord_for_var(
            dev_coords, arg_host_coords, surviving_var
        )
        if target_stick_expr is None:
            continue
        result = compute_restickify_target_layout(
            stl, arg.layout, target_stick_expr, arg_host_coords, dev_coords
        )
        if result is not None:
            return result

    raise Unsupported(
        f"{label}: cannot restickify layout to move {avoid_var} off the stick"
    )


def _canonical_stl_from_collapsed_host(
    arg: PropArg,
    matmul_var: sympy.Symbol,
    reduction_type: str,
    label: str,
) -> SpyreTensorLayout:
    """Build the canonical STL for a matmul input by collapsing host dims.

    Constructs the canonical ``SpyreTensorLayout`` directly from the host
    geometry, collapsing any mixed-radix dims that jointly carry ``matmul_var``
    into one logical dim.  This is the single authoritative path for Pass 3.

    Single-host-dim case (Case A): the host already has exactly one dim for
    ``matmul_var``; build the STL directly from it (``compute_restickify_target_layout``
    can produce different results depending on candidate order, so we bypass it).

    Multi-host-dim case (Case B): validate the chain is affine, full-range,
    and contiguous, then merge the dims.

    Raises ``Unsupported`` on invalid chains.  Does not call ``matching_dim`` or
    ``compute_restickify_target_layout``.
    """
    host_coords = host_coordinates(arg.layout, arg.dep, None)
    host_dims = [
        dim for dim, coord in enumerate(host_coords) if matmul_var in coord.free_symbols
    ]
    host_size = [concretize_expr(size) for size in arg.layout.size]
    host_stride = [concretize_expr(stride) for stride in arg.layout.stride]

    if len(host_dims) == 1:
        # Case A: single host dim — build STL from host size/stride directly,
        # with that dim as the stick.
        stick_dim = host_dims[0]
        canonical_size = host_size
        canonical_stride = host_stride
    else:
        zeroed_index = arg.dep.index.xreplace({matmul_var: sympy.S.Zero})
        var_delta = sympy.expand(arg.dep.index - zeroed_index)
        var_stride = sympy.expand(var_delta.coeff(matmul_var))
        var_range = concretize_expr(arg.dep.ranges[matmul_var])

        carrier_dims = [dim for dim in host_dims if host_size[dim] > 1]
        affine_full_range = (
            len(carrier_dims) > 1
            and not var_stride.free_symbols
            and var_stride.is_Integer
            and var_stride > 0
            and sympy.simplify(var_delta - var_stride * matmul_var) == 0
            and all(
                coord.free_symbols <= {matmul_var}
                and sympy.simplify(coord.xreplace({matmul_var: sympy.S.Zero})) == 0
                for dim, coord in enumerate(host_coords)
                if dim in host_dims
            )
            and math.prod(host_size[dim] for dim in carrier_dims) == var_range
        )
        chain = sorted(carrier_dims, key=lambda dim: host_stride[dim], reverse=True)
        contiguous_chain = bool(chain) and all(
            host_stride[outer] == host_stride[inner] * host_size[inner]
            for outer, inner in zip(chain, chain[1:])
        )
        if (
            not affine_full_range
            or not contiguous_chain
            or host_stride[chain[-1]] != var_stride
        ):
            raise Unsupported(
                f"{reduction_type}: cannot canonicalize factorized {label}_var="
                f"{matmul_var}; expected an affine full-range contiguous host "
                f"dimension chain, got host coordinates {host_coords}, "
                f"size={list(arg.layout.size)}, stride={list(arg.layout.stride)}, "
                f"dep={arg.dep.name}, index={arg.dep.index}, "
                f"ranges={dict(arg.dep.ranges)}"
            )

        # Collapse the mixed-radix host dims into one logical dim, preserving all
        # unrelated dims (including size-one batch dims) to keep rank conventions.
        collapsed_dims = set(host_dims)
        insert_at = min(host_dims)
        canonical_size = []
        canonical_stride = []
        stick_dim = -1
        for dim, (size, stride) in enumerate(zip(host_size, host_stride)):
            if dim == insert_at:
                stick_dim = len(canonical_size)
                canonical_size.append(var_range)
                canonical_stride.append(int(var_stride))
            if dim not in collapsed_dims:
                canonical_size.append(size)
                canonical_stride.append(stride)
        assert stick_dim >= 0

    dim_order = [dim for dim in range(len(canonical_size)) if dim != stick_dim]
    dim_order.append(stick_dim)
    return SpyreTensorLayout(
        canonical_size,
        canonical_stride,
        arg.layout.dtype,
        dim_order,
        ElementArrangement.STANDARD,
    )


def find_stick_compatible_input_layout(
    arg: PropArg,
    reduction_var: sympy.Symbol,
    reduction_type: str,
    label: str,
) -> SpyreTensorLayout:
    """Find the required STL for a matmul input by iterating all candidate layouts.

    1. Return the first layout whose stick already carries reduction_var and no
       outer axis also carries it (zero cost, no restickify needed).
    2. Else return the first layout that can be restickified to put reduction_var
       on the stick via compute_restickify_target_layout.
    3. (BATCH_MATMUL_OP only) Else collapse mixed-radix host dims and construct
       the canonical STL directly.
    4. Else raise Unsupported.
    """
    logger.debug(
        "[find_stick_compatible_input_layout] label=%r reduction_type=%r\n"
        "  arg.dep.name      = %s\n"
        "  reduction_var     = %s\n"
        "  arg.layout.size   = %s\n"
        "  arg.layout.stride = %s\n"
        "  arg.layouts       = %s",
        label,
        reduction_type,
        arg.dep.name,
        reduction_var,
        list(arg.layout.size),
        list(arg.layout.stride),
        arg.layouts,
    )
    # Skip candidates whose stick expression the backend cannot represent
    # (e.g. floor(var/N) from a cross-stick access); they are not usable inputs
    # and another candidate may work.
    candidates = [
        (stl, coords)
        for stl in arg.layouts
        if (coords := try_device_coordinates(stl, arg.dep, None)) is not None
    ]

    # Pass 1: already stick-compatible.
    # stick_compatible() checks cross-tensor compatibility; here we only need
    # to know if this input's stick coord already carries the target loop variable.
    # For BATCH_MATMUL_OP: also reject layouts where the reduction variable appears
    # on any outer axis in addition to the stick (factorized layout, e.g. SDPA's
    # [L, D/64, H, 64]).  The backend treats a variable present on both an outer
    # axis and the stick as two separate contraction dimensions, producing wrong
    # output.  Such candidates fall through to Pass 2 / Pass 3 so a restickify
    # collapses the factorization first.
    # Other reduction types (exx2, layernormnorm, …): the backend correctly handles
    # a tiled layout where the reduction var appears on one outer axis (floor(v/64))
    # and the stick (Mod(v, 64)), so no special check is needed.
    # Non-STANDARD arrangements (QFP8WT etc.) carry their own contraction
    # structure; the backend handles them regardless of device coord shape,
    # so return immediately without checking the stick.
    for stl, dev_coords in candidates:
        if stl.element_arrangement != ElementArrangement.STANDARD:
            return stl
        if reduction_var not in dev_coords[-1].free_symbols:
            continue
        if reduction_type == BATCH_MATMUL_OP and any(
            reduction_var in c.free_symbols for c in dev_coords[:-1]
        ):
            continue
        return stl

    # Pass 2: can be restickified — find the resolvable device coord for reduction_var
    # and use it as target_stick_expr for compute_restickify_target_layout.
    # Skip non-STANDARD arrangements: compute_restickify_target_layout always
    # produces a STANDARD layout and would silently drop the arrangement.
    # For BATCH_MATMUL_OP: skip candidates where reduction_var is on the stick AND
    # on any outer axis (factorized, e.g. SDPA [L,D/64,H,64]).
    # compute_restickify_target_layout cannot correctly collapse this factorization.
    # Such candidates are handled by Pass 3 instead.
    arg_host_coords = host_coordinates(arg.layout, arg.dep, None)
    for stl, dev_coords in candidates:
        if stl.element_arrangement != ElementArrangement.STANDARD:
            continue
        if (
            reduction_type == BATCH_MATMUL_OP
            and reduction_var in dev_coords[-1].free_symbols
            and any(reduction_var in c.free_symbols for c in dev_coords[:-1])
        ):
            continue
        target_stick_expr = _dev_coord_for_var(
            dev_coords, arg_host_coords, reduction_var
        )
        if target_stick_expr is None:
            continue
        result = compute_restickify_target_layout(
            stl, arg.layout, target_stick_expr, arg_host_coords, dev_coords
        )
        if result is not None:
            return result

    # Pass 3 (BATCH_MATMUL_OP only): all candidates were factorized (reduction_var
    # on outer axes), so Pass 2 found nothing.  Build the canonical STL directly
    # from the host geometry, collapsing mixed-radix dims if needed.
    # This fires for:
    #   Case A: single host dim but only factorized STL candidates — Pass 2
    #           correctly skipped them; derive from host geometry instead.
    #   Case B: multiple host dims (mixed-radix view) — collapse and construct.
    # Does NOT fire when Pass 2 returned successfully (normal weight tensors where
    # the generated var is on the stick and reduction_var is only on outer axes in
    # a non-factorized sense).
    if reduction_type == BATCH_MATMUL_OP:
        canonical = _canonical_stl_from_collapsed_host(
            arg, reduction_var, reduction_type, label
        )
        logger.debug(
            "[find_stick_compatible_input_layout] Pass 3: canonical STL"
            " for %r %s_var=%s → %s",
            label,
            label,
            reduction_var,
            canonical,
        )
        return canonical

    raise Unsupported(
        f"{reduction_type}: cannot restickify any input layout of {label} to carry {label}_var={reduction_var}"
    )


def _matmul_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[PropArg],
) -> list[SpyreTensorLayout]:
    """
    Matmul has fixed in/out stick requirements so handled specially.
    Algorithm is
       1. Identify reduction symbol (K) and generated symbol (N) via set arithmetic
          on the free symbols of each input's index expression — no host-dim lookup needed
       2. For both input args, find a required STL with the correct stick symbol
       3. Compute the output STL and construct the FixedInOutNode cost function
    """
    data = op.data
    _check_supported_input_sticks(args, data.reduction_type)
    out_coords = host_coordinates(output, output_dep, None)

    logger.debug(
        "[_matmul_layouts] output (%s):\n"
        "  host size   = %s\n"
        "  host stride = %s\n"
        "  dep ranges  = %s\n"
        "  dep index   = %s\n"
        "  host coords = %s",
        output_dep.name,
        list(output.size),
        list(output.stride),
        dict(output_dep.ranges),
        output_dep.index,
        out_coords,
    )
    if logger.isEnabledFor(logging.DEBUG):
        for i, arg in enumerate(args):
            stl = arg.layouts[0]
            h_coords = host_coordinates(arg.layout, arg.dep, None)
            d_coords = device_coordinates(stl, arg.dep, None)
            logger.debug(
                "[_matmul_layouts] input[%d] (%s):\n"
                "  host size   = %s\n"
                "  host stride = %s\n"
                "  dep ranges  = %s\n"
                "  dep index   = %s\n"
                "  STL         = %s\n"
                "  host coords = %s\n"
                "  dev coords  = %s",
                i,
                arg.dep.name,
                list(arg.layout.size),
                list(arg.layout.stride),
                dict(arg.dep.ranges),
                arg.dep.index,
                stl,
                h_coords,
                d_coords,
            )

    # ReadWrites.reads is an OrderedSet: two semantic operands collapse to one
    # MemoryDep when they are aliases with identical access.  Restore the pair.
    if len(args) == 1:
        args = [args[0], args[0]]

    # identify_matmul_inputs either confirms positional order or falls back to it.
    # Map positionally — object identity breaks for self-alias (same dep object).
    identify_matmul_inputs([a.dep for a in args], output_dep)
    x, y = args[0], args[1]

    # Hardware stick constraints (DF16):
    #   Input1 (x): stick on reduction_var (loop var absent from output)
    #   Input2 (y): stick on generated_var (loop var present in output, absent from x)
    #   Output:     stick on generated_var
    reduction_var = find_reduction_var((x.dep,), output_dep)
    n_size = get_matmul_n_size(op)

    x_req_stl = find_stick_compatible_input_layout(
        x, reduction_var, data.reduction_type, "x"
    )

    if n_size == 1:
        # N has no loop symbol after size-one simplification, so there is no
        # generated_var to discover.  Build an explicit sparse-stick layout for y
        # so K is not mistaken for a second contraction dimension.
        y_dim_order = list(range(len(y.layout.size))) + [-1]
        y_req_stl = SpyreTensorLayout(
            [concretize_expr(s) for s in y.layout.size],
            [concretize_expr(s) for s in y.layout.stride],
            y.layout.dtype,
            y_dim_order,
        )
        # Output stick is on the last host dim (N=1 collapses N to a scalar position).
        out_dims = len(output.size)
        out_stick_dim = out_dims - 1
    else:
        generated_var = find_matmul_generated_var(y.dep, x.dep, output_dep, op)
        y_req_stl = find_stick_compatible_input_layout(
            y, generated_var, data.reduction_type, "y"
        )
        _out_stick_dim = next(
            (i for i, c in enumerate(out_coords) if generated_var in c.free_symbols),
            None,
        )
        if _out_stick_dim is None:
            raise Unsupported(
                f"{data.reduction_type}: generated_var={generated_var} not found "
                f"in output coords {out_coords}"
            )
        out_stick_dim = _out_stick_dim

    out_dims = len(output.size)
    out_dim_order = list(range(out_dims - 2))
    if out_stick_dim == out_dims - 1:
        out_dim_order = out_dim_order + [out_dims - 2, out_dims - 1]
    else:
        out_dim_order = out_dim_order + [out_dims - 1, out_dims - 2]
    # Concretize for C++ SpyreTensorLayout constructor.
    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]

    out_stl = SpyreTensorLayout(c_size, c_stride, output.dtype, out_dim_order)

    op.restick_cost_fn = FixedInOutNode.from_args(
        [x, y],
        out_stl,
        [x_req_stl, y_req_stl],
        op,
    )
    return [out_stl]


def _conv_reduction_var(x: PropArg, reduction_candidates: set) -> sympy.Symbol | None:
    """Pick conv's contraction var (`in`) from the reduction candidates.

    A conv reduces over three loop vars -- the input channel `in` and the two
    kernel taps `ki`/`kj` -- so find_reduction_var (which requires exactly one)
    does not apply.  The layout-relevant contraction is `in`: it is the input
    channel and the activation's stick dim.  `ki`/`kj` are windowed reductions
    folded into the activation's spatial coordinates, never its stick.  So `in`
    is the reduction candidate that appears on the stick of some candidate
    activation layout (NHWC: stick = Mod(in, 64)).
    """
    for stl in x.layouts:
        stick_syms = device_coordinates(stl, x.dep, None)[-1].free_symbols
        hit = reduction_candidates & stick_syms
        if hit:
            return next(iter(hit))
    return None


def _conv_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[PropArg],
) -> list[SpyreTensorLayout]:
    """Layout propagation for the two-input conv2d reduction.

    conv2d has the same input/output stick structure as matmul -- activation
    (x) sticks on the contraction var, weight (y) and output stick on the
    generated var (out-channel) -- so this mirrors _matmul_layouts.  The one
    difference is the contraction var: conv reduces over {in, ki, kj}, and the
    stick-relevant one is `in` (see _conv_reduction_var).
    """
    data = op.data
    _check_supported_input_sticks(args, data.reduction_type)
    out_coords = host_coordinates(output, output_dep, None)

    # No len(args)==1 guard needed: conv activation and weight are never aliased.
    identify_matmul_inputs([a.dep for a in args], output_dep)
    x, y = args[0], args[1]

    reduction_candidates = x.dep.index.free_symbols - output_dep.index.free_symbols
    reduction_var = _conv_reduction_var(x, reduction_candidates)
    if reduction_var is None:
        raise Unsupported(
            f"{data.reduction_type}: could not identify the input-channel "
            f"contraction var among {reduction_candidates}"
        )
    generated_var = find_matmul_generated_var(y.dep, x.dep, output_dep)

    x_req_stl = find_stick_compatible_input_layout(
        x, reduction_var, data.reduction_type, "x"
    )
    y_req_stl = find_stick_compatible_input_layout(
        y, generated_var, data.reduction_type, "y"
    )

    out_stick_dim = next(
        (i for i, c in enumerate(out_coords) if generated_var in c.free_symbols),
        None,
    )
    if out_stick_dim is None:
        raise Unsupported(
            f"{data.reduction_type}: generated_var={generated_var} not found in "
            f"output coords {out_coords}"
        )

    # The output must stick on generated_var (out-channel). Unlike matmul, whose
    # N dim is always in the last two positions, conv's out-channel can sit
    # anywhere (index 1 for an NCHW [mb, out, i, j] output), so move it to the
    # stick (innermost) position explicitly and keep the other dims' order.
    out_dims = len(output.size)
    out_dim_order = [d for d in range(out_dims) if d != out_stick_dim] + [out_stick_dim]
    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]
    out_stl = SpyreTensorLayout(c_size, c_stride, output.dtype, out_dim_order)
    op.restick_cost_fn = FixedInOutNode.from_args(
        [x, y], out_stl, [x_req_stl, y_req_stl], op
    )
    return [out_stl]


def _multi_arg_pointwise_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[PropArg],
) -> list[SpyreTensorLayout]:
    """
    Multi-arg pointwise is a join point so handled specially.
    Algorithm is
       1. Compute set of output stick expressions possible given the input layouts,
          keeping only those that produce a supported stick expression on every input.
       2. Compute an out STL for each; fall back to alternate output dims if none survive.
       3. Determine output ElementArrangement based on input EAs (propagate staggered EA if present)
       4. Construct the AllSameNode cost function since in and out sticks must always match
    """

    # Determine output ElementArrangement from inputs
    # Collect all unique EAs from input SpyreTensorLayouts
    input_eas = set()
    for arg in args:
        if arg.layouts:
            # Get EA from first SpyreTensorLayout (all should have same EA for this input)
            input_eas.add(arg.layouts[0].element_arrangement)

    # Determine output EA and enforce the multi-arg mixed-EA compatibility rule
    # over STAGGERED_EAS = {DL16_TO_FP32, FP32_TO_DL16}. (EXX2 is intentionally
    # out of scope here: it is a reduction ordering handled by the layernorm
    # layouts, and is_ea_compatible in split_multi_ops likewise excludes it from
    # the broadcast pattern. The ops that bypass that check -- layernormnorm /
    # layernormscale carrying EXX2 -- have dedicated layout handlers and never
    # reach this multi-arg-pointwise path, so no EXX2 operand can slip through
    # case 3.1/3.2 here.) The cases are:
    #   1.  all STANDARD                            -> STANDARD output
    #   2a. >1 distinct staggered EA                -> unsupported (raise)
    #   2b. one staggered EA, no STANDARD operands  -> staggered output (via 3.1)
    #   3.  one staggered EA mixed with STANDARD operands:
    #       3.1 every STANDARD operand can broadcast   -> staggered output
    #       3.2 every staggered operand can broadcast  -> STANDARD output
    #       3.3 otherwise (a STANDARD and a staggered full operand coexist)
    #                                                  -> unsupported (raise)
    # Prefer 3.1 in the overlap because propagation does not yet know which
    # candidate layout the optimizer will commit for the staggered producer.
    # A future extension may admit 2a/3.3 by inserting an explicit EA conversion
    # at extra cost.
    staggered_inputs = input_eas & STAGGERED_EAS

    if len(staggered_inputs) > 1:
        # Case 2a: multiple distinct staggered EAs - not supported.
        raise Unsupported(
            f"Multi-arg pointwise with multiple staggered EAs not supported: {input_eas}"
        )
    elif len(staggered_inputs) == 1:
        staggered_ea = next(iter(staggered_inputs))

        # A device *stick* is a broadcast (carries at most one distinct host
        # element) exactly when it maps to a size-1 / sparse host axis. In an STL
        # the stick is the last device dim and the layout constructor
        # (spyre_tensor_impl.cpp) sets `stride_map[-1] == -1` precisely in that
        # case. This is the correct, dim_order-independent test — reading
        # `arg.layout.size[-1]` only works under an identity dim_order (e.g. host
        # size [1, 64, 1] with the stick on a size-1 axis that is not last would
        # be mishandled).
        #
        # `arg.layouts` is a per-consumer list of *candidate* device layouts (see
        # _get_prop_args: `list(buf.layouts)`), from which the cost solver later
        # picks one. So an operand "can broadcast" as long as it has at least one
        # broadcast candidate; pruning the non-broadcast candidates in place is
        # safe and steers the solver to a compatible stick.
        def _broadcast_split(arg):
            """Partition arg.layouts into (broadcast, non_broadcast) candidates."""
            if len(arg.layout.size) == 0:
                return list(arg.layouts), []  # scalar: always broadcastable
            broadcast: list[SpyreTensorLayout] = []
            non_broadcast: list[SpyreTensorLayout] = []
            for stl in arg.layouts:
                (broadcast if stl.stride_map[-1] == -1 else non_broadcast).append(stl)
            return broadcast, non_broadcast

        std_split = [
            (arg, *_broadcast_split(arg))
            for arg in args
            if arg.layouts
            and arg.layouts[0].element_arrangement == ElementArrangement.STANDARD
        ]
        stag_split = [
            (arg, *_broadcast_split(arg))
            for arg in args
            if arg.layouts and arg.layouts[0].element_arrangement == staggered_ea
        ]

        if all(broadcast for _, broadcast, _ in std_split):
            # Case 3.1 (and case 2b with no STANDARD operands): preserve the
            # staggered arrangement and keep only broadcast-compatible STANDARD
            # candidates.
            output_ea = staggered_ea
            for arg, broadcast, non_broadcast in std_split:
                if non_broadcast:
                    arg.layouts[:] = broadcast
        elif stag_split and all(broadcast for _, broadcast, _ in stag_split):
            # Case 3.2: every staggered operand has a broadcast candidate, so
            # the operation can use STANDARD ordering.
            #
            # Safety: device coordinates depend only on device_size/stride_map,
            # not the EA label (see device_coordinates), and `_is_supported_layout`
            # rebuilds each operand from its host size/stride with the *output* EA
            # (here STANDARD). Reinterpreting a broadcastable staggered operand as
            # STANDARD is therefore sound only if its staggered device geometry is
            # identical to the STANDARD layout of the same host shape. A size-1
            # stick usually makes the staggering vacuous, but we do NOT assume it:
            # verify per kept candidate and raise rather than silently emit wrong
            # coordinates (e.g. under a non-identity dim_order or a shape whose
            # staggering is not a genuine no-op).
            for arg, broadcast, _ in stag_split:
                c_in_size = [concretize_expr(s) for s in arg.layout.size]
                c_in_stride = [concretize_expr(s) for s in arg.layout.stride]
                std_geom = SpyreTensorLayout(
                    c_in_size,
                    c_in_stride,
                    arg.layout.dtype,
                    list(range(len(c_in_size))),
                    ElementArrangement.STANDARD,
                )
                for cand in broadcast:
                    if (
                        cand.device_size != std_geom.device_size
                        or cand.stride_map != std_geom.stride_map
                    ):
                        raise Unsupported(
                            f"Multi-arg pointwise with mixed EA: staggered operand "
                            f"{arg.dep.name} is broadcastable but its {staggered_ea} "
                            f"device layout "
                            f"({cand.device_size}/{cand.stride_map}) differs from the "
                            f"STANDARD layout of the same shape "
                            f"({std_geom.device_size}/{std_geom.stride_map}); "
                            f"reinterpreting it as STANDARD would change its "
                            f"coordinates. De-staggering via an explicit EA "
                            f"conversion is not supported yet."
                        )
            output_ea = ElementArrangement.STANDARD
            for arg, broadcast, non_broadcast in stag_split:
                if non_broadcast:
                    arg.layouts[:] = broadcast
            staggered_inputs = set()
        else:
            # Case 3.3: a STANDARD operand and a staggered operand are both
            # non-broadcast full tensors — a genuine mixed-order op we cannot
            # represent yet (future: insert an explicit EA conversion at extra
            # cost). Report the offending STANDARD operand (there is always one
            # here, since 3.1 failed).
            arg = next(arg for arg, broadcast, _ in std_split if not broadcast)
            stl = arg.layouts[0]
            c_stride = [concretize_expr(s) for s in arg.layout.stride]
            mapped = next(
                (d for d, hs in enumerate(c_stride) if hs == stl.stride_map[-1]),
                None,
            )
            mapped_size = (
                concretize_expr(arg.layout.size[mapped])
                if mapped is not None
                else "unknown"
            )
            raise Unsupported(
                f"Multi-arg pointwise with mixed EA: STANDARD input {arg.dep.name} "
                f"must broadcast (device stick dimension size 1) to be compatible "
                f"with a staggered EA, and no staggered operand is broadcastable "
                f"either. Its stick maps to host dim {mapped} of size {mapped_size}"
            )
    else:
        # Case 1: all STANDARD (or non-staggered) EAs - use STANDARD.
        output_ea = ElementArrangement.STANDARD

    ind_names, _, ind_sizes = indirect_info_from_op(op)
    stick_exprs = {
        dc[-1]
        for arg in args
        for stl in arg.layouts
        if arg.dep.name not in ind_names
        for dc in [try_device_coordinates(stl, arg.dep, ind_sizes)]
        if dc is not None
    }

    # Bool physical format resolution: see resolve_output_formats's docstring.
    bool_device_dtype = (
        infer_bool_device_dtype(args) if output.dtype == torch.bool else None
    )
    out_device_dtype, out_dtype_for_layout = resolve_output_formats(
        output.dtype, bool_device_dtype
    )

    # If the indexing and device element size are identical
    # across all inputs and the output we can just propagate the device layout.
    in_coords = [host_coordinates(arg.layout, arg.dep, ind_sizes) for arg in args]
    out_coords = host_coordinates(output, output_dep, ind_sizes)
    can_use_same_layout = True

    if len(stick_exprs) > 1 or any(len(arg.layouts) > 1 for arg in args):
        can_use_same_layout = False
    else:
        for arg, arg_coors in zip(args, in_coords):
            if (
                arg_coors != out_coords
                or arg.layout.size != output.size
                or arg.dep.index != output_dep.index
                or not same_device_size(arg.layout.dtype, out_dtype_for_layout)
            ):
                can_use_same_layout = False
                break

    stick_size = get_elem_in_stick(out_dtype_for_layout)
    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]

    def _is_supported_layout(dim_order):
        for arg in args:
            # Project output dim_order to input, dropping leading dims missing due to broadcast.
            rank_diff = len(output.size) - len(arg.layout.size)
            projected_dim_order = [d - rank_diff for d in dim_order if d >= rank_diff]
            c_in_size = [concretize_expr(s) for s in arg.layout.size]
            c_in_stride = [concretize_expr(s) for s in arg.layout.stride]
            in_stl = SpyreTensorLayout(
                c_in_size,
                c_in_stride,
                out_dtype_for_layout,
                projected_dim_order,
                output_ea,
            )
            coord = try_device_coordinates(in_stl, arg.dep, ind_sizes)
            if coord is None or not is_stick_expr_offset_free(coord[-1], stick_size):
                return False
            # For indirect-access value tensors, the stick dimension cannot
            # depend on the index symbol. If it does, a restickify will be needed
            # to move the indirect coordinate away from the stick.
            if arg.dep.name not in ind_names and ind_sizes:
                stick_coord = coord[-1]
                # Check if stick coordinate depends on any index symbol
                for index_sym in ind_sizes:
                    if index_sym in stick_coord.free_symbols:
                        return False
        return True

    def _try_stick_dim(stick_dim):
        dim_order = _compute_dim_order(stick_dim, c_size, out_coords)
        if _is_supported_layout(dim_order):
            results.append(
                SpyreTensorLayout(
                    c_size, c_stride, out_dtype_for_layout, dim_order, output_ea
                )
            )

    results: list[SpyreTensorLayout] = []

    if can_use_same_layout:
        template_stl = next(iter(args[0].layouts))
        results.append(
            SpyreTensorLayout(
                template_stl.device_size,
                template_stl.stride_map,
                out_device_dtype,
                output_ea,
            )
        )
    elif not stick_exprs:
        _try_stick_dim(-1)
    else:
        offset_free_stick_exprs = {
            e for e in stick_exprs if is_stick_expr_offset_free(e, stick_size)
        }
        # Sort stick exprs for determinism
        for stick_expr in sorted(offset_free_stick_exprs, key=iter_var_id):
            _try_stick_dim(_pick_stick_dim(stick_expr, out_coords))

    # Always scan all dims so that dims absent from any input stick expression
    # (e.g. the outer broadcast dim) are also offered as candidates. Deduplicate
    # against layouts already produced by the input-stick loop above.
    # Skip for staggered-EA ops: their output layout is dictated by the staggered
    # input EA and adding STANDARD candidates would corrupt downstream ops.
    # EA omitted from key: the loop below is skipped for staggered ops, so all
    # candidates added (and looked up) here use STANDARD EA — geometry suffices.
    #
    # Aligned dims are scanned first; unaligned dims (padded later by
    # insert_restickify_padding) only when nothing aligned yielded a candidate.
    seen_keys = {(tuple(r.device_size), tuple(r.stride_map)) for r in results}
    all_dims = range(len(output.size)) if not staggered_inputs else []
    aligned_dims, unaligned_dims = _dims_by_alignment(all_dims, output.size, stick_size)
    for dims in (aligned_dims, unaligned_dims):
        for alt_stick_dim in dims:
            pre_len = len(results)
            _try_stick_dim(alt_stick_dim)
            if len(results) > pre_len:
                key = (tuple(results[-1].device_size), tuple(results[-1].stride_map))
                if key in seen_keys:
                    results.pop()
                else:
                    seen_keys.add(key)
        if results:
            break

    # LX in-place: promote a same-frame input's layout to FIRST so the beam
    # commits it on a cost tie, avoiding a free-but-in-place-defeating permutation
    # (allocator.py _determine_in_place). Two skips guard it:
    #   - footprint mismatch: an fp8-unpack layout has a different total device
    #     element count than the plain output, so its stride_map cannot tile the
    #     output's host strides (copy_tensor rejects it at runtime).
    #   - staggered EA present: the output EA is dictated by that input, so a
    #     STANDARD promotion would corrupt the arrangement of downstream converts
    #     (e.g. rmsnorm fp32-upcast: weight * x_normed.to(fp16)).
    natural_footprints = {
        math.prod([s for s in r.device_size if s > 0]) for r in results
    }
    for arg in args if not staggered_inputs else []:
        if (
            arg.layout.size != output.size
            # Sympy structural (syntactic) equality: two semantically identical
            # expressions with different variable names or term order compare
            # unequal here. This is intentionally conservative — only inputs
            # whose index is exactly the same expression as the output's qualify
            # as same-frame candidates for in-place layout reuse.
            or arg.dep.index != output_dep.index
            or not same_device_size(arg.layout.dtype, output.dtype)
        ):
            continue
        src_stl = next(iter(arg.layouts))
        candidate = SpyreTensorLayout(
            src_stl.device_size,
            src_stl.stride_map,
            get_device_dtype(output.dtype),
        )
        if (
            math.prod([s for s in candidate.device_size if s > 0])
            not in natural_footprints
        ):
            continue
        # Stick must be offset-free; per-input feasibility is left to
        # AllSameNode (INF-costs incompatible).
        out_coord = device_coordinates(candidate, output_dep, ind_sizes)
        if not is_stick_expr_offset_free(out_coord[-1], stick_size):
            continue
        # Move to front (or insert if new): the in-place layout must win on
        # cost ties so the allocator's positional in-place check can fire.
        key = (tuple(candidate.device_size), tuple(candidate.stride_map))
        results = [
            r for r in results if (tuple(r.device_size), tuple(r.stride_map)) != key
        ]
        results.insert(0, candidate)

    if not results:
        # Reaches here when no stick-compatible output layout survives — e.g. in
        # the mixed-EA case (3.1) a STANDARD operand pruned to its only broadcast
        # candidate then fails `_is_supported_layout`. Surface the EA context so
        # the diagnostic is as localized as the pre-pruning gate's was.
        raise Unsupported(
            f"Multi-arg pointwise ({op.get_name()}): no supported output layout "
            f"found with size={output.size}, output_ea={output_ea}, "
            f"coordinates={out_coords}. Input EAs: "
            f"{[a.layouts[0].element_arrangement for a in args if a.layouts]}"
        )

    if len(results) > 1:
        logger.info(
            f"Multi-arg pointwise ({op.get_name()}): producing {len(results)} candidate output layouts."
        )

    op.restick_cost_fn = AllSameNode.from_args(args, results, output_dep, op)
    return results


def _topk_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[PropArg],
) -> list[SpyreTensorLayout]:
    _check_supported_input_sticks(args, "topk")
    x = args[0]
    x_coords = host_coordinates(x.layout, x.dep, None)
    out_coords = host_coordinates(output, output_dep, None)

    # Reduction var: in x's index but absent from output's.
    reduction_var = find_reduction_var((x.dep,), output_dep)

    # Coords that survive the reduction into the output.
    surviving_coords = [
        c
        for c in x_coords
        if len(c.free_symbols) > 0 and matching_dim(out_coords, c) is not None
    ]

    # Collect candidate output stick dims. A valid input stick passes through;
    # a stick on the reduction var requires a restickify, so every surviving
    # coord becomes a candidate.
    out_stick_dims: set[int | None] = set()
    for stl in x.layouts:
        x_stick_expr = device_coordinates(stl, x.dep, None)[-1]
        if reduction_var in x_stick_expr.free_symbols:
            for c in surviving_coords:
                out_stick_dims.add(matching_dim(out_coords, c))
        else:
            out_stick_dims.add(matching_dim(out_coords, x_stick_expr))

    # Build one output STL per candidate stick dim.
    # Note: the stick dim STL will never be added so will never be
    #       selected as a candidate output STL
    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]
    results: list[SpyreTensorLayout] = []
    for out_stick_dim in out_stick_dims:
        if out_stick_dim is None:
            out_dim_order = list(range(len(output.size))) + [-1]
        else:
            out_dim_order = [d for d in range(len(output.size)) if d != out_stick_dim]
            out_dim_order += [out_stick_dim]
        results.append(SpyreTensorLayout(c_size, c_stride, output.dtype, out_dim_order))

    op.restick_cost_fn = AllSameNode.from_args(args, results, output_dep, op)
    return results


def _keep_by_index_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[PropArg],
) -> list[SpyreTensorLayout]:
    """Layout propagation for keep_by_index.

    Output shape matches values input (arg0). Search dimension cannot be the
    stick for arg0 or output. Indices (arg1) k dimension cannot be the stick.

    Algorithm:
      1. Identify search_var: dimension in values not in indices
      2. Find required layout for values where search_var is NOT on stick
      3. Find required layout for indices where k dimension is NOT on stick
      4. Compute output STL with stick matching values' output stick position
    """
    _check_supported_input_sticks(args, "keep_by_index")
    values = args[0]
    indices = args[1]
    out_coords = host_coordinates(output, output_dep, None)
    values_coords = host_coordinates(values.layout, values.dep, None)

    # Find search_var: the loop variable in values not present in indices
    # (unlike reductions, keep_by_index preserves all dimensions)
    indices_coords = host_coordinates(indices.layout, indices.dep, None)
    search_var = None
    for v_coord in values_coords:
        if len(v_coord.free_symbols) == 0:
            continue
        # Check if this coordinate appears in indices
        found_in_indices = any(v_coord.equals(i_coord) for i_coord in indices_coords)
        if not found_in_indices:
            search_var = next(iter(v_coord.free_symbols))
            break

    if search_var is None:
        raise Unsupported("keep_by_index: could not identify search dimension")

    values_req_stl = _find_layout_avoiding_var_on_stick(
        values, search_var, "keep_by_index"
    )

    # Find reduction_var: loop variable in indices not in values (the k dimension)
    reduction_var = None
    for i_coord in indices_coords:
        if len(i_coord.free_symbols) == 0:
            continue
        found_in_values = any(i_coord.equals(v_coord) for v_coord in values_coords)
        if not found_in_values:
            reduction_var = next(iter(i_coord.free_symbols))
            break

    if reduction_var is None:
        raise Unsupported("keep_by_index: could not identify k dimension")

    indices_req_stl = _find_layout_avoiding_var_on_stick(
        indices, reduction_var, "keep_by_index"
    )

    # Compute output STL: stick position matches values' required layout
    x_stick_expr = device_coordinates(values_req_stl, values.dep, None)[-1]
    out_stick_dim = matching_dim(out_coords, x_stick_expr)

    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]

    if out_stick_dim is None:
        out_dim_order = list(range(len(output.size))) + [-1]
    else:
        out_dim_order = [d for d in range(len(output.size)) if d != out_stick_dim]
        out_dim_order += [out_stick_dim]

    out_stl = SpyreTensorLayout(c_size, c_stride, output.dtype, out_dim_order)

    op.restick_cost_fn = FixedInOutNode.from_args(
        [values, indices], out_stl, [values_req_stl, indices_req_stl], op
    )
    return [out_stl]


def compute_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[PropArg],
) -> list[SpyreTensorLayout]:
    """
    Main driver for propagating layouts. There are two tasks performed
    1. Compute candidate output STLs given a set of STLs for each input arg.
    2. Attach a restick cost function based on the type of op.
    """
    data = op.data

    # Log device coordinates for indirect value args. Useful for debugging
    # gather/scatter layout propagation.
    if logger.isEnabledFor(logging.DEBUG):
        indirect_index_names, _, ind_sizes = indirect_info_from_op(op)
        if indirect_index_names:
            for arg in args:
                if arg.dep.name in indirect_index_names:
                    continue
                for j, stl in enumerate(arg.layouts):
                    try:
                        d_coords_str = str(device_coordinates(stl, arg.dep, ind_sizes))
                    except Unsupported:
                        d_coords_str = "<unsupported>"
                    logger.debug(
                        f"  indirect value {arg.dep.name} STL[{j}]"
                        f"\n    d_coords={d_coords_str}"
                    )

    if len(args) > 1 and isinstance(data, Pointwise):
        return _multi_arg_pointwise_layouts(op, output, output_dep, args)

    if isinstance(data, Reduction) and data.reduction_type in [
        BATCH_MATMUL_OP,
        BATCH_MATMUL_FP8_OP,
    ]:
        return _matmul_layouts(op, output, output_dep, args)

    if isinstance(data, Reduction) and data.reduction_type == CONV2D_FWD_OP:
        return _conv_layouts(op, output, output_dep, args)

    if isinstance(data, Reduction) and data.reduction_type == "exx2":
        return _exx2_layout(op, output, output_dep, args)

    if is_topk(op):
        return _topk_layouts(op, output, output_dep, args)

    if is_keep_by_index(op):
        return _keep_by_index_layouts(op, output, output_dep, args)

    aten_op = next(iter(data.origins)).target if data.origins else None

    if aten_op == spyreop.layernormnorm.default:
        # layernormnorm is pointwise but special: it has multiple args, input and
        # output must have matching size/stride, and x's stick must match
        # mean/norm_mean (last logical dim).
        in_layout = args[0].layout
        if in_layout.size != output.size or in_layout.stride != output.stride:
            raise Unsupported(
                f"views not supported for spyre.layernormnorm({in_layout.size})=>{output.size})"
            )
        return _layernormnorm_layout(op, output, output_dep, args)

    if aten_op == aten.clone.default:
        # clone materializes a new buffer in a fixed row-major layout regardless of
        # input stick — equivalent to a restickify. No restickify before it is needed,
        # unless there is an offset in the stick dimension.
        return _clone_layout(op, output, output_dep, args)

    # All other single arg ops
    layouts = []
    for stl in args[0].layouts:
        result = _single_arg_op_layout(
            op, output, output_dep, args[0].dep, args[0].layout, stl
        )
        layouts.extend(result)
    if not layouts:
        raise Unsupported(
            f"{op.get_name()} ({aten_op}): no supported output layout found for "
            f"any of {len(args[0].layouts)} candidate input layouts; "
            f"output size={output.size}"
        )
    op.restick_cost_fn = AllSameNode.from_args(args, layouts, output_dep, op)
    return layouts


def _all_constant_layouts(op: Operation) -> list[SpyreTensorLayout]:
    """Return one STL per valid stick dimension for a constant-valued buffer.

    A constant tensor (ones_like, full, zeros_like, ...) has no real memory
    access pattern — every element is the same scalar broadcast from a
    SpyreConstantFallback.  Because the content is uniform, any stick layout
    is correct.  Offering all valid choices lets the optimizer pick whichever
    is compatible with the rest of the graph at zero cost, avoiding a needless
    restickify.

    NOTE: constant-fill ops in the main propagation loop currently use
    generic_layout instead (a single default-layout candidate) to avoid
    exponential beam-state growth in loop-unrolled graphs.  This function is
    still used for the SpyreEmptyFallback mutation target path (in-place update
    ops with no real inputs) where the enumeration is bounded and correct.
    It will also be restored for the single-consumer constant case once that
    optimization lands.
    """
    output: FixedLayout = op.get_layout()
    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]
    stick_size = get_elem_in_stick(output.dtype)
    layouts = [
        SpyreTensorLayout(
            c_size,
            c_stride,
            output.dtype,
            [d for d in range(len(c_size)) if d != stick_dim] + [stick_dim],
        )
        for stick_dim in range(len(c_size))
        if c_size[stick_dim] % stick_size == 0 and c_size[stick_dim] >= stick_size
    ]
    if not layouts:
        layouts = [generic_layout(op)]
    return layouts


def generic_layout(op: Operation) -> SpyreTensorLayout:
    output: FixedLayout = op.get_layout()
    # Concretize for C++ SpyreTensorLayout constructor.
    c_size = [concretize_expr(s) for s in output.size]
    return SpyreTensorLayout(c_size, output.dtype)


def _one_mem_dep(deps) -> MemoryDep | None:
    mem_deps = [dep for dep in deps if isinstance(dep, MemoryDep)]
    if len(mem_deps) != 1:
        return None
    return mem_deps[0]


def _same_host_layout(lhs, rhs) -> bool:
    return (
        lhs.device == rhs.device
        and lhs.dtype == rhs.dtype
        and tuple(lhs.size) == tuple(rhs.size)
        and tuple(lhs.stride) == tuple(rhs.stride)
        and lhs.offset == rhs.offset
    )


def _target_device_layout(target, name: str):
    layout = target.get_layout()
    if isinstance(layout, FixedTiledLayout):
        return layout.device_layout

    # This runs after input layout propagation but before restickify
    # optimization/finalization, so graph inputs still carry propagated
    # candidate layouts on the TensorBox rather than a finalized committed_stl.
    graph_input = V.graph.graph_inputs.get(name)
    layouts = getattr(graph_input, "layouts", None)
    if not layouts:
        # Also check candidate layouts on the producing ComputedBuffer —
        # graph intermediates are not in graph_inputs.
        # Exclude SpyreEmptyFallback and SpyreConstantFallback: those are
        # synthetic buffers whose layouts are derived via separate paths.
        # Exactly one candidate is required; the assert below enforces this.
        buf = V.graph.get_buffer(name) if name else None
        if buf is not None and not isinstance(
            buf, (SpyreEmptyFallback, SpyreConstantFallback)
        ):
            buf_layouts = getattr(buf, "layouts", None)
            if buf_layouts:
                layouts = buf_layouts

    if not layouts or len(layouts) != 1:
        return None
    return next(iter(layouts))


def _find_alt_target_stl(
    target_layout: FixedLayout,
    target_stl: SpyreTensorLayout,
    output_dep: MemoryDep,
) -> SpyreTensorLayout | None:
    """
    Find an alternative SpyreTensorLayout with an offset-free stick expression
    for a mutation target. Returns None if the current layout is already valid,
    or raises Unsupported if no valid alternative exists.
    """
    # A bool target's stick size comes from the format it is physically stored
    # in (target_stl), not from target_layout.dtype -- see bool_layout_dtype's
    # docstring.
    dtype_for_layout = (
        bool_layout_dtype(target_stl.device_dtype, "mutation target")
        if target_layout.dtype == torch.bool
        else target_layout.dtype
    )
    stick_size = get_elem_in_stick(dtype_for_layout)
    write_stick = device_coordinates(target_stl, output_dep, None)[-1]
    if is_stick_expr_offset_free(write_stick, stick_size):
        return None

    c_size = [concretize_expr(s) for s in target_layout.size]
    c_stride = [concretize_expr(s) for s in target_layout.stride]
    candidates = _candidate_output_stls(
        target_layout, output_dep, c_size, c_stride, write_stick, dtype_for_layout
    )
    if not candidates:
        raise Unsupported(
            f"no offset-free alternative stick dim for mutation target "
            f"(write stick {write_stick!r}, size={target_layout.size})"
        )
    return candidates[0]


def _resolve_copy_back_candidates(operations: list[Operation]) -> None:
    """Commit safe lowering-marked copy-back candidates.

    ``aten.copy_`` lowering marks structural candidates but leaves the explicit
    copy intact.  Once layout propagation has computed producer layouts, this
    resolver proves the full safety condition.  Failed candidates remain normal
    copies.
    """
    writer_by_name: dict[str, Operation] = {}
    write_counts: Counter[str] = Counter()
    names_read: set[str] = set()

    for op in operations:
        read_writes = op.get_read_writes()
        for write in read_writes.writes:
            writer_by_name[write.name] = op
            write_counts[write.name] += 1
        for read in read_writes.reads:
            if isinstance(read, MemoryDep):
                names_read.add(read.name)

    graph_inputs = set(V.graph.graph_input_names)
    graph_outputs = set(V.graph.get_output_names())
    removed_ops: list[Operation] = []
    mutated_inputs: set[str] = set()

    for copy_op in operations:
        if not (
            getattr(copy_op, COPY_BACK_CANDIDATE_ATTR, False)
            and isinstance(copy_op, ComputedBuffer)
            and isinstance(copy_op.layout, MutationLayoutSHOULDREMOVE)
        ):
            continue

        read_writes = copy_op.get_read_writes()
        source = _one_mem_dep(read_writes.reads)
        destination = _one_mem_dep(read_writes.writes)
        if source is None or destination is None:
            continue
        if source.index != destination.index:
            continue

        target = copy_op.layout.get_buffer()
        target_name = target.get_name()
        if target_name not in graph_inputs:
            continue
        if target_name in graph_outputs or source.name in graph_outputs:
            continue
        if target_name in names_read or target_name in mutated_inputs:
            continue

        producer = writer_by_name.get(source.name)
        if producer is None or producer is copy_op:
            continue
        if not isinstance(producer, ComputedBuffer):
            continue
        if not config.ignore_span_overflow_hints and isinstance(
            producer.data, Pointwise
        ):
            # Layouts are still plain FixedLayout here (finalize_layouts
            # hasn't run yet), so we can't yet tell whether this specific
            # producer will actually get auto-tiled. Conservatively preserve
            # the copy-back for any Pointwise producer while the feature is
            # on, the same way the old (now-removed) chunk_large_tensors
            # guard did: `config.chunk_large_tensors and
            # isinstance(producer.data, Pointwise)`, with no layout-type
            # check either.
            continue
        if isinstance(producer.layout, MutationLayoutSHOULDREMOVE):
            continue
        if write_counts[source.name] != 1:
            continue
        if not _same_host_layout(producer.get_layout(), target.get_layout()):
            continue

        target_stl = _target_device_layout(target, target_name)
        if target_stl is None:
            continue
        producer_layouts = getattr(producer, "layouts", None)
        if not producer_layouts or target_stl not in producer_layouts:
            continue
        # Only elide when the producer has a single unambiguous layout. With
        # multiple candidates the optimizer may not commit to target_stl, so
        # eliding the copy would be incorrect.
        if len(producer_layouts) != 1:
            continue

        producer.layout = copy_op.layout
        producer.layouts = [target_stl]
        producer.committed_stl = target_stl
        setattr(producer, ELIDED_COPY_BACK_ATTR, True)
        mutated_inputs.add(target_name)
        removed_ops.append(copy_op)
        logger.info(
            "removed copy-back %s; %s now writes %s",
            copy_op.get_name(),
            producer.get_name(),
            target_name,
        )

    for op in removed_ops:
        for write in op.get_read_writes().writes:
            V.graph.removed_buffers.add(write.name)
        operations.remove(op)


def _eager_view_input_layout(
    real_input: torch.Tensor,
    ptl: FixedLayout,
    name: str,
) -> "FixedLayout | None":
    """Rewrite a placeholder view's FixedLayout to "layout = base, dep = view".

    Eager-mode placeholders arrive with view size/stride/offset baked onto
    FixedLayout. Downstream passes (and inline-slice paths) instead expect
    base-storage size/stride with the offset in ``layout.offset`` so
    ``FixedLayout.make_indexer`` weaves it into ``MemoryDep.index``.

    Returns the replacement layout, or ``None`` if no rewrite is needed.
    """
    base = real_input._base
    storage_offset = real_input.storage_offset()

    # The two gates below are intentionally orthogonal:
    #
    #   Example       | sub-region | offset | Action
    #   --------------|------------|--------|-------------------------------
    #   x[1:]         | y          | y      | size/stride <- base; offset
    #   x[:6]         | y          | n      | size/stride <- base
    #   x.t()[1:]     | n          | y      | keep view size/stride; offset
    #   x.t()         | n          | n      | no rewrite
    #
    # Sub-region requires stride preserved AND size differs. A pure
    # transpose differs on both, but rewriting it to base size/stride
    # would silently strip the permutation -- it falls to the offset-only
    # branch instead.
    #
    # Stride equality alone isn't sufficient: a size-1 dimension has an
    # arbitrary stride in PyTorch, so a transpose/permute touching one can
    # coincidentally match its base's stride tuple too. A genuine
    # sub-region can only shrink -- every dim of the view must be <= the
    # same dim of base -- which a transpose/permute never satisfies.
    is_sub_region = (
        base is not None
        and tuple(real_input.stride()) == tuple(base.stride())
        and tuple(real_input.size()) != tuple(base.size())
        and all(real_input.size(d) <= base.size(d) for d in range(real_input.dim()))
    )
    if not (is_sub_region or storage_offset != 0):
        return None

    if is_sub_region:
        # Offset (via make_indexer) + dep.ranges recover the view extent.
        new_size = list(base.size())
        new_stride = list(base.stride())
    else:
        # Keep the view's permuted size/stride, just attach the offset.
        new_size = list(real_input.size())
        new_stride = list(real_input.stride())

    # Verify the offset is device-stick-aligned by computing the real
    # device stick coordinate for a full read of this view, using the same
    # device-coordinate machinery (compute_coordinates +
    # is_stick_expr_offset_free) already relied on elsewhere in this module
    # for equivalent checks. A flat host-offset heuristic can't see per-row
    # stick padding -- a row boundary can be device-stick-aligned even when
    # the row length itself isn't a multiple of elem_in_stick -- so the check
    # has to happen in device space, not host space.
    # TODO: unaligned stick-dim offsets need alt-layout retargeting;
    # currently rejected to avoid silent miscompute downstream.
    stl = real_input.device_tensor_layout()
    elem_in_stick = get_elem_in_stick(ptl.dtype)
    rank = len(real_input.shape)
    ivars = sympy.symbols(f"_offset_check_i0:{rank}", integer=True, nonnegative=True)
    var_ranges = {v: s for v, s in zip(ivars, real_input.shape)}
    flat_index = storage_offset + sum(new_stride[d] * ivars[d] for d in range(rank))
    stick_expr = compute_coordinates(
        list(stl.device_size), list(stl.stride_map), var_ranges, flat_index
    )[-1]
    if not is_stick_expr_offset_free(stick_expr, elem_in_stick):
        raise Unsupported(
            f"graph input {name} has a non-stick-aligned device stick "
            f"coordinate ({stick_expr}) at storage_offset={storage_offset}; "
            f"not yet supported"
        )

    return FixedLayout(
        device=ptl.device,
        dtype=ptl.dtype,
        size=new_size,
        stride=new_stride,
        offset=sympy.Integer(storage_offset),
    )


def propagate_spyre_tensor_layouts(
    graph: GraphLowering,
) -> None:
    operations = graph.operations
    # Convert InputBuffers from FixedLayout to SpyreTensorLayouts
    if len(graph.graph_input_names) > 0:
        for name, real_input in zip(graph.graph_input_names, V.get_real_inputs()):
            if isinstance(real_input, torch.Tensor):
                stl = real_input.device_tensor_layout()
                if stl is None:
                    # A CPU tensor lifted as a graph input, or a host tensor
                    # feeding a FallbackKernel has no Spyre layout;
                    # leave its FixedLayout and .layouts untouched,
                    # mirroring the non-Spyre ComputedBuffer skip below.
                    continue
                tb = graph.graph_inputs[name]
                if (
                    not isinstance(tb, TensorBox)
                    or not isinstance(tb.data, StorageBox)
                    or not isinstance(tb.data.data, InputBuffer)
                ):
                    raise Unsupported(
                        f"graph input {name} is not a TensorBox(StorageBox(InputBuffer))"
                    )
                ptl = tb.data.data.layout
                if not isinstance(ptl, FixedLayout):
                    raise Unsupported(f"graph input {name} does not have a FixedLayout")
                new_layout = _eager_view_input_layout(real_input, ptl, name)
                if new_layout is not None:
                    tb.data.data.layout = new_layout
                tb.layouts = [stl]

    # Alt layout each graph input has been forced to by a mutation write, so a
    # second write can detect a conflicting alt.
    forced_mutation_alts: dict[str, SpyreTensorLayout] = {}

    # Operations are in topological order (guaranteed by GraphLowering).
    # Visit them and use the input SpyreTensorLayouts and the operation being
    # performed to compute the set of possible output SpyreTensorLayouts.
    for op in operations:
        if op.is_no_op():
            op.layouts = [generic_layout(op)]
            op.restick_cost_fn = AnyInNode.from_args()
        elif isinstance(op, ComputedBuffer):
            layout = op.maybe_get_layout()
            if layout is None or layout.device.type != DEVICE_NAME:
                continue
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                target = op.layout.target
                while isinstance(target, ReinterpretView):
                    target = target.data
                target_name = target.get_name() if hasattr(target, "get_name") else ""
                # Look up the actual buffer node (unwraps TensorBox/StorageBox
                # wrappers that coarse_tile.py places around SpyreEmptyFallback).
                target_buf = V.graph.get_buffer(target_name) if target_name else None
                target_stl = _target_device_layout(target, target_name)
                if target_stl is None:
                    target_buf_layouts = getattr(target_buf, "layouts", None)
                    if not isinstance(target_buf, SpyreEmptyFallback) and (
                        target_buf_layouts and len(target_buf_layouts) > 1
                    ):
                        # target_buf is an ordinary multi-candidate ComputedBuffer
                        # (issue #3845): this mutation op (e.g. copy_forced) and
                        # target_buf are co-outputs — both write the same buffer, so
                        # they must commit to the same layout. Pass target_buf as a
                        # co-output to AllSameNode so the beam search enforces exact
                        # layout equality. Restickify may only be inserted on real
                        # input edges (e.g. the src tensor); the output buffer itself
                        # cannot be restickified.
                        assert target_buf is not None
                        target_write_dep = _one_mem_dep(
                            target_buf.get_read_writes().writes
                        )
                        assert target_write_dep is not None, (
                            f"{op.get_name()}: mutation target {target_name!r} "
                            f"must have exactly one write MemoryDep to synthesize "
                            f"a coupling edge"
                        )
                        rw = op.get_read_writes()
                        output_dep = next(iter(rw.writes))
                        args = _get_prop_args(rw.reads)  # real inputs only (e.g. src)
                        op.layouts = list(
                            target_buf_layouts
                        )  # inherit target's candidates
                        # target_write_dep is the co-output: same buffer, must agree on layout
                        op.restick_cost_fn = AllSameNode.from_args(
                            args, op.layouts, [output_dep, target_write_dep], op
                        )
                        continue
                    if not isinstance(target_buf, SpyreEmptyFallback):
                        # op gets no .layouts/.restick_cost_fn at all; any
                        # downstream consumer that requires them (e.g.
                        # optimize_restickify.py's hasattr(op,
                        # "restick_cost_fn") asserts) will fail loudly, but
                        # not with this root cause visible -- warn here so
                        # that failure is traceable back to this skip.
                        logger.warning(
                            "MutationLayoutSHOULDREMOVE target_stl=None, "
                            "skipping layout assignment: op=%s target_name=%r "
                            "buf_type=%s",
                            op.get_operation_name(),
                            target_name,
                            type(target_buf).__name__,
                        )
                        continue
                    # SpyreEmptyFallback mutation target has no device layout yet
                    # -- expected, not exceptional; handled just below.
                    logger.debug(
                        "MutationLayoutSHOULDREMOVE target_stl=None: "
                        "op=%s target_name=%r buf_type=%s",
                        op.get_operation_name(),
                        target_name,
                        type(target_buf).__name__,
                    )
                    # SpyreEmptyFallback target has no device layout yet.
                    # Treat the mutation op like a normal pointwise op: run
                    # _multi_arg_pointwise_layouts with the non-target inputs.
                    # This enforces input-compatibility and slice constraints,
                    # so the backend DDL slice check passes.
                    rw = op.get_read_writes()
                    output_dep = next(iter(rw.writes))
                    all_args = _get_prop_args(rw.reads)
                    # Exclude the target read-back (dep.name == target_name) from
                    # layout candidate derivation: it IS the output buffer, not an
                    # independent input.
                    new_value_args = [a for a in all_args if a.dep.name != target_name]
                    # Build a dep for the SpyreEmptyFallback target so the beam
                    # search couples all mutation ops writing it via a co-output
                    # edge. We cannot find it in rw.reads — the mutation target is
                    # encoded in MutationLayoutSHOULDREMOVE, not in inner_fn, so
                    # it never appears in get_read_writes().reads. Construct a
                    # MemoryDep with target_name but same index shape as output_dep.
                    #
                    if not new_value_args:
                        # No real tensor inputs: this is a coarse-tile identity fill
                        # (e.g. coarse_tile_fill writing inf/0 into the accumulator).
                        # A scalar fill has no stick orientation preference, so use
                        # AnyInNode and let the real writers (combine, copy) determine
                        # the STL. Borrow target_buf.layouts so the beam has a non-empty
                        # candidate list; AnyInNode commits at zero cost regardless.
                        assert hasattr(target_buf, "layouts"), (
                            f"SpyreEmptyFallback {target_name!r} has no layouts — "
                            "expected SpyreEmptyFallback to have layouts set before any "
                            "mutation op writes it"
                        )
                        candidates = target_buf.layouts
                        op.layouts = candidates
                        op.restick_cost_fn = AnyInNode.from_args()
                        continue
                    # Build a co-output dep for the mutation target so the beam
                    # couples all mutation ops (that have real inputs) writing it
                    # to the same STL. The target has all valid STLs as candidates
                    # so this dep adds no spurious constraint.
                    target_co_dep = MemoryDep(
                        name=target_name,
                        index=output_dep.index,
                        var_names=output_dep.var_names,
                        size=output_dep.size,
                    )
                    out_deps = [output_dep, target_co_dep]
                    if (
                        isinstance(op.data, Reduction)
                        and op.data.reduction_type == BATCH_MATMUL_OP
                    ):
                        # x and y are the two genuine matmul inputs (not the target
                        # read-back). _matmul_layouts derives a single out_stl
                        # deterministically from the target layout and installs its
                        # own FixedInOutNode, so no separate AllSameNode join is needed.
                        # target_co_dep is intentionally not threaded in: matmul has a
                        # fixed, deterministic output STL so two BATCH_MATMUL ops writing
                        # the same SpyreEmptyFallback will always agree.
                        assert len(new_value_args) == 2, (
                            "BATCH_MATMUL_OP mutation op should have exactly "
                            f"two non-target inputs, got {len(new_value_args)} "
                            f"for {op.get_name()}"
                        )
                        accum_layout = target_buf.get_layout()
                        candidates = _matmul_layouts(
                            op, accum_layout, output_dep, new_value_args
                        )
                        target_buf.layouts = candidates
                        op.layouts = candidates
                    elif isinstance(op.data, Reduction):
                        # The non-target input is the reduction's un-reduced,
                        # higher-rank input. Must use _single_arg_op_layout, not the
                        # broadcast-oriented pointwise join path.
                        assert len(new_value_args) == 1, (
                            "Reduction mutation op should have exactly one non-target "
                            f"input, got {len(new_value_args)} for {op.get_name()}"
                        )
                        accum_layout = target_buf.get_layout()
                        in_arg = new_value_args[0]
                        candidates = []
                        for stl in in_arg.layouts:
                            candidates.extend(
                                _single_arg_op_layout(
                                    op,
                                    accum_layout,
                                    output_dep,
                                    in_arg.dep,
                                    in_arg.layout,
                                    stl,
                                )
                            )
                        if not candidates:
                            raise Unsupported(
                                f"{op.get_name()}: no supported output layout "
                                f"found for any of {len(in_arg.layouts)} "
                                f"candidate input layouts; target size="
                                f"{accum_layout.size}"
                            )
                        op.restick_cost_fn = AllSameNode.from_args(
                            all_args, candidates, out_deps, op
                        )
                        target_buf.layouts = candidates
                        op.layouts = candidates
                    else:
                        target_layout = target_buf.get_layout()
                        candidates = _multi_arg_pointwise_layouts(
                            op, target_layout, output_dep, new_value_args
                        )
                        # Propagate candidates forward: the mutation target is
                        # write-only from this op's perspective (excluded from
                        # new_value_args), so no read edge carries the layout
                        # information forward naturally.  Overwrite target_buf.layouts
                        # so that ops later in topo order that READ the target see
                        # the narrowed candidate set from this write, not the initial
                        # wide set assigned at allocation time.
                        target_buf.layouts = candidates
                        op.restick_cost_fn = AllSameNode.from_args(
                            all_args, candidates, out_deps, op
                        )
                        op.layouts = candidates
                    continue
                rw = op.get_read_writes()
                output_dep = next(iter(rw.writes))
                args = _get_prop_args(rw.reads)

                # Find an alternative layout if the write has an unsupported stick
                # expression (e.g. offset like v+32). Force the optimizer to use
                # this layout for the mutation target.
                # Note: SpyreEmptyFallback targets are not graph inputs so skip
                # the alt-layout path (which only applies to graph inputs).
                target_layout = target.get_layout()
                if isinstance(target_layout, FixedLayout) and not isinstance(
                    target_buf, SpyreEmptyFallback
                ):
                    alt_stl = _find_alt_target_stl(
                        target_layout, target_stl, output_dep
                    )
                    if alt_stl is not None:
                        graph_input = V.graph.graph_inputs.get(target_name)
                        assert graph_input is not None
                        # A graph input holds only one device layout, so two
                        # writes needing different alts cannot both be expressed.
                        # TODO: support this by chaining relayouts between writes
                        # through temp buffers.
                        prior_alt = forced_mutation_alts.get(target_name)
                        if prior_alt is not None and prior_alt != alt_stl:
                            raise Unsupported(
                                f"multiple mutations to graph input {target_name} "
                                f"require conflicting alternative layouts "
                                f"({prior_alt!r} vs {alt_stl!r}); chaining "
                                f"relayouts between writes is not yet supported"
                            )
                        forced_mutation_alts[target_name] = alt_stl
                        graph_input.layouts = [alt_stl]
                        op._restickify_plan = (target_name, target_stl, alt_stl)
                        target_stl = alt_stl
                op.layouts = [target_stl]
                op.restick_cost_fn = AllSameNode.from_args(
                    args, [target_stl], output_dep, op
                )
                continue
            op.decide_layout()
            rw = op.get_read_writes()
            output_dep = next(iter(rw.writes))
            args = _get_prop_args(rw.reads)
            output = op.get_layout()
            if not args:
                mem_reads = [r for r in rw.reads if isinstance(r, MemoryDep)]
                is_constant_fill = bool(mem_reads) and all(
                    isinstance(V.graph.get_buffer(r.name), SpyreConstantFallback)
                    for r in mem_reads
                )
                if is_constant_fill:
                    # Constant-fill ops (zeros_like, full, ones_like, ...) have no real
                    # memory layout — they can materialize in any stick orientation.
                    # For now, using a single generic layout candidate to reduce beam
                    # state space explosion.  Optimizations to follow.
                    op.layouts = [generic_layout(op)]
                else:
                    logger.warning(
                        f"{op.get_name()} has no propagatable args but reads non-constant "
                        f"buffers {[r.name for r in mem_reads]}; falling back to generic layout"
                    )
                    op.layouts = [generic_layout(op)]
                op.restick_cost_fn = AnyInNode.from_args()
            elif isinstance(op.data, (Pointwise, Reduction)):
                op.layouts = compute_layouts(op, output, output_dep, args)
            else:
                logger.warning(f"Warning: unhandled node type {type(op.data)}")
        elif isinstance(op, FallbackKernel):
            # FallbackKernel.create in PyTorch produces three cases:
            #   Case 1 (single tensor)  -> MultiOutputLayout + 1 MultiOutput
            #   Case 2 (tuple of N)     -> MultiOutputLayout + N MultiOutputs
            #   Case 3 (void/in-place)  -> NoneLayout       + 0 MultiOutputs
            # The FallbackKernel itself never carries a real tensor layout
            # (MultiOutputLayout / NoneLayout both raise from get_layout()).
            # The trailing MultiOutputs are handled in their own branch below.
            pass
        elif isinstance(op, MultiOutput):
            op.layouts = [generic_layout(op)]
            op.restick_cost_fn = AnyInNode.from_args()
        elif isinstance(op, SpyreConstantFallback):
            op.layouts = [generic_layout(op)]
            op.restick_cost_fn = AnyInNode.from_args()
        elif isinstance(op, SpyreEmptyFallback):
            # Full-buffer placeholder allocated by _allocate_full_buffer.
            # Offer all valid STLs so the beam can commit to whichever layout
            # the mutation writers (fill, combine, copy) require.  The co-output
            # dep on each writer enforces that the committed STL matches the
            # writer's output STL — so the beam picks the right one rather than
            # defaulting to generic_layout (stick) and forcing infeasibility for
            # flat writers.
            op.layouts = _all_constant_layouts(op)
            op.restick_cost_fn = AnyInNode.from_args()
        elif isinstance(op, DeviceCopy):
            # spyre -> cpu: the output is a host tensor and carries no Spyre
            #     layout. Leave `.layouts` unset.
            # cpu -> spyre: the output is a fresh on-device buffer with no
            #     inherited tiling, so give it a new device layout.
            if op.get_layout().device.type == DEVICE_NAME:
                op.layouts = [generic_layout(op)]
                op.restick_cost_fn = AnyInNode.from_args()

        elif isinstance(
            op,
            (
                BroadcastAsyncFallback,
                WaitWorkFallback,
                AllReduceAsyncFallback,
            ),
        ):
            input_name = op.inputs[0].get_name()
            input_buf = V.graph.get_buffer(input_name)
            op.layouts = list(input_buf.layouts)
            op.restick_cost_fn = AnyInNode.from_args()
        elif isinstance(op, AllGatherAsyncFallback):
            op.layouts = [generic_layout(op)]
            op.restick_cost_fn = AnyInNode.from_args()
        elif isinstance(op, ExternKernel):
            logger.warning(f"unhandled node type {type(op)}")
        else:
            logger.warning(f"unhandled operation type {type(op)}")

    _resolve_copy_back_candidates(operations)


def propagate_mutation_layouts(
    nodes: list,
) -> list:
    """
    Second phase of layout propagation for mutation ops.

    ComputedBuffers with MutationLayoutSHOULDREMOVE are skipped in
    propagate_spyre_tensor_layouts because the scheduler needs to see the
    mutation layout during its initialisation to set up mutation tracking.
    This pass runs as a _pre_fusion_custom_pass (after scheduler init) to
    assign FixedTiledLayout to those remaining mutation ops.
    """
    for n in nodes:
        if not (isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer)):
            continue
        if not isinstance(n.node.layout, MutationLayoutSHOULDREMOVE):
            continue
        if isinstance(n.node.data, (Pointwise, Reduction)):
            real = n.node.layout.real_layout()
            if isinstance(real, FixedTiledLayout):
                n.node.layout = real
            else:
                rw = n.read_writes
                output_dep = next(iter(rw.writes))
                args = _get_prop_args(rw.reads)
                output = n.node.get_layout()
                layouts = list(compute_layouts(n.node, output, output_dep, args))
                n.node.layout = layouts[0]
        elif isinstance(n.node.data, Reduction):
            real = n.node.layout.real_layout()
            if isinstance(real, FixedTiledLayout):
                n.node.layout = real
            else:
                logger.warning(
                    "propagate_mutation_layouts: unhandled mutation Reduction"
                    f" op {n.node.get_name()}: real_layout is {type(real)}"
                )
        else:
            logger.warning(
                f"propagate_mutation_layouts: unhandled mutation op {type(n.node.data)}"
            )

    return nodes
