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
    ElementArrangement,
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
)
from .errors import Unsupported
from .constants import (
    BATCH_MATMUL_OP,
    COPY_BACK_CANDIDATE_ATTR,
    DEVICE_NAME,
    ELIDED_COPY_BACK_ATTR,
    REDUCTIONS_NON_STICK_DIM_ONLY,
    STAGGERED_EAS,
    TOPK_OPS,
)
from .ir import FixedTiledLayout, SpyreConstantFallback
from .pass_utils import (
    compute_restickify_target_layout,
    concretize_expr,
    find_matmul_generated_var,
    find_reduction_var,
    identify_matmul_inputs,
    host_coordinates,
    device_coordinates,
    try_device_coordinates,
    indirect_info_from_op,
    is_stick_expr_offset_free,
    iter_var_id,
)
from .optimize_restickify import AllSameNode, AnyInNode, FixedInOutNode
from .views import matching_dim

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
            if hasattr(buf, "layouts"):
                res.append(PropArg(arg, layout, list(buf.layouts)))
            else:
                if not isinstance(layout, FixedTiledLayout):
                    raise RuntimeError(f"{buf} does not have FixedTiledLayout")
                res.append(PropArg(arg, layout, [layout.device_layout]))
    return res


def same_device_size(t1: torch.dtype, t2: torch.dtype) -> bool:
    return get_elem_in_stick(t1) == get_elem_in_stick(t2)


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
    stick_expr, output, output_dep, c_size, c_stride
) -> SpyreTensorLayout | None:
    """If stick_expr is offset-free, build an output STL with it mapped to the right dim.

    Returns None if stick_expr has an offset (caller should fall back to scanning).
    """
    stick_size = get_elem_in_stick(output.dtype)
    if not is_stick_expr_offset_free(stick_expr, stick_size):
        return None
    out_coords = host_coordinates(output, output_dep, None)
    out_stick_dim = _pick_stick_dim(stick_expr, out_coords)
    return _make_output_stl(output, output_dep, c_size, c_stride, out_stick_dim)


def _make_output_stl(
    output, output_dep, c_size, c_stride, stick_dim
) -> SpyreTensorLayout | None:
    """Build a candidate output STL with stick_dim last and verify the resulting stick is offset-free.

    Returns None if the resulting stick expression has an offset.
    """
    stick_size = get_elem_in_stick(output.dtype)
    out_coords = host_coordinates(output, output_dep, None)
    dim_order = _compute_dim_order(stick_dim, c_size, out_coords)
    stl = SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)
    coords = device_coordinates(stl, output_dep, None)
    if is_stick_expr_offset_free(coords[-1], stick_size):
        return stl
    return None


def _candidate_output_stls(
    output: FixedLayout,
    output_dep: MemoryDep,
    c_size: list,
    c_stride: list,
    stick_size: int,
    skip_stick_expr: sympy.Expr,
) -> list[SpyreTensorLayout]:
    """Enumerate candidate output STLs by trying each dim as the stick.

    Skip the dim that already produces an unsupported stick.
    """
    out_coords = host_coordinates(output, output_dep, None)
    skip_dim = _pick_stick_dim(skip_stick_expr, out_coords)

    result = []
    for alt_stick_dim in range(len(output.size)):
        if alt_stick_dim == skip_dim:
            continue
        if concretize_expr(output.size[alt_stick_dim]) % stick_size != 0:
            # TODO: Support dimensions with size not divisible by stick_size via padding (See #1756)
            continue
        stl = _make_output_stl(output, output_dep, c_size, c_stride, alt_stick_dim)
        if stl is not None:
            result.append(stl)
    return result


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


def _single_arg_op_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    dep: MemoryDep,
    in_layout: FixedLayout,
    stl: SpyreTensorLayout,
) -> list[SpyreTensorLayout]:
    """
    Compute the output STL(s) for a single-arg op given one candidate input STL.
    Called once per candidate input STL to produce corresponding output STL(s).
    """
    data = op.data
    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]
    stick_size = get_elem_in_stick(output.dtype)

    if isinstance(data, Reduction):
        x_dev_coords = device_coordinates(stl, dep, None)
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
                x_stick_expr, output, output_dep, c_size, c_stride
            )
            if out_stl is not None:
                return [out_stl]

        # Try alternative layouts when input layout is not supported
        in_coords = host_coordinates(in_layout, dep, None)
        out_coords = host_coordinates(output, output_dep, None)
        stick_dim = matching_dim(in_coords, x_stick_expr)
        layouts = []
        for in_dim in range(len(in_layout.size)):
            if in_dim == stick_dim:
                continue
            if concretize_expr(in_layout.size[in_dim]) % stick_size != 0:
                # TODO: Support dimensions with size not divisible by stick_size via padding (See #1756)
                continue
            in_coord = in_coords[in_dim]
            # Map input dim to output dim. If input dim carries reduction var, it's collapsed
            if reduction_var is not None and reduction_var in in_coord.free_symbols:
                out_stick_dim = -1
            else:
                out_stick_dim = _pick_stick_dim(in_coord, out_coords)
                if out_stick_dim < 0:
                    continue
            out_stl = _make_output_stl(
                output, output_dep, c_size, c_stride, out_stick_dim
            )
            if out_stl is not None:
                layouts.append(out_stl)

        return layouts

    # Single-arg pointwise
    assert isinstance(data, Pointwise)
    origin_node = next(iter(data.origins))
    aten_op = origin_node.target
    match aten_op:
        case prims.convert_element_type.default if not same_device_size(
            in_layout.dtype, output.dtype
        ):
            # Type conversion may require padding when input has padding due to stick
            # alignment. For example, 4x16 FP16 has 48 elements of padding (64 total),
            # which becomes 64 FP32 elements when converted. We need to reflect this
            # in the output host size so the constructor creates the correct device layout.
            in_stick_expr = device_coordinates(stl, dep, None)[-1]
            if not is_stick_expr_offset_free(in_stick_expr, stl.elems_per_stick()):
                return []

            input_ea = stl.element_arrangement

            # Determine output EA based on conversion direction and input EA
            if in_layout.dtype == torch.float16 and output.dtype == torch.float32:
                # FP16 → FP32 conversion
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

    in_coords = host_coordinates(in_layout, dep, None)
    out_coords = host_coordinates(output, output_dep, None)
    if (
        in_coords == out_coords
        and in_layout.size == output.size
        and dep.index == output_dep.index
        and same_device_size(in_layout.dtype, output.dtype)
    ):
        # Input and output tensors are being accessed identically and elem size is the same.
        # We can simply propagate the device_layout including ElementArrangement.
        stl = SpyreTensorLayout(
            stl.device_size,
            stl.stride_map,
            get_device_dtype(output.dtype),
            stl.element_arrangement,
        )
        return [stl]

    in_device_coords = device_coordinates(stl, dep, None)
    stick_expr = in_device_coords[-1]

    # Try to preserve input layout, fall back to scanning all output dims
    out_stl = _output_stl_from_stick_expr(
        stick_expr, output, output_dep, c_size, c_stride
    )
    if out_stl is not None:
        return [out_stl]
    return _candidate_output_stls(
        output, output_dep, c_size, c_stride, stick_size, stick_expr
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
    stick_size = get_elem_in_stick(output.dtype)

    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]
    out_stl = SpyreTensorLayout(
        c_size, c_stride, output.dtype, list(range(len(output.size)))
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
        output, output_dep, c_size, c_stride, stick_size, stick_expr
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
    reduction_var = find_reduction_var(x.dep, output_dep)
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
    reduction_var = find_reduction_var(x.dep, output_dep)
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


def find_stick_compatible_input_layout(
    arg: PropArg,
    reduction_var: sympy.Symbol,
    reduction_type: str,
    label: str,
) -> SpyreTensorLayout:
    """Find the required STL for a matmul input by iterating all candidate layouts.

    1. Return the first layout whose stick already carries reduction_var (zero cost).
    2. Else return the first layout that can be restickified to put reduction_var on the stick.
    3. Else raise Unsupported.
    """
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
    for stl, dev_coords in candidates:
        if reduction_var in dev_coords[-1].free_symbols:
            return stl

    # Pass 2: can be restickified — find the resolvable device coord for reduction_var
    # and use it as target_stick_expr for compute_restickify_target_layout.
    arg_host_coords = host_coordinates(arg.layout, arg.dep, None)
    for stl, dev_coords in candidates:
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

    x_dep, y_dep = identify_matmul_inputs([a.dep for a in args], output_dep)
    if x_dep is None or y_dep is None:
        raise Unsupported(f"{data.reduction_type}: could not identify Input1/Input2")
    # Map identified deps back to PropArgs.
    if x_dep is args[0].dep:
        x, y = args[0], args[1]
    else:
        x, y = args[1], args[0]

    # Hardware stick constraints (DF16):
    #   Input1 (x): stick on reduction_var (loop var absent from output)
    #   Input2 (y): stick on generated_var (loop var present in output, absent from x)
    #   Output:     stick on generated_var
    reduction_var = find_reduction_var(x.dep, output_dep)
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
            f"{data.reduction_type}: generated_var={generated_var} not found in output coords {out_coords}"
        )

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

    # Determine output EA based on input EAs. The full EA-compatibility rule is
    # enforced later by validate_ops via the shared is_ea_compatible predicate;
    # here we only reject the one case propagation itself cannot represent
    # (more than one distinct staggered EA) and otherwise pick the output EA.
    # We deliberately do NOT run is_ea_compatible here: validate_ops skips
    # layernorm ops carrying EXX2, and this join point sees those ops too, so a
    # blanket gate here would over-reject valid layernorm/EXX2 combinations.
    staggered_inputs = input_eas & STAGGERED_EAS

    if len(staggered_inputs) > 1:
        # Multiple different staggered EAs - not supported
        raise Unsupported(
            f"Multi-arg pointwise with multiple staggered EAs not supported: {input_eas}"
        )
    elif len(staggered_inputs) == 1:
        # One staggered EA mixed with STANDARD inputs (the broadcast pattern).
        output_ea = next(iter(staggered_inputs))

        # A STANDARD operand can broadcast against a staggered-EA operand only if
        # its device *stick* dimension enumerates at most one distinct host
        # element, i.e. the stick maps to a size-1 (broadcast) host axis. The
        # element arrangement is a device-layout property, so we must test the
        # host axis the device stick actually maps to, not a fixed host axis.
        #
        # In an STL the stick is the last device dim and `stride_map[-1]` is its
        # host stride; the layout constructor (spyre_tensor_impl.cpp) sets that
        # entry to -1 exactly when the mapped host axis has size 1 (or the stick
        # is sparse). So `stride_map[-1] == -1` is the correct, dim_order-
        # independent test. Reading `arg.layout.size[-1]` instead only works when
        # dim_order is the identity (stick == last host dim); under a non-identity
        # dim_order the device stick may map to a size-1 axis that is not last
        # (e.g. host size [1, 64, 1] with the stick on a size-1 axis), which the
        # trailing-dim check would mishandle.
        for arg in args:
            if not arg.layouts:
                continue
            for stl in arg.layouts:
                if stl.element_arrangement != ElementArrangement.STANDARD:
                    continue
                if len(arg.layout.size) == 0:
                    # Scalar - always compatible.
                    continue
                if stl.stride_map[-1] == -1:
                    # Device stick maps to a size-1 (broadcast) / sparse host
                    # axis: compatible.
                    continue
                # Stick maps to a real host axis; identify it for the message.
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
                    f"with a staggered EA. Its stick maps to host dim {mapped} of size "
                    f"{mapped_size}"
                )
    else:
        # All STANDARD or other EAs - use STANDARD
        output_ea = ElementArrangement.STANDARD

    ind_names, _, ind_sizes = indirect_info_from_op(op)
    stick_exprs = {
        device_coordinates(stl, arg.dep, ind_sizes)[-1]
        for arg in args
        for stl in arg.layouts
        if arg.dep.name not in ind_names
    }

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
                or not same_device_size(arg.layout.dtype, output.dtype)
            ):
                can_use_same_layout = False
                break

    stick_size = get_elem_in_stick(output.dtype)
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
                c_in_size, c_in_stride, output.dtype, projected_dim_order, output_ea
            )
            coord = device_coordinates(in_stl, arg.dep, ind_sizes)
            if not is_stick_expr_offset_free(coord[-1], stick_size):
                return False
        return True

    def _try_stick_dim(stick_dim):
        dim_order = _compute_dim_order(stick_dim, c_size, out_coords)
        if _is_supported_layout(dim_order):
            results.append(
                SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order, output_ea)
            )

    results: list[SpyreTensorLayout] = []

    if can_use_same_layout:
        template_stl = next(iter(args[0].layouts))
        results.append(
            SpyreTensorLayout(
                template_stl.device_size,
                template_stl.stride_map,
                get_device_dtype(output.dtype),
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

    # Try alternative layouts if no valid layouts found
    if not results:
        for alt_stick_dim in range(len(output.size) - 1):
            # TODO: Support dimensions with size not divisible by stick_size via padding (See #1756)
            if concretize_expr(output.size[alt_stick_dim]) % stick_size != 0:
                continue
            _try_stick_dim(alt_stick_dim)

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
        raise Unsupported(
            f"Multi-arg pointwise ({op.get_name()}): no supported output layout found "
            f"with size={output.size} and coordinates={out_coords}"
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
    reduction_var = find_reduction_var(x.dep, output_dep)

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

    if isinstance(data, Reduction) and data.reduction_type == BATCH_MATMUL_OP:
        return _matmul_layouts(op, output, output_dep, args)

    if isinstance(data, Reduction) and data.reduction_type == "exx2":
        return _exx2_layout(op, output, output_dep, args)

    if isinstance(data, Reduction) and data.reduction_type in TOPK_OPS:
        return _topk_layouts(op, output, output_dep, args)

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
    """
    output: FixedLayout = op.get_layout()
    c_size = [concretize_expr(s) for s in output.size]
    c_stride = [concretize_expr(s) for s in output.stride]
    layouts = [
        SpyreTensorLayout(
            c_size,
            c_stride,
            output.dtype,
            [d for d in range(len(c_size)) if d != stick_dim] + [stick_dim],
        )
        for stick_dim in range(len(c_size))
        if c_size[stick_dim] > 1  # no sticks of size 1
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
    stick_size = get_elem_in_stick(target_layout.dtype)
    write_stick = device_coordinates(target_stl, output_dep, None)[-1]
    if is_stick_expr_offset_free(write_stick, stick_size):
        return None

    c_size = [concretize_expr(s) for s in target_layout.size]
    c_stride = [concretize_expr(s) for s in target_layout.stride]
    candidates = _candidate_output_stls(
        target_layout, output_dep, c_size, c_stride, stick_size, write_stick
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
                    # All spyre tensors are created with device layouts.
                    # Therefore we expect all graph inputs to have them.
                    raise Unsupported(
                        f"missing device_tensor_layout on graph input {name}"
                    )
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
                target_stl = _target_device_layout(target, target_name)
                if target_stl is None:
                    continue
                rw = op.get_read_writes()
                output_dep = next(iter(rw.writes))
                args = _get_prop_args(rw.reads)

                # Find an alternative layout if the write has an unsupported stick
                # expression (e.g. offset like v+32). Force the optimizer to use
                # this layout for the mutation target.
                target_layout = target.get_layout()
                if isinstance(target_layout, FixedLayout):
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
                    op.layouts = _all_constant_layouts(op)
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
        elif isinstance(op, DeviceCopy):
            # spyre -> cpu: the output is a host tensor and carries no Spyre
            #     layout. Leave `.layouts` unset.
            # cpu -> spyre: the output is a fresh on-device buffer with no
            #     inherited tiling, so give it a new device layout.
            if op.get_layout().device.type == DEVICE_NAME:
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
