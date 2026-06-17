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

import sympy
import torch
from .logging_utils import get_inductor_logger
from torch._inductor.ir import (
    ComputedBuffer,
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

from torch_spyre._C import (
    ElementArrangement,
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
)
from .errors import Unsupported
from . import config
from .constants import (
    BATCH_MATMUL_OP,
    COPY_BACK_CANDIDATE_ATTR,
    ELIDED_COPY_BACK_ATTR,
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
    is_stick_expr_offset_free,
    indirect_index_dep_names,
    indirect_access_subs_from_op,
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
    out_coords = host_coordinates(output, output_dep)
    out_stick_dim = _pick_stick_dim(stick_expr, out_coords)
    return _make_output_stl(output, output_dep, c_size, c_stride, out_stick_dim)


def _make_output_stl(
    output, output_dep, c_size, c_stride, stick_dim
) -> SpyreTensorLayout | None:
    """Build a candidate output STL with stick_dim last and verify the resulting stick is offset-free.

    Returns None if the resulting stick expression has an offset.
    """
    stick_size = get_elem_in_stick(output.dtype)
    out_coords = host_coordinates(output, output_dep)
    dim_order = _compute_dim_order(stick_dim, c_size, out_coords)
    stl = SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)
    coords = device_coordinates(stl, output_dep)
    if is_stick_expr_offset_free(coords[-1], stick_size):
        return stl
    return None


def _candidate_output_stls(
    output: FixedLayout,
    output_dep: MemoryDep,
    c_size: list,
    c_stride: list,
    stick_size: int,
) -> list[SpyreTensorLayout]:
    """Enumerate candidate output STLs by trying each non-last dim as the stick.

    Skips any candidate whose resulting device stick expression has an offset.
    """
    result = []
    for alt_stick_dim in range(len(output.size) - 1):
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
        for stl in arg.layouts:
            stick_expr = device_coordinates(stl, arg.dep)[-1]
            if not is_stick_expr_offset_free(stick_expr, stl.elems_per_stick()):
                raise Unsupported(
                    f"{op_label}: input arg{i} has stick expression with offset "
                    f"{stick_expr!r} (likely from slicing the stick dimension); "
                    f"this op requires a fixed input layout and double-restickify is not yet supported"
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
        x_dev_coords = device_coordinates(stl, dep)
        out_coords = host_coordinates(output, output_dep)
        x_stick_expr = x_dev_coords[-1]

        # Try to preserve input layout
        out_stl = _output_stl_from_stick_expr(
            x_stick_expr, output, output_dep, c_size, c_stride
        )
        if out_stl is not None:
            return [out_stl]

        # Try alternative layouts when input layout is not supported
        in_coords = host_coordinates(in_layout, dep)
        reduction_var = next(
            iter(dep.index.free_symbols - output_dep.index.free_symbols), None
        )
        layouts = []
        for in_dim in range(len(in_layout.size)):
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
            in_stick_expr = device_coordinates(stl, dep)[-1]
            if not is_stick_expr_offset_free(in_stick_expr, stl.elems_per_stick()):
                return []

            in_elems_per_stick = get_elem_in_stick(in_layout.dtype)
            stick_dim_size = in_layout.size[-1]
            fmt = ElementArrangement.STANDARD
            unaligned = concretize_expr(stick_dim_size % in_elems_per_stick)

            if unaligned > 0:
                outer_sizes = [concretize_expr(s) for s in output.size[:-1]]
                outer_strides = [concretize_expr(s) for s in output.stride[:-1]]
                c_size = outer_sizes + [in_elems_per_stick]
                c_stride = outer_strides + [1]
                if in_layout.dtype == torch.float16 and output.dtype == torch.float32:
                    fmt = ElementArrangement.DL16_TO_FP32

            stl = SpyreTensorLayout(
                c_size, c_stride, output.dtype, list(range(len(c_size))), fmt
            )
            return [stl]

    in_coords = host_coordinates(in_layout, dep)
    out_coords = host_coordinates(output, output_dep)
    if (
        in_coords == out_coords
        and in_layout.size == output.size
        and dep.index == output_dep.index
        and same_device_size(in_layout.dtype, output.dtype)
    ):
        # Input and output tensors are being accessed identically and elem size is the same.
        # We can simply propagate the device_layout.
        stl = SpyreTensorLayout(
            stl.device_size, stl.stride_map, get_device_dtype(output.dtype)
        )
        return [stl]

    in_device_coords = device_coordinates(stl, dep)
    stick_expr = in_device_coords[-1]

    # Try to preserve input layout, fall back to scanning all output dims
    out_stl = _output_stl_from_stick_expr(
        stick_expr, output, output_dep, c_size, c_stride
    )
    if out_stl is not None:
        return [out_stl]
    return _candidate_output_stls(output, output_dep, c_size, c_stride, stick_size)


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
    in_device_coords = device_coordinates(in_stl, in_dep)
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
    out_coords = host_coordinates(output, output_dep)
    in_layout = args[0].layout
    in_host_coords = host_coordinates(in_layout, in_dep)
    required_in_stl = None
    for candidate in _candidate_output_stls(
        output, output_dep, c_size, c_stride, stick_size
    ):
        target_stick = device_coordinates(candidate, output_dep)[-1]
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
    arg_dev_coords = [device_coordinates(stl, arg.dep) for stl in arg.layouts]

    # Pass 1: already stick-compatible.
    # stick_compatible() checks cross-tensor compatibility; here we only need
    # to know if this input's stick coord already carries the target loop variable.
    for stl, dev_coords in zip(arg.layouts, arg_dev_coords):
        if reduction_var in dev_coords[-1].free_symbols:
            return stl

    # Pass 2: can be restickified — find the resolvable device coord for reduction_var
    # and use it as target_stick_expr for compute_restickify_target_layout.
    arg_host_coords = host_coordinates(arg.layout, arg.dep)
    for stl, dev_coords in zip(arg.layouts, arg_dev_coords):
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
    out_coords = host_coordinates(output, output_dep)

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
       3. Construct the AllSameNode cost function since in and out sticks must always match
    """

    indirect_index_names = indirect_index_dep_names(op)
    # Collect all unique non-zero stick expressions from input layouts
    stick_exprs = {
        stick_expr
        for arg in args
        for stl in arg.layouts
        if arg.dep.name not in indirect_index_names
        and (stick_expr := device_coordinates(stl, arg.dep)[-1]) != 0
    }

    # If the indexing and device element size are identical
    # across all inputs and the output we can just propagate the device layout.
    in_coords = [host_coordinates(arg.layout, arg.dep) for arg in args]
    out_coords = host_coordinates(output, output_dep)
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
                c_in_size, c_in_stride, output.dtype, projected_dim_order
            )
            coord = device_coordinates(in_stl, arg.dep)
            if not is_stick_expr_offset_free(coord[-1], stick_size):
                return False
        return True

    def _try_stick_dim(stick_dim):
        dim_order = _compute_dim_order(stick_dim, c_size, out_coords)
        if _is_supported_layout(dim_order):
            results.append(SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order))

    results: list[SpyreTensorLayout] = []

    if can_use_same_layout:
        template_stl = next(iter(args[0].layouts))
        results.append(
            SpyreTensorLayout(
                template_stl.device_size,
                template_stl.stride_map,
                get_device_dtype(output.dtype),
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
    x_coords = host_coordinates(x.layout, x.dep)
    out_coords = host_coordinates(output, output_dep)

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
        x_stick_expr = device_coordinates(stl, x.dep)[-1]
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

    # Log substituted device coordinates for indirect index args. Useful for
    # debugging gather/scatter layout propagation, and also the canonical
    # example of how to call device_coordinates() with indirect_load_subs
    # pre-scheduler (keeping indirect_access_subs_from_op exercised and visible).
    if logger.isEnabledFor(logging.DEBUG):
        indirect_index_names = indirect_index_dep_names(op)
        if indirect_index_names:
            indirect_subs = indirect_access_subs_from_op(op)
            for arg in args:
                if arg.dep.name in indirect_index_names:
                    continue
                for j, stl in enumerate(arg.layouts):
                    try:
                        d_coords: object = device_coordinates(
                            stl, arg.dep, indirect_subs
                        )
                    except Exception:
                        d_coords = "<error>"
                    logger.debug(
                        f"  indirect value {arg.dep.name} STL[{j}] d_coords (with IndirectAccess subs)={d_coords}"
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
        if isinstance(producer.layout, MutationLayoutSHOULDREMOVE):
            continue
        if config.chunk_large_tensors and isinstance(producer.data, Pointwise):
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

    # Operations are in topological order (guaranteed by GraphLowering).
    # Visit them and use the input SpyreTensorLayouts and the operation being
    # performed to compute the set of possible output SpyreTensorLayouts
    it = iter(operations)
    for op in it:
        if op.is_no_op():
            op.layouts = [generic_layout(op)]
            op.restick_cost_fn = AnyInNode.from_args()
        elif isinstance(op, ComputedBuffer):
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                target = op.layout.target
                while isinstance(target, ReinterpretView):
                    target = target.data
                target_stl = _target_device_layout(
                    target,
                    target.get_name() if hasattr(target, "get_name") else "",
                )
                if target_stl is None:
                    continue
                rw = op.get_read_writes()
                output_dep = next(iter(rw.writes))
                args = _get_prop_args(rw.reads)
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
            op = next(it, None)
            if not isinstance(op, MultiOutput):
                raise RuntimeError("FallbackKernel must be followed by MultiOutput")
            op.layouts = [generic_layout(op)]
            op.restick_cost_fn = AnyInNode.from_args()
        elif isinstance(op, SpyreConstantFallback):
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
