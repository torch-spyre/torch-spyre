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


from typing import NamedTuple

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
    Operation,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.virtualized import V

from torch_spyre._C import (
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
)
from .errors import Unsupported
from .constants import BATCH_MATMUL_OP, TOPK_OPS
from .ir import FixedTiledLayout, SpyreConstantFallback
from .pass_utils import (
    compute_restickify_target_layout,
    concretize_expr,
    host_coordinates,
    device_coordinates,
    iter_var_id,
)
from .optimize_restickify import AllSameNode, AnyInNode, FixedInOutNode
from .views import matching_dim

# ---------------------------------------------------------------------------
# TODO(issue#1371): once SpyreTensorLayout is migrated to c10::SymInt, all
# concretize_expr calls in this file can be removed.
# ---------------------------------------------------------------------------

logger = get_inductor_logger("propagate_layouts")

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
            if hasattr(buf, "layouts"):
                res.append(PropArg(arg, layout, list(buf.layouts)))
            else:
                if not isinstance(layout, FixedTiledLayout):
                    raise RuntimeError(f"{buf} does not have FixedTiledLayout")
                res.append(PropArg(arg, layout, [layout.device_layout]))
    return res


def same_device_size(t1: torch.dtype, t2: torch.dtype) -> bool:
    return get_elem_in_stick(t1) == get_elem_in_stick(t2)


def _single_arg_op_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    dep: MemoryDep,
    in_layout: FixedLayout,
    stl: SpyreTensorLayout,
) -> SpyreTensorLayout:
    """
    Compute the output STL for a single-arg op given one candidate input STL.
    Called once per candidate input STL to produce the corresponding output STL.
    """
    data = op.data

    if isinstance(data, Reduction):
        if data.reduction_type == "exx2":
            x_coords = host_coordinates(in_layout, dep)
            x_dev_coords = device_coordinates(stl, dep)
            x_stick_expr = x_dev_coords[-1]
            x_stick_dim = matching_dim(x_coords, x_stick_expr)
            if x_stick_dim is None or x_stick_dim != len(in_layout.size) - 1:
                # TODO: Insert a restickify to enable the operation to be performed
                raise Unsupported(f"exx2: illegal device layout {stl}")
            dim_order = list(range(len(output.size))) + [-1]
            c_size = [concretize_expr(s) for s in output.size]
            c_stride = [concretize_expr(s) for s in output.stride]
            return SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)
        else:
            # Propagate input stick to output if the dim survives, else put stick last.
            x_coords = host_coordinates(in_layout, dep)
            x_dev_coords = device_coordinates(stl, dep)
            out_coords = host_coordinates(output, output_dep)
            x_stick_expr = x_dev_coords[-1]
            out_stick_dim = matching_dim(out_coords, x_stick_expr)
            if out_stick_dim is None:
                out_dim_order = list(range(len(output.size))) + [-1]
            else:
                out_dim_order = [
                    d for d in range(len(output.size)) if d != out_stick_dim
                ]
                out_dim_order = out_dim_order + [out_stick_dim]
            c_size = [concretize_expr(s) for s in output.size]
            c_stride = [concretize_expr(s) for s in output.stride]
            return SpyreTensorLayout(c_size, c_stride, output.dtype, out_dim_order)

    # Single-arg pointwise
    assert isinstance(data, Pointwise)
    origin_node = next(iter(data.origins))
    aten_op = origin_node.target
    match aten_op:
        case aten.clone.default:
            # Clone is generated by an explicit `contiguous()`; on spyre that means use the default row major tiling.
            # Concretize for C++ SpyreTensorLayout constructor.
            c_size = [concretize_expr(s) for s in output.size]
            c_stride = [concretize_expr(s) for s in output.stride]
            return SpyreTensorLayout(
                c_size,
                c_stride,
                output.dtype,
                list(range(len(output.size))),
            )

        case spyreop.overwrite.default:
            return SpyreTensorLayout(output.size, output.dtype)

        case _:
            in_coords = host_coordinates(in_layout, dep)
            out_coords = host_coordinates(output, output_dep)
            if (
                in_coords == out_coords
                and dep.index == output_dep.index
                and same_device_size(in_layout.dtype, output.dtype)
            ):
                # Input and output tensors are being accessed identically and elem size is the same.
                # We can simply propagate the device_layout.
                return SpyreTensorLayout(
                    stl.device_size,
                    stl.stride_map,
                    get_device_dtype(output.dtype),
                )
            else:
                # TODO: We should be able to preserve the input stride_map
                #       unless the operation is changing elems_per_stick.
                #       For now, use the default layout for a mostly row major dimension
                #       ordering, adjusted to put the stick dimension last and move all
                #       non-stick size one dimensions to the right to avoid tiling them.
                in_device_coords = device_coordinates(stl, dep)
                stick_expr = in_device_coords[-1]
                maybe_stick_dim = matching_dim(out_coords, stick_expr)
                out_stick_dim = -1 if maybe_stick_dim is None else maybe_stick_dim
                dim_order = [
                    d
                    for d in range(len(output.size))
                    if d != out_stick_dim and out_coords[d] != 0
                ]
                dim_order += [
                    d
                    for d in range(len(output.size))
                    if d != out_stick_dim and out_coords[d] == 0
                ]
                dim_order += [out_stick_dim]
                # Concretize for C++ SpyreTensorLayout constructor.
                c_size = [concretize_expr(s) for s in output.size]
                c_stride = [concretize_expr(s) for s in output.stride]
                return SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)


def _matmul_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[PropArg],
) -> list[SpyreTensorLayout]:
    """
    Matmul has fixed in/out stick requirements so handled specially.
    Algorithm is
       1. For both input args, compuate a layout that is representative
       2. For output arg, compute the output layout
       3. Construct the FixdInOutNode cost function
    """
    data = op.data
    out_coords = host_coordinates(output, output_dep)

    x = args[0]
    y = args[1]
    x_stl = next(iter(x.layouts))
    y_stl = next(iter(y.layouts))
    x_coords = host_coordinates(x.layout, x.dep)
    x_dev_coords = device_coordinates(x_stl, x.dep)
    y_coords = host_coordinates(y.layout, y.dep)
    y_dev_coords = device_coordinates(y_stl, y.dep)

    x_stick_expr = x_dev_coords[-1]
    y_stick_expr = y_dev_coords[-1]
    if (
        matching_dim(x_coords, x_stick_expr) is None
        or matching_dim(y_coords, y_stick_expr) is None
    ):
        raise Unsupported(
            f"{data.reduction_type}: failed to map stick_dims to host coords"
        )

    # Hardware stick constraints (DF16):
    #   Input1 (x): stick on reduction_dim (the x coord that does NOT appear in output)
    #   Input2 (y): stick on generated_dim (the y coord that appears in output)
    #   Output:     stick on generated_dim
    if matching_dim(out_coords, x_stick_expr) is not None:
        reduction_coord = next(
            c
            for c in x_coords
            if len(c.free_symbols) > 0 and matching_dim(out_coords, c) is None
        )
    else:
        reduction_coord = x_stick_expr

    if matching_dim(out_coords, y_stick_expr) is None:
        generated_coord = next(
            c
            for c in y_coords
            if len(c.free_symbols) > 0
            and matching_dim(out_coords, c) is not None
            and matching_dim(x_coords, c) is None
        )
    else:
        generated_coord = y_stick_expr

    if reduction_coord == x_dev_coords[-1]:
        x_req_stl = x_stl
    else:
        _x = compute_restickify_target_layout(
            x_stl, x.layout, reduction_coord, x_coords, x_dev_coords
        )
        if _x is None:
            raise Unsupported(
                f"{data.reduction_type}: cannot restickify x to reduction_coord={reduction_coord}"
            )
        x_req_stl = _x

    if generated_coord == y_dev_coords[-1]:
        y_req_stl = y_stl
    else:
        _y = compute_restickify_target_layout(
            y_stl, y.layout, generated_coord, y_coords, y_dev_coords
        )
        if _y is None:
            raise Unsupported(
                f"{data.reduction_type}: cannot restickify y to generated_coord={generated_coord}"
            )
        y_req_stl = _y

    out_stick_dim = matching_dim(out_coords, generated_coord)
    if out_stick_dim is None:
        raise Unsupported(
            f"{data.reduction_type}: failed to map output stick_dim to host coords {out_coords} {generated_coord}"
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
        [x, y], out_stl, [x_req_stl, y_req_stl]
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
       1. Compute set of output stick expressions possible given the input layouts
       2. Compute an out STL for each
       3. Construct the AllSameNode cost function since in and out sticks must always match
    """
    stick_exprs = {
        device_coordinates(stl, arg.dep)[-1]
        for arg in args
        for stl in arg.layouts
        if device_coordinates(stl, arg.dep)[-1] != 0
    }

    if len(stick_exprs) > 1:
        logger.info(
            f"Multi-stick pointwise ({op.get_name()}): producing {len(stick_exprs)} output layouts."
        )

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
                or arg.dep.index != output_dep.index
                or not same_device_size(arg.layout.dtype, output.dtype)
            ):
                can_use_same_layout = False
                break

    results: list[SpyreTensorLayout] = []
    # Sort stick exprs for determinism
    for stick_expr in sorted(stick_exprs, key=iter_var_id) if stick_exprs else [None]:
        if can_use_same_layout:
            template_stl = next(iter(args[0].layouts))
            stl = SpyreTensorLayout(
                template_stl.device_size,
                template_stl.stride_map,
                get_device_dtype(output.dtype),
            )
        else:
            if stick_expr is None:
                out_stick_dim = -1
            else:
                maybe_stick_dim = matching_dim(out_coords, stick_expr)
                out_stick_dim = -1 if maybe_stick_dim is None else maybe_stick_dim
            dim_order = [
                d
                for d in range(len(output.size))
                if d != out_stick_dim and out_coords[d] != 0
            ]
            dim_order += [
                d
                for d in range(len(output.size))
                if d != out_stick_dim and out_coords[d] == 0
            ]
            dim_order += [out_stick_dim]
            c_size = [concretize_expr(s) for s in output.size]
            c_stride = [concretize_expr(s) for s in output.stride]
            stl = SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)
        results.append(stl)
    op.restick_cost_fn = AllSameNode.from_args(args, results, output_dep)
    return results


def _topk_layouts(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[PropArg],
) -> list[SpyreTensorLayout]:
    x = args[0]
    x_coords = host_coordinates(x.layout, x.dep)
    out_coords = host_coordinates(output, output_dep)

    # Reduction coordinate: in x's host coords but absent from output's host coords.
    reduction_coord = next(
        c
        for c in x_coords
        if len(c.free_symbols) > 0 and matching_dim(out_coords, c) is None
    )
    reduction_dim = matching_dim(x_coords, reduction_coord)

    # Coords that survive the reduction into the output.
    surviving_coords = [
        c
        for c in x_coords
        if len(c.free_symbols) > 0 and matching_dim(out_coords, c) is not None
    ]

    # Collect candidate output stick dims. A valid input stick passes through;
    # a stick on the reduction dim requires a restickify, so every surviving
    # coord becomes a candidate.
    out_stick_dims: set[int | None] = set()
    for stl in x.layouts:
        x_stick_expr = device_coordinates(stl, x.dep)[-1]
        if matching_dim(x_coords, x_stick_expr) == reduction_dim:
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

    op.restick_cost_fn = AllSameNode.from_args(args, results, output_dep)
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

    if len(args) > 1 and isinstance(data, Pointwise):
        return _multi_arg_pointwise_layouts(op, output, output_dep, args)

    if isinstance(data, Reduction) and data.reduction_type == BATCH_MATMUL_OP:
        return _matmul_layouts(op, output, output_dep, args)

    if isinstance(data, Reduction) and data.reduction_type in TOPK_OPS:
        return _topk_layouts(op, output, output_dep, args)

    aten_op = next(iter(data.origins)).target if data.origins else None
    if aten_op == spyreop.layernormnorm.default:
        # layernormnorm is pointwise but special: it has multiple args, input and output
        # must have matching size/stride, but only the first arg drives the output layout.
        in_layout = args[0].layout
        if in_layout.size != output.size or in_layout.stride != output.stride:
            raise Unsupported(
                f"views not supported for spyre.layernormnorm({in_layout.size})=>{output.size})"
            )
        layouts = [
            _single_arg_op_layout(op, output, output_dep, args[0].dep, in_layout, stl)
            for stl in args[0].layouts
        ]
        op.restick_cost_fn = AllSameNode.from_args(args[:1], layouts, output_dep)
        return layouts

    if aten_op == aten.clone.default:
        # clone materializes a new buffer in a fixed row-major layout regardless of
        # input stick — equivalent to a restickify. No restickify before it is needed.
        stl = _single_arg_op_layout(
            op,
            output,
            output_dep,
            args[0].dep,
            args[0].layout,
            next(iter(args[0].layouts)),
        )
        op.restick_cost_fn = AnyInNode.from_args()
        return [stl]

    # All other single arg ops
    layouts = [
        _single_arg_op_layout(op, output, output_dep, args[0].dep, args[0].layout, stl)
        for stl in args[0].layouts
    ]
    op.restick_cost_fn = AllSameNode.from_args(args, layouts, output_dep)
    return layouts


def generic_layout(op: Operation) -> SpyreTensorLayout:
    output: FixedLayout = op.get_layout()
    # Concretize for C++ SpyreTensorLayout constructor.
    c_size = [concretize_expr(s) for s in output.size]
    return SpyreTensorLayout(c_size, output.dtype)


def propagate_spyre_tensor_layouts(
    operations: list[Operation],
) -> None:
    # Convert InputBuffers from FixedLayout to SpyreTensorLayouts
    if len(V.graph.graph_input_names) > 0:
        for name, real_input in zip(V.graph.graph_input_names, V.get_real_inputs()):
            if isinstance(real_input, torch.Tensor):
                stl = real_input.device_tensor_layout()
                if stl is None:
                    # All spyre tensors are created with device layouts.
                    # Therefore we expect all graph inputs to have them.
                    raise Unsupported(
                        f"missing device_tensor_layout on graph input {name}"
                    )
                tb = V.graph.graph_inputs[name]
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
                continue
            op.decide_layout()
            rw = op.get_read_writes()
            output_dep = next(iter(rw.writes))
            args = _get_prop_args(rw.reads)
            output = op.get_layout()
            if isinstance(op.data, (Pointwise, Reduction)):
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
        if isinstance(n.node.data, Pointwise):
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
        else:
            logger.warning(
                f"propagate_mutation_layouts: unhandled mutation op {type(n.node.data)}"
            )

    return nodes
