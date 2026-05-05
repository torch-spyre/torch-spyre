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

import logging
from typing import Any
from .ir import SpyreConstantFallback

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
from .constants import BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .pass_utils import (
    SchedNodeArg,
    concretize_expr,
    get_mem_deps_from_rw,
    host_coordinates,
    device_coordinates,
)
from .views import matching_dim
# ---------------------------------------------------------------------------
# TODO(issue#1371): once SpyreTensorLayout is migrated to c10::SymInt, all
# concretize_expr calls in this file can be removed.
# ---------------------------------------------------------------------------

logger = get_inductor_logger("stickify")

aten = torch.ops.aten
spyreop = torch.ops.spyre


def same_device_size(t1: torch.dtype, t2: torch.dtype) -> bool:
    return get_elem_in_stick(t1) == get_elem_in_stick(t2)


def restickify_device_size(
    old_device_size: list,
    idc: list,
    old_stick_expr,
    target_stick_expr,
    host_size: list,
    old_sd: int,
    new_sd: int,
    stick_size: int = 64,
) -> list:
    """Compute device_size for a restickify by swapping old_stick_expr with target_stick_expr.

    Uses idc coordinate expressions to identify which device dims cover old_sd vs new_sd:
    - Last device dim (stick): size is always stick_size
    - Dims involving old_var (outer stick): size becomes host_size[new_sd] // stick_size
    - Dims involving new_var (non-stick for new_sd): size becomes host_size[old_sd]
    - coord==0 with no explicit outer stick: degenerate outer stick (host_size[old_sd] == stick_size);
      absorbs the new outer stick when host_size[new_sd] > stick_size
    - All other dims: unchanged
    """
    old_var = next(iter(old_stick_expr.free_symbols))
    new_var = next(iter(target_stick_expr.free_symbols))
    # True when old_sd has an explicit outer-stick dim (host_size[old_sd] > stick_size).
    # False means old_sd fit in one stick: the degenerate outer stick has coord==0, size==1.
    has_old_outer_stick = any(
        old_var in idc[j].free_symbols for j in range(len(idc) - 1)
    )
    result = []
    for j, coord in enumerate(idc):
        if j == len(idc) - 1:
            # Last device dim is always the stick, size is always stick_size.
            result.append(stick_size)
        elif old_var in coord.free_symbols:
            result.append(host_size[new_sd] // stick_size)
        elif new_var in coord.free_symbols:
            result.append(host_size[old_sd])
        elif (
            coord == sympy.S.Zero
            and not has_old_outer_stick
            and host_size[new_sd] > stick_size
        ):
            # Degenerate outer stick (host_size[old_sd] == stick_size, so floor(old_var/64) == 0).
            # This placeholder absorbs the new outer stick when new_sd requires tiling.
            result.append(host_size[new_sd] // stick_size)
        else:
            result.append(old_device_size[j])
    return result


def restickify_stride_map(
    old_stride_map: list,
    idc: list,
    old_stick_expr,
    target_stick_expr,
    host_size: list,
    host_stride: list,
    old_sd: int,
    new_sd: int,
    stick_size: int = 64,
) -> list:
    """Compute stride_map for a restickify by swapping old_stick_expr with target_stick_expr.

    Uses idc coordinate expressions to identify which device dims cover old_sd vs new_sd,
    then rescales their stride_map values by host_stride[new_sd] / host_stride[old_sd]
    and vice versa. All other dims are unchanged.
    """
    old_var = next(iter(old_stick_expr.free_symbols))
    new_var = next(iter(target_stick_expr.free_symbols))
    # True when old_sd has an explicit outer-stick dim (host_size[old_sd] > stick_size).
    # False means old_sd fit in one stick: the degenerate outer stick has coord==0, size==1.
    has_old_outer_stick = any(
        old_var in idc[j].free_symbols for j in range(len(idc) - 1)
    )
    result = []
    for j, coord in enumerate(idc):
        if j == len(idc) - 1:
            # Last device dim is always the stick; stride_map is the host stride of new_sd.
            result.append(host_stride[new_sd])
        elif old_var in coord.free_symbols:
            result.append(
                old_stride_map[j] * host_stride[new_sd] // host_stride[old_sd]
            )
        elif new_var in coord.free_symbols:
            result.append(
                old_stride_map[j] * host_stride[old_sd] // host_stride[new_sd]
            )
        elif (
            coord == sympy.S.Zero
            and not has_old_outer_stick
            and host_size[new_sd] > stick_size
        ):
            # Degenerate outer stick (host_size[old_sd] == stick_size, so floor(old_var/64) == 0).
            # This placeholder absorbs the new outer stick when new_sd requires tiling.
            result.append(host_stride[new_sd] * stick_size)
        else:
            result.append(old_stride_map[j])
    return result


def schedule_restickify(
    op: Operation,
    arg: SchedNodeArg,
    target_stick_expr,
    ic: list,
    idc: list,
    restickify_plan: dict,
) -> FixedTiledLayout:
    """Record a restickify needed for arg to match target_stick_expr.

    Computes the target FixedTiledLayout by replacing the current stick
    coordinate expression with target_stick_expr in the device layout, then
    appends an entry to restickify_plan[op] for the insert_restickify pass to act on.
    """
    dl = arg.layout.device_layout
    new_sd = matching_dim(ic, target_stick_expr)
    assert new_sd is not None, (
        f"Could not find a host dimension matching stick expr {target_stick_expr} in {ic}"
    )
    host_size = list(arg.layout.size)
    host_stride = list(arg.layout.stride)
    old_sd = matching_dim(ic, idc[-1])
    assert old_sd is not None, (
        f"Could not find a host dimension matching current stick expr {idc[-1]} in {ic}"
    )
    old_stick_expr = idc[-1]
    old_stride_map = list(dl.stride_map)

    device_size = restickify_device_size(
        list(dl.device_size),
        idc,
        old_stick_expr,
        target_stick_expr,
        host_size,
        old_sd,
        new_sd,
    )
    stride_map = restickify_stride_map(
        old_stride_map,
        idc,
        old_stick_expr,
        target_stick_expr,
        host_size,
        host_stride,
        old_sd,
        new_sd,
    )

    stl = SpyreTensorLayout(device_size, stride_map, dl.device_dtype)

    target_layout = FixedTiledLayout(
        arg.layout.device, arg.layout.dtype, arg.layout.size, arg.layout.stride, stl
    )
    restickify_plan.setdefault(op.get_name(), []).append(
        {"arg_name": arg.dep.name, "target_layout": target_layout}
    )
    return target_layout


def pointwise_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[SchedNodeArg],
    restickify_plan: dict[str, list[dict[str, Any]]],
) -> FixedTiledLayout:
    data = op.data
    origin_node = next(iter(data.origins))
    aten_op = origin_node.target

    if len(args) == 1:
        x = args[0]
        match aten_op:
            case aten.clone.default:
                # Clone is generated by an explicit `contiguous()`; on spyre that means use the default row major tiling.
                # Concretize for C++ SpyreTensorLayout constructor.
                c_size = [concretize_expr(s) for s in output.size]
                c_stride = [concretize_expr(s) for s in output.stride]
                stl = SpyreTensorLayout(
                    c_size,
                    c_stride,
                    output.dtype,
                    list(range(len(output.size))),
                )
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )

            case spyreop.overwrite.default:
                stl = SpyreTensorLayout(output.size, output.dtype)
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )
            case _:
                x_stl = x.layout.device_layout
                in_coords = host_coordinates(x.layout, x.dep)
                out_coords = host_coordinates(output, output_dep)
                if (
                    in_coords == out_coords
                    and x.dep.index == output_dep.index
                    and same_device_size(x.layout.dtype, output.dtype)
                ):
                    # Input and output tensors are being accessed identically and elem size is the same.
                    # We can simply propagate the device_layout.
                    stl = SpyreTensorLayout(
                        x_stl.device_size,
                        x_stl.stride_map,
                        get_device_dtype(output.dtype),
                    )
                else:
                    # TODO: We should be able to preserve the input stride_map
                    #       unless the operation is changing elems_per_stick.
                    #       For now, use the default layout for a mostly row major dimension
                    #       ordering, adjusted to put the stick dimension last and move all
                    #       non-stick size one dimensions to the right to avoid tiling them.
                    in_device_coords = device_coordinates(x.layout, x.dep)
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
                    stl = SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)

                # FixedTiledLayout keeps original (possibly symbolic) size/stride.
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )

    elif aten_op == spyreop.layernormnorm.default:
        # Output layout is determined by layout of first argument only
        x = args[0]
        x_stl = x.layout.device_layout
        if x.layout.size != output.size or x.layout.stride != output.stride:
            raise Unsupported(
                f"views not supported for spyre.layernormnorm({x.layout.size})=>{output.size}) "
            )
        stl = SpyreTensorLayout(x_stl.device_size, x_stl.stride_map, x_stl.device_dtype)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    else:
        in_coords = [host_coordinates(arg.layout, arg.dep) for arg in args]
        in_device_coords = [device_coordinates(arg.layout, arg.dep) for arg in args]
        out_coords = host_coordinates(output, output_dep)

        # Stick compatability check.
        # For all tensors whose stick dimension is being iterated over,
        # the indexing expression must be identical.
        stick_exprs = set()
        for idc in in_device_coords:
            if idc[-1] != 0:
                stick_exprs.add(idc[-1])
        stick_expr = next(iter(stick_exprs)) if stick_exprs else None

        if len(stick_exprs) > 1:
            # This is a legal PyTorch operation that requires inserting restickify operations.
            logger.warning(
                f"Injecting restickify to resolve pointwise op with nonuniform stick indexing: {stick_exprs}."
            )
            # Arbitrary choice: let arg[0] define the stick variable and restickify all
            # others that have a conflict.
            stick_expr = in_device_coords[0][-1]
            assert stick_expr != 0, (
                "Expected arg 0 to have non-zero stick indexing expression"
            )
            for ic, idc, arg in zip(in_coords[1:], in_device_coords[1:], args[1:]):
                if idc[-1] != stick_expr:
                    schedule_restickify(op, arg, stick_expr, ic, idc, restickify_plan)

        # If the indexing and device element size are identical
        # across all inputs and the output we can just propagate the device layout.
        can_use_same_layout = True
        for arg, arg_coors in zip(args, in_coords):
            if (
                arg_coors != out_coords
                or arg.dep.index != output_dep.index
                or not same_device_size(arg.layout.dtype, output.dtype)
            ):
                can_use_same_layout = False
                break

        if can_use_same_layout:
            template_stl = args[0].layout.device_layout
            stl = SpyreTensorLayout(
                template_stl.device_size,
                template_stl.stride_map,
                get_device_dtype(output.dtype),
            )
        else:
            # Use row major adjusted to put stick dimension last
            # and move all non-stick size one dimensions to the right to avoid tiling them.
            if len(stick_exprs) == 0:
                maybe_stick_dim = None
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
            # Concretize for C++ SpyreTensorLayout constructor.
            c_size = [concretize_expr(s) for s in output.size]
            c_stride = [concretize_expr(s) for s in output.stride]
            stl = SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)

        # FixedTiledLayout keeps original (possibly symbolic) size/stride.
        result = FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )

        if logger.isEnabledFor(logging.DEBUG):
            input_info = ", ".join(
                [f"in{i}:{list(arg.layout.size)}" for i, arg in enumerate(args)]
            )
            logger.debug(
                f"{getattr(op, '__name__', repr(op))} layout: {input_info} -> out:{list(result.size)}, "
                f"device_size={list(result.device_layout.device_size)}"
            )

        return result


def reduction_layout(
    op: Operation,
    output: FixedLayout,
    output_dep: MemoryDep,
    args: list[SchedNodeArg],
    restickify_plan: dict[str, list[dict[str, Any]]],
) -> FixedTiledLayout:
    data = op.data
    if data.reduction_type == BATCH_MATMUL_OP:
        x = args[0]
        y = args[1]
        x_coords = host_coordinates(x.layout, x.dep)
        x_dev_coords = device_coordinates(x.layout, x.dep)
        y_coords = host_coordinates(y.layout, y.dep)
        y_dev_coords = device_coordinates(y.layout, y.dep)
        out_coords = host_coordinates(output, output_dep)
        x_stick_expr = x_dev_coords[-1]
        y_stick_expr = y_dev_coords[-1]
        x_stick_dim = matching_dim(x_coords, x_stick_expr)
        y_stick_dim = matching_dim(y_coords, y_stick_expr)
        if x_stick_dim is None or y_stick_dim is None:
            raise Unsupported(
                f"{data.reduction_type}: failed to map stick_dims to host coords"
            )

        # Hardware stick constraints (DF16):
        #   Input1 (x): stick on reduction_dim (the x coord that does NOT appear in output)
        #   Input2 (y): stick on generated_dim (the y coord that appears in output)
        #   Output:     stick on generated_dim
        if matching_dim(out_coords, x_stick_expr) is not None:
            logger.warning(
                f"Injecting restickify on {data.reduction_type} x input to move stick to reduction_dim"
            )
            reduction_coord = next(
                c
                for c in x_coords
                if len(c.free_symbols) > 0 and matching_dim(out_coords, c) is None
            )
            tl = schedule_restickify(
                op,
                x,
                reduction_coord,
                x_coords,
                x_dev_coords,
                restickify_plan,
            )
            x_stick_expr = device_coordinates(tl, x.dep)[-1]
        if matching_dim(out_coords, y_stick_expr) is None:
            logger.warning(
                f"Injecting restickify on {data.reduction_type} y input to move stick to generated_dim"
            )
            generated_coord = next(
                c
                for c in y_coords
                if len(c.free_symbols) > 0
                and matching_dim(out_coords, c) is not None
                and matching_dim(x_coords, c) is None
            )
            tl = schedule_restickify(
                op,
                y,
                generated_coord,
                y_coords,
                y_dev_coords,
                restickify_plan,
            )
            y_stick_expr = device_coordinates(tl, y.dep)[-1]
        out_stick_dim = matching_dim(out_coords, y_stick_expr)
        if out_stick_dim is None:
            raise Unsupported(
                f"{data.reduction_type}: failed to map output stick_dim to host coords {out_coords} {y_stick_expr}"
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
        stl = SpyreTensorLayout(c_size, c_stride, output.dtype, out_dim_order)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    elif data.reduction_type == "exx2":
        x = args[0]
        x_coords = host_coordinates(x.layout, x.dep)
        x_dev_coords = device_coordinates(x.layout, x.dep)
        x_stick_expr = x_dev_coords[-1]
        x_stick_dim = matching_dim(x_coords, x_stick_expr)
        if x_stick_dim is None or x_stick_dim != len(x.layout.size) - 1:
            # TODO: Insert a restickify to enable the operation to be performed
            raise Unsupported(f"exx2: illegal device layout {x.layout}")

        dim_order = list(range(len(output.size))) + [-1]
        # Concretize for C++ SpyreTensorLayout constructor.
        c_size = [concretize_expr(s) for s in output.size]
        c_stride = [concretize_expr(s) for s in output.stride]
        stl = SpyreTensorLayout(c_size, c_stride, output.dtype, dim_order)

        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    else:
        x = args[0]
        x_coords = host_coordinates(x.layout, x.dep)
        x_dev_coords = device_coordinates(x.layout, x.dep)
        out_coords = host_coordinates(output, output_dep)
        x_stick_expr = x_dev_coords[-1]
        out_stick_dim = matching_dim(out_coords, x_stick_expr)
        if out_stick_dim is None:
            out_dim_order = list(range(len(output.size))) + [-1]
        else:
            out_dim_order = [
                d for d in list(range(len(output.size))) if d != out_stick_dim
            ]
            out_dim_order = out_dim_order + [out_stick_dim]
        # Concretize for C++ SpyreTensorLayout constructor.
        c_size = [concretize_expr(s) for s in output.size]
        c_stride = [concretize_expr(s) for s in output.stride]
        stl = SpyreTensorLayout(c_size, c_stride, output.dtype, out_dim_order)
        result = FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{data.reduction_type} layout: in:{list(args[0].layout.size)} -> out:{list(result.size)}, "
                f"device_size={list(result.device_layout.device_size)}"
            )

        return result


def generic_layout(op: Operation) -> FixedTiledLayout:
    output: FixedLayout = op.get_layout()
    # Concretize for C++ SpyreTensorLayout constructor.
    c_size = [concretize_expr(s) for s in output.size]
    # Use the generic stick format
    stl = SpyreTensorLayout(c_size, output.dtype)
    return FixedTiledLayout(
        output.device, output.dtype, output.size, output.stride, stl
    )


def propagate_spyre_tensor_layouts(
    operations: list[Operation],
) -> None:
    # Convert InputBuffers from FixedLayout to FixedTiledLayouts
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
                        "graph input {name} is not a TensorBox(StorageBox(InputBuffer))"
                    )
                ptl = tb.data.data.layout
                if not isinstance(ptl, FixedLayout):
                    raise Unsupported("graph input {name} does not have a FixedLayout")
                tb.data.data.layout = FixedTiledLayout(
                    ptl.device, ptl.dtype, ptl.size, ptl.stride, stl
                )

    # Operations are in topological order (guaranteed by GraphLowering).
    # Visit them and use the inputs' FixedTiledLayouts and the operation being
    # performed to convert each output FixedLayout to a FixedTiledLayout.
    restickify_plan: dict[str, list[dict[str, Any]]] = {}

    it = iter(operations)
    for op in it:
        if op.is_no_op():
            op.layout = generic_layout(op)
        elif isinstance(op, ComputedBuffer):
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                # Mutation ops (e.g. spyre.overwrite) must keep their
                # MutationLayoutSHOULDREMOVE so the scheduler correctly
                # treats them as in-place writes to the target buffer.
                # Their FixedTiledLayout is assigned later in
                # propagate_mutation_layouts, after the scheduler has
                # set up mutation tracking.
                continue
            op.decide_layout()
            rw = op.get_read_writes()
            output_dep = next(iter(rw.writes))
            args = get_mem_deps_from_rw(rw)
            output = op.get_layout()
            if isinstance(op.data, Pointwise):
                op.layout = pointwise_layout(
                    op, output, output_dep, args, restickify_plan
                )
            elif isinstance(op.data, Reduction):
                op.layout = reduction_layout(
                    op, output, output_dep, args, restickify_plan
                )
            else:
                logger.warning(f"Warning: unhandled node type {type(op.data)}")
        elif isinstance(op, FallbackKernel):
            op = next(it, None)
            if not isinstance(op, MultiOutput):
                raise RuntimeError("FallbackKernel must be followed by MultiOutput")
            op.layout = generic_layout(op)
        elif isinstance(op, ExternKernel):
            if isinstance(op, SpyreConstantFallback):
                op.layout = generic_layout(op)
            else:
                logger.warning(f"unhandled node type {type(op)}")
        else:
            logger.warning(f"unhandled operation type {type(op)}")

    V.graph.restickify_plan = restickify_plan


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
    from .pass_utils import get_mem_deps

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
                args = get_mem_deps(n)
                output = n.node.get_layout()
                n.node.layout = pointwise_layout(n.node, output, output_dep, args, {})
        else:
            logger.warning(
                f"propagate_mutation_layouts: unhandled mutation op {type(n.node.data)}"
            )

    return nodes
