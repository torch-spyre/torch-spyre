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

import torch
from torch._inductor.ir import (
    ComputedBuffer,
    FallbackKernel,
    FixedLayout,
    InputBuffer,
    MultiOutput,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.scheduler import (
    BaseSchedulerNode,
    SchedulerNode,
    ExternKernelSchedulerNode,
    NopKernelSchedulerNode,
)
from torch._inductor.virtualized import V

from torch_spyre._C import (
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
    compute_view_layout,
)
from .errors import Unsupported
from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .pass_utils import SchedNodeArg, get_mem_deps, map_dims_to_vars, is_wildcard


aten = torch.ops.aten
spyreop = torch.ops.spyre


def is_sparse(stl: SpyreTensorLayout) -> bool:
    return stl.dim_map[-1] == -1


def device_layout_like(
    layout: FixedTiledLayout, dtype: torch.dtype
) -> SpyreTensorLayout:
    """
    Return a SpyreTensorLayout with the same tiling pattern as layout adjusted for the device_size of dtype.
    """
    if get_elem_in_stick(layout.dtype) == get_elem_in_stick(dtype):
        return SpyreTensorLayout(
            layout.device_layout.device_size,
            layout.device_layout.dim_map,
            get_device_dtype(dtype),
        )
    else:
        adjusted_device_size = list(layout.device_layout.device_size)
        stick_dim_idx = -3 if len(adjusted_device_size) > 2 else -2
        old = get_elem_in_stick(layout.dtype)
        new = get_elem_in_stick(dtype)
        if old > new:
            scaling_factor = old / new
            adjusted_device_size[-1] = int(adjusted_device_size[-1] * scaling_factor)
            adjusted_device_size[stick_dim_idx] = int(
                (adjusted_device_size[stick_dim_idx] + scaling_factor - 1)
                / scaling_factor
            )
        else:
            scaling_factor = new / old
            adjusted_device_size[-1] = int(adjusted_device_size[-1] / scaling_factor)
            adjusted_device_size[stick_dim_idx] = int(
                adjusted_device_size[stick_dim_idx] * scaling_factor
            )
        return SpyreTensorLayout(
            adjusted_device_size, layout.device_layout.dim_map, get_device_dtype(dtype)
        )


def derive_dim_order(template: SpyreTensorLayout, rank: int) -> list[int]:
    # 1 D template; nothing to learn
    if len(template.dim_map) == 2:
        return list(range(rank))

    # Recognize default tiling
    if (template.dim_map[-1] == template.dim_map[-3]) and (
        len(template.dim_map) == len(set(template.dim_map)) + 1
    ):
        # Invert tiling to construct matching dim_order
        dim_order = template.dim_map[-2:-1] + template.dim_map[:-2]
        dim_order = [d for d in dim_order if d < rank and d >= 0]
        if len(dim_order) == rank:
            return dim_order
        # Add any missing ranks to the front
        for r in range(rank):
            if r not in dim_order:
                dim_order = [r] + dim_order
        return dim_order

    print(f"Warning: could not derive dim order from {template}")
    return list(range(rank))


def pointwise_layout(n: SchedulerNode, args: list[SchedNodeArg]) -> FixedTiledLayout:
    pw: Pointwise = n.node.data
    output: FixedLayout = n.node.get_layout()
    origin_node = next(iter(pw.origins))
    op = origin_node.target
    if len(args) == 1:
        x = args[0]
        x_stl = x.layout.device_layout
        match op:
            case spyreop.slice.default:
                if not is_sparse(x_stl):
                    raise Unsupported("slice on non-sparse tensor")
                if len(x.layout.size) != 1:
                    raise Unsupported("slice on non 1-D tensor")
                stl = SpyreTensorLayout(output.size, output.dtype)

            case spyreop.swap.default:
                if not is_sparse(x_stl):
                    raise Unsupported("swap on non-sparse tensor")
                if len(x.layout.size) != 1:
                    raise Unsupported("swap on non 1-D tensor")
                stl = SpyreTensorLayout(output.size, output.dtype, [0, -1])

            case aten.clone.default:
                if is_sparse(x_stl):
                    # TODO: Determine whether we already support cloning a sparse tensor
                    #       or what functionality needs to be added to enable it.  Restickify?
                    raise Unsupported("clone on sparse tensor")

                # Clone is generated by an explicit `contiguous()`; force row major layout via dim_order.
                stl = SpyreTensorLayout(
                    output.size, output.dtype, list(range(len(output.size)))
                )

            case _:
                in_size = x.layout.size
                out_size = output.size

                if in_size == out_size:
                    # Sizes match exactly; propagate the input's SpyreTensorLayout
                    stl = device_layout_like(x.layout, output.dtype)
                elif [s for s in in_size if s != 1] == [s for s in out_size if s != 1]:
                    # Squeezed sizes match; use view machinery to compute output layout
                    try:
                        stl = compute_view_layout(
                            torch.Size(in_size), torch.Size(out_size), x_stl
                        )
                    except RuntimeError:
                        # TODO: This is a legal PyTorch operation.
                        # However, it may require us to inject a restickify to perform it.
                        raise Unsupported(
                            f"incompatible sizes: {op}({in_size})=>{out_size}) "
                        )
                else:
                    # This should have been rejected by PyTorch. This is not a legal operation.
                    raise Unsupported(f"size mismatch: {op}({in_size})=>{out_size}) ")

        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    elif op == spyreop.layernormnorm.default:
        # Output layout is determined by layout of first argument only
        x = args[0]
        x_stl = x.layout.device_layout
        if not x.layout.size == output.size:
            raise Unsupported(
                f"size mismatch:  layernormnorm({x.layout.size})=>{output.size}) "
            )
        stl = SpyreTensorLayout(x_stl.device_size, x_stl.dim_map, x_stl.device_dtype)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    else:
        # Stick compatability check.
        # All stick dimensions not being broadcast must correspond to the same variable in the iteration space.
        input_dim_to_vars = [
            map_dims_to_vars(arg.layout, arg.dep.index) for arg in args
        ]
        stick_vars = set()
        for arg, arg_dim_map in zip(args, input_dim_to_vars):
            stick_dim = arg.layout.device_layout.host_stick_dim()
            if stick_dim is not None and stick_dim in arg_dim_map:
                sv = arg_dim_map[stick_dim]
                if not is_wildcard(sv):
                    stick_vars.add(arg_dim_map[stick_dim])
        if len(stick_vars) > 1:
            # TODO: This is a legal PyTorch operation that we cannot execute without inserting restickify operations.
            raise Unsupported("Pointwise op with multiple non-broadcasted stick dims")

        # Case 1: There exists a non-broadcasting input.
        # Propagate its device_layout to the output.
        for arg in args:
            if arg.layout.size == output.size:
                stl = device_layout_like(arg.layout, output.dtype)
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )

        # Case 2: All inputs are broadcasting at least one dimension.
        #         If we haven't already rejected the operation as needing
        #         restickification, it must be the case that output is a dense tensor.
        #         Hueristically pick the tensor with most non-broadcast dimensions
        #         and derive a dim_order to use for the output from it.
        chosen = args[0]
        used_dims = len(input_dim_to_vars[0])
        for arg, arg_dim_map in zip(args, input_dim_to_vars):
            if len(arg_dim_map) > used_dims:
                chosen = arg
                used_dims = len(arg_dim_map)

        dim_order = derive_dim_order(chosen.layout.device_layout, len(output.size))
        stl = SpyreTensorLayout(output.size, output.dtype, dim_order)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )


def reduction_layout(n: SchedulerNode, args: list[SchedNodeArg]) -> FixedTiledLayout:
    red: Reduction = n.node.data
    output: FixedLayout = n.node.get_layout()
    output_dims = map_dims_to_vars(output, list(n.read_writes.writes)[0].index)
    if red.reduction_type == MATMUL_REDUCTION_OP:
        x_stl = args[0].layout.device_layout
        y_stl = args[1].layout.device_layout
        if is_sparse(x_stl) or is_sparse(y_stl):
            raise Unsupported(f"matmul on sparse tensors {x_stl} {y_stl}")
        if x_stl.host_stick_dim() == 0 and y_stl.host_stick_dim() == 0:
            out_dim_order = [1, 0]
        elif x_stl.host_stick_dim() != 0 and y_stl.host_stick_dim() != 0:
            out_dim_order = [0, 1]
        else:
            raise Unsupported(f"matmul stick dimensions mismatch {x_stl} {y_stl}")
        stl = SpyreTensorLayout(output.size, output.dtype, out_dim_order)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    elif red.reduction_type == BATCH_MATMUL_OP:
        x_layout = args[0].layout
        y_layout = args[1].layout
        x_stl = x_layout.device_layout
        y_stl = y_layout.device_layout
        x_dims = len(x_layout.size)
        y_dims = len(y_layout.size)
        out_dims = len(output.size)
        if is_sparse(x_stl) or is_sparse(y_stl):
            raise Unsupported(f"bmm on sparse tensors {x_stl} {y_stl}")
        out_dim_order = list(range(out_dims - 2))
        if (x_stl.host_stick_dim() == (x_dims - 1)) and (
            y_stl.host_stick_dim() == (y_dims - 1)
        ):
            out_dim_order = out_dim_order + [out_dims - 2, out_dims - 1]
        elif (x_stl.host_stick_dim() == (x_dims - 1)) and (
            y_stl.host_stick_dim() == (y_dims - 1)
        ):
            out_dim_order = out_dim_order + [out_dims - 1, out_dims - 2]
        else:
            raise Unsupported(f"bmm stick dimensions mismatch {x_stl} {y_stl}")
        stl = SpyreTensorLayout(output.size, output.dtype, out_dim_order)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    elif red.reduction_type == "exx2":
        x = args[0]
        x_stl = x.layout.device_layout
        if is_sparse(x_stl) or x_stl.host_stick_dim() != (len(x.layout.size) - 1):
            raise Unsupported(f"exx2 unsupported layout {x_stl}")
        dim_map = list(range(len(output.size))) + [-1]
        stl = SpyreTensorLayout(output.size, output.dtype, dim_map)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    else:
        x = args[0]
        x_stl = x.layout.device_layout
        out_dim_order = derive_dim_order(x_stl, len(output.size))
        in_dims = map_dims_to_vars(x.layout, x.dep.index)
        stick_dim_var = in_dims.get(x_stl.host_stick_dim(), None)
        if stick_dim_var not in output_dims.values():
            out_dim_order = out_dim_order + [-1]
        stl = SpyreTensorLayout(output.size, output.dtype, out_dim_order)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )


def generic_layout(n: ExternKernelSchedulerNode) -> FixedTiledLayout:
    output: FixedLayout = n.node.get_layout()
    # Use the generic stick format
    stl = SpyreTensorLayout(output.size, output.dtype)
    return FixedTiledLayout(
        output.device, output.dtype, output.size, output.stride, stl
    )


def propagate_spyre_tensor_layouts(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
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

    # Nodes are in topological order (guarenteed by caller).
    # Visit them and use the inputs' FixedTiledLayouts and the operation being
    # performed by the node to convert its output FixedLayouts to FixedTiledLayouts.

    it = iter(nodes)
    for n in it:
        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            n.node.decide_layout()
            if isinstance(n.node.data, Pointwise):
                output_layout = pointwise_layout(n, get_mem_deps(n))
                n.node.layout = output_layout
            elif isinstance(n.node.data, Reduction):
                output_layout = reduction_layout(n, get_mem_deps(n))
                n.node.layout = output_layout
            else:
                print(f"Warning: unhandled node type {type(n.node)}")
        elif isinstance(n, ExternKernelSchedulerNode):
            if isinstance(n.node, FallbackKernel):
                n = next(it, None)
                if not (
                    isinstance(n, ExternKernelSchedulerNode)
                    and isinstance(n.node, MultiOutput)
                ):
                    raise RuntimeError("FallbackKernel must be followed by MultiOutput")

                output_layout = generic_layout(n)
                n.node.layout = output_layout
            else:
                print(f"Warning: unhandled node type {type(n.node)}")
        elif isinstance(n, NopKernelSchedulerNode):
            output_layout = generic_layout(n)
            n.node.layout = output_layout
        else:
            print(f"Warning: unhandled scheduler node type {type(n)}")

    return nodes
