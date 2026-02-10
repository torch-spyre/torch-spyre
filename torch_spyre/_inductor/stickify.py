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

from torch_spyre._C import SpyreTensorLayout
from . import Unsupported
from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout
from .pass_utils import SchedNodeArg, get_mem_deps, map_dims_to_vars


aten = torch.ops.aten
spyreop = torch.ops.spyre


def is_sparse(stl: SpyreTensorLayout) -> bool:
    return stl.device_size[-1] == -1


def pointwise_layout(n: SchedulerNode, args: list[SchedNodeArg]) -> FixedTiledLayout:
    pw: Pointwise = n.node.data
    output: FixedLayout = n.node.get_layout()
    op = pw.get_origin_node().target
    if len(args) == 1:
        x = args[0]
        x_stl = x.layout.device_layout
        match op:
            case spyreop.layernormscale.default:
                if not x.layout.size == output.size:
                    raise Unsupported(
                        f"size mismatch:  layernormscale({x.layout.size})=>{output.size}) "
                    )
                stl = SpyreTensorLayout(
                    x_stl.device_size, x_stl.dim_map, x_stl.device_dtype
                )
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )
            case spyreop.slice.default:
                if not is_sparse(x_stl):
                    raise Unsupported("slice on non-sparse tensor")
                if len(x.layout.size) != 1:
                    raise Unsupported("slice on non 1-D tensor")
                stl = SpyreTensorLayout(output.size, output.dtype)
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )
            case spyreop.swap.default:
                if not is_sparse(x_stl):
                    raise Unsupported("swap on non-sparse tensor")
                if len(x.layout.size) != 1:
                    raise Unsupported("swap on non 1-D tensor")
                stl = SpyreTensorLayout(output.size, output.dtype, [0, -1])
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )
            case aten.clone.default:
                if is_sparse(x_stl):
                    raise Unsupported("clone on sparse tensor")
                # FIXME: Blindly using dense generic stick layout. Should derive from inputs
                stl = SpyreTensorLayout(output.size, output.dtype)
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )
            case _:
                # Generic pointwise unary: output dim order is same as input
                if not x.layout.size == output.size:
                    raise Unsupported(
                        f"size mismatch:  {op}({x.layout.size})=>{output.size}) "
                    )
                # FIXME: Blindly using dense generic stick layout. Should derive from inputs
                stl = SpyreTensorLayout(output.size, output.dtype)
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )
    elif op == spyreop.layernormnorm.default:
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
        output_dims = map_dims_to_vars(output, list(n.read_writes.writes)[0].index)
        input_dims = [map_dims_to_vars(arg.layout, arg.dep.index) for arg in args]
        input_dim_idx = [0] * len(args)
        for i in range(len(output_dims)):
            var = output_dims[i]
            for j in range(len(args)):
                if var in input_dims[j]:
                    if input_dims[j][input_dim_idx[j]] != var:
                        # TODO: This is overly conservative.
                        #        SDSCs can support pointwise ops where non-stick dimensions differ in stride order
                        raise Unsupported(
                            "pointwise op with non-aligned input dimensions"
                        )
                    input_dim_idx[j] += 1

        # FIXME: Blindly using dense generic stick layout. Should derive from inputs
        stl = SpyreTensorLayout(output.size, output.dtype)
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
            raise Unsupported(f"matmul on non-dense tensors {x_stl} {y_stl}")
        if x_stl.host_stick_dim() == 0 and y_stl.host_stick_dim() == 0:
            out_host_dim_order = [1, 0]
        elif x_stl.host_stick_dim() != 0 and y_stl.host_stick_dim() != 0:
            out_host_dim_order = [0, 1]
        else:
            raise Unsupported(f"matmul stick dimensions mismatch {x_stl} {y_stl}")
        stl = SpyreTensorLayout(output.size, output.dtype, out_host_dim_order)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    elif red.reduction_type == BATCH_MATMUL_OP:
        x_stl = args[0].layout.device_layout
        y_stl = args[1].layout.device_layout
        if is_sparse(x_stl) or is_sparse(y_stl):
            raise Unsupported(
                f"{red.reduction_type} on non-dense tensors {x_stl} {y_stl}"
            )
        if x_stl.dim_map != y_stl.dim_map:
            raise Unsupported(f"{red.reduction_type} layout mismatch {x_stl} {y_stl}")
        # TODO: FIXME forcing generic stick layout. Should compute the output device_size and dim_map directly from input STL
        stl = SpyreTensorLayout(output.size, output.dtype)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    elif red.reduction_type == "exx2":
        # TODO: FIXME forcing generic stick layout.  Should compute the output device_size and dim_map directly from input STL
        dim_map = list(range(len(output.size))) + [-1]
        stl = SpyreTensorLayout(output.size, output.dtype, dim_map)
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    else:
        input = args[0]
        stick_dim = input.layout.device_layout.host_stick_dim()
        input_dims = map_dims_to_vars(input.layout, input.dep.index)
        stick_var = input_dims[stick_dim]
        is_stick_reduction = stick_var not in output_dims.values()
        sparse_tensor = is_stick_reduction
        # TODO: FIXME forcing generic stick layout.  Should compute the lowlevel device_size and dim_map directly from input STL
        dim_map = list(range(len(output.size))) + ([-1] if sparse_tensor else [])
        stl = SpyreTensorLayout(output.size, output.dtype, dim_map)
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
