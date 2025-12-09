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

from typing import NamedTuple, Sequence

import sympy

import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.scheduler import BaseSchedulerNode, SchedulerNode
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import V

from torch_spyre._C import SpyreTensorLayout, StickFormat
from . import Unsupported
from .constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP
from .ir import FixedTiledLayout


aten = torch.ops.aten
spyreop = torch.ops.spyre


def stl_host_dim_order(self: SpyreTensorLayout) -> list[int]:
    return self.dim_map[1:]


def stl_stick_dim(self: SpyreTensorLayout) -> int:
    return self.dim_map[-1]


def stl_is_stick_reduction(self: SpyreTensorLayout, axis: list[int]) -> bool:
    stick_dim = self.stick_dim()
    if stick_dim in axis:
        if len(axis) > 1:
            Unsupported(f"reduction on both stick and non-stick dimensions {axis}")
        return True
    else:
        return False


setattr(SpyreTensorLayout, "host_dim_order", stl_host_dim_order)
setattr(SpyreTensorLayout, "stick_dim", stl_stick_dim)
setattr(SpyreTensorLayout, "is_stick_reduction", stl_is_stick_reduction)


def stride_order_vars(index: sympy.Expr) -> Sequence[sympy.Symbol]:
    """
    Order the free variables in an index expression in decreasing stride order.
    """
    strides = {
        s: sympy_subs(index, {s: 1}) - sympy_subs(index, {s: 0})
        for s in index.free_symbols
    }
    ordered_strides: Sequence[tuple[sympy.Symbol, sympy.Expr]] = sorted(
        strides.items(), key=lambda item: item[1], reverse=True
    )
    return [item[0] for item in ordered_strides]


class Arg(NamedTuple):
    dep: MemoryDep
    layout: FixedTiledLayout


def pointwise_layout(n: SchedulerNode, args: list[Arg]) -> FixedTiledLayout:
    pw: Pointwise = n.node.data
    output: FixedLayout = n.node.get_layout()
    op = pw.get_origin_node().target
    if len(args) == 1:
        x = args[0]
        match op:
            case spyreop.layernormnorm.default:
                raise Unsupported("TODO: layernormnorm")
            case spyreop.layernormscale.default:
                raise Unsupported("TODO: layernormscale")
            case spyreop.slice.default:
                if x.layout.device_layout.format != StickFormat.Sparse:
                    raise Unsupported("slice on non-sparse tensor")
                if len(x.layout.size) != 1:
                    raise Unsupported("slice on non 1-D tensor")
                stl = SpyreTensorLayout(output.size, output.dtype)
                stl.format = StickFormat.Dense
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )
            case spyreop.swap.default:
                if x.layout.device_layout.format != StickFormat.Sparse:
                    raise Unsupported("swap on non-sparse tensor")
                if len(x.layout.size) != 1:
                    raise Unsupported("swap on non 1-D tensor")
                stl = SpyreTensorLayout(output.size, output.dtype)
                stl.format = StickFormat.Sparse
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )
            case aten.clone.default:
                if not x.layout.device_layout.format == StickFormat.Dense:
                    raise Unsupported("clone on sparse tensor")
                stl = SpyreTensorLayout(output.size, output.dtype)
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )
            case _:
                # Generic pointwise unary: output layout is same as input
                if not x.layout.size == output.size:
                    raise Unsupported(
                        f"size mismatch:  {op}({x.layout.size})=>{output.size}) "
                    )
                stl = SpyreTensorLayout(
                    output.size,
                    output.dtype,
                    x.layout.device_layout.host_dim_order(),
                    x.layout.device_layout.format,
                )
                return FixedTiledLayout(
                    output.device, output.dtype, output.size, output.stride, stl
                )
    else:
        output_dims = stride_order_vars(list(n.read_writes.writes)[0].index)
        input_dims = [stride_order_vars(arg.dep.index) for arg in args]
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
        output_format = None
        stick_dim_var = output_dims[-1]
        for i in range(len(args)):
            if stick_dim_var in input_dims[i]:
                if output_format is None:
                    output_format = args[i].layout.device_layout.format
                elif output_format != args[i].layout.device_layout.format:
                    raise Unsupported(
                        "pointwise op with incompatible input stick formats"
                    )
        # TODO: Pretending bools are float16.
        out_dtype = torch.float16 if output.dtype == torch.bool else output.dtype
        stl = SpyreTensorLayout(output.size, out_dtype)
        stl.format = output_format
        return FixedTiledLayout(
            output.device, out_dtype, output.size, output.stride, stl
        )


def reduction_layout(n: SchedulerNode, args: list[Arg]) -> FixedTiledLayout:
    red: Reduction = n.node.data
    output: FixedLayout = n.node.get_layout()
    output_dims = stride_order_vars(list(n.read_writes.writes)[0].index)
    if (
        red.reduction_type == MATMUL_REDUCTION_OP
        or red.reduction_type == BATCH_MATMUL_OP
    ):
        x_stl = args[0].layout.device_layout
        y_stl = args[1].layout.device_layout
        if x_stl.format != StickFormat.Dense or y_stl.format != StickFormat.Dense:
            raise Unsupported(
                f"{red.reduction_type} on non-dense tensors {x_stl} {y_stl}"
            )
        if x_stl.host_dim_order() != y_stl.host_dim_order():
            raise Unsupported(
                f"{red.reduction_type} stick dimensions mismatch {x_stl} {y_stl}"
            )
        stl = SpyreTensorLayout(output.size, output.dtype, x_stl.host_dim_order())
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    elif red.reduction_type == "exx2":
        stl = SpyreTensorLayout(output.size, output.dtype)
        stl.format = StickFormat.SparseMulti
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )
    else:
        input = args[0]
        input_dims = stride_order_vars(input.dep.index)
        stick_var = input_dims[-1]
        is_stick_reduction = stick_var not in output_dims
        stl = SpyreTensorLayout(output.size, output.dtype)
        stl.format = StickFormat.Sparse if is_stick_reduction else StickFormat.Dense
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )


def propagate_spyre_tensor_layouts(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    def mem_deps(n: SchedulerNode) -> list[Arg]:
        res: list[Arg] = []
        for arg in n.read_writes.reads:
            if isinstance(arg, MemoryDep):
                buf = V.graph.get_buffer(arg.name)
                layout = buf.get_layout()
                if not isinstance(layout, FixedTiledLayout):
                    raise RuntimeError(f"{buf} does not have FixedTiledLayout")
                res.append(Arg(arg, layout))
        return res

    # Convert InputBuffers from FixedLayout to FixedTiledLayouts
    for name, real_input in zip(V.graph.graph_input_names, V.get_real_inputs()):
        if isinstance(real_input, torch.Tensor):
            stl = real_input.device_tensor_layout()
            if stl is None:
                # All spyre tensors are created with device layouts.
                # Therefore we expect all graph inputes to have them.
                raise Unsupported(f"missing device_tensor_layout on graph input {name}")
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
    for n in nodes:
        if isinstance(n, SchedulerNode) and isinstance(n.node, ComputedBuffer):
            n.node.decide_layout()
            if isinstance(n.node.data, Pointwise):
                output_layout = pointwise_layout(n, mem_deps(n))
                n.node.layout = output_layout
            elif isinstance(n.node.data, Reduction):
                output_layout = reduction_layout(n, mem_deps(n))
                n.node.layout = output_layout
            else:
                print(f"Warning: unhandled node type {type(n.node)}")

        else:
            print(f"Warning: unhandled scheduler node type {type(n)}")

    return nodes
