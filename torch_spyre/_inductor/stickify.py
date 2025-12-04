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

from typing import NamedTuple, Sequence, Tuple, Union

import sympy
import torch
from sympy import Expr
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
from torch.fx.experimental.symbolic_shapes import (
    guard_size_oblivious,
    is_nested_int,
)
from torch_spyre._C import SpyreTensorLayout, StickFormat
from . import Unsupported

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


def stl_spyre_fixed_layout(
    self: SpyreTensorLayout, device: torch.device, size: torch.Size, dtype: torch.dtype
):
    cur_stride = (
        1
        if self.format == StickFormat.Dense
        else 128 // dtype.itemsize  # TODO - get from self.device_strides?
    )
    stride: list[int | torch.SymInt] = [-1] * len(size)
    for d in reversed(self.host_dim_order()):
        stride[d] = cur_stride
        cur_stride = cur_stride * size[d]
    return FixedTiledLayout(device, dtype, list(size), stride, self)


setattr(SpyreTensorLayout, "host_dim_order", stl_host_dim_order)
setattr(SpyreTensorLayout, "stick_dim", stl_stick_dim)
setattr(SpyreTensorLayout, "is_stick_reduction", stl_is_stick_reduction)
setattr(SpyreTensorLayout, "spyre_fixed_layout", stl_spyre_fixed_layout)


class FixedTiledLayout(FixedLayout):
    device_layout: SpyreTensorLayout

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: list[Expr],
        stride: list[Expr],
        device_layout: SpyreTensorLayout,
    ) -> None:
        super().__init__(device, dtype, size, stride)
        self.device_layout = device_layout

    def __str__(self) -> str:
        device_index_str = "" if self.device.index is None else f":{self.device.index}"
        return (
            f"{type(self).__name__}('{self.device.type}{device_index_str}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}, device_layout={self.device_layout})"
        )

    def get_allocation_size(self) -> list[Expr]:
        # TODO: Eventually this will include padding, etc.
        return self.size

    __repr__ = __str__


def tensor_get_spyre_layout(self: torch.Tensor) -> SpyreTensorLayout:
    if not hasattr(self, "spyre_layout"):
        print(f"Warning: {self} lacks spyre_layout; assuming generic stick layout")
        self.spyre_layout = SpyreTensorLayout(self.size(), self.dtype)
    return self.spyre_layout


def spyre_matmul_result_shape(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[Sequence[int], SpyreTensorLayout]:
    x_layout: SpyreTensorLayout = x.get_spyre_layout()
    y_layout: SpyreTensorLayout = y.get_spyre_layout()
    if x_layout.format != StickFormat.Dense or y_layout.format != StickFormat.Dense:
        raise Unsupported(f"matmul on non-dense tensors {x_layout} {y_layout}")
    if x_layout.host_dim_order() != y_layout.host_dim_order():
        raise Unsupported(f"matmul stick dimensions mismatch {x_layout} {y_layout}")
    res_size = [x.size()[0], y.size()[1]]
    res_layout = SpyreTensorLayout(res_size, x.dtype, x_layout.host_dim_order())
    return res_size, res_layout


def spyre_bmm_result_shape(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[Sequence[int], SpyreTensorLayout]:
    x_layout: SpyreTensorLayout = x.get_spyre_layout()
    y_layout: SpyreTensorLayout = y.get_spyre_layout()
    if x_layout.format != StickFormat.Dense or y_layout.format != StickFormat.Dense:
        raise Unsupported(f"bmm on non-dense tensors {x_layout} {y_layout}")
    if x_layout.host_dim_order() != y_layout.host_dim_order():
        raise Unsupported(f"bmm stick dimensions mismatch {x_layout} {y_layout}")
    res_size = [x.size()[0], x.size()[1], y.size()[-1]]
    res_layout = SpyreTensorLayout(res_size, x.dtype, x_layout.host_dim_order())
    return res_size, res_layout


def spyre_reduction_result_shape(
    x: torch.Tensor, axis: Union[int, list[int]], keepdims: bool = False
) -> Tuple[Sequence[int], SpyreTensorLayout]:
    # Normalize axis
    x_size = x.size()
    if isinstance(axis, int):
        axis = [axis]
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += len(x_size) if len(x_size) else 1

    # Compute result shape + DCI
    x_layout: SpyreTensorLayout = x.get_spyre_layout()
    is_stick_reduction = x_layout.is_stick_reduction(axis)
    res_size = list(x_size)
    res_order = x_layout.host_dim_order()
    for d in axis:
        if keepdims:
            res_size[d] = 1
        else:
            res_size[d] = -1
            res_order[d] = -1
            res_order = [rd if rd < d else rd - 1 for rd in res_order]
    res_size = [rs for rs in res_size if rs >= 0]
    res_order = [rd for rd in res_order if rd >= 0]
    res_format = StickFormat.Sparse if is_stick_reduction else StickFormat.Dense
    res_layout = SpyreTensorLayout(res_size, x.dtype, res_order, format=res_format)
    return res_size, res_layout


def spyre_pointwise_result_shape(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[Sequence[int], SpyreTensorLayout]:
    """
    Compute the shape of the result of a pointwise binary operation.
    The code is based on torch.broadcast_shapes with Spyre enhancements.
    """
    x_size = x.size()
    y_size = y.size()
    res_size = [1] * max(len(x_size), len(y_size))
    x_broadcasted = [False] * len(res_size)
    y_broadcasted = [False] * len(res_size)
    for i in range(-1, -1 - len(x_size), -1):
        res_size[i] = x_size[i]

    for i in range(-1, -1 - len(y_size), -1):
        # NB: handle nested ints specially to avoid invalid guarding on Ne(j0, 1).
        if is_nested_int(y_size[i]):
            # Broadcasting is allowed for (j0, 1) or (j0, j0);
            # not (j0, j1), (j0, 5), etc.
            if is_nested_int(res_size[i]) and guard_size_oblivious(
                y_size[i] == res_size[i]
            ):
                continue
        else:
            if guard_size_oblivious(y_size[i] == res_size[i]):
                continue
            if guard_size_oblivious(y_size[i] == 1) and not guard_size_oblivious(
                res_size[i] == 1
            ):
                y_broadcasted[i] = True
                continue

        if res_size[i] != 1:
            raise RuntimeError(
                "Shape mismatch: objects cannot be broadcast to a single shape"
            )
        res_size[i] = y_size[i]
        x_broadcasted[i] = True

    x_layout = x.get_spyre_layout()
    y_layout = y.get_spyre_layout()
    if x_layout.format == y_layout.format:
        res_format = x_layout.format
    elif x_layout.format == StickFormat.Dense and y_broadcasted[x_layout.stick_dim()]:
        res_format = StickFormat.Dense
    elif y_layout.format == StickFormat.Dense and x_broadcasted[y_layout.stick_dim]:
        res_format = StickFormat.Dense
    else:
        raise Unsupported(
            f"binop with incompatible DCIs: {x_layout} {y_layout} {x_broadcasted} {y_broadcasted}"
        )

    # TODO: Forcing generic stick dimension order
    dim_order = list(range(len(res_size)))
    return res_size, SpyreTensorLayout(res_size, x.dtype, dim_order, format=res_format)


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
            case spyreop.compact.default:
                raise Unsupported("TODO: compact")
            case spyreop.exx2.default:
                raise Unsupported("TODO: exx2")
            case spyreop.layernorm.default:
                raise Unsupported("TODO: layernorm")
            case spyreop.layernormnorm:
                raise Unsupported("TODO: layernormnorm")
            case spyreop.layernormscale:
                raise Unsupported("TODO: layernormscale")
            case spyreop.slice.default:
                raise Unsupported("TODO: slice")
            case spyreop.swap.default:
                raise Unsupported("TODO: swap")
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
                    print(f"Checking: {i} {j} {var} {input_dims[j][input_dim_idx[j]]}")
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
        stl = SpyreTensorLayout(output.size, output.dtype)
        stl.format = output_format
        return FixedTiledLayout(
            output.device, output.dtype, output.size, output.stride, stl
        )


def reduction_layout(n: SchedulerNode, args: list[Arg]) -> FixedTiledLayout:
    print(f"Convert reduction: {n} {args}")
    raise Unsupported("TODO")


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
                # TODO: This should become a hard-error
                print(
                    f"Warning: missing device_tensor_layout on graph input {name}; assuming generic stick layout"
                )
                stl = SpyreTensorLayout(real_input.size(), real_input.dtype)
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
