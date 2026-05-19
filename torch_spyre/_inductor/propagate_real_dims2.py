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


import sympy
import torch
from .logging_utils import get_inductor_logger
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.virtualized import V
from .errors import Unsupported
from .constants import BATCH_MATMUL_OP
from .pass_utils import host_coordinates
from .views import matching_dim
from torch.utils.weak import WeakTensorKeyDictionary

logger = get_inductor_logger("propagate_real_dims")


_real_dims = {}

_real_tensor_dims = WeakTensorKeyDictionary()


def declare_real_dim(name, size):
    """
    Declare a real dimension
    """
    _real_dims[name] = size


def annotate_real_dims(tensor, real_dims):
    """
    Annotate tensor with real dimensions: [(name, size), ...]
    """
    _real_tensor_dims[tensor] = real_dims
    return tensor


def _get_buffer(dep):
    return V.graph.get_buffer(dep.name)



def _lone_sym(coord: sympy.Expr) -> sympy.Symbol:
    return next(iter(coord.free_symbols))


def compute_input_real_dims(dep: MemoryDep) -> dict:
    """Map loop vars to real dim names for a single input dep."""
    buf = _get_buffer(dep)
    coords = host_coordinates(buf.get_layout(), dep)
    result = {}
    for i, coord in enumerate(coords):
        if coord.free_symbols:
            result[_lone_sym(coord)] = buf.real_dims[i]
    return result


def op_out_coords(op: ComputedBuffer) -> list:
    output_dep = next(iter(op.get_read_writes().writes))
    return host_coordinates(op.get_layout(), output_dep)


def coords_to_real_dims(coords: list, rdims: dict) -> list:
    """Map coordinate expressions to real dim names via their loop variable."""
    return [rdims[_lone_sym(c)] for c in coords if c.free_symbols]


def get_input_real_dims(inputs: list) -> dict:
    """
    Manage real_dims for all inputs, mapping dims from upstream node to this node.
    Must handle views, etc.
    """
    rdims = {}
    for inp in inputs:
        rdims.update(compute_input_real_dims(inp))
    return rdims


def get_reduction_dim(dep: MemoryDep, out_coords: list) -> sympy.Symbol:
    """Return the reduction loop variable: the input coord absent from the output."""
    in_coords = host_coordinates(_get_buffer(dep).get_layout(), dep)
    reduction_coord = next(
        c for c in in_coords
        if c.free_symbols and matching_dim(out_coords, c) is None
    )
    return _lone_sym(reduction_coord)


def _reduction_real_dims(op, inputs):
    """
    Works for single input reductions and matmul
    """
    rdims = get_input_real_dims(inputs)

    # Part 2: reduction dimension mapping
    out_coords = op_out_coords(op)
    reduction_var = get_reduction_dim(inputs[0], out_coords)
    op.real_dims = coords_to_real_dims(out_coords, rdims)
    op.real_ranges = op.real_dims + [rdims[reduction_var]]


def _pointwise_real_dims(op, inputs):

    rdims = get_input_real_dims(inputs)

    # Part 2: pointwise dimension mapping (no reduction)
    out_coords = op_out_coords(op)
    op.real_dims = coords_to_real_dims(out_coords, rdims)
    op.real_ranges = op.real_dims


def _compute_real_dims(op, inputs):
    """
    Augment op with real ranges and output real dims
    """
    if isinstance(op.data, Reduction):
        return _reduction_real_dims(op, inputs)
    if isinstance(op.data, Pointwise):
        return _pointwise_real_dims(op, inputs)
    raise NotImplementedError(f"real dims not implemented for {type(op.data)}")


def propagate_real_dims(
    operations: list[Operation],
) -> None:
    """
    Propagate real dims from inputs though graph
    """
    if len(V.graph.graph_input_names) > 0:
        for name, real_input in zip(V.graph.graph_input_names, V.get_real_inputs()):
            if isinstance(real_input, torch.Tensor):
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
                tb.real_dims = _real_tensor_dims.get(real_input)

    it = iter(operations)
    for op in it:
        if op.is_no_op():
            op.real_dims = []
            op.iterations = []
        elif isinstance(op, ComputedBuffer):
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                continue
            rw = op.get_read_writes()
            inputs = []
            for input in rw.reads:
                if isinstance(input, MemoryDep):
                    inputs.append(input)
            if isinstance(op.data, (Pointwise, Reduction)):
                _compute_real_dims(op, inputs)
            else:
                logger.warning(f"Warning: unhandled node type {type(op.data)}")
        else:
            logger.warning(f"unhandled operation type {type(op)}")

    # debug

    print("OPS")
    for op in iter(operations):
        print(op.get_operation_name(), op.real_ranges)

    print("TENSORS")
    for buf in V.graph.buffers:
        print(buf.name, buf.real_dims)
