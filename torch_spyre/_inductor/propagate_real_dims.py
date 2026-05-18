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
from .views import compute_coordinates
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


def _compute_real_layout(real_dims):
    """
    Compute real size and stride based on real dims
    """
    size = []
    stride = [1]
    for s in reversed(real_dims):
        stride.append(stride[-1] * _real_dims[s])
        size.append(_real_dims[s])
    return list(reversed(size)), list(reversed(stride[:-1]))


def _matmul_real_dims(op, inputs):
    """
    Augment matmul with real ranges and output real dims
    """
    ranges0 = inputs[0].ranges
    index0 = inputs[0].index
    real_dims0 = _get_buffer(inputs[0]).real_dims
    real_size0, real_stride0 = _compute_real_layout(real_dims0)

    ranges1 = inputs[1].ranges
    index1 = inputs[1].index
    real_dims1 = _get_buffer(inputs[1]).real_dims
    real_size1, real_stride1 = _compute_real_layout(real_dims1)

    # compute coordinates based on real dims
    real_coords0 = compute_coordinates(real_size0, real_stride0, ranges0, index0)
    real_coords1 = compute_coordinates(real_size1, real_stride1, ranges1, index1)

    # identify order of op dims
    keys = ranges0.keys()
    vars0 = index0.free_symbols
    vars1 = index1.free_symbols
    dim0 = next(iter(keys - vars1))
    dim1 = next(iter(vars0 & vars1))
    dim2 = next(iter(keys - vars0))

    # match iteration variables to real dims
    matches = {dim0: None, dim1: None, dim2: None}
    for i, coord in enumerate(real_coords0):
        if coord.is_symbol:
            matches[next(iter(coord.free_symbols))] = real_dims0[i]
    for i, coord in enumerate(real_coords1):
        if coord.is_symbol:
            matches[next(iter(coord.free_symbols))] = real_dims1[i]

    # add output real dims
    op.real_ranges = [matches[dim0], matches[dim1], matches[dim2]]

    # add op real ranges
    op.real_dims = [matches[dim0], matches[dim2]]


def _compute_real_dims(op, inputs):
    """
    Augment op with real ranges and output real dims
    """
    if isinstance(op.data, Reduction) and op.data.reduction_type == BATCH_MATMUL_OP:
        return _matmul_real_dims(op, inputs)

    # TODO handle other op types
    op.real_ranges = _get_buffer(inputs[0]).real_dims
    op.real_dims = _get_buffer(inputs[0]).real_dims


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
