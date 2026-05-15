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

from torch.utils.weak import WeakTensorKeyDictionary

logger = get_inductor_logger("propagate_real_dims")

_real_dims = WeakTensorKeyDictionary()


def annotate_real_dims(tensor, info):
    """
    Annotate device tensor with real dimensions
    """
    _real_dims[tensor] = info
    return tensor


def _get_real_dims(dep):
    return V.graph.get_buffer(dep.name).real_dims


def _matmul_real_dims(op, inputs):
    # TODO use coordinates to find the real dims

    # the output dims
    op.real_dims = [
        _get_real_dims(inputs[0])[0],
        _get_real_dims(inputs[1])[1],
    ]

    # the op dims
    op.it_real_dims = [
        _get_real_dims(inputs[0])[0],
        _get_real_dims(inputs[0])[1],
        _get_real_dims(inputs[1])[1],
    ]


def _compute_real_dims(op, inputs):
    data = op.data

    if isinstance(data, Reduction) and data.reduction_type == BATCH_MATMUL_OP:
        return _matmul_real_dims(op, inputs)

    # TODO
    op.real_dims = _get_real_dims(inputs[0])
    op.it_real_dims = _get_real_dims(inputs[0])


def propagate_real_dims(
    operations: list[Operation],
) -> None:
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
                tb.real_dims = _real_dims.get(real_input)

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

    # debug print
    print("OPS")
    for op in iter(operations):
        print(op.get_operation_name(), op.it_real_dims)

    print("TENSORS")
    for buf in V.graph.buffers:
        print(buf.name, buf.real_dims)
