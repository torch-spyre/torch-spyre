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

from torch_spyre._inductor.constants import MATMUL_REDUCTION_OP, BATCH_MATMUL_OP, TRANSPOSE_OP
from torch_spyre._inductor import Unsupported
from .compute_ops import generate_sfp_op, generate_matmul, generate_bmm
from .data_ops import (
    generate_slice,
    generate_transpose,
    generate_transpose_3d_stick,
)


def generate_sdsc(pointers, *, op, dimensions, inputs, outputs, reduction, **kwargs):
    if op == MATMUL_REDUCTION_OP:
        return generate_matmul(
            pointers,
            op=op,
            dimensions=dimensions,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
    if op == BATCH_MATMUL_OP:
        return generate_bmm(
            pointers,
            op=op,
            dimensions=dimensions,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
    if op == "swap":
        return generate_transpose(
            pointers,
            op=op,
            dimensions=[dimensions[0], 64],
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
    if op == "slice":
        return generate_slice(
            pointers,
            op=op,
            dimensions=dimensions,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
    if op == TRANSPOSE_OP and len(dimensions) == 2:
        return generate_transpose(
            pointers,
            op=op,
            dimensions=dimensions,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
    if op == TRANSPOSE_OP and len(dimensions) == 3:
        transposed_dims = [
            dim % len(dimensions) for dim in kwargs["op_info"]["transposed_dims"]
        ]
        is_stick_transpose = (
            0 in transposed_dims or 1 in transposed_dims
        ) and 2 in transposed_dims
        if is_stick_transpose:
            return generate_transpose_3d_stick(
                pointers,
                op=op,
                dimensions=dimensions,
                inputs=inputs,
                outputs=outputs,
                transposed_dims=transposed_dims,
                **kwargs,
            )
        else:
            # Non-stick transpose currently unsupported
            raise Unsupported("Transposition not changing the stick dimension")
    return generate_sfp_op(
        pointers,
        op=op,
        dimensions=dimensions,
        inputs=inputs,
        outputs=outputs,
        reduction=reduction,
        **kwargs,
    )
