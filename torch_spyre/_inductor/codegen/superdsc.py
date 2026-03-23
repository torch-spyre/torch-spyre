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

from torch_spyre._inductor.constants import (
    MATMUL_REDUCTION_OP,
    BATCH_MATMUL_OP,
    TRANSPOSE_OP,
    CLONE_OP,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import OpSpec
from torch_spyre._inductor.constants import SEGMENT_OFFSETS
from .compute_ops import generate_sfp_op, generate_matmul, generate_bmm
from .data_ops import (
    generate_slice,
    generate_transpose,
    generate_transpose_3d_stick,
    generate_transpose_4d_stick,
    generate_identity,
)

logger = get_inductor_logger("codegen.superdsc")

_argument_names = ["arg0", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6"]


def compile_op_spec(kernel_name: str, op_spec: OpSpec) -> tuple[Any, list[int]]:
    inputs = []
    outputs = []
    arg_map = []
    for index, ts in enumerate(op_spec.args):
        # use node seq (idx in nodes) to verify whether to reuse lx for this buffer,
        # in case same Op used twice in sequence and only want pin 1 of them
        lx_addr = None
        for k, addr in getattr(ts, "allocation", {}).items():
            if kernel_name.split("_")[-1] == k.replace("lx:", ""):
                lx_addr = addr

        if ts.is_input:
            inputs.append(
                {
                    "name": _argument_names[index],
                    "it_dim_map": ts.it_dim_map,
                    "device_layout": ts.device_layout,
                    "lx_addr": lx_addr,
                }
            )
            arg_map.append(ts.arg_index)
        else:
            outputs.append(
                {
                    "name": _argument_names[index],
                    "it_dim_map": ts.it_dim_map,
                    "device_layout": ts.device_layout,
                    "lx_addr": lx_addr,
                }
            )
            arg_map.append(ts.arg_index)
    kernel_descriptor = {
        "name": kernel_name,
        "reduction": op_spec.is_reduction,
        "op": op_spec.op,
        "dimensions": op_spec.iteration_space,
        "inputs": inputs,
        "outputs": outputs,
    }
    if op_spec.op_info is not None:
        kernel_descriptor["op_info"] = op_spec.op_info
    pointers = dict(zip(_argument_names, SEGMENT_OFFSETS))
    dt_sdsc = generate_sdsc(pointers, **kernel_descriptor)
    return dt_sdsc, arg_map


def generate_sdsc(pointers, *, op, dimensions, inputs, outputs, reduction, **kwargs):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"SDSC generation: op={op}, dimensions={dimensions}, "
            f"is_reduction={reduction}, num_inputs={len(inputs)}, num_outputs={len(outputs)}"
        )

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
    if op == "to_dtype":
        if (
            inputs[0]["device_layout"].device_dtype
            == outputs[0]["device_layout"].device_dtype
        ):
            return generate_identity(
                pointers,
                op=CLONE_OP,
                dimensions=dimensions,
                inputs=inputs,
                outputs=outputs,
                **kwargs,
            )
        else:
            raise Unsupported(
                f"to_dtype from {inputs[0]['device_layout'].device_dtype} to {outputs[0]['device_layout'].device_dtype}"
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
        if (
            inputs[0]["device_layout"].host_stick_dim() in transposed_dims
        ):  # stick transpose implemented through restickify
            return generate_transpose_3d_stick(
                pointers,
                op=op,
                dimensions=dimensions,
                inputs=inputs,
                outputs=outputs,
                transposed_dims=transposed_dims,
                **kwargs,
            )
        else:  # non-stick transpose implemented through identity
            return generate_identity(
                pointers,
                op=op,
                dimensions=dimensions,
                inputs=inputs,
                outputs=outputs,
                **kwargs,
            )
    if op == TRANSPOSE_OP and len(dimensions) == 4:
        transposed_dims = [
            dim % len(dimensions) for dim in kwargs["op_info"]["transposed_dims"]
        ]
        if (
            inputs[0]["device_layout"].host_stick_dim() in transposed_dims
        ):  # stick transpose implemented through restickify
            return generate_transpose_4d_stick(
                pointers,
                op=op,
                dimensions=dimensions,
                inputs=inputs,
                outputs=outputs,
                transposed_dims=transposed_dims,
                **kwargs,
            )
        else:  # non-stick transpose implemented through identity
            return generate_identity(
                pointers,
                op=op,
                dimensions=dimensions,
                inputs=inputs,
                outputs=outputs,
                **kwargs,
            )
    if op == CLONE_OP:
        return generate_identity(
            pointers,
            op=op,
            dimensions=dimensions,
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
    return generate_sfp_op(
        pointers,
        op=op,
        dimensions=dimensions,
        inputs=inputs,
        outputs=outputs,
        reduction=reduction,
        **kwargs,
    )
