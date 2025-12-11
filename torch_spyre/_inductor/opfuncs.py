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

UNIMPLEMENTED = "unimplemented"


opfunc_mapping: dict[str, str] = {}


def _initialize_opfunc_mapping():
    from .spyre_kernel import SpyreKernelOverrides

    # Inductor reduction operations
    reductions = {
        "argmax": "argmax",
        "argmin": "argmin",
        "exx2": "exx2",
        "welford_reduce": UNIMPLEMENTED,
        "welford_combine": UNIMPLEMENTED,
        "any": UNIMPLEMENTED,
        "matmul": "matmul",
        "batchmatmul": "batchmatmul",
        "max": "max",
        "min": "min",
        "prod": UNIMPLEMENTED,
        "sum": "sum",
        "xor_sum": UNIMPLEMENTED,
        "mean": "mean",
    }

    # Default all Inductor ops to UNIMPLEMENTED
    pointwise_ops = {
        attr: UNIMPLEMENTED
        for attr in dir(SpyreKernelOverrides)
        if callable(getattr(SpyreKernelOverrides, attr)) and not attr.startswith("_")
    }
    # Implemented pointwise ops whose opfunc is the same as the inductor op
    same_name = [
        "abs",
        "add",
        "exp",
        "gelu",
        "layernormnorm",
        "layernormscale",
        "log",
        "mul",
        "reciprocal",
        "rsqrt",
        "sigmoid",
        "softplus",
        "sqrt",
        "sub",
        "tanh",
    ]
    for i in same_name:
        pointwise_ops[i] = i
    # Implemented pointwise ops that need to be renamed
    pointwise_ops["truediv"] = "realdiv"
    pointwise_ops["relu"] = "relufwd"
    pointwise_ops["eq"] = "equal"
    pointwise_ops["ne"] = "notequal"
    pointwise_ops["ge"] = "greaterequal"
    pointwise_ops["where"] = "where3"

    return pointwise_ops | reductions


def get_spyre_op(op: str) -> str:
    """Map PyTorch inductor ops to Spyre OpFuncs"""

    if not opfunc_mapping:
        opfunc_mapping.update(_initialize_opfunc_mapping())

    spyre_op = opfunc_mapping.get(op, UNIMPLEMENTED)
    return spyre_op
