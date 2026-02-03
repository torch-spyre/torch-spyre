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

BATCH_MATMUL_OP = "batchmatmul"
CLONE_OP = "clone"
MATMUL_REDUCTION_OP = "matmul"
TRANSPOSE_OP = "transpose"

DEVICE_NAME = "spyre"


SEGMENT_OFFSETS = [
    0x0,
    0x400000000,
    0x800000000,
    0xC00000000,
    0x1000000000,
    0x1400000000,
    0x1800000000,
]

SPYRE_FP32_OPS = [
    "add",
    "sub",
    "mul",
    "where",
    "realdiv",
    "relu",
    "reciprocal",
    "layernormscale",
    "abs",
    "neg",
    "exp",
    "sigmoid",
    "exx2",
    "layernormnorm",
]
