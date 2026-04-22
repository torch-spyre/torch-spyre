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
IDENTITY_OP = "identity"
RESTICKIFY_OP = "ReStickifyOpHBM"

# Type casting operators from deeptools
DL16TOFP32_OP = "dl16tofp32"
FP32TODL16_OP = "fp32todl16"

# identical dtypes due to PR #1605 loads bfloat16 tensors using DL16
# DL16TOBF16_OP = "dl16tobf16"
# FP8TODL16_OP = "fp8todl16"

# not available in deeptools but can be supported using DL16, refer PR #1605
# FP32TOBF16_OP = "fp32tobf16"
# BF16TOFP32_OP = "bf16tofp32"

# not implemented / available in deeptools yet
# FP32TOFP8_OP = "fp32tofp8"
# FP8TOFP32_OP = "fp8tofp32"
# BF16TODL16_OP = "bf16todl16"
# INT32TOINT16_OP = "int32toint16"
# INT16TOINT32_OP = "int16toint32"
# INT32TOINT8_OP = "int32toint8"
# INT8TOINT32_OP = "int8toint32"

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

INTERMEDIATES_SEGMENT = 0x0
SEGMENT_SIZE = 0x400000000

SPYRE_FP32_OPS = [
    "add",
    "sub",
    "mul",
    "where",
    "realdiv",
    "relufwd",
    "reciprocal",
    "layernormscale",
    "abs",
    "neg",
    "exp",
    "sigmoid",
    "exx2",
    "layernormnorm",
    "identity",
    "overwrite",
    "topkvalue",
    "topkindex",
    "floor",
    "to_dtype",
]

TOPK_OPS = {"topkvalue", "topkindex"}

LAYOUT_LABELS = ["OUTPUT", "KERNEL", "INPUT", "KERNEL_IDX"]
MATMUL_LAYOUT_LABELS = ["INPUT", "KERNEL", "OUTPUT", "KERNEL_IDX"]


# Populate more valid labels from deeptools here if needed
INPUT_DIM_LABELS = ["mb", "x", "y", "i", "j", "ki", "kj"]
OUTPUT_DIM_LABELS = ["out"]
MATMUL_DIM_LABELS = ["x", "mb", "y", "ki", "kj", "out", "in"]
