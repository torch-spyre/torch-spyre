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
BATCH_MATMUL_FP8_OP = "batchmatmulfp8"

# Type casting operators from deeptools
DL16TOFP32_OP = "dl16tofp32"
FP32TODL16_OP = "fp32todl16"

DEVICE_NAME = "spyre"

# Marker on a ComputedBuffer that should be considered for copy-back removal.
# ``aten.copy_`` lowering sets this on the explicit copy-back mutation op; layout
# propagation later proves feasibility and either removes the copy or leaves it
# intact.
COPY_BACK_CANDIDATE_ATTR = "_spyre_copy_back_candidate"

# Marker on a ComputedBuffer whose layout was retargeted so that the producer
# writes a graph input directly. Downstream passes use this to distinguish a
# compute mutation op from a pure-copy mutation op.
ELIDED_COPY_BACK_ATTR = "_spyre_writes_copy_back_target"

# FX ``custom`` metadata key for BMMs created from a shared 2D weight whose
# logical batch dim is statically 1.  The downstream OpSpec key carries the same
# fact after lowering, where FX metadata is no longer directly available.
SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY = "_spyre_shared_weight_unit_bmm"
SHARED_WEIGHT_UNIT_BMM_INFO_KEY = "shared_weight_unit_bmm"


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
    "mean",
    "sum",
    "max",
    "min",
    "layernormscale",
    "abs",
    "neg",
    "exp",
    "sigmoid",
    "exx2",
    "layernormnorm",
    "identity",
    "sqrt",
    "rsqrt",
    "topkvalue",
    "topkindex",
    "floor",
    "to_dtype",
    "maximum",
    "minimum",
    "greaterthan",
    "greaterequal",
    "lesserthan",
    "lesserequal",
    "equal",
    "notequal",
]

TOPK_OPS = {"topkvalue", "topkindex"}

LAYOUT_LABELS = ["OUTPUT", "KERNEL", "INPUT", "KERNEL_IDX"]
MATMUL_LAYOUT_LABELS = ["INPUT", "KERNEL", "OUTPUT", "KERNEL_IDX"]


# Populate more valid labels from deeptools here if needed
INPUT_DIM_LABELS = ["mb", "x", "y", "i", "j", "ki", "kj"]
OUTPUT_DIM_LABELS = ["out"]
MATMUL_DIM_LABELS = ["ki", "kj", "y", "x", "mb", "out", "in"]
