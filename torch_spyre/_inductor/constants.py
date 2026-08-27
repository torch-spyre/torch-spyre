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
from torch_spyre._C import ElementArrangement

BATCH_MATMUL_OP = "batchmatmul"
IDENTITY_OP = "identity"
RESTICKIFY_OP = "ReStickifyOpHBM"
DEPTHWISE_CONV2D_OP = "depthwiseconv2dnative"
BATCH_MATMUL_FP8_OP = "batchmatmulfp8"
KEEP_BY_INDEX_OP = "keepbyindex"
MATMUL_REDUCTION_OPS = frozenset({BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP})

# Reduction ops that cannot reduce along the stick dimension.
# Native prod reduction is not currently available in the backend.
# See backend issue #4409.
REDUCTIONS_NON_STICK_DIM_ONLY = {"prod"}

# Type casting operators from deeptools
DL16TOFP32_OP = "dl16tofp32"
FP32TODL16_OP = "fp32todl16"
FP8TODL16_OP = "fp8todl16"
FP32TOINT32_OP = "fp32toint32"
INT32TOFP32_OP = "int32tofp32"

DEVICE_NAME = "spyre"

# The staggered EAs produced by the bidirectional fp16<->fp32 on-device
# conversions, whose device coordinates are non-sequential and (unlike QFP8CH)
# can be restored by the reverse conversion. propagate_layouts uses this set for
# two things: (1) deciding a dtype conversion must PRESERVE the input device
# layout (rescale in place) rather than reconstruct a dense one, and (2) picking
# the output EA / gating the broadcast handling in a multi-arg pointwise.
#
# NOTE: this is deliberately narrower than "all non-STANDARD EAs". is_ea_compatible
# does NOT use this set — it treats any single non-STANDARD EA (except EXX2) as
# broadcastable. QFP8CH is intentionally excluded here because membership also
# forces the convert-preserve path, which would mishandle the degenerate qfp8ch
# convert layout.
STAGGERED_EAS = frozenset(
    {
        ElementArrangement.DL16_TO_FP32,
        ElementArrangement.FP32_TO_DL16,
    }
)


def is_ea_compatible(eas) -> bool:
    """Return True if the given ElementArrangements can co-exist on one multi-arg
    pointwise op.

    Compatible when either:
      1. Every operand shares a single EA (any EA, including all-STANDARD), or
      2. The broadcast pattern: exactly one distinct *non-STANDARD* EA is present
         (on one or more operands) and every remaining operand is STANDARD. The
         STANDARD operands broadcast against the non-STANDARD ("staggered")
         element ordering, so a single such ordering is fine but two different
         ones are not.

    EXX2 is excluded from the broadcast pattern: it is a reduction mode (two
    values per stick), not a broadcastable ordering. It is only valid when *all*
    operands share it (case 1); layernorm ops carrying EXX2 are handled by a
    separate skip in ``validate_ops``.

    This predicate governs EA-*set* membership only. The additional device-layout
    constraint that STANDARD operands in the mixed case must broadcast at the
    stick dim (stick size 1) is enforced separately in
    ``_multi_arg_pointwise_layouts`` where the concrete layouts are available.

    Args:
        eas: An iterable of ElementArrangement values (duplicates allowed).
    """
    unique = set(eas)

    # Case 1: every operand shares a single EA (covers all-STANDARD too).
    if len(unique) <= 1:
        return True

    # Case 2: mixed EAs are only the broadcast pattern — exactly one distinct
    # non-STANDARD EA (and it must not be EXX2), with the rest STANDARD.
    #
    # NOTE: this accepts QFP8CH + STANDARD, but no current graph produces that
    # combination — QFP8CH tensors are consumed by an fp8 matmul or the fp8->fp16
    # convert, never a multi-arg pointwise. So QFP8CH is intentionally kept OUT of
    # STAGGERED_EAS (which doubles as the "convert must preserve the device
    # layout" gate; adding QFP8CH there would mis-handle the degenerate qfp8ch
    # convert layout). If QFP8CH broadcast ever becomes real, split those two
    # uses rather than widening STAGGERED_EAS.
    non_standard = unique - {ElementArrangement.STANDARD}
    return len(non_standard) == 1 and ElementArrangement.EXX2 not in non_standard


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

# The intermediates pool must leave headroom below the full segment size --
# 2 GiB is reserved for other segment-7 consumers (e.g. kernel-address/dim
# symbol bookkeeping), so the pool itself may never grow to claim the whole
# segment.
MAX_POOL_SIZE_BYTES = SEGMENT_SIZE - 2 * 1024**3

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
    "prod",
]

# FP8 E4M3 numeric limits
FP8_E4M3FN_INFO = torch.finfo(torch.float8_e4m3fn)
FP8_E4M3FN_MAX = float(FP8_E4M3FN_INFO.max)
FP8_E4M3FN_MIN = float(FP8_E4M3FN_INFO.min)

# Operations that directly handle FP8 dtypes (SEN143_FP8)
SPYRE_FP8_OPS = {
    "qfp8ch",  # Channel-wise FP8 quantization (output: FP8)
    "fp8todl16",  # FP8 to FP16 conversion (input: FP8)
    "batchmatmulfp8",  # FP8 bmm (inputs: FP8)
    "qfp8wt",  # FP8 quantization (output: FP8)
}

TOPK_OPS = {"topkvalue", "topkindex"}

LAYOUT_LABELS = ["OUTPUT", "KERNEL", "INPUT", "KERNEL_IDX"]
MATMUL_LAYOUT_LABELS = ["INPUT", "KERNEL", "OUTPUT", "KERNEL_IDX"]
CONV2D_LAYOUT_LABELS = ["OUTPUT", "INPUT", "KERNEL", "KERNEL_IDX"]

AVGPOOL2D_OP = "avgpoolfwd"
# Pool opfunc names, mirroring TOPK_OPS. Add maxpool/minpool here as they land so
# _is_pool stays a single membership test rather than a growing chain of ==.
POOL_OPS = {AVGPOOL2D_OP}

# Conv opfunc names. conv2d is a two-input reduction (activation + weight) with
# windowed spatial dims -- a hybrid of the matmul and pool patterns. Kept as a
# set so _is_conv is a single membership test as fp8/int8/int4 variants land.
CONV2D_FWD_OP = "conv2d"
# Both the forward conv2d (aten.convolution direct lowering, PR #3284) and the
# depthwise conv2d (spyre.conv2d, PR #3510) op strings are convolutions for the
# purposes of codegen dispatch (_is_conv). DEPTHWISE_CONV2D_OP is defined above.
CONV_OPS = {CONV2D_FWD_OP, DEPTHWISE_CONV2D_OP}

# Two-input reductions dispatched together in spyre_kernel.store_reduction:
# matmul (activation @ weight) and conv2d (activation * weight, reduced over
# in/ki/kj) both build [input, weight, output] tensor args.
TWO_INPUT_REDUCTION_OPS = frozenset(
    {BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP, CONV2D_FWD_OP, KEEP_BY_INDEX_OP}
)

# Depthwise conv is a two-input reduction like TWO_INPUT_REDUCTION_OPS but is
# dispatched in its own branch in spyre_kernel.store_reduction because it
# builds its tensor args differently (one filter per input channel).
DEPTHWISE_CONV_REDUCTION_OPS = frozenset({DEPTHWISE_CONV2D_OP})

# Single-input reductions: everything store_reduction dispatches to its
# fallback branch (exactly one input TensorArg). These are PyTorch/Inductor
# reduction_type strings (sum/mean/max/min/prod) plus Spyre-specific reduction
# ops (exx2, topkvalue/topkindex, avgpoolfwd) -- there is no upstream registry
# of supported reduction_type strings to derive this from, so it is written
# down here explicitly.
SINGLE_INPUT_REDUCTION_OPS = frozenset(
    {"sum", "mean", "max", "min", "prod", "exx2", *TOPK_OPS, AVGPOOL2D_OP}
)

# Populate more valid labels from deeptools here if needed
INPUT_DIM_LABELS = ["mb", "x", "y", "i", "j", "ki", "kj"]
OUTPUT_DIM_LABELS = ["out"]
MATMUL_DIM_LABELS = ["ki", "kj", "y", "x", "mb", "out", "in"]
CONV2D_DIM_LABELS = ["mb", "out", "i", "j", "ki", "kj"]
# Canonical avgpool iteration-space order: batch, out-H, out-W, channel,
# kernel-H, kernel-W. These SDSC labels are owned by the codegen layer; dim-role
# survival is derived from the node's live output ranges
# (OpSpec.node_output_ranges), never from these strings, so SDSC naming does not
# leak above codegen.
POOL_DIM_LABELS = ["mb", "i", "j", "out", "ki", "kj"]
# Canonical conv2d iteration-space order, mirroring POOL_DIM_LABELS: batch,
# out-H, out-W, out-channel, in-channel (the contraction dim), kernel-H,
# kernel-W. Like the pool labels, these SDSC strings are owned by the codegen
# layer. Codegen maps each iteration symbol to a role structurally, from the
# args' access expressions (set membership and co-occurrence in
# device_coordinates), never from sizes or positions -- see
# _match_labels_by_structure and _CONV_ROLE_LABELS in codegen/superdsc.py.
# Squeezed size-1 roles (e.g. batch N==1) never appear as symbols and drop out
# for free, so the mapping stays aligned with the surviving iteration-space
# dims.
CONV_DIM_LABELS = ["mb", "i", "j", "out", "in", "ki", "kj"]
