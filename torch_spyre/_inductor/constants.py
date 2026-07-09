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

from torch_spyre._C import ElementArrangement

BATCH_MATMUL_OP = "batchmatmul"
IDENTITY_OP = "identity"
RESTICKIFY_OP = "ReStickifyOpHBM"
BATCH_MATMUL_FP8_OP = "batchmatmulfp8"

# Reduction ops that cannot reduce along the stick dimension.
# Native prod reduction is not currently available in the backend.
# See backend issue #4409.
REDUCTIONS_NON_STICK_DIM_ONLY = {"prod"}

# Type casting operators from deeptools
DL16TOFP32_OP = "dl16tofp32"
FP32TODL16_OP = "fp32todl16"
FP8TODL16_OP = "fp8todl16"

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
    "prod",
]

# Operations that directly handle FP8 dtypes (SEN143_FP8)
# FP8 E4M3 numeric limits
FP8_E4M3_MAX = 448.0

SPYRE_FP8_OPS = {
    "qfp8ch",  # Channel-wise FP8 quantization (output: FP8)
    "fp8todl16",  # FP8 to FP16 conversion (input: FP8)
}

TOPK_OPS = {"topkvalue", "topkindex"}

LAYOUT_LABELS = ["OUTPUT", "KERNEL", "INPUT", "KERNEL_IDX"]
MATMUL_LAYOUT_LABELS = ["INPUT", "KERNEL", "OUTPUT", "KERNEL_IDX"]


# Populate more valid labels from deeptools here if needed
INPUT_DIM_LABELS = ["mb", "x", "y", "i", "j", "ki", "kj"]
OUTPUT_DIM_LABELS = ["out"]
MATMUL_DIM_LABELS = ["ki", "kj", "y", "x", "mb", "out", "in"]
