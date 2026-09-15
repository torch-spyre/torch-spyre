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

"""Per-op ElementArrangement propagation and correctness rules.

This module encodes the static EA transfer rules for each operation class in
the Spyre backend.  It answers two questions for every op:

  1. **Propagation** — what EA should the output carry?
  2. **Correctness** — given the input EA(s), does the hardware op produce
     numerically correct results, or must the caller restore the input EA to
     STANDARD first?

The scope is limited to the two reversible staggered EAs (DL16_TO_FP32 and
FP32_TO_DL16) that arise from on-device FP16↔FP32 conversions.  QFP8CH,
QFP8WT, EXX2, and FP8 are out of scope for now.

Reference: ``docs/ea_rules.md`` (see also the inline per-class docstrings).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

from torch_spyre._C import ElementArrangement

# ---------------------------------------------------------------------------
# Public EA sets (re-exported for callers that need them)
# ---------------------------------------------------------------------------

#: The two reversible staggered EAs produced by on-device FP16↔FP32 conversions.
STAGGERED_EAS: frozenset[ElementArrangement] = frozenset(
    {ElementArrangement.DL16_TO_FP32, ElementArrangement.FP32_TO_DL16}
)

_STANDARD = ElementArrangement.STANDARD


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class EaAction(Enum):
    """What a caller must do when it receives an EaResult."""

    OK = auto()
    """The op is numerically correct; proceed with the given output EA."""

    RESTORE_INPUT = auto()
    """The op cannot produce correct results with the current input EA(s).
    The caller must insert a reverse conversion upstream (fp32todl16 or
    dl16tofp32) to restore STANDARD EA before running this op."""


@dataclass(frozen=True)
class EaResult:
    """Return value from every EA rule function.

    Attributes:
        action: Whether the op is OK or the caller must restore input EA.
        output_ea: The EA the output tensor should carry.  Meaningful only
            when ``action == EaAction.OK``.
    """

    action: EaAction
    output_ea: ElementArrangement

    # Convenience constructors ---------------------------------------------------

    @staticmethod
    def ok(output_ea: ElementArrangement) -> "EaResult":
        return EaResult(EaAction.OK, output_ea)

    @staticmethod
    def restore() -> "EaResult":
        return EaResult(EaAction.RESTORE_INPUT, _STANDARD)


# ---------------------------------------------------------------------------
# Class 0 — Single-arg pointwise
# ---------------------------------------------------------------------------


def single_arg_pointwise_ea(input_ea: ElementArrangement) -> EaResult:
    """EA rule for single-arg pointwise ops (neg, abs, relu, exp, …).

    A single-arg pointwise applies the same function to every memory slot
    independently.  There is no cross-input pairing problem, so any EA is
    accepted and the output carries the same EA as the input.

    Output EA rule: propagate the input EA unchanged.
    """
    return EaResult.ok(input_ea)


# ---------------------------------------------------------------------------
# Class 1 — Multi-arg pointwise
# ---------------------------------------------------------------------------


def multi_arg_pointwise_ea(
    input_eas: Sequence[ElementArrangement],
    *,
    broadcast_mask: Sequence[bool] | None = None,
) -> EaResult:
    """EA rule for multi-arg pointwise ops (add, mul, where, …).

    The hardware computes ``out[i] = f(A[i], B[i], …)`` slot-by-slot.  For the
    result to be numerically correct, all non-broadcast inputs must share the
    same EA so that the same tensor element occupies slot ``i`` in every operand.

    Args:
        input_eas: EA of each input tensor.
        broadcast_mask: Optional boolean sequence, same length as
            ``input_eas``.  ``True`` at position ``k`` means tensor ``k`` is a
            size-1 broadcast along the stick dimension and is therefore always
            compatible regardless of its EA.  When ``None``, no input is
            treated as a broadcast (conservative default).

    Returns:
        ``EaResult.ok(shared_ea)`` when all non-broadcast inputs share a single
        EA; ``EaResult.restore()`` when they do not.

    Output EA rule:
        - All STANDARD → STANDARD.
        - All share one staggered EA (or broadcast) → that staggered EA.
        - Mix of different EAs (excluding broadcasts) → RESTORE_INPUT.
    """
    if broadcast_mask is None:
        broadcast_mask = [False] * len(input_eas)

    non_broadcast_eas = {
        ea for ea, is_bc in zip(input_eas, broadcast_mask) if not is_bc
    }

    if len(non_broadcast_eas) <= 1:
        # All non-broadcast inputs agree (or there are none).
        ea = next(iter(non_broadcast_eas), _STANDARD)
        return EaResult.ok(ea)

    # More than one distinct EA among non-broadcast inputs — mismatched pairing.
    return EaResult.restore()


# ---------------------------------------------------------------------------
# Class 2 — Reduction
# ---------------------------------------------------------------------------


def reduction_ea(
    input_ea: ElementArrangement,
    *,
    reducing_stick_dim: bool,
) -> EaResult:
    """EA rule for reduction ops (sum, mean, max, min, …).

    Two sub-cases depending on which dimension is being reduced:

    **Case A — reducing along the stick dimension** (``reducing_stick_dim=True``):
        The hardware accumulates all elements within each stick.  Since
        addition/max/min are order-independent, any EA (STANDARD or staggered)
        produces a correct result.  After the stick dim is consumed, the
        staggered ordering within that stick no longer exists, so the output
        EA becomes STANDARD.  Exception: a staggered EA carried by a *non-stick*
        dimension is unaffected and propagates to the output.

    **Case B — reducing along a non-stick dimension** (``reducing_stick_dim=False``):
        Each full stick is accumulated as a unit across the outer dimension.
        The intra-stick ordering is never touched, so a stick-staggered EA
        propagates to the output unchanged.  A non-stick staggered EA, however,
        corresponds to the dimension being reduced — its ordering is consumed
        and the output EA becomes STANDARD.

    Args:
        input_ea: EA of the input tensor.
        reducing_stick_dim: True if the reduction loop variable is on the stick
            dimension; False if it is on an outer (non-stick) dimension.

    Returns:
        Always ``EaResult.ok(…)`` — reductions never require a restore because
        addition/max/min are commutative.

    Output EA rule:
        - Stick reduction: STANDARD (staggered ordering consumed);
          non-stick staggered EA propagates through.
        - Non-stick reduction: propagate stick staggered EA; STANDARD when the
          non-stick dim's staggered EA is consumed.

    Note:
        When a reduction produces a size-1 tensor along the reduced dimension
        the current implementation propagates the input EA; this is a known
        limitation (see module docstring).
    """
    if input_ea not in STAGGERED_EAS:
        # STANDARD or other non-staggered EA (EXX2, QFP8CH, …) — out of scope.
        return EaResult.ok(input_ea)

    if reducing_stick_dim:
        # Case A: staggered ordering on the stick is consumed → STANDARD.
        # (Non-stick staggered EA is not modelled here; callers handle that
        # dimension separately if needed.)
        return EaResult.ok(_STANDARD)
    else:
        # Case B: stick ordering survives; non-stick stagger consumed → STANDARD.
        # For now the stick EA is what we track, so propagate it.
        return EaResult.ok(input_ea)


# ---------------------------------------------------------------------------
# Class 3 — Restickify (Transpose)
# ---------------------------------------------------------------------------


def restickify_ea(
    stick_ea: ElementArrangement,
    target_ea: ElementArrangement,
) -> tuple[ElementArrangement, ElementArrangement]:
    """EA rule for restickify (transpose of stick ↔ non-stick dimension).

    Restickify swaps the stick dimension with one target non-stick dimension.
    The staggered status of those two dimensions is exchanged; all other
    non-stick dimensions are unaffected.

    Args:
        stick_ea: EA currently on the stick dimension.
        target_ea: EA currently on the target non-stick dimension (the dim
            that will become the new stick after the swap).

    Returns:
        ``(new_stick_ea, new_non_stick_ea)`` — the EA that will be on the new
        stick after the swap and the EA that will move to the (formerly-stick)
        non-stick dimension.

    Output EA rule:
        - The staggered EA of the current stick moves to the new non-stick dim.
        - The staggered EA of the target non-stick dim moves to the new stick.
        - Unswapped dimensions are unaffected (caller is responsible for them).
    """
    # After the swap: the old stick becomes a non-stick (carries old stick_ea),
    # and the target non-stick becomes the new stick (carries target_ea).
    return target_ea, stick_ea


# ---------------------------------------------------------------------------
# Class 4 — MatMul  C[M, N] = Σ_K  A[M, K] × B[K, N]
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatmulEaResult:
    """Detailed result for matmul EA analysis.

    Attributes:
        action: OK, or RESTORE_INPUT if K-contraction is mis-paired.
        output_ea: The EA the output C should carry (meaningful only when
            ``action == EaAction.OK``).
        a_restore: True if A's EA must be restored to STANDARD before the op.
        b_restore: True if B's EA must be restored to STANDARD before the op.
    """

    action: EaAction
    output_ea: ElementArrangement
    a_restore: bool
    b_restore: bool


def matmul_ea(
    a_ea: ElementArrangement,
    b_ea: ElementArrangement,
    *,
    a_ea_on_stick: bool,
    b_ea_on_stick: bool,
) -> MatmulEaResult:
    """EA rule for matmul  C[M, N] = Σ_K  A[M, K] × B[K, N].

    Hardware stick assignments:
        A [M, K]: stick dim = K  → layout [M, K/64, 64]
        B [K, N]: stick dim = N  → layout [K/64, 64, N/64, 64]
        C [M, N]: stick dim = N  → layout [M, N/64, 64]

    Three independent effects govern correctness (see module-level doc):

    1. **K-contraction correctness**: A stick stagger scrambles K-indices in
       A's sticks; a B non-stick stagger reorders B's rows in memory —  both
       mis-pair K.  Exception: when A has a stick stagger *and* B has a
       non-stick stagger, both permutations are the same so they cancel.

    2. **C column positions**: B stick stagger reorders C's N-columns.
       Recoverable by re-indexing (output carries stick stagger on N).

    3. **C row positions**: A non-stick stagger reorders C's M-rows.
       Recoverable by re-indexing (output carries non-stick stagger on M).

    Args:
        a_ea: EA of input A.
        b_ea: EA of input B.
        a_ea_on_stick: True if A's staggered EA lives on the stick dim (K).
        b_ea_on_stick: True if B's staggered EA lives on the stick dim (N).

    Returns:
        ``MatmulEaResult`` with ``action``, ``output_ea``, and per-input
        restore flags.
    """
    a_staggered = a_ea in STAGGERED_EAS
    b_staggered = b_ea in STAGGERED_EAS

    # Short-circuit: both STANDARD — trivially correct.
    if not a_staggered and not b_staggered:
        return MatmulEaResult(EaAction.OK, _STANDARD, False, False)

    a_stick_stagger = a_staggered and a_ea_on_stick
    a_nonstick_stagger = a_staggered and not a_ea_on_stick
    b_stick_stagger = b_staggered and b_ea_on_stick
    b_nonstick_stagger = b_staggered and not b_ea_on_stick

    # Determine K-contraction correctness.
    # A stick stagger mis-pairs K unless cancelled by B non-stick stagger.
    # B non-stick stagger mis-pairs K unless cancelled by A stick stagger.
    k_mispaired = (a_stick_stagger or b_nonstick_stagger) and not (
        a_stick_stagger and b_nonstick_stagger
    )

    if k_mispaired:
        # Cannot produce correct results — caller must restore.
        a_restore = a_stick_stagger
        b_restore = b_nonstick_stagger
        return MatmulEaResult(EaAction.RESTORE_INPUT, _STANDARD, a_restore, b_restore)

    # K-contraction is correct.  Determine output EA from surviving effects.
    # C columns: permuted if B has stick stagger (on N).
    # C rows:    permuted if A has non-stick stagger (on M).
    if b_stick_stagger and a_nonstick_stagger:
        # Both row and column permutations — output carries both effects.
        # Model as a combined staggered EA on the output (column stagger from B).
        output_ea = b_ea
    elif b_stick_stagger:
        output_ea = b_ea  # C columns permuted (stick stagger on N)
    elif a_nonstick_stagger:
        output_ea = a_ea  # C rows permuted (non-stick stagger on M)
    else:
        # Cancellation case: A stick stagger + B non-stick stagger → STANDARD.
        output_ea = _STANDARD

    return MatmulEaResult(EaAction.OK, output_ea, False, False)
