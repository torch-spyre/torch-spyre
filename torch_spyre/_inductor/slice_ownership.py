# Copyright 2026 The Torch-Spyre Authors.
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

"""Pure coordinate proofs for tensor-slice ownership."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import sympy


def project_affine_slice_axis(
    coordinate: sympy.Expr,
    device_extent: int,
    split: int,
    iteration_space: Mapping[sympy.Symbol, sympy.Expr],
) -> tuple[sympy.Symbol, sympy.Expr] | None:
    """Prove that one loop owns slices of a multi-loop affine coordinate.

    A reshaped Q read has stick coordinate ``2 * head + pair``. Splitting
    sixteen sticks four ways cuts the head loop, while the two rotation-pair
    values stay inside every owned slice. Counting free symbols alone rejects
    that legal projection. Bound the *whole* affine rectangle for each loop
    partition, including every other loop, before using its zero-origin line
    in the existing ownership proof. This proves containment, not permission
    to change an access, its physical layout, or the owners.

    Deliberately decline nonlinear, reversed, symbolic and ambiguous cases.
    No Cartesian enumeration or relaxed proof budget is needed: extrema of
    an ascending affine rectangle occur at its endpoints.
    """
    symbols = coordinate.free_symbols
    if len(symbols) < 2 or not symbols <= iteration_space.keys():
        return None
    origin = coordinate.xreplace(dict.fromkeys(symbols, sympy.S.Zero))
    coefficients = {symbol: coordinate.coeff(symbol) for symbol in symbols}
    if (
        origin.is_Integer is not True
        or any(c.is_Integer is not True or c <= 0 for c in coefficients.values())
        or sympy.expand(
            coordinate - origin - sum(c * s for s, c in coefficients.items())
        )
        != 0
    ):
        return None
    extents = {}
    for symbol in symbols:
        extent = iteration_space[symbol]
        extent = sympy.sympify(extent[0] if isinstance(extent, tuple) else extent)
        if extent.is_Integer is not True or extent <= 0:
            return None
        extents[symbol] = int(extent)
    width = device_extent // split
    projections = []
    for loop, extent in extents.items():
        if extent % split:
            continue
        step = coefficients[loop]
        other_span = sum(
            coefficients[s] * (size - 1) for s, size in extents.items() if s != loop
        )
        loop_width = extent // split
        for slot in range(split):
            low = origin + step * slot * loop_width
            high = low + step * (loop_width - 1) + other_span
            if not slot * width <= low <= high < (slot + 1) * width:
                break
        else:
            projections.append((loop, origin + step * loop))
    return projections[0] if len(projections) == 1 else None


def project_floor_affine_slice_axis(
    coordinate: sympy.Expr,
    device_extent: int,
    split: int,
    iteration_space: Mapping[sympy.Symbol, sympy.Expr],
) -> tuple[sympy.Symbol, sympy.Expr] | None:
    """Certify a single floor of an ascending affine coordinate.

    Attention repack writes feature sticks at ``2*head + floor(width/64)``.
    The width loop stays within a head shard, but is not itself affine. For
    integer loops, an integer-affine A satisfies A + floor(B) = floor(A+B).
    After clearing the positive integer denominator D, containment of the
    integer numerator in [D*lo, D*hi) proves containment of its floor in
    [lo, hi). Reuse the affine proof on that scaled physical axis, then return
    the ORIGINAL access's representative line. No emitted access changes.

    Only one floor with unit coefficient is admitted. Nonlinear numerators,
    symbolic divisors, reversed accesses and crossing slices still fail the
    existing affine containment proof.
    """
    floors = coordinate.atoms(sympy.floor)
    if len(floors) != 1:
        return None
    quotient = next(iter(floors))
    integer_part = sympy.expand(coordinate - quotient)
    symbols = coordinate.free_symbols
    origin = integer_part.xreplace(dict.fromkeys(symbols, sympy.S.Zero))
    coefficients = {symbol: integer_part.coeff(symbol) for symbol in symbols}
    if (
        origin.is_Integer is not True
        or any(c.is_Integer is not True for c in coefficients.values())
        or sympy.expand(
            integer_part - origin - sum(c * s for s, c in coefficients.items())
        )
        != 0
    ):
        return None
    numerator, denominator = sympy.together(
        integer_part + quotient.args[0]
    ).as_numer_denom()
    if denominator.is_Integer is not True or denominator <= 0:
        return None
    projected = project_affine_slice_axis(
        sympy.expand(numerator),
        device_extent * int(denominator),
        split,
        iteration_space,
    )
    if projected is None:
        return None
    loop, _ = projected
    return loop, coordinate.xreplace(dict.fromkeys(symbols - {loop}, sympy.S.Zero))


@dataclass(frozen=True)
class SliceOwnershipProof:
    name: str
    certificate: str
    prove: Callable[..., tuple[sympy.Symbol, sympy.Expr] | None]
    floor_recovery: bool
    requires: tuple[str, ...]
    tests: tuple[str, ...]


SLICE_OWNERSHIP_PROOFS = (
    SliceOwnershipProof(
        "Affine rectangle containment",
        "project_affine_slice_axis",
        project_affine_slice_axis,
        False,
        (
            "one contained loop partition",
            "complete read rectangle",
            "unchanged physical coordinates",
        ),
        (
            "test_q_rope_ownership_keeps_the_whole_pair_in_its_slice",
            "test_q_projection_rope_handoff_preserves_pairs_and_other_consumer",
            "test_ownership_callers_keep_their_admission_policy",
        ),
    ),
    SliceOwnershipProof(
        "Floor-affine rectangle containment",
        "project_floor_affine_slice_axis",
        project_floor_affine_slice_axis,
        True,
        (
            "one ascending affine floor",
            "complete physical read rectangle",
            "unchanged coordinates and caller admission",
        ),
        (
            "test_floor_slice_projection_rejects_unproven_access",
            "test_attention_output_repack_handoff",
            "test_ownership_callers_keep_their_admission_policy",
        ),
    ),
)


def project_slice_axis(
    coordinate: sympy.Expr,
    device_extent: int,
    split: int,
    iteration_space: Mapping[sympy.Symbol, sympy.Expr],
    *,
    floor_only: bool = False,
) -> tuple[sympy.Symbol, sympy.Expr] | None:
    """Apply the caller's containment policy without changing its access."""
    for proof in SLICE_OWNERSHIP_PROOFS:
        if floor_only and not proof.floor_recovery:
            continue
        projected = proof.prove(coordinate, device_extent, split, iteration_space)
        if projected is not None:
            return projected
    return None


def factor_slice_axis(
    coordinate: sympy.Expr,
    device_extent: int,
    split: int,
    slot: sympy.Expr,
    iteration_space: Mapping[sympy.Symbol, sympy.Expr],
) -> list[tuple[int, sympy.Expr, int, sympy.Expr]] | None:
    """Factor an exact dense mixed-radix axis for the ownership proof only.

    K's ``2*head + pair`` can split both loops. V's
    ``2*head + floor(feature/64)`` can split heads and whole sticks. Unlike
    containment in one loop, this requires an exact zero-origin, hole-free
    product with independent digits and rectangular slice boundaries. It does
    not rewrite addresses or authorize an intra-stick layout change.
    """
    digits = []
    seen = set()
    for term in sympy.Add.make_args(sympy.expand(coordinate)):
        coefficient, digit = term.as_coeff_Mul()
        if coefficient.is_Integer is not True or coefficient <= 0:
            return None
        if isinstance(digit, sympy.Symbol):
            loop, divisor = digit, 1
        elif digit.func == sympy.floor:
            numerator, denominator = digit.args[0].as_numer_denom()
            if (
                not isinstance(numerator, sympy.Symbol)
                or denominator.is_Integer is not True
                or denominator <= 0
            ):
                return None
            loop, divisor = numerator, int(denominator)
        else:
            return None
        if loop in seen or loop not in iteration_space:
            return None
        seen.add(loop)
        extent = iteration_space[loop]
        extent = sympy.sympify(extent[0] if isinstance(extent, tuple) else extent)
        if extent.is_Integer is not True or extent <= 0 or extent % divisor:
            return None
        radix = int(extent) // divisor
        if radix <= 1:
            return None
        digits.append((int(coefficient), digit, radix))
    if not 2 <= len(digits) <= 4:
        return None
    digits.sort(key=lambda item: item[0], reverse=True)
    stride = device_extent
    remaining = split
    result = []
    for coefficient, digit, radix in digits:
        if stride % radix or coefficient != stride // radix:
            return None
        stride //= radix
        factor = min(radix, remaining)
        if radix % factor or remaining % factor:
            return None
        remaining //= factor
        owner = sympy.Mod(sympy.floor(slot / remaining), factor)
        result.append((radix, digit, factor, owner))
    return result if stride == remaining == 1 else None


def ownership_proofs_markdown() -> str:
    """Describe the exact proof order and the restricted recovery policy."""
    lines = [
        "| Ownership proof | Certificate | Requires | Covering tests |",
        "|---|---|---|---|",
    ]
    for proof in SLICE_OWNERSHIP_PROOFS:
        policy = "; floor-only recovery" if proof.floor_recovery else "; emitted access"
        tests = "; ".join(f"`{test}`" for test in proof.tests)
        lines.append(
            f"| {proof.name}{policy} | `{proof.certificate}` | "
            f"{', '.join(proof.requires)} | {tests} |"
        )
    return "\n".join(lines) + "\n"
