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


@dataclass(frozen=True)
class SliceOwnershipProof:
    name: str
    certificate: str
    prove: Callable[..., tuple[sympy.Symbol, sympy.Expr] | None]
    requires: tuple[str, ...]
    tests: tuple[str, ...]


SLICE_OWNERSHIP_PROOFS = (
    SliceOwnershipProof(
        "Affine rectangle containment",
        "project_affine_slice_axis",
        project_affine_slice_axis,
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
)


def project_slice_axis(
    coordinate: sympy.Expr,
    device_extent: int,
    split: int,
    iteration_space: Mapping[sympy.Symbol, sympy.Expr],
) -> tuple[sympy.Symbol, sympy.Expr] | None:
    """Apply the caller's containment policy without changing its access."""
    for proof in SLICE_OWNERSHIP_PROOFS:
        projected = proof.prove(coordinate, device_extent, split, iteration_space)
        if projected is not None:
            return projected
    return None


def ownership_proofs_markdown() -> str:
    """Describe the exact emitted-access proof order."""
    lines = [
        "| Ownership proof | Certificate | Requires | Covering tests |",
        "|---|---|---|---|",
    ]
    for proof in SLICE_OWNERSHIP_PROOFS:
        tests = "; ".join(f"`{test}`" for test in proof.tests)
        lines.append(
            f"| {proof.name}; emitted access | `{proof.certificate}` | "
            f"{', '.join(proof.requires)} | {tests} |"
        )
    return "\n".join(lines) + "\n"
