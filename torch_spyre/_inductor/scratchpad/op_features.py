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

"""Cost-model ``OpFeatures`` for a *candidate* core division.

The extractor (:mod:`torch_spyre._inductor.dump_cost_model`) reads each
op's **committed** ``op_it_space_splits``, so it yields features for the division
the compiler already chose. A co-optimizer needs features for every division in a
buffer's candidate menu, because the division is exactly what it is searching
over.

That turns out to need no re-derivation. ``CoreDivision`` stores
``(output_splits, reduction_splits)`` -- the coeff-keyed ``ItSpaceSplits`` pair
produced by :func:`pass_utils.splits_by_index_coeff` -- which is the same type,
in the same encoding, that ``op_it_space_splits`` holds and that
:func:`pass_utils.apply_splits_from_index_coeff` consumes. So a candidate is
evaluated by temporarily installing its pair on the op and re-running the
extractor: every division-dependent field (``cores``, ``reduction_cores``,
``matmul_rows_per_core`` / ``_cols_per_core``, ``tile_rows_per_core``) is then
recomputed by the extractor rather than by a second, drifting copy of its
axis-decoding rules.

Residency (``ArgTraffic.mem``) is deliberately *not* resolved here. It is the
other half of what the co-optimizer searches over, it is a plain per-argument
flag, and no other feature depends on it -- so features are emitted once per
(op, division) and residency is applied at scoring time by
:func:`with_residency`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Optional

from torch_spyre._inductor.cost_model import OpFeatures
from torch_spyre._inductor.logging_utils import get_inductor_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision

logger = get_inductor_logger("scratchpad.op_features")

# The attribute the extractor reads the division from.
_SPLITS_ATTR = "op_it_space_splits"


def features_for_division(op, division: "CoreDivision") -> Optional[OpFeatures]:
    """``OpFeatures`` for ``op`` as if it were divided per ``division``.

    Returns ``None`` when the op cannot be featurized (the extractor is
    best-effort and swallows its own failures, so a ``None`` here means the op
    itself was rejected, not that the division was bad).

    Temporarily swaps ``op_it_space_splits``. The swap is restored on every path
    including failure, so a caller that iterates a menu leaves the op exactly as
    it found it -- important because the same ``op`` objects stay live in the
    graph after capture.
    """
    from torch_spyre._inductor.dump_cost_model import extract_op_features

    had = hasattr(op, _SPLITS_ATTR)
    saved = getattr(op, _SPLITS_ATTR, None)
    try:
        setattr(op, _SPLITS_ATTR, (division.output_splits, division.reduction_splits))
        return extract_op_features(op)
    except Exception:  # noqa: BLE001 - featurization is best-effort by design
        logger.debug("could not featurize op for a candidate division", exc_info=True)
        return None
    finally:
        if had:
            setattr(op, _SPLITS_ATTR, saved)
        else:  # never had one: do not leave an attribute the op did not carry
            try:
                delattr(op, _SPLITS_ATTR)
            except AttributeError:
                pass


def features_for_menu(op, divisions) -> list[Optional[OpFeatures]]:
    """``features_for_division`` over a buffer's whole candidate menu, index for
    index with ``divisions`` so a menu index selects its features directly."""
    return [features_for_division(op, cd) for cd in divisions]


def with_residency(features: OpFeatures, lx_names: AbstractSet[str]) -> OpFeatures:
    """``features`` with each argument's ``mem`` set from ``lx_names``.

    The cost model charges an LX-resident argument no HBM traffic, so this is
    what turns a placement decision into a cost. Returns a new object; the input
    is left alone so one extracted menu can be scored against many candidate
    placements.
    """
    return dataclasses.replace(
        features,
        args=[
            dataclasses.replace(a, mem=("lx" if a.name in lx_names else "hbm"))
            for a in features.args
        ],
    )
