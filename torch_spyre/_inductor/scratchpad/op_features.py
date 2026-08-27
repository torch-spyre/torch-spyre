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

A co-optimizer needs features for every division in a buffer's candidate menu,
not just the committed division. ``CoreDivision`` stores sparse, symbol-keyed
output and reduction splits, so this module restores its complete symbol-keyed
map and passes it directly to :func:`dump_cost_model.extract_op_features`.
No candidate is encoded into ``op_it_space_splits``: that legacy
coefficient-keyed representation is reserved for the Scheduler boundary.

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
from torch_spyre._inductor.pass_utils import iteration_space_from_op

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision

logger = get_inductor_logger("scratchpad.op_features")


def _work_slices(op, division: "CoreDivision") -> dict:
    """Restore a complete symbol-keyed split map from a sparse candidate."""
    return {
        symbol: int(
            division.output_splits.get(symbol, division.reduction_splits.get(symbol, 1))
        )
        for symbol in iteration_space_from_op(op)
    }


def features_for_division(op, division: "CoreDivision") -> Optional[OpFeatures]:
    """``OpFeatures`` for ``op`` as if it were divided per ``division``.

    Returns ``None`` when the op cannot be featurized (the extractor is
    best-effort and swallows its own failures, so a ``None`` here means the op
    itself was rejected, not that the division was bad).

    The candidate is passed as a complete symbol-keyed map, leaving the live
    operation and its Scheduler-boundary transport untouched.
    """
    from torch_spyre._inductor.dump_cost_model import extract_op_features

    try:
        return extract_op_features(op, _work_slices(op, division))
    except Exception:  # noqa: BLE001 - featurization is best-effort by design
        logger.debug("could not featurize op for a candidate division", exc_info=True)
        return None


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
