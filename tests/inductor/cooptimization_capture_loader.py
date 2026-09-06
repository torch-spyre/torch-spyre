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

"""Rehydrate captured co-optimizer dumps into **real** substrate objects.

The fixture JSON is produced by running the real ``CoOptimizingAllocator`` over
softmax/mlp/swiglu/sdpa/... and serializing the buffers handed to the solver
(candidate menus + ``cd_parent_matches`` + placement/cost fields) plus the
solver's chosen division + address per buffer.

Two capture fields predate the landed data model and are mapped on load:

``placement``
    Redundant with ``residency_reason`` -- the two are perfectly in sync across
    all 308 captured buffers (``placement == (residency_reason is None)``), and
    only ``residency_reason`` exists on the real class, so ``placement`` is
    dropped.

``boundary_cost`` / ``spill_write_cost``
    The real class carries a ``boundary: BufferType`` classification instead, and
    each engine derives its own cost from it. In every capture
    ``spill_write_cost == size`` (a full producer write) and ``boundary_cost`` is
    nonzero for exactly one buffer per graph -- the graph output -- where it is
    also exactly ``size`` (the unavoidable write-out). So ``boundary_cost > 0``
    identifies the output and the mapping below is exact rather than heuristic.
    See ``test_capture_boundary_mapping_is_exact``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType,
    CoreDivision,
    CoreDivisionBuffer,
)

DEFAULT_CAPTURE_PATH = Path(__file__).parent / "cooptimization_captures.json"
LARGE_CAPTURE_PATH = Path(__file__).parent / "cooptimization_captures_large.json"


@dataclass
class CapturedGraph:
    """One captured graph: the solver INPUT (buffers) + the SOLVED reference."""

    buffers: list[CoreDivisionBuffer]
    # name -> {"chosen_division": int|None, "address": int|None, "resident": bool}
    solved: dict[str, dict] = field(default_factory=dict)

    def by_name(self) -> dict[str, CoreDivisionBuffer]:
        return {b.name: b for b in self.buffers}


def _cd(d: dict) -> CoreDivision:
    # Keys serialize as strings; the real substrate keys them by int stride.
    return CoreDivision(
        output_splits={int(k): v for k, v in d["output_splits"].items()},
        reduction_splits={int(k): v for k, v in d["reduction_splits"].items()},
    )


def _buf(d: dict) -> CoreDivisionBuffer:
    # A nonzero boundary_cost marks the graph output, whose write-out happens
    # whether or not it is resident; everything else is an intermediate whose
    # producer write residency turns into a free LX write. That distinction is
    # exactly what ``BufferType`` encodes for the landed spill-cost formula.
    boundary = BufferType.Output if d["boundary_cost"] else BufferType.Intermediate
    return CoreDivisionBuffer(
        name=d["name"],
        size=d["size"],
        uses=list(d["uses"]),
        first_use_is_read=d["first_use_is_read"],
        in_place_parents=list(d["in_place_parents"]),
        residency_reason=d["residency_reason"],
        core_divisions=[_cd(cd) for cd in d["core_divisions"]],
        parents=list(d["parents"]),
        cd_parent_matches={
            p: [tuple(pair) for pair in pairs]
            for p, pairs in d["cd_parent_matches"].items()
        },
        boundary=boundary,
    )


def load_captures(
    path: str | Path = DEFAULT_CAPTURE_PATH,
) -> dict[str, list[CapturedGraph]]:
    """Parse the capture JSON -> ``{case_name: [CapturedGraph, ...]}``."""
    with open(path) as f:
        raw = json.load(f)
    out: dict[str, list[CapturedGraph]] = {}
    for case, graphs in raw.items():
        out[case] = [
            CapturedGraph(
                buffers=[_buf(b) for b in g["inputs"]],
                solved={s["name"]: s for s in g["solved"]},
            )
            for g in graphs
        ]
    return out


# The capture pins the committed/legacy division at ``core_divisions[0]`` (the
# allocator's fixed-division seed), so the SA seed is ``chosen_division = 0`` for
# every buffer, with pi from a FirstFit pass over the sizes.
SEED_DIVISION_INDEX = 0


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CAPTURE_PATH
    cases = load_captures(path)
    print(f"{'case':8} {'graphs':>6} {'buffers':>7} {'pinned':>6} {'resident*':>9}")
    for case, graphs in cases.items():
        for gi, g in enumerate(graphs):
            pinned = sum(1 for b in g.buffers if b.residency_reason is not None)
            resident = sum(1 for s in g.solved.values() if s["resident"])
            assert set(g.solved) <= set(g.by_name()), (
                f"{case}[{gi}] solved/name mismatch"
            )
            tag = f"{case}[{gi}]" if len(graphs) > 1 else case
            print(
                f"{tag:8} {len(graphs):>6} {len(g.buffers):>7} {pinned:>6} {resident:>9}"
            )
    print("(*resident = solved reference; the SA engine must re-derive it)")
