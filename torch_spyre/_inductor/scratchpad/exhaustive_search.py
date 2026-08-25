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

import math
import time
from collections.abc import Sequence
from typing import Callable, Optional, cast

from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivisionBuffer,
    CoreDivisionLayoutSolver,
    LifetimeBoundBuffer,
    MemoryPlanSolver,
    ceil_div,
)
from torch_spyre._inductor.logging_utils import get_inductor_logger

__all__ = ["ExhaustiveSearchSolver"]

logger = get_inductor_logger("scratchpad.exhaustive_search")


class ExhaustiveSearchSolver(CoreDivisionLayoutSolver):
    """DFS-based joint core-division + placement solver.

    Wraps a placement-only :class:`MemoryPlanSolver` factory and exhaustively
    searches the candidate-division cross-product, scoring each leaf by
    building a fresh inner solver (a solver is single-use) over per-core
    sizes derived from the chosen divisions and calling its
    :meth:`plan_layout`.  The combination that minimises total HBM bytes
    wins; the winning ``chosen_division`` is committed to every buffer, then
    the inner solver is built and called once more for the final placement.

    Bounded by ≤ K^N leaves where N counts buffers with >1 division candidate
    (most carry only the upstream-committed seed).  Per-leaf cost is one
    :meth:`plan_layout` call; ``cd_parent_matches`` (precomputed by
    :class:`CoOptimizingAllocator`) replaces the graph-level ``get_ncores`` check
    that was previously rerun per leaf on graph mutations.
    """

    def __init__(
        self,
        buffers: Sequence[LifetimeBoundBuffer],
        size: int,
        inner_factory: Callable[[Sequence[LifetimeBoundBuffer], int], MemoryPlanSolver],
        alignment: int = 128,
    ):
        super().__init__(buffers, size, alignment)
        self._inner_factory = inner_factory

    # ------------------------------------------------------------------
    # MemoryPlanSolver contract
    # ------------------------------------------------------------------

    def plan_layout(self, log_lx_usage: bool = False) -> list[LifetimeBoundBuffer]:
        # Dispatch is per buffer, not per class (mirrors CpSatLayoutSolver): a
        # buffer with no candidate divisions to choose among places here just
        # as well as through the joint entry point.
        return cast(
            "list[LifetimeBoundBuffer]",
            self.plan_layout_and_core_divisions(),
        )

    # ------------------------------------------------------------------
    # CoreDivisionLayoutSolver contract
    # ------------------------------------------------------------------

    def plan_layout_and_core_divisions(self) -> list[CoreDivisionBuffer]:
        buffers_list = cast("list[CoreDivisionBuffer]", self.buffers)

        buf_by_name: dict[str, CoreDivisionBuffer] = {b.name: b for b in buffers_list}

        # CoreDivisionBuffer.size is the total device footprint (pre-division); that
        # is exactly the HBM cost when a buffer is not pinned to LX.
        buf_total_bytes: dict[str, int] = {b.name: b.size for b in buffers_list}

        # Per-buffer chosen-division index, mutated in-place during DFS.
        chosen: dict[str, int] = {b.name: 0 for b in buffers_list}
        best_total: float = math.inf
        best_chosen: dict[str, int] = dict(chosen)

        # Only buffers with >1 candidate are search variables; single-candidate
        # buffers stay at index 0 throughout.
        variable_buffers = [b for b in buffers_list if len(b.core_divisions) > 1]

        def _residency_reason(b: CoreDivisionBuffer, ci: int) -> Optional[str]:
            """Residency verdict for ``b`` under division ``ci``.

            Starts from the pre-computed ``b.residency_reason`` (computed by
            :class:`CoOptimizingAllocator` with ``division_is_fixed=False``), then
            adds the division-compatibility check via ``cd_parent_matches``: if no
            (parent_div, ci) pair in ``cd_parent_matches[parent]`` matches the
            parent's currently-chosen division, the per-core views disagree and the
            buffer must spill.
            """
            if b.residency_reason is not None:
                return b.residency_reason
            for p_name in b.parents:
                if p_name not in buf_by_name:
                    continue
                pi = chosen[p_name]
                pairs = b.cd_parent_matches.get(p_name, [])
                if not any(pp == pi and pc == ci for pp, pc in pairs):
                    return "core div mismatch"
            return None

        def _valid_inplace_parents(b: CoreDivisionBuffer, ci: int) -> list[str]:
            """In-place parents whose per-core sizes are compatible with division ``ci``.

            ``_check_in_place_relationships`` requires ``child.size <= parent.size``
            for plain :class:`LifetimeBoundBuffer` pairs (no ``core_divisions``).
            Only include parents whose per-core size is >= the child's under the
            respective chosen divisions so the assertion in the inner solver holds.
            """
            b_per_core = ceil_div(b.size, b.core_divisions[ci].output_partition)
            valid = []
            for p_name in b.in_place_parents:
                p = buf_by_name.get(p_name)
                if p is None:
                    valid.append(p_name)
                    continue
                pi = chosen[p_name]
                p_per_core = ceil_div(p.size, p.core_divisions[pi].output_partition)
                if b_per_core <= p_per_core:
                    valid.append(p_name)
            return valid

        def _make_temp_bufs() -> list[LifetimeBoundBuffer]:
            return [
                LifetimeBoundBuffer(
                    name=b.name,
                    size=ceil_div(
                        b.size, b.core_divisions[chosen[b.name]].output_partition
                    ),
                    uses=b.uses,
                    first_use_is_read=b.first_use_is_read,
                    in_place_parents=_valid_inplace_parents(b, chosen[b.name]),
                    residency_reason=_residency_reason(b, chosen[b.name]),
                )
                for b in buffers_list
            ]

        def score_leaf() -> int:
            # A solver is single-use, so score each leaf with a fresh one.
            solver = self._inner_factory(_make_temp_bufs(), self.limit)
            allocation = solver.plan_layout()
            pinned = {a.name for a in allocation if a.address is not None}
            return sum(v for k, v in buf_total_bytes.items() if k not in pinned)

        def recurse(var_idx: int) -> None:
            nonlocal best_total, best_chosen
            if var_idx == len(variable_buffers):
                hbm = score_leaf()
                if hbm < best_total:
                    best_total = hbm
                    best_chosen = dict(chosen)
                return
            b = variable_buffers[var_idx]
            prev = chosen[b.name]
            for opt_idx in range(len(b.core_divisions)):
                chosen[b.name] = opt_idx
                recurse(var_idx + 1)
            chosen[b.name] = prev

        t1 = time.perf_counter()
        recurse(0)
        t_search = time.perf_counter() - t1

        n_paths = math.prod(len(b.core_divisions) for b in buffers_list)
        winner = {
            b.name: b.core_divisions[best_chosen[b.name]].label
            for b in variable_buffers
            if best_chosen[b.name] != 0
        }
        logger.info(
            "co-opt search: %d paths in %.1fms; winner=%s",
            n_paths,
            t_search * 1e3,
            winner,
        )

        # Commit winning choices and run final placement.
        for b in buffers_list:
            b.chosen_division = best_chosen[b.name]

        chosen = best_chosen
        final_solver = self._inner_factory(_make_temp_bufs(), self.limit)
        final_alloc = final_solver.plan_layout()
        addr_by_name = {a.name: a.address for a in final_alloc}
        for b in buffers_list:
            b.address = addr_by_name.get(b.name)

        self.spill_reasons = dict(final_solver.spill_reasons)
        return buffers_list
