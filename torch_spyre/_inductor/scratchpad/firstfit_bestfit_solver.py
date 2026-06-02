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

import heapq
from dataclasses import dataclass, field, replace
from typing import Optional, Callable

from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)


def round_up_to_alignment(arg: int, alignment: int) -> int:
    return ((arg + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class Gap:
    start: int
    end: int
    in_place_parents: list[str] = field(default_factory=list)


def _assert_in_place_relationships(buffers: list[LifetimeBoundBuffer]) -> None:
    """Assert that all declared in-place parent/child pairs satisfy required invariants."""
    buf_by_name = {b.name: b for b in buffers}
    for child in buffers:
        for parent_name in child.in_place_parents:
            parent = buf_by_name[parent_name]
            assert parent.end_time == child.start_time + 1, (
                f"In-place parent {parent_name}.end_time={parent.end_time} must equal "
                f"child {child.name}.start_time+1={child.start_time + 1}"
            )
            assert child.size <= parent.size, (
                f"In-place child {child.name}.size={child.size} "
                f"must be <= parent {parent_name}.size={parent.size}"
            )


def _topological_sort(
    buffers: list[LifetimeBoundBuffer],
    f: Callable[[LifetimeBoundBuffer], int] = lambda b: 0,
) -> list[LifetimeBoundBuffer]:
    """Topological sort via Kahn's algorithm; ties broken by (f, original_index)."""
    name_to_idx = {b.name: i for i, b in enumerate(buffers)}
    in_degree = [0] * len(buffers)
    children: list[list[int]] = [[] for _ in buffers]

    for i, child in enumerate(buffers):
        for parent_name in child.in_place_parents:
            p = name_to_idx[parent_name]
            children[p].append(i)
            in_degree[i] += 1

    heap: list[tuple[int, int]] = []
    for i, buf in enumerate(buffers):
        if in_degree[i] == 0:
            heapq.heappush(heap, (f(buf), i))

    result: list[LifetimeBoundBuffer] = []
    while heap:
        _, i = heapq.heappop(heap)
        result.append(buffers[i])
        for j in children[i]:
            in_degree[j] -= 1
            if in_degree[j] == 0:
                buf = buffers[j]
                heapq.heappush(heap, (buf.end_time - buf.start_time, j))

    assert len(result) == len(buffers), (
        "Cycle detected in in-place parent relationships"
    )
    return result


class FirstFitLayoutSolver(MemoryPlanSolver):
    """Allocates buffers shortest-lifetime-first, placing each in the first gap that fits.

    Buffers are sorted topologically (parents before children) with ties broken by ascending
    lifetime, then placed one at a time. For each buffer, free address gaps during its lifetime
    are computed; the buffer is placed at the start of the first gap large enough to hold it
    (rounded up to alignment). In-place reuse is attempted first: if a declared parent has already
    been placed and its address falls within a free gap, the child inherits that address.
    Buffers that cannot fit within self.limit are evicted (address=None).
    """

    def _all_minus(
        self,
        gaps: list[Gap],
        interval: tuple[int, int],
        minimum_size: int,
    ) -> list[Gap]:
        """Return gaps with interval subtracted, dropping remainders < minimum_size."""
        result = []
        for gap in gaps:
            a, b = gap.start, gap.end
            if a < interval[0]:
                if b < interval[0]:
                    if b - a >= minimum_size:
                        result.append(Gap(a, b))
                else:
                    if interval[0] - a >= minimum_size:
                        result.append(Gap(a, interval[0]))
            if b > interval[1]:
                if a > interval[1]:
                    if b - a >= minimum_size:
                        result.append(Gap(a, b))
                else:
                    if b - interval[1] >= minimum_size:
                        result.append(Gap(interval[1], b))
        return result

    def _build_gaps(
        self,
        buffer: LifetimeBoundBuffer,
        placed: list[LifetimeBoundBuffer],
    ) -> list[Gap]:
        """Build free gaps for buffer, annotated with valid in-place parent addresses.

        Pass 1: subtract address intervals of all already-placed buffers that overlap
        buffer's lifetime, except declared in-place parents (their slots are candidates
        for reuse, not conflicts).
        Pass 2: for each remaining gap, record which declared parents fit entirely within it.
        """
        gaps: list[Gap] = [Gap(0, self.limit)]
        parent_names = set(buffer.in_place_parents)

        for other in placed:
            if other.address is None:
                continue
            if other.name in parent_names:
                continue
            if not (
                other.start_time < buffer.end_time
                and buffer.start_time < other.end_time
            ):
                continue
            gaps = self._all_minus(
                gaps, (other.address, other.address + other.size), buffer.size
            )

        placed_by_name = {b.name: b for b in placed}
        for i, gap in enumerate(gaps):
            new_parents = list(gap.in_place_parents)
            for parent_name in parent_names:
                parent = placed_by_name.get(parent_name)
                if parent is None or parent.address is None:
                    continue
                addr_p = parent.address
                if gap.start <= addr_p and addr_p + parent.size <= gap.end:
                    new_parents.append(parent_name)
            gaps[i] = replace(gap, in_place_parents=new_parents)

        return gaps

    def _pick_gap(
        self,
        gaps: list[Gap],
        size: int,
    ) -> Optional[Gap]:
        """Return the first fitting gap, or None."""
        for gap in gaps:
            addr = round_up_to_alignment(gap.start, self.alignment)
            if addr + size <= gap.end:
                return gap
        return None

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        if not buffers:
            return []
        assert all(buf.address is None for buf in buffers), (
            "Buffers cannot be previously or partially planned"
        )
        _assert_in_place_relationships(buffers)

        buffers_filtered = [
            buffer for buffer in buffers if buffer.end_time >= buffer.start_time + 1
        ]
        buffers_sorted = _topological_sort(
            buffers_filtered, lambda b: b.end_time - b.start_time
        )

        names_to_addresses: dict[str, int] = {}
        for i, buffer in enumerate(buffers_sorted):
            placed = buffers_sorted[:i]
            gaps = self._build_gaps(buffer, placed)
            gap = self._pick_gap(
                [gap for gap in gaps if gap.in_place_parents],
                buffer.size,
            )
            if gap is not None:
                parent = gap.in_place_parents[0]
                buffer.address = names_to_addresses[parent]
                names_to_addresses[buffer.name] = buffer.address
            else:
                gap = self._pick_gap(gaps, buffer.size)
                if gap is not None:
                    buffer.address = round_up_to_alignment(gap.start, self.alignment)
                    names_to_addresses[buffer.name] = buffer.address

        return buffers


class BestFitLayoutSolver(FirstFitLayoutSolver):
    """Like FirstFitLayoutSolver but places each buffer in the tightest fitting gap.

    Inherits all logic from FirstFitLayoutSolver; only the gap-selection policy
    differs: instead of picking the first gap large enough to hold the buffer,
    this picks the gap that minimises leftover space after placement.
    """

    def _pick_gap(
        self,
        gaps: list[Gap],
        size: int,
    ) -> Optional[Gap]:
        """Return the tightest fitting gap, or None."""
        best_gap: Optional[Gap] = None
        best_waste: int = 0
        for gap in gaps:
            addr = round_up_to_alignment(gap.start, self.alignment)
            if addr + size <= gap.end:
                waste = gap.end - addr - size
                if best_gap is None or waste < best_waste:
                    best_gap = gap
                    best_waste = waste
        return best_gap
