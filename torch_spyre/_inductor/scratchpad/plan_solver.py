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


from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
import math

from torch_spyre._inductor.logging_utils import get_inductor_logger

logger = get_inductor_logger("scratchpad.plan_solver")


@dataclass
class LifetimeBoundBuffer:
    """
    Defines the data fields required for a plan solver.

    ``uses`` is the sorted list of operation indices at which the buffer is
    accessed (as returned by ``calculate_liveness``).  It must be non-empty:
    the ``start_time``/``end_time`` properties index into it and the
    FirstFit/BestFit scoring divides by ``len(uses)``, so callers must only
    construct buffers for names that are actually used.  ``first_use_is_read``
    is True for graph inputs (all accesses are reads) and False for computed
    buffers (first access is a write, all subsequent accesses are reads).

    ``start_time`` and ``end_time`` are convenience properties derived from
    ``uses``: ``uses[0]`` and ``uses[-1] + 1`` respectively.
    """

    name: str
    size: int
    uses: list[int]
    first_use_is_read: bool = False
    address: Optional[int] = None
    in_place_parents: list[str] = field(default_factory=list)

    @property
    def start_time(self) -> int:
        return self.uses[0]

    @property
    def end_time(self) -> int:
        return self.uses[-1] + 1


def _assert_in_place_relationships(buffers: list["LifetimeBoundBuffer"]) -> None:
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


class MemoryPlanSolver(ABC):
    """
    An abstract class for defining algorithms which solve
    memory layout patterns based on provided sizes, lifetimes.
    """

    def __init__(self, size: int, alignment: int = 128):
        """Initialize the solver with a fixed scratchpad capacity and alignment.

        Args:
            size (int): Total scratchpad size in bytes. Buffers whose aligned
                placement would exceed this limit are evicted (address=None).
            alignment (int): Byte alignment boundary. Every buffer is placed at
                the next address that is a multiple of this value. Defaults to 128
                (one Spyre stick).
        """
        self.limit = size
        self.alignment = alignment

    @abstractmethod
    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer], log_lx_usage: bool = False
    ) -> list[LifetimeBoundBuffer]:
        """
        Utilizes an implementation defined algorithm to determine
        if and where buffers should be placed in scratchpad memory based
        on their attributes.

        Args:
            buffers (list[LifetimeBoundBuffer]): The set of candidate buffers for memory planning
            log_lx_usage (bool): If True, emit per-timestep scratchpad usage at DEBUG level.

        Returns:
            list[LifetimeBoundBuffer]: The set of buffers with their placements defined.
        """
        pass


class GreedyLayoutSolver(MemoryPlanSolver):
    def __init__(self, size: int, alignment: int = 128):
        super().__init__(size, alignment)
        # `usage` tracks live placements during planning. It is specific to the
        # greedy time-stepping algorithm; the gap-based solvers don't use it.
        self.usage: list[LifetimeBoundBuffer] = []

    def _get_lowest_addr_in_use(self):
        return min(
            (rec.address for rec in self.usage if rec.address is not None),
            default=0,
        )

    def _get_highest_addr_in_use(self):
        return max(
            (rec.address + rec.size for rec in self.usage if rec.address is not None),
            default=0,
        )

    def _find_free_block(self, size_needed: int) -> Optional[int]:
        assert all(x.address is not None for x in self.usage)
        curr_lo = self._get_lowest_addr_in_use()
        curr_hi = self._get_highest_addr_in_use()
        if self.limit < size_needed:
            return None

        if not self.usage or curr_lo >= size_needed:
            return 0

        address = math.ceil(curr_hi / self.alignment) * self.alignment
        if address + size_needed <= self.limit:
            return address

        # Search for a gap between existing allocations
        self.usage.sort(key=lambda x: (x.address is None, x.address))
        for i in range(len(self.usage) - 1):
            assert (current_address := self.usage[i].address) is not None
            assert (next_address := self.usage[i + 1].address) is not None
            frag_st = (
                math.ceil((current_address + self.usage[i].size) / self.alignment)
                * self.alignment
            )
            if next_address - frag_st >= size_needed:
                return frag_st

        return None

    def _try_allocate(self, buffer: LifetimeBoundBuffer):
        # Check if the current buffer can be in-placed
        for in_place_opt in buffer.in_place_parents:
            matched_obj = next((u for u in self.usage if u.name == in_place_opt), None)
            if matched_obj is not None and buffer.size <= matched_obj.size:
                buffer.address = matched_obj.address
                self.usage.append(buffer)
                self.usage.remove(matched_obj)
                return None

        # Decide where to allocate the block from
        addr = self._find_free_block(buffer.size)

        # Push the allocation result to the buffer and the usage table
        if addr is not None:
            buffer.address = addr
            self.usage.append(buffer)
        else:
            buffer.address = None

    def _try_deallocate(self, bufs: list[LifetimeBoundBuffer] | LifetimeBoundBuffer):
        if isinstance(bufs, LifetimeBoundBuffer):
            bufs = [bufs]

        for buf in bufs:
            if buf in self.usage:
                self.usage.remove(buf)

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer], log_lx_usage: bool = False
    ) -> list[LifetimeBoundBuffer]:
        """Allocates addresses to the provided buffer list

        Accepts a set of buffers with pre-defined sizes and lifetimes. These buffers are
        allocated addresses with 0 -> `limit` where the maximum starting address of
        buffers are at most `self.limit` - `LifetimeBoundBuffer.size` - 1. The algorithm
        increments through logical time where time increments 1 unit for each
        step in a computation graph. At each step the lifetimes of all buffers are
        evaluated for allocation and deallocation based on its lifetime relative
        to the time being evaluated. As an optimization, times where no buffers
        enter or exit scope are not evaluated.

        When a buffer enters scope, the current usage is evaluated in the following
        manner:
            1. Check if there is a permissible in-place buffer already allocated
            2. Is there enough space from address 0 -> first usage.
            3. Is there enough space for the current buffer from the max address
                to the maximum memory address. Allocate as current_max + 1 + alignment.
            4. Is there space between allocations. Check for gaps between current
                allocations and find where gaps exceed current size. Allocate if
                current gap is larger than current size + alignment.

        Args:
            buffers (list[LifetimeBoundBuffer]): The set of buffers to be planned.

        Returns:
            list[LifetimeBoundBuffer]: The supplied buffers with addresses assigned.
        """
        if not buffers:
            return []
        assert all(buf.address is None for buf in buffers), (
            "Buffers cannot be previously or partially planned"
        )
        _assert_in_place_relationships(buffers)

        self.usage = []

        # Walk through all transition points in chronological order.
        # Include end_time + 1 so deallocation fires even when no other
        # buffer starts or ends at that tick.
        times = set()
        for b in buffers:
            times.add(b.start_time)
            times.add(b.end_time)
        sorted_times = sorted(times)

        for idx in sorted_times:
            # Deallocate all expired buffers before allocating new ones so that
            # freed slots are immediately available at the same time step.
            for buffer in buffers:
                if idx == buffer.end_time:
                    self._try_deallocate(buffer)

            for buffer in buffers:
                if idx == buffer.start_time:
                    self._try_allocate(buffer)

        if log_lx_usage and logger.isEnabledFor(10):  # logging.DEBUG
            logger.debug("scratchpad limit: %d KB", self.limit // 1024)
            for idx in range(sorted_times[0], sorted_times[-1]):
                live = []
                used = 0
                for b in buffers:
                    if b.address is not None and b.start_time <= idx < b.end_time:
                        live.append(f"{b.name}_{b.size // 1024}KB@{hex(b.address)}")
                        used += b.size
                logger.debug("t=%d: %d KB  [%s]", idx, used // 1024, ", ".join(live))

        return buffers
