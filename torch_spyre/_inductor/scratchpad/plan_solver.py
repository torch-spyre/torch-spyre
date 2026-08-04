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


from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
import math
from torch_spyre._inductor.logging_utils import get_inductor_logger
from enum import Enum

logger = get_inductor_logger("scratchpad.plan_solver")


class SolveError(Exception):
    """Raised when a solver is unable to find a solution"""


class BufferType(Enum):
    Intermediate = 0
    Input = 1
    Output = 2


def ceil_div(a: int, b: int) -> int:
    """Integer ceiling division. Used wherever a footprint is divided down by a
    core count, so every such site rounds identically (no float intermediate)."""
    return -(-a // b)


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
    # define the reason for excluding the buffer based on allocator
    # or solver logic paths.
    residency_reason: Optional[str] = None

    @property
    def read_count(self) -> int:
        """Reports the number of reads base on the number of uses."""
        return max(0, len(self.uses) - 1)

    @property
    def start_time(self) -> int:
        return self.uses[0]

    @property
    def end_time(self) -> int:
        return self.uses[-1] + 1

    @property
    def min_footprint(self) -> int:
        """Smallest LX footprint the buffer can take, for the capacity check"""
        return self.size

    def overlaps_in_time(self, other: "LifetimeBoundBuffer") -> bool:
        """Returns true iff self and other overlap in time."""
        return self.start_time < other.end_time and other.start_time < self.end_time


@dataclass
class CoreDivision:
    """One permissible core-division of a buffer's producing op.

    ``output_splits`` / ``reduction_splits`` are the stride/coeff-keyed encoding
    produced by :func:`pass_utils.splits_by_index_coeff` -- exactly the shape
    stored in ``op.op_it_space_splits``. Solvers are expected to use these to size
    the buffer (per-core footprint = total / ``output_partition``).
    """

    output_splits: dict[int, int] = field(default_factory=dict)
    reduction_splits: dict[int, int] = field(default_factory=dict)

    @property
    def cores_used(self) -> int:
        return math.prod(self.output_splits.values()) * math.prod(
            self.reduction_splits.values()
        )

    @property
    def is_clean(self) -> bool:
        """True when no reduction axis is split, so the output is fully sliced
        across cores (no per-core partial sums)."""
        return not self.reduction_splits

    @property
    def output_partition(self) -> int:
        """How many cores the output buffer is sliced across."""
        return math.prod(self.output_splits.values())

    def signature_key(self):
        """Per-core slicing signature, or ``None`` for a reduction-split division
        (a ``None`` never compares equal, so partial-reduction divisions never
        match)."""
        return tuple(sorted(self.output_splits.items())) if self.is_clean else None

    @property
    def label(self) -> str:
        out = ",".join(f"s{s}/{f}" for s, f in sorted(self.output_splits.items()))
        red = ",".join(f"~s{s}/{f}" for s, f in sorted(self.reduction_splits.items()))
        return " ".join(p for p in (out, red) if p) or "whole"


@dataclass
class CoreDivisionBuffer(LifetimeBoundBuffer):
    """A :class:`LifetimeBoundBuffer` carrying the joint core-division metadata

    The placement-only solvers (greedy/first-fit/best-fit) never look at these
    fields, so they stay on this subclass rather than the shared base.
    """

    core_divisions: list[CoreDivision] = field(default_factory=list)
    # Producer buffer names; defines the producer->consumer edges for matching.
    parents: list[str] = field(default_factory=list[str])
    # parent_buf_name -> (parent_div_idx, this_div_idx) pairs that induce the
    # *same per-core slicing of the parent*, precomputed by the allocator via
    # ``_per_core_view_on_buf`` (physical device-dim view equality, correct
    # across reductions/reshapes). These are the sole slicing-match predicate;
    # an absent/empty entry means no compatible division, so the gate forbids
    # the merge/residency across that edge.
    cd_parent_matches: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    chosen_division: Optional[int] = None
    boundary: BufferType = BufferType.Intermediate

    @property
    def min_footprint(self) -> int:
        """Smallest per-core footprint any candidate division allows. With no
        candidates there is nothing to divide by, so it falls back to ``size``
        (the placement-only case ``_wrap`` also dispatches on)."""
        if not self.core_divisions:
            return self.size
        return min(
            ceil_div(self.size, cd.output_partition) for cd in self.core_divisions
        )


def _assert_in_place_relationships(
    buffers: Sequence["LifetimeBoundBuffer"],
) -> None:
    """Assert that all declared in-place parent/child pairs satisfy required invariants."""
    buf_by_name = {b.name: b for b in buffers}
    for child in buffers:
        for parent_name in child.in_place_parents:
            parent = buf_by_name.get(parent_name)
            if parent:
                assert parent.end_time == child.start_time + 1, (
                    f"In-place parent {parent_name}.end_time={parent.end_time} must equal "
                    f"child {child.name}.start_time+1={child.start_time + 1}"
                )
                # With core_divisions ``size`` is the *total* footprint, so a static
                # size check doesn't apply; the per-core match is enforced against the
                # chosen division in ``CpSatLayoutSolver._add_inplace_relaxation``. Only
                # the division-fixed case (plain ``LifetimeBoundBuffer``, no
                # ``core_divisions``) keeps the static check.
                if not (
                    getattr(parent, "core_divisions", None)
                    or getattr(child, "core_divisions", None)
                ):
                    assert child.size <= parent.size, (
                        f"In-place child {child.name}.size={child.size} "
                        f"must be <= parent {parent_name}.size={parent.size}"
                    )


class MemoryPlanSolver(ABC):
    """Solves *placement*: where, if anywhere, each buffer lives in scratchpad.

    Every solver implements this. Each buffer's core division is already fixed
    by the time a placement-only solver sees it, so the buffer's ``size`` is the
    footprint to pack. :class:`CoreDivisionLayoutSolver` extends the contract for
    solvers that can also choose the division.
    """

    def __init__(self, size: int, alignment: int = 128):
        """Initialize the solver with a fixed scratchpad capacity and alignment.

        Args:
            size (int): Total scratchpad size in bytes. Buffers whose aligned
                placement would exceed this limit are evicted (address=None).
            alignment (int): Byte alignment boundary. Every buffer is placed at
                the next address that is a multiple of this value. Defaults to
                128 (one Spyre stick), which is also what every concrete solver
                defaults to.
        """
        self.limit = size
        self.alignment = alignment
        self.spill_reasons: dict[str, str] = {}

    def excluded(self, buffer: "LifetimeBoundBuffer") -> Optional[str]:
        """Why ``buffer`` may not reside in LX, or ``None`` if it may."""
        if buffer.residency_reason is not None:
            return buffer.residency_reason
        if buffer.min_footprint > self.limit:
            return (
                f"min footprint {buffer.min_footprint} B > LX capacity {self.limit} B"
            )
        return None

    def record_exclusions(
        self, buffers: Sequence["LifetimeBoundBuffer"]
    ) -> dict[str, str]:
        """Compute, store, and return the ``name -> reason`` map of every buffer
        barred from LX residency.

        This is the piece a solver that keeps barred buffers in its model (e.g.
        CP-SAT, which pins them non-resident rather than dropping them) needs on
        its own; :meth:`partition` layers the placeable/excluded split on top.
        The returned map is also stored in :attr:`spill_reasons`.
        """
        self.spill_reasons = {
            buffer.name: reason
            for buffer in buffers
            if (reason := self.excluded(buffer)) is not None
        }
        return self.spill_reasons

    def partition(
        self, buffers: Sequence["LifetimeBoundBuffer"]
    ) -> tuple[list["LifetimeBoundBuffer"], list["LifetimeBoundBuffer"]]:
        """Split ``buffers`` into ``(placeable, excluded)``, recording every
        exclusion in :attr:`spill_reasons` via :meth:`record_exclusions`.
        """
        excluded_reasons = self.record_exclusions(buffers)
        placeable = [b for b in buffers if b.name not in excluded_reasons]
        excluded = [b for b in buffers if b.name in excluded_reasons]
        return placeable, excluded

    @abstractmethod
    def plan_layout(
        self, buffers: Sequence[LifetimeBoundBuffer], log_lx_usage: bool = False
    ) -> list[LifetimeBoundBuffer]:
        """
        Utilizes an implementation defined algorithm to determine
        if and where buffers should be placed in scratchpad memory based
        on their attributes.

        ``buffers`` is a :class:`Sequence` (not ``list``) because ``Sequence`` is
        covariant in its element type: that lets a caller hand over a
        ``list[CoreDivisionBuffer]`` -- a subtype of ``LifetimeBoundBuffer`` -- and
        still type-check.

        Args:
            buffers (Sequence[LifetimeBoundBuffer]): The set of candidate buffers
                for memory planning
            log_lx_usage (bool): If True, emit per-timestep scratchpad usage at DEBUG level.

        Returns:
            list[LifetimeBoundBuffer]: The set of buffers with their placements defined.
        """


class CoreDivisionLayoutSolver(MemoryPlanSolver):
    """A solver that chooses each buffer's *core division* jointly with its
    placement, rather than accepting a division fixed upstream.

    The two decisions are coupled: the division sets the per-core footprint the
    placement has to fit, and residency requires a producer and its consumers to
    slice the shared buffer the same way. Solving them together lets a buffer
    take the division that lets it reside.

    Such a solver still satisfies :meth:`plan_layout` -- placement-only is the
    special case where there is nothing to choose.
    """

    @abstractmethod
    def plan_layout_and_core_divisions(
        self, buffers: Sequence[CoreDivisionBuffer]
    ) -> list[CoreDivisionBuffer]:
        """Choose each buffer's core division and its LX placement together.

        On top of the :meth:`plan_layout` contract, implementations write the
        index of the chosen division back to ``chosen_division`` for the
        allocator to commit.

        Args:
            buffers: Candidate buffers, each carrying its enumerated candidate
                core divisions.

        Returns:
            The same buffers, with placements and chosen divisions defined.
        """
