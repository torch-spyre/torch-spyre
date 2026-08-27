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
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
import math
from torch_spyre._inductor.logging_utils import get_inductor_logger
from enum import Enum

if TYPE_CHECKING:
    from torch_spyre._inductor.scratchpad.lx_relayout import LXRelayoutPlan

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

    ``uses`` is the strictly increasing list of operation indices at which the
    buffer is accessed (as returned by ``calculate_liveness``).  It is normally
    non-empty, and callers that read ``start_time``/``end_time`` require that,
    since those properties index into it; the FirstFit/BestFit scoring divides
    by ``len(uses)`` plus a write bonus, which is non-zero for a computed buffer
    even when ``uses`` is empty.  Emptiness is nevertheless allowed at
    construction -- see :meth:`__post_init__` for the registration state that
    needs it.  ``first_use_is_read`` is True for graph inputs (all accesses are
    reads) and False for computed buffers (first access is a write, all
    subsequent accesses are reads).

    Both properties of ``uses`` are asserted in ``__post_init__``.  Strictness
    is what makes ``read_count`` trustworthy: one entry per accessing op means
    that for a computed buffer ``read_count == 0`` is exactly "written, never
    read", which the in-place invariants rely on (see
    :func:`check_in_place_parent_is_read`).  A repeated index would describe a
    buffer written and read by the same op, i.e. with a single live tick, and
    would let such a buffer pass as an in-place parent.

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
    # Optional exclusive lifetime end for storage reused by a counted loop.
    # Keep this separate from ``uses``: it changes address overlap, but must not
    # manufacture a read or inflate residency/spill benefit.
    lifetime_end_override: Optional[int] = None
    # Buffers that must be placed atomically with this one. Despite the name,
    # this is one-to-many: only the group root carries the complete partner list.
    paired_with: list["LifetimeBoundBuffer"] = field(
        default_factory=list, repr=False, compare=False
    )
    # LX relayout plans for which this buffer is the source.
    lx_relayout_plans: list["LXRelayoutPlan"] = field(
        default_factory=list, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Not also asserted non-empty: buffers are sometimes registered before
        # their uses are known and filled in afterwards (see
        # ``make_buffer_registry`` in tests/inductor/test_scratchpad_patterns.py),
        # and an empty list is vacuously strictly increasing. Callers that read
        # ``start_time``/``end_time`` still require a non-empty list.
        #
        # This runs at construction only, so it does not see later mutation of
        # ``uses``; the in-place invariants therefore test the property they need
        # directly rather than inferring it from ``read_count``.
        assert all(a < b for a, b in zip(self.uses, self.uses[1:])), (
            f"buffer {self.name} has uses={self.uses}, which is not strictly "
            "increasing; uses carries one distinct index per accessing operation"
        )
        if self.lifetime_end_override is not None and self.uses:
            assert self.lifetime_end_override >= self.uses[-1] + 1, (
                f"buffer {self.name} has lifetime_end_override="
                f"{self.lifetime_end_override} before nominal exclusive end "
                f"{self.uses[-1] + 1}"
            )

    @property
    def read_count(self) -> int:
        """Number of reads.  For a computed buffer the first use is the producing
        write, so every use but that one is a read; when ``first_use_is_read``
        (a graph input) every use is a read.  Exact because ``uses`` holds one
        distinct index per accessing op.

        This counts the buffer's reads, not the reads residency would save: an
        input's first read is the clone-in that pinning cannot avoid, so a cost
        model has to discount it separately (see
        :meth:`_LifetimeBufferWithCpVars.spill_cost`).  The ``max`` only guards
        the transient empty-``uses`` state described in :meth:`__post_init__`.
        """
        return max(0, len(self.uses) - (0 if self.first_use_is_read else 1))

    @property
    def start_time(self) -> int:
        return self.uses[0]

    @property
    def end_time(self) -> int:
        nominal = self.uses[-1] + 1
        return max(nominal, self.lifetime_end_override or nominal)

    @property
    def min_footprint(self) -> int:
        """Smallest LX footprint the buffer can take, for the capacity check"""
        return self.size

    def overlaps_in_time(self, other: "LifetimeBoundBuffer") -> bool:
        """Returns true iff self and other overlap in time."""
        return self.start_time < other.end_time and other.start_time < self.end_time


@dataclass(frozen=True)
class TileAxis:
    """One coarse-tiling level.

    ``host_dim`` is a *positional* index: into ``op_out_coords(op)`` for an
    output axis, or into the op's ordered reduction loop variables (see
    :func:`wsr.coarse_tile.reduction_loop_vars`) for a reduction axis.
    ``is_reduction`` selects which frame ``host_dim`` indexes. ``count`` is the
    split factor -- how many equal tiles the axis is cut into.
    """

    host_dim: int
    count: int
    is_reduction: bool = False


@dataclass(frozen=True)
class TileSpec:
    """An ordered, outermost-first tuple of :class:`TileAxis` levels.

    Ordered -- where the core-division splits are dicts and so order-free --
    because tile levels *nest*: swapping two levels is a different plan. Frozen
    and hashable so ``==`` is exactly the "same tiling shape" test the group
    derivation keys on. The empty spec is *untiled*, and is the inert default
    every :class:`CoreDivision` carries while ``auto_coarse_tiling`` is off.
    """

    axes: tuple[TileAxis, ...] = ()

    @property
    def is_untiled(self) -> bool:
        return not self.axes

    @property
    def depth(self) -> int:
        """Number of nested tile levels."""
        return len(self.axes)

    @property
    def tile_count(self) -> int:
        """Total number of loop tiles across every level (all axes)."""
        return math.prod(a.count for a in self.axes)

    @property
    def output_tile_count(self) -> int:
        """Product of the split factors over output (non-reduction) axes only.

        This is the factor by which a tiled op's own per-tile scratch shrinks,
        and so the factor :attr:`CoreDivisionBuffer.min_footprint` divides by. A
        reduction-tiled level does *not* shrink that buffer -- the op's own
        output is the accumulator, which keeps the full output extent -- so
        reduction axes are excluded here even though they count in
        :attr:`tile_count`.
        """
        return math.prod(a.count for a in self.axes if not a.is_reduction)

    @property
    def is_clean(self) -> bool:
        """True when no reduction axis is tiled (mirrors
        :attr:`CoreDivision.is_clean`)."""
        return not any(a.is_reduction for a in self.axes)

    @property
    def label(self) -> str:
        if not self.axes:
            return "untiled"
        return "/".join(
            f"{'~' if a.is_reduction else ''}d{a.host_dim}:{a.count}" for a in self.axes
        )


@dataclass
class CoreDivision:
    """One permissible core-division of a buffer's producing op.

    ``output_splits`` / ``reduction_splits`` are the stride/coeff-keyed encoding
    produced by :func:`pass_utils.splits_by_index_coeff` -- exactly the shape
    stored in ``op.op_it_space_splits``. Solvers are expected to use these to size
    the buffer (per-core footprint = total / ``output_partition``).

    ``tiling`` pairs a coarse tiling onto this division as *one* candidate rather
    than a parallel list. A division is only meaningful relative to a tiling
    (the legal set and the coeff encoding both move with it), so the two must be
    chosen together. The empty :class:`TileSpec` is untiled and inert.
    """

    output_splits: dict[int, int] = field(default_factory=dict)
    reduction_splits: dict[int, int] = field(default_factory=dict)
    tiling: TileSpec = field(default_factory=TileSpec)

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
        (the placement-only case ``_wrap`` also dispatches on).

        A tiled candidate's own buffer is per-tile scratch, so its footprint
        shrinks by the output tile count as well as the core count -- this is
        the LX-residency win entering the footprint math. Reduction tile levels
        are excluded (see :attr:`TileSpec.output_tile_count`); with
        ``auto_coarse_tiling`` off every ``cd.tiling`` is empty and this reduces
        to the previous ``ceil_div(size, output_partition)`` exactly."""
        if not self.core_divisions:
            return self.size
        return min(
            ceil_div(self.size, cd.output_partition * cd.tiling.output_tile_count)
            for cd in self.core_divisions
        )


def check_in_place_parent_is_read(
    parent: "LifetimeBoundBuffer", child_name: str
) -> None:
    """Reject an in-place parent whose storage is never read before handover.

    The child takes the parent's storage over at the parent's last use, so that
    use has to be a read. For a computed buffer the first use is the write, so a
    single use means the buffer is written and never read -- handing it to a
    child would overwrite data nothing ever consumed, and it would make parent
    and child come alive on the same tick while sharing storage. Graph inputs are
    exempt: all their uses are reads, so one use is enough.

    Split out of :func:`_check_in_place_relationships` because the
    permutation-based layout solvers call it directly, where they resolve
    declared pairs (``_compute_inplace_partners``), alongside their own copy of
    the abutment check -- their incremental machinery samples the contact
    profiles at the single tick the pair overlaps and so cannot re-derive a
    longer overlap. The size invariant is the one they do *not* take as a
    precondition: an oversized child is simply not placed in-place (see
    ``_can_inplace``), so checking it here would reject inputs they handle
    correctly.
    """
    # Tested as "a use strictly after the first" rather than via ``read_count``:
    # the two agree whenever ``uses`` is strictly increasing, but ``uses`` is
    # validated at construction and can be mutated afterwards, and this way a
    # repeated index cannot pass as a read.
    has_read_after_write = len(parent.uses) > 1 and parent.uses[-1] > parent.uses[0]
    if not (parent.first_use_is_read or has_read_after_write):
        raise ValueError(
            f"In-place parent {parent.name} is a computed buffer that is never "
            f"read (uses={parent.uses}), so it cannot hand its storage to child "
            f"{child_name}"
        )


def _check_in_place_relationships(
    buffers: Sequence["LifetimeBoundBuffer"],
) -> None:
    """Reject any declared in-place pair that violates a required invariant."""
    buf_by_name = {b.name: b for b in buffers}
    for child in buffers:
        for parent_name in child.in_place_parents:
            parent = buf_by_name.get(parent_name)
            if parent:
                if parent.end_time != child.start_time + 1:
                    raise ValueError(
                        f"In-place parent {parent_name}.end_time={parent.end_time} "
                        f"must equal child {child.name}.start_time+1="
                        f"{child.start_time + 1}"
                    )
                check_in_place_parent_is_read(parent, child.name)
                # With core_divisions ``size`` is the *total* footprint, so a static
                # size check doesn't apply; the per-core match is enforced against the
                # chosen division in ``CpSatLayoutSolver._add_inplace_relaxation``. Only
                # the division-fixed case (plain ``LifetimeBoundBuffer``, no
                # ``core_divisions``) keeps the static check.
                if not (
                    getattr(parent, "core_divisions", None)
                    or getattr(child, "core_divisions", None)
                ):
                    if child.size > parent.size:
                        raise ValueError(
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

    supports_paired_buffers = False

    def __init__(
        self, buffers: Sequence["LifetimeBoundBuffer"], size: int, alignment: int = 128
    ):
        """Initialize the solver with its buffers, a fixed scratchpad capacity,
        and alignment.

        ``buffers`` is a :class:`Sequence` (not ``list``) because ``Sequence`` is
        covariant in its element type: that lets a caller hand over a
        ``list[CoreDivisionBuffer]`` -- a subtype of ``LifetimeBoundBuffer`` -- and
        still type-check.

        Args:
            buffers (Sequence[LifetimeBoundBuffer]): The set of candidate buffers
                for memory planning. A solver instance is single-use: construct a
                fresh one for each buffer set to plan.
            size (int): Total scratchpad size in bytes. Buffers whose aligned
                placement would exceed this limit are evicted (address=None).
            alignment (int): Byte alignment boundary. Every buffer is placed at
                the next address that is a multiple of this value. Defaults to
                128 (one Spyre stick), which is also what every concrete solver
                defaults to.
        """
        self.buffers: list["LifetimeBoundBuffer"] = list(buffers)
        assert self.supports_paired_buffers or not any(
            buffer.paired_with for buffer in self.buffers
        ), f"{type(self).__name__} does not support paired-buffer placement"
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

    def record_exclusions(self) -> dict[str, str]:
        """Compute, store, and return the ``name -> reason`` map of every buffer
        in :attr:`buffers` barred from LX residency.

        This is the piece a solver that keeps barred buffers in its model (e.g.
        CP-SAT, which pins them non-resident rather than dropping them) needs on
        its own; :meth:`partition` layers the placeable/excluded split on top.
        The returned map is also stored in :attr:`spill_reasons`.
        """
        self.spill_reasons = {
            buffer.name: reason
            for buffer in self.buffers
            if (reason := self.excluded(buffer)) is not None
        }
        return self.spill_reasons

    def partition(
        self,
    ) -> tuple[list["LifetimeBoundBuffer"], list["LifetimeBoundBuffer"]]:
        """Split :attr:`buffers` into ``(placeable, excluded)``, recording every
        exclusion in :attr:`spill_reasons` via :meth:`record_exclusions`.
        """
        excluded_reasons = self.record_exclusions()
        placeable = [b for b in self.buffers if b.name not in excluded_reasons]
        excluded = [b for b in self.buffers if b.name in excluded_reasons]
        return placeable, excluded

    @abstractmethod
    def plan_layout(self, log_lx_usage: bool = False) -> list[LifetimeBoundBuffer]:
        """
        Utilizes an implementation defined algorithm to determine
        if and where :attr:`buffers` should be placed in scratchpad memory based
        on their attributes.

        Args:
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
    def plan_layout_and_core_divisions(self) -> list[CoreDivisionBuffer]:
        """Choose each buffer's core division and its LX placement together.

        On top of the :meth:`plan_layout` contract, implementations write the
        index of the chosen division back to ``chosen_division`` for the
        allocator to commit. Operates on :attr:`buffers`, each of which must
        carry its enumerated candidate core divisions.

        Returns:
            The same buffers, with placements and chosen divisions defined.
        """
