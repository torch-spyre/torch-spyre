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
    :func:`assert_in_place_parent_is_read`).  A repeated index would describe a
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
        return self.uses[-1] + 1

    @property
    def min_footprint(self) -> int:
        """Smallest LX footprint the buffer can take, for the capacity check"""
        return self.size

    def overlaps_in_time(self, other: "LifetimeBoundBuffer") -> bool:
        """Returns true iff self and other overlap in time."""
        return self.start_time < other.end_time and other.start_time < self.end_time


@dataclass(frozen=True)
class TileAxis:
    """One coarse-tile level: tile ``host_dim`` into ``count`` equal tiles.

    ``host_dim`` is a positional index into the op's output coordinates
    (``op_out_coords(op)``), or -- when ``is_reduction`` -- into the op's
    ordered reduction loop variables. That is the frame ``coarse_tile``
    ultimately consumes and the one ``op_it_space_splits`` already uses.
    """

    host_dim: int
    count: int
    is_reduction: bool = False

    def __post_init__(self):
        # A count of 1 is a no-op level (the hint path drops split_count==1
        # outright); the untiled plan is the *empty* TileSpec, never a
        # unit-count axis.
        assert self.count >= 2, f"TileAxis count must be >= 2, got {self.count}"
        assert self.host_dim >= 0, (
            f"TileAxis host_dim must be >= 0, got {self.host_dim}"
        )


@dataclass(frozen=True)
class TileSpec:
    """A coarse tiling as data: an ordered, outermost-first tuple of levels.

    Ordered, where core-division splits are dicts, because tile levels nest --
    swapping two levels is a different plan. Frozen and hashable so ``==`` is
    the "same tiling shape" test group derivation runs on. ``TileSpec()`` is
    the untiled plan; note *undecided* is expressed by
    ``chosen_division is None`` on the buffer, never by an empty spec.
    """

    axes: tuple[TileAxis, ...] = ()

    def __post_init__(self):
        # Accept any iterable of TileAxis but store a tuple, so instances
        # stay hashable regardless of how callers built the sequence.
        if not isinstance(self.axes, tuple):
            object.__setattr__(self, "axes", tuple(self.axes))

    @property
    def is_untiled(self) -> bool:
        return not self.axes

    @property
    def depth(self) -> int:
        """Nesting depth of the tile loop this spec describes."""
        return len(self.axes)

    @property
    def tile_count(self) -> int:
        """Total trip count of the tile-loop nest (1 when untiled)."""
        return math.prod(ax.count for ax in self.axes)

    @property
    def output_tile_count(self) -> int:
        """How many tiles the op's own output write is sliced into.

        Excludes reduction levels: tiling a reduction axis chunks the *input*
        walk while the per-tile output scratch (the accumulator) keeps the
        full output extent, so only output-axis counts shrink the footprint.
        """
        return math.prod(ax.count for ax in self.axes if not ax.is_reduction)

    @property
    def is_clean(self) -> bool:
        """True when no reduction axis is tiled (no partial-sum combining)."""
        return not any(ax.is_reduction for ax in self.axes)

    @property
    def label(self) -> str:
        return (
            " ".join(
                f"{'~' if ax.is_reduction else ''}d{ax.host_dim}/{ax.count}"
                for ax in self.axes
            )
            or "untiled"
        )


@dataclass
class CoreDivision:
    """One permissible core division of a buffer's producing op.

    ``output_splits`` / ``reduction_splits`` are the stride/coeff-keyed encoding
    produced by :func:`pass_utils.splits_by_index_coeff` -- exactly the shape
    stored in ``op.op_it_space_splits``. Solvers use these to size the buffer
    (per-core footprint = total / ``output_partition``).

    Carries no tiling. A division and a tiling are separate candidate lists
    related many-to-many (see :class:`TilingCandidate`), because the legal
    division set is tiling-relative: tiling shrinks a dim's extent, admitting
    fewer factors, while shrinking its per-core span, admitting more splits, so
    the per-spec sets are neither subsets nor supersets of one another.

    **One division may only be referenced by several tiling candidates once the
    encoding is tiling-invariant, which it is not yet.** ``splits_by_index_coeff``
    keys each symbol by its *coefficient in the write index*, and coarse tiling
    rewrites exactly those coefficients (``_divide_ranges`` / ``_rescale_index``),
    so the same dict denotes different physical splits under different tilings --
    the conflation shape the co-optimizing path already shipped once. Re-keying
    positionally (``data.ranges`` / ``reduction_ranges`` positions, which
    ``_divide_ranges`` leaves untouched) is the prerequisite for a second
    candidate, and belongs with the enumerator that first produces one. Until
    then every buffer carries exactly one, untiled candidate, and
    :func:`single_tile_idx` is what makes a second fail loudly rather than
    silently share a division across frames.
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


@dataclass(frozen=True)
class TilingCandidate:
    """One coarse tiling an op could take, plus the divisions legal under it.

    The two candidate lists on a :class:`CoreDivisionBuffer` -- ``tile_specs``
    and ``core_divisions`` -- are the two node sets of a **many-to-many**
    relation, and ``division_idxs`` is this spec's slice of the edge set: the
    indices into ``core_divisions`` that are legal *on this spec's frame*.

    Many-to-many because the per-spec legal sets are neither subsets nor
    supersets of one another: tiling shrinks a dim's extent (admitting fewer
    factors) while shrinking its per-core span (admitting more splits), so the
    adjacency is irregular and has to be stored rather than recovered from a
    predicate at solve time.

    The relation is **totally participating in both directions** -- a spec with
    no legal division can never be chosen, and a division no spec references is
    unreachable. Both are asserted in
    :meth:`CoreDivisionBuffer.__post_init__`; violating either is a build bug,
    not a plan the solver should be asked to reject.

    Anything that depends on *both* endpoints belongs on the edge, not on
    either list -- the per-core footprint
    (:meth:`CoreDivisionBuffer.per_core_footprint`) and the per-core view,
    which is only valid on this spec's predicted frame.
    """

    spec: TileSpec = TileSpec()
    division_idxs: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.division_idxs, tuple):
            object.__setattr__(self, "division_idxs", tuple(self.division_idxs))

    @property
    def label(self) -> str:
        return self.spec.label


@dataclass
class CoreDivisionBuffer(LifetimeBoundBuffer):
    """A :class:`LifetimeBoundBuffer` carrying the joint core-division metadata

    The placement-only solvers (greedy/first-fit/best-fit) never look at these
    fields, so they stay on this subclass rather than the shared base.

    ``core_divisions`` and ``tile_specs`` are two lists, related many-to-many
    through :attr:`TilingCandidate.division_idxs` (see that class). A plan is
    one *edge*: ``(chosen_tiling, chosen_division)``. ``None`` on either means
    undecided, which is distinct from a chosen candidate whose tiling is the
    empty spec -- undecided is a bug, untiled is a plan.
    """

    core_divisions: list[CoreDivision] = field(default_factory=list)
    # Coarse tilings this op could take, each naming the divisions legal under
    # it. Defaults to the single untiled candidate over every division, which
    # is the inert shape every pre-coarse-tiling caller gets.
    tile_specs: list[TilingCandidate] = field(default_factory=list)
    # Producer buffer names; defines the producer->consumer edges for matching.
    parents: list[str] = field(default_factory=list[str])
    # parent_buf_name -> (parent_div_idx, this_div_idx) pairs that induce the
    # *same per-core slicing of the parent*, precomputed by the allocator via
    # ``_per_core_view_on_buf`` (physical device-dim view equality, correct
    # across reductions/reshapes). These are the sole slicing-match predicate;
    # an absent/empty entry means no compatible division, so the gate forbids
    # the merge/residency across that edge.
    #
    # Keyed on division indices alone, which is what the two-list form buys:
    # residency across an intra-group edge forces producer and consumer onto
    # the *same* spec, so the spec is shared across the edge rather than
    # ranging freely on both sides.
    cd_parent_matches: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    chosen_division: Optional[int] = None
    chosen_tiling: Optional[int] = None
    boundary: BufferType = BufferType.Intermediate

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.tile_specs and self.core_divisions:
            self.tile_specs = [
                TilingCandidate(TileSpec(), tuple(range(len(self.core_divisions))))
            ]
        self.assert_relation_total()

    def assert_relation_total(self) -> None:
        """Assert the spec/division relation participates totally both ways.

        Re-runnable, because the allocator fills these lists in stages; every
        mutation of either list has to leave this true.
        """
        n = len(self.core_divisions)
        referenced: set[int] = set()
        for t, cand in enumerate(self.tile_specs):
            assert cand.division_idxs, (
                f"buffer {self.name} tiling candidate {t} ({cand.label}) lists no "
                "legal core division, so it can never be chosen; it must not be "
                "offered to the solver"
            )
            for d in cand.division_idxs:
                assert 0 <= d < n, (
                    f"buffer {self.name} tiling candidate {t} ({cand.label}) "
                    f"references division {d}, out of range for {n} divisions"
                )
            referenced.update(cand.division_idxs)
        assert len(referenced) == n, (
            f"buffer {self.name} has core divisions "
            f"{sorted(set(range(n)) - referenced)} that no tiling candidate "
            "references, so they are unreachable"
        )

    def edges(self) -> list[tuple[int, int]]:
        """Every legal ``(tile_idx, div_idx)`` pair, in list order.

        This is the domain a solver selects one element from, and the index
        space its per-edge tables (footprint, view) are built over.
        """
        return [
            (t, d) for t, cand in enumerate(self.tile_specs) for d in cand.division_idxs
        ]

    def per_core_footprint(self, tile_idx: int, div_idx: int) -> int:
        """Per-core LX footprint of this buffer under one edge.

        The single definition of the divisor, because it is the one quantity
        that depends on *both* endpoints: ``output_partition`` is spatial (how
        many cores hold a slice concurrently) and ``output_tile_count`` is
        temporal (how many sequential tiles the op's own output write is cut
        into, reduction levels excluded -- a tiled reduction keeps the full
        output extent as its accumulator). Anything that sizes a buffer for
        placement must come through here; splitting the two divisors across
        call sites is how one of them gets forgotten.
        """
        cd = self.core_divisions[div_idx]
        spec = self.tile_specs[tile_idx].spec
        return ceil_div(self.size, cd.output_partition * spec.output_tile_count)

    @property
    def min_footprint(self) -> int:
        """Smallest per-core footprint any legal edge allows. With no candidates
        there is nothing to divide by, so it falls back to ``size`` (the
        placement-only case ``_wrap`` also dispatches on).

        Minimised over *edges*, not over divisions: a division's footprint is
        not defined without the spec it is taken under.
        """
        if not self.core_divisions:
            return self.size
        return min(self.per_core_footprint(t, d) for t, d in self.edges())


def single_tile_idx(buf: "CoreDivisionBuffer") -> int:
    """The index of ``buf``'s only tiling candidate, asserting there is one.

    Every solver in the tree today chooses a *division* but not a *tiling*, so
    it prices each division under one spec. That is sound only while the buffer
    offers exactly one, and this is where that assumption is stated and checked
    rather than assumed at each sizing site. When a solver gains a real tiling
    variable, its per-candidate tables become per-*edge*
    (:meth:`CoreDivisionBuffer.edges`) selected by a channelled ``(tile, div)``
    pair, and these call sites are exactly the ones that have to change --
    which is what the assert makes impossible to miss.
    """
    assert buf.tile_specs, f"buffer {buf.name} offers no tiling candidate"
    assert len(buf.tile_specs) == 1, (
        f"buffer {buf.name} offers {len(buf.tile_specs)} tiling candidates, but "
        "this solver models only a core-division choice; pricing every division "
        "under one spec would be wrong. The model needs a tiling variable and "
        "per-edge tables before it can accept these candidates."
    )
    return 0


def assert_in_place_parent_is_read(
    parent: "LifetimeBoundBuffer", child_name: str
) -> None:
    """Assert an in-place parent's storage is read before it is handed over.

    The child takes the parent's storage over at the parent's last use, so that
    use has to be a read. For a computed buffer the first use is the write, so a
    single use means the buffer is written and never read -- handing it to a
    child would overwrite data nothing ever consumed, and it would make parent
    and child come alive on the same tick while sharing storage. Graph inputs are
    exempt: all their uses are reads, so one use is enough.

    Split out of :func:`_assert_in_place_relationships` because the
    permutation-based layout solvers enforce this one invariant on its own. For
    them the other two are placement-time gates rather than preconditions: a
    child that outgrows its parent, or a pair whose lifetimes do not abut, is
    simply not placed in-place (see ``_can_inplace``), so asserting either here
    would reject inputs those solvers handle correctly.
    """
    # Tested as "a use strictly after the first" rather than via ``read_count``:
    # the two agree whenever ``uses`` is strictly increasing, but ``uses`` is
    # validated at construction and can be mutated afterwards, and this way a
    # repeated index cannot pass as a read.
    has_read_after_write = len(parent.uses) > 1 and parent.uses[-1] > parent.uses[0]
    assert parent.first_use_is_read or has_read_after_write, (
        f"In-place parent {parent.name} is a computed buffer that is never read "
        f"(uses={parent.uses}), so it cannot hand its storage to child "
        f"{child_name}"
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
                assert_in_place_parent_is_read(parent, child.name)
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
