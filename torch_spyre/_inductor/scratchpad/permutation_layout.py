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


"""Permutation-based incremental layout solving.

A permutation fixes an allocation order; :class:`PermutationBasedLayoutSolver`
places buffers in that order and maintains addresses incrementally under
``swap``/``rotate`` via the order-based contact profiles, while
:class:`ReferencePermutationBasedLayoutSolver` is a from-scratch oracle used
for differential testing. The :class:`Profile` step function is the contact
data structure both build on. Search policies (e.g. the simulated-annealing search)
drive this substrate by composition; it knows nothing about how the
permutation is chosen.
"""

from typing import Optional
from abc import ABC, abstractmethod
import bisect
import heapq
import math

from torch_spyre._inductor.scratchpad.contact_profile import Profile
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer


def buffer_quality(buf: LifetimeBoundBuffer) -> float:
    """The contribution buffer ``buf`` makes to a plan's :meth:`quality` when it
    is fully allocated below capacity.

    Weights the buffer's size by how heavily it is used: each access counts
    once, plus an extra half for a buffer whose first access is a write (a
    computed buffer, ``first_use_is_read`` False) since its initial store also
    touches the slot. Formally
    ``(len(buf.uses) + (0 if buf.first_use_is_read else 0.5)) * buf.size``.
    """
    return (len(buf.uses) + (0.0 if buf.first_use_is_read else 0.5)) * buf.size


# ===========================================================================
# Permutation-based layout solvers
# ===========================================================================


class PermutationBasedLayoutSolverBase(ABC):
    """Shared state and interface for capacity-bounded allocation plans.

    A plan places a set of :class:`LifetimeBoundBuffer` objects into a
    fixed-capacity scratchpad following a *permutation*: an explicit allocation
    order given as a list of buffer indices. Buffer ``permutation[k]`` is
    allocated on top of every already-placed buffer whose lifetime overlaps it
    (respecting in-place parents), rounded up to ``alignment``.

    Addresses are maintained internally and are **not** written back to the
    buffer objects until :meth:`finalize`. Two buffers that are alive at the
    same logical tick may not occupy overlapping address ranges, with the sole
    exception of an in-place parent/child pair, which may share an identical
    address (``P.end_time == C.start_time + 1``).

    The objective being optimized is :meth:`quality`: the summed
    :func:`buffer_quality` (use-weighted size) of every buffer that fits
    *entirely* below ``capacity``. A buffer whose placement would cross the
    capacity line is *evicted* -- its address is ``None`` (the single source of
    truth for eviction) and it is neither counted nor written back on
    :meth:`finalize`. Eviction is upward-closed: anything that would rest on an
    evicted buffer is evicted too.

    Subclasses implement :meth:`_build` (initial placement) and :meth:`swap`
    (incremental re-placement after exchanging two adjacent permutation
    entries).

    Args:
        buffers: The buffers to place. Indices into this list are the values
            used in ``permutation`` and as keys throughout the plan.
        permutation: Allocation order as a permutation of
            ``range(len(buffers))``.
        capacity: Scratchpad capacity in bytes.
        alignment: Byte alignment boundary for placed addresses. Defaults to 128
            (one Spyre stick).
    """

    def __init__(
        self,
        buffers: list[LifetimeBoundBuffer],
        permutation: list[int],
        capacity: int,
        alignment: int = 128,
    ):
        n = len(buffers)
        assert sorted(permutation) == list(range(n)), (
            "permutation must be a permutation of range(len(buffers))"
        )
        self.buffers = buffers
        self.permutation = list(permutation)
        self.capacity = capacity
        self.alignment = alignment
        self._name_to_idx = {buf.name: i for i, buf in enumerate(buffers)}

        # Per-buffer size as a flat list, for fast access in the placement hot
        # loop (avoids a dataclass attribute lookup per candidate). Immutable.
        self._sizes = [buf.size for buf in buffers]

        # Per-buffer quality contribution (use-weighted size) as a flat list,
        # summed into total_quality for every fully-allocated buffer. Immutable.
        self._qualities = [buffer_quality(buf) for buf in buffers]

        # Per-buffer set of possible in-place partners (its declared parents and
        # the children that declare it). Static -- a function of names and
        # in_place_parents -- so computed once and consulted instead of probing
        # every candidate during placement. See _placement_decision.
        self._inplace_partners = self._compute_inplace_partners()

        # Lifetime-interval data for the saturation early-stop (Part 2; used by
        # the incremental solver's sequential placers, see _sequential_place).
        # Static -- a function of lifetimes only -- so computed once here.
        self._build_interval_data()

        # Internal address per buffer index; None means evicted (does not fit
        # below capacity). Populated by _build and kept in sync by swap. Not
        # written to buffer objects until finalize.
        self.addresses: list[Optional[int]] = [0] * n

        # Sum of buffer_quality(buf) over all fully-allocated buffers (address +
        # size <= capacity). Maintained incrementally; exposed via quality().
        # Also, the count of these buffers, exposed via count_allocated().
        self.total_quality: float = 0.0
        self.total_allocated_count: int = 0

        self._build()

    @abstractmethod
    def _build(self) -> None:
        """Compute addresses for every buffer in permutation order.

        Populates ``self.addresses`` and ``self.total_quality`` (and any
        subclass-specific structures). Called once from ``__init__``.
        """

    @abstractmethod
    def swap(self, i: int) -> float:
        """Swap permutation entries ``i`` and ``i + 1`` and re-place buffers.

        Args:
            i: Position in the permutation; entries ``i`` and ``i + 1`` are
                exchanged.

        Returns:
            The change in :meth:`quality` caused by the swap (new minus old).
        """

    # --- shared helpers -----------------------------------------------------

    def rotate(self, i: int, j: int) -> float:
        """Modify the permutation by taking ``self.permutation[i]`` out of the
        permutation and reinserting it at position ``j``. Returns the change in
        :meth:`quality` caused by the rotation (new minus old)."""
        # A product of swaps, even over the full distance, beats a permutation-edit +
        # _build(): most of the swaps are O(1) no-ops, so the chain is far cheaper
        # than an O(n^2) rebuild in the realistic (sparse-overlap) regime. (A rebuild
        # only wins for dense overlap, where it is a symptom of swap propagation
        # degenerating -- a thing to fix, not to route around. See
        # benchmarks/copy_vs_swap_results.md.)
        delta = 0.0
        if i < j:
            for k in range(i, j):
                delta += self.swap(k)
        elif j < i:
            for k in range(i - 1, j - 1, -1):
                delta += self.swap(k)
        return delta

    def _align_up(self, addr: int) -> int:
        """Round ``addr`` up to the next multiple of ``self.alignment``."""
        return math.ceil(addr / self.alignment) * self.alignment

    def _top(self, idx: int) -> Optional[int]:
        """Return ``address + size`` for a placed buffer (its exclusive top), or
        ``None`` if ``idx`` is evicted (has no address)."""
        if self.addresses[idx] is None:
            return None
        return self.addresses[idx] + self._sizes[idx]  # type: ignore

    def is_fully_allocated(self, idx: int) -> bool:
        """True if buffer ``idx`` has an address (and so fits below ``capacity``).

        ``None`` is the single source of truth for eviction: a buffer carries a
        concrete address iff it fits entirely below ``capacity`` (the capacity
        gate lives in :meth:`_placement_decision`), so "has an address" and
        "fully allocated" coincide.
        """
        return self.addresses[idx] is not None

    def overlaps(self, i: int, j: int) -> bool:
        """True if buffers ``i`` and ``j`` are alive at a common tick.

        Lifetimes are half-open intervals ``[start_time, end_time)``, so an
        in-place parent and child (``parent.end_time == child.start_time + 1``)
        overlap at exactly that boundary tick (``child.start_time``).
        """
        return self.buffers[i].overlaps_in_time(self.buffers[j])

    def _in_place_pair(self, i: int, j: int) -> Optional[tuple[int, int]]:
        """Return ``(parent_idx, child_idx)`` if ``i`` and ``j`` form an in-place
        pair, else ``None``.

        The relationship is declared on the child via ``in_place_parents``; it is
        symmetric for placement purposes, so either argument may be the parent.
        """
        bi = self.buffers[i]
        bj = self.buffers[j]
        if bj.name in bi.in_place_parents:
            return (j, i)  # j is the parent of i
        if bi.name in bj.in_place_parents:
            return (i, j)  # i is the parent of j
        return None

    def _compute_inplace_partners(self) -> list[set[int]]:
        """For each buffer index, the set of buffers it could share a slot with
        in-place: ``{j : _in_place_pair(i, j) is not None}``. This is exactly its
        declared parents plus the children that declare it -- a static function
        of names and ``in_place_parents``, so it is computed once and lets
        :meth:`_placement_decision` probe only real partners instead of testing
        every candidate.
        """
        n = len(self.buffers)
        partners: list[set[int]] = [set() for _ in range(n)]
        for child, buf in enumerate(self.buffers):
            for pname in buf.in_place_parents:
                parent = self._name_to_idx.get(pname)
                if parent is not None:
                    partners[child].add(parent)
                    partners[parent].add(child)
        return partners

    def _can_inplace(self, parent: int, child: int) -> bool:
        """True if ``child`` is allowed to share ``parent``'s address.

        A child may only reuse a parent's storage if it fits within it; a
        larger child would still need the parent's inputs while writing past
        the parent's footprint.
        """
        return self.buffers[child].size <= self.buffers[parent].size

    def _placement_decision(
        self, idx: int, candidates: list[int]
    ) -> tuple[Optional[int], Optional[int]]:
        """Decide ``idx``'s address given the buffers it must sit on top of.

        ``candidates`` are already-placed buffer indices that overlap ``idx`` in
        time. For the reference plan these are *all* time-overlapping buffers;
        for the incremental plan they are ``idx``'s direct below-neighbours --
        both yield the same decision, because the highest top among them is the
        same and that is all the rule depends on.

        ``idx`` is placed on top of everything it overlaps. The one exception is
        an in-place partner ``P`` (``P.end_time == idx.start_time + 1`` or vice
        versa): ``idx`` may instead drop into ``P``'s slot, reusing ``P``'s
        address, but *only* when every other overlapping buffer already tops out
        at or below ``P``'s address -- otherwise ``idx`` would land partway into
        occupied space. When that holds, dropping onto ``P`` still leaves ``idx``
        above all the others (it saves ``P``'s footprint rather than stacking on
        top of it).

        This method is the single eviction authority: ``None`` is returned as the
        address whenever ``idx`` does not fit entirely below ``capacity``.
        Eviction is upward-closed, so ``idx`` is evicted if *any* candidate is
        itself evicted (``idx`` would rest on a buffer that has no address) --
        detected without computing the ``max``, since a ``None`` candidate
        dominates. Otherwise ``idx``'s aligned top must not cross ``capacity``.

        Returns:
            ``(address, partner)`` where ``address`` is ``None`` when ``idx`` is
            evicted, and ``partner`` is the candidate whose address was reused
            in-place (or ``None`` if ``idx`` was stacked / evicted).
        """
        if not candidates:
            # Lone buffer: it sits on the floor at address 0, but a buffer larger
            # than the whole scratchpad is evicted (the real hole in "floor => 0").
            if self._sizes[idx] > self.capacity:
                return None, None
            return 0, None
        addr = self.addresses
        sizes = self._sizes
        # A None (evicted) candidate dominates: idx would rest on it, so idx is
        # evicted too. Detect this before the max (None has no finite top).
        if any(addr[p] is None for p in candidates):
            return None, None
        # _top inlined as addr[p] + sizes[p] on locals: this max runs once per
        # placed buffer over all its candidates and is the placement hot loop.
        max_top = max(addr[p] + sizes[p] for p in candidates)  # type: ignore
        # Try to drop into an in-place partner's slot. Only ``idx``'s precomputed
        # in-place partners can qualify, so probe those that are present among
        # the candidates rather than testing every candidate. At most one can
        # qualify: if two did, each would have to top out below the other's
        # address, which is impossible -- so iteration order does not matter.
        partners = self._inplace_partners[idx]
        if partners:
            for partner in partners.intersection(candidates):
                pair = self._in_place_pair(idx, partner)
                assert pair is not None  # partner came from the in-place set
                if not self._can_inplace(*pair):
                    continue
                partner_addr = addr[partner]
                assert partner_addr is not None  # the partner is allocated
                others_top = max(
                    (
                        addr[q] + sizes[q]  # type: ignore
                        for q in candidates
                        if q != partner
                    ),
                    default=0,
                )
                if others_top <= partner_addr:
                    # In-place reuse fits whenever the partner does (the child is
                    # no larger than the partner), but gate on capacity uniformly.
                    if partner_addr + sizes[idx] > self.capacity:
                        return None, None
                    return partner_addr, partner
        aligned_addr = self._align_up(max_top)
        if aligned_addr + sizes[idx] > self.capacity:
            return None, None
        return aligned_addr, None

    def _address_from_candidates(
        self, idx: int, candidates: list[int]
    ) -> Optional[int]:
        """Return only the address from :meth:`_placement_decision`."""
        return self._placement_decision(idx, candidates)[0]

    # --- saturation early-stop (Part 2; incremental sequential placers) ------

    def _build_interval_data(self) -> None:
        """Precompute the lifetime-interval structures the saturation early-stop
        reads. Static (a function of lifetimes only), built once in ``__init__``
        and shared by reference in :meth:`copy`.

        Reuses the :meth:`_build_profiles` breakpoint idiom: the sorted unique
        lifetime endpoints cut the timeline into ``K`` half-open intervals
        ``[interval_starts[k], interval_starts[k + 1])`` (``K`` can be 0 when
        ``n == 0``). For each:

        - ``_total_at[k]`` -- how many buffers are alive on interval ``k`` (a
          delta sweep over the endpoints, accumulated into a running count).
        - ``_buf_intervals[idx]`` -- the half-open range ``[lo, hi)`` of interval
          indices buffer ``idx`` covers (``bisect`` of its start/end).
        """
        bufs = self.buffers
        starts = sorted({b.start_time for b in bufs} | {b.end_time for b in bufs})
        self._interval_starts = starts
        k = max(0, len(starts) - 1)
        self._num_intervals = k
        total = [0] * k
        # Delta sweep: +1 at each start interval, -1 at each end interval; the
        # prefix sum over intervals is the alive count.
        deltas = [0] * (k + 1)
        for b in bufs:
            deltas[bisect.bisect_left(starts, b.start_time)] += 1
            deltas[bisect.bisect_left(starts, b.end_time)] -= 1
        running = 0
        for i in range(k):
            running += deltas[i]
            total[i] = running
        self._total_at = total
        self._buf_intervals = [
            (
                bisect.bisect_left(starts, b.start_time),
                bisect.bisect_left(starts, b.end_time),
            )
            for b in bufs
        ]

    def _sequential_place(self, get_candidates) -> None:
        """Place every buffer in permutation order, with the saturation
        early-stop. Resets and repopulates ``addresses``, ``inplace_reuse``,
        ``total_quality`` and ``total_allocated_count``.

        ``get_candidates(pos, idx)`` returns the already-placed candidate list
        for the buffer at permutation position ``pos`` -- the only thing that
        differs between the incremental ``_build`` (a ``prior``-scan) and
        ``_recompute_all_addresses`` (an ``overlap_dict`` lookup).

        Early-stop: an interval is *done* once it can accept nothing more --
        either it already carries an evicted buffer (``has_none_at``) or every
        buffer alive on it has been placed (``placed_at == total_at``). Once all
        intervals are done, every remaining buffer is alive only over saturated
        intervals and therefore rests (transitively) on an evicted buffer, so it
        is evicted too; we stop and bulk-set the tail to ``None``. This is
        result-identical to running the full loop (see the module/plan notes),
        so it never changes addresses -- only the work to compute them.
        """
        n = len(self.buffers)
        perm = self.permutation
        self.inplace_reuse: dict[int, int] = {}
        self.total_quality = 0.0
        self.total_allocated_count = 0

        k = self._num_intervals
        total_at = self._total_at
        buf_intervals = self._buf_intervals
        placed_at = [0] * k
        has_none_at = [False] * k
        done_at = [total_at[t] == 0 for t in range(k)]
        not_done = k - sum(done_at)

        stop = n  # permutation position at which the early-stop fired (n => none)
        for pos in range(n):
            if not_done == 0:
                stop = pos
                break
            idx = perm[pos]
            addr, partner = self._placement_decision(idx, get_candidates(pos, idx))
            self.addresses[idx] = addr
            if partner is not None:
                self.inplace_reuse[idx] = partner
            evicted = addr is None
            if not evicted:
                self.total_quality += self._qualities[idx]
                self.total_allocated_count += 1
            lo, hi = buf_intervals[idx]
            for t in range(lo, hi):
                placed_at[t] += 1
                if evicted:
                    has_none_at[t] = True
                if not done_at[t] and (has_none_at[t] or placed_at[t] == total_at[t]):
                    done_at[t] = True
                    not_done -= 1

        for pos in range(stop, n):
            self.addresses[perm[pos]] = None

    def quality(self) -> float:
        """Summed :func:`buffer_quality` of all buffers fully allocated below
        capacity (O(1))."""
        return self.total_quality

    def count_allocated(self) -> int:
        """Count of all buffers fully allocated below capacity (O(1))."""
        return self.total_allocated_count

    def finalize(self) -> None:
        """Write back each buffer's address to the buffer object.

        ``self.addresses[idx]`` is already the single source of truth: a concrete
        address for a buffer that fits below ``capacity``, or ``None`` for an
        evicted one (which is not committed). So the write-back is a direct copy.
        """
        for idx, buf in enumerate(self.buffers):
            buf.address = self.addresses[idx]


class ReferencePermutationBasedLayoutSolver(PermutationBasedLayoutSolverBase):
    """Simple, obviously-correct O(n^2) reference plan.

    Placement scans all previously-placed, time-overlapping buffers for each
    buffer; ``swap`` mutates the permutation and rebuilds from scratch. Kept as
    a permanent oracle for differential testing against the incremental
    :class:`PermutationBasedLayoutSolver`.
    """

    def _build(self) -> None:
        n = len(self.buffers)
        self.addresses = [0] * n
        self.total_quality = 0.0
        self.total_allocated_count = 0
        for pos in range(n):
            idx = self.permutation[pos]
            prior = self.permutation[:pos]
            candidates = [p for p in prior if self.overlaps(idx, p)]
            self.addresses[idx] = self._address_from_candidates(idx, candidates)
            if self.is_fully_allocated(idx):
                self.total_quality += self._qualities[idx]
                self.total_allocated_count += 1

    def swap(self, i: int) -> float:
        """Swap permutation entries ``i``/``i+1`` and rebuild from scratch."""
        old_total = self.total_quality
        perm = self.permutation
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        self._build()
        return self.total_quality - old_total


class PermutationBasedLayoutSolver(PermutationBasedLayoutSolverBase):
    """Incremental capacity-bounded allocation plan.

    Maintains, for each buffer, a *contact profile* -- a step function over its
    lifetime giving the buffer directly below / above it in the per-column
    stacking order (or None at the ends). Swapping two adjacent permutation
    entries transposes them only over their shared column range, so the profiles
    are updated by O(segments) splices rather than rebuilt; addresses are then
    re-placed for the buffers the change actually reaches, propagated along the
    time-overlap dependency graph.

    The contact relation is purely order-based (a function of the permutation
    and lifetimes): at a column the alive buffers are ordered by permutation
    position, and ``below_profile[c]`` at that column is ``c``'s immediate
    predecessor in that order. In-place placement is ignored by the relation
    (parent-before-child means parent-below-child); it still affects addresses,
    which are computed separately.

    Attributes:
        below_profile: ``below_profile[c]`` maps each column of ``c``'s lifetime
            to the buffer directly below ``c`` there, or None.
        above_profile: the inverse relation; used to find which buffers may need
            re-placing when ``c``'s top moves.
        inplace_reuse: ``inplace_reuse[x] = y`` when buffer ``x`` reused
            partner ``y``'s address in-place (``x`` was placed at ``y``'s
            address).
    """

    def _build(self) -> None:
        n = len(self.buffers)
        self.addresses = [0] * n
        # Place every buffer in permutation order (candidates are the earlier,
        # time-overlapping buffers), with the saturation early-stop. This sets
        # addresses, inplace_reuse and the running totals.
        self._sequential_place(
            lambda pos, idx: [
                p for p in self.permutation[:pos] if self.overlaps(idx, p)
            ]
        )
        # Persistent position index, maintained in O(1) by swap().
        self.position: list[int] = [0] * n
        for p, idx in enumerate(self.permutation):
            self.position[idx] = p
        # Time-overlap sets. Lifetimes never change, so this is computed once
        # and lets the address recompute find a buffer's candidates in O(degree)
        # instead of scanning all n buffers.
        self.overlap_dict: dict[int, set[int]] = {i: set() for i in range(n)}
        for a in range(n):
            for b in range(a + 1, n):
                if self.overlaps(a, b):
                    self.overlap_dict[a].add(b)
                    self.overlap_dict[b].add(a)
        # Minimum |i - j| at which rotate() uses the remove/reinsert fast path
        # (_fast_rotate) instead of the adjacent-swap chain; below it the chain
        # is cheaper because most of its swaps are O(1) no-ops. n//8 (~0.125n) is
        # picked from the measured crossover -- the fraction of n above which the
        # fast path wins -- which is ~0.04-0.15n at medium overlap density and
        # ~0.13-0.37n at low density (it falls as density rises, since the swap
        # chain's per-overlap propagation grows super-linearly while the fast
        # path is ~independent of distance). So n//8 sits below the
        # medium-density crossover (engaging the fast path where it clearly pays)
        # and is mildly conservative at low density (it may engage a touch early,
        # but both paths are sub-millisecond there). It is an instance attribute
        # so callers/tests can override it -- set it to 1 to force the fast path
        # on every rotation.
        self._rotate_remove_insert_threshold = max(2, n // 8)
        self._build_profiles()

    def _build_profiles(self) -> None:
        """Build the below/above contact profiles from ground truth.

        At each column the buffers alive there are totally ordered by
        permutation position (the bottom-to-top stacking order); a buffer's
        below/above neighbour is its immediate predecessor / successor in that
        per-column order, or None at the ends. Sweeping the breakpoint intervals
        and reading adjacent pairs gives each buffer's contact step function over
        its lifetime. In-place placement is ignored -- the relation is purely a
        function of the permutation and lifetimes.
        """
        n = len(self.buffers)
        self.below_profile: dict[int, Profile] = {}
        self.above_profile: dict[int, Profile] = {}
        if n == 0:
            return
        bufs = self.buffers
        below_segs: dict[int, tuple[list[int], list[Optional[int]]]] = {
            i: ([], []) for i in range(n)
        }
        above_segs: dict[int, tuple[list[int], list[Optional[int]]]] = {
            i: ([], []) for i in range(n)
        }
        breakpoints = sorted({b.start_time for b in bufs} | {b.end_time for b in bufs})
        for t0 in breakpoints[:-1]:
            alive = sorted(
                (i for i in range(n) if bufs[i].start_time <= t0 < bufs[i].end_time),
                key=lambda i: self.position[i],
            )
            for idx, c in enumerate(alive):
                below = alive[idx - 1] if idx > 0 else None
                above = alive[idx + 1] if idx + 1 < len(alive) else None
                below_segs[c][0].append(t0)
                below_segs[c][1].append(below)
                above_segs[c][0].append(t0)
                above_segs[c][1].append(above)
        for i in range(n):
            bs, bl = below_segs[i]
            bs.append(bufs[i].end_time)
            self.below_profile[i] = Profile.from_segments(bs, bl)
            as_, al = above_segs[i]
            as_.append(bufs[i].end_time)
            self.above_profile[i] = Profile.from_segments(as_, al)

    def swap(self, i: int) -> float:
        """Swap permutation entries ``i`` and ``i+1`` and re-place incrementally.

        A no-op when the swapped buffers do not overlap in time. Otherwise:

        1. Over their shared column range the two buffers' per-column order
           transposes and nothing else changes, so the contact profiles are
           updated by a handful of splices (:meth:`_update_profiles_for_swap`).
        2. Addresses are then re-placed for the buffers the change reaches,
           processed in a min-heap by position (dependencies always point to
           earlier positions, so a buffer is settled before anything resting on
           it; ``position`` is maintained in O(1)). Two kinds of edge feed the
           dirty set:

           - *Order-above.* When ``z``'s address changes, the buffers directly
             above it -- ``above_profile[z]`` -- are dirtied. This is the cheap
             contact-profile frontier and it is exactly right whenever the
             buffer a dependent rests on is also its order-below neighbour.

           - *In-place transition.* In-placement makes the contact order and the
             rest-on order diverge: a transparent in-place child sits low while
             its taller parent pokes through and binds the buffer above the
             child. While that in-placement is stable the order-above frontier
             still suffices (the child's address tracks the parent it reuses, so
             a change in the parent reaches the buffer above the child through
             the child). The gap is at the *transition*: when a buffer ``z``'s
             in-place status flips (activates or deactivates), the poke-through
             appears or vanishes, so the buffer resting on it must be revisited
             even though nothing it can see changed value. So on a status change
             we dirty the order-above neighbour of *both* members of the pair at
             their shared (overlap) tick -- the parent's above-neighbour is the
             child, and the child's above-neighbour is the buffer that gains or
             loses the poke-through.

        Returns:
            The change in :meth:`quality` (new minus old).
        """
        n = len(self.buffers)
        assert 0 <= i < n - 1
        perm = self.permutation
        x, y = perm[i], perm[i + 1]
        perm[i], perm[i + 1] = y, x
        self.position[x], self.position[y] = i + 1, i
        if not self.overlaps(x, y):
            # Independent buffers: their order does not affect any address.
            return 0

        # 1. Transpose the contact profiles over the shared column range.
        a = max(self.buffers[x].start_time, self.buffers[y].start_time)
        b = min(self.buffers[x].end_time, self.buffers[y].end_time)
        self._update_profiles_for_swap(x, y, a, b)

        # 2. Re-place affected addresses, propagating along order-above edges and
        # in-place transitions (see the method docstring). Seed with the swapped
        # pair and whatever rested on them before the swap.
        old_total = self.total_quality
        seed: set[int] = {x, y}
        for lbl in (
            self.above_profile[x].label_set() | self.above_profile[y].label_set()
        ):
            if lbl is not None:
                seed.add(lbl)
        heap = [(self.position[idx], idx) for idx in seed]
        heapq.heapify(heap)
        queued = set(seed)
        # Buffers already settled as evicted by the flip-to-None fast path below.
        # heapq has no cheap delete, so a flipped buffer is skipped (lazily) if it
        # is later popped from the normal heap.
        flipped: set[int] = set()

        def _dirty(w: Optional[int], pos_z: int) -> None:
            if w is not None and w not in queued and self.position[w] > pos_z:
                queued.add(w)
                heapq.heappush(heap, (self.position[w], w))

        while heap:
            _, z = heapq.heappop(heap)
            queued.discard(z)
            if z in flipped:
                continue
            pos_z = self.position[z]
            old_addr = self.addresses[z]
            old_partner = self.inplace_reuse.get(z)
            if self.is_fully_allocated(z):
                self.total_quality -= self._qualities[z]
                self.total_allocated_count -= 1
            self._recompute_address(z)
            if self.addresses[z] is None:
                # z is evicted and final (position ordering: nothing below it
                # changes after it is popped, so it cannot un-evict). Everything
                # resting transitively on z is therefore evicted too -- exactly
                # z's order-above closure (a buffer is evicted iff an evicted
                # buffer lies in its candidate set, i.e. is one of its order-below
                # neighbours). Bulk-flip that closure to None directly instead of
                # re-deriving each via _recompute_address (which would just
                # short-circuit to None). z's own quality was already removed
                # above and is not re-added.
                self._flip_evicted_closure(z, flipped)
                continue
            self.total_quality += self._qualities[z]
            self.total_allocated_count += 1
            new_partner = self.inplace_reuse.get(z)
            if self.addresses[z] != old_addr:
                for w in self.above_profile[z].label_set():
                    _dirty(w, pos_z)
            if new_partner != old_partner:
                # In-place status changed: revisit the buffers resting on the
                # pair at the tick where parent and child overlap.
                for partner in (old_partner, new_partner):
                    if partner is None:
                        continue
                    pair = self._in_place_pair(z, partner)
                    assert pair is not None  # partner is a recorded in-place reuse
                    parent, child = pair
                    t = self.buffers[child].start_time
                    _dirty(self.above_profile[child].label_at(t), pos_z)
                    _dirty(self.above_profile[parent].label_at(t), pos_z)
        return self.total_quality - old_total

    def _flip_evicted_closure(self, z: int, flipped: set[int]) -> None:
        """Evict every buffer resting (transitively) on the just-evicted ``z``.
        This method updates ``flipped``.

        These are exactly ``z``'s order-above closure: ``w`` rests on ``z`` iff
        ``z`` is one of ``w``'s order-below neighbours (``w in above_profile[z]``),
        and an evicted order-below neighbour forces ``w`` evicted regardless of
        its other supports. Each is set to ``None`` (clearing any in-place reuse
        and decrementing the running totals) and marked ``flipped`` so the normal
        heap skips it when popped. Order-above neighbours always have strictly
        higher position, so the closure is finite and never revisits ``z``.
        """
        stack = [w for w in self.above_profile[z].label_set() if w is not None]
        while stack:
            w = stack.pop()
            if w in flipped:
                continue
            flipped.add(w)
            if self.is_fully_allocated(w):
                self.total_quality -= self._qualities[w]
                self.total_allocated_count -= 1
            self.addresses[w] = None
            self.inplace_reuse.pop(w, None)
            for u in self.above_profile[w].label_set():
                if u is not None and u not in flipped:
                    stack.append(u)

    def _update_profiles_for_swap(self, x: int, y: int, a: int, b: int) -> None:
        """Transpose ``x`` (was lower) and ``y`` (was upper) in the contact
        profiles over the shared column range ``[a, b)``.

        Captures both views before mutating, then runs the same splice logic
        once per side (downward and upward are exact mirrors).
        """
        old_x_below = self.below_profile[x].segments(a, b)
        old_y_above = self.above_profile[y].segments(a, b)
        self._splice_half(
            self.below_profile, self.above_profile, x, y, a, b, old_x_below
        )
        self._splice_half(
            self.above_profile, self.below_profile, y, x, a, b, old_y_above
        )

    @staticmethod
    def _splice_half(
        primary: dict[int, Profile],
        reverse: dict[int, Profile],
        lo: int,
        hi: int,
        a: int,
        b: int,
        old_lo: tuple[list[int], list[Optional[int]]],
    ) -> None:
        """One side of the transposition. ``lo`` was directly below ``hi`` (in
        the ``primary`` direction) over ``[a, b)``; after the swap ``hi`` is.

        - ``primary[lo]`` over ``[a, b)`` becomes ``hi``.
        - ``primary[hi]`` over ``[a, b)`` inherits ``lo``'s old ``primary`` view.
        - Each buffer ``lo`` pointed at keeps the relationship but now via
          ``hi``, so its ``reverse`` profile relabels ``lo -> hi`` over that
          segment.
        """
        primary[lo].splice(a, b, [a, b], [hi])
        seg_starts, seg_labels = old_lo
        primary[hi].splice(a, b, list(seg_starts), list(seg_labels))
        for k, label in enumerate(seg_labels):
            if label is not None:
                reverse[label].relabel(seg_starts[k], seg_starts[k + 1], {lo: hi})

    def _recompute_address(self, z: int) -> None:
        """Re-place ``z``'s address from the buffers it actually rests on, read
        off the (already-spliced) contact profile.

        This is :meth:`contact_at` over ``z``'s below-profile breakpoints,
        inlined: walking the profile segments hands us each order-below label
        directly, so we skip ``contact_at``'s per-breakpoint bisect (its hot
        cost), and -- since the candidate set is used unordered -- the
        ``_in_place_pair`` ordering as well. Per segment the candidates are the
        order-below buffer plus, across an active in-place transition (the
        partner it reused is still alive at the segment's first column, so the
        two are co-located there), that co-located partner. This is a provably
        sufficient candidate set for :meth:`_placement_decision`: it preserves
        the maximum top and surfaces exactly the co-located buffers the in-place
        legality test needs, so it yields the same address and partner as
        scanning the full earlier-overlapping set, while touching only ``z``'s
        own contact segments.

        Two distinct meanings of ``None`` meet here and must not be conflated:

        - A profile *label* ``m is None`` means *floor* -- nothing is below ``z``
          on that segment -- so it contributes no candidate (``continue``).
        - A label ``m`` that is a real neighbour but whose ``addresses[m] is
          None`` is an *evicted* neighbour. It is still added to ``cand``;
          :meth:`_placement_decision` then short-circuits it to an evicted
          (``None``) placement for ``z``, since ``z`` would rest on it. The tail
          below clears ``inplace_reuse[z]`` when the result is evicted (partner
          ``None``), so an evicted buffer never keeps a stale reuse entry.
        """
        cand: set[int] = set()
        prof = self.below_profile[z]
        starts, labels = prof.starts, prof.labels
        reuse = self.inplace_reuse
        bufs = self.buffers
        for i, m in enumerate(labels):
            if m is None:
                continue  # floor (no neighbour), not an evicted neighbour
            cand.add(m)
            reused = reuse.get(m)
            if reused is not None:
                rbuf = bufs[reused]
                if rbuf.start_time <= starts[i] < rbuf.end_time:
                    cand.add(reused)
        addr, partner = self._placement_decision(z, list(cand))
        self.addresses[z] = addr
        if partner is None:
            self.inplace_reuse.pop(z, None)
        else:
            self.inplace_reuse[z] = partner

    # --- rotate: remove-one / reinsert-elsewhere fast path ------------------

    def rotate(self, i: int, j: int) -> float:
        """Take ``permutation[i]`` out of the permutation and reinsert it at
        position ``j``; return the change in :meth:`quality` (new minus old).

        Two strategies, chosen by distance ``|i - j|``:

        - **Swap chain (short moves).** ``super().rotate`` walks the element to
          its destination by adjacent :meth:`swap` calls. Most of those swaps
          are O(1) no-ops, so for a short hop this is far cheaper than touching
          the whole permutation.
        - **Remove / reinsert (long moves), :meth:`_fast_rotate`.** For a long
          hop the swap chain re-places the moved element (and the buffers it
          passes) over and over. Instead we edit the permutation once, recompute
          every address in the new order (reusing the static ``overlaps`` sets,
          never the O(n^2) reference scan), and patch the contact profiles for
          the single move -- all in time independent of ``|i - j|``.

        The crossover ``|i - j| >=`` :attr:`_rotate_remove_insert_threshold`
        selects the fast path; the threshold is a tunable instance attribute
        (set it to 1 to force the fast path on every rotation).
        """
        if i == j:
            return 0
        if abs(i - j) < self._rotate_remove_insert_threshold:
            return super().rotate(i, j)
        return self._fast_rotate(i, j)

    def _fast_rotate(self, i: int, j: int) -> float:
        """Remove ``permutation[i]`` and reinsert it at ``j`` in one shot.

        Edits the permutation and ``position`` index, recomputes all addresses
        in the new order (:meth:`_recompute_all_addresses`), then updates the
        contact profiles by an incremental single-move patch
        (:meth:`_patch_profiles_for_move`). Returns the quality delta.
        """
        old_total = self.total_quality
        x = self.permutation[i]
        # Capture x's pre-move contact profiles; the patch needs the old
        # adjacency to stitch x's former neighbours back together. (Cheap
        # shallow copies of the two step functions.)
        old_below = Profile(
            list(self.below_profile[x].starts), list(self.below_profile[x].labels)
        )
        old_above = Profile(
            list(self.above_profile[x].starts), list(self.above_profile[x].labels)
        )
        self._move_in_permutation(i, j)
        self._recompute_all_addresses()
        self._patch_profiles_for_move(x, old_below, old_above)
        return self.total_quality - old_total

    def _move_in_permutation(self, i: int, j: int) -> None:
        """Pop ``permutation[i]`` and reinsert it at ``j``; refresh ``position``
        over the affected range."""
        perm = self.permutation
        x = perm.pop(i)
        perm.insert(j, x)
        lo, hi = (i, j) if i < j else (j, i)
        for p in range(lo, hi + 1):
            self.position[perm[p]] = p

    def _recompute_all_addresses(self) -> None:
        """Re-place every buffer in the current permutation order, reusing the
        static ``overlaps`` sets (never the O(n^2) reference scan).

        Rebuilds ``addresses``, ``inplace_reuse``, ``total_quality`` and
        ``total_allocated_count`` from scratch but in O(sum of overlap degrees),
        which is what makes the long-move rotate independent of ``|i - j|``.

        Unlike :meth:`_recompute_address` (the swap path), this builds candidates
        from the static, order-independent ``overlaps`` set rather than the
        contact profiles. It runs inside :meth:`_fast_rotate` *before* the
        profiles are patched for the move, so at this point ``below_profile``
        still describes the pre-move order and cannot be trusted as a candidate
        source; ``overlaps`` is valid regardless of order. The saturation
        early-stop applies here too (this is a forward sweep in permutation
        order, like ``_build``).
        """
        pos = self.position
        self._sequential_place(
            lambda p, idx: [w for w in self.overlap_dict[idx] if pos[w] < p]
        )

    @staticmethod
    def _iter_common(
        prof_a: Profile, prof_b: Profile
    ) -> "list[tuple[int, int, Optional[int], Optional[int]]]":
        """Walk two profiles over their shared span, yielding ``(lo, hi, a, b)``
        for each maximal sub-interval on which both labels are constant."""
        assert prof_a.span_start == prof_b.span_start
        assert prof_a.span_end == prof_b.span_end
        cuts = sorted(set(prof_a.starts) | set(prof_b.starts))
        out: list[tuple[int, int, Optional[int], Optional[int]]] = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            out.append((lo, hi, prof_a.label_at(lo), prof_b.label_at(lo)))
        return out

    def _patch_profiles_for_move(
        self, x: int, old_below: Profile, old_above: Profile
    ) -> None:
        """Update the contact profiles for the single move of ``x`` to its new
        position, producing profiles byte-identical to a from-scratch rebuild.

        Two stages, both order-based (in-place placement is irrelevant here):

        1. **Remove x.** Over each column x used to occupy, its old below
           neighbour ``a`` and above neighbour ``b`` become adjacent: splice
           ``above_profile[a] := b`` and ``below_profile[b] := a`` (handling the
           ``None`` ends, where the survivor becomes bottom/top).
        2. **Reinsert x.** x's contact neighbours can only be members of
           ``overlaps[x]``; sweep the breakpoints those members induce across
           x's lifetime. On each sub-interval the alive subset is constant, so
           x's new below neighbour is the alive member with the greatest
           ``position < position[x]`` and its above neighbour the one with the
           least greater position (``None`` if none). Rebuild x's own profiles
           and splice x into each neighbour's profile.
        """
        bufs = self.buffers
        s_x, e_x = bufs[x].start_time, bufs[x].end_time

        # --- 1. Remove x: stitch its former below/above neighbours together. --
        for lo, hi, a, b in self._iter_common(old_below, old_above):
            if a is not None:
                self.above_profile[a].splice(lo, hi, [lo, hi], [b])
            if b is not None:
                self.below_profile[b].splice(lo, hi, [lo, hi], [a])

        # --- 2. Reinsert x at its new position. ------------------------------
        pos = self.position
        pos_x = pos[x]
        members = self.overlap_dict[x]
        cuts = {s_x, e_x}
        for w in members:
            if bufs[w].start_time > s_x:
                cuts.add(bufs[w].start_time)
            if bufs[w].end_time < e_x:
                cuts.add(bufs[w].end_time)
        cut_list = sorted(c for c in cuts if s_x <= c <= e_x)

        below_starts: list[int] = []
        below_labels: list[Optional[int]] = []
        above_starts: list[int] = []
        above_labels: list[Optional[int]] = []
        for lo, hi in zip(cut_list[:-1], cut_list[1:]):
            below = None  # greatest position below pos_x among alive members
            below_pos = -1
            above = None  # least position above pos_x among alive members
            above_pos = len(self.permutation)
            for w in members:
                if bufs[w].start_time <= lo < bufs[w].end_time:
                    pw = pos[w]
                    if pw < pos_x:
                        if pw > below_pos:
                            below_pos, below = pw, w
                    elif above is None or pw < above_pos:
                        above_pos, above = pw, w
            below_starts.append(lo)
            below_labels.append(below)
            above_starts.append(lo)
            above_labels.append(above)
            # Splice x into each new neighbour's profile over [lo, hi).
            if below is not None:
                self.above_profile[below].splice(lo, hi, [lo, hi], [x])
            if above is not None:
                self.below_profile[above].splice(lo, hi, [lo, hi], [x])
        below_starts.append(e_x)
        above_starts.append(e_x)
        self.below_profile[x] = Profile.from_segments(below_starts, below_labels)
        self.above_profile[x] = Profile.from_segments(above_starts, above_labels)

    def contact_at(self, c: int, t: int) -> Optional[int] | tuple[int, int]:
        """What occupies the address slot directly below ``c`` at column ``t``,
        derived on demand from the order ``below_profile`` and
        :attr:`inplace_reuse` (nothing extra is stored).

        Inspection/test-only: the placement hot path does not call this; it
        inlines the same derivation over a buffer's own below-profile segments
        in :meth:`_recompute_address`. Kept as a readable, single-column oracle
        for tests and debugging. Three outcomes:

        - ``None`` -- nothing is below ``c`` at ``t`` (``c`` is on the floor).
        - ``int m`` -- a single buffer ``m`` is directly below ``c``; ``c`` rests
          on ``m``.
        - ``(parent, child)`` -- the slot directly below ``c`` is shared by an
          in-place pair at *their* transition column (the one tick on which
          ``parent`` and ``child`` are both alive and co-located at the same
          address). ``c`` rests on ``parent`` (the larger member -- in-place
          requires ``child.size <= parent.size``, so it tops out highest);
          ``child`` is the smaller buffer buried in the same slot.

        The tuple's meaning is role-based, not position-based: which member is
        ``c``'s order-below neighbour depends on the reuse direction. When the
        child reused the parent, the child is the order-below neighbour and the
        parent pokes up from beneath it; when the parent reused the child, the
        parent is the order-below neighbour and the child is buried below it.
        Either way ``parent`` is what ``c`` rests on and both are returned.
        """
        m = self.below_profile[c].label_at(t)
        if m is None:
            return None
        partner = self.inplace_reuse.get(m)
        if partner is not None:
            pair = self._in_place_pair(m, partner)
            assert pair is not None  # m reuses partner: they form a pair
            # m is always alive at t (it is c's order-below), so the pair is at
            # its transition column iff the other member (partner) is alive too.
            obuf = self.buffers[partner]
            if obuf.start_time <= t < obuf.end_time:
                return pair  # (parent, child)
        return m

    def copy(self) -> "PermutationBasedLayoutSolver":
        """Return an independent layout snapshot that can be mutated (via
        :meth:`swap` / :meth:`rotate`) without affecting this one.

        Structures fixed for the lifetime of the plan -- ``buffers``,
        ``_name_to_idx``, ``overlaps`` -- are shared by reference; only the
        dynamic layout state (permutation, addresses, positions, contact
        profiles and running totals) is deep-copied. So this costs O(n + profile
        size), not a rebuild. The result is always a plain
        :class:`PermutationBasedLayoutSolver`, regardless of subclass.
        """
        clone = PermutationBasedLayoutSolver.__new__(PermutationBasedLayoutSolver)
        # Shared, immutable-during-planning structures.
        clone.buffers = self.buffers
        clone._name_to_idx = self._name_to_idx
        clone.capacity = self.capacity
        clone.alignment = self.alignment
        clone.overlap_dict = self.overlap_dict
        clone._sizes = self._sizes
        clone._qualities = self._qualities
        clone._inplace_partners = self._inplace_partners
        # Lifetime-interval data for the saturation early-stop (static).
        clone._interval_starts = self._interval_starts
        clone._num_intervals = self._num_intervals
        clone._total_at = self._total_at
        clone._buf_intervals = self._buf_intervals
        # Rotate-policy knob (a cheap scalar; carried so a clone rotates the
        # same way as its source and tests can flip it on a clone).
        clone._rotate_remove_insert_threshold = self._rotate_remove_insert_threshold
        # Deep-copied dynamic state.
        clone.permutation = list(self.permutation)
        clone.addresses = list(self.addresses)
        clone.position = list(self.position)
        clone.total_quality = self.total_quality
        clone.total_allocated_count = self.total_allocated_count
        clone.inplace_reuse = dict(self.inplace_reuse)
        clone.below_profile = {
            k: Profile(list(p.starts), list(p.labels))
            for k, p in self.below_profile.items()
        }
        clone.above_profile = {
            k: Profile(list(p.starts), list(p.labels))
            for k, p in self.above_profile.items()
        }
        return clone
