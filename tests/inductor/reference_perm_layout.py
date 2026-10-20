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


"""A from-scratch permutation layout solver, as a test oracle.

A permutation fixes an allocation order;
:class:`ReferencePermutationBasedLayoutSolver` places the buffers in that
order by scanning, for each buffer, every earlier-placed buffer it overlaps,
and rebuilds every address from scratch on each operation. The shipped packer,
``torch_spyre._C.NativePermutationLayoutSolver``, does the same asymptotic work
-- it too re-places from scratch, quadratically in the worst case -- but with
far better constants. This class exists only to say what the answer should be,
so it takes none of those shortcuts and its correctness is meant to be evident
from reading it rather than argued for.

This is test code: nothing under ``torch_spyre/`` may import it.
"""

from typing import Optional
import math

from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer,
    check_in_place_parent_is_read,
)

# The native packer holds every tick in an int64, so a plan it is meant to
# agree with cannot carry a use it could not represent.
_INT64_MAX = 2**63 - 1


def _quality_for(buf: LifetimeBoundBuffer, size: int) -> float:
    """The :func:`buffer_quality` value ``buf`` would have at footprint ``size``.

    Factored out so a plan can re-score a buffer after :meth:`resize` without
    mutating the shared buffer object: the use-weight is a function of ``buf``'s
    access pattern only, so only the size varies.
    """
    return (len(buf.uses) + (0.0 if buf.first_use_is_read else 0.5)) * size


def buffer_quality(buf: LifetimeBoundBuffer) -> float:
    """The contribution buffer ``buf`` makes to a plan's :meth:`quality` when it
    is fully allocated below capacity.

    Weights the buffer's size by how heavily it is used: each access counts
    once, plus an extra half for a buffer whose first access is a write (a
    computed buffer, ``first_use_is_read`` False) since its initial store also
    touches the slot. Formally
    ``(len(buf.uses) + (0 if buf.first_use_is_read else 0.5)) * buf.size``.
    """
    return _quality_for(buf, buf.size)


# ===========================================================================
# Permutation-based layout solvers
# ===========================================================================


class ReferencePermutationBasedLayoutSolver:
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

    Every operation re-places every buffer from scratch: correctness is meant
    to be evident rather than argued for, so nothing here is maintained
    incrementally and no state survives a move.

    Args:
        buffers: The buffers to place. Indices into this list are the values
            used in ``permutation`` and as keys throughout the plan.
        permutation: Allocation order as a permutation of
            ``range(len(buffers))``.
        capacity: Scratchpad capacity in bytes.
        alignment: Byte alignment boundary for placed addresses. Defaults to 128
            (one Spyre stick).
        eligible: Optional per-buffer LX-eligibility flags (indexed like
            ``buffers``). ``None`` means every buffer is eligible -- the layout-
            only default, byte-for-byte identical to the pre-eligibility solver.
            An ineligible buffer keeps its permutation slot but is routed to HBM:
            it is transparent to the stack (contributes no address, no quality,
            and nothing rests on it). Toggle it live with :meth:`set_eligible`.
    """

    def __init__(
        self,
        buffers: list[LifetimeBoundBuffer],
        permutation: list[int],
        capacity: int,
        alignment: int = 128,
        eligible: Optional[list[bool]] = None,
    ):
        n = len(buffers)
        # Checked in the native constructor's order, so that a plan wrong in more
        # than one way reports the same first complaint from either packer.
        if alignment <= 0:
            raise ValueError("alignment must be positive")
        if sorted(permutation) != list(range(n)):
            raise ValueError("permutation must be a permutation of range(len(buffers))")
        for i, buf in enumerate(buffers):
            if buf.size < 0:
                raise ValueError(f"buffer {i}: size must be non-negative")
            # LifetimeBoundBuffer deliberately allows empty uses (registration can
            # precede them), but a packer reads start_time/end_time off them.
            if not buf.uses:
                raise ValueError("buffer uses must be non-empty")
            # Python has no overflow here, but the native packer derives
            # end_time as uses[-1] + 1 in int64. Rejected in both so the choice
            # of packer stays invisible.
            if buf.uses[-1] >= _INT64_MAX:
                raise ValueError(f"buffer {i}: last use must be below INT64_MAX")
        self.buffers = buffers
        self.permutation = list(permutation)
        self.capacity = capacity
        self.alignment = alignment
        self._name_to_idx = {buf.name: i for i, buf in enumerate(buffers)}
        # Names are the identity in-place parents are resolved by, so a
        # duplicate makes ``in_place_parents=["a"]`` ambiguous -- the dict
        # comprehension above silently keeps the last such buffer. Reject
        # instead of resolving to an arbitrary one.
        if len(self._name_to_idx) != n:
            raise ValueError("buffer names must be unique")

        # Per-buffer size as a flat list, for fast access in the placement hot
        # loop (avoids a dataclass attribute lookup per candidate). Mutable via
        # :meth:`resize` (which never touches the shared buffer objects), so
        # :meth:`copy` deep-copies it.
        self._sizes = [buf.size for buf in buffers]

        # Per-buffer quality contribution (use-weighted size) as a flat list,
        # summed into total_quality for every fully-allocated buffer. Mutable via
        # :meth:`resize` (tracks ``_sizes``); deep-copied by :meth:`copy`.
        self._qualities = [buffer_quality(buf) for buf in buffers]

        # Per-buffer LX-eligibility. An ineligible buffer holds its slot but is
        # skipped by the placer and excluded from the contact order (routed to
        # HBM). Mutable via :meth:`set_eligible`; deep-copied by :meth:`copy`.
        # Defaults to all-True, which reproduces the layout-only solver exactly.
        self._eligible = [True] * n if eligible is None else list(eligible)
        if len(self._eligible) != n:
            raise ValueError("eligible must have one flag per buffer")

        # Per-buffer set of possible in-place partners (its declared parents and
        # the children that declare it). Static -- a function of names and
        # in_place_parents -- so computed once and consulted instead of probing
        # every candidate during placement. See _placement_decision.
        self._inplace_partners = self._compute_inplace_partners()

        # Internal address per buffer index; None means evicted (does not fit
        # below capacity). Populated by _build and kept in sync by swap. Not
        # written to buffer objects until finalize.
        self.addresses: list[Optional[int]] = [0] * n

        # Sum of buffer_quality(buf) over all fully-allocated buffers (address +
        # size <= capacity), exposed via quality(); and the count of those
        # buffers, exposed via count_allocated(). Both recomputed by _build.
        self.total_quality: float = 0.0
        self.total_allocated_count: int = 0

        self._build()

    # --- shared helpers -----------------------------------------------------

    def _check_index(self, idx: int, method: str) -> None:
        """Reject a buffer index outside ``range(len(buffers))``.

        Python would read ``idx=-1`` as the last buffer and silently mutate (or
        report on) that one, where the native packer raises. The two are meant to
        be interchangeable, so raise its ``ValueError``, with its message.
        """
        if not 0 <= idx < len(self.buffers):
            raise ValueError(f"{method} index out of range")

    def _check_swap_index(self, i: int) -> None:
        """Reject a swap position with no successor to swap with.

        Valid positions are ``0 .. len(buffers) - 2``. Lives on the base but is
        called from each concrete :meth:`swap`, which is all the base declares.
        Same reasoning as :meth:`_check_index`: unchecked, ``swap(-1)`` exchanges
        the last permutation entry with the first.
        """
        if not 0 <= i < len(self.buffers) - 1:
            raise ValueError("swap index out of range")

    def _check_indices(self, i: int, j: int, method: str) -> None:
        """:meth:`_check_index` for a method taking two buffer indices. Called by
        every :meth:`rotate` before its ``i == j`` no-op, so an out-of-range
        ``rotate(999, 999)`` raises instead of returning 0.0 (as in the native
        packer)."""
        n = len(self.buffers)
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"{method} index out of range")

    def resize(self, idx: int, new_size: int) -> float:
        """Change buffer ``idx``'s footprint to ``new_size`` in place and
        re-place. Returns the change in :meth:`quality` (new minus old).

        The buffer's lifetime and permutation slot are unchanged, so only its
        size-derived footprint and quality move; the shared buffer object is
        **not** mutated (the plan tracks size in ``_sizes``). One of the two
        packer extensions the co-optimizing engine needs, and harmless to
        placement-only use.

        ``new_size`` must be non-negative, as in the native packer: a negative
        footprint puts a buffer's top below its own address, breaking the "rest
        on the max top" invariant placement is built on. Zero is allowed: the
        allocator clamps unsized entries to 0.
        """
        self._check_index(idx, "resize")
        if new_size < 0:
            raise ValueError("resize size must be non-negative")
        old_total = self.total_quality
        self._sizes[idx] = new_size
        self._qualities[idx] = _quality_for(self.buffers[idx], new_size)
        self._build()
        return self.total_quality - old_total

    def set_eligible(self, idx: int, flag: bool) -> float:
        """Toggle buffer ``idx``'s LX-eligibility and re-place. Returns the
        change in :meth:`quality` (new minus old); a no-op (returns ``0.0``) when
        the flag is unchanged.

        An ineligible buffer keeps its permutation slot but is routed to HBM
        (transparent to the stack). The other co-optimization packer extension:
        the SA engine flips this as a division change makes a buffer's tiling edge
        (in)compatible.
        """
        # Before the unchanged-flag no-op, so an out-of-range ``set_eligible``
        # cannot return 0.0 instead of raising (as in the native packer).
        self._check_index(idx, "set_eligible")
        if self._eligible[idx] == flag:
            return 0.0
        old_total = self.total_quality
        self._eligible[idx] = flag
        self._build()
        return self.total_quality - old_total

    def rotate(self, i: int, j: int) -> float:
        """Modify the permutation by taking ``self.permutation[i]`` out of the permutation and
        reinserting it at position ``j``. Returns the change in :meth:`quality` caused by the
        rotation (new minus old)."""
        self._check_indices(i, j, "rotate")
        delta = 0.0
        if i < j:
            for k in range(i, j):
                delta += self.swap(k)
        elif j < i:
            for k in range(i - 1, j - 1, -1):
                delta += self.swap(k)
        return delta

    def _align_up(self, addr: int) -> int:
        """Round ``addr`` up to the next multiple of ``self.alignment``.

        Integer ceiling division, not ``math.ceil(addr / alignment)``: the float
        form loses precision above ``2**53`` and there *under*-aligns, handing
        back an address below ``addr`` (and so two live buffers the same slot).
        The C++ packer computes this exactly, so the float form was also the one
        place the two could disagree on in-range input.
        """
        return -(-addr // self.alignment) * self.alignment

    def _top(self, idx: int) -> Optional[int]:
        """Return ``address + size`` for a placed buffer (its exclusive top), or
        ``None`` if ``idx`` is evicted (has no address)."""
        if self.addresses[idx] is None:
            return None
        return self.addresses[idx] + self._sizes[idx]  # type: ignore

    def top_or_inf(self, idx: int) -> float:
        """:meth:`_top` as a float, with ``inf`` for an evicted buffer.

        The public form the annealing search sorts on: an evicted buffer sorts as
        if it sat arbitrarily high, so it is treated as above any placed buffer
        and never reordered below one. Lives on the packer (rather than in the
        search, where it used to read ``buffers[idx].size``) so that it reads the
        plan-local ``_sizes``, which :meth:`resize` mutates.
        """
        self._check_index(idx, "top_or_inf")
        top = self._top(idx)
        return math.inf if top is None else float(top)

    def is_fully_allocated(self, idx: int) -> bool:
        """True if buffer ``idx`` has an address (and so fits below ``capacity``).

        ``None`` is the single source of truth for eviction: a buffer carries a
        concrete address iff it fits entirely below ``capacity`` (the capacity
        gate lives in :meth:`_placement_decision`), so "has an address" and
        "fully allocated" coincide.
        """
        self._check_index(idx, "is_fully_allocated")
        return self._is_allocated(idx)

    def _is_allocated(self, idx: int) -> bool:
        """:meth:`is_fully_allocated` without the bounds check, for the internal
        callers that hold an index they just derived. Mirrors the native packer's
        split between the exported accessor and the raw predicate."""
        return self.addresses[idx] is not None

    def overlaps(self, i: int, j: int) -> bool:
        """True if buffers ``i`` and ``j`` are alive at a common tick.

        Lifetimes are half-open intervals ``[start_time, end_time)``, so an
        in-place parent and child (``parent.end_time == child.start_time + 1``)
        overlap at exactly that boundary tick (``child.start_time``).
        """
        self._check_indices(i, j, "overlaps")
        return self._overlaps(i, j)

    def _overlaps(self, i: int, j: int) -> bool:
        """:meth:`overlaps` without the bounds check. Same reasoning as
        :meth:`_is_allocated`."""
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
                    # A write-only computed parent has nothing to hand over, so
                    # the pair is not expressible rather than merely unprofitable.
                    # Checked here because this is the one place that resolves
                    # declared pairs. Of the three in-place invariants checked
                    # by ``_check_in_place_relationships`` only the size one is
                    # a placement-time gate here rather than a precondition --
                    # an oversized child is simply not placed in-place, see
                    # ``_can_inplace`` -- so checking it would reject plans this
                    # solver handles.
                    check_in_place_parent_is_read(self.buffers[parent], buf.name)
                    # Single-tick handoff: a valid in-place pair overlaps at
                    # exactly one tick, the child's first. Both in-place
                    # candidate generators upstream enforce it
                    # (``allocator._determine_in_place`` and
                    # ``_determine_in_place_division_invariant``), and
                    # ``_check_in_place_relationships`` re-checks it. The
                    # native packer relies on it and rejects a violation here,
                    # so the oracle must reject the same plans.
                    if (
                        self.buffers[parent].end_time
                        != self.buffers[child].start_time + 1
                    ):
                        raise ValueError(
                            f"in-place pair ({self.buffers[parent].name}, "
                            f"{buf.name}) must hand off at a single tick: parent "
                            f"end_time {self.buffers[parent].end_time} != child "
                            f"start_time {self.buffers[child].start_time} + 1"
                        )
                    partners[child].add(parent)
                    partners[parent].add(child)
        return partners

    def _can_inplace(self, parent: int, child: int) -> bool:
        """True if ``child`` is allowed to share ``parent``'s address.

        A child may only reuse a parent's storage if it fits within it; a
        larger child would still need the parent's inputs while writing past
        the parent's footprint.

        Reads the plan-local ``_sizes`` (not ``buffers[...].size``) so a
        :meth:`resize` that crosses the fit boundary flips in-place legality.
        """
        return self._sizes[child] <= self._sizes[parent]

    def _placement_decision(
        self, idx: int, candidates: list[int]
    ) -> tuple[Optional[int], Optional[int]]:
        """Decide ``idx``'s address given the buffers it must sit on top of.

        ``candidates`` are the already-placed buffer indices that overlap
        ``idx`` in time -- here, *all* of them. (The native packer passes only
        ``idx``'s direct below-neighbours, which yields the same decision: the
        highest top among them is the same, and that is all the rule depends
        on.)

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
        max_top = max(addr[p] + sizes[p] for p in candidates)  # type: ignore
        # Try to drop into an in-place partner's slot. At most one partner can
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
                    (addr[q] + sizes[q] for q in candidates if q != partner),  # type: ignore
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

    def _build(self) -> None:
        """Place every buffer from scratch, in permutation order.

        Each buffer is placed on top of every earlier-placed, time-overlapping
        one -- the whole specification, scanned directly.
        """
        n = len(self.buffers)
        self.addresses = [0] * n
        self.total_quality = 0.0
        self.total_allocated_count = 0
        for pos in range(n):
            idx = self.permutation[pos]
            # An ineligible buffer is routed to HBM: no address, no quality, and
            # excluded from every later buffer's candidate set (so it is
            # transparent to the stack, not an evicting support).
            if not self._eligible[idx]:
                self.addresses[idx] = None
                continue
            prior = self.permutation[:pos]
            candidates = [
                p for p in prior if self._overlaps(idx, p) and self._eligible[p]
            ]
            self.addresses[idx] = self._address_from_candidates(idx, candidates)
            if self._is_allocated(idx):
                self.total_quality += self._qualities[idx]
                self.total_allocated_count += 1

    def swap(self, i: int) -> float:
        """Swap permutation entries ``i``/``i+1`` and rebuild from scratch."""
        self._check_swap_index(i)
        old_total = self.total_quality
        perm = self.permutation
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        self._build()
        return self.total_quality - old_total
