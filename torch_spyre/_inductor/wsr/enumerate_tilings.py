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

"""The coarse tilings an op could take, as a space and as a list.

:class:`TilingSpace` answers "may this op take *this* tiling" one spec at a
time, and :func:`enumerate_tile_options` is the cross product over it --
exhaustive and deterministic, so a solver consuming the list has a complete
candidate set while one that generates specs as it goes gets the same verdicts
without materializing them. Because the enumeration *is* the cross product, a
spec the space admits is one the list would have carried, apart from the
``max_options`` truncation the list applies and the space does not.

Level *order* is canonical, which is what makes that equality hold in both
directions: an output spec's levels ascend by ``host_dim``, and a spec in any
other order is refused rather than admitted. Nest order is consequently not a
decision variable -- no term in the cost model depends on it, so carrying both
orders would double the state space, split a tiling group on a distinction
without a difference, and buy nothing. Reintroduce it alongside a term that
prices it, not before.


The strategy is **exact divisors**: a split count is
admissible only if it divides its dim's extent exactly, because coarse tiling
emits equal-sized loop tiles. This is not a new rule -- it is exactly what
:func:`_split_candidates_for_host_dim` already computes for the span-overflow
path -- so this module *reuses* those predicates rather than restating them.
Two consequences of the strategy are visible to users and neither is a
bug: a prime extent is effectively untileable (its only divisors are 1 and
itself, and the self-split is a unit tile), and padding a dim to a composite
extent is the lever that opens divisors.

Reductions are enumerated here too, but **single-level only**: never nested with
an output axis and never two reduction dims at once. Those shapes are the ones
the reduction-tiling path gets wrong today, so the enumerator must not offer
them. The space itself is output-only; the list appends the reduction options.

Two deliberate departures from the span-overflow path:

* **It fails closed on the stick host dim.** ``_split_candidates_for_host_dim``
  admits a *whole-stick* split of the stick-carrying dim, but applying such a
  tiling produces an undersized boundary ``full_buf`` and silently wrong results
  (the stick-dim upscale bug). So this enumerator drops the stick dim entirely
  rather than trusting the stick-alignment predicate the 448 path relies on.
  It names the stick dim by coordinate identity only, and offers an op it
  cannot name one for no output tiling: a size-based guess that missed would
  leave the true stick dim offered and alignment-checked against the guess.
* **It does not run on span pressure.** ``_candidate_host_dims`` only offers dims
  that relieve an overflowing span; enumerating from it would return the untiled
  option alone for an op under no pressure, and the solver would never tile it.
  This enumerates every legally-splittable non-stick dim regardless of
  pressure.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Sequence

from torch._inductor.ir import ComputedBuffer, Reduction

from .. import config
from ..errors import Unsupported
from ..ir import FixedTiledLayout
from ..logging_utils import get_inductor_logger
from ..pass_utils import host_coordinates
from ..scratchpad.plan_solver import TileAxis, TileSpec
from .coarse_tile import _stick_host_dim, reduction_loop_vars
from .span_overflow_hint_analysis import (
    _MAX_AUTO_TILE_SPLIT_COUNT,
    _MAX_SPLITS_PER_DIM,
    _input_read_deps,
    _layout_has_static_span_metadata,
    _post_tile_stick_alignment_error,
    _split_candidates_for_host_dim,
)

logger = get_inductor_logger("wsr.enumerate_tilings")

# Default caps for the enumerator. Split counts stay bounded by
# ``_MAX_AUTO_TILE_SPLIT_COUNT`` (imported, NOT migrated to config.py);
# these two bound the *shape* of the option set, not individual splits.
_MAX_TILE_DIMS = 2
_MAX_TILE_OPTIONS = 64


def _output_stick_host_dim(op: ComputedBuffer) -> int | None:
    """The op's within-stick output host dim, or ``None`` when coordinate
    identity cannot name it (see the module docstring)."""
    layout = op.get_layout()
    if getattr(layout, "device_layout", None) is None:
        return None
    return _stick_host_dim(op, layout.device_layout)


def _output_split_counts(op: ComputedBuffer, host_dim: int) -> list[int]:
    """Legal split counts (> 1) for output ``host_dim``, exact divisors only.

    Delegates to ``_split_candidates_for_host_dim`` -- which already composes
    exact divisibility, ``_MAX_AUTO_TILE_SPLIT_COUNT``, the Reduction unit-extent
    rejection, and both stick-alignment checks -- and drops the trivial ``1``.
    """
    try:
        candidates = _split_candidates_for_host_dim(op, host_dim)
    except Unsupported:
        return []
    return [s for s in candidates if s > 1]


def _unit_tile_breaks_a_reader(
    op: ComputedBuffer, host_dim: int, count: int, readers: Sequence[ComputedBuffer]
) -> bool:
    """Whether tiling ``host_dim`` ``count`` ways leaves a 1-extent tile that a
    reader views through another rank or shape, which ``_squeezed_retile_dims``
    refuses for a reader outside the tiling group. Which readers end up outside
    is not known yet, so any reader counts."""
    ranges = tuple(op.data.ranges)
    if int(ranges[host_dim]) // count != 1:
        return False
    for reader in readers:
        reader_ranges = tuple(reader.data.ranges)
        if len(reader_ranges) < len(ranges) or (
            reader_ranges[host_dim] != 1 and reader_ranges != ranges
        ):
            return True
    return False


def _reduction_split_cuts_input_stick(op: ComputedBuffer, red_var, split: int) -> bool:
    """Return True if tiling reduction loop var ``red_var`` by ``split`` cuts a
    physical stick in any input the reduction dim controls.

    The reduction analogue of ``_input_stick_alignment_error``: it uses the
    reduction loop var (from :func:`reduction_loop_vars`) as the target symbol
    instead of an output host dim's symbols, and reuses the same low-level
    helpers. Fails closed -- an input whose coordinates cannot be derived is
    treated as cut.
    """
    for dep, layout in _input_read_deps(op):
        if not _layout_has_static_span_metadata(layout):
            continue
        try:
            input_coords = host_coordinates(layout, dep, None)
        except (TypeError, ValueError, RuntimeError, KeyError, IndexError):
            return True
        for input_host_dim, coord in enumerate(input_coords):
            if red_var in coord.free_symbols:
                if (
                    _post_tile_stick_alignment_error(layout, input_host_dim, split)
                    is not None
                ):
                    return True
    return False


def _reduction_split_counts(op: ComputedBuffer, red_pos: int) -> list[int]:
    """Legal split counts (> 1) for reduction dim ``red_pos``, exact divisors.

    Exact divisors of the reduction extent, minus the unit-tile split (rejected
    for Reduction ops, matching ``_split_candidates_for_host_dim``), minus any
    split that cuts an input stick, bounded by ``_MAX_AUTO_TILE_SPLIT_COUNT``.
    """
    reduction_ranges = list(getattr(op.data, "reduction_ranges", []))
    if red_pos >= len(reduction_ranges):
        return []
    try:
        full = int(reduction_ranges[red_pos])
    except (TypeError, ValueError):
        return []
    if full <= 1:
        return []
    try:
        red_var = reduction_loop_vars(op)[red_pos]
    except (IndexError, StopIteration, AssertionError):
        return []
    divisors = sorted(
        {
            d
            for i in range(1, math.isqrt(full) + 1)
            if full % i == 0
            for d in (i, full // i)
        }
    )
    legal: list[int] = []
    for split in divisors:
        if split <= 1:
            continue
        if full // split <= 1:  # unit-tile rejection (Reduction)
            continue
        if split > _MAX_AUTO_TILE_SPLIT_COUNT:
            continue
        if _reduction_split_cuts_input_stick(op, red_var, split):
            continue
        legal.append(split)
    return legal


def _canonical_key(spec: TileSpec) -> tuple:
    """Deterministic ordering key -- explicitly NOT ``_combo_cost``.

    Shallower nests first, then output axes before reduction axes, then by the
    axis tuple. Truncation from the tail therefore drops the deepest, most
    speculative options first; the untiled option is ranked and kept separately.
    """
    return (
        spec.depth,
        tuple((a.is_reduction, a.host_dim, a.count) for a in spec.axes),
    )


def _finalize_options(options: list[TileSpec], max_options: int) -> list[TileSpec]:
    """Dedup, canonically order, and truncate -- untiled first and never dropped."""
    seen: set[TileSpec] = set()
    unique: list[TileSpec] = []
    for spec in options:
        if spec not in seen:
            seen.add(spec)
            unique.append(spec)
    rest = sorted((s for s in unique if not s.is_untiled), key=_canonical_key)
    # The untiled option is mandatory and always leads, so truncation
    # from the tail can never drop it.
    result = [TileSpec()] + rest
    if max_options is not None and len(result) > max_options:
        result = result[:max_options]
    return result


@dataclasses.dataclass
class TilingSpace:
    """One op's legal coarse tilings as a space to move in, not a list.

    The output half of :func:`enumerate_tile_options`, asked one spec at a
    time: which dims are tileable, what counts each admits, and whether a proposed
    :class:`TileSpec` is legal. The enumeration is the cross product over
    exactly these answers, so a spec :meth:`admits` accepts is one the list
    would have carried -- generation changes when an option is materialized,
    not which options exist. The one asymmetry is deliberate: ``max_options``
    truncates the list and constrains the space not at all, which is the whole
    reason a search generates rather than enumerates.

    The domains are the *legal* counts per output dim, ``1`` excluded (a unit
    split is the untiled option, which every spec omits rather than spells
    out), already capped at ``max_splits_per_dim``. Empty for a dim that cannot
    be tiled at all, including the stick host dim, which
    :func:`build_tiling_space` drops. Output levels only: a reduction level is
    never admitted.
    """

    # Most output levels one spec may nest.
    max_dims: int

    output_counts: dict[int, list[int]]

    @property
    def output_dims(self) -> list[int]:
        """Tileable output host dims, ascending -- the order a spec's levels
        are enumerated and proposed in."""
        return sorted(self.output_counts)

    @property
    def is_empty(self) -> bool:
        """True when the untiled spec is the only one admitted."""
        return not self.output_counts

    def counts(self, host_dim: int) -> list[int]:
        """Legal split counts for output ``host_dim``."""
        return self.output_counts.get(host_dim, [])

    def admits(self, spec: TileSpec) -> bool:
        """Whether ``op`` may take ``spec``: output levels only, every level's
        count legal for its dim, no dim tiled twice, the canonical level order,
        and at most ``max_dims`` levels."""
        if spec.is_untiled:
            return True
        if any(axis.is_reduction for axis in spec.axes):
            return False
        if any(axis.count not in self.counts(axis.host_dim) for axis in spec.axes):
            return False
        if any(a.host_dim >= b.host_dim for a, b in zip(spec.axes, spec.axes[1:])):
            return False  # non-canonical level order (module docstring)
        return spec.depth <= self.max_dims

    def enumerate(self) -> list[TileSpec]:
        """Every spec this space admits, untiled first: the cross product over
        ``max_dims``-subsets of the tileable dims."""
        options: list[TileSpec] = [TileSpec()]
        per_dim = [(dim, self.output_counts[dim]) for dim in self.output_dims]
        for depth in range(1, min(self.max_dims, len(per_dim)) + 1):
            for combo in itertools.combinations(per_dim, depth):
                dims = [dim for dim, _ in combo]
                for counts in itertools.product(*(counts for _, counts in combo)):
                    options.append(
                        TileSpec(
                            tuple(
                                TileAxis(host_dim=dim, count=count)
                                for dim, count in zip(dims, counts)
                            )
                        )
                    )
        return options

    def neighbours(self, spec: TileSpec) -> list[TileSpec]:
        """The specs one level-edit away from ``spec``: change a level's count,
        remove a level, add an output level. Ordered and deduplicated, so a
        search proposing from this is deterministic; illegal results are dropped
        by :meth:`admits`.

        There is no reorder move, and an added level lands in *canonical*
        position rather than innermost: nest order is not a decision variable
        (see the module docstring), so a swap would be a free, always-accepted
        step buying no information. Every candidate is canonicalized on the way
        out, so even a non-canonical ``spec`` handed in from elsewhere has a way
        back into the space rather than being stranded.
        """
        axes = spec.axes
        out: list[TileSpec] = []
        for i, axis in enumerate(axes):
            for count in self.counts(axis.host_dim):
                if count != axis.count:
                    level = dataclasses.replace(axis, count=count)
                    out.append(TileSpec(axes[:i] + (level,) + axes[i + 1 :]))
        for i in range(len(axes)):
            out.append(TileSpec(axes[:i] + axes[i + 1 :]))
        tiled = {axis.host_dim for axis in axes if not axis.is_reduction}
        for host_dim in self.output_dims:
            if host_dim in tiled:
                continue
            for count in self.output_counts[host_dim]:
                out.append(TileSpec(axes + (TileAxis(host_dim=host_dim, count=count),)))
        return [
            candidate
            for candidate in dict.fromkeys(canonical_tiling(c) for c in out)
            if candidate != spec and self.admits(candidate)
        ]


def canonical_tiling(spec: TileSpec) -> TileSpec:
    """``spec`` with its levels in the order :meth:`TilingSpace.admits` requires
    -- output axes before reduction axes, each ascending by ``host_dim``."""
    return TileSpec(
        tuple(sorted(spec.axes, key=lambda a: (a.is_reduction, a.host_dim)))
    )


def _tileable(op: object) -> bool:
    """Whether ``op`` may be offered a coarse tiling at all. Not without a
    ``FixedTiledLayout``: no device layout to check stick alignment against
    (e.g. a KV-cache write). Not in a loop group already, which
    ``CoarseTilingPass`` would clobber by stamping ``dim_hints`` wholesale:
    ``dim_hints`` marks the ops the hint and span-overflow passes tiled,
    ``loop_info`` the members ``coarse_tile`` adds around them (a copy-out, a
    restickify of a tiled read)."""
    return (
        isinstance(op, ComputedBuffer)
        and isinstance(op.layout, FixedTiledLayout)
        and not getattr(op, "dim_hints", [])
        and getattr(op, "loop_info", None) is None
    )


def build_tiling_space(
    op: ComputedBuffer,
    *,
    max_dims: int = _MAX_TILE_DIMS,
    max_splits_per_dim: int = _MAX_SPLITS_PER_DIM,
    readers: Sequence[ComputedBuffer] = (),
) -> TilingSpace:
    """The :class:`TilingSpace` for ``op``; empty domains for an op that cannot
    be coarse-tiled at all (:func:`_tileable`), which is not an error -- untiled
    is always legal. ``readers`` are the ops that read ``op``'s output (see
    :func:`_unit_tile_breaks_a_reader`)."""
    output_counts: dict[int, list[int]] = {}
    stick_dim = _output_stick_host_dim(op) if _tileable(op) else None
    if stick_dim is not None:
        n_out = len(op.data.ranges) if hasattr(op.data, "ranges") else 0
        for host_dim in range(n_out):
            if host_dim == stick_dim:
                continue  # fail closed on the stick dim (module docstring)
            counts = [
                count
                for count in _output_split_counts(op, host_dim)
                if not _unit_tile_breaks_a_reader(op, host_dim, count, readers)
            ][:max_splits_per_dim]
            if counts:
                output_counts[host_dim] = counts
    return TilingSpace(max_dims=max_dims, output_counts=output_counts)


def enumerate_tile_options(
    op: ComputedBuffer,
    *,
    max_dims: int = _MAX_TILE_DIMS,
    max_splits_per_dim: int = _MAX_SPLITS_PER_DIM,
    max_options: int = _MAX_TILE_OPTIONS,
) -> list[TileSpec]:
    """Return the coarse tilings ``op`` could legally take, untiled first.

    The set always contains the empty (untiled) :class:`TileSpec` and
    every single- or nested-output tiling over exact divisors of non-stick
    output dims (up to ``max_dims`` dims tiled at once), plus every single-level
    reduction tiling when ``op`` is a Reduction and ``enable_reduction_tiling``
    is set. It never emits a nested output+reduction spec or a multi-reduction
    spec. Deterministic; the solver prices and chooses among these.

    A thin consumer of :class:`TilingSpace`, which holds the output predicates:
    this adds the reduction options, then orders, deduplicates and truncates.
    """
    options = build_tiling_space(
        op, max_dims=max_dims, max_splits_per_dim=max_splits_per_dim
    ).enumerate()
    if (
        _tileable(op)
        and isinstance(op.data, Reduction)
        and config.enable_reduction_tiling
    ):
        for red_pos in range(len(getattr(op.data, "reduction_ranges", []))):
            for count in _reduction_split_counts(op, red_pos)[:max_splits_per_dim]:
                options.append(
                    TileSpec(
                        (TileAxis(host_dim=red_pos, count=count, is_reduction=True),)
                    )
                )
    return _finalize_options(options, max_options)
