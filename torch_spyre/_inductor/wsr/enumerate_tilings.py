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
them.

Two deliberate departures from the span-overflow path:

* **It fails closed on the stick host dim.** ``_split_candidates_for_host_dim``
  admits a *whole-stick* split of the stick-carrying dim, but applying such a
  tiling produces an undersized boundary ``full_buf`` and silently wrong results
  (the stick-dim upscale bug). So this enumerator drops the stick dim entirely
  rather than trusting the stick-alignment predicate the 448 path relies on.
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

from torch._inductor.ir import ComputedBuffer, Reduction

from .. import config
from ..errors import Unsupported
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
    _within_stick_host_dim,
)

# Default caps for the enumerator. Split counts stay bounded by
# ``_MAX_AUTO_TILE_SPLIT_COUNT`` (imported, NOT migrated to config.py);
# these two bound the *shape* of the option set, not individual splits.
_MAX_TILE_DIMS = 2
_MAX_TILE_OPTIONS = 64


def _output_stick_host_dim(op: ComputedBuffer) -> int | None:
    """The op's within-stick output host dim, or ``None`` when unresolved.

    Tiling this dim is excluded (see the module docstring): fail closed. Prefer
    the coordinate-identity resolver ``_stick_host_dim``; fall back to the
    size-based ``_within_stick_host_dim``.
    """
    layout = op.get_layout()
    if getattr(layout, "device_layout", None) is None:
        return None
    try:
        dim = _stick_host_dim(op, layout.device_layout)
    except (AttributeError, TypeError, ValueError, RuntimeError, KeyError, IndexError):
        dim = None
    if dim is None:
        try:
            dim = _within_stick_host_dim(layout)
        except (AttributeError, TypeError, ValueError, IndexError):
            dim = None
    return dim


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

    Everything :func:`enumerate_tile_options` needs, asked one spec at a time:
    which dims are tileable, what counts each admits, and whether a proposed
    :class:`TileSpec` is legal. The enumeration is the cross product over
    exactly these answers, so a spec :meth:`admits` accepts is one the list
    would have carried -- generation changes when an option is materialized,
    not which options exist. The one asymmetry is deliberate: ``max_options``
    truncates the list and constrains the space not at all, which is the whole
    reason a search generates rather than enumerates.

    The domains are the *legal* counts per dim, ``1`` excluded (a unit split is
    the untiled option, which every spec omits rather than spells out), already
    capped at ``max_splits_per_dim``. Empty for a dim that cannot be tiled at
    all, including the stick host dim, which :func:`build_tiling_space` drops.
    """

    op: ComputedBuffer
    # Most output levels one spec may nest. Reduction levels are capped at one
    # by :meth:`admits` instead, and cannot be nested with an output level.
    max_dims: int
    output_counts: dict[int, list[int]]
    reduction_counts: dict[int, list[int]]

    @property
    def output_dims(self) -> list[int]:
        """Tileable output host dims, ascending -- the order a spec's levels
        are enumerated and proposed in."""
        return sorted(self.output_counts)

    @property
    def is_empty(self) -> bool:
        """True when the untiled spec is the only one this op can take."""
        return not self.output_counts and not self.reduction_counts

    def counts(self, host_dim: int, is_reduction: bool = False) -> list[int]:
        """Legal split counts for one axis, in the frame ``is_reduction``
        selects (see :class:`TileAxis`)."""
        source = self.reduction_counts if is_reduction else self.output_counts
        return source.get(host_dim, [])

    def admits(self, spec: TileSpec) -> bool:
        """Whether ``op`` may take ``spec``: every level's count legal for its
        axis, no axis tiled twice, and the shape rules the enumerator applies
        -- at most ``max_dims`` output levels, and a reduction level only ever
        alone (never nested with an output axis, never two at once)."""
        if spec.is_untiled:
            return True
        axes = [(axis.is_reduction, axis.host_dim) for axis in spec.axes]
        if len(set(axes)) != len(axes):
            return False
        if any(
            axis.count not in self.counts(axis.host_dim, axis.is_reduction)
            for axis in spec.axes
        ):
            return False
        if not spec.is_clean:
            return spec.depth == 1
        return spec.depth <= self.max_dims

    def enumerate(self) -> list[TileSpec]:
        """Every spec this space admits, untiled first.

        Materializes what :meth:`admits` decides, so the two cannot drift: the
        output half is the cross product over ``max_dims``-subsets of the
        tileable dims, the reduction half is one level at a time.
        """
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
        for red_pos in sorted(self.reduction_counts):
            for count in self.reduction_counts[red_pos]:
                options.append(
                    TileSpec(
                        (TileAxis(host_dim=red_pos, count=count, is_reduction=True),)
                    )
                )
        return options

    def neighbours(self, spec: TileSpec) -> list[TileSpec]:
        """The specs one level-edit away from ``spec``: change a level's count,
        remove a level, add an output level innermost, swap two adjacent
        levels. Ordered and deduplicated, so a search proposing from this is
        deterministic; illegal results are dropped by :meth:`admits`.

        **Output axes only**, which is the v1 scope: no move ever *adds* a
        reduction level, so a search seeded at the untiled spec never reaches
        one (a spec that already carries one can still drop it or recount it).
        Reordering is by *adjacent* swaps alone -- they generate every
        permutation over repeated moves, and a one-move alphabet is what makes
        the walk local.
        """
        axes = spec.axes
        out: list[TileSpec] = []
        for i, axis in enumerate(axes):
            for count in self.counts(axis.host_dim, axis.is_reduction):
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
        for i in range(len(axes) - 1):
            out.append(TileSpec(axes[:i] + (axes[i + 1], axes[i]) + axes[i + 2 :]))
        return [
            candidate
            for candidate in dict.fromkeys(out)
            if candidate != spec and self.admits(candidate)
        ]


def build_tiling_space(
    op: ComputedBuffer,
    *,
    max_dims: int = _MAX_TILE_DIMS,
    max_splits_per_dim: int = _MAX_SPLITS_PER_DIM,
) -> TilingSpace:
    """The :class:`TilingSpace` for ``op``; empty domains for an op that cannot
    be coarse-tiled at all, which is not an error -- untiled is always legal."""
    output_counts: dict[int, list[int]] = {}
    reduction_counts: dict[int, list[int]] = {}
    if isinstance(op, ComputedBuffer):
        stick_dim = _output_stick_host_dim(op)
        n_out = len(op.data.ranges) if hasattr(op.data, "ranges") else 0
        for host_dim in range(n_out):
            if host_dim == stick_dim:
                continue  # fail closed on the stick dim (module docstring)
            counts = _output_split_counts(op, host_dim)[:max_splits_per_dim]
            if counts:
                output_counts[host_dim] = counts
        if isinstance(op.data, Reduction) and config.enable_reduction_tiling:
            for red_pos in range(len(getattr(op.data, "reduction_ranges", []))):
                counts = _reduction_split_counts(op, red_pos)[:max_splits_per_dim]
                if counts:
                    reduction_counts[red_pos] = counts
    return TilingSpace(
        op=op,
        max_dims=max_dims,
        output_counts=output_counts,
        reduction_counts=reduction_counts,
    )


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

    A thin consumer of :class:`TilingSpace`, which holds the predicates: this
    orders, deduplicates and truncates what the space enumerates.
    """
    space = build_tiling_space(
        op, max_dims=max_dims, max_splits_per_dim=max_splits_per_dim
    )
    return _finalize_options(space.enumerate(), max_options)
