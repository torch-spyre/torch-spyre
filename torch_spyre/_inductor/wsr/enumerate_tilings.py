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

"""Enumerate the coarse tilings an op could take.

A *pure, unconsumed* enumerator for the coarse-tiling optimization.
Nothing calls it yet -- the solver that prices and chooses among these options
arrives separately. Its whole contract is to answer "what tilings could this op
legally take", exhaustively and deterministically, so the solver has a complete
candidate set to search.

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
    spec. Deterministic and unconsumed; the solver prices and
    chooses among these.
    """
    options: list[TileSpec] = [TileSpec()]
    if not isinstance(op, ComputedBuffer):
        return options

    # --- output-range options -------------------------------------------------
    stick_dim = _output_stick_host_dim(op)
    n_out = len(op.data.ranges) if hasattr(op.data, "ranges") else 0
    per_dim: list[tuple[int, list[int]]] = []
    for host_dim in range(n_out):
        if host_dim == stick_dim:
            continue  # fail closed on the stick dim (module docstring)
        counts = _output_split_counts(op, host_dim)[:max_splits_per_dim]
        if counts:
            per_dim.append((host_dim, counts))

    for k in range(1, min(max_dims, len(per_dim)) + 1):
        for dims_combo in itertools.combinations(per_dim, k):
            dim_indices = [d for d, _ in dims_combo]
            split_lists = [counts for _, counts in dims_combo]
            for splits in itertools.product(*split_lists):
                axes = tuple(
                    TileAxis(host_dim=d, count=s) for d, s in zip(dim_indices, splits)
                )
                options.append(TileSpec(axes))

    # --- reduction options: single-level only ---------------------------------
    if isinstance(op.data, Reduction) and config.enable_reduction_tiling:
        n_red = len(getattr(op.data, "reduction_ranges", []))
        for red_pos in range(n_red):
            for split in _reduction_split_counts(op, red_pos)[:max_splits_per_dim]:
                options.append(
                    TileSpec(
                        (TileAxis(host_dim=red_pos, count=split, is_reduction=True),)
                    )
                )

    return _finalize_options(options, max_options)
