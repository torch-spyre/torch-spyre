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
import itertools
import math
import sympy

from enum import Enum


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
    and hashable. The empty spec is *untiled*, and is the inert default every
    :class:`CoreDivision` carries unless a solver chose otherwise.
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
    def level_counts(self) -> tuple[int, ...]:
        """Trip count of each level, outermost first: the loop nest alone,
        without the dims it tiles. ``host_dim`` is positional in each op's own
        output, so two specs can be equal and tile different dims of a buffer
        they share, or differ and tile the same one. Ops therefore share a loop
        group on equal nests (``derive_tiling_groups``), and whether a consumer
        reads its producer tile by tile is checked per edge."""
        return tuple(a.count for a in self.axes)

    @property
    def output_tile_count(self) -> int:
        """Product of the split factors over output (non-reduction) axes only.

        This is the factor by which a tiled op's own per-tile scratch shrinks,
        and so a factor a buffer's per-core footprint divides by. A
        reduction-tiled level does *not* shrink that buffer -- the op's own
        output is the accumulator, which keeps the full output extent -- so
        reduction axes are excluded here even though they count in
        :attr:`tile_count`.
        """
        return math.prod(a.count for a in self.axes if not a.is_reduction)

    def count_for(self, key: tuple[int, bool]) -> int:
        """Total split factor this spec puts on the axis ``key``, a
        ``(host_dim, is_reduction)`` pair; 1 for an axis it does not tile."""
        return math.prod(
            a.count for a in self.axes if (a.host_dim, a.is_reduction) == key
        )

    @property
    def is_clean(self) -> bool:
        """True when no reduction axis is tiled, so every tile produces a
        finished slice of the output rather than a partial accumulation.

        Landed with stage 1 and removed again by #4519 as unused; the
        solver-driven tiling path in ``GraphBuilder._tiling_candidates``
        is the caller that makes it live, and it is the single filter that keeps
        reduction tilings (numerically fragile on e.g. softmax's max/sum) out of
        the candidate menu."""
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

    ``splits`` is keyed by the producer's iteration symbols -- one entry per
    axis with a split factor.
    ``reduction_syms`` names the subset of those keys that split a reduction
    axis rather than an output axis.
    ``tiling`` pairs a coarse tiling onto this division as one candidate. The
    empty :class:`TileSpec` is untiled and inert.
    ``tile_splits`` is that tiling in the keys ``splits`` uses: one
    ``(iteration symbol, trip count)`` per level, outermost first. A tiling is
    a split in time the way a core division is a split in space, so a view can
    carry both and be matched the same way. Empty when untiled.
    """

    splits: dict[sympy.Symbol, int] = field(default_factory=dict)
    reduction_syms: frozenset[sympy.Symbol] = field(default_factory=frozenset)
    tiling: TileSpec = field(default_factory=TileSpec)
    tile_splits: tuple[tuple[sympy.Symbol, int], ...] = ()

    @property
    def cores_used(self) -> int:
        return math.prod(self.splits.values())

    @property
    def output_splits(self) -> dict[sympy.Symbol, int]:
        return {s: v for s, v in self.splits.items() if s not in self.reduction_syms}

    @property
    def reduction_splits(self) -> dict[sympy.Symbol, int]:
        return {s: v for s, v in self.splits.items() if s in self.reduction_syms}

    @property
    def output_partition(self) -> int:
        """How many cores the output buffer is sliced across."""
        return math.prod(self.output_splits.values())

    @property
    def label(self) -> str:
        """Human-readable rendering of this division's splits, e.g.
        ``"s0/4 ~s1/2"`` (output split by 4 on symbol 0, reduction split by 2
        on symbol 1), or ``"whole"`` for the untouched, undivided candidate."""
        out = ",".join(
            f"s{s}/{f}"
            for s, f in sorted(self.output_splits.items(), key=lambda i: str(i[0]))
        )
        red = ",".join(
            f"~s{s}/{f}"
            for s, f in sorted(self.reduction_splits.items(), key=lambda i: str(i[0]))
        )
        return " ".join(p for p in (out, red) if p) or "whole"


def sym_core_divs_for(
    name: str, divisions: Sequence[CoreDivision]
) -> dict[sympy.Symbol, sympy.Symbol]:
    """Symbolic stand-in for a chosen ``op_it_space_splits``: one symbol per
    stride coefficient seen across buffer ``name``'s candidate divisions, so
    the cost model can carry an undecided split as an unknown rather than a
    concrete value."""
    keys = dict.fromkeys(itertools.chain.from_iterable(cd.splits for cd in divisions))
    return {
        key: sympy.Symbol(f"split_{name}_{key}", integer=True, positive=True)
        for key in keys
    }


def sym_tile_counts_for(
    name: str, divisions: Sequence[CoreDivision]
) -> dict[tuple[int, bool], sympy.Symbol]:
    """Symbolic stand-in for a chosen ``tiling``, the coarse-tiling sibling
    of :func:`sym_core_divs_for`: one symbol per tiled axis (a ``(host_dim,
    is_reduction)`` pair) seen across the candidate divisions, bound to that
    candidate's count on the axis, 1 where it leaves the axis untiled. Keyed on
    the axis rather than the nesting level, so the symbols do not depend on how
    a candidate orders its levels -- the cost of a nest (a product of per-level
    trip counts) does not either. Empty while no candidate is tiled."""
    keys = dict.fromkeys(
        (axis.host_dim, axis.is_reduction)
        for cd in divisions
        for axis in cd.tiling.axes
    )
    return {
        (dim, is_red): sympy.Symbol(
            f"tile_{name}_{'r' if is_red else 'o'}{dim}",
            integer=True,
            positive=True,
        )
        for (dim, is_red) in keys
    }


def division_symbol(buffer_name: str) -> sympy.Symbol:
    """The objective symbol for ``buffer_name``'s chosen division index. One
    constructor so a term built from a buffer's *name* (a relayout copy pricing
    its source) and the engine's binding built from the node agree on name and
    assumptions."""
    return sympy.Symbol(f"division_{buffer_name}", integer=True, nonnegative=True)
