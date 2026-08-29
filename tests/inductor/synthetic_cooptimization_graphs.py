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

"""Hand-built co-optimization graphs targeting structural coverage gaps.

The captured corpus (``cooptimization_captures.json``) is small -- four building
blocks, ``n`` <= 25, only ``sdpa`` with real pins or multi-op region structure
-- too thin for anything whose value only shows up on multiple regions or a
non-trivial ``n``. Real captures need the substrate and carry a ``solved``
reference validated by ``test_cooptimization_types.py``; these synthetic fixtures
deliberately do *not* -- they have no ground-truth engine output, so they
exercise only the **shape-invariant** SA guarantees (output contract,
>= baseline, determinism, region flood, recolor-finds-regions), never the
empirical schedule-quality assertions calibrated on the real corpus.

Each graph is a plain list of ``CoreDivisionBuffer`` built to isolate one
structure the captures under-cover:

* ``chain_long`` / ``chain_short`` -- one uniform region of very different
  op-count but similar coloring variety: the anchor-fairness case. Anchors are
  picked uniformly over ops, so a region draws recolor proposals in proportion to
  its op-count, which under-serves a small region carrying comparable coloring
  variety.
* ``wide_join`` -- a fan-in diamond whose sink cannot satisfy every parent's
  tiling at once: many accepted internal seams + ``num_children`` spill scaling.
* ``multi_region`` -- three uniform regions separated by boundary edges that
  carry only the trivial pair, so the flood stops for free at each boundary
  (boundaries emerge from the flood rather than being detected) and recolor has
  several regions to pick.
* ``k_split_consumers`` -- a clean producer read by reduction-split (K-split)
  consumers via the PSUM ring, plus a reduction-split *producer* that is gated
  out of LX because no child edge carries its partial-write index -- the
  compatibility gate, not the fixed pin.
* ``pinned_heavy`` -- resident buffers interleaved with several pinned buffers
  spanning distinct ``residency_reason`` values, so the fixed pin gate is
  exercised while pins still gate neighbors and contribute always-HBM traffic.
* ``big_chain`` -- a ~50-buffer multi-region graph for compile-time-budget /
  burst-scaling exercise and a heavier determinism load.
"""

from __future__ import annotations

from tests.inductor.cooptimization_capture_loader import CapturedGraph
from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType,
    CoreDivision,
    CoreDivisionBuffer,
)

_SIZE = 65536  # bytes; a plausible per-buffer footprint (mirrors capture scale)


# --- menu helpers ----------------------------------------------------------- #
def _trivial() -> CoreDivision:
    """Index-0 committed/seed division: whole-buffer, undivided."""
    return CoreDivision(output_splits={}, reduction_splits={})


def _osplit(k: int) -> CoreDivision:
    """An output split by ``k`` on dim 1 (output_partition == k)."""
    return CoreDivision(output_splits={1: k}, reduction_splits={})


def _rsplit(k: int) -> CoreDivision:
    """A reduction (K) split by ``k`` -- output_partition stays 1."""
    return CoreDivision(output_splits={}, reduction_splits={1: k})


# The standard three-entry output menu shared by most synthetic ops: index 0
# trivial (the seed), index 1 split-by-2, index 2 split-by-4. Tilings 1 and 2
# are the non-trivial recolor anchors.
_STD_MENU = [_trivial(), _osplit(2), _osplit(4)]


def _buf(
    name: str,
    parents: list[str],
    matches: dict[str, list[tuple[int, int]]],
    *,
    lifetime: tuple[int, int],
    menu: list[CoreDivision] | None = None,
    size: int = _SIZE,
    residency_reason: str | None = None,
    in_place_parents: list[str] | None = None,
) -> CoreDivisionBuffer:
    """One synthetic buffer. ``lifetime`` is ``(first_use, last_use)`` -- adjacent
    buffers are given overlapping windows so a tight capacity forces real packing
    pressure (spills / eligibility toggles), mirroring the captures' rolling
    ``uses``. ``matches`` maps each parent name to its ``(parent_idx, child_idx)``
    compatible menu-index pairs."""
    first, last = lifetime
    return CoreDivisionBuffer(
        name=name,
        size=size,
        uses=[first, last],
        first_use_is_read=bool(parents),
        in_place_parents=in_place_parents or [],
        residency_reason=residency_reason,
        core_divisions=list(menu if menu is not None else _STD_MENU),
        parents=list(parents),
        cd_parent_matches=dict(matches),
        # No synthetic buffer is a graph output, so every one is an intermediate
        # whose producer write residency saves (matches the captures, where
        # boundary_cost is zero for all but the single output buffer).
        boundary=BufferType.Intermediate,
    )


# Edge-compatibility shorthands (menu-index pairs on a parent->child edge):
_UNIFORM = [(0, 0), (1, 1), (2, 2)]  # a tiling propagates end to end
_BOUNDARY = [(0, 0)]  # only the trivial tiling crosses -> region boundary


# --- graph builders --------------------------------------------------------- #
def _chain(prefix: str, length: int) -> list[CoreDivisionBuffer]:
    """A uniform pointwise chain ``b0 -> b1 -> ... `` of ``length`` ops; every edge
    propagates any tiling (one region spanning the whole chain)."""
    bufs = [_buf(f"{prefix}0", [], {}, lifetime=(0, 1))]
    for i in range(1, length):
        prev = f"{prefix}{i - 1}"
        bufs.append(_buf(f"{prefix}{i}", [prev], {prev: _UNIFORM}, lifetime=(i, i + 1)))
    return bufs


def chain_long() -> list[CoreDivisionBuffer]:
    return _chain("c", 12)


def chain_short() -> list[CoreDivisionBuffer]:
    return _chain("c", 3)


def wide_join() -> list[CoreDivisionBuffer]:
    """``root`` fans out to ``m0..m5``, which all feed one ``sink``. Each middle is
    compatible with the root on the uniform relation, but the sink only matches
    each middle on a *distinct* tiling, so no single sink coloring satisfies every
    parent -- the flood assigns the sink once and accepts the rest as seams."""
    root = _buf("root", [], {}, lifetime=(0, 7))
    middles = [
        _buf(f"m{i}", ["root"], {"root": _UNIFORM}, lifetime=(1, 8)) for i in range(6)
    ]
    # sink[c] matches m_i on tiling index (i % 2) + 1 -> parents disagree.
    sink_matches = {f"m{i}": [(0, 0), ((i % 2) + 1, (i % 2) + 1)] for i in range(6)}
    sink = _buf("sink", [f"m{i}" for i in range(6)], sink_matches, lifetime=(8, 9))
    return [root, *middles, sink]


def multi_region() -> list[CoreDivisionBuffer]:
    """Three uniform 3-op chains joined by boundary edges (only the trivial pair
    crosses), so a split tiling floods within a region and stops at each boundary
    -- three distinct regions, not one."""
    bufs: list[CoreDivisionBuffer] = []
    prev_tail: str | None = None
    t = 0
    for r in range(3):
        for j in range(3):
            name = f"r{r}_{j}"
            if j == 0 and prev_tail is not None:
                # region entry: boundary edge from the previous region's tail.
                matches = {prev_tail: _BOUNDARY}
                parents = [prev_tail]
            elif j == 0:
                matches, parents = {}, []
            else:
                p = f"r{r}_{j - 1}"
                matches, parents = {p: _UNIFORM}, [p]
            bufs.append(_buf(name, parents, matches, lifetime=(t, t + 2)))
            t += 1
        prev_tail = f"r{r}_2"
    return bufs


def k_split_consumers() -> list[CoreDivisionBuffer]:
    """A clean ``producer`` read by two K-split consumers via the PSUM ring, plus a
    reduction-split ``kprod`` whose partial-write index never appears on its child
    edge -- so ``kprod`` is gated out of LX whenever it selects that division
    (the compatibility gate, not the fixed pin)."""
    # producer: clean output menu; consumers read it split on the reduction axis.
    producer = _buf("producer", [], {}, lifetime=(0, 3))
    # consumer menu: index 0 trivial, index 1 output-split-2, index 2 K-split-2.
    kmenu = [_trivial(), _osplit(2), _rsplit(2)]
    # producer's clean index 0 is compatible with the consumer's trivial (0,0) and
    # its K-split read (0,2): a K-split reading a clean parent via the ring.
    c0 = _buf(
        "kcons0",
        ["producer"],
        {"producer": [(0, 0), (0, 2)]},
        lifetime=(1, 2),
        menu=kmenu,
    )
    c1 = _buf(
        "kcons1",
        ["producer"],
        {"producer": [(0, 0), (0, 2)]},
        lifetime=(2, 3),
        menu=kmenu,
    )
    # kprod: a producer with a reduction-split menu entry. Its child edge only
    # carries a pair for its clean index 0, so choosing the reduction index (2)
    # leaves it with no compatible child pair -> gated out of LX.
    kprod = _buf("kprod", [], {}, lifetime=(3, 5), menu=kmenu)
    tail = _buf("ktail", ["kprod"], {"kprod": [(0, 0), (1, 1)]}, lifetime=(4, 5))
    return [producer, c0, c1, kprod, tail]


def pinned_heavy() -> list[CoreDivisionBuffer]:
    """Resident ops interleaved with several pinned buffers spanning distinct
    ``residency_reason`` values. Pins still gate neighbors via
    ``cd_parent_matches`` and contribute always-HBM traffic."""
    reasons = ["extern kernel user", "mutation target", "graph output (no clone)"]
    bufs = [_buf("head", [], {}, lifetime=(0, 1))]
    prev = "head"
    t = 1
    for i in range(3):
        pin = _buf(
            f"pin{i}",
            [prev],
            {prev: _UNIFORM},
            lifetime=(t, t + 2),
            residency_reason=reasons[i],
        )
        cur = _buf(
            f"res{i}",
            [f"pin{i}"],
            {f"pin{i}": _UNIFORM},
            lifetime=(t + 1, t + 3),
        )
        bufs.extend([pin, cur])
        prev = f"res{i}"
        t += 2
    return bufs


def big_chain() -> list[CoreDivisionBuffer]:
    """~48 buffers across several regions -- compile-time-budget / burst-scaling
    exercise and a heavier determinism load."""
    bufs: list[CoreDivisionBuffer] = []
    prev_tail: str | None = None
    t = 0
    for r in range(8):
        for j in range(6):
            name = f"g{r}_{j}"
            if j == 0 and prev_tail is not None:
                matches, parents = {prev_tail: _BOUNDARY}, [prev_tail]
            elif j == 0:
                matches, parents = {}, []
            else:
                p = f"g{r}_{j - 1}"
                matches, parents = {p: _UNIFORM}, [p]
            bufs.append(_buf(name, parents, matches, lifetime=(t, t + 3)))
            t += 1
        prev_tail = f"g{r}_5"
    return bufs


# name -> [CapturedGraph] (single graph per case), mirroring ``load_captures``'s shape
# so the SA test harness can chain synthetic and captured cases uniformly.
def synthetic_graphs() -> dict[str, list[CapturedGraph]]:
    builders = {
        "chain_long": chain_long,
        "chain_short": chain_short,
        "wide_join": wide_join,
        "multi_region": multi_region,
        "k_split_consumers": k_split_consumers,
        "pinned_heavy": pinned_heavy,
        "big_chain": big_chain,
    }
    return {name: [CapturedGraph(buffers=build())] for name, build in builders.items()}


if __name__ == "__main__":
    for case, graphs in synthetic_graphs().items():
        g = graphs[0]
        pinned = sum(1 for b in g.buffers if b.residency_reason is not None)
        anchors = sum(
            1
            for b in g.buffers
            if any(cd.output_partition > 1 for cd in b.core_divisions)
        )
        print(
            f"{case:20} {len(g.buffers):3d} buffers  "
            f"{pinned:2d} pinned  {anchors:3d} anchor-eligible"
        )
