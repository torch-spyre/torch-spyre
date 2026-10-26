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

"""Per-division-pair relayout candidate pricing for the CP-SAT solver.

Pins ``solver_relayout_pair_cost`` (the per-pair gate + price the CP-SAT
candidate tables are built from) and ``governing_run_split`` (the shared
geometry rule) against the same known views and fitted-law rows the reporting
path is pinned to, so the enumeration path and the reporting path cannot
drift apart.
"""

from types import SimpleNamespace
from collections import Counter

import pytest
from sympy import Eq, Mod, Piecewise, Symbol, floor

from torch_spyre._inductor.cost_model import OpFeatures, relayout_ns
from torch_spyre._inductor.dump_cost_model import governing_run_split
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.scratchpad import lx_relayout
from torch_spyre._inductor.scratchpad.allocator import _intern_view_group
from torch_spyre._inductor import config
from torch_spyre._inductor.scratchpad.lx_relayout import (
    core_domain_rejection,
    grouped_gather_rejection,
    RelayoutCandidate,
    solver_relayout_pair_cost,
)

_CORE_ID = Symbol("core_id")

# Logical [8, 256, 512] fp16 commits device layout [256, 8, 8, 64]: an M hint
# splits device dim 0, a B hint splits dim 2 (see lx_relayout_cost.md). These
# are the two views of the canonical measured configuration: producer
# {B:4, M:2} against consumer {B:2, M:4} on 8 cores.
_DEVICE_DIMS = [256, 8, 8, 64]
# Views carry their physical core count explicitly, as production views built
# by work_division_from_view do; the committed gate rejects views that omit it.
_SRC = PerCoreView(
    ((0, 2), (2, 4)),
    ((0, floor(_CORE_ID / 4)), (2, Mod(_CORE_ID, 4))),
    num_cores=8,
)
_DST = PerCoreView(
    ((0, 4), (2, 2)),
    ((0, floor(_CORE_ID / 2)), (2, Mod(_CORE_ID, 2))),
    num_cores=8,
)
_OUT_ELEMS = 256 * 8 * 8 * 64  # 2.10 MB at fp16
_DTYPE_BYTES = 2


def test_governing_run_split_picks_the_finer_view():
    # Source: innermost split dim 2, (8//4)*64 = 128 elems (256 B), split 4.
    # Destination: (8//2)*64 = 256 elems (512 B), split 2. Source governs.
    assert governing_run_split(_SRC, _DST, _DEVICE_DIMS) == (128, 4)
    # Direction-symmetric, as the law requires.
    assert governing_run_split(_DST, _SRC, _DEVICE_DIMS) == (128, 4)


def test_pair_cost_reproduces_the_canonical_row():
    # The 2.10 MB / 8 cores / run 256 B / split 4 row: direct measurement
    # 8.721 us, model 8.778 us. The enumeration price must be the same number
    # the reporting path produces (one law, two callers).
    cost = solver_relayout_pair_cost(
        _SRC, _DST, 8, _DEVICE_DIMS, _OUT_ELEMS, _DTYPE_BYTES
    )
    assert cost == pytest.approx(8778, rel=0.01)
    # Direction does not enter (measured: 8.721 vs 8.701 us reversed).
    reverse = solver_relayout_pair_cost(
        _DST, _SRC, 8, _DEVICE_DIMS, _OUT_ELEMS, _DTYPE_BYTES
    )
    assert reverse == pytest.approx(cost, rel=1e-9)


def test_equal_views_are_not_a_relayout():
    # An equal pair belongs to cd_parent_matches (free residency), never here.
    assert (
        solver_relayout_pair_cost(_SRC, _SRC, 8, _DEVICE_DIMS, _OUT_ELEMS, _DTYPE_BYTES)
        is None
    )


def test_grouped_gather_is_priced_like_a_permutation_of_the_same_tensor():
    # Destination with 4 distinct owners on 8 cores is a grouped gather
    # (#3440): each destination slice is assembled from two source slices and
    # held by two cores. The sweep measured the grouped shuffle at or below the
    # permutation law, so it is priced by that law on the finer side's geometry,
    # exactly as a permutation between the same two views would be.
    grouped = PerCoreView(((2, 4),), ((2, Mod(_CORE_ID, 4)),), num_cores=8)
    cost = solver_relayout_pair_cost(
        _SRC, grouped, 8, _DEVICE_DIMS, _OUT_ELEMS, _DTYPE_BYTES
    )
    assert cost is not None and cost > 0
    run, split = governing_run_split(_SRC, grouped, _DEVICE_DIMS)
    assert (run, split) == governing_run_split(_SRC, _SRC, _DEVICE_DIMS), (
        "the source is the finer side of a gather and governs the price"
    )
    assert cost == solver_relayout_pair_cost(
        _SRC, _DST, 8, _DEVICE_DIMS, _OUT_ELEMS, _DTYPE_BYTES
    ), "same tensor, same cores, same governing geometry: same price"


@pytest.mark.parametrize("source_cores", [2, 8])
def test_broadcast_report_uses_the_solver_price_domain(source_cores):
    from dataclasses import replace

    from torch._inductor.virtualized import V

    from torch_spyre._inductor import dump_cost_model as dcm

    source = PerCoreView(((0, source_cores),), ((0, _CORE_ID),), num_cores=source_cores)
    destination = PerCoreView(((0, 2),), ((0, Mod(_CORE_ID, 2)),), num_cores=8)
    # Each of the two destination pieces is held by four cores.
    plan = lx_relayout.LXRelayoutPlan(
        "source", ("consumer",), source, destination, source_cores
    )
    graph = SimpleNamespace(_spyre_lx_relayout_copies={plan.edge: ("copy", plan)})
    with V.set_graph_handler(graph):
        reported = dcm._relayout_features(
            SimpleNamespace(get_name=lambda: "copy"), [64], 2
        )

    assert reported is not None
    assert reported.relayout_cores == source_cores
    price = relayout_ns(reported)
    old_execution_domain = relayout_ns(replace(reported, cores=8, relayout_cores=0))
    assert price > 0
    assert price == pytest.approx(old_execution_domain * 8 / source_cores)
    solver_price = solver_relayout_pair_cost(
        source, destination, source_cores, [64], 64, 2, destination_num_cores=8
    )
    assert solver_price == pytest.approx(price)


@pytest.mark.parametrize("fragments", [8, 10, 11, 16])
def test_gather_fanin_respects_shuffle_register_budget(fragments):
    # A row/column split feeding a row-only matmul view. The innermost
    # physical split is only two, within the cost law's fitted range, while
    # each destination gathers `fragments` columns. Gemma's 10-way gather
    # fails Deeptools register initialization if admitted here.
    cores = 2 * fragments
    source = PerCoreView(
        ((0, fragments), (1, 2)),
        ((0, floor(_CORE_ID / 2)), (1, Mod(_CORE_ID, 2))),
        num_cores=cores,
    )
    destination = PerCoreView(((1, 2),), ((1, Mod(_CORE_ID, 2)),), num_cores=cores)
    dims = [4 * fragments, 256, 64]
    assert governing_run_split(source, destination, dims)[1] == 2
    supported = fragments <= 8
    assert (
        lx_relayout.movement_supported(source, destination, cores, cores) == supported
    )
    cost = solver_relayout_pair_cost(
        source, destination, cores, dims, 4 * fragments * 256 * 64, 2
    )
    assert (cost is not None) == supported


# A row-split normalization's fp16 output on 32 cores: logical [1, 512, 4096]
# in the stick-outer device layout [1, 64 stick columns, 512 rows, 64]. Its row
# reduction needs whole rows on a core, so each core owns 16 rows of every stick
# column: a run of 16 * 64 elements, governing split 32.
_NORM_DIMS = [1, 64, 512, 64]
_NORM_ELEMS = 64 * 512 * 64
_ROWS32 = PerCoreView(((2, 32),), ((2, _CORE_ID),), num_cores=32)
# An 8 x 4 tile of the same tensor on the same 32 cores (8 row groups, 4 stick
# column groups): a complete partition, so moving to it is a permutation.
_TILE8X4 = PerCoreView(
    ((1, 4), (2, 8)),
    ((1, floor(_CORE_ID / 8)), (2, Mod(_CORE_ID, 8))),
    num_cores=32,
)


def _law(run_elems, split, out_elems=_NORM_ELEMS, cores=32):
    return relayout_ns(
        OpFeatures(
            name="lx_relayout",
            is_reduction=False,
            out_elems=out_elems,
            cores=cores,
            dtype_bytes=_DTYPE_BYTES,
            args=[],
            is_lx_relayout=True,
            relayout_run_elems=run_elems,
            relayout_split=split,
        )
    )


@pytest.mark.parametrize("groups", [8])
def test_split_past_fitted_range_is_declined_not_clamped(groups):
    # A GROUPED movement keeps the old range [2, 8]: the law is fitted on
    # permutations, and a gather into a matmul's replicated operand measured
    # 1.8-16x above it at every split. Here the row-split producer feeds the
    # x view of a `groups` x (32 / groups) x 1 matmul cut: each row slice is held
    # by 32 / groups cores (fan-in 4 or 8), governing split 32. The emitter can
    # move it (movement_supported admits it), but the solver must not OFFER it
    # at a price the law was not fitted for, so the pair is declined outright.
    destination = PerCoreView(
        ((2, groups),), ((2, Mod(_CORE_ID, groups)),), num_cores=32
    )
    assert governing_run_split(_ROWS32, destination, _NORM_DIMS) == (1024, 32)
    assert lx_relayout.movement_supported(_ROWS32, destination, 32, 32)
    assert (
        solver_relayout_pair_cost(
            _ROWS32, destination, 32, _NORM_DIMS, _NORM_ELEMS, _DTYPE_BYTES
        )
        is None
    )


def test_split_product_must_equal_core_count():
    # solver_relayout_movement_supported requires split products equal to num_cores on
    # BOTH sides: the canonical pair (products 8) cannot host a permutation on
    # a 4-core solve. (A view whose slot expression is out of range for
    # num_cores is a caller error, not a declined pair - the allocator's
    # cores_used equality gate guarantees views are built for num_cores
    # before pricing.) The views omit their physical count so the split-product
    # gate, not the ownership-domain check, is what declines the pair.
    src = PerCoreView(_SRC.work_slice_dims, _SRC.core_to_slot)
    dst = PerCoreView(_DST.work_slice_dims, _DST.core_to_slot)
    assert (
        solver_relayout_pair_cost(src, dst, 4, _DEVICE_DIMS, _OUT_ELEMS, _DTYPE_BYTES)
        is None
    )


@config.patch({"lx_same_tile_group_relayout": True})
def test_coarse_tiled_endpoints_are_declined_at_the_edge_gate():
    """Cross-group endpoints must stop before any layout inspection.

    Real sibling and nested group IDs pin exact scope equality. Recording
    layout properties distinguish this gate from later layout rejection;
    untiled and equal-group controls must reach that later check.
    """
    from torch._inductor.ir import ComputedBuffer

    from torch_spyre._inductor.loop_info import CoarseTileInfo
    from torch_spyre._inductor.scratchpad.lx_relayout import (
        solver_relayout_edge_context,
    )

    def tiled(group):
        return CoarseTileInfo(
            loop_group_id=group,
            loop_count=[4] * len(group),
            loop_tiled_dims=[[0]] * len(group),
        )

    def probe(loop_info):
        class _Probe(ComputedBuffer):
            layout_accessed = False

            @property
            def layout(self):
                type(self).layout_accessed = True

        obj = _Probe.__new__(_Probe)
        obj.loop_info = loop_info
        return obj

    unknown = object()
    rejected = [
        (unknown, None),
        (None, unknown),
        (unknown, unknown),
        (tiled((0,)), None),
        (None, tiled((0,))),
        (tiled((0,)), tiled((1,))),
        (tiled((0,)), tiled((0, 1))),
        (tiled((0, 1)), tiled((0,))),
        (tiled((0, 1)), tiled((0, 2))),
    ]
    # No graph: the loop-info and layout checks precede its first use.
    for prod_info, cons_info in rejected:
        prod, cons = probe(prod_info), probe(cons_info)
        assert solver_relayout_edge_context(None, prod, cons, "buf0", {}) is None
        assert not type(prod).layout_accessed and not type(cons).layout_accessed, (
            "a cross-group endpoint reached checks past the loop_info gate"
        )

    # Distinct metadata objects with equal IDs must also pass the scope gate.
    for prod_info, cons_info in [
        (None, None),
        (tiled((0,)), tiled((0,))),
        (tiled((0, 1)), tiled((0, 1))),
    ]:
        prod, cons = probe(prod_info), probe(cons_info)
        assert solver_relayout_edge_context(None, prod, cons, "buf0", {}) is None
        assert type(prod).layout_accessed


# --- Solver gate vs committed gate ------------------------------------------
#
# The solver prices candidates before divisions exist, so it cannot call the
# committed path's gates; it mirrors them. The committed movement gate is being
# rewritten (#4284 replaces ``_compatible_partitions`` with
# ``movement_supported``; #3440 widens it to grouped gathers and broadcasts).
# The solver's gate must stay "never looser": everything it admits, the
# committed gate admits too, and the shapes the fitted law does not cover are
# declined whatever the committed gate says.


# Two owners replicated over eight cores (a broadcast source), and the same
# ownership as _SRC's dim-2 quarter spelled two ways over four cores.
_TWO_OWNERS_ON_8 = PerCoreView(((2, 2),), ((2, Mod(_CORE_ID, 2)),), num_cores=8)
_QUARTER_PLAIN = PerCoreView(((2, 4),), ((2, _CORE_ID),), num_cores=4)
_QUARTER_MOD = PerCoreView(((2, 4),), ((2, Mod(_CORE_ID, 4)),), num_cores=4)
_HALF_ON_4 = PerCoreView(
    ((0, 2), (2, 2)), ((0, floor(_CORE_ID / 2)), (2, Mod(_CORE_ID, 2))), num_cores=4
)


def test_same_ownership_in_two_spellings_is_not_a_relayout():
    """``core_id`` and ``core_id % 4`` name the same owners over four cores
    (#4283). Structural equality would call that movement; ownership equality
    must not, at every place the solver path compares views."""
    dims = [256, 8, 4, 64]
    assert _QUARTER_PLAIN != _QUARTER_MOD  # structurally distinct ...
    assert _QUARTER_PLAIN.same_partition(_QUARTER_MOD)  # ... same owners
    assert (
        solver_relayout_pair_cost(
            _QUARTER_PLAIN, _QUARTER_MOD, 4, dims, 256 * 8 * 4 * 64, _DTYPE_BYTES
        )
        is None
    )
    with pytest.raises(ValueError, match="equal"):
        RelayoutCandidate(
            parent="buf0",
            consumer="buf1",
            source_division=0,
            consumer_division=1,
            group=0,
            source_view=_QUARTER_PLAIN,
            destination_view=_QUARTER_MOD,
            cost_ns=1.0,
            source_footprint_bytes=1,
            destination_footprint_bytes=1,
        )


def test_candidate_without_a_measured_span_is_rejected():
    """The committed path declines a relayout whose span it cannot measure
    (#3440); the solver path must not carry such a candidate into a solve,
    where a zero-sized copy would place for free."""
    with pytest.raises(ValueError, match="no measured span"):
        RelayoutCandidate(
            parent="buf0",
            consumer="buf1",
            source_division=0,
            consumer_division=1,
            group=0,
            source_view=_SRC,
            destination_view=_DST,
            cost_ns=1.0,
            source_footprint_bytes=128,
            destination_footprint_bytes=0,
        )


# Two complete column slices on 2 cores, each fed to four of 8 destination cores
# (#3440's "larger domain" broadcast case), and the same ownership described on
# a core domain that is not the transfer's.
_TWO_ON_2 = PerCoreView(((0, 2),), ((0, Mod(_CORE_ID, 2)),), num_cores=2)
_FOUR_ON_8 = PerCoreView(
    ((0, 4),),
    (
        (
            0,
            Piecewise(
                (0, Eq(_CORE_ID, 0)), (1, _CORE_ID < 4), (2, Eq(_CORE_ID, 4)), (3, True)
            ),
        ),
    ),
    num_cores=8,
)
_TWO_ON_4 = PerCoreView(((0, 2),), ((0, Mod(_CORE_ID, 2)),), num_cores=4)
_TWO_ON_32 = PerCoreView(((0, 2),), ((0, floor(_CORE_ID / 16)),), num_cores=32)


def test_broadcast_to_a_larger_core_domain_is_admitted_and_priced():
    """A broadcast crosses core counts; the price takes the destination's for
    the gate and prices on the SOURCE's cores: the law's per-core bytes are then
    the source slice, which is what every destination core receives, and the
    sweep found broadcast time flat in fan-out."""
    assert lx_relayout.movement_supported(_TWO_ON_2, _FOUR_ON_8, 2, 8)
    dims = [256, 8, 8, 64]
    cost = solver_relayout_pair_cost(
        _TWO_ON_2, _FOUR_ON_8, 2, dims, 256 * 8 * 8 * 64, 2, destination_num_cores=8
    )
    assert cost is not None and cost > 0
    # Ownership described on another core domain than the transfer: declined.
    assert not lx_relayout.movement_supported(_TWO_ON_4, _TWO_ON_32, 2, 32)
    assert (
        solver_relayout_pair_cost(
            _TWO_ON_4,
            _TWO_ON_32,
            2,
            dims,
            256 * 8 * 8 * 64,
            2,
            destination_num_cores=32,
        )
        is None
    )


@pytest.mark.parametrize("fragments", [4, 8, 16])
def test_complete_partition_expansion_respects_fanin(fragments):
    """Complete 16-to-32 repartitions reach the fan-in guard, including eight."""
    source = PerCoreView(((0, 16),), ((0, _CORE_ID),), num_cores=16)
    rows = 16 // fragments
    destination = PerCoreView(
        ((0, rows), (1, 2 * fragments)),
        ((0, Mod(_CORE_ID, rows)), (1, floor(_CORE_ID / rows))),
        num_cores=32,
    )
    edges = lx_relayout.transfer_edges(
        dict(source.work_slice_dims),
        dict(destination.work_slice_dims),
        lx_relayout._core_slices(source, 16),
        lx_relayout._core_slices(destination, 32),
    )
    assert destination.split_product == 32
    assert set(Counter(src for src, _ in edges).values()) == {2 * fragments}
    assert set(Counter(dst for _, dst in edges).values()) == {fragments}
    assert lx_relayout.movement_supported(source, destination, 16, 32) == (
        fragments <= 8
    )


@pytest.mark.parametrize("columns,supported", [(8, True), (4, False)])
def test_expansion_rejects_replicated_multisource_destination(columns, supported):
    """Replication cannot use the complete-destination gather exception."""
    source = PerCoreView(((0, 16),), ((0, _CORE_ID),), num_cores=16)
    destination = PerCoreView(
        ((0, 4), (1, columns)),
        ((0, Mod(_CORE_ID, 4)), (1, Mod(floor(_CORE_ID / 4), columns))),
        num_cores=32,
    )
    edges = lx_relayout.transfer_edges(
        dict(source.work_slice_dims),
        dict(destination.work_slice_dims),
        lx_relayout._core_slices(source, 16),
        lx_relayout._core_slices(destination, 32),
    )
    # Both geometries pass uniformity and the register budget; only the
    # rejected arm has replicated destination slices (16 slices on 32 cores).
    assert set(Counter(src for src, _ in edges).values()) == {8}
    assert set(Counter(dst for _, dst in edges).values()) == {4}
    assert destination.split_product == 4 * columns
    assert lx_relayout.movement_supported(source, destination, 16, 32) == supported


@pytest.mark.parametrize("balanced", [True, False])
def test_expansion_rejects_nonuniform_fanout(balanced):
    """Keep coverage and fan-in valid while changing only source fan-out."""
    # Four slices on eight cores use the existing replicated-broadcast path.
    # This checks its shared fan-out guard, not the complete-destination arm.
    destination = PerCoreView(
        ((0, 4),),
        (
            (
                0,
                Piecewise(
                    (0, Eq(_CORE_ID, 0)),
                    (1, _CORE_ID < (4 if balanced else 3)),
                    (2, Eq(_CORE_ID, 4 if balanced else 3)),
                    (3, True),
                ),
            ),
        ),
        num_cores=8,
    )
    source_map = lx_relayout._core_slices(_TWO_ON_2, 2)
    destination_map = lx_relayout._core_slices(destination, 8)
    assert {row[0] for row in destination_map.values()} == {0, 1, 2, 3}
    edges = lx_relayout.transfer_edges(
        dict(_TWO_ON_2.work_slice_dims),
        dict(destination.work_slice_dims),
        source_map,
        destination_map,
    )
    assert set(Counter(dst for _, dst in edges).values()) == {1}
    assert Counter(src for src, _ in edges) == (
        {0: 4, 1: 4} if balanced else {0: 3, 1: 5}
    )
    assert lx_relayout.movement_supported(_TWO_ON_2, destination, 2, 8) == balanced


@pytest.mark.parametrize("balanced", [True, False])
def test_complete_destination_rejects_nonuniform_fanout(balanced):
    source = PerCoreView(((0, 4),), ((0, _CORE_ID),), num_cores=4)
    rows, columns = (4, 3) if balanced else (3, 4)
    destination = PerCoreView(
        ((0, rows), (1, columns)),
        ((0, floor(_CORE_ID / columns)), (1, Mod(_CORE_ID, columns))),
        num_cores=12,
    )
    edges = lx_relayout.transfer_edges(
        dict(source.work_slice_dims),
        dict(destination.work_slice_dims),
        lx_relayout._core_slices(source, 4),
        lx_relayout._core_slices(destination, 12),
    )
    assert destination.split_product == destination.num_cores == 12
    assert (
        len(
            {
                tuple(row.items())
                for row in lx_relayout._core_slices(destination, 12).values()
            }
        )
        == 12
    )
    assert set(Counter(dst for _, dst in edges).values()) == ({1} if balanced else {2})
    assert Counter(src for src, _ in edges) == (
        {0: 3, 1: 3, 2: 3, 3: 3} if balanced else {0: 4, 1: 8, 2: 8, 3: 4}
    )
    assert lx_relayout.movement_supported(source, destination, 4, 12) == balanced


def test_destination_views_are_interned_by_ownership():
    """Two consumers whose candidates land on the same physical view of one
    parent must share ONE group (one shuffle serves both), even when the two
    views spell the slot expression differently."""
    groups: dict = {}
    assert _intern_view_group(groups, _QUARTER_PLAIN) == 0
    assert _intern_view_group(groups, _QUARTER_MOD) == 0  # same owners -> same group
    assert _intern_view_group(groups, _HALF_ON_4) == 1  # different owners -> new group
    assert _intern_view_group(groups, _QUARTER_PLAIN) == 0
    assert list(groups) == [_QUARTER_PLAIN, _HALF_ON_4]  # first spelling stays the key


def test_consumer_rules_are_shared_with_the_committed_collector(monkeypatch):
    """The consumer half of #3440's movement rules, one function per rule, used
    by the committed collector and the solver's enumeration alike: the core
    domains must be equal or the consumer must span every compute core, and a
    grouped gather (fewer destination owners than source cores) needs a matmul
    consumer, while a permutation or a broadcast destination does not."""
    matmul, pointwise = object(), object()
    monkeypatch.setattr(lx_relayout, "_is_matmul_op", lambda op: op is matmul)
    with config.patch({"sencores": 32}):
        assert core_domain_rejection(8, 8) is None
        assert core_domain_rejection(2, 32) is None
        assert "fewer physical cores" in core_domain_rejection(32, 8)
        assert "all compute cores" in core_domain_rejection(2, 16)
    grouped = PerCoreView(((2, 4),), ((2, Mod(_CORE_ID, 4)),), num_cores=8)
    assert grouped_gather_rejection(matmul, 8, grouped) is None
    assert "requires a matmul consumer" in grouped_gather_rejection(
        pointwise, 8, grouped
    )
    assert grouped_gather_rejection(pointwise, 8, _DST) is None  # a permutation
    assert grouped_gather_rejection(pointwise, 2, _FOUR_ON_8) is None  # a broadcast


@pytest.mark.parametrize(
    "invalid", [None, "offset", "compact_stride", "inner_tail", "second_reader"]
)
def test_prefix_lx_contraction_requires_exact_backing_stride(invalid):
    from dataclasses import replace
    import torch
    from torch._inductor.dependencies import MemoryDep, ReadWrites
    from torch._inductor.ir import ComputedBuffer, Pointwise, Reduction
    from torch._inductor.runtime.hints import ReductionHint
    from torch.utils._ordered_set import OrderedSet
    from torch_spyre._C import ElementArrangement
    from torch_spyre._inductor.constants import BATCH_MATMUL_OP
    from torch_spyre._inductor.ir import FixedTiledLayout

    m, physical, logical, n = 512, 3328, 3200, 4096
    a, b, k = [Symbol(name, integer=True, nonnegative=True) for name in ("a", "b", "k")]
    stl = SimpleNamespace(
        device_size=[52, m, 64],
        stride_map=[64, physical, 1],
        elems_per_stick=lambda: 64,
        element_arrangement=ElementArrangement.STANDARD,
    )
    if invalid == "inner_tail":
        stl.device_size, stl.stride_map = [m, 52, 64], [physical, 64, 1]
    layout = FixedTiledLayout(
        torch.device("cpu"), torch.float16, [m, physical], [physical, 1], stl
    )
    source = ComputedBuffer(
        name="source",
        layout=layout,
        data=Pointwise(
            device=torch.device("cpu"),
            dtype=torch.float16,
            inner_fn=lambda idx: 0,
            ranges=[m, physical],
        ),
    )
    consumer = ComputedBuffer(
        name="down",
        layout=layout.unallocated_copy(),
        data=Reduction(
            device=torch.device("cpu"),
            dtype=torch.float16,
            src_dtype=torch.float16,
            inner_fn=lambda idx, red: (0, 0),
            ranges=[m, n],
            reduction_ranges=[logical],
            reduction_type=BATCH_MATMUL_OP,
            reduction_hint=ReductionHint.DEFAULT,
        ),
    )
    stride = logical if invalid == "compact_stride" else physical
    offset = 64 if invalid == "offset" else 0
    read = MemoryDep("source", stride * a + k + offset, (a, b, k), (m, n, logical))
    weight = MemoryDep("weight", logical * b + k, (a, b, k), (m, n, logical))
    write = MemoryDep("down", n * a + b, (a, b), (m, n))
    consumer.get_read_writes = lambda: ReadWrites(
        OrderedSet([read, weight]), OrderedSet([write]), OrderedSet()
    )
    axis = lx_relayout.prefix_lx_contraction_axis(source, consumer)
    assert axis == (0 if invalid in (None, "second_reader") else None)
    if invalid == "second_reader":
        other = replace(consumer, name="other", layout=layout.unallocated_copy())
        other.get_read_writes = lambda: ReadWrites(
            OrderedSet([read, weight]),
            OrderedSet([MemoryDep("other", n * a + b, (a, b), (m, n))]),
            OrderedSet(),
        )
        graph = SimpleNamespace(
            operations=[source, consumer],
            get_buffer=lambda name: source,
            get_output_names=lambda: [],
        )
        assert lx_relayout.prefix_lx_gather_source(graph, "source", [0, 1])
        graph.operations = [source, other]
        assert lx_relayout.prefix_lx_gather_source(graph, "source", [0, 1])
        graph.operations = [source, consumer, other]
        assert not lx_relayout.prefix_lx_gather_source(graph, "source", [0, 1, 2])
