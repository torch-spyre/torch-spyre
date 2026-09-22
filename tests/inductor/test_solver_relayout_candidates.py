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

import pytest
from sympy import Eq, Mod, Piecewise, Symbol, floor

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


def test_split_past_fitted_range_is_declined_not_clamped():
    # Governing split 16 is outside the law's fitted range [2, 8], where it
    # over-predicts 12-40%. The reporting path clamps (the shuffle already
    # exists); the solver path must not OFFER an option at a price the law was
    # never fitted for, so the pair is declined outright.
    dims = [256, 8, 16, 64]
    src16 = PerCoreView(((2, 16),), ((2, _CORE_ID),))
    dst16 = PerCoreView(((0, 16),), ((0, _CORE_ID),))
    assert governing_run_split(src16, dst16, dims) == (64, 16)
    assert (
        solver_relayout_pair_cost(
            src16, dst16, 16, dims, 256 * 8 * 16 * 64, _DTYPE_BYTES
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


def test_coarse_tiled_endpoints_are_declined_at_the_edge_gate():
    """An endpoint carrying loop_info can never host a relayout: the fitted
    law has no loop_trip factor, a tiled producer's buffer is per-tile
    scratch, and a tiled consumer reads through a per-iteration staging op.

    Probes are real-enough ComputedBuffers (created via __new__ to pass the
    isinstance check) whose ``layout`` is a recording property: the gate is
    the first check, so a tiled endpoint must be rejected WITHOUT the layout
    ever being touched - distinguishing gate-rejection from the downstream
    rejections that any fake would also hit. (The full compile-path variant
    lives in test_solver_relayout_e2e.py, skipped until the co-opt substrate
    survives coarse-tiled graphs at all; its non-vacuity was verified by
    running it with the gate reverted.)"""
    from torch._inductor.ir import ComputedBuffer

    from torch_spyre._inductor.scratchpad.lx_relayout import (
        solver_relayout_edge_context,
    )

    def probe(tiled):
        class _Probe(ComputedBuffer):
            layout_accessed = False

            @property
            def layout(self):
                type(self).layout_accessed = True

        obj = _Probe.__new__(_Probe)
        obj.loop_info = object() if tiled else None
        return obj

    # No graph: both gates under test sit before the first use of it.
    graph = None
    for prod_tiled, cons_tiled in ((True, False), (False, True), (True, True)):
        prod, cons = probe(prod_tiled), probe(cons_tiled)
        assert solver_relayout_edge_context(graph, prod, cons, "buf0", {}) is None
        assert not type(prod).layout_accessed and not type(cons).layout_accessed, (
            "a coarse-tiled endpoint reached checks past the loop_info gate"
        )

    # Control: an untiled pair proceeds past the gate into the layout check,
    # proving the probes are capable of registering deeper progression.
    prod, cons = probe(False), probe(False)
    assert solver_relayout_edge_context(graph, prod, cons, "buf0", {}) is None
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
