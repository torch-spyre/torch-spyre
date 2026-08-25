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
from sympy import Mod, Symbol, floor

from torch_spyre._inductor.dump_cost_model import governing_run_split
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.scratchpad.lx_relayout import solver_relayout_pair_cost

_CORE_ID = Symbol("core_id")

# Logical [8, 256, 512] fp16 commits device layout [256, 8, 8, 64]: an M hint
# splits device dim 0, a B hint splits dim 2 (see lx_relayout_cost.md). These
# are the two views of the canonical measured configuration: producer
# {B:4, M:2} against consumer {B:2, M:4} on 8 cores.
_DEVICE_DIMS = [256, 8, 8, 64]
_SRC = PerCoreView(((0, 2), (2, 4)), ((0, floor(_CORE_ID / 4)), (2, Mod(_CORE_ID, 4))))
_DST = PerCoreView(((0, 4), (2, 2)), ((0, floor(_CORE_ID / 2)), (2, Mod(_CORE_ID, 2))))
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


def test_grouped_gather_is_declined():
    # Destination with 4 distinct owners on 8 cores is a grouped gather
    # (#3440): a multicast, not a permutation. Its term is uncalibrated, so
    # the enumeration must decline to price it rather than use permutation
    # constants (same stance as the extractor).
    grouped = PerCoreView(((2, 4),), ((2, Mod(_CORE_ID, 4)),))
    assert (
        solver_relayout_pair_cost(
            _SRC, grouped, 8, _DEVICE_DIMS, _OUT_ELEMS, _DTYPE_BYTES
        )
        is None
    )


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
    # _compatible_partitions requires split products equal to num_cores on
    # BOTH sides: the canonical pair (products 8) cannot host a permutation on
    # a 4-core solve. (A view whose slot expression is out of range for
    # num_cores is a caller error, not a declined pair - the allocator's
    # cores_used equality gate guarantees views are built for num_cores
    # before pricing.)
    assert (
        solver_relayout_pair_cost(_SRC, _DST, 4, _DEVICE_DIMS, _OUT_ELEMS, _DTYPE_BYTES)
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
                return None

        obj = _Probe.__new__(_Probe)
        obj.loop_info = object() if tiled else None
        return obj

    for prod_tiled, cons_tiled in ((True, False), (False, True), (True, True)):
        prod, cons = probe(prod_tiled), probe(cons_tiled)
        assert solver_relayout_edge_context(prod, cons, "buf0", {}) is None
        assert not type(prod).layout_accessed and not type(cons).layout_accessed, (
            "a coarse-tiled endpoint reached checks past the loop_info gate"
        )

    # Control: an untiled pair proceeds past the gate into the layout check,
    # proving the probes are capable of registering deeper progression.
    prod, cons = probe(False), probe(False)
    assert solver_relayout_edge_context(prod, cons, "buf0", {}) is None
    assert type(prod).layout_accessed
