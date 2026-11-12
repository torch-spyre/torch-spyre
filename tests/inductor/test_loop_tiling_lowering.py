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

"""The loop-tiling plan type and its builder (``wsr/loop_tiling_lowering.py``).

The plan holds positions and a trip count and nothing that can be read back from the
graph. The builder is held to the legacy planner (``plan_coarse_tile_groups``) by
running both at the same call, on the same undivided graph.
"""

import contextlib
import dataclasses
from unittest import mock

import pytest
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)
from torch._inductor.ir import ComputedBuffer

from torch_spyre._inductor import config
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.wsr import coarse_tile
from torch_spyre._inductor.wsr.loop_tiling_lowering import (
    LoopTilingGroup,
    LoopTilingPlan,
    OpTiling,
    TiledDim,
    build_loop_tiling_plan,
)

from test_span_overflow_gate import (
    CASES,
    _forced_one_level,
    _permuted,
    _plain,
    _shaped,
    compile_span_overflow,
)

_SHAPE = (8, 8, 64, 512)


def test_the_plan_types_are_frozen_and_hold_only_positions():
    """Nothing derivable from the live graph (kind, consumers, ranges, per-read
    lists) is a field: a plan that held it could go stale as the passes run."""
    names = lambda cls: [f.name for f in dataclasses.fields(cls)]  # noqa: E731
    assert names(TiledDim) == ["position", "in_reduction"]
    assert names(OpTiling) == ["op_name", "tiled"]
    assert names(LoopTilingGroup) == ["index", "trip_count", "ops"]
    assert names(LoopTilingPlan) == ["groups"]
    dim = TiledDim(1, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        dim.position = 2  # type: ignore[misc]
    group = LoopTilingGroup(0, 2, (OpTiling("op0", dim), OpTiling("op1", None)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        group.trip_count = 3  # type: ignore[misc]
    assert LoopTilingPlan((group,)).groups[0].ops[1].tiled is None


def _legacy_position(info):
    if info is None:
        return None
    if info.loop_tiled_dims[0]:
        return (info.loop_tiled_dims[0][0], False)
    if info.loop_tiled_reduction_dims[0]:
        return (info.loop_tiled_reduction_dims[0][0], True)
    return None


@contextlib.contextmanager
def _built_beside_the_legacy_planner(into, *, regroup=None):
    """At every call of ``plan_coarse_tile_groups`` build the new plan from the same
    arguments first (the graph is still undivided), then run the legacy planner.

    ``into`` gets ``(legacy positions by op name, new plan or the Unsupported it
    raised, trip counts)`` per call. ``regroup(operations, groups)`` may change the
    groups the builder (only) is given."""
    original = coarse_tile.plan_coarse_tile_groups

    def spy(operations, groups):
        given = regroup(operations, groups) if regroup else groups
        try:
            built = build_loop_tiling_plan(operations, given)
        except Unsupported as error:
            built = error
        legacy = original(operations, groups)
        positions = {
            o.get_operation_name(): _legacy_position(legacy.get(id(o)))
            for ops, _ in groups
            for o in ops
            if isinstance(o, ComputedBuffer)
        }
        into.append((positions, built, [int(c) for _, lv in groups for _, c in lv]))
        return legacy

    with mock.patch.object(coarse_tile, "plan_coarse_tile_groups", spy):
        yield


def _assert_same_as_legacy(calls):
    assert calls, "the planner was never reached"
    for positions, built, counts in calls:
        assert isinstance(built, LoopTilingPlan), built
        assert [g.trip_count for g in built.groups] == counts
        for group in built.groups:
            for op in group.ops:
                tiled = op.tiled
                got = None if tiled is None else (tiled.position, tiled.in_reduction)
                assert got == positions[op.op_name], (op.op_name, got, positions)
        # A ComputedBuffer the legacy planner left alone is in the plan with tiled=None.
        planned = {o.op_name for g in built.groups for o in g.ops}
        assert planned == set(positions)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_plan_equals_the_legacy_planners_positions_on_the_gate_cases(case):
    fn, shapes, limit, *_ = CASES[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    calls = []
    with _built_beside_the_legacy_planner(calls):
        compile_span_overflow(fn, args, limit)
    _assert_same_as_legacy(calls)
    # Some case must tile an op, or the equality above is vacuous.
    assert any(
        o.tiled is not None
        for _, built, _ in calls
        for g in built.groups
        for o in g.ops
    )


@pytest.mark.parametrize(
    "make_input, levels",
    [(_plain, [(0, 8)]), (_permuted, [(0, 8)]), (_permuted, [(1, 8)])],
    ids=["plain-0", "permuted-0", "permuted-1"],
)
def test_the_plan_equals_the_legacy_planners_positions_on_forced_plans(
    make_input, levels
):
    calls = []
    with _built_beside_the_legacy_planner(calls):
        _forced_one_level(_SHAPE, levels, make_input)
    _assert_same_as_legacy(calls)


def test_a_group_of_two_ops_is_planned_op_by_op():
    fn, shapes, limit, *_ = CASES["pointwise_chain"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    calls = []
    with _built_beside_the_legacy_planner(calls):
        compile_span_overflow(fn, args, limit)
    (_, built, _), *_ = calls
    assert [len(g.ops) for g in built.groups] == [2]
    first, second = (o.tiled for o in built.groups[0].ops)
    assert first is not None and first == second


def test_a_reduction_with_reduction_tiling_disabled_is_refused():
    """The real auto-tiler already refuses with the flag off, so the flag is turned
    off for the builder's call only."""
    fn, shapes, limit, *_ = CASES["matmul_k_tiled"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    seen = []
    original = coarse_tile.plan_coarse_tile_groups

    def spy(operations, groups):
        build_loop_tiling_plan(operations, groups)  # fine with the flag on
        with config.patch({"enable_reduction_tiling": False}):
            try:
                build_loop_tiling_plan(operations, groups)
            except Unsupported as error:
                seen.append(str(error))
        return original(operations, groups)

    with mock.patch.object(coarse_tile, "plan_coarse_tile_groups", spy):
        compile_span_overflow(fn, args, limit)
    assert seen and "enable_reduction_tiling" in seen[0], seen


def test_a_partial_reduction_read_in_the_group_is_refused():
    """A pointwise op that reads a K-tiled reduction in the same loop would see a
    partial sum. The grouping never forms this group, so hand the builder one."""
    seen = []

    def sum_then_its_consumer(operations, groups):
        (ops, levels), *_ = groups
        names = {o.get_name() for o in ops if isinstance(o, ComputedBuffer)}
        consumer = next(
            o
            for o in operations
            if isinstance(o, ComputedBuffer)
            and o not in ops
            and any(n in names for n in o.get_read_names())
        )
        return [([*ops, consumer], levels)]

    def fn(x):
        return x.sum(dim=1) * 2

    x = _shaped((64, 8192), 1)
    calls = []
    with _built_beside_the_legacy_planner(calls, regroup=sum_then_its_consumer):
        try:
            compile_span_overflow(fn, [x], 256 * 1024)
        except Unsupported as error:  # the real grouping refuses first, or not
            seen.append(error)
    assert calls, "the planner was never reached"
    (_, built, _), *_ = calls
    assert isinstance(built, Unsupported), built
    assert "partial reduction result consumed" in str(built)


def test_two_levels_are_refused():
    seen = []
    original = coarse_tile.plan_coarse_tile_groups

    def spy(operations, groups):
        ((ops, levels), *_) = groups
        try:
            build_loop_tiling_plan(operations, [(ops, [*levels, *levels])])
        except Unsupported as error:
            seen.append(str(error))
        return original(operations, groups)

    fn, shapes, limit, *_ = CASES["pointwise_copy_out"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    with mock.patch.object(coarse_tile, "plan_coarse_tile_groups", spy):
        compile_span_overflow(fn, args, limit)
    assert seen and "nested" in seen[0].lower(), seen


def test_the_group_index_counts_from_the_offset():
    fn, shapes, limit, *_ = CASES["pointwise_independent_pair"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    seen = []
    original = coarse_tile.plan_coarse_tile_groups

    def spy(operations, groups):
        seen.append(
            (
                [g.index for g in build_loop_tiling_plan(operations, groups).groups],
                [g.index for g in build_loop_tiling_plan(operations, groups, 5).groups],
            )
        )
        return original(operations, groups)

    with mock.patch.object(coarse_tile, "plan_coarse_tile_groups", spy):
        compile_span_overflow(fn, args, limit)
    assert seen == [([0], [5])], seen
