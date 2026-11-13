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
import os
from unittest import mock

import pytest
import sympy
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)
from torch._inductor.ir import ComputedBuffer

from torch_spyre._inductor import config, passes
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.wsr import coarse_tile
from torch_spyre._inductor.wsr.loop_tiling_lowering import (
    LoopTilingGroup,
    LoopTilingPlan,
    OpTiling,
    TiledDim,
    build_loop_tiling_plan,
)

from utils_inductor import expects_raise
from test_span_overflow_gate import (
    _relative_error,
    CASES,
    MB,
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

    with (
        mock.patch.object(coarse_tile, "plan_coarse_tile_groups", spy),
        config.patch({"loop_tiling_lowering": "legacy"}),
    ):
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

    with (
        mock.patch.object(coarse_tile, "plan_coarse_tile_groups", spy),
        config.patch({"loop_tiling_lowering": "legacy"}),
    ):
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
    with (
        mock.patch.object(coarse_tile, "plan_coarse_tile_groups", spy),
        config.patch({"loop_tiling_lowering": "legacy"}),
    ):
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

    with (
        mock.patch.object(coarse_tile, "plan_coarse_tile_groups", spy),
        config.patch({"loop_tiling_lowering": "legacy"}),
    ):
        compile_span_overflow(fn, args, limit)
    assert seen == [([0], [5])], seen


def test_the_switch_defaults_and_rejects_nonsense():
    assert config._choose_loop_tiling_lowering("plan") == "plan"
    assert config._choose_loop_tiling_lowering("legacy") == "legacy"
    with pytest.raises(ValueError, match="bogus"):
        config._choose_loop_tiling_lowering("bogus")
    # Unset in the environment the default is the plan.
    if "SPYRE_INDUCTOR_LOOP_TILING_LOWERING" not in os.environ:
        assert config.loop_tiling_lowering == "plan"


@expects_raise
def test_a_group_holding_an_op_with_loop_info_is_refused():
    from torch_spyre._inductor.loop_info import CoarseTileInfo

    fn, shapes, limit, *_ = CASES["pointwise_copy_out"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    original = coarse_tile.coarse_tile_post_stickify

    def spy(graph, groups, group_idx_offset=0):
        groups[0][0][0].loop_info = CoarseTileInfo(
            loop_group_id=(9,),
            loop_count=[2],
            loop_tiled_dims=[[0]],
            loop_tiled_reduction_dims=[[]],
        )
        return original(graph, groups=groups, group_idx_offset=group_idx_offset)

    with (
        mock.patch.object(coarse_tile, "coarse_tile_post_stickify", spy),
        mock.patch.object(passes, "coarse_tile_post_stickify", spy),
        config.patch({"loop_tiling_lowering": "plan"}),
        pytest.raises(Exception, match="existing loop_info"),
    ):
        compile_span_overflow(fn, args, limit)


@pytest.mark.parametrize("case", ["pointwise_copy_out", "pointwise_chain"])
def test_the_new_lowering_stamps_a_shim_and_no_stored_lists(case):
    """Every looped op carries a LoopShim, and the stage-7a.2 pass that adds terms to
    a legacy record is not involved."""
    from torch_spyre._inductor.loop_info import CoarseTileInfo, LoopShim
    from torch_spyre._inductor.loop_node import all_ops

    fn, shapes, limit, *_ = CASES[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    infos = []
    original = passes._hand_off_loop_nodes

    def spy(graph):
        infos.extend(
            getattr(op, "loop_info", None)
            for op in all_ops(graph.operations)
            if getattr(op, "loop_info", None) is not None
        )
        return original(graph)

    with (
        mock.patch.object(passes, "_hand_off_loop_nodes", spy),
        mock.patch.object(coarse_tile, "add_loop_terms") as add_loop_terms,
        config.patch({"loop_tiling_lowering": "plan"}),
    ):
        compile_span_overflow(fn, args, limit)
    assert infos and all(isinstance(i, LoopShim) for i in infos), infos
    assert not any(isinstance(i, CoarseTileInfo) for i in infos)
    add_loop_terms.assert_not_called()


def _carries(fn, args, limit):
    from test_span_overflow_loop_terms import _compile_seeing_carries

    with config.patch({"loop_tiling_lowering": "plan"}):
        return _compile_seeing_carries(fn, args, limit)


def test_the_k_tiled_accumulator_is_declared_as_a_carry_under_plan():
    """The record of stage 7a.3: the accumulator's storage, the combine as its update,
    no FX origin; the allocator takes the storage for a carry."""
    fn, shapes, limit, *_ = CASES["matmul_k_tiled"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    compiled, carries = _carries(fn, args, limit)
    assert len(carries) == 1, carries
    _, records = carries[0]
    assert len(records) == 1, records
    storage, update, is_storage, on_storage, on_update, origin = records[0]
    assert update.startswith("coarse_tile_combine_"), update
    assert is_storage and on_storage and on_update, records
    assert origin is None, origin
    assert compiled.added[-2:] == [
        a for a in compiled.added if "_fill_" in a or "_combine_" in a
    ], compiled.added


@expects_raise
def test_a_nested_accumulator_is_refused():
    """A matmul too large for one level nests two (``[4, 4]``): refused for the loop
    builder, with the issue that tracks it (#6088).

    TODO: when nested loops are supported this becomes a positive test."""
    args = [_shaped((4096, 4096), seed) for seed in range(2)]
    with (
        config.patch({"loop_tiling_lowering": "plan"}),
        pytest.raises(Exception, match="#6088"),
    ):
        compile_span_overflow(lambda a, b: a @ b, args, 4 * MB)


@expects_raise
def test_an_inexpressible_group_raises_under_plan_and_does_not_run_legacy():
    """A group the plan cannot lower raises; it does not fall back to the legacy
    transform."""
    fn, shapes, limit, *_ = CASES["matmul_k_tiled"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]

    def fail(*a, **k):
        raise AssertionError("the legacy transform ran under the plan setting")

    with (
        config.patch({"loop_tiling_lowering": "plan"}),
        mock.patch.object(coarse_tile, "_apply_plan", fail),
        mock.patch.object(coarse_tile, "_insert_all_reduction_ops", fail),
        mock.patch(
            "torch_spyre._inductor.wsr.loop_tiling_lowering._plan_carries",
            side_effect=Unsupported("carry cannot be declared"),
        ),
        pytest.raises(Exception, match="carry cannot be declared"),
    ):
        compile_span_overflow(fn, args, limit)


class _RecordPostCallState:
    """Substituted for the pre-scheduling pipeline: records, per loop op, what the
    allocator's ``apply_tilings`` leaves on it after the lowering returns."""

    seen: dict = {}


def _allocator_post_call_state(lowering):
    """Compile ``(x + y) * 2`` with the allocator tiling every op d0 by 4, under
    ``lowering``; per op in a loop: ``(has symbols_on_tile, joint-division flag)``."""
    from test_solver_auto_coarse_tiling import (
        _D0_BY_4,
        CollectTilingPasses,
    )
    from torch._inductor import config as t_inductor_config

    from torch_spyre._inductor.loop_node import all_ops
    from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator
    from torch_spyre._inductor.work_division_constraints import (
        JOINT_TILING_AND_DIVISION_ATTR,
    )
    from torch_spyre.constants import DEVICE_NAME

    seen: dict = {}

    class Record(CollectTilingPasses):
        def __call__(self, graph):
            super().__call__(graph)
            seen.update(
                {
                    op.get_name(): (
                        hasattr(op, "symbols_on_tile"),
                        bool(getattr(op, JOINT_TILING_AND_DIVISION_ATTR, False)),
                    )
                    for op in all_ops(graph.operations)
                    if getattr(op, "loop_info", None) is not None
                }
            )

    x = torch.randn(64, 64, 128, dtype=torch.float16)
    y = torch.randn(64, 64, 128, dtype=torch.float16)
    fn = lambda x, y: (x + y) * 2  # noqa: E731
    torch._dynamo.reset()
    with (
        mock.patch.object(
            CoOptimizingAllocator, "_tiling_candidates", lambda *a, **k: [_D0_BY_4]
        ),
        torch.no_grad(),
        t_inductor_config.patch(force_disable_caches=True),
        config.patch(
            {
                "co_optimizing_lx_planning": True,
                "auto_coarse_tiling": True,
                "layout_solver": "cpsat",
                "allow_all_ops_in_lx_planning": True,
                "loop_tiling_lowering": lowering,
            }
        ),
        mock.patch.object(passes, "CustomPreSchedulingPasses", Record),
    ):
        out = torch.compile(fn, fullgraph=True)(x.to(DEVICE_NAME), y.to(DEVICE_NAME))
    torch.testing.assert_close(out.cpu(), fn(x, y), atol=0.1, rtol=0.05)
    return seen


def test_the_post_call_contract_is_kept_under_plan():
    """The allocator reads two things after the lowering returns: ``symbols_on_tile`` on
    each op it tiled, and ``JOINT_TILING_AND_DIVISION_ATTR`` on the tiled ops and on
    the ops the lowering added. Both are set by ``apply_tilings`` around the call, so
    they must hold under the plan exactly as under the legacy lowering."""
    legacy = _allocator_post_call_state("legacy")
    plan = _allocator_post_call_state("plan")
    assert plan == legacy
    assert plan, "no op was tiled"
    assert any(n.startswith("coarse_tile_copy_") for n in plan), sorted(plan)
    assert all(joint for _, joint in plan.values()), plan
    assert any(symbols for symbols, _ in plan.values()), plan


def _mixed_graph():
    from for_each_tile_fixtures import split_m_fn

    def fn(x, y, a, b):
        return split_m_fn(x, y), a + b

    shapes = [(256, 256), (256, 64), (4096, 1024), (4096, 1024)]
    return fn, [_shaped(shape, seed) for seed, shape in enumerate(shapes)], 4 * MB


def test_no_for_each_tile_op_has_a_loop_shim():
    """A graph with a ``for_each_tile`` loop and a span-overflow loop, compiled under
    the plan: each loop node holds one kind of record, the ``for_each_tile`` one a
    ``CoarseTileInfo`` and the span-overflow one a ``LoopShim``, never the reverse."""
    from test_span_overflow_loop_terms import _nodes_of

    from torch_spyre._inductor.loop_info import CoarseTileInfo, LoopShim
    from torch_spyre._inductor.loop_node import all_ops

    fn, args, limit = _mixed_graph()
    kinds: list[set] = []
    original = passes._hand_off_loop_nodes

    def spy(graph):
        kinds.extend(
            {type(getattr(op, "loop_info", None)) for op in all_ops(node.body)}
            for node in _nodes_of(graph.operations)
        )
        return original(graph)

    with (
        mock.patch.object(passes, "_hand_off_loop_nodes", spy),
        config.patch({"loop_tiling_lowering": "plan"}),
    ):
        compiled = compile_span_overflow(fn, args, limit)
    assert sorted(map(sorted, ([t.__name__ for t in k] for k in kinds))) == [
        ["CoarseTileInfo"],
        ["LoopShim"],
    ], kinds
    assert CoarseTileInfo and LoopShim
    x, y, a, b = (t.float() for t in args)
    assert _relative_error(compiled.outputs, (x @ y, a + b)) < 2e-2


def test_a_for_each_tile_loop_beside_a_span_overflow_loop_matches_legacy():
    """The ``for_each_tile`` kernel is untouched by the setting."""
    from test_loop_tiling_parity import _assert_parity

    fn, args, limit = _mixed_graph()
    _assert_parity(fn, args, limit)


def _stub_op(ranges):
    from types import SimpleNamespace

    return SimpleNamespace(
        data=SimpleNamespace(ranges=[sympy.Integer(r) for r in ranges]),
        get_operation_name=lambda: "op0",
    )


def _dep(index):
    from torch._inductor.dependencies import MemoryDep

    return MemoryDep("buf0", index, (), ())


def test_the_step_is_the_per_trip_delta_of_an_affine_index():
    from torch._inductor.utils import sympy_index_symbol

    from torch_spyre._inductor.wsr.loop_tiling_lowering import TiledDim, _tile_step

    d0, d1 = sympy_index_symbol("d0"), sympy_index_symbol("d1")
    step = _tile_step(
        _stub_op([64, 32]), TiledDim(0, False), _dep(32 * d0 + d1), sympy.Integer(8)
    )
    assert step == 256


def test_an_index_that_is_only_partly_affine_is_stepped_by_its_delta_or_refused():
    """``d0 + 7*floor(d0/8)`` has coefficient 1 at ``d0 = 0``, so ``coefficient x
    extent`` would read the wrong address on every trip after the first. A tile of 8
    moves it by 15 every trip, and that is the step; a tile of 5 moves it by 5 or 6
    depending on the trip, so there is no step and the group is refused."""
    from torch._inductor.utils import sympy_index_symbol

    from torch_spyre._inductor.wsr.loop_tiling_lowering import TiledDim, _tile_step

    d0 = sympy_index_symbol("d0")
    dep = _dep(d0 + 7 * sympy.floor(d0 / 8))
    tiled = TiledDim(0, False)
    assert _tile_step(_stub_op([64, 32]), tiled, dep, sympy.Integer(8)) == 15
    with pytest.raises(Unsupported, match="affine"):
        _tile_step(_stub_op([60, 32]), tiled, dep, sympy.Integer(5))


@expects_raise
def test_a_member_the_loop_does_not_tile_may_not_read_a_tiled_members_tile():
    """The tile is rewritten every trip and is smaller than the full range an untiled
    member would iterate over it.

    TODO: if such a member is ever lowered (it would have to be tiled too), this
    becomes a positive test."""
    import dataclasses

    from torch_spyre._inductor.wsr import loop_tiling_lowering as lowering

    fn, shapes, limit, *_ = CASES["pointwise_chain"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    original = lowering.build_loop_tiling_plan

    def untile_the_consumer(operations, groups, group_idx_offset=0):
        plan = original(operations, groups, group_idx_offset)
        (group,) = plan.groups
        ops = (group.ops[0], dataclasses.replace(group.ops[1], tiled=None))
        return lowering.LoopTilingPlan((dataclasses.replace(group, ops=ops),))

    with (
        mock.patch.object(lowering, "build_loop_tiling_plan", untile_the_consumer),
        config.patch({"loop_tiling_lowering": "plan"}),
        pytest.raises(Exception, match="does not tile"),
    ):
        compile_span_overflow(fn, args, limit)


def test_lx_offsets_compare_by_which_buffers_share_an_address():
    from test_loop_tiling_parity import _comparable

    two = "a allocation={'lx': 0} b allocation={'lx': 262144} c allocation={'lx': 0}"
    swapped = (
        "a allocation={'lx': 262144} b allocation={'lx': 0} c allocation={'lx': 262144}"
    )
    merged = "a allocation={'lx': 0} b allocation={'lx': 0} c allocation={'lx': 0}"
    assert _comparable(two) == _comparable(swapped)
    assert _comparable(two) != _comparable(merged)


def _compile_with_menu(fn, args, menu):
    """Compile ``fn`` under the plan with the allocator offered ``menu(op)`` for each
    op; returns ``(device result, cpu result, {op: loop_info}, {op: per-read advances})``."""
    from test_solver_auto_coarse_tiling import CollectTilingPasses
    from torch._inductor import config as t_inductor_config

    from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator

    cpu = fn(*args)
    torch._dynamo.reset()
    CollectTilingPasses.tiling = {}
    CollectTilingPasses.read_advances = {}
    with (
        mock.patch.object(
            CoOptimizingAllocator, "_tiling_candidates", lambda a, op, cores: menu(op)
        ),
        torch.no_grad(),
        t_inductor_config.patch(force_disable_caches=True),
        config.patch(
            {
                "co_optimizing_lx_planning": True,
                "auto_coarse_tiling": True,
                "layout_solver": "cpsat",
                "allow_all_ops_in_lx_planning": True,
                "loop_tiling_lowering": "plan",
            }
        ),
        mock.patch.object(passes, "CustomPreSchedulingPasses", CollectTilingPasses),
    ):
        device = torch.compile(fn, fullgraph=True)(*(a.to("spyre") for a in args))
    return (
        device.cpu(),
        cpu,
        dict(CollectTilingPasses.tiling),
        dict(CollectTilingPasses.read_advances),
    )


def _one_row_consumer_reads_the_whole_producer(infos, advances):
    """The 64-trip nest's read of what the other nest produced advances one producer
    row (64 * 128 elements) per trip."""
    consumer_nest = [
        name
        for name, info in infos.items()
        if tuple(info.loop_count) == (64,) and not name.startswith("coarse_tile_copy_")
    ]
    assert consumer_nest, sorted(infos)
    assert any(
        advance[0] == 64 * 128 for name in consumer_nest for advance in advances[name]
    ), {name: advances[name] for name in consumer_nest}


def test_a_unit_row_consumer_in_the_next_nest_reads_every_row_of_a_copied_out_producer():
    """The producer takes two tiles of 64 rows; the repeat after it takes one row per
    tile in a nest of its own and reads the producer's full buffer. Its own tiling
    divides its row dim down to one element, which leaves no symbol for that dim in
    the read. The legacy transform returned about 95% wrong values here (every trip
    read the producer's row 0), and carries a fix for it in its own record. The plan
    reads the step from the index before dividing, so the dim has not vanished."""
    from test_solver_auto_coarse_tiling import (
        TileAxis,
        TileSpec,
        _reads_graph_inputs_only,
    )

    x = torch.randn(128, 64, 128, dtype=torch.float16)
    y = torch.randn(128, 64, 128, dtype=torch.float16)
    halves = TileSpec((TileAxis(host_dim=0, count=2),))
    unit_rows = TileSpec((TileAxis(host_dim=0, count=64),))
    device, cpu, infos, advances = _compile_with_menu(
        lambda x, y: (x + y)[:64].repeat(1, 2, 1) * 2,
        (x, y),
        lambda op: [halves] if _reads_graph_inputs_only(op) else [unit_rows],
    )
    torch.testing.assert_close(device, cpu, atol=0.1, rtol=0.05)
    _one_row_consumer_reads_the_whole_producer(infos, advances)


def test_a_unit_row_consumer_in_the_next_nest_reads_every_row_of_a_tiled_reduction():
    """As above, but the producer is a sum tiled along its reduced dim, so the repeat
    reads the reduction's accumulator and not a copied-out buffer (also about 95%
    wrong under the legacy transform)."""
    from test_solver_auto_coarse_tiling import (
        TileAxis,
        TileSpec,
        _reads_graph_inputs_only,
    )

    x = torch.randn(128, 8, 64, 128, dtype=torch.float16)
    reduced_halves = TileSpec((TileAxis(host_dim=0, count=2, is_reduction=True),))
    unit_rows = TileSpec((TileAxis(host_dim=0, count=64),))
    device, cpu, infos, advances = _compile_with_menu(
        lambda x: x.sum(dim=1)[:64].repeat(1, 2, 1) * 2,
        (x,),
        lambda op: [reduced_halves] if _reads_graph_inputs_only(op) else [unit_rows],
    )
    torch.testing.assert_close(device, cpu, atol=0.1, rtol=0.05)
    assert any(any(info.loop_tiled_reduction_dims) for info in infos.values()), infos
    _one_row_consumer_reads_the_whole_producer(infos, advances)
