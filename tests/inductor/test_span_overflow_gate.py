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

"""Compiled span-overflow graphs, as a gate for replacing the lowering (stage 7a.0).

Span-overflow coarse tiling (``ignore_span_overflow_hints=False``) is opt-in. The
planner tests in ``test_span_overflow_hint_analysis.py`` run on mocked ops, so no
test compiles a graph the pass actually tiles. The pass only fires when an op's
per-core span exceeds ``MAX_SPAN_BYTES`` (256 MB), which is why the real
fixtures were large; here the limit is lowered in every module that holds its
own copy of the constant, so the cases compile in seconds.

Each case records what the current lowering produces:

* the group structure the planner forms (ops per group, kind of each op, trip
  count of each level), which a replacement lowering must preserve;
* the ops the transform adds, by name pattern (``coarse_tile_copy_*``,
  ``..._fill_*``, ``..._combine_*``), which ``scratchpad.coarse_tiling`` and work
  division depend on (``apply_tilings`` sets ``JOINT_TILING_AND_DIVISION_ATTR`` on
  ops "not in before and with a loop_info");
* the result against an fp32 reference;
* that the generated code is deterministic once run-specific ids are
  normalised, so a later comparison against a new lowering can be trusted.

Cases outside what a replacement will support (more than one level, a chain the
planner cannot join) are pinned as rejections.

The limit for each case is set just under the largest tile-able tensor (half of
it): a smaller limit makes the planner ask for more tiling than it will do, and
it is rejected instead.
"""

import contextlib
import functools
import itertools
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import pytest
import regex as re
import torch
from torch._inductor.exc import InductorError
from torch._inductor.utils import run_and_get_code

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._inductor import config
from torch_spyre._inductor import nonstick_dim_order, passes, work_division
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.scratchpad import allocator as allocator_module
from torch_spyre._inductor.scratchpad import coarse_tiling, ilp_solver_ortools
from torch_spyre._inductor.scratchpad.coarse_tiling import MAX_SEARCH_TILE_DEPTH
from torch_spyre._inductor.scratchpad.plan_solver import SolveError, TileAxis, TileSpec
from torch_spyre._inductor.wsr import span_overflow_hint_analysis
from torch_spyre._inductor.wsr.coarse_tile import reject_nested_loops
from torch_spyre.constants import DEVICE_NAME

from utils_inductor import expects_raise

MB = 1024 * 1024


@contextlib.contextmanager
def span_limit(nbytes):
    """Lower ``MAX_SPAN_BYTES`` in every module that imported its own copy."""
    with (
        mock.patch.object(span_overflow_hint_analysis, "MAX_SPAN_BYTES", nbytes),
        mock.patch.object(work_division, "MAX_SPAN_BYTES", nbytes),
        mock.patch.object(nonstick_dim_order, "MAX_SPAN_BYTES", nbytes),
    ):
        yield


def _shaped(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=generator) * 0.05).half()


class _Compiled:
    def __init__(self, outputs, groups, added, code, paths):
        self.outputs = outputs
        self.groups = groups  # [(ops in group, [data type of each], [trip counts])]
        self.added = added  # names of ops the transform added, in order
        self.code = code  # normalised generated code
        # Who applied each group: "pass" (passes.py, before the solve or in a
        # failed solve's fallback) or "solve" (apply_tilings, after the solve).
        self.paths = paths


_RUN_SPECIFIC = (
    (r"DebugHandle\(id=\d+", "DebugHandle(id=N"),
    (r"debug_handle=\d+", "debug_handle=N"),
    (r"while_carry_\d+_", "while_carry_N_"),
    (r"# AOT ID: \['\d+_", "# AOT ID: ['N_"),
    (r"ProvenanceTransform\([^)]*\)", ""),
    (r"/tmp/torchinductor_[^/\s'\"]+/[^'\"\s]*", "/tmp/PATH"),
)


def normalise(text):
    for pattern, replacement in _RUN_SPECIFIC:
        text = re.sub(pattern, replacement, text)
    return text


def compile_span_overflow(fn, args, limit_bytes, *, in_solve=True):
    """Compile ``fn`` with span-overflow on and the span limit lowered.

    The lowering is watched from the pass and from the solve's apply, both
    calling ``passes.tile_groups`` as it is now, so a wrapper there sees either
    path. ``in_solve=False`` keeps every group with the pass."""
    formed, added, paths = [], [], []
    original = passes.tile_groups

    def spy(graph, groups, group_idx_offset=0, *, path):
        before = {op.get_operation_name() for op in graph.operations}
        formed.extend(
            (
                len(ops),
                [type(getattr(op, "data", op)).__name__ for op in ops],
                [int(count) for _, count in levels],
            )
            for ops, levels in groups
        )
        paths.extend(path for _ in groups)
        result = original(graph, groups=groups, group_idx_offset=group_idx_offset)
        added.extend(
            op.get_operation_name()
            for op in graph.operations
            if op.get_operation_name() not in before
        )
        return result

    in_pass = mock.patch.object(
        passes, "solve_decides_span_overflow_tiling", lambda: False
    )
    torch._dynamo.reset()
    with (
        mock.patch.object(passes, "tile_groups", functools.partial(spy, path="pass")),
        mock.patch.object(
            coarse_tiling, "tile_groups", functools.partial(spy, path="solve")
        ),
        contextlib.nullcontext() if in_solve else in_pass,
        span_limit(limit_bytes),
        config.patch({"ignore_span_overflow_hints": False}),
        torch._inductor.config.patch(force_disable_caches=True),
    ):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        out, code = run_and_get_code(compiled, *(a.to(DEVICE_NAME) for a in args))
    outputs = out if isinstance(out, tuple) else (out,)
    return _Compiled(
        tuple(o.cpu().float() for o in outputs),
        formed,
        added,
        normalise("\n".join(code)),
        paths,
    )


def _relative_error(outputs, reference):
    reference = reference if isinstance(reference, tuple) else (reference,)
    return max(
        ((o - r).abs().max() / r.abs().max()).item()
        for o, r in zip(outputs, reference, strict=True)
    )


# name -> (function, input shapes, span limit, expected groups, expected added-op
# patterns). A group is (ops in it, [data type of each op], [trip count per level]).
CASES = {
    # The pointwise copy-out path: a tile is computed into a scratch buffer and a
    # coarse_tile_copy_* op writes it into the full output.
    "pointwise_copy_out": (
        lambda a, b: a + b,
        [(4096, 1024)] * 2,
        4 * MB,
        [(1, ["Pointwise"], [2])],
        ["op", r"coarse_tile_copy_"],
    ),
    # Two ops that tile the same way share one loop, each with its own copy-out.
    # They need not depend on each other, so a group is not always a chain.
    "pointwise_chain": (
        lambda a, b, c: (a + b) * c,
        [(4096, 1024)] * 3,
        4 * MB,
        [(2, ["Pointwise", "Pointwise"], [2])],
        ["op", r"coarse_tile_copy_"],
    ),
    "pointwise_independent_pair": (
        lambda a, b, c, d: (a + b, c * d),
        [(4096, 1024)] * 4,
        4 * MB,
        [(2, ["Pointwise", "Pointwise"], [2])],
        ["op", "op", r"coarse_tile_copy_", r"coarse_tile_copy_"],
    ),
    # A matmul tiled on an OUTPUT dim has no accumulator: like a pointwise op it
    # computes a tile into a scratch buffer and a coarse_tile_copy_* op writes it out.
    "matmul_output_tiled_m": (
        lambda a, b: a @ b,
        [(8192, 64), (64, 256)],
        2 * MB,
        [(1, ["Reduction"], [2])],
        ["op", r"coarse_tile_copy_"],
    ),
    "matmul_output_tiled_n": (
        lambda a, b: a @ b,
        [(256, 64), (64, 8192)],
        2 * MB,
        [(1, ["Reduction"], [2])],
        ["op", r"coarse_tile_copy_"],
    ),
    # Joined groups: only an output-range tile can join a producer or consumer.
    "matmul_pointwise_consumer": (
        lambda a, b: (a @ b) * 2.0,
        [(8192, 64), (64, 256)],
        2 * MB,
        [(2, ["Reduction", "Pointwise"], [2])],
        ["op", r"coarse_tile_copy_"],
    ),
    "matmul_add_second_input": (
        lambda a, b, c: (a @ b) + c,
        [(8192, 64), (64, 256), (8192, 256)],
        2 * MB,
        [(2, ["Reduction", "Pointwise"], [2])],
        ["op", r"coarse_tile_copy_"],
    ),
    "matmul_relu_consumer": (
        lambda a, b: torch.relu(a @ b),
        [(8192, 64), (64, 256)],
        2 * MB,
        [(2, ["Reduction", "Pointwise"], [2])],
        ["op", r"coarse_tile_copy_"],
    ),
    # The producer is smaller than the limit, so it is not tiled and not in the group.
    "matmul_pointwise_producer": (
        lambda a, b: (a * 2.0) @ b,
        [(8192, 64), (64, 256)],
        2 * MB,
        [(1, ["Reduction"], [2])],
        ["op", r"coarse_tile_copy_"],
    ),
    "bmm_output_tiled": (
        lambda a, b: torch.bmm(a, b),
        [(8, 2048, 64), (8, 64, 256)],
        4 * MB,
        [(1, ["Reduction"], [2])],
        ["op", r"coarse_tile_copy_"],
    ),
    # A matmul tiled on its REDUCTION dim has an accumulator (fill and combine) and
    # can never join a group: a K tile would break the loop-carried accumulation.
    "matmul_k_tiled": (
        lambda a, b: a @ b,
        [(2048, 2048), (2048, 256)],
        1 * MB,
        [(1, ["Reduction"], [8])],
        ["op", "op", r"_fill_", r"_combine_"],
    ),
}


def _matches(added, patterns):
    """``added`` names, in order, each matching its pattern; ``op`` means op<N>."""
    if len(added) != len(patterns):
        return False
    for name, pattern in zip(added, patterns, strict=True):
        if pattern == "op":
            if not re.fullmatch(r"op\d+", name):
                return False
        elif not re.search(pattern, name):
            return False
    return True


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_current_lowering_on_a_compiled_span_overflow_graph(case):
    fn, shapes, limit, groups, patterns = CASES[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    compiled = compile_span_overflow(fn, args, limit)

    assert compiled.groups == groups, compiled.groups
    assert _matches(compiled.added, patterns), (compiled.added, patterns)
    reference = fn(*[a.float() for a in args])
    # fp16 inputs scaled to 0.05 against an fp32 reference; the worst case here is
    # the K-tiled accumulation, about 4e-3. 2e-2 leaves headroom without hiding a
    # wrong tile or a NaN.
    assert _relative_error(compiled.outputs, reference) < 2e-2


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_generated_code_is_deterministic_after_normalising(case):
    """A comparison against a new lowering is only worth anything if two compiles
    of the same graph agree. Every case, in one process; for five of them
    (a copy-out, a K-tiled matmul, a joined group, an output-tiled matmul and an
    independent pair) two separate processes also gave identical text, which this
    does not repeat: a spurious difference from a token the normaliser misses
    would show up here as a failure."""
    fn, shapes, limit, _, _ = CASES[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    first = compile_span_overflow(fn, args, limit)
    second = compile_span_overflow(fn, args, limit)
    assert first.code == second.code


# Who applies each case's groups under CP-SAT. The pass keeps a K tile and a
# tiling of the stick dim, which the solve cannot take.
TILED_BY = {
    "pointwise_copy_out": ["solve"],
    "pointwise_chain": ["solve"],
    "pointwise_independent_pair": ["solve"],
    "matmul_output_tiled_m": ["solve"],
    "matmul_output_tiled_n": ["pass"],
    "matmul_pointwise_consumer": ["solve"],
    "matmul_add_second_input": ["solve"],
    "matmul_relu_consumer": ["solve"],
    "matmul_pointwise_producer": ["solve"],
    "bmm_output_tiled": ["solve"],
    "matmul_k_tiled": ["pass"],
}

_CPSAT = {
    "lx_planning": True,
    "co_optimizing_lx_planning": True,
    "layout_solver": "cpsat",
}

needs_ortools = pytest.mark.skipif(
    ilp_solver_ortools.cp_model is None,
    reason="the CP-SAT solve, which takes span-overflow tilings, needs ortools",
)


def test_every_case_says_who_tiles_it():
    assert sorted(TILED_BY) == sorted(CASES)


@needs_ortools
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_solve_takes_the_groups_it_can_and_builds_the_same_loops(case):
    """The same planner groups, the same ops added and the same result, whether
    the pass applies a group before the solve or the solve applies it after."""
    fn, shapes, limit, groups, patterns = CASES[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    with config.patch(_CPSAT):
        solved = compile_span_overflow(fn, args, limit)
        passed = compile_span_overflow(fn, args, limit, in_solve=False)

    assert passed.paths == ["pass"] * len(groups), passed.paths
    assert solved.paths == TILED_BY[case], solved.paths
    assert solved.groups == passed.groups == groups, (solved.groups, passed.groups)
    assert _matches(solved.added, patterns), (solved.added, patterns)
    reference = fn(*[a.float() for a in args])
    # As in test_the_current_lowering_on_a_compiled_span_overflow_graph.
    assert _relative_error(solved.outputs, reference) < 2e-2


@needs_ortools
def test_a_failed_solve_applies_the_groups_left_to_it():
    """When the solve fails, the groups left to it are applied before greedy
    placement, by the pass's own apply, so the compile still tiles them."""
    fn, shapes, limit, groups, patterns = CASES["pointwise_chain"]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    offered = []

    def fail(solver, cost_expr):
        offered.append(
            sorted(
                buf.name
                for buf in solver.buffers
                if any(
                    not cd.tiling.is_untiled
                    for cd in getattr(buf, "core_divisions", None) or ()
                )
            )
        )
        raise SolveError("failed by the test")

    with (
        config.patch(_CPSAT),
        mock.patch.object(
            ilp_solver_ortools.CpSatLayoutSolver,
            "plan_layout_and_core_divisions",
            fail,
        ),
    ):
        compiled = compile_span_overflow(fn, args, limit)

    # The solve was offered the chain's two tiled ops, and failed.
    assert len(offered) == 1 and len(offered[0]) == 2, offered
    assert compiled.paths == ["pass"], compiled.paths
    assert compiled.groups == groups, compiled.groups
    assert _matches(compiled.added, patterns), (compiled.added, patterns)
    reference = fn(*[a.float() for a in args])
    assert _relative_error(compiled.outputs, reference) < 2e-2


def test_solve_decides_span_overflow_tiling_matches_select_allocator():
    """The predicate holds exactly where span-overflow tiling and LX planning are
    on and ``select_allocator`` builds an allocator that takes the tilings."""
    from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator

    flags = (
        "ignore_span_overflow_hints",
        "lx_planning",
        "co_optimizing_lx_planning",
        "allow_exhaustive_search",
    )
    solvers = ("greedy", "bestfit", "firstfit", "simulated_annealing", "cpsat")
    cp_models = {None, ilp_solver_ortools.cp_model}
    for *values, solver, cp_model in itertools.product(
        *[(True, False)] * len(flags), solvers, cp_models
    ):
        settings = {**dict(zip(flags, values)), "layout_solver": solver}
        with (
            config.patch(settings),
            mock.patch.object(ilp_solver_ortools, "cp_model", cp_model),
        ):
            try:
                allocator = allocator_module.select_allocator()
            except ValueError:
                allocator = None
            expected = (
                not settings["ignore_span_overflow_hints"]
                and settings["lx_planning"]
                and isinstance(allocator, CoOptimizingAllocator)
                and allocator.takes_span_overflow_tilings
            )
            assert allocator_module.solve_decides_span_overflow_tiling() == expected, (
                settings,
                cp_model,
            )


def _stand_in(name):
    reads = SimpleNamespace(reads=[])
    return SimpleNamespace(
        get_name=lambda: name,
        get_operation_name=lambda: name,
        get_read_writes=lambda: reads,
    )


def test_a_group_is_left_to_the_solve_only_whole():
    """One member the solve cannot take keeps its whole group with the pass, and
    a graph with sharded placements keeps every group."""
    a, b, c = (_stand_in(name) for name in ("buf0", "buf1", "buf2"))
    groups = [([a, b], [(10000, 2)]), ([c], [(10001, 2)])]
    spec = TileSpec((TileAxis(host_dim=0, count=2),))
    hints = [(op, [object()]) for op in (a, b, c)]

    def for_solve(graph, op, *args):
        return (None, "a restickify") if op is b else (spec, None)

    def defer(**graph_attrs):
        graph = SimpleNamespace(operations=[a, b, c], **graph_attrs)
        kept = allocator_module.defer_span_overflow_tiling(graph, groups, hints, None)
        return kept, allocator_module.deferred_span_overflow_tilings(graph)

    with (
        mock.patch.object(allocator_module, "_DivisionPins", lambda *a, **k: None),
        mock.patch.object(
            allocator_module, "_span_overflow_tiling_for_solve", for_solve
        ),
    ):
        assert defer() == ([groups[0]], {"buf2": spec})
        assert defer(_spyre_sharded_placements={"x": object()}) == (groups, {})


def test_the_fallback_applies_the_groups_left_and_defers_the_ops_it_adds():
    """The ops the pass's apply adds join the deferred work division, as they
    would have when the pass applied the groups before work division."""
    from torch._inductor.ir import ComputedBuffer, Pointwise

    added = MagicMock(spec=ComputedBuffer, data=MagicMock(spec=Pointwise))
    added.get_name.return_value = "coarse_tile_copy_buf2"
    graph = SimpleNamespace(operations=[_stand_in("buf2")])
    setattr(graph, work_division._DEFERRED_WORK_DIVISION, {"buf2"})
    group = ([graph.operations[0]], [(10001, 2)])
    applied = []

    def apply(graph, groups):
        applied.append(groups)
        graph.operations.append(added)

    record = allocator_module._DeferredSpanOverflowTiling(
        {"buf2": None}, [group], apply
    )
    setattr(graph, allocator_module._DEFERRED_SPAN_OVERFLOW_TILING, record)
    allocator_module.apply_deferred_span_overflow_tiling(graph)
    allocator_module.apply_deferred_span_overflow_tiling(graph)  # nothing left

    assert applied == [[group]]
    deferred = work_division.deferred_work_division(graph)
    assert deferred == {"buf2", "coarse_tile_copy_buf2"}


def test_an_op_left_to_the_solve_is_offered_its_tiling_alone():
    """Whatever ``auto_coarse_tiling`` says, and with no untiled option."""
    from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator

    spec = TileSpec((TileAxis(host_dim=0, count=2),))
    op = SimpleNamespace(
        loop_info=None, get_operation_name=lambda: "op0", get_name=lambda: "buf0"
    )
    allocator = SimpleNamespace(_forced_tilings={"buf0": spec})
    for auto in (False, True):
        with config.patch({"auto_coarse_tiling": auto}):
            assert CoOptimizingAllocator._tiling_candidates(allocator, op, 4) == [spec]


@expects_raise
def test_a_two_level_group_is_rejected():
    """A matmul too large for one level nests two (``[4, 4]``). The lowering builds
    one standalone loop per group, so the group is refused (#6088).

    TODO: when nested loops are supported this becomes a positive test of the
    two-level group; the case compiled before the rejection and matched fp32 to
    4e-3."""
    args = [_shaped((4096, 4096), seed) for seed in range(2)]
    with pytest.raises(
        InductorError, match=r"nested loop tiling is not supported.*#6088"
    ):
        compile_span_overflow(lambda a, b: a @ b, args, 8 * MB)


# The cases the planner already rejects, and why. A replacement keeps rejecting them.
REJECTED = {
    # The second matmul's tiled dim cannot be joined to its producer's group.
    "bmm_chain": (
        lambda a, b, c: torch.bmm(torch.bmm(a, b), c),
        [(4, 2048, 64), (4, 64, 256), (4, 256, 64)],
        512 * 1024,
        "it reads auto-tiled producer",
    ),
    # A pointwise op larger than one tile level can hold is not tiled further, and
    # the post-planning span check rejects what remains.
    "pointwise_needing_two_levels": (
        lambda a, b: a + b,
        [(4096, 4096)] * 2,
        4 * MB,
        "per-core tensor span",
    ),
}


@pytest.mark.parametrize("case", sorted(REJECTED))
@expects_raise
def test_an_unsupported_graph_is_rejected(case):
    fn, shapes, limit, message = REJECTED[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    with pytest.raises(InductorError, match=message):
        compile_span_overflow(fn, args, limit)


class _NamedOp:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


def test_one_level_groups_are_accepted_and_the_check_does_nothing():
    ops = [_NamedOp("buf0")]
    reject_nested_loops([], "x")
    reject_nested_loops([(ops, [(0, 4)])], "x")
    reject_nested_loops([(ops, [(0, 4)]), (ops, [(1, 2)])], "x")


def test_a_group_with_two_levels_is_rejected_naming_the_caller_and_the_ops():
    ops = [_NamedOp("buf0"), _NamedOp("buf1")]
    with pytest.raises(
        Unsupported, match=r"my pass: nested loop tiling.*buf0, buf1.*#6088"
    ):
        reject_nested_loops([(ops, [(0, 4), (1, 4)])], "my pass")


def test_one_nested_group_among_flat_ones_is_still_found():
    flat = [_NamedOp("buf0")]
    nested = [_NamedOp("buf1")]
    with pytest.raises(Unsupported, match="3 levels over buf1"):
        reject_nested_loops([(flat, [(0, 2)]), (nested, [(1, 2), (2, 2), (3, 2)])], "x")


def test_the_allocator_asks_the_enumerator_for_one_level_only():
    """The SA co-optimizer raises rather than falls back when the apply refuses a
    tiling, so the search must not offer one the lowering will refuse (#6088).

    Calls the allocator's two real entry points, ``_tiling_candidates`` (the menu)
    and ``_division_space`` (the solver's tiling space), on a stand-in ``self``, with
    the enumerator mocked, and checks the ``max_dims`` each one passes. The
    enumerator's own behaviour at ``max_dims=1`` is tested in
    ``test_enumerate_tilings.test_max_dims_one_gives_no_nested_specs``."""
    from types import SimpleNamespace

    from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator
    from torch_spyre._inductor.scratchpad.plan_solver import TileSpec
    from torch_spyre._inductor.wsr import enumerate_tilings

    calls = []

    def fake_options(op, **kwargs):
        calls.append(("enumerate_tile_options", kwargs.get("max_dims")))
        return [TileSpec()]

    def fake_space(op, **kwargs):
        calls.append(("build_tiling_space", kwargs.get("max_dims")))
        return None

    op = SimpleNamespace(
        loop_info=None, get_operation_name=lambda: "buf0", get_name=lambda: "buf0"
    )
    allocator = SimpleNamespace(
        _tilings_from_menu=True,
        _prescribed_ops=(),
        _readers_by_name={},
        _get_op_name=lambda o: "pointwise",
        _solver_chooses_tilings=True,
    )
    with (
        config.patch({"auto_coarse_tiling": True}),
        mock.patch.object(enumerate_tilings, "enumerate_tile_options", fake_options),
        mock.patch.object(enumerate_tilings, "build_tiling_space", fake_space),
    ):
        CoOptimizingAllocator._tiling_candidates(allocator, op, 4)
        CoOptimizingAllocator._division_space(allocator, op)

    assert MAX_SEARCH_TILE_DEPTH == 1
    assert calls == [
        ("enumerate_tile_options", MAX_SEARCH_TILE_DEPTH),
        ("build_tiling_space", MAX_SEARCH_TILE_DEPTH),
    ], calls


def test_no_third_enumerator_call_in_the_allocator_escapes_the_bound():
    """The test above covers the two search call sites; the third checks a
    span-overflow tiling left to the solve. This is the structural
    backstop for a new one: every call to the enumerator in ``allocator.py`` must
    pass ``max_dims=MAX_SEARCH_TILE_DEPTH``. (``enumerate_tilings.py`` calls itself
    with the bound it was given, and the allocator is its only other caller.)"""
    import ast
    import inspect

    from torch_spyre._inductor.scratchpad import allocator

    tree = ast.parse(inspect.getsource(allocator))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None))
        in ("enumerate_tile_options", "build_tiling_space")
    ]
    assert len(calls) == 3, [ast.unparse(c)[:60] for c in calls]
    for call in calls:
        bound = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}.get("max_dims")
        assert bound == "MAX_SEARCH_TILE_DEPTH", ast.unparse(call)


@pytest.mark.parametrize("limit", [256 * 1024, 16 * 1024], ids=["256KB", "16KB"])
def test_a_nested_for_each_tile_loop_is_not_affected(limit):
    """for_each_tile builds its own nests and never reaches the rejection.

    ``span_overflow_groups`` skips an op that carries ``dim_hints``, which the
    ``for_each_tile`` lowering stamps, so no group forms over a loop body however
    low the limit goes (16 KB tempts it to tile everything: each tensor here is
    128 KB or less). The test checks that directly: no group formed, so there was
    nothing for ``reject_nested_loops`` to refuse, and the nest is intact and right.

    The 256 KB case is a control (nothing overflows). The 16 KB case is the one
    with power: with that skip removed the planner groups the loop body and the
    compile fails with "group (1,) would overwrite the existing loop_info of
    while_loop_body_graph_0_0_op5"."""
    from for_each_tile_fixtures import nested_split_m_then_k_fn

    x = _shaped((256, 256), 0)
    y = _shaped((256, 64), 1)
    compiled = compile_span_overflow(nested_split_m_then_k_fn, [x, y], limit)
    assert compiled.groups == [], compiled.groups
    assert _relative_error(compiled.outputs, x.float() @ y.float()) < 2e-2
    # still two nested loops: the nest the for_each_tile lowering built is intact
    assert compiled.code.count("LoopSpec(") >= 2, compiled.code.count("LoopSpec(")


def _forced_one_level(shape, levels, make_input):
    """Compile ``abs(x) + 1`` with the span-overflow plan forced to ``levels``.

    Reads the generated code directly: ``compare_with_cpu``'s ``source_check`` is
    never called (it compares ``device == "spyre"`` where ``device`` is a
    ``torch.device``, which is ``False``), so a check passed there asserts nothing.
    Returns ``(fraction of elements more than 0.05 from fp32, loop count, trip
    counts)``. A wrong layout puts whole tiles in the wrong place, so the
    max-normalised error of the other helpers is misleading here."""
    from test_span_overflow_hint_analysis import _forced_span_plan

    torch.manual_seed(0xAFFE)
    x = make_input()
    reference = torch.abs(x.float()) + 1
    torch._dynamo.reset()
    with (
        mock.patch(
            "torch_spyre._inductor.wsr.coarse_tile_span_overflow."
            "plan_span_overflow_tile",
            _forced_span_plan(shape, levels),
        ),
        config.patch({"sencores": 4, "ignore_span_overflow_hints": False}),
        torch._inductor.config.patch(force_disable_caches=True),
    ):
        compiled = torch.compile(
            lambda x: torch.abs(x) + 1, backend="inductor", fullgraph=True
        )
        out, code = run_and_get_code(compiled, x.to(DEVICE_NAME))
    source = "\n".join(code)
    counts = sorted({int(c) for c in re.findall(r"count=sympify\('(\d+)'\)", source)})
    wrong = ((out.cpu().float() - reference).abs() > 0.05).float().mean().item()
    return wrong, source.count("LoopSpec("), counts


_PERMUTE_SHAPE = (8, 8, 64, 512)


def _plain():
    return torch.randn(*_PERMUTE_SHAPE, dtype=torch.float16)


def _permuted():
    return torch.randn(*_PERMUTE_SHAPE, dtype=torch.float16).permute(1, 0, 2, 3)


@pytest.mark.parametrize(
    "make_input, levels, trips",
    [
        (_plain, [(0, 8)], [8]),
        (_plain, [(1, 8)], [8]),
        (_permuted, [(0, 4)], [4]),
    ],
    ids=["plain_8_dim0", "plain_8_dim1", "permuted_4_dim0"],
)
def test_a_forced_one_level_plan_of_a_buffer_is_right(make_input, levels, trips):
    """Controls for the permuted test below: the forced plan is applied (one loop
    with the forced trip count, read from the generated code) and the result is
    right for a plain buffer at 8 tiles and for the permuted buffer at 4."""
    wrong, loops, counts = _forced_one_level(_PERMUTE_SHAPE, levels, make_input)
    assert (loops, counts) == (1, trips), (loops, counts)
    assert wrong == 0.0, wrong


@pytest.mark.parametrize("dim", [0, 1])
def test_a_permuted_buffer_tiled_eight_ways_at_one_level_is_right(dim):
    """The stride coincidence of ``test_permuted_same_size_dims_tile_numeric``: the
    tile count equals the size of the two leading dims. Reached with one level when
    the plan is forced (the planner picks ``[4, 4]``, which is right and is that
    test, refused today because it nests).

    Host dim 0 of this buffer tiled 8 ways is squeezed to extent 1 in the copy-out.
    Its per-trip advance was taken as the product of the sizes to its right (262144,
    which is dim 1's stride), where the buffer's own stride is 32768, so tiles 1 to 7
    were written to the wrong place (#6093). It is now the buffer's stride."""
    wrong, loops, counts = _forced_one_level(_PERMUTE_SHAPE, [(dim, 8)], _permuted)
    assert (loops, counts) == (1, [8]), (loops, counts)
    assert wrong == 0.0, wrong
