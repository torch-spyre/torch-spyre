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
from unittest import mock

import pytest
import regex as re
import torch
from torch._inductor.utils import run_and_get_code

import torch_spyre  # noqa: F401  registers the "spyre" device
from torch_spyre._inductor import config
from torch_spyre._inductor import nonstick_dim_order, passes, work_division
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.scratchpad.coarse_tiling import MAX_SEARCH_TILE_DEPTH
from torch_spyre._inductor.wsr import coarse_tile, span_overflow_hint_analysis
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
    def __init__(self, outputs, groups, added, code):
        self.outputs = outputs
        self.groups = groups  # [(ops in group, [data type of each], [trip counts])]
        self.added = added  # names of ops the transform added, in order
        self.code = code  # normalised generated code


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


def compile_span_overflow(fn, args, limit_bytes):
    """Compile ``fn`` with span-overflow on and the span limit lowered."""
    formed, added = [], []
    original = coarse_tile.coarse_tile_post_stickify

    def spy(graph, groups, group_idx_offset=0):
        before = {op.get_operation_name() for op in graph.operations}
        formed.extend(
            (
                len(ops),
                [type(getattr(op, "data", op)).__name__ for op in ops],
                [int(count) for _, count in levels],
            )
            for ops, levels in groups
        )
        result = original(graph, groups=groups, group_idx_offset=group_idx_offset)
        added.extend(
            op.get_operation_name()
            for op in graph.operations
            if op.get_operation_name() not in before
        )
        return result

    torch._dynamo.reset()
    with (
        mock.patch.object(coarse_tile, "coarse_tile_post_stickify", spy),
        mock.patch.object(passes, "coarse_tile_post_stickify", spy),
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


@pytest.mark.parametrize(
    "case", ["pointwise_copy_out", "matmul_k_tiled", "matmul_pointwise_consumer"]
)
def test_the_generated_code_is_deterministic_after_normalising(case):
    """A comparison against a new lowering is only worth anything if two compiles
    of the same graph agree."""
    fn, shapes, limit, _, _ = CASES[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    first = compile_span_overflow(fn, args, limit)
    second = compile_span_overflow(fn, args, limit)
    assert first.code == second.code


@expects_raise
def test_a_two_level_group_is_rejected():
    """A matmul too large for one level nests two (``[4, 4]``). The lowering builds
    one standalone loop per group, so the group is refused (#6088).

    TODO: when nested loops are supported this becomes a positive test of the
    two-level group; the case compiled before the rejection and matched fp32 to
    4e-3."""
    args = [_shaped((4096, 4096), seed) for seed in range(2)]
    with pytest.raises(Exception, match=r"nested loop tiling is not supported.*#6088"):
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
    with pytest.raises(Exception, match=message):
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


def test_the_allocator_search_is_bounded_to_one_level():
    """The SA co-optimizer raises rather than falls back when the apply refuses a
    tiling, so the search must not offer one the lowering will refuse (#6088).

    This is a structural check, not a behavioural one: the two places the
    allocator asks the enumerator for tilings sit behind ``auto_coarse_tiling`` and
    solver selection on a fully built allocator, so reaching them needs more
    stubbing than it is worth. What it does catch is the realistic regression, a
    new enumerator call without the bound. The enumerator itself is tested in
    ``test_enumerate_tilings.test_max_dims_one_gives_no_nested_specs``."""
    import ast
    import inspect

    from torch_spyre._inductor.scratchpad import allocator

    assert MAX_SEARCH_TILE_DEPTH == 1
    tree = ast.parse(inspect.getsource(allocator))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None))
        in ("enumerate_tile_options", "build_tiling_space")
    ]
    assert len(calls) == 2, [ast.unparse(c)[:60] for c in calls]
    for call in calls:
        bound = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}.get("max_dims")
        assert bound == "MAX_SEARCH_TILE_DEPTH", ast.unparse(call)


def test_a_nested_for_each_tile_loop_is_not_affected():
    """for_each_tile builds its own nests and never reaches the rejection. Run one
    with span-overflow on: it must still compile and be right."""
    from for_each_tile_fixtures import nested_split_m_then_k_fn

    x = _shaped((256, 256), 0)
    y = _shaped((256, 64), 1)
    reference = x.float() @ y.float()
    torch._dynamo.reset()
    with (
        config.patch({"ignore_span_overflow_hints": False}),
        torch._inductor.config.patch(force_disable_caches=True),
    ):
        out = torch.compile(
            nested_split_m_then_k_fn, backend="inductor", fullgraph=True
        )(x.to(DEVICE_NAME), y.to(DEVICE_NAME))
    assert _relative_error((out.cpu().float(),), reference) < 2e-2
