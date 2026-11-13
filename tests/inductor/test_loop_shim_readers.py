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

"""Readers of a loop shim op that need the device (the device-free ones are in
``test_loop_shim.py``).

The cost features price an op's traffic by the levels at which its operands advance.
For a ``CoarseTileInfo`` op that verdict is stamped in the stored lists; a
``LoopShim`` has no lists, so the reader falls back to the index rule (an operand
advances at a level iff its index has a nonzero coefficient on that level's loop
variable). This test pins that the two give the same features, on real compiled ops.
"""

import dataclasses
from unittest import mock

import contextlib

import pytest
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)
from torch._inductor.ir import ComputedBuffer

from torch_spyre._inductor import config, passes
from torch_spyre._inductor.cost_features import extract_op_features
from torch_spyre._inductor.loop_info import CoarseTileInfo, LoopShim
from torch_spyre._inductor.loop_node import all_ops
from torch_spyre._inductor.wsr import coarse_tile

from test_span_overflow_gate import CASES, _shaped, compile_span_overflow


def _shim_of(info):
    return LoopShim(
        loop_group_id=tuple(info.loop_group_id),
        loop_count=list(info.loop_count),
        loop_tiled_dims=[list(d) for d in info.loop_tiled_dims],
        loop_tiled_reduction_dims=[list(d) for d in info.loop_tiled_reduction_dims],
    )


def _features_both_ways(case):
    """For each looped op of ``case``: (name, features with its CoarseTileInfo,
    features with the equal LoopShim)."""
    rows = []
    original = coarse_tile.coarse_tile_post_stickify

    def spy(graph, groups, group_idx_offset=0):
        result = original(graph, groups=groups, group_idx_offset=group_idx_offset)
        for op in all_ops(graph.operations):
            info = getattr(op, "loop_info", None)
            if not isinstance(op, ComputedBuffer) or not isinstance(
                info, CoarseTileInfo
            ):
                continue
            legacy = extract_op_features(op)
            op.loop_info = _shim_of(info)
            try:
                shim = extract_op_features(op)
            finally:
                op.loop_info = info
            rows.append((op.get_operation_name(), legacy, shim))
        return result

    fn, shapes, limit, *_ = CASES[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    with (
        mock.patch.object(coarse_tile, "coarse_tile_post_stickify", spy),
        mock.patch.object(passes, "coarse_tile_post_stickify", spy),
        config.patch({"loop_tiling_lowering": "legacy"}),
    ):
        compile_span_overflow(fn, args, limit)
    return rows


@pytest.mark.parametrize(
    "case",
    [
        "pointwise_copy_out",
        "pointwise_chain",
        "matmul_output_tiled_n",
        "matmul_pointwise_consumer",
        "matmul_k_tiled",
    ],
)
def test_cost_features_of_a_shim_op_equal_those_of_its_coarsetileinfo(case):
    rows = _features_both_ways(case)
    assert rows, "the case formed no loop"
    for name, legacy, shim in rows:
        assert dataclasses.asdict(shim) == dataclasses.asdict(legacy), name


# ---------------------------------------------------------------------------
# End to end: the same compile with the shim swapped in must not change the kernel
# ---------------------------------------------------------------------------

from test_span_overflow_loop_terms import _normalised  # noqa: E402
from test_span_overflow_gate import _relative_error  # noqa: E402


@contextlib.contextmanager
def _shim_swapped_in(enabled=True):
    """Inside the block every op the transform stamped gets a ``LoopShim`` carrying
    the same four fields in place of its ``CoarseTileInfo``, straight after the
    transform, so every later reader (cost features, work division, LX planning,
    the kernel) sees a shim op."""
    original = coarse_tile.coarse_tile_post_stickify

    def spy(graph, groups, group_idx_offset=0):
        result = original(graph, groups=groups, group_idx_offset=group_idx_offset)
        if enabled:
            for op in all_ops(graph.operations):
                info = getattr(op, "loop_info", None)
                if isinstance(info, CoarseTileInfo):
                    op.loop_info = _shim_of(info)
        return result

    with (
        mock.patch.object(coarse_tile, "coarse_tile_post_stickify", spy),
        mock.patch.object(passes, "coarse_tile_post_stickify", spy),
        config.patch({"loop_tiling_lowering": "legacy"}),
    ):
        yield


def _compile(case, *, swap_in_the_shim):
    fn, shapes, limit, *_ = CASES[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    with _shim_swapped_in(swap_in_the_shim):
        compiled = compile_span_overflow(fn, args, limit)
    return compiled, _relative_error(compiled.outputs, fn(*[a.float() for a in args]))


@pytest.mark.parametrize("case", sorted(CASES))
def test_swapping_in_the_shim_does_not_change_the_kernel(case):
    with_info, error_info = _compile(case, swap_in_the_shim=False)
    with_shim, error_shim = _compile(case, swap_in_the_shim=True)
    assert _normalised(with_shim.code) == _normalised(with_info.code)
    assert with_shim.groups == with_info.groups
    assert with_shim.added == with_info.added
    assert error_shim == error_info < 2e-2


@pytest.mark.parametrize("dim", [0, 1])
def test_a_dim_squeezed_to_extent_one_still_advances_with_the_shim(dim):
    """Host dim 0 of the permuted buffer tiled 8 ways is squeezed out of the copy-out
    (no ``d{i}`` symbol survives), so nothing in ``loop_tiled_dims`` names its level;
    only the index says it advances. Without the shim's advancing levels the level
    symbol is never minted and the address stays pinned across the trips."""
    from test_span_overflow_gate import _forced_one_level, _permuted

    with _shim_swapped_in():
        wrong, loops, counts = _forced_one_level((8, 8, 64, 512), [(dim, 8)], _permuted)
    assert (loops, counts) == (1, [8]), (loops, counts)
    assert wrong == 0.0, wrong
