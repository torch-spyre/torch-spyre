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

"""Parity of the loop-tiling lowering, one compile per lowering setting.

``LOWERINGS`` names the settings of ``config.loop_tiling_lowering`` every test here
runs under; ``compile_with`` compiles under one of them. The first tests pin cases
the gate (``test_span_overflow_gate``) does not reach, on the legacy transform, so a
later lowering is held to what legacy does, not to what it was thought to do.
"""

from unittest import mock

import pytest
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)

from torch_spyre._inductor import config, passes

from utils_inductor import expects_raise
from test_span_overflow_gate import (
    CASES,
    MB,
    _relative_error,
    _shaped,
    compile_span_overflow,
)

LOWERINGS = ["legacy"]  # the pinned reduction cases below: "plan" lowers pointwise only
# Cases beyond the gate's: a tiled producer read outside the loop (by a small slice
# that is not itself tiled), with and without the producer being a graph output.
EXTRA_CASES = {
    "pointwise_outside_consumers": (
        lambda a, b: (lambda y: y[:8] * 2 + y[8:16])(a + b),
        [(4096, 1024)] * 2,
        4 * MB,
    ),
    "pointwise_outside_consumer_and_graph_output": (
        lambda a, b: (lambda y: (y, y[:8] * 2))(a + b),
        [(4096, 1024)] * 2,
        4 * MB,
    ),
}
PLAN_VS_LEGACY = [
    "pointwise_copy_out",
    "pointwise_chain",
    "pointwise_independent_pair",
    *EXTRA_CASES,
]


def compile_with(lowering, fn, args, limit_bytes):
    """``compile_span_overflow`` under ``config.loop_tiling_lowering == lowering``."""
    with config.patch({"loop_tiling_lowering": lowering}):
        return compile_span_overflow(fn, args, limit_bytes)


def _is_k_tiled(compiled):
    """A fill before the loop and a combine in it: an accumulator was introduced."""
    return any("_fill_" in n for n in compiled.added) and any(
        "_combine_" in n for n in compiled.added
    )


def _signed_powers_of_two(shape, seed, scale):
    """Mostly 1 with a few 2 and 0.5: every product is exact in fp16 and moderate."""
    generator = torch.Generator().manual_seed(seed)
    pick = torch.rand(*shape, generator=generator)
    values = torch.ones(*shape)
    values[pick < scale] = 2.0
    values[pick > 1 - scale] = 0.5
    return values.half()


# name -> (reduction, input builder). The inputs are chosen so a wrong identity fill
# shows: an amax of negative values (a 0 fill wins), an amin of positive values, and a
# sum of zero-mean values (a few units per row, so a fill of 1 is a visible shift).
_REDUCTIONS = {
    "sum": (lambda x: x.sum(dim=1), lambda: _shaped((64, 8192), 1)),
    "amax": (lambda x: x.amax(dim=1), lambda: -(_shaped((64, 8192), 2).abs() + 0.5)),
    "amin": (lambda x: x.amin(dim=1), lambda: _shaped((64, 8192), 3).abs() + 0.5),
}


@pytest.mark.parametrize("lowering", LOWERINGS)
@pytest.mark.parametrize("name", sorted(_REDUCTIONS))
def test_a_k_tiled_reduction_over_a_tiled_dim_is_right(lowering, name):
    fn, make = _REDUCTIONS[name]
    x = make()
    compiled = compile_with(lowering, fn, [x], 256 * 1024)
    assert _is_k_tiled(compiled), compiled.added
    assert _relative_error(compiled.outputs, fn(x.float())) < 2e-2


@pytest.mark.parametrize("lowering", LOWERINGS)
def test_a_k_tiled_reduction_in_fp32(lowering):
    """The accumulator and its combine keep fp32 (a fp32 matmul is not supported by
    the backend, so the fp32 accumulator is exercised through a sum)."""

    def fn(x):
        return x.sum(dim=1)

    x = _shaped((64, 8192), 7).float()
    compiled = compile_with(lowering, fn, [x], 512 * 1024)
    assert _is_k_tiled(compiled), compiled.added
    assert _relative_error(compiled.outputs, fn(x)) < 2e-2


@pytest.mark.parametrize("lowering", LOWERINGS)
@expects_raise
def test_a_prod_over_a_tiled_dim_is_rejected(lowering):
    """``prod`` over a span-overflowing dim is refused: the auto-tiler cannot join the
    reduction with its producer.

    TODO: when it can, this becomes a positive test with an identity fill of 1
    (the input is powers of two, so every product is exact in fp16)."""
    x = _signed_powers_of_two((64, 8192), 4, 0.0015)
    with pytest.raises(Exception, match="Cannot auto-tile"):
        compile_with(lowering, lambda x: x.prod(dim=1), [x], 256 * 1024)


@pytest.mark.parametrize("lowering", LOWERINGS)
def test_a_k_tiled_matmul_whose_accumulator_is_a_graph_output(lowering):
    """The matmul result is returned itself and also read by a later op."""

    def fn(a, b):
        c = a @ b
        return c, c * 2

    args = [_shaped(shape, i) for i, shape in enumerate([(2048, 2048), (2048, 256)])]
    compiled = compile_with(lowering, fn, args, 1 * MB)
    assert _is_k_tiled(compiled), compiled.added
    assert _relative_error(compiled.outputs, fn(*[a.float() for a in args])) < 2e-2


def _nodes(fn, args, limit, lowering):
    """Compile under ``lowering``; also return the loop nodes (name, trip count,
    leaf op names) as the pass loop leaves them just before the scheduler."""
    from test_span_overflow_loop_terms import _nodes_of

    from torch_spyre._inductor.loop_node import all_ops

    seen = []
    original = passes._hand_off_loop_nodes

    def spy(graph):
        seen.extend(
            (
                n.get_name(),
                n.trip_count,
                [o.get_operation_name() for o in all_ops(n.body)],
            )
            for n in _nodes_of(graph.operations)
        )
        return original(graph)

    with mock.patch.object(passes, "_hand_off_loop_nodes", spy):
        compiled = compile_with(lowering, fn, args, limit)
    return compiled, seen


@pytest.mark.parametrize("case", PLAN_VS_LEGACY)
def test_pointwise_plan_matches_legacy_kernel_text(case):
    """The same graph under both lowerings: the same groups, the same added ops, the
    same loop nodes, the same normalised kernel text (LX offsets included) and the
    same numbers."""
    from test_span_overflow_loop_terms import _normalised

    fn, shapes, limit, *_ = {**CASES, **EXTRA_CASES}[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    legacy, legacy_nodes = _nodes(fn, args, limit, "legacy")
    plan, plan_nodes = _nodes(fn, args, limit, "plan")
    assert legacy_nodes, "no loop node was built"
    assert plan_nodes == legacy_nodes
    assert plan.groups == legacy.groups
    assert plan.added == legacy.added
    assert _normalised(plan.code) == _normalised(legacy.code)
    reference = fn(*[a.float() for a in args])
    assert _relative_error(plan.outputs, reference) < 2e-2
