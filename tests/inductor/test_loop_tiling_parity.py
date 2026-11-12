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
# The slices are 256 rows: with 8 the work division of the consumer is a cost-model
# tie that legacy alone resolves differently from one compile to the next.
EXTRA_CASES = {
    "pointwise_outside_consumers": (
        lambda a, b: (lambda y: y[:256] * 2 + y[256:512])(a + b),
        [(4096, 1024)] * 2,
        4 * MB,
    ),
    "pointwise_outside_consumer_and_graph_output": (
        lambda a, b: (lambda y: (y, y[:256] * 2))(a + b),
        [(4096, 1024)] * 2,
        4 * MB,
    ),
}
# A bias vector read by a joined consumer tiled on N: its buffer is rank 1 where the op
# iterates two dims, and its index still holds the tiled dim's symbol.
EXTRA_CASES["matmul_bias_tiled_n"] = (
    lambda a, b, c: (a @ b) + c,
    [(256, 64), (64, 8192), (8192,)],
    2 * MB,
)
PLAN_VS_LEGACY = [
    "pointwise_copy_out",
    "pointwise_chain",
    "pointwise_independent_pair",
    *EXTRA_CASES,
    "matmul_output_tiled_m",
    "matmul_output_tiled_n",
    "matmul_pointwise_consumer",
    "matmul_add_second_input",
    "matmul_relu_consumer",
    "matmul_pointwise_producer",
    "bmm_output_tiled",
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
def test_plan_matches_legacy_kernel_text(case):
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


_SHAPE = (8, 8, 64, 512)


def _forced(lowering, levels, make_input):
    """``_forced_one_level`` under ``lowering``, also returning the normalised
    generated code."""
    from torch._inductor.utils import run_and_get_code

    import test_span_overflow_gate as gate
    from test_span_overflow_loop_terms import _normalised

    code = []

    def capture(*args, **kwargs):
        out, source = run_and_get_code(*args, **kwargs)
        code.append("\n".join(source))
        return out, source

    with (
        mock.patch.object(gate, "run_and_get_code", capture),
        config.patch({"loop_tiling_lowering": lowering}),
    ):
        result = gate._forced_one_level(_SHAPE, levels, make_input)
    return result, _normalised(code[0])


@pytest.mark.parametrize("which, dim", [("plain", 0), ("permuted", 0), ("permuted", 1)])
def test_forced_plans_match_legacy_kernel_text(which, dim):
    """The permuted buffer tiled 8 ways at one level: dim 0 is squeezed to extent 1,
    and its step is the buffer's own stride (#6093). The plan reads it from the index
    before the divide."""
    from test_span_overflow_gate import _permuted, _plain

    make = {"plain": _plain, "permuted": _permuted}[which]
    legacy, legacy_code = _forced("legacy", [(dim, 8)], make)
    plan, plan_code = _forced("plan", [(dim, 8)], make)
    assert plan == legacy
    assert plan[0] == 0.0 and plan[1:] == (1, [8]), plan
    assert plan_code == legacy_code


def _forced_view(lowering, view, levels):
    """``abs(view(x)) + 1`` with the span-overflow plan forced to ``levels``."""
    from test_span_overflow_hint_analysis import _forced_span_plan
    from torch._inductor.utils import run_and_get_code

    from torch_spyre.constants import DEVICE_NAME

    torch.manual_seed(0xAFFE)
    x = torch.randn(64, 8, 512, dtype=torch.float16)
    fn = lambda x: torch.abs(view(x)) + 1  # noqa: E731
    shape = tuple(view(x).shape)
    torch._dynamo.reset()
    with (
        mock.patch(
            "torch_spyre._inductor.wsr.coarse_tile_span_overflow."
            "plan_span_overflow_tile",
            _forced_span_plan(shape, levels),
        ),
        config.patch(
            {
                "sencores": 4,
                "ignore_span_overflow_hints": False,
                "loop_tiling_lowering": lowering,
            }
        ),
        torch._inductor.config.patch(force_disable_caches=True),
    ):
        out, _ = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), x.to(DEVICE_NAME)
        )
    return ((out.cpu().float() - fn(x.float())).abs() > 0.05).float().mean().item()


@expects_raise
def test_a_read_through_a_permuted_view_is_refused_under_plan():
    """The op iterates (8, 64, 512) over an input stored (64, 8, 512): its dim 0 is the
    buffer's dim 1, so 'dim 0' of the buffer has another stride (4096, not 512).
    Legacy compiles it (it leaves the group whole); the plan refuses it, naming the
    view, rather than carry a step for a view it does not model.

    TODO: when a read through a reordering view is lowered, this becomes a positive
    test."""
    permute = lambda t: t.permute(1, 0, 2)  # noqa: E731
    assert _forced_view("legacy", permute, [(0, 8)]) == 0.0
    with pytest.raises(Exception, match="view that reorders"):
        _forced_view("plan", permute, [(0, 8)])


def test_a_read_through_a_plain_view_is_converted_under_plan():
    assert _forced_view("plan", lambda t: t, [(0, 64)]) == 0.0


@pytest.mark.parametrize(
    "lowering",
    [
        pytest.param(
            lowering,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="#6243: span-overflow tiling returns wrong values for a "
                "transposed consumer of a tiled pointwise producer (about 60% of "
                "elements); the grouping puts the consumer in the producer's loop, "
                "so the plan gives the same wrong values as the legacy lowering",
            ),
        )
        for lowering in ("legacy", "plan")
    ],
)
def test_a_transposed_consumer_is_checked_numerically_only(lowering):
    """Numerics only: the two lowerings agree on the values, not on a text that
    #6243 makes meaningless."""
    fn = lambda a, b: (a + b).t() * 2  # noqa: E731
    args = [_shaped((4096, 1024), seed) for seed in range(2)]
    compiled = compile_with(lowering, fn, args, 4 * MB)
    reference = fn(*[a.float() for a in args])
    wrong = ((compiled.outputs[0] - reference).abs() > 0.05).float().mean().item()
    assert wrong == 0.0, wrong
