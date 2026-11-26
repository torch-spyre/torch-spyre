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

"""What the loop-tiling lowering does to real graphs beyond the span-overflow gate.

``test_span_overflow_gate.py`` pins the gate's cases (pointwise, matmul, joined, bmm,
K-tiled); this file adds the cases that gate does not reach: a tiled producer read
outside the loop, a bias vector read by a joined consumer tiled on N, a bmm that
consumes a tiled bmm, K-tiled reductions with the identity fills that show a wrong
fill, an fp32 accumulator and a graph-output accumulator, plus the squeezed and
permuted forced plans.

Each case asserts what the lowering builds (groups, the ops it adds, the loop node)
and that the values match an fp32 reference. There is no second lowering to compare
against; the expectations are the table below.
"""

import pytest
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)

from test_span_overflow_gate import (
    MB,
    _forced_one_level,
    _matches,
    _permuted,
    _plain,
    _relative_error,
    _shaped,
    compile_span_overflow,
)
from test_span_overflow_loop_terms import _compile_seeing_nodes
from utils_inductor import expects_raise


def _inputs(*shapes):
    return lambda: [_shaped(shape, seed) for seed, shape in enumerate(shapes)]


def _signed_powers_of_two(shape, seed, scale):
    """Mostly 1 with a few 2 and 0.5: every product is exact in fp16 and moderate."""
    generator = torch.Generator().manual_seed(seed)
    pick = torch.rand(*shape, generator=generator)
    values = torch.ones(*shape)
    values[pick < scale] = 2.0
    values[pick > 1 - scale] = 0.5
    return values.half()


def _matmul_and_its_double(a, b):
    c = a @ b
    return c, c * 2


# A K-tiled reduction adds its first op, the accumulator, a fill (outside the loop) and
# a combine (inside it). The inputs are chosen so a wrong identity fill shows: an amax of
# negative values (a 0 fill wins), an amin of positive values, and a sum of zero-mean
# values (a few units per row, so a fill of 1 is a visible shift).
_K_TILED = ["op", "op", r"_fill_", r"_combine_"]
_COPY_OUT = ["op", r"coarse_tile_copy_"]

# name -> (function, inputs, span limit, groups, added-op patterns, loop trips, ops in
# the loop node). A group is (ops, [data type of each], [trip count per level]).
LOWERED = {
    # A tiled producer read outside the loop, by small slices of 256 rows that are not
    # tiled themselves (8 rows would leave the consumer's work division a cost-model tie).
    "pointwise_outside_consumers": (
        lambda a, b: (lambda y: y[:256] * 2 + y[256:512])(a + b),
        _inputs((4096, 1024), (4096, 1024)),
        4 * MB,
        [(1, ["Pointwise"], [2])],
        _COPY_OUT,
        2,
        2,
    ),
    "pointwise_outside_consumer_and_graph_output": (
        lambda a, b: (lambda y: (y, y[:256] * 2))(a + b),
        _inputs((4096, 1024), (4096, 1024)),
        4 * MB,
        [(1, ["Pointwise"], [2])],
        _COPY_OUT,
        2,
        2,
    ),
    # A bias vector read by a joined consumer tiled on N: its buffer is rank 1 where
    # the op iterates two dims, and its index still holds the tiled dim's symbol.
    "matmul_bias_tiled_n": (
        lambda a, b, c: (a @ b) + c,
        _inputs((256, 64), (64, 8192), (8192,)),
        2 * MB,
        [(2, ["Reduction", "Pointwise"], [2])],
        _COPY_OUT,
        2,
        3,
    ),
    # A bmm that consumes a tiled bmm over a shared batch dim joins its producer's
    # group (two Reductions, one loop). A pointwise producer or a 2D matmul feeding a
    # matmul never does: the grouper refuses it ("cannot join them").
    "bmm_consuming_a_tiled_bmm": (
        lambda a, b, c: torch.bmm(torch.bmm(a, b), c),
        _inputs((8, 2048, 64), (8, 64, 256), (8, 256, 64)),
        4 * MB,
        [(2, ["Reduction", "Reduction"], [2])],
        _COPY_OUT,
        2,
        3,
    ),
    "k_tiled_sum": (
        lambda x: x.sum(dim=1),
        lambda: [_shaped((128, 8192), 1)],
        512 * 1024,
        [(1, ["Reduction"], [4])],
        _K_TILED,
        4,
        2,
    ),
    "k_tiled_amax": (
        lambda x: x.amax(dim=1),
        lambda: [-(_shaped((64, 8192), 2).abs() + 0.5)],
        256 * 1024,
        [(1, ["Reduction"], [4])],
        _K_TILED,
        4,
        2,
    ),
    "k_tiled_amin": (
        lambda x: x.amin(dim=1),
        lambda: [_shaped((64, 8192), 3).abs() + 0.5],
        256 * 1024,
        [(1, ["Reduction"], [4])],
        _K_TILED,
        4,
        2,
    ),
    # The accumulator and its combine keep fp32 (an fp32 matmul is not supported by
    # the backend, so the fp32 accumulator is exercised through a sum).
    "k_tiled_sum_in_fp32": (
        lambda x: x.sum(dim=1),
        lambda: [_shaped((64, 8192), 7).float()],
        512 * 1024,
        [(1, ["Reduction"], [4])],
        _K_TILED,
        4,
        2,
    ),
    # The matmul result is returned itself and also read by a later op.
    "k_tiled_matmul_whose_accumulator_is_a_graph_output": (
        _matmul_and_its_double,
        _inputs((2048, 2048), (2048, 256)),
        1 * MB,
        [(1, ["Reduction"], [8])],
        _K_TILED,
        8,
        2,
    ),
}


@pytest.mark.parametrize("case", sorted(LOWERED))
def test_the_lowering_builds_what_the_case_expects(case):
    fn, make_inputs, limit, groups, added, trips, members = LOWERED[case]
    args = make_inputs()
    compiled, nodes, violations = _compile_seeing_nodes(fn, args, limit)
    assert compiled.groups == groups, compiled.groups
    assert _matches(compiled.added, added), (compiled.added, added)
    assert [(n[2], len(n[3])) for n in nodes] == [(trips, members)], nodes
    assert violations == [], violations
    reference = fn(*[a.float() for a in args])
    assert _relative_error(compiled.outputs, reference) < 2e-2


@expects_raise
def test_a_prod_over_a_tiled_dim_is_rejected():
    """``prod`` over a span-overflowing dim is refused: the auto-tiler cannot join the
    reduction with its producer.

    TODO: when it can, this becomes a positive case of ``LOWERED`` with an identity
    fill of 1 (the input is powers of two, so every product is exact in fp16)."""
    x = _signed_powers_of_two((64, 8192), 4, 0.0015)
    with pytest.raises(Exception, match="Cannot auto-tile"):
        compile_span_overflow(lambda x: x.prod(dim=1), [x], 256 * 1024)


# --- the permuted buffer tiled eight ways at one level ---------------------------

_SHAPE = (8, 8, 64, 512)


@pytest.mark.parametrize("which, dim", [("plain", 0), ("permuted", 0), ("permuted", 1)])
def test_forced_plans_are_right(which, dim):
    """Dim 0 of the permuted buffer is squeezed to extent 1, and its step is the
    buffer's own stride (#6093). The plan reads it from the index before the divide."""
    make = {"plain": _plain, "permuted": _permuted}[which]
    wrong, loops, counts = _forced_one_level(_SHAPE, [(dim, 8)], make)
    assert (loops, counts) == (1, [8]), (loops, counts)
    assert wrong == 0.0, wrong


def _forced_view(view, levels):
    """``abs(view(x)) + 1`` with the span-overflow plan forced to ``levels``; the
    fraction of elements more than 0.05 from fp32."""
    from unittest import mock

    from test_span_overflow_hint_analysis import _forced_span_plan
    from torch._inductor.utils import run_and_get_code

    from torch_spyre._inductor import config
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
        config.patch({"sencores": 4, "ignore_span_overflow_hints": False}),
        torch._inductor.config.patch(force_disable_caches=True),
    ):
        out, _ = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), x.to(DEVICE_NAME)
        )
    return ((out.cpu().float() - fn(x.float())).abs() > 0.05).float().mean().item()


@pytest.mark.parametrize("dim, count", [(0, 8), (1, 8), (2, 8), (0, 2)])
def test_a_read_through_a_permuted_view_is_right(dim, count):
    """The op iterates (8, 64, 512) over an input stored (64, 8, 512): its dim 0 is the
    buffer's dim 1, so the buffer's stride for 'dim 0' is not the op's (4096, not 512).
    The step comes from the access's own index before the divide, so the view's order
    does not matter, including when the tiled dim is squeezed to extent 1."""
    assert _forced_view(lambda t: t.permute(1, 0, 2), [(dim, count)]) == 0.0


def test_a_read_through_a_plain_view_is_converted():
    assert _forced_view(lambda t: t, [(0, 64)]) == 0.0


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#6243: span-overflow tiling returns wrong values for a transposed "
    "consumer of a tiled pointwise producer (about 60% of elements); the grouping "
    "puts the consumer in the producer's loop",
)
def test_a_transposed_consumer_of_a_tiled_producer_is_right():
    fn = lambda a, b: (a + b).t() * 2  # noqa: E731
    args = [_shaped((4096, 1024), seed) for seed in range(2)]
    compiled = compile_span_overflow(fn, args, 4 * MB)
    reference = fn(*[a.float() for a in args])
    wrong = ((compiled.outputs[0] - reference).abs() > 0.05).float().mean().item()
    assert wrong == 0.0, wrong
