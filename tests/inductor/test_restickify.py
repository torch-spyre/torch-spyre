# Copyright 2025 The Torch-Spyre Authors.
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

# Tests for restickify insertion in pointwise operations.
#
# Restickify is triggered when a transposed (non-contiguous) tensor is used
# in a pointwise op alongside a contiguous tensor, and the layouts are
# stick-incompatible. The compiler inserts a restickify kernel to convert
# the layout before the pointwise op proceeds.
#
# Shapes use multiples of 64 (stick size = 64 fp16 elements) to ensure
# stick-aligned inputs that exercise the restickify path rather than fallback.

import itertools
import math

import pytest
from unittest.mock import patch

import torch
from torch._inductor.virtualized import V
from torch.spyre import SpyreTensorLayout

import torch_spyre._inductor.optimize_restickify as _optimize_restickify
from torch._inductor.exc import InductorError
from torch_spyre._inductor import config
from utils_inductor import _compile_and_run, compare_with_cpu

DEVICE = torch.device("spyre")
S = 128  # must be a multiple of 64
T = 64  # side length for 4D tests (all dims equal)


@pytest.fixture(autouse=True)
def _seed_rng():
    """Pin the RNG before every test so the ``torch.randn`` inputs do not depend
    on prior tests' draws.  A few tolerance-based tests reduce (e.g. p.sum())
    over their inputs, and allclose's rtol*|ref| slack collapses when that
    reduction lands near zero, so an unlucky order-dependent draw could flake.
    A fixed seed makes the draws reproducible regardless of run order; the
    exact-ramp (_arange) tests draw no RNG and are unaffected."""
    torch.manual_seed(42)


# -------- Helpers ---------- #
def _compute_cost(restickify_plan):
    assert restickify_plan is not None, "restickify_plan should not be None"
    return sum(
        math.prod(int(s) for s in entry["target_layout"].size)
        for entries in restickify_plan.values()
        for entry in entries
    )


def _compile_and_run_plan_capture(fn, *args):
    import torch_spyre._inductor.passes as _passes

    captured = {}
    finalize_layouts = _passes.finalize_layouts

    def capturing_finalize_layouts(graph):
        finalize_layouts(graph)
        captured["plan"] = dict(V.graph.restickify_plan)

    with patch.object(_passes, "finalize_layouts", capturing_finalize_layouts):
        spyre_result = _compile_and_run(fn, args, DEVICE)

    return spyre_result, captured.get("plan", {})


def _strict_size1_input_arm(fn, x, expected_arm):
    """Run ``fn`` (a size-1 new-stick / output-elided restickify), assert the
    result matches CPU bit-exactly, AND assert _pad_restickify_input took the
    expected buffer-materialisation arm.

    ``expected_arm`` is ``"producer"`` (the input is a compute op whose output
    buffer we own and tag in place) or ``"clone"`` (the input is a graph input we
    cannot re-allocate, so an identity clone is materialised and tagged).  The
    arm assertion guards against the read being optimised away: if a future
    Inductor folded a graph-input transpose to a no-op (so no restickify read of
    the input survived), the clone arm would not fire and this test would fail
    loudly rather than silently validate nothing.

    ``_pad_restickify_input`` materialises the input inline, branching on whether
    the input buffer is a ``ComputedBuffer`` (producer) or a graph input (clone);
    the wrapper below classifies each call the same way to record which arm ran.
    """
    from torch._inductor.ir import ComputedBuffer

    import torch_spyre._inductor.padding as _padding

    arms = []
    orig = _padding._pad_restickify_input

    def capturing(op, graph):
        _in_dep, in_buf, _in_layout = _padding._restickify_input(op, graph)
        arms.append("producer" if isinstance(in_buf, ComputedBuffer) else "clone")
        return orig(op, graph)

    with patch.object(_padding, "_pad_restickify_input", capturing):
        spyre = _compile_and_run(fn, (x,), DEVICE)
    cpu = fn(x)
    assert torch.equal(spyre.cpu(), cpu), (
        f"\nMISMATCH shape={tuple(x.shape)}\n cpu   =\n{cpu}\n spyre =\n{spyre.cpu()}\n"
    )
    assert expected_arm in arms, (
        f"expected _pad_restickify_input {expected_arm!r} arm to fire, saw "
        f"{arms}; the restickify input read may have been optimised away"
    )


def _compare(
    fn,
    *args,
    device_args=None,
    check_strides=True,
    optimal_cost=None,
    skip_correctness=False,
):
    """Run fn on Spyre, assert correctness against CPU, and optionally assert the restickify
    plan has cost == optimal_cost.

    device_args: if provided, used for the Spyre run instead of args (allows pre-placed
    tensors with custom layouts). args are still used for the CPU reference.
    """
    run_args = device_args if device_args is not None else args
    if optimal_cost is None:
        spyre_result = _compile_and_run(fn, run_args, DEVICE)
    else:
        spyre_result, plan = _compile_and_run_plan_capture(fn, *run_args)
        actual_cost = _compute_cost(plan)
        assert actual_cost == optimal_cost, (
            f"restickify cost: expected {optimal_cost}, got {actual_cost}"
        )
    if not skip_correctness:
        compare_with_cpu(fn, *args, target=spyre_result, run_eager=False)
    if (
        check_strides and device_args is None
    ):  # skip when device_args differ from CPU args: strides intentionally won't match
        cpu_result = fn(*args)
        assert cpu_result.stride() == spyre_result.stride(), (
            f"Stride mismatch: CPU {cpu_result.stride()} vs Spyre {spyre_result.stride()}"
        )


def _make_tensors(n, *shape):
    """Make n scaled fp16 tensors of the given shape. Scale keeps values small enough for chained matmuls."""
    return [torch.randn(*shape, dtype=torch.float16) * 0.1 for _ in range(n)]


def _make_2d_tensors(s1, s2):
    # A, B: shape [s1, s2]; X, Y: shape [s2, s1]
    A = torch.randn((s1, s2), dtype=torch.float16)
    B = torch.randn((s1, s2), dtype=torch.float16)
    X = torch.randn((s2, s1), dtype=torch.float16)
    Y = torch.randn((s2, s1), dtype=torch.float16)
    return A, B, X, Y


def _arange(*shape, base=0, span=1000):
    """A distinct-value ramp that is EXACT in fp16.

    Build the ramp in int64 and take the modulo BEFORE casting: fp16 cannot
    represent integers above 2048 exactly (``torch.arange`` itself overflows to
    inf past ~65504), and odd integers in (1024, 2048] are not representable, so
    a direct fp16 arange — or any band reaching past 1023 — silently rounds and
    turns the oracle's exact-equality check into a lie.

    ``base + span`` must stay <= 1024 so every value lands on an exact fp16
    integer (ULP < 1); a misplaced stick then shows up as a wrong value, never a
    rounding artifact.  ``base`` lets a second argument occupy a disjoint band
    from the first, so a swapped element is caught even between two cat inputs.
    """
    assert base + span <= 1024, f"band [{base}, {base + span}) exceeds fp16-exact 1024"
    n = 1
    for s in shape:
        n *= s
    ramp = (torch.arange(n, dtype=torch.int64) % span) + base
    return ramp.to(torch.float16).reshape(shape)


def _strict(fn, *args):
    spyre = _compile_and_run(fn, args, DEVICE)
    cpu = fn(*args)
    shapes = [tuple(a.shape) for a in args]
    assert torch.equal(spyre.cpu(), cpu), (
        f"\nMISMATCH shapes={shapes}\n cpu   =\n{cpu}\n spyre =\n{spyre.cpu()}\n"
    )


# -------- Pointwise tests ----------

# 2-arg tests — run on a full set of size pairs
SIZES_2D_FULL = [
    (256, 128),
    (128, 256),
    (128, 128),
    (64, 128),
    (128, 64),
]


@pytest.fixture(params=SIZES_2D_FULL, ids=lambda p: f"{p[0]}x{p[1]}")
def tensors_2arg(request):
    s1, s2 = request.param
    return _make_2d_tensors(s1, s2)


def test_2arg_at_plus_x(tensors_2arg):
    A, _, X, _ = tensors_2arg
    _compare(lambda a, x: a.t() + x, A, X, optimal_cost=A.numel())


def test_2arg_x_plus_at(tensors_2arg):
    A, _, X, _ = tensors_2arg
    _compare(lambda a, x: x + a.t(), A, X, optimal_cost=A.numel())


def test_2arg_xt_plus_a(tensors_2arg):
    A, _, X, _ = tensors_2arg
    _compare(lambda a, x: x.t() + a, A, X, optimal_cost=X.numel())


def test_2arg_a_plus_xt(tensors_2arg):
    A, _, X, _ = tensors_2arg
    _compare(lambda a, x: a + x.t(), A, X, optimal_cost=X.numel())


# 3-arg and 4-arg tests — run on a smaller set of size pairs
SIZES_2D_SMALL = [
    (256, 128),
    (128, 128),
]


@pytest.fixture(params=SIZES_2D_SMALL, ids=lambda p: f"{p[0]}x{p[1]}")
def tensors_multiarg(request):
    s1, s2 = request.param
    return _make_2d_tensors(s1, s2)


def test_3arg_at_bt_x(tensors_multiarg):
    A, B, X, _ = tensors_multiarg
    _compare(lambda a, b, x: a.t() + b.t() + x, A, B, X, optimal_cost=X.numel())


def test_3arg_at_x_bt(tensors_multiarg):
    A, B, X, _ = tensors_multiarg
    _compare(lambda a, b, x: a.t() + x + b.t(), A, B, X, optimal_cost=X.numel())


def test_3arg_x_at_bt(tensors_multiarg):
    A, B, X, _ = tensors_multiarg
    _compare(lambda a, b, x: x + a.t() + b.t(), A, B, X, optimal_cost=X.numel())


def test_3arg_at_x_y(tensors_multiarg):
    A, _, X, Y = tensors_multiarg
    _compare(lambda a, x, y: a.t() + x + y, A, X, Y, optimal_cost=A.numel())


def test_4arg_at_bt_x_y(tensors_multiarg):
    A, B, X, Y = tensors_multiarg
    _compare(
        lambda a, b, x, y: a.t() + b.t() + x + y, A, B, X, Y, optimal_cost=A.numel()
    )


def test_4arg_at_x_bt_y(tensors_multiarg):
    A, B, X, Y = tensors_multiarg
    _compare(
        lambda a, b, x, y: a.t() + x + b.t() + y, A, B, X, Y, optimal_cost=2 * A.numel()
    )


def test_4arg_x_at_y_bt(tensors_multiarg):
    A, B, X, Y = tensors_multiarg
    _compare(
        lambda a, b, x, y: x + a.t() + y + b.t(), A, B, X, Y, optimal_cost=2 * A.numel()
    )


def test_4arg_at_x_y_bt(tensors_multiarg):
    A, B, X, Y = tensors_multiarg
    _compare(
        lambda a, b, x, y: a.t() + x + y + b.t(), A, B, X, Y, optimal_cost=2 * A.numel()
    )


def test_4arg_at_x_y_z(tensors_multiarg):
    A, _, X, Y = tensors_multiarg
    Z = torch.randn_like(X)
    _compare(lambda a, x, y, z: a.t() + x + y + z, A, X, Y, Z, optimal_cost=A.numel())


def test_4arg_x_at_y_z(tensors_multiarg):
    A, _, X, Y = tensors_multiarg
    Z = torch.randn_like(X)
    _compare(lambda a, x, y, z: x + a.t() + y + z, A, X, Y, Z, optimal_cost=A.numel())


def test_4arg_x_y_at_z(tensors_multiarg):
    A, _, X, Y = tensors_multiarg
    Z = torch.randn_like(X)
    _compare(lambda a, x, y, z: x + y + a.t() + z, A, X, Y, Z, optimal_cost=A.numel())


def test_4arg_x_y_z_at(tensors_multiarg):
    A, _, X, Y = tensors_multiarg
    Z = torch.randn_like(X)
    _compare(lambda a, x, y, z: x + y + z + a.t(), A, X, Y, Z, optimal_cost=A.numel())


# 3D tests
SIZES_3D = [(2, 256, 128), (4, 128, 64)]


@pytest.fixture(params=SIZES_3D, ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}")
def tensors_3d(request):
    s0, s1, s2 = request.param
    a = torch.randn((s0, s1, s2), dtype=torch.float16)
    x = torch.randn((s0, s2, s1), dtype=torch.float16)
    return a, x


def test_3d_transpose12_plus_x(tensors_3d):
    a, x = tensors_3d
    _compare(lambda a, x: a.transpose(1, 2) + x, a, x)


def test_3d_x_plus_transpose12(tensors_3d):
    a, x = tensors_3d
    _compare(lambda a, x: x + a.transpose(1, 2), a, x)


# 4D tests:
SIZES_4D = [(2, 256, 3, 128), (2, 128, 4, 64)]


@pytest.fixture(params=SIZES_4D, ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}x{p[3]}")
def tensors_4d(request):
    s0, s1, s2, s3 = request.param
    a = torch.randn((s0, s1, s2, s3), dtype=torch.float16)
    x = torch.randn((s0, s3, s2, s1), dtype=torch.float16)
    return a, x


def test_4d_transpose13_plus_x(tensors_4d):
    a, x = tensors_4d
    _compare(lambda a, x: a.transpose(1, 3) + x, a, x)


def test_4d_x_plus_transpose13(tensors_4d):
    a, x = tensors_4d
    _compare(lambda a, x: x + a.transpose(1, 3), a, x)


# View + unsqueeze tests


def test_view_unsqueeze_add():
    d0, d1, d2, d3, d4 = 2, 3, 4, 2, 64
    a = torch.randn((1, d0, d1 * d3 * d4), dtype=torch.float16) * 0.1
    b = torch.randn((1, d0, d1 * d3 * d4), dtype=torch.float16) * 0.1
    c = torch.randn((1, d0, d2, d3, d4), dtype=torch.float16) * 0.1

    def func(a, b, c):
        x = a + b
        z = x.view(1, d0, d1, d3, d4)
        return z.unsqueeze(2) + c.unsqueeze(3)

    _compare(func, a, b, c)


# Expand tests
SIZES_EXPAND = [(128, 256)]


@pytest.fixture(params=SIZES_EXPAND, ids=lambda p: f"{p[0]}x{p[1]}")
def tensors_expand(request):
    s0, s1 = request.param
    x = torch.randn((s0, s1, s1), dtype=torch.float16)
    y = torch.randn((s1, s0), dtype=torch.float16)
    return x, y


def test_expand_x_plus_yt_expand(tensors_expand):
    x, y = tensors_expand
    _compare(lambda x, y: x + y.transpose(0, 1).unsqueeze(1).expand(x.shape), x, y)


def test_expand_yt_expand_plus_x(tensors_expand):
    x, y = tensors_expand
    _compare(
        lambda x, y: y.transpose(0, 1).unsqueeze(1).expand(x.shape) + x,
        x,
        y,
        check_strides=False,  # Stride differes from CPU even before restickify, skipping stride check
    )


# Expand + transpose tests: b.unsqueeze(0 or 1).expand(s,s) forces layout
# choice because the expand side cannot always be restickified — the optimizer
# must choose the a.t() side's stick instead.


def test_expand_unsqueeze0_expand_plus_at():
    s = 128
    a = torch.randn((s, s), dtype=torch.float16) * 0.1
    b = torch.randn((s,), dtype=torch.float16) * 0.1
    _compare(
        lambda a, b: b.unsqueeze(0).expand(s, s) + a.t(), a, b, check_strides=False
    )


def test_expand_at_plus_unsqueeze0_expand():
    s = 128
    a = torch.randn((s, s), dtype=torch.float16) * 0.1
    b = torch.randn((s,), dtype=torch.float16) * 0.1
    _compare(lambda a, b: a.t() + b.unsqueeze(0).expand(s, s), a, b)


def test_expand_unsqueeze1_expand_plus_at():
    s = 128
    a = torch.randn((s, s), dtype=torch.float16) * 0.1
    b = torch.randn((s,), dtype=torch.float16) * 0.1
    _compare(
        lambda a, b: b.unsqueeze(1).expand(s, s) + a.t(), a, b, check_strides=False
    )


def test_expand_at_plus_unsqueeze1_expand():
    s = 128
    a = torch.randn((s, s), dtype=torch.float16) * 0.1
    b = torch.randn((s,), dtype=torch.float16) * 0.1
    _compare(lambda a, b: a.t() + b.unsqueeze(1).expand(s, s), a, b)


# cat after two-stick add: the add produces two candidate sticks; the cat
# forces a mutation op downstream and requires the chosen stick to be
# compatible with the cat output layout.


def test_cat_after_at_plus_b():
    s = 128
    a = torch.randn((s, s), dtype=torch.float16) * 0.1
    b = torch.randn((s, s), dtype=torch.float16) * 0.1
    c = torch.randn((s, s), dtype=torch.float16) * 0.1
    _compare(lambda a, b, c: torch.cat([a.t() + b, c]), a, b, c, check_strides=False)


# 2-arg tests with size-1
SIZES_4D_SIZE1 = [(128, 256)]


@pytest.fixture(params=SIZES_4D_SIZE1, ids=lambda p: f"1x{p[0]}x1x{p[1]}")
def tensors_size1(request):
    s1, s2 = request.param
    X = torch.randn((1, s2, 1, s1), dtype=torch.float16)
    Y = torch.randn((1, s1, 1, s2), dtype=torch.float16)
    return X, Y


def test_2arg_size1_x_plus_yt13(tensors_size1):
    X, Y = tensors_size1
    _compare(lambda x, y: x + y.transpose(1, 3), X, Y)


def test_2arg_size1_yt13_plus_x(tensors_size1):
    X, Y = tensors_size1
    _compare(lambda x, y: y.transpose(1, 3) + x, X, Y)


# ------- Matmul Tests ---------

MATMUL_SIZES = [(128, 256), (64, 128)]


@pytest.fixture(params=MATMUL_SIZES, ids=[f"{a}x{b}" for a, b in MATMUL_SIZES])
def matmul_tensors_ab(request):
    a, b = request.param
    x = torch.randn((a, b), dtype=torch.float16) * 0.1
    y = torch.randn((a, b), dtype=torch.float16) * 0.1
    return x, y


@pytest.fixture(params=MATMUL_SIZES, ids=[f"{a}x{b}" for a, b in MATMUL_SIZES])
def matmul_tensors_ab_ba(request):
    a, b = request.param
    x = torch.randn((a, b), dtype=torch.float16) * 0.1
    y = torch.randn((b, a), dtype=torch.float16) * 0.1
    return x, y


def test_matmul_x_y(matmul_tensors_ab_ba):
    x, y = matmul_tensors_ab_ba
    _compare(lambda x, y: torch.matmul(x, y), x, y, optimal_cost=0)


def test_matmul_xt_y(matmul_tensors_ab):
    x, y = matmul_tensors_ab
    _compare(lambda x, y: torch.matmul(x.t(), y), x, y, optimal_cost=x.numel())


def test_matmul_x_yt(matmul_tensors_ab):
    x, y = matmul_tensors_ab
    _compare(lambda x, y: torch.matmul(x, y.t()), x, y, optimal_cost=y.numel())


def test_matmul_xt_yt(matmul_tensors_ab_ba):
    x, y = matmul_tensors_ab_ba
    _compare(
        lambda x, y: torch.matmul(x.t(), y.t()),
        x,
        y,
        optimal_cost=x.numel() + y.numel(),
    )


# ------- Batched Matmul Tests ---------

BMM_SIZES = [(3, 128, 64)]


@pytest.fixture(params=BMM_SIZES, ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}")
def bmm_tensors_ab(request):
    batch, a, b = request.param
    x = torch.randn((batch, a, b), dtype=torch.float16) * 0.1
    y = torch.randn((batch, a, b), dtype=torch.float16) * 0.1
    return x, y


@pytest.fixture(params=BMM_SIZES, ids=lambda p: f"{p[0]}x{p[1]}x{p[2]}")
def bmm_tensors_ab_ba(request):
    batch, a, b = request.param
    x = torch.randn((batch, a, b), dtype=torch.float16) * 0.1
    y = torch.randn((batch, b, a), dtype=torch.float16) * 0.1
    return x, y


def test_bmm_xt_y(bmm_tensors_ab):
    x, y = bmm_tensors_ab
    _compare(lambda x, y: torch.matmul(x.transpose(1, 2), y), x, y)


def test_bmm_x_yt(bmm_tensors_ab):
    x, y = bmm_tensors_ab
    _compare(lambda x, y: torch.matmul(x, y.transpose(1, 2)), x, y)


def test_bmm_xt_yt(bmm_tensors_ab_ba):
    x, y = bmm_tensors_ab_ba
    _compare(lambda x, y: torch.matmul(x.transpose(1, 2), y.transpose(1, 2)), x, y)


@pytest.mark.xfail(reason="Matmul with size 1 dims not yet supported")
def test_matmul_unit_n_2d():
    """2D matmul where N=1 (output is a column vector)."""
    x = torch.rand(5, 64, dtype=torch.float16)
    y = torch.rand(64, 1, dtype=torch.float16)
    _compare(lambda x, y: x @ y, x, y)


@pytest.mark.xfail(reason="Matmul with size 1 dims not yet supported")
def test_matmul_unit_n_2d_reduce():
    """2D matmul where Y is sparse"""
    x = torch.rand(5, 64, dtype=torch.float16)
    y = torch.rand(64, 64, dtype=torch.float16)
    _compare(lambda x, y: x @ y.sum(dim=-1, keepdim=True), x, y)


@pytest.mark.xfail(reason="Matmul with size 1 dims not yet supported")
def test_bmm_unit_n():
    """Batched matmul with N=1 output dimension (decode-style query matmul).

    Both x and y are (32, 1, 128); y is transposed before the matmul giving
    out: (32, 1, 1).  When N=1 the N symbol is constant-folded away, producing
    a sparse y layout.  Verifies the compiler handles it correctly.
    """
    x = torch.randn((32, 1, 128), dtype=torch.float16) * 0.1
    y = torch.randn((32, 1, 128), dtype=torch.float16) * 0.1
    _compare(lambda x, y: torch.matmul(x, y.transpose(1, 2)), x, y)


@pytest.mark.xfail(reason="Matmul with size 1 dims not yet supported")
def test_bmm_unit_n_self():
    """same tensor to matmul with and without transpose"""
    x = torch.randn((32, 1, 128), dtype=torch.float16) * 0.1
    _compare(lambda x: torch.matmul(x, x.transpose(1, 2)), x)


# ------- FallbackKernel + restickify regression test ---------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
def test_fallback_with_restickify():
    # FallbackKernel (torch.sin) produces a MultiOutput node. Verify the optimizer
    # handles it via AnyInNode and still makes a correct restickify decision downstream.
    x, y = _make_tensors(2, S, S)
    _compare(lambda x, y: torch.sin(x) + y.t(), x, y, optimal_cost=S * S)


# ------- Mutation + restickify regression test ---------


def test_bmm_with_inplace_mutation():
    # Regression test: copy_() creates a mutation_renames chain in the Inductor
    # scheduler. Combined with a bmm whose weight needs restickifying, this
    # previously caused a topo-sort cycle when compute_dependencies() was called
    # a second time inside insert_restickify.
    B, M, K, N = 1, 8, 64, 64
    x = torch.randn((B, M, K), dtype=torch.float16)
    weight = torch.randn((N, K), dtype=torch.float16)
    cache = torch.zeros((B, M, K), dtype=torch.float16)

    def func(x, weight, cache):
        cache.copy_(x)
        return torch.bmm(cache, weight.t().unsqueeze(0).expand(B, -1, -1))

    _compare(func, x, weight, cache)


def test_mutation_target_multi_candidate_layout():
    """Issue #3845: a mutation op and its target can diverge on device layout in
    the beam search.  ``c = a + b`` is an ordinary ComputedBuffer with two
    genuine candidate STLs (row-stick vs. column-stick, since 128x256 is
    non-square); ``copy_forced(d, c)`` then mutates it.  Unlike the
    SpyreEmptyFallback accumulator case above (test_bmm_with_inplace_mutation,
    where the target starts as a fresh torch.zeros() buffer and already has
    working multi-candidate coupling), the mutation op here has no genuine
    MemoryDep read-edge to its target at all -- propagate_layouts synthesizes
    a coupling edge from the target's own write MemoryDep so the beam search
    can price the constraint that this op's output STL must match whatever
    the target eventually commits to.  This reproducer is independent of
    coarse_tiling and spyre_hint, isolating the root propagate_layouts gap.
    """
    M, N = 128, 256
    a = torch.randn((M, N), dtype=torch.float16) * 0.01
    b = torch.randn((M, N), dtype=torch.float16) * 0.01
    d = torch.randn((M, N), dtype=torch.float16) * 0.01

    def fn(a, b, d):
        c = a + b
        c = torch.ops.spyre.copy_forced(d, c)
        return c

    _compare(fn, a, b, d)


def test_mutation_target_restick_on_src_not_target():
    # d is (N, M) so d.t() is (M, N) col-major — a different layout than c = a+b
    # (row-major).  The restickify must land on d.t() (src), not on c (the mutation
    # target).  validate_no_restickify_on_mutation_targets is wired into passes.py by
    # this PR and fires during compilation if the target is incorrectly restickified.
    M, N = 128, 256
    a = torch.randn((M, N), dtype=torch.float16) * 0.01
    b = torch.randn((M, N), dtype=torch.float16) * 0.01
    d = torch.randn((N, M), dtype=torch.float16) * 0.01

    def fn(a, b, d):
        c = a + b
        c = torch.ops.spyre.copy_forced(d.t(), c)
        return c

    _compare(fn, a, b, d, optimal_cost=M * N)


def test_two_mutations_same_target_copy_f():
    # Two copy_forced calls writing the same SpyreEmptyFallback with conflicting
    # input layouts.  The optimizer must assign both mutation ops the same output
    # STL so the backend sees a consistent layout for acc.
    M, N = 128, 256
    a = torch.randn((N, M), dtype=torch.float16) * 0.01  # a.t() is MxN col-stick
    b = torch.randn((M, N), dtype=torch.float16) * 0.01  # MxN row-stick

    def fn(a, b):
        acc = torch.ops.spyre.empty([M, N], device=a.device, dtype=a.dtype)
        acc = torch.ops.spyre.copy_forced(a.t() + a.t(), acc)
        acc = torch.ops.spyre.copy_forced(b + b, acc)
        return acc

    result = _compile_and_run(fn, (a.to(DEVICE), b.to(DEVICE)), DEVICE).cpu()
    expected = b + b
    assert torch.allclose(result, expected, atol=1e-2), (
        f"max_diff={(result - expected).abs().max().item():.4f}"
    )


def test_two_mutations_same_target_copy_():
    # Same as test_two_mutations_same_target_copy_f but using PyTorch copy_()
    # instead of spyre.copy_forced.  The mutation target is a graph-input tensor,
    # so both ops must commit the same STL for acc.
    M, N = 128, 256
    a = torch.randn((N, M), dtype=torch.float16) * 0.01  # a.t() is MxN col-stick
    b = torch.randn((M, N), dtype=torch.float16) * 0.01  # MxN row-stick
    acc = torch.zeros((M, N), dtype=torch.float16)

    def fn(a, b, acc):
        acc.copy_(a.t() + a.t())
        acc.copy_(b + b)
        return acc

    _compare(fn, a, b, acc)


def test_two_mutations_read_between_writes_minimal_copy_f():
    # Minimal write/read/write: write acc, read acc into snapshot, write acc again,
    # return snapshot.  Both mutation ops must commit the same STL for acc.
    M, N = 128, 256
    a = torch.randn((N, M), dtype=torch.float16) * 0.01
    b = torch.randn((M, N), dtype=torch.float16) * 0.01

    def fn(a, b):
        acc = torch.ops.spyre.empty([M, N], device=a.device, dtype=a.dtype)
        acc = torch.ops.spyre.copy_forced(a.t(), acc)  # write 1: col-stick
        snapshot = acc * b  # READ acc
        acc = torch.ops.spyre.copy_forced(b, acc)  # write 2: row-stick
        return snapshot

    result = _compile_and_run(fn, (a.to(DEVICE), b.to(DEVICE)), DEVICE).cpu()
    expected = a.t().contiguous() * b
    assert torch.allclose(result, expected, atol=1e-2), (
        f"max_diff={(result - expected).abs().max().item():.4f}"
    )


def test_two_mutations_read_between_writes_minimal_copy_():
    # Same as test_two_mutations_read_between_writes_minimal_copy_f but using copy_().
    M, N = 128, 256
    a = torch.randn((N, M), dtype=torch.float16) * 0.01
    b = torch.randn((M, N), dtype=torch.float16) * 0.01
    acc = torch.zeros((M, N), dtype=torch.float16)

    def fn(a, b, acc):
        acc.copy_(a.t())  # write 1: col-stick
        snapshot = acc * b  # READ acc
        acc.copy_(b)  # write 2: row-stick
        return snapshot

    _compare(fn, a, b, acc)


@pytest.mark.skip(
    reason="crashes in _clone_layout (propagate_layouts) before the conflicting-layout "
    "assert fires: `return snapshot * acc` forces a flat 1D clone of acc (size=[32768]) "
    "between the two writes; device_coordinates on the col-stick acc via that flat index "
    "produces the unsupported stick expr floor((Mod(d0,16384))/256). Issue #4093"
)
def test_two_mutations_read_between_writes_v2_copy_f():
    # write / read / write returning snapshot * acc (both outputs live).
    # Skipped: `return snapshot * acc` forces a flat 1D clone of acc between the
    # two writes; device_coordinates on the col-stick acc via that flat index
    # produces an unsupported stick expression in _clone_layout.
    M, N = 128, 256
    a = torch.randn((N, M), dtype=torch.float16) * 0.01  # a.t() is MxN col-stick
    b = torch.randn((M, N), dtype=torch.float16) * 0.01  # MxN row-stick

    def fn(a, b):
        acc = torch.ops.spyre.empty([M, N], device=a.device, dtype=a.dtype)
        acc = torch.ops.spyre.copy_forced(a.t() + a.t(), acc)  # write 1: col-stick
        snapshot = acc * b  # READ acc
        acc = torch.ops.spyre.copy_forced(b + b, acc)  # write 2: row-stick
        return snapshot * acc

    result = _compile_and_run(fn, (a.to(DEVICE), b.to(DEVICE)), DEVICE).cpu()
    a_t = a.t().contiguous()
    expected = (a_t + a_t) * b * (b + b)
    assert torch.allclose(result, expected, atol=1e-2), (
        f"max_diff={(result - expected).abs().max().item():.4f}"
    )


def test_two_mutations_read_between_writes_v2_copy_():
    # Same as test_two_mutations_read_between_writes_v2_copy_f but using copy_().
    M, N = 128, 256
    a = torch.randn((N, M), dtype=torch.float16) * 0.01
    b = torch.randn((M, N), dtype=torch.float16) * 0.01
    acc = torch.zeros((M, N), dtype=torch.float16)

    def fn(a, b, acc):
        acc.copy_(a.t() + a.t())  # write 1: col-stick
        snapshot = acc * b  # READ acc
        acc.copy_(b + b)  # write 2: row-stick
        return snapshot * acc

    _compare(fn, a, b, acc)


# Optimizer correctness + optimality tests: verify both output values and
# minimum-cost restickify plan across a range of graph patterns.


def test_opt_parens_one_conflict():
    """((a + b) + (c.t() + d)) + (e + f) — conflict only in inner group."""
    a, b, c, d, e, f = _make_tensors(6, S, S)
    _compare(
        lambda a, b, c, d, e, f: ((a + b) + (c.t() + d)) + (e + f),
        a,
        b,
        c,
        d,
        e,
        f,
        optimal_cost=S * S,
    )


def test_opt_adds_then_matmul_x():
    """(a + b.t() + c.t() + d.t()) @ e — upstream optimal + forced matmul x cost."""
    a, b, c, d, e = _make_tensors(5, S, S)
    _compare(
        lambda a, b, c, d, e: (a + b.t() + c.t() + d.t()) @ e,
        a,
        b,
        c,
        d,
        e,
        optimal_cost=2 * S * S,
    )


def test_opt_adds_then_matmul_y():
    """a @ (b + c.t()) — beam picks upstream stick to avoid extra matmul cost."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: a @ (b + c.t()), a, b, c, optimal_cost=S * S)


def test_opt_adds_then_matmul_y_long_chain():
    """a @ (b + c.t() + d.t() + e.t()) — majority transposed going into y."""
    a, b, c, d, e = _make_tensors(5, S, S)
    _compare(
        lambda a, b, c, d, e: a @ (b + c.t() + d.t() + e.t()),
        a,
        b,
        c,
        d,
        e,
        optimal_cost=2 * S * S,
    )


def test_opt_matmul_x_and_y_conflict():
    """a.t() @ (b + c.t()) — x wrong stick + y upstream conflict."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: a.t() @ (b + c.t()), a, b, c, optimal_cost=2 * S * S)


def test_opt_matmul_then_adds():
    """(a @ b) + c.t() — matmul output stick vs transposed input."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: (a @ b) + c.t(), a, b, c, optimal_cost=S * S)


def test_opt_matmul_then_long_adds():
    """(a @ b) + c.t() + d.t() — keep matmul stick, restickify one input."""
    a, b, c, d = _make_tensors(4, S, S)
    _compare(lambda a, b, c, d: (a @ b) + c.t() + d.t(), a, b, c, d, optimal_cost=S * S)


def test_opt_chained_matmuls():
    """(a @ b) @ c — no restickify needed."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: (a @ b) @ c, a, b, c, optimal_cost=0)


def test_opt_two_independent_conflicts():
    """(a+b.t()) + (e.t()+f.t()+g) — two separate conflicts."""
    a, b, e, f, g = _make_tensors(5, S, S)
    _compare(
        lambda a, b, e, f, g: (a + b.t()) + (e.t() + f.t() + g),
        a,
        b,
        e,
        f,
        g,
        optimal_cost=2 * S * S,
    )


def test_opt_fanout_intermediate():
    """buf = a + b.t(); (buf + c) + (buf + d.t()) — buf consumed twice."""
    a, b, c, d = _make_tensors(4, S, S)

    def fn(a, b, c, d):
        buf = a + b.t()
        return buf + c + (buf + d.t())

    _compare(fn, a, b, c, d, optimal_cost=2 * S * S)


def test_opt_diamond():
    """buf = a + b.t(); buf + buf — same intermediate read twice."""
    a, b = _make_tensors(2, S, S)

    def fn(a, b):
        buf = a + b.t()
        return buf + buf

    _compare(fn, a, b, optimal_cost=S * S)


def test_opt_matmul_rect_x_wrong_stick():
    """(64x128).t() @ (64x192) — cost uses buffer size not reduction dim."""
    M, K, N = 64, 128, 192
    (a,) = _make_tensors(1, M, K)
    (b,) = _make_tensors(1, M, N)
    _compare(lambda a, b: a.t() @ b, a, b, optimal_cost=M * K)


def test_opt_sum_between_pointwise():
    """(a + b.t()).sum(1) + c — reduction between two pointwise stages."""
    a, b = _make_tensors(2, S, S)
    (c,) = _make_tensors(1, S)
    # Note: sum() below may fail correctness depending which stick flows in
    # because propagate_layouts does not yet properly detect incompatibility
    # of sparse/non-sparse sticks in a pointwise op.  Disabling correctness
    # check until that is resolved
    _compare(
        lambda a, b, c: (a + b.t()).sum(0) + c,
        a,
        b,
        c,
        optimal_cost=S * S,
        skip_correctness=True,
    )


def test_opt_chain_transposed_intermediate():
    """(a.t() + b).t() + c — intermediate consumed transposed."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: (a.t() + b).t() + c, a, b, c, optimal_cost=S * S)


def test_opt_beam_trim(monkeypatch):
    """Three ops each with 2 candidate layouts: beam grows to 8 before trimming.

    BEAM_WIDTH=2 forces trimming at every step; verifies correctness is preserved.
    """
    monkeypatch.setattr(_optimize_restickify, "BEAM_WIDTH", 2)
    a, b, c, d, e, f = _make_tensors(6, S, S)
    _compare(
        lambda a, b, c, d, e, f: (a.t() + b) + (c.t() + d) + (e.t() + f),
        a,
        b,
        c,
        d,
        e,
        f,
    )


def test_opt_4d_one_conflict():
    """a.transpose(0,3) + b + c + d — one input with stick on dim 0."""
    a, b, c, d = _make_tensors(4, T, T, T, T)
    _compare(
        lambda a, b, c, d: a.transpose(0, 3) + b + c + d,
        a,
        b,
        c,
        d,
        optimal_cost=T**4,
    )


def test_opt_4d_mixed_conflicts():
    """a.transpose(0,3) + b.transpose(1,3) + c.transpose(2,3) + d — three non-matching sticks."""
    a, b, c, d = _make_tensors(4, T, T, T, T)
    _compare(
        lambda a, b, c, d: (
            a.transpose(0, 3) + b.transpose(1, 3) + c.transpose(2, 3) + d
        ),
        a,
        b,
        c,
        d,
        optimal_cost=3 * T**4,
    )


def test_opt_4d_majority_wins():
    """a.transpose(0,3) + b.transpose(0,3) + c.transpose(0,3) + d — three stick on dim 0."""
    a, b, c, d = _make_tensors(4, T, T, T, T)
    _compare(
        lambda a, b, c, d: (
            a.transpose(0, 3) + b.transpose(0, 3) + c.transpose(0, 3) + d
        ),
        a,
        b,
        c,
        d,
        optimal_cost=T**4,
    )


def test_opt_4d_chain_transposed_intermediate():
    """(a.transpose(2,3) + b).transpose(2,3) + c — 4D version of transposed intermediate."""
    a, b, c = _make_tensors(3, T, T, T, T)
    _compare(
        lambda a, b, c: (a.transpose(2, 3) + b).transpose(2, 3) + c,
        a,
        b,
        c,
        optimal_cost=T**4,
    )


def test_opt_two_matmuls_wrong_inputs():
    """(a.t() @ b) + (c @ d.t()) — each matmul has one wrong-stick input."""
    a, b, c, d = _make_tensors(4, S, S)
    _compare(
        lambda a, b, c, d: (a.t() @ b) + (c @ d.t()),
        a,
        b,
        c,
        d,
        optimal_cost=2 * S * S,
    )


def test_opt_matmul_both_inputs_upstream_conflict():
    """(a + b.t()) @ (c + d.t()) — both inputs have upstream stick conflicts."""
    a, b, c, d = _make_tensors(4, S, S)
    _compare(
        lambda a, b, c, d: (a + b.t()) @ (c + d.t()),
        a,
        b,
        c,
        d,
        optimal_cost=2 * S * S,
    )


# ------- Intentional failure -------------------


def test_wrong_optimal_cost_fails():
    """This tests checks if the optimal cost is mismatching so proper
    assertion failure is detected"""

    a, b, c, d, e = _make_tensors(5, S, S)

    def func(a, b, c, d, e):
        return (a + b.t() + c.t() + d.t()) @ e

    correct_expected_cost = 2 * S * S

    with pytest.raises(
        AssertionError,
        match=f"restickify cost: expected 0, got {correct_expected_cost}",
    ):
        _compare(func, a, b, c, d, e, optimal_cost=0)


# ------- Constant tensor STL tests ---------


def test_constant_plus_xt():
    """ones_like(x) + x.t() — constant tensor should adopt x.t()'s stick (cost 0 once constant layout optimization is implemented)."""
    x = torch.randn((S, S), dtype=torch.float16)
    _compare(lambda x: torch.ones_like(x) + x.t(), x)


def test_constant_in_conflict_chain():
    """ones_like(x) + x.t() + y — constant adopts winning STL, doesn't add to conflict cost."""
    x, y = _make_tensors(2, S, S)
    _compare(lambda x, y: torch.ones_like(x) + x.t() + y, x, y, optimal_cost=S * S)


def test_constant_matmul_x():
    """ones_like(y) @ y — constant should get col-major STL that matmul x needs, cost 0."""
    y = _make_tensors(1, S, S)[0]
    _compare(lambda y: torch.ones_like(y) @ y, y, optimal_cost=0)


def test_two_constants_plus_xt():
    """ones_like(x) + zeros_like(x) + x.t() — two flexible constants (cost 0 once constant layout optimization is implemented)."""
    x = torch.randn((S, S), dtype=torch.float16)
    _compare(lambda x: torch.ones_like(x) + torch.zeros_like(x) + x.t(), x)


def test_full_plus_xt():
    """torch.full + x.t() — full tensor constant should adopt x.t()'s stick (cost 0 once constant layout optimization is implemented)."""
    x = torch.randn((S, S), dtype=torch.float16)
    _compare(
        lambda x: torch.full((S, S), 0.5, dtype=torch.float16, device=x.device) + x.t(),
        x,
    )


def test_fill_plus_xt():
    """empty_like + fill_ + x.t() — mutation-based constant should adopt x.t()'s stick (cost 0 once constant layout optimization is implemented)."""
    x = torch.randn((S, S), dtype=torch.float16)

    def fn(x):
        e = torch.empty_like(x)
        e.fill_(1.0)
        return e + x.t()

    _compare(fn, x)


def test_constant_two_consumers():
    """zeros_like fed to two consumers with conflicting layouts.

    With the current algorithm, the constant gets the generic (row-major) layout.
    out1 = z + a is free (both row-major); out2 = z + x.t() requires one restickify
    of z, costing A.numel().

    Once constant layout optimization is implemented, the constant will instead adopt
    x.t()'s stick for free and out1 will pay the restickify — same total cost, but
    the constant itself will never need a restickify.
    """
    A = torch.randn((S, S), dtype=torch.float16)
    X = torch.randn((S, S), dtype=torch.float16)

    def fn(a, x):
        z = torch.zeros_like(
            a
        )  # gets generic (row-major) layout until optimization lands
        out1 = z + a  # consumer 1: row-major — free
        out2 = z + x.t()  # consumer 2: col-major — restickify of z
        return out1 + out2

    _compare(fn, A, X, optimal_cost=A.numel())


def test_constant_three_consumers():
    """zeros_like fed to three consumers; first wins the layout, others may restickify.

    Correctness check only — cost depends on fusion decisions.
    """
    A = torch.randn((S, S), dtype=torch.float16)
    X = torch.randn((S, S), dtype=torch.float16)
    B = torch.randn((S, S), dtype=torch.float16)

    def fn(a, x, b):
        z = torch.zeros_like(a)  # flexible
        out1 = z + a  # consumer 1: row-major
        out2 = z + x.t()  # consumer 2: col-major — may restickify
        out3 = z + b  # consumer 3: row-major — same as consumer 1, free
        return out1 + out2 + out3

    _compare(fn, A, X, B)


def test_arange_plus_xt():
    """arange.view + x.t() — correctness check only.

    arange lowers to FallbackKernel which gets a fixed generic layout, so the
    downstream add may still need a restickify.  No optimal_cost asserted.
    """
    x = torch.randn((S, S), dtype=torch.float16)
    _compare(
        lambda x: (
            torch.arange(S * S, dtype=torch.float16, device=x.device).view(S, S) + x.t()
        ),
        x,
    )


# ------- Constant-fill inputs ---------


def test_amax_full_and_amax_live_maximum():
    """maximum(amax(full(-inf), dim=-1), amax(t, dim=-1)) — zero-stick output from
    constant-fill reduction must be a valid candidate for the pointwise output."""
    B, H, Lq, Lk = 1, 32, 128, 256
    t = torch.randn((B, H, Lq, Lk), dtype=torch.float16)

    def f(t):
        full = torch.full((B, H, Lq, Lk), float("-inf"), device=t.device, dtype=t.dtype)
        t_max = torch.amax(full, dim=-1)
        u_max = torch.amax(t, dim=-1)
        return torch.maximum(t_max, u_max)

    _compare(f, t, optimal_cost=0)


# ------- Unsupported stick configurations ---------


def test_sparse_dense_pointwise():
    """a.sum(-1) + b - reduction followed by pointwise without broadcasting."""
    a = torch.randn((S, S, S), dtype=torch.float16).to(DEVICE)
    b = torch.randn((S, S), dtype=torch.float16).to(DEVICE)

    with pytest.raises(
        InductorError, match="No mechanism to gather elements from multiple sticks"
    ):
        _compare(lambda a, b: a.amin(-1) + b, a, b)


# ------- Restickify padding: strided input raises Unsupported ---------


def test_pad_restickify_strided_input_raises():
    """A strided-read input to transpose+clone must raise, not silently corrupt
    output.

    ``x[:, :, ::2, :]`` reads dim -2 with step 2, so its coordinate is ``2*d``.
    Codegen carries only a contiguous tail, so a strided read is unpaddable and
    the compiler must fail loudly rather than return wrong data.  (A *narrowing*
    slice, by contrast, is now padded -- see the OFFSET_STICK_OK cases.)
    """
    x = torch.randn((2, 2, 128, 128), dtype=torch.float16)
    with pytest.raises(
        RuntimeError,
        match="strided input on host dim",
    ):
        _compile_and_run(
            lambda x: x[:, :, ::2, :].transpose(-2, -1).clone(), (x,), DEVICE
        )


def test_pad_restickify_strided_producer_raises():
    """A strided read fed by a *producer* must also raise, not miscompile.

    Same geometry as test_pad_restickify_strided_input_raises, but the ``+ 1``
    makes the strided tensor a produced (internal) buffer rather than a bare
    graph input, exercising the other input path.  The strided read is equally
    unpaddable, so this path must also fail loudly rather than return wrong
    data.
    """
    x = torch.randn((2, 2, 128, 128), dtype=torch.float16)
    with pytest.raises(
        RuntimeError,
        match="strided input on host dim",
    ):
        _compile_and_run(
            lambda x: (x + 1)[:, :, ::2, :].transpose(-2, -1).clone(), (x,), DEVICE
        )


# ------- Restickify padding: sliced-transpose stick expr classification -------


# Sliced transposes that ARE valid and must compile correctly, spanning the
# range of slice placements and extents on the becomes-stick dim.  Codegen reads
# a whole-stick window from the slice base -- [slice_start, slice_start +
# round_up_to_stick(extent)) -- and the input-padding pass grows the allocation
# to cover it, so a narrowing slice near the dim end (whose window over-reads the
# declared size) is padded, not rejected.  Only a *strided* read on the
# becomes-stick dim still raises (see test_pad_restickify_strided_input_raises).
OFFSET_STICK_OK = [
    # Slice the becomes-stick dim (dim0 -> last dim after transpose(0, 1)), whose
    # declared size 128 is stick-aligned; read window fits, gate skips.
    (lambda x: x[3:67].transpose(0, 1).clone(), (128, 128)),
    # Same, with an unaligned extent (63): read window still fits.
    (lambda x: x[3:66].transpose(0, 1).clone(), (128, 128)),
    # Stick-aligned slice start on the becomes-stick dim.
    (lambda x: x[:, :, 64:128, :].transpose(-2, -1).clone(), (2, 2, 128, 128)),
    # Slice on the becomes-stick dim, aligned (single-stick) extent.
    (lambda x: x[:, 64:128].transpose(0, 1).clone(), (128, 128)),
    # Slice on the becomes-stick dim, 1.5-stick extent.
    (lambda x: x[:, :96].transpose(0, 1).clone(), (128, 128)),
    # Narrowing slice near the dim end: read window [125, 125+64) over-reads the
    # 128-row declared size, so the pass grows the allocation to 192 (3 sticks).
    (lambda x: x[125:128].transpose(0, 1).clone(), (128, 128)),
    # Wider over-reading slice: extent 70 -> read window [30, 30+128) over-reads
    # the declared size; the allocation grows to 192.
    (lambda x: x[30:100].transpose(0, 1).clone(), (128, 128)),
]


@pytest.mark.parametrize(
    "fn,shape", OFFSET_STICK_OK, ids=lambda p: p if isinstance(p, tuple) else ""
)
def test_sliced_transpose_stick_expr_compiles(fn, shape):
    """A valid sliced transpose compiles correctly regardless of where the slice
    lands or its extent -- an over-reading narrowing slice on the becomes-stick
    dim is padded, and only a strided read is rejected (see
    test_pad_restickify_strided_input_raises)."""
    x = torch.randn(shape, dtype=torch.float16)
    result = _compile_and_run(fn, (x,), DEVICE)
    compare_with_cpu(fn, x, target=result, run_eager=False)


# Strict versions of the becomes-stick-dim slices above: a slice that lands on
# the dim the transpose turns into the stick is the case most likely to misplace
# a stick lane, and randn + tolerance can mask that.  A distinct-value ramp with
# torch.equal catches a single displaced lane exactly.
@pytest.mark.parametrize(
    "fn,shape", OFFSET_STICK_OK, ids=lambda p: p if isinstance(p, tuple) else ""
)
def test_sliced_transpose_stick_expr_strict(fn, shape):
    x = _arange(*shape)
    _strict(fn, x)


# ------- Restickify padding: large tensors (tolerance oracle) ---------
#
# transpose(-2,-1)+clone with an unaligned new stick dim, padded up to a stick
# boundary.  These tensors are too large for the fp16-exact ramp the strict
# tests below use (``_arange`` caps at 1024 elements per value band), so they
# compare ``randn`` data with a tolerance instead.  The exact geometry of every
# small padded shape is covered strictly below.
RESTICKIFY_PAD_LARGE_SIZES = [(1025, 1024), (2, 1025, 1024), (2, 2, 1025, 1024)]


@pytest.mark.parametrize(
    "shape", RESTICKIFY_PAD_LARGE_SIZES, ids=lambda p: "x".join(map(str, p))
)
def test_pad_large_transpose_clone(shape):
    x = torch.randn(shape, dtype=torch.float16)
    _compare(lambda x: x.transpose(-2, -1).clone(), x, check_strides=False)


# ------- Restickify input padding fused into a producer ---------
#
# When the restickify's input is produced by an internal op, the padding can be
# folded into that producer instead of inserting a separate copy; a restickify
# reading a bare graph input has no producer and falls back to the copy.  These
# tests exercise both outcomes (asserting on the debug log which path fired) and
# check that the result is correct either way.
#
# The tests put BOTH binary operands behind a computation so the transposed side
# (not a bare graph input) is the one restickified -- a bare graph-input operand
# would otherwise be the one chosen and hit the fallback.


def _run_capturing_padding_log(fn, *args):
    """Run fn on Spyre, returning (result, fused_fire_count, all_log_records)
    captured from insert_restickify_padding's debug log."""
    import logging

    import torch_spyre._inductor.padding as _padding

    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    prev_level = _padding.logger.level
    _padding.logger.addHandler(handler)
    _padding.logger.setLevel(logging.DEBUG)
    try:
        result = _compile_and_run(fn, args, DEVICE)
    finally:
        _padding.logger.removeHandler(handler)
        _padding.logger.setLevel(prev_level)

    # Every padded input logs "padded input"; a graph-input clone additionally
    # logs "inserted clone". A producer fusion is a padded input with no clone,
    # so the producer-arm count is padded inputs minus inserted clones.
    padded = sum("padded input" in m for m in records)
    cloned = sum("inserted clone" in m for m in records)
    fused = padded - cloned
    return result, fused, records


RESTICKIFY_FUSE_SIZES = [(67, 67), (2, 67, 67)]


@pytest.fixture(params=RESTICKIFY_FUSE_SIZES, ids=lambda p: "x".join(map(str, p)))
def fuse_tensors(request):
    shape = request.param
    x = torch.randn(shape, dtype=torch.float16)
    y = torch.randn(shape, dtype=torch.float16)
    return x, y


def test_pad_fused_into_producer(fuse_tensors):
    """(x*2).T + relu(y): the transposed side is an internal single-consumer
    pointwise producer, so the padding fuses into it (no copy).  Both operands
    are unaligned, so the same restickify needs both input and output padding."""
    x, y = fuse_tensors

    def fn(x, y):
        return (x * 2).transpose(-2, -1) + torch.relu(y)

    result, fused, _ = _run_capturing_padding_log(fn, x, y)
    assert fused >= 1, "expected producer-fusion to fire, but it did not"
    compare_with_cpu(fn, x, y, target=result, run_eager=False)


def test_pad_fused_into_matmul_producer():
    """A sliced matmul output (a@b)[:,c:]+z: the producer is a matmul
    (a reduction), not a pointwise op.  It should still fuse rather than fall
    back to a copy, and the result must be correct."""
    a = torch.randn((67, 64), dtype=torch.float16)
    b = torch.randn((64, 67), dtype=torch.float16)
    z = torch.randn((67, 64), dtype=torch.float16)

    def fn(a, b, z):
        return (a @ b)[:, 3:] + z

    result, fused, _ = _run_capturing_padding_log(fn, a, b, z)
    assert fused >= 1, "matmul (Reduction) producer should fuse, not fall back"
    compare_with_cpu(fn, a, b, z, target=result, run_eager=False)


def test_pad_graph_input_falls_back():
    """Restickifying a bare graph input has no producer to fuse into, so the
    padding must fall back to a copy -- and still be correct."""
    x = torch.randn((67, 67), dtype=torch.float16)
    y = torch.randn((67, 67), dtype=torch.float16)

    def fn(x, y):
        return x.transpose(-2, -1) + y

    result, fused, _ = _run_capturing_padding_log(fn, x, y)
    assert fused == 0, "graph-input restickify should not fuse into a producer"
    compare_with_cpu(fn, x, y, target=result, run_eager=False)


def test_pad_multi_consumer_producer_fuses_with_coreader():
    """A producer read by a restickify AND a non-restickify co-reader (here
    p.sum()) still fuses, and the co-reader's result stays correct -- growing
    the shared producer for the restickify does not disturb the other reader."""
    x = torch.randn((67, 128), dtype=torch.float16)
    z = torch.randn((128, 67), dtype=torch.float16)

    def fn(x, z):
        p = x * 2  # two consumers: the transpose (restickify) and the sum
        return p.transpose(-2, -1) + z + p.sum()

    result, fused, _ = _run_capturing_padding_log(fn, x, z)
    assert fused >= 1, "producer with a non-restickify co-reader should still fuse"
    compare_with_cpu(fn, x, z, target=result, run_eager=False)


def _shared_producer_two_restickify_case():
    """The (inputs, fn) for a producer ``p = x * 2`` whose sole consumers are two
    transposes taking different new stick dims -- so p fans out to two restickify
    nodes with different paddings.  Shared by the two tests below, which assert
    different things about it (fusion + correctness vs the two-node graph shape)."""
    x = torch.randn((67, 53, 128), dtype=torch.float16)
    za = torch.randn((128, 53, 67), dtype=torch.float16)
    zb = torch.randn((67, 128, 53), dtype=torch.float16)

    def fn(x, za, zb):
        p = x * 2  # sole consumers are the two transposes below (restickifies)
        return p.transpose(0, 2) + za, p.transpose(1, 2) + zb

    return (x, za, zb), fn


def test_pad_shared_all_restickify_consumers_fuse():
    """A producer read only by restickify ops fuses even with several consumers.
    Two transposes of the same producer take different new stick dims, so each
    needs a different padding; both must apply and both results be correct."""
    (x, za, zb), fn = _shared_producer_two_restickify_case()
    result, fused, _ = _run_capturing_padding_log(fn, x, za, zb)
    assert fused >= 1, "all-restickify shared producer should fuse"
    compare_with_cpu(fn, x, za, zb, target=result, run_eager=False)


def _is_restickify_op(op) -> bool:
    """True when ``op`` is a spyre.restickify ComputedBuffer, detected via its
    origin FX node's target."""
    from torch._inductor.ir import ComputedBuffer

    if not isinstance(op, ComputedBuffer):
        return False
    origins = op.origins
    if not origins:
        return False
    return next(iter(origins)).target is torch.ops.spyre.restickify.default


def _restickify_readers_by_source(fn, *args):
    """Run fn on Spyre and return {producer_name: [restickify buffer names]},
    a map of every source buffer to the restickify ops that read it, captured
    from graph.operations right after insert_restickify splices them in."""
    import torch_spyre._inductor.passes as _passes

    insert_restickify = _passes.insert_restickify
    by_source: dict[str, list[str]] = {}

    def capturing_insert_restickify(graph):
        insert_restickify(graph)
        for op in graph.operations:
            if not _is_restickify_op(op):
                continue
            for read in op.get_read_writes().reads:
                name = getattr(read, "name", None)
                if name is not None:
                    by_source.setdefault(name, []).append(op.get_name())

    with patch.object(_passes, "insert_restickify", capturing_insert_restickify):
        _compile_and_run(fn, args, DEVICE)
    return by_source


def test_shared_producer_gets_two_restickify_nodes():
    """insert_restickify keys its plan by consumer, not by source, so a producer
    that fans out to two consumers each wanting a different layout gets a
    *separate* restickify node per consumer -- both reading the one producer.
    We assert the two-node shape directly rather than via the fusion log."""
    (x, za, zb), fn = _shared_producer_two_restickify_case()
    by_source = _restickify_readers_by_source(fn, x, za, zb)
    shared = [src for src, readers in by_source.items() if len(readers) >= 2]
    assert shared, (
        "expected one producer read by >=2 restickify nodes, got "
        f"{ {s: r for s, r in by_source.items()} }"
    )


# ------- Restickify padding: strict (distinct values + torch.equal) ---------
#
# The tolerance-based tests above compare ``randn`` data with atol=rtol=0.1,
# whose fp16 value collisions in [-3, 3] can MASK an element landing in the
# wrong stick.  The tests below feed a distinct-per-element ramp (``_arange``)
# and require exact equality (``_strict``), so a single misplaced element fails.
# They cover the transpose+clone geometries where a misplaced element is most
# likely: an unaligned stick split across blocks, multiple leading batch dims,
# and size-1 dims in or around the stick.

SPLIT_2D = [(65, 4), (67, 4), (128, 67), (130, 33), (67, 128)]


@pytest.mark.parametrize("shape", SPLIT_2D, ids=lambda p: f"{p[0]}x{p[1]}")
def test_strict_2d_transpose_clone(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(0, 1).clone(), x)


# transpose(-2, -1).clone() with >=2 leading batch dims: every batch plane must
# survive.  Covers a single stick block (..64..) and multiple blocks (..65..),
# an unaligned middle (old-stick) dim of size 4, and deeper/larger batch nests.
SPLIT_ND = [
    (4, 91, 72),
    (2, 3, 65, 4),
    (2, 3, 64, 4),
    (2, 2, 65, 64),
    (3, 5, 65, 4),
    (2, 4, 130, 33),
    (2, 2, 3, 65, 4),
    (2, 67, 128),  # single leading batch, unaligned new stick
    (2, 2, 67, 128),  # two leading batch dims, unaligned new stick
]


@pytest.mark.parametrize("shape", SPLIT_ND, ids=lambda p: "x".join(map(str, p)))
def test_strict_nd_transpose_clone(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(-2, -1).clone(), x)


# transpose(1, -1).clone() swaps an inner batch dim with the stick dim, a
# different demoted-middle geometry than transpose(-2, -1).
SPLIT_T1_LAST = [(2, 4, 67, 128), (2, 3, 65, 4), (2, 2, 67, 128)]


@pytest.mark.parametrize("shape", SPLIT_T1_LAST, ids=lambda p: "x".join(map(str, p)))
def test_strict_nd_transpose_1_last_clone(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(1, -1).clone(), x)


# transpose(0, -1).clone() swaps the OUTERMOST dim with the stick dim, with both
# the source and destination stick dims sub-64 (e.g. 2 and 7) so neither fills a
# full stick.  Must still place every element exactly.
SPLIT_T0_LAST = [(7, 67, 2), (7, 65, 2), (5, 3, 2), (7, 67, 63)]


@pytest.mark.parametrize("shape", SPLIT_T0_LAST, ids=lambda p: "x".join(map(str, p)))
def test_strict_transpose_0_last_clone(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(0, -1).clone(), x)


# ======================================================================
# Size-1 dims in and around the stick.
#
# A size-1 host dim collapses upstream Inductor's within-stick loop symbol, which
# these transpose+clone/contiguous geometries must handle without dropping or
# zeroing a plane.  Each family below targets a distinct restickify arm/branch:
#
#   family                        arm / branch exercised
#   ----------------------------  ------------------------------------------
#   SIZE1_INPUT_STICK             size-1 leaves stick, real enters:
#     _clone (.exp -> folds to reinterpret_tensor, allclose oracle)
#     _contiguous_producer        in-place producer grow (_pad_restickify_input)
#     _clone_graph_input          inserted identity clone we grow
#   SIZE1_NEW_STICK               size-1 moves INTO stick; pass declines the
#                                 read, codegen restores the elided OUTPUT stick
#   SIZE1_MULTI_STICK             size-1 in stick + more size-1 dims (clone arm)
#   SIZE1_SURVIVING_BATCH         real batch survives outside BOTH sticks; new
#                                 stick a full 64 (no output-middle pad)
#   REAL_DIM_INTO_STICK + sweeps  real dim into stick via bare-input clone;
#                                 full 23-perm sweep across both arms
#   SIZE1_MULTI_BLOCK             new stick spans >=2 stick blocks
#   SIZE1_MULTI_BLOCK_MID_DIM     + real dim between old stick and plane
#                                 (size-1 old stick not farthest from plane)
# ======================================================================


# A size-1 dim IN the input stick: the transpose moves this size-1 dim out of
# the stick and a real dim in, and every element must come back correctly.
#
# The .exp().transpose().clone() chain folds to a reinterpret_tensor, so no
# restickify is emitted and no padding runs here -- this is the plain
# size-1-in-stick round-trip.  The producer-grow and graph-input-clone padding
# paths are exercised by the .contiguous() and graph-input variants below.  The
# transcendental's last-ULP host/device drift means this asserts allclose rather
# than exact equality.
SIZE1_INPUT_STICK = [(7, 65, 1), (5, 3, 1)]


@pytest.mark.parametrize(
    "shape", SIZE1_INPUT_STICK, ids=lambda p: "x".join(map(str, p))
)
def test_size1_input_stick_transpose_0_last_clone(shape):
    def fn(x):
        return x.exp().transpose(0, -1).clone()

    x = torch.ones(*shape, dtype=torch.float16)
    spyre = _compile_and_run(fn, (x,), DEVICE)
    torch.testing.assert_close(spyre.cpu(), fn(x), atol=1e-2, rtol=1e-2)


# Same size-1-in-stick shapes, produced buffer, but ending in .contiguous()
# instead of .clone() -- the one construction that reaches _pad_restickify_input's
# IN-PLACE PRODUCER GROW: the transpose moves a real dim (7 / 5) into the stick,
# so its unaligned new-stick dim must be padded to a stick boundary, and inductor
# keeps the pointwise producer as a real ComputedBuffer we own and grow in place
# (.clone() would fold to a reinterpret_tensor instead).  Ramp span 511 doubled by
# x + x stays < 1024 (fp16-exact), so torch.equal catches a mis-placed or
# uninitialised row precisely.
@pytest.mark.parametrize(
    "shape", SIZE1_INPUT_STICK, ids=lambda p: "x".join(map(str, p))
)
def test_size1_input_stick_transpose_0_last_contiguous_producer(shape):
    x = _arange(*shape, span=511)
    _strict(lambda x: (x + x).transpose(0, -1).contiguous(), x)


# Same size-1-in-stick contiguous producer, but the real dim that moves INTO the
# stick is > 64, so the restored output stick CROSSES a stick boundary (out=67 ->
# 64 + 3).  This forces the output-buffer grow that covers the full 64-plane
# sweep; its geometry is asserted device-free by the white-box size-1 tests at the
# end of this file.  Ramp span 511 doubled by x + x stays < 1024 (fp16-exact), so
# torch.equal catches a displaced lane.
SIZE1_INPUT_STICK_CROSSING = [(7, 67, 1), (5, 67, 1), (3, 130, 1)]


@pytest.mark.parametrize(
    "shape", SIZE1_INPUT_STICK_CROSSING, ids=lambda p: "x".join(map(str, p))
)
def test_size1_input_stick_crossing_transpose_0_last_contiguous_producer(shape):
    x = _arange(*shape, span=511)
    _strict(lambda x: (x + x).transpose(0, -1).contiguous(), x)


# A MULTI-STAGE pointwise producer (two or more fused ops) feeding the same
# transpose + .contiguous() restickify.  A fused chain iterates one order, but the
# transpose flips it on one interior edge: a buffer written in the original order
# and read transposed must not come back as a lane permutation (same values, wrong
# positions).  The single-stage case above cannot exercise this -- it has no
# interior buffer.  Each direct producer is read in the producer's own order and
# only the store is permuted, so no interior buffer is dual-ordered.
#
# Two axes both matter and are swept independently (a full cross-product would add
# nothing -- the crossing edge is the same regardless of the combination):
#   * producer SHAPE, on a fixed linear 2-then-3-stage chain -- covers a size-1
#     last dim and a non-size-1 last dim (dim 2), proving it is not size-1-specific;
#   * producer DAG SHAPE, on the fixed [2,3,4] repro -- multi-input, diamond, and
#     pointwise-stages-on-both-sides-of-the-transpose.
# Ramp span 255 through the chains stays < 1024 (fp16-exact), so torch.equal pins
# a displaced lane exactly.
MULTISTAGE_PRODUCER_CASES = [
    # (builder, shape, id): linear-chain shape sweep.
    (
        lambda x: (x + x).mul(2.0).transpose(0, -1).contiguous(),
        (2, 3, 4),
        "2stage_2x3x4",
    ),
    (
        lambda x: (x + x).mul(2.0).transpose(0, -1).contiguous(),
        (5, 3, 2),
        "2stage_5x3x2",
    ),
    (
        lambda x: (x + x).mul(2.0).transpose(0, -1).contiguous(),
        (5, 3, 1),
        "2stage_5x3x1",
    ),
    # A third stage extends the producer chain one more step.
    (
        lambda x: (x + x).mul(2.0).add(1.0).transpose(0, -1).contiguous(),
        (2, 3, 4),
        "3stage_2x3x4",
    ),
    (
        lambda x: (x + x).mul(2.0).add(1.0).transpose(0, -1).contiguous(),
        (5, 3, 2),
        "3stage_5x3x2",
    ),
    (
        lambda x: (x + x).mul(2.0).add(1.0).transpose(0, -1).contiguous(),
        (5, 3, 1),
        "3stage_5x3x1",
    ),
    # DAG-shape sweep on the [2,3,4] repro.
    (lambda x: (x + x * 3.0).transpose(0, -1).contiguous(), (2, 3, 4), "multi_input"),
    (
        lambda x: ((x + 1.0) * (x + 2.0)).transpose(0, -1).contiguous(),
        (2, 3, 4),
        "diamond",
    ),
    (
        lambda x: (x + x).mul(2.0).add(1.0).transpose(0, -1).mul(3.0).contiguous(),
        (2, 3, 4),
        "stages_both_sides",
    ),
]


@pytest.mark.parametrize(
    "builder,shape",
    [(b, s) for b, s, _ in MULTISTAGE_PRODUCER_CASES],
    ids=[i for _, _, i in MULTISTAGE_PRODUCER_CASES],
)
def test_multistage_producer_transpose_contiguous(builder, shape):
    x = _arange(*shape, span=255)
    _strict(builder, x)


# A size-1 dim moving INTO the stick (the mirror of SIZE1_INPUT_STICK): the
# transpose destination stick dim has host size 1, so upstream Inductor elides
# the OUTPUT operand's within-stick loop symbol.  The pass declines the
# size-1-new-stick read (there is no over-read to cover) and codegen restores the
# elided OUTPUT stick, so both must come back bit-exact.
#
# Each entry is (shape, transpose_dims); .contiguous() (not .clone(), which folds
# to reinterpret_tensor) keeps a real producer ComputedBuffer and forces the
# restickify.  Ramp span 511 doubled by x + x stays < 1024 (fp16-exact), so
# torch.equal catches a mis-placed lane precisely.
SIZE1_NEW_STICK = [
    ((1, 1, 5, 67), (1, 3)),  # size-1 dim 1 moves into stick; real 5 survives
    ((1, 3, 1, 67), (0, 3)),  # size-1 dim 0 moves into stick; real 3 survives
]


@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_NEW_STICK,
    ids=[
        f"{'x'.join(map(str, s))}_t{'_'.join(map(str, d))}" for s, d in SIZE1_NEW_STICK
    ],
)
def test_size1_new_stick_transpose_contiguous(shape, dims):
    x = _arange(*shape, span=511)
    _strict(lambda x: (x + x).transpose(*dims).contiguous(), x)


# Same size-1 dim moving INTO the stick, but the restickify input is a bare graph
# input.  Uses .clone() -- when the elided operand is the OUTPUT, the clone does
# NOT fold to reinterpret_tensor (the destination stick's size-1 dim keeps a real
# restickify), so this exercises the elided-output path for a plain input.
@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_NEW_STICK,
    ids=[
        f"{'x'.join(map(str, s))}_t{'_'.join(map(str, d))}" for s, d in SIZE1_NEW_STICK
    ],
)
def test_size1_new_stick_transpose_clone_graph_input(shape, dims):
    x = _arange(*shape, span=511)
    _strict(lambda x: x.transpose(*dims).clone(), x)


# Symmetric mirror of SIZE1_INPUT_STICK_CROSSING, on the elided-OUTPUT side.  A
# size-1 dim moves INTO the stick (output-elided), and the real dim moving OUT of
# the stick is > 64, so the INTACT operand -- here the INPUT -- crosses a stick
# boundary.  _restickify_restore_elided_dim binds the shared symbol to the
# outermost size-64 gap dim on the intact input just as the mirror does on the
# intact output, so the input buffer needs the same grow.  _pad_restickify_input
# supplies it by ensuring the read comes from a buffer we own -- a producer we
# grow in place, or, for a graph input, a materialised identity clone we grow --
# via padding._pad_elided_dim, which prepends the size-64 gap dim the restore then
# reuses.  ``[1,67,7]`` is the mirror of ``[7,67,1]``; ``[1,67,130]`` /
# ``[1,130,3]`` push the intact-input stick past 64 to force the crossing.  Ramp
# span 511 doubled by x + x stays < 1024 (fp16-exact), so torch.equal pins a
# displaced lane precisely.
SIZE1_NEW_STICK_CROSSING = [
    ((1, 67, 7), (0, 2)),  # mirror of [7,67,1]
    ((1, 67, 130), (0, 2)),  # intact input stick 130 -> crosses (64 + 66)
    ((1, 130, 3), (0, 2)),  # crossing with a real middle dim 130
]


@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_NEW_STICK_CROSSING,
    ids=[
        f"{'x'.join(map(str, s))}_t{'_'.join(map(str, d))}"
        for s, d in SIZE1_NEW_STICK_CROSSING
    ],
)
def test_size1_new_stick_crossing_transpose_contiguous_producer(shape, dims):
    # (x + x) is a real compute op, so its output is a buffer we own: the producer
    # arm tags it directly.  A bare transpose would instead read the graph input.
    x = _arange(*shape, span=511)
    _strict_size1_input_arm(
        lambda x: (x + x).transpose(*dims).contiguous(), x, "producer"
    )


@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_NEW_STICK_CROSSING,
    ids=[
        f"{'x'.join(map(str, s))}_t{'_'.join(map(str, d))}"
        for s, d in SIZE1_NEW_STICK_CROSSING
    ],
)
def test_size1_new_stick_crossing_transpose_clone_graph_input(shape, dims):
    # A bare transpose of a graph input keeps the input as a direct restickify
    # read (no compute op interposes a producer buffer), so the clone arm fires:
    # an identity clone is materialised and tagged for the grow.
    x = _arange(*shape, span=511)
    _strict_size1_input_arm(lambda x: x.transpose(*dims).clone(), x, "clone")


# Same size-1-in-stick shapes, but the restickify input is a bare graph input
# (no producer op), exercising the other input path -- an inserted identity
# clone we grow.  A copy preserves the input bit-for-bit, so this asserts exact
# equality on the ramp (unlike the allclose above).
@pytest.mark.parametrize(
    "shape", SIZE1_INPUT_STICK, ids=lambda p: "x".join(map(str, p))
)
def test_size1_input_stick_transpose_0_last_clone_graph_input(shape):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(0, -1).clone(), x)


# A size-1 dim in the input stick PLUS at least one more size-1 host dim
# elsewhere, so more than one size-1 dim is present at once.  The transpose must
# place every real batch plane correctly regardless of where the extra size-1
# dims sit.  The clone path (x.transpose(*dims).clone()) covers two arrangements:
#   * the extra size-1 dims are the two UNTOUCHED dims, possibly interleaved with
#     a real dim between them (the first five entries), and
#   * an extra size-1 dim sits OUTSIDE both sticks, leading or middle, with one or
#     two of them (the last five entries).
# Each entry is (shape, transpose_dims); the transpose swaps the size-1
# input-stick dim with a real dim.  Distinct-ramp + torch.equal catches a
# mis-placed plane exactly.
SIZE1_MULTI_STICK = [
    # Extra size-1 dims are the untouched dims (or interleaved with a real dim).
    ((1, 1, 64, 1), (0, -1)),  # three size-1 dims (0, 1, 3)
    ((1, 1, 67, 1), (0, -1)),  # three size-1 dims, unaligned stick
    ((1, 5, 67, 1), (0, -1)),  # interleaved: size-1 at 0 and 3, real 5/67 between
    ((7, 1, 64, 1), (1, 3)),  # interleaved: size-1 at 1 and 3, real 7/64 between
    ((5, 1, 67, 1), (1, 3)),  # interleaved: size-1 at 1 and 3, real 5/67 between
    # Extra size-1 dim(s) outside both sticks (leading or middle).
    ((1, 4, 64, 1), (2, 3)),  # extra size-1 leading (dim0)
    ((4, 1, 64, 1), (2, 3)),  # extra size-1 in the middle (dim1)
    ((1, 4, 1, 64, 1), (3, 4)),  # two extra size-1, batch outer
    ((4, 1, 1, 64, 1), (3, 4)),  # two extra size-1, batch/size-1 interleaved
    ((1, 1, 4, 64, 1), (3, 4)),  # two extra size-1, batch inner
]


@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_MULTI_STICK,
    ids=[f"{'x'.join(map(str, s))}_t{d}" for s, d in SIZE1_MULTI_STICK],
)
def test_size1_multi_input_stick_transpose_clone(shape, dims):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(*dims).clone(), x)


# A size-1 dim in the input stick where a real batch/leading dim (extent > 1)
# survives OUTSIDE both the old (size-1) and new sticks -- e.g. (4, 64, 1)
# transpose(1, 2), whose batch dim 0 stays leading while dims 1 and 2 swap.
# Every surviving batch plane must come back correct.
#
# The new (destination) stick is a full 64 here on purpose: an unaligned
# destination would additionally need output-middle padding, covered separately
# above.  Each entry is (shape, transpose_dims).
SIZE1_SURVIVING_BATCH = [
    ((4, 64, 1), (1, 2)),  # batch dim 0 = 4 survives; new stick = dim 1
    ((2, 64, 1), (1, 2)),  # smaller batch
    ((3, 5, 64, 1), (2, 3)),  # batch 3 + spatial 5 both survive; new stick dim 2
    ((2, 2, 64, 1), (2, 3)),  # two leading dims survive
    ((2, 3, 4, 64, 1), (3, 4)),  # deep batch nest
]


@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_SURVIVING_BATCH,
    ids=[f"{'x'.join(map(str, s))}_t{d}" for s, d in SIZE1_SURVIVING_BATCH],
)
def test_size1_input_stick_surviving_batch_transpose_clone(shape, dims):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(*dims).clone(), x)


# A REAL (extent > 1) dim moving into the stick when the restickify input is a
# bare graph input (no producer op to grow in place).  This is the clone-arm
# mirror of test_multistage_producer_transpose_contiguous: the pass inserts an
# identity clone of the input, grows its new-stick dim, and expresses the
# transpose as a Scatter that reads the clone straight and permutes on the store.
#
# Shape [3,2,5,67] has NO size-1 dim, so it covers the real-dim (non-size-1) clone
# arm.  The size-1 shape [3,1,5,67] is covered exhaustively (every perm) by
# test_full_perm_sweep_transpose_clone_graph_input below, so it is not repeated
# here.
#
# Must use .permute(...).contiguous(): .clone() on a bare input folds to
# reinterpret_tensor and never restickifies.  Each perm moves a real dim (size 3
# or 5) into the stick; the reordered variants ALSO permute the outer real dims,
# so the store carries a compound transpose.  Ramp span 511 keeps every value
# < 1024 (fp16-exact), so torch.equal pins a mis-placed lane exactly.
REAL_DIM_INTO_STICK_PERMS = [
    (0, 1, 3, 2),  # size-5 dim into stick, outer dims in order
    (2, 1, 0, 3),  # size-3 dim into an outer slot, outer dims reordered
    (0, 3, 1, 2),  # size-5 dim into stick, outer real dims reordered (compound)
    (3, 2, 1, 0),  # full reversal -- size-3 dim into stick, all reordered
]


@pytest.mark.parametrize(
    "perm", REAL_DIM_INTO_STICK_PERMS, ids=lambda p: "".join(map(str, p))
)
def test_real_dim_into_stick_transpose_clone_graph_input(perm):
    x = _arange(3, 2, 5, 67, span=511)
    _strict(lambda x: x.permute(*perm).contiguous(), x)


# Full sweep of every non-identity permutation of [3,1,5,67] via
# permute(p).contiguous(), across both restickify input arms.  This generalises
# test_real_dim_into_stick_transpose_clone_graph_input (which pins a subset of
# perms for the clone arm) to the whole permutation group and to the producer arm.
#
#   clone arm:    x.permute(p).contiguous()            -- restickify reads a bare
#                 graph input; the pass inserts + grows an identity clone.
#   producer arm: (x + x * 3.0).permute(p).contiguous() -- a pointwise op
#                 materialises a ComputedBuffer the pass grows in place.  x * 3.0
#                 keeps the producer from folding away, so this exercises a path
#                 the clone arm does not.
#
# The producer arm interposes a pointwise op whose transpose-crossing intermediate
# is staged in LX and work-divided across several cores, so it also exercises the
# multi-core LX staging path (see #3340).  The two producer sweeps below cover
# complementary regimes: the padding sweep pins sencores=1 so no work-division
# occurs (padding restickify in isolation); the aligned sweep runs at the default
# core count so the transpose reorders work-divided dims.  Every perm on both arms
# must be bit-exact.
_SWEEP_SHAPE = (3, 1, 5, 67)
_ALL_PERMS = [p for p in itertools.permutations(range(4)) if p != (0, 1, 2, 3)]


@pytest.mark.parametrize("perm", _ALL_PERMS, ids=lambda p: "".join(map(str, p)))
def test_full_perm_sweep_transpose_clone_graph_input(perm):
    # Clone arm copies the input through unchanged, so any band <= 1024 is
    # fp16-exact; span 511 widens distinct-lane coverage.
    x = _arange(*_SWEEP_SHAPE, span=511)
    _strict(lambda x: x.permute(*perm).contiguous(), x)


# Producer-arm padding sweep on the UNALIGNED shape [3,1,5,67] (last dim 67 pads to
# 128 = two sticks, exercising the restickify padding this PR adds).  sencores=1
# removes work-division, so the LX transpose-crossing defect cannot fire and the
# result is purely the padding restickify -- which is bit-exact for every perm.
@config.patch({"sencores": 1})
@pytest.mark.parametrize("perm", _ALL_PERMS, ids=lambda p: "".join(map(str, p)))
def test_full_perm_sweep_transpose_contiguous_producer(perm):
    # t + t * 3.0 == 4 * t, so the input band must satisfy 4 * span <= 1024 to stay
    # fp16-exact (else CPU-fp16 vs device-DLFloat16 rounding is a false mismatch).
    x = _arange(*_SWEEP_SHAPE, span=255)
    _strict(lambda x: (x + x * 3.0).permute(*perm).contiguous(), x)


# The producer arm on the aligned shape [3,1,5,64] at the default core count, so
# every perm that reorders the two real outer dims (3 and 5) drives them into the
# multi-core work-divided regime -- the transpose-crossing LX staging path (#3340).
# On the aligned shape (last dim 64 = one stick) there is no padding, so this
# exercises that path in isolation.  Every perm must be bit-exact.
_ALIGNED_SWEEP_SHAPE = (3, 1, 5, 64)


@pytest.mark.parametrize("perm", _ALL_PERMS, ids=lambda p: "".join(map(str, p)))
def test_full_perm_sweep_aligned_producer(perm):
    x = _arange(*_ALIGNED_SWEEP_SHAPE, span=255)
    _strict(lambda x: (x + x * 3.0).permute(*perm).contiguous(), x)


# Producer-arm restickify at the DEFAULT core count on the unaligned shape
# [5,3,66], so the transpose-crossing intermediate is work-divided across cores
# (unlike the sencores=1 sweep above).  A two-stage pointwise (x + x*3) feeds a
# permute+contiguous whose output lands a different host dim within the stick --
# the "into" restickify.  The staged intermediate is a *transitive* producer of
# that restickify (it feeds an already-spilled intermediate, which feeds the
# restickify), and its output must land bit-exact for every "into" perm.  The
# even stick extent (66) keeps the padded stick clear of the odd-last-dim LX
# work-division defect, so the "into" perms are correct regardless of residency.
#
# perm (1,0,2) swaps only the two non-stick outer dims (dim2=66 stays the stick),
# so no host dim moves into the stick and no restickify fires: it exercises the
# same multi-core LX work-division path as the aligned sweep above (#3340), not an
# "into" case.  Every perm here must be bit-exact.
_INTO_SWEEP_SHAPE = (5, 3, 66)


@pytest.mark.parametrize(
    "perm",
    [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)],
    ids=lambda p: "".join(map(str, p)),
)
def test_full_perm_sweep_into_producer_default_cores(perm):
    # t + t * 3.0 == 4 * t, so span must satisfy 4 * span <= 1024 for fp16 exactness.
    x = _arange(*_INTO_SWEEP_SHAPE, span=255)
    _strict(lambda x: (x + x * 3.0).permute(*perm).contiguous(), x)


# A size-1 dim in the input stick whose NEW stick dim spans >=2 stick blocks
# (host size > 64), aligned or unaligned, with and without leading batch dims.
# Every stick block must land correctly.  Distinct-ramp + torch.equal.
SIZE1_MULTI_BLOCK = [
    ((1, 128, 1), (1, 2)),  # 2 aligned blocks, no batch
    ((1, 192, 1), (1, 2)),  # 3 aligned blocks, no batch
    ((1, 67, 1), (1, 2)),  # 2 unaligned blocks, no batch
    ((4, 128, 1), (1, 2)),  # 2 aligned blocks + batch 4
    ((2, 128, 1), (1, 2)),  # 2 aligned blocks + batch 2
    ((4, 67, 1), (1, 2)),  # 2 unaligned blocks + batch
    ((4, 192, 1), (1, 2)),  # 3 blocks + batch
    ((2, 3, 67, 1), (2, 3)),  # 2 unaligned blocks + leading batch nest
]


@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_MULTI_BLOCK,
    ids=[f"{'x'.join(map(str, s))}_t{d}" for s, d in SIZE1_MULTI_BLOCK],
)
def test_size1_multi_block_transpose_clone(shape, dims):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(*dims).clone(), x)


# A size-1 old stick with a MULTI-BLOCK new stick (host > 64) AND a real
# (size > 1) dim positioned BETWEEN the old stick and the batch/plane dims, so
# the collapsed old stick lands at a device dim that is NOT the farthest from the
# plane.  Every stick block must land correctly regardless of where the size-1
# old stick sits.  Distinct-ramp + torch.equal catches a misplaced block exactly.
SIZE1_MULTI_BLOCK_MID_DIM = [
    ((1, 1, 2, 128, 1), (3, 4)),  # real dim (2) between old stick and plane
    ((1, 1, 3, 128, 1), (3, 4)),  # size-3 mid dim
    ((1, 1, 2, 192, 1), (3, 4)),  # 3 blocks
    ((1, 1, 2, 67, 1), (3, 4)),  # unaligned blocks
    ((1, 3, 2, 128, 1), (3, 4)),  # plane + mid dim both real
]


@pytest.mark.parametrize(
    "shape,dims",
    SIZE1_MULTI_BLOCK_MID_DIM,
    ids=[f"{'x'.join(map(str, s))}_t{d}" for s, d in SIZE1_MULTI_BLOCK_MID_DIM],
)
def test_size1_multi_block_mid_dim_transpose_clone(shape, dims):
    x = _arange(*shape)
    _strict(lambda x: x.transpose(*dims).clone(), x)


# A restickify input sliced with a contiguous OFFSET on a NON-stick host dim
# (e.g. x[1:3]), with an unaligned new stick (67) that needs padding.  This is a
# valid, paddable slice and must compile correctly.  Distinct-ramp + torch.equal
# catches a misplaced plane exactly.
OFFSET_NONSTICK_INPUT = [
    # Graph-input leading-dim offset (rows 1..2 of 4), unaligned new stick (67).
    (lambda x: x[1:3].transpose(1, 2).clone(), (4, 67, 128)),
    # Offset on a middle (non-leading, non-stick) dim.
    (lambda x: x[:, 1:3].transpose(2, 3).clone(), (2, 4, 67, 128)),
]


@pytest.mark.parametrize(
    "fn,shape",
    OFFSET_NONSTICK_INPUT,
    ids=["x".join(map(str, s)) for _, s in OFFSET_NONSTICK_INPUT],
)
def test_offset_nonstick_input_transpose_clone(fn, shape):
    x = _arange(*shape)
    _strict(fn, x)


# A STRIDED (step > 1) read of a restickify input (``x[::2]``) is not paddable:
# the strided rows are non-adjacent, so a copy would read the wrong data.  With
# an unaligned new stick (67) that needs padding, this must fail loudly rather
# than silently miscompile.
def test_strided_input_transpose_clone_raises():
    x = _arange(4, 67, 128)
    with pytest.raises(RuntimeError, match="strided input on host dim"):
        _compile_and_run(lambda x: x[::2].transpose(1, 2).clone(), (x,), DEVICE)


# A BROADCAST read (a dim iterated wider than its size) coexisting with an
# unaligned new-stick dim that needs padding.  The rope view
# ``k.view(B, S, H, D)`` on a ``[B, S, H, 2, 1, D/2]`` input folds the size-2 dim
# into the last dim, so after ``transpose(2, 3)`` that dim is read with a
# zero-coefficient coord (``floor(v/64)`` / ``Mod(v, 64)``) -- a re-read, not a
# stride.  Unlike ``x[::2]`` above, this is fine: codegen derives the read's
# device strides from the device layout, so the broadcast reads correctly while
# the unaligned new stick (S=67) is padded.  Every element must land exactly.
def test_broadcast_input_transpose_clone():
    def fn(k):
        B, S, H = k.shape[0], k.shape[1], k.shape[2]
        D = k.shape[-1] * 2
        return k.view(B, S, H, D).transpose(1, 2).transpose(2, 3).clone()

    x = _arange(1, 67, 1, 2, 1, 64)
    _strict(fn, x)


# Matmul with a SUB-STICK contraction dim (D) reached through a permuted key,
# k.permute(0,2,3,1), which restickifies k into [B,H,D,Lk] with the sub-stick D
# demoted to a non-stick device dim.  When D does not fill a whole stick (D=48)
# the restickify's widened read runs past D's initialized lanes; D must be padded
# to a stick boundary (and its K-tail zero-filled) so the contraction ignores the
# pad rows.  Read-side padding grows D's device dim device-size-only, which covers
# the over-read for every D here (an already-full D=64 is a no-op grow).
# Small-span ramps keep the fp16 accumulation exact so torch.equal is a true
# oracle; D spans sub-stick (33, 48), aligned (64), and multi-block (96) sizes.
SUBSTICK_MATMUL_D = [33, 48, 63, 64, 96]


@pytest.mark.parametrize("D", SUBSTICK_MATMUL_D, ids=lambda d: f"D{d}")
def test_substick_contraction_permuted_key(D):
    B, H, Lq, Lk = 2, 4, 16, 32
    q = _arange(B, Lq, H, D, span=3)
    k = _arange(B, Lk, H, D, span=3)

    def fn(q, k):
        return torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))

    _strict(fn, q, k)


def test_2d_sparse_broadcast_dense_pointwise():
    """a.sum(-1) + b - reduction output broadcast into pointwise with dense b."""
    a = torch.randn((S, S), dtype=torch.float16)
    b = torch.randn((S, S), dtype=torch.float16)
    _compare(lambda a, b: a.amin(-1) + b, a, b, optimal_cost=S * S)


def test_3d_sparse_broadcast_dense_pointwise():
    """a.sum(-1) + b - reduction output broadcast into pointwise with dense b."""
    a = torch.randn((S, S, S), dtype=torch.float16)
    b = torch.randn((S, S, S), dtype=torch.float16)
    _compare(lambda a, b: a.amin(-1) + b, a, b, optimal_cost=S * S * S)


def test_sparse_dense_pointwise_d0_stick():
    """a.sum(-1) + b where b has a d0 stick — verifies sparse detection with alt-dim candidate."""

    a = torch.randn((S, S, S), dtype=torch.float16).to(DEVICE)
    b_layout = SpyreTensorLayout([S, S], [S, 1], torch.float16, [1, 0])
    b = torch.randn((S, S), dtype=torch.float16).to(device_layout=b_layout)
    with pytest.raises(
        InductorError, match="No mechanism to gather elements from multiple sticks"
    ):
        _compare(lambda a, b: a.amin(-1) + b, a, b)


def test_sparse_broadcast_dense_pointwise_d0_stick():
    """a.sum(-1) + b where b has a d0 stick — verifies sparse detection with alt-dim candidate."""

    a = torch.randn((S, S), dtype=torch.float16)
    b = torch.randn((S, S), dtype=torch.float16)
    b_layout = SpyreTensorLayout([S, S], [S, 1], torch.float16, [1, 0])
    b_dev = b.to(device_layout=b_layout)
    _compare(
        lambda a, b: a.amin(-1) + b,
        a,
        b,
        device_args=[a.to(DEVICE), b_dev],
        optimal_cost=0,
    )


def test_broadcast_dense_pointwise():
    a = torch.randn((S), dtype=torch.float16)
    b = torch.randn((S, S), dtype=torch.float16)
    _compare(lambda a, b: a + b, a, b, optimal_cost=0)


def test_broadcast_3d_dense_pointwise():
    a = torch.randn((S, S), dtype=torch.float16)
    b = torch.randn((S, S, S), dtype=torch.float16)
    _compare(lambda a, b: a + b, a, b, optimal_cost=0)


def test_unsqueeze_broadcast_dense_pointwise():
    """a.unsqueeze(-1) + b - unsqueeze broadcast followed by pointwise."""
    a = torch.randn((S,), dtype=torch.float16)
    b = torch.randn((S, S), dtype=torch.float16)
    _compare(lambda a, b: a.unsqueeze(-1) + b, a, b, optimal_cost=S * S)


def test_unsqueeze_broadcast_dense_pointwise_d0_stick():
    """a.unsqueeze(-1) + b where b has a d0 stick — verifies sparse detection with alt-dim candidate."""

    a = torch.randn((S,), dtype=torch.float16)
    b = torch.randn((S, S), dtype=torch.float16)
    b_layout = SpyreTensorLayout([S, S], [S, 1], torch.float16, [1, 0])
    b_dev = b.to(device_layout=b_layout)
    _compare(
        lambda a, b: a.unsqueeze(-1) + b,
        a,
        b,
        device_args=[a.to(DEVICE), b_dev],
        optimal_cost=0,
    )


def test_unsqueeze_expand_broadcast_dense_pointwise():
    """a.unsqueeze(-1).expand(S, S) + b - unsqueeze+expand broadcast followed by pointwise."""
    a = torch.randn((S,), dtype=torch.float16)
    b = torch.randn((S, S), dtype=torch.float16)
    _compare(lambda a, b: a.unsqueeze(-1).expand(S, S) + b, a, b, optimal_cost=S * S)


def test_unsqueeze_expand_broadcast_dense_pointwise_d0_stick():
    """a.unsqueeze(-1).expand(S, S) + b where b has a d0 stick — verifies sparse detection with alt-dim candidate."""

    a = torch.randn((S,), dtype=torch.float16)
    b = torch.randn((S, S), dtype=torch.float16)
    b_layout = SpyreTensorLayout([S, S], [S, 1], torch.float16, [1, 0])
    b_dev = b.to(device_layout=b_layout)
    _compare(
        lambda a, b: a.unsqueeze(-1).expand(S, S) + b,
        a,
        b,
        device_args=[a.to(DEVICE), b_dev],
        optimal_cost=0,
    )


# ------- Broadcast outer-product tests ---------


def test_broadcast_outer_diff():
    """acs.unsqueeze(-1) - acs.unsqueeze(-2): outer-product subtraction over [BH, C].

    Both reads of acs conflict (stick on d1 vs d2); the optimizer picks stick on
    dim 0 (BH) at cost 2 * acs.numel() — two restickifies of the 2D input.
    """
    BH, C = 64, 64
    acs = torch.randn((BH, C), dtype=torch.float16)
    _compare(
        lambda acs: torch.exp(acs.unsqueeze(-1) - acs.unsqueeze(-2)),
        acs,
        optimal_cost=2 * acs.numel(),
        check_strides=False,
    )


def test_broadcast_expand_first():
    """expand on unsqueeze(-1) only — one input expanded, one bare unsqueeze."""
    BH, C = 64, 64
    acs = torch.randn((BH, C), dtype=torch.float16)
    _compare(
        lambda acs: torch.exp(acs.unsqueeze(-1).expand(-1, -1, C) - acs.unsqueeze(-2)),
        acs,
        optimal_cost=2 * acs.numel(),
        check_strides=False,
    )


def test_broadcast_expand_second():
    """expand on unsqueeze(-2) only — one input expanded, one bare unsqueeze."""
    BH, C = 64, 64
    acs = torch.randn((BH, C), dtype=torch.float16)
    _compare(
        lambda acs: torch.exp(acs.unsqueeze(-1) - acs.unsqueeze(-2).expand(-1, C, -1)),
        acs,
        optimal_cost=2 * acs.numel(),
        check_strides=False,
    )


def test_broadcast_expand_both():
    """expand on both unsqueezes — both inputs fully expanded to [BH, C, C]."""
    BH, C = 64, 64
    acs = torch.randn((BH, C), dtype=torch.float16)
    _compare(
        lambda acs: torch.exp(
            acs.unsqueeze(-1).expand(-1, -1, C) - acs.unsqueeze(-2).expand(-1, C, -1)
        ),
        acs,
        optimal_cost=2 * acs.numel(),
        check_strides=False,
    )


# ------- Single-arg op with a size-1 interior dim ---------


def test_single_arg_size1_interior_dim():
    """Slicing one position out of a [B, H, L, D] tensor yields a size-1 interior dim.

    The size-1 seq dim is offered as a stick candidate for the single-arg op that
    produces it; its stick expression concretizes to 1, which device_coordinates
    rejects. That candidate must be skipped, not abort the compile. Mirrors the
    per-position KV-cache write in decoder inference (input [1, 2, L, 128] ->
    [1, 2, 1, 128]).
    """
    x = torch.randn((1, 2, T, 128), dtype=torch.float16)
    _compare(lambda x: x[:, :, 1:2, :].contiguous(), x)


def test_single_arg_size1_after_staggered_ea():
    """RoPE output written into a KV cache at a single token position.

    A RoPE-style mul+sum produces a staggered 7-dim device layout for k of
    shape [1, NKV, T, HD]. The single-position KV write slices k to
    [1, NKV, 1, HD] and scatters it into the cache via copy_. The mutation op
    is a single-arg op whose output is [1, NKV, 1, HD]; when propagate_layouts
    evaluates the staggered input STL against this op's dep, the stick
    expression concretizes to the literal 1 (the size-1 seq dim). That
    candidate must be skipped via try_device_coordinates, not abort the compile.

    test_single_arg_size1_interior_dim does not cover this because its input
    STL is plain row-major; here the input STL is a committed staggered-EA
    layout from the RoPE op.
    """
    NKV, HD = 2, 128
    half = HD // 2
    KVLEN = 3 * T  # cache longer than one block

    def fn(x, sf, kc):
        # RoPE: [1, NKV, T, HD] -> staggered-EA output
        B, H, L, D = x.shape
        h = D // 2
        x_ = x.transpose(1, 2).reshape(B, L, H, 2, h)
        sf6 = sf[:, :, None, :, :, :]
        k = sf6.mul(x_.unsqueeze(-3)).sum(4, keepdim=True).flatten(3)
        k = k.transpose(1, 2)
        # Single-position KV write: slice [1, NKV, 1, HD] into cache
        kw = k[:, :, 1:2, :]
        kc[:, :, T : T + 1, :] = kw
        return kc

    x = torch.randn((1, NKV, T, HD), dtype=torch.float16)
    sf = torch.randn((1, T, 2, 2, half), dtype=torch.float16)
    kc = torch.randn((1, NKV, KVLEN, HD), dtype=torch.float16)
    _compare(fn, x, sf, kc)


# --- size-1 old-stick under-allocation regression guard ----------------------
#
# This test looks contrived -- most of its body is arena setup, not the op under
# test.  That setup is the point: it is the recipe that first surfaced the size-1
# old-stick under-allocation bug, kept here so the bug cannot silently return.  On
# `(x + x).transpose(0, -1).contiguous()` over the (mb, out, 1) family the output
# allocation was one plane short and the store overran it.
#
# A plain equality check would not catch that -- the extra plane lands on whatever
# happens to be there, often benign.  The warming ops below make the overrun
# observable deterministically: they free the region the store would overhang and
# leave a zeros residue, so an out-of-bounds plane reads 0.0 while the correct
# output is a nonzero ramp -- a mismatch on every overrun plane.

# Shapes in the (mb, out, 1) size-1 old-stick family, spread across mb and out so
# they land at differing pool offsets.
_COVERAGE_GAP_SHAPES = [
    (7, 67, 1),
    (7, 66, 1),
    (7, 65, 1),
    (7, 63, 1),
    (5, 67, 1),
    (9, 67, 1),
]


@pytest.mark.parametrize(
    "shape", _COVERAGE_GAP_SHAPES, ids=lambda p: "x".join(map(str, p))
)
def test_restickify_coverage_gap(shape):
    # Warm the arena so a short allocation would overrun into a freed, zeroed
    # region (see the block comment): the large shapes leave the zeros residue,
    # the small ones position the target's pool over it.
    for warm_shape in [(2, 1025, 1024), (2, 2, 1025, 1024), (67, 64), (67, 67)]:
        z = torch.zeros(warm_shape, dtype=torch.float16)
        _compile_and_run(lambda x: x.transpose(-2, -1).clone(), (z,), DEVICE)

    x = _arange(*shape, span=511)
    _strict(lambda t: (t + t).transpose(0, -1).contiguous(), x)


# Broadcasting an elided size-one axis into a new stick is an identity fill.
_BROADCAST_INTO_STICK_SHAPES = [
    ((1, 8, 1), 64),
    ((1, 8, 1), 67),
    ((3, 5, 1), 64),
    ((1, 1, 1), 64),
]


@pytest.mark.parametrize(
    "shape,bwidth",
    _BROADCAST_INTO_STICK_SHAPES,
    ids=[f"{'x'.join(map(str, s))}_b{b}" for s, b in _BROADCAST_INTO_STICK_SHAPES],
)
def test_restickify_broadcast_into_stick(shape, bwidth):
    x = _arange(*shape, span=511)
    _strict(lambda t: t.expand(*shape[:-1], bwidth).contiguous(), x)


# --------------------------------------------------------------------------
# White-box (device-free) geometry of the size-1 stick allocation grow.
#
# A restickify whose old stick has host size 1 collapses one operand's
# within-stick coordinate to a constant, so that operand's STL carries the old
# stick as a size-1 ``stride_map == -1`` singleton dim, while the restore
# (_restickify_restore_elided_dim) rebuilds the descriptor to sweep a full
# 64-plane stick.  The grow is two coupled halves: padding._pad_elided_dim
# prepends an outermost size-64 ``-1`` gap dim to the (intact) operand's STL so
# the allocation covers the sweep, and the restore REUSES that dim (binds the
# shared iteration symbol to it) rather than inserting its own, keeping the
# descriptor total equal to the grown allocation.  These tests pin each half
# against the real STL / OpSpec APIs, without a device (the functional
# lane-displacement checks are the *_stick_crossing_* tests above).
_SIZE1_COLLAPSED_DEVICE_SIZE = [67, 1, 1, 64]
_SIZE1_COLLAPSED_STRIDE_MAP = [7, 64, -1, 1]
_SIZE1_HOST_SIZE = [1, 67, 7]
_SIZE1_HOST_STRIDE = [469, 7, 1]
_SIZE1_STICK = 64  # SEN169_FP16 elems_per_stick


def test_size1_grow_prepends_size64_gap_dim():
    from torch_spyre._C import DataFormats, ElementArrangement
    from torch_spyre._C import SpyreTensorLayout as _CSpyreTensorLayout
    from torch_spyre._inductor.ir import FixedTiledLayout
    from torch_spyre._inductor.padding import _pad_elided_dim

    # _C.SpyreTensorLayout takes device_size LITERALLY (unlike torch.spyre's
    # SpyreTensorLayout, which treats the first arg as a host shape and re-derives
    # the device layout); we assert on exact device dims here.
    stl = _CSpyreTensorLayout(
        list(_SIZE1_COLLAPSED_DEVICE_SIZE),
        list(_SIZE1_COLLAPSED_STRIDE_MAP),
        DataFormats.SEN169_FP16,
        ElementArrangement.STANDARD,
    )
    layout = FixedTiledLayout(
        DEVICE, torch.float16, list(_SIZE1_HOST_SIZE), list(_SIZE1_HOST_STRIDE), stl
    )

    # _pad_elided_dim reads buf.get_layout() and writes buf.layout; a 2-line stub
    # is all the ComputedBuffer surface it touches.
    class _StubBuf:
        def __init__(self, layout):
            self.layout = layout

        def get_layout(self):
            return self.layout

    buf = _StubBuf(layout)
    _pad_elided_dim(buf)

    dl = buf.layout.device_layout
    # An outermost size-64 gap dim (stride_map -1) is prepended; the original
    # device dims follow unchanged.
    assert list(dl.device_size) == [_SIZE1_STICK, *_SIZE1_COLLAPSED_DEVICE_SIZE]
    assert list(dl.stride_map) == [-1, *_SIZE1_COLLAPSED_STRIDE_MAP]
    # Allocation is 128B/stick * prod(device_size[:-1]) (get_device_size_in_bytes
    # in spyre_tensor_impl.cpp); the prepended dim grows it exactly 64x.
    alloc_bytes = 128 * math.prod(list(dl.device_size)[:-1])
    assert alloc_bytes == 64 * 8576 == 548864


def _size1_restickify_op_spec():
    """A restickify op_spec matching [7,67,1] (x+x).transpose(0,-1).contiguous():
    the INPUT stick is elided (within-stick coord is a constant), the OUTPUT is
    intact and carries the padding-prepended size-64 gap dim as its outermost dim.
    """
    import sympy
    from torch_spyre._C import DataFormats
    from torch_spyre._inductor.op_spec import OpSpec, TensorArg

    def tensor_arg(is_input, arg_index, device_size, coords):
        return TensorArg(
            is_input=is_input,
            arg_index=arg_index,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=list(device_size),
            device_coordinates=list(coords),
            allocation={"hbm": 0x400000000 + arg_index * 0x1000},
        )

    c = sympy.Symbol("c")
    # Elided INPUT: within-stick coordinate is a constant (no free symbol); the
    # old stick shows up as a size-1 dim.
    in_arg = tensor_arg(
        is_input=True,
        arg_index=0,
        device_size=[67, 1, 1, 64],
        coords=[c, sympy.Integer(0), sympy.Integer(0), sympy.Integer(0)],
    )
    # Intact OUTPUT: the grow prepended an outermost size-64, symbol-less gap dim;
    # the within-stick coord carries the free symbol.
    out_arg = tensor_arg(
        is_input=False,
        arg_index=1,
        device_size=[64, 67, 1, 64],
        coords=[sympy.Integer(0), c, sympy.Integer(0), sympy.Mod(c, 64)],
    )
    return OpSpec(
        op="restickify",
        is_reduction=False,
        iteration_space={c: (sympy.Integer(67), 1)},
        args=[in_arg, out_arg],
        op_info={},
    )


def test_size1_restore_reuses_prepended_dim():
    from torch_spyre._inductor.spyre_kernel import _restickify_restore_elided_dim

    op_spec = _size1_restickify_op_spec()
    out_arg = op_spec.args[1]

    _restickify_restore_elided_dim(op_spec)

    # The outermost size-64 dim is REUSED: the restore binds its shared symbol to
    # the existing dim rather than inserting a second one, so the intact operand
    # still has exactly ONE outermost size-64 dim (no 64x double count).
    assert list(out_arg.device_size) == [64, 67, 1, 64]
    assert out_arg.device_size.count(64) == 2  # gap dim + within-stick dim
    # The shared symbol (rs*) now drives that outermost dim ...
    outer_syms = tuple(out_arg.device_coordinates[0].free_symbols)
    assert len(outer_syms) == 1
    shared = outer_syms[0]
    assert shared.name.startswith("rs")
    # ... and it iterates with range 1, so it contributes no real stride: the
    # size-64 device slot is pure back-gap padding, not a 64-plane sweep.
    assert op_spec.iteration_space[shared] == (_SIZE1_STICK, 1)


def test_size1_restore_asserts_when_phantom_absent():
    from torch_spyre._inductor.spyre_kernel import _restickify_restore_elided_dim

    # If the padding grow did not run, the intact operand has no prepended
    # size-64 gap dim; the restore must hard-assert rather than silently insert a
    # second dim (which would double the descriptor total 64x vs the allocation).
    op_spec = _size1_restickify_op_spec()
    out_arg = op_spec.args[1]
    # Drop the prepended gap dim to simulate the un-grown STL.
    out_arg.device_size = [67, 1, 64]
    out_arg.device_coordinates = out_arg.device_coordinates[1:]

    with pytest.raises(AssertionError, match="padding-prepended"):
        _restickify_restore_elided_dim(op_spec)


def test_round_up_to_stick_geometry():
    """round_up_to_stick underlies the input-padding gate's read-window arithmetic
    (padded_dim_size = slice_offset + round_up_to_stick(extent)).  The gate itself
    is covered end-to-end by the OFFSET_STICK_OK / *_raises functional tests above;
    this pins the rounding helper it relies on.
    """
    from torch_spyre._inductor.padding import round_up_to_stick

    # No-op on a stick multiple; rounds a partial stick up to the next boundary.
    assert round_up_to_stick(0, torch.float16) == 0
    assert round_up_to_stick(64, torch.float16) == 64
    assert round_up_to_stick(3, torch.float16) == 64
    assert round_up_to_stick(70, torch.float16) == 128
    assert round_up_to_stick(128, torch.float16) == 128
    assert round_up_to_stick(129, torch.float16) == 192
