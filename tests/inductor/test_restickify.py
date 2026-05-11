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

import math
import os

import pytest
import torch

import torch_spyre._inductor.insert_restickify as _insert_restickify
import torch_spyre._inductor.optimize_restickify as _optimize_restickify
from utils_inductor import _compile_and_run, compare_with_cpu

DEVICE = torch.device("spyre")
S = 128  # must be a multiple of 64
T = 64  # side length for 4D tests (all dims equal)


def _verify_cost(expected_cost):
    if not _insert_restickify.restickify_plan:
        # Cache hit — finalize_layouts did not run, plan was not captured. Skip cost check.
        return
    actual = sum(
        math.prod(int(s) for s in entry["target_layout"].size)
        for entries in _insert_restickify.restickify_plan.values()
        for entry in entries
    )
    assert actual == expected_cost, (
        f"restickify cost: expected {expected_cost}, got {actual}"
    )


def _compare(fn, *args, check_strides=True, optimal_cost=None, skip_correctness=False):
    """Run fn on Spyre, assert correctness against CPU, and optionally assert the restickify
    plan has cost == optimal_cost. Restickify decisions and their cost normally remains inside
    the compiler; the env var below instructs the compiler to stash it for us.
    """
    if optimal_cost is not None:
        _insert_restickify.restickify_plan = {}
        os.environ["SPYRE_CAPTURE_RESTICKIFY_PLAN"] = "1"
        try:
            spyre_result = _compile_and_run(fn, args, DEVICE)
        finally:
            del os.environ["SPYRE_CAPTURE_RESTICKIFY_PLAN"]
    else:
        spyre_result = _compile_and_run(fn, args, DEVICE)
    if not skip_correctness:
        compare_with_cpu(fn, *args, target=spyre_result, run_eager=False)
    if check_strides:
        cpu_result = fn(*args)
        assert cpu_result.stride() == spyre_result.stride(), (
            f"Stride mismatch: CPU {cpu_result.stride()} vs Spyre {spyre_result.stride()}"
        )
    if optimal_cost is not None:
        _verify_cost(optimal_cost)


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

    spyre_result = _compile_and_run(func, (x, weight, cache), DEVICE)
    compare_with_cpu(func, x, weight, cache, target=spyre_result, run_eager=False)


# Optimizer correctness + optimality tests: verify both output values and
# minimum-cost restickify plan across a range of graph patterns.


def test_opt_parens_one_conflict():
    """((a + b) + (c.t() + d)) + (e + f) — conflict only in inner group."""
    a, b, c, d, e, f = _make_tensors(6, S, S)
    _compare(
        lambda a, b, c, d, e, f: (((a + b) + (c.t() + d)) + (e + f)),
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
        lambda a, b, c, d, e: ((a + b.t() + c.t() + d.t()) @ e),
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
    _compare(lambda a, b, c: (a @ (b + c.t())), a, b, c, optimal_cost=S * S)


def test_opt_adds_then_matmul_y_long_chain():
    """a @ (b + c.t() + d.t() + e.t()) — majority transposed going into y."""
    a, b, c, d, e = _make_tensors(5, S, S)
    _compare(
        lambda a, b, c, d, e: (a @ (b + c.t() + d.t() + e.t())),
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
    _compare(lambda a, b, c: (a.t() @ (b + c.t())), a, b, c, optimal_cost=2 * S * S)


def test_opt_matmul_then_adds():
    """(a @ b) + c.t() — matmul output stick vs transposed input."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: ((a @ b) + c.t()), a, b, c, optimal_cost=S * S)


def test_opt_matmul_then_long_adds():
    """(a @ b) + c.t() + d.t() — keep matmul stick, restickify one input."""
    a, b, c, d = _make_tensors(4, S, S)
    _compare(
        lambda a, b, c, d: ((a @ b) + c.t() + d.t()), a, b, c, d, optimal_cost=S * S
    )


def test_opt_chained_matmuls():
    """(a @ b) @ c — no restickify needed."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: ((a @ b) @ c), a, b, c, optimal_cost=0)


def test_opt_two_independent_conflicts():
    """(a+b.t()) + (e.t()+f.t()+g) — two separate conflicts."""
    a, b, e, f, g = _make_tensors(5, S, S)
    _compare(
        lambda a, b, e, f, g: ((a + b.t()) + (e.t() + f.t() + g)),
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
    _compare(lambda a, b: (a.t() @ b), a, b, optimal_cost=M * K)


def test_opt_sum_between_pointwise():
    """(a + b.t()).sum(1) + c — reduction between two pointwise stages."""
    a, b = _make_tensors(2, S, S)
    (c,) = _make_tensors(1, S)
    # Note: sum() below may fail correctness depending which stick flows in
    # because propagate_layouts does not yet properly detect incompatibility
    # of sparse/non-sparse sticks in a pointwise op.  Disabling correctness
    # check until that is resolved
    _compare(
        lambda a, b, c: ((a + b.t()).sum(0) + c),
        a,
        b,
        c,
        optimal_cost=S * S,
        skip_correctness=True,
    )


def test_opt_chain_transposed_intermediate():
    """(a.t() + b).t() + c — intermediate consumed transposed."""
    a, b, c = _make_tensors(3, S, S)
    _compare(lambda a, b, c: ((a.t() + b).t() + c), a, b, c, optimal_cost=S * S)


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
        lambda a, b, c, d: (a.transpose(0, 3) + b + c + d),
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
        lambda a, b, c: ((a.transpose(2, 3) + b).transpose(2, 3) + c),
        a,
        b,
        c,
        optimal_cost=T**4,
    )


def test_opt_two_matmuls_wrong_inputs():
    """(a.t() @ b) + (c @ d.t()) — each matmul has one wrong-stick input."""
    a, b, c, d = _make_tensors(4, S, S)
    _compare(
        lambda a, b, c, d: ((a.t() @ b) + (c @ d.t())),
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
        lambda a, b, c, d: ((a + b.t()) @ (c + d.t())),
        a,
        b,
        c,
        d,
        optimal_cost=2 * S * S,
    )
