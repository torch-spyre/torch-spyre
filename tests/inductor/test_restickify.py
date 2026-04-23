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

import pytest
import torch

from utils_inductor import _compile_and_run, compare_with_cpu

DEVICE = torch.device("spyre")


def _compare(fn, *args, check_strides=True):
    spyre_result = _compile_and_run(fn, args, DEVICE)
    compare_with_cpu(fn, *args, target=spyre_result, run_eager=False)
    if check_strides:
        cpu_result = fn(*args)
        assert cpu_result.stride() == spyre_result.stride(), (
            f"Stride mismatch: CPU {cpu_result.stride()} vs Spyre {spyre_result.stride()}"
        )


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
    _compare(lambda a, x: a.t() + x, A, X)


def test_2arg_x_plus_at(tensors_2arg):
    A, _, X, _ = tensors_2arg
    _compare(lambda a, x: x + a.t(), A, X)


def test_2arg_xt_plus_a(tensors_2arg):
    A, _, X, _ = tensors_2arg
    _compare(lambda a, x: x.t() + a, A, X)


def test_2arg_a_plus_xt(tensors_2arg):
    A, _, X, _ = tensors_2arg
    _compare(lambda a, x: a + x.t(), A, X)


# 3-arg and 4-arg tests — run on a smaller set of size pairs
SIZES_2D_SMALL = [
    (256, 128),
]


@pytest.fixture(params=SIZES_2D_SMALL, ids=lambda p: f"{p[0]}x{p[1]}")
def tensors_multiarg(request):
    s1, s2 = request.param
    return _make_2d_tensors(s1, s2)


def test_3arg_at_bt_x(tensors_multiarg):
    A, B, X, _ = tensors_multiarg
    _compare(lambda a, b, x: a.t() + b.t() + x, A, B, X)


def test_3arg_at_x_bt(tensors_multiarg):
    A, B, X, _ = tensors_multiarg
    _compare(lambda a, b, x: a.t() + x + b.t(), A, B, X)


def test_3arg_x_at_bt(tensors_multiarg):
    A, B, X, _ = tensors_multiarg
    _compare(lambda a, b, x: x + a.t() + b.t(), A, B, X)


def test_3arg_at_x_y(tensors_multiarg):
    A, _, X, Y = tensors_multiarg
    _compare(lambda a, x, y: a.t() + x + y, A, X, Y)


def test_4arg_at_bt_x_y(tensors_multiarg):
    A, B, X, Y = tensors_multiarg
    _compare(lambda a, b, x, y: a.t() + b.t() + x + y, A, B, X, Y)


def test_4arg_at_x_bt_y(tensors_multiarg):
    A, B, X, Y = tensors_multiarg
    _compare(lambda a, b, x, y: a.t() + x + b.t() + y, A, B, X, Y)


def test_4arg_x_at_y_bt(tensors_multiarg):
    A, B, X, Y = tensors_multiarg
    _compare(lambda a, b, x, y: x + a.t() + y + b.t(), A, B, X, Y)


def test_4arg_at_x_y_bt(tensors_multiarg):
    A, B, X, Y = tensors_multiarg
    _compare(lambda a, b, x, y: a.t() + x + y + b.t(), A, B, X, Y)


def test_4arg_a_bt_c_d_square():
    s = 128
    A = torch.randn((s, s), dtype=torch.float16)
    B = torch.randn((s, s), dtype=torch.float16)
    C = torch.randn((s, s), dtype=torch.float16)
    D = torch.randn((s, s), dtype=torch.float16)
    _compare(lambda a, b, c, d: a + b.t() + c + d, A, B, C, D)


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


def test_matmul_xt_y(matmul_tensors_ab):
    x, y = matmul_tensors_ab
    _compare(lambda x, y: torch.matmul(x.t(), y), x, y)


def test_matmul_x_yt(matmul_tensors_ab):
    x, y = matmul_tensors_ab
    _compare(lambda x, y: torch.matmul(x, y.t()), x, y)


def test_matmul_xt_yt(matmul_tensors_ab_ba):
    x, y = matmul_tensors_ab_ba
    _compare(lambda x, y: torch.matmul(x.t(), y.t()), x, y)


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
