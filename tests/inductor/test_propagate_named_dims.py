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

# Tests for propagate_named_dims — the pass that annotates each op's output
# with semantic named dim labels (B, H, Lq, etc.) by propagating them from
# annotated graph inputs through the op graph.
#
# Each test compiles a small function, intercepts the pass via patching, and
# asserts that the graph output buffer carries the expected named dims.
# No coarse tiling hints are used — these tests cover propagation only.

import math

import pytest
import torch
from torch._inductor.ir import ComputedBuffer
from unittest.mock import patch

import torch_spyre._inductor.passes as _passes
import torch_spyre._inductor.propagate_named_dims as _pnd
from utils_inductor import _compile_and_run

DEVICE = torch.device("spyre")

# Dim sizes shared by the attention-shaped tests.
B, H, Lq, Lk, D = 12, 32, 256, 256, 128


# -------- Helper --------


def _run_and_capture(fn, args, declarations, annotations):
    """Declare dims, annotate device tensors, compile, return output named dims.

    args must be device tensors — the pass matches annotations via object identity
    against V.get_real_inputs(), which are the runtime device tensors.
    _compile_and_run's .to(device) is a no-op for already-on-device tensors.
    """
    for name, size in declarations.items():
        _pnd.declare_tensor_dim(name, size)
    for tensor, dims in annotations.items():
        _pnd.name_tensor_dims(tensor, dims)

    captured = {}
    real_fn = _passes.propagate_named_dims

    def capturing_propagate(graph):
        real_fn(graph)
        # _dim_prop_info is set by propagate_named_dims and deleted by
        # assign_dim_hints (the next pass). Capture the output dims here,
        # in the window between the two passes.
        output_names = set(graph.get_output_names())
        output_ops = [
            op
            for op in graph.operations
            if isinstance(op, ComputedBuffer)
            and op.get_name() in output_names
            and hasattr(op, "_dim_prop_info")
        ]
        if output_ops:
            captured["named_dims"] = list(output_ops[-1]._dim_prop_info.named_dims)

    with patch.object(_passes, "propagate_named_dims", capturing_propagate):
        _compile_and_run(fn, args, DEVICE)

    return captured.get("named_dims", [])


# -------- Basic 2-D tests --------

_M, _N = 64, 128


def test_2d_add():
    """2-D pointwise add of two [M,N] tensors; output dims match."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)
    y = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a, b):
        return a + b

    result = _run_and_capture(
        fn,
        [x, y],
        declarations={"M": _M, "N": _N},
        annotations={x: ["M", "N"], y: ["M", "N"]},
    )
    assert result == ["M", "N"], f"got {result}"


def test_2d_add_transposed():
    """2-D pointwise add where one input is transposed: [M,N] + [N,M].t()."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)
    y = torch.randn(_N, _M, dtype=torch.float16, device=DEVICE)

    def fn(a, b):
        return a + b.t()

    result = _run_and_capture(
        fn,
        [x, y],
        declarations={"M": _M, "N": _N},
        annotations={x: ["M", "N"], y: ["N", "M"]},
    )
    assert result == ["M", "N"], f"got {result}"


def test_2d_reduce_on_M():
    """2-D sum reduction over M: [M,N] -> [N]."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a):
        return a.sum(dim=0)

    result = _run_and_capture(
        fn,
        [x],
        declarations={"M": _M, "N": _N},
        annotations={x: ["M", "N"]},
    )
    assert result == ["N"], f"got {result}"


def test_2d_reduce_on_N():
    """2-D sum reduction over N: [M,N] -> [M]."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a):
        return a.sum(dim=1)

    result = _run_and_capture(
        fn,
        [x],
        declarations={"M": _M, "N": _N},
        annotations={x: ["M", "N"]},
    )
    assert result == ["M"], f"got {result}"


def test_2d_reduce_on_M_contiguous_before():
    """Contiguous before reduction: [M,N].contiguous().sum(dim=0) -> [N]."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a):
        return a.contiguous().sum(dim=0)

    result = _run_and_capture(
        fn,
        [x],
        declarations={"M": _M, "N": _N},
        annotations={x: ["M", "N"]},
    )
    assert result == ["N"], f"got {result}"


def test_2d_reduce_on_M_contiguous_after():
    """Contiguous after reduction: [M,N].sum(dim=0).contiguous() -> [N]."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a):
        return a.sum(dim=0).contiguous()

    result = _run_and_capture(
        fn,
        [x],
        declarations={"M": _M, "N": _N},
        annotations={x: ["M", "N"]},
    )
    assert result == ["N"], f"got {result}"


def test_2d_reduce_on_N_contiguous_before():
    """Contiguous before reduction: [M,N].contiguous().sum(dim=1) -> [M]."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a):
        return a.contiguous().sum(dim=1)

    result = _run_and_capture(
        fn,
        [x],
        declarations={"M": _M, "N": _N},
        annotations={x: ["M", "N"]},
    )
    assert result == ["M"], f"got {result}"


def test_2d_reduce_on_N_contiguous_after():
    """Contiguous after reduction: [M,N].sum(dim=1).contiguous() -> [M]."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a):
        return a.sum(dim=1).contiguous()

    result = _run_and_capture(
        fn,
        [x],
        declarations={"M": _M, "N": _N},
        annotations={x: ["M", "N"]},
    )
    assert result == ["M"], f"got {result}"


def test_2d_transposed_reduce_on_M():
    """Transpose [N,M] -> [M,N], then reduce over M: output [N]."""
    x = torch.randn(_N, _M, dtype=torch.float16, device=DEVICE)

    def fn(a):
        return a.t().sum(dim=0)

    result = _run_and_capture(
        fn,
        [x],
        declarations={"M": _M, "N": _N},
        annotations={x: ["N", "M"]},
    )
    assert result == ["N"], f"got {result}"


def test_2d_transposed_reduce_on_N():
    """Transpose [N,M] -> [M,N], then reduce over N: output [M]."""
    x = torch.randn(_N, _M, dtype=torch.float16, device=DEVICE)

    def fn(a):
        return a.t().sum(dim=1)

    result = _run_and_capture(
        fn,
        [x],
        declarations={"M": _M, "N": _N},
        annotations={x: ["N", "M"]},
    )
    assert result == ["M"], f"got {result}"


# -------- 4-D attention-shaped tests --------


def test_no_permute():
    """Baseline: input already in [B, H, Lq, D] order, output dims match."""
    queries = torch.randn(B, H, Lq, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q):
        return (q * scale).contiguous()

    result = _run_and_capture(
        fn,
        [queries],
        declarations={"B": B, "H": H, "Lq": Lq, "D": D},
        annotations={queries: ["B", "H", "Lq", "D"]},
    )
    assert result == ["B", "H", "Lq", "D"], f"got {result}"


def test_permute_then_contiguous():
    """Permuted input [B, Lq, H, D] -> permute(0,2,1,3) * scale -> contiguous."""
    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q):
        return (q.permute(0, 2, 1, 3) * scale).contiguous()

    result = _run_and_capture(
        fn,
        [queries],
        declarations={"B": B, "H": H, "Lq": Lq, "D": D},
        annotations={queries: ["B", "Lq", "H", "D"]},
    )
    assert result == ["B", "H", "Lq", "D"], f"got {result}"


def test_permute_no_contiguous():
    """Permuted input [B, Lq, H, D] -> permute(0,2,1,3) * scale, no contiguous."""
    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q, s):
        return q.permute(0, 2, 1, 3) * s

    result = _run_and_capture(
        fn,
        [queries, scale],
        declarations={"B": B, "H": H, "Lq": Lq, "D": D},
        annotations={queries: ["B", "Lq", "H", "D"]},
    )
    assert result == ["B", "H", "Lq", "D"], f"got {result}"


def test_contiguous_then_mul():
    """contiguous before multiply: permute -> contiguous -> * scale."""
    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q):
        return q.permute(0, 2, 1, 3).contiguous() * scale

    result = _run_and_capture(
        fn,
        [queries],
        declarations={"B": B, "H": H, "Lq": Lq, "D": D},
        annotations={queries: ["B", "Lq", "H", "D"]},
    )
    assert result == ["B", "H", "Lq", "D"], f"got {result}"


def test_permute_matmul():
    """Two permuted inputs into matmul; checks reduction dim propagation.

    queries [B, L, H, D] -> permute -> [B, H, L, D]
    keys    [B, L, H, D] -> permute -> [B, H, D, L]
    output  matmul -> [B, H, L, L]
    """
    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)
    keys = torch.randn(B, Lk, H, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q, k):
        q_perm = q.permute(0, 2, 1, 3) * scale
        k_perm = k.permute(0, 2, 3, 1) * scale
        return torch.matmul(q_perm, k_perm)

    result = _run_and_capture(
        fn,
        [queries, keys],
        declarations={"B": B, "H": H, "L": Lq, "D": D},
        annotations={
            queries: ["B", "L", "H", "D"],
            keys: ["B", "L", "H", "D"],
        },
    )
    assert result == ["B", "H", "L", "L"], f"got {result}"


def test_view_reshape_a():
    """View/reshape with shared name for equal-size dims (A==D, both called AD).

    Dims of size 2 share one name AD; all other dims are distinct.
    w, x: [1, AD, B*AD*E] annotated ["AD", "B", "AD", "E"]
    y:    [1, AD, C, AD, E] annotated ["AD", "C", "AD", "E"]
    z:    [1, AD, C, 1, 1, 1] annotated ["AD", "C"]
    output shape: [1, AD, C, B, AD, E]
    """
    ad, b, c, e = 2, 3, 4, 64
    w = torch.randn(1, ad, b * ad * e, dtype=torch.float16, device=DEVICE) * 0.1
    x = torch.randn(1, ad, b * ad * e, dtype=torch.float16, device=DEVICE) * 0.1
    y = torch.randn(1, ad, c, ad, e, dtype=torch.float16, device=DEVICE) * 0.1
    z = torch.randn(1, ad, c, 1, 1, 1, dtype=torch.float16, device=DEVICE) * 0.1

    def fn(w, x, y, z):
        t = w + x
        t = t.view(1, ad, b, ad, e)
        t = t.unsqueeze(2) + y.unsqueeze(3)
        return t + z

    result = _run_and_capture(
        fn,
        [w, x, y, z],
        declarations={"AD": ad, "B": b, "C": c, "E": e},
        annotations={
            w: ["AD", "B", "AD", "E"],
            x: ["AD", "B", "AD", "E"],
            y: ["AD", "C", "AD", "E"],
            z: ["AD", "C"],
        },
    )
    assert result == ["AD", "C", "B", "AD", "E"], f"got {result}"


def test_view_reshape_b():
    """View/reshape with all unique dim sizes.

    w, x: [1, A, B*D*E] annotated ["A", "B", "D", "E"]
    y:    [1, A, C, D, E] annotated ["A", "C", "D", "E"]
    z:    [1, A, C, 1, 1, 1] annotated ["A", "C"]
    output shape: [1, A, C, B, D, E]
    """
    a, b, c, d, e = 2, 3, 4, 5, 64
    w = torch.randn(1, a, b * d * e, dtype=torch.float16, device=DEVICE) * 0.1
    x = torch.randn(1, a, b * d * e, dtype=torch.float16, device=DEVICE) * 0.1
    y = torch.randn(1, a, c, d, e, dtype=torch.float16, device=DEVICE) * 0.1
    z = torch.randn(1, a, c, 1, 1, 1, dtype=torch.float16, device=DEVICE) * 0.1

    def fn(w, x, y, z):
        t = w + x
        t = t.view(1, a, b, d, e)
        t = t.unsqueeze(2) + y.unsqueeze(3)
        return t + z

    result = _run_and_capture(
        fn,
        [w, x, y, z],
        declarations={"A": a, "B": b, "C": c, "D": d, "E": e},
        annotations={
            w: ["A", "B", "D", "E"],
            x: ["A", "B", "D", "E"],
            y: ["A", "C", "D", "E"],
            z: ["A", "C"],
        },
    )
    assert result == ["A", "C", "B", "D", "E"], f"got {result}"


def test_permute_exp():
    """Permute + exp — no constant, split_multi_ops does not fire, passes cleanly.

    queries [B, Lq, H, D] -> permute(0,2,1,3) -> exp()
    Loop vars stay in input stride order; propagation works correctly.
    """
    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)

    def fn(q):
        return q.permute(0, 2, 1, 3).exp()

    result = _run_and_capture(
        fn,
        [queries],
        declarations={"B": B, "H": H, "Lq": Lq, "D": D},
        annotations={queries: ["B", "Lq", "H", "D"]},
    )
    assert result == ["B", "H", "Lq", "D"], f"got {result}"


def test_permute_matmul_distinct_lqlk():
    """Two permuted inputs into matmul with Lq != Lk; passes with distinct sizes.

    queries [B, Lq, H, D] -> permute -> [B, H, Lq, D]
    keys    [B, Lk, H, D] -> permute -> [B, H, D, Lk]
    output  matmul -> [B, H, Lq, Lk]
    """
    _B, _H, _Lq, _Lk, _D = 2, 4, 16, 32, 64
    queries = torch.randn(_B, _Lq, _H, _D, dtype=torch.float16, device=DEVICE)
    keys = torch.randn(_B, _Lk, _H, _D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(_D)

    def fn(q, k):
        q_perm = q.permute(0, 2, 1, 3) * scale
        k_perm = k.permute(0, 2, 3, 1) * scale
        return torch.matmul(q_perm, k_perm)

    result = _run_and_capture(
        fn,
        [queries, keys],
        declarations={"B": _B, "H": _H, "Lq": _Lq, "Lk": _Lk, "D": _D},
        annotations={
            queries: ["B", "Lq", "H", "D"],
            keys: ["B", "Lk", "H", "D"],
        },
    )
    assert result == ["B", "H", "Lq", "Lk"], f"got {result}"


def test_permute_mul_equal_dims():
    """Permute + constant multiply where the two middle dims share the same size.

    queries [B, L, L, D] -> permute(0,2,1,3) * 0.5 -> [B, L, L, D]
    """
    _B, _L, _D = 2, 16, 64
    queries = torch.randn(_B, _L, _L, _D, dtype=torch.float16, device=DEVICE)

    def fn(q):
        return q.permute(0, 2, 1, 3) * 0.5

    result = _run_and_capture(
        fn,
        [queries],
        declarations={"B": _B, "L": _L, "D": _D},
        annotations={queries: ["B", "L", "L", "D"]},
    )
    assert result == ["B", "L", "L", "D"], f"got {result}"


def test_broadcast_unsqueeze_mul():
    """Regression: broadcast intermediate must not use write-dep index substitution.

    amax forces c_reduced to materialize as a ComputedBuffer. Without the fix,
    zeroing the missing D sym inflates strides by D=128, mapping H to the wrong loop var.

    x is left unannotated to exercise the untracked fallback path.
    """
    x = torch.randn(B, H, Lq, D, dtype=torch.float16, device=DEVICE)
    c = torch.randn(B, H, Lk, D, dtype=torch.float16, device=DEVICE)

    def fn(a, c_full):
        c_reduced = c_full.amax(dim=-1)  # [B,H,Lk,D] -> [B,H,Lk]
        return c_reduced.unsqueeze(-1) * a

    result = _run_and_capture(
        fn,
        [x, c],
        declarations={"B": B, "H": H, "Lq": Lq, "Lk": Lk, "D": D},
        annotations={
            c: ["B", "H", "Lk", "D"],
        },
    )
    assert result == ["B", "H", "Lk", "_untracked_128"], f"got {result}"


# -------- Equal-size dims with distinct names --------


def test_permute_matmul_equal_lqlk_distinct_names():
    """Like test_permute_matmul but with distinct names Lq/Lk for the equal dims."""
    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)
    keys = torch.randn(B, Lk, H, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q, k):
        q_perm = q.permute(0, 2, 1, 3) * scale
        k_perm = k.permute(0, 2, 3, 1) * scale
        return torch.matmul(q_perm, k_perm)

    result = _run_and_capture(
        fn,
        [queries, keys],
        declarations={"B": B, "H": H, "Lq": Lq, "Lk": Lk, "D": D},
        annotations={
            queries: ["B", "Lq", "H", "D"],
            keys: ["B", "Lk", "H", "D"],
        },
    )
    assert result == ["B", "H", "Lq", "Lk"], f"got {result}"


def test_permuted_intermediate_then_reduce():
    """Permuted intermediate buffer fed into a reduction.

    q [B, Lq, H, D] -> (q * scale).permute(0,2,1,3) -> intermediate [B, H, Lq, D]
    -> sum over D -> output [B, H, Lq]

    Tests that compute_input_named_dims correctly handles a permuted ComputedBuffer
    as input to a Reduction op.
    """
    q = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q):
        return (q * scale).permute(0, 2, 1, 3).sum(dim=-1)

    result = _run_and_capture(
        fn,
        [q],
        declarations={"B": B, "H": H, "Lq": Lq, "D": D},
        annotations={q: ["B", "Lq", "H", "D"]},
    )
    assert result == ["B", "H", "Lq"], f"got {result}"


def test_permuted_intermediate_then_pointwise():
    """Permuted ComputedBuffer fed into a second Pointwise op (no contiguous between).

    q [B, Lq, H, D] -> (q * scale).permute(0,2,1,3) -> exp() -> intermediate [B,H,Lq,D]
    -> * bias -> output [B, H, Lq, D]

    The exp() intermediate is a ComputedBuffer with non-contiguous (permuted) strides.
    dep.index for that buffer has the same free symbols as write_dep.index but different
    strides. The missing_syms check correctly uses write_dep.index; the simpler == check
    would also use write_dep.index here. This test validates the permuted-intermediate
    case that originally motivated write_dep.index substitution.
    """
    q = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)
    bias = torch.randn(B, H, Lq, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q, bias):
        inter = (q * scale).permute(0, 2, 1, 3).exp()
        return inter * bias

    result = _run_and_capture(
        fn,
        [q, bias],
        declarations={"B": B, "H": H, "Lq": Lq, "D": D},
        annotations={q: ["B", "Lq", "H", "D"]},
    )
    assert result == ["B", "H", "Lq", "D"], f"got {result}"


def test_view_reshape_a_distinct_names():
    """Like test_view_reshape_a but with distinct names A/D for the equal-size dims."""
    a, b, c, d, e = 2, 3, 4, 2, 64
    w = torch.randn(1, a, b * d * e, dtype=torch.float16, device=DEVICE) * 0.1
    x = torch.randn(1, a, b * d * e, dtype=torch.float16, device=DEVICE) * 0.1
    y = torch.randn(1, a, c, d, e, dtype=torch.float16, device=DEVICE) * 0.1
    z = torch.randn(1, a, c, 1, 1, 1, dtype=torch.float16, device=DEVICE) * 0.1

    def fn(w, x, y, z):
        t = w + x
        t = t.view(1, a, b, d, e)
        t = t.unsqueeze(2) + y.unsqueeze(3)
        return t + z

    result = _run_and_capture(
        fn,
        [w, x, y, z],
        declarations={"A": a, "B": b, "C": c, "D": d, "E": e},
        annotations={
            w: ["A", "B", "D", "E"],
            x: ["A", "B", "D", "E"],
            y: ["A", "C", "D", "E"],
            z: ["A", "C"],
        },
    )
    assert result == ["A", "C", "B", "D", "E"], f"got {result}"


def test_view_reshape_then_reduce():
    """View/reshape intermediate fed into a Reduction.

    w, x: [1, A, B*D*E] -> add -> view(1, A, B, D, E) -> sum(dim=-1) -> [1, A, B, D]

    The view intermediate has fused dims in its layout (dep.index has multi-symbol
    coords). Tests that compute_input_named_dims handles this correctly for Reduction.
    """
    a, b, d, e = 2, 3, 5, 64
    w = torch.randn(1, a, b * d * e, dtype=torch.float16, device=DEVICE) * 0.1
    x = torch.randn(1, a, b * d * e, dtype=torch.float16, device=DEVICE) * 0.1

    def fn(w, x):
        t = w + x
        t = t.view(1, a, b, d, e)
        return t.sum(dim=-1)

    result = _run_and_capture(
        fn,
        [w, x],
        declarations={"A": a, "B": b, "D": d, "E": e},
        annotations={
            w: ["A", "B", "D", "E"],
            x: ["A", "B", "D", "E"],
        },
    )
    assert result == ["A", "B", "D"], f"got {result}"


def test_permute_mul_equal_dims_distinct_names():
    """Like test_permute_mul_equal_dims but with distinct names H/Lq for equal dims."""
    _B, _H, _Lq, _D = 2, 16, 16, 64
    queries = torch.randn(_B, _Lq, _H, _D, dtype=torch.float16, device=DEVICE)

    def fn(q):
        return q.permute(0, 2, 1, 3) * 0.5

    result = _run_and_capture(
        fn,
        [queries],
        declarations={"B": _B, "H": _H, "Lq": _Lq, "D": _D},
        annotations={queries: ["B", "Lq", "H", "D"]},
    )
    assert result == ["B", "H", "Lq", "D"], f"got {result}"


def test_reshape_1d_to_2d_exp():
    """1-D tensor [4096] annotated ['A'] -> reshape(64,64) raises Unsupported.

    A single named dim split by reshape cannot be propagated accurately.
    """
    _A = 4096
    x = torch.randn(_A, dtype=torch.float16, device=DEVICE)

    def fn(x):
        return x.reshape(64, 64).exp()

    with pytest.raises(Exception, match="reshape split a named dim"):
        _run_and_capture(
            fn,
            [x],
            declarations={"A": _A},
            annotations={x: ["A"]},
        )


# -------- Stride-0 broadcast (torch.expand) tests --------


def test_broadcast_expand_leading_dim():
    """Broadcast annotation not yet supported: [1,N] annotated ["M","N"] produces _untracked_.

    The size-1 leading dim is skipped; _consume_names(["M","N"], 128) fails because
    "M"=64 is at the front of remaining. Both loop vars fall back to _untracked_.
    See broadcast_named_dims_fix.txt for the full fix.
    """
    x = torch.randn(1, _N, dtype=torch.float16, device=DEVICE)
    y = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a, b):
        return a.expand(_M, _N) + b

    result = _run_and_capture(
        fn,
        [x, y],
        declarations={"M": _M, "N": _N},
        annotations={x: ["M", "N"]},
    )
    assert result == ["_untracked_64", "_untracked_128"], f"got {result}"


def test_broadcast_expand_trailing_dims():
    """Broadcast annotation not yet supported: [M,1] annotated ["M","N"] produces _untracked_.

    The M dim is assigned correctly; the trailing size-1 dim is skipped and "N"
    remains in remaining. Inductor elides the N loop var from dep.ranges entirely
    so it is never found. Falls back to _untracked_ for N.
    See broadcast_named_dims_fix.txt for the full fix.
    """
    x = torch.randn(_M, 1, dtype=torch.float16, device=DEVICE)
    y = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a, b):
        return a.expand(_M, _N) + b

    result = _run_and_capture(
        fn,
        [x, y],
        declarations={"M": _M, "N": _N},
        annotations={x: ["M", "N"]},
    )
    assert result == ["M", "_untracked_128"], f"got {result}"


def test_broadcast_expand_middle_dim():
    """Broadcast annotation not yet supported: [B,1,D] annotated ["B","H","D"] mis-maps.

    B is assigned correctly; the middle size-1 dim is skipped and "H" stays in
    remaining. _consume_names(["H","D"], D2=64) fails because "H"=32 is at the
    front. D falls back to _untracked_ as well.
    See broadcast_named_dims_fix.txt for the full fix.
    """
    _B2, _H2, _D2 = 4, 32, 64
    x = torch.randn(_B2, 1, _D2, dtype=torch.float16, device=DEVICE)
    y = torch.randn(_B2, _H2, _D2, dtype=torch.float16, device=DEVICE)

    def fn(a, b):
        return a.expand(_B2, _H2, _D2) + b

    result = _run_and_capture(
        fn,
        [x, y],
        declarations={"B2": _B2, "H2": _H2, "D2": _D2},
        annotations={x: ["B2", "H2", "D2"]},
    )
    assert result == ["B2", "_untracked_32", "_untracked_64"], f"got {result}"
