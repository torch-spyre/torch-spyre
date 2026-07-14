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

# Tests for propagate_named_dims and the dim_info mechanism.
#
# These tests verify that named dim labels (B, H, Lq, etc.) are correctly
# propagated. They do not exercise the Spyre backend — functions are compiled
# and the pass is intercepted via patching before any backend codegen runs.

import collections
import math

import pytest
import torch
from torch._inductor.ir import ComputedBuffer
from unittest.mock import patch

import torch_spyre._inductor.passes as _passes
import torch_spyre._inductor.propagate_named_dims as _pnd
from torch_spyre._inductor import spyre_hint as _spyre_hint
from utils_inductor import _compile_and_run

DEVICE = torch.device("spyre")

# loop_var symbols (d0, d1, ...) are assigned by Inductor outermost-first and
# are stable across compilations for a given function shape.  d0 is the outermost
# surviving loop variable for both pointwise and reduction ops.
# (Verified: test_2d_add and test_2d_reduce_on_M both assign d0 across 3 runs.)
_CaptureResult = collections.namedtuple(
    "_CaptureResult", ["propagated_dims", "dim_hints"]
)

# Dim sizes shared by the attention-shaped tests.
B, H, Lq, Lk, D = 12, 32, 256, 256, 128


# -------- Helper --------


def _run_and_capture(
    fn,
    args,
    named_dims,
    tensor_dims,
    *,
    expected_propagated_dims=None,
    expected_dim_hints=None,
):
    """Declare dims, annotate device tensors, compile, return _CaptureResult.

    named_dims: the output op's _dim_prop_info.named_dims (captured between
        propagate_named_dims and assign_dim_hints).
    dim_hints: the output op's dim_hints list (captured after assign_dim_hints).

    If expected_propagated_dims is provided, asserts result.propagated_dims matches.
    If expected_dim_hints is provided, asserts that the captured dim_hints match.
    Each dim_hints entry is a dict:
        {"loop_var": "d0", "dim_names": ["B"], "split_count": 4, "is_reduction": False}
    loop_var is the sympy Symbol rendered as a string, or None for broadcast ops.

    Only the final output op is captured — same scope as named_dims.
    """
    for name, size in named_dims.items():
        _pnd.declare_tensor_dim(name, size)
    for tensor, dims in tensor_dims.items():
        _pnd.name_tensor_dims(tensor, dims)

    captured = {}
    real_propagate = _passes.propagate_named_dims
    real_assign = _passes.assign_dim_hints

    def capturing_propagate(graph):
        real_propagate(graph)
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

    def capturing_assign(graph):
        real_assign(graph)
        output_names = set(graph.get_output_names())
        output_ops = [
            op
            for op in graph.operations
            if isinstance(op, ComputedBuffer)
            and op.get_name() in output_names
            and getattr(op, "dim_hints", None)
        ]
        if output_ops:
            captured["dim_hints"] = list(output_ops[-1].dim_hints)

    with (
        patch.object(_passes, "propagate_named_dims", capturing_propagate),
        patch.object(_passes, "assign_dim_hints", capturing_assign),
        patch("torch_spyre.execution.kernel_runner.prepare_kernel"),
        patch("torch_spyre.execution.kernel_runner.launch_jobplan"),
        patch("torch_spyre.execution.async_compile.subprocess.run"),
    ):
        _compile_and_run(fn, args, DEVICE)

    result = _CaptureResult(
        propagated_dims=captured.get("named_dims", []),
        dim_hints=captured.get("dim_hints", []),
    )

    if expected_propagated_dims is not None:
        assert result.propagated_dims == expected_propagated_dims, (
            f"named_dims mismatch:\n  got:      {result.propagated_dims}"
            f"\n  expected: {expected_propagated_dims}"
        )

    if expected_dim_hints is not None:
        actual = [
            {
                "loop_var": str(h.loop_var) if h.loop_var is not None else None,
                "dim_names": h.dim_names,
                "split_count": h.split_count,
                "is_reduction": h.is_reduction,
            }
            for h in result.dim_hints
        ]
        assert actual == expected_dim_hints, (
            f"dim_hints mismatch:\n  got:      {actual}\n  expected: {expected_dim_hints}"
        )

    return result


# -------- Basic 2-D tests --------

_M, _N = 128, 256


def test_2d_add():
    """2-D pointwise add of two [M,N] tensors; output dims match."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)
    y = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a, b):
        with _spyre_hint(num_tiles_per_dim={"M": 2}):
            return a + b

    _run_and_capture(
        fn,
        [x, y],
        named_dims={"M": _M, "N": _N},
        tensor_dims={x: ["M", "N"], y: ["M", "N"]},
        expected_dim_hints=[
            {
                "loop_var": "d0",
                "dim_names": ["M"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["M", "N"],
    )


def test_2d_add_transposed():
    """2-D pointwise add where one input is transposed: [M,N] + [N,M].t()."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)
    y = torch.randn(_N, _M, dtype=torch.float16, device=DEVICE)

    def fn(a, b):
        with _spyre_hint(num_tiles_per_dim={"M": 2}):
            return a + b.t()

    _run_and_capture(
        fn,
        [x, y],
        named_dims={"M": _M, "N": _N},
        tensor_dims={x: ["M", "N"], y: ["N", "M"]},
        expected_dim_hints=[
            {
                "loop_var": "d0",
                "dim_names": ["M"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["M", "N"],
    )


def test_2d_reduce_on_M():
    """2-D sum reduction over M: [M,N] -> [N]."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a):
        with _spyre_hint(num_tiles_per_dim={"N": 2}):
            return a.sum(dim=0)

    _run_and_capture(
        fn,
        [x],
        named_dims={"M": _M, "N": _N},
        tensor_dims={x: ["M", "N"]},
        expected_dim_hints=[
            {
                "loop_var": "d0",
                "dim_names": ["N"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["N"],
    )


def test_2d_reduce_on_N():
    """2-D sum reduction over N: [M,N] -> [M]."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a):
        with _spyre_hint(num_tiles_per_dim={"M": 2}):
            return a.sum(dim=1)

    _run_and_capture(
        fn,
        [x],
        named_dims={"M": _M, "N": _N},
        tensor_dims={x: ["M", "N"]},
        expected_dim_hints=[
            {
                "loop_var": "d0",
                "dim_names": ["M"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["M"],
    )


def test_2d_reduce_on_M_contiguous_before():
    """Contiguous before reduction: [M,N].contiguous().sum(dim=0) -> [N]."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a):
        with _spyre_hint(num_tiles_per_dim={"N": 2}):
            return a.contiguous().sum(dim=0)

    _run_and_capture(
        fn,
        [x],
        named_dims={"M": _M, "N": _N},
        tensor_dims={x: ["M", "N"]},
        expected_dim_hints=[
            {
                "loop_var": "d0",
                "dim_names": ["N"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["N"],
    )


def test_2d_reduce_on_M_contiguous_after():
    """Contiguous after reduction: [M,N].sum(dim=0).contiguous() -> [N]."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a):
        with _spyre_hint(num_tiles_per_dim={"N": 2}):
            return a.sum(dim=0).contiguous()

    _run_and_capture(
        fn,
        [x],
        named_dims={"M": _M, "N": _N},
        tensor_dims={x: ["M", "N"]},
        expected_dim_hints=[
            {
                "loop_var": "d0",
                "dim_names": ["N"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["N"],
    )


def test_2d_reduce_on_N_contiguous_before():
    """Contiguous before reduction: [M,N].contiguous().sum(dim=1) -> [M]."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a):
        with _spyre_hint(num_tiles_per_dim={"M": 2}):
            return a.contiguous().sum(dim=1)

    _run_and_capture(
        fn,
        [x],
        named_dims={"M": _M, "N": _N},
        tensor_dims={x: ["M", "N"]},
        expected_dim_hints=[
            {
                "loop_var": "d0",
                "dim_names": ["M"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["M"],
    )


def test_2d_reduce_on_N_contiguous_after():
    """Contiguous after reduction: [M,N].sum(dim=1).contiguous() -> [M]."""
    x = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a):
        with _spyre_hint(num_tiles_per_dim={"M": 2}):
            return a.sum(dim=1).contiguous()

    _run_and_capture(
        fn,
        [x],
        named_dims={"M": _M, "N": _N},
        tensor_dims={x: ["M", "N"]},
        expected_dim_hints=[
            {
                "loop_var": "d0",
                "dim_names": ["M"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["M"],
    )


def test_2d_transposed_reduce_on_M():
    """Transpose [N,M] -> [M,N], then reduce over M: output [N]."""
    x = torch.randn(_N, _M, dtype=torch.float16, device=DEVICE)

    def fn(a):
        with _spyre_hint(num_tiles_per_dim={"N": 2}):
            return a.t().sum(dim=0)

    _run_and_capture(
        fn,
        [x],
        named_dims={"M": _M, "N": _N},
        tensor_dims={x: ["N", "M"]},
        expected_dim_hints=[
            {
                "loop_var": "d0",
                "dim_names": ["N"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["N"],
    )


def test_2d_transposed_reduce_on_N():
    """Transpose [N,M] -> [M,N], then reduce over N: output [M]."""
    x = torch.randn(_N, _M, dtype=torch.float16, device=DEVICE)

    def fn(a):
        with _spyre_hint(num_tiles_per_dim={"M": 2}):
            return a.t().sum(dim=1)

    _run_and_capture(
        fn,
        [x],
        named_dims={"M": _M, "N": _N},
        tensor_dims={x: ["N", "M"]},
        expected_dim_hints=[
            {
                "loop_var": "d0",
                "dim_names": ["M"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["M"],
    )


# -------- 4-D attention-shaped tests --------


def test_no_permute():
    """Baseline: input already in [B, H, Lq, D] order, output dims match."""
    queries = torch.randn(B, H, Lq, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q):
        with _spyre_hint(num_tiles_per_dim={"Lq": 2}):
            return (q * scale).contiguous()

    _run_and_capture(
        fn,
        [queries],
        named_dims={"B": B, "H": H, "Lq": Lq, "D": D},
        tensor_dims={queries: ["B", "H", "Lq", "D"]},
        expected_dim_hints=[
            {
                "loop_var": "d2",
                "dim_names": ["Lq"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["B", "H", "Lq", "D"],
    )


def test_permute_then_contiguous():
    """Permuted input [B, Lq, H, D] -> permute(0,2,1,3) * scale -> contiguous."""
    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q):
        with _spyre_hint(num_tiles_per_dim={"Lq": 2}):
            return (q.permute(0, 2, 1, 3) * scale).contiguous()

    _run_and_capture(
        fn,
        [queries],
        named_dims={"B": B, "H": H, "Lq": Lq, "D": D},
        tensor_dims={queries: ["B", "Lq", "H", "D"]},
        expected_dim_hints=[
            {
                "loop_var": "d2",
                "dim_names": ["Lq"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["B", "H", "Lq", "D"],
    )


def test_permute_no_contiguous():
    """Permuted input [B, Lq, H, D] -> permute(0,2,1,3) * scale, no contiguous."""
    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q, s):
        with _spyre_hint(num_tiles_per_dim={"Lq": 2}):
            return q.permute(0, 2, 1, 3) * s

    _run_and_capture(
        fn,
        [queries, scale],
        named_dims={"B": B, "H": H, "Lq": Lq, "D": D},
        tensor_dims={queries: ["B", "Lq", "H", "D"]},
        expected_dim_hints=[
            {
                "loop_var": "d2",
                "dim_names": ["Lq"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["B", "H", "Lq", "D"],
    )


def test_contiguous_then_mul():
    """contiguous before multiply: permute -> contiguous -> * scale."""
    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q):
        with _spyre_hint(num_tiles_per_dim={"Lq": 2}):
            return q.permute(0, 2, 1, 3).contiguous() * scale

    _run_and_capture(
        fn,
        [queries],
        named_dims={"B": B, "H": H, "Lq": Lq, "D": D},
        tensor_dims={queries: ["B", "Lq", "H", "D"]},
        expected_dim_hints=[
            {
                "loop_var": "d2",
                "dim_names": ["Lq"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["B", "H", "Lq", "D"],
    )


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
        with _spyre_hint(num_tiles_per_dim={"L": 2}):
            q_perm = q.permute(0, 2, 1, 3) * scale
            k_perm = k.permute(0, 2, 3, 1) * scale
            return torch.matmul(q_perm, k_perm)

    _run_and_capture(
        fn,
        [queries, keys],
        named_dims={"B": B, "H": H, "L": Lq, "D": D},
        tensor_dims={
            queries: ["B", "L", "H", "D"],
            keys: ["B", "L", "H", "D"],
        },
        expected_dim_hints=[
            {
                "loop_var": "d3",
                "dim_names": ["L"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["B", "H", "L", "L"],
    )


def test_view_reshape_a():
    """View/reshape with shared name for equal-size dims (A==D, both called AD).

    No expected_dim_hints: b=3 creates a fractional coordinate expression
    (2*c0 + 2*c2/3) in views.py that the coordinate normalizer does not support,
    so any hint triggers an InductorError during compilation.

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

    _run_and_capture(
        fn,
        [w, x, y, z],
        named_dims={"AD": ad, "B": b, "C": c, "E": e},
        tensor_dims={
            w: ["AD", "B", "AD", "E"],
            x: ["AD", "B", "AD", "E"],
            y: ["AD", "C", "AD", "E"],
            z: ["AD", "C"],
        },
        expected_propagated_dims=["AD", "C", "B", "AD", "E"],
    )


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

    _run_and_capture(
        fn,
        [w, x, y, z],
        named_dims={"A": a, "B": b, "C": c, "D": d, "E": e},
        tensor_dims={
            w: ["A", "B", "D", "E"],
            x: ["A", "B", "D", "E"],
            y: ["A", "C", "D", "E"],
            z: ["A", "C"],
        },
        # no dim_hints: 4-input fusion hits the 5-tensor bundle limit with a hint
        expected_propagated_dims=["A", "C", "B", "D", "E"],
    )


def test_permute_exp():
    """Permute + exp — no constant, split_multi_ops does not fire, passes cleanly.

    queries [B, Lq, H, D] -> permute(0,2,1,3) -> exp()
    Loop vars stay in input stride order; propagation works correctly.
    """
    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)

    def fn(q):
        with _spyre_hint(num_tiles_per_dim={"Lq": 2}):
            return q.permute(0, 2, 1, 3).exp()

    _run_and_capture(
        fn,
        [queries],
        named_dims={"B": B, "H": H, "Lq": Lq, "D": D},
        tensor_dims={queries: ["B", "Lq", "H", "D"]},
        expected_dim_hints=[
            {
                "loop_var": "d2",
                "dim_names": ["Lq"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["B", "H", "Lq", "D"],
    )


def test_permute_matmul_distinct_lqlk():
    """Two permuted inputs into matmul with Lq != Lk; passes with distinct sizes.

    No expected_dim_hints: all dims are too small to tile safely
    (largest is Lk=32; split_count=2 gives tile size 16, below the 64-element minimum).

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

    _run_and_capture(
        fn,
        [queries, keys],
        named_dims={"B": _B, "H": _H, "Lq": _Lq, "Lk": _Lk, "D": _D},
        tensor_dims={
            queries: ["B", "Lq", "H", "D"],
            keys: ["B", "Lk", "H", "D"],
        },
        expected_propagated_dims=["B", "H", "Lq", "Lk"],
    )


def test_permute_mul_equal_dims():
    """Permute + constant multiply where the two middle dims share the same size.

    No expected_dim_hints: all dims are too small to tile safely
    (L=16, D=64; split_count=2 gives tile sizes 8 and 32, both below the 64-element minimum).

    queries [B, L, L, D] -> permute(0,2,1,3) * 0.5 -> [B, L, L, D]
    """
    _B, _L, _D = 2, 16, 64
    queries = torch.randn(_B, _L, _L, _D, dtype=torch.float16, device=DEVICE)

    def fn(q):
        return q.permute(0, 2, 1, 3) * 0.5

    _run_and_capture(
        fn,
        [queries],
        named_dims={"B": _B, "L": _L, "D": _D},
        tensor_dims={queries: ["B", "L", "L", "D"]},
        expected_propagated_dims=["B", "L", "L", "D"],
    )


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
        with _spyre_hint(num_tiles_per_dim={"Lk": 2}):
            return c_reduced.unsqueeze(-1) * a

    _run_and_capture(
        fn,
        [x, c],
        named_dims={"B": B, "H": H, "Lq": Lq, "Lk": Lk, "D": D},
        tensor_dims={
            c: ["B", "H", "Lk", "D"],
        },
        expected_dim_hints=[
            {
                "loop_var": "d2",
                "dim_names": ["Lk"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["B", "H", "Lk", "_untracked_128"],
    )


# -------- Equal-size dims with distinct names --------


def test_permute_matmul_equal_lqlk_distinct_names():
    """Like test_permute_matmul but with distinct names Lq/Lk for the equal dims."""
    queries = torch.randn(B, Lq, H, D, dtype=torch.float16, device=DEVICE)
    keys = torch.randn(B, Lk, H, D, dtype=torch.float16, device=DEVICE)
    scale = 1.0 / math.sqrt(D)

    def fn(q, k):
        q_perm = q.permute(0, 2, 1, 3) * scale
        k_perm = k.permute(0, 2, 3, 1) * scale
        with _spyre_hint(num_tiles_per_dim={"Lq": 2}):
            return torch.matmul(q_perm, k_perm)

    _run_and_capture(
        fn,
        [queries, keys],
        named_dims={"B": B, "H": H, "Lq": Lq, "Lk": Lk, "D": D},
        tensor_dims={
            queries: ["B", "Lq", "H", "D"],
            keys: ["B", "Lk", "H", "D"],
        },
        expected_dim_hints=[
            {
                "loop_var": "d2",
                "dim_names": ["Lq"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["B", "H", "Lq", "Lk"],
    )


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
        with _spyre_hint(num_tiles_per_dim={"Lq": 2}):
            return (q * scale).permute(0, 2, 1, 3).sum(dim=-1)

    _run_and_capture(
        fn,
        [q],
        named_dims={"B": B, "H": H, "Lq": Lq, "D": D},
        tensor_dims={q: ["B", "Lq", "H", "D"]},
        expected_dim_hints=[
            {
                "loop_var": "d2",
                "dim_names": ["Lq"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["B", "H", "Lq"],
    )


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
        with _spyre_hint(num_tiles_per_dim={"Lq": 2}):
            inter = (q * scale).permute(0, 2, 1, 3).exp()
            return inter * bias

    _run_and_capture(
        fn,
        [q, bias],
        named_dims={"B": B, "H": H, "Lq": Lq, "D": D},
        tensor_dims={q: ["B", "Lq", "H", "D"]},
        expected_dim_hints=[
            {
                "loop_var": "d2",
                "dim_names": ["Lq"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["B", "H", "Lq", "D"],
    )


def test_view_reshape_a_distinct_names():
    """Like test_view_reshape_a but with distinct names A/D for the equal-size dims.

    No expected_dim_hints: same b=3 fractional coordinate limitation as test_view_reshape_a.
    """
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

    _run_and_capture(
        fn,
        [w, x, y, z],
        named_dims={"A": a, "B": b, "C": c, "D": d, "E": e},
        tensor_dims={
            w: ["A", "B", "D", "E"],
            x: ["A", "B", "D", "E"],
            y: ["A", "C", "D", "E"],
            z: ["A", "C"],
        },
        expected_propagated_dims=["A", "C", "B", "D", "E"],
    )


def test_view_reshape_then_reduce():
    """View/reshape intermediate fed into a Reduction.

    No expected_dim_hints: same b=3 fractional coordinate limitation as test_view_reshape_a.

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

    _run_and_capture(
        fn,
        [w, x],
        named_dims={"A": a, "B": b, "D": d, "E": e},
        tensor_dims={
            w: ["A", "B", "D", "E"],
            x: ["A", "B", "D", "E"],
        },
        expected_propagated_dims=["A", "B", "D"],
    )


def test_permute_mul_equal_dims_distinct_names():
    """Like test_permute_mul_equal_dims but with distinct names H/Lq for equal dims.

    No expected_dim_hints: same small-dim limitation as test_permute_mul_equal_dims.
    """
    _B, _H, _Lq, _D = 2, 16, 16, 64
    queries = torch.randn(_B, _Lq, _H, _D, dtype=torch.float16, device=DEVICE)

    def fn(q):
        return q.permute(0, 2, 1, 3) * 0.5

    _run_and_capture(
        fn,
        [queries],
        named_dims={"B": _B, "H": _H, "Lq": _Lq, "D": _D},
        tensor_dims={queries: ["B", "Lq", "H", "D"]},
        expected_propagated_dims=["B", "H", "Lq", "D"],
    )


def test_reshape_split_untracked_dim_tolerated():
    """Reshape that splits an _untracked_ intermediate dim is tolerated, not raised.

    Mirrors a k/v projection reshaped into heads: an *unannotated* input produces
    an intermediate whose dims carry only _untracked_ placeholder names (no
    meaningful name to preserve).  A view then splits that dim into two loop
    vars, hitting the len(loop_vars) > len(names) branch in
    compute_input_named_dims.  Because the split name is only _untracked_, the
    pass assigns fresh untracked names per loop var and keeps going instead of
    aborting like test_reshape_1d_to_2d_exp (which splits a *real* named dim).

    Contrast with test_reshape_1d_to_2d_exp: that splits the meaningful name "A"
    and raises "reshape split a named dim"; here nothing meaningful is split, so
    compilation succeeds and the output dims are all _untracked_.

    x is left unannotated (no tensor_dims entry) so its consuming op is assigned
    _untracked_ names; y anchors a valid named_dims declaration for the run.
    """
    _T, _Hh, _Dd = 8, 4, 64  # HD = 256 splits into H=4, D=64
    x = torch.randn(_T, _Hh * _Dd, dtype=torch.float16, device=DEVICE) * 0.1
    y = torch.randn(_T, _Hh, _Dd, dtype=torch.float16, device=DEVICE) * 0.1

    def fn(x, y):
        # x.exp() forces an intermediate ComputedBuffer with _untracked_ names,
        # then the view splits its untracked [HD] dim into [H, D].
        t = x.exp().view(_T, _Hh, _Dd)
        return t + y

    # The assertion is simply that compilation does not raise: without the
    # _untracked_ tolerance branch, the split of x.exp()'s untracked dim would
    # hit "reshape split a named dim" and abort.  We deliberately do not pin the
    # output op's propagated_dims -- the final op reads annotated y, so its names
    # come from y, not from the untracked split path under test; asserting them
    # would test the wrong op.  Reaching this line means the branch tolerated
    # the split.
    _run_and_capture(
        fn,
        [x, y],
        named_dims={"T": _T, "H": _Hh, "D": _Dd},
        tensor_dims={y: ["T", "H", "D"]},
    )


def test_reshape_1d_to_2d_exp():
    """1-D tensor [4096] annotated ['A'] -> reshape(64,64) raises Unsupported.

    A single named dim split by reshape cannot be propagated accurately.
    No expected_dim_hints: compilation raises before assign_dim_hints runs.
    """
    _A = 4096
    x = torch.randn(_A, dtype=torch.float16, device=DEVICE)

    def fn(x):
        return x.reshape(64, 64).exp()

    with pytest.raises(Exception, match="reshape split a named dim"):
        _run_and_capture(
            fn,
            [x],
            named_dims={"A": _A},
            tensor_dims={x: ["A"]},
        )


# -------- Stride-0 broadcast (torch.expand) tests --------


def test_broadcast_expand_leading_dim():
    """Broadcast annotation not yet supported: [1,N] annotated ["M","N"] produces _untracked_.

    The size-1 leading dim is skipped; _consume_names(["M","N"], 256) fails because
    "M"=128 is at the front of remaining. Both loop vars fall back to _untracked_.
    See broadcast_named_dims_fix.txt for the full fix.

    loop_var is None because the broadcast path cannot identify the output loop var.
    """
    x = torch.randn(1, _N, dtype=torch.float16, device=DEVICE)
    y = torch.randn(_M, _N, dtype=torch.float16, device=DEVICE)

    def fn(a, b):
        with _spyre_hint(num_tiles_per_dim={"M": 2}):
            return a.expand(_M, _N) + b

    _run_and_capture(
        fn,
        [x, y],
        named_dims={"M": _M, "N": _N},
        tensor_dims={x: ["M", "N"]},
        expected_dim_hints=[
            {
                "loop_var": None,
                "dim_names": ["M"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["_untracked_128", "_untracked_256"],
    )


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
        with _spyre_hint(num_tiles_per_dim={"M": 2}):
            return a.expand(_M, _N) + b

    _run_and_capture(
        fn,
        [x, y],
        named_dims={"M": _M, "N": _N},
        tensor_dims={x: ["M", "N"]},
        expected_dim_hints=[
            {
                "loop_var": "d0",
                "dim_names": ["M"],
                "split_count": 2,
                "is_reduction": False,
            },
        ],
        expected_propagated_dims=["M", "_untracked_256"],
    )


def test_broadcast_expand_middle_dim():
    """Broadcast annotation not yet supported: [B,1,D] annotated ["B","H","D"] mis-maps.

    No expected_dim_hints: all dims too small to tile safely (B=4, H=32, D=64;
    split_count=2 gives tile sizes 2, 16, and 32, all below the 64-element minimum).

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

    _run_and_capture(
        fn,
        [x, y],
        named_dims={"B2": _B2, "H2": _H2, "D2": _D2},
        tensor_dims={x: ["B2", "H2", "D2"]},
        expected_propagated_dims=["B2", "_untracked_32", "_untracked_64"],
    )


# -------- Indirect-access (gather) tests --------

_GM, _GN, _GP = 128, 256, 32
_GQ = 192
_GA, _GB, _GC = 64, 8, 64


def test_gather_advanced_indexing_2d():
    """x[i]: x[M,N] gathered by i[P]. The gathered M dim is addressed by an
    indirect index (0 loop vars); the pass must not raise Unsupported."""
    x = torch.randn(_GM, _GN, dtype=torch.float16, device=DEVICE)
    i = torch.randint(0, _GM, (_GP,), dtype=torch.int32, device=DEVICE)

    def fn(x, i):
        return x[i]

    _run_and_capture(
        fn,
        [x, i],
        named_dims={"M": _GM, "N": _GN, "P": _GP},
        tensor_dims={x: ["M", "N"], i: ["P"]},
        expected_propagated_dims=["P", "N"],
    )


def test_gather_advanced_indexing_with_exp():
    """x[i].exp(): a unary fused onto the gather still drives the gather's input
    read through compute_input_named_dims; must not raise."""
    x = torch.randn(_GM, _GN, dtype=torch.float16, device=DEVICE)
    i = torch.randint(0, _GM, (_GP,), dtype=torch.int32, device=DEVICE)

    def fn(x, i):
        return x[i].exp()

    _run_and_capture(
        fn,
        [x, i],
        named_dims={"M": _GM, "N": _GN, "P": _GP},
        tensor_dims={x: ["M", "N"], i: ["P"]},
        expected_propagated_dims=["P", "N"],
    )


def test_gather_3d_data():
    """x[i] with 3-D data x[A,B,C] gathered by i[P]: the gathered A dim is
    index-selected (0 loop vars); the inner B and C dims propagate."""
    x = torch.randn(_GA, _GB, _GC, dtype=torch.float16, device=DEVICE)
    i = torch.randint(0, _GA, (_GP,), dtype=torch.int32, device=DEVICE)

    def fn(x, i):
        return x[i]

    _run_and_capture(
        fn,
        [x, i],
        named_dims={"A": _GA, "B": _GB, "C": _GC, "P": _GP},
        tensor_dims={x: ["A", "B", "C"], i: ["P"]},
        expected_propagated_dims=["P", "B", "C"],
    )


def test_index_select_2d():
    """torch.index_select(x, 0, i): same indirect read as x[i]; must not raise."""
    x = torch.randn(_GM, _GN, dtype=torch.float16, device=DEVICE)
    i = torch.randint(0, _GM, (_GP,), dtype=torch.int32, device=DEVICE)

    def fn(x, i):
        return torch.index_select(x, 0, i)

    _run_and_capture(
        fn,
        [x, i],
        named_dims={"M": _GM, "N": _GN, "P": _GP},
        tensor_dims={x: ["M", "N"], i: ["P"]},
        expected_propagated_dims=["P", "N"],
    )


def test_gather_2d_index():
    """x[i]: x[M,N] gathered by 2-D index i[P,Q]. Output is [P,Q,N]; the
    gathered M dim has a 2-D indirect index (0 loop vars); inner N propagates."""
    x = torch.randn(_GM, _GN, dtype=torch.float16, device=DEVICE)
    i = torch.randint(0, _GM, (_GP, _GQ), dtype=torch.int32, device=DEVICE)

    def fn(x, i):
        return x[i]

    _run_and_capture(
        fn,
        [x, i],
        named_dims={"M": _GM, "N": _GN, "P": _GP, "Q": _GQ},
        tensor_dims={x: ["M", "N"], i: ["P", "Q"]},
        expected_propagated_dims=["P", "Q", "N"],
    )
