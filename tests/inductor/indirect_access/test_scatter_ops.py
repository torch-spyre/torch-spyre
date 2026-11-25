# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""Scatter-family and related indirect writes: scatter / scatter_add /
scatter_reduce, index_reduce, index_fill, masked_fill, and put.

Multi-sencores is compiled-only: dests whose indexed dim can split across cores
request patch_sencores (eager once at default config; compiled at 1 and chip-max).
Small dests stay eager + compiled at default config. Eager is never a SENCORES variant.
"""

import pytest
import torch

from utils_inductor import DEVICE, cached_randn
from conftest import (
    gap_one_core,
    _xfail_existing,
    compare_mode,
    expect_dynamic_true_refused,
    rejection,
)

# fp16 compute after a gather (unary, binary, small reductions).  The largest error
# measured against an fp32 reference is 5.1e-3 (fp16's own compute error: up to 1.7e-3),
# so 1e-2 leaves about 2x margin.  Pure data movement uses atol=0 instead.
_TOL_F16 = 1e-2
# fp32 compute.  The largest error measured against an fp32 CPU reference is 4.3e-7
# (layer norm).
_TOL_F32 = 1e-5


_MASKED_FILL_FP32 = rejection(
    4714,
    "masked_fill fp32: the bool mask cast to fp32 gives a staggered EA whose shuffle "
    "needs whole stick pairs; the stick dim of 32 is not a whole pair.",
    r"requires whole stick pairs",
)
_IR_DUP_EAGER = (
    6638,
    "index_reduce_ with a repeated index returns wrong values: the decomposition "
    "overwrites, so only one source row per destination row is kept (unique indices "
    "are a documented precondition).",
    r"eager spyre <-> cpu\S* mismatch\s+Tensor-likes are not (?:close|equal)!",
)
_IR_DUP_COMPILED = (
    6638,
    "compile index_reduce_ with a repeated index returns wrong values: the "
    "decomposition overwrites, so only one source row per destination row is kept "
    "(unique indices are a documented precondition).",
    r"compiled spyre <-> cpu\S* mismatch\s+Tensor-likes are not (?:close|equal)!",
)
_IR_STICK_DIM = rejection(
    4404,
    "index_reduce along the stick (last) dim is rejected: its gather/scatter along the stick does not lower.",
    r"index_reduce along the stick \(last\) dim",
)
_PUT_EAGER = rejection(
    4540,
    "aten::put_ is not registered on Spyre.",
    r"Could not run 'aten::put_'",
)
_PUT_COMPILED = rejection(
    4636,
    "compile put_: AssertionError restickify op_spec has 3 args.",
    r"restickify op_spec has 3 args",
)
_MUTATING_FALLBACK = r"upstream lowers this form to a mutating fallback"


def _scatter_fallback(issue, what):
    """Compiled scatter forms that upstream lowers to scatter_reduce_.two are rejected."""
    return rejection(
        issue,
        f"compile {what} lowers to aten.scatter_reduce_.two, a mutating fallback with no Spyre kernel.",
        _MUTATING_FALLBACK,
    )


_SR_TWO = rejection(
    4361,
    "aten::scatter_reduce.two_out is not registered on Spyre.",
    r"Could not run 'aten::scatter_reduce.two_out'",
)


def _router_index(T, E, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.empty(T, K, dtype=torch.int64)
    for t in range(T):
        idx[t] = torch.randperm(E, generator=g)[:K]
    return idx


# ---------------------------------------------------------------------------
# index_reduce
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestIndexReduceEagerCompile:
    """index_reduce_ standalone: row-level indexed reduction on dim=0 (mean/prod/amax/amin)."""

    def setup_method(self):
        torch.manual_seed(0xBEEF)

    def _xfail_duplicates(self, execution_mode):
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/6638
        _xfail_existing(execution_mode, eager=_IR_DUP_EAGER, compiled=_IR_DUP_COMPILED)

    def _run(self, execution_mode, dst, src, idx, reduce_op, tol):
        def fn(dst, src, idx):
            return dst.clone().index_reduce_(
                0, idx, src, reduce=reduce_op, include_self=False
            )

        compare_mode(execution_mode, fn, dst, src, idx, atol=tol, rtol=tol)

    def test_index_reduce_mean_dim0(self, execution_mode):
        """index_reduce_ mean on dim=0; duplicate indices average rows into destination.

        PyTorch index_reduce_ does not accept reduce='sum' (use scatter_add for sum).
        The index repeats row 2, and the decomposition overwrites, so the result is wrong
        in eager and compiled (#6638).
        """
        self._xfail_duplicates(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_mean01", dtype=torch.float16)
        src = cached_randn(
            (3, 8), differentiation="irdc_mean01_src", dtype=torch.float16
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "mean", _TOL_F16)

    def test_index_reduce_prod_dim0(self, execution_mode):
        """index_reduce_ prod on dim=0; duplicate indices multiply rows into destination."""
        self._xfail_duplicates(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_prod01", dtype=torch.float16)
        src = cached_randn(
            (3, 8), differentiation="irdc_prod01_src", dtype=torch.float16
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "prod", _TOL_F16)

    def test_index_reduce_amax_dim0(self, execution_mode):
        """index_reduce_ amax on dim=0; duplicate indices take element-wise max across rows."""
        self._xfail_duplicates(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_amax01", dtype=torch.float16)
        src = cached_randn(
            (3, 8), differentiation="irdc_amax01_src", dtype=torch.float16
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "amax", _TOL_F16)

    def test_index_reduce_amin_dim0(self, execution_mode):
        """index_reduce_ amin on dim=0; duplicate indices take element-wise min across rows."""
        self._xfail_duplicates(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_amin01", dtype=torch.float16)
        src = cached_randn(
            (3, 8), differentiation="irdc_amin01_src", dtype=torch.float16
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "amin", _TOL_F16)

    def test_index_reduce_mean_float32(self, execution_mode):
        """index_reduce_ mean in float32; validates 4-byte word path for row accumulation."""
        self._xfail_duplicates(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_f32_01", dtype=torch.float32)
        src = cached_randn(
            (3, 8), differentiation="irdc_f32_01_src", dtype=torch.float32
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "mean", _TOL_F32)

    def test_index_reduce_single_index(self, execution_mode):
        """index_reduce_ with a single-element index; boundary case — P=1 row update."""
        dst = cached_randn((8, 16), differentiation="irdc_p1_01", dtype=torch.float16)
        src = cached_randn(
            (1, 16), differentiation="irdc_p1_01_src", dtype=torch.float16
        )
        idx = torch.tensor([3], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "amax", _TOL_F16)

    @pytest.mark.parametrize("reduce_op", ["mean", "prod", "amax", "amin"])
    @pytest.mark.parametrize("include_self", [False, True])
    def test_index_reduce_unique_indices(self, execution_mode, reduce_op, include_self):
        """index_reduce_ with unique indices on dim=0: supported (#6187), matches CPU.

        Every destination row receives at most one source row. Measured against CPU
        fp16 the largest error is 2e-3, so _TOL_F16 leaves 5x margin.
        """
        dst = cached_randn((4, 8), differentiation="irdc_uniq", dtype=torch.float16)
        src = cached_randn((3, 8), differentiation="irdc_uniq_src", dtype=torch.float16)
        idx = torch.tensor([2, 0, 3], dtype=torch.int64)

        def fn(dst, src, idx):
            return dst.clone().index_reduce_(
                0, idx, src, reduce=reduce_op, include_self=include_self
            )

        compare_mode(execution_mode, fn, dst, src, idx, atol=_TOL_F16, rtol=_TOL_F16)

    def test_index_reduce_stick_dim_rejected(self, execution_mode):
        """index_reduce_ along the last (stick) dim is rejected with Unsupported (#4404)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4404
        _xfail_existing(execution_mode, eager=_IR_STICK_DIM, compiled=_IR_STICK_DIM)
        dst = cached_randn((4, 64), differentiation="irdc_stick", dtype=torch.float16)
        src = cached_randn(
            (4, 2), differentiation="irdc_stick_src", dtype=torch.float16
        )
        idx = torch.tensor([1, 5], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, s, i: d.clone().index_reduce_(
                1, i, s, reduce="amax", include_self=False
            ),
            dst,
            src,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )


# ---------------------------------------------------------------------------
# index_fill
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestIndexFillEagerCompile:
    """index_fill_ standalone: fills selected rows/columns with a scalar constant."""

    def setup_method(self):
        torch.manual_seed(0xF111)

    def test_index_fill_dim0_float16(self, execution_mode):
        """index_fill_ on dim=0 with float16; fills rows [0, 2] with 3.0."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4414
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4472
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4414,
                "aten::index_fill_.int_Scalar is not registered on Spyre.",
                r"Could not run 'aten::index_fill_.int_Scalar'",
            ),
            compiled=rejection(
                4472,
                "compile index_fill_ stores a rank-0 Constant.",
                r"store value of unexpected type <class '[\w.]*Constant'>",
            ),
        )
        x = cached_randn((4, 16), differentiation="ifill_f16_01", dtype=torch.float16)
        idx = torch.tensor([0, 2], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x.clone().index_fill_(0, i, 3.0),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_fill_dim1_float16(self, execution_mode):
        """index_fill_ on dim=1 with float16; fills columns [1, 3] with -1.0."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4414
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4414,
                "aten::index_fill_.int_Scalar is not registered on Spyre.",
                r"Could not run 'aten::index_fill_.int_Scalar'",
            ),
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        x = cached_randn((8, 16), differentiation="ifill_f16_02", dtype=torch.float16)
        idx = torch.tensor([1, 3], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x.clone().index_fill_(1, i, -1.0),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_fill_1d_dest_length_8(self, execution_mode):
        """index_fill_ 1-D dest length 8 — compile leftover (#4399)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4414
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4414,
                "aten::index_fill_.int_Scalar is not registered on Spyre.",
                r"Could not run 'aten::index_fill_.int_Scalar'",
            ),
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        x = torch.zeros(8, dtype=torch.float16)
        idx = torch.tensor([0, 2], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda t, i: t.clone().index_fill_(0, i, 1.0),
            x,
            idx,
            atol=0,
            rtol=0,
        )


# ---------------------------------------------------------------------------
# masked_fill (standalone)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestMaskedFillStandaloneEagerCompile:
    """masked_fill as the primary operation on a plain tensor (not downstream after gather)."""

    def setup_method(self):
        torch.manual_seed(0xA55A)

    def test_masked_fill_float16(self, execution_mode):
        """masked_fill on (8,16) float16; every even position receives -inf sentinel.

        Exact (atol=0): masked_fill copies or overwrites, so on dl16-snapped
        inputs the device result is bit-identical to the CPU result.
        """
        x = cached_randn((8, 16), differentiation="mfill_f16_01", dtype=torch.float16)
        mask = torch.arange(8 * 16).reshape(8, 16) % 2 == 0
        compare_mode(
            execution_mode,
            lambda x, m: x.masked_fill(m, float("-inf")),
            x,
            mask,
            atol=0,
            rtol=0,
        )

    def test_masked_fill_float32(self, execution_mode):
        """masked_fill on (8,16) float32; validates 4-byte word path for mask-based fill."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4714
        _xfail_existing(
            execution_mode, eager=_MASKED_FILL_FP32, compiled=_MASKED_FILL_FP32
        )
        x = cached_randn((8, 16), differentiation="mfill_f32_01", dtype=torch.float32)
        mask = torch.arange(8 * 16).reshape(8, 16) % 2 == 0
        compare_mode(
            execution_mode,
            lambda x, m: x.masked_fill(m, float("-inf")),
            x,
            mask,
            atol=0,
            rtol=0,
        )

    def test_masked_fill_bfloat16(self, execution_mode):
        """masked_fill on (8,16) bfloat16; dtype parity check for bf16 masked writes."""
        x = cached_randn((8, 16), differentiation="mfill_bf16_01", dtype=torch.bfloat16)
        mask = torch.arange(8 * 16).reshape(8, 16) % 3 == 0
        compare_mode(
            execution_mode,
            lambda x, m: x.masked_fill(m, 0.0),
            x,
            mask,
            atol=0,
            rtol=0,
        )

    def test_masked_fill_2d_row_mask(self, execution_mode):
        """masked_fill with a column-broadcast mask (shape [8,1]); fills entire rows."""
        x = cached_randn((8, 32), differentiation="mfill_row_01", dtype=torch.float16)
        mask = torch.tensor(
            [True, False, True, False, True, False, True, False]
        ).unsqueeze(1)
        compare_mode(
            execution_mode,
            lambda x, m: x.masked_fill(m, -1e4),
            x,
            mask,
            atol=0,
            rtol=0,
        )


# ---------------------------------------------------------------------------
# put (flat-indexed write)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestPutFlatIndexWriteEagerCompile:
    """torch.Tensor.put_ standalone: flat-indexed write into the flattened tensor view."""

    def setup_method(self):
        torch.manual_seed(0x9876)

    def _xfail_put(self, execution_mode):
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4540
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4636
        _xfail_existing(execution_mode, eager=_PUT_EAGER, compiled=_PUT_COMPILED)

    def test_put_overwrite_float16(self, execution_mode):
        """put_ accumulate=False on float16; flat positions [0, 5, 11] receive values [1, 2, 3]."""
        self._xfail_put(execution_mode)
        x = cached_randn((3, 4), differentiation="put_ow_f16_01", dtype=torch.float16)
        idx = torch.tensor([0, 5, 11], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        compare_mode(
            execution_mode,
            lambda x, i, v: x.clone().put_(i, v, accumulate=False),
            x,
            idx,
            vals,
            atol=0,
            rtol=0,
        )

    def test_put_accumulate_float16(self, execution_mode):
        """put_ accumulate=True on float16; duplicate index [0,0] adds twice to flat pos 0."""
        self._xfail_put(execution_mode)
        x = cached_randn((3, 4), differentiation="put_acc_f16_01", dtype=torch.float16)
        idx = torch.tensor([0, 0, 5], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        compare_mode(
            execution_mode,
            lambda x, i, v: x.clone().put_(i, v, accumulate=True),
            x,
            idx,
            vals,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_put_overwrite_float32(self, execution_mode):
        """put_ accumulate=False on float32; validates 4-byte word path for flat writes."""
        self._xfail_put(execution_mode)
        x = cached_randn((3, 4), differentiation="put_ow_f32_01", dtype=torch.float32)
        idx = torch.tensor([0, 5, 11], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        compare_mode(
            execution_mode,
            lambda x, i, v: x.clone().put_(i, v, accumulate=False),
            x,
            idx,
            vals,
            atol=0,
            rtol=0,
        )

    def test_put_accumulate_float32(self, execution_mode):
        """put_ accumulate=True on float32; duplicate index [0,0] adds twice to flat pos 0."""
        self._xfail_put(execution_mode)
        x = cached_randn((3, 4), differentiation="put_acc_f32_01", dtype=torch.float32)
        idx = torch.tensor([0, 0, 5], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        compare_mode(
            execution_mode,
            lambda x, i, v: x.clone().put_(i, v, accumulate=True),
            x,
            idx,
            vals,
            atol=_TOL_F32,
            rtol=_TOL_F32,
        )

    def test_put_bfloat16(self, execution_mode):
        """put_ accumulate=False on bfloat16; validates 2-byte word flat write."""
        self._xfail_put(execution_mode)
        x = cached_randn((3, 4), differentiation="put_bf16_01", dtype=torch.bfloat16)
        idx = torch.tensor([0, 5, 11], dtype=torch.int64)
        vals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        compare_mode(
            execution_mode,
            lambda x, i, v: x.clone().put_(i, v, accumulate=False),
            x,
            idx,
            vals,
            atol=0,
            rtol=0,
        )

    def test_put_single_position(self, execution_mode):
        """put_ with a single flat index; boundary case P=1.

        Eager is missing aten::put_ (#4540). Compiled matches CPU.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4540
        _xfail_existing(execution_mode, eager=_PUT_EAGER)
        x = cached_randn((4, 8), differentiation="put_p1_01", dtype=torch.float16)
        idx = torch.tensor([7], dtype=torch.int64)
        vals = torch.tensor([-9.0], dtype=torch.float16)
        compare_mode(
            execution_mode,
            lambda x, i, v: x.clone().put_(i, v, accumulate=False),
            x,
            idx,
            vals,
            atol=0,
            rtol=0,
        )


# ---------------------------------------------------------------------------
# scatter_add
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterAddGroupedMmOffs:
    """Gemma grouped_mm offs — int64 dest [128], T=192. Eager #4396; compiled is rejected (#4396).

    Dest length 128 splits across cores; compiled runs at sencores=1 and chip-max.
    """

    def setup_method(self):
        torch.manual_seed(0)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    @gap_one_core
    def test_scatter_add_grouped_mm_offs_int64(self, execution_mode):
        """scatter_add dim=0 int64 counts [128], T=192 — Gemma grouped_mm offs (#4396)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4396
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4396,
                "aten::scatter_add.out is not registered on Spyre.",
                r"Could not run 'aten::scatter_add.out'",
            ),
            compiled=_scatter_fallback(4396, "scatter_add int64"),
        )
        G, T = 128, 192
        counts = torch.zeros(G, dtype=torch.int64)
        index = torch.randint(0, G, (T,), dtype=torch.int64)
        src = torch.ones(T, dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda c, i, s: c.scatter_add(0, i, s),
            counts,
            index,
            src,
            atol=0,
            rtol=0,
        )


# ---------------------------------------------------------------------------
# torch.scatter / scatter_(value=)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterRouterMaps:
    """Llama 4 K=1 and Qwen3 K=8 densify maps — dest (192, 128). #4403 / #4404 / #4354.

    Dest (192, 128) splits across cores; compiled runs at sencores=1 and chip-max.
    """

    def setup_method(self):
        torch.manual_seed(0)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    @gap_one_core
    def test_scatter_llama4_router_k1(self, execution_mode):
        """torch.scatter dim=1 K=1 into -inf dest (192, 128) — Llama 4 router (#4403)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1002
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4404
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=rejection(
                1002,
                "aten::scatter.src_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.src_out'",
            ),
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        T, E, K = 192, 128, 1
        dst = torch.full((T, E), float("-inf"), dtype=torch.float16)
        idx = _router_index(T, E, K)
        src = (
            (torch.arange(T, dtype=torch.float32) / 7.0).to(torch.float16).unsqueeze(1)
        )
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter(1, i, s),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    @gap_one_core
    def test_scatter_qwen3_router_k8(self, execution_mode):
        """torch.scatter dim=1 K=8 into zeros dest (192, 128) — Qwen3 MoE router (#4403)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1002
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4404
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=rejection(
                1002,
                "aten::scatter.src_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.src_out'",
            ),
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        T, E, K = 192, 128, 8
        dst = torch.zeros(T, E, dtype=torch.float16)
        idx = _router_index(T, E, K)
        src = cached_randn((T, K), differentiation="sc_qwen3", dtype=torch.float16)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter(1, i, s),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterValueStamp:
    """Scalar scatter_ extra dest ranks — grouped_mm offs and KV-cache slots. #4436 / #4472."""

    def setup_method(self):
        torch.manual_seed(0)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    @gap_one_core
    def test_scatter_value_grouped_mm_offs(self, execution_mode):
        """scatter_(0, idx, 1.0) onto fp16 dest [128], T=192 — leftover extra (#4436 Python).

        Eager is aten::scatter.value_out (#4473). Compiled fails in
        insert_restickify_padding (#6163): the restickify is written into a
        slice of another buffer, which cannot be padded in place.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4473
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/6163
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4473,
                "aten::scatter.value_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.value_out'",
            ),
            compiled=rejection(
                6163,
                "compile scatter_(value=) on an fp16 dest [128]: insert_restickify_padding cannot pad a restickify written into a slice of another buffer.",
                r"insert_restickify_padding: restickify written into a slice of another buffer",
            ),
        )
        G, T = 128, 192
        dst = torch.zeros(G, dtype=torch.float16)
        idx = torch.randint(0, G, (T,), dtype=torch.int64)

        def fn(d, i):
            out = d.clone()
            out.scatter_(0, i, 1.0)
            return out

        compare_mode(execution_mode, fn, dst, idx, atol=0, rtol=0)

    @gap_one_core
    def test_scatter_value_kv_cache_slots(self, execution_mode):
        """scatter_ stamp 0.0 into bf16 dest (256, 8, 128) at slots [3, 17] (#4436)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4473
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4472
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4473,
                "aten::scatter.value_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.value_out'",
            ),
            compiled=rejection(
                4472,
                "compile scatter_(value=) stores a rank-0 Constant.",
                r"store value of unexpected type <class '[\w.]*Constant'>",
            ),
        )
        dst = cached_randn((256, 8, 128), differentiation="sc_kv", dtype=torch.bfloat16)
        idx = torch.zeros(2, 8, 128, dtype=torch.int64)
        idx[0] = 3
        idx[1] = 17

        def fn(d, i):
            out = d.clone()
            out.scatter_(0, i, 0.0)
            return out

        compare_mode(execution_mode, fn, dst, idx, atol=0, rtol=0)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatter1DDestRestickify:
    """1-D dest below stick size / dest [64] — extra, not a YAML unique-abort name."""

    def test_scatter_1d_dest_length_5_fp32(self, execution_mode):
        """scatter_ dest length 5 fp32 — leftover extra (#4474 Python).

        Eager is aten::scatter.src_out (#1002). Compiled this log was
        normalize_coordinates offset==0 (#4399), not the filed #4474
        insert_restickify_padding string.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1002
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=rejection(
                1002,
                "aten::scatter.src_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.src_out'",
            ),
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        x = torch.zeros(5, dtype=torch.float32)
        index = torch.tensor([1, 3], dtype=torch.int32)
        src = torch.tensor([100.0, 200.0], dtype=torch.float32)

        def fn(d, i, s):
            out = d.clone()
            out.scatter_(0, i, s)
            return out

        compare_mode(execution_mode, fn, x, index, src, atol=0, rtol=0)

    @gap_one_core
    def test_scatter_1d_dest_length_64(self, execution_mode, patch_sencores):
        """scatter_ dest [64] fp16 — compile assert offset == 0 (#4399)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1002
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=rejection(
                1002,
                "aten::scatter.src_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.src_out'",
            ),
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        x = torch.zeros(64, dtype=torch.float16)
        index = torch.tensor([1, 3], dtype=torch.int64)
        src = torch.tensor([1.0, 2.0], dtype=torch.float16)

        def fn(d, i, s):
            out = d.clone()
            out.scatter_(0, i, s)
            return out

        compare_mode(execution_mode, fn, x, index, src, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# scatter_reduce / scatter_(reduce=)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterReduceLegacyGroupedMm:
    """Legacy Tensor.scatter_(reduce=) on grouped_mm dest ranks. Eager and compiled: #4413.

    Dest length 128 splits across cores; compiled runs at sencores=1 and chip-max.
    """

    def setup_method(self):
        torch.manual_seed(0)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    @gap_one_core
    def test_scatter_reduce_add_grouped_mm_int64(self, execution_mode):
        """scatter_(reduce='add') int64 dest [128], T=192 (#4413)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4413
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4413,
                "aten::scatter.reduce_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.reduce_out'",
            ),
            compiled=_scatter_fallback(4413, "scatter_(reduce=)"),
        )
        G, T = 128, 192
        dst = torch.ones(G, dtype=torch.int64)
        idx = torch.randint(0, G, (T,), dtype=torch.int64)
        src = torch.ones(T, dtype=torch.int64)

        def fn(d, i, s):
            out = d.clone()
            out.scatter_(0, i, s, reduce="add")
            return out

        compare_mode(execution_mode, fn, dst, idx, src, atol=0, rtol=0)

    @gap_one_core
    def test_scatter_reduce_multiply_grouped_mm_fp16(self, execution_mode):
        """scatter_(reduce='multiply') fp16 dest [128], T=192; dest starts at ones (#4413)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4413
        _xfail_existing(
            execution_mode,
            eager=rejection(
                4413,
                "aten::scatter.reduce_out is not registered on Spyre.",
                r"Could not run 'aten::scatter.reduce_out'",
            ),
            compiled=_scatter_fallback(4413, "scatter_(reduce=)"),
        )
        G, T = 128, 192
        dst = torch.ones(G, dtype=torch.float16)
        idx = torch.randint(0, G, (T,), dtype=torch.int64)
        src = cached_randn((T,), differentiation="sr_mul_src", dtype=torch.float16) + 1

        def fn(d, i, s):
            out = d.clone()
            out.scatter_(0, i, s, reduce="multiply")
            return out

        compare_mode(execution_mode, fn, dst, idx, src, atol=_TOL_F16, rtol=_TOL_F16)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterReduceTwoColliding:
    """aten::scatter_reduce.two_out colliding dest. Eager and compiled: #4361.

    dim=0 dest (32, 64) splits at chip-max; dim=1 dest (4, 64) stays single-config.
    """

    def setup_method(self):
        torch.manual_seed(0)

    def _xfail_colliding_dim0(self, execution_mode):
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4361
        _xfail_existing(
            execution_mode,
            eager=_SR_TWO,
            compiled=_scatter_fallback(4361, "scatter_reduce with colliding indices"),
        )

    @gap_one_core
    def test_scatter_reduce_amax_colliding_dim0(self, execution_mode, patch_sencores):
        """scatter_reduce amax dim=0 dest/src (32, 64) non-unique index (#4361)."""
        self._xfail_colliding_dim0(execution_mode)
        dst = torch.randn(32, 64, dtype=torch.float16)
        src = torch.randn(32, 64, dtype=torch.float16)
        idx = torch.randint(0, 32, (32, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter_reduce(0, i, s, reduce="amax"),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    @gap_one_core
    def test_scatter_reduce_prod_colliding_dim0(self, execution_mode, patch_sencores):
        """scatter_reduce prod dim=0 dest/src (32, 64) non-unique index (#4361)."""
        self._xfail_colliding_dim0(execution_mode)
        dst = torch.randn(32, 64, dtype=torch.float16).abs() + 0.5
        src = torch.randn(32, 64, dtype=torch.float16).abs() + 0.5
        idx = torch.randint(0, 32, (32, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter_reduce(0, i, s, reduce="prod"),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    @gap_one_core
    def test_scatter_reduce_mean_colliding_dim0(self, execution_mode, patch_sencores):
        """scatter_reduce mean dim=0 dest/src (32, 64) non-unique index (#4361)."""
        self._xfail_colliding_dim0(execution_mode)
        dst = torch.randn(32, 64, dtype=torch.float16)
        src = torch.randn(32, 64, dtype=torch.float16)
        idx = torch.randint(0, 32, (32, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter_reduce(0, i, s, reduce="mean"),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    @gap_one_core
    def test_scatter_reduce_amin_must_write_dim0(self, execution_mode, patch_sencores):
        """scatter_reduce amin dim=0 dest=100 so dest is not already the min (#4361)."""
        self._xfail_colliding_dim0(execution_mode)
        dst = torch.full((32, 64), 100.0, dtype=torch.float16)
        src = torch.randn(32, 64, dtype=torch.float16)
        idx = torch.randint(0, 32, (32, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter_reduce(0, i, s, reduce="amin"),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_scatter_reduce_sum_dim1_fp16(self, execution_mode):
        """scatter_reduce sum dim=1 fp16 dest (4, 64) src (4, 32).

        Eager is aten::scatter_reduce.two_out (#4361). Compiled is KeyError tmp0 (#4637),
        not #4346 (indirect_sizes {}) and not #4354 (ranges_from_index_vars).
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4361
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4637
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=_SR_TWO,
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        dst = (
            (torch.arange(4 * 64, dtype=torch.float32) / 7.0)
            .reshape(4, 64)
            .to(torch.float16)
        )
        src = (
            (torch.arange(4 * 32, dtype=torch.float32) / 7.0)
            .reshape(4, 32)
            .to(torch.float16)
        )
        idx = torch.zeros(4, 32, dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter_reduce(1, i, s, reduce="sum"),
            dst,
            idx,
            src,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_scatter_reduce_sum_dim1_fp32(self, execution_mode):
        """scatter_reduce sum dim=1 fp32 — ReStickifyOpHBM leftover (#1796)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4361
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1796
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=_SR_TWO,
            compiled=rejection(
                4399,
                "compile scatter-family write (index_fill_ / scatter / scatter_reduce): cannot re-tile buf, no size-1 stick target.",
                r"cannot re-tile \S+: no size-1 stick target",
            ),
        )
        dst = cached_randn((4, 64), differentiation="sr_f32", dtype=torch.float32)
        src = cached_randn((4, 32), differentiation="sr_f32s", dtype=torch.float32)
        idx = torch.zeros(4, 32, dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda d, i, s: d.scatter_reduce(1, i, s, reduce="sum"),
            dst,
            idx,
            src,
            atol=_TOL_F32,
            rtol=_TOL_F32,
        )


class TestScatterReduceCompileDynamic:
    """compile(dynamic=True) scatter_reduce sum dim=0 is refused (#5578)."""

    @expect_dynamic_true_refused
    def test_scatter_reduce_sum_dim0_dynamic(self):
        """compile(dynamic=True) scatter_reduce sum dim=0 is refused."""
        dst = (
            (torch.arange(4 * 64, dtype=torch.float32) / 7.0)
            .reshape(4, 64)
            .to(torch.float16)
        )
        src = (
            (torch.arange(2 * 64, dtype=torch.float32) / 3.0)
            .reshape(2, 64)
            .to(torch.float16)
        )
        idx = torch.zeros(2, 64, dtype=torch.int64)
        ref = dst.scatter_reduce(0, idx, src, reduce="sum")
        got = torch.compile(
            lambda d, i, s: d.scatter_reduce(0, i, s, reduce="sum"),
            dynamic=True,
        )(dst.to(DEVICE), idx.to(DEVICE), src.to(DEVICE)).cpu()
        torch.testing.assert_close(got, ref, atol=_TOL_F16, rtol=_TOL_F16)
