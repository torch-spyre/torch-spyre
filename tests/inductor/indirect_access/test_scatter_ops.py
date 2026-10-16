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

"""Scatter-family and related indirect writes: scatter / scatter_add /
scatter_reduce, index_reduce, index_fill, masked_fill, and put.

Multi-sencores is compiled-only: dests whose indexed dim can split across cores
request patch_sencores (eager once at default config; compiled at 1 and chip-max).
Small dests stay eager + compiled at default config. Eager is never a SENCORES variant.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import DEVICE, cached_randn  # noqa: E402
from conftest import _xfail_existing, compare_mode  # noqa: E402

_ATOL_F16 = 1e-2
_ATOL_F32 = 1e-5


_IR_OUT = (4634, "aten::index_reduce.out is not registered on Spyre.")
_IR_MEAN_CONST = (4472, "compile index_reduce_ mean stores a rank-0 Constant.")
_MASKED_EAGER = (4356, "aten::masked_fill_.Scalar is not registered on Spyre.")
_PUT_EAGER = (4540, "aten::put_ is not registered on Spyre.")
_PUT_COMPILED = (
    4636,
    "compile put_ has no output layout for aten.index_put.default.",
)
_SR_TWO = (4361, "aten::scatter_reduce.two_out is not registered on Spyre.")


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

    def _xfail_kernel(self, execution_mode):
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4634
        _xfail_existing(execution_mode, eager=_IR_OUT, compiled=_IR_OUT)

    def _xfail_mean(self, execution_mode):
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4634
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4472
        _xfail_existing(execution_mode, eager=_IR_OUT, compiled=_IR_MEAN_CONST)

    def _run(self, execution_mode, dst, src, idx, reduce_op, atol):
        def fn(dst, src, idx):
            return dst.clone().index_reduce_(
                0, idx, src, reduce=reduce_op, include_self=False
            )

        compare_mode(execution_mode, fn, dst, src, idx, atol=atol, rtol=atol)

    def test_index_reduce_mean_dim0(self, execution_mode):
        """index_reduce_ mean on dim=0; duplicate indices average rows into destination.

        PyTorch index_reduce_ does not accept reduce='sum' (use scatter_add for sum).
        Eager is missing aten::index_reduce.out (#4634). Compiled mean is Constant (#4472).
        """
        self._xfail_mean(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_mean01", dtype=torch.float16)
        src = cached_randn(
            (3, 8), differentiation="irdc_mean01_src", dtype=torch.float16
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "mean", _ATOL_F16)

    def test_index_reduce_prod_dim0(self, execution_mode):
        """index_reduce_ prod on dim=0; duplicate indices multiply rows into destination."""
        self._xfail_kernel(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_prod01", dtype=torch.float16)
        src = cached_randn(
            (3, 8), differentiation="irdc_prod01_src", dtype=torch.float16
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "prod", _ATOL_F16)

    def test_index_reduce_amax_dim0(self, execution_mode):
        """index_reduce_ amax on dim=0; duplicate indices take element-wise max across rows."""
        self._xfail_kernel(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_amax01", dtype=torch.float16)
        src = cached_randn(
            (3, 8), differentiation="irdc_amax01_src", dtype=torch.float16
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "amax", _ATOL_F16)

    def test_index_reduce_amin_dim0(self, execution_mode):
        """index_reduce_ amin on dim=0; duplicate indices take element-wise min across rows."""
        self._xfail_kernel(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_amin01", dtype=torch.float16)
        src = cached_randn(
            (3, 8), differentiation="irdc_amin01_src", dtype=torch.float16
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "amin", _ATOL_F16)

    def test_index_reduce_mean_float32(self, execution_mode):
        """index_reduce_ mean in float32; validates 4-byte word path for row accumulation."""
        self._xfail_mean(execution_mode)
        dst = cached_randn((4, 8), differentiation="irdc_f32_01", dtype=torch.float32)
        src = cached_randn(
            (3, 8), differentiation="irdc_f32_01_src", dtype=torch.float32
        )
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "mean", _ATOL_F32)

    def test_index_reduce_single_index(self, execution_mode):
        """index_reduce_ with a single-element index; boundary case — P=1 row update."""
        self._xfail_kernel(execution_mode)
        dst = cached_randn((8, 16), differentiation="irdc_p1_01", dtype=torch.float16)
        src = cached_randn(
            (1, 16), differentiation="irdc_p1_01_src", dtype=torch.float16
        )
        idx = torch.tensor([3], dtype=torch.int64)
        self._run(execution_mode, dst, src, idx, "amax", _ATOL_F16)


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
            eager=(4414, "aten::index_fill_.int_Scalar is not registered on Spyre."),
            compiled=(
                4472,
                "compile index_fill_ stores a rank-0 Constant.",
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
            eager=(4414, "aten::index_fill_.int_Scalar is not registered on Spyre."),
            compiled=(
                4399,
                "compile index_fill_ dim=1 hits normalize_coordinates offset==0.",
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
            eager=(4414, "aten::index_fill_.int_Scalar is not registered on Spyre."),
            compiled=(
                4399,
                "compile index_fill_ dest length 8 hits normalize_coordinates offset==0.",
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

    def _xfail_eager(self, execution_mode):
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4356
        _xfail_existing(execution_mode, eager=_MASKED_EAGER)

    def test_masked_fill_float16(self, execution_mode):
        """masked_fill on (8,16) float16; every even position receives -inf sentinel.

        Compiled uses _ATOL_F16: atol=0 is 1 fp16 ULP on unmasked elems, not a missed write
        (leftover #4356). Eager is missing aten::masked_fill_.Scalar (#4356).
        """
        self._xfail_eager(execution_mode)
        x = cached_randn((8, 16), differentiation="mfill_f16_01", dtype=torch.float16)
        mask = torch.arange(8 * 16).reshape(8, 16) % 2 == 0
        compare_mode(
            execution_mode,
            lambda x, m: x.masked_fill(m, float("-inf")),
            x,
            mask,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_masked_fill_float32(self, execution_mode):
        """masked_fill on (8,16) float32; validates 4-byte word path for mask-based fill."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4356
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4073
        _xfail_existing(
            execution_mode,
            eager=_MASKED_EAGER,
            compiled=(4073, "compile masked_fill fp32 hits where3 on IEEE_FP32."),
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
        self._xfail_eager(execution_mode)
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
        self._xfail_eager(execution_mode)
        x = cached_randn((8, 32), differentiation="mfill_row_01", dtype=torch.float16)
        mask = torch.tensor(
            [True, False, True, False, True, False, True, False]
        ).unsqueeze(1)
        compare_mode(
            execution_mode,
            lambda x, m: x.masked_fill(m, -1e4),
            x,
            mask,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
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
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
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
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
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
        """put_ with a single flat index; boundary case P=1."""
        self._xfail_put(execution_mode)
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
    """Gemma grouped_mm offs — int64 dest [128], T=192. #4395.

    Dest length 128 splits across cores; compiled runs at sencores=1 and chip-max.
    """

    def setup_method(self):
        torch.manual_seed(0)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    def test_scatter_add_grouped_mm_offs_int64(self, execution_mode):
        """scatter_add dim=0 int64 counts [128], T=192 — Gemma grouped_mm offs (#4395)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4396
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4395
        _xfail_existing(
            execution_mode,
            eager=(4396, "aten::scatter_add.out is not registered on Spyre."),
            compiled=(
                4395,
                "compile scatter_add int64 grouped_mm offs mismatches CPU.",
            ),
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

    def test_scatter_llama4_router_k1(self, execution_mode):
        """torch.scatter dim=1 K=1 into -inf dest (192, 128) — Llama 4 router (#4403)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1002
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4404
        _xfail_existing(
            execution_mode,
            eager=(1002, "aten::scatter.src_out is not registered on Spyre."),
            compiled=(
                4404,
                "compile Llama 4 K=1 restickify restore size-64 gap dim.",
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
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_scatter_qwen3_router_k8(self, execution_mode):
        """torch.scatter dim=1 K=8 into zeros dest (192, 128) — Qwen3 MoE router (#4403)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1002
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4404
        _xfail_existing(
            execution_mode,
            eager=(1002, "aten::scatter.src_out is not registered on Spyre."),
            compiled=(
                4404,
                "compile Qwen3 K=8 restickify restore size-64 gap dim.",
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
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterValueStamp:
    """Scalar scatter_ extra dest ranks — grouped_mm offs and KV-cache slots. #4436 / #4472."""

    def setup_method(self):
        torch.manual_seed(0)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    def test_scatter_value_grouped_mm_offs(self, execution_mode):
        """scatter_(0, idx, 1.0) onto fp16 dest [128], T=192 — leftover extra (#4436 Python).

        Eager is aten::scatter.value_out (#4473). Compiled this log was
        ranges_from_index_vars (#4354 leftover string), not #4436 mismatch
        and not #4472 Constant.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4473
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4354
        _xfail_existing(
            execution_mode,
            eager=(4473, "aten::scatter.value_out is not registered on Spyre."),
            compiled=(
                4354,
                "compile scatter_(value=) grouped_mm hits ranges_from_index_vars.",
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

    def test_scatter_value_kv_cache_slots(self, execution_mode):
        """scatter_ stamp 0.0 into bf16 dest (256, 8, 128) at slots [3, 17] (#4436)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4473
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4472
        _xfail_existing(
            execution_mode,
            eager=(4473, "aten::scatter.value_out is not registered on Spyre."),
            compiled=(
                4472,
                "compile scatter_(value=) KV dest stores a rank-0 Constant.",
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
            eager=(1002, "aten::scatter.src_out is not registered on Spyre."),
            compiled=(
                4399,
                "compile 1-D dest length 5 hits normalize_coordinates offset==0.",
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

    def test_scatter_1d_dest_length_64(self, execution_mode, patch_sencores):
        """scatter_ dest [64] fp16 — compile assert offset == 0 (#4399)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1002
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(
            execution_mode,
            eager=(1002, "aten::scatter.src_out is not registered on Spyre."),
            compiled=(
                4399,
                "compile 1-D dest [64] hits normalize_coordinates offset==0.",
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
    """Legacy Tensor.scatter_(reduce=) on grouped_mm dest ranks. #4407.

    Dest length 128 splits across cores; compiled runs at sencores=1 and chip-max.
    """

    def setup_method(self):
        torch.manual_seed(0)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    def test_scatter_reduce_add_grouped_mm_int64(self, execution_mode):
        """scatter_(reduce='add') int64 dest [128], T=192 (#4407)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4413
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4407
        _xfail_existing(
            execution_mode,
            eager=(4413, "aten::scatter.reduce_out is not registered on Spyre."),
            compiled=(
                4407,
                "compile scatter_(reduce='add') grouped_mm int64 mismatches CPU.",
            ),
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

    def test_scatter_reduce_multiply_grouped_mm_fp16(self, execution_mode):
        """scatter_(reduce='multiply') fp16 dest [128], T=192; dest starts at ones (#4407)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4413
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4407
        _xfail_existing(
            execution_mode,
            eager=(4413, "aten::scatter.reduce_out is not registered on Spyre."),
            compiled=(
                4407,
                "compile scatter_(reduce='multiply') grouped_mm mismatches CPU.",
            ),
        )
        G, T = 128, 192
        dst = torch.ones(G, dtype=torch.float16)
        idx = torch.randint(0, G, (T,), dtype=torch.int64)
        src = cached_randn((T,), differentiation="sr_mul_src", dtype=torch.float16) + 1

        def fn(d, i, s):
            out = d.clone()
            out.scatter_(0, i, s, reduce="multiply")
            return out

        compare_mode(execution_mode, fn, dst, idx, src, atol=_ATOL_F16, rtol=_ATOL_F16)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestScatterReduceTwoColliding:
    """aten::scatter_reduce.two_out colliding dest — extra vs YAML reduce_out. #4351.

    dim=0 dest (32, 64) splits at chip-max; dim=1 dest (4, 64) stays single-config.
    """

    def setup_method(self):
        torch.manual_seed(0)

    def _xfail_colliding_dim0(self, execution_mode):
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4361
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4351
        _xfail_existing(
            execution_mode,
            eager=_SR_TWO,
            compiled=(
                4351,
                "compile scatter_reduce colliding dim=0 mismatches CPU.",
            ),
        )

    def test_scatter_reduce_amax_colliding_dim0(self, execution_mode, patch_sencores):
        """scatter_reduce amax dim=0 dest/src (32, 64) non-unique index (#4351)."""
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
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_scatter_reduce_prod_colliding_dim0(self, execution_mode, patch_sencores):
        """scatter_reduce prod dim=0 dest/src (32, 64) non-unique index (#4351)."""
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
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_scatter_reduce_mean_colliding_dim0(self, execution_mode, patch_sencores):
        """scatter_reduce mean dim=0 dest/src (32, 64) non-unique index (#4351)."""
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
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_scatter_reduce_amin_must_write_dim0(self, execution_mode, patch_sencores):
        """scatter_reduce amin dim=0 dest=100 so dest is not already the min (#4351)."""
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
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_scatter_reduce_sum_dim1_fp16(self, execution_mode):
        """scatter_reduce sum dim=1 fp16 dest (4, 64) src (4, 32).

        Eager is aten::scatter_reduce.two_out (#4361). Compiled is KeyError tmp0 (#4637),
        not #4346 (indirect_sizes {}) and not #4354 (ranges_from_index_vars).
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4361
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4637
        _xfail_existing(
            execution_mode,
            eager=_SR_TWO,
            compiled=(4637, "compile scatter_reduce sum dim=1 fp16 KeyError tmp0."),
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
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_scatter_reduce_sum_dim1_fp32(self, execution_mode):
        """scatter_reduce sum dim=1 fp32 — ReStickifyOpHBM leftover (#1796)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4361
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1796
        _xfail_existing(
            execution_mode,
            eager=_SR_TWO,
            compiled=(
                1796,
                "compile scatter_reduce sum dim=1 fp32 ReStickifyOpHBM IEEE_FP32.",
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
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )


class TestScatterReduceCompileDynamic:
    """compile(dynamic=True) scatter_reduce sum dim=0.

    Unique abort is alignment input is not concrete: s11 (#4638), not #4346
    (indirect symbol tmp0 not found in indirect_sizes {}).
    """

    def test_scatter_reduce_sum_dim0_dynamic(self):
        """compile(dynamic=True) scatter_reduce sum dim=0 — alignment s11 (#4638)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4638
        pytest.xfail(
            reason="compile(dynamic=True) scatter_reduce hits alignment input "
            "is not concrete: s11. See issue #4638."
        )
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
        torch.testing.assert_close(got, ref, atol=_ATOL_F16, rtol=_ATOL_F16)
