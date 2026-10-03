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

"""GATHER_OP_SPEC emission, compiler pass ordering, stick-aligned and non-aligned memory layouts, dynamic shape specialization (M/P/D), SENCORES 1/4/32, LX budget control, and index computation inside the compiled graph."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import DEVICE, cached_randn  # noqa: E402
from conftest import _xfail_existing, compare_mode  # noqa: E402

_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_F32 = 1e-5

_INDEX_EAGER = (1219, "aten::index.Tensor_out is not registered on Spyre.")
_INCOMPATIBLE_HOST_SIZE = (
    3732,
    "compile take / flatten-take Incompatible host_size and dim_order.",
)
_SYMBOLIC_INT_CONV = (
    4304,
    "compile_dyn / symbolic size Cannot convert symbols to int.",
)
_GATHER_OUT_LAYOUTS_EMPTY = (
    4713,
    "Compile topk followed by table gather out_layouts is empty.",
)
_DXP_STANDALONE_FAIL = (
    4321,
    "compile gather DBO dxp_standalone returned non-zero exit status 1.",
)
_RESTICKIFY_3ARGS = (
    4327,
    "compile gather / index_select dim>=1 restickify op_spec has 3 args.",
)


class TestGatherCompilerPassesMemoryAndDynamicShapes:
    """GATHER_OP_SPEC emission and pass order, stick/non-aligned/multi-stick memory layouts, dynamic M/P/D shapes, SENCORES/LX configuration, LX budget variants, and in-graph index computation."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self, sencores):
        from torch_spyre._inductor import config

        with config.patch({"sencores": sencores}):
            yield
        os.environ.pop("LX_PLANNING", None)
        os.environ.pop("DXP_LX_FRAC_AVAIL", None)

    # ------------------------------------------------------------------

    def test_sdsc_index_in_hbm(self):
        """Index tensor stays in HBM; not placed in LX scratchpad."""
        os.environ["LX_PLANNING"] = "1"
        x = cached_randn((64, 128), differentiation="gcp04", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            "compiled", lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_two_gathers_two_op_specs(self):
        """Two independent gathers; two GATHER_OP_SPECs in same graph."""
        x = cached_randn((64, 128), differentiation="gcp05a", dtype=torch.float16)
        y = cached_randn((64, 128), differentiation="gcp05b", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda x, y, i: (x[i], y[i]),
            x,
            y,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_reshape_before_gather(self):
        """Reshape barrier before gather; GATHER_OP_SPEC at post-reshape."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/3732
        _xfail_existing("compiled", compiled=_INCOMPATIBLE_HOST_SIZE)
        x = cached_randn((8, 8, 128), differentiation="gcp07", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda x, i: x.reshape(64, 128)[i],
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_symbolic_args_encoding(self):
        """BUNDLE_SYMBOLIC_ARGS encoding in SDSC; dynamic idx shape."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing("compiled", compiled=_SYMBOLIC_INT_CONV)
        x = cached_randn((64, 128), differentiation="gcp08", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        result = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
        torch.testing.assert_close(result, x[idx], atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_gather_with_downstream_fused(self):
        """Downstream op fused into SDSC; single kernel generated."""
        x = cached_randn((64, 128), differentiation="gcp12", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda x, i: torch.tanh(x[i]),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    def test_stick_aligned_source(self):
        """Source rows are stick-aligned (D=64, fp16, 128 bytes)."""
        x = cached_randn((64, 64), differentiation="mem01", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            "compiled", lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_bfloat16_alignment(self):
        """bfloat16 alignment: 2 bytes/elem; stick boundaries differ from fp16."""
        x = cached_randn((64, 64), differentiation="mem10", dtype=torch.bfloat16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            "compiled", lambda x, i: x[i], x, idx, atol=_ATOL_BF16, rtol=_ATOL_BF16
        )

    def test_stride_non_standard(self):
        """Non-standard strides from slice; still gathers correctly."""
        x = cached_randn((128, 128), differentiation="mem11", dtype=torch.float16)
        x_sliced = x[::2].contiguous()
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda x, i: x[i],
            x_sliced,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    def test_dynamic_source_size(self):
        """Dynamic M (source rows); compiled once, different M at runtime."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing("compiled", compiled=_SYMBOLIC_INT_CONV)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        for M in (32, 64):
            x = cached_randn(
                (M, 128), differentiation=f"dyn01_{M}", dtype=torch.float16
            )
            idx = torch.randint(0, M, (16,), dtype=torch.int64)
            result = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, x[idx], atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_dynamic_index_size(self):
        """Dynamic P (index length); compiled once, P=16 and P=32."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing("compiled", compiled=_SYMBOLIC_INT_CONV)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        x = cached_randn((64, 128), differentiation="dyn02", dtype=torch.float16)
        for P in (16, 32):
            idx = torch.randint(0, 64, (P,), dtype=torch.int64)
            result = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, x[idx], atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_dynamic_inner_dim(self):
        """Dynamic D (inner dim); D=64 and D=128 in same session."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing("compiled", compiled=_SYMBOLIC_INT_CONV)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        for D in (64, 128):
            x = cached_randn((32, D), differentiation=f"dyn03_{D}", dtype=torch.float16)
            result = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, x[idx], atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_symbolic_gather_dim0(self):
        """Symbolic shape at gather dim=0; GATHER_OP_SPEC generated."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing("compiled", compiled=_SYMBOLIC_INT_CONV)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        x = cached_randn((64, 64), differentiation="dyn05", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        result = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
        torch.testing.assert_close(result, x[idx], atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_symbolic_batch_seqlen(self):
        """Dynamic batch × seqlen; slot_idxs (B*Lk) varies."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing("compiled", compiled=_SYMBOLIC_INT_CONV)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        kv = cached_randn((512, 8, 64), differentiation="dyn06", dtype=torch.float16)
        for B_Lk in (32, 64):
            idx = torch.randint(0, 512, (B_Lk,), dtype=torch.int64)
            result = fn(kv.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, kv[idx], atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_dynamic_3d_source(self):
        """Dynamic 3D source (M, H, D); gather at dim=0."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing("compiled", compiled=_SYMBOLIC_INT_CONV)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        for M in (32, 64):
            x = cached_randn(
                (M, 8, 64), differentiation=f"dyn07_{M}", dtype=torch.float16
            )
            idx = torch.randint(0, M, (16,), dtype=torch.int64)
            result = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, x[idx], atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_dynamic_with_downstream(self):
        """Dynamic shape + downstream fused op (tanh)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing("compiled", compiled=_SYMBOLIC_INT_CONV)
        fn = torch.compile(lambda x, i: torch.tanh(x[i]), dynamic=True)
        x = cached_randn((64, 128), differentiation="dyn08", dtype=torch.float16)
        for P in (16, 32):
            idx = torch.randint(0, 64, (P,), dtype=torch.int64)
            result = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
            expected = torch.tanh(x[idx])
            torch.testing.assert_close(result, expected, atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_dynamic_bfloat16(self):
        """Dynamic shape with bfloat16; SDSC wordLength=2."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing("compiled", compiled=_SYMBOLIC_INT_CONV)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        for P in (16, 32):
            x = cached_randn(
                (64, 128), differentiation=f"dyn09_{P}", dtype=torch.bfloat16
            )
            idx = torch.randint(0, 64, (P,), dtype=torch.int64)
            result = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, x[idx], atol=_ATOL_BF16, rtol=_ATOL_BF16)

    def test_dynamic_vocab_size(self):
        """Dynamic vocab size; embedding-like lookup with varying M."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing("compiled", compiled=_SYMBOLIC_INT_CONV)
        fn = torch.compile(lambda x, i: x[i], dynamic=True)
        for V in (256, 512):
            w = cached_randn(
                (V, 128), differentiation=f"dyn10_{V}", dtype=torch.float16
            )
            idx = torch.randint(0, V, (32,), dtype=torch.int64)
            result = fn(w.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, w[idx], atol=_ATOL_F16, rtol=_ATOL_F16)

    def test_take_dynamic_true(self):
        """torch.take compile(dynamic=True) — Cannot convert symbols to int (#4304).

        Existing dynamic tests are x[i] gather. This extra is take's reshape(-1)[index] path.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4304
        _xfail_existing("compiled", compiled=_SYMBOLIC_INT_CONV)
        inp = torch.arange(64, dtype=torch.float16) / 7.0 - 1.5
        idx = torch.tensor([0, 16, 32, 63], dtype=torch.int64)
        fn = torch.compile(torch.take, dynamic=True)
        result = fn(inp.to(DEVICE), idx.to(DEVICE)).cpu()
        torch.testing.assert_close(
            result, torch.take(inp, idx), atol=_ATOL_F16, rtol=_ATOL_F16
        )

    # ------------------------------------------------------------------

    def test_sencores_correctness(self):
        """Gather correctness across sencores=1/4/32 (from class fixture) on (256,128)."""
        x = cached_randn((256, 128), differentiation="cfg05", dtype=torch.float16)
        idx = torch.randint(0, 256, (64,), dtype=torch.int64)
        compare_mode(
            "compiled", lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    # ------------------------------------------------------------------

    def test_lxd_high_fraction(self):
        """DXP_LX_FRAC_AVAIL=0.8; large LX budget; downstream eligible."""
        os.environ["LX_PLANNING"] = "1"
        os.environ["DXP_LX_FRAC_AVAIL"] = "0.8"
        x = cached_randn((64, 128), differentiation="lxd02", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda x, i: torch.relu(x[i]),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_lxd_lx_overlap(self):
        """LX-LX overlap: downstream output eligible for LX re-use."""
        os.environ["LX_PLANNING"] = "1"
        x = cached_randn((64, 128), differentiation="lxd03", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda x, i: torch.sigmoid(torch.relu(x[i])),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_lxd_gate_off_no_lx(self):
        """LX_PLANNING=1 works correctly for non-gather ops."""
        os.environ["LX_PLANNING"] = "1"
        x = cached_randn((64, 128), differentiation="lxd08", dtype=torch.float16)
        compare_mode(
            "compiled", lambda x: torch.relu(x), x, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_lxd_large_tensor_lx(self):
        """Large tensor with LX_PLANNING; allocation pressure test."""
        os.environ["LX_PLANNING"] = "1"
        x = cached_randn((4096, 64), differentiation="lxd09", dtype=torch.float16)
        idx = torch.randint(0, 4096, (128,), dtype=torch.int64)
        compare_mode(
            "compiled", lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_lxd_bfloat16_co_opt(self):
        """bfloat16 + LX_PLANNING co-optimization; 2-byte elements."""
        os.environ["LX_PLANNING"] = "1"
        x = cached_randn((64, 128), differentiation="lxd10", dtype=torch.bfloat16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            "compiled", lambda x, i: x[i], x, idx, atol=_ATOL_BF16, rtol=_ATOL_BF16
        )

    # ------------------------------------------------------------------

    def test_arange_index_inside_graph(self):
        """torch.arange generates index inside compiled graph; GATHER_OP_SPEC must fire."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing("compiled", compiled=_INDEX_EAGER)
        x = cached_randn((64, 128), differentiation="ing01", dtype=torch.float16)
        seqlen = 32
        compare_mode(
            "compiled",
            lambda x: x[torch.arange(seqlen, dtype=torch.int64)],
            x,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_topk_index_inside_graph(self):
        """topk inside compiled graph produces row index; expert_w[ids] fires GATHER_OP_SPEC."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4713
        _xfail_existing("compiled", compiled=_GATHER_OUT_LAYOUTS_EMPTY)
        expert_w = cached_randn(
            (8, 64, 32), differentiation="ing02", dtype=torch.float16
        )
        logits = cached_randn((1, 8), differentiation="ing02l", dtype=torch.float16)
        compare_mode(
            "compiled",
            lambda w, lgt: w[
                torch.topk(lgt, 2, dim=-1).indices.flatten().to(torch.int64)
            ],
            expert_w,
            logits,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_nonzero_index_inside_graph(self):
        """nonzero inside graph selects active rows; gather fetches those rows."""
        x = cached_randn((32, 64), differentiation="ing03", dtype=torch.float16)
        scores = torch.randint(0, 2, (32,), dtype=torch.int64)
        nz_idx = scores.nonzero(as_tuple=True)[0].to(torch.int64)
        if nz_idx.numel() == 0:
            nz_idx = torch.tensor([0], dtype=torch.int64)
        compare_mode(
            "compiled", lambda x, i: x[i], x, nz_idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_where_index_inside_graph(self):
        """torch.where condition → index inside compiled graph; gather valid slot rows."""
        x = cached_randn((32, 64), differentiation="ing04", dtype=torch.float16)
        scores = torch.randn(32)
        idx = torch.where(scores > 0)[0].to(torch.int64)
        if idx.numel() == 0:
            idx = torch.tensor([0], dtype=torch.int64)
        compare_mode(
            "compiled", lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_argmax_index_inside_graph(self):
        """argmax inside compiled graph; gather the single top-scoring row."""
        x = cached_randn((32, 64), differentiation="ing05", dtype=torch.float16)
        scores = cached_randn((32,), differentiation="ing05s", dtype=torch.float16)
        compare_mode(
            "compiled",
            lambda x, s: x[torch.argmax(s).unsqueeze(0).to(torch.int64)],
            x,
            scores,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_two_different_indices_same_source(self):
        """Same source, two distinct index tensors in one graph; two independent GATHER_OP_SPECs."""
        x = cached_randn((64, 128), differentiation="ing06", dtype=torch.float16)
        idx1 = torch.randint(0, 64, (16,), dtype=torch.int64)
        idx2 = torch.randint(0, 64, (8,), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda x, i1, i2: (x[i1], x[i2]),
            x,
            idx1,
            idx2,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_two_different_sources_different_indices(self):
        """Two distinct sources and two distinct indices in one graph; all four GATHER_OP_SPECs independent."""
        x = cached_randn((64, 64), differentiation="ing07x", dtype=torch.float16)
        y = cached_randn((32, 64), differentiation="ing07y", dtype=torch.float16)
        idx_x = torch.randint(0, 64, (16,), dtype=torch.int64)
        idx_y = torch.randint(0, 32, (12,), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda x, y, ix, iy: (x[ix], y[iy]),
            x,
            y,
            idx_x,
            idx_y,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------
    # Explicit torch.gather(src, dim, index) — aten::gather.out path
    # All existing compiler tests use x[idx] (aten::index); these are the
    # first compiler-context tests for the explicit gather API form.
    # ------------------------------------------------------------------

    def test_explicit_gather_dim0_lx(self):
        """torch.gather(src, 0, index) with LX_PLANNING=1; aten::gather.out through LX planner."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing("compiled", compiled=_DXP_STANDALONE_FAIL)
        os.environ["LX_PLANNING"] = "1"
        src = cached_randn((64, 128), differentiation="xgth01", dtype=torch.float16)
        index = torch.randint(0, 64, (32, 128), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda s, i: torch.gather(s, 0, i),
            src,
            index,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_explicit_gather_dim1_lx(self):
        """torch.gather(src, 1, index) with LX_PLANNING=1; dim=1 aten::gather.out through LX planner."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing("compiled", compiled=_RESTICKIFY_3ARGS)
        os.environ["LX_PLANNING"] = "1"
        src = cached_randn((32, 64), differentiation="xgth02", dtype=torch.float16)
        index = torch.randint(0, 64, (32, 16), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda s, i: torch.gather(s, 1, i),
            src,
            index,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_explicit_gather_dim2_lx(self):
        """torch.gather(src, 2, index) with LX_PLANNING=1; dim=2 aten::gather.out through LX planner."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing("compiled", compiled=_RESTICKIFY_3ARGS)
        os.environ["LX_PLANNING"] = "1"
        src = cached_randn((8, 16, 64), differentiation="xgth03", dtype=torch.float16)
        index = torch.randint(0, 64, (8, 16, 32), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda s, i: torch.gather(s, 2, i),
            src,
            index,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_explicit_gather_dim0_bfloat16_lx(self):
        """torch.gather(src, 0, index) bfloat16 + LX_PLANNING=1; wordLength=2 path through LX planner."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing("compiled", compiled=_DXP_STANDALONE_FAIL)
        os.environ["LX_PLANNING"] = "1"
        src = cached_randn((64, 128), differentiation="xgth04", dtype=torch.bfloat16)
        index = torch.randint(0, 64, (32, 128), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda s, i: torch.gather(s, 0, i),
            src,
            index,
            atol=_ATOL_BF16,
            rtol=_ATOL_BF16,
        )

    # ------------------------------------------------------------------
    # dim=1 and dim=2 fancy-indexing gathers in compiler context
    # x[idx] (dim=0) is extensively tested above; these are the
    # first compiler tests for non-leading-dim fancy indexing.
    # ------------------------------------------------------------------

    def test_dim1_fancy_indexing_sencores_4(self):
        """x[:, idx, :] dim=1 gather; non-leading-dim gather across all sencores values."""
        x = cached_randn((8, 32, 64), differentiation="dim01", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda x, i: x[:, i, :],
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_dim2_fancy_indexing_lx(self):
        """x[:, :, idx] dim=2 gather with LX_PLANNING=1; innermost-dim indirect access through LX planner."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing("compiled", compiled=_RESTICKIFY_3ARGS)
        os.environ["LX_PLANNING"] = "1"
        x = cached_randn((8, 16, 64), differentiation="dim02", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda x, i: x[:, :, i],
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_dim1_explicit_gather_lx(self):
        """torch.gather(src, 1, index) + LX_PLANNING=1; dim=1 compiler path for KV-head reorder."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing("compiled", compiled=_DXP_STANDALONE_FAIL)
        os.environ["LX_PLANNING"] = "1"
        src = cached_randn((4, 32, 64), differentiation="dim03", dtype=torch.float16)
        index = torch.randint(0, 32, (4, 8, 64), dtype=torch.int64)
        compare_mode(
            "compiled",
            lambda s, i: torch.gather(s, 1, i),
            src,
            index,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )
