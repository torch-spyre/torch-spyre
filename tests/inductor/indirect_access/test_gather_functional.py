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

"""2D–4D gather shapes, diverse (M,N) sizes, all index access patterns (ascending/descending/strided/hotspot/random/repeated/2D), and PyTorch gather API correctness."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import cached_randn  # noqa: E402
from conftest import _xfail_existing, compare_mode  # noqa: E402

_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_F32 = 1e-5

_INDEX_EAGER = (1219, "aten::index.Tensor_out is not registered on Spyre.")
_GATHER_EAGER = (4328, "aten::gather.out is not registered on Spyre.")
_REMAINDER_EAGER = (4416, "aten::remainder.Tensor_out is not registered on Spyre.")
_DXP_STANDALONE_FAIL = (
    4321,
    "compile gather DBO dxp_standalone returned non-zero exit status 1.",
)
_RESTICKIFY_3ARGS = (
    4327,
    "compile gather / index_select dim>=1 restickify op_spec has 3 args.",
)
_INCOMPATIBLE_HOST_SIZE = (
    3732,
    "compile take / flatten-take Incompatible host_size and dim_order.",
)
_WRAPPER_SYNTAX_ERROR = (
    4333,
    "Failed to import inductor wrapper inner SyntaxError: invalid syntax.",
)
_MULTIPLE_INDIRECT_SYMBOLS = (4471, "Unsupported: multiple indirect symbols in arg.")


class TestGatherShapeRankAndIndexPatterns:
    """2D/3D/4D shapes, diverse (M,N) sizes, non-power-of-2 dims, all index patterns (full-range/ascending/descending/repeated/interleaved/strided/hotspot), and torch.gather/index_select/take_along_dim API correctness."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "shape,P,diff_key",
        [
            ((64, 64), 32, "gfc01"),
            ((16, 1024), 8, "gfc02"),
            ((8, 4096), 4, "gfc03"),
            ((4, 16), 2, "gfc05"),
        ],
    )
    def test_gather_2d_basic_shapes(self, execution_mode, shape, P, diff_key):
        """Basic 2D gather across diverse (M, N) shapes and output sizes."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn(shape, differentiation=diff_key, dtype=torch.float16)
        idx = torch.randint(0, shape[0], (P,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gather_2d_large_m256(self, execution_mode, patch_sencores):
        """(256, 64) P=128 — M large enough to split across cores."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((256, 64), differentiation="gfc04", dtype=torch.float16)
        idx = torch.randint(0, 256, (128,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gather_2d_non_power_of_2_inner(self, execution_mode):
        """N=100 not a multiple of 64; stick padding must not leak."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((16, 100), differentiation="gfc06", dtype=torch.float16)
        idx = torch.randint(0, 16, (8,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gather_2d_full_range(self, execution_mode):
        """Index = full row permutation; every row accessed exactly once."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 64), differentiation="gfc07", dtype=torch.float16)
        idx = torch.randperm(32, dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gather_2d_all_same_row(self, execution_mode):
        """All idx values = 0; repeated reads of row 0."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 64), differentiation="gfc08", dtype=torch.float16)
        idx = torch.zeros(16, dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gather_2d_ascending_idx(self, execution_mode):
        """Sequential ascending idx [0..15]; cache-friendly access."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 128), differentiation="gfc09", dtype=torch.float16)
        idx = torch.arange(16, dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gather_2d_descending_idx(self, execution_mode):
        """Descending idx [31..16]; reverse access order."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 128), differentiation="gfc10", dtype=torch.float16)
        idx = torch.arange(31, 15, -1, dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gather_2d_repeated_idx(self, execution_mode):
        """idx has repeated values; duplicate output rows must be identical."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 64), differentiation="gfc11", dtype=torch.float16)
        idx = torch.tensor(
            [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9],
            dtype=torch.int64,
        )
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gather_3d_wide(self, execution_mode):
        """3D tensor (8,32,512), dim=0 gather, P=16."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((8, 32, 512), differentiation="gfc12", dtype=torch.float16)
        idx = torch.randint(0, 8, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gather_3d_small(self, execution_mode):
        """Smallest valid 3D tensor (4,4,16), sub-stick inner dims."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((4, 4, 16), differentiation="gfc13", dtype=torch.float16)
        idx = torch.randint(0, 4, (2,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_gather_3d_interior_dim(self, execution_mode):
        """x[:, idx, :] — interior dim=1 gather, (8,16,64)."""
        x = cached_randn((8, 16, 64), differentiation="gfc14", dtype=torch.float16)
        idx = torch.randint(0, 16, (8,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 1, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_4d_data(self, execution_mode):
        """4D tensor (2,8,4,64), gather at dim=0, P=4."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((2, 8, 4, 64), differentiation="gfc15", dtype=torch.float16)
        idx = torch.randint(0, 2, (4,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_index_select_new_shape(self, execution_mode):
        """torch.index_select(x,0,idx) on (16,256) with P=12."""
        x = cached_randn((16, 256), differentiation="gfc16", dtype=torch.float16)
        idx = torch.randint(0, 16, (12,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_take_along_dim(self, execution_mode):
        """torch.take_along_dim(x, idx, dim=0), same-rank index (16,64)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4416
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4333
        _xfail_existing(
            execution_mode,
            eager=_REMAINDER_EAGER,
            compiled=_WRAPPER_SYNTAX_ERROR,
        )
        x = cached_randn((32, 64), differentiation="gfc17", dtype=torch.float16)
        idx = torch.randint(0, 32, (16, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take_along_dim(x, i, dim=0),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_torch_gather_dim0(self, execution_mode):
        """torch.gather(x, 0, idx) functional form; same-rank (16,64) index."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing(
            execution_mode,
            eager=_GATHER_EAGER,
            compiled=_DXP_STANDALONE_FAIL,
        )
        x = cached_randn((32, 64), differentiation="gfc18", dtype=torch.float16)
        idx = torch.randint(0, 32, (16, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.gather(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_torch_gather_dim0_fewer_index_cols(self, execution_mode):
        """torch.gather dim=0 when index.cols < input.cols — extra #3500.

        Existing dim=0 gathers use the same inner dim for src and index.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing(
            execution_mode,
            eager=_GATHER_EAGER,
            compiled=_DXP_STANDALONE_FAIL,
        )
        x = cached_randn((4, 64), differentiation="gfc_3500", dtype=torch.float16)
        idx = torch.randint(0, 4, (4, 32), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.gather(x, 0, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_result_exact(self, execution_mode):
        """Integer-representable fp16 values (powers of 2); no rounding at output."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        vals = [float(2**k) for k in range(64)]
        x = torch.tensor([vals] * 16, dtype=torch.float16)
        idx = torch.randint(0, 16, (8,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_gather_2d_interleaved_idx(self, execution_mode):
        """Interleaved idx [0,16,1,17,...]; alternating low/high row access."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 128), differentiation="gfc20", dtype=torch.float16)
        idx = torch.tensor(
            [v for pair in zip(range(16), range(16, 32)) for v in pair],
            dtype=torch.int64,
        )
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    # ------------------------------------------------------------------

    def test_idx_hotspot_80pct(self, execution_mode):
        """80% of idx values point to the same row (heavy hotspot)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 128), differentiation="gidx02", dtype=torch.float16)
        hot = [5] * 19 + [torch.randint(0, 32, (1,)).item() for _ in range(5)]
        idx = torch.tensor(hot[:24], dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_idx_strided(self, execution_mode):
        """Strided access [0,4,8,...]; stride=4 row selection."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((64, 64), differentiation="gidx04", dtype=torch.float16)
        idx = torch.arange(0, 64, 4, dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_idx_oversize(self, execution_mode):
        """P=32 > M=16; each input row read multiple times."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((16, 64), differentiation="gidx06", dtype=torch.float16)
        idx = torch.randint(0, 16, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_idx_two_elements(self, execution_mode):
        """Minimal index P=2; smallest valid gather output."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 128), differentiation="gidx07", dtype=torch.float16)
        idx = torch.randint(0, 32, (2,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_idx_single_row(self, execution_mode):
        """P=1 — single-element index; validates compile succeeds."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 256), differentiation="gidx08", dtype=torch.float16)
        idx = torch.randint(0, 32, (1,), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_idx_2d_shape(self, execution_mode):
        """2D-shaped index (8,4) on (16,64) value tensor."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((16, 64), differentiation="gidx10", dtype=torch.float16)
        idx = torch.randint(0, 16, (8, 4), dtype=torch.int64)
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_idx_boundary_values(self, execution_mode):
        """Both boundary rows (0 and M-1=31) plus mid values."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 64), differentiation="gidx13", dtype=torch.float16)
        idx = torch.tensor(
            [0, 31, 5, 10, 15, 20, 25, 3, 8, 13, 18, 23, 28, 1, 16, 30],
            dtype=torch.int64,
        )
        compare_mode(
            execution_mode, lambda x, i: x[i], x, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    # ------------------------------------------------------------------

    def test_torch_gather_dim1(self, execution_mode):
        """torch.gather(x, 1, idx_expanded) at dim=1; selects columns per batch row."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing(
            execution_mode,
            eager=_GATHER_EAGER,
            compiled=_DXP_STANDALONE_FAIL,
        )
        x = cached_randn((8, 32, 64), differentiation="gfc_gd1", dtype=torch.float16)
        idx_raw = torch.randint(0, 32, (8, 16), dtype=torch.int64)
        idx = idx_raw.unsqueeze(-1).expand(8, 16, 64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.gather(x, 1, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_torch_gather_dim1_2d_fp16(self, execution_mode):
        """torch.gather dim=1 2-D fp16 inp (4, 64) idx (4, 32) — extra #4327.

        Existing dim=1 gathers are 3-D broadcast-expand or GQA. This is the 2-D restickify repro.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing(
            execution_mode,
            eager=_GATHER_EAGER,
            compiled=_RESTICKIFY_3ARGS,
        )
        x = (
            (torch.arange(256, dtype=torch.float32) / 7.0 - 1.5)
            .reshape(4, 64)
            .to(torch.float16)
        )
        idx = torch.randint(0, 64, (4, 32), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.gather(x, 1, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_take_along_dim_dim1(self, execution_mode):
        """torch.take_along_dim at dim=1; selects sequence positions per batch head."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4416
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4333
        _xfail_existing(
            execution_mode,
            eager=_REMAINDER_EAGER,
            compiled=_WRAPPER_SYNTAX_ERROR,
        )
        x = cached_randn((4, 32, 64), differentiation="gfc_tad1", dtype=torch.float16)
        idx = torch.randint(0, 32, (4, 16, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take_along_dim(x, i, dim=1),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_take_along_dim_none_flat(self, execution_mode):
        """take_along_dim(x_flat, idx, dim=0): flatten-then-gather (dim=None mode); API gap closure."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4416
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/3732
        _xfail_existing(
            execution_mode,
            eager=_REMAINDER_EAGER,
            compiled=_INCOMPATIBLE_HOST_SIZE,
        )
        x = cached_randn((16, 32), differentiation="gfc_tad_none", dtype=torch.float16)
        flat_idx = torch.randint(0, 16 * 32, (64,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take_along_dim(x.reshape(-1), i, dim=0),
            x,
            flat_idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_take_along_dim_float32(self, execution_mode):
        """take_along_dim at dim=1 on float32 source; selects 16 positions per head from (4,32,64)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4416
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4333
        _xfail_existing(
            execution_mode,
            eager=_REMAINDER_EAGER,
            compiled=_WRAPPER_SYNTAX_ERROR,
        )
        x = cached_randn((4, 32, 64), differentiation="tad_f32_01", dtype=torch.float32)
        idx = torch.randint(0, 32, (4, 16, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take_along_dim(x, i, dim=1),
            x,
            idx,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_take_along_dim_bfloat16(self, execution_mode):
        """take_along_dim at dim=1 on bfloat16 source; same shape as float32 variant for dtype parity."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4416
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4333
        _xfail_existing(
            execution_mode,
            eager=_REMAINDER_EAGER,
            compiled=_WRAPPER_SYNTAX_ERROR,
        )
        x = cached_randn(
            (4, 32, 64), differentiation="tad_bf16_01", dtype=torch.bfloat16
        )
        idx = torch.randint(0, 32, (4, 16, 64), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take_along_dim(x, i, dim=1),
            x,
            idx,
            atol=_ATOL_BF16,
            rtol=_ATOL_BF16,
        )

    def test_advanced_index_2d_coordinate(self, execution_mode):
        """x[idx0, idx1] two-tensor coordinate indexing; each (idx0[i], idx1[i]) pair selects one element."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4471
        _xfail_existing(
            execution_mode,
            eager=_INDEX_EAGER,
            compiled=_MULTIPLE_INDIRECT_SYMBOLS,
        )
        x = cached_randn((8, 16), differentiation="adv2d_01", dtype=torch.float16)
        idx0 = torch.tensor([2, 0, 5, 7, 1], dtype=torch.int64)
        idx1 = torch.tensor([1, 15, 3, 0, 8], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, a, b: x[a, b],
            x,
            idx0,
            idx1,
            atol=0,
            rtol=0,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestGatherNonLeadingDimEagerCompile:
    """Non-leading-dim gather (dim=1 independent index, dim=2) in separate eager and compiled records.

    The existing TestGatherShapeRankAndIndexPatterns.test_torch_gather_dim1 uses a
    broadcast-expanded index (same value tiled across the last dim).  These tests use a
    fully-independent 3-D index where every element can point to a different position —
    the pattern required for GQA head selection and feature-sparse attention.

    dim=2 gather has no coverage at all in the suite; tests here close that gap.

    Each test produces two records:
        test_name[eager]   — ATen eager dispatch path
        test_name[compiled] — torch.compile → Spyre inductor path
    A failure in one record does NOT prevent the other from running.
    """

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    # -- dim=1: fully-independent index (attention head / sequence position) --

    def test_gather_dim1_gqa_head_select(self, execution_mode):
        """torch.gather(x, 1, idx): GQA — select K=8 heads from [B=4, H=32, Dh=64].

        idx[b, k, d] independently addresses a different head for each (batch, head-out, dim)
        position.  This is the pattern broadcast-expand cannot express.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing(
            execution_mode,
            eager=_GATHER_EAGER,
            compiled=_DXP_STANDALONE_FAIL,
        )
        B, H, Dh, K = 4, 32, 64, 8
        x = cached_randn((B, H, Dh), differentiation="ndl01", dtype=torch.float16)
        idx = torch.randint(0, H, (B, K, Dh), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.gather(x, 1, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_dim1_sequence_position_select(self, execution_mode):
        """torch.gather(x, 1, idx): select S=16 positions from [B=4, L=128, D=64].

        Decode output projection pattern: pick specific token positions from a full
        sequence tensor.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing(
            execution_mode,
            eager=_GATHER_EAGER,
            compiled=_DXP_STANDALONE_FAIL,
        )
        B, L, D, S = 4, 128, 64, 16
        x = cached_randn((B, L, D), differentiation="ndl02", dtype=torch.float16)
        idx = torch.randint(0, L, (B, S, D), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.gather(x, 1, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # -- dim=2: feature / channel selection (no prior coverage in the suite) -

    def test_gather_dim2_feature_select(self, execution_mode):
        """torch.gather(x, 2, idx): select K=32 features from [B=2, H=8, F=128].

        Sparse feature selection within each attention head — top-k key channels
        for approximate attention.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing(
            execution_mode,
            eager=_GATHER_EAGER,
            compiled=_RESTICKIFY_3ARGS,
        )
        B, H, F, K = 2, 8, 128, 32
        x = cached_randn((B, H, F), differentiation="ndl05", dtype=torch.float16)
        idx = torch.randint(0, F, (B, H, K), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.gather(x, 2, i),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_take_along_dim_dim2(self, execution_mode):
        """torch.take_along_dim(x, idx, dim=2): channel selection via take_along_dim."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4416
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing(
            execution_mode,
            eager=_REMAINDER_EAGER,
            compiled=_RESTICKIFY_3ARGS,
        )
        B, H, F, K = 2, 8, 128, 32
        x = cached_randn((B, H, F), differentiation="ndl07", dtype=torch.float16)
        idx = torch.randint(0, F, (B, H, K), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take_along_dim(x, i, dim=2),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_dim2_bfloat16(self, execution_mode):
        """torch.gather(x, 2, idx) on bfloat16 source; validates wordLength=2 path."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing(
            execution_mode,
            eager=_GATHER_EAGER,
            compiled=_RESTICKIFY_3ARGS,
        )
        B, H, F, K = 2, 4, 64, 16
        x = cached_randn((B, H, F), differentiation="ndl08", dtype=torch.bfloat16)
        idx = torch.randint(0, F, (B, H, K), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.gather(x, 2, i),
            x,
            idx,
            atol=2e-2,
            rtol=2e-2,
        )
