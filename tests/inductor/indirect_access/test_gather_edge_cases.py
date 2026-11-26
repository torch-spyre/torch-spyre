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

"""Boundary conditions, NaN/Inf source values, non-contiguous index tensors, mask-based gather, all gather API forms, non-contiguous sources, known bug regressions (#2662/#2661/#2654/#2653), and numeric correctness checks."""

import pytest
import torch
import torch.nn.functional as F

from utils_inductor import DEVICE, cached_randn
from conftest import _xfail_existing, compare_mode, rejection

# fp16 compute after a gather (unary, binary, small reductions).  The largest error
# measured against an fp32 reference is 5.1e-3 (fp16's own compute error: up to 1.7e-3),
# so 1e-2 leaves about 2x margin.  Pure data movement uses atol=0 instead.
_TOL_F16 = 1e-2
# bf16 compute after a gather.  bf16 keeps 8 significant bits (one ulp ~ 4e-3); the
# largest error measured against an fp32 reference is 7.1e-3 (bf16's own compute
# error: up to 3.0e-3), so 2e-2 leaves about 2.8x margin.
_TOL_BF16 = 2e-2
# fp32 compute.  The largest error measured against an fp32 CPU reference is 4.3e-7
# (layer norm).
_TOL_F32 = 1e-5

_INDEX_2D_EAGER = rejection(
    6335,
    "Eager Spyre aten::index.Tensor supports only x[index] with a single 1-D int32/int64 index tensor on dim 0.",
    r"supports only x\[index\] with a single 1-D int32/int64 index tensor on dim 0",
)
_BOOL_SOURCE_EAGER = rejection(
    4332,
    "Eager Spyre aten::index.Tensor does not support bool source tensors.",
    r"does not support bool source tensors",
)
_EAGER_MATMUL_MISMATCH = (
    6355,
    "Eager FP16 matmul after a gather mismatches CPU.",
    r"eager spyre <-> (precomputed )?cpu\S*( ref)? mismatch\s+Tensor-likes are not (?:close|equal)!",
)
_NUMERICAL_MISMATCH = (
    4723,
    "Eager gather followed by a sum accumulate mismatches CPU.",
    r"spyre <-> cpu\S* mismatch\s+Tensor-likes are not (?:close|equal)!",
)
_CLAMP_MIN_EAGER = rejection(
    6340,
    "aten::clamp_min.out is not registered on Spyre.",
    r"Could not run 'aten::clamp_min.out'",
)
_STICK_INCOMPATIBLE = rejection(
    3732,
    "compile single-row gather: Pointwise no mechanism to resolve stick incompatibility.",
    r"no mechanism to resolve stick incompatibility",
)
_MASKED_SELECT = rejection(
    4371,
    "aten::masked_select is not registered on Spyre.",
    r"Could not run 'aten::masked_select'",
)
_NONZERO = rejection(
    6335,
    "Eager Spyre aten::index.Tensor supports only x[index] with a single 1-D int32/int64 index tensor on dim 0.",
    r"supports only x\[index\] with a single 1-D int32/int64 index tensor on dim 0",
)
_BOOL_OPERANDS = rejection(
    4332,
    "torch.bool result from operands with device format(s) {SEN169_FP16, IEEE_INT32}.",
    r"torch\.bool result from operands with device format",
)
_RESTICKIFY_3ARGS = rejection(
    4327,
    "compile gather / index_select / take_along_dim at dim>=1: cannot re-tile the source, no size-1 stick target (the issue reported an 'op_spec has 3 args' abort).",
    r"cannot re-tile \S+: no size-1 stick target",
)
_GATHER_STICK_MISMATCH = (
    4327,
    "compile gather at dim>=1 compiles and returns values that differ from CPU "
    "(the issue reported an 'op_spec has 3 args' abort, then a 'cannot re-tile' error).",
    r"compiled spyre <-> cpu\S* mismatch\s+Tensor-likes are not (?:close|equal)!",
)
_TAKE_EAGER = rejection(
    4307,
    "aten::take is not registered on Spyre.",
    r"Could not run 'aten::take'",
)
_TAKE_FLAT_RESTICKIFY = rejection(
    4327,
    "compile take on a flattened 2-D source: AssertionError restickify op_spec has 3 args.",
    r"restickify op_spec has 3 args",
)
_REMAINDER = rejection(
    4416,
    "aten::remainder.Tensor_out is not registered on Spyre.",
    r"Could not run 'aten::remainder.Tensor_out'",
)
_REMAINDER_COMPILED = rejection(
    4416,
    "compile take_along_dim: fused gather kernel contains the unimplemented operation remainder.",
    r"unimplemented operation remainder",
)
_GATHER_OUT = rejection(
    4328,
    "aten::gather.out is not registered on Spyre.",
    r"Could not run 'aten::gather.out'",
)
_DXP_STANDALONE = (
    4321,
    "compile gather dxp_standalone returned non-zero exit status 1.",
    r"SubprocException",
)
_MATMUL_MISMATCH = (
    4722,
    "compile gather followed by matmul / linear numerical mismatch.",
    r"spyre <-> cpu\S* mismatch\s+Tensor-likes are not (?:close|equal)!",
)
_ACCUMULATE_MISMATCH = (
    4723,
    "compile gather accumulate numerical mismatch.",
    r"spyre <-> cpu\S* mismatch\s+Tensor-likes are not (?:close|equal)!",
)


class TestGatherBoundaryNumericsAndAPIForms:
    """Boundary conditions (single row/element, P>M, D=1), NaN/Inf source values, non-contiguous index tensors, mask-based gather (masked_select/bool/nonzero), all gather API forms, non-contiguous sources, bug regressions (#2662/#2661/#2654/#2653), and numeric correctness."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    # ------------------------------------------------------------------

    def test_single_row_source(self, execution_mode):
        """Source has a single row (M=1); all indices must be 0."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/3732
        _xfail_existing(execution_mode, compiled=_STICK_INCOMPATIBLE)
        x = cached_randn((1, 64), differentiation="ec01", dtype=torch.float16)
        idx = torch.zeros(8, dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_all_same_index(self, execution_mode):
        """All indices identical; output is repeated rows."""
        x = cached_randn((64, 128), differentiation="ec04", dtype=torch.float16)
        idx = torch.full((16,), 7, dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_odd_inner_dim(self, execution_mode):
        """Odd D=65; cross-stick boundary element."""
        x = cached_randn((64, 65), differentiation="ec10", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_gather_1d_source(self, execution_mode):
        """1D source tensor; gather is scalar indexing."""
        x = cached_randn((256,), differentiation="ec12", dtype=torch.float16)
        idx = torch.randint(0, 256, (32,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_exact_integer_gather(self, execution_mode):
        """Integer value source; output is exact (atol=0)."""
        x = torch.randint(-500, 500, (32, 64), dtype=torch.int32)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    # ------------------------------------------------------------------

    def test_masked_select_1d(self, execution_mode):
        """torch.masked_select on 1D tensor; positive elements only."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4371
        _xfail_existing(execution_mode, eager=_MASKED_SELECT, compiled=_MASKED_SELECT)
        x = cached_randn((64,), differentiation="msk01", dtype=torch.float16)
        mask = x > 0
        compare_mode(
            execution_mode,
            lambda x, m: torch.masked_select(x, m),
            x,
            mask,
            atol=0,
            rtol=0,
        )

    def test_masked_select_2d(self, execution_mode):
        """torch.masked_select on 2D tensor; positive elements flattened."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4371
        _xfail_existing(execution_mode, eager=_MASKED_SELECT, compiled=_MASKED_SELECT)
        x = cached_randn((8, 32), differentiation="msk02", dtype=torch.float16)
        mask = x > 0
        compare_mode(
            execution_mode,
            lambda x, m: torch.masked_select(x, m),
            x,
            mask,
            atol=0,
            rtol=0,
        )

    def test_bool_index_gather_1d(self, execution_mode):
        """x[mask] bool indexing on 1D; selects positive-valued elements."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4498
        _xfail_existing(execution_mode, eager=_NONZERO, compiled=_NONZERO)
        x = cached_randn((64,), differentiation="msk03", dtype=torch.float16)
        mask = x > 0
        compare_mode(execution_mode, lambda x, m: x[m], x, mask, atol=0, rtol=0)

    def test_bool_index_gather_2d(self, execution_mode):
        """x[mask] bool indexing on 2D dim=0; selects rows where mask is True."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4498
        _xfail_existing(execution_mode, eager=_NONZERO, compiled=_NONZERO)
        x = cached_randn((32, 64), differentiation="msk04", dtype=torch.float16)
        mask = torch.randint(0, 2, (32,)).bool()
        compare_mode(execution_mode, lambda x, m: x[m], x, mask, atol=0, rtol=0)

    def test_nonzero_gather(self, execution_mode):
        """torch.nonzero → index; gather rows at nonzero scalar positions."""
        x = cached_randn((64, 32), differentiation="msk05", dtype=torch.float16)
        scores = torch.randint(0, 10, (64,), dtype=torch.int64)
        nz = torch.nonzero(scores, as_tuple=True)[0].int()
        if nz.numel() == 0:
            nz = torch.tensor([0], dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, nz, atol=0, rtol=0)

    def test_mask_then_index_gather(self, execution_mode):
        """Create index from mask; use as int gather index."""
        src = cached_randn((64, 128), differentiation="msk06", dtype=torch.float16)
        scores = torch.randn(64)
        keep_mask = scores > 0
        idx = torch.where(keep_mask)[0].int()
        if idx.numel() == 0:
            idx = torch.tensor([0], dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], src, idx, atol=0, rtol=0)

    def test_causal_attention_mask(self, execution_mode):
        """Causal mask applied to scores then gather valid row subset."""
        scores = cached_randn((32, 64), differentiation="msk09", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda s, i: torch.where(s[i] > 0, s[i], torch.zeros_like(s[i])),
            scores,
            idx,
            atol=0,
            rtol=0,
        )

    def test_gather_masked_output(self, execution_mode):
        """Gather + apply output mask for padding suppression."""
        x = cached_randn((64, 128), differentiation="msk10", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        out_mask = torch.randint(0, 2, (16, 128)).bool()
        compare_mode(
            execution_mode,
            lambda x, i, m: x[i].masked_fill(m, 0.0),
            x,
            idx,
            out_mask,
            atol=0,
            rtol=0,
        )

    def test_where_based_gather(self, execution_mode):
        """torch.where → conditional gather between two sources."""
        a = cached_randn((32, 64), differentiation="msk11a", dtype=torch.float16)
        b = cached_randn((32, 64), differentiation="msk11b", dtype=torch.float16)
        cond = torch.randint(0, 2, (32, 64)).bool()
        compare_mode(
            execution_mode,
            lambda a, b, c: torch.where(c, a, b),
            a,
            b,
            cond,
            atol=0,
            rtol=0,
        )

    def test_masked_select_bfloat16(self, execution_mode):
        """masked_select on bfloat16 tensor; gather positive-valued elements."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4371
        _xfail_existing(execution_mode, eager=_MASKED_SELECT, compiled=_MASKED_SELECT)
        x = cached_randn((64,), differentiation="msk12", dtype=torch.bfloat16)
        mask = x > 0
        compare_mode(
            execution_mode,
            lambda x, m: torch.masked_select(x, m),
            x,
            mask,
            atol=0,
            rtol=0,
        )

    def test_gather_from_bool_tensor(self, execution_mode):
        """Gather from a bool tensor; exact output."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4332
        _xfail_existing(
            execution_mode, eager=_BOOL_SOURCE_EAGER, compiled=_BOOL_OPERANDS
        )
        x = torch.randint(0, 2, (64, 32)).bool()
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    # ------------------------------------------------------------------

    def test_index_select_dim1(self, execution_mode):
        """index_select at dim=1; column selection."""
        x = cached_randn((32, 128), differentiation="idxs02", dtype=torch.float16)
        idx = torch.randint(0, 128, (64,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 1, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_select_3d_dim0(self, execution_mode):
        """index_select on 3D tensor at dim=0."""
        x = cached_randn((64, 8, 128), differentiation="idxs03", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_select_single_index(self, execution_mode):
        """index_select with P=1; single-row selection."""
        x = cached_randn((64, 128), differentiation="idxs04", dtype=torch.float16)
        idx = torch.tensor([42], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_select_bfloat16(self, execution_mode):
        """index_select on bfloat16; SDSC wordLength=2."""
        x = cached_randn((64, 128), differentiation="idxs07", dtype=torch.bfloat16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_select_int32_result(self, execution_mode):
        """index_select on integer tensor; exact output."""
        x = torch.randint(-1000, 1000, (32, 64), dtype=torch.int32)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_select_4d(self, execution_mode):
        """index_select on 4D tensor at dim=0."""
        x = cached_randn((64, 4, 8, 32), differentiation="idxs09", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_select_with_downstream(self, execution_mode):
        """index_select + relu downstream fused."""
        x = cached_randn((64, 128), differentiation="idxs11", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.relu(torch.index_select(x, 0, i)),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_select_float32(self, execution_mode):
        """index_select on float32; 4 bytes/elem."""
        x = cached_randn((32, 64), differentiation="idxs12", dtype=torch.float32)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_select_fp32_dim1(self, execution_mode):
        """index_select dim=1 on (4, 64, 128) float32 — extra vs dim=0 float32 (#1796)."""
        x = cached_randn(
            (4, 64, 128), differentiation="idxs_f32_d1", dtype=torch.float32
        )
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 1, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_select_large(self, execution_mode, patch_sencores):
        """Large index_select (M=4096, P=512)."""
        x = cached_randn((4096, 64), differentiation="idxs13", dtype=torch.float16)
        idx = torch.randint(0, 4096, (512,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    # ------------------------------------------------------------------

    def test_api_torch_take(self, execution_mode):
        """torch.take(x, flat_idx); flat index into flattened tensor."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4307
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing(execution_mode, eager=_TAKE_EAGER, compiled=_RESTICKIFY_3ARGS)
        x = cached_randn((64, 128), differentiation="api05", dtype=torch.float16)
        flat_idx = torch.randint(0, 64 * 128, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take(x, i),
            x,
            flat_idx,
            atol=0,
            rtol=0,
        )

    def test_api_repeat_interleave(self, execution_mode):
        """torch.repeat_interleave row-level expansion."""
        x = cached_randn((16, 64), differentiation="api06", dtype=torch.float16)
        compare_mode(
            execution_mode,
            lambda x: torch.repeat_interleave(x, 2, dim=0),
            x,
            atol=0,
            rtol=0,
        )

    def test_api_fullgraph(self, execution_mode):
        """fullgraph=True compile; no graph breaks; eager path also correct."""
        x = cached_randn((64, 128), differentiation="api09", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        if execution_mode == "eager":
            compare_mode(
                execution_mode,
                lambda x, i: x[i],
                x,
                idx,
                atol=0,
                rtol=0,
            )
        else:
            fn = torch.compile(lambda x, i: x[i], fullgraph=True)
            result = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
            torch.testing.assert_close(result, x[idx], atol=_TOL_F16, rtol=_TOL_F16)

    def test_api_take_along_dim_argmax(self, execution_mode):
        """take_along_dim(x, argmax, dim=0); pick argmax rows."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4416
        _xfail_existing(execution_mode, eager=_REMAINDER, compiled=_REMAINDER_COMPILED)
        x = cached_randn((32, 64), differentiation="api18", dtype=torch.float16)
        idx = x.argmax(dim=0, keepdim=True)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take_along_dim(x, i, dim=0),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_api_slice_and_gather(self, execution_mode):
        """Slice + gather: first half of rows then index."""
        x = cached_randn((128, 64), differentiation="api19", dtype=torch.float16)
        half = x[:64]
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], half, idx, atol=0, rtol=0)

    def test_api_repeat_interleave_gather(self, execution_mode):
        """repeat_interleave to expand batch then gather."""
        x = cached_randn((8, 64), differentiation="api20", dtype=torch.float16)
        expanded = x.repeat_interleave(4, dim=0)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x[i],
            expanded,
            idx,
            atol=0,
            rtol=0,
        )

    def test_api_gather_then_softmax(self, execution_mode):
        """Gather logits for target tokens → softmax over vocab."""
        logits = cached_randn((32, 512), differentiation="api22", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.softmax(x[i].float(), dim=-1).half(),
            logits,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    # ------------------------------------------------------------------

    def test_ncs_transposed(self, execution_mode):
        """Source is transposed then gather on new dim=0."""
        x = cached_randn((128, 64), differentiation="ncs01", dtype=torch.float16)
        xt = x.t().contiguous()
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], xt, idx, atol=0, rtol=0)

    def test_ncs_permuted_3d(self, execution_mode):
        """3D tensor permuted (1,0,2) then contiguous; gather at dim=0."""
        x = cached_randn((8, 64, 128), differentiation="ncs03", dtype=torch.float16)
        xp = x.permute(1, 0, 2).contiguous()
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], xp, idx, atol=0, rtol=0)

    def test_ncs_expanded_dims(self, execution_mode):
        """Source with expanded singleton dim; gather after expand."""
        x = cached_randn((1, 64, 128), differentiation="ncs05", dtype=torch.float16)
        xe = x.expand(8, 64, 128).reshape(512, 128).contiguous()
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], xe, idx, atol=0, rtol=0)

    def test_ncs_after_view(self, execution_mode):
        """Source viewed to new shape; gather on re-shaped tensor."""
        x = cached_randn((8, 64, 128), differentiation="ncs06", dtype=torch.float16)
        xv = x.reshape(512, 128)
        idx = torch.randint(0, 512, (64,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], xv, idx, atol=0, rtol=0)

    def test_ncs_chunk_then_gather(self, execution_mode):
        """Source split into 4 chunks; gather on third chunk."""
        x = cached_randn((256, 64), differentiation="ncs08", dtype=torch.float16)
        chunk = torch.chunk(x, 4, dim=0)[2].contiguous()
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x[i],
            chunk,
            idx,
            atol=0,
            rtol=0,
        )

    def test_ncs_cat_then_gather(self, execution_mode):
        """Concat two sources; gather on concatenated tensor."""
        a = cached_randn((32, 64), differentiation="ncs09a", dtype=torch.float16)
        b = cached_randn((32, 64), differentiation="ncs09b", dtype=torch.float16)
        ab = torch.cat([a, b], dim=0)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], ab, idx, atol=0, rtol=0)

    def test_ncs_stack_then_gather(self, execution_mode):
        """Stack 32 row vectors; gather rows from stacked tensor."""
        base = cached_randn((32, 128), differentiation="ncs_stk01", dtype=torch.float16)
        x = torch.stack(list(base.unbind(0)), dim=0)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    # ------------------------------------------------------------------

    def test_gather_then_matmul_downstream_correct(self, execution_mode):
        """Gather rows then matmul in one compiled graph; downstream op matches CPU."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/6355
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4722
        _xfail_existing(
            execution_mode, eager=_EAGER_MATMUL_MISMATCH, compiled=_MATMUL_MISMATCH
        )
        x = cached_randn((64, 128), differentiation="bug03x", dtype=torch.float16)
        w = cached_randn((128, 64), differentiation="bug03w", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, w, i: x[i] @ w,
            x,
            w,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_two_gathers_sequential_independent_correct(self, execution_mode):
        """Two gather ops as separate calls; each result matches CPU independently."""
        x = cached_randn((64, 128), differentiation="bug06a", dtype=torch.float16)
        y = cached_randn((64, 128), differentiation="bug06b", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)
        compare_mode(execution_mode, lambda y, i: y[i], y, idx, atol=0, rtol=0)

    # ------------------------------------------------------------------

    def test_cor_gather_zero_inner(self, execution_mode):
        """Inner dim D=1; minimal column width."""
        x = cached_randn((64, 1), differentiation="cor02", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_cor_bfloat16_large_index(self, execution_mode, patch_sencores):
        """bfloat16 source; large P=256 index."""
        x = cached_randn((512, 64), differentiation="cor03", dtype=torch.bfloat16)
        idx = torch.randint(0, 512, (256,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_cor_float32_exact(self, execution_mode):
        """float32 round-trip; values preserved exactly."""
        x = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32)
        idx = torch.arange(0, 64, 2, dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_cor_two_sources_same_index(self, execution_mode):
        """Same index applied to two source tensors independently."""
        a = cached_randn((64, 128), differentiation="cor05a", dtype=torch.float16)
        b = cached_randn((64, 128), differentiation="cor05b", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda a, b, i: a[i] + b[i],
            a,
            b,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_cor_inner_dim_multiple_sticks(self, execution_mode):
        """D=512 (8 sticks at fp16); wide row gather."""
        x = cached_randn((64, 512), differentiation="cor08", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_cor_gather_then_norm(self, execution_mode):
        """Gather + F.normalize downstream."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/6340
        _xfail_existing(execution_mode, eager=_CLAMP_MIN_EAGER)
        x = cached_randn((64, 128), differentiation="cor09", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: F.normalize(x[i].float(), dim=-1).half(),
            x,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_cor_gather_3d_non_leading(self, execution_mode):
        """3D gather at dim=1 (not outermost); batch×seq indexing."""
        x = cached_randn((4, 64, 128), differentiation="cor11", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 1, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_cor_gather_with_scale(self, execution_mode):
        """Gather + scale factor; linear downstream correctness."""
        x = cached_randn((64, 128), differentiation="cor12", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x[i] * 0.5,
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_cor_gather_accumulate(self, execution_mode):
        """Gather + sum accumulate; output is scalar per batch."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4723
        _xfail_existing(
            execution_mode, eager=_NUMERICAL_MISMATCH, compiled=_ACCUMULATE_MISMATCH
        )
        x = cached_randn((64, 128), differentiation="cor13", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x[i].sum(dim=-1),
            x,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    # ------------------------------------------------------------------

    def test_nan_propagation_in_gathered_rows(self, execution_mode):
        """Gather from source with NaN at specific rows; NaN must propagate to output."""
        x = torch.randn(32, 64, dtype=torch.float16)
        x[5, :] = float("nan")
        x[20, :] = float("nan")
        idx = torch.tensor([5, 0, 20, 10], dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_neg_inf_padding_rows_gathered(self, execution_mode):
        """-inf padding rows (attention masking pattern); gather must preserve -inf."""
        x = cached_randn((32, 64), differentiation="nan02", dtype=torch.float16)
        x[0, :] = float("-inf")
        x[31, :] = float("-inf")
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_pos_inf_source_rows(self, execution_mode):
        """+inf values in source row; gather propagates +inf to all output copies."""
        x = cached_randn((32, 64), differentiation="nan03", dtype=torch.float16)
        x[7, :] = float("inf")
        idx = torch.tensor([7, 7, 0, 1, 7], dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_mixed_nan_neg_inf_source(self, execution_mode):
        """Source with both NaN and -inf rows; gather reads each type correctly."""
        x = cached_randn((32, 64), differentiation="nan04", dtype=torch.float16)
        x[0, :] = float("nan")
        x[31, :] = float("-inf")
        idx = torch.tensor([0, 31, 5, 15], dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_nan_in_unselected_rows_does_not_contaminate(self, execution_mode):
        """NaN in rows NOT indexed; selected output rows must be free of NaN."""
        x = cached_randn((32, 64), differentiation="nan05", dtype=torch.float16)
        x[10, :] = float("nan")
        x[20, :] = float("nan")
        idx = torch.arange(8, dtype=torch.int64)
        result_ref = x[idx]
        assert not result_ref.isnan().any()
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_causal_neg_inf_mask_gather_valid_rows(self, execution_mode):
        """Causal mask: future-position rows are -inf; gather reads only valid past rows."""
        x = cached_randn((64, 128), differentiation="nan06", dtype=torch.float16)
        x[32:, :] = float("-inf")
        idx = torch.arange(32, dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    # ------------------------------------------------------------------

    def test_noncontig_idx_stride2(self, execution_mode):
        """Non-contiguous index tensor with stride=2; every other element selected."""
        x = cached_randn((32, 128), differentiation="ncidx01", dtype=torch.float16)
        idx_full = torch.randint(0, 32, (32,), dtype=torch.int64)
        idx = idx_full[::2]
        assert not idx.is_contiguous()
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_noncontig_2d_idx_transposed(self, execution_mode):
        """Transposed 2D index tensor; shape (8,4) with non-unit strides."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/6335
        _xfail_existing(execution_mode, eager=_INDEX_2D_EAGER)
        x = cached_randn((32, 64), differentiation="ncidx04", dtype=torch.float16)
        idx_2d = torch.randint(0, 32, (4, 8), dtype=torch.int64)
        idx = idx_2d.T
        assert not idx.is_contiguous()
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    # ------------------------------------------------------------------

    def test_empty_index_zero_output(self, execution_mode):
        """P=0 empty index; output shape must be (0, D) with no elements accessed."""
        x = cached_randn((32, 64), differentiation="ec_empty01", dtype=torch.float16)
        idx = torch.zeros(0, dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_minimal_gather_m1_p1_d1(self, execution_mode):
        """Absolute minimum: M=1, P=1, D=1; scalar gather from single-element source."""
        x = torch.tensor([[42.0]], dtype=torch.float16)
        idx = torch.tensor([0], dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    @pytest.mark.skip(
        reason="Negative int64 indices reach the hardware gather un-normalized: \
        bad response blocks / ComputeHardwareError, then StreamInErrorState poisons the session. See issue #4731."
    )
    def test_negative_index_int64(self, execution_mode):
        """Negative int64 index values; PyTorch defines x[tensor([-1])] as the last row."""
        x = cached_randn((16, 64), differentiation="neg_idx01", dtype=torch.float16)
        idx = torch.tensor([-1, -2, 0, 5, -1], dtype=torch.int64)
        compare_mode(execution_mode, lambda x, i: x[i], x, idx, atol=0, rtol=0)

    def test_index_select_dim2_on_4d(self, execution_mode):
        """index_select at dim=2 on (2,4,32,64); head-level sequence position selection."""
        x = cached_randn(
            (2, 4, 32, 64), differentiation="ec_dim2_01", dtype=torch.float16
        )
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 2, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_select_negative_dim(self, execution_mode):
        """index_select with dim=-1 (negative dim normalization); selects 3 of 32 columns from (8,32)."""
        x = cached_randn((8, 32), differentiation="isel_neg01", dtype=torch.float16)
        idx = torch.tensor([2, 0, 2], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, -1, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_index_select_empty_index(self, execution_mode):
        """index_select with empty index tensor; output has zero rows on dim=0."""
        x = cached_randn((8, 32), differentiation="isel_emp01", dtype=torch.float16)
        idx = torch.tensor([], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.index_select(x, 0, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_take_float32(self, execution_mode):
        """torch.take flat-index on float32 source; selects 3 elements from 12-element flat view."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4307
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing(
            execution_mode, eager=_TAKE_EAGER, compiled=_TAKE_FLAT_RESTICKIFY
        )
        x = cached_randn((3, 4), differentiation="take_f32_01", dtype=torch.float32)
        idx = torch.tensor([0, 5, 11], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take(x, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_take_bfloat16(self, execution_mode):
        """torch.take flat-index on bfloat16 source; validates 2-byte word flat indexing."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4307
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing(
            execution_mode, eager=_TAKE_EAGER, compiled=_TAKE_FLAT_RESTICKIFY
        )
        x = cached_randn((3, 4), differentiation="take_bf16_01", dtype=torch.bfloat16)
        idx = torch.tensor([0, 5, 11], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take(x, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_take_duplicate_flat_index(self, execution_mode):
        """torch.take with duplicate flat indices; same flat position selected multiple times."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4307
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing(
            execution_mode, eager=_TAKE_EAGER, compiled=_TAKE_FLAT_RESTICKIFY
        )
        x = cached_randn((3, 4), differentiation="take_dup_01", dtype=torch.float16)
        idx = torch.tensor([0, 0, 5], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take(x, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_take_empty_flat_index(self, execution_mode):
        """torch.take with empty index tensor; output is a zero-element 1-D tensor."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4307
        _xfail_existing(execution_mode, eager=_TAKE_EAGER)
        x = cached_randn((3, 4), differentiation="take_emp_01", dtype=torch.float16)
        idx = torch.tensor([], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take(x, i),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_take_rank_equal_1d(self, execution_mode):
        """take 1-D input × 1-D index; output shape follows index (#4306 Case A).

        Existing take tests use 2-D source + 1-D index (input.rank > index.rank).
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4307
        _xfail_existing(execution_mode, eager=_TAKE_EAGER)
        inp = torch.arange(64, dtype=torch.float16) / 7.0 - 1.5
        idx = torch.tensor([0, 16, 32, 63], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take(x, i),
            inp,
            idx,
            atol=0,
            rtol=0,
        )

    def test_take_rank_lesser_2d_index(self, execution_mode):
        """take 1-D input × 2-D index; output is 2-D (#4306 Case B)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4307
        _xfail_existing(execution_mode, eager=_TAKE_EAGER)
        inp = torch.arange(256, dtype=torch.float16) / 7.0 - 1.5
        idx = torch.tensor([[0, 64], [128, 255]], dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.take(x, i),
            inp,
            idx,
            atol=0,
            rtol=0,
        )

    def test_masked_select_float32(self, execution_mode):
        """masked_select on float32 tensor; gathers elements where alternating mask is True."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4371
        _xfail_existing(execution_mode, eager=_MASKED_SELECT, compiled=_MASKED_SELECT)
        x = cached_randn((3, 4), differentiation="msel_f32_01", dtype=torch.float32)
        mask = torch.arange(12).reshape(3, 4) % 2 == 0
        compare_mode(
            execution_mode,
            lambda x, m: torch.masked_select(x, m),
            x,
            mask,
            atol=_TOL_F32,
            rtol=_TOL_F32,
        )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
class TestGatherOutParameterEagerCompile:
    """torch.gather out= pre-allocated output buffer in separate eager and compiled records.

    The out= parameter allows callers to provide a pre-allocated output tensor so the
    kernel writes into existing memory rather than allocating a new buffer.  This is
    important for latency-sensitive inference pipelines where every allocation on the
    hot path matters.

    No existing test in any indirect_access file covers this API form.

    Each test produces two records:
        test_name[eager]    — ATen eager dispatch path
        test_name[compiled] — torch.compile → Spyre inductor path
    A failure in one does NOT prevent the other from running.
    """

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    def test_gather_out_dim0_basic(self, execution_mode):
        """torch.gather(x, 0, idx, out=buf): out= with leading-dim gather [M=128, N=64, P=32]."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing(execution_mode, eager=_GATHER_OUT, compiled=_DXP_STANDALONE)
        M, N, P = 128, 64, 32
        x = cached_randn((M, N), differentiation="out01", dtype=torch.float16)
        idx = torch.randint(0, M, (P, N), dtype=torch.int64)

        def fn(x, idx):
            out = torch.empty(P, N, dtype=torch.float16, device=x.device)
            return torch.gather(x, 0, idx, out=out)

        compare_mode(execution_mode, fn, x, idx, atol=_TOL_F16, rtol=_TOL_F16)

    def test_gather_out_dim1(self, execution_mode):
        """torch.gather(x, 1, idx, out=buf): out= with dim=1 gather for GQA head selection."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing(execution_mode, eager=_GATHER_OUT, compiled=_DXP_STANDALONE)
        B, H, D, K = 2, 32, 64, 8
        x = cached_randn((B, H, D), differentiation="out02", dtype=torch.float16)
        idx = torch.randint(0, H, (B, K, D), dtype=torch.int64)

        def fn(x, idx):
            out = torch.empty(B, K, D, dtype=torch.float16, device=x.device)
            return torch.gather(x, 1, idx, out=out)

        compare_mode(execution_mode, fn, x, idx, atol=_TOL_F16, rtol=_TOL_F16)

    def test_gather_out_dim2(self, execution_mode):
        """torch.gather(x, 2, idx, out=buf): out= with dim=2 feature selection."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing(
            execution_mode, eager=_GATHER_OUT, compiled=_GATHER_STICK_MISMATCH
        )
        B, H, F, K = 2, 8, 128, 32
        x = cached_randn((B, H, F), differentiation="out03", dtype=torch.float16)
        idx = torch.randint(0, F, (B, H, K), dtype=torch.int64)

        def fn(x, idx):
            out = torch.empty(B, H, K, dtype=torch.float16, device=x.device)
            return torch.gather(x, 2, idx, out=out)

        compare_mode(execution_mode, fn, x, idx, atol=_TOL_F16, rtol=_TOL_F16)

    def test_gather_out_dim0_single_row(self, execution_mode):
        """out= with P=1 single-output gather; boundary case for buffer allocation."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing(execution_mode, eager=_GATHER_OUT, compiled=_DXP_STANDALONE)
        M, N = 64, 128
        x = cached_randn((M, N), differentiation="out04", dtype=torch.float16)
        idx = torch.randint(0, M, (1, N), dtype=torch.int64)

        def fn(x, idx):
            out = torch.empty(1, N, dtype=torch.float16, device=x.device)
            return torch.gather(x, 0, idx, out=out)

        compare_mode(execution_mode, fn, x, idx, atol=_TOL_F16, rtol=_TOL_F16)

    def test_gather_out_dim0_reused_buffer(self, execution_mode):
        """out= buffer passed to two consecutive gather calls; second call overwrites first."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing(execution_mode, eager=_GATHER_OUT, compiled=_DXP_STANDALONE)
        M, N, P = 64, 32, 16
        x = cached_randn((M, N), differentiation="out05", dtype=torch.float16)
        idx1 = torch.randint(0, M, (P, N), dtype=torch.int64)
        idx2 = torch.randint(0, M, (P, N), dtype=torch.int64)

        def fn(x, idx1, idx2):
            out = torch.empty(P, N, dtype=torch.float16, device=x.device)
            torch.gather(x, 0, idx1, out=out)
            return torch.gather(x, 0, idx2, out=out)

        compare_mode(execution_mode, fn, x, idx1, idx2, atol=_TOL_F16, rtol=_TOL_F16)

    def test_gather_out_dim0_bfloat16(self, execution_mode):
        """out= gather on bfloat16 source; validates wordLength=2 on the out= path."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4321
        _xfail_existing(execution_mode, eager=_GATHER_OUT, compiled=_DXP_STANDALONE)
        M, N, P = 64, 64, 16
        x = cached_randn((M, N), differentiation="out06", dtype=torch.bfloat16)
        idx = torch.randint(0, M, (P, N), dtype=torch.int64)

        def fn(x, idx):
            out = torch.empty(P, N, dtype=torch.bfloat16, device=x.device)
            return torch.gather(x, 0, idx, out=out)

        compare_mode(execution_mode, fn, x, idx, atol=_TOL_BF16, rtol=_TOL_BF16)
