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

"""Sequential and composed gather chains: gather→gather, gather→matmul, RoPE cos/sin lookup, K/V split, chunked prefill, RoPE interleaved (GPT-J/NeoX), and end-to-end prefill/decode/beam/speculative pipelines."""

import pytest
import torch
import torch.nn.functional as F

from utils_inductor import cached_randn
from conftest import _xfail_existing, compare_mode, rejection, gap_one_core

# fp16 compute after a gather (unary, binary, small reductions).  The largest error
# measured against an fp32 reference is 5.1e-3 (fp16's own compute error: up to 1.7e-3),
# so 1e-2 leaves about 2x margin.  Pure data movement uses atol=0 instead.
_TOL_F16 = 1e-2
# bf16 compute after a gather.  bf16 keeps 8 significant bits (one ulp ~ 4e-3); the
# largest error measured against an fp32 reference is 7.1e-3 (bf16's own compute
# error: up to 3.0e-3), so 2e-2 leaves about 2.8x margin.
_TOL_BF16 = 2e-2


_GATHER_EAGER = rejection(
    4328,
    "aten::gather.out is not registered on Spyre.",
    r"Could not run 'aten::gather.out'",
)
_SCATTER_ADD_EAGER = rejection(
    4396,
    "aten::scatter_add.out is not registered on Spyre.",
    r"Could not run 'aten::scatter_add.out'",
)
_SLICE_OFFSET_ALIGNMENT = rejection(
    5383,
    "compile copy_from_d2d of a sliced view requires a stick-aligned storage_offset.",
    r"copy_from_d2d of a sliced view requires a stick-aligned storage_offset",
)
_POINTWISE_NO_LAYOUT = rejection(
    6388,
    "compile RoPE: cos/sin gathered from a cache and sliced feed pointwise ops: Multi-arg pointwise has no supported output layout.",
    r"Multi-arg pointwise .*no supported output layout found",
)
_GATHER_STICK_MISMATCH = (
    4327,
    "compile gather at dim>=1 compiles and returns values that differ from CPU "
    "(the issue reported an 'op_spec has 3 args' abort, then a 'cannot re-tile' error).",
    r"compiled spyre <-> cpu\S* mismatch\s+Tensor-likes are not (?:close|equal)!",
)
_MATMUL_MISMATCH = (
    4722,
    "compile gather followed by matmul / linear numerical mismatch.",
    r"spyre <-> cpu\S* mismatch\s+Tensor-likes are not (?:close|equal)!",
)
_SCATTER_STICK = (
    4395,
    "compile gather followed by scatter_add produces numerical mismatch vs CPU.",
    r"compiled spyre <-> cpu\S* mismatch\s+Tensor-likes are not (?:close|equal)!",
)
_CHUNK_REDUCE_MUTATION = rejection(
    4754,
    "compile chunk reduce: no offset-free alternative stick dim for mutation target.",
    r"no offset-free alternative stick dim for mutation target",
)


class TestGatherComposedChainsAndEndToEndPipelines:
    """Sequential gathers, gather→matmul/LN/residual, RoPE gather+apply (rotate-half and interleaved), chunked prefill/decode, parallel K/V, and end-to-end decode/beam/speculative pipelines."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield

    # ------------------------------------------------------------------

    def test_two_sequential_gathers(self, execution_mode):
        """Two back-to-back gather ops; second gather feeds into first output."""
        table1 = cached_randn((64, 32), differentiation="ch01t1", dtype=torch.float16)
        table2 = cached_randn((64, 32), differentiation="ch01t2", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda t1, t2, i: t1[i] + t2[i],
            table1,
            table2,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_gather_then_gather_on_output(self, execution_mode):
        """Second gather re-indexes the output of the first gather."""
        table = cached_randn((64, 32), differentiation="ch02", dtype=torch.float16)
        idx1 = torch.randint(0, 64, (32,), dtype=torch.int64)
        idx2 = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda t, i1, i2: t[i1][i2],
            table,
            idx1,
            idx2,
            atol=0,
            rtol=0,
        )

    @gap_one_core
    def test_gather_then_matmul(self, execution_mode):
        """Gather followed by matmul; fused downstream linear."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4722
        _xfail_existing(
            execution_mode, eager=_MATMUL_MISMATCH, compiled=_MATMUL_MISMATCH
        )
        emb = cached_randn((256, 128), differentiation="ch03e", dtype=torch.float16)
        w = cached_randn((128, 64), differentiation="ch03w", dtype=torch.float16)
        idx = torch.randint(0, 256, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda e, w, i: (e[i] @ w),
            emb,
            w,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    @gap_one_core
    def test_gather_then_layer_norm(self, execution_mode):
        """Gather output into layer_norm; end-to-end embedding + norm."""
        emb = cached_randn((256, 128), differentiation="ch04e", dtype=torch.float16)
        idx = torch.randint(0, 256, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda e, i: F.layer_norm(e[i].float(), [128]).half(),
            emb,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_gather_add_residual(self, execution_mode):
        """Gathered embedding added to residual stream."""
        emb = cached_randn((256, 128), differentiation="ch05e", dtype=torch.float16)
        residual = cached_randn((32, 128), differentiation="ch05r", dtype=torch.float16)
        idx = torch.randint(0, 256, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda e, r, i: e[i] + r,
            emb,
            residual,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_gather_split_rope_apply(self, execution_mode):
        """Gather cos/sin cache, split Q/K, apply RoPE."""
        cos_sin = cached_randn(
            (4096, 128), differentiation="ch06cs", dtype=torch.float16
        )
        pos = torch.randint(0, 4096, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda cs, p: torch.chunk(cs[p], 2, dim=-1),
            cos_sin,
            pos,
            atol=0,
            rtol=0,
        )

    def test_gather_two_tables_add(self, execution_mode):
        """Lookup from two separate embedding tables, element-wise add."""
        t1 = cached_randn((128, 64), differentiation="ch07t1", dtype=torch.float16)
        t2 = cached_randn((128, 64), differentiation="ch07t2", dtype=torch.float16)
        idx = torch.randint(0, 128, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda t1, t2, i: t1[i] + t2[i],
            t1,
            t2,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    @gap_one_core
    def test_gather_then_scatter_add(self, execution_mode):
        """Gather expert rows then scatter_add_ into dest (8, 32, 64) — MoE extra (#4395).

        Standalone grouped_mm scatter_add is in test_scatter_ops.py.
        """
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4396
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4395
        _xfail_existing(
            execution_mode, eager=_SCATTER_ADD_EAGER, compiled=_SCATTER_STICK
        )
        expert_w = cached_randn(
            (8, 32, 64), differentiation="ch08", dtype=torch.float16
        )
        ids = torch.randint(0, 8, (16,), dtype=torch.int64)
        dst = torch.zeros(8, 32, 64, dtype=torch.float16)

        def fn(w, i, dst):
            gathered = w[i]
            idx = i.view(-1, 1, 1).expand_as(gathered)
            return dst.clone().scatter_add_(0, idx, gathered)

        compare_mode(
            execution_mode, fn, expert_w, ids, dst, atol=_TOL_F16, rtol=_TOL_F16
        )

    # ------------------------------------------------------------------

    def test_chunk_gather_cos_sin(self, execution_mode):
        """Gather cos_sin[pos]; torch.chunk → cos, sin halves."""
        cos_sin = cached_randn(
            (4096, 128), differentiation="chnk01", dtype=torch.float16
        )
        pos = torch.randint(0, 4096, (64,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda cs, p: torch.chunk(cs[p], 2, dim=-1),
            cos_sin,
            pos,
            atol=0,
            rtol=0,
        )

    @gap_one_core
    def test_split_gather_qkv(self, execution_mode):
        """QKV split after token embedding gather + linear projection."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4722
        _xfail_existing(
            execution_mode, eager=_MATMUL_MISMATCH, compiled=_MATMUL_MISMATCH
        )
        emb = cached_randn((512, 128), differentiation="chnk02e", dtype=torch.float16)
        W = cached_randn((128, 384), differentiation="chnk02w", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda e, W, i: torch.split(e[i] @ W, 128, dim=-1),
            emb,
            W,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_rope_gather_broadcast(self, execution_mode):
        """Gather RoPE freqs then broadcast over head dim."""
        freqs = cached_randn((4096, 64), differentiation="chnk04", dtype=torch.float16)
        pos = torch.randint(0, 4096, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda f, p: f[p].unsqueeze(1),
            freqs,
            pos,
            atol=0,
            rtol=0,
        )

    def test_gather_concat(self, execution_mode):
        """Two gathers from separate caches; concat along seq dim."""
        k1 = cached_randn((256, 8, 64), differentiation="chnk05a", dtype=torch.float16)
        k2 = cached_randn((256, 8, 64), differentiation="chnk05b", dtype=torch.float16)
        i1 = torch.randint(0, 256, (32,), dtype=torch.int64)
        i2 = torch.randint(0, 256, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda k1, k2, i1, i2: torch.cat([k1[i1], k2[i2]], dim=0),
            k1,
            k2,
            i1,
            i2,
            atol=0,
            rtol=0,
        )

    @gap_one_core
    def test_chunk_then_reduce(self, execution_mode):
        """Gather + chunk + per-chunk sum reduction; each quarter summed independently."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4754
        _xfail_existing(
            execution_mode,
            eager=_CHUNK_REDUCE_MUTATION,
            compiled=_CHUNK_REDUCE_MUTATION,
        )
        x = cached_randn((64, 256), differentiation="chnk06", dtype=torch.float16)
        idx = torch.randint(0, 64, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.stack(
                [c.sum(dim=-1) for c in torch.chunk(x[i], 4, dim=-1)]
            ),
            x,
            idx,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_rope_gather_mul(self, execution_mode):
        """Gather cos[pos] * q + sin[pos] * rotate90(q) RoPE pattern."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(execution_mode, eager=_SLICE_OFFSET_ALIGNMENT)
        cos = cached_randn((4096, 64), differentiation="chnk07c", dtype=torch.float16)
        sin = cached_randn((4096, 64), differentiation="chnk07s", dtype=torch.float16)
        q = cached_randn((32, 64), differentiation="chnk07q", dtype=torch.float16)
        pos = torch.randint(0, 4096, (32,), dtype=torch.int64)
        half = q.shape[-1] // 2
        compare_mode(
            execution_mode,
            lambda c, s, q, p: (
                q * c[p] + torch.cat([-q[:, half:], q[:, :half]], dim=-1) * s[p]
            ),
            cos,
            sin,
            q,
            pos,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_gather_unbind_stack(self, execution_mode):
        """Gather + unbind + stack at new axis."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(execution_mode, eager=_SLICE_OFFSET_ALIGNMENT)
        x = cached_randn((64, 4, 32), differentiation="chnk08", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.stack(torch.unbind(x[i], dim=1), dim=0),
            x,
            idx,
            atol=0,
            rtol=0,
        )

    def test_chunk_gather_parallel_k_v(self, execution_mode):
        """Parallel K and V gather in single fn; both correct."""
        k = cached_randn((512, 8, 64), differentiation="chnk10k", dtype=torch.float16)
        v = cached_randn((512, 8, 64), differentiation="chnk10v", dtype=torch.float16)
        idx = torch.randint(0, 512, (64,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda k, v, i: (k[i], v[i]),
            k,
            v,
            idx,
            atol=0,
            rtol=0,
        )

    def test_gather_reshape_chunk(self, execution_mode):
        """Gather + reshape + chunk for multi-head split."""
        emb = cached_randn((512, 512), differentiation="chnk11", dtype=torch.float16)
        idx = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda e, i: torch.chunk(e[i].reshape(32, 8, 64), 2, dim=1),
            emb,
            idx,
            atol=0,
            rtol=0,
        )

    def test_gather_split_apply_merge(self, execution_mode):
        """Gather → split → per-head op → merge back."""
        emb = cached_randn((512, 256), differentiation="chnk12", dtype=torch.float16)
        idx = torch.randint(0, 512, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda e, i: torch.cat(
                [h.relu() for h in torch.chunk(e[i], 4, dim=-1)], dim=-1
            ),
            emb,
            idx,
            atol=0,
            rtol=0,
        )

    # ------------------------------------------------------------------

    def test_e2e_decode_token_lookup(self, execution_mode):
        """GE2E-01: Full decode step: token embed + KV read + minimal attention."""
        vocab = cached_randn((512, 128), differentiation="e2e01v", dtype=torch.float16)
        kv = cached_randn((512, 8, 64), differentiation="e2e01kv", dtype=torch.float16)
        tok_id = torch.randint(0, 512, (1,), dtype=torch.int64)
        slot = torch.randint(0, 512, (1,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda v, kv, t, s: (v[t], kv[s]),
            vocab,
            kv,
            tok_id,
            slot,
            atol=0,
            rtol=0,
        )

    def test_e2e_decode_single_step(self, execution_mode):
        """GE2E-03: Single autoregressive decode; 1 new token lookup."""
        vocab = cached_randn((512, 128), differentiation="e2e03v", dtype=torch.float16)
        kv_k = cached_randn((512, 8, 64), differentiation="e2e03k", dtype=torch.float16)
        kv_v = cached_randn(
            (512, 8, 64), differentiation="e2e03v2", dtype=torch.float16
        )
        tok = torch.randint(0, 512, (1,), dtype=torch.int64)
        slots = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda v, k, vv, t, s: (v[t], k[s], vv[s]),
            vocab,
            kv_k,
            kv_v,
            tok,
            slots,
            atol=0,
            rtol=0,
        )

    def test_e2e_prefill_kv_fill(self, execution_mode):
        """GE2E-05: Gather prefill KV tokens from paged cache; 32 positions from 128-slot pool."""
        cache = cached_randn((128, 8, 64), differentiation="e2e05", dtype=torch.float16)
        pos = torch.arange(32, dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda kv, p: kv[p],
            cache,
            pos,
            atol=0,
            rtol=0,
        )

    def test_e2e_kv_decode_extend(self, execution_mode):
        """GE2E-07: Decode extends prefill; gather 128 prefill + 1 decode position from paged cache."""
        cache = cached_randn((256, 8, 64), differentiation="e2e07", dtype=torch.float16)
        all_pos = torch.arange(129, dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda kv, p: kv[p],
            cache,
            all_pos,
            atol=0,
            rtol=0,
        )

    def test_e2e_logit_slice(self, execution_mode):
        """GE2E-08: Logit extraction at last token position for sampling."""
        logits = cached_randn(
            (12, 64, 512), differentiation="e2e08", dtype=torch.float16
        )
        compare_mode(
            execution_mode,
            lambda x: x[:, -1, :],
            logits,
            atol=0,
            rtol=0,
        )

    def test_e2e_continuous_batch(self, execution_mode):
        """GE2E-10: Continuous batch: 4 prefill + 8 decode requests in one gather."""
        kv = cached_randn((512, 8, 64), differentiation="e2e10", dtype=torch.float16)
        prefill_slots = torch.randint(0, 512, (128,), dtype=torch.int64)
        decode_slots = torch.randint(0, 512, (8,), dtype=torch.int64)
        all_slots = torch.cat([prefill_slots, decode_slots])
        compare_mode(
            execution_mode,
            lambda x, i: x[i],
            kv,
            all_slots,
            atol=0,
            rtol=0,
        )

    @gap_one_core
    def test_e2e_speculative_token(self, execution_mode):
        """GE2E-11: Speculative decode: draft token prob lookup."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4328
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4327
        _xfail_existing(
            execution_mode,
            eager=_GATHER_EAGER,
            compiled=_GATHER_STICK_MISMATCH,
        )
        probs = cached_randn((16, 512), differentiation="e2e11", dtype=torch.float16)
        draft_tok = torch.randint(0, 512, (5,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda p, d: torch.gather(
                p[:5], 1, d.unsqueeze(0).expand(5, -1)
            ).diagonal(),
            probs,
            draft_tok,
            atol=_TOL_F16,
            rtol=_TOL_F16,
        )

    def test_e2e_full_pipeline_bfloat16(self, execution_mode):
        """GE2E-14: bfloat16 end-to-end: embed + KV read + decode output."""
        vocab = cached_randn((512, 128), differentiation="e2e14v", dtype=torch.bfloat16)
        kv = cached_randn((512, 8, 64), differentiation="e2e14kv", dtype=torch.bfloat16)
        tok = torch.randint(0, 512, (8,), dtype=torch.int64)
        slots = torch.randint(0, 512, (32,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda v, kv, t, s: (v[t], kv[s]),
            vocab,
            kv,
            tok,
            slots,
            atol=0,
            rtol=0,
        )

    # ------------------------------------------------------------------

    @gap_one_core
    def test_rope_neox_qk_both(self, execution_mode):
        """NeoX rotate-half RoPE applied to both Q and K after position gather."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/6388
        _xfail_existing(
            execution_mode,
            eager=_SLICE_OFFSET_ALIGNMENT,
            compiled=_POINTWISE_NO_LAYOUT,
        )
        cos_sin = cached_randn(
            (4096, 128), differentiation="ropi02", dtype=torch.float16
        )
        q = cached_randn((16, 64), differentiation="ropi02q", dtype=torch.float16)
        k = cached_randn((16, 64), differentiation="ropi02k", dtype=torch.float16)
        pos = torch.randint(0, 4096, (16,), dtype=torch.int64)

        def fn(cache, q, k, pos):
            half = q.shape[-1] // 2
            cos = cache[pos, :half]
            sin = cache[pos, half : half * 2]

            def apply_rope(x):
                return torch.cat(
                    [
                        x[:, :half] * cos - x[:, half:] * sin,
                        x[:, half:] * cos + x[:, :half] * sin,
                    ],
                    dim=-1,
                )

            return apply_rope(q), apply_rope(k)

        compare_mode(
            execution_mode, fn, cos_sin, q, k, pos, atol=_TOL_F16, rtol=_TOL_F16
        )

    def test_rope_interleaved_separate_cos_sin_caches(self, execution_mode):
        """Interleaved RoPE with separate cos/sin caches; both gathered then applied to Q and K."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4399
        _xfail_existing(execution_mode, eager=_SLICE_OFFSET_ALIGNMENT)
        cos = cached_randn((2048, 64), differentiation="ropi03c", dtype=torch.float16)
        sin = cached_randn((2048, 64), differentiation="ropi03s", dtype=torch.float16)
        q = cached_randn((32, 64), differentiation="ropi03q", dtype=torch.float16)
        k = cached_randn((32, 64), differentiation="ropi03k", dtype=torch.float16)
        pos = torch.randint(0, 2048, (32,), dtype=torch.int64)

        def fn(cos, sin, q, k, pos):
            c, s = cos[pos], sin[pos]
            half = q.shape[-1] // 2
            q_rot = torch.cat([-q[:, half:], q[:, :half]], dim=-1)
            k_rot = torch.cat([-k[:, half:], k[:, :half]], dim=-1)
            return q * c + q_rot * s, k * c + k_rot * s

        compare_mode(
            execution_mode, fn, cos, sin, q, k, pos, atol=_TOL_F16, rtol=_TOL_F16
        )
