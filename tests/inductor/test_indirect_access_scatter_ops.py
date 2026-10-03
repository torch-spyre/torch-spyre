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

import pytest
import torch
from utils_inductor import DEVICE, compare_with_cpu

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mode_flags(mode: str):
    """Return (run_eager, run_compile) booleans from the 'mode' parameter string."""
    return mode == "eager", mode == "compile"


# =============================================================================
# SCATTER
# =============================================================================
class TestScatterOp:
    """Extended gap scenarios for torch.scatter / Tensor.scatter_."""

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_zero_volume_inner_dim(self, mode: str):
        """Zero-volume inner dimension scatter — shape (8, 0, 128) must be a safe no-op.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128)."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4473")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: kv_heads=8, head_dim=128 — zero-volume along middle dim
        self_t = torch.zeros(8, 0, 128, dtype=torch.bfloat16)
        src_t = torch.zeros(4, 0, 128, dtype=torch.bfloat16)
        idx_t = torch.zeros(4, 0, 128, dtype=torch.int64)

        def fn(dest, index, source):
            return dest.scatter(0, index, source)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_1d_single_element_source(self, mode: str):
        """Single-element source scatter — src shape (1,) writes one value into a 1D dest.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128).

        Scatters exactly one source element to one indexed position in a 128-element
        buffer.  This is not a broadcasting test; index and source both have shape (1,).
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4473")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4926")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: head_dim=128 — write one source value to one indexed position
        self_t = torch.zeros(128, dtype=torch.float32)
        src_t = torch.randn(1, dtype=torch.float32)
        idx_t = torch.randint(0, 128, (1,), dtype=torch.int64)

        def fn(dest, index, source):
            return dest.scatter(0, index, source)

        compare_with_cpu(
            fn, self_t, idx_t, src_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("dim", [0, 1])
    def test_scatter_empty_row_index(self, mode: str, dim: int):
        """Zero-element index and source — safe no-op, dest returned unchanged.
        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128, hidden=5120).

        dim=0: dest (8, 128), src/index (0, 128) — zero rows, empty along the scatter dim.
        dim=1: dest (8, 128), src/index (8, 0) — zero cols, empty along the scatter dim.
        Both cases must return dest unchanged.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4473")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: kv_heads=8, head_dim=128
        self_t = torch.randn(8, 128, dtype=torch.bfloat16)
        if dim == 0:
            src_t = torch.randn(
                0, 128, dtype=torch.bfloat16
            )  # zero rows — empty along dim 0
            idx_t = torch.zeros(0, 128, dtype=torch.int64)
        else:
            src_t = torch.randn(
                8, 0, dtype=torch.bfloat16
            )  # zero cols — empty along dim 1
            idx_t = torch.zeros(8, 0, dtype=torch.int64)

        def fn(dest, index, source):
            return dest.scatter(dim, index, source)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_transposed_source_strided_index(self, mode: str):
        """Transposed (non-contiguous) source + strided index into strided destination.
        Model shape: Granite-4.1-20B (kv_heads=8, head_dim=128, hidden=8192).

        Exercises the full non-contiguous path: dest is a step-2 strided view,
        src is transposed (T), index is a narrow sub-view — all three inputs
        have non-unit strides simultaneously.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4473")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4927")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: kv_heads=8, head_dim=128 — step-2 strided view across kv_heads
        # Destination: step-2 strided view of a larger base, shape (8, 128)
        base_dest = torch.zeros(16, 128, dtype=torch.bfloat16)

        # Source: (128, 8) transposed to (8, 128) — non-contiguous
        src_base = torch.randn(128, 8, dtype=torch.bfloat16)
        src_t = src_base.t()  # shape (8, 128), non-contiguous

        # Index: narrow sub-view, shape (8, 128) with values < 8 (dest stride-2 view rows)
        idx_base = torch.randint(0, 8, (16, 128), dtype=torch.int64)
        idx_t = idx_base.narrow(0, 0, 8)  # shape (8, 128), non-zero storage offset

        def fn(base, index, source):
            out = base.clone()
            view = out[::2, :]
            view.scatter_(0, index, source)
            return out

        compare_with_cpu(
            fn,
            base_dest,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_1d_inplace_scalar_fill(self, mode: str):
        """1D in-place scatter_ scalar fill into a head_dim vector.
        Model shape: Gemma-4-26B (head_dim=256).

        Covers 1D-IP: the only existing 1D scatter test (test_scatter_dim_broadcasting)
        uses the out-of-place form. This exercises scatter_() on a 1D tensor by filling
        16 randomly selected positions in a head_dim=256 buffer with a constant 1.0.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4473")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4399")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: head_dim=256 — scatter_ writes a constant into 16 selected slots
        self_t = torch.zeros(256, dtype=torch.bfloat16)
        idx_t = torch.randperm(256, dtype=torch.int64)[:16]

        def fn(dest, index):
            out = dest.clone()
            out.scatter_(0, index, 1.0)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_4d_outofplace_attention_logits(self, mode: str):
        """4D out-of-place scatter on (batch, kv_heads, seq, head_dim) attention geometry.
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128).

        Covers 4D-OOP: no existing test calls scatter() on a 4D tensor.
        Scatters updated token logits into 4 selected sequence positions (dim=2).
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4401")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: batch=2, kv_heads=10, seq=16, head_dim=128
        # Unique indices along the scatter dim (dim=2, size=16): same 4 positions used for
        # every (batch, kv_head, head_dim) slice so there are no duplicate writes.
        self_t = torch.zeros(2, 10, 16, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 10, 4, 128, dtype=torch.bfloat16)
        seq_idx = torch.randperm(16, dtype=torch.int64)[:4]  # 4 unique seq positions
        idx_t = seq_idx.view(1, 1, 4, 1).expand(
            2, 10, 4, 128
        )  # broadcast, no duplicates

        def fn(dest, index, source):
            return dest.scatter(2, index, source)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_4d_inplace_channel_update(self, mode: str):
        """4D in-place scatter_ — updates selected kv_head channels in a KV-cache buffer.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128).

        Covers 4D-IP: scatter_() on a 4D tensor. Scatters 4 source rows into selected
        kv_heads positions (dim=1) inside a (batch, kv_heads, seq, head_dim) buffer.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4401")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: batch=2, kv_heads=8, seq=16, head_dim=128
        # Unique indices along scatter dim (dim=1, size=8): same 4 kv_head positions for
        # every (batch, seq, head_dim) slice so there are no duplicate writes.
        self_t = torch.zeros(2, 8, 16, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 4, 16, 128, dtype=torch.bfloat16)
        kv_idx = torch.randperm(8, dtype=torch.int64)[:4]  # 4 unique kv_head positions
        idx_t = kv_idx.view(1, 4, 1, 1).expand(
            2, 4, 16, 128
        )  # broadcast, no duplicates

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_(1, index, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_5d_inplace_layer_cache(self, mode: str):
        """5D in-place scatter_ on layer-KV-cache geometry (batch, layers, kv_heads, seq, head_dim).
        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128).

        Covers 5D-IP: the only existing 5D scatter test (test_scatter_5d_middle_axis)
        uses the out-of-place form. Scatters 4 updated token positions into dim=3 (seq)
        of a full layer-cache buffer in-place.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4401")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: batch=2, layers=4, kv_heads=8, seq=16, head_dim=128
        # Unique indices along scatter dim (dim=3, size=16): same 4 seq positions for
        # every (batch, layers, kv_heads, head_dim) slice so there are no duplicate writes.
        self_t = torch.zeros(2, 4, 8, 16, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 4, 8, 4, 128, dtype=torch.bfloat16)
        seq_idx = torch.randperm(16, dtype=torch.int64)[:4]  # 4 unique seq positions
        idx_t = seq_idx.view(1, 1, 1, 4, 1).expand(
            2, 4, 8, 4, 128
        )  # broadcast, no duplicates

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_(3, index, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_3d_inplace_kv_cache_update(self, mode: str):
        """3D in-place scatter_ on KV-cache geometry (kv_heads, seq, head_dim).
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128).

        Covers 3D-IP: the only existing 3D scatter test (test_scatter_zero_volume_inner_dim)
        uses the out-of-place form on a zero-volume tensor. This exercises scatter_() on
        a real non-empty 3D tensor, writing 4 source rows into selected seq positions
        (dim=1) of a (kv_heads, seq, head_dim) KV-cache buffer.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4401")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: kv_heads=8, seq=16, head_dim=128
        # Unique indices along scatter dim (dim=1, size=16): same 4 seq positions for
        # every (kv_head, head_dim) slice so there are no duplicate writes.
        self_t = torch.zeros(8, 16, 128, dtype=torch.bfloat16)
        src_t = torch.randn(8, 4, 128, dtype=torch.bfloat16)
        seq_idx = torch.randperm(16, dtype=torch.int64)[:4]  # 4 unique seq positions
        idx_t = seq_idx.view(1, 4, 1).expand(8, 4, 128)  # broadcast, no duplicates

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_(1, index, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_6d_inplace_multi_layer_cache(self, mode: str):
        """6D in-place scatter_ on multi-layer multi-expert cache geometry
        (experts, batch, layers, kv_heads, seq, head_dim).
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128, bf16 equivalent).

        Covers 6D-IP: no existing test uses scatter_() on a 6D tensor. Scatters 2
        updated token positions into dim=4 (seq) of a 6D expert-layer-KV-cache buffer.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4401")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B (model-derived geometry, bf16 dtype):
        # experts=2, batch=2, layers=2, kv_heads=8, seq=8, head_dim=128
        # Unique indices along scatter dim (dim=4, size=8): same 2 seq positions for
        # every outer slice so there are no duplicate writes.
        self_t = torch.zeros(2, 2, 2, 8, 8, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 2, 2, 8, 2, 128, dtype=torch.bfloat16)
        seq_idx = torch.randperm(8, dtype=torch.int64)[:2]  # 2 unique seq positions
        idx_t = seq_idx.view(1, 1, 1, 1, 2, 1).expand(
            2, 2, 2, 8, 2, 128
        )  # broadcast, no duplicates

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_(4, index, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )


# =============================================================================
# SCATTER_ADD
# =============================================================================
class TestScatterAddOp:
    """Extended gap scenarios for torch.scatter_add / Tensor.scatter_add_."""

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_add_nonzero_storage_offset(self, mode: str):
        """Non-zero storage_offset on self, index, and src simultaneously.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256)."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4473")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: kv_heads=16, head_dim=256 — offset sub-views into a kv_heads×head_dim buffer
        base_self = torch.zeros(32, 256, dtype=torch.bfloat16)
        base_src = torch.randn(32, 256, dtype=torch.bfloat16)
        base_idx = torch.randint(0, 16, (32, 256), dtype=torch.int64)

        # Sub-views with non-zero storage_offset — 16 rows each
        self_w = base_self.narrow(0, 8, 16)  # offset = 8*256 elements
        src_w = base_src.narrow(0, 4, 16)  # offset = 4*256 elements
        idx_w = base_idx.narrow(0, 2, 16)  # offset = 2*256 elements

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_add_(0, index, source)
            return out

        compare_with_cpu(
            fn,
            self_w,
            idx_w,
            src_w,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_add_100k_single_bin_hotspot(self, mode: str):
        """100 000-element single-bucket hotspot — atomic-contention stress test.
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128).

        All 100 000 updates target row 0 exclusively, exercising maximum atomic
        contention on one accumulator cell.  This is a stress / no-crash test:
        the tolerance is deliberately wide because BF16 accumulation order is
        unspecified and both CPU and Spyre may produce different rounding.

        The tolerance (atol=128, rtol=0) catches only catastrophically wrong
        results (off by more than ~0.1 % of 100 000) while accepting the
        BF16-level rounding divergence (~hundreds of ULPs at this scale).
        For tight numerical accuracy, see test_scatter_add_bf16_subnormal_accumulation.
        """
        if mode == "eager":
            pytest.xfail(reason="known issue- 4396")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: kv_heads=10, head_dim=128 — all updates target kv_head row 0
        num_updates = 100_000
        self_t = torch.zeros(10, 128, dtype=torch.bfloat16)
        src_t = torch.ones(num_updates, 128, dtype=torch.bfloat16)
        idx_t = torch.zeros(num_updates, 128, dtype=torch.int64)  # all target row 0

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_add_(0, index, source)
            return out

        # atol=128 ≈ 0.128% of 100 000 — catches catastrophic errors, not tight correctness
        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=128,
            rtol=0,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_add_bf16_subnormal_accumulation(self, mode: str):
        """BF16 subnormal/denormal accumulation using smallest_subnormal.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128).

        Uses torch.finfo(bfloat16).smallest_subnormal to guarantee a true
        denormal value rather than a magic constant that may flush to zero on
        some platforms. Verifies the Spyre backend does not trap on denormals
        or silently flush them before accumulation.
        """
        if mode == "eager":
            pytest.xfail(reason="known issue- 4396")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: kv_heads=8, head_dim=128 — subnormals accumulate into row 0
        finfo = torch.finfo(torch.bfloat16)
        val = getattr(finfo, "smallest_subnormal", 2**-133)
        self_t = torch.zeros(8, 128, dtype=torch.bfloat16)
        # Multiple subnormal values accumulating into the same bucket exercises
        # both the denormal representation and repeated addition without flush.
        src_t = torch.full((4, 128), val, dtype=torch.bfloat16)
        idx_t = torch.zeros(4, 128, dtype=torch.int64)  # all accumulate into row 0

        def fn(dest, index, source):
            return torch.scatter_add(dest, 0, index, source)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_add_1d_both(self, mode: str):
        """1D scatter_add — both OOP and IP on a 1D head_dim vector.
        Model shape: Granite-4.1-20B (head_dim=128). Covers 1D-OOP and 1D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4396")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: head_dim=128 flat accumulation buffer
        self_t = torch.zeros(128, dtype=torch.bfloat16)
        src_t = torch.ones(16, dtype=torch.bfloat16)
        idx_t = torch.randint(0, 128, (16,), dtype=torch.int64)

        def fn_oop(dest, index, source):
            return torch.scatter_add(dest, 0, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.scatter_add_(0, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_add_3d_both(self, mode: str):
        """3D scatter_add on (kv_heads, seq, head_dim) — both OOP and IP.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256). Covers 3D-OOP and 3D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4396")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: (kv_heads=16, seq=8, head_dim=256) — accumulate along seq (dim=1)
        self_t = torch.zeros(16, 8, 256, dtype=torch.bfloat16)
        src_t = torch.randn(16, 4, 256, dtype=torch.bfloat16)
        idx_t = torch.randint(0, 8, (16, 4, 256), dtype=torch.int64)

        def fn_oop(dest, index, source):
            return torch.scatter_add(dest, 1, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.scatter_add_(1, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_add_4d_both(self, mode: str):
        """4D scatter_add on (batch, kv_heads, seq, head_dim) — both OOP and IP.
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128). Covers 4D-OOP and 4D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4396")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: (batch=2, kv_heads=10, seq=8, head_dim=128) — accumulate along seq (dim=2)
        self_t = torch.zeros(2, 10, 8, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 10, 4, 128, dtype=torch.bfloat16)
        idx_t = torch.randint(0, 8, (2, 10, 4, 128), dtype=torch.int64)

        def fn_oop(dest, index, source):
            return torch.scatter_add(dest, 2, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.scatter_add_(2, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_add_5d_both(self, mode: str):
        """5D scatter_add on (batch, layers, kv_heads, seq, head_dim) — both OOP and IP.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128). Covers 5D-OOP and 5D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4396")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (batch=2, layers=4, kv_heads=8, seq=8, head_dim=128) — accumulate along seq (dim=3)
        self_t = torch.zeros(2, 4, 8, 8, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 4, 8, 4, 128, dtype=torch.bfloat16)
        idx_t = torch.randint(0, 8, (2, 4, 8, 4, 128), dtype=torch.int64)

        def fn_oop(dest, index, source):
            return torch.scatter_add(dest, 3, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.scatter_add_(3, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_add_6d_both(self, mode: str):
        """6D scatter_add on (experts, batch, layers, kv_heads, seq, head_dim) — both OOP and IP.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128, bf16 equiv). Covers 6D-OOP and 6D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4396")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4409")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (experts=2, batch=2, layers=2, kv_heads=8, seq=4, head_dim=128) — accumulate along seq (dim=4)
        self_t = torch.zeros(2, 2, 2, 8, 4, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 2, 2, 8, 2, 128, dtype=torch.bfloat16)
        idx_t = torch.randint(0, 4, (2, 2, 2, 8, 2, 128), dtype=torch.int64)

        def fn_oop(dest, index, source):
            return torch.scatter_add(dest, 4, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.scatter_add_(4, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )


# =============================================================================
# SCATTER_REDUCE
# =============================================================================
class TestScatterReduceOp:
    """Extended gap scenarios for torch.scatter_reduce / Tensor.scatter_reduce_."""

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("reduce_mode", ["sum", "prod", "mean", "amax", "amin"])
    def test_scatter_reduce_nan_inf_all_modes(self, mode: str, reduce_mode: str):
        """NaN and Inf propagation across all 5 reduction modes (not just sum).
        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128)."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile" and reduce_mode == "sum":
            pytest.xfail(reason="known issue- 4409")
        if mode == "compile" and reduce_mode in ("prod", "mean", "amax", "amin"):
            pytest.xfail(reason="known issue- 4928")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: kv_heads=8, head_dim=128 — NaN/Inf injected into first two rows
        self_t = torch.zeros(8, 128, dtype=torch.float32)
        src_t = torch.ones(4, 128, dtype=torch.float32)
        src_t[0, 0] = float("nan")
        src_t[1, 1] = float("inf")
        idx_t = torch.tensor(
            [[0] * 128, [1] * 128, [2] * 128, [3] * 128], dtype=torch.int64
        )

        def fn(dest, index, source):
            return dest.scatter_reduce(
                0, index, source, reduce=reduce_mode, include_self=True
            )

        compare_with_cpu(
            fn, self_t, idx_t, src_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_reduce_dynamic_symint_shapes(self, mode: str):
        """Dynamic SymInt shape tracing through scatter_reduce in TorchInductor.
        Model shape: Granite-4.1-20B (hidden=8192, kv_heads=8, head_dim=128)."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- epic-43")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: batch=2, seq=8, hidden=8192; mark seq dim dynamic
        B, S, D = 2, 8, 8192
        max_S = 32

        self_t = torch.zeros(B, max_S, D, dtype=torch.float32)
        src_t = torch.randn(B, S, D, dtype=torch.float32)
        idx_t = torch.randint(0, max_S, (B, S, D), dtype=torch.int64)

        def fn(dest, index, source):
            return dest.scatter_reduce(
                1, index, source, reduce="sum", include_self=True
            )

        if run_compile and hasattr(torch, "_dynamo"):
            torch._dynamo.mark_dynamic(src_t, 1)

        compare_with_cpu(
            fn, self_t, idx_t, src_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_reduce_5d_attention_geometry(self, mode: str):
        """5D attention geometry (Batch, Heads, Layers, Seq, HeadDim) scatter_reduce along dim=3.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256)."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4874")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4874")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: (batch=2, kv_heads=16, layers=2, seq=16, head_dim=256)
        self_t = torch.zeros(2, 16, 2, 32, 256, dtype=torch.bfloat16)
        src_t = torch.randn(2, 16, 2, 16, 256, dtype=torch.bfloat16)
        idx_t = torch.randint(0, 32, (2, 16, 2, 16, 256), dtype=torch.int64)

        def fn(dest, index, source):
            return dest.scatter_reduce(
                3, index, source, reduce="sum", include_self=True
            )

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.05,
            rtol=0.05,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_reduce_1d_inplace(self, mode: str):
        """1D in-place scatter_reduce_ on a head_dim vector.
        Model shape: Ministral-3-14B (head_dim=128). Covers 1D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4932")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: head_dim=128 flat accumulation buffer
        self_t = torch.ones(128, dtype=torch.float32)
        src_t = torch.rand(16, dtype=torch.float32) + 0.5
        idx_t = torch.randint(0, 128, (16,), dtype=torch.int64)

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_reduce_(0, index, source, reduce="sum", include_self=True)
            return out

        compare_with_cpu(
            fn, self_t, idx_t, src_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_reduce_2d_inplace(self, mode: str):
        """2D in-place scatter_reduce_ — covers 2D-IP (all existing 2D tests use OOP).
        Model shape: Granite-4.1-20B (kv_heads=8, head_dim=128)."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4932")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: (kv_heads=8, head_dim=128)
        self_t = torch.ones(8, 128, dtype=torch.float32)
        src_t = torch.rand(4, 128, dtype=torch.float32) + 0.5
        idx_t = torch.randint(0, 8, (4, 128), dtype=torch.int64)

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_reduce_(0, index, source, reduce="sum", include_self=True)
            return out

        compare_with_cpu(
            fn, self_t, idx_t, src_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_reduce_3d_inplace(self, mode: str):
        """3D in-place scatter_reduce_ — covers 3D-IP (existing 3D test uses OOP).
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256)."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4932")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: (kv_heads=16, seq=8, head_dim=256) — reduce along seq (dim=1)
        self_t = torch.ones(16, 8, 256, dtype=torch.float32)
        src_t = torch.rand(16, 4, 256, dtype=torch.float32) + 0.5
        idx_t = torch.randint(0, 8, (16, 4, 256), dtype=torch.int64)

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_reduce_(1, index, source, reduce="sum", include_self=True)
            return out

        compare_with_cpu(
            fn, self_t, idx_t, src_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_reduce_4d_both(self, mode: str):
        """4D scatter_reduce — both OOP and IP on (batch, kv_heads, seq, head_dim).
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128). Covers 4D-OOP and 4D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4932")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: (batch=2, kv_heads=10, seq=8, head_dim=128) — reduce along seq (dim=2)
        self_t = torch.ones(2, 10, 8, 128, dtype=torch.float32)
        src_t = torch.rand(2, 10, 4, 128, dtype=torch.float32) + 0.5
        idx_t = torch.randint(0, 8, (2, 10, 4, 128), dtype=torch.int64)

        def fn_oop(dest, index, source):
            return dest.scatter_reduce(
                2, index, source, reduce="sum", include_self=True
            )

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.scatter_reduce_(2, index, source, reduce="sum", include_self=True)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_reduce_5d_inplace(self, mode: str):
        """5D in-place scatter_reduce_ — covers 5D-IP (existing 5D test uses OOP).
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128)."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4932")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (batch=2, layers=4, kv_heads=8, seq=8, head_dim=128) — reduce along seq (dim=3)
        self_t = torch.ones(2, 4, 8, 8, 128, dtype=torch.bfloat16)
        src_t = torch.rand(2, 4, 8, 4, 128, dtype=torch.bfloat16) + 0.5
        idx_t = torch.randint(0, 8, (2, 4, 8, 4, 128), dtype=torch.int64)

        def fn(dest, index, source):
            out = dest.clone()
            out.scatter_reduce_(3, index, source, reduce="sum", include_self=True)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_scatter_reduce_6d_both(self, mode: str):
        """6D scatter_reduce — both OOP and IP on expert-layer-KV-cache geometry.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128, bf16 equiv). Covers 6D-OOP and 6D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4932")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (experts=2, batch=2, layers=2, kv_heads=8, seq=4, head_dim=128)
        self_t = torch.ones(2, 2, 2, 8, 4, 128, dtype=torch.float32)
        src_t = torch.rand(2, 2, 2, 8, 2, 128, dtype=torch.float32) + 0.5
        idx_t = torch.randint(0, 4, (2, 2, 2, 8, 2, 128), dtype=torch.int64)

        def fn_oop(dest, index, source):
            return dest.scatter_reduce(
                4, index, source, reduce="sum", include_self=True
            )

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.scatter_reduce_(4, index, source, reduce="sum", include_self=True)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )


# =============================================================================
# INDEX_ADD
# =============================================================================
class TestIndexAddOp:
    """Extended gap scenarios for torch.index_add / Tensor.index_add_."""

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_add_channels_last_4d(self, mode: str):
        """channels-last 4D tensor index_add along channel dimension (dim=1).
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128) — channels=kv_heads=10, spatial=head_dim=128."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4874")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4874")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: kv_heads=10 as channels, head_dim=128 split as H=8, W=16
        self_t = torch.zeros(2, 10, 8, 16, dtype=torch.float32).to(
            memory_format=torch.channels_last
        )
        src_shape = list(self_t.shape)
        src_shape[1] = 4  # updating 4 of 10 kv-head channels
        src_t = torch.randn(*src_shape, dtype=torch.float32).to(
            memory_format=torch.channels_last
        )
        idx_t = torch.randperm(10)[:4]  # 4 unique channel indices

        def fn(dest, index, source):
            return torch.index_add(dest, 1, index, source)

        compare_with_cpu(
            fn, self_t, idx_t, src_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_add_nonzero_storage_offset(self, mode: str):
        """Non-zero storage_offset on sliced window inputs — pointer arithmetic correctness.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128)."""
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: kv_heads=8, head_dim=128 — pool of 64 kv-head rows, slice 8 rows
        pool_self = torch.zeros(64, 128, dtype=torch.bfloat16)
        pool_src = torch.randn(64, 128, dtype=torch.bfloat16)

        # Sub-views at non-zero offsets — each window is 8 kv-head rows of width 128
        self_w = pool_self.narrow(0, 16, 8)  # storage_offset = 16*128
        src_w = pool_src.narrow(0, 8, 4)  # storage_offset = 8*128
        idx_t = torch.randperm(8)[:4]

        def fn(dest, index, source):
            return torch.index_add(dest, 0, index, source)

        compare_with_cpu(
            fn,
            self_w,
            idx_t,
            src_w,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.05,
            rtol=0.05,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_add_1d_inplace_correctness(self, mode: str):
        """1D in-place index_add_ correctness — first positive correctness test for IP form.
        Model shape: Ministral-3-14B (head_dim=128). Covers 1D-IP correctness."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 3507")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4844")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: 1D head_dim=128 buffer — add 8 values at random positions
        self_t = torch.zeros(128, dtype=torch.bfloat16)
        src_t = torch.randn(8, dtype=torch.bfloat16)
        idx_t = torch.randperm(128, dtype=torch.int64)[:8]

        def fn(dest, index, source):
            out = dest.clone()
            out.index_add_(0, index, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_add_5d_both(self, mode: str):
        """5D index_add on (batch, layers, kv_heads, seq, head_dim) — both OOP and IP.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128). Covers 5D-OOP and 5D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4874")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4874")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (batch=2, layers=4, kv_heads=8, seq=8, head_dim=128) — add along seq (dim=3)
        self_t = torch.zeros(2, 4, 8, 8, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 4, 8, 4, 128, dtype=torch.bfloat16)
        idx_t = torch.randperm(8, dtype=torch.int64)[:4]

        def fn_oop(dest, index, source):
            return torch.index_add(dest, 3, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.index_add_(3, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_add_6d_both(self, mode: str):
        """6D index_add on expert-layer-KV-cache geometry — both OOP and IP.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128, bf16 equiv). Covers 6D-OOP and 6D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4874")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4874")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (experts=2, batch=2, layers=2, kv_heads=8, seq=4, head_dim=128)
        self_t = torch.zeros(2, 2, 2, 8, 4, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 2, 2, 8, 2, 128, dtype=torch.bfloat16)
        idx_t = torch.randperm(4, dtype=torch.int64)[:2]

        def fn_oop(dest, index, source):
            return torch.index_add(dest, 4, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.index_add_(4, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )


# =============================================================================
# INDEX_COPY
# =============================================================================
class TestIndexCopyOp:
    """Extended gap scenarios for torch.index_copy / Tensor.index_copy_."""

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_copy_5d_noncontiguous_strided(self, mode: str):
        """Non-contiguous 5D transposed destination — strided DMA coordinate correctness.
        Model shape: Granite-3.3-8B (batch=2, kv_heads=8, layers=4, seq=4, head_dim=128)."""
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (batch=2, kv_heads=8, layers=4, seq=4, head_dim=128)
        # transpose kv_heads and seq dims to exercise non-contiguous strided layout
        base = torch.zeros(2, 8, 4, 4, 128, dtype=torch.bfloat16)
        dest_t = base.transpose(1, 3)  # shape: (2, 4, 4, 8, 128), non-contiguous
        num_slices = 2
        src_shape = list(dest_t.shape)
        src_shape[2] = num_slices
        src_t = torch.randn(*src_shape, dtype=torch.bfloat16)
        idx_t = torch.tensor([0, 3], dtype=torch.int64)

        def fn(dest, index, source):
            out = dest.clone()
            out.index_copy_(2, index, source)
            return out

        compare_with_cpu(
            fn,
            dest_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.05,
            rtol=0.05,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_copy_zero_volume_trailing_dim(self, mode: str):
        """Zero-volume trailing dimension — index_copy must be a safe no-op.
        Model shape: Ministral-3-14B (kv_heads=8, seq=4, head_dim=0 zero-volume)."""
        if mode == "compile":
            pytest.xfail(reason="known issue- 4929")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: kv_heads=8 rows, seq=4 cols, trailing head_dim=0 (zero-volume)
        self_t = torch.zeros(8, 4, 0, dtype=torch.bfloat16)
        src_t = torch.zeros(8, 2, 0, dtype=torch.bfloat16)
        idx_t = torch.tensor([1, 3], dtype=torch.int64)

        def fn(dest, index, source):
            out = dest.clone()
            out.index_copy_(1, index, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_copy_unique_indices(self, mode: str):
        """Unique target indices — deterministic copy, CPU is a valid oracle.
        Model shape: Granite-4.1-20B (kv_heads=8, head_dim=128).

        Uses unique indices so there is no write-order ambiguity on any backend.
        compare_with_cpu is a correct oracle here because the result is fully
        determined regardless of execution order.
        """
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: kv_heads=8, head_dim=128 — 3 distinct rows copied to indices 1, 3, 5
        self_t = torch.zeros(8, 128, dtype=torch.bfloat16)
        src_t = torch.tensor(
            [[1.0] * 128, [2.0] * 128, [3.0] * 128], dtype=torch.bfloat16
        )
        idx_t = torch.tensor([1, 3, 5], dtype=torch.int64)

        def fn(dest, index, source):
            out = dest.clone()
            out.index_copy_(0, index, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_copy_1d_both(self, mode: str):
        """1D index_copy — both OOP and IP on a 1D head_dim vector.
        Model shape: Gemma-4-12B (head_dim=128). Covers 1D-OOP and 1D-IP."""
        torch.manual_seed(0)
        if mode == "compile":
            pytest.xfail(reason="known issue- 4399")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: head_dim=128 flat buffer — copy 4 positions
        self_t = torch.zeros(128, dtype=torch.bfloat16)
        src_t = torch.randn(4, dtype=torch.bfloat16)
        idx_t = torch.randperm(128, dtype=torch.int64)[:4]

        def fn_oop(dest, index, source):
            return torch.index_copy(dest, 0, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.index_copy_(0, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_copy_4d_both(self, mode: str):
        """4D index_copy on (batch, kv_heads, seq, head_dim) — both OOP and IP.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256). Covers 4D-OOP and 4D-IP."""
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: (batch=2, kv_heads=16, seq=8, head_dim=256) — copy 4 seq positions (dim=2)
        self_t = torch.zeros(2, 16, 8, 256, dtype=torch.bfloat16)
        src_t = torch.randn(2, 16, 4, 256, dtype=torch.bfloat16)
        idx_t = torch.randperm(8, dtype=torch.int64)[:4]

        def fn_oop(dest, index, source):
            return torch.index_copy(dest, 2, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.index_copy_(2, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_copy_6d_both(self, mode: str):
        """6D index_copy on expert-layer-KV-cache geometry — both OOP and IP.
        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128). Covers 6D-OOP and 6D-IP."""
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: (experts=2, batch=2, layers=2, kv_heads=8, seq=4, head_dim=128)
        self_t = torch.zeros(2, 2, 2, 8, 4, 128, dtype=torch.bfloat16)
        src_t = torch.randn(2, 2, 2, 8, 2, 128, dtype=torch.bfloat16)
        idx_t = torch.randperm(4, dtype=torch.int64)[:2]

        def fn_oop(dest, index, source):
            return torch.index_copy(dest, 4, index, source)

        def fn_ip(dest, index, source):
            out = dest.clone()
            out.index_copy_(4, index, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )


# =============================================================================
# INDEX_FILL
# =============================================================================
class TestIndexFillOp:
    """Extended gap scenarios for torch.index_fill / Tensor.index_fill_."""

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("fill_val", [float("nan"), float("inf"), float("-inf")])
    def test_index_fill_special_ieee754_values(self, mode: str, fill_val: float):
        """Special IEEE-754 fills — NaN, +Inf, -Inf bit-pattern preservation.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256)."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4414")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4472")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: kv_heads=16 rows, head_dim=256 cols — fill 2 kv-head rows
        self_t = torch.ones(16, 256, dtype=torch.float32)
        idx_t = torch.tensor([0, 8], dtype=torch.int64)

        def fn(dest, index):
            return torch.index_fill(dest, 0, index, fill_val)

        compare_with_cpu(
            fn, self_t, idx_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_fill_5d_hyperplane(self, mode: str):
        """5D tensor hyperplane broadcast fill along an intermediate dim.
        Model shape: Gemma-4-12B (batch=2, kv_heads=10, layers=4, seq=8, head_dim=128)."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4414")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4472")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: (batch=2, kv_heads=10, layers=4, seq=8, head_dim=128) — fill 3 layer slices
        self_t = torch.zeros(2, 10, 4, 8, 128, dtype=torch.bfloat16)
        idx_t = torch.tensor([0, 2, 3], dtype=torch.int64)

        def fn(dest, index):
            return torch.index_fill(dest, 2, index, 42.0)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_fill_nonunit_strided_slice(self, mode: str):
        """Non-unit strided slice fill — interleaved rows must remain unmodified.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128) — step-3 stride across kv-heads."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4414")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4472")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: 24 kv-head rows of width 128; step-3 view gives 8 rows
        base = torch.randn(24, 128, dtype=torch.bfloat16)
        idx_t = torch.tensor([0, 1], dtype=torch.int64)

        def fn(base_tensor, index):
            out = base_tensor.clone()
            view = out[::3, :]  # step-3 stride along dim 0 — shape (8, 128)
            view.index_fill_(0, index, 99.0)
            return out

        compare_with_cpu(
            fn,
            base,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_fill_complex64_scalar(self, mode: str):
        """Complex scalar fill on complex64 — real and imaginary channels updated correctly.
        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128) — complex64 over kv_heads×head_dim.

        The existing dtype sweep uses float32/int32/bool fills. This test
        explicitly exercises complex scalar broadcasting to verify that both
        the real and imaginary interleaved channels are written without
        stomping on each other or leaving one channel at its original value.
        """
        if mode == "eager":
            pytest.xfail(reason="known issue- 4414")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4472")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: kv_heads=8 rows, head_dim=128 cols — complex64 fill on 2 kv-head rows
        self_t = torch.zeros(8, 128, dtype=torch.complex64)
        idx_t = torch.tensor([0, 4], dtype=torch.int64)
        fill_val = complex(3.5, -2.1)

        def fn(dest, index):
            return dest.index_fill(0, index, fill_val)

        compare_with_cpu(
            fn, self_t, idx_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_fill_1d_both(self, mode: str):
        """1D index_fill — both OOP and IP on a 1D head_dim vector.
        Model shape: Granite-4.1-20B (head_dim=128). Covers 1D-OOP and 1D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4414")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4399")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: head_dim=128 flat buffer — fill 8 positions with 99.0
        self_t = torch.zeros(128, dtype=torch.bfloat16)
        idx_t = torch.randperm(128, dtype=torch.int64)[:8]

        def fn_oop(dest, index):
            return torch.index_fill(dest, 0, index, 99.0)

        def fn_ip(dest, index):
            out = dest.clone()
            out.index_fill_(0, index, 99.0)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_fill_4d_both(self, mode: str):
        """4D index_fill on (batch, kv_heads, seq, head_dim) — both OOP and IP.
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128). Covers 4D-OOP and 4D-IP."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4414")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4472")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: (batch=2, kv_heads=10, seq=8, head_dim=128) — fill 4 seq positions (dim=2)
        self_t = torch.zeros(2, 10, 8, 128, dtype=torch.bfloat16)
        idx_t = torch.tensor([1, 3, 5, 7], dtype=torch.int64)

        def fn_oop(dest, index):
            return torch.index_fill(dest, 2, index, 42.0)

        def fn_ip(dest, index):
            out = dest.clone()
            out.index_fill_(2, index, 42.0)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_fill_6d_both(self, mode: str):
        """6D index_fill on expert-layer-KV-cache geometry — both OOP and IP.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256). Covers 6D-OOP and 6D-IP."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4414")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4472")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: (experts=2, batch=2, layers=2, kv_heads=16, seq=4, head_dim=256) — fill 2 seq positions (dim=4)
        self_t = torch.zeros(2, 2, 2, 16, 4, 256, dtype=torch.float32)
        idx_t = torch.tensor([1, 3], dtype=torch.int64)

        def fn_oop(dest, index):
            return torch.index_fill(dest, 4, index, 7.0)

        def fn_ip(dest, index):
            out = dest.clone()
            out.index_fill_(4, index, 7.0)
            return out

        compare_with_cpu(
            fn_oop, self_t, idx_t, run_eager=run_eager, run_compile=run_compile
        )
        compare_with_cpu(
            fn_ip, self_t, idx_t, run_eager=run_eager, run_compile=run_compile
        )


# =============================================================================
# INDEX_SELECT
# =============================================================================
class TestIndexSelectOp:
    """Extended gap scenarios for torch.index_select."""

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_select_5d_intermediate_dim(self, mode: str):
        """5D tensor gathering along intermediate dimension (dim=2).
        Model shape: Granite-4.1-20B (batch=2, kv_heads=8, layers=8, seq=4, head_dim=128)."""
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: (batch=2, kv_heads=8, layers=8, seq=4, head_dim=128) — select 4 layers
        self_t = torch.randn(2, 8, 8, 4, 128, dtype=torch.bfloat16)
        idx_t = torch.tensor([0, 2, 5, 7], dtype=torch.int64)

        def fn(source, index):
            return torch.index_select(source, 2, index)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.05,
            rtol=0.05,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_select_out_aliases_input(self, mode: str):
        """out= buffer is a sub-slice of the input tensor — memory-overlap hazard.

        x[:4] is passed as the out= buffer while x itself is the source.
        PyTorch must either handle the overlap correctly (producing the same
        result as a fresh allocation) or raise a descriptive error. The test
        accepts either outcome — what it guards against is silent data
        corruption producing a result that differs from a clean gather.
        """
        run_eager, run_compile = _mode_flags(mode)

        x = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
        idx_t = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
        # Reference: clean gather before any mutation
        expected = x[idx_t].clone()

        def run(source, index):
            # out aliases the first 4 rows of source — genuine memory overlap
            out = source[:4]
            try:
                torch.index_select(source, 0, index, out=out)
                return out.clone()
            except RuntimeError:
                # Overlap detected and rejected by backend — acceptable outcome
                return expected.clone().to(source.device)
            # NotImplementedError is intentionally NOT caught — an unsupported op
            # is a real backend gap and should fail the test visibly

        # CPU reference
        cpu_result = run(x.clone(), idx_t)

        # Spyre
        if run_eager:
            spyre_result = run(x.clone().to(DEVICE), idx_t.to(DEVICE)).cpu()
            torch.testing.assert_close(spyre_result, cpu_result, atol=0.0, rtol=0.0)
        if run_compile:
            compiled_run = torch.compile(run)
            spyre_result = compiled_run(x.clone().to(DEVICE), idx_t.to(DEVICE)).cpu()
            torch.testing.assert_close(spyre_result, cpu_result, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_select_massive_repeated_single_row(self, mode: str):
        """10 000-element index all pointing to row 0 — cache-broadcast correctness."""
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        self_t = torch.randn(8, 64, dtype=torch.float32)
        idx_t = torch.zeros(10_000, dtype=torch.int64)  # all point to row 0

        def fn(source, index):
            return torch.index_select(source, 0, index)

        compare_with_cpu(
            fn, self_t, idx_t, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_select_dynamic_index_length(self, mode: str):
        """Dynamic index length — compiled graph must handle varying K without recompilation.
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128).

        Calls the compiled function three times with index tensors of different
        lengths (K=2, 5, 10) to verify TorchInductor emits a kernel that
        accepts a symbolic index size rather than specialising on a fixed K.
        In eager mode this simply validates correctness across all three sizes.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: kv_heads=10 rows of width head_dim=128
        x = torch.randn(10, 128, dtype=torch.bfloat16)

        def fn(source, index):
            return torch.index_select(source, 0, index)

        compiled_fn = (
            torch.compile(fn) if (run_compile and hasattr(torch, "compile")) else None
        )

        for k in (2, 5, 10):
            idx_t = torch.randperm(10)[:k]

            if run_eager:
                compare_with_cpu(
                    fn,
                    x,
                    idx_t,
                    run_eager=True,
                    run_compile=False,
                    atol=1e-2,
                    rtol=1.6e-2,
                )

            if compiled_fn is not None:
                x_dev = x.to(DEVICE)
                idx_dev = idx_t.to(DEVICE)
                expected = fn(x, idx_t)  # CPU reference for this k
                result = compiled_fn(x_dev, idx_dev)
                torch.testing.assert_close(
                    result.cpu(), expected, atol=1e-2, rtol=1.6e-2
                )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_select_1d(self, mode: str):
        """1D index_select on a head_dim vector (read-only op, OOP only).
        Model shape: Granite-3.3-8B (head_dim=128). Covers 1D-OOP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4956")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4956")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: head_dim=128 — select 8 positions
        self_t = torch.randn(128, dtype=torch.bfloat16)
        idx_t = torch.randperm(128, dtype=torch.int64)[:8]

        def fn(source, index):
            return torch.index_select(source, 0, index)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_select_4d(self, mode: str):
        """4D index_select on (batch, kv_heads, seq, head_dim) (OOP only).
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256). Covers 4D-OOP."""
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: (batch=2, kv_heads=16, seq=8, head_dim=256) — select 4 seq positions (dim=2)
        self_t = torch.randn(2, 16, 8, 256, dtype=torch.bfloat16)
        idx_t = torch.tensor([0, 2, 4, 6], dtype=torch.int64)

        def fn(source, index):
            return torch.index_select(source, 2, index)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_select_6d(self, mode: str):
        """6D index_select on expert-layer-KV-cache geometry (OOP only).
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128). Covers 6D-OOP."""
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: (experts=2, batch=2, layers=2, kv_heads=10, seq=4, head_dim=128) — select 2 seq positions (dim=4)
        self_t = torch.randn(2, 2, 2, 10, 4, 128, dtype=torch.bfloat16)
        idx_t = torch.tensor([1, 3], dtype=torch.int64)

        def fn(source, index):
            return torch.index_select(source, 4, index)

        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )


# =============================================================================
# INDEX_PUT
# =============================================================================
class TestIndexPutOp:
    """Extended gap scenarios for torch.index_put / Tensor.index_put_."""

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_put_5d_coordinate_tuple(self, mode: str):
        """5D coordinate index tuple — 4 index vectors addressing 4 dims of a 5D tensor.
        Model shape: Granite-3.3-8B (batch=2, kv_heads=8, layers=4, seq=4, head_dim=128)."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 692")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4451")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (batch=2, kv_heads=8, layers=4, seq=4, head_dim=128)
        # 4-index tuple addressing batch × kv_heads × layers × seq, leaving head_dim free.
        # Unique flat coords over the 4 indexed dims (2×8×4×4=256 total slots) → no duplicates.
        self_t = torch.zeros(2, 8, 4, 4, 128, dtype=torch.bfloat16)
        n = 16
        flat = torch.randperm(2 * 8 * 4 * 4, dtype=torch.int64)[:n]
        i0 = flat // (8 * 4 * 4)
        i1 = (flat // (4 * 4)) % 8
        i2 = (flat // 4) % 4
        i3 = flat % 4
        vals = torch.randn(n, 128, dtype=torch.bfloat16)

        def fn(dest, a, b, c, d, v):
            out = dest.clone()
            out.index_put_((a, b, c, d), v, accumulate=False)
            return out

        compare_with_cpu(
            fn,
            self_t,
            i0,
            i1,
            i2,
            i3,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_put_sparse_3d_bool_mask_transposed(self, mode: str):
        """Sparse 3D boolean mask on a transposed (non-contiguous) tensor.
        Model shape: Ministral-3-14B (kv_heads=8, seq=4, head_dim=128)."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 692")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4870")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: (seq=4, kv_heads=8, head_dim=128) transposed to (kv_heads=8, seq=4, head_dim=128)
        torch.manual_seed(0)
        base = torch.zeros(4, 8, 128, dtype=torch.bfloat16)
        dest_t = base.transpose(
            0, 1
        ).contiguous()  # shape (8, 4, 128), contiguous after transpose
        mask = torch.rand(8, 4, 128) > 0.85  # sparse True positions (~15% density)
        num_true = int(mask.sum().item())
        vals = torch.randn(num_true, dtype=torch.bfloat16)

        def fn(dest, m, v):
            out = dest.clone()
            out.index_put_((m,), v, accumulate=False)
            return out

        compare_with_cpu(
            fn,
            dest_t,
            mask,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_index_put_half_format_heavy_collision(self, mode: str, dtype: torch.dtype):
        """Heavy accumulation collision in float16 and bfloat16.
        Model shape: Granite-4.1-20B (kv_heads=8, head_dim=128).

        5 000 updates targeting exactly 8 kv-head rows (625 collisions per row)
        exercises half-precision accumulator rounding under maximum collision
        contention. Covers both float16 and bfloat16 — distinct from the base
        file's bfloat16-only test with low-contention random indices.
        """
        if mode == "eager":
            pytest.xfail(reason="known issue- 692")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4411")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: kv_heads=8 rows of width head_dim=128
        num_bins, updates_per_bin = 8, 625
        num_updates = num_bins * updates_per_bin
        self_t = torch.zeros(num_bins, 128, dtype=dtype)
        # Round-robin assignment: update k targets row k % num_bins
        idx_t = torch.arange(num_updates, dtype=torch.int64) % num_bins
        vals = torch.ones(num_updates, 128, dtype=dtype)

        def fn(dest, index, v):
            out = dest.clone()
            out.index_put_((index,), v, accumulate=True)
            return out

        # atol=1.0 accounts for half-precision rounding accumulating over 500 adds
        compare_with_cpu(
            fn,
            self_t,
            idx_t,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1.0,
            rtol=1e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_full_tensor_slice_assignment(self, mode: str):
        """Full-tensor slice assignment out[:] = v — broadcasts scalar-filled tensor to all positions.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256).

        This is not an index_put test.  index_put_((), v) is the semantic equivalent
        but crashes with an internal assert on CPU in PyTorch 2.13 (IndexKernelUtils.h:8).
        This test documents the workaround (out[:] = v) and validates the slice-assignment
        path on Spyre for full-tensor overwrites.
        """
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: kv_heads=16 rows, head_dim=256 cols — full-tensor replace
        self_t = torch.zeros(16, 256, dtype=torch.bfloat16)
        vals = torch.ones(16, 256, dtype=torch.bfloat16) * 7.0

        def fn(dest, v):
            out = dest.clone()
            out[:] = v
            return out

        compare_with_cpu(
            fn,
            self_t,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_put_1d_both(self, mode: str):
        """1D index_put — both OOP and IP on a 1D head_dim vector.
        Model shape: Ministral-3-14B (head_dim=128). Covers 1D-OOP and 1D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 692")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4399")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: head_dim=128 flat buffer — write 8 positions
        self_t = torch.zeros(128, dtype=torch.bfloat16)
        idx_t = torch.randperm(128, dtype=torch.int64)[:8]
        vals = torch.randn(8, dtype=torch.bfloat16)

        def fn_oop(dest, index, v):
            return torch.index_put(dest, (index,), v, accumulate=False)

        def fn_ip(dest, index, v):
            out = dest.clone()
            out.index_put_((index,), v, accumulate=False)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            idx_t,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            idx_t,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_put_4d_both(self, mode: str):
        """4D index_put on (batch, kv_heads, seq, head_dim) — both OOP and IP.
        Model shape: Granite-4.1-20B (kv_heads=8, head_dim=128). Covers 4D-OOP and 4D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 692")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4451")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: (batch=2, kv_heads=8, seq=8, head_dim=128)
        # 3-index tuple over batch × kv_heads × seq (2×8×8=128 total slots) → unique coords.
        self_t = torch.zeros(2, 8, 8, 128, dtype=torch.bfloat16)
        n = 4
        flat = torch.randperm(2 * 8 * 8, dtype=torch.int64)[:n]
        i0 = flat // (8 * 8)
        i1 = (flat // 8) % 8
        i2 = flat % 8
        vals = torch.randn(n, 128, dtype=torch.bfloat16)

        def fn_oop(dest, a, b, c, v):
            return torch.index_put(dest, (a, b, c), v, accumulate=False)

        def fn_ip(dest, a, b, c, v):
            out = dest.clone()
            out.index_put_((a, b, c), v, accumulate=False)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            i0,
            i1,
            i2,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            i0,
            i1,
            i2,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_index_put_6d_both(self, mode: str):
        """6D index_put on expert-layer-KV-cache geometry — both OOP and IP.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256). Covers 6D-OOP and 6D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 692")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4451")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: (experts=2, batch=2, layers=2, kv_heads=16, seq=4, head_dim=256)
        # 5-index tuple over experts × batch × layers × kv_heads × seq (2×2×2×16×4=512 slots)
        # → unique flat coords, no write-order ambiguity with accumulate=False.
        self_t = torch.zeros(2, 2, 2, 16, 4, 256, dtype=torch.bfloat16)
        n = 8
        flat = torch.randperm(2 * 2 * 2 * 16 * 4, dtype=torch.int64)[:n]
        i0 = flat // (2 * 2 * 16 * 4)
        i1 = (flat // (2 * 16 * 4)) % 2
        i2 = (flat // (16 * 4)) % 2
        i3 = (flat // 4) % 16
        i4 = flat % 4
        vals = torch.randn(n, 256, dtype=torch.bfloat16)

        def fn_oop(dest, a, b, c, d, e, v):
            return torch.index_put(dest, (a, b, c, d, e), v, accumulate=False)

        def fn_ip(dest, a, b, c, d, e, v):
            out = dest.clone()
            out.index_put_((a, b, c, d, e), v, accumulate=False)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            i0,
            i1,
            i2,
            i3,
            i4,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            i0,
            i1,
            i2,
            i3,
            i4,
            vals,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )


# =============================================================================
# MASKED_SCATTER
# =============================================================================
class TestMaskedScatterOp:
    """Extended gap scenarios for torch.masked_scatter / Tensor.masked_scatter_."""

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_all_true_full_replacement(self, mode: str):
        """All-True dense mask — entire tensor replaced by sequential source elements.
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128) — 1280-element flat replacement."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: kv_heads=10 × head_dim=128 = 1280 elements — all replaced sequentially
        self_t = torch.zeros(10, 128, dtype=torch.bfloat16)
        mask = torch.ones(10, 128, dtype=torch.bool)
        src_t = torch.arange(1280, dtype=torch.bfloat16)

        def fn(dest, m, source):
            return torch.masked_scatter(dest, m, source)

        compare_with_cpu(
            fn,
            self_t,
            mask,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_5d_transposed_target(self, mode: str):
        """True non-contiguous 5D target — row-major traversal of mask True bits.
        Model shape: Granite-3.3-8B (batch=2, kv_heads=8, layers=4, seq=4, head_dim=128).

        The destination is base.transpose(1, 3) with NO .contiguous() call, so
        it is a genuinely non-contiguous tensor (strides are non-unit and not
        in descending order).  masked_scatter must traverse both dest and mask
        in C-contiguous logical order regardless of the physical memory layout.
        CPU is the reference; the result is compared element-wise via
        compare_with_cpu.
        """
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (batch=2, kv_heads=8, layers=4, seq=4, head_dim=128)
        # transpose kv_heads and seq to make a genuinely non-contiguous 5D layout
        torch.manual_seed(1)
        base = torch.zeros(2, 8, 4, 4, 128, dtype=torch.bfloat16)
        dest_t = base.transpose(1, 3)  # shape (2,4,4,8,128), genuinely non-contiguous
        assert not dest_t.is_contiguous(), "dest must be non-contiguous for this test"
        mask = torch.rand(*dest_t.shape) > 0.75
        num_true = int(mask.sum().item())
        src_t = torch.randn(num_true, dtype=torch.bfloat16)

        def fn(dest, m, source):
            return torch.masked_scatter(dest, m, source)

        compare_with_cpu(
            fn,
            dest_t,
            mask,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.05,
            rtol=0.05,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_dynamic_shapes_inductor(self, mode: str):
        """Dynamic batch dimension and dynamic source length under TorchInductor.
        Model shape: Ministral-3-14B (hidden=5120, kv_heads=8) — dynamic (B, S) slices.

        masked_scatter requires a prefix-sum scan over the mask to compute per-
        element source offsets. This test verifies TorchInductor can compile
        that scan kernel with a symbolic batch size B and symbolic True-count K
        rather than specialising on fixed constants.

        The compiled function is called three times with distinct (B, S) pairs —
        (8, 40), (4, 20), (6, 32) — varying both dimensions to exercise
        guard-free re-use of a single compiled kernel. In eager mode the test
        simply validates correctness across all three sizes.
        """
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        def fn(dest, m, source):
            return torch.masked_scatter(dest, m, source)

        compiled_fn = (
            torch.compile(fn) if (run_compile and hasattr(torch, "compile")) else None
        )

        # Ministral-3-14B: hidden=5120 — dynamic batch × seq slices of a fixed hidden dim
        # Three distinct (B, S) pairs vary both B and S to exercise guard-free kernel reuse.
        torch.manual_seed(7)
        for B, S in ((8, 40), (4, 20), (6, 32)):
            self_t = torch.zeros(B, S, dtype=torch.bfloat16)
            mask = torch.rand(B, S) > 0.5  # ~50% True — K varies with (B, S)
            num_true = int(mask.sum().item())
            src_t = torch.randn(num_true, dtype=torch.bfloat16)

            expected = fn(self_t, mask, src_t)  # CPU reference

            if run_eager:
                compare_with_cpu(
                    fn,
                    self_t,
                    mask,
                    src_t,
                    run_eager=True,
                    run_compile=False,
                    atol=1e-2,
                    rtol=1.6e-2,
                )

            if compiled_fn is not None:
                result = compiled_fn(self_t, mask, src_t)
                torch.testing.assert_close(result.cpu(), expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_0d_inplace(self, mode: str):
        """0D in-place masked_scatter_ — scalar self with scalar True mask.
        Covers 0D-IP (existing 0D test uses OOP form only)."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        x = torch.tensor(0.0, dtype=torch.float32)
        mask = torch.tensor(True)
        src = torch.tensor([42.0], dtype=torch.float32)

        def fn(d, m, s):
            out = d.clone()
            out.masked_scatter_(m, s)
            return out

        compare_with_cpu(fn, x, mask, src, run_eager=run_eager, run_compile=run_compile)

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_4d_inplace(self, mode: str):
        """4D in-place masked_scatter_ — (batch, kv_heads, seq, head_dim).
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128). Covers 4D-IP."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (batch=2, kv_heads=8, seq=8, head_dim=128) — ~25% True density
        torch.manual_seed(5)
        self_t = torch.zeros(2, 8, 8, 128, dtype=torch.bfloat16)
        mask = torch.rand(2, 8, 8, 128) > 0.75
        src_t = torch.randn(int(mask.sum().item()), dtype=torch.bfloat16)

        def fn(dest, m, source):
            out = dest.clone()
            out.masked_scatter_(m, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            mask,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_5d_inplace(self, mode: str):
        """5D in-place masked_scatter_ — (batch, layers, kv_heads, seq, head_dim).
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128, bf16 equiv). Covers 5D-IP."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (batch=2, layers=4, kv_heads=8, seq=4, head_dim=128) — ~25% True
        torch.manual_seed(6)
        self_t = torch.zeros(2, 4, 8, 4, 128, dtype=torch.bfloat16)
        mask = torch.rand(2, 4, 8, 4, 128) > 0.75
        src_t = torch.randn(int(mask.sum().item()), dtype=torch.bfloat16)

        def fn(dest, m, source):
            out = dest.clone()
            out.masked_scatter_(m, source)
            return out

        compare_with_cpu(
            fn,
            self_t,
            mask,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_6d_both(self, mode: str):
        """6D masked_scatter — both OOP and IP on expert-layer-KV-cache geometry.
        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128). Covers 6D-OOP and 6D-IP."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: (experts=2, batch=2, layers=2, kv_heads=8, seq=4, head_dim=128) — ~25% True
        torch.manual_seed(7)
        self_t = torch.zeros(2, 2, 2, 8, 4, 128, dtype=torch.bfloat16)
        mask = torch.rand(2, 2, 2, 8, 4, 128) > 0.75
        src_t = torch.randn(int(mask.sum().item()), dtype=torch.bfloat16)

        def fn_oop(dest, m, source):
            return torch.masked_scatter(dest, m, source)

        def fn_ip(dest, m, source):
            out = dest.clone()
            out.masked_scatter_(m, source)
            return out

        compare_with_cpu(
            fn_oop,
            self_t,
            mask,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            self_t,
            mask,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )


# =============================================================================
# TORCH.PUT / PUT_
# =============================================================================
class TestPutOp:
    """Op-level tests for torch.put / Tensor.put_ — flat-index write into flattened tensor view.

    Covers:
      PUT-01a/b/c  baseline contig/non-contig dst, reshaped index
      PUT-02       all-zero index hotspot stress (accumulate=True)
      PUT-03       empty index no-op
      PUT-04/05    duplicate index error paths (eager only)
      PUT-06       3D flat-index address translation
      PUT-NEW-01/02/03  4D/5D/6D model-shaped geometries
      PUT-GAP-A    dtype sweep — float16, bfloat16
      PUT-GAP-B    non-contiguous src / index / all-three
      PUT-GAP-C    all 8 scalar/one-element shape combos
      PUT-GAP-D    empty destination shapes
      PUT-GAP-E    large parallel accumulation (grainsize > 3000)
    """

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    def test_put_1d_dst_contig(self, mode: str, accumulate: bool):
        """PUT-01a: 1D contiguous dst — baseline flat-index write, both accumulate modes.
        Model shape: Granite-3.3-8B — 1D flat KV cache slot vector of length head_dim=128."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4844")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: head_dim=128 as a flat 1D cache slot buffer
        dst = torch.zeros(128, dtype=torch.bfloat16)
        src = torch.randn(8, dtype=torch.bfloat16)
        if accumulate:
            idx = torch.randint(0, 128, (8,), dtype=torch.int64)
        else:
            idx = torch.randperm(128, dtype=torch.int64)[:8]

        def fn(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    def test_put_2d_dst_noncontig(self, mode: str, accumulate: bool):
        """PUT-01b: 2D non-contiguous dst — flat index must respect multi-stride address layout.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128) — stride-2 view over kv-heads."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4636")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: kv_heads=8, head_dim=128 — stride-2 view leaves every other head
        base = torch.zeros(16, 128, dtype=torch.bfloat16)
        dst = base[::2, :]  # stride-2 view, shape (8, 128), non-contiguous
        src = torch.randn(4, dtype=torch.bfloat16)
        if accumulate:
            idx = torch.randint(0, dst.numel(), (4,), dtype=torch.int64)
        else:
            idx = torch.randperm(dst.numel(), dtype=torch.int64)[:4]

        def fn(d, i, s):
            return torch.put(d.clone(), i, s, accumulate=accumulate)

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    def test_put_reshaped_index(self, mode: str, accumulate: bool):
        """PUT-01c: 2D reshaped index tensor — put must accept non-1D index of same numel.
        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128) — 2×4 reshaped index into kv cache."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4636")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: kv_heads=8 × head_dim=128 flat — 2×4 index addresses 8 positions
        dst = torch.zeros(8, 128, dtype=torch.bfloat16)
        src = torch.randn(8, dtype=torch.bfloat16)
        if accumulate:
            idx = torch.randint(0, dst.numel(), (2, 4), dtype=torch.int64)
        else:
            idx = torch.randperm(dst.numel(), dtype=torch.int64)[:8].reshape(2, 4)

        def fn(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_put_allzero_index_hotspot(self, mode: str, dtype: torch.dtype):
        """PUT-02: All-zero index hotspot stress — accumulate=True, all updates target slot 0.

        Result must equal orig + source.sum() within half-precision rounding tolerance.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4930")
        run_eager, run_compile = _mode_flags(mode)

        size = 100 if dtype in (torch.float16, torch.bfloat16) else 200
        orig = torch.randn(1, dtype=dtype)
        idx = torch.zeros(size, dtype=torch.int64)
        src = torch.randn(size, dtype=dtype)
        atol = 0.5 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        rtol = 0.1 if dtype in (torch.float16, torch.bfloat16) else 1e-3

        def fn(d, i, s):
            return d.put(i, s, accumulate=True)

        compare_with_cpu(
            fn,
            orig,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    def test_put_empty_index(self, mode: str, accumulate: bool):
        """PUT-03: Zero-element index and source — dst must be returned bit-for-bit unchanged.
        Model shape: Granite-4.1-20B (kv_heads=8, head_dim=128)."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540, 4930")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4636")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: kv_heads=8 × head_dim=128 — empty index is a no-op
        dst = torch.randn(8, 128, dtype=torch.bfloat16)
        idx = torch.zeros(0, dtype=torch.int64)
        src = torch.zeros(0, dtype=torch.bfloat16)

        def fn(d, i, s):
            out = d.clone()
            out.put_(i, s, accumulate=accumulate)
            return out

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    def test_put_duplicate_index_no_accumulate_raises(self):
        """PUT-04: Duplicate indices + accumulate=False → nondeterminism RuntimeError (eager only).

        Duplicate-index detection is an eager-only runtime check. torch.compile traces
        the op statically and never inspects actual index values, so it does not raise —
        this is documented PyTorch behaviour. Only eager mode is tested here.
        """
        torch.manual_seed(0)
        pytest.xfail(reason="known issue- 4540")
        a = torch.randn(10, dtype=torch.float32)
        idx = torch.tensor([0, 0], dtype=torch.int64)
        vals = torch.tensor([0.0, 1.0], dtype=torch.float32)

        with pytest.raises(RuntimeError):
            a.put(idx, vals, accumulate=False)
        with pytest.raises(RuntimeError):
            a.put_(idx, vals, accumulate=False)

    def test_put_duplicate_index_accumulate_raises(self):
        """PUT-05: Duplicate indices + accumulate=True → nondeterminism RuntimeError (eager only).

        Same as PUT-04: duplicate-index detection only happens at eager runtime.
        torch.compile does not raise for this case.
        """
        torch.manual_seed(0)
        pytest.xfail(reason="known issue- 4540")
        a = torch.randn(10, dtype=torch.float32)
        idx = torch.tensor([0, 0], dtype=torch.int64)
        vals = torch.tensor([0.0, 1.0], dtype=torch.float32)

        with pytest.raises(RuntimeError):
            a.put(idx, vals, accumulate=True)
        with pytest.raises(RuntimeError):
            a.put_(idx, vals, accumulate=True)

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_put_3d_flat_index_address_translation(self, mode: str):
        """PUT-06: 3D tensor flat-index address translation — flat index maps through 3D multi-stride layout.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256) — 3D cache: (pages=2, kv_heads=8, head_dim=16)."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4636")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: page-table shaped (pages=2, kv_heads=8, head_dim=16) — 256 elements flat
        # flat index k → 3D position (k//128, (k//16)%8, k%16)
        dst = torch.zeros(2, 8, 16, dtype=torch.bfloat16)
        # 4 known flat positions mapping to distinct 3D coordinates
        idx = torch.tensor([0, 64, 128, 255], dtype=torch.int64)
        src = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.bfloat16)

        def fn(d, i, s):
            return torch.put(d, i, s, accumulate=False)

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    def test_put_4d_outofplace(self, mode: str, accumulate: bool):
        """4D out-of-place torch.put on (batch, kv_heads, seq, head_dim).
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128). Covers 4D-OOP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4399")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: (batch=2, kv_heads=10, seq=4, head_dim=128) — 1280 flat elements
        dst = torch.zeros(2, 10, 4, 128, dtype=torch.bfloat16)
        src = torch.randn(8, dtype=torch.bfloat16)
        if accumulate:
            idx = torch.randint(0, dst.numel(), (8,), dtype=torch.int64)
        else:
            idx = torch.randperm(dst.numel(), dtype=torch.int64)[:8]

        def fn(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    def test_put_5d_both(self, mode: str, accumulate: bool):
        """5D torch.put and put_ — both OOP and IP on layer-KV-cache geometry.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128). Covers 5D-OOP and 5D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4399")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (batch=2, layers=4, kv_heads=8, seq=4, head_dim=128) — 32768 flat elements
        dst = torch.zeros(2, 4, 8, 4, 128, dtype=torch.bfloat16)
        src = torch.randn(8, dtype=torch.bfloat16)
        if accumulate:
            idx = torch.randint(0, dst.numel(), (8,), dtype=torch.int64)
        else:
            idx = torch.randperm(dst.numel(), dtype=torch.int64)[:8]

        def fn_oop(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        def fn_ip(d, i, s):
            out = d.clone()
            out.put_(i, s, accumulate=accumulate)
            return out

        compare_with_cpu(
            fn_oop,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )
        compare_with_cpu(
            fn_ip,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    def test_put_6d_both(self, mode: str, accumulate: bool):
        """6D torch.put and put_ on expert-layer-KV-cache geometry — both OOP and IP.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128, bf16 equiv). Covers 6D-OOP and 6D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4399")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: (experts=2, batch=2, layers=2, kv_heads=8, seq=4, head_dim=128)
        dst = torch.zeros(2, 2, 2, 8, 4, 128, dtype=torch.bfloat16)
        src = torch.randn(8, dtype=torch.bfloat16)
        if accumulate:
            idx = torch.randint(0, dst.numel(), (8,), dtype=torch.int64)
        else:
            idx = torch.randperm(dst.numel(), dtype=torch.int64)[:8]

        def fn_oop(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        def fn_ip(d, i, s):
            out = d.clone()
            out.put_(i, s, accumulate=accumulate)
            return out

        compare_with_cpu(
            fn_oop,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )
        compare_with_cpu(
            fn_ip,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    # -------------------------------------------------------------------------
    # PUT-GAP-A: dtype sweep — float16, bfloat16
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_put_dtype_sweep(self, mode: str, accumulate: bool, dtype: torch.dtype):
        """PUT-GAP-A: put/put_ correctness across float16 and bfloat16.
        Model shape: Ministral-3-14B — 1D head_dim=128 and 2D kv_heads×head_dim=8×128.

        The upstream PyTorch test_put covers all_types_and_complex_and(half, bfloat16).
        The existing TestPutOp only explicitly tests fp16/bf16/fp32 in the hotspot
        scenario.  This test exercises the general flat-index write path for the two
        half-precision dtypes supported on Spyre. complex64/complex128 and float64
        are not supported on Spyre and are excluded.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4844")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: head_dim=128 (1D) and kv_heads×head_dim=8×128=1024 (2D) shapes
        atol, rtol = 1e-2, 1e-2

        def make(*shape):
            if dtype.is_floating_point or dtype.is_complex:
                return torch.randn(*shape, dtype=dtype)
            return torch.zeros(*shape, dtype=dtype)

        for dst_shape in [(128,), (8, 128)]:
            dst = make(*dst_shape)
            src = make(8)
            if accumulate:
                idx = torch.randint(0, dst.numel(), (8,), dtype=torch.int64)
            else:
                idx = torch.randperm(dst.numel(), dtype=torch.int64)[:8]

            def fn(d, i, s):
                return torch.put(d, i, s, accumulate=accumulate)

            compare_with_cpu(
                fn,
                dst,
                idx,
                src,
                run_eager=run_eager,
                run_compile=run_compile,
                atol=atol,
                rtol=rtol,
            )

    # -------------------------------------------------------------------------
    # PUT-GAP-B: non-contiguous source, non-contiguous index, all-three
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    def test_put_noncontig_source(self, mode: str, accumulate: bool):
        """PUT-GAP-B (i): Non-contiguous source tensor.
        Model shape: Granite-4.1-20B (head_dim=128) — 1D flat KV slot buffer.

        Source is a step-2 slice of a larger 1-D buffer (non-unit stride).
        The destination and index are fully contiguous.  torch.put must
        flatten and read the source through its stride rather than assuming
        a packed memory layout.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4844")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: head_dim=128 — step-2 slice of 16-element buffer gives 8 values
        dst = torch.zeros(128, dtype=torch.bfloat16)
        src_base = torch.randn(16, dtype=torch.bfloat16)
        src = src_base[::2]  # shape (8,), stride 2 — non-contiguous
        assert not src.is_contiguous()

        if accumulate:
            idx = torch.randint(0, 128, (8,), dtype=torch.int64)
        else:
            idx = torch.randperm(128, dtype=torch.int64)[:8]

        def fn(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    def test_put_noncontig_index(self, mode: str, accumulate: bool):
        """PUT-GAP-B (ii): Non-contiguous index tensor.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256) — 1D flat write into 16×256 cache.

        Index is a step-2 slice (non-unit stride); destination and source are
        contiguous.  torch.put flattens the index through its stride to resolve
        the target flat positions.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4844")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: head_dim=256 — flat buffer; step-2 index slice addresses 8 positions
        dst = torch.zeros(256, dtype=torch.bfloat16)
        src = torch.randn(8, dtype=torch.bfloat16)
        if accumulate:
            idx_base = torch.randint(0, 256, (16,), dtype=torch.int64)
        else:
            idx_base = torch.cat(
                [
                    torch.randperm(256, dtype=torch.int64)[:8],
                    torch.zeros(8, dtype=torch.int64),  # padding (not used)
                ]
            )
        idx = idx_base[::2]  # shape (8,), stride 2 — non-contiguous
        assert not idx.is_contiguous()

        def fn(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    def test_put_all_three_noncontig(self, mode: str, accumulate: bool):
        """PUT-GAP-B (iii): dst, src, and index all non-contiguous simultaneously.
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128) — 2D kv cache, all non-contig.

        This is the hardest non-contiguous path: torch.put must convert the
        destination to a flattened contiguous view and read both src and index
        through their respective non-unit strides.  Corresponds directly to the
        upstream parametrize over (dst_contig, src_contig, idx_contig) = (F,F,F).
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4636")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: kv_heads=10, head_dim=128 — step-2 rows of (20, 128) → shape (10, 128)
        base_dst = torch.zeros(20, 128, dtype=torch.bfloat16)
        dst = base_dst[::2, :]  # shape (10, 128), stride[0]=256

        # Non-contig src: step-2 slice → shape (8,)
        src_base = torch.randn(16, dtype=torch.bfloat16)
        src = src_base[::2]

        # Non-contig idx: stride-2 slice → shape (8,)
        if accumulate:
            idx_raw = torch.randint(0, dst.numel(), (16,), dtype=torch.int64)
        else:
            idx_raw = torch.cat(
                [
                    torch.randperm(dst.numel(), dtype=torch.int64)[:8],
                    torch.zeros(8, dtype=torch.int64),
                ]
            )
        idx = idx_raw[::2]

        assert not dst.is_contiguous()
        assert not src.is_contiguous()
        assert not idx.is_contiguous()

        def fn(d, i, s):
            return torch.put(d.clone(), i, s, accumulate=accumulate)

        compare_with_cpu(
            fn,
            dst,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1.6e-2,
        )

    # -------------------------------------------------------------------------
    # PUT-GAP-C: all 8 scalar / one-element shape combinations
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    @pytest.mark.parametrize(
        "size_t,size_i,size_s",
        [
            # All 8 combinations of ()/(1,) for dst × idx × src
            ((), (), ()),
            ((), (), (1,)),
            ((), (1,), ()),
            ((), (1,), (1,)),
            ((1,), (), ()),
            ((1,), (), (1,)),
            ((1,), (1,), ()),
            ((1,), (1,), (1,)),
        ],
    )
    def test_put_scalar_size_combinations(
        self,
        mode: str,
        accumulate: bool,
        size_t: tuple,
        size_i: tuple,
        size_s: tuple,
    ):
        """PUT-GAP-C: All 8 combinations of scalar/one-element shapes for dst × idx × src.

        Mirrors the upstream PyTorch scalar-size loop in test_put:
            for size_t, size_i, size_s in product([(), (1,)], repeat=3)

        For accumulate=True the result must equal (dst_init + source).item().
        For accumulate=False the result must equal source.item().
        Both the out-of-place (torch.put) and in-place (put_) variants are
        checked.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # eager: all combinations fail — torch.put not yet supported (#4540)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")

        # compile: failures depend on the index shape (size_i):
        #   size_i=()  → both accumulate modes fail (#4451)
        #   size_i=(1,) → only accumulate=True fails (#4932); accumulate=False passes
        if mode == "compile":
            if size_i == ():
                pytest.xfail(reason="known issue- 4451")
            elif size_i == (1,) and accumulate:
                pytest.xfail(reason="known issue- 4932")

        dst = torch.randn(size_t, dtype=torch.float32)
        # Index high=1: the only valid flat index for a scalar/1-element dst is 0
        idx = torch.zeros(size_i, dtype=torch.int64)
        src = torch.randn(size_s, dtype=torch.float32)

        def fn_outplace(d, i, s):
            return torch.put(d, i, s, accumulate=accumulate)

        def fn_inplace(d, i, s):
            out = d.clone()
            out.put_(i, s, accumulate=accumulate)
            return out

        compare_with_cpu(
            fn_outplace, dst, idx, src, run_eager=run_eager, run_compile=run_compile
        )
        compare_with_cpu(
            fn_inplace, dst, idx, src, run_eager=run_eager, run_compile=run_compile
        )

    # -------------------------------------------------------------------------
    # PUT-GAP-D: empty destination shapes
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("accumulate", [False, True])
    @pytest.mark.parametrize(
        "dst_shape,idx_shape",
        [
            # Empty destination shapes from the upstream test_put_empty suite
            ((0,), (0,)),
            ((0,), (0, 1, 2, 0)),
            ((0, 1, 2, 0), (0,)),
            ((0, 1, 2, 0), (0, 1, 2, 0)),
            ((1, 2, 3), (0,)),
            ((1, 2, 3), (0, 1, 2, 0)),
        ],
    )
    def test_put_empty_dst_shapes(
        self,
        mode: str,
        accumulate: bool,
        dst_shape: tuple,
        idx_shape: tuple,
    ):
        """PUT-GAP-D: Empty destination shapes — put_ must leave dst bit-for-bit unchanged.

        Mirrors the upstream parametrization in test_put_empty:
            dst_shape in [(0,), (0,1,2,0), (1,2,3)]
            indices_shape in [(0,), (0,1,2,0)]

        When the index tensor is empty there are no writes; both accumulate modes
        must return a tensor that equals the original destination.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # eager: all combinations fail — torch.put_ not yet supported (#4540)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")

        # compile: failure depends on dst_shape:
        #   (0,) dest          → #4931 (both accumulate modes)
        #   (0,1,2,0) dest     → passes (both accumulate modes)
        #   (1,2,3)/(0,) idx   → #4636 (both accumulate modes)
        #   (1,2,3)/(0,1,2,0) → #4636 (accumulate=False), #4451 (accumulate=True)
        if mode == "compile":
            if dst_shape == (0,):
                pytest.xfail(reason="known issue- 4931")
            elif dst_shape == (1, 2, 3) and idx_shape == (0,):
                pytest.xfail(reason="known issue- 4636")
            elif (
                dst_shape == (1, 2, 3) and idx_shape == (0, 1, 2, 0) and not accumulate
            ):
                pytest.xfail(reason="known issue- 4636")
            elif dst_shape == (1, 2, 3) and idx_shape == (0, 1, 2, 0) and accumulate:
                pytest.xfail(reason="known issue- 4451")

        dst = torch.randn(*dst_shape, dtype=torch.float32)
        idx = torch.empty(*idx_shape, dtype=torch.int64)
        src = torch.randn(*idx_shape, dtype=torch.float32)

        def fn(d, i, s):
            out = d.clone()
            out.put_(i, s, accumulate=accumulate)
            return out

        compare_with_cpu(
            fn, dst, idx, src, run_eager=run_eager, run_compile=run_compile
        )

    # -------------------------------------------------------------------------
    # PUT-GAP-E: large parallel accumulation (grainsize > 3000)
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_put_large_parallel_accumulate(self, mode: str):
        """PUT-GAP-E: Large parallel accumulation — all updates target slot 0, size > 3000.

        The upstream test_put_accumulate deliberately uses sizes of (200,) and
        (3002,) to trigger CPU parallelism (grainsize = 3000).  The existing
        TestPutOp.test_put_allzero_index_hotspot only uses 100/200 updates —
        below the parallel threshold.

        This test uses 3002 updates, all targeting flat index 0, so the result
        must equal orig[0] + source.sum() within fp32 tolerance. float64 is
        not supported on Spyre.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4930")
        run_eager, run_compile = _mode_flags(mode)

        size = 3002
        orig = torch.randn(1, dtype=torch.float32)
        idx = torch.zeros(size, dtype=torch.int64)
        src = torch.randn(size, dtype=torch.float32)

        def fn(d, i, s):
            return d.put(i, s, accumulate=True)

        compare_with_cpu(
            fn,
            orig,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-2,
            rtol=1e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_put_large_parallel_accumulate_low_precision(
        self, mode: str, dtype: torch.dtype
    ):
        """PUT-GAP-E (low-precision): Large accumulation for float16 and bfloat16.

        Uses 100 updates (matching the upstream reduced count for low-precision
        types to avoid overflow), but exercises the same all-indices-to-slot-0
        pattern.  Wider tolerance accounts for half-precision rounding.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4540")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4540, 4930")
        run_eager, run_compile = _mode_flags(mode)

        size = 100
        orig = torch.randn(1, dtype=dtype)
        idx = torch.zeros(size, dtype=torch.int64)
        src = torch.randn(size, dtype=dtype)

        def fn(d, i, s):
            return d.put(i, s, accumulate=True)

        compare_with_cpu(
            fn,
            orig,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=0.5,
            rtol=0.1,
        )


# =============================================================================
# TORCH.INDEX_REDUCE / INDEX_REDUCE_
# =============================================================================
class TestIndexReduceOp:
    """Op-level tests for torch.index_reduce / Tensor.index_reduce_."""

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_index_reduce_all_modes_dims(self, mode: str, reduce_mode: str, dim: int):
        """IDR-01: All reduce modes × all dims — correctness against Python reference loop.
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128) — 3D tensor [10, 8, 16]."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4634, 4411, 4874")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: kv_heads=10, split head_dim=128 as (8, 16) for 3D coverage
        shape = [10, 8, 16]
        dest = torch.rand(*shape, dtype=torch.bfloat16) + 0.5  # avoid zeros for prod

        def fn(d, i, s):
            return d.index_reduce(dim, i, s, reduce=reduce_mode, include_self=True)

        # Unique indices — no collisions
        idx_unique = torch.tensor([0, 1, 2], dtype=torch.int64)
        src_shape_unique = list(shape)
        src_shape_unique[dim] = len(idx_unique)
        src_unique = torch.rand(*src_shape_unique, dtype=torch.bfloat16) + 0.5
        compare_with_cpu(
            fn,
            dest,
            idx_unique,
            src_unique,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

        # Duplicate indices — slot 0 receives two updates
        idx_dup = torch.tensor([0, 0, 1], dtype=torch.int64)
        src_shape_dup = list(shape)
        src_shape_dup[dim] = len(idx_dup)
        src_dup = torch.rand(*src_shape_dup, dtype=torch.bfloat16) + 0.5
        compare_with_cpu(
            fn,
            dest,
            idx_dup,
            src_dup,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    def test_index_reduce_include_self_false_identity_init(
        self, mode: str, reduce_mode: str
    ):
        """IDR-02: include_self=False — unvisited slots reset to reduction identity.
        Model shape: Granite-4.1-20B (kv_heads=8, head_dim=128).

        prod → 1.0, amax → -inf, amin → +inf, mean → 0.0 (count=0 → stays 0).
        Visited slots computed from src only (no self contribution).
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4634, 4472")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: kv_heads=8 rows of head_dim=128 — rows 4–7 unvisited
        dest = torch.full((8, 128), 5.0, dtype=torch.bfloat16)  # non-identity fill
        src = torch.rand(4, 128, dtype=torch.bfloat16) + 0.5
        idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)  # rows 4–7 unvisited

        def fn(d, i, s):
            return d.index_reduce(0, i, s, reduce=reduce_mode, include_self=False)

        compare_with_cpu(
            fn,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=1e-4,
            rtol=1e-4,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    def test_index_reduce_4d_both(self, mode: str, reduce_mode: str):
        """4D index_reduce on (batch, kv_heads, seq, head_dim) — both OOP and IP.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256). Covers 4D-OOP and 4D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4411")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: (batch=2, kv_heads=16, seq=8, head_dim=256) — reduce along seq (dim=2)
        dest = torch.rand(2, 16, 8, 256, dtype=torch.float32) + 0.5
        src = torch.rand(2, 16, 4, 256, dtype=torch.float32) + 0.5
        idx = torch.randint(0, 8, (4,), dtype=torch.int64)

        def fn_oop(d, i, s):
            return d.index_reduce(2, i, s, reduce=reduce_mode, include_self=True)

        def fn_ip(d, i, s):
            out = d.clone()
            out.index_reduce_(2, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn_oop,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    def test_index_reduce_5d_both(self, mode: str, reduce_mode: str):
        """5D index_reduce on (batch, layers, kv_heads, seq, head_dim) — both OOP and IP.
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128). Covers 5D-OOP and 5D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4411")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: (batch=2, layers=4, kv_heads=10, seq=8, head_dim=128) — reduce along seq (dim=3)
        dest = torch.rand(2, 4, 10, 8, 128, dtype=torch.float32) + 0.5
        src = torch.rand(2, 4, 10, 4, 128, dtype=torch.float32) + 0.5
        idx = torch.randint(0, 8, (4,), dtype=torch.int64)

        def fn_oop(d, i, s):
            return d.index_reduce(3, i, s, reduce=reduce_mode, include_self=True)

        def fn_ip(d, i, s):
            out = d.clone()
            out.index_reduce_(3, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn_oop,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    def test_index_reduce_6d_both(self, mode: str, reduce_mode: str):
        """6D index_reduce on expert-layer-KV-cache geometry — both OOP and IP.
        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128). Covers 6D-OOP and 6D-IP."""
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4411")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: (experts=2, batch=2, layers=2, kv_heads=8, seq=4, head_dim=128)
        dest = torch.rand(2, 2, 2, 8, 4, 128, dtype=torch.float32) + 0.5
        src = torch.rand(2, 2, 2, 8, 2, 128, dtype=torch.float32) + 0.5
        idx = torch.randint(0, 4, (2,), dtype=torch.int64)

        def fn_oop(d, i, s):
            return d.index_reduce(4, i, s, reduce=reduce_mode, include_self=True)

        def fn_ip(d, i, s):
            out = d.clone()
            out.index_reduce_(4, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn_oop,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )
        compare_with_cpu(
            fn_ip,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )


# =============================================================================
# TORCH.MASKED_SCATTER / MASKED_SCATTER_
# =============================================================================
class TestMaskedScatterOpAdditional:
    """Gap scenarios for torch.masked_scatter / masked_scatter_ not covered in TestMaskedScatterOp.

    Covers:
      MSK-NEW-01  Bool dtype source/dest
      MSK-NEW-02  Memory-overlap raises RuntimeError
      MSK-NEW-03  All 4 contig/non-contig dest × mask combos
      MSK-NEW-04  Multi-shape sweep
      MSK-NEW-05  0-d self tensor
      MSK-NEW-06  No side-effects on source/mask
      MSK-NEW-07  All-False mask no-op
      MSK-GAP-A   source.numel() < mask.sum() → RuntimeError
      MSK-GAP-C   Zero-volume / empty-dimension shapes
    """

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_bool_dtype(self, mode: str):
        """Bool dtype source/dest — True/False values scattered at True mask positions."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        # In-place: dst=[F,F,F], src=[T,T,T], mask=[F,T,F] → dst[1]=True
        dst_ip = torch.tensor([False, False, False])
        src_t = torch.tensor([True, True, True])
        mask1 = torch.tensor([False, True, False])

        def fn_inplace(d, m, s):
            out = d.clone()
            out.masked_scatter_(m, s)
            return out

        compare_with_cpu(
            fn_inplace,
            dst_ip,
            mask1,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
        )

        # Out-of-place: mask=[T,F,T] → result=[T,T,T]
        dst_oop = torch.tensor([False, True, False])
        mask2 = torch.tensor([True, False, True])

        def fn_outplace(d, m, s):
            return d.masked_scatter(m, s)

        compare_with_cpu(
            fn_outplace,
            dst_oop,
            mask2,
            src_t,
            run_eager=run_eager,
            run_compile=run_compile,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_memory_overlap_raises(self, mode: str):
        """Expanded (non-owning) dst must raise RuntimeError before writing."""
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        x = torch.rand(1, dtype=torch.float32).expand(6)  # non-owning view
        src = torch.rand(3, dtype=torch.float32)
        mask = torch.tensor([True, False, True, True, False, False])

        # Test that writing into the non-owning expanded view raises
        if run_eager:
            with pytest.raises(RuntimeError):
                x.masked_scatter_(mask, src)

        if run_compile and hasattr(torch, "compile"):
            with pytest.raises(RuntimeError):
                torch.compile(lambda d, m, s: d.masked_scatter_(m, s))(x, mask, src)

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_noncontig_dest_mask_combos(self, mode: str):
        """All 4 contig/non-contig combos of dest × mask produce identical results."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        # Build reference tensors then derive non-contig views
        t_base = torch.zeros(5, 2, dtype=torch.float32)
        t_contig = t_base.clone()
        t_ncontig = t_base.clone().transpose(0, 1)  # shape (2,5), non-contig

        mask_base = torch.tensor(
            [[False, True], [False, True], [False, False], [True, True], [True, True]]
        )
        mask_contig = mask_base.clone()
        mask_ncontig = mask_base.clone().transpose(0, 1)  # shape (2,5), non-contig

        source = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)

        def fn_cc(d, m, s):
            # contiguous dest, contiguous mask
            out = d.contiguous().clone()
            out.masked_scatter_(m.contiguous(), s)
            return out

        def fn_cn(d, m, s):
            # contiguous dest, non-contiguous mask
            out = d.contiguous().clone()
            out.masked_scatter_(m, s)
            return out

        def fn_nc(d, m, s):
            # non-contiguous dest, contiguous mask
            out = d.clone()
            out.masked_scatter_(m.contiguous(), s)
            return out

        def fn_nn(d, m, s):
            # non-contiguous dest, non-contiguous mask
            out = d.clone()
            out.masked_scatter_(m, s)
            return out

        # All 4 combos: CC, CN, NC, NN
        compare_with_cpu(
            fn_cc,
            t_contig,
            mask_contig,
            source,
            run_eager=run_eager,
            run_compile=run_compile,
        )
        compare_with_cpu(
            fn_cn,
            t_contig,
            mask_ncontig,
            source,
            run_eager=run_eager,
            run_compile=run_compile,
        )
        compare_with_cpu(
            fn_nc,
            t_ncontig,
            mask_contig,
            source,
            run_eager=run_eager,
            run_compile=run_compile,
        )
        compare_with_cpu(
            fn_nn,
            t_ncontig,
            mask_ncontig,
            source,
            run_eager=run_eager,
            run_compile=run_compile,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("shape", [(2, 5), (5, 10, 3), (4, 5, 4, 3)])
    def test_masked_scatter_multishape_sweep(self, mode: str, shape: tuple):
        """Multi-shape sweep — 2D, 3D, 4D tensors with ~60% True density masks."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        torch.manual_seed(42)
        dest = torch.randn(*shape, dtype=torch.float32)
        mask = torch.rand(*shape) < 0.6
        num_true = int(mask.sum().item())
        src = torch.randn(num_true, dtype=torch.float32)

        def fn(d, m, s):
            return d.masked_scatter(m, s)

        compare_with_cpu(
            fn, dest, mask, src, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_scalar_0d_self(self, mode: str):
        """MSK-NEW-04b: 0-dimensional scalar self with scalar True mask → value replaced, shape preserved."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        x = torch.tensor(0, dtype=torch.float32)
        mask = torch.tensor(True)
        src = torch.tensor([42.0], dtype=torch.float32)

        def fn(d, m, s):
            return d.masked_scatter(m, s)

        compare_with_cpu(fn, x, mask, src, run_eager=run_eager, run_compile=run_compile)

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_no_side_effects(self, mode: str):
        """Out-of-place call must not modify self, mask, or source."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        torch.manual_seed(3)
        x = torch.zeros(5, dtype=torch.float32)
        mask = torch.tensor([True, False, True, False, True])
        src = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)

        x_clone = x.clone()
        mask_clone = mask.clone()
        src_clone = src.clone()

        def fn(d, m, s):
            return d.masked_scatter(m, s)

        compare_with_cpu(fn, x, mask, src, run_eager=run_eager, run_compile=run_compile)

        # Originals must be untouched on CPU (out-of-place op must not modify inputs)
        torch.testing.assert_close(x, x_clone)
        torch.testing.assert_close(src, src_clone)
        assert torch.equal(mask, mask_clone)

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    def test_masked_scatter_scalar_false_mask_noop(self, mode: str):
        """0-dimensional scalar False mask — dest returned unchanged (no True positions)."""
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        mask = torch.tensor(0, dtype=torch.bool)  # scalar False
        src = torch.tensor([99.0, 88.0], dtype=torch.float32)

        def fn(d, m, s):
            return d.masked_scatter(m, s)

        compare_with_cpu(fn, x, mask, src, run_eager=run_eager, run_compile=run_compile)

    # -------------------------------------------------------------------------
    # MSK-GAP-A: source smaller than the number of True values → RuntimeError
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("variant", ["inplace", "outplace"])
    def test_masked_scatter_source_too_small_raises(self, mode: str, variant: str):
        """MSK-GAP-A: source.numel() < mask.sum() must raise RuntimeError.

        The upstream PyTorch test_masked_scatter explicitly checks (CPU only,
        to avoid CUDA kernel-synchronisation issues) that when the source tensor
        has fewer elements than the number of True positions in the mask a
        RuntimeError is raised with the message "source < number of ones in mask".

        This test covers:
        - in-place  (masked_scatter_)
        - out-of-place (masked_scatter)
        - both eager and compile modes

        compare_with_cpu is not applicable — the test asserts an error is raised,
        not a value.  In compile mode the error must propagate out of the compiled
        function.
        """
        run_eager, run_compile = _mode_flags(mode)

        dest = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)
        # mask has 3 True elements; source only has 2 — one short
        mask = torch.tensor(
            [False, False, False, False, True, False, True, False, True, False]
        )
        src_short = torch.tensor([0.0, 0.0], dtype=torch.float32)  # numel=2 < 3 True

        if run_eager:
            if variant == "inplace":
                with pytest.raises(RuntimeError):
                    dest.clone().masked_scatter_(mask, src_short)
            else:
                with pytest.raises(RuntimeError):
                    torch.masked_scatter(dest, mask, src_short)

        if run_compile and hasattr(torch, "compile"):
            if variant == "inplace":
                compiled_fn = torch.compile(
                    lambda d, m, s: d.clone().masked_scatter_(m, s)
                )
            else:
                compiled_fn = torch.compile(
                    lambda d, m, s: torch.masked_scatter(d, m, s)
                )
            with pytest.raises(RuntimeError):
                compiled_fn(dest, mask, src_short)

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize(
        "n_true,src_size",
        [
            (3, 0),  # 3 True positions, 0-element source
            (3, 1),  # 3 True positions, 1-element source
            (3, 2),  # 3 True positions, 2-element source (one short)
            (5, 4),  # 5 True positions, 4-element source
        ],
    )
    def test_masked_scatter_source_too_small_parametrized(
        self, mode: str, n_true: int, src_size: int
    ):
        """MSK-GAP-A (parametrized): RuntimeError for various (n_true, src_size) pairs
        where src_size < n_true.

        Covers the boundary: 0-element source, 1-element source, and near-boundary
        cases.  All combinations must raise RuntimeError on both in-place and
        out-of-place variants in both eager and compile modes.
        compare_with_cpu is not applicable — error assertion only.
        """
        run_eager, run_compile = _mode_flags(mode)

        # Build a dest and mask with exactly n_true True positions
        dest = torch.zeros(10, dtype=torch.float32)
        mask_vals = [False] * 10
        for i in range(n_true):
            mask_vals[i] = True
        mask = torch.tensor(mask_vals)
        src_short = torch.zeros(src_size, dtype=torch.float32)

        for fn_inplace in [True, False]:
            if run_eager:
                if fn_inplace:
                    with pytest.raises(RuntimeError):
                        dest.clone().masked_scatter_(mask, src_short)
                else:
                    with pytest.raises(RuntimeError):
                        torch.masked_scatter(dest, mask, src_short)

            if run_compile and hasattr(torch, "compile"):
                if fn_inplace:
                    cfn = torch.compile(lambda d, m, s: d.clone().masked_scatter_(m, s))
                else:
                    cfn = torch.compile(lambda d, m, s: torch.masked_scatter(d, m, s))
                with pytest.raises(RuntimeError):
                    cfn(dest, mask, src_short)

    # -------------------------------------------------------------------------
    # MSK-GAP-C: zero-volume / empty-dimension tensor shapes
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize(
        "dest_shape,mask_shape",
        [
            # (0,)  — 1D zero-length
            ((0,), (0,)),
            # (N, 0) — zero columns
            ((4, 0), (4, 0)),
            # (0, N) — zero rows
            ((0, 5), (0, 5)),
            # (N, 0, M) — zero-volume middle dim, matching upstream (5, 0, 5)
            ((5, 0, 5), (5, 0, 5)),
            # (N, 0, M) with broadcast-compatible mask of same shape
            ((3, 0, 4), (3, 0, 4)),
        ],
    )
    def test_masked_scatter_zero_volume_shapes(
        self, mode: str, dest_shape: tuple, mask_shape: tuple
    ):
        """MSK-GAP-C: Zero-volume / empty-dimension destinations — must be a safe no-op.

        When the destination (or mask) contains a zero-size dimension the total
        number of elements is zero.  An all-False (or vacuously all-True) mask
        over zero elements has zero True positions, so an empty source is valid
        and the operation must return a tensor identical to the original
        destination.

        Mirrors the upstream empty-tensor cases in test_masked_scatter:
            dest = torch.empty((5, 0, 5), ...)
            mask = torch.ones_like(dest, dtype=torch.bool)
            src  = torch.empty((0,), ...)
            dest.masked_scatter_(mask, src)   # no-op, dest unchanged

        Both in-place and out-of-place variants are checked.
        compare_with_cpu is used: CPU result (unchanged dest) is the reference.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        dest = torch.randn(*dest_shape, dtype=torch.float32)
        # All-False mask — zero True positions, empty source is always valid
        mask = torch.zeros(*mask_shape, dtype=torch.bool)
        src = torch.empty(0, dtype=torch.float32)

        def fn_inplace(d, m, s):
            out = d.clone()
            out.masked_scatter_(m, s)
            return out

        def fn_outplace(d, m, s):
            return torch.masked_scatter(d, m, s)

        compare_with_cpu(
            fn_inplace, dest, mask, src, run_eager=run_eager, run_compile=run_compile
        )
        compare_with_cpu(
            fn_outplace, dest, mask, src, run_eager=run_eager, run_compile=run_compile
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize(
        "dest_shape",
        [
            (5, 0, 5),
            (3, 0, 4),
            (0, 8),
        ],
    )
    def test_masked_scatter_zero_volume_ones_mask(self, mode: str, dest_shape: tuple):
        """MSK-GAP-C (ones mask): Zero-volume dest with all-True mask and empty source.

        Mirrors the second upstream empty-tensor variant:
            mask = torch.ones_like(dest, dtype=torch.bool)
            src  = torch.empty((0,), ...)
            dest.masked_scatter_(mask, src)

        An all-True mask over a zero-volume tensor still has zero True elements
        (because numel() == 0), so the empty source is valid and the call must
        be a safe no-op.
        compare_with_cpu validates that both eager and compile modes match CPU.
        """
        if mode == "eager":
            pytest.xfail(reason="known issue- 4437")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4437")
        run_eager, run_compile = _mode_flags(mode)

        dest = torch.empty(*dest_shape, dtype=torch.float32)
        mask = torch.ones(*dest_shape, dtype=torch.bool)  # all-True, but 0 elements
        src = torch.empty(0, dtype=torch.float32)

        def fn_inplace(d, m, s):
            out = d.clone()
            out.masked_scatter_(m, s)
            return out

        def fn_outplace(d, m, s):
            return torch.masked_scatter(d, m, s)

        compare_with_cpu(
            fn_inplace, dest, mask, src, run_eager=run_eager, run_compile=run_compile
        )
        compare_with_cpu(
            fn_outplace, dest, mask, src, run_eager=run_eager, run_compile=run_compile
        )


# =============================================================================
# TORCH.INDEX_REDUCE / INDEX_REDUCE_
# =============================================================================


class TestIndexReduceUpstreamGaps:
    """Upstream-gap scenarios for torch.index_reduce / Tensor.index_reduce_.

    IDR-GAP-A  index_reduce_ (in-place) is not tested — only out-of-place used so far
    IDR-GAP-B  non-contiguous dest / src / index — all-contig only so far
    IDR-GAP-C  int32 index dtype — only int64 used so far
    IDR-GAP-D  float16 / bfloat16 dtype coverage — only float32 so far
    IDR-GAP-E  include_self=False × dim 1 and dim 2 — only dim 0 tested
    IDR-GAP-F  non-contiguous view / sliced destination (regression: #144846)
    IDR-GAP-G  include_self=False identity init expanded to all dims + index_reduce_

    Every test is parametrized over ["eager", "compile"] and uses compare_with_cpu
    as the sole assertion mechanism.
    """

    # -------------------------------------------------------------------------
    # IDR-GAP-A: in-place index_reduce_ dedicated tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_index_reduce_inplace_all_modes_dims(
        self, mode: str, reduce_mode: str, dim: int
    ):
        """IDR-GAP-A: index_reduce_ (in-place) correctness — all 4 modes × dims 0/1/2.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128) — 3D tensor [8, 8, 16].

        The existing TestIndexReduceOp.test_index_reduce_all_modes_dims calls the
        out-of-place d.index_reduce(...).  This test uses the in-place form
        out.index_reduce_(dim, idx, src, reduce=..., include_self=True) wrapped
        in a clone so compare_with_cpu can compare the modified tensor.

        include_self=True: dest values contribute to the reduction.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4634, 4932")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: kv_heads=8, head_dim=128 split as (8, 8, 16) for 3D coverage
        shape = [8, 8, 16]
        num_src = 4
        src_shape = list(shape)
        src_shape[dim] = num_src

        dest = torch.rand(*shape, dtype=torch.bfloat16) + 0.5  # avoid zeros for prod
        src = torch.rand(*src_shape, dtype=torch.bfloat16) + 0.5
        idx = torch.randint(0, shape[dim], (num_src,), dtype=torch.int64)

        def fn(d, i, s):
            out = d.clone()
            out.index_reduce_(dim, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_index_reduce_inplace_include_self_false(
        self, mode: str, reduce_mode: str, dim: int
    ):
        """IDR-GAP-A + IDR-GAP-E + IDR-GAP-G: index_reduce_ with include_self=False,
        all modes, all dims.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128) — 3D tensor [8, 8, 16].

        Combines three gaps in one parametrized test:
        - Uses the in-place API (IDR-GAP-A)
        - Covers dim 1 and dim 2 (IDR-GAP-E)
        - Expands the identity-init check to all dims (IDR-GAP-G)

        Unvisited slots are reset to the reduction identity before accumulation:
          prod → 1.0,  amax → -inf,  amin → +inf,  mean → 0.0
        The CPU reference from compare_with_cpu verifies both visited and
        unvisited slot values.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4634, 4932")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: kv_heads=8, head_dim=128 split as (8, 8, 16)
        shape = [8, 8, 16]
        num_src = 2  # deliberately fewer than shape[dim]
        src_shape = list(shape)
        src_shape[dim] = num_src

        dest = torch.full(shape, 5.0, dtype=torch.bfloat16)  # non-identity fill
        src = torch.rand(*src_shape, dtype=torch.bfloat16) + 0.5
        idx = torch.arange(num_src, dtype=torch.int64)  # touches first num_src slots

        def fn(d, i, s):
            out = d.clone()
            out.index_reduce_(dim, i, s, reduce=reduce_mode, include_self=False)
            return out

        compare_with_cpu(
            fn,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    # -------------------------------------------------------------------------
    # IDR-GAP-B: non-contiguous destination / source / index
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    def test_index_reduce_noncontig_dest(self, mode: str, reduce_mode: str):
        """IDR-GAP-B (i): Non-contiguous destination — step-2 row stride.
        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128) — step-2 row view.

        dest is a [::2, :] view of a larger backing buffer (stride[0] = 2×cols),
        so it is non-contiguous.  src and idx are contiguous.
        index_reduce_ must correctly address the strided destination layout.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4634, 4932")
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: kv_heads=8, head_dim=128 — step-2 view of 16-row buffer
        base = torch.rand(16, 128, dtype=torch.bfloat16) + 0.5
        dest = base[::2, :]  # shape (8, 128), non-contiguous
        assert not dest.is_contiguous()
        src = torch.rand(4, 128, dtype=torch.bfloat16) + 0.5
        idx = torch.randint(0, dest.shape[0], (4,), dtype=torch.int64)

        def fn(d, i, s):
            out = d.clone()
            out.index_reduce_(0, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    def test_index_reduce_noncontig_source(self, mode: str, reduce_mode: str):
        """IDR-GAP-B (ii): Non-contiguous source — transposed 2D tensor.
        Model shape: Granite-4.1-20B (kv_heads=8, head_dim=128) — transposed (128, 8) → (8, 128).

        src is a transposed view (non-contiguous strides).  dest and idx are
        contiguous.  index_reduce_ must read src through its logical strides.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4634, 4411")
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: kv_heads=8, head_dim=128 — transpose (128, 8) → (8, 128)
        dest = torch.rand(8, 128, dtype=torch.bfloat16) + 0.5
        src_base = torch.rand(128, 4, dtype=torch.bfloat16) + 0.5
        src = src_base.t()  # shape (4, 128), non-contiguous
        assert not src.is_contiguous()

        def fn(d, i, s):
            out = d.clone()
            out.index_reduce_(0, i, s, reduce=reduce_mode, include_self=True)
            return out

        # Unique indices — no collisions
        idx_unique = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
        compare_with_cpu(
            fn,
            dest,
            idx_unique,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

        # Duplicate indices — row 0 receives two updates
        idx_dup = torch.tensor([0, 0, 2, 4], dtype=torch.int64)
        compare_with_cpu(
            fn,
            dest,
            idx_dup,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize("mode", ["eager", "compile"])
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    def test_index_reduce_noncontig_index(self, mode: str, reduce_mode: str):
        """IDR-GAP-B (iii): Non-contiguous index — step-2 stride slice.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256) — step-2 index over kv-heads.

        idx is derived from a wider buffer via [::2], giving non-unit stride.
        dest and src are contiguous.
        """
        torch.manual_seed(0)
        if mode == "eager":
            pytest.xfail(reason="known issue- 4634")
        if mode == "compile":
            pytest.xfail(reason="known issue- 4634, 4932")
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: kv_heads=16 rows, head_dim=256 — step-2 stride selects 8 indices
        dest = torch.rand(16, 256, dtype=torch.bfloat16) + 0.5
        src = torch.rand(4, 256, dtype=torch.bfloat16) + 0.5

        def fn(d, i, s):
            out = d.clone()
            out.index_reduce_(0, i, s, reduce=reduce_mode, include_self=True)
            return out

        # Unique non-contiguous index — no collisions: selects [0, 2, 4, 6] kv-heads
        idx_base_unique = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
        idx_unique = idx_base_unique[::2]  # [0, 2, 4, 6], stride 2 — non-contiguous
        assert not idx_unique.is_contiguous()
        compare_with_cpu(
            fn,
            dest,
            idx_unique,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

        # Duplicate non-contiguous index — kv-head 0 receives two updates
        idx_base_dup = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], dtype=torch.int64)
        idx_dup = idx_base_dup[::2]  # [0, 0, 2, 2], stride 2 — non-contiguous
        assert not idx_dup.is_contiguous()
        compare_with_cpu(
            fn,
            dest,
            idx_dup,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(reason="Known issue #4634"),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(reason="Known issue #4634/#4411"),
            ),
        ],
    )
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    def test_index_reduce_all_three_noncontig(self, mode: str, reduce_mode: str):
        """IDR-GAP-B (iv): dest, src, and index all non-contiguous simultaneously.
        Model shape: Gemma-4-12B (kv_heads=10, head_dim=128) — all-noncontig 2D kv cache.

        Corresponds to the upstream (dest_noncontig, src_noncontig, index_noncontig)
        = (True, True, True) parametrize combination.  All three tensors have
        non-unit strides; index_reduce_ must correctly resolve addresses for each.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-12B: kv_heads=10, head_dim=128 — step-2 rows of (20, 128) → (10, 128)
        base_dest = torch.rand(20, 128, dtype=torch.bfloat16) + 0.5
        dest = base_dest[::2, :]  # shape (10, 128), non-contiguous

        # Non-contig src: transposed (128, 4) → (4, 128)
        src_base = torch.rand(128, 4, dtype=torch.bfloat16) + 0.5
        src = src_base.t()  # shape (4, 128), non-contiguous

        assert not dest.is_contiguous()
        assert not src.is_contiguous()

        def fn(d, i, s):
            out = d.clone()
            out.index_reduce_(0, i, s, reduce=reduce_mode, include_self=True)
            return out

        # Unique non-contiguous index — selects [0, 2, 4, 6] kv-heads, no collisions
        idx_base_unique = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
        idx_unique = idx_base_unique[::2]  # [0, 2, 4, 6], stride 2 — non-contiguous
        assert not idx_unique.is_contiguous()
        compare_with_cpu(
            fn,
            dest,
            idx_unique,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

        # Duplicate non-contiguous index — kv-head 0 receives two updates
        idx_base_dup = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], dtype=torch.int64)
        idx_dup = idx_base_dup[::2]  # [0, 0, 2, 2], stride 2 — non-contiguous
        assert not idx_dup.is_contiguous()
        compare_with_cpu(
            fn,
            dest,
            idx_dup,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    # -------------------------------------------------------------------------
    # IDR-GAP-C: int32 index dtype
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(reason="Known issue #4634"),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(reason="Known issue #4634/#4411"),
            ),
        ],
    )
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    @pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
    def test_index_reduce_index_dtype(
        self, mode: str, reduce_mode: str, index_dtype: torch.dtype
    ):
        """IDR-GAP-C: int32 and int64 index dtypes — both must produce identical results.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128).

        The upstream suite parametrizes over index_dtypes = [torch.int, torch.long].
        Existing tests use only int64.  This test covers both dtypes for both the
        out-of-place and in-place APIs across all four reduce modes.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Granite-3.3-8B: kv_heads=8 rows, head_dim=128 cols — int32/int64 index into kv-heads
        dest = torch.rand(8, 128, dtype=torch.bfloat16) + 0.5
        src = torch.rand(4, 128, dtype=torch.bfloat16) + 0.5

        def fn_out(d, i, s):
            return d.index_reduce(0, i, s, reduce=reduce_mode, include_self=True)

        def fn_in(d, i, s):
            out = d.clone()
            out.index_reduce_(0, i, s, reduce=reduce_mode, include_self=True)
            return out

        for idx in [
            torch.tensor([0, 2, 4, 6], dtype=index_dtype),  # unique kv-heads
            torch.tensor(
                [0, 0, 2, 4], dtype=index_dtype
            ),  # duplicate: kv-head 0 hit twice
        ]:
            compare_with_cpu(
                fn_out,
                dest,
                idx,
                src,
                run_eager=run_eager,
                run_compile=run_compile,
                atol=5e-3,
                rtol=5e-3,
            )
            compare_with_cpu(
                fn_in,
                dest,
                idx,
                src,
                run_eager=run_eager,
                run_compile=run_compile,
                atol=5e-3,
                rtol=5e-3,
            )

    # -------------------------------------------------------------------------
    # IDR-GAP-D: float16 / bfloat16 dtype coverage
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(reason="Known issue #4634"),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(reason="Known issue #4634/#4411"),
            ),
        ],
    )
    @pytest.mark.parametrize("reduce_mode", ["mean", "amax", "amin"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_index_reduce_low_precision_dtypes(
        self, mode: str, reduce_mode: str, dtype: torch.dtype
    ):
        """IDR-GAP-D: float16 and bfloat16 dtype coverage — both out-of-place and in-place.
        Model shape: Granite-3.3-8B (kv_heads=8, head_dim=128).

        The existing main test uses float32 only.  This test covers float16 and
        bfloat16 for the three modes where low-precision is defined (mean, amax,
        amin).  prod is omitted here as repeated multiplication quickly underflows
        to zero in fp16/bf16.

        Tolerance is widened for low-precision: half-precision accumulation
        introduces rounding that can differ from float32 reference.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        if dtype in (torch.float16, torch.bfloat16):
            atol, rtol = 5e-2, 5e-2
        else:
            atol, rtol = 1e-4, 1e-4

        # Granite-3.3-8B: kv_heads=8 rows, head_dim=128 cols — values in [0.5, 1.5] to avoid fp16 overflow
        dest = (torch.rand(8, 128, dtype=torch.float32) + 0.5).to(dtype)
        src = (torch.rand(4, 128, dtype=torch.float32) + 0.5).to(dtype)

        def fn_out(d, i, s):
            return d.index_reduce(0, i, s, reduce=reduce_mode, include_self=True)

        def fn_in(d, i, s):
            out = d.clone()
            out.index_reduce_(0, i, s, reduce=reduce_mode, include_self=True)
            return out

        for idx in [
            torch.tensor([0, 2, 4, 6], dtype=torch.int64),  # unique kv-heads
            torch.tensor(
                [0, 0, 2, 4], dtype=torch.int64
            ),  # duplicate: kv-head 0 hit twice
        ]:
            compare_with_cpu(
                fn_out,
                dest,
                idx,
                src,
                run_eager=run_eager,
                run_compile=run_compile,
                atol=atol,
                rtol=rtol,
            )
            compare_with_cpu(
                fn_in,
                dest,
                idx,
                src,
                run_eager=run_eager,
                run_compile=run_compile,
                atol=atol,
                rtol=rtol,
            )

    # -------------------------------------------------------------------------
    # IDR-GAP-E + IDR-GAP-G: include_self=False × all dims, out-of-place +
    #                         identity-init expanded across dims
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(reason="Known issue #4634"),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(reason="Known issue #4634/#4472/#4874"),
            ),
        ],
    )
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_index_reduce_include_self_false_all_dims(
        self, mode: str, reduce_mode: str, dim: int
    ):
        """IDR-GAP-E + IDR-GAP-G: include_self=False out-of-place, all dims.
        Model shape: Ministral-3-14B (kv_heads=8, head_dim=128) — 3D tensor [8, 8, 16].

        The existing test_index_reduce_include_self_false_identity_init only covers
        dim=0.  This test parametrizes over dims 0, 1, 2 to verify that the
        identity-reset and accumulate-from-src-only logic is correct along every
        axis of a 3D tensor.

        Unvisited slots retain the reduction identity (prod→1, amax→-inf,
        amin→+inf, mean→0); visited slots are computed from src only.
        CPU is the authoritative reference via compare_with_cpu.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Ministral-3-14B: kv_heads=8, head_dim=128 split as (8, 8, 16) for 3D coverage
        shape = [8, 8, 16]
        num_src = 2
        src_shape = list(shape)
        src_shape[dim] = num_src

        dest = torch.full(shape, 5.0, dtype=torch.bfloat16)
        src = torch.rand(*src_shape, dtype=torch.bfloat16) + 0.5
        idx = torch.arange(num_src, dtype=torch.int64)

        def fn(d, i, s):
            return d.index_reduce(dim, i, s, reduce=reduce_mode, include_self=False)

        compare_with_cpu(
            fn,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    # -------------------------------------------------------------------------
    # IDR-GAP-F: non-contiguous view / sliced destination
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(reason="Known issue #4634"),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(reason="Known issue #4634/#4874"),
            ),
        ],
    )
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    def test_index_reduce_sliced_view_dest(self, mode: str, reduce_mode: str):
        """IDR-GAP-F: index_reduce_ on a non-contiguous view slice of a larger tensor.
        Model shape: Granite-4.1-20B (kv_heads=8, head_dim=128) — (batch=2, kv_heads+pad=11, head_dim=128).

        Mirrors the upstream TorchInductor regression test (issue #144846):
            x_base[:, 3:, :]  — non-contiguous slice leaving 3 padding cols behind

        The base tensor is allocated with extra padding; index_reduce_ is
        applied directly to the slice (non-zero storage_offset and non-default strides).
        The result in the slice must match the CPU reference; unsliced cols remain unchanged.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Granite-4.1-20B: (batch=2, kv_heads=8 + 3 padding = 11, head_dim=128)
        # slice [:, 3:, :] → shape (2, 8, 128) — kv_heads view with non-zero storage_offset
        x_base = torch.rand(2, 11, 128, dtype=torch.bfloat16) + 0.5
        idx = torch.randint(0, 8, (4,), dtype=torch.int64)
        source = torch.rand(2, 4, 128, dtype=torch.bfloat16) + 0.5

        def fn(base, i, s):
            out = base.clone()
            view = out[
                :, 3:, :
            ]  # shape (2, 8, 128), non-contiguous (storage_offset > 0)
            view.index_reduce_(1, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn,
            x_base,
            idx,
            source,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(
                "eager",
                marks=pytest.mark.xfail(reason="Known issue #4634"),
            ),
            pytest.param(
                "compile",
                marks=pytest.mark.xfail(reason="Known issue #4634/#4411"),
            ),
        ],
    )
    @pytest.mark.parametrize("reduce_mode", ["prod", "mean", "amax", "amin"])
    def test_index_reduce_narrow_view_dest(self, mode: str, reduce_mode: str):
        """IDR-GAP-F (narrow): index_reduce_ on a narrow() view — non-zero storage_offset.
        Model shape: Gemma-4-26B (kv_heads=16, head_dim=256) — narrow 16 of 32 kv-head rows.

        Uses narrow() instead of slicing to produce a view with a different
        storage_offset.  This exercises the pointer-arithmetic path through
        the backend's DMA address computation for non-zero offsets.
        """
        torch.manual_seed(0)
        run_eager, run_compile = _mode_flags(mode)

        # Gemma-4-26B: kv_heads=16, head_dim=256 — pool of 32 rows; narrow to 16 at offset 8
        pool = torch.rand(32, 256, dtype=torch.bfloat16) + 0.5
        dest = pool.narrow(
            0, 8, 16
        )  # rows 8–23, storage_offset = 8*256, shape (16, 256)
        assert dest.storage_offset() > 0
        src = torch.rand(4, 256, dtype=torch.bfloat16) + 0.5
        idx = torch.randint(0, 16, (4,), dtype=torch.int64)

        def fn(d, i, s):
            out = d.clone()
            out.index_reduce_(0, i, s, reduce=reduce_mode, include_self=True)
            return out

        compare_with_cpu(
            fn,
            dest,
            idx,
            src,
            run_eager=run_eager,
            run_compile=run_compile,
            atol=5e-3,
            rtol=5e-3,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
