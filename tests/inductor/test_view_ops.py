# Copyright 2026 The Torch-Spyre Authors.
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

"""
View Ops Tests

Consolidates operation-level tests for the following view ops:
  §2   reshape / view
  §3   squeeze
  §4   unsqueeze
  §5   expand
  §6   transpose
  §7   t (2D transpose)
  §9   clone
  §10  cat
  §11  select
  §12  narrow / slice
  §13  split

Excluded from this file (kept in their own files):
  §8   permute                 → test_view_ops_permute.py
  FVT  module-level / e2e      → test_view_ops_fvt.py

Pass criteria: compare_with_cpu unless noted per subsection.

Stick alignment reference:
  fp16 / bf16  →  64 elements per stick  (128 B / 2 B)
  fp32         →  32 elements per stick  (128 B / 4 B)
"""

import pytest
import torch
import torch.nn.functional as F

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

CONSUMERS = ["pointwise", "sum_last"]
CONSUMERS_SCALAR = ["pointwise"]  # 0D scalar output: sum(dim=-1) is undefined


def _apply_consumer(t, consumer):
    """Apply the named consumer to tensor t."""
    if consumer == "pointwise":
        return t * 1.0
    return t.sum(dim=-1)


_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_F32 = 1e-5
_RTOL = 1e-2


def _get_tolerances(dtype):
    if dtype == torch.float16:
        return _ATOL_F16, _RTOL
    elif dtype == torch.bfloat16:
        return _ATOL_BF16, _RTOL
    elif dtype == torch.float32:
        return _ATOL_F32, _RTOL
    return _ATOL_F16, _RTOL


def _compare_modes(execution_mode, fn, *args, atol=None, rtol=None):
    if atol is None or rtol is None:
        dtype = next(
            (a.dtype for a in args if isinstance(a, torch.Tensor)), torch.float32
        )
        def_atol, def_rtol = _get_tolerances(dtype)
        atol = def_atol if atol is None else atol
        rtol = def_rtol if rtol is None else rtol

    compare_with_cpu(
        fn,
        *args,
        atol=atol,
        rtol=rtol,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


# ===========================================================================
# §2 — reshape / view
# ===========================================================================

# ---------------------------------------------------------------------------
# §2.1 — Gemma CPU-traced shapes (aligned)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestReshapeCPUTraced:
    """Gemma CPU-traced aligned shapes not covered by module-level tests."""

    def test_reshape_qsplit_full_attn(self, execution_mode, dtype):
        """[1,32,8192]→view(1,32,16,512) * 1.0→[1,32,16,512]. Full-attn D=512; no module-level test."""
        x = cached_randn((1, 32, 8192), dtype=dtype,
                         differentiation=f"reshape_qsplit_full_attn_{dtype}")
        _compare_modes(execution_mode, lambda t: t.view(1, 32, 16, 512) * 1.0, x)

    def test_reshape_round_trip_flatten_split(self, execution_mode, dtype):
        """[1,32,16,256]→reshape(1,32,-1).reshape(1,32,16,256) * 1.0. Round-trip chain; no module-level test."""
        x = cached_randn((1, 32, 16, 256), dtype=dtype,
                         differentiation=f"reshape_round_trip_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: t.reshape(1, 32, -1).reshape(1, 32, 16, 256) * 1.0,
            x,
        )


# ---------------------------------------------------------------------------
# §2.2 — Spyre-traced shapes (aligned)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestReshapeSpyreTraced:
    """Spyre-traced shapes not covered by module-level tests."""

    def test_reshape_kvsplit_spyre_s64(self, execution_mode, dtype):
        """[1,64,2048]→view(1,64,-1,256) * 1.0→[1,64,8,256]. K/V split, Spyre seq=64; no module-level test."""
        x = cached_randn((1, 64, 2048), dtype=dtype,
                         differentiation=f"reshape_kvsplit_s64_{dtype}")
        _compare_modes(execution_mode, lambda t: t.view(1, 64, -1, 256) * 1.0, x)

    def test_reshape_inferred_dim_aligned(self, execution_mode, dtype):
        """[2,32,16,256]→reshape(2,-1,256) * 1.0→[2,512,256]. Inner unchanged 256→256; no module-level test."""
        x = cached_randn((2, 32, 16, 256), dtype=dtype,
                         differentiation=f"reshape_inferred_dim_{dtype}")
        _compare_modes(execution_mode, lambda t: t.reshape(2, -1, 256) * 1.0, x)


# ---------------------------------------------------------------------------
# §2.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestReshapeUnaligned:
    """All ranks 1D–6D with non-multiple innermost dimensions."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_1d_to_2d(self, execution_mode, dtype, consumer):
        """[520]→reshape(8,65)→[8,65]. Inner 65: 65%64=1. Both consumers exercised."""
        pytest.xfail(
            "known failure across all dtypes and consumers; "
            "tracked in https://github.com/torch-spyre/torch-spyre/issues/1353"
        )
        x = cached_randn((520,), dtype=dtype, differentiation=f"reshape_1d_to_2d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(8, 65), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_2d_to_1d(self, execution_mode, dtype, consumer):
        """[8,65]→reshape(-1)→[520]. 1D output; sum_last reduces to scalar (valid)."""
        pytest.xfail(
            "known failure across all dtypes and consumers (UnalignedStickSplit); "
            "tracked in https://github.com/torch-spyre/torch-spyre/issues/5197"
        )
        x = cached_randn((8, 65), dtype=dtype, differentiation=f"reshape_2d_to_1d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(-1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_3d_to_2d_33(self, execution_mode, dtype, consumer):
        """[4,8,33]→reshape(32,33)→[32,33]. Inner 33 < 64. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32 and execution_mode == "eager":
            pytest.xfail(
                "known failure for sum_last/float32/eager; "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/1353"
            )
        if consumer == "sum_last" and dtype == torch.float32 and execution_mode == "compiled":
            pytest.xfail(
                "known failure for sum_last/float32/compiled; "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((4, 8, 33), dtype=dtype, differentiation=f"reshape_3d_to_2d_33_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(32, 33), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_3d_to_4d_260(self, execution_mode, dtype, consumer):
        """[1,32,2080]→reshape(1,32,8,260)→[1,32,8,260]. Output inner 260: 260%64=4. Both consumers."""
        pytest.xfail(
            "known failure across all dtypes and consumers; "
            "tracked in https://github.com/torch-spyre/torch-spyre/issues/1353"
        )
        x = cached_randn((1, 32, 2080), dtype=dtype, differentiation=f"reshape_3d_to_4d_260_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 32, 8, 260), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_4d_to_3d_129(self, execution_mode, dtype, consumer):
        """[1,32,8,129]→reshape(1,32,1032)→[1,32,1032]. Inner 1032; 1032%64=8. Both consumers."""
        pytest.xfail()
        x = cached_randn((1, 32, 8, 129), dtype=dtype,
                         differentiation=f"reshape_4d_to_3d_129_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 32, 1032), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_5d_to_3d(self, execution_mode, dtype, consumer):
        """[1,2,4,8,7]→reshape(1,8,56)→[1,8,56]. Input inner 7; output 56: 56%64≠0. Both consumers."""
        pytest.xfail(
            "known failure across all dtypes and consumers (Unexpected stick expression / no supported output layout); "
            "tracked in https://github.com/torch-spyre/torch-spyre/issues/5214"
        )
        x = cached_randn((1, 2, 4, 8, 7), dtype=dtype,
                         differentiation=f"reshape_5d_to_3d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 8, 56), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_6d_to_3d(self, execution_mode, dtype, consumer):
        """[1,2,4,8,4,7]→reshape(1,8,224)→[1,8,224]. Inner 224: 224%64=32. Both consumers."""
        pytest.xfail(
            "known failure across all dtypes and consumers (Unexpected stick expression / no supported output layout); "
            "tracked in https://github.com/torch-spyre/torch-spyre/issues/5214"
        )
        x = cached_randn((1, 2, 4, 8, 4, 7), dtype=dtype,
                         differentiation=f"reshape_6d_to_3d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 8, 224), consumer), x)


# ---------------------------------------------------------------------------
# §2.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestReshapeSize1:
    """0D–6D inputs with size-1 dimensions."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_0d_scalar_to_1d(self, execution_mode, dtype, consumer):
        """[]→reshape(1)→[1]. 0D→1D; inner=1. sum_last reduces [1] to scalar (valid)."""
        x = torch.tensor(1.0, dtype=dtype)
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_1d_size1_to_3d(self, execution_mode, dtype, consumer):
        """[1]→reshape(1,1,1)→[1,1,1]. All dims=1. Both consumers."""
        x = cached_randn((1,), dtype=dtype, differentiation=f"reshape_1d_size1_to_3d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 1, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_2d_remove_trailing_size1(self, execution_mode, dtype, consumer):
        """[128,1]→reshape(128)→[128]. 1D output; sum_last reduces to scalar. Both consumers."""
        x = cached_randn((128, 1), dtype=dtype, differentiation=f"reshape_2d_rm_trailing1_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(128), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_3d_trailing_size1_to_2d(self, execution_mode, dtype, consumer):
        """[4,32,1]→reshape(4,32)→[4,32]. Inner changes 1→32. Both consumers."""
        x = cached_randn((4, 32, 1), dtype=dtype, differentiation=f"reshape_3d_trailing1_to_2d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(4, 32), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_4d_two_size1_to_2d(self, execution_mode, dtype, consumer):
        """[1,8,1,256]→reshape(8,256)→[8,256]. Inner unchanged 256→256. Both consumers."""
        if consumer == "sum_last" and dtype == torch.bfloat16:
            pytest.xfail()
        x = cached_randn((1, 8, 1, 256), dtype=dtype,
                         differentiation=f"reshape_4d_two_size1_to_2d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(8, 256), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_4d_size1_trailing_to_3d(self, execution_mode, dtype, consumer):
        """[2,8,32,1]→reshape(2,8,32)→[2,8,32]. Inner changes 1→32. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float16:
            pytest.xfail()
        x = cached_randn((2, 8, 32, 1), dtype=dtype,
                         differentiation=f"reshape_4d_trailing1_to_3d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(2, 8, 32), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_5d_size1_inner_to_4d(self, execution_mode, dtype, consumer):
        """[1,2,1,8,256]→reshape(1,2,8,256)→[1,2,8,256]. Inner unchanged 256. Both consumers."""
        x = cached_randn((1, 2, 1, 8, 256), dtype=dtype,
                         differentiation=f"reshape_5d_size1_to_4d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 2, 8, 256), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_6d_size1_trailing_to_5d(self, execution_mode, dtype, consumer):
        """[1,2,4,8,16,1]→reshape(1,2,4,8,16)→[1,2,4,8,16]. Inner changes 1→16. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 16, 1), dtype=dtype,
                         differentiation=f"reshape_6d_trailing1_to_5d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 2, 4, 8, 16), consumer), x)


# ---------------------------------------------------------------------------
# §2.5 — Restickification scenarios
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestReshapeRestickify:
    """Tests where a reshape causes the Spyre compiler to insert a restickify."""

    def test_restickify_reshape_then_bmm(self, execution_mode):
        """reshape changes K stick dim; matmul requires stick on K → restickify."""
        x = cached_randn((2, 32, 16, 256), dtype=torch.bfloat16,
                         differentiation="reshape_restickify_bmm_x")
        w = cached_randn((256, 256), dtype=torch.bfloat16,
                         differentiation="reshape_restickify_bmm_w")

        def fn(t, weight):
            reshaped = t.reshape(64, 16, 256)
            w_exp = weight.unsqueeze(0).expand(64, 256, 256)
            return torch.bmm(reshaped, w_exp)

        _compare_modes(execution_mode, fn, x, w, atol=0.5, rtol=0.1)

    def test_restickify_reshape_then_add_incompatible_layout(self, execution_mode):
        """Two view-derived tensors with incompatible stick layouts fed into add."""
        x = cached_randn((4, 128, 64), dtype=torch.bfloat16,
                         differentiation="reshape_restickify_add")

        def fn(t):
            return t.reshape(4, 64, 128) + t.reshape(4, 64, 128)

        _compare_modes(execution_mode, fn, x)

    @pytest.mark.xfail
    def test_restickify_reshape_then_dtype_cast(self, execution_mode):
        """Non-offset-free layout after reshape; dtype cast crosses stick-width boundary."""
        x = cached_randn((1, 32, 65), dtype=torch.float16,
                         differentiation="reshape_restickify_cast")

        def fn(t):
            return t.reshape(32, 65).to(torch.float32)

        _compare_modes(execution_mode, fn, x)

    def test_restickify_reshape_then_layernorm(self, execution_mode):
        """Reshape leaves stick on wrong dim; layer_norm requires stick on last dim."""
        x = cached_randn((1, 32, 2816), dtype=torch.bfloat16,
                         differentiation="reshape_restickify_layernorm")

        def fn(t):
            return F.layer_norm(t.reshape(32, 2816), [2816])

        _compare_modes(execution_mode, fn, x)


# ===========================================================================
# §3 — squeeze
# ===========================================================================

# ---------------------------------------------------------------------------
# §3.1 — No-op cases
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSqueezeNoop:
    """squeeze on a non-size-1 dim — verifies the no-op path."""

    def test_squeeze_noop_1d_aligned(self, execution_mode, dtype):
        """[256]→squeeze(0) * 1.0→[256]. No-op; inner 256 unchanged."""
        x = cached_randn((256,), dtype=dtype, differentiation=f"squeeze_noop_1d_aligned_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(0) * 1.0, x)

    def test_squeeze_noop_1d_unaligned(self, execution_mode, dtype):
        """[63]→squeeze(0) * 1.0→[63]. No-op; inner 63 unchanged."""
        x = cached_randn((63,), dtype=dtype, differentiation=f"squeeze_noop_1d_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(0) * 1.0, x)

    def test_squeeze_noop_4d_no_singletons(self, execution_mode, dtype):
        """[2,16,32,256]→squeeze() * 1.0→[2,16,32,256]. No-op; inner 256 unchanged."""
        x = cached_randn((2, 16, 32, 256), dtype=dtype,
                         differentiation=f"squeeze_noop_4d_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze() * 1.0, x)


# ---------------------------------------------------------------------------
# §3.2 — Aligned result after squeeze
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSqueezeAligned:
    """1D–6D inputs producing aligned innermost dim after squeeze."""

    def test_squeeze_2d_leading_to_1d_aligned(self, execution_mode, dtype):
        """[1,256]→squeeze(0) * 1.0→[256]. Non-trailing squeeze; inner 256 unchanged."""
        x = cached_randn((1, 256), dtype=dtype, differentiation=f"squeeze_2d_leading_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(0) * 1.0, x)

    def test_squeeze_3d_batch_to_2d_aligned(self, execution_mode, dtype):
        """[1,32,2816]→squeeze(0) * 1.0→[32,2816]. Non-trailing squeeze; inner 2816 unchanged."""
        x = cached_randn((1, 32, 2816), dtype=dtype,
                         differentiation=f"squeeze_3d_batch_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(0) * 1.0, x)

    def test_squeeze_3d_mid_decode_aligned(self, execution_mode, dtype):
        """[2,1,256]→squeeze(1) * 1.0→[2,256]. Non-trailing squeeze; inner 256 unchanged."""
        x = cached_randn((2, 1, 256), dtype=dtype, differentiation=f"squeeze_3d_mid_decode_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(1) * 1.0, x)

    def test_squeeze_4d_decode_seq_aligned(self, execution_mode, dtype):
        """[1,1,16,256]→squeeze(1) * 1.0→[1,16,256]. Non-trailing squeeze; inner 256 unchanged."""
        x = cached_randn((1, 1, 16, 256), dtype=dtype,
                         differentiation=f"squeeze_4d_decode_seq_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(1) * 1.0, x)

    def test_squeeze_4d_tuple_dims_aligned(self, execution_mode, dtype):
        """[1,16,1,256]→squeeze((0,2)) * 1.0→[16,256]. Neither dim is trailing; inner 256 unchanged."""
        x = cached_randn((1, 16, 1, 256), dtype=dtype,
                         differentiation=f"squeeze_4d_tuple_dims_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze((0, 2)) * 1.0, x)

    def test_squeeze_4d_trailing_to_3d_aligned(self, execution_mode, dtype):
        """[32,8,64,1]→squeeze(-1) * 1.0→[32,8,64]. Trailing squeeze; inner changes 1→64."""
        x = cached_randn((32, 8, 64, 1), dtype=dtype,
                         differentiation=f"squeeze_4d_trailing_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(-1) * 1.0, x)

    def test_squeeze_5d_all_singletons_aligned(self, execution_mode, dtype):
        """[1,1,32,1,256]→squeeze() * 1.0→[32,256]. Inner 256 unchanged."""
        x = cached_randn((1, 1, 32, 1, 256), dtype=dtype,
                         differentiation=f"squeeze_5d_all_singletons_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze() * 1.0, x)

    def test_squeeze_6d_leading_to_5d_aligned(self, execution_mode, dtype):
        """[1,2,4,1,8,256]→squeeze(0) * 1.0→[2,4,1,8,256]. Non-trailing squeeze; inner 256 unchanged."""
        x = cached_randn((1, 2, 4, 1, 8, 256), dtype=dtype,
                         differentiation=f"squeeze_6d_leading_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(0) * 1.0, x)


# ---------------------------------------------------------------------------
# §3.3 — Unaligned result after squeeze
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSqueezeUnaligned:
    """1D–6D inputs producing unaligned innermost dim after squeeze."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_2d_leading_unaligned_33(self, execution_mode, dtype, consumer):
        """[1,33]→squeeze(0)→[33]. Non-trailing squeeze; inner 33 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 33), dtype=dtype, differentiation=f"squeeze_2d_unaligned_33_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_3d_mid_unaligned_70(self, execution_mode, dtype, consumer):
        """[2,1,70]→squeeze(1)→[2,70]. Non-trailing squeeze; inner 70 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((2, 1, 70), dtype=dtype, differentiation=f"squeeze_3d_mid_unaligned_70_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_4d_decode_unaligned_65(self, execution_mode, dtype, consumer):
        """[1,1,16,65]→squeeze(1)→[1,16,65]. Non-trailing squeeze; inner 65 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 1, 16, 65), dtype=dtype,
                         differentiation=f"squeeze_4d_decode_unaligned_65_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_4d_tuple_unaligned_33(self, execution_mode, dtype, consumer):
        """[1,16,1,33]→squeeze((0,2))→[16,33]. Neither dim trailing; inner 33 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 16, 1, 33), dtype=dtype,
                         differentiation=f"squeeze_4d_tuple_unaligned_33_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze((0, 2)), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_5d_all_singletons_unaligned(self, execution_mode, dtype, consumer):
        """[1,1,32,1,30]→squeeze()→[32,30]. Inner 30 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 1, 32, 1, 30), dtype=dtype,
                         differentiation=f"squeeze_5d_all_singletons_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_6d_leading_unaligned_13(self, execution_mode, dtype, consumer):
        """[1,2,4,1,8,13]→squeeze(0)→[2,4,1,8,13]. Non-trailing squeeze; inner 13 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 2, 4, 1, 8, 13), dtype=dtype,
                         differentiation=f"squeeze_6d_leading_unaligned_13_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(0), consumer), x)


# ---------------------------------------------------------------------------
# §3.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSqueezeSize1:
    """0D–6D edge cases involving size-1 dims, scalars, and multi-squeeze chains."""

    @pytest.mark.parametrize("consumer", CONSUMERS_SCALAR)
    def test_squeeze_1d_size1_to_scalar(self, execution_mode, dtype, consumer):
        """[1]→squeeze(0)→[] scalar. 0D output; pointwise only."""
        x = cached_randn((1,), dtype=dtype, differentiation=f"squeeze_1d_size1_to_scalar_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS_SCALAR)
    def test_squeeze_2d_all_size1_to_scalar(self, execution_mode, dtype, consumer):
        """[1,1]→squeeze()→[] scalar. 0D output; pointwise only."""
        x = cached_randn((1, 1), dtype=dtype, differentiation=f"squeeze_2d_all_size1_scalar_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_2d_trailing_size1(self, execution_mode, dtype, consumer):
        """[128,1]→squeeze(-1)→[128]. Trailing squeeze; inner changes 1→128. Both consumers."""
        x = cached_randn((128, 1), dtype=dtype, differentiation=f"squeeze_2d_trailing_size1_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(-1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_3d_all_size1_to_1d(self, execution_mode, dtype, consumer):
        """[1,1,256]→squeeze()→[256]. Inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 1, 256), dtype=dtype,
                         differentiation=f"squeeze_3d_all_size1_to_1d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_4d_mixed_size1_to_1d(self, execution_mode, dtype, consumer):
        """[1,1,32,1]→squeeze()→[32]. Three size-1 dims removed; inner changes 1→32."""
        x = cached_randn((1, 1, 32, 1), dtype=dtype,
                         differentiation=f"squeeze_4d_mixed_size1_to_1d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_5d_size1_inner(self, execution_mode, dtype, consumer):
        """[2,1,32,1,256]→squeeze()→[2,32,256]. Inner 256 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float16:
            pytest.xfail()
        x = cached_randn((2, 1, 32, 1, 256), dtype=dtype,
                         differentiation=f"squeeze_5d_size1_inner_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_6d_size1_to_3d(self, execution_mode, dtype, consumer):
        """[1,1,4,1,8,256]→squeeze()→[4,8,256]. Inner 256 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16):
            pytest.xfail()
        x = cached_randn((1, 1, 4, 1, 8, 256), dtype=dtype,
                         differentiation=f"squeeze_6d_size1_to_3d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_chain_multi_then_reshape(self, execution_mode, dtype, consumer):
        """[1,8,1,256]→squeeze(0).squeeze(1).reshape(-1)→[2048]. Both consumers."""
        if consumer == "sum_last" and dtype == torch.bfloat16:
            pytest.xfail()
        x = cached_randn((1, 8, 1, 256), dtype=dtype,
                         differentiation=f"squeeze_chain_multi_reshape_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.squeeze(0).squeeze(1).reshape(-1), consumer),
            x,
        )


# ===========================================================================
# §4 — unsqueeze
# ===========================================================================

# ---------------------------------------------------------------------------
# §4.1 — Gemma CPU-traced shapes (aligned)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestUnsqueezeCPUTraced:
    """Gemma CPU-traced aligned shapes not covered by module-level tests."""

    def test_unsqueeze_4d_gqa_full_attn_to_5d(self, execution_mode, dtype):
        """[1,2,32,512]→unsqueeze(2) * 1.0→[1,2,1,32,512]. Full-attn GQA; no module-level test."""
        x = cached_randn((1, 2, 32, 512), dtype=dtype,
                         differentiation=f"unsqueeze_4d_gqa_full_attn_{dtype}")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(2) * 1.0, x)


# ---------------------------------------------------------------------------
# §4.2 — Spyre-traced shapes (aligned)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestUnsqueezeSpyreTraced:
    """Spyre-traced shapes not covered by module-level tests."""

    def test_unsqueeze_2d_linear_weight_kv(self, execution_mode, dtype):
        """[2048,2816]→unsqueeze(0) * 1.0→[1,2048,2816]. Non-trailing; no module-level test."""
        x = cached_randn((2048, 2816), dtype=dtype,
                         differentiation=f"unsqueeze_2d_weight_kv_{dtype}")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(0) * 1.0, x)

    def test_unsqueeze_2d_linear_weight_mlp(self, execution_mode, dtype):
        """[2816,2112]→unsqueeze(0) * 1.0→[1,2816,2112]. Non-trailing; no module-level test."""
        x = cached_randn((2816, 2112), dtype=dtype,
                         differentiation=f"unsqueeze_2d_weight_mlp_{dtype}")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(0) * 1.0, x)


# ---------------------------------------------------------------------------
# §4.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestUnsqueezeUnaligned:
    """0D–5D inputs (producing 1D–6D) with unaligned innermost dim."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_0d_to_1d_unaligned(self, execution_mode, dtype, consumer):
        """[]→unsqueeze(0)→[1]. 0D→1D; inner=1. sum_last reduces [1]→scalar (valid). Both consumers."""
        x = torch.tensor(1.5, dtype=dtype)
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_1d_to_2d_unaligned_70(self, execution_mode, dtype, consumer):
        """[70]→unsqueeze(0)→[1,70]. Non-trailing; inner 70 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((70,), dtype=dtype, differentiation=f"unsqueeze_1d_to_2d_70_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_2d_to_3d_unaligned_33(self, execution_mode, dtype, consumer):
        """[32,33]→unsqueeze(0)→[1,32,33]. Non-trailing; inner 33 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((32, 33), dtype=dtype, differentiation=f"unsqueeze_2d_to_3d_33_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_3d_to_4d_rope_unaligned_65(self, execution_mode, dtype, consumer):
        """[1,32,65]→unsqueeze(2)→[1,32,1,65]. Non-trailing; inner 65 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 32, 65), dtype=dtype,
                         differentiation=f"unsqueeze_3d_to_4d_65_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_4d_to_5d_gqa_unaligned_65(self, execution_mode, dtype, consumer):
        """[2,8,32,65]→unsqueeze(2)→[2,8,1,32,65]. Non-trailing; inner 65 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float16:
            pytest.xfail()
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((2, 8, 32, 65), dtype=dtype,
                         differentiation=f"unsqueeze_4d_to_5d_65_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_5d_to_6d_unaligned_13(self, execution_mode, dtype, consumer):
        """[1,2,4,8,13]→unsqueeze(3)→[1,2,4,1,8,13]. Non-trailing (dim 3); inner 13 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 2, 4, 8, 13), dtype=dtype,
                         differentiation=f"unsqueeze_5d_to_6d_13_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(3), consumer), x)


# ---------------------------------------------------------------------------
# §4.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestUnsqueezeSize1:
    """0D–5D inputs (producing 1D–6D). Trailing unsqueeze(-1) creates size-1 innermost."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_0d_to_1d_aligned(self, execution_mode, dtype, consumer):
        """[]→unsqueeze(0)→[1]. 0D→1D; result size-1. Both consumers."""
        x = torch.tensor(2.0, dtype=dtype)
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_1d_size1_to_2d(self, execution_mode, dtype, consumer):
        """[1]→unsqueeze(0)→[1,1]. Size-1 input stays size-1. Both consumers."""
        x = cached_randn((1,), dtype=dtype, differentiation=f"unsqueeze_1d_size1_to_2d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_2d_trailing_size1_aligned(self, execution_mode, dtype, consumer):
        """[32,256]→unsqueeze(-1)→[32,256,1]. Trailing; new inner=1. Both consumers."""
        x = cached_randn((32, 256), dtype=dtype,
                         differentiation=f"unsqueeze_2d_trailing_size1_aligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(-1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_2d_trailing_size1_unaligned(self, execution_mode, dtype, consumer):
        """[4,8,70]→unsqueeze(-1)→[4,8,70,1]. Trailing; new inner=1; preceding 70 unaligned. Both consumers."""
        x = cached_randn((4, 8, 70), dtype=dtype,
                         differentiation=f"unsqueeze_2d_trailing_size1_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(-1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_3d_dual_axis(self, execution_mode, dtype, consumer):
        """[32,256]→unsqueeze(0).unsqueeze(3)→[1,32,256,1]. Trailing unsqueeze adds size-1 innermost. Both consumers."""
        x = cached_randn((32, 256), dtype=dtype,
                         differentiation=f"unsqueeze_3d_dual_axis_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(0).unsqueeze(3), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_4d_size1_innermost_to_5d(self, execution_mode, dtype, consumer):
        """[1,8,32,1]→unsqueeze(2)→[1,8,1,32,1]. Non-trailing (dim 2); inner=1 unchanged. Both consumers."""
        x = cached_randn((1, 8, 32, 1), dtype=dtype,
                         differentiation=f"unsqueeze_4d_size1_innermost_to_5d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_5d_trailing_size1_to_6d(self, execution_mode, dtype, consumer):
        """[1,2,4,8,256]→unsqueeze(-1)→[1,2,4,8,256,1]. Trailing; new inner=1. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 256), dtype=dtype,
                         differentiation=f"unsqueeze_5d_trailing_size1_to_6d_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(-1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_5d_to_6d_aligned(self, execution_mode, dtype, consumer):
        """[1,2,4,8,256]→unsqueeze(3)→[1,2,4,1,8,256]. Non-trailing (dim 3); inner 256 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16):
            pytest.xfail()
        x = cached_randn((1, 2, 4, 8, 256), dtype=dtype,
                         differentiation=f"unsqueeze_5d_to_6d_aligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(3), consumer), x)


# ===========================================================================
# §5 — expand
# ===========================================================================

# ---------------------------------------------------------------------------
# §5.1 — Gemma CPU-traced shapes (aligned)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestExpandCPUTraced:
    """Gemma CPU-traced aligned shapes not covered by module-level tests."""

    def test_expand_rope_freq_heads(self, execution_mode, dtype):
        """[1,32,1,256]→expand(1,32,16,256) * 1.0→[1,32,16,256]. RoPE frequency broadcast; inner 256 unchanged."""
        x = cached_randn((1, 32, 1, 256), dtype=dtype,
                         differentiation=f"expand_rope_freq_heads_{dtype}")
        _compare_modes(execution_mode, lambda t: t.expand(1, 32, 16, 256) * 1.0, x)

    def test_expand_bmm_prep_then_view(self, execution_mode, dtype):
        """[1,16,32,256]→expand(1,16,32,256).view(16,32,256)→[16,32,256]. Same-shape expand then view."""
        x = cached_randn((1, 16, 32, 256), dtype=dtype,
                         differentiation=f"expand_bmm_prep_view_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: t.expand(1, 16, 32, 256).view(16, 32, 256),
            x,
        )

    def test_expand_chain_then_reshape_heads(self, execution_mode, dtype):
        """[1,8,1,32,256]→expand(1,8,2,32,256).reshape(1,16,32,256)→[1,16,32,256]. Forces clone."""
        x = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"expand_chain_reshape_heads_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: t.expand(1, 8, 2, 32, 256).reshape(1, 16, 32, 256),
            x,
        )

    def test_expand_chain_bias_add(self, execution_mode, dtype):
        """[2816]→expand(32,2816)+x[32,2816]. Bias broadcast then add."""
        b = cached_randn((2816,), dtype=dtype, differentiation=f"expand_bias_add_b_{dtype}")
        x = cached_randn((32, 2816), dtype=dtype,
                         differentiation=f"expand_bias_add_x_{dtype}")

        def fn(bias, inp):
            return bias.expand(32, 2816) + inp

        _compare_modes(execution_mode, fn, b, x)


# ---------------------------------------------------------------------------
# §5.2 — Spyre-traced shapes (aligned)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestExpandSpyreTraced:
    """Spyre-traced shapes not covered by module-level tests."""

    def test_expand_spyre_weight_kv(self, execution_mode, dtype):
        """[1,2816,2048]→expand(1,2816,2048) * 1.0→[1,2816,2048]. Same-shape; inner 2048 unchanged."""
        x = cached_randn((1, 2816, 2048), dtype=dtype,
                         differentiation=f"expand_spyre_weight_kv_{dtype}")
        _compare_modes(execution_mode, lambda t: t.expand(1, 2816, 2048) * 1.0, x)

    def test_expand_spyre_weight_mlp(self, execution_mode, dtype):
        """[1,2816,2112]→expand(1,2816,2112) * 1.0→[1,2816,2112]. Same-shape; inner 2112 unchanged."""
        x = cached_randn((1, 2816, 2112), dtype=dtype,
                         differentiation=f"expand_spyre_weight_mlp_{dtype}")
        _compare_modes(execution_mode, lambda t: t.expand(1, 2816, 2112) * 1.0, x)


# ---------------------------------------------------------------------------
# §5.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestExpandUnaligned:
    """0D–6D inputs with unaligned innermost dimension."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_0d_scalar_to_4d_unaligned(self, execution_mode, dtype, consumer):
        """[]→expand(1,16,32,70)→[1,16,32,70]. Inner 70: 70%64=6. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (AssertionError); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5219"
            )
        x = torch.tensor(1.0, dtype=dtype)
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(1, 16, 32, 70), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_1d_bias_unaligned_70(self, execution_mode, dtype, consumer):
        """[70]→expand(2,32,70)→[2,32,70]. Inner 70 unchanged. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (SubprocException / DtException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5237"
            )
        x = cached_randn((70,), dtype=dtype, differentiation=f"expand_1d_bias_70_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(2, 32, 70), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_2d_batch_unaligned_33(self, execution_mode, dtype, consumer):
        """[1,33]→expand(4,33)→[4,33]. Inner 33 unchanged. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (SubprocException / DtException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5237"
            )
        x = cached_randn((1, 33), dtype=dtype, differentiation=f"expand_2d_batch_33_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(4, 33), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_3d_batch_unaligned_33(self, execution_mode, dtype, consumer):
        """[1,32,33]→expand(4,32,33)→[4,32,33]. Inner 33 unchanged. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (SubprocException / DtException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5237"
            )
        x = cached_randn((1, 32, 33), dtype=dtype,
                         differentiation=f"expand_3d_batch_33_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(4, 32, 33), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_5d_gqa_unaligned_65(self, execution_mode, dtype, consumer):
        """[1,8,1,32,65]→expand(1,8,2,32,65)→[1,8,2,32,65]. Inner 65 unchanged. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (SubprocException / DtException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5237"
            )
        x = cached_randn((1, 8, 1, 32, 65), dtype=dtype,
                         differentiation=f"expand_5d_gqa_65_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(1, 8, 2, 32, 65), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_4d_mask_unaligned_seq33(self, execution_mode, dtype, consumer):
        """[1,1,33,33]→expand(2,8,33,33)→[2,8,33,33]. Inner 33 unchanged. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (SubprocException / DtException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5237"
            )
        x = cached_randn((1, 1, 33, 33), dtype=dtype,
                         differentiation=f"expand_4d_mask_33_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(2, 8, 33, 33), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_3d_minus_one_unaligned(self, execution_mode, dtype, consumer):
        """[2,1,70]→expand(-1,16,-1)→[2,16,70]. -1 preserves dim; inner 70 unchanged. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (SubprocException / DtException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5237"
            )
        x = cached_randn((2, 1, 70), dtype=dtype,
                         differentiation=f"expand_3d_minus_one_70_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(-1, 16, -1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_6d_unaligned_inner(self, execution_mode, dtype, consumer):
        """[1,2,1,4,1,13]→expand(1,2,8,4,16,13)→[1,2,8,4,16,13]. 6D; inner 13 unchanged. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (SubprocException / DtException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5237"
            )
        x = cached_randn((1, 2, 1, 4, 1, 13), dtype=dtype,
                         differentiation=f"expand_6d_unaligned_13_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.expand(1, 2, 8, 4, 16, 13), consumer),
            x,
        )


# ---------------------------------------------------------------------------
# §5.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestExpandSize1:
    """0D–6D. Size-1 innermost can stay size-1 or be broadcast to aligned."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_0d_scalar_to_4d_aligned(self, execution_mode, dtype, consumer):
        """[]→expand(1,16,32,256)→[1,16,32,256]. 0D broadcast; inner 256 unchanged. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (AssertionError); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5219"
            )
        x = torch.tensor(1.0, dtype=dtype)
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(1, 16, 32, 256), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_1d_size1_keep_inner(self, execution_mode, dtype, consumer):
        """[1]→expand(2,32,1)→[2,32,1]. Innermost stays size-1. Both consumers."""
        x = cached_randn((1,), dtype=dtype, differentiation=f"expand_1d_size1_keep_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(2, 32, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_1d_size1_to_aligned(self, execution_mode, dtype, consumer):
        """[1]→expand(256)→[256]. 1D size-1 broadcast to aligned 256. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (AssertionError); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5219"
            )
        x = cached_randn((1,), dtype=dtype, differentiation=f"expand_1d_size1_to_aligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(256), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_2d_size1_inner_to_2816(self, execution_mode, dtype, consumer):
        """[32,1]→expand(32,2816)→[32,2816]. Size-1 innermost broadcast to aligned 2816. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (AssertionError); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5219"
            )
        x = cached_randn((32, 1), dtype=dtype, differentiation=f"expand_2d_size1_to_2816_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(32, 2816), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_3d_moe_trailing1_broadcast(self, execution_mode, dtype, consumer):
        """[32,8,1]→expand(32,8,2816)→[32,8,2816]. MoE routing trailing-1 broadcast. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (AssertionError); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5219"
            )
        x = cached_randn((32, 8, 1), dtype=dtype,
                         differentiation=f"expand_3d_moe_trailing1_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(32, 8, 2816), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_4d_all_size1_except_inner(self, execution_mode, dtype, consumer):
        """[1,1,1,256]→expand(4,16,32,256)→[4,16,32,256]. All outer size-1; inner 256 unchanged. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (SubprocException / DtException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5237"
            )
        x = cached_randn((1, 1, 1, 256), dtype=dtype,
                         differentiation=f"expand_4d_all_size1_except_inner_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(4, 16, 32, 256), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_4d_mqa_16x(self, execution_mode, dtype, consumer):
        """[1,1,1,256]→expand(1,16,32,256)→[1,16,32,256]. MQA (KV=1) 16× broadcast. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (SubprocException / DtException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5237"
            )
        x = cached_randn((1, 1, 1, 256), dtype=dtype,
                         differentiation=f"expand_4d_mqa_16x_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(1, 16, 32, 256), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_5d_size1_inner_aligned(self, execution_mode, dtype, consumer):
        """[16,1,4,16,1]→expand(16,8,4,16,256)→[16,8,4,16,256]. 5D; size-1 inner → aligned 256. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (AssertionError); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5219"
            )
        x = cached_randn((16, 1, 4, 16, 1), dtype=dtype,
                         differentiation=f"expand_5d_size1_inner_aligned_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.expand(16, 8, 4, 16, 256), consumer),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_5d_size1_inner_keep(self, execution_mode, dtype, consumer):
        """[16,1,4,16,1]→expand(16,8,4,16,1)→[16,8,4,16,1]. 5D; inner stays size-1. Both consumers."""
        x = cached_randn((16, 1, 4, 16, 1), dtype=dtype,
                         differentiation=f"expand_5d_size1_inner_keep_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.expand(16, 8, 4, 16, 1), consumer),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_6d_size1_to_aligned(self, execution_mode, dtype, consumer):
        """[1,1,1,4,1,256]→expand(2,8,4,4,16,256)→[2,8,4,4,16,256]. 6D; multiple size-1 broadcast. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (SubprocException / DtException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5237"
            )
        x = cached_randn((1, 1, 1, 4, 1, 256), dtype=dtype,
                         differentiation=f"expand_6d_size1_to_aligned_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.expand(2, 8, 4, 4, 16, 256), consumer),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_6d_aligned(self, execution_mode, dtype, consumer):
        """[1,2,1,4,1,256]→expand(8,2,4,4,16,256)→[8,2,4,4,16,256]. 6D aligned; inner 256 unchanged. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail(
                "known failure for sum_last across all dtypes (SubprocException / DtException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5237"
            )
        x = cached_randn((1, 2, 1, 4, 1, 256), dtype=dtype,
                         differentiation=f"expand_6d_aligned_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.expand(8, 2, 4, 4, 16, 256), consumer),
            x,
        )


# ===========================================================================
# §6 — transpose
# ===========================================================================

# ---------------------------------------------------------------------------
# §6.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestTransposeUnaligned:
    """2D–6D inputs with unaligned innermost dimension."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_2d_unaligned_inner_swap(self, execution_mode, dtype, consumer):
        """[65,256]→transpose(0,1)→[256,65]. Inner changes 256→65; 65%64=1. Both consumers."""
        if dtype == torch.float16:
            pytest.xfail()
        x = cached_randn((65, 256), dtype=dtype,
                         differentiation=f"transpose_2d_unaligned_swap_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_3d_last_two_unaligned(self, execution_mode, dtype, consumer):
        """[4,32,70]→transpose(1,2)→[4,70,32]. Inner changes 70→32; input inner 70 unaligned. Both consumers."""
        if dtype in (torch.float16, torch.bfloat16):
            pytest.xfail()
        x = cached_randn((4, 32, 70), dtype=dtype,
                         differentiation=f"transpose_3d_last_two_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(1, 2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_4d_key_unaligned_65(self, execution_mode, dtype, consumer):
        """[1,16,32,65]→transpose(2,3)→[1,16,65,32]. Inner changes 65→32. Both consumers."""
        if dtype in (torch.float16, torch.bfloat16):
            pytest.xfail()
        x = cached_randn((1, 16, 32, 65), dtype=dtype,
                         differentiation=f"transpose_4d_key_65_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(2, 3), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_4d_head_swap_unaligned_inner(self, execution_mode, dtype, consumer):
        """[2,32,8,70]→transpose(1,2)→[2,8,32,70]. Inner 70 unchanged; 70%64=6. Both consumers."""
        if dtype == torch.float16:
            pytest.xfail()
        if dtype == torch.float32:
            pytest.xfail(
                "known failure for float32; "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((2, 32, 8, 70), dtype=dtype,
                         differentiation=f"transpose_4d_head_swap_70_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(1, 2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_5d_unaligned(self, execution_mode, dtype, consumer):
        """[1,8,1,32,65]→transpose(1,3)→[1,32,1,8,65]. Inner 65 unchanged. Both consumers."""
        if dtype == torch.float16:
            pytest.xfail()
        if dtype == torch.float32:
            pytest.xfail(
                "known failure for float32; "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 8, 1, 32, 65), dtype=dtype,
                         differentiation=f"transpose_5d_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(1, 3), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_6d_unaligned(self, execution_mode, dtype, consumer):
        """[1,2,4,8,16,13]→transpose(1,4)→[1,16,4,8,2,13]. Inner 13 unchanged; 13 < 64. Both consumers."""
        if dtype == torch.float32:
            pytest.xfail(
                "known failure for float32; "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 2, 4, 8, 16, 13), dtype=dtype,
                         differentiation=f"transpose_6d_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(1, 4), consumer), x)


# ---------------------------------------------------------------------------
# §6.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestTransposeSize1:
    """Size-1 dimension interactions."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_2d_size1_trailing(self, execution_mode, dtype, consumer):
        """[256,1]→transpose(0,1)→[1,256]. Trailing size-1 moves to leading. Both consumers."""
        x = cached_randn((256, 1), dtype=dtype,
                         differentiation=f"transpose_2d_size1_trailing_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_3d_size1_middle(self, execution_mode, dtype, consumer):
        """[4,1,256]→transpose(0,1)→[1,4,256]. Size-1 at middle; inner 256 unchanged. Both consumers."""
        x = cached_randn((4, 1, 256), dtype=dtype,
                         differentiation=f"transpose_3d_size1_middle_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_4d_size1_gqa_expand(self, execution_mode, dtype, consumer):
        """[1,8,1,256]→transpose(1,2)→[1,1,8,256]. GQA-like; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 8, 1, 256), dtype=dtype,
                         differentiation=f"transpose_4d_size1_gqa_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(1, 2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_5d_size1_to_trailing(self, execution_mode, dtype, consumer):
        """[1,8,32,1,256]→transpose(3,4)→[1,8,32,256,1]. Size-1 moved to innermost. Both consumers."""
        x = cached_randn((1, 8, 32, 1, 256), dtype=dtype,
                         differentiation=f"transpose_5d_size1_to_trailing_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(3, 4), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_6d_all_size1_except_inner(self, execution_mode, dtype, consumer):
        """[1,1,1,1,1,256]→transpose(0,4)→[1,1,1,1,1,256]. No-op in practice; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 1, 1, 1, 1, 256), dtype=dtype,
                         differentiation=f"transpose_6d_all_size1_except_inner_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(0, 4), consumer), x)


# ===========================================================================
# §7 — t (2D transpose)
# ===========================================================================

# ---------------------------------------------------------------------------
# §7.2 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestTUnaligned:
    """Unaligned / swapped dimensions."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_1d_noop(self, execution_mode, dtype, consumer):
        """[256].t()→[256]. 1D t() is a no-op; identity. Both consumers."""
        x = cached_randn((256,), dtype=dtype,
                         differentiation=f"t_1d_noop_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_2d_unaligned_inner_to_outer(self, execution_mode, dtype, consumer):
        """[65,256].t()→[256,65]. New inner 65; 65%64=1. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float16:
            pytest.xfail()
        x = cached_randn((65, 256), dtype=dtype,
                         differentiation=f"t_2d_unaligned_inner_to_outer_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_2d_both_unaligned(self, execution_mode, dtype, consumer):
        """[33,70].t()→[70,33]. New inner 33 < 64; both dims unaligned. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float16:
            pytest.xfail()
        x = cached_randn((33, 70), dtype=dtype,
                         differentiation=f"t_2d_both_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_2d_aligned_swap(self, execution_mode, dtype, consumer):
        """[128,64].t()→[64,128]. New inner 128; both aligned; inner changes. Both consumers."""
        if consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16):
            pytest.xfail()
        x = cached_randn((128, 64), dtype=dtype,
                         differentiation=f"t_2d_aligned_swap_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)


# ---------------------------------------------------------------------------
# §7.3 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestTSize1:
    """Size-1 dimension interactions with t()."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_2d_size1_trailing(self, execution_mode, dtype, consumer):
        """[256,1].t()→[1,256]. Trailing size-1 → leading; new inner 256. Both consumers."""
        x = cached_randn((256, 1), dtype=dtype,
                         differentiation=f"t_2d_size1_trailing_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_2d_size1_leading(self, execution_mode, dtype, consumer):
        """[1,256].t()→[256,1]. Size-1 leading → trailing; new inner 1 (ultra-unaligned). Both consumers."""
        x = cached_randn((1, 256), dtype=dtype,
                         differentiation=f"t_2d_size1_leading_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_2d_both_size1(self, execution_mode, dtype, consumer):
        """[1,1].t()→[1,1]. All dims=1; no layout change. Both consumers."""
        x = cached_randn((1, 1), dtype=dtype,
                         differentiation=f"t_2d_both_size1_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)


# ===========================================================================
# §9 — clone
# ===========================================================================

# ---------------------------------------------------------------------------
# §9.2 — Gemma Spyre-traced shapes (aligned)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestCloneSpyreTraced:
    """Shapes from gemma4_fx_module_trace_v2.html (Spyre compile, seq=64)."""

    def test_clone_moe_token_3d(self, execution_mode, dtype):
        """[1,8,64].clone(). MoE token routing; inner 64 aligned."""
        x = cached_randn((1, 8, 64), dtype=dtype,
                         differentiation=f"clone_moe_token_3d_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_moe_hidden_3d(self, execution_mode, dtype):
        """[1,8,2816].clone(). MoE hidden state; inner 2816 aligned."""
        x = cached_randn((1, 8, 2816), dtype=dtype,
                         differentiation=f"clone_moe_hidden_3d_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_attn_output_4d_s64(self, execution_mode, dtype):
        """[1,16,64,256].clone(). Spyre attn output; aligned."""
        x = cached_randn((1, 16, 64, 256), dtype=dtype,
                         differentiation=f"clone_attn_output_4d_s64_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_attn_output_full_attn_s64(self, execution_mode, dtype):
        """[1,16,64,512].clone(). Full-attention Spyre output; inner 512 aligned."""
        x = cached_randn((1, 16, 64, 512), dtype=dtype,
                         differentiation=f"clone_attn_output_full_attn_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_after_permute_head_swap(self, execution_mode, dtype):
        """[1,64,16,256].permute(0,2,1,3).clone()→[1,16,64,256]. Clone of non-contiguous permuted tensor; restickify path."""
        x = cached_randn((1, 64, 16, 256), dtype=dtype,
                         differentiation=f"clone_after_permute_{dtype}")
        _compare_modes(execution_mode, lambda t: t.permute(0, 2, 1, 3).clone(), x)


# ---------------------------------------------------------------------------
# §9.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestCloneUnaligned:
    """1D–6D inputs with unaligned innermost dimension."""

    def test_clone_1d_unaligned(self, execution_mode, dtype):
        """[63].clone(). Inner 63 < 64."""
        x = cached_randn((63,), dtype=dtype,
                         differentiation=f"clone_1d_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_2d_unaligned_inner(self, execution_mode, dtype):
        """[32,70].clone(). Inner 70; 70%64=6."""
        x = cached_randn((32, 70), dtype=dtype,
                         differentiation=f"clone_2d_unaligned_inner_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_3d_unaligned_after_transpose(self, execution_mode, dtype):
        """[4,256,65].transpose(1,2).clone()→[4,65,256]. Clone of non-contiguous; output inner 256."""
        if dtype == torch.float32 and execution_mode == "eager":
            pytest.xfail(
                "known failure (no mechanism to resolve stick incompatibility); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5216"
            )
        if dtype == torch.float32 and execution_mode == "compiled":
            pytest.xfail(
                "known failure (ReStickifyOpHBM on DataFormats.IEEE_FP32); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5216"
            )
        x = cached_randn((4, 256, 65), dtype=dtype,
                         differentiation=f"clone_3d_after_transpose_{dtype}")
        _compare_modes(execution_mode, lambda t: t.transpose(1, 2).clone(), x)

    def test_clone_4d_unaligned_inner_33(self, execution_mode, dtype):
        """[2,8,32,33].clone(). Inner 33 < 64."""
        x = cached_randn((2, 8, 32, 33), dtype=dtype,
                         differentiation=f"clone_4d_unaligned_33_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_4d_unaligned_inner_70(self, execution_mode, dtype):
        """[2,8,32,70].clone(). Inner 70 > 64 non-multiple; covers >64 bucket at 4D."""
        x = cached_randn((2, 8, 32, 70), dtype=dtype,
                         differentiation=f"clone_4d_unaligned_70_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_5d_unaligned(self, execution_mode, dtype):
        """[1,2,4,8,13].clone(). Inner 13 < 64."""
        x = cached_randn((1, 2, 4, 8, 13), dtype=dtype,
                         differentiation=f"clone_5d_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_6d_unaligned(self, execution_mode, dtype):
        """[1,2,4,8,16,7].clone(). Inner 7 < 64."""
        x = cached_randn((1, 2, 4, 8, 16, 7), dtype=dtype,
                         differentiation=f"clone_6d_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_6d_unaligned_inner_70(self, execution_mode, dtype):
        """[1,2,4,8,16,70].clone(). Inner 70 > 64 non-multiple; covers >64 bucket at 6D."""
        x = cached_randn((1, 2, 4, 8, 16, 70), dtype=dtype,
                         differentiation=f"clone_6d_unaligned_70_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)


# ---------------------------------------------------------------------------
# §9.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestCloneSize1:
    """0D–5D with size-1 dimensions."""

    def test_clone_scalar_0d(self, execution_mode, dtype):
        """[] scalar.clone(). 0D scalar materialise."""
        x = torch.tensor(1.0, dtype=dtype)
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_1d_size1(self, execution_mode, dtype):
        """[1].clone(). Size-1 vector; extreme unaligned."""
        x = cached_randn((1,), dtype=dtype,
                         differentiation=f"clone_1d_size1_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_3d_size1_trailing(self, execution_mode, dtype):
        """[32,64,1].clone(). Size-1 innermost."""
        x = cached_randn((32, 64, 1), dtype=dtype,
                         differentiation=f"clone_3d_size1_trailing_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_4d_size1_after_unsqueeze(self, execution_mode, dtype):
        """[1,8,32,256].unsqueeze(2).clone()→[1,8,1,32,256]. Clone after unsqueeze; inner 256 unchanged."""
        x = cached_randn((1, 8, 32, 256), dtype=dtype,
                         differentiation=f"clone_4d_size1_after_unsqueeze_{dtype}")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(2).clone(), x)

    def test_clone_5d_gqa_size1_dim(self, execution_mode, dtype):
        """[1,8,1,32,256].clone(). GQA-like; size-1 at dim-2; inner 256 aligned."""
        x = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"clone_5d_gqa_size1_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)


# ===========================================================================
# §10 — cat
# ===========================================================================

# ---------------------------------------------------------------------------
# §10.1 — Gemma CPU-traced shapes (aligned)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestCatCPUTraced:
    """KV RoPE half-rotation concat — no matching module-level test."""

    def test_cat_rope_halves_kv_sliding(self, execution_mode, dtype):
        """[1,32,8,128] × 2 → cat(dim=-1) → [1,32,8,256] * 1.0. KV RoPE concat."""
        a = cached_randn((1, 32, 8, 128), dtype=dtype,
                         differentiation=f"cat_rope_halves_kv_a_{dtype}")
        b = cached_randn((1, 32, 8, 128), dtype=dtype,
                         differentiation=f"cat_rope_halves_kv_b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: torch.cat([a, b], dim=-1) * 1.0, a, b
        )


# ---------------------------------------------------------------------------
# §10.2 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestCatUnaligned:
    """1D–6D inputs with unaligned innermost dimension."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_1d_unaligned_halves(self, execution_mode, dtype, consumer):
        """[33] + [33] → cat(dim=0) → [66]. Output inner 66; 66%64=2. Both consumers."""
        pytest.xfail(
            "known failure across all dtypes and consumers; "
            "tracked in https://github.com/torch-spyre/torch-spyre/issues/3916"
        )
        a = cached_randn((33,), dtype=dtype,
                         differentiation=f"cat_1d_unaligned_a_{dtype}")
        b = cached_randn((33,), dtype=dtype,
                         differentiation=f"cat_1d_unaligned_b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b]), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_2d_trailing_unaligned(self, execution_mode, dtype, consumer):
        """[32,33] + [32,33] → cat(dim=-1) → [32,66]. Output inner 66 unaligned. Both consumers."""
        if consumer == "sum_last":
            pytest.xfail()
        if dtype == torch.float32:
            pytest.xfail(
                "known failure for float32 (no mechanism to resolve stick incompatibility); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5216"
            )
        a = cached_randn((32, 33), dtype=dtype,
                         differentiation=f"cat_2d_trailing_unaligned_a_{dtype}")
        b = cached_randn((32, 33), dtype=dtype,
                         differentiation=f"cat_2d_trailing_unaligned_b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=-1), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_3d_seq_dim_unaligned(self, execution_mode, dtype, consumer):
        """[1,33,256] + [1,33,256] → cat(dim=1) → [1,66,256]. Seq dim unaligned; inner 256 unchanged. Both consumers."""
        pytest.xfail()
        a = cached_randn((1, 33, 256), dtype=dtype,
                         differentiation=f"cat_3d_seq_unaligned_a_{dtype}")
        b = cached_randn((1, 33, 256), dtype=dtype,
                         differentiation=f"cat_3d_seq_unaligned_b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=1), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_4d_unaligned_inner_halves(self, execution_mode, dtype, consumer):
        """[1,16,32,33] × 2 → cat(dim=-1) → [1,16,32,66]. Unaligned inputs → unaligned output. Both consumers."""
        if dtype == torch.float32:
            pytest.xfail(
                "known failure for float32 (no mechanism to resolve stick incompatibility); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5216"
            )
        if consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16):
            pytest.xfail(
                "known failure for sum_last/float16/bfloat16 (spyre <-> cpu mismatch); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5216"
            )
        a = cached_randn((1, 16, 32, 33), dtype=dtype,
                         differentiation=f"cat_4d_unaligned_inner_a_{dtype}")
        b = cached_randn((1, 16, 32, 33), dtype=dtype,
                         differentiation=f"cat_4d_unaligned_inner_b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=-1), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_5d_seq_unaligned(self, execution_mode, dtype, consumer):
        """[1,8,1,33,256] × 2 → cat(dim=3) → [1,8,1,66,256]. Seq dim unaligned; inner 256 unchanged. Both consumers."""
        pytest.xfail()
        a = cached_randn((1, 8, 1, 33, 256), dtype=dtype,
                         differentiation=f"cat_5d_seq_unaligned_a_{dtype}")
        b = cached_randn((1, 8, 1, 33, 256), dtype=dtype,
                         differentiation=f"cat_5d_seq_unaligned_b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=3), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_6d_inner_unaligned(self, execution_mode, dtype, consumer):
        """[1,2,4,8,16,13] × 2 → cat(dim=-1) → [1,2,4,8,16,26]. 6D; inner 26 < 64. Both consumers."""
        if dtype == torch.float32:
            pytest.xfail(
                "known failure for float32 (no mechanism to resolve stick incompatibility); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5216"
            )
        if consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16):
            pytest.xfail(
                "known failure for sum_last/float16/bfloat16 (spyre <-> cpu mismatch); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5216"
            )
        a = cached_randn((1, 2, 4, 8, 16, 13), dtype=dtype,
                         differentiation=f"cat_6d_inner_unaligned_a_{dtype}")
        b = cached_randn((1, 2, 4, 8, 16, 13), dtype=dtype,
                         differentiation=f"cat_6d_inner_unaligned_b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=-1), consumer), a, b
        )


# ---------------------------------------------------------------------------
# §10.3 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestCatSize1:
    """Size-1 dimension interactions."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_3d_size1_seq_expand(self, execution_mode, dtype, consumer):
        """[1,1,256] + [1,32,256] → cat(dim=1) → [1,33,256]. Size-1 seq prepended. Both consumers."""
        pytest.xfail()
        a = cached_randn((1, 1, 256), dtype=dtype,
                         differentiation=f"cat_3d_size1_seq_a_{dtype}")
        b = cached_randn((1, 32, 256), dtype=dtype,
                         differentiation=f"cat_3d_size1_seq_b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=1), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_4d_size1_trailing_halves(self, execution_mode, dtype, consumer):
        """[32,8,16,1] × 2 → cat(dim=-1) → [32,8,16,2]. Two size-1 halves → size-2 inner. Both consumers."""
        if dtype == torch.float16:
            pytest.xfail(
                "known failure for 16 elements/stick dtype (float16); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5196"
            )
        if dtype in (torch.bfloat16, torch.float32):
            pytest.xfail(
                "known failure for 32 elements/stick dtypes (bfloat16, float32); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5196"
            )
        a = cached_randn((32, 8, 16, 1), dtype=dtype,
                         differentiation=f"cat_4d_size1_trailing_a_{dtype}")
        b = cached_randn((32, 8, 16, 1), dtype=dtype,
                         differentiation=f"cat_4d_size1_trailing_b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=-1), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_5d_gqa_kv_cache(self, execution_mode, dtype, consumer):
        """[1,8,1,32,256] × 2 → cat(dim=2) → [1,8,2,32,256]. GQA-like cache; inner 256 unchanged. Both consumers."""
        pytest.xfail()
        a = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"cat_5d_gqa_kv_a_{dtype}")
        b = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"cat_5d_gqa_kv_b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=2), consumer), a, b
        )


# ===========================================================================
# §11 — select
# ===========================================================================

# ---------------------------------------------------------------------------
# §11.2 — Float shapes, 1D–6D
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSelectFloat:
    """Float dtype shapes covering 1D→0D through 6D→5D."""

    def test_select_1d_to_scalar(self, execution_mode, dtype):
        """[256] → select(0,0) → [] * 1.0. 0D output; sum undefined on scalar."""
        x = cached_randn((256,), dtype=dtype,
                         differentiation=f"select_1d_to_scalar_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(0, 0) * 1.0, x)

    def test_select_2d_leading_dim(self, execution_mode, dtype):
        """[8,256] → select(0,3) → [256] * 1.0. Leading dim selected; inner 256 unchanged."""
        x = cached_randn((8, 256), dtype=dtype,
                         differentiation=f"select_2d_leading_dim_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(0, 3) * 1.0, x)

    def test_select_3d_seq_select(self, execution_mode, dtype):
        """[1,32,256] → select(1,0) → [1,256] * 1.0. Middle dim selected; inner 256 unchanged."""
        x = cached_randn((1, 32, 256), dtype=dtype,
                         differentiation=f"select_3d_seq_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(1, 0) * 1.0, x)

    def test_select_4d_head_select(self, execution_mode, dtype):
        """[1,16,32,256] → select(1,0) → [1,32,256] * 1.0. Head dim selected; inner 256 unchanged."""
        x = cached_randn((1, 16, 32, 256), dtype=dtype,
                         differentiation=f"select_4d_head_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(1, 0) * 1.0, x)

    def test_select_4d_last_token_trailing(self, execution_mode, dtype):
        """[1,16,32,256] → select(2,-1) → [1,16,256] * 1.0. Seq dim selected; inner 256 unchanged."""
        x = cached_randn((1, 16, 32, 256), dtype=dtype,
                         differentiation=f"select_4d_last_token_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(2, -1) * 1.0, x)

    def test_select_5d_middle_dim(self, execution_mode, dtype):
        """[1,8,2,32,256] → select(2,0) → [1,8,32,256] * 1.0. GQA-like; inner 256 unchanged."""
        x = cached_randn((1, 8, 2, 32, 256), dtype=dtype,
                         differentiation=f"select_5d_middle_dim_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(2, 0) * 1.0, x)

    def test_select_6d_to_5d(self, execution_mode, dtype):
        """[1,2,4,8,16,256] → select(3,0) → [1,2,4,16,256] * 1.0. 6D→5D; inner 256 unchanged."""
        x = cached_randn((1, 2, 4, 8, 16, 256), dtype=dtype,
                         differentiation=f"select_6d_to_5d_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(3, 0) * 1.0, x)


# ---------------------------------------------------------------------------
# §11.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSelectUnaligned:
    """Unaligned innermost dimension."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_select_2d_leading_unaligned(self, execution_mode, dtype, consumer):
        """[8,70] → select(0,0) → [70]. Inner 70 unchanged; 70%64=6. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((8, 70), dtype=dtype,
                         differentiation=f"select_2d_leading_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.select(0, 0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_select_3d_trailing_unaligned_new_inner(self, execution_mode, dtype, consumer):
        """[4,256,33] → select(1,0) → [4,33]. Inner 33 unchanged from dim-2; 33 < 64. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((4, 256, 33), dtype=dtype,
                         differentiation=f"select_3d_trailing_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.select(1, 0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_select_4d_unaligned_inner(self, execution_mode, dtype, consumer):
        """[1,16,32,65] → select(1,0) → [1,32,65]. Inner 65 unchanged; 65%64=1. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float16:
            pytest.xfail()
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 16, 32, 65), dtype=dtype,
                         differentiation=f"select_4d_unaligned_inner_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.select(1, 0), consumer), x)


# ---------------------------------------------------------------------------
# §11.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSelectSize1:
    """Size-1 dimension interactions."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_select_2d_size1_leading(self, execution_mode, dtype, consumer):
        """[1,256] → select(0,0) → [256]. Removes size-1 leading dim. Both consumers."""
        x = cached_randn((1, 256), dtype=dtype,
                         differentiation=f"select_2d_size1_leading_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.select(0, 0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_select_3d_size1_gqa(self, execution_mode, dtype, consumer):
        """[1,8,1] → select(2,0) → [1,8]. Select removes trailing size-1; inner changes 1→8. Both consumers."""
        x = cached_randn((1, 8, 1), dtype=dtype,
                         differentiation=f"select_3d_size1_gqa_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.select(2, 0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_select_5d_size1_dim(self, execution_mode, dtype, consumer):
        """[1,8,1,32,256] → select(2,0) → [1,8,32,256]. Select the size-1 dim; inner 256 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16):
            pytest.xfail()
        x = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"select_5d_size1_dim_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.select(2, 0), consumer), x)


# ===========================================================================
# §12 — narrow / slice
# ===========================================================================

# ---------------------------------------------------------------------------
# §12.2 — Gemma Spyre-traced shapes
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestNarrowSpyreTraced:
    """Shapes from gemma4_fx_module_trace_v2.html."""

    def test_slice_3d_last_element(self, execution_mode, dtype):
        """[64,128,64]→narrow(2,0,1)→[64,128,1] * 1.0. Inner changes 64→1."""
        x = cached_randn((64, 128, 64), dtype=dtype,
                         differentiation=f"narrow_3d_last_element_{dtype}")
        _compare_modes(execution_mode, lambda t: t.narrow(2, 0, 1) * 1.0, x)

    def test_slice_3d_half_inner(self, execution_mode, dtype):
        """[1,8,64]→narrow(2,0,32)→[1,8,32] * 1.0. Inner changes 64→32."""
        x = cached_randn((1, 8, 64), dtype=dtype,
                         differentiation=f"narrow_3d_half_inner_{dtype}")
        _compare_modes(execution_mode, lambda t: t.narrow(2, 0, 32) * 1.0, x)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestNarrowSpyreTracedBf16:
    """bf16-only shape from Spyre trace."""

    def test_slice_3d_single_element_inner_bf16(self, execution_mode):
        """[1,8,64]→narrow(2,0,1)→[1,8,1] * 1.0. bf16 only; inner changes 64→1."""
        x = cached_randn((1, 8, 64), dtype=torch.bfloat16,
                         differentiation="narrow_3d_single_element_bf16")
        _compare_modes(execution_mode, lambda t: t.narrow(2, 0, 1) * 1.0, x)


# ---------------------------------------------------------------------------
# §12.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestNarrowUnaligned:
    """1D–6D inputs with unaligned output innermost dimension."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_1d_unaligned_result(self, execution_mode, dtype, consumer):
        """[256]→narrow(0,0,65)→[65]. Output inner 65; 65%64=1. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((256,), dtype=dtype,
                         differentiation=f"narrow_1d_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(0, 0, 65), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_2d_trailing_unaligned(self, execution_mode, dtype, consumer):
        """[8,128]→narrow(1,0,33)→[8,33]. Inner changes 128→33; 33 < 64. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((8, 128), dtype=dtype,
                         differentiation=f"narrow_2d_trailing_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(1, 0, 33), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_3d_inner_unaligned(self, execution_mode, dtype, consumer):
        """[1,32,128]→narrow(2,0,70)→[1,32,70]. Inner changes 128→70; 70%64=6. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float16:
            pytest.xfail()
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 32, 128), dtype=dtype,
                         differentiation=f"narrow_3d_inner_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(2, 0, 70), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_4d_head_dim_unaligned(self, execution_mode, dtype, consumer):
        """[1,16,32,256]→narrow(3,0,65)→[1,16,32,65]. Inner changes 256→65. Both consumers."""
        if consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16):
            pytest.xfail()
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 16, 32, 256), dtype=dtype,
                         differentiation=f"narrow_4d_head_dim_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(3, 0, 65), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_5d_inner_below_stick(self, execution_mode, dtype, consumer):
        """[1,2,4,8,128]→narrow(4,0,13)→[1,2,4,8,13]. Inner changes 128→13 < 64. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 2, 4, 8, 128), dtype=dtype,
                         differentiation=f"narrow_5d_inner_below_stick_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(4, 0, 13), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_6d_inner_unaligned(self, execution_mode, dtype, consumer):
        """[1,2,4,8,16,128]→narrow(5,0,33)→[1,2,4,8,16,33]. 6D; inner changes 128→33 < 64. Both consumers."""
        if consumer == "sum_last" and dtype == torch.float16:
            pytest.xfail()
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32 (SubprocException); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        x = cached_randn((1, 2, 4, 8, 16, 128), dtype=dtype,
                         differentiation=f"narrow_6d_inner_unaligned_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(5, 0, 33), consumer), x)


# ---------------------------------------------------------------------------
# §12.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestNarrowSize1:
    """Size-1 dimension interactions."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_2d_size1_leading(self, execution_mode, dtype, consumer):
        """[1,256]→narrow(0,0,1)→[1,256]. Narrow on size-1 dim; no-op; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 256), dtype=dtype,
                         differentiation=f"narrow_2d_size1_leading_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(0, 0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_3d_to_size1_inner(self, execution_mode, dtype, consumer):
        """[1,32,256]→narrow(2,0,1)→[1,32,1]. Slice to size-1 innermost; inner changes 256→1. Both consumers."""
        x = cached_randn((1, 32, 256), dtype=dtype,
                         differentiation=f"narrow_3d_to_size1_inner_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(2, 0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_4d_already_size1_dim(self, execution_mode, dtype, consumer):
        """[1,8,1,256]→narrow(2,0,1)→[1,8,1,256]. Narrow the existing size-1 dim; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 8, 1, 256), dtype=dtype,
                         differentiation=f"narrow_4d_already_size1_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(2, 0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_5d_inner_to_size1(self, execution_mode, dtype, consumer):
        """[1,8,2,32,64]→narrow(4,0,1)→[1,8,2,32,1]. 5D; innermost sliced to 1. Both consumers."""
        x = cached_randn((1, 8, 2, 32, 64), dtype=dtype,
                         differentiation=f"narrow_5d_inner_to_size1_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(4, 0, 1), consumer), x)


# ===========================================================================
# §13 — split
# ===========================================================================

# ---------------------------------------------------------------------------
# §13.1 — Gemma CPU-traced shapes (aligned)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSplitCPUTraced:
    """CPU-traced QKV split — no matching module-level test."""

    def test_split_qkv_heads_trailing(self, execution_mode, dtype):
        """[1,32,4096]→split(2048,dim=-1)→([1,32,2048],[1,32,2048]). QKV split; * 1.0 per chunk."""
        x = cached_randn((1, 32, 4096), dtype=dtype,
                         differentiation=f"split_qkv_heads_trailing_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([c * 1.0 for c in t.split(2048, dim=-1)]),
            x,
        )


# ---------------------------------------------------------------------------
# §13.2 — Additional aligned shapes
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSplitAligned:
    """Further aligned shapes covering 2D–5D."""

    def test_split_2d_equal_cols(self, execution_mode, dtype):
        """[32,256]→split(128,dim=1)→([32,128],[32,128]) * 1.0. Inner changes 256→128."""
        x = cached_randn((32, 256), dtype=dtype,
                         differentiation=f"split_2d_equal_cols_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([c * 1.0 for c in t.split(128, dim=1)]),
            x,
        )

    def test_split_3d_seq_dim(self, execution_mode, dtype):
        """[2,64,256]→split(32,dim=1)→[2,32,256] × 2. * 1.0; inner 256 unchanged."""
        x = cached_randn((2, 64, 256), dtype=dtype,
                         differentiation=f"split_3d_seq_dim_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([c * 1.0 for c in t.split(32, dim=1)]),
            x,
        )

    def test_split_4d_head_split(self, execution_mode, dtype):
        """[1,32,16,256]→split(8,dim=2)→[1,32,8,256] × 2. * 1.0; inner 256 unchanged."""
        x = cached_randn((1, 32, 16, 256), dtype=dtype,
                         differentiation=f"split_4d_head_split_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([c * 1.0 for c in t.split(8, dim=2)]),
            x,
        )

    def test_split_5d_rope_half(self, execution_mode, dtype):
        """[1,32,16,2,128]→split(1,dim=3)→[1,32,16,1,128] × 2. * 1.0; inner 128 unchanged."""
        x = cached_randn((1, 32, 16, 2, 128), dtype=dtype,
                         differentiation=f"split_5d_rope_half_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([c * 1.0 for c in t.split(1, dim=3)]),
            x,
        )


# ---------------------------------------------------------------------------
# §13.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSplitUnaligned:
    """2D–5D inputs with unaligned output innermost dimension."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_2d_unaligned_result_trailing(self, execution_mode, dtype, consumer):
        """[8,130]→split(65,dim=-1)→[8,65] × 2. Inner changes 130→65; 65%64=1. Both consumers."""
        if execution_mode == "eager" and (
            consumer == "pointwise"
            or (consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16))
        ):
            pytest.xfail(
                "known issue tracked in https://github.com/torch-spyre/torch-spyre/issues/5383"
            )
        if execution_mode == "eager" and consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32/eager; "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        if execution_mode == "compiled" and consumer == "sum_last":
            pytest.xfail(
                "known failure for compiled/sum_last; "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/3916"
            )
        if execution_mode == "compiled" and consumer == "pointwise":
            pytest.xfail(
                "known failure (no mechanism to resolve stick incompatibility); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5216"
            )
        x = cached_randn((8, 130), dtype=dtype,
                         differentiation=f"split_2d_unaligned_trailing_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(65, dim=-1)]),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_3d_unaligned_trailing(self, execution_mode, dtype, consumer):
        """[4,32,140]→split(70,dim=-1)→[4,32,70] × 2. Inner 70; 70%64=6. Both consumers."""
        if execution_mode == "eager" and (
            consumer == "pointwise"
            or (consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16))
        ):
            pytest.xfail(
                "known issue tracked in https://github.com/torch-spyre/torch-spyre/issues/5383"
            )
        if execution_mode == "eager" and consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32/eager; "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        if execution_mode == "compiled":
            pytest.xfail(
                "known failure (no mechanism to resolve stick incompatibility); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5216"
            )
        x = cached_randn((4, 32, 140), dtype=dtype,
                         differentiation=f"split_3d_unaligned_trailing_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(70, dim=-1)]),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_4d_unaligned_inner(self, execution_mode, dtype, consumer):
        """[1,16,32,130]→split(65,dim=-1)→[1,16,32,65] × 2. Inner 65 unaligned. Both consumers."""
        if execution_mode == "eager" and (
            consumer == "pointwise"
            or (consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16))
        ):
            pytest.xfail(
                "known issue tracked in https://github.com/torch-spyre/torch-spyre/issues/5383"
            )
        if execution_mode == "eager" and consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure for sum_last/float32/eager; "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5224"
            )
        if execution_mode == "compiled":
            pytest.xfail(
                "known failure (no mechanism to resolve stick incompatibility); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5216"
            )
        x = cached_randn((1, 16, 32, 130), dtype=dtype,
                         differentiation=f"split_4d_unaligned_inner_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(65, dim=-1)]),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_5d_unaligned_non_trailing(self, execution_mode, dtype, consumer):
        """[1,2,4,8,256]→split(2,dim=2)→[1,2,2,8,256] × 2. Non-trailing split; inner 256 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16):
            pytest.xfail()
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure (no mechanism to resolve stick incompatibility); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5216"
            )
        x = cached_randn((1, 2, 4, 8, 256), dtype=dtype,
                         differentiation=f"split_5d_unaligned_non_trailing_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(2, dim=2)]),
            x,
        )


# ---------------------------------------------------------------------------
# §13.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSplitSize1:
    """Size-1 dimension interactions."""

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_3d_trailing_size1_result(self, execution_mode, dtype, consumer):
        """[4,32,2]→split(1,dim=-1)→[4,32,1] × 2. Inner changes 2→1. Both consumers."""
        if execution_mode == "eager":
            pytest.xfail(
                "known issue tracked in https://github.com/torch-spyre/torch-spyre/issues/5383"
            )
        if execution_mode == "compiled":
            pytest.xfail(
                "known failure (no supported output layout); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5214"
            )
        x = cached_randn((4, 32, 2), dtype=dtype,
                         differentiation=f"split_3d_trailing_size1_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(1, dim=-1)]),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_4d_size1_dim(self, execution_mode, dtype, consumer):
        """[1,8,32,256]→split(1,dim=0)→([1,8,32,256],). Single chunk. Both consumers."""
        if consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16):
            pytest.xfail()
        x = cached_randn((1, 8, 32, 256), dtype=dtype,
                         differentiation=f"split_4d_size1_dim_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.split(1, dim=0)[0], consumer),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_5d_size1_head_chunks(self, execution_mode, dtype, consumer):
        """[1,8,1,32,256]→split(4,dim=1)→[1,4,1,32,256] × 2. GQA-like chunks; inner 256 unchanged. Both consumers."""
        if consumer == "sum_last" and dtype in (torch.float16, torch.bfloat16):
            pytest.xfail()
        if consumer == "sum_last" and dtype == torch.float32:
            pytest.xfail(
                "known failure (no mechanism to resolve stick incompatibility); "
                "tracked in https://github.com/torch-spyre/torch-spyre/issues/5216"
            )
        x = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"split_5d_size1_head_chunks_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(4, dim=1)]),
            x,
        )
