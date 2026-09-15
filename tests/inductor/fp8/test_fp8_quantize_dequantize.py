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

from __future__ import annotations
import pytest
import torch
from torch_spyre.constants import DEVICE_NAME

E4M3 = getattr(torch, "float8_e4m3fn", None)
FP16 = torch.float16
FP32 = torch.float32
BF16 = torch.bfloat16
FP8_MAX = 448.0
DEVICE = DEVICE_NAME
BACKEND = "inductor"


def _spyre_available() -> bool:
    try:
        import torch_spyre  # noqa: F401

        return torch.spyre.is_available()
    except (ImportError, AttributeError):
        return False


def _quantize_op_available():
    try:
        _ = torch.ops.spyre.quantize_fp8_with_scale
        return True
    except AttributeError:
        return False


def _weight_quantize_op_available():
    try:
        _ = torch.ops.spyre.quantize_weight_fp8_with_scale
        return True
    except AttributeError:
        return False


def _dequantize_op_available():
    try:
        _ = torch.ops.spyre.dequantize_fp8_with_scale
        return True
    except AttributeError:
        return False


pytestmark = pytest.mark.skipif(
    not _spyre_available(), reason="Spyre device not available"
)

skip_no_quantize = pytest.mark.skipif(
    not _quantize_op_available(), reason="quantize_fp8_with_scale not registered"
)
skip_no_weight_quantize = pytest.mark.skipif(
    not _weight_quantize_op_available(),
    reason="quantize_weight_fp8_with_scale not registered",
)
skip_no_dequantize = pytest.mark.skipif(
    not _dequantize_op_available(),
    reason="dequantize_fp8_with_scale not registered",
)

_RAW_QUANTIZE_PARAMS = [
    pytest.param("quantize_fp8_with_scale", id="activation", marks=skip_no_quantize),
    pytest.param(
        "quantize_weight_fp8_with_scale", id="weight", marks=skip_no_weight_quantize
    ),
]

activation_roundtrip_marks = [skip_no_quantize, skip_no_dequantize]


def _quantize_fn(x, scale):
    return torch.ops.spyre.quantize_fp8_with_scale(x, scale)


def _weight_quantize_fn(x, scale):
    return torch.ops.spyre.quantize_weight_fp8_with_scale(x, scale)


def _dequantize_fn(x, scale):
    return torch.ops.spyre.dequantize_fp8_with_scale(x, scale)


def _quant_dequant_fn(x, scale):
    fp8 = torch.ops.spyre.quantize_fp8_with_scale(x, scale)
    return torch.ops.spyre.dequantize_fp8_with_scale(fp8, scale)


def _weight_quant_dequant_fn(x, scale):
    fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(x, scale)
    return torch.ops.spyre.dequantize_fp8_with_scale(fp8, scale)


def _dequantize_ref(x_fp8_cpu, scale_cpu):
    return x_fp8_cpu.to(FP16) * scale_cpu.to(FP16)


def _roundtrip_ref(x_cpu, scale_cpu):
    inv = 1.0 / scale_cpu.to(FP16)
    fp8 = (x_cpu.to(FP16) * inv).clamp(-FP8_MAX, FP8_MAX).to(E4M3)
    return fp8.to(FP16) * scale_cpu.to(FP16)


def _compile(fn, fullgraph=False, dynamic=False):
    return torch.compile(fn, backend=BACKEND, fullgraph=fullgraph, dynamic=dynamic)


# Values 1..16 cycling — all exactly representable in E4M3fn, so atol=0.0 is valid.
def _make_arange_fp16(shape, device=DEVICE):
    n = 1
    for d in shape:
        n *= d
    return ((torch.arange(n, dtype=FP32) % 16) + 1).to(FP16).reshape(shape).to(device)


def _make_arange_fp8(shape, device=DEVICE):
    n = 1
    for d in shape:
        n *= d
    return ((torch.arange(n, dtype=FP32) % 16) + 1).to(E4M3).reshape(shape).to(device)


@pytest.fixture(autouse=True)
def _isolate_compile_state():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


class TestQuantizeFP8:
    """Both quantize ops: output dtype/shape/device, 2D–4D inputs, BF16 input, per-row scale."""

    @pytest.mark.parametrize("op_name", _RAW_QUANTIZE_PARAMS)
    def test_output_dtype_shape_device(self, op_name):
        """Baseline 2D (2,8): output E4M3, shape and Spyre device preserved, contiguous."""
        x = torch.full((2, 8), 2.0, dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        fn = _compile(lambda x, s: getattr(torch.ops.spyre, op_name)(x, s))
        out = fn(x, scale)
        assert out.dtype == E4M3, f"expected E4M3, got {out.dtype}"
        assert out.shape == torch.Size((2, 8)), f"expected (2,8), got {out.shape}"
        assert out.device.type == DEVICE
        assert out.is_contiguous()

    @skip_no_quantize
    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/3023: "
        ".cpu() on activation FP8 output fails with invalid device size/stride map"
    )
    def test_activation_quantize_2d_cpu_copy(self):
        """Activation FP8 (2,8) non-uniform output: .cpu() must preserve logical values."""
        x = _make_arange_fp16((2, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        fn = _compile(_quantize_fn)
        out = fn(x, scale)
        assert out.dtype == E4M3
        assert out.shape == torch.Size((2, 8))
        out_cpu = out.cpu()
        expected = x.cpu().to(FP32).clamp(-FP8_MAX, FP8_MAX).to(E4M3)
        assert out_cpu.dtype == E4M3
        torch.testing.assert_close(
            out_cpu.float(), expected.float(), atol=0.0, rtol=0.0
        )

    @pytest.mark.parametrize("op_name", _RAW_QUANTIZE_PARAMS)
    @pytest.mark.parametrize(
        "shape",
        [
            (4, 8),
            (2, 128),
            (2, 4096),
        ],
        ids=["4x8", "2x128", "2x4096"],
    )
    def test_output_2d_stick_aligned_shapes(self, op_name, shape):
        """Representative 2D stick-aligned shapes: output E4M3, shape and device preserved."""
        x = torch.full(shape, 2.0, dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        fn = _compile(lambda x, s: getattr(torch.ops.spyre, op_name)(x, s))
        out = fn(x, scale)
        assert out.dtype == E4M3
        assert out.shape == torch.Size(shape)
        assert out.device.type == DEVICE
        assert out.is_contiguous()

    @pytest.mark.parametrize("op_name", _RAW_QUANTIZE_PARAMS)
    @pytest.mark.parametrize("shape", [(2, 4), (4, 4)], ids=["2x4", "4x4"])
    def test_output_k4_shapes(self, op_name, shape):
        """K=4 (sub-stick) 2D shapes: output E4M3, shape preserved."""
        x = torch.full(shape, 1.0, dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        fn = _compile(lambda x, s: getattr(torch.ops.spyre, op_name)(x, s))
        out = fn(x, scale)
        assert out.dtype == E4M3
        assert out.shape == torch.Size(shape)

    @pytest.mark.parametrize("op_name", _RAW_QUANTIZE_PARAMS)
    def test_output_per_row_scale(self, op_name):
        """Per-row scale tensor (M,1): output shape (2,8) preserved."""
        x = torch.tensor([[2.0] * 8, [4.0] * 8], dtype=FP16, device=DEVICE)
        scale = torch.tensor([[1.0], [2.0]], dtype=FP16, device=DEVICE)
        fn = _compile(lambda x, s: getattr(torch.ops.spyre, op_name)(x, s))
        out = fn(x, scale)
        assert out.dtype == E4M3
        assert out.shape == torch.Size((2, 8))

    @skip_no_quantize
    @pytest.mark.parametrize(
        "shape",
        [
            (2, 2, 8),
            (4, 128, 512),
            (1, 128, 4096),
        ],
        ids=["2x2x8", "4x128x512", "1x128x4096"],
    )
    def test_output_3d_activation_shapes(self, shape):
        """3D activation shapes: output E4M3, shape preserved."""
        x = torch.full(shape, 1.0, dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quantize_fn)(x, scale)
        assert out.dtype == E4M3
        assert out.shape == torch.Size(shape)
        assert out.device.type == DEVICE

    @skip_no_quantize
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 4, 32, 128),
            pytest.param(
                (2, 8, 64, 64),
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3026: "
                    "3D/4D head_dim=64 fails to compile: compound stick expression not supported"
                ),
            ),
        ],
        ids=["1x4x32x128", "2x8x64x64"],
    )
    def test_output_4d_activation_shapes(self, shape):
        """4D [batch, heads, seq, head_dim] activation shapes."""
        x = torch.full(shape, 1.0, dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quantize_fn)(x, scale)
        assert out.dtype == E4M3
        assert out.shape == torch.Size(shape)
        assert out.device.type == DEVICE

    @skip_no_quantize
    @skip_no_dequantize
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 2, 8),
            (2, 128),
            (4, 128, 512),
        ],
        ids=["1x2x8", "2x128", "4x128x512"],
    )
    def test_activation_scale1_roundtrip(self, shape):
        """Activation quantize(scale=1.0) -> dequantize: element-order preserved."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale)
        ref = x.cpu().clamp(-FP8_MAX, FP8_MAX).to(E4M3).to(FP16)
        torch.testing.assert_close(
            out.cpu(),
            ref,
            atol=0.0,
            rtol=0.0,
            msg=f"shape={shape}: scale=1 roundtrip differs from CPU ref",
        )


class TestFP8Dequantize:
    """dequantize_fp8_with_scale: value correctness, shape coverage, edge values, and sign preservation."""

    pytestmark = skip_no_dequantize

    @pytest.mark.parametrize(
        "shape,scale_val",
        [
            ((2, 8), 1.0),
            ((4, 8), 2.0),
            ((2, 2, 8), 1.0),
        ],
        ids=["2x8_s1", "4x8_s2", "2x2x8_s1"],
    )
    def test_2d_3d_uniform_fill_with_scale(self, shape, scale_val):
        """2D and 3D non-uniform FP8 inputs with varying scale: dequantize against CPU ref."""
        x_fp8 = _make_arange_fp8(shape)
        scale = torch.tensor(scale_val, dtype=FP16, device=DEVICE)
        ref = _dequantize_ref(x_fp8.cpu(), scale.cpu())
        out = _compile(_dequantize_fn)(x_fp8, scale)
        assert out.dtype == FP16
        assert out.shape == torch.Size(shape)
        torch.testing.assert_close(
            out.cpu(),
            ref,
            atol=0.0,
            rtol=0.0,
            msg=f"shape={shape} scale={scale_val}: dequantize differs from CPU ref",
        )

    def test_2d_distinct_row_constants(self):
        """Two rows with distinct per-element values: no row mixing, correct element order."""
        n = 8
        row0 = torch.arange(1, n + 1, dtype=FP32).to(E4M3)
        row1 = torch.arange(n + 1, 2 * n + 1, dtype=FP32).to(E4M3)
        x_fp8_cpu = torch.stack([row0, row1])
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _dequantize_ref(x_fp8_cpu, scale.cpu())
        out = _compile(_dequantize_fn)(x_fp8_cpu.to(DEVICE), scale)
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)
        assert not out.cpu()[0].equal(out.cpu()[1])

    def test_2d_ascending_element_values(self):
        """2D input with power-of-2 ascending values [1,2,4,...,128] in each row."""
        vals = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
        x_fp8_cpu = torch.tensor([vals, vals], dtype=FP32).to(E4M3)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _dequantize_ref(x_fp8_cpu, scale.cpu())
        out = _compile(_dequantize_fn)(x_fp8_cpu.to(DEVICE), scale)
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 8),
            pytest.param(
                (1, 16),
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/2527: "
                    "M=1 shapes crash dxp_standalone with DtException SIGABRT"
                ),
            ),
        ],
        ids=["1x8", "1x16"],
    )
    def test_2d_m1_shapes(self, shape):
        """M=1 non-uniform FP8 input."""
        x_fp8 = _make_arange_fp8(shape)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _dequantize_ref(x_fp8.cpu(), scale.cpu())
        out = _compile(_dequantize_fn)(x_fp8, scale)
        assert out.dtype == FP16
        assert out.shape == torch.Size(shape)
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("M", [3, 5, 7, 8, 16], ids=["M3", "M5", "M7", "M8", "M16"])
    def test_2d_k8_m_sweep(self, M):
        """K=8 with varying M including odd and non-power-of-2 values."""
        x_fp8 = _make_arange_fp8((M, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _dequantize_ref(x_fp8.cpu(), scale.cpu())
        out = _compile(_dequantize_fn)(x_fp8, scale)
        assert out.dtype == FP16
        assert out.shape == torch.Size((M, 8))
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/3033: "
        "dequantize wrong values for K≥16 (all shapes here have K=512-4096)"
    )
    @pytest.mark.parametrize(
        "shape,scale_val",
        [
            ((1, 128, 512), 0.01),
            ((4, 128, 512), 0.1),
            ((1, 128, 1024), 0.5),
            ((1, 128, 2048), 1.0),
            ((1, 128, 4096), 2.0),
        ],
        ids=["1x128x512", "4x128x512", "1x128x1024", "1x128x2048", "1x128x4096"],
    )
    def test_3d_production_shapes(self, shape, scale_val):
        """3D shapes [batch, seq, hidden] with non-uniform input."""
        x_fp8 = _make_arange_fp8(shape)
        scale = torch.tensor(scale_val, dtype=FP16, device=DEVICE)
        ref = _dequantize_ref(x_fp8.cpu(), scale.cpu())
        out = _compile(_dequantize_fn)(x_fp8, scale)
        assert out.dtype == FP16
        assert out.shape == torch.Size(shape)
        assert not out.cpu().isnan().any()
        torch.testing.assert_close(
            out.cpu(),
            ref,
            atol=0.0,
            rtol=0.0,
            msg=f"shape={shape} scale={scale_val}: differs from CPU ref",
        )

    def test_zero_filled_fp8_input(self):
        """FP8(0.0) * scale=1.0: all elements must be FP16(0.0)."""
        x_fp8 = torch.full((2, 8), 0.0, dtype=FP32).to(E4M3).to(DEVICE)
        out = _compile(_dequantize_fn)(
            x_fp8, torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ).cpu()
        assert (out == 0.0).all()

    def test_fp8_max_input_value(self):
        """FP8(448.0) * scale=1.0: no clamp or saturation in dequantize."""
        x_fp8_cpu = torch.full((2, 8), FP8_MAX, dtype=FP32).to(E4M3)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _dequantize_ref(x_fp8_cpu, scale.cpu())
        out = _compile(_dequantize_fn)(x_fp8_cpu.to(DEVICE), scale)
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)

    def test_fp8_min_input_value(self):
        """FP8(-448.0) * scale=1.0: sign and magnitude passed through unchanged."""
        x_fp8_cpu = torch.full((2, 8), -FP8_MAX, dtype=FP32).to(E4M3)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _dequantize_ref(x_fp8_cpu, scale.cpu())
        out = _compile(_dequantize_fn)(x_fp8_cpu.to(DEVICE), scale)
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)

    def test_positive_negative_fp8_input(self):
        """Positive FP8 fill -> positive FP16 output; negative -> negative."""
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        fn = _compile(_dequantize_fn)
        pos = torch.full((2, 8), 2.0, dtype=FP32).to(E4M3).to(DEVICE)
        neg = torch.full((2, 8), -2.0, dtype=FP32).to(E4M3).to(DEVICE)
        assert (fn(pos, scale).cpu() > 0).all()
        assert (fn(neg, scale).cpu() < 0).all()

    def test_2d_fp8_scale_4x(self):
        """2D non-uniform dequantize with scale=4.0."""
        x_fp8 = _make_arange_fp8((2, 8))
        scale = torch.tensor(4.0, dtype=FP16, device=DEVICE)
        ref = _dequantize_ref(x_fp8.cpu(), scale.cpu())
        out = _compile(_dequantize_fn)(x_fp8, scale)
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)

    def test_mixed_sign_row_data(self):
        """Row0 positive ascending, row1 negative descending: sign and element order."""
        x_fp8_cpu = torch.tensor(
            [
                [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
                [-2.0, -4.0, -8.0, -16.0, -32.0, -64.0, -128.0, -256.0],
            ],
            dtype=FP32,
        ).to(E4M3)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _dequantize_ref(x_fp8_cpu, scale.cpu())
        out = _compile(_dequantize_fn)(x_fp8_cpu.to(DEVICE), scale)
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)
        assert (out.cpu()[0] > 0).all() and (out.cpu()[1] < 0).all()

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/3027: "
        "hardware FP8→FP16 decodes ~6.25% of E4M3fn bit patterns incorrectly"
    )
    def test_all_e4m3_bit_patterns(self):
        """All 256 E4M3 bit patterns must convert to the corresponding FP16 values."""
        raw = torch.arange(256, dtype=torch.uint8)
        x_fp8_cpu = raw.view(E4M3).reshape(32, 8)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _dequantize_ref(x_fp8_cpu, scale.cpu())
        out = _compile(_dequantize_fn)(x_fp8_cpu.to(DEVICE), scale)
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0, equal_nan=True)

    def test_shape_guard_recompile(self):
        """(2,8) then (4,8): shape guard must recompile, correct output for both shapes."""
        from torch._dynamo.testing import CompileCounterWithBackend

        counter = CompileCounterWithBackend(BACKEND)
        fn = torch.compile(_dequantize_fn, backend=counter, fullgraph=True)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        x2 = _make_arange_fp8((2, 8))
        ref2 = _dequantize_ref(x2.cpu(), scale.cpu())
        torch.testing.assert_close(fn(x2, scale).cpu(), ref2, atol=0.0, rtol=0.0)
        x4 = _make_arange_fp8((4, 8))
        ref4 = _dequantize_ref(x4.cpu(), scale.cpu())
        torch.testing.assert_close(fn(x4, scale).cpu(), ref4, atol=0.0, rtol=0.0)
        assert counter.frame_count == 2, (
            f"expected 2 specialized graphs (dynamic=False, two distinct shapes), "
            f"got {counter.frame_count}"
        )

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/2526: "
        "non-contiguous FP8 input fails to compile (no stick-incompatibility resolution)"
    )
    def test_transposed_fp8_input(self):
        """Transposed non-contiguous FP8 input via .t()."""
        base = _make_arange_fp8((8, 2))
        x_t = base.t()
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _dequantize_ref(base.cpu().t().contiguous(), scale.cpu())
        out = _compile(_dequantize_fn)(x_t, scale)
        assert out.dtype == FP16
        assert out.shape == torch.Size((2, 8))
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)


class TestWeightRoundtrip:
    """quantize_weight_fp8_with_scale → dequantize roundtrip: stick-aligned K, K=8, scale arithmetic."""

    pytestmark = [
        skip_no_weight_quantize,
        skip_no_dequantize,
        pytest.mark.skip(
            reason="https://github.com/torch-spyre/torch-spyre/issues/3022: "
            "fused weight quant/dequant crashes dxp_standalone compiler"
        ),
    ]

    @pytest.mark.parametrize(
        "shape,scale_val",
        [
            ((2, 128), 1.0),
            ((16, 128), 1.0),
            ((2, 4096), 1.0),
        ],
        ids=["2x128", "16x128", "2x4096"],
    )
    def test_2d_uniform_stick_aligned(self, shape, scale_val):
        """Stick-aligned K (>=128), non-uniform input: weight roundtrip must match CPU ref."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(scale_val, dtype=FP16, device=DEVICE)
        ref = _roundtrip_ref(x.cpu(), scale.cpu())
        out = _compile(_weight_quant_dequant_fn)(x, scale)
        torch.testing.assert_close(
            out.cpu(),
            ref,
            atol=0.0,
            rtol=0.0,
            msg=f"weight roundtrip shape={shape}: differs from CPU ref",
        )

    def test_2d_uniform_scale_arithmetic(self):
        """(2,128) non-uniform input, scale=2.0: scale arithmetic applied per element."""
        x = _make_arange_fp16((2, 128))
        scale = torch.tensor(2.0, dtype=FP16, device=DEVICE)
        ref = _roundtrip_ref(x.cpu(), scale.cpu())
        out = _compile(_weight_quant_dequant_fn)(x, scale)
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("M", [16, 32, 64], ids=["M16", "M32", "M64"])
    def test_2d_k8_large_m_uniform(self, M):
        """K=8, M >= rows_per_stick(16), non-uniform input."""
        x = _make_arange_fp16((M, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _roundtrip_ref(x.cpu(), scale.cpu())
        out = _compile(_weight_quant_dequant_fn)(x, scale)
        torch.testing.assert_close(
            out.cpu(),
            ref,
            atol=0.0,
            rtol=0.0,
            msg=f"weight K=8 M={M} non-uniform: differs from CPU ref",
        )

    @pytest.mark.parametrize(
        "shape", [(2, 8), (4, 8), (8, 8)], ids=["2x8", "4x8", "8x8"]
    )
    def test_2d_k8_small_m(self, shape):
        """K=8, M < rows_per_stick(16), non-uniform input."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _roundtrip_ref(x.cpu(), scale.cpu())
        out = _compile(_weight_quant_dequant_fn)(x, scale)
        torch.testing.assert_close(
            out.cpu(),
            ref,
            atol=0.0,
            rtol=0.0,
            msg=f"weight K=8 shape={shape}: differs from CPU ref",
        )

    def test_2d_nonuniform_data(self):
        """Non-uniform (2,128) rows: weight roundtrip must preserve per-element values."""
        vals0 = [float(i % 16 + 1) for i in range(128)]
        vals1 = [float(16 - i % 16) for i in range(128)]
        x = torch.tensor([vals0, vals1], dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _roundtrip_ref(x.cpu(), scale.cpu())
        out = _compile(_weight_quant_dequant_fn)(x, scale)
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)


class TestRoundtripBaseline:
    """Activation quantize → dequantize baseline: 2D/3D fill, distinct rows, and determinism."""

    pytestmark = activation_roundtrip_marks

    @pytest.mark.parametrize(
        "shape,scale_val",
        [
            ((2, 8), 1.0),
            ((4, 8), 2.0),
            ((8, 8), 1.0),
            ((2, 2, 8), 2.0),
        ],
        ids=["2x8_s1", "4x8_s2", "8x8_s1", "2x2x8_s2"],
    )
    def test_2d_3d_uniform_fill_with_scale(self, shape, scale_val):
        """2D and 3D non-uniform inputs with varying scale: roundtrip against CPU ref."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(scale_val, dtype=FP16, device=DEVICE)
        ref = _roundtrip_ref(x.cpu(), scale.cpu())
        out = _compile(_quant_dequant_fn)(x, scale)
        assert out.dtype == FP16
        assert out.shape == torch.Size(shape)
        torch.testing.assert_close(
            out.cpu(),
            ref,
            atol=0.0,
            rtol=0.0,
            msg=f"shape={shape} scale={scale_val}: roundtrip differs from CPU ref",
        )

    def test_3d_scalar_scale_broadcast(self):
        """3D non-uniform input (2,2,8) with scalar scale broadcast via (1,1,1) tensor."""
        x = _make_arange_fp16((2, 2, 8))
        scale = torch.tensor([[[2.0]]], dtype=FP16, device=DEVICE)
        ref = _roundtrip_ref(x.cpu(), torch.tensor(2.0, dtype=FP16))
        out = _compile(_quant_dequant_fn)(x, scale)
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)

    def test_2d_distinct_row_constants(self):
        """Row0 values 1..8, row1 values 9..16: each row distinct, no row mixing."""
        row0 = torch.arange(1, 9, dtype=FP16)
        row1 = torch.arange(9, 17, dtype=FP16)
        x = torch.stack([row0, row1]).to(DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _roundtrip_ref(x.cpu(), scale.cpu())
        out = _compile(_quant_dequant_fn)(x, scale)
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)
        assert not out.cpu()[0].equal(out.cpu()[1])

    def test_2d_ascending_element_values(self):
        """Per-element ascending [1,2,4,...,128]: element-order preserved through roundtrip."""
        vals = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
        x = torch.tensor([vals, vals], dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "seed,shape",
        [(42, (2, 8)), (0, (4, 8)), (7, (2, 2, 8))],
        ids=["seed42_2x8", "seed0_4x8", "seed7_2x2x8"],
    )
    def test_seeded_random_fp16_input(self, seed, shape):
        """Random FP16 inputs with fixed seeds: catches element-specific bit-pattern bugs."""
        torch.manual_seed(seed)
        x = (torch.rand(shape) * 10.0 - 5.0).to(FP16).to(DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
            msg=f"random inputs seed={seed} shape={shape}: roundtrip differs from CPU ref",
        )

    def test_monotonic_uniform_input_magnitudes(self):
        """Three ascending fills 2.0 < 4.0 < 8.0: roundtrip output mean is monotonic."""
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        fn = _compile(_quant_dequant_fn)
        out2 = fn(torch.full((2, 8), 2.0, dtype=FP16, device=DEVICE), scale).cpu()
        out4 = fn(torch.full((2, 8), 4.0, dtype=FP16, device=DEVICE), scale).cpu()
        out8 = fn(torch.full((2, 8), 8.0, dtype=FP16, device=DEVICE), scale).cpu()
        assert out2.mean() < out4.mean() < out8.mean()

    def test_two_calls_same_input(self):
        """Non-uniform input called twice on same compiled fn: outputs byte-identical."""
        x = _make_arange_fp16((2, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        fn = _compile(_quant_dequant_fn)
        assert fn(x, scale).cpu().equal(fn(x, scale).cpu())


class TestRoundtripShapes:
    """Activation roundtrip shape sweep: M=1, K=4/8/16, K-unaligned, 3D/4D, LLaMA/matrix shapes."""

    pytestmark = activation_roundtrip_marks

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/2527: "
        "M=1 shapes crash dxp_standalone with DtException SIGABRT"
    )
    @pytest.mark.parametrize(
        "shape", [(1, 8), (1, 16), (1, 32)], ids=["1x8", "1x16", "1x32"]
    )
    def test_2d_m1_shapes(self, shape):
        """M=1 2D shapes, non-uniform input."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale)
        assert out.dtype == FP16
        assert out.shape == torch.Size(shape)
        torch.testing.assert_close(
            out.cpu(), _roundtrip_ref(x.cpu(), scale.cpu()), atol=0.0, rtol=0.0
        )

    @pytest.mark.parametrize(
        "shape", [(2, 4), (4, 4), (2, 2, 4)], ids=["2x4", "4x4", "2x2x4"]
    )
    def test_2d_k4_shapes(self, shape):
        """K=4 2D and 3D shapes, non-uniform input."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 16),
            (4, 32),
            (4, 128),
            (2, 64),
            (4, 256),
            (2, 4096),
            (4, 4096),
        ],
        ids=["2x16", "4x32", "4x128", "2x64", "4x256", "2x4096", "4x4096"],
    )
    def test_2d_varying_k(self, shape):
        """2D shapes with varying K from 16 to 4096, non-uniform input."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale).cpu()
        assert not out.isnan().any()
        torch.testing.assert_close(
            out,
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
            msg=f"shape={shape}",
        )

    def test_2d_k16_all_columns(self):
        """(2,16) arange values 1..16: all 16 K-dimension columns must appear in output."""
        values = torch.arange(1, 17, dtype=FP16)
        x = torch.stack((values, values)).to(DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale).cpu()
        assert not out.isnan().any()
        torch.testing.assert_close(
            out, _roundtrip_ref(x.cpu(), scale.cpu()), atol=0.0, rtol=0.0
        )

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4312: "
        "spyre_fill_tensor raises RuntimeError for numel()==0"
    )
    def test_2d_zero_batch(self):
        """Zero-sized first dimension (0,8): must return empty tensor, not crash."""
        x = torch.zeros(0, 8, dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale)
        assert out.shape == torch.Size((0, 8))
        assert out.dtype == FP16

    @pytest.mark.parametrize(
        "M",
        [
            16,
            32,
            64,
            128,
            pytest.param(
                256,
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3029: "
                    "K=8 activation roundtrip wrong values for M≥256"
                ),
            ),
        ],
        ids=["M16", "M32", "M64", "M128", "M256"],
    )
    def test_2d_k8_large_pow2_m(self, M):
        """K=8 large power-of-2 M values, non-uniform input."""
        x = _make_arange_fp16((M, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
            msg=f"M={M} K=8 non-uniform: roundtrip differs",
        )

    @pytest.mark.parametrize("M", [3, 5, 7], ids=["M3", "M5", "M7"])
    def test_2d_k8_odd_m(self, M):
        """K=8 odd M values, non-uniform input."""
        x = _make_arange_fp16((M, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize("M", [6, 10, 12], ids=["M6", "M10", "M12"])
    def test_2d_k8_non_pow2_m(self, M):
        """K=8 non-power-of-2 M values, non-uniform input."""
        x = _make_arange_fp16((M, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/3029: "
        "K=8 activation roundtrip wrong values for M≥256"
    )
    @pytest.mark.parametrize("M", [512, 1024], ids=["M512", "M1024"])
    def test_2d_k8_very_large_m(self, M):
        """K=8 very large M, non-uniform input."""
        x = _make_arange_fp16((M, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape,scale_val",
        [
            ((2048, 4096), 1.0),
            ((4096, 4096), 1.0),
            pytest.param(
                (1, 4096),
                1.0,
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/2527: "
                    "M=1 shapes crash dxp_standalone with DtException SIGABRT"
                ),
            ),
        ],
        ids=["2048x4096", "4096x4096", "1x4096"],
    )
    def test_2d_llama_shapes(self, shape, scale_val):
        """LLaMA-scale 2D shapes, non-uniform input."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(scale_val, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape,scale_val",
        [
            ((128, 128), 0.5),
            ((4096, 1024), 2.0),
        ],
        ids=["128x128", "4096x1024"],
    )
    def test_2d_matrix_shaped_activations(self, shape, scale_val):
        """Large 2D matrix shapes. Power-of-2 scales only (FP16 reciprocal is exact)."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(scale_val, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale).cpu()
        assert not out.isnan().any()
        torch.testing.assert_close(
            out, _roundtrip_ref(x.cpu(), scale.cpu()), atol=0.0, rtol=0.0
        )

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/2527: "
        "M=1 shapes crash dxp_standalone with DtException SIGABRT"
    )
    def test_3d_shape_2x16x8(self):
        """3D (2,16,8) non-uniform input."""
        x = _make_arange_fp16((2, 16, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale)
        torch.testing.assert_close(
            out.cpu(), _roundtrip_ref(x.cpu(), scale.cpu()), atol=0.0, rtol=0.0
        )

    @pytest.mark.parametrize(
        "S",
        [
            pytest.param(
                17,
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3029: "
                    "3D (2,17,8) fails in pytest (8/272 wrong at row 16) but passes "
                    "in standalone repro; possibly compile-ordering-dependent"
                ),
            ),
            pytest.param(
                32,
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3029: "
                    "3D (2,32,8) K=8 roundtrip: 256/512 elements wrong (50%), finite values"
                ),
            ),
            pytest.param(
                64,
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3029: "
                    "3D (2,64,8) K=8 roundtrip: 256/1024 elements wrong (25%), finite values"
                ),
            ),
            128,
            pytest.param(
                512,
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3029: "
                    "3D (2,512,8) K=8 roundtrip: expected 256 wrong elements, same DDL addressing bug"
                ),
            ),
        ],
        ids=["S17", "S32", "S64", "S128", "S512"],
    )
    def test_3d_shapes_varying_seq_k8(self, S):
        """3D (2,S,8) non-uniform input."""
        x = _make_arange_fp16((2, S, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param(
                (2, 128, 64),
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3026: "
                    "3D/4D head_dim=64 fails to compile: compound stick expression not supported"
                ),
            ),
            (4, 64, 256),
            (2, 32, 4096),
            (1, 512, 4096),
        ],
        ids=["2x128x64", "4x64x256", "2x32x4096", "1x512x4096"],
    )
    def test_3d_production_shapes(self, shape):
        """3D activation shapes, non-uniform input."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param(
                (63, 8),
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3029: "
                    "K=8 activation roundtrip wrong values (value mismatch)"
                ),
            ),
            pytest.param(
                (63, 13),
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3024: "
                    "stick-unaligned K causes compile crash (unsupported coordinate expression)"
                ),
            ),
            (63, 129),
            pytest.param(
                (17, 13),
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3024: "
                    "stick-unaligned K causes compile crash (unsupported coordinate expression)"
                ),
            ),
            pytest.param(
                (100, 50),
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3024: "
                    "stick-unaligned K causes compile crash (unsupported coordinate expression)"
                ),
            ),
            (7, 77),
            (67, 256),
        ],
        ids=["63x8", "63x13", "63x129", "17x13", "100x50", "7x77", "67x256"],
    )
    def test_2d_stick_unaligned_shapes(self, shape):
        """2D stick-unaligned shapes, non-uniform input."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param(
                (3, 7, 9),
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3024: "
                    "stick-unaligned K causes compile crash (unsupported coordinate expression)"
                ),
            ),
            (67, 71, 77),
            (2, 63, 128),
        ],
        ids=["3x7x9", "67x71x77", "2x63x128"],
    )
    def test_3d_stick_unaligned_shapes(self, shape):
        """3D stick-unaligned shapes, non-uniform input."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param(
                (2, 8, 64, 64),
                marks=pytest.mark.skip(
                    reason="https://github.com/torch-spyre/torch-spyre/issues/3026: "
                    "3D/4D head_dim=64 fails to compile: compound stick expression not supported"
                ),
            ),
            (1, 4, 32, 128),
        ],
        ids=["2x8x64x64", "1x4x32x128"],
    )
    def test_4d_attention_layout(self, shape):
        """4D [batch, heads, seq_len, head_dim] non-uniform input."""
        x = _make_arange_fp16(shape)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )


class TestRoundtripScale:
    """Activation roundtrip scale coverage: scalar range, zero/inf/negative, and per-row scales."""

    pytestmark = activation_roundtrip_marks

    # Power-of-2 scales only: non-power-of-2 (e.g. 0.1) have inexact FP16 reciprocals,
    # causing ±1 ULP differences at FP8 boundaries that make atol=0.0 invalid.
    @pytest.mark.parametrize(
        "scale_val", [0.5, 1.0, 2.0, 4.0, 8.0], ids=["s0.5", "s1", "s2", "s4", "s8"]
    )
    def test_scalar_scale_range(self, scale_val):
        """Scalar power-of-2 scale sweep, non-uniform input: roundtrip matches CPU ref."""
        x = _make_arange_fp16((2, 8))
        scale = torch.tensor(scale_val, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale)
        assert not out.cpu().isnan().any()
        torch.testing.assert_close(
            out.cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
            msg=f"scale={scale_val}: roundtrip differs",
        )

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/3037: "
        "scale=0 produces +inf output instead of 0"
    )
    def test_zero_scale(self):
        """scale=0: output must be all zero."""
        x = torch.full((2, 8), 1.0, dtype=FP16, device=DEVICE)
        scale = torch.tensor(0.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale).cpu()
        assert not out.isnan().any()
        assert (out == 0.0).all()

    def test_inf_scale(self):
        """scale=Inf: x/Inf -> 0 -> FP8(0) -> 0. Must not produce NaN."""
        x = _make_arange_fp16((2, 8))
        out = _compile(_quant_dequant_fn)(
            x, torch.tensor(float("inf"), dtype=FP16, device=DEVICE)
        ).cpu()
        assert (out == 0.0).all()

    def test_negative_scale(self):
        """Negative scale inverts sign: non-uniform input, scale=-1.0."""
        x = _make_arange_fp16((2, 8))
        scale = torch.tensor(-1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    def test_distinct_scale_values(self):
        """x=4.3 with scale=1.0 vs scale=1.5: lands on different FP8 grid points."""
        x = torch.full((2, 8), 4.3, dtype=FP16, device=DEVICE)
        fn = _compile(_quant_dequant_fn)
        o1 = fn(x, torch.tensor(1.0, dtype=FP16, device=DEVICE)).cpu()
        o2 = fn(x, torch.tensor(1.5, dtype=FP16, device=DEVICE)).cpu()
        assert not o1.equal(o2)

    def test_per_row_scale_2d(self):
        """Per-row scale tensor (M,1): each row applies its own scale."""
        row0 = torch.arange(1, 9, dtype=FP16)
        row1 = torch.arange(9, 17, dtype=FP16)
        x = torch.stack([row0, row1]).to(DEVICE)
        scale = torch.tensor([[1.0], [2.0]], dtype=FP16, device=DEVICE)
        ref = _roundtrip_ref(x.cpu(), scale.cpu())
        out = _compile(_quant_dequant_fn)(x, scale)
        assert out.dtype == FP16
        torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)

    def test_per_row_distinct_scale_values(self):
        """Per-row scale (1.0 vs 1.5) on x=4.3: rows must differ in output."""
        x = torch.full((2, 8), 4.3, dtype=FP16, device=DEVICE)
        scale = torch.tensor([[1.0], [1.5]], dtype=FP16, device=DEVICE)
        ref = _roundtrip_ref(x.cpu(), scale.cpu())
        out = _compile(_quant_dequant_fn)(x, scale).cpu()
        torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)
        assert not out[0].equal(out[1])


class TestRoundtripEdgeCases:
    """Activation roundtrip edge values: zero fill, FP8 max/min, clamp, inf, NaN, sign."""

    pytestmark = activation_roundtrip_marks

    def test_zero_filled_input(self):
        """All-zero input: roundtrip must produce all-zero output."""
        x = torch.zeros(2, 8, dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale).cpu()
        assert (out == 0.0).all()

    def test_above_fp8_max_input(self):
        """Input 1000.0 > FP8_MAX=448.0: clamped to 448.0 before FP8 cast."""
        x = torch.full((2, 8), 1000.0, dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    def test_below_fp8_min_input(self):
        """Input -1000.0 < FP8_MIN=-448.0: clamped to -448.0 before FP8 cast."""
        x = torch.full((2, 8), -1000.0, dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    def test_fp8_max_value_input(self):
        """Exact FP8_MAX=448.0 input: no clamping applied, roundtrip exact."""
        x = torch.full((2, 8), FP8_MAX, dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    def test_near_zero_input(self):
        """Near-zero 1e-4: underflows to FP8(0). No NaN or crash."""
        x = torch.full((2, 8), 1e-4, dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale).cpu()
        assert not out.isnan().any()
        torch.testing.assert_close(
            out, _roundtrip_ref(x.cpu(), scale.cpu()), atol=0.0, rtol=0.0
        )

    def test_positive_negative_input(self):
        """Positive values stay positive, negative stay negative through roundtrip."""
        x_pos = _make_arange_fp16((2, 8))
        x_neg = -_make_arange_fp16((2, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        fn = _compile(_quant_dequant_fn)
        assert (fn(x_pos, scale).cpu() > 0).all()
        assert (fn(x_neg, scale).cpu() < 0).all()

    def test_mixed_sign_row_data(self):
        """Row0 positive ascending, row1 negative: per-row sign preserved."""
        x = torch.tensor(
            [
                [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
                [-2.0, -4.0, -8.0, -16.0, -32.0, -64.0, -128.0, -256.0],
            ],
            dtype=FP16,
            device=DEVICE,
        )
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale).cpu()
        torch.testing.assert_close(
            out, _roundtrip_ref(x.cpu(), scale.cpu()), atol=0.0, rtol=0.0
        )
        assert (out[0] > 0).all() and (out[1] < 0).all()

    def test_fp16_subnormal_input(self):
        """FP16 subnormal (1e-7): underflows to FP8(0). No NaN."""
        x = torch.full((2, 8), 1e-7, dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale).cpu()
        assert not out.isnan().any()
        torch.testing.assert_close(
            out, _roundtrip_ref(x.cpu(), scale.cpu()), atol=0.0, rtol=0.0
        )

    def test_positive_inf_input(self):
        """+Inf input: clamped to FP8_MAX. Output must not be NaN or Inf."""
        x = torch.full((2, 8), float("inf"), dtype=FP16, device=DEVICE)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ref = _roundtrip_ref(x.cpu(), scale.cpu())
        out = _compile(_quant_dequant_fn)(x, scale).cpu()
        assert not out.isnan().any()
        assert not out.isinf().any()
        torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)

    def test_nan_input(self):
        """NaN input: op must not crash. Output shape must be preserved."""
        x = torch.full((2, 8), float("nan"), dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(
            x, torch.tensor(1.0, dtype=FP16, device=DEVICE)
        ).cpu()
        assert out.shape == torch.Size((2, 8))


class TestRoundtripInputVariants:
    """Activation roundtrip with non-standard inputs: non-contiguous (transposed) and BF16."""

    pytestmark = activation_roundtrip_marks

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/2526: "
        "non-contiguous input fails to compile (no stick-incompatibility resolution)"
    )
    def test_transposed_non_contiguous_input(self):
        """Transposed non-contiguous input, non-uniform."""
        base = _make_arange_fp16((8, 2))
        x = base.t()
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn)(x, scale)
        assert out.dtype == FP16
        torch.testing.assert_close(
            out.cpu(),
            _roundtrip_ref(base.cpu().t().contiguous(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )

    def test_ascending_values_with_scale(self):
        """Ascending per-element values [0.5..4.0] with scale=2.0."""
        vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        x = torch.tensor([vals, vals], dtype=FP16, device=DEVICE)
        scale = torch.tensor(2.0, dtype=FP16, device=DEVICE)
        torch.testing.assert_close(
            _compile(_quant_dequant_fn)(x, scale).cpu(),
            _roundtrip_ref(x.cpu(), scale.cpu()),
            atol=0.0,
            rtol=0.0,
        )


class TestRoundtripGraphBehavior:
    """Activation roundtrip compilation: fullgraph=True, repeated-call determinism, shape guard recompile."""

    pytestmark = activation_roundtrip_marks

    def test_quantize_dequantize_fullgraph(self):
        """Both ops in a single fullgraph compile without graph break."""
        x = _make_arange_fp16((2, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        out = _compile(_quant_dequant_fn, fullgraph=True)(x, scale)
        torch.testing.assert_close(
            out.cpu(), _roundtrip_ref(x.cpu(), scale.cpu()), atol=0.0, rtol=0.0
        )

    def test_three_calls_same_input(self):
        """Non-uniform input called three times: output byte-identical across all calls."""
        x = _make_arange_fp16((2, 8))
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        fn = _compile(_quant_dequant_fn)
        outs = [fn(x, scale).cpu() for _ in range(3)]
        assert outs[0].equal(outs[1]) and outs[0].equal(outs[2])

    def test_shape_guard_recompile(self):
        """(2,8) then (3,8): shape guard must recompile for correct row-3 output."""
        fn = _compile(_quant_dequant_fn, fullgraph=True)
        scale = torch.tensor(1.0, dtype=FP16, device=DEVICE)
        x2 = _make_arange_fp16((2, 8))
        fn(x2, scale)
        x3 = _make_arange_fp16((3, 8))
        out3 = fn(x3, scale)
        ref3 = _roundtrip_ref(x3.cpu(), scale.cpu())
        torch.testing.assert_close(
            out3.cpu(),
            ref3,
            atol=0.0,
            rtol=0.0,
            msg="(3,8) after (2,8) without reset: shape guard must trigger recompile",
        )


class TestFP8EagerMode:
    """quantize_fp8_with_scale and quantize_weight_fp8_with_scale in eager mode (no torch.compile)."""

    def setup_method(self):
        torch._dynamo.reset()

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4313: quantize ops return None in eager mode"
    )
    @pytest.mark.skipif(
        not _quantize_op_available(), reason="quantize_fp8_with_scale not registered"
    )
    def test_quantize_fp8_with_scale_eager(self):
        """quantize_fp8_with_scale eager: output shape, dtype, and values match CPU oracle."""
        M, K = 16, 128
        torch.manual_seed(50)
        a_cpu = torch.rand(M, K, dtype=FP16)
        a = a_cpu.to(DEVICE)
        sa = torch.tensor(1.0, dtype=FP16, device=DEVICE)

        a_fp8 = torch.ops.spyre.quantize_fp8_with_scale(a, sa)

        assert a_fp8.shape == (M, K)
        assert a_fp8.dtype == E4M3
        assert a_fp8.device.type == DEVICE
        ref = a_cpu.float().clamp(-FP8_MAX, FP8_MAX).to(E4M3)
        torch.testing.assert_close(a_fp8.cpu().float(), ref.float(), atol=0.0, rtol=0.0)

    @pytest.mark.skip(
        reason="https://github.com/torch-spyre/torch-spyre/issues/4313: quantize ops return None in eager mode"
    )
    @pytest.mark.skipif(
        not _weight_quantize_op_available(),
        reason="quantize_weight_fp8_with_scale not registered",
    )
    def test_quantize_weight_fp8_with_scale_eager(self):
        """quantize_weight_fp8_with_scale eager: output shape, dtype, and values match CPU oracle."""
        K, N = 128, 128
        torch.manual_seed(51)
        w_cpu = torch.rand(K, N, dtype=FP16)
        w = w_cpu.to(DEVICE)
        sw = torch.tensor(1.0, dtype=FP16, device=DEVICE)

        w_fp8 = torch.ops.spyre.quantize_weight_fp8_with_scale(w, sw)

        assert w_fp8.shape == (K, N)
        assert w_fp8.dtype == E4M3
        assert w_fp8.device.type == DEVICE
        ref = w_cpu.float().clamp(-FP8_MAX, FP8_MAX).to(E4M3)
        torch.testing.assert_close(w_fp8.cpu().float(), ref.float(), atol=0.0, rtol=0.0)
