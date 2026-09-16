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

"""
Test bidirectional FP16↔FP32 type conversion with ElementArrangement.

Tests in mode: compile, eager
FP16 types: FP16, BF16
Tests all 4 conversion cases:
1. FP16→FP32 with STANDARD → DL16_TO_FP32
2. FP16→FP32 with FP32_TO_DL16 → STANDARD
3. FP32→FP16 with STANDARD → FP32_TO_DL16
4. FP32→FP16 with DL16_TO_FP32 → STANDARD
"""

import pytest
import torch
from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout
from torch_spyre._inductor.dtype_ops import DtypeOpTable
from torch_spyre._inductor.constants import DEVICE_NAME


def _run(fn, *args, mode="compile"):
    if mode == "compile":
        fn = torch.compile(fn)
    return fn(*args)


def get_ea(tensor):
    if tensor.device.type != DEVICE_NAME:
        return None
    try:
        layout = get_spyre_tensor_layout(tensor)
        return layout.element_arrangement if layout else None
    except RuntimeError:
        return None


def assert_ea(tensor, expected_ea):
    actual_ea = get_ea(tensor)
    assert actual_ea == expected_ea, f"Expected: {expected_ea}, Got: {actual_ea}"


def assert_val(fn, x, result):
    x_cpu = x.cpu()
    result_cpu = fn(x_cpu)
    torch.testing.assert_close(result.cpu(), result_cpu, rtol=1e-3, atol=1e-3)


def ea_of(dev):
    return ElementArrangement.STANDARD if dev == DEVICE_NAME else None


_TEST_CASES = [
    (device, mode, fp16)
    for device in [DEVICE_NAME]
    for mode in ["compile", "eager"]
    for fp16 in DtypeOpTable.fp16_types()
]

_TEST_IDS = [
    f"{device}-{mode}-{fp16}".replace("torch.", "")
    for device, mode, fp16 in _TEST_CASES
]


@pytest.mark.parametrize(
    "device, mode, fp16",
    _TEST_CASES,
    ids=_TEST_IDS,
)
def test_fp16_to_fp32_standard_input(device, mode, fp16):
    """Test FP16/BF16→FP32 with STANDARD input creates DL16_TO_FP32 (#2843)."""

    def fn(x):
        return x.to(torch.float32)

    x = torch.randn(4, 128, device=device, dtype=fp16)
    result = _run(fn, x, mode=mode)

    # Verify output EA
    assert_ea(result, ElementArrangement.DL16_TO_FP32)

    # Note: Cannot compare tensors with non-STANDARD EA directly with CPU
    # The result has DL16_TO_FP32 EA which differs from CPU's STANDARD EA

    print(f"✓ {fp16}→FP32 with STANDARD input produces DL16_TO_FP32 ({mode})")


@pytest.mark.parametrize("device", ["spyre"])
@pytest.mark.parametrize("mode", ["compile", "eager"])
def test_mixed_bf16_fp32_add_rejects_ea_mismatch(device, mode):
    """A bf16/fp32 add must raise on EA mismatch instead of silently executing (#2843)."""

    def fn(x, y):
        return torch.add(x, y)

    x_fp32 = torch.randn(5120, dtype=torch.float32, device=device)
    y_bf16 = torch.randn(5120, dtype=torch.bfloat16, device=device)

    with pytest.raises(Exception, match="element arrangement|EA"):
        _run(fn, x_fp32, y_bf16)


@pytest.mark.parametrize(
    "device, mode, fp16",
    _TEST_CASES,
    ids=_TEST_IDS,
)
def test_fp32_to_fp16_standard_input(device, mode, fp16):
    """Test FP32→FP16 with STANDARD input creates FP32_TO_DL16."""

    def fn(x):
        return x.to(dtype=fp16)

    x = torch.randn(4, 128, device=device, dtype=torch.float32)
    result = _run(fn, x, mode=mode)

    # Eager and compiled casts use the same compiled D2D conversion path, so
    # both preserve the hardware conversion's staggered element arrangement.
    assert_ea(result, ElementArrangement.FP32_TO_DL16)

    # Note: Cannot compare tensors with non-STANDARD EA directly with CPU
    # The result has FP32_TO_DL16 EA which differs from CPU's STANDARD EA

    print("✓ FP32→{fp16} with STANDARD input produces FP32_TO_DL16 ({mode})")


@pytest.mark.parametrize("device", ["spyre"])
@pytest.mark.parametrize(
    "fp16",
    DtypeOpTable.fp16_types(),
    ids=lambda dt: str(dt).replace("torch.", ""),
)
def test_fp16_to_fp32_restoration(device, fp16):
    """Test FP16→FP32 with FP32_TO_DL16 input restores to STANDARD."""

    @torch.compile
    def fn(x):
        # FP32 → FP16 (creates FP32_TO_DL16)
        x_fp16 = x.to(dtype=fp16)
        # FP16 → FP32 (should restore to STANDARD)
        return x_fp16.to(torch.float32)

    x = torch.randn(4, 128, device=device, dtype=torch.float32)
    result = fn(x)

    # Verify output EA is STANDARD (restored)
    assert_ea(result, ElementArrangement.STANDARD)

    # Verify correctness
    assert_val(fn, x, result)

    print("✓ FP16→FP32 restoration (FP32_TO_DL16 → STANDARD) works")


@pytest.mark.parametrize("device", ["spyre"])
@pytest.mark.parametrize(
    "fp16",
    DtypeOpTable.fp16_types(),
    ids=lambda dt: str(dt).replace("torch.", ""),
)
def test_fp32_to_fp16_restoration(device, fp16):
    """Test FP32→FP16 with DL16_TO_FP32 input restores to STANDARD."""

    @torch.compile
    def fn(x):
        # FP16 → FP32 (creates DL16_TO_FP32)
        x_fp32 = x.to(torch.float32)
        # FP32 → FP16 (should restore to STANDARD)
        return x_fp32.to(dtype=fp16)

    x = torch.randn(4, 128, device=device, dtype=fp16)
    result = fn(x)

    # Verify output EA is STANDARD (restored)
    assert_ea(result, ElementArrangement.STANDARD)

    # Verify correctness
    assert_val(fn, x, result)

    print("✓ FP32→FP16 restoration (DL16_TO_FP32 → STANDARD) works")


@pytest.mark.parametrize("device", ["spyre"])
@pytest.mark.parametrize(
    "fp16",
    DtypeOpTable.fp16_types(),
    ids=lambda dt: str(dt).replace("torch.", ""),
)
def test_bidirectional_roundtrip_fp16_start(device, fp16):
    """Test FP16→FP32→FP16 roundtrip."""

    @torch.compile
    def fn(x):
        # FP16(STANDARD) → FP32(DL16_TO_FP32) → FP16(STANDARD)
        x_fp32 = x.to(torch.float32)
        return x_fp32.to(dtype=fp16)

    x = torch.randn(4, 128, device=device, dtype=fp16)
    result = fn(x)

    # Verify final EA is STANDARD
    assert_ea(result, ElementArrangement.STANDARD)

    # Verify correctness
    assert_val(fn, x, result)

    print("✓ FP16→FP32→FP16 roundtrip works")


@pytest.mark.parametrize("device", ["spyre"])
@pytest.mark.parametrize(
    "fp16",
    DtypeOpTable.fp16_types(),
    ids=lambda dt: str(dt).replace("torch.", ""),
)
def test_bidirectional_roundtrip_fp32_start(device, fp16):
    """Test FP32→FP16→FP32 roundtrip."""

    @torch.compile
    def fn(x):
        # FP32(STANDARD) → FP16(FP32_TO_DL16) → FP32(STANDARD)
        x_fp16 = x.to(dtype=fp16)
        return x_fp16.to(torch.float32)

    x = torch.randn(4, 128, device=device, dtype=torch.float32)
    result = fn(x)

    # Verify final EA is STANDARD
    assert_ea(result, ElementArrangement.STANDARD)

    # Verify correctness
    assert_val(fn, x, result)

    print("✓ FP32→FP16→FP32 roundtrip works")


def _stagger_fn(x, fp16):
    """fp32 → fp16(staggered) → stagger_to_standard_ea → standard EA fp16."""
    return torch.ops.spyre.stagger_to_standard_ea(x.to(dtype=fp16))


@pytest.mark.parametrize(
    "x",
    [
        # 1-D: stick-aligned
        torch.randn(64, dtype=torch.float32),
        torch.randn(128, dtype=torch.float32),
        # 1-D: non-stick-aligned (padded to 64-multiple)
        torch.nn.functional.pad(torch.randn(44, dtype=torch.float32), (0, 20)),
        # 2-D: stick-aligned
        torch.randn(4, 64, dtype=torch.float32),
        torch.randn(7, 128, dtype=torch.float32),
        # 2-D: non-stick-aligned (padded)
        torch.nn.functional.pad(torch.randn(7, 44, dtype=torch.float32), (0, 20)),
        # 3-D: stick-aligned
        torch.randn(2, 4, 64, dtype=torch.float32),
        torch.randn(3, 5, 128, dtype=torch.float32),
        # 3-D: non-stick-aligned (padded)
        torch.nn.functional.pad(torch.randn(2, 4, 44, dtype=torch.float32), (0, 20)),
        # 4-D: stick-aligned
        torch.randn(2, 3, 4, 64, dtype=torch.float32),
        torch.randn(2, 3, 4, 128, dtype=torch.float32),
        # 4-D: non-stick-aligned (padded)
        torch.nn.functional.pad(torch.randn(2, 3, 4, 44, dtype=torch.float32), (0, 20)),
    ],
)
@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize(
    "fp16",
    DtypeOpTable.fp16_types(),
    ids=lambda dt: str(dt).replace("torch.", ""),
)
def test_stagger_to_standard_ea(x, fp16):
    """stagger_to_standard_ea restores standard EA after fp32→fp16 (fp32todl16).

    Verifies:
      1. Output values match a plain x.to(fp16) on CPU (logical correctness).
      2. Output EA is STANDARD (layout correctness).
    """
    expected = x.to(fp16)

    compiled_fn = torch.compile(_stagger_fn, backend="inductor")

    # 1. Value correctness: Spyre result matches CPU fp16 cast.
    # fp32→fp16 rounding differs slightly (Spyre uses DF16); use fp16 tolerances.
    result = compiled_fn(x.to("spyre"), fp16).cpu()
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    # 2. Layout correctness: output EA must be STANDARD.
    spyre_result = compiled_fn(x.to("spyre"), fp16)
    ea = get_spyre_tensor_layout(spyre_result).element_arrangement
    assert ea == ElementArrangement.STANDARD, f"Expected STANDARD EA, got {ea}"


_SLICED_RMSNORM_SHAPES = {
    "gemma3-1b": (4, 1, 256, 1152),
    "gemma4-global": (32, 4, 512, 5376),
}


@pytest.mark.parametrize(
    "shape", list(_SLICED_RMSNORM_SHAPES), ids=list(_SLICED_RMSNORM_SHAPES)
)
@pytest.mark.parametrize("part", ["q", "k"])
def test_fp16_to_fp32_on_qkv_slice(shape, part):
    """A staggered upcast of a fused-QKV slice uses the slice's row span."""
    num_q_heads, num_kv_heads, head_dim, hidden = _SLICED_RMSNORM_SHAPES[shape]
    tokens = 8
    w_q = num_q_heads * head_dim
    w_kv = num_kv_heads * head_dim
    heads = num_q_heads if part == "q" else num_kv_heads
    start = 0 if part == "q" else w_q
    width = w_q if part == "q" else w_kv

    def fn(x, w_qkv, weight):
        qkv = x @ w_qkv
        part_view = qkv[:, start : start + width].reshape(tokens, heads, head_dim)
        x32 = part_view.float()
        normalized = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (normalized * (1.0 + weight.float())).to(x.dtype), x32

    args = (
        torch.randn(tokens, hidden, dtype=torch.float16) / 8,
        torch.randn(hidden, w_q + 2 * w_kv, dtype=torch.float16) / 8,
        torch.randn(head_dim, dtype=torch.float16) / 8,
    )
    expected, _ = fn(*args)
    result, upcast = torch.compile(fn, dynamic=False)(
        *(arg.to(DEVICE_NAME) for arg in args)
    )

    torch.testing.assert_close(result.cpu(), expected, rtol=0.01, atol=0.03)
    assert_ea(upcast, ElementArrangement.DL16_TO_FP32)


# ---------------------------------------------------------------------------
# Eager-path unit tests
# ---------------------------------------------------------------------------
def _build_eager_ea_tests():
    eager_to = {
        "to_kw_device_dtype": lambda x, d, dt: x.to(device=d, dtype=dt),
        "to_pos_device_kw_dtype": lambda x, d, dt: x.to(d, dtype=dt),
        "to_pos_device_dtype": lambda x, d, dt: x.to(d, dt),
        "to_torch_device_dtype": lambda x, d, dt: x.to(torch.device(d), dt),
    }

    device_pairs = [
        (DEVICE_NAME, DEVICE_NAME),
        (DEVICE_NAME, "cpu"),
        ("cpu", DEVICE_NAME),
        ("cpu", "cpu"),
    ]

    unsupported_dci_pairs = [
        (torch.float32, torch.float16),
    ]

    test_cases = []
    test_ids = []

    for src_dev, dst_dev in device_pairs:
        same_device = src_dev == dst_dev
        for fp16 in DtypeOpTable.fp16_types():
            is_unsupported = (torch.float32, fp16) in unsupported_dci_pairs
            if not same_device and is_unsupported:
                continue

            for _id, _to in eager_to.items():
                test_cases.append((src_dev, dst_dev, fp16, _to))

                dt_name = str(fp16).replace("torch.", "")
                test_ids.append(f"{src_dev}-{dst_dev}-{dt_name}-{_id}")

    return test_cases, test_ids


EAGER_TO_TEST_CASES, EAGER_TO_TEST_IDS = _build_eager_ea_tests()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "src_dev, dst_dev, fp16, eager_to",
    EAGER_TO_TEST_CASES,
    ids=EAGER_TO_TEST_IDS,
)
def test_eager_ea(src_dev, dst_dev, fp16, eager_to):
    """Verify eager mode EA across device transfer combinations."""
    same_spyre_device = src_dev == dst_dev == DEVICE_NAME

    # FP16 -> FP32 casting flow
    x16 = torch.randn(4, 128, device=src_dev, dtype=fp16)
    assert_ea(x16, ea_of(src_dev))

    y32 = eager_to(x16, dst_dev, torch.float32)
    assert_ea(
        y32,
        ElementArrangement.DL16_TO_FP32 if same_spyre_device else ea_of(dst_dev),
    )

    z16 = eager_to(y32, src_dev, fp16)
    assert_ea(z16, ea_of(src_dev))

    # FP32 -> FP16 casting flow
    x32 = torch.randn(4, 128, device=src_dev, dtype=torch.float32)
    assert_ea(x32, ea_of(src_dev))

    y16 = eager_to(x32, dst_dev, fp16)
    assert_ea(
        y16,
        ElementArrangement.FP32_TO_DL16 if same_spyre_device else ea_of(dst_dev),
    )

    z32 = eager_to(y16, src_dev, torch.float32)
    assert_ea(z32, ea_of(src_dev))


# ---------------------------------------------------------------------------
# Per-op EA propagation rules — EA tag + numerical correctness
#
# Pattern:
#   - assert_ea()   : the output carries the expected ElementArrangement
#   - assert_val()  : when output EA is STANDARD, values match CPU directly
#   - roundtrip     : when output EA is staggered, apply the reverse convert
#                     to get STANDARD, then compare with CPU
# ---------------------------------------------------------------------------

DEVICE = "spyre"


def _to_standard(t, fp16):
    """Roundtrip a staggered fp16/bf16 tensor back to STANDARD EA.

    DL16_TO_FP32  (fp16→fp32 stagger): restore via fp32→fp16  (fp32todl16)
    FP32_TO_DL16  (fp32→fp16 stagger): restore via fp16→fp32  (dl16tofp32)
    Either direction: apply the reverse convert so EA becomes STANDARD, then
    cast back to fp16 for a fair value comparison.
    """
    ea = get_ea(t)
    if ea == ElementArrangement.DL16_TO_FP32:
        # staggered fp32 → restore via fp32→fp16 → now STANDARD fp16
        return t.to(dtype=fp16)
    if ea == ElementArrangement.FP32_TO_DL16:
        # staggered fp16 → restore via fp16→fp32 → STANDARD fp32 → cast to fp16
        return t.to(torch.float32).to(dtype=fp16)
    return t  # already STANDARD


def _assert_numerically_close(
    spyre_result, cpu_fn, cpu_input, *, fp16, rtol=1e-2, atol=1e-2
):
    """Compare spyre_result with CPU reference, handling staggered EA via roundtrip."""
    ea = get_ea(spyre_result)
    if ea in (ElementArrangement.DL16_TO_FP32, ElementArrangement.FP32_TO_DL16):
        comparable = _to_standard(spyre_result, fp16).cpu()
    else:
        comparable = spyre_result.cpu()
    expected = cpu_fn(cpu_input)
    torch.testing.assert_close(comparable, expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Class 0 — Single-arg pointwise: always OK, propagate EA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize(
    "fp16", DtypeOpTable.fp16_types(), ids=lambda dt: str(dt).replace("torch.", "")
)
def test_rule_single_arg_pointwise_propagates_stagger(device, fp16):
    """Single-arg pointwise (neg) on a staggered tensor propagates EA unchanged.

    EA rule: output_ea == input_ea  (DL16_TO_FP32 in, DL16_TO_FP32 out)
    Numerical check: roundtrip to STANDARD, compare with CPU.
    """

    @torch.compile
    def fn(x):
        # fp16 → fp32 creates DL16_TO_FP32
        x_fp32 = x.to(torch.float32)
        # neg is single-arg pointwise — must propagate DL16_TO_FP32
        return torch.neg(x_fp32)

    x = torch.randn(4, 128, device=device, dtype=fp16)
    result = fn(x)

    # EA must be propagated unchanged from the fp32 staggered input
    assert_ea(result, ElementArrangement.DL16_TO_FP32)

    # Numerical check via roundtrip: neg(staggered fp32) → fp16 restore → compare
    _assert_numerically_close(
        result,
        lambda t: torch.neg(t.to(torch.float32)).to(fp16),
        x.cpu(),
        fp16=fp16,
    )


@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize(
    "fp16", DtypeOpTable.fp16_types(), ids=lambda dt: str(dt).replace("torch.", "")
)
def test_rule_single_arg_pointwise_standard_stays_standard(device, fp16):
    """Single-arg pointwise (abs) on STANDARD tensor stays STANDARD."""

    @torch.compile
    def fn(x):
        return torch.abs(x)

    x = torch.randn(4, 128, device=device, dtype=fp16)
    result = fn(x)

    assert_ea(result, ElementArrangement.STANDARD)
    # STANDARD output: compare directly with CPU
    torch.testing.assert_close(result.cpu(), fn(x.cpu()), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Class 1 — Multi-arg pointwise: all same EA → propagate; mixed → needs restore
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize(
    "fp16", DtypeOpTable.fp16_types(), ids=lambda dt: str(dt).replace("torch.", "")
)
def test_rule_multi_arg_pointwise_same_stagger_propagates(device, fp16):
    """add(staggered, staggered) — same EA on both inputs → output propagates EA.

    Both inputs share DL16_TO_FP32 → add is elementwise-correct → output is
    DL16_TO_FP32.
    """

    @torch.compile
    def fn(x, y):
        x_fp32 = x.to(torch.float32)  # DL16_TO_FP32
        y_fp32 = y.to(torch.float32)  # DL16_TO_FP32
        return torch.add(x_fp32, y_fp32)

    x = torch.randn(4, 128, device=device, dtype=fp16)
    y = torch.randn(4, 128, device=device, dtype=fp16)
    result = fn(x, y)

    assert_ea(result, ElementArrangement.DL16_TO_FP32)

    _assert_numerically_close(
        result,
        lambda t: torch.add(t[0].to(torch.float32), t[1].to(torch.float32)).to(fp16),
        (x.cpu(), y.cpu()),
        fp16=fp16,
    )


@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize(
    "fp16", DtypeOpTable.fp16_types(), ids=lambda dt: str(dt).replace("torch.", "")
)
def test_rule_multi_arg_pointwise_broadcast_standard_with_stagger(device, fp16):
    """add(staggered full, STANDARD broadcast) → output keeps staggered EA.

    A size-1 broadcast input is always compatible regardless of its EA.
    """

    @torch.compile
    def fn(x, scale):
        x_fp32 = x.to(torch.float32)  # DL16_TO_FP32
        # scale is fp32 with shape [1] — broadcasts along stick → STANDARD
        return torch.add(x_fp32, scale)

    x = torch.randn(4, 128, device=device, dtype=fp16)
    scale = torch.tensor([0.5], device=device, dtype=torch.float32)
    result = fn(x, scale)

    assert_ea(result, ElementArrangement.DL16_TO_FP32)

    _assert_numerically_close(
        result,
        lambda t: torch.add(t[0].to(torch.float32), t[1]).to(fp16),
        (x.cpu(), scale.cpu()),
        fp16=fp16,
    )


# ---------------------------------------------------------------------------
# Class 2 — Reduction on stick dim: output becomes STANDARD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize(
    "fp16", DtypeOpTable.fp16_types(), ids=lambda dt: str(dt).replace("torch.", "")
)
def test_rule_reduction_stick_dim_becomes_standard(device, fp16):
    """mean along the stick dim of a DL16_TO_FP32 tensor → output is STANDARD.

    Reduction is order-independent, so the staggered ordering is consumed and
    the output EA becomes STANDARD.  Values must match CPU directly.
    """

    @torch.compile
    def fn(x):
        x_fp32 = x.to(torch.float32)  # DL16_TO_FP32
        # reduce along dim=-1 (the stick dim) → EA consumed → STANDARD
        return x_fp32.mean(dim=-1)

    x = torch.randn(4, 128, device=device, dtype=fp16)
    result = fn(x)

    assert_ea(result, ElementArrangement.STANDARD)
    # STANDARD output: compare directly
    torch.testing.assert_close(
        result.cpu(),
        fn(x.cpu()),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize(
    "fp16", DtypeOpTable.fp16_types(), ids=lambda dt: str(dt).replace("torch.", "")
)
def test_rule_reduction_non_stick_dim_propagates_stagger(device, fp16):
    """mean along a non-stick dim of a DL16_TO_FP32 tensor → output keeps EA.

    Reducing over the outer (non-stick) dimension accumulates sticks as units;
    the intra-stick ordering is untouched, so EA propagates.
    """

    @torch.compile
    def fn(x):
        x_fp32 = x.to(torch.float32)  # DL16_TO_FP32, shape [4, 128]
        # reduce along dim=0 (outer / non-stick dim) → EA survives
        return x_fp32.mean(dim=0)  # output shape [128], still DL16_TO_FP32

    x = torch.randn(4, 128, device=device, dtype=fp16)
    result = fn(x)

    assert_ea(result, ElementArrangement.DL16_TO_FP32)

    _assert_numerically_close(
        result,
        lambda t: t.to(torch.float32).mean(dim=0).to(fp16),
        x.cpu(),
        fp16=fp16,
    )


# ---------------------------------------------------------------------------
# Class 3 — Restickify: swaps staggered EA between stick and non-stick dim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", [DEVICE])
@pytest.mark.parametrize(
    "fp16", DtypeOpTable.fp16_types(), ids=lambda dt: str(dt).replace("torch.", "")
)
def test_rule_restickify_swaps_ea(device, fp16):
    """Transpose (restickify) of a staggered tensor moves EA to the new stick.

    After fp16→fp32 (DL16_TO_FP32 on last dim), a transpose moves the stick
    to dim 0.  The staggered EA follows the former stick dimension to its new
    (non-stick) position; the new stick (former outer dim) is STANDARD.
    The full chain fp16→fp32→transpose→fp32→fp16 must match CPU.
    """

    @torch.compile
    def fn(x):
        x_fp32 = x.to(torch.float32)  # [4, 128] DL16_TO_FP32 on dim 1
        x_t = x_fp32.t()  # [128, 4] — restickify swaps dims
        # Restore to fp16 to produce STANDARD EA for comparison
        return x_t.to(dtype=fp16)

    x = torch.randn(4, 128, device=device, dtype=fp16)
    result = fn(x)

    # After fp32→fp16 restoration the output EA must be STANDARD
    assert_ea(result, ElementArrangement.STANDARD)
    torch.testing.assert_close(
        result.cpu(),
        fn(x.cpu()),
        rtol=1e-2,
        atol=1e-2,
    )


# Made with Bob
