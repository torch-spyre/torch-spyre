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
from torch._inductor.exc import InductorError
from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout
from torch_spyre._inductor import config
from torch_spyre._inductor.dtype_ops import DtypeOpTable
from torch_spyre._inductor.constants import DEVICE_NAME
from utils_inductor import (
    cached_randn,
    shapes2key,
    expects_raise,
)
from torch_spyre._inductor.propagate_hints import spyre_hint
from torch_spyre._inductor.wsr import propagate_named_dims as pnd


def _dtype_name(dt):
    return str(dt).split(".")[-1]


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
    torch.testing.assert_close(result.cpu(), result_cpu, rtol=1e-2, atol=1e-2)


def ea_of(dev):
    return ElementArrangement.STANDARD if dev == DEVICE_NAME else None


# Shapes whose whole content fits in a single stick, so the tensor has no
# spatial dim outside the stick dim for the conversion op to loop over.
STICK_ONLY_SHAPES = [
    (),
    (1,),
    (1, 1),
    (1, 1, 1),
    (1, 1, 1, 1),
]

# Stick aligned shape
BASIC_SHAPES = [(4, 128)]

_SHAPES = STICK_ONLY_SHAPES + BASIC_SHAPES

_TEST_CASES = [
    (device, mode, fp16, shape)
    for device in [DEVICE_NAME]
    for mode in ["compile", "eager"]
    for fp16 in DtypeOpTable.fp16_types()
    for shape in _SHAPES
]

_TEST_IDS = [
    f"{device}-{mode}-{fp16}-{shapes2key((shape,))}".replace("torch.", "")
    for device, mode, fp16, shape in _TEST_CASES
]


@pytest.mark.parametrize(
    "device, mode, fp16, shape",
    _TEST_CASES,
    ids=_TEST_IDS,
)
def test_fp16_to_fp32_standard_input(device, mode, fp16, shape):
    """Test FP16/BF16→FP32 with STANDARD input creates DL16_TO_FP32 (#2843)."""

    def fn(x):
        return x.to(torch.float32)

    x = torch.randn(shape, device=device, dtype=fp16)
    result = _run(fn, x, mode=mode)

    # Verify output EA
    expected_ea = (
        ElementArrangement.STANDARD
        if shape in STICK_ONLY_SHAPES
        else ElementArrangement.DL16_TO_FP32
    )
    assert_ea(result, expected_ea)

    # Verify correctness
    assert_val(fn, x, result)

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
    "device, mode, fp16, shape",
    _TEST_CASES,
    ids=_TEST_IDS,
)
def test_fp32_to_fp16_standard_input(device, mode, fp16, shape):
    """Test FP32→FP16 with STANDARD input creates FP32_TO_DL16."""

    def fn(x):
        return x.to(dtype=fp16)

    x = torch.randn(shape, device=device, dtype=torch.float32)
    result = _run(fn, x, mode=mode)

    # Eager and compiled casts use the same compiled D2D conversion path, so
    # both preserve the hardware conversion's staggered element arrangement.
    expected_ea = (
        ElementArrangement.STANDARD
        if shape in STICK_ONLY_SHAPES
        else ElementArrangement.FP32_TO_DL16
    )
    assert_ea(result, expected_ea)

    # Verify correctness
    assert_val(fn, x, result)

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


@pytest.mark.parametrize("shape", STICK_ONLY_SHAPES)
@pytest.mark.parametrize("start_dtype", [torch.float32, torch.float16])
def test_roundtrip_stick_only_shape(shape, start_dtype):
    """A conversion round trip works when the tensor is a scalar or all size-1 dims.

    A type conversion needs one spatial dim beyond the stick, which these shapes
    do not have; codegen supplies a virtual one. Both directions are exercised
    because a single fp16->fp32 output is staggered and not comparable to CPU.
    Random values make a lane read from the wrong place show up as a mismatch.
    """
    other_dtype = torch.float16 if start_dtype == torch.float32 else torch.float32

    def fn(t):
        return t.to(other_dtype).to(start_dtype)

    x = torch.randn(shape, dtype=start_dtype)
    result = torch.compile(fn, backend="inductor")(x.to("spyre"))

    ea = get_spyre_tensor_layout(result).element_arrangement
    assert ea == ElementArrangement.STANDARD, f"Expected STANDARD EA, got {ea}"
    torch.testing.assert_close(result.cpu(), fn(x), rtol=1e-3, atol=1e-3)


def test_int32_to_fp32_partial_stick_1d_then_rsqrt():
    """A conversion of a 1-D tensor that ends inside a stick pads it to whole sticks.

    The only loop variable is the stick, so codegen adds a virtual row before it;
    the stick must still be padded. int32 and fp32 share a stick width, so this
    covers the padding without a width change.
    """

    def fn(x):
        return torch.rsqrt(x.to(torch.float32))

    x = torch.randint(1, 1000, (44,), dtype=torch.int32)
    result = torch.compile(fn, dynamic=False)(x.to("spyre"))
    torch.testing.assert_close(result.cpu(), fn(x), rtol=1e-2, atol=1e-2)


# An op consuming an upcast FP32 value before the downcast back, on stick lengths
# that end inside a stick. The upcast value is staggered: each FP16 stick spans a
# pair of FP32 sticks. Padding the conversions is enough while work division
# leaves the stick dim unsplit, or while the stick dim rounded up to FP32 sticks
# is a whole number of pairs; (68,) is the unsplit case.
_PARTIAL_STICK_UPCAST_PASS = [(68,), (196,), (232,), (1000,), (4, 100), (4, 104)]
# Here work division splits the stick dim and the round-up ends in the middle of
# a pair, so the pair's second stick is left to no core.
# TODO: make work division handle staggered FP32, splitting the stick dim by
# whole stick pairs.
_PARTIAL_STICK_UPCAST_NEEDS_PAIRS = [(4100,), (4, 68), (4, 196)]


@pytest.mark.parametrize(
    "shape",
    [
        *_PARTIAL_STICK_UPCAST_PASS,
        *[
            pytest.param(
                shape,
                marks=pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="TODO: size work division for an upcast value by whole "
                    "stick pairs, so the last pair's second stick is processed",
                ),
            )
            for shape in _PARTIAL_STICK_UPCAST_NEEDS_PAIRS
        ],
    ],
    ids=str,
)
def test_upcast_consumed_on_partial_stick(shape):
    """An op between an FP16 -> FP32 -> FP16 round trip keeps every element.

    The values are exact in DLFloat16, so any element the device moves or drops
    shows up as a mismatch.
    """
    torch._dynamo.reset()

    def fn(x):
        y = x.to(torch.float32)
        return (y + y).to(torch.float16)

    x = (torch.randint(-64, 64, shape) / 8).to(torch.float16)
    result = torch.compile(fn)(x.to("spyre"))
    torch.testing.assert_close(result.cpu(), fn(x), rtol=0, atol=0)


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


@pytest.mark.parametrize("lx_planning", [False, True], ids=["hbm", "lx"])
@pytest.mark.parametrize(
    "fp16",
    DtypeOpTable.fp16_types(),
    ids=lambda dt: str(dt).replace("torch.", ""),
)
def test_stagger_to_standard_ea_one_core_row_order(lx_planning, fp16):
    """Keep conflicting physical row segments local to one core (#4703)."""
    torch._dynamo.reset()
    # Distinct, exactly representable rows make a row permutation unambiguous.
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)
    x = x.expand(2, 4, 64).contiguous()
    with config.patch({"sencores": 1, "lx_planning": lx_planning}):
        compiled_fn = torch.compile(_stagger_fn, backend="inductor", fullgraph=True)
        result = compiled_fn(x.to(DEVICE_NAME), fp16)
    torch.testing.assert_close(result.cpu(), x.to(fp16), atol=0, rtol=0)
    assert_ea(result, ElementArrangement.STANDARD)


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


# (input dtype, on-device conversion, its staggered arrangement, conversion back)
_STAGGERING_CONVERSIONS = {
    "dl16_to_fp32": (
        torch.float16,
        lambda t: t.float(),
        ElementArrangement.DL16_TO_FP32,
        lambda y: y.half(),
    ),
    "fp32_to_dl16": (
        torch.float32,
        lambda t: t.half(),
        ElementArrangement.FP32_TO_DL16,
        lambda y: y.float(),
    ),
}

_FRESH_ALLOCATION_PROGRAMS = {
    "cat_dim0": lambda a, b: torch.cat([a, b]),
    "cat_stick_dim": lambda a, b: torch.cat([a, b], dim=1),
    "stack": lambda a, b: torch.stack([a, b]),
    "pad_rows": lambda a, b: torch.nn.functional.pad(a, (0, 0, 0, 4)),
}


def _small_int_fp16(shape, modulus, base=0):
    # Integers <= 1024 are exact in DLFloat16, so results compare bit-exact.
    return (torch.arange(shape[0] * shape[1]) % modulus + base).reshape(shape).half()


@pytest.mark.parametrize("conversion", list(_STAGGERING_CONVERSIONS))
@pytest.mark.parametrize("program", list(_FRESH_ALLOCATION_PROGRAMS))
def test_staggered_into_fresh_allocation(program, conversion):
    """cat, stack and F.pad of converted tensors keep the staggered arrangement."""
    combine = _FRESH_ALLOCATION_PROGRAMS[program]
    dtype, convert, expected_ea, restore = _STAGGERING_CONVERSIONS[conversion]
    a = _small_int_fp16((4, 64), 7).to(dtype)
    b = _small_int_fp16((4, 64), 5, base=10).to(dtype)

    def fn(a, b):
        return combine(convert(a), convert(b))

    out = torch.compile(fn)(a.to(DEVICE_NAME), b.to(DEVICE_NAME))
    assert_ea(out, expected_ea)

    def round_trip(a, b):
        return restore(fn(a, b))

    restored = torch.compile(round_trip)(a.to(DEVICE_NAME), b.to(DEVICE_NAME))
    assert_ea(restored, ElementArrangement.STANDARD)
    assert torch.equal(restored.cpu(), round_trip(a, b))


_WHOLE_STICK_DIM_PROGRAMS = {
    # Width 100 is not a multiple of 64, but each write covers the whole stick dim.
    "dl16_to_fp32_cat_rows": (
        torch.float16,
        lambda a, b: torch.cat([a.float(), b.float()]),
        ElementArrangement.DL16_TO_FP32,
        lambda y: y.half(),
    ),
    "fp32_to_dl16_pad_rows": (
        torch.float32,
        lambda a, b: torch.nn.functional.pad(a.half(), (0, 0, 0, 4)),
        ElementArrangement.FP32_TO_DL16,
        lambda y: y.float(),
    ),
}


@pytest.mark.parametrize("program", list(_WHOLE_STICK_DIM_PROGRAMS))
def test_staggered_whole_stick_dim_write(program):
    """A write of the whole stick dim keeps the arrangement at a width not a
    multiple of 64, when its last stagger unit is allocated whole."""
    dtype, fn, expected_ea, restore = _WHOLE_STICK_DIM_PROGRAMS[program]
    a = _small_int_fp16((4, 100), 7).to(dtype)
    b = _small_int_fp16((4, 100), 5, base=10).to(dtype)

    out = torch.compile(fn)(a.to(DEVICE_NAME), b.to(DEVICE_NAME))
    assert_ea(out, expected_ea)

    def round_trip(a, b):
        return restore(fn(a, b))

    restored = torch.compile(round_trip)(a.to(DEVICE_NAME), b.to(DEVICE_NAME))
    assert_ea(restored, ElementArrangement.STANDARD)
    assert torch.equal(restored.cpu(), round_trip(a, b))


@pytest.mark.parametrize("conversion", list(_STAGGERING_CONVERSIONS))
def test_cat_of_staggered_along_stick_dim_with_tail(conversion):
    """A write at a whole-unit offset may end the stick dim in a partial unit."""
    dtype, convert, expected_ea, restore = _STAGGERING_CONVERSIONS[conversion]
    a = _small_int_fp16((4, 64), 7).to(dtype)
    b = _small_int_fp16((4, 100), 5, base=10).to(dtype)

    def fn(a, b):
        return torch.cat([convert(a), convert(b)], dim=1)

    out = torch.compile(fn)(a.to(DEVICE_NAME), b.to(DEVICE_NAME))
    assert_ea(out, expected_ea)

    def round_trip(a, b):
        return restore(fn(a, b))

    restored = torch.compile(round_trip)(a.to(DEVICE_NAME), b.to(DEVICE_NAME))
    assert_ea(restored, ElementArrangement.STANDARD)
    assert torch.equal(restored.cpu(), round_trip(a, b))


@pytest.mark.parametrize("conversion", list(_STAGGERING_CONVERSIONS))
def test_staggered_pad_fills_partial_last_unit(conversion):
    """A non-zero pad fill ending in a partial stagger unit lands on its elements."""
    dtype, convert, expected_ea, restore = _STAGGERING_CONVERSIONS[conversion]
    a = _small_int_fp16((4, 64), 7).to(dtype)

    def fn(a):
        return torch.nn.functional.pad(convert(a), (0, 36), value=5.0)

    out = torch.compile(fn)(a.to(DEVICE_NAME))
    assert_ea(out, expected_ea)

    def round_trip(a):
        return restore(fn(a))

    restored = torch.compile(round_trip)(a.to(DEVICE_NAME))
    assert_ea(restored, ElementArrangement.STANDARD)
    assert torch.equal(restored.cpu(), round_trip(a))


def _index_write(a, b, idx):
    # The value is computed before the allocation, so it is laid out by then.
    value = a.float() * 2
    out = torch.empty(4, 64, dtype=torch.float32, device=a.device)
    out[idx] = value
    return out


_ORDERLESS_READ_PROGRAMS = {
    # The STANDARD reads here carry no element arrangement: a fill, an index.
    "cat_zeros": lambda a, b, idx: torch.cat(
        [a.float(), torch.zeros(4, 64, dtype=torch.float32, device=a.device)]
    ),
    "cat_full": lambda a, b, idx: torch.cat(
        [a.float(), torch.full((4, 64), 3.0, dtype=torch.float32, device=a.device)]
    ),
    "index_write": _index_write,
}


@pytest.mark.parametrize("program", list(_ORDERLESS_READ_PROGRAMS))
def test_staggered_write_with_orderless_reads(program):
    """A constant fill or an index read does not mix arrangements."""
    fn = _ORDERLESS_READ_PROGRAMS[program]
    a = _small_int_fp16((4, 64), 7)
    b = _small_int_fp16((4, 64), 5, base=10)
    idx = torch.tensor([2, 0, 3, 1])

    out = torch.compile(fn)(a.to(DEVICE_NAME), b.to(DEVICE_NAME), idx.to(DEVICE_NAME))
    assert_ea(out, ElementArrangement.DL16_TO_FP32)

    def round_trip(a, b, idx):
        return fn(a, b, idx).half()

    restored = torch.compile(round_trip)(
        a.to(DEVICE_NAME), b.to(DEVICE_NAME), idx.to(DEVICE_NAME)
    )
    assert_ea(restored, ElementArrangement.STANDARD)
    assert torch.equal(restored.cpu(), round_trip(a, b, idx))


@expects_raise
def test_pad_of_staggered_with_odd_fp32_sticks_raises():
    """A DL16_TO_FP32 stick dim of 3 fp32 sticks has no room for its last pair."""
    a = _small_int_fp16((4, 96), 7)

    def fn(a):
        return torch.nn.functional.pad(a.float(), (0, 0, 0, 4))

    with pytest.raises(InductorError, match="not whole units"):
        torch.compile(fn)(a.to(DEVICE_NAME))


@expects_raise
def test_cat_of_staggered_and_standard_raises():
    """One allocation cannot hold a staggered and a STANDARD writer."""
    a = _small_int_fp16((4, 64), 7)
    c = torch.ones(4, 64, dtype=torch.float32)

    def fn(a, c):
        return torch.cat([a.float(), c])

    with pytest.raises(InductorError, match="mixed element arrangements"):
        torch.compile(fn)(a.to(DEVICE_NAME), c.to(DEVICE_NAME))


@expects_raise
def test_pad_of_staggered_inside_stick_pair_raises():
    """A pad that splits DL16_TO_FP32 stick pairs cannot keep the stagger."""
    a = _small_int_fp16((4, 64), 7)

    def fn(a):
        # Width 128 leaves room for whole pairs; both fills and the copy split one.
        return torch.nn.functional.pad(a.float(), (32, 32))

    with pytest.raises(InductorError, match="not whole units"):
        torch.compile(fn)(a.to(DEVICE_NAME))


# ---------------------------------------------------------------------------
# D2H EA restore, H2D write back
# ---------------------------------------------------------------------------
def _build_ea_d2h_tests():
    # Shapes common to both conversions: the stick dim is a whole number of
    # 64-element sticks.
    STICK_ALIGNED_SHAPES = [
        (64,),
        (128,),
        (4, 64),
        (4, 128),
        (7, 128),
        (2, 4, 64),
        (3, 5, 128),
        (4, 8, 128),
        (8, 16, 64),
        (8, 16, 128),
        (2, 3, 4, 64),
        (2, 3, 4, 128),
        (2, 4, 8, 64),
    ]

    # n (2, 63]: 2 <= n <= 63 shapes
    SMALL_N_SHAPES = [
        (2,),
        (3,),
        (4,),
        (16,),
        (32,),
        (63,),
        (2, 2),
        (4, 16),
        (4, 32),
    ]

    # base shapes
    BASE_SHAPES = STICK_ALIGNED_SHAPES + SMALL_N_SHAPES + STICK_ONLY_SHAPES

    # fp16 -> fp32 doubles the stick count, so the output is always whole stick
    # pairs once the stick dim is at least 64 wide. Stick dims narrower than 64
    # only get a single fp32 stick; these are rejected on D2H.
    FP16_TO_FP32_SHAPES = BASE_SHAPES + [
        # Partial last stick or pair
        (68,),
        (4, 68),
        (8, 16, 68),
        (2, 96),
        (4, 96),
        (8, 16, 96),
        (4, 100),
        (8, 16, 100),
        (4, 160),
        (8, 16, 160),
        # Narrower than one stick (rejected on D2H)
        (44,),
        (4, 16),
        (4, 32),
        (4, 44),
        (7, 44),
        (2, 4, 44),
        (2, 3, 4, 44),
    ]

    # fp32 -> fp16 halves the stick count, so the compiler only accepts an even
    # number of 32-element fp32 sticks (rescale_stl_for_dtype). Shapes with an odd
    # stick count (68, 96, 160 and 32 wide) are not supported.
    FP32_TO_FP16_SHAPES = BASE_SHAPES + [
        # One stick + a partial stick
        (4, 100),
        (8, 16, 100),
        # Single partial stick
        (44,),
        (4, 16),
        (4, 44),
        (7, 44),
        (2, 4, 44),
        (2, 3, 4, 44),
    ]

    EA = ElementArrangement
    FP32 = torch.float32
    params = {}
    for FP16 in DtypeOpTable.fp16_types():
        tests = [
            (FP16, FP32, EA.DL16_TO_FP32, FP16_TO_FP32_SHAPES),
            (FP32, FP16, EA.FP32_TO_DL16, FP32_TO_FP16_SHAPES),
        ]
        for src_dtype, dst_dtype, base_expected_ea, shapes in tests:
            conv_name = f"{_dtype_name(src_dtype)}_to_{_dtype_name(dst_dtype)}"
            for shape in shapes:
                expected_ea = (
                    ElementArrangement.STANDARD
                    if shape in STICK_ONLY_SHAPES
                    else base_expected_ea
                )
                key = f"{conv_name}_{shapes2key((shape,))}"
                params[key] = (src_dtype, dst_dtype, expected_ea, shape)
    return params


EA_D2H_TEST_PARAMS = _build_ea_d2h_tests()


def _rescale_rejected(src_dtype, dst_dtype, shape):
    # FP32 -> FP16/BF16 conversion requires an even number of FP32 sticks because two
    # 32-element FP32 sticks combine to form one 64-element FP16/BF16 stick.  If the
    # input shape results in an odd number of FP32 sticks (e.g., shape (32,) has 1
    # stick, shape (4, 32) has 1 stick in the last dim), `rescale_stl_for_dtype` rejects
    # the layout as it cannot form whole output sticks. See #3809, #5361.
    _RESCALE_REJECTED_SHAPES = [
        (32,),
        (4, 32),
    ]

    return (
        src_dtype == torch.float32
        and dst_dtype in (torch.float16, torch.bfloat16)
        and shape in _RESCALE_REJECTED_SHAPES
    )


def _d2h_unsupported(expected_ea, shape):
    # fp16 -> fp32 restores whole fp32 stick pairs. A stick dim narrower than
    # 64 only gets a single fp32 stick, so half of each pair has nowhere to go.
    return expected_ea == ElementArrangement.DL16_TO_FP32 and shape[-1] < 64


def _convert_one_row_per_core(x, dst_dtype):
    """Convert on device with one M row per core.

    Work division does not account for stick padding yet, so a single row per
    core keeps the converted values correct for padded stick dims.
    """
    names = list("TBMN"[-x.dim() :])  # last dim is the stick dim N
    torch._dynamo.reset()  # also clears declared dim names
    for name, size in zip(names, x.shape):
        pnd.declare_tensor_dim(name, size)
    x_dev = x.to(DEVICE_NAME)
    pnd.name_tensor_dims(x_dev, names)

    def convert(a):
        if a.dim() < 2:  # no M dim to divide
            return a.to(dst_dtype)
        with spyre_hint(work_div={"M": x.shape[-2]}):
            return a.to(dst_dtype)

    # static: a dynamic recompile is a different kernel
    return torch.compile(convert, dynamic=False)(x_dev)


@pytest.mark.parametrize(
    "src_dtype, dst_dtype, expected_ea, shape",
    EA_D2H_TEST_PARAMS.values(),
    ids=EA_D2H_TEST_PARAMS.keys(),
)
def test_ea_restore_d2h(src_dtype, dst_dtype, expected_ea, shape):
    # Small integers are exact in fp32, fp16 and bf16, so no rounding noise
    values = torch.ceil(cached_randn(shape, dtype=torch.float32) * 5)
    x = values.to(src_dtype)
    ref = values.to(dst_dtype)

    if _rescale_rejected(src_dtype, dst_dtype, shape):
        with pytest.raises(InductorError, match="cannot rescale device layout"):
            _convert_one_row_per_core(x, dst_dtype)
        return

    y = _convert_one_row_per_core(x, dst_dtype)
    assert_ea(y, expected_ea)

    if _d2h_unsupported(expected_ea, shape):
        with pytest.raises(RuntimeError, match="D2H requires whole stick pairs"):
            y.cpu()  # D2H -> ea_restore
        return

    got = y.cpu()  # D2H -> ea_restore
    assert got.dtype == ref.dtype
    assert got.shape == ref.shape
    assert torch.equal(got, ref)


EA_H2D_TEST_PARAMS = EA_D2H_TEST_PARAMS

EA_H2D_BASIC_TEST_PARAMS = {
    key: params for key, params in EA_H2D_TEST_PARAMS.items() if params[3] in _SHAPES
}


def _make_x_and_new(shape, src_dtype, dst_dtype, offset=100.0):
    """Small integers, exact in fp16, bf16 and fp32, for reproducible compares.

    bf16 only represents integers exactly up to 256, so the offset keeps
    values + offset + 1 under that. A +1000 offset would round in bf16.
    """
    values = torch.ceil(cached_randn(shape, dtype=torch.float32) * 5)
    x = values.to(src_dtype)  # source data on host
    new = (values + offset).to(dst_dtype)  # data to be H2D-copied
    return x, new


@pytest.mark.parametrize(
    "src_dtype, dst_dtype, expected_ea, shape",
    EA_H2D_TEST_PARAMS.values(),
    ids=EA_H2D_TEST_PARAMS.keys(),
)
def test_ea_restore_h2d(src_dtype, dst_dtype, expected_ea, shape):
    x, new = _make_x_and_new(shape, src_dtype, dst_dtype)

    if _rescale_rejected(src_dtype, dst_dtype, shape):
        with pytest.raises(InductorError, match="cannot rescale device layout"):
            _convert_one_row_per_core(x, dst_dtype)
        return

    # Create staggered tensor on device
    y = _convert_one_row_per_core(x, dst_dtype)
    assert_ea(y, expected_ea)

    if _d2h_unsupported(expected_ea, shape):
        # ea_restore now runs for H2D too and applies the same whole-stick-pair
        # check as D2H, so the copy_ itself is rejected.
        with pytest.raises(RuntimeError, match="requires whole stick pairs"):
            y.copy_(new)
        return

    # H2D `copy_` of new (STANDARD host data) into the staggered y.
    # The EA tag is unchanged, copy_ writes only the data.
    y.copy_(new)
    assert_ea(y, expected_ea)

    # D2H restore must now produce exactly `new`.
    got = y.cpu()
    assert got.dtype == new.dtype
    assert got.shape == new.shape
    assert torch.equal(got, new), (
        f"{shape} {src_dtype}->{dst_dtype}: H2D copy_ into staggered tensor "
        f"mismatch ({(got != new).sum().item()}/{new.numel()} wrong)"
    )


@pytest.mark.parametrize(
    "src_dtype, dst_dtype, expected_ea, shape",
    EA_H2D_BASIC_TEST_PARAMS.values(),
    ids=EA_H2D_BASIC_TEST_PARAMS.keys(),
)
def test_ea_restore_h2d_compiled_consumer(src_dtype, dst_dtype, expected_ea, shape):
    x, new = _make_x_and_new(shape, src_dtype, dst_dtype)

    y = _convert_one_row_per_core(x, dst_dtype)
    assert_ea(y, expected_ea)

    # H2D copy into the staggered buffer.
    y.copy_(new)

    # Compiled consumer: (y + 1).to(src_dtype). Reading y through a kernel
    # verifies that the data on the device really is in staggered order.
    def consume(a):
        return (a + 1.0).to(src_dtype)

    torch._dynamo.reset()
    z = torch.compile(consume, dynamic=False)(y)
    got = z.cpu()

    # `new` is already dst_dtype, so add 1 in dst_dtype like the device does.
    expected = (new + 1.0).to(src_dtype)
    assert torch.equal(got, expected), (
        f"{shape} {src_dtype}->{dst_dtype}: compiled consumer after H2D "
        f"mismatch ({(got != expected).sum().item()}/{expected.numel()} wrong)"
    )


@pytest.mark.parametrize(
    "src_dtype, dst_dtype, expected_ea, shape",
    EA_H2D_BASIC_TEST_PARAMS.values(),
    ids=EA_H2D_BASIC_TEST_PARAMS.keys(),
)
def test_ea_h2d_d2h_h2d_chain(src_dtype, dst_dtype, expected_ea, shape):
    x, new1 = _make_x_and_new(shape, src_dtype, dst_dtype, offset=100.0)
    _, new2 = _make_x_and_new(shape, src_dtype, dst_dtype, offset=50.0)
    assert not torch.equal(new1, new2)  # distinct payloads

    y = _convert_one_row_per_core(x, dst_dtype)
    assert_ea(y, expected_ea)

    # H2D #1 -> D2H
    y.copy_(new1)
    assert_ea(y, expected_ea)
    assert torch.equal(y.cpu(), new1), f"{shape}: chain read1 mismatch"

    # H2D #2 (same staggered tensor, different data) -> D2H
    y.copy_(new2)
    assert_ea(y, expected_ea)
    assert torch.equal(y.cpu(), new2), f"{shape}: chain read2 mismatch"

    # A third round, back to the first payload, so it can't be a one-shot fluke.
    y.copy_(new1)
    assert_ea(y, expected_ea)
    assert torch.equal(y.cpu(), new1), f"{shape}: chain read3 mismatch"


# Made with Bob
