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
import math
from torch._dynamo.exc import BackendCompilerFailed
from torch._inductor.exc import InductorError
from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout
from torch_spyre._inductor.dtype_ops import DtypeOpTable
from torch_spyre._inductor.constants import DEVICE_NAME
from utils_inductor import shapes2key, expects_raise


EA = ElementArrangement


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


def assert_val(fn, x, result, rtol=1e-3, atol=1e-3):
    """Compare ``result`` to ``fn`` on CPU; ``x`` is a tensor or a tuple of them."""
    result_ea = get_ea(result)
    if result_ea == EA.STANDARD:
        xs = x if isinstance(x, tuple) else (x,)
        result_cpu = fn(*(t.cpu() for t in xs))
        torch.testing.assert_close(result.cpu(), result_cpu, rtol=rtol, atol=atol)


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
@pytest.mark.parametrize("mode", ["compile", "eager"])
def test_mixed_bf16_fp32_add_shuffles_ea_mismatch(device, mode):
    """A bf16/fp32 add mixes EAs; a shuffle makes them match (#2843)."""

    def fn(x, y):
        return torch.add(x, y)

    x_fp32 = torch.randn(5120, dtype=torch.float32)
    y_bf16 = torch.randn(5120, dtype=torch.bfloat16)

    result = _run(fn, x_fp32.to(device), y_bf16.to(device), mode=mode)

    assert_ea(result, ElementArrangement.STANDARD)
    assert_val(fn, (x_fp32, y_bf16), result, rtol=1e-2, atol=1e-2)


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


# Shapes whose whole content fits in a single stick, so the tensor has no
# spatial dim outside the stick dim for the conversion op to loop over.
STICK_ONLY_SHAPES = [
    (),
    (1,),
    (1, 1),
    (1, 1, 1),
    (1, 1, 1, 1),
]


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
# pair of FP32 sticks, and the last pair's second stick must still be processed
# when work division splits the stick dim.
_PARTIAL_STICK_UPCAST_SHAPES = [
    (68,),
    (196,),
    (232,),
    (1000,),
    (4100,),
    (4, 68),
    (4, 100),
    (4, 104),
    (4, 196),
]


@pytest.mark.parametrize("shape", _PARTIAL_STICK_UPCAST_SHAPES, ids=str)
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
    numel = math.prod(shape)
    return (torch.arange(numel) % modulus + base).reshape(shape).half()


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


@pytest.mark.parametrize("tokens", [1, 64])
@pytest.mark.parametrize(
    "fp16",
    DtypeOpTable.fp16_types(),
    ids=lambda dt: str(dt).replace("torch.", ""),
)
def test_fp32_reduction_cast_into_matmul(tokens, fp16):
    """A matmul reads an fp32 sum cast to fp16 with K on the stick.

    This is the MoE expert combine. The cast's output is tagged FP32_TO_DL16,
    and the matmul used to take a candidate of it without K on the stick.
    Integers of magnitude <= 256 keep every partial sum exact in bf16 and
    DLFloat16.
    """
    torch._dynamo.reset()

    def fn(y, w, m):
        return (y.float() * w.float()).sum(0).to(fp16) @ m

    y = torch.randint(-1, 2, (2, tokens, 128)).to(fp16)
    w = torch.randint(0, 2, (2, tokens, 1)).to(fp16)
    m = torch.randint(-1, 2, (128, 128)).to(fp16)
    result = torch.compile(fn)(y.to(DEVICE_NAME), w.to(DEVICE_NAME), m.to(DEVICE_NAME))
    torch.testing.assert_close(result.cpu(), fn(y, w, m), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Shuffle op correctness verification to apply or restore element arrangement
# ---------------------------------------------------------------------------
_WHOLE_STICK_PAIRS = "requires whole FP32 stick pairs"
_RESCALE_REJECTION = "cannot rescale device layout"
_NOT_DIVISIBLE = "cannot be divided safely"
_BACKEND_ERROR = "Cannot find a valid minimum value for data stage"


class TestEAShuffle:
    # Shapes for both directions: stick-aligned, shorter than a stick, ending
    # inside a stick; ranks 1 to 4. (256,) has more than one fp32 stick pair, to
    # verify mb=1. (4, 192) and (4, 320) hold an odd number (>= 3) of fp32 stick
    # pairs when staggered as DL16_TO_FP32.
    _STICK_ALIGNED_SHAPES = [
        (64,),
        (128,),
        (256,),
        (2, 64),
        (4, 128),
        (64, 256),
        (2, 4, 64),
        (2, 3, 4, 64),
        (4, 192),
        (4, 320),
        (2,),
        (32,),
        (63,),
        (4, 16),
        (4, 32),
        (44,),
        (100,),
        (7, 44),
        (2, 4, 100),
    ]
    # fp16 -> fp32 (DL16_TO_FP32) also pads an odd number of fp32 sticks to a whole
    # pair. fp32 -> fp16 (FP32_TO_DL16) only accepts an even number of 32-element
    # fp32 sticks (rescale_stl_for_dtype), so these are left out of it.
    _DL16_TO_FP32_SHAPES = _STICK_ALIGNED_SHAPES + [(68,), (4, 69), (4, 96), (4, 160)]
    _FP32_TO_DL16_SHAPES = _STICK_ALIGNED_SHAPES

    # Odd number (>= 3) of fp32 stick pairs per row: dbo-opt fails on the shuffle
    # kernel when a core gets few rows (even pair counts are exact, and so are
    # (64, 320) and wider with 64 rows; (64, 192) depends on the test order).
    _ODD_PAIR_SHAPES = [(1, 192), (4, 160), (4, 192), (4, 320), (4, 448)]
    # Rows so long that the LX planner puts the shuffle's input and output in
    # different memories, one in LX and one in HBM: dbo-opt fails too. With
    # ``LX_PLANNING=0`` they compile and are exact, as are narrower rows (both
    # sides in LX) and more rows per core (both sides in HBM, as (64, 151936)).
    _WIDE_SHAPES = [(2, 128256)]
    _BACKEND_FAILURES = dict.fromkeys(_ODD_PAIR_SHAPES + _WIDE_SHAPES, _BACKEND_ERROR)

    # Shapes the compiler rejects today, with a fragment of the error. They are
    # left out of the shuffle cases on the device and checked by
    # test_ea_shuffle_rejected.
    _DL16_TO_FP32_REJECTED = {
        # fewer than a whole pair of 32-element fp32 sticks
        **dict.fromkeys([(2,), (32,), (4, 16), (4, 32)], _WHOLE_STICK_PAIRS),
        # a rank >= 3 read of the shuffled stick dim is not affine
        (2, 4, 64): _NOT_DIVISIBLE,
        (2, 3, 4, 64): _NOT_DIVISIBLE,
        **dict.fromkeys(_ODD_PAIR_SHAPES, _BACKEND_ERROR),
    }
    _FP32_TO_DL16_REJECTED = {
        # one 32-element fp32 stick is not a whole 64-element fp16 stick
        **dict.fromkeys([(32,), (4, 32)], _RESCALE_REJECTION),
    }
    _DIRECTIONS = [
        (EA.DL16_TO_FP32, "dl16_to_fp32", _DL16_TO_FP32_SHAPES, _DL16_TO_FP32_REJECTED),
        (EA.FP32_TO_DL16, "fp32_to_dl16", _FP32_TO_DL16_SHAPES, _FP32_TO_DL16_REJECTED),
    ]
    _RUN_CASES = [
        pytest.param(ea, s, id=f"{name}-{shapes2key((s,))}")
        for ea, name, shapes, rejected in _DIRECTIONS
        for s in shapes
        if s not in rejected
    ]
    _REJECTED_CASES = [
        pytest.param(ea, s, m, id=f"{name}-{shapes2key((s,))}")
        for ea, name, _, rejected in _DIRECTIONS
        for s, m in rejected.items()
    ]

    # Pointwise ops and chains of them; ``s`` is the staggered operand and ``std``
    # the STANDARD one. The last chain reads ``s`` three times (one shared shuffle).
    POINTWISE_FNS = {
        "add": torch.add,
        "mul": torch.mul,
        "eq": torch.eq,
        "rsub": lambda s, std: std - s,
        "add_then_mul": lambda s, std: (s + std) * 2.0,
        "mul_then_add": lambda s, std: s * std + std,
        "rsub_then_mul": lambda s, std: (std - s) * 2.0,
        "reuse_staggered": lambda s, std: (s * std + s) + std,
    }

    # Pointwise ops that mix a staggered operand with a STANDARD one; ``x16`` is a
    # 16-bit float tensor and ``y32`` an fp32 one, both STANDARD inputs.
    MIXED_EA_PROGRAMS = {
        "downcast_plus_std16": lambda x16, y32: y32.to(x16.dtype) + x16,
        "std16_times_downcast": lambda x16, y32: x16 * y32.to(x16.dtype),
        "upcast_plus_std32": lambda x16, y32: x16.float() + y32,
        "upcast_clone_plus_std32": lambda x16, y32: x16.float().clone() + y32,
        # flips y32 into the staggered order, then shuffles the result back
        "flip_then_shuffle_back": lambda x16, y32: (
            ((x16.float() * 2.0) * y32 + x16.float()) + y32
        ),
    }
    # Shapes with only the stick dim need the virtual mb=1 row for an fp32 -> fp32
    # shuffle once the row has more than one fp32 stick pair. (64, 448) is an odd
    # number of pairs with enough rows, and (64, 151936) is wide with both sides of
    # the shuffle in HBM.
    _MIXED_EA_SHAPES = [
        (2, 64),
        (4, 128),
        (64, 256),
        (128,),
        (256,),
        (512,),
        (1, 256),
        (64, 448),
        (64, 151936),
    ]
    _MIXED_EA_REJECTED = {(2, 32): _WHOLE_STICK_PAIRS, **_BACKEND_FAILURES}

    _FP16 = pytest.mark.parametrize(
        "fp16",
        DtypeOpTable.fp16_types(),
        ids=lambda dt: str(dt).replace("torch.", ""),
    )

    @staticmethod
    def _operands(ea, shape, fp16):
        """Inputs for ``ea``: ``x`` is converted to the dtype of ``y`` (STANDARD)."""
        if ea == EA.DL16_TO_FP32:
            x_dtype, y_dtype = fp16, torch.float32
        else:
            x_dtype, y_dtype = torch.float32, fp16
        x = _small_int_fp16(shape, 7).to(x_dtype)
        y = _small_int_fp16(shape, 5, base=3).to(y_dtype)
        return x, y

    def _check(self, fn, shape, fp16, ea=EA.DL16_TO_FP32):
        """Run ``fn(x, y)`` compiled on the device; it must equal the CPU result."""
        x, y = self._operands(ea, shape, fp16)
        result = _run(fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME))
        assert_ea(result, EA.STANDARD)
        assert_val(fn, (x, y), result)

    @staticmethod
    def _cast_then(op):
        """``op`` on ``x`` converted to the dtype of ``y``: shuffles the staggered x."""
        return lambda x, y: op(x.to(y.dtype), y)

    @_FP16
    @pytest.mark.parametrize("ea, shape", _RUN_CASES)
    def test_ea_shuffle(self, ea, shape, fp16):
        """A staggered operand added to a STANDARD one is shuffled, on every shape."""
        self._check(self._cast_then(torch.add), shape, fp16, ea)

    @expects_raise
    @_FP16
    @pytest.mark.parametrize("ea, shape, match", _REJECTED_CASES)
    def test_ea_shuffle_rejected(self, ea, shape, match, fp16):
        """Shapes the compiler cannot shuffle yet fail with a stable error."""
        with pytest.raises((InductorError, BackendCompilerFailed), match=match):
            self._check(self._cast_then(torch.add), shape, fp16, ea)

    @_FP16
    @pytest.mark.parametrize(
        "ea", [EA.DL16_TO_FP32, EA.FP32_TO_DL16], ids=["dl16_to_fp32", "fp32_to_dl16"]
    )
    @pytest.mark.parametrize(
        "pointwise", list(POINTWISE_FNS.values()), ids=list(POINTWISE_FNS)
    )
    def test_shuffle_pointwise(self, ea, fp16, pointwise):
        """Each pointwise op, and chains of them, read a shuffled operand."""
        self._check(self._cast_then(pointwise), (2, 64), fp16, ea)

    @_FP16
    @pytest.mark.parametrize("shape", _MIXED_EA_SHAPES, ids=lambda s: shapes2key((s,)))
    @pytest.mark.parametrize("program", list(MIXED_EA_PROGRAMS))
    def test_mixed_ea_pointwise(self, program, shape, fp16):
        """A pointwise op on a staggered and a STANDARD operand matches the CPU."""
        self._check(self.MIXED_EA_PROGRAMS[program], shape, fp16)

    @expects_raise
    @_FP16
    @pytest.mark.parametrize(
        "shape, match",
        [
            pytest.param(s, m, id=shapes2key((s,)))
            for s, m in _MIXED_EA_REJECTED.items()
        ],
    )
    def test_mixed_ea_pointwise_rejected(self, shape, match, fp16):
        """A staggered fp32 operand the compiler cannot shuffle fails cleanly."""
        with pytest.raises((InductorError, BackendCompilerFailed), match=match):
            self._check(self.MIXED_EA_PROGRAMS["upcast_plus_std32"], shape, fp16)


# Made with Bob
