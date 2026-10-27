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
from torch_spyre._inductor.dtype_ops import DtypeOpTable
from torch_spyre._inductor.constants import DEVICE_NAME
from utils_inductor import cached_randn, shapes2key, expects_raise


EA = ElementArrangement


def _run(fn, *args, mode="compile"):
    if mode == "compile":
        fn = torch.compile(fn)
    # specialize_int: ``shuffle`` reads ``ea.value`` inside the traced function,
    # and the target EA must stay a constant rather than become a symbolic int.
    with torch._dynamo.config.patch(specialize_int=True):
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
# Shuffle op
# ---------------------------------------------------------------------------
def shuffle(x, ea):
    return torch.ops.spyre.shuffle(x, ea.value)


class TestEAShuffle:
    """Tests for explicit torch.ops.spyre.shuffle functionality."""

    BASIC_SHAPES = [(2, 64)]

    SHAPES = [
        # 1-D: stick-aligned
        (64,),
        (128,),
        # 1-D: non-stick-aligned
        (44,),
        (100,),
        # 2-D: stick-aligned
        (4, 64),
        (7, 128),
        # 2-D: non-stick-aligned
        (7, 44),
        (3, 100),
        # 3-D: stick-aligned
        (2, 4, 64),
        (3, 5, 128),
        # 3-D: non-stick-aligned
        (2, 4, 44),
        # 4-D: stick-aligned
        (2, 3, 4, 64),
        (2, 3, 4, 128),
        # 4-D: non-stick-aligned
        (2, 3, 4, 44),
    ]

    OPS_DICT = {
        "add": torch.add,
        "mul": torch.mul,
        "eq": torch.eq,
    }

    @staticmethod
    def _case(in_dtype, out_dtype=None, in_ea=None, out_ea=None, rtol=1e-3, atol=1e-3):
        in_ea = in_ea if in_ea is not None else EA.STANDARD
        out_ea = out_ea if out_ea is not None else EA.STANDARD
        if out_dtype is not None:
            src = "dl16" if in_dtype == torch.float16 else "fp32"
            dst = "fp32" if out_dtype == torch.float32 else "dl16"
            name = f"{src}_to_{dst}_to_std"
        else:
            name = f"std_to_{in_ea.name.lower()}"
        return (name, in_dtype, out_dtype, in_ea, out_ea, rtol, atol)

    TEST_CASES = [
        # Case 1: fp16 -> fp32 (staggered DL16_TO_FP32) -> shuffle to STANDARD
        _case(torch.float16, torch.float32),
        # Case 2: fp32 -> fp16 (staggered FP32_TO_DL16) -> shuffle to STANDARD
        _case(torch.float32, torch.float16, rtol=1e-2, atol=1e-2),
        # Case 3: fp32 (STANDARD) -> shuffle to FP32_TO_DL16
        _case(torch.float32, None, EA.FP32_TO_DL16, EA.FP32_TO_DL16, 0, 0),
        # Case 4: fp16 (STANDARD) -> shuffle to DL16_TO_FP32
        _case(torch.float16, None, EA.DL16_TO_FP32, EA.DL16_TO_FP32, 0, 0),
    ]

    @pytest.mark.parametrize("shape", SHAPES, ids=[shapes2key((s,)) for s in SHAPES])
    @pytest.mark.parametrize(
        "name, base_dtype, cast_dtype, target_ea, exp_ea, rtol, atol",
        TEST_CASES,
        ids=[case[0] for case in TEST_CASES],
    )
    def test_shuffle(
        self, shape, name, base_dtype, cast_dtype, target_ea, exp_ea, rtol, atol
    ):
        def fn(x):
            x_proc = x.to(cast_dtype) if cast_dtype is not None else x
            return shuffle(x_proc, target_ea)

        x = cached_randn(shape, dtype=base_dtype)
        result = _run(fn, x.to(DEVICE_NAME))

        assert_ea(result, exp_ea)
        assert_val(fn, x, result, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "shape", BASIC_SHAPES, ids=[shapes2key((s,)) for s in BASIC_SHAPES]
    )
    @pytest.mark.parametrize(
        "fp16", DtypeOpTable.fp16_types(), ids=lambda dt: str(dt).replace("torch.", "")
    )
    def test_shuffle_preserves_geometry(self, shape, fp16):
        x_dev = cached_randn(shape, dtype=fp16).to(DEVICE_NAME)
        in_layout = get_spyre_tensor_layout(x_dev)

        y_dev = _run(lambda t: shuffle(t, EA.DL16_TO_FP32), x_dev)
        out_layout = get_spyre_tensor_layout(y_dev)

        assert y_dev.dtype == x_dev.dtype
        assert y_dev.shape == x_dev.shape
        assert out_layout.element_arrangement == EA.DL16_TO_FP32
        assert out_layout.device_size == in_layout.device_size
        assert out_layout.stride_map == in_layout.stride_map
        assert out_layout.device_dtype == in_layout.device_dtype

    @pytest.mark.parametrize(
        "shape", BASIC_SHAPES, ids=[shapes2key((s,)) for s in BASIC_SHAPES]
    )
    def test_shuffle_feeds_downstream_op(self, shape):
        def fn(x, y):
            # x is fp16 -> fp32 (staggered) -> shuffle to std -> add to y (std)
            return (shuffle(x.to(torch.float32), EA.STANDARD) + y) * 2.0

        x = cached_randn(shape, dtype=torch.float16)
        y = cached_randn(shape, dtype=torch.float32)

        result = _run(fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME))

        assert_ea(result, EA.STANDARD)
        assert_val(fn, (x, y), result, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "shape", BASIC_SHAPES, ids=[shapes2key((s,)) for s in BASIC_SHAPES]
    )
    @pytest.mark.parametrize(
        "op",
        list(OPS_DICT.values()),
        ids=list(OPS_DICT.keys()),
    )
    def test_shuffle_implicit_pointwise(self, shape, op):
        def fn(x_stg, y_std):
            return op(x_stg, y_std)

        x = cached_randn(shape, dtype=torch.float32)
        y = cached_randn(shape, dtype=torch.float32)

        x_dev = x.to(DEVICE_NAME)
        y_dev = y.to(DEVICE_NAME)

        x_stg = shuffle(x_dev, EA.FP32_TO_DL16)
        result = _run(fn, x_stg, y_dev)

        assert_ea(result, EA.STANDARD)
        assert_val(fn, (x, y), result)

    @expects_raise
    @pytest.mark.parametrize(
        "shape", BASIC_SHAPES, ids=[shapes2key((s,)) for s in BASIC_SHAPES]
    )
    def test_shuffle_invalid_ea(self, shape):
        x = cached_randn(shape, dtype=torch.float16).to(DEVICE_NAME)
        with pytest.raises(Exception, match="shuffle: invalid EA"):
            _run(lambda t: torch.ops.spyre.shuffle(t, 99999), x)


# Made with Bob
