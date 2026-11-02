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
Tests for ElementArrangement: bidirectional FP16↔FP32 type conversion, the
shuffle of staggered operands to STANDARD (``insert_ea_shuffle``,
``optimize_shuffling``) and the read-only ``validate_ops`` check.

Tests in mode: compile, eager
FP16 types: FP16, BF16
Tests all 4 conversion cases:
1. FP16→FP32 with STANDARD → DL16_TO_FP32
2. FP16→FP32 with FP32_TO_DL16 → STANDARD
3. FP32→FP16 with STANDARD → FP32_TO_DL16
4. FP32→FP16 with DL16_TO_FP32 → STANDARD
"""

import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import sympy
import torch
from torch._inductor import config as inductor_config
from torch._inductor.dependencies import MemoryDep
from torch._inductor.exc import InductorError
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Scatter,
)
from torch._inductor.virtualized import V
from torch_spyre._C import (
    ElementArrangement,
    SpyreTensorLayout,
    get_device_dtype,
    get_spyre_tensor_layout,
)
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor import element_arrangement, passes
from torch_spyre._inductor.codegen.compute_ops import gen_coord_info_value
from torch_spyre._inductor.constants import DEVICE_NAME, STAGGERED_EAS
from torch_spyre._inductor.dtype_ops import DtypeOpTable
from torch_spyre._inductor.element_arrangement import (
    insert_ea_shuffle,
    optimize_shuffling,
    validate_ops,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.ir import FixedTiledLayout
from torch_spyre._inductor.passes import CustomPreSchedulingPasses
from utils_inductor import cached_randn, expects_raise, shapes2key


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
    numel = torch.Size(shape).numel()
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


# ---------------------------------------------------------------------------
# Shuffle of staggered operands
# ---------------------------------------------------------------------------
# TODO: replace with the issue number of the backend-compiler report (the
# shuffle kernel fails in dbo-opt) once it is filed.
_BACKEND_SHUFFLE_ISSUE = "#TBD"


class TestEAShuffle:
    """A multi-arg op mixing STANDARD and staggered operands gets a shuffle.

    Both staggered arrangements (fp16 -> fp32 gives DL16_TO_FP32, fp32 -> fp16
    gives FP32_TO_DL16) are met by a STANDARD operand. ``test_ea_shuffle`` covers
    every shape group on the device with one pointwise op, and
    ``test_shuffle_pointwise`` covers every pointwise op and chain on one small
    shape. The remaining tests check the stick coordinate info that codegen
    emits for a shuffle (``gen_coord_info_value``) on the same shapes, without a
    device. Inputs are small integers, so device results compare exactly.
    """

    # Shape groups. Every shape is distinct; a group is kept small by taking one
    # representative per rank and per stick-count class.
    #
    # STICK_ONLY_SHAPES (module level, shared with test_roundtrip_stick_only_shape)
    # hold a scalar or all size-1 dims.

    # The stick dim is a whole number of 64-element sticks.
    _STICK_ALIGNED_SHAPES = [
        (64,),
        (128,),
        (2, 64),
        (4, 64),
        (4, 128),
        (2, 4, 64),
        (3, 5, 128),
        (2, 3, 4, 64),
        (64, 256),
        # an odd number (>= 3) of fp32 stick pairs when staggered as DL16_TO_FP32
        (4, 192),
        (4, 320),
    ]
    # The stick dim is shorter than a stick, or ends a stick early.
    _SMALL_N_SHAPES = [(2,), (16,), (32,), (63,), (2, 2), (4, 16), (4, 32)]
    # The stick dim ends inside a stick.
    _PARTIAL_STICK_SHAPES = [(44,), (100,), (7, 44), (4, 100), (2, 4, 44), (8, 16, 100)]

    # fp16 -> fp32: the staggered operand is the fp32 one (DL16_TO_FP32). Its
    # stick dim may hold an odd number of fp32 sticks, padded to a whole pair.
    _DL16_TO_FP32_SHAPES = (
        STICK_ONLY_SHAPES
        + _STICK_ALIGNED_SHAPES
        + _SMALL_N_SHAPES
        + _PARTIAL_STICK_SHAPES
        + [(68,), (4, 68), (4, 69), (4, 96), (4, 160)]
    )
    # fp32 -> fp16: the staggered operand is the fp16 one (FP32_TO_DL16). The
    # compiler only accepts an even number of 32-element fp32 sticks
    # (rescale_stl_for_dtype), so the odd-stick shapes above are left out.
    _FP32_TO_DL16_SHAPES = (
        STICK_ONLY_SHAPES
        + _STICK_ALIGNED_SHAPES
        + _SMALL_N_SHAPES
        + _PARTIAL_STICK_SHAPES
    )

    # Shapes the compiler rejects today, with a fragment of the error. They are
    # left out of the shuffle cases on the device and checked by
    # test_ea_shuffle_rejected. TODO: when a rejection is lifted, move its shape
    # to a group above so that it is checked by test_ea_shuffle.
    _DL16_TO_FP32_REJECTED = {
        # fewer than a whole pair of 32-element fp32 sticks
        (2,): "whole FP32 stick pairs",
        (16,): "whole FP32 stick pairs",
        (32,): "whole FP32 stick pairs",
        (2, 2): "whole FP32 stick pairs",
        (4, 16): "whole FP32 stick pairs",
        (4, 32): "whole FP32 stick pairs",
        # a rank >= 3 read of the shuffled stick dim is not affine
        (2, 4, 64): "cannot be divided safely",
        (2, 3, 4, 64): "cannot be divided safely",
    }
    _FP32_TO_DL16_REJECTED = {
        # one 32-element fp32 stick is not a whole 64-element fp16 stick
        (32,): "cannot rescale device layout",
        (4, 32): "cannot rescale device layout",
    }

    # (staggered arrangement, shape) over both conversion directions, for the
    # tests that need no device.
    _SHUFFLE_CASES = [
        pytest.param(
            EA.DL16_TO_FP32, s, id=f"dl16_to_fp32-{shapes2key((s,)) or 'scalar'}"
        )
        for s in _DL16_TO_FP32_SHAPES
    ] + [
        pytest.param(
            EA.FP32_TO_DL16, s, id=f"fp32_to_dl16-{shapes2key((s,)) or 'scalar'}"
        )
        for s in _FP32_TO_DL16_SHAPES
    ]

    # The same shapes on the device, without the rejected ones. A broadcast stick
    # (STICK_ONLY_SHAPES) gets no shuffle: its order is moot.
    _DL16_TO_FP32_RUN = sorted(
        set(_DL16_TO_FP32_SHAPES)
        - set(_DL16_TO_FP32_REJECTED)
        - set(STICK_ONLY_SHAPES),
        key=_DL16_TO_FP32_SHAPES.index,
    )
    _FP32_TO_DL16_RUN = sorted(
        set(_FP32_TO_DL16_SHAPES)
        - set(_FP32_TO_DL16_REJECTED)
        - set(STICK_ONLY_SHAPES),
        key=_FP32_TO_DL16_SHAPES.index,
    )
    _RUN_CASES = [
        pytest.param(EA.DL16_TO_FP32, s, id=f"dl16_to_fp32-{shapes2key((s,))}")
        for s in _DL16_TO_FP32_RUN
    ] + [
        pytest.param(EA.FP32_TO_DL16, s, id=f"fp32_to_dl16-{shapes2key((s,))}")
        for s in _FP32_TO_DL16_RUN
    ]
    _REJECTED_CASES = [
        pytest.param(EA.DL16_TO_FP32, s, m, id=f"dl16_to_fp32-{shapes2key((s,))}")
        for s, m in _DL16_TO_FP32_REJECTED.items()
    ] + [
        pytest.param(EA.FP32_TO_DL16, s, m, id=f"fp32_to_dl16-{shapes2key((s,))}")
        for s, m in _FP32_TO_DL16_REJECTED.items()
    ]

    # Pointwise ops and chains of them; ``s`` is the staggered operand and ``std``
    # the STANDARD one.
    POINTWISE_FNS = {
        "add": torch.add,
        "mul": torch.mul,
        "eq": torch.eq,
        "rsub": lambda s, std: std - s,
        "add_then_mul": lambda s, std: (s + std) * 2.0,
        "mul_then_add": lambda s, std: s * std + std,
        "rsub_then_mul": lambda s, std: (std - s) * 2.0,
    }

    # Sizes of a stick dim that hold an odd number of fp32 sticks.
    _ODD_FP32_STICK_SIZES = sorted(
        {s[-1] for s in _DL16_TO_FP32_SHAPES if s and -(-s[-1] // 32) % 2 == 1}
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

    @staticmethod
    def _stick_geometry(ea, shape):
        """``(size, elems_per_stick)`` of the stick dim of the staggered tensor."""
        return (shape[-1] if shape else 1), (32 if ea == EA.DL16_TO_FP32 else 64)

    @staticmethod
    def _coord_info(size, nsplits, eps, ea, device_size=None):
        return gen_coord_info_value(
            size=size,
            nsplits=nsplits,
            elems_per_stick=eps,
            is_stick_dim=True,
            element_arrangement=ea,
            is_ea_shuffle=True,
            stick_device_size=device_size,
        )

    @staticmethod
    def _levels_of(info):
        """The ``(cardinality, step)`` levels of a stick dim, innermost first."""
        folds = info["folds"]
        return [
            (attr["factor_"], func["Affine"]["alpha_"])
            for attr, func in zip(
                folds["dim_prop_attr"][3:], folds["dim_prop_func"][3:]
            )
        ][::-1]

    @staticmethod
    def _stick_offsets(levels):
        """Sorted element offsets visited by nested ``(cardinality, step)`` levels."""
        offsets = [0]
        for card, step in levels:
            offsets = [o + i * step for o in offsets for i in range(card)]
        return sorted(offsets)

    @staticmethod
    def _drop_trivial(levels):
        return [lvl for i, lvl in enumerate(levels) if i == 0 or lvl[0] != 1]

    @staticmethod
    def _expected_levels(size, eps):
        """Reference staggered levels, written out per elements-per-stick."""
        if eps == 64:
            return [(4, 1), (2, 32), (8, 4), (-(-size // 64), 64)]
        sticks = -(-size // 32)
        return [(4, 1), (8, 8), (2, 4), (sticks // 2, 64)]

    # -- on the device -----------------------------------------------------

    @pytest.mark.parametrize(
        "fp16",
        DtypeOpTable.fp16_types(),
        ids=lambda dt: str(dt).replace("torch.", ""),
    )
    @pytest.mark.parametrize("ea, shape", _RUN_CASES)
    def test_ea_shuffle(self, ea, shape, fp16):
        """A staggered operand added to a STANDARD one is shuffled, on every shape."""
        x, y = self._operands(ea, shape, fp16)

        def fn(x, y):
            # the converted operand is staggered and meets a STANDARD one
            return torch.add(x.to(y.dtype), y)

        result = _run(fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME))

        assert_ea(result, EA.STANDARD)
        assert torch.equal(result.cpu(), fn(x, y))

    @expects_raise
    @pytest.mark.parametrize(
        "fp16",
        DtypeOpTable.fp16_types(),
        ids=lambda dt: str(dt).replace("torch.", ""),
    )
    @pytest.mark.parametrize("ea, shape, match", _REJECTED_CASES)
    def test_ea_shuffle_rejected(self, ea, shape, match, fp16):
        """Shapes the compiler cannot shuffle yet fail with a stable error."""
        x, y = self._operands(ea, shape, fp16)

        def fn(x, y):
            return torch.add(x.to(y.dtype), y)

        with pytest.raises(InductorError, match=match):
            _run(fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME))

    @pytest.mark.parametrize(
        "fp16",
        DtypeOpTable.fp16_types(),
        ids=lambda dt: str(dt).replace("torch.", ""),
    )
    @pytest.mark.parametrize(
        "ea", [EA.DL16_TO_FP32, EA.FP32_TO_DL16], ids=["dl16_to_fp32", "fp32_to_dl16"]
    )
    @pytest.mark.parametrize(
        "pointwise", list(POINTWISE_FNS.values()), ids=list(POINTWISE_FNS)
    )
    def test_shuffle_pointwise(self, ea, fp16, pointwise):
        """Each pointwise op, and chains of them, read a shuffled operand."""
        x, y = self._operands(ea, (2, 64), fp16)

        def fn(x, y):
            return pointwise(x.to(y.dtype), y)

        result = _run(fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME))

        assert_ea(result, EA.STANDARD)
        assert torch.equal(result.cpu(), fn(x, y))

    # -- stick coordinate info of a shuffle (no device) --------------------

    @pytest.mark.parametrize("nsplits", [1, 4])
    @pytest.mark.parametrize("ea, shape", _SHUFFLE_CASES)
    def test_coord_info(self, ea, shape, nsplits):
        """The levels visit every element of the padded per-core stick dim once."""
        size, eps = self._stick_geometry(ea, shape)
        # the device pads the stick dim of each core to whole units (stick pairs
        # for fp32)
        per_core = -(-size // 64) * 64
        info = self._coord_info(size, nsplits, eps, ea, per_core * nsplits)
        funcs = info["folds"]["dim_prop_func"]
        attrs = info["folds"]["dim_prop_attr"]
        levels = self._levels_of(info)

        assert levels == self._drop_trivial(self._expected_levels(per_core, eps))
        assert info["elemArr"] == len(levels)
        assert len(funcs) == len(attrs) == 3 + len(levels)
        # the core fold advances by the logical per-core size
        assert funcs[0] == {"Affine": {"alpha_": size, "beta_": 0}}
        assert attrs[0] == {"factor_": nsplits, "label_": "core_fold"}
        assert attrs[1] == {"factor_": 1, "label_": "corelet_fold"}
        assert attrs[2] == {"factor_": 1, "label_": "row_fold"}
        # outermost level first
        assert [a["label_"] for a in attrs[3:]] == [
            f"elem_arr_{i}" for i in reversed(range(len(levels)))
        ]
        assert self._stick_offsets(levels) == list(range(per_core))

    @pytest.mark.parametrize("ea, shape", _SHUFFLE_CASES)
    def test_coord_info_same_levels(self, ea, shape):
        size, eps = self._stick_geometry(ea, shape)
        per_core = -(-size // 64) * 64
        assert self._coord_info(
            size, 1, eps, EA.DL16_TO_FP32, per_core
        ) == self._coord_info(size, 1, eps, EA.FP32_TO_DL16, per_core)

    @pytest.mark.parametrize("ea, shape", _SHUFFLE_CASES)
    def test_coord_info_standard(self, ea, shape):
        """The STANDARD side of a shuffle is a plain stick, then the sticks."""
        size, eps = self._stick_geometry(ea, shape)
        per_core = -(-size // eps) * eps
        levels = self._levels_of(self._coord_info(size, 1, eps, EA.STANDARD, per_core))
        assert levels == self._drop_trivial([(eps, 1), (per_core // eps, eps)])
        assert self._stick_offsets(levels) == list(range(per_core))

    @pytest.mark.parametrize("ea", STAGGERED_EAS, ids=lambda ea: ea.name.lower())
    @pytest.mark.parametrize(
        "eps, size, expected",
        [
            (64, 64, [(4, 1), (2, 32), (8, 4)]),
            (64, 128, [(4, 1), (2, 32), (8, 4), (2, 64)]),
            (32, 64, [(4, 1), (8, 8), (2, 4)]),
            (32, 128, [(4, 1), (8, 8), (2, 4), (2, 64)]),
        ],
        ids=["fp16-64", "fp16-128", "fp32-64", "fp32-128"],
    )
    def test_coord_info_levels(self, ea, eps, size, expected):
        info = self._coord_info(size, 1, eps, ea, size)
        assert self._levels_of(info) == expected

    @pytest.mark.parametrize("ea", STAGGERED_EAS, ids=lambda ea: ea.name.lower())
    @pytest.mark.parametrize("eps", [32, 64])
    @pytest.mark.parametrize("nsplits", [1, 4, 32])
    def test_coord_info_non_stick_dim(self, nsplits, eps, ea):
        got = gen_coord_info_value(
            size=8,
            nsplits=nsplits,
            elems_per_stick=eps,
            is_stick_dim=False,
            element_arrangement=ea,
            is_ea_shuffle=True,
            stick_device_size=256,
        )
        assert got["elemArr"] == 1
        assert got["folds"]["dim_prop_attr"][0]["factor_"] == nsplits

    @expects_raise
    @pytest.mark.parametrize("ea", STAGGERED_EAS, ids=lambda ea: ea.name.lower())
    @pytest.mark.parametrize("size", _ODD_FP32_STICK_SIZES)
    def test_coord_info_odd_sticks_raises(self, size, ea):
        with pytest.raises(Unsupported, match="whole FP32 stick pairs"):
            self._coord_info(size, 1, 32, ea)

    @expects_raise
    @pytest.mark.parametrize("ea", STAGGERED_EAS, ids=lambda ea: ea.name.lower())
    @pytest.mark.parametrize("eps", [16, 128])
    def test_coord_info_bad_eps_raises(self, eps, ea):
        with pytest.raises(Unsupported, match="not defined for"):
            self._coord_info(128, 1, eps, ea)


# ---------------------------------------------------------------------------
# Shuffle optimization (optimize_shuffling)
# ---------------------------------------------------------------------------
class _ShuffleRecorder(CustomPreSchedulingPasses):
    """Pre-scheduling pipeline that records the EA shuffles around the pass.

    ``records["before"]`` is taken right after ``insert_ea_shuffle`` (what it
    inserted) and ``records["after"]`` right after ``optimize_shuffling``. Each
    is the list of source dtypes, one entry per shuffle copy.
    """

    records: dict = {}

    def __init__(self):
        super().__init__()
        self.passes = [self._record(p) for p in self.passes]

    @staticmethod
    def _shuffle_source_dtypes(graph):
        dtypes = []
        for op in graph.operations:
            if not getattr(op, "ea_shuffle", False):
                continue
            (src,) = {
                d.name for d in op.get_read_writes().reads if isinstance(d, MemoryDep)
            }
            dtypes.append(graph.get_buffer(src).get_dtype())
        return dtypes

    @staticmethod
    def _shuffle_layouts(graph):
        """``(source layout, shuffle layout)`` of every shuffle copy."""
        pairs = []
        for op in graph.operations:
            if not getattr(op, "ea_shuffle", False):
                continue
            (src,) = {
                d.name for d in op.get_read_writes().reads if isinstance(d, MemoryDep)
            }
            pairs.append(
                (
                    graph.get_buffer(src).get_layout().device_layout,
                    op.get_layout().device_layout,
                )
            )
        return pairs

    @classmethod
    def _record(cls, pass_fn):
        key = {insert_ea_shuffle: "before", optimize_shuffling: "after"}.get(pass_fn)
        if key is None:
            return pass_fn

        def wrapped(graph):
            pass_fn(graph)
            cls.records[key] = cls._shuffle_source_dtypes(graph)
            if key == "before":
                cls.records["layouts"] = cls._shuffle_layouts(graph)

        wrapped.__name__ = pass_fn.__name__
        return wrapped


def _compile_recording(fn, *args):
    """Compile ``fn`` and return ``(result, records)`` from ``_ShuffleRecorder``."""
    _ShuffleRecorder.records = {}
    torch.compiler.reset()
    with (
        inductor_config.patch(force_disable_caches=True),
        patch.object(passes, "CustomPreSchedulingPasses", _ShuffleRecorder),
    ):
        result = torch.compile(fn)(*args)
    torch.compiler.reset()
    return result, _ShuffleRecorder.records


# ---------------------------------------------------------------------------
# Device-free graphs for insert_ea_shuffle / optimize_shuffling
#
# The passes run on committed layouts only, so a graph of mock ``ComputedBuffer``
# ops with real ``FixedTiledLayout``s is enough. Only the IR rewrites (inserting
# the shuffle copy, re-pointing a reader) are replaced by a fake that edits the
# ``reads`` list of the mock op; tests assert the resulting ``(name, reads)``.
# ---------------------------------------------------------------------------
def _ea_buf(
    name, ea=EA.STANDARD, reads=(), dtype=torch.float16, shuffle=False, **attrs
):
    """A mock pointwise ``ComputedBuffer`` of shape (4, 128) in arrangement ``ea``.

    ``attrs`` override attributes of the mock (``data``, ``loop_info``, ...);
    ``broadcast=True`` makes the stick a broadcast (``stride_map[-1] == -1``).
    """
    broadcast = attrs.pop("broadcast", False)
    stl = SpyreTensorLayout([4, 128], dtype).with_element_arrangement(ea)
    if broadcast:
        stl = SpyreTensorLayout(
            device_size=list(stl.device_size),
            stride_map=[*stl.stride_map[:-1], -1],
            device_dtype=stl.device_dtype,
            element_arrangement=ea,
        )
    op = MagicMock(spec=ComputedBuffer)
    op.layout = FixedTiledLayout(torch.device("spyre"), dtype, [4, 128], [128, 1], stl)
    op.data = MagicMock(spec=Pointwise)
    op.loop_info = None
    op._read_copy_elision_record = None
    op.ea_shuffle = shuffle
    op.reads = list(reads)
    op.get_name.return_value = name
    op.get_layout.side_effect = lambda: op.layout
    op.get_dtype.return_value = dtype
    op.get_read_names.side_effect = lambda: list(op.reads)
    op.get_read_writes.side_effect = lambda: SimpleNamespace(
        reads=[MemoryDep(r, sympy.Integer(0), (), ()) for r in op.reads],
        writes=[MemoryDep(name, sympy.Integer(0), (), ())],
    )
    for key, value in attrs.items():
        setattr(op, key, value)
    return op


@pytest.fixture
def run_pass(monkeypatch):
    """``run_pass(pass_fn, ops, outputs=())`` runs a pass on a fake graph and
    returns the ``(name, reads)`` of the operations left in it."""

    def fake_insert_shuffle_op(graph, src_name, target_ea, before_op):
        shuffle = _ea_buf(f"shuffle_of_{src_name}", target_ea, [src_name], shuffle=True)
        graph.operations.insert(graph.operations.index(before_op), shuffle)
        return shuffle

    def fake_redirect(op, name_map, operations, **_):
        op.reads = [name_map.get(r, r) for r in op.reads]
        return op

    monkeypatch.setattr(
        element_arrangement, "_insert_shuffle_op", fake_insert_shuffle_op
    )
    monkeypatch.setattr(
        element_arrangement, "redirect_computed_buffer_reads", fake_redirect
    )
    monkeypatch.setattr(
        element_arrangement, "indirect_info_from_op", lambda op: (set(), {}, None)
    )

    def run(pass_fn, ops, outputs=()):
        graph = SimpleNamespace(
            operations=list(ops), removed_buffers=set(), graph_inputs={}
        )
        graph.get_output_names = lambda: list(outputs)
        graph.get_buffer = lambda n: next(
            o for o in graph.operations if o.get_name() == n
        )
        pass_fn(graph)
        return [(op.get_name(), op.reads) for op in graph.operations]

    return run


class TestOptimizeShuffling:
    """``optimize_shuffling`` removes or cheapens the shuffles
    ``insert_ea_shuffle`` adds.

    The structural assertions look at the IR between ``insert_ea_shuffle`` and
    ``optimize_shuffling``; each test also checks the numerical result.
    """

    SHAPE = (4, 128)

    @staticmethod
    def _run_recording(fn, *args):
        """Compile ``fn``; return ``(result, before, after)`` shuffle source dtypes."""
        result, records = _compile_recording(fn, *args)
        return result, records["before"], records["after"]

    def test_no_shuffle(self):
        """Operands that share one arrangement need no shuffle at all."""

        def fn(x, y):
            return x.to(torch.float32) + y.to(torch.float32)

        x = cached_randn(self.SHAPE, dtype=torch.float16)
        y = cached_randn(self.SHAPE, dtype=torch.float16)

        result, before, after = self._run_recording(
            fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME)
        )

        assert before == [] and after == []
        assert_val(fn, (x, y), result, rtol=1e-2, atol=1e-2)

    def test_shuffle_kept(self):
        """A staggered fp16 operand meeting a STANDARD one still needs its shuffle."""

        def fn(x, y):
            return x.to(torch.float16) + y

        x = cached_randn(self.SHAPE, dtype=torch.float32)
        y = cached_randn(self.SHAPE, dtype=torch.float16)

        result, before, after = self._run_recording(
            fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME)
        )

        # The shuffle reads the fp32 -> fp16 cast; a downcast has no narrower
        # input to move the shuffle onto, so it is neither removed nor rewritten.
        assert before == [torch.float16]
        assert after == [torch.float16]
        assert_ea(result, EA.STANDARD)
        assert_val(fn, (x, y), result, rtol=1e-2, atol=1e-2)

    def test_sink_through_upcast(self):
        """``fp16 -> fp32`` then a shuffle becomes a shuffle of the fp16 input."""

        def fn(x, y):
            return (x.to(torch.float32) + y) * 2.0

        x = cached_randn(self.SHAPE, dtype=torch.float16)
        y = cached_randn(self.SHAPE, dtype=torch.float32)

        result, before, after = self._run_recording(
            fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME)
        )

        # Before: the staggered fp32 cast output is shuffled (4 B/element).
        # After: the fp16 input is shuffled and the cast itself restores STANDARD.
        assert before == [torch.float32]
        assert after == [torch.float16]
        assert_ea(result, EA.STANDARD)
        assert_val(fn, (x, y), result, rtol=1e-2, atol=1e-2)

    def test_merge_duplicates(self):
        """Two consumers of one staggered value share a single shuffle."""

        def fn(x, y):
            a = x.to(torch.float16)
            return a + y, a * y

        x = cached_randn(self.SHAPE, dtype=torch.float32)
        y = cached_randn(self.SHAPE, dtype=torch.float16)

        (r1, r2), before, after = self._run_recording(
            fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME)
        )

        assert len(before) >= len(after) == 1
        for r, ref in zip((r1, r2), fn(x, y)):
            assert_ea(r, EA.STANDARD)
            torch.testing.assert_close(r.cpu(), ref, rtol=1e-2, atol=1e-2)

    @staticmethod
    def _shuffle(name, src, ea=EA.STANDARD, **kwargs):
        return _ea_buf(name, ea, [src], shuffle=True, **kwargs)

    def test_dead(self, run_pass):
        ops = [_ea_buf("a", EA.DL16_TO_FP32), self._shuffle("s", "a")]
        assert run_pass(optimize_shuffling, ops) == [("a", [])]

    def test_dead_output_kept(self, run_pass):
        ops = [_ea_buf("a", EA.DL16_TO_FP32), self._shuffle("s", "a")]
        assert run_pass(optimize_shuffling, ops, outputs=["s"]) == [
            ("a", []),
            ("s", ["a"]),
        ]

    def test_noop(self, run_pass):
        """The source already has the target layout, so readers read it."""
        ops = [
            _ea_buf("a"),
            self._shuffle("s", "a"),
            _ea_buf("out", EA.STANDARD, ["s"]),
        ]
        assert run_pass(optimize_shuffling, ops, outputs=["out"]) == [
            ("a", []),
            ("out", ["a"]),
        ]

    def test_required_kept(self, run_pass):
        ops = [
            _ea_buf("a", EA.DL16_TO_FP32),
            self._shuffle("s", "a"),
            _ea_buf("out", EA.STANDARD, ["s"]),
        ]
        assert run_pass(optimize_shuffling, ops, outputs=["out"]) == [
            ("a", []),
            ("s", ["a"]),
            ("out", ["s"]),
        ]

    def test_chain_cancels(self, run_pass):
        """``STANDARD -> staggered -> STANDARD`` leaves no shuffle."""
        ops = [
            _ea_buf("a"),
            self._shuffle("s1", "a", EA.DL16_TO_FP32),
            self._shuffle("s2", "s1"),
            _ea_buf("out", EA.STANDARD, ["s2"]),
        ]
        assert run_pass(optimize_shuffling, ops, outputs=["out"]) == [
            ("a", []),
            ("out", ["a"]),
        ]

    def test_chain_collapses(self, run_pass):
        """``A -> B -> C`` through two shuffles becomes ``A -> C``."""
        ops = [
            _ea_buf("a", EA.FP32_TO_DL16),
            self._shuffle("s1", "a", EA.DL16_TO_FP32),
            self._shuffle("s2", "s1"),
            _ea_buf("out", EA.STANDARD, ["s2"]),
        ]
        result = run_pass(optimize_shuffling, ops, outputs=["out"])
        assert ("s2", ["a"]) in result and "s1" not in dict(result)

    def test_dup_merged(self, run_pass):
        ops = [
            _ea_buf("a", EA.DL16_TO_FP32),
            self._shuffle("s1", "a"),
            self._shuffle("s2", "a"),
            _ea_buf("out1", EA.STANDARD, ["s1"]),
            _ea_buf("out2", EA.STANDARD, ["s2"]),
        ]
        result = dict(run_pass(optimize_shuffling, ops, outputs=["out1", "out2"]))
        assert result["out1"] == result["out2"] == ["s1"]
        assert "s2" not in result

    def test_upcast_sunk(self, run_pass):
        """``x(fp16) -upcast-> s -shuffle-> t`` shuffles the narrow ``x`` instead."""
        fp32 = torch.float32
        ops = [
            _ea_buf("x"),
            _ea_buf("s", EA.DL16_TO_FP32, ["x"], dtype=fp32),
            self._shuffle("t", "s", dtype=fp32),
            _ea_buf("out", EA.STANDARD, ["t"], dtype=fp32),
        ]
        result = run_pass(optimize_shuffling, ops, outputs=["out"])
        names = dict(result)
        assert "t" not in names and "shuffle_of_x" in names

    @pytest.mark.parametrize(
        "attrs",
        [
            {"layout": MagicMock(spec=MutationLayoutSHOULDREMOVE)},
            {"data": MagicMock(spec=Scatter)},
            {"loop_info": object()},
            {"_read_copy_elision_record": object()},
        ],
        ids=["mutation", "scatter", "coarse_tiled", "copy_elision"],
    )
    def test_unredirectable_reader(self, run_pass, attrs):
        """A no-op shuffle stays if one of its readers cannot be re-pointed."""
        ops = [
            _ea_buf("a"),
            self._shuffle("s", "a"),
            _ea_buf("ok", EA.STANDARD, ["s"]),
            _ea_buf("bad", EA.STANDARD, ["s"], **attrs),
        ]
        result = dict(run_pass(optimize_shuffling, ops, outputs=["ok", "bad"]))
        assert result["s"] == ["a"] and result["ok"] == ["s"]

    def test_reader_reads_source(self, run_pass):
        """Renaming would merge two reads of one op, so the shuffle stays."""
        ops = [
            _ea_buf("a"),
            self._shuffle("s", "a"),
            _ea_buf("out", EA.STANDARD, ["s", "a"]),
        ]
        result = dict(run_pass(optimize_shuffling, ops, outputs=["out"]))
        assert result["s"] == ["a"] and result["out"] == ["s", "a"]

    def test_non_computed_reader(self, run_pass):
        extern = MagicMock(spec=Operation)
        extern.reads = ["s"]
        extern.get_name = lambda: "ext"
        extern.get_read_writes = lambda: SimpleNamespace(
            reads=[MemoryDep("s", sympy.Integer(0), (), ())], writes=[]
        )
        ops = [_ea_buf("a"), self._shuffle("s", "a"), extern]
        result = dict(run_pass(optimize_shuffling, ops, outputs=["ext"]))
        assert result["s"] == ["a"] and result["ext"] == ["s"]


# ---------------------------------------------------------------------------
# insert_ea_shuffle: exactness on device and the geometry of the inserted copy
# ---------------------------------------------------------------------------
class TestInsertEAShuffle:
    """Mixed staggered/STANDARD pointwise ops compile and match CPU exactly."""

    @staticmethod
    def _staggered_add(a, b):
        """``a`` (fp16) upcast to a staggered fp32 meets the STANDARD fp32 ``b``."""
        return a.float() + b

    @staticmethod
    def _staggered_add_inputs(shape):
        return _small_int_fp16(shape, 7), _small_int_fp16(shape, 5, base=3).float()

    @staticmethod
    def _assert_exact_on_device(fn, *cpu_args):
        """Compile ``fn`` on the device and compare with CPU bit for bit."""
        torch.compiler.reset()
        with inductor_config.patch(force_disable_caches=True):
            result = torch.compile(fn)(*(x.to(DEVICE_NAME) for x in cpu_args))
        torch.compiler.reset()
        assert torch.equal(result.cpu(), fn(*cpu_args))

    PROGRAMS = {
        "fp32_to_fp16_plus_fp16": (
            torch.float32,
            torch.float16,
            lambda a, b: a.to(torch.float16) + b,
        ),
        "fp16_times_fp32_to_fp16": (
            torch.float32,
            torch.float16,
            lambda a, b: b * a.to(torch.float16),
        ),
        "fp16_to_fp32_plus_fp32": (
            torch.float16,
            torch.float32,
            lambda a, b: a.float() + b,
        ),
        "fp16_to_fp32_clone_plus_fp32": (
            torch.float16,
            torch.float32,
            lambda a, b: a.float().clone() + b,
        ),
    }
    SHAPES = [(2, 64), (4, 128), (64, 256)]

    @pytest.mark.parametrize(
        "program, shape",
        [
            pytest.param(
                p,
                s,
                id=f"{p}-{shapes2key((s,))}",
                marks=pytest.mark.xfail(
                    p.startswith("fp16_to_fp32") and s == (64, 256),
                    strict=True,
                    raises=InductorError,
                    reason=(
                        "the backend compiler: Cannot find a valid minimum value"
                        f"{_BACKEND_SHUFFLE_ISSUE}"
                    ),
                ),
            )
            for p, s in itertools.product(PROGRAMS, SHAPES)
        ],
    )
    def test_mixed_ea_pointwise(self, program, shape):
        a_dtype, b_dtype, fn = self.PROGRAMS[program]
        a = _small_int_fp16(shape, 7).to(a_dtype)
        b = _small_int_fp16(shape, 5, base=3).to(b_dtype)
        self._assert_exact_on_device(fn, a, b)

    def test_shuffle_geometry(self):
        """The copy has the committed input layout, only the arrangement changes."""
        a, b = self._staggered_add_inputs((4, 128))
        _, records = _compile_recording(
            self._staggered_add, a.to(DEVICE_NAME), b.to(DEVICE_NAME)
        )
        assert records["layouts"], "no shuffle was inserted"
        for src, shuffle in records["layouts"]:
            assert src.element_arrangement in STAGGERED_EAS
            assert shuffle.element_arrangement == EA.STANDARD
            assert list(shuffle.device_size) == list(src.device_size)
            assert list(shuffle.stride_map) == list(src.stride_map)
            assert shuffle.device_dtype == src.device_dtype

    # The fp32 widths and the odd-pair backend failures are covered by
    # TestEAShuffle. Here the shuffle's input and output land in different
    # memories (one in LX, one in HBM) when the LX planner cannot fit both.
    WIDE_SHAPES = [(64, 151936), (2, 128256)]

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param(
                s,
                marks=pytest.mark.xfail(
                    spyre_config.lx_planning,
                    strict=True,
                    raises=InductorError,
                    reason=(
                        "the backend compiler (dbo-opt) rejects a shuffle whose "
                        "input and output are in different memories (LX and HBM), "
                        f"{_BACKEND_SHUFFLE_ISSUE}"
                    ),
                ),
            )
            for s in WIDE_SHAPES
        ],
        ids=lambda s: shapes2key((s,)),
    )
    def test_wide_rows_lx(self, shape):
        self._assert_exact_on_device(
            self._staggered_add, *self._staggered_add_inputs(shape)
        )

    @pytest.mark.parametrize(
        "shape", WIDE_SHAPES, ids=[shapes2key((s,)) for s in WIDE_SHAPES]
    )
    def test_wide_rows_no_lx(self, shape):
        with spyre_config.patch(lx_planning=False):
            self._assert_exact_on_device(
                self._staggered_add, *self._staggered_add_inputs(shape)
            )

    @pytest.mark.parametrize(
        "in_ea, out_ea, kwargs, shuffled",
        [
            (EA.DL16_TO_FP32, EA.STANDARD, {}, True),
            (EA.FP32_TO_DL16, EA.STANDARD, {}, True),
            # A broadcast stick reads one element of the stick: order is moot.
            (EA.DL16_TO_FP32, EA.STANDARD, {"broadcast": True}, False),
            (EA.STANDARD, EA.STANDARD, {}, False),
            # The op itself produces a staggered result: nothing to restore.
            (EA.DL16_TO_FP32, EA.DL16_TO_FP32, {}, False),
            (EA.DL16_TO_FP32, EA.FP32_TO_DL16, {}, False),
        ],
        ids=[
            "dl16_to_fp32",
            "fp32_to_dl16",
            "broadcast_stick",
            "standard_input",
            "staggered_output_same",
            "staggered_output_other",
        ],
    )
    def test_operand(self, run_pass, in_ea, out_ea, kwargs, shuffled):
        ops = [
            _ea_buf("a", in_ea, **kwargs),
            _ea_buf("b"),
            _ea_buf("out", out_ea, ["a", "b"]),
        ]
        reads = dict(run_pass(insert_ea_shuffle, ops))
        assert reads["out"] == (["shuffle_of_a", "b"] if shuffled else ["a", "b"])

    def test_single_operand(self, run_pass):
        ops = [_ea_buf("a", EA.DL16_TO_FP32), _ea_buf("out", EA.STANDARD, ["a"])]
        assert dict(run_pass(insert_ea_shuffle, ops))["out"] == ["a"]

    def test_both_staggered(self, run_pass):
        ops = [
            _ea_buf("a", EA.DL16_TO_FP32),
            _ea_buf("b", EA.DL16_TO_FP32),
            _ea_buf("out", EA.STANDARD, ["a", "b"]),
        ]
        reads = dict(run_pass(insert_ea_shuffle, ops))
        assert reads["out"] == ["shuffle_of_a", "shuffle_of_b"]

    @expects_raise
    def test_coarse_tiled_raises(self, run_pass):
        ops = [
            _ea_buf("a", EA.DL16_TO_FP32),
            _ea_buf("b"),
            _ea_buf("out", EA.STANDARD, ["a", "b"], loop_info=object()),
        ]
        with pytest.raises(Unsupported, match="coarse-tiled loop"):
            run_pass(insert_ea_shuffle, ops)


# ---------------------------------------------------------------------------
# validate_ops is a read-only check
# ---------------------------------------------------------------------------
class TestValidateOps:
    @staticmethod
    def _stl(ea):
        return SpyreTensorLayout(
            device_size=[2, 64],
            stride_map=[64, 1],
            device_dtype=get_device_dtype(torch.float16),
            element_arrangement=ea,
        )

    @staticmethod
    def _fake_op(buffer_names, pointwise=True):
        """A pointwise op that reads ``buffer_names``."""
        op = MagicMock()
        op.data = MagicMock(spec=Pointwise) if pointwise else MagicMock()
        op.data.origins = []
        deps = []
        for name in buffer_names:
            dep = MagicMock(spec=MemoryDep)
            dep.name = name
            deps.append(dep)
        op.get_read_writes.return_value = SimpleNamespace(reads=deps, writes=[])
        return op

    def _validate(self, ops, buffer_eas):
        buffers = {
            name: SimpleNamespace(layouts=[self._stl(ea) for ea in eas])
            for name, eas in buffer_eas.items()
        }
        graph = SimpleNamespace(operations=list(ops), get_buffer=buffers.__getitem__)
        before = list(graph.operations)
        with V.set_graph_handler(graph):
            validate_ops(graph)
        assert graph.operations == before, "validate_ops must not edit the graph"

    def test_mixed_ok(self):
        """Case 3.3 (one staggered EA with STANDARD operands) is no longer an error."""
        self._validate(
            [self._fake_op(["a", "b"])],
            {"a": [EA.STANDARD], "b": [EA.DL16_TO_FP32]},
        )

    def test_same_ea_ok(self):
        self._validate(
            [self._fake_op(["a", "b"])],
            {"a": [EA.FP32_TO_DL16], "b": [EA.FP32_TO_DL16]},
        )

    def test_single_operand_skipped(self):
        self._validate([self._fake_op(["a"])], {"a": [EA.DL16_TO_FP32]})

    def test_non_pointwise_skipped(self):
        self._validate(
            [self._fake_op(["a", "b"], pointwise=False)],
            {"a": [EA.DL16_TO_FP32], "b": [EA.FP32_TO_DL16]},
        )

    @expects_raise
    def test_two_staggered_rejected(self):
        with pytest.raises(Unsupported, match="Incompatible ElementArrangement"):
            self._validate(
                [self._fake_op(["a", "b"])],
                {"a": [EA.DL16_TO_FP32], "b": [EA.FP32_TO_DL16]},
            )

    @expects_raise
    def test_exx2_rejected(self):
        with pytest.raises(Unsupported, match="Incompatible ElementArrangement"):
            self._validate(
                [self._fake_op(["a", "b"])],
                {"a": [EA.EXX2], "b": [EA.STANDARD]},
            )


# Made with Bob
