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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch._inductor import config as inductor_config
from torch._inductor.dependencies import MemoryDep
from torch._inductor.exc import InductorError
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
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
from torch_spyre._inductor import passes
from torch_spyre._inductor.codegen.compute_ops import (
    _gen_shuffle_stick_coord_info,
    _shuffle_stick_levels,
    gen_coord_info_value,
)
from torch_spyre._inductor.constants import DEVICE_NAME, STAGGERED_EAS
from torch_spyre._inductor.dtype_ops import DtypeOpTable
from torch_spyre._inductor.element_arrangement import (
    _can_redirect_readers,
    _needs_shuffle,
    validate_ops,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.passes import CustomPreSchedulingPasses, _get_pass_name
from utils_inductor import cached_randn, shapes2key, expects_raise


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


def _shape_id(shape):
    return shapes2key((shape,)) or "scalar"


def _dedupe(*shape_lists):
    return list(dict.fromkeys(s for shapes in shape_lists for s in shapes))


# Widths of the fp32 stick dim from the review: 32, 69 and 96 hold an odd number
# of fp32 sticks, the odd-pair widths (192 ... 704) fail in the backend compiler.
REVIEW_WIDTHS = [32, 64, 69, 96, 100, 128, 192, 256, 320, 384, 448, 576, 640, 704]
REVIEW_WIDTHS += [768, 896, 1024]

_STICK_ALIGNED_SHAPES = [
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
_SMALL_N_SHAPES = [(2,), (3,), (4,), (16,), (32,), (63,), (2, 2), (4, 16), (4, 32)]
_BASE_SHAPES = _STICK_ALIGNED_SHAPES + _SMALL_N_SHAPES + STICK_ONLY_SHAPES
_REVIEW_SHAPES = [(2, 64), (4, 128), (64, 256)]

# Non-stick-aligned shapes (the stick dim ends inside a stick).
_PARTIAL_STICK_SHAPES = [
    (44,),
    (100,),
    (7, 44),
    (3, 100),
    (4, 44),
    (4, 100),
    (2, 4, 44),
    (8, 16, 100),
    (2, 3, 4, 44),
]

# fp16 -> fp32: the staggered operand is the fp32 one (DL16_TO_FP32).
_DL16_TO_FP32_SHAPES = _dedupe(
    _BASE_SHAPES,
    _REVIEW_SHAPES,
    _PARTIAL_STICK_SHAPES,
    # Partial last stick or pair
    [(68,), (4, 68), (8, 16, 68), (2, 96), (4, 96), (8, 16, 96)],
    [(4, 160), (8, 16, 160)],
    [(r, w) for r in (1, 4, 64) for w in REVIEW_WIDTHS],
)
# fp32 -> fp16: the compiler only accepts an even number of 32-element fp32
# sticks (rescale_stl_for_dtype), so 68, 96, 160 and 32 wide are left out.
_FP32_TO_DL16_SHAPES = _dedupe(
    _BASE_SHAPES,
    _REVIEW_SHAPES,
    _PARTIAL_STICK_SHAPES,
    [(r, w) for r in (1, 4, 64) for w in REVIEW_WIDTHS if -(-w // 32) % 2 == 0],
)


def _fp32_stick_pairs(width):
    """Stick pairs in a row of ``width`` fp32 elements, padded to whole pairs."""
    sticks = -(-width // 32)
    return -(-sticks // 2)


def _odd_fp32_pairs(shape):
    """True if a row of an fp32 staggered tensor holds an odd number (>= 3) of
    stick pairs (a backend failure)."""
    if not shape:
        return False
    pairs = _fp32_stick_pairs(shape[-1])
    return pairs >= 3 and pairs % 2 == 1


# direction -> (dtype of the converted operand, dtype of the STANDARD operand,
#               conversion to the staggered dtype), given the fp16 flavour
_SHUFFLE_DIRECTIONS = {
    "dl16_to_fp32": (lambda f: f, lambda f: torch.float32, lambda f: torch.float32),
    "fp32_to_dl16": (lambda f: torch.float32, lambda f: f, lambda f: f),
}
_SHUFFLE_SHAPES = {
    "dl16_to_fp32": _DL16_TO_FP32_SHAPES,
    "fp32_to_dl16": _FP32_TO_DL16_SHAPES,
}


def _shuffle_cases():
    cases = []
    for direction, shapes in _SHUFFLE_SHAPES.items():
        for shape in shapes:
            marks = []
            if direction == "dl16_to_fp32" and _odd_fp32_pairs(shape):
                marks.append(
                    pytest.mark.xfail(
                        strict=True,
                        raises=InductorError,
                        reason=(
                            "the backend compiler (dbo-opt) rejects the shuffle "
                            "kernel when a row holds an odd number (>= 3) of fp32 "
                            f"stick pairs, {_BACKEND_SHUFFLE_ISSUE}"
                        ),
                    )
                )
            cases.append(
                pytest.param(
                    direction,
                    shape,
                    marks=marks,
                    id=f"{direction}-{_shape_id(shape)}",
                )
            )
    return cases


_SHUFFLE_CASES = _shuffle_cases()


def _shuffle_operands(direction, shape, fp16):
    """Inputs and conversion for ``direction``: x is converted, y is STANDARD."""
    x_dtype, y_dtype, to_dtype = (f(fp16) for f in _SHUFFLE_DIRECTIONS[direction])
    x = _small_int_fp16(shape, 7).to(x_dtype)
    y = _small_int_fp16(shape, 5, base=3).to(y_dtype)
    return x, y, lambda t: t.to(to_dtype)


class TestEAShuffle:
    """A multi-arg op mixing STANDARD and staggered operands gets a shuffle.

    Both staggered arrangements (fp16 -> fp32 gives DL16_TO_FP32, fp32 -> fp16
    gives FP32_TO_DL16) are met by a STANDARD operand, in both operand orders,
    over aligned, partial-stick, small, stick-only and review shapes. Inputs
    are small integers, so the results compare exactly.
    """

    OPS_DICT = {
        "add": torch.add,
        "mul": torch.mul,
        "eq": torch.eq,
        # the STANDARD operand comes first
        "rsub": lambda s, y: y - s,
    }

    @pytest.mark.parametrize("fp16", DtypeOpTable.fp16_types())
    @pytest.mark.parametrize("direction, shape", _SHUFFLE_CASES)
    @pytest.mark.parametrize(
        "op",
        list(OPS_DICT.values()),
        ids=list(OPS_DICT.keys()),
    )
    def test_shuffle_implicit_pointwise(self, direction, shape, fp16, op):
        x, y, convert = _shuffle_operands(direction, shape, fp16)

        def fn(x, y_std):
            # the converted operand is staggered and meets a STANDARD one
            return op(convert(x), y_std)

        result = _run(fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME))

        assert_ea(result, EA.STANDARD)
        assert torch.equal(result.cpu(), fn(x, y))

    @pytest.mark.parametrize("fp16", DtypeOpTable.fp16_types())
    @pytest.mark.parametrize("direction, shape", _SHUFFLE_CASES)
    def test_shuffle_feeds_downstream_op(self, direction, shape, fp16):
        x, y, convert = _shuffle_operands(direction, shape, fp16)

        def fn(x, y):
            # staggered -> shuffled to std for the add with y (std) -> next op
            return (convert(x) + y) * 2.0

        result = _run(fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME))

        assert_ea(result, EA.STANDARD)
        assert torch.equal(result.cpu(), fn(x, y))


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
        name = _get_pass_name(pass_fn)
        key = {"insert_ea_shuffle": "before", "optimize_shuffling": "after"}.get(name)
        if key is None:
            return pass_fn

        def wrapped(graph):
            pass_fn(graph)
            cls.records[key] = cls._shuffle_source_dtypes(graph)
            if key == "before":
                cls.records["layouts"] = cls._shuffle_layouts(graph)

        wrapped.__name__ = name
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


def _run_recording(fn, *args):
    """Compile ``fn`` and return ``(result, before, after)`` shuffle source dtypes."""
    result, records = _compile_recording(fn, *args)
    return result, records["before"], records["after"]


class TestEAShuffleOptimize:
    """``optimize_shuffling`` removes or cheapens the shuffles
    ``insert_ea_shuffle`` adds.

    The structural assertions look at the IR between ``insert_ea_shuffle`` and
    ``optimize_shuffling``; each test also checks the numerical result.
    """

    SHAPE = (4, 128)

    def test_no_shuffle_when_arrangements_match(self):
        """Operands that share one arrangement need no shuffle at all."""

        def fn(x, y):
            return x.to(torch.float32) + y.to(torch.float32)

        x = cached_randn(self.SHAPE, dtype=torch.float16)
        y = cached_randn(self.SHAPE, dtype=torch.float16)

        result, before, after = _run_recording(fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME))

        assert before == [] and after == []
        assert_val(fn, (x, y), result, rtol=1e-2, atol=1e-2)

    def test_required_shuffle_is_kept(self):
        """A staggered fp16 operand meeting a STANDARD one still needs its shuffle."""

        def fn(x, y):
            return x.to(torch.float16) + y

        x = cached_randn(self.SHAPE, dtype=torch.float32)
        y = cached_randn(self.SHAPE, dtype=torch.float16)

        result, before, after = _run_recording(fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME))

        # The shuffle reads the fp32 -> fp16 cast; a downcast has no narrower
        # input to move the shuffle onto, so it is neither removed nor rewritten.
        assert before == [torch.float16]
        assert after == [torch.float16]
        assert_ea(result, EA.STANDARD)
        assert_val(fn, (x, y), result, rtol=1e-2, atol=1e-2)

    def test_shuffle_sinks_through_upcast(self):
        """``fp16 -> fp32`` then a shuffle becomes a shuffle of the fp16 input."""

        def fn(x, y):
            return (x.to(torch.float32) + y) * 2.0

        x = cached_randn(self.SHAPE, dtype=torch.float16)
        y = cached_randn(self.SHAPE, dtype=torch.float32)

        result, before, after = _run_recording(fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME))

        # Before: the staggered fp32 cast output is shuffled (4 B/element).
        # After: the fp16 input is shuffled and the cast itself restores STANDARD.
        assert before == [torch.float32]
        assert after == [torch.float16]
        assert_ea(result, EA.STANDARD)
        assert_val(fn, (x, y), result, rtol=1e-2, atol=1e-2)

    def test_duplicate_shuffles_are_merged(self):
        """Two consumers of one staggered value share a single shuffle."""

        def fn(x, y):
            a = x.to(torch.float16)
            return a + y, a * y

        x = cached_randn(self.SHAPE, dtype=torch.float32)
        y = cached_randn(self.SHAPE, dtype=torch.float16)

        (r1, r2), before, after = _run_recording(
            fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME)
        )

        assert len(before) >= len(after) == 1
        for r, ref in zip((r1, r2), fn(x, y)):
            assert_ea(r, EA.STANDARD)
            torch.testing.assert_close(r.cpu(), ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# insert_ea_shuffle: exactness on device and the geometry of the inserted copy
# ---------------------------------------------------------------------------
def _staggered_add(a, b):
    """``a`` (fp16) upcast to a staggered fp32 meets the STANDARD fp32 ``b``."""
    return a.float() + b


def _staggered_add_inputs(shape):
    return _small_int_fp16(shape, 7), _small_int_fp16(shape, 5, base=3).float()


def _assert_exact_on_device(fn, *cpu_args):
    """Compile ``fn`` on the device and compare with CPU bit for bit."""
    torch.compiler.reset()
    with inductor_config.patch(force_disable_caches=True):
        result = torch.compile(fn)(*(t.to(DEVICE_NAME) for t in cpu_args))
    torch.compiler.reset()
    assert torch.equal(result.cpu(), fn(*cpu_args))


class TestEAShuffleInsert:
    """Mixed staggered/STANDARD pointwise ops compile and match CPU exactly."""

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
        "fp16_to_fp32_plus_fp32": (torch.float16, torch.float32, _staggered_add),
        "fp16_to_fp32_clone_plus_fp32": (
            torch.float16,
            torch.float32,
            lambda a, b: a.float().clone() + b,
        ),
    }
    SHAPES = [(2, 64), (4, 128), (64, 256)]

    @pytest.mark.parametrize("shape", SHAPES, ids=[shapes2key((s,)) for s in SHAPES])
    @pytest.mark.parametrize("program", list(PROGRAMS))
    def test_mixed_ea_pointwise_is_exact(self, program, shape):
        a_dtype, b_dtype, fn = self.PROGRAMS[program]
        a = _small_int_fp16(shape, 7).to(a_dtype)
        b = _small_int_fp16(shape, 5, base=3).to(b_dtype)
        _assert_exact_on_device(fn, a, b)

    def test_shuffle_keeps_input_geometry(self):
        """The copy has the committed input layout, only the arrangement changes."""
        a, b = _staggered_add_inputs((4, 128))
        _, records = _compile_recording(
            _staggered_add, a.to(DEVICE_NAME), b.to(DEVICE_NAME)
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
    def test_wide_rows_with_lx_planning(self, shape):
        _assert_exact_on_device(_staggered_add, *_staggered_add_inputs(shape))

    @pytest.mark.parametrize(
        "shape", WIDE_SHAPES, ids=[shapes2key((s,)) for s in WIDE_SHAPES]
    )
    def test_wide_rows_without_lx_planning(self, shape):
        with spyre_config.patch(lx_planning=False):
            _assert_exact_on_device(_staggered_add, *_staggered_add_inputs(shape))


# ---------------------------------------------------------------------------
# validate_ops is a read-only check
# ---------------------------------------------------------------------------
def _stl(ea, dtype=torch.float16, broadcast=False):
    elems = 64 if dtype == torch.float16 else 32
    return SpyreTensorLayout(
        device_size=[2, elems],
        stride_map=[elems, -1 if broadcast else 1],
        device_dtype=get_device_dtype(dtype),
        element_arrangement=ea,
    )


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


def _fake_graph(ops, buffer_eas):
    buffers = {
        name: SimpleNamespace(layouts=[_stl(ea) for ea in eas])
        for name, eas in buffer_eas.items()
    }
    return SimpleNamespace(operations=list(ops), get_buffer=buffers.__getitem__)


def _validate(ops, buffer_eas):
    graph = _fake_graph(ops, buffer_eas)
    before = list(graph.operations)
    with V.set_graph_handler(graph):
        validate_ops(graph)
    assert graph.operations == before, "validate_ops must not edit the graph"


class TestValidateOps:
    def test_mixed_standard_and_staggered_is_accepted(self):
        """Case 3.3 (one staggered EA with STANDARD operands) is no longer an error."""
        _validate(
            [_fake_op(["a", "b"])],
            {"a": [EA.STANDARD], "b": [EA.DL16_TO_FP32]},
        )

    def test_same_ea_is_accepted(self):
        _validate(
            [_fake_op(["a", "b"])],
            {"a": [EA.FP32_TO_DL16], "b": [EA.FP32_TO_DL16]},
        )

    def test_single_operand_is_skipped(self):
        _validate([_fake_op(["a"])], {"a": [EA.DL16_TO_FP32]})

    def test_non_pointwise_is_skipped(self):
        _validate(
            [_fake_op(["a", "b"], pointwise=False)],
            {"a": [EA.DL16_TO_FP32], "b": [EA.FP32_TO_DL16]},
        )

    @expects_raise
    def test_two_staggered_eas_are_rejected(self):
        with pytest.raises(Unsupported, match="Incompatible ElementArrangement"):
            _validate(
                [_fake_op(["a", "b"])],
                {"a": [EA.DL16_TO_FP32], "b": [EA.FP32_TO_DL16]},
            )

    @expects_raise
    def test_exx2_mixed_with_standard_is_rejected(self):
        with pytest.raises(Unsupported, match="Incompatible ElementArrangement"):
            _validate(
                [_fake_op(["a", "b"])],
                {"a": [EA.EXX2], "b": [EA.STANDARD]},
            )


# ---------------------------------------------------------------------------
# Helpers of insert_ea_shuffle / optimize_shuffling
# ---------------------------------------------------------------------------
class TestNeedsShuffle:
    @pytest.mark.parametrize(
        "in_ea, broadcast, out_ea, expected",
        [
            (EA.DL16_TO_FP32, False, EA.STANDARD, True),
            (EA.FP32_TO_DL16, False, EA.STANDARD, True),
            # A broadcast stick reads one element of the stick: order is moot.
            (EA.DL16_TO_FP32, True, EA.STANDARD, False),
            (EA.STANDARD, False, EA.STANDARD, False),
            # The op itself produces a staggered result: nothing to restore.
            (EA.DL16_TO_FP32, False, EA.DL16_TO_FP32, False),
            (EA.DL16_TO_FP32, False, EA.FP32_TO_DL16, False),
        ],
    )
    def test_needs_shuffle(self, in_ea, broadcast, out_ea, expected):
        layout = SimpleNamespace(device_layout=_stl(in_ea, broadcast=broadcast))
        assert _needs_shuffle(layout, out_ea) is expected


def _reader(**attrs):
    """A plain pointwise ``ComputedBuffer`` reader that can be redirected."""
    op = MagicMock(spec=ComputedBuffer)
    op.layout = MagicMock()
    op.data = MagicMock(spec=Pointwise)
    op.loop_info = None
    op._read_copy_elision_record = None
    op.get_read_names.return_value = ["old"]
    for name, value in attrs.items():
        setattr(op, name, value)
    return op


class TestCanRedirectReaders:
    def test_plain_readers(self):
        assert _can_redirect_readers([_reader(), _reader()], "new")

    def test_no_readers(self):
        assert _can_redirect_readers([], "new")

    @pytest.mark.parametrize(
        "attrs",
        [
            {"layout": MagicMock(spec=MutationLayoutSHOULDREMOVE)},
            {"data": MagicMock(spec=Scatter)},
            {"loop_info": object()},
            {"_read_copy_elision_record": object()},
            {"get_read_names": MagicMock(return_value=["old", "new"])},
        ],
        ids=["mutation", "scatter", "coarse_tiled", "copy_elision", "reads_new"],
    )
    def test_reader_that_cannot_be_redirected(self, attrs):
        assert not _can_redirect_readers([_reader(), _reader(**attrs)], "new")

    def test_not_a_computed_buffer(self):
        assert not _can_redirect_readers([MagicMock()], "new")


# ---------------------------------------------------------------------------
# Shuffle stick levels and coordinate info (codegen, no device needed)
# ---------------------------------------------------------------------------
def _stick_offsets(levels):
    """Sorted element offsets visited by nested ``(cardinality, step)`` levels."""
    offsets = [0]
    for card, step in levels:
        offsets = [o + i * step for o in offsets for i in range(card)]
    return sorted(offsets)


STAGGERED = [EA.DL16_TO_FP32, EA.FP32_TO_DL16]

# Stick-dim sizes (elements, one core) to cover, per elements-per-stick: the
# stick dims of every shape above (aligned, partial, small, review widths).
_STICK_DIM_SIZES = sorted(
    {1, 2, 3, 4, 16, 32, 44, 63, 64, 68, 69, 96, 100, 128, 160, 192}
    | set(REVIEW_WIDTHS)
    | {s[-1] for s in _DL16_TO_FP32_SHAPES + _FP32_TO_DL16_SHAPES if s}
)
# (size, elems per stick) pairs that have staggered levels without a device size:
# an fp32 stick dim needs a whole number of stick pairs.
_LEVEL_CASES = [
    (size, eps)
    for eps in (32, 64)
    for size in _STICK_DIM_SIZES
    if eps == 64 or (-(-size // 32)) % 2 == 0
]


def _expected_levels(size, eps):
    """Reference staggered levels, written out per elements-per-stick."""
    if eps == 64:
        return [(4, 1), (2, 32), (8, 4), (-(-size // 64), 64)]
    sticks = -(-size // 32)
    return [(4, 1), (8, 8), (2, 4), (sticks // 2, 64)]


def _drop_trivial(levels):
    return [lvl for i, lvl in enumerate(levels) if i == 0 or lvl[0] != 1]


class TestShuffleStickLevels:
    @pytest.mark.parametrize("ea", STAGGERED)
    @pytest.mark.parametrize("size", _STICK_DIM_SIZES)
    def test_fp16_levels_permute_the_padded_sticks(self, size, ea):
        """The levels visit every element of the (padded) sticks exactly once."""
        sticks = -(-size // 64)
        levels = _shuffle_stick_levels(size, 64, ea)
        assert levels == _drop_trivial(_expected_levels(size, 64))
        assert levels[:3] == [(4, 1), (2, 32), (8, 4)]
        assert levels[3:] == ([(sticks, 64)] if sticks > 1 else [])
        assert _stick_offsets(levels) == list(range(64 * sticks))

    @pytest.mark.parametrize("ea", STAGGERED)
    @pytest.mark.parametrize(
        "size", [s for s in _STICK_DIM_SIZES if (-(-s // 32)) % 2 == 0]
    )
    def test_fp32_levels_permute_the_padded_stick_pairs(self, size, ea):
        sticks = -(-size // 32)
        pairs = sticks // 2
        levels = _shuffle_stick_levels(size, 32, ea)
        assert levels == _drop_trivial(_expected_levels(size, 32))
        assert levels[:3] == [(4, 1), (8, 8), (2, 4)]
        assert levels[3:] == ([(pairs, 64)] if pairs > 1 else [])
        assert _stick_offsets(levels) == list(range(32 * sticks))

    @pytest.mark.parametrize("size, eps", _LEVEL_CASES)
    def test_both_staggered_eas_share_levels(self, size, eps):
        assert _shuffle_stick_levels(size, eps, EA.DL16_TO_FP32) == (
            _shuffle_stick_levels(size, eps, EA.FP32_TO_DL16)
        )

    @expects_raise
    @pytest.mark.parametrize("ea", STAGGERED)
    @pytest.mark.parametrize(
        "size", [s for s in _STICK_DIM_SIZES if (-(-s // 32)) % 2 == 1]
    )
    def test_fp32_odd_stick_count_without_device_size_raises(self, size, ea):
        with pytest.raises(Unsupported, match="whole FP32 stick pairs"):
            _shuffle_stick_levels(size, 32, ea)

    @pytest.mark.parametrize("ea", STAGGERED)
    @pytest.mark.parametrize(
        "size", [s for s in _STICK_DIM_SIZES if (-(-s // 32)) % 2 == 1]
    )
    def test_fp32_levels_follow_the_device_size(self, size, ea):
        """The device pads the stick dim to whole pairs; the count follows it."""
        padded = 64 * _fp32_stick_pairs(size)
        levels = _shuffle_stick_levels(size, 32, ea, padded)
        assert levels == _shuffle_stick_levels(padded, 32, ea)
        assert _stick_offsets(levels) == list(range(padded))

    @pytest.mark.parametrize("ea", STAGGERED)
    @pytest.mark.parametrize("eps", [32, 64])
    @pytest.mark.parametrize("size", [64, 128, 192, 256, 384])
    def test_device_size_smaller_than_size_is_ignored(self, size, eps, ea):
        assert _shuffle_stick_levels(size, eps, ea, eps) == _shuffle_stick_levels(
            size, eps, ea
        )

    @pytest.mark.parametrize("eps", [32, 64])
    @pytest.mark.parametrize("size", _STICK_DIM_SIZES)
    def test_standard_levels(self, size, eps):
        sticks = -(-size // eps)
        expected = [(eps, 1), (sticks, eps)]
        assert _shuffle_stick_levels(size, eps, EA.STANDARD) == _drop_trivial(expected)

    @expects_raise
    @pytest.mark.parametrize("ea", STAGGERED)
    @pytest.mark.parametrize("eps", [16, 128])
    def test_unsupported_elements_per_stick_raises(self, eps, ea):
        with pytest.raises(Unsupported, match="not defined for"):
            _shuffle_stick_levels(128, eps, ea)


def _shuffle_coord_info_cases():
    """(size, nsplits, eps, ea, device_size) over cores, widths and arrangements."""
    cases = []
    for ea in STAGGERED:
        for eps in (64, 32):
            for nsplits in (1, 2, 4, 32):
                for size in _STICK_DIM_SIZES:
                    # device size over all cores, padded to whole units per core
                    unit = eps * (2 if eps == 32 else 1)
                    per_core = -(-size // unit) * unit
                    cases.append((size, nsplits, eps, ea, per_core * nsplits))
    return cases


_COORD_CASES = _shuffle_coord_info_cases()
_COORD_IDS = [f"{c[3].name}-eps{c[2]}-n{c[1]}-s{c[0]}" for c in _COORD_CASES]


class TestShuffleStickCoordInfo:
    @pytest.mark.parametrize("case", _COORD_CASES, ids=_COORD_IDS)
    def test_structure_over_cores_widths_and_arrangements(self, case):
        size, nsplits, eps, ea, device_size = case
        info = _gen_shuffle_stick_coord_info(
            size, nsplits, eps, "nopad", ea, device_size
        )
        funcs = info["folds"]["dim_prop_func"]
        attrs = info["folds"]["dim_prop_attr"]
        per_core = device_size // nsplits
        levels = _shuffle_stick_levels(size, eps, ea, per_core)

        assert levels == _drop_trivial(_expected_levels(per_core, eps))
        assert info["elemArr"] == len(levels)
        assert info["padding"] == "nopad"
        assert len(funcs) == len(attrs) == 3 + len(levels)
        # the core fold advances by the logical per-core size
        assert funcs[0] == {"Affine": {"alpha_": size, "beta_": 0}}
        assert attrs[0] == {"factor_": nsplits, "label_": "core_fold"}
        assert attrs[1] == {"factor_": 1, "label_": "corelet_fold"}
        assert attrs[2] == {"factor_": 1, "label_": "row_fold"}
        # outermost level first
        assert [a["factor_"] for a in attrs[3:]] == [c for c, _ in reversed(levels)]
        assert [f["Affine"]["alpha_"] for f in funcs[3:]] == [
            s for _, s in reversed(levels)
        ]
        assert [a["label_"] for a in attrs[3:]] == [
            f"elem_arr_{i}" for i in reversed(range(len(levels)))
        ]
        # the levels still cover the whole per-core stick dim once
        assert _stick_offsets(levels) == list(range(per_core))

    @pytest.mark.parametrize("ea", STAGGERED)
    @pytest.mark.parametrize("width", [64, 128, 256])
    def test_fp16_reference_structure(self, width, ea):
        info = _gen_shuffle_stick_coord_info(width, 1, 64, "nopad", ea)
        funcs = info["folds"]["dim_prop_func"]
        attrs = info["folds"]["dim_prop_attr"]
        sticks = width // 64
        outer = [] if sticks == 1 else [sticks]
        assert [a["factor_"] for a in attrs[3:]] == [*outer, 8, 2, 4]
        assert [f["Affine"]["alpha_"] for f in funcs[3:]] == [
            *([64] if outer else []),
            4,
            32,
            1,
        ]

    @pytest.mark.parametrize("ea", STAGGERED)
    @pytest.mark.parametrize("width", [64, 128, 256])
    def test_fp32_reference_structure(self, width, ea):
        info = _gen_shuffle_stick_coord_info(width, 1, 32, "nopad", ea)
        funcs = info["folds"]["dim_prop_func"]
        attrs = info["folds"]["dim_prop_attr"]
        pairs = width // 64
        outer = [] if pairs == 1 else [pairs]
        assert [a["factor_"] for a in attrs[3:]] == [*outer, 2, 8, 4]
        assert [f["Affine"]["alpha_"] for f in funcs[3:]] == [
            *([64] if outer else []),
            4,
            8,
            1,
        ]

    @pytest.mark.parametrize("ea", STAGGERED)
    @pytest.mark.parametrize("size", [32, 69, 96, 160, 192, 320, 448, 576, 704])
    def test_odd_logical_size_padded_to_whole_pairs_on_the_device(self, size, ea):
        """An odd fp32 stick count sits in whole pairs on the device."""
        device_size = 64 * _fp32_stick_pairs(size)
        info = _gen_shuffle_stick_coord_info(
            size, 1, 32, "nopad", ea, device_size=device_size
        )
        ref = _gen_shuffle_stick_coord_info(device_size, 1, 32, "nopad", ea)
        funcs, ref_funcs = info["folds"]["dim_prop_func"], ref["folds"]["dim_prop_func"]
        assert info["elemArr"] == ref["elemArr"]
        assert info["folds"]["dim_prop_attr"] == ref["folds"]["dim_prop_attr"]
        assert funcs[1:] == ref_funcs[1:]
        # the core fold still advances by the logical per-core size
        assert funcs[0] == {"Affine": {"alpha_": size, "beta_": 0}}

    @pytest.mark.parametrize("case", _COORD_CASES, ids=_COORD_IDS)
    def test_gen_coord_info_value_routes_the_stick_dim_of_a_shuffle(self, case):
        size, nsplits, eps, ea, device_size = case
        got = gen_coord_info_value(
            size=size,
            nsplits=nsplits,
            elems_per_stick=eps,
            is_stick_dim=True,
            element_arrangement=ea,
            is_ea_shuffle=True,
            stick_device_size=device_size,
        )
        assert got == _gen_shuffle_stick_coord_info(
            size, nsplits, eps, "nopad", ea, device_size
        )

    @pytest.mark.parametrize("ea", STAGGERED)
    @pytest.mark.parametrize("eps", [32, 64])
    @pytest.mark.parametrize("nsplits", [1, 4, 32])
    def test_gen_coord_info_value_leaves_other_dims_of_a_shuffle(
        self, nsplits, eps, ea
    ):
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


# Made with Bob
