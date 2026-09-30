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

import math

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


# Shapes used by both bidirectional roundtrip tests.
# [4, 128]: stick-aligned (128 = 2 fp16 sticks of 64).
# [4, 96] and [5, 4, 96]: sub-stick (96 = 1.5 fp16 sticks; the last is partial).
# [2, 3, 5] and [2, 3, 1]: extents inside the first half-stick, where the live
# elements fit one FP32 stick but the stagger still reaches the pair, so the
# widening conversion's capacity comes from insert_staggered_ea_padding rather
# than from the extent itself.  [2, 3, 1] also exercises the sentinel stick dim.
_ROUNDTRIP_SHAPES = [
    pytest.param((4, 128), id="aligned_4x128"),
    pytest.param((4, 96), id="substick_4x96"),
    pytest.param((5, 4, 96), id="substick_5x4x96"),
    pytest.param((2, 3, 5), id="substick_2x3x5"),
    pytest.param((2, 3, 1), id="substick_2x3x1"),
]


@pytest.mark.parametrize("device", ["spyre"])
@pytest.mark.parametrize("shape", _ROUNDTRIP_SHAPES)
@pytest.mark.parametrize(
    "fp16",
    DtypeOpTable.fp16_types(),
    ids=lambda dt: str(dt).replace("torch.", ""),
)
def test_bidirectional_roundtrip_fp16_start(device, shape, fp16):
    """Test FP16→FP32→FP16 roundtrip for stick-aligned and sub-stick shapes."""
    torch._dynamo.reset()

    @torch.compile
    def fn(x):
        # FP16(STANDARD) → FP32(DL16_TO_FP32) → FP16(STANDARD)
        x_fp32 = x.to(torch.float32)
        # Without an op in between, inductor folds the pair into an FP16 copy.
        return (x_fp32 * 2.0).to(dtype=fp16)

    x = torch.randn(shape, device=device, dtype=fp16)
    result = fn(x)

    # Verify final EA is STANDARD
    assert_ea(result, ElementArrangement.STANDARD)

    # Verify correctness
    assert_val(fn, x, result)

    print("✓ FP16→FP32→FP16 roundtrip works")


@pytest.mark.parametrize("device", ["spyre"])
@pytest.mark.parametrize("shape", _ROUNDTRIP_SHAPES)
@pytest.mark.parametrize(
    "fp16",
    DtypeOpTable.fp16_types(),
    ids=lambda dt: str(dt).replace("torch.", ""),
)
def test_bidirectional_roundtrip_fp32_start(device, shape, fp16):
    """Test FP32→FP16→FP32 roundtrip for stick-aligned and sub-stick shapes."""
    torch._dynamo.reset()

    @torch.compile
    def fn(x):
        # FP32(STANDARD) → FP16(FP32_TO_DL16) → FP32(STANDARD)
        x_fp16 = x.to(dtype=fp16)
        return x_fp16.to(torch.float32)

    x = torch.randn(shape, device=device, dtype=torch.float32)
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


def _upcast_consumed(x):
    return x.to(torch.float32) + 0.0


def _downcast(x):
    return x.to(torch.float16)


@pytest.mark.parametrize(
    "fn, dtype",
    [
        pytest.param(_upcast_consumed, torch.float16, id="fp16_to_fp32"),
        pytest.param(_downcast, torch.float32, id="fp32_to_fp16"),
    ],
)
def test_conversion_on_size1_stick_dim_stays_in_its_buffer(fn, dtype):
    """A conversion of an (n, 1) tensor accesses its FP32 stick pair once.

    Each row's element sits in a stick of its own, and the FP32 side is padded
    to a stick pair on a dim of its own. Access beyond the buffer is invisible in
    the values, as only the first FP32 stick holds an element, but faults the
    device once it leaves mapped memory; 704 rows, with the pair as the last
    buffer of the HBM pool, are enough for that.
    """
    torch._dynamo.reset()
    x = (torch.randint(-64, 64, (704, 1)) / 8).to(dtype)
    result = torch.compile(fn)(x.to(DEVICE_NAME))
    torch.testing.assert_close(result.cpu(), fn(x), rtol=0, atol=0)


# An op consuming an upcast FP32 value before the downcast back, on stick lengths
# that end inside a stick. The upcast value is staggered: each FP16 stick spans a
# pair of FP32 sticks. (68,) leaves the stick dim unsplit; in (232,), (1000,),
# (4, 100) and (4, 104) the stick dim rounded up to FP32 sticks is a whole number
# of pairs; in the rest, work division splits the stick dim where the round-up
# ends in the middle of a pair, so it has to split by whole pairs.
_PARTIAL_STICK_UPCAST_SHAPES = [
    (68,),
    (232,),
    (1000,),
    (4, 100),
    (4, 104),
    (196,),
    (4100,),
    (4, 68),
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


# A staggered FP32 value spans two sticks per FP16 stick, so an extent that is
# not a whole FP16 stick leaves the second one partly live.  Cover the sub-stick
# extents (below one FP16 stick) and the half-stick multiples, where the wide
# side needs an odd number of FP32 sticks rounded up to the pair.
_CONSUMED_WIDE_EXTENTS = [1, 5, 31, 32, 33, 63, 96]


@pytest.mark.parametrize("device", ["spyre"])
@pytest.mark.parametrize("extent", _CONSUMED_WIDE_EXTENTS, ids=lambda n: f"n{n}")
@pytest.mark.parametrize(
    "fp16",
    DtypeOpTable.fp16_types(),
    ids=lambda dt: str(dt).replace("torch.", ""),
)
def test_consumed_wide_value_keeps_both_staggered_sticks(device, extent, fp16):
    """A consumer between two conversions covers both sticks of a staggered pair.

    Unlike the roundtrips above, the wide value is read by an op rather than
    converted straight back, so the pair of FP32 sticks holding one FP16 stick
    has to survive into that op's own iteration.  An op reaching only the first
    stick drops half the elements, interleaved through the extent rather than
    left in a tail, which a bare roundtrip cannot show: with nothing consuming
    the wide value the pair of casts folds to an identity and no conversion
    reaches the device at all.
    """
    torch._dynamo.reset()

    def fn(x):
        return (x.to(torch.float32) * 2.0).to(dtype=fp16)

    x = torch.randn(2, 3, extent, device=device, dtype=fp16)
    result = torch.compile(fn)(x)

    assert_ea(result, ElementArrangement.STANDARD)
    assert_val(fn, x, result)


@pytest.mark.parametrize("rows", [4, 8], ids=lambda n: f"rows{n}")
def test_widened_slice_cast_to_int32_gathers_rows(rows):
    """A one-stick slice of a widened value, cast to INT32, indexes a gather.

    This is how MoE expert routing builds its gather indices: each expert id is
    broadcast across an FP16 stick, widened, and sliced to one FP32 stick.  The
    INT32 cast reads the staggered FP32 value but writes an unstaggered output,
    one stick per row, which it must fill without reaching into the next row.
    """
    torch._dynamo.reset()

    def fn(widened, table):
        index = widened.to(torch.float32)[..., :32].to(torch.int32)[..., 0]
        return table[index]

    ids = (torch.arange(rows) * 7 + 3) % 64
    widened = ids.to(torch.float16)[None, :, None].expand(1, rows, 64).contiguous()
    table = (torch.arange(64 * 128) % 257).reshape(64, 128).to(torch.float16)
    result = torch.compile(fn, dynamic=False)(
        widened.to(DEVICE_NAME), table.to(DEVICE_NAME)
    )

    torch.testing.assert_close(result.cpu(), table[ids][None], rtol=0, atol=0)


# Extents a single FP16 stick holds but a single FP32 stick does not, bracketed by
# the ones on either side that fit (32) or fill (64) the FP16 stick.
_NARROWED_EXTENTS = [5, 32, 33, 48, 63, 64, 96]


def _narrow_then_widen_eager(x, fp16):
    return x.to(dtype=fp16).to(torch.float32)


def _narrow_consume_widen(x, fp16):
    return (x.to(dtype=fp16) * 2.0).to(torch.float32)


@pytest.mark.parametrize("device", ["spyre"])
@pytest.mark.parametrize("shape_prefix", [(4,), (2, 3)], ids=["4xn", "2x3xn"])
@pytest.mark.parametrize("extent", _NARROWED_EXTENTS, ids=lambda n: f"n{n}")
@pytest.mark.parametrize(
    "fn, mode",
    [
        pytest.param(_narrow_then_widen_eager, "eager", id="eager"),
        pytest.param(_narrow_consume_widen, "compile", id="compile_consumed"),
    ],
)
@pytest.mark.parametrize(
    "fp16",
    DtypeOpTable.fp16_types(),
    ids=lambda dt: str(dt).replace("torch.", ""),
)
def test_narrowed_value_widens_into_every_fp32_stick(
    device, shape_prefix, extent, fn, mode, fp16
):
    """Widening a narrowed value returns the elements past the first FP32 stick.

    An FP16 stick holding more than 32 elements widens into two FP32 sticks, both
    holding host elements, so the STANDARD result has to map the second one to
    the host rather than leave it as unaddressed capacity.  Eager casts run as
    separate conversions; the compiled case consumes the narrowed value before
    widening it.  The values are integers exact in every format involved,
    so a dropped element shows as a mismatch instead of hiding in the tolerance.
    """
    torch._dynamo.reset()
    shape = (*shape_prefix, extent)
    host = (torch.arange(math.prod(shape)) % 257).reshape(shape).to(torch.float32)

    result = _run(fn, host.to(device), fp16, mode=mode)

    assert_ea(result, ElementArrangement.STANDARD)
    torch.testing.assert_close(result.cpu(), fn(host, fp16), rtol=0, atol=0)


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


def test_fp16_to_fp32_on_heads_view_input_reads_as_a_slice():
    """An upcast of a heads view of fused QKV lays out every head.

    Eager decode passes q as ``qkv[:, :256].view(1, 2, 128)``, which keeps qkv's
    device layout: one device dim holds all 12 fp16 sticks, at ``2*head +
    floor(elem/64)``. The view reads as a slice, so the output gets its own
    heads dim rather than a rescale of qkv's.
    """

    def fn(q):
        x32 = q.float()
        # The multiply keeps inductor from folding the round trip into an FP16 copy.
        return (x32 * 3.0).half(), x32

    qkv = torch.randn(1, 768, dtype=torch.float16)
    expected, _ = fn(qkv[:, :256].view(1, 2, 128))
    result, upcast = torch.compile(fn)(qkv.to(DEVICE_NAME)[:, :256].view(1, 2, 128))

    torch.testing.assert_close(result.cpu(), expected, atol=0.005, rtol=0.005)
    assert_ea(upcast, ElementArrangement.DL16_TO_FP32)
    assert list(get_spyre_tensor_layout(upcast).device_size) == [1, 4, 2, 32]


@pytest.mark.parametrize("tokens", [16, 40])
def test_eager_upcast_of_a_stick_with_no_dim_counting_its_sticks(tokens):
    """An eager upcast of a hidden state with its stick on the token dim compiles.

    A TP=2 decode hands RMSNorm a ``[tokens, 4096]`` value laid out as
    ``[4096, 64]``/``[1, 4096]``: the tokens on the stick, and no device dim to
    count its sticks, so the upcast builds its layout as for a slice. 40 tokens
    take two fp32 sticks. The host transfer rejects that stride map, so the layout
    is allocated on the device directly and only the layout is checked.
    """
    from torch_spyre._C import (
        SpyreTensorLayout,
        get_device_dtype,
        spyre_empty_with_layout,
    )

    layout = SpyreTensorLayout(
        [4096, 64],
        [1, 4096],
        get_device_dtype(torch.float16),
        ElementArrangement.STANDARD,
    )
    hidden = spyre_empty_with_layout(
        (tokens, 4096),
        (4096, 1),
        torch.float16,
        layout,
        device=torch.device(DEVICE_NAME),
    )

    upcast = hidden.to(torch.float32)

    assert_ea(upcast, ElementArrangement.DL16_TO_FP32)
    assert get_spyre_tensor_layout(upcast).stride_map[-1] == upcast.stride(0)


@pytest.mark.parametrize("start", [0, 1024], ids=lambda n: f"start{n}")
def test_fp16_to_fp32_on_transposed_slice_keeps_input_stick(start):
    """A sliced staggered upcast puts its stick where the input's stick lands.

    Transposing the column slice moves the input's stick dim to output dim 0,
    which is not the output's last dim.
    """
    torch._dynamo.reset()
    tokens, width = 8, 256

    def fn(x):
        x32 = x[:, start : start + width].t().float()
        return (x32 * x32).sum(0).to(x.dtype), x32

    x = torch.randn(tokens, 1536, dtype=torch.float16)
    expected, _ = fn(x)
    result, upcast = torch.compile(fn, dynamic=False)(x.to(DEVICE_NAME))

    torch.testing.assert_close(result.cpu(), expected, rtol=0.01, atol=0.01)
    assert_ea(upcast, ElementArrangement.DL16_TO_FP32)
    assert get_spyre_tensor_layout(upcast).stride_map[-1] == upcast.stride(0)


def _downcast_stick_slice(start, stop):
    def fn(a, b):
        return (b.float() * 2)[:, 0, start:stop].half() + a

    return fn


@pytest.mark.parametrize(
    "start, extent", [(0, 64), (64, 128)], ids=["first_pair", "second_pair"]
)
def test_downcast_of_one_stick_slice_reads_the_stick_pair(start, extent):
    """A downcast of a one-FP32-stick slice reads both sticks of its pair.

    The staggered value already holds whole pairs, so no gap dim is added: the
    pair lives on the num-sticks dim.  The slice's host range fits one FP32
    stick, but the downcast iterates a whole FP16 stick, spread over the pair.
    The size-2 outer dim selected by ``[:, 0]`` is a host dim, not the pair.
    """
    torch._dynamo.reset()
    fn = _downcast_stick_slice(start, start + 32)
    a = torch.randn(4, 32, dtype=torch.float16)
    b = torch.randn(4, 2, extent, dtype=torch.float16)
    result = torch.compile(fn, dynamic=False)(a.to(DEVICE_NAME), b.to(DEVICE_NAME))

    torch.testing.assert_close(result.cpu(), fn(a, b), rtol=0.005, atol=0.005)


def test_downcast_of_slice_starting_inside_a_stick_pair_is_unsupported():
    """A slice starting at the pair's second FP32 stick fails loudly."""
    torch._dynamo.reset()
    fn = _downcast_stick_slice(32, 64)
    a = torch.randn(4, 32, dtype=torch.float16)
    b = torch.randn(4, 2, 64, dtype=torch.float16)
    with pytest.raises(Exception, match="starting at stick 1, inside a stick pair"):
        torch.compile(fn, dynamic=False)(a.to(DEVICE_NAME), b.to(DEVICE_NAME))


def test_downcast_of_view_stepping_one_stick_is_unsupported():
    """A downcast whose outer host dim steps one FP32 stick fails loudly.

    Each FP16 output stick takes one row of that dim, while a stick pair spans
    two rows.
    """
    torch._dynamo.reset()

    def fn(a, b):
        return (b.float() * 2).view(4, 4, 32)[:, 0:2, :].half() + a

    a = torch.randn(4, 2, 32, dtype=torch.float16)
    b = torch.randn(4, 128, dtype=torch.float16)
    with pytest.raises(Exception, match="starting at a varying stick"):
        torch.compile(fn, dynamic=False)(a.to(DEVICE_NAME), b.to(DEVICE_NAME))


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


# Made with Bob
