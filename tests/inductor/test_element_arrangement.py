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
from unittest import mock

import pytest
import sympy
import torch
from torch._dynamo.exc import BackendCompilerFailed
from torch._inductor.dependencies import MemoryDep
from torch._inductor.exc import InductorError
from torch._inductor.ir import ComputedBuffer, Pointwise
from torch._inductor.virtualized import V
from torch_spyre._C import (
    ElementArrangement,
    SpyreTensorLayout,
    get_spyre_tensor_layout,
)
from torch_spyre._inductor.constants import DEVICE_NAME
from torch_spyre._inductor.dtype_ops import DtypeOpTable
from torch_spyre._inductor.element_arrangement import validate_ops
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.propagate_hints import spyre_hint
from torch_spyre._inductor.wsr import propagate_named_dims as pnd
from utils_inductor import (
    cached_randn,
    expects_raise,
    shapes2key,
)

EA = ElementArrangement


def _dtype_name(dt):
    return str(dt).split(".")[-1]


def _run(fn, *args, mode="compile", reset_dynamo=False):
    if mode == "compile":
        if reset_dynamo:
            torch._dynamo.reset()
        fn = torch.compile(fn, dynamic=False)
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


def assert_val(fn, x, result, intermediate=None):
    """Compare a device result with the CPU result of ``fn``.

    The tolerance follows the lowest-precision dtype involved. Pass
    ``intermediate`` when ``fn`` rounds through a dtype that is neither the
    input nor the output (for example fp32 -> fp16 -> fp32).
    """
    xs = x if isinstance(x, tuple) else (x,)
    result_cpu = fn(*(t.cpu() for t in xs))
    dtypes = {t.dtype for t in xs}
    dtypes.add(result.dtype)
    if intermediate is not None:
        dtypes.add(intermediate)
    if torch.bfloat16 in dtypes:
        rtol, atol = 1.6e-2, 1.6e-2
    elif torch.float16 in dtypes:
        # For randn-scale values (<5), fp16 half-ulp is <= ~2e-3.
        rtol, atol = 2e-3, 2e-3
    else:
        rtol, atol = 1e-5, 1e-5
    torch.testing.assert_close(result.cpu(), result_cpu, rtol=rtol, atol=atol)


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
    # A size-1 stick dim with several rows also holds one element per stick (#5566).
    (2, 1),
    (3, 1, 1),
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

    # Verify correctness (value is rounded through fp16/bf16 in between)
    assert_val(fn, x, result, intermediate=fp16)

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

    # Verify correctness (value is rounded through fp16/bf16 in between)
    assert_val(fn, x, result, intermediate=fp16)

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
        {(0,): 4096, (1,): tokens},
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


@pytest.mark.parametrize("n", [1, 8, 33, 100])
def test_float_of_mask_from_upcast_keeps_its_arrangement(n):
    """A bool computed from a staggered value is stored in fp32 sticks, so
    `.float()` on it is a same-width copy that keeps DL16_TO_FP32."""

    def fn(x):
        xf = x.float()
        valid = (xf <= 4.0).float()
        return (xf * valid).half()

    x = _small_int_fp16((1, n), 9).reshape(n)
    out = torch.compile(fn)(x.to(DEVICE_NAME))
    assert torch.equal(out.cpu(), fn(x))


@pytest.mark.parametrize("cols", [64, 96, 128])
@pytest.mark.parametrize("row", [0, 1])
def test_downcast_of_one_row_of_an_upcast(row, cols):
    """A row of an upcast value, converted back: the upcast's input also offers a
    layout with its stick on the rows, which cannot address one row and must be
    skipped rather than fail the conversion."""

    def fn(x):
        return x.float()[row].half()

    x = _small_int_fp16((64, cols), 13)
    out = torch.compile(fn, dynamic=False)(x.to(DEVICE_NAME))
    assert torch.equal(out.cpu(), fn(x))


# ---------------------------------------------------------------------------
# D2H EA restore, H2D write back
# ---------------------------------------------------------------------------
def _build_ea_dci_tests():
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
    # pairs once the stick dim is wider than 32 (padded by #5048). Stick dims of
    # at most 32 only get a single fp32 stick; these are rejected on D2H.
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
        # 44 is padded to a stick pair; 16 and 32 are rejected on D2H
        (44,),
        (4, 16),
        (4, 32),
        (4, 44),
        (7, 44),
        (2, 4, 44),
        (2, 3, 4, 44),
    ]

    # fp32 -> fp16 halves the stick count; an odd number of 32-element fp32 sticks
    # (32 wide, 68, 96, 160) is padded to a whole pair by #5048.
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


EA_D2H_TEST_PARAMS = _build_ea_dci_tests()


def _whole_stick_pairs_rejected(expected_ea, shape):
    # fp16 -> fp32 restores whole fp32 stick pairs. A stick dim of at most one fp32
    # stick (32 elements) only gets a single fp32 stick, so half of each pair has
    # nowhere to go. Wider dims are padded to a whole pair by #5048.
    # Note: shape[-1] is safe for 0-dim shapes () because expected_ea is forced
    # to STANDARD in those cases, so the expected_ea == DL16_TO_FP32 check
    # short-circuits and will not reach shape[-1].
    return expected_ea == ElementArrangement.DL16_TO_FP32 and shape[-1] <= 32


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

    y = _convert_one_row_per_core(x, dst_dtype)
    assert_ea(y, expected_ea)

    if _whole_stick_pairs_rejected(expected_ea, shape):
        with pytest.raises(RuntimeError, match="requires whole stick pairs"):
            y.cpu()
        return

    got = y.cpu()
    assert got.dtype == ref.dtype
    assert got.shape == ref.shape
    assert torch.equal(got, ref)


EA_H2D_TEST_PARAMS = EA_D2H_TEST_PARAMS


def _shape_of(param):
    values = param.values if hasattr(param, "values") else param
    return values[3]


EA_H2D_BASIC_TEST_PARAMS = {
    key: params
    for key, params in EA_H2D_TEST_PARAMS.items()
    if _shape_of(params) in _SHAPES
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

    # Create staggered tensor on device
    y = _convert_one_row_per_core(x, dst_dtype)
    assert_ea(y, expected_ea)

    if _whole_stick_pairs_rejected(expected_ea, shape):
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


def test_ea_staggered_host_storage_offset():
    """Host slices with pair-aligned offsets succeed; mid-pair offsets raise."""
    # DL16_TO_FP32: pair is 64 elements (2 FP32 sticks)
    x_dl16 = torch.arange(4 * 128, dtype=torch.float16).reshape(4, 128)
    y_fp32 = _convert_one_row_per_core(x_dl16, torch.float32)
    assert_ea(y_fp32, ElementArrangement.DL16_TO_FP32)

    # Mid-pair offset (32 elements) -> rejected
    host_buf = torch.empty(32 + 4 * 128, dtype=torch.float32)
    mid_pair_slice = host_buf[32:].view(4, 128)
    with pytest.raises(RuntimeError, match="not aligned to a stick pair boundary"):
        mid_pair_slice.copy_(y_fp32)

    # Pair-aligned offset (64 elements) -> succeeds
    host_buf2 = torch.empty(64 + 4 * 128, dtype=torch.float32)
    pair_aligned_slice = host_buf2[64:].view(4, 128)
    pair_aligned_slice.copy_(y_fp32)
    assert torch.equal(pair_aligned_slice, y_fp32.cpu())

    # DL16_TO_FP32 with odd stick-pair width (ea_boundary needed, shape (4, 68)):
    # 68 elements = 3 sticks (padded to 4 sticks / 128 elements in DCI)
    x_odd = torch.arange(4 * 68, dtype=torch.float16).reshape(4, 68)
    y_odd_fp32 = _convert_one_row_per_core(x_odd, torch.float32)
    assert_ea(y_odd_fp32, ElementArrangement.DL16_TO_FP32)

    # Mid-pair offset (32 elements) -> rejected
    host_buf_odd = torch.empty(32 + 4 * 68, dtype=torch.float32)
    mid_pair_slice_odd = host_buf_odd[32:].view(4, 68)
    with pytest.raises(RuntimeError, match="not aligned to a stick pair boundary"):
        mid_pair_slice_odd.copy_(y_odd_fp32)

    # Pair-aligned offset (64 elements) with boundary slice -> succeeds
    host_buf2_odd = torch.empty(64 + 4 * 68, dtype=torch.float32)
    pair_aligned_slice_odd = host_buf2_odd[64:].view(4, 68)
    pair_aligned_slice_odd.copy_(y_odd_fp32)
    assert torch.equal(pair_aligned_slice_odd, y_odd_fp32.cpu())

    # H2D: Pair-aligned offset (64 elements) with boundary expand -> succeeds
    h2d_host_buf = torch.arange(64 + 4 * 68, dtype=torch.float32)
    h2d_host_slice = h2d_host_buf[64:].view(4, 68)
    y_h2d = _convert_one_row_per_core(
        torch.zeros(4, 68, dtype=torch.float16), torch.float32
    )
    assert_ea(y_h2d, ElementArrangement.DL16_TO_FP32)
    y_h2d.copy_(h2d_host_slice)
    assert torch.equal(y_h2d.cpu(), h2d_host_slice)

    # FP32_TO_DL16: stick is 64 elements (1 FP16 stick)
    x_fp32 = torch.arange(4 * 128, dtype=torch.float32).reshape(4, 128)
    y_dl16 = _convert_one_row_per_core(x_fp32, torch.float16)
    assert_ea(y_dl16, ElementArrangement.FP32_TO_DL16)

    # Mid-stick offset (32 elements) -> rejected
    host_buf_fp16 = torch.empty(32 + 4 * 128, dtype=torch.float16)
    mid_stick_slice = host_buf_fp16[32:].view(4, 128)
    with pytest.raises(RuntimeError, match="not aligned to a stick boundary"):
        mid_stick_slice.copy_(y_dl16)

    # Stick-aligned offset (64 elements) -> succeeds
    host_buf2_fp16 = torch.empty(64 + 4 * 128, dtype=torch.float16)
    stick_aligned_slice = host_buf2_fp16[64:].view(4, 128)
    stick_aligned_slice.copy_(y_dl16)
    assert torch.equal(stick_aligned_slice, y_dl16.cpu())


def test_ea_staggered_device_storage_offset():
    """Verify whether device-side slices with pair-aligned offsets restore correctly."""
    # DL16_TO_FP32 D2H: pair-aligned device slice (64 elements / 2 FP32 sticks)
    x_dl16 = torch.arange(4 * 128, dtype=torch.float16).reshape(4, 128)
    y_fp32 = _convert_one_row_per_core(x_dl16, torch.float32)
    assert_ea(y_fp32, ElementArrangement.DL16_TO_FP32)

    dev_slice_fp32 = y_fp32[:, 64:]
    got_fp32 = dev_slice_fp32.cpu()
    expected_fp32 = x_dl16[:, 64:].to(torch.float32)
    assert torch.equal(got_fp32, expected_fp32)

    # FP32_TO_DL16 D2H: stick-aligned device slice (64 elements / 1 FP16 stick)
    x_fp32 = torch.arange(4 * 128, dtype=torch.float32).reshape(4, 128)
    y_dl16 = _convert_one_row_per_core(x_fp32, torch.float16)
    assert_ea(y_dl16, ElementArrangement.FP32_TO_DL16)

    dev_slice_dl16 = y_dl16[:, 64:]
    got_dl16 = dev_slice_dl16.cpu()
    expected_dl16 = x_fp32[:, 64:].to(torch.float16)
    assert torch.equal(got_dl16, expected_dl16)


# ---------------------------------------------------------------------------
# Shuffle op correctness verification to apply or restore element arrangement
# ---------------------------------------------------------------------------
_WHOLE_STICK_PAIRS = "requires whole stick pairs"
_RESCALE_REJECTION = "cannot rescale device layout"
_NOT_DIVISIBLE = "cannot be divided safely"


def _rms_norm_fp32(h):
    """Gemma 4 RMSNorm: normalize in fp32 and cast back to the input dtype."""
    x = h.to(torch.float32)
    return (x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)).to(h.dtype)


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
        (2,),
        (32,),
        (63,),
        (4, 16),
        (4, 32),
        (44,),
        (100,),
        (7, 44),
        (2, 4, 100),
        # odd stick pair padded by PR 5048
        (1, 192),
        (4, 192),
        (4, 320),
        (4, 448),
    ]
    # fp16 -> fp32 (DL16_TO_FP32) also pads an odd number of fp32 sticks to a whole
    # pair. fp32 -> fp16 (FP32_TO_DL16) only accepts an even number of 32-element
    # fp32 sticks (rescale_stl_for_dtype), so these are left out of it.
    _DL16_TO_FP32_SHAPES = _STICK_ALIGNED_SHAPES + [
        (68,),
        (4, 69),
        (4, 96),
        (4, 160),
        (64, 192),
        (64, 448),
        (2, 128256),
        (64, 151936),
    ]
    _FP32_TO_DL16_SHAPES = _STICK_ALIGNED_SHAPES

    # Shapes the compiler rejects today, with a fragment of the error. They are
    # left out of the shuffle cases on the device and checked by
    # test_ea_shuffle_rejected.
    _DL16_TO_FP32_REJECTED = {
        # fewer than a whole pair of 32-element fp32 sticks
        **dict.fromkeys([(2,), (32,), (4, 16), (4, 32)], _WHOLE_STICK_PAIRS),
    }
    _FP32_TO_DL16_REJECTED: dict[tuple[int, ...], str] = {}
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

    # Reduction ops mixing a staggered operand with a STANDARD one.
    MIXED_EA_REDUCTIONS = {
        "sum_dim0": lambda x16, y32: x16.float().sum(dim=0) + y32.sum(dim=0),
        "sum_dim1": lambda x16, y32: x16.float().sum(dim=-1) + y32.sum(dim=-1),
        "mean_dim0": lambda x16, y32: x16.float().mean(dim=0) + y32.mean(dim=0),
        "mean_dim1": lambda x16, y32: x16.float().mean(dim=-1) + y32.mean(dim=-1),
        "sum_mixed_add_dim0": lambda x16, y32: (x16.float() + y32).sum(dim=0),
        "sum_mixed_add_dim1": lambda x16, y32: (x16.float() + y32).sum(dim=-1),
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
        # odd stick pair padded by PR 5048
        (1, 192),
        (4, 192),
        (4, 320),
        (4, 448),
        (64, 448),
        # Gemma 4 hidden state shapes
        (1, 64, 2816),
        (1, 1, 2816),
        (1, 2816),
    ]
    _MIXED_EA_REJECTED = {(2, 32): _WHOLE_STICK_PAIRS}

    # Gemma 4 patterns that mix a staggered operand with a STANDARD one, with the
    # arrangement of the result.
    GEMMA4_MIXED_EA_PROGRAMS = {
        "downcast_add_upcast": (
            lambda x16, y32: y32.to(x16.dtype) + y32,
            EA.STANDARD,
        ),
        # Two chained fp32 RMSNorms with a downcast weight. The last multiply by
        # the downcast weight leaves the result staggered.
        "double_rms_norm": (
            lambda x16, y32: (
                _rms_norm_fp32(_rms_norm_fp32(x16) * y32.to(x16.dtype))
                * y32.to(x16.dtype)
            ),
            EA.FP32_TO_DL16,
        ),
    }

    GEMMA4_MIXED_EA_SHAPES = [
        (1, 64, 2816),
        (1, 1, 2816),
        (1, 2816),
    ]

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

    def _check(
        self,
        fn,
        shape,
        fp16,
        ea=EA.DL16_TO_FP32,
        reset_dynamo=False,
        result_ea=EA.STANDARD,
    ):
        """Run ``fn(x, y)`` compiled on the device; it must equal the CPU result."""
        x, y = self._operands(ea, shape, fp16)
        result = _run(
            fn, x.to(DEVICE_NAME), y.to(DEVICE_NAME), reset_dynamo=reset_dynamo
        )
        assert_ea(result, result_ea)
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
        self._check(self._cast_then(pointwise), (2, 64), fp16, ea, reset_dynamo=True)

    @_FP16
    @pytest.mark.parametrize("shape", _MIXED_EA_SHAPES, ids=lambda s: shapes2key((s,)))
    @pytest.mark.parametrize("program", list(MIXED_EA_PROGRAMS))
    def test_mixed_ea_pointwise(self, request, program, shape, fp16):
        """A pointwise op on a staggered and a STANDARD operand matches the CPU."""
        self._check(self.MIXED_EA_PROGRAMS[program], shape, fp16, reset_dynamo=True)

    @_FP16
    @pytest.mark.parametrize(
        "shape", GEMMA4_MIXED_EA_SHAPES, ids=lambda s: shapes2key((s,))
    )
    @pytest.mark.parametrize("program", list(GEMMA4_MIXED_EA_PROGRAMS))
    def test_mixed_ea_gemma4(self, program, shape, fp16):
        """Gemma 4 RMS norm that has mixed EA cases with size=1 inputs."""
        fn, result_ea = self.GEMMA4_MIXED_EA_PROGRAMS[program]
        self._check(fn, shape, fp16, reset_dynamo=True, result_ea=result_ea)

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
            self._check(
                self.MIXED_EA_PROGRAMS["upcast_plus_std32"],
                shape,
                fp16,
                reset_dynamo=True,
            )

    @_FP16
    @pytest.mark.parametrize(
        "shape", [(2, 64), (4, 128), (64, 256)], ids=lambda s: shapes2key((s,))
    )
    @pytest.mark.parametrize("program", list(MIXED_EA_REDUCTIONS))
    def test_mixed_ea_reduction(self, program, shape, fp16):
        """A reduction op on a staggered and a STANDARD operand matches the CPU."""
        self._check(self.MIXED_EA_REDUCTIONS[program], shape, fp16, reset_dynamo=True)

    @expects_raise
    @_FP16
    @pytest.mark.parametrize(
        "shape, match",
        [
            pytest.param(s, m, id=shapes2key((s,)))
            for s, m in _MIXED_EA_REJECTED.items()
        ],
    )
    def test_mixed_ea_reduction_rejected(self, shape, match, fp16):
        """Staggered fp32 reduction operands rejected by compiler fail cleanly."""
        with pytest.raises((InductorError, BackendCompilerFailed), match=match):
            self._check(
                self.MIXED_EA_REDUCTIONS["sum_mixed_add_dim0"],
                shape,
                fp16,
                reset_dynamo=True,
            )

    @pytest.mark.xfail(
        raises=InductorError,
        strict=True,
        reason="6676: EA shuffle for sliced operands is not yet supported",
        match="no mechanism to resolve stick incompatibility",
    )
    @_FP16
    def test_ea_shuffle_of_sliced_operand(self, fp16):
        """A staggered operand sliced out of a larger buffer and shuffled."""

        def fn(x, y):
            return x[:, 32:96].to(y.dtype) + y[:, 32:96]

        x = torch.randn(4, 128, dtype=torch.float32).to(DEVICE_NAME)
        y = torch.randn(4, 128, dtype=fp16).to(DEVICE_NAME)
        _run(fn, x, y, reset_dynamo=True)


class TestEAValidateOps:
    """Unit tests for the validate_ops pass on Pointwise and Reduction ops."""

    @staticmethod
    def _make_mock_op(op_type, inputs, op_name="add"):
        """Create a mock ComputedBuffer with either Pointwise or Reduction data."""
        mock_data = mock.MagicMock(spec=op_type)
        origin_node = mock.MagicMock()
        origin_node.target.name.return_value = f"aten::{op_name}.Tensor"
        mock_data.origins = {origin_node}
        mock_op = mock.MagicMock(spec=ComputedBuffer)
        mock_op.data = mock_data
        sym = sympy.Symbol("i")
        read_deps = [MemoryDep(name, sym, (sym,), (64,)) for name in inputs]
        mock_op.get_read_writes.return_value = mock.MagicMock(reads=read_deps)
        return mock_op

    @staticmethod
    def _make_mock_graph(operations, buffer_eas):
        mock_graph = mock.MagicMock()
        mock_graph.operations = operations
        buffers = {}
        for name, ea in buffer_eas.items():
            buf = mock.MagicMock()
            layout = mock.MagicMock(spec=SpyreTensorLayout)
            layout.element_arrangement = ea
            buf.layouts = [layout]
            buffers[name] = buf
        mock_graph.get_buffer.side_effect = lambda n: buffers.get(n)
        return mock_graph

    @pytest.mark.parametrize("op_cls", [Pointwise])
    def test_validate_ops_compatible(self, op_cls):
        """All-standard and standard+one-staggered (broadcast) pass validation."""
        op = self._make_mock_op(op_cls, ["buf0", "buf1"])
        graph = self._make_mock_graph(
            [op], {"buf0": EA.STANDARD, "buf1": EA.DL16_TO_FP32}
        )
        with V.set_graph_handler(graph):
            validate_ops(graph)

    @pytest.mark.parametrize("op_cls", [Pointwise])
    def test_validate_ops_incompatible_raises(self, op_cls):
        """Multiple distinct staggered EAs raise Unsupported."""
        op = self._make_mock_op(op_cls, ["buf0", "buf1"])
        graph = self._make_mock_graph(
            [op], {"buf0": EA.DL16_TO_FP32, "buf1": EA.FP32_TO_DL16}
        )
        with V.set_graph_handler(graph):
            with pytest.raises(Unsupported, match="Incompatible ElementArrangement"):
                validate_ops(graph)

    def test_validate_ops_layernorm_exx2_allowed(self):
        """Layernorm ops legitimately mixing EXX2 pass validation."""
        op = self._make_mock_op(Pointwise, ["buf0", "buf1"], op_name="layernormnorm")
        graph = self._make_mock_graph([op], {"buf0": EA.STANDARD, "buf1": EA.EXX2})
        with V.set_graph_handler(graph):
            validate_ops(graph)


def test_ea_h2d_gapped_host_rows():
    """H2D of a host view with gaps between rows into a staggered tensor."""
    # Row stride 192 is not a whole multiple of the width 128 (issue 6326).
    base = (torch.arange(4 * 192) % 251).to(torch.float32).reshape(4, 192)
    src = base[:, :128]
    y = _convert_one_row_per_core(
        torch.zeros(4, 128, dtype=torch.float16), torch.float32
    )
    assert_ea(y, ElementArrangement.DL16_TO_FP32)
    y.copy_(src)
    assert torch.equal(y.cpu(), src)
