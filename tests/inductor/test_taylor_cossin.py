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

"""Tests for the Spyre cos / sin decompositions.

``aten.cos`` and ``aten.sin`` used to be CPU-offloaded fallback ops. They are
now decomposed on device by ``spyre_cos`` / ``spyre_sin`` in
``torch_spyre/_inductor/decompositions.py``, built entirely from arithmetic
primitives Spyre lowers natively (``floor``, ``mul``, ``add``, ``sub``).
This file tests those two functions -- directly on CPU for the numerics, and
through ``torch.cos`` / ``torch.sin`` under ``torch.compile`` for the
registration and on-device path.

All model YAML files (tests/resource/models/*.yaml, non-_spyre variants)
show fp32 as the input dtype at the cos/sin call site:

  Model                          shape (prefill)    shape (decode)
  granite-4.1-8b                 [1, 29, 128]       [1, 1, 128]
  granite-3.3-8b-instruct        [1, 41, 128]       —
  Meta-Llama-3.1-8B-Instruct     [1, 12, 128]       [1, 1, 128]
  Qwen2.5-7B-Instruct            [1, 39, 128]       —
  Ministral-3-14B-Instruct-2512  [1, 14, 128]       —
  Mistral-Small-3.2-24B (text)   [1, 855, 128]      [1, 1, 128]
  Mistral-Small-3.2-24B (vision) [1064, 64]         —
  gpt-oss-20b (non-contiguous)   [1, 11, 32]        —
  gemma-4-26B-A4B-it             [1, 34, 256]       [1, 1, 256]

Algorithm under test
--------------------
Step 1 — Cody-Waite two-term range reduction (ops: floor, mul, add, sub):

    PI is split into two parts so that PI_HI + PI_LO == π exactly in fp64:
        PI_HI = 3.1415927410125732   (nearest fp32 to π)
        PI_LO = -8.742278012618954e-8 (fp64 residual: π - PI_HI)

    k    = floor(x / π + 0.5)              # round-to-nearest via floor
    x_r  = (x − k × PI_HI) − k × PI_LO    # two-step subtraction, |x_r| ≤ π/2
    sign = 1 − 2 × (k − 2 × floor(k / 2)) # (−1)^k

    torch.round is NOT used — it is not implemented in the Spyre codegen.
    floor(x + 0.5) is the standard round-to-nearest equivalent using only floor.

    Using a single-term π (PI_fp32 = 3.1415927) accumulates error at
    ~8.7e-8 per unit of k; at k=318 (x≈1000) the residual x_r is off by
    ~2.8e-5, dominating the polynomial error. The two-term split reduces
    the x_r error to machine-noise levels (< 2e-12 for |x| ≤ 128000).

Step 2 — degree-9 Horner polynomial on |x_r| ≤ π/2:

    cos(x_r) ≈ 1 + x²(−1/2 + x²(1/24 + x²(−1/720 + x²/40320)))
    sin(x_r) ≈ x_r(1 + x²(−1/6 + x²(1/120 + x²(−1/5040 + x²/362880))))
    cos(x)   = sign × cos(x_r)
    sin(x)   = sign × sin(x_r)

    The polynomial itself has a ~2.5e-5 rounding floor in fp32 for cos
    (from partial cancellation of large terms near |x_r| = π/2).  This is
    irreducible without switching to fp64 evaluation.

Accuracy summary (fp32 in, measured against an fp64 reference).  cos and sin
match on RoPE-realistic inputs but not on a dense linspace sweep, where cos is
the limiting op by ~7x -- the polynomial's cancellation floor near |x_r| = pi/2
applies to cos at every range, while sin only reaches it once the range
reduction itself starts contributing:

    range                          cos        sin
    |x| <= pi                   2.5e-5     3.7e-6
    |x| <= 100                  2.8e-5     3.7e-6
    |x| <= 1000                 5.5e-5     3.0e-5
    RoPE-realistic              5.3e-5     5.3e-5   (head_dim 128/256, seq 1064)

    safe tolerance:              1e-4

fp16 / bf16 in: the decomposition upcasts to fp32 and casts back.  On device,
measured against cos/sin of the input as the device holds it, this lands at
4.9e-4 (fp16) and 2.4e-3 (bf16) at every range from pi to 1000 -- in both cases
the output cast, not the polynomial, is the limit (fp16 ULP at 1.0 is ~9.8e-4,
bf16 ~3.9e-3).  Evaluating the range reduction at the input width instead gives
0.75 absolute error for fp16 at |x| ~ 1000; a reduction-only fp32 window is
honored but weaker (2.4e-3), so the whole body is upcast.

Comparing a *device* fp16 result against a CPU reference needs care: Spyre's
fp16 is a 1-6-9 format (9 mantissa bits, not IEEE's 10), so H2D re-rounds an
IEEE fp16 input by up to 1 ULP, shifting cos/sin by up to 0.5 at |x| ~ 1000
regardless of implementation -- a CPU fallback measures the same 0.495.  Device
tests below therefore reference the round-tripped input.  bf16 and fp32 H2D are
bit-exact and need no such care.

Integral / bool in: promoted to a float result on device, no CPU fallback.  The
result dtype is asked of aten rather than reimplemented -- ``elementwise_dtypes``
with ``INT_TO_FLOAT``, the promotion kind ``torch._refs.cos`` is built with -- so
int32, int64 and bool all yield fp32 (the default dtype), matching ``torch.cos``
exactly, measured at 3.2e-6 on CPU and 3.2e-6 on device.  cos/sin therefore need
no dtype rule of their own and no ``fallback_ops`` entry.

The fp32 *compute* width is a separate matter and is spelled with an explicit
``.to``, not taken from that same call: ``elementwise_dtypes`` derives the
computation dtype from ``_computation_dtype_map``, and
``torch_spyre._inductor.patches.spyre_data_types`` empties that map for the whole
Inductor compile so upstream refs do not silently widen fp16 on a device whose
native dtype is fp16.  Under a Spyre compile it reports ``compute=float16`` where
an eager call reports ``compute=float32``, so decorating the decompositions with
``elementwise_type_promotion_wrapper`` looks idiomatic but widens nothing on the
path that matters: measured 0.7501 max error at fp16, identical to no upcast at
all, versus 5.0e-4 for the explicit cast.  ``test_low_precision_upcasts`` and
``test_low_precision_compiles_and_accurate`` are what catch that regression.

KNOWN LIMITATION for fp16 / bf16 -- issue #2818, pre-existing and not specific to
cos/sin: the closing fp32 -> fp16/bf16 cast is wrong on device unless the
innermost dim spans an EVEN number of 32-element fp32 sticks, i.e.
``ceil(size[-1] / 32) % 2 == 0``.  Two fp32 sticks pair into one 64-element fp16
stick and an odd count leaves a dangling half stick.  Measured on (4, N): correct
at N = 48, 64, 112, 128, 192, 240, 256, 320; wrong (over half the elements taking
values absent from the correct result) or a hard "Invalid device sizes and stride
map" at N = 16, 32, 65, 80, 96, 129, 160, 224.  The rule is the pairing, not
"multiple of 64" -- N=48 passes, N=96 does not.  A bare
``(x.float() * 2.0).to(torch.float16)`` fails identically with no cos/sin in the
graph, while pure fp16 pointwise ops are bit-exact, so this is the backend cast
and not the decomposition.  ``test_low_precision_odd_stick_count_xfail`` pins it
so the day #2818 is fixed shows up as an XPASS.  All model head dims (64, 128,
256) are even-stick and unaffected.

Note: fp32 itself can only represent values near 128000 with spacing ~0.008,
which is larger than π; a linspace sweep at |x|_max=128000 is meaningless in
fp32. The tests below are bounded to the largest real RoPE input (seq_len=1064,
inv_freq[0]=1.0 → |x|_max=1063).
"""

import math
import warnings

import pytest
import torch

from torch_spyre._inductor.decompositions import (
    get_spyre_decomp_table,
    spyre_cos,
    spyre_sin,
)
from torch_spyre.ops.fallbacks import FallbackWarning, fallback_ops

# Dtypes the decomposition must not evaluate the range reduction in: it upcasts
# these to fp32 and casts back.  Tolerance is the dtype's own ULP at 1.0, since
# the output cast -- not the polynomial -- is the accuracy limit.
_UPCAST_DTYPES = [
    pytest.param(torch.float16, 1e-3, id="fp16"),
    pytest.param(torch.bfloat16, 8e-3, id="bf16"),
]

# Every integral and boolean dtype ``aten.cos`` / ``aten.sin`` accept.  All of
# them promote to the default floating dtype, so all of them are the
# decomposition's business, whether or not Spyre has a device format for them.
_ATEN_INTEGRAL_DTYPES = [
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]

# ---------------------------------------------------------------------------
# Section 0: registration -- cheap guard, no device needed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", ["cos", "sin"])
def test_registered_as_decomposition_and_not_a_fallback(op_name):
    """cos / sin must reach Inductor as a decomposition, for every dtype.

    ``get_spyre_decomp_table`` drops ``fallback_ops`` *before* applying
    ``spyre_decompositions``, so a stray ``fallback_ops`` entry would not shadow
    the decomposition on the compile path -- but it would still register an eager
    CPU-offload kernel and advertise a CPU path that nothing needs, since the
    decomposition's own promotion lets it serve every real dtype aten accepts.
    (Complex is the one it does not, and needs no fallback either: a complex
    tensor cannot be placed on a Spyre device at all, so no compile sees one --
    ``_taylor_dtypes`` documents this.)  Assert both directions so neither
    drifts back.
    """
    op = getattr(torch.ops.aten, op_name).default
    assert op in get_spyre_decomp_table(), (
        f"{op} is missing from the Spyre decomposition table; RoPE would be back "
        f"on a CPU round-trip"
    )
    assert op not in fallback_ops, (
        f"{op} is in fallback_ops; the decomposition promotes integral input "
        f"itself, so a CPU fallback is dead weight"
    )


@pytest.mark.parametrize("dtype", _ATEN_INTEGRAL_DTYPES)
@pytest.mark.parametrize("op_name", ["cos", "sin"])
def test_promotes_integral_and_bool_to_float(op_name, dtype):
    """Integral / bool input promotes to a float result, exactly as aten does.

    The dtype contract is aten's own: the decompositions take their result dtype
    from ``elementwise_dtypes(..., INT_TO_FLOAT)``, the promotion kind
    ``torch._refs.cos`` is built with.  Assert against ``torch.cos`` /
    ``torch.sin`` rather than a hardcoded fp32 so this keeps holding if the
    default dtype changes.

    Parametrized over *every* integral and boolean dtype ``aten.cos`` accepts,
    not just the ones Spyre has a device format for, because these bodies are
    plain torch functions: the promotion has to be right before the question of
    what the device can hold even arises.

    Without the promotion the result is cast back to the integral input dtype and
    truncates to 0 / +-1 -- measured max error 0.58 int32, 0.99 int64, 0.46 bool.
    """
    decomp, ref = {"cos": (spyre_cos, torch.cos), "sin": (spyre_sin, torch.sin)}[
        op_name
    ]
    x = (torch.arange(64) % 2 if dtype is torch.bool else torch.arange(64) % 7).to(
        dtype
    )
    out = decomp(x)
    expected = ref(x)
    assert out.dtype == expected.dtype, (
        f"{dtype} {op_name}: got {out.dtype}, aten gives {expected.dtype}"
    )
    max_err = (out.double() - expected.double()).abs().max().item()
    assert max_err < 1e-4, f"{dtype} {op_name} error {max_err:.3e} >= 1e-4"


# ---------------------------------------------------------------------------
# Section 1: CPU-only numerical accuracy of the decomposition bodies
# ---------------------------------------------------------------------------


class TestTaylorAccuracyCpu:
    """Validate decomposition accuracy on CPU across a broad input range.

    ``spyre_cos`` / ``spyre_sin`` are plain functions over torch ops, so they
    evaluate on CPU tensors unchanged. These tests need no Spyre device.

    The meaningful upper bound for fp32 tests is |x|_max ~ 1063 (the actual
    maximum RoPE embedding value for seq_len=1064, inv_freq[0]=1.0).  Values
    larger than ~10000 cannot be meaningfully tested in fp32 because the fp32
    representable spacing (0.008 at 128000) exceeds π, making the ground-truth
    cos/sin input itself ill-defined.
    """

    # The ranges are nested, so the widest sweep subsumes the narrower ones at
    # this linspace density.  |x|<=pi is kept as well because it is the only
    # one that isolates the polynomial from the range reduction (k == 0 for
    # most of it), which is where the cos error floor near |x_r| = pi/2 lives.
    @pytest.mark.parametrize("x_max", [math.pi, 1000.0])
    @pytest.mark.parametrize("op_name", ["cos", "sin"])
    def test_fp32_accuracy(self, op_name, x_max):
        """Max fp32 error < 1e-4 across the swept range."""
        decomp, ref = {"cos": (spyre_cos, torch.cos), "sin": (spyre_sin, torch.sin)}[
            op_name
        ]
        x = torch.linspace(-x_max, x_max, steps=10001, dtype=torch.float32)
        # fp64 reference: measures the decomposition, not torch's own fp32 error
        max_err = (decomp(x).double() - ref(x.double())).abs().max().item()
        assert max_err < 1e-4, (
            f"|x|_max={x_max}: max {op_name} error {max_err:.3e} >= 1e-4"
        )

    @pytest.mark.parametrize("dtype, tol", _UPCAST_DTYPES)
    @pytest.mark.parametrize("op_name", ["cos", "sin"])
    def test_low_precision_upcasts(self, op_name, dtype, tol):
        """fp16 / bf16 are evaluated in fp32, then cast back.

        Evaluating the range reduction at the input width instead gives ~3.5e-3
        at |x| <= 10 and ~0.25 at |x| <= 1000 for fp16 -- far outside the
        dtype's own ULP -- so this pins the upcast, not just the output dtype.
        """
        decomp, ref = {"cos": (spyre_cos, torch.cos), "sin": (spyre_sin, torch.sin)}[
            op_name
        ]
        x = torch.linspace(-1000.0, 1000.0, steps=10001, dtype=dtype)
        out = decomp(x)
        assert out.dtype == dtype, f"expected {dtype} out, got {out.dtype}"
        max_err = (out.double() - ref(x.double())).abs().max().item()
        assert max_err < tol, f"{dtype} {op_name} error {max_err:.3e} >= {tol:.0e}"

    @pytest.mark.parametrize("op_name", ["cos", "sin"])
    def test_fp64_is_not_narrowed(self, op_name):
        """fp64 in, fp64 out -- the upcast rule must never *lower* precision.

        fp64 cannot reach a Spyre graph (H2D rejects it: "Spyre backend does not
        support dtype Double"), so this is a CPU-only contract: the bodies are
        plain torch functions and narrowing one to fp32 here would discard input
        precision for nothing.  The error bound is the polynomial's own, which is
        an fp32-era constant and does not improve with a wider input.
        """
        decomp, ref = {"cos": (spyre_cos, torch.cos), "sin": (spyre_sin, torch.sin)}[
            op_name
        ]
        x = torch.linspace(-1000.0, 1000.0, steps=4096, dtype=torch.float64)
        out = decomp(x)
        assert out.dtype == torch.float64, f"fp64 narrowed to {out.dtype}"
        max_err = (out - ref(x)).abs().max().item()
        assert max_err < 1e-4, f"fp64 {op_name} error {max_err:.3e} >= 1e-4"

    def test_boundary_values(self):
        """Values at k*π: cos(k*π) = (−1)^k, sin(k*π) = 0."""
        boundaries = torch.tensor(
            [k * math.pi for k in range(-4, 5)], dtype=torch.float32
        )
        assert (spyre_cos(boundaries) - torch.cos(boundaries)).abs().max() < 1e-4
        assert (spyre_sin(boundaries) - torch.sin(boundaries)).abs().max() < 1e-4

    def test_rope_realistic_inputs(self):
        """Inputs drawn from the actual RoPE frequency formula.

        inv_freq[k] = 1 / (10000 ^ (2k / head_dim)).
        emb = cat([inv_freq @ pos_ids, inv_freq @ pos_ids]).
        Covers head_dim ∈ {64, 128, 256} and seq_len ∈ {1, 29, 128, 855, 1064}.
        Worst case: head_dim=128/256, seq_len=1064 → max error ~ 5.3e-5.
        """
        for head_dim in (64, 128, 256):
            d = head_dim // 2
            inv_freq = 1.0 / (10000.0 ** (torch.arange(d, dtype=torch.float32) / d))
            for seq_len in (1, 29, 128, 855, 1064):
                pos_ids = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
                emb = torch.cat([pos_ids * inv_freq, pos_ids * inv_freq], dim=-1)

                cos_err = (spyre_cos(emb) - torch.cos(emb)).abs().max().item()
                sin_err = (spyre_sin(emb) - torch.sin(emb)).abs().max().item()
                assert cos_err < 1e-4, (
                    f"head_dim={head_dim}, seq_len={seq_len}: cos error {cos_err:.3e}"
                )
                assert sin_err < 1e-4, (
                    f"head_dim={head_dim}, seq_len={seq_len}: sin error {sin_err:.3e}"
                )


# ---------------------------------------------------------------------------
# Section 2: Spyre compilation and correctness, via torch.cos / torch.sin
# ---------------------------------------------------------------------------

# The decomposition is purely pointwise (floor, mul, add, sub), so shape does
# not affect lowering -- one representative shape per rank is enough, matching
# the (256,) / (67, 256) / (67, 71, 256) convention that
# test_inductor_ops.py::test_pointwise_unary_op already uses for cos/sin.
# Sizes here are borrowed from real model call sites so the ranks stay
# realistic, but they are not claimed to be model-level coverage: the inputs
# are torch.randn, not actual RoPE embeddings.  test_rope_end_to_end covers
# the real embedding values, and TestTaylorAccuracyCpu covers the numerics
# across every model head_dim / seq_len combination on CPU.
_SPYRE_SHAPES = [
    pytest.param((256,), id="1d"),
    pytest.param((1064, 64), id="2d_pixtral"),  # largest real |x| (seq_len 1064)
    pytest.param((1, 855, 128), id="3d_prefill"),
    pytest.param((1, 1, 256), id="3d_decode"),  # size-1 dims
]


def _compile_on_spyre(op, x, *, allow_fallback=None):
    """Compile ``op`` for Spyre and return the CPU-side result.

    Any FallbackWarning fails the test: a CPU offload here would mean the
    decomposition did not take effect, which is the whole point of the change.
    ``allow_fallback`` is a message substring for the one warning that is not
    about that -- see ``test_integral_promotes_on_device``, where the *input
    conversion* offloads for reasons that belong to the cast op rather than to
    cos/sin.  It is allowed by name rather than by silencing the category.
    """

    @torch.compile(dynamic=False)
    def fn(t):
        return op(t)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FallbackWarning)
        out = fn(x.to("spyre")).cpu()

    unexpected = [
        str(w.message)
        for w in caught
        if issubclass(w.category, FallbackWarning)
        and not (allow_fallback and allow_fallback in str(w.message))
    ]
    assert not unexpected, f"unexpected CPU fallback(s): {unexpected}"
    return out


@pytest.mark.parametrize("shape", _SPYRE_SHAPES)
def test_cos_compiles_and_accurate(shape):
    """torch.cos compiles on Spyre without a fallback and matches CPU."""
    x = torch.randn(*shape, dtype=torch.float32)
    out = _compile_on_spyre(torch.cos, x)
    torch.testing.assert_close(out, torch.cos(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", _SPYRE_SHAPES)
def test_sin_compiles_and_accurate(shape):
    """torch.sin compiles on Spyre without a fallback and matches CPU."""
    x = torch.randn(*shape, dtype=torch.float32)
    out = _compile_on_spyre(torch.sin, x)
    torch.testing.assert_close(out, torch.sin(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype, tol", _UPCAST_DTYPES)
@pytest.mark.parametrize("op_name", ["cos", "sin"])
def test_low_precision_compiles_and_accurate(op_name, dtype, tol):
    """fp16 / bf16 inputs stay accurate on device via the fp32 upcast.

    Without the upcast the range reduction runs at the input width and the
    result is unusable at large |x| (0.75 absolute error for fp16 at |x| ~ 1000),
    so sweep well past the point where that shows up.

    The reference is cos/sin of the input *as the device holds it*, not of the
    original IEEE tensor.  Spyre's fp16 is a 1-6-9 format (9 mantissa bits, not
    IEEE's 10), so H2D re-rounds an fp16 input by up to 1 ULP, which shifts
    cos/sin by up to 0.5 at |x| ~ 1000 no matter how the op is implemented -- a
    CPU fallback included.  Scoring against the original tensor would charge the
    decomposition for that and mask the thing under test; bf16 is unaffected
    (H2D is bit-exact) and takes the same path here.

    Codegen note, recorded for future reference and not addressed here: this is
    the one path whose intermediates do not all fit in LX.  ``output_code.py``
    under ``TORCH_COMPILE_DEBUG=1`` shows a single 25-OpSpec fused kernel
    (identical for fp16 and bf16) in which ``OpSpec(op='dl16tofp32')`` at op #0
    writes its widened fp32 to an HBM pool buffer rather than to LX -- re-read
    twice, since the widened value has two users -- and op #23 writes back to
    that same buffer for ``OpSpec(op='fp32todl16')`` at op #24 to narrow into the
    output.  So two intermediates round-trip through HBM, both belonging to the
    conversions rather than to the polynomial; the fp32 and integral paths keep
    every intermediate in LX with one HBM write for the result.  Worth revisiting
    only if an fp16 cos/sin call site turns out to be hot -- every supported
    model is fp32 here.
    """
    op = getattr(torch, op_name)
    x = torch.linspace(-1000.0, 1000.0, 4096, dtype=dtype)
    held = x.to("spyre").cpu()  # what the device actually stores
    out = _compile_on_spyre(op, x)
    assert out.dtype == dtype, f"expected {dtype} out, got {out.dtype}"
    ref = op(held.double()).to(dtype)
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


# Integral dtypes that can actually reach the device, out of
# ``_ATEN_INTEGRAL_DTYPES``.  Two are excluded, neither for a cos/sin reason:
#
# * int16 is refused at H2D -- "Unsupported DCI data format conversion: src=9
#   dst=9 (cpu_type=int16, dev_type=int16)" -- so no int16 tensor exists on
#   device for cos/sin to serve.  ``test_promotes_integral_and_bool_to_float``
#   still covers the promotion on the CPU side.
# * uint8 places and reads back exactly, and the compiled result is correct, but
#   the process then dies in senlib at teardown (SIGSEGV, reproduced standalone
#   with an H2D roundtrip and no cos/sin in the graph).  Excluded so it cannot
#   take the suite down; the crash is a backend issue, not this decomposition's.
_DEVICE_INTEGRAL_DTYPES = [
    pytest.param(torch.int32, False, id="int32"),
    pytest.param(torch.int64, True, id="int64"),
    pytest.param(torch.int8, True, id="int8"),
    pytest.param(torch.bool, True, id="bool"),
]


@pytest.mark.parametrize("dtype, cast_offloads", _DEVICE_INTEGRAL_DTYPES)
@pytest.mark.parametrize("op_name", ["cos", "sin"])
def test_integral_promotes_on_device(op_name, dtype, cast_offloads):
    """Integral / bool cos-sin runs on device, promoted, and cos/sin itself
    never offloads.

    This is what removes the need for a ``fallback_ops`` entry: the polynomial
    runs on device for every integral dtype and only the *input cast* differs.
    int32 -> fp32 is element-size preserving, so no stick reordering is emitted
    and the promotion costs nothing in layout terms -- nothing offloads at all.
    int64 / int8 -> fp32 and bool -> fp32 do offload the cast, for reasons that
    differ per dtype and none of which is cos/sin's:

    * int64 has no *distinct* device representation today: int32 is its only
      physical form (H2D downcasts it -- "Backend Spyre does not support int64" in
      ``types_mapping.h`` -- and ``get_device_dtype(torch.int64)`` is
      ``IEEE_INT32`` at 32 elements per stick, indistinguishable from int32).
      That is a deliberate choice rather than a hardware ceiling: the full int64
      range could be carried across the mantissa bits of several fp32 elements,
      but no LLM workload has called for it, so the backend downcasts instead.
      Either way there is no int64 -> fp32 conversion for the backend to support:
      the conversion that would actually execute is int32 -> fp32, already in the
      table as ``int32tofp32``.  The offload happens because the support check is
      keyed on the *logical* torch dtype -- ``is_supported(int64, float32)`` is
      False, so ``convert_element_type`` takes the CPU path before any physical
      format is consulted.  Resolving the source through ``get_device_dtype``
      first, as bool sources already do via ``get_bool_src_operator``, would keep
      it on device.  A needless host round-trip, not a cast the device cannot
      perform.
    * int8 does have a device format of its own (``SENINT8``), and there the gap
      is the plain one: ``DtypeOpTable`` carries no int8 -> fp32 entry at all, so
      the cast has nowhere to go but CPU.  The promoted polynomial still runs on
      device and the result matches aten.
    * bool -> fp32 *is* supported (``is_supported`` returns True, via
      ``dl16tofp32``).  It offloads here because these inputs are *host* bools:
      a DMA-copied bool InputBuffer has a different HBM element ordering, and
      ``dl16tofp32`` reorders sticks, so ``to_dtype`` in
      ``_inductor/lowering.py`` deliberately routes that one case to CPU rather
      than read the buffer back shuffled.  A device-computed bool takes the
      native path, and bool -> fp16 is an IDENTITY byte copy that is safe even
      from host.  So this offload is a correctness guard, not a gap.

    The split is visible in ``output_code.py`` under ``TORCH_COMPILE_DEBUG=1``,
    and it is the *only* difference between these dtypes: the FX graph is
    identical for all of them -- one
    ``prims.convert_element_type(arg0_1, torch.float32)`` node ahead of the
    polynomial nodes -- so the divergence is entirely in how that one node
    lowers.  int32 fuses it into the kernel as ``OpSpec(op='int32tofp32')`` at op
    #0, widening ``IEEE_INT32`` straight into LX at no HBM cost; int64 / int8 /
    host-bool instead emit
    ``buf0 = torch.ops.spyre.to_dtype_cpu.default(arg0_1, torch.float32)`` in
    ``call()`` ahead of the kernel, whose OpSpec list then starts at ``mul``.
    Recorded for future reference rather than as a task: it is a D2H+H2D round
    trip on an unaligned buffer, but the polynomial is unaffected either way --
    every remaining op stays on device with all intermediates in LX and one HBM
    write for the result -- and no LLM workload calls cos/sin on an integral
    tensor today.

    ``cast_offloads`` records which dtypes offload, and the allowance is by
    warning text so a cos/sin offload still fails the test.  Either way it is the
    cast op's business, not this decomposition's: cos/sin of an int64 or bool
    tensor works, and the assertions below check dtype and value against aten.
    For int64 the only real caveat is the range Spyre supports at all, since H2D
    truncates int64 to int32 before any op sees the tensor.

    Shape (4, 17) is deliberately not stick-aligned: unlike the fp16/bf16 path
    (see the module docstring on #2818) the integral promotion has no stick-count
    constraint, because it does not change the element byte size.
    """
    op = getattr(torch, op_name)
    x = (torch.arange(68) % 2 if dtype is torch.bool else torch.arange(68) % 7).to(
        dtype
    )
    x = x.reshape(4, 17)
    out = _compile_on_spyre(
        op, x, allow_fallback="conversion from" if cast_offloads else None
    )
    expected = op(x)
    assert out.dtype == expected.dtype, (
        f"{dtype} {op_name}: got {out.dtype}, aten gives {expected.dtype}"
    )
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype, tol", _UPCAST_DTYPES)
@pytest.mark.parametrize("op_name", ["cos", "sin"])
@pytest.mark.xfail(
    reason="#2818: fp32->fp16/bf16 cast is wrong when ceil(size[-1]/32) is odd",
    strict=False,  # some odd-stick shapes raise, others return wrong values
)
def test_low_precision_odd_stick_count_xfail(op_name, dtype, tol):
    """Pin the #2818 exposure of the fp32 upcast at an odd fp32-stick count.

    (2, 3, 32) spans ceil(32/32) == 1 fp32 stick along the innermost dim, an odd
    count, so the closing fp32 -> fp16/bf16 cast lands in #2818 and over half the
    elements come back with values that are not in the correct result at all.  A
    bare ``(x.float() * 2.0).to(dtype)`` on this shape is wrong by the same amount
    with no cos/sin in the graph, so the defect is the backend cast; this test
    exists to make the fix visible as an XPASS rather than to blame cos/sin.

    Every shape in ``_SPYRE_SHAPES`` has an even fp32-stick count, which is why
    the rest of the suite cannot see this.
    """
    op = getattr(torch, op_name)
    x = torch.linspace(-3.0, 3.0, 192, dtype=dtype).reshape(2, 3, 32)
    held = x.to("spyre").cpu()  # 1-6-9 fp16 round-trip, as elsewhere
    out = _compile_on_spyre(op, x)
    ref = op(held.double()).to(dtype)
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


def test_cos_sin_non_contiguous():
    """Non-contiguous input: gpt-oss-20b actual stride pattern [352, 1, 11].

    The logical shape is [1, 11, 32] with stride[1]=1 and stride[2]=11
    (transposed last two dims relative to row-major).  Pointwise ops
    delegate stride handling to the compiler; this confirms no crash and
    correct output.
    """
    shape = (1, 11, 32)
    storage = torch.randn(352, dtype=torch.float32)  # stride[0]=352
    x = storage.as_strided(shape, stride=(352, 1, 11))

    @torch.compile(dynamic=False)
    def fn(t):
        return torch.cos(t), torch.sin(t)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        cos_out, sin_out = fn(x.to("spyre"))
    torch.testing.assert_close(cos_out.cpu(), torch.cos(x), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sin_out.cpu(), torch.sin(x), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Section 3: End-to-end RoPE kernel test
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard rotate_half used in apply_rotary_pos_emb."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def test_rope_end_to_end():
    """Full RoPE pipeline on Spyre using the cos/sin decompositions.

    Mirrors the HuggingFace Transformers pattern:
        emb       = cat([inv_freq @ pos_ids, inv_freq @ pos_ids])  # fp32
        cos_emb   = cos(emb)                                       # fp32, STANDARD EA
        sin_emb   = sin(emb)
        q_embed   = q * cos_emb + rotate_half(q) * sin_emb
        k_embed   = k * cos_emb + rotate_half(k) * sin_emb

    Uses Llama-3.1-8B prefill shape: emb [1, S=12, D=128], q/k [B=1, H=8, S=12, D=128].
    All tensors kept in fp32 for a clean EA-STANDARD path.  In production models
    the same fp32 cos/sin would be cast to fp16 before the rotary multiply, but
    that cast produces a FP32_TO_DL16 (staggered) EA on Spyre which cannot be
    directly multiplied against STANDARD fp16 q/k unless the staggered operand
    broadcasts on its stick dimension.  What this test pins down is that the
    decomposed ops compile and execute correctly on Spyre inside a real RoPE
    graph; the fp16 integration path in real models is a separate concern (the
    model compiles today because the cast and rotary apply typically fall inside
    a single graph, or cos/sin are pre-computed as STANDARD fp16 via a different
    lowering path).

    EA note: fp32 cos/sin output has STANDARD EA.  All q/k tensors are fp32
    STANDARD.  The cos_emb/sin_emb tensors are unsqueezed from [1, S, D] to
    [1, 1, S, D] so they broadcast over the H dimension in the rotary multiply;
    this is a size-1 broadcast on dim 1 which is EA-compatible.
    """
    B, H, S, D = 1, 8, 12, 128

    @torch.compile(dynamic=False)
    def rope_spyre(q, k, emb):
        """cos/sin + rotary multiply, all in fp32, all STANDARD EA."""
        cos_emb = torch.cos(emb).unsqueeze(1)  # [1, 1, S, D]
        sin_emb = torch.sin(emb).unsqueeze(1)
        return (
            q * cos_emb + _rotate_half(q) * sin_emb,
            k * cos_emb + _rotate_half(k) * sin_emb,
        )

    def rope_ref(q, k, emb):
        cos_emb = torch.cos(emb).unsqueeze(1)
        sin_emb = torch.sin(emb).unsqueeze(1)
        return (
            q * cos_emb + _rotate_half(q) * sin_emb,
            k * cos_emb + _rotate_half(k) * sin_emb,
        )

    torch.manual_seed(0)
    d = D // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(d, dtype=torch.float32) / d))
    pos_ids = torch.arange(S, dtype=torch.float32)
    freqs = torch.outer(pos_ids, inv_freq)  # [S, d]
    emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(0)  # [1, S, D]

    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32)

    q_ref, k_ref = rope_ref(q, k, emb)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        q_out, k_out = rope_spyre(q.to("spyre"), k.to("spyre"), emb.to("spyre"))

    torch.testing.assert_close(q_out.cpu(), q_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(k_out.cpu(), k_ref, atol=1e-4, rtol=1e-4)
