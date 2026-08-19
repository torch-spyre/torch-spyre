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

"""Regression tests: in-place ops (``mul_``/``add_``/``sub_``/``div_``/
``relu_``/...) must be backed by a Spyre device kernel that mutates their own
storage and leaves unrelated live buffers alone.

Background. Registered eager ops are standalone-compiled. A compiled in-place
kernel bakes its write-destination address at trace time and reuses it across
calls, so if such a kernel is generated for an in-place op it can write to a
stale address, overwriting an unrelated buffer that happens to be live there --
the shape of the ``google/gemma-3-1b-it`` compile corruption, where an in-place
``scores *= scale`` inside attention overwrote the outer graph's live residual.

``torch-spyre`` therefore registers in-place ops as functional-compute +
``self.copy_`` back (a runtime-addressed device copy), matching the existing
``normal_``/``uniform_``/``zero_`` kernels. Without that registration these ops
have no Spyre kernel and fall to a generic ``CompositeExplicitAutograd`` /
decomposition path that is not address-safe under ``torch.compile``.

The registrations are derived from ``eager.COMPILED_OPS`` (the functional ops
that get a compiled kernel), so the two lists cannot drift; the parametrization
of :func:`test_inplace_op_has_spyre_kernel` is derived the same way,
:func:`test_relu_inplace_is_not_a_compiled_kernel` catches an in-place op being
put back on the compiled path, and
:func:`test_pow_inplace_is_not_paired_by_name` pins the one pair the derivation
must *reject*.

The deterministic guard here is :func:`test_inplace_op_has_spyre_kernel`: it
fails if the registration is dropped. Because that guard is parametrized over a
derived list -- and pytest reports an empty parameter set as a silent skip --
:func:`test_derivation_is_not_degenerate` guards the guard, failing if the
derivation stops finding the overloads it is meant to cover. The remaining tests assert the behaviour
that registration must uphold (correct in-place values with ``self`` identity
preserved, a co-live buffer left intact, and PyTorch's in-place dtype
contract). The full end-to-end model repro lives in vllm-spyre
``tests/inductor/test_gemma_residual_clobber.py``.
"""

import pytest
import torch

import torch_spyre  # noqa: F401


DEVICE = "spyre"
DTYPE = torch.float16
ATOL = 0.05

aten = torch.ops.aten

# The compile-time failure the backend raises for an unsupported rounding mode.
# Narrower than ``Exception`` so the rounding-mode xfails cannot mask an
# unrelated break; see :func:`test_div_rounding_mode`.
_DIV_MODE_ERROR = torch._inductor.exc.InductorError

# (in-place packet, overload, scalar operand) for each registered arithmetic op.
INPLACE_SCALAR = [
    (aten.mul_, "Scalar", 3.0),
    (aten.add_, "Scalar", 2.0),
    (aten.sub_, "Scalar", 1.5),
    (aten.div_, "Scalar", 4.0),
]
INPLACE_TENSOR = [
    (aten.mul_, "Tensor"),
    (aten.add_, "Tensor"),
    (aten.sub_, "Tensor"),
    (aten.div_, "Tensor"),
]


def _apply(op_name, a, b, **kwargs):
    """Run in-place ``op_name`` (e.g. ``mul_``) of ``a`` by ``b``; return ``a``."""
    getattr(a, op_name)(b, **kwargs)
    return a


# In-place overloads whose coverage is load-bearing: the gemma root-cause op,
# the ``relu_`` gap, the ``div_`` rounding-mode overloads, and the four
# arithmetic packets. ``test_derivation_is_not_degenerate`` pins these by name
# so a derivation that silently stops finding them fails loudly.
ANCHOR_OPS = [
    "aten::add_.Scalar",
    "aten::add_.Tensor",
    "aten::div_.Scalar",
    "aten::div_.Scalar_mode",
    "aten::div_.Tensor",
    "aten::div_.Tensor_mode",
    "aten::mul_.Scalar",
    "aten::mul_.Tensor",
    "aten::relu_",
    "aten::sub_.Scalar",
    "aten::sub_.Tensor",
]

# The derivation finds 37 overloads today. A floor (rather than an equality)
# lets ops be added or legitimately dropped without churn, while still catching
# a broad collapse that leaves only the anchors standing.
MIN_DERIVED_OPS = 30


def _spyre_kernel_ops():
    """Every in-place overload that ``register_inplace_kernels`` should cover.

    Derived from ``eager.COMPILED_OPS`` the same way the registration is, so a
    newly added compiled op is checked automatically instead of relying on this
    test's hand-written lists (which is how ``relu_`` and ``div_``'s ``_mode``
    overloads went unchecked).

    Deliberately re-derived here rather than read back from whatever
    ``register_inplace_kernels`` registered: consuming the registration's own
    output would make the guard vacuous, since deleting the registration -- the
    regression this module exists to catch -- would empty the list instead of
    failing it. See :func:`test_derivation_is_not_degenerate`, which keeps that
    independence from turning an empty derivation into a silent pass.
    """
    from torch_spyre.ops import eager

    compiled = {op.name() for op in eager._compile_kernel_overloads(eager.COMPILED_OPS)}
    ops = []
    for name in compiled:
        base = name.partition(".")[0].split("::")[1]
        inplace_packet = getattr(aten, base + "_", None)
        if inplace_packet is None:
            continue
        for overload in inplace_packet.overloads():
            try:
                inplace_op = getattr(inplace_packet, overload)
            except RuntimeError:
                continue
            sibling = eager._functional_sibling(inplace_op)
            if sibling is None or sibling.name() not in compiled:
                continue
            ops.append(inplace_op)
    return sorted(set(ops), key=lambda o: o.name())


def test_derivation_is_not_degenerate():
    """``_spyre_kernel_ops`` must keep finding the overloads it is meant to.

    Without this, the module's main guard can pass while checking nothing.
    :func:`test_inplace_op_has_spyre_kernel` is parametrized over a *derived*
    list, and pytest reports an empty parameter set as a **skip with exit code
    0** -- so a derivation that breaks (``_functional_sibling`` starting to
    return ``None``, ``COMPILED_OPS`` being renamed, the overload filter
    tightening) would collect zero cases and the suite would go green while
    silently testing nothing. Verified: stubbing ``_functional_sibling`` to
    return ``None`` takes the derivation from 37 overloads to 0.

    This test is *not* parametrized, so it always runs and always asserts.
    """
    ops = _spyre_kernel_ops()
    names = {op.name() for op in ops}

    missing = [name for name in ANCHOR_OPS if name not in names]
    assert not missing, (
        f"derivation no longer finds {missing}; _spyre_kernel_ops is broken or "
        "these ops lost their compiled functional sibling -- the parametrized "
        "guard below would silently collect nothing"
    )
    assert len(ops) >= MIN_DERIVED_OPS, (
        f"derivation collapsed to {len(ops)} overloads (expected at least "
        f"{MIN_DERIVED_OPS}); the parametrized guard below is no longer "
        "covering the registration"
    )


@pytest.mark.parametrize("op", _spyre_kernel_ops(), ids=lambda o: o.name())
def test_inplace_op_has_spyre_kernel(op):
    """Each derived in-place overload must have a Spyre (PrivateUse1) kernel.

    This is the deterministic regression signal: dropping
    ``register_inplace_kernels`` leaves these ops with no device kernel (they
    fall to a decomposition that generates the address-unsafe compiled in-place
    kernel), and this assertion fails.
    """
    assert torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), "PrivateUse1"), (
        f"{op.name()} has no Spyre kernel; in-place registration was dropped"
    )


def test_pow_inplace_is_not_paired_by_name():
    """``pow_`` must NOT be registered: the pairing would be unsafe.

    ``pow_.Scalar(Tensor self, Scalar exponent)`` and
    ``pow.Scalar(Scalar self, Tensor exponent)`` share an overload name with
    *swapped* operands, so pairing on the name alone would build a kernel that
    computes ``other ** self``. ``_functional_sibling``'s signature check
    rejects it. (Matching ``pow_.Scalar`` to ``pow.Tensor_Scalar`` by signature
    would be operand-correct, but the device's functional ``pow`` is itself
    wrong today -- ``torch.pow(2.0, 3.0)`` returns 16.0, and
    ``pow.Tensor_Tensor`` raises "unimplemented operation pow" -- so the
    conservative name requirement stays.)
    """
    from torch_spyre.ops import eager

    for op in (aten.pow_.Scalar, aten.pow_.Tensor):
        assert eager._functional_sibling(op) is None, (
            f"{op.name()} was paired with a functional op; its same-named "
            "overload has swapped operands, so the pairing is unsafe"
        )
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(
            op.name(), "PrivateUse1"
        ), f"{op.name()} must not get a Spyre kernel while functional pow is wrong"


def test_relu_inplace_is_not_a_compiled_kernel():
    """``relu_`` must go through the ``copy_`` path, not a compiled in-place one.

    ``aten.relu_`` used to sit in the compiled-op list, i.e. it got exactly the
    address-baking in-place kernel this module exists to prevent. Guard against
    it (or any other in-place op) being added back there.
    """
    from torch_spyre.ops import eager

    inplace = [op for op in eager.COMPILED_OPS if str(op).endswith("_")]
    assert not inplace, (
        f"in-place ops must not be in COMPILED_OPS (found {inplace}); a compiled "
        "in-place kernel bakes its write address and can overwrite a live buffer"
    )


@pytest.mark.parametrize(
    "packet,overload,scalar", INPLACE_SCALAR, ids=lambda v: getattr(v, "__name__", v)
)
class TestInPlaceScalar:
    def test_values_and_identity(self, packet, overload, scalar):
        """The op mutates ``self`` in place with the correct values."""
        op = packet.__name__
        torch.manual_seed(0)
        base = torch.randn(8, 16, dtype=DTYPE)
        dev = base.to(DEVICE)
        returned = _apply(op, dev, scalar)

        ref = _apply(op, base.clone(), scalar)
        assert returned.data_ptr() == dev.data_ptr()  # true in-place, not a copy
        torch.testing.assert_close(
            dev.to("cpu").float(), ref.float(), atol=ATOL, rtol=0
        )

    def test_leaves_second_live_tensor_intact(self, packet, overload, scalar):
        """An in-place op on one live device tensor must not corrupt another.

        ``keep`` is allocated first and read back AFTER the in-place op on the
        separately-allocated ``work``; a baked/stale write address shows up as
        ``keep`` changing.
        """
        op = packet.__name__
        torch.manual_seed(1)
        keep_cpu = torch.randn(64, 64, dtype=DTYPE)
        work_cpu = torch.randn(64, 64, dtype=DTYPE)
        keep = keep_cpu.to(DEVICE)
        work = work_cpu.to(DEVICE)

        _apply(op, work, scalar)

        torch.testing.assert_close(
            keep.to("cpu").float(), keep_cpu.float(), atol=ATOL, rtol=0
        )


@pytest.mark.parametrize(
    "packet", [p for p, _ in INPLACE_TENSOR], ids=lambda p: p.__name__
)
def test_inplace_tensor_operand(packet):
    """Tensor (not scalar) operand overload of each in-place op."""
    op = packet.__name__
    torch.manual_seed(2)
    base = torch.randn(8, 16, dtype=DTYPE)
    other = torch.randn(8, 16, dtype=DTYPE).abs() + 1.0  # nonzero for div_
    dev = base.to(DEVICE)
    _apply(op, dev, other.to(DEVICE))

    ref = _apply(op, base.clone(), other)
    torch.testing.assert_close(dev.to("cpu").float(), ref.float(), atol=ATOL, rtol=0)


@pytest.mark.parametrize(
    "rounding_mode",
    [
        None,
        # 'floor'/'trunc' fail in the *functional* div kernel, independently of
        # any in-place wrapper: torch.div(x, 3.0, rounding_mode='floor') raises
        # InductorError("No FX node for buf1") for a scalar operand and
        # "Cannot resolve target for 'div_rn'/'truediv'" for a tensor one. The
        # in-place kernel inherits that gap; it does not introduce it.
        #
        # ``raises`` is the specific compile failure, not bare ``Exception``:
        # these xfails must not double as a blanket licence for this path to
        # break some other way. A wrong *value* or a failure in the derivation
        # or ``copy_`` wrapper raises something else and still fails the suite,
        # while ``strict=True`` catches the day the kernel starts working.
        pytest.param(
            "floor", marks=pytest.mark.xfail(raises=_DIV_MODE_ERROR, strict=True)
        ),
        pytest.param(
            "trunc", marks=pytest.mark.xfail(raises=_DIV_MODE_ERROR, strict=True)
        ),
    ],
    ids=lambda m: str(m),
)
@pytest.mark.parametrize(
    "scalar_operand", [True, False], ids=["Scalar_mode", "Tensor_mode"]
)
def test_div_rounding_mode(rounding_mode, scalar_operand):
    """``div_.Tensor_mode``/``Scalar_mode`` get kernels, so they need values checked.

    These take a kwarg-only ``rounding_mode``; the kernel forwards ``**kwargs``
    to the functional op, and nothing else in this module exercises that path.
    The ``rounding_mode=None`` case is the one the backend supports today; the
    rounding modes are strict-xfail on the specific compile error so this starts
    passing (and tells us) once the functional kernel gains support, while any
    *other* failure on this path still fails the suite.
    """
    op = "div_"
    torch.manual_seed(4)
    base = torch.randn(8, 16, dtype=DTYPE) * 8.0
    other = 3.0 if scalar_operand else torch.randn(8, 16, dtype=DTYPE).abs() + 1.0
    dev_other = other if scalar_operand else other.to(DEVICE)

    dev = base.to(DEVICE)
    _apply(op, dev, dev_other, rounding_mode=rounding_mode)

    ref = _apply(op, base.clone(), other, rounding_mode=rounding_mode)
    torch.testing.assert_close(dev.to("cpu").float(), ref.float(), atol=ATOL, rtol=0)


def test_div_rounding_mode_matches_functional_op():
    """The in-place kernel must agree with the functional op it wraps.

    Whatever ``torch.div(..., rounding_mode=...)`` does on device, ``div_`` must
    do the same -- that equivalence is the in-place kernel's actual contract,
    and unlike the CPU comparison above it holds regardless of backend gaps.
    """
    torch.manual_seed(4)
    base = torch.randn(8, 16, dtype=DTYPE) * 8.0

    functional = torch.div(base.to(DEVICE), 3.0, rounding_mode=None)
    dev = base.to(DEVICE)
    dev.div_(3.0, rounding_mode=None)
    torch.testing.assert_close(
        dev.to("cpu").float(), functional.to("cpu").float(), atol=0, rtol=0
    )


def test_relu_inplace_values_and_identity():
    """``relu_`` must mutate ``self`` in place with correct values.

    ``relu_`` was previously registered as a compiled in-place kernel; it now
    goes through the same functional + ``copy_`` path as the arithmetic ops.
    """
    torch.manual_seed(5)
    base = torch.randn(8, 16, dtype=DTYPE)
    dev = base.to(DEVICE)
    returned = dev.relu_()

    assert returned.data_ptr() == dev.data_ptr()
    torch.testing.assert_close(
        dev.to("cpu").float(), base.clone().relu_().float(), atol=ATOL, rtol=0
    )


@pytest.mark.parametrize(
    "method",
    [
        "abs_",
        "exp_",
        "neg_",
        "tanh_",
        "sigmoid_",
        "reciprocal_",
        "sqrt_",
        "rsqrt_",
        "log_",
        "bitwise_not_",
        "logical_not_",
    ],
)
def test_newly_covered_unary_inplace(method):
    """Unary in-place ops newly routed through ``copy_`` must match the
    functional op they wrap.

    Widening the registration from 4 arithmetic ops to everything derived from
    ``COMPILED_OPS`` brings these along; compare against the *device* functional
    op so the check tests the wrapper, not the backend's numerics.
    """
    torch.manual_seed(6)
    if method == "bitwise_not_":
        base = torch.randint(-8, 8, (8, 16), dtype=torch.int32)
    elif method == "logical_not_":
        base = torch.randint(0, 2, (8, 16)).to(DTYPE)
    elif method in ("sqrt_", "rsqrt_", "log_", "reciprocal_"):
        base = torch.rand(8, 16, dtype=DTYPE) + 0.5  # positive domain
    else:
        base = torch.randn(8, 16, dtype=DTYPE)

    functional = getattr(torch, method[:-1])(base.to(DEVICE)).to("cpu")

    dev = base.to(DEVICE)
    returned = getattr(dev, method)()
    assert returned.data_ptr() == dev.data_ptr()  # true in-place, not a copy
    torch.testing.assert_close(
        dev.to("cpu").float(), functional.float(), atol=0, rtol=0
    )


def test_inplace_rejects_uncastable_result_dtype():
    """An in-place op whose promoted result cannot cast back to ``self`` raises.

    PyTorch's in-place contract requires ``can_cast(result_type, self.dtype)``;
    eager CPU/CUDA raise here. The ``copy_``-back kernel would otherwise
    silently downcast (int32 ``self`` truncating a float32 result), so the
    kernel checks explicitly. Guards that check against being dropped.
    """
    dev = torch.ones(8, 16, dtype=torch.int32).to(DEVICE)
    with pytest.raises(RuntimeError, match="can't be cast to the desired"):
        dev.mul_(2.5)


def test_inplace_inside_compiled_graph():
    """An in-place op mutating a fresh buffer inside a compiled graph must not
    corrupt a co-live buffer that is re-read afterwards.

    This mirrors the gemma-3-1b structure: an opaque region performs an
    in-place ``scores *= scale`` while the outer graph's residual is live, and
    the residual is re-read after the region. Runs eager vs
    ``torch.compile`` and requires bit-for-bit agreement.
    """

    def block(resid, x):
        scores = x * 2.0  # fresh buffer, offset 0 -- like an attention matmul out
        scores *= 0.5  # the in-place op under test
        return resid + scores.sum(dim=-1, keepdim=True)

    torch.manual_seed(3)
    resid_cpu = torch.randn(64, 64, dtype=DTYPE)
    x_cpu = torch.randn(64, 64, dtype=DTYPE)

    eager = block(resid_cpu.to(DEVICE), x_cpu.to(DEVICE)).to("cpu").float()
    compiled = torch.compile(block, backend="inductor", fullgraph=True, dynamic=False)
    got = compiled(resid_cpu.to(DEVICE), x_cpu.to(DEVICE)).to("cpu").float()

    torch.testing.assert_close(got, eager, atol=ATOL, rtol=0)
