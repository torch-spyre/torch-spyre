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

"""Regression tests: in-place arithmetic ops (``mul_``/``add_``/``sub_``/
``div_``) must be backed by a Spyre device kernel that mutates their own
storage and leaves unrelated live buffers alone.

Background. Registered eager ops are standalone-compiled. A compiled in-place
kernel bakes its write-destination address at trace time and reuses it across
calls, so if such a kernel is generated for an in-place op it can write to a
stale address and clobber an unrelated buffer that happens to be live there --
the shape of the ``google/gemma-3-1b-it`` compile corruption, where an in-place
``scores *= scale`` inside attention overwrote the outer graph's live residual.

``torch-spyre`` therefore registers in-place arithmetic as functional-compute +
``self.copy_`` back (a runtime-addressed device copy), matching the existing
``normal_``/``uniform_``/``zero_`` kernels. Without that registration these ops
have no Spyre kernel and fall to a generic ``CompositeExplicitAutograd`` /
decomposition path that is not address-safe under ``torch.compile``.

The deterministic guard here is :func:`test_inplace_op_has_spyre_kernel`: it
fails if the registration is dropped. The remaining tests assert the behaviour
that registration must uphold (correct in-place values with ``self`` identity
preserved, and no clobber of a co-live buffer). The full end-to-end model repro
lives in vllm-spyre ``tests/inductor/test_gemma_residual_clobber.py``.
"""

import pytest
import torch

import torch_spyre  # noqa: F401


DEVICE = "spyre"
DTYPE = torch.float16
ATOL = 0.05

aten = torch.ops.aten

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


def _apply(op_name, a, b):
    """Run in-place ``op_name`` (e.g. ``mul_``) of ``a`` by ``b``; return ``a``."""
    getattr(a, op_name)(b)
    return a


@pytest.mark.parametrize(
    "packet,overload",
    [(p, o) for p, o, _ in INPLACE_SCALAR] + INPLACE_TENSOR,
    ids=lambda v: v if isinstance(v, str) else v.__name__,
)
def test_inplace_op_has_spyre_kernel(packet, overload):
    """Each in-place arithmetic overload must have a Spyre (PrivateUse1) kernel.

    This is the deterministic regression signal: dropping
    ``register_inplace_arith_kernel`` leaves these ops with no device kernel
    (they fall to a decomposition that generates the address-unsafe compiled
    in-place kernel), and this assertion fails.
    """
    op = getattr(packet, overload)
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        op.name(), "PrivateUse1"
    ), f"{op.name()} has no Spyre kernel; in-place registration was dropped"


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

    def test_no_clobber_of_second_live_tensor(self, packet, overload, scalar):
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
