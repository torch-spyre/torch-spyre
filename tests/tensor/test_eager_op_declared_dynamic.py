# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""Eager ops on a tensor declared with ``.to("spyre", dynamic=...)`` (#6030).

``.to(dynamic=)`` calls ``mark_dynamic`` on the tensor, and ``dynamic=False``
does not override a mark. An eager op runs through a ``dynamic=False`` compiled
frame (``compile_once`` in ``ops/eager.py`` and ``_OPWrapper`` in
``_inductor/decompositions.py``), which specialises on the size, so a marked
tensor used to make it raise ``ConstraintViolationError``, or reach the Spyre
lowering as a symbolic dim. Both sites now hand the kernel an unmarked
``detach()`` of every marked tensor. A user ``torch.compile`` over the same
tensor is untouched, and the last class checks that it is still strict.
"""

import torch
import torch.nn.functional as F
from torch.fx.experimental.symbolic_shapes import ConstraintViolationError
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_spyre  # noqa: F401

DEVICE = torch.device("spyre")
DECLARATION = {0: {"min": 8, "max": 128, "granularity": 8}}
COLS = 64
SIZES = [128, 64, 32, 128]  # the last repeats a size: a cache hit, not a new trace
MARKS = ("_dynamo_dynamic_indices", "_dynamo_dynamic_range")


def _declared(rows):
    cpu = torch.rand(rows, COLS, dtype=torch.float16)
    return cpu, cpu.to(DEVICE, dynamic=DECLARATION)


def _is_marked(t):
    return any(getattr(t, name, None) for name in MARKS)


class TestCloneDeclaredDynamic(TestCase):
    def test_clone_matches_cpu(self) -> None:
        for rows in SIZES:
            with self.subTest(rows=rows):
                cpu, x = _declared(rows)
                torch.testing.assert_close(x.clone().cpu(), cpu)

    def test_clone_of_resized_tensor_matches_itself(self) -> None:
        # resize_ within the declared range keeps the same tensor object (and
        # marks) across sizes. Its contents are not defined by resize_, so
        # compare the clone with the tensor itself, read back on the host.
        _, x = _declared(128)
        for rows in SIZES:
            with self.subTest(rows=rows):
                x.resize_(rows, COLS)
                torch.testing.assert_close(x.clone().cpu(), x.cpu())

    def test_copy_into_static_destination(self) -> None:
        for rows in SIZES:
            with self.subTest(rows=rows):
                cpu, x = _declared(rows)
                dst = torch.zeros(rows, COLS, dtype=torch.float16).to(DEVICE)
                dst.copy_(x)
                torch.testing.assert_close(dst.cpu(), cpu)

    def test_copy_into_declared_destination(self) -> None:
        # The other direction, as reported on a [256, 32] int32 table: the
        # marked tensor is dst, not src. (min=8 is the reservation floor.)
        declaration = {0: {"min": 8, "max": 256, "granularity": 8}}
        for rows in [256, 128, 8]:
            with self.subTest(rows=rows):
                host = torch.randint(0, 1000, (rows, 32), dtype=torch.int32)
                dst = torch.zeros(rows, 32, dtype=torch.int32).to(
                    DEVICE, dynamic=declaration
                )
                self.assertIs(dst.copy_(host.to(DEVICE)), dst)
                self.assertIn(0, dst._dynamo_dynamic_indices)
                torch.testing.assert_close(dst.cpu(), host)

    def test_marks_survive_the_eager_op(self) -> None:
        # The strict promise is for the caller's own torch.compile code, so the
        # op must neither drop the marks from x nor leak them onto the result.
        _, x = _declared(128)
        y = x.clone()
        self.assertIn(0, x._dynamo_dynamic_indices)
        self.assertFalse(_is_marked(y))


class TestOtherEagerOpsOnDeclaredTensor(TestCase):
    """Same cause, other entry points. Before the fix a plain, in-place or out=
    ``add`` failed with ``InductorError: Spyre backend does not support: add ...
    iteration space has size s33``, ``slice_scatter`` with a declared ``src``
    with ``ConstraintViolationError``, and ops that go through ``_OPWrapper``
    (a Spyre decomposition, here ``gelu``) with the same ``InductorError``.
    """

    def test_out_of_place_op_matches_cpu(self) -> None:
        for rows in SIZES:
            with self.subTest(rows=rows):
                cpu, x = _declared(rows)
                torch.testing.assert_close(
                    (x + 1).cpu(), cpu + 1, rtol=1e-3, atol=1e-3
                )
                self.assertIn(0, x._dynamo_dynamic_indices)

    def test_inplace_op_returns_the_callers_tensor(self) -> None:
        for rows in SIZES:
            with self.subTest(rows=rows):
                cpu, x = _declared(rows)
                self.assertIs(x.add_(1), x)
                self.assertIn(0, x._dynamo_dynamic_indices)
                torch.testing.assert_close(x.cpu(), cpu + 1, rtol=1e-3, atol=1e-3)

    def test_out_variant_returns_the_callers_tensor(self) -> None:
        for rows in SIZES:
            with self.subTest(rows=rows):
                a_cpu, a = _declared(rows)
                b_cpu, b = _declared(rows)
                _, out = _declared(rows)
                self.assertIs(torch.add(a, b, out=out), out)
                self.assertIn(0, out._dynamo_dynamic_indices)
                torch.testing.assert_close(
                    out.cpu(), a_cpu + b_cpu, rtol=1e-3, atol=1e-3
                )

    def test_slice_scatter_with_declared_src(self) -> None:
        for rows in SIZES:
            with self.subTest(rows=rows):
                cpu, x = _declared(rows)
                src_cpu, src = _declared(rows)
                result = x.slice_scatter(src, 0, 0, x.shape[0])
                torch.testing.assert_close(
                    result.cpu(), cpu.slice_scatter(src_cpu, 0, 0, rows)
                )

    def test_decomposed_op_matches_cpu(self) -> None:
        # gelu has a Spyre decomposition, so eager gelu goes through _OPWrapper
        # and not through compile_once.
        for rows in SIZES:
            with self.subTest(rows=rows):
                cpu, x = _declared(rows)
                torch.testing.assert_close(
                    F.gelu(x).cpu(), F.gelu(cpu), rtol=1e-2, atol=1e-2
                )
                self.assertIn(0, x._dynamo_dynamic_indices)


class TestUserCompileStaysStrict(TestCase):
    """The marks are stripped only for the internal eager kernels. A user ``torch.compile``
    that specialises a declared dim must keep raising, so an unintended
    specialisation stays loud instead of silently recompiling per size. This
    holds with or without the fix; it guards against widening the fix to the
    declaration itself (e.g. ``maybe_mark_dynamic``).
    """

    def test_specialising_body_still_raises(self) -> None:
        torch._dynamo.reset()
        _, x = _declared(128)

        def specialises(t):
            return t + 1 if t.shape[0] == 128 else t

        compiled = torch.compile(specialises, backend="eager")
        with self.assertRaisesRegex(ConstraintViolationError, "specialized"):
            compiled(x)


if __name__ == "__main__":
    run_tests()

