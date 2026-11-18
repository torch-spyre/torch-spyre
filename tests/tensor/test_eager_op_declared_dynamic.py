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
from torch_spyre.ops.eager import (
    _DYNAMO_DYNAMIC_MARKS,
    _has_dynamo_marks,
    _static_shape_call,
)

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
                torch.testing.assert_close((x + 1).cpu(), cpu + 1, rtol=1e-3, atol=1e-3)
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

    def test_list_argument_op_matches_cpu(self) -> None:
        # cat takes a list of tensors, so the marked tensors sit inside a list.
        for rows in SIZES:
            with self.subTest(rows=rows):
                a_cpu, a = _declared(rows)
                b_cpu, b = _declared(rows)
                torch.testing.assert_close(
                    torch.cat([a, b]).cpu(), torch.cat([a_cpu, b_cpu])
                )
                self.assertIn(0, a._dynamo_dynamic_indices)

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
    """The marks are stripped only for the internal eager kernels. A user
    ``torch.compile`` that specialises a declared dim must keep raising, so an
    unintended specialisation stays loud instead of silently recompiling per
    size. This holds with or without the fix; it guards against widening the
    fix to the declaration itself (e.g. ``maybe_mark_dynamic``).
    """

    def test_specialising_body_still_raises(self) -> None:
        torch._dynamo.reset()
        _, x = _declared(128)

        def specialises(t):
            return t + 1 if t.shape[0] == 128 else t

        compiled = torch.compile(specialises, backend="eager")
        with self.assertRaisesRegex(ConstraintViolationError, "specialized"):
            compiled(x)


class TestStaticShapeCall(TestCase):
    """The stripping helper on CPU tensors, with a stand-in for the kernel."""

    def test_mark_attributes_are_all_known(self) -> None:
        # Drift guard: the mark names are private torch API. If an upgrade adds
        # or renames one, fail here and not by a silent return of #6030.
        decorators = torch._dynamo.decorators
        # Set next to the marks (torch >= 2.14) as an optional size hint for a
        # marked dim. It does not make a dim symbolic, so it is no mark.
        companions = {"_dynamo_hint_overrides"}
        markers = {
            "mark_dynamic": lambda t: torch._dynamo.mark_dynamic(t, 0, min=2, max=8),
            "maybe_mark_dynamic": lambda t: torch._dynamo.maybe_mark_dynamic(t, 0),
            "mark_unbacked": lambda t: decorators.mark_unbacked(t, 0),
            "mark_unbacked(strict)": lambda t: decorators.mark_unbacked(
                t, 0, strict=True
            ),
        }
        for name, mark in markers.items():
            with self.subTest(marker=name):
                t = torch.zeros(4, 3)
                mark(t)
                added = {n for n in vars(t) if n.startswith("_dynamo")}
                self.assertTrue(added)
                self.assertLessEqual(added - companions, set(_DYNAMO_DYNAMIC_MARKS))
                self.assertTrue(_has_dynamo_marks(t))
                self.assertFalse(_has_dynamo_marks(t.detach()))

    def test_list_arguments_and_results(self) -> None:
        # Tensors inside a list argument are found, and the caller's own tensor
        # comes back from inside a list result.
        marked, plain = torch.zeros(4, 3), torch.zeros(4, 3)
        torch._dynamo.mark_dynamic(marked, 0)

        def kernel(tensors, other):
            self.assertFalse(any(_has_dynamo_marks(t) for t in (*tensors, other)))
            tensors[0].add_(1)
            return [tensors[0], other, (tensors[1],)]

        result = _static_shape_call(kernel)([marked, plain], other=marked)
        self.assertIs(result[0], marked)
        self.assertIs(result[1], marked)
        self.assertIs(result[2][0], plain)
        self.assertEqual(marked.sum().item(), 12)  # the write reached the caller
        self.assertTrue(_has_dynamo_marks(marked))

        # Nothing marked: the kernel gets the caller's own objects back untouched.
        seen = []
        _static_shape_call(lambda *a: seen.extend(a))(plain, [plain])
        self.assertIs(seen[0], plain)
        self.assertIs(seen[1][0], plain)

    def test_metadata_change_on_the_twin_raises(self) -> None:
        # A resize on the twin would not reach the caller's tensor.
        marked = torch.zeros(4, 3)
        torch._dynamo.mark_dynamic(marked, 0)
        with self.assertRaisesRegex(RuntimeError, "changed the size"):
            _static_shape_call(lambda t: t.resize_(2, 3))(marked)


if __name__ == "__main__":
    run_tests()
