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

# Owner(s): ["module: cpp"]

import os
import regex as re
import unittest
import psutil
import warnings
from contextlib import contextmanager
import pytest

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    set_warn_always_context,
    subtest,
    TestCase,
)

# 0-dim scalar roundtrip: shared dtype × factory lambdas (used by parametrized fill test).
_SCALAR_ROUNDTRIP_DTYPE_CASES = [
    (torch.int8, lambda fn: fn((), dtype=torch.float32).to(dtype=torch.int8)),
    (torch.bool, lambda fn: fn((), dtype=torch.bool)),
    (torch.int64, lambda fn: fn((), dtype=torch.int64)),
    (
        torch.float8_e4m3fn,
        lambda fn: fn((), dtype=torch.float32).to(dtype=torch.float8_e4m3fn),
    ),
    (torch.float16, lambda fn: fn((), dtype=torch.float16)),
    (torch.float32, lambda fn: fn((), dtype=torch.float32)),
]

# Applied per parametrized variant (see test_cross_device_copy_scalar_fill).
# TODO: ISSUE: https://github.com/torch-spyre/torch-spyre/issues/1172
_SCALAR_FILL_XFAIL = pytest.mark.xfail(reason="Support 0-dim tensors in Spyre")

# TODO: ISSUE: https://github.com/torch-spyre/torch-spyre/issues/1153 (to_dtype / Inductor)
_SCALAR_ADD_XFAIL_TO_DTYPE = pytest.mark.xfail(
    reason="Support scalar eager add with to_dtype lowering in Spyre"
)
# TODO: ISSUE: https://github.com/torch-spyre/torch-spyre/issues/1474 (DataFormats.SEN143_FP8)
_SCALAR_ADD_XFAIL_FP8 = pytest.mark.xfail(
    reason="Support scalar eager add for DataFormats.SEN143_FP8 in Spyre"
)
# TODO: ISSUE: https://github.com/torch-spyre/torch-spyre/issues/925
_SCALAR_ADD_SKIP_INT = pytest.mark.skip(
    reason="Spyre backend does not support int32/int16 dtype - causes segfault/crash in data format converter"
)

# TODO:ISSUE: https://github.com/torch-spyre/torch-spyre/issues/1588
_SCALAR_ADD_SKIP_UINT8 = pytest.mark.skip(
    reason="Spyre hardware requires 128-byte aligned buffers - small uint8 tensors cause alignment violations"
)

_SCALAR_ADD_FALLBACK_FULL_WARN = r"torch\.ops\.spyre\.full is falling back to cpu"


@instantiate_parametrized_tests
class TestSpyre(TestCase):
    def test_initializes(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "spyre")

    @unittest.skip("Skip for now")
    def test_autograd_init(self):
        # Make sure autograd is initialized
        torch.ones(2, requires_grad=True, device="spyre").sum().backward()

        pid = os.getpid()
        task_path = f"/proc/{pid}/task"
        all_threads = psutil.Process(pid).threads()

        all_thread_names = set()

        for t in all_threads:
            with open(f"{task_path}/{t.id}/comm") as file:
                thread_name = file.read().strip()
            all_thread_names.add(thread_name)

        for i in range(torch.spyre._device_daemon.NUM_DEVICES):
            self.assertIn(f"pt_autograd_{i}", all_thread_names)

    def test_empty_factory(self):
        a = torch.empty(50, device="spyre", dtype=torch.float16)
        self.assertEqual(a.device.type, "spyre")

        a.fill_(3.5)

        a_cpu = a.cpu()
        self.assertTrue(a_cpu.eq(3.5).all())

    def test_ones_factory(self):
        a = torch.ones(50, device="spyre", dtype=torch.float16)
        self.assertEqual(a.device.type, "spyre")
        a_cpu = a.cpu()
        self.assertTrue(a_cpu.eq(1.0).all())

    def test_str(self):
        a = torch.tensor([1, 2], dtype=torch.float16).to("spyre")
        a_repr = str(a)

        def normalize_device(s):
            return re.sub(r"(device='spyre):\d+'", r"\1:0'", s)

        a_repr = normalize_device(a_repr)

        # Check the the print includes all elements and Spyre device
        expected_a_repr = "tensor([1., 2.], dtype=torch.float16, device='spyre:0')"
        self.assertEqual(expected_a_repr, a_repr)

    def test_repr(self):
        a = torch.tensor([1.234242424234, 2], dtype=torch.float16).to("spyre")
        try:
            a_repr = f"{a}"
        except RuntimeError as err:
            self.fail(f"Printing tensor failed with runtime error {err}")

        def normalize_device(s):
            return re.sub(r"(device='spyre):\d+'", r"\1:0'", s)

        a_repr = normalize_device(a_repr)

        # Check the the print includes all elements and Spyre device
        expected_a_repr = (
            "tensor([1.2344, 2.0000], dtype=torch.float16, device='spyre:0')"
        )
        self.assertEqual(expected_a_repr, a_repr)

    def test_printing(self):
        t = torch.ones((2, 3), device="spyre", dtype=torch.float16)

        # Try printing
        try:
            print(t)
            print("Tensor printing works!")
        except NotImplementedError as e:
            self.fail(f"Spyre backend should support tensor printing: {e}")

    @contextmanager
    def _device_transfer_warnings(
        self,
        expect_downcast_warning=False,
        ignored_warning_message=None,
    ):
        """CPU↔spyre transfer tests: PyTorch-style warn_always + recorded warnings.

        Uses ``set_warn_always_context(True)`` (same as ``TestCase.assertWarnsOnceRegex``)
        so TORCH_WARN_ONCE behaves consistently, then applies Spyre-specific int64
        downcast expectations (and optional ignored fallback warnings).
        """
        with set_warn_always_context(True):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                if ignored_warning_message is not None:
                    warnings.filterwarnings(
                        "ignore",
                        message=ignored_warning_message,
                    )
                yield

        msg = "does not support int64"
        recorded = [str(w.message) for w in rec]
        if expect_downcast_warning:
            self.assertTrue(
                any(msg in s for s in recorded),
                f"Expected warning containing '{msg}', got: {recorded}",
            )
        else:
            self.assertEqual(len(recorded), 0, f"Expected no warnings, got: {recorded}")

    def _assert_roundtrip_close(self, x_original, x_roundtrip, dtype):
        if dtype == torch.float8_e4m3fn:
            torch.testing.assert_close(
                x_original.float(),
                x_roundtrip.float(),
            )
        else:
            torch.testing.assert_close(
                x_original,
                x_roundtrip,
                rtol=2e-3,
                atol=1e-5,
                check_dtype=False,
            )

    def _assert_zero_dim_roundtrip(self, x_original, x_roundtrip, dtype):
        self.assertEqual(x_original.ndim, 0)
        self.assertEqual(x_original.numel(), 1)
        self.assertEqual(x_roundtrip.ndim, 0)
        self.assertEqual(x_roundtrip.numel(), 1)
        self._assert_roundtrip_close(x_original, x_roundtrip, dtype)

    def _roundtrip_to_spyre_and_back(self, x, expect_warning=False):
        self.assertEqual(x.device.type, "cpu", "initial device is not cpu")

        with self._device_transfer_warnings(expect_downcast_warning=expect_warning):
            x_spyre = x.to("spyre")

        self.assertEqual(x_spyre.device.type, "spyre", "to device is not spyre")
        self.assertEqual(x_spyre.dtype, x.dtype)

        return x_spyre.to("cpu")

    def _run_scalar_add_case(self, dtype, scalar_factory, ignored_warning_message=None):
        dtypes_with_downcast_warning = {torch.int64}
        a = scalar_factory()
        self.assertEqual(a.ndim, 0)
        self.assertEqual(a.numel(), 1)

        with self._device_transfer_warnings(
            expect_downcast_warning=dtype in dtypes_with_downcast_warning,
            ignored_warning_message=ignored_warning_message,
        ):
            b = a.to(device="spyre").add(2.0).to(device="cpu")

        self.assertEqual(b.ndim, 1)
        self.assertEqual(b.numel(), 1)

        expected = a + 2
        if dtype == torch.float8_e4m3fn:
            torch.testing.assert_close(b.float(), expected.reshape(1).float())
        else:
            torch.testing.assert_close(
                b,
                expected.reshape(1),
                rtol=2e-3,
                atol=1e-5,
                check_dtype=False,
            )

    # ISSUE: https://github.com/torch-spyre/torch-spyre/issues/1187
    @parametrize(
        "factory_name",
        [
            subtest("zeros", name="zeros", decorators=[_SCALAR_FILL_XFAIL]),
            subtest("ones", name="ones", decorators=[_SCALAR_FILL_XFAIL]),
        ],
    )
    def test_cross_device_copy_scalar_fill(self, factory_name):
        tensor_factory = getattr(torch, factory_name)
        for dtype, scalar_factory in _SCALAR_ROUNDTRIP_DTYPE_CASES:
            x = scalar_factory(tensor_factory)
            x_cpu = self._roundtrip_to_spyre_and_back(
                x, expect_warning=dtype in {torch.int64}
            )
            self._assert_zero_dim_roundtrip(x, x_cpu, dtype)

    # Scalar 0-dim: spyre -> add(2.) -> cpu. int8/bool/int64: Inductor to_dtype in add (#1153).
    # float8: SEN143_FP8 unsupported. float16/float32: pass; ignore known full() CPU fallback warn.
    @parametrize(
        "dtype, scalar_factory, ignored_warning_message",
        [
            subtest(
                (torch.int8, lambda: torch.tensor(10, dtype=torch.int8), None),
                name="int8",
                decorators=[_SCALAR_ADD_XFAIL_TO_DTYPE],
            ),
            subtest(
                (torch.bool, lambda: torch.tensor(True, dtype=torch.bool), None),
                name="bool",
                decorators=[_SCALAR_ADD_XFAIL_TO_DTYPE],
            ),
            subtest(
                (torch.int64, lambda: torch.tensor(10, dtype=torch.int64), None),
                name="int64",
                decorators=[_SCALAR_ADD_XFAIL_TO_DTYPE],
            ),
            subtest(
                (torch.uint8, lambda: torch.tensor(10, dtype=torch.uint8), None),
                name="uint8",
                decorators=[_SCALAR_ADD_SKIP_UINT8],
            ),
            subtest(
                (torch.int16, lambda: torch.tensor(10, dtype=torch.int16), None),
                name="int16",
                decorators=[_SCALAR_ADD_SKIP_INT],
            ),
            subtest(
                (torch.int32, lambda: torch.tensor(10, dtype=torch.int32), None),
                name="int32",
                decorators=[_SCALAR_ADD_SKIP_INT],
            ),
            subtest(
                (
                    torch.float8_e4m3fn,
                    lambda: torch.tensor(10.0, dtype=torch.float32).to(
                        dtype=torch.float8_e4m3fn
                    ),
                    None,
                ),
                name="float8_e4m3fn",
                decorators=[_SCALAR_ADD_XFAIL_FP8],
            ),
            subtest(
                (
                    torch.float16,
                    lambda: torch.tensor(10, dtype=torch.float16),
                    _SCALAR_ADD_FALLBACK_FULL_WARN,
                ),
                name="float16",
            ),
            subtest(
                (
                    torch.float32,
                    lambda: torch.tensor(10, dtype=torch.float32),
                    _SCALAR_ADD_FALLBACK_FULL_WARN,
                ),
                name="float32",
            ),
        ],
    )
    def test_cross_device_copy_scalar_add(
        self, dtype, scalar_factory, ignored_warning_message
    ):
        self._run_scalar_add_case(
            dtype,
            scalar_factory,
            ignored_warning_message=ignored_warning_message,
        )

    def test_cross_device_copy(self):
        a = torch.rand(10, dtype=torch.float16)
        b = a.to(device="spyre").add(2.0).to(device="cpu")
        self.assertEqual(b, a + 2)

    def test_cross_device_copy_dtypes(self):
        dtype_configs = {
            torch.int8: lambda: (torch.rand(64, 64) * 100).to(dtype=torch.int8),
            torch.bool: lambda: torch.randint(0, 2, (64, 64), dtype=torch.bool),
            torch.int64: lambda: torch.randint(
                -32768, 32767, (64, 64), dtype=torch.int64
            ),
            torch.float8_e4m3fn: lambda: torch.rand(64, 64).to(
                dtype=torch.float8_e4m3fn
            ),
            torch.float16: lambda: torch.rand(64, 64, dtype=torch.float16),
            torch.float32: lambda: torch.rand(64, 64, dtype=torch.float32),
        }

        dtypes_with_downcast_warning = {torch.int64}

        for dtype, tensor_factory in dtype_configs.items():
            x = tensor_factory()
            x_cpu = self._roundtrip_to_spyre_and_back(
                x, expect_warning=dtype in dtypes_with_downcast_warning
            )
            self._assert_roundtrip_close(x, x_cpu, dtype)

    @unittest.skip("Skip for now")
    def test_data_dependent_output(self):
        cpu_a = torch.randn(10)
        a = cpu_a.to(device="spyre")
        mask = a.gt(0)
        out = torch.masked_select(a, mask)

        self.assertEqual(out, cpu_a.masked_select(cpu_a.gt(0)))

    # simple test to make sure allocation size is different between spyre and cpu
    # this will be built out more once we have an op running in spyre
    # currently this never finishes because of an issue with closing the
    # program -- that will be solved in separate PR
    # (this was tested in isolation)
    def test_allocation_size(self):
        x = torch.tensor([1, 2], dtype=torch.float16, device="spyre")
        y = torch.tensor([1, 2], dtype=torch.float16)
        x_storage_nbytes = x.untyped_storage().nbytes()
        self.assertEqual(x_storage_nbytes, 128)
        self.assertNotEqual(
            x_storage_nbytes,
            y.untyped_storage().nbytes(),
            "spyre vs cpu allocation size should differ",
        )

    def test_spyre_round_trip(self):
        dtypes = [torch.float16]
        for dtype in dtypes:
            x = torch.tensor([1, 2], dtype=dtype)
            self.assertEqual(x.device.type, "cpu", "initial device is not cpu")
            x_spyre = x.to("spyre")
            self.assertEqual(x_spyre.device.type, "spyre", "to device is not spyre")
            x_cpu = x_spyre.to("cpu")
            torch.testing.assert_close(
                x,
                x_cpu,
                msg=f"round trip copy produces incorrect results for dtype={dtype}",
            )

    def test_default_on_import(self):
        import torch_spyre  # noqa: F401

        self.assertTrue(torch.spyre.get_downcast_warning())

    def test_set_get_roundtrip(self):
        import torch_spyre  # noqa: F401

        torch.spyre.set_downcast_warning(False)
        self.assertFalse(torch.spyre.get_downcast_warning())
        torch.spyre.set_downcast_warning(True)
        self.assertTrue(torch.spyre.get_downcast_warning())

    def test_warning_emitted_when_enabled(self):
        import torch_spyre  # noqa: F401

        t = torch.randint(-32768, 32767, (64, 64), dtype=torch.int64)
        torch.spyre.set_downcast_warning(True)
        with set_warn_always_context(True):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                t.to(device="spyre")
        self.assertTrue(
            any("does not support int64" in str(w.message) for w in rec),
            msg="expected int64 downcast warning when enabled",
        )

    def test_warning_suppressed_when_disabled(self):
        import torch_spyre  # noqa: F401

        torch.spyre.set_downcast_warning(False)
        t = torch.randint(-32768, 32767, (64, 64), dtype=torch.int64)
        with set_warn_always_context(True):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                t.to(device="spyre")
        self.assertEqual(len(rec), 0)

    def test_allocation_and_copy_dtypes(self):
        # allocation and device to host cases
        for dtype in [
            torch.float16,
            torch.float32,
            torch.bool,
            torch.int8,
            torch.bfloat16,
        ]:
            x = torch.empty(64, dtype=dtype, device="spyre")
            x.cpu()

        for dtype in [torch.float64]:
            with self.assertRaises(RuntimeError):
                x = torch.empty(64, dtype=dtype, device="spyre")
                x.cpu()

        # allocation and host to device cases
        for dtype in [
            torch.float16,
            torch.float32,
            torch.bool,
            torch.int8,
            torch.bfloat16,
        ]:
            x = torch.empty(64, dtype=dtype)
            x.to("spyre")

        for dtype in [torch.float64]:
            with self.assertRaises(RuntimeError):
                x = torch.empty(64, dtype=dtype)
                x.to("spyre")

    def test_detach(self):
        # exercises the shallow copy code path
        for dtype in [torch.float16, torch.float32, torch.bool, torch.int8]:
            x = torch.empty(64, dtype=dtype, device="spyre")
            x.detach().cpu()

    def test_hooks_on_import(self):
        dev = torch._C._get_accelerator()
        self.assertEqual(str(dev), "spyre")

    def test_memory_allocated(self):
        torch.spyre.memory.reset_peak_memory_stats()
        torch.spyre.memory.reset_accumulated_memory_stats()

        prev_allocated = torch.spyre.memory.memory_allocated()
        prev_max_allocated = torch.spyre.memory.max_memory_allocated()

        self.assertEqual(
            prev_allocated, prev_max_allocated
        )  # Due to reset_peak_memory_stats
        x = torch.rand((64, 64), dtype=torch.float16)
        mem_size = x.numel() * x.element_size()  # 8192 bytes
        self.assertEqual(x.device.type, "cpu")
        self.assertEqual(torch.spyre.memory.memory_allocated(), prev_allocated)

        x = x.to("spyre")
        self.assertEqual(x.device.type, "spyre")
        self.assertEqual(
            torch.spyre.memory.memory_allocated(), prev_allocated + mem_size
        )

        del x
        self.assertEqual(torch.spyre.memory.memory_allocated(), prev_allocated)

        # Test max
        self.assertEqual(
            torch.spyre.memory.max_memory_allocated(), prev_max_allocated + mem_size
        )

    def test_spyre_device_count_and_set_device(self):
        count = torch.spyre.device_count()

        assert isinstance(count, int)
        assert count > 0

        orig = torch.spyre.current_device()

        try:
            for i in range(min(2, count)):
                torch.spyre.set_device(i)
                assert torch.spyre.current_device() == i

            with pytest.raises(Exception):
                torch.spyre.set_device(count)

            with pytest.raises(Exception):
                torch.spyre.set_device(-1)
        finally:
            torch.spyre.set_device(orig)

    def test_instantiate_device_type_tests_mro(self):
        """Verify that instantiate_device_type_tests works with TestCase
        base class and only_for=("privateuse1",).

        Previously, inheriting from PrivateUse1TestBase caused an MRO
        conflict when instantiate_device_type_tests tried to create a
        dynamic subclass that also inherits PrivateUse1TestBase.
        Using plain TestCase + only_for avoids the conflict.
        """
        from torch.testing._internal.common_device_type import (
            instantiate_device_type_tests,
        )

        class _TestMROCheck(TestCase):
            def test_device_is_spyre(self):
                pass

        ns = {"_TestMROCheck": _TestMROCheck}
        # This must not raise TypeError about MRO
        instantiate_device_type_tests(_TestMROCheck, ns, only_for=("privateuse1",))

        # instantiate_device_type_tests should create a class named
        # _TestMROCheckPRIVATEUSE1 in the namespace
        assert "_TestMROCheckPRIVATEUSE1" in ns, (
            f"Expected _TestMROCheckPRIVATEUSE1 in namespace, got {list(ns)}"
        )

        # The generated class should be instantiable (valid MRO)
        cls = ns["_TestMROCheckPRIVATEUSE1"]
        assert issubclass(cls, TestCase)

    def test_device_to_device(self):
        """Test simple device-to-device copy using tensor.copy_() method."""
        src = torch.randn(3, dtype=torch.float16, device="spyre")
        dst = torch.empty(3, dtype=torch.float16, device="spyre")

        dst.copy_(src)

        # Verify the copy worked
        assert torch.allclose(src.cpu(), dst.cpu())
        assert src.data_ptr() != dst.data_ptr()

    def test_device_to_device_with_view(self):
        """Test more complex device-to-device copy using tensor.copy_() method."""
        a = torch.randn(512, 512).to("spyre")
        b = torch.zeros((512, 512), device="spyre")
        c = b.view((64, 8, 512))
        b.copy_(a)
        assert torch.allclose(a.cpu(), b.cpu())
        assert torch.allclose(a.cpu().view(64, 8, 512), c.cpu())


if __name__ == "__main__":
    run_tests()
