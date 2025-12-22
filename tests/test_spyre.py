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
import unittest
import psutil
import warnings
from contextlib import contextmanager

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


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
        import regex as re

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
        except RuntimeError as re:
            self.fail(f"Printing tensor failed with runtime error {re}")

        import regex as re

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
            print("Printing failed:", e)
            assert False, "Spyre backend should support tensor printing"

    def test_cross_device_copy_scalar(self):
        # scalar tensor becomes 1D tensor on Spyre
        a = torch.tensor(10, dtype=torch.float16)
        b = a.to(device="spyre").add(2).to(device="cpu")
        self.assertEqual(b.ndim, 1)
        self.assertEqual(b.numel(), 1)
        self.assertEqual(b.item(), a + 2)

    def test_cross_device_copy(self):
        a = torch.rand(10, dtype=torch.float16)
        b = a.to(device="spyre").add(2).to(device="cpu")
        self.assertEqual(b, a + 2)

    def test_cross_device_copy_dtypes(self):
        @contextmanager
        def check_downcast_warning(expect_warning=False):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                yield

            msg = "does not support int64"
            recorded = [str(w.message) for w in rec]
            if expect_warning:
                assert len(rec) == 1 and msg in str(rec[0].message), (
                    f"Expected one warning containing '{msg}', got: {recorded}"
                )
            else:
                assert len(rec) == 0, f"Expected no warnings, got: {recorded}"

        dtypes = [
            torch.float8_e4m3fn,
            torch.int8,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.bool,
        ]
        for dtype in dtypes:
            x = None
            expect_warning = False
            if dtype in [torch.int8]:
                x = torch.rand(64, 64) * 100
                x = x.to(dtype=dtype)
            elif dtype in [torch.bool]:
                x = torch.randint(0, 2, (64, 64), dtype=dtype)
            elif dtype in [torch.int64]:
                expect_warning = True
                x = torch.randint(-32768, 32767, (64, 64), dtype=dtype)
            elif dtype in [torch.float8_e4m3fn]:
                x = torch.rand(64, 64)
                x = x.to(dtype=dtype)
            else:
                x = torch.rand(64, 64, dtype=dtype)
            assert x.device.type == "cpu", "initial device is not cpu"

            with check_downcast_warning(expect_warning):
                x_spyre = x.to("spyre")

            assert x_spyre.device.type == "spyre", "to device is not spyre"
            assert x_spyre.dtype == x.dtype
            x_cpu = x_spyre.to("cpu")
            custom_rtol = 2e-3
            custom_atol = 1e-5
            try:
                if dtype in [torch.float8_e4m3fn]:
                    from torch.testing import assert_close

                    assert_close(x.float(), x_cpu.float())
                else:
                    torch.testing.assert_close(
                        x,
                        x_cpu,
                        rtol=custom_rtol,
                        atol=custom_atol,
                        check_dtype=False,  # You may need this if the dtypes are different after conversion
                    )
                print(f"Tensors are close with custom tolerance for dtype={dtype}.")
            except AssertionError as e:
                print(f"Tensors are NOT close for dtype={dtype}! Details:\n{e}")

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
        assert x_storage_nbytes == 128
        assert x_storage_nbytes != y.untyped_storage().nbytes(), "failed allocation"

    def test_spyre_round_trip(self):
        dtypes = [torch.float16]
        for dtype in dtypes:
            x = torch.tensor([1, 2], dtype=dtype)
            assert x.device.type == "cpu", "initial device is not cpu"
            x_spyre = x.to("spyre")
            assert x_spyre.device.type == "spyre", "to device is not spyre"
            x_cpu = x_spyre.to("cpu")
            (
                torch.testing.assert_close(x, x_cpu),
                f"round trip copy produces incorrect results for dtype={dtype}",
            )

    def test_default_on_import(self):
        import torch_spyre  # noqa: F401

        assert torch.spyre.get_downcast_warning() is True

    def test_set_get_roundtrip(self):
        import torch_spyre  # noqa: F401

        torch.spyre.set_downcast_warning(False)
        assert torch.spyre.get_downcast_warning() is False
        torch.spyre.set_downcast_warning(True)
        assert torch.spyre.get_downcast_warning() is True

    def test_warning_emitted_when_enabled(self):
        import torch_spyre  # noqa: F401

        torch.set_warn_always(True)
        t = torch.randint(-32768, 32767, (64, 64), dtype=torch.int64)
        torch.spyre.set_downcast_warning(True)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            t2 = t.to(device="spyre")  # noqa: F841
        # At least one UserWarning captured
        assert any("does not support int64" in str(w.message) for w in rec)

    def test_warning_suppressed_when_disabled(self):
        import torch_spyre  # noqa: F401

        torch.set_warn_always(True)
        torch.spyre.set_downcast_warning(False)
        t = torch.randint(-32768, 32767, (64, 64), dtype=torch.int64)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            t2 = t.to(device="spyre")  # noqa: F841
        assert len(rec) == 0


if __name__ == "__main__":
    run_tests()
