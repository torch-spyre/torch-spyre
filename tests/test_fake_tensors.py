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

"""Tests for FakeTensors on Spyre.

Some/all of these can be removed once upstream Issue:
https://github.com/pytorch/pytorch/issues/203190 is resolved.
"""

from torch.testing._internal.common_utils import run_tests, TestCase
import pytest


class TestFakeTensors(TestCase):
    def test_avoid_device_init(self):
        import torch  # noqa: F401
        import torch_spyre  # noqa: F401
        from torch._subclasses.fake_tensor import FakeTensorMode

        self.assertTrue(FakeTensorMode().avoid_device_init)

    def test_check_lift(self):
        import torch  # noqa: F401
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        with FakeTensorMode():
            self.assertTrue(torch._C._only_lift_cpu_tensors())
            t = torch.tensor(2.0, device="spyre")
            self.assertIsInstance(t, FakeTensor)
            self.assertEqual(t.device.type, "spyre")
        self.assertFalse(torch._C._only_lift_cpu_tensors())

    @pytest.mark.requires_spyre_profiler
    def test_memcpy_compile_using_profiler(self):
        """
        Test a simple compilation function.

        We expect no additional device copies happening during this compilation.
        """

        import torch
        from torch.profiler import profile, ProfilerActivity

        def fn(x):
            return x + 1

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        ) as prof:
            with torch._inductor.config.patch(force_disable_caches=True):
                x = torch.ones([1], dtype=torch.float16, device="spyre")
                fn_compiled = torch.compile(fn)
                fn_compiled(x).to("cpu")
            torch.spyre.synchronize()

        def count(prefix):
            return sum(e.count for e in prof.key_averages() if e.key.startswith(prefix))

        # in case FakeTensors are using the real device
        # compilation will show up a lot more memcpy's
        # around 54 H2D and 27 D2H were observed earlier
        self.assertEqual(count("Memcpy (HtoD)"), 2)
        self.assertEqual(count("Memcpy (DtoH)"), 1)
        self.assertEqual(count("launch::H2D"), 2)
        self.assertEqual(count("launch::D2H"), 1)


if __name__ == "__main__":
    run_tests()
