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
    def test_no_device_copies_under_fake_mode(self):
        import torch
        from torch._inductor.fx_passes import joint_graph
        from torch.profiler import profile, ProfilerActivity
        from torch._subclasses.fake_tensor import FakeTensorMode

        def _memcpys(prof):
            return sum(
                e.count
                for e in prof.key_averages()
                if e.key.startswith(("Memcpy", "launch::H2D", "launch::D2H"))
            )

        torch.ones(1, device="spyre")  # runtime init outside the profiled region
        acts = [ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]

        with profile(activities=acts) as prof:
            with FakeTensorMode():
                torch.tensor(2.0, device="spyre")
                torch.tensor([1.0, 2.0], device="spyre") * 2
        self.assertEqual(_memcpys(prof), 0)

        # The compile-time path that produced the original 78 copies. Clear the
        # per-process cache so this doesn't depend on test order.
        joint_graph.lazy_init.cache_clear()
        with profile(activities=acts) as prof:
            joint_graph.lazy_init(torch.device("spyre"))
        self.assertEqual(_memcpys(prof), 0)


if __name__ == "__main__":
    run_tests()
