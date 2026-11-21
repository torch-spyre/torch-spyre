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

"""Eager ``dist.all_reduce_coalesced`` and ``dist.all_reduce`` inside the
coalescing manager on the spyreccl backend (not the compiled path; see
``test_all_reduce_coalesced_compiled.py``)."""

import os

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from utils import _collective_expected, _collective_rank_input

# Skip all tests if RANK is not defined, or WORLD_SIZE is not set or less than 2
if "RANK" not in os.environ:
    pytest.skip(
        "RANK environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

if "WORLD_SIZE" not in os.environ:
    pytest.skip(
        "WORLD_SIZE environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

try:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size < 2:
        pytest.skip(
            f"WORLD_SIZE is {world_size}, need at least 2 for distributed tests",
            allow_module_level=True,
        )
except ValueError:
    pytest.skip(
        "WORLD_SIZE environment variable is not a valid integer, skipping distributed tests",
        allow_module_level=True,
    )

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"


@instantiate_parametrized_tests
class TestAllReduceCoalesced(TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the distributed environment once for all tests."""
        if not dist.distributed_c10d.is_backend_available(C10D_BACKEND):
            raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
        if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
            raise RuntimeError(
                f"Error: Missing a C10 Backend for 'spyre'! Expected {C10D_BACKEND}"
            )

        if not dist.is_initialized():
            dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

        cls.comm_size = dist.get_world_size()
        cls.comm_rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        """Clean up the distributed environment after all tests."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _finish(self, work, async_op):
        if async_op:
            self.assertIsNotNone(work, "async_op=True must return a Work handle")
            work.wait()
            self.assertTrue(work.is_completed())
        else:
            self.assertIsNone(work, "async_op=False must return None")

    SHAPES = [(64,), (4, 128), (3, 104)]

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("async_op", [False, True])
    def test_all_reduce_coalesced(self, dtype, async_op):
        tensors = [
            _collective_rank_input(self.comm_rank, s, dtype).to(DEVICE)
            for s in self.SHAPES
        ]
        with pytest.warns(FutureWarning):
            work = dist.all_reduce_coalesced(tensors, async_op=async_op)
        if async_op:
            # This deprecated API returns the work's future, not the work.
            self.assertIsInstance(work, torch._C.Future)
            work.wait()
        else:
            self.assertIsNone(work)
        for t, s in zip(tensors, self.SHAPES):
            self.assertEqual(
                t.cpu(),
                _collective_expected("allreduce", s, self.comm_size, dtype),
                rtol=0,
                atol=0,
            )

    @parametrize("async_op", [False, True])
    def test_coalescing_manager(self, async_op):
        tensors = [
            _collective_rank_input(self.comm_rank, s, torch.float16).to(DEVICE)
            for s in self.SHAPES
        ]
        with dist.distributed_c10d._coalescing_manager(async_ops=async_op) as cm:
            for t in tensors:
                dist.all_reduce(t)
        if async_op:
            cm.wait()
        for t, s in zip(tensors, self.SHAPES):
            self.assertEqual(
                t.cpu(),
                _collective_expected("allreduce", s, self.comm_size, torch.float16),
                rtol=0,
                atol=0,
            )

    def test_rejects_non_sum_before_any_transfer(self):
        tensors = [
            torch.zeros(64, dtype=torch.float16, device=DEVICE) for _ in range(2)
        ]
        with self.assertRaisesRegex(RuntimeError, "only supports SUM operation"):
            with pytest.warns(FutureWarning):
                dist.all_reduce_coalesced(tensors, op=dist.ReduceOp.MAX)


if __name__ == "__main__":
    run_tests()
