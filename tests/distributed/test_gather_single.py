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

"""Eager ``dist.gather_single`` / ``dist.gather_into_tensor`` on the spyreccl
backend. The list ``dist.gather`` is tested in ``test_gather.py``."""

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

from utils import _unique_rank_input

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
class TestGatherSingle(TestCase):
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

    SHAPES = [(64,), (4, 128), (3, 100)]

    @parametrize("shape", SHAPES)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("root", ["first", "last"])
    @parametrize("async_op", [False, True])
    def test_gather_single(self, shape, dtype, root, async_op):
        dst = 0 if root == "first" else self.comm_size - 1
        x = _unique_rank_input(self.comm_rank, shape, dtype).to(DEVICE)
        out = None
        if self.comm_rank == dst:
            out = torch.zeros(
                (self.comm_size * shape[0], *shape[1:]), dtype=dtype, device=DEVICE
            )
        work = dist.gather_single(x, out, dst=dst, async_op=async_op)
        self._finish(work, async_op)
        if self.comm_rank != dst:
            return
        expected = torch.cat(
            [_unique_rank_input(r, shape, dtype) for r in range(self.comm_size)]
        )
        self.assertEqual(out.cpu(), expected, rtol=0, atol=0)

    def test_gather_into_tensor(self):
        shape = (4, 128)
        x = _unique_rank_input(self.comm_rank, shape, torch.float16).to(DEVICE)
        out = None
        if self.comm_rank == 0:
            out = torch.zeros(
                (self.comm_size * 4, 128), dtype=torch.float16, device=DEVICE
            )
        with pytest.warns(FutureWarning):
            dist.gather_into_tensor(x, out, dst=0)
        if self.comm_rank == 0:
            expected = torch.cat(
                [
                    _unique_rank_input(r, shape, torch.float16)
                    for r in range(self.comm_size)
                ]
            )
            self.assertEqual(out.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    run_tests()
