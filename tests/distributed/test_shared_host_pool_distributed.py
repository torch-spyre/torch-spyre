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

import os

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_spyre._C import SharedHostPool  # type: ignore[attr-defined]

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


class TestSharedHostPoolCrossProcess(TestCase):
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

    def test_cross_process_shared_pool(self):
        """
        A shared host pool created in one process must be visible to another
        process attaching the same name.
        """
        num_slots = 1
        slot_bytes = 10
        name = self.id()

        if self.comm_rank == 0:
            # Process 0: create the shared host pool.
            pool = SharedHostPool.create_or_attach(name, num_slots, slot_bytes)
            self.assertIsNotNone(pool)

        dist.barrier()  # Ensure process 0 has created the pool

        if self.comm_rank == 1:
            # Process 1: attach the same pool and confirm its geometry.
            pool = SharedHostPool.create_or_attach(name, num_slots, slot_bytes)
            self.assertEqual(pool.slot_count(), num_slots)
            self.assertEqual(pool.name(), name)
            self.assertEqual(pool.total_bytes(), pool.slot_count() * pool.slot_bytes())
            # Attaching with a different slot count only raises if this process
            # truly attached to process 0's pool.
            with self.assertRaises(RuntimeError):
                SharedHostPool.create_or_attach(name, num_slots + 1, slot_bytes)

        # Keep process 0's pool alive until process 1 is done with it.
        dist.barrier()


if __name__ == "__main__":
    run_tests()
