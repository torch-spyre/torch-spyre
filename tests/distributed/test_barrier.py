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

import torch
import torch.distributed as dist
import time
import os
import pytest
from torch.testing._internal.common_utils import run_tests, TestCase

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


class TestBarrier(TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the distributed environment once for all tests."""
        # Check that the c10d backend was loaded properly
        if not dist.distributed_c10d.is_backend_available(C10D_BACKEND):
            raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
        if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
            raise RuntimeError(
                f"Error: Missing a C10 Backend for 'spyre'! Expected {C10D_BACKEND}"
            )

        # Initialize the distributed environment
        if not dist.is_initialized():
            dist.init_process_group(device_id=DEVICE)

        cls.comm_size = dist.get_world_size()
        cls.comm_rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        """Clean up the distributed environment after all tests."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_barrier_basic(self):
        """Test basic barrier functionality - all ranks should proceed together."""
        # Simple barrier test - just verify it doesn't hang or crash
        dist.barrier()
        # If we reach here, barrier worked
        self.assertTrue(True, "Barrier completed successfully")

    def test_barrier_synchronization_timing(self):
        """
        Test that barrier synchronizes all ranks by verifying timing.
        Each rank sleeps for a different duration, then all should wait
        at the barrier until the slowest rank arrives.
        """
        # Each rank sleeps for rank * 0.5 seconds
        # So rank 0: 0s, rank 1: 0.5s, rank 2: 1.0s, etc.
        sleep_duration = self.comm_rank * 0.5
        max_sleep = (self.comm_size - 1) * 0.5

        start_time = time.time()

        # Sleep for rank-dependent duration
        time.sleep(sleep_duration)

        # Barrier - all ranks wait here
        dist.barrier()

        elapsed_time = time.time() - start_time

        # The elapsed time should be at least as long as the maximum sleep time
        # across all ranks (with some tolerance for timing variations)
        tolerance = 0.2  # 200ms tolerance for timing variations
        self.assertGreaterEqual(
            elapsed_time,
            max_sleep - tolerance,
            f"Rank {self.comm_rank}: Barrier did not wait long enough. "
            f"Expected at least {max_sleep}s, got {elapsed_time:.3f}s",
        )

        # Also verify it didn't wait too long (sanity check)
        # Should not take more than max_sleep + 1 second
        self.assertLess(
            elapsed_time,
            max_sleep + 1.0,
            f"Rank {self.comm_rank}: Barrier waited too long. "
            f"Expected around {max_sleep}s, got {elapsed_time:.3f}s",
        )

    def test_barrier_multiple_calls(self):
        """Test that multiple consecutive barriers work correctly."""
        for i in range(3):
            # Add small rank-dependent delays between barriers
            time.sleep(self.comm_rank * 0.1)
            dist.barrier()

        # If we reach here, all barriers completed successfully
        self.assertTrue(True, "Multiple barriers completed successfully")


if __name__ == "__main__":
    run_tests()
