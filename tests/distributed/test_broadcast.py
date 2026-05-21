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

# ------------
# Temporary hack
torch.spyre._impl._lazy_init()
# ------------

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"


class TestBroadcast(TestCase):
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
        # Add 'cpu:gloo' since we want to use the backend as well
        if not dist.is_initialized():
            dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

        cls.comm_size = dist.get_world_size()
        cls.comm_rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        """Clean up the distributed environment after all tests."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _test_broadcast_helper(self, shape, dtype, root, fill_value):
        """
        Helper method to test broadcast with specific parameters.

        Args:
            shape: Tensor shape
            dtype: Tensor data type
            root: Root rank for broadcast
            fill_value: Value to fill the tensor at root rank
        """
        # Create expected tensor (what all ranks should have after broadcast)
        expected_tensor = torch.zeros(shape, dtype=dtype)
        expected_tensor.fill_(fill_value)

        # At root rank: create tensor with expected value
        # At other ranks: create tensor with different value (ones)
        if self.comm_rank == root:
            x = expected_tensor.clone()
        else:
            x = torch.ones(shape, dtype=dtype)

        # Send input tensor to Spyre device
        x_device = x.to(DEVICE)

        # Broadcast from root rank
        dist.broadcast(x_device, root)

        # Get result back to CPU for verification
        result = x_device.to("cpu")

        # Verify result matches expected tensor at ALL ranks (including root)
        # print(f"Shape: {shape}")
        # print(f"Expected {expected_tensor}")
        # print(f"Got {result}")
        self.assertTrue(
            torch.allclose(result, expected_tensor, rtol=1e-5, atol=1e-5),
            f"Rank {self.comm_rank}: Broadcast result incorrect. "
            f"Expected {expected_tensor[:10] if expected_tensor.numel() > 10 else expected_tensor}, "
            f"got {result[:10] if result.numel() > 10 else result}",
        )

    def test_broadcast_from_rank_zero_float16(self):
        """Test broadcast from rank 0 with float16 data type."""
        self._test_broadcast_helper(
            shape=(128,), dtype=torch.float16, root=0, fill_value=2.0
        )

    def test_broadcast_from_rank_zero_float32(self):
        """Test broadcast from rank 0 with float32 data type."""
        self._test_broadcast_helper(
            shape=(256,), dtype=torch.float32, root=0, fill_value=3.5
        )

    def test_broadcast_from_rank_zero_2d_tensor(self):
        """Test broadcast from rank 0 with 2D tensor."""
        self._test_broadcast_helper(
            shape=(512, 1024), dtype=torch.float16, root=0, fill_value=4.0
        )

    def test_broadcast_from_rank_zero_large_tensor(self):
        """Test broadcast from rank 0 with large tensor."""
        self._test_broadcast_helper(
            shape=(1024, 2048), dtype=torch.float32, root=0, fill_value=1.5
        )

    def test_broadcast_from_non_zero_root(self):
        """
        Test broadcast from a non-zero root rank.
        Uses rank 1 if available, otherwise uses the last rank.
        """
        # Choose root rank: rank 1 if world size > 1, otherwise rank 0
        root_rank = 1 if self.comm_size > 1 else 0

        self._test_broadcast_helper(
            shape=(256,), dtype=torch.float32, root=root_rank, fill_value=7.5
        )

    def test_broadcast_from_last_rank(self):
        """Test broadcast from the last rank in the world."""
        root_rank = self.comm_size - 1

        self._test_broadcast_helper(
            shape=(128,), dtype=torch.float16, root=root_rank, fill_value=9.0
        )

    def test_broadcast_multiple_tensors_sequential(self):
        """Test broadcasting multiple tensors sequentially from rank 0."""
        # First broadcast
        self._test_broadcast_helper(
            shape=(64,), dtype=torch.float32, root=0, fill_value=1.0
        )

        # Second broadcast with different parameters
        self._test_broadcast_helper(
            shape=(128,), dtype=torch.float16, root=0, fill_value=2.0
        )

    def test_broadcast_zero_tensor(self):
        """Test broadcast with tensor filled with zeros."""
        self._test_broadcast_helper(
            shape=(128,), dtype=torch.float32, root=0, fill_value=0.0
        )

    def test_broadcast_negative_values(self):
        """Test broadcast with negative values."""
        self._test_broadcast_helper(
            shape=(128,), dtype=torch.float32, root=0, fill_value=-5.5
        )

    def test_broadcast_invalid_root_rank(self):
        """Test that broadcast with invalid root rank raises an error."""
        # Only test on rank 0 to avoid multiple error messages
        if self.comm_rank != 0:
            return

        invalid_root = self.comm_size + 10  # Invalid rank (out of bounds)
        x = torch.ones(10, dtype=torch.float32, device=DEVICE)

        # This should raise an error
        with self.assertRaises(Exception):
            dist.broadcast(x, invalid_root)


if __name__ == "__main__":
    run_tests()
