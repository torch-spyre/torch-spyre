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

from utils import _assert_tensor_equal

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


class TestGather(TestCase):
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

    def _test_gather_helper(self, shape, dtype, dst):
        """
        Helper method to test gather with specific parameters.

        Args:
            shape: Tensor shape
            dtype: Tensor data type
            dst: Destination rank for gather
        """
        # Calculate total number of elements in the tensor
        num_elements = torch.tensor(shape).prod().item()

        if dtype == torch.float16:
            assert self.comm_size * num_elements <= 1024, (
                f"float16 exact-integer range exceeded: rank {self.comm_rank}"
            )

        # Create contiguous range for this rank: rank 0 gets [0..num_elements-1],
        # rank 1 gets [num_elements..2*num_elements-1], etc.
        start_value = self.comm_rank * num_elements
        end_value = start_value + num_elements

        # Create tensor with contiguous values for this rank
        input_tensor = torch.arange(start_value, end_value, dtype=dtype).reshape(shape)
        input_device = input_tensor.to(DEVICE)

        if self.comm_rank == dst:
            gather_list = [
                torch.zeros_like(input_device) for _ in range(self.comm_size)
            ]
            dist.gather(input_device, gather_list=gather_list, dst=dst)

            # Build all expected slices at once and compare in bulk.
            # All ranks contribute contiguous blocks: rank r owns [r*N .. (r+1)*N - 1],
            # so the full expected output is simply arange(comm_size * N) reshaped.
            all_results = torch.stack(
                [gather_list[r].to("cpu") for r in range(self.comm_size)]
            )
            all_expected = torch.arange(
                self.comm_size * num_elements, dtype=dtype
            ).reshape(self.comm_size, *shape)
            _assert_tensor_equal(
                all_results,
                all_expected,
                dtype,
                f"Rank {self.comm_rank}: gather result incorrect at destination rank {dst}",
            )
        else:
            dist.gather(input_device, gather_list=None, dst=dst)
            self.assertTrue(True, "Non-destination rank completed gather successfully")

    def test_gather_float16(self):
        """Test gather to rank 0 with float16 tensors."""
        self._test_gather_helper(shape=(128,), dtype=torch.float16, dst=0)

    def test_gather_float32(self):
        """Test gather to rank 0 with float32 tensors."""
        self._test_gather_helper(shape=(256,), dtype=torch.float32, dst=0)

    def test_gather_int32(self):
        """Test gather to rank 0 with int32 tensors."""
        self._test_gather_helper(shape=(192,), dtype=torch.int32, dst=0)

    def test_gather_2d_tensor_float16(self):
        """Test gather with 2D tensor shapes using float16."""
        self._test_gather_helper(shape=(4, 64), dtype=torch.float16, dst=0)

    def test_gather_2d_tensor_float32(self):
        """Test gather with 2D tensor shapes using float32."""
        self._test_gather_helper(shape=(4, 64), dtype=torch.float32, dst=0)

    def test_gather_2d_tensor_int32(self):
        """Test gather with 2D tensor shapes using int32."""
        self._test_gather_helper(shape=(4, 64), dtype=torch.int32, dst=0)

    def test_gather_rank_non_zero_float16(self):
        """Test gather to non-zero destination rank with float16 tensors."""
        dst_rank = min(1, self.comm_size - 1)
        self._test_gather_helper(shape=(128,), dtype=torch.float16, dst=dst_rank)

    def test_gather_rank_non_zero_float32(self):
        """Test gather to non-zero destination rank with float32 tensors."""
        dst_rank = min(1, self.comm_size - 1)
        self._test_gather_helper(shape=(256,), dtype=torch.float32, dst=dst_rank)

    def test_gather_rank_non_zero_int32(self):
        """Test gather to non-zero destination rank with int32 tensors."""
        dst_rank = min(1, self.comm_size - 1)
        self._test_gather_helper(shape=(192,), dtype=torch.int32, dst=dst_rank)

    def test_gather_2d_tensor_rank_non_zero_float16(self):
        """Test gather to non-zero destination rank with 2D tensor shapes using float16."""
        dst_rank = min(1, self.comm_size - 1)
        self._test_gather_helper(shape=(4, 64), dtype=torch.float16, dst=dst_rank)

    def test_gather_2d_tensor_rank_non_zero_float32(self):
        """Test gather to non-zero destination rank with 2D tensor shapes using float32."""
        dst_rank = min(1, self.comm_size - 1)
        self._test_gather_helper(shape=(4, 64), dtype=torch.float32, dst=dst_rank)

    def test_gather_2d_tensor_rank_non_zero_int32(self):
        """Test gather to non-zero destination rank with 2D tensor shapes using int32."""
        dst_rank = min(1, self.comm_size - 1)
        self._test_gather_helper(shape=(4, 64), dtype=torch.int32, dst=dst_rank)


if __name__ == "__main__":
    run_tests()
