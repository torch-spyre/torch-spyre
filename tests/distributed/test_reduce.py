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


class TestReduce(TestCase):
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

    def _test_reduce_helper(self, shape, dtype, dst, async_op=False):
        """
        Helper method to test reduce with specific parameters.

        Args:
            shape: Tensor shape
            dtype: Tensor data type
            dst: Destination rank for reduce
            async_op: If True, launch the collective asynchronously and call
                      work.wait() before inspecting the result
        """
        # Calculate total number of elements in the tensor
        num_elements = torch.tensor(shape).prod().item()

        if num_elements > 256:
            self.skipTest(
                f"Reduce test case is not designed for more than 256 elements, got {num_elements}"
            )

        # Create bounded float16 values for this rank to avoid overflow during SUM.
        start_value = self.comm_rank + 1
        input_tensor = (
            (torch.arange(num_elements, dtype=torch.float32).div(10.0).add(start_value))
            .to(dtype)
            .reshape(shape)
        )
        input_device = input_tensor.to(DEVICE)

        # Reduce with SUM operation to destination rank
        work = dist.reduce(
            input_device, dst=dst, op=dist.ReduceOp.SUM, async_op=async_op
        )

        if async_op:
            self.assertIsNotNone(work, "async_op=True must return a Work handle")
            work.wait()
            self.assertTrue(work.is_completed())
        else:
            self.assertIsNone(work, "async_op=False must return None")

        if self.comm_rank == dst:
            result = input_device.to("cpu")

            # Expected result: comm_size * (i / 10.0) + sum_{r=1..comm_size}(r)
            offset = self.comm_size * (self.comm_size + 1) / 2
            expected = (
                (
                    torch.arange(num_elements, dtype=torch.float32)
                    .div(10.0)
                    .mul(self.comm_size)
                    .add(offset)
                )
                .to(dtype)
                .reshape(shape)
            )

            _assert_tensor_equal(
                result,
                expected,
                dtype,
                f"Rank {self.comm_rank}: reduce result incorrect at destination rank {dst}",
                atol=0.2,
            )
        else:
            self.assertTrue(True, "Non-destination rank completed reduce successfully")

    def test_reduce_float16(self):
        """Test reduce to rank 0 with float16 tensors."""
        self._test_reduce_helper(shape=(128,), dtype=torch.float16, dst=0)

    def test_reduce_2d_tensor_float16(self):
        """Test reduce with 2D tensor shapes using float16."""
        self._test_reduce_helper(shape=(4, 64), dtype=torch.float16, dst=0)

    def test_reduce_rank_non_zero_float16(self):
        """Test reduce to non-zero destination rank with float16 tensors."""
        dst_rank = min(1, self.comm_size - 1)
        self._test_reduce_helper(shape=(128,), dtype=torch.float16, dst=dst_rank)

    def test_reduce_2d_tensor_rank_non_zero_float16(self):
        """Test reduce to non-zero destination rank with 2D tensor shapes using float16."""
        dst_rank = min(1, self.comm_size - 1)
        self._test_reduce_helper(shape=(4, 64), dtype=torch.float16, dst=dst_rank)

    def test_reduce_float16_async(self):
        """Test reduce to rank 0 with float16 tensors using async_op=True."""
        self._test_reduce_helper(
            shape=(128,), dtype=torch.float16, dst=0, async_op=True
        )

    def test_reduce_2d_tensor_float16_async(self):
        """Test reduce with 2D float16 tensors using async_op=True."""
        self._test_reduce_helper(
            shape=(4, 64), dtype=torch.float16, dst=0, async_op=True
        )

    def test_reduce_rank_non_zero_float16_async(self):
        """Test reduce to non-zero destination rank with float16 tensors using async_op=True."""
        dst_rank = min(1, self.comm_size - 1)
        self._test_reduce_helper(
            shape=(128,), dtype=torch.float16, dst=dst_rank, async_op=True
        )

    def test_reduce_2d_tensor_rank_non_zero_float16_async(self):
        """Test reduce with 2D float16 tensors to a non-zero destination rank using async_op=True."""
        dst_rank = min(1, self.comm_size - 1)
        self._test_reduce_helper(
            shape=(4, 64), dtype=torch.float16, dst=dst_rank, async_op=True
        )


if __name__ == "__main__":
    run_tests()
