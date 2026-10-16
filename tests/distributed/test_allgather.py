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
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from inductor.utils_inductor import expects_raise
from utils import (
    _assert_tensor_equal,
    _bfloat16_collective_expected,
    _bfloat16_rank_input,
    _collective_expected,
    _collective_rank_input,
    _native_bfloat16_tensor,
)

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
class TestAllGather(TestCase):
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

    def _test_allgather_helper(self, shape, dtype, async_op=False):
        """
        Helper method to test allgather with specific parameters.

        Args:
            shape: Tensor shape
            dtype: Tensor data type
            async_op: If True, launch the collective asynchronously and call
                      work.wait() before inspecting the result
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

        output_list = [torch.zeros_like(input_device) for _ in range(self.comm_size)]

        work = dist.all_gather(output_list, input_device, async_op=async_op)

        if async_op:
            self.assertIsNotNone(work, "async_op=True must return a Work handle")
            work.wait()
            self.assertTrue(work.is_completed())
        else:
            self.assertIsNone(work, "async_op=False must return None")

        # Build all expected slices at once and compare in bulk.
        # All ranks contribute contiguous blocks: rank r owns [r*N .. (r+1)*N - 1],
        # so the full expected output is simply arange(comm_size * N) reshaped.
        all_results = torch.stack(
            [output_list[r].to("cpu") for r in range(self.comm_size)]
        )
        all_expected = torch.arange(self.comm_size * num_elements, dtype=dtype).reshape(
            self.comm_size, *shape
        )
        _assert_tensor_equal(
            all_results,
            all_expected,
            dtype,
            f"Rank {self.comm_rank}: allgather result incorrect",
        )

    def test_allgather_float16(self):
        """Test allgather with float16 tensors."""
        self._test_allgather_helper(shape=(128,), dtype=torch.float16)

    def test_allgather_float32(self):
        """Test allgather with float32 tensors."""
        self._test_allgather_helper(shape=(256,), dtype=torch.float32)

    def test_allgather_int32(self):
        """Test allgather with int32 tensors."""
        self._test_allgather_helper(shape=(192,), dtype=torch.int32)

    def test_allgather_2d_tensor_float16(self):
        """Test allgather with 2D tensor shapes using float16."""
        self._test_allgather_helper(shape=(4, 64), dtype=torch.float16)

    def test_allgather_2d_tensor_float32(self):
        """Test allgather with 2D tensor shapes using float32."""
        self._test_allgather_helper(shape=(4, 64), dtype=torch.float32)

    def test_allgather_2d_tensor_int32(self):
        """Test allgather with 2D tensor shapes using int32."""
        self._test_allgather_helper(shape=(4, 64), dtype=torch.int32)

    def test_allgather_float16_async(self):
        """Test allgather with float16 tensors using async_op=True."""
        self._test_allgather_helper(shape=(128,), dtype=torch.float16, async_op=True)

    def test_allgather_float32_async(self):
        """Test allgather with float32 tensors using async_op=True."""
        self._test_allgather_helper(shape=(256,), dtype=torch.float32, async_op=True)

    def test_allgather_int32_async(self):
        """Test allgather with int32 tensors using async_op=True."""
        self._test_allgather_helper(shape=(192,), dtype=torch.int32, async_op=True)

    def test_allgather_2d_tensor_float16_async(self):
        """Test allgather with 2D float16 tensors using async_op=True."""
        self._test_allgather_helper(shape=(4, 64), dtype=torch.float16, async_op=True)

    @parametrize("shape", [(128,), (4, 64), (4, 96)])
    @parametrize("async_op", [False, True])
    def test_allgather_bfloat16(self, shape, async_op):
        x = _bfloat16_rank_input(self.comm_rank, shape).to(DEVICE)
        outputs = [torch.empty_like(x) for _ in range(self.comm_size)]
        work = dist.all_gather(outputs, x, async_op=async_op)
        if async_op:
            self.assertIsNotNone(work)
            work.wait()
        else:
            self.assertIsNone(work)
        self.assertEqual(
            torch.cat([part.cpu() for part in outputs]),
            _bfloat16_collective_expected("allgather", shape, self.comm_size),
            rtol=0,
            atol=0,
        )

    @parametrize("async_op", [False, True])
    def test_allgather_float16_padded(self, async_op):
        shape = (4, 96)
        x = _collective_rank_input(self.comm_rank, shape, torch.float16).to(DEVICE)
        outputs = [torch.empty_like(x) for _ in range(self.comm_size)]
        work = dist.all_gather(outputs, x, async_op=async_op)
        if async_op:
            work.wait()
        self.assertEqual(
            torch.cat([part.cpu() for part in outputs]),
            _collective_expected("allgather", shape, self.comm_size, torch.float16),
            rtol=0,
            atol=0,
        )

    @parametrize("prefix", [False, True])
    @expects_raise
    def test_allgather_bfloat16_rejects_non_dl16_storage(self, prefix):
        x = _native_bfloat16_tensor(DEVICE)
        if prefix:
            x = x[:64]
        with self.assertRaisesRegex(
            RuntimeError, "BF16 collectives require DL16 device storage"
        ):
            outputs = [torch.empty_like(x) for _ in range(self.comm_size)]
            dist.all_gather(outputs, x)

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("async_op", [False, True])
    @parametrize("input_kind", ["matmul", "vocab_prefix", "row_prefix"])
    @parametrize("vocab_size", [96, 128])
    def test_allgather_matmul_logits(self, dtype, async_op, input_kind, vocab_size):
        # The eager LM head unpads a matmul output before gathering logits.
        # With one row, a vocabulary prefix is contiguous even though its
        # descriptor still covers the wider allocation.
        rows = 1 if input_kind == "vocab_prefix" else 4
        base_rows = 8 if input_kind == "row_prefix" else rows
        width = vocab_size + 64 if input_kind == "vocab_prefix" else vocab_size
        hidden = torch.zeros((base_rows, 64), dtype=dtype)
        hidden[:, 0] = torch.arange(1, base_rows + 1, dtype=dtype)
        weights = [
            _collective_rank_input(rank, (64, width), dtype)
            * (1 + torch.arange(width) // 64).to(dtype)
            for rank in range(self.comm_size)
        ]
        base = hidden.to(DEVICE) @ weights[self.comm_rank].to(DEVICE)
        logits = base[:rows, :vocab_size]
        self.assertTrue(logits.is_contiguous())
        outputs = [torch.empty_like(logits) for _ in range(self.comm_size)]
        work = dist.all_gather(outputs, logits, async_op=async_op)
        # In async mode the work must own any materialized input until done.
        del logits, base
        if async_op:
            self.assertIsNotNone(work)
            work.wait()
            self.assertTrue(work.is_completed())
        else:
            self.assertIsNone(work)
        for rank, output in enumerate(outputs):
            expected = (hidden.float() @ weights[rank].float())[:rows, :vocab_size].to(
                dtype
            )
            self.assertEqual(output.dtype, dtype)
            self.assertEqual(output.cpu(), expected, rtol=0, atol=0)

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @expects_raise
    def test_allgather_rejects_nonzero_storage_offset(self, dtype):
        x = torch.ones((5, 96), dtype=dtype, device=DEVICE)[1:]
        outputs = [torch.empty_like(x) for _ in range(self.comm_size)]
        with self.assertRaisesRegex(RuntimeError, "nonzero storage offsets"):
            dist.all_gather(outputs, x)


if __name__ == "__main__":
    run_tests()
