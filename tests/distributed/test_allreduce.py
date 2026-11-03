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

import math
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

from utils import (
    NUMEL_512KB,
    _assert_reduction_close,
    _assert_tensor_equal,
    _bfloat16_collective_expected,
    _bfloat16_rank_input,
    _calculate_collective_tolerance,
    _check_work_handle,
    _chunk_numel,
    _collective_expected,
    _collective_rank_input,
    _exact_collective_case,
    _integer_case,
    _native_bfloat16_tensor,
    _ramp_case,
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
class TestAllReduce(TestCase):
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("async_op", [False, True])
    def test_allreduce_scalar(self, dtype, async_op):
        """SUM covers a scalar's logical value within its padded allocation."""
        scale = 2.0**20 if dtype == torch.bfloat16 else 16.0
        x = torch.tensor(scale * (self.comm_rank + 1), dtype=dtype).to(DEVICE)
        work = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=async_op)
        if async_op:
            self.assertIsNotNone(work)
            work.wait()
            self.assertTrue(work.is_completed())
        else:
            self.assertIsNone(work)
        expected = torch.tensor(
            scale * self.comm_size * (self.comm_size + 1) / 2, dtype=dtype
        )
        result = x.cpu()
        self.assertEqual(result.shape, torch.Size([]))
        self.assertEqual(result, expected, rtol=0, atol=0)

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

    def _test_allreduce_helper(self, shape, dtype, async_op=False):
        """
        Helper method to test allreduce with specific parameters.

        Args:
            shape: Tensor shape
            dtype: Tensor data type
            async_op: If True, launch the collective asynchronously and call
                      work.wait() before inspecting the result
        """
        # Calculate total number of elements in the tensor
        num_elements = torch.tensor(shape).prod().item()

        if num_elements > 256:
            self.skipTest(
                f"AllReduce test case is not designed for more than 256 elements, got {num_elements}"
            )

        # Bounded float16 values for this rank to avoid overflow during SUM.
        input_tensor, expected, atol = _ramp_case(
            num_elements, self.comm_rank, self.comm_size, dtype
        )
        input_device = input_tensor.reshape(shape).to(DEVICE)

        # Allreduce with SUM operation — all ranks receive the result
        work = dist.all_reduce(input_device, op=dist.ReduceOp.SUM, async_op=async_op)

        if async_op:
            self.assertIsNotNone(work, "async_op=True must return a Work handle")
            work.wait()
            self.assertTrue(work.is_completed())
        else:
            self.assertIsNone(work, "async_op=False must return None")

        result = input_device.to("cpu")

        _assert_tensor_equal(
            result,
            expected.reshape(shape),
            dtype,
            f"Rank {self.comm_rank}: allreduce result incorrect",
            atol=atol,
        )

    def test_allreduce_float16(self):
        """Test allreduce with float16 tensors."""
        self._test_allreduce_helper(shape=(128,), dtype=torch.float16)

    def test_allreduce_2d_tensor_float16(self):
        """Test allreduce with 2D tensor shapes using float16."""
        self._test_allreduce_helper(shape=(4, 64), dtype=torch.float16)

    def test_allreduce_3d_tensor_float16(self):
        """Test allreduce with 3D tensor shapes using float16."""
        self._test_allreduce_helper(shape=(2, 2, 64), dtype=torch.float16)

    def test_allreduce_4d_tensor_float16(self):
        """Test allreduce with 4D tensor shapes using float16."""
        self._test_allreduce_helper(shape=(2, 1, 2, 64), dtype=torch.float16)

    def test_allreduce_float16_async(self):
        """Test allreduce with float16 tensors using async_op=True."""
        self._test_allreduce_helper(shape=(128,), dtype=torch.float16, async_op=True)

    def test_allreduce_2d_tensor_float16_async(self):
        """Test allreduce with 2D tensor shapes using float16 and async_op=True."""
        self._test_allreduce_helper(shape=(4, 64), dtype=torch.float16, async_op=True)

    def test_allreduce_3d_tensor_float16_async(self):
        """Test allreduce with 3D tensor shapes using float16 and async_op=True."""
        self._test_allreduce_helper(
            shape=(2, 2, 64), dtype=torch.float16, async_op=True
        )

    def test_allreduce_4d_tensor_float16_async(self):
        """Test allreduce with 4D tensor shapes using float16 and async_op=True."""
        self._test_allreduce_helper(
            shape=(2, 1, 2, 64), dtype=torch.float16, async_op=True
        )

    def _test_allreduce_exact_helper(self, pattern, shape=(128,), async_op=False):
        """
        Allreduce a pattern whose SUM is exact in any reduction order, so the
        result must match bit for bit (atol=0, rtol=0).

        Args:
            pattern: Name understood by utils._exact_collective_case
            shape: Tensor shape
            async_op: If True, launch the collective asynchronously
        """
        own, expected = _exact_collective_case(
            pattern, math.prod(shape), self.comm_rank, self.comm_size
        )
        x = own.reshape(shape).to(DEVICE)

        work = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=async_op)
        _check_work_handle(self, work, async_op)

        self.assertEqual(
            x.cpu(),
            expected.reshape(shape),
            rtol=0,
            atol=0,
            msg=f"Rank {self.comm_rank}: allreduce {pattern} not exact",
        )

    def _test_allreduce_integer_helper(self, case, shape, async_op=False):
        """
        Allreduce non-negative integer inputs (exact in DLFloat16) so only the
        reduction itself can round. Sums <= 1024 must be bit exact; larger sums
        must be within accumulation depth * DLFloat16 ULP.

        Args:
            case: "range" (rank r sends r*n .. (r+1)*n - 1) or "random"
            shape: Tensor shape
            async_op: If True, launch the collective asynchronously
        """
        numel = math.prod(shape)
        own, expected = _integer_case(case, numel, self.comm_rank, self.comm_size)
        x = own.reshape(shape).to(DEVICE)

        work = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=async_op)
        _check_work_handle(self, work, async_op)

        _assert_reduction_close(
            x.cpu().flatten(),
            expected,
            self.comm_size,
            f"Rank {self.comm_rank}: allreduce {case} {shape}",
        )

    # --- Always correct: exact results (atol=0) ---

    def test_allreduce_all_zeros(self):
        """Allreduce of zeros stays exactly zero."""
        self._test_allreduce_exact_helper("zeros", shape=(128,))

    def test_allreduce_all_zeros_2d(self):
        """Allreduce of 2D zeros stays exactly zero."""
        self._test_allreduce_exact_helper("zeros", shape=(4, 64))

    def test_allreduce_all_ones(self):
        """Allreduce of ones is exactly the world size."""
        self._test_allreduce_exact_helper("ones", shape=(128,))

    def test_allreduce_all_ones_2d(self):
        """Allreduce of 2D ones is exactly the world size."""
        self._test_allreduce_exact_helper("ones", shape=(4, 64))

    def test_allreduce_all_ones_3d(self):
        """Allreduce of 3D ones is exactly the world size."""
        self._test_allreduce_exact_helper("ones", shape=(2, 2, 64))

    def test_allreduce_all_ones_4d(self):
        """Allreduce of 4D ones is exactly the world size."""
        self._test_allreduce_exact_helper("ones", shape=(2, 1, 2, 64))

    def test_allreduce_all_ones_async(self):
        """Allreduce of ones is exact with async_op=True."""
        self._test_allreduce_exact_helper("ones", shape=(128,), async_op=True)

    def test_allreduce_uniform_constant(self):
        """Every rank sends 4.0, so the sum is exactly 4.0 * world size."""
        self._test_allreduce_exact_helper("uniform_constant", shape=(128,))

    def test_allreduce_powers_of_two(self):
        """Rank r sends 2^(r % 4); the sum is exact in any order."""
        self._test_allreduce_exact_helper("powers_of_two", shape=(128,))

    def test_allreduce_powers_of_two_async(self):
        """Rank r sends 2^(r % 4); exact with async_op=True."""
        self._test_allreduce_exact_helper("powers_of_two", (128,), async_op=True)

    def test_allreduce_sign_cancellation(self):
        """Even ranks send +2.0, odd ranks -2.0: exact cancellation."""
        self._test_allreduce_exact_helper("sign_cancellation", shape=(128,))

    def test_allreduce_single_rank_impulse(self):
        """Only rank 0 contributes (values 1..1021, wrapping); others send zeros."""
        self._test_allreduce_exact_helper("single_rank_impulse", shape=(128,))

    def test_allreduce_single_rank_impulse_async(self):
        """Single-rank impulse is exact with async_op=True."""
        self._test_allreduce_exact_helper(
            "single_rank_impulse", shape=(128,), async_op=True
        )

    def test_allreduce_disjoint_shards(self):
        """Each rank owns one disjoint slice; a wrong chunk offset is a wrong value."""
        self._test_allreduce_exact_helper(
            "disjoint_shards", shape=(64 * self.comm_size,)
        )

    def test_allreduce_disjoint_shards_async(self):
        """Disjoint shards are exact with async_op=True."""
        self._test_allreduce_exact_helper(
            "disjoint_shards", shape=(64 * self.comm_size,), async_op=True
        )

    # --- Should be correct: DLFloat16 rounding (exact where possible, else
    # within accumulation depth * ULP) ---

    def test_allreduce_integer_range(self):
        """Rank r sends r*n .. (r+1)*n - 1."""
        self._test_allreduce_integer_helper("range", shape=(64,))

    def test_allreduce_integer_range_2d(self):
        """Integer range with a 2D tensor."""
        self._test_allreduce_integer_helper("range", shape=(2, 64))

    def test_allreduce_integer_range_async(self):
        """Integer range with async_op=True."""
        self._test_allreduce_integer_helper("range", shape=(128,), async_op=True)

    def test_allreduce_chunk_one_stick_per_rank(self):
        """Exactly one 64-element stick per rank (reduce-scatter chunking)."""
        self._test_allreduce_integer_helper(
            "random", shape=(_chunk_numel("one_stick_per_rank", self.comm_size),)
        )

    def test_allreduce_chunk_non_multiple(self):
        """A size that is not a multiple of the world size in sticks."""
        self._test_allreduce_integer_helper(
            "random", shape=(_chunk_numel("non_multiple", self.comm_size),)
        )

    def test_allreduce_chunk_one_stick_per_rank_async(self):
        """One stick per rank with async_op=True."""
        self._test_allreduce_integer_helper(
            "random",
            shape=(_chunk_numel("one_stick_per_rank", self.comm_size),),
            async_op=True,
        )

    def test_allreduce_large_tensor_4096(self):
        """4096 elements."""
        self._test_allreduce_integer_helper("random", shape=(4096,))

    def test_allreduce_large_tensor_65536(self):
        """65536 elements."""
        self._test_allreduce_integer_helper("random", shape=(65536,))

    def test_allreduce_large_tensor_262144(self):
        """262144 elements (512 KB per rank): the largest size verified to compile."""
        self._test_allreduce_integer_helper("random", shape=(NUMEL_512KB,))

    def test_reduction_check_rejects_corruption(self):
        """The exact/tolerance check must not accept a corrupted result."""
        expected = torch.tensor([1.0, 1000.0, 2000.0, 4000.0])
        tol = _calculate_collective_tolerance(self.comm_size, 4000.0, min_floor=0.0)
        _assert_reduction_close(expected.clone(), expected, self.comm_size, "clean")

        in_exact_region = expected.clone()
        in_exact_region[1] += 1.0
        with self.assertRaises(pytest.fail.Exception):
            _assert_reduction_close(in_exact_region, expected, self.comm_size, "exact")

        beyond_tolerance = expected.clone()
        beyond_tolerance[3] += 2.0 * tol
        with self.assertRaises(pytest.fail.Exception):
            _assert_reduction_close(beyond_tolerance, expected, self.comm_size, "tol")

        within_tolerance = expected.clone()
        within_tolerance[3] += tol / 2.0
        _assert_reduction_close(within_tolerance, expected, self.comm_size, "in tol")

    @parametrize("shape", [(128,), (4, 64), (2, 2, 64), (2, 1, 2, 64), (4, 96)])
    @parametrize("async_op", [False, True])
    def test_allreduce_bfloat16(self, shape, async_op):
        x = _bfloat16_rank_input(self.comm_rank, shape).to(DEVICE)
        work = dist.all_reduce(x, async_op=async_op)
        if async_op:
            self.assertIsNotNone(work)
            work.wait()
        else:
            self.assertIsNone(work)
        self.assertEqual(
            x.cpu(),
            _bfloat16_collective_expected("allreduce", shape, self.comm_size),
            rtol=0,
            atol=0,
        )

    @parametrize("async_op", [False, True])
    def test_allreduce_float16_padded(self, async_op):
        shape = (4, 96)
        x = _collective_rank_input(self.comm_rank, shape, torch.float16).to(DEVICE)
        work = dist.all_reduce(x, async_op=async_op)
        if async_op:
            work.wait()
        self.assertEqual(
            x.cpu(),
            _collective_expected("allreduce", shape, self.comm_size, torch.float16),
            rtol=0,
            atol=0,
        )

    def test_allreduce_bfloat16_rejects_non_dl16_storage(self):
        x = _native_bfloat16_tensor(DEVICE)
        with self.assertRaisesRegex(
            RuntimeError, "BF16 collectives require DL16 device storage"
        ):
            dist.all_reduce(x)

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("prefix", [False, True])
    def test_allreduce_rejects_unsupported_storage_view(self, dtype, prefix):
        expected = torch.ones((5, 96), dtype=dtype)
        base = expected.to(DEVICE)
        x = base[:-1] if prefix else base[1:]
        self.assertEqual(x.storage_offset() == 0, prefix)
        error = (
            "matching device storage extent" if prefix else "nonzero storage offsets"
        )
        with self.assertRaisesRegex(RuntimeError, error):
            dist.all_reduce(x)
        self.assertTrue(torch.equal(base.cpu(), expected))


if __name__ == "__main__":
    run_tests()
