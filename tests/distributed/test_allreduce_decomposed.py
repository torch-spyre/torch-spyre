# Copyright 2025 The Torch-Spyre Authors.
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

"""Tests for compile-time allreduce decomposition (reduce + broadcast).

Verifies that _c10d_functional.all_reduce is decomposed into reduce_async +
wait_work + broadcast_async at compile time, and produces correct results.

Usage:
    torchrun --nproc-per-node 2 tests/distributed/test_allreduce_decomposed.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_spyre  # noqa: F401

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
        "WORLD_SIZE environment variable is not a valid integer, "
        "skipping distributed tests",
        allow_module_level=True,
    )

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"
_GROUP_NAME = "default"


class AllReduceDecomposedModule(torch.nn.Module):
    """Module that performs allreduce using functional collective ops."""

    def __init__(self, group_name: str = _GROUP_NAME) -> None:
        super().__init__()
        self._group_name = group_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.ops._c10d_functional.all_reduce(x, "sum", self._group_name)
        return torch.ops._c10d_functional.wait_tensor(y)


class AllReduceWithComputeModule(torch.nn.Module):
    """Module with compute interleaved around allreduce."""

    def __init__(self, group_name: str = _GROUP_NAME) -> None:
        super().__init__()
        self._group_name = group_name

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_scaled = x * 2.0
        reduced = torch.ops._c10d_functional.all_reduce(
            x_scaled, "sum", self._group_name
        )
        z = y + 1.0
        result = torch.ops._c10d_functional.wait_tensor(reduced)
        return result + z


class TestAllReduceDecomposed(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.spyre._impl._lazy_init()

        if not dist.distributed_c10d.is_backend_available(C10D_BACKEND):
            raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
        if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
            raise RuntimeError(
                f"Error: Missing a C10 Backend for 'spyre'! Expected {C10D_BACKEND}"
            )

        if not dist.is_initialized():
            dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

        c10d._register_process_group(_GROUP_NAME, dist.group.WORLD)

        cls.comm_size = dist.get_world_size()
        cls.comm_rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        torch.compiler.reset()

    def test_allreduce_decomposed_fp16(self):
        """Verify decomposed allreduce produces correct sum across ranks."""
        x = torch.ones((128,), dtype=torch.float16, device=DEVICE)
        module = AllReduceDecomposedModule()
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.dtype, torch.float16)
        expected = torch.full((128,), float(self.comm_size), dtype=torch.float16)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: allreduce result incorrect. "
            f"Expected {expected[0].item()}, got {result[0].to('cpu').item()}",
        )

    def test_allreduce_decomposed_fp16_larger(self):
        """Verify decomposed allreduce works with larger fp16 tensors."""
        x = torch.ones((256,), dtype=torch.float16, device=DEVICE)
        module = AllReduceDecomposedModule()
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.shape, x.shape)
        expected = torch.full((256,), float(self.comm_size), dtype=torch.float16)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: allreduce fp16 result incorrect",
        )

    def test_allreduce_decomposed_with_interleaved_compute(self):
        """Verify compute around decomposed allreduce compiles correctly."""
        x = torch.ones((128,), dtype=torch.float16, device=DEVICE)
        y = torch.ones((128,), dtype=torch.float16, device=DEVICE)
        module = AllReduceWithComputeModule()
        compiled_module = torch.compile(module)
        result = compiled_module(x, y)

        # x_scaled = 2.0, allreduce(sum) = 2.0 * world_size, z = 2.0
        # result = (2.0 * world_size) + 2.0
        expected_val = 2.0 * self.comm_size + 2.0
        expected = torch.full((128,), expected_val, dtype=torch.float16)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: allreduce with compute incorrect. "
            f"Expected {expected_val}, got {result[0].to('cpu').item()}",
        )

    def test_allreduce_decomposed_rank_scaled(self):
        """Verify allreduce with rank-dependent input values."""
        val = float(self.comm_rank + 1)
        x = torch.full((64,), val, dtype=torch.float16, device=DEVICE)
        module = AllReduceDecomposedModule()
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        # Sum of 1 + 2 + ... + world_size = world_size * (world_size + 1) / 2
        expected_val = self.comm_size * (self.comm_size + 1) / 2.0
        expected = torch.full((64,), expected_val, dtype=torch.float16)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: rank-scaled allreduce incorrect. "
            f"Expected {expected_val}, got {result[0].to('cpu').item()}",
        )


if __name__ == "__main__":
    run_tests()
