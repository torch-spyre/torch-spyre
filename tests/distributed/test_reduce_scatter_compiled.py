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

"""Tests for compiled reduce_scatter_tensor.

Verifies that _c10d_functional.reduce_scatter_tensor is lowered into
reducescatter_plan (compile-time) + reducescatter_run (runtime) calls
and produces correct results.

Usage:
    torchrun --nproc-per-node 2 tests/distributed/test_reduce_scatter_compiled.py
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


class ReduceScatterModule(torch.nn.Module):
    """Module that performs reduce_scatter_tensor on a single tensor."""

    def __init__(self, group_size: int, group_name: str = _GROUP_NAME) -> None:
        super().__init__()
        self._group_size = group_size
        self._group_name = group_name

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        result = torch.ops._c10d_functional.reduce_scatter_tensor(
            tensor, "sum", self._group_size, self._group_name
        )
        return torch.ops._c10d_functional.wait_tensor(result)


class TestReduceScatterCompiled(TestCase):
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

    def test_reduce_scatter_compiled_basic(self):
        """Each rank contributes a full tensor; output is reduced scatter."""
        # Input: each rank has a (128,) tensor filled with (rank + 1).
        # With 2 ranks, reduce_scatter splits input into 2 chunks of 64 and
        # sums corresponding chunks across ranks.
        # Rank 0 input: [1,1,...,1] (128 elems)
        # Rank 1 input: [2,2,...,2] (128 elems)
        # After reduce_scatter with sum:
        #   Rank 0 gets chunk 0: 1+2 = 3
        #   Rank 1 gets chunk 1: 1+2 = 3
        x = torch.full(
            (128,), float(self.comm_rank + 1), dtype=torch.float16, device=DEVICE
        )

        module = ReduceScatterModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.shape[0], 128 // self.comm_size)

        expected_val = self.comm_size * (self.comm_size + 1) / 2.0
        expected = torch.full(
            (128 // self.comm_size,), expected_val, dtype=torch.float16
        )
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: reduce_scatter basic incorrect. "
            f"Expected {expected_val}, got {result[0].to('cpu').item()}",
        )

    def test_reduce_scatter_compiled_fp16(self):
        """Verify dtype and output shape preservation for fp16 inputs."""
        x = torch.ones((128,), dtype=torch.float16, device=DEVICE)

        module = ReduceScatterModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.shape[0], 128 // self.comm_size)

    def test_reduce_scatter_compiled_with_compute(self):
        """Compute interleaved around reduce_scatter."""

        class ComputeModule(torch.nn.Module):
            def __init__(self, group_size: int, group_name: str = _GROUP_NAME):
                super().__init__()
                self._group_size = group_size
                self._group_name = group_name

            def forward(self, x, bias):
                x_scaled = x * 2.0
                result = torch.ops._c10d_functional.reduce_scatter_tensor(
                    x_scaled, "sum", self._group_size, self._group_name
                )
                scattered = torch.ops._c10d_functional.wait_tensor(result)
                return scattered + bias

        x = torch.full(
            (128,), float(self.comm_rank + 1), dtype=torch.float16, device=DEVICE
        )
        bias = torch.ones((128 // self.comm_size,), dtype=torch.float16, device=DEVICE)

        module = ComputeModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x, bias)

        # x_scaled = 2*(rank+1), reduce_scatter sum -> 2*sum(rank+1), + bias=1
        rs_val = 2.0 * self.comm_size * (self.comm_size + 1) / 2.0
        expected_val = rs_val + 1.0
        expected = torch.full(
            (128 // self.comm_size,), expected_val, dtype=torch.float16
        )
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: reduce_scatter with compute incorrect. "
            f"Expected {expected_val}, got {result[0].to('cpu').item()}",
        )

    def test_reduce_scatter_compiled_repeated_execution(self):
        """WSI reuse: compiled reduce_scatter executed multiple times."""
        module = ReduceScatterModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)

        for iteration in range(4):
            x = torch.full(
                (128,),
                float(self.comm_rank + 1),
                dtype=torch.float16,
                device=DEVICE,
            )
            result = compiled_module(x)

            self.assertEqual(result.shape[0], 128 // self.comm_size)

            expected_val = self.comm_size * (self.comm_size + 1) / 2.0
            expected = torch.full(
                (128 // self.comm_size,), expected_val, dtype=torch.float16
            )
            self.assertTrue(
                torch.allclose(result.to("cpu"), expected),
                f"Rank {self.comm_rank}: iteration {iteration}, "
                f"reduce_scatter incorrect. "
                f"Expected {expected_val}, got {result[0].to('cpu').item()}",
            )

    def test_reduce_scatter_coalesced_compiled(self):
        """Coalesced reduce_scatter — multiple tensors in one call."""

        class CoalescedModule(torch.nn.Module):
            def __init__(self, group_size: int, group_name: str = _GROUP_NAME):
                super().__init__()
                self._group_size = group_size
                self._group_name = group_name

            def forward(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
                results = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
                    tensors, "sum", self._group_size, self._group_name
                )
                return [torch.ops._c10d_functional.wait_tensor(r) for r in results]

        a = torch.full(
            (128,),
            float(self.comm_rank + 1),
            dtype=torch.float16,
            device=DEVICE,
        )
        b = torch.full(
            (256,),
            float(self.comm_rank + 1),
            dtype=torch.float16,
            device=DEVICE,
        )

        module = CoalescedModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        r_a, r_b = compiled_module([a, b])

        self.assertEqual(r_a.shape[0], 128 // self.comm_size)
        self.assertEqual(r_b.shape[0], 256 // self.comm_size)

        expected_val = self.comm_size * (self.comm_size + 1) / 2.0
        expected_a = torch.full(
            (128 // self.comm_size,), expected_val, dtype=torch.float16
        )
        expected_b = torch.full(
            (256 // self.comm_size,), expected_val, dtype=torch.float16
        )
        self.assertTrue(
            torch.allclose(r_a.to("cpu"), expected_a),
            f"Rank {self.comm_rank}: coalesced tensor 0 incorrect. "
            f"Expected {expected_val}, got {r_a[0].to('cpu').item()}",
        )
        self.assertTrue(
            torch.allclose(r_b.to("cpu"), expected_b),
            f"Rank {self.comm_rank}: coalesced tensor 1 incorrect. "
            f"Expected {expected_val}, got {r_b[0].to('cpu').item()}",
        )


if __name__ == "__main__":
    run_tests()
