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

"""Tests for compiled all_gather_into_tensor_coalesced.

Verifies that _c10d_functional.all_gather_into_tensor_coalesced is decomposed
into individual allgather_plan (compile-time) + allgather_run (runtime) calls
and produces correct results for each tensor in the list.

Usage:
    torchrun --nproc-per-node 2 tests/distributed/test_all_gather_coalesced_compiled.py
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


class AllGatherCoalescedModule(torch.nn.Module):
    """Module that performs coalesced allgather on a list of tensors."""

    def __init__(self, group_size: int, group_name: str = _GROUP_NAME) -> None:
        super().__init__()
        self._group_size = group_size
        self._group_name = group_name

    def forward(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        results = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
            tensors, self._group_size, self._group_name
        )
        return [torch.ops._c10d_functional.wait_tensor(r) for r in results]


class TestAllGatherCoalescedCompiled(TestCase):
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

    def test_all_gather_coalesced_compiled_basic(self):
        """Two same-shape tensors gathered correctly."""
        a = torch.full((64,), float(self.comm_rank), dtype=torch.float16, device=DEVICE)
        b = torch.full(
            (64,), float(self.comm_rank + 10), dtype=torch.float16, device=DEVICE
        )

        module = AllGatherCoalescedModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        r_a, r_b = compiled_module([a, b])

        self.assertEqual(r_a.shape[0], 64 * self.comm_size)
        self.assertEqual(r_b.shape[0], 64 * self.comm_size)

        r_a_cpu = r_a.to("cpu")
        r_b_cpu = r_b.to("cpu")
        for rank in range(self.comm_size):
            chunk_a = r_a_cpu[rank * 64 : (rank + 1) * 64]
            expected_a = torch.full((64,), float(rank), dtype=torch.float16)
            self.assertTrue(
                torch.allclose(chunk_a, expected_a),
                f"Rank {self.comm_rank}: tensor a chunk for rank {rank} incorrect",
            )

            chunk_b = r_b_cpu[rank * 64 : (rank + 1) * 64]
            expected_b = torch.full((64,), float(rank + 10), dtype=torch.float16)
            self.assertTrue(
                torch.allclose(chunk_b, expected_b),
                f"Rank {self.comm_rank}: tensor b chunk for rank {rank} incorrect",
            )

    def test_all_gather_coalesced_compiled_mixed_shapes(self):
        """Tensors of different shapes in one coalesced call."""
        a = torch.full((64,), float(self.comm_rank), dtype=torch.float16, device=DEVICE)
        b = torch.full(
            (128,), float(self.comm_rank), dtype=torch.float16, device=DEVICE
        )

        module = AllGatherCoalescedModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        r_a, r_b = compiled_module([a, b])

        self.assertEqual(r_a.shape[0], 64 * self.comm_size)
        self.assertEqual(r_b.shape[0], 128 * self.comm_size)

        r_a_cpu = r_a.to("cpu")
        r_b_cpu = r_b.to("cpu")
        for rank in range(self.comm_size):
            chunk_a = r_a_cpu[rank * 64 : (rank + 1) * 64]
            chunk_b = r_b_cpu[rank * 128 : (rank + 1) * 128]
            expected = torch.full((1,), float(rank), dtype=torch.float16)
            self.assertTrue(
                torch.allclose(chunk_a, expected.expand(64)),
                f"Rank {self.comm_rank}: mixed-shapes tensor a, "
                f"rank {rank} chunk incorrect",
            )
            self.assertTrue(
                torch.allclose(chunk_b, expected.expand(128)),
                f"Rank {self.comm_rank}: mixed-shapes tensor b, "
                f"rank {rank} chunk incorrect",
            )

    def test_all_gather_coalesced_compiled_fp16(self):
        """Verify dtype and output shape preservation for fp16 inputs."""
        a = torch.ones((64,), dtype=torch.float16, device=DEVICE)
        b = torch.ones((128,), dtype=torch.float16, device=DEVICE)

        module = AllGatherCoalescedModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        r_a, r_b = compiled_module([a, b])

        self.assertEqual(r_a.dtype, torch.float16)
        self.assertEqual(r_b.dtype, torch.float16)
        self.assertEqual(r_a.shape[0], 64 * self.comm_size)
        self.assertEqual(r_b.shape[0], 128 * self.comm_size)

    def test_all_gather_coalesced_compiled_with_compute(self):
        """Compute interleaved around coalesced allgather."""

        class CoalescedWithComputeModule(torch.nn.Module):
            def __init__(self, group_size: int, group_name: str = _GROUP_NAME):
                super().__init__()
                self._group_size = group_size
                self._group_name = group_name

            def forward(self, x, y):
                x_scaled = x * 2.0
                results = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
                    [x_scaled], self._group_size, self._group_name
                )
                z = y + 1.0
                gathered = torch.ops._c10d_functional.wait_tensor(results[0])
                return gathered + z

        x = torch.full(
            (64,), float(self.comm_rank + 1), dtype=torch.float16, device=DEVICE
        )
        y = torch.ones((64 * self.comm_size,), dtype=torch.float16, device=DEVICE)

        module = CoalescedWithComputeModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x, y)

        result_cpu = result.to("cpu")
        for rank in range(self.comm_size):
            chunk = result_cpu[rank * 64 : (rank + 1) * 64]
            # x_scaled = 2*(rank+1), gathered, then + z where z = 2.0
            expected_val = 2.0 * (rank + 1) + 2.0
            expected = torch.full((64,), expected_val, dtype=torch.float16)
            self.assertTrue(
                torch.allclose(chunk, expected),
                f"Rank {self.comm_rank}: coalesced allgather with compute, "
                f"rank {rank} chunk incorrect. "
                f"Expected {expected_val}, got {chunk[0].item()}",
            )

    def test_all_gather_coalesced_compiled_repeated_execution(self):
        """WSI reuse: compiled coalesced allgather executed multiple times."""
        module = AllGatherCoalescedModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)

        for iteration in range(4):
            a = torch.full(
                (64,), float(self.comm_rank), dtype=torch.float16, device=DEVICE
            )
            b = torch.full(
                (64,),
                float(self.comm_rank + 10),
                dtype=torch.float16,
                device=DEVICE,
            )
            r_a, r_b = compiled_module([a, b])

            self.assertEqual(r_a.shape[0], 64 * self.comm_size)
            r_a_cpu = r_a.to("cpu")
            r_b_cpu = r_b.to("cpu")
            for rank in range(self.comm_size):
                chunk_a = r_a_cpu[rank * 64 : (rank + 1) * 64]
                expected_a = torch.full((64,), float(rank), dtype=torch.float16)
                self.assertTrue(
                    torch.allclose(chunk_a, expected_a),
                    f"Rank {self.comm_rank}: iteration {iteration}, "
                    f"tensor a chunk for rank {rank} incorrect",
                )

                chunk_b = r_b_cpu[rank * 64 : (rank + 1) * 64]
                expected_b = torch.full((64,), float(rank + 10), dtype=torch.float16)
                self.assertTrue(
                    torch.allclose(chunk_b, expected_b),
                    f"Rank {self.comm_rank}: iteration {iteration}, "
                    f"tensor b chunk for rank {rank} incorrect",
                )


if __name__ == "__main__":
    run_tests()
