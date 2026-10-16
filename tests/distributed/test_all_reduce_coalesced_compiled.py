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

"""Tests for compiled all_reduce_coalesced (decomposed into per-tensor allreduce).

Verifies that _c10d_functional.all_reduce_coalesced is decomposed into
individual allreduce_plan (compile-time) + allreduce_run (runtime) calls
and produces correct results for each tensor in the list.

Usage:
    torchrun --nproc-per-node 2 tests/distributed/test_all_reduce_coalesced_compiled.py
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


class AllReduceCoalescedModule(torch.nn.Module):
    """Module that performs coalesced allreduce on a list of tensors."""

    def __init__(self, group_name: str = _GROUP_NAME) -> None:
        super().__init__()
        self._group_name = group_name

    def forward(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        results = torch.ops._c10d_functional.all_reduce_coalesced(
            tensors, "sum", self._group_name
        )
        return [torch.ops._c10d_functional.wait_tensor(r) for r in results]


class TestAllReduceCoalescedCompiled(TestCase):
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

    def test_all_reduce_coalesced_compiled_basic(self):
        """Two same-shape tensors reduced correctly."""
        a = torch.ones((128,), dtype=torch.float16, device=DEVICE)
        b = torch.full((128,), 2.0, dtype=torch.float16, device=DEVICE)

        module = AllReduceCoalescedModule()
        compiled_module = torch.compile(module)
        r_a, r_b = compiled_module([a, b])

        expected_a = torch.full((128,), float(self.comm_size), dtype=torch.float16)
        expected_b = torch.full((128,), 2.0 * self.comm_size, dtype=torch.float16)
        self.assertTrue(
            torch.allclose(r_a.to("cpu"), expected_a),
            f"Rank {self.comm_rank}: coalesced tensor 0 incorrect. "
            f"Expected {expected_a[0].item()}, got {r_a[0].to('cpu').item()}",
        )
        self.assertTrue(
            torch.allclose(r_b.to("cpu"), expected_b),
            f"Rank {self.comm_rank}: coalesced tensor 1 incorrect. "
            f"Expected {expected_b[0].item()}, got {r_b[0].to('cpu').item()}",
        )

    def test_all_reduce_coalesced_compiled_mixed_shapes(self):
        """Tensors of different shapes in one coalesced call."""
        a = torch.full(
            (64,), float(self.comm_rank + 1), dtype=torch.float16, device=DEVICE
        )
        b = torch.full(
            (256,), float(self.comm_rank + 1), dtype=torch.float16, device=DEVICE
        )

        module = AllReduceCoalescedModule()
        compiled_module = torch.compile(module)
        r_a, r_b = compiled_module([a, b])

        # sum of (rank+1) for all ranks = ws*(ws+1)/2
        expected_val = self.comm_size * (self.comm_size + 1) / 2.0
        expected_a = torch.full((64,), expected_val, dtype=torch.float16)
        expected_b = torch.full((256,), expected_val, dtype=torch.float16)
        self.assertTrue(
            torch.allclose(r_a.to("cpu"), expected_a),
            f"Rank {self.comm_rank}: mixed-shapes tensor 0 (64,) incorrect. "
            f"Expected {expected_val}, got {r_a[0].to('cpu').item()}",
        )
        self.assertTrue(
            torch.allclose(r_b.to("cpu"), expected_b),
            f"Rank {self.comm_rank}: mixed-shapes tensor 1 (256,) incorrect. "
            f"Expected {expected_val}, got {r_b[0].to('cpu').item()}",
        )

    def test_all_reduce_coalesced_compiled_fp16(self):
        """Verify dtype preservation for fp16 inputs."""
        a = torch.ones((128,), dtype=torch.float16, device=DEVICE)
        b = torch.ones((128,), dtype=torch.float16, device=DEVICE)

        module = AllReduceCoalescedModule()
        compiled_module = torch.compile(module)
        r_a, r_b = compiled_module([a, b])

        self.assertEqual(r_a.dtype, torch.float16)
        self.assertEqual(r_b.dtype, torch.float16)

    def test_all_reduce_coalesced_compiled_with_compute(self):
        """Compute interleaved around coalesced allreduce."""

        class CoalescedWithComputeModule(torch.nn.Module):
            def __init__(self, group_name: str = _GROUP_NAME) -> None:
                super().__init__()
                self._group_name = group_name

            def forward(self, x, y):
                x_scaled = x * 2.0
                y_shifted = y + 1.0
                results = torch.ops._c10d_functional.all_reduce_coalesced(
                    [x_scaled, y_shifted], "sum", self._group_name
                )
                r_x = torch.ops._c10d_functional.wait_tensor(results[0])
                r_y = torch.ops._c10d_functional.wait_tensor(results[1])
                return r_x + r_y

        x = torch.ones((128,), dtype=torch.float16, device=DEVICE)
        y = torch.ones((128,), dtype=torch.float16, device=DEVICE)

        module = CoalescedWithComputeModule()
        compiled_module = torch.compile(module)
        result = compiled_module(x, y)

        # x_scaled = 2.0, allreduce -> 2.0 * ws
        # y_shifted = 2.0, allreduce -> 2.0 * ws
        # result = 2*ws + 2*ws = 4*ws
        expected_val = 4.0 * self.comm_size
        expected = torch.full((128,), expected_val, dtype=torch.float16)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: coalesced with compute incorrect. "
            f"Expected {expected_val}, got {result[0].to('cpu').item()}",
        )

    def test_all_reduce_coalesced_compiled_repeated_execution(self):
        """WSI reuse: compiled coalesced allreduce executed multiple times."""
        module = AllReduceCoalescedModule()
        compiled_module = torch.compile(module)

        expected_a = torch.full((128,), float(self.comm_size), dtype=torch.float16)
        expected_b = torch.full((128,), 3.0 * self.comm_size, dtype=torch.float16)

        for iteration in range(4):
            # Fresh tensors each iteration — allreduce mutates in-place.
            a = torch.ones((128,), dtype=torch.float16, device=DEVICE)
            b = torch.full((128,), 3.0, dtype=torch.float16, device=DEVICE)
            r_a, r_b = compiled_module([a, b])
            self.assertTrue(
                torch.allclose(r_a.to("cpu"), expected_a),
                f"Rank {self.comm_rank}: iteration {iteration}, "
                f"tensor 0 incorrect. Expected {expected_a[0].item()}, "
                f"got {r_a[0].to('cpu').item()}",
            )
            self.assertTrue(
                torch.allclose(r_b.to("cpu"), expected_b),
                f"Rank {self.comm_rank}: iteration {iteration}, "
                f"tensor 1 incorrect. Expected {expected_b[0].item()}, "
                f"got {r_b[0].to('cpu').item()}",
            )


if __name__ == "__main__":
    run_tests()
