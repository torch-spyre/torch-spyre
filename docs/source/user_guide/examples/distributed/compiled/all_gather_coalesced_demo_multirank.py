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

"""Compiled all_gather_into_tensor_coalesced demo.

This example demonstrates how torch.compile lowers
all_gather_into_tensor_coalesced into per-tensor allgather plan/run pairs
via Spyre's compile-time collective ops.  The coalesced call takes a list
of tensors and gathers each one independently:

At graph load (compile-time, one plan per distinct shape):
    plan_handle_0 = allgather_plan(num_elems, dtype, group_size, group)
    plan_handle_1 = allgather_plan(num_elems, dtype, group_size, group)

At every forward pass (runtime):
    y0 = allgather_run(x0, plan_handle_0, group_size)
    y1 = allgather_run(x1, plan_handle_1, group_size)
    y0 = wait_work(y0)
    y1 = wait_work(y1)

Usage:
    torchrun --nproc-per-node 2 all_gather_coalesced_demo_multirank.py
"""

import os

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

import torch_spyre  # noqa: F401

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"
_GROUP_NAME = "default"


def demo_basic_coalesced(comm_rank, comm_size):
    """Basic coalesced allgather — two tensors gathered in one call."""
    print(f"[Rank {comm_rank}] Demo 1: Basic coalesced allgather")

    class CoalescedModule(torch.nn.Module):
        def __init__(self, group_size):
            super().__init__()
            self._group_size = group_size

        def forward(self, tensors):
            results = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
                tensors, self._group_size, _GROUP_NAME
            )
            return [torch.ops._c10d_functional.wait_tensor(r) for r in results]

    a = torch.full((64,), float(comm_rank), dtype=torch.float16, device=DEVICE)
    b = torch.full((64,), float(comm_rank + 10), dtype=torch.float16, device=DEVICE)
    print(f"[Rank {comm_rank}] Input a: {comm_rank}.0, b: {comm_rank + 10}.0")

    compiled_fn = torch.compile(CoalescedModule(comm_size))
    r_a, r_b = compiled_fn([a, b])

    r_a_cpu = r_a.to("cpu")
    r_b_cpu = r_b.to("cpu")
    print(f"[Rank {comm_rank}] Result a shape: {r_a.shape} (64 * {comm_size})")
    print(f"[Rank {comm_rank}] Result b shape: {r_b.shape} (64 * {comm_size})")

    for rank in range(comm_size):
        chunk_a = r_a_cpu[rank * 64 : (rank + 1) * 64]
        chunk_b = r_b_cpu[rank * 64 : (rank + 1) * 64]
        expected_a = torch.full((64,), float(rank), dtype=torch.float16)
        expected_b = torch.full((64,), float(rank + 10), dtype=torch.float16)
        assert torch.allclose(chunk_a, expected_a), (
            f"Rank {comm_rank}: tensor a, rank {rank} chunk FAILED"
        )
        assert torch.allclose(chunk_b, expected_b), (
            f"Rank {comm_rank}: tensor b, rank {rank} chunk FAILED"
        )
    print(f"[Rank {comm_rank}] PASSED")


def demo_mixed_shapes(comm_rank, comm_size):
    """Coalesced allgather with tensors of different shapes.

    Each tensor gets its own WSI plan.  The plan cache deduplicates by
    (dtype, num_elems, group_size), so tensors with the same shape share
    a plan.
    """
    print(f"[Rank {comm_rank}] Demo 2: Mixed-shape coalesced allgather")

    class MixedShapeModule(torch.nn.Module):
        def __init__(self, group_size):
            super().__init__()
            self._group_size = group_size

        def forward(self, tensors):
            results = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
                tensors, self._group_size, _GROUP_NAME
            )
            return [torch.ops._c10d_functional.wait_tensor(r) for r in results]

    a = torch.full((64,), float(comm_rank), dtype=torch.float16, device=DEVICE)
    b = torch.full((128,), float(comm_rank), dtype=torch.float16, device=DEVICE)
    print(f"[Rank {comm_rank}] Input: two tensors (64, 128), value={comm_rank}.0")

    compiled_fn = torch.compile(MixedShapeModule(comm_size))
    r_a, r_b = compiled_fn([a, b])

    print(f"[Rank {comm_rank}] Result a shape: {r_a.shape}, b shape: {r_b.shape}")

    r_a_cpu = r_a.to("cpu")
    r_b_cpu = r_b.to("cpu")
    for rank in range(comm_size):
        chunk_a = r_a_cpu[rank * 64 : (rank + 1) * 64]
        chunk_b = r_b_cpu[rank * 128 : (rank + 1) * 128]
        expected = float(rank)
        assert torch.allclose(
            chunk_a, torch.full((64,), expected, dtype=torch.float16)
        ), f"Rank {comm_rank}: tensor a, rank {rank} FAILED"
        assert torch.allclose(
            chunk_b, torch.full((128,), expected, dtype=torch.float16)
        ), f"Rank {comm_rank}: tensor b, rank {rank} FAILED"
    print(f"[Rank {comm_rank}] PASSED")


def demo_coalesced_with_compute(comm_rank, comm_size):
    """Compute interleaved with coalesced allgather."""
    print(f"[Rank {comm_rank}] Demo 3: Coalesced allgather with compute")

    class ComputeModule(torch.nn.Module):
        def __init__(self, group_size):
            super().__init__()
            self._group_size = group_size

        def forward(self, x, y):
            x_scaled = x * 2.0
            results = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
                [x_scaled], self._group_size, _GROUP_NAME
            )
            z = y + 1.0
            gathered = torch.ops._c10d_functional.wait_tensor(results[0])
            return gathered + z

    x = torch.full((64,), float(comm_rank + 1), dtype=torch.float16, device=DEVICE)
    y = torch.ones((64 * comm_size,), dtype=torch.float16, device=DEVICE)
    print(f"[Rank {comm_rank}] Input x: {comm_rank + 1}.0, y: 1.0")

    compiled_fn = torch.compile(ComputeModule(comm_size))
    result = compiled_fn(x, y)

    result_cpu = result.to("cpu")
    for rank in range(comm_size):
        chunk = result_cpu[rank * 64 : (rank + 1) * 64]
        # x_scaled = 2*(rank+1), gathered, + z where z = 2.0
        expected_val = 2.0 * (rank + 1) + 2.0
        print(
            f"[Rank {comm_rank}] Chunk {rank}: "
            f"{chunk[0].item()}, expected: {expected_val}"
        )
        assert torch.allclose(
            chunk, torch.full((64,), expected_val, dtype=torch.float16)
        ), f"Rank {comm_rank}: rank {rank} chunk FAILED"
    print(f"[Rank {comm_rank}] PASSED")


if __name__ == "__main__":
    if not dist.distributed_c10d.is_backend_available(C10D_BACKEND):
        raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
    if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
        raise RuntimeError(
            f"Error: Missing a C10 Backend for 'spyre'! Expected {C10D_BACKEND}"
        )

    print("Initializing distributed process group...")
    dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

    comm_size = dist.get_world_size()
    comm_rank = dist.get_rank()

    c10d._register_process_group(_GROUP_NAME, dist.group.WORLD)

    print(f"[Rank {comm_rank}] World size: {comm_size}")
    print(f"[Rank {comm_rank}] Device: {DEVICE}")

    demo_basic_coalesced(comm_rank, comm_size)
    demo_mixed_shapes(comm_rank, comm_size)
    demo_coalesced_with_compute(comm_rank, comm_size)

    print(f"\n[Rank {comm_rank}] All demos PASSED!")
    dist.destroy_process_group()
