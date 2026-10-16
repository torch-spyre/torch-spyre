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

"""Compiled reduce_scatter_tensor demo.

This example demonstrates how torch.compile lowers reduce_scatter_tensor
into reducescatter_plan/reducescatter_run pairs via Spyre's compile-time
collective ops.  reduce_scatter is the inverse of allgather: each rank
contributes its full input tensor, the tensors are reduced (summed),
and each rank receives a disjoint slice of the result.

At graph load (compile-time, one plan per distinct shape):
    plan_handle = reducescatter_plan(num_elems, dtype, group_size, reduce_op, group)

At every forward pass (runtime):
    y = reducescatter_run(x, plan_handle, group_size)
    y = wait_work(y)

Usage:
    torchrun --nproc-per-node 2 reduce_scatter_demo_multirank.py
"""

import os

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

import torch_spyre  # noqa: F401

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"
_GROUP_NAME = "default"


def demo_basic_reduce_scatter(comm_rank, comm_size):
    """Basic reduce_scatter — each rank contributes a full tensor."""
    print(f"[Rank {comm_rank}] Demo 1: Basic reduce_scatter")

    class ReduceScatterModule(torch.nn.Module):
        def __init__(self, group_size):
            super().__init__()
            self._group_size = group_size

        def forward(self, x):
            result = torch.ops._c10d_functional.reduce_scatter_tensor(
                x, "sum", self._group_size, _GROUP_NAME
            )
            return torch.ops._c10d_functional.wait_tensor(result)

    x = torch.full((128,), float(comm_rank + 1), dtype=torch.float16, device=DEVICE)
    print(f"[Rank {comm_rank}] Input: {comm_rank + 1}.0 (128 elems)")

    compiled_fn = torch.compile(ReduceScatterModule(comm_size))
    result = compiled_fn(x)

    result_cpu = result.to("cpu")
    print(
        f"[Rank {comm_rank}] Result shape: {result.shape} "
        f"(128 / {comm_size} = {128 // comm_size})"
    )

    # sum of (rank+1) for all ranks = ws*(ws+1)/2
    expected_val = comm_size * (comm_size + 1) / 2.0
    expected = torch.full((128 // comm_size,), expected_val, dtype=torch.float16)
    assert torch.allclose(result_cpu, expected), (
        f"Rank {comm_rank}: expected {expected_val}, got {result_cpu[0].item()}"
    )
    print(f"[Rank {comm_rank}] PASSED (value={result_cpu[0].item()})")


def demo_reduce_scatter_with_compute(comm_rank, comm_size):
    """Compute interleaved with reduce_scatter."""
    print(f"[Rank {comm_rank}] Demo 2: Reduce_scatter with compute")

    class ComputeModule(torch.nn.Module):
        def __init__(self, group_size):
            super().__init__()
            self._group_size = group_size

        def forward(self, x, bias):
            x_scaled = x * 2.0
            result = torch.ops._c10d_functional.reduce_scatter_tensor(
                x_scaled, "sum", self._group_size, _GROUP_NAME
            )
            scattered = torch.ops._c10d_functional.wait_tensor(result)
            return scattered + bias

    x = torch.full((128,), float(comm_rank + 1), dtype=torch.float16, device=DEVICE)
    bias = torch.ones((128 // comm_size,), dtype=torch.float16, device=DEVICE)
    print(f"[Rank {comm_rank}] Input x: {comm_rank + 1}.0, bias: 1.0, scale: 2.0")

    compiled_fn = torch.compile(ComputeModule(comm_size))
    result = compiled_fn(x, bias)

    result_cpu = result.to("cpu")
    # x_scaled = 2*(rank+1), reduce_scatter sum, + bias=1
    rs_val = 2.0 * comm_size * (comm_size + 1) / 2.0
    expected_val = rs_val + 1.0
    expected = torch.full((128 // comm_size,), expected_val, dtype=torch.float16)
    print(
        f"[Rank {comm_rank}] Result: {result_cpu[0].item()}, expected: {expected_val}"
    )
    assert torch.allclose(result_cpu, expected), f"Rank {comm_rank}: FAILED"
    print(f"[Rank {comm_rank}] PASSED")


def demo_coalesced_reduce_scatter(comm_rank, comm_size):
    """Coalesced reduce_scatter — multiple tensors in one call."""
    print(f"[Rank {comm_rank}] Demo 3: Coalesced reduce_scatter")

    class CoalescedModule(torch.nn.Module):
        def __init__(self, group_size):
            super().__init__()
            self._group_size = group_size

        def forward(self, tensors):
            results = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
                tensors, "sum", self._group_size, _GROUP_NAME
            )
            return [torch.ops._c10d_functional.wait_tensor(r) for r in results]

    a = torch.full((128,), float(comm_rank + 1), dtype=torch.float16, device=DEVICE)
    b = torch.full((256,), float(comm_rank + 1), dtype=torch.float16, device=DEVICE)
    print(f"[Rank {comm_rank}] Input: two tensors (128, 256), value={comm_rank + 1}.0")

    compiled_fn = torch.compile(CoalescedModule(comm_size))
    r_a, r_b = compiled_fn([a, b])

    print(f"[Rank {comm_rank}] Result a shape: {r_a.shape}, b shape: {r_b.shape}")

    expected_val = comm_size * (comm_size + 1) / 2.0
    expected_a = torch.full((128 // comm_size,), expected_val, dtype=torch.float16)
    expected_b = torch.full((256 // comm_size,), expected_val, dtype=torch.float16)
    assert torch.allclose(r_a.to("cpu"), expected_a), (
        f"Rank {comm_rank}: tensor a FAILED"
    )
    assert torch.allclose(r_b.to("cpu"), expected_b), (
        f"Rank {comm_rank}: tensor b FAILED"
    )
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

    demo_basic_reduce_scatter(comm_rank, comm_size)
    demo_reduce_scatter_with_compute(comm_rank, comm_size)
    demo_coalesced_reduce_scatter(comm_rank, comm_size)

    print(f"\n[Rank {comm_rank}] All demos PASSED!")
    dist.destroy_process_group()
