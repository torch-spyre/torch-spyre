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

"""Compiled allreduce demo.

This example demonstrates how torch.compile lowers allreduce into a
two-phase plan/run pattern via Spyre's compile-time collective ops:

At graph load (compile-time):
    plan_handle = allreduce_plan(dtype, "sum", group)

At every forward pass (runtime):
    y = allreduce_run(x, plan_handle)
    y = wait_work(y)

The plan creates the WorkScheduleInfo (communication plan) once, and the
run op reuses it on every invocation — avoiding repeated planning overhead.

Usage:
    torchrun --nproc-per-node 2 all_reduce_compiled.py
"""

import os

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

import torch_spyre  # noqa: F401

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"
_GROUP_NAME = "default"


def demo_basic_allreduce(comm_rank, comm_size):
    """Basic compiled allreduce — each rank contributes ones, result = world_size."""
    print(f"[Rank {comm_rank}] Demo 1: Basic compiled allreduce")

    class AllReduceModule(torch.nn.Module):
        def forward(self, x):
            y = torch.ops._c10d_functional.all_reduce(x, "sum", _GROUP_NAME)
            return torch.ops._c10d_functional.wait_tensor(y)

    x = torch.ones((128,), dtype=torch.float16, device=DEVICE)
    print(f"[Rank {comm_rank}] Input: all ones, shape={x.shape}")

    compiled_fn = torch.compile(AllReduceModule())
    result = compiled_fn(x)

    result_cpu = result.to("cpu")
    expected = float(comm_size)
    print(f"[Rank {comm_rank}] Result: {result_cpu[:5]}...")
    print(f"[Rank {comm_rank}] Expected: {expected} (world_size={comm_size})")

    assert torch.allclose(
        result_cpu, torch.full((128,), expected, dtype=torch.float16)
    ), f"Rank {comm_rank}: FAILED"
    print(f"[Rank {comm_rank}] PASSED")


def demo_overlap_opportunity(comm_rank, comm_size):
    """Compute interleaved with allreduce — demonstrates overlap potential.

    The compiler schedules allreduce_run as early as its dependencies allow,
    maximizing the overlap window between communication and independent compute.
    """
    print(f"[Rank {comm_rank}] Demo 2: Communication-compute overlap opportunity")

    class OverlapModule(torch.nn.Module):
        def forward(self, x, y):
            reduced = torch.ops._c10d_functional.all_reduce(x, "sum", _GROUP_NAME)

            # Independent compute — can potentially overlap with communication
            z = y * 3.0 + 1.0

            # Wait for allreduce to complete
            reduced_result = torch.ops._c10d_functional.wait_tensor(reduced)

            # Combine communication result with local compute
            return reduced_result + z

    x = torch.ones((256,), dtype=torch.float16, device=DEVICE) * (comm_rank + 1)
    y = torch.ones((256,), dtype=torch.float16, device=DEVICE)

    print(f"[Rank {comm_rank}] Input x: {comm_rank + 1}.0, y: 1.0")

    compiled_fn = torch.compile(OverlapModule())
    result = compiled_fn(x, y)

    result_cpu = result.to("cpu")
    # allreduce(x) = sum(1, 2, ..., world_size) = world_size*(world_size+1)/2
    # z = 3.0 + 1.0 = 4.0
    # result = allreduce_sum + z
    allreduce_sum = comm_size * (comm_size + 1) / 2.0
    expected_val = allreduce_sum + 4.0
    print(f"[Rank {comm_rank}] Result[0]: {result_cpu[0].item()}")
    print(f"[Rank {comm_rank}] Expected: {expected_val}")
    print(f"[Rank {comm_rank}]   (allreduce_sum={allreduce_sum} + z=4.0)")

    assert torch.allclose(
        result_cpu, torch.full((256,), expected_val, dtype=torch.float16)
    ), f"Rank {comm_rank}: FAILED"
    print(f"[Rank {comm_rank}] PASSED")


def demo_multiple_allreduce(comm_rank, comm_size):
    """Multiple allreduce operations in one compiled graph."""
    print(f"[Rank {comm_rank}] Demo 3: Multiple compiled allreduce operations")

    class MultiAllReduceModule(torch.nn.Module):
        def forward(self, a, b):
            ar_a = torch.ops._c10d_functional.all_reduce(a, "sum", _GROUP_NAME)
            ar_b = torch.ops._c10d_functional.all_reduce(b, "sum", _GROUP_NAME)
            result_a = torch.ops._c10d_functional.wait_tensor(ar_a)
            result_b = torch.ops._c10d_functional.wait_tensor(ar_b)
            return result_a * result_b

    a = torch.full((64,), 2.0, dtype=torch.float16, device=DEVICE)
    b = torch.full((64,), 3.0, dtype=torch.float16, device=DEVICE)

    print(f"[Rank {comm_rank}] Input a: 2.0, b: 3.0")

    compiled_fn = torch.compile(MultiAllReduceModule())
    result = compiled_fn(a, b)

    result_cpu = result.to("cpu")
    # allreduce(a) = 2.0 * world_size, allreduce(b) = 3.0 * world_size
    # result = (2*ws) * (3*ws) = 6 * ws^2
    expected_val = 6.0 * comm_size * comm_size
    print(f"[Rank {comm_rank}] Result[0]: {result_cpu[0].item()}")
    print(f"[Rank {comm_rank}] Expected: {expected_val}")

    assert torch.allclose(
        result_cpu, torch.full((64,), expected_val, dtype=torch.float16)
    ), f"Rank {comm_rank}: FAILED"
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

    demo_basic_allreduce(comm_rank, comm_size)
    demo_overlap_opportunity(comm_rank, comm_size)
    demo_multiple_allreduce(comm_rank, comm_size)

    print(f"\n[Rank {comm_rank}] All demos PASSED!")
    dist.destroy_process_group()
