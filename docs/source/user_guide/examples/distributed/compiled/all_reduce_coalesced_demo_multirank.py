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

"""Compiled all_reduce_coalesced demo.

This example demonstrates how torch.compile lowers all_reduce_coalesced
into per-tensor allreduce plan/run pairs via Spyre's compile-time
collective ops.  The coalesced call takes a list of tensors and reduces
each one independently:

At graph load (compile-time, one plan per distinct shape):
    plan_handle_0 = allreduce_plan(dtype, "sum", group)
    plan_handle_1 = allreduce_plan(dtype, "sum", group)  # deduped if same shape

At every forward pass (runtime):
    y0 = allreduce_run(x0, plan_handle_0)
    y1 = allreduce_run(x1, plan_handle_1)
    y0 = wait_work(y0)
    y1 = wait_work(y1)

Usage:
    torchrun --nproc-per-node 2 all_reduce_coalesced_demo_multirank.py
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
    """Basic coalesced allreduce — two tensors reduced in one call."""
    print(f"[Rank {comm_rank}] Demo 1: Basic coalesced allreduce")

    class CoalescedModule(torch.nn.Module):
        def forward(self, tensors):
            results = torch.ops._c10d_functional.all_reduce_coalesced(
                tensors, "sum", _GROUP_NAME
            )
            return [torch.ops._c10d_functional.wait_tensor(r) for r in results]

    a = torch.ones((128,), dtype=torch.float16, device=DEVICE)
    b = torch.full((128,), 2.0, dtype=torch.float16, device=DEVICE)
    print(f"[Rank {comm_rank}] Input a: all 1.0, b: all 2.0")

    compiled_fn = torch.compile(CoalescedModule())
    r_a, r_b = compiled_fn([a, b])

    r_a_cpu = r_a.to("cpu")
    r_b_cpu = r_b.to("cpu")
    expected_a = float(comm_size)
    expected_b = 2.0 * comm_size
    print(
        f"[Rank {comm_rank}] Result a[0]: {r_a_cpu[0].item()}, expected: {expected_a}"
    )
    print(
        f"[Rank {comm_rank}] Result b[0]: {r_b_cpu[0].item()}, expected: {expected_b}"
    )

    assert torch.allclose(
        r_a_cpu, torch.full((128,), expected_a, dtype=torch.float16)
    ), f"Rank {comm_rank}: tensor a FAILED"
    assert torch.allclose(
        r_b_cpu, torch.full((128,), expected_b, dtype=torch.float16)
    ), f"Rank {comm_rank}: tensor b FAILED"
    print(f"[Rank {comm_rank}] PASSED")


def demo_mixed_shapes(comm_rank, comm_size):
    """Coalesced allreduce with tensors of different shapes.

    Each tensor gets its own WSI plan. The plan cache deduplicates by
    (dtype, num_elems), so tensors with the same shape share a plan.
    """
    print(f"[Rank {comm_rank}] Demo 2: Mixed-shape coalesced allreduce")

    class MixedShapeModule(torch.nn.Module):
        def forward(self, tensors):
            results = torch.ops._c10d_functional.all_reduce_coalesced(
                tensors, "sum", _GROUP_NAME
            )
            return [torch.ops._c10d_functional.wait_tensor(r) for r in results]

    val = float(comm_rank + 1)
    a = torch.full((64,), val, dtype=torch.float16, device=DEVICE)
    b = torch.full((256,), val, dtype=torch.float16, device=DEVICE)
    c = torch.full((128,), val, dtype=torch.float16, device=DEVICE)
    print(f"[Rank {comm_rank}] Input: three tensors (64, 256, 128), value={val}")

    compiled_fn = torch.compile(MixedShapeModule())
    r_a, r_b, r_c = compiled_fn([a, b, c])

    # sum of (rank+1) for all ranks = ws*(ws+1)/2
    expected_val = comm_size * (comm_size + 1) / 2.0
    print(f"[Rank {comm_rank}] Result a[0]: {r_a.to('cpu')[0].item()}")
    print(f"[Rank {comm_rank}] Result b[0]: {r_b.to('cpu')[0].item()}")
    print(f"[Rank {comm_rank}] Result c[0]: {r_c.to('cpu')[0].item()}")
    print(f"[Rank {comm_rank}] Expected all: {expected_val}")

    for name, result, size in [("a", r_a, 64), ("b", r_b, 256), ("c", r_c, 128)]:
        assert torch.allclose(
            result.to("cpu"),
            torch.full((size,), expected_val, dtype=torch.float16),
        ), f"Rank {comm_rank}: tensor {name} FAILED"
    print(f"[Rank {comm_rank}] PASSED")


def demo_coalesced_with_compute(comm_rank, comm_size):
    """Compute interleaved with coalesced allreduce.

    Demonstrates that independent compute can be placed around the
    coalesced collective, just as with individual allreduce calls.
    """
    print(f"[Rank {comm_rank}] Demo 3: Coalesced allreduce with compute")

    class ComputeModule(torch.nn.Module):
        def forward(self, x, y):
            x_scaled = x * 2.0
            y_shifted = y + 1.0
            results = torch.ops._c10d_functional.all_reduce_coalesced(
                [x_scaled, y_shifted], "sum", _GROUP_NAME
            )
            r_x = torch.ops._c10d_functional.wait_tensor(results[0])
            r_y = torch.ops._c10d_functional.wait_tensor(results[1])
            return r_x + r_y

    x = torch.ones((128,), dtype=torch.float16, device=DEVICE)
    y = torch.ones((128,), dtype=torch.float16, device=DEVICE)
    print(f"[Rank {comm_rank}] Input x: 1.0, y: 1.0")

    compiled_fn = torch.compile(ComputeModule())
    result = compiled_fn(x, y)

    result_cpu = result.to("cpu")
    # x_scaled = 2.0, allreduce -> 2.0 * ws
    # y_shifted = 2.0, allreduce -> 2.0 * ws
    # result = 2*ws + 2*ws = 4*ws
    expected_val = 4.0 * comm_size
    print(f"[Rank {comm_rank}] Result[0]: {result_cpu[0].item()}")
    print(f"[Rank {comm_rank}] Expected: {expected_val}")

    assert torch.allclose(
        result_cpu, torch.full((128,), expected_val, dtype=torch.float16)
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

    demo_basic_coalesced(comm_rank, comm_size)
    demo_mixed_shapes(comm_rank, comm_size)
    demo_coalesced_with_compute(comm_rank, comm_size)

    print(f"\n[Rank {comm_rank}] All demos PASSED!")
    dist.destroy_process_group()
