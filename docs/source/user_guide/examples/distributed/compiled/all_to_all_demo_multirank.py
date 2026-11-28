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

# Run with:  torchrun --nproc-per-node 2 all_to_all_demo_multirank.py
#
# all_to_all_single cuts each rank's input along dim 0 into one slice per
# rank, sends slice j to rank j, and concatenates the slices it receives in
# source-rank order.  The splits can be uneven; the output's dim 0 is then
# sum(output_split_sizes), which can differ from the input's.
#
# Spyre limits: every slice must be a whole number of 128-byte sticks, and
# dim-0 rows must be contiguous on device.  Both hold for a 1-D tensor, or a
# 2-D tensor whose rows are at most one stick wide (64 fp16 / 32 fp32
# elements), which is what this demo uses.  Zero-size slices are not yet
# supported.

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

ROW = 64  # one fp16 stick per row


def run_demo():
    device = torch.device("spyre")

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}/{world_size} using device {device}")

    c10d._register_process_group("default", dist.group.WORLD)

    # Uneven pattern: rank s sends s + 1 rows to every rank (itself included).
    # With 2 ranks: rank 0 sends [1, 1] (2 rows in), rank 1 sends [2, 2]
    # (4 rows in), and both receive [1, 2] (3 rows out), so dim 0 changes.
    def rows_sent(src, dst):
        return src + 1

    input_splits = [rows_sent(rank, d) for d in range(world_size)]
    output_splits = [rows_sent(s, rank) for s in range(world_size)]

    # Every row is filled with 10 * (sender + 1) + destination, so each row of
    # the output shows who sent it and that it was meant for this rank.
    rows = [
        torch.full((ROW,), 10.0 * (rank + 1) + d)
        for d, n in enumerate(input_splits)
        for _ in range(n)
    ]
    x = torch.stack(rows).to(torch.float16).to(device)
    print(f"Rank {rank} - Input rows (first element): {x[:, 0]}")
    print(f"Rank {rank} - input_splits={input_splits} output_splits={output_splits}")

    def fn(t):
        # Pre-exchange computation
        y = t * 2

        # Uneven all_to_all - lowered to alltoall_plan + alltoall_run
        exchanged = torch.ops._c10d_functional.all_to_all_single(
            y, output_splits, input_splits, "default"
        )

        # Wait for the exchange to complete, then post-exchange computation
        received = torch.ops._c10d_functional.wait_tensor(exchanged)
        return received + 1

    print(f"Rank {rank} - Compiling function...")
    compiled_fn = torch.compile(fn)

    print(f"Rank {rank} - Executing all_to_all")
    out = compiled_fn(x)

    expected = [
        2 * (10.0 * (s + 1) + rank) + 1
        for s, n in enumerate(output_splits)
        for _ in range(n)
    ]
    print("\n")
    print(f"Rank {rank} - After all_to_all (first element of each row): {out[:, 0]}")
    print(f"Rank {rank} - Expected: {expected} = 2 * (10 * (sender + 1) + {rank}) + 1")
    print(f"\n[Rank {rank}] Input shape: {x.shape} -> Output shape: {out.shape}\n")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_demo()

"""
Rank 0/2 using device spyre
Rank 0 - Input rows (first element): tensor([10., 11.], dtype=torch.float16, device='spyre:0')
Rank 0 - input_splits=[1, 1] output_splits=[1, 2]
Rank 0 - Compiling function...
Rank 0 - Executing all_to_all


Rank 0 - After all_to_all (first element of each row): tensor([21., 41., 41.], dtype=torch.float16, device='spyre:0')
Rank 0 - Expected: [21.0, 41.0, 41.0] = 2 * (10 * (sender + 1) + 0) + 1

[Rank 0] Input shape: torch.Size([2, 64]) -> Output shape: torch.Size([3, 64])

Rank 1/2 using device spyre
Rank 1 - Input rows (first element): tensor([20., 20., 21., 21.], dtype=torch.float16, device='spyre:1')
Rank 1 - input_splits=[2, 2] output_splits=[1, 2]
Rank 1 - Compiling function...
Rank 1 - Executing all_to_all


Rank 1 - After all_to_all (first element of each row): tensor([23., 43., 43.], dtype=torch.float16, device='spyre:1')
Rank 1 - Expected: [23.0, 43.0, 43.0] = 2 * (10 * (sender + 1) + 1) + 1

[Rank 1] Input shape: torch.Size([4, 64]) -> Output shape: torch.Size([3, 64])
"""
