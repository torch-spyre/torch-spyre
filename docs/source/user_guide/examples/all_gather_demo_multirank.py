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

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


def run_demo():
    device = torch.device("spyre")

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}/{world_size} using device {device}")

    c10d._register_process_group("default", dist.group.WORLD)

    # Each rank creates a unique tensor - must be at least 128 bytes for spyre-comms
    # Using 8x8 float32 = 256 bytes (meets minimum requirement)
    x = torch.ones(8, 8).to(device) * (rank + 1.0)
    print(f"Rank {rank} - Initial tensor: {x[0, :4]}")

    independent = torch.ones(16, 8).to(device) * 5.0

    def fn(t, ind):
        # Pre-gather computation
        y = t * 2

        # All-gather from all ranks - lowered to all_gather_async
        # Output shape[0] = input shape[0] * world_size = 8*2 = 16
        y_gathered = torch.ops._c10d_functional.all_gather_into_tensor(
            y, world_size, "default"
        )

        # Independent computation (compiler in future can schedule this
        # to overlap with all_gather)
        ind_result = ind * ind  # 5^2 = 25

        # Wait for all_gather to complete
        y_ready = torch.ops._c10d_functional.wait_tensor(y_gathered)

        # Element-wise operation on the gathered result
        # y_ready shape: [16, 8] — first 8 rows from rank0 (val=2), next 8 from rank1 (val=4)
        # z = gathered * 3 + 25
        z = y_ready * 3 + ind_result
        return z

    print(f"Rank {rank} - Compiling function...")
    compiled_fn = torch.compile(fn)

    print(f"Rank {rank} - Executing all_gather")
    out = compiled_fn(x, independent)

    print("\n")
    print(f"Rank {rank} - After all_gather (first half): {out[0, :4]}")
    print(f"Rank {rank} - After all_gather (second half): {out[8, :4]}")
    # With 2 ranks: rank0 contributes 1*2=2, rank1 contributes 2*2=4
    # First 8 rows: 2*3 + 25 = 31
    # Last 8 rows:  4*3 + 25 = 37
    print(f"Rank {rank} - Expected first half: 31.0 = (1*2)*3 + 5^2")
    print(f"Rank {rank} - Expected second half: 37.0 = (2*2)*3 + 5^2")
    print(f"\n[Rank {rank}] Output shape: {out.shape}\n")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_demo()

"""
Rank 0/2 using device spyre
Rank 1/2 using device spyre
Rank 0 - Initial tensor: tensor([1., 1., 1., 1.], device='spyre:0')
Rank 1 - Initial tensor: tensor([2., 2., 2., 2.], device='spyre:0')
Rank 0 - Compiling function...
Rank 1 - Compiling function...
Rank 0 - Executing all_gather
Rank 1 - Executing all_gather


Rank 0 - After all_gather (first half): tensor([31., 31., 31., 31.], device='spyre:0')
Rank 0 - After all_gather (second half): tensor([37., 37., 37., 37.], device='spyre:0')
Rank 0 - Expected first half: 31.0 = (1*2)*3 + 5^2
Rank 0 - Expected second half: 37.0 = (2*2)*3 + 5^2

[Rank 0] Output shape: torch.Size([16, 8])



Rank 1 - After all_gather (first half): tensor([31., 31., 31., 31.], device='spyre:0')
Rank 1 - After all_gather (second half): tensor([37., 37., 37., 37.], device='spyre:0')
Rank 1 - Expected first half: 31.0 = (1*2)*3 + 5^2
Rank 1 - Expected second half: 37.0 = (2*2)*3 + 5^2

[Rank 1] Output shape: torch.Size([16, 8])
"""
