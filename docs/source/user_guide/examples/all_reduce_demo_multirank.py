import os

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

import torch_spyre  # noqa: F401


def run_demo():
    torch.spyre._impl._lazy_init()

    rank = int(os.environ.get("RANK", "0"))
    device = torch.device(f"spyre:{rank}")

    dist.init_process_group("cpu:gloo,spyre:spyreccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}/{world_size} using device {device}")

    c10d._register_process_group("default", dist.group.WORLD)

    # Each rank contributes its own value: rank+1
    x = torch.full((8, 8), float(rank + 1), dtype=torch.float16, device=device)
    print(f"Rank {rank} - Initial tensor: {x[0, :4]}")

    independent = (torch.ones(8, 8) * 10.0).to(dtype=torch.float16, device=device)

    def fn(t, ind):
        # Pre-reduce computation
        y = t + t

        # All-reduce (sum) across all ranks - lowered to allreduce_async
        y_reduced = torch.ops._c10d_functional.all_reduce(y, "sum", "default")

        # Independent computation (potential overlap with collective)
        ind_result = ind * ind * ind  # 10^3 = 1000

        # Wait for all_reduce to complete
        y_ready = torch.ops._c10d_functional.wait_tensor(y_reduced)

        # Combine results
        z = y_ready + ind_result
        return z

    print(f"Rank {rank} - Compiling function...")
    compiled_fn = torch.compile(fn)

    print(f"Rank {rank} - Executing all_reduce")
    out = compiled_fn(x, independent)

    # Expected: sum of 2*(rank+1) for all ranks + 1000
    # = 2 * (1 + 2 + ... + world_size) + 1000
    # For world_size=2: 2*(1+2) + 1000 = 6 + 1000 = 1006
    expected_sum = 2 * (world_size * (world_size + 1) // 2)
    expected_value = float(expected_sum + 1000)

    print("\n")
    print(f"Rank {rank} - After all_reduce: {out[0, :4]}")
    print(f"Rank {rank} - Expected: {expected_value} = 2*sum(1..{world_size}) + 10^3")
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
Rank 0 - Executing all_reduce
Rank 1 - Executing all_reduce


Rank 0 - After all_reduce: tensor([1006., 1006., 1006., 1006.], device='spyre:0')
Rank 0 - Expected: 1006.0 = 2*sum(1..2) + 10^3

[Rank 0] Output shape: torch.Size([8, 8])


Rank 1 - After all_reduce: tensor([1006., 1006., 1006., 1006.], device='spyre:0')
Rank 1 - Expected: 1006.0 = 2*sum(1..2) + 10^3

[Rank 1] Output shape: torch.Size([8, 8])
"""
