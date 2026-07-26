import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


def run_demo():
    device = torch.device("spyre")

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    root_rank = 0

    print(f"Rank {rank}/{world_size} using device {device}")

    c10d._register_process_group("default", dist.group.WORLD)

    # Create tensor - must be at least 128 bytes for spyre-comms
    # Using 8x8 float32 = 256 bytes (meets minimum requirement)
    if rank == root_rank:
        x = torch.ones(8, 8).to(device) * 42.0
        print(f"Rank {rank} (ROOT) - Initial tensor: {x[0, :4]}")
    else:
        x = torch.zeros(8, 8).to(device)
        # print(f"Rank {rank} - Initial tensor: {x[0, :4]}")

    independent = torch.ones(8, 8).to(device) * 10.0

    def fn(t, ind):
        # Pre-broadcast computation
        y = t + t

        # Broadcast from root rank - lowered to broadcast_async
        y_bcast = torch.ops._c10d_functional.broadcast(y, root_rank, "default")

        # Independent computation (compiler in future can schedule this to overlap with broadcast)
        ind_result = ind * ind * ind  # 10^3 = 1000

        # Wait for broadcast to complete
        y_ready = torch.ops._c10d_functional.wait_tensor(y_bcast)

        # Combine results: (42*2)*2 + 1000 = 168 + 1000 = 1168
        z = y_ready * 2 + ind_result
        return z

    print(f"Rank {rank} - Compiling function...")
    compiled_fn = torch.compile(fn)

    print(f"Rank {rank} - Executing broadcast")
    out = compiled_fn(x, independent)

    print("\n")
    print(f"Rank {rank} - After broadcast: {out[0, :4]}")
    print(f"Rank {rank} - Expected: 1168.0 = (42*2)*2 + 10^3")
    print(f"\n[Rank {rank}] Output shape: {out.shape}\n")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_demo()

"""
Rank 1/2 using device spyre
Rank 0/2 using device spyre
Rank 1 - Compiling function...
Rank 0 (ROOT) - Initial tensor: tensor([42., 42., 42., 42.], device='spyre:0')
Rank 1 - Executing broadcast
Rank 0 - Compiling function...
Rank 0 - Executing broadcast


Rank 1 - After broadcast: tensor([1168., 1168., 1168., 1168.], device='spyre:0')
Rank 1 - Expected: 1168.0 = (42*2)*2 + 10^3

[Rank 1] Output shape: torch.Size([8, 8])



Rank 0 - After broadcast: tensor([1168., 1168., 1168., 1168.], device='spyre:0')
Rank 0 - Expected: 1168.0 = (42*2)*2 + 10^3

[Rank 0] Output shape: torch.Size([8, 8])
"""
