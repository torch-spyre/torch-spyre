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
import argparse
import torch
import torch.distributed as dist
import os

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"


def run_test(comm_rank, comm_size, async_op=False):
    """Run an allgather test where each rank contributes and receives from all.

    Args:
        comm_rank: Rank of the current process
        comm_size: Total number of processes
        async_op: If True, launch the collective asynchronously and overlap CPU
                  work with the hardware operation before calling work.wait().
    """
    global DEVICE

    num_elements = 128

    # Create contiguous range for this rank: rank 0 gets [0..num_elements-1],
    # rank 1 gets [num_elements..2*num_elements-1], etc.
    start_value = comm_rank * num_elements
    end_value = start_value + num_elements
    input_tensor = torch.arange(start_value, end_value, dtype=torch.float16)

    print("-" * 70)
    print(f"[{comm_rank} of {comm_size}] Input Tensor: {input_tensor.shape}")
    print(f"[{comm_rank} of {comm_size}] {input_tensor[:10]} .. {input_tensor[-10:]}")

    # Send input tensor to Spyre device
    input_device = input_tensor.to(DEVICE)

    # Prepare output tensors - all ranks need this for allgather
    output_list = [torch.zeros_like(input_device) for _ in range(comm_size)]

    # Build expected values on CPU while the collective runs on hardware
    expected = []
    for rank_idx in range(comm_size):
        rank_start = rank_idx * num_elements
        rank_end = rank_start + num_elements
        expected.append(torch.arange(rank_start, rank_end, dtype=torch.float16))

    if async_op:
        # Launch allgather asynchronously — returns a Work handle immediately
        print(f"[{comm_rank} of {comm_size}] Allgather Tensor (async): Spyre")
        work = dist.all_gather(output_list, input_device, async_op=True)

        # Note: Opportunity for overlapping of host activities with asynchronous communication.

        # Block until the async collective has completed
        work.wait()
    else:
        # Allgather with the collective library
        print(f"[{comm_rank} of {comm_size}] Allgather Tensor: Spyre")
        dist.all_gather(output_list, input_device)

    # Check the result at all ranks
    print(f"[{comm_rank} of {comm_size}] Gathered tensors from all ranks:")
    all_correct = True
    for rank_idx in range(comm_size):
        result = output_list[rank_idx].to("cpu")
        print(f"  From rank {rank_idx}: {result[:10]} .. {result[-10:]}")

        if torch.allclose(result, expected[rank_idx]):
            print(f"  Rank {rank_idx} tensor is correct")
        else:
            print(f"  Rank {rank_idx} tensor is incorrect!")
            all_correct = False

    if all_correct:
        print(f"[{comm_rank} of {comm_size}] All gathered tensors are correct")
    else:
        raise RuntimeError(
            f"[{comm_rank} of {comm_size}] Some gathered tensors are incorrect"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed allgather example")
    parser.add_argument(
        "--async",
        dest="async_op",
        action="store_true",
        default=False,
        help="Launch allgather asynchronously (async_op=True)",
    )
    args = parser.parse_args()

    # Check that the c10d backend was loaded properly
    if dist.distributed_c10d.is_backend_available(C10D_BACKEND) is False:
        raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
    if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
        raise RuntimeError(
            f"Error: Missing a C10 Backend for {'spyre'}! Expected {C10D_BACKEND}"
        )

    # Initialize the distributed environment
    # Add 'cpu:gloo' since we want to use the backend as well
    print("# Initialize Distributed Group ")
    dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

    comm_size = dist.get_world_size()
    comm_rank = dist.get_rank()

    run_test(comm_rank, comm_size, async_op=args.async_op)

    dist.destroy_process_group()
