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
    """Run a reduce test where all ranks contribute and root receives the sum.

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
    print(
        f"[{comm_rank} of {comm_size}] Input Tensor (Before Reduce): {input_tensor.shape}"
    )
    print(f"[{comm_rank} of {comm_size}] {input_tensor[:10]} .. {input_tensor[-10:]}")

    # Send input tensor to Spyre device
    input_device = input_tensor.to(DEVICE)

    expected_tensor = None
    if comm_rank == 0:
        # Expected result: sum of all ranks' contributions at each position
        # Position i gets: (0*num_elements + i) + (1*num_elements + i) + ... + ((comm_size-1)*num_elements + i)
        # = i*comm_size + num_elements*(0 + 1 + ... + (comm_size-1))
        # = i*comm_size + num_elements*comm_size*(comm_size-1)/2
        expected_tensor = torch.zeros(num_elements, dtype=torch.float16)
        for i in range(num_elements):
            expected_tensor[i] = (
                i * comm_size + num_elements * comm_size * (comm_size - 1) / 2
            )

    if async_op:
        # Launch reduce asynchronously — returns a Work handle immediately
        print(f"[{comm_rank} of {comm_size}] Reduce Tensor (SUM, async): Spyre")
        work = dist.reduce(input_device, dst=0, op=dist.ReduceOp.SUM, async_op=True)

        # Note: Opportunity for overlapping of host activities with asynchronous communication.

        # Block until the async collective has completed
        work.wait()
    else:
        # Reduce with the collective library (SUM operation to root rank 0)
        print(f"[{comm_rank} of {comm_size}] Reduce Tensor (SUM): Spyre")
        dist.reduce(input_device, dst=0, op=dist.ReduceOp.SUM)

    # Check the result at root
    if comm_rank == 0:
        result = input_device.to("cpu")
        print(
            f"[{comm_rank} of {comm_size}] Reduced Tensor at root (SUM of all ranks):"
        )
        print(f"[{comm_rank} of {comm_size}] {result[:10]} .. {result[-10:]}")
        print(f"  Expected values: {expected_tensor[:10]} .. {expected_tensor[-10:]}")

        if torch.allclose(result, expected_tensor):
            print(f"[{comm_rank} of {comm_size}] Reduced tensor is correct")
        else:
            raise RuntimeError(
                f"[{comm_rank} of {comm_size}] Reduced tensor is incorrect: "
                f"expected {expected_tensor[:10]} but got {result[:10]}"
            )
    else:
        mode = " (async, input consumed)" if async_op else " (input consumed)"
        print(f"[{comm_rank} of {comm_size}] Non-root rank completed reduce{mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed reduce example")
    parser.add_argument(
        "--async",
        dest="async_op",
        action="store_true",
        default=False,
        help="Launch reduce asynchronously (async_op=True)",
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
