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
import math
import torch
import torch.distributed as dist
import os

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"


def run_test(comm_rank, comm_size, async_op=False, num_elements=128):
    """Run an allreduce test where all ranks contribute and all receive the sum.

    Args:
        comm_rank: Rank of the current process
        comm_size: Total number of processes
        async_op: If True, launch the collective asynchronously and overlap CPU
                  work with the hardware operation before calling work.wait().
        num_elements: Number of elements per rank tensor. Reduce this to lower
                      the peak float16 magnitude when testing at large world sizes.
    """
    global DEVICE

    # Create contiguous range for this rank: rank 0 gets [0..num_elements-1],
    # rank 1 gets [num_elements..2*num_elements-1], etc.
    start_value = comm_rank * num_elements
    end_value = start_value + num_elements
    input_tensor = torch.arange(start_value, end_value, dtype=torch.float16)

    print("-" * 70)
    print(
        f"[{comm_rank} of {comm_size}] Input Tensor (Before Allreduce): {input_tensor.shape}"
    )
    print(f"[{comm_rank} of {comm_size}] {input_tensor[:10]} .. {input_tensor[-10:]}")

    # Send input tensor to Spyre device
    input_device = input_tensor.to(DEVICE)

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
        # Launch allreduce asynchronously — returns a Work handle immediately
        print(f"[{comm_rank} of {comm_size}] Allreduce Tensor (SUM, async): Spyre")
        work = dist.all_reduce(input_device, op=dist.ReduceOp.SUM, async_op=True)

        # Note: Opportunity for overlapping of host activities with asynchronous communication.

        # Block until the async collective has completed
        work.wait()
    else:
        # Allreduce with the collective library (SUM operation)
        print(f"[{comm_rank} of {comm_size}] Allreduce Tensor (SUM): Spyre")
        dist.all_reduce(input_device, op=dist.ReduceOp.SUM)

    # Check the result at all ranks
    result = input_device.to("cpu")
    print(f"[{comm_rank} of {comm_size}] Reduced Tensor (SUM of all ranks):")
    print(f"[{comm_rank} of {comm_size}] {result[:10]} .. {result[-10:]}")
    print(f"  Expected values: {expected_tensor[:10]} .. {expected_tensor[-10:]}")

    # Tolerance: 1 ULP of float16 at the maximum *chunk* partial-sum magnitude.
    # dist.all_reduce dispatches to ReduceScatterAllGather: each rank accumulates
    # only a 1/N slice of the tensor (num_elements/comm_size elements), receiving
    # one contribution from each rank. The maximum value any rank sums locally is
    # bounded by comm_size × max_single_element, where max_single_element is the
    # largest value any rank contributes: (comm_size-1)*num_elements + (num_elements-1).
    # This is O(N * num_elements) — one factor of N smaller than the full reduce.py
    # magnitude — so 1 ULP at that scale is the appropriate bound. The allgather
    # phase that follows is pure data movement and introduces no rounding.
    # ULP of float16 at value v = 2^(floor(log2(v)) - 10).
    # note: This formula assumes ReduceScatterAllGather is selected at runtime.
    # If a different algorithm is chosen (e.g. BiTreeBcast, GatherSumBcast), the
    # error model changes and this tolerance may need revisiting.
    max_single_element = float((comm_size - 1) * num_elements + (num_elements - 1))
    chunk_max = comm_size * max_single_element
    atol = 2.0 ** (math.floor(math.log2(chunk_max)) - 10)
    print(
        f"  Tolerance: atol={atol} "
        f"(1 ULP of float16 at chunk max {chunk_max}, algo=ReduceScatterAllGather assumed)"
    )

    if torch.allclose(result, expected_tensor, atol=atol, rtol=0.0):
        print(f"[{comm_rank} of {comm_size}] Reduced tensor is correct")
    else:
        raise RuntimeError(
            f"[{comm_rank} of {comm_size}] Reduced tensor is incorrect: "
            f"expected {expected_tensor[:10]} but got {result[:10]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed allreduce example")
    parser.add_argument(
        "--async",
        dest="async_op",
        action="store_true",
        default=False,
        help="Launch allreduce asynchronously (async_op=True)",
    )
    parser.add_argument(
        "--num-elements",
        dest="num_elements",
        type=int,
        default=128,
        help=(
            "Number of elements per rank tensor (default: 128). "
            "Reduce this to lower peak float16 magnitude at large world sizes."
        ),
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

    run_test(comm_rank, comm_size, async_op=args.async_op, num_elements=args.num_elements)

    dist.destroy_process_group()
