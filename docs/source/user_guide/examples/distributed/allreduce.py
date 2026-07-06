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
import os

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"


def run_test(comm_rank, comm_size):
    """Run an allreduce test where all ranks contribute and all receive the sum."""
    global DEVICE

    # Each rank creates a tensor filled with its rank+1 value
    input_tensor = torch.zeros(128, dtype=torch.float16)
    input_tensor.fill_(float(comm_rank + 1))

    print("-" * 70)
    print(
        f"[{comm_rank} of {comm_size}] Input Tensor (Before Allreduce): {input_tensor.shape}"
    )
    print(f"[{comm_rank} of {comm_size}] {input_tensor[:10]}")

    # Send input tensor to Spyre device
    input_device = input_tensor.to(DEVICE)

    # Allreduce with the collective library (SUM operation)
    print(f"[{comm_rank} of {comm_size}] Allreduce Tensor (SUM): Spyre")
    dist.all_reduce(input_device, op=dist.ReduceOp.SUM)

    # Check the result at all ranks
    result = input_device.to("cpu")
    print(f"[{comm_rank} of {comm_size}] Reduced Tensor (SUM of all ranks):")
    print(f"[{comm_rank} of {comm_size}] {result[:10]}")

    # Expected result: sum of (1 + 2 + 3 + ... + comm_size)
    expected_sum = sum(range(1, comm_size + 1))
    expected_tensor = torch.zeros(128, dtype=torch.float16)
    expected_tensor.fill_(float(expected_sum))

    print(f"  Expected value per element: {expected_sum}")

    if torch.allclose(result, expected_tensor):
        print(f"[{comm_rank} of {comm_size}] Reduced tensor is correct")
    else:
        raise RuntimeError(
            f"[{comm_rank} of {comm_size}] Reduced tensor is incorrect: "
            f"expected {expected_tensor[:10]} but got {result[:10]}"
        )


if __name__ == "__main__":
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

    run_test(comm_rank, comm_size)

    dist.destroy_process_group()

# Made with Bob
