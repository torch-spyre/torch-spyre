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


def run_test(expected_tensor, comm_rank, comm_size):
    """Run a broadcast test with the given expected tensor."""
    global DEVICE
    if 0 != comm_rank:
        x = torch.ones_like(expected_tensor)
    else:
        x = expected_tensor

    # Send input tensor to Spyre device
    print("-" * 70)
    print(f"[{comm_rank} of {comm_size}] Tensor Input: {x.shape}")
    print(f"[{comm_rank} of {comm_size}] {x[:10]}")
    x_device = x.to(DEVICE)

    # Broadcast with the collective library
    print(f"[{comm_rank} of {comm_size}] Broadcast Tensor: Spyre")
    dist.broadcast(x_device, 0)

    result = x_device.to("cpu")
    print(f"[{comm_rank} of {comm_size}] Tensor after collective")
    print(f"[{comm_rank} of {comm_size}] {result[:10]}")

    # Check the result
    if torch.allclose(result, expected_tensor):
        print(f"[{comm_rank} of {comm_size}] Tensor is correct")
    else:
        raise RuntimeError(
            f"[{comm_rank} of {comm_size}] Tensor is incorrect: "
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

    exp_result = torch.zeros(128, dtype=torch.float16)
    exp_result.fill_(2.0)
    run_test(exp_result, comm_rank, comm_size)

    exp_result2 = torch.zeros(512, 1024, dtype=torch.float16)
    exp_result2.fill_(4.0)
    run_test(exp_result2, comm_rank, comm_size)

    dist.destroy_process_group()
