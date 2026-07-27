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

import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_spyre  # noqa: F401

# Skip all tests if RANK is not defined, or WORLD_SIZE is not set or less than 2
if "RANK" not in os.environ:
    pytest.skip(
        "RANK environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

if "WORLD_SIZE" not in os.environ:
    pytest.skip(
        "WORLD_SIZE environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

try:
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size < 2:
        pytest.skip(
            f"WORLD_SIZE is {world_size}, need at least 2 for distributed tests",
            allow_module_level=True,
        )
except ValueError:
    pytest.skip(
        "WORLD_SIZE environment variable is not a valid integer, skipping distributed tests",
        allow_module_level=True,
    )

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"
_GROUP_NAME = "default"


class BroadcastCompiledModule(torch.nn.Module):
    """Module that performs broadcast using functional collective ops."""

    def __init__(self, src_rank: int = 0, group_name: str = _GROUP_NAME) -> None:
        super().__init__()
        self._src_rank = src_rank
        self._group_name = group_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.ops._c10d_functional.broadcast(x, self._src_rank, self._group_name)
        return torch.ops._c10d_functional.wait_tensor(y)


class TestBroadcastCompiled(TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the distributed process group once for the whole test class."""
        torch.spyre._impl._lazy_init()

        if not dist.distributed_c10d.is_backend_available(C10D_BACKEND):
            raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
        if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
            raise RuntimeError(
                f"Error: Missing a C10 Backend for 'spyre'! Expected {C10D_BACKEND}"
            )

        if not dist.is_initialized():
            dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

        c10d._register_process_group(_GROUP_NAME, dist.group.WORLD)

        cls.comm_size = dist.get_world_size()
        cls.comm_rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        """Destroy the distributed process group after all tests complete."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        """Reset compiler caches before each test."""
        super().setUp()
        torch.compiler.reset()

    def test_broadcast_compiled_fp32(self):
        """Verify compiled broadcast works with fp32."""
        x = torch.ones((128,), dtype=torch.float32, device=DEVICE)
        module = BroadcastCompiledModule()
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        expected = torch.ones((128,), dtype=torch.float32)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: broadcast result incorrect",
        )

    def test_broadcast_compiled_fp16(self):
        """Verify compiled broadcast preserves fp16 and correctness."""
        x = torch.ones((256,), dtype=torch.float16, device=DEVICE)
        module = BroadcastCompiledModule()
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.shape, x.shape)
        expected = torch.ones((256,), dtype=torch.float16)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: broadcast result incorrect",
        )

    def test_broadcast_compiled_with_interleaved_compute(self):
        """Verify compute between broadcast_async and wait_work compiles correctly."""

        class BroadcastWithCompute(torch.nn.Module):
            def __init__(self, src_rank=0, group_name=_GROUP_NAME):
                super().__init__()
                self._src_rank = src_rank
                self._group_name = group_name

            def forward(self, x, y):
                bcast = torch.ops._c10d_functional.broadcast(
                    x, self._src_rank, self._group_name
                )
                # Compute placed between async broadcast and wait
                z = y * 2.0
                result = torch.ops._c10d_functional.wait_tensor(bcast)
                return result + z

        x = torch.ones((128,), dtype=torch.float32, device=DEVICE)
        y = torch.ones((128,), dtype=torch.float32, device=DEVICE)
        module = BroadcastWithCompute()
        compiled_module = torch.compile(module)
        result = compiled_module(x, y)

        expected = torch.ones((128,), dtype=torch.float32) * 3.0
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: broadcast with interleaved compute incorrect",
        )


if __name__ == "__main__":
    run_tests()
