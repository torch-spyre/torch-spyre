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

torch.spyre._impl._lazy_init()

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"
_GROUP_NAME = "default"

# Elements exchanged with each peer. spyre-comms requires the per-rank chunk to
# be a multiple of 128 bytes, so 128 fp16/fp32 elements (256/512 bytes) is safe.
_CHUNK = 128


class AllToAllCompiledModule(torch.nn.Module):
    """Module that performs equal-split all_to_all using functional c10d ops."""

    def __init__(self, group_size: int, group_name: str = _GROUP_NAME) -> None:
        super().__init__()
        self._group_size = group_size
        self._group_name = group_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        splits = [x.shape[0] // self._group_size] * self._group_size
        y = torch.ops._c10d_functional.all_to_all_single(
            x, splits, splits, self._group_name
        )
        return torch.ops._c10d_functional.wait_tensor(y)


class TestAllToAllCompiled(TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the distributed process group once for the whole test class."""
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

    def _make_input(self, dtype: torch.dtype) -> torch.Tensor:
        """Chunk j of rank r's input holds the value r * 100 + j."""
        chunks = [
            torch.full((_CHUNK,), float(self.comm_rank * 100 + j), dtype=dtype)
            for j in range(self.comm_size)
        ]
        return torch.cat(chunks).to(DEVICE)

    def _check_result(self, result: torch.Tensor, dtype: torch.dtype) -> None:
        """Chunk i of rank r's output equals chunk r of rank i's input."""
        self.assertEqual(result.dtype, dtype)
        self.assertEqual(result.shape[0], _CHUNK * self.comm_size)

        result_cpu = result.to("cpu")
        for src in range(self.comm_size):
            chunk = result_cpu[src * _CHUNK : (src + 1) * _CHUNK]
            expected_val = float(src * 100 + self.comm_rank)
            expected = torch.full((_CHUNK,), expected_val, dtype=dtype)
            self.assertTrue(
                torch.allclose(chunk, expected),
                f"Rank {self.comm_rank}: all_to_all chunk from rank {src} "
                f"incorrect. Expected {expected_val}, got {chunk[0].item()}",
            )

    def test_all_to_all_compiled_fp16(self):
        """Verify compiled all_to_all works with fp16."""
        x = self._make_input(torch.float16)
        compiled_module = torch.compile(AllToAllCompiledModule(self.comm_size))
        self._check_result(compiled_module(x), torch.float16)

    def test_all_to_all_compiled_fp32(self):
        """Verify compiled all_to_all works with fp32."""
        x = self._make_input(torch.float32)
        compiled_module = torch.compile(AllToAllCompiledModule(self.comm_size))
        self._check_result(compiled_module(x), torch.float32)

    def test_all_to_all_compiled_does_not_modify_input(self):
        """all_to_all is out-of-place: the input tensor must be unchanged."""
        x = self._make_input(torch.float16)
        x_before = x.to("cpu").clone()
        compiled_module = torch.compile(AllToAllCompiledModule(self.comm_size))
        compiled_module(x)
        self.assertTrue(torch.equal(x.to("cpu"), x_before))

    def test_all_to_all_compiled_with_interleaved_compute(self):
        """Verify compute around compiled all_to_all works correctly."""

        class AllToAllWithComputeModule(torch.nn.Module):
            def __init__(self, group_size: int, group_name: str = _GROUP_NAME):
                super().__init__()
                self._group_size = group_size
                self._group_name = group_name

            def forward(self, x, y):
                x_scaled = x * 2.0
                splits = [x.shape[0] // self._group_size] * self._group_size
                exchanged = torch.ops._c10d_functional.all_to_all_single(
                    x_scaled, splits, splits, self._group_name
                )
                z = y + 1.0
                result = torch.ops._c10d_functional.wait_tensor(exchanged)
                return result + z

        x = self._make_input(torch.float16)
        y = torch.ones((_CHUNK * self.comm_size,), dtype=torch.float16, device=DEVICE)
        compiled_module = torch.compile(AllToAllWithComputeModule(self.comm_size))
        result = compiled_module(x, y)

        result_cpu = result.to("cpu")
        for src in range(self.comm_size):
            chunk = result_cpu[src * _CHUNK : (src + 1) * _CHUNK]
            expected_val = 2.0 * (src * 100 + self.comm_rank) + 2.0
            expected = torch.full((_CHUNK,), expected_val, dtype=torch.float16)
            self.assertTrue(
                torch.allclose(chunk, expected),
                f"Rank {self.comm_rank}: all_to_all with compute chunk from "
                f"rank {src} incorrect. Expected {expected_val}, "
                f"got {chunk[0].item()}",
            )

    def test_all_to_all_compiled_repeated_execution(self):
        """Compiled all_to_all executed multiple times (WSI reuse test)."""
        x = self._make_input(torch.float16)
        compiled_module = torch.compile(AllToAllCompiledModule(self.comm_size))

        for _ in range(4):
            self._check_result(compiled_module(x), torch.float16)


if __name__ == "__main__":
    run_tests()
