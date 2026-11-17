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
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from utils import (
    _bfloat16_collective_expected,
    _bfloat16_rank_input,
    _collective_expected,
    _collective_rank_input,
    _native_bfloat16_tensor,
    _overpadded_collective_tensor,
)

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


class AllGatherCompiledModule(torch.nn.Module):
    """Module that performs all_gather using functional collective ops."""

    def __init__(self, group_size: int, group_name: str = _GROUP_NAME) -> None:
        super().__init__()
        self._group_size = group_size
        self._group_name = group_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.ops._c10d_functional.all_gather_into_tensor(
            x, self._group_size, self._group_name
        )
        return torch.ops._c10d_functional.wait_tensor(y)


@instantiate_parametrized_tests
class TestAllGatherCompiled(TestCase):
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

    @parametrize("dtype,dtype_code", [(torch.float16, 5), (torch.bfloat16, 15)])
    @parametrize("width,padded_width", [(128, 256), (24640, 26624)])
    def test_all_gather_runtime_overpadded_storage(
        self, dtype, dtype_code, width, padded_width
    ):
        # Exercise the compiled runtime directly with a finalized producer layout.
        # empty_like would allocate fewer bytes than this plan transfers.
        plan = torch.ops.spyre.allgather_plan(
            padded_width, dtype_code, self.comm_size, _GROUP_NAME
        )
        for step in range(3):
            values = _collective_rank_input(self.comm_rank, (1, width), dtype)
            source = _overpadded_collective_tensor(
                values * (step + 1), padded_width, DEVICE
            )
            result = torch.ops.spyre.allgather_run(source, plan, self.comm_size)
            # Generated code releases its temporary before the wait as well.
            del source
            result = torch.ops.spyre.wait_work(result)
            expected = torch.cat(
                [
                    _collective_rank_input(rank, (1, width), dtype) * (step + 1)
                    for rank in range(self.comm_size)
                ]
            )
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.cpu(), expected, rtol=0, atol=0)

    def test_all_gather_compiled_fp32(self):
        """Verify compiled all_gather works with fp32."""
        x = torch.full(
            (128,), float(self.comm_rank), dtype=torch.float32, device=DEVICE
        )
        module = AllGatherCompiledModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.shape[0], 128 * self.comm_size)

        result_cpu = result.to("cpu")
        for rank in range(self.comm_size):
            chunk = result_cpu[rank * 128 : (rank + 1) * 128]
            expected = torch.full((128,), float(rank), dtype=torch.float32)
            self.assertTrue(
                torch.allclose(chunk, expected),
                f"Rank {self.comm_rank}: all_gather chunk for rank {rank} incorrect",
            )

    def test_all_gather_compiled_fp16(self):
        """Verify compiled all_gather preserves fp16."""
        x = torch.ones((256,), dtype=torch.float16, device=DEVICE)
        module = AllGatherCompiledModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.shape[0], 256 * self.comm_size)

    def test_all_gather_compiled_fp16_non_stick_aligned(self):
        """Verify all_gather works when the last dim is not stick-aligned.

        A stick holds 64 fp16 elements (128 bytes). A tensor with last dim=96
        requires 2 sticks on device (128 elements), so the device buffer is
        larger than the logical tensor. This exercises the path where
        device_size > logical size due to stick padding.
        """
        x = torch.full(
            (4, 96), float(self.comm_rank), dtype=torch.float16, device=DEVICE
        )
        module = AllGatherCompiledModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.shape, torch.Size([4 * self.comm_size, 96]))

        result_cpu = result.to("cpu")
        for rank in range(self.comm_size):
            chunk = result_cpu[rank * 4 : (rank + 1) * 4]
            expected = torch.full((4, 96), float(rank), dtype=torch.float16)
            if not torch.allclose(chunk, expected):
                diff = chunk != expected
                mismatch_idx = diff.nonzero()
                print(
                    f"Rank {self.comm_rank}: chunk[{rank}] has "
                    f"{diff.sum().item()} mismatches out of {chunk.numel()}"
                )
                if mismatch_idx.numel() > 0:
                    first_bad = mismatch_idx[0]
                    r, c = first_bad[0].item(), first_bad[1].item()
                    print(
                        f"  First mismatch at [{r},{c}]: "
                        f"got {chunk[r, c].item()}, expected {expected[r, c].item()}"
                    )
                print(
                    f"  Row 0 around stick boundary [60:70]: {chunk[0, 60:70].tolist()}"
                )
                print(f"  Has NaN: {torch.isnan(chunk).any().item()}")
            self.assertTrue(
                torch.allclose(chunk, expected),
                f"Rank {self.comm_rank}: all_gather non-stick-aligned chunk "
                f"for rank {rank} incorrect",
            )

    def test_all_gather_compiled_with_interleaved_compute(self):
        """Verify compute around compiled all_gather works correctly."""

        class AllGatherWithComputeModule(torch.nn.Module):
            def __init__(self, group_size: int, group_name: str = _GROUP_NAME):
                super().__init__()
                self._group_size = group_size
                self._group_name = group_name

            def forward(self, x, y):
                x_scaled = x * 2.0
                gathered = torch.ops._c10d_functional.all_gather_into_tensor(
                    x_scaled, self._group_size, self._group_name
                )
                z = y + 1.0
                result = torch.ops._c10d_functional.wait_tensor(gathered)
                return result + z

        x = torch.full(
            (64,), float(self.comm_rank + 1), dtype=torch.float16, device=DEVICE
        )
        y = torch.ones((64 * self.comm_size,), dtype=torch.float16, device=DEVICE)

        module = AllGatherWithComputeModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x, y)

        result_cpu = result.to("cpu")
        for rank in range(self.comm_size):
            chunk = result_cpu[rank * 64 : (rank + 1) * 64]
            # x_scaled = 2*(rank+1), gathered chunk for each rank, then +2.0 (y+1)
            expected_val = 2.0 * (rank + 1) + 2.0
            expected = torch.full((64,), expected_val, dtype=torch.float16)
            self.assertTrue(
                torch.allclose(chunk, expected),
                f"Rank {self.comm_rank}: all_gather with compute chunk "
                f"for rank {rank} incorrect. "
                f"Expected {expected_val}, got {chunk[0].item()}",
            )

    def test_all_gather_compiled_sequential_same_shape(self):
        """Two sequential all_gather calls in one graph."""

        class DoubleAllGatherModule(torch.nn.Module):
            def __init__(self, group_size: int, group_name: str = _GROUP_NAME):
                super().__init__()
                self._group_size = group_size
                self._group_name = group_name

            def forward(self, x):
                y = x + x
                gathered = torch.ops._c10d_functional.all_gather_into_tensor(
                    y, self._group_size, self._group_name
                )
                g1 = torch.ops._c10d_functional.wait_tensor(gathered)
                # Take a slice back to original chunk size for the second gather
                chunk = g1.narrow(0, 0, x.shape[0])
                gathered2 = torch.ops._c10d_functional.all_gather_into_tensor(
                    chunk, self._group_size, self._group_name
                )
                g2 = torch.ops._c10d_functional.wait_tensor(gathered2)
                return g2

        x = torch.full(
            (64,), float(self.comm_rank + 1), dtype=torch.float16, device=DEVICE
        )
        module = DoubleAllGatherModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.shape[0], 64 * self.comm_size)

        # y = 2*(rank+1), first gather concatenates all ranks' values
        # chunk = first 64 elements = rank 0's contribution = 2*(0+1) = 2.0
        # second gather: all ranks contribute this same chunk (2.0)
        result_cpu = result.to("cpu")
        for rank in range(self.comm_size):
            chunk = result_cpu[rank * 64 : (rank + 1) * 64]
            expected = torch.full((64,), 2.0, dtype=torch.float16)
            self.assertTrue(
                torch.allclose(chunk, expected),
                f"Rank {self.comm_rank}: sequential all_gather chunk "
                f"for rank {rank} incorrect. "
                f"Expected 2.0, got {chunk[0].item()}",
            )

    def test_all_gather_compiled_repeated_execution(self):
        """Compiled all_gather executed multiple times (WSI reuse test).

        This mimics the Granite TP pattern: a compiled graph with all_gather
        is run repeatedly (warmup + decode steps). Catches regressions where
        bundle artifacts are not preserved across repeated executeBundle calls
        on the same WSI.
        """
        x = torch.full(
            (128,), float(self.comm_rank), dtype=torch.float16, device=DEVICE
        )
        module = AllGatherCompiledModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)

        for iteration in range(4):
            result = compiled_module(x)
            self.assertEqual(result.shape[0], 128 * self.comm_size)

            result_cpu = result.to("cpu")
            for rank in range(self.comm_size):
                chunk = result_cpu[rank * 128 : (rank + 1) * 128]
                expected = torch.full((128,), float(rank), dtype=torch.float16)
                self.assertTrue(
                    torch.allclose(chunk, expected),
                    f"Rank {self.comm_rank}: iteration {iteration}, "
                    f"all_gather chunk for rank {rank} incorrect. "
                    f"Expected {float(rank)}, got {chunk[0].item()}",
                )

    @parametrize("shape", [(128,), (4, 96)])
    def test_allgather_compiled_bfloat16(self, shape):
        def fn(x):
            x = x * 2.0
            y = torch.ops._c10d_functional.all_gather_into_tensor(
                x, self.comm_size, _GROUP_NAME
            )
            return torch.ops._c10d_functional.wait_tensor(y) * 0.5

        compiled = torch.compile(fn, fullgraph=True)
        for _ in range(2):
            x = _bfloat16_rank_input(self.comm_rank, shape).to(DEVICE)
            result = compiled(x)
            self.assertEqual(result.dtype, torch.bfloat16)
            self.assertEqual(
                result.cpu(),
                _bfloat16_collective_expected("allgather", shape, self.comm_size),
                rtol=0,
                atol=0,
            )

    def test_allgather_compiled_float16_padded_intermediate(self):
        def fn(x):
            x = x * 2.0
            y = torch.ops._c10d_functional.all_gather_into_tensor(
                x, self.comm_size, _GROUP_NAME
            )
            return torch.ops._c10d_functional.wait_tensor(y) * 0.5

        shape = (4, 96)
        compiled = torch.compile(fn, fullgraph=True)
        for _ in range(2):
            x = _collective_rank_input(self.comm_rank, shape, torch.float16).to(DEVICE)
            self.assertEqual(
                compiled(x).cpu(),
                _collective_expected("allgather", shape, self.comm_size, torch.float16),
                rtol=0,
                atol=0,
            )

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("chunk_rank", [0, 1])
    def test_allgather_compiled_padded_view(self, dtype, chunk_rank):
        shape = (4, 96)

        def fn(x):
            gathered = torch.ops._c10d_functional.all_gather_into_tensor(
                x * 2.0, self.comm_size, _GROUP_NAME
            )
            base = torch.ops._c10d_functional.wait_tensor(gathered)
            chunk = base.narrow(0, chunk_rank * shape[0], shape[0])
            gathered_view = torch.ops._c10d_functional.all_gather_into_tensor(
                chunk, self.comm_size, _GROUP_NAME
            )
            return base * 0.5, torch.ops._c10d_functional.wait_tensor(
                gathered_view
            ) * 0.5

        compiled = torch.compile(fn, fullgraph=True)
        expected_chunk = _collective_rank_input(chunk_rank, shape, dtype)
        for _ in range(2):
            x = _collective_rank_input(self.comm_rank, shape, dtype).to(DEVICE)
            base, gathered_view = compiled(x)
            self.assertEqual(
                base.cpu(),
                _collective_expected("allgather", shape, self.comm_size, dtype),
                rtol=0,
                atol=0,
            )
            self.assertEqual(
                gathered_view.cpu(),
                torch.cat([expected_chunk] * self.comm_size),
                rtol=0,
                atol=0,
            )

    def test_allgather_compiled_bfloat16_rejects_non_dl16_storage(self):
        x = _native_bfloat16_tensor(DEVICE)
        with self.assertRaisesRegex(
            RuntimeError, "BF16 collectives require DL16 device storage"
        ):
            torch.ops.spyre.allgather_run(x, -1, self.comm_size)

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("prefix", [False, True])
    def test_allgather_rejects_unsupported_storage_view(self, dtype, prefix):
        expected = torch.ones((5, 96), dtype=dtype)
        base = expected.to(DEVICE)
        x = base[:-1] if prefix else base[1:]
        self.assertEqual(x.storage_offset() == 0, prefix)
        error = (
            "matching device storage extent" if prefix else "nonzero storage offsets"
        )
        with self.assertRaisesRegex(RuntimeError, error):
            torch.ops.spyre.allgather_run(x, -1, self.comm_size)
        self.assertTrue(torch.equal(base.cpu(), expected))


if __name__ == "__main__":
    run_tests()
