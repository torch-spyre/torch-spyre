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

import torch_spyre  # noqa: F401
from inductor.utils_inductor import expects_raise

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

# Uneven splits move whole 128-byte sticks. In 2-D tests each row is one
# stick, the widest row whose default device layout keeps dim-0 rows
# contiguous; wider rows are tiled outside dim 0 and are rejected at compile.
_STICK_BYTES = 128


def _row_elems(dtype: torch.dtype) -> int:
    return _STICK_BYTES // torch.tensor([], dtype=dtype).element_size()


def _uneven_rows(world: int) -> list[list[int]]:
    """pattern[s][d]: rows rank s sends to rank d; consistent across ranks."""
    return [[(s + d) % world + 1 for d in range(world)] for s in range(world)]


def _tag(src: int, dst: int, row: int) -> float:
    """Value of row ``row`` of the slice src sends to dst.

    At most 7 * 64 + 7 * 8 + 8 = 512 for up to 8 ranks, so ``2 * tag + 1``
    (the interleaved-compute check) stays below 2048, where fp16 is exact.
    """
    return float(src * 64 + dst * 8 + row)


class UnevenAllToAllModule(torch.nn.Module):
    """all_to_all_single with explicit (possibly uneven) split sizes."""

    def __init__(self, output_splits, input_splits, group_name=_GROUP_NAME):
        super().__init__()
        self._output_splits = list(output_splits)
        self._input_splits = list(input_splits)
        self._group_name = group_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.ops._c10d_functional.all_to_all_single(
            x, self._output_splits, self._input_splits, self._group_name
        )
        return torch.ops._c10d_functional.wait_tensor(y)


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


@instantiate_parametrized_tests
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

    # ------------------------------------------------------------------
    # Uneven splits
    # ------------------------------------------------------------------

    def _uneven_case(self, dtype, one_d):
        """Inputs, splits and expected output for the uneven pattern.

        Rank s sends (s + d) % W + 1 rows to rank d; each row is filled with
        _tag(s, d, row) so every received row identifies its source.
        """
        pattern = _uneven_rows(self.comm_size)
        me = self.comm_rank
        width = _row_elems(dtype)
        send = pattern[me]
        recv = [pattern[src][me] for src in range(self.comm_size)]

        def rows(pairs):
            return torch.stack(
                [torch.full((width,), _tag(s, d, r), dtype=dtype) for s, d, r in pairs]
            )

        x = rows([(me, d, r) for d, n in enumerate(send) for r in range(n)])
        expected = rows([(s, me, r) for s, n in enumerate(recv) for r in range(n)])
        if one_d:
            # 1-D: dim 0 counts elements, so each row is `width` of them.
            x, expected = x.reshape(-1), expected.reshape(-1)
            send = [n * width for n in send]
            recv = [n * width for n in recv]
        return x, recv, send, expected, width

    def _check_uneven(self, result, expected, width):
        result_cpu = result.to("cpu")
        self.assertEqual(result_cpu.shape, expected.shape)
        self.assertTrue(
            torch.equal(result_cpu, expected),
            f"Rank {self.comm_rank}: uneven all_to_all mismatch; row tags got "
            f"{result_cpu.reshape(-1, width)[:, 0].tolist()}, expected "
            f"{expected.reshape(-1, width)[:, 0].tolist()}",
        )

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("one_d", [False, True])
    def test_all_to_all_compiled_uneven_splits(self, dtype, one_d):
        """Uneven splits: the output's dim 0 is sum(output_split_sizes)."""
        x, recv, send, expected, width = self._uneven_case(dtype, one_d)
        compiled_module = torch.compile(UnevenAllToAllModule(recv, send))
        self._check_uneven(compiled_module(x.to(DEVICE)), expected, width)

    def test_all_to_all_compiled_uneven_with_interleaved_compute(self):
        """Compute before and after an uneven all_to_all, which changes dim 0."""
        x, recv, send, expected, width = self._uneven_case(torch.float16, False)

        class Module(torch.nn.Module):
            def forward(self, t, y):
                exchanged = torch.ops._c10d_functional.all_to_all_single(
                    t * 2.0, recv, send, _GROUP_NAME
                )
                return torch.ops._c10d_functional.wait_tensor(exchanged) + y

        y = torch.ones(expected.shape, dtype=torch.float16)
        compiled_module = torch.compile(Module())
        result = compiled_module(x.to(DEVICE), y.to(DEVICE))
        self._check_uneven(result, expected * 2.0 + 1.0, width)

    def test_all_to_all_compiled_uneven_repeated_execution(self):
        """Uneven compiled all_to_all executed multiple times (WSI reuse)."""
        x, recv, send, expected, width = self._uneven_case(torch.float16, False)
        compiled_module = torch.compile(UnevenAllToAllModule(recv, send))
        x_dev = x.to(DEVICE)
        for _ in range(4):
            self._check_uneven(compiled_module(x_dev), expected, width)

    # ------------------------------------------------------------------
    # Rejections: each must fail at compile time, before reaching spyre-comms
    # ------------------------------------------------------------------

    @expects_raise
    def test_all_to_all_compiled_rejects_rows_not_contiguous(self):
        """Rows wider than one stick are tiled outside dim 0 on device.

        TODO: support these by restickifying to a row-contiguous layout.
        """
        splits = [2] * self.comm_size
        x = torch.ones((2 * self.comm_size, 128), dtype=torch.float16).to(DEVICE)
        compiled_module = torch.compile(UnevenAllToAllModule(splits, splits))
        with pytest.raises(Exception, match="rows are not contiguous"):
            compiled_module(x)

    @expects_raise
    def test_all_to_all_compiled_rejects_zero_split(self):
        """A zero-size split is rejected at compile time.

        TODO: spyre-comms alltoallw segfaults applying a plan with a zero
        split; make this a positive test once it handles them.
        """
        width = _row_elems(torch.float16)
        splits = [0] + [2] * (self.comm_size - 1)
        x = torch.ones((sum(splits), width), dtype=torch.float16).to(DEVICE)
        compiled_module = torch.compile(UnevenAllToAllModule(splits, splits))
        with pytest.raises(Exception, match="zero-size splits are not supported"):
            compiled_module(x)


if __name__ == "__main__":
    run_tests()
