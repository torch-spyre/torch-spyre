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

"""Eager ``dist.all_to_all_single`` and the list ``dist.all_to_all`` on the
spyreccl backend (not the compiled path; see ``test_all_to_all_compiled.py``
and, for the ``spyre::alltoall_plan/run`` ops, ``test_all_to_all_ops.py``)."""

import os

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from utils import _unique_rank_input

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


@instantiate_parametrized_tests
class TestAllToAll(TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the distributed environment once for all tests."""
        if not dist.distributed_c10d.is_backend_available(C10D_BACKEND):
            raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
        if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
            raise RuntimeError(
                f"Error: Missing a C10 Backend for 'spyre'! Expected {C10D_BACKEND}"
            )

        if not dist.is_initialized():
            dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

        cls.comm_size = dist.get_world_size()
        cls.comm_rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        """Clean up the distributed environment after all tests."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _finish(self, work, async_op):
        if async_op:
            self.assertIsNotNone(work, "async_op=True must return a Work handle")
            work.wait()
            self.assertTrue(work.is_completed())
        else:
            self.assertIsNone(work, "async_op=False must return None")

    # Row shapes whose dim-0 rows are contiguous on device: 1-D, and 2-D with a
    # last dim of one stick (64) or less (32, padded to one stick).
    ROWS = [(), (64,), (32,)]

    def _splits(self, src, dst):
        # Uneven, rank-dependent rows sent from src to dst.
        return 1 + (src + dst) % 2

    def _row_unit(self, row):
        # 1-D tensors need 64 elements (one fp16 stick) per split.
        return 64 if row == () else 1

    def _input(self, rank, row, splits_from):
        rows = sum(splits_from) * self._row_unit(row)
        return _unique_rank_input(rank, (rows, *row), torch.float16)

    def _expected(self, row, uneven):
        g = self.comm_size
        unit = self._row_unit(row)
        pieces = []
        for src in range(g):
            sends = [self._splits(src, d) if uneven else 1 for d in range(g)]
            x = self._input(src, row, sends)
            start = sum(sends[: self.comm_rank]) * unit
            pieces.append(x[start : start + sends[self.comm_rank] * unit])
        return torch.cat(pieces)

    @parametrize("row", ROWS)
    @parametrize("uneven", [False, True])
    @parametrize("async_op", [False, True])
    def test_all_to_all_single(self, row, uneven, async_op):
        g, rank = self.comm_size, self.comm_rank
        unit = self._row_unit(row)
        sends = [self._splits(rank, d) if uneven else 1 for d in range(g)]
        recvs = [self._splits(s, rank) if uneven else 1 for s in range(g)]
        x = self._input(rank, row, sends).to(DEVICE)
        out = torch.zeros((sum(recvs) * unit, *row), dtype=torch.float16, device=DEVICE)
        kwargs = {}
        if uneven:
            kwargs = dict(
                output_split_sizes=[r * unit for r in recvs],
                input_split_sizes=[s * unit for s in sends],
            )
        work = dist.all_to_all_single(out, x, async_op=async_op, **kwargs)
        self._finish(work, async_op)
        self.assertEqual(out.cpu(), self._expected(row, uneven), rtol=0, atol=0)

    @parametrize("row", [(), (64,)])
    def test_functional_all_to_all_single(self, row):
        g, rank = self.comm_size, self.comm_rank
        unit = self._row_unit(row)
        sends = [self._splits(rank, d) for d in range(g)]
        recvs = [self._splits(s, rank) for s in range(g)]
        x = self._input(rank, row, sends).to(DEVICE)
        out = torch.ops._c10d_functional.all_to_all_single(
            x,
            [r * unit for r in recvs],
            [s * unit for s in sends],
            dist.group.WORLD.group_name,
        )
        out = torch.ops._c10d_functional.wait_tensor(out)
        self.assertEqual(out.cpu(), self._expected(row, True), rtol=0, atol=0)

    @parametrize("row", [(), (64,)])
    @parametrize("async_op", [False, True])
    def test_all_to_all_list(self, row, async_op):
        g, rank = self.comm_size, self.comm_rank
        unit = self._row_unit(row)
        sends = [self._splits(rank, d) for d in range(g)]
        recvs = [self._splits(s, rank) for s in range(g)]
        x = self._input(rank, row, sends)
        inputs = [c.contiguous().to(DEVICE) for c in x.split([s * unit for s in sends])]
        outputs = [
            torch.zeros((r * unit, *row), dtype=torch.float16, device=DEVICE)
            for r in recvs
        ]
        work = dist.all_to_all(outputs, inputs, async_op=async_op)
        self._finish(work, async_op)
        expected = self._expected(row, True).split([r * unit for r in recvs])
        for out, exp in zip(outputs, expected):
            self.assertEqual(out.cpu(), exp, rtol=0, atol=0)

    def test_rejects_tiled_rows(self):
        # (g * 4, 128) fp16: the default layout tiles dim 1 outside dim 0, so
        # dim-0 rows are not contiguous on device.
        x = torch.zeros((self.comm_size * 4, 128), dtype=torch.float16, device=DEVICE)
        out = torch.zeros_like(x)
        with self.assertRaisesRegex(RuntimeError, "rows are not contiguous"):
            dist.all_to_all_single(out, x)

    def test_rejects_partial_stick_split(self):
        # 32 fp16 elements per rank is half a stick.
        x = torch.zeros(self.comm_size * 32, dtype=torch.float16, device=DEVICE)
        out = torch.zeros_like(x)
        with self.assertRaisesRegex(RuntimeError, "multiple of 128 bytes"):
            dist.all_to_all_single(out, x)

    def test_rejects_zero_split(self):
        # Every rank sends everything to itself; spyre-comms crashes on the
        # zero splits, so they are rejected before any transfer.
        g, rank = self.comm_size, self.comm_rank
        x = torch.zeros(64, dtype=torch.float16, device=DEVICE)
        out = torch.zeros_like(x)
        splits = [64 if j == rank else 0 for j in range(g)]
        with self.assertRaisesRegex(RuntimeError, "zero-size .* splits"):
            dist.all_to_all_single(out, x, splits, splits)


if __name__ == "__main__":
    run_tests()
