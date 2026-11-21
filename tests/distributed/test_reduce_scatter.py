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

"""Eager ``dist.reduce_scatter_tensor`` / ``dist.reduce_scatter_single``, the
list ``dist.reduce_scatter`` and the coalesced form on the spyreccl backend (not
the compiled path; see ``test_reduce_scatter_compiled.py``)."""

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

from utils import _collective_rank_input

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
class TestReduceScatter(TestCase):
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

    # Per-rank chunk shapes. (64,) is one stick; (1, 64) has A == 1, so the
    # staging permute is trivial; (4, 128) has a last dim of two sticks, so its
    # default device layout tiles dim 1 outside dim 0; (3, 104) pads every
    # row; (2, 3, 64) is 3-D, so the result is copied into the output. A naive
    # flat-slab split gets the last three wrong.
    CHUNKS = [(64,), (1, 64), (4, 128), (3, 104), (2, 3, 64)]

    def _input(self, rank, chunk, dtype):
        shape = (self.comm_size * chunk[0], *chunk[1:])
        return _collective_rank_input(rank, shape, dtype)

    def _expected(self, chunk, dtype):
        total = torch.stack(
            [self._input(r, chunk, dtype) for r in range(self.comm_size)]
        ).sum(dim=0)
        return total.chunk(self.comm_size)[self.comm_rank]

    @parametrize("chunk", CHUNKS)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("async_op", [False, True])
    def test_reduce_scatter_tensor(self, chunk, dtype, async_op):
        x = self._input(self.comm_rank, chunk, dtype).to(DEVICE)
        out = torch.zeros(chunk, dtype=dtype, device=DEVICE)
        work = dist.reduce_scatter_tensor(out, x, async_op=async_op)
        self._finish(work, async_op)
        self.assertEqual(out.cpu(), self._expected(chunk, dtype), rtol=0, atol=0)
        # spyre-comms accumulates in its input buffer; the caller's input must
        # not be modified.
        self.assertEqual(
            x.cpu(), self._input(self.comm_rank, chunk, dtype), rtol=0, atol=0
        )

    @parametrize("chunk", [(64,), (3, 104)])
    @parametrize("async_op", [False, True])
    def test_reduce_scatter_single(self, chunk, async_op):
        x = self._input(self.comm_rank, chunk, torch.float16).to(DEVICE)
        out = torch.zeros(chunk, dtype=torch.float16, device=DEVICE)
        work = dist.reduce_scatter_single(out, x, async_op=async_op)
        self._finish(work, async_op)
        self.assertEqual(
            out.cpu(), self._expected(chunk, torch.float16), rtol=0, atol=0
        )

    @parametrize("chunk", [(64,), (4, 128), (3, 104)])
    @parametrize("async_op", [False, True])
    def test_reduce_scatter_list(self, chunk, async_op):
        x = self._input(self.comm_rank, chunk, torch.float16)
        inputs = [c.contiguous().to(DEVICE) for c in x.chunk(self.comm_size)]
        out = torch.zeros(chunk, dtype=torch.float16, device=DEVICE)
        work = dist.reduce_scatter(out, inputs, async_op=async_op)
        self._finish(work, async_op)
        self.assertEqual(
            out.cpu(), self._expected(chunk, torch.float16), rtol=0, atol=0
        )

    @parametrize("chunk", [(64,), (4, 128)])
    def test_functional_reduce_scatter_tensor(self, chunk):
        x = self._input(self.comm_rank, chunk, torch.float16).to(DEVICE)
        out = torch.ops._c10d_functional.reduce_scatter_tensor(
            x, "sum", self.comm_size, dist.group.WORLD.group_name
        )
        out = torch.ops._c10d_functional.wait_tensor(out)
        self.assertEqual(
            out.cpu(), self._expected(chunk, torch.float16), rtol=0, atol=0
        )

    @parametrize("async_op", [False, True])
    def test_coalescing_manager(self, async_op):
        chunks = [(64,), (4, 128), (3, 104)]
        inputs = [
            self._input(self.comm_rank, c, torch.float16).to(DEVICE) for c in chunks
        ]
        outputs = [torch.zeros(c, dtype=torch.float16, device=DEVICE) for c in chunks]
        with dist.distributed_c10d._coalescing_manager(async_ops=async_op) as cm:
            for out, x in zip(outputs, inputs):
                dist.reduce_scatter_tensor(out, x)
        if async_op:
            cm.wait()
        for out, c in zip(outputs, chunks):
            self.assertEqual(
                out.cpu(), self._expected(c, torch.float16), rtol=0, atol=0
            )

    def test_rejects_unaligned_1d_chunk(self):
        x = torch.zeros(self.comm_size * 32, dtype=torch.float16, device=DEVICE)
        out = torch.zeros(32, dtype=torch.float16, device=DEVICE)
        with self.assertRaisesRegex(RuntimeError, "must be a multiple of the stick"):
            dist.reduce_scatter_tensor(out, x)

    def test_rejects_non_sum(self):
        x = torch.zeros(self.comm_size * 64, dtype=torch.float16, device=DEVICE)
        out = torch.zeros(64, dtype=torch.float16, device=DEVICE)
        with self.assertRaisesRegex(RuntimeError, "only supports SUM operation"):
            dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.MAX)

    def test_rejects_float32(self):
        x = torch.zeros(self.comm_size * 64, dtype=torch.float32, device=DEVICE)
        out = torch.zeros(64, dtype=torch.float32, device=DEVICE)
        with self.assertRaisesRegex(RuntimeError, "only supports float16 and bfloat16"):
            dist.reduce_scatter_tensor(out, x)


if __name__ == "__main__":
    run_tests()
