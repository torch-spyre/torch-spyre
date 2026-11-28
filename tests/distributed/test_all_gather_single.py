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

"""Eager ``dist.all_gather_into_tensor`` / ``dist.all_gather_single`` and the
coalesced all_gather forms on the spyreccl backend (not the compiled path; see
``test_all_gather_compiled.py``)."""

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
class TestAllGatherSingle(TestCase):
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

    # (64,) is one stick per rank. (4, 128) has a last dim of two sticks, so
    # its device layout tiles dim 1 outside dim 0; (3, 100) pads every row;
    # (2, 3, 64) is 3-D. A naive flat-slab gather gets the last three wrong.
    SHAPES = [(64,), (4, 128), (3, 100), (2, 3, 64)]

    def _expected(self, shape, dtype):
        return torch.cat(
            [_unique_rank_input(r, shape, dtype) for r in range(self.comm_size)]
        )

    def _output(self, shape, dtype):
        return torch.zeros(
            (self.comm_size * shape[0], *shape[1:]), dtype=dtype, device=DEVICE
        )

    @parametrize("shape", SHAPES)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("async_op", [False, True])
    def test_all_gather_into_tensor(self, shape, dtype, async_op):
        x = _unique_rank_input(self.comm_rank, shape, dtype).to(DEVICE)
        out = self._output(shape, dtype)
        work = dist.all_gather_into_tensor(out, x, async_op=async_op)
        self._finish(work, async_op)
        self.assertEqual(out.cpu(), self._expected(shape, dtype), rtol=0, atol=0)
        # The input is not modified.
        self.assertEqual(
            x.cpu(), _unique_rank_input(self.comm_rank, shape, dtype), rtol=0, atol=0
        )

    @parametrize("shape", [(64,), (3, 100)])
    @parametrize("async_op", [False, True])
    def test_all_gather_single(self, shape, async_op):
        x = _unique_rank_input(self.comm_rank, shape, torch.float16).to(DEVICE)
        out = self._output(shape, torch.float16)
        work = dist.all_gather_single(out, x, async_op=async_op)
        self._finish(work, async_op)
        self.assertEqual(
            out.cpu(), self._expected(shape, torch.float16), rtol=0, atol=0
        )

    @parametrize("shape", [(64,), (4, 128)])
    def test_functional_all_gather_into_tensor(self, shape):
        x = _unique_rank_input(self.comm_rank, shape, torch.float16).to(DEVICE)
        out = torch.ops._c10d_functional.all_gather_into_tensor(
            x, self.comm_size, dist.group.WORLD.group_name
        )
        out = torch.ops._c10d_functional.wait_tensor(out)
        self.assertEqual(
            out.cpu(), self._expected(shape, torch.float16), rtol=0, atol=0
        )

    @parametrize("async_op", [False, True])
    def test_coalescing_manager(self, async_op):
        shapes = [(64,), (4, 128), (3, 100)]
        inputs = [
            _unique_rank_input(self.comm_rank, s, torch.float16).to(DEVICE)
            for s in shapes
        ]
        outputs = [self._output(s, torch.float16) for s in shapes]
        with dist.distributed_c10d._coalescing_manager(async_ops=async_op) as cm:
            for out, x in zip(outputs, inputs):
                dist.all_gather_into_tensor(out, x)
        if async_op:
            cm.wait()
        for out, s in zip(outputs, shapes):
            self.assertEqual(
                out.cpu(), self._expected(s, torch.float16), rtol=0, atol=0
            )

    @parametrize("async_op", [False, True])
    def test_all_gather_coalesced(self, async_op):
        shapes = [(64,), (4, 128)]
        inputs = [
            _unique_rank_input(self.comm_rank, s, torch.float16).to(DEVICE)
            for s in shapes
        ]
        outputs = [
            [
                torch.zeros(s, dtype=torch.float16, device=DEVICE)
                for _ in range(self.comm_size)
            ]
            for s in shapes
        ]
        work = dist.all_gather_coalesced(outputs, inputs, async_op=async_op)
        if async_op:
            # This deprecated API returns the work's future, not the work.
            self.assertIsInstance(work, torch._C.Future)
            work.wait()
        else:
            self.assertIsNone(work)
        for outs, s in zip(outputs, shapes):
            for r, out in enumerate(outs):
                self.assertEqual(
                    out.cpu(),
                    _unique_rank_input(r, s, torch.float16),
                    rtol=0,
                    atol=0,
                )

    def test_get_future(self):
        # The future completes without a call to work.wait(), and only after
        # the output has been assembled from the per-rank buffers.
        shape = (3, 100)
        x = _unique_rank_input(self.comm_rank, shape, torch.float16).to(DEVICE)
        out = self._output(shape, torch.float16)
        work = dist.all_gather_into_tensor(out, x, async_op=True)
        work.get_future().wait()
        self.assertEqual(
            out.cpu(), self._expected(shape, torch.float16), rtol=0, atol=0
        )
        work.wait()
        self.assertTrue(work.is_completed())

    def test_rejects_wrong_output_size(self):
        x = torch.zeros(64, dtype=torch.float16, device=DEVICE)
        out = torch.zeros(64 * (self.comm_size + 1), dtype=torch.float16, device=DEVICE)
        with self.assertRaisesRegex(RuntimeError, "all_gather_single: expected"):
            dist.all_gather_into_tensor(out, x)


if __name__ == "__main__":
    run_tests()
