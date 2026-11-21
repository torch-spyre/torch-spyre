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

"""Eager tests for the spyre::alltoall_plan / spyre::alltoall_run ops.

These call the ops that compiled all_to_all lowers to directly, without
torch.compile, so a failure here is in the C++ op or in spyre-comms
(alltoallw), not in Inductor.  The plan takes split sizes in device elements of
the flat device buffer.  Each 2-D input here has rows exactly one stick wide
(128 bytes: 64 fp16 or 32 fp32 elements), the widest row whose default layout
keeps dim-0 rows contiguous on device; a wider row is split into stick tiles
stored outside dim 0, and a flat split would then mix rows.  1-D inputs are
always contiguous and are split in whole sticks.
"""

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

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"
_GROUP_NAME = "default"

_STICK_BYTES = 128


def _row_elems(dtype: torch.dtype) -> int:
    """Elements in one stick: the row width used for every input below."""
    return _STICK_BYTES // torch.tensor([], dtype=dtype).element_size()


def _dtype_code(dtype: torch.dtype) -> int:
    # Same encoding as torch_spyre._inductor.ir._dtype_to_int.
    from torch_spyre._inductor.ir import _dtype_to_int

    return _dtype_to_int(dtype)


def _even_rows(world: int, rows: int) -> list[list[int]]:
    """pattern[s][d]: rows rank s sends to rank d."""
    return [[rows] * world for _ in range(world)]


def _uneven_rows(world: int) -> list[list[int]]:
    """An uneven but cross-rank-consistent pattern: s sends (s + d) % W + 1."""
    return [[(s + d) % world + 1 for d in range(world)] for s in range(world)]


def _zero_send_rows(world: int) -> list[list[int]]:
    """Every rank sends nothing to the next rank; other peers get 2 rows."""
    return [
        [0 if d == (s + 1) % world else 2 for d in range(world)] for s in range(world)
    ]


_PATTERNS = {
    "even": lambda w: _even_rows(w, 2),
    "uneven": _uneven_rows,
}


def _tag(src: int, dst: int, row: int) -> float:
    """Value written into row ``row`` of the slice src sends to dst.

    At most 7 * 64 + 7 * 8 + 8 = 512 for up to 8 ranks (rows per slice are
    at most the world size), well inside fp16's exact-integer range.
    """
    return float(src * 64 + dst * 8 + row)


@instantiate_parametrized_tests
class TestAllToAllOps(TestCase):
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

        # alltoall_plan gathers every rank's splits over this group.
        c10d._register_process_group(_GROUP_NAME, dist.group.WORLD)

        cls.comm_size = dist.get_world_size()
        cls.comm_rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        """Clean up the distributed environment after all tests."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _make_input(self, send_rows: list[int], dtype, one_d: bool) -> torch.Tensor:
        """Concatenate, by destination, the row-tagged slices this rank sends."""
        rows = []
        for dst, n in enumerate(send_rows):
            for r in range(n):
                rows.append(
                    torch.full(
                        (_row_elems(dtype),), _tag(self.comm_rank, dst, r), dtype=dtype
                    )
                )
        x = torch.stack(rows)
        return x.reshape(-1) if one_d else x

    def _expected(self, recv_rows: list[int], dtype, one_d: bool) -> torch.Tensor:
        """Concatenate, by source, the slices every peer sends to this rank."""
        rows = []
        for src, n in enumerate(recv_rows):
            for r in range(n):
                rows.append(
                    torch.full(
                        (_row_elems(dtype),), _tag(src, self.comm_rank, r), dtype=dtype
                    )
                )
        x = torch.stack(rows)
        return x.reshape(-1) if one_d else x

    def _run(self, pattern, dtype=torch.float16, one_d=False, iterations=1):
        me = self.comm_rank
        send_rows = pattern[me]
        recv_rows = [pattern[src][me] for src in range(self.comm_size)]
        row = _row_elems(dtype)
        # A 1-D tensor's dim 0 counts elements, so each row is `row` of it.
        unit = row if one_d else 1

        handle = torch.ops.spyre.alltoall_plan(
            [n * row for n in recv_rows],
            [n * row for n in send_rows],
            _dtype_code(dtype),
            _GROUP_NAME,
        )
        x_cpu = self._make_input(send_rows, dtype, one_d)
        expected = self._expected(recv_rows, dtype, one_d)
        output_dim0 = sum(recv_rows) * unit
        self.assertEqual(x_cpu.shape[0], sum(send_rows) * unit)

        x = x_cpu.to(DEVICE)
        for _ in range(iterations):
            out = torch.ops.spyre.alltoall_run(x, handle, output_dim0)
            out = torch.ops.spyre.wait_work(out)
            out_cpu = out.to("cpu")
            self.assertEqual(out_cpu.shape, expected.shape)
            self.assertTrue(
                torch.equal(out_cpu, expected),
                f"Rank {me}: alltoall output mismatch for pattern {pattern}; "
                f"first rows got {out_cpu.reshape(-1, row)[:, 0].tolist()}, "
                f"expected {expected.reshape(-1, row)[:, 0].tolist()}",
            )
        # all_to_all is out of place: the input must be unchanged.
        self.assertTrue(torch.equal(x.to("cpu"), x_cpu))
        return handle

    @parametrize("pattern", list(_PATTERNS))
    @parametrize("dtype", [torch.float16, torch.float32])
    def test_alltoall_ops(self, pattern, dtype):
        """Rows land at the right destination, in source-rank order."""
        self._run(_PATTERNS[pattern](self.comm_size), dtype=dtype)

    @parametrize("pattern", ["even", "uneven"])
    @parametrize("dtype", [torch.float16, torch.float32])
    def test_alltoall_ops_1d(self, pattern, dtype):
        """1-D tensors split by element, in whole sticks."""
        self._run(_PATTERNS[pattern](self.comm_size), dtype=dtype, one_d=True)

    def test_alltoall_ops_plan_reuse(self):
        """One plan applied repeatedly (the compiled path's WSI reuse)."""
        self._run(_uneven_rows(self.comm_size), iterations=4)

    def test_alltoall_ops_plan_is_cached(self):
        """Identical splits share a plan handle; different splits do not."""
        pattern = _uneven_rows(self.comm_size)
        first = self._run(pattern)
        self.assertEqual(self._run(pattern), first)
        self.assertNotEqual(self._run(_even_rows(self.comm_size, 2)), first)

    def test_alltoall_ops_cache_keys_on_whole_pattern(self):
        """Plans are cached by the whole exchange, not this rank's splits.

        Only the last rank's sends change between the two patterns, so rank 0
        has the same local splits both times.  Rank 0 must still build a new
        plan with everyone else: building a plan is collective, so if rank 0
        reused its cached plan the other ranks would wait for it forever.
        """
        world = self.comm_size
        first = [[1] * world for _ in range(world)]
        second = [row[:] for row in first]
        second[world - 1][world - 1] = 2  # the last rank keeps an extra row
        h1 = self._run(first)
        h2 = self._run(second)
        self.assertNotEqual(h1, h2)
        # Each pattern maps back to its own plan, on every rank.
        self.assertEqual(self._run(first), h1)
        self.assertEqual(self._run(second), h2)

    @expects_raise
    def test_alltoall_ops_rejects_zero_split(self):
        """A zero-size split is rejected before it reaches spyre-comms.

        TODO: spyre-comms alltoallw segfaults in Collective::Convert when a
        plan with a zero split is applied.  When it handles them, make this a
        positive test (rows from _zero_send_rows) and drop the C++ check.
        """
        pattern = _zero_send_rows(self.comm_size)
        me = self.comm_rank
        row = _row_elems(torch.float16)
        recv = [pattern[src][me] * row for src in range(self.comm_size)]
        send = [n * row for n in pattern[me]]
        with pytest.raises(RuntimeError, match="zero-size .* splits are not supported"):
            torch.ops.spyre.alltoall_plan(
                recv, send, _dtype_code(torch.float16), _GROUP_NAME
            )

    @expects_raise
    def test_alltoall_ops_rejects_mismatched_peers(self):
        """Ranks that disagree on the splits all get an error, not a hang.

        Rank 0 expects one more stick from rank 1 than rank 1 sends.  Every
        rank sees the whole gathered pattern, so every rank raises.
        """
        world = self.comm_size
        row = _row_elems(torch.float16)
        recv = [row] * world
        send = [row] * world
        if self.comm_rank == 0:
            recv[1] += row
        with pytest.raises(RuntimeError, match="rank 1 sends .* but rank 0 expects"):
            torch.ops.spyre.alltoall_plan(
                recv, send, _dtype_code(torch.float16), _GROUP_NAME
            )

    @expects_raise
    def test_alltoall_ops_rejects_wrong_split_count(self):
        """The plan needs exactly one split per rank."""
        splits = [_row_elems(torch.float16)] * (self.comm_size + 1)
        with pytest.raises(RuntimeError, match="number of splits must equal"):
            torch.ops.spyre.alltoall_plan(
                splits, splits, _dtype_code(torch.float16), _GROUP_NAME
            )

    @expects_raise
    def test_alltoall_ops_rejects_input_not_matching_plan(self):
        """alltoall_run checks the input's device size against the plan."""
        row = _row_elems(torch.float16)
        splits = [2 * row] * self.comm_size
        handle = torch.ops.spyre.alltoall_plan(
            splits, splits, _dtype_code(torch.float16), _GROUP_NAME
        )
        x = torch.zeros((self.comm_size, row), dtype=torch.float16).to(DEVICE)
        with pytest.raises(RuntimeError, match="the plan expects"):
            torch.ops.spyre.alltoall_run(x, handle, 2 * self.comm_size)


if __name__ == "__main__":
    run_tests()
