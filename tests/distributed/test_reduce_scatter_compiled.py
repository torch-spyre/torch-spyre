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

"""Tests for compiled reduce_scatter_tensor.

Verifies that _c10d_functional.reduce_scatter_tensor is lowered into
reducescatter_plan (compile-time) + reducescatter_run (runtime) calls
and produces correct results.

Usage:
    torchrun --nproc-per-node 2 tests/distributed/test_reduce_scatter_compiled.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch._inductor.exc import InductorError
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from inductor.utils_inductor import expects_raise

import torch_spyre  # noqa: F401

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
        "WORLD_SIZE environment variable is not a valid integer, "
        "skipping distributed tests",
        allow_module_level=True,
    )

DEVICE = torch.device(f"spyre:{os.getenv('RANK', '0')}")
C10D_BACKEND = "spyreccl"
_GROUP_NAME = "default"


class ReduceScatterModule(torch.nn.Module):
    """Module that performs reduce_scatter_tensor on a single tensor."""

    def __init__(self, group_size: int, group_name: str = _GROUP_NAME) -> None:
        super().__init__()
        self._group_size = group_size
        self._group_name = group_name

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        result = torch.ops._c10d_functional.reduce_scatter_tensor(
            tensor, "sum", self._group_size, self._group_name
        )
        return torch.ops._c10d_functional.wait_tensor(result)


def _position_input(shape: tuple[int, ...], rank: int) -> torch.Tensor:
    """Position-dependent input: (arange % 509) + rank.

    Distinct values per position expose a permuted or misplaced chunk, which
    a constant fill cannot.  Every partial sum stays below 2048, so the values
    are exact in fp16 and through DL16.
    """
    numel = 1
    for s in shape:
        numel *= s
    return ((torch.arange(numel) % 509).to(torch.float16) + rank).reshape(shape)


def _rs_reference(shape: tuple[int, ...], world: int, rank: int) -> torch.Tensor:
    full = sum(_position_input(shape, r).float() for r in range(world))
    return full.chunk(world, 0)[rank].half()


class TestReduceScatterCompiled(TestCase):
    @classmethod
    def setUpClass(cls):
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
        if dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        torch.compiler.reset()

    def test_reduce_scatter_compiled_basic(self):
        """Each rank contributes a full tensor; output is reduced scatter."""
        # Input: each rank has a (256,) tensor filled with (rank + 1).
        # reduce_scatter splits it into group_size chunks (whole sticks for up
        # to 4 ranks) and sums corresponding chunks across ranks, so every
        # rank gets sum(rank + 1) = ws*(ws+1)/2.
        x = torch.full(
            (256,), float(self.comm_rank + 1), dtype=torch.float16, device=DEVICE
        )

        module = ReduceScatterModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.shape[0], 256 // self.comm_size)

        expected_val = self.comm_size * (self.comm_size + 1) / 2.0
        expected = torch.full(
            (256 // self.comm_size,), expected_val, dtype=torch.float16
        )
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: reduce_scatter basic incorrect. "
            f"Expected {expected_val}, got {result[0].to('cpu').item()}",
        )

    def test_reduce_scatter_compiled_fp16(self):
        """Verify dtype and output shape preservation for fp16 inputs."""
        x = torch.ones((256,), dtype=torch.float16, device=DEVICE)

        module = ReduceScatterModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.shape[0], 256 // self.comm_size)

    def test_reduce_scatter_compiled_with_compute(self):
        """Compute interleaved around reduce_scatter."""

        class ComputeModule(torch.nn.Module):
            def __init__(self, group_size: int, group_name: str = _GROUP_NAME):
                super().__init__()
                self._group_size = group_size
                self._group_name = group_name

            def forward(self, x, bias):
                x_scaled = x * 2.0
                result = torch.ops._c10d_functional.reduce_scatter_tensor(
                    x_scaled, "sum", self._group_size, self._group_name
                )
                scattered = torch.ops._c10d_functional.wait_tensor(result)
                return scattered + bias

        x = torch.full(
            (256,), float(self.comm_rank + 1), dtype=torch.float16, device=DEVICE
        )
        bias = torch.ones((256 // self.comm_size,), dtype=torch.float16, device=DEVICE)

        module = ComputeModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x, bias)

        # x_scaled = 2*(rank+1), reduce_scatter sum -> 2*sum(rank+1), + bias=1
        rs_val = 2.0 * self.comm_size * (self.comm_size + 1) / 2.0
        expected_val = rs_val + 1.0
        expected = torch.full(
            (256 // self.comm_size,), expected_val, dtype=torch.float16
        )
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: reduce_scatter with compute incorrect. "
            f"Expected {expected_val}, got {result[0].to('cpu').item()}",
        )

    def test_reduce_scatter_compiled_repeated_execution(self):
        """WSI reuse: compiled reduce_scatter executed multiple times."""
        module = ReduceScatterModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)

        for iteration in range(4):
            x = torch.full(
                (256,),
                float(self.comm_rank + 1),
                dtype=torch.float16,
                device=DEVICE,
            )
            result = compiled_module(x)

            self.assertEqual(result.shape[0], 256 // self.comm_size)

            expected_val = self.comm_size * (self.comm_size + 1) / 2.0
            expected = torch.full(
                (256 // self.comm_size,), expected_val, dtype=torch.float16
            )
            self.assertTrue(
                torch.allclose(result.to("cpu"), expected),
                f"Rank {self.comm_rank}: iteration {iteration}, "
                f"reduce_scatter incorrect. "
                f"Expected {expected_val}, got {result[0].to('cpu').item()}",
            )

    def test_reduce_scatter_coalesced_compiled(self):
        """Coalesced reduce_scatter — multiple tensors in one call."""

        class CoalescedModule(torch.nn.Module):
            def __init__(self, group_size: int, group_name: str = _GROUP_NAME):
                super().__init__()
                self._group_size = group_size
                self._group_name = group_name

            def forward(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
                results = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
                    tensors, "sum", self._group_size, self._group_name
                )
                return [torch.ops._c10d_functional.wait_tensor(r) for r in results]

        a = torch.full(
            (256,),
            float(self.comm_rank + 1),
            dtype=torch.float16,
            device=DEVICE,
        )
        b = torch.full(
            (512,),
            float(self.comm_rank + 1),
            dtype=torch.float16,
            device=DEVICE,
        )

        module = CoalescedModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        r_a, r_b = compiled_module([a, b])

        self.assertEqual(r_a.shape[0], 256 // self.comm_size)
        self.assertEqual(r_b.shape[0], 512 // self.comm_size)

        expected_val = self.comm_size * (self.comm_size + 1) / 2.0
        expected_a = torch.full(
            (256 // self.comm_size,), expected_val, dtype=torch.float16
        )
        expected_b = torch.full(
            (512 // self.comm_size,), expected_val, dtype=torch.float16
        )
        self.assertTrue(
            torch.allclose(r_a.to("cpu"), expected_a),
            f"Rank {self.comm_rank}: coalesced tensor 0 incorrect. "
            f"Expected {expected_val}, got {r_a[0].to('cpu').item()}",
        )
        self.assertTrue(
            torch.allclose(r_b.to("cpu"), expected_b),
            f"Rank {self.comm_rank}: coalesced tensor 1 incorrect. "
            f"Expected {expected_val}, got {r_b[0].to('cpu').item()}",
        )

    def test_reduce_scatter_coalesced_compiled_multidim(self):
        """Coalesced reduce_scatter with 2-D inputs — only dim 0 is split."""
        # Two tensors of different 2-D shapes.  After reduce_scatter:
        #   (128, 32) → (128 // group_size, 32)
        #   (256, 16) → (256 // group_size, 16)
        # Each rank fills with (rank + 1); sum across ranks = ws*(ws+1)/2.

        class CoalescedMultidimModule(torch.nn.Module):
            def __init__(self, group_size: int, group_name: str = _GROUP_NAME):
                super().__init__()
                self._group_size = group_size
                self._group_name = group_name

            def forward(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
                results = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
                    tensors, "sum", self._group_size, self._group_name
                )
                return [torch.ops._c10d_functional.wait_tensor(r) for r in results]

        fill = float(self.comm_rank + 1)
        a = torch.full((128, 32), fill, dtype=torch.float16, device=DEVICE)
        b = torch.full((256, 16), fill, dtype=torch.float16, device=DEVICE)

        module = CoalescedMultidimModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        r_a, r_b = compiled_module([a, b])

        self.assertEqual(r_a.shape, torch.Size([128 // self.comm_size, 32]))
        self.assertEqual(r_b.shape, torch.Size([256 // self.comm_size, 16]))

        expected_val = self.comm_size * (self.comm_size + 1) / 2.0
        expected_a = torch.full(
            (128 // self.comm_size, 32), expected_val, dtype=torch.float16
        )
        expected_b = torch.full(
            (256 // self.comm_size, 16), expected_val, dtype=torch.float16
        )
        self.assertTrue(
            torch.allclose(r_a.to("cpu"), expected_a),
            f"Rank {self.comm_rank}: coalesced multidim tensor 0 incorrect. "
            f"Expected {expected_val}, got {r_a[0, 0].to('cpu').item()}",
        )
        self.assertTrue(
            torch.allclose(r_b.to("cpu"), expected_b),
            f"Rank {self.comm_rank}: coalesced multidim tensor 1 incorrect. "
            f"Expected {expected_val}, got {r_b[0, 0].to('cpu').item()}",
        )

    def test_reduce_scatter_compiled_multidim(self):
        """reduce_scatter on a 2-D input — only dim 0 is split by group_size."""
        # Input shape: (128, 64).  After reduce_scatter: (128 // group_size, 64).
        # Each rank fills with (rank + 1), so the sum across ranks is ws*(ws+1)/2.
        x = torch.full(
            (128, 64),
            float(self.comm_rank + 1),
            dtype=torch.float16,
            device=DEVICE,
        )

        module = ReduceScatterModule(group_size=self.comm_size)
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        expected_rows = 128 // self.comm_size
        self.assertEqual(result.shape, torch.Size([expected_rows, 64]))

        expected_val = self.comm_size * (self.comm_size + 1) / 2.0
        expected = torch.full((expected_rows, 64), expected_val, dtype=torch.float16)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: reduce_scatter multidim incorrect. "
            f"Expected {expected_val}, got {result[0, 0].to('cpu').item()}",
        )

    @parametrize(
        "shape",
        [
            (256,),
            (128, 64),
            (128, 128),
            (64, 256),
            (256, 4096),
            (4, 8, 64),
            (8, 4, 128),
        ],
        name_fn=lambda shape: "x".join(map(str, shape)),
    )
    def test_reduce_scatter_compiled_position(self, shape):
        """Each rank receives exactly its own dim-0 chunk of the sum (#4273).

        Shapes whose last dim spans several sticks used to receive a
        column-stick block instead of their rows.
        """
        x = _position_input(shape, self.comm_rank).to(DEVICE)
        result = torch.compile(ReduceScatterModule(group_size=self.comm_size))(x)

        expected = _rs_reference(shape, self.comm_size, self.comm_rank)
        self.assertEqual(result.shape, expected.shape)
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    def test_reduce_scatter_compiled_position_view_input(self):
        """A dim-0 slice input is materialized before the collective."""

        class SliceModule(torch.nn.Module):
            def __init__(self, group_size: int):
                super().__init__()
                self._group_size = group_size

            def forward(self, x):
                result = torch.ops._c10d_functional.reduce_scatter_tensor(
                    x[64:], "sum", self._group_size, _GROUP_NAME
                )
                return torch.ops._c10d_functional.wait_tensor(result)

        shape = (192, 128)
        x = _position_input(shape, self.comm_rank).to(DEVICE)
        result = torch.compile(SliceModule(self.comm_size))(x)

        full = sum(_position_input(shape, r).float() for r in range(self.comm_size))
        expected = full[64:].chunk(self.comm_size, 0)[self.comm_rank].half()
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    def test_reduce_scatter_coalesced_compiled_position(self):
        """Coalesced reduce_scatter places each tensor's chunks correctly."""

        class CoalescedModule(torch.nn.Module):
            def __init__(self, group_size: int):
                super().__init__()
                self._group_size = group_size

            def forward(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
                results = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
                    tensors, "sum", self._group_size, _GROUP_NAME
                )
                return [torch.ops._c10d_functional.wait_tensor(r) for r in results]

        shapes = [(128, 128), (8, 4, 128), (128, 64)]
        inputs = [_position_input(s, self.comm_rank).to(DEVICE) for s in shapes]
        results = torch.compile(CoalescedModule(self.comm_size))(inputs)

        for shape, result in zip(shapes, results):
            expected = _rs_reference(shape, self.comm_size, self.comm_rank)
            torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    @expects_raise
    def test_reduce_scatter_compiled_sub_stick_1d_rejected(self):
        """A 1-D per-rank chunk smaller than a stick raises at compile time.

        spyre-comms has no algorithm for a split below one 128-byte stick and
        terminates the process; the lowering must reject it first.
        TODO: support by padding the per-rank chunk to whole sticks.
        """
        x = _position_input((100,), self.comm_rank).to(DEVICE)
        with pytest.raises(InductorError, match="must be a multiple of the stick"):
            torch.compile(ReduceScatterModule(group_size=self.comm_size))(x)

    @expects_raise
    def test_reduce_scatter_compiled_sub_stick_per_rank_rejected(self):
        """32 elements per rank (e.g. (128,) at g=4) is rejected, for any g.

        This is the shape that terminated all ranks inside
        context->reduce_scatter at g=4 (#4273).
        """
        x = _position_input((32 * self.comm_size,), self.comm_rank).to(DEVICE)
        with pytest.raises(InductorError, match="must be a multiple of the stick"):
            torch.compile(ReduceScatterModule(group_size=self.comm_size))(x)

    @expects_raise
    @parametrize(
        "elems_per_rank",
        [32, 100],
        name_fn=lambda n: f"{n}_per_rank",
    )
    def test_reduce_scatter_plan_sub_stick_raises(self, elems_per_rank):
        """reducescatter_plan rejects sub-stick slabs before calling spyre-comms.

        This is the call that crashed: spyre-comms finds no algorithm for a
        split below one 128-byte stick and terminates the process from inside
        plan creation (DT_ERROR, no Python exception). The plan op must raise
        first. If this check regresses the test cannot fail cleanly -- every
        rank exits -- which still fails the run.
        """
        from torch_spyre._inductor.ir import _dtype_to_int

        num_elems = elems_per_rank * self.comm_size
        with pytest.raises(RuntimeError, match="slabs of whole sticks"):
            torch.ops.spyre.reducescatter_plan(
                num_elems,
                _dtype_to_int(torch.float16),
                self.comm_size,
                "sum",
                _GROUP_NAME,
            )

    @expects_raise
    def test_reduce_scatter_run_unstaged_input_raises(self):
        """reducescatter_run rejects an input that was not staged by the lowering.

        An unstaged (128, 128) input was the wrong-data path: spyre-comms gave
        each rank a column-stick block instead of its rows, with no error.
        """
        from torch_spyre._inductor.ir import _dtype_to_int

        x = _position_input((128, 128), self.comm_rank).to(DEVICE)
        # Plan for a valid staged buffer of the same size, so only the run-time
        # layout check can reject this call.
        staged_elems = 128 * 128
        plan = torch.ops.spyre.reducescatter_plan(
            staged_elems,
            _dtype_to_int(torch.float16),
            self.comm_size,
            "sum",
            _GROUP_NAME,
        )
        with pytest.raises(RuntimeError, match="expected a staged"):
            torch.ops.spyre.reducescatter_run(x, plan, self.comm_size)


instantiate_parametrized_tests(TestReduceScatterCompiled)

if __name__ == "__main__":
    run_tests()
