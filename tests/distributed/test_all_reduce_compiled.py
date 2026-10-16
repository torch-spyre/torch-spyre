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

"""Tests for compiled allreduce (plan/run two-phase pattern).

Verifies that _c10d_functional.all_reduce is lowered to allreduce_plan
(compile-time) + allreduce_run (runtime) and produces correct results.

Usage:
    torchrun --nproc-per-node 2 tests/distributed/test_all_reduce_compiled.py
"""

import os

import pytest
import regex as re
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_spyre  # noqa: F401
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.wsr import for_each_tile

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


class AllReduceModule(torch.nn.Module):
    """Module that performs allreduce using functional collective ops."""

    def __init__(self, group_name: str = _GROUP_NAME) -> None:
        super().__init__()
        self._group_name = group_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.ops._c10d_functional.all_reduce(x, "sum", self._group_name)
        return torch.ops._c10d_functional.wait_tensor(y)


class AllReduceWithComputeModule(torch.nn.Module):
    """Module with compute interleaved around allreduce."""

    def __init__(self, group_name: str = _GROUP_NAME) -> None:
        super().__init__()
        self._group_name = group_name

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_scaled = x * 2.0
        reduced = torch.ops._c10d_functional.all_reduce(
            x_scaled, "sum", self._group_name
        )
        z = y + 1.0
        result = torch.ops._c10d_functional.wait_tensor(reduced)
        return result + z


_CARRY_M, _CARRY_K, _CARRY_N = 64, 256, 128


def _split_k_then_allreduce(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A for_each_tile accumulator whose only post-loop use is an all_reduce."""

    def body(acc, ops):
        x_tile, y_tile = ops
        return acc + x_tile @ y_tile, None

    acc, _ = for_each_tile(
        body,
        (x, y),
        dims=(-1, 0),
        tile_size=64,
        init=torch.zeros(_CARRY_M, _CARRY_N, device=x.device, dtype=x.dtype),
    )
    reduced = torch.ops._c10d_functional.all_reduce(acc, "sum", _GROUP_NAME)
    return torch.ops._c10d_functional.wait_tensor(reduced) * 2.0


def _loop_update_allocations(source: str) -> list[str]:
    """Allocation kinds of the carry update, the first ``add`` in the LoopSpec."""
    loop = source[source.index("LoopSpec(") :]
    update = loop[loop.index("op='add'") :]
    next_op = update.find("OpSpec(")
    update = update if next_op < 0 else update[:next_op]
    return re.findall(r"allocation=\{'(\w+)'", update)


class TestAllReduceCompiled(TestCase):
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

    def test_allreduce_compiled_fp16(self):
        """Verify compiled allreduce produces correct sum across ranks."""
        x = torch.ones((128,), dtype=torch.float16, device=DEVICE)
        module = AllReduceModule()
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.dtype, torch.float16)
        expected = torch.full((128,), float(self.comm_size), dtype=torch.float16)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: allreduce result incorrect. "
            f"Expected {expected[0].item()}, got {result[0].to('cpu').item()}",
        )

    def test_allreduce_compiled_fp16_larger(self):
        """Verify compiled allreduce works with larger fp16 tensors."""
        x = torch.ones((256,), dtype=torch.float16, device=DEVICE)
        module = AllReduceModule()
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.shape, x.shape)
        expected = torch.full((256,), float(self.comm_size), dtype=torch.float16)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: allreduce fp16 result incorrect",
        )

    def test_allreduce_compiled_with_interleaved_compute(self):
        """Verify compute around compiled allreduce works correctly."""
        x = torch.ones((128,), dtype=torch.float16, device=DEVICE)
        y = torch.ones((128,), dtype=torch.float16, device=DEVICE)
        module = AllReduceWithComputeModule()
        compiled_module = torch.compile(module)
        result = compiled_module(x, y)

        # x_scaled = 2.0, allreduce(sum) = 2.0 * world_size, z = 2.0
        # result = (2.0 * world_size) + 2.0
        expected_val = 2.0 * self.comm_size + 2.0
        expected = torch.full((128,), expected_val, dtype=torch.float16)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: allreduce with compute incorrect. "
            f"Expected {expected_val}, got {result[0].to('cpu').item()}",
        )

    def test_allreduce_compiled_rank_scaled(self):
        """Verify allreduce with rank-dependent input values."""
        val = float(self.comm_rank + 1)
        x = torch.full((64,), val, dtype=torch.float16, device=DEVICE)
        module = AllReduceModule()
        compiled_module = torch.compile(module)
        result = compiled_module(x)

        # Sum of 1 + 2 + ... + world_size = world_size * (world_size + 1) / 2
        expected_val = self.comm_size * (self.comm_size + 1) / 2.0
        expected = torch.full((64,), expected_val, dtype=torch.float16)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: rank-scaled allreduce incorrect. "
            f"Expected {expected_val}, got {result[0].to('cpu').item()}",
        )

    def test_allreduce_compiled_sequential_same_shape(self):
        """Two sequential allreduce calls on the same tensor shape in one graph."""

        class DoubleAllReduceModule(torch.nn.Module):
            def __init__(self, group_name: str = _GROUP_NAME) -> None:
                super().__init__()
                self._group_name = group_name

            def forward(self, t, ind):
                y = t + t
                y_reduced = torch.ops._c10d_functional.all_reduce(
                    y, "sum", self._group_name
                )
                y_ready = torch.ops._c10d_functional.wait_tensor(y_reduced)
                y_reduced2 = torch.ops._c10d_functional.all_reduce(
                    y_ready, "sum", self._group_name
                )
                ind_result = ind * ind * ind
                y_ready2 = torch.ops._c10d_functional.wait_tensor(y_reduced2)
                return y_ready2 + ind_result

        t = torch.full(
            (128,), float(self.comm_rank + 1), dtype=torch.float16, device=DEVICE
        )
        ind = torch.full((128,), 10.0, dtype=torch.float16, device=DEVICE)

        module = DoubleAllReduceModule()
        compiled_module = torch.compile(module)
        result = compiled_module(t, ind)

        # t + t = 2*(rank+1)
        # first allreduce: sum of 2*(rank+1) for all ranks = 2 * ws*(ws+1)/2
        first_ar = 2.0 * self.comm_size * (self.comm_size + 1) / 2.0
        # second allreduce: first_ar is the same on all ranks, so sum = first_ar * ws
        second_ar = first_ar * self.comm_size
        # ind_result = 10^3 = 1000
        expected_val = second_ar + 1000.0
        expected = torch.full((128,), expected_val, dtype=torch.float16)
        self.assertTrue(
            torch.allclose(result.to("cpu"), expected),
            f"Rank {self.comm_rank}: sequential allreduce incorrect. "
            f"Expected {expected_val}, got {result[0].to('cpu').item()}",
        )

    def test_allreduce_compiled_repeated_execution(self):
        """Multiple same-shape allreduces in one graph, executed multiple times.

        This mimics the Granite TP pattern: a model with several allreduce
        calls (one per transformer layer) compiled and then run repeatedly
        (warmup + decode steps). Catches regressions where bundle artifacts
        are not preserved across repeated executeBundle calls on the same WSI.
        """

        class MultiLayerAllReduceModule(torch.nn.Module):
            def __init__(self, group_name: str = _GROUP_NAME) -> None:
                super().__init__()
                self._group_name = group_name

            def forward(self, x):
                # Simulate multiple TP layers each doing allreduce
                h = x * 2.0
                h = torch.ops._c10d_functional.all_reduce(h, "sum", self._group_name)
                h = torch.ops._c10d_functional.wait_tensor(h)

                h = h + 1.0
                h = torch.ops._c10d_functional.all_reduce(h, "sum", self._group_name)
                h = torch.ops._c10d_functional.wait_tensor(h)

                h = h * 0.5
                h = torch.ops._c10d_functional.all_reduce(h, "sum", self._group_name)
                h = torch.ops._c10d_functional.wait_tensor(h)
                return h

        x = torch.ones((128,), dtype=torch.float16, device=DEVICE)
        module = MultiLayerAllReduceModule()
        compiled_module = torch.compile(module)

        ws = float(self.comm_size)
        # layer 1: 1*2=2, allreduce -> 2*ws
        # layer 2: 2*ws+1, allreduce -> (2*ws+1)*ws
        # layer 3: (2*ws+1)*ws*0.5, allreduce -> (2*ws+1)*ws*0.5*ws
        expected_val = (2.0 * ws + 1.0) * ws * 0.5 * ws
        expected = torch.full((128,), expected_val, dtype=torch.float16)

        # Run multiple times — simulates warmup + decode iterations
        for iteration in range(4):
            result = compiled_module(x)
            self.assertTrue(
                torch.allclose(result.to("cpu"), expected),
                f"Rank {self.comm_rank}: iteration {iteration} incorrect. "
                f"Expected {expected_val}, got {result[0].to('cpu').item()}",
            )

    def test_allreduce_of_loop_carry_keeps_carry_in_lx(self):
        """A loop accumulator reduced across ranks stays in LX while it accumulates.

        The collective needs an HBM operand. Without a post-loop drain the carry
        itself was that operand, so every trip read and wrote it in HBM.
        0/1 operands keep every partial sum exact in fp16.
        """
        g = torch.Generator().manual_seed(0)
        x = torch.randint(0, 2, (_CARRY_M, _CARRY_K), generator=g).half()
        y = torch.randint(0, 2, (_CARRY_K, _CARRY_N), generator=g).half()
        scale = self.comm_rank + 1
        rank_sum = self.comm_size * (self.comm_size + 1) / 2
        expected = (x.float() @ y.float()) * rank_sum * 2.0

        compiled = torch.compile(_split_k_then_allreduce, fullgraph=True)
        x_dev, y_dev = (x * scale).to(DEVICE), y.to(DEVICE)
        with spyre_config.patch(
            {"lx_planning": True, "co_optimizing_lx_planning": True}
        ):
            first, sources = run_and_get_code(compiled, x_dev, y_dev)
            second = compiled(x_dev, y_dev)

        for call, result in (("first", first), ("second", second)):
            torch.testing.assert_close(
                result.cpu().float(),
                expected,
                atol=0,
                rtol=0,
                msg=f"Rank {self.comm_rank}: {call} call incorrect",
            )
        source = "\n".join(sources)
        self.assertIn("allreduce_run(", source)
        self.assertEqual(_loop_update_allocations(source), ["lx", "lx", "lx"])


if __name__ == "__main__":
    run_tests()
