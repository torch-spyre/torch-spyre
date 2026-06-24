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

import math
from collections.abc import Sequence
from contextlib import contextmanager
import functools
from typing import Callable, TypeVarTuple, Unpack, Optional, override

import unittest
from unittest.mock import patch
import torch

from torch._inductor import config as t_inductor_config
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.passes import CustomPreSchedulingPasses
from torch_spyre._inductor import passes
from torch_spyre._inductor import config as ts_inductor_config


Ts = TypeVarTuple("Ts")


class CustomPreSchedulingPassesWithOurPasses(CustomPreSchedulingPasses):
    """torch_spyre._inductor.patches.enable_spyre_context sets
    torch._inductor.config._post_fusion_custom_pass to
    torch_spyre._inductor.passes.CustomPostFusionPasses(), so we have to monkey patch that class
    to add the ability to add custom passes."""

    test_instance: Optional["TestScratchpadUsage"] = None

    @classmethod
    def initialize(cls, test_instance: "TestScratchpadUsage"):
        cls.test_instance = test_instance

    @override
    def __call__(self, graph: GraphLowering) -> None:
        assert self.test_instance is not None, (
            "CustomPreSchedulingPassesWithOurPasses.test_instance must be set to an instance of "
            "TestScratchpadUsage before get_passes is called"
        )
        super().__call__(graph)
        for f in self.test_instance.our_pre_scheduling_passes:
            f(graph)


class TestScratchpadUsage(unittest.TestCase):
    our_pre_scheduling_passes: list[Callable[[GraphLowering], None]] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patchers = []

    def setUp(self):
        torch.manual_seed(0xAFFE)

        self.patchers.append(t_inductor_config.patch("force_disable_caches", True))
        self.patchers.append(ts_inductor_config.patch("sencores", 1))

        CustomPreSchedulingPassesWithOurPasses.initialize(self)
        self.patchers.append(
            patch.object(
                passes,
                "CustomPreSchedulingPasses",
                CustomPreSchedulingPassesWithOurPasses,
            )
        )

        for p in self.patchers:
            p.__enter__()

        torch.compiler.reset()

    def tearDown(self):
        for p in self.patchers:
            p.__exit__(None, None, None)

        torch.compiler.reset()

    def rand_device(self, shape: Sequence[int]):
        result = torch.rand(shape, dtype=torch.float16, device="spyre")
        return result

    @contextmanager
    def pre_scheduling_iterating_pass(
        self,
        f: Callable[[GraphLowering], None],
    ):
        """Context manager to add a post fusion custom pass that processes each node independently
        using `f`."""

        def new_pass(graph: GraphLowering) -> None:
            f(graph)

        self.our_pre_scheduling_passes.append(new_pass)
        yield
        self.our_pre_scheduling_passes.remove(new_pass)

    def compile_and_collect_mem_usage(
        self, f: Callable[[Unpack[Ts]], torch.Tensor], args: tuple[Unpack[Ts]]
    ) -> tuple[torch.Tensor, dict[str, str]]:
        mem_usages = {}

        def visitor(graph: GraphLowering) -> None:
            nonlocal mem_usages
            operations = graph.operations
            for op in operations:
                buf_name = op.name
                buffer = graph.get_buffer(buf_name)
                layout = buffer.get_layout()
                device_layout = layout.device_layout
                allocation = getattr(layout, "allocation", {})
                mem_usages[buf_name] = {
                    "location": "LX" if "lx" in allocation else "HBM",
                    "size": math.prod(device_layout.device_size[:-1]) * 128,
                }

        with self.pre_scheduling_iterating_pass(visitor):
            compiled_kernel = torch.compile(f, fullgraph=True)
            result = compiled_kernel(*args).to("cpu")

        return (result, mem_usages)

    def run_test(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
        **kwargs,
    ):
        """Run the current class's test procedure on the given model and arguments. Override this
        in each subclass."""
        cpu_result = model(*(t.to("cpu") for t in args))

        with ts_inductor_config.patch(lx_planning=True):
            device_result, mem_usages = self.compile_and_collect_mem_usage(model, args)

        self.assertTrue(
            any(mem_usage["location"] == "LX" for mem_usage in mem_usages.values()),
            "Expected at least one buffer to be allocated in LX, but none were",
        )

        atol = kwargs.get("atol", 1e-4)
        self.assertTrue(
            torch.allclose(cpu_result, device_result, atol=atol), "Results do not match"
        )

    def common(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
        **kwargs,
    ):
        """This method runs some sanity checks common to all subclasses and then calls
        `run_test`."""
        for t in args:
            self.assertIsInstance(t, torch.Tensor)
            self.assertEqual(t.device.type, "spyre")
        return self.run_test(model, args, **kwargs)

    def test_softmax(self):
        f = functools.partial(torch.softmax, dim=0)
        x = self.rand_device((512, 1024))
        self.common(f, (x,))


class TestMeasureHBMUsageScratchPad(TestScratchpadUsage):
    def measure_hbm_transfers(
        self, model: Callable[[Unpack[Ts]], torch.Tensor], args: tuple[Unpack[Ts]]
    ) -> tuple[torch.Tensor | None, int]:
        """Estimates the HBM transfers for a given operation. This assumes that any buffer that
        has an entry in its allocations that starts with "lx" is free and that any other node's HBM
        transfers are accurately returned by `mem_usage_by_node`."""
        result, mem_usages = self.compile_and_collect_mem_usage(model, args)
        hbm_transfers = sum(
            mem_usage["size"]
            for mem_usage in mem_usages.values()
            if mem_usage["location"] == "HBM"
        )
        return (result, hbm_transfers)

    @override
    def run_test(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
        **kwargs,
    ):
        """Test that estimates the total amount of HBM transfers with LX planning turned off and
        turned on, and then compares them."""
        with ts_inductor_config.patch(lx_planning=False):
            result_without_lx, hbm_without_lx = self.measure_hbm_transfers(model, args)

        with ts_inductor_config.patch(lx_planning=True):
            result_with_lx, hbm_with_lx = self.measure_hbm_transfers(model, args)

        self.assertLess(
            hbm_with_lx,
            hbm_without_lx,
            "Expected LX planning to reduce HBM transfers, but it did not",
        )
        self.assertTrue(
            torch.allclose(result_without_lx, result_with_lx, atol=1e-5),
            "Results do not match between LX planning on and off",
        )

    # TODO: Add additional ops


@unittest.skipUnless(
    ts_inductor_config.co_optimizing_lx_planning,
    "CO_OPTIMIZING_LX_PLANNING is off; skipping cooptimization tests",
)
class TestMeasureHBMUsageCoOptimizing(TestMeasureHBMUsageScratchPad):
    """Compares HBM transfers between DefaultAllocator and
    StrategyBCoOptimizingAllocator. The cooptimizing allocator should be ≤ default on every shape,
    and should strictly improve on cases where adjacent ops disagree on which
    iteration-space dim to split — the canonical example is softmax(dim=0)
    where work_distribution picks rows for the pointwise ops and cols for the
    reduction ops, forcing 3 of 4 shared buffers to HBM under DefaultAllocator.

    Skipped unless `CO_OPTIMIZING_LX_PLANNING=1` is set in the environment;
    otherwise the cooptimization code path doesn't activate and there's
    nothing to compare.
    """

    @override
    def setUp(self):
        super().setUp()
        # Cooptimization needs > 1 core to have anything to optimize.
        self.patchers.append(ts_inductor_config.patch("sencores", 4))
        self.patchers[-1].__enter__()

    @override
    def run_test(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
        strict: bool = False,
        **kwargs,
    ):
        """Compare HBM transfers with cooptimization off vs on. If
        `strict`, asserts coopt < default; otherwise coopt ≤ default."""
        with ts_inductor_config.patch(lx_planning=True):
            with ts_inductor_config.patch(co_optimizing_lx_planning=False):
                result_default, hbm_default = self.measure_hbm_transfers(model, args)
            torch.compiler.reset()
            with ts_inductor_config.patch(co_optimizing_lx_planning=True):
                result_coopt, hbm_coopt = self.measure_hbm_transfers(model, args)

        cmp = self.assertLess if strict else self.assertLessEqual
        rel = "<" if strict else "≤"
        cmp(
            hbm_coopt,
            hbm_default,
            f"Expected cooptimization to be {rel} default HBM, got "
            f"coopt={hbm_coopt} default={hbm_default}",
        )
        self.assertTrue(
            torch.allclose(result_default, result_coopt, atol=1e-4),
            "Results do not match between cooptimization on and off",
        )

    def test_softmax_dim0_strictly_lower_hbm(self):
        """The canonical motivating case from the design doc. softmax(dim=0)
        has every adjacent op pair disagreeing on which dim to split, so
        DefaultAllocator only pins 1 of 4 shared buffers; Strategy B should
        flip the pointwise ops to cols and pin all 4 → strictly lower HBM."""
        f = functools.partial(torch.softmax, dim=0)
        x = self.rand_device((512, 1024))
        self.common(f, (x,), strict=True)

    def test_softmax_dim_neg1_no_regression(self):
        """softmax(dim=-1) is the well-behaved baseline where DefaultAllocator
        already pins everything pinnable. Strategy B must match (no regression)."""
        f = functools.partial(torch.softmax, dim=-1)
        x = self.rand_device((512, 1024))
        self.common(f, (x,))


class TestCloneAtGraphBoundaries(TestScratchpadUsage):
    """End-to-end tests for clone insertion at graph input/output boundaries.

    The allocator now inserts clone ops on-demand inside _push_allocation rather than
    as a separate pre-scheduling pass.  These tests verify that:
    - graph inputs read by multiple ops get a clone that lands in LX
    - graph outputs that are also read inside the graph get a clone (for the HBM return
      value), while the original buffer is pinned to LX

    Enabling ``lx_boundary_clones`` flips ``clone_at_graph_boundaries()`` on and
    makes the inserted clone outputs LX-eligible, so the boundary clone path is
    exercised.
    """

    def setUp(self):
        self.patchers.append(ts_inductor_config.patch("lx_boundary_clones", True))
        super().setUp()

    def _compile_and_inspect(
        self,
        f: Callable,
        args: tuple,
    ) -> tuple:
        """Compile f, capture op count and mem_usages after the allocator runs.

        Handles both single-tensor and tuple outputs.
        Returns (result_on_cpu, n_ops, mem_usages).
        """
        n_ops_captured: list[int] = []
        mem_usages: dict[str, dict] = {}

        def visitor(graph: GraphLowering) -> None:
            n_ops_captured.append(len(graph.operations))
            for op in graph.operations:
                buf_name = op.name
                buffer = graph.get_buffer(buf_name)
                layout = buffer.get_layout()
                device_layout = layout.device_layout
                allocation = getattr(layout, "allocation", {})
                mem_usages[buf_name] = {
                    "location": "LX" if "lx" in allocation else "HBM",
                    "size": math.prod(device_layout.device_size[:-1]) * 128,
                }

        with self.pre_scheduling_iterating_pass(visitor):
            compiled_kernel = torch.compile(f, fullgraph=True)
            raw = compiled_kernel(*args)
            if isinstance(raw, tuple):
                result = tuple(r.to("cpu") for r in raw)
            else:
                result = raw.to("cpu")

        n_ops = n_ops_captured[0] if n_ops_captured else 0
        return result, n_ops, mem_usages

    def test_input_clone_when_read_by_multiple_ops(self):
        """A graph input read by two different ops is cloned; the clone lands in LX."""
        x = self.rand_device((64, 1024))

        def fn(x):
            # x is consumed by both exp_op and add_op → two reads → eligible for input clone
            return torch.exp(x) + x

        with ts_inductor_config.patch(lx_planning=False):
            ref_result, n_ops_no_lx, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(lx_planning=True):
            result, n_ops_with_lx, mem_usages = self._compile_and_inspect(fn, (x,))

        self.assertGreater(
            n_ops_with_lx,
            n_ops_no_lx,
            f"Expected the input clone to add an op: {n_ops_no_lx} ops without LX, "
            f"{n_ops_with_lx} with LX",
        )
        self.assertTrue(
            any(u["location"] == "LX" for u in mem_usages.values()),
            "Expected at least one LX-allocated buffer after input cloning",
        )
        # Clone is an exact copy; LX planning must not change the numerical result.
        self.assertTrue(
            torch.equal(ref_result, result),
            "LX input clone changed the numerical result",
        )

    def test_output_clone_when_intermediate_is_also_graph_output(self):
        """A buffer that is both a graph output and read inside the graph is pinned to LX;
        a clone of it is inserted as the actual (HBM) graph output returned to the caller."""
        x = self.rand_device((64, 1024))

        def fn(x):
            # After CSE, y = exp(x) is produced once.
            # y is a graph output AND is read by add_op → eligible for output clone.
            y = torch.exp(x)
            z = y + 1  # add_op reads y
            return y, z

        with ts_inductor_config.patch(lx_planning=False):
            (ref_y, ref_z), n_ops_no_lx, _ = self._compile_and_inspect(fn, (x,))

        torch.compiler.reset()

        with ts_inductor_config.patch(lx_planning=True):
            (result_y, result_z), n_ops_with_lx, mem_usages = self._compile_and_inspect(
                fn, (x,)
            )

        self.assertGreater(
            n_ops_with_lx,
            n_ops_no_lx,
            f"Expected the output clone to add an op: {n_ops_no_lx} ops without LX, "
            f"{n_ops_with_lx} with LX",
        )
        self.assertTrue(
            any(u["location"] == "LX" for u in mem_usages.values()),
            "Expected at least one LX-allocated buffer after output cloning",
        )
        # Clone is an exact copy; LX planning must not change the numerical result.
        self.assertTrue(
            torch.equal(ref_y, result_y), "LX output clone changed result y"
        )
        self.assertTrue(
            torch.equal(ref_z, result_z), "LX output clone changed result z"
        )


if __name__ == "__main__":
    unittest.main()
