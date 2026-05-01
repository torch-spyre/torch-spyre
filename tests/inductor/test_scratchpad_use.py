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

from collections.abc import Sequence
from contextlib import contextmanager
import functools
from typing import Any, Callable, TypeVarTuple, Unpack, Optional, override

import unittest
from unittest.mock import patch
import torch

from torch._inductor.virtualized import V
from torch._inductor import config as t_inductor_config
from torch._inductor.ir import Operation

from torch_spyre._inductor.scratchpad import mem_usage_by_op
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
    def __call__(self, operations: list[Operation]) -> None:
        assert self.test_instance is not None, (
            "CustomPreSchedulingPassesWithOurPasses.test_instance must be set to an instance of "
            "TestScratchpadUsage before get_passes is called"
        )
        super().__call__(operations)
        for f in self.test_instance.our_pre_scheduling_passes:
            f(operations)


class TestScratchpadUsage(unittest.TestCase):
    our_pre_scheduling_passes: list[Callable[[list[Operation]], None]] = []

    def setUp(self):
        torch.manual_seed(0xAFFE)
        self.patchers = []

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
        f: Callable[[Operation], None],
    ):
        """Context manager to add a post fusion custom pass that processes each node independently
        using `f`."""

        def new_pass(nodes: list[Operation]) -> None:
            for node in nodes:
                f(node)

        self.our_pre_scheduling_passes.append(new_pass)
        yield
        self.our_pre_scheduling_passes.remove(new_pass)

    def compile_and_collect_mem_usage(
        self, f: Callable[[Unpack[Ts]], torch.Tensor], args: tuple[Unpack[Ts]]
    ) -> tuple[torch.Tensor, list[dict[str, dict[str, Any]]]]:
        mem_usages = []

        def visitor(node: Operation) -> None:
            nonlocal mem_usages
            mem_usage = mem_usage_by_op(node)
            mem_usage = {
                key: value
                for key, value in mem_usage.items()
                if isinstance(value, dict)
            }
            for buffer_name, usage in mem_usage.items():
                buffer = V.graph.get_buffer(buffer_name)
                layout = buffer.get_layout()
                allocation = getattr(layout, "allocation", {})
                usage["location"] = "LX" if "lx" in allocation else "HBM"

            mem_usages.append(mem_usage)

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
            any(
                usage["location"] == "LX"
                for mem_usage in mem_usages
                for usage in mem_usage.values()
            ),
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
            usage["size"]
            for mem_usage in mem_usages
            for usage in mem_usage.values()
            if usage["location"] == "HBM"
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


if __name__ == "__main__":
    unittest.main()
