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
from typing import Any, Callable, TypeVarTuple, Unpack, Optional

import unittest
import torch

from torch._inductor.scheduler import BaseSchedulerNode, SchedulerNode
from torch._inductor.virtualized import V
from torch._inductor import config as t_inductor_config

from torch_spyre._inductor.scratchpad import mem_usage_by_node
from torch_spyre._inductor.passes import CustomNodePassBase, CustomPostFusionPasses
from torch_spyre._inductor import passes
from torch_spyre._inductor import config as ts_inductor_config

from tests.inductor.utils_inductor import cached_randn

Ts = TypeVarTuple("Ts")


class CustomPostFusionPassesWithOurPasses(CustomNodePassBase):
    test_instance = Optional["TestScratchpadUsage"]
    base_pass_list: list[
        Callable[[list[BaseSchedulerNode]], list[BaseSchedulerNode]]
    ] = []

    @classmethod
    def initialize(cls, test_instance: "TestScratchpadUsage"):
        cls.base_pass_list = CustomPostFusionPasses().get_passes()
        cls.test_instance = test_instance
        passes.CustomPostFusionPasses = CustomPostFusionPassesWithOurPasses

    def get_passes(self):
        assert self.test_instance is not None, (
            "CustomPostFusionPassesWithOurPasses.test_instance must be set to an instance of "
            "TestScratchpadUsage before get_passes is called"
        )
        return self.test_instance.our_scheduler_post_passes + self.base_pass_list


class TestScratchpadUsage(unittest.TestCase):
    our_scheduler_post_passes: list[
        Callable[[list[BaseSchedulerNode]], list[BaseSchedulerNode]]
    ] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        torch.manual_seed(0xAFFE)
        t_inductor_config.force_disable_caches = True
        torch.compiler.reset()
        if not CustomPostFusionPassesWithOurPasses.base_pass_list:
            # Monkey patch CustomPostFusionPasses to call our passes as well. We can't do it in
            # the class definition or in setUpClass because we need access to the test instance.
            CustomPostFusionPassesWithOurPasses.initialize(self)

    def cached_randn_device(self, shape: Sequence[int], *args, **kwargs):
        result = cached_randn(shape, *args, **kwargs)
        return result.to("spyre")

    @contextmanager
    def post_fusion_mapping_pass(
        self,
        f: Callable[[BaseSchedulerNode], BaseSchedulerNode],
    ):
        """Context manager to add a post fusion custom pass that processes each node independently
        using `f`."""

        def new_pass(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
            return [f(node) for node in nodes]

        self.our_scheduler_post_passes.append(new_pass)
        yield
        self.our_scheduler_post_passes.remove(new_pass)

    def compile_and_collect_mem_usage(
        self, f: Callable[[Unpack[Ts]], torch.Tensor], args: tuple[Unpack[Ts]]
    ) -> tuple[torch.Tensor, list[dict[str, dict[str, Any]]]]:
        mem_usages = []

        def visitor(node: BaseSchedulerNode) -> BaseSchedulerNode:
            nonlocal mem_usages
            if isinstance(node, SchedulerNode):
                mem_usage = mem_usage_by_node(node)
                for buffer_name, usage in mem_usage.items():
                    buffer = V.graph.get_buffer(buffer_name)
                    layout = buffer.get_layout()
                    allocation = getattr(layout, "allocation", {})
                    usage["location"] = (
                        "LX"
                        if any(key.startswith("lx") for key in allocation)
                        else "HBM"
                    )

                mem_usages.append(mem_usage)
            return node

        with self.post_fusion_mapping_pass(visitor):
            compiled_kernel = torch.compile(f, fullgraph=True)
            result = compiled_kernel(*args).to("cpu")

        return (result, mem_usages)

    def run_test(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
    ):
        """Run the current class's test procedure on the given model and arguments. Override this
        in each subclass."""
        with ts_inductor_config.patch(lx_planning=True):
            _, mem_usages = self.compile_and_collect_mem_usage(model, args)

        print(mem_usages)

        self.assertTrue(
            any(
                usage["location"] == "LX"
                for mem_usage in mem_usages
                for usage in mem_usage.values()
            )
        )

    def common(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
    ):
        """This method runs some sanity checks common to all subclasses and then calls
        `run_test`."""
        for t in args:
            self.assertIsInstance(t, torch.Tensor)
            self.assertEqual(t.device.type, "spyre")
        return self.run_test(model, args)

    def test_softmax(self):
        f = functools.partial(torch.softmax, dim=0)
        x = self.cached_randn_device((512, 1024))
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

    def run_test(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
    ):
        """Test that estimates the total amount of HBM transfers with LX planning turned off and
        turned on, and then compares them."""
        with ts_inductor_config.patch(lx_planning=False):
            result_without_lx, hbm_without_lx = self.measure_hbm_transfers(model, args)

        with ts_inductor_config.patch(lx_planning=True):
            result_with_lx, hbm_with_lx = self.measure_hbm_transfers(model, args)

        self.assertLess(hbm_with_lx, hbm_without_lx)

        delta = torch.abs(result_without_lx - result_with_lx).max().item()
        self.assertLess(delta, 1e-5)


if __name__ == "__main__":
    unittest.main()
