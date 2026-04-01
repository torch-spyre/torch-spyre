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
from typing import Any, Callable, TypeVarTuple, Unpack

import unittest
import torch
import os

from torch._inductor.scheduler import BaseSchedulerNode, SchedulerNode
from tests.inductor.utils_inductor import cached_randn

from torch_spyre._inductor.scratchpad import mem_usage_by_node
from torch_spyre._inductor import passes
from torch._inductor.virtualized import V

Ts = TypeVarTuple("Ts")


class TestScratchpadUsage(unittest.TestCase):
    our_scheduler_post_passes: list[
        Callable[[list[BaseSchedulerNode]], list[BaseSchedulerNode]]
    ] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.old_scheduler_post_passes = passes.scheduler_post_passes
        passes.scheduler_post_passes = self.post_passes_bypass

    def post_passes_bypass(
        self, nodes: list[BaseSchedulerNode]
    ) -> list[BaseSchedulerNode]:
        nodes = self.old_scheduler_post_passes(nodes)
        for our_pass in self.our_scheduler_post_passes:
            nodes = our_pass(nodes)
        return nodes

    def setUp(self):
        torch.manual_seed(0xAFFE)
        torch.compiler.reset()

    def cached_randn_device(self, shape: Sequence[int], *args, **kwargs):
        result = cached_randn(shape, *args, **kwargs)
        return result.to("spyre")

    @contextmanager
    def set_lx_planning(self, enable: bool):
        """Context manager to set the LX_PLANNING environment variable to enable or disable LX
        planning."""
        old_value = os.environ.get("LX_PLANNING", "0")
        os.environ["LX_PLANNING"] = "1" if enable else "0"
        modifying = os.environ.get("LX_PLANNING") != old_value
        if modifying:
            # Clear build cache so that old builds with the previous value of LX_PLANNING don't
            # interfere with the test
            torch.compiler.reset()
        yield
        if modifying:
            # Again clear build cache so that builds with the modified value of LX_PLANNING don't
            # interfere with future tests
            torch.compiler.reset()
            os.environ["LX_PLANNING"] = old_value

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
    ) -> tuple[torch.Tensor | None, list[dict[str, dict[str, Any]]]]:
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
            try:
                result = compiled_kernel(*args)
            except:  # noqa: E722
                # When https://github.com/torch-spyre/torch-spyre/issues/1257 is fixed, we can remove
                # the try/except block here and the None in the return type.
                result = None

        return (result, mem_usages)

    def run_test(
        self,
        model: Callable[[Unpack[Ts]], torch.Tensor],
        args: tuple[Unpack[Ts]],
    ):
        """Run the current class's test procedure on the given model and arguments. Override this
        in each subclass."""
        with self.set_lx_planning(True):
            _, emus = self.compile_and_collect_mem_usage(model, args)

        self.assertTrue(
            any(usage["location"] == "LX" for emu in emus for usage in emu.values())
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
        result, emus = self.compile_and_collect_mem_usage(model, args)
        hbm_transfers = sum(
            usage["size"]
            for emu in emus
            for usage in emu.values()
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
        with self.set_lx_planning(False):
            result_without_lx, hbm_without_lx = self.measure_hbm_transfers(model, args)

        with self.set_lx_planning(True):
            result_with_lx, hbm_with_lx = self.measure_hbm_transfers(model, args)

        self.assertLess(hbm_with_lx, hbm_without_lx)

        if result_without_lx is not None and result_with_lx is not None:
            delta = torch.abs(result_without_lx - result_with_lx).max().item()
            self.assertLess(delta, 1e-5)


if __name__ == "__main__":
    unittest.main()
