from typing import Any, Callable

import unittest
import torch
import os

from torch._inductor.scheduler import BaseSchedulerNode, SchedulerNode
from tests.inductor.utils_inductor import cached_randn

import torch._inductor.config as inductor_config
from torch_spyre._inductor.scratchpad import mem_usage_by_node
from torch._inductor.virtualized import V


class BaseScratchpadTest(unittest.TestCase):
    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        """All derived classes should override this; this base class should be skipped."""
        assert False, "Override the kernel method in all subclasses"

    @classmethod
    def setUpClass(cls):
        if cls is BaseScratchpadTest:
            raise unittest.SkipTest("Skip BaseScratchpadTest tests, it's a base class")
        super(BaseScratchpadTest, cls).setUpClass()

    def setUp(self):
        torch.manual_seed(0xAFFE)

        # Prepare a device tensor
        x = cached_randn((512, 1024))
        self.x = x.to("spyre")

        torch.compiler.reset()

    def set_lx_planning(self, enable: bool):
        old_value = os.environ.get("LX_PLANNING", "0")
        os.environ["LX_PLANNING"] = "1" if enable else "0"
        if os.environ.get("LX_PLANNING") != old_value:
            # Clear build cache
            torch.compiler.reset()

    def add_post_fusion_mapping_pass(
        self,
        f: Callable[[BaseSchedulerNode], BaseSchedulerNode],
    ):
        """Set a post fusion custom pass that processes each node independently."""
        old_pass = inductor_config._post_fusion_custom_pass

        def new_pass(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
            return [f(node) for node in nodes]

        if old_pass is None:
            inductor_config._post_fusion_custom_pass = new_pass
        else:

            def combined_pass(
                nodes: list[BaseSchedulerNode],
            ) -> list[BaseSchedulerNode]:
                return new_pass(old_pass(nodes))

            inductor_config._post_fusion_custom_pass = combined_pass

    def extended_mem_usages(
        self,
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

        self.add_post_fusion_mapping_pass(visitor)

        compiled_kernel = torch.compile(self.kernel)
        try:
            result = compiled_kernel(self.x).cpu()
        except:  # noqa: E722
            # When https://github.com/torch-spyre/torch-spyre/issues/1257 is fixed, we can remove
            # the try/except block here and the None in the return type.
            result = None

        return (result, mem_usages)

    def test_lx_is_used(self):
        """Basic test to see if the LX is used in any allocation when LX planning is turned on."""
        self.set_lx_planning(True)

        _, emus = self.extended_mem_usages()
        self.assertTrue(
            any(usage["location"] == "LX" for emu in emus for usage in emu.values())
        )

    def measure_hbm_transfers(self) -> tuple[torch.Tensor | None, int]:
        """Estimates the HBM transfers for a given operation. This assumes that any buffer that
        has an entry in its allocations that starts with "lx" is free and that any other node's HBM
        transfers are accurately returned by `mem_usage_by_node`."""
        result, emus = self.extended_mem_usages()
        hbm_transfers = sum(
            usage["size"]
            for emu in emus
            for usage in emu.values()
            if usage["location"] == "HBM"
        )
        return (result, hbm_transfers)

    def test_compare_hbm_use(self):
        """Test that estimates the total amount of HBM transfers with LX planning turned off and
        turned on, and then compares them."""
        self.set_lx_planning(False)
        result_without_lx, hbm_without_lx = self.measure_hbm_transfers()

        self.set_lx_planning(True)
        result_with_lx, hbm_with_lx = self.measure_hbm_transfers()

        self.assertLess(hbm_with_lx, hbm_without_lx)

        if result_without_lx is not None and result_with_lx is not None:
            delta = torch.abs(result_without_lx - result_with_lx).max().item()
            self.assertLess(delta, 1e-5)


class SoftMaxScratchPadTest(BaseScratchpadTest):
    def kernel(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=0)


if __name__ == "__main__":
    unittest.main()
