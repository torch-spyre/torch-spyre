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

"""Compile-time regression test for buffer origins inside invoke_subgraph bodies.

A ComputedBuffer lowered inside an ``invoke_subgraph`` HOP body -- what a
``torch.compiler.nested_compile_region`` block reused across layers lowers to --
carries an ``origins`` set spanning TWO fx.Graphs: the parent graph's
``invoke_subgraph`` / ``repeated_subgraph0`` nodes AND the subgraph's own compute
node. Observed origins for such a buffer, with graph membership::

    origins   = ['invoke_subgraph', 'repeated_subgraph0', 'mul']
    is_local  = [False,             False,                True]

``split_multi_ops`` anchors its FX insertion on an origin node. The pre-fix
``next(iter(op.origins))`` could hand back either *foreign* parent-graph node,
and FX rejects an anchor from another graph::

    torch._inductor.exc.InductorError: AssertionError:
        Node to insert before is not in graph.

``split_multi_ops._origin_in_graph`` selects the origin whose ``.graph`` is the
graph being edited (also used by ``scratchpad/graph_editor.py``).

Verified in both directions: with the fix, compilation proceeds past the pass;
with ``_origin_in_graph`` reverted to the first-origin heuristic, the assertion
above is raised (3/3 runs).

"""

import unittest
from unittest import mock

import torch
from torch import nn
from torch.compiler import nested_compile_region

import torch_spyre  # noqa: F401  installs the inductor passes
from torch_spyre.constants import DEVICE_NAME

# NB: plain unittest, NOT torch.testing._internal.common_utils.run_tests --
# run_tests() calls torch.manual_seed(), which fires torch-spyre's custom-device
# seed hook and eagerly initializes the Spyre VFIO device, turning this
# compile-only test into a device test that fails when a card is busy. See the
# same note in tests/test_nested_compile_region_guard.py.

# Sentinel raised from the stubbed backend-compiler boundary. Reaching it means
# every Inductor pass -- including the one under test -- has already run.
_REACHED_BACKEND = "spyre-test: reached SDSC backend compiler"


class _Block(nn.Module):
    def forward(self, h):
        # The matmul is load-bearing: it blocks fusion into the parent graph, so
        # the region body lowers as its own GraphLowering (the only way a buffer
        # gets cross-graph origins). The trailing pointwise chain then puts
        # mul + add in one loop body, which is what makes split_multi_ops
        # materialize an intermediate and reach its FX insertion.
        return torch.relu(h @ h.t() * 3.0 + 1.0)


def _region_block(block):
    # nested_compile_region cannot mark a bound method, so wrap it.
    def wrapper(*args, **kwargs):
        return block.forward(*args, **kwargs)

    return nested_compile_region(wrapper)


class TestInvokeSubgraphSplitMultiOps(unittest.TestCase):
    """Compile a reused region whose body drives split_multi_ops."""

    def setUp(self):
        super().setUp()
        # Cut the pipeline at the SDSC backend compiler. Dynamo, the subgraph
        # lowering, and the whole pre-scheduling pass pipeline (split_multi_ops
        # included) have all run by the time async_compile.sdsc() shells out to
        # dxp_standalone, so raising here keeps the test on the pass under test
        # and off unfinished device support for region bodies. Patched in the
        # async_compile namespace only, so nothing else is affected.
        patcher = mock.patch(
            "torch_spyre.execution.async_compile.subprocess.run",
            side_effect=RuntimeError(_REACHED_BACKEND),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def test_region_body_compiles_past_split_multi_ops(self):
        """Lowering a reused region body must not raise from origin selection.

        Before the fix, Inductor codegen of the subgraph body raised::

            InductorError: AssertionError: Node to insert before is not in graph

        because the insertion anchor was a parent-graph node. The assertion here
        is narrow on purpose: compilation must get all the way to the stubbed
        backend compiler, which is only reachable once the pass pipeline has
        completed.
        """
        blocks = [_region_block(_Block()) for _ in range(3)]

        def outer(h):
            for b in blocks:
                h = b(h)
            return h

        h = torch.randn(64, 64, dtype=torch.float16, device=DEVICE_NAME)
        compiled = torch.compile(outer, dynamic=False, fullgraph=True)

        with self.assertRaises(Exception) as ctx:
            compiled(h)

        chain = []
        cur: BaseException | None = ctx.exception
        while cur is not None:
            chain.append(f"{type(cur).__name__}: {cur}")
            cur = cur.__cause__ or cur.__context__
        joined = "\n".join(chain)

        # The regression itself: a foreign-graph FX insertion anchor.
        self.assertNotIn(
            "Node to insert before is not in graph",
            joined,
            "split_multi_ops anchored FX insertion on an origin node from "
            "another graph -- _origin_in_graph is not filtering origins to the "
            f"graph being lowered.\n{joined}",
        )

        # Positive half: compilation really did reach the backend boundary,
        # rather than dying somewhere earlier for an unrelated reason. Without
        # this, any early failure would satisfy the assertion above.
        self.assertIn(
            _REACHED_BACKEND,
            joined,
            f"compilation did not reach the SDSC backend compiler.\n{joined}",
        )


if __name__ == "__main__":
    unittest.main()
