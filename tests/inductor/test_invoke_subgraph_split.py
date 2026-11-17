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

"""Regression test for invoke_subgraph + split_multi_ops graph identity.

A ``torch.compiler.nested_compile_region`` block with a *fused multi-op*
pointwise body (``relu(h * 3 + 1)`` -> mul, add, relu in one loop body, so
``split_multi_ops`` fires and reaches its FX-node insertion), on Spyre-device
tensors, called N times so the region lowers to an ``invoke_subgraph`` HOP.
Before the fix this failed during ``torch.compile`` with::

    torch._inductor.exc.InductorError: AssertionError:
        Node to insert before is not in graph.

Root cause: the subgraph ComputedBuffer's ``origins`` set spans TWO
``fx.Graph`` objects -- the parent's ``invoke_subgraph`` / ``get_attr`` nodes
AND the subgraph-local ``mul`` node. ``split_multi_ops`` picked
``next(iter(op.origins))``, which could return the parent's invoke_subgraph
node; that node is not in the subgraph's ``gl.graph``, so
``gl.graph.inserting_before(orig_node)`` asserted. The fix
(``pass_utils.origin_in_graph``) selects the origin whose ``.graph is
gl.graph``.
"""

import unittest

import torch
from torch import nn
from torch._inductor import config as t_inductor_config
from torch.compiler import nested_compile_region

import torch_spyre  # noqa: F401  registers "spyre" + installs the inductor passes
from torch_spyre.constants import DEVICE_NAME


class _Block(nn.Module):
    def forward(self, h):
        # Fused multi-op pointwise body: mul -> add -> relu in one loop body,
        # forcing split_multi_ops to materialize an intermediate and hit the
        # FX-node insertion that used to assert.
        return torch.relu(h * 3.0 + 1.0)


def _region(block):
    # nested_compile_region cannot mark a bound method, so wrap it.
    def wrapper(*args, **kwargs):
        return block.forward(*args, **kwargs)

    return nested_compile_region(wrapper)


class TestInvokeSubgraphSplit(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # Load-bearing, not hygiene: a cached FX graph is replayed without
        # re-running the Spyre pre-scheduling passes, so split_multi_ops never
        # fires and this test would pass against the broken code.
        patcher = t_inductor_config.patch("fx_graph_cache", False)
        patcher.__enter__()
        self.addCleanup(patcher.__exit__, None, None, None)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def test_nested_region_multi_op_compiles(self):
        """A reused region with a fused multi-op body must compile and run.

        Used to raise InductorError("Node to insert before is not in graph.")
        from split_multi_ops' FX insertion, during compilation.
        """
        blocks = [_region(_Block()) for _ in range(3)]

        def outer(h):
            for b in blocks:
                h = b(h)
            return h

        seen_hops = []

        def backend(gm, example_inputs):
            # Assert on what Inductor actually receives: if the regions were
            # inlined into the parent graph there is no cross-graph origin set,
            # and the test would silently guard nothing.
            seen_hops.append(
                [
                    n
                    for n in gm.graph.nodes
                    if "invoke_subgraph" in str(getattr(n, "target", ""))
                ]
            )
            from torch._inductor.compile_fx import compile_fx

            return compile_fx(gm, example_inputs)

        h = torch.randn(2, 64, dtype=torch.float16, device=DEVICE_NAME)
        compiled = torch.compile(outer, backend=backend, dynamic=False, fullgraph=True)

        out = compiled(h)

        self.assertEqual(tuple(out.shape), (2, 64))
        self.assertTrue(seen_hops, "compile backend never ran")
        self.assertGreaterEqual(
            len(seen_hops[0]),
            2,
            "expected repeated invoke_subgraph calls, got "
            f"{[n.name for n in seen_hops[0]]}",
        )

    def test_region_transposed_intermediate_layout(self):
        """A transposed intermediate passed into a reused region must keep its
        device layout, not one rebuilt from its host strides.

        x.t() * 1.0 produces a buffer with strides [1, 128] -- the stick
        dimension is dim 0, not the default dim 1.  Before the fix,
        propagate_spyre_tensor_layouts fell through to the host-stride fallback
        for invoke_subgraph inputs (hasattr(op, "operands") was always True but
        op.operands was None), and the rebuilt layout placed the stick on the
        wrong dimension, giving ~81% wrong values with no error raised.
        """

        @nested_compile_region
        def r(h):
            return h + 1.0

        def outer(x):
            # Two calls: a single call is inlined and no invoke_subgraph remains.
            return r(r(x.t() * 1.0))

        seen_hops = []

        def backend(gm, example_inputs):
            seen_hops.append(
                [
                    n
                    for n in gm.graph.nodes
                    if "invoke_subgraph" in str(getattr(n, "target", ""))
                ]
            )
            from torch._inductor.compile_fx import compile_fx

            return compile_fx(gm, example_inputs)

        x = torch.randn(64, 128, dtype=torch.float16, device=DEVICE_NAME)
        compiled = torch.compile(outer, backend=backend, dynamic=False, fullgraph=True)
        out = compiled(x)

        self.assertGreaterEqual(
            len(seen_hops[0]),
            2,
            f"expected two invoke_subgraph calls, got {[n.name for n in seen_hops[0]]}",
        )
        torch.testing.assert_close(out.cpu(), outer(x.cpu()), atol=0.1, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
