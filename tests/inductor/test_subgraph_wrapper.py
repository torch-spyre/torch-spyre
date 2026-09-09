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

"""Unit tests for SpyreSubgraphPythonWrapperCodegen (invoke_subgraph wrapper).

``torch.compiler.nested_compile_region`` bodies are lowered as a separate
``GraphLowering`` with their own wrapper codegen, obtained from
``SpyrePythonWrapperCodegen.create(is_subgraph=True, ...)``. Before
``SpyreSubgraphPythonWrapperCodegen`` existed, that returned the *stock*
``SubgraphPythonWrapperCodegen``, which knows nothing about Spyre buffers.

These tests pin the three contracts of that wrapper:

1. ``create(is_subgraph=True)`` returns the Spyre subclass (the dispatch).
2. The subclass gets the Spyre *buffer* codegen from
   ``_SpyreWrapperCodegenMixin`` -- both the method the stock wrapper lacks
   entirely (``generate_const_tensor_fallback``) and the one it silently gets
   wrong (``make_buffer_allocation``).
3. Role-specific behavior stays on the stock base: the subgraph's own
   ``sizevars`` is patched, but ``write_header`` remains a no-op so the parent
   module's imports are not emitted twice.

"""

import unittest

import torch
import torch_spyre  # noqa: F401  registers the "spyre" device
from sympy import Integer
from torch import fx
from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise
from torch._inductor.virtualized import V

from torch_spyre._C import ElementArrangement, SpyreTensorLayout
from torch_spyre._inductor.ir import FixedTiledLayout, SpyreConstantFallback
from torch_spyre._inductor.wrapper import (
    SpyrePythonWrapperCodegen,
    SpyreSubgraphPythonWrapperCodegen,
    noop_simplify_loops_impl,
)


def _enter_fresh_graph(test):
    """Push a bare GraphLowering as V.graph and return it.

    A wrapper's ``__init__`` reads ``V.graph`` (to patch its sizevars), so each
    wrapper must be built under the graph it belongs to -- the parent wrapper
    under the parent graph, the subgraph wrapper under the subgraph's own.
    Registered for cleanup on ``test`` so the handler unwinds in LIFO order.

    ``graph_outputs`` is normally assigned mid-lowering; a GraphLowering that
    never runs lowering leaves it unset, so fill it in for the buffer-reuse and
    free paths that consult ``get_output_names()``.
    """
    ctx = V.set_graph_handler(GraphLowering(fx.symbolic_trace(lambda: None)))
    ctx.__enter__()
    test.addCleanup(ctx.__exit__, None, None, None)
    V.graph.graph_outputs = []
    return V.graph


def _make_ftl_buffer(name="buf0", host_size=(64, 64)):
    """Real ComputedBuffer carrying a FixedTiledLayout.

    A FixedTiledLayout is what makes ``make_buffer_allocation`` take the Spyre
    branch; a stock layout falls through to ``super()``. Mirrors
    ``_make_ftl_buffer`` in test_hbm_pool_planning.py.
    """
    strides = [int(s) for s in FlexibleLayout.contiguous_strides(list(host_size))]
    device_layout = SpyreTensorLayout(
        list(host_size),
        strides,
        torch.float16,
        list(range(len(host_size))),
        ElementArrangement.STANDARD,
    )
    layout = FixedTiledLayout(
        torch.device("cpu"),
        torch.float16,
        [Integer(s) for s in host_size],
        [Integer(s) for s in strides],
        device_layout,
    )
    data = Pointwise(
        device=torch.device("cpu"),
        dtype=torch.float16,
        inner_fn=lambda index: Integer(1),
        ranges=[Integer(s) for s in host_size],
    )
    return ComputedBuffer(name=name, layout=layout, data=data)


class TestSpyreSubgraphWrapper(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # The parent wrapper belongs to the outer graph. Build it here, then
        # each test pushes a *second* graph for the subgraph itself, matching
        # how Inductor lowers an invoke_subgraph body inside a parent graph.
        parent_graph = _enter_fresh_graph(self)
        self.parent = SpyrePythonWrapperCodegen()
        self.parent_sizevars = parent_graph.sizevars

    def _make_subgraph_wrapper(self, name="subgraph_0"):
        _enter_fresh_graph(self)
        return SpyrePythonWrapperCodegen.create(
            is_subgraph=True, subgraph_name=name, parent_wrapper=self.parent
        )

    def test_create_returns_spyre_subgraph_wrapper(self):
        """create(is_subgraph=True) must build the Spyre subclass.

        This is the behavior delta of the change: the same call used to return
        a stock SubgraphPythonWrapperCodegen. Assert the exact type -- an
        isinstance check would also pass for the stock base class, since the
        Spyre wrapper inherits from it.
        """
        wrapper = self._make_subgraph_wrapper()
        self.assertIs(type(wrapper), SpyreSubgraphPythonWrapperCodegen)
        self.assertIs(wrapper.parent_wrapper, self.parent)

    def test_const_tensor_fallback_emits_spyre_constant(self):
        """A SpyreConstantFallback must codegen inside a subgraph body.

        ``SpyreConstantFallback.codegen`` calls
        ``wrapper.generate_const_tensor_fallback(self)`` unconditionally, and
        that method exists only on the Spyre wrapper. This is the concrete
        AttributeError that made a Spyre subgraph wrapper necessary: such a
        node is materialized inside a region body by split_multi_ops.
        """
        wrapper = self._make_subgraph_wrapper()
        node = SpyreConstantFallback(
            torch.ops.spyre.constant.default,
            3.5,
            torch.float16,
            torch.device("spyre"),
        )

        node.codegen(wrapper)

        self.assertIn(
            f'{node.get_name()} = spyre_constant_tensor(3.5, torch.device("spyre"), '
            "torch.float16)",
            wrapper.lines[-1],
        )

        # Witness that the override is load-bearing rather than redundant: the
        # stock wrapper has no such attribute. If a future torch grows this
        # method, this assertion fails and the witness is merely stale -- the
        # Spyre assertion above is the one that guards real behavior.
        stock = SubgraphPythonWrapperCodegen("subgraph_stock", self.parent)
        with self.assertRaises(AttributeError):
            node.codegen(stock)

    def test_buffer_allocation_uses_spyre_layout_allocator(self):
        """A FixedTiledLayout buffer must allocate via spyre_empty_with_layout.

        The failure mode this guards is silent rather than loud: handed a
        FixedTiledLayout, the stock wrapper emits a plain ``empty_strided``
        with no device layout at all, so a subgraph body would compile and run
        while allocating the wrong thing. Assert the absence of empty_strided
        explicitly.
        """
        wrapper = self._make_subgraph_wrapper()

        line = wrapper.make_buffer_allocation(_make_ftl_buffer())

        self.assertIn("spyre_empty_with_layout(", line)
        self.assertIn("SpyreTensorLayout(", line)
        self.assertNotIn("empty_strided", line)

    def test_subgraph_patches_own_sizevars_but_not_header(self):
        """The mixin must be wired at the right altitude.

        Two halves of one contract, and both break if the mixin is layered
        wrongly:

        * ``_patch_sizevars`` is called from ``__init__`` (not inherited as a
          side effect), so the subgraph's *own* sizevars gets the noop loop
          simplifier -- Spyre's device layout is not visible to Inductor, so
          its loop simplification would be wrong inside a region body too.
        * ``write_header`` stays the stock no-op. A subgraph is emitted as a
          ``def`` inside the parent's module and relies on the parent's
          imports; re-emitting the Spyre header would duplicate
          ``del async_compile`` and break the generated module.
        """
        wrapper = self._make_subgraph_wrapper()
        subgraph_graph = V.graph

        # Distinct GraphLowerings, so this proves the patch is per-graph and
        # not merely inherited from the parent's already-patched allocator.
        self.assertIsNot(subgraph_graph.sizevars, self.parent_sizevars)

        # Check the bound function first: if the patch is missing, the stock
        # simplifier raises from deep inside torch's sizevars on the call
        # below, which reports an IndexError rather than the real problem.
        self.assertIs(
            subgraph_graph.sizevars._simplify_loops_impl.__func__,
            noop_simplify_loops_impl,
        )

        sizes = [Integer(4), Integer(8)]
        simplified, reindex, prune = subgraph_graph.sizevars._simplify_loops_impl(
            [], sizes, []
        )
        self.assertEqual(simplified, sizes)
        sentinel = object()
        self.assertIs(reindex(sentinel), sentinel)
        self.assertIs(prune(sentinel), sentinel)

        wrapper.write_header()
        self.assertEqual(wrapper.imports.getvalue(), "")
        self.assertEqual(wrapper.header.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
