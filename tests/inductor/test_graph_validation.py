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

"""Unit tests for torch_spyre._inductor.graph_validation.

Tests exercise the validate_graph() public entry point. No Spyre
hardware is required — all tests build lightweight GraphLowering
instances with known buffer/operation configurations.
"""

import unittest
from unittest.mock import MagicMock

import torch
from sympy import Integer
from torch import fx
from torch._inductor.dependencies import MemoryDep, StarDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, FlexibleLayout, Pointwise
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from torch_spyre._inductor.graph_validation import (
    GraphValidationError,
    validate_graph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_HOST_SIZE = (64, 64)


def _enter_fresh_graph(test: unittest.TestCase) -> GraphLowering:
    """Push a bare GraphLowering as V.graph and register cleanup."""
    ctx = V.set_graph_handler(GraphLowering(fx.symbolic_trace(lambda: None)))
    ctx.__enter__()
    test.addCleanup(ctx.__exit__, None, None, None)
    V.graph.graph_outputs = []
    return V.graph


def _make_buffer(name: str, host_size: tuple[int, ...] = _HOST_SIZE) -> ComputedBuffer:
    """Create a minimal ComputedBuffer with the given name."""
    strides = [int(s) for s in FlexibleLayout.contiguous_strides(list(host_size))]
    layout = FlexibleLayout(
        torch.device("cpu"),
        torch.float16,
        [Integer(s) for s in host_size],
        [Integer(s) for s in strides],
    )
    data = Pointwise(
        device=torch.device("cpu"),
        dtype=torch.float16,
        inner_fn=lambda index: Integer(1),
        ranges=[Integer(s) for s in host_size],
    )
    buf = ComputedBuffer(name=name, layout=layout, data=data)
    return buf


def _register_buffer(graph: GraphLowering, buf: ComputedBuffer) -> str:
    """Register a buffer on the graph in the standard way."""
    name = graph.register_buffer(buf, set_name=True)
    return name


def _register_and_add_operation(
    graph: GraphLowering,
    buf: ComputedBuffer,
) -> str:
    """Register a buffer AND add it as an operation."""
    name = _register_buffer(graph, buf)
    graph.register_operation(buf)
    return name


def _make_read_writes(reads: list[str], writes: list[str]):
    """Create a mock ReadWrites with the given buffer names as reads/writes."""
    mock_rw = MagicMock()
    mock_rw.reads = OrderedSet()
    for r in reads:
        dep = MagicMock(spec=MemoryDep)
        dep.name = r
        mock_rw.reads.add(dep)
    mock_rw.writes = OrderedSet()
    for w in writes:
        dep = MagicMock(spec=MemoryDep)
        dep.name = w
        mock_rw.writes.add(dep)
    return mock_rw


# ---------------------------------------------------------------------------
# Tests: Happy Path
# ---------------------------------------------------------------------------


class TestGraphValidationHappyPath(unittest.TestCase):
    """A well-formed graph should pass all checks without error."""

    def test_empty_graph_is_valid(self):
        graph = _enter_fresh_graph(self)
        validate_graph(graph)

    def test_single_buffer_single_op(self):
        graph = _enter_fresh_graph(self)
        buf = _make_buffer("placeholder")
        _register_and_add_operation(graph, buf)
        validate_graph(graph)

    def test_multiple_buffers_and_ops(self):
        graph = _enter_fresh_graph(self)
        for _ in range(5):
            buf = _make_buffer("placeholder")
            _register_and_add_operation(graph, buf)
        validate_graph(graph)

    def test_removed_buffer_in_removed_set(self):
        """A buffer in removed_buffers should not cause a validation error
        as long as it is still in the buffers list."""
        graph = _enter_fresh_graph(self)
        buf = _make_buffer("placeholder")
        name = _register_and_add_operation(graph, buf)
        graph.removed_buffers.add(name)
        graph.name_to_buffer.pop(name, None)
        graph.operations.remove(buf)
        validate_graph(graph)

    def test_graph_with_output(self):
        graph = _enter_fresh_graph(self)
        buf = _make_buffer("placeholder")
        _register_and_add_operation(graph, buf)
        graph.graph_outputs = [buf]
        validate_graph(graph)

    def test_pass_name_in_error_messages(self):
        """pass_name should appear in error messages when provided."""
        graph = _enter_fresh_graph(self)
        buf = _make_buffer("placeholder")
        _register_buffer(graph, buf)
        dup_buf = _make_buffer("placeholder")
        dup_buf.name = buf.get_name()
        graph.buffers.append(dup_buf)
        with self.assertRaises(GraphValidationError) as ctx:
            validate_graph(graph, pass_name="test_pass")
        self.assertIn("test_pass", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests: Buffer Name Uniqueness (INV-1)
# ---------------------------------------------------------------------------


class TestBufferNameUniqueness(unittest.TestCase):
    def setUp(self):
        self.graph = _enter_fresh_graph(self)

    def test_duplicate_buffer_names_detected(self):
        """Two buffers with the same name should be caught."""
        buf1 = _make_buffer("placeholder")
        _register_buffer(self.graph, buf1)
        buf2 = _make_buffer("placeholder")
        buf2.name = buf1.get_name()
        self.graph.buffers.append(buf2)

        with self.assertRaises(GraphValidationError) as ctx:
            validate_graph(self.graph)
        self.assertIn("unique", str(ctx.exception).lower())

    def test_unique_names_pass(self):
        for _ in range(3):
            buf = _make_buffer("placeholder")
            _register_buffer(self.graph, buf)
        validate_graph(self.graph)


# ---------------------------------------------------------------------------
# Tests: name_to_buffer Consistency (INV-3)
# ---------------------------------------------------------------------------


class TestNameToBufferConsistency(unittest.TestCase):
    def setUp(self):
        self.graph = _enter_fresh_graph(self)

    def test_missing_name_to_buffer_entry(self):
        """A live buffer not in name_to_buffer should be caught."""
        buf = _make_buffer("placeholder")
        _register_buffer(self.graph, buf)
        name = buf.get_name()
        self.graph.name_to_buffer.pop(name)

        with self.assertRaises(GraphValidationError) as ctx:
            validate_graph(self.graph)
        self.assertIn("name_to_buffer", str(ctx.exception))

    def test_stale_name_to_buffer_for_removed_buffer(self):
        """A removed buffer still in name_to_buffer should be caught."""
        buf = _make_buffer("placeholder")
        name = _register_buffer(self.graph, buf)
        self.graph.removed_buffers.add(name)

        with self.assertRaises(GraphValidationError) as ctx:
            validate_graph(self.graph)
        self.assertIn("removed", str(ctx.exception).lower())

    def test_name_to_buffer_references_wrong_buffer(self):
        """name_to_buffer entry pointing to wrong buffer object is caught."""
        buf1 = _make_buffer("placeholder")
        name1 = _register_buffer(self.graph, buf1)
        buf2 = _make_buffer("placeholder")
        _register_buffer(self.graph, buf2)
        self.graph.name_to_buffer[name1] = buf2

        with self.assertRaises(GraphValidationError) as ctx:
            validate_graph(self.graph)
        self.assertIn("name_to_buffer", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests: name_to_op Consistency (INV-4)
# ---------------------------------------------------------------------------


class TestNameToOpConsistency(unittest.TestCase):
    def setUp(self):
        self.graph = _enter_fresh_graph(self)

    def test_missing_name_to_op_entry(self):
        """A live operation not in name_to_op should be caught."""
        buf = _make_buffer("placeholder")
        _register_and_add_operation(self.graph, buf)
        op_name = buf.get_operation_name()
        self.graph.name_to_op.pop(op_name)

        with self.assertRaises(GraphValidationError) as ctx:
            validate_graph(self.graph)
        self.assertIn("name_to_op", str(ctx.exception))

    def test_operation_with_none_operation_name(self):
        """An operation with operation_name=None should be caught."""
        buf = _make_buffer("placeholder")
        _register_and_add_operation(self.graph, buf)
        buf.operation_name = None

        with self.assertRaises(GraphValidationError) as ctx:
            validate_graph(self.graph)
        self.assertIn("operation_name", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests: Reads from Defined Buffers (INV-5)
# ---------------------------------------------------------------------------


class TestReadsFromDefinedBuffers(unittest.TestCase):
    def setUp(self):
        self.graph = _enter_fresh_graph(self)

    def test_read_from_undefined_buffer(self):
        """An operation reading a buffer not in name_to_buffer or
        graph_inputs should be caught."""
        buf = _make_buffer("placeholder")
        _register_and_add_operation(self.graph, buf)
        mock_rw = _make_read_writes(reads=["nonexistent_buf"], writes=[buf.get_name()])
        buf.get_read_writes = lambda: mock_rw

        with self.assertRaises(GraphValidationError) as ctx:
            validate_graph(self.graph)
        self.assertIn("undefined", str(ctx.exception).lower())

    def test_read_from_graph_input_is_valid(self):
        """Reading from a graph input should not raise."""
        buf = _make_buffer("placeholder")
        _register_and_add_operation(self.graph, buf)
        self.graph.graph_inputs["input_buf"] = MagicMock()
        mock_rw = _make_read_writes(reads=["input_buf"], writes=[buf.get_name()])
        buf.get_read_writes = lambda: mock_rw
        validate_graph(self.graph)

    def test_star_dep_is_ignored(self):
        """StarDep reads (ordering dependencies) should not be validated
        as buffer reads."""
        buf = _make_buffer("placeholder")
        _register_and_add_operation(self.graph, buf)
        mock_rw = MagicMock()
        star = MagicMock(spec=StarDep)
        star.name = "nonexistent"
        mock_rw.reads = OrderedSet([star])
        mock_rw.writes = OrderedSet()
        buf.get_read_writes = lambda: mock_rw
        validate_graph(self.graph)


# ---------------------------------------------------------------------------
# Tests: Graph Outputs Valid (INV-6)
# ---------------------------------------------------------------------------


class TestGraphOutputsValid(unittest.TestCase):
    def setUp(self):
        self.graph = _enter_fresh_graph(self)

    def test_output_referencing_undefined_buffer(self):
        """A graph output whose name is not in name_to_buffer or
        graph_inputs should be caught."""
        mock_output = MagicMock()
        mock_output.get_name.return_value = "nonexistent_output"
        self.graph.graph_outputs = [mock_output]

        with self.assertRaises(GraphValidationError) as ctx:
            validate_graph(self.graph)
        self.assertIn("output", str(ctx.exception).lower())

    def test_valid_output(self):
        """An output that references a registered buffer should pass."""
        buf = _make_buffer("placeholder")
        _register_buffer(self.graph, buf)
        self.graph.graph_outputs = [buf]
        validate_graph(self.graph)


# ---------------------------------------------------------------------------
# Tests: No Reads from Removed Buffers (INV-8)
# ---------------------------------------------------------------------------


class TestNoReadsFromRemovedBuffers(unittest.TestCase):
    def setUp(self):
        self.graph = _enter_fresh_graph(self)

    def test_read_from_removed_buffer(self):
        """An operation reading from a buffer in removed_buffers should be
        caught."""
        producer = _make_buffer("placeholder")
        prod_name = _register_and_add_operation(self.graph, producer)
        consumer = _make_buffer("placeholder")
        _register_and_add_operation(self.graph, consumer)

        self.graph.removed_buffers.add(prod_name)
        self.graph.name_to_buffer.pop(prod_name, None)
        self.graph.operations.remove(producer)

        mock_rw = _make_read_writes(reads=[prod_name], writes=[consumer.get_name()])
        consumer.get_read_writes = lambda: mock_rw

        with self.assertRaises(GraphValidationError) as ctx:
            validate_graph(self.graph)
        err_msg = str(ctx.exception).lower()
        self.assertTrue(
            "removed" in err_msg or "undefined" in err_msg,
            f"Expected 'removed' or 'undefined' in error: {ctx.exception}",
        )


# ---------------------------------------------------------------------------
# Tests: GraphValidationError attributes
# ---------------------------------------------------------------------------


class TestGraphValidationErrorAttributes(unittest.TestCase):
    def test_invariant_and_pass_name_stored(self):
        err = GraphValidationError("INV-1", "detail", pass_name="my_pass")
        self.assertEqual(err.invariant, "INV-1")
        self.assertEqual(err.pass_name, "my_pass")
        self.assertIn("my_pass", str(err))
        self.assertIn("INV-1", str(err))

    def test_empty_pass_name(self):
        err = GraphValidationError("INV-1", "detail")
        self.assertNotIn("[after", str(err))


if __name__ == "__main__":
    unittest.main()
