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

"""IR-level / mocked-IR unit tests for WhileLoop -> for_each_tile lowering.

No Spyre device or backend compiler is required. Covers four areas, each
in its own class group:
  1. An eager-mode sanity check that the nested for_each_tile fixture's
     reference implementation matches plain matmul (TestNestedForEach
     TileFixture).
  2. while_loop_bridge's generic while_loop -> coarse-tile-group bridge:
     CarryBinding/carry_bindings_for and splice_while_loop's buffer
     transplant, carry/xs-leaf read redirection, and mutated-carry
     re-read guard, exercised against hand-built mocks
     (TestCarryBindingsFor, TestSpliceWhileLoop).
  3. for_each_tile_lowering's splice_while_loops pass and try_prove_
     for_each_tile shape prover, exercised against real ir.WhileLoop
     nodes built by lowering the vendored fixtures through GraphLowering
     (TestSpliceWhileLoops, TestTryProveForEachTile).
  4. Pass-pipeline registration: splice_while_loops runs first, ahead of
     every other pre-scheduling pass (TestPassPipelineRegistration).

For end-to-end compilation + numerical correctness against a CPU
reference, see test_for_each_tile_e2e.py.
"""

import unittest
from unittest import mock

import torch
from torch._inductor.virtualized import V

from tests.inductor.for_each_tile_fixtures import (
    capture_post_grad_while_loop,
    matmul_inputs,
    nested_split_m_then_k_fn,
    nested_split_m_then_k_reference,
    split_k_fn,
    split_m_elementwise_fn,
    split_m_fn,
)
from torch_spyre._inductor.wsr.for_each_tile_lowering import (
    try_prove_for_each_tile,
)


class TestNestedForEachTileFixture(unittest.TestCase):
    """Eager-mode sanity check for the nested for_each_tile fixture's reference."""

    def test_reference_matches_plain_matmul(self):
        (X, Y), expected = matmul_inputs()
        actual = nested_split_m_then_k_reference(X, Y)
        torch.testing.assert_close(actual, expected)

    def test_eager_fn_matches_plain_matmul(self):
        (X, Y), expected = matmul_inputs()
        actual = nested_split_m_then_k_fn(X, Y)
        torch.testing.assert_close(actual, expected)


class TestCarryBindingsFor(unittest.TestCase):
    def test_one_carry_positional_match(self):
        from torch_spyre._inductor.wsr.while_loop_bridge import (
            CarryBinding,
            carry_bindings_for,
        )

        while_op = mock.Mock()
        while_op.carried_inputs = ["init0"]
        while_op.body_subgraph.graph.graph_outputs = ["out0"]

        bindings = carry_bindings_for(while_op)

        self.assertEqual(len(bindings), 1)
        self.assertIsInstance(bindings[0], CarryBinding)
        self.assertEqual(bindings[0].carry_index, 0)
        self.assertEqual(bindings[0].initial, "init0")
        self.assertEqual(bindings[0].body_output, "out0")
        self.assertTrue(bindings[0].scratch_name)

    def test_multiple_carries_preserve_order(self):
        from torch_spyre._inductor.wsr.while_loop_bridge import carry_bindings_for

        while_op = mock.Mock()
        while_op.carried_inputs = ["init0", "init1", "init2"]
        while_op.body_subgraph.graph.graph_outputs = ["out0", "out1", "out2"]

        bindings = carry_bindings_for(while_op)

        self.assertEqual([b.carry_index for b in bindings], [0, 1, 2])
        self.assertEqual([b.initial for b in bindings], ["init0", "init1", "init2"])
        self.assertEqual([b.body_output for b in bindings], ["out0", "out1", "out2"])
        # Every binding gets a distinct scratch name.
        names = [b.scratch_name for b in bindings]
        self.assertEqual(len(names), len(set(names)))

    def test_no_carries(self):
        from torch_spyre._inductor.wsr.while_loop_bridge import carry_bindings_for

        while_op = mock.Mock()
        while_op.carried_inputs = []
        while_op.body_subgraph.graph.graph_outputs = []

        bindings = carry_bindings_for(while_op)

        self.assertEqual(bindings, [])


class TestSpliceWhileLoop(unittest.TestCase):
    def test_removes_while_op_and_inserts_body_ops(self):
        import torch_spyre._inductor.wsr.while_loop_bridge as bridge

        body_op_a = mock.Mock(name="body_op_a", spec=["get_operation_name"])
        body_op_b = mock.Mock(name="body_op_b", spec=["get_operation_name"])
        multi_output = mock.Mock(name="multi_output")
        multi_output.inputs = []

        before = mock.Mock(name="before")
        before.inputs = []

        while_op = mock.Mock()
        while_op.carried_inputs = []
        while_op.inputs = []
        while_op.body_subgraph.graph.graph_outputs = []
        while_op.body_subgraph.graph.graph_inputs = {}
        while_op.body_subgraph.graph.operations = [body_op_a, body_op_b]
        while_op.body_subgraph.graph.name_to_op = {}
        while_op.body_subgraph.graph.name_to_buffer = {}

        graph = mock.Mock()
        graph.operations = [before, while_op, multi_output]
        graph.name_to_op = {}
        graph.name_to_buffer = {}
        graph.buffers = []

        spliced = bridge.splice_while_loop(graph, while_op, carries=[])

        self.assertEqual(spliced, [body_op_a, body_op_b])
        self.assertNotIn(while_op, graph.operations)
        self.assertIn(body_op_a, graph.operations)
        self.assertIn(body_op_b, graph.operations)

    def test_splices_at_while_op_position(self):
        import torch_spyre._inductor.wsr.while_loop_bridge as bridge

        body_op = mock.Mock(name="body_op", spec=["get_operation_name"])
        before = mock.Mock(name="before")
        before.inputs = []
        after = mock.Mock(name="after")
        after.inputs = []

        while_op = mock.Mock()
        while_op.carried_inputs = []
        while_op.inputs = []
        while_op.body_subgraph.graph.graph_outputs = []
        while_op.body_subgraph.graph.graph_inputs = {}
        while_op.body_subgraph.graph.operations = [body_op]
        while_op.body_subgraph.graph.name_to_op = {}
        while_op.body_subgraph.graph.name_to_buffer = {}

        graph = mock.Mock()
        graph.operations = [before, while_op, after]
        graph.name_to_op = {}
        graph.name_to_buffer = {}
        graph.buffers = []

        bridge.splice_while_loop(graph, while_op, carries=[])

        self.assertEqual(graph.operations, [before, body_op, after])

    def test_direct_input_ref_and_buffer_transplant_with_real_carry(self):
        """Regression coverage for Bug A (buffer transplant) and Bug B
        (carry/xs-leaf read-side redirection), using mocks that exercise the
        actual code paths rather than just silencing AttributeErrors.

        Shape: one carry (index 0, a pass-through: body_output IS the
        placeholder) plus one non-carry xs leaf (index 1). consumer_op has
        no `.data` (so it is not routed through redirect_computed_buffer_
        reads/ComputedBuffer reconstruction -- that machinery needs a real
        frozen ComputedBuffer and is exercised end-to-end against real
        compiled graphs in test_for_each_tile_e2e.py instead); it
        holds direct .inputs references to both placeholders, mirroring
        DynamicScalar/ExternKernelOut's real read shape. producer_op is a
        distinct op that "produces" the carry's own buffer, standing in for
        an op whose output must become visible to the outer graph
        (Bug A's regression surface).
        """
        import torch_spyre._inductor.wsr.while_loop_bridge as bridge

        carry_placeholder = mock.Mock(name="carry_placeholder", spec=["get_name"])
        carry_placeholder.get_name.return_value = "while_loop_body_graph_0_0_arg0_1"

        xs_placeholder = mock.Mock(name="xs_placeholder", spec=["get_name"])
        xs_placeholder.get_name.return_value = "while_loop_body_graph_0_0_arg1_1"

        real_carry_init = mock.Mock(name="real_carry_init", spec=["get_name"])
        real_carry_init.get_name.return_value = "outer_carry_buf"

        real_xs_input = mock.Mock(name="real_xs_input", spec=["get_name"])
        real_xs_input.get_name.return_value = "outer_xs_buf"

        # consumer_op has no `.data` -- direct-object-reference read shape
        # (DynamicScalar/ExternKernelOut), routed through
        # _substitute_direct_input_refs, not redirect_computed_buffer_reads.
        consumer_op = mock.Mock(
            name="consumer_op", spec=["get_operation_name", "inputs"]
        )
        consumer_op.get_operation_name.return_value = "consumer_op"
        consumer_op.inputs = [carry_placeholder, xs_placeholder]

        produced_buf = mock.Mock(name="produced_buf", spec=["get_name"])
        produced_buf.get_name.return_value = "while_loop_body_graph_0_0_buf5"

        producer_op = mock.Mock(
            name="producer_op",
            spec=["get_operation_name", "get_outputs", "inputs"],
        )
        producer_op.get_operation_name.return_value = "producer_op"
        producer_op.get_outputs.return_value = [produced_buf]
        producer_op.inputs = []

        while_op = mock.Mock()
        while_op.carried_inputs = [real_carry_init]
        while_op.inputs = [real_carry_init, real_xs_input]
        while_op.body_subgraph.graph.graph_outputs = [carry_placeholder]
        while_op.body_subgraph.graph.graph_inputs = {
            "while_loop_body_graph_0_0_arg0_1": carry_placeholder,
            "while_loop_body_graph_0_0_arg1_1": xs_placeholder,
        }
        while_op.body_subgraph.graph.operations = [producer_op, consumer_op]
        while_op.body_subgraph.graph.name_to_op = {"producer_op": producer_op}
        while_op.body_subgraph.graph.name_to_buffer = {
            "while_loop_body_graph_0_0_buf5": produced_buf
        }

        graph = mock.Mock()
        graph.operations = [while_op]
        graph.name_to_op = {}
        graph.name_to_buffer = {}
        graph.buffers = []

        carries = bridge.carry_bindings_for(while_op)
        self.assertEqual(len(carries), 1)

        bridge.splice_while_loop(graph, while_op, carries)

        # Bug B: consumer_op's direct .inputs references to both
        # placeholders must be rewritten to the real outer-graph objects --
        # the carry (pass-through: body_output IS the placeholder) to
        # while_op.carried_inputs[0], the xs leaf to while_op.inputs[1].
        self.assertEqual(consumer_op.inputs, [real_carry_init, real_xs_input])

        # Bug A: producer_op's own output buffer must become visible to the
        # OUTER graph's registries, not just the (mocked) inner body_graph's.
        self.assertIn("while_loop_body_graph_0_0_buf5", graph.name_to_buffer)
        self.assertIs(
            graph.name_to_buffer["while_loop_body_graph_0_0_buf5"], produced_buf
        )
        self.assertIn(produced_buf, graph.buffers)
        self.assertEqual(graph.name_to_op.get("producer_op"), producer_op)

    def test_mutated_carry_read_elsewhere_detected_as_extra_reader(self):
        """A read of a mutated carry's placeholder AFTER its own producer
        is a write-after-read hazard: ``_extra_readers_of_placeholder`` is
        what ``splice_while_loop`` now consults to find it (see
        ``_snapshot_carry_placeholder``'s docstring) -- rather than the
        earlier design this test used to cover, where any such read raised
        ``RuntimeError`` outright. That guard was superseded by commit
        75058820's snapshot mechanism: a hazardous read is now redirected
        to a pre-write snapshot of the carry's old value instead of being
        rejected, since online-softmax's own `correction = exp(m - m_new)`
        legitimately needs both m's old and new values in the same pass
        (see test_carry_mode_online_softmax). This test covers the
        detection step in isolation, at the mock level; the fixture above
        is the end-to-end proof the resulting snapshot is numerically
        correct.
        """
        import torch_spyre._inductor.wsr.while_loop_bridge as bridge

        # A second, unrelated op that reads the mutated carry's own
        # per-iteration output after its producer has already run.
        rogue_reader = mock.Mock(
            name="rogue_reader",
            spec=["get_operation_name", "get_name", "get_read_writes"],
        )
        rogue_reader.get_operation_name.return_value = "rogue_reader"
        rogue_reader.get_name.return_value = None
        rogue_dep = mock.Mock(name="rogue_dep", spec=["name"])
        rogue_dep.name = "while_loop_body_graph_0_0_arg0_1"
        rogue_reader.get_read_writes.return_value = mock.Mock(reads=[rogue_dep])

        producer = mock.Mock(name="producer", spec=["get_operation_name", "get_name"])
        producer.get_operation_name.return_value = "producer"
        producer.get_name.return_value = "while_loop_body_graph_0_0_buf7"

        extra_readers = bridge._extra_readers_of_placeholder(
            "while_loop_body_graph_0_0_arg0_1",
            "while_loop_body_graph_0_0_buf7",
            [producer, rogue_reader],
        )

        self.assertEqual(extra_readers, [rogue_reader])

    def test_read_before_producer_is_not_an_extra_reader(self):
        """A read of the placeholder that happens at-or-before the
        producer's own position is safe by program order (it sees the OLD
        value, same as the producer computing the new one from it) -- only
        reads AFTER the producer are write-after-read hazards.
        """
        import torch_spyre._inductor.wsr.while_loop_bridge as bridge

        producer = mock.Mock(name="producer", spec=["get_operation_name", "get_name"])
        producer.get_operation_name.return_value = "producer"
        producer.get_name.return_value = "while_loop_body_graph_0_0_buf7"

        safe_reader = mock.Mock(
            name="safe_reader",
            spec=["get_operation_name", "get_name", "get_read_writes"],
        )
        safe_reader.get_operation_name.return_value = "safe_reader"
        safe_reader.get_name.return_value = None
        safe_dep = mock.Mock(name="safe_dep", spec=["name"])
        safe_dep.name = "while_loop_body_graph_0_0_arg0_1"
        safe_reader.get_read_writes.return_value = mock.Mock(reads=[safe_dep])

        extra_readers = bridge._extra_readers_of_placeholder(
            "while_loop_body_graph_0_0_arg0_1",
            "while_loop_body_graph_0_0_buf7",
            [safe_reader, producer],
        )

        self.assertEqual(extra_readers, [])

    def test_get_read_writes_failure_raises_unsupported(self):
        """A post-producer op whose get_read_writes() raises must not be
        silently treated as "doesn't read the placeholder" -- that could
        mask a real write-after-read hazard. See Unsupported's use here.
        """
        import torch_spyre._inductor.wsr.while_loop_bridge as bridge
        from torch_spyre._inductor.errors import Unsupported

        producer = mock.Mock(name="producer", spec=["get_operation_name", "get_name"])
        producer.get_operation_name.return_value = "producer"
        producer.get_name.return_value = "while_loop_body_graph_0_0_buf7"

        broken_reader = mock.Mock(
            name="broken_reader",
            spec=["get_operation_name", "get_name", "get_read_writes"],
        )
        broken_reader.get_operation_name.return_value = "broken_reader"
        broken_reader.get_name.return_value = None
        broken_reader.get_read_writes.side_effect = RuntimeError("boom")

        with self.assertRaises(Unsupported):
            bridge._extra_readers_of_placeholder(
                "while_loop_body_graph_0_0_arg0_1",
                "while_loop_body_graph_0_0_buf7",
                [producer, broken_reader],
            )


def _find_while_loop_ir_op(fn, args):
    """Compile fn(*args) under GraphLowering and return the WhileLoop ir.Operation.

    Uses the same capture_post_grad_while_loop entry point the fixture
    module offers, then re-lowers the returned FX graph module through a
    fresh GraphLowering to reach the ir.Operation level this prover
    operates on (mirrors how CustomPreSchedulingPasses receives graph.operations).

    GraphLowering.run() requires an active V.fake_mode with a real
    ShapeEnv (WhileLoop.create's unbacked-symbol renaming touches
    V.fake_mode.shape_env.unbacked_renamings unconditionally) -- the
    fake_mode/shape_env the original torch.compile trace already attached
    to this graph module's own node.meta["val"] fake tensors is reused here,
    since any unbacked symbols the graph already refers to only exist in
    that original shape_env.
    """
    from torch._inductor.graph import GraphLowering
    from torch._inductor import ir

    _out, gm = capture_post_grad_while_loop(fn, args)

    fake_mode = None
    for node in gm.graph.nodes:
        val = node.meta.get("val") if hasattr(node, "meta") else None
        candidate = getattr(val, "fake_mode", None)
        if candidate is not None:
            fake_mode = candidate
            break
    assert fake_mode is not None, "could not recover a fake_mode from gm node.meta"

    graph = GraphLowering(gm, example_inputs=list(args), shape_env=fake_mode.shape_env)
    with V.set_graph_handler(graph), V.set_fake_mode(fake_mode):
        graph.run(*args)
    while_ops = [op for op in graph.operations if isinstance(op, ir.WhileLoop)]
    assert len(while_ops) == 1, f"expected exactly one WhileLoop, got {len(while_ops)}"
    return while_ops[0]


class TestSpliceWhileLoops(unittest.TestCase):
    def _run_graph(self, fn, args):
        """Lower fn(*args) through a fresh GraphLowering and return it.

        Mirrors _find_while_loop_ir_op's fake_mode/shape_env recovery above:
        GraphLowering.run() requires an active V.fake_mode with a real
        ShapeEnv (WhileLoop.create's unbacked-symbol renaming touches
        V.fake_mode.shape_env.unbacked_renamings unconditionally), so the
        fake_mode the original torch.compile trace attached to this graph
        module's own node.meta["val"] fake tensors is reused here.
        """
        from torch._inductor.graph import GraphLowering

        _out, gm = capture_post_grad_while_loop(fn, args)

        fake_mode = None
        for node in gm.graph.nodes:
            val = node.meta.get("val") if hasattr(node, "meta") else None
            candidate = getattr(val, "fake_mode", None)
            if candidate is not None:
                fake_mode = candidate
                break
        assert fake_mode is not None, "could not recover a fake_mode from gm node.meta"

        graph = GraphLowering(
            gm, example_inputs=list(args), shape_env=fake_mode.shape_env
        )
        with V.set_graph_handler(graph), V.set_fake_mode(fake_mode):
            graph.run(*args)
        return graph

    def test_map_mode_group_gets_loop_info(self):
        from torch._inductor import ir
        from torch._inductor.virtualized import V

        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            splice_while_loops,
        )

        (X, Y), _ref = matmul_inputs()
        graph = self._run_graph(split_m_fn, (X, Y))
        with V.set_graph_handler(graph):
            self.assertTrue(
                any(isinstance(op, ir.WhileLoop) for op in graph.operations)
            )

            splice_while_loops(graph)

            self.assertFalse(
                any(isinstance(op, ir.WhileLoop) for op in graph.operations)
            )
            tiled_ops = [
                op for op in graph.operations if getattr(op, "loop_info", None)
            ]
            self.assertTrue(
                tiled_ops, "expected at least one op with loop_info stamped"
            )
            for op in tiled_ops:
                self.assertTrue(op.dim_hints, f"{op} missing synthesized dim_hints")

    def test_carry_mode_group_gets_loop_info(self):
        from torch._inductor import ir
        from torch._inductor.virtualized import V

        from torch_spyre._inductor.loop_info import LoopCarryRecord
        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            splice_while_loops,
        )

        (X, Y), _ref = matmul_inputs()
        graph = self._run_graph(split_k_fn, (X, Y))
        with V.set_graph_handler(graph):
            splice_while_loops(graph)

            self.assertFalse(
                any(isinstance(op, ir.WhileLoop) for op in graph.operations)
            )
            tiled_ops = [
                op for op in graph.operations if getattr(op, "loop_info", None)
            ]
            self.assertTrue(
                tiled_ops, "expected at least one op with loop_info stamped"
            )
            for op in tiled_ops:
                self.assertTrue(op.dim_hints, f"{op} missing synthesized dim_hints")

            records_by_name = {
                op.get_name(): record
                for op in graph.operations
                if isinstance(
                    record := getattr(op, "_loop_carry_record", None),
                    LoopCarryRecord,
                )
            }
            storage_records = {
                name: record
                for name, record in records_by_name.items()
                if record.storage_name == name
            }
            self.assertTrue(storage_records, "expected loop-carry storage metadata")
            for storage_name, record in storage_records.items():
                self.assertIn(record.update_name, records_by_name)
                self.assertIs(records_by_name[record.update_name], record)
                self.assertEqual(records_by_name[storage_name], record)


class TestTryProveForEachTile(unittest.TestCase):
    def test_map_mode_accepted_with_trip_count(self):
        (X, Y), _ref = matmul_inputs()
        while_op = _find_while_loop_ir_op(split_m_fn, (X, Y))

        result = try_prove_for_each_tile(while_op)

        self.assertTrue(result.accepted, result.reason)
        self.assertIsNotNone(result.trip_count)

    def test_carry_mode_accepted_with_trip_count(self):
        (X, Y), _ref = matmul_inputs()
        while_op = _find_while_loop_ir_op(split_k_fn, (X, Y))

        result = try_prove_for_each_tile(while_op)

        self.assertTrue(result.accepted, result.reason)
        self.assertIsNotNone(result.trip_count)

    def test_declines_non_matching_shape(self):
        while_op = mock.Mock()
        while_op.cond_subgraph.graph.operations = []
        while_op.cond_subgraph.graph.graph_outputs = []

        result = try_prove_for_each_tile(while_op)

        self.assertFalse(result.accepted)
        self.assertTrue(result.reason)


class TestPassPipelineRegistration(unittest.TestCase):
    def test_splice_while_loops_is_first_pass(self):
        from torch_spyre._inductor.passes import CustomPreSchedulingPasses
        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            splice_while_loops,
        )

        pipeline = CustomPreSchedulingPasses()

        self.assertIs(pipeline.passes[0], splice_while_loops)


def test_tile_dim_marker_lowering_produces_distinct_operation():
    """lower_tile_dim_marker must NOT elide -- confirms the Pointwise.create
    fallback (spec Section 7) actually forces a distinct ir.Operation, unlike
    the bare-identity lowering this replaces. Captures the GraphLowering
    directly via a GraphLowering.run monkeypatch (mirroring _post_grad_graphs'
    own monkeypatch-capture style in for_each_tile_fixtures.py) rather than
    TestSpliceWhileLoops._run_graph, which requires an actual scan/while_loop
    shape this bare op call does not have.

    Adaptation from the original spec: wraps the compile in
    `torch._inductor.config.patch("force_disable_caches", True)`. Without
    it, a second run of this exact test (fxgraph cache warm from a prior
    run) hits FxGraphCache and skips GraphLowering.run entirely, making the
    `captured` list empty and the test fail nondeterministically depending
    on cache state -- not a redesign, just the same cache-disable pattern
    already used elsewhere in this test suite (e.g. test_padding.py,
    test_dedup_constants.py, test_inductor_fx_passes.py).
    """
    import torch
    from torch._inductor import config as t_inductor_config

    from torch._inductor.graph import GraphLowering

    import torch_spyre  # noqa: F401  (registers the spyre device + lowerings)
    from torch_spyre.constants import DEVICE_NAME

    captured: list[GraphLowering] = []
    original_run = GraphLowering.run

    def capturing_run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        captured.append(self)
        return result

    def fn(x):
        return torch.ops.spyre.tile_dim_marker(x, 1)

    X = torch.randn(4, 8, device=DEVICE_NAME)
    with (
        t_inductor_config.patch("force_disable_caches", True),
        mock.patch.object(GraphLowering, "run", capturing_run),
    ):
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        compiled(X)

    assert captured, "GraphLowering.run was never invoked"
    graph = captured[0]
    marker_ops = [
        op
        for op in graph.operations
        if getattr(op, "tile_marker_dim", None) is not None
    ]
    assert len(marker_ops) == 1, (
        f"expected exactly one op carrying tile_marker_dim, found "
        f"{len(marker_ops)} in {[type(o) for o in graph.operations]}"
    )
    assert marker_ops[0].tile_marker_dim == 1


class TestConsumeTileDimMarkers(unittest.TestCase):
    """_consume_tile_dim_markers: marker map + marker erasure."""

    def _run_graph(self, fn, args):
        """Lower fn(*args) through a fresh GraphLowering and return it.

        Same pattern as TestSpliceWhileLoops._run_graph: calling
        GraphLowering.run() directly on a standalone instance (rather than
        driving a full torch.compile) stops short of codegen(), so
        splice_while_loops -- a pre-scheduling pass that only runs from
        _update_scheduler during codegen() -- never fires. That leaves the
        WhileLoop op intact in graph.operations for this test to splice
        itself and inspect the intermediate (post-splice, pre-marker-
        consumption) state, which a full torch.compile capture cannot do:
        by the time such a capture's own GraphLowering.run returns, the
        real pipeline has already spliced AND consumed markers on the same
        graph object, leaving no WhileLoop for the test to find.
        """
        from torch._inductor.graph import GraphLowering

        from tests.inductor.for_each_tile_fixtures import capture_post_grad_while_loop

        _out, gm = capture_post_grad_while_loop(fn, args)

        fake_mode = None
        for node in gm.graph.nodes:
            val = node.meta.get("val") if hasattr(node, "meta") else None
            candidate = getattr(val, "fake_mode", None)
            if candidate is not None:
                fake_mode = candidate
                break
        assert fake_mode is not None, "could not recover a fake_mode from gm node.meta"

        graph = GraphLowering(
            gm, example_inputs=list(args), shape_env=fake_mode.shape_env
        )
        with V.set_graph_handler(graph), V.set_fake_mode(fake_mode):
            graph.run(*args)
        return graph

    def test_marker_erased_and_mapped_after_split_m_splice(self):
        """Marker resolution for split_m_fn's StarDep-shaped matmul consumer.

        split_m_fn's marker's sole consumer is a matmul -- an aten-fallback
        ExternKernelOut even on this device-less CPU fixture, i.e. the
        StarDep/_substitute_direct_input_refs branch (see
        test_marker_inlined_preserves_advance_term_on_computed_buffer_
        consumer's docstring below, which pins the *other* branch
        specifically because this test doesn't reach it). Per
        _consume_tile_dim_markers's own docstring, that branch
        deliberately keeps the marker materialized in graph.operations
        (never erased) rather than erasing it the way the ComputedBuffer/
        inline branch does -- a StarDep consumer has no inner_fn to fuse
        the marker's per-iteration transform into, so the marker must
        remain a real, addressable buffer for it to read. This test's name
        predates that fix and is now a slight misnomer (nothing here is
        erased); kept for continuity with its git history rather than
        renamed.
        """
        from torch._inductor import ir

        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            _body_loop_var,
            _consume_tile_dim_markers,
            _stacking_carry_indices,
            try_prove_for_each_tile,
        )
        from torch_spyre._inductor.wsr.while_loop_bridge import (
            carry_bindings_for,
            splice_while_loop,
        )

        (X, Y), _ref = matmul_inputs()
        graph = self._run_graph(split_m_fn, (X, Y))

        while_ops = [op for op in graph.operations if isinstance(op, ir.WhileLoop)]
        self.assertEqual(len(while_ops), 1)
        while_op = while_ops[0]

        result = try_prove_for_each_tile(while_op)
        self.assertTrue(result.accepted)
        loop_var = _body_loop_var(while_op)
        self.assertIsNotNone(loop_var)

        with V.set_graph_handler(graph):
            carries = carry_bindings_for(
                while_op, _stacking_carry_indices(while_op, loop_var)
            )
            group_ops = splice_while_loop(
                graph, while_op, carries, trip_count=result.trip_count
            )

            # lower_tile_dim_marker (lowering.py) stamps tile_marker_dim
            # directly on the realized ComputedBuffer -- the exact object
            # that ends up as this `op` in group_ops/graph.operations -- not
            # on `op.data` (that is one level deeper: the Pointwise/
            # Reduction IR expression, which never carries it). Matches the
            # getattr(op, "tile_marker_dim", None) convention
            # TestLowerTileDimMarker already uses above.
            marker_dims_before = [
                op.tile_marker_dim
                for op in group_ops
                if getattr(op, "tile_marker_dim", None) is not None
            ]
            self.assertTrue(
                marker_dims_before, "expected at least one tile_marker_dim-tagged op"
            )

            marker_map = _consume_tile_dim_markers(group_ops, graph.operations)

            self.assertTrue(
                marker_map, "expected a non-empty (consumer_op, dep) -> dim map"
            )
            self.assertTrue(
                all(isinstance(dim, int) for dim in marker_map.values()),
            )

            # split_m_fn's matmul is StarDep-shaped (see this test's own
            # docstring above), so the marker(s) deliberately survive in
            # graph.operations rather than being erased -- assert they are
            # still present, still tagged with their original dims, and
            # (per _validate_contiguous's gapless-contiguity requirement,
            # see _consume_tile_dim_markers's own comment on this) still
            # members of group_ops too.
            remaining_markers = [
                op
                for op in graph.operations
                if getattr(op, "tile_marker_dim", None) is not None
            ]
            self.assertEqual(
                sorted(op.tile_marker_dim for op in remaining_markers),
                sorted(marker_dims_before),
                "StarDep-shaped marker consumers keep their markers "
                "materialized in graph.operations (never erased) -- see "
                "_consume_tile_dim_markers's own docstring",
            )
            self.assertTrue(
                all(op in group_ops for op in remaining_markers),
                "surviving markers must remain members of group_ops too, "
                "or _validate_contiguous's gapless-contiguity check on "
                "this group would fail later",
            )

    def test_marker_inlined_preserves_advance_term_on_computed_buffer_consumer(self):
        """Pins the ComputedBuffer/inliner branch specifically.

        test_marker_erased_and_mapped_after_split_m_splice (above) only
        exercises split_m_fn, whose marker's sole consumer is a matmul --
        an aten-fallback ExternKernelOut even on this device-less CPU
        fixture, i.e. the StarDep/_substitute_direct_input_refs erasure
        branch, NOT the ComputedBuffer/_inline_marker_into_consumer branch.
        Real (device-backed) for_each_tile bodies commonly read a
        marker-tagged tile from a Pointwise/Reduction ComputedBuffer
        instead (e.g. test_carry_mode_online_softmax's ``k_tile.transpose
        (-1, -2)``-fed matmul operand, or any elementwise op on a tile) --
        that branch was, until this test, exercised only by e2e numeric
        assertions on the real Spyre device, with nothing at the unit
        level able to catch a regression to it.

        split_m_elementwise_fn's ``x_tile * 2.0`` gives a genuine Pointwise
        ComputedBuffer consumer of the marker even on CPU (matmul, unlike
        elementwise ops, always aten-falls-back). This test pins the exact
        regression a future "simplify back to a plain rename" would
        reintroduce (see _consume_tile_dim_markers's own docstring): the
        marker's own inner_fn contributes a genuine, non-identity
        per-iteration advance term to its read index (the ``+ 24*u0`` tile-
        slice offset) that a bare NameSwapHandler rename would silently
        drop. Assert that term is actually still present, syntactically, on
        the consumer's post-erasure read -- not just that erasure happened
        at all (which test_marker_erased_and_mapped_after_split_m_splice
        already covers, and which a buggy rename would ALSO satisfy).
        """
        from torch._inductor import ir
        from torch._inductor.dependencies import MemoryDep

        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            _body_loop_var,
            _consume_tile_dim_markers,
            _stacking_carry_indices,
            try_prove_for_each_tile,
        )
        from torch_spyre._inductor.wsr.while_loop_bridge import (
            carry_bindings_for,
            splice_while_loop,
        )

        (X, Y), _ref = matmul_inputs()
        graph = self._run_graph(split_m_elementwise_fn, (X, Y))

        while_ops = [op for op in graph.operations if isinstance(op, ir.WhileLoop)]
        self.assertEqual(len(while_ops), 1)
        while_op = while_ops[0]

        result = try_prove_for_each_tile(while_op)
        self.assertTrue(result.accepted)
        loop_var = _body_loop_var(while_op)
        self.assertIsNotNone(loop_var)

        with V.set_graph_handler(graph):
            carries = carry_bindings_for(
                while_op, _stacking_carry_indices(while_op, loop_var)
            )
            group_ops = splice_while_loop(
                graph, while_op, carries, trip_count=result.trip_count
            )

            marker_op = next(
                op
                for op in group_ops
                if getattr(op, "tile_marker_dim", None) is not None
            )
            marker_name = marker_op.get_name()
            self.assertIsInstance(marker_op, ir.ComputedBuffer)

            # The marker's OWN read index (of its real upstream input) is
            # the ground truth this test expects to survive erasure intact.
            marker_reads = [
                d for d in marker_op.get_read_writes().reads if isinstance(d, MemoryDep)
            ]
            self.assertEqual(len(marker_reads), 1)
            marker_input_name = marker_reads[0].name
            marker_own_index = marker_reads[0].index
            self.assertIn(
                loop_var,
                marker_own_index.free_symbols,
                "fixture assumption violated: expected the marker's own "
                "read index to carry the per-iteration advance term "
                f"({loop_var}); got {marker_own_index!r}",
            )

            # The consumer (split_m_elementwise_fn's `x_tile * 2.0`) must
            # be a real ComputedBuffer -- confirming this test actually
            # reaches _inline_marker_into_consumer, not
            # _substitute_direct_input_refs.
            consumer_before = next(
                op
                for op in group_ops
                if isinstance(op, ir.ComputedBuffer)
                and op is not marker_op
                and any(
                    isinstance(d, MemoryDep) and d.name == marker_name
                    for d in op.get_read_writes().reads
                )
            )
            consumer_name = consumer_before.get_name()

            _consume_tile_dim_markers(group_ops, graph.operations)

            new_consumer = next(
                op for op in graph.operations if op.get_name() == consumer_name
            )
            self.assertIsInstance(new_consumer, ir.ComputedBuffer)
            post_reads = [
                d
                for d in new_consumer.get_read_writes().reads
                if isinstance(d, MemoryDep) and d.name == marker_input_name
            ]
            self.assertEqual(
                len(post_reads),
                1,
                "expected exactly one post-erasure read of the marker's "
                f"own upstream input {marker_input_name!r}",
            )
            post_index = post_reads[0].index

            # The pin: the composed index must still carry loop_var's
            # coefficient from the marker's OWN index, unchanged -- proof
            # the marker's real per-iteration coordinate transform was
            # composed in, not merely renamed past. A plain
            # NameSwapHandler-style rename would instead reuse the
            # consumer's OWN pre-erasure (marker-relative) index, which
            # never mentioned loop_var at all -- that bug would make this
            # specific assertion fail while still passing
            # test_marker_erased_and_mapped_after_split_m_splice's weaker
            # "marker map is non-empty and erasure happened" checks.
            self.assertIn(
                loop_var,
                post_index.free_symbols,
                "post-erasure consumer read lost the marker's own "
                f"per-iteration advance term ({loop_var}) -- got "
                f"{post_index!r}. This is the exact silent-wrong-answer "
                "regression a plain rename-based marker erasure would "
                "reintroduce.",
            )
            self.assertEqual(
                post_index.coeff(loop_var),
                marker_own_index.coeff(loop_var),
                "post-erasure consumer read's loop_var coefficient must "
                "match the marker's own index's loop_var coefficient "
                "exactly -- the composed index should carry the marker's "
                "real coordinate transform through unchanged, not some "
                "other (e.g. renamed-and-unchanged, or miscomposed) value.",
            )

    def test_split_k_marker_resolves_reduction_dim(self):
        """IR-level value assertion for split_k_fn's reduction-dim marker.

        Uses self._run_graph (this class's own established pattern, see its
        docstring above) rather than a full torch.compile capture: driving
        split_k_fn through real codegen hits a pre-existing, out-of-scope
        read-copy/stick-layout gap in propagate_layouts.py (tracked as issue
        #4460, see test_carry_mode_split_k's own XFAIL docstring in
        test_for_each_tile_e2e.py) that has nothing to do with marker
        resolution.

        split_k_fn's matmul lowers on this CPU fixture as an aten-fallback
        ExternKernelOut -- a StarDep-shaped consumer, same as split_m_fn's
        matmul (see test_marker_erased_and_mapped_after_split_m_splice
        above). lookup_marker_dim deliberately returns None for a
        StarDep-mapped entry (see its own docstring: StarDep has no
        .index/.ranges to resolve a consumer-space position from, and its
        only real caller, _hint_ranges_pos, already filters out every
        non-ComputedBuffer op before calling it) -- so the reduction-dim
        value assertion belongs on _consume_tile_dim_markers's own marker
        map, which is what actually records K's position for this op
        shape, not on lookup_marker_dim's consumer-space remapping.
        """
        from torch._inductor import ir
        from torch._inductor.dependencies import StarDep

        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            _body_loop_var,
            _consume_tile_dim_markers,
            _marker_dim,
            _stacking_carry_indices,
            try_prove_for_each_tile,
        )
        from torch_spyre._inductor.wsr.while_loop_bridge import (
            carry_bindings_for,
            splice_while_loop,
        )

        import torch
        from tests.inductor.for_each_tile_fixtures import M, K, N

        X, Y = torch.randn(M, K), torch.randn(K, N)
        graph = self._run_graph(split_k_fn, (X, Y))

        while_ops = [op for op in graph.operations if isinstance(op, ir.WhileLoop)]
        self.assertEqual(len(while_ops), 1)
        while_op = while_ops[0]

        result = try_prove_for_each_tile(while_op)
        self.assertTrue(result.accepted)
        loop_var = _body_loop_var(while_op)
        self.assertIsNotNone(loop_var)

        with V.set_graph_handler(graph):
            carries = carry_bindings_for(
                while_op, _stacking_carry_indices(while_op, loop_var)
            )
            group_ops = splice_while_loop(
                graph, while_op, carries, trip_count=result.trip_count
            )

            # split_k_fn's matmul is a StarDep-shaped consumer (see this
            # test's own docstring above), so _consume_tile_dim_markers
            # keeps both markers materialized rather than erasing them
            # (its own "StarDep-shaped consumer" branch comment) and
            # redirects the matmul's input references to the markers
            # themselves. The marker map is therefore keyed by StarDeps
            # naming the MARKERS (buffers still present in group_ops post-
            # call), not by the raw arg0_1/arg1_1 input names those
            # markers used to be erased down to -- capture the marker names
            # before the call, since group_ops is mutated in place.
            markers_before = {
                op.get_name(): _marker_dim(op)
                for op in group_ops
                if _marker_dim(op) is not None
            }
            self.assertEqual(
                len(markers_before), 2, "expected exactly one marker per matmul input"
            )

            marker_map = _consume_tile_dim_markers(group_ops, graph.operations)

            matmul_op = next(
                op for op in group_ops if isinstance(op, ir.ExternKernelOut)
            )
            matmul_name = matmul_op.get_name()

            x_marker_name, y_marker_name = None, None
            for name, dim in markers_before.items():
                if dim == 1:
                    x_marker_name = name
                elif dim == 0:
                    y_marker_name = name

            x_dim = marker_map.get((matmul_name, StarDep(name=x_marker_name)))
            y_dim = marker_map.get((matmul_name, StarDep(name=y_marker_name)))
            self.assertEqual(
                x_dim,
                1,
                "split_k_fn tiles X along dims=(-1, ...) -- dim 1 of X's "
                "[M, K] shape, the reduction (K) dim",
            )
            self.assertEqual(
                y_dim,
                0,
                "split_k_fn tiles Y along dims=(..., 0) -- dim 0 of Y's "
                "[K, N] shape, the reduction (K) dim",
            )

    def test_lookup_marker_dim_resolves_reduction_dim(self):
        """lookup_marker_dim's is_reduction=True return path (F4.1).

        No existing fixture in this file reaches this path: every CPU
        fixture's marker consumer is either a StarDep-shaped matmul
        (split_m_fn/split_k_fn -- lookup_marker_dim returns None
        immediately for those, see test_split_k_marker_resolves_
        reduction_dim's own docstring) or a Pointwise ComputedBuffer
        (split_m_elementwise_fn), never a Reduction ComputedBuffer.
        Confirmed directly against online_softmax_fn (whose amax/sum
        reductions read tiles derived FROM the marker's consumer, not the
        marker itself): both of its two real markers still resolve to a
        StarDep-shaped matmul consumer on this CPU fixture, same as split_m_
        fn/split_k_fn -- the reduction-dim resolution path genuinely has no
        CPU-fixture-driven route in this file, so this test builds the
        minimal IR directly instead (per the fix-wave brief's own
        guidance), using this class's established mock.Mock(spec=[...])
        convention (see TestSpliceWhileLoop above).

        Mirrors _hint_ranges_pos's own resolution order: op.data is a real
        Reduction (isinstance-checked; mock.Mock(spec=Reduction) passes
        that check), and the marker-identified MemoryDep's index (``r0 +
        3*u0``) is built so u0's (the WhileLoop-splice loop_var) advance
        coefficient (3) coincidentally matches r0's own coefficient (1)
        times r0's range (3) -- the exact coefficient-coincidence equation
        lookup_marker_dim's docstring describes. op_out_coords is
        monkeypatched to a coordinate list that does NOT mention r0, so
        _loop_var_to_ranges_pos misses (r0 is not an output dim) and
        resolution must fall through to the reduction_loop_vars branch --
        which is the one thing this test actually pins: reduction_loop_vars
        derives r0 as op's sole reduction var (it is on the read but not on
        the write), so the candidate's `is_reduction` flag comes out True,
        not from a hint the test injects directly.
        """
        import sympy

        from torch._inductor.dependencies import MemoryDep
        from torch._inductor.ir import Reduction

        import torch_spyre._inductor.wsr.coarse_tile as coarse_tile_mod
        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            _MARKER_MAPS,
            clear_marker_maps,
            lookup_marker_dim,
        )

        r0, u0 = sympy.symbols("r0 u0")

        # The marker's own read: op's per-iteration advance (u0, coeff 3)
        # coincides with r0's own coefficient (1) times r0's range (3).
        marker_dep = MemoryDep(
            name="marker_buf",
            index=r0 + 3 * u0,
            var_names=(r0,),
            size=(sympy.Integer(3),),
        )
        # op's write does not mention r0 -- r0 is a pure reduction var.
        out_dep = MemoryDep(
            name="reduction_op",
            index=sympy.Symbol("d0"),
            var_names=(sympy.Symbol("d0"),),
            size=(sympy.Integer(1),),
        )

        op = mock.Mock(spec=["get_read_writes", "get_name", "data"])
        op.get_name.return_value = "reduction_op"
        op.data = mock.Mock(spec=Reduction)
        rw = mock.Mock()
        rw.reads = [marker_dep]
        rw.writes = {out_dep}
        op.get_read_writes.return_value = rw

        operations = ["sentinel_operations_list"]
        clear_marker_maps()
        self.addCleanup(clear_marker_maps)
        _MARKER_MAPS[id(operations)] = {("reduction_op", marker_dep): 0}

        graph = mock.Mock(spec=["operations"])
        graph.operations = operations

        with mock.patch.object(
            coarse_tile_mod, "op_out_coords", return_value=[sympy.Integer(0)]
        ):
            with V.set_graph_handler(graph):
                result = lookup_marker_dim(op, u0)

        self.assertEqual(
            result,
            (0, True),
            "expected lookup_marker_dim to resolve u0 to reduction "
            "position 0 (is_reduction=True) via r0, the sole var in the "
            "marker-identified dep's ranges that satisfies the "
            "coefficient-coincidence equation",
        )

        # Non-vacuity check: with the marker map entry removed (simulating
        # a marker that was never recorded), lookup_marker_dim must return
        # None instead -- confirming the True-path result above is not
        # returned unconditionally.
        _MARKER_MAPS[id(operations)] = {}
        with mock.patch.object(
            coarse_tile_mod, "op_out_coords", return_value=[sympy.Integer(0)]
        ):
            with V.set_graph_handler(graph):
                empty_map_result = lookup_marker_dim(op, u0)
        self.assertIsNone(
            empty_map_result,
            "expected lookup_marker_dim to return None once the marker "
            "map entry is removed -- if this fails, the reduction-dim "
            "resolution above wasn't actually reading from the map",
        )

    def test_hint_ranges_pos_raises_on_unmarked_loop_var_gap(self):
        """_hint_ranges_pos's raise-on-gap guard (F4.2).

        Markers are authoritative and there is no fallback heuristic: an
        op whose read index genuinely mentions a WhileLoop-splice
        loop_var, but which resolves to no output dim, no reduction dim,
        AND no marker-map entry, is an unrecognized shape that must raise
        loudly (coarse_tile.py's _hint_ranges_pos, ~line 2136-2149) rather
        than silently return a guessed/absent position. This safety
        argument -- the entire justification in this branch for deleting
        the old _loop_var_pos_from_reads heuristic with no fallback -- had
        no test coverage anywhere in this file before this test.

        Builds the minimal op/hint directly (this class's established
        mock.Mock(spec=[...]) convention): op's own read index genuinely
        mentions loop_var u0, op.data is not a Reduction (so the reduction-
        ranges branch quickly misses), _loop_var_to_ranges_pos is
        monkeypatched to always miss (op is not tiled along an output dim
        for this hint), and clear_marker_maps() ensures lookup_marker_dim
        has no map entry to resolve against either -- exhausting every
        resolution channel _hint_ranges_pos tries before its final
        raise-on-gap check.
        """
        import sympy

        from torch._inductor.dependencies import MemoryDep

        import torch_spyre._inductor.wsr.coarse_tile as coarse_tile_mod
        from torch_spyre._inductor.propagate_hints import DimHint
        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            clear_marker_maps,
        )

        u0 = sympy.Symbol("u0")
        d0 = sympy.Symbol("d0")

        # op's own read index genuinely mentions loop_var u0 -- the one
        # condition that must be true for a miss to raise rather than
        # return (None, False) as a legitimate loop-invariant op would.
        read_dep = MemoryDep(
            name="some_buf", index=u0 + d0, var_names=(d0,), size=(sympy.Integer(4),)
        )
        op = mock.Mock(spec=["get_read_writes", "get_name", "data"])
        op.get_name.return_value = "gap_op"
        op.data = mock.Mock()  # deliberately not a Reduction instance
        rw = mock.Mock()
        rw.reads = [read_dep]
        rw.writes = {mock.Mock(index=sympy.Integer(0))}
        op.get_read_writes.return_value = rw

        hint = DimHint(
            dim_names=["u0"],
            split_count=1,
            loop_var=u0,
            is_reduction=False,
            loop_var_range=2,
        )

        clear_marker_maps()
        self.addCleanup(clear_marker_maps)

        graph = mock.Mock(spec=["operations"])
        graph.operations = ["sentinel_operations_list"]

        with mock.patch.object(
            coarse_tile_mod, "_loop_var_to_ranges_pos", return_value=None
        ):
            with V.set_graph_handler(graph):
                with self.assertRaises(
                    AssertionError,
                    msg="expected _hint_ranges_pos to raise when "
                    "loop_var appears in op's own read index but resolves "
                    "through no channel (output dim, reduction dim, or "
                    "marker map)",
                ):
                    coarse_tile_mod._hint_ranges_pos(
                        op, hint, out_coords=[sympy.Integer(0)]
                    )

        # Non-vacuity check: when the read index does NOT mention loop_var
        # at all (a genuinely loop-invariant op), _hint_ranges_pos must NOT
        # raise -- it must return (None, False) instead. This confirms the
        # raise above is conditioned on "loop_var appears in the read
        # index", not unconditional.
        invariant_dep = MemoryDep(
            name="some_buf", index=d0, var_names=(d0,), size=(sympy.Integer(4),)
        )
        invariant_op = mock.Mock(spec=["get_read_writes", "get_name", "data"])
        invariant_op.get_name.return_value = "invariant_op"
        invariant_op.data = mock.Mock()
        invariant_rw = mock.Mock()
        invariant_rw.reads = [invariant_dep]
        invariant_rw.writes = {mock.Mock(index=sympy.Integer(0))}
        invariant_op.get_read_writes.return_value = invariant_rw

        with mock.patch.object(
            coarse_tile_mod, "_loop_var_to_ranges_pos", return_value=None
        ):
            with V.set_graph_handler(graph):
                result = coarse_tile_mod._hint_ranges_pos(
                    invariant_op, hint, out_coords=[sympy.Integer(0)]
                )
        self.assertEqual(
            result,
            (None, False),
            "expected a genuinely loop-invariant op (whose read index "
            "never mentions loop_var) to resolve to (None, False) without "
            "raising -- if this fails, the raise above isn't actually "
            "conditioned on the mentions_loop_var check",
        )

    def test_consume_tile_dim_markers_raises_on_wrong_consumer_count(self):
        """_consume_tile_dim_markers's consumer-count guard (F4.3).

        A tile_dim_marker op is expected to have exactly one consuming
        read within its spliced body (for_each_tile_lowering.py,
        ~line 762-769). Zero consumers or more than one are both
        unrecognized shapes and must raise AssertionError rather than
        silently pick a default -- untested anywhere in this file before
        this test. Covers both wrong-count shapes in one test (zero, then
        two), each built directly via this class's established
        mock.Mock(spec=[...]) convention (see TestSpliceWhileLoop above)
        rather than a full torch.compile, since driving a real for_each_
        tile fixture into either of these malformed shapes is not
        possible through the ordinary lowering path -- both are meant to
        be unreachable there, which is exactly why they need direct
        construction to exercise at all.
        """
        import sympy

        from torch._inductor.dependencies import MemoryDep
        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            _consume_tile_dim_markers,
        )

        def make_marker_op():
            marker_op = mock.Mock(
                name="marker_op",
                spec=["get_read_writes", "get_name", "tile_marker_dim"],
            )
            marker_op.get_name.return_value = "marker_buf"
            marker_op.tile_marker_dim = 0
            marker_rw = mock.Mock()
            marker_rw.reads = [
                MemoryDep(
                    name="input_buf", index=sympy.Integer(0), var_names=(), size=()
                )
            ]
            marker_op.get_read_writes.return_value = marker_rw
            return marker_op

        def make_consumer(name):
            consumer_op = mock.Mock(name=name, spec=["get_read_writes", "get_name"])
            consumer_op.get_name.return_value = name
            consumer_rw = mock.Mock()
            consumer_rw.reads = [
                MemoryDep(
                    name="marker_buf", index=sympy.Integer(0), var_names=(), size=()
                )
            ]
            consumer_op.get_read_writes.return_value = consumer_rw
            return consumer_op

        # Zero consumers: the marker op has no consuming read at all.
        zero_marker_op = make_marker_op()
        unrelated_op = mock.Mock(
            name="unrelated_op", spec=["get_read_writes", "get_name"]
        )
        unrelated_op.get_name.return_value = "unrelated_op"
        unrelated_rw = mock.Mock()
        unrelated_rw.reads = []
        unrelated_op.get_read_writes.return_value = unrelated_rw

        zero_group_ops = [zero_marker_op, unrelated_op]
        zero_operations = list(zero_group_ops)
        zero_graph = mock.Mock(spec=["operations"])
        zero_graph.operations = zero_operations

        with V.set_graph_handler(zero_graph):
            with self.assertRaisesRegex(AssertionError, r"has 0 consuming reads"):
                _consume_tile_dim_markers(zero_group_ops, zero_operations)

        # Two consumers: two distinct ops both read the marker's output.
        two_marker_op = make_marker_op()
        consumer_a = make_consumer("consumer_a")
        consumer_b = make_consumer("consumer_b")

        two_group_ops = [two_marker_op, consumer_a, consumer_b]
        two_operations = list(two_group_ops)
        two_graph = mock.Mock(spec=["operations"])
        two_graph.operations = two_operations

        with V.set_graph_handler(two_graph):
            with self.assertRaisesRegex(AssertionError, r"has 2 consuming reads"):
                _consume_tile_dim_markers(two_group_ops, two_operations)

    def test_nested_for_each_tile_markers_resolve_correctly(self):
        """Two tile_dim_marker-tagged reads at two nesting levels resolve.

        nested_split_m_then_k_fn wraps an outer M-tiling for_each_tile
        around an inner K-tiling for_each_tile -- the exact ambiguity the
        marker mechanism exists to eliminate (this is the case that would
        have been silently wrong under the deleted
        _loop_var_pos_from_reads heuristic). This drives the REAL
        splice_while_loops entry point (registered as this pipeline's
        first CustomPreSchedulingPasses pass) directly, rather than
        manually replaying splice_while_loop twice.

        Requires a real Spyre-device compile, not plain CPU tensors:
        CustomPreSchedulingPasses.__call__ early-returns
        (_operations_have_spyre_device check) for a device-less graph, so
        splice_while_loops -- and therefore marker consumption -- never
        runs at all on CPU tensors. (Confirmed directly: on CPU tensors,
        the top-level graph still has 1 unspliced ir.WhileLoop even after
        the full compile returns, because the whole pipeline was skipped,
        not because splicing failed.)

        splice_while_loops is the pipeline's *first* pass; every later
        pass in the pipeline (propagate_named_dims, assign_dim_hints, ...,
        propagate_spyre_tensor_layouts, codegen) runs after it and is
        irrelevant to what this test checks. Issue #4460's stick-layout/
        read-copy gap (the same gap test_carry_mode_split_k is xfailed
        for in test_for_each_tile_e2e.py) is now fixed for this fixture's
        shape, but nested_split_m_then_k_fn's inner loop still hits a
        distinct, out-of-scope issue #4581 gap (an outer-tile-carry
        symbol not recognized as indirect inside the inner WhileLoop
        body) during codegen, well after splice_while_loops has already
        completed -- so instead of driving the pipeline all the way
        through codegen, this test monkeypatches splice_while_loops
        itself (the name torch_spyre._inductor.passes imports and calls
        directly) to capture a *snapshot* of graph.operations right as it
        returns, and tolerates the InductorError issue #4581 raises
        afterward in a later, unrelated pass.

        The capture must be a snapshot (``list(graph.operations)``, a new
        list object), not a live reference to ``graph`` or to
        ``graph.operations`` itself. ``graph.operations`` is the SAME list
        object throughout the whole compile -- deadcode_elimination (the
        very next pass after splice_while_loops) and every later pass keep
        mutating it in place. Critically, DCE deletes an unspliced-but-
        still-present WhileLoop op for a completely unrelated reason: with
        no splice, the WhileLoop's output is never wired into anything
        downstream, so DCE treats it as ordinary dead code and removes it
        -- exactly as if splicing had succeeded. That means a *live* read of
        ``graph.operations`` taken after the full compile returns cannot
        tell "the marker/splice mechanism actually ran" apart from "the
        marker/splice mechanism was completely disabled": both leave 0
        WhileLoop ops by the time such a read happens. Only a snapshot
        taken inside the monkeypatch, before DCE or any later pass can
        touch the list again, actually pins down the state right after
        splice_while_loops returns.
        """
        import torch
        import torch_spyre  # noqa: F401  registers the "spyre" device
        from torch_spyre.constants import DEVICE_NAME

        import torch_spyre._inductor.passes as passes_mod
        from tests.inductor.for_each_tile_fixtures import (
            capture_post_grad_while_loop,
            nested_split_m_then_k_fn,
        )
        from torch._inductor import ir
        from torch._inductor.exc import InductorError

        X = torch.randn(8, 12, device=DEVICE_NAME, dtype=torch.float16)
        Y = torch.randn(12, 6, device=DEVICE_NAME, dtype=torch.float16)

        captured = {}
        original_splice_while_loops = passes_mod.splice_while_loops

        def capturing_splice_while_loops(graph):
            result = original_splice_while_loops(graph)
            captured["graph"] = graph
            # Snapshot -- a NEW list object -- taken at the exact instant
            # splice_while_loops returns, before deadcode_elimination (the
            # very next pass) or anything after it can mutate
            # graph.operations further. See this test's docstring for why
            # a live read of graph.operations after the full compile
            # returns cannot distinguish "spliced correctly" from
            # "splicing was a no-op and DCE pruned the orphaned WhileLoop
            # as unrelated dead code."
            captured["operations"] = list(graph.operations)
            return result

        passes_mod.splice_while_loops = capturing_splice_while_loops
        try:
            capture_post_grad_while_loop(nested_split_m_then_k_fn, (X, Y))
        except InductorError as exc:
            # Expected: codegen (much later, unrelated to splicing) hits
            # issue #4581 after splice_while_loops has already completed
            # and this test's capture has already fired. Any OTHER
            # exception is a real, unexpected finding -- do not swallow
            # it.
            self.assertIn(
                "indirect symbol",
                str(exc),
                "expected the known issue #4581 indirect-symbol gap, got a "
                f"different InductorError: {exc!r}",
            )
        finally:
            passes_mod.splice_while_loops = original_splice_while_loops

        self.assertIn("graph", captured, "splice_while_loops was never called/captured")
        self.assertIn(
            "operations", captured, "splice_while_loops was never called/captured"
        )
        operations = captured["operations"]
        # Right after splice_while_loops returns -- read from the snapshot,
        # NOT from graph.operations re-read now (see docstring: DCE and
        # later passes have already mutated that live list further by the
        # time this line runs) -- both the outer and inner WhileLoop must
        # already be spliced and every marker at both nesting levels
        # already consumed.
        remaining_while_ops = [op for op in operations if isinstance(op, ir.WhileLoop)]
        self.assertEqual(
            remaining_while_ops,
            [],
            "expected both nesting levels to be fully spliced",
        )
        # nested_split_m_then_k_fn's inner loop is split_k-shaped (matmul
        # consumer -- an aten-fallback ExternKernelOut, a StarDep-shaped
        # read; see this test's own docstring above). Per
        # _consume_tile_dim_markers's docstring, a StarDep-shaped
        # consumer's marker is deliberately kept materialized in
        # graph.operations (never erased) rather than erased the way a
        # ComputedBuffer consumer's marker is -- so unlike the outer
        # M-tiling marker (consumed by a Pointwise/Reduction ComputedBuffer
        # inside the outer body, erased normally), the inner K-tiling
        # marker is expected to still be present here. Exactly one marker
        # should remain: the fixture has exactly one StarDep-shaped
        # (matmul) marker consumer, at the inner nesting level.
        remaining_markers = [
            op for op in operations if getattr(op, "tile_marker_dim", None) is not None
        ]
        self.assertEqual(
            len(remaining_markers),
            1,
            "expected exactly one surviving marker (the inner split_k "
            "matmul's StarDep-shaped consumer -- see "
            "_consume_tile_dim_markers's own docstring on why that branch "
            "keeps its marker materialized rather than erasing it)",
        )

    def test_nested_for_each_tile_markers_snapshot_catches_noop_splice_stub(self):
        """Mutation coverage for the snapshot fix above.

        A `splice_while_loops` stub that does nothing but `return None` --
        never calling the real splicer, never mutating graph.operations
        itself -- leaves an unspliced WhileLoop genuinely present at the
        instant it returns. Confirms the fixed (snapshot-based) test body
        actually catches that: the snapshot must show 1 remaining
        WhileLoop op, so the first assertion in
        test_nested_for_each_tile_markers_resolve_correctly's body must
        fail against it.

        This is a DIFFERENT mutation from either of this file's other two
        for-each-tile-marker mutation tests:
          - a try_prove_for_each_tile-rejection mutation (not present in
            this file; see the fix-round-1 commit message) blocks
            splice_while_loops from ever attempting to splice a given
            WhileLoop at all -- a different code path.
          - test_nested_for_each_tile_markers_snapshot_catches_leftover_
            marker_injection (below) targets the SECOND assertion
            (remaining_markers) by injecting a fake marker into the
            snapshot AFTER capture, so it is unaffected by the live-
            reference-vs-snapshot issue this test targets.
        Neither of those exercises "splice_while_loops runs but does no
        real re-wiring work, and DCE prunes the evidence afterward
        regardless" -- the exact gap the snapshot fix above closes. Without
        the snapshot fix (i.e. reading live graph.operations after the
        full compile returns), this exact stub was independently confirmed
        to slip through: DCE deletes the orphaned, unspliced WhileLoop as
        ordinary dead code by the time such a live read happens, so the
        buggy test body would see 0 remaining WhileLoop ops here too --
        indistinguishable from a correct splice.
        """
        import torch
        import torch_spyre  # noqa: F401  registers the "spyre" device
        from torch_spyre.constants import DEVICE_NAME

        import torch_spyre._inductor.passes as passes_mod
        from tests.inductor.for_each_tile_fixtures import (
            capture_post_grad_while_loop,
            nested_split_m_then_k_fn,
        )
        from torch._inductor import ir
        from torch._inductor.exc import InductorError

        X = torch.randn(8, 12, device=DEVICE_NAME, dtype=torch.float16)
        Y = torch.randn(12, 6, device=DEVICE_NAME, dtype=torch.float16)

        captured = {}
        original_splice_while_loops = passes_mod.splice_while_loops

        def noop_splice_while_loops(graph):
            # Deliberately broken: never calls the real splicer, never
            # rewires or mutates anything. The WhileLoop op it was handed
            # is still fully intact in graph.operations right now.
            captured["operations"] = list(graph.operations)
            return None

        passes_mod.splice_while_loops = noop_splice_while_loops
        try:
            capture_post_grad_while_loop(nested_split_m_then_k_fn, (X, Y))
        except InductorError:
            # With no splice at all, later passes may fail in ways that
            # have nothing to do with issue #4460 -- any InductorError here
            # is fine to swallow; this mutation test only cares about the
            # snapshot taken above.
            pass
        finally:
            passes_mod.splice_while_loops = original_splice_while_loops

        self.assertIn(
            "operations", captured, "noop_splice_while_loops was never called"
        )
        operations = captured["operations"]
        remaining_while_ops = [op for op in operations if isinstance(op, ir.WhileLoop)]
        # This is the mutation catch: the no-op stub leaves the WhileLoop
        # genuinely present in the snapshot. If this assertion ever starts
        # passing (i.e. remaining_while_ops == []), the snapshot fix has
        # regressed back to something DCE can erase before capture.
        self.assertEqual(
            len(remaining_while_ops),
            1,
            "expected the no-op splice_while_loops stub to leave exactly "
            "one unspliced WhileLoop in the snapshot -- if this fails, "
            "the snapshot is no longer catching a disabled splice pass",
        )

    def test_nested_for_each_tile_markers_snapshot_catches_leftover_marker_injection(
        self,
    ):
        """Mutation coverage for the remaining_markers assertion (F1's fix).

        Drives the exact same real capture as
        test_nested_for_each_tile_markers_resolve_correctly (a genuine,
        unmutated splice_while_loops run), then -- AFTER the snapshot is
        taken -- tags one real op already in the snapshot with
        ``tile_marker_dim = 0``, simulating a marker that for some reason
        survived splicing/consumption. The fixed assertion
        (``getattr(op, "tile_marker_dim", None) is not None``) must catch
        this: it is the mutation test for the SECOND assertion in
        test_nested_for_each_tile_markers_resolve_correctly's body
        (remaining_markers), the counterpart to
        test_nested_for_each_tile_markers_snapshot_catches_noop_splice_stub
        (above), which mutates the *splice* to prove the *WhileLoop*
        assertion is non-vacuous -- this test mutates the *marker* state
        instead, to prove the *remaining_markers* assertion is non-vacuous.

        This directly guards against F1's actual bug: the pre-fix check
        (``hasattr(op, "data") and hasattr(op.data, "tile_marker_dim")``)
        looked one level too deep (at ``op.data``, the Pointwise/Reduction
        IR expression, which never carries the attribute) instead of at
        ``op`` itself (the ComputedBuffer, which is what
        ``lower_tile_dim_marker`` actually stamps). That pre-fix check
        would stay vacuously ``[]`` against this exact mutation -- confirmed
        directly while developing this fix, not merely asserted here.
        """
        import torch
        import torch_spyre  # noqa: F401  registers the "spyre" device
        from torch_spyre.constants import DEVICE_NAME

        import torch_spyre._inductor.passes as passes_mod
        from tests.inductor.for_each_tile_fixtures import (
            capture_post_grad_while_loop,
            nested_split_m_then_k_fn,
        )
        from torch._inductor.exc import InductorError

        X = torch.randn(8, 12, device=DEVICE_NAME, dtype=torch.float16)
        Y = torch.randn(12, 6, device=DEVICE_NAME, dtype=torch.float16)

        captured = {}
        original_splice_while_loops = passes_mod.splice_while_loops

        def capturing_splice_while_loops(graph):
            result = original_splice_while_loops(graph)
            captured["operations"] = list(graph.operations)
            return result

        passes_mod.splice_while_loops = capturing_splice_while_loops
        try:
            capture_post_grad_while_loop(nested_split_m_then_k_fn, (X, Y))
        except InductorError:
            # Same tolerated, unrelated issue #4460 gap as the test this
            # mirrors; irrelevant to this test's own mutation.
            pass
        finally:
            passes_mod.splice_while_loops = original_splice_while_loops

        self.assertIn(
            "operations", captured, "splice_while_loops was never called/captured"
        )
        operations = captured["operations"]

        # Sanity check on the real, unmutated snapshot: exactly the one
        # StarDep-shaped (inner split_k matmul) marker survives here, same
        # as test_nested_for_each_tile_markers_resolve_correctly asserts
        # (see that test's docstring for why -- _consume_tile_dim_markers
        # deliberately keeps a StarDep-shaped consumer's marker
        # materialized rather than erasing it).
        real_markers_before_injection = [
            op for op in operations if getattr(op, "tile_marker_dim", None) is not None
        ]
        self.assertEqual(
            len(real_markers_before_injection),
            1,
            "fixture assumption violated: expected exactly one surviving "
            "(StarDep-branch) marker before this test's own injection",
        )

        # The mutation: tag a real op already in the snapshot -- one that
        # does not already carry tile_marker_dim -- to simulate an EXTRA
        # marker left behind by an incomplete splice/consumption, on top
        # of the one real survivor above.
        victim = next(
            op
            for op in operations
            if not hasattr(op, "tile_marker_dim")
            and op not in real_markers_before_injection
        )
        self.assertFalse(
            hasattr(victim, "tile_marker_dim"),
            "fixture assumption violated: victim op already carries "
            "tile_marker_dim before injection",
        )
        victim.tile_marker_dim = 0
        try:
            remaining_markers = [
                op
                for op in operations
                if getattr(op, "tile_marker_dim", None) is not None
            ]
            # This is the mutation catch: the fixed assertion must see the
            # injected marker IN ADDITION TO the one real survivor. If
            # this assertion ever starts passing with victim missing, the
            # remaining_markers check has regressed back to something that
            # cannot see a real, ComputedBuffer-level tile_marker_dim tag
            # -- e.g. F1's original one-level-too-deep `op.data` bug.
            self.assertEqual(
                sorted(remaining_markers, key=id),
                sorted([*real_markers_before_injection, victim], key=id),
                "expected the injected tile_marker_dim to be caught by the "
                "remaining_markers filter alongside the one real survivor "
                "-- if this fails, the filter is no longer catching a "
                "marker tagged directly on the op",
            )
        finally:
            del victim.tile_marker_dim

    @unittest.expectedFailure
    def test_nested_for_each_tile_value_correct(self):
        # Issue #4460 (stick-layout/read-copy reconciliation gap in
        # propagate_layouts.py, inherited from nested_split_m_then_k_fn's
        # split_k-shaped inner loop) is fixed for this fixture's shape.
        # Compilation now proceeds further and hits a distinct,
        # out-of-scope issue #4581 gap (an outer-tile-carry symbol not
        # recognized as indirect inside the inner WhileLoop body) during
        # codegen. test_carry_mode_split_k (test_for_each_tile_e2e.py)
        # remains xfailed on the original #4460 gap for its own shape (a
        # StarDep-shaped matmul consumer, which does not hit #4581). Not a
        # defect in this plan's marker mechanism; out of scope for this
        # plan.
        import torch
        import torch_spyre  # noqa: F401  registers the "spyre" device
        from torch_spyre.constants import DEVICE_NAME

        from tests.inductor.for_each_tile_fixtures import (
            nested_split_m_then_k_fn,
            nested_split_m_then_k_reference,
        )

        torch._dynamo.reset()
        X = torch.randn(8, 12, device=DEVICE_NAME, dtype=torch.float16)
        Y = torch.randn(12, 6, device=DEVICE_NAME, dtype=torch.float16)
        expected = nested_split_m_then_k_reference(X.cpu(), Y.cpu()).to(DEVICE_NAME)

        compiled = torch.compile(
            nested_split_m_then_k_fn, backend="inductor", fullgraph=True
        )
        actual = compiled(X, Y)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
