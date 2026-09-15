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

No Spyre device or backend compiler is required. Covers three areas, each
in its own class group:
  1. while_loop_bridge's generic while_loop -> coarse-tile-group bridge:
     CarryBinding/carry_bindings_for and splice_while_loop's buffer
     transplant, carry/xs-leaf read redirection, and mutated-carry
     re-read guard, exercised against hand-built mocks
     (TestCarryBindingsFor, TestSpliceWhileLoop).
  2. for_each_tile_lowering's splice_while_loops pass and try_prove_
     for_each_tile shape prover, exercised against real ir.WhileLoop
     nodes built by lowering the vendored fixtures through GraphLowering
     (TestSpliceWhileLoops, TestTryProveForEachTile).
  3. Pass-pipeline registration: splice_while_loops runs first, ahead of
     every other pre-scheduling pass (TestPassPipelineRegistration).

For end-to-end compilation + numerical correctness against a CPU
reference, see test_for_each_tile_e2e.py.
"""

import unittest
from unittest import mock

from torch._inductor.virtualized import V

from tests.inductor.for_each_tile_fixtures import (
    capture_post_grad_while_loop,
    matmul_inputs,
    split_k_fn,
    split_m_elementwise_fn,
    split_m_fn,
)
from torch_spyre._inductor.wsr.for_each_tile_lowering import (
    try_prove_for_each_tile,
)


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

            remaining_markers = [
                op
                for op in graph.operations
                if getattr(op, "tile_marker_dim", None) is not None
            ]
            self.assertEqual(
                remaining_markers,
                [],
                "marker ops must be erased from graph.operations after consumption",
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
            marker_map = _consume_tile_dim_markers(group_ops, graph.operations)

            matmul_op = next(
                op for op in group_ops if isinstance(op, ir.ExternKernelOut)
            )
            matmul_name = matmul_op.get_name()

            x_dim = marker_map.get((matmul_name, StarDep(name="arg0_1")))
            y_dim = marker_map.get((matmul_name, StarDep(name="arg1_1")))
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


if __name__ == "__main__":
    unittest.main()
