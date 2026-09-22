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

import operator
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

    # matmul_inputs() is fp32 at M,K,N=256,256,64: chunking K into 4 tiles
    # of 64 and summing the partial products reorders fp32 accumulation
    # relative to a single whole matmul, producing rounding noise beyond
    # assert_close's tight fp32 defaults (atol=1e-05, rtol=1.3e-06) -- not a
    # logic bug. Loosened explicitly rather than left at the default.
    ATOL = 2e-4
    RTOL = 1e-3

    def test_reference_matches_plain_matmul(self):
        (X, Y), expected = matmul_inputs()
        actual = nested_split_m_then_k_reference(X, Y)
        torch.testing.assert_close(actual, expected, atol=self.ATOL, rtol=self.RTOL)

    def test_eager_fn_matches_plain_matmul(self):
        (X, Y), expected = matmul_inputs()
        actual = nested_split_m_then_k_fn(X, Y)
        torch.testing.assert_close(actual, expected, atol=self.ATOL, rtol=self.RTOL)


class TestCarryBindingsFor(unittest.TestCase):
    def test_fx_identity_detects_stride_repaired_passthrough_carry(self):
        from torch_spyre._inductor.wsr.while_loop_bridge import (
            _body_fx_carry_is_passthrough,
        )

        fx_graph = torch.fx.Graph()
        carry = fx_graph.placeholder("carry")
        xs = fx_graph.placeholder("xs")
        updated = fx_graph.call_function(operator.add, (carry, 1))
        fx_graph.output((updated, xs))

        while_op = mock.Mock()
        while_op.body_subgraph.graph.module = torch.fx.GraphModule({}, fx_graph)

        self.assertFalse(_body_fx_carry_is_passthrough(while_op, 0))
        self.assertTrue(_body_fx_carry_is_passthrough(while_op, 1))

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
    def test_accumulator_view_tags_backing_storage(self):
        """A view-backed carry records ownership on its mutable Buffer."""
        from torch._inductor import ir

        import torch_spyre._inductor.wsr.while_loop_bridge as bridge

        def computed_buffer(name, size, stride):
            data = mock.MagicMock(spec=ir.Pointwise)
            data.ranges = list(size)
            op = ir.ComputedBuffer(
                name=name,
                layout=ir.FixedLayout(
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    size=list(size),
                    stride=list(stride),
                ),
                data=data,
            )
            op.operation_name = name
            return op

        storage = computed_buffer("carry_storage", (2, 3), (3, 1))
        target = ir.ReinterpretView(
            data=ir.StorageBox(storage),
            layout=ir.FixedLayout(
                device=torch.device("cpu"),
                dtype=torch.float32,
                size=[3, 2],
                stride=[1, 3],
            ),
        )
        real_input = ir.TensorBox(target)
        producer = computed_buffer("carry_update", (3, 2), (2, 1))
        binding = bridge.CarryBinding(
            carry_index=0,
            initial=real_input,
            body_output=producer,
            scratch_name="unused_scratch",
        )
        graph = mock.Mock()
        graph.operations = []
        graph.mark_buffer_mutated = mock.Mock()

        with V.set_graph_handler(graph):
            body_ops = bridge._rewire_accumulator_output(
                graph,
                mock.Mock(),
                binding,
                [producer],
                real_input,
            )

        self.assertEqual(body_ops, [producer])
        self.assertIs(producer.layout.target, target)
        record = storage._loop_carry_record
        self.assertEqual(record.storage_name, "carry_storage")
        self.assertEqual(record.update_name, "carry_update")
        self.assertIs(producer._loop_carry_record, record)
        self.assertFalse(hasattr(target, "_loop_carry_record"))

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
                info = op.loop_info
                self.assertEqual(info.loop_group_id, (0,))
                self.assertIsNone(info.propagation)

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
                info = op.loop_info
                self.assertEqual(info.loop_group_id, (0,))
                self.assertIsNone(info.propagation)

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

    def test_nested_late_created_ops_inherit_ancestor_loop_info(self):
        """Inner splice-created markers and snapshots belong to both loops."""
        from torch._inductor import ir
        from torch._inductor.virtualized import V

        from tests.inductor.for_each_tile_fixtures import (
            attention_inputs,
            nested_online_softmax_fn,
        )
        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            splice_while_loops,
        )

        graph = self._run_graph(nested_online_softmax_fn, attention_inputs())
        with V.set_graph_handler(graph):
            splice_while_loops(graph)

        self.assertFalse(any(isinstance(op, ir.WhileLoop) for op in graph.operations))

        markers = [
            op
            for op in graph.operations
            if getattr(op, "tile_marker_dim", None) is not None
        ]
        self.assertTrue(markers, "expected surviving STAR_DEP_KEPT markers")
        self.assertIn(
            (0, 1),
            {op.loop_info.loop_group_id for op in markers},
            "an inner marker synthesized during its splice lost the outer level",
        )

        snapshots = [
            op
            for op in graph.operations
            if "while_loop_carry_snapshot" in op.get_name()
        ]
        self.assertTrue(snapshots, "expected the online-softmax carry snapshot")
        for op in snapshots:
            self.assertEqual(op.loop_info.loop_group_id, (0, 1))

    def test_noncontiguous_input_materialization_streams_one_tile(self):
        """A prefix view is staged one tile at a time after loop splicing."""
        from torch._inductor import ir
        from torch._inductor.dependencies import MemoryDep

        from torch_spyre._inductor.wsr import for_each_tile
        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            _identity_load,
            splice_while_loops,
        )

        def tile_sequence(x):
            def body(_, operands):
                (x_tile,) = operands
                return None, x_tile * 2

            _, out = for_each_tile(
                body,
                (x,),
                dims=(2,),
                tile_size=64,
                out_dim=2,
            )
            return out

        # A KV-cache prefix has a gap after every head: the physical sequence
        # extent is 320 while the logical prefix passed to attention is 256.
        backing = torch.randn(1, 8, 320, 128)
        prefix = backing[:, :, :256, :]
        self.assertEqual(prefix.stride(), (327680, 40960, 128, 1))

        graph = self._run_graph(tile_sequence, (prefix,))
        with V.set_graph_handler(graph):
            splice_while_loops(graph)

            identities = [
                (op, identity)
                for op in graph.operations
                if isinstance(op, ir.ComputedBuffer)
                and (identity := _identity_load(op)) is not None
            ]
            input_copies = [
                (op, identity)
                for op, identity in identities
                if identity[0] in graph.graph_input_names
            ]
            self.assertEqual(len(input_copies), 1)
            input_copy, (source_name, source_index, identity_indices) = input_copies[0]

            # WhileLoop.create originally materializes [4, 64, 1, 8, 128].
            # Once spliced into a four-trip counted loop, retaining that shape
            # would copy the complete prefix on every trip.  The compiler must
            # instead reuse one compact [Lk_tile, H, D] staging buffer.
            self.assertEqual(list(input_copy.data.ranges), [1, 64, 1, 8, 128])
            self.assertEqual(list(input_copy.layout.size), [1, 64, 1, 8, 128])
            self.assertEqual(list(input_copy.layout.stride), [0, 128, 0, 8192, 1])
            self.assertEqual(
                input_copy.loop_info.squeezed_advance_per_read,
                [[[(8192, 1)]]],
            )
            self.assertEqual(source_index.coeff(identity_indices[0]), 8192)
            self.assertEqual(source_index.coeff(identity_indices[3]), 40960)

            loop_var = input_copy.dim_hints[0].loop_var
            self.assertIsNotNone(loop_var)
            direct_readers = []
            for consumer in graph.operations:
                reads = [
                    dep
                    for dep in consumer.get_read_writes().reads
                    if isinstance(dep, MemoryDep) and dep.name == input_copy.get_name()
                ]
                if reads:
                    direct_readers.append(consumer)
                for dep in reads:
                    self.assertEqual(dep.index.coeff(loop_var), 0)

            non_identity_readers = [
                reader for reader in direct_readers if _identity_load(reader) is None
            ]
            self.assertEqual(
                len(non_identity_readers),
                1,
                [
                    (reader.get_name(), _identity_load(reader))
                    for reader in direct_readers
                ],
            )
            direct_record = non_identity_readers[0]._read_copy_elision_record
            self.assertEqual(direct_record.copy_name, input_copy.get_name())
            self.assertEqual(direct_record.source_name, source_name)

            self.assertFalse(
                any(
                    getattr(op, "loop_info", None)
                    and list(op.data.ranges) == [4, 64, 1, 8, 128]
                    for op, _identity in identities
                ),
                "the full-cache exact-stride copy remained inside the loop",
            )


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

        # Lowered on the captured graph's OWN placeholders, not on `args`:
        # dynamo/AOT order the post-grad graph's placeholders by nothing the
        # caller controls (paged_gather_kv_fn's q, table, k, v arrive in a
        # different order than they are passed), so feeding `args`
        # positionally binds inputs to the wrong placeholders and blows up in
        # lowering on a shape mismatch. Fake tensors are what the real
        # Inductor pipeline runs GraphLowering on anyway.
        placeholders = [
            node.meta["val"] for node in gm.graph.nodes if node.op == "placeholder"
        ]
        graph = GraphLowering(
            gm, example_inputs=placeholders, shape_env=fake_mode.shape_env
        )
        with V.set_graph_handler(graph), V.set_fake_mode(fake_mode):
            graph.run(*placeholders)
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

    def test_marker_with_two_computed_buffer_consumers_maps_both(self):
        """Paged attention's shape: one marker, two ComputedBuffer consumers.

        paged_gather_kv_fn slices one page index out of the tiled block table
        and hands it to two ``index_select``s (K's page and V's), so the
        table's single dim=0 marker has two consuming reads, both
        inline-branch shaped. This is the same supported multi-consumer shape
        softmax_row_tiled_fn reaches through torch.softmax's amax/sub
        siblings, pinned here at the IR level rather than only end to end,
        and on a marker that carries a real per-trip advance.

        Asserts more than "it no longer raises". Each consumer must get its
        OWN map entry (dropping either silently loses that op's tiled-dim
        provenance), each post-inline read must still carry the marker's own
        per-trip advance term with the marker's own coefficient (the
        silent-wrong-numerics regression
        test_marker_inlined_preserves_advance_term_on_computed_buffer_
        consumer pins for one consumer -- composing the transform into the
        first consumer and merely renaming past the second would satisfy a
        weaker check), and the marker must end up erased, since every one of
        its consumers took the inline branch.
        """
        from torch._inductor import ir
        from torch._inductor.dependencies import MemoryDep

        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            MarkerResolution,
            _body_loop_var,
            _consume_tile_dim_markers,
            _marker_resolution,
            _stacking_carry_indices,
            try_prove_for_each_tile,
        )
        from torch_spyre._inductor.wsr.while_loop_bridge import (
            carry_bindings_for,
            splice_while_loop,
        )

        from tests.inductor.for_each_tile_fixtures import (
            paged_gather_kv_fn,
            paged_gather_kv_inputs,
        )

        args = paged_gather_kv_inputs()
        graph = self._run_graph(paged_gather_kv_fn, args)

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

            markers = [
                op
                for op in group_ops
                if getattr(op, "tile_marker_dim", None) is not None
            ]
            self.assertEqual(
                len(markers),
                1,
                "fixture assumption violated: only the block table is "
                f"tiled here, so exactly one marker is expected; got {markers!r}",
            )
            marker_op = markers[0]
            marker_name = marker_op.get_name()

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
                f"read index to carry the per-trip advance ({loop_var}); "
                f"got {marker_own_index!r}",
            )

            consumer_names = {
                op.get_name()
                for op in group_ops
                if isinstance(op, ir.ComputedBuffer)
                and op is not marker_op
                and any(
                    isinstance(d, MemoryDep) and d.name == marker_name
                    for d in op.get_read_writes().reads
                )
            }
            self.assertEqual(
                len(consumer_names),
                2,
                "fixture assumption violated: expected the page index to "
                "be read by two separate ComputedBuffer gathers (K's and "
                f"V's index_select); got {sorted(consumer_names)}",
            )

            marker_map = _consume_tile_dim_markers(group_ops, graph.operations)

        mapped_names = {name for name, _dep in marker_map}
        self.assertEqual(
            mapped_names,
            consumer_names,
            "every consumer of the marker must get its own map entry",
        )
        self.assertEqual(set(marker_map.values()), {0})

        for (name, dep), _dim in marker_map.items():
            self.assertEqual(
                dep.name,
                marker_input_name,
                f"{name}'s mapped dep must name the marker's own upstream "
                "input, i.e. be the post-inline read",
            )
            self.assertEqual(
                dep.index.coeff(loop_var),
                marker_own_index.coeff(loop_var),
                f"{name}'s post-inline read lost or altered the marker's "
                f"own per-trip advance term ({loop_var}) -- got "
                f"{dep.index!r}, marker's own index is {marker_own_index!r}. "
                "Every consumer must have the marker's real coordinate "
                "transform composed in, not just the first one.",
            )

        self.assertEqual(
            _marker_resolution(marker_op),
            MarkerResolution.INLINE_ERASED,
            "with both consumers inlined, the marker no longer needs to be "
            "materialized",
        )
        self.assertNotIn(marker_op, graph.operations)
        self.assertNotIn(marker_op, group_ops)

    def test_split_k_marker_resolves_reduction_dim(self):
        """IR-level value assertion for split_k_fn's reduction-dim marker.

        Uses self._run_graph (this class's own established pattern, see its
        docstring above) rather than a full torch.compile capture: this
        keeps the test focused on marker resolution alone, independent of
        the full lowering/codegen pipeline (see test_carry_mode_split_k in
        test_for_each_tile_e2e.py, issue #4460, for that pipeline's own
        now-passing end-to-end coverage of this fixture family).

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
        """_consume_tile_dim_markers's consumer-count guards (F4.3).

        A tile_dim_marker op must have at least one consuming read within
        its spliced body, and at most one StarDep-shaped consuming read
        (multiple ComputedBuffer consuming reads ARE supported -- see
        torch.softmax's amax/sub siblings, test_softmax_row_tiled_small).
        Zero consumers and multiple StarDep consumers are both unrecognized/
        unhandled shapes and must raise AssertionError rather than silently
        pick a default or (for the StarDep case) silently produce wrong
        numerics -- untested anywhere in this file before this test. Covers
        both wrong-count shapes in one test (zero, then two StarDep), each
        built directly via this class's established mock.Mock(spec=[...])
        convention (see TestSpliceWhileLoop above) rather than a full
        torch.compile. The zero-consumer shape is believed unreachable
        through the ordinary lowering path, which is why it needs direct
        construction to exercise at all. The two-StarDep-consumer shape is
        NOT unreachable: it is reached for real by
        for_each_tile_fixtures.py's sibling_nested_fn/sibling_nested_
        stardep_fn (two independent WhileLoops both consuming a single
        outer marker on their shared tiled operand via StarDep -- see
        those fixtures' docstrings, issue #4581 territory). It is still
        built directly here too, for a fast, isolated unit test of the
        guard itself rather than a full torch.compile round-trip.
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
            with self.assertRaisesRegex(
                AssertionError, r"has 2 StarDep-shaped consuming reads"
            ):
                _consume_tile_dim_markers(two_group_ops, two_operations)

    def test_consume_tile_dim_markers_stamps_marker_resolution(self):
        """_consume_tile_dim_markers stamps tile_marker_resolution per marker.

        Builds two markers: one whose single consumer is an ordinary
        ComputedBuffer-shaped read (MemoryDep, inline-erase branch) and one
        whose single consumer is StarDep-shaped (kept-materialized branch).
        Confirms each marker ends up stamped with the matching
        MarkerResolution member, and that the inline-erased marker is
        removed from operations/group_ops while the StarDep-kept one
        survives in both -- the pre-existing behavior this task must not
        change, now observable through the new field.

        The StarDep consumer is a bare mock.Mock(spec=[...]), matching
        test_consume_tile_dim_markers_raises_on_wrong_consumer_count's
        convention -- that branch never calls get_read_writes() on
        anything but the mocked reads list. The inline-erase branch is
        different: _inline_marker_into_consumer/replace_computed_buffer_body
        constructs a genuine new ComputedBuffer and _consume_tile_dim_
        markers immediately re-derives its post-inline dep from a REAL
        new_consumer.get_read_writes() call (not a mock return value), which
        recurses into real Inductor tracing (ComputedBuffer.get_read_writes
        -> extract_read_writes -> data.get_pointwise_size()/make_loader())
        that only works against genuine ir.Pointwise/ir.FixedLayout objects
        and a real SizeVarAllocator on V.graph -- a bare mock.Mock() for
        `.data`/`.layout` (confirmed empirically) fails inside that real
        tracing machinery well before reaching this pass's own logic, so
        the inline marker/consumer pair here is built from minimal real IR
        pieces instead.
        """
        import sympy

        from torch._inductor.dependencies import MemoryDep, StarDep
        from torch._inductor.ir import FixedLayout, Pointwise
        from torch._inductor.sizevars import SizeVarAllocator
        from torch._inductor.virtualized import ops
        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            MarkerResolution,
            _consume_tile_dim_markers,
            _marker_resolution,
        )

        def make_marker_op(name):
            marker_op = mock.Mock(
                name=name,
                spec=[
                    "get_read_writes",
                    "get_name",
                    "tile_marker_dim",
                    "data",
                    "layout",
                ],
            )
            marker_op.get_name.return_value = name
            marker_op.tile_marker_dim = 0
            marker_op.data = mock.Mock()
            marker_op.layout = mock.Mock(size=(4,), stride=(1,), offset=0)
            marker_rw = mock.Mock()
            d0 = sympy.Symbol("d0")
            marker_rw.reads = [
                MemoryDep(name="input_buf", index=d0, var_names=(d0,), size=(4,))
            ]
            # _marker_substitution (for_each_tile_lowering.py) also reads
            # the marker's own WRITE dep off this same get_read_writes()
            # call, to recover the var_names it substitutes the inline
            # consumer's load-site coordinates into.
            marker_rw.writes = [
                MemoryDep(name=name, index=d0, var_names=(d0,), size=(4,))
            ]
            marker_op.get_read_writes.return_value = marker_rw
            return marker_op

        def make_inline_consumer(name, marker_name):
            consumer_op = mock.Mock(
                name=name,
                spec=[
                    "get_read_writes",
                    "get_name",
                    "get_operation_name",
                    "data",
                    "layout",
                    "operation_name",
                    "_split_size",
                    "_original_inner_fn",
                    "_original_ranges",
                    "_original_reduction_ranges",
                    "origins",
                    "origin_node",
                ],
            )
            consumer_op.get_name.return_value = name
            consumer_op.get_operation_name.return_value = name

            def inner_fn(index):
                return ops.load(marker_name, index[0])

            # A real Pointwise/FixedLayout pair -- required because
            # _consume_tile_dim_markers calls the real
            # ComputedBuffer.get_read_writes() on the reconstructed
            # consumer below (see this test's docstring); a mock.Mock()
            # `.data`/`.layout` cannot satisfy that real tracing path.
            consumer_op.data = Pointwise(
                device=torch.device("cpu"),
                dtype=torch.float32,
                inner_fn=inner_fn,
                ranges=[sympy.Integer(4)],
            )
            consumer_op.layout = FixedLayout(torch.device("cpu"), torch.float32, [4])
            consumer_op.operation_name = name
            consumer_op._split_size = None
            consumer_op._original_inner_fn = None
            consumer_op._original_ranges = None
            consumer_op._original_reduction_ranges = None
            consumer_op.origins = set()
            consumer_op.origin_node = None
            consumer_rw = mock.Mock()
            consumer_rw.reads = [
                MemoryDep(
                    name=marker_name, index=sympy.Integer(0), var_names=(), size=()
                )
            ]
            consumer_op.get_read_writes.return_value = consumer_rw
            return consumer_op

        def make_stardep_consumer(name, marker_name):
            consumer_op = mock.Mock(name=name, spec=["get_read_writes", "get_name"])
            consumer_op.get_name.return_value = name
            consumer_rw = mock.Mock()
            consumer_rw.reads = [StarDep(name=marker_name, mode=None)]
            consumer_op.get_read_writes.return_value = consumer_rw
            return consumer_op

        inline_marker = make_marker_op("inline_marker")
        inline_consumer = make_inline_consumer("inline_consumer", "inline_marker")
        stardep_marker = make_marker_op("stardep_marker")
        stardep_consumer = make_stardep_consumer("stardep_consumer", "stardep_marker")

        group_ops = [inline_marker, inline_consumer, stardep_marker, stardep_consumer]
        operations = list(group_ops)
        # Real SizeVarAllocator: is_zero_elements() (hit while re-tracing
        # the reconstructed inline consumer's get_read_writes(), see this
        # test's docstring) calls V.graph.sizevars.statically_known_true(),
        # which a plain mock.Mock() cannot satisfy.
        graph = mock.Mock(
            spec=["operations", "name_to_buffer", "name_to_op", "sizevars"]
        )
        graph.operations = operations
        graph.name_to_buffer = {}
        graph.name_to_op = {}
        graph.sizevars = SizeVarAllocator()

        with V.set_graph_handler(graph):
            _consume_tile_dim_markers(group_ops, operations)

        self.assertEqual(
            _marker_resolution(inline_marker),
            MarkerResolution.INLINE_ERASED,
            "expected the ComputedBuffer/MemoryDep-consumed marker to be "
            "stamped INLINE_ERASED",
        )
        self.assertEqual(
            _marker_resolution(stardep_marker),
            MarkerResolution.STAR_DEP_KEPT,
            "expected the StarDep-consumed marker to be stamped STAR_DEP_KEPT",
        )
        self.assertNotIn(
            inline_marker,
            operations,
            "inline-erased marker must still be removed from operations "
            "(pre-existing behavior, must not regress)",
        )
        self.assertIn(
            stardep_marker,
            operations,
            "StarDep-kept marker must still remain in operations "
            "(pre-existing behavior, must not regress)",
        )

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
        read-copy gap (the same gap test_carry_mode_split_k in
        test_for_each_tile_e2e.py now passes against) is fixed for this
        fixture's shape, and issue #4706's OS-5 symbol-consistency gap on
        splice_while_loops's synthetic carry/tile-read redirect `identity`
        op (a distinct follow-up to issue #4581) is fixed too, per
        create_tensor_arg's device_tile_advance_expr handling
        (torch_spyre/_inductor/wsr/for_each_tile_lowering.py). The fixture
        now compiles cleanly all the way through. This test monkeypatches
        splice_while_loops itself (the
        name torch_spyre._inductor.passes imports and calls directly) to
        capture a *snapshot* of graph.operations right as it returns, and
        tolerates the new OS-5 InductorError afterward in a later,
        unrelated pass -- the same structure the original test used for
        the "indirect symbol" error it used to tolerate, updated to the
        new error this fix's guard change causes the pipeline to reach.

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

        X = torch.randn(256, 256, device=DEVICE_NAME, dtype=torch.float16)
        Y = torch.randn(256, 64, device=DEVICE_NAME, dtype=torch.float16)

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

    def test_triple_nested_stardep_outer_resolves_correctly(self):
        """Three-level nesting, STAR_DEP_KEPT at the outer level: marker
        splicing/resolution completes correctly and the fixture compiles
        cleanly to completion.

        Directly analogous to test_nested_for_each_tile_markers_resolve_
        correctly's finding for the (simpler, 2-level)
        nested_split_m_then_k_fn fixture: splice_while_loops fully splices
        both WhileLoop ops (0 remain) and both outer-level STAR_DEP_KEPT
        markers survive correctly-tagged. Issue #4706's OS-5 symbol-
        consistency gap on a synthetic `identity` op inserted by
        splice_while_loops's carry/tile-read redirect -- the same gap
        documented on test_nested_for_each_tile_markers_resolve_correctly
        and on test_nested_for_each_tile_value_correct -- is fixed for
        depth=3 nesting too (not just depth=2).
        """
        import torch
        import torch_spyre  # noqa: F401
        from torch_spyre.constants import DEVICE_NAME
        from tests.inductor.for_each_tile_fixtures import (
            capture_post_grad_while_loop,
            triple_nested_stardep_outer_fn,
        )

        X = torch.randn(2, 256, 256, device=DEVICE_NAME, dtype=torch.float16)
        Y = torch.randn(2, 256, 64, device=DEVICE_NAME, dtype=torch.float16)
        capture_post_grad_while_loop(triple_nested_stardep_outer_fn, (X, Y))

    def test_triple_nested_stardep_middle_resolves_correctly(self):
        """Three-level nesting, STAR_DEP_KEPT at the middle level: marker
        splicing/resolution completes correctly and the fixture compiles
        cleanly to completion.

        Same fix as test_triple_nested_stardep_outer_resolves_correctly --
        see that test's docstring for the full explanation.
        """
        import torch
        import torch_spyre  # noqa: F401
        from torch_spyre.constants import DEVICE_NAME
        from tests.inductor.for_each_tile_fixtures import (
            capture_post_grad_while_loop,
            triple_nested_stardep_middle_fn,
        )

        X = torch.randn(2, 256, 256, device=DEVICE_NAME, dtype=torch.float16)
        Y = torch.randn(2, 256, 64, device=DEVICE_NAME, dtype=torch.float16)
        capture_post_grad_while_loop(triple_nested_stardep_middle_fn, (X, Y))

    def test_triple_nested_stardep_inner_resolves_correctly(self):
        """Three-level nesting, STAR_DEP_KEPT at the inner level: marker
        splicing/resolution completes correctly and the fixture compiles
        cleanly to completion.

        Same fix as test_triple_nested_stardep_outer_resolves_correctly --
        see that test's docstring for the full explanation.
        """
        import torch
        import torch_spyre  # noqa: F401
        from torch_spyre.constants import DEVICE_NAME
        from tests.inductor.for_each_tile_fixtures import (
            capture_post_grad_while_loop,
            triple_nested_stardep_inner_fn,
        )

        X = torch.randn(2, 256, 256, device=DEVICE_NAME, dtype=torch.float16)
        Y = torch.randn(2, 256, 64, device=DEVICE_NAME, dtype=torch.float16)
        capture_post_grad_while_loop(triple_nested_stardep_inner_fn, (X, Y))

    def test_triple_nested_stardep_multilevel_resolves_correctly(self):
        """Three-level nesting, STAR_DEP_KEPT at two levels at once: marker
        splicing/resolution completes correctly and the fixture compiles
        cleanly to completion.

        Same fix as test_triple_nested_stardep_outer_resolves_correctly --
        see that test's docstring for the full explanation.

        NOTE: independent of that, this fixture currently exercises
        the SAME two outer-level markers as
        test_triple_nested_stardep_outer_resolves_correctly, not independent
        two-marker interaction. Task 3 found that triple_nested_stardep_
        multilevel_fn's inner-level marker is completely absent after
        splicing when the outer level is also STAR_DEP_KEPT (a separate,
        real, independently-confirmed finding -- see that fixture's
        docstring in for_each_tile_fixtures.py). This test does NOT prove
        the intended independent-multi-marker-interaction case pending a
        fix to that inner-marker-loss finding.
        """
        import torch
        import torch_spyre  # noqa: F401
        from torch_spyre.constants import DEVICE_NAME
        from tests.inductor.for_each_tile_fixtures import (
            capture_post_grad_while_loop,
            triple_nested_stardep_multilevel_fn,
        )

        X = torch.randn(2, 256, 256, device=DEVICE_NAME, dtype=torch.float16)
        Y = torch.randn(2, 256, 64, device=DEVICE_NAME, dtype=torch.float16)
        capture_post_grad_while_loop(triple_nested_stardep_multilevel_fn, (X, Y))

    def test_sibling_nested_resolves_correctly(self):
        """Two sibling (non-nested) for_each_tile loops sharing one outer
        marker do NOT compile -- a real, already triple-confirmed
        architectural gap, not a fixture bug.

        sibling_nested_fn's outer for_each_tile stamps exactly one dim=0
        marker on x_tile, and hands that same x_tile directly to both
        sibling_a's and sibling_b's inner for_each_tile calls. Each lowers
        to its own WhileLoop op, so the single marker ends up with 2
        consuming reads via StarDep -- but _consume_tile_dim_markers's
        hard assertion requires exactly 1. This is issue #4581 territory
        (out of scope for this plan); see sibling_nested_fn's docstring in
        for_each_tile_fixtures.py for the full mechanism writeup. This test
        asserts (a) the eager reference is correct on its own (no device,
        no compile) and (b) compiling raises the documented assertion,
        matched on a naming-independent substring so it survives unrelated
        upstream op-naming shifts.
        """
        import torch
        import torch_spyre  # noqa: F401
        import pytest
        from torch._inductor.exc import InductorError
        from torch_spyre.constants import DEVICE_NAME
        from tests.inductor.for_each_tile_fixtures import (
            capture_post_grad_while_loop,
            sibling_nested_fn,
            sibling_nested_reference,
        )

        X = torch.randn(256, 256, dtype=torch.float16)
        Y = torch.randn(256, 64, dtype=torch.float16)
        expected = sibling_nested_reference(X, Y)
        actual = sibling_nested_fn(X, Y)
        torch.testing.assert_close(actual, expected)

        X_dev = X.to(DEVICE_NAME)
        Y_dev = Y.to(DEVICE_NAME)
        with pytest.raises(
            InductorError, match="consuming reads within its spliced body"
        ):
            capture_post_grad_while_loop(sibling_nested_fn, (X_dev, Y_dev))

    def test_sibling_nested_stardep_resolves_correctly(self):
        """Two sibling for_each_tile loops, one STAR_DEP_KEPT and one
        INLINE_ERASED, sharing one outer marker, do NOT compile -- the same
        real, already triple-confirmed architectural gap as
        test_sibling_nested_resolves_correctly.

        sibling_nested_stardep_fn is structurally identical to
        sibling_nested_fn in the way that matters here: a single outer
        dim=0 marker on x_tile is handed directly to two sibling inner
        for_each_tile calls, each becoming its own WhileLoop consumer of
        that one marker via StarDep -- 2 consuming reads where
        _consume_tile_dim_markers requires exactly 1. See
        sibling_nested_stardep_fn's docstring in for_each_tile_fixtures.py
        for the full mechanism writeup. This test asserts (a) the eager
        reference is correct on its own (no device, no compile) and (b)
        compiling raises the documented assertion, matched on a naming-
        independent substring.
        """
        import torch
        import torch_spyre  # noqa: F401
        import pytest
        from torch._inductor.exc import InductorError
        from torch_spyre.constants import DEVICE_NAME
        from tests.inductor.for_each_tile_fixtures import (
            capture_post_grad_while_loop,
            sibling_nested_stardep_fn,
            sibling_nested_stardep_reference,
        )

        X = torch.randn(256, 256, dtype=torch.float16)
        Y = torch.randn(256, 64, dtype=torch.float16)
        expected = sibling_nested_stardep_reference(X, Y)
        actual = sibling_nested_stardep_fn(X, Y)
        torch.testing.assert_close(actual, expected)

        X_dev = X.to(DEVICE_NAME)
        Y_dev = Y.to(DEVICE_NAME)
        with pytest.raises(
            InductorError, match="consuming reads within its spliced body"
        ):
            capture_post_grad_while_loop(sibling_nested_stardep_fn, (X_dev, Y_dev))

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

        X = torch.randn(256, 256, device=DEVICE_NAME, dtype=torch.float16)
        Y = torch.randn(256, 64, device=DEVICE_NAME, dtype=torch.float16)

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

        X = torch.randn(256, 256, device=DEVICE_NAME, dtype=torch.float16)
        Y = torch.randn(256, 64, device=DEVICE_NAME, dtype=torch.float16)

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
        # The marker_resolution-aware guard in
        # _synthesize_dim_hints_for_group (issue #4581's fix) makes real
        # progress -- the pipeline now runs past the original codegen-time
        # "indirect symbol" lookup failure. Issue #4706's OS-5
        # symbol-consistency gap on splice_while_loops's synthetic
        # `identity` op (create_tensor_arg now strips WhileLoop-splice
        # loop_var symbols like u5 out of the static device_coordinates
        # and folds their contribution into device_tile_advance_expr
        # instead) is fixed too, for this fixture's shape. Compilation,
        # scheduling, and codegen now all complete -- but the test still
        # fails at the final numeric assertion, and the failure is
        # nondeterministic run-to-run on this same fixed-seed-free fixture
        # (confirmed via repeated clean-cache runs). This points to a
        # memory-safety-class bug (a stale or aliased HBM read, likely in
        # hbm_pool allocation lifetime or carry read/write scheduling
        # order) rather than a deterministic addressing/indexing bug in
        # the tiled_symbols/splice-var binding layer -- see issue #4701
        # for the full investigation writeup and next steps.
        # test_carry_mode_split_k (test_for_each_tile_e2e.py) now passes:
        # its StarDep-shaped matmul consumer turned out to hit the same
        # issue #4706 OS-5 symbol-consistency layer as this fixture, and
        # does not hit the #4701 nondeterministic memory-safety gap this
        # test remains xfailed on (confirmed via isolated stash/pop
        # bisection -- see that test's own docstring).
        import torch
        import torch_spyre  # noqa: F401  registers the "spyre" device
        from torch_spyre.constants import DEVICE_NAME

        from tests.inductor.for_each_tile_fixtures import (
            nested_split_m_then_k_fn,
            nested_split_m_then_k_reference,
        )

        torch._dynamo.reset()
        X = torch.randn(256, 256, device=DEVICE_NAME, dtype=torch.float16)
        Y = torch.randn(256, 64, device=DEVICE_NAME, dtype=torch.float16)
        expected = nested_split_m_then_k_reference(X.cpu(), Y.cpu()).to(DEVICE_NAME)

        compiled = torch.compile(
            nested_split_m_then_k_fn, backend="inductor", fullgraph=True
        )
        actual = compiled(X, Y)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-2, rtol=1e-2)


class TestStampDirectLoopInfo(unittest.TestCase):
    """_stamp_direct_loop_info builds loop_group_id/loop_count directly."""

    def _run_graph(self, fn, args):
        """Lower fn(*args) through a fresh GraphLowering and return it.

        Same pattern as TestSpliceWhileLoops._run_graph/TestConsumeTileDim
        Markers._run_graph: a standalone GraphLowering.run() call stops
        short of codegen(), so splice_while_loops (a pre-scheduling pass
        that only fires from _update_scheduler during codegen()) never
        runs -- leaving the WhileLoop op intact in graph.operations for
        this test to splice and stamp itself.
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

        # Lowered on the captured graph's OWN placeholders, not on `args`:
        # dynamo/AOT order the post-grad graph's placeholders by nothing the
        # caller controls, so feeding `args` positionally can bind inputs to
        # the wrong placeholders (see TestConsumeTileDimMarkers._run_graph's
        # docstring for a fixture where this actually happens).
        placeholders = [
            node.meta["val"] for node in gm.graph.nodes if node.op == "placeholder"
        ]
        graph = GraphLowering(
            gm, example_inputs=placeholders, shape_env=fake_mode.shape_env
        )
        with V.set_graph_handler(graph), V.set_fake_mode(fake_mode):
            graph.run(*placeholders)
        return graph

    def test_single_level_stamps_group_id_and_count(self):
        from torch._inductor import ir

        from tests.inductor.for_each_tile_fixtures import matmul_inputs, split_k_fn
        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            _body_loop_var,
            _stamp_direct_loop_info,
            try_prove_for_each_tile,
        )
        from torch_spyre._inductor.wsr.while_loop_bridge import (
            carry_bindings_for,
            splice_while_loop,
        )

        (X, Y), _ref = matmul_inputs()
        graph = self._run_graph(split_k_fn, (X, Y))

        while_ops = [op for op in graph.operations if isinstance(op, ir.WhileLoop)]
        self.assertEqual(len(while_ops), 1)
        while_op = while_ops[0]

        result = try_prove_for_each_tile(while_op)
        self.assertTrue(result.accepted, result.reason)
        loop_var = _body_loop_var(while_op)
        self.assertIsNotNone(loop_var)

        with V.set_graph_handler(graph):
            carries = carry_bindings_for(while_op)
            group_ops = splice_while_loop(
                graph, while_op, carries, trip_count=result.trip_count
            )

            _stamp_direct_loop_info(group_ops, loop_var, result.trip_count, group_idx=0)

        stamped = [op for op in group_ops if getattr(op, "loop_info", None)]
        self.assertTrue(stamped, "no op received a loop_info stamp")
        for op in stamped:
            info = op.loop_info
            self.assertEqual(info.loop_group_id, (0,))
            self.assertEqual(info.loop_count, [result.trip_count])
            self.assertIsNone(info.propagation)

    def test_marker_resolved_dim_appears_in_loop_tiled_dims(self):
        """A marker-resolved tile read surfaces in loop_tiled_dims.

        Deviates from the brief's literal snippet in fixture choice:
        split_m_fn's own tile-marker consumer is `x_tile @ y_whole`, a
        matmul that lowers to an aten-fallback ExternKernelOut on this
        CPU-fixture path -- a StarDep-shaped consumer (see
        split_m_elementwise_fn's own docstring in for_each_tile_fixtures.py)
        that lookup_marker_dim deliberately returns None for (no
        index/ranges to resolve a position from). Verified empirically:
        with split_m_fn, no op in group_ops ever gets a non-empty
        loop_tiled_dims, so the assertion below would fail even against a
        correct implementation. split_m_elementwise_fn's intervening
        `x_tile * 2.0` lowers to a genuine Pointwise ComputedBuffer, whose
        MemoryDep read of the marker IS what _consume_tile_dim_markers'
        ComputedBuffer branch (_inline_marker_into_consumer) fuses in and
        maps -- giving lookup_marker_dim a real MemoryDep to resolve.
        """
        from torch._inductor import ir

        from tests.inductor.for_each_tile_fixtures import split_m_elementwise_fn
        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            _body_loop_var,
            _consume_tile_dim_markers,
            _stamp_direct_loop_info,
            try_prove_for_each_tile,
        )
        from torch_spyre._inductor.wsr.while_loop_bridge import (
            carry_bindings_for,
            splice_while_loop,
        )

        X = torch.randn(128, 12)
        Y = torch.randn(12, 6)
        graph = self._run_graph(split_m_elementwise_fn, (X, Y))

        while_ops = [op for op in graph.operations if isinstance(op, ir.WhileLoop)]
        self.assertEqual(len(while_ops), 1)
        while_op = while_ops[0]

        result = try_prove_for_each_tile(while_op)
        self.assertTrue(result.accepted, result.reason)
        loop_var = _body_loop_var(while_op)
        self.assertIsNotNone(loop_var)

        with V.set_graph_handler(graph):
            carries = carry_bindings_for(while_op)
            group_ops = splice_while_loop(
                graph, while_op, carries, trip_count=result.trip_count
            )
            _consume_tile_dim_markers(group_ops, graph.operations)

            _stamp_direct_loop_info(group_ops, loop_var, result.trip_count, group_idx=0)

        tiled_ops = [
            op
            for op in group_ops
            if getattr(op, "loop_info", None) and op.loop_info.loop_tiled_dims[-1]
        ]
        self.assertTrue(
            tiled_ops,
            "no op in split_m_elementwise_fn's body has a marker-resolved tiled dim",
        )

    def test_tiled_dims_per_read_has_one_entry_per_read_dep(self):
        """tiled_dims_per_read/output_tiled_dims match op.get_read_writes().

        Deviates from the brief's literal snippet in two ways, both
        verified empirically against the real fixture/API before writing
        this test (see task-4-report.md):

        1. Fixture: split_m_fn's own tile-marker consumer (`x_tile @
           y_whole`) is a StarDep-shaped ExternKernelOut consumer -- same
           reason test_marker_resolved_dim_appears_in_loop_tiled_dims
           (Task 3) switched to split_m_elementwise_fn. Reused here rather
           than reintroducing split_m_fn's dead end.
        2. Read-dep filter: the brief's snippet filters reads/writes with
           `hasattr(d, "index")`, but StarDep.index is a property that
           raises NotImplementedError (not AttributeError) when accessed
           -- hasattr() only swallows AttributeError in Python 3, so that
           filter crashes on any op with a StarDep read/write (confirmed
           via a throwaway probe against this exact fixture, e.g. the
           ExternKernelOut matmul op and every AssertScalar/DynamicScalar
           op in the spliced group). isinstance(dep, MemoryDep) is the
           correct filter and is what the real implementation uses too.
        """
        from torch._inductor import ir
        from torch._inductor.dependencies import MemoryDep

        from tests.inductor.for_each_tile_fixtures import split_m_elementwise_fn
        from torch_spyre._inductor.wsr.for_each_tile_lowering import (
            _body_loop_var,
            _consume_tile_dim_markers,
            _stamp_direct_loop_info,
            try_prove_for_each_tile,
        )
        from torch_spyre._inductor.wsr.while_loop_bridge import (
            carry_bindings_for,
            splice_while_loop,
        )

        X = torch.randn(128, 12)
        Y = torch.randn(12, 6)
        graph = self._run_graph(split_m_elementwise_fn, (X, Y))

        while_ops = [op for op in graph.operations if isinstance(op, ir.WhileLoop)]
        self.assertEqual(len(while_ops), 1)
        while_op = while_ops[0]

        result = try_prove_for_each_tile(while_op)
        self.assertTrue(result.accepted, result.reason)
        loop_var = _body_loop_var(while_op)
        self.assertIsNotNone(loop_var)

        with V.set_graph_handler(graph):
            carries = carry_bindings_for(while_op)
            group_ops = splice_while_loop(
                graph, while_op, carries, trip_count=result.trip_count
            )
            _consume_tile_dim_markers(group_ops, graph.operations)

            _stamp_direct_loop_info(group_ops, loop_var, result.trip_count, group_idx=0)

            # op.get_read_writes() needs the live GraphLowering (V.graph)
            # context, same as the stamping call above -- stays inside the
            # `with` block rather than reading loop_info alone afterward.
            saw_advancing_read = False
            for op in group_ops:
                if not getattr(op, "loop_info", None):
                    continue
                info = op.loop_info
                rw = op.get_read_writes()
                reads = [dep for dep in rw.reads if isinstance(dep, MemoryDep)]
                self.assertEqual(
                    len(info.tiled_dims_per_read),
                    len(reads),
                    f"{op.get_name()}: tiled_dims_per_read entry count must "
                    "match the op's own read-dep count",
                )
                # Each per-read entry carries one list per nesting level
                # (parallel to loop_tiled_dims's own [loop_tiled_dims]
                # wrapping) -- this function only ever stamps one level, so
                # entry[-1] is that level's (pos, extent) list.
                for dep, per_read_levels in zip(reads, info.tiled_dims_per_read):
                    per_level = per_read_levels[-1]
                    if dep.index.coeff(loop_var) != 0:
                        # Extent is this dep's own per-trip tile size (its
                        # matched range var's own dep.ranges value), not
                        # trip_count -- these only coincide when tile_size==1.
                        # See _extent_at_pos's docstring: stamping trip_count
                        # here silently doubled the read-side device advance
                        # on any tile_size>1 fixture (confirmed on
                        # add_tiled_fn). split_m_elementwise_fn's tile_size=64
                        # means the real per-trip extent is 64, not
                        # result.trip_count (2).
                        pos = info.loop_tiled_dims[-1][0]
                        matched_var = next(
                            var
                            for var, rng in dep.ranges.items()
                            if dep.index.coeff(var) * rng == dep.index.coeff(loop_var)
                        )
                        self.assertEqual(
                            per_level,
                            [(pos, dep.ranges[matched_var])],
                            f"{op.get_name()}: advancing read must carry "
                            "(resolved_pos, per-trip tile extent)",
                        )
                        saw_advancing_read = True
                    else:
                        self.assertEqual(
                            per_level,
                            [],
                            f"{op.get_name()}: non-advancing read must stay empty",
                        )

                writes = [dep for dep in rw.writes if isinstance(dep, MemoryDep)]
                if writes and writes[0].index.coeff(loop_var) == 0:
                    self.assertEqual(
                        info.output_tiled_dims[-1],
                        [],
                        f"{op.get_name()}: non-advancing write must stay empty",
                    )

        self.assertTrue(
            saw_advancing_read,
            "no op in split_m_elementwise_fn's body has a loop_var-advancing "
            "read -- test would pass vacuously",
        )


if __name__ == "__main__":
    unittest.main()
