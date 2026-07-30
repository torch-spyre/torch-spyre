# Copyright 2025 The Torch-Spyre Authors.
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

"""Device-free unit tests for the provenance forwarding helpers and observer."""

import dataclasses
import logging
import logging.handlers
from types import SimpleNamespace

import pytest
import torch
import torch.fx as fx
from torch._inductor import config as inductor_config
from torch._inductor.virtualized import V

from torch_spyre._inductor.dedup_constants import _drop_constant
from torch_spyre._inductor.loop_info import copy_op_metadata
from torch_spyre._inductor.op_spec import ProvenanceTransform
from torch_spyre._inductor.provenance import (
    _SPYRE_PROV_HISTORY_ATTR,
    build_debug_handle,
    preserve_provenance,
    merge_provenance,
    decompose_provenance,
    SpyreGraphTransformObserver,
    reset_provenance_warnings,
)
from torch_spyre._inductor.split_multi_ops import _make_intermediate_bufs


@pytest.fixture
def prov_logs():
    # Observer tests opt in explicitly; production defaults to level 0 (off).
    with inductor_config.patch("trace.provenance_tracking_level", 1):
        reset_provenance_warnings()
        logger = logging.getLogger("spyre.inductor.provenance")
        handler = logging.handlers.MemoryHandler(capacity=10000)
        prev_level = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(handler)
        try:
            yield handler.buffer
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)


class _Buf:
    """Minimal ComputedBuffer stand-in: mutable origins/origin_node."""

    def __init__(self, origins=None, origin_node=None, name="buf"):
        self.origins = set(origins or ())
        self.origin_node = origin_node
        self._name = name

    def get_name(self):
        return self._name

    def get_operation_name(self):
        return self._name


@dataclasses.dataclass(frozen=True)
class _FrozenBuf:
    """Frozen provenance carrier with a mutable origins container."""

    origins: set[object] = dataclasses.field(default_factory=set)
    origin_node: object | None = None


class TestPreserveProvenance:
    def test_copies_origins_and_node(self):
        old = _Buf(origins={"a", "b"}, origin_node="a")
        new = _Buf()
        preserve_provenance(old, new, pass_name="test_preserve")
        assert new.origins == {"a", "b"}
        assert new.origin_node == "a"

    def test_copies_history(self):
        old = _Buf(origins={"a"})
        history = (ProvenanceTransform("fusion", "fuse"),)
        setattr(old, _SPYRE_PROV_HISTORY_ATTR, history)
        new = _Buf()
        preserve_provenance(old, new, pass_name="test_preserve")
        assert getattr(new, _SPYRE_PROV_HISTORY_ATTR) == (
            *history,
            ProvenanceTransform("rewrite", "test_preserve"),
        )

    def test_does_not_clobber_existing_origin_node(self):
        # A pass that already set origin_node on the new buffer keeps its value.
        old = _Buf(origin_node="a")
        new = _Buf(origin_node="b")
        preserve_provenance(old, new, pass_name="test_preserve")
        assert new.origin_node == "b"

    def test_combines_existing_histories_without_clobbering(self):
        old = _Buf()
        new = _Buf()
        old_transform = ProvenanceTransform("fusion", "old_fusion")
        new_transform = ProvenanceTransform("rewrite", "new_rewrite")
        setattr(old, _SPYRE_PROV_HISTORY_ATTR, (old_transform,))
        setattr(new, _SPYRE_PROV_HISTORY_ATTR, (new_transform,))

        preserve_provenance(old, new, pass_name="test_preserve")

        assert getattr(new, _SPYRE_PROV_HISTORY_ATTR) == (
            old_transform,
            new_transform,
            ProvenanceTransform("rewrite", "test_preserve"),
        )

    def test_preserves_legitimate_repeated_records(self):
        old = _Buf()
        new = _Buf()
        repeated = ProvenanceTransform("rewrite", "same_pass")
        history = (repeated, repeated)
        setattr(old, _SPYRE_PROV_HISTORY_ATTR, history)
        setattr(new, _SPYRE_PROV_HISTORY_ATTR, history)
        preserve_provenance(old, new, pass_name="test_preserve")
        assert getattr(new, _SPYRE_PROV_HISTORY_ATTR) == (
            *history,
            ProvenanceTransform("rewrite", "test_preserve"),
        )

    def test_unions_into_existing_origins(self):
        # origins is unioned in place, not rebound: pre-existing origins survive.
        old = _Buf(origins={"a"})
        new = _Buf(origins={"z"})
        preserve_provenance(old, new, pass_name="test_preserve")
        assert new.origins == {"a", "z"}

    def test_frozen_carrier_copies_primary_and_records_rewrite(self):
        old = _Buf(origins={"a"}, origin_node="a")
        new = _FrozenBuf()

        preserve_provenance(old, new, pass_name="test_preserve")

        assert new.origins == {"a"}
        assert new.origin_node == "a"
        assert getattr(new, _SPYRE_PROV_HISTORY_ATTR)[-1] == ProvenanceTransform(
            "rewrite", "test_preserve"
        )


class TestCopyOpMetadata:
    def test_does_not_copy_provenance_history(self):
        old = _Buf()
        source_history = (ProvenanceTransform("fusion", "source_fusion"),)
        destination_history = (ProvenanceTransform("rewrite", "destination"),)
        setattr(old, _SPYRE_PROV_HISTORY_ATTR, source_history)
        new = _Buf()
        setattr(new, _SPYRE_PROV_HISTORY_ATTR, destination_history)

        copy_op_metadata(old, new)

        assert getattr(new, _SPYRE_PROV_HISTORY_ATTR) == destination_history


class TestMergeProvenance:
    def test_unions_origins_and_appends_fusion_record(self):
        s1, s2 = _Buf(origins={"a"}), _Buf(origins={"b", "c"})
        new = _Buf()
        merge_provenance(
            [s1, s2],
            new,
            pass_name="spyre_fuse_nodes",
            reason="same tile",
        )
        assert new.origins == {"a", "b", "c"}
        assert getattr(new, _SPYRE_PROV_HISTORY_ATTR)[-1] == ProvenanceTransform(
            "fusion", "spyre_fuse_nodes", "same tile"
        )

    def test_clears_single_source_origin_node(self):
        s1 = _Buf(origins={"a"}, origin_node="a")
        s2 = _Buf(origins={"b"}, origin_node="b")
        new = _Buf(origin_node="a")
        merge_provenance([s1, s2], new, pass_name="spyre_fuse_nodes")
        assert new.origin_node is None

    def test_frozen_carrier_clears_primary(self):
        source = _Buf(origins={"a"}, origin_node="a")
        new = _FrozenBuf(origin_node="a")

        merge_provenance([source], new, pass_name="test_merge")

        assert new.origins == {"a"}
        assert new.origin_node is None
        assert getattr(new, _SPYRE_PROV_HISTORY_ATTR)[-1] == ProvenanceTransform(
            "fusion", "test_merge"
        )


class TestDecomposeProvenance:
    def test_each_child_inherits_parent(self):
        old = _Buf(origins={"a"}, origin_node="a")
        c0, c1 = _Buf(name="c0"), _Buf(name="c1")
        decompose_provenance(old, [c0, c1], pass_name="split_multi_ops")
        for c in (c0, c1):
            assert c.origins == {"a"}
            assert c.origin_node == "a"
            assert getattr(c, _SPYRE_PROV_HISTORY_ATTR)[-1] == ProvenanceTransform(
                "decomposition", "split_multi_ops"
            )

    def test_children_have_independent_origins(self):
        # Each child gets its own origins set; mutating one must not affect another.
        old = _Buf(origins={"a"})
        c0, c1 = _Buf(name="c0"), _Buf(name="c1")
        decompose_provenance(old, [c0, c1], pass_name="split_multi_ops")
        c0.origins.add("x")
        assert c1.origins == {"a"}

    def test_semantic_child_keeps_own_origin_and_primary(self):
        old = _Buf(origins={"parent"}, origin_node="parent")
        child = _Buf(origins={"child"}, origin_node="child")

        decompose_provenance(
            old,
            [child],
            pass_name="split_multi_ops",
            inherit_origins=False,
        )

        assert child.origins == {"child"}
        assert child.origin_node == "child"
        assert getattr(child, _SPYRE_PROV_HISTORY_ATTR)[-1] == ProvenanceTransform(
            "decomposition", "split_multi_ops"
        )

    def test_semantic_frozen_child_records_history(self):
        old = _Buf(origins={"parent"}, origin_node="parent")
        child = _FrozenBuf(origins={"child"}, origin_node="child")

        decompose_provenance(
            old,
            [child],
            pass_name="split_multi_ops",
            inherit_origins=False,
        )

        assert child.origins == {"child"}
        assert child.origin_node == "child"
        assert getattr(child, _SPYRE_PROV_HISTORY_ATTR)[-1] == ProvenanceTransform(
            "decomposition", "split_multi_ops"
        )

    def test_frozen_child_inherits_primary_by_default(self):
        old = _Buf(origins={"parent"}, origin_node="parent")
        child = _FrozenBuf()

        decompose_provenance(old, [child], pass_name="split_multi_ops")

        assert child.origins == {"parent"}
        assert child.origin_node == "parent"
        assert getattr(child, _SPYRE_PROV_HISTORY_ATTR)[-1] == ProvenanceTransform(
            "decomposition", "split_multi_ops"
        )

    def test_preserves_complete_history(self):
        old = _Buf(origins={"parent"}, origin_node="parent")
        history = tuple(
            ProvenanceTransform("rewrite", f"pass_{index}") for index in range(100)
        )
        setattr(old, _SPYRE_PROV_HISTORY_ATTR, history)
        child = _Buf()

        decompose_provenance(old, [child], pass_name="split_multi_ops")

        recorded = getattr(child, _SPYRE_PROV_HISTORY_ATTR)
        assert recorded[:-1] == history
        assert recorded[-1] == ProvenanceTransform("decomposition", "split_multi_ops")


class _FakeGraphLowering:
    """Minimal lowering surface exercised by _make_intermediate_bufs."""

    def __init__(self, graph, operations):
        self.graph = graph
        self.operations = operations
        self.env = {}
        self.name_to_buffer = {}

    def run_node(self, node):
        buffer = _Buf(
            origins={node},
            origin_node=node,
            name=f"split_buf_{len(self.env)}",
        )
        tensor_box = SimpleNamespace(
            data=SimpleNamespace(data=buffer),
            get_name=buffer.get_name,
        )
        self.operations.append(buffer)
        self.env[node] = tensor_box
        return tensor_box


class TestSplitMultiOpsProvenance:
    def test_materialized_children_keep_source_identity_and_history(self):
        graph = fx.Graph()
        input_node = graph.placeholder("arg")
        input_node.meta["val"] = torch.empty(2, device="meta")
        orig_node = graph.call_function(
            torch.ops.aten.add.Tensor,
            (input_node, input_node),
        )
        orig_node.meta.update(
            {
                "stack_trace": (
                    '  File "/tmp/model.py", line 41, in forward\n    return x + 1\n'
                ),
                "original_aten": torch.ops.aten.add.Tensor,
            }
        )

        parent = _Buf(
            origins={orig_node},
            origin_node=orig_node,
            name="parent_buf",
        )
        prior = ProvenanceTransform("rewrite", "prior_pass")
        setattr(parent, _SPYRE_PROV_HISTORY_ATTR, (prior,))

        operations = []
        graph_lowering = _FakeGraphLowering(graph, operations)
        _make_intermediate_bufs(
            [
                (
                    "constant",
                    1,
                    (),
                    {"fill_value": 1.0, "dtype": torch.float32},
                ),
                ("relu", 2, (0,), {}),
            ],
            {1: torch.float32, 2: torch.float32},
            {0: input_node.name},
            SimpleNamespace(dtype=torch.float32, device=torch.device("cpu")),
            operations,
            0,
            graph_lowering,
            orig_node,
            parent,
            final_op_name="add",
        )

        assert len(operations) == 2
        handles = [build_debug_handle(buffer) for buffer in operations]
        assert all(handle is not None for handle in handles)
        assert {handle.source.to_str() for handle in handles} == {"/tmp/model.py:41:0"}
        assert {handle.aten_op for handle in handles} == {
            str(torch.ops.spyre.constant.default),
            str(torch.ops.aten.relu.default),
        }
        assert len({handle.id for handle in handles}) == 2
        assert len({handle.ir_chain for handle in handles}) == 2

        expected_reasons = {"materialize constant", "materialize relu"}
        for buffer, handle in zip(operations, handles):
            child_node = next(iter(buffer.origins))
            assert orig_node not in buffer.origins
            assert buffer.origin_node is child_node
            assert child_node.meta["from_node"][0].name == orig_node.name
            assert child_node.meta["from_node"][0].pass_name == "split_multi_ops"
            assert handle.transform_history[0] == prior
            assert handle.transform_history[-1].kind == "decomposition"
            assert handle.transform_history[-1].pass_name == "split_multi_ops"
            assert handle.transform_history[-1].reason in expected_reasons


class _NodeListTarget(list):
    """Stand-in for list[BaseSchedulerNode]; each unit's buffer is .node."""


def _unit(buf):
    class _N:
        def __init__(self, b):
            self.node = b

    return _N(buf)


class TestObserverDetection:
    def test_regression_warns(self, prov_logs):
        b = _Buf(origins={"a"})
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "bad_pass_regress", kind="node"):
            b.origins = set()  # pass wrongly drops provenance
        assert any("bad_pass_regress" in r.getMessage() for r in prov_logs)

    def test_preserved_no_warning(self, prov_logs):
        b = _Buf(origins={"a"})
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "good_pass_keep", kind="node"):
            pass  # no change
        assert not any("good_pass_keep" in r.getMessage() for r in prov_logs)

    def test_new_unattributed_buffer_warns(self, prov_logs):
        target = _NodeListTarget([])
        with SpyreGraphTransformObserver(target, "creates_bare_buf", kind="node"):
            target.append(_unit(_Buf(origins=set())))
        assert any("creates_bare_buf" in r.getMessage() for r in prov_logs)

    def test_allowlisted_new_buffer_silent(self, prov_logs):
        target = _NodeListTarget([])
        with SpyreGraphTransformObserver(target, "insert_restickify", kind="node"):
            target.append(_unit(_Buf(origins=set())))  # source-less by design
        assert not any("spyre-provenance" in r.getMessage() for r in prov_logs)

    def test_partial_origin_loss_warns(self, prov_logs):
        b = _Buf(origins={"a", "b"})
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "partial_drop_pass", kind="node"):
            b.origins = {"a"}  # loses "b" but is not empty
        assert any("partial_drop_pass" in r.getMessage() for r in prov_logs)

    def test_sourceless_creation_pass_partial_loss_warns(self, prov_logs):
        b = _Buf(origins={"a", "b"})
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "insert_restickify", kind="node"):
            b.origins = {"a"}
        assert any("insert_restickify" in r.getMessage() for r in prov_logs)

    def test_level_zero_disables_observer(self, prov_logs):
        b = _Buf(origins={"a"})
        target = _NodeListTarget([_unit(b)])
        with inductor_config.patch("trace.provenance_tracking_level", 0):
            with SpyreGraphTransformObserver(target, "drop_at_level_0", kind="node"):
                b.origins = set()
        assert not any("drop_at_level_0" in r.getMessage() for r in prov_logs)

    @pytest.mark.parametrize("level", [1, 2])
    def test_positive_levels_enable_observer(self, prov_logs, level):
        pass_name = f"drop_at_level_{level}"
        b = _Buf(origins={"a"})
        target = _NodeListTarget([_unit(b)])
        with inductor_config.patch("trace.provenance_tracking_level", level):
            with SpyreGraphTransformObserver(target, pass_name, kind="node"):
                b.origins = set()
        assert any(pass_name in r.getMessage() for r in prov_logs)

    def test_never_raises_on_bad_target(self, prov_logs):
        # Enumeration failure must not propagate.
        with SpyreGraphTransformObserver(object(), "weird_target", kind="node"):
            pass

    def test_replacement_buffer_partial_loss_warns(self, prov_logs):
        # A pass that swaps a buffer for a fresh SAME-NAME object holding a
        # subset of origins ({a,b} -> {a}) must still be flagged; identity-keyed
        # snapshots would miss it because the object id changed.
        target = _NodeListTarget([_unit(_Buf(origins={"a", "b"}, name="x"))])
        with SpyreGraphTransformObserver(target, "replace_pass", kind="node"):
            target[0] = _unit(_Buf(origins={"a"}, name="x"))  # same name, lost "b"
        assert any("replace_pass" in r.getMessage() for r in prov_logs)

    def test_origin_node_loss_warns(self, prov_logs):
        # A pass that keeps origins but clears origin_node still loses the
        # authoritative source/aten pointer; a non-allowlisted pass must warn.
        b = _Buf(origins={"a"}, origin_node="a")
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "drops_origin_node", kind="node"):
            b.origin_node = None  # origins intact, authoritative pointer gone
        assert any("drops_origin_node" in r.getMessage() for r in prov_logs)

    def test_transform_history_loss_warns(self, prov_logs):
        b = _Buf(origins={"a"})
        setattr(b, _SPYRE_PROV_HISTORY_ATTR, (ProvenanceTransform("fusion", "fuse"),))
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "drops_history", kind="node"):
            setattr(b, _SPYRE_PROV_HISTORY_ATTR, ())
        assert any("transformation history" in r.getMessage() for r in prov_logs)

    def test_intentional_remap_exemption_is_silent(self, prov_logs, monkeypatch):
        import torch_spyre._inductor.provenance as provenance

        pass_name = "intentional_remap"
        monkeypatch.setattr(
            provenance,
            "INTENTIONAL_PROVENANCE_REMAP_PASSES",
            frozenset({pass_name}),
        )
        b = _Buf(origins={"old"}, origin_node="old")
        setattr(
            b,
            _SPYRE_PROV_HISTORY_ATTR,
            (ProvenanceTransform("rewrite", "old_pass"),),
        )
        target = _NodeListTarget([_unit(b)])

        with SpyreGraphTransformObserver(target, pass_name, kind="node"):
            b.origins = {"new"}
            b.origin_node = "new"
            setattr(b, _SPYRE_PROV_HISTORY_ATTR, ())

        assert not any(pass_name in record.getMessage() for record in prov_logs)

    def test_sourceless_creation_pass_origin_node_loss_warns(self, prov_logs):
        # Source-less helper creation does not excuse provenance loss on an
        # existing buffer reconstructed by the same pass.
        b = _Buf(origins={"a"}, origin_node="a")
        target = _NodeListTarget([_unit(b)])
        with SpyreGraphTransformObserver(target, "insert_restickify", kind="node"):
            b.origin_node = None
        assert any("insert_restickify" in r.getMessage() for r in prov_logs)

    def test_graphlowering_new_unattributed_buffer_warns(self, prov_logs):
        graph = SimpleNamespace(operations=[])
        pass_name = "creates_bare_graphlowering_buffer"

        with SpyreGraphTransformObserver(graph, pass_name, kind="graphlowering"):
            graph.operations.append(_Buf(origins=set(), name="new"))

        assert any(pass_name in record.getMessage() for record in prov_logs)

    def test_removed_buffer_without_forwarding_warns(self, prov_logs):
        removed = _Buf(origins={"lost"}, origin_node="lost", name="removed")
        survivor = _Buf(origins={"kept"}, origin_node="kept", name="survivor")
        graph = SimpleNamespace(operations=[removed, survivor])
        pass_name = "drops_removed_buffer"

        with SpyreGraphTransformObserver(graph, pass_name, kind="graphlowering"):
            graph.operations.remove(removed)

        assert any(
            pass_name in record.getMessage()
            and "removed a buffer" in record.getMessage()
            for record in prov_logs
        )

    def test_dedup_forwards_removed_constant_provenance(self, prov_logs):
        canonical = _Buf(origins={"a"}, origin_node="a", name="canonical")
        duplicate = _Buf(origins={"b"}, origin_node="b", name="duplicate")
        canonical_history = (ProvenanceTransform("rewrite", "canonical_pass"),)
        duplicate_history = (ProvenanceTransform("rewrite", "duplicate_pass"),)
        setattr(canonical, _SPYRE_PROV_HISTORY_ATTR, canonical_history)
        setattr(duplicate, _SPYRE_PROV_HISTORY_ATTR, duplicate_history)
        graph = SimpleNamespace(
            operations=[canonical, duplicate],
            removed_buffers=set(),
            name_to_buffer={"canonical": canonical, "duplicate": duplicate},
            name_to_op={"canonical": canonical, "duplicate": duplicate},
            name_to_users={"canonical": [], "duplicate": []},
        )
        pass_name = "dedup_and_promote_constants"

        with V.set_graph_handler(graph):
            with SpyreGraphTransformObserver(graph, pass_name, kind="graphlowering"):
                _drop_constant(graph.operations, duplicate, canonical)

        assert graph.operations == [canonical]
        assert canonical.origins == {"a", "b"}
        assert canonical.origin_node is None
        history = getattr(canonical, _SPYRE_PROV_HISTORY_ATTR)
        assert history[-1] == ProvenanceTransform(
            "fusion", pass_name, "duplicate constant"
        )
        assert duplicate_history[0] in history
        assert not any(pass_name in record.getMessage() for record in prov_logs)

    def test_deadcode_removal_is_silent(self, prov_logs):
        dead = _Buf(origins={"dead"}, origin_node="dead", name="dead")
        graph = SimpleNamespace(operations=[dead])

        with SpyreGraphTransformObserver(
            graph,
            "deadcode_elimination",
            kind="graphlowering",
        ):
            graph.operations.remove(dead)

        assert not any(
            "deadcode_elimination" in record.getMessage() for record in prov_logs
        )


class TestPipelineWrapping:
    def test_node_pipeline_observes_each_pass(self, prov_logs):
        from torch_spyre._inductor.passes import _SpyreNodePassPipeline

        def dropping_node_pass(nodes):
            for n in nodes:
                n.node.origins = set()  # wrongly clears provenance
            return nodes

        pipeline = _SpyreNodePassPipeline([dropping_node_pass])
        # Force the device guard on so the loop body runs off-device.
        pipeline._has_spyre_device = lambda target: True
        pipeline([_unit(_Buf(origins={"a"}))])
        assert any("dropping_node_pass" in r.getMessage() for r in prov_logs)

    def test_node_pipeline_reconciles_returned_list(self, prov_logs):
        from torch_spyre._inductor.passes import _SpyreNodePassPipeline

        def replacing_node_pass(nodes):
            assert len(nodes) == 1
            return [_unit(_Buf(origins={"a"}, name="x"))]

        pipeline = _SpyreNodePassPipeline([replacing_node_pass])
        pipeline._has_spyre_device = lambda target: True
        pipeline([_unit(_Buf(origins={"a", "b"}, name="x"))])

        assert any("replacing_node_pass" in record.getMessage() for record in prov_logs)

    def test_graphlowering_pipeline_observes_each_pass(self, prov_logs, monkeypatch):
        import torch_spyre._inductor.passes as passes_mod

        class _FakeGraph:
            def __init__(self, ops):
                self.operations = ops

        def dropping_gl_pass(graph):
            for op in graph.operations:
                op.origins = set()

        monkeypatch.setattr(
            passes_mod, "_operations_have_spyre_device", lambda ops: True
        )
        pipeline = passes_mod.CustomPreSchedulingPasses()
        pipeline.passes = [dropping_gl_pass]
        pipeline(_FakeGraph([_Buf(origins={"a"})]))
        assert any("dropping_gl_pass" in r.getMessage() for r in prov_logs)

    def test_warning_dedup_resets_per_pipeline_run(self, prov_logs):
        from torch_spyre._inductor.passes import _SpyreNodePassPipeline

        def dedup_reset_pass(nodes):
            for n in nodes:
                n.node.origins = set()  # drop provenance every run
            return nodes

        pipeline = _SpyreNodePassPipeline([dedup_reset_pass])
        pipeline._has_spyre_device = lambda target: True

        # The buffer grows per emission and is never cleared between runs:
        # this is what proves the observer now genuinely re-emits per compile
        # (the cumulative count strictly increases on the second run) rather
        # than being silenced by Python's per-process warning registry, which
        # this test would have caught under the old warnings.warn emission.
        pipeline([_unit(_Buf(origins={"a"}))])
        after_first = sum(1 for r in prov_logs if "dedup_reset_pass" in r.getMessage())
        pipeline([_unit(_Buf(origins={"a"}))])
        after_second = sum(1 for r in prov_logs if "dedup_reset_pass" in r.getMessage())
        assert after_first >= 1
        assert after_second > after_first  # reset -> a new run warns again
