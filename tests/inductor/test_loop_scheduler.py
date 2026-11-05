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

"""Unit tests for ``loop_scheduler``: ``LoopSchedulerNode`` and the patches.

Everything here runs on real Inductor objects on the CPU: a real
``GraphLowering``, real ``ComputedBuffer``s and ``ExternKernel``s, a real
``LoopNode`` and a real ``Scheduler``.  The scheduler's front half (node
creation through ``compute_ancestors``) is driven step by step in the order
``Scheduler._init`` runs it, because the compile path does not hand the
scheduler a ``LoopNode`` yet (``passes.py`` still expands the nodes).
"""

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import pytest
import sympy
import torch
from torch import fx
from torch._inductor import ir
from torch._inductor.debug import DebugContext
from torch._inductor.dependencies import StarDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    StorageBox,
    TensorBox,
)
from torch._inductor.scheduler import (
    ExternKernelSchedulerNode,
    Scheduler,
    SchedulerNode,
)
from torch._inductor.virtualized import V

from torch_spyre._inductor import loop_scheduler as ls
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.loop_node import LoopNode
from torch_spyre._inductor.loop_scheduler import (
    SUPPORTED_RELEASES,
    LoopSchedulerNode,
    apply_loop_scheduler_patches,
    loop_scheduler_patches_for_compile,
    loop_scheduler_supported,
)
from torch_spyre._inductor.scheduler import (
    CountedLoopSchedulerNode,
    _all_scheduler_nodes,
)

_PATCHED = (
    "create_scheduler_node",
    "update_zero_dim_cpu_tensor",
    "compute_ancestors",
    "dead_node_elimination",
)


def _current():
    return {name: Scheduler.__dict__[name] for name in _PATCHED}


def _version(text):
    return mock.patch.object(torch, "__version__", text)


class _Graph(unittest.TestCase):
    """A real ``GraphLowering`` on the CPU to build IR and a ``Scheduler`` in."""

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        graph = GraphLowering(gm)
        graph.graph_outputs = []
        self._ctx = [V.set_graph_handler(graph), DebugContext()]
        for ctx in self._ctx:
            ctx.__enter__()
        self.addCleanup(self._exit)
        self.outputs = []
        V.graph.get_output_names = lambda: list(self.outputs)

    def _exit(self):
        for ctx in reversed(self._ctx):
            ctx.__exit__(None, None, None)

    @staticmethod
    def _input(name):
        buf = V.graph.name_to_buffer.get(name)
        if buf is None:
            buf = InputBuffer(
                name=name,
                layout=FixedLayout(torch.device("cpu"), torch.float32, [8], [1]),
            )
            V.graph.name_to_buffer[name] = buf
            V.graph.graph_inputs[name] = TensorBox(StorageBox(buf))
        return buf

    def _buf(self, name, reads="x", mutates=None, flip=False):
        """A ``ComputedBuffer`` reading ``reads`` (an input or a buffer),
        element ``7 - i`` instead of ``i`` when ``flip``."""
        src = V.graph.name_to_buffer.get(reads) or self._input(reads)
        box = TensorBox(StorageBox(src))
        pw = Pointwise.create(
            device=torch.device("cpu"),
            dtype=torch.float32,
            inner_fn=lambda index: box.make_loader()([7 - index[0]] if flip else index),
            ranges=[8],
        )
        layout = (
            FixedLayout(torch.device("cpu"), torch.float32, [8], [1])
            if mutates is None
            else MutationLayoutSHOULDREMOVE(V.graph.name_to_buffer[mutates])
        )
        buf = ComputedBuffer(name=name, layout=layout, data=pw.data.data)
        buf.operation_name = name
        V.graph.name_to_buffer[name] = buf
        V.graph.name_to_op[name] = buf
        return buf

    def _extern(self, name, reads="x"):
        src = V.graph.name_to_buffer.get(reads) or self._input(reads)
        op = ir.ExternKernel(
            None,
            FixedLayout(torch.device("cpu"), torch.float32, [8], [1]),
            [src],
        )
        op.name = name
        op.operation_name = name
        V.graph.name_to_buffer[name] = op
        V.graph.name_to_op[name] = op
        return op

    @staticmethod
    def _loop(name, body, trip=4):
        return LoopNode(
            name, body, induction=sympy.Symbol(f"u_{name}"), trip_count=trip
        )

    @staticmethod
    def _scheduler():
        return Scheduler([])

    @staticmethod
    def _front_half(sched, operations):
        """``Scheduler._init`` from node creation to ``compute_ancestors``,
        each stage the real Inductor method, in ``_init``'s order."""
        sched.nodes = [sched.create_scheduler_node(n) for n in operations]
        sched.update_zero_dim_cpu_tensor()
        for node in sched.nodes:
            node.prune_deps()
        sched.name_to_node = {n.get_name(): n for n in sched.nodes}
        sched.name_to_buf = {
            buf.get_name(): buf for n in sched.nodes for buf in n.get_outputs()
        }
        sched.name_to_fused_node = dict(sched.name_to_node)
        sched.mutation_real_name = {}
        sched.mutation_renames = {}
        sched.compute_dependencies()
        sched.nodes = sched.topological_sort_schedule(sched.nodes)
        sched.dead_node_elimination()
        sched.name_to_fused_node = {n.get_name(): n for n in sched.nodes}
        sched.compute_ancestors()
        return sched.nodes


# ---------------------------------------------------------------------------
# The patches are installed for the block and restored exactly
# ---------------------------------------------------------------------------


class TestPatchInstallRestore(unittest.TestCase):
    def setUp(self):
        self._v = _version("2.14.0")
        self._v.start()
        self.addCleanup(self._v.stop)

    def test_replaces_four_methods_and_restores_identical_originals(self):
        before = _current()
        with apply_loop_scheduler_patches():
            inside = _current()
            for name in _PATCHED:
                self.assertIsNot(inside[name], before[name], name)
        after = _current()
        for name in _PATCHED:
            self.assertIs(after[name], before[name], name)

    def test_restores_after_exception_in_block(self):
        before = _current()
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with apply_loop_scheduler_patches():
                raise RuntimeError("boom")
        for name, fn in _current().items():
            self.assertIs(fn, before[name], name)

    def test_nested_use_restores_each_level(self):
        before = _current()
        with apply_loop_scheduler_patches():
            outer = _current()
            with apply_loop_scheduler_patches():
                inner = _current()
                for name in _PATCHED:
                    self.assertIsNot(inner[name], outer[name], name)
            for name, fn in _current().items():
                self.assertIs(fn, outer[name], name)
        for name, fn in _current().items():
            self.assertIs(fn, before[name], name)

    def test_patched_surface_is_listed(self):
        surface = " ".join(ls.PATCHED_SURFACE)
        for name in _PATCHED:
            self.assertIn(name, surface)
        for name in (
            "compute_dependencies",
            "topological_sort_schedule",
            "_populate_stream_assignments",
            "refresh_group_node_dependencies",
            "init_group_node",
            "add_fake_dep",
            "set_read_writes",
            "has_side_effects",
            "NodeUser",
            "SchedulerBuffer",
            "get_stream_idx",
            "get_origin_node",
            "get_mempool",
        ):
            self.assertIn(name, surface)

    def test_enable_spyre_context_installs_and_restores(self):
        from torch_spyre._inductor.patches import enable_spyre_context

        before = _current()
        with enable_spyre_context([]):
            inside = _current()
            for name in _PATCHED:
                self.assertIsNot(inside[name], before[name], name)
        for name, fn in _current().items():
            self.assertIs(fn, before[name], name)


# ---------------------------------------------------------------------------
# Release gate
# ---------------------------------------------------------------------------


class TestReleaseGate(_Graph):
    def test_supported_releases(self):
        # One torch release at a time: the one pyproject.toml pins.  The previous
        # release is refused too, not just a future one.
        for text, expected in (
            ("2.12.0", False),
            ("2.13.0+cpu", False),
            ("2.14.0", True),
            ("2.14.1+cpu", True),
            ("2.15.0", False),
        ):
            with self.subTest(text), _version(text):
                self.assertIs(loop_scheduler_supported(), expected)

    def test_supported_release_is_the_one_pyproject_pins(self):
        # SUPPORTED_RELEASES is a literal, so bumping the torch pin without
        # moving it would turn every counted loop off.  Make that fail in the
        # change that bumps the pin.  tools/get_torch_minor.py is the helper the
        # CI image build uses to read the same pin.
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "get_torch_minor", root / "tools" / "get_torch_minor.py"
        )
        assert spec is not None and spec.loader is not None
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        major, minor = tool.extract_torch_minor(
            (root / "pyproject.toml").read_bytes()
        ).split(".")
        self.assertEqual(
            [(int(major), int(minor))],
            list(SUPPORTED_RELEASES),
            "pyproject.toml pins a different torch than SUPPORTED_RELEASES: "
            "update loop_scheduler.SUPPORTED_RELEASES (Step 10.6a of the "
            "upgrade-pytorch-version skill)",
        )

    def test_apply_raises_on_unsupported_and_patches_nothing(self):
        before = _current()
        for text in ("2.12.0", "2.13.0+cpu", "2.15.0"):
            with self.subTest(text), _version(text):
                with self.assertRaises(Unsupported):
                    with apply_loop_scheduler_patches():
                        pass
                for name, fn in _current().items():
                    self.assertIs(fn, before[name], name)

    def test_per_compile_unsupported_raises_only_on_a_loop_node(self):
        before = _current()
        for text in ("2.12.0", "2.13.0+cpu", "2.15.0"):
            with self.subTest(text), _version(text):
                with loop_scheduler_patches_for_compile():
                    # A compile with no LoopNode is unaffected.
                    a = self._buf(f"a_{text}")
                    self.outputs = [a.get_name()]
                    nodes = self._front_half(self._scheduler(), [a])
                    self.assertEqual([n.get_name() for n in nodes], [a.get_name()])
                    self.assertIsInstance(nodes[0], SchedulerNode)
                    # A LoopNode is refused loudly, not with NotImplementedError.
                    t = self._buf(f"t_{text}")
                    loop = self._loop(f"loop_{text}", [t])
                    with self.assertRaises(Unsupported):
                        self._scheduler().create_scheduler_node(loop)
                for name, fn in _current().items():
                    self.assertIs(fn, before[name], name)

    def test_per_compile_supported_installs_the_patches(self):
        before = _current()
        for text in ("2.14.0", "2.14.1+cpu"):
            with self.subTest(text), _version(text):
                with loop_scheduler_patches_for_compile():
                    t = self._buf(f"t_{text}")
                    loop = self._loop(f"loop_{text}", [t])
                    snode = self._scheduler().create_scheduler_node(loop)
                    self.assertIsInstance(snode, LoopSchedulerNode)
                for name, fn in _current().items():
                    self.assertIs(fn, before[name], name)


# ---------------------------------------------------------------------------
# LoopSchedulerNode built from real scheduler nodes
# ---------------------------------------------------------------------------


class TestLoopSchedulerNode(_Graph):
    def setUp(self):
        super().setUp()
        self._v = _version("2.14.0")
        self._v.start()
        self.addCleanup(self._v.stop)
        self._p = apply_loop_scheduler_patches()
        self._p.__enter__()
        self.addCleanup(self._p.__exit__, None, None, None)

    def _flat_loop(self):
        t = self._buf("t", reads="x")
        u = self._buf("u", reads="t")
        return self._loop("loopnode0", [t, u]), t, u

    def test_is_a_counted_loop_with_the_loop_node(self):
        loop, _, _ = self._flat_loop()
        snode = self._scheduler().create_scheduler_node(loop)
        self.assertIsInstance(snode, CountedLoopSchedulerNode)
        self.assertIs(snode.node, loop)
        self.assertEqual(snode.get_name(), "loopnode0")
        self.assertEqual(snode.get_first_name(), "loopnode0")
        self.assertEqual(snode.loop_count, 4)
        self.assertEqual(snode.min_order, 0)
        self.assertEqual(snode.max_order, 0)

    def test_external_reads_exclude_the_loops_own_buffers(self):
        loop, _, _ = self._flat_loop()
        snode = self._scheduler().create_scheduler_node(loop)
        self.assertEqual(set(snode.get_buffer_names()), {"t", "u"})
        self.assertEqual({d.name for d in snode.read_writes.reads}, {"x"})
        self.assertEqual({d.name for d in snode.read_writes.writes}, {"t", "u"})
        self.assertNotIn("t", {d.name for d in snode.unmet_dependencies})

    def test_internal_read_at_another_index_is_not_external(self):
        # merge_list subtracts writes from reads only when the deps are
        # equal; a read of an internal buffer at another index survives it.
        t = self._buf("t", reads="x")
        u = self._buf("u", reads="t", flip=True)
        snode = self._scheduler().create_scheduler_node(self._loop("loopnode0", [t, u]))
        self.assertEqual({d.name for d in snode.read_writes.reads}, {"x"})

    def test_add_fake_dep_ignores_internal_adds_external(self):
        loop, _, _ = self._flat_loop()
        self._input("y")
        snode = self._scheduler().create_scheduler_node(loop)
        snode.add_fake_dep(StarDep("t"))
        self.assertNotIn("t", {d.name for d in snode.read_writes.reads})
        snode.add_fake_dep(StarDep("other"))
        self.assertIn("other", {d.name for d in snode.read_writes.reads})
        # Through set_read_writes, so unmet_dependencies is refreshed too
        # (B.11 failure 5).
        self.assertIn("other", {d.name for d in snode.unmet_dependencies})

    def test_can_inplace_is_false(self):
        loop, _, _ = self._flat_loop()
        snode = self._scheduler().create_scheduler_node(loop)
        for dep in snode.read_writes.reads:
            self.assertFalse(snode.can_inplace(dep))

    def test_get_nodes_returns_the_leaves_of_a_flat_loop(self):
        loop, t, u = self._flat_loop()
        snode = self._scheduler().create_scheduler_node(loop)
        leaves = snode.get_nodes()
        self.assertEqual([n.get_name() for n in leaves], ["t", "u"])
        self.assertTrue(all(isinstance(n, SchedulerNode) for n in leaves))
        self.assertEqual([n.node for n in leaves], [t, u])
        # The body's buffers resolve to the node that sits in Scheduler.nodes.
        for buf in snode.get_outputs():
            self.assertIs(buf.defining_op, snode)

    def test_nested_loops_nest(self):
        a = self._buf("a", reads="x")
        b = self._buf("b", reads="a")
        c = self._buf("c", reads="b")
        inner = self._loop("loopnode1", [b], trip=2)
        outer = self._loop("loopnode0", [a, inner, c], trip=4)
        snode = self._scheduler().create_scheduler_node(outer)
        kids = snode.get_nodes()
        self.assertEqual([type(k) for k in kids][1], LoopSchedulerNode)
        self.assertIs(kids[1].node, inner)
        self.assertEqual(kids[1].loop_count, 2)
        self.assertEqual([n.get_name() for n in kids[1].get_nodes()], ["b"])
        self.assertEqual(
            [n.get_name() for n in _all_scheduler_nodes([snode])], ["a", "b", "c"]
        )
        # Every leaf buffer resolves to the OUTERMOST node.
        for buf in snode.get_outputs():
            self.assertIs(buf.defining_op, snode)
        self.assertEqual({d.name for d in snode.read_writes.reads}, {"x"})

    def test_has_side_effects_from_children(self):
        acc = self._buf("acc", reads="x")
        loop, _, _ = self._flat_loop()
        snode = self._scheduler().create_scheduler_node(loop)
        self.assertFalse(snode.has_side_effects())
        upd = self._buf("upd", reads="x", mutates="acc")
        with mock.patch.object(
            SchedulerNode,
            "has_side_effects",
            lambda self: isinstance(self.node.layout, MutationLayoutSHOULDREMOVE),
        ):
            loop2 = self._loop("loopnode9", [upd])
            self.assertTrue(
                self._scheduler().create_scheduler_node(loop2).has_side_effects()
            )
        self.assertIsNotNone(acc)

    def test_update_mutated_names_renames_external_reads(self):
        # An external node mutated ``x`` earlier: its new name is ``m``.  The
        # loop must read ``m`` (and so depend on the mutator), as a flat
        # reader would; the children's reads are renamed too.
        loop, _, _ = self._flat_loop()
        snode = self._scheduler().create_scheduler_node(loop)
        snode.update_mutated_names({"x": "m"})
        self.assertEqual({d.name for d in snode.read_writes.reads}, {"m"})
        self.assertIn("m", {d.name for d in snode.unmet_dependencies})
        t_node = snode.get_nodes()[0]
        self.assertEqual({d.name for d in t_node.read_writes.reads}, {"m"})


# ---------------------------------------------------------------------------
# The front half of Scheduler._init on a real LoopNode
# ---------------------------------------------------------------------------


class TestSchedulerFrontHalf(_Graph):
    def setUp(self):
        super().setUp()
        self._v = _version("2.14.0")
        self._v.start()
        self.addCleanup(self._v.stop)
        self._p = apply_loop_scheduler_patches()
        self._p.__enter__()
        self.addCleanup(self._p.__exit__, None, None, None)

    def test_loop_after_producer_internal_buffers_live(self):
        pre = self._buf("pre", reads="x")
        t = self._buf("t", reads="pre")
        u = self._buf("u", reads="t")
        post = self._buf("post", reads="u")
        self.outputs = ["post"]
        loop = self._loop("loopnode0", [t, u])
        sched = self._scheduler()
        nodes = self._front_half(sched, [loop, pre, post])
        self.assertEqual([n.get_name() for n in nodes], ["pre", "loopnode0", "post"])
        # ``t`` is read only inside the loop (B.11 failure 7).
        self.assertNotIn("t", V.graph.removed_buffers)
        self.assertNotIn("u", V.graph.removed_buffers)
        loop_snode = nodes[1]
        self.assertEqual(set(loop_snode.ancestors), {"pre"})
        self.assertEqual(set(nodes[2].ancestors), {"pre", "loopnode0"})
        # Children are not in Scheduler.nodes; they take the loop's order,
        # and map to the loop in name_to_fused_node, as fused members do.
        for child in loop_snode.get_nodes():
            self.assertIs(sched.name_to_fused_node[child.get_name()], loop_snode)
            self.assertEqual(child.min_order, loop_snode.min_order)
            self.assertEqual(child.max_order, loop_snode.max_order)
            self.assertEqual(child.ancestors, loop_snode.ancestors)

    def test_dead_loop_is_removed_with_its_body(self):
        t = self._buf("t", reads="x")
        u = self._buf("u", reads="t")
        keep = self._buf("keep", reads="x")
        self.outputs = ["keep"]
        loop = self._loop("loopnode0", [t, u])
        nodes = self._front_half(self._scheduler(), [loop, keep])
        self.assertEqual([n.get_name() for n in nodes], ["keep"])
        for name in ("loopnode0", "t", "u"):
            self.assertIn(name, V.graph.removed_operations)
        self.assertIn("t", V.graph.removed_buffers)

    def test_unread_body_buffer_of_a_live_loop(self):
        # Records today's rule: the loop is atomic, so a body op nobody reads
        # still runs, and its buffer is kept (not marked removed while its op
        # stores it).  Dead compute, not wrong code.
        t = self._buf("t", reads="x")
        dead = self._buf("dead", reads="t")
        out = self._buf("out", reads="t")
        self.outputs = ["out"]
        loop = self._loop("loopnode0", [t, dead, out])
        nodes = self._front_half(self._scheduler(), [loop])
        self.assertEqual(
            [n.get_name() for n in nodes[0].get_nodes()], ["t", "dead", "out"]
        )
        for name in ("t", "dead", "out"):
            self.assertNotIn(name, V.graph.removed_buffers)

    def test_child_mutating_an_internal_buffer_is_no_self_dependency(self):
        # B.11 failure 4: compute_dependencies adds StarDep(t) for upd, whose
        # prior writer is the loop's own child.
        t = self._buf("t", reads="x")
        upd = self._buf("upd", reads="t", mutates="t")
        out = self._buf("out", reads="t")
        self.outputs = ["out"]
        loop = self._loop("loopnode0", [t, upd, out])
        nodes = self._front_half(self._scheduler(), [loop])
        self.assertEqual([n.get_name() for n in nodes], ["loopnode0"])
        loop_snode = nodes[0]
        self.assertEqual(
            {d.name for d in loop_snode.unmet_dependencies}
            & set(loop_snode.get_buffer_names()),
            set(),
        )

    def test_child_mutates_external_buffer_later_child_reads_it(self):
        # The named risky case of the self-dependency fix: a child mutates a
        # buffer that a LATER child of the same trip reads.
        acc = self._buf("acc", reads="x")
        upd = self._buf("upd", reads="acc", mutates="acc")
        rd = self._buf("rd", reads="acc")
        post = self._buf("post", reads="rd")
        self.outputs = ["post", "acc"]
        loop = self._loop("loopnode0", [upd, rd])
        with mock.patch.object(
            SchedulerNode,
            "has_side_effects",
            lambda self: isinstance(self.node.layout, MutationLayoutSHOULDREMOVE),
        ):
            nodes = self._front_half(self._scheduler(), [acc, loop, post])
        self.assertEqual([n.get_name() for n in nodes], ["acc", "loopnode0", "post"])
        loop_snode = nodes[1]
        # Body order is the program order; the scheduler never reorders it.
        self.assertEqual([n.get_name() for n in loop_snode.get_nodes()], ["upd", "rd"])
        self.assertIn("acc", loop_snode.ancestors)

    def test_extern_in_body_lands_immediately_before_its_loop(self):
        pre = self._buf("pre", reads="x")
        ext = self._extern("ext", reads="x")
        t = self._buf("t", reads="pre")
        self.outputs = ["t", "ext"]
        loop = self._loop("loopnode0", [ext, t])
        sched = self._scheduler()
        sched.nodes = [sched.create_scheduler_node(n) for n in [pre, loop]]
        sched.update_zero_dim_cpu_tensor()
        names = [n.get_name() for n in sched.nodes]
        self.assertEqual(names, ["pre", "ext", "loopnode0"])
        self.assertIsInstance(sched.nodes[1], ExternKernelSchedulerNode)
        self.assertNotIn("ext", [n.get_name() for n in sched.nodes[2].get_nodes()])

    def test_each_loops_hoisted_op_lands_before_its_own_loop(self):
        # The prototype's one module-level list put every hoisted op before
        # the FIRST loop.
        e0 = self._extern("e0", reads="x")
        t0 = self._buf("t0", reads="x")
        mid = self._buf("mid", reads="t0")
        e1 = self._extern("e1", reads="mid")
        t1 = self._buf("t1", reads="mid")
        sched = self._scheduler()
        ops = [self._loop("loopnode0", [e0, t0]), mid]
        ops.append(self._loop("loopnode1", [e1, t1]))
        sched.nodes = [sched.create_scheduler_node(n) for n in ops]
        sched.update_zero_dim_cpu_tensor()
        self.assertEqual(
            [n.get_name() for n in sched.nodes],
            ["e0", "loopnode0", "mid", "e1", "loopnode1"],
        )

    def test_additional_deps_of_a_child_reach_the_loop(self):
        # compute_dependencies reads V.graph.additional_*_deps by the name of
        # a node in Scheduler.nodes; a child's entry must not be dropped.
        side = self._buf("side", reads="x")
        t = self._buf("t", reads="x")
        self.outputs = ["t", "side"]
        V.graph.additional_star_deps["t"].add("side")
        loop = self._loop("loopnode0", [t])
        nodes = self._front_half(self._scheduler(), [loop, side])
        self.assertEqual([n.get_name() for n in nodes], ["side", "loopnode0"])
        self.assertIn("side", set(nodes[1].ancestors))

    def test_loop_reading_a_mutated_input_runs_after_the_mutator(self):
        # ``mut`` mutates ``a``; the loop reads ``a`` afterwards.  Through
        # update_mutated_names the loop must depend on ``mut``.
        a = self._buf("a", reads="x")
        mut = self._buf("mut", reads="x", mutates="a")
        t = self._buf("t", reads="a")
        self.outputs = ["t", "a"]
        loop = self._loop("loopnode0", [t])
        with mock.patch.object(
            SchedulerNode,
            "has_side_effects",
            lambda self: isinstance(self.node.layout, MutationLayoutSHOULDREMOVE),
        ):
            nodes = self._front_half(self._scheduler(), [a, mut, loop])
        self.assertEqual([n.get_name() for n in nodes], ["a", "mut", "loopnode0"])
        self.assertIn("mut", set(nodes[2].ancestors))

    def test_extern_in_nested_body_lands_before_the_outermost_loop(self):
        ext = self._extern("ext", reads="x")
        a = self._buf("a", reads="x")
        b = self._buf("b", reads="a")
        self.outputs = ["b", "ext"]
        outer = self._loop("loopnode0", [a, self._loop("loopnode1", [ext, b])])
        nodes = self._front_half(self._scheduler(), [outer])
        self.assertEqual([n.get_name() for n in nodes], ["ext", "loopnode0"])

    def test_hoisted_state_is_per_scheduler(self):
        ext = self._extern("ext", reads="x")
        t = self._buf("t", reads="x")
        loop = self._loop("loopnode0", [ext, t])
        first = self._scheduler()
        first.nodes = [first.create_scheduler_node(loop)]
        # A second scheduler, built while the first still holds its hoisted
        # node, sees none of it.
        u = self._buf("u", reads="x")
        second = self._scheduler()
        second.nodes = [second.create_scheduler_node(u)]
        second.update_zero_dim_cpu_tensor()
        self.assertEqual([n.get_name() for n in second.nodes], ["u"])
        first.update_zero_dim_cpu_tensor()
        self.assertEqual([n.get_name() for n in first.nodes], ["ext", "loopnode0"])
        # Flushed: a second call splices nothing again.
        first.update_zero_dim_cpu_tensor()
        self.assertEqual([n.get_name() for n in first.nodes], ["ext", "loopnode0"])

    def test_no_loop_node_takes_the_original_path(self):
        a = self._buf("a", reads="x")
        b = self._buf("b", reads="a")
        self.outputs = ["b"]
        nodes = self._front_half(self._scheduler(), [a, b])
        self.assertEqual([n.get_name() for n in nodes], ["a", "b"])
        self.assertTrue(all(type(n) is SchedulerNode for n in nodes))

    def test_hoisted_op_reading_a_loop_buffer_is_refused(self):
        # Hoisting it before the loop would put it before its producer.
        t = self._buf("t", reads="x")
        ext = self._extern("ext", reads="t")
        loop = self._loop("loopnode0", [t, ext])
        with self.assertRaisesRegex(Unsupported, "ext"):
            self._scheduler().create_scheduler_node(loop)

    def test_body_with_no_kernel_is_refused(self):
        ext = self._extern("ext", reads="x")
        loop = self._loop("loopnode0", [ext])
        with self.assertRaises(Unsupported):
            self._scheduler().create_scheduler_node(loop)


if __name__ == "__main__":
    pytest.main([__file__])
