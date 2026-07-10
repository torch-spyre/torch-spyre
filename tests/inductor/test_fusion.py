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

from unittest.mock import patch

from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.test_case import run_tests

from torch_spyre._inductor import config
from torch_spyre._inductor import fusion
from torch_spyre._inductor.fusion import (
    _can_extend_bundle,
    _op_count,
    spyre_fuse_nodes,
)


class _FakeNode:
    """Duck-typed stand-in for a scheduler node used by the fusion predicate."""

    def __init__(self, reduction: bool = False, count: int = 1) -> None:
        self._reduction = reduction
        self._count = count

    def is_reduction(self) -> bool:
        return self._reduction

    def get_nodes(self) -> list:
        return [self] * self._count


class TestCanExtendBundle(InductorTestCase):
    def test_op_count_counts_leaf_nodes(self):
        self.assertEqual(_op_count(_FakeNode(count=3)), 3)

    def test_reduction_candidate_never_extends(self):
        with patch.object(config, "max_fused_ops", 100):
            self.assertFalse(
                _can_extend_bundle([_FakeNode()], _FakeNode(reduction=True))
            )

    def test_bundle_with_reduction_rejects_more(self):
        with patch.object(config, "max_fused_ops", 100):
            self.assertFalse(
                _can_extend_bundle([_FakeNode(reduction=True)], _FakeNode())
            )

    def test_within_cap_allows(self):
        with patch.object(config, "max_fused_ops", 4):
            self.assertTrue(_can_extend_bundle([_FakeNode(), _FakeNode()], _FakeNode()))

    def test_exceeds_cap_rejects(self):
        with patch.object(config, "max_fused_ops", 2):
            self.assertFalse(
                _can_extend_bundle([_FakeNode(), _FakeNode()], _FakeNode())
            )

    def test_cap_counts_leaf_ops_of_loop_group(self):
        with patch.object(config, "max_fused_ops", 3):
            self.assertFalse(
                _can_extend_bundle([_FakeNode(count=2)], _FakeNode(count=2))
            )

    def test_at_cap_boundary_allows(self):
        # total == cap must be allowed (spec: reject only when total > cap).
        with patch.object(config, "max_fused_ops", 3):
            self.assertTrue(_can_extend_bundle([_FakeNode(), _FakeNode()], _FakeNode()))

    def test_reduction_detected_via_leaf_not_container(self):
        # A loop-group whose container reports non-reduction but wraps a
        # reduction leaf must still be treated as a reduction. Guards against
        # trusting the container's own is_reduction().
        class _Group:
            def is_reduction(self) -> bool:
                return False  # container does not reflect its members

            def get_nodes(self) -> list:
                return [_FakeNode(), _FakeNode(reduction=True)]

        with patch.object(config, "max_fused_ops", 100):
            self.assertFalse(_can_extend_bundle([_FakeNode()], _Group()))
            self.assertFalse(_can_extend_bundle([_Group()], _FakeNode()))


class _FakeLeaf:
    """Fusible-node fake; registered via _FUSIBLE_NODE_TYPES in tests."""

    def __init__(self, name: str, reduction: bool = False, count: int = 1) -> None:
        self._name = name
        self._reduction = reduction
        self._count = count

    def is_reduction(self) -> bool:
        return self._reduction

    def get_nodes(self) -> list:
        return [self] * self._count

    def get_name(self) -> str:
        return self._name


def _grouped_names(out) -> list:
    # _make_fused is patched to return the node list, so each group is a list.
    return [[n.get_name() for n in g] for g in out]


class TestSpyreFuseNodes(InductorTestCase):
    def _ctx(self, cap):
        # Patch the fusible-type gate to accept _FakeLeaf and make _make_fused
        # return the raw list so we can assert boundaries without building a
        # real FusedSchedulerNode.
        return [
            patch.object(config, "bundle_symbolic_args", True),
            patch.object(config, "max_fused_ops", cap),
            patch.object(fusion, "_FUSIBLE_NODE_TYPES", (_FakeLeaf,)),
            patch.object(fusion, "_make_fused", lambda ns: list(ns)),
        ]

    def _run(self, nodes, cap):
        ctxs = self._ctx(cap)
        for c in ctxs:
            c.start()
        try:
            return spyre_fuse_nodes(nodes)
        finally:
            for c in reversed(ctxs):
                c.stop()

    def test_reduction_isolated_into_own_bundle(self):
        out = self._run(
            [_FakeLeaf("a"), _FakeLeaf("r", reduction=True), _FakeLeaf("b")], cap=100
        )
        self.assertEqual(_grouped_names(out), [["a"], ["r"], ["b"]])

    def test_cap_splits_long_run(self):
        out = self._run([_FakeLeaf(str(i)) for i in range(5)], cap=2)
        self.assertEqual([len(g) for g in out], [2, 2, 1])

    def test_op_exceeding_cap_survives_as_singleton(self):
        out = self._run([_FakeLeaf("big", count=5), _FakeLeaf("n")], cap=2)
        self.assertEqual(_grouped_names(out), [["big"], ["n"]])

    def test_no_fusion_when_symbolic_args_off(self):
        nodes = [_FakeLeaf("a"), _FakeLeaf("b")]
        with patch.object(config, "bundle_symbolic_args", False):
            self.assertIs(spyre_fuse_nodes(nodes), nodes)

    def test_non_fusible_node_forces_boundary(self):
        # A node not in _FUSIBLE_NODE_TYPES (e.g. a Fallback) breaks the run and
        # passes through as-is between the two surrounding bundles.
        class _Fallback:
            def get_name(self) -> str:
                return "f"

        fb = _Fallback()
        out = self._run([_FakeLeaf("a"), fb, _FakeLeaf("b")], cap=100)
        self.assertEqual(len(out), 3)
        self.assertEqual([n.get_name() for n in out[0]], ["a"])
        self.assertIs(out[1], fb)
        self.assertEqual([n.get_name() for n in out[2]], ["b"])


if __name__ == "__main__":
    run_tests()
