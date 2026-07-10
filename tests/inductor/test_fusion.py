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
from torch_spyre._inductor.fusion import _can_extend_bundle, _op_count


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


if __name__ == "__main__":
    run_tests()
