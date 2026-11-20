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

"""Unit tests for the bundle-grouping policy shared by the fusion pass and the
pre-scheduling bundle estimator.

:func:`group_contiguous_fusable` is exercised directly: it is generic over the
item type, so the grouping rule can be checked with plain integers, without a
scheduler, a graph or a device.

:func:`estimate_bundles` is exercised with bare ``ComputedBuffer`` instances
(allocated via ``__new__``, carrying only the attributes the grouping consults),
since building real ones needs a compile.
"""

import unittest
from unittest import mock
import torch
from torch._inductor.ir import ComputedBuffer, NopKernel
from torch._inductor.scheduler import NopKernelSchedulerNode, SchedulerNode
from torch_spyre._inductor import config
from torch_spyre._inductor.constants import DEVICE_NAME
from torch_spyre._inductor.fusion import (
    _is_fusable_node,
    _is_hoistable_node,
    estimate_bundles,
    group_contiguous_fusable,
)


def _is_even(x: int) -> bool:
    """Stand-in predicate: fusable items are even, boundaries are odd.

    Any predicate would do; this one keeps the expected groupings easy to read.
    """
    return x % 2 == 0


class TestGroupContiguousFusable(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(group_contiguous_fusable([], _is_even), [])

    def test_all_fusable_is_one_run(self):
        self.assertEqual(
            group_contiguous_fusable([0, 2, 4, 6], _is_even), [[0, 2, 4, 6]]
        )

    def test_single_fusable(self):
        self.assertEqual(group_contiguous_fusable([2], _is_even), [[2]])

    def test_all_boundary_each_its_own_group(self):
        self.assertEqual(group_contiguous_fusable([1, 3, 5], _is_even), [[1], [3], [5]])

    def test_lone_boundary_comes_back_as_single_element_group(self):
        # This is what makes the refactor behaviour-preserving in
        # ``spyre_fuse_nodes``: a boundary node is a length-1 run, which
        # ``_make_fused`` returns unchanged rather than wrapping.
        self.assertEqual(group_contiguous_fusable([1], _is_even), [[1]])

    def test_alternating(self):
        self.assertEqual(
            group_contiguous_fusable([0, 1, 2, 3, 4], _is_even),
            [[0], [1], [2], [3], [4]],
        )

    def test_boundary_at_start(self):
        self.assertEqual(
            group_contiguous_fusable([1, 0, 2, 4], _is_even), [[1], [0, 2, 4]]
        )

    def test_boundary_at_end(self):
        self.assertEqual(
            group_contiguous_fusable([0, 2, 4, 1], _is_even), [[0, 2, 4], [1]]
        )

    def test_boundary_at_both_ends(self):
        self.assertEqual(
            group_contiguous_fusable([1, 0, 2, 3], _is_even),
            [[1], [0, 2], [3]],
        )

    def test_runs_are_maximal_and_order_is_preserved(self):
        items = [0, 2, 1, 4, 6, 8, 3, 5, 10]
        self.assertEqual(
            group_contiguous_fusable(items, _is_even),
            [[0, 2], [1], [4, 6, 8], [3], [5], [10]],
        )
        # Every item appears exactly once, in the original order.
        groups = group_contiguous_fusable(items, _is_even)
        self.assertEqual([item for group in groups for item in group], items)


def _buffer(tag: str, device: str, zero_elements: bool = False) -> ComputedBuffer:
    """A bare ``ComputedBuffer`` carrying only what the grouping consults.

    The grouping reads nothing but the type, the device and ``is_no_op()``, so the
    buffer needs no layout or loop body -- just to satisfy the ``isinstance``
    check and answer those two calls. ``zero_elements`` stands in for a buffer
    with no elements, which the scheduler turns into a
    ``NopKernelSchedulerNode``. ``tag`` identifies it in assertions.
    """
    op = ComputedBuffer.__new__(ComputedBuffer)
    op.tag = tag
    op.get_device = lambda: torch.device(device)
    op.is_no_op = lambda: zero_elements
    return op


def _spyre(tag: str) -> ComputedBuffer:
    return _buffer(tag, DEVICE_NAME)


def _cpu(tag: str) -> ComputedBuffer:
    """A ComputedBuffer off-device: a bundle boundary that is still modellable."""
    return _buffer(tag, "cpu")


class _Extern:
    """Stands in for an extern/fallback op: not a ComputedBuffer at all."""

    def __init__(self, tag: str):
        self.tag = tag

    def is_no_op(self) -> bool:
        return False


def _nop_kernel(tag: str) -> NopKernel:
    """A bare ``NopKernel`` (e.g. the ``ConcatKernel`` behind ``cat``)."""
    op = NopKernel.__new__(NopKernel)
    op.tag = tag
    return op


def _tags(bundles) -> list[list[str]]:
    """The grouping as tags, so an assertion reads as the bundle shape."""
    return [[op.tag for op in bundle] for bundle in bundles]


class TestEstimateBundles(unittest.TestCase):
    """The pre-scheduling estimate the co-optimizer's cost objective groups by."""

    def test_spyre_run_is_one_bundle(self):
        ops = [_spyre("a"), _spyre("b"), _spyre("c")]
        self.assertEqual(_tags(estimate_bundles(ops)), [["a", "b", "c"]])

    def test_extern_op_forces_a_boundary(self):
        ops = [_spyre("a"), _spyre("b"), _Extern("x"), _spyre("c")]
        self.assertEqual(_tags(estimate_bundles(ops)), [["a", "b"], ["x"], ["c"]])

    def test_off_device_buffer_forces_a_boundary(self):
        # A ComputedBuffer is not enough; it has to be on the Spyre device.
        ops = [_spyre("a"), _cpu("h"), _spyre("b")]
        self.assertEqual(_tags(estimate_bundles(ops)), [["a"], ["h"], ["b"]])

    def test_no_operations(self):
        self.assertEqual(estimate_bundles([]), [])

    def test_nop_kernel_is_hoisted_ahead_of_the_run(self):
        # ``spyre_fuse_nodes`` hoists the NopKernelSchedulerNode this becomes,
        # so it must not split the estimate into [a] and [b].
        ops = [_spyre("a"), _nop_kernel("cat"), _spyre("b")]
        self.assertEqual(_tags(estimate_bundles(ops)), [["cat"], ["a", "b"]])

    def test_zero_element_buffer_is_hoisted_not_fused(self):
        # A Spyre ComputedBuffer with no elements is a nop, not a run member.
        ops = [_spyre("a"), _buffer("pad", DEVICE_NAME, True), _spyre("b")]
        self.assertEqual(_tags(estimate_bundles(ops)), [["pad"], ["a", "b"]])

    def test_unvetted_nop_is_a_boundary(self):
        ops = [_spyre("a"), _UnvettedNop("new"), _spyre("b")]
        self.assertEqual(_tags(estimate_bundles(ops)), [["a"], ["new"], ["b"]])

    def test_fusion_off_gives_one_bundle_per_op(self):
        # The real pass leaves every node alone when bundling is off, so the
        # estimate must not claim a fusion that will not happen.
        ops = [_spyre("a"), _spyre("b"), _spyre("c")]
        with mock.patch.object(config, "bundle_symbolic_args", False):
            self.assertEqual(_tags(estimate_bundles(ops)), [["a"], ["b"], ["c"]])


def _scheduler_node(tag: str) -> SchedulerNode:
    """A bare Spyre ``SchedulerNode``: only the type and device are consulted."""
    node = SchedulerNode.__new__(SchedulerNode)
    node.tag = tag
    node.get_device = lambda: torch.device(DEVICE_NAME)
    return node


class _UnvettedNop:
    """Stands in for an IR kind whose ``is_no_op()`` is true but which is neither
    a ``ComputedBuffer`` nor a ``NopKernel`` -- one a future PyTorch might add."""

    def __init__(self, tag: str):
        self.tag = tag

    def is_no_op(self) -> bool:
        return True


def _nop_node(tag: str, op=None) -> NopKernelSchedulerNode:
    """A bare ``NopKernelSchedulerNode`` wrapping ``op``, by default an empty
    Spyre buffer (what ``lowering.empty`` produces): only the type and the
    wrapped operation are consulted."""
    node = NopKernelSchedulerNode.__new__(NopKernelSchedulerNode)
    node.tag = tag
    node.node = op if op is not None else _buffer(tag, DEVICE_NAME, True)
    return node


class TestHoistNopNodes(unittest.TestCase):
    """The scheduler-node predicates that ``spyre_fuse_nodes`` groups by."""

    def _group(self, nodes):
        return _tags(
            group_contiguous_fusable(nodes, _is_fusable_node, _is_hoistable_node)
        )

    def test_nop_is_moved_ahead_of_the_run(self):
        # Without hoisting, the nop would split the run into [a, b] and [c, d].
        nodes = [
            _scheduler_node("a"),
            _scheduler_node("b"),
            _nop_node("nop"),
            _scheduler_node("c"),
            _scheduler_node("d"),
        ]
        self.assertEqual(self._group(nodes), [["nop"], ["a", "b", "c", "d"]])

    def test_nop_at_end_of_run_is_hoisted(self):
        # Intentional: a nop only allocates, so emitting it ahead of the run it
        # follows is safe even with no fusable node after it.
        nodes = [_scheduler_node("a"), _nop_node("nop")]
        self.assertEqual(self._group(nodes), [["nop"], ["a"]])

    def test_nop_with_no_pending_run_stays_in_place(self):
        nodes = [_nop_node("nop"), _scheduler_node("a"), _scheduler_node("b")]
        self.assertEqual(self._group(nodes), [["nop"], ["a", "b"]])

    def test_nop_kernel_is_hoisted(self):
        nodes = [_scheduler_node("a"), _nop_node("cat", _nop_kernel("cat"))]
        self.assertEqual(self._group(nodes), [["cat"], ["a"]])

    def test_unvetted_nop_is_a_boundary(self):
        # Only nops known to codegen as a bare allocation are reordered.
        nodes = [
            _scheduler_node("a"),
            _nop_node("new", _UnvettedNop("new")),
            _scheduler_node("b"),
        ]
        self.assertEqual(self._group(nodes), [["a"], ["new"], ["b"]])


if __name__ == "__main__":
    unittest.main()
