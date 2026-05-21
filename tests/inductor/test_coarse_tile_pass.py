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

"""Unit tests for coarse_tile IR pass.

All tests use lightweight mocks so they run without a Spyre device or a full
compilation pipeline.
"""

import unittest
from unittest.mock import MagicMock
from sympy import Integer, Symbol, simplify

from torch._inductor.ir import Pointwise, Reduction

from torch_spyre._inductor.coarse_tile import coarse_tile, _divide_ranges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pointwise(ranges):
    """Return a fake Pointwise with the given ranges."""
    pw = MagicMock(spec=Pointwise)
    pw.ranges = list(ranges)
    return pw


def _make_reduction(ranges, reduction_ranges):
    """Return a fake Reduction with the given ranges and reduction_ranges."""
    red = MagicMock(spec=Reduction)
    red.ranges = list(ranges)
    red.reduction_ranges = list(reduction_ranges)
    return red


def _make_op(data, name="op0"):
    """Return a fake ComputedBuffer wrapping data."""
    from torch._inductor.ir import ComputedBuffer

    op = MagicMock(spec=ComputedBuffer)
    op.data = data
    op.get_operation_name.return_value = name
    # Ensure loop_group_id / loop_count are not pre-set
    del op.loop_group_id
    del op.loop_count
    return op


def _make_non_computed_op(name="extern0"):
    """Return a fake non-ComputedBuffer operation."""
    from torch._inductor.ir import Operation

    op = MagicMock(spec=Operation)
    op.get_operation_name.return_value = name
    return op


# ---------------------------------------------------------------------------
# Tests for _divide_ranges
# ---------------------------------------------------------------------------


class TestDivideRanges(unittest.TestCase):
    def test_pointwise_single_dim_divided(self):
        data = _make_pointwise([Integer(64)])
        op = _make_op(data)
        _divide_ranges(op, Integer(4), tiled_dims=None)
        self.assertEqual(data.ranges[0], Integer(16))

    def test_pointwise_symbolic_count(self):
        k = Symbol("K", positive=True)
        n = Symbol("N", positive=True)
        data = _make_pointwise([n])
        op = _make_op(data)
        _divide_ranges(op, k, tiled_dims=None)
        # n / K should simplify to n/K
        self.assertEqual(simplify(data.ranges[0] - n / k), 0)

    def test_pointwise_multidim_default_tiles_outermost_only(self):
        data = _make_pointwise([Integer(32), Integer(8)])
        op = _make_op(data)
        _divide_ranges(op, Integer(4), tiled_dims=None)
        # Only dim 0 is divided by 4; dim 1 is unchanged
        self.assertEqual(data.ranges[0], Integer(8))
        self.assertEqual(data.ranges[1], Integer(8))

    def test_tiled_dims_2(self):
        data = _make_pointwise([Integer(32), Integer(16), Integer(4)])
        op = _make_op(data)
        _divide_ranges(op, Integer(4), tiled_dims=2)
        self.assertEqual(data.ranges[0], Integer(8))
        self.assertEqual(data.ranges[1], Integer(4))
        self.assertEqual(data.ranges[2], Integer(4))  # untouched

    def test_tiled_dims_0_no_change(self):
        data = _make_pointwise([Integer(32)])
        op = _make_op(data)
        original = list(data.ranges)
        _divide_ranges(op, Integer(4), tiled_dims=0)
        self.assertEqual(data.ranges, original)

    def test_empty_ranges_no_change(self):
        data = _make_pointwise([])
        op = _make_op(data)
        _divide_ranges(op, Integer(4), tiled_dims=None)
        self.assertEqual(data.ranges, [])

    def test_reduction_outer_dims_divided_inner_untouched(self):
        data = _make_reduction([Integer(64)], [Integer(128)])
        op = _make_op(data)
        _divide_ranges(op, Integer(4), tiled_dims=None)
        # Outer range is divided
        self.assertEqual(data.ranges[0], Integer(16))
        # Reduction range is left alone
        self.assertEqual(data.reduction_ranges[0], Integer(128))

    def test_non_loops_type_skipped(self):
        """_divide_ranges should silently skip ops whose data is not Loops."""
        from torch._inductor.ir import Operation

        op = _make_op(MagicMock(spec=Operation))
        # Should not raise
        _divide_ranges(op, Integer(4), tiled_dims=None)


# ---------------------------------------------------------------------------
# Tests for coarse_tile
# ---------------------------------------------------------------------------


class TestCoarseTile(unittest.TestCase):
    def _run(self, all_ops, groups, **kwargs):
        coarse_tile(all_ops, groups, **kwargs)

    def test_single_group_stamps_attributes(self):
        data = _make_pointwise([Integer(64)])
        op = _make_op(data, "op0")
        self._run([op], [([op], Integer(4))])
        self.assertEqual(op.loop_group_id, (0,))
        self.assertEqual(op.loop_count, Integer(4))
        self.assertEqual(data.ranges[0], Integer(16))

    def test_two_groups_get_distinct_ids(self):
        d0 = _make_pointwise([Integer(32)])
        d1 = _make_pointwise([Integer(64)])
        op0 = _make_op(d0, "op0")
        op1 = _make_op(d1, "op1")
        self._run([op0, op1], [([op0], Integer(4)), ([op1], Integer(8))])
        self.assertEqual(op0.loop_group_id, (0,))
        self.assertEqual(op1.loop_group_id, (1,))
        self.assertEqual(op0.loop_count, Integer(4))
        self.assertEqual(op1.loop_count, Integer(8))
        self.assertEqual(d0.ranges[0], Integer(8))
        self.assertEqual(d1.ranges[0], Integer(8))

    def test_empty_groups_list_is_noop(self):
        data = _make_pointwise([Integer(32)])
        op = _make_op(data, "op0")
        original = list(data.ranges)
        self._run([op], [])
        self.assertFalse(
            hasattr(op, "loop_group_id") and op.loop_group_id != MagicMock()
        )
        self.assertEqual(data.ranges, original)

    def test_non_computed_buffer_skipped(self):
        """Non-ComputedBuffer ops in a group are skipped without error."""

        op_extern = _make_non_computed_op("extern0")
        data = _make_pointwise([Integer(16)])
        op_computed = _make_op(data, "op0")
        self._run([op_extern, op_computed], [([op_extern, op_computed], Integer(2))])
        # computed op gets stamped
        self.assertEqual(op_computed.loop_group_id, (0,))
        # extern op does not get stamped (it was skipped)
        # MagicMock will auto-create attributes so we check it wasn't set
        # by verifying the computed op range was divided
        self.assertEqual(data.ranges[0], Integer(8))

    def test_symbolic_count(self):
        k = Symbol("K", positive=True)
        n = Symbol("N", positive=True)
        data = _make_pointwise([n])
        op = _make_op(data, "op0")
        self._run([op], [([op], k)])
        self.assertEqual(op.loop_count, k)
        self.assertEqual(simplify(data.ranges[0] - n / k), 0)

    def test_non_contiguous_group_raises(self):
        """A group whose ops are not contiguous in operations should raise."""
        d0 = _make_pointwise([Integer(32)])
        d1 = _make_pointwise([Integer(32)])
        d2 = _make_pointwise([Integer(32)])
        op0 = _make_op(d0, "op0")
        op1 = _make_op(d1, "op1")
        op2 = _make_op(d2, "op2")
        # group contains op0 and op2 but not op1 (non-contiguous)
        with self.assertRaises(RuntimeError):
            self._run([op0, op1, op2], [([op0, op2], Integer(4))])

    def test_op_not_in_operations_raises(self):
        """An op in a group that is absent from the operations list should raise."""
        data = _make_pointwise([Integer(32)])
        op_known = _make_op(data, "op0")
        op_unknown = _make_op(_make_pointwise([Integer(8)]), "unknown")
        with self.assertRaises(RuntimeError):
            self._run([op_known], [([op_unknown], Integer(2))])

    def test_multiple_ops_in_single_group(self):
        d0 = _make_pointwise([Integer(32)])
        d1 = _make_pointwise([Integer(64)])
        op0 = _make_op(d0, "op0")
        op1 = _make_op(d1, "op1")
        self._run([op0, op1], [([op0, op1], Integer(4))])
        # Both ops share loop_group_id (0,)
        self.assertEqual(op0.loop_group_id, (0,))
        self.assertEqual(op1.loop_group_id, (0,))
        self.assertEqual(d0.ranges[0], Integer(8))
        self.assertEqual(d1.ranges[0], Integer(16))

    def test_per_group_tiled_dims_override(self):
        """Each group may carry its own tiled_dims as a third tuple element."""
        # op0: 2D [32, 16] — tile dim 0 (tiled_dims=1, default)
        # op1: 2D [8, 64]  — tile dim 1 (tiled_dims override via third element)
        d0 = _make_pointwise([Integer(32), Integer(16)])
        d1 = _make_pointwise([Integer(8), Integer(64)])
        op0 = _make_op(d0, "op0")
        op1 = _make_op(d1, "op1")
        self._run(
            [op0, op1],
            [
                ([op0], Integer(4)),  # uses default tiled_dims=None → dim 0
                ([op1], Integer(4), 2),  # per-group override: tile first 2 dims
            ],
        )
        # op0: dim 0 divided, dim 1 unchanged
        self.assertEqual(d0.ranges[0], Integer(8))
        self.assertEqual(d0.ranges[1], Integer(16))
        # op1: both dims divided
        self.assertEqual(d1.ranges[0], Integer(2))
        self.assertEqual(d1.ranges[1], Integer(16))

    def test_per_group_tiled_dims_none_overrides_kwarg(self):
        """Per-group tiled_dims=None overrides a non-None kwarg default."""
        d0 = _make_pointwise([Integer(32), Integer(16)])
        op0 = _make_op(d0, "op0")
        # kwarg default would tile 2 dims, but group says None → tile only dim 0
        self._run(
            [op0],
            [([op0], Integer(4), None)],
            tiled_dims=2,
        )
        self.assertEqual(d0.ranges[0], Integer(8))
        self.assertEqual(d0.ranges[1], Integer(16))  # untouched


if __name__ == "__main__":
    unittest.main()
