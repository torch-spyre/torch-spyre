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


"""Unit tests for the scan helpers in torch_spyre._inductor.

``find_scan_var``, ``scan_axis``, ``scan_space_position`` and
``scan_ordered_space`` are pure functions of the IR node and index expressions,
so they are exercised here without hardware. The ``non_zero_pos_to_idx_*`` tests
check the registered CPU reference kernel, not the Spyre path.
"""

from types import SimpleNamespace

import pytest
import torch
import sympy

import torch_spyre  # noqa: F401  (registers spyre::non_zero_pos_to_idx)
from torch_spyre._inductor.errors import Unsupported
from torch._inductor.ir import Scan
from torch_spyre._inductor.pass_utils import (
    scan_axis,
    scan_ordered_space,
    find_scan_var,
    scan_space_position,
)


def _scan_op(pointwise, scan, axis=0):
    """A bare ``Scan`` over ``pointwise`` + ``scan`` ranges, scanned at ``axis``."""
    data = Scan.__new__(Scan)
    object.__setattr__(data, "ranges", list(pointwise))
    object.__setattr__(data, "scan_ranges", list(scan))
    object.__setattr__(data, "size", list(pointwise) + list(scan))

    def reindex(index, scan_index):
        # What ``Scan.create`` builds.
        return [*index[:axis], *scan_index, *index[axis:]]

    object.__setattr__(data, "reindex", reindex)
    return SimpleNamespace(data=data, get_name=lambda: "buf0")


def _rw(ranges):
    return SimpleNamespace(writes=[SimpleNamespace(ranges=ranges)])


d0, d1, d2 = sympy.symbols("d0 d1 d2")


def test_find_scan_var_is_the_last_write_loop():
    rw = _rw({d0: 4, d1: 128, d2: 64})
    assert find_scan_var(_scan_op([4, 128], [64], axis=0), rw) == d2


def test_find_scan_var_skips_unit_pointwise_dims():
    # (1, 4, 64) scanned over dim 2: the unit dim has no loop.
    rw = _rw({d0: 4, d1: 64})
    assert find_scan_var(_scan_op([1, 4], [64], axis=2), rw) == d1


def test_find_scan_var_ignores_other_loops_with_the_same_extent():
    rw = _rw({d0: 64, d1: 64})
    assert find_scan_var(_scan_op([64], [64]), rw) == d1


def test_find_scan_var_rejects_merged_loops():
    # Pointwise and scan merged into one loop of 128.
    with pytest.raises(Unsupported, match="not 1 pointwise"):
        find_scan_var(_scan_op([8], [16]), _rw({d0: 128}))


def test_find_scan_var_rejects_wrong_scan_extent():
    with pytest.raises(Unsupported, match="extent 16"):
        find_scan_var(_scan_op([8], [16]), _rw({d0: 8, d1: 32}))


def test_find_scan_var_rejects_multiple_scan_dims():
    with pytest.raises(Unsupported, match="2 dims"):
        find_scan_var(_scan_op([4], [8, 8]), _rw({d0: 4}))


def test_find_scan_var_rejects_non_scan():
    with pytest.raises(Unsupported, match="not a scan"):
        find_scan_var(SimpleNamespace(data=object(), get_name=lambda: "b"))


@pytest.mark.parametrize(
    "pointwise, scan, axis",
    [
        ([4, 128], [64], 0),
        ([4, 128], [64], 1),
        ([4, 128], [64], 2),
        ([128], [64], 0),
        ([1, 128], [64], 1),  # unit dim before the scan dim
        ([4, 1], [64], 2),  # unit dim after the scan dim
    ],
)
def test_scan_axis_and_position(pointwise, scan, axis):
    op = _scan_op(pointwise, scan, axis)
    assert scan_axis(op) == axis
    # Unit dims before the scan dim take no slot in the iteration space.
    units_before = sum(1 for n in ([*pointwise[:axis]]) if n == 1)
    assert scan_space_position(op) == axis - units_before


def test_scan_axis_rejects_unrecognized_reindex():
    op = _scan_op([4], [64])
    object.__setattr__(op.data, "reindex", lambda pw, sc: [*pw, sc[0] + 1])
    with pytest.raises(Unsupported, match="does not place"):
        scan_axis(op)


def test_scan_axis_rejects_multiple_scan_dims():
    with pytest.raises(Unsupported, match="2 scan dims"):
        scan_axis(_scan_op([4], [8, 8]))


def test_scan_ordered_space_moves_scan_dim_to_its_axis():
    d0, d1, d2 = sympy.symbols("d0 d1 d2")
    # Write dep lists pointwise vars first, the scan var (d2) last.
    space = {d0: 4, d1: 128, d2: 64}
    assert list(scan_ordered_space(space, d2, 0)) == [d2, d0, d1]
    assert list(scan_ordered_space(space, d2, 1)) == [d0, d2, d1]
    assert list(scan_ordered_space(space, d2, 2)) == [d0, d1, d2]


def test_scan_ordered_space_ignores_strides_of_unit_dims():
    d0, d1 = sympy.symbols("d0 d1")
    space = {d0: 1, d1: 64}
    assert list(scan_ordered_space(space, d1, 0)) == [d1, d0]


def test_scan_ordered_space_rejects_bad_input():
    d0, d1 = sympy.symbols("d0 d1")
    with pytest.raises(Unsupported, match="not in"):
        scan_ordered_space({d0: 4}, d1, 0)
    with pytest.raises(Unsupported, match="out of range"):
        scan_ordered_space({d0: 4, d1: 4}, d1, 2)


class _RecordLoads:
    """Ops handler that records the buffer name of every load."""

    def __init__(self):
        self.loads = []

    def load(self, name, index):
        self.loads.append(name)


def test_graph_editor_swaps_names_in_every_scan_inner_fn():
    from torch._inductor.virtualized import V

    from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor

    op = _scan_op([4], [64])
    scan = op.data

    def load_a(index):
        V.ops.load("a", index)

    object.__setattr__(scan, "inner_fn", load_a)
    object.__setattr__(scan, "inner_fns", (load_a, load_a))
    # The rebuild copies every dataclass field, so the bare Scan needs them all.
    for name in Scan.__dataclass_fields__:
        if not hasattr(scan, name):
            object.__setattr__(scan, name, None)
    object.__setattr__(scan, "origins", set())
    object.__setattr__(scan, "origin_node", None)
    object.__setattr__(scan, "traceback", None)
    editor = GraphEditor.__new__(GraphEditor)

    new = editor._create_loop_hack_inner_fn(scan, {"a": "b"})
    assert isinstance(new, Scan)
    for fn in (new.inner_fn, *new.inner_fns):
        rec = _RecordLoads()
        with V.set_ops_handler(rec):
            fn([0, 0])
        assert rec.loads == ["b"]
    assert GraphEditor.is_rewritable_consumer(SimpleNamespace(data=scan))


def _ref(x, dim, padval):
    return torch.ops.spyre.non_zero_pos_to_idx(x, dim, padval)


def test_non_zero_pos_to_idx_spec_example():
    x = torch.tensor([0.0, 5.0, 0.0, 3.0])
    out = _ref(x, 0, -1.0)
    assert out.tolist() == [-1.0, 0.0, -1.0, 1.0]


def test_non_zero_pos_to_idx_keeps_shape_and_dtype():
    x = torch.randn(3, 4, 5).half()
    out = _ref(x, 1, 0.0)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_non_zero_pos_to_idx_all_zero_row_gets_padval():
    x = torch.zeros(2, 6)
    out = _ref(x, 1, 7.0)
    assert torch.equal(out, torch.full((2, 6), 7.0))


def test_non_zero_pos_to_idx_counter_is_per_row_along_dim():
    x = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]])
    out = _ref(x, 1, -1.0)
    assert out.tolist() == [[0.0, -1.0, 1.0], [-1.0, 0.0, 1.0]]


def test_non_zero_pos_to_idx_negative_dim_matches_positive():
    x = torch.randint(0, 2, (3, 4, 5)).float()
    assert torch.equal(_ref(x, -1, -1.0), _ref(x, 2, -1.0))
    assert torch.equal(_ref(x, -2, -1.0), _ref(x, 1, -1.0))
