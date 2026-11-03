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

"""Tests for the read-only loop-fact accessors in ``loop_facts``."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch  # noqa: F401
import sympy
from torch._inductor.ir import ExternKernel
from torch._inductor.scheduler import SchedulerNode

from torch_spyre._inductor.cost_model_pass import _loop_group_id as cost_gid
from torch_spyre._inductor.loop_facts import (
    counted_loop_path,
    enclosing_loops,
    loop_depth,
    scheduler_enclosing_loops,
)
from torch_spyre._inductor.loop_info import CoarseTileInfo
from torch_spyre._inductor.scheduler import _loop_count, _loop_group_id
from torch_spyre._inductor.scratchpad.utils import counted_loop_group_path


def _info(gid, counts, tiled=None, tiled_red=None):
    n = len(counts) if hasattr(counts, "__len__") else 0
    return CoarseTileInfo(
        loop_group_id=gid,
        loop_count=counts,
        loop_tiled_dims=tiled if tiled is not None else [[] for _ in range(n)],
        loop_tiled_reduction_dims=(
            tiled_red if tiled_red is not None else [[] for _ in range(n)]
        ),
    )


def _op(info=None):
    return SimpleNamespace(loop_info=info) if info is not None else SimpleNamespace()


def _extern(info):
    op = MagicMock(spec=ExternKernel)
    op.loop_info = info
    return op


def _snode(info):
    sn = MagicMock(spec=SchedulerNode)
    sn.node = _op(info)
    return sn


def test_flat_op():
    op = _op(_info((0,), [4], [[1]], [[]]))
    (lvl,) = enclosing_loops(op)
    assert lvl.trip_count == 4
    assert lvl.group_path == (0,)
    assert lvl.tiled_dims == (1,)
    assert lvl.tiled_reduction_dims == ()
    assert lvl.induction is None
    assert loop_depth(op) == 1


def test_nested_op():
    op = _op(_info((0, 1), [2, 8], [[0], [1]], [[], [0]]))
    outer, inner = enclosing_loops(op)
    assert (outer.group_path, outer.trip_count) == ((0,), 2)
    assert (inner.group_path, inner.trip_count) == ((0, 1), 8)
    assert inner.tiled_reduction_dims == (0,)
    assert loop_depth(op) == 2


def test_no_loop():
    assert enclosing_loops(_op()) == ()
    assert loop_depth(_op()) == 0
    assert counted_loop_path(_op()) == ()


def test_malformed_group_id():
    op = _op(_info(3, [4]))
    assert enclosing_loops(op) == ()
    assert counted_loop_path(op) == ()


def test_extern_excluded_by_default():
    op = _extern(_info((0,), [4]))
    assert counted_loop_path(op) == ()
    assert counted_loop_path(op, include_extern=True) == (0,)


def test_scheduler_skips_nodes_without_loop_info():
    node = MagicMock()
    node.get_nodes.return_value = [
        _snode(None),
        _snode(_info((2,), [sympy.Integer(5)])),
    ]
    (lvl,) = scheduler_enclosing_loops(node)
    assert lvl.group_path == (2,)
    assert lvl.trip_count == 5


def test_scheduler_node_without_scheduler_nodes_yields_empty():
    node = MagicMock()
    node.get_nodes.return_value = [MagicMock()]
    assert scheduler_enclosing_loops(node) == ()


def test_default_tiled_reduction_dims_is_still_a_loop():
    # Like test_restickify_cost_model: loop_tiled_reduction_dims left empty.
    op = _op(CoarseTileInfo((0,), [32], [[]]))
    (lvl,) = enclosing_loops(op)
    assert lvl.trip_count == 32
    assert lvl.tiled_dims == () and lvl.tiled_reduction_dims == ()
    assert len(enclosing_loops(op)) == len(counted_loop_path(op)) == 1


def test_short_loop_count_is_malformed():
    assert enclosing_loops(_op(_info((0, 1), [4], [[0]], [[]]))) == ()


def test_agrees_with_existing_readers():
    ops = [
        _op(),
        _op(_info((0,), [4], [[1]], [[]])),
        _op(_info((0, 1), [2, 8], [[0], [1]], [[], [0]])),
        _op(_info((3, 0, 2), [2, 3, 4], [[0], [], [1]], [[], [], []])),
        _op(CoarseTileInfo((0,), [32], [[]])),
    ]
    for op in ops:
        path = counted_loop_group_path(op)
        assert counted_loop_path(op) == path
        levels = enclosing_loops(op)
        assert len(levels) == len(path)
        gid = cost_gid(op) or ()
        assert [lvl.group_path for lvl in levels] == [
            gid[: i + 1] for i in range(len(gid))
        ]
        if levels:
            info = op.loop_info
            assert [lvl.trip_count for lvl in levels] == list(info.loop_count)
            assert [lvl.tiled_dims for lvl in levels] == [
                tuple(d) for d in info.loop_tiled_dims
            ] + [()] * (len(levels) - len(info.loop_tiled_dims))


def _sched_node(info):
    node = MagicMock()
    node.get_nodes.return_value = [_snode(info)]
    node.get_name.return_value = "buf0"
    return node


def test_scheduler_readers_well_formed():
    node = _sched_node(_info((3, 0), [2, 8]))
    assert _loop_group_id(node) == (3, 0)
    assert _loop_count(node, 0) == 2
    assert _loop_count(node, 1) == 8
    assert _loop_group_id(_sched_node(None)) is None


def test_scheduler_readers_malformed_ids_read_as_no_loop():
    # A bare-int or empty id used to raise TypeError / return () from
    # _loop_group_id; it now reads as "no loop", like the other accessors.
    for gid in (3, ()):
        assert _loop_group_id(_sched_node(_info(gid, [4]))) is None


def test_scheduler_loop_count_missing_depth_is_loud():
    node = _sched_node(_info((0,), [4]))
    with pytest.raises(AssertionError, match="no loop_count for depth 1"):
        _loop_count(node, 1)
    # A loop_count shorter than the id reads as "no loop" for both readers.
    short = _sched_node(_info((0, 1), [4]))
    assert _loop_group_id(short) is None
    with pytest.raises(AssertionError, match="no loop_count for depth 0"):
        _loop_count(short, 0)
