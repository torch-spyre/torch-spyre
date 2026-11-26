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

import torch  # noqa: F401
import sympy
from torch._inductor.ir import ExternKernel

from torch_spyre._inductor.cost_model_pass import _loop_group_id as cost_gid
from torch_spyre._inductor.loop_facts import (
    counted_loop_path,
    enclosing_loops,
    loop_depth,
)
from torch_spyre._inductor.loop_node import LoopMember
from torch_spyre._inductor.scratchpad.utils import counted_loop_group_path


def _info(gid, counts, tiled=None, tiled_red=None):
    n = len(counts) if hasattr(counts, "__len__") else 0
    return LoopMember(
        loop_group_id=gid,
        loop_count=counts,
        loop_tiled_dims=tiled if tiled is not None else [[] for _ in range(n)],
        loop_tiled_reduction_dims=(
            tiled_red if tiled_red is not None else [[] for _ in range(n)]
        ),
    )


def _op(info=None):
    return SimpleNamespace(loop_member=info) if info is not None else SimpleNamespace()


def _extern(info):
    op = MagicMock(spec=ExternKernel)
    op.loop_member = info
    return op


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


def test_default_tiled_reduction_dims_is_still_a_loop():
    # Like test_restickify_cost_model: loop_tiled_reduction_dims left empty.
    op = _op(LoopMember((0,), [32], [[]]))
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
        _op(LoopMember((0,), [32], [[]])),
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
            info = op.loop_member
            assert [lvl.trip_count for lvl in levels] == list(info.loop_count)
            assert [lvl.tiled_dims for lvl in levels] == [
                tuple(d) for d in info.loop_tiled_dims
            ] + [()] * (len(levels) - len(info.loop_tiled_dims))


class TestBroadcastBatchVars:
    """``pass_utils.broadcast_batch_vars`` reads the tiled dims through the
    accessors.

    It is kept on purpose, and so is this test. ``find_matmul_generated_var``
    calls it when given ``op`` (``propagate_layouts`` and ``read_copy_elision``
    both pass it) as a second line of defence for when
    ``coarse_tile._check_matmul_broadcast_batch_tiling``, which rejects its input
    first, is loosened or bypassed. No compiled test reaches it (265 matmul,
    attention and loop tests never called it), so it is pinned here on stand-in
    deps: a ``SimpleNamespace`` index cannot tell a real ``MemoryDep`` from a stub.
    """

    @staticmethod
    def _op(loop_member):
        return SimpleNamespace(
            loop_member=loop_member,
            data=SimpleNamespace(ranges=[sympy.Integer(8), sympy.Integer(16)]),
        )

    @staticmethod
    def _info(tiled):
        return LoopMember(
            loop_group_id=(0,),
            loop_count=[2],
            loop_tiled_dims=[tiled],
            loop_tiled_reduction_dims=[[]],
        )

    @staticmethod
    def _deps(x_vars):
        d0, d1 = sympy.symbols("d0 d1")
        out = SimpleNamespace(var_names=[d0, d1], index=d0 * 16 + d1)
        x = SimpleNamespace(index=sum(x_vars, sympy.Integer(0)))
        return x, out, d0, d1

    def test_a_tiled_dim_x_is_broadcast_over_is_returned(self):
        from torch_spyre._inductor.pass_utils import broadcast_batch_vars

        d0, d1 = sympy.symbols("d0 d1")
        x, out, _, _ = self._deps([d1])  # x has no d0: broadcast over dim 0
        assert broadcast_batch_vars(self._op(self._info([0])), x, out) == {d0}

    def test_a_tiled_dim_x_also_carries_is_not_returned(self):
        from torch_spyre._inductor.pass_utils import broadcast_batch_vars

        d0, d1 = sympy.symbols("d0 d1")
        x, out, _, _ = self._deps([d0, d1])
        assert broadcast_batch_vars(self._op(self._info([0])), x, out) == set()

    def test_an_untiled_op_returns_nothing(self):
        from torch_spyre._inductor.pass_utils import broadcast_batch_vars

        x, out, _, _ = self._deps([sympy.Symbol("d1")])
        assert broadcast_batch_vars(self._op(self._info([])), x, out) == set()
        assert broadcast_batch_vars(self._op(None), x, out) == set()
