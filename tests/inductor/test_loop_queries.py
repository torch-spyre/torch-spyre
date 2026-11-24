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

"""Unit tests for ``loop_queries.LoopProgram`` (CPU, stub ops)."""

from types import SimpleNamespace

import pytest
import sympy
import torch
from torch._inductor.dependencies import MemoryDep, ReadWrites
from torch._inductor.ir import ExternKernel
from torch.utils._ordered_set import OrderedSet

from torch_spyre._inductor.loop_node import (
    LoopCarryRecord,
    LoopMember,
    LoopNode,
    OpsView,
)
from torch_spyre._inductor.loop_queries import LoopProgram
from torch_spyre._inductor.pass_utils import trip_count_estimate


def _dep(name):
    """``"x@512"`` is a second dep on ``x`` with a different index."""
    base, _, off = name.partition("@")
    return MemoryDep(base, sympy.Integer(int(off or 0)), (), ())


class _Op:
    """A stub op: a name, reads/writes and an optional ``loop_member``."""

    def __init__(self, name, gid=None, counts=None, reads=(), writes=None):
        self._name = name
        if gid is not None:
            self.loop_member = LoopMember(
                loop_group_id=tuple(gid),
                loop_count=list(counts),
                loop_tiled_dims=[[] for _ in gid],
                loop_tiled_reduction_dims=[[] for _ in gid],
            )
        self._reads = [_dep(r) for r in reads]
        self._writes = [_dep(w) for w in (writes if writes is not None else [name])]

    def get_name(self):
        return self._name

    def get_operation_name(self):
        return self._name

    def get_device(self):
        return torch.device("cpu")

    def get_outputs(self):
        return [self]

    def get_read_writes(self):
        return ReadWrites(
            OrderedSet(self._reads), OrderedSet(self._writes), OrderedSet()
        )

    def __repr__(self):
        return f"_Op({self._name})"


class _Extern(ExternKernel):
    """A stub extern kernel (not an allocation)."""

    def __init__(self, name):  # noqa: D107 - the dataclass init is not wanted
        self._name = name

    def get_name(self):
        return self._name

    def get_operation_name(self):
        return self._name

    def get_read_writes(self):
        return ReadWrites(OrderedSet(), OrderedSet([_dep(self._name)]), OrderedSet())

    def __repr__(self):
        return f"_Extern({self._name})"

    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other


class _Alloc(_Extern):
    """A zero-element allocation: an extern kernel whose ``is_no_op()`` is true."""

    def is_no_op(self):
        return True


def _node(name, body, trip=4, induction=None, carries=()):
    return LoopNode(name, body, induction=induction, trip_count=trip, carries=carries)


def _Op_in(p, name):
    return next(op for op in p.ops() if op.get_name() == name)


def test_slots_follow_program_order_through_nested_nodes():
    a, b, c, d = _Op("a"), _Op("b"), _Op("c"), _Op("d")
    tree = [a, _node("n0", [b, _node("n1", [c], trip=3)], trip=4), d]
    p = LoopProgram(tree)
    assert [x.get_name() for x in p.ops()] == ["a", "b", "c", "d"]
    assert [p.slot(x) for x in (a, b, c, d)] == [0, 1, 2, 3]


def test_view_list_and_graph_give_the_same_program():
    tree = [_Op("a"), _node("n0", [_Op("b")])]
    for src in (tree, OpsView(tree), SimpleNamespace(operations=tree)):
        assert [x.get_name() for x in LoopProgram(src).ops()] == ["a", "b"]


def test_executions_multiply_the_trip_counts_of_the_enclosing_nodes():
    c = _Op("c")
    p = LoopProgram([_node("n0", [_node("n1", [c], trip=3)], trip=4)])
    assert p.executions(c) == 12 and p.executions_estimate(c) == 12
    assert [lp.node.get_name() for lp in p.loops_of(c)] == ["n0", "n1"]


def test_an_allocation_and_a_non_member_in_a_body_run_every_trip():
    alloc, clone = _Alloc("buf"), _Op("clone")  # clone has no loop_member
    p = LoopProgram([_node("n0", [alloc, clone, _Op("k")], trip=4)])
    assert p.executions(alloc) == 4 and p.executions(clone) == 4


def test_an_extern_in_a_body_runs_once_before_its_outermost_loop():
    pre, ext = _Op("pre"), _Extern("ext")
    p = LoopProgram([pre, _node("n0", [_Op("k0"), _node("n1", [ext], trip=2)], trip=4)])
    assert p.loops_of(ext) == () and p.executions(ext) == 1
    assert p.time(ext) == p.span(p.outermost(_Op_in(p, "k0")))[0] == 1


def test_shared_loops_and_same_iteration():
    b, c, e = _Op("b"), _Op("c"), _Op("e")
    n1 = _node("n1", [c], trip=3)
    p = LoopProgram([_node("n0", [b, n1], trip=4), e])
    assert [lp.node for lp in p.shared_loops(b, c)] == [p.loops_of(b)[0].node]
    assert p.shared_loops(b, e) == ()
    assert p.same_iteration(b, b) and not p.same_iteration(b, c)


def test_a_pending_loop_is_answered_from_the_lowerings_record():
    x, y = _Op("x", gid=(0,), counts=[5]), _Op("y", gid=(0,), counts=[5])
    p = LoopProgram([_Op("a"), x, y])
    (loop,) = p.loops_of(x)
    assert loop.node is None and loop.path == (0,) and loop is p.loops_of(y)[0]
    assert p.span(loop) == (1, 2) and p.executions(x) == 5


# Review Focus 1-3
def test_an_empty_node_has_no_span():
    n = _node("n0", [])
    p = LoopProgram([_Op("a"), n])
    assert p.span(p.loop_for(n)) is None


def test_a_symbolic_count_stays_symbolic_and_its_estimate_resolves():
    s = sympy.Symbol("s0", positive=True, integer=True)
    c = _Op("c")
    p = LoopProgram([_node("n0", [c], trip=s)])
    assert p.executions(c) == s and p.executions_estimate(c) == trip_count_estimate(s)


def test_an_op_not_in_the_program_raises_with_its_name():
    with pytest.raises(KeyError, match="ghost"):
        LoopProgram([_Op("a")]).slot(_Op("ghost"))


# Task 1 review follow-ups
def test_a_top_level_extern_with_a_pending_member_runs_once_outside_loops():
    ext = _Extern("ext")
    ext.loop_member = LoopMember(
        loop_group_id=(0,),
        loop_count=[5],
        loop_tiled_dims=[[]],
        loop_tiled_reduction_dims=[[]],
    )
    p = LoopProgram([_Op("a"), ext])
    assert p.loops_of(ext) == () and p.executions(ext) == 1


def test_nested_pending_loops_are_shared_by_group_path():
    a = _Op("a", gid=(0, 1), counts=[2, 3])
    b = _Op("b", gid=(0, 1), counts=[2, 3])
    c = _Op("c", gid=(0,), counts=[2])
    p = LoopProgram([a, b, c])
    outer = p.loops_of(a)[0]
    assert outer is p.loops_of(b)[0] and outer is p.loops_of(c)[0]
    inner = p.loops_of(a)[1]
    assert inner is p.loops_of(b)[1] and p.loops_of(c) == (outer,)
    assert [p.executions(x) for x in (a, b, c)] == [6, 6, 2]
    assert p.span(inner) == (0, 1) and p.span(outer) == (0, 2)


# Buffer queries
def test_a_value_born_outside_and_read_inside_lives_for_the_whole_loop():
    x = _Op("x", writes=["x"])
    r = _Op("r", reads=["x"])
    p = LoopProgram([x, _node("n0", [_Op("k0"), r, _Op("k2")], trip=4), _Op("z")])
    assert p.uses("x") == [0, 2] and p.live_interval("x") == (0, 3)


def test_a_body_temporary_keeps_a_per_trip_lifetime():
    alloc = _Alloc("t")
    w = _Op("w", writes=["t"])
    r = _Op("r", reads=["t"])
    p = LoopProgram([_node("n0", [_Op("k"), alloc, w, r, _Op("k2")], trip=4)])
    assert p.live_interval("t") == (1, 3)


def test_a_write_inside_a_loop_does_not_widen():
    x, w = _Op("x", writes=["x"]), _Op("w", writes=["x"])
    p = LoopProgram([x, _node("n0", [_Op("k"), w, _Op("k2")], trip=4)])
    assert p.live_interval("x") == (0, 2)


def test_a_carry_storage_covers_its_loop():
    init = _Op("init", writes=["acc"])
    upd = _Op("upd", reads=["acc"], writes=["acc"])
    rec = LoopCarryRecord(storage_name="acc", update_name="upd")
    n = _node("n0", [_Op("k"), upd, _Op("k2")], trip=4, carries=[rec])
    assert LoopProgram([init, n]).live_interval("acc") == (0, 3)


def test_runtime_reads_count_each_reading_op_per_execution():
    x, r = _Op("x", writes=["x"]), _Op("r", reads=["x"])
    inner = _node("n1", [r], trip=3)
    p = LoopProgram([x, _node("n0", [inner], trip=4), _Op("z", reads=["x"])])
    assert p.runtime_reads("x") == 13


def test_an_unused_buffer_has_no_uses_and_no_lifetime():
    p = LoopProgram(SimpleNamespace(operations=[_Op("a")], graph_input_names=["in0"]))
    assert p.uses("in0") == [] and p.live_interval("in0") is None
    assert p.live_interval("nope") is None
    assert p.input_names() == ("in0",) and LoopProgram([]).input_names() == ()


def test_one_op_reaching_a_buffer_twice_counts_once():
    r = _Op("r", reads=["x", "x@512"])
    p = LoopProgram([_Op("x", writes=["x"]), _node("n0", [r], trip=4)])
    assert p.uses("x") == [0, 1] and p.runtime_reads("x") == 4
