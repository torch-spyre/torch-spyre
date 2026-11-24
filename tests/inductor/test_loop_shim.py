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

"""Index-derived advances, for an op the loop-tiling lowering made.

Device-free tests on stub ops. A ``LoopMember`` with ``strict_loop_vars`` (what the
loop-tiling lowering stamps) lists the variable of every level in the op's
``dim_hints``, so ``loop_facts`` reads its advances from the op's own index: the
coefficient of each loop variable it lists. ("Shim" in the test names is that op.)
"""

from types import SimpleNamespace

import pytest
import sympy
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import sympy_index_symbol

from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.loop_facts import (
    enclosing_loops,
    output_advance,
    output_advances_anywhere,
    read_advance,
    read_advances_anywhere,
    advancing_levels,
)
from torch_spyre._inductor.loop_node import LoopMember
from torch_spyre._inductor.propagate_hints import DimHint

d0, d1 = sympy_index_symbol("d0"), sympy_index_symbol("d1")
u0 = sympy.Symbol("u0", integer=True, nonnegative=True)
u1 = sympy.Symbol("u1", integer=True, nonnegative=True)


def _dep(name, index):
    return MemoryDep(name, index, (d0, d1), (4096, 512))


def _hint(variable, trips):
    return DimHint(
        dim_names=[],
        split_count=1,
        loop_var=variable,
        is_reduction=False,
        loop_var_range=sympy.Integer(trips),
    )


def _op(loop_member, reads, writes, hints=()):
    op = SimpleNamespace(
        loop_member=loop_member,
        dim_hints=list(hints),
        get_read_writes=lambda: SimpleNamespace(reads=reads, writes=writes),
    )
    return op


def _shim(levels=1, tiled=None):
    return LoopMember(
        loop_group_id=(3,) + (0,) * (levels - 1),
        loop_count=[sympy.Integer(8)] * levels,
        loop_tiled_dims=tiled if tiled is not None else [[1]] + [[]] * (levels - 1),
        loop_tiled_reduction_dims=[[] for _ in range(levels)],
        strict_loop_vars=True,
    )


def test_shim_levels_are_read_by_enclosing_loops():
    shim = _shim()
    legacy = LoopMember(
        loop_group_id=(3,),
        loop_count=[sympy.Integer(8)],
        loop_tiled_dims=[[1]],
        loop_tiled_reduction_dims=[[]],
    )
    assert enclosing_loops(_op(shim, [], [])) == enclosing_loops(_op(legacy, [], []))
    (level,) = enclosing_loops(_op(shim, [], []))
    assert level.trip_count == 8 and level.tiled_dims == (1,)


def test_read_advance_of_a_shim_op_is_the_loop_coefficient_of_its_index():
    dep = _dep("arg0_1", 1024 * d0 + d1 + 512 * u0)
    op = _op(_shim(), [dep], [], [_hint(u0, 8)])
    assert read_advance(op, dep) == (512,)
    assert read_advances_anywhere(op, dep) is True


def test_an_access_without_the_variable_does_not_advance():
    dep = _dep("arg0_1", 1024 * d0 + d1)
    op = _op(_shim(), [dep], [], [_hint(u0, 8)])
    assert read_advance(op, dep) == (0,)
    assert read_advances_anywhere(op, dep) is False


def test_a_dependency_that_is_not_a_read_of_the_op_gives_nothing():
    dep, other = _dep("a", 512 * u0), _dep("b", 512 * u0)
    op = _op(_shim(), [dep], [], [_hint(u0, 8)])
    assert read_advance(op, other) == ()
    assert read_advances_anywhere(op, other) is False


def test_output_advance_uses_the_first_write():
    first, second = _dep("out", 256 * u0 + d1), _dep("out2", 999 * u0)
    op = _op(_shim(), [], [first, second], [_hint(u0, 8)])
    assert output_advance(op) == (256,)
    assert output_advances_anywhere(op) is True


def test_an_output_that_does_not_advance():
    op = _op(_shim(), [], [_dep("out", d1 + 1024 * d0)], [_hint(u0, 8)])
    assert output_advance(op) == (0,)
    assert output_advances_anywhere(op) is False


def test_two_levels_use_the_loop_variables_outermost_first():
    dep = _dep("arg0_1", 4 * u0 + 100 * u1 + d0)
    op = _op(_shim(levels=2), [dep], [], [_hint(u0, 2), _hint(u1, 4)])
    assert read_advance(op, dep) == (4, 100)
    only_inner = _dep("arg1_1", 100 * u1 + d0)
    op2 = _op(_shim(levels=2), [only_inner], [], [_hint(u0, 2), _hint(u1, 4)])
    assert read_advance(op2, only_inner) == (0, 100)
    assert read_advances_anywhere(op2, only_inner) is True


def test_the_advancing_levels_of_a_shim_op_are_the_levels_any_access_moves_on():
    read = _dep("arg0_1", 100 * u1 + d0)
    other = _dep("arg1_1", d0)
    write = _dep("buf0", 4 * u0 + d0)
    hints = [_hint(u0, 2), _hint(u1, 4)]
    op = _op(_shim(levels=2), [read, other], [write], hints)
    assert advancing_levels(op) == frozenset({0, 1})
    no_write_term = _op(_shim(levels=2), [read, other], [_dep("buf0", d0)], hints)
    assert advancing_levels(no_write_term) == frozenset({1})
    still = _op(_shim(levels=2), [other], [_dep("buf0", d0)], hints)
    assert advancing_levels(still) == frozenset()
    assert advancing_levels(_op(_shim(levels=2), [], [], hints)) == frozenset()


def test_caller_supplied_reads_are_used_in_place_of_get_read_writes():
    """scratchpad passes the memoized op_read_writes reads; the lookup must use them,
    not re-trace the op."""
    x, y = _dep("x", 128 * d0 + d1), _dep("y", 4 * u0 + d1)
    op = _op(_shim(), [x, y], [], [_hint(u0, 8)])
    op.get_read_writes = None  # would raise if called
    assert read_advances_anywhere(op, y, reads=[x, y])
    assert not read_advances_anywhere(op, x, reads=[x, y])
    assert not read_advances_anywhere(op, _dep("z", u0), reads=[x, y])


def test_an_output_with_no_memory_write_does_not_advance():
    op = _op(_shim(), [_dep("x", 4 * u0)], [], [_hint(u0, 8)])
    assert output_advance(op) == ()
    assert output_advances_anywhere(op) is False


def test_an_ordinary_hint_is_not_a_loop_variable():
    """A span-overflow hint (loop_var is an iteration symbol, no loop_var_range)
    must not be taken for the loop's variable."""
    dep = _dep("arg0_1", 512 * u0 + d1)
    ordinary = DimHint(
        dim_names=["_span_overflow"],
        split_count=2,
        loop_var=d1,
        is_reduction=False,
        hint_id=10000,
    )
    op = _op(_shim(), [dep], [], [ordinary, _hint(u0, 8)])
    assert read_advance(op, dep) == (512,)


def test_a_for_each_tile_op_that_lists_no_loop_variable_does_not_advance():
    """It is inside the loop but indexes nothing with its variable (the loop's fill,
    for one): zero at every level, not an error."""
    dep = _dep("arg0_1", d0 + d1)
    legacy = LoopMember(
        loop_group_id=(3, 0),
        loop_count=[sympy.Integer(8)] * 2,
        loop_tiled_dims=[[], []],
        loop_tiled_reduction_dims=[[], []],
    )
    op = _op(legacy, [dep], [], [])
    assert read_advance(op, dep) == (0, 0)
    assert read_advances_anywhere(op, dep) is False
    assert advancing_levels(op) == frozenset()


def test_a_shim_op_with_no_loop_variable_hint_is_loud():
    """The lowering lists the variable of every level on every op it makes, so a shim
    op that lists none is a compiler bug: pinning its address across trips would be
    the wrong-value class of #5573."""
    dep = _dep("arg0_1", 512 * u0)
    op = _op(_shim(), [dep], [], [])
    with pytest.raises(RuntimeError, match="loop variable"):
        read_advance(op, dep)


def test_a_loop_variable_count_that_differs_from_the_depth_is_loud():
    """The pairing of variable to level is then unknown."""
    dep = _dep("arg0_1", 512 * u0)
    op = _op(_shim(levels=2), [dep], [], [_hint(u0, 8)])
    with pytest.raises(RuntimeError, match="loop variable"):
        read_advance(op, dep)


def test_a_non_affine_use_of_the_variable_is_unsupported():
    dep = _dep("arg0_1", d0 + sympy.floor(u0 / 2))
    op = _op(_shim(), [dep], [], [_hint(u0, 8)])
    with pytest.raises(Unsupported, match="non-affin"):
        read_advance(op, dep)


def test_a_coarsetileinfo_op_is_read_from_its_index_too():
    """A ``for_each_tile`` op carries no more than the loop's own facts; whatever
    stored lists it still has are ignored, so an empty record gives the same advance."""
    info = LoopMember(
        loop_group_id=(3,),
        loop_count=[sympy.Integer(8)],
        loop_tiled_dims=[[1]],
        loop_tiled_reduction_dims=[[]],
    )
    read, write = _dep("arg0_1", 512 * u0 + d1), _dep("out", 256 * u0 + d1)
    op = _op(info, [read], [write], [_hint(u0, 8)])
    assert read_advance(op, read) == (512,)
    assert read_advances_anywhere(op, read) is True
    assert output_advance(op) == (256,)
    assert output_advances_anywhere(op) is True
    assert advancing_levels(op) == frozenset({0})


# ---------------------------------------------------------------------------
# The readers that can see a shim op (the sites below are what an audit of
# the readers found)
# ---------------------------------------------------------------------------

from unittest import mock  # noqa: E402

from torch._inductor.ir import Pointwise  # noqa: E402

from torch_spyre._inductor import read_copy_elision, spyre_kernel  # noqa: E402
from torch_spyre._inductor.ir import FixedTiledLayout  # noqa: E402
from torch_spyre._inductor.scratchpad.allocator import (  # noqa: E402
    _coarse_tile_loop_trip_count,
    _is_coarse_tiled,
)


def _equal_infos():
    shim = _shim()
    legacy = LoopMember(
        loop_group_id=shim.loop_group_id,
        loop_count=list(shim.loop_count),
        loop_tiled_dims=[list(d) for d in shim.loop_tiled_dims],
        loop_tiled_reduction_dims=[list(d) for d in shim.loop_tiled_reduction_dims],
    )
    return shim, legacy


def test_the_allocator_treats_a_shim_op_as_coarse_tiled():
    """``_coarse_tile_loop_trip_count`` answers 1 for anything that is not coarse
    tiled, which for a shim op would silently drop the loop's trips from the
    relayout prices."""
    shim, legacy = _equal_infos()
    for info in (shim, legacy):
        op = SimpleNamespace(loop_member=info)
        assert _is_coarse_tiled(op) is True
        assert _coarse_tile_loop_trip_count(op) == 8
    assert _is_coarse_tiled(SimpleNamespace()) is False
    assert _coarse_tile_loop_trip_count(SimpleNamespace(loop_member=None)) == 1


def _kernel_for(op_loop_info, *, reads=()):
    """A kernel positioned on a stub op, with the real bookkeeping and a stub
    ``_index_tile_advance``."""
    kernel = spyre_kernel.SpyreKernel.__new__(spyre_kernel.SpyreKernel)
    kernel._tile_advance_seen = {}
    kernel._tile_advance_rw = None
    node = SimpleNamespace(
        loop_member=op_loop_info,
        get_operation_name=lambda: "op7",
        get_read_writes=lambda: SimpleNamespace(reads=list(reads), writes=[]),
    )
    kernel.current_node = SimpleNamespace(node=node)
    return kernel


def _patched_v():
    return mock.patch.object(
        spyre_kernel, "V", SimpleNamespace(graph=SimpleNamespace(scheduler=None))
    )


def test_the_kernel_takes_a_shim_ops_advance_from_the_index():
    """No stored record exists; the advance is the index's."""
    kernel = _kernel_for(_shim(), reads=[_dep("arg0_1", 512 * u0)])
    advance = 512 * sympy.Symbol("u0")
    kernel._index_tile_advance = lambda tensor, is_input, name: (advance, None)
    with _patched_v():
        assert kernel._tile_advance(None, True, "arg0_1") == advance


def test_a_zero_advance_of_a_shim_op_is_none():
    kernel = _kernel_for(_shim(), reads=[_dep("arg0_1", d1)])
    kernel._index_tile_advance = lambda tensor, is_input, name: (sympy.S.Zero, None)
    with _patched_v():
        assert kernel._tile_advance(None, True, "arg0_1") is None


def test_the_kernel_counts_each_read_of_a_shim_op_and_a_repeat_load_is_none():
    """``_index_tile_advance`` picks a read by how many the kernel has counted;
    the kernel counts them itself in ``_tile_advance``, or
    every read would answer 'no advance'."""
    reads = [_dep("x", 512 * u0), _dep("x", 256 * u0)]
    kernel = _kernel_for(_shim(), reads=reads)
    seen_at_call = []

    def index_advance(tensor, is_input, name):
        seen_at_call.append(kernel._tile_advance_seen.get(name, 0))
        return sympy.Integer(len(seen_at_call)) * sympy.Symbol("u0"), None

    kernel._index_tile_advance = index_advance
    with _patched_v():
        first = kernel._tile_advance(None, True, "x")
        second = kernel._tile_advance(None, True, "x")
        third = kernel._tile_advance(None, True, "x")  # nothing left to read
    assert seen_at_call == [1, 2]
    assert first == sympy.Symbol("u0") and second == 2 * sympy.Symbol("u0")
    assert third is None


def test_a_for_each_tile_op_that_lists_no_loop_variable_has_no_advance():
    """It is inside a loop but nothing it indexes moves with it (the loop's fill, for
    one): the index gives no advance, which is "no advance", not an error."""
    _, legacy = _equal_infos()
    kernel = _kernel_for(legacy, reads=[_dep("arg0_1", 512 * u0)])
    kernel._index_tile_advance = lambda tensor, is_input, name: None
    with _patched_v():
        assert kernel._tile_advance(None, True, "arg0_1") is None


def test_a_shim_op_with_no_index_advance_is_loud():
    kernel = _kernel_for(_shim(), reads=[_dep("arg0_1", 512 * u0)])
    kernel._index_tile_advance = lambda tensor, is_input, name: None
    with _patched_v(), pytest.raises(RuntimeError, match="loop variable"):
        kernel._tile_advance(None, True, "arg0_1")


def test_a_write_takes_the_first_write_and_counts_no_read():
    """A write is not matched against the buffer's reads: the counter only moves for
    reads, so a read of the same buffer afterwards still takes its first match."""
    kernel = _kernel_for(_shim(), reads=[_dep("buf", 512 * u0)])
    seen = []

    def index_advance(tensor, is_input, name):
        seen.append((is_input, kernel._tile_advance_seen.get(name, 0)))
        return sympy.Symbol("u0") * 4, None

    kernel._index_tile_advance = index_advance
    with _patched_v():
        kernel._tile_advance(None, False, "buf")
        kernel._tile_advance(None, True, "buf")
    assert seen == [(False, 0), (True, 1)]


def test_read_copy_elision_keeps_a_copy_for_a_shim_consumer():
    """Decided: it does not migrate; only a ``for_each_tile`` consumer (a
    ``LoopMember`` without ``strict_loop_vars``) has a saved direct form, so a consumer
    the loop-tiling lowering made keeps its copy, and says so, not crashes."""
    consumer_is_shim, copy_is_shim = True, False
    _, legacy = _equal_infos()
    other = _dep("other", d0)
    cp = _dep("cp", d0 + d1)
    out = _dep("out", d0 + d1)
    layout = object()
    consumer = SimpleNamespace(
        get_name=lambda: "c",
        data=mock.MagicMock(spec=Pointwise),
        get_read_writes=lambda: SimpleNamespace(reads=[cp, other], writes=[out]),
        get_layout=lambda: layout,
        loop_member=_shim() if consumer_is_shim else legacy,
    )
    copy_op = SimpleNamespace(
        get_layout=lambda: mock.MagicMock(spec=FixedTiledLayout),
        get_read_writes=lambda: SimpleNamespace(
            reads=[_dep("src", d0 + d1)], writes=[cp]
        ),
        loop_member=_shim() if copy_is_shim else legacy,
    )
    direct = SimpleNamespace(
        get_read_writes=lambda: SimpleNamespace(
            reads=[_dep("src", d0 + d1), other], writes=[out]
        ),
        get_layout=lambda: layout,
    )
    record = SimpleNamespace(
        consumer_name="c",
        source_name="src",
        copy_name="cp",
    )
    with mock.patch.object(
        read_copy_elision, "_clone_direct_consumer", lambda *a: direct
    ):
        built, reason = read_copy_elision._prove_matmul_direct_read(
            consumer, copy_op, record
        )
    assert built is None
    assert reason == "copy reads its source more than once, or consumer is not looped"
