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

"""The temporary loop_info shim and the index-derived advances for it.

Device-free tests on stub ops. A ``LoopShim`` (what the loop-tiling lowering stamps
instead of a ``CoarseTileInfo``) carries the plan's own content and no advance
lists, so ``loop_facts`` reads a shim op's advances from the op's own index: the
coefficient of each loop variable its ``dim_hints`` list.
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
)
from torch_spyre._inductor.loop_info import CoarseTileInfo, LoopShim
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


def _op(loop_info, reads, writes, hints=()):
    op = SimpleNamespace(
        loop_info=loop_info,
        dim_hints=list(hints),
        get_read_writes=lambda: SimpleNamespace(reads=reads, writes=writes),
    )
    return op


def _shim(levels=1, tiled=None):
    return LoopShim(
        loop_group_id=(3,) + (0,) * (levels - 1),
        loop_count=[sympy.Integer(8)] * levels,
        loop_tiled_dims=tiled if tiled is not None else [[1]] + [[]] * (levels - 1),
        loop_tiled_reduction_dims=[[] for _ in range(levels)],
    )


def test_shim_levels_are_read_by_enclosing_loops():
    shim = _shim()
    legacy = CoarseTileInfo(
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


def test_a_shim_op_with_no_loop_variable_hint_is_loud():
    dep = _dep("arg0_1", 512 * u0)
    op = _op(_shim(), [dep], [], [])
    with pytest.raises(RuntimeError, match="loop variable"):
        read_advance(op, dep)


def test_a_non_affine_use_of_the_variable_is_unsupported():
    dep = _dep("arg0_1", d0 + sympy.floor(u0 / 2))
    op = _op(_shim(), [dep], [], [_hint(u0, 8)])
    with pytest.raises(Unsupported, match="non-affin"):
        read_advance(op, dep)


def test_a_coarsetileinfo_op_still_reads_its_lists():
    """The existing path: the stored per-read pairs, substituted into the index."""
    legacy = CoarseTileInfo(
        loop_group_id=(3,),
        loop_count=[sympy.Integer(8)],
        loop_tiled_dims=[[1]],
        loop_tiled_reduction_dims=[[]],
        tiled_dims_per_read=[[[(1, sympy.Integer(512))]]],
        output_tiled_dims=[[(1, sympy.Integer(512))]],
    )
    read, write = _dep("arg0_1", 1024 * d0 + d1), _dep("out", 1024 * d0 + d1)
    op = _op(legacy, [read], [write])
    assert read_advance(op, read) == (512,)
    assert read_advances_anywhere(op, read) is True
    assert output_advance(op) == (512,)
    assert output_advances_anywhere(op) is True
