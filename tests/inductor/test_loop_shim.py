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
    shim_advancing_levels,
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


def test_the_advancing_levels_of_a_shim_op_are_the_levels_any_access_moves_on():
    read = _dep("arg0_1", 100 * u1 + d0)
    other = _dep("arg1_1", d0)
    write = _dep("buf0", 4 * u0 + d0)
    hints = [_hint(u0, 2), _hint(u1, 4)]
    op = _op(_shim(levels=2), [read, other], [write], hints)
    assert shim_advancing_levels(op) == frozenset({0, 1})
    no_write_term = _op(_shim(levels=2), [read, other], [_dep("buf0", d0)], hints)
    assert shim_advancing_levels(no_write_term) == frozenset({1})
    still = _op(_shim(levels=2), [other], [_dep("buf0", d0)], hints)
    assert shim_advancing_levels(still) == frozenset()
    assert shim_advancing_levels(_op(_shim(levels=2), [], [], hints)) == frozenset()


def test_the_advancing_levels_of_a_coarsetileinfo_op_are_not_asked_of_the_index():
    dep = _dep("arg0_1", 100 * u1 + d0)
    op = _op(_equal_infos()[1], [dep], [], [_hint(u0, 2)])
    assert shim_advancing_levels(op) == frozenset()


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


# ---------------------------------------------------------------------------
# The readers that can see a shim op (the inventory is tests/inductor/
# loop_shim_audit.py; the sites below are what it found)
# ---------------------------------------------------------------------------

import ast  # noqa: E402
import pathlib  # noqa: E402
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
    legacy = CoarseTileInfo(
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
        op = SimpleNamespace(loop_info=info)
        assert _is_coarse_tiled(op) is True
        assert _coarse_tile_loop_trip_count(op) == 8
    assert _is_coarse_tiled(SimpleNamespace()) is False
    assert _coarse_tile_loop_trip_count(SimpleNamespace(loop_info=None)) == 1


# Where a bare ``isinstance(x, CoarseTileInfo)`` is correct, and why. Anything else
# must test for the shim as well (``COUNTED_LOOP_INFO``).
_BARE_ISINSTANCE_ALLOWED = {
    # pass 0: runs before any shim can exist, on for_each_tile ops only
    "wsr/for_each_tile_lowering.py": "for_each_tile ops are stamped at pass 0",
    # declines by design: it keeps a copy for an op without a complete address record
    "read_copy_elision.py": "keeps the copy for a shim consumer (decided)",
}


def test_every_bare_isinstance_coarsetileinfo_site_is_classified():
    root = pathlib.Path(__file__).resolve().parents[2] / "torch_spyre" / "_inductor"
    unclassified = []
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "isinstance"
                and len(node.args) == 2
                and ast.unparse(node.args[1]) == "CoarseTileInfo"
                and rel not in _BARE_ISINSTANCE_ALLOWED
            ):
                unclassified.append(f"{rel}:{node.lineno}")
    assert unclassified == [], (
        "these tests would read as False for a LoopShim op; widen them to "
        f"COUNTED_LOOP_INFO or classify them above: {unclassified}"
    )


def _kernel_for(op_loop_info, *, reads=()):
    """A kernel positioned on a stub op, with the real bookkeeping and a stub
    ``_index_tile_advance``."""
    kernel = spyre_kernel.SpyreKernel.__new__(spyre_kernel.SpyreKernel)
    kernel._general_tile_advance_seen = {}
    node = SimpleNamespace(
        loop_info=op_loop_info,
        get_operation_name=lambda: "op7",
        get_read_writes=lambda: SimpleNamespace(reads=list(reads), writes=[]),
    )
    kernel.current_node = SimpleNamespace(node=node)

    def never(*args, **kwargs):
        raise AssertionError("the stored-record builder ran for a shim op")

    kernel._general_tile_advance = never
    return kernel


def _patched_v():
    return mock.patch.object(
        spyre_kernel, "V", SimpleNamespace(graph=SimpleNamespace(scheduler=None))
    )


def test_the_kernel_skips_the_advance_check_for_a_shim_op():
    """No stored record exists to compare with; the advance is the index's."""
    kernel = _kernel_for(_shim(), reads=[_dep("arg0_1", 512 * u0)])
    advance = 512 * sympy.Symbol("u0")
    kernel._index_tile_advance = lambda tensor, is_input, name: (advance, None)
    with _patched_v():
        assert kernel._checked_tile_advance(None, True, "arg0_1") == advance


def test_a_zero_advance_of_a_shim_op_is_none():
    kernel = _kernel_for(_shim(), reads=[_dep("arg0_1", d1)])
    kernel._index_tile_advance = lambda tensor, is_input, name: (sympy.S.Zero, None)
    with _patched_v():
        assert kernel._checked_tile_advance(None, True, "arg0_1") is None


def test_the_kernel_counts_each_read_of_a_shim_op_and_a_repeat_load_is_none():
    """``_index_tile_advance`` picks a read by how many the kernel has counted;
    the record builder is bypassed, so the shim branch must count them itself, or
    every read of a shim op would answer 'no advance'."""
    reads = [_dep("x", 512 * u0), _dep("x", 256 * u0)]
    kernel = _kernel_for(_shim(), reads=reads)
    seen_at_call = []

    def index_advance(tensor, is_input, name):
        seen_at_call.append(kernel._general_tile_advance_seen.get(name, 0))
        return sympy.Integer(len(seen_at_call)) * sympy.Symbol("u0"), None

    kernel._index_tile_advance = index_advance
    with _patched_v():
        first = kernel._checked_tile_advance(None, True, "x")
        second = kernel._checked_tile_advance(None, True, "x")
        third = kernel._checked_tile_advance(None, True, "x")  # nothing left to read
    assert seen_at_call == [1, 2]
    assert first == sympy.Symbol("u0") and second == 2 * sympy.Symbol("u0")
    assert third is None


def test_a_shim_op_with_no_index_advance_is_loud():
    kernel = _kernel_for(_shim(), reads=[_dep("arg0_1", 512 * u0)])
    kernel._index_tile_advance = lambda tensor, is_input, name: None
    with _patched_v(), pytest.raises(RuntimeError, match="loop variable"):
        kernel._checked_tile_advance(None, True, "arg0_1")


@pytest.mark.parametrize(
    "consumer_is_shim, copy_is_shim",
    [(True, False), (False, True), (True, True)],
    ids=["shim_consumer", "shim_copy", "both_shim"],
)
def test_read_copy_elision_keeps_a_copy_when_either_op_is_a_shim(
    consumer_is_shim, copy_is_shim
):
    """Decided: it does not migrate; an op without a complete loop-address record
    keeps its copy, and says so, not crashes. It tests the consumer and the copy
    separately and returns the same reason for both, so each combination is run:
    a test with both shim could not tell which check declined."""
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
        loop_info=_shim() if consumer_is_shim else legacy,
    )
    copy_op = SimpleNamespace(
        get_layout=lambda: mock.MagicMock(spec=FixedTiledLayout),
        get_read_writes=lambda: SimpleNamespace(
            reads=[_dep("src", d0 + d1)], writes=[cp]
        ),
        loop_info=_shim() if copy_is_shim else legacy,
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
        direct_tiled_dims_per_level=None,
        direct_squeezed_advance_per_level=None,
    )
    with mock.patch.object(
        read_copy_elision, "_clone_direct_consumer", lambda *a: direct
    ):
        built, reason = read_copy_elision._prove_matmul_direct_read(
            consumer, copy_op, record
        )
    assert built is None
    assert reason == "copy has no complete loop-address record"
