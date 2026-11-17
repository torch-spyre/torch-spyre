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

"""Reading a loop access's advance from its index (stage 7.2).

``test_loop_advance_check.py`` pins the decision logic with both builders replaced.
This file tests the index-based builder and what it relies on, with real sympy
indexes, real dependencies and the real device projection:

* ``_loop_coefficient``: the loop variable's step in an index, and what it refuses;
* ``_matching_read_deps``: which read an access is, shared with the stored-record
  builder so the two cannot pick different reads (a read through a mutation alias
  is the case that used to differ);
* ``SpyreKernel._index_tile_advance``: dependency choice, level mapping and the
  projection to device elements;
* ``_split_loop_term`` and ``_NameSwapHandler(loop_terms=...)``: how the direct read
  that replaces a staged copy keeps the copy's loop term.
"""

from types import SimpleNamespace

import pytest
import sympy
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)
from torch._inductor.dependencies import MemoryDep, ReadWrites
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from torch_spyre._inductor import spyre_kernel
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.propagate_hints import DimHint
from torch_spyre._inductor.spyre_kernel import _loop_coefficient, _matching_read_deps
from torch_spyre._inductor.wsr.for_each_tile_lowering import _split_loop_term
from torch_spyre._inductor.wsr.loop_tiling_common import _NameSwapHandler

U0, U1 = sympy.symbols("u0 u1", integer=True)
D0, D1, D2 = sympy.symbols("d0 d1 d2", integer=True)
S = sympy.Symbol("s")


class TestLoopCoefficient:
    def test_an_affine_index_gives_its_step(self):
        assert _loop_coefficient(32 * U0 + D0, U0) == 32

    def test_an_unexpanded_product_is_read_correctly(self):
        """Expr.coeff returns 0 here; the true step is 32."""
        index = sympy.Mul(32, U0 + 1, evaluate=False) + D0
        assert index.coeff(U0) == 0
        assert _loop_coefficient(index, U0) == 32

    def test_a_constant_offset_does_not_change_the_step(self):
        assert _loop_coefficient(3 * U0 + 7 + D0, U0) == 3

    def test_a_variable_that_is_absent_has_step_zero(self):
        assert _loop_coefficient(D0 + 5, U0) == 0

    def test_the_other_loop_variable_is_not_this_ones_step(self):
        assert _loop_coefficient(8 * U0 + 16 * U1 + D0, U1) == 16

    def test_floor_of_an_integer_variable_times_an_integer_is_affine(self):
        """The loop variable is an integer, so floor(8*u0) is 8*u0."""
        assert _loop_coefficient(4096 * sympy.floor(8 * U0), U0) == 32768

    @pytest.mark.parametrize(
        "index",
        [
            sympy.floor(U0 / 2) + D0,
            64 * sympy.Mod(U0, 2) + D0,
            U0 * U0 + D0,
        ],
        ids=["floor_half", "mod_two", "square"],
    )
    def test_a_non_affine_use_is_refused(self, index):
        with pytest.raises(Unsupported, match="non-affinely"):
            _loop_coefficient(index, U0)

    def test_a_symbolic_step_is_refused(self):
        with pytest.raises(Unsupported, match="symbolic step"):
            _loop_coefficient(S * U0 + D0, U0)


def _dep(name, index):
    return MemoryDep(name, sympy.sympify(index), (), ())


class TestMatchingReadDeps:
    def test_matches_by_name(self):
        reads = [_dep("a", D0), _dep("b", D0), _dep("a", D1)]
        assert [i for i, _ in _matching_read_deps(reads, "a", {})] == [0, 2]

    def test_a_read_through_a_mutation_alias_matches_its_resolved_name(self):
        """load() passes the underlying buffer's name; the dependency has the alias's.
        A plain name comparison finds nothing and the access looks like it does not
        advance (flash-v2's B-tiled copy-out reads were pinned to tile 0 this way)."""
        reads = [_dep("buf19", 64 * D0), _dep("other", D0)]
        alias = {"buf19": "buf2"}
        assert [d.name for _, d in _matching_read_deps(reads, "buf2", alias)] == [
            "buf19"
        ]
        assert _matching_read_deps(reads, "buf19", alias) == []

    def test_no_match_is_empty(self):
        assert _matching_read_deps([_dep("a", D0)], "z", {}) == []


def _kernel(*, reads=(), writes=(), hints=(), levels=1, consumed=None):
    """A kernel whose current node has the given dependencies and loop hints."""
    kernel = spyre_kernel.SpyreKernel.__new__(spyre_kernel.SpyreKernel)
    kernel._general_tile_advance_seen = dict(consumed or {})
    kernel._tile_advance_symbols = {}
    node = SimpleNamespace(
        dim_hints=[
            DimHint(
                dim_names=["_while_loop"],
                split_count=1,
                loop_var=v,
                is_reduction=False,
                loop_var_range=4,
            )
            for v in hints
        ],
        loop_info=SimpleNamespace(
            loop_group_id=tuple(range(levels)),
            loop_count=[4] * levels,
            loop_tiled_dims=[[] for _ in range(levels)],
            loop_tiled_reduction_dims=[[] for _ in range(levels)],
        ),
        get_operation_name=lambda: "op7",
        get_read_writes=lambda: ReadWrites(
            OrderedSet(reads), OrderedSet(writes), OrderedSet()
        ),
    )
    kernel.current_node = SimpleNamespace(node=node)
    return kernel


def _tensor(device_size, stride_map):
    return SimpleNamespace(
        layout=SimpleNamespace(
            device_layout=SimpleNamespace(
                device_size=device_size, stride_map=stride_map
            )
        )
    )


# A [4, 64] buffer: device dim 0 is rows (host stride 64), dim 1 the stick (stride 1).
TENSOR = _tensor([4, 64], [64, 1])


class TestIndexTileAdvance:
    def test_a_read_advances_by_the_projected_step(self):
        kernel = _kernel(
            reads=[_dep("buf", 64 * U0 + D1)], hints=[U0], consumed={"buf": 1}
        )
        expr, dep = kernel._index_tile_advance(TENSOR, True, "buf")
        symbol = kernel._tile_advance_symbols[0]
        assert expr == sympy.floor(64 * symbol)
        assert dep.name == "buf"

    def test_a_write_uses_the_first_write(self):
        kernel = _kernel(writes=[_dep("out", 64 * U0 + D1)], hints=[U0])
        expr, _ = kernel._index_tile_advance(TENSOR, False, "out")
        assert expr == sympy.floor(64 * kernel._tile_advance_symbols[0])

    def test_the_consumed_count_picks_which_read_of_a_buffer(self):
        reads = [_dep("buf", D1), _dep("buf", 64 * U0 + D1)]
        first = _kernel(reads=reads, hints=[U0], consumed={"buf": 1})
        second = _kernel(reads=reads, hints=[U0], consumed={"buf": 2})
        assert first._index_tile_advance(TENSOR, True, "buf")[0] == 0
        assert second._index_tile_advance(TENSOR, True, "buf")[0] != 0

    def test_two_levels_map_hints_to_levels_in_order(self):
        kernel = _kernel(
            reads=[_dep("buf", 256 * U0 + 64 * U1 + D1)],
            hints=[U0, U1],
            levels=2,
            consumed={"buf": 1},
        )
        tensor = _tensor([8, 4, 64], [256, 64, 1])
        expr, _ = kernel._index_tile_advance(tensor, True, "buf")
        s0, s1 = kernel._tile_advance_symbols[0], kernel._tile_advance_symbols[1]
        assert expr == sympy.floor(256 * s0) + sympy.floor(64 * s1)

    def test_a_read_through_a_mutation_alias_is_found(self):
        """The index builder resolves aliases as the stored-record builder does."""
        kernel = _kernel(
            reads=[_dep("buf19", 64 * U0 + D1)], hints=[U0], consumed={"buf2": 1}
        )
        scheduler = SimpleNamespace(mutation_real_name={"buf19": "buf2"})
        with V.set_graph_handler(SimpleNamespace(scheduler=scheduler)):
            result = kernel._index_tile_advance(TENSOR, True, "buf2")
        assert result is not None
        assert result[0] == sympy.floor(64 * kernel._tile_advance_symbols[0])

    def test_no_hinted_variable_means_no_index_value(self):
        kernel = _kernel(reads=[_dep("buf", D1)], hints=[], consumed={"buf": 1})
        assert kernel._index_tile_advance(TENSOR, True, "buf") is None

    def test_a_hint_count_that_differs_from_the_depth_keeps_the_record(self):
        """Level i's variable is the i-th hint only with one hint per level."""
        kernel = _kernel(
            reads=[_dep("buf", 64 * U0 + D1)], hints=[U0], levels=2, consumed={"buf": 1}
        )
        assert kernel._index_tile_advance(TENSOR, True, "buf") is None

    def test_a_non_affine_index_is_refused_not_read_as_no_advance(self):
        kernel = _kernel(
            reads=[_dep("buf", sympy.floor(U0 / 2) + D1)],
            hints=[U0],
            consumed={"buf": 1},
        )
        with pytest.raises(Unsupported):
            kernel._index_tile_advance(TENSOR, True, "buf")

    def test_no_matching_read_means_no_index_value(self):
        kernel = _kernel(reads=[_dep("other", D1)], hints=[U0], consumed={"buf": 1})
        assert kernel._index_tile_advance(TENSOR, True, "buf") is None


I0 = sympy.Symbol("_fet_identity_i0", integer=True)
I1 = sympy.Symbol("_fet_identity_i1", integer=True)


class TestSplitLoopTerm:
    def test_the_loop_term_is_set_apart(self):
        loop_term, rest = _split_loop_term(128 * I0 + 8192 * U0 + I1, {U0})
        assert loop_term == 8192 * U0
        assert rest == 128 * I0 + I1

    def test_an_index_without_a_loop_term_has_a_zero_one(self):
        loop_term, rest = _split_loop_term(128 * I0 + I1, {U0})
        assert loop_term == 0
        assert rest == 128 * I0 + I1

    def test_a_cross_term_with_an_iteration_variable_is_refused(self):
        """c*i*u has the copy's own variable in it, which the consumer does not have."""
        assert _split_loop_term(128 * I0 + 4 * I0 * U0, {U0}) is None

    def test_a_stray_symbol_stays_in_the_index_for_the_identity_check_to_reject(self):
        """An indirect or unbacked symbol is not a loop variable: it must not be
        stripped and re-added; it stays, so the pure-identity check fails on it."""
        stray = sympy.Symbol("tmp0")
        loop_term, rest = _split_loop_term(128 * I0 + 8192 * U0 + stray, {U0})
        assert loop_term == 8192 * U0
        assert stray in rest.free_symbols

    def test_a_loop_term_that_mixes_in_a_stray_symbol_is_refused(self):
        stray = sympy.Symbol("tmp0")
        assert _split_loop_term(128 * I0 + stray * U0, {U0}) is None


class _Recorder:
    """Stands in for V.ops: records the loads that reach the bottom of the stack."""

    def __init__(self):
        self.loads = []

    def load(self, name, index):
        self.loads.append((name, index))
        return name


class TestNameSwapLoopTerms:
    NAME_MAP = {"copy": ("src", [sympy.Integer(128)], [sympy.Integer(128)])}

    def test_a_swapped_load_gets_the_term(self):
        recorder = _Recorder()
        handler = _NameSwapHandler(recorder, self.NAME_MAP, {"copy": 8192 * U0})
        handler.load("copy", 128 * D0)
        assert recorder.loads == [("src", 128 * D0 + 8192 * U0)]

    def test_a_direct_load_of_the_source_is_left_alone(self):
        """The consumer may also read the source on its own, with its own index."""
        recorder = _Recorder()
        handler = _NameSwapHandler(recorder, self.NAME_MAP, {"copy": 8192 * U0})
        handler.load("src", 24576 * D0)
        assert recorder.loads == [("src", 24576 * D0)]

    def test_without_loop_terms_the_swap_is_unchanged(self):
        recorder = _Recorder()
        _NameSwapHandler(recorder, self.NAME_MAP).load("copy", 128 * D0)
        assert recorder.loads == [("src", 128 * D0)]
