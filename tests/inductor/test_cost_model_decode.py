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

"""Unit tests for the matmul M/N/K decode in ``dump_cost_model._matmul_features``.

Regression guard for the batchmatmul decode bug: for a 3D ``[B, M, N]`` output the
batch stride is the LARGEST write-index coefficient, so the old "largest coeff = M"
rule mis-picked the batch as M for B>=2 -- ``matmul_rows_per_core`` came out as the
batch size instead of M/m, corrupting pt_eff and the spill term. The fix excludes the
batch var (via the named-dim map when present, else by dropping the largest-coeff
var(s)) before choosing M and N.

No Spyre device or backend compiler is required; the iteration space and split maps
are injected so the pure decode logic is exercised in isolation.
"""

import dataclasses
from types import SimpleNamespace
import unittest
from unittest import mock

import pytest
import sympy
import torch
from sympy import Symbol
from torch._inductor.dependencies import MemoryDep, ReadWrites
from torch._inductor.ir import InputBuffer, MutationLayoutSHOULDREMOVE, Scatter
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

import torch_spyre._inductor.dump_cost_model as dcm
from torch_spyre._C import DataFormats, SpyreTensorLayout
from torch_spyre._inductor import work_division as wd
from torch_spyre._inductor.cost_model import ArgTraffic
from torch_spyre._inductor.dump_cost_model import (
    _level_loop_vars,
    _loop_factor_for_index,
    _loop_var_advances,
    _stamped_advances,
)
from torch_spyre._inductor.ir import FixedTiledLayout


class _FakeDep:
    def __init__(self, index):
        self.index = index


class _FakeRW:
    def __init__(self, write_index, read_index):
        self.writes = [_FakeDep(write_index)]
        self.reads = [_FakeDep(read_index)]


class _FakeData:
    # reduction (K) range -> drives macs = out_elems * K
    reduction_ranges = [2048]


class _FakeOp:
    """Minimal stand-in for a batchmatmul ComputedBuffer the decode reads."""

    def __init__(self, write_index, read_index, splits, wdli=None):
        self.data = _FakeData()
        self.op_it_space_splits = splits  # truthy -> decode runs
        self._w = write_index
        self._r = read_index
        if wdli is not None:
            self.work_div_loop_info = wdli

    def get_read_writes(self):
        return _FakeRW(self._w, self._r)


def _patch(monkeypatch, it_space, split_map):
    monkeypatch.setattr(dcm, "iteration_space_from_op", lambda op: it_space)
    monkeypatch.setattr(dcm, "apply_splits_from_index_coeff", lambda *a, **k: split_map)


def test_bmm_b_ge_2_excludes_batch_from_m(monkeypatch):
    """B=4, M=1024, N=1024, K=2048, forced split 1x16x2x1. Batch stride is largest."""
    b, m, n, kk = sympy.symbols("b m n kk", positive=True, integer=True)
    it_space = {b: 4, m: 1024, n: 1024, kk: 2048}
    write_index = 1048576 * b + 1024 * m + n  # strides: B > M > N (N is the stick, 1)
    split_map = {b: 1, m: 16, n: 2, kk: 1}
    _patch(monkeypatch, it_space, split_map)

    op = _FakeOp(write_index, 1024 * b + m + kk, {"d0": 1, "d1": 16, "d2": 2, "d3": 1})
    macs, rows_per_core, cols_per_core, a_bytes, b_bytes, k_split, m_split, n_split = (
        dcm._matmul_features(op, out_elems=4 * 1024 * 1024, dtype_bytes=2)
    )

    # THE regression: M/m must be 1024/16 = 64, NOT the batch size (4).
    assert rows_per_core == 64
    assert cols_per_core == 512  # N/n = 1024/2
    assert (m_split, n_split, k_split) == (16, 2, 1)
    assert a_bytes == 1024 * 2048 * 2 and b_bytes == 2048 * 1024 * 2
    assert macs == 4 * 1024 * 1024 * 2048  # includes the batch


def test_bmm_named_dim_map_picks_m_n(monkeypatch):
    """With work_div_loop_info present, M/N are identified by name (exact)."""
    b, m, n, kk = sympy.symbols("b m n kk", positive=True, integer=True)
    it_space = {b: 4, m: 1024, n: 1024, kk: 2048}
    _patch(monkeypatch, it_space, {b: 1, m: 16, n: 2, kk: 1})
    op = _FakeOp(
        1048576 * b + 1024 * m + n,
        1024 * b + m + kk,
        {"d0": 1, "d1": 16, "d2": 2, "d3": 1},
        wdli={b: ["B"], m: ["M"], n: ["N"], kk: ["K"]},
    )
    _, rows_per_core, cols_per_core, *_ = dcm._matmul_features(op, 4 * 1024 * 1024, 2)
    assert rows_per_core == 64 and cols_per_core == 512


def test_symbol_keyed_ownership_precedes_scheduler_transport(monkeypatch):
    """Pre-Scheduler reporting reads ownership, not lossy coefficient transport."""
    b, m, n, kk = sympy.symbols("b m n kk", positive=True, integer=True)
    it_space = {b: 4, m: 1024, n: 1024, kk: 2048}
    write_index = 1048576 * b + 1024 * m + n
    owned = {b: 1, m: 16, n: 2, kk: 1}
    _patch(monkeypatch, it_space, {b: 1, m: 1, n: 1, kk: 1})
    op = _FakeOp(write_index, 1024 * b + m + kk, {1: 1})
    op.iteration_space_ownership = SimpleNamespace(work_slices=owned)

    _, rows_per_core, cols_per_core, _, _, k_split, m_split, n_split = (
        dcm._matmul_features(op, 4 * 1024 * 1024, 2)
    )

    assert rows_per_core == 64 and cols_per_core == 512
    assert (m_split, n_split, k_split) == (16, 2, 1)


def test_plain_matmul_b1_unchanged(monkeypatch):
    """B=1 collapses the batch (2 output vars) -> decode is the plain-2D case, 8x4."""
    m, n, kk = sympy.symbols("m n kk", positive=True, integer=True)
    it_space = {m: 1024, n: 1024, kk: 2048}
    _patch(monkeypatch, it_space, {m: 8, n: 4, kk: 1})
    op = _FakeOp(1024 * m + n, m + kk, {"d0": 8, "d1": 4, "d2": 1})
    _, rows_per_core, cols_per_core, _, _, _, m_split, n_split = dcm._matmul_features(
        op, 1024 * 1024, 2
    )
    assert rows_per_core == 128 and cols_per_core == 256  # 1024/8, 1024/4
    assert (m_split, n_split) == (8, 4)


CACHE_ROWS, CACHE_COLS = 65536, 128  # buf15's [65536, 128] fp16 cache
CACHE_ELEMS = CACHE_ROWS * CACHE_COLS  # 8,388,608: what the store is charged now


def _isym(name):
    """The (integer, positive) assumptions real Inductor loop vars carry; without
    them sympy will not simplify a stick coordinate to a bare symbol."""
    return Symbol(name, integer=True, positive=True)


def _cache_fixed_layout():
    """The host-test pattern for a real Spyre layout, with no runtime device
    init: SpyreTensorLayout over the host shape, wrapped in FixedTiledLayout."""
    size = [CACHE_ROWS, CACHE_COLS]
    stride = [CACHE_COLS, 1]
    device_layout = SpyreTensorLayout(size, stride, torch.float16, [0, 1])
    return FixedTiledLayout(
        torch.device("cpu"), torch.float16, size, stride, device_layout
    )


def _cache_store_dep():
    """buf15's real store: index ``d1 + 128*tmp0`` over loops d0 = 512 and
    d1 = 128. The row loop d0 reaches the address ONLY through the runtime slot
    read from the index tensor, so it has no coefficient in the flat index."""
    d0, d1, tmp0 = _isym("d0"), _isym("d1"), _isym("tmp0")
    return MemoryDep("buf15", d1 + 128 * tmp0, (d0, d1), (512, 128))


def _cache_store_op(dep, slot_index=None):
    """A stand-in mutation op around real objects: real InputBuffer target, real
    mutation layout, real read/writes; only the op wrapper is hand-built."""
    op = mock.Mock()
    slot_expr = _isym("d0") if slot_index is None else slot_index

    def output_indexer(index):
        with V.set_graph_handler(SimpleNamespace(sizevars=SizeVarAllocator())):
            load_index = slot_expr.xreplace(dict(zip(dep.var_names, index)))
            return [
                V.ops.indirect_indexing(
                    V.ops.load("slots", load_index), CACHE_ROWS, check=False
                ),
                index[1],
            ]

    op.data = mock.Mock(spec=Scatter)
    op.data.ranges = list(dep.size)
    op.data.output_indexer = output_indexer
    op.dim_hints = []
    with V.set_graph_handler(mock.Mock()):
        op.get_layout.return_value = MutationLayoutSHOULDREMOVE(
            InputBuffer(name="buf15", layout=_cache_fixed_layout())
        )
    op.get_read_writes.return_value = ReadWrites(
        reads=OrderedSet(
            [
                MemoryDep(
                    "slots",
                    _isym("d0") if slot_index is None else slot_index,
                    dep.var_names,
                    dep.size,
                )
            ]
        ),
        writes=OrderedSet([dep]),
        index_exprs=OrderedSet(),
    )
    return op


class StickGeometryTest(unittest.TestCase):
    """Only a unit-stride stick variable admits per-row padding."""

    def test_unit_stride_is_admitted(self):
        d1 = _isym("d1")
        self.assertEqual(dcm._unit_stride_stick_var(sympy.Mod(d1, 64), 64), d1)
        self.assertEqual(dcm._unit_stride_stick_var(d1, 64), d1)

    def test_strided_and_offset_coordinates_are_not_admitted(self):
        d1 = _isym("d1")
        # Both pass the generic stick-expression helper -- Mod() with one free
        # symbol -- but their rows do not start at stick 0.
        self.assertIsNone(dcm._unit_stride_stick_var(sympy.Mod(3 * d1, 64), 64))
        self.assertIsNone(dcm._unit_stride_stick_var(sympy.Mod(d1 + 5, 64), 64))

    def test_a_constant_stick_coordinate_proves_no_variable(self):
        self.assertIsNone(dcm._unit_stride_stick_var(sympy.Integer(0), 64))


class StoredElemsTest(unittest.TestCase):
    """The loop nest -> elements stored, with the stick symbol in sticks."""

    def test_cache_write_counts_every_row(self):
        d0, d1 = _isym("d0"), _isym("d1")
        self.assertEqual(
            dcm._stored_elems({d0: 512, d1: 2}, {d1: 64}, CACHE_ELEMS), 65536
        )

    def test_partial_row_pads_per_row_not_per_flat_total(self):
        # 3 rows x 100 fp16: 3 x ceil(100/64) x 64 = 384, not flat ceil(300/64) x 64.
        d0, d1 = _isym("d0"), _isym("d1")
        self.assertEqual(dcm._stored_elems({d0: 3, d1: 2}, {d1: 64}, 1 << 20), 384)

    def test_symbolic_and_degenerate_nests_keep_the_committed_charge(self):
        d0, d1, n = _isym("d0"), _isym("d1"), _isym("n")
        self.assertIsNone(dcm._stored_elems({d0: n, d1: 2}, {d1: 64}, 1 << 20))
        self.assertIsNone(dcm._stored_elems({}, {}, 1 << 20))
        self.assertIsNone(dcm._stored_elems({d0: 512, d1: 2}, {d1: 64}, 65536))


class IndirectWriteElemsTest(unittest.TestCase):
    """The store's own geometry, through the real layout helpers."""

    def test_row_loop_through_an_indirect_slot_is_counted(self):
        dep = _cache_store_dep()
        d0, d1 = _isym("d0"), _isym("d1")
        self.assertTrue(dep.is_indirect())
        self.assertEqual(set(dep.ranges), {d0, d1})
        td = wd.TensorDep(dep, _cache_fixed_layout())
        adjusted, stick_vars = wd.adjust_it_space_for_sticks(dict(dep.ranges), [td])
        self.assertEqual(stick_vars, {d1: 64})
        self.assertEqual(adjusted[d1], 2)
        self.assertEqual(dcm._stored_elems(adjusted, stick_vars, CACHE_ELEMS), 65536)

    def test_mutation_store_is_narrowed_end_to_end(self):
        self.assertEqual(
            dcm._indirect_write_elems(_cache_store_op(_cache_store_dep()), CACHE_ELEMS),
            65536,
        )

    def test_modulo_does_not_hide_a_strided_store(self):
        dep = _cache_store_dep()
        d1 = _isym("d1")
        strided = MemoryDep(dep.name, dep.index + 64 * d1, dep.var_names, dep.size)
        self.assertIsNone(
            dcm._indirect_write_elems(_cache_store_op(strided), CACHE_ELEMS)
        )

    def test_column_dependent_or_unknown_slots_keep_the_committed_charge(self):
        dep = _cache_store_dep()
        op = _cache_store_op(dep, 128 * _isym("d0") + _isym("d1"))
        self.assertIsNone(dcm._indirect_write_elems(op, CACHE_ELEMS))
        op.get_read_writes.return_value.reads.add(
            MemoryDep("slots", _isym("d0"), dep.var_names, dep.size)
        )
        self.assertIsNone(dcm._indirect_write_elems(op, CACHE_ELEMS))
        op.data.output_indexer = lambda x: x
        self.assertIsNone(dcm._indirect_write_elems(op, CACHE_ELEMS))


class SymbolicPriceTest(unittest.TestCase):
    """Indirect buffers are never resident, so symbolic and HBM prices agree."""

    def test_symbolic_and_concrete_prices_agree_at_hbm(self):
        sym_is_lx = Symbol("is_lx_buf15", integer=True, nonnegative=True)

        def store(is_lx):
            return ArgTraffic(
                name="buf15",
                role="output",
                is_lx=is_lx,
                elems=65536,
                dims=[CACHE_ROWS, CACHE_COLS],
                loop_factor=1,
                is_boundary=False,
            )

        symbolic = store(sym_is_lx).hbm_elems()
        self.assertFalse(isinstance(symbolic, int))  # genuinely symbolic, not cast
        self.assertEqual(int(symbolic.subs(sym_is_lx, 0)), store(False).hbm_elems())
        self.assertEqual(store(False).hbm_elems(), 65536)


def test_macs_are_one_output_pass_times_the_writes_loop_factor(monkeypatch):
    """``matmul_macs`` is the work of the WHOLE counted loop.

    ``out_elems * K`` is one pass over the output buffer; the extractor passes the
    write's own loop factor (how many times that buffer is produced over the nest),
    and the work is the product. Which factor each loop shape gets is tested through
    the extractor below.
    """
    m, n, kk = sympy.symbols("m n kk", positive=True, integer=True)
    it_space = {m: 64, n: 1024, kk: 2048}
    _patch(monkeypatch, it_space, {m: 1, n: 1, kk: 1})
    op = _FakeOp(1024 * m + n, m + kk, {"d0": 1, "d1": 1, "d2": 1})
    out_elems = 64 * 1024
    per_pass = out_elems * 2048
    assert dcm._matmul_features(op, out_elems, 2)[0] == per_pass  # single pass
    assert dcm._matmul_features(op, out_elems, 2, 128)[0] == 128 * per_pass


def test_the_extractor_scales_a_per_trip_body_matmul_by_its_trip(monkeypatch):
    """Wiring: ``extract_op_features`` passes the loop's tiling to ``_matmul_features``.

    Without it a per-trip body matmul reports one trip of work while its traffic is
    charged ``loop_trip`` times.
    """
    from torch_spyre._inductor.constants import BATCH_MATMUL_OP

    m, n, kk = sympy.symbols("m n kk", positive=True, integer=True)
    _patch(monkeypatch, {m: 64, n: 128, kk: 64}, {m: 1, n: 1, kk: 1})
    monkeypatch.setattr(dcm, "_indirect_write_elems", lambda *_: None)
    op = _FakeOp(128 * m + n, m + kk, {"d0": 1, "d1": 1, "d2": 1})
    op.data = SimpleNamespace(
        reduction_ranges=[64], reduction_type=BATCH_MATMUL_OP, ranges=[64, 128]
    )
    layout = SimpleNamespace(allocation=None, device_layout=None)
    op.name = "buf1"
    op.dim_hints = []
    op.get_operation_name = lambda: "op_buf1"
    op.get_layout = lambda: layout
    op.get_dtype = lambda: SimpleNamespace(itemsize=2)
    op.get_size = lambda: [64, 128]
    graph = SimpleNamespace(
        graph_input_names=[],
        get_output_names=lambda: [],
        get_buffer=lambda name: None,
    )

    def macs(loop_info):
        op.loop_info = loop_info
        with V.set_graph_handler(graph):
            return dcm.extract_op_features(op).matmul_macs

    def loop(tiled_out):
        return SimpleNamespace(
            loop_count=[16],
            loop_tiled_dims=[tiled_out],
            loop_tiled_reduction_dims=[[]],
        )

    assert macs(loop([])) == 16 * 64 * 128 * 64  # per-trip body op
    assert macs(loop([0])) == 64 * 128 * 64  # output-tiled: already the total


# Reads advancing through a counted loop visit each tile once. A stationary read
# revisits its source each trip; nested loop variables must stay with their levels.
u0, u1, d0, d1, d2 = sympy.symbols("u0 u1 d0 d1 d2", integer=True)


def _hint(var, trip):
    """Stand-in for ``propagate_hints.DimHint``: only the two fields read here."""
    return SimpleNamespace(dim_names=[], loop_var=var, loop_var_range=trip)


def _op(*hints):
    return SimpleNamespace(dim_hints=list(hints))


# ------------------------------------------------------------- level pairing


def _factor(index, levels, op, stamped=None):
    """The read/write factor the extractor computes for ``index``."""
    loop_vars = _level_loop_vars(op, levels)
    return _loop_factor_for_index(
        index, levels, _loop_var_advances(index, loop_vars, stamped)
    )


def test_an_advancing_read_is_not_multiplied_by_the_trip_count():
    # One level of 128 trips; the op tiles none of its own dims (the measured case).
    levels = [(128, set(), 0)]
    advancing = 704 * d2 + 1982464 * u0
    invariant = 2816 * d0 + d2
    op = _op(_hint(u0, sympy.Integer(128)))
    assert _factor(advancing, levels, op) == 1
    assert _factor(invariant, levels, op) == 128
    # Without the loop variable the advancing read is charged the trip count.
    assert _loop_factor_for_index(advancing, levels) == 128


def test_a_variable_whose_range_is_not_the_levels_trip_is_not_paired():
    levels = [(128, set(), 0)]
    assert _level_loop_vars(_op(_hint(u0, sympy.Integer(4))), levels) == [None]


def test_hints_without_a_range_are_ignored():
    # An ordinary spyre_hint scope variable is a real iteration-range variable (already
    # in dep.ranges); only for_each_tile variables carry loop_var_range.
    levels = [(128, set(), 0)]
    assert _level_loop_vars(_op(_hint(u0, None)), levels) == [None]
    assert _level_loop_vars(SimpleNamespace(), levels) == [None]


def test_a_hint_count_that_differs_from_the_level_count_pairs_nothing():
    # Two levels but one loop variable: the pairing is not known, so keep the price
    # every other op gets, and say so in the debug log.
    levels = [(4, set(), 0), (4, set(), 0)]
    with mock.patch.object(dcm.logger, "debug") as debug:
        assert _level_loop_vars(_op(_hint(u0, 4)), levels) == [None, None]
    debug.assert_called_once()
    assert _factor(8 * u0 + d0, levels, _op(_hint(u0, 4))) == 16


def test_nested_loops_of_equal_trip_count_pair_each_variable_with_its_own_level():
    """Two nested loops of 4 trips: ``u0`` is the outer variable, ``u1`` the inner one.

    Pairing by trip count alone gave both levels both variables, so an index that
    carried only ``u0`` (advancing with the outer loop, re-entered by the inner one)
    was charged 1 instead of 4.
    """
    levels = [(4, set(), 0), (4, set(), 0)]
    op = _op(_hint(u0, 4), _hint(u1, 4))
    assert _level_loop_vars(op, levels) == [u0, u1]
    assert _factor(8 * u0 + d0, levels, op) == 4  # walked outer, re-read inner
    assert _factor(8 * u1 + d0, levels, op) == 4  # re-read outer, walked inner
    assert _factor(8 * u0 + 2 * u1, levels, op) == 1  # walked at both
    assert _factor(d0, levels, op) == 16  # re-entered at both


def test_nested_loops_of_different_trip_count_keep_their_own_variables():
    levels = [(2, set(), 0), (4, set(), 0)]
    op = _op(_hint(u0, 2), _hint(u1, 4))
    assert _factor(8 * u0, levels, op) == 4
    assert _factor(8 * u1, levels, op) == 2


# ----------------------------------------------- pinned reads (lowering's rule)


def test_a_loop_variable_without_a_coefficient_does_not_advance_the_read():
    """Synthetic stamp-consistency check, not an observed lowered-kernel case.

    The lowering advances a dependency only when its index has a nonzero
    coefficient on the loop variable (``_stamp_direct_loop_info``). ``u0`` being a
    free symbol is not enough: the synthetic index ``4096*FloorDiv(u0, 2)`` has
    coefficient 0, so under that rule it would be stamped pinned, and the price
    follows the same rule (re-entered every trip). No lowered kernel is known to
    produce this read."""
    from torch.utils._sympy.functions import FloorDiv

    levels = [(8, set(), 0)]
    op = _op(_hint(u0, 8))
    pinned = 4096 * FloorDiv(u0, 2) + d2
    assert _loop_var_advances(pinned, [u0]) == [False]
    assert _factor(pinned, levels, op) == 8
    assert _factor(4096 * u0 + d2, levels, op) == 1  # coefficient 4096: advances


def test_the_lowerings_stamp_decides_over_the_index():
    """Synthetic stamp-consistency check, not an observed lowered-kernel case.

    When the lowering's stamp and the index disagree, the stamp wins: code
    generation follows the stamp, so the price follows it too. The read below keeps
    ``u0`` in its index but carries an empty ("pinned") stamp. It is constructed for
    the check; no reachable read with ``u0`` in its index and a pinned stamp is
    known."""
    levels = [(8, set(), 0)]
    op = _op(_hint(u0, 8))
    pool_read = 64 * u0 + d2
    pinned_stamp = _stamped_advances([[]], None, 1)
    advancing_stamp = _stamped_advances([[(0, sympy.Integer(1))]], None, 1)
    squeezed_stamp = _stamped_advances([[]], [[(sympy.Integer(64), 1)]], 1)
    assert (pinned_stamp, advancing_stamp, squeezed_stamp) == ([False], [True], [True])
    assert _factor(pool_read, levels, op, pinned_stamp) == 8
    assert _factor(d2, levels, op, advancing_stamp) == 1
    # A stamp that does not cover every level is not used.
    assert _stamped_advances([[], []], None, 1) is None
    assert _stamped_advances([[]], [[], []], 1) is None


# ------------------------------------------------ wiring in extract_op_features


class _StubGraph:
    graph_input_names: list = []

    def get_output_names(self):
        return []

    def get_buffer(self, name):
        return None


def _looped_op(trips, tiled_out, hints, write_index, read_indices, data=None):
    """An op the real extractor can walk, carrying a ``for_each_tile`` loop nest."""
    layout = SimpleNamespace(allocation=None, device_layout=None)
    rw = SimpleNamespace(
        reads=[
            SimpleNamespace(name=f"arg{i}", index=index)
            for i, index in enumerate(read_indices)
        ],
        writes=[SimpleNamespace(index=write_index)],
    )
    return SimpleNamespace(
        name="buf1",
        data=data,
        dim_hints=list(hints),
        loop_info=SimpleNamespace(
            loop_count=list(trips),
            loop_tiled_dims=[list(level) for level in tiled_out],
            loop_tiled_reduction_dims=[[] for _ in trips],
        ),
        get_name=lambda: "buf1",
        get_operation_name=lambda: "op_buf1",
        get_layout=lambda: layout,
        get_dtype=lambda: SimpleNamespace(itemsize=2),
        get_size=lambda: [64],
        get_read_writes=lambda: rw,
    )


def _features(monkeypatch, op, it_space, graph=None, **kwargs):
    from torch._inductor.virtualized import V

    monkeypatch.setattr(dcm, "iteration_space_from_op", lambda _op: it_space)
    monkeypatch.setattr(dcm, "_indirect_write_elems", lambda *_: None)
    with V.set_graph_handler(graph or _StubGraph()):
        return dcm.extract_op_features(op, **kwargs)


def _factors(monkeypatch, op, it_space):
    return {a.name: a.loop_factor for a in _features(monkeypatch, op, it_space).args}


@pytest.mark.parametrize("trips", [None, 1, 8])
def test_operand_geometry_preserves_the_single_pass_read_estimate(monkeypatch, trips):
    """Only looped matmuls may replace the existing read-run estimate."""
    from torch_spyre._inductor.constants import BATCH_MATMUL_OP

    op = _looped_op(
        [] if trips is None else [trips],
        [] if trips is None else [[]],
        [] if trips is None else [_hint(u0, sympy.Integer(trips))],
        write_index=64 * d0 + d1,
        read_indices=[64 * d0 + d2],
        data=SimpleNamespace(
            ranges=[64, 64], reduction_ranges=[64], reduction_type=BATCH_MATMUL_OP
        ),
    )
    # The operand-specific proof can find a different run than the general one.
    monkeypatch.setattr(dcm, "_operand_read_geometry", lambda *_: (128, 4096))
    monkeypatch.setattr(dcm, "_read_run_bytes", lambda *_: 256)
    read = _read(_features(monkeypatch, op, {d0: 64, d1: 64, d2: 64}), "arg0")
    expected = (128, 4096) if trips == 8 else (256, None)
    assert (read.read_run_bytes, read.read_tile_elems) == expected


def test_the_extractor_walks_an_expert_bank_read_once(monkeypatch):
    """The bug itself: a per-expert body op whose bank read advances with the expert
    loop was priced 128 reads of the bank.  Removing the fold in the extractor fails
    this test."""
    op = _looped_op(
        [128],
        [[]],  # the op tiles none of its own dims
        [_hint(u0, sympy.Integer(128))],
        write_index=64 * d0 + d2,
        read_indices=[704 * d2 + 1982464 * u0, 64 * d0 + d2],
    )
    factors = _factors(monkeypatch, op, {d0: 64, d2: 704})
    assert factors["arg0"] == 1  # the bank read: walked once across the expert loop
    assert factors["arg1"] == 128  # the activation: re-entered every trip
    assert factors["op_buf1"] == 128  # the per-trip output buffer


def test_the_extractor_walks_a_kv_page_read_of_a_page_loop_once(monkeypatch):
    """Not a mixture-of-experts shape: an attention step over one KV page per trip.
    The page read advances with the page loop, the query is re-entered."""
    op = _looped_op(
        [8],
        [[]],
        [_hint(u0, sympy.Integer(8))],
        write_index=64 * d0 + d1,
        read_indices=[64 * d1 + 4096 * u0 + d2, 64 * d0 + d2],
    )
    factors = _factors(monkeypatch, op, {d0: 64, d1: 64, d2: 64})
    assert factors["arg0"] == 1
    assert factors["arg1"] == 8
    assert factors["op_buf1"] == 8


def test_the_extractor_leaves_a_row_tiled_loop_unchanged(monkeypatch):
    """A row-tiled loop (the op tiles its own row dim d0) already names its variable
    in the level's symbols; the fold adds nothing and the invariant read keeps the
    trip count."""
    op = _looped_op(
        [8],
        [[0]],
        [_hint(u0, sympy.Integer(8))],
        write_index=64 * d0 + d1,
        read_indices=[64 * d0 + d2, 64 * d1 + d2],
        data=SimpleNamespace(
            ranges=[64, 64], reduction_ranges=[64], reduction_type=None
        ),
    )
    factors = _factors(monkeypatch, op, {d0: 64, d1: 64, d2: 64})
    assert factors == {"op_buf1": 1, "arg0": 1, "arg1": 8}


def test_the_extractor_pairs_nested_equal_trip_loops_by_level(monkeypatch):
    op = _looped_op(
        [4, 4],
        [[], []],
        [_hint(u0, 4), _hint(u1, 4)],
        write_index=d0,
        read_indices=[8 * u0 + d0, 8 * u1 + d0, 8 * u0 + 2 * u1 + d0],
    )
    factors = _factors(monkeypatch, op, {d0: 64})
    assert factors["arg0"] == 4  # advances with the outer loop only
    assert factors["arg1"] == 4  # advances with the inner loop only
    assert factors["arg2"] == 1  # advances with both


# ---------------------------------------- matmul work = one pass x the write's factor


def _looped_matmul_features(
    monkeypatch,
    *,
    size,
    k,
    trips,
    tiled_out,
    tiled_red=None,
    hints=(),
    write_index,
    read_indices,
    it_space,
    work_slices=None,
    ranges=None,
    output_tiled_dims=None,
):
    """``extract_op_features`` on a batch-matmul body op inside a loop nest."""
    from torch_spyre._inductor.constants import BATCH_MATMUL_OP

    data = SimpleNamespace(
        ranges=list(size[-2:] if ranges is None else ranges),
        reduction_ranges=[k],
        reduction_type=BATCH_MATMUL_OP,
    )
    op = _looped_op(trips, tiled_out, hints, write_index, read_indices, data=data)
    op.get_size = lambda: list(size)
    if tiled_red is not None:
        op.loop_info.loop_tiled_reduction_dims = [list(lv) for lv in tiled_red]
    if output_tiled_dims is not None:
        op.loop_info.output_tiled_dims = output_tiled_dims
    monkeypatch.setattr(dcm, "iteration_space_from_op", lambda _op: it_space)
    monkeypatch.setattr(dcm, "_indirect_write_elems", lambda *_: None)
    with V.set_graph_handler(_StubGraph()):
        return dcm.extract_op_features(op, work_slices)


def _output_factor(feature):
    return next(a.loop_factor for a in feature.args if a.role == "output")


def test_a_stacked_slice_write_is_not_multiplied_by_the_trip_count(monkeypatch):
    """Review repro (E=128, T=64, N=128, K=64; the loop tiles none of the op's dims).

    A per-expert body matmul that re-writes one ``[T, N]`` buffer each trip produces
    it 128 times. One that writes its own slice ``out[u0, m, n]`` of a stacked
    ``[E, T, N]`` buffer produces that buffer once: ``out_elems`` already covers every
    trip. Both are 128 experts of ``T*N*K`` work; scaling the stacked write by the trip
    count as well gave 8,589,934,592.

    The stacked write is synthetic: the test checks that the factor follows the
    write's index, not that a lowered body ``batchmatmul`` reaches this
    squeezed-advance write (not established).
    """
    E, T, N, K = 128, 64, 128, 64
    m, n, r0 = sympy.symbols("m n r0", integer=True)
    common = dict(
        k=K,
        trips=[E],
        tiled_out=[[]],
        hints=[_hint(u0, sympy.Integer(E))],
        read_indices=[K * m + r0, N * K * u0 + N * r0 + n],
        it_space={m: T, n: N, r0: K},
    )
    per_trip = _looped_matmul_features(
        monkeypatch, size=[T, N], write_index=N * m + n, **common
    )
    stacked = _looped_matmul_features(
        monkeypatch, size=[E, T, N], write_index=T * N * u0 + N * m + n, **common
    )
    assert (_output_factor(per_trip), per_trip.matmul_macs) == (E, 67_108_864)
    assert (_output_factor(stacked), stacked.matmul_macs) == (1, 67_108_864)


def test_macs_of_a_paged_attention_step_and_a_row_loop(monkeypatch):
    """Non-expert ``for_each_tile`` shapes.

    * A page loop: one attention score matmul per KV page (64 query rows x 128 page
      tokens x head dim 64), 16 trips, re-writing one score buffer -- 16 pages of work.
    * A row loop over the same op's query rows: the loop tiles an output dim, the
      output buffer is full-extent and the raw product already is the total.
    """
    m, n, r0 = sympy.symbols("m n r0", integer=True)
    common = dict(
        size=[64, 128],
        k=64,
        write_index=128 * m + n,
        read_indices=[64 * m + r0, 64 * n + r0],
        it_space={m: 64, n: 128, r0: 64},
    )
    page_loop = _looped_matmul_features(
        monkeypatch, trips=[16], tiled_out=[[]], hints=[_hint(u0, 16)], **common
    )
    row_loop = _looped_matmul_features(
        monkeypatch, trips=[8], tiled_out=[[0]], hints=[_hint(u0, 8)], **common
    )
    assert page_loop.matmul_macs == 16 * 64 * 128 * 64
    assert row_loop.matmul_macs == 64 * 128 * 64


def test_a_reduction_tiled_matmul_keeps_its_trip_scaling(monkeypatch):
    """A coarse loop over K: the write has no reduction variable, so the same output
    tile is produced every trip with ``K / trips`` of the reduction; the work of the
    nest is ``trips`` times the per-tile product (unchanged convention)."""
    m, n, r0 = sympy.symbols("m n r0", integer=True)
    feature = _looped_matmul_features(
        monkeypatch,
        size=[64, 128],
        k=16,  # the per-tile K slice of a K=64 reduction in 4 trips
        trips=[4],
        tiled_out=[[]],
        tiled_red=[[0]],
        write_index=128 * m + n,
        read_indices=[64 * m + r0, 128 * r0 + n],
        it_space={m: 64, n: 128, r0: 16},
    )
    assert _output_factor(feature) == 4
    assert feature.matmul_macs == 64 * 128 * 64


def test_a_mixed_nest_scales_only_the_level_that_re_writes(monkeypatch):
    """An outer row loop (tiles the output row dim, 2 trips) around an inner
    ``for_each_tile`` loop that tiles none of the op's dims (4 trips). The output is
    walked by the outer level and re-written by the inner one: 1 * 4 passes. The old
    all-or-nothing rule saw "tiles an output dim" and did not scale at all."""
    m, n, r0 = sympy.symbols("m n r0", integer=True)
    feature = _looped_matmul_features(
        monkeypatch,
        size=[64, 128],
        k=64,
        trips=[2, 4],
        tiled_out=[[0], []],
        hints=[_hint(u0, 2), _hint(u1, 4)],
        write_index=128 * m + n,
        read_indices=[64 * m + r0, 8192 * u1 + 128 * r0 + n],
        it_space={m: 64, n: 128, r0: 64},
    )
    assert _output_factor(feature) == 4
    assert feature.matmul_macs == 4 * 64 * 128 * 64
    factors = {a.name: a.loop_factor for a in feature.args}
    assert factors["arg0"] == 4  # walked by the row loop, re-read by the inner loop
    assert factors["arg1"] == 2  # re-read by the row loop, walked by the inner loop


# ------------------- the same work on both matmul models (allocator and report)
#
# The scratchpad allocator prices with ``use_bundled_cost_model=False``: the UPSTREAM
# model rebuilds per-trip M/N/K axes from the features and calls
# ``work_division._matmul_execution_cost``. The report (``cost_model_pass``) uses the
# default BUNDLED model, which reads ``matmul_macs``. Both must charge the work of
# the whole loop, and neither may multiply a buffer that already spans the trips.


def _allocator_compute_ns(feature):
    from torch_spyre._inductor import cost_model
    from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS

    assert _COST_PARAMS.use_bundled_cost_model is False
    return cost_model._matmul_ns_upstream([feature], _COST_PARAMS)


def _report_compute_ns(feature):
    from torch_spyre._inductor import cost_model

    params = cost_model.CostParams()
    assert params.use_bundled_cost_model is True
    return cost_model._matmul_ns_bundled([feature], params)


def _passes(feature):
    from torch_spyre._inductor import cost_model

    k_axis = cost_model._matmul_axes_for_split_cost(feature)[3]
    return cost_model._matmul_passes(feature, k_axis[0])


def _expert_matmuls(monkeypatch, trips):
    """One expert projection (T=64, N=128, K=64) per trip of a ``trips``-trip loop:
    re-writing one ``[T, N]`` buffer each trip (stationary output), and writing its
    own slice of a stacked ``[trips, T, N]`` buffer."""
    T, N, K = 64, 128, 64
    m, n, r0 = sympy.symbols("m n r0", integer=True)
    common = dict(
        k=K,
        trips=[trips],
        tiled_out=[[]],
        hints=[_hint(u0, sympy.Integer(trips))],
        read_indices=[K * m + r0, N * K * u0 + N * r0 + n],
        it_space={m: T, n: N, r0: K},
        work_slices={m: 4, n: 8, r0: 1},
    )
    stationary = _looped_matmul_features(
        monkeypatch, size=[T, N], write_index=N * m + n, **common
    )
    stacked = _looped_matmul_features(
        monkeypatch, size=[trips, T, N], write_index=T * N * u0 + N * m + n, **common
    )
    return stationary, stacked


def test_loop_passes_do_not_multiply_the_batch_split_penalty_twice(monkeypatch):
    """The same physical BMM batch split costs once per invocation in either form."""
    from torch_spyre._inductor import cost_model as cm
    from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS

    E, B, M, N, K = 8, 2, 512, 128, 64
    b, m, n, r0 = sympy.symbols("b m n r0", integer=True)
    common = dict(
        k=K,
        trips=[E],
        tiled_out=[[]],
        hints=[_hint(u0, sympy.Integer(E))],
        ranges=[B, M, N],
        read_indices=[M * K * b + K * m + r0, B * K * N * u0 + K * N * b + N * r0 + n],
        it_space={b: B, m: M, n: N, r0: K},
        work_slices={b: 2, m: 2, n: 8, r0: 1},
    )
    body = _looped_matmul_features(
        monkeypatch, size=[B, M, N], write_index=M * N * b + N * m + n, **common
    )
    stacked = _looped_matmul_features(
        monkeypatch,
        size=[E, B, M, N],
        write_index=B * M * N * u0 + M * N * b + N * m + n,
        **common,
    )
    assert (_passes(body), _passes(stacked)) == (E, 1)
    expected = E * _COST_PARAMS.mm_batch_split_ns_per_step
    assert expected > 0
    without_batch = dataclasses.replace(_COST_PARAMS, mm_batch_split_ns_per_step=0)
    for feature in (body, stacked):
        assert cm._matmul_batch_split_ns([feature], _COST_PARAMS) == expected
        assert _allocator_compute_ns(feature) - cm._matmul_ns_upstream(
            [feature], without_batch
        ) == pytest.approx(expected)
    assert _allocator_compute_ns(body) == pytest.approx(_allocator_compute_ns(stacked))


def test_both_matmul_models_charge_every_trip_of_a_stationary_output_loop(
    monkeypatch,
):
    """Review of #4996 (cyang49): the allocator's model ignored the corrected work.

    A 128-expert loop whose body matmul re-writes one ``[T, N]`` buffer each trip
    does 128 experts of work. Before, the allocator's model priced one expert: it
    rebuilds ``B = out_elems / (M * N) = 1`` and never read ``matmul_macs``."""
    one, _ = _expert_matmuls(monkeypatch, 1)
    stationary, _ = _expert_matmuls(monkeypatch, 128)
    assert _passes(stationary) == 128
    assert _allocator_compute_ns(stationary) == pytest.approx(
        128 * _allocator_compute_ns(one), rel=1e-12
    )
    assert _report_compute_ns(stationary) == pytest.approx(
        128 * _report_compute_ns(one), rel=1e-12
    )


def test_both_matmul_models_charge_a_stacked_output_loop_once(monkeypatch):
    """The stacked write ``out[u0, m, n]`` into ``[E, T, N]``: ``B`` already spans the
    128 trips, so neither model multiplies again. It is the same 128 experts of work
    as the stationary-output loop, on both models."""
    one, _ = _expert_matmuls(monkeypatch, 1)
    stationary, stacked = _expert_matmuls(monkeypatch, 128)
    assert _passes(stacked) == 1
    assert _allocator_compute_ns(stacked) == pytest.approx(
        _allocator_compute_ns(stationary), rel=1e-12
    )
    assert _allocator_compute_ns(stacked) == pytest.approx(
        128 * _allocator_compute_ns(one), rel=1e-12
    )
    assert _report_compute_ns(stacked) == pytest.approx(
        _report_compute_ns(stationary), rel=1e-12
    )


def test_both_matmul_models_charge_a_page_loop_and_a_reduction_tiled_loop_per_trip(
    monkeypatch,
):
    """Non-expert shapes. An attention-score step over one KV page per trip (16 pages)
    re-writes its score buffer; a coarse loop over K revisits one output with each K
    slice (4 trips of K/4). Each pass is one trip's matmul, so both are charged 16 and
    4 passes; a row loop walks its output once and keeps one pass."""
    m, n, r0 = sympy.symbols("m n r0", integer=True)
    scores = dict(
        size=[64, 128],
        k=64,
        write_index=128 * m + n,
        read_indices=[64 * m + r0, 64 * n + r0],
        it_space={m: 64, n: 128, r0: 64},
        work_slices={m: 4, n: 8, r0: 1},
    )
    page_one = _looped_matmul_features(
        monkeypatch, trips=[1], tiled_out=[[]], hints=[_hint(u0, 1)], **scores
    )
    page_loop = _looped_matmul_features(
        monkeypatch, trips=[16], tiled_out=[[]], hints=[_hint(u0, 16)], **scores
    )
    row_loop = _looped_matmul_features(
        monkeypatch, trips=[8], tiled_out=[[0]], hints=[_hint(u0, 8)], **scores
    )
    assert _passes(page_loop) == 16
    assert _allocator_compute_ns(page_loop) == pytest.approx(
        16 * _allocator_compute_ns(page_one), rel=1e-12
    )
    assert _passes(row_loop) == 1
    assert _allocator_compute_ns(row_loop) == pytest.approx(
        _allocator_compute_ns(page_one), rel=1e-12
    )

    k_slices = dict(
        size=[64, 128],
        k=16,
        write_index=128 * m + n,
        read_indices=[64 * m + r0, 128 * r0 + n],
        it_space={m: 64, n: 128, r0: 16},
        work_slices={m: 4, n: 8, r0: 1},
    )
    k_one = _looped_matmul_features(
        monkeypatch, trips=[1], tiled_out=[[]], tiled_red=[[0]], **k_slices
    )
    k_loop = _looped_matmul_features(
        monkeypatch, trips=[4], tiled_out=[[]], tiled_red=[[0]], **k_slices
    )
    assert _passes(k_loop) == 4
    assert _allocator_compute_ns(k_loop) == pytest.approx(
        4 * _allocator_compute_ns(k_one), rel=1e-12
    )


def test_both_matmul_models_charge_a_mixed_nest_level_by_level(monkeypatch):
    """Outer row loop (walks the output, 2 trips) around an inner ``for_each_tile``
    loop that re-writes it (4 trips): 1 x 4 passes on both models."""
    m, n, r0 = sympy.symbols("m n r0", integer=True)
    common = dict(
        size=[64, 128],
        k=64,
        write_index=128 * m + n,
        read_indices=[64 * m + r0, 8192 * u1 + 128 * r0 + n],
        it_space={m: 64, n: 128, r0: 64},
        work_slices={m: 4, n: 8, r0: 1},
    )
    single = _looped_matmul_features(
        monkeypatch,
        trips=[2, 1],
        tiled_out=[[0], []],
        hints=[_hint(u0, 2), _hint(u1, 1)],
        **common,
    )
    nest = _looped_matmul_features(
        monkeypatch,
        trips=[2, 4],
        tiled_out=[[0], []],
        hints=[_hint(u0, 2), _hint(u1, 4)],
        **common,
    )
    assert (_passes(single), _passes(nest)) == (1, 4)
    assert _allocator_compute_ns(nest) == pytest.approx(
        4 * _allocator_compute_ns(single), rel=1e-12
    )
    assert _report_compute_ns(nest) == pytest.approx(
        4 * _report_compute_ns(single), rel=1e-12
    )


def test_a_record_without_a_loop_keeps_its_allocator_price(monkeypatch):
    """A single-pass record, and a record whose MAC count is missing, are priced
    exactly as before: one call of the execution model with the rebuilt axes."""
    import dataclasses

    from torch_spyre._inductor import cost_model

    one, _ = _expert_matmuls(monkeypatch, 1)
    axes = cost_model._matmul_axes_for_split_cost(one)
    b_axis, m_axis, n_axis, k_axis, shared = axes
    direct = 1000.0 * wd._matmul_execution_cost(
        b_axis,
        m_axis,
        n_axis,
        k_axis,
        cost_model.config.sencores,
        shared_weight=shared,
        include_hbm=False,
    )
    assert _allocator_compute_ns(one) == pytest.approx(direct, rel=1e-12)
    no_macs = dataclasses.replace(one, matmul_macs=0)
    assert _allocator_compute_ns(no_macs) == pytest.approx(direct, rel=1e-12)


def test_the_extractor_follows_a_pinned_stamp_on_a_read_carrying_the_loop_variable(
    monkeypatch,
):
    """Synthetic stamp-consistency check, not an observed lowered-kernel case.

    The lowering's per-read stamp, when it covers every read, is what the price
    follows: here the second read keeps ``u0`` in its index but is stamped pinned,
    so it is priced as re-read every trip. The read is constructed for the check;
    no reachable read of this shape is known."""
    op = _looped_op(
        [8],
        [[]],
        [_hint(u0, sympy.Integer(8))],
        write_index=64 * d0 + d2,
        read_indices=[4096 * u0 + d2, 64 * u0 + d2],
    )
    op.loop_info.tiled_dims_per_read = [[[(0, sympy.Integer(1))]], [[]]]
    op.loop_info.squeezed_advance_per_read = []
    factors = _factors(monkeypatch, op, {d0: 64, d2: 64})
    assert factors["arg0"] == 1  # stamped advancing
    assert factors["arg1"] == 8  # stamped pinned despite u0 in its index


def test_the_extractor_marks_the_reads_that_advance_with_the_loop_variable(monkeypatch):
    """``advances_with_loop_var`` is what confines the partitioned-read price to the
    for_each_tile loop's tiled operand: set for a read whose index carries the loop
    variable, clear for the re-entered read and for the output."""
    op = _looped_op(
        [128],
        [[]],
        [_hint(u0, sympy.Integer(128))],
        write_index=64 * d0 + d2,
        read_indices=[704 * d2 + 1982464 * u0, 64 * d0 + d2],
    )
    feature = _features(monkeypatch, op, {d0: 64, d2: 704})
    advances = {a.name: a.advances_with_loop_var for a in feature.args}
    assert advances["arg0"] is True
    assert advances["arg1"] is False
    assert advances["op_buf1"] is False


def test_partitionability_requires_a_legal_menu_split_of_the_read_index():
    index = 64 * d1 + d2
    assert not dcm._has_partitioning_candidate(index, None)
    assert not dcm._has_partitioning_candidate(index, [])
    assert not dcm._has_partitioning_candidate(index, [{d0: 32, d1: 1}])
    assert not dcm._has_partitioning_candidate(index, [{d1: sympy.Symbol("split")}])
    assert not dcm._has_partitioning_candidate(None, [{d1: 2}])
    assert dcm._has_partitioning_candidate(index, [{d0: 32}, {d1: 2}])
    assert dcm._has_partitioning_candidate(index, [{d2: sympy.Integer(4)}])


def test_the_extractor_uses_the_same_menu_for_every_priced_candidate(monkeypatch):
    op = _looped_op(
        [128],
        [[]],
        [_hint(u0, sympy.Integer(128))],
        write_index=64 * d0 + d2,
        read_indices=[704 * d2 + 1982464 * u0],
    )
    # One candidate only replicates this read; the other partitions it. Both
    # retain the estimate because the legal menu offers a partitioning choice.
    menu = [{d0: 16, d2: 1}, {d0: 8, d2: 2}]
    for chosen in menu:
        features = _features(
            monkeypatch,
            op,
            {d0: 32, d2: 128},
            work_slices=chosen,
            candidate_work_slices=menu,
        )
        assert next(
            a for a in features.args if a.name == "arg0"
        ).has_partitioning_candidate

    # The shape and bytes are unchanged. Remove the partitioning option and
    # applicability declines; this is not a byte-size cutoff or a chosen-split test.
    for menu_without_partition in (None, [{d0: 8, d2: 1}, {d0: 16, d2: 1}]):
        features = _features(
            monkeypatch,
            op,
            {d0: 32, d2: 128},
            work_slices={d0: 16, d2: 1},
            candidate_work_slices=menu_without_partition,
        )
        assert not any(a.has_partitioning_candidate for a in features.args)


def test_an_op_without_a_for_each_tile_variable_marks_no_read_as_advancing(monkeypatch):
    """Coarse tiling stamps a loop but no loop variable on the op's hints, so no read
    is marked, whatever its ``loop_factor``."""
    op = _looped_op(
        [4],
        [[0]],
        [],
        write_index=64 * d0 + d1,
        read_indices=[64 * d0 + d2, 64 * d1 + d2],
        data=SimpleNamespace(
            ranges=[64, 64], reduction_ranges=[64], reduction_type=None
        ),
    )
    feature = _features(monkeypatch, op, {d0: 64, d1: 64, d2: 64})
    assert not any(a.advances_with_loop_var for a in feature.args)


# ------------------------- loop-delivery eligibility: one per-read verdict
#
# ``advances_with_loop_var`` gates the loop-delivery term (with ``loop_factor == 1``
# and a partitioning candidate). It follows the same per-read verdict as
# ``loop_factor``: the lowering's stamp, else a nonzero coefficient on the level's
# variable. Each test asserts the flag and the factor together, since only their
# combination prices.


def _read(feature, name):
    return next(a for a in feature.args if a.name == name)


def test_a_stamped_pinned_read_carrying_the_loop_variable_is_not_eligible(
    monkeypatch,
):
    op = _looped_op(
        [8],
        [[]],
        [_hint(u0, sympy.Integer(8))],
        write_index=64 * d0 + d2,
        read_indices=[4096 * u0 + d2],
    )
    op.loop_info.tiled_dims_per_read = [[[]]]
    op.loop_info.squeezed_advance_per_read = []
    pinned = _read(_features(monkeypatch, op, {d0: 64, d2: 64}), "arg0")
    assert (pinned.advances_with_loop_var, pinned.loop_factor) == (False, 8)


def test_a_squeezed_advance_stamp_is_eligible_without_the_variable_in_the_index(
    monkeypatch,
):
    """The rebased direct read ``read_copy_elision`` builds: the loop variable left
    the index, and the stamp carries the advance (one expert bank slice per trip).
    It is the same walked read as before the rebase, so it is eligible."""
    op = _looped_op(
        [8],
        [[]],
        [_hint(u0, sympy.Integer(8))],
        write_index=64 * d0 + d2,
        read_indices=[d2],
    )
    op.loop_info.tiled_dims_per_read = [[[]]]
    op.loop_info.squeezed_advance_per_read = [[[(sympy.Integer(4096), 1)]]]
    walked = _read(
        _features(monkeypatch, op, {d0: 64, d2: 64}, candidate_work_slices=[{d2: 4}]),
        "arg0",
    )
    assert (walked.advances_with_loop_var, walked.loop_factor) == (True, 1)
    assert walked.has_partitioning_candidate


def test_a_zero_coefficient_loop_variable_is_not_eligible_without_stamps(monkeypatch):
    from torch.utils._sympy.functions import FloorDiv

    op = _looped_op(
        [8],
        [[]],
        [_hint(u0, sympy.Integer(8))],
        write_index=64 * d0 + d2,
        read_indices=[4096 * FloorDiv(u0, 2) + d2],
    )
    pinned = _read(_features(monkeypatch, op, {d0: 64, d2: 64}), "arg0")
    assert (pinned.advances_with_loop_var, pinned.loop_factor) == (False, 8)


def test_a_read_walked_only_by_the_levels_tiled_dim_is_a_coarse_loop_read(
    monkeypatch,
):
    """The level tiles the op's row dim ``d0`` and the read carries ``d0``, so the
    loop walks it once (factor 1), but its ``u0`` term has coefficient 0: the
    for_each_tile variable does not advance it. That is the coarse-loop operand the
    loop-delivery term leaves out, so it is not eligible."""
    from torch.utils._sympy.functions import FloorDiv

    op = _looped_op(
        [8],
        [[0]],
        [_hint(u0, sympy.Integer(8))],
        write_index=64 * d0 + d1,
        read_indices=[64 * d0 + 4096 * FloorDiv(u0, 2) + d2],
        data=SimpleNamespace(
            ranges=[64, 64], reduction_ranges=[64], reduction_type=None
        ),
    )
    read = _read(_features(monkeypatch, op, {d0: 64, d1: 64, d2: 64}), "arg0")
    assert (read.advances_with_loop_var, read.loop_factor) == (False, 1)


# Through the callers. The allocator builds its objective from
# ``CoOptimizingAllocator._extract_op_features`` (the legal menu goes in) and prices
# with its own ``_COST_PARAMS``; the report calls ``extract_op_features(op)`` with no
# menu and the default params.


def _row_tiled_expert_matmul(read_stamps):
    """One expert per trip of an 8-trip loop that also tiles the output rows ``d0``.

    ``arg0`` is the activation tile; its index carries ``d0`` and ``u0``.
    ``arg1`` is the expert bank, walked by ``u0``. ``read_stamps`` is the
    lowering's per-read stamp list.
    """
    from torch_spyre._inductor.constants import BATCH_MATMUL_OP

    op = _looped_op(
        [8],
        [[0]],
        [_hint(u0, sympy.Integer(8))],
        write_index=64 * d0 + d1,
        read_indices=[64 * d0 + 4096 * u0 + d2, 64 * d2 + d1 + 4096 * u0],
        data=SimpleNamespace(
            ranges=[64, 64], reduction_ranges=[64], reduction_type=BATCH_MATMUL_OP
        ),
    )
    op.get_size = lambda: [64, 64]
    op.loop_info.tiled_dims_per_read = read_stamps
    op.loop_info.squeezed_advance_per_read = []
    return op


_ADVANCES = [[(0, sympy.Integer(64))]]  # one level, advancing
_PINNED = [[]]  # one level, the explicit "pinned" verdict
_IT_SPACE = {d0: 64, d1: 64, d2: 64}
_MENU = [{d0: 2, d1: 4}, {d1: 8}]  # 8 cores: the delivery estimate is nonzero


def _allocator_features(
    monkeypatch, op, chosen=0, *, menu=_MENU, it_space=_IT_SPACE, graph=None, is_lx=None
):
    """The allocator's extraction with its legal ``menu``. ``chosen`` is a menu
    index, or a ``{dim: split symbol}`` map that keeps the splits symbolic."""
    from torch_spyre._inductor.scratchpad import sa_cooptimizer
    from torch_spyre._inductor.scratchpad.allocator import CoOptimizingAllocator
    from torch_spyre._inductor.scratchpad.plan_solver import CoreDivision

    monkeypatch.setattr(sa_cooptimizer, "iteration_space_from_op", lambda _: it_space)
    monkeypatch.setattr(dcm, "iteration_space_from_op", lambda _: it_space)
    monkeypatch.setattr(dcm, "_indirect_write_elems", lambda *_: None)
    buffers = {
        "buf1": SimpleNamespace(
            sym_core_divs=menu[chosen] if isinstance(chosen, int) else chosen,
            core_divisions=[CoreDivision(splits=dict(s)) for s in menu],
        )
    }
    with V.set_graph_handler(graph or _StubGraph()):
        return CoOptimizingAllocator._extract_op_features(
            None, None, "buf1", buffers, is_lx or {}, op=op
        )


def _delivery_ns(feature, params=None):
    from torch_spyre._inductor.cost_model import _partitioned_operand_read_excess
    from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS

    return _partitioned_operand_read_excess([feature], params or _COST_PARAMS)


def _without(feature, name):
    import dataclasses

    return dataclasses.replace(
        feature, args=[a for a in feature.args if a.name != name]
    )


def test_the_allocator_prices_loop_delivery_only_for_reads_the_loop_variable_walks(
    monkeypatch,
):
    """Both reads advance by the lowering's stamps: both are walked once and both
    get the allocator's loop-delivery estimate."""
    walked = _allocator_features(
        monkeypatch, _row_tiled_expert_matmul([_ADVANCES, _ADVANCES])
    )
    flags = {
        a.name: (a.advances_with_loop_var, a.loop_factor, a.has_partitioning_candidate)
        for a in walked.args
        if a.role == "input"
    }
    assert flags == {"arg0": (True, 1, True), "arg1": (True, 1, True)}
    assert _delivery_ns(walked) > _delivery_ns(_without(walked, "arg0")) > 0


def test_a_read_whose_advance_moved_to_its_restickify_copy_gets_no_loop_delivery(
    monkeypatch,
):
    """Synthetic stamp-consistency check, not an observed lowered-kernel case.

    Through the allocator: the activation read below keeps ``u0`` in its index but
    carries an empty ("pinned") stamp, and the row tile ``d0`` still walks it
    (factor 1). The free-symbol rule priced it as a loop-walked operand; by the
    stamp it is not one, so only the bank keeps the estimate. The read is
    constructed for the check: no reachable read with ``u0`` in its index and a
    pinned stamp is known, and whether ``insert_restickify`` (which the test name
    recalls) or any other pass produces one is not established."""
    moved = _allocator_features(
        monkeypatch, _row_tiled_expert_matmul([_PINNED, _ADVANCES])
    )
    activation, bank = _read(moved, "arg0"), _read(moved, "arg1")
    assert (activation.advances_with_loop_var, activation.loop_factor) == (False, 1)
    assert activation.has_partitioning_candidate
    assert (bank.advances_with_loop_var, bank.loop_factor) == (True, 1)
    assert _delivery_ns(moved) == pytest.approx(_delivery_ns(_without(moved, "arg0")))
    assert _delivery_ns(moved) > 0


def test_the_report_omits_the_loop_delivery_estimate(monkeypatch):
    """The report extracts without the legal menu, so no read has partitioning
    evidence and the estimate is omitted for every shape, eligible or not. Its
    totals therefore need not rank eligible looped-matmul plans as the allocator
    does."""
    from torch_spyre._inductor.cost_model import CostParams

    op = _row_tiled_expert_matmul([_ADVANCES, _ADVANCES])
    report = _features(monkeypatch, op, _IT_SPACE)
    assert [a.advances_with_loop_var for a in report.args if a.role == "input"] == [
        True,
        True,
    ]
    assert not any(a.has_partitioning_candidate for a in report.args)
    assert _delivery_ns(report, CostParams()) == 0


def test_a_rebased_bank_read_keeps_the_estimate_of_its_pre_rebase_form(monkeypatch):
    """The gained case, through the allocator's extraction. The pre-rebase bank read
    carries ``u0``; the rebased one (``read_copy_elision``) drops it and keeps the
    advance in the squeezed stamp. One physical read, one price. Under the
    free-symbol rule the rebased form lost the estimate. (Today the allocator never
    sees a rebased matmul read: its pricing projection rebases pointwise copies only,
    and the elision itself runs after planning.)"""
    before = _row_tiled_expert_matmul([_PINNED, _ADVANCES])
    rebased = _row_tiled_expert_matmul([_PINNED, [[]]])
    rebased.get_read_writes().reads[1].index = 64 * d2 + d1
    rebased.loop_info.squeezed_advance_per_read = [[[]], [[(sympy.Integer(4096), 1)]]]
    bank_before = _read(_allocator_features(monkeypatch, before), "arg1")
    priced = _allocator_features(monkeypatch, rebased)
    bank_after = _read(priced, "arg1")
    assert (bank_after.advances_with_loop_var, bank_after.loop_factor) == (True, 1)
    assert (
        bank_after.has_partitioning_candidate == bank_before.has_partitioning_candidate
    )
    assert _delivery_ns(priced) == pytest.approx(
        _delivery_ns(_allocator_features(monkeypatch, before))
    )


# DMA requests of looped matmul operand reads, at the MoE expert-loop shape: 128
# experts, 128 tokens, hidden 2816, inter 704. Each run of a core's read is one DMA
# request. A column split that leaves each core one stick (128 B) of every bank row
# makes thousands of them per trip. Expected values are hand arithmetic from the
# calibrated constants: 7.5 ns per request at 4-16 cores (11 and 22 cores take it),
# 3.75 at 32, a 150 B/ns peak, and 150/32 B/ns per core for the delivery estimate.
_E, _T, _K, _N = 128, 128, 2816, 704
_ONE_STICK = {d0: 2, d1: 11}  # 22 cores
_K_SPLIT = {d0: 8, d2: 4}  # 32 cores
_GATE_MENU = [_ONE_STICK, _K_SPLIT, {d0: 32}]


class _LayoutGraph(_StubGraph):
    """Graph inputs with real device layouts, by name: ``{name: (size, layout)}``."""

    def __init__(self, buffers):
        self.graph_input_names = list(buffers)
        self._buffers = buffers

    def get_buffer(self, name):
        if name not in self._buffers:
            return None
        size, layout = self._buffers[name]
        return SimpleNamespace(
            get_layout=lambda: SimpleNamespace(device_layout=layout, allocation=None),
            get_size=lambda: list(size),
        )


def _moe_gate(hidden=_K, cols=_N, bank_dtype=DataFormats.SEN169_FP16):
    """``out[m, n] = sum_k act[m, k] * bank[u0, k, n]``, one expert per trip, with
    its graph and iteration space. The activation lies in stick planes
    ``[K/64, T, 64]``; the bank keeps each row's sticks together, ``[E, K, N/64, 64]``."""
    from torch_spyre._inductor.constants import BATCH_MATMUL_OP

    sizes = (_T, cols, hidden)
    op = _looped_op(
        [_E],
        [[]],
        [_hint(u0, sympy.Integer(_E))],
        write_index=cols * d0 + d1,
        read_indices=[],
        data=SimpleNamespace(
            ranges=[_T, cols], reduction_ranges=[hidden], reduction_type=BATCH_MATMUL_OP
        ),
    )
    reads = [
        MemoryDep("arg0", hidden * d0 + d2, (d0, d1, d2), sizes),
        MemoryDep("arg1", hidden * cols * u0 + cols * d2 + d1, (d0, d1, d2), sizes),
    ]
    write = MemoryDep("buf1", cols * d0 + d1, (d0, d1, d2), sizes)
    op.get_read_writes = lambda: SimpleNamespace(reads=reads, writes=[write])
    op.get_size = lambda: [_T, cols]
    graph = _LayoutGraph(
        {
            "arg0": (
                [_T, hidden],
                SpyreTensorLayout(
                    device_size=[hidden // 64, _T, 64],
                    stride_map=[64, hidden, 1],
                    device_dtype=DataFormats.SEN169_FP16,
                ),
            ),
            "arg1": (
                [_E, hidden, cols],
                SpyreTensorLayout(
                    device_size=[_E, hidden, cols // 64, 64],
                    stride_map=[hidden * cols, cols, 64, 1],
                    device_dtype=bank_dtype,
                ),
            ),
        }
    )
    return op, graph, {d0: _T, d1: cols, d2: hidden}


def _requests_ns(ops, params=None):
    from torch_spyre._inductor.cost_model import _loop_operand_request_excess
    from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS

    return _loop_operand_request_excess(ops, params or _COST_PARAMS)


@pytest.mark.parametrize(
    "split,act_run,bank_run",
    [
        (_ONE_STICK, _T // 2 * 128, 128),  # one stick of every bank row
        (_K_SPLIT, _T // 8 * 128, _K // 4 * _N * 2),  # whole bank rows
        ({d0: 32}, _T // 32 * 128, _K * _N * 2),  # tokens do not index the bank
    ],
)
def test_the_extractor_measures_each_matmul_operands_dma_run(
    monkeypatch, split, act_run, bank_run
):
    """A core's contiguous run of each input, from its device layout, and the read's
    own footprint per trip (one expert, not the op's M*N*K space or the whole bank).
    A matmul keeps no single-read transport geometry, so the transport term never
    prices it."""
    op, graph, space = _moe_gate()
    feature = _features(
        monkeypatch,
        op,
        space,
        graph=graph,
        work_slices={s: split.get(s, 1) for s in space},
        candidate_work_slices=_GATE_MENU,
    )
    act, bank = _read(feature, "arg0"), _read(feature, "arg1")
    assert (act.read_run_bytes, act.read_tile_elems) == (act_run, _T * _K)
    assert (bank.read_run_bytes, bank.read_tile_elems) == (bank_run, _K * _N)
    assert (feature.transport_read_run_bytes, feature.transport_tile_elems) == (
        None,
        None,
    )


def test_operand_geometry_declines_an_indirect_bank_index(monkeypatch):
    op, graph, space = _moe_gate()
    kwargs = dict(graph=graph, work_slices=_ONE_STICK, candidate_work_slices=_GATE_MENU)
    before = _read(_features(monkeypatch, op, space, **kwargs), "arg1")
    assert (before.read_run_bytes, before.read_tile_elems) == (128, _K * _N)
    rw = op.get_read_writes()
    # An unknown loaded index is outside the iteration space; unlike u0 it is
    # not a for_each_tile offset that the geometry proof can pin per invocation.
    bank = dataclasses.replace(
        rw.reads[1],
        index=rw.reads[1].index.subs(
            u0, sympy.Symbol("idx", integer=True, nonnegative=True)
        ),
    )
    op.get_read_writes = lambda: SimpleNamespace(
        reads=[rw.reads[0], bank], writes=rw.writes
    )
    feature = _features(monkeypatch, op, space, **kwargs)
    read = _read(feature, "arg1")
    with V.set_graph_handler(graph):
        previous_run = dcm._read_run_bytes(op, bank, _ONE_STICK)
    assert (read.read_run_bytes, read.read_tile_elems) == (previous_run, None)


def test_operand_geometry_needs_a_menu_for_symbolic_stick_splits(monkeypatch):
    op, graph, space = _moe_gate()
    splits = {d0: 1, d1: 1, d2: sympy.Symbol("split_k", integer=True, positive=True)}
    kwargs = dict(graph=graph, work_slices=splits)
    unknown = _read(_features(monkeypatch, op, space, **kwargs), "arg0")
    proven = _read(
        _features(monkeypatch, op, space, candidate_work_slices=_GATE_MENU, **kwargs),
        "arg0",
    )
    with V.set_graph_handler(graph):
        previous_run = dcm._read_run_bytes(op, op.get_read_writes().reads[0], splits)
    assert (unknown.read_run_bytes, unknown.read_tile_elems) == (previous_run, None)
    assert proven.read_run_bytes is not None
    assert proven.read_tile_elems == _T * _K


@pytest.mark.parametrize(
    "shape,split,ns,cores",
    [
        ((_K, _N), _ONE_STICK, 7.5, 22),  # requests 26.354 ms > delivery 1.538 ms
        ((_K, _N), _K_SPLIT, 3.75, 32),  # long runs: no excess
        ((704, 2816), {d1: 11}, 7.5, 11),  # 512 B: requests 4.051 < delivery 6.459
    ],
)
def test_a_looped_operand_read_is_charged_the_slower_of_requests_and_delivery(
    monkeypatch, shape, split, ns, cores
):
    """Through the allocator's extraction and params: the bank's DMA requests and
    its loop-delivery estimate are excess time over the same bytes/peak, so the read
    is charged the larger, never the sum. Short runs add requests; long runs, or a
    delivery-bound read, keep the previous price. The report extracts without the
    legal menu, so it has no delivery estimate and charges the requests alone."""
    from torch_spyre._inductor.cost_model import CostParams, predict_ops
    from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS

    op, graph, space = _moe_gate(*shape)
    feature = _allocator_features(
        monkeypatch,
        op,
        menu=[split, {d0: 8}],
        it_space=space,
        graph=graph,
    )
    assert feature.cores == cores
    bank = _read(feature, "arg1")
    payload = bank.read_tile_elems * 2
    requests = _E * (payload / bank.read_run_bytes * ns - payload / 150)
    delivery = max(0.0, _E * payload * (32 / (cores * 150) - 1 / 150))
    assert _delivery_ns(feature) == pytest.approx(delivery, rel=1e-9)
    added = max(0.0, requests - delivery)
    assert _requests_ns([feature]) == pytest.approx(added, rel=1e-9)
    stripped = dataclasses.replace(
        feature,
        args=[
            dataclasses.replace(a, read_run_bytes=None, read_tile_elems=None)
            for a in feature.args
        ],
    )
    assert predict_ops([feature], _COST_PARAMS) - predict_ops(
        [stripped], _COST_PARAMS
    ) == pytest.approx(added, rel=1e-9)
    report = _features(
        monkeypatch,
        op,
        space,
        graph=graph,
        work_slices={s: split.get(s, 1) for s in space},
    )
    assert _requests_ns([report], CostParams()) == pytest.approx(
        max(0.0, requests), rel=1e-9
    )


@pytest.mark.parametrize(
    "menu,bank_dtype,declined",
    [
        (_GATE_MENU, DataFormats.SEN169_FP16, set()),
        # 44 stick planes per row: a K split of 3 cannot keep whole sticks.
        ([*_GATE_MENU, {d2: 3}], DataFormats.SEN169_FP16, {"arg0"}),
        (_GATE_MENU, DataFormats.IEEE_FP32, {"arg1"}),
    ],
    ids=["fp16", "uneven-k-split", "fp32-bank"],
)
def test_the_symbolic_operand_price_equals_the_concrete_price_at_every_candidate(
    monkeypatch, menu, bank_dtype, declined
):
    """The allocator's symbolic run and objective, at each legal candidate, equal
    that candidate's concrete extraction and price. A read whose geometry is not
    proven at every candidate, or not fp16, declines everywhere, never a price for
    some candidates only. CP-SAT keeps the term at every pinned candidate."""
    from test_cost_model_replication import _solve_pinned

    from torch_spyre._inductor.cost_model import predict_ops
    from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS

    op, graph, space = _moe_gate(bank_dtype=bank_dtype)
    names = {d1: "split_n", d2: "split_k", d0: "split_m"}  # _solve_pinned's order
    symbols = {
        d: sympy.Symbol(n, integer=True, positive=True) for d, n in names.items()
    }
    kwargs = dict(menu=menu, it_space=space, graph=graph)
    symbolic = _allocator_features(monkeypatch, op, symbols, **kwargs)
    term = sympy.sympify(_requests_ns([symbolic]))
    objective = sympy.sympify(predict_ops([symbolic], _COST_PARAMS))
    pinned = [tuple(c.get(d, 1) for d in names) for c in menu]
    priced = []
    for i, candidate in enumerate(menu):
        concrete = _allocator_features(monkeypatch, op, i, **kwargs)
        at = {symbols[d]: candidate.get(d, 1) for d in names}
        for name in ("arg0", "arg1"):
            run = _read(symbolic, name).read_run_bytes
            assert (run is None) == (name in declined)
            assert run is None or sympy.sympify(run).free_symbols
            assert (None if run is None else sympy.sympify(run).subs(at)) == _read(
                concrete, name
            ).read_run_bytes
        expected = float(_requests_ns([concrete]))
        priced.append(expected)
        assert float(term.subs(at)) == pytest.approx(expected, abs=1e-3)
        assert float(objective.subs(at)) == pytest.approx(
            float(predict_ops([concrete], _COST_PARAMS)), rel=1e-9
        )
        assert _solve_pinned(term, pinned, i) == pytest.approx(expected, abs=2.0)
    # At the one-stick candidate only the bank's runs are short: declined, it adds
    # nothing. (At 32 token cores the activation's 512 B runs are priced too.)
    assert (priced[0] > 0) == ("arg1" not in declined)


def test_a_stamped_level_without_a_loop_variable_follows_its_stamp(monkeypatch):
    """Review repro (H=16 heads, T=64, N=128, K=64; ``[H, T, N]`` output).

    The loop tiles output dim 0 with tile size 1, so the iteration space has no
    symbol for it (``_tiled_symbols_per_level`` skips unit-size ranges) and the level
    declares a dim but names no symbol. The level has no paired ``for_each_tile``
    variable (no loop-variable hint). The lowering's stamp says the output advances
    at that level, and the stamp is consulted whether or not a variable is paired:
    the write is walked once and the matmul work is the loop's true total
    ``16 * T * N * K``, not 16 times it.

    Synthetic, mirroring the reviewer's repro. Which lowerings reach it (a
    ``spyre_hint`` or solver coarse loop, or a hint/level count mismatch) is the
    reviewer's statement, not independently established here.
    """
    H, T, N, K = 16, 64, 128, 64
    m, n, r0 = sympy.symbols("m n r0", integer=True)
    feature = _looped_matmul_features(
        monkeypatch,
        size=[H, T, N],
        ranges=[1, T, N],  # the unit-size head range has no iteration symbol
        k=K,
        trips=[H],
        tiled_out=[[0]],
        output_tiled_dims=[[(0, 1)]],
        write_index=N * m + n,
        read_indices=[K * m + r0, N * r0 + n],
        it_space={m: T, n: N, r0: K},
    )
    assert _output_factor(feature) == 1
    assert feature.matmul_macs == H * T * N * K


def test_a_stamped_coarse_loop_read_keeps_loop_delivery_disabled(monkeypatch):
    """Complete coarse-loop stamps affect traffic, without enabling loop delivery.

    The allocator has a legal partitioning menu, but no paired for_each_tile
    variable. Applying #4995's stamp-precedence fix must keep that distinction.
    """
    op = _row_tiled_expert_matmul([_ADVANCES, _PINNED])
    op.dim_hints = []
    feature = _allocator_features(monkeypatch, op)
    read = _read(feature, "arg0")
    assert read.loop_factor == 1
    assert read.has_partitioning_candidate
    assert not read.advances_with_loop_var
    assert _delivery_ns(feature) == 0
