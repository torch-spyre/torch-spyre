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

from types import SimpleNamespace
import unittest
from unittest import mock

import sympy
import torch
from sympy import Symbol
from torch._inductor.dependencies import MemoryDep, ReadWrites
from torch._inductor.ir import InputBuffer, MutationLayoutSHOULDREMOVE, Scatter
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

import torch_spyre._inductor.dump_cost_model as dcm
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor import work_division as wd
from torch_spyre._inductor.cost_model import ArgTraffic
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
    indices = [SimpleNamespace(name="slots")]
    op.data = mock.Mock(spec=Scatter)
    op.data.output_indexer = lambda _: indices
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
