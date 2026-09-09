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

import sympy

import torch_spyre._inductor.dump_cost_model as dcm


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
