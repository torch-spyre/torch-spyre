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

"""Indexed writes through flattened/reshaped views (torch-spyre#5125).

A paged KV cache is allocated as ``[num_blocks, block_size, kv_heads, head_dim]``
and written one *slot* at a time through its flattened view
``cache.view(-1, kv_heads, head_dim)``. The view changes rank (4-D -> 3-D) and its
leading dim (``num_blocks * block_size`` slots) spans two dims of the base.

Compiled on Spyre, ``index_put_``, ``index_copy_`` and ``cache[slots] = values``
used to silently leave every requested row unchanged on this destination, while
the same ops on a directly allocated 3-D cache, or on a clone of the view,
passed. The scattered slot index was sized and placed against the 4-D base
instead of the view, so it landed in the wrong device dim with the wrong extent.

This file covers six view topologies:

* ``Test2DBase1DView``       — ``[M, N].view(-1)`` (2-D→1-D, scalar scatter)
* ``Test3DBase2DView``       — ``[A, B, C].view(-1, C)`` (3-D→2-D, 2 leading dims merged)
* ``Test4DBase3DView``       — ``[B, S, H, D].view(-1, H, D)`` (KV-cache; 4-D→3-D)
* ``Test4DBase2DView``       — ``[A, B, C, D].view(-1, D)`` (4-D→2-D, 3 dims merged)
* ``Test5DBase4DView``       — ``[A, B, C, D, E].view(-1, C, D, E)`` (5-D→4-D)
* ``Test3DBaseTrailingMerge``— ``[A, B, C].view(A, -1)`` (trailing 2 dims merged, scatter on dim 0)

Each topology is tested for three dtypes:

* ``bfloat16`` — reported dtype; elems_per_stick=64; copy-in is lossless → exact
  comparison (rtol=0, atol=0).
* ``float32``  — elems_per_stick=32 (different stick geometry); copy-in lossless
  → exact comparison (rtol=0, atol=0).
* ``float16``  — elems_per_stick=64 (same as bfloat16); the copy-in path through
  the Spyre device format introduces ≤1 ULP rounding on the written rows, so a
  loose tolerance (atol=1e-2, rtol=1e-2) is used.  The key assertion is that the
  written rows are non-zero (the bug dropped them to zero entirely).

Every case writes through the view of a plain ``.to("spyre")`` cache (default
device layout, so the scatter dim is typically not at device position 0 and a
relayout is required) and compares the *whole base* cache with the CPU result.
Slots are unique in every case: with a repeated index the winner of a scatter
is undefined.
"""

import unittest

import pytest
import torch
from torch._inductor.exc import InductorError

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_OPERATIONS = ("index_put_", "index_copy_", "advanced_indexing")


# Tolerance per dtype.  bfloat16 and float32 copy-in paths are lossless so
# exact comparison is valid.  float16 goes through the Spyre device format
# with ≤1 ULP rounding on written rows; a loose tolerance is required.
_TOLERANCES = {
    torch.bfloat16: (0, 0),
    torch.float32: (0, 0),
    torch.float16: (1e-2, 1e-2),  # (rtol, atol)
}


def _run_and_compare(test_case, update_fn, dtype, base_shape, view_shape, slots):
    """Compile *update_fn* on Spyre and assert it matches the CPU reference.

    Args:
        test_case: ``unittest.TestCase`` instance (for assertions).
        update_fn: callable ``(base, values, slots) -> base`` that performs
            an in-place scatter through a view.  Must return the *base* tensor
            so the whole buffer is compared.
        dtype: element dtype.
        base_shape: shape of the allocated buffer.
        view_shape: shape the function will call ``.view()`` with.
        slots: list[int] of *unique* slot indices (into view's leading dim).
    """
    generator = torch.Generator().manual_seed(42)
    # "row" shape: everything after the scattered leading dim of the view.
    row_shape = list(view_shape[1:]) if len(view_shape) > 1 else []
    base = torch.zeros(base_shape, dtype=dtype)
    values = torch.randn([len(slots)] + row_shape, dtype=dtype, generator=generator)
    slot_idx = torch.tensor(slots, dtype=torch.int64)

    # CPU reference
    expected = update_fn(base.clone(), values.clone(), slot_idx.clone())

    # Verify the reference actually wrote something (guards against a broken
    # update_fn that leaves the buffer all-zero, which would make the test vacuous).
    flat_view = expected.reshape(view_shape)
    test_case.assertTrue(
        flat_view[slot_idx].any(),
        "CPU reference did not write any non-zero value — check update_fn",
    )

    # Spyre compiled result
    actual = torch.compile(update_fn, dynamic=False)(
        base.to("spyre"), values.to("spyre"), slot_idx.to("spyre")
    ).to("cpu")

    rtol, atol = _TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# 2-D base → 1-D view   [M, N].view(-1)
# Scatter would be element-granularity, but Spyre's scatter engine operates on
# rows (full sticks).  A bfloat16 stick holds 64 elements; scattering into
# individual elements within the same stick is undefined on hardware.
# Even a directly-allocated 1-D tensor of 8192 bf16 elements fails scatter
# with "restickify op_spec has 3 args" because the device layout is [128, 64]
# (128 rows × 64 elems/stick) and there is no way to express an arbitrary
# element address as a single scatter row index.
#
# This class documents the limitation and asserts that the compiler raises
# rather than silently producing wrong results.
# ---------------------------------------------------------------------------

_2D_M = 64
_2D_N = 128  # 64×128 = 8192 elements; stick=64 bf16 → 128 rows
_2D_SLOTS = {
    "low": [0, 1, 5, 127],
    "mid": [128, 255, 512, 1023],
    "boundary": [127, 128, 255, 256],
}


def _2d_index_put(base, values, slots):
    base.view(-1).index_put_((slots,), values)
    return base


def _2d_index_copy(base, values, slots):
    base.view(-1).index_copy_(0, slots, values)
    return base


def _2d_advanced(base, values, slots):
    base.view(-1)[slots] = values
    return base


_2D_OPS = {
    "index_put_": _2d_index_put,
    "index_copy_": _2d_index_copy,
    "advanced_indexing": _2d_advanced,
}


class Test2DBase1DView(unittest.TestCase):
    """Fully-flat 1-D scatter through a 2-D base — known unsupported on Spyre.

    Spyre's scatter engine operates on rows (full sticks of 64 bfloat16
    elements), not individual elements.  A bfloat16 stick holds 64 elements,
    so a fully-flat view ``[M, N].view(-1)`` produces a device layout of
    ``[M*N//64, 64]`` (rows × elements/stick).  There is no way to express an
    arbitrary element address as a single scatter row index; even a
    directly-allocated 1-D tensor of 8192 bf16 elements fails with
    ``"restickify op_spec has 3 args"`` at the backend codegen stage.

    ``@pytest.mark.xfail(strict=True)`` means:
      * while the hardware limit exists  → CI shows ``xfail`` (expected), green.
      * if the hardware gains element-scatter → the test passes the inner body,
        pytest sees an unexpected pass (XPASS) and fails the suite — that is
        the signal to remove or update this decorator.
    """

    def _check(self, op_name, slots_name, dtype):
        _run_and_compare(
            self,
            _2D_OPS[op_name],
            dtype,
            base_shape=(_2D_M, _2D_N),
            view_shape=(_2D_M * _2D_N,),
            slots=_2D_SLOTS[slots_name],
        )

    _XFAIL_REASON = (
        "Spyre scatter engine operates on rows (sticks), not individual "
        "elements; element-granularity scatter into a fully-flat 1-D view "
        "is unsupported at the hardware level (tracked separately from #5125). "
        "strict=True: an unexpected pass (XPASS) signals that hardware "
        "element-scatter has been added and this decorator should be removed. "
        "Note: subTest is intentionally NOT used here — pytest xfail + "
        "unittest.subTest don't compose (subTest swallows exceptions into "
        "sub-results, so the outer method appears to pass even when every "
        "subtest failed, turning strict xfail into an XPASS false-alarm)."
    )

    @pytest.mark.xfail(raises=InductorError, strict=True, reason=_XFAIL_REASON)
    def test_bfloat16(self):
        """Known unsupported: element-scatter into a fully-flat 1-D view (bfloat16)."""
        for op in _OPERATIONS:
            for sl in _2D_SLOTS:
                self._check(op, sl, torch.bfloat16)

    @pytest.mark.xfail(raises=InductorError, strict=True, reason=_XFAIL_REASON)
    def test_float32(self):
        """Known unsupported: element-scatter into a fully-flat 1-D view (float32)."""
        for op in _OPERATIONS:
            for sl in _2D_SLOTS:
                self._check(op, sl, torch.float32)

    @pytest.mark.xfail(raises=InductorError, strict=True, reason=_XFAIL_REASON)
    def test_float16(self):
        """Known unsupported: element-scatter into a fully-flat 1-D view (float16)."""
        for op in _OPERATIONS:
            for sl in _2D_SLOTS:
                self._check(op, sl, torch.float16)


# ---------------------------------------------------------------------------
# 3-D base → 2-D view   [A, B, C].view(-1, C)
# Two leading dims are merged; scatter is over the first dim of the view.
# ---------------------------------------------------------------------------

_3D_A = 32
_3D_B = 16
_3D_C = 64  # trailing dim, kept intact
_3D_SLOTS = {
    "low": [0, 1, 5, 15],
    "mid": [16, 31, 32, 47],
    "boundary": [15, 16, 31, 32],
}


def _3d_index_put(base, values, slots):
    base.view(-1, _3D_C).index_put_((slots,), values)
    return base


def _3d_index_copy(base, values, slots):
    base.view(-1, _3D_C).index_copy_(0, slots, values)
    return base


def _3d_advanced(base, values, slots):
    base.view(-1, _3D_C)[slots] = values
    return base


_3D_OPS = {
    "index_put_": _3d_index_put,
    "index_copy_": _3d_index_copy,
    "advanced_indexing": _3d_advanced,
}


class Test3DBase2DView(unittest.TestCase):
    """3-D base ``[32, 16, 64]`` written through ``view(-1, 64)`` (2-D view)."""

    def _check(self, op_name, slots_name, dtype):
        _run_and_compare(
            self,
            _3D_OPS[op_name],
            dtype,
            base_shape=(_3D_A, _3D_B, _3D_C),
            view_shape=(_3D_A * _3D_B, _3D_C),
            slots=_3D_SLOTS[slots_name],
        )

    def test_bfloat16(self):
        for op in _OPERATIONS:
            for sl in _3D_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.bfloat16)

    def test_float32(self):
        for op in _OPERATIONS:
            for sl in _3D_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.float32)

    def test_float16(self):
        for op in _OPERATIONS:
            for sl in _3D_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.float16)


# ---------------------------------------------------------------------------
# 4-D base → 3-D view   [NUM_BLOCKS, BLOCK_SIZE, H, D].view(-1, H, D)
# This is the KV-cache shape from the original bug report.
# ---------------------------------------------------------------------------

NUM_BLOCKS = 512
BLOCK_SIZE = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128

_KV_SLOTS = {
    "issue_repro": [47, 63, 80, 95],
    "first_block": [0, 1, 5, 15],
    "block_boundaries": [15, 16, 31, 32],
}


def _kv_index_put(cache, values, slots):
    cache.view(-1, NUM_KV_HEADS, HEAD_DIM).index_put_((slots,), values)
    return cache


def _kv_index_copy(cache, values, slots):
    cache.view(-1, NUM_KV_HEADS, HEAD_DIM).index_copy_(0, slots, values)
    return cache


def _kv_advanced(cache, values, slots):
    cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[slots] = values
    return cache


_KV_OPS = {
    "index_put_": _kv_index_put,
    "index_copy_": _kv_index_copy,
    "advanced_indexing": _kv_advanced,
}


class Test4DBase3DView(unittest.TestCase):
    """KV-cache: 4-D base ``[512, 16, 8, 128]`` written through ``view(-1, 8, 128)``."""

    def _check(self, op_name, slots_name, dtype):
        _run_and_compare(
            self,
            _KV_OPS[op_name],
            dtype,
            base_shape=(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
            view_shape=(NUM_BLOCKS * BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
            slots=_KV_SLOTS[slots_name],
        )

    def test_bfloat16(self):
        for op in _OPERATIONS:
            for sl in _KV_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.bfloat16)

    def test_float32(self):
        for op in _OPERATIONS:
            for sl in _KV_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.float32)

    def test_float16(self):
        for op in _OPERATIONS:
            for sl in _KV_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.float16)


# ---------------------------------------------------------------------------
# 4-D base → 2-D view   [A, B, C, D].view(-1, D)
# Three leading dims are merged into one; only the innermost dim D is kept.
# ---------------------------------------------------------------------------

_4D2_A = 8
_4D2_B = 16
_4D2_C = 4
_4D2_D = 128  # kept intact; must be multiple of elems_per_stick
_4D2_SLOTS = {
    "low": [0, 1, 5, 15],
    "mid": [64, 127, 128, 255],
    "boundary": [63, 64, 127, 128],
}


def _4d2_index_put(base, values, slots):
    base.view(-1, _4D2_D).index_put_((slots,), values)
    return base


def _4d2_index_copy(base, values, slots):
    base.view(-1, _4D2_D).index_copy_(0, slots, values)
    return base


def _4d2_advanced(base, values, slots):
    base.view(-1, _4D2_D)[slots] = values
    return base


_4D2_OPS = {
    "index_put_": _4d2_index_put,
    "index_copy_": _4d2_index_copy,
    "advanced_indexing": _4d2_advanced,
}


class Test4DBase2DView(unittest.TestCase):
    """4-D base ``[8, 16, 4, 128]`` written through ``view(-1, 128)`` (2-D view).

    Three leading dims are folded into one scatter dim — exercises the
    multi-dim merge path in ``_merge_device_dims_for_view``.
    """

    def _check(self, op_name, slots_name, dtype):
        _run_and_compare(
            self,
            _4D2_OPS[op_name],
            dtype,
            base_shape=(_4D2_A, _4D2_B, _4D2_C, _4D2_D),
            view_shape=(_4D2_A * _4D2_B * _4D2_C, _4D2_D),
            slots=_4D2_SLOTS[slots_name],
        )

    def test_bfloat16(self):
        for op in _OPERATIONS:
            for sl in _4D2_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.bfloat16)

    def test_float32(self):
        for op in _OPERATIONS:
            for sl in _4D2_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.float32)

    def test_float16(self):
        for op in _OPERATIONS:
            for sl in _4D2_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.float16)


# ---------------------------------------------------------------------------
# 5-D base → 4-D view   [A, B, C, D, E].view(-1, C, D, E)
# Two leading dims merged; the remaining three trailing dims are preserved.
# ---------------------------------------------------------------------------

_5D_A = 32
_5D_B = 8
_5D_C = 4
_5D_D = 8
_5D_E = 64  # must be multiple of elems_per_stick
_5D_SLOTS = {
    "low": [0, 1, 5, 15],
    "mid": [32, 63, 64, 127],
    "boundary": [31, 32, 63, 64],
}


def _5d_index_put(base, values, slots):
    base.view(-1, _5D_C, _5D_D, _5D_E).index_put_((slots,), values)
    return base


def _5d_index_copy(base, values, slots):
    base.view(-1, _5D_C, _5D_D, _5D_E).index_copy_(0, slots, values)
    return base


def _5d_advanced(base, values, slots):
    base.view(-1, _5D_C, _5D_D, _5D_E)[slots] = values
    return base


_5D_OPS = {
    "index_put_": _5d_index_put,
    "index_copy_": _5d_index_copy,
    "advanced_indexing": _5d_advanced,
}


class Test5DBase4DView(unittest.TestCase):
    """5-D base ``[32, 8, 4, 8, 64]`` written through ``view(-1, 4, 8, 64)`` (4-D view).

    Two leading dims are merged — extends coverage to a 5-D source with
    multi-trailing-dim rows.
    """

    def _check(self, op_name, slots_name, dtype):
        _run_and_compare(
            self,
            _5D_OPS[op_name],
            dtype,
            base_shape=(_5D_A, _5D_B, _5D_C, _5D_D, _5D_E),
            view_shape=(_5D_A * _5D_B, _5D_C, _5D_D, _5D_E),
            slots=_5D_SLOTS[slots_name],
        )

    def test_bfloat16(self):
        for op in _OPERATIONS:
            for sl in _5D_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.bfloat16)

    def test_float32(self):
        for op in _OPERATIONS:
            for sl in _5D_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.float32)

    def test_float16(self):
        for op in _OPERATIONS:
            for sl in _5D_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.float16)


# ---------------------------------------------------------------------------
# 3-D base → 2-D view (trailing merge)   [A, B, C].view(A, -1)
# Leading dim A is untouched; the scatter is over dim 0.
# The two *trailing* dims (B, C) are merged, which is a different device-dim
# adjacency pattern from the leading-dim merges above.
# ---------------------------------------------------------------------------

_TR_A = 64  # number of rows (scatter target)
_TR_B = 8
_TR_C = 128  # kept together with B; B*C must be multiple of elems_per_stick
_TR_SLOTS = {
    "low": [0, 1, 5, 15],
    "mid": [16, 31, 32, 47],
    "boundary": [31, 32, 47, 48],
}


def _tr_index_put(base, values, slots):
    base.view(_TR_A, -1).index_put_((slots,), values)
    return base


def _tr_index_copy(base, values, slots):
    base.view(_TR_A, -1).index_copy_(0, slots, values)
    return base


def _tr_advanced(base, values, slots):
    base.view(_TR_A, -1)[slots] = values
    return base


_TR_OPS = {
    "index_put_": _tr_index_put,
    "index_copy_": _tr_index_copy,
    "advanced_indexing": _tr_advanced,
}


class Test3DBaseTrailingMerge(unittest.TestCase):
    """3-D base ``[64, 8, 128]`` written through ``view(64, -1)`` (trailing merge).

    Unlike the leading-dim merge cases, here the scatter axis (dim 0) is
    unchanged; it is the *trailing* dims that are folded together.  This
    exercises the trailing-adjacency branch of ``_merge_device_dims_for_view``.
    """

    def _check(self, op_name, slots_name, dtype):
        _run_and_compare(
            self,
            _TR_OPS[op_name],
            dtype,
            base_shape=(_TR_A, _TR_B, _TR_C),
            view_shape=(_TR_A, _TR_B * _TR_C),
            slots=_TR_SLOTS[slots_name],
        )

    def test_bfloat16(self):
        for op in _OPERATIONS:
            for sl in _TR_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.bfloat16)

    def test_float32(self):
        for op in _OPERATIONS:
            for sl in _TR_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.float32)

    def test_float16(self):
        for op in _OPERATIONS:
            for sl in _TR_SLOTS:
                with self.subTest(op=op, slots=sl):
                    self._check(op, sl, torch.float16)


if __name__ == "__main__":
    unittest.main()
