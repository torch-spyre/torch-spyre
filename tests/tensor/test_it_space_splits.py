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

"""Unit tests for splits_by_index_coeff and apply_splits_from_index_coeff.

These functions encode pre-scheduler core-division splits so they can be
recovered after the scheduler renames iteration-space symbols.  The key
invariant: the coefficient of a symbol in a tensor's flat index expression
is determined by the layout strides, which are fixed across the
pre-scheduling / codegen boundary.
"""

import sympy

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch_spyre._inductor.pass_utils import (
    apply_splits_from_index_coeff,
    splits_by_index_coeff,
)

# Pre-scheduler symbols
i0, i1, i2, i3 = sympy.symbols("i0 i1 i2 i3", integer=True)
# Post-scheduler symbols (renamed by the scheduler)
s0, s1, s2, s3 = sympy.symbols("s0 s1 s2 s3", integer=True)
# Reduction symbol
r0 = sympy.Symbol("r0", integer=True)
sr = sympy.Symbol("sr", integer=True)


class TestItSpaceSplits(TestCase):
    def setUp(self):
        torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Pointwise: all dims in write dep, no reduction
    # ------------------------------------------------------------------

    def test_pointwise_2d(self):
        # [M, N] pointwise, output write index = N*i0 + i1
        write_idx = 8 * i0 + i1
        splits = {i0: 4, i1: 2}
        stored = splits_by_index_coeff(splits, write_idx, write_idx)
        # Both output splits encoded, reduction dict empty
        self.assertEqual(stored, ({8: 4, 1: 2}, {}))

        recovered = apply_splits_from_index_coeff(
            stored, 8 * s0 + s1, 8 * s0 + s1, {s0: 8, s1: 8}
        )
        self.assertEqual(recovered, {s0: 4, s1: 2})

    def test_pointwise_3d(self):
        # [B, M, N] pointwise, output write index = M*N*i0 + N*i1 + i2
        write_idx = 32 * i0 + 8 * i1 + i2
        splits = {i0: 4, i1: 2, i2: 1}
        stored = splits_by_index_coeff(splits, write_idx, write_idx)
        # i2 split=1 is skipped
        self.assertEqual(stored, ({32: 4, 8: 2}, {}))

        recovered = apply_splits_from_index_coeff(
            stored, 32 * s0 + 8 * s1 + s2, 32 * s0 + 8 * s1 + s2, {s0: 4, s1: 8, s2: 8}
        )
        self.assertEqual(recovered, {s0: 4, s1: 2, s2: 1})

    def test_unity_splits_not_stored(self):
        # All splits are 1 — nothing stored
        write_idx = 8 * i0 + i1
        stored = splits_by_index_coeff({i0: 1, i1: 1}, write_idx, write_idx)
        self.assertEqual(stored, ({}, {}))

        recovered = apply_splits_from_index_coeff(
            stored, 8 * s0 + s1, 8 * s0 + s1, {s0: 8, s1: 8}
        )
        self.assertEqual(recovered, {s0: 1, s1: 1})

    # ------------------------------------------------------------------
    # Reduction: matmul — reduction dim absent from write dep
    # ------------------------------------------------------------------

    def test_matmul_2d(self):
        # [M, K] x [K, N] -> [M, N], M=8, N=4, K=16
        # write dep: N*i0 + i1 = 4*i0 + i1
        # first read dep (x): K*i0 + r0 = 16*i0 + r0
        write_idx = 4 * i0 + i1
        x_read_idx = 16 * i0 + r0
        splits = {i0: 4, i1: 2, r0: 8}
        stored = splits_by_index_coeff(splits, write_idx, x_read_idx)
        # i0->4 (coeff 4 in write), i1->2 (coeff 1 in write), r0->8 (coeff 1 in x)
        self.assertEqual(stored, ({4: 4, 1: 2}, {1: 8}))

        # Post-scheduler: same shapes, renamed symbols
        recovered = apply_splits_from_index_coeff(
            stored,
            4 * s0 + s1,
            16 * s0 + sr,
            {s0: 8, s1: 4, sr: 16},
        )
        self.assertEqual(recovered, {s0: 4, s1: 2, sr: 8})

    def test_matmul_square(self):
        # Square matmul M=N=K=8: i0 and r0 have same range but different splits
        # write dep: 8*i0 + i1, x read dep: 8*i0 + r0
        write_idx = 8 * i0 + i1
        x_read_idx = 8 * i0 + r0
        splits = {i0: 4, i1: 2, r0: 4}
        stored = splits_by_index_coeff(splits, write_idx, x_read_idx)
        # i0 and r0 share range 8 but they are in separate namespaces (output vs reduction)
        self.assertEqual(stored, ({8: 4, 1: 2}, {1: 4}))

        recovered = apply_splits_from_index_coeff(
            stored,
            8 * s0 + s1,
            8 * s0 + sr,
            {s0: 8, s1: 8, sr: 8},
        )
        self.assertEqual(recovered, {s0: 4, s1: 2, sr: 4})

    def test_batch_matmul(self):
        # [B, M, K] x [K, N] -> [B, M, N], B=2, M=8, N=4, K=16
        # write dep: M*N*i0 + N*i1 + i2 = 32*i0 + 4*i1 + i2
        # first read dep (x): K*M*i0 + K*i1 + r0 = 128*i0 + 16*i1 + r0
        write_idx = 32 * i0 + 4 * i1 + i2
        x_read_idx = 128 * i0 + 16 * i1 + r0
        splits = {i0: 2, i1: 4, i2: 2, r0: 8}
        stored = splits_by_index_coeff(splits, write_idx, x_read_idx)
        # i0->2 (coeff 32), i1->4 (coeff 4), i2->2 (coeff 1), r0->8 (coeff 1 in x)
        self.assertEqual(stored, ({32: 2, 4: 4, 1: 2}, {1: 8}))

        recovered = apply_splits_from_index_coeff(
            stored,
            32 * s0 + 4 * s1 + s2,
            128 * s0 + 16 * s1 + sr,
            {s0: 2, s1: 8, s2: 4, sr: 16},
        )
        self.assertEqual(recovered, {s0: 2, s1: 4, s2: 2, sr: 8})

    def test_reduction_unity_split_not_stored(self):
        # Reduction dim gets split=1 (not eligible for core division)
        write_idx = 4 * i0 + i1
        x_read_idx = 16 * i0 + r0
        splits = {i0: 4, i1: 1, r0: 1}
        stored = splits_by_index_coeff(splits, write_idx, x_read_idx)
        self.assertEqual(stored, ({4: 4}, {}))

        recovered = apply_splits_from_index_coeff(
            stored,
            4 * s0 + s1,
            16 * s0 + sr,
            {s0: 8, s1: 4, sr: 16},
        )
        self.assertEqual(recovered, {s0: 4, s1: 1, sr: 1})

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_size1_dim_not_assigned_split(self):
        # Tensor [4, 1, 8]: output strides [8, 8, 1] — size-1 dim shares stride
        # with its neighbour. On the apply side, the size-1 symbol must get the
        # default split (1), not the split belonging to the neighbouring dim.
        write_idx = 8 * i0 + 8 * i1 + i2  # i1 has range 1, same coeff as i0
        splits = {i0: 4, i1: 1, i2: 1}
        stored = splits_by_index_coeff(splits, write_idx, write_idx)
        # Only i0 produces a non-unity split; i1 and i2 are skipped
        self.assertEqual(stored, ({8: 4}, {}))

        # Post-scheduler: same shapes, s1 has range 1
        recovered = apply_splits_from_index_coeff(
            stored,
            8 * s0 + 8 * s1 + s2,
            8 * s0 + 8 * s1 + s2,
            {s0: 4, s1: 1, s2: 8},
        )
        # s1 must NOT receive the split=4 meant for s0
        self.assertEqual(recovered, {s0: 4, s1: 1, s2: 1})

    def test_broadcast_dim_not_assigned_split(self):
        # Broadcast dim has stride 0 — coeff==0, so absent from stored splits
        # and correctly defaults to 1 on the apply side.
        # Tensor [M, N] where second dim is broadcast (stride=0): index = M*i0
        write_idx = 8 * i0  # i1 is broadcast, not in index
        splits = {i0: 4, i1: 1}
        stored = splits_by_index_coeff(splits, write_idx, write_idx)
        self.assertEqual(stored, ({8: 4}, {}))

        recovered = apply_splits_from_index_coeff(
            stored, 8 * s0, 8 * s0, {s0: 8, s1: 8}
        )
        self.assertEqual(recovered, {s0: 4, s1: 1})

    def test_single_dim(self):
        # 1D tensor
        write_idx = i0
        stored = splits_by_index_coeff({i0: 8}, write_idx, write_idx)
        self.assertEqual(stored, ({1: 8}, {}))

        recovered = apply_splits_from_index_coeff(stored, s0, s0, {s0: 32})
        self.assertEqual(recovered, {s0: 8})

    def test_empty_splits(self):
        # No splits at all (scalar or all unity)
        write_idx = i0
        stored = splits_by_index_coeff({}, write_idx, write_idx)
        self.assertEqual(stored, ({}, {}))

        recovered = apply_splits_from_index_coeff(stored, s0, s0, {s0: 8})
        self.assertEqual(recovered, {s0: 1})


if __name__ == "__main__":
    run_tests()
