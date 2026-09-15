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

# Tests for project_dim_order, which re-expresses a multi-arg pointwise output's
# dim_order as a dim_order for one of its operands.
#
# These are pure-function tests: no compilation, no backend codegen, no device.
# The invariant under test is the one SpyreTensorLayout::init asserts
# (spyre_tensor_impl.cpp) — len(dim_order) == rank, or rank + 1 when the last
# entry is -1 (sparse stick).

import pytest

from torch_spyre._inductor.propagate_layouts import project_dim_order


def _assert_layout_invariant(dim_order, rank):
    """Mirror the TORCH_CHECK in SpyreTensorLayout::init."""
    sparse = len(dim_order) > 0 and dim_order[-1] == -1
    length_ok = len(dim_order) == rank or (sparse and len(dim_order) == rank + 1)
    assert length_ok, f"{dim_order} incompatible with rank {rank}"
    # The non-stick entries must be a permutation of the operand's dims.
    core = dim_order[:-1] if sparse else dim_order
    expected = list(range(rank))
    assert sorted(core) == expected, f"{dim_order} core is not {expected}"


class TestBroadcast:
    """Operand rank <= output rank: the case the projection always handled."""

    def test_equal_rank_is_identity(self):
        assert project_dim_order([1, 3, 0, 2], 4, 4) == [1, 3, 0, 2]

    def test_leading_broadcast_dims_dropped(self):
        # Operand [C, D] against output [A, B, C, D]: output dims 0,1 don't exist
        # on the operand and are dropped; 2,3 shift down to 0,1.
        assert project_dim_order([1, 3, 0, 2], 4, 2) == [1, 0]

    def test_scalar_operand(self):
        assert project_dim_order([1, 3, 0, 2], 4, 0) == []

    @pytest.mark.parametrize("arg_rank", [0, 1, 2, 3, 4])
    def test_invariant_holds(self, arg_rank):
        _assert_layout_invariant(project_dim_order([1, 3, 0, 2], 4, arg_rank), arg_rank)


class TestSparseStickMarker:
    """A trailing -1 means the stick maps to no host dim. It must survive."""

    def test_marker_preserved_at_equal_rank(self):
        # Regression: the old `if d >= rank_diff` filter dropped -1 whenever
        # rank_diff >= 0, quietly turning a sparse-stick operand layout dense.
        assert project_dim_order([1, 3, 0, 2, -1], 4, 4) == [1, 3, 0, 2, -1]

    def test_marker_preserved_under_broadcast(self):
        assert project_dim_order([1, 3, 0, 2, -1], 4, 2) == [1, 0, -1]

    def test_marker_not_treated_as_a_dim(self):
        # -1 must not be shifted into a real dim index.
        projected = project_dim_order([1, 3, 0, 2, -1], 4, 6)
        assert projected.count(-1) == 1
        assert projected[-1] == -1

    @pytest.mark.parametrize("arg_rank", [1, 2, 3, 4, 5, 6])
    def test_invariant_holds(self, arg_rank):
        _assert_layout_invariant(
            project_dim_order([1, 3, 0, 2, -1], 4, arg_rank), arg_rank
        )


class TestHigherRankOperand:
    """Operand rank > output rank: previously unhandled, and the crash."""

    def test_qwen3_rope_gather(self):
        # The real failure from hf-adapters#330. An index_select on a fused RoPE
        # chain: the gather's value operand is still the rank-6
        # [B, L, H, 2, 1, D/2] = [1, 64, 8, 2, 1, 64] intermediate (flatten and
        # transpose are views), while the gather's output is rank-4
        # [B, H, 1, D] = [1, 8, 1, 128].
        #
        # The old projection produced [3, 5, 2, 4, 1]: five entries against six
        # host sizes, and the -1 marker shifted into a bogus dim index 1. That
        # tripped "Incompatible host_size and dim_order".
        projected = project_dim_order([1, 3, 0, 2, -1], out_rank=4, arg_rank=6)
        assert projected != [3, 5, 2, 4, 1]
        _assert_layout_invariant(projected, 6)

    def test_extra_leading_dims_get_entries(self):
        # Operand rank 6, output rank 4: the operand's two extra leading dims
        # must appear, and the shared trailing dims stay right-aligned.
        assert project_dim_order([1, 3, 0, 2], 4, 6) == [0, 1, 3, 5, 2, 4]

    @pytest.mark.parametrize("arg_rank", [5, 6])
    @pytest.mark.parametrize("dim_order", [[1, 3, 0, 2], [1, 3, 0, 2, -1]])
    def test_invariant_holds(self, dim_order, arg_rank):
        _assert_layout_invariant(project_dim_order(dim_order, 4, arg_rank), arg_rank)
