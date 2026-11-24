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

"""A reader that takes its facts from a shim op's index: the cost features.

A strict ``LoopMember`` carries no per-read lists, so ``cost_features.extract_op_features`` reads
whether an operand advances at a level from the access's index. The device-free
accessor tests are in ``test_loop_shim.py``; this file runs the reader on the shim ops
of real compiled graphs and checks the loop facts it reports against what the plan
decided for each op.
"""

import pytest
import torch  # noqa: F401  (must precede torch_spyre._inductor imports)
from torch._inductor.ir import ComputedBuffer

from torch_spyre._inductor import passes
from torch_spyre._inductor.cost_features import extract_op_features
from torch_spyre._inductor.loop_node import all_ops

from test_span_overflow_gate import CASES, _shaped, compile_span_overflow
from unittest import mock


def _features_of_the_looped_ops(case):
    """``[(loop_trip, tiles_output_dim, tiles_reduction_dim, [(role, advances)])]`` of
    every shim op of ``case``, in graph order, read straight after the lowering;
    ``advances`` is whether that operand moves with the loop variable."""
    rows = []
    original = passes.tile_groups

    def spy(graph, groups, group_idx_offset=0):
        result = original(graph, groups=groups, group_idx_offset=group_idx_offset)
        for op in all_ops(graph.operations):
            info = getattr(op, "loop_member", None)
            if isinstance(op, ComputedBuffer) and getattr(
                info, "strict_loop_vars", False
            ):
                features = extract_op_features(op)
                rows.append(
                    (
                        features.loop_trip,
                        features.tiles_output_dim,
                        features.tiles_reduction_dim,
                        [(a.role, a.advances_with_loop_var) for a in features.args],
                    )
                )
        return result

    fn, shapes, limit, *_ = CASES[case]
    args = [_shaped(shape, seed) for seed, shape in enumerate(shapes)]
    with mock.patch.object(passes, "tile_groups", spy):
        compile_span_overflow(fn, args, limit)
    return rows


# case -> one row per looped op. Every op of an output-tiled group tiles an output dim,
# the copy-out included; the K-tiled matmul tiles its reduction dim and its combine,
# which only folds the partial result into the accumulator, tiles neither. An operand
# advances when the loop reads a new slice of it each trip: a graph input does, the tile
# one op leaves for the next (rewritten every trip) does not.
_OUT, _IN, _MOVES, _STAYS = "output", "input", True, False
EXPECTED_FEATURES = {
    "pointwise_copy_out": [
        (2, True, False, [(_OUT, _STAYS), (_IN, _MOVES), (_IN, _MOVES)]),
        (2, True, False, [(_OUT, _STAYS), (_IN, _STAYS)]),
    ],
    "pointwise_chain": [
        (2, True, False, [(_OUT, _STAYS), (_IN, _MOVES), (_IN, _MOVES)]),
        (2, True, False, [(_OUT, _STAYS), (_IN, _STAYS), (_IN, _MOVES)]),
        (2, True, False, [(_OUT, _STAYS), (_IN, _STAYS)]),
    ],
    "matmul_output_tiled_n": [
        (2, True, False, [(_OUT, _STAYS), (_IN, _STAYS), (_IN, _MOVES)]),
        (2, True, False, [(_OUT, _STAYS), (_IN, _STAYS)]),
    ],
    "matmul_pointwise_consumer": [
        (2, True, False, [(_OUT, _STAYS), (_IN, _STAYS), (_IN, _MOVES)]),
        (2, True, False, [(_OUT, _STAYS), (_IN, _STAYS), (_IN, _STAYS)]),
        (2, True, False, [(_OUT, _STAYS), (_IN, _STAYS)]),
    ],
    "matmul_k_tiled": [
        (8, False, True, [(_OUT, _STAYS), (_IN, _MOVES), (_IN, _MOVES)]),
        (8, False, False, [(_OUT, _STAYS), (_IN, _STAYS), (_IN, _STAYS)]),
    ],
}


@pytest.mark.parametrize("case", sorted(EXPECTED_FEATURES))
def test_the_cost_features_of_a_shim_op_report_the_loop_the_plan_built(case):
    assert _features_of_the_looped_ops(case) == EXPECTED_FEATURES[case]
