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

"""Compile-time decomposition of collective operations into primitives.

This module is intentionally free of Inductor IR dependencies so it can be
unit-tested independently. It returns plain lists of (op_type, params) tuples
that the lowering layer translates into IR nodes.
"""


def decompose_allreduce(
    reduce_op: str = "sum",
    group_name: str = "default",
    root_rank: int = 0,
) -> list[tuple[str, dict]]:
    """Decompose allreduce into reduce + broadcast.

    allreduce(x, op, group) →
        y = reduce(x, op, dst_rank=root_rank, group)
        z = broadcast(y, src_rank=root_rank, group)

    Returns a list of (op_type, params) pairs describing the decomposition.
    """
    return [
        (
            "reduce",
            {
                "reduce_op": reduce_op,
                "dst_rank": root_rank,
                "group_name": group_name,
            },
        ),
        (
            "broadcast",
            {
                "src_rank": root_rank,
                "group_name": group_name,
            },
        ),
    ]
