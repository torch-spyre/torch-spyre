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

"""The per-op loop record, ``ir.Operation.loop_info``.

``LoopInfo`` is stamped onto ``ComputedBuffer`` ops by the ``for_each_tile`` lowering
and by the loop-tiling lowering (span-overflow and allocator tiling), and read through
the ``loop_facts`` accessors. It is the last per-op copy of the loop's facts; item 9 of
#5863 replaces it with the enclosing ``LoopNode``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy


@dataclass
class LoopInfo:
    """Loop metadata stamped on a ``ComputedBuffer`` by a loop lowering.

    It holds the loop's own facts only. An access's per-trip advance is the loop
    variable's coefficient in the access's index (``loop_facts.read_advance``).

    Attributes
    ----------
    loop_group_id:
        Tuple encoding the nesting path, e.g. ``(0,)`` for an outermost
        group, ``(0, 0)`` for a nested group inside group 0.
    loop_count:
        List of trip counts, one per nesting level from outermost to
        innermost.  ``len(loop_count) == len(loop_group_id)`` always holds.
    loop_tiled_dims:
        List of lists, one sub-list per nesting level.  Each sub-list
        contains the ``data.ranges`` positional indices that are tiled at
        that level.  An empty sub-list means the op is loop-invariant at
        that level.
    loop_tiled_reduction_dims:
        List of lists, one sub-list per nesting level.  Each sub-list
        contains the ``data.reduction_ranges`` positional indices that are
        tiled at that level.  An empty sub-list means no reduction dim is
        tiled at that level.  Parallel to ``loop_tiled_dims``.
    strict_loop_vars:
        True on an op the loop-tiling lowering made: its ``dim_hints`` list the
        loop variable of EVERY level, so an op that lists none is a compiler bug
        and not a loop-invariant op. False on a ``for_each_tile`` op, where an op
        that lists no loop variable (the loop's fill, say) simply does not move
        with the loop. Read by ``loop_facts`` and the kernel's advance lookup.
    """

    loop_group_id: tuple[int, ...]
    loop_count: list[sympy.Expr]
    loop_tiled_dims: list[list[int]]
    loop_tiled_reduction_dims: list[list[int]] = field(default_factory=list)
    strict_loop_vars: bool = False
