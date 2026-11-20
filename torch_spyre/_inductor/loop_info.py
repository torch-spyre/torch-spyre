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

"""Coarse-tiling loop metadata attached to ir.Operation objects.

``CoarseTileInfo`` is stamped onto ``ComputedBuffer`` ops by ``coarse_tile()``
and consumed by the scheduler, kernel codegen, and buffer-propagation pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sympy

if TYPE_CHECKING:
    pass


@dataclass
class CoarseTileInfo:
    """Loop metadata stamped on a ``ComputedBuffer`` by the ``for_each_tile`` lowering.

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
    """

    loop_group_id: tuple[int, ...]
    loop_count: list[sympy.Expr]
    loop_tiled_dims: list[list[int]]
    loop_tiled_reduction_dims: list[list[int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Op-metadata helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopShim:
    """TEMPORARY, deleted with ``loop_info.py`` at stage 9.

    What the loop-tiling lowering (``wsr/loop_tiling_lowering.py``) stamps as an
    op's ``loop_info`` instead of a ``CoarseTileInfo``, for the readers that have
    not yet moved to the ``LoopNode``. It holds the plan's own content: which loop
    the op is in, how many trips, and which dim is tiled at each level. It holds
    NOTHING about advances: the four advance fields of ``CoarseTileInfo`` are
    absent on purpose, so a reader that wants them either has a shim branch
    (``loop_facts`` derives a shim op's advances from the op's own index) or fails
    in a test instead of reading an empty list.
    """

    loop_group_id: tuple[int, ...]
    loop_count: list[sympy.Expr]
    loop_tiled_dims: list[list[int]]
    loop_tiled_reduction_dims: list[list[int]]


# What a reader tests ``loop_info`` against when it needs the loop's own facts (group
# id, count, tiled dims) and nothing about advances. ``isinstance(x, CoarseTileInfo)``
# alone reads as False for a ``LoopShim``; stage 9 deletes this with the shim.
COUNTED_LOOP_INFO = (CoarseTileInfo, LoopShim)
