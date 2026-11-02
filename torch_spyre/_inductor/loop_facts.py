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

"""Read-only loop facts over the per-op ``loop_info`` metadata.

Every function derives its answer from ``getattr(op, "loop_info", None)`` on
each call; nothing is cached and nothing is mutated.  This gives later stages
of the loop refactor one stable read API: when the storage behind it changes,
only this module changes, not its readers.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy
from torch._inductor.ir import ExternKernel, Operation
from torch._inductor.scheduler import BaseSchedulerNode, SchedulerNode


@dataclass(frozen=True)
class LoopLevel:
    """One nesting level of a counted loop enclosing an operation.

    ``group_path`` is the prefix of the op's ``loop_group_id`` up to and
    including this level.  ``induction`` is ``None``: the current metadata
    stores no induction variable, so readers must not depend on it yet.
    """

    group_path: tuple[int, ...]
    trip_count: sympy.Expr
    induction: sympy.Symbol | None
    tiled_dims: tuple[int, ...]
    tiled_reduction_dims: tuple[int, ...]


def _group_path(loop_info) -> tuple[int, ...]:
    """The ``loop_group_id`` of ``loop_info`` as a tuple; ``()`` if absent/malformed."""
    gid = getattr(loop_info, "loop_group_id", None)
    try:
        return tuple(gid) if gid else ()
    except TypeError:  # malformed id (e.g. a bare int)
        return ()


def _levels(loop_info) -> tuple[LoopLevel, ...]:
    """One ``LoopLevel`` per ``loop_group_id`` entry; ``()`` if none or malformed.

    ``loop_tiled_dims`` / ``loop_tiled_reduction_dims`` may be shorter than the
    path (``loop_tiled_reduction_dims`` defaults to ``[]``); a level with no
    entry gets ``()``.  A ``loop_count`` shorter than the path is genuinely
    malformed (a trip count cannot be invented) and yields ``()``.
    """
    if loop_info is None:
        return ()
    path = _group_path(loop_info)
    counts = loop_info.loop_count
    if not path or len(counts) < len(path):
        return ()
    tiled = loop_info.loop_tiled_dims
    tiled_red = loop_info.loop_tiled_reduction_dims

    def at(lists, i):
        return tuple(lists[i]) if i < len(lists) else ()

    return tuple(
        LoopLevel(
            group_path=path[: i + 1],
            trip_count=sympy.sympify(counts[i]),
            induction=None,
            tiled_dims=at(tiled, i),
            tiled_reduction_dims=at(tiled_red, i),
        )
        for i in range(len(path))
    )


def enclosing_loops(op: Operation) -> tuple[LoopLevel, ...]:
    """Loops enclosing ``op``, outermost first; ``()`` if none or malformed."""
    return _levels(getattr(op, "loop_info", None))


def loop_depth(op: Operation) -> int:
    """Number of counted loops enclosing ``op``."""
    return len(enclosing_loops(op))


def counted_loop_path(
    op: Operation, *, include_extern: bool = False
) -> tuple[int, ...]:
    """The counted-loop group path ``op`` runs in, outermost first; ``()`` if none.

    An ``ExternKernel`` keeps its ``loop_info`` (e.g. a hoisted loop-body
    constant) but runs once, outside the loop, so it yields ``()`` unless
    ``include_extern`` is set.  A malformed id yields ``()`` rather than
    raising.
    """
    if isinstance(op, ExternKernel) and not include_extern:
        return ()
    return _group_path(getattr(op, "loop_info", None))


def scheduler_enclosing_loops(node: BaseSchedulerNode) -> tuple[LoopLevel, ...]:
    """Levels of the first ``SchedulerNode`` in ``node`` that has a ``loop_info``.

    An ``ExternKernelSchedulerNode`` holds no ``SchedulerNode`` and so yields
    ``()``, which is how the scheduler already excludes extern kernels.
    """
    for snode in node.get_nodes():
        if isinstance(snode, SchedulerNode) and snode.node is not None:
            loop_info = getattr(snode.node, "loop_info", None)
            if loop_info is not None:
                return _levels(loop_info)
    return ()
