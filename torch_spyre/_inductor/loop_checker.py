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

"""Structural checks on the ``LoopNode`` tree in ``graph.operations``.

A pass that adds, moves or rewrites ops in a loop can leave the tree wrong
without any error: an op in two bodies, an index that uses another loop's
variable, a carry whose update is gone. :func:`check_loops` runs between
passes and names the pass that produced the violation.

What it checks:

1. Every op appears exactly once in the tree.
2. Every loop variable in an op's read and write indices is the induction
   variable of a loop that encloses the op.
3. Every carry's update is an op in its node's body.

What it cannot check: that a read which SHOULD advance carries the induction
variable. That was the padding bug (#5573); only a comparison against the
kernel finds it.

What it does not check, and why. A once-only op (an extern kernel that keeps
its ``loop_info``, such as a ``SpyreConstantFallback`` created in a body)
stays in the body until the scheduler hoists it (``loop_scheduler``), so
"once-ops are outside the node" is not an invariant before that. On the loop
tests, a check for it fired on every graph that creates a constant in a body.

A carry's STORAGE is not required to be in the body either: it is the
carry's initial buffer, which is written before the loop.

Default: on under pytest or with ``TORCH_SPYRE_CHECK_LOOPS=1``; off otherwise.
:func:`~torch_spyre._inductor.loop_node.building_loop` suspends it while a loop
is half built.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from typing import Any

import sympy
from torch._inductor import ir
from torch._inductor.dependencies import MemoryDep

from .errors import Unsupported
from .loop_node import LoopNode, all_ops, in_building_loop


class LoopStructureError(Unsupported):
    """The ``LoopNode`` tree is inconsistent. ``where`` names the pass."""


def checking_enabled() -> bool:
    flag = os.environ.get("TORCH_SPYRE_CHECK_LOOPS")
    if flag is not None:
        return flag == "1"
    return "pytest" in sys.modules


def _walk(
    operations: list[ir.Operation], enclosing: tuple[LoopNode, ...] = ()
) -> Iterator[tuple[ir.Operation, tuple[LoopNode, ...]]]:
    for op in operations:
        if isinstance(op, LoopNode):
            yield from _walk(op.body, (*enclosing, op))
        else:
            yield op, enclosing


def _inductions(operations: list[ir.Operation]) -> set[sympy.Symbol]:
    found: set[sympy.Symbol] = set()
    for op in operations:
        if isinstance(op, LoopNode):
            if op.induction is not None:
                found.add(op.induction)
            found |= _inductions(op.body)
    return found


def _index_symbols(op: ir.Operation) -> set[sympy.Symbol]:
    """Free symbols of every memory dependency's index (a ``StarDep`` has none)."""
    used: set[sympy.Symbol] = set()
    try:
        rw = op.get_read_writes()
    except Exception:  # an op that cannot trace itself has nothing to check
        return used
    for dep in (*rw.reads, *rw.writes):
        if isinstance(dep, MemoryDep):
            used |= dep.index.free_symbols
    return used


def find_violations(operations: list[ir.Operation]) -> list[str]:
    """Every structural violation in ``operations``, as messages."""
    problems: list[str] = []
    all_inductions = _inductions(operations)

    seen: set[int] = set()
    for op, enclosing in _walk(operations):
        if id(op) in seen:
            problems.append(f"{op.get_name()} appears more than once in the tree")
        seen.add(id(op))
        mine = {n.induction for n in enclosing if n.induction is not None}
        stray = (_index_symbols(op) & all_inductions) - mine
        if stray:
            problems.append(
                f"{op.get_name()} indexes with {sorted(map(str, stray))}, not "
                "the induction variable of any loop enclosing it"
            )

    def carries(entries: list[ir.Operation]) -> None:
        for entry in entries:
            if not isinstance(entry, LoopNode):
                continue
            names = {o.get_name() for o in all_ops(entry.body)}
            for record in entry.carries:
                if record.update_name not in names:
                    problems.append(
                        f"{entry.get_name()}: carry update {record.update_name!r} "
                        "is not an op in its body"
                    )
            carries(entry.body)

    carries(operations)
    return problems


def check_loops(graph: Any, *, where: str) -> None:
    """Raise :class:`LoopStructureError` if the tree is inconsistent.

    A no-op when checking is off, while a loop is being built, or when the
    graph holds no ``LoopNode``.
    """
    if not checking_enabled() or in_building_loop():
        return
    operations = getattr(graph.operations, "real", graph.operations)
    if not any(isinstance(op, LoopNode) for op in operations):
        return
    problems = find_violations(operations)
    if problems:
        raise LoopStructureError(
            f"after {where}: the loop structure is inconsistent:\n  "
            + "\n  ".join(problems)
        )
