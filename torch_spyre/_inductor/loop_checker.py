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

from .logging_utils import get_inductor_logger
from .loop_node import LoopNode, all_ops, in_building_loop

logger = get_inductor_logger("loop_checker")


class LoopStructureError(RuntimeError):
    """The ``LoopNode`` tree is inconsistent: a compiler bug, not a user error.

    Not a subclass of ``Unsupported``: that reads "the Spyre backend does not
    support ...", which tells a user their program is at fault when a pass broke
    an invariant, and a test that asserts ``Unsupported`` around a whole compile
    would pass on this error and hide the regression.

    ``where`` is the name of the pass that left the tree inconsistent;
    ``problems`` are the individual violations.
    """

    def __init__(self, where: str, problems: list[str]) -> None:
        self.where = where
        self.problems = list(problems)
        super().__init__(
            f"after {where}: the loop structure is inconsistent "
            "(a compiler bug; set TORCH_SPYRE_CHECK_LOOPS=0 to skip this check):\n  "
            + "\n  ".join(self.problems)
        )


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


def checking_enabled() -> bool:
    """``TORCH_SPYRE_CHECK_LOOPS`` decides when it is set; otherwise on under pytest.

    An unrecognised value raises, so a typo cannot silently turn the check off.
    """
    flag = os.environ.get("TORCH_SPYRE_CHECK_LOOPS")
    if flag is not None:
        value = flag.strip().lower()
        if value in _TRUE:
            return True
        if value in _FALSE:
            return False
        raise ValueError(
            f"TORCH_SPYRE_CHECK_LOOPS={flag!r}: expected one of "
            f"{sorted(_TRUE | _FALSE - {''})}"
        )
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
    """Free symbols of every memory dependency's index (a ``StarDep`` has none).

    Calls ``get_read_writes()`` and not the ``pass_utils.op_read_writes`` memo
    on purpose: the memo is invalidated by hand at each site that rewrites an
    op, so a pass that forgot to would leave a stale entry, and a checker that
    read it would miss exactly the violation that pass just created.
    """
    used: set[sympy.Symbol] = set()
    try:
        rw = op.get_read_writes()
    except Exception as exc:  # a checker must not crash a compile
        logger.debug(
            "loop_checker: %s cannot trace its reads and writes (%s: %s); "
            "its indices are not checked",
            op.get_name(),
            type(exc).__name__,
            exc,
        )
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
        if not all_inductions:
            # No loop has an induction variable, so no index can use a stray
            # one: skip tracing every op's dependencies.
            continue
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
        raise LoopStructureError(where, problems)
