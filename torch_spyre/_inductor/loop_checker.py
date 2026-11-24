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
4. An op that runs in a counted loop that a node covers (its ``loop_member``
   names that loop) sits inside that node: not outside it, and not in the body
   of another loop's node.
5. A node's body holds no op that is not a member of the loop anywhere in its
   body.
6. An op in nodes the pipeline built points at exactly those nodes and carries no
   ``loop_member``: once the nodes are built they own the loop facts, and a pass
   that adds an op to a loop says what it tiles with ``record_member``. An op whose
   index uses an enclosing loop's variable must hold a tiling entry in those nodes
   (an op without one reads as "not in the loop", which would drop its advance
   silently).

A loop no node covers is not checked: a loop-tiling lowering requests its
nodes and the pass loop builds them once the pass has returned. So rule 4 fires
only when a node already claims the op's loop, which is how a copy-back placed
after its loop by a flat index (#6101) is told from a loop not yet built.

Rules 4 and 5 compare each op's loop path with the node it sits in, using the
path that builds the nodes (:func:`counted_loop_path`, which gives an
``ExternKernel`` the empty path). They see only the leaf ops, so a node built by
hand from ops with no ``loop_member`` (the unit tests do) is not checked by them.
Rule 6 skips nodes the pipeline did not build.

What it cannot check: that a read which SHOULD advance carries the induction
variable. That was the padding bug (#5573); only a comparison against the
kernel finds it.

What it does not check, and why. A once-only op (an extern kernel that keeps
its ``loop_member``, such as a ``SpyreConstantFallback`` created in a body)
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
from .loop_facts import counted_loop_path
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


def _node_key(node: LoopNode, depth: int) -> tuple[int, ...] | None:
    """The group path prefix of ``node``'s members at ``depth``, from its first
    member that has one (the rule ``collapse`` uses)."""
    for leaf in all_ops(node.body):
        path = counted_loop_path(leaf)
        if len(path) > depth:
            return path[: depth + 1]
    return None


def _covered_keys(
    entries: list[ir.Operation], depth: int = 0, out: set[tuple[int, ...]] | None = None
) -> set[tuple[int, ...]]:
    """The group path key of every node in ``entries``, at every depth."""
    out = set() if out is None else out
    for entry in entries:
        if isinstance(entry, LoopNode):
            key = _node_key(entry, depth)
            if key is not None:
                out.add(key)
            _covered_keys(entry.body, depth + 1, out)
    return out


def _claimed(path: tuple[int, ...], covered: set[tuple[int, ...]]) -> bool:
    """Whether some node covers ``path`` or a prefix of it."""
    return any(path[: len(key)] == key for key in covered)


def _membership_problems(
    entries: list[ir.Operation],
    depth: int,
    enclosing: tuple[int, ...],
    problems: list[str],
    covered: set[tuple[int, ...]],
) -> None:
    """Rule 4 for ``entries``, which sit directly in a node whose key is
    ``enclosing`` (``()`` at the top level). ``covered`` holds every node's key."""
    for entry in entries:
        if isinstance(entry, LoopNode):
            key = _node_key(entry, depth)
            if key is not None:
                if key[: len(enclosing)] != enclosing:
                    problems.append(
                        f"{entry.get_name()} is a loop of group path {key} "
                        f"nested in a loop of group path {enclosing}"
                    )
                _membership_problems(entry.body, depth + 1, key, problems, covered)
            else:
                _membership_problems(
                    entry.body, depth + 1, enclosing, problems, covered
                )
            continue
        path = counted_loop_path(entry)
        if path and not enclosing:
            if not _claimed(path, covered):
                continue  # a flat loop no node covers
            problems.append(
                f"{entry.get_name()} runs in the counted loop {path} but sits "
                "outside every loop node"
            )
        elif path and path[: len(enclosing)] != enclosing:
            problems.append(
                f"{entry.get_name()} runs in the counted loop {path} but sits "
                f"in the body of the loop {enclosing}"
            )


def _non_members(entries: list[ir.Operation], depth: int, problems: list[str]) -> None:
    """Rule 5: an op in a node's body that is not a member of its loop.

    The scheduler runs a node's whole body every trip, so such an op would run
    every trip although it carries no loop. It is reported wherever it sits in the
    body. (Before stage 8.5 an op at the head or tail was allowed, because the
    flat view put an op inserted "before the loop" there and ``peel_non_members``
    moved it out at the hand-off; every edit now names its position, and that
    repair is gone.) An ``ExternKernel`` is allowed anywhere: a once-op stays in
    the body until the scheduler hoists it.
    """
    for entry in entries:
        if not isinstance(entry, LoopNode):
            continue
        key = _node_key(entry, depth)
        if key is not None:
            for op in entry.body:
                if not (
                    isinstance(op, LoopNode)
                    or bool(counted_loop_path(op))
                    or isinstance(op, ir.ExternKernel)
                ):
                    problems.append(
                        f"{op.get_name()} is in the body of {entry.get_name()} "
                        "but is not a member of the loop"
                    )
        _non_members(entry.body, depth + 1, problems)


def _node_chain_problems(
    op: ir.Operation, enclosing: tuple[LoopNode, ...], problems: list[str]
) -> None:
    """Rule 6: an op in nodes the pipeline built points at exactly those nodes
    (``_loop_chain``) and carries no ``loop_member``.

    The nodes own the loop facts once they are built (stage 9.4.3). A pass that adds
    an op to a loop calls a ``loop_node`` helper, which records the chain, and
    ``record_member``, which records what the op tiles. An op edited into a body
    by hand has no chain; one stamped after the build would carry a record nothing
    reads. Nodes built by hand (unit tests) are not checked.
    """
    if not enclosing or not all(n.facts_recorded for n in enclosing):
        return
    if tuple(getattr(op, "_loop_chain", ())) != enclosing:
        problems.append(
            f"{op.get_name()}: its recorded node chain "
            f"{[n.get_name() for n in getattr(op, '_loop_chain', ())]} is not the "
            f"nodes around it {[n.get_name() for n in enclosing]}"
        )
    elif hasattr(op, "loop_member"):
        problems.append(
            f"{op.get_name()}: carries a loop_member inside built nodes "
            f"{[n.get_name() for n in enclosing]}, which nothing reads; a pass that "
            "puts an op in a loop says what it tiles with record_member"
        )


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
        symbols = _index_symbols(op)
        stray = (symbols & all_inductions) - mine
        if stray:
            problems.append(
                f"{op.get_name()} indexes with {sorted(map(str, stray))}, not "
                "the induction variable of any loop enclosing it"
            )
        if (
            symbols & mine
            and all(n.facts_recorded for n in enclosing)
            and any(op.get_name() not in n.tiling for n in enclosing)
        ):
            problems.append(
                f"{op.get_name()} advances with the loop variable "
                f"{sorted(map(str, symbols & mine))} but its nodes hold no tiling for "
                "it, so it reads as not in the loop: a pass that puts an op that "
                "advances with a loop into it says what it tiles with record_member "
                "(or builds it with copy_op_metadata from a member)"
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
    for op, enclosing in _walk(operations):
        _node_chain_problems(op, enclosing, problems)
    _membership_problems(operations, 0, (), problems, _covered_keys(operations))
    _non_members(operations, 0, problems)
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
