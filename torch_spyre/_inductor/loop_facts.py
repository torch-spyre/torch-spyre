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

"""Read-only loop facts: the one read API for "which loops is this op in".

Once the pipeline has built an op's ``LoopNode``s they own the facts (group path,
trip count, tiled dims, ``strict_loop_vars``) and every function here asks them
(through ``op._loop_chain``). An op no node holds yet, a loop still being lowered,
answers from its ``loop_member``, the lowering's build-time record
(:func:`pending_levels`). Nothing is cached and nothing is mutated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ExternKernel, Operation

from .errors import Unsupported


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


def _group_path(loop_member) -> tuple[int, ...]:
    """The ``loop_group_id`` of ``loop_member`` as a tuple; ``()`` if absent/malformed."""
    gid = getattr(loop_member, "loop_group_id", None)
    try:
        return tuple(gid) if gid else ()
    except TypeError:  # malformed id (e.g. a bare int)
        return ()


def _levels(loop_member) -> tuple[LoopLevel, ...]:
    """One ``LoopLevel`` per ``loop_group_id`` entry; ``()`` if none or malformed.

    ``loop_tiled_dims`` / ``loop_tiled_reduction_dims`` may be shorter than the
    path (``loop_tiled_reduction_dims`` defaults to ``[]``); a level with no
    entry gets ``()``.  A ``loop_count`` shorter than the path is genuinely
    malformed (a trip count cannot be invented) and yields ``()``.
    """
    if loop_member is None:
        return ()
    path = _group_path(loop_member)
    if not path:
        return ()
    counts = getattr(loop_member, "loop_count", None) or ()
    if len(counts) < len(path):
        return ()
    tiled = getattr(loop_member, "loop_tiled_dims", None) or ()
    tiled_red = getattr(loop_member, "loop_tiled_reduction_dims", None) or ()

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


def pending_levels(op: Operation) -> tuple[LoopLevel, ...]:
    """The levels ``op.loop_member`` (the lowering's build-time record) describes;
    ``()`` if none or malformed. Readers use :func:`enclosing_loops`; this is the
    record itself, for an op no node holds yet."""
    return _levels(getattr(op, "loop_member", None))


def _in_built_node(op: Operation) -> bool:
    """True if ``op`` is a MEMBER of the loops around it: it sits in built nodes AND
    each of them holds its tiling entry. Being placed in a body is not enough: an LX
    clone the allocator puts beside a body consumer runs every trip but is not part
    of the loop's computation, and treating it as looped made the same-tile-group
    relayout offer a candidate for it (a wrong plan on the eager SDPA graphs,
    ``test_gather_attention``). An entry exists exactly for the ops that had a
    ``loop_member`` stamp."""
    chain = getattr(op, "_loop_chain", ())
    if not (isinstance(chain, tuple) and chain):
        return False
    name = op.get_name()
    return all(node.facts_recorded and name in node.tiling for node in chain)


def enclosing_loops(op: Operation) -> tuple[LoopLevel, ...]:
    """Loops enclosing ``op``, outermost first; ``()`` if none or malformed.

    Read from the ``LoopNode``s around ``op`` once the pipeline has built them
    (:func:`node_levels`; the loop checker, rule 6, keeps them equal to the stamp).
    An op no node holds yet (a loop being lowered, before its nodes are built) or no
    longer holds (hoisted out of its loop) answers from its ``loop_member``.
    """
    if _in_built_node(op):
        return node_levels(op)
    return pending_levels(op)


def node_levels(op: Operation) -> tuple[LoopLevel, ...]:
    """The same levels as :func:`enclosing_loops`, read from the ``LoopNode``s
    around ``op`` (``op._loop_chain``, set where the node is built and by the
    ``loop_node`` edit helpers) instead of from ``op.loop_member``.

    ``induction`` is the node's, where ``enclosing_loops`` reports ``None``. ``()``
    for an op no node holds.
    """
    chain = getattr(op, "_loop_chain", ())
    if not chain:
        return ()
    name = op.get_name()
    levels = []
    for i, node in enumerate(chain):
        tiled, tiled_reduction = node.tiling.get(name, ((), ()))
        levels.append(
            LoopLevel(
                group_path=tuple(n.group_id for n in chain[: i + 1]),
                trip_count=sympy.sympify(node.trip_count),
                induction=node.induction,
                tiled_dims=tiled,
                tiled_reduction_dims=tiled_reduction,
            )
        )
    return tuple(levels)


def strict_loop_vars(op: Operation) -> bool:
    """True if the loop-tiling lowering made the loops around ``op``: every member
    lists the loop variable of every level in its ``dim_hints``. From the nodes once
    built, from the stamp for an op no node holds."""
    if _in_built_node(op):
        return any(node.strict_loop_vars for node in op._loop_chain)  # type: ignore[attr-defined]
    return bool(getattr(getattr(op, "loop_member", None), "strict_loop_vars", False))


def is_looped(op: Operation) -> bool:
    """True if ``op`` is a member of a counted loop: its nodes hold its tiling (or, for
    an op no node holds yet, it carries the lowering's ``loop_member``). An op merely
    placed in a loop body, such as an LX clone beside a consumer, is not a member."""
    return bool(enclosing_loops(op))


def loop_depth(op: Operation) -> int:
    """Number of counted loops enclosing ``op``."""
    return len(enclosing_loops(op))


def counted_loop_path(
    op: Operation, *, include_extern: bool = False
) -> tuple[int, ...]:
    """The counted-loop group path ``op`` runs in, outermost first; ``()`` if none.

    An ``ExternKernel`` keeps its ``loop_member`` (e.g. a hoisted loop-body
    constant) but runs once, outside the loop, so it yields ``()`` unless
    ``include_extern`` is set.  A malformed id yields ``()`` rather than
    raising.
    """
    if isinstance(op, ExternKernel) and not include_extern:
        return ()
    if _in_built_node(op):
        return tuple(node.group_id for node in op._loop_chain)  # type: ignore[attr-defined]
    return _group_path(getattr(op, "loop_member", None))


# ---------------------------------------------------------------------------
# Per-read and output advance (read from the index)
# ---------------------------------------------------------------------------


def _memory_reads(op) -> list[MemoryDep]:
    return [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]


def _loop_coefficient(index: sympy.Expr, variable: sympy.Symbol) -> sympy.Expr:
    """The coefficient of the loop variable ``variable`` in ``index``.

    (The lowering computes the step it writes into the index separately, as the delta
    of the undivided index over one trip: ``wsr.loop_tiling_lowering._tile_step``. This
    reads that term back from the final index.)

    ``Expr.coeff`` is not a safe reading of "how far one trip moves this access": it
    returns 0 for an unexpanded product (``32*(u0 + 1)``) and for any use of the
    variable it cannot see through (``floor(u0/2)``, ``Mod(u0, 2)``). Read as "no
    advance", those would be silently wrong once the stored record, which makes such
    a disagreement loud, is deleted. So the index is evaluated at 0, 1 and 2:

    * it must be affine in the variable (the second difference is zero), and
    * the step must be a number (a symbolic step cannot be projected to device
      elements),

    otherwise this raises ``Unsupported``. A variable that does not occur in the
    index has coefficient 0. The loop variable is an integer, so ``floor(8*u0)`` is
    ``8*u0`` and has coefficient 8.

    Lives here (``spyre_kernel`` imports it) because the advance accessors below
    read a shim op's advances with it.
    """
    if variable not in index.free_symbols:
        return sympy.S.Zero
    at = [sympy.expand(index.xreplace({variable: k})) for k in (0, 1, 2)]
    if sympy.simplify(at[2] - 2 * at[1] + at[0]) != 0:
        raise Unsupported(
            f"the loop variable {variable} is used non-affinely in the index "
            f"{index}; a per-trip advance cannot be read from it"
        )
    step = sympy.simplify(at[1] - at[0])
    if not step.is_number:
        raise Unsupported(
            f"the loop variable {variable} has the symbolic step {step} in the "
            f"index {index}; it cannot be projected to device elements"
        )
    return step


def _index_advance(op, dep: MemoryDep) -> tuple[sympy.Expr, ...]:
    """An op's advance per level: the coefficient of each loop variable in
    ``dep.index``, in host elements. The loop variables are the ``dim_hints``
    entries that carry a ``loop_var_range``, outermost first; an ordinary
    iteration-symbol hint is not one.

    An op of a ``for_each_tile`` loop that lists no loop variable has none in its
    index, so nothing it reads or writes moves with the loop: zero at every level.
    (The loop's fill and other once-only members are like that.) An op the
    loop-tiling lowering made (``strict_loop_vars``) always lists the variable of every
    level, so one that lists none is a compiler bug, not a zero: that raises, as it
    did before part A. Otherwise the count must equal the loop depth, else the
    pairing of variable to level is unknown and nothing here can guess it.
    """
    variables = [
        h.loop_var
        for h in getattr(op, "dim_hints", None) or ()
        if h.loop_var is not None and h.loop_var_range is not None
    ]
    depth = loop_depth(op)
    if not variables:
        if strict_loop_vars(op):
            raise RuntimeError(
                f"{op.get_operation_name() if hasattr(op, 'get_operation_name') else op} "
                f"is made by the loop-tiling lowering but lists none of its {depth} "
                "loop variable(s): its advances cannot be read from the index"
            )
        return (sympy.S.Zero,) * depth
    if len(variables) != depth:
        raise RuntimeError(
            f"{op.get_operation_name() if hasattr(op, 'get_operation_name') else op} "
            f"has {depth} loop level(s) but {len(variables)} loop variable "
            "hint(s): its advances cannot be read from the index"
        )
    return tuple(_loop_coefficient(dep.index, v) for v in variables)


def _first_memory_write(op) -> MemoryDep | None:
    return next(
        (d for d in op.get_read_writes().writes if isinstance(d, MemoryDep)), None
    )


def read_advance(op: Operation, dep: MemoryDep) -> tuple[sympy.Expr, ...]:
    """Host-element advance per trip of read ``dep``, one per level, outermost first.

    ``sympy.S.Zero`` for a level at which the read does not advance; ``()`` when
    ``op`` has no ``loop_member`` or ``dep`` is not one of its ``MemoryDep`` reads.
    The value is in host elements: projecting it to the device
    (``views.tiling_expr_to_device_expr``) needs a concrete layout and stays with
    the caller. Read from the index: the coefficient of the level's loop variable
    (``_loop_coefficient``, which refuses a non-affine use).
    """
    if not is_looped(op) or not isinstance(dep, MemoryDep):
        return ()
    if dep not in _memory_reads(op):
        return ()
    return _index_advance(op, dep)


def output_advance(op: Operation) -> tuple[sympy.Expr, ...]:
    """Host-element advance per trip of ``op``'s write, one per level.

    The write is ``op``'s first ``MemoryDep`` write, as in the kernel.  ``()``
    when ``op`` has no ``loop_member`` or no ``MemoryDep`` write.
    """
    if not is_looped(op):
        return ()
    dep = _first_memory_write(op)
    if dep is None:
        return ()
    return _index_advance(op, dep)


def read_advances_anywhere(
    op: Operation, dep: MemoryDep, *, reads: Sequence[MemoryDep] | None = None
) -> bool:
    """True if read ``dep`` of ``op`` advances at any level.

    ``reads`` is ``op``'s ``MemoryDep`` reads in order, for a caller that already
    holds them (e.g. from ``pass_utils.op_read_writes``); without it they are
    re-derived from ``op.get_read_writes()``.
    """
    if not is_looped(op) or not isinstance(dep, MemoryDep):
        return False
    if dep not in (_memory_reads(op) if reads is None else reads):
        return False
    return any(a != 0 for a in _index_advance(op, dep))


def output_advances_anywhere(op: Operation) -> bool:
    """True if ``op``'s write advances at any level (``False`` for an op with no
    ``MemoryDep`` write).

    The loop fact of ``scratchpad.utils._is_tiled_advancing``, without its
    ``FixedTiledLayout`` gate (a layout question, not a loop fact).
    """
    if not is_looped(op):
        return False
    dep = _first_memory_write(op)
    return dep is not None and any(a != 0 for a in _index_advance(op, dep))


def advancing_levels(op: Operation) -> frozenset[int]:
    """The loop levels at which any read or the write of ``op`` moves.

    The kernel names a loop-level symbol for a level only if something at that
    level advances; the answer comes from the index. Empty for an op that is in no
    loop.
    """
    if not is_looped(op):
        return frozenset()
    levels: set[int] = set()
    for dep in _memory_reads(op):
        levels.update(i for i, a in enumerate(_index_advance(op, dep)) if a != 0)
    write = _first_memory_write(op)
    if write is not None:
        levels.update(i for i, a in enumerate(_index_advance(op, write)) if a != 0)
    return frozenset(levels)
