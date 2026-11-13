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

from collections.abc import Sequence
from dataclasses import dataclass

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ExternKernel, Operation
from torch._inductor.scheduler import BaseSchedulerNode, SchedulerNode

from .errors import Unsupported
from .loop_info import LoopShim


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


# ---------------------------------------------------------------------------
# Per-read and output advance
# ---------------------------------------------------------------------------


def _host_dim_to_index_symbol(ir_node, dim: int) -> sympy.Symbol:
    """``SpyreKernel._host_dim_to_index_symbol``: the kernel's own mapping.

    Imported at call time so that this module, which ``scratchpad`` readers
    import, does not import ``spyre_kernel`` at load time.
    """
    from .spyre_kernel import SpyreKernel

    return SpyreKernel._host_dim_to_index_symbol(ir_node, dim)


def _memory_reads(op) -> list[MemoryDep]:
    return [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]


def _read_record(op, dep: MemoryDep, reads: Sequence[MemoryDep] | None = None):
    """``(loop_info, dep_idx)`` for ``dep`` as a read of ``op``, else ``None``.

    The kernel gives the k-th read of a (mutation-resolved) buffer name the
    record of the k-th ``MemoryDep`` read with that name, which is the
    record at that read's own position among the ``MemoryDep`` reads; so
    ``dep``'s record is the one at ``dep``'s position.  A ``dep`` equal to two
    reads takes the first.  ``reads``, when given, is ``op``'s ``MemoryDep``
    reads in order, used in place of re-deriving them.
    """
    loop_info = getattr(op, "loop_info", None)
    if loop_info is None or not isinstance(dep, MemoryDep):
        return None
    if reads is None:
        reads = _memory_reads(op)
    if dep not in reads:
        return None
    return loop_info, reads.index(dep)


def _at(lists, i):
    return lists[i] if lists is not None and i < len(lists) else []


def _per_level_advance(
    ir_node, dep: MemoryDep, per_level_dims, squeezed_per_level, depth: int
) -> tuple[sympy.Expr, ...]:
    """One host-element advance per level, built as the kernel builds it.

    Mirrors ``SpyreKernel._general_tile_advance`` before its device
    projection: substitute ``extent * s`` for each tiled dim's index symbol,
    zero every other free symbol of ``dep.index``, add ``s * extent *
    host_stride`` for each squeezed pair.  ``s`` is a fresh symbol, and the
    reported value is the term at ``s = 1`` minus the term at ``s = 0``: the
    per-trip step, without the constant offset (e.g. a slice start) that
    ``dep.index`` may carry and the kernel's projection drops.  For a term
    linear in ``s`` (every term the ``for_each_tile`` corpus produces) this
    is the coefficient the kernel projects; a term non-linear in ``s`` (e.g.
    through ``floor``) has no single per-trip step, and the kernel's
    projection of one is not reproduced here.
    """
    n = max(depth, len(per_level_dims or ()), len(squeezed_per_level or ()))
    out: list[sympy.Expr] = []
    for level in range(n):
        pairs = _at(per_level_dims, level)
        squeezed = _at(squeezed_per_level, level)
        if not pairs and not squeezed:
            out.append(sympy.S.Zero)
            continue
        s = sympy.Dummy("level")
        term = sympy.S.Zero
        if pairs:
            subs = {
                _host_dim_to_index_symbol(ir_node, d): extent * s for d, extent in pairs
            }
            subs.update(
                {sym: sympy.S.Zero for sym in dep.index.free_symbols if sym not in subs}
            )
            term += dep.index.subs(subs)
        for host_stride, extent in squeezed:
            term += s * extent * host_stride
        out.append(sympy.simplify(term.subs(s, 1) - term.subs(s, 0)))
    return tuple(out)


def _loop_coefficient(index: sympy.Expr, variable: sympy.Symbol) -> sympy.Expr:
    """The coefficient of the loop variable ``variable`` in ``index``.

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
    at = [sympy.expand(index.subs(variable, k)) for k in (0, 1, 2)]
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


def _shim_advance(op, dep: MemoryDep) -> tuple[sympy.Expr, ...]:
    """A shim op's advance per level: the coefficient of each loop variable in
    ``dep.index``. The loop variables are the ``dim_hints`` entries that carry a
    ``loop_var_range``, outermost first; an ordinary iteration-symbol hint is not
    one. The count must equal the loop depth, else the pairing of variable to level
    is unknown and nothing here can guess it."""
    variables = [
        h.loop_var
        for h in getattr(op, "dim_hints", None) or ()
        if h.loop_var is not None and h.loop_var_range is not None
    ]
    depth = _depth(op.loop_info)
    if len(variables) != depth:
        raise RuntimeError(
            f"{op.get_operation_name() if hasattr(op, 'get_operation_name') else op} "
            f"has {depth} loop level(s) but {len(variables)} loop variable "
            "hint(s): its advances cannot be read from the index"
        )
    return tuple(_loop_coefficient(dep.index, v) for v in variables)


def _depth(loop_info) -> int:
    return len(_group_path(loop_info))


def read_advance(op: Operation, dep: MemoryDep) -> tuple[sympy.Expr, ...]:
    """Host-element advance per trip of read ``dep``, one per level, outermost first.

    ``sympy.S.Zero`` for a level at which the read does not advance; ``()``
    when ``op`` has no ``loop_info`` or ``dep`` is not one of its
    ``MemoryDep`` reads.  The value is in host elements: projecting it to the
    device (``views.tiling_expr_to_device_expr``) needs a concrete layout and
    stays with the caller.

    For a term that is linear in the level's loop variable (every term the
    ``for_each_tile`` lowering produces today) this is the coefficient the
    kernel projects.  For a non-linear term the two can differ:
    ``4096 * floor(8 * s)`` gives ``32768`` here (the term at ``s = 1`` minus
    the term at ``s = 0``), while ``SpyreKernel._general_tile_advance`` takes the
    coefficient of ``s``, which is ``0``, and so emits no advance.  Do not
    substitute one for the other without deciding which is right.
    """
    found = _read_record(op, dep)
    if found is None:
        return ()
    loop_info, idx = found
    if isinstance(loop_info, LoopShim):
        return _shim_advance(op, dep)
    tiled = loop_info.tiled_dims_per_read
    if idx >= len(tiled):
        # The kernel emits no advance for a read past the record.
        return (sympy.S.Zero,) * _depth(loop_info)
    squeezed = getattr(loop_info, "squeezed_advance_per_read", None) or []
    return _per_level_advance(
        op, dep, tiled[idx], _at(squeezed, idx), _depth(loop_info)
    )


def _first_memory_write(op) -> MemoryDep | None:
    return next(
        (d for d in op.get_read_writes().writes if isinstance(d, MemoryDep)), None
    )


def output_advance(op: Operation) -> tuple[sympy.Expr, ...]:
    """Host-element advance per trip of ``op``'s write, one per level.

    The write is ``op``'s first ``MemoryDep`` write, as in the kernel.  ``()``
    when ``op`` has no ``loop_info`` or no ``MemoryDep`` write.  The same
    caveat as :func:`read_advance` applies to a term that is not linear in the
    level's loop variable.
    """
    loop_info = getattr(op, "loop_info", None)
    if loop_info is None:
        return ()
    dep = _first_memory_write(op)
    if dep is None:
        return ()
    if isinstance(loop_info, LoopShim):
        return _shim_advance(op, dep)
    return _per_level_advance(
        op,
        dep,
        loop_info.output_tiled_dims,
        getattr(loop_info, "squeezed_advance_output", None) or [],
        _depth(loop_info),
    )


def read_advances_anywhere(
    op: Operation, dep: MemoryDep, *, reads: Sequence[MemoryDep] | None = None
) -> bool:
    """True if read ``dep`` of ``op`` has a tiled or squeezed entry at any level.

    Answers from the record, as ``scratchpad.utils._is_read_advancing_anywhere``
    does, not from ``any(a != 0 for a in read_advance(op, dep))``: the two
    differ for a record whose substituted term is zero (a tiled dim whose
    symbol is absent from ``dep.index``), and the existing readers treat such
    a read as advancing.  A read past the end of ``tiled_dims_per_read`` is
    not advancing.

    ``reads`` is ``op``'s ``MemoryDep`` reads in order, for a caller that
    already holds them (e.g. from ``pass_utils.op_read_writes``); without it
    they are re-derived from ``op.get_read_writes()``.
    """
    found = _read_record(op, dep, reads)
    if found is None:
        return False
    loop_info, idx = found
    if isinstance(loop_info, LoopShim):
        return any(a != 0 for a in _shim_advance(op, dep))
    if idx >= len(loop_info.tiled_dims_per_read):
        return False
    if any(loop_info.tiled_dims_per_read[idx]):
        return True
    squeezed = getattr(loop_info, "squeezed_advance_per_read", None) or []
    return idx < len(squeezed) and any(squeezed[idx])


def output_advances_anywhere(op: Operation) -> bool:
    """True if ``op``'s write has a tiled or squeezed entry at any level.

    The record test of ``scratchpad.utils._is_tiled_advancing``, without its
    ``FixedTiledLayout`` gate (a layout question, not a loop fact).  Like
    ``read_advances_anywhere`` it reads the record, not the substituted term,
    and, like it, does not ask whether ``op`` has a ``MemoryDep`` write.
    """
    loop_info = getattr(op, "loop_info", None)
    if loop_info is None:
        return False
    if isinstance(loop_info, LoopShim):
        dep = _first_memory_write(op)
        return dep is not None and any(a != 0 for a in _shim_advance(op, dep))
    if any(dims for dims in loop_info.output_tiled_dims):
        return True
    squeezed = getattr(loop_info, "squeezed_advance_output", None) or []
    return any(level for level in squeezed)


def shim_advancing_levels(op: Operation) -> frozenset[int]:
    """The loop levels at which any read or the write of a ``LoopShim`` op moves.

    The kernel names a loop-level symbol for a level only if something at that
    level advances.  A ``CoarseTileInfo`` op answers that from its stored lists;
    a shim op has none, so the answer comes from the index.  Empty for an op
    that does not carry a shim.
    """
    if not isinstance(getattr(op, "loop_info", None), LoopShim):
        return frozenset()
    levels: set[int] = set()
    for dep in _memory_reads(op):
        levels.update(i for i, a in enumerate(_shim_advance(op, dep)) if a != 0)
    write = _first_memory_write(op)
    if write is not None:
        levels.update(i for i, a in enumerate(_shim_advance(op, write)) if a != 0)
    return frozenset(levels)
