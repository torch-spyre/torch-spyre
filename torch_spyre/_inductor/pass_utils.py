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

import io
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Sequence, TypeVar, Union

import regex
import torch
import sympy
from sympy import Expr, Symbol
from torch._inductor.ir import (
    Buffer,
    ComputedBuffer,
    FixedLayout,
    Loops,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
)
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.graph import GraphLowering
from torch._inductor.dependencies import MemoryDep, ReadWrites, StarDep, is_indirect
from torch._inductor.virtualized import V
from torch_spyre._C import SpyreTensorLayout, get_elem_in_stick
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.op_spec import IndirectAccess, TensorWorkDivision

from . import config
from .core_mapping import core_to_slice_mapping
from .constants import (
    ELIDED_COPY_BACK_ATTR,
    KEEP_BY_INDEX_OP,
    MATMUL_REDUCTION_OPS,
    TOPK_OPS,
)
from .ir import FixedTiledLayout, SpyreConstantFallback
from .logging_utils import get_inductor_logger
from .loop_info import copy_op_metadata
from .provenance import preserve_provenance
from .views import (
    AlignmentInputs,
    build_alignment_inputs,
    compute_coordinates,
    matching_dim,
)

# PyTorch's default lower bound for size symbols (sizes 0/1 are specialised).
_SHAPE_ENV_DEFAULT_LOWER = 2
logger = get_inductor_logger("pass_utils")


class SchedNodeArg(NamedTuple):
    dep: MemoryDep
    layout: "FixedTiledLayout"


@dataclass(frozen=True)
class AlignmentAccess:
    """One tensor access before its coordinates are normalized for codegen."""

    device_layout: Any
    index: sympy.Expr


def _fixed_read_layout(buf) -> "FixedTiledLayout":
    layout = buf.get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        # Reading real_layout() through a mutation layout is only valid once
        # the target buffer's own layout is a committed FixedTiledLayout.
        # Three producers of this shape:
        #  - the copy-back elision optimization (propagate_layouts.py), which
        #    stamps ELIDED_COPY_BACK_ATTR on the producer; or
        #  - coarse_tile.py's nested output-dim + reduction-dim tiling
        #    (_insert_reduction_copy_op), which mutates directly into a
        #    SpyreEmptyFallback accumulator (accum_tile) — a legitimate
        #    in-group consumer (e.g. the next outer-tile iteration's copy-in)
        #    reads that copy op's own output the same way an ordinary
        #    producer's output would be read; or
        #  - coarse_tile.py's copy_out path for a MutationLayoutSHOULDREMOVE op
        #    whose target is a locally-created graph-output buffer (e.g.
        #    copy_forced(src, c) where c is returned directly) -- _insert_copy_op's
        #    inserted coarse_tile_copy_* op reads the mutation op's own output
        #    the same way. The mutation target there is an ordinary
        #    ComputedBuffer, not a SpyreEmptyFallback, so this case is
        #    recognized by layout alone.
        mutation_target = layout.get_buffer()
        is_elided = getattr(buf, ELIDED_COPY_BACK_ATTR, False)
        is_committed_target = isinstance(mutation_target.get_layout(), FixedTiledLayout)
        if not (is_elided or is_committed_target):
            raise RuntimeError(f"unexpected mutation layout on read buffer {buf}")
        layout = layout.real_layout()
    if not isinstance(layout, FixedTiledLayout):
        raise RuntimeError(f"{buf} does not have FixedTiledLayout")
    return layout


def get_mem_deps(n: SchedulerNode) -> list[SchedNodeArg]:
    res: list[SchedNodeArg] = []
    for arg in n.read_writes.reads:
        if isinstance(arg, MemoryDep):
            buf = V.graph.get_buffer(arg.name)
            res.append(SchedNodeArg(arg, _fixed_read_layout(buf)))
    return res


def op_read_writes(op: Operation) -> ReadWrites:
    """``op.get_read_writes()`` memoized on the op instance.

    ``ComputedBuffer.get_read_writes`` re-runs sympy dependency extraction on
    every call and is not cached upstream, yet its result does not depend on the
    only thing the LX planner mutates (``op_it_space_splits``). The scratchpad
    pass calls it hundreds of times, so we cache it under a private key only this
    helper reads -- a non-planner caller (e.g. later-pass codegen) still goes
    through the real method.
    """
    rw = op.__dict__.get("_ts_cached_read_writes")
    if rw is None:
        rw = op.get_read_writes()
        op.__dict__["_ts_cached_read_writes"] = rw
    return rw


def op_short_name(op: Operation) -> str:
    """Resolve an operation's short name, including fused FX origins."""

    for fx_node in (getattr(op, "origin_node", None), *getattr(op, "origins", ())):
        target = getattr(fx_node, "target", None)
        for attr in ("_opname", "__name__", "name"):
            if name := getattr(target, attr, None):
                return str(name)
    return "None"


def invalidate_op_read_writes(op: Operation) -> None:
    """Drop any memoized :func:`op_read_writes` result for ``op``.

    Call this immediately after mutating an op's dependencies in place -- e.g.
    swapping a load name in its ``inner_fn`` -- so the next
    :func:`op_read_writes` re-traces instead of returning stale reads/writes.
    The memo is keyed on the op instance and is otherwise never invalidated: its
    result is independent of ``op_it_space_splits`` (the only thing the LX
    planner normally mutates), so a plain split change needs no invalidation.
    """
    op.__dict__.pop("_ts_cached_read_writes", None)


def concretize_expr(expr: Union[Expr, int]) -> int:
    """Concretize a sympy expression to a Python int.

    Used at boundaries where concrete values are required (e.g. C++
    constructors that only accept ``int``, comparison operators inside
    algorithms such as work-division and coordinate computation).

    Key invariant: only structural parameters (sizes, strides, split
    counts) are concretized.  Symbolic loop variables inside coordinate
    output expressions are never touched, so the generated coordinate
    expressions remain symbolic and will carry through to the SDSC when
    symbolic SDSC generation is implemented.
    """
    if isinstance(expr, int):
        return expr
    if isinstance(expr, sympy.Integer):
        return int(expr)
    if hasattr(expr, "free_symbols") and expr.free_symbols:
        return V.graph.sizevars.optimization_hint(expr)
    return int(expr)


def _user_min_or_none(expr: Expr) -> Optional[int]:
    """Return the user-supplied ``mark_dynamic(min=...)``, or ``None``.

    PyTorch initialises the lower bound for size symbols to 2 (sizes 0
    and 1 are specialised), so a recorded lower bound of 2 is
    indistinguishable from "user did not pass min". We treat
    ``lower == 2`` as "no min provided".

    Known limitation: a user who legitimately passes
    ``mark_dynamic(min=2, max=...)`` will be silently treated as if
    they had not passed min at all. The call site in
    ``compute_granularity`` will then take the default-divisor branch
    (and emit the "defaulting granularity to ..." warning) instead of
    honouring the user value. There is no way to disambiguate the two
    cases from the ShapeEnv alone -- resolving this needs PyTorch to
    expose the user-provided min separately from the bound. See #2284
    for the design discussion.
    """
    vr = V.graph.sizevars.shape_env.bound_sympy(expr)
    if not isinstance(vr.lower, sympy.Integer):
        return None
    lower = int(vr.lower)
    # min=2 collides with PyTorch's default lower bound and is treated
    # as "unset" here
    return None if lower == _SHAPE_ENV_DEFAULT_LOWER else lower


def finite_upper_or_none(expr: Expr) -> Optional[int]:
    """Return the ShapeEnv finite upper bound for ``expr``, or ``None``.
    A bound is usable iff it is a positive concrete
    ``sympy.Integer``; ``sympy.oo``, non-integers, and non-positive
    values all return ``None``.
    """
    vr = V.graph.sizevars.shape_env.bound_sympy(expr)
    if isinstance(vr.upper, sympy.Integer) and vr.upper.is_finite and int(vr.upper) > 0:
        return int(vr.upper)
    return None


def compute_granularity(expr: Expr, max_size: int) -> int:
    """Return the granularity for a symbolic dimension.

    Admissible runtime values are ``{G, 2G, ..., max_size}``. If the
    user passed ``mark_dynamic(min=...)`` we honour it after validation;
    otherwise we pick the smallest divisor of ``max_size`` that
    satisfies ``config.max_buckets`` and ``config.min_default_granularity``.

    Callers must only invoke this for symbolic ``expr``. See #2284,
    #2287, #2288, #2289 for the full design.

    Deferred: when the symbolic dim is the stick dim of its tensor the
    granularity also needs to be a multiple of ``elems_per_stick(dtype)``.
    Handled in a follow-up once the stick-dim symbolic path is enabled.
    """
    assert hasattr(expr, "free_symbols") and expr.free_symbols, (
        f"compute_granularity called on non-symbolic expr={expr!r}"
    )

    max_buckets = config.max_buckets
    min_default_g = config.min_default_granularity

    # When ShapeEnv has no finite upper bound, max_size came from
    # optimization_hint (via compute_max_size below, merged in #2003), not from
    # mark_dynamic(max=...). The granularity is then only as trustworthy
    # as that hint -- warn the user so they can pin it explicitly with
    # mark_dynamic(max=...).
    if finite_upper_or_none(expr) is None:
        warnings.warn(
            f"max for symbolic dim {expr} came from optimization_hint, not from "
            f"mark_dynamic(max=...). Proceeding with max={max_size} as a "
            f"best-effort estimate. Set max explicitly via mark_dynamic to "
            f"lock the bucket structure.",
            stacklevel=2,
        )

    user_min = _user_min_or_none(expr)
    if user_min is not None:
        if max_size % user_min != 0:
            raise Unsupported(
                f"mark_dynamic(min={user_min}) must divide max={max_size}; "
                f"got {max_size} % {user_min} = {max_size % user_min}"
            )
        if max_size // user_min > max_buckets:
            raise Unsupported(
                f"mark_dynamic(min={user_min}) produces {max_size // user_min} "
                f"buckets, exceeds max_buckets={max_buckets}. Increase min "
                f"to reduce the bucket count, or raise config.max_buckets."
            )
        return user_min

    # No user min: pick the smallest divisor d of max_size where
    # d >= min_default_g and max_size / d <= max_buckets.
    for divisor in sorted(sympy.divisors(max_size)):
        if divisor < min_default_g:
            continue
        if max_size // divisor <= max_buckets:
            warnings.warn(
                f"mark_dynamic(min=...) not provided for symbolic dim "
                f"{expr}; defaulting granularity to {divisor} "
                f"(max={max_size}, {max_size // divisor} buckets). "
                f"Set min explicitly to override.",
                stacklevel=2,
            )
            return divisor

    # Unreachable for sane inputs: max_size is always a divisor of
    # itself and gives 1 bucket, so the loop above always finds a hit.
    # Kept as a defensive raise.
    raise Unsupported(
        f"No valid granularity for max={max_size} under "
        f"max_buckets={max_buckets}, min_default_granularity={min_default_g}"
    )


def concretize_index(index: sympy.Expr, loop_vars: set) -> sympy.Expr:
    """Replace non-loop symbolic variables in an index expression with concrete values.

    With ``dynamic=True``, the host index may contain symbolic strides. When
    ``normalize_coordinates`` isolates each loop variable's contribution
    by substituting 0 for all other free symbols, the size symbol ``s1``
    is also zeroed.  This function replaces size symbols with their concrete
    hints so that coordinate expressions are structurally identical to static-shape
    compilation while loop variable symbols are preserved.
    """

    # Handle non-symbolic index (e.g., scalar tensors with index=0)
    if not isinstance(index, sympy.Basic):
        return sympy.sympify(index)

    # Exclude indirect (gather/scatter) index symbols such as ``tmp0``. Under
    # PT 2.12, ``optimization_hint`` concretizes an unbacked indirect symbol to
    # ``config.unbacked_symint_fallback`` (8192) instead of raising, which would
    # drop the symbol from the coordinate and break named-dim propagation for
    # gathers. Only genuine dynamic-shape size symbols (s0, s1, ...) should be
    # concretized here; indirect symbols must stay symbolic.
    size_syms = {s for s in (index.free_symbols - loop_vars) if not is_indirect(s.name)}
    if not size_syms:
        return index
    # Try each symbol individually
    subs = {}
    for s in size_syms:
        try:
            hint = V.graph.sizevars.optimization_hint(s)
            subs[s] = hint  # Successfully concretized
        except (TypeError, ValueError):
            # Can't concretize this symbol, skip it
            pass

    if not subs:
        return index  # No symbols concretized, return original
    result = index.subs(subs)
    return result


def compute_max_size(expr: Union[Expr, int]) -> int:
    """Return the maximum value a symbolic size expression can take.

    Uses the ShapeEnv upper bound when one is recorded (i.e. the symbol was
    created with an explicit ``max=`` constraint using mark_dynamic API). Falls
    back to ``optimization_hint`` when no finite upper bound exists.

    Needed for dynamic shape support.
    """
    if isinstance(expr, int):
        return expr
    if isinstance(expr, sympy.Integer):
        return int(expr)
    if not (hasattr(expr, "free_symbols") and expr.free_symbols):
        return int(expr)
    bound = finite_upper_or_none(expr)
    if bound is not None:
        return bound
    # No finite ShapeEnv bound: fall back to the permissive hint. size_hint was
    # removed in PT 2.12; optimization_hint is its replacement and keeps the
    # intended "best-effort max estimate" semantics (a heuristic/fallback for
    # unbacked symbols) rather than raising.
    return V.graph.sizevars.optimization_hint(expr)


def compute_symbolic_bounds(expr: Union[Expr, int]) -> "tuple[int, int] | None":
    """Return (max_size, granularity) bounds for a symbolic expression from ShapeEnv.

    Returns None for concrete expressions (no free symbols).
    max_size is the computed maximum size,
    granularity from compute_granularity.
    """
    if isinstance(expr, (int, sympy.Integer)):
        return None
    if not (hasattr(expr, "free_symbols") and expr.free_symbols):
        return None
    shape_env = V.graph.sizevars.shape_env
    if shape_env is None:
        return None

    max_size = compute_max_size(expr)
    granularity = compute_granularity(expr, max_size)

    return (max_size, granularity)


def get_mem_deps_from_rw(read_writes: ReadWrites) -> list[SchedNodeArg]:
    res: list[SchedNodeArg] = []
    for arg in read_writes.reads:
        # Indirect deps are index tensors (e.g. gather indices) whose access
        # pattern is data-dependent; they cannot drive work-division planning.
        if (
            isinstance(arg, MemoryDep)
            and isinstance(arg.index, sympy.Basic)
            and not arg.is_indirect()
        ):
            buf = V.graph.get_buffer(arg.name)
            res.append(SchedNodeArg(arg, _fixed_read_layout(buf)))
    return res


def op_out_coords(op: ComputedBuffer) -> list[sympy.Expr]:
    """Return host coordinates for the output dep of a ComputedBuffer."""
    output_dep = next(iter(op.get_read_writes().writes))
    return host_coordinates(op.get_layout(), output_dep, indirect_sizes_from_op(op))


def is_restickify_coords(in_coords: list[Expr], out_coords: list[Expr]) -> bool:
    """Return whether a single-input pointwise copy is a RESTICKIFY (vs IDENTITY).

    ``in_coords`` / ``out_coords`` are the operands' device-space coordinates.
    It is a restickify iff a *different* host dim lands within the stick. A
    broadcast into a new stick dim is an identity fill, not a stick swap.

    The authoritative test, shared by the codegen store side and the padding
    pass matcher (``is_restickify_op``) so the two cannot disagree.
    """
    if all(e == 0 for e in in_coords) and not all(e == 0 for e in out_coords):
        return False  # broadcast: scalar input expanding to non-scalar output
    out_stick_syms = out_coords[-1].free_symbols
    if out_stick_syms == in_coords[-1].free_symbols:
        return False
    in_stick_syms = in_coords[-1].free_symbols
    if not in_stick_syms and out_stick_syms:
        in_syms = set().union(*(coord.free_symbols for coord in in_coords))
        if out_stick_syms.isdisjoint(in_syms):
            return False
    return True


def _scatter_index_buf_names_ordered(op: ComputedBuffer) -> list[str]:
    """Return names of the index tensors used in a Scatter op's output_indexer.

    For Scatter ops the indirect index is encoded in the output_indexer
    closure. Extract the index buffer names directly from the 'indices'
    closure variable, preserving the order of `indices` (position within that
    list is the scattered dimension). Returns [] if op isn't a Scatter, or if
    the closure doesn't expose an 'indices' variable in the expected shape
    (e.g. because Inductor renamed it), in which case a warning is logged
    since downstream passes will silently miss the scatter index tensors.
    """
    from torch._inductor.ir import Scatter

    if not isinstance(op.data, Scatter):
        return []

    fn = op.data.output_indexer
    if fn.__closure__ is None:
        return []

    freevars = fn.__code__.co_freevars
    try:
        cells = {
            name: cell.cell_contents for name, cell in zip(freevars, fn.__closure__)
        }
    except ValueError:
        return []

    indices = None
    if "indices" in cells:
        indices = cells["indices"]
    elif "index_loader" in cells:
        # Fallback: PyTorch Inductor may use index_loader instead of direct indices.
        # Try to extract the indices from index_loader's closure.
        index_loader = cells["index_loader"]
        if hasattr(index_loader, "__closure__") and index_loader.__closure__:
            loader_freevars = index_loader.__code__.co_freevars
            try:
                loader_cells = {
                    name: cell.cell_contents
                    for name, cell in zip(loader_freevars, index_loader.__closure__)
                }
                if "indices" in loader_cells:
                    indices = loader_cells["indices"]
            except (ValueError, AttributeError):
                pass

    if indices is None:
        logger.warning(
            "Scatter.output_indexer closure has no 'indices' variable or "
            "'index_loader' — Inductor structure may have changed. "
            "Scatter index tensors will not be excluded from stick compatibility "
            "checks. (freevars: %s)",
            list(freevars),
        )
        return []

    names = []
    for idx_tensor in indices:
        if idx_tensor is None:
            continue
        # Unwrap TensorBox -> StorageBox -> Buffer to get the name
        node = idx_tensor
        while hasattr(node, "data"):
            node = node.data
        if hasattr(node, "name") and node.name is not None:
            names.append(node.name)
    return names


def _find_scatter_index_buf_names(op: ComputedBuffer) -> set[str]:
    """Return names of deps whose loaded values are used as indices in scatter output_indexer."""
    return set(_scatter_index_buf_names_ordered(op))


def _build_indirect_store_subs(
    op: ComputedBuffer,
) -> "tuple[dict[sympy.Symbol, sympy.Expr], dict[sympy.Symbol, int] | None]":
    """Map indirect symbols in scatter writes to (IndexedBase subs, sizes).

    For Scatter ops, the scatter indices are loop vars in write_dep.index.
    Identify which loop vars come from scatter index buffers by extracting
    those buffers and seeing which symbols appear in their index expressions.
    Returns ({sym: IndexedBase[...]}, None) -- sizes is always None since the
    scattered-dim size isn't recoverable from op alone; see compute_coordinates,
    which treats sizes=None as "skip unknown symbols silently."
    """
    from sympy import IndexedBase

    rw = op.get_read_writes()
    writes = [
        d
        for d in rw.writes
        if isinstance(d, MemoryDep) and isinstance(d.index, sympy.Basic)
    ]
    if not writes:
        return {}, None
    write_dep = writes[0]

    # Extract scatter index symbols (symbols in write_dep.index not in loop ranges).
    all_write_syms = write_dep.index.free_symbols
    loop_syms = set(write_dep.ranges.keys())
    scatter_index_syms = all_write_syms - loop_syms

    if not scatter_index_syms:
        # No scatter symbols found.
        return {}, None

    # Try to map scatter symbols to index buffers from the closure.
    index_buf_names = _scatter_index_buf_names_ordered(op)
    if index_buf_names:
        # Build map of all read deps by name
        read_deps = [d for d in rw.reads if isinstance(d, MemoryDep)]
        dep_by_name = {d.name: d for d in read_deps}

        # Recover each symbol's relative creation order from its numeric
        # tmp<N> suffix. Inductor's LoopBody.add_indirect assigns indirect
        # placeholder symbols with a monotonic counter as output_indexer
        # iterates over `indices` in position order (lowering.py's
        # index_output_size_and_inner_fn), and the placeholder is later
        # substituted 1:1 with the real tmp<N> CSE variable. So sorting by
        # suffix recovers the same order as `indices` -- free_symbols itself
        # is an unordered set and cannot be zipped directly.
        def _sym_sort_key(s: sympy.Symbol) -> tuple[int, int, str]:
            m = regex.search(r"(\d+)$", s.name)
            return (0, int(m.group(1)), "") if m else (1, 0, s.name)

        ordered_syms = sorted(scatter_index_syms, key=_sym_sort_key)

        # CSE can dedupe two indices entries that load the same buffer at the
        # same index expression, so len(ordered_syms) may be < len(indices).
        # Only pair positionally when counts agree -- pairing at a mismatched
        # count would silently attribute one index buffer's symbol to
        # another, which is the exact bug this replaces.
        subs = {}
        if len(ordered_syms) == len(index_buf_names):
            for sym, index_buf_name in zip(ordered_syms, index_buf_names):
                if index_buf_name not in dep_by_name:
                    continue
                index_dep = dep_by_name[index_buf_name]
                subs[sym] = IndexedBase(index_dep.name)[index_dep.index]
        else:
            logger.warning(
                "_build_indirect_store_subs: %d scatter symbols but %d index "
                "buffers (CSE likely deduped identical index loads) -- "
                "cannot safely pair symbols to buffers positionally; "
                "falling back to placeholder subs. symbols=%s indices=%s",
                len(ordered_syms),
                len(index_buf_names),
                ordered_syms,
                index_buf_names,
            )
        if subs:
            return subs, None

    # Fallback: if we couldn't extract index buffer names from the closure,
    # create placeholder subs so that scatter_access_subs can be built.
    # The actual buffer names don't matter for layout enforcement.
    subs = {
        sym: IndexedBase(f"scatter_idx_{i}")[sym]
        for i, sym in enumerate(scatter_index_syms)
    }

    # The valid range for a scatter-index symbol isn't recoverable here (it's
    # the mutation target's scattered-dim size, not visible from op alone).
    # Return None, matching indirect_info_from_op's documented convention:
    # sizes=None tells compute_coordinates to skip unknown symbols silently
    # rather than raising Unsupported or misreading a fabricated size.
    return subs, None


def indirect_info_from_op(
    op: "ComputedBuffer | None",
) -> "tuple[set[str], dict[sympy.Symbol, sympy.Expr], dict[sympy.Symbol, int] | None]":
    """Return (dep_names, access_subs, sizes) for a ComputedBuffer in one inner_fn pass.

    Pass op=None when there is no ComputedBuffer (e.g. structural callers that only
    check stick compatibility or layout shape). Returns (set(), {}, None), where
    sizes=None tells compute_coordinates to skip unknown symbols silently rather than
    raising Unsupported. None is returned when op has no indirect reads. If indirect
    reads exist but none resolve to a known buffer (unexpected), sizes={} is returned;
    in normalize_coordinates this still produces opaque-Term fallback (same as None),
    but in compute_coordinates an unknown symbol would raise Unsupported.
    """
    if op is None:
        return set(), {}, None

    from torch._inductor.ir import Scatter

    # For scatter ops, extract info from the write side instead of reads.
    if isinstance(op.data, Scatter):
        subs, sizes = _build_indirect_store_subs(op)
        scatter_names: set[str] = set()
        for expr in subs.values():
            if hasattr(expr, "base") and hasattr(expr.base, "name"):
                scatter_names.add(expr.base.name)
        access_subs = {
            sym: IndirectAccess(sympy.Symbol(expr.base.name))
            for sym, expr in subs.items()
            if hasattr(expr, "base")
        }
        return scatter_names, access_subs, sizes

    # For gather and other ops, use the read side.
    subs, sizes = _build_indirect_load_subs(op)
    names: set[str] = {expr.base.name for expr in subs.values()}
    names |= _find_scatter_index_buf_names(op)
    access_subs = {
        sym: IndirectAccess(sympy.Symbol(expr.base.name)) for sym, expr in subs.items()
    }
    return names, access_subs, sizes


def indirect_sizes_from_op(
    op: "ComputedBuffer | None",
) -> "dict[sympy.Symbol, int] | None":
    """Build {indirect_sym → size} for a ComputedBuffer (pre-scheduler).

    Returns the valid index range for each indirect symbol, captured from the
    size argument of indirect_indexing() during inner_fn re-execution.
    Pass op=None when there is no ComputedBuffer; returns None.
    """
    _, _, sizes = indirect_info_from_op(op)
    return sizes


class _LoadSentinel:
    """Opaque token returned by load(); carries the buffer name through ops."""

    def __init__(self, name: str):
        self.name = name


class _IndirectIndexFinder:
    """Re-executes inner_fn to map each indirect load to its index buffer and valid range.

    Inductor bakes the index range into the inner_fn closure as the size argument to
    ops.indirect_indexing() — invisible in printed IR, only accessible by re-execution.
    This handler intercepts those calls to recover both the source buffer name and the size.
    """

    def __init__(self):
        from torch._inductor.ops_handler import MockHandler

        self._mock = MockHandler()
        self._pending_indirect_index_buf: str | None = None
        self._pending_indirect_index_size: int | None = None
        self.indirect_index_by_buf: dict[str, str] = {}
        self.indirect_index_size_by_buf: dict[str, int] = {}

    def load(self, name: str, index):
        if self._pending_indirect_index_buf is not None:
            if self._pending_indirect_index_size is None:
                raise Unsupported(
                    f"indirect_indexing() set pending buf {self._pending_indirect_index_buf!r} "
                    "but did not set pending size; single-slot protocol violated"
                )
            self.indirect_index_by_buf[name] = self._pending_indirect_index_buf
            self.indirect_index_size_by_buf[name] = self._pending_indirect_index_size
            self._pending_indirect_index_buf = None
            self._pending_indirect_index_size = None
        return _LoadSentinel(name)

    def indirect_indexing(self, index_var, size, check=True, wrap_neg=True):
        # Assumes load() is called immediately after — Inductor's aten.index
        # lowering always emits indirect_indexing() directly before the
        # consuming load(), so the single slot is never overwritten in between.
        if isinstance(index_var, _LoadSentinel):
            if self._pending_indirect_index_buf is not None:
                raise Unsupported(
                    f"indirect_indexing({index_var.name}) called before load() consumed "
                    f"the previous pending slot ({self._pending_indirect_index_buf}); "
                    "chained indirect indexing is not supported"
                )
            self._pending_indirect_index_buf = index_var.name
            self._pending_indirect_index_size = int(size)
        return sympy.S.Zero

    def __getattr__(self, attr):
        return getattr(self._mock, attr)


def _find_indirect_index_bufs(
    op: ComputedBuffer,
) -> "tuple[dict[str, str], dict[str, int]]":
    """Re-execute inner_fn and return ({data_buf: index_buf}, {data_buf: size}) mappings."""
    from torch._inductor.virtualized import V as _V

    finder = _IndirectIndexFinder()
    with _V.set_ops_handler(finder):
        op.data.inner_fn(*op.data.inner_fn_args())
    return finder.indirect_index_by_buf, finder.indirect_index_size_by_buf


def _build_indirect_load_subs(
    op: ComputedBuffer,
) -> "tuple[dict[sympy.Symbol, sympy.Expr], dict[sympy.Symbol, int] | None]":
    """Map indirect symbols to (IndexedBase subs, sizes).

    Pre-scheduler only: re-executes inner_fn via _IndirectIndexFinder to learn
    which buffer's load produced each indirect index and what size it carries.
    Returns ({sym: IndexedBase[...]}, {sym: size}).
    """
    from sympy import IndexedBase

    rw = op.get_read_writes()
    reads = [
        d
        for d in rw.reads
        if isinstance(d, MemoryDep) and isinstance(d.index, sympy.Basic)
    ]
    if not any(d.is_indirect() for d in reads):
        return {}, None
    indirect_index_buf_map, indirect_index_size_map = _find_indirect_index_bufs(op)
    dep_by_name = {d.name: d for d in reads}
    subs = {}
    sizes = {}
    for d in reads:
        if not d.is_indirect():
            continue
        indirect_index_buf = indirect_index_buf_map.get(d.name)
        if indirect_index_buf is None:
            continue
        indirect_index_dep = dep_by_name[indirect_index_buf]
        size = indirect_index_size_map.get(d.name)
        indirect_syms = [s for s in d.index.free_symbols if s not in d.ranges]
        if len(indirect_syms) > 1:
            raise Unsupported(f"multiple indirect symbols in {d.name}: {indirect_syms}")
        for sym in indirect_syms:
            subs[sym] = IndexedBase(indirect_index_dep.name)[indirect_index_dep.index]
            if size is not None:
                sizes[sym] = size
    return subs, sizes


def indirect_store_sizes(
    dep: MemoryDep, layout: "FixedTiledLayout"
) -> "dict[sympy.Symbol, int]":
    """Map each store-side indirect symbol to its destination row extent.

    The indirect symbol selects a scatter-destination row; its valid range is
    the destination's host extent along the scattered dimension, recovered by
    matching the symbol's linear coefficient in the write index to a host
    stride. device_coordinates() needs these integer ranges (not IndirectAccess
    markers) to process the row symbol as an ordinary loop var; unlike gather,
    there is no load-side indirect_indexing() call to recover the size from on
    the store side, so we derive it from the layout instead.
    """
    host_size = [concretize_expr(s) for s in layout.size]
    host_stride = [concretize_expr(s) for s in layout.stride]
    sizes: dict[sympy.Symbol, int] = {}
    index = dep.index
    for sym in index.free_symbols:
        if sym in dep.ranges:
            continue
        coeff = index.coeff(sym)
        for dim, st in enumerate(host_stride):
            if st == coeff:
                sizes[sym] = host_size[dim]
                break
    return sizes


def _wrap_indirect_subs(
    raw: dict[sympy.Symbol, sympy.Expr],
) -> "dict[sympy.Symbol, sympy.Expr]":
    """Convert index buffer references to IndirectAccess markers.

    Takes mappings like {sym → IndexedBase(name)[index]} and converts them to
    {sym → IndirectAccess(name)}. Used by both load and store builders to mark
    which dimensions are accessed indirectly.
    """
    return {
        sym: IndirectAccess(sympy.Symbol(expr.base.name)) for sym, expr in raw.items()
    }


def indirect_access_subs_from_op(
    op: ComputedBuffer,
) -> "dict[sympy.Symbol, sympy.Expr]":
    """Find all indirect accesses in an operation and mark them appropriately.

    Called before scheduling to identify which dimensions are accessed through
    runtime indices. Handles both gather (indirect reads) and scatter (indirect
    writes). Returns substitutions that mark these dimensions with IndirectAccess
    so the work division planner knows not to split them incorrectly.

    Uses indirect_info_from_op() for the load side (already returns IndirectAccess
    markers), then merges scatter store-side subs via _build_indirect_store_subs.
    """
    _, load_subs, _ = indirect_info_from_op(op)
    store_subs_raw, _ = _build_indirect_store_subs(op)
    store_subs = _wrap_indirect_subs(store_subs_raw)
    return {**load_subs, **store_subs}


def indirect_store_subs_from_op(
    op: ComputedBuffer,
) -> "dict[sympy.Symbol, sympy.Expr]":
    """Find indirect accesses in scatter writes only.

    This is the write-only version of `indirect_access_subs_from_op`. Marks
    the scatter destination's row dimension as IndirectAccess so it stays at a
    shared base address across cores (same treatment as gather value tables).
    Returns empty for non-scatter operations.
    """
    store_subs_raw, _ = _build_indirect_store_subs(op)
    return _wrap_indirect_subs(store_subs_raw)


def indirect_access_subs_from_kernel(
    indirect_vars: "dict[sympy.Symbol, Any]",
) -> "dict[sympy.Symbol, sympy.Expr]":
    """Build {indirect_sym → IndirectAccess(name)} from SpyreKernel.indirect_vars (post-scheduler).

    Used after scheduling, where indirect_vars directly maps the fresh symbol
    returned by indirect_indexing() to its source TensorAccess.
    No re-execution of inner_fn needed — the mapping is available live.
    The resulting subs can be passed to device_coordinates() or applied
    directly to already-computed coordinate expressions.
    """
    return {
        sym: IndirectAccess(sympy.Symbol(ta.name)) for sym, ta in indirect_vars.items()
    }


def host_coordinates(
    layout: FixedLayout,
    dep: MemoryDep,
    indirect_sizes: "dict[sympy.Symbol, int] | None",
) -> list[sympy.Expr]:
    """Compute host-space coordinate expressions for a tensor access.

    Args:
        layout: Host layout of the tensor being accessed.
        dep: Memory dependency describing the access index and loop ranges.
        indirect_sizes: {indirect_sym → size} from indirect_sizes_from_op(), or
            None for structural callers (stick-compatibility checks, layout
            matching) where indirect coordinates are irrelevant.

    Returns:
        One coordinate expression per host dimension.
    """
    # Concretize size/stride so compute_coordinates can use plain ``<``/``>``
    # comparisons.  var_ranges and index stay symbolic so the *output*
    # coordinate expressions remain symbolic.
    # TODO(issue#1373): remove concretization once compute_coordinates handles
    #              symbolic comparisons natively.
    concrete_size = [concretize_expr(s) for s in layout.size]
    concrete_stride = [concretize_expr(s) for s in layout.stride]
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    return compute_coordinates(
        concrete_size, concrete_stride, dep.ranges, index, indirect_sizes=indirect_sizes
    )


def identify_matmul_inputs(
    inputs: list[MemoryDep],
    write_dep: MemoryDep,
) -> tuple[MemoryDep, MemoryDep]:
    """Identify Input1 (x) and Input2 (y) of a BatchMatmul op.

    Uses positional order: lower_bmm always emits x before y, so inputs[0] is
    x and inputs[1] is y.  This is valid even when N=1 (the N symbol is
    constant-folded out of index expressions).

    Raises ValueError if len(inputs) != 2.
    """
    if len(inputs) != 2:
        raise ValueError(
            f"identify_matmul_inputs: expected 2 inputs, got {len(inputs)}"
        )
    return inputs[0], inputs[1]


def find_reduction_var(inputs: Sequence[MemoryDep], out_dep: MemoryDep) -> sympy.Symbol:
    """Return the single input iteration symbol reduced from the output.

    Symbols must have a range on the input dependency that uses them. Indirect
    gather/scatter address symbols are not iteration dimensions.
    """
    reduction_vars = {
        sym
        for inp in inputs
        for sym in inp.index.free_symbols
        if sym in inp.ranges and not is_indirect(sym.name)
    } - out_dep.index.free_symbols
    if len(reduction_vars) != 1:
        raise Unsupported(
            f"expected exactly 1 reduction variable, got {reduction_vars}"
        )
    return next(iter(reduction_vars))


def broadcast_batch_vars(op: Operation, x_dep: MemoryDep, out_dep: MemoryDep) -> set:
    """Return output loop vars for dims coarse-tiled that x is broadcast over.

    When a matmul operand (x) is broadcast over a batch dim (e.g. torch.matmul
    of a [T,H] tensor unsqueezed against a [E,H,F] weight) AND that dim is
    coarse-tiled with >1 element per tile, the tile-loop variable for that dim
    survives in the output's (and y's) index expression -- unlike at 1
    element/tile, where the per-tile address offset is a pure constant handled
    entirely outside the per-tile index expression. This makes the leftover
    batch-dim loop var indistinguishable from the true generated (N) dim via
    set membership on dep.index.free_symbols alone (see issue #3888): both are
    "in y and the output, but not in x".

    op.loop_info.loop_tiled_dims (stamped by coarse-tiling before layout
    propagation runs, per passes.py's pass ordering) lists, per nesting
    level, the raw op.data.ranges positions this loop group tiles -- for the
    innermost tile-body op itself, output_tiled_dims/tiled_dims_per_read are
    both empty (this op's own write is already the tile-local slice, so it
    has nothing further to report), so loop_tiled_dims is the only surviving
    signal identifying which raw dims the enclosing loop nest tiles at all.
    A raw tiled dim is a spurious "excess" batch var iff x's own read lacks
    that dim's squeezed symbol entirely (broadcast), rather than genuinely
    being tiled on it too.
    """
    loop_info = getattr(op, "loop_info", None)
    if loop_info is None or not loop_info.loop_tiled_dims:
        return set()

    # Map each squeezed output-dim position (index into out_dep.var_names) to
    # its raw op.data.ranges position -- mirrors
    # SpyreKernel._host_dim_to_index_symbol's squeeze arithmetic, restricted to
    # the output (non-reduction) side.
    ranges = getattr(op.data, "ranges", None)
    if ranges is None:
        return set()
    raw_to_squeezed: dict[int, int] = {}
    it_idx = 0
    for host_idx, r in enumerate(ranges):
        if int(r) != 1:
            raw_to_squeezed[host_idx] = it_idx
            it_idx += 1

    excess: set = set()
    for level_dims in loop_info.loop_tiled_dims:
        for pos in level_dims:
            squeezed = raw_to_squeezed.get(pos)
            if squeezed is None or squeezed >= len(out_dep.var_names):
                continue
            var = out_dep.var_names[squeezed]
            # x is genuinely tiled on this dim too -- not a broadcast dim.
            if var in x_dep.index.free_symbols:
                continue
            excess.add(var)
    return excess


def find_matmul_generated_var(
    y_dep: MemoryDep,
    x_dep: MemoryDep,
    out_dep: MemoryDep,
    op: Operation | None = None,
) -> sympy.Symbol:
    """Return the single loop variable that appears in y's and the output's index but not in x's.

    This is the N (generation) dimension of a matmul.

    When op is given, excludes candidates that are actually a coarse-tiled
    batch dim x is broadcast over rather than the true generated dim -- see
    broadcast_batch_vars.

    coarse_tile.py's plan-time check (_check_matmul_broadcast_batch_tiling)
    already rejects the >1-element/tile broadcast-batch configuration before
    layout propagation runs, so in the tiled case this disambiguation should
    only ever need to resolve the (always-supported) 1-element/tile case. The
    two checks are intentionally layered as belt-and-suspenders: the plan-time
    check gives an early, precise diagnostic; this one keeps propagation
    correct even if that guard is ever loosened or bypassed.

    Raises Unsupported if the count is not exactly 1.
    """
    y_syms = y_dep.index.free_symbols
    x_syms = x_dep.index.free_symbols
    out_syms = out_dep.index.free_symbols
    logger.debug(
        "[find_matmul_generated_var] looking for N (generated dim = in y & out, not in x)\n"
        "  x   index=%-20s  free=%s\n"
        "  y   index=%-20s  free=%s\n"
        "  out index=%-20s  free=%s\n"
        "  (y & out) = %s  =>  minus x = %s",
        x_dep.index,
        x_syms,
        y_dep.index,
        y_syms,
        out_dep.index,
        out_syms,
        y_syms & out_syms,
        (y_syms & out_syms) - x_syms,
    )
    generated_vars = (y_syms & out_syms) - x_syms
    logger.debug("  generated_vars = %s", generated_vars)
    if op is not None and len(generated_vars) > 1:
        generated_vars = generated_vars - broadcast_batch_vars(op, x_dep, out_dep)
        logger.debug("  generated_vars (after broadcast filter) = %s", generated_vars)
    if len(generated_vars) != 1:
        raise Unsupported(
            f"expected exactly 1 generated variable, got {generated_vars}"
        )
    return next(iter(generated_vars))


def is_stick_expr_offset_free(stick_expr: sympy.Expr, elems_per_stick: int) -> bool:
    """Check if a stick expression is free of constant offsets.

    Returns True for stick expressions with no additive offset:
    - Mod(var, elems_per_stick) where var is a single symbol
    - A bare variable (symbol)
    - Zero
    """
    is_supported_mod = (
        isinstance(stick_expr, sympy.Mod)
        and len(stick_expr.args[0].free_symbols) == 1
        and stick_expr.args[1] == elems_per_stick
    )
    is_bare_var = stick_expr.is_symbol
    is_zero = stick_expr == sympy.S.Zero
    return is_supported_mod or is_bare_var or is_zero


def _is_stick_expr_with_offset(stick_expr: sympy.Expr, elems_per_stick: int) -> bool:
    """Return True if stick_expr is an offset variant: Mod(var, N) + c or var + c."""
    if not isinstance(stick_expr, sympy.Add):
        return False
    free_args = [a for a in stick_expr.args if a.free_symbols]
    return len(free_args) == 1 and is_stick_expr_offset_free(
        free_args[0], elems_per_stick
    )


def _check_stick_expr_supported(stick_expr: sympy.Expr, elems_per_stick: int) -> None:
    """Raise Unsupported for stick expressions may be valid but are not yet supported."""
    offset_free = is_stick_expr_offset_free(stick_expr, elems_per_stick)
    has_offset = _is_stick_expr_with_offset(stick_expr, elems_per_stick)
    if not (offset_free or has_offset):
        raise Unsupported(
            f"Unexpected stick expression {stick_expr!r}: expected "
            f"Mod(var, {elems_per_stick}), a bare variable, 0, or any of those "
            f"with a constant offset"
        )


def device_coordinates(
    stl: SpyreTensorLayout,
    dep: MemoryDep,
    indirect_sizes: "dict[sympy.Symbol, int] | None",
) -> list[sympy.Expr]:
    """Compute device-space coordinate expressions for a tensor access.

    Args:
        stl: Device layout (SpyreTensorLayout) of the tensor being accessed.
        dep: Memory dependency describing the access index and loop ranges.
        indirect_sizes: {indirect_sym → size} from indirect_sizes_from_op(), or
            None for structural callers (stick-compatibility checks, layout
            matching) where indirect coordinates are irrelevant.

    Returns:
        One coordinate expression per device dimension; the last element is
        the stick expression.
    """
    coords = alignment_coordinates(stl, dep.index, dep.ranges, indirect_sizes)
    _check_stick_expr_supported(coords[-1], stl.elems_per_stick())
    return coords


def alignment_coordinates(
    stl: SpyreTensorLayout,
    index: sympy.Expr,
    it_space: dict[sympy.Symbol, sympy.Expr],
    indirect_sizes: "dict[sympy.Symbol, int] | None",
    *,
    repeat_info_out: "dict[sympy.Symbol, dict] | None" = None,
) -> list[sympy.Expr]:
    """Build the coordinates consumed by tensor alignment.

    Scheduler validation and codegen must prepare identical inputs for
    ``align_tensors``.  Keep index concretization and coordinate construction in
    this one helper so the validation preview cannot drift from real codegen.
    """

    # device_size and stride_map come from the C++ SpyreTensorLayout and are
    # already concrete, so no concretization is needed here.
    index = concretize_index(index, set(it_space))
    coords = compute_coordinates(
        stl.device_size,
        stl.stride_map,
        it_space,
        index,
        indirect_sizes,
        repeat_info_out=repeat_info_out,
    )
    return coords


def build_operation_alignment_inputs(
    raw_iteration_space: dict[sympy.Symbol, sympy.Expr],
    accesses: Sequence[AlignmentAccess],
    *,
    indirect_sizes: "dict[sympy.Symbol, int] | None" = None,
    repeat_info: "dict[sympy.Symbol, dict] | None" = None,
    op: ComputedBuffer | None = None,
    read_writes: ReadWrites | None = None,
    aligned_iteration_space: (dict[sympy.Symbol, tuple[sympy.Expr, int]] | None) = None,
) -> AlignmentInputs:
    """Build the complete, immutable input to tensor alignment.

    Codegen may supply its already-finalized iteration space when an operation
    has an op-specific extension.  Scheduler preview supplies ``op`` and
    ``read_writes`` and gets the standard split-aware space.  Coordinate and
    indirect-symbol preparation are shared in both cases.
    """

    if aligned_iteration_space is None:
        if op is None or read_writes is None:
            raise ValueError(
                "alignment input construction requires either a finalized "
                "iteration space or an operation and its dependencies"
            )
        aligned_iteration_space = iteration_space_with_splits(
            op, read_writes, raw_iteration_space
        )

    if indirect_sizes is None and op is not None:
        indirect_sizes = indirect_sizes_from_op(op)
    resolved_indirect_sizes = dict(indirect_sizes or {})
    if resolved_indirect_sizes:
        # Dependency extraction can recreate an equivalent Symbol with
        # different assumptions.  Bind the Symbol present in the real index to
        # the recovered range so preview and codegen normalize the same input.
        size_by_name = {str(dim): size for dim, size in resolved_indirect_sizes.items()}
        for access in accesses:
            for dim in access.index.free_symbols - raw_iteration_space.keys():
                if dim not in resolved_indirect_sizes and str(dim) in size_by_name:
                    resolved_indirect_sizes[dim] = size_by_name[str(dim)]

    repeat_snapshot = {
        symbol: dict(info) for symbol, info in (repeat_info or {}).items()
    }
    tensors = [
        {
            "size": list(access.device_layout.device_size),
            "coordinates": alignment_coordinates(
                access.device_layout,
                access.index,
                raw_iteration_space,
                resolved_indirect_sizes,
                repeat_info_out=repeat_snapshot,
            ),
        }
        for access in accesses
    ]
    return build_alignment_inputs(
        aligned_iteration_space,
        tensors,
        resolved_indirect_sizes,
        repeat_snapshot,
    )


def try_device_coordinates(
    stl: SpyreTensorLayout,
    dep: MemoryDep,
    indirect_sizes: "dict[sympy.Symbol, int] | None",
) -> list[sympy.Expr] | None:
    """Like ``device_coordinates`` but returns ``None`` instead of raising when
    the layout's stick expression is one the backend cannot represent.

    Use this to probe whether a layout is representable under a given dep —
    for example, when iterating candidate input STLs and wanting to skip any
    whose stick concretizes to an unsupported expression (e.g. the literal 1
    when the stick dimension is size-1 in the current op's loop ranges).
    """
    try:
        return device_coordinates(stl, dep, indirect_sizes)
    except Unsupported:
        return None


def _find_entry_output_dim(op) -> "tuple[int, int, int] | None":
    """Locate a gather's index-entry dim on the output; the one coordinate search.

    The entry dim is the index tensor's STICK dim (its rows are selected at
    runtime), which on the gather output is a NON-stick dim (the row selector).
    Multi-core work division splits this dim in whole index sticks, so a split
    is only legal when the output can hold a stick-aligned slice -- i.e. when the
    output extent is a whole multiple of the index ``eps``.

    Returns ``(eps, out_pos, out_extent)`` -- the index elems_per_stick, the
    entry dim's device position in the output layout, and the output
    device_size there -- or ``None`` when ``op`` is not a gather-style indirect
    access, the output layout is not yet a committed ``FixedTiledLayout``, or the
    entry dim coincides with the output's own stick dim (a geometry the
    stick-alignment padding does not cover). Valid from
    ``enforce_indirect_access_layout`` onward, once every buffer's layout is
    committed.
    """
    subs = indirect_access_subs_from_op(op)
    if not subs:
        return None
    index_names = {e.args[0].name for e in subs.values() if e.args}

    rw = op.get_read_writes()
    out_dep = next(iter(rw.writes), None)
    if out_dep is None:
        return None

    # Fail safe: callers use this only to *enable* a split / pad a paddable
    # gather output. If the output can't be analysed (e.g. a scatter writes its
    # destination indirectly, so device_coordinates can't resolve the entry
    # coord), return None so the caller stays conservative rather than erroring.
    try:
        out_stl = _fixed_read_layout(op).device_layout
        out_coords = device_coordinates(out_stl, out_dep, None)
        for d in rw.reads:
            if not (isinstance(d, MemoryDep) and d.name in index_names):
                continue
            idx_stl = _fixed_read_layout(V.graph.get_buffer(d.name)).device_layout
            stick_expr = device_coordinates(idx_stl, d, None)[-1]
            if len(stick_expr.free_symbols) != 1:
                continue
            stick_var = next(iter(stick_expr.free_symbols))
            eps = idx_stl.elems_per_stick()
            # The entry dim must be a NON-stick dim of the output (exclude the
            # last, stick, coordinate). On a scatter the entry coord is an
            # IndirectAccess (its free symbol is the index buffer, not the
            # iteration stick var), so no match -> None -> partial scatter stays
            # forbidden, which is correct: its in-place dest can't be padded.
            for pos, coord in enumerate(out_coords[:-1]):
                if coord.free_symbols == {stick_var}:
                    return eps, pos, int(out_stl.device_size[pos])
    except (RuntimeError, AssertionError, KeyError, ValueError, Unsupported):
        return None
    return None


def is_output_stick_aligned_for_entry(op) -> bool:
    """Whether a gather output already holds a whole-stick slice for its entry dim.

    The work-division guard uses this to decide the partial-last-stick split:
    True means the stick-aligned split is legal (the output extent is a multiple
    of the index ``eps`` -- either naturally, or because the padding pass grew
    it); False means keep the split forbidden. Returns False for anything that is
    not a paddable gather entry (scatter, non-committed layout), so the guard
    stays conservative.
    """
    found = _find_entry_output_dim(op)
    if found is None:
        return False
    eps, _out_pos, out_extent = found
    return out_extent % eps == 0


def padded_entry_output_stl(op) -> "SpyreTensorLayout | None":
    """The gather output's device layout grown so the entry dim is a whole stick.

    The padding pass applies this returned layout; the coordinate search and the
    size math both stay here. Returns ``None`` when there is nothing to do -- not
    a paddable gather entry, or the entry dim is already stick-aligned. The
    logical size is unchanged: only the physical ``device_size`` grows, and the
    D2H copy extracts the logical view from the larger allocation.
    """
    found = _find_entry_output_dim(op)
    if found is None:
        return None
    eps, out_pos, out_extent = found
    if out_extent % eps == 0:
        return None
    out_stl = _fixed_read_layout(op).device_layout
    device_size = list(out_stl.device_size)
    device_size[out_pos] = ((out_extent + eps - 1) // eps) * eps
    logger.info(
        "padded_entry_output_stl: %s entry dim (pos %d) %d -> %d for "
        "stick-aligned multi-core split",
        op.get_name(),
        out_pos,
        out_extent,
        device_size[out_pos],
    )
    return SpyreTensorLayout(
        device_size=device_size,
        stride_map=list(out_stl.stride_map),
        device_dtype=out_stl.device_dtype,
    )


def _shared_indirect_coords(op: "ComputedBuffer") -> list[list[Expr]]:
    """Device-coordinate lists for tensors shared across cores with runtime rows.

    Covers gather value tables (indirect reads) and scatter destinations
    (indirect writes). Each returned list is the device_coordinates of one shared
    tensor, with the runtime-chosen row marked by IndirectAccess. These tensors
    stay at the same base address on every core, so their data dimensions must
    not be split. Returns empty for regular operations.
    """
    coords_lists: list[list[Expr]] = []

    # Gather value tables: filtered out of the normal arg list, recovered here.
    # device_coordinates needs the *integer* index range for each indirect symbol
    # (indirect_sizes), not the IndirectAccess marker map. Compute the coordinates
    # with the raw indirect symbol treated as a normal loop var, then xreplace the
    # subs to mark the runtime-chosen row.
    subs = indirect_access_subs_from_op(op)
    if subs:
        ind_sizes = indirect_sizes_from_op(op)
        for d in op.get_read_writes().reads:
            if isinstance(d, MemoryDep) and d.is_indirect():
                layout = _fixed_read_layout(V.graph.get_buffer(d.name))
                coords = device_coordinates(layout.device_layout, d, ind_sizes)
                coords_lists.append([c.xreplace(subs) for c in coords])

    # Scatter destination: the write, with its runtime-chosen row marked the same
    # way (integer row extent for coordinates, then xreplace to IndirectAccess).
    store_subs = indirect_store_subs_from_op(op)
    if store_subs:
        write = next(iter(op.get_read_writes().writes))
        # Resolve the scatter destination's layout, leniently unwrapping the
        # in-place-mutation layout (mirrors work_division._resolve_layout; inlined
        # to avoid a pass_utils -> work_division import cycle). _fixed_read_layout
        # is too strict here — it rejects a scatter's mutation output.
        layout = op.get_layout()
        if isinstance(layout, MutationLayoutSHOULDREMOVE):
            layout = layout.real_layout()
        ind_sizes = indirect_store_sizes(write, layout)
        coords = device_coordinates(layout.device_layout, write, ind_sizes)
        coords_lists.append([c.xreplace(store_subs) for c in coords])

    return coords_lists


def _non_indirect_coord_syms(coords: list[Expr]) -> set[Symbol]:
    """Extract coordinate symbols, skipping the runtime-chosen row dimension."""
    syms: set[Symbol] = set()
    for coord in coords:
        if hasattr(coord, "has") and coord.has(IndirectAccess):
            # A coordinate can mix the runtime-chosen row with data syms when
            # the layout folds the row dim into an outer one -- a paged KV cache
            # reads as `d0 + 128*IndirectAccess(idx)`. Those data syms still
            # address into the shared table, so report them (#3984).
            inner: set[Symbol] = set()
            for term in coord.atoms(IndirectAccess):
                inner |= term.free_symbols
            syms |= coord.free_symbols - inner
            continue
        syms |= coord.free_symbols
    return syms


def shared_indirect_data_syms(op: "ComputedBuffer") -> set[Symbol]:
    """Find data dimensions of shared tables that must not be split.

    These are the column/stick dimensions of tables shared across cores (gather
    value tables or scatter destinations). We can't split these dimensions
    because the table needs to stay at the same base address on every core.
    Splitting a data dimension would require different base addresses per core,
    breaking the shared access pattern. Returns empty for regular operations.
    """
    syms: set[Symbol] = set()
    for coords in _shared_indirect_coords(op):
        syms |= _non_indirect_coord_syms(coords)
    return syms


def indirect_forbidden_split_syms(op: "ComputedBuffer") -> set[Symbol]:
    """Iteration dims that must not be core-split for an indirect op.

    1. **Shared-table data dims** — the non-row dims of a shared gather/scatter
       table must not advance per core (all cores share the same base address).

    2. **Partial-last-stick entry dims** — splitting a partial last index stick
       across cores straddles the stick boundary. Forbidden unless the gather
       output was already padded to a stick boundary by
       ``enforce_indirect_access_layout``, which makes the split safe. Scatter
       output rows are chosen at runtime so can never be padded; they stay unsplit.
    """
    syms = shared_indirect_data_syms(op)  # (1)

    # (2) Forbid a partial-last-stick index-entry dim UNLESS the gather output is
    # provably stick-aligned for it. The partial check keys off the INDEX's
    # *logical* entry count (d.ranges[stick_var], e.g. 40), which is never padded
    # -- the stick-alignment fix grows the gather OUTPUT's device_size, not the
    # index. is_output_stick_aligned_for_entry reads that (possibly padded)
    # output extent, so the split is allowed only when the output can hold a
    # whole-stick slice. It stays forbidden when the output is NOT aligned: a
    # SCATTER (its in-place dest can't be resized -> returns False) or a gather
    # whose output the pass could not grow -- those fall back to a single core,
    # not miscompile. Applies to BOTH gather and scatter
    output_aligned = is_output_stick_aligned_for_entry(op)
    subs = indirect_access_subs_from_op(op)
    index_names = {e.args[0].name for e in subs.values() if e.args}
    for d in op.get_read_writes().reads:
        if not (isinstance(d, MemoryDep) and d.name in index_names):
            continue
        layout = _fixed_read_layout(V.graph.get_buffer(d.name))
        stick_expr = device_coordinates(layout.device_layout, d, None)[-1]
        if len(stick_expr.free_symbols) != 1:
            continue
        stick_var = next(iter(stick_expr.free_symbols))
        eps = layout.device_layout.elems_per_stick()
        partial = (
            stick_var in d.ranges and concretize_expr(d.ranges[stick_var]) % eps != 0
        )
        if partial and not output_aligned:
            syms.add(stick_var)

    return syms


def indirect_store_entry_syms(
    op: "ComputedBuffer", forbidden: "set[Symbol] | None" = None
) -> set[Symbol]:
    """Find which dimensions of a scatter can be safely parallelized.

    For scatter operations, we can split along the index-entry dimension (giving
    each core different source rows to write). The destination stays shared with
    its row chosen at runtime. This is safe because each core writes to different
    entries.

    ``forbidden`` may be a precomputed ``indirect_forbidden_split_syms(op)`` to
    avoid recomputing it (the caller often already has it); it is computed here
    when omitted.
    """
    from torch._inductor.ir import Scatter

    if not isinstance(op.data, Scatter) or op.data.scatter_mode is not None:
        return set()
    if not indirect_store_subs_from_op(op):
        return set()
    if forbidden is None:
        forbidden = indirect_forbidden_split_syms(op)
    return set(iteration_space_from_op(op)) - forbidden


def iter_var_id(stick_expr) -> int:
    """Iteration variable index from a stick expr: Mod(d2,64) -> 2, d2 -> 2.
    Returns -1 for constant-zero (scalar/broadcast, no real stick).
    NOTE: this is the loop variable index (suffix of dN), NOT a tensor dimension index."""
    if stick_expr == sympy.S.Zero or not stick_expr.free_symbols:
        return -1
    sym = next(iter(stick_expr.free_symbols))
    name = str(sym)
    i = len(name) - 1
    while i >= 0 and name[i].isdigit():
        i -= 1
    return int(name[i + 1 :])


def iteration_space(n: SchedulerNode) -> dict[sympy.Symbol, sympy.Expr]:
    if isinstance(n.node.data, Pointwise):
        # The iteration space of a Pointwise is that of its output
        return next(iter(n.read_writes.writes)).ranges.copy()
    elif isinstance(n.node.data, Reduction):
        # Output dims from the write dep; reduction dims appended from read deps.
        # Inductor shares sympy symbols across all tensor accesses in a Reduction's
        # inner_fn, so the if-not-in guard correctly deduplicates without producing
        # spurious dims even for multi-input reductions (matmul, conv2d, etc.).
        result = next(iter(n.read_writes.writes)).ranges.copy()
        for dep in n.read_writes.reads:
            if isinstance(dep, StarDep):
                continue
            for sym, size in dep.ranges.items():
                if sym not in result:
                    result[sym] = size
        return result
    else:
        raise Unsupported("Unexpected node type")


def iteration_space_from_op(op: ComputedBuffer) -> dict[sympy.Symbol, sympy.Expr]:
    """Pre-scheduler version of iteration_space: uses op.get_read_writes() instead
    of SchedulerNode.read_writes."""
    rw = op_read_writes(op)
    if isinstance(op.data, Pointwise):
        return next(iter(rw.writes)).ranges.copy()
    elif isinstance(op.data, Reduction):
        # Output dims from write dep; reduction dims appended from read deps.
        # Inductor shares sympy symbols across all tensor accesses in a Reduction's
        # inner_fn, so the if-not-in guard correctly deduplicates without producing
        # spurious dims even for multi-input reductions (matmul, conv2d, etc.).
        result = next(iter(rw.writes)).ranges.copy()
        for dep in rw.reads:
            if isinstance(dep, StarDep):
                continue
            for sym, size in dep.ranges.items():
                if sym not in result:
                    result[sym] = size
        return result
    else:
        raise Unsupported("Unexpected node type")


_V = TypeVar("_V")

# Type alias for the two-namespace split storage: (output_splits, reduction_splits).
# output_splits is keyed by the symbol's coefficient in the write dep's index.
# reduction_splits is keyed by the symbol's coefficient in the first read dep's index.
# The two dicts use different reference indices so their keys never collide.
ItSpaceSplits = tuple[dict[sympy.Expr, int], dict[sympy.Expr, int]]


def coeff_through_floor(expr: sympy.Expr, sym: sympy.Symbol) -> sympy.Expr:
    """``expr.coeff(sym)``, but also finds ``sym``'s coefficient when it
    only appears inside a ``floor(...)`` wrapper.

    ``device_tile_advance_expr`` (the only caller of this helper today) is
    always a sympy ``Add`` where each tiled symbol contributes exactly one
    additive term -- one minted symbol per coarse-tiling level, produced by
    ``views.tiling_expr_to_device_expr``. That term is either a plain
    ``Mul`` (``k*sym``) or ``sympy.floor(k*sym/d)``. ``sympy.Expr.coeff()``
    looks through ``Mul``/``Add`` fine but refuses to look inside
    ``floor()``, silently returning 0 for a genuine free symbol. This
    isolates ``sym``'s own term first, then unwraps one ``floor()`` layer
    before delegating to ``.coeff()``.

    Tiles are always a whole number of sticks, so ``floor()``'s division
    here must always be exact; a non-integer result means an earlier pass
    or ``spyre_hint`` produced an invalid sub-stick tile boundary, and is
    reported as ``Unsupported`` rather than silently truncated.
    """
    terms = expr.args if isinstance(expr, sympy.Add) else (expr,)
    own_term = next((t for t in terms if sym in t.free_symbols), None)
    if own_term is None:
        return sympy.S.Zero
    if isinstance(own_term, sympy.floor):
        coeff = own_term.args[0].coeff(sym)
    else:
        coeff = own_term.coeff(sym)
    if coeff != 0 and not coeff.is_Integer:
        raise Unsupported(
            f"Tile-advance coefficient {coeff} for symbol {sym} in "
            f"{expr!r} is not an integer number of sticks -- tiling below "
            "stick granularity is not supported (check the originating "
            "spyre_hint or coarse-tiling pass)"
        )
    return coeff


def _coeff_splits_from_index(
    splits: dict[sympy.Symbol, _V],
    index: sympy.Expr,
    *,
    skip: "Callable[[_V], bool] | None" = None,
) -> dict[sympy.Expr, _V]:
    """Return a coeff→value dict for symbols with a non-zero coefficient in index.

    The coefficient of a symbol in a flat tensor index expression is stable
    across the pre-scheduling / codegen boundary (same layout strides on both
    sides), so it serves as a symbol-identity key that survives the scheduler's
    renaming.  Symbols absent from index (coeff=0) are not included.

    Entries for which ``skip(value)`` returns True are omitted.
    """
    result: dict[sympy.Expr, _V] = {}
    for sym, value in splits.items():
        if skip is not None and skip(value):
            continue
        coeff = index.coeff(sym)
        if coeff != 0:
            result[coeff] = value
    return result


def splits_by_index_coeff(
    splits: dict[sympy.Symbol, int],
    write_index: sympy.Expr,
    read_index: sympy.Expr,
) -> ItSpaceSplits:
    """Encode a symbol→split dict as scheduler coefficient transport.

    This is intentionally a boundary representation: pre-scheduler passes keep
    symbol-keyed ownership. Coefficients are stable across scheduler renaming,
    but cannot distinguish distinct non-unity splits sharing one coefficient.
    This encoder retains the established iteration-order last-axis-wins behavior;
    ``finalize_work_division_for_scheduler`` warns when that transport is lossy.

    Only non-unity splits are stored; 1 is the default on the apply side.
    Dimensions absent from both indexes are intentionally omitted.
    """
    skip = lambda v: v <= 1  # noqa: E731
    output_splits = _coeff_splits_from_index(splits, write_index, skip=skip)
    # Reduction splits: symbols with coeff==0 in write_index but coeff!=0 in read_index
    reduction_only = {
        sym: val for sym, val in splits.items() if write_index.coeff(sym) == 0
    }
    reduction_splits = _coeff_splits_from_index(reduction_only, read_index, skip=skip)
    return output_splits, reduction_splits


def make_iteration_space_ownership(
    op: Operation, splits: dict[sympy.Symbol, int]
) -> TensorWorkDivision:
    """Return complete symbol-keyed ownership for ``op``'s iteration space."""
    iter_space = iteration_space_from_op(op)
    work_slices = {sym: int(splits.get(sym, 1)) for sym in iter_space}
    dim_splits = tuple(work_slices.values())
    contiguous_dim = (
        len(dim_splits) - 1
        if (
            isinstance(op, ComputedBuffer)
            and isinstance(op.data, Reduction)
            and op.data.reduction_type in MATMUL_REDUCTION_OPS
            and config.core_id_k_fast_emission
        )
        else None
    )
    # Keep the selected mapping with the splits for the planned codegen handoff.
    # Scheduler still reconstructs it from coefficient transport today; meanwhile
    # retaining it lets the boundary diagnose aliases with different ownership.
    return TensorWorkDivision(
        work_slices,
        core_to_slice_mapping(
            tuple(iter_space),
            dim_splits,
            math.prod(dim_splits),
            contiguous_dim=contiguous_dim,
        ),
        num_cores=math.prod(dim_splits),
    )


def commit_iteration_space_ownership(
    op: Operation, splits: dict[sympy.Symbol, int]
) -> None:
    """Commit pre-scheduler symbol-keyed split and core ownership."""
    op.iteration_space_ownership = make_iteration_space_ownership(op, splits)


def select_work_division_transport_indexes(
    op: Operation, rw: ReadWrites, it_space: dict[sympy.Symbol, sympy.Expr]
) -> tuple[sympy.Expr, sympy.Expr]:
    """Choose the write/read indexes for legacy coefficient-keyed transport.

    Pre-scheduler work division is keyed by iteration symbols, but Scheduler's
    ``op_it_space_splits`` transport keys output splits by a write-index
    coefficient and reduction-only splits by a read-index coefficient. Encoding
    and later decoding must select the same indexes: changing the read can map a
    reduction symbol to a different coefficient.

    The write reference is the first ``MemoryDep`` write. Normally the read
    reference is the first non-indirect ``MemoryDep`` read, falling back to the
    first memory read and then the write for read-free operations. ``keep_by_index``
    operations instead choose the read whose free symbols overlap the most with
    the iteration symbols absent from the write, because that read best exposes
    their reduction axes.

    Raises:
        Unsupported: if the operation has no ``MemoryDep`` write and therefore
            cannot be represented by this transport.
    """
    write = next((d for d in rw.writes if isinstance(d, MemoryDep)), None)
    if write is None:
        raise Unsupported(f"{op.get_name()} has no MemoryDep write for work division")
    write_index = write.index
    if is_keep_by_index(op):
        reduction_vars = set(it_space) - write_index.free_symbols
        read_index = read_with_max_reduction_overlap(rw.reads, reduction_vars)
        if read_index is not None:
            return write_index, read_index
    first_read = next((d for d in rw.reads if isinstance(d, MemoryDep)), None)
    read = next(
        (d for d in rw.reads if isinstance(d, MemoryDep) and not d.is_indirect()),
        first_read,
    )
    return write_index, read.index if read is not None else write_index


def finalize_work_division_for_scheduler(graph: GraphLowering) -> None:
    """Convert committed symbol ownership to legacy scheduler split transport.

    Coefficient keys cannot distinguish every symbol.  Preserve the established
    iteration-order last-wins behavior, but warn once per affected operation.
    """
    for op in graph.operations:
        ownership = getattr(op, "iteration_space_ownership", None)
        if ownership is None:
            continue
        rw = op_read_writes(op)
        try:
            write_index, read_index = select_work_division_transport_indexes(
                op, rw, iteration_space_from_op(op)
            )
        except Unsupported:
            continue
        collisions: list[str] = []
        for label, index, symbols in (
            ("output", write_index, ownership.work_slices),
            (
                "reduction",
                read_index,
                {
                    sym: split
                    for sym, split in ownership.work_slices.items()
                    if write_index.coeff(sym) == 0
                },
            ),
        ):
            seen: dict[sympy.Expr, tuple[sympy.Symbol, int, sympy.Expr]] = {}
            for sym, split in symbols.items():
                if split <= 1:
                    continue
                coeff = index.coeff(sym)
                if coeff == 0:
                    collisions.append(f"{label}:{sym}=absent")
                    continue
                value = (sym, split, ownership.core_id_to_work_slice[sym])
                if coeff in seen and seen[coeff][1:] != value[1:]:
                    collisions.append(
                        f"{label}: coeff {coeff} {seen[coeff][0]}={seen[coeff][1:]} "
                        f"-> {sym}={value[1:]}"
                    )
                seen[coeff] = value
        if collisions:
            logger.warning(
                "lossy work-division scheduler transport for %s; using "
                "iteration-order last-axis-wins: %s",
                op.get_name(),
                "; ".join(collisions),
            )
        op.op_it_space_splits = splits_by_index_coeff(
            ownership.work_slices, write_index, read_index
        )
        delattr(op, "iteration_space_ownership")


def apply_splits_from_index_coeff(
    coeff_splits: ItSpaceSplits,
    write_index: sympy.Expr,
    read_index: sympy.Expr,
    sched_it_space: dict[sympy.Symbol, sympy.Expr],
) -> dict[sympy.Symbol, int]:
    """Reconstruct a scheduler-symbol→split dict from an ItSpaceSplits pair.

    Output dims (non-zero coeff in write_index) are looked up in
    coeff_splits[0]; reduction dims (zero coeff in write_index) are looked up
    in coeff_splits[1] via their coefficient in read_index.  Symbols not found
    in either dict default to 1.
    """
    output_coeff_splits, reduction_coeff_splits = coeff_splits
    result: dict[sympy.Symbol, int] = {sym: 1 for sym in sched_it_space}
    for sym, size in sched_it_space.items():
        # Skip iteration vars with trivial range.  For symbolic ranges we
        # cannot statically determine triviality (and a symbolic size
        # carries no compile-time guarantee that it is 1), so we assume
        # they are non-trivial — consistent with views.compute_coordinates.
        # TODO(issue#1373): replace with a sympy-aware predicate.
        if isinstance(size, (int, sympy.Integer)) and int(size) <= 1:
            continue
        wc = write_index.coeff(sym)
        if wc != 0:
            if wc in output_coeff_splits:
                result[sym] = output_coeff_splits[wc]
        else:
            rc = read_index.coeff(sym)
            if rc != 0 and rc in reduction_coeff_splits:
                result[sym] = reduction_coeff_splits[rc]
    return result


def iteration_space_with_splits(
    op: Operation,
    rw: ReadWrites,
    it_space: dict[sympy.Symbol, sympy.Expr],
) -> dict[sympy.Symbol, tuple[sympy.Expr, int]]:
    """Attach the operation's committed work-division splits to loop symbols."""

    splits: dict[sympy.Symbol, int] = {}
    if hasattr(op, "op_it_space_splits"):
        write_index, read_index = select_work_division_transport_indexes(
            op, rw, it_space
        )
        splits = apply_splits_from_index_coeff(
            op.op_it_space_splits, write_index, read_index, it_space
        )
    return {dim: (extent, int(splits.get(dim, 1))) for dim, extent in it_space.items()}


# The following restickify helpers are used only by the restickify
# but are here to avoid circular dependences in those files


def restickify_device_size(
    old_device_size: list,
    old_sd_outer_dim: int,
    old_sd_host_size: int,
    new_sd_outer_dim: int,
    new_sd_host_size: int,
    stick_size: int,
) -> list:
    """Computes the new device size after a restickify is performed
    moving the stick from old_sd to new_sd."""
    new_device_size = list(old_device_size)
    new_device_size[-1] = stick_size
    new_device_size[old_sd_outer_dim] = (
        new_sd_host_size + stick_size - 1
    ) // stick_size
    new_device_size[new_sd_outer_dim] = old_sd_host_size
    return new_device_size


def restickify_stride_map(
    old_stride_map: list,
    old_sd_outer_dim: int,
    old_sd_host_stride: int,
    new_sd_outer_dim: int,
    new_sd_host_stride: int,
    stick_size: int,
) -> list:
    """Computes the new stride_map after a restickify is performed moving the stick from old_sd to new_sd."""
    new_stride_map = list(old_stride_map)
    new_stride_map[-1] = new_sd_host_stride
    new_stride_map[old_sd_outer_dim] = new_sd_host_stride * stick_size
    new_stride_map[new_sd_outer_dim] = old_sd_host_stride
    return new_stride_map


def compute_restickify_target_layout(
    stl: SpyreTensorLayout,
    host_layout: FixedLayout,
    target_stick_expr,
    ic: list,
    idc: list,
) -> "SpyreTensorLayout | None":
    """Compute the target STL that results from moving stl's stick to target_stick_expr.
    Returns None if the restickify is infeasible.
    """
    new_sd = matching_dim(ic, target_stick_expr)
    if new_sd is None:
        return None
    host_size = [concretize_expr(s) for s in host_layout.size]
    host_stride = [concretize_expr(s) for s in host_layout.stride]
    old_sd = matching_dim(ic, idc[-1])
    if old_sd is None:
        return None
    old_stick_expr = idc[-1]
    old_stride_map = list(stl.stride_map)
    old_var = next(iter(old_stick_expr.free_symbols))
    new_var = next(iter(target_stick_expr.free_symbols))
    stick_size = get_elem_in_stick(host_layout.dtype)
    old_sd_outer_dim = next(
        (j for j in range(len(idc) - 1) if old_var in idc[j].free_symbols),
        next((j for j in range(len(idc) - 1) if idc[j] == sympy.S.Zero), None),
    )
    if old_sd_outer_dim is None:
        return None
    candidates = [j for j in range(len(idc) - 1) if new_var in idc[j].free_symbols]
    if not candidates:
        return None
    new_sd_outer_dim = candidates[0]
    device_size = restickify_device_size(
        list(stl.device_size),
        old_sd_outer_dim,
        host_size[old_sd],
        new_sd_outer_dim,
        host_size[new_sd],
        stick_size,
    )
    stride_map = restickify_stride_map(
        old_stride_map,
        old_sd_outer_dim,
        host_stride[old_sd],
        new_sd_outer_dim,
        host_stride[new_sd],
        stick_size,
    )
    return SpyreTensorLayout(device_size, stride_map, stl.device_dtype)


def stick_compatible(coords: "list[list[sympy.Expr]]") -> bool:
    """Return True if all tensors are stick-compatible.

    coords: list of device_coordinates() results, one per tensor.

    Compatible means: the union of stick variables (free symbols in the last
    device coordinate) across all tensors has at most one element, and is
    disjoint from the union of nonstick variables (free symbols in all other
    device coordinates, excluding each tensor's own stick variable).
    """
    stick_vars: set[sympy.Symbol] = set()
    nonstick_vars: set[sympy.Symbol] = set()
    for dc in coords:
        tensor_stick_vars = dc[-1].free_symbols
        stick_vars |= tensor_stick_vars
        for coord in dc[:-1]:
            nonstick_vars |= coord.free_symbols - tensor_stick_vars
    return len(stick_vars) <= 1 and stick_vars.isdisjoint(nonstick_vars)


def compute_restickify_needed(
    in_stl: SpyreTensorLayout,
    in_host: FixedLayout,
    in_dep: MemoryDep,
    out_stl: SpyreTensorLayout,
    out_dep: MemoryDep,
    op: "ComputedBuffer | None" = None,
) -> "tuple[bool, SpyreTensorLayout | None]":
    """Determine whether a restickify is needed for one (in_stl, out_stl) pair.

    in_dep and out_dep may differ when the output buffer is accessed with a
    different index than the input (e.g. a transposed read).

    op: when provided, index-role deps (gather indices) are never stick-constrained
    and always return (False, None).

    Returns:
      (False, None)   — stick-compatible: no restickify needed
      (True, stl)     — restickify needed, stl is the target STL for the restickified input
      (True, None)    — restickify needed but infeasible
    """
    ind_names, _, ind_sizes = indirect_info_from_op(op)
    if in_dep.name in ind_names:
        return False, None
    idc = try_device_coordinates(in_stl, in_dep, ind_sizes)
    out_idc = try_device_coordinates(out_stl, out_dep, ind_sizes)
    if idc is None or out_idc is None:
        # One of the layouts has a stick expression the backend cannot
        # represent (e.g. floor(var/N) from a cross-stick access). Such a
        # candidate can never be a feasible restickify source/target.
        #
        # Return (True, None): the (needed=True, tgt=None) pair is the
        # "infeasible restickify" signal on this function's contract. The beam
        # search maps it to INF cost and discards the candidate — see
        # EdgeCostMap._compute_and_cache_cost in optimize_restickify.py. This is
        # preferable to aborting the whole pass when another candidate is valid.
        return True, None
    assert idc, "device_coordinates returned empty list for input"
    assert out_idc, "device_coordinates returned empty list for output"
    # Input stick with an offset always needs restickify to remove the offset.
    in_stick_offset_free = is_stick_expr_offset_free(idc[-1], in_stl.elems_per_stick())
    if in_stick_offset_free and stick_compatible([idc, out_idc]):
        return False, None
    ic = host_coordinates(in_host, in_dep, ind_sizes)
    target_stick = out_idc[-1]

    if target_stick == sympy.S.Zero and not in_stick_offset_free:
        # No output dim carries the input's stick var, so compute_restickify_target_layout
        # would fail to match. Promote the reduction var to the stick dimension so the
        # restickify removes the offset.
        reduction_vars = in_dep.index.free_symbols - out_dep.index.free_symbols
        if reduction_vars:
            red_var = next(iter(reduction_vars))
            target_stick = sympy.Mod(red_var, in_stl.elems_per_stick())
    return True, compute_restickify_target_layout(
        in_stl, in_host, target_stick, ic, idc
    )


def copy_fx_custom_meta(src: "torch.fx.Node", dst: "torch.fx.Node") -> None:
    """Copy meta["custom"] from one FX node to another.

    Call this whenever a pass creates a new FX node replacing an existing one,
    so that custom metadata (including spyre hints) is not silently dropped.
    """
    if "custom" in src.meta:
        dst.meta["custom"] = src.meta["custom"]


def _repoint_mutation_targets(
    operations: list[Operation], old_buf: Buffer, new_buf: Buffer
) -> None:
    """Repoint any ``MutationLayoutSHOULDREMOVE.target`` chain aimed at ``old_buf``.

    Reconstructing a ``ComputedBuffer`` (see ``replace_computed_buffer_body``,
    ``redirect_computed_buffer_reads``) swaps the new object into ``operations``
    and ``V.graph.name_to_buffer``, but a mutation op elsewhere in the graph may
    hold a direct object reference to the old buffer via
    ``MutationLayoutSHOULDREMOVE.target`` -- set once, at the mutation op's
    original lowering time, and never re-resolved by name afterwards (unlike
    ordinary reads, which always go through ``V.graph.get_buffer(name)``).  Left
    unpatched, that op keeps mutating the orphaned old object forever: its
    layout is never promoted past ``FixedLayout``, which later fails the
    ``isinstance(layout, FixedTiledLayout)`` assert in
    ``work_division._resolve_layout`` (see issue #3944/#3945).

    ``target`` may be the bare buffer, or wrapped in one or more
    ``MutableBox``/``BaseView`` layers (``TensorBox(StorageBox(buf))``,
    ``ReinterpretView``, ...) -- both wrapper families expose the next layer
    as ``.data``, so a single attribute name covers both.
    """
    for candidate in operations:
        layout = getattr(candidate, "layout", None)
        if not isinstance(layout, MutationLayoutSHOULDREMOVE):
            continue
        target = layout.target
        if target is old_buf:
            layout.target = new_buf
            continue
        # Buffer/ComputedBuffer (a bare target) has no `.data`, so the walk
        # is guaranteed to terminate there without wrongly descending into
        # an already-bare buffer.
        holder = target
        while hasattr(holder, "data"):
            inner = holder.data
            if inner is old_buf:
                holder.data = new_buf
                break
            holder = inner


def replace_computed_buffer_body(
    op: ComputedBuffer,
    new_data: Loops,
    operations: list[Operation],
    *,
    pass_name: str,
    reason: str | None = None,
) -> ComputedBuffer:
    """Replace the body (``data``) of a ``ComputedBuffer`` with ``new_data``.

    ``ComputedBuffer`` is a frozen dataclass, so its ``data`` field cannot be
    mutated in place.  This function constructs a new ``ComputedBuffer`` with
    the updated body and swaps it into ``operations``, copying all metadata
    fields that downstream passes depend on: ``operation_name``, ``origins``,
    ``origin_node``, and the ``_split_size`` / ``_original_*`` fields used by
    ``get_default_sizes_body``.  The ``get_default_sizes_body`` cache is
    cleared on the new buffer so stale size results from the old body are not
    reused.  Also repoints any ``MutationLayoutSHOULDREMOVE.target`` elsewhere
    in ``operations`` that referenced the old object (see
    ``_repoint_mutation_targets``).

    Returns the replacement ComputedBuffer.
    """
    # Always wrap the original inner_fn via WrapperHandler; never rebuild
    # index expressions from scratch (they go stale — see issue #2797).
    new_buf = ComputedBuffer(
        name=op.get_name(),
        layout=op.layout,
        data=new_data,
        _split_size=op._split_size,
        _original_inner_fn=op._original_inner_fn,
        _original_ranges=op._original_ranges,
        _original_reduction_ranges=op._original_reduction_ranges,
    )
    new_buf.operation_name = op.operation_name
    preserve_provenance(op, new_buf, pass_name=pass_name, reason=reason)
    copy_op_metadata(op, new_buf)
    ComputedBuffer.get_default_sizes_body.clear_cache(new_buf)

    op_idx = operations.index(op)
    operations[op_idx] = new_buf
    _repoint_mutation_targets(operations, op, new_buf)
    return new_buf


class NameSwapHandler(WrapperHandler):
    """Patch an inner_fn's ``load`` calls to read renamed buffers.

    Used after inserting a producer upstream (e.g. a restickify or an identity
    clone) that supersedes an existing input: the consumer's inner_fn still
    names the old buffer, so wrap it to remap each ``load(old_name, ...)`` to
    ``load(new_name, ...)``.

    This is the canonical WrapperHandler wrapping pattern for compiler passes:
    wrap, never rebuild index expressions from scratch (they go stale — see
    CLAUDE.md "Compiler Pass Conventions" and issue #2797).
    """

    def __init__(self, inner, name_map: dict[str, str]):
        super().__init__(inner)
        self._name_map = name_map

    def load(self, name, index):
        return super().load(self._name_map.get(name, name), index)


def redirect_computed_buffer_reads(
    op: ComputedBuffer,
    name_map: dict[str, str],
    operations: list[Operation],
    *,
    pass_name: str,
    reason: str | None = None,
) -> ComputedBuffer:
    """Redirect ``op``'s reads through ``name_map`` and reconstruct the buffer.

    Wraps ``op.data.inner_fn`` with ``NameSwapHandler`` so every ``load`` of a
    remapped buffer resolves to its replacement, then reconstructs the frozen
    ``ComputedBuffer`` so the instance-keyed ``get_default_sizes_body`` cache is
    cleanly invalidated (the reconstruct is the reason both this helper and
    ``replace_computed_buffer_body`` rebuild rather than mutate in place). Also
    repoints any ``MutationLayoutSHOULDREMOVE.target`` elsewhere in
    ``operations`` that referenced the old object (see
    ``_repoint_mutation_targets``).

    Returns the replacement ComputedBuffer.
    """
    # Patch inner_fn once with the full name_map covering all remapped args.
    orig_inner = op.data.inner_fn

    def new_inner_fn(*args, _map=name_map, _orig_inner=orig_inner):
        with V.set_ops_handler(NameSwapHandler(V.ops, _map)):
            return _orig_inner(*args)

    object.__setattr__(op.data, "inner_fn", new_inner_fn)

    # Reconstruct ComputedBuffer as a fresh object so the instance-keyed cache
    # on get_default_sizes_body can be cleanly invalidated below.
    new_buf = ComputedBuffer(
        name=op.get_name(),
        layout=op.layout,
        data=op.data,
        _split_size=op._split_size,
        _original_inner_fn=op._original_inner_fn,
        _original_ranges=op._original_ranges,
        _original_reduction_ranges=op._original_reduction_ranges,
    )
    new_buf.operation_name = op.operation_name
    preserve_provenance(op, new_buf, pass_name=pass_name, reason=reason)
    copy_op_metadata(op, new_buf)

    op_idx = operations.index(op)
    operations[op_idx] = new_buf
    V.graph.name_to_buffer[new_buf.get_name()] = new_buf
    _repoint_mutation_targets(operations, op, new_buf)

    # Invalidate the sizes/body cache so it is recomputed on next access with
    # the patched inner_fn.
    ComputedBuffer.get_default_sizes_body.clear_cache(new_buf)
    return new_buf


def lower_pad_sequence(
    arg_fx_node: torch.fx.Node,
    padded_size: list[int],
    device: torch.device,
    dtype: torch.dtype,
    dim: int,
    insert_before: torch.fx.Node,
    orig_stl: SpyreTensorLayout,
    fill_value: float = 0.0,
) -> tuple[Buffer, list[Operation]]:
    """Lower an IR-level pad sequence that extends a buffer along one dimension.

    Allocates a padded buffer of ``padded_size``, fills the pad region with
    ``fill_value``, then copies the original data into offset 0 along ``dim``.
    Only one dimension may differ between ``padded_size`` and the original shape.

    Uses torch.ops.aten.constant_pad_nd which lowers to a 4-op IR sequence:
      1. ComputedBuffer - output buffer allocation (FixedLayout)
      2. SpyreConstantFallback - fill constant (FixedLayout)
      3. ComputedBuffer - fill padding region (MutationLayoutSHOULDREMOVE)
      4. ComputedBuffer - copy input data (MutationLayoutSHOULDREMOVE)

    constant_pad_nd is called with align_to_stick=True to ensure the padding region
    is filled with stick-aligned offsets. This is required because the dim is
    ensured to be a stick dimension here.

    ``orig_stl`` is the ``SpyreTensorLayout`` of the unpadded buffer and is used
    to derive the padded buffer's device layout, preserving the within-stick host
    dimension.  Raises ``RuntimeError`` if the within-stick dimension cannot be
    determined from ``orig_stl``.

    Deduplication of identical constants across multiple pad calls happens later
    at the IR level via dedup_and_promote_constants.

    Returns ``(padded_buf, new_ops)`` where ``padded_buf`` is the allocated buffer
    and ``new_ops`` is the list of new IR operations in topological order.
    """

    graph_lowering = V.graph
    fx_graph = graph_lowering.graph

    # Count operations before lowering so we can identify newly added ones.
    ops_before = len(graph_lowering.operations)

    original_shape = list(arg_fx_node.meta["val"].shape)
    assert len(padded_size) == len(original_shape), (
        f"lower_pad_sequence: padded_size rank {len(padded_size)} != "
        f"original rank {len(original_shape)}"
    )
    padded_dims = [
        i for i in range(len(padded_size)) if padded_size[i] != original_shape[i]
    ]
    assert padded_dims == [dim], (
        f"lower_pad_sequence: expected exactly dim={dim} to be padded, "
        f"but padded_size={padded_size} differs from original={original_shape} at dims={padded_dims}"
    )
    original_size_dim: int = original_shape[dim]
    pad_extent = padded_size[dim] - original_size_dim
    assert pad_extent > 0, (
        f"lower_pad_sequence: pad_extent={pad_extent} for dim={dim}; "
        f"padded_size={padded_size}, original_size_dim={original_size_dim}"
    )

    # Build pad tuple for constant_pad_nd: (left, right) pairs in reverse dimension order
    # We're padding only one dimension, so most pairs are (0, 0)
    pad_tuple = []
    for i in range(len(original_shape) - 1, -1, -1):
        if i == dim:
            # Pad at the end of this dimension
            pad_tuple.extend([0, pad_extent])
        else:
            pad_tuple.extend([0, 0])

    with fx_graph.inserting_before(insert_before):
        # Single constant_pad_nd call (lowers to 4 IR operations)
        pad_fx = fx_graph.create_node(
            "call_function",
            torch.ops.aten.constant_pad_nd.default,
            args=(arg_fx_node, pad_tuple, fill_value),
            kwargs={"align_to_stick": True},
        )
        pad_fx.meta["val"] = torch.empty(padded_size, dtype=dtype, device=device)

    # Lower the constant_pad_nd node, assigning FixedTiledLayouts immediately.
    # propagate_spyre_tensor_layouts already ran, so the new op keep FlexibleLayout
    # unless we assign here.
    pad_tb = graph_lowering.run_node(pad_fx)
    graph_lowering.env[pad_fx] = pad_tb
    padded_buf = pad_tb.data.data  # TensorBox -> StorageBox -> Buffer

    # Collect all newly added operations (appended at the end of graph.operations).
    new_ops = graph_lowering.operations[ops_before:]

    assert new_ops[0] == padded_buf

    # Verify structure: constant_pad_nd lowers to 4 operations
    #   op0: ComputedBuffer - output buffer allocation (FixedLayout)
    #   op1: SpyreConstantFallback - fill constant (FixedLayout)
    #   op2: ComputedBuffer - fill padding region (MutationLayoutSHOULDREMOVE)
    #   op3: ComputedBuffer - copy input data (MutationLayoutSHOULDREMOVE)
    assert (
        len(new_ops) == 4
        and isinstance(new_ops[0], ComputedBuffer)
        and isinstance(new_ops[0].get_layout(), FixedLayout)
        and isinstance(new_ops[1], SpyreConstantFallback)
        and isinstance(new_ops[1].get_layout(), FixedLayout)
        and isinstance(new_ops[2], ComputedBuffer)
        and isinstance(new_ops[2].get_layout(), MutationLayoutSHOULDREMOVE)
        and isinstance(new_ops[3], ComputedBuffer)
        and isinstance(new_ops[3].get_layout(), MutationLayoutSHOULDREMOVE)
    )

    # --- Build the device layout (SpyreTensorLayout) for the padded buffer. ---
    #
    # We need to know two things to construct the padded STL:
    #   1. The "core" host shape — the dimensions that orig_stl was actually
    #      built from.  mm_to_bmm_pass sometimes adds a leading batch=1 dim to
    #      padded_size (the view the matmul inner_fn uses) while leaving the
    #      underlying buffer 2D.  Passing that phantom dim to SpyreTensorLayout
    #      would produce a degenerate 4D device layout with a -1 sentinel stride
    #      for the size-1 device dim, which causes compute_coordinates to emit a
    #      constant nonzero stick offset and normalize_coordinates to assert.
    #      We strip phantom dims by comparing padded_size rank against the host
    #      rank implied by orig_stl: stride_map has one entry per device dim, and
    #      device dims = host dims + 1 (the extra entry is the within-stick dim),
    #      so orig_host_ndim = len(stride_map) - 1.
    #   2. Which host dimension is the within-stick dimension.  SpyreTensorLayout
    #      takes an explicit dim_order whose last element names the within-stick
    #      host dim; we must carry this over from the original buffer so that the
    #      padded buffer's device coordinates use the same stick dimension.  We
    #      identify it by matching orig_stl.stride_map[-1] (the within-stick
    #      element stride, always 1 for contiguous layouts) against the original
    #      buffer's host strides.

    # Step 1 — strip phantom batch dims to get the core host shape.
    orig_host_ndim = len(list(orig_stl.stride_map)) - 1
    n_phantom = len(padded_size) - orig_host_ndim
    padded_core = padded_size[n_phantom:]

    # Step 2 — identify the within-stick host dim in the view (which may include
    # phantom leading dims) by matching the within-stick element stride.
    # TODO: replace this sm_last heuristic with _resize_device_layout from
    # coarse_tile.py (device-native reconstruction that handles transposed and
    # non-contiguous layouts without guessing from stride_map[-1]).
    sm_last = int(list(orig_stl.stride_map)[-1])
    orig_host_stride = list(arg_fx_node.meta["val"].stride())
    within_stick_dim_view = next(
        (i for i, s in enumerate(orig_host_stride) if int(s) == sm_last), None
    )
    if within_stick_dim_view is None:
        raise RuntimeError(
            f"lower_pad_sequence: cannot determine within-stick host dimension for "
            f"buffer {arg_fx_node.name!r}: orig_stl.stride_map[-1]={sm_last} not found "
            f"in view strides {orig_host_stride}.  orig_stl={list(orig_stl.device_size)} "
            f"stride_map={list(orig_stl.stride_map)}, padded_size={padded_size}"
        )

    # Step 3 — translate the within-stick dim index from view space to core space
    # (subtract the number of phantom dims that were stripped in step 1).
    within_stick_dim_core = within_stick_dim_view - n_phantom

    # Step 4 — build dim_order for SpyreTensorLayout: all non-stick dims in their
    # natural order, followed by the within-stick dim last.  This tells the STL
    # constructor which host dim maps to the innermost device (within-stick) axis.
    dim_order_core = [
        i for i in range(len(padded_core)) if i != within_stick_dim_core
    ] + [within_stick_dim_core]

    # Step 5 — compute row-major strides for the padded core shape.  These are
    # host strides, not device strides; SpyreTensorLayout derives the device
    # layout (sticks, rows, …) from the host shape + dim_order.
    core_stride = [1] * len(padded_core)
    for i in range(len(padded_core) - 2, -1, -1):
        core_stride[i] = core_stride[i + 1] * padded_core[i + 1]

    padded_stl = SpyreTensorLayout(padded_core, core_stride, dtype, dim_order_core)
    host_layout = padded_buf.layout
    padded_buf.layout = FixedTiledLayout(
        host_layout.device,
        host_layout.dtype,
        host_layout.size,
        host_layout.stride,
        padded_stl,
    )

    # LX planning (scratchpad.py) accesses op.origin_node directly on the ComputedBuffer,
    # so we set it here explicitly.
    object.__setattr__(padded_buf, "origin_node", pad_fx)

    # propagate_spyre_tensor_layouts already ran before this pass, so any op
    # lowered here keeps FlexibleLayout unless we assign a FixedTiledLayout
    # immediately. The constant buffer (new_ops[1]) is a scalar tensor (size=[]).
    const_buf = new_ops[1]
    const_layout = const_buf.get_layout()
    const_stl = SpyreTensorLayout(const_layout.size, const_layout.dtype)
    const_buf.layout = FixedTiledLayout(
        const_layout.device,
        const_layout.dtype,
        const_layout.size,
        const_layout.stride,
        const_stl,
    )

    # Mutation ops are intentionally left untouched

    assert (
        len(new_ops) == 4
        and isinstance(new_ops[0].get_layout(), FixedTiledLayout)
        and isinstance(new_ops[1].get_layout(), FixedTiledLayout)
        and isinstance(new_ops[2].get_layout(), MutationLayoutSHOULDREMOVE)
        and isinstance(new_ops[3].get_layout(), MutationLayoutSHOULDREMOVE)
    )

    return padded_buf, list(new_ops)


@dataclass(frozen=True)
class PerCoreView:
    """Geometric description of a buffer's per-core slicing.

    - work_slice_dims: (device-dim index, split factor) pairs, one per
      split dim.
    - core_to_slot: (device-dim index, slice-index expression in core_id)
      pairs giving each core's position along that split dim.
    - num_cores: physical cores over which ``core_to_slot`` is defined. This
      can exceed the number of logical slices when data is replicated.

    The first two fields are keyed by the buffer's device-dim index — not by
    op-local iter symbols — so the value depends only on the buffer's physical
    slicing.

    Example: a 2D buffer split 4-ways on dim 0 across 4 cores has
        work_slice_dims = ((0, 4),)
        core_to_slot    = ((0, Mod(core_id, 4)),)
    so core_id=2 owns slot 2 along dim 0.
    """

    work_slice_dims: tuple[tuple[int, int], ...]
    core_to_slot: tuple[tuple[int, Expr], ...]
    num_cores: int | None = None


def _is_matmul_op(op: Operation) -> bool:
    return (
        isinstance(op, ComputedBuffer)
        and isinstance(op.data, Reduction)
        and op.data.reduction_type in MATMUL_REDUCTION_OPS
    )


def is_topk(op: Operation) -> bool:
    """Return True iff ``op`` is a ``ComputedBuffer`` computing a topk reduction."""
    return (
        isinstance(op, ComputedBuffer)
        and isinstance(op.data, Reduction)
        and op.data.reduction_type in TOPK_OPS
    )


def is_keep_by_index(op: Operation) -> bool:
    """Return True iff ``op`` is a ``ComputedBuffer`` computing a keep_by_index reduction."""
    return (
        isinstance(op, ComputedBuffer)
        and isinstance(op.data, Reduction)
        and op.data.reduction_type == KEEP_BY_INDEX_OP
    )


def read_with_max_reduction_overlap(
    reads: Any,
    reduction_vars: set[sympy.Symbol],
) -> Optional[sympy.Expr]:
    """Return the non-indirect read index with the most reduction-var coefficients.

    For keep_by_index, the indices tensor carries the k (reduction) dimension
    while the values tensor doesn't, so the read that should drive
    ``splits_by_index_coeff`` / ``apply_splits_from_index_coeff`` isn't
    necessarily the first non-indirect read. Instead, pick whichever
    non-indirect read's index has nonzero coefficients on the most
    ``reduction_vars`` symbols. Returns None if no read has any overlap.
    """
    best_read = None
    best_count = 0
    for d in reads:
        if isinstance(d, MemoryDep) and not d.is_indirect():
            red_count = sum(
                1 for red_var in reduction_vars if d.index.coeff(red_var) != 0
            )
            if red_count > best_count:
                best_count = red_count
                best_read = d.index
    return best_read


# TODO: Select and store the core mapping before LX planning, then pass the
# winning mapping to codegen.
class _ViewPrep(NamedTuple):
    """Candidate-invariant precompute shared across every core-division
    candidate of one ``(op, dep, buf_name)``.

    Everything here depends only on the op, its dep, and the target buffer (not
    on ``op.op_it_space_splits``), so ``_prepare_per_core_view`` computes it once
    and ``_per_core_view_from_prep`` reuses it per candidate -- hoisting the
    sympy-heavy op-level work out of the per-candidate loop.
    """

    iter_space: dict
    write_index: "sympy.Expr"
    read_index: "sympy.Expr"
    # concretize_expr(dep.index.coeff(sym)) over the *full* iteration space, so
    # the per-candidate path does a dict lookup instead of a sympy .coeff() call.
    dep_coeff: dict
    # The target dependency projected into the buffer's physical device
    # coordinates.  A single physical axis can contain multiple logical loop
    # symbols after a reshape; their relative host strides determine whether a
    # split owns one contiguous slice or several interleaved slices.
    dep_device_coordinates: tuple["sympy.Expr", ...]
    device_size: Any
    stride_map: Any
    elems_per_stick: int
    device_stride_to_dim: dict
    stick_host_stride: Optional[int]
    num_stick_dim: Optional[int]
    num_stick: int
    num_stick_stride: int
    is_matmul: bool


def _prepare_per_core_view(
    op: Operation,
    dep: MemoryDep,
    buf_name: str,
    *,
    parts: "Optional[tuple[dict, sympy.Expr, sympy.Expr]]" = None,
) -> Optional[_ViewPrep]:
    """Compute the candidate-invariant pieces of a per-core view once.

    Returns ``None`` when ``buf_name``'s layout is not a ``FixedTiledLayout`` --
    the view is then unrepresentable for *every* candidate, so callers map
    ``None`` to the unrepresentable result without entering the per-candidate
    path.

    ``parts`` supplies ``(iter_space, write_index, read_index)`` explicitly.
    Default (None) reads them off ``op`` -- the *pre*-scheduler ranges, which is
    what LX planning sees. Post-fusion callers pass the ``SchedulerNode``'s
    ranges instead, so they model the order codegen will actually emit.
    """
    if parts is not None:
        iter_space, write_index, read_index = parts
    else:
        # The op-level write_index / read_index (for *any* buffer the op writes /
        # reads, not necessarily buf_name) bridge stride-keyed coeff_splits back
        # to scheduler symbols.
        rw = op_read_writes(op)
        write_index = next(iter(rw.writes)).index
        read_index = next((d.index for d in rw.reads), write_index)
        iter_space = iteration_space_from_op(op)

    buf_op = V.graph.get_buffer(buf_name)
    buf_layout = buf_op.layout
    if not isinstance(buf_layout, FixedTiledLayout):
        return None

    if is_topk(op):
        return None

    dev_layout = buf_layout.device_layout
    device_size = dev_layout.device_size
    stride_map = dev_layout.stride_map
    elems_per_stick = dev_layout.device_dtype.elems_per_stick()

    # Device-dim placement maps -- depend only on the buffer layout.
    device_stride_to_dim: dict[int, int] = {}
    for i, s in enumerate(stride_map):
        if s <= 0:
            continue
        prev = device_stride_to_dim.get(s)
        if prev is None or device_size[i] != 1:
            device_stride_to_dim[s] = i

    stick_host_stride, num_stick_dim, num_stick, num_stick_stride = None, None, 0, 0
    if stride_map[-1] > 0:
        stick_host_stride = stride_map[-1]
        num_stick_dim = device_stride_to_dim.get(stick_host_stride * elems_per_stick)
        if num_stick_dim is not None:
            num_stick = device_size[num_stick_dim]
            num_stick_stride = stride_map[num_stick_dim]

    # Per-symbol host stride on this buffer, over the full iteration space.
    # ``apply_splits_from_index_coeff`` returns a dict keyed by exactly the
    # iteration-space symbols, so precomputing the coeff for every iter symbol
    # covers every symbol the per-candidate path can ask for.
    dep_coeff = {sym: concretize_expr(dep.index.coeff(sym)) for sym in iter_space}
    coordinates = try_device_coordinates(dev_layout, dep, None)
    if coordinates is None:
        return None
    dep_device_coordinates = tuple(coordinates)

    return _ViewPrep(
        iter_space=iter_space,
        write_index=write_index,
        read_index=read_index,
        dep_coeff=dep_coeff,
        dep_device_coordinates=dep_device_coordinates,
        device_size=device_size,
        stride_map=stride_map,
        elems_per_stick=elems_per_stick,
        device_stride_to_dim=device_stride_to_dim,
        stick_host_stride=stick_host_stride,
        num_stick_dim=num_stick_dim,
        num_stick=num_stick,
        num_stick_stride=num_stick_stride,
        is_matmul=_is_matmul_op(op),
    )


def _per_core_view_from_prep(
    prep: Optional[_ViewPrep],
    splits: dict[sympy.Symbol, int] | tuple[dict, dict],
    reduction_splits: Optional[dict[sympy.Symbol, int]] = None,
) -> tuple[PerCoreView, bool, bool]:
    """Evaluate a precomputed view for symbol splits or scheduler transport."""
    num_cores = math.prod(
        value
        for group in (splits if isinstance(splits, tuple) else (splits,))
        for value in group.values()
    )
    # 3-tuple: (view, has_partial_reduction, representable). ``representable`` is
    # False only on the give-up returns below (a split that slices this buffer
    # can't be placed on a device dim); cross-op view comparisons must treat it
    # as "no match", since its empty view means "couldn't tell", not "whole".
    unrepresentable = (
        PerCoreView(work_slice_dims=(), core_to_slot=(), num_cores=num_cores),
        False,
        False,
    )

    # No real split -> whole-buffer view, representable regardless of layout. Must
    # precede the ``prep is None`` guard to match the original ordering.
    if isinstance(splits, tuple):
        if not any(n > 1 for d in splits for n in d.values()):
            return (
                PerCoreView(work_slice_dims=(), core_to_slot=(), num_cores=num_cores),
                False,
                True,
            )
        if prep is None:
            return unrepresentable
        per_sym = apply_splits_from_index_coeff(
            splits, prep.write_index, prep.read_index, prep.iter_space
        )
        has_partial_reduction = any(n > 1 for n in splits[1].values())
    else:
        if not any(n > 1 for n in splits.values()):
            return (
                PerCoreView(work_slice_dims=(), core_to_slot=(), num_cores=num_cores),
                False,
                True,
            )
        if prep is None:
            return unrepresentable
        per_sym = {sym: int(splits.get(sym, 1)) for sym in prep.iter_space}
        has_partial_reduction = any(n > 1 for n in (reduction_splits or {}).values())

    # Step 2: keep splits that actually slice this buffer, keyed by their host
    # stride on buf (precomputed in ``dep_coeff``). host_stride == 0 means the
    # split contracts an axis not present on this buffer (canonical case: a
    # K-split's output dep) and is dropped from the geometry. The
    # has_partial_reduction flag is op-level -- set whenever the op has any
    # reduction-axis split -- and is independent of which dep we're inspecting.
    splits_by_stride: dict[int, tuple[int, "sympy.Symbol"]] = {}
    for sym, split in per_sym.items():
        host_stride = prep.dep_coeff.get(sym, 0)
        if split <= 1 or host_stride == 0:
            continue
        splits_by_stride[host_stride] = (int(split), sym)

    device_size = prep.device_size
    stride_map = prep.stride_map
    elems_per_stick = prep.elems_per_stick
    device_stride_to_dim = prep.device_stride_to_dim
    stick_host_stride = prep.stick_host_stride
    num_stick_dim = prep.num_stick_dim
    num_stick = prep.num_stick
    num_stick_stride = prep.num_stick_stride
    iter_space = prep.iter_space

    # Step 3: place each split on a device dim via stride lookup.
    #
    # stride_map[i] is a device-dim → host-stride mapping. The stickified
    # host dim decomposes into two device dims (per dim_map_to_stride_map in C++):
    #   - within-stick dim: always at position n-1, with
    #     stride_map[-1] = host_stride[stick_dim] and dev_size = elems_per_stick.
    #   - outer-stick (num_stick) dim: stride = stride_map[-1] * elems_per_stick,
    #     dev_size = ceil(host_size[stick_dim] / elems_per_stick).
    # A split whose host stride h equals stick_host_stride lands on the
    # stickified host dim; sticks are atomic, so it must use the outer-stick
    # dim. Skip stride_map entries <= 0 — sentinels for collapsed or
    # broadcast dims.
    #
    # Example: host [64, 128] sticked to device [2, 64, 64] with
    # stride_map=[64, 128, 1] and elems_per_stick=64. stick_host_stride=1,
    # num_stick_dim=dim 0 (stride 64). With M-split×4 (h=128) and N-split×2
    # (h=1), N's h matches stick_host_stride → outer-stick dim 0; M's h=128
    # → dim 1. Result: work_slice_dims={0: 2, 1: 4}.
    # ``device_stride_to_dim`` and the stick vars are candidate-invariant and
    # come from ``prep`` (bound above).
    work_slice_dims: dict[int, int] = {}
    sym_to_device_dim: dict["sympy.Symbol", int] = {}
    for h, (split, sym) in sorted(splits_by_stride.items()):
        dev_dim = device_stride_to_dim.get(h)
        if h == stick_host_stride:
            dev_dim = num_stick_dim
        # A lower-rank reshape can flatten outer axes into the stickified axis.
        # Mapping that loop only by its host-stride then falsely attributes the
        # split to num_stick_dim. The full physical capacity of that host axis
        # is known exactly, so an iteration spanning beyond it proves compound
        # ownership that PerCoreView cannot express. Keep the buffer in HBM.
        if h == stick_host_stride and dev_dim is not None:
            iter_extent_expr = iter_space[sym]
            if isinstance(iter_extent_expr, tuple):
                iter_extent_expr = iter_extent_expr[0]
            iter_extent = concretize_expr(iter_extent_expr)
            stickified_extent = num_stick * elems_per_stick
            if iter_extent > stickified_extent:
                logger.debug(
                    f"iteration {sym} extent {iter_extent} spans beyond mapped "
                    f"stickified dim {dev_dim} extent {stickified_extent}; "
                    f"returning empty_view"
                )
                return unrepresentable

        # Multi-stick-stride rescue: a consumer view subdivides the stickified
        # axis at k sticks per step (h = k * num_stick_stride). Only safe when
        # split*k fully covers num_stick_dim — partial coverage would
        # misreport the per-dim factor. Example (test_view_unsqueeze_add):
        # device_size=[2, 6, 1, 64], num_stick_dim=1 (stride 64); split=3,
        # h=128, k=2, split*k=6 == device_size[1] → place on dim 1, factor 6.
        if dev_dim is None and num_stick_stride > 0 and h % num_stick_stride == 0:
            k = h // num_stick_stride
            if split * k == num_stick:
                dev_dim = num_stick_dim
                split *= k

        # A reshape can place more than one logical loop symbol on one physical
        # device axis.  Splitting an inner symbol while an unsplit outer symbol
        # also contributes to that axis gives each core several interleaved
        # regions, which PerCoreView's single ``(axis, slice)`` cannot express.
        #
        # For example, ``4 * outer + floor(inner / 64)`` over an 8-stick axis,
        # with ``inner`` split in two, gives the two core groups ownership of
        # {0, 1, 4, 5} and {2, 3, 6, 7}; it is not the contiguous 2-way split
        # represented by ``work_slice_dims=(axis, 2)``.  Host-index coefficient
        # order identifies those outer contributors without depending on the
        # iteration order of SymPy's ``free_symbols`` set.
        if dev_dim is not None:
            axis_symbols = prep.dep_device_coordinates[dev_dim].free_symbols
            if any(
                other != sym
                and per_sym.get(other, 1) <= 1
                and prep.dep_coeff.get(other, 0) > h
                for other in axis_symbols
            ):
                logger.debug(
                    f"split iteration {sym} is interleaved by an outer loop "
                    f"on device dim {dev_dim}; returning empty_view"
                )
                return unrepresentable
        # TODO: two known unhandled failure modes fall through to the
        # empty_view fallback (cases catalogued in
        # per_core_view_failing_cases.md):
        #   (A) Collapsed-axis info loss — device_layout built from a
        #       higher-rank host tensor while dep is indexed via a lower-rank
        #       reshape view; the work-split factor spans multiple device
        #       dims but reaches us as a single (h, factor) (e.g.
        #       test_matmul_tiled_y, test_qkv_attn_paths_fms_*_gqa).
        #   (B) Multi-stick stride with partial coverage —
        #       h = k * stride_map[num_stick_dim], k > 1, but
        #       split * k < num_stick (rescue above only
        #       handles the full-coverage case where they are equal).
        # In both cases no single (dev_dim, factor) placement faithfully
        # represents the per-core slicing; empty_view keeps the buffer on
        # HBM via the caller's mismatch logic. Future work: extend the
        # PerCoreView schema to express multi-dim or strided splits, or
        # refuse the buffer earlier in scratchpad planning.
        if (
            dev_dim is None
            or dev_dim in work_slice_dims
            or device_size[dev_dim] % split != 0
        ):
            logger.debug(
                f"could not place split h={h} factor={split} on "
                f"stride_map={stride_map} device_size={device_size}; "
                f"returning empty_view"
            )
            return unrepresentable
        work_slice_dims[dev_dim] = split
        sym_to_device_dim[sym] = dev_dim

    # Step 4: model the same physical ownership SDSC will emit. LX compatibility
    # requires producer and consumer to assign each slice to the same physical
    # core; matching split factors alone is insufficient.
    num_cores = int(math.prod(per_sym.values()))
    iter_symbols = tuple(iter_space)
    dim_splits = tuple(int(per_sym[sym]) for sym in iter_symbols)
    contiguous_dim = (
        len(dim_splits) - 1
        if prep.is_matmul and config.core_id_k_fast_emission
        else None
    )
    core_to_slot = core_to_slice_mapping(
        iter_symbols,
        dim_splits,
        num_cores,
        contiguous_dim=contiguous_dim,
    )
    # Re-key by the buffer's device-dim index (canonical) instead of the op's
    # iter symbol name. Two ops with the same per-core slicing on this buffer
    # compare equal even if they name their iter axes differently.
    pruned_core_to_slot: list[tuple[int, "Expr"]] = []
    for sym, dev_dim in sym_to_device_dim.items():
        pruned_core_to_slot.append((dev_dim, core_to_slot[sym]))
    pruned_core_to_slot.sort(key=lambda x: x[0])

    view = PerCoreView(
        work_slice_dims=tuple(sorted(work_slice_dims.items())),
        core_to_slot=tuple(pruned_core_to_slot),
        num_cores=num_cores,
    )
    return (view, has_partial_reduction, True)


def _per_core_view_on_buf(
    op: Operation,
    dep: MemoryDep,
    buf_name: str,
    cache: Optional[dict] = None,
) -> tuple[PerCoreView, bool, bool]:
    """Build a PerCoreView describing how `op` slices `buf_name` via `dep`.

    Thin wrapper over ``_prepare_per_core_view`` + ``_per_core_view_from_prep``,
    reading the candidate division from ``op.op_it_space_splits``. Callers
    sweeping many candidates of one op/edge should call those two directly to
    amortize the op-level precompute.

    Returns `(view, has_partial_reduction, representable)`. ``has_partial_reduction``
    is True when the op has a reduction split (partial sums left on most cores);
    callers act on it only for write-deps. ``representable`` is False only on the
    give-up cases (a split that slices this buffer can't be placed on a device
    dim), which cross-op comparisons must treat as a non-match. Pass `cache` to
    memoize, keyed by (op name, symbol-keyed splits, dep, buf_name).

    The op name is part of the key because the result also depends on op-derived
    write_index / read_index / iter_space / matmul-ness, not just (splits, dep,
    buf_name): two different ops can share the same (splits, dep, buf_name) — e.g.
    a producer's write-dep and a consumer's read-dep on the same buffer at the
    same index — and must NOT alias the same entry (both
    ``ScratchpadAllocator._cd_parent_matches`` and ``get_ncores_for_buffers``
    share one cache across a producer and consumer of the same buffer).
    """
    ownership = getattr(op, "iteration_space_ownership", None)
    splits = ownership.work_slices if ownership is not None else {}
    key = None
    if cache is not None:
        key = (op.get_name(), frozenset(splits.items()), dep, buf_name)
        hit = cache.get(key)
        if hit is not None:
            return hit

    if not any(n > 1 for n in splits.values()):
        num_cores = ownership.num_cores if ownership is not None else 1
        result = (
            PerCoreView(work_slice_dims=(), core_to_slot=(), num_cores=num_cores),
            False,
            True,
        )
        if cache is not None:
            cache[key] = result
        return result

    prep = _prepare_per_core_view(op, dep, buf_name)
    write_index = prep.write_index if prep is not None else sympy.S.Zero
    reduction_splits = {
        sym: split for sym, split in splits.items() if write_index.coeff(sym) == 0
    }
    result = _per_core_view_from_prep(prep, splits, reduction_splits)
    if cache is not None:
        cache[key] = result
    return result


def per_core_view_scheduled(
    node: "SchedulerNode", dep: MemoryDep, buf_name: str
) -> tuple[PerCoreView, bool, bool]:
    """:func:`_per_core_view_on_buf` against the *post*-scheduler ranges.

    Same result shape, but the iteration space and indices come from
    ``node.read_writes`` / :func:`iteration_space` rather than the pre-scheduler
    ``op.get_read_writes()``. Inductor's ``loop_ordering_after_fusion`` can
    permute a fused op's ranges after LX planning has already committed, and
    ``core_to_slice_mapping`` is positional, so only this post-fusion view
    reflects the core->slice assignment codegen will really emit.
    """
    op = node.node
    coeff_splits: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
    if not any(n > 1 for d in coeff_splits for n in d.values()):
        # No real split -> whole-buffer view; every core holds all of it.
        return (
            PerCoreView(work_slice_dims=(), core_to_slot=(), num_cores=1),
            False,
            True,
        )

    rw = node.read_writes
    write_dep = next((d for d in rw.writes if isinstance(d, MemoryDep)), None)
    if write_dep is None:
        # StarDep-only writer: no index to reason about, so treat as
        # unrepresentable rather than guessing.
        num_cores = math.prod(
            value for group in coeff_splits for value in group.values()
        )
        return (
            PerCoreView(work_slice_dims=(), core_to_slot=(), num_cores=num_cores),
            False,
            False,
        )
    read_index = next(
        (d.index for d in rw.reads if isinstance(d, MemoryDep)), write_dep.index
    )
    parts = (iteration_space(node), write_dep.index, read_index)
    prep = _prepare_per_core_view(op, dep, buf_name, parts=parts)
    return _per_core_view_from_prep(prep, coeff_splits)


def format_operations(operations: list[Operation]) -> str:
    """Format LLIR operations including torch-spyre custom metadata"""
    buf = io.StringIO()
    for op in operations:
        buf.write(f"{op.get_operation_name()}: {type(op).__name__}")
        if isinstance(op, ComputedBuffer):
            buf.write(f"\n  buffer={op.get_name()}")
            buf.write(f"\n  layout={op.layout}")
            if allocation := getattr(op.layout, "allocation", None):
                buf.write(f"\n  allocation={allocation}")
            if splits := getattr(op, "op_it_space_splits", None):
                rw = op.get_read_writes()
                write_index = next(iter(rw.writes)).index
                read_index = next((d.index for d in rw.reads), write_index)
                it_space = iteration_space_from_op(op)
                readable_splits = apply_splits_from_index_coeff(
                    splits, write_index, read_index, it_space
                )
                buf.write(f"\n  op_it_space_splits={readable_splits}")
            if dim_hints := getattr(op, "dim_hints", None):
                buf.write(f"\n  dim_hints={dim_hints}")
            if loop_info := getattr(op, "loop_info", None):
                buf.write(f"\n  loop_info={loop_info}")
            buf.write(f"\n  {op.data}")
        buf.write("\n\n")
    return buf.getvalue()
