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

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, TypeVar, Union

import torch
import sympy
from sympy import Expr
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
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.dependencies import MemoryDep, ReadWrites
from torch._inductor.virtualized import V
from torch_spyre._C import SpyreTensorLayout, get_elem_in_stick
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.op_spec import IndirectAccess

from . import config
from .codegen.superdsc import (
    _get_core_to_slice_mapping,
    _k_fast_core_to_slice_mapping,
    _should_use_k_fast_mapping,
)
from .constants import BATCH_MATMUL_OP, ELIDED_COPY_BACK_ATTR
from .ir import FixedTiledLayout, SpyreConstantFallback
from .logging_utils import get_inductor_logger
from .loop_info import copy_op_metadata
from .views import compute_coordinates, matching_dim

# PyTorch's default lower bound for size symbols (sizes 0/1 are specialised).
_SHAPE_ENV_DEFAULT_LOWER = 2
logger = get_inductor_logger("pass_utils")


class SchedNodeArg(NamedTuple):
    dep: MemoryDep
    layout: "FixedTiledLayout"


def _fixed_read_layout(buf) -> "FixedTiledLayout":
    layout = buf.get_layout()
    if isinstance(layout, MutationLayoutSHOULDREMOVE):
        if not getattr(buf, ELIDED_COPY_BACK_ATTR, False):
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
        return V.graph.sizevars.size_hint(expr)
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


def _finite_upper_or_none(expr: Expr) -> Optional[int]:
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

    Wiring: this helper has no call sites yet. The pointwise
    work-division PR (#2499) will plug it into the ``size_hint`` call
    sites in ``work_division.py`` and ``codegen/superdsc.py``,
    alongside ``compute_max_size``.

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
    # size_hint (via compute_max_size below, merged in #2003), not from
    # mark_dynamic(max=...). The granularity is then only as trustworthy
    # as that hint -- warn the user so they can pin it explicitly with
    # mark_dynamic(max=...).
    if _finite_upper_or_none(expr) is None:
        warnings.warn(
            f"max for symbolic dim {expr} came from size_hint, not from "
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

    size_syms = index.free_symbols - loop_vars
    if not size_syms:
        return index
    # Try each symbol individually
    subs = {}
    for s in size_syms:
        try:
            hint = V.graph.sizevars.size_hint(s)
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
    back to ``size_hint`` when no finite upper bound exists.

    Needed for dynamic shape support.

    # TODO: To be used in size_hint call-sites in superdsc.py and work_division.py
    #       to get the maxSize in SDSC and work planning respectively
    """
    if isinstance(expr, int):
        return expr
    if isinstance(expr, sympy.Integer):
        return int(expr)
    if not (hasattr(expr, "free_symbols") and expr.free_symbols):
        return int(expr)
    bound = _finite_upper_or_none(expr)
    if bound is not None:
        return bound
    return V.graph.sizevars.size_hint(expr)


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
    return host_coordinates(op.get_layout(), output_dep)


def _find_scatter_index_buf_names(op: ComputedBuffer) -> set[str]:
    """Return names of deps whose loaded values are used as indices in scatter output_indexer.

    For Scatter ops the indirect index is encoded in the output_indexer closure.
    Extract the index buffer names directly from the 'indices' closure variable.
    """
    from torch._inductor.ir import Scatter

    if not isinstance(op.data, Scatter):
        return set()

    fn = op.data.output_indexer
    if fn.__closure__ is None:
        return set()

    freevars = fn.__code__.co_freevars
    try:
        cells = {
            name: cell.cell_contents for name, cell in zip(freevars, fn.__closure__)
        }
    except ValueError:
        return set()

    if "indices" not in cells:
        logger.warning(
            "Scatter.output_indexer closure has no 'indices' variable — "
            "Inductor may have renamed it. Scatter index tensors will not be "
            "excluded from stick compatibility checks. (freevars: %s)",
            list(freevars),
        )
        return set()
    indices = cells["indices"]
    names = set()
    for idx_tensor in indices:
        if idx_tensor is None:
            continue
        # Unwrap TensorBox -> StorageBox -> Buffer to get the name
        node = idx_tensor
        while hasattr(node, "data"):
            node = node.data
        if hasattr(node, "name") and node.name is not None:
            names.add(node.name)
    return names


def indirect_index_dep_names(op: ComputedBuffer) -> set[str]:
    """Return names of deps whose loaded values are used as indices in loads or stores."""
    names = {expr.base.name for expr in _build_indirect_load_subs(op).values()}
    names |= _find_scatter_index_buf_names(op)
    return names


class _LoadSentinel:
    """Opaque token returned by load(); carries the buffer name through ops."""

    def __init__(self, name: str):
        self.name = name


class _IndirectIndexFinder:
    """Re-executes inner_fn to discover which buffer's load feeds each indirect load.

    load() returns a _LoadSentinel carrying the buffer name.
    indirect_indexing() receives that sentinel and records which buffer was
    used as the index, so the next load() call can record the relationship.
    All other ops delegate to MockHandler.
    """

    def __init__(self):
        from torch._inductor.ops_handler import MockHandler

        self._mock = MockHandler()
        self._pending_indirect_index_buf: str | None = None
        self.indirect_index_by_buf: dict[str, str] = {}

    def load(self, name: str, index):
        if self._pending_indirect_index_buf is not None:
            self.indirect_index_by_buf[name] = self._pending_indirect_index_buf
            self._pending_indirect_index_buf = None
        return _LoadSentinel(name)

    def indirect_indexing(self, index_var, size, check=True, wrap_neg=True):
        # Assumes load() is called immediately after — Inductor's aten.index
        # lowering always emits indirect_indexing() directly before the
        # consuming load(), so the single slot is never overwritten in between.
        if isinstance(index_var, _LoadSentinel):
            self._pending_indirect_index_buf = index_var.name
        return sympy.S.Zero

    def __getattr__(self, attr):
        return getattr(self._mock, attr)


def _find_indirect_index_bufs(op: ComputedBuffer) -> dict[str, str]:
    """Re-execute inner_fn and return {data_buf: indirect_index_buf} mapping."""
    from torch._inductor.virtualized import V as _V

    finder = _IndirectIndexFinder()
    with _V.set_ops_handler(finder):
        op.data.inner_fn(*op.data.inner_fn_args())
    return finder.indirect_index_by_buf


def _build_indirect_load_subs(op: ComputedBuffer) -> dict[sympy.Symbol, sympy.Expr]:
    """Map indirect symbols in MemoryDep.index to IndexedBase[source_index] exprs.

    Pre-scheduler only: re-executes inner_fn via _IndirectIndexFinder to learn
    which buffer's load produced each indirect index.
    """
    from sympy import IndexedBase

    rw = op.get_read_writes()
    reads = [
        d
        for d in rw.reads
        if isinstance(d, MemoryDep) and isinstance(d.index, sympy.Basic)
    ]
    if not any(d.is_indirect() for d in reads):
        return {}
    indirect_index_buf_map = _find_indirect_index_bufs(op)
    dep_by_name = {d.name: d for d in reads}
    result = {}
    for d in reads:
        if not d.is_indirect():
            continue
        indirect_index_buf = indirect_index_buf_map.get(d.name)
        if indirect_index_buf is None:
            continue
        indirect_index_dep = dep_by_name[indirect_index_buf]
        for sym in d.index.free_symbols:
            if sym not in d.ranges:
                result[sym] = IndexedBase(indirect_index_dep.name)[
                    indirect_index_dep.index
                ]
    return result


def indirect_access_subs_from_op(
    op: ComputedBuffer,
) -> "dict[sympy.Symbol, sympy.Expr]":
    """Build {indirect_sym → IndirectAccess(name)} for a ComputedBuffer (pre-scheduler).

    Used before scheduling, when indirect_vars is not yet available.
    Re-executes inner_fn via _IndirectIndexFinder to discover which buffer's
    load produced each indirect index. The resulting subs can be passed to
    device_coordinates() to replace indirect symbols with IndirectAccess(name).
    """
    raw = _build_indirect_load_subs(op)
    return {
        sym: IndirectAccess(sympy.Symbol(expr.base.name)) for sym, expr in raw.items()
    }


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


def host_coordinates(layout: FixedLayout, dep: MemoryDep) -> list[sympy.Expr]:
    # Concretize size/stride so compute_coordinates can use plain ``<``/``>``
    # comparisons.  var_ranges and index stay symbolic so the *output*
    # coordinate expressions remain symbolic.
    # TODO(issue#1373): remove concretization once compute_coordinates handles
    #              symbolic comparisons natively.
    concrete_size = [concretize_expr(s) for s in layout.size]
    concrete_stride = [concretize_expr(s) for s in layout.stride]
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    return compute_coordinates(concrete_size, concrete_stride, dep.ranges, index)


def identify_matmul_inputs(
    inputs: list[MemoryDep],
    write_dep: MemoryDep,
) -> tuple[MemoryDep, MemoryDep] | tuple[None, None]:
    """Identify Input1 (x) and Input2 (y) of a BatchMatmul op.

    Uses the BatchMatmul semantic dimension definitions:
      reduction_dim: in Input1, Input2,  NOT Output
      generated_dim: in Input2, Output,  NOT Input1
      preserved_dim: in Input1, Output,  NOT Input2
      noreuse_dim:   in Input1, Input2,  Output

    Identifies y by its generated_dim (N): present in y and the output, absent
    from x.  This is more robust than identifying x by its preserved_dim (M):
    when M=1, M is constant-folded out of both x's and the output's index
    simultaneously, making the preserved_dim test blind.  N is immune — even
    N=1 ranges stay in the output's index expression.

    Returns (None, None) if y cannot be identified.
    """
    assert len(inputs) == 2
    a, b = inputs[0], inputs[1]
    out_syms = write_dep.index.free_symbols
    syms_a = a.index.free_symbols
    syms_b = b.index.free_symbols

    # b has generated_dim → b is y, a is x
    if (syms_b & out_syms) - syms_a:
        return a, b
    # a has generated_dim → a is y, b is x
    if (syms_a & out_syms) - syms_b:
        return b, a
    return None, None


def find_reduction_var(x_dep: MemoryDep, out_dep: MemoryDep) -> sympy.Symbol:
    """Return the single loop variable that appears in x's index but not in the output's.

    Raises Unsupported if the count is not exactly 1.
    """
    reduction_vars = x_dep.index.free_symbols - out_dep.index.free_symbols
    if len(reduction_vars) != 1:
        raise Unsupported(
            f"expected exactly 1 reduction variable, got {reduction_vars}"
        )
    return next(iter(reduction_vars))


def find_matmul_generated_var(
    y_dep: MemoryDep, x_dep: MemoryDep, out_dep: MemoryDep
) -> sympy.Symbol:
    """Return the single loop variable that appears in y's and the output's index but not in x's.

    This is the N (generation) dimension of a matmul.
    Raises Unsupported if the count is not exactly 1.
    """
    generated_vars = (
        y_dep.index.free_symbols & out_dep.index.free_symbols
    ) - x_dep.index.free_symbols
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
    indirect_load_subs: "dict[sympy.Symbol, sympy.Expr] | None" = None,
) -> list[sympy.Expr]:
    """Compute device-space coordinates for a tensor access.

    indirect_load_subs: optional {indirect_sym → IndirectAccess(name)} mapping produced by
        indirect_access_subs_from_op() (pre-scheduler) or indirect_access_subs_from_kernel()
        (post-scheduler). When provided, indirect symbols in the coordinates are
        replaced with IndirectAccess expressions, giving indirect-aware coordinates.
    """
    # device_size and stride_map come from the C++ SpyreTensorLayout and are
    # already concrete, so no concretization is needed here.
    index = concretize_index(dep.index, set(dep.ranges.keys()))
    coords = compute_coordinates(
        stl.device_size,
        stl.stride_map,
        dep.ranges,
        index,
        indirect_load_subs,
    )
    _check_stick_expr_supported(coords[-1], stl.elems_per_stick())
    return coords


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
        # The iteration space of a Reduction is that of its input
        return next(iter(n.read_writes.reads)).ranges.copy()
    else:
        raise Unsupported("Unexpected node type")


def iteration_space_from_op(op: ComputedBuffer) -> dict[sympy.Symbol, sympy.Expr]:
    """Pre-scheduler version of iteration_space: uses op.get_read_writes() instead
    of SchedulerNode.read_writes."""
    rw = op.get_read_writes()
    if isinstance(op.data, Pointwise):
        return next(iter(rw.writes)).ranges.copy()
    elif isinstance(op.data, Reduction):
        return next(iter(rw.reads)).ranges.copy()
    else:
        raise Unsupported("Unexpected node type")


_V = TypeVar("_V")

# Type alias for the two-namespace split storage: (output_splits, reduction_splits).
# output_splits is keyed by the symbol's coefficient in the write dep's index.
# reduction_splits is keyed by the symbol's coefficient in the first read dep's index.
# The two dicts use different reference indices so their keys never collide.
ItSpaceSplits = tuple[dict[sympy.Expr, int], dict[sympy.Expr, int]]


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
    """Encode a symbol→split dict as a pair of coeff-keyed dicts.

    Output dims (those present in write_index) are encoded using their
    coefficient in write_index.  Reduction dims (absent from write_index) are
    encoded using their coefficient in read_index.  The two dicts form separate
    namespaces so their keys never collide, even when output and reduction dims
    happen to share the same stride value in different tensors.

    Only non-unity splits are stored; 1 is the default on the apply side.
    """
    skip = lambda v: v <= 1  # noqa: E731
    output_splits = _coeff_splits_from_index(splits, write_index, skip=skip)
    # Reduction splits: symbols with coeff==0 in write_index but coeff!=0 in read_index
    reduction_only = {
        sym: val for sym, val in splits.items() if write_index.coeff(sym) == 0
    }
    reduction_splits = _coeff_splits_from_index(reduction_only, read_index, skip=skip)
    return output_splits, reduction_splits


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
    if op is not None and in_dep.name in indirect_index_dep_names(op):
        return False, None
    idc = device_coordinates(in_stl, in_dep)
    out_idc = device_coordinates(out_stl, out_dep)
    assert idc, "device_coordinates returned empty list for input"
    assert out_idc, "device_coordinates returned empty list for output"
    # Input stick with an offset always needs restickify to remove the offset.
    in_stick_offset_free = is_stick_expr_offset_free(idc[-1], in_stl.elems_per_stick())
    if in_stick_offset_free and stick_compatible([idc, out_idc]):
        return False, None
    ic = host_coordinates(in_host, in_dep)
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


def replace_computed_buffer_body(
    op: ComputedBuffer,
    new_data: Loops,
    operations: list[Operation],
) -> ComputedBuffer:
    """Replace the body (``data``) of a ``ComputedBuffer`` with ``new_data``.

    ``ComputedBuffer`` is a frozen dataclass, so its ``data`` field cannot be
    mutated in place.  This function constructs a new ``ComputedBuffer`` with
    the updated body and swaps it into ``operations``, copying all metadata
    fields that downstream passes depend on: ``operation_name``, ``origins``,
    ``origin_node``, and the ``_split_size`` / ``_original_*`` fields used by
    ``get_default_sizes_body``.  The ``get_default_sizes_body`` cache is
    cleared on the new buffer so stale size results from the old body are not
    reused.

    Returns the replacement ComputedBuffer.
    """
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
    new_buf.origins = op.origins
    new_buf.origin_node = op.origin_node
    copy_op_metadata(op, new_buf)
    ComputedBuffer.get_default_sizes_body.clear_cache(new_buf)

    op_idx = operations.index(op)
    operations[op_idx] = new_buf
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

    Both fields are keyed by the buffer's device-dim index — not by op-
    local iter symbols — so the value depends only on the buffer's
    physical slicing.

    Example: a 2D buffer split 4-ways on dim 0 across 4 cores has
        work_slice_dims = ((0, 4),)
        core_to_slot    = ((0, Mod(core_id, 4)),)
    so core_id=2 owns slot 2 along dim 0.
    """

    work_slice_dims: tuple[tuple[int, int], ...]
    core_to_slot: tuple[tuple[int, Expr], ...]


def _is_matmul_op(op: Operation) -> bool:
    return (
        isinstance(op, ComputedBuffer)
        and isinstance(op.data, Reduction)
        and op.data.reduction_type == BATCH_MATMUL_OP
    )


# TODO: refactor core assignment so the LX planner consumes determined
# assignments instead of re-deriving them here.
def _per_core_view_on_buf(
    op: Operation,
    dep: MemoryDep,
    buf_name: str,
    cache: Optional[dict] = None,
) -> tuple[PerCoreView, bool]:
    """Build a PerCoreView describing how `op` slices `buf_name` via `dep`.

    Returns `(view, has_partial_reduction)`. The flag is True when this
    op's iteration space contains a reduction split — meaning the
    producer leaves partial sums on most cores. Callers act on it only
    for write-deps; a read-dep on a K-split input still has a valid work
    slice.

    Steps:
      1. Recover {iter-symbol: split} from op.op_it_space_splits.
      2. Filter to splits that actually slice this buffer (host_stride
         != 0 in dep.index).
      3. Place each remaining split on a device dim via stride lookup,
         producing work_slice_dims keyed by device-dim index.
      4. Build the core-to-slot mapping (k_fast-aware) and re-key it
         by device-dim so it's independent of op-local symbol names.

    Pass an optional `cache` dict to memoize results across calls,
    keyed by (op.op_it_space_splits, dep, buf_name).
    """
    coeff_splits: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
    if cache is not None:
        # dicts aren't hashable; freeze each into a frozenset of items so
        # the key is hashable and order-independent.
        out, red = coeff_splits
        key = (frozenset(out.items()), frozenset(red.items()), dep, buf_name)
        hit = cache.get(key)
        if hit is not None:
            return hit

    # Step 1: recover {iter-symbol: split} from op.op_it_space_splits.
    # The op-level write_index / read_index (for *any* buffer the op
    # writes / reads, not necessarily buf_name) bridge stride-keyed
    # coeff_splits back to scheduler symbols.
    rw = op.get_read_writes()
    empty_view = (PerCoreView(work_slice_dims=(), core_to_slot=()), False)
    if not any(n > 1 for d in coeff_splits for n in d.values()):
        result = empty_view
        if cache is not None:
            cache[key] = result
        return result
    write_index = next(iter(rw.writes)).index
    read_index = next((d.index for d in rw.reads), write_index)
    iter_space = iteration_space_from_op(op)
    per_sym = apply_splits_from_index_coeff(
        coeff_splits, write_index, read_index, iter_space
    )

    # Step 2: keep splits that actually slice this buffer, keyed by
    # their host stride on buf via dep.index.coeff(sym). host_stride == 0
    # means the split contracts an axis not present on this buffer
    # (canonical case: a K-split's output dep) and is dropped from the
    # geometry. The has_partial_reduction flag is op-level — set whenever
    # the op has any reduction-axis split — and is independent of which
    # dep we're inspecting here.
    has_partial_reduction = any(n > 1 for n in coeff_splits[1].values())
    splits_by_stride: dict[int, tuple[int, "sympy.Symbol"]] = {}
    for sym, split in per_sym.items():
        host_stride = concretize_expr(dep.index.coeff(sym))
        if split <= 1 or host_stride == 0:
            continue
        splits_by_stride[host_stride] = (int(split), sym)

    buf_layout = V.graph.get_buffer(buf_name).layout
    if not isinstance(buf_layout, FixedTiledLayout):
        return empty_view
    dev_layout = buf_layout.device_layout
    device_size = dev_layout.device_size
    stride_map = dev_layout.stride_map
    elems_per_stick = dev_layout.device_dtype.elems_per_stick()

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

    work_slice_dims: dict[int, int] = {}
    sym_to_device_dim: dict["sympy.Symbol", int] = {}
    for h, (split, sym) in sorted(splits_by_stride.items()):
        dev_dim = device_stride_to_dim.get(h)
        if h == stick_host_stride:
            dev_dim = num_stick_dim
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
            return empty_view
        work_slice_dims[dev_dim] = split
        sym_to_device_dim[sym] = dev_dim

    # Step 4: build the core→slot mapping using the same gate codegen
    # uses (_should_use_k_fast_mapping), so K-fast matmul ops compare
    # under the K-cohort-adjacent ordering they will actually emit.
    num_cores = int(math.prod(per_sym.values()))
    is_matmul = _is_matmul_op(op)
    if _should_use_k_fast_mapping(is_matmul, iter_space, per_sym):
        _mapping_func = _k_fast_core_to_slice_mapping
    else:
        _mapping_func = _get_core_to_slice_mapping
    core_to_slot_by_name = _mapping_func(iter_space, per_sym, num_cores)
    # Re-key by the buffer's device-dim index (canonical) instead of the op's
    # iter symbol name. Two ops with the same physical per-core slicing on
    # this buffer compare equal even if they name their iter axes differently
    # (e.g. one op calls cols `d0`, another calls cols `d1`). Drop unsplit
    # dims: _get_core_to_slice_mapping emits Integer(0) for any dim with
    # split=1, which doesn't affect per-core byte placement but would make
    # two ops with different iter-space arities compare unequal.
    pruned_core_to_slot: list[tuple[int, "Expr"]] = []
    for sym, dev_dim in sym_to_device_dim.items():
        expr = core_to_slot_by_name.get(str(sym))
        if expr is not None:
            pruned_core_to_slot.append((dev_dim, expr))
    pruned_core_to_slot.sort(key=lambda x: x[0])

    view = PerCoreView(
        work_slice_dims=tuple(sorted(work_slice_dims.items())),
        core_to_slot=tuple(pruned_core_to_slot),
    )
    result = (view, has_partial_reduction)
    if cache is not None:
        cache[key] = result
    return result
