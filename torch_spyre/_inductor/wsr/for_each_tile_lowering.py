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

"""for_each_tile WhileLoop recognition, splicing, and direct loop-info stamping.

This module recognizes the exact WhileLoop shape torch-spyre#4136's
for_each_tile frontend (via decompose_scan_to_while_loop) produces, and owns
every step of turning one into tiled, schedulable ops:

1. Prove: try_prove_for_each_tile inspects a WhileLoop's cond_subgraph and
   decides whether it is for_each_tile-shaped, deriving a provable trip
   count. Every for_each_tile-specific assumption lives in this module;
   while_loop_bridge.py (the generic splice mechanism) knows none of them.

2. Splice: once accepted, splice_while_loops hands the WhileLoop to
   while_loop_bridge.splice_while_loop, which flattens the loop body into
   graph.operations in place. For a nested for_each_tile, splicing runs to a
   fixed point, outermost level first -- an inner WhileLoop only becomes
   visible once the outer splice has flattened its body in.

3. Identify the tile: each tile's real read/write is marked by a
   tile_dim_marker op (lowering.py's lower_tile_dim_marker).
   _consume_tile_dim_markers finds every marker's consuming op(s), inlines
   the marker's own per-iteration coordinate transform directly into each
   ComputedBuffer consumer (or, for a StarDep-shaped consumer with no
   inner_fn to inline into, redirects the reference and keeps the marker
   materialized), and erases the marker where possible. This produces the
   ground truth lookup_marker_dim later uses to say which read/write of an
   op is the tile and which of its own index variables is the tiled one.

4. Stamp: _stamp_direct_loop_info directly constructs and stamps a
   CoarseTileInfo on every op in a spliced group, one nesting level at a
   time, from ground truth already available at this point -- trip count
   (from the prover), loop var (from the spliced body), and tiled-dim
   position (from lookup_marker_dim, or a structural fallback anchored on
   the op's own output coordinates when no marker resolves it). This never
   calls coarse_tile_pre_stickify: re-deriving tiling from index
   coefficients independently at each nesting level risks a level's already-
   committed metadata going stale once a later-spliced, more-deeply-nested
   level renames or rewires the buffers that metadata depended on.
   splice_while_loops defers every stamp call to a single final phase, after
   every level of a nested for_each_tile has been spliced, so every op that
   will ever exist for this compile is present before any stamping happens
   -- see splice_while_loops's own docstring for why per-iteration stamping
   cannot work for a nested for_each_tile.

Real cond-graph shape (against a live compiled graph for both split_m_fn
(map mode) and split_k_fn (carry mode)): decompose_scan_to_while_loop always
lowers for_each_tile's cond_fn to a cond_subgraph.graph with exactly one
ir.Operation -- a scalar (size=[]) bool ComputedBuffer -- whose inner_fn
does exactly:

    tmp0 = ops.load(<cond graph's own first placeholder>, 0)
    tmp1 = ops.constant(N, torch.int64)
    tmp2 = tmp0 < tmp1
    return tmp2

i.e. `lt(iteration_sym, N)` with N a plain Python int/constant baked in by
the tracer (for_each_tile.py's `_step_counter`/`count_mode` logic always
carries the trip counter as carried_inputs[0], and the cond subgraph's own
first placeholder is that same carry positionally). `N` is not exposed as a
separate symbolic node anywhere reachable from the IR level -- it only shows
up as the literal second operand of the `<` -- so rather than parse
inner_fn's closure cells (an internal, unstable implementation detail of
torch._inductor.ir.make_pointwise/ops_wrapper), this module *runs* inner_fn
once under a small recording ops handler that intercepts `load`/`constant`
and returns opaque placeholders for everything else. This is the same "wrap
the ops handler, don't reconstruct index expressions" pattern CLAUDE.md
mandates for ComputedBuffer.inner_fn elsewhere in this codebase (see issue
#2797), applied here for read-only shape recognition rather than mutation.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import logging
from typing import TYPE_CHECKING, Any

import sympy
import torch

from torch._inductor.ops_handler import DefaultHandler, WrapperHandler
from torch._inductor.virtualized import V

from .. import timing_recorder
from ..deadcode_elimination import deadcode_elimination
from ..logging_utils import get_inductor_logger
from ..pass_utils import format_operations

if TYPE_CHECKING:
    from torch._inductor import ir
    from torch._inductor.dependencies import Dep

logger = get_inductor_logger("wsr.for_each_tile_lowering")


@dataclasses.dataclass(frozen=True)
class ProverResult:
    """Outcome of trying to recognize a WhileLoop as a for_each_tile loop."""

    accepted: bool
    trip_count: sympy.Expr | None = None
    reason: str = ""


class _CondInnerFnRecorder(DefaultHandler):
    """Records the loads/constants/comparison op a cond inner_fn issues.

    Every other ops call (there should be none for the shape this prover
    recognizes) is routed through `_default` and answered with an opaque
    placeholder string so `inner_fn` can run to completion without needing a
    real kernel-codegen context.
    """

    def __init__(self) -> None:
        self.loads: list[tuple[str, Any]] = []
        self.constants: list[Any] = []
        self.compare_ops: list[str] = []

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if name == "load":
            self.loads.append((args[0], args[1]))
            return f"__load_{len(self.loads) - 1}__"
        if name == "constant":
            self.constants.append(args[0])
            return f"__constant_{len(self.constants) - 1}__"
        if name in ("lt", "le", "gt", "ge", "eq", "ne"):
            self.compare_ops.append(name)
            return f"__cmp_{name}__"
        # Anything else means this cond graph does not match the known
        # for_each_tile shape (a single load-vs-constant comparison); record
        # the op name so the caller can decline with a useful reason.
        self.compare_ops.append(f"unexpected:{name}")
        return f"__unexpected_{name}__"


class _IdentityLoadRecorder(DefaultHandler):
    """Recognize a Pointwise body that returns exactly one load.

    ``WhileLoop.create`` uses such bodies to repair an input's strides before
    handing it to the loop body.  After a for_each_tile WhileLoop is spliced,
    that otherwise-benign whole-input materialization sits inside the counted
    loop.  This recorder lets the post-splice contraction below prove the copy
    is an identity without inspecting ``inner_fn`` closures.
    """

    def __init__(self) -> None:
        self.value = object()
        self.loads: list[tuple[str, Any]] = []

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if name != "load" or self.loads:
            raise ValueError("not a single-load identity")
        self.loads.append((args[0], args[1]))
        return self.value


def _first_placeholder_name(cond_graph) -> str | None:
    """The cond subgraph's own first graph input -- the iteration carry."""
    graph_inputs = getattr(cond_graph, "graph_inputs", None)
    if not graph_inputs:
        return None
    return next(iter(graph_inputs), None)


def _extract_trip_count(cond_graph) -> sympy.Expr | None:
    """Find the single lt(iteration_sym, N)-shaped comparison cond_graph computes.

    for_each_tile's cond_fn (after decompose_scan_to_while_loop) reduces to
    exactly one boolean scalar ComputedBuffer computing
    `ops.load(<first placeholder>, 0) < ops.constant(N, ...)`. Returns N as a
    sympy.Expr, or None if the shape does not match.
    """
    graph_outputs = getattr(cond_graph, "graph_outputs", None)
    if not graph_outputs or len(graph_outputs) != 1:
        return None
    operations = getattr(cond_graph, "operations", None)
    if not operations or len(operations) != 1:
        return None

    op = operations[0]
    data = getattr(op, "data", None)
    inner_fn = getattr(data, "inner_fn", None)
    if inner_fn is None:
        return None
    # The comparison is a scalar bool -- no output ranges to index over.
    get_size = getattr(data, "get_size", None)
    if get_size is None or list(get_size()) != []:
        return None
    if getattr(data, "dtype", None) != torch.bool:
        return None

    first_placeholder = _first_placeholder_name(cond_graph)
    if first_placeholder is None:
        return None

    recorder = _CondInnerFnRecorder()
    with V.set_ops_handler(recorder):
        inner_fn(())

    if recorder.compare_ops != ["lt"]:
        return None
    if len(recorder.loads) != 1 or len(recorder.constants) != 1:
        return None

    (loaded_name, loaded_index) = recorder.loads[0]
    if loaded_name != first_placeholder:
        return None
    if loaded_index != 0:
        return None

    bound = recorder.constants[0]
    if isinstance(bound, bool):
        return None
    if not isinstance(bound, (int, sympy.Expr)):
        return None
    return sympy.sympify(bound)


def try_prove_for_each_tile(while_op: "ir.WhileLoop") -> ProverResult:
    """Decide whether while_op matches for_each_tile's known WhileLoop shape."""
    cond_subgraph = getattr(while_op, "cond_subgraph", None)
    cond_graph = getattr(cond_subgraph, "graph", None) if cond_subgraph else None
    if cond_graph is None:
        return ProverResult(accepted=False, reason="no cond_subgraph.graph to inspect")

    trip_count = _extract_trip_count(cond_graph)
    if trip_count is None:
        return ProverResult(
            accepted=False,
            reason=(
                "cond_subgraph did not reduce to a single provable "
                "lt(iteration_sym, N) comparison"
            ),
        )
    return ProverResult(accepted=True, trip_count=trip_count)


def _body_loop_var(while_op: "ir.WhileLoop") -> sympy.Symbol | None:
    """Find the real per-iteration index symbol the spliced body already uses.

    for_each_tile's frontend always carries the trip counter as
    carried_inputs[0] (see for_each_tile.py's _step_counter/count_mode
    logic), so the body subgraph's own first graph input is that same carry
    positionally. decompose_scan_to_while_loop's body lowers each tile's
    scan-index arithmetic to a leading DynamicScalar op that reads that
    first placeholder (via ops.load/.item()) and defines a fresh unbacked
    symbol (e.g. ``u0``); every tiled op's real index expressions
    (ExternKernelOut offsets, ComputedBuffer write indices, ...) are then
    written in terms of that symbol -- confirmed against a live compiled
    graph for both split_m_fn (map mode) and split_k_fn (carry mode).

    _stamp_direct_loop_info's loop_var parameter must be exactly this
    symbol: lookup_marker_dim/coarse_tile.py's op_out_coords/
    reduction_loop_vars resolve loop_var by searching for it inside an
    op's own index expressions, so a freshly-minted, disconnected
    sympy.Symbol would never resolve and every op would land with empty
    tiled dims.

    Returns None if the body subgraph does not have this exact shape (no
    DynamicScalar reading the first placeholder), signaling the caller to
    decline rather than stamp loop_info nothing will ever resolve against.
    """
    from torch._inductor import ir

    body_subgraph = getattr(while_op, "body_subgraph", None)
    body_graph = getattr(body_subgraph, "graph", None) if body_subgraph else None
    if body_graph is None:
        return None

    graph_inputs = getattr(body_graph, "graph_inputs", None)
    if not graph_inputs:
        return None
    first_placeholder = next(iter(graph_inputs), None)
    if first_placeholder is None:
        return None

    for op in getattr(body_graph, "operations", None) or ():
        if not isinstance(op, ir.DynamicScalar):
            continue
        defs = op.get_unbacked_symbol_defs()
        if len(defs) != 1:
            continue
        inputs = getattr(op, "inputs", None) or []
        input_names = [i.get_name() for i in inputs if hasattr(i, "get_name")]
        if input_names == [first_placeholder]:
            return next(iter(defs))
    return None


_MARKER_MAPS: dict[int, dict[tuple[str, "Dep"], int]] = {}
"""Per-compile marker-map registry, keyed by id(operations).

NOT safe to let outlive one compile: CPython aggressively reuses a freed
list's id, so a stale entry left behind by a prior compile can collide
with -- and be silently mistaken for -- a live compile's own entry sharing
the same buffer-name/dep-shape key. clear_marker_maps() must be called once
per compile, before _consume_tile_dim_markers runs, to guarantee this never
happens -- passes.py's per-compile pipeline entry point does this, alongside
the analogous reset_provenance_warnings() call, for the same "each compile
starts from a clean slate" reason.
"""


def clear_marker_maps() -> None:
    """Discard every entry in the module-level _MARKER_MAPS registry.

    Must be called exactly once per compile, before _consume_tile_dim_markers
    runs for that compile (passes.py's per-compile pipeline __call__ does
    this). See _MARKER_MAPS's own docstring for why skipping this is unsafe,
    not just leaky: id(operations) can be reused by CPython for an unrelated
    later compile's operations list, letting that later compile silently
    resolve against a dead compile's stale entry instead of raising or
    returning None.
    """
    _MARKER_MAPS.clear()


def _stacking_carry_indices(
    while_op: "ir.WhileLoop",
    loop_var: sympy.Symbol,
    trip_count: "sympy.Expr | int | None" = None,
) -> frozenset[int]:
    """Which carry positions are ``scan``-``ys`` stacking carries, not accumulators.

    ``for_each_tile``'s map mode has no user carry at all: ``scan`` requires
    one, so the frontend threads a step counter as the carry and puts the
    per-tile output in ``ys`` (see for_each_tile.py's ``map_mode`` branch and
    ``_stacked_to_full``). ``decompose_scan_to_while_loop`` then materializes
    that ``ys`` accumulation as ANOTHER ``carried_inputs`` entry, so at the
    ``ir.WhileLoop`` level it is positionally indistinguishable from a real
    accumulator carry -- yet it needs the opposite treatment (see
    while_loop_bridge.py's ``CarryBinding.stacking``).

    The distinguishing evidence, taken from the IR rather than from the
    frontend's own metadata (which does not survive to this point):

    1. The body does not compute a new value for this carry position -- its
       ``body_output`` IS the body's own placeholder for it, threaded
       through unchanged. A real accumulator's ``body_output`` is a
       different, op-produced buffer (``split_k_fn``: ``buf6``, its own
       ``acc + x @ y`` result).
    2. Some body op nonetheless WRITES it, in place, through a
       ``MutationLayoutSHOULDREMOVE`` whose target is a view of that
       placeholder -- so the carry is not merely a read-only pass-through
       leaf (``split_m_fn``'s X ``xs`` leaf is exactly that, and must NOT be
       folded).
    3. That write's per-iteration position depends on ``loop_var``: the
       target view's own offset mentions it. This is what makes it a stack
       of tiles rather than one whole-buffer overwrite, and it is the fact
       the fold arithmetic relies on. EXCEPT when ``trip_count == 1``: a
       single-trip loop's write offset has only one possible value, so
       Inductor's own symbolic simplification legitimately drops
       ``loop_var`` from it entirely, leaving offset=0 rather than an
       expression mentioning ``loop_var``. Treat that case as
       satisfying evidence 3 too, rather than falling through to the
       pass-through-leaf classification -- ``fold_stacked_carry_layout``
       already documents and handles this same trip_count=1 degeneracy on
       the fold-arithmetic side (its ``size[1] == 1`` branch), so detection
       must recognize the same shape or the fold never runs at all.

    Requiring all three keeps every other carry shape -- accumulator,
    read-only pass-through leaf, scalar step counter -- on the pre-existing
    path untouched.
    """
    from torch._inductor import ir
    from torch._inductor.ir import MutableBox

    body_graph = while_op.body_subgraph.graph
    placeholder_names = list(body_graph.graph_inputs.keys())
    body_outputs = body_graph.graph_outputs

    single_trip = (
        trip_count is not None and sympy.simplify(sympy.sympify(trip_count) - 1) == 0
    )

    # Placeholders written in place, per iteration, at a loop_var-dependent
    # offset (evidence 2 + 3).
    tile_written: set[str] = set()
    for op in body_graph.operations:
        layout = getattr(op, "layout", None)
        if not isinstance(layout, ir.MutationLayoutSHOULDREMOVE):
            continue
        target = layout.target
        while isinstance(target, MutableBox):
            target = target.data
        target_layout = getattr(target, "layout", None)
        if target_layout is None:
            continue
        offset = sympy.sympify(getattr(target_layout, "offset", 0))
        if loop_var not in offset.free_symbols and not single_trip:
            continue
        name = getattr(layout.get_buffer(), "get_name", lambda: None)()
        if name is not None:
            tile_written.add(name)

    stacking: set[int] = set()
    for i, placeholder_name in enumerate(placeholder_names):
        if i >= len(body_outputs):
            break
        out_name = getattr(body_outputs[i], "get_name", lambda: None)()
        if out_name != placeholder_name:  # evidence 1
            continue
        if placeholder_name in tile_written:
            stacking.add(i)
    return frozenset(stacking)


def _marker_dim(op: "ir.Operation") -> "int | None":
    """Return op's tile_marker_dim if it carries one, else None.

    lower_tile_dim_marker (lowering.py) stamps ``tile_marker_dim`` directly
    on the realized ``ComputedBuffer`` -- ``pw.data.data`` there, where
    ``pw`` is the ``TensorBox`` it returns, ``pw.data`` its ``StorageBox``,
    and ``pw.data.data`` the ``ComputedBuffer`` that ends up as this exact
    ``op`` object in ``graph.operations``/``group_ops``. So the attribute
    lives on ``op`` itself, not on ``op.data`` (``op.data`` is one level
    deeper still -- the ``Pointwise``/``Reduction`` IR expression node, which
    never carries it).
    """
    return getattr(op, "tile_marker_dim", None)


class MarkerResolution(enum.Enum):
    """How _consume_tile_dim_markers resolved one tile_dim_marker op.

    INLINE_ERASED: every one of the marker's consumers is a ComputedBuffer
    (or was inlined into one) -- the marker's transform was fused directly
    into each such consumer's inner_fn and the marker op was removed from
    both operations and group_ops.

    STAR_DEP_KEPT: at least one of the marker's consumers reaches it via a
    StarDep, not an ordinary MemoryDep -- the marker cannot be fused away
    and stays live as a real, addressable buffer (see
    _consume_tile_dim_markers's own comment on why removing it from
    operations would break coarse_tile.py's _validate_contiguous
    gapless-block invariant), even if other consumers of the same marker
    were inlined.
    """

    INLINE_ERASED = "inline_erased"
    STAR_DEP_KEPT = "star_dep_kept"


def _marker_resolution(op: "ir.Operation") -> "MarkerResolution | None":
    """Return op's tile_marker_resolution if it carries one, else None.

    Stamped by _consume_tile_dim_markers at the same branch point that
    decides whether to erase the marker from operations (INLINE_ERASED) or
    keep it materialized (STAR_DEP_KEPT). Like tile_marker_dim, this lives
    as a plain attribute on the op itself, not on op.data.
    """
    return getattr(op, "tile_marker_resolution", None)


def _marker_axis_is_consumed(axis_coords: "list[sympy.Expr] | None") -> bool:
    """True iff the marker's tiled-axis load-site coordinate is consumed at the consumer.

    True iff the captured coordinate(s) are non-empty and carry no free symbols at
    all (all pure constants) -- stricter than "no iteration variable", so a
    dynamic-size symbol conservatively keeps the existing resolution path. A
    surviving tiled coordinate always carries a free symbol (d0/i0/q0/...), so it is
    never misclassified. A kept size-1 tiled axis is classified consumed: harmless,
    it cannot be split and the per-trip advance still comes from
    _structural_resolve / squeezed_advance_per_read.
    """
    if not axis_coords:
        return False
    return all(not c.free_symbols for c in axis_coords)


def _delinearize_index(index: sympy.Expr, size, stride, offset) -> list[sympy.Expr]:
    """Invert a FixedLayout-style flat index back into per-dim coordinates.

    ``ops.load(name, index)`` always receives ``index`` already flattened
    to a single sympy expression by the loaded buffer's own
    ``make_indexer()`` -- see ``Buffer.make_loader``/``_fixed_indexer`` --
    never the original ``Sequence[Expr]`` coordinate list a ``Loops``
    ``inner_fn``'s own index is parameterized by. Inlining the marker's
    own read (see ``_InlineMarkerHandler``) needs that coordinate list
    back, so undo ``_fixed_indexer``'s ``sum(idx[i] * stride[i]) +
    offset`` here: every per-dim loop-index symbol (``i0``, ``r0``, ``d0``,
    ...) appears in exactly one additive term of that sum with
    coefficient ``stride[i]`` (the same one-symbol-per-term linearity
    every coordinate-recovery helper in this package already assumes --
    see coarse_tile.py's ``_loop_var_to_ranges_pos``/the deleted
    ``_loop_var_pos_from_reads``), so dividing each matched term by its
    own stride recovers ``idx[i]`` directly; a size-1 dim contributes no
    term at all (``_fixed_indexer`` skips it), so its coordinate is
    simply 0.

    Two distinct dims sharing the same non-zero stride literal (a
    degenerate/unusual layout, not observed against any fixture in this
    repo, but not structurally impossible either) would make
    ``remaining.coeff(st)`` silently SUM both dims' coefficients into one
    merged coordinate instead of raising -- exactly the kind of silent
    wrong-answer this module exists to prevent elsewhere. Guard against it
    explicitly: raise rather than let two dims collide on one recovered
    coordinate.
    """
    remaining = sympy.expand(index - offset)
    coords: list[sympy.Expr] = []
    seen_strides: dict[sympy.Expr, int] = {}
    for i, (sz, st) in enumerate(zip(size, stride)):
        if sz == 1:
            coords.append(sympy.Integer(0))
            continue
        if st != 0 and st in seen_strides:
            raise AssertionError(
                f"_delinearize_index: dims {seen_strides[st]} and {i} both "
                f"have stride {st!r} (sizes {size!r}); cannot recover "
                "distinct coordinates for both from a single flattened "
                "index without conflating them."
            )
        if st != 0:
            seen_strides[st] = i
        # A size>1 broadcast dim (stride 0) contributes no term to
        # `remaining` and its coordinate is always 0 -- but that must be
        # special-cased rather than falling into the general
        # `remaining.coeff(st)` branch below: sympy's `.coeff(0)` does not
        # mean "coefficient of the constant term" here, it returns the
        # WHOLE `remaining` expression unchanged. Using `remaining.coeff(st)`
        # unconditionally would silently smuggle the entire remaining sum
        # into this one dim's "coordinate" instead of 0.
        coords.append(remaining.coeff(st) if st != 0 else sympy.Integer(0))
    return coords


def _marker_substitution(
    marker_op: "ir.Operation",
) -> tuple[str, sympy.Expr, tuple[sympy.Symbol, ...], tuple[sympy.Expr, ...]]:
    """Get (marker's own input name, marker's own read index expr, var_names, size).

    The marker's ``inner_fn`` (``lower_tile_dim_marker``, lowering.py) is
    always exactly ``return loader(index)`` -- a single load of the
    marker's own upstream input at some index that is generally NOT the
    identity (it can carry an extra per-iteration advance term, e.g.
    ``+ 24*u0``). Rather than re-executing that ``inner_fn`` live (which
    would replay a stale FX ``Proxy``/``OpsValue`` captured from whatever
    trace built the marker in the first place -- crashing
    (``LightTracer.create_arg`` raises ``NotImplementedError`` on the
    leaked ``OpsValue``) or silently reusing the wrong graph node when
    spliced into a different consumer's live retrace), extract the
    marker's own read as a pure symbolic expression via
    ``get_read_writes()`` and let the caller substitute into it -- the same
    "index expressions are symbolic, substitute don't re-execute"
    discipline this module and ``coarse_tile.py`` already follow
    everywhere else (see CLAUDE.md's "wrap, never reconstruct").

    Returns the marker's own single MemoryDep read's ``(name, index)``,
    where ``index`` is expressed in terms of the marker's own WRITE dep's
    ``var_names`` (its own output-coordinate symbols, ``d0, d1, ...`` in
    positional order) -- the caller substitutes its own load-site
    coordinates for those symbols positionally. Also returns that same
    WRITE dep's own ``size``, positionally aligned with ``var_names``: both
    have size-1 dims already squeezed out by Inductor's
    ``index_vars_squeeze``/canonicalize machinery, so the caller can use
    ``size`` to tell which of the marker's *layout* dims (``_InlineMarkerHandler``'s
    own ``self._size``, which does NOT have size-1 dims squeezed out) a
    given ``var_names`` entry actually corresponds to.
    """
    from torch._inductor.dependencies import MemoryDep

    rw = marker_op.get_read_writes()
    write_deps = [d for d in rw.writes if isinstance(d, MemoryDep)]
    read_deps = [d for d in rw.reads if isinstance(d, MemoryDep)]
    if len(write_deps) != 1 or len(read_deps) != 1:
        raise AssertionError(
            f"tile_dim_marker op {marker_op.get_name()!r} has "
            f"{len(write_deps)} MemoryDep writes and {len(read_deps)} "
            "MemoryDep reads; expected exactly 1 of each to inline its "
            "body into a consumer."
        )
    write_dep, read_dep = write_deps[0], read_deps[0]
    return read_dep.name, read_dep.index, write_dep.var_names, write_dep.size


class _InlineMarkerHandler(WrapperHandler):
    """Intercept a load of one erased tile_dim_marker, inlining its own body.

    A plain name-swap (``pass_utils.NameSwapHandler``, via
    ``redirect_computed_buffer_reads``) is wrong here: it rewrites
    ``load(marker_name, index)`` to ``load(marker_input_name, index)``,
    reusing the CONSUMER's own (already-flattened) index expression
    unchanged. That index was computed by flattening the CONSUMER's
    coordinate list through the MARKER's OWN layout indexer (e.g. index
    ``12*d0 + d2`` into the marker's own ``[2, 12]``-shaped write) -- it is
    not, and must not be reused as, an index into the marker's raw,
    unsliced INPUT (e.g. ``arg0_1``, the whole per-trip-invariant operand,
    whose corresponding read is ``12*d0 + d1 + 24*u0`` -- note the extra
    ``+ 24*u0`` term the marker's own body contributes, encoding exactly the
    per-iteration tile offset _hint_ranges_pos/lookup_marker_dim need to
    see). Swapping only the name and keeping the consumer's own flat index
    silently drops that offset term entirely -- every trip reads the SAME
    window of the underlying tensor instead of advancing, a silent
    wrong-answer bug rather than a crash.

    The correct erasure substitutes the CONSUMER's own load-site index
    (delinearized back into per-dim coordinates by ``_delinearize_index``,
    since ``ops.load`` always hands us an already-flattened single
    expression, not the coordinate list the marker's own index is
    parameterized by) for the marker's own output-coordinate symbols
    inside the marker's own read-index expression (``_marker_substitution``)
    -- a pure sympy substitution, never a live re-execution of the
    marker's ``inner_fn`` (see ``_marker_substitution``'s docstring for why
    that would be wrong). This keeps the "wrap, never reconstruct"
    convention (CLAUDE.md, issue #2797): the marker's ORIGINAL index
    expression is reused verbatim, never re-derived by hand -- only the
    free variables are substituted, exactly as any other index-composition
    in this codebase already does.
    """

    def __init__(
        self, inner, marker_name: str, marker_op: "ir.Operation", capture=None
    ):
        super().__init__(inner)
        self._marker_name = marker_name
        # Marker operand's tiled axis + a one-shot collector for the load-site
        # coordinate along it (feeds the consumed-axis classification).
        self._marker_dim = _marker_dim(marker_op)
        self._capture = capture
        layout = marker_op.layout
        self._size = layout.size
        self._stride = layout.stride
        self._offset = layout.offset
        (
            self._marker_input_name,
            self._marker_read_index,
            self._marker_var_names,
            self._marker_write_size,
        ) = _marker_substitution(marker_op)

    def load(self, name, index):
        if name == self._marker_name:
            coords = _delinearize_index(index, self._size, self._stride, self._offset)
            if (
                self._capture is not None
                and self._marker_dim is not None
                and 0 <= self._marker_dim < len(coords)
            ):
                # Key by the consumer load-site index and REPLACE, so a
                # re-evaluated inner_fn yields one record per distinct read rather
                # than accumulating. The coordinate is in the marker's own
                # unsqueezed axes: a pure constant means the axis was consumed
                # (sliced base); a free symbol means genuine tiling.
                self._capture[index] = coords[self._marker_dim]
            # self._size (this handler's own layout.size) has NOT had size-1
            # dims squeezed out, but self._marker_var_names/
            # self._marker_write_size (from the marker's own WRITE dep) HAVE
            # -- Inductor's index_vars_squeeze/canonicalize already dropped
            # them. Zipping coords (one per self._size slot) directly against
            # var_names (one per squeezed slot) would silently misalign and
            # truncate the moment the two lists differ in length -- e.g. a
            # size-1 dim anywhere but the position(s) every current fixture
            # happens to put it at. Filter coords down to only the positions
            # whose size is not 1 before zipping, so the two lists are
            # positionally comparable by construction rather than by
            # coincidence of today's fixture shapes.
            non_unit_coords = [c for c, sz in zip(coords, self._size) if sz != 1]
            if len(non_unit_coords) != len(self._marker_var_names):
                raise AssertionError(
                    f"tile_dim_marker {self._marker_name!r}: "
                    f"{len(non_unit_coords)} non-size-1 load-site coordinates "
                    f"(from size {self._size!r}) but "
                    f"{len(self._marker_var_names)} marker var_names (from "
                    f"write size {self._marker_write_size!r}); cannot "
                    "substitute positionally."
                )
            subs = dict(zip(self._marker_var_names, non_unit_coords))
            # simultaneous=True is required: sympy.Expr.subs(dict) otherwise
            # applies substitutions sequentially, one symbol at a time, so a
            # dict like {d0: d1, d1: d2} first rewrites d0->d1 and THEN
            # rewrites that same fresh d1 -> d2, silently merging two
            # distinct coordinates into one wrong composed index (e.g.
            # {d0: d1, d1: d2} applied sequentially to `d0 + 2*d1` yields
            # `3*d1` instead of the correct `d1 + 2*d2`) whenever the
            # substitution's target set overlaps its source set, which a
            # coordinate permutation (e.g. a transposed tile read) does.
            composed = self._marker_read_index.subs(subs, simultaneous=True)
            return super().load(self._marker_input_name, composed)
        return super().load(name, index)


def _inline_marker_into_consumer(
    consumer_op: "ir.Operation",
    marker_op: "ir.Operation",
    operations: list["ir.Operation"],
) -> "tuple[ir.Operation, bool]":
    """Erase marker_op by inlining its body into consumer_op's load of it.

    See _InlineMarkerHandler for why a plain name-swap
    (pass_utils.redirect_computed_buffer_reads) is wrong for this case.
    Patches consumer_op.data.inner_fn in place (Loops is a frozen dataclass,
    hence object.__setattr__ -- same as redirect_computed_buffer_reads does),
    then delegates the reconstruct-and-swap-into-`operations` step to
    ``pass_utils.replace_computed_buffer_body``, which already performs
    exactly that (metadata copy, provenance, cache invalidation, mutation-
    target/nested-WhileLoop repointing) for a caller supplying a full new
    body object rather than a bare name map.

    Returns ``(new_consumer, consumed)``: the replacement buffer and whether the
    marker's tiled axis was consumed at the consumer. The new read/write info is
    materialized once here to fill the one-shot collector, which is then closed
    (the closed-over ``capture`` is rebound to None) so later re-evaluations of
    this long-lived inner_fn stop recording.
    """
    from torch._inductor.virtualized import V

    from torch_spyre._inductor.pass_utils import (
        _invalidate_body_caches,
        replace_computed_buffer_body,
    )

    marker_name = marker_op.get_name()

    orig_inner = consumer_op.data.inner_fn
    capture: "dict | None" = {}

    def new_inner_fn(*args, _orig_inner=orig_inner):
        with V.set_ops_handler(
            _InlineMarkerHandler(V.ops, marker_name, marker_op, capture)
        ):
            return _orig_inner(*args)

    object.__setattr__(consumer_op.data, "inner_fn", new_inner_fn)
    _invalidate_body_caches(consumer_op.data)

    result = replace_computed_buffer_body(
        consumer_op,
        consumer_op.data,
        operations,
        pass_name="_consume_tile_dim_markers",
        reason=f"inline erased tile_dim_marker {marker_name!r} body",
    )
    # Fill the one-shot collector once (this evaluates the replacement body),
    # classify, then close it by rebinding the closed-over `capture` to None:
    # later re-evaluations of this long-lived inner_fn pass None to the handler.
    result.get_read_writes()
    consumed = _marker_axis_is_consumed(
        list(capture.values()) if capture is not None else []
    )
    capture = None
    return result, consumed


def _consume_tile_dim_markers(
    group_ops: list["ir.Operation"],
    operations: list["ir.Operation"],
) -> dict[tuple[str, "Dep"], int]:
    """Find every tile_dim_marker-tagged op in group_ops, erase it, map its dim.

    For each marker op (an op whose realized ComputedBuffer carries
    tile_marker_dim -- see lowering.py's lower_tile_dim_marker): find every
    consuming use among group_ops, record (op.get_name(), dep) -> dim in
    the returned map for each one, then erase the marker and remove the
    marker op from `operations`.

    A marker can have more than one consuming use: e.g. torch.softmax's
    default decomposition (amax, sub, exp, sum, div) reads its input tensor
    directly from two sibling ops (amax and sub both read the
    pre-decomposition value, per torch._inductor.decomposition's aten
    softmax decomp), so a for_each_tile tile whose body calls softmax on
    the whole tile produces a marker read by two independent ComputedBuffer
    consumers rather than one op chained through another. Each consuming
    read is resolved independently -- inlining the
    marker's transform into each ComputedBuffer consumer in turn, or
    redirecting each StarDep-shaped consumer's reference -- and the marker
    itself is erased only once every consuming read has been resolved (see
    below on why any StarDep consumer forces STAR_DEP_KEPT for the whole
    marker, even when other consumers were inlined).

    A marker's consumer can hold its read in either of two shapes (both
    already handled elsewhere in this package for the analogous WAR-hazard
    carry-snapshot redirect -- see while_loop_bridge.py's
    ``_snapshot_carry_placeholder``, whose ``hasattr(reader, "data")``
    branch is the same test used below):

    - A ``ComputedBuffer`` consumer (has ``.data``, an inner_fn-backed
      Pointwise/Reduction/Scan/Sort body): the read surfaces as a
      ``MemoryDep`` named after the marker in ``get_read_writes().reads``.
      Erased via ``_inline_marker_into_consumer``, which wraps (never
      reconstructs, per CLAUDE.md) the inner_fn with ``_InlineMarkerHandler``
      so every load of the marker re-issues the MARKER'S OWN inner_fn
      (``marker_op.data.make_loader()``) at the consumer's load-site index,
      rather than merely renaming past it. A plain name-swap
      (``pass_utils.redirect_computed_buffer_reads``/``NameSwapHandler``) is
      NOT used here: the marker's own body performs a genuine, non-identity
      per-iteration coordinate transform (the tile's slice offset -- e.g. an
      extra ``+ 24*u0`` advance term baked into the marker's own read index
      by ``lower_tile_dim_marker``), which a bare name-swap would silently
      drop, since ``NameSwapHandler.load`` passes the consumer's own index
      straight through unchanged. A plain rename produces silently wrong
      device-side numerics: every loop trip ends up reading the identical
      (tile-0-only) slice of the underlying tensor instead of advancing
      through it.
    - An ``InputsKernel``-family consumer (``ExternKernelOut``,
      ``FallbackKernel``, ``ConcatKernel``, ... -- no inner_fn, e.g. the CPU
      aten-fallback matmul for a for_each_tile tile read on a
      device-less/CPU fixture): the read surfaces as a ``StarDep`` (name
      only, no index/ranges) named after the marker, and is held as a
      direct Python object reference in the consumer's ``.inputs`` list (or
      ``.layout.target`` for a MutationLayoutSHOULDREMOVE write) rather than
      through any named load. Erased via
      ``while_loop_bridge._substitute_direct_input_refs``, which patches
      that reference in place to point at the marker's own input object
      instead -- the same helper (and the same object-identity-preserving
      unwrap-one-StorageBox-level care it documents) that
      ``_snapshot_carry_placeholder`` already relies on for this exact read
      shape.

    Zero consuming uses (of either shape) is an unrecognized shape and
    raises -- this pass runs on a freshly spliced body whose only consumers
    of a marker's output should be the op(s) for_each_tile's frontend wrote
    to read that tile, so no consumer at all means an assumption this
    module owns (see the module docstring) no longer holds and a silent
    guess would be worse than a loud failure. More than one StarDep-shaped
    consuming use also raises (see the dedicated check above) -- that shape
    (sibling for_each_tile loops sharing one outer tile, issue #4581) is not
    yet handled correctly, unlike multiple ComputedBuffer consuming uses.

    Must run before any hint synthesis -- callers of _hint_ranges_pos
    consult the module-level _MARKER_MAPS registry this function populates,
    and must see every marker already resolved and erased.

    Internally keyed by ``consumer_op.get_name()`` rather than
    ``consumer_op`` itself or ``id(consumer_op)``: ``ir.Operation``/
    ``ComputedBuffer`` are ``(unsafe_hash=False, eq=True)`` dataclasses, so
    Python sets their ``__hash__`` to ``None`` and they cannot be dict/set
    keys directly (see ``coarse_tile.py``'s ``plan: dict[int,
    CoarseTileInfo]`` docstring for the same, already-established
    convention in this codebase for using ``id()`` instead of the object).
    But ``id(op)`` itself is NOT safe here the way it is for that other
    convention's own single-pass, single-object lifetime: a *later*
    coarse-tiling sub-pass (e.g. ``_insert_all_read_copy_ops`` /
    ``_patch_consumer_to_read_copy``) can rebuild this exact consumer
    AGAIN via ``replace_computed_buffer_body`` -- for a completely
    unrelated read of the same op -- minting a new object with a new
    ``id()`` before ``lookup_marker_dim`` is ever called from
    ``_hint_ranges_pos`` -- silently orphaning an otherwise-still-matching
    map entry keyed by the old ``id()``, even when the rebuild was for a
    completely unrelated read of the same op. ``op.get_name()`` (the
    buffer name) is what stays
    stable across such a reconstruction -- every rebuild-via-
    ``replace_computed_buffer_body`` site in this package (this one
    included) preserves the original name, and other Spyre metadata
    (``PropagationPlan.outside_consumer_names``, etc.) already keys by name
    for exactly this reason -- see that field's own docstring on name
    stability. The returned map is keyed by ``(op.get_name(), dep)`` --
    ``lookup_marker_dim`` (the map's only reader) recomputes ``op.get_name()``
    itself, so callers never need the key's ``str`` half spelled out
    explicitly. For a StarDep-shaped consumer the dep stored is the StarDep
    itself (not a MemoryDep) --
    lookup_marker_dim's own read-walk already iterates every read
    regardless of type when matching by identity/equality against the map,
    and only special-cases MemoryDep for the (inapplicable to StarDep,
    which has no index) reduction-coordinate check.
    """
    from torch._inductor.dependencies import MemoryDep, StarDep

    from torch_spyre._inductor.wsr.while_loop_bridge import (
        _substitute_direct_input_refs,
    )

    marker_map: dict[tuple[str, Dep], int] = {}
    group_op_ids = {id(op) for op in group_ops}

    for marker_op in list(group_ops):
        dim = _marker_dim(marker_op)
        if dim is None:
            continue
        marker_name = marker_op.get_name()

        consumers: list[tuple[ir.Operation, Dep]] = []
        for candidate in group_ops:
            # No `id(candidate) not in group_op_ids` check here: every
            # candidate iterated is, by construction, an element of
            # group_ops itself, so it is trivially always present in
            # group_op_ids (which this loop never mutates) -- that
            # disjunct could never be True and would only mislead a
            # future reader into thinking group_ops/group_op_ids can
            # desync mid-loop here. (group_op_ids IS mutated later in
            # this function, once a marker/consumer is actually erased
            # or replaced -- see below -- just not during this scan.)
            if candidate is marker_op:
                continue
            rw = candidate.get_read_writes()
            for dep in rw.reads:
                if isinstance(dep, (MemoryDep, StarDep)) and dep.name == marker_name:
                    consumers.append((candidate, dep))

        if len(consumers) == 0:
            raise AssertionError(
                f"tile_dim_marker op {marker_name!r} has 0 consuming reads "
                "within its spliced body; expected at least one. This is "
                "an unrecognized for_each_tile shape -- "
                "_consume_tile_dim_markers only knows how to erase a marker "
                "whose output is read by one or more downstream ops."
            )

        # More than one StarDep-shaped consumer of the SAME marker is a
        # distinct, still-unhandled shape from the multi-ComputedBuffer case
        # above (e.g. torch.softmax's amax/sub siblings): it arises when two
        # independent sibling for_each_tile loops (each lowering to its own
        # ir.WhileLoop) are both handed the same outer tile directly -- see
        # sibling_nested_fn/sibling_nested_stardep_fn in
        # for_each_tile_fixtures.py (issue #4581). Redirecting each StarDep
        # consumer's reference to marker_op independently (the StarDep branch
        # below) is only proven correct for a single such consumer; the
        # two-sibling-WhileLoop shape produces silently wrong numerics, not a
        # crash, if this guard is bypassed -- so this must keep raising
        # rather than silently accept a shape the resolution logic doesn't
        # actually handle correctly. Only count StarDep consumers here; multiple
        # ComputedBuffer consumers remain supported by the loop below.
        star_dep_consumer_count = sum(
            1 for consumer_op, _ in consumers if not hasattr(consumer_op, "data")
        )
        if star_dep_consumer_count > 1:
            raise AssertionError(
                f"tile_dim_marker op {marker_name!r} has "
                f"{star_dep_consumer_count} StarDep-shaped consuming reads "
                "within its spliced body; expected at most one. Multiple "
                "sibling for_each_tile loops consuming the same outer tile "
                "directly (issue #4581) are not yet supported -- see this "
                "function's docstring."
            )

        # ComputedBuffer has no plain `.inputs` list attribute (that
        # attribute belongs to the InputsKernel family -- FallbackKernel,
        # ConcatKernel, etc). A ComputedBuffer's own upstream reads instead
        # come from its inner_fn, surfaced via get_read_writes().reads. A
        # tile_dim_marker op has exactly one MemoryDep read (the tile it
        # marks).
        marker_reads = [
            dep
            for dep in marker_op.get_read_writes().reads
            if isinstance(dep, MemoryDep)
        ]
        if len(marker_reads) != 1:
            raise AssertionError(
                f"tile_dim_marker op {marker_name!r} has "
                f"{len(marker_reads)} MemoryDep reads; expected exactly 1 "
                "(the tile it marks)."
            )
        marker_input_name = marker_reads[0].name

        # A marker CAN have more than one consuming read: e.g.
        # torch.softmax's default decomposition (amax, sub, exp, sum, div)
        # reads its own input tensor directly from two sibling ops (amax
        # and sub both read the pre-decomposition placeholder), so a
        # for_each_tile tile whose body calls softmax on the whole, untiled
        # tile produces a marker with two independent ComputedBuffer
        # consumers rather than one op chained through another. Each
        # consuming read is
        # resolved independently below -- inlining the marker's transform
        # into each ComputedBuffer consumer in turn, or (for a StarDep
        # consumer) redirecting each such reference -- and the marker
        # itself is erased only once every consuming read has been
        # resolved.
        any_star_dep_consumer = False
        for consumer_op, consumer_dep in consumers:
            # Set by the ComputedBuffer branch; stays False (unchanged behavior)
            # for a StarDep consumer, which has no captured load-site coordinate.
            _marker_axis_consumed = False
            if hasattr(consumer_op, "data"):
                # A consumer can independently read marker_input_name BEFORE
                # inlining too -- e.g. a body op shaped like
                # `x @ tile_dim_marker(x)` reads the marker's own input
                # directly as one operand and through the marker as the
                # other. Snapshot those pre-existing reads so the check below
                # only requires exactly one NEW read of marker_input_name
                # (the one _inline_marker_into_consumer just composed in),
                # not exactly one in total.
                pre_inline_reads = [
                    d
                    for d in consumer_op.get_read_writes().reads
                    if isinstance(d, MemoryDep) and d.name == marker_input_name
                ]
                new_consumer, _marker_axis_consumed = _inline_marker_into_consumer(
                    consumer_op, marker_op, operations
                )
                # _inline_marker_into_consumer swaps `operations[op_idx]` in
                # place but returns a new object whose reads no longer include
                # `consumer_dep` at all -- the marker's own inner_fn (its
                # genuine per-iteration coordinate transform, e.g. an extra
                # `+ 24*u0` advance term) is now composed directly into the
                # consumer's read of `marker_input_name`, replacing the old,
                # marker-relative read. lookup_marker_dim looks up entries
                # keyed by the CURRENT read it finds on `op` at lookup time, so
                # the map must be keyed by that post-inline dep (the one now
                # naming `marker_input_name`, with the marker's own index
                # composed in), not by the stale pre-inline `consumer_dep`
                # (which named the marker itself and will never appear among
                # new_consumer's reads again).
                new_reads = [
                    d
                    for d in new_consumer.get_read_writes().reads
                    if isinstance(d, MemoryDep) and d.name == marker_input_name
                ]
                brand_new_reads = [d for d in new_reads if d not in pre_inline_reads]
                if len(brand_new_reads) != 1:
                    raise AssertionError(
                        f"consumer {consumer_op.get_name()!r} has "
                        f"{len(brand_new_reads)} newly-inlined MemoryDep reads "
                        f"named {marker_input_name!r} (of {len(new_reads)} "
                        f"total, {len(pre_inline_reads)} pre-existing); "
                        "expected exactly 1 new one (the inlined read that "
                        f"used to go through erased marker {marker_name!r})."
                    )
                new_dep = brand_new_reads[0]
            else:
                # StarDep-shaped consumer (ExternKernelOut/FallbackKernel/
                # ConcatKernel/... -- including a nested ir.WhileLoop, whose own
                # .carried_inputs/.additional_inputs are the read shape
                # _substitute_direct_input_refs's docstring calls its "fourth
                # read shape"): no inner_fn to wrap, so
                # redirect_computed_buffer_reads does not apply, and there is no
                # load index to compose the marker's own transform into the way
                # _inline_marker_into_consumer does for a ComputedBuffer
                # consumer above.
                #
                # The marker's own ComputedBuffer performs a genuine,
                # non-identity per-iteration coordinate transform (the tile's
                # slice/offset -- see lower_tile_dim_marker's docstring), the
                # same as for the ComputedBuffer-consumer branch above.
                # Pointing the consumer's reference at the marker's own
                # upstream input (marker_input_name) instead of at the
                # marker itself discards that transform entirely: every
                # consumer read then sees the raw, untiled operand with no
                # per-iteration offset at all -- on a nested for_each_tile
                # (map/map), the inner loop's captured outer-tile operand
                # stays pinned to outer trip 0's slice on every trip,
                # corrupting every outer iteration after the first. This is
                # Spyre-codegen-specific and stays invisible on CPU
                # eager/CPU Inductor, which never exercises a StarDep-shaped
                # nested-WhileLoop marker consumer's codegen.
                #
                # The correct erasure-equivalent for this read shape is to keep
                # the marker's ComputedBuffer materialized (never remove it from
                # group_ops/operations) and redirect the consumer's reference to
                # the marker ITSELF rather than to its upstream input --
                # equivalent in effect to _inline_marker_into_consumer's
                # per-load composition, just realized as a standalone buffer
                # instead of fused into the consumer's own body, since a
                # StarDep-shaped consumer has no body to fuse into. Only a
                # stale-by-identity, same-name reference (splice_while_loop's
                # own upstream passes can leave a consumer's direct object
                # reference pointing at an object that predates the marker's
                # final reconstruction, even though it already names the
                # marker correctly) needs patching at all --
                # _substitute_direct_input_refs's name-based resolve() is a
                # no-op for any reference that already points at marker_op by
                # identity, and safely repoints any reference that doesn't.
                _substitute_direct_input_refs([consumer_op], {marker_name: marker_op})
                new_consumer = consumer_op
                # No object reconstruction happened (unlike the ComputedBuffer
                # branch) -- consumer_op's own identity is unchanged, and its
                # dep still names marker_name (the marker is not erased, so
                # nothing renamed it) -- unlike the erase-and-redirect path this
                # replaced, there is no post-substitution name change to
                # re-derive a new dep from.
                new_dep = consumer_dep
                any_star_dep_consumer = True

            # Skip a consumed read: its `dim` names a marker axis that no longer
            # exists in the read, and recording it is what would let
            # lookup_marker_dim's coefficient coincidence mis-resolve it. A
            # retained read of a mixed op is still entered and still resolves.
            if not _marker_axis_consumed:
                marker_map[(new_consumer.get_name(), new_dep)] = dim
            if id(consumer_op) in group_op_ids:
                group_op_ids.discard(id(consumer_op))
                group_op_ids.add(id(new_consumer))
                group_ops[group_ops.index(consumer_op)] = new_consumer

        # Only erase the marker once every consuming read has been resolved
        # above -- with multiple consumers, erasing after the first would
        # leave later consumers' MemoryDep/StarDep reads dangling (naming a
        # marker no longer present in `operations`).
        #
        # If ANY consumer reached the marker via a StarDep, the marker must
        # stay materialized for that consumer's sake (see the StarDep branch
        # above) -- even when OTHER consumers were ComputedBuffers and had
        # the marker's transform inlined into them directly. Erasing here
        # would only be safe if every consumer was inlined.
        if any_star_dep_consumer:
            marker_op.tile_marker_resolution = MarkerResolution.STAR_DEP_KEPT
        else:
            # Only the ComputedBuffer/inline branch actually fuses the
            # marker's transform into the consumer and erases the marker;
            # the StarDep branch below deliberately keeps marker_op alive
            # in BOTH group_ops and operations (see its comment) -- it must
            # still codegen as a real, addressable buffer for the StarDep
            # consumer to read, and _validate_contiguous (coarse_tile.py)
            # requires every group's ops to occupy a gapless block of
            # `operations`, so removing it from `operations` while keeping
            # it out of `group_ops` breaks that contiguity check for any
            # group whose block the marker sits inside. Passes that must not treat a
            # surviving marker as an ordinary tile op instead guard on
            # `_marker_dim(op) is not None` individually (see
            # _plan_read_copies in coarse_tile.py for the first such guard)
            # -- or, where the finer INLINE_ERASED/STAR_DEP_KEPT distinction
            # matters, on `_marker_resolution(op)`.
            marker_op.tile_marker_resolution = MarkerResolution.INLINE_ERASED
            operations.remove(marker_op)
            if marker_op in group_ops:
                group_ops.remove(marker_op)
            group_op_ids.discard(id(marker_op))

    _MARKER_MAPS.setdefault(id(operations), {}).update(marker_map)
    return marker_map


def lookup_marker_dim(
    op: "ir.Operation", loop_var: sympy.Symbol
) -> "tuple[int, bool] | None":
    """Resolve loop_var's tiled-dim position for op via the marker map.

    Walks op's own reads and looks each one up in whichever _MARKER_MAPS
    entry was populated for the operations list this op belongs to, to
    confirm the read the marker map recorded for `op` is still present and
    to identify which of THAT read's own index variables carries loop_var's
    per-trip advance. Returns None if no mapped read resolves, signaling the
    caller (coarse_tile.py's _hint_ranges_pos) to raise rather than guess.

    The marker map's stored int (see _consume_tile_dim_markers) is NOT the
    position this function returns -- it is tile_dim_marker's own `dim`
    argument (for_each_tile.py's `spec.dim`), a position in the MARKER's
    OWN tensor shape (the tile operand's layout), unrelated to the
    consumer op's `data.ranges`/`data.reduction_ranges` numbering that every
    caller of _hint_ranges_pos requires (see its own docstring: "The
    position indexes op.data.ranges when the second element is False and
    op.data.reduction_ranges when it is True"). Passing the marker's raw
    dim straight through silently mis-selects the tiled position whenever
    the tile's own shape ordering differs from the consumer's -- e.g. a
    matmul reading a stacked tile leaf, where the marker's dim indexes the
    2-D tile [rows, cols] but the matmul's own output/reduction dims are
    numbered differently. A CPU aten-fallback matmul never surfaces this
    because it is a StarDep consumer (see below), which never reaches this
    position-mapping code at all -- so this class of bug is only visible
    on real Spyre-device codegen paths.

    So instead of trusting the map's stored int, re-derive the consumer's
    own position directly, scoped to exactly the one dep the marker map
    already identified as the tile read: find the read's own index
    variable `var` whose extent matches loop_var's per-trip advance
    (dep.index.coeff(loop_var) == dep.index.coeff(var) * dep.ranges[var]),
    then map `var` into op's own output coordinates
    (_loop_var_to_ranges_pos) or, if that misses and op is a Reduction,
    into op's own reduction vars (reduction_loop_vars.index).

    More than one var in dep.ranges can satisfy the same coefficient-
    coincidence equation on the SAME read -- e.g. a reduction dim whose
    extent numerically coincides with the tile size (flash-attention's
    online-softmax body, where D == SOFTMAX_TILE_SIZE). Because markers are
    authoritative and there is no fallback heuristic once a marker has
    identified the read, this ambiguity cannot be resolved by guessing:
    collect EVERY candidate var on the marker-identified dep (don't return
    on the first one found), and if more than one survives, raise rather
    than silently pick one. (This has not been observed to trigger against
    any fixture in this repo, including online-softmax's own D ==
    SOFTMAX_TILE_SIZE coincidence -- that coincidence lands on a read the
    marker map does NOT identify as the tile, so it never reaches this
    per-dep candidate collection at all -- but the check must still exist
    so a future shape that does collide on the marker's own dep fails
    loudly instead of guessing.)

    A mapped dep can be either a MemoryDep (ComputedBuffer/inner_fn-backed
    consumer) or a StarDep (InputsKernel-family consumer, e.g.
    ExternKernelOut -- see _consume_tile_dim_markers). StarDep has no
    .index/.ranges (.index raises NotImplementedError) and no coordinate
    space to resolve a position in at all -- but this is moot, not a gap:
    plan_coarse_tile_groups (coarse_tile.py, this function's only real
    caller path) already skips every non-ComputedBuffer op before ever
    calling _hint_ranges_pos, and every StarDep-shaped consumer
    (ExternKernelOut and the rest of the InputsKernel family) has no
    `.data`/inner_fn and so is never a ComputedBuffer. A StarDep-mapped
    entry therefore never needs a resolved position in practice; return
    None for it rather than guess.

    Scoped to ONLY the marker map belonging to the CURRENT compile's own
    ``V.graph.operations`` list -- never every entry in the module-level
    ``_MARKER_MAPS`` registry (see ``_MARKER_MAPS``'s own docstring for why
    searching every entry in the registry would be unsafe). ``V.graph`` is
    the live ``GraphLowering`` for whichever compile is currently running
    this pass pipeline, so ``V.graph.operations`` is guaranteed to be the
    SAME list object ``_consume_tile_dim_markers`` was given for this exact
    compile.
    """
    from torch._inductor.dependencies import Dep, MemoryDep
    from torch._inductor.ir import Reduction
    from torch._inductor.virtualized import V

    from torch_spyre._inductor.wsr.coarse_tile import (
        _loop_var_to_ranges_pos,
        op_out_coords,
        reduction_loop_vars,
    )

    rw = op.get_read_writes()
    op_name = op.get_name()
    marker_map = _MARKER_MAPS.get(id(V.graph.operations))
    if marker_map is not None:
        for dep in rw.reads:
            if not isinstance(dep, Dep):
                continue
            if (op_name, dep) not in marker_map:
                continue
            if not isinstance(dep, MemoryDep):
                # StarDep-shaped mapped entry: no index/ranges to resolve a
                # position from, and (per docstring) never actually reached
                # by a real caller. Keep searching other reads rather than
                # claim a position that doesn't exist.
                continue

            index = dep.index
            if not isinstance(index, sympy.Basic):
                continue
            sym_coeff = index.coeff(loop_var)
            if sym_coeff == 0:
                continue

            out_coords = op_out_coords(op)
            red_vars = (
                reduction_loop_vars(op)
                if isinstance(getattr(op, "data", None), Reduction)
                else []
            )
            # Collect EVERY var on this one dep that satisfies the
            # coefficient-coincidence equation -- do not return on the
            # first match. More than one candidate here is the narrow,
            # within-one-dep ambiguity the deleted _loop_var_pos_from_reads
            # guarded via cross-read corroboration (see this function's
            # own docstring); with a single ground-truth dep and no second
            # read to corroborate against, the only safe response to
            # multiple candidates is to raise, not to silently pick one.
            candidates: list[tuple[int, bool, sympy.Symbol]] = []
            for var, rng in dep.ranges.items():
                var_coeff = index.coeff(var)
                if var_coeff == 0:
                    continue
                if sympy.simplify(sym_coeff - var_coeff * rng) != 0:
                    continue
                pos = _loop_var_to_ranges_pos(out_coords, var)
                if pos is not None:
                    candidates.append((pos, False, var))
                elif var in red_vars:
                    candidates.append((red_vars.index(var), True, var))
            if len(candidates) > 1:
                names = ", ".join(str(c[2]) for c in candidates)
                raise AssertionError(
                    f"WhileLoop-splice hint's loop_var {loop_var} resolved "
                    f"to {len(candidates)} candidate index variables "
                    f"({names}) on op {op_name!r}'s marker-identified read "
                    f"{dep!r}, all equally satisfying the coefficient-"
                    "coincidence check. The marker map identifies WHICH "
                    "read is the tile, but not which of that read's own "
                    "index variables is the one loop_var actually "
                    "advances -- a numeric coincidence between two dims' "
                    "extents (e.g. a reduction dim's size matching the "
                    "tile size) can satisfy the same equation for more "
                    "than one variable. Markers are authoritative and "
                    "there is no fallback heuristic for this: raising "
                    "here surfaces the gap instead of silently picking "
                    "one candidate over the other."
                )
            if candidates:
                pos, is_reduction, _ = candidates[0]
                return pos, is_reduction
    return None


def _stamp_direct_loop_info(
    group_ops: list["ir.Operation"],
    loop_var: sympy.Symbol,
    trip_count: sympy.Expr,
    group_idx: int,
) -> None:
    """Directly construct and stamp one CoarseTileInfo level per op.

    Ground truth only -- trip count from try_prove_for_each_tile, loop_var
    from _body_loop_var, per-op tiled-dim resolution from
    lookup_marker_dim. Never calls coarse_tile_pre_stickify: the program
    already states trip count, loop var, tile dim, and carry roles as
    ground truth, so re-deriving them via double-blind per-level
    coarse_tile_pre_stickify inference is unnecessary and unsafe -- one
    level's committed metadata (e.g. tiled_dims_per_read computed against a
    provisional buffer name) can go stale once a later-spliced nested level
    renames or rewires the buffers it depended on. propagation is never
    produced -- every CoarseTileInfo built here leaves it None -- because
    PropagationPlan has zero consumers outside coarse_tile.py's own Pass
    1/2/3, which while_loop groups skip entirely.

    Called once per while_loop nesting level, in group_idx-ascending
    (outermost-first) order -- see splice_while_loops's own docstring for
    why stamping is deferred to a single final phase in this order. Each
    call extends whatever single CoarseTileInfo a strictly-outer level's
    own call already stamped, never overwriting it. ``op.loop_info`` is
    ALWAYS a single ``CoarseTileInfo`` (never a list of them) -- consistent
    with every other stamping site in the codebase (coarse_tile.py's own
    ``op.loop_info = dataclasses.replace(info, ...)``, padding.py,
    read_copy_elision.py, insert_restickify.py, and
    work_division_constraints.py's own reader, which does
    ``for level_dims in loop_info.loop_tiled_dims`` directly against
    ``ctx.op.loop_info`` with no list-of-CoarseTileInfo indirection
    anywhere). ``CoarseTileInfo``'s own per-level list fields
    (loop_group_id/loop_count/loop_tiled_dims/loop_tiled_reduction_dims/
    tiled_dims_per_read's and output_tiled_dims's per-level entries) already
    encode every nesting level inside ONE object, outermost first (per
    loop_info.py's docstring). See the `existing is not None` branch below
    for the append-not-prepend convention this relies on when a later,
    strictly-inner call extends an already-stamped op.

    tiled_dims_per_read/output_tiled_dims are filled from
    ``op.get_read_writes()`` ground truth: a dep advances at this level iff
    its own index has a nonzero coefficient on loop_var, in which case its
    extent for this (single) level is simply trip_count -- no multi-level
    extent composition is needed here, since each call only ever computes
    one new level's contribution before appending it onto any existing
    per-read entries. ``op.get_read_writes()`` returns reads in the same
    order regardless of which level's call invokes it (it is a pure
    function of the op's own current IR, not of loop_info), so appending
    this level's per-read entry at the same read-index an outer call already
    populated is positionally safe.

    squeezed_advance_per_read/squeezed_advance_output cover a dep whose
    index carries a nonzero coefficient on loop_var that neither
    _extent_at_pos nor _structural_resolve could attribute to one of this
    op's own tiled dims (e.g. a point-shaped or already-resolved-elsewhere
    read/write) -- mirroring coarse_tile.py's own
    _point_splice_advance_for_dep, which records the identical
    (coefficient, 1) shape for its own splice-only advances. This is
    stamped explicitly here, at ground-truth time, specifically so
    SpyreKernel._general_tile_advance never needs to re-derive whether a
    splice symbol merely appearing in dep.index implies a real per-trip
    advance -- it does not in general (a restickified V-tile pool read is
    genuinely pinned across trips despite loop_var being a free symbol of
    its dep.index by construction). A dep whose coefficient on
    loop_var is exactly zero gets no entry at all here, which is itself the
    explicit "pinned, do not advance" verdict -- not a gap for
    _general_tile_advance to fill in.

    Also appends a minimal ``DimHint(loop_var=loop_var,
    loop_var_range=trip_count)`` onto ``op.dim_hints``. This is NOT a
    revival of ``_synthesize_dim_hints_for_group`` (no hint_id minting, no
    is_reduction/dim_names/split_count advisory content -- those fields are
    left at their dataclass defaults and unread by any consumer on this
    path). It exists only to keep ``loop_var_ranges_from_dim_hints`` (see
    pass_utils.py) working: that helper -- and its callers
    ``op_out_coords`` and ``_build_indirect_store_subs`` -- reads
    ``{h.loop_var: h.loop_var_range for h in op.dim_hints}`` to recognize a
    WhileLoop-splice loop_var that is deliberately not a ``dep.ranges`` key.
    Without this, ``_build_indirect_store_subs`` misclassifies loop_var as
    a runtime scatter-row symbol (its only signal is "not a loop range
    key"), which crashed as "indirect symbol not found in indirect_sizes
    {}" once this function stopped populating dim_hints. See
    _build_indirect_store_subs's own docstring for the identical bug this
    once already fixed for the old mechanism.
    """
    from torch._inductor.dependencies import MemoryDep
    from torch._inductor.ir import Reduction

    from torch_spyre._inductor.errors import Unsupported
    from torch_spyre._inductor.loop_info import CoarseTileInfo
    from torch_spyre._inductor.propagate_hints import DimHint
    from torch_spyre._inductor.wsr.coarse_tile import (
        _loop_var_to_ranges_pos,
        op_out_coords,
        reduction_loop_vars,
    )

    def _structural_resolve(
        dep: "MemoryDep", prior_levels: "list[tuple[int, sympy.Expr]] | None"
    ) -> "tuple[int, sympy.Expr] | None":
        """Structural fallback: resolve (pos, extent) for dep without a
        marker, when lookup_marker_dim's op-level resolution didn't apply
        or didn't cover this particular dep.

        Only attempted when dep.index actually carries loop_var (per-dep,
        not per-op, since reads and the write can each independently need
        it). Uses op's own output coordinates (ground truth for the op's
        own tiled-dim positions) to find loop_var's ranges position, then
        derives this dep's own per-trip extent from its own coefficient on
        loop_var divided by the mapped symbol's coefficient in this SAME
        dep's index -- reusing trip_count here would be wrong whenever this
        dep's own per-trip step differs from loop_var's per-trip step.

        op_out_coords can raise Unsupported for >=3 levels of nesting, when
        an outer level's own loop_var is not yet covered by this op's
        dim_hints -- caught here and treated identically to "no structural
        match", i.e. no stamp for this dep, not a crash. This is strictly
        more permissive than the no-fallback behavior (every dep hit the
        no-stamp outcome before this fallback existed), so it cannot
        regress a case that worked before.

        `prior_levels` is this SAME dep's own already-stamped (pos, extent)
        entries, flattened across every strictly-inner level a prior call
        already stamped (``existing.tiled_dims_per_read[i]`` /
        ``existing.output_tiled_dims`` flattened, before this level's own
        entry is appended -- None/empty on the first, outermost call).
        Reject a candidate `structural_pos` only when THIS dep already has
        a stamped level at that same position from a strictly-inner call --
        that is a genuine collision: two independent levels both claiming
        the same device dim, each minting its own advance symbol in
        _general_tile_advance, so the dim would advance twice per trip
        instead of once. Rejecting merely because `loop_var` shares free
        symbols with another coordinate is NOT safe: a WhileLoop-splice-
        folded write index (e.g. coordinate ``d0 + 2*u5``, where ``d0`` is
        this op's own local tile coordinate) can legitimately have that
        shape with no real collision at all -- rejecting unconditionally
        there leaves output_tiled_dims empty and breaks codegen addressing.
        So the check is scoped precisely to "this dep already has a stamped
        level at this exact position from another level," never to "this
        coordinate has extra free symbols."
        """
        dep_loop_var_coeff = dep.index.coeff(loop_var)
        if dep_loop_var_coeff == 0:
            return None
        try:
            out_coords = op_out_coords(op)
        except Unsupported:
            return None
        structural_pos = _loop_var_to_ranges_pos(out_coords, loop_var)
        if structural_pos is None:
            return None
        # Reject only a genuine collision: this dep already carries a
        # stamped level at structural_pos from a strictly-inner call -- see
        # this function's own docstring. A coordinate where loop_var shares
        # free symbols with another var is NOT by itself a reason to
        # reject -- see the same docstring.
        if prior_levels and any(pos == structural_pos for pos, _ in prior_levels):
            return None
        # mapped_sym: this SAME dep's own index variable occupying
        # structural_pos -- resolved the same convention
        # _loop_var_to_ranges_pos itself uses (a var's position is where it
        # is found in op_out_coords), scanning dep.ranges (this dep's own
        # iteration variables, positionally aligned with op.data.ranges by
        # construction) rather than loop_var, mirroring lookup_marker_dim's
        # own dep.ranges.items() scan above.
        mapped_sym = None
        for var in dep.ranges:
            if _loop_var_to_ranges_pos(out_coords, var) == structural_pos:
                mapped_sym = var
                break
        if mapped_sym is None:
            return None
        mapped_coeff = dep.index.coeff(mapped_sym)
        if mapped_coeff == 0:
            return None
        # sympy.Mod(a, b) only evaluates to a concrete integer when both a
        # and b are numeric; with a symbolic coefficient (e.g. an
        # as-yet-unbound trip-count symbol) it stays an unevaluated Mod
        # expression, which is truthy under `!= 0` and would reject a
        # structurally valid resolution. Only apply the divisibility check
        # when both coefficients are actually numbers; otherwise fall
        # through and accept the structural resolution as-is.
        mod_check = sympy.Mod(dep_loop_var_coeff, mapped_coeff)
        if mod_check.is_number and mod_check != 0:
            return None
        extent = dep_loop_var_coeff / mapped_coeff
        return structural_pos, extent

    def _extent_at_pos(
        dep: "MemoryDep", pos: int, is_reduction: bool
    ) -> "sympy.Expr | None":
        """This dep's own per-trip tile extent at marker-resolved `pos`.

        lookup_marker_dim resolves POSITION only (via a coefficient-
        coincidence match against loop_var, then discards the matched
        var/range -- see its own docstring). The actual extent to stamp
        is that matched var's own range in THIS dep (dep.ranges[var]),
        not trip_count: trip_count is the loop's iteration count, while
        the extent tiled_dims_per_read/output_tiled_dims must carry is
        the tile's own per-trip size in the dim's host-range units (see
        loop_info.py's tiled_dims_per_read docstring) -- these coincide
        only when tile_size==1. Stamping trip_count instead of the real
        per-trip extent doubles (or otherwise miscomputes) the read-side
        device advance, silently reading past/aliasing wrong rows on
        later trips whenever tile_size != 1.

        `pos` is in the same namespace lookup_marker_dim resolved it in:
        op_out_coords positions when is_reduction is False, or this op's
        own reduction_loop_vars positions when True (mirrors
        lookup_marker_dim's own out_coords/red_vars split). Using
        op_out_coords unconditionally here would always miss for a
        reduction-position resolution -- op_out_coords is scoped to the
        op's WRITE dep only (see pass_utils.op_out_coords) and never
        contains a reduction position -- so a dep that genuinely tiles
        along a reduction dim (e.g. a K/V tile read inside an
        online-softmax reduction) would fall through to the
        squeezed-advance fallback below, which is only valid for
        point-shaped reads (see that branch's comment), silently
        producing a wrong device advance for a real, multi-element tiled
        read.

        Re-derives the same coefficient-coincidence equation
        lookup_marker_dim used to find pos in the first place
        (dep.index.coeff(var) * dep.ranges[var] == dep.index.coeff(
        loop_var)), scoped to dep's own ranges rather than trusting
        `pos` alone -- `pos` is a position in the relevant namespace, and
        more than one dep.ranges var can share it only when they're
        genuinely the same tiled dim, so re-matching here is safe and
        mirrors _structural_resolve's identical pattern just above.
        """
        if is_reduction:
            red_vars = (
                reduction_loop_vars(op)
                if isinstance(getattr(op, "data", None), Reduction)
                else []
            )
            if pos >= len(red_vars):
                return None
            target_var = red_vars[pos]
        else:
            out_coords = op_out_coords(op)
            if pos >= len(out_coords):
                return None
            target_var = None
        dep_loop_var_coeff = dep.index.coeff(loop_var)
        for var, rng in dep.ranges.items():
            if is_reduction:
                if var != target_var:
                    continue
            elif _loop_var_to_ranges_pos(out_coords, var) != pos:
                continue
            var_coeff = dep.index.coeff(var)
            if var_coeff == 0:
                continue
            if sympy.simplify(dep_loop_var_coeff - var_coeff * rng) == 0:
                return rng
        return None

    for op in group_ops:
        existing: CoarseTileInfo | None = getattr(op, "loop_info", None)

        prior_hints = list(getattr(op, "dim_hints", None) or [])
        op.dim_hints = [
            *prior_hints,
            DimHint(
                dim_names=[],
                split_count=1,
                loop_var=loop_var,
                is_reduction=False,
                loop_var_range=trip_count,
            ),
        ]

        resolved = lookup_marker_dim(op, loop_var)
        loop_tiled_dims: list[int] = []
        loop_tiled_reduction_dims: list[int] = []
        resolved_pos: int | None = None
        resolved_is_reduction = False
        # tiled_dims_per_read/output_tiled_dims (unlike loop_tiled_dims/
        # loop_tiled_reduction_dims, which are separate lists and always
        # store a raw op.data.reduction_ranges index) use ONE shared
        # position space where a reduction dim is offset by
        # n_output_dims == len(op.data.ranges) -- see loop_info.py's
        # tiled_dims_per_read docstring ("n_output_dims + reduction_pos
        # for reduction dims") and spyre_kernel.py's
        # _host_dim_to_index_symbol, which decodes exactly that offset.
        # _extent_at_pos operates in this same shared space (its
        # is_reduction branch indexes reduction_loop_vars directly, which
        # is reduction_pos, not the offset value), so resolved_pos must be
        # converted here before being passed to it or stored below.
        tiled_dims_per_read_pos: int | None = None
        if resolved is not None:
            ranges_pos, is_reduction = resolved
            resolved_pos = ranges_pos
            resolved_is_reduction = is_reduction
            if is_reduction:
                loop_tiled_reduction_dims.append(ranges_pos)
                n_output_dims = len(op.data.ranges) if hasattr(op.data, "ranges") else 0
                tiled_dims_per_read_pos = n_output_dims + ranges_pos
            else:
                loop_tiled_dims.append(ranges_pos)
                tiled_dims_per_read_pos = ranges_pos
        elif (marker_dim := _marker_dim(op)) is not None:
            # op is itself a tile_dim_marker (not a marker CONSUMER, which
            # lookup_marker_dim above already covers) that survived splicing
            # as a real, addressable op -- e.g. a StarDep-consumed marker
            # kept materialized as a still-nested WhileLoop's carried input
            # (MarkerResolution.STAR_DEP_KEPT; see splice_while_loops's own
            # docstring on _recordable_op_names). Such an op's own write
            # never carries loop_var (only its READ of the underlying
            # tensor does), so lookup_marker_dim's marker-map lookup (keyed
            # by CONSUMER op name) finds no entry, and _structural_resolve's
            # op_out_coords-anchored fallback below also can't resolve it.
            # tile_marker_dim is ground truth for this op's own tiled
            # position (for_each_tile.py's spec.dim, stamped by
            # lower_tile_dim_marker onto this exact op/tensor -- no
            # cross-shape translation risk here, unlike lookup_marker_dim's
            # marker-to-consumer case, since the marker IS this op).
            resolved_pos = marker_dim
            loop_tiled_dims.append(resolved_pos)
            tiled_dims_per_read_pos = marker_dim

        rw = op.get_read_writes()
        # StarDep has no .index (raises NotImplementedError, not
        # AttributeError, so hasattr(dep, "index") is not a safe filter
        # here) -- isinstance against MemoryDep is the correct guard, same
        # as lookup_marker_dim's own filtering above.
        reads = [dep for dep in rw.reads if isinstance(dep, MemoryDep)]
        existing_per_read = list(existing.tiled_dims_per_read) if existing else []
        new_tiled_dims_per_read: list[list[tuple[int, sympy.Expr]]] = []
        new_squeezed_advance_per_read: list[list[tuple[sympy.Expr, sympy.Expr]]] = []
        for read_idx, dep in enumerate(reads):
            per_level: list[tuple[int, sympy.Expr]] = []
            squeezed_level: list[tuple[sympy.Expr, sympy.Expr]] = []
            extent = (
                _extent_at_pos(dep, resolved_pos, resolved_is_reduction)
                if resolved_pos is not None and dep.index.coeff(loop_var) != 0
                else None
            )
            if extent is not None and tiled_dims_per_read_pos is not None:
                per_level.append((tiled_dims_per_read_pos, extent))
            else:
                prior_levels = (
                    [entry for level in existing_per_read[read_idx] for entry in level]
                    if read_idx < len(existing_per_read)
                    else None
                )
                structural = _structural_resolve(dep, prior_levels)
                if structural is not None:
                    per_level.append(structural)
                else:
                    # loop_var is a genuine free symbol of this dep's index
                    # but neither a marker nor op_out_coords could resolve
                    # it to one of this op's own tiled dims -- e.g. a
                    # ReStickifyOpHBM/pool read whose dep.index carries no
                    # loop_var term at all (dep genuinely pinned: coeff==0,
                    # nothing to record here), or -- the case this branch
                    # exists for -- a point-shaped or otherwise dim-less
                    # dependency where loop_var is the SOLE source of the
                    # per-trip address step (mirrors coarse_tile.py's
                    # _point_splice_advance_for_dep, which records the
                    # identical shape for its own splice-only advances).
                    # Stamp that coefficient explicitly into
                    # squeezed_advance_per_read/output rather than leaving
                    # it for SpyreKernel._general_tile_advance to re-derive
                    # from raw dep.index.free_symbols -- that re-derivation
                    # cannot distinguish "loop_var present but this dep is
                    # pinned" from "loop_var present and this dep genuinely
                    # advances," which would silently corrupt a
                    # restickified V-tile pool buffer's address on trip 2+.
                    # A zero coefficient here means this dep's address truly
                    # does not depend on loop_var -- leave squeezed_level
                    # empty, an explicit "pinned" verdict, not an omission.
                    coeff = dep.index.coeff(loop_var)
                    if coeff != 0:
                        squeezed_level.append((coeff, sympy.Integer(1)))
            new_tiled_dims_per_read.append(per_level)
            new_squeezed_advance_per_read.append(squeezed_level)

        output_tiled_dims_level: list[tuple[int, sympy.Expr]] = []
        squeezed_advance_output_level: list[tuple[sympy.Expr, sympy.Expr]] = []
        writes = [dep for dep in rw.writes if isinstance(dep, MemoryDep)]
        if len(writes) > 1:
            raise Unsupported(
                f"op {op.get_name()!r} has {len(writes)} MemoryDep writes; "
                "_stamp_direct_loop_info assumes at most one so it can stamp "
                "a single output_tiled_dims/squeezed_advance_output entry"
            )
        if writes:
            write_dep = writes[0]
            write_extent = (
                _extent_at_pos(write_dep, resolved_pos, resolved_is_reduction)
                if resolved_pos is not None and write_dep.index.coeff(loop_var) != 0
                else None
            )
            if write_extent is not None and tiled_dims_per_read_pos is not None:
                output_tiled_dims_level.append((tiled_dims_per_read_pos, write_extent))
            else:
                prior_output_levels = (
                    [entry for level in existing.output_tiled_dims for entry in level]
                    if existing
                    else None
                )
                structural = _structural_resolve(write_dep, prior_output_levels)
                if structural is not None:
                    output_tiled_dims_level.append(structural)
                    # No marker resolved this op's own tiled position (the
                    # `resolved_pos is None` branch above left
                    # loop_tiled_dims empty), but the write dep structurally
                    # resolves to a real tiled position at this level --
                    # e.g. a stacking-carry write-out op that is neither a
                    # marker nor a marker's consumer, so lookup_marker_dim/
                    # _marker_dim both return None for it, yet its write
                    # genuinely advances per trip. loop_tiled_dims must
                    # record this position too: it feeds
                    # spyre_kernel.py's create_op_spec (gated on
                    # loop_tiled_dims, independently of output_tiled_dims)
                    # as well as several coarse_tile.py consumers (e.g.
                    # the reduction/consumer tiling check at
                    # coarse_tile.py:960) that compare loop_tiled_dims
                    # directly. Leaving it empty here caused the minted
                    # advance symbol to appear in
                    # TensorArg.device_tile_advance_expr (built from
                    # output_tiled_dims via _general_tile_advance) but
                    # never in OpSpec.tiled_symbols (built from
                    # loop_tiled_dims via create_op_spec) -- exactly the
                    # failure mode create_op_spec's own comment warns
                    # about, silently pinning this op's write address
                    # across every trip instead of advancing it.
                    structural_pos, _ = structural
                    if structural_pos not in loop_tiled_dims:
                        loop_tiled_dims.append(structural_pos)
                else:
                    # Same explicit-stamping rationale as the read-side
                    # else-branch above: record a genuinely splice-var-only
                    # write advance directly, rather than leaving it for
                    # _general_tile_advance to re-derive from raw
                    # dep.index.free_symbols.
                    write_coeff = write_dep.index.coeff(loop_var)
                    if write_coeff != 0:
                        squeezed_advance_output_level.append(
                            (write_coeff, sympy.Integer(1))
                        )

        if existing is None:
            op.loop_info = CoarseTileInfo(
                loop_group_id=(group_idx,),
                loop_count=[trip_count],
                loop_tiled_dims=[loop_tiled_dims],
                loop_tiled_reduction_dims=[loop_tiled_reduction_dims],
                tiled_dims_per_read=[
                    [per_read] for per_read in new_tiled_dims_per_read
                ],
                output_tiled_dims=[output_tiled_dims_level],
                squeezed_advance_per_read=[
                    [per_read] for per_read in new_squeezed_advance_per_read
                ],
                squeezed_advance_output=[squeezed_advance_output_level],
            )
        else:
            # CANONICAL explanation of the outermost-first append-not-prepend
            # convention (other docstrings in this module cross-reference
            # this comment rather than re-stating it):
            #
            # _stamp_direct_loop_info is called once per while_loop nesting
            # level, in group_idx-ascending (outermost-first) order --
            # splice_while_loops defers every call to a final phase that
            # iterates pending_levels in splice-acceptance order, and the
            # outer while_loop is always accepted (and so appended to
            # pending_levels) before a nested while_loop can become visible.
            # So the FIRST call for a given op is the OUTERMOST level, and
            # already correctly occupies index 0 (the existing=None branch
            # above). Every per-level list field's documented convention is
            # outermost-first (loop_info.py's own docstring; scheduler.py's
            # and coarse_tile.py's reliance on loop_group_id[0] being the
            # outermost level), so each later, strictly-inner call's own
            # contribution must be APPENDED onto the end of what an outer
            # call already stamped -- prepending here would shift the outer
            # level's own key out of index 0, making an inner-nested op's
            # loop_group_id[0] disagree with its outer siblings'
            # loop_group_id[0] and breaking scheduler.py's _build_loop_group
            # grouping, which would produce sibling, not nested,
            # CountedLoopSchedulerNodes.
            merged_tiled_dims_per_read = list(existing.tiled_dims_per_read)
            if merged_tiled_dims_per_read and len(merged_tiled_dims_per_read) == len(
                new_tiled_dims_per_read
            ):
                merged_tiled_dims_per_read = [
                    [*prior_levels, new_level]
                    for prior_levels, new_level in zip(
                        merged_tiled_dims_per_read, new_tiled_dims_per_read
                    )
                ]
            else:
                # Read shape changed between levels (should not happen for
                # the same op, but guard rather than silently misalign
                # positionally) -- fall back to this level's own reads with
                # no prior-level history rather than raising.
                merged_tiled_dims_per_read = [
                    [per_read] for per_read in new_tiled_dims_per_read
                ]

            # Same outermost-first append convention as tiled_dims_per_read
            # above, kept in its own block since it has an independent
            # existing-shape guard (squeezed_advance_per_read can be empty
            # on `existing` even when tiled_dims_per_read is not, e.g. an
            # outer level that resolved every dep structurally and stamped
            # no splice-only advances at all).
            existing_squeezed_per_read = list(existing.squeezed_advance_per_read)
            if existing_squeezed_per_read and len(existing_squeezed_per_read) == len(
                new_squeezed_advance_per_read
            ):
                merged_squeezed_advance_per_read = [
                    [*prior_levels, new_level]
                    for prior_levels, new_level in zip(
                        existing_squeezed_per_read, new_squeezed_advance_per_read
                    )
                ]
            else:
                merged_squeezed_advance_per_read = [
                    [per_read] for per_read in new_squeezed_advance_per_read
                ]

            op.loop_info = dataclasses.replace(
                existing,
                loop_group_id=(*existing.loop_group_id, group_idx),
                loop_count=[*existing.loop_count, trip_count],
                loop_tiled_dims=[*existing.loop_tiled_dims, loop_tiled_dims],
                loop_tiled_reduction_dims=[
                    *existing.loop_tiled_reduction_dims,
                    loop_tiled_reduction_dims,
                ],
                tiled_dims_per_read=merged_tiled_dims_per_read,
                output_tiled_dims=[
                    *existing.output_tiled_dims,
                    output_tiled_dims_level,
                ],
                squeezed_advance_per_read=merged_squeezed_advance_per_read,
                squeezed_advance_output=[
                    *existing.squeezed_advance_output,
                    squeezed_advance_output_level,
                ],
            )


def _recordable_op_names(group_ops: list["ir.Operation"]) -> list[str]:
    """Names group_ops will resolve to by the time every level is spliced.

    A nested (not-yet-accepted-this-iteration) for_each_tile's own
    ir.WhileLoop can still be sitting inside group_ops here, unspliced --
    it only becomes visible to try_prove_for_each_tile on a LATER iteration
    of splice_while_loops's `while True:` driver, once ITS nesting level
    gets accepted and spliced in turn. Recording group_ops's own names
    verbatim would therefore include this WhileLoop's name and its
    MultiOutput children's names -- neither of which survive that later
    splice_while_loop call, which replaces the WhileLoop wholesale with its
    body_subgraph's own ops (under entirely different names) and drops the
    WhileLoop's MultiOutput children from graph.operations outright (see
    splice_while_loop's own docstring and its trailing graph.operations
    filter). A name recorded for either would never resolve at stamp time.

    But every op that a later splice of this nested WhileLoop will
    eventually materialize -- including an accumulator-add op that does not
    exist as a distinct object yet -- already has a fixed, predictable name:
    while_op.body_subgraph.graph.operations is the body subgraph's own op
    list, already carrying its final (fully-prefixed) names before its own
    splice ever runs -- e.g. a doubly-nested body's ops already read as
    "..._while_loop_body_graph_0_0_while_loop_body_graph_0_bufN" even before
    the inner WhileLoop's own splice_while_loop call. So this level's
    op_names must recurse into any nested ir.WhileLoop's own
    body_subgraph.graph.operations (arbitrarily deep, for >2 nesting
    levels) and record ITS names instead of the WhileLoop's own -- that is
    what gives an op materialized only by a later, inner splice a chance to
    receive this (outer) level's own stamped contribution too, which is the
    entire point of deferring stamping in the first place. The WhileLoop's
    own MultiOutput children (which never survive its splice either way)
    are dropped with no replacement.

    One more staleness gap this recursion must account for: a nested,
    not-yet-spliced body's own operations snapshot (taken here, before that
    body's OWN splice_while_loop/`_consume_tile_dim_markers` calls ever run)
    still includes its own tile_dim_marker op(s) -- see `_marker_dim`. Most
    of those are erased outright (not rebuilt under the same name, unlike a
    marker's *consumer*, which _consume_tile_dim_markers rebuilds via
    replace_computed_buffer_body and which therefore DOES keep a stable,
    recordable name -- see splice_while_loops's own docstring) once that
    nested level's own `_consume_tile_dim_markers` call actually runs, on a
    LATER iteration of the `while True:` driver. A marker's name recorded
    here would therefore usually never resolve at stamp time, so markers are
    excluded from the RECURSED (still-nested, not-yet-consumed) branch the
    same way the WhileLoop's own MultiOutput children are.

    This exclusion must NOT apply to the top-level `group_ops` this function
    is originally called with (as opposed to a nested body's operations,
    reached only via the recursive call above): `splice_while_loops` always
    calls `_consume_tile_dim_markers(group_ops, graph.operations)` for
    THIS level's own group_ops before calling this function, so by the time
    this function sees them, every one of this level's own markers has
    already been resolved one of two ways (see MarkerResolution): a
    ComputedBuffer-consumer marker is INLINE_ERASED and no longer present in
    group_ops at all (nothing to exclude), but a StarDep-consumer marker
    (e.g. one whose sole consumer is a still-nested inner WhileLoop's
    carried input -- exactly the outer-M/inner-K shape this module exists
    for) is deliberately kept materialized (STAR_DEP_KEPT) as a real,
    addressable buffer that group_ops still legitimately contains and that
    DOES need this level's own stamp -- excluding it here silently drops it
    from `pending_levels`, leaving its `dim_hints`/`loop_info` unstamped and
    producing a codegen-time "indirect symbol ... not found in
    indirect_sizes" failure. So the marker check below only fires while
    recursing into a still-nested WhileLoop's own body (`nested`), never at
    this function's own top-level `group_ops`.
    """
    from torch._inductor import ir

    def _walk(ops: list["ir.Operation"], nested: bool) -> list[str]:
        names: list[str] = []
        for op in ops:
            if isinstance(op, ir.WhileLoop):
                nested_ops = list(op.body_subgraph.graph.operations)
                names.extend(_walk(nested_ops, nested=True))
                continue
            nested_inputs = getattr(op, "inputs", None) or ()
            if any(isinstance(inp, ir.WhileLoop) for inp in nested_inputs):
                # MultiOutput child of a still-nested WhileLoop; dropped by
                # its later splice, not replaced -- see this function's
                # docstring.
                continue
            if nested and _marker_dim(op) is not None:
                # tile_dim_marker op belonging to a still-nested level;
                # usually erased outright by that level's own (not yet run)
                # marker consumption -- see this function's docstring. Not
                # applied at the top level, where group_ops's own markers
                # have already been resolved (INLINE_ERASED removed them
                # already; STAR_DEP_KEPT ones are real ops that must still
                # be recorded).
                continue
            names.append(op.get_name())
        return names

    return _walk(group_ops, nested=False)


def _identity_load(
    op: "ir.Operation",
) -> tuple[str, sympy.Expr, tuple[sympy.Symbol, ...]] | None:
    """Return the sole load performed by a pure pointwise identity.

    The generated exact-stride normalizations this recognizes are ordinary
    Pointwise buffers, not a dedicated IR node.  Run their body under a
    recording handler so accepting one is based on behavior (one load whose
    value is returned unchanged), not an origin name or a fragile graph
    pattern.
    """
    from torch._inductor import ir

    if not isinstance(op, ir.ComputedBuffer) or not isinstance(op.data, ir.Pointwise):
        return None

    indices = tuple(
        sympy.Symbol(f"_fet_identity_i{i}", integer=True)
        for i in range(len(op.data.ranges))
    )
    recorder = _IdentityLoadRecorder()
    try:
        with V.set_ops_handler(recorder):
            result = op.data.inner_fn(indices)
    except (AssertionError, TypeError, ValueError):
        return None
    # V.ops is an OpsWrapper, so scalar handler results normally come back as
    # OpsValue(value).  Accept the unwrapped form too for direct unit tests.
    if (
        getattr(result, "value", result) is not recorder.value
        or len(recorder.loads) != 1
    ):
        return None
    name, index = recorder.loads[0]
    return name, sympy.sympify(index), indices


def _contract_exact_stride_input_materializations(
    graph: Any,
    pending_levels: list[tuple[sympy.Symbol, sympy.Expr, int, list[str]]],
) -> None:
    """Turn loop-local whole-input copies into one-tile streaming copies.

    Upstream ``WhileLoop.create`` requires exact strides for a body input.  A
    non-contiguous K/V prefix therefore arrives in the spliced graph as a pure
    identity with shape ``[trip_count, tile, ...]``.  Merely splicing the loop
    leaves that full materialization inside every counted-loop trip, although
    its consumer selects exactly one leading slice with the loop variable.

    Prove that relationship from the consumer dependency, shrink the identity's
    trip axis to one, transfer the loop advance to the identity's source read,
    and pin the consumer to the reusable tile-local result.  Pure identity
    copies chained after the first materialization (the matching exact-stride
    repair on a pass-through body output) are contracted along with it.

    The recognition is deliberately narrow.  Any ambiguous axis, non-identity
    producer, non-affine source step, or mismatched loop scope is left unchanged.
    """
    from torch._inductor import ir
    from torch._inductor.dependencies import MemoryDep
    from torch._inductor.ir import FixedLayout

    from torch_spyre._inductor.loop_info import ReadCopyElisionRecord
    from torch_spyre._inductor.wsr.coarse_tile import (
        _LoopVarRebaseHandler,
        _NameSwapHandler,
        _divide_ranges,
        _patch_retiled_load_indexes,
        _splice_loop_vars,
    )

    operations = graph.operations
    identities = {
        op.get_name(): identity
        for op in operations
        if (identity := _identity_load(op)) is not None
    }
    if not identities:
        return

    direct_read_roots: set[str] = set()
    contracted_names: set[str] = set()

    for loop_var, trip_count, group_idx, _op_names in reversed(pending_levels):
        # Every op carrying this level records its absolute group index in the
        # corresponding loop_group_id slot.  Different top-level HOPs can have
        # different tuple lengths, so resolve the slot per op instead of using
        # group_idx as a positional index.
        def level_index(op: "ir.Operation") -> int | None:
            loop_info = getattr(op, "loop_info", None)
            group_ids = getattr(loop_info, "loop_group_id", ())
            try:
                return group_ids.index(group_idx)
            except ValueError:
                return None

        selected: dict[str, tuple[int, sympy.Expr, bool]] = {}

        # Roots are identities whose consumer explicitly selects one slice
        # with this loop's variable.  The coefficient has to equal exactly one
        # producer-axis stride, and that axis has to have trip_count elements.
        for consumer in operations:
            consumer_level = level_index(consumer)
            consumer_info = getattr(consumer, "loop_info", None)
            if consumer_level is None or consumer_info is None:
                continue
            reads = [
                dep
                for dep in consumer.get_read_writes().reads
                if isinstance(dep, MemoryDep)
            ]
            for dep_idx, dep in enumerate(reads):
                producer = graph.try_get_buffer(dep.name)
                if dep.name not in identities or not isinstance(
                    producer, ir.ComputedBuffer
                ):
                    continue
                producer_level = level_index(producer)
                if producer_level != consumer_level:
                    continue
                per_read = (
                    consumer_info.tiled_dims_per_read[dep_idx]
                    if dep_idx < len(consumer_info.tiled_dims_per_read)
                    else []
                )
                squeezed = (
                    consumer_info.squeezed_advance_per_read[dep_idx]
                    if dep_idx < len(consumer_info.squeezed_advance_per_read)
                    else []
                )
                has_level_advance = (
                    consumer_level < len(per_read) and bool(per_read[consumer_level])
                ) or (consumer_level < len(squeezed) and bool(squeezed[consumer_level]))
                coefficient = sympy.simplify(dep.index.coeff(loop_var))
                layout = getattr(producer, "layout", None)
                if (
                    not has_level_advance
                    or coefficient == 0
                    or not isinstance(layout, FixedLayout)
                ):
                    continue
                axes = [
                    dim
                    for dim, (size, stride) in enumerate(
                        zip(layout.size, layout.stride, strict=True)
                    )
                    if sympy.simplify(size - trip_count) == 0
                    and sympy.simplify(stride - coefficient) == 0
                ]
                # Repeated extents/strides can make more than one producer
                # axis fit the arithmetic.  The dependency then does not prove
                # which axis the loop selects, so do not guess: retaining the
                # full materialization is the correctness-preserving fallback.
                if len(axes) != 1:
                    continue
                axis = axes[0]
                source_name, source_index, identity_indices = identities[dep.name]
                source_step = sympy.simplify(source_index.coeff(identity_indices[axis]))
                if source_step == 0:
                    continue
                selected[dep.name] = (axis, source_step, True)
                if source_name in graph.graph_input_names:
                    direct_read_roots.add(dep.name)

        # Exact-stride normalization can also add a second whole-size identity
        # on the pass-through body output.  Contract a pure identity chain as
        # long as its source has already been selected and its matching local
        # axis is unambiguous.  Only the root reads the external source with an
        # advancing address; descendants read the root's rewritten tile-local
        # scratch at a fixed address.
        changed = True
        while changed:
            changed = False
            for name, (
                source_name,
                source_index,
                identity_indices,
            ) in identities.items():
                if name in selected or source_name not in selected:
                    continue
                op = graph.try_get_buffer(name)
                source = graph.try_get_buffer(source_name)
                if not isinstance(op, ir.ComputedBuffer) or not isinstance(
                    source, ir.ComputedBuffer
                ):
                    continue
                if level_index(op) != level_index(source):
                    continue
                source_axis = selected[source_name][0]
                source_layout = getattr(source, "layout", None)
                layout = getattr(op, "layout", None)
                if not isinstance(source_layout, FixedLayout) or not isinstance(
                    layout, FixedLayout
                ):
                    continue
                source_stride = source_layout.stride[source_axis]
                axes = [
                    dim
                    for dim, size in enumerate(layout.size)
                    if sympy.simplify(size - trip_count) == 0
                    and sympy.simplify(
                        source_index.coeff(identity_indices[dim]) - source_stride
                    )
                    == 0
                ]
                # A chain link must identify exactly one matching local axis.
                # Choosing the first of several matches could contract a
                # different dimension and silently change the copied values.
                if len(axes) != 1:
                    continue
                selected[name] = (axes[0], sympy.S.Zero, False)
                changed = True

        if not selected:
            continue

        group_ids = {
            getattr(graph.try_get_buffer(name), "loop_info").loop_group_id
            for name in selected
        }
        if len(group_ids) != 1:
            continue
        group_id = next(iter(group_ids))
        group_ops = [
            op
            for op in operations
            if getattr(getattr(op, "loop_info", None), "loop_group_id", None)
            == group_id
        ]

        retiled_infos = {}
        for name, (axis, source_step, owns_advance) in selected.items():
            producer = graph.try_get_buffer(name)
            assert isinstance(producer, ir.ComputedBuffer)
            divide_result = _divide_ranges(producer, trip_count, [axis])
            if divide_result.retiled_info is None:
                continue
            retiled_infos[name] = divide_result.retiled_info
            contracted_names.add(name)

            if owns_advance:
                producer_info = producer.loop_info
                producer_level = level_index(producer)
                assert producer_level is not None
                reads = [
                    dep
                    for dep in producer.get_read_writes().reads
                    if isinstance(dep, MemoryDep)
                ]
                if len(reads) != 1:
                    continue
                squeezed = copy.deepcopy(producer_info.squeezed_advance_per_read)
                if not squeezed:
                    squeezed = [[[] for _ in producer_info.loop_count] for _ in reads]
                squeezed[0][producer_level] = [(source_step, sympy.Integer(1))]
                producer.loop_info = dataclasses.replace(
                    producer_info,
                    squeezed_advance_per_read=squeezed,
                )

        if not retiled_infos:
            continue

        # The advancing consumer now reads iteration zero of a scratch tile;
        # its old per-level advance belongs to the identity's source instead.
        selected_names = set(retiled_infos)
        for consumer in group_ops:
            consumer_level = level_index(consumer)
            consumer_info = getattr(consumer, "loop_info", None)
            if consumer_level is None or consumer_info is None:
                continue
            reads = [
                dep
                for dep in consumer.get_read_writes().reads
                if isinstance(dep, MemoryDep)
            ]
            tiled = copy.deepcopy(consumer_info.tiled_dims_per_read)
            squeezed = copy.deepcopy(consumer_info.squeezed_advance_per_read)
            metadata_changed = False
            for dep_idx, dep in enumerate(reads):
                if dep.name not in selected_names or dep.index.coeff(loop_var) == 0:
                    continue
                if dep_idx < len(tiled) and consumer_level < len(tiled[dep_idx]):
                    tiled[dep_idx][consumer_level] = []
                if dep_idx < len(squeezed) and consumer_level < len(squeezed[dep_idx]):
                    squeezed[dep_idx][consumer_level] = []
                metadata_changed = True
            if metadata_changed:
                consumer.loop_info = dataclasses.replace(
                    consumer_info,
                    tiled_dims_per_read=tiled,
                    squeezed_advance_per_read=squeezed,
                )

        _patch_retiled_load_indexes(
            group_id,
            group_ops,
            retiled_infos,
            operations,
        )

    # Preserve the direct source form for the existing post-layout proof.
    # Until layout selection has proved that the source's device encoding and
    # per-core ownership are compatible, the contracted copy remains the
    # authoritative fallback.  Attaching these records only after every
    # retile rewrite is important: replace_computed_buffer_body deliberately
    # drops a saved record whenever it changes a consumer body.
    for copy_name in direct_read_roots:
        copy_op = graph.try_get_buffer(copy_name)
        identity = identities.get(copy_name)
        if not isinstance(copy_op, ir.ComputedBuffer) or identity is None:
            continue
        source_name, source_index, identity_indices = identity
        source_base = sympy.simplify(
            source_index.subs({index: sympy.S.Zero for index in identity_indices})
        )
        source_strides = [
            sympy.simplify(source_index.coeff(index)) for index in identity_indices
        ]
        residual = sympy.simplify(
            source_index
            - source_base
            - sum(
                (
                    stride * index
                    for stride, index in zip(source_strides, identity_indices)
                ),
                sympy.S.Zero,
            )
        )
        if residual != 0 or source_base != 0:
            continue

        readers = []
        for op in operations:
            if not isinstance(op, ir.ComputedBuffer):
                continue
            matching_reads = [
                dep
                for dep in op.get_read_writes().reads
                if isinstance(dep, MemoryDep) and dep.name == copy_name
            ]
            if matching_reads:
                readers.append((op, matching_reads))
        candidate_readers = [
            reader for reader in readers if reader[0].get_name() not in contracted_names
        ]
        if len(candidate_readers) != 1 or len(candidate_readers[0][1]) != 1:
            continue
        consumer, (copy_dep,) = candidate_readers[0]
        # _NameSwapHandler intentionally discards the compact copy's constant
        # offset.  A nonzero one would need coordinate decomposition rather
        # than stride substitution, so retain the staging fallback for it.
        copy_offset = sympy.simplify(
            copy_dep.index.subs(
                {symbol: sympy.S.Zero for symbol in copy_dep.index.free_symbols}
            )
        )
        if copy_offset != 0:
            continue

        copy_info = getattr(copy_op, "loop_info", None)
        copy_reads = [
            dep for dep in copy_op.get_read_writes().reads if isinstance(dep, MemoryDep)
        ]
        if copy_info is None or len(copy_reads) != 1:
            continue
        full_strides = list(copy_op.layout.stride)
        name_map = {copy_name: (source_name, full_strides, source_strides)}
        loop_var_zeros = {
            symbol: sympy.S.Zero for symbol in _splice_loop_vars(consumer)
        }
        consumer_inner = consumer.data.inner_fn

        def direct_inner_fn(
            *args,
            _inner=consumer_inner,
            _map=name_map,
            _source_name=source_name,
            _loop_var_zeros=loop_var_zeros,
        ):
            with V.set_ops_handler(
                _LoopVarRebaseHandler(V.ops, _source_name, _loop_var_zeros)
            ):
                with V.set_ops_handler(_NameSwapHandler(V.ops, _map)):
                    return _inner(*args)

        direct_tiled = copy_info.tiled_dims_per_read[0]
        direct_squeezed = (
            copy_info.squeezed_advance_per_read[0]
            if copy_info.squeezed_advance_per_read
            else [[] for _ in copy_info.loop_count]
        )
        consumer._read_copy_elision_record = ReadCopyElisionRecord(  # type: ignore[attr-defined]
            consumer_name=consumer.get_name(),
            copy_name=copy_name,
            source_name=source_name,
            direct_inner_fn=direct_inner_fn,
            direct_tiled_dims_per_level=tuple(
                tuple(tuple(pair) for pair in level) for level in direct_tiled
            ),
            direct_squeezed_advance_per_level=tuple(
                tuple(tuple(pair) for pair in level) for level in direct_squeezed
            ),
        )


def splice_while_loops(graph) -> None:
    """CustomPreSchedulingPasses entry point: splice every for_each_tile WhileLoop.

    Runs to a fixed point (handles nested for_each_tile, whose inner
    WhileLoop only appears after the outer one's body has been spliced in).
    Directly constructs and stamps CoarseTileInfo per accepted group from
    ground truth (trip count, loop var, tile_dim_marker resolution, carry
    roles) via _stamp_direct_loop_info -- never hands the group to
    coarse_tile_pre_stickify for re-inference, since coarse_tile_pre_stickify's
    index-coefficient inference runs once per nesting level, blind to prior
    runs, and a level's committed metadata can go stale once a later-spliced
    nested level renames or rewires the buffers it depended on (see
    _stamp_direct_loop_info's own docstring for the full rationale).

    Splicing itself runs outermost-first for a nested for_each_tile: the
    outer WhileLoop is visible in graph.operations immediately, while an
    inner WhileLoop only becomes visible once the outer splice flattens its
    body in on a later iteration of the `while True:` loop below. But ALL
    stamping is deferred to a single final phase, run only after every level
    has been spliced -- i.e. after this function's own `while True:` driver
    has exited. This is necessary, not merely convenient: an op materialized
    only by an INNER level's splice (e.g. an accumulator-add whose write
    buffer is a fresh per-outer-trip allocation) does not exist yet at the
    time an outer level's splice iteration finishes, so stamping outer-first
    per-iteration (the old structure) would call _stamp_direct_loop_info for
    the outer level before that op exists -- it can never receive the outer
    level's tiling contribution, silently corrupting its device-address
    advance. Deferring every _stamp_direct_loop_info call to after the last
    splice guarantees every op that will ever exist for this compile is
    already present before any stamping happens.

    Per-level ops are recorded by NAME (`op.get_name()`, via
    `_recordable_op_names`), never by object reference:
    `_consume_tile_dim_markers`, invoked here on a LATER splice iteration
    for a different, still-nested level, can replace a ComputedBuffer
    object in `graph.operations` via `pass_utils.replace_computed_buffer_body`
    (see `_inline_marker_into_consumer` above), which mints a brand-new
    object at the same list index under the SAME name. An object reference
    collected on an earlier splice iteration would silently go stale by the
    time the stamp phase runs; the name stays stable across such a rebuild
    (the same convention `_consume_tile_dim_markers`'s own docstring
    documents for its marker map). `_recordable_op_names` additionally
    recurses into any still-nested (not yet accepted this iteration)
    ir.WhileLoop found in group_ops, recording its body_subgraph's own
    (already fixed, pre-splice) op names instead of the WhileLoop's own --
    otherwise an op materialized only once THAT WhileLoop is itself spliced
    on a later iteration (e.g. the accumulator-add) would never be recorded
    for this (outer) level at all, since it does not exist as a distinct
    object yet. See `_recordable_op_names`'s own docstring for why those
    names are already fixed and predictable ahead of that later splice.

    Some operations have no predictable pre-splice name: a nested level's
    splice can create carry snapshots, and a marker excluded from an
    ancestor's prospective names can survive as STAR_DEP_KEPT. Each nested
    WhileLoop therefore carries the indices of its accepted ancestors. When
    that loop is later spliced, its final recordable names are appended to
    every ancestor's pending name list. This makes those late-created ops
    members of the complete loop nest instead of a sibling group containing
    only their immediate level. A pre-loop carry copy that the splice
    inserts before the loop (carry ownership, #4838) is appended to the
    ancestors only: it runs once per enclosing trip, outside its own loop.

    Stamping itself still proceeds level-0-first (outermost first) within
    the single final phase: level 0's call stamps first (existing=None,
    loop_group_id=(0,)), and a strictly-inner level's later call resolves
    the SAME live object again by name and APPENDS -- e.g. level 1 onto
    level 0 yields loop_group_id=(0, 1). See the `existing is not None`
    branch inside _stamp_direct_loop_info for why append, not prepend, is
    required here.
    """
    from torch._inductor import ir

    from torch_spyre._inductor.wsr.coarse_tile import _rebase_point_splice_reads
    from torch_spyre._inductor.wsr.while_loop_bridge import (
        carry_bindings_for,
        splice_while_loop,
    )

    if logger.isEnabledFor(logging.INFO):
        with timing_recorder.stage("stage:splice_while_loops:log_before"):
            logger.info(
                "BEFORE SPLICE_WHILE_LOOPS\n%s", format_operations(graph.operations)
            )

    group_idx = 0
    pending_levels: list[tuple[sympy.Symbol, sympy.Expr, int, list[str]]] = []

    while True:
        while_ops = [op for op in graph.operations if isinstance(op, ir.WhileLoop)]
        if not while_ops:
            break

        progressed = False
        for while_op in while_ops:
            result = try_prove_for_each_tile(while_op)
            if not result.accepted:
                continue  # leave untouched; falls through to upstream's default path

            loop_var = _body_loop_var(while_op)
            if loop_var is None:
                continue  # body shape doesn't match; leave untouched

            # Operations synthesized while splicing this loop (notably a
            # carry snapshot and STAR_DEP_KEPT tile markers) did not exist
            # when any enclosing level recorded its prospective body names.
            # Remember the actual ancestor chain on each still-nested
            # WhileLoop so these late-created operations can be added to all
            # enclosing levels before the final stamping phase.
            ancestor_level_indices = tuple(
                getattr(while_op, "_for_each_tile_ancestor_level_indices", ())
            )

            carries = carry_bindings_for(
                while_op,
                _stacking_carry_indices(while_op, loop_var, result.trip_count),
            )
            names_before_splice = {op.get_name() for op in graph.operations}
            group_ops = splice_while_loop(
                graph,
                while_op,
                carries,
                trip_count=result.trip_count,
            )
            # The splice can also add an op OUTSIDE this loop's body: a
            # carry's pre-loop ownership copy. It runs once per trip of every
            # enclosing level, never per trip of this one, so it joins the
            # ancestors' names below but not this level's.
            group_names = {op.get_name() for op in group_ops}
            pre_loop_names = [
                op.get_name()
                for op in graph.operations
                if op.get_name() not in names_before_splice
                and op.get_name() not in group_names
            ]

            _consume_tile_dim_markers(group_ops, graph.operations)

            recordable_names = _recordable_op_names(group_ops)
            for ancestor_idx in ancestor_level_indices:
                ancestor_names = pending_levels[ancestor_idx][3]
                ancestor_name_set = set(ancestor_names)
                for name in (*pre_loop_names, *recordable_names):
                    if name not in ancestor_name_set:
                        ancestor_names.append(name)
                        ancestor_name_set.add(name)

            for op in group_ops:
                if isinstance(op, ir.WhileLoop):
                    op._for_each_tile_ancestor_level_indices = (
                        *ancestor_level_indices,
                        group_idx,
                    )

            pending_levels.append(
                (
                    loop_var,
                    result.trip_count,
                    group_idx,
                    recordable_names,
                )
            )

            group_idx += 1
            progressed = True

        if not progressed:
            # Every remaining WhileLoop was declined; stop rather than loop forever.
            break

    # Stamp phase: every level has been spliced, so every op that will ever
    # exist for this compile is present in graph.operations now. Resolve each
    # level's recorded names back to LIVE ir.Operation objects -- never the
    # objects recorded during the splice phase above, which may have gone
    # stale (see this function's own docstring) -- and stamp in level order
    # (group_idx ascending, i.e. outermost first).
    from torch_spyre._inductor.errors import Unsupported

    name_to_op = {op.get_name(): op for op in graph.operations}
    for loop_var, trip_count, level_group_idx, op_names in pending_levels:
        missing = [name for name in op_names if name not in name_to_op]
        if missing:
            raise Unsupported(
                f"for_each_tile level {level_group_idx} recorded op(s) "
                f"{missing} that no longer exist in graph.operations at "
                "stamp time"
            )
        resolved_ops = [name_to_op[name] for name in op_names]
        _stamp_direct_loop_info(resolved_ops, loop_var, trip_count, level_group_idx)

    # ``WhileLoop.create`` may have compacted a non-contiguous sliced operand
    # into a full ``[trip_count, tile, ...]`` temporary.  It was outside the
    # loop before splicing but is now an ordinary member of every counted
    # iteration.  Stream one tile through that temporary instead of copying
    # the complete K/V cache on each trip.
    _contract_exact_stride_input_materializations(graph, pending_levels)

    # Read-side counterpart of the stamping above: _stamp_direct_loop_info
    # records a point-shaped splice read's per-trip step in
    # squeezed_advance_per_read (mirroring coarse_tile.py's own
    # _point_splice_advance_for_dep -- see that function's docstring), but
    # leaves the read's index itself unrebased (e.g. paged attention's
    # in-body page index, dep.index == 32*u0). main reaches
    # _rebase_point_splice_reads for this same shape via
    # coarse_tile_pre_stickify (_coarse_tile_common calls it unconditionally
    # at the end of planning), which this branch never calls for
    # WhileLoop-splice groups -- see this function's own docstring. Without
    # this call the raw indirect symbol survives into
    # propagate_spyre_tensor_layouts, which rejects it (not an
    # iteration_space key). Run once, after every level's stamping has been
    # merged in above, so squeezed_advance_per_read reflects each op's final,
    # fully-merged state rather than a partially-stamped intermediate.
    _rebase_point_splice_reads(graph.operations)

    if pending_levels:
        # Splicing/unrolling leaves dead carry-snapshot and marker scaffolding
        # behind. The pipeline's deadcode_elimination pass runs before this one,
        # so nothing downstream would otherwise clean this up until scheduling --
        # run it here too so the AFTER dump below reflects the IR later passes
        # actually see, not transient splice debris. Idempotent: a later
        # deadcode_elimination call finds nothing new to remove.
        deadcode_elimination(graph)

    if logger.isEnabledFor(logging.INFO):
        if pending_levels:
            with timing_recorder.stage("stage:splice_while_loops:log_after"):
                logger.info(
                    "AFTER SPLICE_WHILE_LOOPS\n%s", format_operations(graph.operations)
                )
        else:
            logger.info("SPLICE_WHILE_LOOPS: no while_loops spliced")
