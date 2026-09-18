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

"""for_each_tile-specific WhileLoop prover.

Recognizes the exact WhileLoop shape torch-spyre#4136's for_each_tile
frontend (via decompose_scan_to_while_loop) produces, derives a provable
trip count, and -- once accepted -- hands off to the generic bridge
(while_loop_bridge.py) plus DimHint synthesis to actually splice and
coarse-tile the body. This module owns every for_each_tile-specific
assumption; while_loop_bridge.py knows none of them.

Real cond-graph shape (confirmed empirically against a live compiled graph
for both split_m_fn (map mode) and split_k_fn (carry mode) -- see the task-4
report for the full investigation): decompose_scan_to_while_loop always
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
mandates for ComputedBuffer.inner_fn elsewhere in this codebase, applied
here for read-only shape recognition rather than mutation.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Any

import sympy
import torch

from torch._inductor.ops_handler import DefaultHandler, WrapperHandler
from torch._inductor.virtualized import V

if TYPE_CHECKING:
    from torch._inductor import ir
    from torch._inductor.dependencies import Dep


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
        self.index_exprs: list[Any] = []
        self.compare_ops: list[str] = []

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if name == "load":
            self.loads.append((args[0], args[1]))
            return f"__load_{len(self.loads) - 1}__"
        if name == "constant":
            self.constants.append(args[0])
            return f"__constant_{len(self.constants) - 1}__"
        if name == "index_expr":
            self.index_exprs.append(args[0])
            return args[0]
        if name in ("lt", "le", "gt", "ge", "eq", "ne"):
            self.compare_ops.append(name)
            return f"__cmp_{name}__"
        # Anything else means this cond graph does not match the known
        # for_each_tile shape (a single load-vs-constant comparison); record
        # the op name so the caller can decline with a useful reason.
        self.compare_ops.append(f"unexpected:{name}")
        return f"__unexpected_{name}__"


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
    if len(recorder.loads) != 1:
        return None

    (loaded_name, loaded_index) = recorder.loads[0]
    if loaded_name != first_placeholder:
        return None
    if loaded_index != 0:
        return None

    bounds = [*recorder.constants, *recorder.index_exprs]
    if len(bounds) != 1:
        return None
    bound = bounds[0]
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

    _synthesize_dim_hints_for_group's DimHint.loop_var must be exactly this
    symbol: coarse_tile.py's _loop_var_to_ranges_pos/
    _loop_var_to_reduction_ranges_pos resolve loop_var by searching for it
    inside an op's own index expressions (via op_out_coords/
    reduction_loop_vars), so a freshly-minted, disconnected sympy.Symbol
    would never resolve and every op would land with empty tiled dims.

    Returns None if the body subgraph does not have this exact shape (no
    DynamicScalar reading the first placeholder), signaling the caller to
    decline rather than synthesize a hint nothing will ever match.
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


_next_synthetic_hint_id_start = 1 << 30  # reserved range, well above real hint scopes

_MARKER_MAPS: dict[int, dict[tuple[str, "Dep"], int]] = {}
"""Per-compile marker-map registry, keyed by id(operations).

NOT safe to let outlive one compile: CPython aggressively reuses a freed
list's id, so a stale entry left behind by a prior compile can collide
with -- and be silently mistaken for -- a live compile's own entry
sharing the same buffer-name/dep-shape (confirmed empirically: running
the same for_each_tile fixture twice in one process produces
byte-identical (op.get_name(), dep) keys across both compiles).
clear_marker_maps() must be called once per compile, before
_consume_tile_dim_markers runs, to guarantee this never happens --
passes.py's per-compile pipeline entry point does this, alongside the
analogous reset_provenance_warnings() call, for the same "each compile
starts from a clean slate" reason.
"""


def clear_marker_maps() -> None:
    """Discard every entry in the module-level _MARKER_MAPS registry.

    Must be called exactly once per compile, before _consume_tile_dim_markers
    runs for that compile (passes.py's per-compile pipeline __call__ does
    this, right alongside reset_provenance_warnings(), which exists for the
    identical "each compile starts fresh" reason). Without this,
    _MARKER_MAPS leaks for the process's entire lifetime (nothing else ever
    deletes an entry), and -- more seriously than the leak itself --
    id(operations) can be reused by CPython for an unrelated later compile's
    operations list, letting that later compile's lookup_marker_dim call
    silently resolve against a dead compile's stale entry instead of
    raising or returning None. Confirmed empirically: two successive
    compiles of the same fixture in one process produced identical
    (op.get_name(), dep) keys; it happened to be harmless there only because
    both entries stored the same dim, which is not a general guarantee.
    """
    _MARKER_MAPS.clear()


def _synthesize_dim_hints_for_group(
    group_ops: list["ir.Operation"],
    loop_var: sympy.Symbol,
    hint_id: int,
    trip_count: sympy.Expr,
) -> None:
    """Stamp one synthesized DimHint per op in group_ops for this while-loop level.

    loop_var is this level's own induction variable -- there is exactly one
    per nesting level, unlike a user spyre_hint() scope which can cover many
    ops arbitrarily.

    ``is_reduction`` here is ADVISORY ONLY, unlike on an ordinary
    ``spyre_hint()`` DimHint where it selects the lookup channel. One
    synthesized hint covers the whole level, so it cannot say, per op,
    whether the level lands on an output dim or a reduction dim OF THAT OP --
    and the two fixtures disagree for structurally identical ``aten.mm``
    bodies: ``split_m_fn``'s matmul reduces over K but the loop tiles M (an
    output dim), while ``split_k_fn``'s reduces over K and the loop tiles K
    itself. ``coarse_tile.py``'s ``_hint_ranges_pos`` therefore resolves the
    channel per op from where ``loop_var`` actually appears, and ignores this
    field for a WhileLoop-splice hint (identified by ``loop_var_range`` being
    non-None). It is still populated from ``reduction_type`` so the hint
    reads sensibly in logs and so any future consumer that keys off it sees
    the op's own reduction-ness rather than a hard-coded False.

    Which dim is tiled, and whether an op is tiled at all, likewise come from
    that resolution: an op that never mentions loop_var simply gets no tiled
    dim recorded for this hint_id, which is the correct outcome for ops that
    are loop-invariant at this level (e.g. an INVARIANT operand's own read).

    loop_var must be the real per-iteration index symbol already present in
    the spliced body's own index expressions (see _body_loop_var) -- not a
    freshly-minted, disconnected sympy.Symbol, for the same reason.
    """
    from torch_spyre._inductor.propagate_hints import DimHint

    for op in group_ops:
        if not hasattr(op, "data"):
            continue
        if _marker_resolution(op) is MarkerResolution.INLINE_ERASED:
            # An inline-erased tile_dim_marker op is gone from operations
            # by the time this runs -- if group_ops still holds a stale
            # reference to it (should not happen post-consumption, but
            # guard defensively), it is not itself a for_each_tile tile
            # read: it IS the marker, already fused into its consumer's
            # inner_fn, with nothing left to hint.
            #
            # A STAR_DEP_KEPT marker, by contrast, is still a live member
            # of group_ops and its own upstream read still carries a real
            # per-iteration offset -- the next nesting level up needs a
            # synthesized hint for it to resolve provenance through
            # lookup_marker_dim/_hint_ranges_pos (issue #4581: the old
            # guard, `_marker_dim(op) is not None`, skipped BOTH marker
            # kinds here, silently starving this exact case of a hint).
            continue
        existing = list(getattr(op, "dim_hints", []) or [])
        is_reduction = getattr(op.data, "reduction_type", None) is not None
        hint = DimHint(
            dim_names=[f"_while_loop_{hint_id}"],
            split_count=1,  # per-level count; coarse_tile derives real counts from `levels`
            loop_var=loop_var,
            is_reduction=is_reduction,
            hint_id=hint_id,
            loop_var_range=trip_count,
        )
        op.dim_hints = [*existing, hint]


def _stacking_carry_indices(
    while_op: "ir.WhileLoop", loop_var: sympy.Symbol
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
       the fold arithmetic relies on.

    Requiring all three keeps every other carry shape -- accumulator,
    read-only pass-through leaf, scalar step counter -- on the pre-existing
    path untouched.
    """
    from torch._inductor import ir
    from torch._inductor.ir import MutableBox

    body_graph = while_op.body_subgraph.graph
    placeholder_names = list(body_graph.graph_inputs.keys())
    body_outputs = body_graph.graph_outputs

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
        if loop_var not in offset.free_symbols:
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
    never carries it). Confirmed empirically: checking ``op.data`` here
    always misses, even for a genuine marker op.
    """
    return getattr(op, "tile_marker_dim", None)


class MarkerResolution(enum.Enum):
    """How _consume_tile_dim_markers resolved one tile_dim_marker op.

    INLINE_ERASED: every consumer of the marker is a ComputedBuffer (or was
    inlined into one) -- the marker's transform was fused directly into
    each consumer's inner_fn and the marker op was removed from both
    operations and group_ops.

    STAR_DEP_KEPT: at least one consumer reaches the marker via a StarDep,
    not an ordinary MemoryDep -- the marker cannot be fused away and stays
    live as a real, addressable buffer (see _consume_tile_dim_markers's own
    comment on why removing it from operations would break
    coarse_tile.py's _validate_contiguous gapless-block invariant). Any
    ComputedBuffer consumer the same marker also has is still inlined; it
    is the StarDep one that keeps the buffer alive.
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
    trace built the marker in the first place, crashing or silently
    reusing the wrong graph node when spliced into a different consumer's
    live retrace -- confirmed empirically: ``LightTracer.create_arg``
    raises ``NotImplementedError`` on the leaked ``OpsValue``), extract the
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
    wrong-answer bug confirmed empirically against test_map_mode_split_m
    and test_carry_mode_online_softmax's real-device matmul consumers (see
    _consume_tile_dim_markers's own docstring and this module's task-5
    report for the full trace).

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

    def __init__(self, inner, marker_name: str, marker_op: "ir.Operation"):
        super().__init__(inner)
        self._marker_name = marker_name
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
            # distinct coordinates into one (confirmed empirically: this
            # produced a wrong composed index, e.g. 129*d2 instead of the
            # correct 128*d1 + d2, for test_carry_mode_online_softmax's
            # transposed k_tile read, since its coordinate permutation's
            # target set overlaps its source set: {d0: d1, d1: d2}).
            composed = self._marker_read_index.subs(subs, simultaneous=True)
            return super().load(self._marker_input_name, composed)
        return super().load(name, index)


def _inline_marker_into_consumer(
    consumer_op: "ir.Operation",
    marker_op: "ir.Operation",
    operations: list["ir.Operation"],
) -> "ir.Operation":
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
    """
    from torch._inductor.virtualized import V

    from torch_spyre._inductor.pass_utils import (
        _invalidate_body_caches,
        replace_computed_buffer_body,
    )

    marker_name = marker_op.get_name()

    orig_inner = consumer_op.data.inner_fn

    def new_inner_fn(*args, _orig_inner=orig_inner):
        with V.set_ops_handler(_InlineMarkerHandler(V.ops, marker_name, marker_op)):
            return _orig_inner(*args)

    object.__setattr__(consumer_op.data, "inner_fn", new_inner_fn)
    _invalidate_body_caches(consumer_op.data)

    return replace_computed_buffer_body(
        consumer_op,
        consumer_op.data,
        operations,
        pass_name="_consume_tile_dim_markers",
        reason=f"inline erased tile_dim_marker {marker_name!r} body",
    )


def _consume_tile_dim_markers(
    group_ops: list["ir.Operation"],
    operations: list["ir.Operation"],
) -> dict[tuple[str, "Dep"], int]:
    """Find every tile_dim_marker-tagged op in group_ops, erase it, map its dim.

    For each marker op (an op whose realized ComputedBuffer carries
    tile_marker_dim -- see lowering.py's lower_tile_dim_marker): find its
    consuming uses among group_ops, record (op.get_name(), dep) -> dim in
    the returned map for each of them, then erase the marker and remove the
    marker op from `operations`.

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
      straight through unchanged. Confirmed empirically (see
      ``lookup_marker_dim``'s own docstring and this module's task-5
      report): a plain rename produced silently wrong device-side numerics
      on ``test_map_mode_split_m``/``test_carry_mode_online_softmax``,
      because every loop trip ended up reading the identical (tile-0-only)
      slice of the underlying tensor instead of advancing through it.
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

    A marker can have SEVERAL consuming uses, and each is erased on its own:
    the inline branch composes the marker's own per-trip transform into one
    consumer's own load index, a rewrite with nothing shared between
    consumers, and the StarDep branch repoints one consumer's own direct
    reference. Paged attention is the shape that needs this -- the tiled
    page-index row is sliced once and handed to an ``index_select`` per
    cache, so K's gather and V's gather are two separate ops both reading
    that one marker. The only thing several consumers changes is WHEN the
    marker itself may be dropped: not until every one of them has taken the
    inline branch (a single StarDep consumer still needs it materialized),
    hence the all-inlined bookkeeping below rather than a per-consumer
    decision.

    Zero consuming uses (of either shape) is an unrecognized shape and
    raises -- this pass runs on a freshly spliced body where a marker's
    output should be read by exactly the op(s) for_each_tile's frontend
    wrote to read that tile; a marker read by nothing at all means an
    assumption this module owns (see the module docstring) no longer holds
    and a silent guess would be worse than a loud failure.

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
    ``_hint_ranges_pos``. Confirmed empirically
    (test_carry_mode_online_softmax): the marker map's ``id()``-keyed entry
    for the K-tile-marker's consumer went stale exactly this way once a
    read-copy for its *other* read (``arg0_1``/Q, unrelated to the marker)
    was inserted, silently orphaning an otherwise-correct, otherwise-still-
    matching map entry. ``op.get_name()`` (the buffer name) is what stays
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

        if not consumers:
            raise AssertionError(
                f"tile_dim_marker op {marker_name!r} has no consuming read "
                "within its spliced body; expected at least one. This is an "
                "unrecognized for_each_tile shape -- _consume_tile_dim_markers "
                "only knows how to erase a marker whose output some downstream "
                "op actually reads."
            )

        # ComputedBuffer has no plain `.inputs` list attribute (that
        # attribute belongs to the InputsKernel family -- FallbackKernel,
        # ConcatKernel, etc). A ComputedBuffer's own upstream reads instead
        # come from its inner_fn, surfaced via get_read_writes().reads --
        # confirmed empirically against a live tile_dim_marker op, which has
        # exactly one MemoryDep read (the tile it marks).
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

        # Grouped by consumer op, not iterated read by read: one call to
        # _inline_marker_into_consumer erases EVERY load of the marker in that
        # consumer's body at once (_InlineMarkerHandler.load intercepts them
        # all), and it returns a NEW object -- so a second call for the same
        # consumer's second read would rewrite an object that is no longer the
        # one in `operations`. Keyed by name for the reason the docstring above
        # gives for the returned map: ir.Operation is unhashable, and a name
        # stays stable across the reconstruction the inline performs.
        consumers_by_op: dict[str, tuple[ir.Operation, list[Dep]]] = {}
        for candidate, dep in consumers:
            _, deps = consumers_by_op.setdefault(candidate.get_name(), (candidate, []))
            deps.append(dep)

        # The marker may only be dropped once every consumer has been inlined;
        # one StarDep consumer keeps it materialized for all of them.
        all_inlined = True

        for consumer_op, consumer_deps in consumers_by_op.values():
            if hasattr(consumer_op, "data"):
                new_consumer = _inline_marker_into_consumer(
                    consumer_op, marker_op, operations
                )
                # _inline_marker_into_consumer swaps `operations[op_idx]` in
                # place but returns a new object whose reads no longer
                # include `consumer_deps` at all -- the marker's own
                # inner_fn (its genuine per-iteration coordinate transform,
                # e.g. an extra `+ 24*u0` advance term) is now composed
                # directly into the consumer's read of `marker_input_name`,
                # replacing the old, marker-relative read.
                # lookup_marker_dim looks up entries keyed by the CURRENT
                # read it finds on `op` at lookup time, so the map must be
                # keyed by those post-inline deps (the ones now naming
                # `marker_input_name`, with the marker's own index composed
                # in), not by the stale pre-inline `consumer_deps` (which
                # named the marker itself and will never appear among
                # new_consumer's reads again).
                new_deps = [
                    d
                    for d in new_consumer.get_read_writes().reads
                    if isinstance(d, MemoryDep) and d.name == marker_input_name
                ]
                if not new_deps:
                    raise AssertionError(
                        f"consumer {consumer_op.get_name()!r} has no "
                        "post-inline MemoryDep read named "
                        f"{marker_input_name!r}; expected the inlined "
                        f"read(s) that used to go through erased marker "
                        f"{marker_name!r}. Counted, not required to be one: "
                        "a consumer reading the same tile at two different "
                        "indices legitimately holds one dep per index, and "
                        "the inline composes the marker's transform into "
                        "each of them."
                    )
            else:
                # StarDep-shaped consumer (ExternKernelOut/FallbackKernel/
                # ConcatKernel/... -- including a nested ir.WhileLoop, whose
                # own .carried_inputs/.additional_inputs are the read shape
                # _substitute_direct_input_refs's docstring calls its
                # "fourth read shape"): no inner_fn to wrap, so
                # redirect_computed_buffer_reads does not apply, and there
                # is no load index to compose the marker's own transform
                # into the way _inline_marker_into_consumer does for a
                # ComputedBuffer consumer above.
                #
                # The marker's own ComputedBuffer performs a genuine,
                # non-identity per-iteration coordinate transform (the
                # tile's slice/offset -- see lower_tile_dim_marker's
                # docstring), the same as for the ComputedBuffer-consumer
                # branch above. Pointing the consumer's reference at the
                # marker's own upstream input (marker_input_name, e.g.
                # arg0_1 -- what an earlier version of this branch did via
                # V.graph.try_get_buffer) discards that transform entirely:
                # every consumer read then sees the raw, untiled operand
                # with no per-iteration offset at all. Confirmed empirically
                # on a nested for_each_tile (map/map) on the real Spyre
                # device: the inner loop's captured outer-tile operand
                # silently stayed pinned to outer trip 0's slice on every
                # trip, corrupting every outer iteration after the first
                # (~48% wrong elements) while remaining invisible on CPU
                # eager/CPU Inductor, since neither exercises Spyre-specific
                # codegen for a StarDep-shaped nested-WhileLoop marker
                # consumer.
                #
                # The correct erasure-equivalent for this read shape is to
                # keep the marker's ComputedBuffer materialized (never
                # remove it from group_ops/operations) and redirect the
                # consumer's reference to the marker ITSELF rather than to
                # its upstream input -- equivalent in effect to
                # _inline_marker_into_consumer's per-load composition, just
                # realized as a standalone buffer instead of fused into the
                # consumer's own body, since a StarDep-shaped consumer has
                # no body to fuse into. Only a stale-by-identity, same-name
                # reference (confirmed empirically: splice_while_loop's own
                # upstream passes can leave a consumer's direct object
                # reference pointing at an object that predates the marker's
                # final reconstruction, even though it already names the
                # marker correctly) needs patching at all --
                # _substitute_direct_input_refs's name-based resolve() is a
                # no-op for any reference that already points at marker_op
                # by identity, and safely repoints any reference that
                # doesn't.
                _substitute_direct_input_refs([consumer_op], {marker_name: marker_op})
                new_consumer = consumer_op
                # No object reconstruction happened (unlike the
                # ComputedBuffer branch) -- consumer_op's own identity is
                # unchanged, and its deps still name marker_name (the marker
                # is not erased, so nothing renamed it) -- unlike the
                # erase-and-redirect path this replaced, there is no
                # post-substitution name change to re-derive new deps from.
                new_deps = consumer_deps
                all_inlined = False

            for new_dep in new_deps:
                marker_map[(new_consumer.get_name(), new_dep)] = dim
            if id(consumer_op) in group_op_ids:
                group_op_ids.discard(id(consumer_op))
                group_op_ids.add(id(new_consumer))
                group_ops[group_ops.index(consumer_op)] = new_consumer

        if all_inlined:
            # Only the ComputedBuffer/inline branch actually fuses the
            # marker's transform into the consumer and erases the marker;
            # the StarDep branch above deliberately keeps marker_op alive
            # in BOTH group_ops and operations (see its comment) -- it must
            # still codegen as a real, addressable buffer for the StarDep
            # consumer to read, and _validate_contiguous (coarse_tile.py)
            # requires every group's ops to occupy a gapless block of
            # `operations`, so removing it from `operations` alone while
            # keeping it out of `group_ops` (tried and reverted -- see git
            # history) breaks that contiguity check for any group whose
            # block the marker sits inside. Passes that must not treat a
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
        else:
            marker_op.tile_marker_resolution = MarkerResolution.STAR_DEP_KEPT

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
    numbered differently. Confirmed empirically: this exact bug produced
    silently wrong (not raising) numerics on test_map_mode_split_m and
    test_carry_mode_online_softmax's real-device matmul consumers, whose
    CPU-fixture-based unit-test counterparts never caught it because a CPU
    aten-fallback matmul is a StarDep consumer (see below), which never
    reaches this position-mapping code at all.

    So instead of trusting the map's stored int, re-derive the consumer's
    own position the same way the deleted _loop_var_pos_from_reads did, but
    scoped to exactly the one dep the marker map already identified as the
    tile read -- no CROSS-READ ambiguity/corroboration logic is needed the
    way that heuristic's cross-read guessing required, since the marker is
    ground truth about which read is the tile: find the read's own index
    variable `var` whose extent matches loop_var's per-trip advance
    (dep.index.coeff(loop_var) == dep.index.coeff(var) * dep.ranges[var]),
    then map `var` into op's own output coordinates
    (_loop_var_to_ranges_pos) or, if that misses and op is a Reduction,
    into op's own reduction vars (reduction_loop_vars.index).

    A narrower, WITHIN-ONE-DEP ambiguity the deleted heuristic also guarded
    against still applies here, and is NOT made moot by having a ground-
    truth marker: more than one var in dep.ranges can satisfy the same
    coefficient-coincidence equation on the SAME read (the heuristic's own
    docstring names the motivating shape -- a reduction dim whose extent
    numerically coincides with the tile size, e.g. flash-attention's
    online-softmax body where D == SOFTMAX_TILE_SIZE). The deleted
    heuristic resolved this via cross-read corroboration (trust a lone
    per-read candidate; require a second, independently-agreeing read
    before trusting a reduction-channel match when a read had more than
    one candidate). That corroboration mechanism doesn't carry over as-is
    (this function deliberately looks at only the one marker-identified
    dep, not every read), but the underlying risk -- picking an arbitrary
    one of several equally-plausible candidates -- is exactly what
    "markers are authoritative, raise on gap, no fallback heuristic"
    rules out. So: collect EVERY candidate var on the marker-identified
    dep (don't return on the first one found), and if more than one
    survives, raise the same actionable gap error _hint_ranges_pos raises
    elsewhere rather than silently guess. (This has not been observed to
    trigger against any test fixture in this repo, including
    online-softmax's own D == SOFTMAX_TILE_SIZE coincidence -- that
    coincidence lands on a read the marker map does NOT identify as the
    tile, so it never reaches this per-dep candidate collection at all --
    but the check must still exist so a future shape that does collide on
    the marker's own dep fails loudly instead of guessing.)

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
    ``_MARKER_MAPS`` registry. ``_MARKER_MAPS`` is keyed by
    ``id(operations)``, and CPython aggressively reuses a freed list's
    id; two unrelated compiles in the same process can (and, confirmed
    empirically, do) end up with byte-identical
    ``(op.get_name(), dep)`` keys whenever they share a buffer-naming/dep
    shape (e.g. two runs of the same for_each_tile fixture). Searching
    every map in the registry, as an earlier version of this function
    did, risks resolving a live compile's lookup against a DIFFERENT,
    unrelated compile's stale entry -- silently returning the wrong
    position whenever that stale entry happens to disagree (harmless only
    by accident when the two happen to agree, as they did for the
    same-fixture-twice repro that surfaced this). ``V.graph`` is the live
    ``GraphLowering`` for whichever compile is currently running this
    pass pipeline (already relied on elsewhere in this module, e.g.
    ``V.graph.try_get_buffer`` in ``_consume_tile_dim_markers``), so
    ``V.graph.operations`` is guaranteed to be the SAME list object
    ``_consume_tile_dim_markers`` was given for this exact compile.
    ``clear_marker_maps()`` (called once per compile from
    ``passes.py``'s pipeline entry point, alongside the analogous
    ``reset_provenance_warnings()``) additionally guarantees no entry
    from a past compile can outlive it even under id reuse.
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


def splice_while_loops(graph) -> None:
    """CustomPreSchedulingPasses entry point: splice every for_each_tile WhileLoop.

    Runs to a fixed point (handles nested for_each_tile, whose inner
    WhileLoop only appears after the outer one's body has been spliced in).
    Calls coarse_tile_pre_stickify immediately per accepted group -- before
    propagate_named_dims/assign_dim_hints ever run for this compile -- since
    those overwrite op.dim_hints from scratch and would otherwise silently
    clobber the synthesized hints this function just stamped.
    """
    from torch._inductor import ir

    from torch_spyre._inductor.wsr.coarse_tile import coarse_tile_pre_stickify
    from torch_spyre._inductor.wsr.while_loop_bridge import (
        carry_bindings_for,
        splice_while_loop,
    )

    hint_id = _next_synthetic_hint_id_start
    group_idx = 0

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

            carries = carry_bindings_for(
                while_op, _stacking_carry_indices(while_op, loop_var)
            )
            group_ops = splice_while_loop(
                graph, while_op, carries, trip_count=result.trip_count
            )

            _consume_tile_dim_markers(group_ops, graph.operations)

            _synthesize_dim_hints_for_group(
                group_ops, loop_var, hint_id, result.trip_count
            )

            from torch_spyre._inductor.pass_utils import compute_max_size

            planning_count = sympy.Integer(compute_max_size(result.trip_count))
            levels = [(hint_id, planning_count)]
            coarse_tile_pre_stickify(
                graph,
                groups=[(group_ops, levels)],
                group_idx_offset=group_idx,
                runtime_loop_count=(
                    result.trip_count if result.trip_count.free_symbols else None
                ),
            )

            group_idx += 1
            hint_id += 1
            progressed = True

        if not progressed:
            # Every remaining WhileLoop was declined; stop rather than loop forever.
            break
