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
from typing import TYPE_CHECKING, Any

import sympy
import torch

from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.virtualized import V

if TYPE_CHECKING:
    from torch._inductor import ir


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

            _synthesize_dim_hints_for_group(
                group_ops, loop_var, hint_id, result.trip_count
            )

            levels = [(hint_id, result.trip_count)]
            coarse_tile_pre_stickify(
                graph, groups=[(group_ops, levels)], group_idx_offset=group_idx
            )

            group_idx += 1
            hint_id += 1
            progressed = True

        if not progressed:
            # Every remaining WhileLoop was declined; stop rather than loop forever.
            break
