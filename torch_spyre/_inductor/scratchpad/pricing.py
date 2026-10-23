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


"""The priced objective: the cost model's prediction over each node's
symbolic work split and residency, built around the solve the same way for
every solver, and its dump for the summarize-sdsc skill."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import NamedTuple, Optional, TYPE_CHECKING

import sympy
from torch._inductor.ir import (
    ComputedBuffer,
)
from torch._inductor.graph import GraphLowering

from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.scratchpad.plan_solver import (
    CoreDivision,
    TileSpec,
    TileAxis,
    division_symbol,
)
from torch_spyre._inductor.scratchpad.utils import (
    mem_usage_by_buf,
)

from torch_spyre._inductor import config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.cost_model import CostParams
from torch_spyre._inductor.scratchpad.buffer_graph import (
    ResourceNode,
    SpyreOptimizationGraph,
)


if TYPE_CHECKING:
    from torch_spyre._inductor.scratchpad.allocator import SolvedPlan

logger = get_inductor_logger("scratchpad.pricing")

_COST_PARAMS = CostParams(
    # we need a expression of both compute, mem_t
    # whereas the default gives max(compute, mem_t)
    # which optimizes compute only when there's a matmul
    overlap_gamma=0.46,
    use_bundled_cost_model=False,
    # The coarse underfill and LX-spill derates: zero exponents drop the tile height
    # from both (``rpc**0 == 1``) and the 1.0 cap pins the underfill there, so each
    # derate is exactly 1. CP-SAT cannot lower their fractional powers of a symbolic
    # tile height at all (the whole cost objective would fall back to the
    # lexicographic one). The annealer can, but is misled by them: the underfill
    # derate divides a whole bundle's memory term by the efficiency of its smallest
    # tile, which over-prices a tile several times over on real decoder blocks.
    coarse_underfill_exp=0.0,
    coarse_underfill_col_exp=0.0,
    coarse_underfill_cap=1.0,
    lx_spill_exp=0.0,
    mm_spill_ws_exp=0.0,
    # The loop-invariant re-read: ``read_bytes`` already charges a tiled matmul's
    # invariant operand once per iteration through its ``loop_factor``, and
    # ``_loop_reread_bytes`` charges the repeats again.
    loop_reread_scale=0.0,
)


class _PricedNode(NamedTuple):
    """What the cost-expression features read about one node: its cost-model
    symbols and its candidate divisions."""

    sym_core_divs: Mapping
    sym_tile_counts: Mapping
    core_divisions: Sequence[CoreDivision]

    @classmethod
    def of(cls, node: ResourceNode) -> "_PricedNode":
        return cls(node.sym_core_divs, node.sym_tile_counts, node.menu)


def cost_expr(
    graph: GraphLowering,
    solve_graph: SpyreOptimizationGraph,
) -> tuple[Optional[sympy.Expr], list]:
    """The priced objective of the open-mode solve over the graph's nodes
    and their candidates, and its per-bundle terms; ``None`` when
    it is withheld or cannot be built (the solver then runs its own
    lexicographic ladder)."""
    from torch_spyre._inductor.scratchpad.graph_builder import (
        GraphBuilder,
        residency_symbols,
    )

    nodes = solve_graph.nodes
    # Built once here, not per op inside the loop below: every op's residency
    # lookup is against this same whole-graph map.
    default_is_lx = residency_symbols(solve_graph)
    pricing_ops = GraphBuilder._pricing_operations(graph, solve_graph)
    pricing_by_name = {op.get_name(): op for op in pricing_ops}

    # Keyed by buffer name, which is what ``predict_by_bundle`` needs to match
    # features to the ops in each estimated bundle. ``mem_usage_by_buf`` keys
    # on ``op.name`` over ``graph.operations``, so every feature here belongs
    # to an op the grouping will see -- a feature whose buffer is absent from
    # ``graph.operations`` would silently drop out of the objective.
    mem_usage = mem_usage_by_buf(graph)
    feature_names = [
        output_name
        for output_name in mem_usage
        if isinstance(graph.get_buffer(output_name), ComputedBuffer)
        and output_name in nodes
        and output_name in pricing_by_name
    ]

    from torch_spyre._inductor.cost_model import predict_bundles

    # Logged, not asserted: dropping a buffer from the objective changes what
    # the solver optimizes without failing anything, so it has to be visible,
    # but it is not worth killing a plan over. Empty today.
    unscored = set(feature_names) - {
        getattr(op, "name", None) for op in graph.operations
    }
    if unscored:
        logger.debug(
            "cost objective omits %d buffer(s) absent from graph.operations: %s",
            len(unscored),
            sorted(unscored),
        )
    if config.auto_coarse_tiling:
        # The cost model is flat in both axes the tiling search moves along:
        # it has no term for tile size and none for cut count, so every
        # candidate tiling scores identically and the choice falls to
        # whichever optimum the multi-worker portfolio reaches first. Hand
        # the solver no cost expression at all and let its lexicographic
        # ladder rank residency, cuts, parallelism and division shape.
        logger.debug(
            "cost objective skipped: auto_coarse_tiling makes tile size and "
            "cut count decision axes the cost model cannot score"
        )
        return None, []

    priced = {name: _PricedNode.of(node) for name, node in nodes.items()}
    bundle_terms: list = []
    # Feature extraction runs inside the same fallback: a candidate tiling
    # can raise Unsupported (an axis with no resolvable loop variable) or
    # ValueError (candidates on an already coarse-tiled op), and either
    # costs the objective, not the compile. TypeError is in the set because
    # a cost-model branch over an undecided ``is_lx`` raises "cannot
    # determine truth value of Relational" (#4233).
    try:
        op_features = {
            output_name: extract_op_features(
                graph,
                output_name,
                priced,
                default_is_lx,
                op=pricing_by_name[output_name],
            )
            for output_name in feature_names
        }
        bundle_terms = predict_bundles(pricing_ops, op_features, params=_COST_PARAMS)
        cost_expr = sympy.sympify(sum(term for _, term in bundle_terms))
    except (ValueError, RuntimeError, TypeError) as e:
        logger.warning(
            "cost objective unavailable (%s: %s); the solver falls back to its "
            "own objective",
            type(e).__name__,
            e,
            exc_info=True,
        )
        if not config._cpsat_warn_on_cost_expr:
            raise
        return None, []

    return cost_expr, bundle_terms


def dump_cost_expr(
    graph: GraphLowering,
    plan: SolvedPlan,
    solve_stats: dict,
    cost_expr,
    bundle_terms,
    capacity: int,
) -> None:
    """The objective as solved: its terms, the chosen symbol values and
    the evaluated prices, for the summarize-sdsc skill."""
    from torch_spyre._inductor.dump_common import emit_json_line, origin_op_name

    # Ops are named twice because the numeric cost dump names them twice:
    # ``op_names`` matches its block heading and is what a human reads,
    # ``op_ids`` matches its ``output opN`` line and is unique, so it is
    # the key that joins the two dumps.
    op_names, op_ids = {}, {}
    for op in graph.operations or ():
        try:
            named = (op.get_name(), origin_op_name(op), op.get_operation_name())
        except Exception:  # pragma: no cover - naming is best-effort
            continue
        op_names[named[0]], op_ids[named[0]] = named[1], named[2]
    context = {
        "op_names": op_names,
        "op_ids": op_ids,
        "env": {
            # Per SOLVE, not per run: the head-major attention path caps it
            # for its own compile and leaves the rest at 32.
            "sencores": config.sencores,
            "lx_capacity": capacity,
            "allocator": "ScratchpadAllocator",
        },
        "solve": dict(solve_stats),
    }
    emit_json_line(
        config.dump_cost_expr_file,
        cost_expr_record(
            cost_expr,
            bundle_terms,
            plan,
            _COST_PARAMS,
            context=context,
        ),
    )


def solved_bindings(plan: SolvedPlan) -> dict:
    """The objective's symbols as the solved plan fixes them: ``is_lx`` is 1
    for a resident node and 0 for a spilled one; a node with a chosen
    division binds its ``division`` index and each per-axis split symbol to
    that division's split (1 for an axis it does not split), and each
    tile-count symbol to that division's tile count (1 for an axis it does
    not tile)."""
    bindings: dict = {}
    for node in plan.nodes:
        bindings[node.sym_is_lx] = 1 if plan.address(node) is not None else 0
        cd = plan.division(node)
        if cd is None:
            continue
        bindings[division_symbol(node.name)] = plan.division_index(node)
        for key, sym in node.sym_core_divs.items():
            bindings[sym] = cd.splits.get(key, 1)
        for key, sym in node.sym_tile_counts.items():
            bindings[sym] = cd.tiling.count_for(key)
    return bindings


def _evaluate(expr: sympy.Expr, bindings: dict) -> float | None:
    try:
        return float(sympy.sympify(expr).xreplace(bindings).evalf())
    except (TypeError, ValueError, AttributeError):
        return None


def cost_expr_record(
    cost_expr: sympy.Expr,
    bundle_terms: Sequence[tuple[list[str], sympy.Expr]],
    plan: SolvedPlan,
    params: object = None,
    *,
    context: dict | None = None,
) -> dict:
    """One dump record for a solved co-optimized graph: the objective's
    per-bundle terms as ``sympy.srepr`` strings (lossless,
    ``parse_expr`` restores them), the solved symbol bindings, and every term
    evaluated under them. The plan's node names (the graph's stores) are what
    a reader joins on.

    ``divisions`` carries each buffer's candidate core counts, the one chosen,
    its producers, its residency ``reason`` when one kept it out of LX, and the
    division pairs the residency gate admitted on each incoming edge -- the
    alternatives a decision was made over, which the objective alone cannot
    show.

    ``context`` is whatever the caller knows and this function cannot see -- the
    environment the plan was made in, and how the solve went -- so a reader gets
    one self-describing record per solve rather than a set of numbers whose
    meaning depends on flags nobody wrote down. Its keys may not collide with
    the record's own."""
    import dataclasses

    bindings = solved_bindings(plan)
    bundles = [
        {
            "ops": list(names),
            "expr": sympy.srepr(sympy.sympify(term)),
            "value_ns": _evaluate(term, bindings),
        }
        for names, term in bundle_terms
    ]
    objective_ns = _evaluate(cost_expr, bindings)
    record = {
        "buffers": [n.name for n in plan.nodes],
        # Buffer sizes in bytes: with the names, a key that tells kernels apart
        # even though every kernel numbers its buffers from buf0.
        "buffer_sizes": {n.name: n.nbytes for n in plan.nodes},
        "params": dataclasses.asdict(params)
        if params is not None
        and dataclasses.is_dataclass(params)
        and not isinstance(params, type)
        else {},
        "bundles": bundles,
        # Reading why a division was chosen needs the alternatives it was
        # chosen over: per buffer the core count and split shape of every
        # candidate, the index the solver took, and the ``(parent, consumer)``
        # index pairs the residency gate admitted on each incoming edge. Pairs
        # are stored as INDICES into the two buffers' ``cores`` lists, so a
        # reader can render them as core counts without the record repeating
        # the divisions. Keyed by the CONSUMER, which is where ``parents`` and
        # ``cd_parent_matches`` are populated. A parent with an EMPTY pair list
        # divides no way this buffer can read locally; a parent absent from
        # ``matches`` altogether was refused an edge outright, which is the
        # louder of the two signals (issue #4655 was of that kind).
        "divisions": {
            n.name: {
                "cores": [cd.cores_used for cd in n.menu],
                "labels": [cd.label for cd in n.menu],
                "chosen": plan.division_index(n),
                "parents": [e.src.name for e in n.inputs for _ in range(e.reads)],
                # Why this buffer never reached the solver, in the allocator's
                # own words ("op not allowed", "partial/offset read"). None
                # means it DID reach the solver, and ``bindings`` says what the
                # solve then decided -- three outcomes a reader must not
                # conflate: excluded, weighed and declined, or resident.
                "reason": n.residency_reason,
                "matches": {
                    e.src.name: [list(pair) for pair in e.pairs]
                    for e in n.inputs
                    if e.pairs is not None
                },
            }
            for n in plan.nodes
            if n.menu
        },
        "bindings": {str(k): v for k, v in bindings.items()},
        "objective_ns": objective_ns,
    }
    # Additive only: context describes the record, it does not get to redefine
    # it. Without this a caller key named `bundles` would replace the terms.
    clashes = set(context or ()) & set(record)
    assert not clashes, f"context may not override record keys: {sorted(clashes)}"
    record.update(context or {})
    # A term that would not evaluate under the solved bindings reads in the JSON
    # exactly like one deliberately left unpriced. The difference matters: the
    # second is normal, the first means the objective and the bindings have
    # drifted apart -- a cost-model change introducing a symbol no engine binds,
    # say. Say so once, where the CP-SAT path already logs "cannot linearize".
    unpriced = sum(1 for entry in bundles if entry["value_ns"] is None)
    if unpriced or objective_ns is None:
        logger.warning(
            "cost dump: %d of %d terms did not evaluate under the solved "
            "bindings%s; objective and bindings may have drifted",
            unpriced,
            len(bundles),
            "" if objective_ns is not None else " (whole objective too)",
        )
    return record


def extract_op_features(graph, output_name, buffers, is_lx, *, op=None):
    """Build symbolic OpFeatures for one ComputedBuffer op (best-effort).

    Same extraction as dump_cost_model.extract_op_features, but keyed off
    each buffer's *symbolic* core-division vars (sym_core_divs) instead of
    concrete values, so the resulting OpFeatures can be fed to
    predict_ops() to build a cost expression over the solver's own
    decision variables. The extractor reads each arg's symbolic residency
    from ``is_lx`` (built once by the caller over all of ``buffers``, not
    per op); ``buffers`` itself supplies this op's own candidate divisions.

    Candidate coarse tilings enter the same way: each tiled axis's
    symbolic count (``sym_tile_counts``) is keyed on the op's own loop
    variable for that axis. With no tiled candidate there are none, and the
    extractor reads the op's committed tiling (``loop_info``) as before.
    """
    from torch_spyre._inductor.dump_cost_model import extract_op_features
    from torch_spyre._inductor.scratchpad.coarse_tiling import (
        try_resolve_tile_axis_loop_vars,
    )
    from torch_spyre._inductor.scratchpad.division_menus import _work_slices

    op = graph.get_buffer(output_name) if op is None else op
    buffer = buffers[output_name]
    division = CoreDivision(splits=buffer.sym_core_divs)
    ws = _work_slices(op, division)
    sym_tiling: dict = {}
    if config.auto_coarse_tiling:
        for (host_dim, is_reduction), symbol in buffer.sym_tile_counts.items():
            spec = TileSpec((TileAxis(host_dim, 1, is_reduction),))
            loop_vars, reason = try_resolve_tile_axis_loop_vars(op, spec)
            if loop_vars is None:
                raise Unsupported(reason)
            (var,) = loop_vars
            sym_tiling[var] = sym_tiling.get(var, 1) * symbol
    return extract_op_features(
        op,
        ws,
        is_lx=is_lx,
        sym_tiling=sym_tiling or None,
        candidate_work_slices=[
            _work_slices(op, candidate) for candidate in buffer.core_divisions
        ],
    )
