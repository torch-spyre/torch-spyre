# Scratchpad refactor: open questions

Running list while the plan (`~/spyre-roadmap/plans/cpsat-graph-buffer-refactor.md`)
is implemented end to end on `scratchpad-refactor-experimental` as an exploration.
Each entry says what I assumed in the meantime, so work did not stop on it. Newest
last. Feasibility results so far: `~/spyre-roadmap/plans/cpsat-graph-buffer-refactor-feasibility.md`.

Status (2026-10-22): stopped at Phase 2 plus 3a (Q11). The zoo benchmark is done:
every new mode (derived, Lazy with points, digits, point menus) reaches the same
objective as today's tables on 64/64 solves, and the agreement checks pass (746/746
derived edges, 1,011/1,011 Lazy edges, 582/582 Lazy spaces). Planning time over the
zoo, against today's tables with relayouts off: tiling compiles 161 s -> 123 s
(derived) and 113 s (Lazy + point menus); default compiles 47 s -> 57-70 s, all of
it CP-SAT time (Q12). The report has the full tables and an implementation guide.

## Progress: the whole plan (requested 2026-10-22, "implement the whole plan with green CI")

| step | state |
|---|---|
| Phase 1 plan text: `Transform` / `TransformKind`, `encode_transform` / `encode_transform_use`, `encode_lifetime`, `NodeVars.start/end`, `edge.merged` / `edge.transform`, `chosen_transforms()` | done; same model (smoke 37/37, unit-test solves 99/99) |
| 2c input clones as constraints (`encode_clone`, `CloneRead`) | done under `LX_VIEW_RELATION=derived`; clone consumers can be Lazy |
| 2d relayouts as constraints (`LX_RELAYOUT_ENCODING=lazy`, `lazy_relayout.py`) | done; 8 models x 2 configs, no errors, objectives far below no-relayout (attn 31455 -> 21199, transformer 83385 -> 58349), residency within 1 buffer of the table relayouts |
| 2e in-place as constraints | covered: an in-place edge between Lazy nodes is the derived view relation plus footprint equality, each footprint a table over the axes it varies with (Q13) |
| 3a `set_solver`, `GraphSolver`, `MarkTableAxis`, `Decomposed` (CP-SAT leaves) | done |
| 3b node absorbs `lx_view`, relayout plans, paired members (pinned transforms) | done |
| 3c graph from the IR (`ir_graph.py`, `LX_IR_GRAPH_CHECK`) | done; agreement run in progress |
| 3d placement solvers read placement views (`PlacementSolver`); greedy drops its isinstance | done |
| 3e exhaustive search as `DecomposedPlacement`; annealer charges `spill_cost` | done; annealer adjacency stays `op_position` (Q14) |
| 3f tests moved with the API | done for 3a-3g |
| 3g one allocator (`division_mode`), `CoOptimizingAllocator` deleted; the allocator builds no buffers: graph from the IR, `solve_graph`, commit from the solved graph | done; same-plans 80/80 before this step, rerun in progress (Q15) |
| Constraint encodings the default (derived relation, Lazy split, lazy relayouts) | done 2026-10-23 (df7359e6); the table encodings and their flags then deleted (Q17) |
| Graph solvers declare `Capabilities`; the allocator takes a graph solver; op rules out of the allocator (`residency_rules`, `division_menus`, `drain_plans`, `pricing`) | done 2026-10-23 (3df2ccf6) |
| Placement-only solvers replaced by fixed divisions (Q16) | done 2026-10-23 (de1bd4f2) |
| Solver classes deleted: `CpSatLayoutSolver`, `MemoryPlanSolver`, `CoreDivisionLayoutSolver`, `graph_solver_for`; the annealer and first-fit are plain engines (Q18) | done 2026-10-23 |
| Final cleanup: delete redundant code | done with the steps above |
| The solver returns its nodes; `SolvedPlan` reads decisions off them (Q18) | done 2026-10-23 (f6f973db) |
| One path: the annealer, graph solvers, `ModelSolver`, `layout_solver` and the factory deleted (Q19) | done 2026-10-23 |

## Q1. Stage 9: extraction or generation from the op?

Stage 9 is done as an extraction: the candidate derivations moved verbatim from
`CoOptimizingAllocator` into `scratchpad/table_candidates.py`, and the solver still
reads candidates from the buffers. Generating them inside the strategy from
`node.op` / `node.space` needs the graph built from the IR (3c), and until then
the cost expression and the SA / exhaustive solvers read the same buffers.

- Assumed: extraction is acceptable for Phase 1; true generation comes with 3c.
- Answer needed: is that the split you want, or should Stage 9 wait for 3c?

## Q2. Apply the plan's later model changes (Transform, lifetime vars) to the code?

The plan changed after Phase 1 was implemented (`RelayoutLayout` -> `Transform`,
`encode_relayout` -> `encode_transform_use`, `NodeVars.start/end`,
`encode_lifetime`). The spike shows free lifetime ends cost nothing (79 captures,
same optimum, median 1.01x). The code still uses the old names.

- Assumed: rename later, once the Phase 2 prototype settles; it is mechanical.
- Answer needed: rename now, or keep the code at the plan revision it was built
  from (22966fe)?

## Q3. Phase 2 "lazy": stop at L1 (per-node tables) or go to L2 (no menu)?

The first lazy strategy follows the plan's 2.0 L1 arm: menus stay as per-node
tables, pair tables go, and every edge's view relation becomes equality of
per-node view ids, tile-view ids and core counts derived from the edge's
`ResidencyEdge`. Overridden edges (input clones, loop carries) keep pair tables.
L2 (no menu) needs views derived in closed form from index maps (2c proper), which
is a re-derivation of `_per_core_view_from_prep` (311 lines with rescues).

- Assumed: build L1 end to end first (`LX_VIEW_RELATION=derived`), benchmark it,
  then attempt L2 for the common digit-form views with a per-node fallback.
- Answer needed: is L1 enough for the decision you want to make?

## Q4. Relayouts under the derived relation

The relayout candidates are priced per view pair (`_cd_parent_relayouts`), so the
first derived-relation strategy turns relayouts off (no candidates generated).

- Assumed: off is acceptable for the first benchmark; the plan's 2d (lazy relayouts:
  one optional layout per edge, shared by reified view equality, closed-form
  legality) comes after L1 works.
- Answer needed: should the benchmark compare against table-with-relayouts or
  table-without-relayouts? (I will report both.)

## Q5. Pushing exploration commits to PR #5676

Every commit on `scratchpad-refactor-experimental` updates PR #5676 (draft, "DO NOT
MERGE") and runs CI. The Phase 1 commits are green.

- Assumed: keep pushing; CI on the exploration is useful signal.
- Answer needed: or should the Phase 2/3 exploration go to a separate branch so
  #5676 stays at Phase 1?

## Q6. MIP backend: one stage still INFEASIBLE

After fixing the product linearization, SCIP still reports `sdpa_1x8x256x64`
stage 0 INFEASIBLE (43k MIP vars) where CP-SAT is optimal: a remaining linearizer
defect or big-M numerics.

- Assumed: not worth chasing for the feasibility question (MIP is a second opinion
  on small models only).
- Answer needed: do you want the MIP backend kept at all?

## Q7. What the Lazy work split tabulates over

`LX_DIVISION_ENCODING=lazy` (with `LX_VIEW_RELATION=derived`) gives a node no
candidate index. Its split is one var per loop axis plus a tiling choice; legality is
posted per tiling frame from its `WorkDivisionContext` (`lazy_division.py`, ported
from the 2f spike). Every per-division quantity (footprint, core cost, tile counts,
view ids, cost-expression terms) is a table over only the axes it varies with,
tabulated over the space's *points*: each frame's factor domains within the core
budget, a superset of the legal divisions that is cheap to list. No legality check
and no menu are involved in those tables.

- Assumed (recommended): tabulate over points for now. Views need per-point view
  computation (`_view_for_div`), which is the cost 2c's closed-form views remove.
- Answer needed: is per-(node, dep) view-id tables over points an acceptable
  fallback, with 2c's digit form as the follow-up that removes the view computation?

## Q8. The allocator still builds menus

The Lazy strategy ignores the menu in the model, but the allocator still enumerates
it: the carry and clone pair tables, residency gating and relayout pricing read it,
and the solve writes its choice back as a menu index (`b.chosen_division`). So the
planning-time saving of "menus gone" (the plan's ~78 s zoo-tiling column) does not
show yet; the benchmark will show model size and solve time only.

Update: point menus (`LX_LAZY_MENUS=1`) skip the enumeration for nodes that will be
Lazy, which takes the zoo-tiling allocator time from 99.6 s to 61.4 s (134.8 s with
today's tables). The carry, clone and restickify-pricing consumers still get full
menus.

- Assumed (recommended): keep the menus until Phase 3 (3c/3g) moves those consumers
  onto the graph; report allocator time separately.
- Answer needed: should removing the menu from the allocator be part of this
  exploration, or is the model-side result enough?

## Q9. Which nodes stay Table under Lazy

A node stays `Table` if its split is pinned or committed, it has no space (not
enumerable), it is an end of an edge whose relation is still a pair table over menu
indices (loop-carry update/read edges, input clones), or its space has more than
4096 points. This is the plan's mixed enumeration (rule 5).

- Assumed (recommended): as above; clones and carries move with 2c's clone encoding.
- Answer needed: none unless you want a different cut-off.

## Q10. Relayouts under Lazy (2d): worth building now?

The derived relation and Lazy run without relayout candidates (Q4). Measured on the
zoo (table with vs without relayouts, same code, separate processes): relayouts make
20 more real buffers resident (610 vs 590, about 3%), plans are identical on 40 of
64 solves, and planning takes about 70% longer (prepare 239.7 s vs 141.5 s, solve
105.2 s vs 66.3 s in total).

- Assumed (recommended): 2d is not built in this exploration; its closed form for
  `movement_supported` is already verified offline (Phase 2.0), and the remaining
  per-pair gates and the price are the work.
- Answer needed: is a ~3% residency gain worth 2d before the rest, or should
  relayouts become an opt-in refinement?

## Q11. How far into Phase 3

3a (backend protocol) is built. 3b-3g are restructurings of the allocator: 3c builds
the graph from the IR beside the buffers and must agree with it, and 3g deletes
`CoOptimizingAllocator`. Their feasibility was judged from the code and the 3c spike
(762/762 `produced_at`), and the plan says Phase 3 needs its own plan.

- Assumed (recommended): stop implementation at Phase 2 (L1 derived relation, Lazy
  work split, digit views, point menus) plus 3a, finish the benchmark, and write the
  implementation guide and results into the feasibility report.
- Answer needed: do you want a 3c prototype (IR builder plus agreement check) too?

## Q12. CP-SAT is slower on every new encoding

Over the zoo's default compiles, CP-SAT time goes from 12.2 s (pair tables) to
21.7 s (derived) and 26.9-31.8 s (Lazy). Default compiles have small menus, so the
allocator has little to save and the total rises 21-48%. On tiling compiles the
allocator saving outweighs it (-24% derived, -30% Lazy with point menus). Equality
of id vars propagates worse than a pair table, and Lazy adds product chains and
reified span terms.

- Assumed (recommended): not investigated further here. The guide lists the
  candidates (keep pair tables where both menus are small, solution hints from a
  table solve, a table instead of the product chain for the core count) and says to
  measure them before making either encoding the default.
- Answer needed: is a per-compile choice (derived / Lazy only when coarse tiling is
  on) acceptable, or must the new encoding be no slower everywhere?

## Q13. 2e as a product equality

The plan writes in-place footprint equality as `out_partition * out_tiles`, a
product equality. Under the Lazy encoding a node's footprint is already a table over
the axes it varies with (the output splits and the tiling option), and an in-place
edge posts footprint equality plus the derived view relation, which is the same
constraint.

- Assumed (recommended): no separate product form; it would add a product chain
  per node for the same feasible set.
- Answer needed: none unless you want the literal product form.

## Q14. The annealer's op adjacency

The plan moves `SaCoOptimizingSolver`'s op adjacency onto execution edges. On real
graphs `op_position` equals each op output's `produced_at`. But 21 annealer tests
build synthetic buffers whose `uses` all start at tick 0 and rely on `op_position`
alone, so deriving positions from the chain makes them collide.

- Resolved (2026-10-23): on the graph path the annealer's positions come from each
  node's `produced_at` (`graph_solvers.buffers_of`); `CandidateTables.positions` is
  gone. The engine still takes `op_position` on its buffers, so the synthetic fixtures
  are unchanged.

## Q15. How far "delete the allocator's buffer construction" goes (answered)

Answer (2026-10-22): "The allocator shouldn't build buffers; we should be able to pass
IR buffers in the graph and solve; the allocator should just coordinate solving and
committing of results."

Done:
- The candidate generator builds the graph plus the strategy's `CandidateTables`:
  `TableCandidates.prepare_graph` (the committed mode is the same generator with every
  menu pinned, Q16).
- The allocator hands the graph to its graph solver (`GraphSolver.solve`, Q18).
- The commit path reads the solved graph (`SolvedPlan`).

`_generate_buffers`, `_build_bound_buffers`, `_append_lx_relayout_destinations`,
`_build_cd_bound_buffers` and `_relayout_copy_buffers` are deleted. Buffers remain only
as the solvers' buffer form (unit tests and capture replay) and as the annealing
co-optimizer's internals, built from the graph and tables (`graph_solvers.buffers_of`).

## Q16. Placement-only solvers (answered)

Answer (2026-10-23): "We don't need placement only solvers. That will be covered by
fixed strategies."

Done: greedy, best-fit, placement-only SA, exhaustive search and the committed-mode
builder are deleted. `co_optimizing_lx_planning=False` is the same solve with every
op's menu pinned to its committed division, priced like the open mode. Greedy's
committed relayout plans, ownership anchoring and the fixed-division residency
branches went with it. A failed open-mode solve replans with the committed divisions;
if that fails too nothing is placed in LX.

Known loss: no relayout serves a *completed reduction* (a reduction finished on fewer
cores, e.g. an MoE down projection). Only the deleted committed collector built those
routes; the solver's relayout edges never have. The two device tests now assert the
HBM read and exact numerics. Restoring it means a completed-reduction `RelayoutEdge`.

## Q17. Obsolete equivalence-only code (answered)

Answer (2026-10-23): "Don't keep obsolete stuff just for equivalency tests."

Done: `LX_VIEW_RELATION`, `LX_DIVISION_ENCODING`, `LX_RELAYOUT_ENCODING` and
`LX_SOLVER_RELAYOUT_GROUPS_PER_EDGE` are deleted, with the table relation and table
work split as CP-SAT encodings and the capped `RelayoutCopyBuffer` path. The factory is
built from the solver's capabilities alone: a model-posting solver gets
`LazyStrategy`, the annealer `TableStrategy`. Old capture pickles holding
`RelayoutCopyBuffer` no longer load; the replay skips them.

## Q18. The solver classes (answered)

Answer (2026-10-23): "I expected the solvers to essentially drop by the end."

Done: the allocator is configured with a graph solver (`ModelSolver` for CP-SAT,
`BufferSolver(SaCoOptimizingSolver)` for the annealer). `CpSatLayoutSolver`, the
`MemoryPlanSolver` / `CoreDivisionLayoutSolver` bases and `graph_solver_for` are
deleted. Tests solve buffers with `build_graph` + `ModelSolver.solve` and read the
decisions through `SolvedPlan`. `build_graph` checks the buffer form's in-place pairs.

Later (2026-10-23): "Shouldn't we be returning the resource nodes from the solver?" A
solver returns every node with its decisions; `SolvedPlan(nodes, tables, alignment)` is
built after the solve and reads the chosen division from `node.work_split` and the fired
relayout from `edge.relayout`.

## Q19. Only the main code path (answered)

Answer (2026-10-23): "Prune and keep only the main code path", choosing to retire the
annealing co-optimizer and keep `Annealing` as a graph backend.

Done: `sa_cooptimizer.py`, `first_fit.py`, the C++ permutation packer,
`graph_solvers.py` (`GraphSolver`, `Capabilities`, `BufferSolver`, `buffers_of`) and
`ilp_solver_ortools.py` (`ModelSolver`) are deleted, with `config.layout_solver`,
`EnumerationFactory` and `TableStrategy`'s `derived` flag. `SpyreOptimizationGraph.solve`
bars oversized nodes, records the solve stats and picks its backend from
`SPYRE_LX_BACKEND`. Code only the annealer reached went too: split spaces on nodes and
buffers (`OpSplitSpace`, `build_op_split_space`, the generated-division relations on
`ResidencyEdge`), `TileReads`, and the allocator's apply-in-place tiling path
(`_apply_chosen_tilings` and its checks, `JOINT_TILING_AND_DIVISION_ATTR`). CP-SAT applies
its tilings in `_materialize_selection` and re-solves, so the committed plan never
carries a tiling. The cost dump drops `off_expression_ns`, `score_ns` and `env.solver`.

## Q20. A configurable division strategy (answered)

Answer (2026-10-23): "I want the strategy to be configurable. We should use lazy the
way we are now but we should be able to fall back to enumerating and picking from the
set and encoding accordingly." Also: delete the `Decomposed` backend (nothing used it
after the exhaustive search went).

Done: `config.lx_division_strategy` (`LX_DIVISION_STRATEGY`), `"lazy"` (default) or
`"enumerated"`. `select_allocator` passes it to `ScratchpadAllocator(division_strategy=)`,
which picks `LazyStrategy` or `TableStrategy` and builds `TableCandidates(lazy=)`: off,
no op gets a `Lazy` space or a point menu, so every menu is the op's legal divisions.
The committed replan keeps the strategy.

Later (2026-10-23): "We should still be able to do the pure table based strategy as
well ... an additional strategy specialization." Done: `"table"` selects
`PairTableStrategy(TableStrategy)`, which relates every edge by a pair table, with
`TableCandidates(pair_tables=True)` building one per edge (clone edges included). Relayouts
stay the lazy ones; they relate their ends through the strategy's `_relate`, which reads
the pair table.
