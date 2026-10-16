# The co-optimizer

:::{admonition} Draft
:class: warning

This is a first draft assembled from `sa_co_optimization.md`, `scratchpad_planning.md`,
`work_division_planning.md`, and the current state of
`torch_spyre/_inductor/scratchpad/`. It is meant as a jumping-off point for review, not a
finished page — see the open questions in [Ongoing work](#ongoing-work)
before relying on any claim there.
:::

This page is the map of **why** torch-spyre co-optimizes several compiler decisions together
instead of solving each one separately, **how** that joint search works in general, and
**which** decisions it currently covers. For the full detail on any one piece, follow the links
into [Scratchpad Planning](scratchpad_planning.md), [Work Division Planning](work_division_planning.md),
[Joint core-division + LX placement](sa_co_optimization.md), and
[Analytical Cost Model](cost_model.md).

**Quick navigation:**

- [Why a co-optimizer](#why-a-co-optimizer)
- [How it works in general](#how-it-works-in-general)
- [What gets co-optimized today](#what-gets-co-optimized-today)
  - [LX placement and work division](#a-lx-placement-and-work-division)
  - [LX placement, work division, and coarse tiling](#b-lx-placement-work-division-and-coarse-tiling)
- [Cost model](#cost-model)
- [Ongoing work](#ongoing-work)
- [Related documents](#related-documents)

## Why a co-optimizer

In the `torch-spyre` compiler pipeline, work division runs before LX planning, and its
objective is simply to pick the best parallelization scheme for each op *individually*. But
consider a simplified program with two memory-bound pointwise ops in a chain: because scratchpad
is private per core, the key performance optimization is making sure op1's output is already
exactly where op2 needs it on each core's own scratchpad — not moved there after the fact,
whether through an HBM round trip or an explicit relayout copy between cores, since both are
real options but neither is free. That means the data core 0 holds in its scratchpad after op1
must be exactly what core 0 needs for op2. An awkward work division breaks this: if op1 splits
along rows and op2 along columns, the data op2 needs on a given core was never produced on
that core, and moving it there is the extra runtime cost this section is about.

Solving work division and LX placement one at a time, freezing the earlier choice before the
later pass runs, leaves exactly this interaction on the table: either compute utilization is
sacrificed for memory locality, or memory locality is sacrificed and HBM traffic balloons. In
either case, joint assignments that *reconcile* several ops onto one shared division — one
that keeps everyone's compute shape good **and** keeps the shared buffer resident — are missed
simply because the earlier pass already committed. The co-optimizer exists to search that
combined space instead of two disjoint ones.

### A simplified example of core division choices on buffer residency

Take `X - X.max(dim=0)`, a 2-op chain (`max(dim=0)` into `sub`, with input `X` feeding both).
Two cores will be used for each op, with a pre-determined work division setting for each. For
example, an input buffer of shape `M x N` with `max` set to split by columns has core0 read and
process the first half of the buffer (`M x N/2`) while core1 handles the other half. If the
columns each core needs are already pinned on that core's scratchpad, the op can read directly
from LX and process successfully. The resulting row vector is written to LX or HBM; assume the
ideal case where it resides on LX. Note that core0's scratchpad then only has half of that row,
as shown in box 1 in the figure below.

If the subsequent `sub` is unfortunately set to process by rows, each core needs the *entire*
row vector of `buf1` to perform the operation — the half row `max` left on LX is not sufficient
for each core to proceed. At compile time, `torch-spyre` recognizes the problem and tries to
remedy it. Without the co-optimizer, work division decisions are considered fixed during
scratchpad planning, so the only option at this stage is to force `max` to write to HBM instead
of LX, as shown in box 2a in the figure. Because every core can read from HBM and fetch the
entire row of `buf1`, `sub` can execute correctly and write its output to HBM (or LX — a
decision that will be made similarly when planning for the next op).

HBM read/write is more expensive than LX, so a better option is for `sub` to process by columns
instead, as shown in box 2b in the figure. More generally: work division decisions for each op
should stay consistent with prior ops' decisions when possible, so the total HBM cost is
minimized. Hence the need for co-optimization.

:::{figure} ../_static/images/lx/co-optimization-v2.svg
:alt: A 2-op chain (max then sub) and the LX scratchpad's occupancy at each step; choosing sub's division to match max's keeps every buffer pinned, choosing a mismatched division spills some of them to HBM
:width: 700px
:align: center
:::

## How it works in general

In a nutshell, the co-optimizer relies on a cost model to estimate the overall compute cost and
communication cost of a given set of work division choices. Matmul compute is the only op type
with its own compute-roofline term — a dependent fused reduction riding in the same bundle as a
matmul gets a smaller additive correction — while every other op is treated as pure memory
traffic. The data-transfer time is estimated by where each buffer lives (HBM or LX), using known
or measured bandwidth for the target hardware. Rather than committing to one work division per
op up front, it evaluates several combinations together and prices the communication cost that
each one implies, then commits only the winning combination at the end of the search. A few
different solver backends can drive this search; the sections below cover each in detail, and
[Cost model](#cost-model) covers the model itself — how it plugs into the solvers, how to
inspect it, and how to recalibrate it.

The co-optimizer is reached through `CoOptimizingAllocator`
(`torch_spyre/_inductor/scratchpad/allocator.py`), gated by `config.co_optimizing_lx_planning`
(env `CO_OPTIMIZING_LX_PLANNING`, **on by default**) and selected via `config.layout_solver`. It
runs as a pre-scheduling pass: `V.graph` is live but `V.graph.scheduler` is still `None`, so
fusion hasn't happened yet. The cost model therefore can't read a kernel's real fused grouping
off the scheduler — it has to *estimate* one from the flat, ordered operation list instead.
Without `co_optimizing_lx_planning`, each solver can still be selected on its own, but then it
only *places* buffers against divisions some earlier pass already fixed — it never searches
divisions itself. The joint behavior is what `CoOptimizingAllocator` adds on top.

`layout_solver`'s options are `"cpsat"`, `"simulated_annealing"`, `"greedy"`, `"firstfit"`, and
`"bestfit"`, but only `"simulated_annealing"`, and `"cpsat"` *with `ortools` installed*, are
natively core-division-capable and reach a purpose-built joint solver directly.
`select_allocator()` rejects every other combination by default and raises `ValueError`
**unless** `config.allow_exhaustive_search` (env `ALLOW_EXHAUSTIVE_SEARCH`) is also set, in
which case `ExhaustiveSearchSolver` will perform a DFS search over
`greedy`'s own division menu (see details below). **Note this DFS is exponential** — `K^N`
leaves over the `N` buffers with a real choice — **and has no size cap or timeout of its own.**
The reject-by-default behavior above is the only guard against that blow-up, so only opt in on
a task small enough for the exponential compile-time cost to be acceptable. This fallback is
not a general substitute for CP-SAT or SA.

### The simplest version: exhaustive search over greedy placement

Before getting into the production solvers, it's worth seeing the simplest possible instance
of "co-optimize divisions and placement jointly," because it makes the shape of the problem
concrete without any annealing or constraint-solving machinery to explain first.

`ExhaustiveSearchSolver` (`torch_spyre/_inductor/scratchpad/exhaustive_search.py`) wraps the
simplest placement-only solver, `GreedyLayoutSolver`, and brute-forces the other half of the
problem — searching — around it. For every buffer it generates a few legitimate candidate
division choices, then tries every combination with an exhaustive DFS, bounded by `K^N` leaves
where `N` is the number of buffers with a real choice and `K` is each one's candidate count. At
each leaf:

1. Builds a fresh set of per-core buffer sizes implied by that combination of chosen divisions.
2. Hands them to a brand-new instance of the wrapped placement solver (a solver is single-use)
   and calls `plan_layout()`.
3. Scores the leaf by summing the total size of every buffer the inner solver could *not* pin
   to LX — the same differential-HBM-traffic idea the production solvers' memory-only
   fallback uses.

The combination with the lowest score wins; its divisions are committed to every buffer, and
the wrapped solver runs once more to produce the final LX/HBM addresses.

For example: a producer with 2 candidate divisions feeding a consumer with 2 of its own is a
`2×2 = 4`-leaf DFS. Say only one of those four pairings avoids a `core_div_mismatch` between
them, per `_cd_parent_matches()` (`allocator.py`) — the same relation the production solvers use
to decide which combinations even make a buffer LX-eligible via its `cd_parent_matches` field,
not a check bolted on separately here. The three mismatched leaves each spill one buffer to HBM
and score accordingly; the matching leaf scores zero (both pin). `ExhaustiveSearchSolver` tries
all four, picks the lowest-scoring leaf as the winner, and commits its divisions.

Note this is a simplified scoring which only ever counts unpinned-HBM bytes — there is no
compute-cost term anywhere in this solver (`plan_layout_and_core_divisions` even carries a
`# TODO: use cost_expr here`), and it never generates a division of its own: it only indexes
into the candidate divisions each buffer already carries in from the upstream work-division
pass. So this simplest version implicitly assumes those upstream divisions are already the
right call for each buffer's compute shape, and only searches for the pairing that keeps the
most of them resident. For a memory-bound graph that's not always true: a division that's
suboptimal for one op's matmul utilization could still lower the *overall* cost if it lets a
shared buffer stay pinned, but this solver has no way to see that trade-off or reach for such a
division unless it happens to already be on the menu. The production solvers below close part
of this gap — CP-SAT and SA both weigh a compute-roofline term alongside HBM traffic — but even
they choose only among divisions `enumerate_work_division_candidates()` offers, not an unbounded
search over every possible division.

This also shows why `ExhaustiveSearchSolver` isn't what ships by default: the compile-time cost
is exponential in the number of buffers with a real choice. `select_allocator()` only reaches for
it as a fallback — when `co_optimizing_lx_planning` is set but the configured `layout_solver`
has no purpose-built core-division-capable solver to co-optimize with (i.e. anything other than
`"cpsat"` with `ortools` installed, or `"simulated_annealing"`) — and only when the caller has
opted in via `config.allow_exhaustive_search` (env `ALLOW_EXHAUSTIVE_SEARCH`); otherwise it
raises rather than silently paying that cost. The two production solvers below exist to make
the common case affordable; the rest of this section covers how `ExhaustiveSearchSolver` itself
is kept affordable when it does run.

**Pruning the DFS's candidate menu: `_enum_split_options`.** The example above assumed each
buffer's candidate list — its "menu" — was already small (2 candidates). In practice,
`select_allocator()` always pairs `ExhaustiveSearchSolver` with
`CoOptimizingAllocator(prune=True)`, which controls exactly that menu: instead of
`enumerate_work_division_candidates()`'s full, standalone-work-division-planner cross product,
`_enum_split_options` (`allocator.py`) builds a much smaller, heuristic candidate list per op,
dispatching on op type — the difference between a `2×2`-leaf DFS and one with every legal
division for every buffer. CP-SAT and the SA solver never set `prune=True`; this pruning exists
specifically to keep *this* DFS's exponential cost affordable, not as a third solving strategy
alongside them.

Generated alternatives are deduped and filtered down to ones that are actually legal (no
stick-count overflow, no collapsed/broadcast dims, hard work-division constraints respected).
Within that, pointwise ops get a handful of dim-flip variants and matmuls get each other's
splits and a shared batch-major option, so that reconciling buffers across ops stays possible
without blowing up the per-op menu size — the specific heuristics mirror the kind of
compute-vs-locality trade-off [Why a co-optimizer](#why-a-co-optimizer) motivates, just applied
per op type here instead of jointly. The leaf-scoring function is the same one described above
(`_generate_buffers + plan_layout`, scored by unpinned HBM bytes), with some of its repeated
work memoized across leaves.

These heuristics are marked TEMP/TODO: the intent is for work division to assign consistent
splits directly, at which point they can be removed. See
[Co-optimization is still limited](scratchpad_planning.md#co-optimization-is-still-limited)
for the remaining gaps in this path.

### The production solvers

Two solver backends implement the joint search at scale, differing in paradigm but sharing
the same inputs and the same cost model:

- **`CpSatLayoutSolver`** (`config.layout_solver = "cpsat"`, the default) models the problem as
  one global constraint-satisfaction model via Google OR-Tools CP-SAT. Core-division candidates
  (and, where landed, tiling candidates) are decision variables, placement is a 2D no-overlap
  packing — each resident buffer an optional `[lifetime] × [address, address+size)` rectangle —
  and producer/consumer slicing-match constraints tie divisions together across edges. CP-SAT
  searches exactly, minimizing predicted HBM traffic.
- **`SaCoOptimizingSolver`** (`config.layout_solver = "simulated_annealing"` with
  `co_optimizing_lx_planning`) anneals a single joint state heuristically. The state is the pair
  `(pi, W)`: the LX layout permutation `pi` (held in a composed `PermutationBasedLayoutSolver`)
  and the division vector `W`, one candidate-menu index per buffer. It seeds every buffer at
  menu index 0 with a FirstFit placement, then runs one geometric cool, a capped number of
  steps per buffer, using three move types — **reorder** (re-place a buffer in `pi`,
  weight 0.5), **flip** (change one buffer's division, or move a coarse-tiling boundary, weight
  0.3), and **recolor** (flood a coordinated division change across a connected region via the
  `cd_parent_matches` compatibility relation, weight 0.2) — and returns the best state seen, so
  the result is never worse than the seed. See [Joint core-division + LX placement](sa_co_optimization.md) for the
  full mechanics.

### Co-optimizer will honor `spyre_hint` for work divisions

A resolved `spyre_hint(work_div=...)` is still honored under the co-optimizer, not overridden
by the joint search. `allocator.py` collapses that op's candidate-division list to the single
division the hint requested *before* either solver runs — the hint acts upstream of the search,
not as a rule the search itself has to learn or respect. Everything else in the graph stays
co-optimized as usual; only that one op loses its freedom.

This also means the cost formula itself has no notion of a `spyre_hint` and never
special-cases one: the formula isn't biased toward the hint, the search just has nothing left
to search over for that buffer.

This is unconditional on the hint surviving to reach the co-optimizer, including hints inside a
`for_each_tile` body — except a hint collection/recovery bug during the scan-to-`while_loop`
retrace currently prevents that propagation (`#5057` fixes it, not yet landed).

Confirmed empirically, not just read in code: a pilot sweep's `bmm_wd` row requesting
`WD_B=1 WD_K=1 WD_M=8 WD_N=4` produced a compiled kernel whose dumped `MODEL FEATS` carry
exactly that split, unperturbed.

### Symbolic cost formula

Both production solvers are driven by a *single* cost expression (`cost_expr`), built once, up
front, by `predict_by_bundle(graph.operations, op_features, ...)`. It's a sympy expression over
two kinds of symbol, one pair per buffer: `sym_is_lx` (1 if the buffer ends up LX-resident, 0 if
it spills to HBM) and `sym_core_divs` (which division was chosen). `predict_ops`'s real model
has many more terms (turnaround, replicated-read pricing, access-pattern bandwidth overrides,
and per-bundle dedup for a graph-boundary input a fused kernel's clone-in loads only once — see
its docstring); simplified down to the shape that matters for co-optimization:

```python
# Build: walk the graph once, symbols only, nothing bound yet.
cost_expr = 0
for op in ops:
    for buf in op.reads_and_writes:
        cost_expr += (1 - sym_is_lx(buf)) * size(buf, sym_core_divs(buf)) / bandwidth
    cost_expr += compute_cost(op, sym_core_divs) if op_is_matmul(op) else 0

# Search: each candidate binds every buffer's symbols at once, then
# evaluates the SAME cost_expr under that one joint guess.
for candidate in candidates:  # CP-SAT's own solver vars, or SA's (chosen, placement) state
    bindings = {sym: candidate[sym] for sym in cost_expr.free_symbols}
    real_cost = cost_expr.xreplace(bindings).evalf()
    if real_cost < best_cost:
        best_cost, winner = real_cost, candidate
```

During the build phase, the inner sum is a buffer's own HBM term, linear in `sym_is_lx` and
zero once it's pinned. `compute_cost` is zero for every op except a matmul, which adds its
roofline term keyed off the same `sym_core_divs`. The whole graph's `cost_expr` is just this
double sum, built once as pure symbols before any decision is made.

The search phase never changes that expression; it instead scores each candidate by
substituting its guess into `cost_expr` and evaluating — CP-SAT's own decision variables, or
SA's `(chosen, placement)` state, are a *different* representation of the same decisions. Once
a winner is picked, only its guess is ever written back onto the buffers (as `.address` and
`.chosen_division`), which is what finally pins down what `sym_is_lx` and `sym_core_divs`
turned out to mean for that buffer.

When no cost expression is usable (symbolic shapes, serialized captures, or a construction
error), both solvers fall back to a cheaper **memory-only** objective: pure differential HBM
traffic, where a resident buffer costs zero and only spills are summed — the compute-roofline
term drops out entirely. This fallback is weaker — on a graph where everything already fits,
every division scores identically and the search has nothing left to optimize on.

## What gets co-optimized today

**Controlling it**: `config.co_optimizing_lx_planning` and `config.layout_solver` gate both
subsections below — see [How it works in general](#how-it-works-in-general) for their options,
and for what happens with a non-core-division-capable `layout_solver` and the
`ExhaustiveSearchSolver` fallback. Those two are all (a) needs; (b) adds one more on top,
`config.auto_coarse_tiling` (env `AUTO_COARSE_TILING`, off by default) — the dedicated switch
for adding coarse tiling to the pair in (a), inert unless `layout_solver` is `"cpsat"` or
`"simulated_annealing"`.

### a) LX placement and work division

This pair is the smallest set of "joint" search, covered above in [Why a
co-optimizer](#why-a-co-optimizer) and [Symbolic cost formula](#symbolic-cost-formula): LX
placement (where a buffer's resident copy lives, or whether it spills to HBM) and work division
(how many cores an op uses and which slice each one owns) are both symbols in the same
`cost_expr`, so the solver can trade one against the other. See [Co-optimizer will honor
`spyre_hint` for work divisions](#co-optimizer-will-honor-spyre_hint-for-work-divisions) for
how a forced division interacts with that search.

**Status**: implemented and tested together — see [Scratchpad
Planning](scratchpad_planning.md) and [Work Division
Planning](work_division_planning.md) for each variable's own standalone mechanics.

### b) LX placement, work division, and coarse tiling

A Spyre has 32 cores, and each core's LX scratchpad has only ~1.5MB usable for planning (2MB,
minus other reservations). Even divided across all 32 cores, a single op's buffer can
still be too large for its most-divided per-core share to fit — and those large buffers are
usually the ones that most need LX for efficiency. `torch-spyre` handles this by breaking the
buffer into tiles and converting the op into a loop, so a working set that doesn't fit LX at
full size can fit one tile at a time. The analogy to work division is direct: work division
splits in *space* (across cores, at the same time); coarse tiling splits in *time* (across
loop iterations, on the same core). The underlying loop-IR mechanism — how a tiled run is
represented and survives scheduling and codegen — is implemented and documented in
[Coarse-Tiling Loop IR for the Spyre Backend](coarse_tiling_loops.md); that document also
points to the design RFC. What's described here is specifically the *solver's* ability to
choose a coarse tiling as part of the joint search, which is a separate, newer effort.

At a high level: neither solver prices tiling as a third term alongside LX placement and work
division in `cost_expr`, but both decide it jointly with them. CP-SAT does so in the *same*
constraint model, just ranked by a different objective. When `auto_coarse_tiling` is on,
`cost_expr` is dropped and residency, division, and tiling are all still decided together, but
now ranked by an ordered sequence of tie-breaks instead: residency first, then fewest tiling
cuts, then parallelism, then division shape, then tile count, each one only resolving ties the
step before leaves open. Once that solve picks a tiling, the graph is physically rewritten to
apply it, and residency and division are solved jointly *again* over the now-tiled graph — a
second joint solve, not a continuation of the first, since tiling is no longer offered as a
choice at that point. Simulated annealing picks a tiling jointly with division too, through
`OpSplitSpace`, but keeps ranking by `cost_expr`, where a tiling pays only through the residency
it makes possible. Its chosen tilings are applied the same way, and its placement stands: there
is no second solve.

**Status: landed for both solvers, off by default; SA fixes still landing.** `CoreDivision.tiling`
/ `min_footprint` in `plan_solver.py` account for a tiling if one is present, and
`scratchpad/coarse_tiling.py`'s `CoarseTilingPass` — taking a `TileSpec` and lowering it to
`DimHint`s — applies the chosen tilings, both shared by the two solvers:

- **CP-SAT**: landed via `#4768` (`main` as of this update). Each op is offered a menu of
  output-axis tilings, each with its own division menu; a producer/consumer pair is only
  compatible if they agree on which slice of the shared buffer each tile and core owns.
  Consecutive ops sharing a loop nest form a loop group, and every producer/consumer edge
  inside one must be compatible. The solve ranks plans with the lexicographic ladder described
  above, not `cost_expr`. Once it picks a tiling, `CoarseTilingPass` applies it to the graph
  and the allocation is solved again over the now-tiled graph, with no further tilings
  offered — that second plan is the one committed.

- **Simulated annealing**: a 7-PR stack (`#4456` through `#4894`) adds this; the first five
  are merged. `OpSplitSpace`'s `TilingSpace` half offers each op its tilings; the **flip** move
  shifts a tiled run's boundary, and the **recolor** move (it floods a coordinated division
  change across a connected region of buffers) carries a tiling along with it. `#4602`
  applies the chosen tilings before the divisions are committed, keeping the anneal's
  placement; a tiling the apply refuses raises rather than falling back. `#4611` prices the
  HBM traffic of the full-extent companion buffer the apply mints for a tiled op whose output
  escapes its run. Still to come: `#4770` prices the loop a tiling introduces, through
  tile-count symbols in `cost_expr`, and lets the anneal place a staging copy for a tiled op's
  read across its run's boundary; `#4894` fixes the gaps between what the search may tile and
  what the apply builds that surface once tilings are applied, as compile failures or
  mis-scaled reads. Until `#4894` lands, keep `auto_coarse_tiling` off under simulated
  annealing. See [The apply round](sa_co_optimization.md#the-apply-round) for the mechanics.

`scratchpad_planning.md`'s "Current limitations"/"Future work" coarse-tiling items now point
here for current status, rather than carrying their own stale claim.

## Cost model

Both production solvers share one cost model (`torch_spyre/_inductor/cost_model.py`) — the
same `predict_by_bundle` / `cost_expr` referenced throughout the sections above. See
[Analytical Cost Model](cost_model.md) for the model in full: how a kernel is priced, every
`SPYRE_DUMP_COST` output shape, the measure-and-score workflow, and the current accuracy
table. This section covers only what's specific to the *co-optimizer's* use of it: how each
solver consumes the same expression differently, and how to recalibrate it against real
hardware.

### How it's integrated: symbolic (CP-SAT) vs. compiled-and-evaluated (SA)

[Symbolic cost formula](#symbolic-cost-formula) covered the shape of the search: build
`cost_expr` once, then score each candidate by substituting its guess and evaluating. The two
solvers differ in what "substitute and evaluate" compiles down to:

- **CP-SAT (symbolic all the way through)**: `_minimize_cost_expr`
  (`scratchpad/ilp_solver_ortools.py`) maps each sympy symbol straight onto a native OR-Tools
  decision variable (`sym_map[t.buffer.sym_is_lx.name] = t.in_buffer`, and similarly for
  `sym_core_divs`), then hands the whole expression, still unevaluated, to the constraint model
  as the objective to minimize via `_SympyExprToCpSat(...).convert(cost_expr)`. CP-SAT never
  evaluates `cost_expr` in Python; it searches the symbolic constraint space directly.
- **Simulated annealing (compiled, then evaluated numerically per move)**: `_build_score_fn`
  (`scratchpad/sa_cooptimizer.py`) instead compiles `cost_expr` once via `sympy.lambdify` into
  a plain Python function of `(chosen, resident)`, and calls that function fresh for every
  candidate state the annealer visits — each move gets a real number back, not a symbolic
  rewrite.

### Inspecting a co-optimization decision

Two useful flags:

- `SPYRE_DUMP_COST`: the same per-bundle breakdown [What it prints when
  turned on](cost_model.md#what-it-prints-when-turned-on) covers in full, with worked
  examples — it applies unchanged here, since the co-optimizer calls the same `explain()`. A
  human-readable report, meant for reading at the terminal rather than parsing.
- `SPYRE_DUMP_COST_EXPR_FILE=<path>`: a companion machine-readable dump, one JSON line per
  solve (`cost_expr_record` in `scratchpad/plan_solver.py`), recording the objective's
  per-bundle terms as lossless `sympy.srepr` strings, the symbol bindings the solver actually
  chose, every term evaluated under those bindings, and the alternative divisions each buffer
  could have taken instead — a self-describing record of one solve, meant to be read without
  needing to know which flags produced it. This one is specific to a *co-optimized* solve.

### Keeping the model honest: recalibrating against real hardware

[Measuring, scoring and refreshing the model](cost_model.md#measuring-scoring-and-refreshing-the-model)
is the real workflow this subsection builds on — read it first for the full `profile_ops.py` →
`run_cost_model_sweep.py` → `sweep_records.json` → `eval_model.py` pipeline, what
`sweep_plan.json`'s 1632 configurations cover, and the [Accuracy](cost_model.md#accuracy)
table (5.0% RMS for broadcast up to 29.8% for matmul-split). The co-optimizer adds nothing to
that pipeline itself — it's the same model, same database, same scorer — so what follows is
only the parts specific to using it from here: a visual complement to the RMS% table, and two
co-optimizer-specific gaps the measure/score loop doesn't by itself cover (whether a forced
hint is honored, and whether the sweep plan tracks what the co-optimizer actually needs).

`eval_model.py --plot <path.png>` is that visual complement: a log-log scatter of predicted
vs. measured `kernel_us`, one point per scored row, colored by category, with the 1:1 line and
the overall MSE/RMSE **in time units** annotated —

```bash
python3 tools/cost_model/eval_model.py --plot /tmp/before.png   # before a cost_model.py edit
# ... edit torch_spyre/_inductor/cost_model.py, or re-run with --params k=v overrides ...
python3 tools/cost_model/eval_model.py --plot /tmp/after.png
```

:::{figure} ../_static/images/lx/cost-model-pilot-scatter.png
:alt: Log-log scatter of predicted vs. measured kernel_us for a 20-row pilot sweep, colored by category, with a 1:1 reference line
:width: 600px
:align: center

A real (not illustrative) `--plot` output from a complete 20-configuration pilot sweep
(`run_cost_model_sweep.py --limit 20`, run to completion in ~8 minutes). Most categories sit
tight on the 1:1 line; the two visible outliers are `matmul_k` (`matmul_k_tiling`, a
coarse-K-tiled matmul), over-predicted by +137% in this run, and `softmax`
(`softmax_row_tiling`'s heavily-tiled config, 8 tiles on 2 cores), under-predicted by -82% —
exactly the kind of category- and config-specific signal this plot is for, that the aggregate
RMS% table alone would not point at directly.
:::

Read the two PNGs side by side: if a shift is confined to one category, check that category's
row in the RMS% table to confirm it. The scatter is log-log because the sweep's `kernel_us`
spans about three orders of magnitude (tens of µs for `transpose`/pointwise up to several ms
for forced-split matmuls); a linear scale would collapse every small kernel into the origin and
show nothing about them. `--plot` needs matplotlib, which is **not** a torch-spyre dependency
(base or optional) — install it separately; the flag fails with a clear message instead of a
traceback when it's missing.

**Does the sweep need to disable the co-optimizer, so a `WD_*`-forced configuration isn't
silently overwritten by the joint search?** No — see [Co-optimizer will honor `spyre_hint` for
work divisions](#co-optimizer-will-honor-spyre_hint-for-work-divisions). An op with no `WD_*`
override is intentionally left for the co-optimizer to decide, since those configurations
exist to measure the *default* co-optimized behavior.

**Does anything check that the requested config is actually what got measured?** It does now.
`sweep_records.json` already had `split_forced`/`split_actual` fields meant for exactly this,
but they were silently dead: the regex that filled them matched an older label format that
`profile_ops.py` no longer prints, so both fields parsed to `None` on every current-format row
without erroring. Fixed in `parse_sweep_logs.py` — new patterns match the current
`WD_M=../WD_N=../WD_K=..`-style label, and `split_actual` falls back to reading the matmul's
`matmul_m_split`/`matmul_n_split`/`reduction_cores` straight out of `feats` (populated on every
run) when the older debug-only `op_it_space_splits` text isn't present. The parser now also
warns on any row where the two disagree, so a hint that silently failed to take shows up as a
loud warning instead of a quietly wrong measurement.

#### Current sweep plan distribution

**Audit of `sweep_plan.json`'s own coverage.** The plan is generated by deduplicating
whatever configurations happen to already have measurements
(`run_cost_model_sweep.py`'s `_configs()`, driven by `_env_from_record`) — there is no
code anywhere that checks the result for balance, so today's distribution is an artifact
of what got measured historically, not a deliberate weighting choice. Measured directly
from `sweep_plan.json` (1632 configs total; `run_cost_model_sweep.py --dry-run` reports
1518 of those because `_SKIP_OPS` excludes `bmm_layout` — a non-default operand layout no
released build emits — and `bmm_3d2d_k_tiling`, quarantined after a real hardware incident
on 2026-08-07 rather than for being a bad config), grouped into
[Accuracy](cost_model.md#accuracy)'s own reporting categories:

  <table>
  <thead><tr><th>group</th><th>category</th><th>configs</th><th>% of 1632</th></tr></thead>
  <tbody>
  <tr><td rowspan="9"><b>matmul/bmm</b><br/>573 (35.1%)</td><td>matmul_split (<code>mmwd</code>)</td><td align="right">281</td><td align="right">17.2%</td></tr>
  <tr><td>bmm_split</td><td align="right">113</td><td align="right">6.9%</td></tr>
  <tr><td>matmul_row</td><td align="right">93</td><td align="right">5.7%</td></tr>
  <tr><td>bmm</td><td align="right">23</td><td align="right">1.4%</td></tr>
  <tr><td>matmul</td><td align="right">18</td><td align="right">1.1%</td></tr>
  <tr><td>matmul_k</td><td align="right">15</td><td align="right">0.9%</td></tr>
  <tr><td>matmul_nested</td><td align="right">13</td><td align="right">0.8%</td></tr>
  <tr><td>bmm_3d2d (skipped)</td><td align="right">9</td><td align="right">0.6%</td></tr>
  <tr><td>bmm_nested</td><td align="right">8</td><td align="right">0.5%</td></tr>
  <tr><td rowspan="2"><b>softmax</b><br/>138 (8.5%)</td><td>softmax</td><td align="right">113</td><td align="right">6.9%</td></tr>
  <tr><td>softmax_unrolled</td><td align="right">25</td><td align="right">1.5%</td></tr>
  <tr><td rowspan="2"><b>reduction</b><br/>206 (12.6%)</td><td>reduction (plain)</td><td align="right">155</td><td align="right">9.5%</td></tr>
  <tr><td>coarse_reduction</td><td align="right">51</td><td align="right">3.1%</td></tr>
  <tr><td colspan="2">broadcast</td><td align="right">226</td><td align="right">13.8%</td></tr>
  <tr><td colspan="2">transport</td><td align="right">172</td><td align="right">10.5%</td></tr>
  <tr><td colspan="2">pointwise</td><td align="right">168</td><td align="right">10.3%</td></tr>
  <tr><td colspan="2">other (incl. <code>bmm_layout</code>, skipped)</td><td align="right">149</td><td align="right">9.1%</td></tr>
  </tbody>
  </table>

  Summed by family, matmul/bmm+softmax together are 711/1632 (43.6%) — well under
  broadcast+transport+pointwise+reduction's combined 921/1632 (56.4%), so "matmul-biased"
  overstates it at the top level. The real skew is *within* the matmul family: `mmwd` alone
  (17.2%) outweighs every other single op by a wide margin, while five matmul/bmm
  sub-categories — `matmul`, `matmul_k`, `matmul_nested`, `bmm_3d2d`, `bmm_nested` — each
  rest on under 20 configs (8 to 18). Their rows in the Accuracy table are the ones to treat
  as statistically thin regardless of how important those ops are architecturally; a
  category needs enough configs to average out per-run noise (the pilot sweep above shows
  single-digit `kernel_us_cv` swings even within one config's 7 reps) before its RMS% number
  means much. Before leaning on the per-category RMS% numbers for a real decision, re-derive
  what coverage and weighting the model actually *needs* — per category, per shape range,
  per core count — and check the plan against that, rather than assuming today's
  distribution reflects anyone's intent. See [Ongoing work](#ongoing-work) for a concrete
  plan to act on this rather than just flagging it.

## Ongoing work

Open items from drafting this page that are worth resolving before calling it done:

### Cost model-related:

- A concrete plan for rebalancing `sweep_plan.json`, picking up the coverage audit above:
  1. Set a minimum-sample floor per category before trusting its RMS% number for a real
     decision — e.g. 20 configs as a starting bar (matching the under-20 threshold already
     flagged above). `matmul`, `matmul_k`, `matmul_nested`, `bmm_3d2d`, and `bmm_nested` fall
     below it today.
  2. Add targeted configs for each under-floor category, varying the dimension most likely to
     matter for that op (core count for `matmul`/`bmm` families; tile count for the
     `_k_tiling`/`_nested` variants) rather than just repeating the same shape — a config that
     doesn't vary anything new doesn't reduce the real uncertainty, only the per-run
     measurement noise.
  3. Measure the new configs: `python3 docs/source/user_guide/examples/profile_ops.py` for a
     one-off, or add them to a local copy of `sweep_plan.json` and run
     `run_cost_model_sweep.py --configs <path>` for a batch; `BENCH_EMIT_RECORDS=1` and
     `SPYRE_DUMP_COST=1` must both be set (the sweep driver already sets them) so the new
     rows carry `feats`, not just `io`.
  4. Fold the result back into the shipped plan: once new measurements exist in a local
     `sweep_records.json`, `run_cost_model_sweep.py --from-records --export-configs
     tools/cost_model/sweep_plan.json` regenerates the committed plan from every
     configuration that database now contains, including the new ones — this is also how any
     future solver-driven-coarse-tiling knob (the gap flagged below) would get its own sweep
     coverage once one exists.
  5. Re-check with `eval_model.py`: confirm the previously-thin categories' RMS% is now
     computed over a real sample, and `--plot` the new scatter against the old one the way
     [Keeping the model honest](#keeping-the-model-honest-recalibrating-against-real-hardware)
     already describes for a `cost_model.py` edit — the same before/after comparison applies
     to a sweep-plan change, since both move the measured population the model is scored
     against.
- New co-optimized axes need their own sweep configs, deliberately added. CP-SAT's
  solver-chosen coarse tiling (`#4768`, `config.auto_coarse_tiling`, described above) landed
  without one: checked directly against the landed commit, it touches neither `profile_ops.py`,
  `tools/cost_model/sweep_plan.json`, nor `eval_model.py` — the new tiling-candidate machinery
  has no sweep knob of its own yet, the same way `WD_M`/`WD_N`/`WD_K` is `mmwd`'s knob for
  forced work-division splits today. Its prediction accuracy is therefore unmeasured right now,
  not hypothetically — nothing in [Accuracy](cost_model.md#accuracy)'s table covers it. The
  same gap holds for simulated annealing's solver-chosen tiling (`#4456` through `#4894` —
  see [b) LX placement, work division, and coarse tiling](#b-lx-placement-work-division-and-coarse-tiling)),
  reachable behind the same flag since `#4602`. Combined with the point
  above (no automatic coverage check), a new optimization axis landing without a deliberate
  sweep-plan update is a real way for its prediction accuracy to go unmeasured indefinitely
  rather than loudly missing.
- Could the model tune itself? Today, fitting is manual: someone reads `eval_model.py`'s RMS%
  table and `--plot` scatter, forms a hypothesis, edits a constant or term in `cost_model.py`,
  and re-scores. Every constant in `CostParams` already carries its own fitting provenance in
  the module docstring (e.g. `alpha ~0.00574 ns/byte, calibrated so a balanced 1R+1W read
  lands at its measured ~105 GB/s effective`) — that record-keeping discipline is worth
  preserving in whatever replaces manual tuning. One direction: evolve the closed-form model
  into a small learned one (e.g. an MLP over the same `OpFeatures` fields) fit by SGD/backprop
  against `sweep_records.json`'s measured `kernel_us` directly, instead of hand-deriving each
  term. Worth weighing against the one related attempt already in the file's history: an
  empirical correction term for the `max(compute, memory)` under-charge was fitted and then
  removed rather than shipped because its fitted value proved unstable across populations
  (`cost_model.md`'s [Two limitations worth knowing](cost_model.md#two-limitations-worth-knowing)) —
  a learned model trades interpretable, individually-justified constants for exactly this kind
  of population-dependent fit, so the same instability risk would need to be addressed (e.g.
  held-out validation across categories, not just an aggregate RMS%) before trusting it over
  the current closed form.

### Tiling-related:

- SA's coarse-tiling stack (`#4456` through `#4894`) is 5 of 7 merged — update [b) LX
  placement, work division, and coarse tiling](#b-lx-placement-work-division-and-coarse-tiling)
  above (and `scratchpad_planning.md`'s pointers to it) again as `#4770` and `#4894` land.
- The existing docs use "tiling" for two different things — `CoreDivision.tiling` (coarse
  tiling proper, a `TileSpec`), and `scratchpad_planning.md`'s co-optimization section calling
  core-division splits for matmuls "the matmuls' tilings." The code no longer does:
  `sa_cooptimizer.py`'s `_flood_region`, whose division-menu index parameter was named `tiling`,
  now takes a `DivisionConfig`. Worth a terminology pass across the documents, so
  "division"/"split" and "tiling" stay consistently distinct.

### Solver performance:

- Nothing memoizes a co-optimizer solve across compiles. Every `torch.compile` that reaches
  `CoOptimizingAllocator` runs a full CP-SAT solve or SA anneal from scratch, even for a graph
  shape the search has already solved before. The caching that does exist
  (`torch_spyre/execution/kernel_cache.py`) is a different thing: its hash
  (`compute_specs_hash`) covers the *already-resolved* `OpSpec`/`LoopSpec` tree — the solver's
  output, not an input that could look a prior solve up — so it only skips recompiling an
  identical finished plan, never the search that produced it. A real "memorize the optimum"
  layer would need a cache keyed on something available *before* the solve (the graph's
  iteration-space shape and candidate menus, not the committed division), checked against a
  cache before `plan_layout_and_core_divisions` runs, with a defined policy for a
  near-but-not-exact shape match (reuse as a warm start vs. require an exact key). No such
  layer, warm-start or otherwise, exists today in `allocator.py`, `ilp_solver_ortools.py`, or
  `sa_cooptimizer.py`.

## Related documents

- [Analytical Cost Model](cost_model.md) — the model in full: how a kernel is priced, every
  `SPYRE_DUMP_COST` output shape, the measure-and-score workflow, and the current accuracy
  table
- [Scratchpad Planning](scratchpad_planning.md) — the allocator architecture, the solvers, LX
  eligibility rules, and the memory-hierarchy background
- [Work Division Planning](work_division_planning.md) — the standalone work-division planner
  and the candidate space the co-optimizer searches
- [Joint core-division + LX placement](sa_co_optimization.md) — the SA co-optimizer's search,
  objective, and test fixtures in full detail
- [Simulated Annealing Layout Planner](simulated_annealing_layout.md) — the placement-only
  annealer, a different class from the joint SA co-optimizer
- [Coarse-Tiling Loop IR for the Spyre Backend](coarse_tiling_loops.md) — how a tiled run is
  represented and survives scheduling and codegen, independent of who chooses the tiling
