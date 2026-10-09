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
- [What gets co-optimized](#what-gets-co-optimized)
  - [LX placement and work division](#a-lx-placement-and-work-division)
  - [LX placement, work division, and coarse tiling](#b-lx-placement-work-division-and-coarse-tiling)
- [Cost model](#cost-model)
- [Ongoing work](#ongoing-work)
- [Related documents](#related-documents)

## Why a co-optimizer

In the `torch-spyre` compiler pipeline, work division runs before LX planning, and its
objective is simply to pick the best parallelization scheme for each op *individually*. But
consider a simplified program with two memory-bound pointwise ops in a chain: the key
performance optimization is making sure op1's output stays on each core's own scratchpad for
op2 to read directly, rather than round-tripping through HBM. Because scratchpad is private per
core, the data core 0 holds in its scratchpad after op1 must be exactly what core 0 needs for
op2. If work division splits op1 along rows and op2 along columns, the data op2 needs on a
given core was never produced on that core — so op1's output has to spill to HBM, and every
core processing op2 pays the slower HBM fetch to get its share.

Solving work division and LX placement one at a time, freezing the earlier choice before the
later pass runs, leaves exactly this interaction on the table: either compute utilization is
sacrificed for memory locality, or memory locality is sacrificed and HBM traffic balloons. In
either case, joint assignments that *reconcile* several ops onto one shared division — one
that keeps everyone's compute shape good **and** keeps the shared buffer resident — are missed
simply because the earlier pass already committed. The co-optimizer exists to search that
combined space instead of two disjoint ones.

## How it works in general

In a nutshell, the co-optimizer uses a cost model to estimate two things per candidate: the
runtime of matmul compute (the only op type with its own compute-roofline term; a dependent
fused reduction riding in the same bundle as a matmul gets a smaller additive correction, and
every other op is treated as pure memory traffic) and the data-transfer time implied by where
each buffer lives (HBM or LX), using known or measured bandwidth for the target hardware.
Rather than committing to one work division per op up front, it evaluates several combinations
together and prices the communication cost that each one implies, then commits only the
winning combination at the end of the search. A few different solver backends can drive this
search; the sections below cover each in detail, and [Cost model](#cost-model) covers the
model itself — how it plugs into the solvers, how to inspect it, and how to recalibrate it.

The co-optimizer is reached through `CoOptimizingAllocator`
(`torch_spyre/_inductor/scratchpad/allocator.py`), gated by `config.co_optimizing_lx_planning`
(env `CO_OPTIMIZING_LX_PLANNING`, on by default) and selected via `config.layout_solver`. It
runs as a **pre-scheduling pass** — `V.graph` is live but `V.graph.scheduler` is still `None`,
so fusion hasn't happened yet and anything the engine needs to know about the kernels its
decisions will land in has to be *estimated* from the flat, ordered operation list rather than
read off the real fused grouping.

### The simplest version: exhaustive search over greedy placement

Before getting into the production solvers, it's worth seeing the simplest possible instance
of "co-optimize divisions and placement jointly," because it makes the shape of the problem
concrete without any annealing or constraint-solving machinery to explain first.

`ExhaustiveSearchSolver` (`torch_spyre/_inductor/scratchpad/exhaustive_search.py`) wraps a
plain, placement-only solver — most simply `GreedyLayoutSolver`, which otherwise only ever
places buffers against divisions some earlier pass already fixed — and brute-forces the other
half of the problem around it. For every buffer that has more than one candidate division, it
tries every combination (an exhaustive DFS, bounded by `K^N` leaves over the `N` buffers that
actually have a choice), and at each leaf:

1. Builds a fresh set of per-core buffer sizes implied by that combination of chosen divisions.
2. Hands them to a brand-new instance of the wrapped placement solver (a solver is single-use)
   and calls `plan_layout()`.
3. Scores the leaf by summing the total size of every buffer the inner solver could *not* pin
   to LX — the same differential-HBM-traffic idea the production solvers' memory-only
   fallback uses.

The combination with the lowest score wins; its divisions are committed to every buffer, and
the wrapped solver runs once more to produce the final addresses.

For example: a producer with 2 candidate divisions feeding a consumer with 2 of its own is a
`2×2 = 4`-leaf DFS. Say only one of those four pairings avoids a `core_div_mismatch` between
them (`cd_parent_matches` — the same relation the production solvers use to decide which
combinations even make a buffer LX-eligible, not a check bolted on separately here). The three
mismatched leaves each spill one buffer to HBM and score accordingly; the matching leaf scores
zero (both pin). `ExhaustiveSearchSolver` tries all four, keeps the zero-scoring one, and
commits its divisions.

This also shows why it's not what ships by default: scoring a leaf means re-running the real
placement solver from scratch, so the cost is exponential in the number of buffers with a
real choice. `select_allocator()` only reaches for `ExhaustiveSearchSolver` as a fallback —
when `co_optimizing_lx_planning` is set but the configured `layout_solver` has no
purpose-built core-division-capable solver to co-optimize with (i.e. anything other than
`"cpsat"` with `ortools` installed, or `"simulated_annealing"`) — and only when the caller has
opted in via `config.allow_exhaustive_search` (env `ALLOW_EXHAUSTIVE_SEARCH`); otherwise it
raises rather than silently paying that cost. The two solvers below exist to make the general
case affordable for the common case; the rest of this section covers how the fallback itself
is kept affordable when it does run.

**Pruning the DFS's candidate menu: `_enum_split_options`.** The example above assumed each
buffer's menu was already small (2 candidates). In practice `select_allocator()` always pairs
`ExhaustiveSearchSolver` with `CoOptimizingAllocator(prune=True)`, which controls exactly that
menu: instead of `enumerate_work_division_candidates()`'s full, standalone-work-division-planner
cross product, `_enum_split_options` (`allocator.py`) builds a much smaller, heuristic
candidate list per op, dispatching on op type — the difference between a `2×2`-leaf DFS and
one with every legal division for every buffer. CP-SAT and SA never set `prune=True` — this
pruning exists specifically to keep *this* DFS's exponential cost affordable, not as a third
solving strategy alongside them.

:::{figure} ../_static/images/lx/co-optimization.svg
:alt: Co-optimization searches over alternative split assignments, scoring each by HBM bytes left unpinned
:width: 700px
:align: center

The pruned search enumerates split variants per op, scores each combination by counting HBM
bytes the solver could not pin, and commits the winning assignment back before the standard
allocator flow.
:::

Generated alternatives are deduped by canonical key and filtered through `_split_fits_sticks`,
which rejects factors that overflow a stickified dim's stick count (those would abort the
SuperDSC bundler) or that land on a collapsed/broadcast dim. The upstream seed is always kept
and is already stick-valid from work division; every other candidate must still satisfy hard
work-division constraints (blocked axes stay unsplit, split domains restrict legal factors).

- **Pointwise ops** get their seed, dim-flip variants (move the seed's single output-dim
  factor onto each compatible alternative output dim, bounded by `DEFAULT_VARIANT_CAP = 6`),
  and the matmul tilings from the shared pool (below). Adopting a neighbouring matmul's tiling
  makes the op's per-core view match the matmul's, so a shared buffer pins to LX *and* the op
  runs at the matmul's high-utilization shape.
- **Matmul splits are not overridden onto a single dim, but neighbours' tilings and a
  batch-major split are offered.** Concentrating a balanced `M/4×N/8` split onto one dim
  (`M/32`) pins the matmul output and the surrounding chain to LX but is a poor matmul shape:
  on `mlp-linear-kn.t` (`SENCORES=32`) it regressed kernel time ~2.5× as process-engine
  utilization fell from 66% to 33%. So the rule remains **prioritize compute utilization for
  compute-bound ops**: the seed split is never flipped onto one dim. Instead,
  `_check_and_add_matmul_option` offers each matmul its seed plus (a) every *other* matmul's
  split transferred into this op's coordinates by axis role (so two matmuls whose
  work-division splits disagree can find a consistent assignment), and (b) a factored
  batch-major `B/M` split. All of these are full-core splits, so compute utilization is
  preserved.
- **Batch-major `B/M` tiling reconciles attention.** Two attention matmuls (`Q·Kᵀ` and
  `scores·V`) contract different axes, so neither can adopt the other's `N`/`K` tiling, but
  both keep the batch (`B`) and `M` output axes. `_factored_bm_splits` emits a single full-core
  `B/b · M/m` split (largest batch factor that fits, from `(8, 4, 2)` with `m = ncores / b`),
  valid for both matmuls and divisible into both stick-count extents. This shared tiling is
  also offered to the **softmax reductions** (`max`/`sum`) in their own output coordinates via
  `_reduction_bm_axes`. Reductions are otherwise left on their seed, but offering them the
  `B/M` split lets the whole softmax chain between the two matmuls reconcile to one tiling. On
  `mha_4h` (`SENCORES=32`) this converges both matmuls and the entire softmax chain on
  `B/4·M/8`, pinning the scores matrix and the chain to LX. Reductions are not given dim-flip
  variants (their reduced axis is fixed), and any candidate that fails to reconcile a shared
  buffer's per-core view self-eliminates during scoring.

The shared matmul-tiling pool is collected once by `_find_distinct_matmul_splits`: each
distinct matmul seed split plus each matmul's factored `B/M` split, deduped. This pool seeds
both the pointwise candidate lists and the cross-matmul transfer.

On `mlp-linear-kn.t` (`SENCORES=32`) the pointwise-seeding path lifted process-engine
utilization from ~66% to ~79% and cut fused kernel time by ~17% (about 2× faster than the
sendnn reference).

The leaf-scoring function itself is the same one described above: it runs the full
`_generate_buffers + plan_layout` pass on the candidate splits and counts the HBM bytes of
every buffer the solver could not pin. Repeated `_per_core_view_on_buf` work is memoized
across leaves, and the split-invariant liveness / filtered-op-view / mem-usage computations
are hoisted out of the per-leaf path.

The factored-`B/M` and cross-matmul transfer options are marked TEMP/TODO: the intent is for
work division to assign consistent splits directly, at which point these compensating options
can be removed. See [Co-optimization is still limited](scratchpad_planning.md#co-optimization-is-still-limited)
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
  menu index 0 with a FirstFit placement, then runs one geometric cool for
  `clamp(40n, 200, 15000)` steps using three move types — **reorder** (re-place a buffer in `pi`,
  weight 0.5), **flip** (change one buffer's division, weight 0.3), and **recolor** (flood a
  coordinated division change across a connected region via the `cd_parent_matches`
  compatibility relation, weight 0.2) — and returns the best state seen, so the result is never
  worse than the seed. See [Joint core-division + LX placement](sa_co_optimization.md) for the
  full mechanics.

**What "co-optimizing" means concretely**: both solvers are driven by a *single* cost
expression (`cost_expr`) built once, up front, from the whole graph's predicted runtime —
`predict_by_bundle(graph.operations, op_features, ...)`, a symbolic expression over every
buffer's `sym_is_lx` (is it LX-resident) and `sym_core_divs` (which division was chosen).
Because both kinds of decision are symbols in the *same* expression, a move that changes one
automatically re-prices the other — there's no separate objective per variable to reconcile
after the fact. That's what distinguishes this from running two independent passes in
sequence. When no cost expression is usable (symbolic shapes, serialized captures, or a
construction error), both solvers fall back to a cheaper **memory-only** objective: pure
differential HBM traffic, where a resident buffer costs zero and only spills are summed. This
fallback is weaker — on a graph where everything already fits, every division scores
identically and the search has nothing left to optimize on.

Without `co_optimizing_lx_planning`, each solver can still be selected on its own, but then it
only *places* buffers against divisions some earlier pass already fixed — it never searches
divisions itself. The joint behavior is what `CoOptimizingAllocator` adds on top.

## What gets co-optimized

**Controlling it**: two flags gate everything below, for both subsections.

- `config.co_optimizing_lx_planning` (env `CO_OPTIMIZING_LX_PLANNING`, **on** by default):
  the on/off switch. Off, every variable here goes back to being decided by its own
  standalone pass, nothing joint.
- `config.layout_solver` (env `LAYOUT_SOLVER`, default `"cpsat"`): which solver runs the
  joint search. Options are `"cpsat"`, `"simulated_annealing"`, `"greedy"`, `"firstfit"`, and
  `"bestfit"` — but only `"simulated_annealing"`, and `"cpsat"` *with `ortools` installed*,
  are natively core-division-capable and reach a purpose-built joint solver directly.

Every other case — `"greedy"`, `"firstfit"`, `"bestfit"`, or `"cpsat"` without `ortools` —
has no core-division-capable solver to co-optimize with, so `select_allocator()` **rejects it
by default**: it raises `ValueError` unless `config.allow_exhaustive_search` (env
`ALLOW_EXHAUSTIVE_SEARCH`) is also set, in which case it falls back to the
`ExhaustiveSearchSolver` DFS from
[The simplest version](#the-simplest-version-exhaustive-search-over-greedy-placement) above.
So `layout_solver="greedy"` with `co_optimizing_lx_planning` left on doesn't just place
buffers once `allow_exhaustive_search` is set — it drives that DFS over `greedy`'s own
division menu. **That DFS is exponential** (`K^N` leaves over the `N` buffers with a real
choice) **and has no size cap or timeout of its own** — the reject-by-default behavior above
is the only guard, so only opt in on a task small enough that the exponential blow-up is
acceptable, not as a general substitute for CP-SAT or SA.

### a) LX placement and work division

**What it is**: this pair is the baseline joint search — neither variable alone is a
"co-optimization" (that needs at least two things decided together), so this is the smallest
real instance of what the rest of this page describes.

- **LX placement** is where each buffer's resident copy lives in LX — represented as
  `address: Optional[int]` on `LifetimeBoundBuffer`; `None` after solving means the buffer
  spilled to HBM. Eligibility is decided once, declaratively, in
  `ScratchpadAllocator._residency_reasons`, and carried to every solver as a single
  `residency_reason` field (`None` = may be pinned, any string = may not) — see
  [Declarative exclusion](scratchpad_planning.md#declarative-exclusion). The actual placement
  search (which address, in what order) is `pi` in the SA solver and the 2D no-overlap packing
  in CP-SAT. Implemented, and the variable the whole scratchpad-planning pass exists to solve —
  see [Scratchpad Planning](scratchpad_planning.md) for the full allocator architecture, and
  note that the standalone, placement-only `SimulatedAnnealingLayoutSolver` (see
  [Simulated Annealing Layout Planner](simulated_annealing_layout.md)) is a *different* class
  from the joint `SaCoOptimizingSolver` — it only ever moves `pi`, never the division vector
  `W` below.
- **Work division** is how many cores an op's output or reduction uses, and which slice of the
  iteration space each core owns — one split factor per iteration-space symbol. Splitting an
  **output** dim gives each core a disjoint output slice; splitting a **reduction** dim gives
  each core a partial result that must be combined. See
  [Work Division Planning](work_division_planning.md) for the standalone 3-pass planner
  (span-reduction, cost-model matmul split, default distribution) that picks this when the
  co-optimizer isn't driving it. As a co-optimized variable, the joint solvers don't re-derive
  divisions; they search over the *same* candidate space the standalone planner would offer,
  pre-enumerated per op by `enumerate_work_division_candidates()` (`work_division.py`) and
  attached to each buffer as a menu of `CoreDivision` candidates
  (`CoreDivisionBuffer.core_divisions`, `plan_solver.py`). In the SA solver this menu index is
  exactly the `W` half of the `(pi, W)` state; the **flip** move changes one buffer's chosen
  index and resizes its per-core footprint, and **recolor** propagates a compatible change
  across a connected region. In CP-SAT, the menu becomes a decision variable alongside
  placement in the same constraint model.

**Controlling it**: just the two base flags above — both `"cpsat"` and `"simulated_annealing"`
reach this pair the same way, with no extra switch needed.

**Status: implemented and tested together** — e.g.
`test_a_flip_that_shrinks_a_footprint_into_capacity_raises_the_count` exercises exactly the
scenario the joint search exists for: a buffer that doesn't fit undivided becomes eligible
once a flip halves its footprint, moving both variables' state at once. The cost expression
also prices division choice directly — `test_core_division_symbol_drives_the_score` shows the
predicted score scaling with the number of cores a division uses, confirming it feeds the
runtime term, not just a memory-fit heuristic.

### b) LX placement, work division, and coarse tiling

**What it is**: cutting an op's iteration space into sequential tile runs — a `TileSpec` of
`TileAxis` entries (output and/or reduction axes, each with a count) — so a working set that
doesn't fit LX at full size can fit one tile at a time. It's a different lever from work
division: division splits the iteration space *across cores at the same time*; coarse tiling
splits it *across time, on the same core(s)*. The underlying loop-IR mechanism — how a tiled
run is represented and survives scheduling and codegen — is implemented and documented in
[Coarse-Tiling Loop IR for the Spyre Backend](coarse_tiling_loops.md); that document also
points to the design RFC. What's described here is specifically the *solver's* ability to
choose a coarse tiling as part of the joint search, which is a separate, newer effort.

**Controlling it**: on top of the two base flags above (which still have to select CP-SAT —
`layout_solver == "cpsat"` — for any of this to apply), `config.auto_coarse_tiling` (env
`AUTO_COARSE_TILING`, off by default) is the dedicated switch for adding coarse tiling to the
pair in (a). It's inert with `layout_solver` set to `"simulated_annealing"` or anything else.

**Status: landed for CP-SAT, still in progress for simulated annealing.** The apply-side
plumbing is landed for both solvers — `scratchpad/coarse_tiling.py` takes a `TileSpec` and
lowers it to `DimHint`s, and `CoreDivision.tiling` / `min_footprint` in `plan_solver.py`
already account for a tiling if one is present:

- **CP-SAT**: landed via `#4768` (`main` as of this update).

  - **Candidates.** An op is offered the output-axis tilings `enumerate_tile_options` finds:
    never the stick dim, never a reduction axis, never an axis one of its reads repeats along
    (the repeated dim of `x.repeat`, whose tiles would have to wrap back over `x`), and none
    that leave a per-core read over the read-distance limit. Ops a `spyre_hint` or
    `for_each_tile` loop already tiles, every op inside a `for_each_tile` region, restickifies
    and mutations are offered only the untiled option. Each tiling gets its own division menu,
    enumerated on the per-tile frame.
  - **Matching.** A producer/consumer pair of divisions is compatible when the two agree on
    core ownership and on tile ownership of the buffer they share, both taken on the untiled
    buffer: tile `t` must touch the same slice on both sides. `TileSpec` equality is not the
    test — `host_dim` is positional in each op's own output, so equal specs can tile different
    dims of a shared buffer (a permuted or reducing consumer), and unequal specs the same one.
    A consumer that reads the buffer more than once has to agree through every read: `a +
    a.permute(1, 0, 2)` pairs with `a` only under a division or tiling on a dim both reads walk
    alike. A buffer that may not live in LX gets no compatible pairs; tiling exists to keep
    buffers in LX, so its producer and consumer never share a nest.
  - **Loop groups.** Consecutive ops that run the same loop nest (the same trip count at each
    level) share a loop group. The solve requires every producer/consumer edge inside a group
    to be a compatible pair, and `CoarseTilingPass` checks each such edge again before applying
    the tiling.
  - **Objective.** The cost expression itself is not used, since it has no term for tile size
    or loop-group boundaries — the solve instead ranks plans lexicographically: LX residency,
    then *cuts* (tiled ops whose value must be copied out of their nest, for a consumer outside
    it or as a graph output), then parallelism, division shape, and last, fewest tiles.
  - **Materialize and re-plan.** When the solve picks any tiling, `CoarseTilingPass` applies it
    and the allocation is solved again over the tiled graph with no further tilings offered;
    that second plan is the one committed. A `SolveError` from the first solve falls back to
    greedy placement over the untouched graph, and one from the second solve over the tiled
    graph.

- **Simulated annealing**: still landing. A 7-PR stack (`pr-4456` through `pr-4894`) extends
  `SaCoOptimizingSolver`'s division representation to carry a tiling choice and extends the
  **recolor** move to propose tiling changes, with companion-buffer copy-out costs (for data
  that escapes a tile run) priced into the cost model along the way. `main`'s
  `sa_cooptimizer.py` does not yet carry this.

`scratchpad_planning.md`'s "Current limitations" still lists "No coarse-tiling integration"
and its "Future work" still lists "Joint operation with the `coarse_tiling` pass" — both now
stale given CP-SAT's landing (`#4768` added the "Solver-driven coarse tiling" section without
removing them), worth flagging to whoever maintains that page rather than silently
contradicting it here.

## Cost model

Both production solvers share one cost model (`torch_spyre/_inductor/cost_model.py`) — the
same `predict_by_bundle` / `cost_expr` referenced throughout the sections above. See
[Analytical Cost Model](cost_model.md) for the model in full: how a kernel is priced, every
`SPYRE_DUMP_COST` output shape, the measure-and-score workflow, and the current accuracy
table. This section covers only what's specific to the *co-optimizer's* use of it: how each
solver consumes the same expression differently, and how to recalibrate it against real
hardware.

### How it's integrated: symbolic (CP-SAT) vs. compiled-and-evaluated (SA)

The two production solvers consume the *same* `cost_expr` — one sympy expression built once,
up front, over every buffer's `sym_is_lx` and `sym_core_divs` symbols — but bind it to a chosen
candidate in two different ways:

- **CP-SAT (symbolic all the way through)**: `_minimize_cost_expr`
  (`scratchpad/ilp_solver_ortools.py`) maps each sympy symbol straight onto a native OR-Tools
  decision variable —

  ```python
  sym_map[t.buffer.sym_is_lx.name] = t.in_buffer
  for key, symbol in t.buffer.sym_core_divs.items():
      sym_map[symbol.name] = t.cp_core_divs[key]
  ...
  cp_cost = _SympyExprToCpSat(model, sym_map, buffer_map).convert(cost_expr)
  ```

  — then hands the *whole expression*, still unevaluated, to the constraint model as the
  objective to minimize. CP-SAT never evaluates `cost_expr` in Python; it searches the
  symbolic constraint space directly.

- **Simulated annealing (compiled, then evaluated numerically per move)**: `_build_score_fn`
  (`scratchpad/sa_cooptimizer.py`) instead resolves every symbol to a plain Python callable of
  `(chosen, resident)` —

  ```python
  value_of[buf.sym_is_lx] = lambda chosen, resident, name=buf.name: (
      1 if name in resident else 0
  )
  ...
  fn = sympy.lambdify(free, cost_expr, modules="math")
  ```

  — compiles `cost_expr` once via `sympy.lambdify` into a fast numeric function, and calls that
  function fresh for every candidate state the annealer visits. Each move gets a real number
  back, not a symbolic rewrite.

Same model, same expression, two different consumption strategies: one treats it as a
constraint to satisfy exactly, the other as a function to sample cheaply, many times, during a
heuristic search.

### Inspecting a co-optimized solve

[What it prints when turned on](cost_model.md#what-it-prints-when-turned-on) covers
`SPYRE_DUMP_COST`'s per-bundle breakdown in full, with worked examples; it applies unchanged
here since the co-optimizer calls the same `explain()`. The one thing specific to a *co-optimized*
solve is `SPYRE_DUMP_COST_EXPR_FILE=<path>`: a companion machine-readable dump, one JSON line
per solve (`cost_expr_record` in `scratchpad/plan_solver.py`), recording the objective's
per-bundle terms as lossless `sympy.srepr` strings, the symbol bindings the solver actually
chose, every term evaluated under those bindings, and the alternative divisions each buffer
could have taken instead — a self-describing record of one solve, meant to be read without
needing to know which flags produced it.

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

Read the two PNGs side by side: points moving toward the 1:1 line is an improvement, points
moving away in one category only is a regression confined to that category (check that
category's row in the RMS% table to confirm), and a cloud that barely shifts between the two
images alongside a near-identical RMSE is a no-op change. The scatter is log-log because the
sweep's `kernel_us` spans about three orders of magnitude (tens of µs for `transpose`/pointwise
up to several ms for forced-split matmuls); a linear scale would collapse every small kernel
into the origin and show nothing about them. `--plot` needs matplotlib, which is **not** a
torch-spyre dependency (base or optional) — install it separately; the flag fails with a clear
message instead of a traceback when it's missing.

**Does the sweep need to disable the co-optimizer, so a `WD_*`-forced configuration isn't
silently overwritten by the joint search?** No. A forced split uses `spyre_hint(work_div=...)`
(see [How it's integrated](#how-its-integrated-symbolic-cp-sat-vs-compiled-and-evaluated-sa)
above for what a hint actually does to the solver's input), and `allocator.py`'s division-menu
builder special-cases a resolved hint before either solver ever runs: it collapses that op's
candidate list to the single fixed division the hint requested
(`_legal_fixed_division`, `reason = "user work_div hint"`), so there is nothing left for
CP-SAT or SA to search over for that op. Confirmed empirically, not just read in code: a pilot
sweep's `bmm_wd` row requesting `WD_B=1 WD_K=1 WD_M=8 WD_N=4` produced a compiled kernel whose
dumped `MODEL FEATS` carry exactly `cores=32, matmul_m_split=8, matmul_n_split=4` — the forced
split, unperturbed. An op with no `WD_*` override is intentionally left for the co-optimizer to
decide, since those configurations exist to measure the *default* co-optimized behavior.

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

#### Open items on this measure/score/plot loop

- **Audit `sweep_plan.json`'s own coverage.** The plan is generated by deduplicating
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
- **New co-optimized axes need their own sweep configs, deliberately added.** CP-SAT's
  solver-chosen coarse tiling (`#4768`, `config.auto_coarse_tiling`, described above) landed
  without one: checked directly against the landed commit, it touches neither `profile_ops.py`,
  `tools/cost_model/sweep_plan.json`, nor `eval_model.py` — the new tiling-candidate machinery
  has no sweep knob of its own yet, the same way `WD_M`/`WD_N`/`WD_K` is `mmwd`'s knob for
  forced work-division splits today. Its prediction accuracy is therefore unmeasured right now,
  not hypothetically — nothing in [Accuracy](cost_model.md#accuracy)'s table covers it. The
  same gap is coming for simulated annealing's stack (`pr-4456` through `pr-4894`) once it
  lands too. Combined with the point above (no automatic coverage check), a new optimization
  axis landing without a deliberate sweep-plan update is a real way for its prediction accuracy
  to go unmeasured indefinitely rather than loudly missing.
- **Could the model tune itself?** Today, fitting is manual: someone reads
  `eval_model.py`'s RMS% table and `--plot` scatter, forms a hypothesis, edits a constant
  or term in `cost_model.py`, and re-scores. Every constant in `CostParams` already carries
  its own fitting provenance in the module docstring (e.g. `alpha ~0.00574 ns/byte,
  calibrated so a balanced 1R+1W read lands at its measured ~105 GB/s effective`) — that
  record-keeping discipline is worth preserving in whatever replaces manual tuning. One
  direction: evolve the closed-form model into a small learned one (e.g. an MLP over the
  same `OpFeatures` fields) fit by SGD/backprop against `sweep_records.json`'s measured
  `kernel_us` directly, instead of hand-deriving each term. Worth weighing against the one
  related attempt already in the file's history: an empirical correction term for the
  `max(compute, memory)` under-charge was fitted and then **removed rather than shipped**
  because its fitted value proved unstable across populations (`cost_model.md`'s
  [Two limitations worth knowing](cost_model.md#two-limitations-worth-knowing)) — a learned
  model trades interpretable, individually-justified constants for exactly this kind of
  population-dependent fit, so the same instability risk would need to be addressed (e.g.
  held-out validation across categories, not just an aggregate RMS%) before trusting it
  over the current closed form.
- **Nothing memoizes a co-optimizer solve across compiles.** Every `torch.compile` that
  reaches `CoOptimizingAllocator` runs a full CP-SAT solve or SA anneal from scratch, even
  for a graph shape the search has already solved before. The caching that does exist
  (`torch_spyre/execution/kernel_cache.py`) is a different thing: its hash
  (`compute_specs_hash`) covers the *already-resolved* `OpSpec`/`LoopSpec` tree — the
  solver's output, not an input that could look a prior solve up — so it only skips
  recompiling an identical finished plan, never the search that produced it. A real
  "memorize the optimum" layer would need a cache keyed on something available *before* the
  solve (the graph's iteration-space shape and candidate menus, not the committed division),
  checked against a cache before `plan_layout_and_core_divisions` runs, with a defined
  policy for a near-but-not-exact shape match (reuse as a warm start vs. require an exact
  key). No such layer, warm-start or otherwise, exists today in `allocator.py`,
  `ilp_solver_ortools.py`, or `sa_cooptimizer.py`.

## Ongoing work

Open items from drafting this page that are worth resolving before calling it done:

- CP-SAT's coarse-tiling stack landed first (`#4768`), updating [c) Coarse
  tiling](#c-coarse-tiling) above accordingly. SA's stack (`pr-4456` through `pr-4894`) is
  still pending — update that section again once it lands rather than leaving CP-SAT's
  landed/SA's pending split stale the way `scratchpad_planning.md`'s own "Current
  limitations"/"Future work" lists were left after `#4768` (flagged above).
- The existing docs *and the current code* use "tiling" for at least two different things —
  `CoreDivision.tiling` (coarse tiling proper, a `TileSpec`), and, separately, (a)
  `scratchpad_planning.md`'s co-optimization section calling core-division splits for matmuls
  "the matmuls' tilings," and (b) `sa_cooptimizer.py`'s own `_flood_region(anchor, tiling)` and
  `_rng.choice(self._nontrivial_menu[anchor])`, where `tiling` names a **division-menu index**,
  not a `TileSpec` — confirmed on `upstream/main` as of this draft (`sa_cooptimizer.py:519-581`).
  So the naming collision isn't just a doc-wording slip; it's in a parameter name in the
  pre-coarse-tiling code the SA stack is extending. Worth a terminology pass across the
  documents *and* that code once coarse tiling lands, so "division"/"split" and "tiling" stay
  consistently distinct.
- **A concrete plan for rebalancing `sweep_plan.json`**, picking up the coverage audit above:
  1. **Set a minimum-sample floor per category** before trusting its RMS% number for a real
     decision — e.g. 20 configs as a starting bar (matching the under-20 threshold already
     flagged above). `matmul`, `matmul_k`, `matmul_nested`, `bmm_3d2d`, and `bmm_nested` fall
     below it today.
  2. **Add targeted configs for each under-floor category**, varying the dimension most
     likely to matter for that op (core count for `matmul`/`bmm` families; tile count for
     the `_k_tiling`/`_nested` variants) rather than just repeating the same shape — a
     config that doesn't vary anything new doesn't reduce the real uncertainty, only the
     per-run measurement noise.
  3. **Measure the new configs**: `python3 docs/source/user_guide/examples/profile_ops.py`
     for a one-off, or add them to a local copy of `sweep_plan.json` and run
     `run_cost_model_sweep.py --configs <path>` for a batch; `BENCH_EMIT_RECORDS=1` and
     `SPYRE_DUMP_COST=1` must both be set (the sweep driver already sets them) so the new
     rows carry `feats`, not just `io`.
  4. **Fold the result back into the shipped plan**: once new measurements exist in a local
     `sweep_records.json`, `run_cost_model_sweep.py --from-records --export-configs
     tools/cost_model/sweep_plan.json` regenerates the committed plan from every
     configuration that database now contains, including the new ones — this is also how
     any future solver-driven-coarse-tiling knob (the gap flagged above) would get its own
     sweep coverage once one exists.
  5. **Re-check with `eval_model.py`**: confirm the previously-thin categories' RMS% is now
     computed over a real sample, and `--plot` the new scatter against the old one the way
     [Keeping the model honest](#keeping-the-model-honest-recalibrating-against-real-hardware)
     already describes for a `cost_model.py` edit — the same before/after comparison applies
     to a sweep-plan change, since both move the measured population the model is scored
     against.

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
