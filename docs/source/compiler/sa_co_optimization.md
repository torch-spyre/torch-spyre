# Joint core-division + LX placement (the SA co-optimizer)

`SaCoOptimizingSolver` decides two things at once: how each buffer's work is divided across
cores, and where the resulting per-core buffers live in the LX scratchpad. The two are coupled —
a finer division shrinks a buffer's per-core footprint, which changes what fits in LX, which
changes whether the division was worth taking — so solving them separately leaves the
interaction on the table.

For the placement-only annealer (a *different* class, with its own schedule) see
[Simulated Annealing Layout Planner](simulated_annealing_layout.md). For the surrounding
allocator and the other solvers, see [Scratchpad Planning](scratchpad_planning.md).

## Where it sits

`config.layout_solver = "simulated_annealing"` with `co_optimizing_lx_planning` routes to
`CoOptimizingAllocator(layout_planning=SaCoOptimizingSolver)`. The allocator builds one
`CoreDivisionBuffer` per graph buffer — carrying the candidate division menu, the
`cd_parent_matches` compatibility relation, and, where it could derive them, the per-candidate
machinery that stands in for both (`division_space` and `residency_edges`) — and hands the list to
the solver, which mutates it in place with a `chosen_division` and an `address`.

It runs as a **pre-scheduling pass**: `V.graph` is live but
`V.graph.scheduler` is still `None`, so fusion has not happened yet. Anything the engine wants to
know about the kernels its decisions will land in has to be *estimated* from the ordered
operation list (see [Bundles](#bundles-are-estimates)).

The engine takes no options. The buffers, the capacity and the alignment are its whole interface;
the search parameters are module constants in `sa_cooptimizer.py`.

## The search

The state is the pair `(pi, W)`: the layout permutation `pi`, held in a composed
`NativePermutationLayoutSolver` packer, and the division vector `W`, one `DivisionConfig` per
buffer. A config is a division as a *value* — the `CoreDivision` itself, a canonical hashable key
identifying the choice it makes, and the menu position it came from, if any. The seed is every
buffer at its first candidate with `pi` from a FirstFit pass. One geometric cool runs
`clamp(40n, 200, 15000)` steps at fixed proposal weights, and the best state seen is what gets
written back — so the result is never worse than the seed.

### Where the candidates come from

Each buffer gets a `_DivisionSource`, and the engine asks nothing else: the seed, the divisions one
step away (`neighbours`, what a flip proposes), and a splitting division to flood from
(`anchor`, what a recolor proposes). A buffer whose producing op has an `OpSplitSpace` *generates*
those (`_GeneratedDivisions`); the rest read the enumerated menu (`_MenuDivisions`). Each
producer→consumer edge likewise gets an `_EdgeRelation`, which the residency gate and the recolor
flood ask: computed per candidate off the buffer's geometry where both ends generate
(`_ViewRelation`, which inverts the per-core view to construct the other end's division), and the
`cd_parent_matches` table projected onto choices otherwise (`_TableRelation`). The one thing still
keyed by menu position is the `chosen_division` written back; a generated division the menu does
not carry is appended to it then — which is the path a tiled division always takes, since the
enumeration carries no tilings.

### The coarse tiling rides on the same candidate

`CoreDivision.tiling` is a `TileSpec`, and `OpSplitSpace` chooses it jointly with the splits. It has
to be joint: tiling rewrites index expressions and `splits_by_index_coeff` keys the output splits by
each symbol's coefficient in the write index, so a `CoreDivision` carried across tilings is
uninterpretable rather than merely illegal. The tiling half of the space is `TilingSpace`
(`wsr/enumerate_tilings.py`), whose predicates `enumerate_tile_options` is the cross product over.

The space is **ragged**. Each tiling is judged in its own per-tile frame, the context
`work_division_context_for_op(op, max_cores, tiling)` builds, as the CP-SAT menu is. A tile level
cuts its axis's extent, so a core split of that axis must divide the smaller extent; it cuts the
per-core span too, so `MAX_SPAN_BYTES` can admit a split map the untiled op cannot take. The per-axis
domains are nested — a divisor of the per-tile extent divides the whole one, and the committed span
floors bind both — so the untiled ones are the widest and the view inverse searching them misses
nothing. The joint verdict is not nested, so whatever falls back from a tiled config to its untiled
twin checks the twin, as the recolor anchor does.

In practice that difference is empty. An op that fits untiled at some split fits at any split
meeting the committed floors, which every tiled-admitted split meets. An op that fits at none never
reaches the search untiled: `span_reduction_pass` refuses it, unless the opt-in span-overflow pass
has tiled it first, and an op already tiled is offered no tiling. So this search still only finds
tilings that pay through LX residency.

That payoff is not a cost term. Every tiling-sensitive term in the cost model is a derate bounded by
1.0 and an untiled op has a working set of 0, so the objective can rank tilings against each other
but never above not tiling. What a tiling does is divide `_per_core_size` by `output_tile_count` as
well as `output_partition`, which can bring a buffer under `_eligible`'s capacity gate and be repaid
in the HBM traffic residency then frees.

The tiling half is attached only where `CoOptimizingAllocator._solver_chooses_tilings` holds: this
engine, and `config.auto_coarse_tiling`, off by default. Unlike the rest of this engine's behaviour
that really is a user setting rather than a consequence of which engine runs. A refusal from the
apply round raises rather than falling back, so any gap between what `OpSplitSpace.admits` believes
it may tile and what `coarse_tile` accepts is a compile failure. And nothing yet prices the loop
cost above the split cap, so the search has no downward pressure on the tiling axis and takes as
much of it as the divisor lattice offers.

## The apply round

`CoOptimizingAllocator._apply_chosen_tilings` collects the chosen `TileSpec`s off the allocation,
keys them by operation name, and runs `CoarseTilingPass` — the only consumer a chosen spec has.
Three things about where it sits.

**Before the commit, not after.** `commit_iteration_space_ownership` builds the ownership off
`iteration_space_from_op`, whose symbols come from the write dep's ranges — which `_divide_ranges`
invalidates, and whose `core_to_slice_mapping` is a function of the whole ordered split tuple, so it
goes stale even when every symbol survives. Committing afterwards derives both halves against the
already-divided op rather than migrating a stale object. `coarse_tile` reads no ownership of its
own, so the one `_distribute_work` left is simply overwritten. Where a tiled dim divides to extent 1
its symbol leaves the iteration space altogether; `make_iteration_space_ownership` would silently
read that axis as unsplit, so `commit_divisions` refuses a chosen division naming a symbol the op
no longer has.

**The anneal's placement stands; there is no second round.** Applying the tiling is what makes those
addresses *true* — the hazard was that the search priced the per-tile footprint while the graph
wrote the full extent, and the apply closes exactly that gap. Re-running a placement engine here
would decouple the layout from the divisions and tilings it was jointly chosen with, which is the
coupling this engine exists for. The companion buffers the apply mints — a full-extent `full_buf`
per op whose output escapes its tiling group — were not in the joint state, so they get no LX
address and stay in HBM until something prices them. That is the remaining known optimism: the
search sees the per-tile shrink but not the companion.

**A refusal raises**, after `CoarseTilingPass.plan_only` — a zero-mutation dry run — so it raises on
an untouched graph rather than leaving a half-transformed one behind. The search is meant to propose
only tilings `coarse_tile` accepts, so a refusal is a defect in `OpSplitSpace.admits` rather than a
graph to route around; dropping the tiling at this point would also invalidate the addresses already
spaced for it.

`_check_priced_footprints` then asserts what the old "refuse a tiled resident buffer" guard was
reaching for, in the form that survives the feature working: the buffer's applied per-core footprint
is the one it was placed at. The search divides the total size by
`output_partition * output_tile_count`; the apply divides the op's ranges per dim and rebuilds the
device layout through `_resize_device_layout`. Those agree only if that resizing divides the device
byte size exactly — plausible, since `build_tiling_space` never tiles the stick dim, but per-dim
padding could re-round, and an applied footprint *larger* than the priced one would overlap whatever
was packed above it.

Three move types:

* **reorder** (weight 0.5) — a best-first reinsertion sweep. Lift one buffer out, probe every
  legal reinsertion position, and try them in descending packer-`quality()` order, accepting the
  first that clears the Metropolis test. Ranking by the `quality()` proxy rather than the true
  objective is deliberate: it costs O(1) per position, and it breaks ties among the many
  score-identical positions that a permutation move usually offers. Its weight drops to 0 while
  every eligible buffer is resident — `pi` only decides which eligible buffers win LX, so with all
  of them already in, only a structural move can still pay.
* **flip** (weight 0.3) — one step from the drawn buffer's division: a single axis's split factor,
  *or* one coarse tile level. Never both at once, which is what keeps the walk local in a ragged
  space.

  **The two arms have different scope.** A step in the division lattice is the drawn buffer's alone:
  set its config, resize its per-core footprint, refresh LX-eligibility for it and its parents. A
  step in the *tiling* lattice moves a **boundary**. A tiling group is a contiguous run of the
  operation list, so re-speccing one op in the middle of a uniform run would split it into a shape
  the apply round prices differently than the search did. Instead, given the run `A..Z` containing
  the drawn op `H`, `_retile_boundary` re-specs `A..H` or `H..Z` — the run splits in two, or, where
  the new spec matches the neighbouring run's, the boundary between them slides. Untiled is a spec
  value like any other, so runs partition the whole operation list and the move *creates* tiled
  regions as readily as it shrinks them. The single-op move survives as the degenerate case, `H` at
  a run end; a mid-run split takes two steps.

  Whether an op can take a tiling is a per-op question (`neighbours` only offers a level the op's
  current splits survive), and over a run those odds multiply, so the sub-run is **truncated** at
  the first op that refuses rather than the move being rejected — sliding the boundary as far as it
  will go. An operation that produces no solver buffer stops the walk for the same reason it breaks
  a run: nothing can carry a tiling to it. Runs are measured over `CoreDivisionBuffer.op_position`,
  because buffer indices are not operation positions.

  Two costs of that, recorded rather than fixed. The tile levels are **concatenated onto the
  neighbour list, not weighted against it**, so from the untiled state most of flip's mass goes to
  the tiling arm — an implicit retune of a weight #4233 records as already optimal — and `|N(x)|`
  now varies with run length on top of that, which the uncorrected Metropolis test reads as a bias
  towards states with more neighbours. Both are stated rather than tuned: retuning against an
  objective that does not yet price companion buffers would mean retuning twice.

* **recolor** (weight 0.2) — draw a splitting anchor division, flood the residency relation
  bidirectionally from it, and recolor everything it reaches.

  This is the search's **long-range** move: an op's legal divisions are not connected by
  one-axis moves (the core budget blocks a factor going up, a span floor blocks it coming down), so
  its anchor is drawn from the whole space. A generated anchor draws its tiling first, and the flood
  carries that tiling to each op that can take it: a coarse tiling group is a run of consecutive ops
  agreeing on one `TileSpec`, so the flood is what forms one.

  The flood's reach is the residency relation's, which is producer/consumer reachability — not
  contiguity. So `_trim_tilings_to_anchor_run` strips the `TileSpec` from every op the flood reached
  outside the anchor's contiguous run, leaving its **splits** untouched: those are what the flood is
  for, and narrowing them to the run would cost the long-range division move measured at −0.71%.
  Stripping is always legal, because the untiled factor domain contains the tiled one.

  *Flip moves a boundary, recolor repaints a region* — that is the division of labour, and it is why
  making flip's tiling arm multi-op does not make the two the same move. Recolor changes divisions
  to make residency edges compatible and redraws a tiling outright; flip changes no division and
  steps one level from the run's current spec, bounded by one existing run.

Both structural moves carry a short cold layout burst, so `pi` has adapted to the new footprints
before the compound move is judged as a unit by one Metropolis test. The burst stops early for the
same reason reorder does — as soon as every eligible buffer is resident.

Once no move applies at all (every eligible buffer resident and neither structural move
available), nothing can change the state again, so the cool ends there rather than spending its
remaining budget.

A run is **bit-for-bit reproducible**: the RNG is seeded, every domain it draws from is
index-ordered, and the score is an integer fixed-point quantity, so there is no float
accumulation to reorder.

:::{warning}
Reproducible is not stable: the trajectory is chaotic, so compare two revisions over several seeds,
never one.
:::

## The objective

**`cost_expr`** is `CoOptimizingAllocator._solve`'s symbolic prediction for the whole graph —
`sympy.sympify(predict_by_bundle(graph.operations, op_features, params=_COST_PARAMS))`, built from
every buffer's own `sym_is_lx`/`sym_core_divs` (the same symbols the CP-SAT engine's cost
expression is built from). `plan_layout_and_core_divisions(cost_expr)` compiles it once, per
solve, into a fast `(chosen, resident) -> fixed-point ns` callable (`_build_score_fn`): every free
symbol in the expression maps back to a getter built off THESE buffers — an argument's residency
from whether its owning buffer's name is in `resident`, a split symbol's value from the config
`chosen[idx]` itself. The compiled formula is evaluated fresh every step; unlike the
`BundleCostObjective` it replaced, there is no incremental per-bundle memoization or dirty
tracking, and so nothing to invalidate on a rejected move.

**Memory-only** is the fallback, taken when `cost_expr` is `None` (the normal case for anything
driving serialized captures, including the tests) or when it can't be compiled here — an
unrecognized free symbol (e.g. a dynamic-shape symbol the allocator's build left in), or a
construction error `_build_score_fn` catches. It counts the HBM traffic a spill adds over
residency, converted once to fixed-point microseconds. Being *differential*, a resident buffer
contributes exactly zero and only spilled buffers are summed. Its weakness is why the cost model
replaced it: a core division only matters through what it lets fit, so on a graph where
everything fits, every division scores the same and the search has nothing to optimize. The
engine logs which objective it took.

:::{warning}
Building `cost_expr` symbolically shares a real limitation with the CP-SAT path: a few cost-model
code paths (`_is_broadcast_op`, `_transport_kind`, and the standalone-reduction/loop-reread rows)
decide their branch by reading `ArgTraffic.mem`, which raises on a symbolic `is_lx` — so any op
whose special bandwidth rate depends on residency (a broadcast, `cat0`/`cat1`, `transpose_outer`,
or a `sumcol`-style reduction) either drops that rate silently or, if the read raises, sinks the
whole expression back to the memory-only fallback. There is no per-bundle escape hatch for this
the way `BundleCostObjective`'s concrete `predict_ops` calls had.
:::

:::{warning}
The cost objective's plans are cheaper **by the cost model's own reckoning**. No device time has
been measured.
:::

## Bundles are estimates

The cost model scores one fused kernel at a time, and bundle membership changes the answer —
external inputs are deduplicated across a bundle, the pointwise arity derate counts its ops, the
underfill derate takes its worst tile. The co-optimizer cannot ask for the real grouping, because
fusion is decided two stages later. `fusion.estimate_bundles` reproduces the rule from the
operation list instead, sharing `group_contiguous_fusable` with the real pass so the two can only
diverge on the predicate.

:::{warning}
The estimate's accuracy has been checked against real fusion on **one** softmax graph, where the
bundle count, run structure and boundary placement were right and membership under-counted by a
node scheduling introduces later. If the real grouping splits differently, the search is
optimizing a cost that is not the cost that gets compiled. Validating the estimate across a
corpus would be valuable.
:::

## Test fixtures

`tests/inductor/cooptimization_captures.json` holds captured solver inputs (candidate menus,
`cd_parent_matches`, placement and cost fields) plus the reference solution, for the
shape-invariant guarantees — output contract, geometric validity, `>=` baseline, determinism.
`cooptimization_captures_large.json` holds 25–100 buffer graphs for the same guarantees at scale,
opt-in via `SA_COOPT_LARGE_CAPTURES=1` because they are slow.

`cost_expr` needs features as well, so building or checking one against the captured corpus needs
`cooptimization_op_features.json` alongside the buffer captures above. The two must come from the
same compile: every Inductor graph names its buffers `buf0..`, so names collide across unrelated
graphs without lining up. Regenerating it requires a Spyre machine, since the feature extractor
reads live Inductor IR.

One contract is worth stating because it was wrong for a while. The allocator sets
`parents = info["op_inputs"]` without intersecting the solver's buffer set, so an op's graph
inputs, constants and extern outputs appear there. The solver **skips** parents it does not own
rather than asserting on them — a buffer the solver does not own is never LX-resident, so the
edge has nothing to gate. Clone-eligible graph inputs are unaffected: those *are* solver buffers
and resolve normally.

## Open work

1. **Validate against device time.** Every score is the cost model's own prediction.
2. **Validate `estimate_bundles` across a corpus**, not one graph.

## Related documents

* [Scratchpad Planning](scratchpad_planning.md) — the allocator, the other solvers, and the
  co-optimization concept
* [Simulated Annealing Layout Planner](simulated_annealing_layout.md) — the placement-only
  annealer and its schedule
* [Work Division Planning](work_division_planning.md) — where the candidate core divisions come
  from
