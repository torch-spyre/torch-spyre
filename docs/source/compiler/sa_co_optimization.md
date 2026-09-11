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
`PermutationBasedLayoutSolver` packer, and the division vector `W`, one `DivisionConfig` per
buffer. A config is a division as a *value* — the `CoreDivision` itself, a canonical hashable key
identifying the choice it makes, and the menu position it came from, if any. The seed is every
buffer at its first candidate with `pi` from a FirstFit pass. One geometric cool runs
`clamp(40n, 200, 15000)` steps at fixed proposal weights, and the best state seen is what gets
written back — so the result is never worse than the seed.

### Where the candidates come from

Each buffer gets a `_DivisionSource`, and the engine asks nothing else: the seed, the divisions one
step away (`neighbours`, what a flip proposes), and a splitting division to flood from
(`anchor`, what a recolor proposes). A buffer whose producing op has an `OpSplitSpace` *generates*
those; the rest read the enumerated menu, deduplicated by choice — a menu carries the same division
at several positions, since a factor-1 axis is dropped from the sparse split map and `{d0: 2, d1: 1}`
enumerates as a second copy of `{d0: 2}`. The two sources answer alike, because a space admits
exactly what the enumeration carries.

Each producer→consumer edge likewise gets an `_EdgeRelation`, which the residency gate and the
recolor flood ask: is this pair of divisions compatible, and what division does the other end need.
Where both ends generate and the allocator handed over a `ResidencyEdge`, that is computed per
candidate off the buffer's geometry — including *constructing* the other end's division by inverting
the per-core view (`invert_per_core_view`), where the flood used to look one up in the pair table.
Otherwise the `cd_parent_matches` table serves, projected onto choices; the two agree, so a graph
where only some ops generate is not a mixture of two answers.

The one thing still keyed by menu position is the `chosen_division` written back, which is the
allocator's contract rather than the engine's: a generated division is normally one the enumeration
already carries, so the position is resolved by choice, and a division the menu does not carry is
appended to it at write-back — which is the path a tiled division always takes, since the
enumeration carries no tilings.

### The coarse tiling rides on the same candidate

`CoreDivision.tiling` is a `TileSpec`, and `OpSplitSpace` chooses it jointly with the splits rather
than alongside them. It has to be joint: tiling rewrites index expressions and
`splits_by_index_coeff` keys the output splits by each symbol's coefficient in the write index, so a
`CoreDivision` carried across tilings is uninterpretable rather than merely illegal.

The tiling half of the space is `TilingSpace` (`wsr/enumerate_tilings.py`), whose predicates
`enumerate_tile_options` is now the cross product over — the same relationship
`WorkDivisionContext` has to `enumerate_work_division_candidates`, so a spec the space admits is one
the list carries. Whether it is attached at all is
`CoOptimizingAllocator._solver_chooses_tilings`, and there is no flag: only a search that generates
divisions can carry a `TileSpec`, so the engine *is* the switch, and `select_allocator` reaches this
one from exactly two settings — `co_optimizing_lx_planning` plus
`layout_solver = "simulated_annealing"`. Handed no tiling space, the space has no tiling half, every
division is untiled, and the search draws and proposes exactly what it did before the field existed.

The predicate has a second conjunct that is not a choice: `TILE_CHOICES_ARE_APPLIED`, which says
whether anything runs `CoarseTilingPass` over a solve's chosen specs. **Nothing does yet**, so today
no engine is offered tilings and this whole section is dead code waiting on its apply step — see the
warning below for why that is a safety requirement and not caution. Wiring that step **deletes** the
constant rather than setting it `True`: it marks a missing implementation, not a mode, and a
constant pinned `True` would be the config flag this deliberately does not have. What survives is
`_commit_divisions`' refusal below, which is an invariant and not a placeholder.

The space is **ragged**, and in one direction only. A tile level cuts its axis's per-tile extent, so
a core split of that axis must divide the smaller extent — `WorkDivisionContext.factor_domain(axis,
tile_count)` narrows accordingly, dropping *large* factors.

The mirror image is deliberately not modelled. `get_per_core_span` divides each dim's range by its
split count, so a tiling shrinks the per-core span, and both `MAX_SPAN_BYTES` and the floor
`span_reduction_pass` commits would then admit *smaller* splits — tiling would add small factors
back. Judged untiled as they are here, per-tiling domains come out *nested* inside the untiled one
rather than incomparable to it. That costs an option, never a verdict, but note which option: span
relief is the in-tree reason coarse tiling exists (`_maybe_coarse_tile_span_overflow`, pass 448), so
this search can only find tilings that pay through LX residency, never ones that pay by making a
bigger core split legal. Two things block doing it here — the span arithmetic runs off the untiled
op's tensor deps, and the floor is already *committed* to the op by `apply_splits` rather than being
a filter to relax.

The payoff is not a cost term. Every tiling-sensitive term in the cost model is a derate bounded by
1.0 and an untiled op has a working set of 0 by definition, so the objective can rank tilings
against each other but never above not tiling. What a tiling does is divide `_per_core_size` by
`output_tile_count` as well as `output_partition`, which can bring a buffer under the capacity gate
in `_eligible` — an engine threshold, not a cost — and be repaid in the HBM traffic residency then
frees. `sym_core_divs` carries symbols for the splits only, so the `TileSpec` itself is invisible to
`cost_expr`.

:::{warning}
A chosen tiling that nothing applies is **not** a harmless no-op, which is why
`TILE_CHOICES_ARE_APPLIED` gates the whole thing rather than merely documenting it. The search
prices the per-tile footprint and the packer lays LX out by it, while the untiled graph writes the
full extent — so the reserved interval is a fraction of the real one, and the bytes above it go to
whatever was packed there next, or off the end of the region.

Measured with the gate forced open. On a real compile
(`test_mlp__simulated_annealing_sc32_coopt`) a resident buffer reserves 256 bytes and will write
16,384 — 64× — and nothing breaks only because it is the sole resident buffer and LX has room; its
numerical check passes, so the error is invisible there. With a second resident buffer it bites
both ways: `~/coopt-repro/stage3_unapplied_tiling_overlap.py` shows a 16,128-byte overlap between
two live buffers in one arrangement and a 12,768-byte overrun of the LX region in another.
`_commit_divisions` refuses such a commit outright, as the guard for the two getting out of step.

The per-tile footprint is also optimistic in a second way, which outlives that gate: an op whose
output escapes its tiling group is read at full extent by the ops outside it and needs a companion
buffer that nothing sizes yet.
:::

Three move types:

* **reorder** (weight 0.5) — a best-first reinsertion sweep. Lift one buffer out, probe every
  legal reinsertion position, and try them in descending packer-`quality()` order, accepting the
  first that clears the Metropolis test. Ranking by the `quality()` proxy rather than the true
  objective is deliberate: it costs O(1) per position, and it breaks ties among the many
  score-identical positions that a permutation move usually offers. Its weight drops to 0 while
  every eligible buffer is resident — `pi` only decides which eligible buffers win LX, so with all
  of them already in, only a structural move can still pay.
* **flip** (weight 0.3) — move one buffer one step: change a single axis's split factor to another
  its domain admits, *or* edit one coarse tile level (add, remove, recount, or swap two adjacent
  levels), then ripple, resizing its per-core footprint and refreshing LX-eligibility for it and
  its parents. Never both at once, which is what keeps the walk local in a ragged space. Stage 0
  measured the factor domains at ~7 legal factors per axis, which is why this is a list to draw
  from rather than a proposal scale to cool.
* **recolor** (weight 0.2) — draw a splitting anchor division, flood the residency relation
  bidirectionally from it, and recolor everything it reaches.

  This is the search's **long-range** move, and it has to be: an op's legal divisions are not
  connected by one-axis moves (the core budget blocks a factor going up, a span floor blocks it
  coming down), so a search whose only structural moves were local scored 0.9% worse on the corpus
  at four seeds. Its anchor is therefore drawn from the whole space — uniformly from the menu's
  splitting entries, or, generated, by redrawing the tiling and then every axis, keeping each draw
  that leaves the division legal. The tiling is drawn first because the space is ragged in that
  order, and it is drawn at all because a coarse tiling *group* is a run of consecutive ops
  agreeing on one `TileSpec` (`derive_tiling_groups`): the flood is what forms one, carrying the
  anchor's tiling to each op that can take it and leaving the far side untiled where it cannot.

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
Reproducible is not stable: the trajectory is chaotic in the mapping from a draw to a move, so any
change to that mapping reshuffles which solves win. Reseeding this engine alone moves the corpus
total by +2.2% to +3.6%. Comparing two revisions on one seed therefore measures nothing — use
several seeds and compare the means (`~/coopt-repro/stage2b_quality.py`).
:::

## The objective

**`cost_expr`** is `CoOptimizingAllocator._solve`'s symbolic prediction for the whole graph —
`sympy.sympify(predict_by_bundle(graph.operations, op_features, params=_COST_PARAMS))`, built from
every buffer's own `sym_is_lx`/`sym_core_divs` (the same symbols the CP-SAT engine's cost
expression is built from). `plan_layout_and_core_divisions(cost_expr)` compiles it once, per
solve, into a fast `(chosen, resident) -> fixed-point ns` callable (`_build_score_fn`): every free
symbol in the expression maps back to a getter built off THESE buffers — an argument's residency
from whether its owning buffer's name is in `resident`, a split symbol's value from the config
`chosen[idx]` itself. The symbols come from a *declaration* frozen per buffer before the search
(`_build_configs`): one symbol per stride coefficient seen across that buffer's candidates. A
config splitting an axis outside its declaration would be priced at the symbol's default of 1
rather than rejected, so the declaration is checked where configs enter the state. The compiled
formula is evaluated fresh every step; unlike the `BundleCostObjective` it replaced, there is no
incremental per-bundle memoization or dirty tracking, and so nothing to invalidate on a rejected
move.

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
