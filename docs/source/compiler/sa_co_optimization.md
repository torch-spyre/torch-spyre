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
those; the rest read the enumerated menu, one config per position — every menu is already
duplicate-free on its own terms, an op's by split map and a clone's by physical partition. A
config's identity is its split map, except at a clone menu position that repeats one: a clone's
entries are synthesized out of different consumers' iteration symbols, which are positional and
repeat across ops, so two of them can share a split map while slicing the buffer differently. The
two sources answer alike, because a space admits exactly what the enumeration carries.

Each producer→consumer edge likewise gets an `_EdgeRelation`, which the residency gate and the
recolor flood ask: is this pair of divisions compatible, and what division does the other end need.
Where both ends generate and the allocator handed over a `ResidencyEdge`, that is computed per
candidate off the buffer's geometry — including *constructing* the other end's division by inverting
the per-core view (`invert_per_core_view`), where the flood used to look one up in the pair table.
Otherwise the `cd_parent_matches` table serves, projected onto choices. The two agree on *which
pairs are compatible* — a space admits exactly what the enumeration carries, so a generated division
is one the menu holds and the table knows its key — which is what keeps a graph where only some ops
generate from being a mixture of two verdicts. They do not agree on *which* compatible division each
hands back: the table's tie-break is the lowest menu position, while the inverse returns the first
solution its own ordering reaches (placements by `(host stride, name)`, then hidden symbols by
ascending factor). So a flood crossing a generated edge can propagate a different — equally
compatible — division than the table would have picked. Both are deterministic; which one searches
better is unmeasured.

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
the list carries. Level order is **canonical** for that to hold in both directions: an output spec's
levels ascend by `host_dim`, `admits` refuses any other order, and the move alphabet has no reorder
step. Nest order is therefore not a decision variable — no term in the objective depends on it, so
carrying both orders of a nest would double the state space, hand the applier an order chosen by
coin flip, fragment a tiling group on a distinction without a difference, and buy nothing.
 Whether it is attached at all is
`CoOptimizingAllocator._solver_chooses_tilings`, and there is no flag: only a search that generates
divisions can carry a `TileSpec`, so the engine *is* the switch, and `select_allocator` reaches this
one from exactly two settings — `co_optimizing_lx_planning` plus
`layout_solver = "simulated_annealing"`. Handed no tiling space, the space has no tiling half, every
division is untiled, and the search draws and proposes exactly what it did before the field existed.

Two scope limits are enforced rather than merely intended. An op that already carries `dim_hints` —
the marker the hint pass (430) and the span-overflow pass (448) both leave set — gets an empty space,
because `CoarseTilingPass` stamps `op.dim_hints` wholesale and would clobber the group that op is
already part of. And v1 is **output axes only**: no move ever adds a reduction level, and
`OpSplitSpace.admits_tiling` refuses a spec carrying one outright, since `tile_counts` skips
reduction levels and would leave such an axis judged against its untiled extent. `TilingSpace.is_empty`
is defined against the moves for the same reason — an op whose only tileable axis is a reduction one
is one a generating search can do nothing with, and calling it movable spends a step of the budget on
every flip drawn for it.

The predicate's second conjunct is `config.auto_coarse_tiling`, off by default. Unlike the rest of
this engine's behaviour it really is a user setting rather than a consequence of which engine runs,
for two reasons that are not about the machinery working. A refusal from the apply round raises
rather than falling back, so any gap between what `OpSplitSpace.admits` believes it may tile and
what `coarse_tile` accepts is a compile failure. And the tiling price the search sees is partial:
`predict_ops` charges a real tiling 4.7–5.2x the measured device slope on a decoder block, and what
reaches the search is only part of that (below), so neither figure is a calibrated brake yet.

`CoOptimizingAllocator._apply_chosen_tilings` is what runs `CoarseTilingPass` over the chosen specs,
in `_post_solve` and **before** the divisions are committed — see *The apply round* below.

The space is **ragged**, and in one direction only. A tile level cuts its axis's per-tile extent, so
a core split of that axis must divide the smaller extent — `WorkDivisionContext.factor_domain(axis,
tile_count)` narrows accordingly, dropping *large* factors.

The mirror image is deliberately not modelled. `get_per_core_span` divides each dim's range by its
split count, so a tiling shrinks the per-core span, and both `MAX_SPAN_BYTES` and the floor
`span_reduction_pass` commits would then admit *smaller* splits — tiling would add small factors
back. Judged untiled as they are here, per-tiling domains come out *nested* inside the untiled one
rather than incomparable to it. That nesting is an **invariant**, not an apology: `_split_key` and
`_TableRelation` assume a tiled config's split half is a key the untiled menu already carries, the
write-back's `_menu_position` append assumes the same, and `_commit_divisions` commits the split half
of a tiled-but-unapplied config into an untiled graph — all three are legal only because the tiled
domain is a subset, and none of them would catch a violation. Modelling the mirror half has to come
with those three. It costs an option, never a verdict, but note which option: span

relief is the in-tree reason coarse tiling exists (`_maybe_coarse_tile_span_overflow`, pass 448), so
this search can only find tilings that pay through LX residency, never ones that pay by making a
bigger core split legal. Two things block doing it here — the span arithmetic runs off the untiled
op's tensor deps, and the floor is already *committed* to the op by `apply_splits` rather than being
a filter to relax.

The payoff is not a derate. Every tiling-sensitive derate in the cost model is bounded by 1.0 and an
untiled op has a working set of 0 by definition, so those can rank tilings against each other but
never above not tiling. What a tiling does is divide `_per_core_size` by `output_tile_count` as well
as `output_partition`, which can bring a buffer under the capacity gate in `_eligible` — an engine
threshold, not a cost — and be repaid in the HBM traffic residency then frees.

### What the tiling costs, as an objective symbol

`CoreDivisionBuffer.sym_tile_counts` declares one symbol per *iteration axis* the buffer's tiling
space offers a level on, beside `sym_core_divs`' one per stride coefficient. Keyed by axis because a
tile count divides an extent exactly as a core split does, and because which of an op's args a level
makes loop-invariant is decided by whether that axis's symbol appears in their index — a static
fact. So only the count is unknown, and a candidate leaving an axis untiled binds its symbol to 1.
`_build_sources` freezes the declaration and `undeclared_tile_axes` holds generated configs to it,
the way `undeclared_splits` does for the split half.

`CoOptimizingAllocator._extract_op_features` passes a `ProspectiveTiling` — those axes and their
symbols — to the extractor, which stamps the two features linear in it: `loop_trip`, and each arg's
`loop_factor`, the multiplier on an operand the loop re-reads once per iteration. That is the whole
channel. `tiles_output_dim` is deliberately **not** set: it gates Python branches (a matmul's
`pt_eff`, the standalone-row-reduction rate) that would move for every *tileable* op whether or not
the search tiles it, so substituting the counts at 1 would no longer reproduce the untiled price.
Nothing is lost by that — everything else it gates keys on `tile_rows_per_core`, which is symbolic
here, and `_tiled_rows` already withholds those derates for a symbolic tile height.

### Residency across a tiling boundary, and the copy that restores it

The backend cannot advance an LX start address: it is never registered as a symbol in the SDSC JSON,
so `affine.apply` has nothing to target. A coarse-tiled op reading a buffer produced *outside* its
run reads it at an address that moves once per tile, so that buffer may not be resident —
`_read_across_a_tiling_boundary` refuses it. The rule is stated on two configs (this buffer untiled,
some consumer not) and that is sufficient: a tiled producer in another run escapes into an HBM
`full_buf` its consumers are repointed at, and one in the same run is loop-internal scratch at a
fixed address. Left ungated it is not a clean refusal but wrong data — 32 cores compiles it and
returns the first tile every trip.

`CoarseTileReadCopyBuffer` gives that residency back. It is the tile-local staging copy
`coarse_tile`'s Pass 1 builds, predicted in `_build_cd_bound_buffers` so the anneal places it —
minted by the apply it could only ever be HBM, where it would cost a write and a read to save
nothing. It is sized and gated on its *reader's* config (one core's share of one tile, absent while
the reader is untiled), carries no division decision, and is excluded from the step budget, which
counts decisions rather than slots. `_read_copy_savings` prices it at `(r - 1) * size` while
resident: the copy op makes one HBM pass and the `r` readers then read LX. A copy standing in for a
single read is never predicted, since it could only occupy LX for nothing.

The apply is handed the pairs that were placed and stages exactly those, so the residency decision
and the decision to stage are one decision. Its reach is bounded by
`_read_copy_can_be_sized`, which refuses a read whose iteration extents do not map one to one onto
the source's committed dims — every broadcast and every matmul operand
(`TODO(span-overflow-read-copy)`). Those reads are dropped rather than refused, so the op keeps
reading the full buffer; it simply cannot be resident while doing so.

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
read that axis as unsplit, so `_commit_divisions` refuses a chosen division naming a symbol the op
no longer has.

**The anneal's placement stands; there is no second round.** Applying the tiling is what makes those
addresses *true* — the hazard was that the search priced the per-tile footprint while the graph
wrote the full extent, and the apply closes exactly that gap. Re-running a placement engine here
would decouple the layout from the divisions and tilings it was jointly chosen with, which is the
coupling this engine exists for. The companion buffers the apply mints — a full-extent `full_buf`
per op whose output escapes its tiling group — were not in the joint state, so they get no LX
address and stay in HBM — which is what the search prices them as, mid-anneal, through
`_companion_bytes` (see [the objective](#the-objective)).

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
padding could re-round, and an applied footprint *larger* than the priced one is the overlap hazard
again. Quantified while nothing applied a tiling at all:
`~/coopt-repro/stage3_unapplied_tiling_overlap.py` produced a 16,128-byte overlap between two live
buffers in one arrangement and a 12,768-byte overrun of the LX region in another, and a real compile
(`test_mlp__simulated_annealing_sc32_coopt`) reserved 256 bytes for a buffer that would write
16,384 — 64× — with its numerical check still passing.

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
  space. Stage 0 measured the factor domains at ~7 legal factors per axis, which is why this is a
  list to draw from rather than a proposal scale to cool.

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

  This is the search's **long-range** move, and it has to be: an op's legal divisions are not
  connected by one-axis moves (the core budget blocks a factor going up, a span floor blocks it
  coming down), so a search whose only structural moves were local scored 0.9% worse on the corpus
  at four seeds. Its anchor is therefore drawn from the whole space — uniformly from the menu's
  splitting entries, or, generated, by redrawing the tiling and then every axis, keeping each draw
  that leaves the division legal. The tiling is drawn first because the space is ragged in that
  order, and it is drawn at all because a coarse tiling *group* is a run of consecutive ops
  agreeing on one `TileSpec` (`derive_tiling_groups`): the flood is what forms one, carrying the
  anchor's tiling to each op that can take it and leaving the far side untiled where it cannot.

  The flood's reach is the residency relation's, which is producer/consumer reachability — not
  contiguity. So `_trim_tilings_to_anchor_run` strips the `TileSpec` from every op the flood reached
  outside the anchor's contiguous run, leaving its **splits** untouched: those are what the flood is
  for, and narrowing them to the run would cost the long-range division move measured at −0.71%.
  Stripping is always legal, because the untiled factor domain contains the tiled one.

  *Flip moves a boundary, recolor repaints a region* — that is the division of labour, and it is why
  making flip's tiling arm multi-op does not make the two the same move. Recolor changes divisions
  to make residency edges compatible and redraws a tiling outright; flip changes no division and
  steps one level from the run's current spec, bounded by one existing run.

  The tiling draw is judged on the tiling's *own* legality, not against the incoming splits, or a
  tiling whose only legal companions are smaller splits — exactly the footprint-shrinking state the
  feature exists to find — would be rejected before the split redraw that would supply them. Splits
  the drawn tiling cannot take drop to all-ones first, so the redraw climbs out of a legal state.

  Untiled is drawn **flat**, at `_UNTILED_ANCHOR_PROB`, rather than as a per-dim opt-out. Per-dim
  opt-outs alone leave the untiled anchor at the *product* over tileable dims (≈1/289 at two dims,
  ≈1/4913 at three), so undividing a region would vanish exactly as the search gained room to
  over-divide it — and nothing else pushes back: the objective sees a tiling only through a monotone
  per-core footprint, so a tiling move is score-neutral (accepted unconditionally) or score-improving.
  There is no loop-cost term yet; #4233 measures program bytes at `74,880 + 21,504 × total_tiles`,
  with backend compile time superlinear in tile count, and names that term as the prerequisite for
  lifting the split cap.

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
(`_build_sources`): one symbol per stride coefficient seen across that buffer's candidates. A
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

**Companion traffic** is added to whichever of the two runs, at the HBM rate, because neither can
express it. A coarse tiling shrinks what a buffer holds at once; it does not shrink what the
buffer moves. For an op whose output escapes its tiling group the apply allocates a full-extent
`full_buf`, drains one tile into it per iteration through an inserted copy op, and repoints every
outside consumer and any graph output at it — none of which is in `op_features`, extracted from
the untiled graph before `_post_solve` applies anything. So the search would see the per-tile
shrink and neither the copy nor the outside consumers' HBM reads, and the error does not fall
with the tile count. `_companion_bytes` charges the difference between the applied graph and the
extracted one, per escaping buffer: the copy's read of the per-tile scratch, free when that
scratch is resident; the copy's write of `full_buf`, already counted for a graph output whose
externally visible write is charged whether or not the buffer is resident; and the outside
consumers' full-extent reads when the buffer is resident, which residency no longer serves. An op
whose consumers all sit inside its own run therefore costs nothing — the shape the recolor flood
exists to build — while tiling a buffer that is not resident costs a full HBM round trip. The term
is zero unless a buffer is tiled, so a search that takes no tiling scores as it did before it
existed.

It rides outside the compiled expression rather than inside it because `cost_expr`'s declaration
is splits and residency: there is no symbol for a tiling to price against. Two under-corrections
follow from what the solver holds: a consumer is counted once however many times it reads, and a
matmul consumer's `replication` would make its read of `full_buf` cost more than one pass.

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
