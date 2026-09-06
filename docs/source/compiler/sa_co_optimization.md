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
`CoreDivisionBuffer` per graph buffer — carrying the candidate division menu and the
`cd_parent_matches` compatibility relation — and hands the list to the solver, which mutates it
in place with a `chosen_division` and an `address`.

It runs as a **pre-scheduling pass**: `V.graph` is live but
`V.graph.scheduler` is still `None`, so fusion has not happened yet. Anything the engine wants to
know about the kernels its decisions will land in has to be *estimated* from the ordered
operation list (see [Bundles](#bundles-are-estimates)).

The engine takes no options. The buffers, the capacity and the alignment are its whole interface;
the search parameters are module constants in `sa_cooptimizer.py`.

## The search

The state is the pair `(pi, W)`: the layout permutation `pi`, held in a composed
`PermutationBasedLayoutSolver` packer, and the division vector `W`, one menu index per buffer. The
seed is every buffer at menu index 0 with `pi` from a FirstFit pass. One geometric cool runs
`clamp(40n, 200, 15000)` steps at fixed proposal weights, and the best state seen is what gets
written back — so the result is never worse than the seed.

Three move types:

* **reorder** (weight 0.5) — a best-first reinsertion sweep. Lift one buffer out, probe every
  legal reinsertion position, and try them in descending packer-`quality()` order, accepting the
  first that clears the Metropolis test. Ranking by the `quality()` proxy rather than the true
  objective is deliberate: it costs O(1) per position, and it breaks ties among the many
  score-identical positions that a permutation move usually offers. Its weight drops to 0 while
  every eligible buffer is resident — `pi` only decides which eligible buffers win LX, so with all
  of them already in, only a structural move can still pay.
* **flip** (weight 0.3) — move one buffer to a different entry in its own division menu, then
  ripple: resize its per-core footprint and refresh LX-eligibility for it and its parents.
* **recolor** (weight 0.2) — flood the `cd_parent_matches` relation bidirectionally from a
  non-trivial (split) anchor tiling and recolor everything it reaches.

Both structural moves carry a short cold layout burst, so `pi` has adapted to the new footprints
before the compound move is judged as a unit by one Metropolis test. The burst stops early for the
same reason reorder does — as soon as every eligible buffer is resident.

Once no move applies at all (every eligible buffer resident and neither structural move
available), nothing can change the state again, so the cool ends there rather than spending its
remaining budget.

A run is **bit-for-bit reproducible**: the RNG is seeded, every domain it draws from is
index-ordered, and the score is an integer fixed-point quantity, so there is no float
accumulation to reorder.

## The objective

**`cost_expr`** is `CoOptimizingAllocator._solve`'s symbolic prediction for the whole graph —
`sympy.sympify(predict_by_bundle(graph.operations, op_features, params=_COST_PARAMS))`, built from
every buffer's own `sym_is_lx`/`sym_core_divs` (the same symbols the CP-SAT engine's cost
expression is built from). `plan_layout_and_core_divisions(cost_expr)` compiles it once, per
solve, into a fast `(chosen, resident) -> fixed-point ns` callable (`_build_score_fn`): every free
symbol in the expression maps back to a getter built off THESE buffers — an argument's residency
from whether its owning buffer's name is in `resident`, a split symbol's value from
`core_divisions[chosen[idx]]`. The compiled formula is evaluated fresh every step; unlike the
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
