
# A simulated annealing-based memory layout planner

The code base contains a **simulated annealing-based memory-layout planner**.
Like all memory layout planners, when given a set of buffers, each with a size and a
half-open lifetime `[start, end)`, it decides where to place them in a fixed-capacity scratchpad so that
the total size of buffers that fit is maximised. Its code is fairly tricky, which is why this
document exists.

The solver's code lives in `torch_spyre/_inductor/scratchpad/`,
which contains the following files.

- **`plan_solver.py`** — the shared `LifetimeBoundBuffer`
  data type, the `MemoryPlanSolver` ABC, and a simple `GreedyLayoutSolver`.
- **`../../csrc/perm_layout_native.cpp`** — the core, and the only packer that ships. A
  *permutation* is an allocation order; `NativePermutationLayoutSolver` places each buffer on top of
  the earlier-placed buffers it overlaps in time (with in-place reuse) and re-places under
  `swap`/`rotate`/`resize`/`set_eligible`. Its from-scratch counterpart,
  `ReferencePermutationBasedLayoutSolver`, lives in `tests/inductor/reference_perm_layout.py` and is
  a test oracle only.
- **`cooling_schedules.py`** — the `CoolingSchedule` family: `ExponentialCoolingSchedule` and the
  default, auto-calibrated `SelfCalibratingReheatingSchedule`.
- **`simulated_annealing.py`** — `SimulatedAnnealingLayoutSolver`, a simulated-annealing search over allocation
  orders (following a paper by Imanishi & Xu) that drives the permutation solver by composition. It
  is wired in as the opt-in `layout_solver = "simulated_annealing"` config option; the default is
  `cpsat`.

Runnable examples that drive the solver in isolation — a fixed-ordering layout plot,
a first-fit vs simulated-annealing quality comparison, and an in-place convergence
study — live in
[`docs/source/user_guide/examples/scratchpad/`](../user_guide/examples/index.md).

**Native packer.** The packer is C++ only. A Python implementation was the original one and
served as the canonical spec until the C++ port took over as the default; it was measured at
roughly 14× slower end-to-end on mid-sized problems at representative capacity (see
[Native packer performance](native_packer_performance.md)) and has since been removed. Because the
`_C` extension is required for torch-spyre to function at all, a missing native packer means a
stale or incomplete build and raises rather than falling back.

**Validation philosophy:** every operation is checked against the from-scratch reference oracle in
`tests/inductor/reference_perm_layout.py` — randomized *differential* tests, a gated *stress* suite
(`TORCH_SPYRE_STRESS_SCRATCHPAD=1`, tens of thousands of seeds), and in places *exhaustive*
enumeration of all small configurations. This is what makes the subtle in-place edge cases
trustworthy. The oracle is deliberately the naive placer: it is slow, but its correctness is
meant to be evident from reading it.

**Key invariants that recur:** lifetimes are half-open; per column, address order equals permutation
order (weakly — ties only for in-place reuse); at most two buffers share one address at one tick
(in-place legality caps it); and an in-place pair overlaps at exactly one "transition" tick.

## How it works

### The placement engine

`NativePermutationLayoutSolver` is the placement engine. Given an allocation order (a permutation of
buffer indices), it places each buffer at `align_up(max top of the earlier-placed buffers it overlaps
in time)`, with an in-place child allowed to reuse a parent's slot. A buffer that would cross the
capacity line is *evicted* — its address is `None`, and eviction is upward-closed, so anything that
would rest on it is evicted too. `quality()` — the use-weighted total size of buffers that fit under
capacity — is maintained as a running sum and read in O(1).

Placement is a pure function of (permutation, sizes, eligibility, lifetimes), so every mutating
operation simply re-places every buffer in permutation order. What makes that affordable is the
static data computed once in the constructor and shared by reference with every `copy()`: the
per-buffer time-overlap sets (so a buffer's candidates are a filter over its overlaps rather than a
scan of all earlier buffers), the sparse in-place-partner sets (so placement never probes an
unrelated candidate for an in-place relationship), and the lifetime-interval decomposition.

Two things ride on that interval decomposition. A buffer with no in-place partner reads its floor
straight off a per-interval running maximum instead of gathering candidates at all. And the placement
loop carries a **saturation early-stop**: an interval is *done* once it already holds an evicted
buffer or every buffer alive on it has been placed, and once all intervals are done every remaining
buffer must rest (transitively) on an evicted one, so the tail is bulk-evicted. The early-stop is
result-identical to running the loop out; it only changes the work.

The mutators short-circuit where the re-place provably cannot change anything: a `swap` of two
buffers that do not overlap in time, or where either is ineligible (routed to HBM and so outside the
stacking order), moves no address and returns a zero delta after updating the order alone. `rotate`
does the same for an ineligible buffer.

None of this changes the asymptotics: both the packer and the naive placer are quadratic in the
worst case, and the shortcuts buy constants — which is the whole of the win, and enough of it that
porting the genuinely sub-quadratic algorithm the original Python packer used was judged not worth
the roughly 2.5× more C++ it would take (see
[Native packer performance](native_packer_performance.md)).

`ReferencePermutationBasedLayoutSolver` in `tests/inductor/reference_perm_layout.py` is the
obviously-correct counterpart: it scans all earlier-placed overlapping buffers for each placement and
takes none of the shortcuts above. It ships nowhere; it is the differential-test oracle and a
readable reference spec of the placement semantics.

### `rotate`

`rotate(i, j)` takes one permutation entry out and reinserts it at another position. The permutation
and its inverse are edited in one pass and the layout is re-placed, so the cost is independent of
`|i − j|` — which is what the annealing search's long reinsertion moves need.

### The annealing search

`SimulatedAnnealingLayoutSolver` is the search that optimises the layout, following the
simulated-annealing algorithm of Imanishi & Xu. Each step picks a buffer and probes every reinsertion
position by bubbling it across a throwaway `copy()` of the plan, recording `quality()` at each, and
accepts a move by the Metropolis criterion.

The annealer *owns* a plan (composition, not inheritance) and probes on copies: `copy()` is a cheap
O(n), so probing on a copy is cheaper than sweeping the live plan and restoring it. The compile path
uses a seeded RNG, so layout planning is deterministic — the same graph always compiles to the same
scratchpad layout, which build reproducibility and compilation caching depend on.

### Cooling schedules

The `CoolingSchedule` interface is acceptance-*responsive*: `reset()` returns the first temperature
and `update(accepted, move_scale)` the next (or `None` to stop). After each step the annealer reports
both whether the step accepted a move and the *move scale* — the mean `|Δquality|` over the
reinsertion positions it probed, ignoring no-op positions — so a schedule can react to the run.
`ExponentialCoolingSchedule` is a fixed geometric cool-down; `SelfCalibratingReheatingSchedule` is
the default.

The **default `SelfCalibratingReheatingSchedule`** needs no tuning beyond the step budget.
It sizes its temperatures to the instance **online** from the streamed move scale, locates the
productive temperature, and spends the budget on **reheating cycles** around it. With
`A = -ln(accept_hi)` and `B = -ln(accept_lo)` (~0.8 and ~0.01), each cycle cools geometrically from
`center·delta` (accepts a mean-magnitude *worsening* move with probability `accept_hi`) down to
`center/delta` (probability `accept_lo`), where `delta = sqrt(B/A)` fixes the band width and
`center = d_hat/sqrt(A·B)` tracks the move scale as an EMA (`d_hat`). `center` is re-derived from
`d_hat` **every step**, so the band drifts down (or re-expands) with the landscape *within* a cycle,
not only at its boundaries — the EMA horizon is `cycle_len / horizons_per_cycle` (`H`, default 2), so
a stale band never persists for a large fraction of a cycle. `cycles = 1` degenerates to a single
tracked cool. Before the first move scale is known, `center` is seeded from the
peak-load estimate placed at the band top — a single, best-tracked step before the data snaps it
onto the right scale. Sizing temperatures from the data (rather than tuned constants that would
silently degenerate into a random walk or a greedy search on a differently scaled instance) is the
robust default while we lack representative example models; it is a *reasonable, non-definitive*
default — two unvalidated bets (reheating beats a single long cool; online learning beats a
pre-committed warm-up), both bounded by best-seen tracking so they can waste budget but never worsen
the result — to revisit once we can benchmark on real workloads. Knobs: `total_steps`,
`cycles` (default 4), and `horizons_per_cycle` (default 2, sets the center-tracking EMA horizon);
the budget is adaptive (`clamp(30·n, 500, 5000)`), so the n=100 example uses 3000 steps and nothing
exceeds 5000.
