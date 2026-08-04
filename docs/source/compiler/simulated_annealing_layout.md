
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
- **`permutation_layout.py`** — the core. A *permutation* is an allocation order;
  `PermutationBasedLayoutSolver` places each buffer on top of the earlier-placed buffers it overlaps
  in time (with in-place reuse), and maintains all addresses **incrementally** under
  `swap`/`rotate`. `ReferencePermutationBasedLayoutSolver` is a slow, obviously-correct O(n²)
  rebuild used as a test oracle, not used in production code.
- **`contact_profile.py`** — `Profile`, the contact-profile data structure the incremental solver is
  built on.
- **`cooling_schedules.py`** — the `CoolingSchedule` family: `ExponentialCoolingSchedule` and the
  default, auto-calibrated `SelfCalibratingReheatingSchedule`.
- **`simulated_annealing.py`** — `SimulatedAnnealingSolverWithBuffers`, a simulated-annealing search over allocation
  orders (following a paper by Imanishi & Xu) that drives the permutation solver by composition. It
  is wired in as the opt-in `layout_solver = "simulated_annealing"` config option; the default stays
  `greedy`.
- **`benchmarks/`** and **`examples/scratchpad/`** — profiling scripts, result docs, and runnable
  examples.

**Validation philosophy:** every incremental operation is checked against the
from-scratch reference oracle — randomized *differential* tests, a gated *stress* suite
(`TORCH_SPYRE_STRESS_SCRATCHPAD=1`, tens of thousands of seeds), and in places *exhaustive*
enumeration of all small configurations. This is what makes the subtle in-place edge cases
trustworthy.

**Key invariants that recur:** lifetimes are half-open; per column, address order equals permutation
order (weakly — ties only for in-place reuse); at most two buffers share one address at one tick
(in-place legality caps it); and an in-place pair overlaps at exactly one "transition" tick.

## How it works

### The incremental placement engine

`PermutationBasedLayoutSolver` is the placement engine. Given an allocation order (a permutation of
buffer indices), it places each buffer at `align_up(max top of the earlier-placed buffers it overlaps
in time)`, with an in-place child allowed to reuse a parent's slot. `quality()` — the total size of
buffers that fit under capacity — is maintained in O(1).

Addresses are maintained **incrementally** rather than rebuilt. Who-is-below-whom is represented by
*contact profiles*: a `Profile` is a step function over a buffer's lifetime giving its directly
below/above neighbour per column. A `swap` of two adjacent permutation entries transposes them only
over their shared column range via O(segments) splices, then propagates address changes along
*order-above* edges (candidates bounded by a precomputed time-overlap set). An **in-place-status
transition rule** handles the "poke-through" case: a transparent in-place child sits low while its
taller parent pokes up to carry the buffer above it, so a change to the parent must reach that
buffer.

`_recompute_address` reads a buffer's placement candidates straight off its contact profile rather
than scanning the whole overlap set. This relies on `contact_at` — the derived "what does this buffer
rest on" view — being symmetric: it reports the `(parent, child)` in-place pair in *both* reuse
directions, surfacing the buried co-located buffer that the in-place legality test needs. The
candidate set built from a buffer's below-profile breakpoints is provably sufficient for the
placement decision (backed by a sufficiency proof and exhaustive small-case checks).

For speed, each buffer's (static, sparse) in-place-partner set is precomputed, so placement never
probes every candidate for an in-place relationship, and `_top` is inlined over a flat sizes array.

`ReferencePermutationBasedLayoutSolver` is the obviously-correct O(n²) counterpart, sharing
`PermutationBasedLayoutSolverBase`: it rebuilds every address from scratch on each operation. It
ships only as the differential-test oracle and a readable reference spec of the placement semantics.

### `rotate`

`rotate(i, j)` moves one permutation entry to another position, choosing between two strategies by
distance (correctness is independent of the choice). For short moves it is a chain of adjacent
`swap`s, most of which are O(1) no-ops. Once `|i − j|` reaches the threshold `max(2, n // 8)` it uses
a remove/reinsert fast path: edit the permutation once, recompute addresses reusing the static
overlap set (never the O(n²) reference scan), and patch the contact profiles for the single move —
making the cost independent of `|i − j|`, which matters for the long dense rotations. The threshold
is an instance attribute callers can override (set it to 1 to force the fast path on every rotate).

### The annealing search

`SimulatedAnnealingSolverWithBuffers` is the search that optimises the layout, following the
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
