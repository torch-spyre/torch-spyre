# Implementation Plan — Coarse Tiling Optimization

| Field | Value |
|---|---|
| Status | Draft |
| Implements | [`draft-unified-tiling-cpsat.md`](draft-unified-tiling-cpsat.md) (collateral doc 1, Phase 1) |
| Branch | `course-chunking-planning` |
| Scope | 59 numbered requirements (R1.1–R10.6) across `scratchpad`, `wsr`, `padding.py` |
| Shape | 7 stages, one PR each; steps within a stage are commits |
| Architecture | Tiling rides on a `LifetimeBoundBuffer` subclass and is applied inside the scratchpad planning pass — see *Architecture* below |
| Prior work | [PR #3736](https://github.com/torch-spyre/torch-spyre/pull/3736) — the test mechanism and its hint-preservation half, in review |

This is the execution plan for the RFC, not a restatement of it. It cites
requirements by number and specifies **what to build, in what order, and which
test pins each requirement**. Every RFC requirement appears exactly once in the
traceability matrix at the end.

## Architecture — where the tiling decision lives

**Decision.** A tiling is carried as candidate data on a `LifetimeBoundBuffer`
subclass, exactly as a core division already is, and it is chosen *and applied*
inside the scratchpad planning pass. There is **no new pass slot**, and no
representation with an identity of its own outside the buffer set.

This supersedes the RFC's pipeline placement (`draft-unified-tiling-cpsat.md`
§5, diagram at `:952-968`), which put a `unified_partition_solve` pass between
`passes.py:443` and `:448`, applied `coarse_tile` at the 448 slot, and left
451-452 and 455 to commit what it decided. Everything below that differs from
the RFC is a deliberate departure, and the RFC should be updated to match once
stage 1 lands.

### The buffer carries it — as one candidate list, not two

`CoreDivisionBuffer` (`plan_solver.py:137-166`) already is the pattern:
`core_divisions` (candidates), `cd_parent_matches` (pairwise compatibility
across an edge), `chosen_division` (the answer, written back). Tiling **extends
the element of that list** rather than adding a list beside it:

```python
@dataclass
class CoreDivision:
    output_splits: dict[int, int]
    reduction_splits: dict[int, int]
    tiling: TileSpec = TileSpec()   # new; the empty spec is untiled
```

**A tiling and a division cannot be two independent lists**, and that is the
single most important constraint on this representation. A division is only
meaningful relative to a tiling, in two separate ways:

- **The legal set changes.** `enumerate_work_division_candidates`
  (`work_division.py:753`) filters on each factor dividing its dim's size (which
  tiling shrinks, admitting *fewer* factors), on every tensor's per-core span
  being within `MAX_SPAN_BYTES` (which tiling also shrinks, admitting *more*
  splits), and on stick granularity via `adjust_it_space_for_sticks` (recomputed
  from post-tile extents). The per-tiling sets are therefore neither subsets nor
  supersets of one another. R2.3's over-tiling fix falls out of exactly this:
  tiling and dividing relieve the same span pressure, so checked jointly they
  reach a smaller tile count than tiling checked alone.
- **The encoding changes.** `splits_by_index_coeff` keys `output_splits` by each
  symbol's *coefficient in `write_index`*, and tiling rewrites index expressions
  (`_divide_ranges`, `_rescale_index`). The same logical split therefore gets a
  different key under a different tiling — so a `CoreDivision` carried across
  tilings is not merely illegal, it is **uninterpretable**. This is the failure
  shape the co-optimizing path already shipped once as coeff-keyed signature
  conflation.

Pairing them on one indexed candidate keeps every downstream mechanism as-is:
one index selects the joint choice, so `chosen_division` still answers it;
`_wrap`'s dispatch is untouched (`ilp_solver_ortools.py:391`); the `AddElement`
tables already hold per-candidate scalars and simply index a richer candidate;
and **`cd_parent_matches` needs no tiling counterpart** — a match is already
`(parent_idx, child_idx)`, and a candidate now carries its tiling, so the
"shared tiling shape" relation is computed by the existing
`_per_core_view_on_buf` machinery provided the view is taken on the tiled frame.

It also keeps the untiled path bit-identical: `TileSpec()` is the default, so
with `auto_coarse_tiling` off the field is inert and no construction site
changes — which is what R8.1's safe-rollout state requires.

One thing must change rather than staying inert.
`CoreDivisionBuffer.min_footprint` divides `size` by `cd.output_partition`; a
tiled op's own buffer is per-tile scratch, so it becomes
`ceil_div(self.size, cd.output_partition * cd.tiling.tile_count)`. That is the
LX-residency win appearing in the footprint math, so it is load-bearing, not
incidental.

The type is then really a partition config, which is what the RFC calls it
(R2.1's `PartitionConfig`). Renaming `CoreDivision` accordingly is the honest
cleanup and touches its ~20 references; extend it while the field is inert and
rename in a follow-up.

A `TileSpec` is an **ordered, outermost-first** tuple of
`TileAxis(host_dim, count, is_reduction)`. Ordered, where the splits are dicts,
because core divisions are a product and so order-free while tile levels nest —
swapping two levels is a different plan. `host_dim` is a positional index (into
`op_out_coords(op)`, or into the op's ordered reduction loop variables), which is
what `coarse_tile` ultimately consumes and what `op_it_space_splits` already
uses. `chosen_division = None` still means *undecided*, distinct from a chosen
candidate whose tiling is empty — undecided is a bug, untiled is a plan.

### There are no `hint_id`s in the representation

Today a tiled level *is* a `hint_id`: `coarse_tile`'s `levels` list carries
`(hint_id, count)` and nothing else, so which dimension each op tiles has to
travel separately, through the `op.dim_hints` side-channel that
`span_overflow_groups` fabricates and its caller stamps
(`passes.py:363-365`). That is three producers minting into one per-compile
counter namespace, which is what forced `_SPAN_OVERFLOW_HINT_ID = 10000` and
would have forced a third reserved base for the solver.

None of it is needed. Two ops belong in one loop nest when their tiles line up,
and that relation is already computed for divisions: `cd_parent_matches` holds
pairwise candidate compatibility across a named producer→consumer edge, decided
by **physical device-dim view equality** through `_per_core_view_on_buf`
(`pass_utils.py:1696`) — correct across reductions and reshapes, and keyed on
nothing but the buffer name. Because the tiling now rides on the candidate, that
same table covers tilings with no second field. So a level is not identified
globally; it is identified by *agreeing across an edge*, and:

- **Level identity** is unnecessary — a group is a run of consecutive ops whose
  chosen candidates carry the same `TileSpec`.
- **Per-op dim binding** is unnecessary as stored data — each candidate's own
  `TileSpec` states its own `host_dim`s.
- **`hint_id`s survive only as an apply-boundary artefact.** The adapter mints
  them, and it mints from a *derived* base — `max(hint_id present in the graph,
  default=-1) + 1` — so there is no reserved constant and no namespace
  convention to violate. `validate_coarse_tile_groups` (`coarse_tile.py:112`)
  cannot see a collision it could have raised on.

### It is applied as a pre-pass, and that is what removes the prediction problem

`ScratchpadAllocator.plan_allocation` (`allocator.py:146-168`) is a template
method:

```text
_run_passes(pre_optimization_passes, graph)   # graph mutation
buffers = _prepare_buffers(graph)             # buffer set built from the graph
allocation = _solve(buffers)
_post_solve(graph, allocation)                # _commit_divisions
_push_allocation(graph, allocation)
```

`ScratchpadOptimizationPass.apply_pass` already declares the contract coarse
tiling needs — *"The order and number of nodes in the graph may change as a
result of an optimization pass"* (`scratchpad/passes.py:31-32`) — and no
allocator supplies any pre-passes today (`allocator.py:119`), so the slot is
free. Applying tiling there means `_prepare_buffers` (`:177`, `:1504`) builds
its buffer set from the **already-mutated** graph, so `full_buf` and every tile
copy exist as real IR before any buffer object is constructed.

Three consequences, all favourable:

1. **The downstream re-decide problem disappears.** A solve at the RFC's 443-448
   slot is erased downstream — `_commit_user_splits` deletes
   `op_it_space_splits` (`work_division.py:935-944`) and `work_distribution`
   re-derives it (`:1670-1677`). Deciding *inside* 455 puts the decision
   downstream of both, so there is nothing left to erase it and 451-452 become a
   warm-start seed rather than a competitor.
2. **Prediction is not needed in order to apply.** R7.1's `PredictedBufferSet`
   exists because the RFC decides on a buffer set that does not exist yet. With
   the buffer set rebuilt from the mutated graph, sizes and lifetimes are
   measured, not predicted. Prediction becomes necessary only where the tiling
   *choice* has to be priced against residency before being applied — the solver
   stages — not at the apply stage.
3. **Ops inserted by tiling get divisions for free.** `_division_map`
   (`:1527-1556`) iterates every op in `graph.operations`, and
   `_fixed_core_division` (`:858`) already tolerates a missing
   `op_it_space_splits` (`getattr(op, ..., None) or ({}, {})`). Because the
   pre-pass runs before it, the inserted copy ops are enumerated and divided by
   the joint solver. No new work.

The cost is that two buffer-building rounds are needed once the solver chooses:
build tiled buffers → solve → apply → rebuild → place. Stage 1 avoids the round
trip by taking the choice as an input rather than solving for it.

### Axis naming is deferred, and it is real work

Deriving a logical axis identity from the dependency graph would let candidates
be enumerated per chain ("axis X ÷ K") instead of per buffer, and would make
diagnostics legible. `propagate_named_dims` does not carry it as-is: it is
user-gated (`_enabled` is False unless the driver called `name_tensor_dims`, or
the graph carries an in-graph `named_dims` hint — `:521-535`), and its fallback
for an unnamed dim is `_untracked_name` → `f"_untracked_{size}"` (`:112-119`),
which is size-keyed and therefore **collides for two distinct axes of equal
extent**. Real axis identity means union-find over `(buffer, dim)` pairs joined
by producer/consumer index agreement — a new pass, not a config flip.

Nothing before the solver stages needs it: the view-equality table gives the
relation without ever naming it.

## Staging — one PR per stage

**A stage is a PR.** The contract each one meets:

- **Lands green.** CI passes on merge. Where a stage leaves a capability
  deliberately unbuilt, its tests say so through the marker below rather than by
  failing.
- **Reviewable alone.** It cites the tree it changes, and its gate is checkable
  without reading the next stage.
- **Leaves no half-state.** Either nothing observable changes (a stage that adds
  unwired machinery) or the change is complete and proved.
- **Steps are its commits.** The step numbering below is the intended commit
  order inside the PR, so a reviewer can read the test step before the
  implementation step it pins.

| Stage | What lands | Requirements | Gate |
|---|---|---|---|
| **1** | Declarative tiling, applied | R2.1, R4.5, R7.3 (half) | pass-430's tiling reproduced through `TileSpec` with identical `loop_info` |
| **2** | The cost-function seam | R3.1, R3.6, R8.5 | no plan changes anywhere |
| **3** | Tiling option enumeration, reductions included | R1.1–R1.9 | enumerator set equals a brute-force reference; no nested/multi-level reduction option |
| **4** | Prediction and tiling-aware views | R2.6, R6.3, R7.1 | predicted view equals the view recomputed after `coarse_tile` |
| **5** | The solver: candidates, cuts, enablement | R2.2–R2.5, R4.1–R4.10, R5.1–R5.8, R6.1, R6.2, R7.2–R7.5, R8.1–R8.4 | gate on ⇒ end-to-end correct; `INFEASIBLE` ⇒ today's behaviour |
| **6** | Padding *(conditional)* | R10.1–R10.6 | opened by R10.3's measurement **or** by a tileability blocker |
| **7** | The M2 objective collapse *(deferred)* | R3.2–R3.5, R3.7 | spill-parity + no core regression vs a captured CP-SAT baseline |

```text
#3736 ──▶ stage 1 ──┐
                    ├──▶ 3 ──▶ 4 ──▶ 5
          stage 2 ──┘
R10.3 bench ─────────────────────▶ (opens stage 6)
stage 7 ─────────────────────────▶ any time, or never
```

Stages 1 and 2 are independent of each other and can run in parallel: stage 1
touches no solver, stage 2 touches no tiling. Both must land before stage 5,
which is the first to put a tiling in front of the solver.

**Stages 6 and 7 are off the critical path by design.** The R10.3 measurement is
independent of everything and should start at stage 1, because its answer is one
of the two things that decides whether stage 6 exists (the other is §3.0's
tileability argument). Stage 7 — collapsing the objective — is deferred
because it changes every existing plan for reasons unrelated to tiling, and
because the retained two-phase objective is enough for stages 3–8 to be built and
proved against (stage 2 says why). When a cost model is ready it arrives through
stage 2's `objective` parameter, so stage 7 can land before, after, or between
the tiling stages without reordering any of them.

**Why the middle is one large stage.** Stages 3, 4 and 5 could each be cut
finer — and stage 5 was three stages in an earlier draft. They are not, because
the seams fall in places where neither half can be judged alone: an enumerator
without reduction support, a solver with candidates but no cut variables, a model
with cuts but nothing applying it. Each of those lands green and proves nothing.
Where a seam is real the plan still uses it (stages 1, 2, 6, 7 are all small);
where it is not, it becomes a commit boundary instead.

#3736 lands ahead of stage 1 and does not depend on it: its marked combos raise
their own stub, so they xfail correctly with nothing else built. It is also the
experiment that settles whether xfail satisfies `mandatory_success`, so it should
merge before any stage adds more marked combos.

### Correctness comes from the model, not from checking its output

The RFC's §5 rule is that the solver emits only legal, systematically-applicable
plans: every admitting predicate — R4.7 cut compatibility, R2.6 view agreement —
must be *sufficient* for applicability, not merely necessary. **If the CP-SAT
model is respected, the output is valid**, so there is nothing to verify after the
fact. Two consequences for how these stages are gated:

- **No output-diffing gates.** A stage's gate asserts a property of the model, or
  the correctness of one applied plan against CPU. It never asserts that a new
  plan resembles an old one. Where a plan legitimately changes, the only question
  left is whether it is *worse*, and that is quality, not validity — stage 7 is
  the sole place this plan gates on such a comparison, and it says so.
- **Prediction error is quality, not correctness.** A sizing or lifetime
  misprediction leaves a legal plan that degrades to a spill and never a wrong
  address (R7.4). So R7.2 and R7.5's predicted-vs-realized records are
  diagnostics — they explain *why* a candidate was chosen when a plan
  disappoints — not gates that catch invalid output.

**The one check that stays is the detector for the premise failing.** A plan
`coarse_tile` cannot apply, or one whose residency was gated on a per-core view
the applied IR does not honour, means some predicate was necessary but not
sufficient — a modelling bug, not a bad input. That is what stage 7's post-apply
legality assertion is for, and why its firing must be a hard failure rather than a
degrade-to-greedy that would mask it. Under this principle it is not one guard
among many; it is the whole safety net, which raises rather than lowers the bar on
getting stage 4's sufficiency right.

### What the old step structure lost, and why

An earlier draft ordered the work as steps 0–5, with **all** tests written first
(steps 1–2, marked xfail) against stubs scaffolded in step 0, and implementation
following in steps 3–5. Per-stage PRs dissolve that:

- **Step 0's scaffolding mostly disappears.** Stubs existed so that a test
  written before its implementation had a name to reach. When a stage builds the
  real thing and its tests together, there is nothing to stub. The rows that
  survive move to the stage that owns them (see *Scaffolding, redistributed*).
- **A test-only PR is not a self-contained stage.** It either duplicates the
  test work of the stage that implements the feature, or splits one reviewable
  change across two PRs. So each stage carries its own tests.
- **What that would have lost, and how it is kept.** The virtue of test-first was
  that the highest-risk guards existed before the code — the R2.6 negative
  cache-sharing test above all. That is preserved by *step order inside the
  stage*: the test step precedes the implementation step it pins, so the PR's own
  history shows the test failing and then passing. Cheaper than a separate PR and
  strictly more reviewable.
- **The marker stays**, for genuine forward references across stages — a stage
  may land tests for a later stage's capability, marked. #3736 is exactly that.

### The expected-failure marker

An xfail marker records *that* a test failed, never *why*. Both standard
mechanisms are too broad for a phase this long:

- `pyproject.toml` sets no `xfail_strict`, so `pytest.mark.xfail` does not even
  fail on unexpected pass.
- `unittest.expectedFailure` is strict about that, but absorbs **any**
  exception. A test opening with
  `ts_inductor_config.patch(unified_tiling=True)` before that gate exists raises
  `AttributeError`, is recorded as a satisfied expectation, and *keeps* being
  satisfied after the feature lands — including when it lands wrong.

So the expectation is narrowed to one declared cause: **a marked test may fail
only by reaching an unbuilt stub.** Every stub raises the built-in
`NotImplementedError` under a fixed message prefix —
`raise NotImplementedError("unified-tiling: enumerate_tile_options")` — and every
marked test uses one shared decorator in place of `unittest.expectedFailure`.

**This has landed** in PR #3736 as
`test_solver_auto_coarse_tiling.py:expected_unimplemented`, body identical to the
below, with the first stub already in place
(`NotImplementedError("unified-tiling: config.auto_coarse_tiling")`, raised
before the compile so the marked combos never reach the device). Its docstring
notes it "belongs in `utils_inductor.py` once a second suite wants it" — stage 2
is that second suite, so **the remaining job is the promotion, not the
implementation**: move it to `tests/inductor/utils_inductor.py` and re-import it
in `test_solver_auto_coarse_tiling.py`. Doing the move in stage 2 keeps
`test_scratchpad_solver.py` from importing a peer test module.

```python
def expected_unimplemented(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            fn(self, *args, **kwargs)
        except NotImplementedError as exc:
            pytest.xfail(f"not built yet: {exc}")
        else:
            self.fail(f"{fn.__name__} passed — remove @expected_unimplemented")
    return wrapper
```

It cannot wrap `unittest.expectedFailure`: that absorbs whatever reaches it, so
a non-`NotImplementedError` failure could never be surfaced through it.
Imperative `pytest.xfail()` is the repo's existing idiom for a runtime-decided
expected failure (`indirect_access_common.py:450`, `:474`, called from
`unittest.TestCase` subclasses).

| Outcome | Plain `expectedFailure` | `expected_unimplemented` |
|---|---|---|
| Reached an unbuilt stub | xfail | xfail, reason names the stub |
| Passed cleanly | fail (strict) | fail — "remove the marker" |
| Real assertion failure | **xfail — hidden** | fail |
| Typo, `AttributeError`, `ImportError` | **xfail — hidden** | fail |

Two properties follow. As implementation advances, a test that xfailed on stub A
starts running further and xfails on stub B — the reason string tracks the
frontier with no edits. And because the marker is imperative rather than a
pytest mark, `-m 'not xfail'` does not deselect these tests; they still execute
and still xfail at runtime, which is the behaviour we want.

**Residual risk, accepted.** `_inductor` already raises bare
`NotImplementedError` in five places these tests can traverse —
`graph_editor.py:247` (inside `push_allocation_with_clone`, so every
boundary-clone test), `propagate_named_dims.py:650` (a pass at `passes.py:426`),
`spyre_kernel.py:491`, `codegen/compute_ops.py:745`, `codegen/bundle.py:546`.
If one fires during a marked test it will read as "not built yet" rather than
failing red, masking a real backend gap. The message prefix is the tell: an
xfail whose reason does not start `not built yet: unified-tiling:` came from one
of those five. Worth a glance whenever the xfail set changes shape unexpectedly.
A dedicated exception subclass would make this structural rather than
conventional; the built-in was chosen deliberately over that.

**Phase completion criterion.** `grep -rn "unified-tiling:" torch_spyre/`
returning nothing means every stub is built. Pair it with the decorator's
*call sites* — `grep -rn "@expected_unimplemented" tests/` — returning nothing:
the two must empty together, and either one alone emptying is a bug. The `@`
matters, since the definition in `utils_inductor.py` and its import in each
suite are both live until the very last marker goes, so an unanchored grep can
never reach zero.

**Confirm that xfail satisfies `mandatory_success` before any stage relies on
it.** All target suites carry `unlisted_test_mode: mandatory_success`, and so
does #3736's new yaml (under `labels: [unit, regression, integration, trunk]`).
That xfail satisfies that mode is an assumption about the OOT runner, not
something this repo states anywhere, and it is load-bearing for every marked test
in the phase. If the runner drives tests through `unittest` rather than pytest,
`pytest.xfail()`'s `XFailed` lands as an error and the whole marked set is a hard
CI failure. #3736 is the cheap experiment — nine marked combos and a real yaml,
so its CI run answers the question outright. Read that result first.

*(RFC testing item 1 claims `test_coarse_tiling.py` has no CI config yaml. That
is stale — `test_coarse_tiling_config.yaml` exists, with
`labels: [core, full, device_critical]`.)*

### Scaffolding, redistributed

What survives of the old step 0, and which stage now owns it:

| Deliverable | Where | Stage | Requirement |
|---|---|---|---|
| Promote `expected_unimplemented` out of the #3736 suite | `tests/inductor/utils_inductor.py` | 2 | — |
| Lift `_allocation_fingerprint` (`:803`) onto `BaseTestScratchpadUsage` (`:73`) so a suite that is not `CoOptAllocatorIntegrationTests` can use it | `test_scratchpad_use.py` | 7 | R3.5 |
| Capture today's CP-SAT plans for the four prescribed models as the parity baseline — recorded nowhere today | `tests/inductor/` fixture | 7 | R3.5 |
| `TileAxis`, `TileSpec`, `CoreDivision.tiling` with derived properties | `scratchpad/plan_solver.py:93` | 1 | R2.1 |
| `objective` keyword-only param on both ABCs and all four `plan_layout` overrides | `plan_solver.py:260,297`; `ilp_solver_ortools.py:348`, `greedy_solver.py:134`, `firstfit_bestfit_solver.py:186`, `simulated_annealing.py:122` | 2 | R3.1 |
| Ignore a non-`None` objective, warn once — in the **three placement-only files only**, explicitly *not* `ilp_solver_ortools.py` | `greedy_solver.py`, `firstfit_bestfit_solver.py`, `simulated_annealing.py` | 2 | R3.6 |
| `CostSpec` type accepted by the signature | `scratchpad/cost_expr.py` | 2 | R3.1 |
| `CostExpressionError`, `validate()`, `lower()` | `scratchpad/cost_expr.py` | 7 | R3.3 |
| `sym_is_lx`, `sym_inv_cores` symbol properties | `plan_solver.py` | 7 | R3.7 |
| `enumerate_tile_options` | new `wsr/enumerate_tilings.py` | 3 | R1.1 |
| `PredictedFrame` / `PredictedBufferSet`, including the reduction accumulator / fill / combine | new `wsr/tile_prediction.py` | 4 | R7.1 |
| `unified_tiling`, `auto_coarse_tiling` gates, default off | `config.py` (match the `os.environ.get(...) == "1"` style at `:22-25`) | 5 | R8.1, R5.8 |
| ~~`_UNIFIED_TILING_HINT_ID = 20000`, a reserved `hint_id` base~~ — **withdrawn**: no `hint_id` reaches the representation, and the apply adapter mints from a derived base (*Architecture*) | — | — | R4.5 |
| CI config yaml for `test_enumerate_tilings.py` (#3736 already added the one for `test_solver_auto_coarse_tiling.py`) | `tests/configs/torch_spyre_tests/inductor/` | 3 | — |

**Four, three, and five are three different counts, and the R3.6 row is the one
that has to be narrow.** There are four `plan_layout` definitions to re-sign
(R3.1) but only three files to put the ignore-and-warn in, because
`BestFitLayoutSolver` subclasses `FirstFitLayoutSolver` (`:248`) and inherits its
`plan_layout` — one warn covers both registry entries. And R3.6's own "four
placement-only solvers" counts *registry keys*
(`_PLACEMENT_SOLVERS`, `allocator.py:2108-2113` = greedy / bestfit / firstfit /
simulated_annealing), with `cpsat` handled ahead of the registry at `:2159` as
"the one solver for which `objective` is honoured". An earlier draft of this row
said "the four solvers above", which swept `ilp_solver_ortools.py` into the
ignore set and would have landed a warn-and-ignore in the one path stage 2
exists to wire up.

R3.1 is source-compatible. Four in-tree callers reach `plan_layout`, and none
passes `log_lx_usage` positionally: `allocator.py:184` and `:1348` pass it by
keyword, `allocator.py:1468` and `simulated_annealing.py:81` pass only
`buffers`.

## Stage 1 — Declarative tiling, applied

The first landable slice, and deliberately not a solver. It answers one
question — *can a tiling be stated as data and applied to a real graph through
the scratchpad pass?* — and it is provable, because the two producers that ship
today give a ground truth to reproduce.

**Deliberately excluded.** Candidate *enumeration* (the choice is an input, not
a search), the objective, cut variables, prediction, axis naming, and reductions
beyond whatever the ground-truth graph already exercises.

### 1.1 The premise spike (do this first, before writing any code)

The RFC's entire motivation rests on one claim: tiling buys LX residency through
the run's *interior*, because `_propagate_tiled_op` sets `output_tiled_dims = []`
so the per-tile scratch stays LX-eligible.

Static reading confirms it — `_is_tiled_advancing`'s docstring
(`scratchpad/utils.py:218-234`) states the rule verbatim: "A loop-internal
buffer (e.g. drained by a copy op every iteration) can be tiled yet have its own
write pinned at a fixed address; such a buffer is LX-eligible."

Confirm it empirically anyway, on today's code, before building anything on it:
take a hint-tiled two-group graph, run the allocator, and assert the
*Background* table row by row — interior scratch and read-side tile copies carry
a `None` `residency_reason`; `full_buf` and the write-side copy op do not. This
is RFC testing item 7's positive half, and it costs a day. If it does not hold,
the whole phase needs rescoping, which is why it is the first commit of the first
PR rather than a task inside a later stage.

### 1.2 The representation

| Deliverable | Where |
|---|---|
| `TileAxis`, `TileSpec` — frozen, ordered, hashable so `==` is the "same tiling shape" test; `is_untiled` / `depth` / `tile_count` / `is_clean` / `label` | `scratchpad/plan_solver.py`, beside `CoreDivision` |
| `CoreDivision.tiling: TileSpec = TileSpec()` — one candidate list, not two (*Architecture*); inert while `auto_coarse_tiling` is off, so no construction site changes | `scratchpad/plan_solver.py:93` |
| `CoreDivisionBuffer.min_footprint` divides by `output_partition * tiling.tile_count` — the only non-inert change, and where the LX-residency win enters the footprint math | `scratchpad/plan_solver.py:157-166` |

### 1.3 Lowering and the pass

| Deliverable | Where |
|---|---|
| `reduction_loop_vars(op)` — the ordered reduction loop variables, made public and *reused* by `_loop_var_to_reduction_ranges_pos` so the derivation has one copy | `wsr/coarse_tile.py:742` |
| `tile_spec_to_dim_hints(op, spec, hint_ids)` — the lowering; `_dims_to_hints` (`coarse_tile_span_overflow.py:152`) is already this for output axes, the reduction case is the inverse of `reduction_loop_vars` | new `scratchpad/coarse_tiling.py` |
| Group derivation: consecutive-run over `graph.operations`, breaking on an untiled op or a spec change | new `scratchpad/coarse_tiling.py` |
| `CoarseTilingPass(ScratchpadOptimizationPass)` — derive groups, mint hint ids from the derived base, stamp `dim_hints`, `validate_coarse_tile_groups`, `coarse_tile(graph, groups, group_idx_offset)` | new `scratchpad/coarse_tiling.py` |

**Group derivation is a consecutive run, not connected components.** Contiguity
is a hard requirement, not an optimization: `validate_coarse_tile_groups`
(`coarse_tile.py:112-121`) rejects a hint scope split across two groups, and
`_apply_plan` relies on the group occupying one contiguous stretch. A component
therefore can never span a break, so the correct derivation is
`hints_to_coarse_tile_groups`' own shape (`coarse_tile_hints.py:291-306`) with
`_hint_key` replaced by the chosen candidate's `TileSpec`. `cd_parent_matches`
is where *non-identical but compatible* pairs will be expressed; stage 1 groups
on spec equality, which is span overflow's own rule.

**`group_idx_offset` is derived, like the hint base.**
`max(loop_group_id[0] present, default=-1) + 1`, exactly as
`_maybe_coarse_tile_span_overflow` does it (`passes.py:373-379`), so a
hint-driven group stamped at pass 430 cannot collide.

**A stickified axis is tileable, but only in whole sticks — and only here does
that come free.** Stick identity is invariant under tiling
(`_stick_host_dim`, `coarse_tile.py:460-462`). The user-hint path runs *pre*
stickification (`passes.py:430`), where the layout is derived *from* the divided
ranges, so nothing can cut a stick and nothing checks `elems_per_stick`. This
pass runs *post* stickification, so it is in the strict regime: on the
stick-carrying host dim the per-tile extent must satisfy
`tile_size % elems_per_stick == 0` (`_post_tile_stick_alignment_error`,
`span_overflow_hint_analysis.py:263-293`), re-checked per input against its own
layout (`_input_stick_alignment_error`), failing closed when the stick dim
cannot be identified. Without the check `_resize_device_layout` **ceils** to a
partial stick — wrong addressing with no exception. Stage 1 reproduces a
pre-stickification tiling and so cannot violate this; stage 3 owns enforcing it
on enumerated options.

### 1.4 How it is proved — replicate the emitted OpSpecs

Take a graph the hint path tiles today, express that same tiling as `TileSpec`s,
apply it through `CoarseTilingPass`, and assert **the emitted spec tree is
identical** to what pass 430 produces.

`OpSpec` / `LoopSpec` (`_inductor/op_spec.py:184`, `:250`) is the right thing to
compare, not `loop_info`. `loop_info` is an internal annotation — matching it says
the two paths agree about their own bookkeeping. The spec tree is the backend's
declarative output, so matching it says they agree about *what the device is told
to do*, which is the claim the representation actually has to earn. Every tiling
consequence shows up there: loop nest structure and trip counts as `LoopSpec`
nesting, per-tile extents and `tiled_symbols` on each `OpSpec`, and the inserted
`full_buf` / copy / combine ops as spec entries in their own right.

The machinery exists and needs no new harness. `capture_op_specs()`
(`indirect_access_common.py:124`) patches `SpyreAsyncCompile.sdsc` to record the
spec list per kernel and return a no-op runner — it skips SuperDSC generation and
executes nothing, so the comparison costs a compile rather than a device run, and
`flatten_op_specs` / `flatten_entries` (`:160`, `:176`) walk the nesting.
`test_coarse_tile_e2e.py` already asserts against this structure by counting
`LoopSpec(` occurrences and regexing `OpSpec` blocks (`:3663`, `:5271`), so
structural comparison is the file's existing idiom — this just makes it an
equality against a captured reference instead of a spot-check.

Keep the `loop_info` comparison as the *inner* check rather than dropping it.
Spec-tree equality tells you **that** the paths diverged; `loop_info` equality
tells you **where**, since it is the annotation everything downstream reads. PR
#3736's `_label_tiling` / `_Level` reader is the right tool for that half — it
reads trip counts per op and attributes each level to the scope that asked for it.

Also pinned here: R4.5's group round-trip and the derived `group_idx_offset`, and
R7.3's ordering half — the pass runs inside `plan_allocation` and leaves the op
count unchanged when it does nothing.

## Stage 2 — The cost-function seam

Retains today's objective and changes no plan. Its whole job is to put the
injection point in place, so a cost model can be substituted when one is ready
without the tiling stages waiting on a reweighting first.

**What is deferred.** R3.2's collapse of the two-phase lexicographic solve into a
single weighted `Minimize` changes every existing CP-SAT plan with tiling off, for
reasons that have nothing to do with tiling. It moves to stage 7, off the
critical path, and takes R3.3, R3.4, R3.5 and R3.7 with it.

**Why the tiling axis does not need it.** The retained two-phase solve already
expresses tiling's *benefit* without being told to: phase 1 minimizes spill over
`eff_size`, and stage 1's `min_footprint` divides by `tiling.tile_count`, so a
tiled candidate is simply a smaller one. The space cost of what tiling
materializes is visible too — `full_buf` and the boundary copies enter the buffer
set as real IR (*Architecture*), so they occupy LX and contribute spill like any
other buffer.

**What the retained objective cannot say.** Two things, and both need covering
elsewhere:

- **The time cost of tiling is invisible.** Extra copy ops execute every
  iteration; nothing in `sum(spill_cost)` or `sum(cores)` prices that. A cost
  model is what fixes this, which is the point of the seam.
- **There is no weighting, so tile depth is unbounded.** Phase 1 is
  *lexicographically* absolute on spill — it will accept arbitrarily many tile
  levels to avoid one spilled buffer, because a deeper nest is never worse by that
  measure. Nothing in the model bounds it. Two things outside the model do: the
  enumerator's cap (R1.7, `_MAX_AUTO_TILE_SPLIT_COUNT = 64`) and R1.2's tiered
  ordering with truncation from the tail. Both are stage 3, and this deferral
  makes them load-bearing rather than conveniences — worth saying out loud,
  because a reader of stage 3 alone would take the cap for a performance guard.

| Work item | Where | Requirement |
|---|---|---|
| `objective` keyword-only param, typed `CostSpec \| sympy.Expr \| None`, on both ABCs and all four `plan_layout` overrides | `plan_solver.py:260,297`; `ilp_solver_ortools.py:348`, `greedy_solver.py:134`, `firstfit_bestfit_solver.py:186`, `simulated_annealing.py:122` | R3.1 |
| Ignore a non-`None` objective, warn once — in the **three placement-only files only** | `greedy_solver.py`, `firstfit_bestfit_solver.py`, `simulated_annealing.py` | R3.6 |
| CP-SAT: `objective=None` selects today's two-phase solve **verbatim**; a non-`None` objective raises `NotImplementedError("unified-tiling: objective lowering")` | `ilp_solver_ortools.py` | R3.1 |

**Raise, don't ignore, on the CP-SAT path.** R3.6's ignore-and-warn exists for the
placement solvers, where the parameter is there only for ABC conformance. CP-SAT
is the one solver that must honour an objective (`allocator.py:2159`), so
accepting one and quietly dropping it would be a lie that survives into stage 7.
A declared stub under the marker prefix is honest, and it shows up in the
completion grep (*Staging*).

**Gate.** No plan changes anywhere. Every existing suite passes unmodified, which
needs no baseline fixture and no comparison — the point of retaining the
objective is that there is nothing to compare.

### 2.1 Tests

Device-free, in `test_scratchpad_solver.py`.

| Class | Covers |
|---|---|
| `TestObjectiveSeam` | R3.1 — `objective` keyword-only on both ABCs and all four overrides, accepts a `sympy.Expr`; R3.6 — the three placement files ignore it and warn once; CP-SAT raises on a non-`None` objective |

R8.5 needs no new work: it is covered by the existing
`TestCpSatAllocatorFallback`.

## Stage 3 — Tiling option enumeration

Pure and unconsumed: the enumerator answers "what tilings could this op take",
nothing calls it yet, and the stage lands green with no behaviour change.
**Reduction axes included** — see below.

### 3.0 The enumeration strategy: exact divisors

A split count is admissible only if it **divides the dim exactly**, because
coarse tiling emits equal-sized loop tiles. That is not a new rule;
`_split_candidates_for_host_dim` already computes the divisor set
(`{i, full_size // i}` over `i` up to `isqrt(full_size)`) and then filters it for
stick alignment and the Reduction unit-extent rejection. Stating it as *the*
strategy has two consequences worth writing down, because both are visible to
users and neither is a bug:

- **A prime dim is effectively untileable.** The divisors of a prime `p` are
  `{1, p}`. Splitting `p` ways leaves unit tiles, which the Reduction path already
  rejects (`full_size // split > 1`) and which is degenerate for Pointwise, so in
  practice a prime extent gets the untiled option and nothing else. R1.3
  guarantees the untiled option is always present, so this is a missed
  opportunity, never a failure.
- **Padding is the lever that opens divisors, and that is a second reason to
  want it.** Padding a dim to a composite extent makes splits available that the
  true extent forbids. The plan currently justifies the discretionary pad
  (R10.2/R10.5) only by R10.3's unaligned-`K` matmul measurement; *tileability* is
  an independent motivation, and a stronger one, since it can turn an untileable
  op into a tileable one rather than shaving a constant factor. Recorded in
  stage 6, which should not be read as gated solely on the matmul number.

### 3.1 Reductions are enumerated here, not later

An earlier draft deferred reduction-axis options to a separate late stage. They
belong here, for the same reason stage 5 is one PR: an enumerator that emits only
output-range options, with reduction support arriving later, is an intermediate
state whose output nothing can price.

Folding them in has a dependency that has to move with them. A reduction-tiled
candidate materializes an accumulator, an identity fill and a combine op
(`_propagate_tiled_reduction_op`, `coarse_tile.py:2501`), so the predictor must
predict those or the candidate cannot be costed — that is stage 4's work, and it
moves forward with this. The *solver's* half — widening R4.6's pin and making the
carry rejection a group constraint — stays in stage 5 (§5.3), because both need
groups, which are a solver output.

| Work item | Where | Requirement |
|---|---|---|
| Emit single-level reduction options; never nested or multi-level | `wsr/enumerate_tilings.py` | R1.8 |
| Do **not** gate on `_validate_reduction_tiling` (`:1233`) — it over-approves the known wrong-numerics shapes | enumerator | R1.4 |

Test extension: assert **no** nested output+reduction or multi-level reduction
option is emitted, anchored on the exact shapes that are known-wrong today in
`test_coarse_tile_e2e.py`:

- `correctness=False`, "nested tiling + reduction correctness bug" —
  `test_min_2d_512x256_reduce_dim0_A4_B4` (`:706`),
  `test_min_2d_512x256_reduce_dim1_A4_B4` (`:747`),
  `test_min_3d_512x256x256_reduce_dim2_A4_B2_C4` (`:802`).
- `@pytest.mark.skip`, "inconsistent loop_count across reduction fill/combine
  nodes" — `test_min_3d_..._reduce_dim0_A4_B2_C4` (`:765`), `..._dim1_...`
  (`:784`), `test_add_min_3d_..._reduce_dim0/1/2_A4_B2_C4` (`:950`, `:976`,
  `:1002`).

Every admitted option must apply **and** match CPU (R1.6) — applicability alone
would pass on all eight, because their failure mode is a silent wrong answer.

### 3.1 Expose the reused predicates

| Work item | Where | Requirement |
|---|---|---|
| Expose the R1.4 predicates as reusable helpers; `_search_min_cost_tile_plan` becomes a thin ranked wrapper | `wsr/span_overflow_hint_analysis.py` | R1.4, R1.2 |

M6 forbids reimplementing a predicate that exists, and the stick-alignment rules
(§1.3) are the reason it matters here rather than being a style preference:
`_split_candidates_for_host_dim` already composes exact divisibility,
`_MAX_AUTO_TILE_SPLIT_COUNT = 64`, the Reduction unit-extent rejection, and both
stick-alignment checks. Enumerating without them yields options that apply
cleanly and compute the wrong answer.

### 3.2 The enumerator

| Work item | Where | Requirement |
|---|---|---|
| `enumerate_tile_options`, output ranges only | `wsr/enumerate_tilings.py` | R1.1–R1.5, R1.7, R1.9 |

### 3.3 Tests

New `tests/inductor/test_enumerate_tilings.py` with its config yaml — not
`test_scratchpad_solver.py`, since the enumerator is a `wsr` module and its tests
need no solver. It covers R1.1–R1.5, R1.7, R1.9, and later R1.8.

Two cases carry more weight than the rest:

- **R1.9's silent-failure guard.** For an op under *no* span pressure but with a
  legally splittable host dim, assert the enumerator returns more than the
  untiled option. Built on `_candidate_host_dims` (`:911`) alone it returns one
  option, the solver never tiles that op, and the LX-residency motivation
  quietly does nothing — no error, no warning.
- **R1.1 completeness.** Brute-force reference on small shapes; the enumerator's
  set must equal it.

R1.6's output-range half also lands here, in `test_coarse_tile_e2e.py`: every
enumerated option must apply **and** match CPU. Applicability alone would pass on
the known wrong-numerics shapes.

## Stage 4 — Prediction and tiling-aware views

The highest-risk stage, isolated so its blast radius is its own PR. R2.6 gates
*residency* on per-core views taken on a predicted frame; a wrong view grants
residency on a slicing agreement that does not hold — wrong data, not a
mispredicted size, and explicitly outside R7.4's degrade-to-spill safety. The
RFC calls it "the highest-risk area in the design", and this repo has already
shipped one bug of that shape (coeff-keyed signature conflation in the
co-optimizing path).

### 4.1 Why prediction is its own module

`GraphEditor` (`scratchpad/graph_editor.py`) is the wrong place to put it: it
does one thing — insert a Pointwise `clone` for LX boundary cloning
(`push_allocation_with_clone`, `:103`) — has no generic insert-op or
allocate-buffer primitive, rejects anything that is not `Pointwise`/`Reduction`
(`is_rewritable_consumer`, `:271`), and knows nothing about loop groups.

Nor does it need to be. Every mutation this phase performs already exists in
`coarse_tile.py`: `_allocate_full_buffer` (`:1519`), `_insert_copy_op`
(`:1657`), `_insert_read_copy_ops` (`:1907`), `_insert_combine_op` (`:2307`),
`_insert_reduction_copy_op` (`:2404`). R4.5 is explicit that "no new application
path is introduced". What is missing is the **inverse** of a mutation: a pure
predictor reporting what `_apply_plan` *would* insert without inserting it. That
is its own concern and gets its own module.

**What prediction is for, now that applying does not need it.** The pre-pass
placement (*Architecture*) rebuilds the buffer set from the mutated graph, so
sizes and lifetimes at *apply* time are measured rather than predicted. That
removes prediction from the apply path entirely — but not from this stage,
because the solver still has to *price* a tiling it has not applied: it compares
candidates, and a candidate not chosen is never built. The R7.2/R7.5
predicted-vs-realized comparison gets *stronger* for it — the realized side is
now a direct measurement of the same graph the placement solve runs on, not a
reconstruction.

Two boundaries hold this together:

- **`coarse_tile.py` gets one edit** — `_propagate_tiled_op`'s branch condition
  (`:1280`, the `_find_outside_consumers` test and `_full_buffer_read_deps` at
  `:1425`) extracted to a module-level pure predicate
  `boundary_role(op, ...) -> BoundaryRole`, which `_apply_plan` calls where it
  currently inlines the condition. No behaviour change, one copy of the rule.
  Dependencies stay one-way (`tile_prediction` → `coarse_tile`,
  `tile_prediction` → `span_overflow_hint_analysis`); putting the predicate in
  the new module instead would create a cycle. **Note the reduction path has its
  own, different rule** — `_propagate_tiled_reduction_op` sets
  `output_tiled_dims = []` unconditionally (`:2645`) with no loop-internal early
  return, and calls `_find_outside_consumers` separately (`:2656`). Because stage 3
  now enumerates reduction options, `boundary_role` must cover that rule **in this
  stage**, not a later one, or the extraction reintroduces the second copy it
  exists to prevent.
- **The solver must not import `tile_prediction`.** `ilp_solver_ortools.py` sees
  only buffers and tables; the allocator calls the predictor and hands results
  across. That is what keeps `test_scratchpad_solver.py` device- and IR-free,
  and it is easy to breach by accident once the module exists.

### 4.2 Work items

| Work item | Where | Requirement |
|---|---|---|
| Extract `_propagate_tiled_op`'s branch condition into `boundary_role()`; `_apply_plan` calls it | `wsr/coarse_tile.py:1280` | — |
| `PredictedFrame` — divided ranges + resized layout, per `(op, TileSpec)`; composes `_planned_tile_extents_per_level` (`coarse_tile.py:309`) and `_post_tile_layout_for_splits` (`span_overflow_hint_analysis.py:175`) | new `wsr/tile_prediction.py` | R7.1 (frame half) |
| `PredictedBufferSet` — the buffers a candidate plus cut assignment materializes (`full_buf`, write copy, read copies), at R7.5's interstitial tick coordinates | `wsr/tile_prediction.py` | R7.1 (buffer-set half) |
| `_prepare_per_core_view` / `_per_core_view_on_buf` accept a predicted frame | `pass_utils.py:1467`, `:1696` | R2.6 |
| `_views_for_divs` `prep_cache` key gains the tile | `allocator.py:2078` (key at `:2092`, coeff at `:2095`) | R2.6, R6.3 |

The RFC requires R2.6 and R7.1 to "share one predictor; they must not drift
apart" — one module makes that structural rather than a promise, which is the
argument against putting the frame beside the enumerator and the buffer set
beside the solver.

### 4.3 Tests

| Class | Covers | RFC test item |
|---|---|---|
| `TestTilingAwareViews` | R2.6 — `prep_cache` key includes the tile; **negative**: two candidates differing only in tiling must not share an entry | 8 |
| `TestPerCoreViewPrediction` | R2.6 — predicted view equals `_prepare_per_core_view` recomputed after `coarse_tile` | 8 |

R7.1 is pinned by a deep-compare of the graph before and after the predictor
runs: it must mutate no IR. R6.3 is `TestTilingAwareViews` plus the R3.7
assertion that relayout symbols are absent.

The negative cache-sharing test is the cheapest guard against this stage's
central risk, and it is the first commit of the PR — before the code it pins.

## Stage 5 — The solver: candidates, cuts, and enablement

The core of the feature, and **deliberately one PR** rather than three. Splitting
it at the natural seams would leave the solver in an intermediate state that
cannot be judged: candidates without cut variables produce groups whose
boundaries nothing models, and cuts without enablement produce a model nothing
applies. Each is green in isolation and meaningless in isolation. So the seams
become commit boundaries inside one review, which is what the *steps are commits*
contract is for.

It is the largest stage by some margin. That is a real cost, accepted for the
reason above; the mitigation is that §§5.1–5.4 are independently reviewable
commits in that order, each with its own tests.

Depends on stage 2 (the objective its candidates are priced by), stage 3 (the
options it chooses among) and stage 4 (the views residency is gated on).

### 5.1 Candidates on the buffer

| Work item | Where | Requirement |
|---|---|---|
| `_enumerate_core_divisions` → candidate enumeration, the inner loop running once **per tiling option** and stamping `tiling` on each `CoreDivision` it emits — required, not an optimization (*Architecture*: both the legal set and the coeff encoding are tiling-relative); dedup keyed on `(splits, tiling)` | `allocator.py:1558` | R2.2, R2.3, R2.4, R2.5 |
| `_cd_parent_matches` → `_config_matches` on tiling-aware views | `allocator.py:1973` | R2.6 |
| **No new wrapper.** Pairing the tiling onto the candidate (*Architecture*) means one index still selects the joint choice, so `_CoreDivisionBufferWithCpVars` (`:243`) needs only two more per-candidate `AddElement` tables (`tile_count`, `full_size`), read by the retained objective through `eff_size`. The two-level `tile`/`div` index and its `AddAllowedAssignments` are **withdrawn** — there is no pair to constrain, and symbol binding waits for stage 7, since until then there is no expression to bind against | `ilp_solver_ortools.py` | R4.8 (half) |
| `full_size` / `boundary_view` tables | `ilp_solver_ortools.py` | R4.10 |

#### Tests

| Class | Covers | RFC test item |
|---|---|---|
| `TestConfigEnumeration` | R2.2 per-tiling division sets, R2.4 seed pair retained + dedup on `(splits, tiling)`, R2.5 `Unsupported` on the empty set | 5 |
| `TestOverTilingFix` | R2.3 — strictly smaller tile count than the `core_split_estimate = 1` path | 9 |

`TestOverTilingFix` needs a **span-pressure** model — a shape that forces tiling
today — since that is the only way to observe the fix;
`_E2E_SHAPE = (1, 8195, 256, 64)` from `test_span_overflow_hint_analysis.py:216`
is the known-overflowing shape.

**Measure compile time at this stage, not later.**
`enumerate_work_division_candidates` now runs per tiling option (R2.2) and
`_views_for_divs`' sympy prep is no longer candidate-invariant (R2.6). Both were
built assuming they are paid once per op.

### 5.2 Cuts and groups in the model

| Work item | Where | Requirement |
|---|---|---|
| Direction-indexed `cut_parents`/`cut_children` claim dicts; `_add_cut_equalities` sweep in `_run` | `ilp_solver_ortools.py` | R4.1, R4.2, R4.7 |
| Read copies as optional rectangles; row-3 eviction implication | `_add_no_overlap_2d`, `:568` | R4.9, R4.8 (rest) |
| Pin `cut = 1` on both boundaries of **hint-driven** reduction-tiled ops | wrapper | R4.6 (half) |
| Warm start via `AddHint`; `num_search_workers = 1`, `random_seed = 0`, `max_deterministic_time` | `_run`, `:452-463` | R8.2, R8.4 |

**"Omit reductions" does not omit R4.6.** Hint-driven reduction-axis tiling
exists today and applies **pre**-stickification at `passes.py:430`. This solve
runs post-stickification, so a graph reaching it can already contain
reduction-tiled ops whether or not the enumerator emits any. R4.6's pin is
therefore an obligation here; §5.3 only widens it from "hint-driven" to
"hint-driven or solver-chosen". Leaving it out lets the solver fuse a
reduction-tiled op into a group, which
`_plan_is_loop_invariant_at_reduction_levels` (`coarse_tile.py:559`) then
rejects at apply time — an illegal emission, which §5 of the RFC classes as a
hard failure rather than an R8.3 fallback.

#### Tests

| Class | Covers | RFC test item |
|---|---|---|
| `TestCutTables` | R4.1 totality, R4.2 untileable pins + structural contiguity, R4.7 directionality and fail-closed, claim orientation after the equality sweep | 4 |
| `TestCutConsequences` | R4.8 (no `boundary_op ⟹ ¬in_buffer` constraint; row-3 eviction only), R4.9 optional rectangles, R4.10 `full_size`/`boundary_view` | 7 |
| `TestReductionPinning` | R4.6 — both boundaries pinned, singleton group | 4 |
| `TestWarmStart` | R8.2 — `AddHint` from the heuristic plan | 11 |
| `TestUnifiedTilingDeterminism` | R8.4 — identical plans across runs | 10 |
| `TestOpOrderInvariant` | R4.4 — op order unchanged across the solve | — |
| `TestBoundaryBufferLxStatus` | R4.8–R4.10, R7.5 — the *Background* table; the 1 / 2 / 0 rule; the untiled→tiled row | 7 |

### 5.3 Reductions in the model

The enumerator already emits reduction options (stage 3) and the predictor
already predicts what they materialize (stage 4), so what is left here is the
solver's side of them.

| Work item | Where | Requirement |
|---|---|---|
| Widen the R4.6 pin from hint-driven to hint-driven **or** solver-chosen | solver wrapper | R4.6 (rest) |
| Reject carry-propagating recurrences as a **constraint on groups**, not a filter on options | solver | R1.4, R1.8 (solver half) |
| `enable_reduction_tiling` keeps its default and meaning | `config.py:82` | R5.6 |

**The carry rejection is group-scoped, so it cannot be an enumeration filter.**
`_seed_buffer_for_carry(op, group_ops)` needs the group (`coarse_tile.py:265`),
and groups are a solver *output* — the per-op enumerator has none. It is
correctly a zero-mutation detector, so the reuse is sound, but it has to become a
constraint on groups containing a reduction-tiled op with a carry seed. Today it
raises `Unsupported` at plan time, which under §5 of the RFC is an illegal
emission the model must rule out by construction.

### 5.4 Enablement

The commits that turn the feature on. Everything before them is unwired or inert.

| Work item | Where | Requirement |
|---|---|---|
| `unified_tiling`, `auto_coarse_tiling` gates, default off; require `cpsat` + co-opt, warn and no-op otherwise | `config.py` | R8.1, R5.8 |
| Order the solve before the apply inside `plan_allocation`, so a `SolveError` still reaches `allocator.py:2211` over unmutated IR | `allocator.py` | R8.3 |
| `_commit_divisions` commits the chosen tiling alongside the chosen division | `allocator.py:1605` | R4.5 |
| Skip `_maybe_coarse_tile_span_overflow` on success; retain it verbatim as the fallback tiler | `passes.py:448` | R8.3 |
| Force `cut = 0` inside a hint scope | solver, driven from `op.dim_hints` | R4.3 |
| Hint pins: hinted op enters as a single-candidate buffer; `SPYRE_INDUCTOR_IGNORE_HINTS` drops pins | enumeration | R5.1–R5.8 |
| Record predicted sizes, lifetimes, views under H5; interstitial tick coordinates | solver + allocator | R7.2, R7.5 |
| Placement re-solve warm-started from residency intent; degrade-to-spill carries a distinct `residency_reason` | `allocator.py`, `plan_solver.py:68` | R7.3, R7.4 |
| Post-apply legality assertion — a hard failure, never a degrade-to-greedy | `passes.py` | §5 |
| Retire #3736's `unhinted` / `partial` markers | `test_solver_auto_coarse_tiling.py` | R5.3 |

**`hint_id` is no longer a problem here.** An earlier draft owed this work a
third reserved `hint_id` base above the user ids and
`_SPAN_OVERFLOW_HINT_ID = 10000` (`coarse_tile_span_overflow.py:45`), or else
`validate_coarse_tile_groups` (`coarse_tile.py:112`) would see one `hint_id` in
two groups and raise during apply — an illegal emission the model could not have
ruled out. The representation carries no `hint_id` at all now, and the apply
adapter mints from a base derived off the graph (*Architecture*), so there is
nothing left to collide.

**The R8.3 fallback needs an ordering guarantee, not a handler.** There is no new
pass slot needing its own `SolveError` handler: the solve is inside
`scratchpad_planning`, which `allocator.py:2211`'s existing try/except already
wraps. What that handler cannot assume any more is clean IR, if the pre-pass has
already mutated the graph. Two ways out — solve first and apply only on success
(keeping the pass a pure applier, which is what stage 1 builds), or make the
tiling pass reversible. The former is strongly preferred and costs nothing.

**Hints are taken as given, not re-validated.** A hint that tiles the stick axis
by a non-stick-multiple is legal at pass 430 and would be illegal at 455 (§1.3).
A hinted op therefore arrives already tiled and already stickified around that
tiling, so R5.1/R5.2's "honour the pin" cannot mean "re-check it with the
post-stickification predicates" — that would reject a plan which is in fact
correct.

R6.1/R6.2 need no code — R6.2 is accepted pessimism, guarded by a regression
check that restickify count does not grow, not by an assertion that it shrinks.

#### Tests

| Class | Covers | RFC test item |
|---|---|---|
| `TestUnifiedTilingGates` | R8.1, R5.8 — the on/off matrix; gate off reproduces today | 1, 2 |
| `TestUnifiedTilingHints` — extend #3736's suite | R5.1, R5.2, R5.5 land with its `hinted`/`partial` modes; add R5.4 `IGNORE_HINTS` drops pins, R5.6, R5.7 | — |
| `TestGroupRoundTrip` | R4.5 — groups round-trip; no `loop_group_id` collision | 4 |
| `TestUnifiedTilingFallback` | R8.3 — `INFEASIBLE` → span-overflow path, IR untouched; timeout → incumbent applied | 11 |

Models to add beyond #3736's three: softmax at dim −1, and the pointwise chains
from `TestCloneAtGraphBoundaries`.

**#3736's suite is the shape to extend, and not by adding axes to the existing
classes.** An earlier draft said to add `unified_tiling` and `auto_coarse_tiling`
to `parameter_axes` and mark the combos from `case_decorators`. That does not
work, for three reasons:

1. **`case_decorators` is not a shared hook.** The metaclass reads it with
   `attrs.get("case_decorators")` (`:308`) — the class's *own* dict, never a
   base's. Same for `parameter_models` and `parameter_axes` (`:296`, `:298`).
   There is exactly one definition in the tree today
   (`CoOptAllocatorIntegrationTests`, `:1025`), it is not inherited by anything,
   and every new suite has to define its own regardless. Only `run_case` is
   resolved on `self` and so genuinely inheritable.
2. **It would stack two expected-failure mechanisms.** The one class that has a
   `case_decorators` already returns `unittest.expectedFailure` for its `cpsat`
   combos (`:1024-1036`). Adding a tiling axis there yields
   `[unittest.expectedFailure, expected_unimplemented]` on the `cpsat × tiling`
   combos, and the outer marker absorbs the inner one's `XFailed` — precisely the
   "absorbs **any** exception" failure the marker exists to prevent, and which it
   already says it "cannot wrap".
3. **`parameter_axes` is a cross product.** Two boolean axes are a 4×
   multiplication of whatever class they are added to:
   `ParameterizedScratchpadUsage` goes 40 → 160 generated methods and
   `TestCloneAtGraphBoundaries` 80 → 320, all of them device compiles on
   suites labelled `device_critical`.

A new class per concern, each carrying its own axes and its own hook, costs one
four-line `case_decorators` apiece and has none of that. `hint_mode` is also the
better axis than a raw `auto_coarse_tiling` bool: it names the three contracts
(pins applied exactly / discovery from nothing / pins survive and discovery
fills in the rest) instead of leaving the reader to infer them from a flag.

Those `case_decorators` returns are then **never edited again**. Each combo stops
xfailing on its own, at the moment the last stub on its path is built, because
the decorator keys on the exception rather than on a hand-maintained list — and
the run turns red the day a combo passes, which is the signal to delete its
entry. This stage retires them by making them fire, not by editing them.

One follow-up on the PR itself: it patches `layout_solver` and
`allow_all_ops_in_lx_planning` but not `co_optimizing_lx_planning`, which R8.1
requires on for `UNIFIED_TILING` to do anything (its own `# TODO: Patch coarse
tiling config here` marks the spot).

## Stage 6 — Padding (conditional)

Opened only by R10.3's measurement, which is independent of every other stage and
should run from stage 1 onward: an unaligned-`K` matmul against a hand-pre-padded
one. If the measurement says the discretionary pad does not pay, R10.2 and R10.5
stay closed and this stage is only its first row.

| Work item | Where | Requirement |
|---|---|---|
| Derived legality pad reaches predicted sizing; lift `padding.py`'s fixed policies; shared-operand emission in `lower_pad_sequence` | `padding.py:73`, `:163`, `:183`; `pass_utils.py:1191` | R10.1, R10.4 |
| Discretionary pad as a candidate row | enumeration | R10.2\*, R10.5\* |

`*` = conditional on R10.3's measurement. Extend `test_padding.py` for R10.1
(derived pad reaches sizing) and R10.4 (two matmuls sharing an operand emit one
padded buffer).

## Stage 7 — The M2 objective collapse (deferred)

**Deferred, and substitutable.** Stage 2 lands the seam; this stage lands what
goes through it. R3.2 replaces `_run`'s two-phase lexicographic solve
(`ilp_solver_ortools.py:446-518` — minimize `sum(spill_cost)`, lock it with
`model.add(sum(hbm_terms) <= round(solver.ObjectiveValue()))` at `:479`, then
maximize `sum(cores)`) with one weighted `Minimize`. That changes **every
existing CP-SAT plan with tiling switched off entirely** — a blast radius with
nothing to do with tiling, which is exactly why it is off the critical path.

Order-independent with respect to stages 1–6: it can land before them, after
them, or in the middle. Expect it to be superseded in part — when Isuru's
`predict_ops` is ready it arrives through stage 2's `objective` parameter, and
§7.2/§7.3 below are the reference for taking it. What cannot be substituted is
the collapse itself (R3.2) and the grammar the parameter accepts (R3.3, R3.4),
since those are properties of this repo's solver rather than of any producer.

### 7.1 Build the quality gate first

**The suite that ought to catch this catches nothing today.**
`CoOptAllocatorIntegrationTests`' prescribed fingerprints
(`test_scratchpad_use.py:883-1013`) look like the right guard — exact-match
`{buf: (location, size, split)}` dicts over four models — but the class **does
not set `metaclass=_ParameterizedScratchpadMeta`**, deliberately (`:776-783`:
"no `test_*` methods are generated and nothing is collected — the
co-optimization compiles are too slow to run on every CI job"). Zero methods
collect. A gate phrased against it passes vacuously, and any "expect fingerprint
churn, re-baseline one model at a time" instruction describes work that will
never come due.

Two further facts make reviving it the wrong fix on its own. Its prescribed
dicts are the **greedy** StrategyB plans (`:787-791`), and its `cpsat` combos are
marked `unittest.expectedFailure` (`:1024-1036`) precisely because CP-SAT lands
somewhere else — so *today's CP-SAT plans are recorded nowhere in the tree*, and
there is nothing to diff the collapse against. And exact-match dicts are a
strictly stronger claim than R3.5 makes: R3.5 asks for spill-parity and no core
regression, and explicitly disclaims bit-identity. Pinning addresses and per-axis
splits would fail on changes R3.5 permits.

So the gate is built, not revived, in three commits that precede the collapse:

1. **Lift `_allocation_fingerprint`** (`:803`) from
   `CoOptAllocatorIntegrationTests` to `BaseTestScratchpadUsage` (`:73`). It
   already returns exactly the two projections R3.5 needs — `location` is
   `"LX"`/`"HBM"`, so the spill set falls out of it, and `split` is the committed
   division, so the core count is `math.prod` of its factors — and it needs no
   change beyond being reachable from a suite that collects.
2. **Capture the baseline.** Run the four models through CP-SAT on today's
   two-phase solve and record `{buf: (location, core_count)}` per model as a
   checked-in fixture. This is the artefact that does not exist yet, and it has
   to be taken *before* the collapse or the comparison has no left-hand side.
3. **Assert parity, not identity.** `TestUnifiedTilingParity` compares against
   that fixture on the two R3.5 projections only: no buffer resident in the
   baseline may spill, and at equal spill no op's core count may drop. Sizes and
   per-axis split shapes are free to move.

Reviving `CoOptAllocatorIntegrationTests` is still worth doing — it is the only
place the desired plans are written down — but it is its own task, gated on
`cpsat` becoming the default `layout_solver` as its docstring says, and it is not
what proves R3.5.

### 7.2 How the cost function is injected

`isuruf/torch-spyre@cost` is a working end-to-end injection and is the reference
for this stage. It threads a cost function through four layers:

| Layer | On the reference branch | Adopt as |
|---|---|---|
| Producer | `_inductor/cost_model.py::predict_ops(op_features) -> sympy.Expr`, fed per op by `allocator._extract_op_features` (`:2068` there) | **not this phase** — see *Producer* below |
| Symbol namespace | `sym_is_lx` / `sym_inv_cores` / `sym_core_divs` properties on `CoreDivisionBuffer`, minting `sympy.Symbol(f"is_lx_{self.name}")` | adopt; stage 5 adds the tiling symbols |
| Transport | `plan_layout_and_core_divisions(buffers, cost_expr)` → `_plan_layout_generic` → `_run` | adopt, renamed to R3.1's keyword-only `objective` |
| Lowering | sympy rewrite → `lambdify(syms, expr, modules=[{"min", "max"}, "math"])` applied to the CP-SAT vars → one `model.minimize` | adopt, with a validator in front |

Three of its decisions are worth taking as-is.

1. **Symbols live on the buffer, not in a table passed alongside it.** Producer
   and solver agree through the buffer name alone, so nothing has to carry a
   `symbol -> var` map across the allocator/solver boundary and the solver's
   binding step is a five-line loop over `tensors.values()`. It also keeps the
   namespace open: stage 5's tiling symbols are new properties, not a new
   parameter.
2. **`lambdify` replaces a hand-written lowerer.** The tree walk this plan
   previously specified (`lower(expr, bindings) -> cp_model.LinearExpr`) is what
   `lambdify` already does — it prints the expression to Python source and
   evaluates it against the CP-SAT vars, so `Add` and constant `Mul` lower
   through the vars' own operator overloads and only the nodes with no operator
   form (`Min`, `Max`) need entries in the custom module. `cost_expr.py` stays,
   owning validation and scaling rather than tree-walking.
3. **Non-linearity in the division axis is a lookup table, not a constraint.**
   The cost model wants time, which goes as `1/cores`; a reciprocal of a decision
   variable is not expressible, so the branch precomputes
   `[32 // cd.cores_used for cd in b.core_divisions]` and ties it to the division
   index with one `AddElement` — the same shape `eff_size` already uses. Any
   per-candidate scalar, however non-linear in the choice, therefore costs
   exactly what R3.7 budgets. This is the pattern stage 5's `tile_count` and
   `full_size` follow.

The branch's `if cost_expr is not None: ... else: <two-phase>` split is
transitional and does **not** come across: R3.2 replaces the two-phase block
rather than sitting beside it, and `TestSinglePhaseObjective` pins one
`Minimize` and one `Solve` on the default path too.

### 7.3 What must change on adoption

Nine items, found by reading the branch against this tree. Items 1–5 and 7 are
defects on the branch as it stands, not porting friction; 6, 8 and 9 are gaps to
close on the way in.

| # | On the branch | Consequence | Fix |
|---|---|---|---|
| 1 | `unnest_min` appends nothing for an arg that is neither `c*Min(...)` nor `Add` | It is silently dropped. Verified: `Min(4, 5*Min(x, y))` — the function's own docstring example — rewrites to `Min(5x, 5y)`, losing the constant bound. Wrong cost, no error | add the missing `else: result.append(arg)`; pin with a rewrite test asserting the flattened arg set, not just the node count |
| 2 | `sym_map` binds `t.buffer.sym_inv_cores -> t.inv_cores` for **every** wrapper | `_LifetimeBufferWithCpVars` has no `inv_cores` and plain `LifetimeBoundBuffer` has no `sym_inv_cores` (it is a `CoreDivisionBuffer` property) → `AttributeError` on any `plan_layout` solve | bind through a `bind_symbols()` wrapper hook. Note it must **not** be empty on the placement-only wrapper: that wrapper has `in_buffer` (`:185`), so it binds `is_lx` and omits only `inv_cores`, or `validate`'s containment check rejects any residency-mentioning objective on a `plan_layout` solve |
| 3 | `core_terms` is read by the DEBUG log block but bound only in the `else` branch | `UnboundLocalError` whenever an injected objective runs under DEBUG logging | branch the log line with the objective |
| 4 | the joint wrapper renames `cores` → `inv_cores` and never sets `self.cores` | `core_terms` is then always empty, so the legacy path's phase 2 silently stops running and every no-objective plan loses its parallelism phase | moot once R3.2 lands; until then keep both attributes |
| 5 | two ad-hoc float scales: `truncate_floats_min` (×10⁴, then `/m` back out) and `my_min` (×10³, returning `min_var * 1e-3`) | Verified: CP-SAT accepts float coefficients in `minimize` but **rejects** them in `AddMinEquality` (`ValueError: Failed to convert integer linear expression`). A float-scaled `Min` nested in another `Min`/`Max` raises | one documented `COST_SCALE` applied once to the whole expression (R3.4); never a per-node scale |
| 6 | `modules=[custom, "math"]` — an unsupported node falls through to `math` or to the vars' own operators | the failure surfaces from inside ortools naming neither the sympy node nor the buffer. Verified: `sqrt`/`log` → `TypeError: must be real number`; `x*y` → `TypeError: __mul__(): incompatible function arguments`; `x**2`, `x/y`, `Piecewise` → `NotImplementedError` from `cp_model` | validate before lowering (R3.3), so every rejection carries its own message |
| 7 | nothing checks `cost_expr.free_symbols` against what the solver binds | an unbound symbol reaches `lambdify` as a free variable → `NameError` at call time. Live today: `sym_core_divs` is minted and written to `op.op_it_space_splits` but never bound | assert containment in `validate`; raise `CostExpressionError` naming the unbound symbols in sorted order (`free_symbols` is a `set` — R8.4) |
| 8 | `32 // cd.cores_used` | hardcodes 32 cores rather than reading `SENCORES`, and floor division is lossy: `32 // 7 = 4` against a true 4.57, a 12% error that can mis-rank two divisions | scale the reciprocal table (`SCALE // cores_used`, `SCALE` large) and take the core count from `get_ncores` |
| 9 | three `print()` calls, a commented-out `NewIntVar`, a function-local `import math` | debug residue | delete |

**Producer, and what this stage owes it.** `predict_ops` and its
`dump_cost_model` feature extractors live only on the reference branch — neither
exists in this tree. That is not a blocker: R3.5 has `objective=None` select a
default objective built from today's terms, so M2 lands, is testable, and is
useful with no producer at all, and `TestCostExprLowering` authors its
expressions directly, which is what keeps it device-free. Porting the analytic
cost model is separate work. What this stage owes it is a surface it can bind to
unchanged — which is why the namespace is extensible (`_extract_op_features`
wants symbols for splits as well as for residency and cores) rather than a
closed set of scalars.

**Expression build cost, measured.** `lambdify` is not the bottleneck (0.77 s to
lower a 3000-symbol sum). Building the sympy expression is: accumulating terms
with `sum()` or `+=` is superlinear — 0.34 s at 500 terms, **37.7 s at 1500** —
against 0.09 s for `sympy.Add(*terms)` at 1500. Any producer, and the default
objective here, must assemble with `Add(*terms)`. Worth an explicit line in
`cost_expr.py`, since the natural way to write it is the slow way.

**Needs folding back into R3.7.** R3.7 bounds binding at one `AddElement` per
symbol and says nothing about the aux vars `Min`/`Max` reification creates — one
`IntVar` plus one `AddMinEquality`/`AddMaxEquality` per surviving node, which is
exactly what `unnest_min`'s flattening exists to keep down. The bound wanted is
"one `AddElement` per symbol **and** one reified var per `Min`/`Max` node
surviving flattening"; `TestCostExprLowering` pins the second half.

### 7.4 Work items

1. `scratchpad/cost_expr.py`: `CostSpec`; `CostExpressionError`;
   `validate(expr, bound)` — the R3.3 accept/reject walk plus item 7's
   containment check, raising with the offending node named; `COST_SCALE`
   derivation and the int64 overflow check (R3.4); `lower(expr, bindings)` =
   validate, flatten nested `Min`/`Max`, scale once, then `lambdify` against the
   `{"min", "max"}` module. Model-level symbols with no buffer to hang from stay
   in this module's namespace — `peak_lx_bytes` is one `AddMaxEquality` over the
   existing `top[b]` vars (`_add_no_overlap_2d`, `:568`), never a time-indexed
   sum (R3.7).
2. `plan_solver.py`: `sym_is_lx` on **`LifetimeBoundBuffer`**, not on
   `CoreDivisionBuffer` — residency belongs to every buffer, as
   `residency_reason` (`:68`) already does, and the placement-only wrapper has
   `in_buffer` to bind it to. Only `sym_inv_cores` / `sym_core_divs` are
   division-specific and belong on `CoreDivisionBuffer` (`:137`).
3. `ilp_solver_ortools.py`: `bind_symbols()` on both wrappers (item 2); the
   scaled reciprocal-cores table (item 8); and replace `_run`'s two-phase block
   (`:467-492`) with a single `Minimize`. The default objective is
   `SumOverBuffers(spill_cost * (1 - in_buffer)) - W * SumOverBuffers(cores)` —
   note the `(1 - in_buffer)` factor, without which `sum(spill_cost)` is a
   *constant* and minimizing it is a no-op. `W` must be derived, not guessed:
   the core term's range is bounded by `Σ max cores` (≤ 32·N), so the spill
   weight has to exceed that over the smallest spill increment, and that
   derivation interacts with R3.4's int64 overflow check rather than sitting
   beside it.
4. `_run` must `Solve` **exactly once**, including when the objective is empty
   or constant. Today it can solve *zero* times — with `hbm_terms` and
   `core_terms` both empty it falls through to `_extract` on an unsolved solver,
   and the DEBUG block reads `solver.ObjectiveValue()` regardless (`:468-504`).
   Phase 2 is currently skipped by testing `sb.cores is None`, which only the
   placement-only wrapper sets (`:193`); collapsing to one phase removes that
   flag's meaning, so the placement-only path needs an explicit skip in its
   place.

Adopt the injection mechanism only. The branch also carries an unrelated
divergence in this file — its `_add_no_overlap_2d` shortens the **parent's**
lifetime on an in-place merge where this tree shortens the **child's**, and its
`_justify` drops the capacity check — and it subclasses a
`LifetimeBoundBufferWithSolverVars` base that does not exist here. None of that
is part of the cost function; leave this tree's versions alone.

### 7.5 Tests

Device-free, in `test_scratchpad_solver.py` — no metaclass there, so markers are
applied per method.

| Class | Covers | RFC test item |
|---|---|---|
| `TestCostExprLowering` | R3.3 accept/reject table (one `CostExpressionError` case per rejected construct, raised by `validate` **before** any ortools error can surface), unbound-symbol containment, `Min`/`Max` flattening preserves the arg set, R3.4 one scale for the whole expression + overflow, R3.7 binding bounds, `SumOverEdges`/`relayout_bytes` **absent** | 3 |
| `TestSinglePhaseObjective` | R3.2 — one `Minimize`, no lock inequality, no second `Solve`, **and exactly one `Solve` when the objective is empty** | 3 |
| `TestUnifiedTilingParity` | R3.5 — spill-parity and no core regression against the §2.1 baseline fixture | 2 |

R8.5 needs no new work: it is covered by the existing
`TestCpSatAllocatorFallback`.

**Gate.** `TestUnifiedTilingParity` passing against the baseline fixture on all
four models: no baseline-resident buffer spills, no core-count drop at equal
spill. Sizes and split shapes may move; a *spill* moving is the stop signal, and
it is diagnosed one model at a time against the fixture, not re-baselined away.

This is the one place the plan gates on comparing output to earlier output, and
it is a **quality** gate, not a validity one (*Staging*: correctness comes from
the model). A collapsed objective can be sound and still price worse than the
lexicographic one it replaces; nothing in the model rules that out, so the only
way to see it is to look.

## Requirement traceability

Legend — **Stage**: 1–7. **Level**: E2E (`test_scratchpad_use.py`,
`test_padding.py`, `test_coarse_tile_e2e.py`, `test_solver_auto_coarse_tiling.py`),
SOL (`test_scratchpad_solver.py`), ENU (`test_enumerate_tilings.py`), BENCH
(measurement, not a test).

| Req | Stage | Level | Pinned by |
|---|---|---|---|
| R1.1 | 3 | ENU | brute-force reference equality on small shapes |
| R1.2 | 3 | ENU | tiered deterministic order; truncation drops from tail; mandatory keeps never dropped; `_combo_cost` absent from the path |
| R1.3 | 3 | ENU | untiled option present in every returned set |
| R1.4 | 3 | ENU | patch each reused predicate, assert called |
| R1.5 | 3 | ENU | extents equal `_planned_tile_extents_per_level` output |
| R1.6 | 3 | E2E | apply each option **and** compare to CPU |
| R1.7 | 3 | ENU | cap defaults; assert not migrated to `config.py` |
| R1.8 | 3 | ENU+E2E | no nested/multi-level option, on the eight known-wrong shapes |
| R1.9 | 3 | ENU | no-span-pressure op still yields > 1 option |
| R2.1 | 1 | SOL | `TileAxis`/`TileSpec`/`CoreDivision.tiling` fields and derived scalars; `TileSpec` equality is the shared-shape test; **negative**: no candidate list parallel to `core_divisions` |
| R2.2 | 5 | SOL | division enumeration runs once per tiling option; **negative**: a candidate legal under one tiling is absent under another, in both directions |
| R2.3 | 5 | SOL+E2E | joint span check; over-tiling fix picks a smaller tile count |
| R2.4 | 5 | SOL | seed pair always retained; signature dedup keyed on `(splits, tiling)` together; no cap |
| R2.5 | 5 | SOL | empty candidate set raises `Unsupported` at today's point |
| R2.6 | 4 | SOL+E2E | `prep_cache` key includes tile; negative cache-sharing test; predicted view == recomputed post-`coarse_tile` |
| R3.1 | 2 | SOL | `objective` keyword-only on both ABCs and all four `plan_layout` overrides; accepts a `sympy.Expr`; CP-SAT raises on a non-`None` value until stage 7 |
| R3.2 | 7 | SOL | one `Minimize`; no lock inequality; no second `Solve`; exactly one `Solve` on an empty objective; no cost-expr/legacy fork |
| R3.3 | 7 | SOL | accept case per node; `CostExpressionError` per rejected construct, raised by `validate` before lowering and naming the node; unbound symbols rejected |
| R3.4 | 7 | SOL | one `COST_SCALE` for the whole expression; overflow raises; no float coefficient reaches `AddMinEquality` |
| R3.5 | 7 | E2E | baseline fixture captured before the collapse; parity asserted on spill set + core count, not exact fingerprints. **Quality gate, not a validity one** |
| R3.6 | 2 | SOL | the four registry placement solvers ignore objective and warn once, from three files; `cpsat` honours it |
| R3.7 | 7 | SOL | ≤ 1 `AddElement` per symbol **and** ≤ 1 reified var per `Min`/`Max` surviving flattening; model size linear; `SumOverEdges`/`relayout_bytes` absent |
| R4.1 | 5 | SOL | triple table total; `cut` determined not merely constrained |
| R4.2 | 5 | SOL | untileable boundary pinned; every cut-free run is a contiguous slice |
| R4.3 | 5 | SOL+E2E | hint scope never split |
| R4.4 | 5 | E2E | op order unchanged across the solve |
| R4.5 | 1/5 | E2E | groups round-trip through `TileSpec`; `group_idx_offset` derived, so no `loop_group_id` collision; no `hint_id` to collide |
| R4.6 | 5 | SOL+E2E | both boundaries pinned; singleton group; hint- and solver-chosen |
| R4.7 | 5 | SOL | directional admission; unverifiable pair fails closed; claim orientation |
| R4.8 | 5 | SOL | no `boundary_op ⟹ ¬in_buffer` constraint; row-3 eviction only |
| R4.9 | 5 | SOL | cut-free reserves no read copy; cut creates one |
| R4.10 | 5 | SOL+E2E | `full_size`/`boundary_view` equal realized values |
| R5.1 | 5 | E2E | hinted op enters as a single-candidate buffer (#3736 `hint_mode=hinted`: every pin applied, no level invented) |
| R5.2 | 5 | E2E | solver never re-tiles or un-tiles a hinted op (#3736 `hint_mode=partial`: pins survive, discovery fills the rest) |
| R5.3 | 5 | E2E | unhinted ops are tiled automatically (#3736's `unhinted` marker retires) |
| R5.4 | 5 | E2E | `SPYRE_INDUCTOR_IGNORE_HINTS=1` drops pins |
| R5.5 | 5 | E2E | **negative**: hint group is not grown with solver neighbours |
| R5.6 | 5 | E2E | reduction hint applies; `enable_reduction_tiling=0` raises `Unsupported` |
| R5.7 | 5 | SOL+E2E | H3 levels 1/2/3 name the offending key |
| R5.8 | 5 | E2E | `AUTO_COARSE_TILING` off ⇒ every unhinted op untiled |
| R6.1 | 5 | E2E | restickifies still inserted, unchanged |
| R6.2 | 5 | E2E | **guard only**: restickify count does not regress |
| R6.3 | 4 | SOL | views land (== R2.6); relayout symbols absent (== R3.7) |
| R7.1 | 4 | SOL | predictor mutates no IR (deep-compare before/after); needed to *price* an unapplied candidate, not to apply one |
| R7.2 | 5 | E2E | records exist and are readable; **diagnostic, not a gate** (*Staging*) |
| R7.3 | 1/5 | E2E | solve → apply → rebuild buffers → place, all inside `plan_allocation`; op count unchanged on the `SolveError` path |
| R7.4 | 5 | E2E | re-solve warm-started; degrade carries a distinct reason |
| R7.5 | 5 | E2E | predicted vs realized lifetimes under rank-order normalization, realized side measured off the mutated graph; **diagnostic, not a gate** |
| R8.1 | 5 | E2E | gates default off; warn and no-op without `cpsat` + co-opt |
| R8.2 | 5 | SOL | `AddHint` seeded from the heuristic plan |
| R8.3 | 5 | E2E | `INFEASIBLE` → span-overflow path, IR untouched; timeout → incumbent applied |
| R8.4 | 5 | E2E | identical plans across two runs |
| R8.5 | 2 | E2E | covered by existing `TestCpSatAllocatorFallback` |
| R10.1 | 6 | E2E | predicted sizes from padded `device_size` |
| R10.2 | 6\* | SOL | pad row unlocks an otherwise-blocked split |
| R10.3 | 6 | BENCH | unaligned-`K` matmul vs hand-pre-padded |
| R10.4 | 6 | E2E | two matmuls sharing an operand emit one padded buffer |
| R10.5 | 6\* | E2E | pad pin lowers to a pin; empty set named at H3 level 2 |
| R10.6 | — | — | **absence**: issue #1756 restriction untouched (Phase 3) |

`6*` = conditional on R10.3's measurement (see also §3.0).

## Highest risks, in order

1. **R2.6 tiling-aware per-core views (stage 4).** A wrong view grants residency
   on a slicing agreement that does not hold — wrong data, and outside R7.4's
   degrade-to-spill safety. The negative cache-sharing test is the cheapest
   guard, and it is the first commit of that stage's PR.
2. **Nothing in the model bounds tile depth (stage 2's deferral).** The retained
   two-phase objective is lexicographically absolute on spill, so a deeper nest is
   never worse by its measure and it will tile arbitrarily to avoid one spilled
   buffer. The only bounds are outside the model — the enumerator's cap (R1.7) and
   R1.2's truncation, both stage 3 — which this deferral promotes from
   conveniences to load-bearing. A cost model through stage 2's seam is what
   actually fixes it.
3. **R4.6 in stage 5 (§5.2).** Omitting the hint-driven pin produces plans
   `coarse_tile` will reject at apply — an illegal emission, not a fallback.
4. **Name-keyed symbol binding (stage 7).** Producer and solver agree only
   through the buffer name, and every failure mode of that agreement is quiet: an
   unbound symbol is a `NameError` from generated code, a dropped `Min` arg is a
   wrong cost with no error at all (both live on the reference branch today).
   Nothing downstream can detect it — the plan is legal, just worse. `validate`'s
   containment check and the flattening test are the only guards.
5. **Mutating the graph inside the allocator (§5.4).** Applying tiling as a
   pre-pass puts IR mutation inside `plan_allocation`, which every allocator
   shares and which `allocator.py:2211` retries on `SolveError`. If the mutation
   ever precedes the solve, that retry restarts over a graph the previous attempt
   already tiled — and the symptom is a double-applied plan, not an exception.
   Ordering the solve first makes it structurally impossible; the guard is an
   assertion that the fallback path sees the op count it started with.
6. **Enumeration cost (§5.1).** `enumerate_work_division_candidates` runs per
   tiling option (R2.2) and `_views_for_divs`' sympy prep is no longer
   candidate-invariant (R2.6). Both were built assuming they are paid once per
   op.
7. **The objective collapse (stage 7).** Changes every existing plan with tiling
   off. Off the critical path now, and gated on a baseline that has to be captured
   before the change — but the baseline does not exist yet, so the capture has to
   precede the collapse whenever it is scheduled.

## Plan corrections — this revision

Four defects in the plan itself, found by re-grounding it against the tree, against
PR #3736, and against the RFC's own `PartitionConfig` pairing. None is an RFC
defect — in each case the RFC was right and the plan had drifted from it.

| Where | What was wrong | Now |
|---|---|---|
| Stage 7's gate | Rested on `CoOptAllocatorIntegrationTests` catching a spill-parity regression. That class deliberately has no metaclass and collects **zero** tests (`test_scratchpad_use.py:776-783`); its dicts are the greedy plans and its `cpsat` combos are `expectedFailure`, so today's CP-SAT plans are recorded nowhere. The gate was unsatisfiable-by-inspection and vacuous in practice. | §7.1 builds the gate instead: lift `_allocation_fingerprint`, capture a CP-SAT baseline, assert R3.5's two projections (spill set, core count) against it. |
| The old §4 | Named the new pass slot but nothing downstream of it. `span_reduction` **deletes** a committed division (`work_division.py:935-944`), `work_distribution` re-derives it, and 455 re-solves it jointly under the co-opt gate R8.1 requires — so the solve's output would have been discarded three times. | Diagnosis stands and is the reason the placement moved; the five rewiring items it produced are **withdrawn**, because deciding inside 455 is downstream of all three passes. See *A departure from the RFC* below. |
| The old §0.3 | The R3.6 row read "the four solvers above", which swept `ilp_solver_ortools.py` into the ignore-and-warn set — the one solver R3.6 exempts. | Row narrowed to the three placement-only files, with the four/three/five counts spelled out. |
| *Architecture*, stage 1 | The first draft of the buffer-carried design gave `TiledBuffer` a `tilings` list **beside** the inherited `core_divisions`, implying a free cross product. Divisions are tiling-relative twice over — the legal set moves in both directions, and `splits_by_index_coeff`'s coefficient keys are rewritten by `_divide_ranges`, so a division carried across tilings is uninterpretable, not merely illegal. Here the RFC was right: this is what its `PartitionConfig` pairing existed for, and the redesign dropped it. | One candidate list whose element pairs both: `CoreDivision.tiling: TileSpec = TileSpec()`. `TiledBuffer` and `tiling_parent_matches` dissolve, the two-level solver index is withdrawn, and `min_footprint` gains the tile factor. |

Four further items were open at the last revision and are now folded into the
stages that own them: the default objective's missing `(1 - in_buffer)` factor
and the undefined dominance weight (§7.4), `sym_is_lx` sitting on the wrong class
and the placement-only `bind_symbols` being empty (§7.3 item 2, §7.4), `_run`'s
zero-`Solve` path (§7.4), `boundary_role` missing the reduction path (§4.1), and
the group-scoped carry rejection (§5.3).

### A departure from the RFC — pipeline placement

Unlike the four above, this is not the plan drifting from the RFC. It is a
deliberate change of design, and the RFC needs updating to match rather than the
plan being corrected back.

The RFC places the solve in its own pass between `passes.py:443` and `:448`, has
it apply `coarse_tile` at the 448 slot, and leaves 451-452 and 455 to commit
what it decided (`draft-unified-tiling-cpsat.md` §5, diagram `:952-968`). This
plan instead carries the tiling as candidate data on a `LifetimeBoundBuffer`
subclass and applies it inside the scratchpad planning pass, with no new slot.
Three things fall out, argued in full under *Architecture*:

- the downstream re-decide the row above diagnosed cannot occur, because the
  decision is taken after every pass that would have overwritten it;
- prediction is no longer on the apply path, since the buffer set is rebuilt
  from the mutated graph — it is still needed to *price* candidates, so R7.1
  stays a stage-4 deliverable;
- ops that tiling inserts are divided by the joint solver without new work,
  because `_division_map` runs after the pre-pass.

Sections needing the corresponding RFC edit: §5's pass diagram, R4.5's `hint_id`
namespace (no third reserved base), R7.1's framing of what prediction is for,
and R8.3's fallback (no new-slot handler; instead an ordering guarantee that the
solve precedes any mutation). Not yet applied to the RFC.

**Incorporating PR #3736.** The PR builds the test mechanism and the
hint-preservation half of stage 7. Adopting it changed the plan in three places:
`expected_unimplemented` becomes a *promotion* to `utils_inductor.py` rather than
a fresh write; stage 7's suite is a sibling with its own `parameter_axes`, with
the three reasons the earlier "add axes to the existing classes" sketch could not
have worked (`attrs.get` is own-dict only, `unittest.expectedFailure` stacking,
4× cross-product growth on `device_critical` suites); and the staging section now
asks for the PR's CI result before any stage relies on xfail satisfying
`mandatory_success`.

## RFC corrections — applied

Found while grounding this plan against the tree, and folded back into
`draft-unified-tiling-cpsat.md`.

| Location | Correction |
|---|---|
| Testing item 1 | Claimed `test_coarse_tiling.py` has no CI config yaml. It does — as do all five named suites, each `unlisted_test_mode: mandatory_success`. Replaced with what that mode implies for landing an xfail. |
| R1.4, R1.9 | `_host_dim_has_legal_nontrivial_split` cited at `:935`; it is at `:936`. |
| R1.4 | Reuse list omitted `_remaining_span_candidates_after_tile` (`:1236`) — the span-*sufficiency* check both public entry points compose (`:1344`, `:1471`), and the one R2.3's joint span feasibility should extend rather than restate. Added with that rationale. |
| R4.5 | Handled `loop_group_id` collision via `group_idx_offset` but not `hint_id`. Added the second namespace: span-overflow mints from `_SPAN_OVERFLOW_HINT_ID = 10000`, so a third producer needs its own reserved base or `validate_coarse_tile_groups` raises during apply — an illegal emission, not something the model could rule out. **Since superseded**: the representation carries no `hint_id`, and the apply adapter mints from a derived base, so the third namespace is unnecessary (*Architecture*). The correction stands as a description of the hazard; the reserved-base remedy does not. |

**One pending fold-back, not yet applied.** R3.7 bounds symbol binding at one
`AddElement` per symbol but is silent on the aux vars `Min`/`Max` reification
creates (§7.3). The plan carries the tighter bound and pins it; R3.7 should gain
the second clause.

**Two earlier flags retracted.** Both came from a misreading of the RFC, not
from the RFC:

- §1 and R1.4 cite `op_out_coords` at `pass_utils.py:363` and describe
  `host_dim` as the frame it indexes. That is exactly right — `op_out_coords`
  is at `:363`, and `host_dim` is a positional index into its return. No change.
- *Background* line 163 writes `{id(op): CoarseTileInfo}` without claiming where
  `CoarseTileInfo` is defined (it is `loop_info.py:33`, imported at
  `coarse_tile.py:93`). Nothing to fix.

**Not a correction — a known forward reference.** The parent link
`draft-compiler-optimization.md` does not resolve on this branch; the roadmap
lives on `optimization-roadmap-draft`, and commit `048e558` removed the local
`draft-compiler-optimization-roadmap.md` copy. It resolves when the branches
converge. Left as-is deliberately.

**Verified sound, not a correction.** The *Background* claim that interior
per-tile scratch stays LX-eligible is stated verbatim in `_is_tiled_advancing`'s
docstring (`scratchpad/utils.py:218-234`). The motivation holds as written;
§1.1 confirms it end-to-end anyway.
