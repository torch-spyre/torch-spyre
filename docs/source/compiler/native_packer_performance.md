# Native packer performance

**Measured 2026-08-12.** These numbers describe the C++
`NativePermutationLayoutSolver` as it stood when it became the default packer for
the simulated-annealing layout solver. They are a snapshot, not a contract: re-run
the harness rather than trusting them after the packer, the annealing driver, or
the cost model changes.

Harness:
[`docs/source/user_guide/examples/scratchpad/profile_native_packer.py`](../user_guide/examples/index.md).
Every invocation below prints the tables it produced and can emit the raw
per-instance timings with `--json`.

## What is being compared

The permutation packer that the annealing search drives has two interchangeable
implementations — the canonical Python `PermutationBasedLayoutSolver` and the C++
accelerator — selected by `config.native_layout_packer` /
`TORCH_SPYRE_NATIVE_PACKER`. They are behaviourally identical, asserted
bit-for-bit by the differential and SA-equivalence suites. See
[Simulated annealing layout planner](simulated_annealing_layout.md).

That identity is what makes the comparison clean: layout quality is not a random
variable across the two arms, so **time is the only thing under test**.

## Method

- Both arms solve **the same instance**, and the ratio is formed per instance.
  Dividing summed per-size totals — the obvious shortcut — makes dispersion
  unrecoverable.
- Timed region is solver construction plus `plan_layout()` (solve and finalize).
  `deepcopy` of the buffer list is deliberately outside it: at ~0.6–1% of a native
  solve it is small, but it is an additive constant present in both arms, so
  including it biases every ratio toward 1.
- Arm order alternates per repeat, so CPU frequency or thermal drift cannot
  systematically favour whichever arm runs first.
- Repeats are reduced by **min** — timing noise is one-sided, so the minimum is
  the better estimate of true cost. The median-of-repeats is computed too and
  agrees.
- Wall (`perf_counter`) and CPU (`process_time`) time are both recorded. In the
  runs below they agree to three decimal places, which is the evidence that
  nothing else on the machine perturbed them.
- An `isinstance` check runs per solve, so a mis-selected packer cannot silently
  turn this into a native-versus-native measurement.
- Instances whose annealing search takes fewer than `--min-steps` steps are
  reported separately. If the initial layout is already optimal, `solve()` returns
  immediately and the timing reflects construction, not the packer.
- Per size the harness reports median paired ratio, IQR, percentile bootstrap 95%
  confidence interval (20k resamples), win count, and a one-sided sign test.

Instances come from the `_random_buffers` generator in
`tests/inductor/test_perm_layout_solver.py` — the same one the differential and
SA-equivalence suites drive, so the benchmark and correctness workloads share a
distribution. `horizon=12` (heavy lifetime overlap), `max_size=200`, 25% in-place
child probability, default reheating schedule.

The step budget is a deterministic function of size: 500 steps for n≤16, 960 at
n=32, 1920 at n=64, 3840 at n=128, and 5000 (capped) at n=256.

Environment for the numbers below: single x86_64 dev box (128 cores), Python
3.12.13, `_C` built `-O3 -DNDEBUG`, BLAS/OMP threads pinned to 1,
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`. No accelerator is involved — layout planning is
single-threaded CPU work.

## Capacity pressure is the dominant variable

Scratchpad capacity matters more than problem size, so quote the rule with any
number. `--cap-rule` selects it:

| rule | capacity | character |
|---|---|---|
| `3xmax` | 3 × largest buffer | does not scale with footprint: pressure climbs with n, and at n=128 only ~25 of 128 buffers place at all |
| `foot4` | footprint // 4 | tight, but pressure holds constant as n grows |
| `foot2` | footprint // 2 | representative; most buffers place |

## Results

Median paired speedup (Python ÷ native) with bootstrap 95% CI, 15 seeds per cell.
`placed` is how many buffers were allocated; the rest are evicted.

| n | `3xmax` | placed | `foot4` | placed | `foot2` | placed |
|--:|--:|--:|--:|--:|--:|--:|
| 8 | 10.56× [3.97, 11.92] | 6/8 | 7.39× [6.00, 8.06] | 2/8 | 9.19× [8.14, 10.39] | 4/8 |
| 16 | 11.08× [10.23, 11.86] | 10/16 | 9.86× [9.57, 10.10] | 6/16 | 12.04× [11.50, 13.39] | 11/16 |
| 32 | 10.70× [9.81, 11.35] | 15/32 | 11.14× [10.52, 11.98] | 14/32 | 14.60× [13.69, 15.31] | 25/32 |
| 64 | 10.68× [10.27, 11.16] | 21/64 | 14.41× [12.83, 14.54] | 40/64 | 16.32× [15.76, 17.48] | 58/64 |
| 128 | 10.52× [10.35, 11.63] | 25/128 | 13.94× [13.09, 14.19] | 74/128 | 14.31× [13.61, 15.02] | 108/128 |
| 256 | 7.59× [6.05, 10.71]¹ | 34/256 | — | — | 5.79× [4.83, 7.15]¹ | 214/256 |

¹ 5 and 3 seeds respectively, one repeat — indicative only.

Pooled over n=8–128:

| rule | median | 95% CI | paired wins | sign test |
|---|--:|--:|--:|--:|
| `3xmax` | **10.62×** | [10.49, 11.08] | 75/75 | p = 3.1e-5 |
| `foot4` | **11.34×** | [10.52, 12.44] | 75/75 | p = 3.1e-5 |
| `foot2` | **13.93×** | [13.50, 14.50] | 74/75 | p = 6.1e-5 |

Absolute scale, `foot2`: at n=128 a solve takes 1.64 s native against 23.0 s
Python; at n=32, 49 ms against 713 ms.

Across all 233 instances and 1066 timed solves, finalized addresses and
`quality()` were identical between the two arms — zero mismatches.

## Reading the numbers

**The win is a large constant factor, and it erodes as n grows.** It is not an
asymptotic improvement. Per-step cost under `foot2`:

| n | native µs/step | Python µs/step | ratio | growth vs previous n |
|--:|--:|--:|--:|---|
| 8 | 15.9 | 146.5 | 9.19× | — |
| 16 | 27.0 | 334.7 | 12.04× | native ×1.70, Python ×2.28 |
| 32 | 51.1 | 742.7 | 14.60× | native ×1.89, Python ×2.22 |
| 64 | 113.7 | 1902.7 | 16.32× | native ×2.23, Python ×2.56 |
| 128 | 428.0 | 6001.6 | 14.31× | native ×3.76, Python ×3.15 |
| 256 | 2950.6 | 17083.0 | 5.79× | native ×6.89, Python ×2.85 |

Past n≈128 the native per-step cost grows faster than the Python one, which
follows from the design. The native packer recomputes placement from scratch in
permutation order after every operation; the Python packer is genuinely
incremental, splicing contact profiles. The port buys a very large constant factor
by deleting per-operation interpreter and pybind overhead, at the cost of the
incremental algorithm, so its advantage narrows as the recompute grows. Earlier
profiling put the crossover with a truly incremental C++ packer near n≈640.

Two things bound how much that matters. Captured real graphs run n≈5–80, which is
the 12–16× region under representative capacity. And the large-n cells are a
stress probe rather than a prediction: `horizon` stays at 12 as n grows, so at
n=256 nearly every buffer overlaps nearly every other, far denser than a real
schedule.

**Small sizes have degenerate cells.** Under `3xmax`, five of fifteen n=8
instances have an already-optimal initial layout, run 0–20 annealing steps, and
therefore time construction rather than the packer. They are the source of the
1.30–5.63× tail. `--min-steps` excludes them; the tables above use it.

**Where the remaining time goes.** Sampling profiles taken while developing the
`max_top_at_` fast path put `RecomputeAll` at roughly 40% of solve self-time, with
about a 60/40 C++/Python split overall, and the candidate gather at about
two-thirds of `RecomputeAll`. The residual Python share is the annealing driver's
own move logic plus per-call round trips (`is_fully_allocated`, `quality`,
`overlaps`), each a separate pybind crossing — a plausible second-order target,
well behind the recompute.

## Not covered

- **Captured graphs.** All instances are synthetic; the capture loader and its
  data are not available here. Real graphs have sparser in-place reuse and
  lifetimes spread over a longer schedule, both of which should favour the native
  packer more than these instances do.
- **Iso-time quality.** At equal step count the two arms produce identical
  layouts, so "more steps in the same wall clock yields better layouts" needs a
  separate matched-budget run.
- **The packer in isolation.** These figures are end-to-end through the annealing
  solver, which is what a compile experiences. A microbenchmark of the packer
  alone, driven by a synthetic operation mix, gives substantially higher ratios
  and is not directly comparable.
- **One machine.** No cross-machine or cross-toolchain variance.
