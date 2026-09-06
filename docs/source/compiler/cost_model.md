# Analytical Cost Model

The cost model predicts how long a compiled graph will take to run, from the loop-level
IR, without running it. It exists so the compiler can compare two candidate plans —
different coarse tile sizes, different work divisions, different scratchpad placements —
cheaply enough to do it during compilation.

It is **off by default** and adds nothing to a normal compile.

Today, turning it on **prints** a prediction and does nothing else — it never changes what
gets compiled. The estimate is also returned as a value, so a pass or a tool can read it
during compilation and choose between two plans; see
[Using it as an API during compilation](#using-it-as-an-api-during-compilation). Nothing in
the compiler consumes it yet.

> **Not to be confused with** `work_division.cost_model_matmul_division`, an unrelated
> model that chooses a matmul work division. This page describes the whole-program
> runtime model in `cost_model.py` and the reporting pass in `cost_model_pass.py`.

## Where it runs

The pass runs after LX planning and before the final Scheduler-boundary work-division
conversion, at the point where the loop-level IR is final: layouts resolved, restickify
inserted, coarse tiling applied, symbol-keyed work division committed, scratchpad placement
done. Nothing after this changes what the program moves or computes — only how it is
packaged into kernels.

```text
                        torch.compile
                              │
                              ▼
   FX graph ──▶ Inductor lowering ──▶ graph.operations   (loop-level IR)
                                              │
     ┌────────────────────────────────────────▼────────────────────────────┐
     │                    CustomPreSchedulingPasses                        │
     │  deadcode ▸ split_multi_ops ▸ layouts ▸ restickify ▸ coarse_tile    │
     │  ▸ work_division ▸ scratchpad_planning                              │
     └────────────────────────────────────────┬────────────────────────────┘
                                              │
                          ═══════════ IR IS FINAL HERE ═══════════
                          bytes, tiling and core division all settled
                                              │
                          cost_model_pass ◀───┘   off unless config.cost_model
                                  │                is set — otherwise it returns
                                  │                before touching the graph
                          dump_cost_model
                                  │
                  finalize Scheduler transport
                                  │
      ┌───────────────────────────┴──────┐
      │ group ops into the kernels the   │
      │ backend will fuse, then price    │
      │ each ONCE with cost_model.py     │
      └───────────────────────────┬──────┘
                                  ▼
                        ┌──────────────────┐
                        │   CostReport     │──────▶ printed breakdown
                        │  .total_us       │
                        │  .groups[]       │──────▶ returned to the caller,
                        └──────────────────┘        stored as last_cost_report
                                  ╎
                                  ╎ compare plan A vs plan B without compiling
                                  ╎ or running either — tile size, work division
                                  ╎ and scratchpad placement are all chosen in
                                  ╎ CustomPreSchedulingPasses, above
                                  └╌╌╌╌╌╌╌╌╌╌╌╌╌╌▶ back to pre-scheduling

                              Inductor Scheduler
                                      │
                                      ▼
                              spyre_fuse_nodes
                                      │
                                      ▼
                                  SuperDSC ──▶ DeepTools ──▶ device
```

The estimate returns to pre-scheduling because tile size, work division and scratchpad
placement are all decided there, so a planner can price two candidate plans before
committing to either.

## Turning it on

Set `SPYRE_DUMP_COST`, or patch the config directly:

| value | effect |
|---|---|
| unset, `""`, `0`, `false`, `off` | disabled — the pass returns before touching the graph |
| `1`, `true`, `yes`, `on` | print a per-kernel breakdown and the program total after pre-scheduling |

Any example in `docs/source/user_guide/examples/` works. The smallest is a single
activation:

```bash
SPYRE_DUMP_COST=1 python docs/source/user_guide/examples/gelu.py
```

Or in Python, which is also how tests reach it:

```python
from torch_spyre._inductor import config

with config.patch({"cost_model": "1"}):
    compiled = torch.compile(model)
```

## What it prints when turned on

### A plain kernel

Softmax over a 4096 × 2048 tensor on 32 cores, not tiled:

```bash
BENCH_OP=softmax_row_tiling BENCH_ROWS=4096 BENCH_COLS=2048 BENCH_TILES=1 \
  SPYRE_DUMP_COST=1 python docs/source/user_guide/examples/profile_ops.py
```

```text
predicted total:      320.0 us over 1 kernel(s), 6 op(s)

      kernel   predicted us  ops
           0          320.0  6
                      160.0    Pointwise HBM   16.8 MB   LX   16.8 MB
                      160.0    div       HBM   16.8 MB   LX   17.3 MB
                        0.0    amax      HBM         -   LX   17.3 MB
                        0.0    sub       HBM         -   LX   34.1 MB
                        0.0    exp       HBM         -   LX   33.6 MB
                        0.0    sum_1     HBM         -   LX   17.3 MB

  Operations are grouped by the loop they run in; the count on each group
  header is that loop's iterations.  HBM = main-memory traffic for the whole
  loop, and '= T x B' breaks it into T iterations of B bytes.  LX = on-chip
  scratchpad traffic, already ONE iteration (an LX buffer is allocated per
  tile).  re-fetch = the part of HBM that is the same bytes read again on a
  later iteration.
  ...
```

The six operations fuse into one kernel: row `0` with `ops 6`, its operations indented
below. Nothing loops here.

Only two of them touch main memory. `Pointwise` — the copy that stages the input into LX —
reads the 16.8 MB argument, and `div` writes the 16.8 MB result; each therefore carries
half the kernel's predicted time. The other four show **0 HBM I/O**: their inputs and
outputs are both in LX, and LX traffic is charged 0, so they take no share of the time.
None of the six both reads and writes HBM.

That is the attribution behaving as documented, not four free operations: the kernel is
priced once as a bundle, and the per-op column splits *that* total by each op's share of
the HBM bytes.

### A coarse-tiled kernel

The same softmax, now split into 8 tiles:

```bash
BENCH_OP=softmax_row_tiling BENCH_ROWS=4096 BENCH_COLS=2048 BENCH_TILES=8 \
  SPYRE_DUMP_COST=1 python docs/source/user_guide/examples/profile_ops.py
```

```text
predicted total:      320.0 us over 1 kernel(s), 7 op(s)

      kernel   predicted us  ops
           0          320.0  7
                      320.0    loop 0, runs 8 times
                      160.0      Pointwise HBM   16.8 MB = 8 x 2.1 MB   LX    2.1 MB
                      160.0      Pointwise HBM   16.8 MB = 8 x 2.1 MB   LX    2.1 MB
                        0.0      amax      HBM         -   LX    2.2 MB
                        0.0      sub       HBM         -   LX    4.3 MB
                        0.0      exp       HBM         -   LX    4.2 MB
                        0.0      sum_1     HBM         -   LX    2.2 MB
                        0.0      div       HBM         -   LX    4.3 MB
```

Seven operations now, and everything runs inside the loop. The two `Pointwise` rows are
the coarse-tiling copies: one stages a tile of the argument into LX at the top of each
iteration, the other writes the finished tile back to HBM. They are the only two that
touch main memory, so they carry the whole predicted time between them — including `div`,
which wrote the result directly in the untiled version and now leaves it in LX for the
copy-out.

The rows report two scales at once:

| what you see | what it is |
|---|---|
| `loop 0, runs 8 times` | the block below it runs eight times |
| `320.0 µs` on that header | time for **all eight** iterations |
| `HBM 16.8 MB` | HBM traffic across **all eight** |
| `= 8 x 2.1 MB` | HBM traffic for **one** iteration |
| `LX 2.1 MB` | LX traffic for **one** iteration |

Nothing needs multiplying by the iteration count. The per-iteration figure is the one that
decides whether a tile fits in LX, and the one re-tiling moves.

The two memories are reported on different bases: an HBM operand is recorded at its untiled
size and divided down, while an LX buffer is allocated per tile and is already one
iteration.

### A kernel with both

Operations in one kernel need not run the same number of times. Under K-tiling the
accumulator's fill op sits outside the loop:

```bash
BENCH_OP=bmm_k_tiling BENCH_B=4 BENCH_ROWS=1024 BENCH_COLS=2048 BENCH_N=1024 \
  BENCH_TILES=8 SPYRE_DUMP_COST=1 python docs/source/user_guide/examples/profile_ops.py
```

```text
predicted total:   288297.4 us over 1 kernel(s), 5 op(s)

      kernel   predicted us  ops
           0       288297.4  5
                   288235.7    loop 0, runs 8 times
                   127337.0      Pointwise HBM 17314.1 MB = 8 x 2164.3 MB
                   127337.0      Pointwise HBM 17314.1 MB = 8 x 2164.3 MB
                    32081.0      bmm       HBM 4362.1 MB = 8 x 545.3 MB
                     1480.7      Pointwise HBM  201.3 MB = 8 x 25.2 MB
                       61.7    not in a loop, runs once
                       61.7      Pointwise HBM    8.4 MB
```

The fill runs **once**, so its `8.4 MB` is all of its traffic; the copies and the matmul
above it run eight times. A kernel has no single iteration count, which is why the count
sits on a block header rather than in a column — each block states it for the operations
it governs and carries their subtotal, here `288235.7 µs` of `288297.4 µs`.

Operations land in **separate** kernels only when something between them cannot be fused.
The program total is the sum over kernels.

## Using it as an API during compilation

The pass returns a `CostReport`, so another pass or a tool can compare plans. `graph` is the
`GraphLowering` any pre-scheduling pass receives; the pass reads `graph.operations` and never
mutates it:

```python
from torch_spyre._inductor.cost_model_pass import cost_model_pass


def my_planning_pass(graph):          # a GraphLowering, as any pre-scheduling pass gets
    report = cost_model_pass(graph)   # None when disabled
    if report is not None and report.total_us < best_so_far:
        ...
```

Call it after coarse tiling, work division and scratchpad placement — earlier, the
operations do not yet carry the tiling the estimate depends on. The report is also readable
afterwards as `last_cost_report` on the pipeline instance, or as
`cost_model_pass.LAST_REPORT`. Both read the same per-thread storage, so a thread always
gets the report for the graph it compiled itself and never one from a concurrent compile.

| field | meaning |
|---|---|
| `CostReport.total_us` | predicted runtime for the whole graph — the number to compare |
| `CostReport.groups` | one entry per kernel, in program order (only the printout sorts by cost) |
| `GroupCost.predicted_us` | that kernel's predicted time |
| `GroupCost.loop_group_ids` | the coarse-tiling loops inside this kernel, for labelling |
| `GroupCost.ops` | per-operation attribution, not independent predictions — see [limitations](#two-limitations-worth-knowing) |
| `GroupCost.loop_trip` | the **deepest** loop in this kernel, not a count that applies to every operation in it — read `OpCost.trip` per operation |
| `OpCost.hbm_bytes` | main-memory traffic across the **whole** loop |
| `OpCost.trip` | how many times **this** operation runs; `1` for one placed outside the loop |
| `OpCost.hbm_per_iter` | HBM traffic in **one** iteration of this operation — the working set |
| `OpCost.lx_bytes` | on-chip traffic, **already** per-iteration (LX is allocated per tile) |
| `OpCost.reread_bytes` | of `hbm_bytes`, the excess over a single pass — see the [appendix](#appendix-the-re-fetch-column) |

`OpCost.trip * hbm_per_iter` reproduces `hbm_bytes` — exactly when the count divides the
traffic, which holds on every recorded op, and otherwise short by under a byte per
iteration. A planner comparing tile counts wants `hbm_per_iter`, since that is the figure
re-tiling moves.

## How a kernel is priced

A fused kernel is the only thing that can be measured — there is no per-operation timing
inside one — so every coefficient in `cost_model.py` was fitted against whole kernels, and a
prediction is comparable to a measurement only over that unit. The pass therefore
reconstructs the kernels the backend will fuse before pricing anything.

`predict_ops` is **not additive** over the operations in a kernel: a shared input is charged
once rather than once each, and the turnaround and overlap terms are taken over kernel
totals. Per-operation prices do not sum to the kernel price, which is why the per-operation
column is an attribution.

`spyre_fuse_nodes` fuses everything it can — contiguous Spyre nodes accumulate in order, and
only a node that is not on the device starts a new bundle. No size limit, no cost heuristic,
no reordering. The pass applies that same rule, with two differences:

- it also breaks a kernel at an operation the extractor cannot model, where the backend
  would not, so the report prices two smaller kernels where the device runs one;
- with `bundle_symbolic_args` off the backend does not fuse at all, and neither does the
  report — every operation becomes its own kernel.

`loop_group_id` labels the loop structure inside a kernel; it does not decide boundaries.

## Two limitations worth knowing

**Compute and memory do not combine by any form we can justify.** A matmul has an arithmetic
cost and a memory cost, and the kernel is charged `max(compute, memory)` — the two overlap,
so it takes the longer. That is the mechanism, but it is not the measurement: it under-charges
by 22–31 % on the mean across every matmul family, because part of the shorter stream fails to
hide behind the longer and nothing charges for it. Adding the two instead over-predicts by a
similar margin. Neither form is right, and no mechanism is known for what sits between them —
a fitted fraction closes the gap numerically but its value is unstable across populations, so
it was removed rather than shipped.

This is the largest single source of error in the model. It is confined to matmul — it is why
`matmul, split` reads 29.8 % and `bmm` up to 40.0 % above, while pointwise, broadcast,
reduction and transport all sit under 10 %. (Softmax's 21.1 % has a different cause: a
per-core throughput floor that used to cover partial-machine runs was removed for the same
reason, having no settled mechanism.) The bias runs one way, so a predicted time for a
matmul-heavy graph is a lower bound rather than an estimate. Ranking is unaffected — see the
note under [Accuracy](#accuracy).

**Per-operation times are an attribution, not predictions.** Each kernel is priced once, as
a whole; the per-op column then splits that total by each operation's share of the
main-memory bytes. The parts sum to the kernel total by construction, but they are not
separately meaningful. Two known distortions: the split carries no compute term, so it
misattributes a compute-bound kernel; and the weights are per-operation while the total
de-duplicates shared inputs, so a share can be off by up to a third of itself (measured on
recorded softmax bundles: 33.3 % where a consistent split gives 25.0 %). An operation whose
output stays in LX shows `0.0` because it adds no HBM traffic of its own — it was fused away,
not free.

## Accuracy

RMS error by category, as scored by `tools/cost_model/eval_model.py` over the recorded
sweep:

| category | RMS | category | RMS |
|---|---:|---|---:|
| broadcast | 5.0 % | softmax | 21.1 % |
| transport | 6.1 % | matmul, row-tiled | 25.1 % |
| reduction | 7.2 % | matmul, split | 29.8 % |
| pointwise | 8.5 % | bmm | 26.5–40.0 % |

The right column carries a **systematic under-prediction**, not scatter: compute and memory
are charged as `max(compute, memory)`, and nothing charges for the part of the shorter stream
that fails to hide behind the longer. Ordering survives it — rank correlation with measured
runs +0.875 to +0.998 across every category — so the model is usable for choosing between
plans and not for predicting a wall-clock number.

Two op families are excluded from these figures because nothing models them: flash attention, and the `write` outer product (`b[1,C] + c[R,1]`), whose fitted surface was removed as an unexplained black box.

## Measuring, scoring and refreshing the model

The model is calibrated against a database of measured kernels, and everything needed to
reproduce or refresh that is in the tree.

| what | where |
|---|---|
| the measurement harness — runs one op under the PyTorch profiler and prints a parseable summary | `docs/source/user_guide/examples/profile_ops.py` |
| measure the whole sweep on this machine | `docs/source/user_guide/examples/run_cost_model_sweep.py` |
| the sweep plan — which configurations the sweep measures | `tools/cost_model/sweep_plan.json` |
| the database itself — one record per measured kernel | generated by the sweep, not committed — see `tools/cost_model/records.py` |
| score the model against the database, offline, with no hardware | `tools/cost_model/eval_model.py` |
| fold a sweep log back into the database | `tools/cost_model/parse_sweep_logs.py` |

Measure one configuration:

```bash
BENCH_OP=softmax_row_tiling BENCH_ROWS=4096 BENCH_COLS=2048 BENCH_TILES=8 \
  python3 docs/source/user_guide/examples/profile_ops.py
```

**Generate the measurement database yourself.** It is measured device time, so it belongs to
the machine and toolchain that produced it, and the compiler is under active development —
kernel performance moves with it. A spot-check while preparing this feature measured 261 µs
for a configuration the reference database has at 390 µs, on the same shape and core count,
because an upstream change now pins a shared graph input into LX. Numbers taken on another
build describe that build.

The sweep needs nothing but hardware. What to measure is the **sweep plan**,
`tools/cost_model/sweep_plan.json`, which ships with the repository — a list of
environments (shapes, core counts, tilings, work divisions), not measurements. So a fresh
checkout builds its own database from zero:

```bash
python3 docs/source/user_guide/examples/run_cost_model_sweep.py --dry-run   # what it would run
python3 docs/source/user_guide/examples/run_cost_model_sweep.py --limit 20  # a timed pilot
python3 docs/source/user_guide/examples/run_cost_model_sweep.py             # 1535 configurations
python3 tools/cost_model/eval_model.py                                      # score, no hardware
```

`--limit` samples evenly across the plan rather than taking the first N, so a pilot's timing
is representative of the whole sweep; multiply it to size the full run. `--resume <log>`
continues an interrupted one, skipping the configurations that already produced a
measurement.

The sweep writes `sweep_records.json` into `tools/cost_model/`. It is not committed: ~14 MB
of JSON that grows with every re-sweep would dominate the history for a file no diff can
usefully show. Each log opens with a provenance header, so every record carries the git sha
and date of the sweep that produced it — measurements from different builds must not be
pooled.

A **reference copy** — collected on PyTorch 2.11, and the one the accuracy figures below were
computed from — is linked in `tools/cost_model/records.py` for orientation. Nothing fetches
it automatically; treat it as a starting point, not as a measurement of your build. If you
have a database elsewhere, point the scoring tools at it:

```bash
export SPYRE_COST_MODEL_RECORDS=/path/to/sweep_records.json
```

Every tool that needs the database resolves it the same way through
`tools/cost_model/records.py`, and prints these instructions if it cannot find one. The
sweep is the exception: it creates the database, so it never requires one.

Scoring is a pure computation over the stored measurements — no hardware, no re-running —
which makes it the loop to use when changing a term:

```bash
python3 tools/cost_model/eval_model.py
```

The derivation of every term, with the isolation experiment behind it, is in the module
docstring of `torch_spyre/_inductor/cost_model.py`.

## Implementation

| file | role |
|---|---|
| `cost_model.py` | the model itself: pure Python, no torch dependency, importable standalone |
| `dump_cost_model.py` | IR → `OpFeatures` extraction, and the older per-op feature dump |
| `cost_model_pass.py` | the pass: grouping, per-group pricing, the report |
| `tests/inductor/test_cost_model_pass.py` | grouping, attribution and disabled-path guards |

The pass never raises: every entry point catches and logs instead.

## Appendix: the `re-fetch` column

Most operands a loop touches are read once overall — each iteration takes a fresh slice and
the slices cover the tensor once. A row-tiled matmul is different: it splits the output rows
across iterations, but every iteration needs the **whole** B matrix. `re-fetch` reports what
that costs.

```bash
BENCH_OP=matmul_row_tiling BENCH_ROWS=4096 BENCH_COLS=2048 BENCH_N=1024 \
  BENCH_TILES=8 SPYRE_DUMP_COST=1 python docs/source/user_guide/examples/profile_ops.py
```

```text
      kernel   predicted us  ops
           0          423.1  1
                      423.1    loop 0, runs 8 times
                      423.1      mm HBM   58.7 MB = 8 x 7.3 MB   re-fetch 29.4 MB
```

The three tensors total 29.4 MB — A is 16.8, the output 8.4, B only 4.2 — yet the loop
moves 58.7 MB. A and the output are read and written once between them. B is read eight
times, 33.6 MB out of a 4.2 MB tensor, and the 29.4 MB of `re-fetch` is the seven extra
passes. It is the smallest tensor and the largest cost.

The column appears only for an input of a matmul whose loop tiles an **output** dimension,
and takes its figure from `cost_model._loop_reread_bytes`. Under **reduction** tiling the same access pattern means
the opposite — each iteration consumes a fresh slice of K, so those operands are read once
and nothing is re-fetched.
