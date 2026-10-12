# Frontend compile-time sweep

Measures how long the Torch-Spyre compiler frontend takes, per pass and per stage, across
a range of graph sizes. Compile time only: nothing here runs on device or reports kernel
latency.

The instrumentation is `torch_spyre/_inductor/timing_recorder.py`, enabled by
`TORCH_SPYRE_TIMING=1`. This directory drives it and reads the records back.

## Running it

One point, three cold samples plus a discarded warmup:

```bash
python3 tools/frontend_timing/run_sweep.py --workload mlp -p seq_len=128 -p layers=2 \
    --out /tmp/records
python3 tools/frontend_timing/summarize.py /tmp/records --passes
```

One tier of the plan, with the rows file a dashboard reads:

```bash
python3 tools/frontend_timing/run_sweep.py --plan tools/frontend_timing/sweep_plan.json \
    --tier nightly --out /tmp/records
python3 tools/frontend_timing/summarize.py /tmp/records --tier nightly \
    --csv /tmp/frontend.csv --json /tmp/rows.json
python3 tools/frontend_timing/scaling.py /tmp/rows.json --extrapolate layers=40
```

Needs a Spyre device. Samples run serially because the device is exclusive per process,
so a parallel sweep would measure contention. Sharding a tier across machines is fine.

### Tiers

`--tier NAME` keeps the points that list that tier; a point with no `tiers` runs in every
tier.

| Tier | Meant to answer |
|---|---|
| `pr` | did a change break a compile, or move it grossly |
| `nightly` | the scaling series at moderate sizes |
| `weekly` | everything: long sequences, depth 8, the A/B arms |

### A/B arms

A point may carry `env`, set for that point's children only, to measure two
configurations against one tree. The plan ships three: `SPYRE_LX_PLANNER_RELAYOUT=0`,
`SENCORES=1` and the unbounded relayout enumeration. An arm is part of a point's identity
(record filename, row name and a separate `arm` field), so it is never averaged into its
own control.

### Records are never reused

A sweep refuses a `--out` that already holds records. The recorder puts the pid in each
filename, so nothing is ever replaced: a second sweep into the same directory leaves both
runs' samples side by side and the summarizer averages them into one point. `--append`
allows it deliberately. The summarizer independently refuses records that disagree on
`git_sha`, torch version or Spyre config, which is the same mistake seen from the other
end; `--allow-mixed` overrides that.

The default destination persists between runs for exactly this reason, and is gitignored.

## Where records go

First hit wins: `--out`, then `$SPYRE_FRONTEND_TIMING_RECORDS`, then `records/` beside
`run_sweep.py`. Nothing is committed.

## The protocol

- **One process per sample.** `TORCHINDUCTOR_CACHE_DIR` is read at import, so only a
  fresh process gets a cache that never held the graph. Each child gets a fresh directory
  and `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`.
- **A discarded warmup**, written to `warmup/`, which the summarizer does not read.
- **The minimum across samples**, not the median. Co-tenant interference on a shared
  pod only ever makes a compile slower -- measured at 11% typical and 63% worst spread
  across three samples of one point -- so the fastest sample is the least contaminated
  one, and the long tail a median would resist is exactly what should be discarded. The
  rows file still carries every sample, so a median stays recoverable.
- **No backend** (`TORCH_SPYRE_FRONTEND_ONLY=1`) unless `--with-backend`. A frontend-only
  compile produces no runnable kernel, which is why the driver never calls the compiled
  function twice.
- **Frontend time is a subtraction**: `stage:compile_fx:spyre_compile` minus the
  `backend_compile` events inside it, because the backend runs per kernel from within
  codegen.

A summary row is named from the builder's effective parameters, defaults included, so it
can be more specific than the record filename, which uses only what you passed.

## The families

Model-shaped families answer "how long does a real shape take"; probes move one axis a
specific pass scales on, so a superlinear pass can be attributed.

| Family | Kind | Axis worth moving | Notes |
|---|---|---|---|
| `granite_layer` | model | `layers` | Granite 3.3 8B: 32 query heads over 8 key/value heads, intermediate 12800 |
| `granite_lm_head` | model | `chunks`, `S` | 4096 -> 49159, split over the vocabulary |
| `granite_embedding` | model | `S` | `index_select` over the token table; the only indirect access |
| `transformer_block` | model | `S` | Llama-3.1-8B dims (intermediate 14336), for continuity with older baselines |
| `mlp` | model | `layers` | SwiGLU stack |
| `flash` | model | `Lk` | block-tiled attention, unrolled at trace time |
| `elementwise_chain` | probe | `ops` | per-operation pass cost, no matmul |
| `fanout` | probe | `consumers` | one buffer read by many operations (#4113) |
| `dup_constants` | probe | `dups` | duplicate padding constants, dedup's natural axis |
| `control_flow` | model | none | scan-HOP coverage only |

Sequence length barely moves compile time, while depth moves it roughly linearly. Sweep
`S` to cover both SDPA decomposition paths, either side of `_SDPA_MAX_SEQUENCE_TILE_SIZE`
(512), and `layers` when the question is scaling.

## What the metrics are called

One grammar, because these names become warehouse map keys:

| Name | Meaning |
|---|---|
| `total_ms` | the whole `stage:compile_fx:spyre_compile` region |
| `backend_ms` | the `stage:SpyreAsyncCompile:backend_compile` events inside it; absent when the backend was skipped |
| `frontend_ms` | `total - backend`, subtracted **within each sample** |
| `graph_operations`, `graph_nodes` | pre-scheduling operations, and FX nodes entering the pre-grad pipeline |
| `stage.<Owner>.<what>_ms` | one stage region, inclusive |
| `pass.<Pipeline>.<pass>_ms` | one pass, inclusive |
| `<region>_self_ms` | the same region's self time: inclusive minus its direct children |
| `counter.<name>` | a count from pass metadata, summed over the compile |
| `counter.<Owner>.<what>.<name>` | a count from one stage region's metadata |
| `peak_rss_kb` | the compiling process's peak RSS (Linux reports KB) |
| `compile_wall_ms` | wall time around the whole compiled call, which is strictly more than `total_ms` |
| `kernels_skipped` | kernels the frontend-only mode declined to compile |

`backend_ms` is matched exactly, never by trailing segment: upstream has a
`dynamo_timed` phase of its own called `backend_compile` -- the call into Inductor --
which *encloses* the Spyre compile region, so a suffix match totals the whole Inductor
compile as backend time and drives every frontend figure negative.

Both time series are recorded because neither answers the other's question. Inclusive is
what the frontend subtraction needs and what shows the nesting; it also sums to more than
the compile, so ranking regions by it puts every ancestor of the hot pass near the top.
Self time is what ranks cost, and the `--passes` table orders by it.

Pass counts are summed per compile under their bare key, to keep the name set small;
per-pass counts stay in the raw records. Stage counts are keyed by their region instead,
because two regions can carry the same key for different things -- the scratchpad records
`buffers` on more than one -- and summing those would report a number that is neither.
Any number a pass puts in its event `meta` that is not a graph size becomes
a counter, with no change needed here.

## The rows file

`--json` writes one object per point, with `measurements` mapping each metric to its
**per-sample values**, not a summary. That mirrors `benchmark_runs.measurements`
(`Map(LowCardinality(String), Array(Float64))`), so variance stays recomputable
downstream. A metric only some samples carry gets a shorter array, never a padded zero.

`schema` is versioned (`frontend-timing-rows/1`). A consumer that does not recognise it
should refuse the file, as `scaling.py` does.

## Scaling

`scaling.py` finds every series -- points agreeing on everything but one axis -- and fits
`log(metric) = a * log(axis) + b`, reporting the exponent with its R-squared. It will not
quote an exponent from fewer than four points or below an R-squared floor.

Nothing compiles Granite 3.3 8B's 40 layers in one graph, so `--extrapolate layers=40`
projects from the depth series. Report it as a projection, with its fit.

## Adding a workload

1. Write a builder in `workloads.py` returning a `Workload`, with sizes as keyword
   arguments. Copy the body from a test that compiles today and name that test in the
   docstring.
2. Register it in `BUILDERS`.
3. Add points to `sweep_plan.json`, each with its `tiers`.

`tests/test_frontend_timing_suite.py` fails if a point names a parameter its builder does
not accept, if a point has no `tiers`, or if a registered family is never swept.

## Adding a metric

1. Wrap the region in the compiler with `timing_recorder.stage("stage:<Owner>:<what>")`.
   Counts that describe a region belong in its `meta`, measured outside the region so
   counting is not charged to the work.
2. Nothing in the summarizer changes for a new `stage:` or `pass:` region, both collected
   by prefix, or for a new counter. Add code only for something that is neither.

## What is not covered

- **Backward graphs.** Passes that only run on training graphs have no coverage.
- **A full model in one graph.** Embedding, layers, norm and head are measured
  separately.
- **`GraphLowering.run` and `GraphLowering.codegen`**, and Dynamo versus AOTAutograd, are
  not timed separately, so that time lands in `stage:compile_fx:spyre_compile` self time
  -- which the `_self_ms` series now reports.
- **The backend share.** No plan point enables the backend; `--with-backend` is a
  whole-sweep flag, so measuring that share means a separate run.
- **A scaling curve in `S`.** Sequence length barely moves compile time, and `S=1` takes
  the decomposition's decode branch, so the `S` points span a decode/prefill switch. They
  are coverage of two paths, not one power law -- `scaling.py` will fit them and should be
  expected to call that fit unreliable.
