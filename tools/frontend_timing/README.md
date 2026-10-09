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
| `weekly` | everything: long sequences, depth 8, the A/B arms, the backend share |

### A/B arms

A point may carry `env`, set for that point's children only, to measure two
configurations against one tree. The plan ships three: `SPYRE_LX_PLANNER_RELAYOUT=0`,
`SENCORES=1` and the unbounded relayout enumeration. An arm is part of a point's identity
(record filename, row name and a separate `arm` field), so it is never averaged into its
own control.

## Where records go

First hit wins: `--out`, then `$SPYRE_FRONTEND_TIMING_RECORDS`, then `records/` beside
`run_sweep.py`. Nothing is committed.

## The protocol

- **One process per sample.** `TORCHINDUCTOR_CACHE_DIR` is read at import, so only a
  fresh process gets a cache that never held the graph. Each child gets a fresh directory
  and `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`.
- **A discarded warmup**, written to `warmup/`, which the summarizer does not read.
- **Median across samples**: pod wall time has a long tail.
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
| `backend_ms` | every `:backend_compile` event inside it |
| `frontend_ms` | `total - backend`, subtracted **within each sample** |
| `graph_operations`, `graph_nodes` | largest pre-scheduling graph, and largest FX graph |
| `stage.<Owner>.<what>_ms` | one stage region |
| `pass.<Pipeline>.<pass>_ms` | one pass |
| `counter.<name>` | an analysis-call count, summed over every pass in the compile |
| `peak_rss_kb` | the compiling process's peak RSS (Linux reports KB) |
| `compile_wall_ms` | wall time around the compile, outside the recorder |
| `kernels_skipped` | kernels the frontend-only mode declined to compile |

Counters are summed per compile to keep the name set small; per-pass counts stay in the
raw records. Any number a pass puts in its event `meta` that is not a graph size becomes
a counter, with no change needed here.

## The rows file

`--json` writes one object per point, with `measurements` mapping each metric to its
**per-sample values**, not a median. That mirrors `benchmark_runs.measurements`
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
  not timed separately, so that time lands in `stage:compile_fx:spyre_compile` self time.
