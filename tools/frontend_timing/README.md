# Frontend compile-time sweep

Measures how long the Torch-Spyre compiler frontend takes, per pass and per stage, at a
range of graph sizes. This is compile time only: no workload here is run on device, and
nothing reports kernel latency.

The instrumentation lives in the compiler (`torch_spyre/_inductor/timing_recorder.py`,
enabled by `TORCH_SPYRE_TIMING=1`). This directory is what drives it and what reads the
records back.

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
so a parallel sweep would measure contention instead of compilation. Sharding a tier
across *machines* is fine and is how the weekly tier is meant to fit.

### Tiers

One plan, three cadences, because three files drift apart. `--tier NAME` keeps the points
declaring that tier; a point with no `tiers` key runs in every tier.

| Tier | Points | Meant to answer |
|---|--:|---|
| `pr` | 7 | did a change break a compile, or move it grossly |
| `nightly` | 26 | the scaling series at moderate sizes, per commit-day |
| `weekly` | 47 | everything: long sequences, depth 8, the A/B arms, the backend share |

Measured, on one Spyre card, at `--samples 1` (so warmup plus one sample per point):

| Point | warmup | sample | ops |
|---|--:|--:|--:|
| `granite_layer` S=128 | 75.3 s | 71.5 s | 52 |
| `flash` Lk=512 | 61.9 s | 67.5 s | 71 |
| `elementwise_chain` ops=64 | 27.9 s | 23.4 s | 64 |
| `mlp` layers=1 | 23.4 s | 19.5 s | 5 |
| `fanout` consumers=8 | 18.9 s | 18.5 s | 16 |
| `dup_constants` dups=4 | 17.2 s | 15.5 s | 7 |
| `control_flow` | 14.2 s | 18.6 s | 9 |

**The whole `pr` tier is 7.9 minutes** that way. At `--samples 3` -- four passes over the
plan, since the warmup is discarded -- expect roughly four times that. The `nightly` and
`weekly` tiers have NOT been timed end to end; time them before wiring either to a job,
and record what you measured here.

Other measured points, same conditions: `granite_layer` S=512 layers=2 105.1 s (104 ops),
S=1024 65.9 s (46 ops), `granite_lm_head` S=128 chunks=4 20.5 s, `granite_embedding`
S=512 20.8 s.

### A/B arms

A point may carry `env`, which sets environment for that point's children only. That is
the only way to measure two configurations against one tree, and it is what the complexity
audit needs: the CP-SAT relayout evidence was taken before that optimization became the
default, so neither arm can be re-measured without it. Three arms ship in the plan --
`SPYRE_LX_PLANNER_RELAYOUT=0`, `SENCORES=1`, and the unbounded relayout enumeration.

An arm is part of a point's identity: it is in the record filename, in the summary row name
and in a separate `arm` field, so an arm is never averaged into its own control.

### Which axes actually grow a graph

Not all of them, and a family whose axis does not move the graph measures nothing:

- **`control_flow` has no graph-size axis, by construction.** `for_each_tile` lowers to a
  loop whose *body* is the graph, so the trip count never reaches the compiler: `M=8` and
  `M=32` produce byte-identical work -- 9 operations, 38 passes, indistinguishable times.
  It is in the plan for coverage, whether the frontend handles a scan HOP at all.
- **Growth axes that do work**: `granite_layer`/`mlp` `layers`, `flash` `Lk / block_size`
  unrolled bodies, `elementwise_chain` `ops`, `fanout` `consumers`, `dup_constants` `dups`,
  and `granite_lm_head` `chunks`.
- **Sequence length is nearly flat for compile time, and that is measured, not assumed.**
  `granite_layer` takes 52.8 s at S=128 and 44.9 s at S=1024, while `layers=2` takes
  78.1 s -- so depth moves compile time roughly linearly and `S` barely moves it at all.
  S=1024 is *cheaper* because above `_SDPA_MAX_SEQUENCE_TILE_SIZE` the decomposition takes
  the tiling path and emits 46 operations rather than 52. Sweep `S` for coverage of those
  two decomposition paths, and sweep `layers` when the question is scaling.

## Where records go

First hit wins: `--out`, then `$SPYRE_FRONTEND_TIMING_RECORDS`, then `records/` beside
`run_sweep.py`. Nothing is committed and no repository path is baked in, so a baseline
can be kept wherever it belongs.

## The protocol, and why it is shaped this way

- **One process per sample.** `TORCHINDUCTOR_CACHE_DIR` is read at import, so no
  in-process cache reset can give a sample a cache that never held this graph. Each
  child gets a fresh directory and `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`.
- **A discarded warmup**, written to `warmup/`, which the summarizer does not read.
- **Median, not mean**, across samples: pod wall time has a long tail and one contended
  sample should not move the number.
- **The backend is skipped** (`TORCH_SPYRE_FRONTEND_ONLY=1`) unless `--with-backend`.
  It dominates wall time and is not what this measures. A frontend-only compile produces
  no runnable kernel, which is why the driver never calls the compiled function twice.
- **Frontend time is a subtraction**: the `stage:compile_fx:spyre_compile` region minus
  the `backend_compile` events inside it, because the backend runs per kernel from within
  codegen rather than after it.

A summary row's point name comes from the record's *effective* parameters, so it can be
more specific than the record filename: the filename is built from what you passed, the
row from what the builder resolved, defaults included.

## The families

Two kinds. Model-shaped families answer "how long does a real shape take"; mechanism probes
move one axis a specific pass scales on, so a superlinear pass can be attributed rather
than merely observed. A probe is an instrument, not a workload anyone runs.

| Family | Kind | Axis worth moving | Notes |
|---|---|---|---|
| `granite_layer` | model | `layers` (`S` for path coverage) | Granite 3.3 8B decoder layers: 32 query heads over 8 key/value heads, intermediate 12800 |
| `granite_lm_head` | model | `chunks`, `S` | 4096 -> 49159, 201M parameters; split over the vocabulary because the whole weight does not fit (see below) |
| `granite_embedding` | model | `S` | the token table as the `index_select` it lowers to; the only indirect access here |
| `transformer_block` | model | `S` | Llama-3.1-8B dims (intermediate 14336), kept as the continuity point for older baselines |
| `mlp` | model | `layers` | SwiGLU stack |
| `flash` | model | `Lk` | block-tiled attention; the loop is unrolled at trace time |
| `elementwise_chain` | probe | `ops` | per-operation pass cost, with no matmul in the way |
| `fanout` | probe | `consumers` | one buffer read by many operations: the #4113 mechanism class |
| `dup_constants` | probe | `dups` | duplicate padding constants: dedup's own natural axis |
| `control_flow` | model | none | scan-HOP coverage only |

Granite's dimensions come from its published config, cross-checked against the shapes in
`tests/resource/models/granite-3.3-8b-instruct.yaml`. Note the difference from Llama that
is easy to miss: intermediate is 12800, not 14336, and Granite is grouped-query.

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

Counters are summed per compile rather than kept per pass: per-pass counters are in the raw
records for attribution work, but as dashboard metrics they would multiply every counter by
every pass, and a name set that grows like that stops being low-cardinality. Anything
numeric a pass puts in its event `meta` that is not a graph size becomes a counter, so a
counter added to the compiler appears here with nothing in this directory edited.

## The rows file

`--json` writes the shape the warehouse wants: one object per sweep point, and
`measurements` mapping each metric name to **the per-sample values, not a median**. That
mirrors `benchmark_runs.measurements`, which is
`Map(LowCardinality(String), Array(Float64))` precisely so variance and percentiles stay
recomputable downstream. The medians in the printed table are derived from these.

A metric only some samples carry yields a shorter array rather than a padded one: an absent
measurement must never read as a zero, which on a percentage delta would look like a total
regression.

`schema` is versioned (`frontend-timing-rows/1`). A consumer that does not recognise it
should refuse the file rather than guess, the way the recorder's own `RECORDER_VERSION`
works. `scaling.py` does exactly that.

Ingesting this into ClickHouse is a follow-up, not part of this directory. The intended
mapping is one `benchmark_runs` row per point, keyed through
`extensions/clickhouse-ingest`'s derived ids, with `measurements` passed through as-is --
so the ingest is a translation with no arithmetic in it. Keeping the arithmetic here, where
it is tested without a database, is the point.

## Scaling and the 40-layer number

`scaling.py` finds every series -- points agreeing on everything but one axis -- and fits
`log(metric) = a * log(axis) + b`, reporting the exponent with its R-squared. It refuses to
quote an exponent from fewer than four points or below an R-squared floor, because a slope
through three noisy points always produces a number and such numbers have been the basis of
complexity claims before.

An example of its output, measured 2026-09-24 on one card at `--samples 1` -- an
illustration of the shape, not a baseline, and nothing checks it for staleness:

```
| series                                | metric                          | n | exponent |    R2 | verdict |
| elementwise_chain[ops] cols1024_rows256 | frontend_ms                   | 4 |     1.14 | 0.989 | ok      |
| elementwise_chain[ops] cols1024_rows256 | graph_operations              | 4 |     1.00 | 1.000 | ok      |
| elementwise_chain[ops] cols1024_rows256 | counter.read_writes.extractions | 4 |   1.00 | 1.000 | ok      |
| elementwise_chain[ops] cols1024_rows256 | peak_rss_kb                   | 4 |     0.64 | 0.922 | ok      |
```

Two things that says. `ops` maps exactly one-to-one onto operations, so the probe's axis is
the graph size it claims to be. And pointwise compile time is only mildly superlinear in
operation count -- which is what makes the model-shaped families worth measuring separately,
since `granite_layer` spends about 216 dependency extractions per operation against this
family's steady handful.

Granite 3.3 8B is 40 decoder layers and nothing here compiles 40 layers in one graph: a
single Granite attention op at `LQ=LK=1024` with 16 key/value groups already takes over ten
minutes (`tests/inductor/test_building_blocks.py` skips it for exactly that reason). So the
40-layer figure is `--extrapolate layers=40` off the measured depth series, and it must be
reported as a projection with the fit it rests on, never as a measurement.

## Adding a workload

1. Write a builder in `workloads.py` returning a `Workload`. Take sizes as keyword
   arguments -- the point of a family is that one parameter moves and the rest hold still.
2. Register it in `BUILDERS`.
3. Add points to `sweep_plan.json`, each with its `tiers`.

`tests/test_frontend_timing_suite.py` will fail if a point names a parameter its builder
does not accept, if a point forgets `tiers` (which would silently put it in the per-PR
lane), or if a registered family is never swept. It reads `workloads.py` with `ast` rather
than importing it, so it needs no built extension.

Derive the body from a test that passes today and name that test in the docstring. The
bodies here are copies rather than imports: a baseline is only comparable if the workload
did not move, and test helpers move for test reasons.

## Adding a metric

1. Wrap the region in the compiler with `timing_recorder.stage("stage:<Owner>:<what>")`.
   Counts that describe a region belong in its `meta`, measured outside the region so
   counting is not charged to the work.
2. Nothing in the summarizer needs to change for a new `stage:` or `pass:` region -- both
   are collected by prefix -- nor for a new counter, which is any non-graph-size number a
   pass puts in its event `meta`. Add code only for something that is neither.

## Workload constraints the backend imposes

Three shape rules bound what the attention families can sweep. All surface as assertions
deep in lowering, so they are checked in the builder instead:

- **`head_dim` must be a multiple of 64.** 64 fp16 elements is one stick, and a fractional
  head lands as `Unsupported coordinate expression 5*c0/2`. Ministral-style dims therefore
  sweep as `E=5120, heads=40` (`head_dim` 128), not `heads=32`.
- **A reshape that splits a named dim must re-annotate it.** Projecting `[B, S, E]` and
  reshaping into `[B, S, H, D]` otherwise fails above `S=512` with `layout dim 2 has 2 loop
  vars but only 1 name(s)`. The builder wraps the reshape and transpose in `spyre_hint`.
  512 is not arbitrary: it is `_SDPA_MAX_SEQUENCE_TILE_SIZE` in `decompositions.py`, above
  which the decomposition must tile the query dimension and therefore needs those names to
  exist. The hints go on the reshapes rather than on the input tensors via the eager
  `name_tensor_dims` API, because q, k and v are computed inside the graph -- naming the
  inputs would never reach the reshape that splits the head dimension.
- **Query and key/value head dims need different names under GQA.** `granite_layer` labels
  them `H` and `Hkv`: one name at two sizes is a conflict, not a shortcut.
- **The language-model head does not fit unsplit.** Measured: work division rejects the
  whole `[4096, 49159]` weight with "per-core tensor span 384.500 MB ... exceeds hardware
  limit of 256.00 MB". `SENCORES` cannot rescue it -- 32 cores is already the maximum and
  fewer cores means more per core -- so `granite_lm_head` splits the projection into
  `chunks` matmuls, as a real implementation does for a vocabulary this size. `chunks=2`
  is the coarsest split measured to fit.

## What is not covered

**Forward only.** Nothing here compiles a backward graph, so passes that only run on
training graphs have no coverage.

**Depth is swept, but not to 40.** `granite_layer` and `mlp` both take `layers`, so the
depth axis is measured on a real decoder layer rather than assumed. What is not measured is
the full model: see "Scaling and the 40-layer number" for why, and for what is reported
instead. A real model's cost is not assumed to be depth times one block -- that is what the
depth series exists to test.

**No embedding, as such.** There is no `embedding` lowering in the backend, so
`granite_embedding` is the same memory access written as `index_select`. The backend's
indirect-access numerics are an expected failure today, which does not matter here: a
frontend-only compile never runs the kernel.

**No full model.** Nothing assembles embedding, layers, norm and head into one graph. The
pieces are measured separately and the depth exponent fitted; composing them is arithmetic
a reader can do, and a claim the suite does not make.

## What the decomposition does not cover yet

`GraphLowering.run` and `GraphLowering.codegen` are not timed, so lowering and codegen
time currently lands in the self-time of `stage:compile_fx:spyre_compile`. Dynamo and
AOTAutograd are not separated either. Both are follow-ups; until then, read
`spyre_compile` self-time as "everything upstream of the Spyre pipelines" rather than as
overhead.
