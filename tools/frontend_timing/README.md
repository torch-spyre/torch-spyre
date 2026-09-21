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

The default sweep:

```bash
python3 tools/frontend_timing/run_sweep.py --plan tools/frontend_timing/sweep_plan.json \
    --out /tmp/records
python3 tools/frontend_timing/summarize.py /tmp/records --csv /tmp/frontend.csv
```

Needs a Spyre device. Samples run serially because the device is exclusive per process,
so a parallel sweep would measure contention instead of compilation. Budget accordingly:
the default plan at one sample per point takes about 10 minutes, and its slowest point
(`flash` at `Lk=1024`) is ~84 s per compile, so the default `--samples 3` -- four passes
over the plan, since the warmup is discarded -- is nearer 30.

**Control flow has no graph-size axis, by construction.** `for_each_tile` lowers to a
loop whose *body* is the graph, so the trip count never reaches the compiler: `M=8` and
`M=32` produce byte-identical work -- 9 operations, 38 passes, indistinguishable times.
That family is in the plan for coverage, whether the frontend handles a scan HOP at all,
not for a scaling curve. Only `flash` (via `Lk / block_size` unrolled bodies) and `mlp`
(via `layers`) grow the graph.

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

## Adding a workload

1. Write a builder in `workloads.py` returning a `Workload`. Take sizes as keyword
   arguments -- the point of a family is that one parameter moves and the rest hold still.
2. Register it in `BUILDERS`.
3. Add points to `sweep_plan.json`.

Derive the body from a test that passes today and name that test in the docstring. The
bodies here are copies rather than imports: a baseline is only comparable if the workload
did not move, and test helpers move for test reasons.

## Adding a metric

1. Wrap the region in the compiler with `timing_recorder.stage("stage:<Owner>:<what>")`.
   Counts that describe a region belong in its `meta`, measured outside the region so
   counting is not charged to the work.
2. Nothing in the summarizer needs to change for a new `stage:` or `pass:` region -- both
   are collected by prefix. Add a column only for something that is neither.

## Workload constraints the backend imposes

Two shape rules bound what the `transformer_block` family can sweep. Both surface as
assertions deep in lowering, so they are checked in the builder instead:

- **`head_dim` must be a multiple of 64.** 64 fp16 elements is one stick, and a fractional
  head lands as `Unsupported coordinate expression 5*c0/2`. Ministral-style dims therefore
  sweep as `E=5120, heads=40` (`head_dim` 128), not `heads=32`.
- **A reshape that splits a named dim must re-annotate it.** Projecting `[B, S, E]` and
  reshaping into `[B, S, H, D]` otherwise fails above `S=512` with `layout dim 2 has 2 loop
  vars but only 1 name(s)`. The builder wraps the reshape and transpose in `spyre_hint`.

## What is not covered

**Forward only.** Nothing here compiles a backward graph, so passes that only run on
training graphs have no coverage.

**One block, not a stack.** Depth is swept on the `mlp` family instead, where it is the only
moving parameter. A real model's cost is not assumed to be depth times one block -- that is
a claim the baseline should be used to test, not to bake in.

## What the decomposition does not cover yet

`GraphLowering.run` and `GraphLowering.codegen` are not timed, so lowering and codegen
time currently lands in the self-time of `stage:compile_fx:spyre_compile`. Dynamo and
AOTAutograd are not separated either. Both are follow-ups; until then, read
`spyre_compile` self-time as "everything upstream of the Spyre pipelines" rather than as
overhead.
