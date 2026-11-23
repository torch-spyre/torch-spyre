# Trace values: per-step and per-span views (schema 87, 90)

A multi-rank decode trace is stored as values in `benchmark_runs` (no new base table; schema 90 optionally adds a copy of the unnested spans): one row per (rank, prompt, repeat) of a
`step_trace` benchmark holds aligned arrays (`step_index`, `step_kind`, `cold`, one array per signal) plus `summary.<kind>.<signal>`
and `headline.<kind>.<signal>`; one row per (span pattern, rank, prompt, repeat) of a `span` benchmark holds `step, seq, parent_seq,
depth, layer, start_ns, dur_ns`. Both record types are in `benchmark_excluded_record_types`, so trend, compare and verdict views skip them.

| View | One row per | Use |
|---|---|---|
| `v_bench_step_records` | rank x step | the unnested arrays; `metrics` map, the four role columns, rules |
| `v_bench_step_rank_agg` | step | cross-rank value of each role (total max, others mean unless `agg.<signal>` says otherwise), slowest / fastest / awaited rank |
| `v_bench_step_lag` | rank x step | total minus the fastest rank, wait spread, awaited rank |
| `v_bench_step_headline` | (run, benchmark, repeat, prompt, kind) | median of the cross-rank total over warm steps, rank medians, skew; NULL without a warm step |
| `v_bench_step_outliers` | slow step | modified z > 3.5 per rank and kind, excess split wait / compute / host |
| `v_bench_rank_coverage` | (run, benchmark, repeat, prompt) | ranks present against `run.world_size` |
| `v_bench_spans` | span | the span arrays unnested with lane, name and case id |
| `v_bench_spans_fast` (90) | span | the same rows from `bench_span_values` |

Signals are named by run props (`role.total|wait|compute|busy`, `agg.<signal>`); nothing in the views names a signal. Always filter by
`run_id`: the filter reaches the aggregation under each view. Invalidated repeats are dropped.

## Writing the rows

`python -m spyre_clickhouse_ingest.trace_bundle trace.json --out DIR --artifact-id UUID --arch ARCH` turns a neutral trace description
into a `benchmarks` bundle, one results file per (rank, prompt, repeat) (the writer folds rows of one benchmark within a file), then
`results --from-bundle DIR`. The description format is in the module docstring; `tests/trace_fixture.py` is a worked example.

## The optional span table (90)

A sandbox seeded from prod fills the table by itself: seeding is an INSERT SELECT into `benchmark_runs`, which fires the MV. `mv_bench_span_values` copies the unnested spans of every `benchmark_runs` row inserted after it exists. Create it before the first span
row; for rows already present run once
`INSERT INTO bench_span_values SELECT <the SELECT of mv_bench_span_values, with FROM benchmark_runs AS r>` (the table collapses re-inserts
by newest `audit_timestamp`). Without the file nothing breaks: `v_bench_spans` reads the same data from `benchmark_runs`.
