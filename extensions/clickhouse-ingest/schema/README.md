# v2 schema DDL

The ClickHouse DDL for the v2 tables this package writes and reads.

## Apply order

Files are numbered because the dependencies are real: a view cannot be created before the table
it selects from, and a materialized view must exist before the rows it should see are inserted.
Applying them in filename order works from an empty database.

| file | declares | depends on |
|---|---|---|
| `10-functional-tests.sql` | `test_cases`, `test_case_runs` | — |
| `20-artifacts.sql` | `artifacts`, `artifact_refs`, `artifact_tags`, `artifact_results` | — |
| `30-benchmarks.sql` | `benchmarks`, `benchmark_runs` | — |
| `40-jenkins-agents.sql` | `jenkins_agents` | — |
| `50-artifact-views.sql` | 6 `v_tag_*` / `v_artifact_*` / `v_tier_trend` views | 10, 20 |
| `51-functional-views.sql` | 4 `v_case_*` / `v_run_tier_counters` / `v_tier_report_completeness` views | 10 |
| `52-cross-views.sql` | `v_run_coverage` | 10, 20 |
| `60-benchmark-views.sql` | 5 `v_benchmark_*` views | 20, 30 |
| `70-vllm-hud-projection.sql` | `oss_ci_benchmark_v3`, `oss_ci_benchmark_metadata` + their MVs | 30 |

The `50`/`51`/`52` split is by what a view reads, not by taste: the artifact and functional view
families are independent, and `v_run_coverage` is separate because it is the one view spanning
both (it joins `artifact_results` to `test_case_runs`).

`70-` is named for what it is — a projection of `benchmark_runs` into the shape the PyTorch HUD
reads — rather than for a table, because the tables it declares carry upstream's names
(`oss_ci_benchmark_*`), not ours.

## Why it lives here

`schema.py` models these tables as data — columns, order, CHECK-constraint vocabularies — and
every row this package inserts is ordered through that model. Until now the DDL itself lived in
another repo (`spyre-frameworks/pipelines/clickhouse/`), so the *shape* was declared in one place
and *modelled* here, with nothing but prose comments tying them together.

That split is what let prod drift: `spyre_v2.benchmarks` was missing the `component` column
`schema.py` requires and that leads the `benchmark_id` hash, and nothing failed — both tables
were 0 rows, so the gap only surfaced when a writer was finally pointed at them. Co-locating the
DDL with the model means a column added to one is reviewed beside the other.

## Applying it

**Nothing applies these files automatically.** There is no migration runner and no
`schema_migrations` ledger for the v2 databases — every statement here has been applied by hand.
So before writing to a v2 table, `DESCRIBE TABLE` it on the target server rather than trusting
this directory or `schema.py`.

Four mechanical gotchas, all hit while applying these:

- A `MergeTree` `ORDER BY` is fixed at creation, so adding a column to a sort key means
  DROP+CREATE, not `ALTER`.
- `MODIFY COLUMN` cannot convert `Float64` to `Array(Float64)` (Code 53, "same-dimensional Array,
  Map or String types") — that change is also a DROP+CREATE, or a rewrite through a temp table on
  a populated one.
- Views must be dropped and recreated, not `CREATE OR REPLACE`d (the server rejects it:
  `renameat2() is not supported`). Dropping a base table silently drops its views, so recreate
  them explicitly, base view first — `v_benchmark_results_enriched` before the four that select
  from it.
- Materialized views fire **on insert only**. They cannot be backfilled from rows already in
  their source table, so a definition change means re-inserting, and the MV must exist before
  the data lands. Upstream ships a backfill `INSERT` beside its own MV for this reason.

Several comments in these files cite row counts and percentages measured when the statement was
written. They are evidence for a design decision, not live figures; re-measure before relying on
one.

## State

On the **prod** server, `spyre_v2` matches these files for every table except the HUD
projection, verified column-for-column (including view signatures for the benchmark views):

- The `oss_ci_benchmark_*` pair in `70-` **does not exist in prod** — that file is the intended
  shape, not a deployed one. It was verified on the dev server: applying it and inserting one
  `benchmark_runs` row propagated through both materialized views, and a live HUD read the result
  with upstream's own queries unmodified.
- Staging `spyre_v2_next` has `component` but **not** the widened `measurements` or the `samples`
  column, so staging and prod differ on the benchmark pair.
