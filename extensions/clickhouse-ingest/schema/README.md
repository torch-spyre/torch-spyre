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

`python -m spyre_clickhouse_ingest.apply_schema --database <db>` converges a database on these
files. The `clickhouse-schema` workflow proves every PR against an empty server (apply twice; the
second pass must change nothing). Live databases are applied by the spyre-frameworks Jenkins job
`Spyre/ops/clickhouse-schema`: staging `spyre_v2_next` first, then prod `spyre_v2` behind an
approval. `--check` prints the pending changes and exits 1 if there are any.

What an apply does, in order:

1. Creates any missing table or materialized view.
2. Runs each `migrations/NNN_*.sql` not yet in the database's `schema_migrations` ledger, once.
3. Compares every existing table and MV with its file, in the server's own formatting. A
   difference **fails the run** — it never ALTERs. Change a live table with a migration, then
   update its `CREATE` here to the resulting shape.
4. Creates missing views and drops+recreates changed ones (a plain view holds no data).

Rules this implies:

- `schema/*.sql` holds only `CREATE` statements; an `ALTER`, `INSERT` or backfill goes in
  `migrations/`.
- A new MV needs a backfill migration for the rows already in its source (MVs fire on insert
  only); cut off at the MV's own `metadata_modification_time` so no row is counted twice — see
  `migrations/002_*`.
- A file marked `-- APPLY: explicit` (`80-otel.sql`, which lives in the v1 `spyre` database)
  applies only with `--include <file>`.

Mechanical constraints behind those rules:

- A `MergeTree` `ORDER BY` is fixed at creation, so adding a column to a sort key means a
  rebuild migration, not `ALTER`.
- `MODIFY COLUMN` cannot convert `Float64` to `Array(Float64)` (Code 53) — also a rebuild.
- The server rejects `CREATE OR REPLACE VIEW` (`renameat2() is not supported`), and dropping a
  base table silently drops its views.

Several comments in these files cite row counts and percentages measured when the statement was
written. They are evidence for a design decision, not live figures; re-measure before relying on
one.
