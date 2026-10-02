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
| `51-functional-views.sql` | 4 `v_case_*` / `v_run_tier_counters` / `v_tier_report_completeness` views | 10, 20 |
| `52-cross-views.sql` | `v_run_coverage` | 10, 20 |
| `60-benchmark-views.sql` | 5 `v_benchmark_*` views | 20, 30 |
| `70-vllm-hud-projection.sql` | `oss_ci_benchmark_v3`, `oss_ci_benchmark_metadata` + their MVs | 30 |

The `50`/`51`/`52` split is by what a view reads, not by taste: the artifact and functional view
families are independent, and `v_run_coverage` is separate because it measures the join between
them (`artifact_results` to `test_case_runs`). The `51` tier views read `artifact_results` only
to learn which tier a run's leg was dispatched for.

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
`<lineage>/monitoring/clickhouse/clickhouse-schema`, which targets that folder's v2 database:
Spyre-Next first, then prod `Spyre` behind an approval. `--check` prints the pending changes and
exits 1 if there are any, 2 if a table or MV has drifted (a difference a pending migration
ALTERs is listed as `migrates`, not drift).

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
- A file marked `-- APPLY: explicit` applies only with `--include <file>`, for DDL that belongs to
  a different database. None is marked today.
- A migration marked `-- RERUNNABLE` is written to be repeated, and `--rerun <file>` runs it
  again after its ledger entry, e.g. `006_*` to re-key ids that writers on an older image still
  mint.
- Every fact and dimension table carries `audit_uuid` (UUIDv7) and `audit_timestamp`
  (DateTime64(3)), as every v1 table does: a stable per-row identity, and an insert time fine
  enough to order rows written within the same second. Exporter-owned (`otel_*`), upstream-shaped
  (`oss_ci_*`) and summing (`run_case_counters`) tables are left as their owners define them.

Mechanical constraints behind those rules:

- A `MergeTree` `ORDER BY` is fixed at creation, so adding a column to a sort key means a
  rebuild migration, not `ALTER`.
- `MODIFY COLUMN` cannot convert `Float64` to `Array(Float64)` (Code 53) — also a rebuild.
- The server rejects `CREATE OR REPLACE VIEW` (`renameat2() is not supported`), and dropping a
  base table silently drops its views.

## Sandboxes

User guide, MCP tool reference and security model: [`docs/sandbox.md`](../docs/sandbox.md).

`python -m spyre_clickhouse_ingest.sandbox` gives one person a database on the **dev** server to
change schema and queries freely. A sandbox never reaches a live database: the only way a change
lands in `spyre_v2` is a PR to this directory, applied by the Jenkins job above.

| command | what it does |
|---|---|
| `bootstrap` | Once per dev server: stores prod's read-only `spyre_user` login (`SANDBOX_SOURCE_PASS`) as the `prod_v2` named collection |
| `create --name <n>` | `sandbox_<n>` from `--schema-dir` (default: this directory), seeded from prod, plus a `sandbox_<n>_admin` login it prints once |
| `seed --name <n>` | Appends another sample; copies only the columns both sides have, so an edited table still seeds |
| `diff --name <n>` | What the sandbox has that `--schema-dir` does not -- the content a PR must carry; exit 1 if any |
| `drop --name <n>`, `list` | Remove one; list all |

`create`, `seed`, `drop` and `bootstrap` connect as the dev admin (`CLICKHOUSE_*`); `diff` runs
as the sandbox login. That login holds `ALL` on its own database, `SELECT` on dev `spyre_v2`,
and read of prod through `remote(prod_v2, table='<t>')` -- nothing else.

The sample is connected, not random: runs are picked first (`--days`, `--runs-per-component`,
`--component`, `--arch`, `--tag`, `--run-id`), then each table is cut to those runs and the
dimension rows they reference, so every join a view makes finds its rows. Dimensions are
inserted before their facts and MV targets are left to the MVs. A filter on a column a run
source lacks (`artifact_results` has no `component`) drops that source, not the filter. A new
base table needs a rule in `Sandbox.SEED`; `test_sandbox.py` fails until it has one.

For self-serve use, `python -m spyre_clickhouse_ingest.sandbox_server` (extra `[server]`)
wraps these operations in an HTTP MCP server that holds the dev admin login itself. A caller's
bearer token names them -- a line of the shared per-user token file, or a token the server
minted at `POST /register` for an OpenShift login (`oc whoami -t`) -- and every tool acts only
on sandboxes whose database comment records that owner. Queries run as the sandbox's own login,
whose password is derived from `SANDBOX_SECRET`; sandboxes expire after a TTL. `schema_diff` and
`verify_schema` take any `<owner>/torch-spyre@<ref>`, so a pushed fork branch is proved the
same way as below.

Turning an experiment into a PR: edit the `CREATE` here to the shape `diff` shows, add the
`migrations/NNN_*.sql` that gets a live table there, then prove both paths on dev:

```bash
sandbox diff --name <n> --schema-dir <branch>/schema       # only `migrate` lines left
sandbox create --name <n>_fresh --schema-dir <branch>/schema --no-seed
apply_schema --database sandbox_<n>_fresh --schema-dir <branch>/schema --check   # 0 pending
sandbox create --name <n>_up --days 3 --runs-per-component 5       # main's schema, real rows
apply_schema --database sandbox_<n>_up --schema-dir <branch>/schema   # migrations run on data
apply_schema --database sandbox_<n>_up --schema-dir <branch>/schema --check      # 0 pending
```

Several comments in these files cite row counts and percentages measured when the statement was
written. They are evidence for a design decision, not live figures; re-measure before relying on
one.
