# spyre-clickhouse-ingest

The schema-v2 ClickHouse schema, derived identity and write path, shared by the Spyre CI ingests.

## Why it is a library

Every id here is **derived, never minted**: the product ingests and the Jenkins-side writer must
reach the same uuid for the same run without coordinating. A second copy that drifts by one
normalisation step produces ids that silently never join — no error, just missing data.

## Install

No PyPI or Artifactory publish. Every consumer installs it straight from the repo.

Inside torch-spyre, install from the CHECKOUT, so the library is always the same commit as the
script importing it:

```
uv pip install "${GITHUB_WORKSPACE}/extensions/clickhouse-ingest"
```

From another repo, where that path does not exist, install from git at `@main`:

```
uv run --no-project \
  --with "git+https://github.com/torch-spyre/torch-spyre@main#subdirectory=extensions/clickhouse-ingest" \
  ...
```

Verified on a build node with the same `uv run --no-project --with` form the baked-image ingest
uses.

`@main` rather than a tag, deliberately: this library's whole purpose is that ONE definition of
the derived ids runs everywhere. A consumer pinned to an older tag is a second definition again --
it just fails later and less visibly than a copied file. The identity functions are covered by
golden-value tests (`tests/test_identity_golden.py`), so `@main` moving is not supposed to be able
to change an id; if it ever does, those tests are the thing that must stop it.

## Layout

| module | contents |
|---|---|
| `schema.py` | the table model: columns, order, DDL CHECK sets, `qualified()`, dep-entry helpers |
| `identity.py` | `run_id_of`, `case_id_for`, `component_of`, `canonical_arch`, `gha_artifact_id` |
| `client.py` | `get_client`, `target_database`, `tables_present` |
| `writer.py` | `insert_test_results`, `cases_already_ingested`, `insert_gha_artifact_result` |
| `junit.py` | JUnit helpers + CI run-coordinate resolution |
| `hw_parse.py` | GHA log → `hw_failure_diagnostics` records (RAS events, phases, pytest counts) |
| `hw_schema.py` | `hw_failure_diagnostics` columns + its `ADD COLUMN IF NOT EXISTS` migration |
| `hw_diagnostics.py` | `build_row`/`insert_rows` for `hw_failure_diagnostics` |
| `gha_logs.py` | fetching GHA job logs via `gh`, with transient-5xx retry |

## Tables modelled

`test_cases`, `test_case_runs`, `benchmarks`, `benchmark_runs` (DDL: `functional_tests_v2.sql`)
and `artifacts`, `artifact_refs`, `artifact_tags`, `artifact_results` (DDL: `artifacts_v2.sql`).
The DDL itself is applied by the CI pipeline that owns the warehouse, not from this repo.

The model holds columns, order and the DDL's CHECK sets — not the DDL itself. `TABLES` is pinned
as an exact set by `tests/test_schema.py`, so adding a table to the DDL without modelling it here
fails rather than drifting. Column order was verified against the live `spyre_v2` tables when the
artifact four were added.

It also states the **dep-entry shape**, which is the one contract a reader cannot infer:
`artifacts.identity_deps` / `context_deps` entries are `"<component>@<id12>"` (or `base=<sha>`,
or a bare name), *not* uuids — `id12` is a hash input to `artifact_id`, so the id cannot be
recovered from the string. Use `dep_component()` / `dep_id12()` and resolve via `props['id12']`.
A dashboard route that assumed uuids matched zero rows and rendered nothing, with no error.

## The identity of what a GHA leg ran

A GHA leg installs this PR's build on top of a prebaked image, so what it ran is a different
artifact from the image Jenkins published. It had no identity, and the writer refuses --
correctly -- to record a verdict against an `artifact_id` it cannot derive, so GHA-native legs
wrote no `artifact_results` row: 7,189 of 8,031 measured legs, against 842 that reported.

Deriving one needs two halves no single machine holds: the base image's own `artifact_id`,
stamped into the image by spyre-frameworks' `_package-image` (#1782) as the
`spyre.artifact.id` label and as `/home/senuser/spyre_artifact_id.txt`; and the delta this leg
installed, known only to the workflow that installed it.

```text
_package-image stamps the image    ->  /home/senuser/spyre_artifact_id.txt
  derive-gha-artifact-id (runner)  ->  gha_artifact_id(base, installed) -> "artifact-id" artifact
    push-to-clickhouse (ingest)    ->  --artifact-id
      insert_gha_artifact_result   ->  artifacts + artifact_results
```

`base_artifact_id()` reads the FILE, not the label: a test leg runs inside the container,
where reading its own label would mean an outbound registry inspect with credentials it does
not have. The derivation must run on the runner for the same reason, so the id travels as an
uploaded artifact -- the channel the threaded `run_id` already uses, because a `workflow_run`
consumer sees none of the producing run's inputs or outputs.

Every step degrades to empty rather than failing: a pre-#1782 image carries no id, and a test
run must never go red over telemetry.

## What is deliberately NOT here

- **v1 write paths whose TARGET TABLE differs per repo.** `hf_test_runs` vs `test_runs` cannot share
  a writer, so those stay per-repo until v1 is retired. Name divergence is the blocker, not the v1
  generation as such: `hw_failure_diagnostics` is the same table with the same columns in every repo,
  which is why its parse/ingest lives here despite being v1.
- **A dependency on `torch_spyre`.** Installing that to obtain a schema module would pull
  torch/numpy/ortools, and ortools has no ppc64le/s390x wheel — the ingest would break on p/z.

## Changing an identity function

Don't, without a migration. Every id ever written derives from these functions and the namespace
`cb0af9bf-2858-5eab-9211-f51190531bf3`. `tests/test_identity_golden.py` pins the contract with
golden values; if a change makes those fail, it invalidates historical rows.
