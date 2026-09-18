# spyre-clickhouse-ingest

The schema-v2 ClickHouse schema, derived identity and write path, shared by the Spyre CI ingests.

## Why it is a library

Every id here is **derived, never minted**: the product ingests and the Jenkins-side writer must
reach the same uuid for the same run without coordinating. A second copy that drifts by one
normalisation step produces ids that silently never join — no error, just missing data.

## Install

No PyPI or Artifactory publish. Every consumer installs it straight from the repo, pinned:

```
uv run --no-project \
  --with "git+https://github.com/torch-spyre/torch-spyre@<tag>#subdirectory=extensions/clickhouse-ingest" \
  ...
```

Verified on a build node with the same `uv run --no-project --with` form the baked-image ingest
uses. Pin a tag, not `@main`.

## Layout

| module | contents |
|---|---|
| `schema.py` | the table model: columns, order, DDL CHECK sets, `qualified()`, dep-entry helpers |
| `identity.py` | `v2_run_id`, `v2_test_case_id`, `v2_component`, `v2_canonical_arch` |
| `client.py` | `get_client`, `v2_database`, `v2_tables_present` |
| `v2_writer.py` | `insert_v2`, `v2_already_ingested` |
| `junit.py` | JUnit helpers + CI run-coordinate resolution |

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

## What is deliberately NOT here

- **v1 write paths.** The product repos write v1 to differently-named tables (`hf_test_runs` vs
  `test_runs`), so those stay per-repo until v1 is retired.
- **A dependency on `torch_spyre`.** Installing that to obtain a schema module would pull
  torch/numpy/ortools, and ortools has no ppc64le/s390x wheel — the ingest would break on p/z.

## Changing an identity function

Don't, without a migration. Every id ever written derives from these functions and the namespace
`cb0af9bf-2858-5eab-9211-f51190531bf3`. `tests/test_identity_golden.py` pins the contract with
golden values; if a change makes those fail, it invalidates historical rows.
