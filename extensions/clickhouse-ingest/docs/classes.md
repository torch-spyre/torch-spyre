# Class reference

[← Back to index](README.md)

Every class below lives in `spyre_clickhouse_ingest/`, is re-exported from the
package `__init__.py`, and is also available as a bare function (`writer.py`'s
`TestResultWriter.insert` is also `insert_test_results`, etc.) — "kept so
installed consumers import one definition, not a copy," per the comment at the
bottom of every module. Use whichever form the call site reads better; they
are the same code.

- [identity.py — derived, never-minted uuid5 identities](#identitypy)
- [schema.py — the v2 table model](#schemapy)
- [client.py — connection + write gate](#clientpy)
- [writer.py — one writer per table pair](#writerpy)
- [junit.py — JUnit + run-coordinate helpers](#junitpy)
- [hw_parse.py / hw_schema.py / hw_diagnostics.py — hardware diagnostics](#hardware-diagnostics-hw_parsepy--hw_schemapy--hw_diagnosticspy)
- [gha_logs.py — GHA job-log fetch](#gha_logspy)
- [apply_schema.py — DDL apply CLI](#apply_schemapy)

---

## identity.py

Every id is a `uuid5` under one fixed namespace
(`cb0af9bf-2858-5eab-9211-f51190531bf3`), hashed from normalised, pipe-joined
parts. **Two independent processes — a GHA runner and a Jenkins agent — must
reach the same id for the same (component, name, arch, …) without ever
talking to each other**, which is the whole reason this module exists instead
of being reimplemented per writer.

### `DerivedId` — base class

| Method | Use case |
|---|---|
| `norm(value)` | Canonical scalar form: stripped and lowercased. Called before hashing *any* field, so `"Torch-Spyre "` and `"torch-spyre"` collide on purpose. |
| `arch(value)` | Folds `amd64`/`x86`/`x86-64`/`x86_64` to one spelling — `x86_64` — so a leg reported either way hashes identically. |
| `hash(*parts)` | `uuid5` of the parts joined by `\|`. The one line every `derive()` below eventually calls. |
| `complete(*values)` | True only when every required field is non-blank. A writer calls this *before* hashing so a blank field refuses the id instead of hashing a hole that silently collides with every other blank. |
| `tag_part(tags)` | Tags as a deduped, sorted, comma-joined string — a **set**, not a sequence, so tag order never changes an id. |
| `disc_part(disc, disc_keys)` | Per-producer discriminators, emitted in `disc_keys` order — the escape hatch a producer uses to fold extra columns into an id without editing this module. |

### `RunId(DerivedId)` — identity of one CI leg

```python
RunId.derive(source, external_run_id, arch, test_type)   # -> uuid, or "" if incomplete
RunId.for_args(args, run_id, arch, tier)                 # threaded --run-id, else derive()
```

**Use case.** Every ingest CLI in this doc calls `run_id_for` (the bare-function
alias) exactly once, early, to get the `run_id` every other row in that call
will carry. `for_args` prefers a *threaded* run id — the orchestrator-minted
uuid an upstream job uploaded as an artifact — over deriving one, because a
threaded id is what lets `test_case_runs` join `artifact_results` for the same
run; falling back to `derive()` only happens on a GHA-only leg with no
orchestrator.

### `CaseId(DerivedId)` — content identity of a test

```python
CaseId.derive(component, classname, name, tags)   # -> uuid, or "" without component/name
CaseId.tags_for(case)                              # JUnit <properties> -> tags array
```

**Use case.** `TestResultWriter.insert` (see [writer.py](#writerpy)) calls
this once per JUnit `<testcase>` so the *same test* — same component,
classname, name, tags — reconciles across every run that ever exercised it,
regardless of which CI system or architecture ran it this time.

### `ArtifactId(DerivedId)` / `GhaArtifactId(ArtifactId)` — identity of a built thing

```python
ArtifactId.derive(component, artifact_name, id12, arch)     # -> uuid, needs component+arch
ArtifactId.from_image(path="")                               # reads the stamped base id
GhaArtifactId.derive(component, base_artifact_id, installed, arch)
GhaArtifactId.installed_digest(installed)                    # sha256[:12] of the sorted delta
```

**Use case.** A GHA test leg installs a PR's build on top of a prebaked image,
so what it *ran* is a different artifact from the image Jenkins published.
`derive_artifact_id.py` (torch-spyre) calls `GhaArtifactId.derive` to hash
only the installed delta onto the base image's own id (read via
`ArtifactId.from_image` from `/home/senuser/spyre_artifact_id.txt`), so the
leg's true artifact identity is derivable without a second machine's
involvement. See [Jenkins shared library](jenkins-shared-library.md) for the
build-side counterpart that *writes* `artifacts` rows for what Jenkins
promotes.

### `CapabilityId(DerivedId)` / `BenchmarkId(DerivedId)`

```python
CapabilityId.derive(component, test_type, subject, name, disc, disc_keys)
BenchmarkId.derive(component, name, tags, disc, disc_keys)
```

Both deliberately leave `backend` **out** of the hash — it's the axis a
capability verdict or a benchmark measurement gets *compared across*, not part
of what identifies the subject.

**Use cases.**
- `ingest_model_ops.py` (torch-spyre) derives `CapabilityId`s for
  `test_type="model_ops"` — "does this build support op X".
- `capability_write.py` (hf-adapters) derives them for
  `test_type="model_support"` — "does this Hub checkpoint run on backend Y" —
  the two share one table pair (`capabilities`/`capability_runs`) and are told
  apart purely by `test_type`, a sibling vocabulary, not a subtype.
- `ingest_vllm_benchmarks.py` (spyre-inference) derives `BenchmarkId`s per
  vLLM benchmark name/tag/discriminator combination.

### `Component`

```python
Component.of(args, default=COMPONENT_DEFAULT)   # --component, else a per-repo default
```

The value every id above folds in — get this wrong and a test keeps its
identity but is silently attributed to the wrong product.

---

## schema.py

One class per v2 table, all sharing `Table`. This is **data, not DDL** — the
actual `CREATE TABLE` statements live in [`schema/*.sql`](../schema/README.md)
(canonically owned by `spyre-frameworks/pipelines/clickhouse/`, mirrored here
— see [Jenkins shared library](jenkins-shared-library.md)); `schema.py` is
what every row gets validated and column-ordered through before it reaches
the wire.

### `Table` — base for every v2 table

| Method | Use case |
|---|---|
| `row(values)` | Orders one row dict by `columns`; raises `SchemaError` on an unknown, missing, or CHECK-violating value — the thing that turns a typo'd column name into a loud Python exception instead of a silent ClickHouse insert failure three layers down. |
| `qualified(db)` | `db.table` when a database is given, bare table name otherwise — what makes `CLICKHOUSE_DB_V2` unset degrade cleanly to "v2 disabled" rather than an error. |
| `insert(client, rows, db)` | Bulk-inserts dicts, ordering every row through the one column list. |
| `insert_identities(client, rows, db)` | Inserts only the identity rows the dimension table doesn't already hold — a `SELECT ... WHERE id IN (...)` before the insert, so re-ingesting a run never doubles a `test_cases` row. |
| `present(client, db, check_columns)` | True when the table exists **and** holds at least the modelled columns. This is the v2 write gate every composite action calls before it writes a single row. |
| `count_rows(client, db, where, params)` | `count()` under a parameterised `WHERE` — what every writer's `already_ingested` check is built from. |

### `DepEntry` — parses `artifacts.identity_deps` / `context_deps`

A dep entry is `"<component>@<id12>"` (or `base=<sha>`, or a bare name) — **not
a uuid**, because `id12` is a hash *input* to `artifact_id`, not the id
itself.

```python
DepEntry.id12(entry)        # the id12 the entry names, or "" if it names none
DepEntry.component(entry)   # the component the entry names, without its pin
```

> A dashboard route that assumed uuids matched zero rows and rendered nothing,
> with no error. This is the one contract a reader cannot infer from the
> column alone — resolve an `id12` via `props['id12']` on the row it names,
> never by treating the string as an id you can look up directly.

### The ten concrete tables

| Class | Table | Identity column | Use case |
|---|---|---|---|
| `TestCases` | `test_cases` | `test_case_id` | Dimension row per unique test. |
| `TestCaseRuns` | `test_case_runs` | — (fact) | One test's outcome in one run. |
| `Benchmarks` | `benchmarks` | `benchmark_id` | Dimension row per unique benchmark. |
| `BenchmarkRuns` | `benchmark_runs` | — (fact) | One benchmark's measurements in one run, per backend. |
| `Capabilities` | `capabilities` | `capability_id` | Dimension row per (subject, capability). |
| `CapabilityRuns` | `capability_runs` | — (fact) | One capability's verdict in one run, per backend. |
| `Artifacts` | `artifacts` | *none* — dup is a producer bug | One built/promoted thing. Primarily Jenkins-written; see [Jenkins shared library](jenkins-shared-library.md). |
| `ArtifactRefs` | `artifact_refs` | — | How a consumer obtains an artifact. |
| `ArtifactTags` | `artifact_tags` | — | What a channel tag pointed to as of `ts`. |
| `ArtifactResults` | `artifact_results` | — (fact) | One leg's verdict on one artifact. |

Every table is also exposed as an `UPPER_CASE` constant
(`TEST_CASES`, `ARTIFACT_RESULTS`, …) and via the `TABLES` dict keyed by name
— pinned exact by `tests/test_schema.py`, so adding a table to the DDL
without modelling it here fails the test suite instead of drifting silently.

---

## client.py

One connection factory, so every ingest — Python CLI or Jenkins-invoked
script alike — resolves `CLICKHOUSE_HOST`/`PORT`/`USER`/`PASS`/`DB` through
identical rules.

```python
get_client(verify=True)              # ClickHouse.connect — fails fast with a clear SystemExit
client_summary()                     # "host:port/database" for logging, never the password
target_database()                    # CLICKHOUSE_DB_V2, or "" when v2 isn't configured
tables_present(client, db, tables)   # the v2 write gate, per table set
```

**Use case.** Every composite action gates its own v2 write on
`clickhouse-host` being non-empty *before* even installing the library — but
`tables_present` is the second gate, inside the script, that lets a database
with v1 tables only (no v2 yet) skip the v2 insert cleanly rather than
crashing on a missing table.

---

## writer.py

One writer class per (identity, fact) table pair, all sharing `RunWriter`'s
dedup-and-flush pattern: check what's already landed, write only the new
identities, then the facts.

### `RunWriter` — base

`_seen(client, db, run_id, component, scopes)` (dedup check, scoped by extra
columns like `source_file` or `report_kind`), `_flush(client, db, ident_rows,
fact_rows)` (writes identities then facts, returns the count), `_warn(count,
message)` (stderr, never fatal — a parse gap must stay visible, not fail the
run).

### `TestResultWriter`

```python
TestResultWriter.already_ingested(client, db, run_id, component, source_file="")
TestResultWriter.insert(client, db, component, run_id, cases, source_file="")
```

**Use case.** `ingest_xml.py`, `ingest_xml_si.py` and
`ingest_xml_hf_adapters.py` all call this for the same reason: turn a parsed
JUnit case list into `test_cases`/`test_case_runs` rows. A case whose identity
can't be derived (no name) is skipped with a warning rather than colliding
into every other unidentifiable case.

### `BenchmarkWriter`

```python
BenchmarkWriter.already_ingested(client, db, run_id, component, report_kind="", source_file="")
BenchmarkWriter.insert(client, db, component, run_id, benchmarks, report_kind="", source_file="")
```

**Use case.** `ingest_vllm_benchmarks.py` (spyre-inference) and the perf leg
of torch-spyre's own `ingest_xml.py` both call this — one row per (benchmark,
backend), samples extended across repeated entries, and any run row with zero
measurements dropped (the DDL's `CHECK` would otherwise fail the *whole*
insert for one bad benchmark).

### `CapabilityWriter`

```python
CapabilityWriter.already_ingested(client, db, run_id, component, test_type="", shard="")
CapabilityWriter.insert(client, db, component, run_id, test_type, results, arch="", disc_keys=(), shard="")
```

**Use case.** Two genuinely different producers share this one writer:
`ingest_model_ops.py` (torch-spyre, `test_type="model_ops"`) and
`capability_write.py` (hf-adapters, `test_type="model_support"`). Both pass a
`shard` when the caller fans out over parallel workers — scoping the dedup
check per shard is what stops the *first* shard to flush from making every
other shard look already-ingested.

### `ArtifactWriter`

```python
ArtifactWriter.artifact_recorded(client, db, artifact_id)
ArtifactWriter.result_recorded(client, db, artifact_id, run_id, result_kind, test_type)
ArtifactWriter.insert_gha_result(client, db, *, artifact_id, component, arch, run_id, test_type, state, ...)
```

**Use case.** `ingest_xml.py`, given a non-empty `--artifact-id` from
`derive-gha-artifact-id`, records both the artifact a GHA leg ran *and* its
verdict as one call — refusing on a partial id (missing component/arch/etc.)
rather than writing a half-identified row that a dashboard join would never
find. The Jenkins side writes `artifacts` rows for build/promotion instead of
test verdicts — see [Jenkins shared library](jenkins-shared-library.md).

---

## junit.py

### `JUnitXml`

`extract_properties(tc_el)` — a `<testcase>`'s `(name, value)` property pairs;
`promote_xpass(raw_cases, suite_attrs)` — relabels bare cases as `xpass` for a
suite's non-strict xpass failures (pytest's own summary counts them as
failures, but the per-case status doesn't say which cases).

### `RunCoordinates`

```python
RunCoordinates.threaded_run_id(args)          # --run-id if a real UUID, else ""
RunCoordinates.gha_run_id(args)               # --gha-run-id if numeric, else ""
RunCoordinates.runner_run_id(args, run_id)    # this leg's own id
RunCoordinates.source_and_external(args, run_id)   # (source, external_run_id)
```

**Use case.** `source_and_external` is what makes `RunId.derive`'s `source`
argument one of exactly three values: `"gha"` (numeric `--gha-run-id`),
`"jenkins"` (a non-empty `--jenkins-run-key`, format `folder/job#123` — see
[Jenkins shared library](jenkins-shared-library.md) for where that string is
built), or `"local"` (neither — a manual re-ingest, joinable within itself but
not to any external run).

---

## Hardware diagnostics — hw_parse.py / hw_schema.py / hw_diagnostics.py

The pipeline that turns raw GHA job logs into `hw_failure_diagnostics` rows.
This table is deliberately **v1-generation everywhere**: unlike the other v1
write paths (which stay per-repo until retired), it has the identical shape
in every consuming repo, so it's modelled here instead of copy-pasted three
times.

### hw_parse.py

| Class | Use case |
|---|---|
| `RasClassifier` | `name_to_reason(name)` — maps a RAS event name (`RAS::CBRB::ResponseTimeout`) to a `failure_reason` label. Extend `_RAS_NAME_TO_REASON` here when a new hardware fault signature shows up in a log. |
| `LogText` | `clean(s)` strips ANSI escapes/control chars; `first_env(pattern, text)` reads a cleaned env-var value out of raw GHA log text. |
| `Timestamps` | `parse(line)` — GHA's ISO-8601 prefix first, the runtime's own `DTLOG` stamp as fallback. |
| `PytestSummary` | `summary_line(chunk_lines)` finds pytest's *own* terminal summary line (not a per-file subtotal or a hardware "3 errors drained" chatter line); `first_int` reads a count out of it. |
| `CrashDetector` | `detect(chunk_lines, chunk)` — signal aborts, segfaults, heap corruption, plus up to 10 backtrace frames. |
| `RasEvents` | `extract_all(chunk_lines)` — every RAS event in a log chunk, in order, double-logged blobs dropped. |
| `LogParser` | `parse(text, run_id, suite_hint, is_pod_level_retry)` — the core: slices a multi-attempt log into one record per (suite, attempt) with outcome, failure reason/phase, hardware IDs and pytest stats. |
| `SuiteFilenames` | `from_filename`/`pick_from_dir` — turns GHA log filenames into suite names, skipping CI-infra jobs. |

**Use case.** `parse_hw_failures.py` (torch-spyre) calls `LogParser.parse`
directly over a directory of downloaded job logs; its output JSON is exactly
what `hw_diagnostics.py`'s `RowBuilder` consumes next.

### hw_schema.py

```python
HwFailureDiagnostics.ensure_extra_columns(client, table="")   # ALTER ... ADD COLUMN IF NOT EXISTS
HwFailureDiagnostics.already_ingested(client, run_id, component, table="")
```

**Use case.** `ensure_extra_columns` is this table's *entire* migration
mechanism — it runs on every ingest, unconditionally, so a column added here
propagates to a live deployment the next time anything writes to it, with no
separate migration step to remember.

### hw_diagnostics.py

`RunContext` (frozen dataclass: `run_id`, `artifact_id`, `component`, `arch`,
`external_run_id`, `run_url` — the coordinates a parsed record doesn't carry
itself), `Fields` (scalar coercions: `ts`, `text`, `number`, `detail_json` —
honest defaults over silent nulls), `RowBuilder.build(rec, ctx)` (one parsed
record + its context → the ordered row), `RecordLoader` (`load`,
`filter_suites`, `insert` — the last with explicit column names so a
row/column-count mismatch raises instead of silently misaligning).

**Use case.** `ingest_hw_diagnostics.py` (torch-spyre) is the CLI wrapping
this half of the pipeline; the `ingest-hw-diagnostics-to-clickhouse` composite
action is what calls it end-to-end from a workflow — see
[GitHub Actions](github-actions.md).

---

## gha_logs.py

Fetching GHA job logs through the `gh` CLI — stdlib + `gh` only, **no
`clickhouse-connect` dependency**, which is why `derive_artifact_id.py` can
load just this half of the package without installing the database driver at
all.

```python
GhCli.run_with_retry(args, max_attempts=5, base_delay=2)   # retries transient 502/503/504
JobLogs.dedupe(jobs)
JobLogs.load(path)                 # from a JSONL job listing
JobLogs.download(jobs, repo, out_dir)
JobLogs.download_or_die(jobs, repo, out_dir)   # fatal iff a non-empty list downloads zero
```

**Use case.** Both `ingest-hw-diagnostics-to-clickhouse` and
`ingest-model-ops-to-clickhouse` composite actions call
`download_job_logs_or_die` to pull every relevant job's raw log before
parsing it.

---

## apply_schema.py

The CLI that turns [`schema/*.sql`](../schema/README.md) into a live
database — the only thing in *this* repo that actually creates the tables
`schema.py` models (the canonical DDL source is
`spyre-frameworks/pipelines/clickhouse/`; see
[Jenkins shared library](jenkins-shared-library.md)).

```python
SchemaApplier.schema_dir()                          # sibling schema/, else the installed copy
SchemaApplier.sql_files(schema_dir)                  # apply order = filename order
SchemaApplier.statements(text)                       # one file -> executable statements
SchemaApplier.required_version(text)                 # "-- NEEDS CLICKHOUSE >= X.Y" floor, if any
SchemaApplier.apply_file(client, path, dry_run, text)
SchemaApplier.apply_all(client, schema_dir, dry_run)  # skips files whose version floor is unmet
```

```bash
python3 -m spyre_clickhouse_ingest.apply_schema --dry-run
python3 -m spyre_clickhouse_ingest.apply_schema   # applies against $CLICKHOUSE_HOST
```

> **Nothing runs this automatically.** There is no migration runner and no
> `schema_migrations` ledger for the v2 databases — every statement has been
> applied by hand. Before writing to a v2 table, `DESCRIBE TABLE` it on the
> target server rather than trusting this directory or `schema.py`.

[← Back to index](README.md) · [Next: GitHub Actions →](github-actions.md)
