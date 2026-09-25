# Getting started

[← Back to index](README.md)

Three separate questions, answered separately: installing the library for
local development, reusing it from a new GitHub Actions workflow, and reusing
it from a new Jenkins pipeline. Most people only need one of the three.

- [Install from source](#install-from-source)
- [Reuse it as a GitHub Actions workflow](#reuse-it-as-a-github-actions-workflow)
- [Reuse it as a Jenkins shared library](#reuse-it-as-a-jenkins-shared-library)
- [Running the test suite](#running-the-test-suite)

---

## Install from source

There is **no PyPI or Artifactory publish**. Every consumer installs straight
from a git checkout, and *which* checkout depends on where you're standing.

### Inside torch-spyre (developing the library itself, or a torch-spyre CI script)

Install from the local checkout, so the library is always exactly the commit
the importing script is running against:

```bash
uv pip install "${GITHUB_WORKSPACE}/extensions/clickhouse-ingest"
# or, outside CI, from the repo root:
uv pip install ./extensions/clickhouse-ingest
```

### From another repo (spyre-inference, hf-adapters, a new consumer)

Install from git at `@main` — **deliberately a moving ref, not a pinned
tag**. The whole point of this package is that one definition of the derived
ids runs everywhere; pinning a tag re-creates the "second copy that drifts"
problem this library exists to prevent, just with extra steps. The identity
functions are covered by golden-value tests
(`tests/test_identity_golden.py`) specifically so that `@main` moving is not
supposed to be able to change an id without those tests failing first.

```bash
uv pip install "git+https://github.com/torch-spyre/torch-spyre@main#subdirectory=extensions/clickhouse-ingest"
```

or, matching the exact invocation the baked-image ingest uses:

```bash
uv run --no-project \
  --with "git+https://github.com/torch-spyre/torch-spyre@main#subdirectory=extensions/clickhouse-ingest" \
  <your-script.py> ...
```

### Why it isn't a `pyproject.toml` dependency in consumer repos

Both spyre-inference and hf-adapters install it as a **per-job CI step**
rather than declaring it in their own `pyproject.toml`. The reason is the
same in both repos: their build step runs `uv sync --frozen`, which would
require a lock-file entry and pin the library to whatever commit was locked
at — meaning a fix to this library wouldn't take effect in that consumer
until someone re-locked *that* repo. Installing it as a step keeps `@main`
meaningfully live.

### Local dependencies (what you get)

```toml
dependencies = [
    "clickhouse-connect>=1.3.0",
    "regex",   # hw_parse.py imports it; stdlib `re` is forbidden repo-wide
]
```

Deliberately **not** a dependency: `torch_spyre` itself. Installing that
would pull in torch/numpy/ortools, and ortools ships no ppc64le/s390x wheel —
the ingest would break on the very architectures the Jenkins side exists to
cover.

---

## Reuse it as a GitHub Actions workflow

Full reference: [GitHub Actions](github-actions.md). Short version for a
**new** repo that wants to start pushing to ClickHouse:

1. **Pick the composite action matching what you're ingesting** —
   `ingest-xml-to-clickhouse` for JUnit/benchmark results,
   `ingest-hw-diagnostics-to-clickhouse` for hardware/RAS failures,
   `ingest-model-ops-to-clickhouse` for capability logs, or write a small
   script following `capability_write.py`'s pattern (import
   `spyre_clickhouse_ingest` directly) if your data doesn't come from a GHA
   job log at all.

2. **Call it from your workflow**, always with `component:` set to your
   repo's own name (the value every id hashes in — get it wrong and your
   rows silently attribute to the wrong product):

   ```yaml
   - name: Ingest XML into shared v2 ClickHouse tables
     uses: torch-spyre/torch-spyre/.github/actions/ingest-xml-to-clickhouse@main
     with:
       xml-dir: xml_artifacts
       workflow: ${{ github.workflow }}
       branch: ${{ github.ref_name }}
       sha: ${{ github.sha }}
       gha-run-id: ${{ github.run_id }}
       trigger-type: regression
       component: your-repo-name
       schema: v2
       clickhouse-host: ${{ secrets.CLICKHOUSE_HOST }}
       clickhouse-port: ${{ secrets.CLICKHOUSE_PORT }}
       clickhouse-user: ${{ secrets.CLICKHOUSE_USER }}
       clickhouse-pass: ${{ secrets.CLICKHOUSE_PASS }}
       clickhouse-db: ${{ secrets.CLICKHOUSE_DB }}
       clickhouse-db-v2: ${{ secrets.CLICKHOUSE_DB_V2 }}
   ```

3. **Set the five (or six, with `_V2`) `CLICKHOUSE_*` secrets** at the repo
   or org level. Leaving `clickhouse-host` empty is a supported, clean no-op
   — useful for a fork or a repo that isn't ready to write yet.

4. **If you're not calling from `workflow_run`**, you don't need the
   download-artifacts dance every existing `push-*-to-clickhouse.yaml`
   does — that exists only because `workflow_run` can't see the triggering
   run's own files. A same-workflow call (like spyre-inference's perf leg in
   `_test_matrix.yaml`) can pass `xml-dir` pointing straight at a directory
   your own job just wrote.

5. **Writing your own ingest script instead of reusing a composite action**
   (e.g. a new kind of result that doesn't fit XML/hw-log/model-ops shapes)?
   Follow `capability_write.py`'s pattern: install the library as a step
   (never in `pyproject.toml`, per above), `import` the pieces you need
   (`get_client`, `run_id_of`, `tables_present`, `target_database`, the
   relevant `Writer`/`insert_*` function, and the relevant `schema.py`
   table classes), gate everything on `target_database()` returning
   non-empty, and swallow — with a printed warning, never a raised
   exception — any v2-specific failure so a v2 problem never costs you the
   v1 rows the run exists to produce.

---

## Reuse it as a Jenkins shared library

Full reference: [Jenkins shared library](jenkins-shared-library.md). Short
version — **this is a different mechanism**, not this Python package called
from Groovy:

1. **Load the shared library once per stage or Jenkinsfile:**

   ```groovy
   library "spyre-frameworks-lib@${env.GIT_BRANCH}"
   ```

2. **For test-result ingestion**, call `pushToClickhouse.pushJUnitXml(...)`.
   This *does* end up running `ingest_xml.py` from this package — but inside
   a `podman run` against your product's own container image, with
   `extensions/clickhouse-ingest` attached via `uv run --with <local-path>`,
   not a git install. That means:
   - Your product's container image must bake a checkout of itself at
     `/home/senuser/<product>/`, including the ingest script (default path
     `.github/scripts/ingest_xml.py`, overridable via `ingestScript`).
   - For schema-v2 writes, the image must also bake
     `extensions/clickhouse-ingest` at
     `/home/senuser/<product>/extensions/clickhouse-ingest` —
     `pushJUnitXml` auto-detects and attaches it if present, and silently
     stays v1-only if it isn't.

3. **For artifact/build tracking** (`artifacts`, `artifact_tags`,
   `artifact_results`), call `pushToClickhouse.pushArtifactRecord(...)`,
   `pushArtifactTag(...)`, or `pushArtifactResult(...)`. These are pure
   Groovy — no Python, no container needed — and write over ClickHouse's
   HTTP interface directly.

4. **Provision credentials**: four Jenkins `string`/`usernamePassword`
   credential IDs (`<id>`, `<id>-host`, `<id>-port`, `<id>-db`; default id
   `spyre-clickhouse-creds`), plus, for v2, a folder property
   `CLICKHOUSE_DB_V2_CRED_ID` naming a credential that holds the v2 database
   name.

5. **Before changing any identity-derivation logic on either side**, read
   [Identity: one formula, four implementations](jenkins-shared-library.md#identity-one-formula-four-implementations).
   This package's `identity.py` and the Jenkins side's id formulas are
   independently maintained and must be changed **together**, verified
   against the same golden values, or ids silently stop joining across the
   GHA/Jenkins boundary.

---

## Running the test suite

```bash
cd extensions/clickhouse-ingest
python3 -m pytest tests/
```

| Test file | What it pins |
|---|---|
| `test_identity_golden.py` | Golden uuid5 values — the contract that makes `@main` safe to move. If a change here fails, it is about to invalidate historical rows. |
| `test_schema.py` | The exact `TABLES` set — a table added to the DDL without a matching `schema.py` class fails this. |
| `test_tables_present.py` | `Table.present()` behaviour against a live-shaped mock. |
| `test_artifact_writer.py`, `test_benchmark_writer.py` | Writer-class behaviour: dedup scoping, partial-id refusal, measurement merging. |
| `test_hw_parse.py`, `test_hw_schema.py` | Log-parsing regex behaviour and the self-migration path. |
| `test_gha_logs.py` | Retry/backoff behaviour against transient `gh` failures. |
| `test_client_env.py` | Env-var resolution rules (blank-as-absent, port parsing, etc.). |

[← Back to index](README.md) · [← Jenkins shared library](jenkins-shared-library.md)
