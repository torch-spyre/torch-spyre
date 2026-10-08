# spyre-clickhouse-ingest

The schema-v2 ClickHouse schema, derived identity and write path, shared by the Spyre CI ingests.

> **Full reference:** class-by-class docs, every GitHub Actions and Jenkins consumer across
> torch-spyre/spyre-inference/hf-adapters/spyre-frameworks, and getting-started guides live in
> [`docs/`](docs/README.md).

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
| `mark_retried.py` | stamps `result.retried` / `result.prior_*` on a retry's JUnit XML (stdlib-only; `spyre-mark-retried`) |
| `hw_parse.py` | GHA log → `hw_failure_diagnostics` records (RAS events, phases, pytest counts) |
| `hw_schema.py` | `hw_failure_diagnostics` columns + its `ADD COLUMN IF NOT EXISTS` migration |
| `hw_diagnostics.py` | `build_row`/`insert_rows` for `hw_failure_diagnostics` |
| `gha_logs.py` | fetching GHA job logs via `gh`, with transient-5xx retry |
| `resolver.py` | any artifact spec -> its one spyre_v2 artifact, existing or derived (`resolve` / `ensure`) |
| `options.py` | the artifact flags every surface shares (`add_artifact_options`) |
| `registry.py` | read-only registry access: an image's per-arch leaf, labels and its tag in a tag family |
| `tag_families.yaml` | the tag families (`SPYRE_TAG_FAMILIES` overrides it) |
| `results.py` | the JUnit/benchmark XML ingest (`python -m spyre_clickhouse_ingest results`) |
| `vllm.py` | vLLM bench JSON -> `benchmarks`/`benchmark_runs` (`report_kind=vllm`): spyre-inference's live perf ingest and vLLM bundles |
| `bundle.py`, `bundle.schema.json` | offline results bundles: `bundle init`/`seal`/`validate`/`ingest` ([below](#offline-results-bundles)) |
| `gha_runs.py` | polls GitHub Actions runs and jobs into `pipeline_runs` (`source='gha'`); the Jenkins rows come from spyre-frameworks |
| `ci_run_timings.py` | one orchestrator run's timeline into `ci_run_timings`, a row per build and test leg (`python -m spyre_clickhouse_ingest ci-run-timings write`); the batch comes from spyre-frameworks |

## Naming an artifact (simple -> advanced)

One option group (`options.add_artifact_options`) is shared by every surface, so a flag means the
same thing everywhere; all of them go through `resolver.resolve` (read-only) or `resolver.ensure`
(records it). The spec forms are listed in `resolver.py`'s docstring.

**CLI** (`python -m spyre_clickhouse_ingest artifacts ...`, or the `spyre-clickhouse-ingest`
script):

```bash
# 1. What does this image name? Read-only, prints the resolution as JSON.
python -m spyre_clickhouse_ingest artifacts resolve --artifact image:icr.io/<repo>@sha256:<list> --arch s390x

# 2. Record it, tagged by its supply-chain family (the registry's tag, else --tag-date).
python -m spyre_clickhouse_ingest artifacts ensure --artifact image:icr.io/<repo>@sha256:<list> \
  --arch s390x --tag-family nightly-supply-chain

# 3. An identity the caller already knows: no registry call, never rebound to another id.
python -m spyre_clickhouse_ingest artifacts ensure \
  --artifact 'rpm:ibm-flex-*.<id12>.*.x86_64;component=flex;name=ibm-flex;id12=<id12>' --arch x86_64 \
  --source ai-chip-toolchain/flex@main@<sha> --identity-dep base=<sha256> --prop k=v \
  --tag rc1 --tag-prop k=v --run-url "$BUILD_URL" --lookup off --registry off --dry-run
```

`--lookup auto|off|only`: an existing record wins / derive only, read no database / it must be
recorded. `--registry auto|off`: called only when the spec needs it / never. A recorded image
digest is answered from the database with no registry call. With no registry answer (off, or
unreachable: no `ICR_*` credentials) an unrecorded image is found by a tag ending in its id12
(`s390x-dev-<id12>`) held by one record, else derived from its digest; an image named only by
any other tag then fails. The output's `registry` says which happened.

In `results` a tag it cannot file (an unknown `--tag-family`) is dropped with a warning; the
other tags and the verdicts are recorded regardless. `--strict` makes `results` exit 1 when a
named artifact's verdicts were not all recorded, or when a tag fell back to `misc`.
Only immutable refs are looked up -- a digest, `name==version`, an rpm NEVRA or glob, a generic
URL with its sha -- never a moving tag or a bare name. `--arch multi` keeps a manifest list's own
digest. `--dry-run` prints what would be written (`"written": true` and the would-be `rows`) and
writes nothing; with `--lookup off` it needs no database.

**Ingest** (`python -m spyre_clickhouse_ingest results ...`): the same flags name the artifact
a run tested; `--platform` is a deprecated alias of `--arch`, and `--tag-date` defaults to the
run's start day. `--artifact-id <id>|<base>|<installed>` is derive-gha-artifact-id's record
(= `--artifact gha:<record>`); a bare `--artifact-id` that is not recorded writes no verdict.

**SDK**:

```python
from spyre_clickhouse_ingest import ensure, resolve

r = resolve("wheel:torch-spyre==0.1+<id12>", "x86_64", client=client, db="spyre_v2")
r = ensure(client, "spyre_v2", "image:icr.io/<repo>@sha256:<list>", "s390x",
           tag_family="nightly-supply-chain", tag_date=day, sources=[(repo, ref, sha)])
r.artifact_id, r.source, r.written  # source: given | existing | label | derived
```

`resolve` returns a `Resolution` (or None when the spec names nothing); `ensure` raises
ValueError instead. An `ArtifactIdentity` passed as the spec is authoritative. `artifacts write`
batch entries take either `artifact` (the hash inputs) or `spec` (plus any resolve option), and go
through `ensure` too.

### Tag families

`tag_families.yaml` (its header documents every key and template field) maps a family to the
registry tag that names an image in it and the v2 tag it is recorded as. `SPYRE_TAG_FAMILIES=<path>`
replaces the whole file; a bad regex, an unknown key or template field, or a duplicate family
fails at load, as does a file with no `misc` family. A `--tag` whose prefix names no family and
is given no `--tag-family` is filed under `misc`, with a warning naming the tag -- never under
`release`, which takes only `release-*` tags. `misc` is never matched by prefix or dated; pass the
real family (`pr`, `main`, `nightly`, ...) instead, or `--tag-family misc` to mean it.
A family's `spellings` are other prefixes a registry uses for it: `cicd-tech-preview-vN` is stored
as `ci-cd-tech-preview-vN`, with the raw spelling in the tag prop `registry_tag`, so one tech
preview is one tag.

## Moving off ingest_xml.py

`.github/scripts/ingest_xml.py` only forwards to `python -m spyre_clickhouse_ingest results`
(same arguments). Delete it once this finds no caller:

```bash
grep -rn 'scripts/ingest_xml\.py' --include='*.y*ml' --include='Jenkinsfile*' --include='*.groovy' \
  --include='*.sh' --include='*.py' --include=Makefile \
  torch-spyre hf-adapters spyre-inference spyre-frameworks spyre-test-framework
```

## Tables modelled

`test_cases`, `test_case_runs`, `benchmarks`, `benchmark_runs` (DDL: `functional_tests_v2.sql`)
and `artifacts`, `artifact_refs`, `artifact_tags`, `artifact_results` (DDL: `artifacts_v2.sql`),
and `pipeline_runs` (DDL: `schema/47-pipeline-runs.sql`), and `ci_run_timings`
(DDL: `schema/48-ci-run-timings.sql`).
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

## Offline results bundles

For a run that cannot reach ClickHouse (an isolated lab, an air-gapped host, a person testing by
hand): record the results as a bundle, carry it to a connected host, and upload it to the
Artifactory inbox. The `v2-results-relay` job (spyre-frameworks, every 15 minutes) ingests it
with `bundle ingest`, through the path a connected run takes. JUnit goes through `results`, and
vLLM bench JSON through `vllm.py` (the module spyre-inference's live perf ingest uses). So the
rows link to the artifact exactly as a connected run's do, and differ only in `source=bundle`.
A bundle records results only: its artifact must already be in spyre_v2, and a manual bundle
adds no tags.

A bundle is a directory, or a `.tgz` of one:

```
bundle.json        # bundle.schema.json, schema_version 1
results/           # kind junit: *.xml (JUnit, or benchmark XML); kind vllm: vLLM bench *.json
attachments/...    # optional: logs, .cmd files, notes; kept with the bundle, not ingested
```

| `bundle.json` key | |
|---|---|
| `schema_version` | `1` |
| `kind` | `junit` (default) or `vllm` |
| `artifact_id` | the tested artifact's spyre_v2 id: tried first |
| `image` | `<repo>@sha256:<digest>`, the image as pulled: the fallback (with `arch`) |
| `component`, `arch`, `test_type` | the suite's owner, where it ran, the tier (`artifact_results.test_type`); `vllm`: `spyre-inference`, `perf` |
| `run_key` | `manual:<who>:<uuid4>`, or a Jenkins `<JOB_NAME>#<BUILD_NUMBER>` (alias `jenkins_run_key`) |
| `started_at`, `ended_at` | ISO-8601 with a zone; `seal` fills them |
| `files` | `[{path, sha256}]` for every other file; `seal` fills it |
| `perf` (`vllm`) | `{model, tensor_parallel, input_len, output_len, cards, head_sha, head_branch, state}`, all optional |
| optional | `run_url`, `workflow`, `runner` {host, user, image, ...}, `env` {...}, `notes`, `uploader`, `sealed_at`; `tag_family`/`tags` for Jenkins keys only |

**The artifact.** At least one of `artifact_id` / `image` is needed; carry both when you have
both. Ingest resolves the id first, else the image on `arch` (`resolve --lookup only`). It never
records or tags an artifact. When both resolve, they must name the same artifact, or the bundle
is rejected. A blank field counts as absent. An `artifact` spec (`id:...`, `image:...`) is read
as these fields.

**The run** is `RunId(source, run_key, arch, test_type)`, with source `bundle` for a manual key
and `jenkins` for a Jenkins key. A Jenkins-keyed bundle therefore gets the run_id a connected
`v2Results` run of that build would get. The relay accepts a Jenkins key only from trusted jobs
(`Spyre-Test/testing/`). Re-ingesting the same bundle is a no-op. A different bundle or run under
the same run_key is rejected. The verdict's `props` carry:
- `source=bundle`;
- `uploader`: the Artifactory account that uploaded it;
- `bundle_sha256`: of bundle.json;
- `bundle_url`: also the `run_url` when the bundle names none.

**vLLM bundles** (`kind: vllm`) hold the runner's output under `results/`:
- `<test_name>.json` from `vllm bench latency|throughput|serve` (`--output-json` /
  `--result-filename`);
- `<test_name>.pytorch.json` when `SAVE_TO_PYTORCH_BENCHMARK_FORMAT=1`. Keep it: the `latency`
  samples and the model name come from it.

`test_name` is `<latency|throughput|serve>_<model>_tp<N>_in<N>_out<N>`, as in spyre-inference's
`vllm-benchmarks/` configs. The mode and shapes in it are the benchmark's identity. They must
agree with `perf.tensor_parallel` / `input_len` / `output_len` when those are given.
`perf.model` names a benchmark whose files name none, and `perf.state` is the verdict (default
`passed`). The rows are `benchmarks` + `benchmark_runs` (`report_kind=vllm`) and one `perf` /
`performance` verdict in `artifact_results`. `.cmd` and `.log` files go in `attachments/`.

### Air-gapped flow

1. **While connected**, from the artifact's dashboard page: pull the image **by digest**, and
   copy the pre-filled `bundle.json` (or the `bundle init` command). The ids are filled in, so
   nothing after this step needs the network.
2. **Offline**, run the tests with their output in `results/`, then seal **on the test host**.
   pytest writes suite timestamps in local time, so `seal` reads them there. For vLLM, the
   oldest result file's time is used, unless `--started-at` was given.
3. **Connected**, `bundle validate` it, then upload it under its inbox path (`seal` and
   `validate` print it).

```
# JUnit
python -m spyre_clickhouse_ingest bundle init --artifact id:<artifact_id> \
    --image <repo>@sha256:<digest> --arch <arch> \
    --component <component> --test-type <tier> --out mybundle   # skip if you copied bundle.json
pytest ... --junitxml=mybundle/results/junit.xml
# vLLM bench: spyre-inference's runner, results straight into the bundle
python -m spyre_clickhouse_ingest bundle init --artifact id:<artifact_id> \
    --image <repo>@sha256:<digest> --arch <arch> --kind vllm --out mybundle \
    --perf head_sha=<spyre-inference sha> --perf cards=1
SAVE_TO_PYTORCH_BENCHMARK_FORMAT=1 python .github/scripts/run_vllm_benchmarks.py \
    --configs-dir vllm-benchmarks/benchmarks/spyre --results-dir mybundle/results ...
mv mybundle/results/*.cmd mybundle/results/*.log mybundle/attachments/
# Then, either way:
python -m spyre_clickhouse_ingest bundle seal mybundle
COPYFILE_DISABLE=1 tar -C mybundle -czf mybundle.tgz .
python -m spyre_clickhouse_ingest bundle validate mybundle.tgz    # on the connected host
```

Use `--image` alone when you have no artifact id. With no package on the test host, write
`bundle.json` by hand and fill `files` from `sha256sum results/* attachments/*`.

### Uploading

The inbox is `sys-ai-sw-accel-team-cos-dev-generic-local/zsp/next/<arch>/v2-results/inbox/<key>/`
on `https://na.artifactory.swg-devops.com/artifactory`:
- `<key>` is the `artifact_id`, or `sha256-<digest>` for a bundle that names only an image;
- the file goes there as `<bundle-name>.tgz`, or as a folder `<bundle-name>/`;
- `<bundle-name>` is the run key with every character outside `A-Za-z0-9._-` replaced by `_`,
  then `-<test_type>`.

Uploading needs deploy permission on that repository.

```
ART=https://na.artifactory.swg-devops.com/artifactory/sys-ai-sw-accel-team-cos-dev-generic-local
INBOX=zsp/next/<arch>/v2-results/inbox/<key>
# One file:
curl -fsS -H "Authorization: Bearer $ARTIFACTORY_TOKEN" -T mybundle.tgz "$ART/$INBOX/<bundle-name>.tgz"
# Or a folder, bundle.json last:
(cd mybundle && for f in results/* attachments/* bundle.json; do [ -f "$f" ] || continue
  curl -fsS -H "Authorization: Bearer $ARTIFACTORY_TOKEN" -T "$f" "$ART/$INBOX/<bundle-name>/$f"; done)
# Or the JFrog CLI:
jf rt upload mybundle.tgz "sys-ai-sw-accel-team-cos-dev-generic-local/$INBOX/<bundle-name>.tgz"
```

Within 15 minutes the relay moves it to `v2-results/processed/<key>/...`, or to
`v2-results/rejected/<key>/...` with a `REJECTED.json` giving the reason. A folder whose listed
files have not all arrived is left in the inbox, and is rejected after 24 hours.
`bundle ingest` exits `0` ingested or duplicate, `1` failed (retried), `2` rejected,
`3` incomplete, and prints one JSON line on stdout. `--dry-run` runs every check on a read-only
connection and writes nothing.

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
