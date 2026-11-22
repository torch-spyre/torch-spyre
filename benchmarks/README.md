# Benchmark definitions

Definitions for [`spyre-bench`](../extensions/spyre-bench/README.md) (`bench/v1alpha1`): what torch-spyre
benchmarks, not where or when they run. The catalog is advisory; nothing here gates a merge or a check.

| File | Contents |
|---|---|
| `project.yaml` | Project, the `ops` and `models` Suites, the `one-device` Environment request and the named Selections (`perf-core` is what one perf-suite report holds) |
| `systems.yaml` | the operator library under test |
| `ops/<operation>.yaml` | one Benchmark per operation name, expanded over input shapes into cases (`<operation>_<shape token>`) |
| `models/<model>.yaml` | one Benchmark per model name, expanded over config (`full_model`, `one_block`) and mode into cases |
| `ingest-ids.golden.tsv` | ids the ingest writes for each case's report, kernel and total rows (kernel names are synthetic) |
| `tools/check_identity_ids.py` | conformance check: recomputes the table with `spyre_clickhouse_ingest` and compares |
| `tools/check_report_coverage.py` | which rows of a perf report the catalog declares (`--strict` fails on an undeclared row) |
| `tools/run_kernel_case.py` | the command of a case: runs the repo's perf entry point once per run and keeps the case's own rows of `report.xml` and of the per-kernel breakdown XMLs |
| `tools/collect_ingest_xml.py` | gathers the `report.xml` of every case that ran ok, flat, for the XML ingest |
| `tools/bench_stub.py` | card-free stand-in for the `plan`, `run`, `verdict` and `report` verbs (smoke runs) |
| `tests/` | pytest suite for the conformance check, the make targets and the on-demand workflow |

## Commands

```bash
make bench-validate                      # schema, packs, cross-resource rules
make bench-explain CASE=matmul_prefill-n4096        # one case with its ids
make bench-check-ids                     # identity blocks against the ingest derivation
make bench-list bench-plan bench-run bench-verdict bench-smoke bench-report
make bench-nightly-smoke                 # plan -> run -> verdict -> report with the stub, no cards
```

These targets require the `spyre-bench` package:

```bash
uv pip install 'spyre-bench @ git+https://github.com/torch-spyre/torch-spyre@main#subdirectory=extensions/spyre-bench'
```

When the engine is not installed, or lacks the verb a target needs, the target prints the install command and exits 2.
A caller that must not fail without the engine (the `bench-validate` pre-commit hook) sets `BENCH_OPTIONAL=1`, which
turns that into a notice and exit 0; `BENCH_REQUIRED` (any non-empty value, including `0`) wins over it and keeps exit 2.
`BENCH_ENVIRONMENT` defaults to `one-device`, the Environment this catalog defines. The request is the `BENCH_*`
variables, which the engine also reads from the environment: `BENCH_PURPOSE`, `BENCH_ENVIRONMENT`,
`BENCH_SELECTION`, `BENCH_CASES`, `BENCH_TIER`, `BENCH_SUITES`, `BENCH_SUBJECT`, `BENCH_SUBJECT_ID`, `BENCH_REUSE`,
`BENCH_SET`, `BENCH_PLAN`, `BENCH_INGEST`, `BENCH_INVOKER`, `BENCH_CLEAN`, `BENCH_MAX_WAIT`, `BENCH_MAX_HOLD`,
`BENCH_LEASE_FILE`. `BENCH_ARCH`, `BENCH_CARDS` and `BENCH_QUEUE` are passed through unchanged; the Makefile never
branches on them. `BENCH_RESULTS_DIR` (default `bench-results/`) holds the run output and, under `plan/`, the plan (`BENCH_PLAN_FILE`); `bench-run` plans first unless `BENCH_PLAN` names a plan. `bench-run`
prefixes the engine with `SENPERFORMANCE=2` unless the caller set it, as `make tests TEST_TYPE=perf` does.

## Identity

Every Benchmark carries an `identity` block (`mode: ingest`, `profile: perf-kernel`), so the engine's stored
`benchmark_id` is the id the ingest writes for the same row. The block lists the six discriminators the perf-suite
parser hashes, in its order. An operation case: `record_type: op`, an empty `config_name`, `input_shapes` from the case's
`workload.flags.input_shapes` (the shape string exactly as the perf suite names it in `report.xml`), `run_mode:
op_benchmark`, and the empty `kernel_name` and `is_total` of an operation-level row. A model case: `record_type: model`,
`config_name` and `run_mode` from the case's flags, an empty `input_shapes`; `batch_size` and `prompt_length` are row
properties, not part of the id. Compiled-kernel and total rows are not cases: the ingest parser keeps writing them, and
their ids are pinned in the golden table.

`make bench-check-ids` (and the `bench-conformance` workflow) derive the ids twice, once from the definitions through
the engine and once by running the ingest's own `report.xml` parser over a report built from the definitions, and fail
when they differ, when an id in the golden table moves, or when a definition declares a `schedule` or lacks `est_hours`.
Changing an id on purpose means `python benchmarks/tools/check_identity_ids.py --update` and a reviewed golden diff.
The report rows of the 25 cases equal the ids the ingest has stored for these series (`SERIES_IDS` in
`tests/test_conformance.py`); kernel rows use made-up kernel names, as real ones change with every compile.

Series the ingest stores that the catalog does not declare, and why: the model rows of two further models (six each)
have no producer in this repository; two older `granite` configurations
(`bs1_pl512_*`) are retired and the suite no longer emits them. None is migrated and none gets a new id.

## Cost and cadence

`spec.est_hours` on a Benchmark is the only cost hint. When and how often a Selection runs belongs to whoever invokes
it (a workflow, a scheduler, a developer), never to these files.

## Nightly and weekly runs (Jenkins) and the on-demand workflow

Nightly and weekly runs are triggered from Jenkins presets, because they must run on the nightly-built image; the
`make bench-*` targets are what Jenkins calls. `.github/workflows/bench-on-demand.yml` is the manual route
(`workflow_dispatch` only, no schedule) on the labelled x86 card runners the repo's perf job uses: validate and
identity conformance, then plan, run, collect, ingest and verdict. It is advisory throughout (`continue-on-error`,
never a required check) and serialised by a concurrency group.

| Input | Default | Meaning |
|---|---|---|
| `ref` | the dispatch ref | branch, tag or SHA to check out |
| `selection` | `ops-core` | Selection of this catalog to run |
| `clickhouse_env` | `dev` | `prod` only when set explicitly; prod never falls back to the dev secrets |
| `ingest` | `true` | write the rows; `false` runs and prints verdicts only |
| `smoke` | `false` | stub engine on a hosted runner: no cards, no secrets used, nothing ingested |

Each case's `report.xml` (and the per-kernel breakdown XMLs holding its kernel rows, matched by their `op`, `input_shape`,
`config` and `mode` tags) is ingested by the existing XML parser, so stored ids are the ones the identity check pins.
In the workflow, rows reach ClickHouse through the `ingest-xml-to-clickhouse` action and the `CLICKHOUSE_*` secrets
(`*_DEV` for dev), inline on the runner; results are never uploaded as workflow artifacts and the log carries stage
outcomes and counts, not measurements. The rows are written under the workflow name `bench-on-demand`, so their series
(which keys on the producing job) does not mix with the perf job's. Use `make bench-nightly-smoke` or a `smoke`
dispatch to exercise the wiring without hardware.

### What a Jenkins invoker has to provide

The targets assume no GitHub Actions state: paths are relative to the checkout (`BENCH_RESULTS_DIR`, default
`bench-results/`, may be absolute), nothing reads runner variables, and the composite action is not needed.

1. **Checkout** of the ref under test at the repo root (the `Makefile`, `mk/`, `benchmarks/` and the perf entry point
   `make tests TEST_TYPE=perf` that `run_kernel_case.py` calls, so `spyre-perf-suite` must be on `PATH` as in the
   image).
2. **Engine**: `spyre-bench` with `plan`, `run`, `verdict` and the ingest package importable (`uv pip install
   ./extensions/spyre-bench ./extensions/clickhouse-ingest lxml` into a venv of its own; pass `BENCH=<venv>/bin/spyre-bench`).
   The benchmark command gets an allowlisted environment only: list any extra non-secret names the card setup exports
   in `BENCH_PASS_ENV`.
3. **Request variables**: `BENCH_PURPOSE` (`nightly` or `weekly`), `BENCH_SELECTION`, `BENCH_ENVIRONMENT=one-device`,
   `BENCH_REUSE=never`, `BENCH_SUBJECT_ID` (the artifact id of the image under test), `BENCH_INVOKER=jenkins`,
   `BENCH_QUIET=1`, `BENCH_REQUIRED=1`.
4. **Steps**: `make bench-validate bench-check-ids`, `make bench-run BENCH_INGEST=0` (plans first), then collect and
   ingest, then `make bench-verdict BENCH_RUN_ID=<the id passed to --run-id below>`.
5. **Collect**: `python3 benchmarks/tools/collect_ingest_xml.py bench-results bench-results/ingest-xml`; it prints the
   engine `run_id`.
6. **Ingest** (what the composite action runs, called directly; `CLICKHOUSE_HOST` and `CLICKHOUSE_PASS` are required,
   `CLICKHOUSE_PORT`, `CLICKHOUSE_USER`, `CLICKHOUSE_DB` and `CLICKHOUSE_DB_V2` as the target needs, and
   `INGEST_SCHEMA=v2`):

   ```bash
   python -m spyre_clickhouse_ingest results --xml-dir bench-results/ingest-xml --workflow <series name> \
     --branch <branch> --sha <sha> --run-id <engine run_id> --jenkins-run-key <job>-<build> --trigger-type perf \
     --arch x86_64 --component torch-spyre --artifact-id <artifact id> --schema v2
   ```

   `--run-id` must be the engine run id (the ingest threads a valid UUID verbatim, so the verdict can find the rows).
   `--workflow` is the series name: it becomes `source_job`, so use a name distinct from the perf legs'.
7. **Verdict** reads the same `CLICKHOUSE_*` connection; the target exits 2 when no stored run id is known or the
   verdict has no rows.

## Runner

Cases run through the runner the Project names, `kernel-bench`, a neutral runner name. A case's command is
`python3 benchmarks/tools/run_kernel_case.py --op <op> --shape <input_shapes> ...` (an `ops` case) or `--model <model>
--config <config> --mode <mode> ...` (a `models` case), plus `--suite-dir <dir> --raw-dir <dir>`, set as
`spec.run.command` (argv array with the engine's `{benchmark}`, `{case.*}`, `{out_dir}` and `{raw_dir}` tokens). A case
whose rows the perf report does not contain fails with exit 3 instead of ingesting a row.

`BENCH_SUITE_FROM=<dir>` points the runner at the output of a perf run that already finished (copied instead of running
the suite again); pass the name through to the command with `BENCH_PASS_ENV=BENCH_SUITE_FROM`.

## Running next to the existing perf leg (`bench_mode`)

The reusable `_test_matrix.yaml` perf job takes `bench_mode` (forwarded by `integration-tests.yaml`): `off` (default)
changes nothing. `side-by-side` additionally feeds the same `report.xml` through this catalog (no second suite run) and
files the engine's own results as a connected CI run (`bench-dir`, `channel-trust: ci`) in the series `bench-<workflow>`,
next to the unchanged legacy ingest. That run has its own run id, carries the per-case rows only (kernel rows stay with
the legacy ingest), and the action refuses `ci` on `pull_request` and `workflow_run` events (the perf job never runs on
those). `replace` skips the legacy ingest and stores the per-case XMLs (kernel rows included) under the legacy workflow
name and run id (v2 only), because a `bench-dir` run derives its run id from the engine run and cannot carry the
orchestrator's. Replace fails the job when the report or its kernel breakdowns hold a row no case declares
(`check_report_coverage.py --strict`).

Both modes are advisory (`continue-on-error`) except `replace`, where the mirror steps are the ingest. Side-by-side
puts a second point on the HUD series until the HUD reads one series per benchmark.

Every ingest of the workflows writes v2 only, including the default `off` legacy ingest. The HUD and the infra-perf
reads must read v2 once this merges, not only when `replace` is adopted.
