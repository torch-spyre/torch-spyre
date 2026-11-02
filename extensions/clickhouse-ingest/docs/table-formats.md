# Upload formats, table by table

Every way of getting results into `spyre_v2` (CI, the CLI, the Python SDK, the dashboard's
upload page and API, an offline bundle) ends in the same writers, so a row means the same thing
whichever door it came through. This page says which file format feeds which table, with an
example you can copy.

- [Which format for which table](#which-format-for-which-table)
- [Direct JSON: one file per table](#direct-json-one-file-per-table) (`kind: tables`)
- [Each table's file](#each-tables-file): `test_case_runs`, `benchmark_runs`, `capability_runs`, `artifact_results`
- [A whole bundle, end to end](#a-whole-bundle-end-to-end)
- [The same data through each interface](#the-same-data-through-each-interface)
- [Re-uploading, correcting, retracting](#re-uploading-correcting-retracting)
- [Tables you do not upload](#tables-you-do-not-upload)
- [Schemas](#schemas)

## Which format for which table

| You have | Use | Rows end up in |
|---|---|---|
| JUnit XML (pytest `--junitxml`) | `kind: junit` | `test_cases`, `test_case_runs`, `artifact_results` |
| `vllm bench` output | `kind: vllm` | `benchmarks`, `benchmark_runs`, `artifact_results` |
| spyre-perf-suite `report_sendnn.txt` | `kind: sendnn` | `benchmarks`, `benchmark_runs`, `artifact_results` |
| Eval or harness results already shaped as benchmark rows | `kind: benchmarks` | `benchmarks`, `benchmark_runs`, `artifact_results` |
| **Anything else, or you want exact control** | **`kind: tables`**: one JSON file per table | the table named by each file |

Pick the runner-specific kind when one fits: it reads the tool's own output, so you write nothing.
Use `kind: tables` when your results come from your own harness, or when you want to say exactly
what each row is.

## Direct JSON: one file per table

A `kind: tables` bundle holds one file per table under `results/`, **named after the table**:

```
bundle.json                  # which artifact, which tier
results/test_case_runs.json  # any of these, one bundle = one leg
results/benchmark_runs.json
results/capability_runs.json
results/artifact_results.json
```

Each file is `{"rows": [ ... ]}`. A row carries the fields that **identify** the thing (a test's
`classname` and `name`, a benchmark's `name` and `props`); you never write ids. The writers hash
those fields into `test_case_id` / `benchmark_id` / `capability_id`, fill the identity tables
(`test_cases`, `benchmarks`, `capabilities`) for you, skip a row already recorded, and enforce the
database's CHECK constraints. So a row that would fail the database fails the validator first, with
the file and row number.

`bundle.json` is the same as for every kind (see the [bundle table](../README.md#the-bundle)):

```json
{"kind": "tables", "artifact_id": "<uuid>", "test_type": "regression", "component": "my-suite"}
```

- `test_type` is the bundle's one tier, and decides which tables it may carry:

| `test_type` | may hold |
|---|---|
| `smoke`, `unit`, `integration`, `regression`, `trunk`, `fvt*`, `svt*` | `test_case_runs`, `artifact_results` |
| `perf`, `quality` | `benchmark_runs`, `artifact_results` |
| `model_ops`, `model_modules`, `model_support`, `model_serving` | `capability_runs`, `artifact_results` |

- `component` names the suite's owner. It is part of every identity hash, so keep it stable.
- If you give no `artifact_results.json`, the verdict is derived: `failed` if any case or capability
  failed, else `passed`.

## Each table's file

### `test_case_runs.json`: functional tests

One row per test outcome.

```json
{"rows": [
  {"classname": "tests.test_attention", "name": "test_causal_mask", "status": "passed",
   "duration_s": 1.5, "tags": ["model__granite"], "measurements": {"latency_ms": 12.5}},
  {"classname": "tests.test_attention", "name": "test_gqa", "status": "failed",
   "fail_message": "shape mismatch"}
]}
```

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | The test's name. **Identity.** |
| `status` | yes | `passed`, `failed`, `error`, `skipped`, `xfail`, `xpass` |
| `classname` | no | Class or module. **Identity.** |
| `tags` | no | `namespace__value`. Plain tags are **identity** (they name the test). `platform__`, `testtype__` and `cadence__` describe this run and are not identity. |
| `duration_s` | no | Seconds |
| `fail_message` | no | Kept to 8192 characters |
| `measurements` | no | Numbers this run measured, e.g. `{"latency_ms": 12.5}` |
| `props` | no | Only `result.*` keys, recorded as given |

### `benchmark_runs.json`: measurements

One row per (benchmark, backend). Use it for performance numbers (`test_type: perf`) and for
accuracy or evaluation results (`test_type: quality`).

```json
{"rows": [
  {"name": "lm_eval/gsm8k", "backend": "spyre", "iterations": 200,
   "props": {"record_type": "quality", "model": "ibm-granite/granite-3.3-8b-instruct"},
   "measurements": {"exact_match": 0.52}},
  {"name": "lm_eval/gsm8k", "backend": "cpu", "iterations": 200,
   "props": {"record_type": "quality", "model": "ibm-granite/granite-3.3-8b-instruct"},
   "measurements": {"exact_match": 0.53}}
]}
```

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | The benchmark's name. **Identity.** |
| `measurements` | yes | `metric: number` or `metric: [samples]`. Finite numbers only |
| `backend` | no | `spyre`, `cpu`, … The axis two runs of one benchmark compare across. **Not identity** |
| `props` | no | Describe the benchmark. **All are identity by default** |
| `identity` | no | Which `props` are identity. Omit for all; `[]` for the name alone |
| `iterations` | no | The n behind a pre-averaged number |
| `run_props` | no | Describe this run of the benchmark. Never identity |
| `tags` | no | Part of identity |

**Choosing identity.** The benchmark id is a hash of the name, tags and the identity props.
Two rows with the same id are *the same benchmark* and their runs form one trend. Put in identity
whatever makes two rows "not comparable" (the model, a batch size). Leave out text that may be
corrected later, such as a question's reference answer:

```json
{"name": "ragq/spyre_guide/q01", "backend": "spyre", "identity": [],
 "props": {"question": "What is the IBM Spyre Accelerator?", "reference_answer": "…", "source_pages": "[13]"},
 "measurements": {"ttft_s": 2.2, "total_s": 118.1},
 "run_props": {"generated_answer": "…"}}
```

Here every run of question 1 is one benchmark, whatever the reference says; the answer, which differs
per run, rides on the run.

### `capability_runs.json`: what works, per model or config

One row per capability verdict (a model op, a module, a serving config).

```json
{"rows": [
  {"test_type": "model_ops", "subject": "granite-3.3-8b", "name": "aten.mul.Tensor",
   "status": "passed", "backend": "spyre", "arch": "s390x"},
  {"test_type": "model_ops", "subject": "granite-3.3-8b", "name": "aten.erf.default",
   "status": "not_implemented", "backend": "cpu", "sig": {"dtype": "bf16"}}
]}
```

| Field | Required | Meaning |
|---|---|---|
| `test_type` | yes | `model_ops`, `model_modules`, `model_support`, `model_serving`; equal to the bundle's |
| `subject` | yes | What it belongs to (a model, a config). **Identity** |
| `name` | yes | The capability. **Identity** |
| `status` | yes | `passed`, `failed`, `not_implemented`, `undetermined` |
| `backend` | no | `spyre` or `cpu`. A column, **not identity**: it is how a verdict reads "runs on Spyre" vs "falls back to CPU" |
| `sig` | no | Extra keys that make `subject` + `name` unique (a dtype, a shape). **Identity** |
| `arch` | no | Where it ran, if different from the bundle's |
| `fail_reason`, `tags`, `props` | no | Recorded with the verdict |

### `artifact_results.json`: the leg's verdict

At most one row. It states how the leg went; you usually leave it out and let the bundle derive it.

```json
{"rows": [{"test_type": "regression", "state": "failed", "duration_s": 312.0,
           "props": {"note": "known flaky on s390x"}}]}
```

| Field | Required | Meaning |
|---|---|---|
| `test_type` | yes | Equal to the bundle's |
| `state` | yes | `passed`, `failed`, `error` |
| `result_kind` | no | `functional`, `performance` or `capability`; derived from `test_type` when omitted |
| `arch`, `duration_s`, `props` | no | |

## A whole bundle, end to end

```bash
mkdir -p rows && cat > rows/test_case_runs.json <<'EOF'
{"rows": [{"classname": "my.suite", "name": "test_one", "status": "passed"}]}
EOF

python -m spyre_clickhouse_ingest results --offline --out mybundle.tgz \
    --tables-dir rows --artifact-id <artifact_id> --arch s390x \
    --trigger-type regression --component my-suite

python -m spyre_clickhouse_ingest results --from-bundle mybundle.tgz --validate-only   # check it
python -m spyre_clickhouse_ingest results --upload mybundle.tgz --wait                 # send it
```

`--validate-only` runs offline on the standard library, and names the file, the row and the problem
for anything it refuses:

```
{"status": "rejected", "reason": "test_case_runs.json rows[0]: status 'nope' is not one of ['error', 'failed', 'passed', 'skipped', 'xfail', 'xpass']"}
```

## The same data through each interface

| Interface | How |
|---|---|
| **CLI** | `results --offline --tables-dir <dir> …` to write a bundle, `--upload` to send it, `--from-bundle` to record it |
| **Python SDK** | `from spyre_clickhouse_ingest.tables import validate_rows` to check rows before you send them; `insert_test_results`, `insert_benchmarks` and `insert_capabilities` are the writers underneath |
| **Dashboard upload page** | Choose `*.json` files named after their tables, or a `.tgz` bundle |
| **Dashboard API** | `POST /api/v2/uploads`: see the [upload guide](https://github.ibm.com/ai-sw-acceleration/spyre-dashboard/blob/dev/docs/v2-uploads.md) |
| **Offline / air-gapped** | Write the bundle, copy it, `--upload` from a connected host. The relay records it |

## Re-uploading, correcting, retracting

- **Same bundle again** is recognised and skipped: its run key is the bundle's content hash.
- **A corrected file** is a different bundle, so it is a new run. It does not overwrite the old
  one; retract the old upload from the Uploads page if it was wrong.
- **Within a run**, a row already recorded for the same identity is skipped.

## Tables you do not upload

These have a producer of their own, or a wrong row would damage other results. They have a
generated schema for reference only.

| Table | Why | How it is written |
|---|---|---|
| `artifacts`, `artifact_refs` | An artifact is identified by what built it | `artifacts` CLI, or `create_artifact` in `bundle.json` for an image no pipeline of ours built |
| `artifact_tags` | A tag moves a channel | The orchestrator, or the `artifacts` CLI |
| `artifact_result_reasons` | Derived from a failed verdict | Failure collectors |
| `pipeline_runs`, `pipeline_run_links`, `ci_run_timings` | CI bookkeeping | Jenkins and GitHub Actions |
| `artifact_subscriptions`, `dispatch_*`, `artifact_dispatches` | Subscriptions and dispatch | The dashboard's subscription editor |
| `rbac_roles`, `rbac_bindings` | Access control | The dashboard's access admin |
| `test_cases`, `benchmarks`, `capabilities` | Identity tables | Filled by the writers from the rows above |

## Schemas

Machine-readable JSON Schemas are generated from the table classes, so they cannot drift:

- [`schemas/<table>.upload.schema.json`](schemas): the upload file for `test_case_runs`,
  `benchmark_runs`, `capability_runs`, `artifact_results`.
- [`schemas/<table>.stored.schema.json`](schemas): the stored row of every table, for reference.

Regenerate with `python -m spyre_clickhouse_ingest.tables --out docs/schemas`; a test fails when the
committed copies differ.
