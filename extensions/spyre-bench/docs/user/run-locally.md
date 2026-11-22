# Running benchmarks locally

**Status: Available** (validate, explain, ids, plan, run). Ingesting and reading verdicts also needs a results database;
without one you still get the raw results and a local summary.

## 1. Install

From a checkout of the repository:

```bash
uv pip install extensions/spyre-bench                    # engine
uv pip install "extensions/spyre-bench[jsonschema]"      # full schema validation (recommended)
uv pip install "extensions/spyre-bench[ingest,clickhouse]"  # ingest-mode ids, reuse lookup, verdicts
```

## 2. Validate the catalog

```bash
spyre-bench validate examples/run-demo --central examples/central
```

Expected:

```text
PASS examples/run-demo: 6 docs, 2 cases, packs {...}
```

Exit 0 is valid, 2 is invalid (the message names the file, the document and the rule). Fix those before anything else.

## 3. See what a case is

```bash
spyre-bench explain op_add_fp16 --catalog examples/run-demo --central examples/central
spyre-bench ids --catalog examples/run-demo --central examples/central
```

`explain` prints the case with every reference resolved: its identity inputs, the environment request, series and
reuse keys. `ids` prints the stored `benchmark_id` of every case. If a case is missing, check the selection (`--tier`,
`--suite`, `--cases`).

## 4. Plan

```bash
spyre-bench plan --catalog examples/run-demo --central examples/central \
    --purpose nightly --arch x86_64 --reference-node --out plan.json
```

Expected:

```text
ACTION  CASE         REPEATS  REASON
run     op_add_fp16  3
run     op_mul_fp16  2
plan: run 2, reuse 0, skip 0
```

`plan` never runs anything. Each case is `run`, `reuse` (a valid stored result exists) or `skip` (with a reason such as
a missing reference node or too little capacity). Useful flags: `--select`, `--cases`, `--tier`, `--reuse`,
`--force`, `--max-age`, `--budget`, `--devices CLASS=N`, `--store clickhouse`. Exit codes: 0 planned, 2 invalid
request or catalog, 4 every case skipped, 5 the store failed.

Every flag has a `BENCH_*` environment twin; the order is flag, environment, request file, default.

## 5. Run

```bash
spyre-bench run --plan plan.json --catalog examples/run-demo --central examples/central \
    --workdir examples/run-demo --out results/bench
```

Expected (last line):

```text
run <uuid>: 2 ok, 0 failed, 0 reused, 0 skipped, 0 not selected; ingest skipped, rows unchecked; exit 0 (ok)
```

`run` executes each case's declared command once per repeat, each in its own process group, with a limited
environment (no tokens or credentials are passed unless you name them with `--pass-env`). Outputs land under
`results/bench/cases/<case>-<hash>/`; `run.json` summarises the run and says `incomplete` until it finishes.

Use `--dry-run` to print the commands without running them.

| Exit | Meaning |
|---|---|
| 0 | every executed case ok, or only reuse cases |
| 1 | a case failed, no rows landed, or a result failed its checks |
| 2 | invalid plan, flags, env file or catalog |
| 4 | nothing ran |
| 5 | ingest or store failure |
| 130, 143 | interrupted |

## 6. Ingest and read the verdict (optional)

Rows reach the database only through the ingest; the engine writes none itself. With credentials for a development
database in the `CLICKHOUSE_*` environment:

```bash
spyre-bench run --plan plan.json --out results/bench --ingest --store clickhouse
spyre-bench verdict --run-dir results/bench      # needs the same CLICKHOUSE_* settings; --store none fails
```

**Do not ingest from a laptop into a shared production database.** Local runs belong in a development database or in
no database. To contribute a local or lab measurement to the shared store, use an [upload](uploading.md), which labels
its trust correctly.

## When it fails

| Symptom | Fix |
|---|---|
| `plan` exits 2 | the message names the invalid field; run `validate` first |
| a case is `skip` with `reference_node_required` | pass `--reference-node` only if the machine really is one; otherwise change the purpose |
| `run` exits 2 with `engine_skew` | the plan was made by another engine version; re-run `plan` |
| a repeat fails with a timeout | raise `--timeout`, or check the benchmark's own `run.timeout` |
| `verdict` finds no stored run ids | add `--ingest` and `--store clickhouse` to `run` |
| nothing prints on ingest | set `--ingest-python` to an interpreter that has the ingest package |

## Who to ask

The owner of the benchmark (the `owner` shown by `explain`) for the benchmark's command; the `spyre-bench` maintainers
for engine behaviour.
