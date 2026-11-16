# 9. Test and debug locally

Everything in a pull request to a catalog should have been run once on your machine, in this order, cheapest first.
Steps 1 to 4 need no hardware and no network.

| Step | Command | Answers |
|---|---|---|
| 1 | `spyre-bench validate <catalog>` | is every document well-formed, every reference resolved, every name unique? |
| 2 | `spyre-bench explain <case>` | what is the identity, the Environment request, the series key? |
| 3 | `spyre-bench ids --catalog <catalog> --inputs` | for `identity: {mode: ingest}`: what exactly is hashed? |
| 4 | `spyre-bench plan ...` | which cases run, reuse or skip, and why? |
| 5 | `spyre-bench run ... --dry-run` | what exact commands would run? |
| 6 | `spyre-bench run ...` with a stub | does the plumbing work, and what does a Result look like? |
| 7 | `spyre-bench run ...` with the real command | does the real harness produce the metrics? |

## The inner loop with a stub

A stub is a command that stands in for the real harness: it takes the same arguments, writes plausible raw output and a
`metrics.json`, and exits 0 (or non-zero on purpose). `examples/docs-benchmark/tools/emit_metrics.py` is one;
`examples/run-demo/stub_bench.py` is another with a `STUB_FAIL` switch. Use a stub to test everything *around* the
harness before you spend hardware time:

```bash validated
spyre-bench validate examples/docs-benchmark
spyre-bench plan --catalog examples/docs-benchmark --purpose pr --select smoke --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan.json
spyre-bench run --plan plan.json --catalog examples/docs-benchmark --workdir examples/docs-benchmark --out results/local
```

A **failing** command exercises the failure paths. `examples/docs-benchmark/ops/op-broken.yaml` has two:

```yaml validated in=docs-benchmark/ops/op-broken.yaml
  variants:
    - name: op_fails
      workload: {type: kernel/op, op: fails, dtype: fp32, shape: [1], load: {executor: iterations, iterations: 1}}
      run: {command: ["{python}", "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"]}
    - name: op_silent
      workload: {type: kernel/op, op: silent, dtype: fp32, shape: [1], load: {executor: iterations, iterations: 1}}
      run: {command: ["{python}", "-c", "pass"]}
```

(They differ in `workload`, not only in `run`: a command is a run attribute, so two variants that differ only in their
command would be one benchmark.)

```bash validated
spyre-bench plan --catalog examples/docs-benchmark --purpose pr --cases op_fails,op_silent --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-fail.json
spyre-bench run --plan plan-fail.json --catalog examples/docs-benchmark --workdir examples/docs-benchmark --out results/fail # exit=1
```

```json validated subset-of=results/fail/run.json
{
  "counts": {"planned": 2, "run": 2, "ok": 0, "failed": 2},
  "cases": [
    {"name": "op_fails", "status": "failed", "reason": "repeats_failed", "repeats": {"planned": 1, "ok": 0, "failed": 1}},
    {"name": "op_silent", "status": "failed", "reason": "no_output", "repeats": {"planned": 1, "ok": 1, "failed": 0}}
  ],
  "exit_code": 1,
  "exit_reason": "cases_failed"
}
```

`op_fails` exited 3 (look in `cases/op_fails-<hash>/repeat-001/stderr.log` for `boom` and in `status.json` for
`returncode`). `op_silent` exited 0 but left no raw output and no `metrics.json`, which the engine treats as a failure
(`no_output`) unless the Benchmark says `run.allow_empty: true`.

## Reading what came out

| File | What to look at |
|---|---|
| `plan.json` | per case: `action` (`run`, `reuse`, `skip`) and `reason`; `labels` and `pinning` (what the Environment asked); `command` (case values filled in, run tokens still literal); `benchmark_id`, `series_key`; `summary`; `leg` (the facts the plan assumed: arch, devices, nodes, reference node); `producer.engine` (versions) |
| `run.json` | `counts`; per case `status`, `reason`, repeats ok/failed; `ingest` and `rows_landed` (`unchecked` when not requested, never a silent pass); `stored_run_ids`; `result_checks` and `result_notes` (warnings such as dropped fingerprint keys); `exit_code` and `exit_reason` |
| `plan.redacted.json` | the plan as run, with secrets masked; not runnable |
| `cases/<slug>/result.json` | the engine Result: `case` identity, `subject`, `environment` (`env_id`, fingerprint, labels), `series`, per-repeat `samples`, `aggregate`; every id recomputes from it |
| `cases/<slug>/repeat-NNN/status.json` | the exact argv run, `returncode`, `status`, `reason`, timings, `notes` (withheld environment names, truncated logs) |
| `cases/<slug>/repeat-NNN/stdout.log`, `stderr.log` | the command's output, capped |
| `cases/<slug>/raw/` | what the command wrote for the ingest |

Skip reasons you will meet in `plan.json` (the `reason` of a `skip`):

| Reason | Meaning |
|---|---|
| `capacity` | the leg offers fewer devices or nodes than the Environment needs, or no exclusive access for a clean request |
| `reference_node_required` | the Environment or Suite requires a reference node and the leg is not one |
| `no_environment` | no Environment resolved (neither the Benchmark nor the Project names one) |
| `profile_unsupported` | the requested profile level is not in the Benchmark's `profile.supported` |
| `capacity_timeout`, `max_hold_too_short` | a clean request whose wait or hold does not fit |

The reasons on a `run` record explain why reuse was not possible (`no_result_for_case`, `different_subject`,
`different_environment`, `validity_below_request`, `repeats_below_request`).

Exit codes of `run`: 0 every executed case ok (or an all-reuse plan); 1 a case failed, rows did not land or a Result
failed its checks; 2 invalid plan, flags, env file or catalog; 4 nothing ran; 5 an ingest, the store or the disk failed;
130/143 interrupted. `plan` exits 0 planned, 2 invalid, 4 every case skipped, 5 the store failed.

## Checking an identity change

Before you edit a System, a Workload or a name, copy the tree and compare. `validate --base-dir DIR` takes a directory
that holds the same-named catalog at the merge base, and warns about every case whose identity changed under an
unchanged case name (that case starts a new series):

```python validated exec
import shutil
from pathlib import Path

base = Path("base")
shutil.copytree(Path("examples") / "docs-benchmark", base / "docs-benchmark")
f = base / "docs-benchmark" / "ops" / "op-scale.yaml"
f.write_text(f.read_text().replace("precision: fp32", "precision: bf16"))
```

```bash validated
spyre-bench validate examples/docs-benchmark --base-dir base
```

In a repository this is `git worktree add ../base <merge-base>` and `--base-dir ../base`. A warning is not an error: a
changed id may be exactly what you intend. It is the reviewer's cue that history restarts.

## The make targets

Repositories that vendor a `mk/bench.mk` (it is not in this repository; each product repository carries its own copy, and its README lists the targets) expose the same loop as targets, with the request in `BENCH_*` variables (every
flag has an environment twin, and the flag wins):

```sh
make bench-validate                       # schema, packs, cross-resource rules
make bench-explain CASE=op_scale_n1024    # one case with its ids
make bench-plan BENCH_SELECTION=smoke     # plan only
make bench-run BENCH_SELECTION=smoke      # plan, then run on the devices you hold
make bench-smoke                          # the Selection named smoke: one case, one repeat, never ingested
make bench-report                         # summarise the results directory
```

`BENCH_PURPOSE`, `BENCH_ENVIRONMENT`, `BENCH_SELECTION`, `BENCH_CASES`, `BENCH_TIER`, `BENCH_SUITES`,
`BENCH_SUBJECT_ID`, `BENCH_REUSE`, `BENCH_SET`, `BENCH_PLAN` and `BENCH_INGEST` are the request; `BENCH_QUIET=1` hides
measured values (the default under `CI`). The targets need the engine installed
(`uv pip install extensions/spyre-bench`) and tell you so when it is missing; set `BENCH_REQUIRED=1` to make that an error.
A repository's own README lists the extra targets it defines (conformance checks, a card-free smoke run).

## Fixtures and synthetic data

- Keep every example value synthetic. Names are `node-a`, `demo-llm-a`; counts and latencies are placeholders.
- A Result for an Analysis can be built from the engine API (`engine.build_result`) in a script: see
  `tests/refresh_docs_results.py`, which writes the file the Analysis guide uses, with a `--check` mode a test runs.
- A stored result for reuse can be faked with `spyre_bench.store.MemoryStore`; `tests/test_run.py` shows the row shape.
- Qualification and invalidation rows have example shapes in `examples/tool-output/`; they are never authored in a
  catalog.

## Changes that touch the ingest or the schema

A catalog change that adds a metric to an existing profile, or an `identity` block, is checked against the real ingest:
install the extra from a checkout (`uv pip install 'extensions/spyre-bench[ingest]'`) and run the repository's
conformance check, then `spyre-bench ids --adapter`, which exits non-zero on a mismatch with the ingest's own key list.

A change under `extensions/clickhouse-ingest` (a parser, a view, a migration, a test that queries through `chdb`) is
tested against the real engine, not mocks:

```sh
cd extensions/clickhouse-ingest
uv run --with chdb --with pytest --with clickhouse-connect python -m pytest -q -p no:cacheprovider tests/
```

`chdb` does not import on every machine (it is a native library); run the same command where it does. A schema change is
also proven in a personal sandbox database (see `extensions/clickhouse-ingest/docs/sandbox.md`), never by editing a
shared or production database; the schema ships only as a pull request.

## Debugging by symptom

| Symptom | First look | Usual cause |
|---|---|---|
| `error: selection is empty` | `--arch`, `--select`, `--tier`, `--suite` | no `--arch` (cases whose Environment names an arch need it), or a filter that matches nothing |
| `FAIL ... 0 cases` | the first `error:` line | one schema error stops case expansion; fix it and re-run |
| a case is `skip capacity` | `plan.json` `leg`, the Environment's requests | no `--devices CLASS=N`/`--nodes` large enough, or a clean request (the command line cannot grant exclusive access) |
| `run` exits 1, `no_output` | `repeat-NNN/stderr.log`, the command's argv in `status.json` | the command wrote nothing to `{raw_dir}` and no `metrics.json` |
| `run` exits 2, `engine_skew` | the plan's `producer.engine` vs `spyre-bench --help` | a plan made by a different engine version; re-plan |
| `rows_landed: unchecked` | `run.json` `rows_landed.reason` | `--ingest` or `--store` was not given; not a failure, not a pass |
| a variable is missing in the command | `status.json` `notes` | outside the allowlist or secret-looking: `--pass-env NAME` |
| an id changed | `validate --base-dir`, `explain` | an identity field was edited; see above |

More in [guide 11](11-troubleshooting-and-faq.md).
