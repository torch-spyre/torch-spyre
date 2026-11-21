# 11. Troubleshooting and FAQ

Every error message below is real and every example is checked by the snippet test.

## Validation errors

### `error: ... YAML anchors and aliases are refused`

Use `base:` and `patch:` instead of `&anchor` and `*alias`. The engine reads YAML 1.1 and refuses aliases so a document
means what it says on one screen.

```yaml validated base=docs-benchmark as=extra.yaml expect="error:YAML anchors and aliases are refused"
apiVersion: bench/v1alpha1
kind: Suite
metadata: {name: a}
spec: {type: perf, note: &n x, again: *n}
```

### `$.metadata.name: 'Bad_Name' does not match ...`

Names are lower-case, start and end with a letter or digit, and may contain `.`, `_`, `@` and `-`.

```yaml validated base=docs-benchmark as=extra.yaml expect="error:does not match '^[a-z0-9]"
apiVersion: bench/v1alpha1
kind: Suite
metadata: {name: Bad_Name}
spec: {type: perf}
```

### `system 'nope': no System of that name` (and the same for Environment, Store, Asset, Dataset, Model)

A reference names something that is not loaded. Check the spelling, that the file is under the catalog directory, and
that a shared document is reachable (`--central`, or a Store defined in `stores.yaml` of the same catalog).

```yaml validated base=docs-benchmark as=ops/op-x.yaml expect="error:no System of that name"
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-x}
spec:
  type: perf
  level: kernel
  tier: sanity
  system: nope
  workload: {type: kernel/op, op: x, dtype: fp32, shape: [4], load: {executor: iterations, iterations: 5}}
  variants: [{name: op_x}]
```

### `$.spec.dtype: 'int3' is not one of [...]`

The Workload type's schema (from the pack) rejected a value. The path says which field; the list says what is allowed.

```yaml validated base=docs-benchmark as=ops/op-x.yaml expect="error:is not one of ['fp32', 'bf16', 'fp16', 'fp8', 'int32', 'int8']"
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-x}
spec:
  type: perf
  level: kernel
  tier: sanity
  system: op-lib-cpu-fp32
  workload: {type: kernel/op, op: x, dtype: int3, shape: [4], load: {executor: iterations, iterations: 5}}
  variants: [{name: op_x}]
```

### `FAIL ...: N docs, 0 cases` with one error

One schema error stops case expansion for the whole catalog, so "0 cases" is a symptom, not a second problem. Fix the first
`error:` line and run again.

### `metric 'kernel.speed' is not registered (did you mean ...?)`

Register it in a pack ([guide 5](05-add-a-metric.md)) or fix the spelling.

```yaml validated base=docs-benchmark as=ops/op-x.yaml expect="error:did you mean 'kernel.time.mean'?"
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-x}
spec:
  type: perf
  level: kernel
  tier: sanity
  system: op-lib-cpu-fp32
  metrics: {required: [kernel.speed]}
  workload: {type: kernel/op, op: x, dtype: fp32, shape: [4], load: {executor: iterations, iterations: 5}}
  variants: [{name: op_x}]
```

### `same benchmark_id as case '...': they differ only in run attributes`

Two cases differ only in `tier`, `environment`, `profile` or `run`. Those are run attributes, not identity, so the cases
are the same benchmark. Make the difference part of the workload or system, or use `plan --environment` to run one
case under several Environments ([guide 2](02-define-a-benchmark.md)).

### `$.spec: unknown key 'schedule'`

Cadence, preferred pools, priority and budgets are not part of a Benchmark. Keep `est_hours`; the invoker owns the rest.

### `$.spec.run.command: '...' is not of type 'array'`

`command` is an argv array, not a shell string ([guide 8](08-run-commands-safely.md)).

### `... is not of type 'string'` on a `sha256`

An all-digit digest was read as a number. Quote it ([guide 4](04-data-assets-models-stores.md)). Likewise quote `"on"`,
`"off"` and version strings such as `"0.2"`.

### `... needs an accelerator request` and `labels.cpu_pinning is derived`

See [guide 3](03-environments-and-node-classes.md): `device_local` and `per_device` need an accelerator device request,
and `env.label.cpu_pinning` comes from `host.cpu_pinning`.

### `metric ... is category quality, not valid for type perf`

A perf Benchmark accepts `performance`, `efficiency` and `diagnostic` metrics; a quality Benchmark accepts `quality` and
`diagnostic`. Change the metric's category or the test type.

### `Workload type 'x/y' is not registered`

The pack that registers the type is not loaded. Pass `--pack DIR` and list the pack in the Project's `packs`.

## Planning errors

### `error: selection is empty`

Most often `--arch` is missing or names another architecture: a case whose Environment names an architecture is only selected when `--arch` (default: the host's) matches it. Next
suspects: a `--select`, `--tier` or `--suite` that matches nothing.

```bash validated
spyre-bench plan --catalog examples/docs-benchmark --purpose pr --select smoke --arch s390x \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-none.json # exit=2
spyre-bench plan --catalog examples/docs-benchmark --purpose pr --select nothing --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-none.json # exit=2
```

The second command fails differently: `unknown case or selection 'nothing'` names the thing you asked for.

### `command token {case.nope} is neither a case value nor a run token`

A token in `run.command` is spelled wrong or names a value the case does not have. The error lists the run tokens;
`spyre-bench explain <case>` lists the case values under `identity.disc`.

### A case is `skip` with `capacity` or `reference_node_required`

The leg does not offer what the Environment asked for. Add `--devices CLASS=N`, `--nodes N` or `--reference-node` to
describe the leg, or choose a smaller Environment ([guide 3](03-environments-and-node-classes.md)).

### A case is `run` with reason `no_result_for_case`

`--reuse` was given but the store holds no matching result. With `--store none` (the default) nothing is ever found.

## Run errors

| Message or state | Cause | Fix |
|---|---|---|
| case `failed`, `repeats_failed` | the command exited non-zero or timed out | `repeat-NNN/stderr.log` and `status.json` |
| case `failed`, `no_output` | exit 0, nothing in `{raw_dir}`, no `metrics.json` | write output, or `run.allow_empty: true` if intended |
| `engine_skew` (exit 2) | the plan came from another engine version | re-plan with the engine you run |
| `run --ingest` exits 2 before running | the ingest package is not importable | `--ingest-python` or `--ingest-pythonpath` |
| `rows_landed: unchecked` | no `--ingest` or no `--store` | pass both to check; unchecked is not passed |
| `no_rows_landed` (exit 1) | the ingest ran but the store lacks the case's rows | read `ingest/ingest-N.log` |
| `command holds <redacted>: this is a redacted copy` | a redacted plan was fed to `run` | use the `plan --out` file, not `plan.redacted.json` |
| exit 5 | the ingest, the store or the disk failed | the message names which |

## FAQ

**Why did my benchmark's id change when I only fixed a typo in a comment?** It did not: comments, `description`,
`owner` and annotations are not identity. Run `spyre-bench validate --base-dir` to see what did change.

**Can one Benchmark run under several Environments?** Yes, as separate runs: choose the Environment per request with
`plan --environment`. They are not separate cases because the Environment is not identity.

**Where do I put the number for a threshold?** In an Expectation (`tolerance.pct`, or a `fixed` baseline), next to the
Benchmark or in the central catalog. Not in the Benchmark, and not in a Workload.

**How do I add a metric to results that come from an existing producer?** Register it ([guide 5](05-add-a-metric.md)),
emit it in the sidecar, then check the profile's ingest path stores it; fixed-field producers need an ingest change.

**Can I run a benchmark without the ingest installed?** Yes. `validate`, `explain`, `plan` (engine-mode ids),
`run` without `--ingest` and `analyze` need only the engine. Ingest-mode ids need `spyre-bench[ingest]`.

**What is the difference between a Dataset and an Asset?** An Asset is bytes (digest, location). A Dataset is what the
bytes mean and which part of them is used (role, schema, subsets). A change of bytes is a new Asset; a different slice
of the same bytes is a new subset.

**Why does `validate` pass but `plan` fail?** Some checks need a case in context: command tokens, reuse, capacity.
Validate catches schema, references and registry rules; plan catches the rest.

**May a pack edit a built-in pack?** No. Add your own prefix and `depends` on it ([guide 7](07-write-a-pack.md)).

**Where do cadence and budgets go?** To whatever invokes the engine. A Benchmark carries only `est_hours`.

**Can I put real measurements in an example?** Not in this repository. Examples are synthetic; keep real figures,
hostnames and internal system names out of every public file.
