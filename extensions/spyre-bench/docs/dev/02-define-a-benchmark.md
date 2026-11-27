# 2. Define a Benchmark end to end

You will add a benchmark called `op-scale` (an elementwise scale of a vector) to a catalog, give it a command to run,
and take it from `validate` to a finished run on a laptop with a stub command. The files are in
`examples/docs-benchmark`; every block below is checked against them.

```text
examples/docs-benchmark/
  project.yaml            Project, Suite, Selections, NodeClass, Environment
  ops/op-scale.yaml       System + Benchmark (the file you are writing)
  tools/emit_metrics.py   the command: writes raw output and a metrics.json sidecar
```

## Step 1. The Project, the Suite and the Selections

A catalog needs one Project. It names the `component` (hashed into every id, so keep it stable), the packs, the
groupings and the series labels (see [guide 1](01-concepts.md)). The folder a Benchmark file sits in is its Suite:

```yaml validated in=docs-benchmark/project.yaml
apiVersion: bench/v1alpha1
kind: Selection
metadata: {name: smoke, description: "one case, one repeat; never ingested"}
spec:
  cases: [op_scale_n1024]
  run: {repeats_min: 1}
---
apiVersion: bench/v1alpha1
kind: Selection
metadata: {name: ops-sanity}
spec:
  selector: {types: [perf], groupings: {tier: {at_most: sanity}}}
  reuse: {mode: if_valid, max_age: 7d}
```

Every repository defines a Selection named `smoke`: one case, one repeat, cheap enough for a developer to run by hand.
A Selection names cases explicitly (`cases`) or by a `selector` over types, levels, suites, tags and groupings, never
both. `tier` is an ordered grouping, so `at_most: sanity` selects the cheapest tier.

## Step 2. The System: the thing under test

A System is a typed spec. Its `type` (`kernel/op-library`) comes from a pack and decides which fields are allowed; the
engine validates the rest against that type's JSON Schema. Two fields are reserved and *not* hashed: `backend` (the
comparison axis, so a CPU result and an accelerator result of the same case sit side by side) and `extensions`.

```yaml validated in=docs-benchmark/ops/op-scale.yaml
apiVersion: bench/v1alpha1
kind: System
metadata: {name: op-lib-cpu-fp32}
spec: {type: kernel/op-library, backend: cpu, token: cpu-fp32, library: torch-cpu, precision: fp32, harness: pytorch-benchmark}
```

Everything else in a System is **identity**: change `precision` and you have a different benchmark.

## Step 3. The Benchmark

```yaml validated file=docs-benchmark/ops/op-scale.yaml
# yaml-language-server: $schema=../../../spyre_bench/core/schemas/resources.schema.json
apiVersion: bench/v1alpha1
kind: System
metadata: {name: op-lib-cpu-fp32}
spec: {type: kernel/op-library, backend: cpu, token: cpu-fp32, library: torch-cpu, precision: fp32, harness: pytorch-benchmark}
---
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-scale, owner: example-team, description: "elementwise scale of a vector"}
spec:
  type: perf
  level: kernel
  tier: sanity
  system: op-lib-cpu-fp32
  est_hours: 0.05
  workload:
    type: kernel/op
    op: scale
    dtype: fp32
    shape: [1024]
    load: {executor: iterations, iterations: 50, warmup_iterations: 5}
  metrics: {required: [kernel.time.mean], headline: [kernel.time.mean]}
  run:
    repeats: 3
    timeout: 5m
    est: 1m
    isolation: process
    command: ["{python}", "tools/emit_metrics.py", "--op", "{case.op}", "--shape", "{case.shape}", "--repeat", "{repeat}", "--raw", "{raw_dir}", "--out", "{repeat_dir}"]
  matrix:
    name_template: op_scale_n{size}
    axes:
      size:
        - {token: "1024", patch: {workload: {shape: [1024]}}}
        - {token: "65536", patch: {tier: core, workload: {shape: [65536]}}}
```

What each field decides:

| Field | Meaning | Identity? |
|---|---|---|
| `type`, `level` | what kind of test (`perf` or `quality`) and which stack level it measures (`kernel`, `layer`, `model`, `service`, ...) | `type` yes (as `case.type`) |
| `tier` | an ordered grouping value; selects "cheap to expensive" sets | no (a run property) |
| `system`, `workload` | what is measured and under what load | yes |
| `environment` | the Environment *request* (devices, host); omit to use the Project's `default_environment` | no |
| `metrics.required` | metrics that must be present for a case to count; `headline` marks the ones to show first | no |
| `profile` | which profile levels (`off`, `light`, `full`) the case supports | no |
| `env` | process environment knobs (`{DEMO_FLAG: 64}`); run attributes | no |
| `run.repeats`, `cache`, `isolation`, `timeout`, `est` | how many times, warm or cold, process or session, cost hints | no |
| `run.command`, `run.ingest` | what to execute; see below | no |
| `est_hours` | cost hint for whoever plans (wall-clock hours of all cases) | no |
| `relations`, `previous_names` | links to related Benchmarks; old names that keep history linked | no |
| `identity` | how the stored id is derived; see "Reproducing ids the ingest already stores" | n/a |

### Expanding into cases

Exactly one of `variants`, `matrix` or `derive` expands the template.

- **`variants`** lists cases by name, each with optional overrides (a different `system`, `workload`, `tier`, `run`).
  Use it when the cases are a short, irregular list. A variant can carry `previous_names`.
- **`matrix`** is the cross product of axes. An axis is a list of `{token, patch}` values, where `patch` is a JSON
  Merge Patch (RFC 7386) over the Benchmark spec, or a numeric sweep `{path, values | range, token}`. `name_template`
  builds the case name from the tokens (and `{system.*}`, `{workload.*}`, `{benchmark}`); `exclude` drops combinations.
  Lists replace under a patch, so write the shape in full (`shape: [65536]`).
- **`derive`** makes one case per distinct System of the cases it selects, for example a startup benchmark per decoder
  System. The Benchmark below lives in the `systems` Suite folder and takes its Systems from the `ops` Suite:

```yaml validated base=docs-benchmark as=systems/op-scale-systems.yaml
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-scale-systems, owner: example-team}
spec:
  type: perf
  level: kernel
  tier: core
  system: op-lib-cpu-fp32
  workload: {type: kernel/op, op: scale, dtype: fp32, shape: [8], load: {executor: iterations, iterations: 5}}
  derive:
    from: {suites: [ops]}
    per: system
    name_template: scale_systems_{system.token}
```

Two cases that differ only in run attributes are the same benchmark and are refused. `tier`, `environment`, `profile`
and the `run` block are run attributes, so a matrix over them does not create cases (use `plan --environment` to run
one case under several Environments instead):

```yaml validated base=docs-benchmark as=ops/op-dup.yaml expect="error:they differ only in run attributes"
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-dup}
spec:
  type: perf
  level: kernel
  tier: sanity
  system: op-lib-cpu-fp32
  workload: {type: kernel/op, op: dup, dtype: fp32, shape: [4], load: {executor: iterations, iterations: 5}}
  variants:
    - name: dup_fast
    - name: dup_slow
      run: {timeout: 20m}
```

## Step 4. The command

`run.command` is an argv array, never a shell string; a string is refused at validation:

```yaml validated base=docs-benchmark as=ops/op-x.yaml expect="error:is not of type 'array'"
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-x}
spec:
  type: perf
  level: kernel
  tier: sanity
  system: op-lib-cpu-fp32
  workload: {type: kernel/op, op: x, dtype: fp32, shape: [4], load: {executor: iterations, iterations: 5}}
  run: {command: "python tools/x.py"}
  variants: [{name: op_x}]
```

Tokens are written `{name}`; `{{` and `}}` are literal braces. At **plan** time the engine fills in the case values
(`{case.op}`, `{case.shape}`, `{system.precision}`; a list joins with commas) and refuses a token it cannot resolve.
At **run** time it fills in `{python}`, `{repeat}`, `{raw_dir}`, `{repeat_dir}`, `{out_dir}` and the rest. The full
table is in the [reference](../README.md#command-templates).

```bash validated
spyre-bench validate examples/docs-benchmark
spyre-bench plan --catalog examples/docs-benchmark --purpose pr --select smoke --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan.json
```

```json validated subset-of=plan.json
{
  "plan_schema": "bench.plan/1",
  "purpose": "pr",
  "summary": {"cases": 1, "run": 1, "reuse": 0, "skip": 0},
  "cases": [
    {
      "name": "op_scale_n1024",
      "action": "run",
      "environment": "one-socket",
      "repeats": 1,
      "labels": {"env.label.node.class": "cpu-pool"},
      "command": ["{python}", "tools/emit_metrics.py", "--op", "scale", "--shape", "1024"]
    }
  ]
}
```

The case values are already substituted and the run tokens are still literal; `run` fills those per repeat. The
Selection's `repeats_min: 1` lowered `repeats` from 3 to 1.

Your command's contract is small:

1. Write whatever the ingest will read into `{raw_dir}` (shared by all repeats of the case).
2. Write per-repeat evidence into `{repeat_dir}`, above all `metrics.json`, the **sidecar** the engine builds the
   Result from. The engine parses no producer format itself.
3. Exit 0 on success. A command that exits 0 but leaves `raw/` empty and writes no `metrics.json` is a failed case
   (`no_output`) unless the Benchmark says `run.allow_empty: true`.

```json validated sidecar
{"samples": {"kernel.time.mean": [0.00101]}, "props": {"run.cache": "warm"}, "warmup_discarded": 0}
```

`samples` maps a registered metric name to finite numbers in base units (seconds, bytes). Guide 5 covers how metrics
get registered and how they reach the store. Guide 8 covers what the command can see.

`run.ingest` is the optional argv that hands the raw output to an ingest step for one case, with the same tokens plus
`{run_key}`, `{component}` and `{profile}`. Leave it out to use the default for the case's identity profile.

```yaml validated base=docs-benchmark as=ops/op-ingest.yaml
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-ingest}
spec:
  type: perf
  level: kernel
  tier: sanity
  system: op-lib-cpu-fp32
  workload: {type: kernel/op, op: ingest, dtype: fp32, shape: [4], load: {executor: iterations, iterations: 5}}
  run:
    command: ["{python}", "tools/emit_metrics.py", "--op", "{case.op}", "--repeat", "{repeat}", "--raw", "{raw_dir}", "--out", "{repeat_dir}"]
    ingest: ["{python}", "-m", "my_ingest", "--raw", "{raw_dir}", "--run-key", "{run_key}", "--case", "{case}"]
  variants: [{name: op_ingest}]
```

## Step 5. Run it

```bash validated
spyre-bench run --plan plan.json --catalog examples/docs-benchmark --workdir examples/docs-benchmark --out results/bench
```

`--workdir` is the directory the command runs in, so `tools/emit_metrics.py` resolves. The engine wrote:

```text
results/bench/
  run.json                bench.run/1 manifest: counts, per-case status, ingest, rows_landed, exit_reason
  plan.redacted.json      the plan with secrets redacted (not runnable)
  cases/op_scale_n1024-<hash8>/
    case.json  raw/       the raw output (what the ingest reads)
    repeat-001/           status.json, stdout.log, stderr.log, metrics.json
    result.json           the engine Result
```

```json validated subset-of=results/bench/run.json
{
  "run_schema": "bench.run/1",
  "purpose": "pr",
  "counts": {"planned": 1, "run": 1, "ok": 1, "failed": 0},
  "cases": [{"name": "op_scale_n1024", "status": "ok", "repeats": {"planned": 1, "ok": 1, "failed": 0}}],
  "ingest": {"requested": false, "status": "skipped"},
  "exit_code": 0,
  "exit_reason": "ok"
}
```

## Reproducing ids the ingest already stores

By default a case's `benchmark_id` is the engine's own hash of `case.*`, `system.*` and `input.*`. If your results must
join rows the ingest already stores (an existing series with history), the Benchmark declares an `identity` block and
the engine calls the ingest's own derivation (`spyre-bench[ingest]`, installed from a checkout):

```yaml validated file=docs-ingest-identity/ops/op-scale.yaml
# yaml-language-server: $schema=../../../spyre_bench/core/schemas/resources.schema.json
apiVersion: bench/v1alpha1
kind: System
metadata: {name: op-lib-cpu-fp32}
spec: {type: kernel/op-library, backend: cpu, library: torch-cpu, precision: fp32, harness: pytorch-benchmark}
---
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-scale, owner: example-team}
spec:
  type: perf
  level: kernel
  tier: sanity
  system: op-lib-cpu-fp32
  est_hours: 0.05
  workload: {type: kernel/op, op: scale, dtype: fp32, load: {executor: iterations, iterations: 50, warmup_iterations: 5}}
  identity:
    mode: ingest
    profile: perf-kernel
    discriminators:
      - {key: record_type, value: op}
      - {key: config_name, value: ""}
      - {key: input_shapes, from: case.shape, join: "_"}
      - {key: run_mode, value: op_benchmark}
      - {key: kernel_name, value: ""}
      - {key: is_total, value: ""}
  matrix:
    name_template: op_scale_n{size}
    axes:
      size:
        - {token: "1024", patch: {workload: {shape: [1024]}}}
        - {token: "4096", patch: {workload: {shape: [4096]}}}
```

```bash validated
spyre-bench validate examples/docs-ingest-identity
spyre-bench ids --catalog examples/docs-ingest-identity --inputs
```

The rules that matter when writing the block (the [reference](../README.md#ingest-identity) has all of them):

- **Pick the profile of the producer that writes your rows.** `perf-kernel` is the kernel-report producer (six keys),
  `vllm` and `vllm-sweep` the serving-benchmark producer, `generic` the bundle upload path. The block must list *exactly*
  that profile's keys, in its order; use `value: ""` for a key the producer leaves absent. Any other list derives an id
  nothing ever writes.
- A discriminator is a literal `value` or a `from` naming a `case.*`, `system.*` or `input.*` identity value. List
  values need `join`. Quote numbers that YAML would turn into floats.
- `component` and `name` default to the Project's component and the Benchmark name; they must be lower-case.
- Do not put groupings (`tier`, a Suite name), `backend` or run-context tags in `tags` or in a discriminator key: they
  are run properties, and the block refuses them.
- After any change, run the repository's conformance check (it recomputes ids with the real ingest) and read the diff
  of the golden table. A changed id is a changed series. See [guide 10](10-contribute.md).

Guide 12 has the real `vllm` and `vllm-sweep` blocks (serving benchmark) and the `perf-kernel` block (kernel report),
with `explain` and `ids` output.

## Tiers, suites and reuse

`tier` and `suite` are groupings declared in the Project. They select sets (`--tier`, `--suite`, a Selection's
`groupings: {tier: {at_most: core}}`) and are recorded on the run (`run.tier`, `run.suite.<name>`), never hashed. A
Suite with `pr_selectable: false` is kept out of pull-request selections; `requires_reference: true` makes its cases
skip unless the leg runs on a reference node.

Reuse is a planning rule: instead of running a case, serve a stored result. All of these must hold:

| Condition | Meaning |
|---|---|
| same case | the same `benchmark_id` |
| same subject | the same build under test (`--subject-id`) |
| same environment | the same `env_id`, or the same values of the Selection's `match.environment.labels` |
| same pinning label | a pinned and an unpinned result never substitute for each other |
| validity | at least the purpose's floor |
| repeats | at least as many un-invalidated repeats as requested |
| age | within `max_age` |

A stored row that lacks `env.id` or `run.validity` is unknown and means `run`. The modes are `never` (the default
when nothing says otherwise), `if_valid`, `prefer` and `only`. `--reuse` on the command line overrides a Selection's own
rule; `--force` names cases to rerun. Without a store configured the planner cannot look anything up:

```bash validated
spyre-bench plan --catalog examples/docs-benchmark --purpose nightly --select ops-sanity --reuse if_valid \
    --arch x86_64 --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-reuse.json
```

```json validated subset-of=plan-reuse.json
{"store": "none", "summary": {"run": 1, "reuse": 0}, "cases": [{"name": "op_scale_n1024", "action": "run", "reason": "no_result_for_case"}]}
```

## Checklist before you open a pull request

1. `spyre-bench validate <catalog>` passes (with `--central` and `--pack` as the repository does).
2. `spyre-bench explain <case>` shows the identity you intended; for an `identity` block, `spyre-bench ids --inputs`.
3. `spyre-bench plan` plans the Selection you changed and the reasons look right.
4. The command runs once by hand with a stub or the real tool ([guide 9](09-test-and-debug-locally.md)).
5. Cost hints (`est_hours`, `run.est`, `run.timeout`) are honest; a `timeout` too short fails real runs.
6. The `smoke` Selection still names an existing case.
7. If an id changed on purpose, you know which series restarts ([guide 10](10-contribute.md)).
