# 1. Concepts and the sixteen kinds

This guide is the map. Read it once, then use the others as recipes.

## The pipeline in one picture

```text
definitions (YAML)  ->  plan.json  ->  run (one process per repeat)  ->  Result  ->  ingest  ->  verdict
 what to measure        what to do      raw output + metrics.json       one per       rows in     pass / regressed /
 and what it needs      per case:       in results/bench/               case          the store   improved / ...
                        run/reuse/skip
```

- **Definitions** are YAML documents in a *catalog* (a folder). `spyre-bench validate` checks them with no hardware and
  no network.
- **A plan** (`spyre-bench plan`) turns a request (purpose, subject, selection) into one record per case: run, reuse a
  stored result, or skip with a reason. It never runs anything.
- **A run** (`spyre-bench run`) executes the planned commands, one subprocess per repeat, and writes a Result per case.
  The engine writes no database rows; it hands raw output to an ingest step and then checks the rows landed.
- **A verdict** (`spyre-bench verdict`) prints what the store's views decided. The engine never decides a verdict.

Who starts the pipeline (a CI workflow, a scheduler, a person) does not change what a result means. A result attaches
to the **subject** (the build under test), is compared by environment fingerprint, and records who asked as a property,
never as part of the series.

## An envelope, sixteen kinds

Every document has the same envelope:

```yaml validated in=docs-benchmark/project.yaml
apiVersion: bench/v1alpha1
kind: Suite
metadata: {name: ops, description: "one Benchmark per operation"}
spec: {type: perf, pr_selectable: true}
```

`kind` is one of sixteen authored kinds. Three more (`Run`, `Result`, `InvalidationEvent`) are written by the tool and
refused in source files:

```yaml validated base=docs-benchmark as=extra.yaml expect="error:is tool output"
apiVersion: bench/v1alpha1
kind: Result
metadata: {name: x}
spec: {}
```

| Kind | What it is | Defined in |
|---|---|---|
| Project | the catalog root: component name, packs and versions, groupings (`tier`, `suite`, ...), series labels | the repository that owns the code under test; one in a central catalog too |
| Suite | a folder of Benchmarks and its policy (`type`, `pr_selectable`, `requires_reference`) | the repository |
| Selection | a named set of cases (explicit names or a selector) with a reuse rule and run defaults | the repository |
| Benchmark | a *template*: one System, one Workload, a matrix or variants that expand into cases | the repository, next to the code it measures |
| System | the configuration under test (a typed spec: library, precision, tensor-parallel size, ...) | with the Benchmark, or shared |
| Workload | the load put on a System: inputs, load executor, SLO; names no hardware | with the Benchmark, or shared |
| Asset | physical data: where the bytes are, how to verify them | the repository |
| Dataset | semantics over assets: role, schema, named subsets | the repository |
| Model | weights (and tokenizer, config, adapters) by role, as assets | the repository |
| Store | how to reach data: endpoints and *references* to credentials, never values | the repository or central |
| Environment | the *request*: devices, interconnect, host, labels, validity floor | central catalog (a repository may define a local one) |
| NodeClass | a class of hosts: match rule, expected host profile, qualification policy | central catalog |
| ValidityPolicy | per purpose, the gates a result must pass to earn `valid` | central catalog |
| Expectation | how to judge one metric of a set of cases | next to the Benchmark, or central |
| Analysis | a query over stored results: derive a number, compare across a variable | the repository |
| Pack | registers types, attributes, metrics, device classes, probes and gates | with the engine, or a directory given with `--pack` |

"Central" means a second catalog directory passed with `--central DIR` (or `BENCH_CENTRAL`). It exists so that the
facts about hosts and policy are written once and reviewed by the people who own them, while each repository keeps the
definitions of what it measures.

## Cases and identity

A Benchmark is a template. Its `matrix`, `variants` or `derive` block expands into **cases**, and a case is what is
planned, run and judged. Its stored id, `benchmark_id`, is *derived* from what the case asks, never minted:

| Id | Derived from |
|---|---|
| `benchmark_id` | the Project's `component`, the Benchmark name, identity tags and the flattened identity values (`case.*`, `system.*`, `input.*`) |
| `env_id` | the environment fingerprint only: no names, no serials |
| `placement_id` | the sorted nodes and devices a repeat ran on |
| `series_key` | `benchmark_id` plus the values of the Project's `series_labels` |

So a change to a workload or system field is a **new benchmark** with a new history. Run attributes never change the
id: repeats, `tier`, `suite`, purpose, profile level, `backend`, the `run` block (command, timeout, estimate), the
environment name. Two cases of one Benchmark that differ only in such an attribute are refused as duplicates:

```bash validated
spyre-bench explain op_scale_n1024 --catalog examples/docs-benchmark
```

Read the `identity:` block of that output to see exactly which values were hashed, and `run_props:` for the ones that
were not. Guide 2 covers the rest, including how to reproduce ids the ingest already stores (`identity: {mode: ingest}`).

## Layers and who owns them

| Layer | Question it answers | Examples | Changes when |
|---|---|---|---|
| Definition | what is measured, on what, needing what | Benchmark, System, Workload, Environment | the code or hardware request changes |
| Policy | what counts as trustworthy and as a regression | ValidityPolicy, NodeClass, Expectation | the team agrees a new bar |
| Request | what is wanted now | purpose, subject id, selection, reuse rule, budget | every invocation |
| Plan | what will happen | `plan.json` | every invocation |
| Evidence | what happened | `run.json`, `result.json`, rows in the store | every run |

Anything that decides *when* (cadence, timers, priority, a preferred pool, a budget) is a request-layer concern for
the invoker. The only scheduling-shaped field a Benchmark may carry is the cost hint `est_hours`; a `schedule` key is
refused:

```yaml validated base=docs-benchmark as=ops/op-x.yaml expect="error:unknown key 'schedule'"
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: op-x}
spec:
  type: perf
  level: kernel
  tier: sanity
  system: op-lib-cpu-fp32
  schedule: {preferred: nightly}
  workload: {type: kernel/op, op: x, dtype: fp32, shape: [4], load: {executor: iterations, iterations: 5}}
  variants: [{name: op_x}]
```

## Purposes, validity and trust

A request has a **purpose** (`pr`, `nightly`, `weekly`, `release`, `reference`, `qualification`, `investigate`). The
ValidityPolicy says, per purpose, what a result must have to be `valid`: a current node qualification, a clean node,
enough repeats, a low enough coefficient of variation, discarded warm-up, observed placement. A result that misses a
gate is capped at a lower validity (`shared_env`, `unqualified_env`, `suspect`, `invalid`, `external`), never silently
promoted. Only `valid` results of a baseline purpose become baselines; reuse needs validity at least the purpose's
floor. Guide 3 and guide 6 cover both sides.

## Naming and file conventions

- `metadata.name` matches `^[a-z0-9]([a-z0-9._@-]{0,126}[a-z0-9])?$`; case names may also use `=`, `/` and `,`.
- A Benchmark file is `<benchmark-name>.yaml` in its Suite folder; the folder name is the primary `run.suite.<name>`.
  Shared documents live in `project.yaml`, `stores.yaml`, `assets.yaml`, `datasets.yaml`, `models.yaml` and
  `workloads.yaml`.
- Fields are snake_case. Attributes and metrics are dot-namespaced (`system.tp`, `llm.itl.median`) with no unit in the
  name. Durations are strings with a unit (`90s`, `10m`, `2h`).
- YAML anchors and aliases are refused; use `base:` and `patch:`. Quote `"off"` and `"on"` (YAML 1.1 reads them as
  booleans) and quote hex digests made only of digits.
- Renaming a Benchmark changes its `benchmark_id`; list the old name in `previous_names` to keep history linked.

```yaml validated base=docs-benchmark as=extra.yaml expect="error:YAML anchors and aliases are refused"
apiVersion: bench/v1alpha1
kind: Suite
metadata: {name: a}
spec: {type: perf, note: &n x, again: *n}
```

## Packs: how the engine learns a domain

The kinds are fixed. What varies by domain lives in packs: the `type` strings you can put in a System or Workload
(`kernel/op`, `llm/serve`), the attribute and metric registry, device classes, runners, probes and gates. The engine
ships `core`, `llm-serving`, `spyre`, `kernel` and `comms`; `examples/packs/docs-queue` is a small third-party pack
registered with `--pack`. A Project lists the packs (and a compatible version range) it needs:

```yaml validated in=docs-benchmark/project.yaml
apiVersion: bench/v1alpha1
kind: Project
metadata: {name: docs-benchmark, owner: example-team}
spec:
  component: docs-benchmark
  packs: [{name: kernel, version: "0.2"}]
  default_suites: [ops]
  default_environment: one-socket
  visibility: public
  groupings:
    - {name: tier, ordered: true, cardinality: one, values: [sanity, core, extended]}
    - {name: suite, ordered: false, cardinality: many}
  series_labels: [env.label.node.class]
```

`series_labels` decides which Environment labels split series: two runs that differ in a listed label never share a
baseline. Choose them for facts that change the number (device count, node class, CPU pinning), not for bookkeeping.

## Where to go next

Start with [guide 2](02-define-a-benchmark.md). If you only need to change hardware requests, read
[guide 3](03-environments-and-node-classes.md). The JSON Schemas of every kind are in `spyre_bench/core/schemas/`
(generated by `spyre-bench gen-schemas`); point your editor's YAML language server at
`resources.schema.json` for completion, as the first line of each example file does.
