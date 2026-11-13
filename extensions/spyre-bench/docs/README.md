# spyre-bench concepts

## The model in one page

A **catalog** is a folder of YAML documents. Each document has an envelope (`apiVersion: bench/v1alpha1`, `kind`,
`metadata`, `spec`) and is one of 16 authored kinds. A **Benchmark** is a template: its `matrix`, `variants` or
`derive` block expands into **cases**, and a case's `benchmark_id` is derived from what it asks, never minted.
Tool-written kinds (`Run`, `Result`, `InvalidationEvent`) are refused in source files.

A **pack** registers types, attributes, metrics, device classes and vocabularies without adding kinds. The built-in
packs are `core`, `llm-serving`, `spyre`, `kernel` and `comms`; `examples/packs/postgres-example` is a
third-party pack registered at run time with `--pack`.

## Kinds

| Kind | Purpose |
|---|---|
| Project | catalog root: packs and versions, groupings, series labels |
| Suite | folder of Benchmarks and its policy |
| Selection | a named set of cases, with a reuse rule |
| Benchmark | a template that expands into cases |
| System | the configuration under test (`type` from a pack, `base`/`patch`) |
| Workload | the load put on a System (`inputs`, `load`, `slo`) |
| Asset, Dataset, Model, Store | physical data, its semantics, weights, and how to reach them |
| Environment | the request: devices, interconnect, host, labels, validity floor |
| NodeClass | a class of hosts, its expected profile and qualification policy |
| ValidityPolicy | per purpose, the gates a result must pass to earn `valid` |
| Expectation | how to judge one metric of a set of cases |
| Analysis | a query over stored results |
| Pack | registration of types, attributes, metrics and device classes |

The per-kind JSON Schemas are in `spyre_bench/core/schemas/`; they are generated from `spyre_bench/gen_schemas.py`.

## Identity

| Id | Derived from |
|---|---|
| `benchmark_id` | component (the Project), benchmark name, identity tags, flattened identity attributes (`case.*`, `system.*`, `input.*`) |
| `env_id` | the fingerprint attributes only, no names or serials |
| `placement_id` | the sorted nodes and devices a repeat ran on |
| `series_key` | `benchmark_id` plus the Project's `series_labels` values |

A change to a Workload or System field is a new benchmark. Repeats, groupings, purpose, profile level, `backend` and
unhashed extension fields are run properties and never change the id. Durations hash in seconds (`5m` equals `300s`).
`spyre-bench validate --base-dir` reports series that restart or end between two trees.

## Naming

- `metadata.name` matches `^[a-z0-9]([a-z0-9._@-]{0,126}[a-z0-9])?$`; case names may also use `=`, `/` and `,`.
- A Benchmark file is named `<benchmark-name>.yaml` and lives in its Suite folder; shared documents live in
  `project.yaml`, `stores.yaml`, `assets.yaml`, `datasets.yaml`, `models.yaml` and `workloads.yaml`.
- Fields are snake_case. Attributes and metrics are dot-namespaced (`system.tp`, `llm.itl.median`) with no unit in the
  name; units are UCUM in the metric registry and values are stored in base units.
- Durations are strings with a unit (`90s`, `10m`, `2h`). Tool flags keep their native spelling under `flags`.
- YAML anchors and aliases are refused; use `base:` and `patch:`. Quote `"off"` and `"on"`: YAML 1.1 reads them as
  booleans.
- Renaming a Benchmark changes its `benchmark_id`; list the old name in `previous_names` to keep history linked.

## Planning

`engine.plan(cat, request, leg, existing, invalidations, now)` returns one record per case: `run`, `reuse`, `skip` or
`missing`, with the reason. Reuse needs the same subject and environment (or the Selection's label keys), a validity
at least as high as requested, enough un-invalidated repeats and an age within `max_age`. Helpers:
`engine.excluded_repeats`, `engine.active_invalidations`, `engine.qualification_current`, `engine.validity_of`,
`engine.select_baseline` and `engine.judge`.

## Running the tests

See the README. The suite covers the schemas, every pack, the examples, identity, planning and reuse, invalidation,
qualification, validity and verdicts, result id recomputation and about 90 negative cases (each edits a copy of an
example and asserts the validator's message). Tests that edit examples work on copies in a temp directory.
