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

## Host placement and the pinning experiment

Vocabulary (keys only; values are free-form strings, compared as text, and every example here is synthetic):

| Key | Class | Meaning |
|---|---|---|
| `host.sockets`, `host.numa_nodes`, `host.cores_per_socket`, `host.smt`, `host.llc_domains`, `host.sub_numa` | host fingerprint | CPU topology of the node; `sub_numa` is a generic on/off flag for sub-NUMA clustering |
| `device.numa_node`, `device.pcie_width`, `device.pcie_gen` | device fingerprint | where the accelerator attaches; on a multi-device host `numa_node` is a comma-separated list of integers, one per device in device order (`0,1`) |
| `run.exec.cpu_pinning` | run | requested policy: `none`, `device_local`, `per_device`, `explicit` |
| `run.exec.cpu_set` | run | CPUs the benchmarked process tree is allowed on, as observed (canonical range string such as `0-3,8`) |
| `run.exec.cpu_set_source` | run | `explicit`, `device_numa`, `unresolved` or `none` |
| `run.exec.pinning_verified` | run | `true`, `false` or `unknown` (not observable; never `true` by default) |
| `run.exec.pinning_violations` | run | count of threads outside the requested set; absent when `unknown` |

Fingerprint keys are registered as `env.fp.<key>` (for example `env.fp.host.sockets`). These nine topology
attributes are **recorded but not hashed** under `fingerprint_version: 1`, so a probe that starts emitting them leaves
every existing `env_id` (and the reuse and baselines built on it) unchanged. A Project opts in with
`fingerprint_version: 2`, which also hashes them; new Results carry the version they were built with, and
`recompute` honours it. The registry marks them `fingerprint_since: 2` (an engine-owned table). The
`core.env_complete` gate (share of fingerprint attributes present >= 0.8) counts the attributes the active version
hashes, `Registry.fingerprint_keys(version)`, so version 1 completeness is unaffected.

**Request.** `Environment.spec.host.cpu_pinning: {policy, cpus, require_verified}`. `cpus` is a CPU list and only
valid with `explicit`. `device_local` pins to the union of the NUMA-local CPUs of the requested devices; `per_device`
lets each thread stay inside the CPUs local to one device. `require_verified: true` turns an observed violation (or a
placement that could not be observed) into validity `suspect` with reason `pinning_violated` (`pinning_unverified`),
so a verdict is `suspected`, never `regressed`. `build_result(..., pinning=request)` applies this cap from the
repeats' `run.exec.pinning_verified` props, the worst repeat winning (`false`, then `unknown`, then `true`). An
explicit `policy: none` (without `require_verified`) and an absent block are the same request and digest alike.
`build_result` takes the request from the case's Environment unless `pinning=` is passed.

**Identity.** Pinning is a run property, not part of `env_id` (a pinned and an unpinned run on one node class are the
same hardware). The request is exposed as the label `env.label.cpu_pinning` (derived, `none` when the block is
absent). List it in the Project's `series_labels` and the two never share a series key. Reuse and rolling baselines
(`environment: id` and label mode alike) also refuse a stored row whose pinning label differs, whether or not the
label is a series label; a row without the label counts as `none`.

**Observation.** `spyre_bench.placement.verify_pinning(request, pid, device_numa)` reads `/proc` and `/sys` (Linux
only), compares every thread of the process tree with the requested set and returns the `run.exec.*` properties;
`placement.validity_facts` feeds `engine.validity_of`. The tree is the root's descendants plus every process in its
process group, session or (non-root) cgroup, so a re-parented worker is still judged; start your benchmark in its own
session or cgroup so unrelated processes do not count. In the root cgroup (a private cgroup namespace) a cgroup
cannot tie a worker to the root, so any process outside the tree that was re-parented to pid 1 makes the result
`unknown` rather than risk a false `true`. Anything it cannot fully read (an unreadable or malformed entry, an empty
mask, a changed root start time) gives `unknown`; only a process that exits mid-scan is skipped. Pass the root's
`/proc` start time recorded at spawn as `expected_starttime` to also catch pid reuse before the first observation.
Verification covers affinity masks only: a cgroup cpuset narrower than the requested set is not distinguished.
`cpus` accepts ids up to 16383.

**Analysis.** `derive` computes a metric from two cases of one subject and environment, per repeat pair
(`difference`, `ratio`, or `slope`: `(first - second) / (x.first - x.second)`), and reports median, mean and CV over
the pairs plus the same operation on the two aggregates. Cases with a different subject, `env_id`, labels, node or
pinning are refused with the reason; repeats pair by `run.repeat`, and a pair with a zero denominator is skipped
(counted in `skipped_pairs`, listed under `warnings`), as are non-finite samples. `compare` tabulates a metric (its own
values, or the derived one) by an attribute within each `across` combination: n, mean, median, CV, the effect of each
`by` value against the baseline and `within_noise` (`|effect| <= max(CV of the cell, CV of the baseline cell)`; `null`
below two repeats). The baseline is `none` when present, else `compare.baseline` is required. `direction` is the
registry's for the compared metric (`null` when unregistered) and `reading` says `better` or `worse` accordingly. A
cell holds exactly one Result (or pair): Results of another subject, `env_id`, node or case, or repeated runs of one
case, are refused; add `env.node` to `across` to keep nodes apart. The `by` cells of one `across` combination must
also be comparable with each other apart from pinning (same subject, `env_id`, labels and node), and a Result whose own
repeats disagree on pinning is refused. Derived values that are not finite are skipped like zero denominators, a
repeated `run.repeat` inside a Result is an error, and `reading` is `null` while `within_noise` is `null`.

```bash
spyre-bench validate examples/pinning-demo
spyre-bench analyze pinning-effect --catalog examples/pinning-demo --results results.json [--json]
```

`examples/pinning-demo` is the end-to-end pattern: one latency Benchmark at two output lengths, node-a and node-b
Environments with and without `device_local` pinning, and the `pinning-effect` Analysis. The Results come from the
launcher; the tests build synthetic ones.
