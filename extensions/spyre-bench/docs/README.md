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
| `benchmark_id` | engine mode: component (the Project), benchmark name, identity tags, flattened identity attributes (`case.*`, `system.*`, `input.*`); ingest mode: the ingest derivation of the Benchmark's `identity` block (below) |
| `env_id` | the fingerprint attributes only, no names or serials |
| `placement_id` | the sorted nodes and devices a repeat ran on |
| `series_key` | `benchmark_id` plus the Project's `series_labels` values |

A change to a Workload or System field is a new benchmark. Repeats, groupings, purpose, profile level, `backend` and
unhashed extension fields are run properties and never change the id. Durations hash in seconds (`5m` equals `300s`).
`spyre-bench validate --base-dir` reports series that restart or end between two trees.

## Ingest identity

By default a case's `benchmark_id` is the engine's own hash of `case.*`, `system.*` and `input.*` values. Results that
must join rows the ingest already stores need the ingest's id, which hashes a different set of inputs. A Benchmark
states how, with an `identity` block:

```yaml
identity:
  mode: ingest              # engine (the default when the block is absent) | ingest
  profile: perf-kernel      # which ingest producer's keys: perf-kernel (default) | vllm | vllm-sweep | generic
  component: demo-ops       # optional; default the Project's component
  name: softmax             # optional; default the Benchmark name
  tags: []                  # identity tags only; the perf-kernel and vllm producers write none
  discriminators:           # the profile's keys, in its order
    - {key: record_type, value: model}
    - {key: config_name, from: case.dtype}
    - {key: input_shapes, from: case.shape, join: "_"}
    - {key: run_mode, value: inference}
    - {key: kernel_name, value: ""}
    - {key: is_total, value: ""}
```

- `value` is a literal; `from` names a `case.*`, `system.*` or `input.*` identity value of the case (the keys
  `spyre-bench explain` lists under `identity.disc`). Exactly one of `value` and `from`. A list value needs `join`
  (the ingest writes scalars, never a JSON list); a mapping cannot feed a discriminator. Authoring trap: PyYAML reads
  `1e5` as a string but a JSON file reads it as the float `100000.0`, which is written `100000.0`; use plain integers or
  quote the value.
- **Profiles.** Each ingest producer hashes its own key list, and the ingest hashes every listed key (an absent one as
  empty), so the block must list exactly the profile's keys in order (`value: ""` for an absent one: the perf-kernel
  op-level record leaves `kernel_name` and `is_total` absent, a kernel record always has a kernel name); any other list
  derives an id that producer never writes. `perf-kernel` is the six keys of the kernel report producer
  (`engine.INGEST_DISC_KEYS`), `vllm` the five of the serving-benchmark producer, `vllm-sweep` those five plus four
  sweep keys. These lists live in `engine.ID_PROFILES`; `ingest_adapter.profile_keys` reads the ingest's own constants
  and `ids` refuses to derive when they differ (see "drift guard" below for the three non-matching outcomes). `generic` has no fixed list: the keys are the prop names the block gives,
  listed sorted and unique (this covers bundle uploads and manual rows, where the producer hashes `sorted(props)`; an
  empty list keys the benchmark by name alone).
- **Not expressible.** Compiled-kernel rows whose `kernel_name` the ingest rewrites to `<stem>@<rank>` (the rank depends
  on the measured durations of the whole run) cannot be derived from a definition. Tags on uploaded rows are hashed raw
  by the ingest, so a row that carries legacy or run-context tags (`nightly`, `platform__...`) hashes them while the
  block refuses them: such ids cannot be reproduced (the conformance table has a row that pins the raw hashing).
- `tags` are identity tags. The block refuses a tag that is, or starts with followed by one of `_ - . : =`, a run-level
  grouping name (`tier`, `suite`, `run`, every Project grouping, including names with `_` such as `release_train`) or
  `backend`; so `tier`, `tier__x`, `tier-x`, `tier.x`, `backend__spyre` are refused while `backends`, `backend1`, `be`
  and `release` are not (they are other words). It also refuses tags equal to a grouping value (a tier or Suite name),
  run-context tags (`platform__`, `testtype__`, `cadence__`, the bare legacy aliases) and result tags: groupings are run
  props, `backend` is unhashed. Discriminator keys follow the same rule (delimiters `__ . : =`, and any `backend`
  segment, so `system.backend` and `Backend` are refused; `run_mode` is fine). `metadata.tags` keep selecting cases and
  feed only the engine-native id.
- **Grammar.** The block is narrower than the producers: `component` and `name` are lower-case
  (`^[a-z0-9_][a-z0-9._@:/-]{0,199}$`; underscores allowed so test names like `latency_demo_tp1_in128_out16` fit) and
  tags match `^[a-z0-9][a-z0-9_-]{0,62}$` (both anchored at the end of the string, so a trailing newline is refused),
  where the producers accept any string and the ingest lower-cases and strips it. Consequently the block cannot express
  a producer name with upper-case or non-ASCII letters or spaces, a tag containing `.` or `/`, or a tag equal to a
  grouping name or value (a Suite or tier name); rows like that exist in the ingest and cannot be reproduced.
- A `from` that is not an identity value of the case (for example `system.backend`, a run property) is an error, as is an
  empty `component` or `name`.

`mode: engine`, or no block, keeps the legacy behaviour exactly: the case ids, series keys and env ids of every example
without a block are pinned by `tests/golden/engine-ids.tsv` (rewrite it from `engine_rows()` in
`tests/test_identity.py`, and only for a deliberate id change).

`Catalog.ids(case)` returns `{benchmark_id}` in engine mode and `{benchmark_id, engine_id, id_mode: ingest}` in ingest
mode. `benchmark_id` is the join key (plan, reuse, series key, Results); `Catalog.engine_id(case)` is the engine-native
hash, always available without the ingest and never a join key. `explain` and Results carry `engine_id` only in ingest
mode, where it differs. `Catalog.ingest_inputs(case)` returns the derivation arguments (component, name, tags, disc,
disc_keys, profile) and `Catalog.id_token(case)` an equality token over the exact string the ingest hashes (so ids the
ingest would collide also collide here) for duplicate and change detection; neither needs the ingest. A Result built in
ingest mode stores the ingest `component`, `name`, `tags`, `disc` and `disc_keys` plus `id_mode: ingest`, and keeps the
Benchmark name and case name as `benchmark` and `case` (analyses select Results by `benchmark`, because `name` is now the
ingest name and two Benchmarks may share one). `recompute` verifies the id with the ingest derivation (and raises
`IngestUnavailable` without it, never skipping the check; a Result without `disc_keys` is reported as a mismatch; the
engine-prop overlap check does not apply to an ingest disc). `engine_id` is a label and is not recomputed.

### Adapter contract and the optional extra

`spyre_bench/ingest_adapter.py` is the only module that imports the ingest. The engine never reimplements the ingest
derivation for a stored id.

```text
benchmark_id(component, name, tags, disc, disc_keys) -> str   # BenchmarkId.derive; '' for a blank component or name
  component, name  str, lower-cased and stripped by the ingest
  tags             identity tags, hashed as a deduplicated sorted set (the ingest does not split or filter them)
  disc             {key: str}; disc_keys the hashed keys in order, an absent key hashing as empty
profile_keys(profile) -> tuple | None   # the ingest's own constants; None only when their module is absent (vendored)
available() -> bool        info() -> dict        reset()
```

It relies on `spyre_clickhouse_ingest.identity.BenchmarkId.derive` and `ID_NAMESPACE`, and optionally on
`spyre_clickhouse_ingest.results._V2_BENCH_ID_KEYS` and `spyre_clickhouse_ingest.vllm.BENCH_ID_KEYS`/`SWEEP_ID_KEYS`. A
known-answer probes run once per process (`ingest_adapter.PROBES`): if any no longer derives its id, the signature
changed, or the derivation raises anything, ids are refused with "the ingest rule changed" and `available()` is False.
`IngestUnavailable` carries the install hint.

**What the probes and the table do and do not cover.** The probes pin: tag stripping, lower-casing, deduplication and
sorting, raw hashing of alias-, namespace- and result-looking tags, empty and non-string tags, trailing underscores; value
stripping (spaces, tabs, newlines, leading and trailing) and lower-casing but no casefold/NFC/NFKC, inner whitespace and
newlines kept, bool/float/None stringification, no truncation of values (300 characters) or names (250); keys kept raw
(case, padding, order, duplicates, an absent key hashing empty); component and name stripped and lower-cased; an
all-empty discriminator part still hashing. The table pins what each producer writes (`results._bench_entries`,
`vllm.bench_entries`, `bundle.generic_entries`, `bundle._bench_entry`, including unsorted props, bool/None props and a
vllm `model` extra that is not hashed). They do not prove the ingest is unchanged elsewhere: a different rule on an input
shape none of them exercises (for example a new tag namespace, or a producer that begins to write tags) is invisible
until a row for it exists, and the compiled-kernel `@rank` rewrite is out of scope.

**Drift guard.** `ids --adapter` (and `engine.check_profile`) report per profile: `match`; `skipped (ingest producer
module absent: ...)` when the ingest has no `results`/`vllm` module at all (a vendored identity-only module);
`unverified (producer modules not importable: ...)` when the module exists but cannot be imported, typically because
`clickhouse-connect` is missing; or `MISMATCH`. `ids --adapter` exits non-zero on `MISMATCH` and on `unverified` unless
`--allow-unverified` is given; deriving ids still works when unverified, because only the identity module is needed.

- **Install.** The extra depends on the `spyre-clickhouse-ingest` distribution, which is on no package index: it works
  only from a checkout, where `uv` resolves it from `../clickhouse-ingest` (`uv pip install 'extensions/spyre-bench[ingest]'`
  or `uv run --extra ingest`). `pip install spyre-bench[ingest]` from an index does not work.
- `validate`, `explain`, `analyze`, `gen-schemas` and `ids --inputs` work without it. Without it `explain` prints
  `benchmark_id: null` and `id_unavailable`; `ids` prints engine-mode cases and fails (exit 2) for ingest-mode ones;
  `plan` and `recompute` of an ingest case fail with the hint. Engine-mode cases never need it.
- **Air-gapped kits.** Without an index the kit must carry the ingest derivation itself. Two options: vendor a module
  named `spyre_clickhouse_ingest.identity` that exposes only `BenchmarkId.derive` and `ID_NAMESPACE` (the profile-key
  comparison is then skipped, because the constants live in other modules, so the conformance table below is the only
  guard against drift), or ship the whole ingest package as a wheel built from the same commit. A vendored copy is
  frozen: refresh it with the engine, rerun the table and the probe, and treat a probe failure as an identity change,
  not a bug to silence. Replacing the adapter file wholesale is also supported if it keeps the three functions above.
- `spyre-bench ids --catalog DIR [--inputs] [--json]` prints each case's id or derivation inputs; `ids --adapter` shows
  the adapter status.

### Conformance table

`tests/golden/ingest-ids.tsv` holds rows of (producer, component, name, tags, disc_keys, disc) with the id the REAL
ingest derives. Rows with a producer name (`perf-kernel-op`, `perf-kernel-model`, `perf-kernel-kernel`,
`perf-kernel-total`, `vllm`, `vllm-sweep`, `vllm-sweep-model`, `generic`, `generic-unsorted-props`,
`generic-bool-none`, `generic-empty-props`, `manual`, `manual-unsorted-props`, `manual-name-only`) are produced
by calling the ingest's own entry builders (`results._bench_entries`, `vllm.bench_entries`, `bundle.generic_entries`,
`bundle._bench_entry`) on tiny synthetic inputs, so they record what each producer writes (absent keys stay absent,
producers add no tags). `probe` rows are hand-edited normalisation probes (case, tag order and duplicates, absent keys,
raw hashing of run-context tags, a pair the ingest itself collides). Rows with an `example_case` must equal what the
example's `identity` block resolves to, so the block-to-arguments mapping is checked even without the ingest; the hash
checks run when it is importable and skip with the reason otherwise, and the producer comparison also needs
`clickhouse-connect`. After an intended change, edit the producer inputs in `tests/refresh_conformance.py` (or the probe
rows) and run

```bash
PYTHONPATH=extensions/clickhouse-ingest python tests/refresh_conformance.py [--check]
```

which rewrites the table. A changed id is a changed series: review the diff. When the ingest changes a producer's keys,
update `ID_PROFILES`, the example blocks and the table together.

## Cost hints, not scheduling

The core Benchmark kind carries no cadence. `est_hours` (optional, all cases) and `run.est`/`run.timeout` (per case)
are generic cost hints, never identity. Preferred and fallback placement, priority, budgets and timers belong to the
planner and presets that consume a catalog; a catalog that needs to hand such values over uses `metadata.annotations`,
which the engine neither reads nor requires. A Benchmark never needs them to validate, plan or run.

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
