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

## Dataset slugs

A public definition names a dataset by an opaque slug (`slot-95`), never by its real name. The artifact store holds
`<slug>/dataset.json` beside the data:

```json
{"display_name": "Demo Questions", "public": true, "license": "MIT", "sha256": "<digest>", "notes": ""}
```

- `Dataset.spec.slug` (default: the name) locates it; `spec.store` (default: the store of its first Asset) says where.
  An Asset may use `source: {store, slug}` in place of `store+path`; the path is then `<slug>`.
- A tool labels a dataset with `display_name` only when `public` is exactly `true`; otherwise it shows the slug.
  `explain --dataset-meta-dir DIR` or `--dataset-store STORE` reads the files; with neither, nothing is read and the
  slug is shown. `validate` never reads a store: a slug is only a string there.
- A missing, unreadable or malformed file, a `slug` field that disagrees with the directory, or a `sha256` that none of
  the Dataset's Assets declares is an error, never a silent fallback (an Asset that declares no digest cannot vouch for
  one: omit `sha256`). Error text carries the slug, not the file.
- Only the dataset label goes through the metadata. Every tool prints Dataset and Asset *names* as written, so name them
  by slug as well. The HTTP source accepts https only and follows no redirects, so the token reaches only the configured host.
- The file is a trusted input: whoever can write the store path decides what is public (a duplicate key is not an error: the
  last one wins). Keep that path write-restricted.
- `dataset.json` is not part of any identity. The store namespace is configured per deployment, never in a public file.
- `spyre-bench dataset add|list|check` creates, lists and verifies all of this in one command: [dataset-add.md](dataset-add.md).

## Metric context

Every metric belongs to a metric class that lists the facets its numbers must carry; see
[metric-context.md](metric-context.md). `validate` warns when a Benchmark does not declare a required author-only facet.

## Kernel pack: dtypes, MoE metrics, trace roles

A kernel `op` workload accepts `dtype` `fp32`, `bf16`, `fp16`, `fp8`, `int32` and `int8`; `df16` is the device-side name
of `fp16` and is written `fp16`. The pack registers `kernel.moe.fetch_bytes` and `kernel.moe.loop_trips` (class
`kernel_time`) and the `role.moe.*` run properties that name which stored trace signal plays which role. A role with no
stored signal reads as not captured, never as zero (`spyre_bench.trace_signals`). A sample is in `examples/moe-demo`.

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

## Planning from the command line

`spyre-bench plan` turns a request into `plan.json`: the cases a selection expands to, their stored ids, and for each
case `run`, `reuse` or `skip` with the reason. It never runs anything.

```bash
spyre-bench plan --catalog benchmarks --central shared --purpose nightly --subject-id <artifact-uuid> \
    --select smoke --reuse if_valid --env-id <env_id> --store clickhouse --out plan.json
```

Flags (each with its `BENCH_*` env twin): `--catalog` (`BENCH_DIR`), `--central` (`BENCH_CENTRAL`), `--pack`
(`BENCH_PACK`), `--purpose` (`BENCH_PURPOSE`), `--environment` (`BENCH_ENVIRONMENT`), `--select`/`--selection`
(`BENCH_SELECTION`), `--cases` (`BENCH_CASES`), `--tier` (`BENCH_TIER`), `--suite` (`BENCH_SUITES`), `--type`
(`BENCH_TYPE`), `--subject-id` (`BENCH_SUBJECT_ID`), `--reuse` (`BENCH_REUSE`), `--force` (`BENCH_FORCE`), `--max-age`
(`BENCH_MAX_AGE`), `--set` (`BENCH_SET`), `--budget` (`BENCH_BUDGET`), `--arch` (`BENCH_ARCH`), `--devices`
(`BENCH_DEVICES`), `--nodes` (`BENCH_NODES`), `--reference-node` (`BENCH_REFERENCE_NODE`), `--env-id`
(`BENCH_ENV_ID`), `--store` (`BENCH_STORE`), `--request` (`BENCH_REQUEST`), `--out` (`BENCH_PLAN_OUT`), `--json`.
Precedence: flag, env twin, request file (`--request`, the engine request shape), default. `--select` and `--cases`
work as a pair: a flag of either beats both env twins, which are not merged with it.

- **Reuse** needs the same subject artifact, the same `env_id` (or the Selection's label keys), the same pinning label,
  validity at least the purpose floor, enough repeats and an age within `--max-age`. A stored row that lacks `env.id` or
  `run.validity` props is unknown and means `run`. `--reuse` on the command line overrides a Selection's own rule
  (`reuse.override: true` in the request); `--force` and `--max-age` need a reuse mode too. No reuse setting anywhere is
  `never`, and then the store is never read.
- **Budget** is judged on the final pass (a reused case costs nothing) and must be a finite number >= 0.
- **Capacity** is unchecked unless `--devices CLASS=N` or `--nodes` is given (`leg.devices`/`leg.nodes` are `null` in
  plan.json then). A case that needs a reference node is skipped unless `--reference-node`.
- **List flags.** `--suite`, `--force`, `--set`, `--central` and `--pack` fall back to their env twins when the flag is
  absent; the flag wins. `--set` merges env and flags per key.
- **Store.** `--store none` (default) or `clickhouse`: read-only through `clickhouse_connect`
  (`pip install 'spyre-bench[clickhouse]'`), connection from `CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`, `CLICKHOUSE_USER`,
  `CLICKHOUSE_PASS`, `CLICKHOUSE_SECURE`, and the database from `BENCH_CLICKHOUSE_DB`, else `CLICKHOUSE_DB_V2`, else
  `CLICKHOUSE_DB`. The interface is `spyre_bench.store.Store`; `MemoryStore` is the in-memory one for tests. Invalidation
  events are not read yet.
- **Exit codes.** 0 planned, 2 invalid request or catalog, 4 every case skipped, 5 the store failed while a lookup was
  needed.

`plan.json` (`plan_schema: bench.plan/1`) holds `producer.engine` (`version`, `plan_schema`, `identity_version`), the
normalised `request`, the `leg`, a `summary` and `cases[]`. Each case carries the engine's plan record plus `benchmark`,
`id_mode`, `id_profile`, `labels`, `pinning`, `command`, `ingest` and `allow_empty`. `identity_version` is bumped with any change to how
an id is derived; `run` refuses a plan made under another one.

### Command templates

A Benchmark states how to run its cases with `run.command`, an argv array (never a shell string), and optionally
`run.ingest`. Neither is identity. Tokens are `{name}`; `{{` and `}}` are literal braces.

| Phase | Tokens |
|---|---|
| `plan` (case values) | `{case}`, `{benchmark}`, `{component}`, `{environment}`, and every identity value by dotted name (`{case.op}`, `{case.shape}`, `{case.load.iterations}`, `{system.precision}`); a list joins with `,`. An unknown token is a plan error |
| `run` | `{run_id}`, `{subject_id}`, `{arch}`, `{repeat}` (from 1), `{repeats}` (the case's total repeats, cold plus warm for `cache: both`), `{cache}`, `{profile_level}`, `{raw_dir}`, `{repeat_dir}`, `{out_dir}`, `{python}` |

A command's environment is an allowlist, matched case-insensitively: `PATH HOME USER LOGNAME SHELL TMPDIR TMP TEMP TERM TZ VIRTUAL_ENV CI LD_LIBRARY_PATH LANG LANGUAGE PWD HOSTNAME NODE_NAME`; proxies (`HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY`, either case); certificates (`SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE`); build (`MAKEFLAGS MAKELEVEL MFLAGS CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH CPATH`); caches (`HF_HOME HF_HUB_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE TRANSFORMERS_CACHE`); and the prefixes `LC_ PYTHON SPYRE_ OMP_ MKL_ OPENBLAS_ NUMEXPR_ XDG_ CUDA_ TORCH_ VLLM_ KMP_ GOMP_ HF_DATASETS_ CMAKE_ CCACHE_`, plus any `--pass-env-prefix PREFIX` (repeatable; `BENCH_PASS_ENV_PREFIXES`, comma list; matched case-insensitively on both sides). Two filters apply to every name inherited this way, prefixes included. (1) A name with a WHOLE segment (split on `_ - .`) in `TOKEN PASS PASSWD PW SECRET KEY CRED CREDENTIAL AUTH COOKIE BEARER DSN PAT SESSION PRIVATE CERTKEY APIKEY SECRETKEY PRIVKEY PRIVATEKEY ACCESSKEY AUTHKEY PASSPHRASE` is withheld (`SPYRE_BYPASS`, `SPYRE_KEYBOARD`, `PWD` pass; `SPYRE_KEY` does not). (2) A value that embeds URL credentials (`scheme://user:pw@host`, empty user included) or is token-shaped is withheld: a JWT, a `Bearer`/`Basic` credential, a vendor prefix (`sk-`, `ghp_`, `gho_`, `ghs_`, `ghu_`, `github_pat_`, `hf_`, `xox?-`, `AKIA`, `glpat-`, `npm_`, `AIza`, `ya29.`, `pypi-`), base64 with `=` padding, or a 32+ character alphanumeric run (no `_`, `-`, `.` or `/`) that mixes letters and digits. UUIDs, dashed host names, paths and pure hex runs (a commit sha, a digest) are not token-shaped. Every withheld name is listed in `status.json` notes with its reason (`secret-looking name` or `credential-looking value`), never the value. The engine's own `BENCH_RUN_ID`, `BENCH_CASE`, `BENCH_REPEAT`, `BENCH_REPEATS`, `BENCH_CACHE`, `BENCH_PROFILE_LEVEL`, `BENCH_RAW_DIR`, `BENCH_REPEAT_DIR`, `BENCH_SUBJECT_ID` and `BENCH_QUIET` are added. Nothing else (inherited `BENCH_*`, `CLICKHOUSE_*`, tokens, ...) reaches a command unless named with `--pass-env NAME` (repeatable; `BENCH_PASS_ENV`, comma list), which passes the variable as it is, secret-looking name or value included (a name matching a passed prefix but secret-looking still needs `--pass-env`). `run.allow_empty: true` declares that a command that exits 0 with no raw output
and no `metrics.json` is fine. `examples/run-demo` is a runnable catalog with a stub command.

## Running a plan

```bash
spyre-bench run --plan plan.json --out results/bench --catalog benchmarks --central shared [--ingest] [--store clickhouse]
```

`run` executes every `run` case, one subprocess per repeat (twice the repeats for `cache: both`, cold then warm), each
in its own session so a timeout, an interrupt or SIGTERM kills the whole process group (SIGKILL); stragglers of a
finished command are killed with it (a pid-reuse window of microseconds remains, because the group leader is already
reaped). `plan.json` is trusted input: its `command` and `ingest` argv run as written, after every record is
shape-checked. `--out` is made absolute first (commands run in `--workdir`). Engine-owned files (`run.json`, `plan.redacted.json`, `case.json`, `status.json`, `result.json`) are written to a new temp name (`O_CREAT|O_EXCL|O_NOFOLLOW`) that then replaces the target, and every directory from `--out` down to the file is checked with `lstat` first, so a link a benchmark plants, before or while it runs, at the file or at a directory above it (`cases`, `cases/<slug>`, `raw`, `repeat-N`) is replaced and never written through; `case.json` and `plan.redacted.json` are rewritten after the benchmarks have run. Log files are opened `O_NOFOLLOW`. A benchmark can still delete or swap its own case directory (that case then fails `no_output`); it cannot make the engine write outside `--out`. `run` owns `cases/<slug>` under `--out`: it
deletes `raw/`, `repeat-*` and `result.json` there, and the previous `run.json` and `ingest/`, before running; until a
run finishes `run.json` says `exit_reason: incomplete`, so an aborted run never looks like the last one's success.
Numbers and the env file are validated before anything is written. It needs no catalog to execute; with `--catalog` it
also verifies that every selected planned case still derives the planned `benchmark_id` and builds an engine Result per
case.

Flags: `--plan` (`BENCH_PLAN`), `--catalog` (`BENCH_DIR`; none by default), `--central` (`BENCH_CENTRAL`), `--pack`
(`BENCH_PACK`), `--case` (repeatable, `run` cases only), `--out` (`RESULTS_DIR`, `BENCH_OUT`; default `results/bench`),
`--workdir` (`BENCH_WORKDIR`, cwd of every command), `--env-file` (`BENCH_ENV_FILE`), `--run-id` (`BENCH_RUN_ID`, then
`RUN_ID` when it is a UUID), `--force-new-run`, `--ingest/--no-ingest` (`BENCH_INGEST=1`), `--ingest-cmd`
(`BENCH_INGEST_CMD`), `--ingest-timeout`, `--ingest-pass-env` (`BENCH_INGEST_PASS_ENV`), `--ingest-python`
(`BENCH_INGEST_PYTHON`), `--ingest-pythonpath` (`BENCH_INGEST_PYTHONPATH`), `--store` (`BENCH_STORE`), `--landed-wait`,
`--timeout` (a cap in seconds), `--pass-env` (`BENCH_PASS_ENV`), `--pass-env-prefix` (`BENCH_PASS_ENV_PREFIXES`),
`--max-log-bytes` (16 MiB per log; the rest is drained and dropped with a marker), `--max-metrics-bytes` (8 MiB; a
larger `metrics.json` is ignored with a note), `--max-raw-bytes` (1 GiB; over it a `result_notes` warning), `--quiet`,
`--dry-run`, `--json`.

```text
results/bench/
  run.json                bench.run/1 manifest: counts, per-case status, ingest, rows_landed, exit_reason
  plan.redacted.json      the plan with secrets redacted (not runnable)
  cases/<slug>-<hash8>/
    case.json  raw/       {raw_dir}: what the command writes for the ingest to read
    repeat-001/           {repeat_dir}: status.json, stdout.log, stderr.log, optional metrics.json
    result.json           engine Result
  ingest/ingest-<n>.log
```

- **Sidecar.** A command may write `{repeat_dir}/metrics.json` = `{"samples": {"<metric>": [finite numbers]},
  "props": {...}, "warmup_discarded": N}`; Result samples come only from it, the engine parses no producer format.
- **Silent commands.** A case whose repeats all exit 0 but leave `raw/` empty is `failed`, reason `no_output`, with
  `--ingest`; without it too, unless a repeat wrote a `metrics.json` or the Benchmark sets `run.allow_empty: true`.
- **Probe file.** `--env-file` = `{"fingerprint": {"env.fp.x": "v"}, "observations": {}, "node": "n", "qualified": true,
  "clean": {"exclusive": "observed"}}`; `qualified` must be a JSON boolean. Keys outside the registry are dropped and
  noted in `result_notes`. Without a file, and for what the file leaves out, the facts are unknown and lower validity
  (no qualification, no placement, unknown cleanliness), never raise it. A path that cannot be read, or is not a
  well-typed JSON object, is exit 2.
- **Pinning.** When the Environment requests `host.cpu_pinning`, the process tree is sampled while it runs and the worst
  sample is recorded as `run.exec.*` on the repeat.
- **Secrets.** Every `argv` the engine writes or prints (results directory, `plan --json`, `--dry-run`) is redacted by
  whole name segment and value shape: values of `--token`, `--password`, `--pw`, `--auth`, `--bearer`, `--cookie`,
  `--dsn`, `--key`/`--access-key`/`--api-key`, `-p x`/`-p=x`/`-px`, `--user a:b`/`-u a:b`; bare `NAME value` pairs and
  `NAME=value` words with a secret name, also inside a word (`--env=HF_TOKEN=x`, `-DPASSWORD=x`, `sh -c "export
  TOKEN=x"`, `Password=x;`); URL credentials (empty user too); Authorization/Cookie headers and Bearer/Basic values;
  JSON secret pairs; and token-shaped words. `--passes`, `--author`, `MONKEY_PATCH=1`, `--jenkins-run-key`, UUIDs and
  host names and pure hex runs (a commit sha) are left alone, as are `--max-tokens 128`, `--auth-mode x` and `--ssh-key-path p`.
  Also hidden: quoted values, `password: x`, dict-style pairs, `-pass x`, `-u x`, and all of it inside `sh -c "..."`. A secret of no
  recognisable shape in free text is not caught. The file `plan --out` writes
  is not redacted (`run` executes it, and refuses a plan holding `<redacted>`): keep secrets out of `run.command`, use
  `--pass-env`. Environment values are never written. A command's log that cannot be written (full disk) fails the
  repeat (`log_write_failed`) without blocking it; output a detached process writes after the command exits is dropped.
- **Ingest.** The engine writes no database rows. With `--ingest`, only for cases that are `ok`, it runs the ingest
  (own session, killed with its children on timeout or interrupt) after the last benchmark has finished. Isolation:
  stdin is `/dev/null`; the working directory is a fresh empty `<out>/ingest/cwd-XXXX` and `HOME` a fresh empty
  `<out>/ingest/home-XXXX` (the engine recreates `<out>/ingest` with mode 0700 at that moment, removing whatever a
  benchmark left there without following links; logs are opened `O_EXCL|O_NOFOLLOW`); the default argv uses `-I`, which
  ignores `PYTHON*`, the user site and the cwd; the environment is the benchmark allowlist without `PYTHON*`, plus
  `CLICKHOUSE_*`, `INGEST_*`, `PYTHONSAFEPATH=1` and `--ingest-pass-env NAME` (only that way do `HOME`, `PYTHON*` or a
  credential-carrying proxy reach it). The package must be importable by the interpreter, which a default ingest probes before anything runs: `--ingest-python PATH`, or
  for a dev checkout `--ingest-pythonpath DIR` (outside `--workdir` and `--out`, judged by realpath; the argv becomes
  `-s -P` with `PYTHONPATH` set to exactly those directories); a probe that fails is exit 2. A custom `--ingest-cmd` is trusted code with the same environment and directory. A benchmark's own interpreter may load a
  `usercustomize` module it planted in the user site (HOME is allowlisted for benchmarks): that is acceptable, a benchmark is
  untrusted by design and holds no secrets; the ingest is the process that must not. A process a benchmark detached with
  `setsid` can still write where it knows the path: the isolation is against the benchmark's files, not a hostile
  same-user process. Logs: `ingest/ingest-<n>.log` (stdout) and `.stderr.log`, capped.
  Resolution: `--ingest-cmd`, else the case's `run.ingest`, else the profile default (`python -I -m
  spyre_clickhouse_ingest results ...`): `perf-kernel` uses `--xml-dir` and threads `--run-id <the engine's run id>`;
  `vllm`/`vllm-sweep` use `--vllm-results-dir` and `generic`/engine mode `--benchmarks-dir --component <plan
  component>`, both with `--trigger-type perf` and `--jenkins-run-key
  manual:spyre-bench:<engine run id>-<16 hex of the case name>`. **Every case gets its own run key** (the ingest
  refuses a second bundle under one key), so N ok bundle cases mean N invocations and N stored run ids. **The stored
  run_id is not the engine's:** the bundle profiles derive it from the run key and ignore `--run-id`, so the engine
  reads the ingest's report (the last stdout line that is a JSON object, with a string `run_id` and status
  `ingested`, `duplicate` or `would-ingest`) and records `stored_run_ids` in `run.json`; any other report, or none
  from a default bundle ingest, is a failure (exit 5; a failed ingest's `reason` is recorded), and
  `cases_ingested`/`cases_failed` say which cases landed. Use `spyre-bench verdict --run-dir <out>`. A retry is not
  idempotent (the ingest stamps `ended_at`, so identical raw output is a different bundle): use a new `--run-id` or
  `--force-new-run`. A run that ingests refuses a `--run-id` an earlier run in the same `--out` may have ingested
  under; a run that does not ingest repeats ids freely; two runs in different directories that share `RUN_ID` are not
  detected. Tokens: `{python} {run_id} {run_key} {subject_id} {arch} {component} {raw_dir} {out_dir} {case}
  {profile}`. Tests drive the real bundle-ingest code with 2 and 3 cases per profile and a mixed plan (warehouse
  faked), build real offline bundles from the defaults and parse all four with the ingest's own argument parser when
  it is importable. The engine's tests do not run the ingest against a live ClickHouse.
- **Rows landed.** With `--ingest` and `--store clickhouse`, the engine polls `benchmark_runs` for each stored run id (up
  to `--landed-wait` seconds): every `ok` ingest-mode case's `benchmark_id` must be present, and the run must hold at
  least one other benchmark id per `ok` engine-mode case (their ids are the ingest's, which the engine cannot know), so
  a partial ingest fails. Otherwise exit 1, reason `no_rows_landed`. Without a store, or without `--ingest`, the check
  is `unchecked`, not passed (`run.json` says why; `--ingest` without a store also warns on stderr).

| Exit | Meaning |
|---|---|
| 0 | every executed case ok, or the plan held only `reuse` cases |
| 1 | a case failed, `no_rows_landed`, or a Result failed its shape or id check or could not be built (`result_invalid`) |
| 2 | invalid plan, flags, env file or catalog, a missing input file; `engine_skew` (plan format or identity version differs) |
| 4 | nothing ran and the plan was not all-reuse |
| 5 | an ingest invocation failed, timed out or gave no report, the store failed, or the disk failed |
| 130, 143 | Ctrl-C, SIGTERM (children killed first; no traceback) |

Precedence: 2, 5, 1, 4, 0. Every command and ingest argv is rendered before the first case runs, so a bad token exits 2
with nothing executed. `--dry-run` prints the rendered commands and creates and runs nothing (exit 1 if a case has no
command, 4 if nothing would run).

## Verdicts

```bash
spyre-bench verdict --run-dir results/bench --json [--fail-on regressed] [--require-rows] [--quiet]
spyre-bench verdict --run-id <stored-uuid>   # or an engine run id of a run.json found in --run-dir / RESULTS_DIR
spyre-bench verdict --artifact <uuid>        # the artifact's newest run that has verdict rows
```

Reads `v_benchmark_metric_verdicts` and prints, per benchmark and metric, `pass` (the view's `unchanged`), `regressed`,
`improved`, `suspected` or `no_baseline`, with the tier, reason, baseline size and values. The engine never decides a
verdict and the command never gates: exit 0 (also for a run with no verdict rows) unless `--fail-on` names statuses and
one occurs (exit 1); exit 2 for bad flags or a store failure, never a silent pass. `--quiet` (`BENCH_QUIET=1`) drops
the measured values. Output: `verdict_schema: bench.verdict/1` with `summary`, `benchmarks[]` (worst status per
benchmark: regressed, suspected, improved, no_baseline, pass) and `verdicts[]`.

Each verdict also carries `anchored`, the metric's verdict against the accepted baseline of its series
(`v_benchmark_anchored_verdicts`): `status` (`regressed`, `suspected`, `improved`, `unchanged`, `no_anchor`), `reason`
(`anchor_regressed`, `anchor_improved`, `anchor_unchanged`, `no_anchor`, or a cap such as `single_run`), the anchor `version`,
the values, and for a metric over the anchor's floor `since_ts`, `n_exceeding_runs` and `days_exceeding`. `summary.anchored`
counts the statuses. It is null for a metric with no anchored row, and absent altogether when the store has no anchored view.
The rolling verdict answers "got worse recently"; the anchored one "worse than the accepted normal, and since when", so a
persistent regression stays visible after the rolling baseline has absorbed it. `--no-anchor` leaves it out and
`--fail-on-anchored regressed` gates on it (exit 1; exit 2 when the store has no anchored view, never a silent pass); like `--fail-on` nothing gates by default.

## Accepting a new normal

```bash
spyre-bench accept --run-dir results/bench --reason "driver 2.4 is the new normal" [--window N] [--benchmark UUID] [--metric M] [--dry-run]
```

Proposes a stored run as the new anchor of its series. It reads the run's samples from `v_benchmark_series_runs`, refuses a
run that is not baseline-eligible (suspect, invalidated or off-purpose) or has a non-finite sample (exit 1, nothing proposed), and writes one
`reference-request-<key>-<run>.json` (never replacing an existing file) (`bench.reference-request/1`, role `anchor`) per scope (component, arch, backend, source job)
under `--out` (default `BENCH_OUT/accept`). Series of the scope that the run does not replace are carried over from the
current anchor with their own time, so a partial accept does not reset them. An anchor needs at least 3 samples; a series
that stores one sample per run needs `--window N`, which pools the series' newest N eligible runs up to the accepted one
(`accept` refuses and says so otherwise). The request is the input of the reference
machinery (reference bundle, approval, tag move): the engine writes no rows and moves no tag, so the anchor changes when the
reference is approved, and until then `verdict` keeps comparing with the old one. `--reason` is required and recorded.
Each value carries a `case` block (`component`, `name`, `tags`, `disc`, `disc_keys`, and `id_profile` when the Result
has one) that rebuilds its `benchmark_id`: taken from the Results of `--run-dir` when given, else from the registered
case; `accept` refuses, naming the ids, when a block cannot rebuild its id. `anchor` is `{role, run_id, as_of}`
(`from_run_id` repeats `run_id` for the reference side that reads it). `reference create --from-request` consumes the file.
Exit 1 not eligible or no case data, 2 bad flags, 5 store failure.

## Uploading results

```bash
spyre-bench upload results/bench                       # seal and check; prints what would be sent (dry run)
spyre-bench upload results/bench --apply --wait        # put it in the results inbox and wait for the relay
spyre-bench upload bundle.tgz --apply                  # send a bundle sealed elsewhere, unchanged
```

`upload` is for results measured outside CI: a lab host, a developer machine, a partner's machine. The engine writes
no rows. The command seals the run directory (every `result.json` under it) into a `bench` bundle through the
ingest's offline module, checks it, and prints the run key, case and repeat counts, both digests and a receipt id
(`RCPT-...`). Without `--apply` nothing leaves the machine (the sealed bundle is written to `RESULTS_DIR/upload/bundle.tgz`, or `--out`). With it, the bundle is put in the Artifactory inbox
(`--base-url`, `--repo`, `--root` or `ARTIFACTORY_BASE_URL`, `ARTIFACTORY_V2_RESULTS_REPO`,
`ARTIFACTORY_V2_RESULTS_ROOT`; `ARTIFACTORY_TOKEN` or `~/.netrc`) and the relay records it; `--wait [MIN]` polls for
its outcome. The dashboard's upload page takes the same bundle if you have a login. `--via api` is refused: the
dashboard accepts a browser login only.

| Flag | Effect |
|---|---|
| `--seal-trust lab\|external` | where the results came from (default `external`); the receiving channel records the trust and a label can only lower it |
| `--redact-host`, `--redact-user` | leave the sealing host and user out of the bundle |
| `--ended-at`, `--arch`, `--test-type perf\|quality` | end time (ISO-8601 with a zone), where it ran (default `run.json`; required when it records none: it is never guessed from the host), the tier |
| `--out` | where to write the sealed bundle (default `RESULTS_DIR/upload/bundle.tgz`) |

Rules that follow from the receiving side:

- **Upgrade the connected ingest first.** The sealing host's ingest must not be newer than the receiving one, and
  `upload` needs one that knows `bench` bundles; an older one is reported with that advice.
- **Times carry a zone.** A time without one is read as UTC, whatever the sealing machine's zone; pass `--ended-at`
  with an explicit zone when the tests ran elsewhere.
- **The whole ingest package must be installed** (`spyre-bench[ingest]`, or its wheel in an air-gapped kit). The
  vendored identity module that derives benchmark ids (see the adapter contract) is not enough, because sealing reads
  `bundle.schema.json` beside the offline module.
- **Re-sending is safe.** The run key is the engine run id and the results digest (`content_sha256`) ignores the
  sealing host and time, so sealing the same results again is the same run even though the bundle digest differs; the
  receiving side skips repeats it has and rejects a repeat whose content changed.
- **Seal flags apply to a results directory.** Given an already sealed bundle, `--redact-*`, `--seal-trust`, `--arch`,
  `--test-type`, `--ended-at` and `--out` are refused instead of ignored; `--wait` needs `--apply`.
- **Keys the receiving channel owns** (`run.trust` and others) are dropped from the sealed copy, with a warning.

Exit codes: 0 sent or dry run, 2 invalid input, missing configuration or a rejected bundle, 5 the transfer failed or
the relay did not answer within `--wait` (status `waiting`: the bundle is in the inbox but not yet confirmed).
Bundle format and trust rules: `extensions/clickhouse-ingest/docs/bench-bundles.md`.

## Invalidations

```bash
spyre-bench invalidate (--run ID [--benchmark ID [--repeat N]] | --node N | --device SERIAL | --placement ID) [--from T --to T] \
    --reason-code C --reason TEXT [--state suspect|invalidated] [--apply [--confirm [TOKEN]]]
spyre-bench invalidate restore ID --reason-code C --reason TEXT [--apply]    # also: set ID --state ..., list, show ID
```

The default is a preview: the store is read and the runs, series, benchmarks, repeats and verdict changes are printed;
nothing is written. `--apply` records one event through a `bench-invalidation` bundle and the ingest (which owns the
matching view, the state machine and the 50-run confirmation threshold). Needs `spyre-bench[ingest,clickhouse]`. Exit 0
ok or preview, 2 refused or bad flags, 5 the ingest or store failed. See [incident-runbook.md](incident-runbook.md).

## References

`spyre-bench reference` turns a run into a comparison target, seals it as a `bench-reference` bundle, submits it for
approval and compares later runs with it, informationally and never as a baseline: [`references.md`](references.md).

## Quality

A quality Benchmark (`type: quality`) records a score over n samples: accuracy of an eval, or agreement with a reference
output. Its result is advisory and carries no clean-node or timing requirement; what makes a score trustworthy is the
data behind it.

### What a quality command writes

The command writes the usual `metrics.json` sidecar into `{repeat_dir}`. For the `llm/lm-eval` workload:

| Key | Meaning |
|---|---|
| `samples["llm.eval.exact_match"]` (or `acc`, `acc_norm`, `f1`) | the score, one value per repeat |
| `samples["llm.eval.<metric>_stderr"]` | the harness's standard error; without it the engine derives a binomial one from `llm.eval.n_samples` |
| `samples["llm.eval.n_samples"]`, `error_frac`, `truncated_frac` | what was scored, how many requests failed, how many generations were cut |
| `props["run.quality.dataset_digest"]` | digest of the question list that was scored |
| `props["run.quality.harness"]`, `harness_version`, `protocol` | the harness, its version, and the digest the harness computed over task version, prompt and filter pipeline. Recorded, never authored |
| `props["run.quality.model_revision"]` | the model commit |
| `props["run.quality.temperature"]`, `seed` | how it sampled; `seed` is a run attribute, never identity |

Per-question 0/1 vectors do not go in `samples`: the views reduce an array to one number. Keep them as an attachment next
to the Result and record its digest in `run.quality.samples_digest`.

### Gates and the quality ValidityPolicy

A `ValidityPolicy` with `applies_to: [quality]` replaces the clean, qualification, cv and warm-up requirements with
`requires.gates`, a list of pack gates evaluated for the row. A policy is either perf or quality; mixed lists are rejected,
and Project `validity_policy` names the perf one. A quality case is judged by the quality policy, so the same purposes
(`pr`, `nightly`, `reference`, ...) can list different gates for the two kinds.

| Gate | Fails when | Caps at | Reason |
|---|---|---|---|
| `llm.dataset_digest` | the reported digest differs from the input's `expect.selection_digest` | invalid | `dataset_mismatch` (`dataset_unreported`; `dataset_unpinned` suspect when the input declares none) |
| `llm.records_complete` | `n_samples` differs from `expect.records` | invalid | `records_incomplete` (`records_unreported`; `records_undeclared` suspect) |
| `llm.no_errors` | `error_frac` is not 0 | invalid | `infrastructure` |
| `llm.no_truncation` | `truncated_frac` is above 1% | suspect | `truncation` |
| `llm.harness_pinned` | harness version or protocol is not recorded | suspect | `protocol_unpinned` |
| `llm.model_pinned` | no model revision | suspect | `model_unpinned` |
| `llm.determinism` | a greedy case sampled, or a sampled case has no seed | suspect | `sampling_in_greedy_case`, `seed_unpinned` |

A gate whose input the command did not report caps at suspect with an `*_unreported` or `*_undeclared` reason
(`errors_unreported`, `truncation_unreported`, `records_undeclared`, `dataset_unpinned`) rather than passing; `llm.determinism`
also reports `sampling_in_greedy_case` and `sampling_mismatch` (the temperature that ran is not the one declared). Only the first
input of a workload that carries `expect` is checked.

A purpose with `reference: true` still needs `floor: valid` and `baseline: true`, but may leave out the clean and
qualification requirements as long as it lists the five data gates `dataset_digest`, `records_complete`, `no_errors`,
`harness_pinned` and `model_pinned`. `max_cv_pct` and `discard_warmup` are rejected on a quality row. A greedy case
(`eval.temperature: 0`) must not set `eval.seed`.

### Comparing a score with a reference

An Analysis with a `reference` block compares each selected Result with a reference, in combined standard errors:

```yaml
kind: Analysis
metadata: {name: quality-vs-reference}
spec:
  select: {benchmarks: [eval-questions]}
  metric: llm.eval.exact_match
  vary: [system.backend]
  reference: {case: eval_questions_demoa_ref_l8, tolerance: {stderr: 3, abs: 0.02}}
```

`reference` is the Result of one case (`case`), or an inline `value` with its `stderr` or `n` and an abstract `label`.
Per row: `pass`, `improved`, `regressed`, `suspected` (`single_run`), `no_power`, `no_stderr`, `invalid`. The allowance is the
larger of `stderr` x combined standard error and `abs`. `regressed` needs a shortfall past the allowance and at least four
combined standard errors; a smaller shortfall past it is `suspected`. When `abs` is set and the run cannot resolve it
(1.645 combined standard errors above `abs`) the row is `no_power`. A Result whose validity is `invalid` or `suspect` is
reported as such and is never `regressed`. See `examples/quality-demo` and `spyre-bench analyze`.

With more than one repeat the engine aggregates the median of the per-repeat scores and of their standard errors; repeats are
not pooled, so for a greedy case (the same samples again) the standard error stays the single-run one, and for a sampled
case it is conservative. A Result no policy judged is reported `suspected` (`validity_unjudged`), and so is every row compared
with a reference Result that is not `valid` (`reference_not_valid`). At p = 0 or 1 the binomial standard error uses a half-count
shrink so a perfect score is never taken to have zero noise.

Ingest: a quality case that uses the `generic` identity profile is handed to the ingest with `--trigger-type quality`
(the `{test_type}` ingest token).

### Adding a quality benchmark, dataset or metric

`examples/quality-demo` is a runnable catalog of one quality case and its reference case; start from it.

1. **The question list.** Declare an `Asset` with `integrity.files` (a sha256 per file) and a `Dataset` over it with a named
   subset (`limit`, `order`). A subset is a fixed list, never "whatever the loader yields first": the same questions in the
   same order are what make two runs comparable. A new file digest is a new input identity, hence a new series.
2. **The Benchmark.** `type: quality`, a `llm/lm-eval` (or `llm/parity`) workload whose `inputs[].expect` states how many
   records the run must score (`records`) and the digest of the selection (`selection_digest`, `sha256:...`). Set
   `eval.temperature: 0` for a greedy case and leave `seed` out; a sampled case sets both and the gate `llm.determinism`
   needs the seed recorded. List the metrics the case records under `metrics.required`.
3. **The command.** `run.command` writes `metrics.json` into `{repeat_dir}` (see "What a quality command writes"). Report the
   dataset digest of what it actually scored and every pin it knows; leave out what it does not know rather than guessing,
   because a missing pin is what the gates look for. A command that cannot count errors must not write `error_frac`.
4. **A metric.** Register it in the pack (`metrics:`) with a unit, a `direction`, `category: quality` for a score and
   `diagnostic` for what supports one (standard error, counts, error shares), and the `statistic`. A score needs a
   direction to be compared; a diagnostic has none and is never judged. Name the standard error `<metric>_stderr` so the
   engine finds it, or let it derive a binomial one from `llm.eval.n_samples`.
5. **Which gates.** The quality ValidityPolicy (central) lists gates per purpose. Start a new case under `pr` with
   `llm.records_complete` and add the others as the command learns to report what they check. A quality reference purpose
   must list the five data gates.
6. **Identity.** With the `generic` ingest profile every prop of the benchmarks file is an identity key; list the same keys,
   sorted, as `identity.discriminators` and keep seeds, concurrency and anything else about the run out of it. The id of the
   engine and of the ingest must agree; a test that derives both for each case catches drift.
7. **Judge it.** Add an `Analysis` with a `reference` block and run `spyre-bench analyze`. Use `abs` for the smallest change
   that matters in practice and read `no_power` as "this many questions cannot resolve it".

Per-architecture differences (which cases run, card limits, models to skip) belong in data next to the product repo's
definitions, selected by an architecture name that only names a file. Nothing in a command, Makefile or workflow branches on it.

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
`placement.validity_facts` feeds `engine.validity_of`. `run` calls it for every repeat of a case that asks for pinning
(or verification) and launches the tree under the requested policy first (`pinning_run`, `pinning_plan`). The tree is the root's descendants plus every process in its
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

`examples/pinning-demo` is the end-to-end pattern: a latency Benchmark at two output lengths (single process and
two ranks), node-a and node-b Environments with and without `device_local` pinning, a stub command and the
`pinning-effect` Analysis; `run` applies the request and records the placement. See [pinning.md](pinning.md) for the
user guide (policies, per-device lists, `{pin_wrapper}`, reading the effect table, `unknown`/`suspect`), the measurement
matrix script `scripts/bench-pinning-matrix` and the developer notes.

`relax:` on an Analysis lets `compare` combine cells that differ in the attributes it names (`subject.id`, `env.id`,
`env.node`, `env.label.<k>`, a trailing `*` allowed, a bare `*` relaxes nothing), for a comparison whose point is that
difference, such as card count. Everything not named stays strict, but relaxing `env.id` also stops comparing the rest of
the environment fingerprint (kernel, driver). `derive` pairs stay strict. `examples/trace-demo` scales a decode Benchmark
over 1, 2 and 4 cards (`decode-scaling`) and keeps a per-card controlled-set query (`rank-spread`, evaluated on stored rows
where `run.rank` is a property of each per-rank row; validate with `--pack examples/packs/trace-example`).
