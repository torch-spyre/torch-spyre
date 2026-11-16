# 4. Datasets, assets, models and stores

Benchmarks read data: a trace of prompts, a calibration set, a set of images, weights. The data model has four kinds,
each answering one question, so that a bytes-level change, a meaning-level change and a where-to-find-it change are
different edits with different consequences:

| Kind | Question | Changes when | Is it identity? |
|---|---|---|---|
| Store | how do I reach it? | an endpoint, mirror or credential reference moves | no |
| Asset | which exact bytes? | the content changes (a new digest or revision) | yes: content-addressed `asset_id` |
| Dataset | what do those bytes mean, and which part? | role, schema or subset changes | yes, per subset: `input.<name>.select` |
| Model | which weights? | a revision of any weight asset changes | yes: `model_id` |

The catalog is `examples/docs-data`:

```text
examples/docs-data/
  project.yaml    Project, Suite, Selection, Environment
  stores.yaml     three Stores: gated artifact store, local node cache, public hub
  assets.yaml     four Assets: a pinned file, an archive, a generator, pinned weights
  datasets.yaml   two Datasets, one with named subsets
  models.yaml     one Model
  serve/          the Benchmark that reads them, and its stub command
```

## Stores: references, never values

```yaml validated in=docs-data/stores.yaml
apiVersion: bench/v1alpha1
kind: Store
metadata: {name: datasets}
spec:
  type: core/artifactory
  endpoint: {env: STORE_BASE_URL}
  root: {env: STORE_DATASETS_PATH}
  auth: {mode: token, refs: [{env: STORE_TOKEN}, {secret: store-token}]}
  visibility: internal
  mirrors: [node-cache]
  resolve: [node-cache, mirrors, primary]
```

- `type` is a store type from a pack (the core pack registers `core/artifactory`, `core/hf`, `core/filesystem` and
  `core/http`); the type's own schema validates the extra fields (`path` for a filesystem store).
- `endpoint` and `root` are *value references*: `{value: ...}` for a public literal, `{env: NAME}` for the name of an
  environment variable. Use `env` for anything an organisation would not publish.
- `auth.refs` lists where a credential comes from (`env`, `file` or `secret`: names only). There is no field that takes
  a credential, and the validator refuses one:

```yaml validated base=docs-data as=stores-extra.yaml expect="error:unknown key 'token'"
apiVersion: bench/v1alpha1
kind: Store
metadata: {name: leaky}
spec:
  type: core/artifactory
  visibility: internal
  auth: {mode: token, token: abc}
```

- **Access-gated stores.** A store whose namespace needs a group membership or token is declared with `visibility:
  internal` and `auth.mode: token`. The definition is safe to publish because it holds only variable names; whether
  the credential exists is the launcher's concern, so a missing credential fails the *fetch* of a run, never `validate`.
  Mark a store `auth: {mode: token, optional: true}` when anonymous access works with lower rate limits.
- `resolve` is the lookup order: `node-cache` first, then `mirrors`, then the `primary`. `mirrors` names other Stores.
  A Store can say `offline: {allowed: true}` for use without a network.

## Assets: exact bytes

```yaml validated in=docs-data/assets.yaml
apiVersion: bench/v1alpha1
kind: Asset
metadata:
  name: sample-trace-files
  annotations: {consumer.env: DEMO_TRACE_PATH}
spec:
  type: fileset
  format: jsonl
  version: "1"
  license: internal
  visibility: internal
  source: {store: datasets, path: traces/sample/1}
  integrity:
    files: [{path: sample.jsonl, sha256: "1111111111111111111111111111111111111111111111111111111111111111", size: 4096}]
```

An Asset has a `type` (`fileset`, `table`, `generator`, `generated`, `stream`), a `source` (exactly one of
`store` + `path`, `uri` (+ `revision`), or `generator`) and `integrity`:

- **Pin by digest or revision, never by a moving name.** For files, list each file's `sha256` (and `size`); or give a
  `manifest` (`path` + `sha256` of a `SHA256SUMS` file). For a hub model give a commit `revision` and set
  `integrity.revision_pinned: true`. An unpinned asset is accepted but is not reproducible; treat it as a smell.
- **`asset_id` is content-addressed**: it hashes the integrity, so a new digest is a new asset, and through the
  Dataset and the Workload's input it is a new benchmark. That is intended: a different dataset is a different question.
- Quote a digest. YAML reads an all-digit string as a number, and a 64-character hex string must be a string:

```yaml validated base=docs-data as=assets-extra.yaml expect="error:is not of type 'string'"
apiVersion: bench/v1alpha1
kind: Asset
metadata: {name: unquoted}
spec:
  type: fileset
  source: {store: datasets, path: p}
  integrity: {files: [{path: f.bin, sha256: 1111111111111111111111111111111111111111111111111111111111111111}]}
```

```yaml validated base=docs-data as=assets-extra.yaml expect="error:does not match '^[0-9a-f]{64}$'"
apiVersion: bench/v1alpha1
kind: Asset
metadata: {name: short}
spec:
  type: fileset
  source: {store: datasets, path: p}
  integrity: {files: [{path: f.bin, sha256: abc123}]}
```

- `visibility` is `internal` or `public`; `license` is recorded so a public catalog can state what it redistributes.
- `access` tunes the fetch: `mode: pull | stream`, a per-asset `resolve` order, and `offline: use-cache | fail`.
- `produced_by` records lineage for an asset that is the *output of a benchmark* (`type: generated`): which Benchmark
  and output made it, and a policy (`regenerate: never | if_inputs_change | always`).

### Hub models and generators

```yaml validated in=docs-data/assets.yaml
apiVersion: bench/v1alpha1
kind: Asset
metadata: {name: random-prompts, description: "a pure function of (tool, seed, params): nothing is stored"}
spec:
  type: generator
  version: "1"
  license: Apache-2.0
  visibility: public
  source: {generator: {tool: random-prompts, seed: 0, params: {range_ratio: 0.0}}}
---
apiVersion: bench/v1alpha1
kind: Asset
metadata: {name: demo-llm-weights}
spec:
  type: fileset
  format: safetensors
  version: "1"
  license: Apache-2.0
  visibility: public
  source: {uri: "hf://models/example-org/demo-llm", revision: 8f2c4a6e1b3d5f708192a3b4c5d6e7f8091a2b3c}
  integrity: {revision_pinned: true}
  access: {mode: pull, resolve: [node-cache, primary], offline: use-cache}
```

A **generator** stores nothing: it is a pure function of its tool, seed and parameters, and every random operation
carries a seed. A hub asset names a full commit hash as `revision`; a branch or tag name would drift under you.

### Archives and consumer variables

Ship an archive as a `fileset` whose one file is the archive; the digest covers the archive, not its contents. The
engine neither extracts nor fetches; a launcher or helper does, and it needs two facts the definition can hand over
as `metadata.annotations`:

```yaml validated in=docs-data/assets.yaml
apiVersion: bench/v1alpha1
kind: Asset
metadata:
  name: sample-images-archive
  annotations: {consumer.env: DEMO_IMAGES_PATH, archive: tar.gz, extract: "images/ into $DEMO_IMAGES_PATH"}
spec:
  type: fileset
  format: tar.gz
  version: "1"
  license: internal
  visibility: internal
  source: {store: datasets, path: images/sample/1}
  integrity:
    files: [{path: images.tar.gz, sha256: "2222222222222222222222222222222222222222222222222222222222222222", size: 1048576}]
```

Annotations are free-form strings that the engine neither reads nor hashes: changing one never changes an id. A
convention that works: `consumer.env` names the environment variable the benchmark's command reads to find the
materialised path (the example command reads `DEMO_TRACE_PATH`), `archive` states the container format and `extract`
says how it becomes the file the command expects. Remember that the command's environment is an allowlist: the launcher
must pass such a variable explicitly (guide 8).

## Datasets: meaning and subsets

```yaml validated in=docs-data/datasets.yaml
apiVersion: bench/v1alpha1
kind: Dataset
metadata: {name: sample-trace}
spec:
  asset: sample-trace-files
  role: trace
  schema: {record: {prompt: text, output_len: int}, id: {field: seq_id}}
  subsets:
    head-200: {limit: 200, order: sequential}
    stratified: {sample: {n: 1000, seed: 0, stratify_by: bucket}}
```

- `asset` is shorthand for one asset; `assets` names several. `role` is one of `trace`, `eval`, `calibration`,
  `shapes`, `fixture`, `schema`.
- `subsets` are named selections. A subset is a **pure function** of the asset contents and the selection: `limit`,
  `offset`, `order` (`sequential` or `shuffle` with a seed), `sample` (a `seed` is required), `filter`, `columns`,
  `split`, `ids`. Never write a subset that depends on the clock or an unseeded random choice.
- `validate: {mode: off | sample | full, records: N}` states how much of the data a fetch step checks.

[Guide 12](12-worked-example-nightly-weekly.md) declares the datasets of `vllm bench` (a generated `random` set, a
curated chat set under a slug, and the shape of a deployment-private recorded trace).

### Naming a dataset: slugs and `dataset.json`

A public definition never carries the real name of a dataset. It names the dataset by an **opaque slug** (`slot-31` below)
that locates the data in the artifact store, and the store says what the slug means:

```yaml validated base=docs-data as=slug-example.yaml
apiVersion: bench/v1alpha1
kind: Asset
metadata: {name: slot-31-files}
spec:
  type: fileset
  format: jsonl
  visibility: internal
  source: {store: datasets, slug: slot-31}
  integrity: {files: [{path: data.jsonl, sha256: "0000000000000000000000000000000000000000000000000000000000000031"}]}
---
apiVersion: bench/v1alpha1
kind: Dataset
metadata: {name: slot-31}
spec: {asset: slot-31-files, role: eval, subsets: {head-50: {limit: 50, order: sequential}}}
```

Beside the data the store holds `<store root>/slot-31/dataset.json` (the store namespace is configured per deployment
and never written into a public file):

```text
{"display_name": "Example Questions", "public": false, "license": "internal", "sha256": "<digest of the data>", "notes": ""}
```

- `source: {store, slug}` is `store+path` with the path `<slug>`. A Dataset may also declare `slug:` (default: its name)
  and `store:` (default: the store of its first Asset) when the metadata lives elsewhere than the data.
- **Only `public: true` shows the real name.** `explain --dataset-meta-dir DIR` (or `--dataset-store STORE`) reads the
  files and labels a dataset `display_name` when the flag is exactly `true`, otherwise with the slug. The label is what
  `explain` prints; every tool prints Dataset and Asset *names* as written, so give them slug names too. Without those
  flags nothing is read and the slug is shown.
- `validate` never touches a store: a slug is only a string there, so a catalog validates offline.
- A missing, unreadable or malformed `dataset.json`, a `slug` field that disagrees with its directory, or a `sha256` that
  none of the Dataset's Assets declares is an **error**, never a silent fallback to the slug. Asset digests are still
  verified by the fetch step as before.
- `dataset.json` is a **trusted input**: whoever can write the store path decides what is public. Keep it write-restricted
  and review changes to a `public` flag like a change to the definitions.
- The file is not identity. Benchmarks that already carry a dataset name in their id keep it; a slug never renames one.

## Models

```yaml validated in=docs-data/models.yaml
apiVersion: bench/v1alpha1
kind: Model
metadata: {name: demo-llm}
spec:
  assets: [{role: weights, asset: demo-llm-weights}]
  family: demo
  architecture: decoder-only
  params: {total: 1000000, active: 1000000}
  native_precision: bf16
```

A Model names its weight assets by role (`weights`, `tokenizer`, `config`, `adapter`). `model_id` is the hash of the set
of (role, `asset_id`) pairs, so bumping a revision is a new benchmark. A System refers to a Model by name through its
type's `model` field (the `llm/vllm` type declares it as a reference to a Model, so the id, not the name, is hashed).

## Using the data in a Workload

A Workload's `inputs` bind a name to exactly one of a Dataset (with a subset), an Asset or a Model:

```yaml validated in=docs-data/serve/serve-demo.yaml
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: serve-demo, owner: example-team}
spec:
  type: perf
  level: service
  tier: sanity
  system: demo-llm-fp16
  workload:
    base: trace-replay
    patch: {inputs: [{name: prompts, dataset: sample-trace, subset: head-200, expect: {records: 200}}]}
  environment: one-card
  metrics: {required: [llm.ttft.p99, llm.itl.median]}
  run:
    repeats: 3
    timeout: 20m
    command: ["{python}", "tools/emit_serve.py", "--repeat", "{repeat}", "--raw", "{raw_dir}", "--out", "{repeat_dir}"]
  variants:
    - name: serve_demo_c1_sample
```

- `subset` picks a named subset; inline `select` fields override it. `expect: {records: N}` asserts the count and, with
  `selection_digest`, the exact selection; `digest: off | on_expect | always` says when the digest is computed.
- `access: path | reader` says whether the command receives a path or reads through the engine's reader.

`spyre-bench explain` shows how the references resolved into identity:

```bash validated
spyre-bench explain serve_demo_c1_sample --catalog examples/docs-data
```

Look for `system.model_id`, `input.prompts.asset` and `input.prompts.select` under `identity.disc`. Change a digest in
`assets.yaml` and all three downstream ids move; `tests/test_docs_examples.py` pins that behaviour. The Dataset's *name*
and subset *name* appear only as props (`input.prompts.dataset`, `input.prompts.subset`), so renaming a subset does not
restart a series but changing what it selects does.

## Offline and air-gapped use

The definitions are the same on a connected laptop and in a sealed room; only the Store changes.

1. List a local store first in `resolve` (`node-cache`) and mark remote assets `access: {offline: use-cache}`.
2. Pin everything: digests for files, commit hashes for hub assets. An air-gapped kit is verified against those
   digests, so an unpinned asset cannot be verified there.
3. Keep `visibility` honest: a public catalog may reference only public stores and `{env: ...}` endpoints.
4. The command's environment allowlist already includes the offline switches of common tools (`HF_HUB_OFFLINE`,
   `TRANSFORMERS_OFFLINE`, `HF_HOME`, `HF_HUB_CACHE`); anything else is passed with `--pass-env`.
5. `validate`, `explain`, `plan` and `ids` need no network. `run` needs only what your command needs.

## Checklist

1. Every Asset has a digest or a pinned revision; digests are quoted.
2. Credentials appear only as `{env: NAME}` or `secret: name`; no endpoint of an internal system in a public catalog.
3. Subsets are pure functions with seeds; `expect.records` is set for any subset used in a gating benchmark.
4. A new asset version is a new Asset (or a new `version` and digest), never an in-place edit of a published one.
5. `explain` shows the `input.*` and `model_id` values you expect, and `validate` passes with `--central` and `--pack`.
6. A dataset is named by a slug in public definitions; its real name lives only in `<slug>/dataset.json`.
