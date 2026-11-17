# 12. Worked example: nightly and weekly runs of the two real benchmarks

Guides 2 to 6 use made-up systems. This one declares the two benchmarks the project actually runs, `vllm bench`
(spyre-inference) and `spyre-perf-suite` (torch-spyre ops, hf-adapters models), and gives each a nightly and a weekly
invocation. For each benchmark you get the dataset, the model, the metrics, the workload, the environment request, the
commands that check them, how a CI job invokes them, and how a result is read.

| Catalog | Declares | Shape taken from |
|---|---|---|
| `examples/real-vllm-bench` | `vllm bench` latency, throughput and serving cases | `benchmarks/` of spyre-inference (public models and datasets substituted; see below) |
| `examples/real-perf-suite` | `spyre-perf-suite` operator cases | `benchmarks/` of torch-spyre (the files are copies) |
| `examples/real-perf-suite-models` | `spyre-perf-suite` model cases | `benchmarks/` of hf-adapters (model files are copies; the model list and Selections are reduced to public models) |

What is in a definition: names, shapes, flags and identity. What is never in one: a measured value, a score, a
threshold, a baseline or a tolerance. Those live in the store and in the policy of the deployment that runs the
benchmark. Where the data comes from (a store's address, a credential) is also deployment configuration: a definition
names an environment variable, never a value. The examples use public models and name every dataset by slug; a deployment adds its
own recorded datasets the same way (see "Datasets" in part A). Every block below is checked by the snippet tests against
the example files and the engine.

Cadence is not part of a definition. A Selection says *what* to run; the CI job that invokes it says *when*, and the
request's `purpose` (`nightly` or `weekly`) is recorded on the rows so that scheduled runs can be told from ad-hoc ones.
Guide 1 explains the split.

## Part A. `vllm bench` (spyre-inference)

`vllm bench` has three modes and each writes a JSON file the ingest reads: `latency` (one batch end to end),
`throughput` (an offline batch) and `serve` (requests against a running server). The catalog covers all three.

```text
examples/real-vllm-bench/
  project.yaml            Project, Suite, Selections smoke / nightly-small / weekly-full
  environments.yaml       what a case needs: 1, 2 or 4 cards, with or without CPU pinning
  stores.yaml             the public model hub and the curated dataset store
  assets-datasets.yaml    the bytes of each dataset
  datasets.yaml           what those bytes mean
  models.yaml             Models
  assets-models.yaml      the weights each Model points at
  decoder/<test>.yaml     one Benchmark per series name
```

### The Project and its Selections

```yaml validated in=real-vllm-bench/project.yaml
apiVersion: bench/v1alpha1
kind: Project
metadata: {name: spyre-inference, owner: spyre-inference}
spec:
  component: spyre-inference
  packs: [{name: llm-serving, version: "0.2"}, {name: spyre, version: "0.2"}]
  default_suites: [decoder]
  default_environment: spyre-1card
  visibility: public
  groupings:
    - {name: tier, ordered: true, cardinality: one, values: [sanity, core, extended]}
    - {name: suite, ordered: false, cardinality: many}
  series_labels: [env.label.device.count, env.label.cpu_pinning]
```

`component` is `spyre-inference` because the ingest hashes it into every stored id (guide 2). `series_labels` keep a
one-card and a four-card run, and a pinned and an unpinned run, in separate series.

The two Selections are the whole difference between the nightly and the weekly set:

```yaml validated in=real-vllm-bench/project.yaml
apiVersion: bench/v1alpha1
kind: Selection
metadata: {name: nightly-small, description: "latency, first-token companion, throughput and one serving point on one small model"}
spec:
  cases:
    - latency_qwen05b_tp1_in64_out64
    - latency_qwen05b_tp1_in64_out1
    - throughput_qwen05b_tp1_in64_out64
    - serving_qwen05b_tp1_c4_n100_len4096_seqs4
---
apiVersion: bench/v1alpha1
kind: Selection
metadata: {name: weekly-full, description: "every declared decoder case: wider models, 1 and 4 cards, the concurrency sweep"}
spec:
  selector: {types: [perf], groupings: {tier: {at_most: extended}}}
```

The nightly Selection names its cases. The weekly one selects by tier, so a case added at any tier up to `extended`
joins the weekly run without editing the Selection. Tiers are declared on each Benchmark.

### Datasets

`vllm bench serve` takes its prompts from a dataset (`--dataset-name`). A dataset is declared as a chain: an Asset (the
bytes) and a Dataset (their role). The two public ones the catalog uses:

```yaml validated in=real-vllm-bench/assets-datasets.yaml
apiVersion: bench/v1alpha1
kind: Asset
metadata: {name: random-prompts, description: "vllm bench's random dataset: a pure function of the seed and the lengths, nothing is stored"}
spec:
  type: generator
  license: Apache-2.0
  visibility: public
  source: {generator: {tool: vllm-bench-random, seed: 0, params: {range_ratio: 0.0}}}
---
apiVersion: bench/v1alpha1
kind: Asset
metadata: {name: slot-16-files, description: "a recorded chat set staged in the curated store under its slug"}
spec:
  type: fileset
  format: json
  visibility: internal
  source: {store: curated, slug: slot-16}
  integrity: {files: [{path: prompts.json, sha256: "0000000000000000000000000000000000000000000000000000000000000010"}]}
```

```yaml validated in=real-vllm-bench/datasets.yaml
apiVersion: bench/v1alpha1
kind: Dataset
metadata: {name: random-lengths, description: "random prompts; the lengths come from the workload"}
spec: {asset: random-prompts, role: trace}
---
apiVersion: bench/v1alpha1
kind: Dataset
metadata: {name: slot-16, description: "recorded chat prompts, replayed through the completions endpoint"}
spec: {asset: slot-16-files, role: trace}
```

`random` needs no stored bytes: it is generated from a seed and the lengths the Workload states. `slot-16` is an opaque slug: its bytes sit in the curated store under
`<slug>/`, next to a `dataset.json` that holds the real name and whether it may be shown ([guide 4](04-data-assets-models-stores.md)).

The recorded traces that the `custom` dataset replays are not public. A deployment declares them the same way, with the
bytes in the project's artifact store and the store's address and credential supplied by the launcher. This block
shows the shape with a placeholder digest (all zeros) and made-up names; it validates, and it would fail the fetch of a
real run until the digest is real:

```yaml validated base=real-vllm-bench as=deployment-trace.yaml
apiVersion: bench/v1alpha1
kind: Store
metadata: {name: datasets}
spec:
  type: core/artifactory
  endpoint: {env: STORE_BASE_URL}
  root: {env: STORE_DATASETS_PATH}
  auth: {mode: token, refs: [{env: STORE_TOKEN}]}
  visibility: internal
---
apiVersion: bench/v1alpha1
kind: Asset
metadata:
  name: recorded-trace-files
  annotations: {consumer.env: BENCH_TRACE_DATASET}
spec:
  type: fileset
  format: jsonl
  visibility: internal
  source: {store: datasets, path: <path in the datasets store>}
  integrity:
    files: [{path: trace.jsonl, sha256: "0000000000000000000000000000000000000000000000000000000000000000"}]
---
apiVersion: bench/v1alpha1
kind: Dataset
metadata: {name: recorded-trace, description: "recorded prompts, replayed in recorded order"}
spec: {asset: recorded-trace-files, role: trace}
```

`consumer.env` names the variable the benchmark's runner reads the file path from. A serving Benchmark then reads the
Dataset through an `inputs: [{name: prompts, dataset: recorded-trace}]` entry and sets `dataset-name: custom` in its
flags.

### Models

```yaml validated in=real-vllm-bench/models.yaml
apiVersion: bench/v1alpha1
kind: Model
metadata: {name: qwen2.5-0.5b-instruct}
spec:
  assets: [{role: weights, asset: qwen2.5-0.5b-instruct-weights}]
```

```yaml validated in=real-vllm-bench/assets-models.yaml
# unpinned: runner resolves the hub default
apiVersion: bench/v1alpha1
kind: Asset
metadata: {name: qwen2.5-0.5b-instruct-weights, annotations: {revision.state: unpinned}}
spec:
  type: fileset
  format: safetensors
  visibility: public
  source: {uri: 'hf://models/Qwen/Qwen2.5-0.5B-Instruct', revision: '0000000000000000000000000000000000000000'}
  integrity: {revision_pinned: true}
```

The all-zero revision is the repository's own convention for a model whose hub revision is not pinned yet (the engine
schema cannot say "unpinned" next to `revision_pinned`); the annotation marks it, and the runner resolves the hub
default. A pinned model carries a 40-hex commit, as `gemma-4-26b-a4b-it-weights` does in the same file. The weekly
Selection adds two wider models: `mistral-7b-instruct-v0.3` and `gemma-4-26b-a4b-it` (four cards).

### Metrics

The ingest reads the scalar fields of the benchmark's JSON and skips the lists behind them. The registry
(`llm-serving` pack, guide 5) names the ones a case can declare:

| JSON field | Mode | Registry metric |
|---|---|---|
| `avg_latency` | latency | `llm.batch_latency.mean` |
| `request_throughput` | serve | `llm.request_throughput` |
| `output_throughput` | serve | `llm.output_throughput` |
| `mean_ttft_ms` / `p99_ttft_ms` | serve | `llm.ttft.mean` / `llm.ttft.p99` |
| `p99_tpot_ms` | serve | `llm.tpot.p99` |
| `median_itl_ms` / `p99_itl_ms` | serve | `llm.itl.median` / `llm.itl.p99` |
| `mean_e2el_ms` / `p99_e2el_ms` | serve | `llm.e2el.mean` / `llm.e2el.p99` |
| `total_token_throughput`, `median_ttft_ms`, `mean_tpot_ms`, `median_tpot_ms`, `mean_itl_ms`, `median_e2el_ms` | serve | not registered |
| `elapsed_time`, `requests_per_second`, `tokens_per_second` | throughput | not registered |

The mapping is by convention (no code reads it). Registry values are in seconds; the field names say milliseconds because
that is how the tool and the store name them. The default metrics of the `throughput` workload are the two serve
throughputs, but a throughput run stores `requests_per_second`, `tokens_per_second` and `elapsed_time`, which have no
registered name. A latency run also stores its percentiles as `p<N>_latency`.
Each stored field is a measurement of its own and the ingest stores all of them. The registry decides which ones a case
*declares* (so planning, Results and Expectations can use them), not which ones are stored. A case declares its metrics
in the Benchmark:

```yaml validated in=real-vllm-bench/decoder/serving-qwen05b-tp1-c4.yaml
  metrics:
    required: [llm.ttft.mean, llm.ttft.p99, llm.tpot.p99, llm.itl.median, llm.itl.p99, llm.e2el.mean, llm.e2el.p99, llm.output_throughput, llm.request_throughput]
    headline: [llm.ttft.p99, llm.output_throughput]
```

The latency and throughput Benchmarks say nothing: the workload type's defaults apply (`llm.batch_latency.mean`; the
two throughputs). Declaring a field that is not registered is refused, with the nearest name suggested:

```yaml validated base=real-vllm-bench as=decoder/serving-bad.yaml expect="error:is not registered"
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: serving-bad}
spec:
  type: perf
  level: service
  tier: core
  system: {type: llm/vllm, backend: spyre}
  workload: {type: llm/serve}
  metrics: {required: [llm.tpot.median]}
  variants:
    - name: serving_bad
      system: {type: llm/vllm, backend: spyre, model: qwen2.5-0.5b-instruct, precision: fp16, tp: 1}
      workload: {type: llm/serve, endpoint: /v1/completions, load: {executor: closed, concurrency: 1, requests: 10}}
```

Registering one of the unregistered fields is a pack change ([guide 7](07-write-a-pack.md)); the ingest already stores
it.

### The workload: a Benchmark per series

The ingest names a stored benchmark after the test, so one Benchmark file carries one series name and its identity
block (guide 2) reproduces the ingest's id. The latency case:

```yaml validated file=real-vllm-bench/decoder/latency-qwen05b-tp1-in64-out64.yaml
# yaml-language-server: $schema=../../../spyre_bench/core/schemas/resources.schema.json
# Shape of the producer's latency Benchmarks; the model and the test name are substituted.
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: latency-qwen05b-tp1-in64-out64, description: 'one batch end to end (vllm bench latency)', owner: spyre-inference}
spec:
  type: perf
  level: model
  tier: sanity
  env: {DTLOG_LEVEL: error}
  system: {type: llm/vllm, backend: spyre}
  workload: {type: llm/latency}
  run:
    cache: warm
    repeats: 1
    command: [uv, run, --active, --no-sync, python3, .github/scripts/run_vllm_benchmarks.py, --configs-dir, vllm-benchmarks/benchmarks/spyre, --suite, nightly, --bench-types, latency, --tests, "{case}", --results-dir, "{raw_dir}"]
  est_hours: 0.25
  identity:
    mode: ingest
    profile: vllm
    name: latency_qwen05b_tp1_in64_out64
    discriminators:
      - {key: record_type, value: model}
      - {key: run_mode, value: latency}
      - {key: tensor_parallel, value: 1}
      - {key: input_len, value: 64}
      - {key: output_len, value: 64}
  variants:
    - name: latency_qwen05b_tp1_in64_out64
      environment: spyre-1card
      system: {type: llm/vllm, backend: spyre, token: qwen2.5-0.5b-instruct, model: qwen2.5-0.5b-instruct, precision: fp16, tp: 1, flags: {load-format: auto, max-model-len: 128}}
      workload: {type: llm/latency, shape: {batch_size: 1, input_len: 64, output_len: 64}, load: {executor: iterations, iterations: 2, warmup_iterations: 1}}
```

- The command is the producer's own runner invocation; `{case}` and `{raw_dir}` are filled in at plan and run time
  (guide 2). The runner takes `--tests {case}` from the producer's config files and rejects a name that has no entry
  there. This catalog substitutes public models for the producer's, so its cases are a template: to run one, add the
  matching entry to the producer's config as well. The sweep names below come from the ingest's parser; no producer
  config writes them yet.
- The `vllm` profile hashes five keys, `record_type`, `run_mode`, `tensor_parallel`, `input_len`, `output_len`, in that
  order. They are the tokens of the test name, so the name and the discriminators must agree.
- `variants[].system` and `workload` are the model, parallelism and shape; `environment` is the request (below).
  `out1` companions (`latency_qwen05b_tp1_in64_out1`) give the first-token latency: prefill plus one token.

A serving case sweeps concurrency. The ingest reads a sweep's `c<N>_n<N>_len<N>_seqs<N>` name tokens as four more
discriminators, so these cases use the `vllm-sweep` profile with nine keys:

```yaml validated in=real-vllm-bench/decoder/serving-qwen05b-tp1-c4.yaml
  identity:
    mode: ingest
    profile: vllm-sweep
    name: serving_qwen05b_tp1_c4_n100_len4096_seqs4
    discriminators:
      - {key: record_type, value: model}
      - {key: run_mode, value: serving}
      - {key: tensor_parallel, value: 1}
      - {key: input_len, value: ""}
      - {key: output_len, value: ""}
      - {key: concurrency, value: 4}
      - {key: num_prompts, value: 100}
      - {key: max_model_len, value: 4096}
      - {key: max_num_seqs, value: 4}
```

A key the name does not carry is `value: ""`; the ingest hashes it as empty. The serving workload names its dataset
as an input. The input is part of the engine's own id (`engine_id`); the stored id of an ingest-mode case hashes only
the name and discriminators above, so a new dataset revision does not change it:

```yaml validated in=real-vllm-bench/decoder/serving-qwen05b-tp1-c4.yaml
      workload:
        type: llm/serve
        endpoint: /v1/completions
        input_len: 64
        output_len: 64
        load: {executor: closed, concurrency: 4, requests: 100}
        inputs: [{name: prompts, dataset: random-lengths}]
        sampling: {ignore_eos: true}
        flags: {dataset-name: random, percentile-metrics: 'ttft,tpot,itl,e2el', metric-percentiles: '99,100'}
```

`percentile-metrics` and `metric-percentiles` are what make the tool report the `p99_` fields in the table above.
Weekly adds `serve_qwen05b_tp1_slot-16` (profile `vllm`, dataset `slot-16`) and the other concurrency points.

### The environment request

```yaml validated in=real-vllm-bench/environments.yaml
apiVersion: bench/v1alpha1
kind: Environment
metadata: {name: spyre-1card, description: "one card"}
spec:
  devices:
    requests: [{name: cards, device_class: spyre/pf, count: 1}]
  host: {cpu_pinning: {policy: none}}
  labels: {device.count: 1}
```

An Environment is a request (how many cards, which CPU pinning policy), not a place: the launcher picks the host (guide
3). A four-card case, like `latency_gemma26b_tp4_in64_out64`, names `spyre-4card`. The `-device-local` twins ask for
pinned CPUs and make the engine record whether the placement held.

### Check it

```bash validated
spyre-bench validate examples/real-vllm-bench
spyre-bench explain latency_qwen05b_tp1_in64_out64 --catalog examples/real-vllm-bench
spyre-bench ids --catalog examples/real-vllm-bench --inputs
```

`explain` resolves everything for one case. The part to read is `ingest_identity`: it is exactly what the ingest will
hash, and `benchmark_id` is the id the store will join on.

```text
case: latency_qwen05b_tp1_in64_out64
benchmark_id: <uuid>
id_mode: ingest
ingest_identity:
  component: spyre-inference
  name: latency_qwen05b_tp1_in64_out64
  disc: {record_type: model, run_mode: latency, tensor_parallel: '1', input_len: '64', output_len: '64'}
  profile: vllm
run_props: {run.tier: sanity, run.suite.decoder: 'true'}
```

### Nightly

The nightly invocation is a small sweep on one small model, with purpose `nightly`:

```bash validated
spyre-bench plan --catalog examples/real-vllm-bench --purpose nightly --select nightly-small --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-a-nightly.json
```

```text
ACTION  CASE                                       REPEATS  REASON
run     latency_qwen05b_tp1_in64_out1              1
run     latency_qwen05b_tp1_in64_out64             1
run     serving_qwen05b_tp1_c4_n100_len4096_seqs4  1
run     throughput_qwen05b_tp1_in64_out64          1
plan: run 4, reuse 0, skip 0
```

```json validated subset-of=plan-a-nightly.json
{
  "purpose": "nightly",
  "summary": {"cases": 4, "run": 4, "reuse": 0, "skip": 0},
  "cases": [
    {"name": "latency_qwen05b_tp1_in64_out1", "action": "run", "environment": "spyre-1card", "id_mode": "ingest",
     "id_profile": "vllm", "run_props": {"run.purpose": "nightly", "run.tier": "sanity"}}
  ]
}
```

The baseline is the store's: a run is compared with the previous runs of its series (the same `benchmark_id`, backend,
architecture and `source_job`, the name the ingest was given with `--workflow`), as guide 6 describes. Nothing in the
catalog states a window or a tolerance; those belong to the store's verdict policy.

### Weekly

The weekly invocation plans the whole decoder suite, with purpose `weekly`:

```bash validated
spyre-bench plan --catalog examples/real-vllm-bench --purpose weekly --select weekly-full --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-a-weekly.json
```

```json validated subset-of=plan-a-weekly.json
{
  "purpose": "weekly",
  "summary": {"cases": 12, "run": 12, "reuse": 0, "skip": 0},
  "cases": [
    {"name": "latency_gemma26b_tp4_in64_out1", "environment": "spyre-4card", "accelerators": 4}
  ]
}
```

The twelve cases are the four nightly cases plus the first-token and throughput companions of the wider models, the
other concurrency points and the recorded-chat case. The four-card cases carry their own request; do not set a single
`--environment` for the whole plan, which would override it. A launcher that runs one leg per card count passes
`--environment` per leg and restricts the leg to the cases of that size.

### How a CI job invokes it

The engine does not schedule. A CI job on a timer (a Jenkins job, a scheduled workflow, cron) sends one request per
run. Its shape is the same for both cadences; only the Selection and the purpose differ:

```text
# nightly                              # weekly
purpose:      nightly                  purpose:      weekly
selection:    nightly-small            selection:    weekly-full
subject:      <artifact id of the build under test>
run id:       <uuid minted by the job; every leg of the run shares it>
reuse:        never                    # a scheduled run always measures
target:       <dev or prod store>      # prod only from the default branch
```

Mapped onto the engine, the request is the `BENCH_*` variables or the same flags: `BENCH_PURPOSE`,
`BENCH_SELECTION`, `BENCH_SUBJECT_ID`, `BENCH_REUSE` and `BENCH_ENVIRONMENT` (see the [reference](../README.md)). A
manual dispatch of a workflow takes the same fields as inputs (`selection`, `purpose`, `run_id`, `target`). The job
then does five things, in this order:

1. validate the catalog and plan the Selection (the commands above);
2. run the plan; each case's command leaves the tool's JSON in the case's `raw_dir`;
3. ingest that raw output with the ingest, passing the job's run id (and, with an ingest that records it, the purpose,
   so the rows carry `run.purpose=nightly` or `weekly`);
4. read the verdict of the run (below);
5. report the job's own outcome: the verdict is advisory, so a regression does not fail the job unless the invoker
   passes `--fail-on`.

The credentials for step 3 exist only in that step; the benchmark's command never sees them (guide 8).

### Reading the result

```text
$ spyre-bench verdict --run-id <run uuid>
run <run uuid>: <n> regressed, <n> suspected, <n> improved, <n> no_baseline, <n> pass (<n> metrics)
pass        latency_qwen05b_tp1_in64_out64  avg_latency  [<tier>]  <reason>
no_baseline serving_qwen05b_tp1_c4_n100_len4096_seqs4  p99_ttft_ms  [<tier>]  <reason>
```

One line per benchmark and stored metric, judged by the store's verdict view against the series' previous runs
(guide 6). `no_baseline` means the series does not have enough prior nightly or weekly runs yet, which is where a new
case starts. `suspected` is a prompt to look, not a finding (guide 6 lists when it is used). The name and metric columns are the stored names, which is why this guide lists the
JSON fields. The text form prints no measured value, and `--quiet` drops them from `--json`.

In the store a series is a `benchmark_id` on one backend and architecture, written by one `source_job`. Cards and CPU
pinning are series labels of the engine's own series key; the stored view does not split on them yet, so keep the
`--workflow` name of a one-card leg and a four-card leg apart if their runs must not be compared.

## Part B. `spyre-perf-suite` (torch-spyre ops, hf-adapters models)

`spyre-perf-suite` runs operators and layers on the device and writes a `report.xml`. Its testcases are named
`perf_<op>_<metric>_<ms|MB>_<input shapes>`, and the ingest turns each (op, shapes) group into one stored benchmark.
Two catalogs describe it: operators in torch-spyre, models in hf-adapters. Both use the `perf-kernel` identity profile
or, where the producer's id cannot be reproduced yet, the engine's own id.

### Datasets

An operator benchmark has no Dataset resource: its input is a shape, written as the shape string the report uses
(`1_512_4096__4096_4096` is two operands). The shape is identity, so a new shape is a new benchmark. A model benchmark
names the model on the workload (`model: gpt2`); there is no Asset for it in these catalogs. If a model's weights must be pinned, declare a Model and its Asset as in part A.

### The Project and its Selections (ops)

```yaml validated in=real-perf-suite/project.yaml
apiVersion: bench/v1alpha1
kind: Selection
metadata:
  name: ops-core
  description: Default operator set.
spec:
  selector:
    suites:
      - ops
    groupings:
      tier:
        at_most: core
---
apiVersion: bench/v1alpha1
kind: Selection
metadata:
  name: ops-extended
  description: Every declared operator.
spec:
  selector:
    suites:
      - ops
    groupings:
      tier:
        at_most: extended
```

The catalog also has `smoke` (one case, never ingested) and `ops-sanity`. Nightly runs `ops-core`; weekly runs
`ops-extended`. Today every declared case is at most `core`, so the two select the same 19 cases: the weekly Selection
is there so that an operator added at tier `extended` joins the weekly run alone.

### Metrics

| Report testcase | Stored column | Notes |
|---|---|---|
| `perf_<op>_wall_clock_ms_…` | `total_duration_ms` | falls back to the `cpu` testcase when there is no wall clock |
| `perf_<op>_cpu_ms_…` | `cpu_ms` | |
| `perf_<op>_spyre_ms_…` | `spyre_ms` | |
| `perf_<op>_kernel_ms_…` | `kernel_mean_ms` | the kernel's mean time per run; the registry name is `kernel.time.mean` |
| `perf_<op>_memory_transfer_ms_…` | `memory_transfer_mean_ms` | |
| `perf_<op>_compile_ms_…` (op reports also spell it `compiler`) | `compile_ms` | |
| `perf_<op>_runtime_ms_…` | `runtime_ms` | |
| `perf_<op>_mem_size_MB_…` | `mem_size_mb` | a footprint, not a duration |
| tags `pt_util`, `ratio` on a testcase | `pt_util_percent`, `ratio` | `pt_util_percent` needs the compiler to emit ideal cycles |

The `kernel` pack registers `kernel.time.mean` (the default of the `kernel/op` workload) and `kernel.total_duration`
(the default of `kernel/layer`). The other columns are stored by the ingest but have no registry name; registering one is
a pack change ([guide 7](07-write-a-pack.md)). An operator Benchmark leaves `metrics` out and takes the default.

### The workload: an operator Benchmark

```yaml validated file=real-perf-suite/ops/matmul.yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/torch-spyre/torch-spyre/main/extensions/spyre-bench/spyre_bench/core/schemas/resources.schema.json
apiVersion: bench/v1alpha1
kind: Benchmark
metadata:
  name: matmul
spec:
  type: perf
  level: kernel
  tier: sanity
  system: op-library
  est_hours: 0.1
  workload:
    type: kernel/op
    op: matmul
    load:
      executor: iterations
      iterations: 100
      warmup_iterations: 3
  run:
    est: 6m
    command:
      - python3
      - benchmarks/tools/run_kernel_case.py
      - --op
      - '{benchmark}'
      - --shape
      - '{case.flags.input_shapes}'
      - --suite-dir
      - '{out_dir}/suite'
      - --raw-dir
      - '{raw_dir}'
      - --repeat-dir
      - '{repeat_dir}'
  identity:
    mode: ingest
    profile: perf-kernel
    discriminators:
      - {key: record_type, value: op}
      - {key: config_name, value: ''}
      - {key: input_shapes, from: case.flags.input_shapes}
      - {key: run_mode, value: op_benchmark}
      - {key: kernel_name, value: ''}
      - {key: is_total, value: ''}
  matrix:
    name_template: matmul_{shape}
    axes:
      shape:
        - token: prefill-n1024
          patch: {workload: {flags: {input_shapes: '[[1, 512, 4096], [4096, 1024]]'}}}
        - token: prefill-n4096
          patch: {workload: {flags: {input_shapes: '[[1, 512, 4096], [4096, 4096]]'}}}
        - token: prefill-n12800
          patch: {workload: {flags: {input_shapes: '[[1, 512, 4096], [4096, 12800]]'}}}
        - token: decode-n1024
          patch: {workload: {flags: {input_shapes: '[[4, 1, 4096], [4096, 1024]]'}}}
        - token: decode-n4096
          patch: {workload: {flags: {input_shapes: '[[4, 1, 4096], [4096, 4096]]'}}}
        - token: decode-n12800
          patch: {workload: {flags: {input_shapes: '[[4, 1, 4096], [4096, 12800]]'}}}
```

- The Benchmark name is the operation name the report uses, because the ingest hashes it. One matrix axis lists the
  shapes; each becomes a case `matmul_prefill-n1024`, `matmul_decode-n4096`, and so on. Replace the shape strings with the report's own when you add a
  real operator: a case whose operation and shape the report does not contain fails at run time instead of ingesting a
  row.
- `perf-kernel` hashes six keys. An operator-level row has an empty `kernel_name` and `is_total`; per-kernel and total rows
  are written by the ingest itself and are not cases.
- `system: op-library` is the System in `systems.yaml`: the operator library under test, with `backend: spyre` as the
  unhashed comparison axis.
- Cost lives in `est_hours`; the iteration counts are the workload.

The Environment is one device:

```yaml validated in=real-perf-suite/project.yaml
apiVersion: bench/v1alpha1
kind: Environment
metadata:
  name: one-device
  description: One accelerator device; the placer decides where.
spec:
  devices:
    requests:
      - name: cards
        device_class: spyre/pf
        count: 1
  labels:
    device.count: 1
```

### The models (hf-adapters)

hf-adapters keeps the model list in one file and generates a Benchmark per model. Each model expands to six cases, two
`config` values (`full-model`, `one-block`) by three `mode` values (`prefill-only`, `decode-only`, `generate`):

```yaml validated in=real-perf-suite-models/hf-models/gpt2.yaml
spec:
  type: perf
  level: layer
  tier: sanity
  system: hf-models-spyre-fp16
  workload:
    type: kernel/layer
    model: "gpt2"
    layer: full_model
    flags: {run_mode: prefill_only}
    load: {executor: iterations, iterations: 5}
  matrix:
    axes:
      config:
        - {token: full-model, patch: {workload: {layer: full_model}}}
        - {token: one-block, patch: {workload: {layer: one_block}}}
      mode:
        - {token: prefill-only, patch: {workload: {flags: {run_mode: prefill_only}}}}
        - {token: decode-only, patch: {workload: {flags: {run_mode: decode_only}}}}
        - {token: generate, patch: {workload: {flags: {run_mode: generate}}}}
```

The tier follows the producer's model list: `core` for the two larger models, `extended` for the rest, `sanity` for the
one model the `smoke` Selection uses. The default metric of `kernel/layer` is `kernel.total_duration`. These Benchmarks
have no `identity` block and no `run.command`: the hf-adapters definitions' own README says nothing there runs in CI
yet and that the engine's ids differ from the existing model rows' until it can pin a legacy discriminator set. They
validate, explain and plan; the engine does not run them. Do not read their ids as the stored ones.

```yaml validated in=real-perf-suite-models/project.yaml
apiVersion: bench/v1alpha1
kind: Selection
metadata: {name: models-nightly, description: "the larger models: full model and one block, all three modes"}
spec:
  selector: {types: [perf], groupings: {tier: {eq: core}}}
---
apiVersion: bench/v1alpha1
kind: Selection
metadata: {name: models-weekly, description: "every declared model"}
spec:
  selector: {types: [perf], groupings: {tier: {at_most: extended}}}
```

### Check it

```bash validated
spyre-bench validate examples/real-perf-suite
spyre-bench explain matmul_prefill-n4096 --catalog examples/real-perf-suite
spyre-bench ids --catalog examples/real-perf-suite --inputs
spyre-bench validate examples/real-perf-suite-models
spyre-bench explain "gpt2/config=one-block/mode=prefill-only" --catalog examples/real-perf-suite-models
```

The ingest identity of `matmul_prefill-n4096` reads:

```text
ingest_identity:
  component: torch-spyre
  name: matmul
  disc: {record_type: op, config_name: '', input_shapes: '[[1, 512, 4096], [4096, 4096]]', run_mode: op_benchmark, kernel_name: '', is_total: ''}
  profile: perf-kernel
```

The producer repository also has a conformance check that recomputes these ids with the ingest's own report parser over a
synthetic report and compares them ([guide 10](10-contribute.md)). Run it whenever an `identity` block changes.

### Nightly

```bash validated
spyre-bench plan --catalog examples/real-perf-suite --purpose nightly --select ops-core --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-b-ops-nightly.json
spyre-bench plan --catalog examples/real-perf-suite-models --purpose nightly --select models-nightly --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-b-models-nightly.json
```

```json validated subset-of=plan-b-ops-nightly.json
{
  "purpose": "nightly",
  "summary": {"cases": 19, "run": 19, "reuse": 0, "skip": 0},
  "cases": [{"name": "attention_prefill", "environment": "one-device", "id_mode": "ingest", "id_profile": "perf-kernel"}]
}
```

```json validated subset-of=plan-b-models-nightly.json
{"purpose": "nightly", "summary": {"cases": 12, "run": 12}}
```

```text
ACTION  CASE                                                               REPEATS  REASON
run     gemma-4-26b-a4b-it/config=full-model/mode=prefill-only             1
...
plan: run 12, reuse 0, skip 0
```

### Weekly

```bash validated
spyre-bench plan --catalog examples/real-perf-suite --purpose weekly --select ops-extended --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-b-ops-weekly.json
spyre-bench plan --catalog examples/real-perf-suite-models --purpose weekly --select models-weekly --arch x86_64 \
    --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan-b-models-weekly.json
```

```json validated subset-of=plan-b-models-weekly.json
{"purpose": "weekly", "summary": {"cases": 36, "run": 36}}
```

### How a CI job invokes it

The request has the same shape as in part A (purpose, Selection, subject, run id, reuse, target); only the names differ:
nightly sends `ops-core` (and `models-nightly` for hf-adapters), weekly sends `ops-extended` (and `models-weekly`).
The operator job's steps are the producer's make targets: validate and check the ids, run the plan, collect each case's
`report.xml` into one directory, ingest it with the run id and the purpose, and read the verdict. The ingest call is
the same one the producer's perf job already makes. An ingest that records the run's purpose takes it as one more
option (`--purpose nightly` or `--purpose weekly`); the option is not in every ingest version:

```text
python -m spyre_clickhouse_ingest results --xml-dir <collected reports> --workflow <series name> \
    --branch <branch> --sha <sha> --run-id <engine run id> --trigger-type perf \
    --arch x86_64 --component torch-spyre --artifact-id <subject id> --purpose nightly --schema v2
```

`--workflow` is the series name; pick one that differs from the producer's regular perf job so the two do not share a
series. Credentials are passed to this step only.

### Reading the result

```text
$ spyre-bench verdict --run-id <run uuid>
run <run uuid>: <n> regressed, <n> suspected, <n> improved, <n> no_baseline, <n> pass (<n> metrics)
pass        matmul  kernel_mean_ms  [<tier>]  <reason>
pass        matmul  memory_transfer_mean_ms  [<tier>]  <reason>
```

One line per operator and stored column. A new shape starts as `no_baseline`; a run that missed the validity bar is
`suspected`. The operator name is the stored name (`matmul`), and the shape is part of the series
key, so two shapes of one operator are two series.

## Checklist for adding your own

1. Start from the nearest file above; keep the file name equal to `metadata.name`.
2. Dataset: Asset (pinned) then Dataset; the store's address and credential stay in variables.
3. Metric: declare registered names in `metrics`; check the table before expecting a stored column to have a registry
   name.
4. Workload and identity: the discriminators agree with the producer's test or operation name; run `explain` and `ids`.
5. Environment: a request, not a host.
6. Nightly and weekly: a Selection each, named for what it runs. The timer, the window and the tolerance are not in the
   catalog.
7. `validate`, `plan` for both Selections, and the producer's conformance check.
