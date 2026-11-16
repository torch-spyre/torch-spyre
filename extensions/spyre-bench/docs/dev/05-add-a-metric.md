# 5. Add a metric

A metric is a name in a registry plus a path by which numbers reach it. Both halves matter: a number nobody registered
has no unit, no direction and no noise policy, so nothing can judge it; a registered metric nobody emits is an empty
column. This guide adds `mq.errors.ratio` (failed sends divided by attempted sends) to the example `docs-queue` pack and
follows it from the registry to a verdict. The files are `examples/packs/docs-queue/pack.yaml` and `examples/docs-pack`.

```bash validated
spyre-bench validate examples/docs-pack --pack examples/packs/docs-queue
```

## Step 1. Register it

Metrics are registered by packs. Add one entry to the pack that owns the domain (for a metric no pack owns, add a new
pack: [guide 7](07-write-a-pack.md)).

```yaml validated in=packs/docs-queue/pack.yaml
    - {name: mq.errors.ratio, unit: "1", direction: lower_is_better, category: performance, statistic: ratio, compare: absolute,
       noise: {min_repeats: 3}, description: "failed sends divided by attempted sends"}
```

| Field | Meaning and rules |
|---|---|
| `name` | dot-namespaced, lower-case, the pack's prefix first (`mq.`), **no unit in the name** (`mq.latency.p99`, not `mq.latency_ms.p99`) |
| `unit` | a UCUM string: `s`, `By`, `W`, `1`, `{request}/s`, `{token}/s`. Values are stored in **base units**: seconds, bytes, never milliseconds |
| `direction` | `lower_is_better`, `higher_is_better` or `none` |
| `category` | `performance`, `efficiency`, `capability`, `quality` or `diagnostic`; a `perf` Benchmark accepts `performance`, `efficiency`, `diagnostic`; a `quality` Benchmark accepts `quality`, `diagnostic` |
| `statistic` | what the number *is* (`value`, `mean`, `median`, `p90`, `p99`, `max`, `min`, `count`, `ratio`, `sum`); documents the meaning, it does not compute anything |
| `compare` | how two values are compared: `relative` (percent), `absolute` or `exact` |
| `noise` | `min_repeats` below which a comparison is not trusted; optional `cv_max` (a fraction) |
| `otel` | optional OpenTelemetry semantic-convention name, for exporters |

Pick the direction carefully. `none` means "context, never judged": an Expectation on such a metric is a validation error.
Pick the category to match the test type that will use it:

```yaml validated base=docs-pack as=queue/expectations-bad.yaml pack=packs/docs-queue expect="error:has no direction, so it cannot be judged"
apiVersion: bench/v1alpha1
kind: Expectation
metadata: {name: depth-expectation}
spec:
  selector: {benchmarks: [produce]}
  metric: mq.depth.max
  baseline: {kind: fixed, value: 100}
  tolerance: {pct: 5}
```

When you change a pack, raise its `version` and keep the Project's range compatible (`packs: [{name: docs-queue,
version: "0.1"}]` accepts any `0.1.x`). Adding a metric is a compatible change; renaming or re-uniting one is not (a
new name is a new metric with no history).

## Step 2. Use it in a Benchmark

A Benchmark declares which metrics a case reports:

```yaml validated in=docs-pack/queue/produce.yaml
  metrics: {required: [mq.throughput, mq.latency.p99, mq.errors.ratio], headline: [mq.throughput]}
```

`required` is the set the case is expected to record, `headline` marks the ones to show first (it must be a subset of
`required`). The workload type's `default_metrics` apply when the Benchmark says nothing. Validation checks that every
name is registered and fits the category rule, and it suggests the nearest name when you misspell one:

```yaml validated base=docs-pack as=queue/produce-typo.yaml pack=packs/docs-queue expect="error:did you mean 'mq.errors.ratio'?"
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: produce-typo}
spec:
  type: perf
  level: service
  tier: sanity
  system: broker-1p
  workload: {type: mq/produce, message_bytes: 128, load: {executor: closed, concurrency: 1, duration: 1m}}
  metrics: {required: [mq.errors.ratios]}
  variants: [{name: produce_typo}]
```

## Step 3. Emit it

The engine parses no tool's output format. Your command writes the numbers to the **sidecar**,
`{repeat_dir}/metrics.json`, once per repeat:

```json validated sidecar
{"samples": {"mq.throughput": [49504.9], "mq.latency.p99": [0.00404], "mq.errors.ratio": [0.0001], "mq.depth.max": [120]}}
```

- `samples` maps a metric name to a list of finite numbers in base units. A non-list or a non-finite value is dropped
  with a note in the Result. The Result's per-case value is the median across repeats of each repeat's median sample.
- Unregistered names are kept, with unit `1`, no direction and no noise policy. The engine does not stop you; a
  metric you meant to compare must be registered.
- A file over `--max-metrics-bytes` (8 MiB by default) is ignored with a note. Emit aggregates, not per-request logs.
- Tool output that the *ingest* reads (a JSON report, a JUnit file) goes in `{raw_dir}`; that is a separate path (below).

The stub in `examples/docs-pack/tools/emit_queue.py` writes both:

```python validated in=docs-pack/tools/emit_queue.py
scale = 1 + a.repeat / 100
raw = Path(a.raw)
raw.mkdir(parents=True, exist_ok=True)
(raw / f"produce-{a.repeat}.json").write_text(json.dumps({"bytes": a.bytes, "msgs_per_s": 50000 / scale}))
(Path(a.out) / "metrics.json").write_text(
    json.dumps(
        {
            "samples": {
                "mq.throughput": [50000 / scale],
                "mq.latency.p99": [0.004 * scale],
                "mq.errors.ratio": [0.0001 * a.repeat],
                "mq.depth.max": [120 * a.repeat],
            }
        }
    )
)
```

Run it and read the aggregate the engine built:

```bash validated
spyre-bench plan --catalog examples/docs-pack --pack examples/packs/docs-queue --purpose nightly --select smoke \
    --arch x86_64 --subject-id 9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2 --out plan.json
spyre-bench run --plan plan.json --catalog examples/docs-pack --pack examples/packs/docs-queue \
    --workdir examples/docs-pack --out results/queue
```

```json validated subset-of=results/queue/cases/*/result.json
{"status": {"aggregate": {"statistic": "median", "n": 1, "metrics": {"mq.errors.ratio": {"value": 0.0001, "unit": "1"}, "mq.depth.max": {"unit": "{message}"}}}}}
```

## Step 4. Make it reach the store

(The table below states the ingest's behaviour as read from `vllm.py`, `results.py` and `bundle.py`; the ingest, not the engine, owns it.) The sidecar feeds the engine's Result. Rows in the store come from the **ingest**, which reads `{raw_dir}`. What the
ingest keeps depends on the case's identity profile (guide 2), and this is where a new metric most often stalls:

| Profile | Where numbers come from | A new metric needs |
|---|---|---|
| `generic` (bundle `benchmarks` or `tables`) | the `measurements` object of each row: `metric: number` or `metric: [samples]`, finite numbers (see `extensions/clickhouse-ingest/docs/table-formats.md`) | nothing in the ingest: the name is stored as given; a stored verdict also needs the view policy to list it |
| `vllm`, `vllm-sweep` | the scalar fields of the serving benchmark's JSON (latency, throughput, `mean_`/`median_`/`p99_` of ttft, tpot, itl, e2el) | a change to the ingest's extractor: other fields are skipped |
| `perf-kernel` | testcases of the kernel report whose names match the ingest's metric patterns | a change to the ingest's parser and, for a new column, to the schema |

[Guide 12](12-worked-example-nightly-weekly.md) maps each scalar field of `vllm bench` and each column of
`spyre-perf-suite` to its registry name, and shows which have none.

So the recipe is: if your harness can emit `generic` rows, a new metric is a pack edit plus a line in your command. If
the rows come from a producer with a fixed field list, the metric also needs an ingest change in the same pull
request or a follow-up, reviewed by the ingest's owners ([guide 10](10-contribute.md)). Stored metric names follow the
producer's conventions; the registry name is what Results, Expectations and Analyses use, so keep a mapping in your
harness when they differ. The stored view policy keys on the stored names (`kernel_mean_ms`, `mean_ttft_ms`), so a metric gets a stored verdict only when that policy lists it.

## Step 5. Decide how noisy it is

Three noise controls exist, at three levels. Use the lowest that is enough.

1. **The metric** (`noise: {min_repeats, cv_max}`): the repeats below which a comparison of this metric is not trusted.
2. **The purpose** (ValidityPolicy `requires.max_cv_pct`, `min_repeats`, `discard_warmup`): a run whose repeat-to-repeat
   variation is above the bar is capped at `suspect` and never becomes a baseline.
3. **The Expectation** (`tolerance.pct`, `noise: {sigmas, floor_pct, source}`): the size of change that counts. The
   effective floor is the largest of `tolerance.pct`, `noise.floor_pct` and `sigmas` times the noise estimate, where
   the estimate is the baseline's scaled median absolute deviation as a percentage (`baseline_mad`), the series' coefficient of variation
   (`series_cv`) or the node qualification (`qualification`).

An Expectation judges a metric the case actually records:

```yaml validated in=docs-pack/queue/expectations.yaml
apiVersion: bench/v1alpha1
kind: Expectation
metadata: {name: produce-throughput, owner: example-team, description: "throughput must not fall more than 5% below the last 20 valid nightly runs of the same series"}
spec:
  selector: {benchmarks: [produce]}
  metric: mq.throughput
  baseline: {kind: rolling, window: 20, min_n: 10, purposes: [nightly, weekly], match: {environment: id}}
  tolerance: {pct: 5}
  noise: {sigmas: 2, source: baseline_mad}
  severity: advisory
```

A metric that is registered but not in the selected cases' `required` list (or the workload type's defaults) is an error,
so an Expectation can never silently judge nothing:

```yaml validated base=docs-pack as=queue/produce-minimal.yaml pack=packs/docs-queue expect="error:is not recorded by case"
apiVersion: bench/v1alpha1
kind: Benchmark
metadata: {name: produce-minimal}
spec:
  type: perf
  level: service
  tier: sanity
  system: broker-1p
  workload: {type: mq/produce, message_bytes: 128, load: {executor: closed, concurrency: 1, duration: 1m}}
  metrics: {required: [mq.throughput]}
  variants: [{name: produce_minimal}]
---
apiVersion: bench/v1alpha1
kind: Expectation
metadata: {name: latency-on-minimal}
spec:
  selector: {benchmarks: [produce-minimal]}
  metric: mq.latency.p99
  baseline: {kind: fixed, value: 0.005}
  tolerance: {pct: 5}
```

The reference judge (`engine.judge`) caps a verdict built from a single repeat at `suspected` (and a single-repeat improvement reads as `pass`): one
repeat cannot show a regression.

## Step 6. Derived metrics

When a number is a function of other cases, do not emit it from the command; compute it in an Analysis with `derive`
(guide 6). The derived name must not shadow a registered metric, and it may be a metric of its own if you register it
too (then an Expectation can judge it). Typical derivations: a per-token cost from two output lengths (`slope`), the
cost of a step (`difference`), a speedup (`ratio`).

## Checklist

1. The name is dot-namespaced, unit-free; the unit is UCUM and the value is in base units.
2. Direction and category are right; `none` for context only.
3. The pack `version` is raised; a compatible Project range still loads it.
4. Your command writes it to `metrics.json`; the Benchmark lists it in `metrics.required`.
5. The identity profile's ingest path stores it, or the change to the ingest is in a pull request.
6. An Expectation exists only after the series has enough valid runs; start with `severity: informational` or
   `advisory`.
