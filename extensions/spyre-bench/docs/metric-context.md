# Metric context

A number is only meaningful with the facts that produced it: cold or warm, load, sequence lengths, data type, build,
device. Metric context makes those facts part of the metric declaration, so a number is never shown bare and two
numbers are compared only when their context agrees.

## Vocabulary

`spyre_bench/core/metric-context.yaml` is the controlled vocabulary.

- A **facet** is one piece of context (`cache_state`, `input_len`, `dtype`, `expert_concurrency`). It has a group
  (workload, state, build, environment, method, judgement), a type, an attribute-registry key when one exists, and an
  `auto` source: `engine`, `probe`, `ingest`, `derive` or `declared`.
- A **metric class** (`throughput`, `latency_itl`, `kernel_time`, `design_parameter`, ...) lists the facets its numbers
  must carry. `a|b` means either facet satisfies the requirement. A `when: key=value` facet applies unless that key is
  known and differs. A `compare: note` facet must be recorded but never blocks a comparison.
- **Required means known**, not a particular value: a shared node (`clean=none`) is complete, `unknown` or absent is
  missing.

## Declaring it in a pack

```yaml
metrics:
  - {name: llm.itl.median, class: latency_itl, unit: s, direction: lower_is_better, category: performance}
  - {name: x.rate, class: throughput, context: {required: [model, dtype]}, unit: "{token}/s", ...}
```

`class` picks the metric class; without it the class is matched from the metric name. `context.required` replaces the
class default for that metric, `context.optional` adds facets shown only in the expanded view. Unknown classes and
facets are pack errors.

## Declaring it in a Benchmark

Facets only the author knows (`token_basis`, `timer`, `ttft_def`, `itl_def`, `sampling`, `ratio_basis`) go in an
optional `context:` block. It never enters the benchmark id.

```yaml
spec:
  context: {timer: wall}
```

## The lint

`spyre-bench validate` warns, never fails, when a Benchmark's required metrics need an author-declared facet that its
`context:` block does not state, when a metric matches no class, or when `context:` names an unknown facet. Facets the
engine, probes, ingest or derivation supply are not checked.

## Rendering

`spyre_bench.metric_context` renders a record `{metric, value, unit, statistic, n, ctx}`:

- `context_line` prints one chip per required facet in a fixed order. A missing facet is an empty chip (`KV -`) and
  `render` appends `[CONTEXT INCOMPLETE: <facets>]`. The number is always shown.
- `expanded_view` groups facets and flags each as derived or `required, not recorded`.
- `compare` returns `comparable`, `differs` (differences listed), `incomplete` or `different_metric`. Incomplete
  numbers are excluded from verdicts. `vary` names the facets the comparison is about (default `build`).
- `page_header` hoists chips equal for every number on a page.

Derived facets: `kv_span` (requested range `input_len` to `input_len + output_len`), `cache_state` (from
`compile_cache` and `shape_warmup`), `statistic` and `sample_n`.
