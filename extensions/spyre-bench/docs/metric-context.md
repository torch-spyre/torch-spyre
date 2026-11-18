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
- **Required means known**, not a particular value: a shared node (`clean=none`) is complete. Absent values and
  placeholders are missing: empty or blank text, `unknown`, `unset`, `null`, `n/a`, `?`, `-`, an em dash, NaN or
  infinity (including numpy scalars), and empty containers. `clean=unknown` is missing, because the design reads
  `unknown` as "not determined".
- **`none`** is a value only for facets that list it in `values` (`clean`, `cpu_pinning`) or set `none_ok: true`
  (`device_type`, `quant_scheme`, `baseline`, `mem_alloc_policy`). For every other facet, `model` included, `none` and
  `n/a` are placeholders.
- **Conditional facets** (`when: backend=spyre`) apply unless the condition key is known and differs. In `compare`, a conditional facet applies if it applies to either side, so the result does not depend on the order
  of the two numbers. With the backend
  unknown, both the backend and the conditional facet are listed as missing.

## Declaring it in a pack

```yaml
metrics:
  - {name: llm.itl.median, class: latency_itl, unit: s, direction: lower_is_better, category: performance}
  - {name: x.rate, class: throughput, context: {required: [model, dtype]}, unit: "{token}/s", ...}
```

`class` picks the metric class; without it, a metric of the engine's own packs is matched from its name, and a metric
of any other pack is `unclassified`: no requirements and no lint output. `context.required` replaces the
class default for that metric, `context.optional` replaces the facets shown only in the expanded view. The renderer
honors these when it is given the pack's metrics: `Vocab.bind(registry.metrics)` returns a vocabulary whose
`missing`, `chips`, `expanded_view`, `completeness` and `compare` resolve through the declarations. A plain `Vocab` falls
back to the name-glob match.

Classes and facets are a closed list: an unknown class or facet in a pack is a pack error. A pack that needs its own
names uses the `x-` prefix (`class: x-mine`, `context.required: [x-foo]`). `x-` classes and facets are accepted and carry
no requirements. Metrics that are counts or diagnostics use the `diagnostic` class, which requires nothing; `comms.*`
metrics use `collective`.

## Declaring it in a Benchmark

Facets only the author knows (`token_basis`, `timer`, `ttft_def`, `itl_def`, `sampling`, `ratio_basis`) go in an
optional `context:` block. It never enters the benchmark id. The block is validated and linted, but result records do
not read it yet: a record carries its own `ctx`.

```yaml
spec:
  context: {timer: wall}
```

## The lint

`spyre-bench validate` never fails on metric context. By default it warns when `context:` names an unknown facet or gives an enum facet a value outside its list, and
prints one summary line when Benchmarks have undeclared author-only facets or metrics without a class. `--lint-context`
lists every one. Facets the engine, probes, ingest or derivation supply are not checked.

## YAML booleans in `context:`

YAML 1.1 reads an unquoted `off` as `false` and `on` as `true`. A `context:` value `prefix_cache: off` is therefore
taken as `false`, which the vocabulary, renderer and `compare` treat as `off`. `true` is ambiguous (`on_cold` or
`on_warm`), so the lint warns; quote the value you mean. An empty value (YAML null) also warns. Quote enum values.

A container in a record's `ctx` holding only unrecorded leaves (`[nan]`, `[None]`) is unrecorded. A container with
values is known but not renderable as a single value, so it shows as a count (`TP 2 values`).
A container that contains itself, nests deeper than 8 levels, or holds more than 10,000 items is malformed and
reads as unrecorded. Ordinary exceptions raised by duck-typed values are swallowed; `BaseException`s propagate.

## Vocabulary changes to mirror in other readers

A reader that ports the vocabulary (the dashboard) must apply these:

- `clean` has the values `enforced`, `observed`, `none`. The raw run property `run.clean.exclusive` still allows
  `unknown` (it is a recorded raw value); a raw `unknown` maps to the facet being missing. There is no `clean?` chip.
- `prefix_cache` has the values `off`, `on_cold`, `on_warm`; boolean `false` is `off`.
- `none_ok: true` marks facets where `none` is a real answer: `device_type`, `quant_scheme`, `baseline`,
  `mem_alloc_policy`. `none` is also legal for facets that list it in `values`.
- Classes `diagnostic` (no requirements) and `collective` (`comms.*`) exist; `x-` classes and facets carry no
  requirements.
- A metric of a non-built-in pack with no `class` is `unclassified`: no requirements. If it declares
  `context.required`, those requirements apply and are linted.

## Rendering

`spyre_bench.metric_context` renders a record `{metric, value, unit, statistic, n, ctx}`:

- `context_line` prints one chip per required facet in a fixed order. A missing facet is an empty chip (`KV —`, an em dash) and
  `render` appends `[CONTEXT INCOMPLETE: <facets>]`. The number is always shown.
- `expanded_view` groups facets and flags each as derived or `required, not recorded`.
- `compare(a, b, vocab, vary=...)` (`vary` is a facet name or a collection of them; numeric facets compare as numbers) returns `comparable`, `differs` (differences listed), `incomplete` or `different_metric`. Incomplete
  numbers are excluded from verdicts. `vary` names the facets the comparison is about (default `build`).
- `page_header` hoists chips equal for every number on a page.

Derived facets: `kv_span` (requested range `input_len` to `input_len + output_len`), `cache_state` (from
`compile_cache` and `shape_warmup`), `statistic` and `sample_n`.
