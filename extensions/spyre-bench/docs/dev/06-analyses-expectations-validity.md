# 6. Write an Analysis, an Expectation and a ValidityPolicy

Three kinds turn stored numbers into decisions:

| Kind | Question | Runs? |
|---|---|---|
| ValidityPolicy | is this result trustworthy enough to use for this purpose? | judged when a Result is built |
| Expectation | did this metric get worse than it should have? | validated and digested by the engine; judged by `engine.judge` (reference rules) and, for stored verdicts today, by the store's verdict view |
| Analysis | why did it change, and what does a variable cost? | a query over stored Results; runs nothing |

The files are in `examples/docs-analysis`:

```text
examples/docs-analysis/
  validity-policy.yaml       three purposes: pr, nightly, reference
  expectations.yaml          a rolling and a fixed Expectation
  analysis/pinning-effect.yaml   derive a per-element cost, compare it across CPU pinning policies
examples/results/docs-analysis.results.json   synthetic Results the Analysis reads
```

## ValidityPolicy: what earns `valid`

```yaml validated in=docs-analysis/validity-policy.yaml
apiVersion: bench/v1alpha1
kind: ValidityPolicy
metadata: {name: docs-default, owner: example-team}
spec:
  purposes:
    pr:
      description: "ordinary CI: usable for a PR check, never a baseline"
      floor: shared_env
      requires: {qualification: any, clean: any, min_repeats: 1, max_cv_pct: 10, discard_warmup: 1, placement: optional}
      baseline: false
      reference: false
    nightly:
      floor: shared_env
      requires: {qualification: current, clean: any, min_repeats: 3, max_cv_pct: 5, discard_warmup: 1, placement: required}
      baseline: true
      reference: false
    reference:
      floor: valid
      requires: {qualification: current, clean: enforced, min_repeats: 5, max_cv_pct: 2, discard_warmup: 2, placement: required}
      baseline: true
      reference: true
```

Per purpose:

- **`requires`** are the gates for `valid`. Missing one *caps* the validity; it never fails the run:

| Gate | Satisfied when | Cap and reason when missed |
|---|---|---|
| `qualification: current` | the node has a passing probe within its NodeClass `expiry_days` | `unqualified_env`, `qualification_not_current` |
| `clean: clean \| enforced` | `enforced` exclusive access, or (`clean` only) observed with a clean load check and enforced device exclusivity | `shared_env`, `clean_<need>_not_met` |
| `min_repeats` | at least that many ok repeats | `suspect`, `repeats_below_min` |
| `max_cv_pct` | the repeat-to-repeat coefficient of variation is at most this | `suspect`, `cv_above_max` |
| `discard_warmup` | at least that many warm-up iterations were discarded | `suspect`, `warmup_not_discarded` |
| `placement: required` | a placement was observed | `suspect`, `placement_missing` |
| pinning (from the Environment) | a `require_verified` request was observed to hold | `suspect`, `pinning_violated` / `pinning_unverified` |

  Other caps: `gate_failed` (a pack gate such as a card health check failed, `suspect`), `not_measured` (`invalid`) and
  external trust (`external`). The order, worst first, is `invalid`, `suspect`, `external`, `unqualified_env`,
  `shared_env`, `valid`. A result gets the worst cap that applies.
- **`floor`** is the lowest validity the result may have to be *used* for this purpose (reuse, comparison). An Environment's
  `policy.validity_floor` may only tighten it.
- **`baseline: true`** lets `valid`, internal results of this purpose serve as baselines. Results from external
  sources never become baselines or earn reference grade, whatever the purpose.
- **`reference: true`** lets such results earn reference grade.

The validator enforces the rules that keep this coherent:

```yaml validated base=docs-analysis as=bad-policy.yaml expect="error:baselines need qualification current"
apiVersion: bench/v1alpha1
kind: ValidityPolicy
metadata: {name: lax}
spec:
  purposes:
    nightly:
      floor: shared_env
      requires: {qualification: any, clean: any, min_repeats: 3, max_cv_pct: 5, discard_warmup: 1, placement: required}
      baseline: true
      reference: false
```

```yaml validated base=docs-analysis as=bad-policy.yaml expect="error:a PR run can never be a baseline or reference"
apiVersion: bench/v1alpha1
kind: ValidityPolicy
metadata: {name: pr-baseline}
spec:
  purposes:
    pr:
      floor: shared_env
      requires: {qualification: current, clean: any, min_repeats: 1, discard_warmup: 0, placement: optional}
      baseline: true
      reference: false
```

A reference purpose needs `floor: valid`, `qualification: current`, a real clean requirement (not `any`) and
`baseline: true`. One ValidityPolicy normally lives in a central catalog; a catalog's own policy wins over a central
one, and with several loaded the Project must name one with `validity_policy`. Purposes are the registered vocabulary
(`pr`, `nightly`, `weekly`, `release`, `reference`, `qualification`, `investigate`).

## Expectation: judging one metric

```yaml validated in=docs-analysis/expectations.yaml
apiVersion: bench/v1alpha1
kind: Expectation
metadata: {name: scale-time-no-regression, owner: example-team}
spec:
  selector: {benchmarks: [op-scale]}
  metric: kernel.time.mean
  baseline: {kind: rolling, window: 20, min_n: 10, purposes: [nightly], match: {environment: id}}
  tolerance: {pct: 5}
  noise: {sigmas: 2, source: baseline_mad}
  severity: advisory
```

- **`selector`** picks the cases it judges: `benchmarks`, `cases`, `suites`, `types`, `levels`, `workload_types`,
  `groupings`. It must select at least one case, and each selected case must record the metric.
- **`baseline.kind: rolling`** compares with the last `window` prior valid runs of the listed `purposes` in the same
  series, requiring `min_n` of them (fewer gives `no_baseline`). `match.environment` is `id` (the same `env_id`) or
  `{labels: [env.label.device.count]}` for a relaxed match. A candidate baseline must be `valid`, from internal trust,
  of a purpose with `baseline: true`, not invalidated, with the same pinning label.
- **`baseline.kind: fixed`** compares with a number you state in base units. This is the **anchor**: a value somebody
  signed off (a vendor figure, an agreed target, a reference measurement), independent of the series' history.

```yaml validated in=docs-analysis/expectations.yaml
apiVersion: bench/v1alpha1
kind: Expectation
metadata: {name: scale-time-vs-target, owner: example-team, description: "a fixed target in base units, for a number someone signed off"}
spec:
  selector: {cases: [op_scale_n4096]}
  metric: kernel.time.mean
  baseline: {kind: fixed, value: 0.005}
  tolerance: {pct: 10}
  severity: informational
```

- **`tolerance.pct`** is the worse-than-baseline change that is acceptable. Direction comes from the metric registry,
  so "worse" is slower for `lower_is_better` and lower for `higher_is_better`.
- **`noise`** keeps the floor from being tighter than the data: `sigmas` times a noise estimate (`baseline_mad`: the
  scaled median absolute deviation of the baseline as a percentage, `series_cv`, or `qualification`), and an optional
  `floor_pct`.
- **`severity`**: `informational`, `advisory` (the default) or `gating`. Severity is recorded with the verdict; whether a
  verdict blocks anything is decided by whoever consumes it. Start advisory.

**What judges an Expectation today.** The engine validates Expectations, digests them into the plan and exposes the
reference judge, `engine.judge` (with `select_baseline` and `baseline_eligible`). `spyre-bench run` and `verdict` do not
call it: the verdict `spyre-bench verdict` prints is read from the store's view `v_benchmark_metric_verdicts`, which
applies its own per-metric policy (tier and floor percentage keyed by the stored metric names), a baseline of the
previous 20 runs of the series, at least 10 of them (else `no_baseline`), a floor of the larger of the policy floor and
twice the scaled MAD, and `suspected` for a first run over the floor (`regressed` needs the previous run over it too).
So `severity` and `tolerance` of an Expectation do not change a stored verdict yet. Write Expectations as the reviewed
statement of intent; they take effect when the view is driven by them.

**The reference rules (`engine.judge`)**, with a worked case: the current value is the median across the run's repeats;
the baseline is the median of the prior runs (or the fixed value). The noise term is `sigmas` (default 2) times
`1.4826 * MAD / |baseline median|` in percent; it applies to a rolling baseline even when the Expectation has no `noise`
block, and is 0 for a fixed baseline unless `noise.source` is `series_cv` or `qualification`. The floor is the largest of
`tolerance.pct`, `noise.floor_pct` and that term. With a baseline of 100 ms, tolerance 5% and a noise term of 2.4%, the
floor is 5%. A current value of 107 ms is 7% worse: `regressed`. 104 ms is `pass`; 92 ms is `improved`. The statuses are
`pass`, `regressed`, `improved`, `suspected` and `no_baseline`. Two things always say `suspected` rather than
`regressed`: a result whose validity is below the purpose's floor (or whose pinning could not be verified), and a
verdict resting on a single repeat. A suspicion is a prompt to look, not a finding.

Validation catches an Expectation that cannot work:

```yaml validated base=docs-analysis as=expectations-bad.yaml expect="error:selector selects no case"
apiVersion: bench/v1alpha1
kind: Expectation
metadata: {name: nothing}
spec:
  selector: {cases: [no_such_case]}
  metric: kernel.time.mean
  baseline: {kind: fixed, value: 0.005}
  tolerance: {pct: 10}
```

## Analysis: explaining a change

An Analysis reads stored Results. It holds some keys constant, varies one, and reports the effect:

```yaml validated file=docs-analysis/analysis/pinning-effect.yaml
# yaml-language-server: $schema=../../../spyre_bench/core/schemas/resources.schema.json
apiVersion: bench/v1alpha1
kind: Analysis
metadata: {name: pinning-effect, owner: example-team}
spec:
  select: {benchmarks: [op-scale]}
  metric: kernel.time.mean
  hold: ["case.*", "system.*", "subject.id"]
  vary: [run.exec.cpu_pinning]
  derive:
    name: kernel.time.per_element
    op: slope
    cases: {first: op_scale_n4096, second: op_scale_n1024}
    x: {first: 4096, second: 1024}
  compare:
    by: run.exec.cpu_pinning
    across: [env.label.node.class]
    metric: kernel.time.per_element
    baseline: none
  output: {noise: series_floor}
```

- **`select`** names the Benchmarks (or, with `via: relations`, the ones related to them).
- **`hold`, `relax`, `vary`** use attribute globs. Unlisted keys are *held*, and an attribute the registry does not
  know is never held; `relax` frees a key that `hold` would otherwise pin (for example `env.id`, to sweep a fingerprint
  attribute such as the host CPU clock).
- **`derive`** computes a metric per repeat pair from two cases of one subject and environment: `difference`
  (`first - second`), `ratio` (`first / second`) or `slope` (`(first - second) / (x.first - x.second)`). Its `name` must
  not shadow a registered metric. The report gives the median, mean and CV over the pairs plus the same operation on
  the two aggregates. Pairs with a zero denominator or a non-finite sample are skipped and counted.
- **`compare`** tabulates a metric by an attribute (`by`) within each `across` combination: n, mean, median, CV, the
  effect of each `by` value against the baseline (`baseline: none` means the `none` value of `by` when present; else name
  a `by` value) and whether the effect is `within_noise` (`|effect| <= max(CV of the cell, CV of the baseline cell)`;
  `null` below two repeats). `reading` says `better` or `worse` from the metric's direction.
- **`decompose`** attributes a parent metric to child cases (`from`, `to`, `show: residual`).
- **`output`** chooses the report shape (`controlled_sets`, `rank_variables`, `noise: series_floor | none`, `chart`).

Run it on the synthetic Results:

```bash validated
spyre-bench validate examples/docs-analysis
spyre-bench analyze pinning-effect --catalog examples/docs-analysis --results examples/results/docs-analysis.results.json --json
```

The `compare` section of that report says pinning to the device-local CPUs costs 10% less per element, outside the
noise, and reads `better`:

```text
cell: env.label.node.class=metal-a, run.exec.cpu_pinning=device_local   effect_pct -10.0   within_noise false   reading better
cell: env.label.node.class=metal-a, run.exec.cpu_pinning=none           (baseline)
```

The engine refuses to compare what is not comparable, with the reason, instead of producing a number: Results of a
different subject, `env_id`, node or case in one cell, or the same case run twice in one cell, are errors. Add
`env.node` to `across` to keep nodes apart. The results file in the example is rebuilt by
`python tests/refresh_docs_results.py` and a test fails if it is stale.

Mistakes the validator catches in an Analysis:

```yaml validated base=docs-analysis as=analysis/bad.yaml expect="error:derive.cases: first and second must be different cases"
apiVersion: bench/v1alpha1
kind: Analysis
metadata: {name: bad-derive}
spec:
  select: {benchmarks: [op-scale]}
  metric: kernel.time.mean
  vary: [run.exec.cpu_pinning]
  derive:
    name: kernel.time.per_element
    op: slope
    cases: {first: op_scale_n1024, second: op_scale_n1024}
    x: {first: 1024, second: 1024}
```

```yaml validated base=docs-analysis as=analysis/bad.yaml expect="error:shadows a registered metric"
apiVersion: bench/v1alpha1
kind: Analysis
metadata: {name: bad-shadow}
spec:
  select: {benchmarks: [op-scale]}
  metric: kernel.time.mean
  vary: [run.exec.cpu_pinning]
  derive:
    name: kernel.time.mean
    op: difference
    cases: {first: op_scale_n4096, second: op_scale_n1024}
```

## Which of the three do I need?

- A number should never silently be trusted: **ValidityPolicy** (and the NodeClass and Environment behind it).
- A number should alert when it moves: **Expectation**, advisory first.
- A question about *why* or *how much*: **Analysis**. It can also be the place to prototype the number an Expectation
  will judge later, once you register the derived name as a metric ([guide 5](05-add-a-metric.md)).

Approved references (a stored, signed-off run that serves as a baseline) are created by the reference workflow outside
the definitions; an Expectation with a `fixed` baseline is the definitional equivalent when the number is known up front.
