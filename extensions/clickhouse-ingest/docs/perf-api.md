# Perf Lab data API

[← Back to index](README.md)

The read API behind the Perf Lab drill-down: overview, suites, benchmark series, run detail, environment,
steps, spans, compare, references. Reference implementation: `spyre_clickhouse_ingest/perf_api.py` (stdlib +
`clickhouse_connect`, one function per endpoint, `route()` for dispatch). Synthetic data and a fixture CLI:
`spyre_clickhouse_ingest/perf_fixture.py`. Database: `spyre_v2` only; no v1 object is read.

## Conventions

| Topic | Rule |
|---|---|
| Base path | `/api/v2/perf`. All endpoints are `GET`, read-only, safe to retry. |
| Body | `{"data": ..., "meta": {as_of, cache: {max_age_s, etag}, ...}}`. A route that extends an existing dashboard route keeps that route's flat body and adds the `data` keys to it (see Reconciliation). |
| Time | ISO 8601 UTC. `from` / `to` window on ingest time `ts` (partition pruning); the time shown for a run is `props.run_utc`, else `ts`. `as_of` (overview, suites) evaluates the page at a past instant. Server time zone is assumed UTC. |
| Windows | Every endpoint has a default window and a maximum: overview and suites 7 d (max 30), series 90 d (max 400), effect 60 d (max 120). Longer is `400 window_too_large`, never a silent clip. |
| Pagination | `limit` (caps per endpoint) and an opaque `cursor` from `meta.next_cursor`. The suites cursor carries the evaluation instant (`to`) and the offset, so its pages do not shift when new runs land; the other listings carry the offset only. A cursor that does not decode to an object is `400 bad_cursor`. |
| Caching | `meta.cache.max_age_s` is the TTL the route should send (`Cache-Control: max-age`, stale-while-revalidate 5x); `etag` is a hash of `data`. Overview and suites 60 s (a request with `as_of` is a past instant and may be cached for a day, the route may raise it), series and compare 120 s, run detail 60 s (its invalidation overlay changes), environment, steps, spans, references 300 s. |
| Errors | `{"error": {code, message, details}}`: `400 bad_parameter|bad_window|window_too_large|bad_cursor|unknown_signal`, `404 unknown_run|unknown_benchmark|unknown_series|unknown_reference|no_matching_steps|no_run_for_subject|no_run_for_env|not_found`, `410 run_invalidated` (spans), `413 result_too_large` (steps), `422 not_a_measurement_series|derive_refused`. An empty result is `200` with `meta.no_data: true` (and `meta.reason` where known), never an error: the UI shows a "no data yet" state. |
| Measured rows | Comparison targets (`run.ref_only = 1`) and trace record types (`reference, step_trace, span, trace_phase, trace_step, trace_kernel, compile_pass`, the `benchmark_excluded_record_types` list) never enter overview, suites, series, run detail benchmarks or compare. Steps and spans read exactly those trace types; references read exactly the comparison targets. |

### Invalidated and suspect rows

`v_invalidated_runs` is the single read path (run and run_benchmark scopes, node/device/placement windows through
the placements, retracted uploads). Each response states what it did:

| State of a run | Overview / suites | Series | Run detail, environment | Steps / spans | Compare |
|---|---|---|---|---|---|
| `invalidated` (a run-scope event, a retracted upload, or every repeat) | overview and suites never see these runs (schema 85 drops their samples, so they have no verdict rows); `invalidated_runs_in_window` counts them from `benchmark_runs` by `props.arch`/`platform` and component. Repeat-by-repeat invalidation through a node window is not a run-scope event and retracted uploads (`benchmark_id` and `repeat` both null); a run invalidated repeat by repeat through a node window still counts there and is flagged in series and run detail. Overview reports `invalidated_runs_in_window`; the suites sparkline skips these runs. | point kept and flagged (`invalidation_state`, `verdict.excluded = true`); `include_invalid=false` drops it | shown, with the events (`reason_code`, actor, scope) | spans `410`; steps and spans skip invalidated repeats | the card carries `invalidation_state`; values are returned, from valid repeats only |
| `partial` (a benchmark or some repeats) | counted | flagged; the point's value, CV and `n_repeats` come from the valid repeats | listed per benchmark with `invalidated_repeats`; metrics come from the valid repeats | invalidated repeats skipped | flagged; values from valid repeats |
| `suspect` | counted | flagged | shown | shown | flagged |
| `restored` | as valid | as valid | as valid | as valid | as valid |

Verdicts (schema 85) read `v_invalidated_runs`: invalidated samples are dropped before aggregation, and
suspect samples and runs whose `run.purpose` is set and not a baseline purpose (`benchmark_baseline_purposes`:
nightly, weekly, merge) never enter a baseline; their own verdict is capped at `suspected` with reason
`suspect_data` or `purpose_not_baseline` (shown as `verdict.reason`). `verdict.baseline_includes_invalidated`
is kept only as a fallback for a database whose verdict views predate schema 85: it lists invalidated run ids
inside `baseline_run_ids`, and is empty when 85 is applied and the ids are not listed.

### Verdict source

Verdicts come from the stored `benchmark_metric_verdicts` (the newest `evaluated_at` per run x series x metric; `baseline_kind` is always `rolling` today), not from the live
`v_benchmark_metric_verdicts` (schema 85), which evaluates every run in the database per call. The table is filled by a
30-minute refresh, so a run younger than that has no verdict: series shows it with `verdict.status = pending`,
overview and suites do not list it yet. Per-benchmark `verdict` is `v_benchmark_gate`'s rollup
(`fail` = a blocking metric `regressed`; `warn` = a regressed or suspected blocking/advisory metric;
`no_baseline` = every metric lacks a baseline; else `pass`), recomputed over the stored rows.

## Reconciliation with the dashboard (`feat/perf-lab`, 2026-11-12)

| Layer | Existing route (dashboard `routes/v2_perf.py`, `v2_perflab.py`) | Verdict |
|---|---|---|
| L0 overview | none (`/perf/tag-matrix`, `/perf/families` summarise per tag) | **new** `GET /perf/overview` |
| L1 suites | `/perf/artifact` (workloads of one run with verdicts), `/perf/family`, `/perf/tag-matrix` | **new** `GET /perf/suites`: latest verdict per benchmark series for an arch/component, no tag or artifact needed |
| L2 series | `/perf/points` (`benchmark_id, metric, higher_is_better, count, points[]` with `run_id, ts, arch, backend, component, artifact_id, value, artifact_name, tags, run_url, verdict {status, delta_pct}, highlight`) | **extend**: `GET /perf/benchmarks/{id}/series` returns those keys unchanged and adds `band`, noise, `subject_changes`, `env_changes`, invalidation. Route it as `/perf/points?baseline=1` if a second path is unwanted. |
| L3 run | `/perf/evidence` (one workload in one run: metrics with sample counts, per-metric verdicts, gate) and `/perf/diff` | **extend**: `GET /perf/runs/{id}` adds repeats, CV, validity, trust, invalidation and layer availability for all workloads of a run; `/perf/evidence` stays the single-workload drill |
| L4 environment | `/perflab/trace-runs/compare` `env` (the `env_*` props of trace bundles) | **new** `GET /perf/runs/{id}/environment` over schema 81/82 (fingerprint, node class, qualification, placement, pinning) |
| L5 steps | none (`trace_step` rows are per-phase step means, single card) | **new** `GET /perf/runs/{id}/steps` (per rank, per step) |
| L6 kernels/spans | `/perflab/trace-runs/{id}` (top kernels per phase), `/trace-runs/{id}/timeline` (single-card launches from the stored bundle) | **new** `GET /perf/runs/{id}/spans` for multi-card lane spans stored as values; the single-card kernel table stays on `/perflab` |
| X compare | `/perf/diff`, `/perf/vs-reference`, `/perf/build-diff`, `/perflab/compare`, `/perflab/trace-runs/compare` | **new** `GET /perf/compare` kinds `derive`, `effect`; kinds `runs` and `reference` add fairness flags (env, subject, pinning, clean) to what `/perf/diff` and `/perf/vs-reference` show |
| references | `/perf/references` is a picker of tags and recent runs | **collision**: the new comparison-target listing needs a different path (proposed `/perf/reference-targets`) or `?kind=targets` on the existing route |

## Endpoints

### `GET /perf/overview`  (L0)

Health tiles per (arch, component).

| Param | Default | Meaning |
|---|---|---|
| `arch`, `component` | all | filters |
| `days` | 7 (1..30) | the window the tile describes |
| `lookback_days` | 30 (days..90) | the period that defines "known" benchmarks for coverage |
| `stale_after_hours` | 36 | freshness threshold |
| `as_of` | now | evaluate at a past instant |

`data.tiles[]`: `arch, component, verdicts {fail, warn, pass, no_baseline, improved}` (benchmark series whose newest valid run in the
window has that gate; `improved` counts passing series with an improved metric), `benchmarks_in_window`,
`freshness {last_run_ts, age_hours, stale}`, `coverage {seen_in_window, known_in_lookback, ratio}`, `runs_in_window`,
`invalidated_runs_in_window`. A series is (component, arch, benchmark, backend, source_job). SQL: stored verdicts over
the lookback, rolled up as `v_benchmark_gate`, newest run per series, `run_id NOT IN` whole-run invalidations.

Anchored status (schema 88, see "Anchored baseline" below): each tile also carries `anchored {regressed, suspected, improved, unchanged,
no_anchor}`, series (benchmark x backend x source_job) in the window counted by the worst newest anchored verdict over their blocking and
advisory metrics; and `data.persistent_regressions[]` lists the metrics whose newest anchored verdict is `regressed` (at most 50, longest
first): `arch, component, benchmark_id, name, backend, source_job, metric, tier, worse_pct, since_ts, n_exceeding_runs, days_exceeding,
anchor_version`. SQL: `v_benchmark_anchored_verdicts` (a live view, not the stored history) bounded by the lookback and filtered by
arch/component. A tile counts series (benchmark x backend x source_job); the list has one entry per regressed metric, so a series with two regressed metrics counts once and is listed twice. A database without schema 88 returns zero counts, an empty list and `meta.notes`; a database with 88 but without one of its dependencies (86, 83) is an error, not a note.

### `GET /perf/suites`  (L1)

| Param | Default | Meaning |
|---|---|---|
| `arch`, `component` | all | filters |
| `days` | 7 (1..30), `as_of` | window |
| `group_by` | `suite` | `suite` (`run.suite.name`), `tier` (`run.tier`), `none`: a run prop as of the run, never part of the series key |
| `status` | all | `fail|warn|no_baseline|pass` |
| `q` | | name substring |
| `limit`, `cursor` | 100 (1..500) | pagination |

`data.items[]` (sorted fail, warn, no_baseline, pass, then name): `benchmark_id, name, component, arch, backend, source_job, run_id,
run_ts, verdict, improved, primary_metric, worst_delta_pct, regressed_metrics[], warn_metrics[], baseline_runs, runs_in_window,
record_type, tags[], group, purpose, history[{ts, verdict}]` (last 10 runs). `data.counts` counts all matches, not the page. `meta.total`.

### `GET /perf/benchmarks/{benchmark_id}/series`  (L2)

| Param | Default | Meaning |
|---|---|---|
| `metric` | the first of the series by policy tier, then run count | |
| `arch`, `backend`, `source_job` | the series with the most runs | `meta.series_options[]` lists the others; AB and multi-card jobs are separate series |
| `from`, `to` / `days` | 90 d (max 400) | |
| `limit` | 200 (1..1000) | newest points |
| `include_invalid` | true | `false` drops whole-run invalidated points |
| `reference_key` | | adds `data.reference {key, version, state, value}` for the metric |

`data`: `benchmark_id, metric, higher_is_better, count, points[]` (the `/perf/points` keys) plus `benchmark`, `series {arch, backend,
source_job, metric, tier, higher_is_better}`, `subject_changes[{at_run_id, ts, from, to, tag, id12}]`, `env_changes[...]`,
`summary {n_points, median_value, median_noise_cv_pct, last_status}`. Point: `run_id, ts, value` (median of all samples), `n_samples,
n_repeats, repeat_medians[], noise_cv_pct, min, max, subject_id, subject {artifact_id, name, id12, build, tag, run_url}, subject_changed,
env_id, env_changed, purpose, validity, invalidation_state, invalidations[]` and `verdict {status, reason, tier, baseline_value, delta_pct,
worse_pct, floor_pct, n_baseline, score, band {low, high, center}, baseline_includes_invalidated[], policy_version}`. The band is
`baseline_value * (1 +/- floor_pct/100)`; the worse side is the regression threshold. Errors: `404 unknown_benchmark|unknown_series`,
`422 not_a_measurement_series` for trace record types.

Anchored baseline (schema 88): `data.anchor` is `{state: "none"}` when the series has no current anchor (and `unavailable`, with `meta.notes`, on
a database without schema 88); otherwise `{state: "anchored", artifact_id, ref_key, version, value, n_samples, from, accepted_by,
accepted_at, from_run_id, source_job, floor_pct, band {low, high, center}, latest, persistent}`. `latest` is the newest point's anchored
verdict and `persistent` is true when it is `regressed`. Every point has `anchored`, null when there is no anchor, else `{status, reason,
anchor_value, delta_pct, worse_pct, floor_pct, since_ts, n_exceeding_runs, days_exceeding, version}`: `status` is `regressed`,
`suspected`, `improved`, `unchanged` or `no_anchor` (before the anchor's time); `reason` is `anchor_regressed`, `anchor_improved`,
`anchor_unchanged`, `no_anchor`, or the rolling view's `single_run`, `suspect_data`, `purpose_not_baseline`. `since_ts` and
`n_exceeding_runs` describe the unbroken streak of eligible runs over the anchored floor while the point is over it. `summary` gains
`last_anchored_status`; both describe the newest valid point and are null (not the run before) when that point has no anchored row yet. The rolling `verdict` is unchanged and still answers "got worse recently"; `anchored` answers "worse than the
accepted normal, and since when".

### `GET /perf/runs/{run_id}`  (L3)

Params `limit` (50, max 200), `cursor`, `benchmark_id`. `data.run`: `run_id, arch, ts, components, props` (purpose, trust, tier, invoker,
trigger, validity, suite, profile level), `clean` (`run.clean.*`), `env_id, source_job, subject, invalidation_state, invalidations[],
validity, validity_reasons[], validity_reasons_source` (`stored` when the run carries `run.validity_reasons`, else `derived` from
few_repeats, cv_above_limit, shared_env, pinning_unverified, pinning_violated, external, invalidated), `trust`. `data.benchmarks[]`:
`benchmark_id, name, backend, record_type, repeats[], invalidation_state, invalidated_repeats[], verdicts {metric: {...}}` and
`metrics {metric: {value, median_of_repeats, min, max, n_repeats, n_samples, cv_pct, repeat_cv_pct, repeats[{repeat, median, n,
samples[]}]}}`. `data.availability`: `benchmarks, environment, placements, steps, spans, step_rows, span_rows, verdicts`: the UI enables a
layer's tab from these flags and shows "no data yet" otherwise. `404 unknown_run`.

### `GET /perf/runs/{run_id}/environment`  (L4)

`data`: `env_id, env_ids, fingerprint {version, values{}, members[]}, labels, pinning_label` (missing label = `none`), `node_class,
node_class_spec {host_profile, qualification}`, `request` (the registered Environment), `nodes {node: {node_class, env_id_at_run,
env_history[{env_id, node_class, valid_from, valid_to}]}}`, `qualification {entries[{node, env_id, outcome, cv_pct, qualified_at,
expires_at, in_force_at_run, current_now}], in_force_at_run}`, `clean {run, per_repeat}`, `placements[{repeat, started_at, env_id,
placement_id, nodes[], devices[{node, serial, pci, firmware, link}], links[], clean{}, pinning{policy, cpu_set, cpu_set_source, verified,
violations}}]`, `pinning[]` (per repeat from `run.exec.*`, available without placements), `availability {fingerprint, placements,
qualification, request, node_history}`. `in_force_at_run` is evaluated at the run time, not now.

### `GET /perf/runs/{run_id}/steps`  (L5)

Params `benchmark_id`, `repeat`, `prompt` (default: the first matching record set; `data.selection.available` lists the others), `signal`
(default the declared `role.total`), `outlier_z` (3.5), `max_rows` (100000). `data`: `roles {total, wait, compute, busy}`, `signals[]`,
`ranks[]`, `steps[]` (`ttft`, `itl_0001`...), `cold[]`, `values[rank][step]`, `cross_rank[{step, value, slowest_rank, fastest_rank,
cold}]` by the signal's declared rule (`max|min|mean|median|sum`; undeclared = mean, `rule_declared: false`), `lag[rank][step]`
(total minus the fastest rank on that step), `wait[rank][step]` and `awaited_rank[step]` (least wait) when `role.wait` exists,
`headline {ttft, itl}` each `{value, n_warm_steps, rule, rule_declared, rank_medians{}, rank_skew, slowest_rank}` (median over warm steps
of the per-step cross-rank value; null when the only step of that kind is cold), `outliers[{rank, step, value, median, z, excess,
excess_wait, excess_compute, excess_host}]` (modified z = 0.6745 (x - median) / MAD over warm steps per rank and kind),
`consistency[]` (the producer's `summary.*` against the mean of the steps), `rank_coverage {present, world_size, complete}`, `reporter`,
`trace_uri`. Repeats that are invalidated are skipped. `413` when the run holds more step rows than `max_rows`.

### `GET /perf/runs/{run_id}/spans`  (L6)

Params `view` = `summary` (default) | `kernels` | `timeline`; filters `rank, step, repeat, prompt, lane, layer, min_dur_ns`; `limit` (2000,
max 20000) / `offset` for kernels and timeline; `top` (50) for the summary. `timeline` requires `rank` and `step`. `data.available
{spans, ranks, steps, lanes}`; `lanes[{rank, lane, n, total_ns, mean_ns, max_ns, p95_ns}]`; `kernels[{lane, name, n, total_ns, share,
mean_ns, max_ns, p95_ns, slowest_rank, ranks}]`; summary adds `rank_summary {per_rank_ns, slowest_device_rank, longest_collective_rank,
shortest_collective_rank}` (a rank that computes longest and waits least is the one the others wait for) and `collective_by_layer[{rank,
layer, total_ns}]`; timeline returns `spans[{rank, repeat, step, seq, parent_seq, depth, layer, lane, name, start_ns, dur_ns}]` ordered
by `start_ns, seq`. Start times are relative to the card's step start, so they are comparable within a card, not across cards.
`410 run_invalidated`.

### `GET /perf/compare`  (X)

`kind` selects the comparison. Common: `limit`.

| `kind` | Params | `data` |
|---|---|---|
| `runs` (default) | each side `a_*` / `b_*`: `*_run`, or `*_subject` (+ `arch`), or `*_env` (+ `benchmark_id`, `from`, `to`); optional `benchmark_id`, `metric` | `a`, `b` run cards (subject, env_id, labels, pinning, purpose, clean, invalidation_state, `resolved`), `comparable {subject, env_id, arch, pinning, clean}`, `fair` (env, arch and pinning equal), `rows[{benchmark_id, name, backend, metric, a, b, delta_pct, worse_pct, noise_pct, within_noise, higher_is_better}]` |
| `derive` | `derive=decode_per_token`, `run_ids` (1..50), `n_benchmark`, `one_benchmark`, `n_tokens`, `metric` (`latency_ms`) | `results[{run_id, value, latency_n, latency_1, per_repeat[], cv_pct, env_id, subject_id, pinning}]` with `value = (latency@N - latency@1) / (N - 1)`, `refused[{run_id, reason}]` (`missing_case`, `missing_metric`, `env_id_differs`); `422 derive_refused` when no run qualifies |
| `effect` | `benchmark_id`, `metric`, `arch`, `from`/`to`, `label` (`cpu_pinning`), `group_by` (`node_class`), `subject_id`, `purpose`, `source_job`, `noise_k` (2) | `groups {node_class: {policy: {runs, median, cv_pct, subjects}}}`, `effects[{node_class, cpu_pinning, baseline_median, median, effect_pct, better_pct, noise_pct, within_noise, n_baseline, n, mixed_subjects}]`; a run with no label counts as `none`; `within_noise` = abs(effect) <= `noise_k` x the larger CV of the two groups' run medians |
| `reference` | `run_id`, `key`, `version`, `metric` | `reference` head, `run` card, `rows[{benchmark_id, name, metric, reference, reference_backend, ours, delta_pct}]`. A reference is a target, never a baseline. |

### `GET /perf/references`  (see the path collision above)

`state` (`current` default | `retired` | `superseded` | `untagged` | `all`), `ref_class` (`perf|quality`), `key`, `version`, `limit`, `cursor`. List item:
`artifact_id, key, version, class, origin, label, arch, as_of, approved_by, approved_at, submitted_by, state, run_id, comparability{},
comparability_verdict` (`comparable` when every `ref.comparability.*` is `same`, `partial` when any is `different`, `not_comparable` when
any is `incompatible`). With `key`: `{reference, values[{benchmark_id, name, backend, metric, value, n, terms}]}`. Only published
versions are listed (`v_bench_references`, schema 86); a published version whose key has no tag row yet is `untagged`.

### Anchored baseline

The rolling baseline (previous 20 eligible runs) absorbs a persistent shift: after about 20 runs at the new level the rolling verdict is
`unchanged` again and the regression goes quiet. The anchored verdict (`v_benchmark_anchored_verdicts`, schema 88) compares the same
runs with a fixed accepted baseline, a current reference version marked `ref.anchor = 1` (an approved reference, or a run accepted as the
new normal), with the same noise rule (floor = max(policy floor, 2 x 1.4826 x MAD% of the anchor's samples), `regressed` needs two
consecutive eligible runs over, suspect or off-purpose runs capped at `suspected`). Only runs after the anchor's time are compared.
These endpoints read the live view, not the stored history, so they are bounded by series and window; unlike the rolling verdict there is
no stored history of anchored verdicts, and a history rewritten by accepting a new anchor is the intended behaviour.

## Bounds, per endpoint

| Endpoint | Tables read | Bound |
|---|---|---|
| overview, suites | `benchmark_metric_verdicts` (partition `toYYYYMM(ts)`), `v_invalidated_runs`, `benchmarks` | verdict `ts` window (max 30 d + lookback 90 d), `LIMIT` 500 / 20000 |
| series | verdicts, `benchmark_runs` (component, benchmark, backend, `ts`), `v_artifact_results_published`, `v_artifacts`, `artifact_tags` | window, `component = c` primary-key prefix, run id lists, `LIMIT` |
| overview (anchored), series (anchor line) | `v_benchmark_anchored_verdicts`, `v_benchmark_anchor` (schema 88: live; the series read is filtered by component, benchmark, arch, backend, metric, run time) | lookback window, `LIMIT` 20000 |
| run, environment | `benchmark_runs WHERE run_id = x` (no component known, generic exclusion search), placements, schema 81 views | one run |
| steps, spans | schema 87 views (`v_bench_step_records`, `v_bench_step_headline`, `v_bench_step_outliers`, `v_bench_spans`), or `v_bench_spans_fast` (schema 90) when it holds every span of the run; invalidated repeats are dropped by the views | one run, `max_rows` / `limit` |
| compare | as run, plus verdicts window for `effect` | run ids / window |
| references | `v_bench_references`, then `benchmark_runs WHERE run_id = <reference verdict run>` | reference list is small; values one run |

No endpoint reads `run_properties` or any `spyre.*` object. `benchmark_runs` has no skip index on `run_id` and its primary key leads
with `component`; per-run reads are cheap only because components are few. If run-keyed reads get hot, add a `run_id` bloom filter
or projection (a schema change, not made here).

## Known limits of the reference implementation

- Measurement rows are not de-duplicated by `(ts, audit_timestamp, audit_uuid)` in series and effect (steps and spans are): a re-ingested leg would double the samples. The effect table also pools every backend of a run.
- `status` and `q` filters on suites apply after a `LIMIT 20000` on series; past that many series the total is understated.
- Compare picks the first backend when a benchmark has several (references, effect).
- ClickHouse errors on out-of-range parameters (a rank above 65535) surface as a 500 from the route.
- Overview, suites and the sparkline hide only run-scope invalidations. Stored verdict rows written before a benchmark- or repeat-scope invalidation are not retracted, so those runs keep their old verdict there (series and run detail show the invalidation).
- Props for a compare card and the environment come from any one row of the run that carries them; a run mixing differing `env.id` values per row is not detected.
