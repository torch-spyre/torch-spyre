# Bench bundles: results with repeats, and the records around them

Five bundle kinds carry the output of `spyre-bench` and the records its results depend on. They use the same
sealing, `files[]` checks, `--validate-only`, `--dry-run`, exit codes (0 ingested or duplicate, 1 failed and
retryable, 2 rejected, 3 incomplete) and receipt as the other kinds ([README: Offline results bundles](../README.md)).

| Kind | Holds | Writes | Who may ingest it |
|---|---|---|---|
| `bench` | one Result per case, each with its repeats, placement and environment | `benchmarks`, `benchmark_runs` (one row per repeat), `environment_fingerprints`, `run_placements`, the run's verdict | any channel; the channel decides the trust recorded |
| `bench-definitions` | `Environment` and `NodeClass` records | `environments`, `node_classes` | a CI-trusted channel only |
| `bench-qualification` | one qualification outcome per node and environment | `environment_qualifications` | a CI-trusted channel only |
| `bench-invalidation` | invalidation events (suspect, invalidated, restored) | `result_invalidations` | a channel of trust `lab` or `ci` |
| `bench-placement` | after-the-run updates of a run's placement and clean decision | new `run_placements` rows | a CI-trusted channel only |

The last four name no artifact, `test_type` or `arch`, carry no trust label, and cannot be put in the inbox
(`--upload` refuses them): a CI job ingests them directly. A definitions record of another kind (Benchmark,
Expectation, ValidityPolicy) is refused until its table exists.

## Sealing a `bench` bundle

```bash
python -m spyre_clickhouse_ingest results --offline --bench-dir RESULTS_DIR --arch x86_64 --out run.tgz
```

`--bench-dir` takes an engine results directory (every `result.json` under it), a folder of Results, or one
Result file. The sealer:

- copies each Result into `results/`, dropping channel-owned keys the engine writes (`run.trust` and the other
  reserved keys below) and saying so on stderr;
- takes the subject `artifact_id` from the Results (they must agree), `--trigger-type` as the tier (`perf` by
  default, or `quality`) and `arch` from `--arch`;
- sets `run_key` to `manual:spyre-bench:<engine run id>` when every Result carries the same run id, so
  sealing the same run again, on another host or by another route, is the same run.

Other seal flags apply as for every kind: `--seal-trust lab|external|none` (a claim that can only lower the
channel's trust; `none` writes no label and is for a CI job), `--redact-host`, `--redact-user`,
`--triggered-at`/`--ended-at` (ISO-8601 with an explicit zone; a zone-less time is read as UTC, whatever the
sealing machine's zone), `--component`, `--run-url`.

Records for the other kinds are sealed with `--kind bench-definitions|bench-qualification|bench-invalidation|bench-placement
--records-file FILE`. Layouts:

```text
bench-definitions     {"source": {"repo","ref","git_sha"}, "records": [{"kind": "Environment|NodeClass", "name", "spec_digest", "spec": {...}}]}
bench-qualification   {"records": [{"node","env_id","probe","outcome": "pass|fail","limit_pct","evidence_run_id","qualified_at","expires_at","checks": {...},"metric"?,"cv_pct"? (a claim)}]}
bench-invalidation    {"events": [{"invalidation_id","scope_kind","scope": {...},"window_from"?,"window_to"?,"state","reason_code","reason","evidence_url"?}]}
bench-placement       {"updates": [{"run_id","repeats"? (absent: every repeat),"clean"?: {...},"nodes"?,"devices"?: [{node,serial,pci,firmware,link}],"links"?}]}
```

## What a Result must hold

`apiVersion: bench/*`, `kind: Result`, and under `status`:

| Field | Rule |
|---|---|
| `case` | `component`, `name`, `tags`, `disc` (discriminators), `props`; the stored `benchmark_id` is recomputed here with the ingest rule, never trusted. A Result whose `case.id_mode` is `ingest` and whose `benchmark_id` differs is rejected; another engine id is kept as the identity prop `engine_id` |
| `subject.artifact_id` | the tested artifact; must equal the bundle's |
| `environment` | `env_id` (UUID), `fingerprint_version`, `fingerprint`, `labels` |
| `repeats[]` | `run.repeat` (a count, unique within the Result), `status`, `samples` (metric to finite numbers; required when `status` is `ok`), `placement` (`nodes`, `devices[]` of `node`, `serial`, `pci`, `firmware`, `link`; `links`), `clean`, `props`, optional `started_at` |
| `run_props` | string-valued; run-level claims such as `run.purpose` and `run.validity` |

Two Results for the same case are refused. The same case cannot appear twice in a run.

## Idempotency

- The run is keyed by `run_key`; a re-sealed bundle of the same run has a different `bundle_sha256` (it holds the
  sealing host, kit and time) but the same `content_sha256`, which covers the results only. `--validate-only`
  prints both for `bench`.
- Every repeat row carries `run.row_sha256`. A repeat that is already stored with identical content is skipped; one
  stored with different content is rejected (exit 2); the rest are written. So sending the bundle again is
  `duplicate`, sending it with more repeats adds only those, and a retry after a failed verdict write completes it.
- The verdict of a run that grows is superseded by a new one over the whole run; the views read the newest.

## Trust

The channel assigns trust (`--channel-trust ci|lab|external`; the relay derives it from the uploading account;
a CI job passes `ci`); the bundle's label can only lower it, and `ci` is never read from a bundle. Every row records
`run.trust`. Below `ci` the writer keeps the data and stores the claims it cannot back as `claim.*` props beside the
downgraded value:

| Claim | Stored for `lab` / `external` |
|---|---|
| `run.purpose` outside the channel's allowed set, or absent (`lab`: `on-demand`, `investigate`; `external`: `external`) | the allowed value; the claim in `claim.run.purpose` |
| `run.validity` | always `unknown`; a claim is kept in `claim.run.validity` |
| `clean.exclusive` `enforced` or `observed` | `unknown`; the claim in the placement's `clean['claim.exclusive']` |
| environment labels | replaced by `env.label.origin=<trust>` on the fingerprint row (first sight only) |
| placement source | `self_reported` (`probe` for `ci`) |

A bundle that carries a channel-owned key (`run.trust`, `run.actor`, `run.channel`, `run.upload_id`, `run.writer`,
`run.bundle_sha256`, `run.content_sha256`, `run.row_sha256`, `run.on_behalf_of`, `run.lib_ref`, `upload_id`, `round`,
or anything under `claim.`) is rejected, never silently overwritten.

## Connected runs (CI)

A CI job records a results directory without an inbox, as the channel it runs in:

```bash
python -m spyre_clickhouse_ingest results --bench-dir RESULTS_DIR --arch s390x --trigger-type perf \
  --workflow jenkins-<component>-bench-s390x --channel-trust ci --run-url "$RUN_URL"
```

The GitHub composite action `ingest-xml-to-clickhouse` takes the same as `bench-dir` and `channel-trust`; a
Jenkins job uses the `v2Results(benchDir: ...)` step. `--workflow` names the regression verdict series
(the first segment of each row's `source_file`) and is honoured only for `ci`; a lower channel files its
rows without one. An empty `--workflow` is the unprefixed series (what the vLLM rows use today); any other
value is the legacy `--workflow` string as it is (no `/`, no control characters, at most 128 characters). Without
`--channel-trust` the run is recorded as `external`.

The series and trust of a run are those of its first send: re-sending the same results with another workflow or
a higher trust is a `duplicate` and changes nothing. The action accepts `channel-trust: ci` only on `push`, `schedule` and
`workflow_dispatch` events, and `bench-dir` combined with another results input.

## Qualification and invalidation

- **Qualification.** The record names the node, `env_id`, the evidence run (already ingested as a `bench` run), a limit and, unless the run holds one metric, the canary `metric`. The
  writer recomputes the coefficient of variation and the repeat count from the evidence run's stored rows, requires
  that run to be CI-trusted and placed on the node, and refuses a `pass` the data does not support. A forged bundle
  cannot create a pass.
- **Placement update.** The clean decision a placer reaches after a run, or the devices a probe finds, is written as new
  `run_placements` rows over the run's newest ones (the effective view reads the newest per run and repeat), merging
  the `clean` keys and marking the source `ci_update`. Sending the same update again writes nothing; a run with no
  placement rows, or a repeat it lacks, is refused.
- **Invalidation.** An event is a scope (`run`, `run_benchmark`, `node`, `device`, `placement`, with an optional time
  window: `window_from` / `window_to` are event fields, not scope fields), a state (`suspect`, `invalidated`, `restored`), a reason code and a reason. It is ingested by
  `bench_invalidation.py` (the same writer the dashboard uses): the sealed file is `results/invalidation.json`; the
  actor is `--uploader` (the verified login), never the bundle; a channel of trust `external` and an inbox upload are
  refused. The state machine, `matched_runs` and the confirmation of an event that covers more runs than the
  threshold (`confirmed: true`, and the preview's `confirm_token` when it names one) are that module's. Re-sending the
  state an id already has is a no-op.
- **Retraction** of a `bench` upload is a `result_uploads` event (`format = bench`); the view `v_invalidated_runs`
  projects it, so a retracted run leaves the readers the same way an invalidated one does. It is not also written as
  an invalidation.

## Upgrade order

A kit that writes a newer `schema_version` or a field this ingest does not know is refused. Upgrade the connected
ingest first, then ship the new kit; `results --check <bundle>` says what is missing. `spyre-bench upload` needs the
whole ingest package on the sealing host (`offline.py` reads `bundle.schema.json` beside it); the vendored
`spyre_clickhouse_ingest.identity` module that only derives benchmark ids is not enough for it.

## Limits (read before relying on it)

- **`env_id` is a claim.** The receiving side does not recompute it from the fingerprint (the engine's `validate` does).
  A lower channel can name another environment's id; its runs then carry that `env.id` but keep their own `run.trust`,
  so a trust-filtered series, and the qualification check (which needs a CI-trusted evidence run measured in that
  environment), do not accept them. A fingerprint row first written by a lower channel is described again when CI sends
  the same `env_id`.
- **A lower channel cannot add repeats to a run recorded with higher trust** (rejected; send it under its own engine run
  id). Below CI the fingerprint version is stored as 1, so a claimed high version cannot outrank CI's description.
- **Failed repeats** (`status` other than `ok`, no samples) have no `benchmark_runs` row, so they are not compared on a
  re-send; the run's verdict records that a case had no ok repeat.
- **Placements are one row per repeat number.** Cases of one run that share a repeat number are merged
  (`placement.merged`), and a later bundle adding a case at an existing repeat does not change the stored placement; use
  a `bench-placement` update for that.
