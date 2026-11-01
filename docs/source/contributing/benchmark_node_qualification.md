# Performance benchmark node qualification

This page defines when an execution environment may produce
**regression-grade** performance observations
([#4950][issue-4950], parent [#4895][issue-4895]).

An observation from an unqualified environment must not enter a regression
baseline. The page states **what** qualifies. It does not state how machines
are provisioned, labelled, given devices, sized for disk, or failed over.
Physical inventory (which machine is primary or failover for a series) stays
in a private document. This page does not add a schema or a data migration;
storage for environment, card, and `num_cards` fields belongs to the
[Multi-AIU proposal][multi-aiu].

## How qualification is consumed

Worker qualification is one conjunct of observation eligibility. Comparison
identity and run identity stay separate layers.

```text
environment_qualified  AND  execution_valid  AND  metrics_valid
  AND  provenance_sufficient  AND  comparison_identity_ok
        ->  regression_eligible observation
```

| Layer | Owns | Where it lives |
| --- | --- | --- |
| Node qualification (this page) | Environment may produce regression-grade data | [#4950][issue-4950] |
| Observation eligibility | Row may enter a baseline or receive a verdict | [#4940][issue-4940], noise via [#5181][pr-5181] |
| Comparison identity | Two rows share one series | [#4939][issue-4939] |
| Run identity | Mandatory fields and refuse reasons | [run-identity contract][run-identity] |

`environment_qualified` is true only when every required gate below passes.
Each failure records a named `ENV_*` reason. Missing data is never a pass
and is never treated as zero. A qualified environment can still produce an
ineligible row (incomplete provenance, for example). An unqualified
environment's rows stay ineligible even when the suite exits 0.

Short-circuiting is allowed when the failing reason is recorded.

## Outcome

```text
qualify(environment) ->
  { status: qualified | unqualified,
    reasons: [CODE, ...],   # empty if and only if qualified
    checked_at: UTC timestamp,
    criteria_version: "4950-2" }
```

`qualified` means the observation may proceed to the other eligibility
gates. `unqualified` means it is excluded from the regression baseline
with the recorded `ENV_*` reasons.

## 1. Architecture and device identity

Required for every regression-grade run.

| Check | Pass | Fail |
| --- | --- | --- |
| Host CPU architecture | Recorded architecture is one of `amd64` / `x86_64`, `s390x`, `ppc64le`, and it matches the intended series | `ENV_ARCH_UNKNOWN`, `ENV_ARCH_MISMATCH` |
| Accelerator class | A Spyre / AIU class is recorded for a card-using job, and it matches the product under test | `ENV_DEVICE_UNKNOWN` |
| Device identity within the run | No mid-run card swap or bind change in the measurement window | `ENV_DEVICE_UNSTABLE` |

Public text names architecture and device **class** only.

(card-count-and-topology)=
## 2. Card count and topology

Required unless the job's first present intended source is an explicit 0.
A job is card-using when it **declares or requests** one or more Spyre
cards. Successful lock is not the test. A failed lock or a short lock
stays card-using: record `ENV_NO_CARD_LOCK` or `ENV_CARD_COUNT_MISMATCH`.
Do not reclassify that job as not applicable. When no intended source is
present, card usage cannot be ruled out, so this section fails closed.

A job is not card-using only when its first present intended source is an
explicit 0. That is the only case where this section does not apply. Do not
treat a missing intended count as zero. A failed lock, a short lock, a
measured count with no declaration, or no card-count evidence at all stays
on the fail path below and is never reclassified as not applicable.

### Intended count

Intended count is the count the job asked for, in this priority order.
Stop at the first present source. Do **not** read intended count from a
measured report field.

1. `jenkins_job.cards` / `CARDS`
2. GitHub Actions or Actions Runner Controller requested pool quantity
3. Multi-AIU **declared** expected count at launch

### Recorded count

Recorded count is the quantity actually obtained: the acquired lock
quantity, or an independently stamped measured count such as report
`num_cards`.

### Multi-AIU

Never use the same `num_cards` field as both intended and recorded. If the
only value present is the measured count, treat it as **recorded** and fail
`ENV_CARD_COUNT_MISSING` because intended is absent. Do not infer a
declaration from that measured count.

Evaluate in this order. Record every failing reason before moving on.
Short-circuit only after that reason is stored.

1. The first present intended source is 0. The job is not card-using.
   Stop. This section does not apply.
2. No intended source is present. Card usage cannot be ruled out. If a
   measured count is present, store it as recorded only and do not treat
   it as a declaration. Whether or not that measured count is present,
   fail `ENV_CARD_COUNT_MISSING` and stop. Do not skip this section.
3. The first present intended source is greater than or equal to 1. The
   job is card-using. Stay on this path if the lock fails or is short.
4. The acquired lock quantity is absent or zero. Fail `ENV_NO_CARD_LOCK`.
5. Choose the recorded count. If the acquired lock is greater than zero
   and no measured count is present, recorded is the lock. If the lock is
   absent or zero and a measured count is present, recorded is the measured
   count. If the lock is greater than zero and a measured count is also
   present, they must be equal and that shared value is recorded; if they
   differ, do not pick a winner, and fail `ENV_CARD_COUNT_MISMATCH`.
   Recorded must equal intended. Otherwise fail `ENV_CARD_COUNT_MISMATCH`
   (including a short lock).
6. Intended card count differs from the other members of the series.
   Fail `ENV_TOPOLOGY_MISMATCH`.

A job that stamps neither a declaration nor a measured count has not
declared zero. Record `ENV_CARD_COUNT_MISSING`. Non-card jobs must stamp
an explicit 0. This page does not infer a card request, or the absence of
one, from the suite name.

### Gates

`ENV_NO_CARD_LOCK` is defined only in this section.
[Resource isolation](resource-isolation) does not define a second lock gate.

| Check | Pass | Fail |
| --- | --- | --- |
| Intended count | First present source is greater than or equal to 1, then continue. An explicit 0 skips this section. No intended source fails, with or without a measured count | `ENV_CARD_COUNT_MISSING` |
| Acquired lock | Card-using job acquired a lock quantity greater than zero | `ENV_NO_CARD_LOCK` |
| Recorded vs intended | Recorded count from step 5 is present and equal to intended. Lock and measured count must agree when both are present. A short lock stays card-using | `ENV_CARD_COUNT_MISMATCH` |
| Series topology | Intended card count matches the other members of the series | `ENV_TOPOLOGY_MISMATCH` |

### `FLEX_DEVICE` is optional

Recorded Flex mode is **optional** system information on `report.xml`.
When the report includes it, keep the value. A missing Flex property does
**not** fail qualification. Do not emit a Flex fail code for absence.

Runtime selection of `PF`, `VF`, or `MOCK` via the `FLEX_DEVICE` environment
variable is a separate mechanism, documented in the
[runtime guide](../runtime/index.md). This gate does not change that
variable and does not require the property on the report.

## 3. Required software state

Required for every regression-grade run.

| Check | Pass | Fail |
| --- | --- | --- |
| OS / kernel family | A machine-readable OS identity is present on the run or on a linked environment record | `ENV_OS_UNKNOWN` |
| Driver / runtime stack | Flex, senlib, and deeptools versions that apply to the stack under test are present in `version_info` | `ENV_RUNTIME_INCOMPLETE` |
| Firmware / card software | When that architecture requires a firmware pin, the recorded state matches the pin, or the program has marked the run unpinned-allowed | `ENV_FIRMWARE_UNKNOWN`, `ENV_FIRMWARE_MISMATCH` |
| Image / artifact | Forward runs carry a published image reference or artifact id. Backfill runs carry resolved historical tip pins | `ENV_IMAGE_UNKNOWN` |

This page does not require inventing component commits that were never
recorded. A missing mandatory stack commit makes the observation ineligible
and can also fail this section when the environment cannot attest it.

(resource-isolation)=
## 4. Resource isolation

Required for every regression-grade run.

Exclusive card lock is not restated here. For a card-using job, lock
absence is `ENV_NO_CARD_LOCK` and a short allocation is
`ENV_CARD_COUNT_MISMATCH`, both defined in
[section 2](card-count-and-topology).

| Check | Pass | Fail |
| --- | --- | --- |
| No peer suite on the same cards | The card pool is not shared with another product-test or image-test at the same time | `ENV_CARD_CONTENTION` |
| Agent class | The environment is on the program's Spyre CI agent class for that architecture | `ENV_AGENT_CLASS_UNQUALIFIED` |
| Disk / scratch | The suite did not fail or degrade because the node ran out of space or evicted caches | `ENV_RESOURCE_EXHAUSTED` |

Shared single-architecture card hosts that build images and run product-test
must not run more than one regression-grade performance set at the same time
on the same cards. One image reference and one product-test at a time is the
interim isolation rule until a second qualified environment exists.

## 5. Environment provenance

Required for every regression-grade run.

| Check | Pass | Fail |
| --- | --- | --- |
| Environment id | A stable id (label set, inventory key, or CI node class id) is on the observation or on a joinable side record | `ENV_ID_MISSING` |
| Join | Dashboard or ClickHouse can attribute the run to that environment without guessing | `ENV_ID_UNJOINABLE` |

The [run-identity contract][run-identity] still marks environment id as not
available on the public API. Landing an `environment_id` (or equivalent) is
a producer and ingest follow-on under the [Multi-AIU proposal][multi-aiu],
not a schema added by this page. Until that field is stamped, a public
caller has no id to join. That result is `ENV_ID_MISSING`, not a pass. A
private inventory map may support ops triage. It does not satisfy this
public gate, and it must not be copied into this page.

## 6. Repeatability and noise

Required. Reuse [#5181][pr-5181]. Do not re-implement regression noise in
Python, and do not add a stricter node-only floor.

`environment_qualified` reads ClickHouse verdicts. It does not compute them.

Workload is the `perf` cell, which runs spyre-perf-suite. It is not a
separate canary suite. Read `v_perf_regression_gate.gate_status` for that
perf run. That view folds `v_benchmark_gate`, which reads
`v_benchmark_metric_verdicts` over `benchmark_runs`. Do not compute the
status in Python, and do not add a second floor.

| `v_perf_regression_gate.gate_status` | Result |
| --- | --- |
| row missing | `ENV_NOISE_UNPROVEN` |
| `no_baseline` | `ENV_NOISE_UNPROVEN` |
| `fail` | `ENV_NOISE_EXCESS` |
| `pass` or `warn` | This section passes |

`no_baseline` here is the folded status: every benchmark gate row is
`no_baseline`. One benchmark at `no_baseline` beside others at `pass`
leaves the folded gate at `pass`. `fail` is a blocking `regressed` metric.
`warn` covers other `regressed` or `suspected` signals and does not fail
this section. An informational metric can be `regressed` while the folded
gate stays `pass`; that is not `ENV_NOISE_EXCESS`. Treating `warn`, or a
single metric, as a node failure would be a stricter floor than #5181.

Re-check after a driver or firmware change by letting the next `perf` runs
flow through the same views. This page does not define a second cadence or
a second numeric floor.

## 7. Failover equivalence

Required only when a failover environment is used for the series.

| Check | Pass | Fail |
| --- | --- | --- |
| Declaration | Private inventory states which environment is primary for that architecture and series | `ENV_FAILOVER_UNDECLARED` |
| Equivalence | Failover matches the primary on architecture, intended card count, software-pin class, and #5181 noise class, **or** the deltas are listed and force a separate baseline | `ENV_FAILOVER_NOT_EQUIVALENT` |
| Tagging | Failover runs carry an environment id so series logic can split or merge on purpose | `ENV_FAILOVER_UNTAGGED` |

Recorded Flex mode may be compared when both sides have it. A missing Flex
value still does not, by itself, fail section 2 or this section.

An environment outside the program's Spyre CI agent class is not
failover-equivalent by default.

Which machines fill the primary and failover roles is private inventory.
Do not copy that inventory into this page.

## What this page does not define

- Jenkins label recipes, device bind steps, disk retention, or artifact layout.
- Detector math or dashboard layout. Noise math stays in [#5181][pr-5181].
- Historical commit resolution or backfill fan-out.
- A parallel environment schema, table, or migration. Those ride the
  [Multi-AIU proposal][multi-aiu].

## Producer gaps

Honest gaps today. Closing them is follow-on work. None of them is a schema
in this change, and none of them turns a missing value into a pass.

| Dimension | Today | Follow-on |
| --- | --- | --- |
| Architecture | Stamped on the run | Keep |
| Card count | Often not a first-class field on perf rows | Stamp a declared intended count, including an explicit 0 for a non-card job, and a separate recorded count. Absence is `ENV_CARD_COUNT_MISSING`, not a skip |
| Flex mode | Often absent on `report.xml` | Keep optional. Absence is not a qualification failure |
| Environment id | Not on the public run API | Multi-AIU / ingest track, coordinated with [run identity][run-identity] |
| Isolation | Card lock exists for product-test; contention is operational | Ops keeps one regression-grade set at a time on a shared card pool |
| Noise | [#5181][pr-5181] `v_perf_regression_gate` | Callers **read** that folded gate. No Python reimplementation. Missing row is `ENV_NOISE_UNPROVEN` |
| Failover | Informal sharing | Private primary / failover inventory only |

[issue-4950]: https://github.com/torch-spyre/torch-spyre/issues/4950
[issue-4895]: https://github.com/torch-spyre/torch-spyre/issues/4895
[issue-4940]: https://github.com/torch-spyre/torch-spyre/issues/4940
[issue-4939]: https://github.com/torch-spyre/torch-spyre/issues/4939
[pr-5181]: https://github.com/torch-spyre/torch-spyre/pull/5181
[run-identity]: https://github.ibm.com/ai-sw-acceleration/spyre-dashboard/blob/main/docs/run-identity-contract.md
[multi-aiu]: https://github.ibm.com/IBM-Waterford-Spyre/Multi-AIU-decode-trace-demo/tree/main/proposal
