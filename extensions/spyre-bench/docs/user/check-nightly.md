# Checking a nightly run

The nightly run executes the scheduled selection on the latest build and ingests it as `purpose=nightly`. Nightly runs
are the series that baselines are built from.

## Where to look

**Status:** the command line is **Available**; the dashboard pages are **Pending** (the drill-down is built and reviewed
but not yet deployed everywhere; your deployment may show only the older run and trend pages).

1. **Overview.** One row per architecture and component with the latest run's verdict counts. Start with "needs a look".
2. **Suite.** One row per benchmark at the latest run; filter by status.
3. **Series.** The trend of one benchmark with its baseline band. Click a point to see why it got its verdict
   ("why this verdict" shows the window, the floor, the reason).
4. **Run.** The repeats against the baseline window, trust and validity, gates, and whether the run is invalidated.
5. **Environment.** What it ran on: fingerprint, labels, qualification, cleanliness, placement, CPU pinning.
6. **Steps and ranks, Kernels and spans.** For multi-card runs, see [Multi-card debugging](multi-card-trace.md).

## From the command line

Both commands need read access to the results database (`CLICKHOUSE_*` settings).

```bash
spyre-bench verdict --artifact <artifact-uuid>
python -m spyre_clickhouse_ingest bench-verdicts --artifact-id <uuid> --arch x86_64 --format markdown   # Pending
```

## A short triage routine

1. Is the run valid? A low validity or an invalidated run explains many odd numbers. Open the run page.
2. Is it `suspected` or `regressed`? `suspected` with `single_run` often means wait one more night.
3. Look at the series. A step change on one date points to a change in the subject (the build) or the environment (a
   different node class or driver); a gradual slope points to drift.
4. Compare the failing run with the last good run of the same series ([Comparing](compare.md)) and read the
   differences in subject and environment first.
5. If the data is bad (broken node, interrupted run), [invalidate](incidents-invalidation.md) it so it stops polluting
   baselines. If the regression is real, file an issue with the benchmark id, run ids and the comparison.

A regression that stays can fall silent after about ten runs because the rolling baseline absorbs it; see
[Reading results](reading-results.md#why-a-regression-can-go-quiet). Do not treat `pass` after a known regression as a
fix.

## When it fails

| Symptom | Fix |
|---|---|
| no nightly run for last night | the schedule or the build failed upstream; ask the CI maintainers |
| a run exists but no verdict rows | it was invalidated, or all metrics lack a baseline |
| many benchmarks `suspected` at once | check validity: a shared cause (one bad node) usually explains it |

## Who to ask

The CI maintainers for missing runs; the benchmark owner for a real regression.
