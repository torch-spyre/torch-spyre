# Comparing runs, subjects and environments

Only like-for-like differences support a claim. Before reading any number, list what differs between the two things you
compare; the platform warns when the comparison is not like-for-like.

## What can differ

| Axis | Examples | Is it a fair comparison? |
|---|---|---|
| **Subject** | build A vs build B of the same image | yes, this is the usual regression question |
| **Environment** | node class 1 vs node class 2; device count | only as an environment study; hold the subject fixed |
| **Placement / pinning** | pinned vs unpinned CPUs | only as a pinning study ([Pinning](pinning.md)) |
| **Time** | last night vs three weeks ago | yes if subject and environment are the same; otherwise it mixes axes |
| **Purpose / trust** | nightly CI vs an external upload | not directly: trust is part of series identity |

Change one axis at a time. If two axes differ, the difference cannot be attributed.

## In the dashboard

**Status: Pending** (the compare views are built and reviewed, not deployed everywhere).

- **Compare** shows two runs side by side: results per metric, and a table of subject and environment fields that
  differ. A "Reference target" card appears when the benchmark has an approved [reference](references.md).
- **Pinned vs unpinned** shows the effect of pinning per node class against a noise band.
- Open compare from the Run page ("compare with...") so both sides keep their ids.

## From the command line

**Available.** `analyze` evaluates an `Analysis` document (stored in the catalog) over Results:

```bash
spyre-bench analyze <analysis-name> --catalog <catalog-dir> --results <results.json> [--json]
```

`--results` is one JSON array of Result documents. A run writes one `result.json` per case (when run with `--catalog`);
combine the ones you want: `jq -s . results/bench/cases/*/result.json > results.json`.

An Analysis can **derive** a metric from two cases of one subject and environment, per repeat pair (`difference`,
`ratio`, or `slope`), and **compare** a metric by one attribute within each combination of others. The output reports
n, mean, median, CV, the effect of each value against the baseline, and `within_noise`
(`|effect| <= max(CV of the cell, CV of the baseline cell)`). Pairs with a different subject, environment, label, node
or pinning are refused with the reason; a `within_noise: true` row is not evidence of a difference.

## Reading the answer

1. **Repeats.** Check `n` and CV first. With one or two repeats there is no noise estimate.
2. **Effect against noise.** Believe an effect only when it exceeds the noise band.
3. **Direction.** The metric registry says whether lower or higher is better; `reading` says `better` or `worse`.
4. **Validity.** Compare valid runs; an invalid or suspect run belongs in an incident, not a comparison.

## When it fails

| Symptom | Fix |
|---|---|
| "refused: different env_id" | the two sides ran on different environments; hold it fixed or choose an environment study |
| "no Result of [...]" | the results file does not contain those cases; check names with `spyre-bench explain` |
| `within_noise` everywhere | too few repeats or a noisy benchmark; collect more repeats |
| zero denominator pairs skipped | listed under `warnings`; inspect the raw samples |

## Who to ask

The benchmark owner for whether a metric is comparable; the platform maintainers for refused comparisons.
