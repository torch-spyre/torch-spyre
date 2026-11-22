# Multi-card and trace debugging

A multi-card benchmark is as slow as its slowest rank. This guide shows how to find which rank, which step and which
kernel, starting from a verdict.

**Status: Planned/Pending.** Per-rank step and span data are designed as values in the existing tables with views over
them; the dashboard pages are built against a synthetic fixture and show a "no data" box where a run has none. Only runs
that were produced with tracing enabled have this data.

## The path

1. **Series and run.** Start from the regressed metric ([Checking a nightly run](check-nightly.md)). Note the world size
   (number of cards) and whether `validity` is sound.
2. **Steps and ranks.** Open the run's steps page. The chart shows per-rank time by step; the heatmap shows rank by
   step. Look for a rank that is consistently slower (a slow card or a poor placement) or steps that spike on several
   ranks (a shared stall such as a collective).
3. **Slow rank summary.** The page marks a rank "slow" when its compute time exceeds about 110 % of the rank median.
   This is derived in the browser from stored values and is labelled as derived.
4. **Kernels and spans.** Pick a rank and step to see the timeline: device busy time, time waiting in collectives, and
   the kernels' share of the rank's time. The same step on every rank is listed so you can see who the others waited for.
5. **Environment.** Check placement and device NUMA attachment on the environment page; a slow rank often sits on a
   different NUMA node than its peers.

## Reading it

- Compute-bound slow rank: its kernels are longer than the same kernels on other ranks. Suspect the device, its
  placement or thermal state.
- Wait-bound ranks: they idle in collectives while one rank is late. The slow rank is the cause, not the waiting ranks.
- All ranks slow together: the change is probably in the subject (a kernel or the communication schedule).

## Collecting trace data

Enable the benchmark's profile level that records spans (a Benchmark's profile level is a run property, not identity).
Profile runs are never baseline-eligible. Keep trace runs few and targeted: they cost more than ordinary runs.

## When it fails

| Symptom | Fix |
|---|---|
| "No step data for this run" | the run was not traced; re-run with the span-recording profile level |
| "No span data for step N" | only some steps carry spans; pick one marked with a triangle |
| rank labels look wrong | the run's rank numbering is the launcher's; compare with the launcher log |

## Who to ask

The benchmark owner; the platform maintainers for missing trace data.
