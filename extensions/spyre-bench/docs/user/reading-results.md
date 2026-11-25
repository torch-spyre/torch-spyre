# Reading results and verdicts

A **verdict** is the platform's judgement of one metric of one benchmark in one run, against a baseline built from
earlier runs of the same series. Verdicts are advisory: they route attention, they do not gate merges.

## The five statuses

| Status | Meaning | What to do |
|---|---|---|
| `pass` | within the allowed band of the baseline (the store's own word is `unchanged`) | nothing |
| `improved` | better than the baseline by more than the band | check it is real (see invalid data below), then celebrate |
| `regressed` | worse than the baseline by more than the band, and the previous run of the series was too | investigate; start in [Checking a nightly run](check-nightly.md) |
| `suspected` | looks worse but the evidence is thin or the data is not fully trusted | look, but do not act alone on it; the **reason** says why |
| `no_baseline` | not enough history to judge | nothing to judge yet; the series needs more runs |

A benchmark's overall status is its worst metric status in the order regressed, suspected, improved, no_baseline, pass.

## Reason codes

Status words in the Status column: **Available** (code in this tree), **Pending** (code in an open change or a local
branch, not merged), **Planned** (design only).

What the verdict view (`spyre-bench verdict` reads it) emits:

| Reason | Meaning | Status |
|---|---|---|
| `single_run` | worse than the band, but the series' previous run was not; one bad run is not yet a regression | Available |
| `below_floor` | status `pass`, but the movement was detectable and stayed under the floor | Available |
| `suspect_data` | the run, benchmark or repeat is marked suspect ([Incidents](incidents-invalidation.md)); judged but never enters a baseline | Pending (verdict view revision) |
| `purpose_not_baseline` | the run's purpose (for example a pull-request check) does not build baselines; judged against the series, never joins it | Pending (verdict view revision) |

Added by the separate verdict reader and advisory check, and only applied to a view `regressed` result:

| Reason | Meaning | Status |
|---|---|---|
| `few_repeats` | fewer valid repeats than the reader's minimum (default 3) | Pending (reader) |
| `validity_<state>` | the run's validity is below `unqualified_env` | Pending (reader) |
| `suspect_repeats` | some repeats of the run are suspect | Pending (reader) |

`spyre-bench verdict` itself prints what the view says; it does not apply the reader's caps. Not every `suspected` carries
a reason beyond the first group.

## Trust and validity are different things

- **Trust** answers *who produced it*, and is derived by the server from the channel the data arrived through, never
  claimed by the uploader: `ci` (written by the automation), `lab` (a person with lab access), `external` (anyone else).
  Only `ci` data can set a baseline or become a reference. Trust is part of a series' identity, so `external` data never
  mixes into a `ci` baseline.
- **Validity** answers *was the measurement sound*, per run: `valid`, then lower states such as `unqualified_env`,
  `suspect`, `external`, `invalid`. A run earns `valid` by passing the gates of its validity policy (qualified
  environment, enough repeats, noise below a limit, optional cleanliness and placement). In the Pending reader a low validity caps a `regressed` verdict at `suspected`. Validity reasons
  also include `pinning_violated` and `pinning_unverified` (a run that required CPU pinning could not prove it,
  [Pinning](pinning.md)); they are validity reasons, not verdict reasons (**Available** in the engine).

A run can be trusted and invalid (CI ran, but the node was noisy), or valid and untrusted (a careful lab measurement).
Read both on the run page.

## How the baseline works

For each series (a benchmark on one environment label set) the store takes the previous **20** eligible runs and needs at
least **10** of them; fewer gives `no_baseline`. The baseline is the median; the allowed band is the larger of the
metric's policy floor and twice the baseline's own robust noise (1.4826 x MAD, as a percentage). A noisy benchmark
therefore tolerates more movement than a quiet one. Excluding invalidated runs, suspect runs and runs whose purpose does not build baselines from
baselines and output is **Pending** (verdict view revision); the older view does not do this.

`regressed` needs persistence: the run and its predecessor both exceeded the band. One slow run is `suspected`
(`single_run`). This is a persistence rule, not a significance test.

## Why a regression can go quiet

The window rolls. If a regression stays, after roughly 10 slow runs the median of the window is the slow value, the band
is centred on it, and the verdict returns to `pass`. Nothing was fixed; the baseline moved. Slow drift behaves the same
way: no single step exceeds the band.

This is the main limit of a **rolling** baseline.

## Anchored baselines

An **anchored** baseline compares against a fixed point instead of the last 20 runs: a chosen good run, a release, or an
approved [reference](references.md). A regression then stays visible until it is fixed or someone deliberately moves the
anchor.

**Status: Planned/Pending.** The rolling baseline above is what the store computes today. Anchors are designed as an
additional comparison, shown next to the rolling verdict; check with the maintainers of your deployment whether they are
enabled. Until then, compare a suspicious run with an older known-good run by hand ([Comparing](compare.md)).

## Reading the verdict from the command line

**Available** (needs read access to the results database):

```bash
spyre-bench verdict --run-dir results/bench --json [--fail-on regressed] [--quiet]
spyre-bench verdict --run-id <stored-run-uuid>
spyre-bench verdict --artifact <artifact-uuid>      # newest run of that artifact that has verdict rows
```

The command prints per benchmark and metric the status, tier, reason, baseline size and values. It exits 0 unless you
pass `--fail-on <status>` and one occurs (exit 1); exit 2 means bad flags or a store failure, never a silent pass.

Use the **stored** run ids (`stored_run_ids` in the run's `run.json`), not the engine's own id: the two differ for some
producers. `--run-dir` does this for you.

## When it fails

| Symptom | Likely cause | Action |
|---|---|---|
| `lists no stored run ids` | the run was not ingested | run with `--ingest` and a store, see [Running locally](run-locally.md) |
| every metric `no_baseline` | new benchmark, new environment label, or too little history | wait for 10 runs, or compare by hand |
| `regressed` on a PR run you expected to be `pass` | the PR run joined a series with a different pinning or environment label | check the environment on the run page |
| no verdict rows at all | the run was invalidated, or the view is not deployed | ask the maintainers |
