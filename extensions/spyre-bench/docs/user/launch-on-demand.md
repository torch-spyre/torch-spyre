# Launching a benchmark on demand for a pull request

Use this when you changed something and want to know whether it moves performance, without waiting for the nightly.

## What exists

| Route | Status |
|---|---|
| A manually dispatched workflow in the repository (`bench-on-demand`), which validates, plans, runs, ingests and posts verdicts for a chosen selection | **Pending** (ships with the benchmark producers) |
| A pull-request comment, for example `/spyre-test bench=sanity`, handled by the CI | **Planned** (grammar designed; ask whether your deployment has it) |
| A "Launch" action in the dashboard that previews the cost before starting | **Planned** |

All routes are advisory: nothing gates the merge, and the check they post is never required.

## Choosing what to run

A request names a **purpose** (`on-demand` for these), an **environment** (the hardware shape), a **selection**
(a named set of cases, a tier such as `sanity`, `core`, `extended`, or a suite) and a **reuse** rule:

| Reuse | Meaning |
|---|---|
| `never` | measure everything fresh (the on-demand workflow's default) |
| `if_valid` | use a stored result for the same subject and environment if it is valid, enough repeats and recent |
| `prefer` | like `if_valid` but run when absent |
| `only` | never measure; report what is stored |

Start with the smallest selection that covers your change (`sanity`). Larger tiers cost device-hours; the planner's
`--budget` refuses an over-budget plan.

## Run the workflow (when available)

From the repository's Actions page, choose `bench-on-demand` and fill in:

- `ref`: branch, tag or SHA to benchmark;
- `selection`: for example `ops-core`;
- `clickhouse_env`: `dev` (default). Leave it on `dev` unless a maintainer asks for production;
- `ingest`: write rows (default on);
- `smoke`: a card-free dry run on a hosted runner that measures and ingests nothing. Use it to check your selection
  resolves.

Expected: a run that finishes with a summary of cases run and a verdict table, and (when posting is enabled) a check
named like `Spyre Perf (x86_64)` on the commit, concluding `neutral` when something is regressed or suspected, `success`
otherwise.

## Check the result

1. Open the check, or run `spyre-bench verdict --artifact <artifact-uuid>`.
2. A PR run is judged against the nightly series but never joins it (`purpose_not_baseline`), so a PR run can read
   `suspected`; that is the cap, not a defect (**Pending**: the cap is in the unmerged verdict view revision). See [Reading results](reading-results.md).
3. For one repeat's detail, open the run in the dashboard ([drill-down](check-nightly.md)).

## Clean runs

Ordinary CI shares machines, so another job can disturb a measurement. A **clean run** reserves the machine's resources
for your run. Ask for it only when you need a reference-quality number; it can wait for the machine to drain.
**Status: Planned** (the placement mechanism is built and proven on one machine; the request flag and launcher are not
yet available). Without it, the run records cleanliness as *observed* and the validity policy decides how much to
trust it.

## When it fails

| Symptom | Fix |
|---|---|
| the workflow fails at validation | the selection or a definition is invalid; run `spyre-bench validate` locally |
| unknown suite or case | the message lists the available names; names are never silently dropped |
| plan exits 4 | every case was skipped (capacity or reference node); read the reasons |
| no check appears | posting needs a token with checks permission; ask the maintainers |
| the comment command is ignored | it is not enabled for your repository; use the workflow |

## Who to ask

The maintainers of the CI for your repository; the benchmark's owner for what a selection contains.
