# FAQ

**Is a regression verdict blocking my PR?** No. Verdicts and the check are advisory.

**Why `suspected` instead of `regressed`?** The evidence is thin: one run (**Available**), or suspect data / a run purpose
that does not build baselines (**Pending**). See the [reason codes](reading-results.md#reason-codes).

**Why did a regression turn green by itself?** The rolling baseline moved. See
[Why a regression can go quiet](reading-results.md#why-a-regression-can-go-quiet).

**Why is my PR run never `regressed`?** **Pending** (verdict view revision): PR runs are judged against the nightly
series but cannot join it, so they are capped at `suspected`.

**Why `no_baseline`?** The series has fewer than 10 earlier eligible runs, usually because the benchmark, its inputs or
its environment label is new. Changing an input is a new benchmark.

**I renamed a benchmark and the history vanished.** Rename changes the id; list the old name in `previous_names`.

**Why can't I mix pinned and unpinned data?** They are separate series by design ([Pinning](pinning.md)).

**Does an external upload affect baselines?** No. Trust is part of the series identity.

**Can I delete a bad run?** Invalidate it instead ([Incidents](incidents-invalidation.md)).

**How many repeats do I need?** At least three to get a verdict at all; five or more for comparisons.

**Where do I find a benchmark's cost?** `est_hours` and `run.est` hints in the definition; they never affect identity.

**Can I run on a laptop?** Yes for validating, planning and running; keep the data out of shared databases
([Running locally](run-locally.md)).

**Where are numbers that do not belong to us?** [References](references.md); they are never baselines.
