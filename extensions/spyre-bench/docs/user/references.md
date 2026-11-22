# References: comparison targets that are not our own runs

A **reference** is a number to compare against that we did not measure in our regular series: a result from another
platform, a published figure, a published value, an external output for a quality check, or a one-off carefully measured
run we want to keep as the target. A reference is a comparison target, never a measurement of ours: it does not join
baselines or verdict series.

**Status: Pending.** The storage and the read views for references are implemented and reviewed but not merged; the
approval page and the submission commands are **Planned**. This guide describes the intended flow so you can prepare.

## What a reference carries

| Field | Meaning |
|---|---|
| key | which benchmark case and metric it is a target for |
| class | `perf` or `quality` (and for quality, a score or an output) |
| origin | `generated` (we produced it offline), `uploaded` (a person supplied a file), `published` (transcribed from a paper or page) |
| label, as-of date | what it is and when it was true |
| comparability | what must match for a comparison to be fair (precision, input length, device count, ...) |
| pinning policy | a reference is judged only against runs with the same pinning policy |
| state | `untagged`, `current`, `superseded`, `retired` |

Each version is immutable; "current" is a tag that moves when a new version is approved.

## Submit

1. Prepare the values and the provenance (source, as-of date, comparability notes). For a published number, record
   where it came from.
2. Submit through the same channels as other results ([Uploading](uploading.md)): the dashboard upload page, the
   command line, or an offline bundle of kind `bench-reference`.
3. Anyone who can sign in can submit; submitting does not make it current.
4. The submission appears as pending.

## Approve

A reviewer with the approval right for the component:

1. Opens the pending reference and reads its provenance and comparability notes.
2. If the benchmark case is not registered yet, uses "register this case now".
3. Approves (it becomes current and the previous version becomes superseded) or rejects with a reason.

If the reference was created from a run that was not on a clean machine, the page warns, and the reference records
that fact.

## Use

Where a reference is current for a benchmark, the series page draws it as a target line, the compare view shows a
"Reference target" card, and quality results show parity against it. The target never changes a verdict status; it
gives context and an [anchor](reading-results.md#anchored-baselines) candidate.

## When it fails

| Symptom | Fix |
|---|---|
| submission rejected as malformed | validate the bundle offline (`--validate-only`), fix and resubmit |
| the reference does not show on a series | it is not approved, or its comparability or pinning policy does not match the series |
| two versions both look current | the newest approved tag wins; the older shows as superseded |

## Who to ask

The approvers for the component (a named group); a maintainer if you have no approval right.
