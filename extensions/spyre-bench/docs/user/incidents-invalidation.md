# Incidents and invalidation

Bad data (a broken node, an interrupted run, a misconfigured driver) damages baselines and hides regressions.
**Invalidation** marks data as untrustworthy without deleting it.

**Status:** the storage and the exclusion in verdicts and baselines are **Pending** (implemented, not merged). The
command line and the dashboard action are **Planned**.

## States

| State | Effect |
|---|---|
| `suspect` | the data is judged but never enters a baseline; its verdict is capped at `suspected` (`suspect_data`; **Pending**) |
| `invalidated` | the data is removed from verdicts and baselines; a wholly invalidated run has no verdict rows, the next run is judged against a clean window |
| `restored` | the invalidation is withdrawn; the data counts again |

## Scopes

Run, benchmark within a run, a single repeat, and (for infrastructure incidents) a node, a device or a placement.
Invalidating a node invalidates the runs that touched it in the chosen window.

## Invalidate

1. Identify the cause and the window: which runs, which nodes.
2. Use the dashboard's invalidation page (or the command line when available). A window that covers more than 50 runs
   requires a **dry-run preview** (the runs and series affected) and a confirmation.
3. Give a reason; every action is logged.
4. Check the series afterwards: the invalidated points are greyed, and the next verdicts are judged without them.

Only people with the invalidation right can do this. Restoring is a separate, narrower right.

## Invalidate or delete?

Never delete. Invalidation keeps the evidence and the history of the decision; a restore is always possible.

## When it fails

| Symptom | Fix |
|---|---|
| the action is not offered | you lack the right; ask a maintainer |
| a series still looks wrong | the bad runs were not all in the window; widen it (with a preview) |
| a verdict changed unexpectedly | an invalidation removed a point from the baseline; check the series page's list of invalidated runs |

## Who to ask

The platform maintainers; the CI maintainers when the cause is a node or a pipeline.
