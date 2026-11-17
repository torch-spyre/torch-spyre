-- One row per request at its latest state; the requester is whoever made its first event.
CREATE VIEW IF NOT EXISTS v_ci_requests AS
SELECT
    r.request_id                 AS request_id,
    argMax(r.plan_id, r.ts)      AS plan_id,
    argMax(r.state, r.ts)        AS state,
    argMin(r.actor, r.ts)        AS requested_by,
    argMin(r.on_behalf_of, r.ts) AS on_behalf_of,
    argMin(r.channel, r.ts)      AS channel,
    argMin(r.profile, r.ts)      AS profile,
    argMin(r.target_kind, r.ts)  AS target_kind,
    argMin(r.subjects, r.ts)     AS subjects,
    argMin(r.preset, r.ts)       AS preset,
    argMin(r.arches, r.ts)       AS arches,
    argMin(r.modes, r.ts)        AS modes,
    argMax(r.cells_requested, r.ts) AS cells_requested,
    argMax(r.cells_skipped, r.ts)   AS cells_skipped,
    argMax(r.cells_blocked, r.ts)   AS cells_blocked,
    argMax(r.estimate, r.ts)        AS estimate,
    argMax(r.canonical, r.ts)       AS canonical,
    argMaxIf(r.job_run, r.ts, r.job_run != '')     AS job_run,
    argMaxIf(r.build_url, r.ts, r.build_url != '') AS build_url,
    argMax(r.error, r.ts)        AS error,
    min(r.ts)                    AS requested_at,
    max(r.ts)                    AS updated_at
FROM ci_requests AS r  -- qualified: the aliases above would shadow the columns
GROUP BY r.request_id;

-- Passed-leg duration per component x test type x arch over the last 60 days, for the launch estimate.
-- Failed legs are left out (they stop early or hang) and so are zero-length rows (metadata-only records
-- that perf and model_* write). `n` is shown with every figure: ppc64le and s390x cells are thin.
CREATE VIEW IF NOT EXISTS v_ci_cell_durations AS
SELECT
    e.component                   AS component,
    e.test_type                   AS test_type,
    e.run_arch                    AS arch,
    count()                       AS n,
    quantile(0.5)(e.duration_s)   AS p50_s,
    quantile(0.9)(e.duration_s)   AS p90_s,
    max(e.ts)                     AS last_ts
FROM v_artifact_results_enriched AS e
WHERE e.state = 'passed'
  AND e.duration_s > 0
  AND e.ts >= now() - INTERVAL 60 DAY
GROUP BY e.component, e.test_type, e.run_arch;
