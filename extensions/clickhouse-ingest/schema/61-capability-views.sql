-- Views over capabilities / capability_runs. Two jobs: the enriched base join, and the
-- per-suite counters v1 stored on model_ops_suites, which v2 derives instead. Apply after
-- 46-capabilities.sql.
--
-- MEASURED, not asserted: of 298 v1 suites, 269 disagreed with their own variants on
-- spyre_enabled_count AND not_implemented_count (a typical row stored 24 enabled where 29
-- distinct operations actually XPASSed) -- cpu_fallback_count alone was right on all 298.
--
-- The v1 columns these replace: spyre_enabled_count -> status='passed' AND backend='spyre';
-- not_implemented_count -> status='not_implemented'; cpu_fallback_count -> status='passed' AND
-- backend='cpu'; spyre_failed_count -> status='failed' AND backend='spyre'. v1's four counters
-- were not disjoint (a FALLBACK is also a pass), which is why v2 splits status from backend.


-- Every capability verdict with its identity and run context resolved -- the base join for
-- everything below. ALWAYS filter by run_id or (component, test_type); unfiltered this joins
-- both tables whole.
CREATE VIEW IF NOT EXISTS v_capability_results AS
SELECT
    cr.ts            AS ts,
    cr.run_id        AS run_id,
    cr.capability_id AS capability_id,
    cr.component     AS component,
    cr.test_type     AS test_type,
    cr.arch          AS arch,
    c.subject        AS subject,
    c.name           AS name,
    c.tags           AS tags,
    cr.status        AS status,
    cr.backend       AS backend,
    cr.fail_reason   AS fail_reason,
    -- The identity's discriminator is hashed INTO capability_id, so read from the identity row.
    c.props          AS capability_props,
    cr.props         AS run_props,
    -- The shard that wrote this row, the dedup scope for a fan-out analysis.
    cr.props['shard'] AS shard
FROM capability_runs AS cr
INNER JOIN capabilities AS c ON c.capability_id = cr.capability_id;


-- The per-(run, subject) counters model_ops_suites stored -- one row per subject per run,
-- v1's suite grain (subject read as the model). countIf, not uniqExactIf, deliberately: these
-- count VERDICTS (one per capability, backend), not distinct capabilities -- see distinct_* below.
CREATE VIEW IF NOT EXISTS v_capability_run_counters AS
SELECT
    run_id,
    component,
    test_type,
    subject,
    any(arch)                                                        AS arch,
    min(ts)                                                          AS started_at,
    count()                                                          AS total_verdicts,
    countIf(status = 'passed' AND backend = 'spyre')                  AS spyre_enabled,
    countIf(status = 'not_implemented')                               AS not_implemented,
    countIf(status = 'passed' AND backend = 'cpu')                    AS cpu_fallback,
    countIf(status = 'failed' AND backend = 'spyre')                  AS spyre_failed,
    countIf(status = 'failed')                                        AS failed_any_backend,
    -- Distinct CAPABILITIES, not verdicts: one operation on three backends is one capability --
    -- the number "how many ops does this model exercise" wants.
    uniqExact(capability_id)                                         AS distinct_capabilities,
    uniqExactIf(capability_id, status = 'passed' AND backend = 'spyre') AS distinct_spyre_enabled,
    uniqExactIf(capability_id, status = 'not_implemented')            AS distinct_not_implemented,
    uniqExactIf(capability_id, status = 'passed' AND backend = 'cpu') AS distinct_cpu_fallback,
    -- Support rate over what was attempted on spyre: not_implemented is excluded from the
    -- denominator (not a failure to fix), and guarded against a zero denominator.
    if(countIf(backend = 'spyre') = 0, 0,
       round(100.0 * countIf(status = 'passed' AND backend = 'spyre')
             / countIf(backend = 'spyre'), 2))                       AS spyre_pass_rate
FROM v_capability_results
GROUP BY run_id, component, test_type, subject;


-- Which capabilities work on the CPU but not on Spyre -- unanswerable in v1, which stored one
-- status where a CPU fallback and a Spyre pass were the same value. backend is a VALUE, so this
-- is a self-join, not a flag.
CREATE VIEW IF NOT EXISTS v_capability_backend_gap AS
SELECT
    cpu.run_id        AS run_id,
    cpu.component     AS component,
    cpu.test_type     AS test_type,
    cpu.subject       AS subject,
    cpu.name          AS name,
    cpu.capability_id AS capability_id,
    spy.status        AS spyre_status,
    spy.fail_reason   AS spyre_fail_reason
FROM v_capability_results AS cpu
INNER JOIN v_capability_results AS spy
        ON  spy.run_id        = cpu.run_id
        AND spy.capability_id = cpu.capability_id
WHERE cpu.backend = 'cpu'   AND cpu.status = 'passed'
  AND spy.backend = 'spyre' AND spy.status != 'passed';


-- Per-capability history across runs: is an operation newly supported, or newly broken.
-- Grouped on the IDENTITY (v1's variant_id was a per-row surrogate -- 51,356 ids for 51,356
-- rows -- so no two runs of one operation ever reconciled).
CREATE VIEW IF NOT EXISTS v_capability_history AS
SELECT
    capability_id,
    component,
    test_type,
    subject,
    name,
    backend,
    count()                                    AS runs_observed,
    countIf(status = 'passed')                 AS runs_passed,
    countIf(status = 'not_implemented')         AS runs_not_implemented,
    countIf(status = 'failed')                 AS runs_failed,
    min(ts)                                    AS first_seen,
    max(ts)                                    AS last_seen,
    -- argMax over ts, not any(): a badge showing the latest verdict must not show a stale pass.
    argMax(status, ts)                         AS latest_status,
    argMax(fail_reason, ts)                    AS latest_fail_reason,
    argMax(run_id, ts)                         AS latest_run_id
FROM v_capability_results
GROUP BY capability_id, component, test_type, subject, name, backend;
