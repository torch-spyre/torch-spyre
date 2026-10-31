-- RERUNNABLE
-- Adds run_case_counters.recovered (a passed case whose earlier attempt failed) and recounts every
-- run from test_case_runs, which also clears counters that drifted from their rows. As in 002, the
-- MV's creation time is the cutoff: older rows are recounted here, later ones by the MV, so no row
-- is counted by both or by neither.
ALTER TABLE run_case_counters ADD COLUMN IF NOT EXISTS recovered UInt64 AFTER xpass;

DROP VIEW IF EXISTS run_case_counters_mv;

TRUNCATE TABLE run_case_counters;

CREATE MATERIALIZED VIEW run_case_counters_mv TO run_case_counters AS
SELECT
    run_id,
    component,
    count()                        AS total_tests,
    countIf(status = 'passed')     AS passed,
    countIf(status = 'failed')     AS failed,
    countIf(status = 'error')      AS errors,
    countIf(status = 'skipped')    AS skipped,
    countIf(status = 'xfail')      AS xfail,
    countIf(status = 'xpass')      AS xpass,
    countIf(status = 'passed' AND (props['result.prior_status'] IN ('failed', 'error')
            OR toUInt32OrZero(props['result.reruns']) > 0)) AS recovered
FROM test_case_runs
GROUP BY run_id, component;

INSERT INTO run_case_counters
    (run_id, component, total_tests, passed, failed, errors, skipped, xfail, xpass, recovered)
SELECT
    run_id,
    component,
    count(),
    countIf(status = 'passed'),
    countIf(status = 'failed'),
    countIf(status = 'error'),
    countIf(status = 'skipped'),
    countIf(status = 'xfail'),
    countIf(status = 'xpass'),
    countIf(status = 'passed' AND (props['result.prior_status'] IN ('failed', 'error')
            OR toUInt32OrZero(props['result.reruns']) > 0))
FROM test_case_runs
WHERE audit_timestamp < (
    SELECT metadata_modification_time FROM system.tables
    WHERE database = currentDatabase() AND name = 'run_case_counters_mv'
)
GROUP BY run_id, component;
