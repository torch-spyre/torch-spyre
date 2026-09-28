-- run_case_counters_mv sees only inserts made after it was created; count the earlier rows.
-- The cutoff is the MV's own creation time, so no row is counted by both paths.
INSERT INTO run_case_counters
SELECT
    run_id,
    component,
    count(),
    countIf(status = 'passed'),
    countIf(status = 'failed'),
    countIf(status = 'error'),
    countIf(status = 'skipped'),
    countIf(status = 'xfail'),
    countIf(status = 'xpass')
FROM test_case_runs
WHERE ts < (
    SELECT metadata_modification_time FROM system.tables
    WHERE database = currentDatabase() AND name = 'run_case_counters_mv'
)
GROUP BY run_id, component;
