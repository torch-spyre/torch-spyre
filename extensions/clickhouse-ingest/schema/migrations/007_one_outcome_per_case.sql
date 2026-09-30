-- RERUNNABLE
-- One outcome per (component, run_id, test_case_id): run_case_counters sums rows, so a case held
-- twice in a run was counted twice. The rules are TestResultWriter's. Within one source file a
-- repeat is a retry, so the later write wins: the later run_attempt, then the later audit_uuid
-- (UUIDv7, so insert order). Across files the rows are separate executions: an executed row beats
-- a reused copy, then the later attempt, then the worse status, so a test that ran beats a skip.
--
-- Safe to repeat (`apply_schema --rerun`): images that bake an older ingest keep writing
-- duplicates. Like 006, recount while no ingest is writing to a touched run.

DROP TABLE IF EXISTS case_outcome_dups;

CREATE TABLE case_outcome_dups (audit_uuid UUID, run_id UUID) ENGINE = MergeTree ORDER BY audit_uuid;

INSERT INTO case_outcome_dups
SELECT arrayJoin(arrayFilter(u -> u != keep, all_uuids)), run_id
FROM
(
    SELECT
        run_id,
        arrayFlatten(groupArray(uuids)) AS all_uuids,
        argMax(latest, (executed, top_attempt, latest_severity, latest)) AS keep
    FROM
    (
        -- A file's own winner first. A reused copy inherits its source row's file name, so
        -- copies group apart from the rows this run executed.
        SELECT
            component, run_id, test_case_id, executed,
            groupArray(audit_uuid) AS uuids,
            argMax(audit_uuid, (attempt, audit_uuid)) AS latest,
            max(attempt) AS top_attempt,
            argMax(severity, (attempt, audit_uuid)) AS latest_severity
        FROM
        (
            SELECT
                component, run_id, test_case_id, audit_uuid,
                props['source_file'] AS source_file,
                toUInt8(props['ran_in'] IN ('', toString(run_id))) AS executed,
                toUInt32OrZero(props['run_attempt']) AS attempt,
                transform(toString(status),
                          ['skipped', 'xfail', 'passed', 'xpass', 'error', 'failed'],
                          [1, 2, 3, 4, 5, 6], 0) AS severity
            FROM test_case_runs
        )
        GROUP BY component, run_id, test_case_id, source_file, executed
    )
    GROUP BY component, run_id, test_case_id
    HAVING length(all_uuids) > 1
);

DELETE FROM test_case_runs WHERE audit_uuid IN (SELECT audit_uuid FROM case_outcome_dups);

DELETE FROM run_case_counters WHERE run_id IN (SELECT DISTINCT run_id FROM case_outcome_dups);

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
WHERE run_id IN (SELECT DISTINCT run_id FROM case_outcome_dups)
GROUP BY run_id, component;

DROP TABLE case_outcome_dups;
