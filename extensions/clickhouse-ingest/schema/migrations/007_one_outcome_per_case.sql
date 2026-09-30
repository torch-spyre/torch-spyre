-- RERUNNABLE
-- One outcome per (component, run_id, test_case_id): run_case_counters sums rows, so a case held
-- twice in a run was counted twice. Keeps the row TestResultWriter.SEVERITY's order picks -- an
-- executed row over a reused copy, then the later run_attempt, then the worse status, then the
-- later write -- and deletes the rest.
--
-- Safe to repeat (`apply_schema --rerun`): images that bake an older ingest keep writing
-- duplicates. Like 006, recount while no ingest is writing to a touched run.

DROP TABLE IF EXISTS case_outcome_dups;

CREATE TABLE case_outcome_dups (audit_uuid UUID, run_id UUID) ENGINE = MergeTree ORDER BY audit_uuid;

INSERT INTO case_outcome_dups
SELECT arrayJoin(arrayFilter(u -> u != keep, uuids)), run_id
FROM
(
    SELECT
        run_id,
        groupArray(audit_uuid) AS uuids,
        argMax(audit_uuid, (executed, attempt, severity, audit_timestamp, audit_uuid)) AS keep
    FROM
    (
        SELECT
            component, run_id, test_case_id, audit_uuid, audit_timestamp,
            toUInt8(props['ran_in'] IN ('', toString(run_id))) AS executed,
            toUInt32OrZero(props['run_attempt']) AS attempt,
            transform(toString(status),
                      ['skipped', 'xfail', 'passed', 'xpass', 'error', 'failed'],
                      [1, 2, 3, 4, 5, 6], 0) AS severity
        FROM test_case_runs
    )
    GROUP BY component, run_id, test_case_id
    HAVING count() > 1
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
