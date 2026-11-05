-- RERUNNABLE
-- run_case_counters sums rows, so rows that restate another row of the same run are deleted.
-- TestResultWriter applies the same rules at insert time. A row is redundant when it is:
--   * an exact copy (same file, attempt, status, duration to the ms, and message);
--   * an older attempt of a file that has a newer attempt of the case;
--   * a reused copy, where the run executed the case itself;
--   * a skip, where another file of the run executed the case.
-- Differing outcomes otherwise stay, because one test_case_id can still be two different tests
-- (a name that differs only in case, hf-adapters' base and _adapter configs).
--
-- Runs after 007 (which re-keys case-sensitive names) and is safe to repeat
-- (`apply_schema --rerun`). Like 006, recount while no ingest is writing to a touched run.

DROP TABLE IF EXISTS case_outcome_dups;

CREATE TABLE case_outcome_dups (audit_uuid UUID, run_id UUID) ENGINE = MergeTree ORDER BY audit_uuid;

INSERT INTO case_outcome_dups
SELECT audit_uuid, run_id
FROM
(
    SELECT
        audit_uuid, run_id, source_file, copied, ran, attempt, top_attempt, copy_rank,
        groupUniqArrayIf(source_file, ran AND attempt = top_attempt)
            OVER (PARTITION BY component, run_id, test_case_id) AS ran_files
    FROM
    (
        SELECT
            component, run_id, test_case_id, audit_uuid, source_file, attempt,
            ran_in NOT IN ('', toString(run_id)) AS copied,
            NOT copied AND status != 'skipped' AS ran,
            max(attempt) OVER (PARTITION BY component, run_id, test_case_id, source_file, copied) AS top_attempt,
            row_number() OVER (
                PARTITION BY component, run_id, test_case_id, source_file, copied, attempt, status,
                             round(duration_s, 3), fail_message
                ORDER BY audit_uuid DESC
            ) AS copy_rank
        FROM
        (
            SELECT
                component, run_id, test_case_id, audit_uuid, status, duration_s, fail_message,
                props['source_file'] AS source_file,
                props['ran_in'] AS ran_in,
                toUInt32OrZero(props['run_attempt']) AS attempt
            FROM test_case_runs
            WHERE (component, run_id, test_case_id) IN (
                SELECT component, run_id, test_case_id
                FROM test_case_runs
                GROUP BY component, run_id, test_case_id
                HAVING count() > 1
            )
        )
    )
)
WHERE copy_rank > 1
   OR attempt < top_attempt
   OR (copied AND notEmpty(ran_files))
   OR (NOT ran AND NOT copied AND arrayExists(f -> f != source_file, ran_files));

DELETE FROM test_case_runs WHERE audit_uuid IN (SELECT audit_uuid FROM case_outcome_dups);

DELETE FROM run_case_counters WHERE run_id IN (SELECT DISTINCT run_id FROM case_outcome_dups);

INSERT INTO run_case_counters (run_id, component, total_tests, passed, failed, errors, skipped, xfail, xpass)
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
