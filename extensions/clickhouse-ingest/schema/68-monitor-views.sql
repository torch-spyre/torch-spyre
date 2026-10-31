-- Silent-failure checks the monitor job reads. Each row is one violation; no row means the check holds. Limits and windows are
-- proposals, set here so a threshold change is a view change reviewed like one.

-- Ingest freshness per writer. `silent` = no row in 7 days, `stale` = newer rows exist but the newest is past the limit.
CREATE VIEW IF NOT EXISTS v_mon_freshness AS
SELECT writer, limit_min, newest_at, rows_24h, rows_7d,
       if(rows_7d = 0, NULL, greatest(0, round(dateDiff('second', newest_at, now()) / 60, 1))) AS minutes_stale,
       multiIf(rows_7d = 0, 'silent', dateDiff('second', newest_at, now()) > limit_min * 60, 'stale', 'ok') AS status
FROM
(
    SELECT 'artifact_results' AS writer, 120 AS limit_min, max(ts) AS newest_at, countIf(ts >= now() - INTERVAL 1 DAY) AS rows_24h, count() AS rows_7d
    FROM artifact_results WHERE ts >= now() - INTERVAL 7 DAY
    UNION ALL
    SELECT 'test_case_runs', 120, max(ts), countIf(ts >= now() - INTERVAL 1 DAY), count()
    FROM test_case_runs WHERE ts >= now() - INTERVAL 7 DAY
    UNION ALL
    SELECT 'benchmark_runs', 1440, max(ts), countIf(ts >= now() - INTERVAL 1 DAY), count()
    FROM benchmark_runs WHERE ts >= now() - INTERVAL 7 DAY
    UNION ALL
    SELECT 'pipeline_runs', 60, max(updated_at), countIf(updated_at >= now() - INTERVAL 1 DAY), count()
    FROM pipeline_runs WHERE updated_at >= now() - INTERVAL 7 DAY
    UNION ALL
    SELECT 'artifact_tags', 240, max(ts), countIf(ts >= now() - INTERVAL 1 DAY), count()
    FROM artifact_tags WHERE ts >= now() - INTERVAL 7 DAY
);

-- A Jenkins orchestrator run that reads green or success while it counted no tests, or whose log tail says it was aborted or timed
-- out. Latest phase of each run only; the last 7 days.
CREATE VIEW IF NOT EXISTS v_mon_fake_pass AS
SELECT run_key, trigger_source, repo, pr_number, started_at, build_url, tests_total,
       multiIf(tests_total = 0, 'no_tests_counted', 'aborted_log_tail') AS reason
FROM pipeline_runs FINAL
WHERE source = 'jenkins' AND pipeline_type = 'orchestrator' AND started_at >= now() - INTERVAL 7 DAY
  AND (verdict = 'green' OR (verdict = '' AND result = 'success'))
  AND (tests_total = 0
       OR positionCaseInsensitive(fail_log_tail, 'Aborted by') > 0
       OR positionCaseInsensitive(fail_log_tail, 'Cancelling nested steps due to timeout') > 0);

-- A tagged artifact whose latest passed functional leg counted no tests. Smoke legs write no cases by design and are left out.
CREATE VIEW IF NOT EXISTS v_mon_zero_total_legs AS
SELECT e.artifact_id AS artifact_id, e.run_id AS run_id, e.component AS component, e.run_arch AS arch, e.test_type AS test_type,
       e.ts AS ts, groupUniqArray(t.tag) AS tags
FROM v_artifact_results_enriched AS e
INNER JOIN artifact_tags AS t ON t.artifact_id = e.artifact_id
WHERE e.result_kind = 'functional' AND e.test_type != 'smoke' AND e.state = 'passed' AND e.total_tests = 0
  AND e.ts >= now() - INTERVAL 7 DAY
GROUP BY e.artifact_id, e.run_id, e.component, e.run_arch, e.test_type, e.ts;
