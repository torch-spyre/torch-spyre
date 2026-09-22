-- Views over test_cases / test_case_runs: per-case history, and the per-run tier counters v2
-- dropped as stored columns. Apply after 10-functional-tests.sql.

-- Per-case detail for one run, the artifact drill-down's expanded row. ALWAYS filter by run_id;
-- unfiltered this joins both tables in full. To select runs by artifact, filter
-- `run_id IN (SELECT ... FROM artifact_results ...)` -- the IN form pushes values down as a
-- key predicate, where an INNER JOIN would stream the whole table through a hash join.
CREATE VIEW IF NOT EXISTS v_case_results AS
SELECT
    cr.run_id      AS run_id,
    cr.test_case_id AS test_case_id,
    cr.component    AS component,
    c.classname     AS classname,
    c.name          AS name,
    c.tags          AS tags,
    cr.status       AS status,
    cr.duration_s   AS duration_s,
    cr.fail_message AS fail_message,
    cr.ts           AS ts
FROM test_case_runs AS cr
LEFT JOIN test_cases AS c
       ON c.test_case_id = cr.test_case_id AND c.component = cr.component;

-- Per-case pass history across runs, for the flaky/regressing panel. Grouped on (component,
-- classname, name), not test_case_id: `tags` is in the identity hash, so re-tagging a test
-- mints a new id and would split its own history.
CREATE VIEW IF NOT EXISTS v_case_trend AS
SELECT
    cr.component  AS component,
    c.classname   AS classname,
    c.name        AS name,
    toDate(cr.ts) AS day,
    count()       AS runs,
    countIf(cr.status = 'passed') AS passed,
    countIf(cr.status IN ('failed', 'error')) AS failed,
    countIf(cr.status = 'skipped') AS skipped,
    passed / runs AS pass_rate,
    avg(cr.duration_s) AS mean_duration_s,
    anyIf(cr.fail_message, cr.fail_message != '') AS sample_fail_message
FROM test_case_runs AS cr
INNER JOIN test_cases AS c
        ON c.test_case_id = cr.test_case_id AND c.component = cr.component
GROUP BY component, classname, name, day;

-- Per-tier counters for one run, so the UI renders integration/regression/trunk from a single
-- execution -- each tier counted by its own tag, since the tier relation is not transitive. The
-- tier list is fixed, not derived from tags present, so a tier with zero matches still returns
-- a zero row rather than being indistinguishable from "absent from the picker".
CREATE VIEW IF NOT EXISTS v_run_tier_counters AS
SELECT
    cr.run_id AS run_id,
    cr.component AS component,
    tier,
    countIf(cr.status = 'passed')  AS passed,
    countIf(cr.status = 'failed')  AS failed,
    countIf(cr.status = 'error')   AS errors,
    countIf(cr.status = 'skipped') AS skipped,
    countIf(cr.status = 'xfail')   AS xfail,
    countIf(cr.status = 'xpass')   AS xpass,
    -- Kept alongside the split columns: existing callers address xfailed as one figure.
    countIf(cr.status IN ('xfail', 'xpass')) AS xfailed,
    count()  AS total,
    if(count() > 0, countIf(cr.status = 'passed') / count(), NULL) AS pass_rate,
    sum(cr.duration_s) AS duration_s
FROM test_case_runs AS cr
INNER JOIN test_cases AS c
        ON c.test_case_id = cr.test_case_id AND c.component = cr.component
ARRAY JOIN ['integration', 'regression', 'trunk', 'unit', 'smoke'] AS tier
WHERE has(c.tags, concat('testtype__', tier))
GROUP BY run_id, component, tier;

-- Completeness of a derived tier report: of the cases tagged for the tier, how many did this
-- run actually execute (needed since the tier relation is not transitive -- a trunk run does
-- not necessarily cover every integration case). If not_covered > 0 it is PARTIAL. Grain is
-- (run_id, component, tier); filter by run_id.
CREATE VIEW IF NOT EXISTS v_tier_report_completeness AS
SELECT
    ran.run_id   AS run_id,
    c.component   AS component,
    tier,
    uniqExact(c.test_case_id)                            AS want_total,
    uniqExactIf(c.test_case_id, has(ran.ids, c.test_case_id)) AS ran_total,
    want_total - ran_total                               AS not_covered,
    if(want_total > 0, ran_total / want_total, NULL)      AS completeness
FROM test_cases AS c
ARRAY JOIN ['integration', 'regression', 'trunk', 'unit', 'smoke'] AS tier
-- The cases each run executed, folded to one row per run, so the per-tier comparison is a set
-- membership test rather than a second pass over the fact table.
CROSS JOIN
(
    SELECT run_id, component, groupUniqArray(test_case_id) AS ids
    FROM test_case_runs
    GROUP BY run_id, component
) AS ran
WHERE has(c.tags, concat('testtype__', tier))
  AND c.component = ran.component
GROUP BY run_id, component, tier;
