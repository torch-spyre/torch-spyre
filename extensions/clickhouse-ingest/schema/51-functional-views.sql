-- Views over test_cases / test_case_runs: per-case history, and the per-run tier counters v2
-- dropped as stored columns, whose tier comes from artifact_results. Apply after 10- and 20-.

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

-- Per-tier counters for one run. The tier is the one its leg was dispatched for
-- (artifact_results.test_type), not the cases' testtype__ tags: a case tagged for five tiers still
-- ran under one. A run with no functional leg has no known tier and is absent.
CREATE VIEW IF NOT EXISTS v_run_tier_counters AS
SELECT
    cr.run_id AS run_id,
    cr.component AS component,
    leg.tier AS tier,
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
INNER JOIN
(
    -- One tier per run: artifact_results holds a row per build/reuse of a leg, and a perf leg
    -- can share a functional leg's run_id.
    SELECT run_id, argMax(test_type, ts) AS tier
    FROM artifact_results
    WHERE result_kind = 'functional'
    GROUP BY run_id
) AS leg ON leg.run_id = cr.run_id
GROUP BY run_id, component, tier;

-- Completeness of a run against its leg's tier: of the cases tagged for that tier, how many
-- this run executed (the tiers do not nest, so a run need not cover its tier's population).
-- not_covered > 0 means PARTIAL. Grain is (run_id, component, tier); filter by run_id.
CREATE VIEW IF NOT EXISTS v_tier_report_completeness AS
SELECT
    ran.run_id    AS run_id,
    ran.component AS component,
    ran.tier      AS tier,
    want.want_total                                   AS want_total,
    ran.ran_total                                     AS ran_total,
    want_total - ran_total                            AS not_covered,
    if(want_total > 0, ran_total / want_total, NULL)  AS completeness
FROM
(
    SELECT cr.run_id AS run_id, cr.component AS component, leg.tier AS tier,
           uniqExactIf(cr.test_case_id, has(c.tags, concat('testtype__', leg.tier))) AS ran_total
    FROM test_case_runs AS cr
    INNER JOIN
    (
        SELECT run_id, argMax(test_type, ts) AS tier
        FROM artifact_results
        WHERE result_kind = 'functional'
        GROUP BY run_id
    ) AS leg ON leg.run_id = cr.run_id
    LEFT JOIN test_cases AS c
           ON c.test_case_id = cr.test_case_id AND c.component = cr.component
    GROUP BY run_id, component, tier
) AS ran
INNER JOIN
(
    SELECT component, substring(tag, 11) AS tier, uniqExact(test_case_id) AS want_total
    FROM test_cases
    ARRAY JOIN tags AS tag
    WHERE startsWith(tag, 'testtype__')
    GROUP BY component, tier
) AS want ON want.component = ran.component AND want.tier = ran.tier;
