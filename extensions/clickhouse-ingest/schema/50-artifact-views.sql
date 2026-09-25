-- Views over the artifact and tag tables: resolve a tag to what it pointed at, and join an
-- artifact to the verdicts recorded against it. Apply after 20-artifacts.sql and
-- 10-functional-tests.sql -- later views select from earlier ones, so file order matters.

-- Tag -> the artifact it points at NOW, one row per (tag, component, arch). is_rolling is
-- emergent (ever pointed at more than one artifact), never stored.
CREATE VIEW IF NOT EXISTS v_tag_resolution AS
SELECT
    t.tag                              AS tag,
    any(t.tag_family)                  AS tag_family,
    a.component                        AS component,
    if(a.arch IN ('amd64', 'x86', 'x86-64'), 'x86_64', a.arch) AS arch,
    argMax(t.artifact_id, t.ts)        AS artifact_id,
    max(t.ts)                          AS resolved_ts,
    count()                            AS promotion_count,
    uniqExact(t.artifact_id) > 1       AS is_rolling  -- within this (component, arch) slot
FROM artifact_tags AS t
INNER JOIN artifacts AS a ON a.artifact_id = t.artifact_id
GROUP BY tag, component, arch;

-- The tag picker: one row per tag, so the UI lists channels without resolving each. arch_list
-- is an array since a dated tag spans all three platforms. is_rolling is OR-ed per
-- (component, arch) slot, not a tag-wide uniqExact -- a dated tag legitimately holds one
-- artifact per component, so a tag-wide count would mark every bundle tag rolling.
CREATE VIEW IF NOT EXISTS v_tag_list AS
SELECT
    t.tag                          AS tag,
    any(t.tag_family)              AS tag_family,
    min(t.ts)                      AS first_ts,
    max(t.ts)                      AS last_ts,
    count()                        AS promotion_count,
    uniqExact(t.artifact_id)       AS artifact_count,
    uniqExact(t.component)         AS component_count,
    arraySort(groupUniqArray(if(t.arch IN ('amd64', 'x86', 'x86-64'), 'x86_64', t.arch))) AS arch_list,
    max(slot_artifacts) > 1        AS is_rolling
FROM
(
    SELECT at.tag AS tag, at.tag_family AS tag_family, at.artifact_id AS artifact_id,
           at.ts AS ts, a.component AS component, a.arch AS arch,
           uniqExact(at.artifact_id) OVER (PARTITION BY at.tag, a.component, a.arch)
               AS slot_artifacts
    FROM artifact_tags AS at
    LEFT JOIN artifacts AS a ON a.artifact_id = at.artifact_id
) AS t
GROUP BY tag;

-- Base results view: every run verdict with its artifact's identity attached -- feeds the
-- trend/tag views and the artifact drill-down. artifact_arch vs run_arch stay separate since a
-- 'multi' manifest is tested on one platform. Counters are derived from test_case_runs, not
-- read off artifact_results (which stores none, to avoid drift on delta-run case copies), over
-- the run's whole row set (executed plus copied); use props['ran_in'] = run_id for only what
-- this run itself executed. suite_ran tells "never executed" from "ran and regressed".
CREATE VIEW IF NOT EXISTS v_artifact_results_enriched AS
SELECT
    r.ts             AS ts,
    r.artifact_id    AS artifact_id,
    r.run_id        AS run_id,
    a.component      AS component,
    a.kind           AS kind,
    a.artifact_name  AS artifact_name,
    if(a.arch IN ('amd64', 'x86', 'x86-64'), 'x86_64', a.arch) AS artifact_arch,
    if(r.arch IN ('amd64', 'x86', 'x86-64'), 'x86_64', r.arch) AS run_arch,
    a.origin         AS origin,
    r.result_kind    AS result_kind,
    r.test_type      AS test_type,
    r.state          AS state,
    -- coalesced: the LEFT JOIN below yields NULL, not 0, under join_use_nulls=1, which would
    -- silently break the `suite_ran = 0` filter.
    coalesce(c.total_tests, 0) AS total_tests,
    coalesce(c.passed, 0)      AS passed,
    coalesce(c.failed, 0)      AS failed,
    coalesce(c.errors, 0)      AS errors,
    coalesce(c.skipped, 0)     AS skipped,
    -- Previously omitted, leaving pass_rate at 92.11% instead of 97.88% (72,887 prod rows).
    coalesce(c.xfail, 0)       AS xfail,
    coalesce(c.xpass, 0)       AS xpass,
    r.duration_s     AS duration_s,
    -- Denominator excludes xfail/xpass: of the cases whose outcome was in question, how many passed.
    if(total_tests - xfail - xpass > 0,
       passed / (total_tests - xfail - xpass), NULL) AS pass_rate,
    total_tests > 0 AS suite_ran,
    -- 'running' is advisory only (a crashed run keeps this row until the 90-day TTL); shown
    -- here for the drill-down, but aggregating callers must exclude it (see v_tier_trend).
    CAST(r.state = 'running' AS UInt8) AS is_advisory
FROM artifact_results AS r
LEFT JOIN artifacts AS a ON a.artifact_id = r.artifact_id
-- LEFT JOIN, not INNER: a run with no case rows must still appear, with total_tests = 0.
LEFT JOIN (
    -- run_case_counters, not test_case_runs: pre-aggregated, one row per run; sum() is still
    -- required since SummingMergeTree collapses on merge, not on read.
    SELECT
        run_id,
        sum(total_tests) AS total_tests,
        sum(passed)      AS passed,
        sum(failed)      AS failed,
        sum(errors)      AS errors,
        sum(skipped)     AS skipped,
        sum(xfail)       AS xfail,
        sum(xpass)       AS xpass
    FROM run_case_counters
    GROUP BY run_id
) AS c ON c.run_id = r.run_id;

-- Combined functional + performance results for every artifact in one tag. INNER JOIN on the
-- tag-resolved artifact_id, not tag directly, is what makes this safe (see v_tag_resolution).
CREATE VIEW IF NOT EXISTS v_tag_results AS
SELECT
    tr.tag         AS tag,
    tr.tag_family  AS tag_family,
    tr.component   AS component,
    tr.arch        AS artifact_arch,   -- already canonical via v_tag_resolution
    tr.resolved_ts AS resolved_ts,
    e.artifact_id  AS artifact_id,
    e.artifact_name AS artifact_name,
    e.run_id      AS run_id,
    e.run_arch     AS run_arch,
    e.result_kind  AS result_kind,
    e.test_type    AS test_type,
    e.state        AS state,
    e.total_tests  AS total_tests,
    e.passed       AS passed,
    e.failed       AS failed,
    e.errors       AS errors,
    e.skipped      AS skipped,
    e.duration_s   AS duration_s,
    e.pass_rate    AS pass_rate,
    e.suite_ran    AS suite_ran,
    e.ts           AS ts
FROM v_tag_resolution AS tr
INNER JOIN v_artifact_results_enriched AS e ON e.artifact_id = tr.artifact_id;

-- The membership list behind a tag: which artifacts are in it, with their addresses. Separate
-- from v_tag_results so an untested member (no results to inner-join against) still shows up.
CREATE VIEW IF NOT EXISTS v_tag_artifacts AS
SELECT
    tr.tag          AS tag,
    tr.tag_family   AS tag_family,
    tr.component    AS component,
    tr.arch         AS arch,           -- already canonical via v_tag_resolution
    tr.artifact_id  AS artifact_id,
    tr.resolved_ts  AS resolved_ts,
    a.artifact_name AS artifact_name,
    a.kind          AS kind,
    a.origin        AS origin,
    a.props['id12'] AS id12,
    a.sources       AS sources,
    groupArray(f.ref) AS refs
FROM v_tag_resolution AS tr
INNER JOIN artifacts AS a ON a.artifact_id = tr.artifact_id
LEFT JOIN artifact_refs AS f ON f.artifact_id = tr.artifact_id
GROUP BY tag, tag_family, component, arch, artifact_id, resolved_ts,
         artifact_name, kind, origin, id12, sources;

-- Daily trend for the overview page, one row per (day, tag_family, result_kind, test_type,
-- run_arch, component) -- the UI aggregates upward. pass_rate is computed from summed counters,
-- not averaged over runs, so a 3-test run doesn't weigh the same as a 3,000-test one. One row
-- per TAG (not tag_family): a rolling channel and its dated alias both contribute, so summing
-- `runs` across tag_family double-counts; an exact total is uniqExact(run_id) from
-- v_artifact_results_enriched -- never max() over tag_family.
CREATE VIEW IF NOT EXISTS v_tier_trend AS
-- The tag join is deduped to one row per artifact first: rolling and dated tags coexist by
-- design, so joining v_tag_resolution directly would fan out and count the same run twice.
-- `runs` counts distinct run_id, not result rows, since one run can carry several.
SELECT
    -- The RESULT's timestamp, not the tag's resolution timestamp -- `d` answers which family an
    -- artifact belongs to, never when its results ran (else a re-tag folds earlier days' results
    -- into the re-tag day; measured 504/1,220 joined rows on the wrong day on prod).
    toDate(e.ts)            AS day,
    d.fam                   AS tag_family,
    e.result_kind           AS result_kind,
    e.test_type              AS test_type,
    e.run_arch               AS run_arch,
    e.component               AS component,
    uniqExact(e.run_id)     AS runs,
    uniqExact(e.artifact_id) AS artifacts,
    uniqExactIf(e.run_id, e.state != 'passed') AS failed_runs,
    sum(e.total_tests)      AS total_tests,
    sum(e.passed)           AS passed,
    sum(e.failed)           AS failed,
    sum(e.errors)           AS errors,
    sum(e.skipped)          AS skipped,
    if(sum(e.total_tests) > 0, sum(e.passed) / sum(e.total_tests), NULL) AS pass_rate,
    avg(e.duration_s)       AS mean_duration_s
FROM
(
    -- Latest resolution per (artifact, family): collapses each family's rolling/dated pair to
    -- one row. Grouping by artifact_id alone would also collapse nightly and weekly together,
    -- since one artifact commonly carries both tags.
    SELECT artifact_id,
           tag_family                      AS fam,
           max(resolved_ts)                AS rts
    FROM v_tag_resolution
    WHERE tag_family IN ('nightly', 'weekly')
    GROUP BY artifact_id, tag_family
) AS d
INNER JOIN v_artifact_results_enriched AS e ON e.artifact_id = d.artifact_id
-- state='running' is advisory display only; excluded so a stale crashed-run row never trends.
WHERE e.state != 'running'
GROUP BY day, tag_family, result_kind, test_type, run_arch, component;
