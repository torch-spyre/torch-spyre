-- Views over the artifact and tag tables: resolve a tag to what it pointed at, and join an
-- artifact to the verdicts recorded against it.
-- Apply after 20-artifacts.sql and 10-functional-tests.sql: v_artifact_results_enriched reads
-- test_case_runs, and later views here select from earlier ones, so order within the file
-- matters too.

-- Tag -> the artifact it points at NOW, one row per (tag, component, arch).
-- The only correct resolution primitive. `is_rolling` is emergent (a tag is rolling iff it
-- has ever pointed at more than one artifact), never stored, so it cannot contradict the
-- rows it summarises.
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

-- The tag picker. One row per tag, so the UI can list channels without resolving each.
-- arch_list is an array because a dated tag spans all three platforms and the picker
-- shows that span before a platform is chosen.
-- is_rolling is computed per (component, arch) slot and then OR-ed, not as
-- uniqExact(artifact_id) over the whole tag: a dated tag legitimately holds one artifact per
-- component, so a tag-wide count marks every bundle tag rolling.
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

-- Base results view: every run verdict with its artifact's identity attached. The input
-- to the trend and tag views, and the artifact drill-down's own source.
-- artifact_arch vs run_arch are deliberately separate: a 'multi' manifest is tested on
-- one platform, so only run_arch answers "which platform did this pass on".
-- The counters are derived from test_case_runs rather than read off artifact_results, which
-- no longer stores them: a stored copy drifts the moment a delta run copies a covering run's
-- cases in. Counted over the run's whole row set (executed plus copied), which is the point
-- of the copy. Use props['ran_in'] = run_id for only what this run itself executed.
-- suite_ran distinguishes "the suite never executed" (no case rows at all) from "it ran and
-- regressed" -- state alone cannot, and a 0/0 row would otherwise chart as 0% pass.
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
    -- coalesced because the LEFT JOIN below yields NULL, not 0, for a run with no case rows
    -- under join_use_nulls=1 -- which would break the `suite_ran = 0` filter callers are told
    -- to use, and silently, since NULL simply fails the predicate.
    coalesce(c.total_tests, 0) AS total_tests,
    coalesce(c.passed, 0)      AS passed,
    coalesce(c.failed, 0)      AS failed,
    coalesce(c.errors, 0)      AS errors,
    coalesce(c.skipped, 0)     AS skipped,
    -- Previously omitted, which left them in total_tests but in no bucket: the four buckets
    -- did not sum to the total (72,887 rows on prod) and pass_rate read 92.11%, not 97.88%.
    coalesce(c.xfail, 0)       AS xfail,
    coalesce(c.xpass, 0)       AS xpass,
    r.duration_s     AS duration_s,
    -- Denominator excludes xfail/xpass: of the cases whose outcome was in question, how many
    -- passed. total_tests keeps the full count for the other convention.
    if(total_tests - xfail - xpass > 0,
       passed / (total_tests - xfail - xpass), NULL) AS pass_rate,
    total_tests > 0 AS suite_ran,
    -- 'running' is advisory only: a crashed run keeps this row until the 90-day TTL. Kept
    -- visible here so the drill-down shows a live run, but aggregating callers must exclude
    -- it, as v_tier_trend and v_run_coverage do.
    CAST(r.state = 'running' AS UInt8) AS is_advisory
FROM artifact_results AS r
LEFT JOIN artifacts AS a ON a.artifact_id = r.artifact_id
-- LEFT JOIN, not INNER: a run with no case rows must still appear, with total_tests = 0.
-- That is exactly the "suite never executed" signal, and an INNER JOIN would delete it.
LEFT JOIN (
    -- run_case_counters, not test_case_runs: pre-aggregated on insert, so this reads one row
    -- per run. sum() is still required -- SummingMergeTree collapses on merge, not on read.
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

-- Combined functional + performance results for every artifact in one tag.
-- The INNER JOIN is on (tag-resolved artifact_id) -- resolving first is what makes this
-- safe where a direct join on tag is not (Trap 1). One row per run of a member artifact.
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

-- The membership list behind a tag: which artifacts are in it, with their addresses.
-- Separate from v_tag_results because a tag member with no test run must still be listed --
-- an inner join to results would hide the untested artifacts the page exists to surface.
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

-- Daily trend for the overview page, split by platform so the three arches chart side by
-- side. Grain is one row per (day, tag_family, result_kind, test_type, run_arch, component);
-- the UI aggregates upward, since summing a day's rows is correct but re-deriving a
-- per-component split from a rolled-up row is not.
-- pass_rate is computed from the summed counters, not averaged over runs: averaging rates
-- weights a 3-test run equally with a 3,000-test one.
-- One row per TAG, so a rolling channel and its dated alias each contribute a row and
-- summing `runs` double-counts. Grouping by tag_family does NOT fix this, since both
-- duplicate tags sit in one family. Counts here are exact only within a single `tag`; an
-- exact total is uniqExact(run_id) from v_artifact_results_enriched. Never max() over
-- tag_family -- it assumes families describe the same runs and drops the disjoint ones.
CREATE VIEW IF NOT EXISTS v_tier_trend AS
-- One row per (day, tag_family, kind, test_type, arch, component).
-- The tag join is deduped to one row per artifact first: rolling and dated tags coexist by
-- design (an artifact carries both `weekly` and `weekly-2026-09-05`), so joining
-- v_tag_resolution directly fans out and counts the same run once per tag.
-- `runs` counts distinct run_id, not result rows -- one run can carry several result rows,
-- and a run-count label must not follow the row count.
SELECT
    -- The RESULT's timestamp, not the tag's resolution timestamp. `d` answers which family an
    -- artifact belongs to, never when its results ran: an artifact re-tagged into the same
    -- family on a later day would otherwise fold every earlier day's results into that day
    -- (measured 504 of 1,220 joined rows landing on the wrong day on prod).
    toDate(e.ts)            AS day,
    d.fam                   AS tag_family,
    e.result_kind           AS result_kind,
    e.test_type             AS test_type,
    e.run_arch              AS run_arch,
    e.component             AS component,
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
    -- Latest resolution per (artifact, family): collapses each family's rolling/dated
    -- pair to one row. Grouping by artifact_id alone would collapse nightly and weekly
    -- together too, picking whichever family resolved later and silently dropping the
    -- other family's contribution to its own trend -- a normal case, since one artifact
    -- commonly carries both tags.
    SELECT artifact_id,
           tag_family                      AS fam,
           max(resolved_ts)                AS rts
    FROM v_tag_resolution
    WHERE tag_family IN ('nightly', 'weekly')
    GROUP BY artifact_id, tag_family
) AS d
INNER JOIN v_artifact_results_enriched AS e ON e.artifact_id = d.artifact_id
-- state='running' is advisory display only and a crashed run leaves a stale row until the
-- TTL reaps it, so it must never contribute to a trend.
WHERE e.state != 'running'
GROUP BY day, tag_family, result_kind, test_type, run_arch, component;
