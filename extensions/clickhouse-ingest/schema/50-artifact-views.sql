-- Views over the artifact and tag tables: resolve a tag to what it pointed at, and join an
-- artifact to the verdicts recorded against it. Apply after 20-artifacts.sql and
-- 10-functional-tests.sql -- later views select from earlier ones, so file order matters.

-- One row per artifact. The table gets a row on every build, reuse and sources write for the
-- same id, so joining it raw multiplies every downstream count by that artifact's row count.
-- Record rows carry run_url/ref but no sources or deps, so those take the latest NON-EMPTY
-- value, and props merge key-wise with later rows winning.
CREATE VIEW IF NOT EXISTS v_artifacts AS
SELECT
    -- Every source column is qualified: an output alias sharing its name would shadow it.
    -- (ts, audit_timestamp) orders writes: ts is second-resolution, so same-second rows tie.
    r.artifact_id                                                                        AS artifact_id,
    argMax(r.component, (r.ts, r.audit_timestamp))                                       AS component,
    argMax(r.arch, (r.ts, r.audit_timestamp))                                            AS arch,
    argMax(r.kind, (r.ts, r.audit_timestamp))                                            AS kind,
    argMax(r.artifact_name, (r.ts, r.audit_timestamp))                                   AS artifact_name,
    argMin(r.origin, (r.ts, r.audit_timestamp))                                          AS origin,
    argMaxIf(r.identity_deps, (r.ts, r.audit_timestamp), notEmpty(r.identity_deps))      AS identity_deps,
    argMaxIf(r.context_deps, (r.ts, r.audit_timestamp), notEmpty(r.context_deps))        AS context_deps,
    argMaxIf(r.sources, (r.ts, r.audit_timestamp), notEmpty(r.sources))                  AS sources,
    arrayFold((acc, x) -> mapUpdate(acc, x.3),
              arraySort(x -> (x.1, x.2), groupArray((r.ts, r.audit_timestamp, r.props))),
              CAST(map(), 'Map(String, String)'))                                        AS props,
    -- ts is the first write (when it was built), as min(ts) over the raw table would give.
    min(r.ts)                                                                            AS ts,
    max(r.ts)                                                                            AS last_ts,
    count()                                                                              AS writes
FROM artifacts AS r
GROUP BY r.artifact_id;

-- Tag -> the artifact it points at NOW, one row per (tag, component, arch, artifact_name).
-- is_rolling is emergent (ever pointed at more than one artifact), never stored.
CREATE VIEW IF NOT EXISTS v_tag_resolution AS
SELECT
    t.tag                              AS tag,
    any(t.tag_family)                  AS tag_family,
    a.component                        AS component,
    if(a.arch IN ('amd64', 'x86', 'x86-64'), 'x86_64', a.arch) AS arch,
    a.artifact_name                    AS artifact_name,
    argMax(t.artifact_id, t.ts)        AS artifact_id,
    max(t.ts)                          AS resolved_ts,
    count()                            AS promotion_count,
    uniqExact(t.artifact_id) > 1       AS is_rolling  -- within this (component, arch, artifact_name) slot
FROM artifact_tags AS t
INNER JOIN v_artifacts AS a ON a.artifact_id = t.artifact_id
GROUP BY tag, component, arch, artifact_name;

-- The tag picker: one row per tag, so the UI lists channels without resolving each. arch_list
-- is an array since a dated tag spans all three platforms. is_rolling is OR-ed per
-- (component, arch, artifact_name) slot, not a tag-wide uniqExact -- a dated tag legitimately
-- holds one artifact per component variant, so a coarser count would mark every bundle tag rolling.
CREATE VIEW IF NOT EXISTS v_tag_list AS
SELECT
    t.tag                          AS tag,
    any(t.tag_family)              AS tag_family,
    min(t.ts)                      AS first_ts,
    max(t.ts)                      AS last_ts,
    count()                        AS promotion_count,
    uniqExact(t.artifact_id)       AS artifact_count,
    uniqExactIf(t.component, t.component != '') AS component_count,
    arraySort(groupUniqArrayIf(if(t.arch IN ('amd64', 'x86', 'x86-64'), 'x86_64', t.arch),
                               t.arch != ''))   AS arch_list,
    max(slot_artifacts) > 1        AS is_rolling
FROM
(
    -- ifNull: an unjoined row is NULL under a reader's join_use_nulls=1, and a NULL slot key
    -- would pool every unjoined row into one slot.
    -- arch is canonicalized here too, matching v_tag_resolution, so amd64/x86_64 share one slot.
    SELECT at.tag AS tag, at.tag_family AS tag_family, at.artifact_id AS artifact_id,
           at.ts AS ts,
           ifNull(a.component, '') AS component,
           if(ifNull(a.arch, '') IN ('amd64', 'x86', 'x86-64'), 'x86_64', ifNull(a.arch, '')) AS arch,
           uniqExact(at.artifact_id) OVER (
               PARTITION BY at.tag,
                            -- (component, arch, artifact_name) when joined, else the artifact itself.
                            if(ifNull(a.component, '') = '',
                               toString(at.artifact_id),
                               ifNull(a.component, '')),
                            if(ifNull(a.arch, '') IN ('amd64', 'x86', 'x86-64'), 'x86_64', ifNull(a.arch, '')),
                            ifNull(a.artifact_name, ''))
               AS slot_artifacts
    FROM artifact_tags AS at
    LEFT JOIN v_artifacts AS a ON a.artifact_id = at.artifact_id
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
    -- Of passed: an earlier attempt of the case failed.
    coalesce(c.recovered, 0)   AS recovered,
    r.duration_s     AS duration_s,
    -- Denominator excludes xfail/xpass: of the cases whose outcome was in question, how many passed.
    if(total_tests - xfail - xpass > 0,
       passed / (total_tests - xfail - xpass), NULL) AS pass_rate,
    total_tests > 0 AS suite_ran,
    -- 'running' is advisory only (a crashed run keeps this row until the 90-day TTL); shown
    -- here for the drill-down, but aggregating callers must exclude it (see v_tier_trend).
    CAST(r.state = 'running' AS UInt8) AS is_advisory,
    -- Why a failed/error leg did not pass ('' otherwise). A reasons row wins only by outranking
    -- the verdict's own props; failed cases are the fallback, then 'unknown'.
    if(r.state NOT IN ('failed', 'error'), '',
       multiIf(coalesce(x.x_conf, 0) > r.own_conf, x.x_reason,
               r.own_reason != '', r.own_reason,
               failed + errors > 0, 'test_failure',
               'unknown')) AS failure_reason,
    if(failure_reason = '', '',
       if(coalesce(x.x_conf, 0) > r.own_conf, x.x_subreason, r.own_subreason)) AS failure_subreason,
    if(failure_reason = '', '',
       substringUTF8(multiIf(coalesce(x.x_conf, 0) > r.own_conf, x.x_detail,
                         r.own_detail != '', r.own_detail,
                         failed + errors > 0,
                         concat(toString(failed + errors), ' of ', toString(total_tests), ' cases failed'),
                         ''), 1, 300)) AS failure_detail,
    if(failure_reason = '', '',
       multiIf(coalesce(x.x_conf, 0) > r.own_conf AND x.x_log_url != '', x.x_log_url,
               r.props['failure_log_url'] != '', r.props['failure_log_url'],
               r.props['run_url'])) AS failure_log_url,
    if(failure_reason = '', toUInt32(0),
       if(coalesce(x.x_conf, 0) > r.own_conf, x.x_wait_s,
          toUInt32OrZero(r.props['failure_wait_s']))) AS failure_wait_s,
    startsWith(failure_reason, 'infra_') AS failure_is_infra,
    multiIf(failure_reason = '', '',
            coalesce(x.x_conf, 0) > r.own_conf, x.x_source,
            r.own_conf >= 2, 'writer',
            'derived') AS failure_source
FROM
(
    -- One row per verdict, the latest: a leg writes a 'running' seed before its final state and a
    -- re-push repeats it, so raw rows count a run's counters twice. A run can hold a functional
    -- and a capability verdict for one artifact, hence result_kind/test_type in the key.
    -- own_*: the reason the verdict row itself carries -- props.failure_* (3, or its failure_confidence), else the stale-leg
    -- cleanup's closed_reason, a dead runner or diagnose_failure's category (2). closed_reason only
    -- explains an 'error' close: on 'failed' it says why the row was missing, not why it failed.
    SELECT
        *,
        if(startsWith(props['diagnosis'], '{'),
           JSONExtractString(props['diagnosis'], 'category'), props['diagnosis']) AS dg_cat,
        multiIf(props['failure_reason'] != '', props['failure_reason'],
                state = 'error' AND startsWith(props['closed_reason'], 'parent_superseded'), 'superseded',
                state = 'error' AND props['closed_reason'] = 'parent_manual_abort', 'aborted',
                state = 'error' AND props['closed_reason'] = 'parent_groovy_compile_error', 'pipeline_error',
                state = 'error' AND props['closed_reason'] = 'ch_write_timeout', 'ingest_error',
                state = 'error' AND props['closed_reason'] = 'parent_hung_jenkins_restart', 'infra_capacity',
                props['runner_died'] IN ('1', 'true'), 'infra_capacity',
                -- the pre-taxonomy spelling of ingest_error/result_lost
                dg_cat = 'infra_result_lost', 'ingest_error',
                dg_cat NOT IN ('', 'unknown'), dg_cat,
                '') AS own_reason,
        multiIf(props['failure_reason'] != '', props['failure_subreason'],
                state = 'error' AND startsWith(props['closed_reason'], 'parent_superseded'), 'by_newer_run',
                state = 'error' AND props['closed_reason'] = 'parent_manual_abort', 'user',
                state = 'error' AND props['closed_reason'] = 'parent_groovy_compile_error', 'groovy_compile',
                state = 'error' AND props['closed_reason'] = 'ch_write_timeout', 'ch_write_timeout',
                state = 'error' AND props['closed_reason'] = 'parent_hung_jenkins_restart', 'jenkins_restart',
                props['runner_died'] IN ('1', 'true'), 'runner_died',
                dg_cat = 'infra_result_lost', 'result_lost',
                '') AS own_subreason,
        multiIf(props['failure_reason'] != '', props['failure_detail'],
                startsWith(props['diagnosis'], '{'),
                if(JSONExtractString(props['diagnosis'], 'evidence') != '',
                   JSONExtractString(props['diagnosis'], 'evidence'),
                   JSONExtractString(props['diagnosis'], 'why')),
                state = 'error', props['closed_reason'],
                '') AS own_detail,
        -- A writer's props rank 3 unless it marked them a default (results' failure_confidence=1).
        toUInt8(multiIf(props['failure_reason'] != '', toUInt8OrDefault(props['failure_confidence'], toUInt8(3)),
                        own_reason != '', 2, 0)) AS own_conf
    FROM artifact_results
    ORDER BY ts DESC, audit_timestamp DESC
    LIMIT 1 BY artifact_id, run_id, result_kind, test_type
) AS r
LEFT JOIN v_artifacts AS a ON a.artifact_id = r.artifact_id
LEFT JOIN (
    -- The best reasons row per verdict: each source's latest (so an unmerged older row cannot
    -- win), then the highest confidence across sources.
    SELECT
        artifact_id,
        run_id,
        result_kind,
        test_type,
        max(s_conf)                                    AS x_conf,
        argMax(s_reason,    (s_conf, s_updated_at))    AS x_reason,
        argMax(s_subreason, (s_conf, s_updated_at))    AS x_subreason,
        argMax(s_detail,    (s_conf, s_updated_at))    AS x_detail,
        argMax(s_log_url,   (s_conf, s_updated_at))    AS x_log_url,
        argMax(s_wait_s,    (s_conf, s_updated_at))    AS x_wait_s,
        argMax(source,      (s_conf, s_updated_at))    AS x_source
    FROM
    (
        SELECT
            artifact_id,
            run_id,
            result_kind,
            test_type,
            source,
            max(updated_at)                           AS s_updated_at,
            argMax(confidence,        updated_at)     AS s_conf,
            argMax(failure_reason,    updated_at)     AS s_reason,
            argMax(failure_subreason, updated_at)     AS s_subreason,
            argMax(failure_detail,    updated_at)     AS s_detail,
            argMax(failure_log_url,   updated_at)     AS s_log_url,
            argMax(failure_wait_s,    updated_at)     AS s_wait_s
        FROM artifact_result_reasons
        GROUP BY artifact_id, run_id, result_kind, test_type, source
    )
    GROUP BY artifact_id, run_id, result_kind, test_type
) AS x ON x.artifact_id = r.artifact_id AND x.run_id = r.run_id
      AND x.result_kind = r.result_kind AND x.test_type = r.test_type
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
        sum(xpass)       AS xpass,
        sum(recovered)   AS recovered
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
    e.xfail        AS xfail,
    e.xpass        AS xpass,
    e.duration_s   AS duration_s,
    e.pass_rate    AS pass_rate,
    e.suite_ran    AS suite_ran,
    e.ts           AS ts,
    e.failure_reason    AS failure_reason,
    e.failure_subreason AS failure_subreason,
    e.failure_detail    AS failure_detail,
    e.failure_log_url   AS failure_log_url,
    e.failure_wait_s    AS failure_wait_s,
    e.failure_is_infra  AS failure_is_infra,
    e.failure_source    AS failure_source
FROM v_tag_resolution AS tr
INNER JOIN v_artifact_results_enriched AS e ON e.artifact_id = tr.artifact_id;

-- Every failed/error verdict with its reason, for "why" roll-ups (per arch, per tag family).
CREATE VIEW IF NOT EXISTS v_artifact_result_reasons AS
SELECT
    artifact_id,
    run_id,
    component,
    artifact_name,
    run_arch,
    result_kind,
    test_type,
    state,
    ts,
    failed,
    errors,
    total_tests,
    failure_reason,
    failure_subreason,
    failure_detail,
    failure_log_url,
    failure_wait_s,
    failure_is_infra,
    failure_source
FROM v_artifact_results_enriched
WHERE failure_reason != '';

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
    -- artifact_refs is ReplacingMergeTree, so unmerged duplicates read back until a merge.
    arrayDistinct(groupArray(f.ref)) AS refs
FROM v_tag_resolution AS tr
INNER JOIN v_artifacts AS a ON a.artifact_id = tr.artifact_id
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
    sum(e.xfail)            AS xfail,
    sum(e.xpass)            AS xpass,
    sum(e.recovered)        AS recovered,
    -- Same denominator as v_artifact_results_enriched: xfail/xpass excluded.
    if(sum(e.total_tests) - sum(e.xfail) - sum(e.xpass) > 0,
       sum(e.passed) / (sum(e.total_tests) - sum(e.xfail) - sum(e.xpass)), NULL) AS pass_rate,
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
