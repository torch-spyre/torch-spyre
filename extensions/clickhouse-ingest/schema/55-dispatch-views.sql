-- Subscription matching, the one definition the artifact-dispatch job, its preview and the
-- dashboard read. Apply after 25-artifact-dispatch.sql and 50-artifact-views.sql (v_artifacts).

-- The latest verdict per (artifact, run, test_type); 'running' is not a verdict.
-- seen_at is when the verdict as it stands landed; ts is the run's own time, which uploads backdate.
CREATE VIEW IF NOT EXISTS v_dispatch_verdicts AS
SELECT
    r.artifact_id                     AS artifact_id,
    r.run_id                          AS run_id,
    r.test_type                       AS test_type,
    argMax(r.state, r.ts)             AS state,
    argMax(r.arch, r.ts)              AS arch,
    min(r.ts)                         AS ts,
    max(r.audit_timestamp)            AS seen_at
FROM v_artifact_results_published AS r
WHERE r.state != 'running'
GROUP BY r.artifact_id, r.run_id, r.test_type;

-- Everything a subscription can fire on, one row per (event_type, event_key, artifact).
-- event_ts is the FIRST row's time: main/nightly/weekly/pr writers re-insert a tag on every reuse
-- (one `main` pair holds 875 rows), and a re-insert is not a new event. seen_at is when the event
-- landed (insert time), the order a poller reads in: results land up to days after their ts.
CREATE VIEW IF NOT EXISTS v_artifact_events AS
SELECT 'artifact.tagged' AS event_type, t.tag AS event_key, t.artifact_id AS artifact_id,
       min(t.ts) AS event_ts, t.tag AS tag, any(t.tag_family) AS tag_family,
       argMin(t.props, t.ts) AS tag_props, '' AS test_type, '' AS state,
       min(t.audit_timestamp) AS seen_at
FROM artifact_tags AS t
GROUP BY t.tag, t.artifact_id
UNION ALL
SELECT 'artifact.recorded', '', a.artifact_id, min(a.ts), '', '', CAST(map(), 'Map(String, String)'), '', '',
       min(a.audit_timestamp)
FROM artifacts AS a
GROUP BY a.artifact_id
UNION ALL
SELECT 'artifact.promoted', '', a.artifact_id, min(a.ts), '', '', CAST(map(), 'Map(String, String)'), '', '',
       min(a.audit_timestamp)
FROM artifacts AS a
WHERE a.origin = 'promoted'
GROUP BY a.artifact_id
UNION ALL
SELECT 'results.recorded', concat(toString(v.run_id), '/', v.test_type), v.artifact_id, v.ts, '', '',
       CAST(map(), 'Map(String, String)'), v.test_type, v.state, v.seen_at
FROM v_dispatch_verdicts AS v;

-- Every (artifact, subscription) pair whose artifact filters match, enabled or not: what a
-- request may run, and the base the event filters below narrow.
CREATE VIEW IF NOT EXISTS v_subscription_artifacts AS
SELECT
    a.artifact_id      AS artifact_id,
    a.component        AS component,
    a.arch             AS arch,
    a.kind             AS kind,
    a.artifact_name    AS artifact_name,
    s.subscription_id  AS subscription_id,
    s.enabled          AS enabled,
    s.event_types      AS event_types,
    s.tag_family       AS sub_tag_family,
    s.tag_pattern      AS sub_tag_pattern,
    s.tag_props        AS sub_tag_props,
    s.exclude_families AS exclude_families,
    s.exclude_tag_patterns AS exclude_tag_patterns,
    s.result_test_types AS result_test_types,
    s.result_states    AS result_states,
    s.not_before       AS not_before,
    s.require          AS require,
    s.unless_exists    AS unless_exists,
    s.max_per_hour     AS max_per_hour,
    s.max_per_day      AS max_per_day,
    s.coalesce_minutes AS coalesce_minutes,
    s.target_type      AS target_type,
    s.target           AS target,
    s.params           AS params,
    s.payload_fields   AS payload_fields,
    s.credential_id    AS credential_id,
    s.owner            AS owner
FROM v_artifacts AS a
CROSS JOIN (SELECT * FROM artifact_subscriptions FINAL) AS s
WHERE (s.component = '' OR s.component = a.component)
  AND (s.arch = '' OR s.arch = a.arch)
  AND (s.kind = '' OR s.kind = a.kind)
  AND (s.artifact_name = '' OR s.artifact_name = a.artifact_name);

-- Who an event triggers: one row per (event, subscription) whose filters match, with `reason` the
-- first thing stopping it ('' = it fires) and the automatic dispatch it got. Rate limits and
-- coalescing depend on the dispatch history, so only `dispatch preview` shows them.
-- match_reason is reason without the enabled / not_before gates: what a draft or a feed reads.
-- matched_at is when the event became due: its own seen_at, or the last required verdict's.
-- v_feed_events repeats the event filters below over query parameters; change both together.
CREATE VIEW IF NOT EXISTS v_artifact_subscribers AS
SELECT
    m.event_type       AS event_type,
    m.event_key        AS event_key,
    m.event_ts         AS event_ts,
    m.tag              AS tag,
    m.tag_family       AS tag_family,
    m.test_type        AS test_type,
    m.state            AS state,
    m.artifact_id      AS artifact_id,
    m.component        AS component,
    m.arch             AS arch,
    m.kind             AS kind,
    m.artifact_name    AS artifact_name,
    m.subscription_id  AS subscription_id,
    m.enabled          AS enabled,
    multiIf(
        has(m.exclude_families, m.tag_family)
            OR arrayExists(p -> match(m.tag, concat('^(?:', p, ')$')), m.exclude_tag_patterns), 'excluded',
        notEmpty(m.missing), concat('waiting: ', arrayStringConcat(m.missing, ', ')),
        notEmpty(m.blocking), concat('unless: ', arrayStringConcat(m.blocking, ', ')),
        '')            AS match_reason,
    multiIf(
        NOT m.enabled, 'disabled',
        m.event_ts < m.not_before, 'before not_before',
        match_reason)  AS reason,
    reason = ''        AS auto,
    m.max_per_hour     AS max_per_hour,
    m.max_per_day      AS max_per_day,
    m.coalesce_minutes AS coalesce_minutes,
    m.target_type      AS target_type,
    m.target           AS target,
    m.params           AS params,
    m.payload_fields   AS payload_fields,
    m.credential_id    AS credential_id,
    m.owner            AS owner,
    -- ifNull: an unjoined row is NULL under a reader's join_use_nulls=1.
    ifNull(d.state, '')      AS dispatch_state,
    ifNull(d.build_url, '')  AS build_url,
    ifNull(d.error, '')      AS dispatch_error,
    d.updated_at             AS dispatched_at,
    m.seen_at                AS seen_at,
    m.matched_at             AS matched_at
FROM
(
    SELECT e.event_type AS event_type, e.event_key AS event_key, e.event_ts AS event_ts,
           e.tag AS tag, e.tag_family AS tag_family, e.test_type AS test_type, e.state AS state,
           e.seen_at AS seen_at,
           sa.artifact_id AS artifact_id, sa.component AS component, sa.arch AS arch,
           sa.kind AS kind, sa.artifact_name AS artifact_name,
           sa.subscription_id AS subscription_id, sa.enabled AS enabled,
           sa.not_before AS not_before, sa.exclude_families AS exclude_families,
           sa.exclude_tag_patterns AS exclude_tag_patterns,
           sa.max_per_hour AS max_per_hour, sa.max_per_day AS max_per_day,
           sa.coalesce_minutes AS coalesce_minutes, sa.target_type AS target_type,
           sa.target AS target, sa.params AS params, sa.payload_fields AS payload_fields,
           sa.credential_id AS credential_id, sa.owner AS owner,
           -- A condition is '<test_type>' or '<test_type>=<state>', a key of the artifact's first_seen.
           arrayFilter(c -> NOT mapContains(v.first_seen, c), sa.require) AS missing,
           arrayFilter(c -> mapContains(v.first_seen, c), sa.unless_exists) AS blocking,
           arrayMax(arrayPushBack(arrayMap(c -> v.first_seen[c], sa.require), e.seen_at)) AS matched_at
    FROM v_artifact_events AS e
    INNER JOIN v_subscription_artifacts AS sa ON sa.artifact_id = e.artifact_id
    LEFT JOIN
    (
        -- When each test type, and each test_type=state, first held for the artifact.
        SELECT artifact_id,
               minMap(map(test_type, seen_at, concat(test_type, '=', state), seen_at)) AS first_seen
        FROM v_dispatch_verdicts
        GROUP BY artifact_id
    ) AS v ON v.artifact_id = e.artifact_id
    WHERE has(sa.event_types, e.event_type)
      AND (e.event_type != 'artifact.tagged' OR (
              (sa.sub_tag_family = '' OR sa.sub_tag_family = e.tag_family)
          AND (sa.sub_tag_pattern = '' OR match(e.tag, concat('^(?:', sa.sub_tag_pattern, ')$')))
          AND arrayAll(k -> e.tag_props[k] = sa.sub_tag_props[k], mapKeys(sa.sub_tag_props))))
      AND (e.event_type != 'results.recorded' OR (
              (empty(sa.result_test_types) OR has(sa.result_test_types, e.test_type))
          AND (empty(sa.result_states) OR has(sa.result_states, e.state))))
) AS m
LEFT JOIN
(
    SELECT subscription_id, artifact_id, event_type, event_key, state, build_url, error, updated_at
    FROM artifact_dispatches FINAL
    WHERE requested_by = 'auto'
) AS d ON d.subscription_id = m.subscription_id AND d.artifact_id = m.artifact_id
      AND d.event_type = m.event_type AND d.event_key = m.event_key;

-- The event feed: v_artifact_subscribers' matching for one ad-hoc filter set, passed as parameters
-- named like the artifact_subscriptions columns ('' / [] / map() = any). No enabled, not_before,
-- rate limit or coalescing: a feed reader polls. Rows with match_reason != '' are kept, to say why.
CREATE VIEW IF NOT EXISTS v_feed_events AS
SELECT
    e.event_type       AS event_type,
    e.event_key        AS event_key,
    e.event_ts         AS event_ts,
    e.seen_at          AS seen_at,
    arrayMax(arrayPushBack(arrayMap(c -> v.first_seen[c], {require:Array(String)}), e.seen_at)) AS matched_at,
    e.tag              AS tag,
    e.tag_family       AS tag_family,
    e.test_type        AS test_type,
    e.state            AS state,
    a.artifact_id      AS artifact_id,
    a.component        AS component,
    a.arch             AS arch,
    a.kind             AS kind,
    a.artifact_name    AS artifact_name,
    arrayFilter(c -> NOT mapContains(v.first_seen, c), {require:Array(String)}) AS missing,
    arrayFilter(c -> mapContains(v.first_seen, c), {unless_exists:Array(String)}) AS blocking,
    multiIf(
        has({exclude_families:Array(String)}, e.tag_family)
            OR arrayExists(p -> match(e.tag, concat('^(?:', p, ')$')), {exclude_tag_patterns:Array(String)}), 'excluded',
        notEmpty(missing), concat('waiting: ', arrayStringConcat(missing, ', ')),
        notEmpty(blocking), concat('unless: ', arrayStringConcat(blocking, ', ')),
        '')            AS match_reason
FROM v_artifact_events AS e
INNER JOIN v_artifacts AS a ON a.artifact_id = e.artifact_id
LEFT JOIN
(
    SELECT artifact_id,
           minMap(map(test_type, seen_at, concat(test_type, '=', state), seen_at)) AS first_seen
    FROM v_dispatch_verdicts
    GROUP BY artifact_id
) AS v ON v.artifact_id = e.artifact_id
WHERE ({component:String} = '' OR {component:String} = a.component)
  AND ({arch:String} = '' OR {arch:String} = a.arch)
  AND ({kind:String} = '' OR {kind:String} = a.kind)
  AND ({artifact_name:String} = '' OR {artifact_name:String} = a.artifact_name)
  AND has({event_types:Array(String)}, e.event_type)
  AND (e.event_type != 'artifact.tagged' OR (
          ({tag_family:String} = '' OR {tag_family:String} = e.tag_family)
      AND ({tag_pattern:String} = '' OR match(e.tag, concat('^(?:', {tag_pattern:String}, ')$')))
      AND arrayAll(k -> e.tag_props[k] = {tag_props:Map(String, String)}[k], mapKeys({tag_props:Map(String, String)}))))
  AND (e.event_type != 'results.recorded' OR (
          (empty({result_test_types:Array(String)}) OR has({result_test_types:Array(String)}, e.test_type))
      AND (empty({result_states:Array(String)}) OR has({result_states:Array(String)}, e.state))));

-- The allow-list the dispatcher last published, one row per line; empty when it published none.
CREATE VIEW IF NOT EXISTS v_dispatch_targets AS
SELECT target_type, target_prefix, credential_id, published_at, job_run
FROM dispatch_targets
WHERE published_at = (SELECT max(published_at) FROM dispatch_targets)
  AND NOT (target_type = '' AND target_prefix = '');
