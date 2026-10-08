-- Subscription matching, the one definition the artifact-dispatch job and the dashboard both
-- read. Apply after 25-artifact-dispatch.sql and 50-artifact-views.sql (v_artifacts).

-- Every (artifact, subscription) pair whose artifact filters match, enabled or not: what a
-- request may run, and the base the tag filters below narrow.
CREATE VIEW IF NOT EXISTS v_subscription_artifacts AS
SELECT
    a.artifact_id      AS artifact_id,
    a.component        AS component,
    a.arch             AS arch,
    a.kind             AS kind,
    a.artifact_name    AS artifact_name,
    s.subscription_id  AS subscription_id,
    s.enabled          AS enabled,
    s.tag_family       AS sub_tag_family,
    s.tag_pattern      AS sub_tag_pattern,
    s.tag_props        AS sub_tag_props,
    s.not_before       AS not_before,
    s.target_type      AS target_type,
    s.target           AS target,
    s.params           AS params,
    s.credential_id    AS credential_id,
    s.owner            AS owner
FROM v_artifacts AS a
CROSS JOIN (SELECT * FROM artifact_subscriptions FINAL) AS s
WHERE (s.component = '' OR s.component = a.component)
  AND (s.arch = '' OR s.arch = a.arch)
  AND (s.kind = '' OR s.kind = a.kind)
  AND (s.artifact_name = '' OR s.artifact_name = a.artifact_name);

-- Who a tag triggers: one row per (tag, artifact, subscription) match, with the automatic
-- dispatch it got. tag_ts is the FIRST time the tag pointed at the artifact: main/nightly/weekly/pr
-- writers re-insert a tag on every reuse (one `main` pair holds 875 rows), which is not a new promotion.
-- auto = this match fires on its own; dispatch_state '' = it has not been dispatched yet.
CREATE VIEW IF NOT EXISTS v_artifact_subscribers AS
SELECT
    m.tag              AS tag,
    m.tag_family       AS tag_family,
    m.artifact_id      AS artifact_id,
    m.tag_ts           AS tag_ts,
    m.component        AS component,
    m.arch             AS arch,
    m.kind             AS kind,
    m.artifact_name    AS artifact_name,
    m.subscription_id  AS subscription_id,
    m.enabled          AS enabled,
    m.enabled AND m.tag_ts >= m.not_before AS auto,
    m.target_type      AS target_type,
    m.target           AS target,
    m.params           AS params,
    m.credential_id    AS credential_id,
    m.owner            AS owner,
    -- ifNull: an unjoined row is NULL under a reader's join_use_nulls=1.
    ifNull(d.state, '')      AS dispatch_state,
    ifNull(d.build_url, '')  AS build_url,
    ifNull(d.error, '')      AS dispatch_error,
    d.updated_at             AS dispatched_at
FROM
(
    SELECT t.tag AS tag, t.tag_family AS tag_family, t.tag_ts AS tag_ts,
           sa.artifact_id AS artifact_id, sa.component AS component, sa.arch AS arch,
           sa.kind AS kind, sa.artifact_name AS artifact_name,
           sa.subscription_id AS subscription_id, sa.enabled AS enabled,
           sa.not_before AS not_before, sa.target_type AS target_type, sa.target AS target,
           sa.params AS params, sa.credential_id AS credential_id, sa.owner AS owner
    FROM
    (
        SELECT tag, artifact_id, any(tag_family) AS tag_family, min(ts) AS tag_ts,
               argMin(props, ts) AS tag_props
        FROM artifact_tags
        GROUP BY tag, artifact_id
    ) AS t
    INNER JOIN v_subscription_artifacts AS sa ON sa.artifact_id = t.artifact_id
    WHERE (sa.sub_tag_family = '' OR sa.sub_tag_family = t.tag_family)
      AND (sa.sub_tag_pattern = '' OR match(t.tag, concat('^(?:', sa.sub_tag_pattern, ')$')))
      AND arrayAll(k -> t.tag_props[k] = sa.sub_tag_props[k], mapKeys(sa.sub_tag_props))
) AS m
LEFT JOIN
(
    SELECT subscription_id, artifact_id, tag, state, build_url, error, updated_at
    FROM artifact_dispatches FINAL
    WHERE requested_by = 'auto'
) AS d ON d.subscription_id = m.subscription_id AND d.artifact_id = m.artifact_id AND d.tag = m.tag;
