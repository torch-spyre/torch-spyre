-- External events as one row per event, however many vantages saw it. Feed text is untrusted and is
-- shown, not interpreted; the only fields that drive anything are kind, status, impact, starts_at,
-- ends_at and our own `component`. An event whose end time has passed is not open, whatever its
-- last reported status: a feed that never marks a window completed must not keep it on the page.

-- One row per vantage first, then per event. Aggregates are named v_* and cur_* because an alias
-- equal to a source column is substituted into the other aggregates (ILLEGAL_AGGREGATION), and an
-- unset time from argMax over NULLs stays NULL. `distinct_versions` > 1 means the vantages saw
-- different text for the same event (an edit, or a bad fetch); the page then says they disagree
-- and does not silently show the newest.
CREATE VIEW IF NOT EXISTS v_external_events_current AS
WITH per_vantage AS
(
    SELECT
        event_id, vantage,
        min(observed_at)                  AS v_first_seen,
        max(observed_at)                  AS v_last_seen,
        argMax(content_hash, observed_at) AS v_hash,
        argMax(status, observed_at)       AS v_status,
        argMax(impact, observed_at)       AS v_impact,
        argMax(kind, observed_at)         AS v_kind,
        argMax(component, observed_at)    AS v_component,
        argMax(source, observed_at)       AS v_source,
        argMax(starts_at, observed_at)    AS v_starts,
        argMax(ends_at, observed_at)      AS v_ends,
        argMax(title, observed_at)        AS v_title,
        argMax(url, observed_at)          AS v_url
    FROM external_events
    GROUP BY event_id, vantage
)
SELECT
    event_id,
    groupArray(vantage)                 AS vantages_seen,
    count()                             AS vantage_count,
    min(v_first_seen)                   AS first_seen,
    max(v_last_seen)                    AS last_seen,
    uniqExact(v_hash)                   AS distinct_versions,
    argMax(v_source, v_last_seen)       AS cur_source,
    argMax(v_component, v_last_seen)    AS cur_component,
    argMax(v_kind, v_last_seen)         AS cur_kind,
    argMax(v_status, v_last_seen)       AS cur_status,
    argMax(v_impact, v_last_seen)       AS cur_impact,
    argMax(v_starts, v_last_seen)       AS cur_starts,
    argMax(v_ends, v_last_seen)         AS cur_ends,
    argMax(v_title, v_last_seen)        AS cur_title,
    argMax(v_url, v_last_seen)          AS cur_url,
    distinct_versions > 1               AS vantages_disagree,
    cur_kind = 'maintenance' AND cur_status = 'scheduled' AND cur_starts IS NOT NULL AND cur_starts > now64(3) AS is_upcoming_maintenance,
    cur_status NOT IN ('resolved', 'completed', 'cancelled') AND (cur_ends IS NULL OR cur_ends > now64(3)) AS is_open
FROM per_vantage
GROUP BY event_id;

-- How each feed is doing from each vantage: the newest fetch and whether the last good one is recent.
-- A source with no 'ok' fetch from any vantage in the stale window is `unreachable`, not "no incidents".
CREATE VIEW IF NOT EXISTS v_external_feed_health AS
SELECT
    source, vantage,
    max(fetched_at)                                              AS last_fetch_at,
    argMax(outcome, fetched_at)                                  AS cur_outcome,
    argMax(failed_phase, fetched_at)                             AS cur_failed_phase,
    maxIf(fetched_at, outcome IN ('ok', 'not_modified'))         AS last_good_raw,
    if(last_good_raw = toDateTime64(0, 3, 'UTC'), NULL, last_good_raw) AS last_good_at,
    countIf(outcome NOT IN ('ok', 'not_modified'))               AS failures_24h,
    count()                                                      AS fetches_24h
FROM external_feed_fetches
WHERE fetched_at >= now64(3) - INTERVAL 24 HOUR
GROUP BY source, vantage;
