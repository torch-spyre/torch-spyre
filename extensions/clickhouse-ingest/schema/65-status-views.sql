-- Component state computed from probe samples. The state is a rule over samples and is never
-- written by a person or a model; annotations sit beside it. Constants, all proposals:
--   bucket 5 minutes; a vantage is alive when its newest heartbeat is under 15 minutes old;
--   a vantage's sample is current when it is in the last 15 minutes; "consecutive" means buckets
--   exactly 5 minutes apart.
-- Quorum over the live vantages that gave a known sample (R reporting, F failing):
--   R = 0                  unknown
--   F = 0                  operational (degraded if any answered slowly)
--   R = 1, F = 1           degraded, labelled single_vantage; partial_outage after 3 consecutive buckets
--   R = 2, F = 1           degraded
--   R = 2, F = 2           partial_outage; major_outage once all failed in 2 consecutive buckets
--   R >= 3, F < R/2        degraded
--   R >= 3, R/2 <= F < R   partial_outage
--   R >= 3, F = R          partial_outage; major_outage once all failed in 2 consecutive buckets
-- A 'degraded' sample (slow answer) counts as working for the quorum. An 'unknown' sample is not a
-- report. Unknown is never shown as operational.

-- Newest sample per component, vantage and bucket. Aliases are cur_*: an alias equal to a source
-- column is substituted into the other aggregates (ILLEGAL_AGGREGATION). The heartbeat window is
-- bounded so the live path does not read the whole history.
CREATE VIEW IF NOT EXISTS v_status_bucket AS
WITH beats AS
(
    SELECT vantage, max(ts) AS last_beat
    FROM status_samples FINAL
    WHERE probe = 'heartbeat' AND ts >= now64(3) - INTERVAL 2 DAY
    GROUP BY vantage
)
SELECT
    s.component AS component,
    s.bucket    AS bucket,
    s.vantage   AS vantage,
    argMax(s.state, s.ts)        AS cur_state,
    argMax(s.latency_ms, s.ts)   AS cur_latency_ms,
    argMax(s.error_class, s.ts)  AS cur_error_class,
    max(s.ts)                    AS cur_ts,
    max(b.last_beat)             AS beat_ts,
    max(b.last_beat) >= now64(3) - INTERVAL 15 MINUTE AS vantage_alive
FROM status_samples AS s FINAL
LEFT JOIN beats AS b ON b.vantage = s.vantage
WHERE s.probe != 'heartbeat' AND s.component != '*'
GROUP BY component, bucket, vantage;

-- The state now: the quorum over each live vantage's newest sample of the last 15 minutes, with the
-- consecutive-bucket debounce read from the last four buckets.
CREATE VIEW IF NOT EXISTS v_status_current AS
WITH newest AS
(
    SELECT
        component, vantage,
        argMax(cur_state, bucket)     AS v_state,
        argMax(cur_ts, bucket)        AS v_ts,
        argMax(vantage_alive, bucket) AS v_alive
    FROM v_status_bucket
    WHERE bucket >= toStartOfFiveMinutes(now()) - INTERVAL 15 MINUTE
    GROUP BY component, vantage
),
per_bucket AS
(
    SELECT
        component, bucket,
        intDiv(toUnixTimestamp(toStartOfFiveMinutes(now())) - toUnixTimestamp(bucket), 300) AS back,
        countIf(cur_state IN ('ok', 'degraded', 'failing') AND vantage_alive) AS n_known,
        countIf(cur_state = 'failing' AND vantage_alive)                      AS n_failing
    FROM v_status_bucket
    WHERE bucket >= toStartOfFiveMinutes(now()) - INTERVAL 15 MINUTE
    GROUP BY component, bucket
),
runs AS
(
    SELECT
        component,
        (all0 AND all1) OR (all1 AND all2)                 AS all_fail_pair,
        (one0 AND one1 AND one2) OR (one1 AND one2 AND one3) AS one_fail_triple
    FROM
    (
        SELECT
            component,
            maxIf(n_known >= 2 AND n_failing = n_known, back = 0) AS all0,
            maxIf(n_known >= 2 AND n_failing = n_known, back = 1) AS all1,
            maxIf(n_known >= 2 AND n_failing = n_known, back = 2) AS all2,
            maxIf(n_known = 1 AND n_failing = 1, back = 0)        AS one0,
            maxIf(n_known = 1 AND n_failing = 1, back = 1)        AS one1,
            maxIf(n_known = 1 AND n_failing = 1, back = 2)        AS one2,
            maxIf(n_known = 1 AND n_failing = 1, back = 3)        AS one3
        FROM per_bucket
        GROUP BY component
    )
),
agg AS
(
    SELECT
        component,
        countIf(v_state IN ('ok', 'degraded', 'failing') AND v_alive)              AS reporting,
        countIf(v_state = 'failing' AND v_alive)                                   AS failing,
        countIf(v_state = 'degraded' AND v_alive)                                  AS slow,
        groupArrayIf(vantage, v_state = 'failing' AND v_alive)                     AS failing_from,
        groupArrayIf(vantage, v_state IN ('ok', 'degraded', 'failing') AND v_alive) AS reported_by,
        countIf(NOT v_alive)                                                       AS prober_offline,
        max(v_ts)                                                                  AS last_sample_at
    FROM newest
    GROUP BY component
)
SELECT
    a.component AS component,
    multiIf(
        a.reporting = 0,                                   'unknown',
        a.failing = 0 AND a.slow = 0,                      'operational',
        a.failing = 0,                                     'degraded',
        a.reporting = 1,                                   if(r.one_fail_triple, 'partial_outage', 'degraded'),
        a.reporting = 2 AND a.failing = 1,                 'degraded',
        a.failing * 2 < a.reporting,                       'degraded',
        a.failing = a.reporting,                           if(r.all_fail_pair, 'major_outage', 'partial_outage'),
                                                           'partial_outage'
    ) AS state,
    a.reporting AS vantages_reporting,
    a.reported_by AS reported_by,
    a.failing_from AS failing_from,
    a.prober_offline AS vantages_offline,
    a.reporting = 0 AS no_data,
    a.reporting = 1 AS single_vantage,
    if(a.last_sample_at = toDateTime64(0, 3, 'UTC'), NULL, a.last_sample_at) AS last_sample_at
FROM agg AS a
LEFT JOIN runs AS r ON r.component = a.component;

-- Per component and day: buckets with a known quorum that were working, were down, or were unknown,
-- and the share working. A bucket is down by the same rule as v_status_current: two reporting
-- vantages both failing, three or more with at least half failing, or one vantage failing in the
-- third consecutive bucket onward. A split of two, or one vantage failing for less than that, is
-- degraded (counted in single_vantage_fail_buckets when it is one vantage). A bucket with no known
-- sample from any vantage is unknown and is left out of the share; the share is NULL when no bucket
-- in the day was known. A degraded bucket counts as working and is also counted on its own. The
-- run of single-vantage failures is a RANGE window over the bucket time, so a gap breaks it.
CREATE VIEW IF NOT EXISTS v_status_daily AS
WITH per_bucket AS
(
    SELECT
        component, bucket,
        countIf(cur_state IN ('ok', 'degraded', 'failing')) AS n_known,
        countIf(cur_state = 'failing')                      AS n_fail,
        countIf(cur_state = 'degraded')                     AS n_slow
    FROM v_status_bucket
    GROUP BY component, bucket
),
flagged AS
(
    SELECT
        component, bucket, n_known, n_fail, n_slow,
        n_known = 1 AND n_fail = 1 AS one_fail,
        sum(one_fail) OVER (PARTITION BY component ORDER BY toUnixTimestamp(bucket)
                            RANGE BETWEEN 600 PRECEDING AND CURRENT ROW) AS one_fail_run
    FROM per_bucket
),
judged AS
(
    SELECT
        component, bucket, n_known, n_fail, n_slow, one_fail,
        (n_known = 2 AND n_fail = 2) OR (n_known >= 3 AND n_fail * 2 >= n_known)
            OR (one_fail AND one_fail_run = 3) AS is_down
    FROM flagged
)
SELECT
    component,
    toDate(bucket)                                          AS day,
    countIf(n_known > 0)                                    AS known_buckets,
    countIf(n_known = 0)                                    AS unknown_buckets,
    countIf(is_down)                                        AS down_buckets,
    countIf(one_fail)                                       AS single_vantage_fail_buckets,
    countIf(n_known > 0 AND NOT is_down AND (n_fail > 0 OR n_slow > 0)) AS degraded_buckets,
    known_buckets - down_buckets                            AS working_buckets,
    if(known_buckets = 0, NULL, round(working_buckets / known_buckets, 4)) AS uptime_share
FROM judged
GROUP BY component, day;
