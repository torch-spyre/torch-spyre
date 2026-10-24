-- Service-level views over CI runs, ingest freshness and incident response. Views only: nothing
-- here stores a target, so changing one is a change to v_slo_targets in a reviewed schema PR.
-- Every target below is a PROPOSAL read off the 14 days measured when it was written, not a
-- commitment. Percentiles use the OrNull combinator: quantileIf over no rows is NaN, which a
-- page cannot tell from a number.

-- slo = 'verdict'   subject = lane;     share of runs reaching a verdict within within_min of being scheduled
-- slo = 'main_green' subject = '*';     share of main-push runs that passed outright (within_min unused, 0)
-- slo = 'freshness' subject = source;   newest pipeline_runs row no older than within_min (a snapshot, share 1)
-- slo = 'ack'       subject = severity; share of incidents acknowledged within within_min of opening
-- The ack numbers are the router's placeholder escalation deadlines; no incident has been measured.
CREATE VIEW IF NOT EXISTS v_slo_targets AS
SELECT t.1 AS slo, t.2 AS subject, toUInt32(t.3) AS within_min, toFloat64(t.4) AS target_share
FROM
(
    SELECT arrayJoin([
        ('verdict',    'merge-queue', 120, 0.80),
        ('verdict',    'spyre-test',  120, 0.70),
        ('verdict',    'main-push',   120, 0.95),
        ('main_green', '*',             0, 0.75),
        ('freshness',  'jenkins',      10, 1.0),
        ('freshness',  'gha',          30, 1.0),
        ('ack',        'critical',     15, 0.90),
        ('ack',        'warning',     120, 0.90)
    ]) AS t
);

-- Time to a verdict per day and lane, from the moment Jenkins scheduled the run to its end. That is
-- the one span every lane has: /spyre-test runs started by the poller carry no comment time, so the
-- developer-facing span (comment to end) is reported beside it and is NULL when unknown. Superseded
-- and aborted runs are counted apart and left out of every figure. A run with no end yet is
-- `unfinished`, and is not in the percentiles.
CREATE VIEW IF NOT EXISTS v_slo_pr_verdict_daily AS
SELECT
    toDate(r.run_started_at)                                                         AS day,
    r.trigger_source                                                                 AS lane,
    countIf(r.outcome != 'superseded')                                               AS runs,
    countIf(r.outcome = 'superseded')                                                AS superseded_runs,
    countIf(r.outcome != 'superseded' AND r.scheduled_to_end_ms IS NULL)             AS unfinished,
    round(quantileIfOrNull(0.5)(r.scheduled_to_end_ms, r.outcome != 'superseded') / 60000, 1)  AS p50_min,
    round(quantileIfOrNull(0.9)(r.scheduled_to_end_ms, r.outcome != 'superseded') / 60000, 1)  AS p90_min,
    round(quantileIfOrNull(0.99)(r.scheduled_to_end_ms, r.outcome != 'superseded') / 60000, 1) AS p99_min,
    round(quantileIfOrNull(0.5)(r.comment_to_end_ms, r.outcome != 'superseded') / 60000, 1)    AS developer_wait_p50_min,
    round(quantileIfOrNull(0.9)(r.comment_to_end_ms, r.outcome != 'superseded') / 60000, 1)    AS developer_wait_p90_min,
    any(t.within_min)                                                                AS target_min,
    any(t.target_share)                                                              AS target_share,
    countIf(r.outcome != 'superseded' AND r.scheduled_to_end_ms <= t.within_min * 60000) AS within_target,
    round(within_target / nullIf(runs - unfinished, 0), 3)                           AS share_within_target
FROM v_ci_runs AS r
INNER JOIN
(
    SELECT subject AS slo_lane, within_min, target_share FROM v_slo_targets WHERE slo = 'verdict'
) AS t ON r.trigger_source = t.slo_lane
GROUP BY day, lane;

-- Main-push runs that passed outright, per day and repo. `warned` (a yellow verdict) is its own
-- count, so green_share and not_failed_share tell the two apart. merge_to_green is NULL until a run
-- has passed and has a merge time.
CREATE VIEW IF NOT EXISTS v_slo_main_green_daily AS
SELECT
    toDate(run_started_at)                                                           AS day,
    repo,
    countIf(outcome != 'superseded')                                                 AS runs,
    countIf(outcome = 'passed')                                                      AS passed,
    countIf(outcome = 'warned')                                                      AS warned,
    countIf(outcome = 'failed')                                                      AS failed,
    countIf(outcome = 'superseded')                                                  AS superseded_runs,
    round(passed / nullIf(passed + warned + failed, 0), 3)                           AS green_share,
    round((passed + warned) / nullIf(passed + warned + failed, 0), 3)                AS not_failed_share,
    round(quantileIfOrNull(0.5)(merge_to_green_ms, merge_to_green_ms IS NOT NULL) / 60000, 1) AS merge_to_green_p50_min,
    round(quantileIfOrNull(0.9)(merge_to_green_ms, merge_to_green_ms IS NOT NULL) / 60000, 1) AS merge_to_green_p90_min,
    (SELECT target_share FROM v_slo_targets WHERE slo = 'main_green' AND subject = '*') AS target_share
FROM v_ci_runs_main_push
GROUP BY day, repo;

-- How stale each ingest source is, now. A source with no row in the last 7 days still appears,
-- with NULL minutes and within_limit 0, so silence is visible instead of absent.
CREATE VIEW IF NOT EXISTS v_slo_ingest_freshness AS
SELECT
    t.slo_source                                                                     AS source,
    t.within_min                                                                     AS limit_min,
    s.rows_7d                                                                        AS rows_7d,
    if(s.rows_7d = 0, NULL, s.newest_at)                                             AS newest_at,
    if(s.rows_7d = 0, NULL, round(dateDiff('second', s.newest_at, now()) / 60, 1))   AS minutes_stale,
    s.rows_7d > 0 AND dateDiff('second', s.newest_at, now()) <= t.within_min * 60    AS within_limit
FROM
(
    SELECT subject AS slo_source, within_min FROM v_slo_targets WHERE slo = 'freshness'
) AS t
LEFT JOIN
(
    SELECT source AS seen_source, max(updated_at) AS newest_at, count() AS rows_7d
    FROM pipeline_runs
    WHERE updated_at >= now() - INTERVAL 7 DAY
    GROUP BY seen_source
) AS s ON s.seen_source = t.slo_source;

-- Incident response per day and cause. Medians are minutes and NULL while no incident has reached
-- that step. The ack share counts only critical and warning incidents, the severities with a
-- target; info incidents are in `opened` and nowhere else.
CREATE VIEW IF NOT EXISTS v_slo_incident_response AS
SELECT
    toDate(i.opened_at)                                                              AS day,
    i.cause_class                                                                    AS cause_class,
    count()                                                                          AS opened,
    countIf(i.severity = 'critical')                                                 AS critical,
    countIf(i.first_ack_at IS NOT NULL)                                              AS acknowledged,
    countIf(i.first_stop_at IS NOT NULL)                                             AS stopped,
    round(quantileIfOrNull(0.5)(i.seconds_to_ack, i.seconds_to_ack IS NOT NULL) / 60, 1)   AS ack_p50_min,
    round(quantileIfOrNull(0.5)(i.seconds_to_stop, i.seconds_to_stop IS NOT NULL) / 60, 1) AS stop_p50_min,
    countIf(t.within_min > 0)                                                        AS with_target,
    countIf(t.within_min > 0 AND i.seconds_to_ack IS NOT NULL
            AND i.seconds_to_ack <= t.within_min * 60)                               AS acked_within_target,
    round(acked_within_target / nullIf(with_target, 0), 3)                           AS ack_share
FROM v_incidents AS i
LEFT JOIN
(
    SELECT subject AS slo_severity, within_min FROM v_slo_targets WHERE slo = 'ack'
) AS t ON i.severity = t.slo_severity
GROUP BY day, cause_class;

-- Burn of each verdict target over the last 7 and 28 days. allowed_bad is the runs the target lets
-- miss; burn_rate above 1 means the budget is spent. A run is bad when its verdict took longer than
-- the target, or it has no end and has already run past it. covered_days is how far back the data
-- actually reaches, because a 28-day window over 14 days of history is not a 28-day figure.
CREATE VIEW IF NOT EXISTS v_slo_error_budget AS
SELECT
    t.slo_lane                                                                       AS lane,
    w.window_days                                                                    AS window_days,
    dateDiff('day', min(w.run_started_at), now())                                    AS covered_days,
    count()                                                                          AS runs,
    countIf(w.span_ms > t.within_min * 60000
            OR (w.span_ms IS NULL AND w.run_started_at < now() - toIntervalMinute(t.within_min))) AS bad,
    any(t.target_share)                                                              AS target_share,
    round(runs * (1 - target_share), 1)                                              AS allowed_bad,
    round(bad / nullIf(allowed_bad, 0), 2)                                           AS burn_rate,
    round(1 - bad / nullIf(runs, 0), 3)                                              AS share_good
FROM
(
    SELECT subject AS slo_lane, within_min, target_share FROM v_slo_targets WHERE slo = 'verdict'
) AS t
INNER JOIN
(
    SELECT trigger_source AS run_lane, run_started_at, scheduled_to_end_ms AS span_ms, window_days
    FROM v_ci_runs
    ARRAY JOIN [7, 28] AS window_days
    WHERE outcome != 'superseded' AND run_started_at >= now() - toIntervalDay(window_days)
) AS w ON w.run_lane = t.slo_lane
GROUP BY lane, window_days;
