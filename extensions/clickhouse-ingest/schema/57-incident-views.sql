-- Incidents with their current state and response times, for the maintenance page and Ask Spyre.

-- Aggregate aliases are cur_*: an alias equal to a source column is substituted into the other aggregates.
-- Timings are measured from the first log row; an unset milestone is NULL, not zero.
-- `signals` counts events attached to the incident, so one cause behind 40 builds reads as 40.
CREATE VIEW IF NOT EXISTS v_incidents AS
WITH log AS
(
    SELECT
        incident_id,
        min(updated_at)                                  AS first_logged_at,
        max(updated_at)                                  AS last_logged_at,
        minIf(updated_at, state = 'acknowledged')        AS first_ack_at,
        minIf(updated_at, state IN ('mitigated', 'resolved', 'dismissed')) AS first_stop_at,
        argMax(state, updated_at)                        AS cur_state,
        argMax(severity, updated_at)                     AS cur_severity,
        argMax(title, updated_at)                        AS cur_title,
        argMax(cause_class, updated_at)                  AS cur_cause_class,
        argMax(signature, updated_at)                    AS cur_signature,
        argMax(owner, updated_at)                        AS cur_owner,
        argMax(owner_email, updated_at)                  AS cur_owner_email,
        argMax(thread_url, updated_at)                   AS cur_thread_url,
        argMax(evidence_url, updated_at)                 AS cur_evidence_url,
        argMax(updated_by, updated_at)                   AS cur_updated_by,
        count()                                          AS log_rows
    FROM incident_log
    GROUP BY incident_id
),
events AS
(
    SELECT incident_id, count() AS signals, min(ts) AS first_signal_at,
           groupUniqArray(source) AS sources, groupUniqArray(subject) AS subjects
    FROM incident_events
    WHERE incident_id != ''
    GROUP BY incident_id
)
SELECT
    l.incident_id AS incident_id, l.cur_state AS state, l.cur_severity AS severity, l.cur_title AS title,
    l.cur_cause_class AS cause_class, l.cur_signature AS signature,
    l.cur_owner AS owner, l.cur_owner_email AS owner_email,
    l.cur_thread_url AS thread_url, l.cur_evidence_url AS evidence_url, l.cur_updated_by AS updated_by,
    l.first_logged_at AS opened_at, l.last_logged_at AS updated_at, l.log_rows AS log_rows,
    e.signals AS signals, e.sources AS sources, e.subjects AS subjects,
    if(e.first_signal_at = toDateTime64(0, 3, 'UTC'), NULL, e.first_signal_at) AS first_signal_at,
    if(l.first_ack_at = toDateTime64(0, 3, 'UTC'), NULL, l.first_ack_at) AS first_ack_at,
    if(l.first_stop_at = toDateTime64(0, 3, 'UTC'), NULL, l.first_stop_at) AS first_stop_at,
    if(l.first_ack_at = toDateTime64(0, 3, 'UTC'), NULL, dateDiff('second', l.first_logged_at, l.first_ack_at)) AS seconds_to_ack,
    if(l.first_stop_at = toDateTime64(0, 3, 'UTC'), NULL, dateDiff('second', l.first_logged_at, l.first_stop_at)) AS seconds_to_stop,
    l.cur_state IN ('open', 'acknowledged')              AS is_active
FROM log AS l
LEFT JOIN events AS e USING (incident_id);
