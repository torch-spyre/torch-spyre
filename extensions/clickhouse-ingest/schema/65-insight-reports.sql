-- Saved Ask Spyre Insights reports: the question, the chart specs and a layout template, never data
-- (opening a report re-runs its queries and re-validates every spec). Append-only, one row per event
-- on a report, like result_uploads and ci_requests: a save, an edit (title, template, visibility) or
-- a delete tombstone. Written by the dashboard with the upload writer's login; v_insight_reports
-- reads the latest event per report.
CREATE TABLE IF NOT EXISTS insight_reports
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    report_id       UUID,                          -- uuid7, minted by the dashboard
    state           LowCardinality(String),        -- saved | deleted
    actor           String,                        -- the verified login that wrote this event
    visibility      LowCardinality(String) DEFAULT 'private',
    title           String DEFAULT '',
    question        String DEFAULT '',
    -- A layout id the dashboard defines (headline_chart, side_by_side, ranked_table); not a CHECK,
    -- so a new layout needs no ALTER.
    template        LowCardinality(String) DEFAULT '',
    charts          String DEFAULT '',             -- JSON [{tool, arguments, spec}]: a read-only tool call and the chart spec over its rows
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_state      CHECK state IN ('saved', 'deleted'),
    CONSTRAINT chk_visibility CHECK visibility IN ('private', 'shared'),
    CONSTRAINT chk_report     CHECK report_id != toUUID('00000000-0000-0000-0000-000000000000')
)
ENGINE = MergeTree()
ORDER BY (report_id, ts);

-- One row per report at its latest state. The owner is whoever made its first event; content comes
-- from the latest 'saved' event so a delete tombstone does not blank what it deleted.
CREATE VIEW IF NOT EXISTS v_insight_reports AS
SELECT
    r.report_id                                    AS report_id,
    argMax(r.state, r.ts)                          AS state,
    argMin(r.actor, r.ts)                          AS owner,
    argMaxIf(r.visibility, r.ts, r.state = 'saved') AS visibility,
    argMaxIf(r.title, r.ts, r.state = 'saved')      AS title,
    argMaxIf(r.question, r.ts, r.state = 'saved')   AS question,
    argMaxIf(r.template, r.ts, r.state = 'saved')   AS template,
    argMaxIf(r.charts, r.ts, r.state = 'saved')     AS charts,
    min(r.ts)                                      AS created_at,
    max(r.ts)                                      AS updated_at
FROM insight_reports AS r  -- qualified: the aliases above would shadow the columns
GROUP BY r.report_id;
