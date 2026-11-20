-- Which model answers each use of the dashboard (chat, summaries, Slack, ...), changed on its Model priority page.
-- Append-only: a change is a new event and the latest per use case wins; an event is never rewritten. Written by the
-- dashboard with the upload writer's login. `before` and `after` are the effective orders as JSON, for the audit trail.
CREATE TABLE IF NOT EXISTS model_priority_events
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    event_id        UUID DEFAULT generateUUIDv7(),
    use_case        LowCardinality(String),        -- default | chat | insights | reports | summaries | slack | analyst | doctor | askhr
    action          LowCardinality(String),        -- set | reset (drop a use case's override)
    model_order     Array(String),                 -- served model ids, first tried first
    disabled        Array(String),
    actor           String,                        -- verified login
    actor_kind      LowCardinality(String) DEFAULT 'human',
    reason          String DEFAULT '',
    before          String DEFAULT '',
    after           String DEFAULT '',
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_use_case   CHECK match(use_case, '^[a-z][a-z0-9_]{1,31}$'),
    CONSTRAINT chk_action     CHECK action IN ('set', 'reset'),
    CONSTRAINT chk_actor      CHECK actor != '',
    CONSTRAINT chk_actor_kind CHECK actor_kind IN ('human', 'anonymous', 'agent', 'service'),
    CONSTRAINT chk_order      CHECK action = 'reset' OR length(model_order) > 0,
    CONSTRAINT chk_event      CHECK event_id != toUUID('00000000-0000-0000-0000-000000000000')
)
ENGINE = MergeTree()
ORDER BY (use_case, ts, event_id);

-- The priority in force per use case: its newest event.
CREATE VIEW IF NOT EXISTS v_model_priority AS
SELECT
    e.use_case                                   AS use_case,
    argMax(e.event_id, (e.ts, e.event_id))       AS event_id,
    max(e.ts)                                    AS ts,
    argMax(e.action, (e.ts, e.event_id))         AS action,
    argMax(e.model_order, (e.ts, e.event_id))    AS model_order,
    argMax(e.disabled, (e.ts, e.event_id))       AS disabled,
    argMax(e.actor, (e.ts, e.event_id))          AS actor,
    argMax(e.reason, (e.ts, e.event_id))         AS reason
FROM model_priority_events AS e  -- qualified: the aliases above would shadow the columns
GROUP BY e.use_case;
