-- Who did what through the dashboard's control routes (live-cluster actions, Experience, cache flush).
-- Append-only, one row per attempt, refused ones included, like result_uploads and ci_requests. Written
-- by the dashboard with the upload writer's login; parameters are sanitized by the writer and never
-- hold secrets.
CREATE TABLE IF NOT EXISTS ops_audit_events
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    event_id        UUID,                          -- uuid7, minted by the dashboard
    request_id      String DEFAULT '',             -- the request's X-Request-Id, or one minted per request
    actor           String DEFAULT '',             -- verified login; empty when the attempt had none
    actor_kind      LowCardinality(String) DEFAULT 'anonymous',
    route           String,                        -- the URL rule, e.g. /api/live-cluster/nodes/<node>/cordon
    method          LowCardinality(String),
    action          LowCardinality(String),        -- cluster.cordon-node, experience.start, cache.invalidate, ...
    permission      LowCardinality(String) DEFAULT '',
    target          String DEFAULT '',             -- node, node/card, model key, cache prefix
    params          String DEFAULT '',             -- sanitized JSON of the request's parameters
    result          LowCardinality(String),        -- ok | refused | error
    status          UInt16 DEFAULT 0,              -- HTTP status returned
    reason          String DEFAULT '',             -- why refused or failed
    source_ip       String DEFAULT '',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_actor_kind CHECK actor_kind IN ('human', 'anonymous', 'agent', 'service'),
    CONSTRAINT chk_result CHECK result IN ('ok', 'refused', 'error'),
    CONSTRAINT chk_event  CHECK event_id != toUUID('00000000-0000-0000-0000-000000000000')
)
ENGINE = MergeTree()
ORDER BY (event_id, ts);

-- One row per event (an event is never rewritten), in the shape the other event stores' views have.
CREATE VIEW IF NOT EXISTS v_ops_audit_events AS
SELECT
    e.event_id                      AS event_id,
    argMax(e.ts, e.audit_timestamp) AS ts,
    argMax(e.request_id, e.audit_timestamp) AS request_id,
    argMax(e.actor, e.audit_timestamp)      AS actor,
    argMax(e.actor_kind, e.audit_timestamp) AS actor_kind,
    argMax(e.route, e.audit_timestamp)      AS route,
    argMax(e.method, e.audit_timestamp)     AS method,
    argMax(e.action, e.audit_timestamp)     AS action,
    argMax(e.permission, e.audit_timestamp) AS permission,
    argMax(e.target, e.audit_timestamp)     AS target,
    argMax(e.params, e.audit_timestamp)     AS params,
    argMax(e.result, e.audit_timestamp)     AS result,
    argMax(e.status, e.audit_timestamp)     AS status,
    argMax(e.reason, e.audit_timestamp)     AS reason,
    argMax(e.source_ip, e.audit_timestamp)  AS source_ip,
    argMax(e.props, e.audit_timestamp)      AS props
FROM ops_audit_events AS e  -- qualified: the aliases above would shadow the columns
GROUP BY e.event_id;
