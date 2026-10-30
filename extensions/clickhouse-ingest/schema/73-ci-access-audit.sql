-- Append-only audit of the dashboard letting someone other than the requester see or cancel a CI
-- request (jobs:view-any, jobs:cancel-any), including the refusals. One row per attempt.
CREATE TABLE IF NOT EXISTS ci_access_audit
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    event_id        UUID DEFAULT generateUUIDv7(),
    actor           String,                        -- the verified login
    actor_kind      LowCardinality(String) DEFAULT 'human',
    on_behalf_of    String DEFAULT '',             -- the grantor when an agent acted under a grant
    action          LowCardinality(String),        -- view | list | cancel
    permission      LowCardinality(String) DEFAULT '',  -- jobs:view-any | jobs:cancel-any
    target          String DEFAULT '',             -- the login that made the request
    ci_request_id   UUID,                          -- the ci_requests row acted on
    result          LowCardinality(String),        -- ok | refused | error
    reason          String DEFAULT '',
    channel         LowCardinality(String) DEFAULT 'ui',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_actor_kind CHECK actor_kind IN ('human', 'anonymous', 'agent', 'service'),
    CONSTRAINT chk_action     CHECK action IN ('view', 'list', 'cancel'),
    CONSTRAINT chk_result     CHECK result IN ('ok', 'refused', 'error'),
    CONSTRAINT chk_channel    CHECK channel IN ('ui', 'chat', 'mcp', 'api'),
    CONSTRAINT chk_request    CHECK ci_request_id != toUUID('00000000-0000-0000-0000-000000000000')
)
ENGINE = MergeTree()
ORDER BY (ci_request_id, ts)
TTL toDateTime(ts) + INTERVAL 400 DAY;
