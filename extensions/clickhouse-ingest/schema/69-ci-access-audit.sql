-- Append-only audit of the dashboard letting someone other than the requester see or cancel a CI
-- request (jobs:view-any, jobs:cancel-any), including the refusals. One row per attempt.
CREATE TABLE IF NOT EXISTS ci_access_audit
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    event_id        UUID DEFAULT generateUUIDv7(),
    action          LowCardinality(String),        -- view | list | cancel
    request_id      UUID,                          -- the ci_requests row acted on
    actor           String,                        -- the verified login
    on_behalf_of    String DEFAULT '',             -- the grantor when an agent acted under a grant
    owner           String DEFAULT '',             -- who made the target request
    result          LowCardinality(String),        -- authorized | denied | cancelled | failed
    permission      LowCardinality(String) DEFAULT '',  -- jobs:view-any | jobs:cancel-any
    channel         LowCardinality(String) DEFAULT 'ui',
    reason          String DEFAULT '',
    build_url       String DEFAULT '',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_action  CHECK action IN ('view', 'list', 'cancel'),
    CONSTRAINT chk_result  CHECK result IN ('authorized', 'denied', 'cancelled', 'failed'),
    CONSTRAINT chk_channel CHECK channel IN ('ui', 'chat', 'mcp', 'api'),
    CONSTRAINT chk_request CHECK request_id != toUUID('00000000-0000-0000-0000-000000000000')
)
ENGINE = MergeTree()
ORDER BY (request_id, ts)
TTL toDateTime(ts) + INTERVAL 400 DAY;
