-- The dashboard's side of launching CI: one row per event on a request (previewed, launched, refused,
-- failed, cancelled). `ops_requests` is the trigger job's own audit; this adds what only the dashboard
-- knows (the plan the person confirmed, the cells it would add or skip, the estimate shown) and joins to it
-- on `job_run`. Append-only, like result_uploads; v_ci_requests (59-) reads the latest event per request.
CREATE TABLE IF NOT EXISTS ci_requests
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    request_id      UUID,                          -- uuid7, one per preview/launch attempt
    plan_id         String,                        -- the previewed plan; a launch must carry the id it was confirmed from
    state           LowCardinality(String),        -- previewed | launched | denied | rejected | failed | cancelled
    actor           String,                        -- the login that caused this event (verified from the token)
    on_behalf_of    String DEFAULT '',             -- set when an agent acted under a delegation grant
    channel         LowCardinality(String),        -- ui | chat | mcp | api
    profile         LowCardinality(String) DEFAULT '',  -- the preset "missing" was measured against
    target_kind     LowCardinality(String) DEFAULT '',  -- pr | main | ref | artifact | tag
    subjects        String DEFAULT '',             -- JSON: the PRs, commit sets, artifacts or tags as the person gave them
    preset          LowCardinality(String) DEFAULT '',
    arches          Array(LowCardinality(String)),
    modes           Array(LowCardinality(String)),  -- the test modes requested; empty = the preset's own
    cells_requested String DEFAULT '',             -- JSON: [{artifact_id, mode, arch}] the launch adds
    cells_skipped   String DEFAULT '',             -- JSON: cells that already hold a result and are not re-run
    cells_blocked   String DEFAULT '',             -- JSON: [{..., reason}] cells that cannot be launched
    estimate        String DEFAULT '',             -- JSON: per-arch p50/p90 minutes and sample sizes as shown
    canonical       String DEFAULT '',             -- the ops/trigger request rebuilt from validated values
    job_run         String DEFAULT '',             -- the trigger build; joins ops_requests.job_run
    build_url       String DEFAULT '',             -- the orchestrator build it launched
    error           String DEFAULT '',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_state   CHECK state IN ('previewed', 'launched', 'denied', 'rejected', 'failed', 'cancelled'),
    CONSTRAINT chk_channel CHECK channel IN ('ui', 'chat', 'mcp', 'api'),
    CONSTRAINT chk_request CHECK request_id != toUUID('00000000-0000-0000-0000-000000000000')
)
ENGINE = MergeTree()
ORDER BY (request_id, ts)
-- Matches ops_requests: who asked for what stays queryable for a release cycle.
TTL toDateTime(ts) + INTERVAL 400 DAY;
