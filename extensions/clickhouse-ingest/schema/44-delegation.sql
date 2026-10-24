-- Delegation: a person lets an agent act for them within a short, bounded grant. Infrastructure
-- telemetry, not an artifact fact. Ids, tool names and hashes only: no argument value, secret or
-- raw log lands here.

-- One row per grant, written once when the person confirms the consent text. Never updated: a
-- grant ends through delegation_grant_events (revoked) or by its expires_at.
CREATE TABLE IF NOT EXISTS delegation_grants
(
    grant_id          String,
    grantor           String,                    -- the person whose rights the grant borrows
    grantee           String,                    -- the agent or service allowed to use it
    tools             Array(String),             -- allowlist of gateway tool names
    scope_json        String DEFAULT '{}',       -- optional narrowing: repos, jobs, nodes
    expires_at        DateTime64(3, 'UTC'),
    max_uses          UInt32 DEFAULT 0,          -- 0 = no limit inside the lifetime
    condition         String DEFAULT '',         -- e.g. only incidents the grantor owns
    created_at        DateTime64(3, 'UTC'),
    created_by        String,                    -- the logged-in person; never the grantee
    consent_text_hash String,                    -- sha256 of the exact text the person confirmed

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_grant_lifetime CHECK expires_at > created_at,
    CONSTRAINT chk_grant_parties CHECK grantor != grantee,
    CONSTRAINT chk_grant_hash CHECK length(consent_text_hash) = 64
)
ENGINE = MergeTree()
ORDER BY (grant_id, created_at);

-- Everything that happens under a grant, one row each: creation, every use, every refusal, revocation.
-- Append-only, so it is the audit trail; uses and revocation are counted from it, never stored.
CREATE TABLE IF NOT EXISTS delegation_grant_events
(
    grant_id     String,
    ts           DateTime64(3, 'UTC') DEFAULT now64(3),
    event        LowCardinality(String),   -- created | used | denied | revoked | expired
    actor        String,                   -- who did it: the grantee for a use, the person for a revoke
    tool         LowCardinality(String) DEFAULT '',
    args_hash    String DEFAULT '',        -- hash of the arguments, never the arguments
    result       LowCardinality(String),   -- ok | denied | error
    on_behalf_of String DEFAULT '',        -- the grantor, on a use made under the grant

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_grant_event CHECK event IN ('created', 'used', 'denied', 'revoked', 'expired'),
    CONSTRAINT chk_grant_result CHECK result IN ('ok', 'denied', 'error')
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (grant_id, ts)
TTL toDateTime(ts) + INTERVAL 365 DAY;
