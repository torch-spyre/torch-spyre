-- One row per self-service ClickHouse login, rewritten on every state change; the row with the
-- greatest `version` wins, so read it through FINAL or argMax. Holds no password or digest.
CREATE TABLE IF NOT EXISTS login_requests
(
    login                    String,
    version                  DateTime64(3, 'UTC'), -- writer-supplied, strictly increasing per login
    requester                String,
    owner                    String,
    tier                     UInt8,
    grants                   String,              -- JSON [[privilege, database, table], ...]
    status                   LowCardinality(String), -- expiring/expired are derived from expires_at
    approver                 String DEFAULT '',
    reason                   String DEFAULT '',
    host                     String DEFAULT 'ANY',
    expires_at               Nullable(DateTime64(3, 'UTC')), -- NULL = never
    needs_approval           UInt8,
    password_issued          UInt8,
    last_rotated_at          Nullable(DateTime64(3, 'UTC')),
    old_password_valid_until Nullable(DateTime64(3, 'UTC')),
    created_at               DateTime64(3, 'UTC'),
    updated_by               String,
    CONSTRAINT chk_status CHECK status IN ('pending', 'active', 'revoked', 'deleted', 'rejected'),
    CONSTRAINT chk_tier CHECK tier IN (1, 2)
)
ENGINE = ReplacingMergeTree(version)
ORDER BY login;
