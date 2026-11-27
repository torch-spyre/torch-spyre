-- Append-only lifecycle log behind a login's detail panel. `detail` is JSON and never a secret.
CREATE TABLE IF NOT EXISTS login_events
(
    event_id UUID DEFAULT generateUUIDv7(),
    ts       DateTime64(3, 'UTC') DEFAULT now64(3),
    login    String,
    actor    String,
    action   LowCardinality(String), -- request|approve|reject|create|claim|extend|rotate|retire|revoke|restore|delete|managers|auto_delete|cleanup
    detail   String DEFAULT '',
    reason   String DEFAULT ''
)
ENGINE = MergeTree
ORDER BY (login, ts);
