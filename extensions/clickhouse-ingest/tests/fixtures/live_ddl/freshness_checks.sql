CREATE TABLE vigil_ops.freshness_checks
(
    `checked_at` DateTime64(3, 'UTC'),
    `check_kind` LowCardinality(String),
    `subject` String,
    `scope` LowCardinality(String) DEFAULT '',
    `track` LowCardinality(String) DEFAULT '',
    `state` LowCardinality(String),
    `age_hours` Nullable(Float64),
    `window_hours` Nullable(Float64),
    `behind_count` UInt32 DEFAULT 0,
    `detail` String DEFAULT '',
    `run_url` String DEFAULT '',
    `source` LowCardinality(String) DEFAULT '',
    `schema_version` UInt8 DEFAULT 1,
    `ingested_at` DateTime64(3, 'UTC') DEFAULT now64(3),
    `props` Map(LowCardinality(String), String),
    `audit_uuid` UUID DEFAULT generateUUIDv7(),
    `audit_timestamp` DateTime64(3) DEFAULT now64(3),
    CONSTRAINT chk_freshness_state CHECK state IN ('fresh', 'stale', 'wrong_provenance', 'dangling', 'absent', 'no_provenance', 'unknown', 'ancestry_unknown', 'check_failed'),
    CONSTRAINT chk_freshness_detail CHECK length(detail) <= 8000
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(checked_at)
ORDER BY (check_kind, subject, scope, track, checked_at)
TTL toDateTime(checked_at) + toIntervalDay(400)
SETTINGS index_granularity = 8192
