CREATE TABLE vigil_ops.events
(
    `ts` DateTime64(3, 'UTC') DEFAULT now64(3),
    `ended_at` Nullable(DateTime64(3, 'UTC')),
    `kind` LowCardinality(String),
    `source` LowCardinality(String),
    `subject_type` LowCardinality(String) DEFAULT 'unknown',
    `subject` String,
    `summary` String DEFAULT '',
    `attrs_num` Map(LowCardinality(String), Float64),
    `evidence_url` String DEFAULT '',
    `evidence_kind` LowCardinality(String) DEFAULT 'recorded',
    `confidence` LowCardinality(String) DEFAULT 'exact',
    `dedup_key` String DEFAULT '',
    `recorded_by` LowCardinality(String),
    `writer_run_key` String DEFAULT '',
    `actor_kind` LowCardinality(String) DEFAULT 'system',
    `actor` String DEFAULT '',
    `props` Map(LowCardinality(String), String),
    `audit_uuid` UUID DEFAULT generateUUIDv7(),
    `audit_timestamp` DateTime64(3) DEFAULT now64(3),
    INDEX ix_events_dedup dedup_key TYPE bloom_filter(0.01) GRANULARITY 4,
    INDEX ix_events_id audit_uuid TYPE bloom_filter(0.01) GRANULARITY 4,
    CONSTRAINT chk_event_keys CHECK (kind != '') AND (subject != '') AND (recorded_by != ''),
    CONSTRAINT chk_event_ended CHECK (ended_at IS NULL) OR (ended_at >= ts),
    CONSTRAINT chk_event_evidence_kind CHECK evidence_kind IN ('recorded', 'inferred', 'reported'),
    CONSTRAINT chk_event_confidence CHECK confidence IN ('exact', 'probable', 'unresolved'),
    CONSTRAINT chk_event_actor CHECK actor_kind IN ('system', 'agent', 'assistant', 'human'),
    CONSTRAINT chk_event_run_subject CHECK (subject_type NOT IN ('run', 'build', 'workflow', 'job_run')) OR match(subject, '^(gha:[^/#]+/[^/#]+/[0-9]+#[0-9]+(/[0-9]+)?|[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*#[0-9]+)$'),
    CONSTRAINT chk_event_writer_run CHECK (writer_run_key = '') OR match(writer_run_key, '^(gha:[^/#]+/[^/#]+/[0-9]+#[0-9]+(/[0-9]+)?|[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*#[0-9]+)$')
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (kind, subject_type, subject, ts)
TTL toDateTime(ts) + multiIf((kind IN ('deploy', 'config_change', 'maintenance')), toIntervalDay(730), toIntervalDay(365))
SETTINGS index_granularity = 8192
