CREATE TABLE spyre_v2.test_case_runs
(
    `ts` DateTime DEFAULT now(),
    `run_id` UUID,
    `test_case_id` UUID,
    `component` LowCardinality(String),
    `status` LowCardinality(String),
    `duration_s` Float32,
    `fail_message` String DEFAULT '',
    `props` Map(LowCardinality(String), String),
    `tags` Array(LowCardinality(String)),
    `measurements` Map(LowCardinality(String), Float64),
    `audit_uuid` UUID DEFAULT generateUUIDv7(),
    `audit_timestamp` DateTime64(3) DEFAULT now64(3),
    INDEX idx_run_id run_id TYPE bloom_filter(0.01) GRANULARITY 1,
    CONSTRAINT chk_status CHECK status IN ('passed', 'failed', 'error', 'skipped', 'xfail', 'xpass')
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (component, run_id, test_case_id)
SETTINGS index_granularity = 8192
