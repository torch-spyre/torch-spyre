CREATE TABLE spyre_v2.artifact_results
(
    `ts` DateTime DEFAULT now(),
    `artifact_id` UUID,
    `run_id` UUID,
    `result_kind` LowCardinality(String),
    `test_type` LowCardinality(String),
    `state` LowCardinality(String),
    `arch` LowCardinality(String),
    `duration_s` Float32,
    `props` Map(LowCardinality(String), String),
    `audit_uuid` UUID DEFAULT generateUUIDv7(),
    `audit_timestamp` DateTime64(3) DEFAULT now64(3),
    INDEX idx_run_id run_id TYPE bloom_filter(0.01) GRANULARITY 1,
    CONSTRAINT chk_state CHECK state IN ('passed', 'failed', 'error', 'running'),
    CONSTRAINT chk_result_kind CHECK result_kind IN ('functional', 'performance', 'capability'),
    CONSTRAINT chk_test_type CHECK test_type IN ('smoke', 'unit', 'integration', 'regression', 'trunk', 'perf', 'quality', 'fvt', 'fvt-static', 'fvt-dynamic', 'svt', 'svt-static', 'svt-dynamic', 'model_ops', 'model_modules', 'model_support', 'model_serving')
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (artifact_id, result_kind, test_type, ts)
TTL ts + toIntervalDay(90) WHERE state = 'running'
SETTINGS index_granularity = 8192
