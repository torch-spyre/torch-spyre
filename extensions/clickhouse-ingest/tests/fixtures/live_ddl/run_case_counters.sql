CREATE TABLE spyre_v2.run_case_counters
(
    `run_id` UUID,
    `component` LowCardinality(String),
    `total_tests` UInt64,
    `passed` UInt64,
    `failed` UInt64,
    `errors` UInt64,
    `skipped` UInt64,
    `xfail` UInt64,
    `xpass` UInt64,
    `recovered` UInt64
)
ENGINE = SummingMergeTree
ORDER BY (run_id, component)
SETTINGS index_granularity = 8192
