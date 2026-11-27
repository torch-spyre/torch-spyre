CREATE TABLE vigil_ops.metrics_latest
(
    `subject_type` LowCardinality(String),
    `subject` String,
    `metric` LowCardinality(String),
    `scope` LowCardinality(String),
    `cur` AggregateFunction(argMax, Float64, DateTime64(3, 'UTC')),
    `cur_unit` AggregateFunction(argMax, LowCardinality(String), DateTime64(3, 'UTC')),
    `cur_labels` AggregateFunction(argMax, Map(LowCardinality(String), String), DateTime64(3, 'UTC')),
    `cur_props` AggregateFunction(argMax, Map(LowCardinality(String), String), DateTime64(3, 'UTC')),
    `as_of` SimpleAggregateFunction(max, DateTime64(3, 'UTC'))
)
ENGINE = AggregatingMergeTree
ORDER BY (subject_type, subject, metric, scope)
TTL toDateTime(as_of) + toIntervalDay(400)
SETTINGS index_granularity = 8192
