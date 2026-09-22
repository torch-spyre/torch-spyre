-- OpenTelemetry telemetry from Jenkins, captured verbatim from the exporter's own schema.
--
-- NOT OURS TO DESIGN. These tables are created by the Jenkins OpenTelemetry plugin's ClickHouse
-- exporter, which owns the column set, the CODECs and the 90-day TTLs. They are checked in so a
-- database can be stood up complete and so a reader can see the shape it is querying -- not so
-- they can be edited. Changing a column here does not change what the exporter writes; it breaks
-- the exporter's insert, and Grafana's OTel data source expects these names.
--
-- We are a READER only. Nothing in this package writes them: pushToClickhouse.groovy reads
-- otel_traces to attribute stage timings, and ~16 Grafana panels read the metrics tables.
--
-- Two families, different keys: traces are keyed (ServiceName, SpanName, Timestamp) with a
-- TraceId lookup table fed by its own MV, and metrics are keyed
-- (ServiceName, MetricName, hour, hash(Attributes), TimeUnix) per instrument type.
--
-- Regenerate rather than hand-edit, so this file stays byte-faithful to what the exporter made:
--   SHOW CREATE TABLE <db>.otel_traces   (repeat per table, unqualify the database)
--
-- NEEDS CLICKHOUSE >= 25.8. otel_logs carries INDEX ... TYPE text(tokenizer = 'array'), whose
-- named-argument form an older server rejects with "Only literals can be skip index arguments".
-- Verified against the live table on prod (26.3), so the syntax is correct for the instance these
-- tables actually live on -- do NOT downgrade it to satisfy an older local container, which would
-- make this file diverge from the table it transcribes.
CREATE TABLE IF NOT EXISTS otel_traces
(
    `Timestamp` DateTime64(9) CODEC(Delta(8), ZSTD(1)),
    `TraceId` String CODEC(ZSTD(1)),
    `SpanId` String CODEC(ZSTD(1)),
    `ParentSpanId` String CODEC(ZSTD(1)),
    `TraceState` String CODEC(ZSTD(1)),
    `SpanName` LowCardinality(String) CODEC(ZSTD(1)),
    `SpanKind` LowCardinality(String) CODEC(ZSTD(1)),
    `ServiceName` LowCardinality(String) CODEC(ZSTD(1)),
    `ResourceAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `ScopeName` String CODEC(ZSTD(1)),
    `ScopeVersion` String CODEC(ZSTD(1)),
    `SpanAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `Duration` UInt64 CODEC(ZSTD(1)),
    `StatusCode` LowCardinality(String) CODEC(ZSTD(1)),
    `StatusMessage` String CODEC(ZSTD(1)),
    `Events.Timestamp` Array(DateTime64(9)) CODEC(ZSTD(1)),
    `Events.Name` Array(LowCardinality(String)) CODEC(ZSTD(1)),
    `Events.Attributes` Array(Map(LowCardinality(String), String)) CODEC(ZSTD(1)),
    `Links.TraceId` Array(String) CODEC(ZSTD(1)),
    `Links.SpanId` Array(String) CODEC(ZSTD(1)),
    `Links.TraceState` Array(String) CODEC(ZSTD(1)),
    `Links.Attributes` Array(Map(LowCardinality(String), String)) CODEC(ZSTD(1)),
    INDEX idx_trace_id TraceId TYPE bloom_filter(0.001) GRANULARITY 1,
    INDEX idx_res_attr_key mapKeys(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_res_attr_value mapValues(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_span_attr_key mapKeys(SpanAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_span_attr_value mapValues(SpanAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_duration Duration TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toDate(Timestamp)
ORDER BY (ServiceName, SpanName, toDateTime(Timestamp))
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE TABLE IF NOT EXISTS otel_traces_trace_id_ts
(
    `TraceId` String CODEC(ZSTD(1)),
    `Start` DateTime CODEC(Delta(4), ZSTD(1)),
    `End` DateTime CODEC(Delta(4), ZSTD(1)),
    INDEX idx_trace_id TraceId TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toDate(Start)
ORDER BY (TraceId, Start)
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE MATERIALIZED VIEW IF NOT EXISTS otel_traces_trace_id_ts_mv TO otel_traces_trace_id_ts
(
    `TraceId` String,
    `Start` DateTime64(9),
    `End` DateTime64(9)
)
AS SELECT
    TraceId,
    min(Timestamp) AS Start,
    max(Timestamp) AS End
FROM otel_traces
WHERE TraceId != ''
GROUP BY TraceId;

CREATE TABLE IF NOT EXISTS otel_logs
(
    `Timestamp` DateTime64(9) COMMENT 'Event timestamp with nanosecond precision' CODEC(Delta(8), ZSTD(1)),
    `TraceId` String COMMENT 'W3C trace identifier' CODEC(ZSTD(1)),
    `SpanId` String COMMENT 'W3C span identifier' CODEC(ZSTD(1)),
    `TraceFlags` UInt8 COMMENT 'W3C trace flags',
    `SeverityText` LowCardinality(String) COMMENT 'Log severity as text' CODEC(ZSTD(1)),
    `SeverityNumber` UInt8 COMMENT 'Log severity as number (1-24)',
    `ServiceName` LowCardinality(String) COMMENT 'Service that emitted the log' CODEC(ZSTD(1)),
    `Body` String COMMENT 'Log message body' CODEC(ZSTD(1)),
    `ResourceSchemaUrl` LowCardinality(String) COMMENT 'Schema URL for the resource' CODEC(ZSTD(1)),
    `ResourceAttributes` Map(LowCardinality(String), String) COMMENT 'Resource attributes as key-value pairs' CODEC(ZSTD(1)),
    `ScopeSchemaUrl` LowCardinality(String) COMMENT 'Schema URL for the instrumentation scope' CODEC(ZSTD(1)),
    `ScopeName` String COMMENT 'Instrumentation scope name' CODEC(ZSTD(1)),
    `ScopeVersion` LowCardinality(String) COMMENT 'Instrumentation scope version' CODEC(ZSTD(1)),
    `ScopeAttributes` Map(LowCardinality(String), String) COMMENT 'Instrumentation scope attributes' CODEC(ZSTD(1)),
    `LogAttributes` Map(LowCardinality(String), String) COMMENT 'Log record attributes' CODEC(ZSTD(1)),
    `EventName` String COMMENT 'Event name for log records representing events' CODEC(ZSTD(1)),
    `__otel_materialized_k8s.cluster.name` LowCardinality(String) MATERIALIZED ResourceAttributes['k8s.cluster.name'] CODEC(ZSTD(1)),
    `__otel_materialized_k8s.container.name` LowCardinality(String) MATERIALIZED ResourceAttributes['k8s.container.name'] CODEC(ZSTD(1)),
    `__otel_materialized_k8s.deployment.name` LowCardinality(String) MATERIALIZED ResourceAttributes['k8s.deployment.name'] CODEC(ZSTD(1)),
    `__otel_materialized_k8s.namespace.name` LowCardinality(String) MATERIALIZED ResourceAttributes['k8s.namespace.name'] CODEC(ZSTD(1)),
    `__otel_materialized_k8s.node.name` LowCardinality(String) MATERIALIZED ResourceAttributes['k8s.node.name'] CODEC(ZSTD(1)),
    `__otel_materialized_k8s.pod.name` LowCardinality(String) MATERIALIZED ResourceAttributes['k8s.pod.name'] CODEC(ZSTD(1)),
    `__otel_materialized_k8s.pod.uid` LowCardinality(String) MATERIALIZED ResourceAttributes['k8s.pod.uid'] CODEC(ZSTD(1)),
    `__otel_materialized_deployment.environment.name` LowCardinality(String) MATERIALIZED ResourceAttributes['deployment.environment.name'] CODEC(ZSTD(1)),
    INDEX idx_trace_id TraceId TYPE text(tokenizer = 'array') GRANULARITY 100000000,
    INDEX idx_res_attr_key mapKeys(ResourceAttributes) TYPE text(tokenizer = 'array') GRANULARITY 100000000,
    INDEX idx_res_attr_value mapValues(ResourceAttributes) TYPE text(tokenizer = 'array') GRANULARITY 100000000,
    INDEX idx_scope_attr_key mapKeys(ScopeAttributes) TYPE text(tokenizer = 'array') GRANULARITY 100000000,
    INDEX idx_scope_attr_value mapValues(ScopeAttributes) TYPE text(tokenizer = 'array') GRANULARITY 100000000,
    INDEX idx_log_attr_key mapKeys(LogAttributes) TYPE text(tokenizer = 'array') GRANULARITY 100000000,
    INDEX idx_log_attr_value mapValues(LogAttributes) TYPE text(tokenizer = 'array') GRANULARITY 100000000,
    INDEX idx_lower_body lower(Body) TYPE text(tokenizer = 'splitByNonAlpha') GRANULARITY 100000000
)
ENGINE = MergeTree
PARTITION BY toDate(Timestamp)
ORDER BY (toStartOfFiveMinutes(Timestamp), ServiceName, Timestamp)
TTL toDateTime(Timestamp) + toIntervalDay(90)
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE TABLE IF NOT EXISTS otel_metrics_gauge
(
    `ResourceAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `ResourceSchemaUrl` String CODEC(ZSTD(1)),
    `ScopeName` String CODEC(ZSTD(1)),
    `ScopeVersion` String CODEC(ZSTD(1)),
    `ScopeAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `ScopeDroppedAttrCount` UInt32 CODEC(ZSTD(1)),
    `ScopeSchemaUrl` String CODEC(ZSTD(1)),
    `ServiceName` LowCardinality(String) CODEC(ZSTD(1)),
    `MetricName` LowCardinality(String) CODEC(ZSTD(1)),
    `MetricDescription` String CODEC(ZSTD(1)),
    `MetricUnit` String CODEC(ZSTD(1)),
    `Attributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `StartTimeUnix` DateTime CODEC(Delta(4), ZSTD(1)),
    `TimeUnix` DateTime CODEC(Delta(4), ZSTD(1)),
    `Value` Float64 CODEC(ZSTD(1)),
    `Flags` UInt32 CODEC(ZSTD(1)),
    `Exemplars.FilteredAttributes` Array(Map(LowCardinality(String), String)) CODEC(ZSTD(1)),
    `Exemplars.TimeUnix` Array(DateTime) CODEC(ZSTD(1)),
    `Exemplars.Value` Array(Float64) CODEC(ZSTD(1)),
    `Exemplars.SpanId` Array(String) CODEC(ZSTD(1)),
    `Exemplars.TraceId` Array(String) CODEC(ZSTD(1)),
    INDEX idx_res_attr_key mapKeys(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_res_attr_value mapValues(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_scope_attr_key mapKeys(ScopeAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_scope_attr_value mapValues(ScopeAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_attr_key mapKeys(Attributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_attr_value mapValues(Attributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_time_minmax TimeUnix TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toDate(TimeUnix)
ORDER BY (ServiceName, MetricName, toStartOfHour(TimeUnix), cityHash64(Attributes), TimeUnix)
TTL toDateTime(TimeUnix) + toIntervalDay(90)
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE TABLE IF NOT EXISTS otel_metrics_sum
(
    `ResourceAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `ResourceSchemaUrl` String CODEC(ZSTD(1)),
    `ScopeName` String CODEC(ZSTD(1)),
    `ScopeVersion` String CODEC(ZSTD(1)),
    `ScopeAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `ScopeDroppedAttrCount` UInt32 CODEC(ZSTD(1)),
    `ScopeSchemaUrl` String CODEC(ZSTD(1)),
    `ServiceName` LowCardinality(String) CODEC(ZSTD(1)),
    `MetricName` LowCardinality(String) CODEC(ZSTD(1)),
    `MetricDescription` String CODEC(ZSTD(1)),
    `MetricUnit` String CODEC(ZSTD(1)),
    `Attributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `StartTimeUnix` DateTime CODEC(Delta(4), ZSTD(1)),
    `TimeUnix` DateTime CODEC(Delta(4), ZSTD(1)),
    `Value` Float64 CODEC(ZSTD(1)),
    `Flags` UInt32 CODEC(ZSTD(1)),
    `Exemplars.FilteredAttributes` Array(Map(LowCardinality(String), String)) CODEC(ZSTD(1)),
    `Exemplars.TimeUnix` Array(DateTime) CODEC(ZSTD(1)),
    `Exemplars.Value` Array(Float64) CODEC(ZSTD(1)),
    `Exemplars.SpanId` Array(String) CODEC(ZSTD(1)),
    `Exemplars.TraceId` Array(String) CODEC(ZSTD(1)),
    `AggregationTemporality` Int32 CODEC(ZSTD(1)),
    `IsMonotonic` Bool CODEC(Delta(1), ZSTD(1)),
    INDEX idx_res_attr_key mapKeys(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_res_attr_value mapValues(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_scope_attr_key mapKeys(ScopeAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_scope_attr_value mapValues(ScopeAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_attr_key mapKeys(Attributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_attr_value mapValues(Attributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_time_minmax TimeUnix TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toDate(TimeUnix)
ORDER BY (ServiceName, MetricName, toStartOfHour(TimeUnix), cityHash64(Attributes), TimeUnix)
TTL toDateTime(TimeUnix) + toIntervalDay(90)
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE TABLE IF NOT EXISTS otel_metrics_histogram
(
    `ResourceAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `ResourceSchemaUrl` String CODEC(ZSTD(1)),
    `ScopeName` String CODEC(ZSTD(1)),
    `ScopeVersion` String CODEC(ZSTD(1)),
    `ScopeAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `ScopeDroppedAttrCount` UInt32 CODEC(ZSTD(1)),
    `ScopeSchemaUrl` String CODEC(ZSTD(1)),
    `ServiceName` LowCardinality(String) CODEC(ZSTD(1)),
    `MetricName` LowCardinality(String) CODEC(ZSTD(1)),
    `MetricDescription` String CODEC(ZSTD(1)),
    `MetricUnit` String CODEC(ZSTD(1)),
    `Attributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `StartTimeUnix` DateTime CODEC(Delta(4), ZSTD(1)),
    `TimeUnix` DateTime CODEC(Delta(4), ZSTD(1)),
    `Count` UInt64 CODEC(Delta(8), ZSTD(1)),
    `Sum` Float64 CODEC(ZSTD(1)),
    `BucketCounts` Array(UInt64) CODEC(ZSTD(1)),
    `ExplicitBounds` Array(Float64) CODEC(ZSTD(1)),
    `Exemplars.FilteredAttributes` Array(Map(LowCardinality(String), String)) CODEC(ZSTD(1)),
    `Exemplars.TimeUnix` Array(DateTime) CODEC(ZSTD(1)),
    `Exemplars.Value` Array(Float64) CODEC(ZSTD(1)),
    `Exemplars.SpanId` Array(String) CODEC(ZSTD(1)),
    `Exemplars.TraceId` Array(String) CODEC(ZSTD(1)),
    `Flags` UInt32 CODEC(ZSTD(1)),
    `Min` Float64 CODEC(ZSTD(1)),
    `Max` Float64 CODEC(ZSTD(1)),
    `AggregationTemporality` Int32 CODEC(ZSTD(1)),
    INDEX idx_res_attr_key mapKeys(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_res_attr_value mapValues(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_scope_attr_key mapKeys(ScopeAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_scope_attr_value mapValues(ScopeAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_attr_key mapKeys(Attributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_attr_value mapValues(Attributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_time_minmax TimeUnix TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toDate(TimeUnix)
ORDER BY (ServiceName, MetricName, toStartOfHour(TimeUnix), cityHash64(Attributes), TimeUnix)
TTL toDateTime(TimeUnix) + toIntervalDay(90)
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE TABLE IF NOT EXISTS otel_metrics_summary
(
    `ResourceAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `ResourceSchemaUrl` String CODEC(ZSTD(1)),
    `ScopeName` String CODEC(ZSTD(1)),
    `ScopeVersion` String CODEC(ZSTD(1)),
    `ScopeAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `ScopeDroppedAttrCount` UInt32 CODEC(ZSTD(1)),
    `ScopeSchemaUrl` String CODEC(ZSTD(1)),
    `ServiceName` LowCardinality(String) CODEC(ZSTD(1)),
    `MetricName` LowCardinality(String) CODEC(ZSTD(1)),
    `MetricDescription` String CODEC(ZSTD(1)),
    `MetricUnit` String CODEC(ZSTD(1)),
    `Attributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `StartTimeUnix` DateTime CODEC(Delta(4), ZSTD(1)),
    `TimeUnix` DateTime CODEC(Delta(4), ZSTD(1)),
    `Count` UInt64 CODEC(Delta(8), ZSTD(1)),
    `Sum` Float64 CODEC(ZSTD(1)),
    `ValueAtQuantiles.Quantile` Array(Float64) CODEC(ZSTD(1)),
    `ValueAtQuantiles.Value` Array(Float64) CODEC(ZSTD(1)),
    `Flags` UInt32 CODEC(ZSTD(1)),
    INDEX idx_res_attr_key mapKeys(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_res_attr_value mapValues(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_scope_attr_key mapKeys(ScopeAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_scope_attr_value mapValues(ScopeAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_attr_key mapKeys(Attributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_attr_value mapValues(Attributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_time_minmax TimeUnix TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toDate(TimeUnix)
ORDER BY (ServiceName, MetricName, toStartOfHour(TimeUnix), cityHash64(Attributes), TimeUnix)
TTL toDateTime(TimeUnix) + toIntervalDay(90)
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;

CREATE TABLE IF NOT EXISTS otel_metrics_exponential_histogram
(
    `ResourceAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `ResourceSchemaUrl` String CODEC(ZSTD(1)),
    `ScopeName` String CODEC(ZSTD(1)),
    `ScopeVersion` String CODEC(ZSTD(1)),
    `ScopeAttributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `ScopeDroppedAttrCount` UInt32 CODEC(ZSTD(1)),
    `ScopeSchemaUrl` String CODEC(ZSTD(1)),
    `ServiceName` LowCardinality(String) CODEC(ZSTD(1)),
    `MetricName` LowCardinality(String) CODEC(ZSTD(1)),
    `MetricDescription` String CODEC(ZSTD(1)),
    `MetricUnit` String CODEC(ZSTD(1)),
    `Attributes` Map(LowCardinality(String), String) CODEC(ZSTD(1)),
    `StartTimeUnix` DateTime CODEC(Delta(4), ZSTD(1)),
    `TimeUnix` DateTime CODEC(Delta(4), ZSTD(1)),
    `Count` UInt64 CODEC(Delta(8), ZSTD(1)),
    `Sum` Float64 CODEC(ZSTD(1)),
    `Scale` Int32 CODEC(ZSTD(1)),
    `ZeroCount` UInt64 CODEC(ZSTD(1)),
    `PositiveOffset` Int32 CODEC(ZSTD(1)),
    `PositiveBucketCounts` Array(UInt64) CODEC(ZSTD(1)),
    `NegativeOffset` Int32 CODEC(ZSTD(1)),
    `NegativeBucketCounts` Array(UInt64) CODEC(ZSTD(1)),
    `Exemplars.FilteredAttributes` Array(Map(LowCardinality(String), String)) CODEC(ZSTD(1)),
    `Exemplars.TimeUnix` Array(DateTime) CODEC(ZSTD(1)),
    `Exemplars.Value` Array(Float64) CODEC(ZSTD(1)),
    `Exemplars.SpanId` Array(String) CODEC(ZSTD(1)),
    `Exemplars.TraceId` Array(String) CODEC(ZSTD(1)),
    `Flags` UInt32 CODEC(ZSTD(1)),
    `Min` Float64 CODEC(ZSTD(1)),
    `Max` Float64 CODEC(ZSTD(1)),
    `AggregationTemporality` Int32 CODEC(ZSTD(1)),
    INDEX idx_res_attr_key mapKeys(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_res_attr_value mapValues(ResourceAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_scope_attr_key mapKeys(ScopeAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_scope_attr_value mapValues(ScopeAttributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_attr_key mapKeys(Attributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_attr_value mapValues(Attributes) TYPE bloom_filter(0.01) GRANULARITY 1,
    INDEX idx_time_minmax TimeUnix TYPE minmax GRANULARITY 1
)
ENGINE = MergeTree
PARTITION BY toDate(TimeUnix)
ORDER BY (ServiceName, MetricName, toStartOfHour(TimeUnix), cityHash64(Attributes), TimeUnix)
TTL toDateTime(TimeUnix) + toIntervalDay(90)
SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;
