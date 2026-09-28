-- Views over benchmarks / benchmark_runs. Two jobs: present the wide shape the dashboard
-- already speaks, and derive the two verdicts v1 stored as columns.

-- Every benchmark measurement with its identity, run context and tag resolved -- the base join
-- for everything below.
CREATE VIEW IF NOT EXISTS v_benchmark_results_enriched AS
SELECT
    r.ts AS ts, r.run_id AS run_id, r.benchmark_id AS benchmark_id,
    r.component AS component,
    r.backend AS backend,
    b.name AS name, b.tags AS tags, b.props AS bench_props,
    b.props['record_type']  AS record_type,
    b.props['config_name']  AS config_name,
    b.props['input_shapes'] AS input_shapes,
    b.props['run_mode']     AS run_mode,
    b.props['kernel_name']  AS kernel_name,
    -- measurements is Map(String, Array(Float64)) on the table (every sample); reduced to one
    -- value per metric here, under the same name, so every view below addresses scalars.
    -- arrayAvg, not samples[1], which would depend on harness ordering.
    mapApply((k, v) -> (k, arrayAvg(v)), r.measurements) AS measurements,
    -- The samples themselves, for variance, a percentile, or a geomean that differs from the mean.
    r.measurements AS samples,
    r.iterations AS iterations,
    r.props AS run_props,
    ar.artifact_id,
    -- Where it ran: the producer's props arch/platform, else the leg's arch. Folded to one
    -- spelling, as identity.py's canonical_arch folds it (Jenkins 'amd64', GHA 'x86_64').
    if((multiIf(r.props['arch'] != '', r.props['arch'],
                r.props['platform'] != '', r.props['platform'],
                ar.arch != '', ar.arch,
                -- Fallback only: parses the `<job>-perf-<arch>/` segment of source_file, a
                -- path convention rather than a recorded field.
                extract(r.props['source_file'], '^[^/]*-perf-([A-Za-z0-9_]+)/')) AS raw_arch)
           IN ('amd64', 'x86', 'x86-64'), 'x86_64', raw_arch) AS arch,
    -- The producing job, which separates suites the stored component does not (hf-adapters' perf
    -- job files its rows as torch-spyre).
    extract(r.props['source_file'], '^([^/]+)/') AS source_job,
    -- When the run happened: run_utc when stamped, else ingest `ts` (DEFAULT now()), which
    -- ts_source names so a reader can tell a run time from an ingest time.
    ifNull(parseDateTimeBestEffortOrNull(r.props['run_utc'], 'UTC')::Nullable(DateTime), r.ts) AS run_ts,
    if(isNull(parseDateTimeBestEffortOrNull(r.props['run_utc'], 'UTC')), 'ingest', 'run_utc') AS ts_source,
    ar.test_type, ar.state
FROM benchmark_runs AS r
INNER JOIN benchmarks AS b USING (benchmark_id)
-- Deduped to one artifact_results row per run first: a plain MergeTree with no dedup key would
-- otherwise let a re-ingested leg double every measurement. argMax on ts keeps the latest row.
LEFT JOIN (
    SELECT run_id,
           argMax(artifact_id, ts) AS artifact_id,
           argMax(arch, ts)        AS arch,
           argMax(test_type, ts)   AS test_type,
           argMax(state, ts)       AS state
    FROM artifact_results
    WHERE result_kind = 'performance'
    GROUP BY run_id
) AS ar USING (run_id);

-- The wide projection the dashboard's METRIC_LABELS dict expects: Map keys become named
-- columns. Every metric is NULL when absent, never 0 (Map's zero-default), since a dashboard
-- +/-5% delta would otherwise read a missing measurement as a 100% regression.
CREATE VIEW IF NOT EXISTS v_benchmark_wide AS
SELECT
    run_id, benchmark_id, component, backend, artifact_id, arch, ts,
    name AS operation_name, record_type, config_name, input_shapes, run_mode,
    if(has(mapKeys(measurements), 'total_duration_ms'), measurements['total_duration_ms'], NULL) AS total_duration_ms,
    if(has(mapKeys(measurements), 'cpu_ms'), measurements['cpu_ms'], NULL) AS cpu_ms,
    if(has(mapKeys(measurements), 'spyre_ms'), measurements['spyre_ms'], NULL) AS spyre_ms,
    if(has(mapKeys(measurements), 'kernel_mean_ms'), measurements['kernel_mean_ms'], NULL) AS kernel_mean_ms,
    if(has(mapKeys(measurements), 'memory_transfer_mean_ms'), measurements['memory_transfer_mean_ms'], NULL) AS memory_transfer_mean_ms,
    if(has(mapKeys(measurements), 'compile_ms'), measurements['compile_ms'], NULL) AS compile_ms,
    if(has(mapKeys(measurements), 'runtime_ms'), measurements['runtime_ms'], NULL) AS runtime_ms,
    if(has(mapKeys(measurements), 'mem_size_mb'), measurements['mem_size_mb'], NULL) AS mem_size_mb,
    if(has(mapKeys(measurements), 'pt_util_percent'), measurements['pt_util_percent'], NULL) AS pt_util_percent,
    iterations
FROM v_benchmark_results_enriched;

-- Replaces perf_kernels.ratio, which v1 stored beside the two values it divides. Two pairings,
-- ratio = value / baseline_value:
--   in_row    - one row's own spyre_ms against its cpu_ms (the harness times both per op).
--   cross_run - torch-spyre against sendnn: same component, benchmark_id, arch and metric. They
--               never share a run, so each row pairs with the latest sendnn run at or before it
--               (ASOF), and only within 7 days (gap_hours) -- older than that the baseline no
--               longer describes the same toolchain.
-- Zeros are dropped: producers write 0 for a metric they did not measure.
CREATE VIEW IF NOT EXISTS v_benchmark_backend_compare AS
SELECT
    run_id, benchmark_id, component, name, arch, record_type, kernel_name,
    backend, 'cpu' AS baseline_backend, 'in_row' AS pairing,
    run_ts AS ts, ts_source, run_id AS baseline_run_id, run_ts AS baseline_ts,
    ts_source AS baseline_ts_source, 0 AS gap_hours,
    'spyre_ms' AS metric,
    measurements['spyre_ms'] AS value,
    measurements['cpu_ms']   AS baseline_value,
    value / baseline_value   AS ratio
FROM v_benchmark_results_enriched
WHERE measurements['spyre_ms'] != 0 AND measurements['cpu_ms'] != 0
UNION ALL
SELECT
    t.run_id, t.benchmark_id, t.component, t.name, t.arch, t.record_type, t.kernel_name,
    t.backend, 'sendnn', 'cross_run',
    t.run_ts, t.ts_source, s.run_id, s.run_ts, s.ts_source,
    dateDiff('hour', s.run_ts, t.run_ts),
    t.metric, t.value, s.value, t.value / s.value
FROM
(
    SELECT run_id, benchmark_id, component, name, arch, record_type, kernel_name, backend,
           run_ts, ts_source, m.1 AS metric, m.2 AS value
    FROM v_benchmark_results_enriched
    ARRAY JOIN CAST(measurements, 'Array(Tuple(String, Float64))') AS m
    WHERE backend != 'sendnn' AND m.2 != 0
) AS t
ASOF INNER JOIN
(
    SELECT run_id, benchmark_id, component, arch, run_ts, ts_source, m.1 AS metric, m.2 AS value
    FROM v_benchmark_results_enriched
    ARRAY JOIN CAST(measurements, 'Array(Tuple(String, Float64))') AS m
    WHERE backend = 'sendnn' AND m.2 != 0
) AS s
ON t.component = s.component AND t.benchmark_id = s.benchmark_id AND t.arch = s.arch
   AND t.metric = s.metric AND t.run_ts >= s.run_ts
WHERE t.run_ts - s.run_ts <= 7 * 86400;

-- Replaces perf_benchmarks.regression_status with a verdict against the previous run of the same
-- benchmark, backend, arch and producing job, in run time, via a window function (an all-pairs
-- self-join costs runs^2). Threshold matches the dashboard's +/-5%. A metric named *throughput*
-- or *_per_second, or pt_util_percent, is higher-is-better; every other metric lower-is-better.
CREATE VIEW IF NOT EXISTS v_benchmark_regression AS
SELECT
    run_id, benchmark_id, component, backend, name, arch, source_job,
    record_type, config_name, input_shapes,
    run_ts AS ts, ts_source,
    baseline_run_id, baseline_ts,
    metric, higher_is_better, new_value, baseline_value,
    round((new_value - baseline_value) / abs(baseline_value) * 100 AS change_pct, 1) AS delta_pct,
    multiIf(baseline_value IS NULL, 'no_baseline',
            if(higher_is_better, -change_pct, change_pct) >  5, 'regressed',
            if(higher_is_better, -change_pct, change_pct) < -5, 'improved',
            'unchanged') AS regression_status
FROM (
    SELECT
        *,
        match(metric, 'throughput|_per_second$') OR metric = 'pt_util_percent' AS higher_is_better,
        lagInFrame(toNullable(new_value)) OVER w AS baseline_value,
        lagInFrame(toNullable(run_id))    OVER w AS baseline_run_id,
        lagInFrame(toNullable(run_ts))    OVER w AS baseline_ts
    FROM (
        -- One value per run and metric, so a re-ingested run is never its own baseline. Zeros
        -- are dropped: producers write 0 for a metric they did not measure.
        SELECT run_id, benchmark_id, component, backend, arch, source_job,
               any(name) AS name, any(record_type) AS record_type,
               any(config_name) AS config_name, any(input_shapes) AS input_shapes,
               min(run_ts) AS run_ts, any(ts_source) AS ts_source,
               m.1 AS metric, avg(m.2) AS new_value
        FROM v_benchmark_results_enriched
        ARRAY JOIN CAST(measurements, 'Array(Tuple(String, Float64))') AS m
        WHERE m.2 != 0
        GROUP BY run_id, benchmark_id, component, backend, arch, source_job, metric
    )
    WINDOW w AS (PARTITION BY component, benchmark_id, backend, arch, source_job, metric
                 ORDER BY run_ts, run_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
);

-- Per-arch trend for one benchmark+metric: the platform comparison the dashboard draws.
CREATE VIEW IF NOT EXISTS v_benchmark_trend AS
SELECT
    toStartOfDay(run_ts) AS day, benchmark_id, name, component, backend, arch,
    m.1 AS metric,
    round(avg(m.2), 4) AS avg_value,
    round(min(m.2), 4) AS min_value,
    round(max(m.2), 4) AS max_value,
    uniqExact(run_id) AS runs
FROM v_benchmark_results_enriched
ARRAY JOIN CAST(measurements, 'Array(Tuple(String, Float64))') AS m
GROUP BY day, benchmark_id, name, component, backend, arch, metric;
