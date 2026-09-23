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
    -- Folded to one spelling, as identity.py's canonical_arch folds it for the hash (Jenkins
    -- says 'amd64', GHA says 'x86_64' for one platform).
    if(ar.arch IN ('amd64', 'x86', 'x86-64'), 'x86_64', ar.arch) AS arch,
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

-- Replaces perf_kernels.ratio, which v1 stored as a third column beside the two it divides:
-- torch_spyre_ms and sendnn_ms were never one row (same benchmark, two backends), so this
-- self-joins them and the ratio cannot disagree with its operands.
CREATE VIEW IF NOT EXISTS v_benchmark_backend_compare AS
SELECT
    t.run_id, t.benchmark_id, t.component, t.name, t.arch, t.record_type, t.kernel_name,
    t.backend AS backend, s.backend AS baseline_backend,
    -- NULL, not the Map zero-default: 0/baseline would read as "100% faster" rather than "not measured".
    if(has(mapKeys(t.measurements), 'duration_ms'), t.measurements['duration_ms'], NULL) AS duration_ms,
    if(has(mapKeys(s.measurements), 'duration_ms'), s.measurements['duration_ms'], NULL) AS baseline_duration_ms,
    duration_ms / nullIf(baseline_duration_ms, 0) AS ratio
FROM v_benchmark_results_enriched AS t
INNER JOIN v_benchmark_results_enriched AS s
        ON t.run_id = s.run_id AND t.benchmark_id = s.benchmark_id
WHERE t.backend != s.backend;

-- Replaces perf_benchmarks.regression_status with a verdict against the immediately preceding
-- run of the same benchmark+backend+arch, via a window function (not an all-pairs self-join,
-- which costs runs^2 and double-reports every finding without a baseline_run_id constraint).
-- Threshold matches the dashboard's +/-5%; an absent baseline metric yields NULL, never a verdict.
CREATE VIEW IF NOT EXISTS v_benchmark_regression AS
SELECT
    run_id, benchmark_id, component, backend, name, arch,
    record_type, config_name, input_shapes,
    ts,
    baseline_run_id,
    metric, new_value, baseline_value,
    if(baseline_value IS NULL OR baseline_value = 0, NULL,
       round((new_value - baseline_value) / abs(baseline_value) * 100, 1)) AS delta_pct,
    multiIf(baseline_value IS NULL OR baseline_value = 0, '',
            (new_value - baseline_value) / abs(baseline_value) * 100 >  5, 'regressed',
            (new_value - baseline_value) / abs(baseline_value) * 100 < -5, 'improved',
            'unchanged') AS regression_status
FROM (
    SELECT
        run_id, benchmark_id, component, backend, name, arch,
        record_type, config_name, input_shapes, ts,
        m.1 AS metric,
        m.2 AS new_value,
        lagInFrame(m.2)      OVER w AS baseline_value,
        lagInFrame(run_id)  OVER w AS baseline_run_id
    FROM (
        SELECT run_id, benchmark_id, component, backend, name, arch, record_type, config_name,
               input_shapes, ts, measurements
        FROM v_benchmark_results_enriched
    )
    ARRAY JOIN CAST(measurements, 'Array(Tuple(String, Float64))') AS m
    WINDOW w AS (PARTITION BY benchmark_id, backend, arch, m.1 ORDER BY ts, run_id)
)
WHERE baseline_run_id != run_id;

-- Per-arch trend for one benchmark+metric: the platform comparison the dashboard draws.
CREATE VIEW IF NOT EXISTS v_benchmark_trend AS
SELECT
    toStartOfDay(ts) AS day, benchmark_id, name, component, backend, arch,
    m.1 AS metric,
    round(avg(m.2), 4) AS avg_value,
    round(min(m.2), 4) AS min_value,
    round(max(m.2), 4) AS max_value,
    uniqExact(run_id) AS runs
FROM v_benchmark_results_enriched
ARRAY JOIN CAST(measurements, 'Array(Tuple(String, Float64))') AS m
GROUP BY day, benchmark_id, name, component, backend, arch, metric;
