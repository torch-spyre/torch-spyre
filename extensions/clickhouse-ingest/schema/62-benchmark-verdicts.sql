-- Benchmark regression verdicts, derived in SQL from benchmark_runs: a per-metric policy, the
-- per-metric verdict, the per-benchmark gate, and a refreshable MV that keeps verdict history.
-- Nothing writes these but the schema itself; a gate reads v_benchmark_gate for its run_id.

-- Per-metric gating policy. A view, so a policy change is a reviewed edit here that the applier
-- recreates and the next refresh picks up. A metric absent here is informational.
CREATE VIEW IF NOT EXISTS benchmark_metric_policy AS
SELECT * FROM values(
    'metric String, tier String, higher_is_better Bool, floor_pct Float64',
    ('kernel_mean_ms',          'blocking',      false, 3),
    ('memory_transfer_mean_ms', 'blocking',      false, 3),
    ('total_duration_ms',       'blocking',      false, 3),
    ('spyre_ms',                'advisory',      false, 3),
    ('runtime_ms',              'advisory',      false, 3),
    ('compile_ms',              'advisory',      false, 5),
    ('pt_util_percent',         'advisory',      true,  3),
    ('cpu_ms',                  'informational', false, 5),
    ('mean_ttft_ms',            'blocking',      false, 5),
    ('mean_tpot_ms',            'blocking',      false, 5),
    ('output_throughput',       'blocking',      true,  5),
    ('request_throughput',      'advisory',      true,  5)
);

-- One row per run x series x metric; a series is (component, benchmark_id, backend, arch,
-- source_job). The baseline is the series' previous 20 runs by run time -- backward-only, so a
-- verdict is stable until older data is backfilled or the policy changes.
-- Two detectors, both required for `regressed`: a one-sided Mann-Whitney U (score = its p) and
-- the median delta beyond floor_pct. With one current sample U's p cannot fall below about
-- 1/(n_baseline+1), so a single-sample run reaches only `suspected` (reason single_run).
CREATE VIEW IF NOT EXISTS v_benchmark_metric_verdicts AS
WITH 10 AS min_baseline, 0.01 AS alpha
SELECT
    run_id, benchmark_id, component, backend, arch, source_job, name, input_shapes,
    run_ts AS ts, metric, tier, higher_is_better,
    'rolling' AS baseline_kind,
    base_runs AS baseline_run_ids,
    length(base_runs) AS n_baseline,
    length(cur) AS n_current,
    round(base_med, 4) AS baseline_value,
    round(cur_med, 4) AS current_value,
    round((cur_med - base_med) / abs(base_med) * 100, 1) AS delta_pct,
    -- delta_pct signed so that positive is always worse.
    round(if(higher_is_better, -1, 1) * delta_pct, 1) AS worse_pct,
    round(if(higher_is_better, p_greater, p_less), 4) AS score,
    -- Never tighter than 2 robust sigmas (1.4826 * MAD) of the baseline's own noise.
    round(greatest(policy_floor, 200 * 1.4826 * base_mad / abs(base_med)), 1) AS floor_pct,
    multiIf(n_baseline < min_baseline,                'no_baseline',
            worse_pct >  floor_pct AND score < alpha, 'regressed',
            worse_pct >  floor_pct,                   'suspected',
            worse_pct < -floor_pct,                   'improved',
                                                      'unchanged') AS status,
    multiIf(status = 'suspected' AND n_current < 2,   'single_run',
            status = 'suspected',                     'one_detector',
            status = 'unchanged' AND score < alpha,   'below_floor',
                                                      '') AS reason
FROM (
    SELECT r.*,
        if(p.has_policy, p.tier, 'informational') AS tier,
        if(p.has_policy, p.higher_is_better,
           match(metric, 'throughput|_per_second$') OR metric = 'pt_util_percent') AS higher_is_better,
        if(p.has_policy, p.floor_pct, 5) AS policy_floor,
        arrayReduce('median', base) AS base_med,
        arrayReduce('median', cur) AS cur_med,
        arrayReduce('median', arrayMap(x -> abs(x - base_med), base)) AS base_mad,
        -- U throws on an empty sample, and if() evaluates both branches, so rows that are
        -- no_baseline anyway get a stand-in baseline.
        if(empty(base), [cur[1]], base) AS b,
        arrayConcat(arrayWithConstant(length(b), 0), arrayWithConstant(length(cur), 1)) AS idx,
        arrayReduce('mannWhitneyUTest(\'less\')',    arrayConcat(b, cur), idx).2 AS p_less,
        arrayReduce('mannWhitneyUTest(\'greater\')', arrayConcat(b, cur), idx).2 AS p_greater
    FROM (
        SELECT *,
            arrayFlatten(groupArray(cur) OVER w) AS base,
            groupArray(run_id) OVER w AS base_runs
        FROM (
            -- Every sample, not the per-run mean: U needs the distribution. Zeros are dropped,
            -- since producers write 0 for a metric they did not measure.
            SELECT run_id, benchmark_id, component, backend, arch, source_job,
                   any(name) AS name, any(input_shapes) AS input_shapes, min(run_ts) AS run_ts,
                   s.1 AS metric, groupArray(s.2) AS cur
            FROM v_benchmark_results_enriched
            ARRAY JOIN arrayFlatten(arrayMap(kv -> arrayMap(v -> (kv.1, v), kv.2),
                       CAST(samples, 'Array(Tuple(String, Array(Float64)))'))) AS s
            WHERE s.2 != 0
            GROUP BY run_id, benchmark_id, component, backend, arch, source_job, metric
        )
        WINDOW w AS (PARTITION BY component, benchmark_id, backend, arch, source_job, metric
                     ORDER BY run_ts, run_id ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
    ) AS r
    LEFT JOIN (SELECT *, true AS has_policy FROM benchmark_metric_policy) AS p USING (metric)
);

-- One row per run x benchmark: fail on a blocking regression, warn on any other
-- blocking/advisory signal. Reads the live verdicts, so a gate gets its answer right after ingest.
CREATE VIEW IF NOT EXISTS v_benchmark_gate AS
SELECT
    run_id, benchmark_id, component, backend, arch,
    any(name) AS name, any(input_shapes) AS input_shapes, max(ts) AS ts,
    multiIf(countIf(status = 'regressed' AND tier = 'blocking') > 0, 'fail',
            countIf(status IN ('regressed', 'suspected') AND tier IN ('blocking', 'advisory')) > 0, 'warn',
            countIf(status = 'no_baseline') = count(), 'no_baseline',
            'pass') AS gate_status,
    groupArrayIf(metric, status = 'regressed' AND tier = 'blocking') AS regressed_metrics,
    groupArrayIf(metric, status IN ('regressed', 'suspected') AND tier IN ('blocking', 'advisory')
                         AND NOT (status = 'regressed' AND tier = 'blocking')) AS warn_metrics,
    argMaxIf(metric, worse_pct, status IN ('regressed', 'suspected')) AS primary_metric,
    maxIf(worse_pct, status IN ('regressed', 'suspected')) AS worst_delta_pct
FROM v_benchmark_metric_verdicts
GROUP BY run_id, benchmark_id, component, backend, arch;

-- Verdicts as they were evaluated, per policy_version, for history that must not move when the
-- policy does. Read with FINAL: the refresh re-evaluates recent runs, so a run whose rows were
-- still arriving is replaced by its settled verdict.
CREATE TABLE IF NOT EXISTS benchmark_metric_verdicts
(
    run_id           UUID,
    benchmark_id     UUID,
    component        LowCardinality(String),
    backend          LowCardinality(String),
    arch             LowCardinality(String),
    source_job       LowCardinality(String),
    ts               DateTime,
    metric           LowCardinality(String),
    tier             LowCardinality(String),
    baseline_kind    LowCardinality(String),
    baseline_run_ids Array(UUID),
    n_baseline       UInt16,
    n_current        UInt16,
    baseline_value   Float64,
    current_value    Float64,
    delta_pct        Float64,
    worse_pct        Float64,
    score            Float64,
    floor_pct        Float64,
    status           LowCardinality(String),
    reason           LowCardinality(String),
    policy_version   String,
    evaluated_at     DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(evaluated_at)
PARTITION BY toYYYYMM(ts)
ORDER BY (component, run_id, benchmark_id, backend, arch, source_job, metric, baseline_kind,
          policy_version);

-- Refreshable, not insert-triggered: a run lands in several inserts, and each would see only its
-- own block while re-joining all history. The 3-day filter applies after the window, so
-- baselines still reach back 20 runs. Reads views, so the applier creates it after them.
CREATE MATERIALIZED VIEW IF NOT EXISTS benchmark_metric_verdicts_mv
REFRESH EVERY 30 MINUTE APPEND TO benchmark_metric_verdicts AS
SELECT
    run_id, benchmark_id, component, backend, arch, source_job, ts, metric, tier,
    baseline_kind, baseline_run_ids, n_baseline, n_current, baseline_value, current_value,
    delta_pct, worse_pct, score, floor_pct, status, reason,
    (SELECT hex(cityHash64(arraySort(groupArray(toString(
                (metric, tier, higher_is_better, floor_pct)))))) FROM benchmark_metric_policy)
        AS policy_version
FROM v_benchmark_metric_verdicts
WHERE ts >= now() - INTERVAL 3 DAY;
