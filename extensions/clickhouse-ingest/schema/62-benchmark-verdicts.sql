-- Benchmark regression verdicts, derived in SQL from benchmark_runs: a per-metric policy and a
-- refreshable MV that keeps verdict history. The verdict and gate views are in
-- 85-benchmark-verdict-views.sql (they read result invalidations, declared after this file).
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
    ('request_throughput',      'advisory',      true,  5),
    -- Accuracy evals (test_type quality): higher is better, unlike the time-like default.
    ('exact_match',             'advisory',      true,  2),
    ('acc_norm',                'advisory',      true,  2),
    ('lm_eval_acc',             'advisory',      true,  2),
    ('rouge1',                  'advisory',      true,  2),
    ('rouge2',                  'advisory',      true,  2),
    ('rougeL',                  'advisory',      true,  2),
    ('bleu_4',                  'advisory',      true,  2),
    ('meteor',                  'advisory',      true,  2),
    ('ndcg_at_10',              'advisory',      true,  2),
    ('spearman',                'advisory',      true,  2),
    ('accuracy',                'advisory',      true,  2),
    ('v_measure',               'advisory',      true,  2),
    ('retrieval_pct',           'advisory',      true,  2)
);

-- Run purposes whose results form baselines. A run with another purpose (pr-check, release,
-- investigate, ...) is judged against the baseline but never joins it, and its verdict is capped
-- at `suspected`. A run with no run.purpose prop is treated as eligible.
CREATE VIEW IF NOT EXISTS benchmark_baseline_purposes AS
SELECT * FROM values('purpose String', 'nightly', 'weekly', 'merge');

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
