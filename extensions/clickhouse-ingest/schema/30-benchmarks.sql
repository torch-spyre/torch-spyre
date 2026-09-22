-- Benchmark schema v2: a dimension + fact pair, the same split test_cases /
-- test_case_runs uses. Read 10-functional-tests.sql first -- the identity rules, the
-- props/tags convention and the run_id contract are stated there, not repeated here.
--
-- Scope: torch-spyre op / kernel / model benchmarks (spyre-perf-suite, plus the sendnn
-- and CPU baselines it measures against -- see `backend`). vLLM / spyre-inference perf
-- stays in results_v3 + run_metadata: those are the schema the upstream PyTorch HUD
-- queries read, and reshaping them would fork SQL we re-mount from upstream on every
-- bump. It reaches this layer by carrying run_id and joining through artifact_results.
--
-- Replaces the v1 benchmark_runs + perf_benchmarks + perf_kernels, sendnn_runs +
-- sendnn_benchmarks, and loz_system_performance_vllm.

CREATE TABLE IF NOT EXISTS benchmarks
(
    ts           DateTime DEFAULT now(),

    -- uuid5(NAMESPACE, "name|sorted(tags)|record_type,config_name,input_shapes,
    -- run_mode,kernel_name,is_total"). tags are in the hash, so they must be sorted
    -- before hashing or one benchmark mints two identities. The discriminators are in
    -- the hash rather than only in props because one operation_name occurs at several
    -- record_types (granite as both model and op), so name+tags alone merges
    -- different benchmarks into one.
    benchmark_id UUID,

    -- Which producer's suite this benchmark belongs to, as in test_cases. torch-spyre's
    -- op harness and vLLM's bench share one namespace and collide the day either names a
    -- benchmark `latency`; since benchmark_id is a content hash, that collision would
    -- silently merge two trend lines. In the hash, so one name under two components is
    -- two identities.
    component    LowCardinality(String),

    name         String,
    tags         Array(LowCardinality(String)),

    -- What distinguishes one benchmark from another, per producer: record_type
    -- (op|model|kernel), config_name, input_shapes, run_mode, kernel_name, is_total,
    -- batch_size, prompt_length, custom_op_file. These are configuration, not
    -- measurement. A Map keeps one dimension serving producers whose identity tuples
    -- disagree.
    props        Map(LowCardinality(String), String),

    CONSTRAINT chk_component CHECK component != '',
    CONSTRAINT chk_name      CHECK name != ''
)
ENGINE = MergeTree()
-- name leads, not benchmark_id: every read picks a benchmark by name, and benchmark_id is
-- only arrived at through it or a join, so name-first prunes and clusters a benchmark's
-- variants. component leads it, matching test_cases: reads scope to a producer first and
-- the LowCardinality prefix prunes before the name range scan. benchmark_id stays in the
-- key to keep the row unique.
ORDER BY (component, name, benchmark_id);

-- benchmark_runs, matching test_case_runs: <dimension>_runs is the convention for the
-- observation half of a pair. The v1 table of this name lives in the `spyre` database, so
-- there is no collision.
CREATE TABLE IF NOT EXISTS benchmark_runs
(
    ts           DateTime DEFAULT now(),

    -- The only two foreign keys. run_id carries all run context (arch, branch, commit,
    -- pr, tag) through artifact_results; benchmark_id carries all benchmark identity. A
    -- column earns a place on a fact only if it is an input to that fact's own key hash
    -- (so it cannot disagree) and a leading ORDER BY column (so it earns pruning).
    run_id      UUID,
    benchmark_id UUID,

    -- Which implementation produced these numbers. This is the axis that makes
    -- perf_kernels coherent: its torch_spyre_ms and sendnn_ms were never one
    -- measurement, they are the same kernel on two backends. As rows they compare by
    -- self-join on benchmark_id, and v1's stored `ratio` becomes derived rather than a
    -- third column that can disagree with the two it divides.
    -- component is carried here too, as test_case_runs does, and leads the sort key so a
    -- per-producer read prunes instead of scanning every run.
    component    LowCardinality(String),
    backend      LowCardinality(String),

    -- Metric key -> its samples, keys verbatim from the producer (total_duration_ms,
    -- cpu_ms, spyre_ms, kernel_mean_ms, compile_ms, mem_size_mb, ratio, ...).
    -- A Map, not columns, because the sparsity is per record_type and no column set
    -- fits: mem_size_mb is set on op rows and never on model rows, batch_size the
    -- inverse. A wide fact is majority-NULL by construction and needs a DDL change per
    -- new metric. Units stay encoded in the key suffix.
    --
    -- An array per key, not one Float64: a metric measured n times is n values. Storing
    -- one froze every statistic at ingest -- a geometric mean over a single value equals
    -- the arithmetic mean by construction. Variance, percentiles and a real geomean are
    -- recomputable only if the samples survive. Readers wanting one number take the mean
    -- via v_benchmark_results_enriched, which reduces this to a scalar Map under the same
    -- column name.
    measurements Map(LowCardinality(String), Array(Float64)),

    -- n behind each mean, as the producer reported it. Redundant with
    -- length(measurements[k]) when the producer sends samples, and the only source when
    -- it sends a pre-averaged number instead -- which is why it is not derived. Without a
    -- sample count a delta cannot be separated from noise. 0 = the producer did not say.
    -- ONE scalar for a row whose metrics may carry different n (avg_latency over 30 runs,
    -- p99 over 1): it is the SUM across the merged entries, since each entry contributes its
    -- own distinct count. So it is neither a per-metric n nor a bound on one -- two entries
    -- at 30 and 1 store 31. Use length(measurements[k]) where the producer sent samples.
    iterations   UInt32 DEFAULT 0,

    props        Map(LowCardinality(String), String),

    -- regression_status is deliberately absent: a stored verdict with no recorded
    -- baseline cannot be checked against the data it summarises. Derived in
    -- v_benchmark_regression against an explicit baseline run_id, under the same column
    -- name so the dashboard contract holds.
    CONSTRAINT chk_measurements CHECK length(measurements) > 0
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (component, run_id, benchmark_id, backend);
-- Deliberately no skip index on benchmark_id, though every trend and regression view
-- groups by it across runs. A bloom filter only prunes when matching rows are contiguous
-- in sort order; here every benchmark_id recurs in many runs, so almost every granule
-- holds some row for any given benchmark -- it would cost storage and prune nothing. If
-- per-benchmark history becomes a hot path the fix is a projection or a benchmark_id-first
-- ORDER BY, not an index.
