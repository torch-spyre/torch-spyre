-- Benchmark schema v2: a dimension + fact pair, the same split test_cases/test_case_runs uses.
-- Read 10-functional-tests.sql first -- identity rules, props/tags and run_id are stated there.
--
-- Scope: torch-spyre op/kernel/model benchmarks (spyre-perf-suite, plus sendnn and CPU baselines
-- it measures against -- see `backend`). vLLM/spyre-inference perf stays in results_v3 +
-- run_metadata (the schema upstream PyTorch HUD reads) and joins in via run_id/artifact_results.
--
-- Replaces v1's benchmark_runs + perf_benchmarks + perf_kernels, sendnn_runs +
-- sendnn_benchmarks, and loz_system_performance_vllm.

CREATE TABLE IF NOT EXISTS benchmarks
(
    ts           DateTime DEFAULT now(),

    -- uuid5(name|sorted(tags)|record_type,config_name,input_shapes,run_mode,kernel_name,is_total).
    -- Discriminators are IN the hash, not just props: one operation_name recurs across
    -- record_types (granite as both model and op), so name+tags alone would merge them.
    benchmark_id UUID,

    -- Which producer's suite this belongs to, as in test_cases -- IN the hash, since torch-spyre's
    -- op harness and vLLM's bench share a namespace and could otherwise collide on one name.
    component    LowCardinality(String),

    name         String,
    tags         Array(LowCardinality(String)),

    -- What distinguishes one benchmark from another, per producer (record_type, config_name,
    -- input_shapes, run_mode, kernel_name, is_total, batch_size, prompt_length, ...) -- a Map
    -- since producers' identity tuples disagree.
    props        Map(LowCardinality(String), String),

    CONSTRAINT chk_component CHECK component != '',
    CONSTRAINT chk_name      CHECK name != ''
)
ENGINE = MergeTree()
-- name leads: every read picks a benchmark by name; component leads that (matching test_cases)
-- so the LowCardinality prefix prunes before the name scan; benchmark_id keeps the row unique.
ORDER BY (component, name, benchmark_id);

-- benchmark_runs, matching test_case_runs' <dimension>_runs convention (no collision with v1's
-- table of this name, which lives in the `spyre` database).
CREATE TABLE IF NOT EXISTS benchmark_runs
(
    ts           DateTime DEFAULT now(),

    -- The only two foreign keys: run_id carries all run context via artifact_results,
    -- benchmark_id all benchmark identity.
    run_id      UUID,
    benchmark_id UUID,

    -- Which implementation produced these numbers -- the axis perf_kernels' torch_spyre_ms vs
    -- sendnn_ms compares by self-join on benchmark_id, so v1's stored `ratio` becomes derived.
    -- component is carried here too and leads the sort key for per-producer pruning.
    component    LowCardinality(String),
    backend      LowCardinality(String),

    -- Metric key -> its samples, verbatim from the producer (total_duration_ms, cpu_ms,
    -- spyre_ms, kernel_mean_ms, compile_ms, mem_size_mb, ratio, ...). A Map, not columns:
    -- sparsity is per record_type (mem_size_mb on op rows, batch_size the inverse), so no
    -- fixed column set fits. An array per key, not one Float64: a metric measured n times is n
    -- values, so variance/percentiles/geomean stay recomputable; readers wanting one number use
    -- v_benchmark_results_enriched, which reduces this to a scalar Map under the same name.
    measurements Map(LowCardinality(String), Array(Float64)),

    -- n behind each mean, as reported -- the only source when the producer sends a pre-averaged
    -- number rather than samples, so a delta can be told from noise (0 = not stated). One
    -- scalar per row is the SUM across merged entries (each contributing its own count), so use
    -- length(measurements[k]) instead wherever the producer sent samples.
    iterations   UInt32 DEFAULT 0,

    props        Map(LowCardinality(String), String),

    -- regression_status is deliberately absent: a stored verdict with no baseline can't be
    -- checked. Derived in v_benchmark_regression against an explicit baseline run_id.
    CONSTRAINT chk_measurements CHECK length(measurements) > 0
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (component, run_id, benchmark_id, backend);
-- No skip index on benchmark_id, though every trend view groups by it: a bloom filter only
-- prunes contiguous matches, and a benchmark_id recurs across nearly every granule. A
-- projection or benchmark_id-first ORDER BY is the fix if per-benchmark history gets hot.
