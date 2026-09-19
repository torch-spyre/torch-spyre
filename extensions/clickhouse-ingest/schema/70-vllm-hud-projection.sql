-- vLLM benchmark results in upstream pytorch/test-infra's `oss_ci_benchmark_v3` shape, plus
-- the dropdown table upstream derives from it.
--
-- WHY THIS SHAPE. vLLM perf reaches us through vLLM's own bench harness, so the record shape
-- is an upstream contract we do not control. Matching it is what lets the PyTorch HUD read
-- our numbers with no query changes, with only the DATABASE redirected.
--
-- THE TABLE NAMES ARE LOAD-BEARING. The HUD resolves these tables in TypeScript, not only in
-- its .sql files, and its CLICKHOUSE_BENCHMARK_DATABASE setting redirects the database while
-- keeping the names. They must be spelled `oss_ci_benchmark_v3` and
-- `oss_ci_benchmark_metadata`, and must be real MergeTree tables: exposing the names as plain
-- VIEWs fails with "Code 182: Storage View does not support PREWHERE", because upstream's
-- metadata query builder emits one. That failure is partial and so easy to misread -- the
-- saved .sql queries carry no PREWHERE and keep working, so only the dropdowns break.
--
-- FED BY MATERIALIZED VIEW, NOT A SECOND INSERT. benchmark_runs is the one written perf fact
-- and both tables here are projections of it. An MV's target is a real table, satisfying the
-- PREWHERE constraint above while keeping a single source of truth and a single insert.
--
-- WHAT WE ADD. `run_id` is ours, not upstream's. Upstream has no artifact concept -- its
-- `dependencies` Map is a provenance hint, not a content identity -- so an upstream-shaped
-- table alone cannot join artifact_results. This one column is the difference between
-- HUD-compatible and HUD-compatible-and-joinable.
--
-- MVs FIRE ON INSERT ONLY, so these tables cannot be rebuilt from rows already in
-- benchmark_runs; a definition change means re-inserting. Upstream ships a backfill INSERT
-- beside its own MV for this reason, and the same recipe applies here.

-- ── upstream's record table ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS oss_ci_benchmark_v3
(
    -- Our join key: uuid5(NS, "{source}|{external_run_id}|{arch}|{test_type}"), the same value
    -- artifact_results.run_id carries. On the Jenkins path params.RUN_ID is already this uuid
    -- -- pass it verbatim; re-hashing it mints a third identity that joins to nothing.
    run_id         UUID,

    -- SECONDS, not milliseconds. Every upstream query reads this with toUnixTimestamp() /
    -- fromUnixTimestamp() and no intDiv, so milliseconds put every row centuries in the future
    -- and each query silently returns nothing.
    timestamp      Int64,
    schema_version LowCardinality(String) DEFAULT 'v3',

    -- A REGISTERED benchmark id, not the benchmark's own name: the HUD routes
    -- /benchmark/v3/dashboard/<id> against this and refuses an unregistered one, so a
    -- per-benchmark name makes the dashboard unreachable. The real name travels in
    -- benchmark.extra_info['benchmark_name'].
    name           String,

    repo           LowCardinality(String),
    head_branch    String,
    head_sha       String,
    workflow_id    Int64,
    run_attempt    UInt32 DEFAULT 0,
    job_id         Int64 DEFAULT 0,

    -- Upstream's full 11-field tuple. The GPU fields are empty for a Spyre run, but trimming
    -- them breaks upstream's own metadata MV, which reads runners[1].'cpu_info' as an arch
    -- fallback. Carrying unused fields is the cost of reading upstream's queries unmodified.
    -- runners[1] is (name = DEVICE, type = ARCH): upstream's MV and its _llms query both read
    -- it that way and the dashboard filters on device=/arch=, so swapping the two empties the
    -- page with no error.
    runners        Array(Tuple(
                       name String, type String, cpu_info String, cpu_count UInt32,
                       mem_info String, avail_mem_in_gb UInt32, gpu_info String,
                       gpu_count UInt32, gpu_mem_info String, avail_gpu_mem_in_gb UInt32,
                       extra_info Map(String, String)
                   )),

    -- extra_info carries the keys the HUD reads by name: device, arch, hardware_type,
    -- use_compile, and `args` as a JSON STRING it JSONExtracts tensor_parallel_size /
    -- input_len / output_len out of.
    benchmark      Tuple(name String, mode String, dtype String,
                         extra_info Map(String, String)),
    model          Tuple(name String, type String, backend String, origins Array(String),
                         extra_info Map(String, String)),
    inputs         Map(String, Tuple(dtype String, extra_info Map(String, String))),
    dependencies   Map(String, Tuple(repo String, branch String, sha String, version String,
                                     extra_info Map(String, String))),

    -- An array because a metric measured n times is n values: the HUD computes both an
    -- arithmetic and a geometric mean, which can only differ if the samples survive.
    metric         Tuple(name String, benchmark_values Array(Float32), target_value Float32,
                         extra_info Map(String, String))
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(toDateTime(timestamp))
-- Upstream's sort key, kept: this table exists to serve upstream's time-series scan, and its
-- queries filter on the time window first. Reads by run go to benchmark_runs instead.
ORDER BY (timestamp, head_branch, head_sha, workflow_id, job_id);

-- One row per (benchmark run, metric): benchmark_runs holds a metric->samples Map, and the HUD
-- wants one row per metric, so the Map is fanned out here.
--
-- The props this reads (repo, head_branch, head_sha, workflow_id, arch, hardware_type) must be
-- written onto benchmark_runs by the ingest, not re-derived here: this view cannot see the CI
-- coordinates, and guessing them would put a wrong commit on a chart.
CREATE MATERIALIZED VIEW IF NOT EXISTS oss_ci_benchmark_v3_mv TO oss_ci_benchmark_v3 AS
SELECT
    r.run_id                               AS run_id,
    toInt64(toUnixTimestamp(r.ts))         AS timestamp,
    'v3'                                   AS schema_version,
    'spyre_e2e_benchmark'                  AS name,
    r.props['repo']                        AS repo,
    r.props['head_branch']                 AS head_branch,
    r.props['head_sha']                    AS head_sha,
    toInt64OrZero(r.props['workflow_id'])  AS workflow_id,
    toUInt32OrZero(r.props['run_attempt']) AS run_attempt,
    toInt64OrZero(r.props['job_id'])       AS job_id,
    [(
        r.backend, r.props['arch'], '', toUInt32(0), '', toUInt32(0), '', toUInt32(0), '',
        toUInt32(0), CAST(map(), 'Map(String, String)')
    )]                                     AS runners,
    (
        'spyre_e2e_benchmark',
        b.props['run_mode'],
        b.props['dtype'],
        map(
            'benchmark_name', b.name,
            'device', r.backend,
            'arch', r.props['arch'],
            'hardware_type', r.props['hardware_type'],
            'use_compile', b.props['use_compile'],
            'args', concat(
                '{"tensor_parallel_size":"', b.props['tensor_parallel'],
                '","input_len":"', b.props['input_len'],
                '","output_len":"', b.props['output_len'], '"}'
            )
        )
    )                                      AS benchmark,
    (
        b.props['model'], 'llm', r.backend, ['huggingface'],
        CAST(map(), 'Map(String, String)')
    )                                      AS model,
    CAST(map(), 'Map(String, Tuple(dtype String, extra_info Map(String, String)))') AS inputs,
    CAST(map(), 'Map(String, Tuple(repo String, branch String, sha String, version String, extra_info Map(String, String)))') AS dependencies,
    (
        m.1, arrayMap(x -> toFloat32(x), m.2), toFloat32(0),
        CAST(map(), 'Map(String, String)')
    )                                      AS metric
FROM benchmark_runs AS r
INNER JOIN benchmarks AS b USING (benchmark_id)
ARRAY JOIN arrayZip(mapKeys(r.measurements), mapValues(r.measurements)) AS m
WHERE r.component = 'spyre-inference';

-- ── upstream's dropdown table, its own definition ───────────────────────────────────────────
-- Exists to keep oss_ci_benchmark_names / _branches fast, and it is where the PREWHERE lands.
-- Copied from upstream's oss_ci_benchmark_v3_materialized_views/schema.sql; the only change is
-- dropping the replicated-engine arguments, which a single-node server does not take.
CREATE TABLE IF NOT EXISTS oss_ci_benchmark_metadata
(
    repo            String,
    benchmark_name  String,
    benchmark_dtype String,
    benchmark_mode  String,
    model_name      String,
    model_backend   String,
    device          String,
    arch            String,
    metric_name     String,
    head_branch     String,
    head_sha        String,
    workflow_id     UInt64,
    timestamp       UInt64
)
ENGINE = MergeTree()
ORDER BY (repo, benchmark_name, benchmark_dtype, benchmark_mode, model_name, model_backend,
          device, arch, metric_name, head_branch, workflow_id, timestamp);

CREATE MATERIALIZED VIEW IF NOT EXISTS oss_ci_benchmark_metadata_mv
TO oss_ci_benchmark_metadata AS
SELECT
    repo AS repo,
    tupleElement(benchmark, 'name')  AS benchmark_name,
    tupleElement(benchmark, 'dtype') AS benchmark_dtype,
    tupleElement(benchmark, 'mode')  AS benchmark_mode,
    tupleElement(model, 'name')      AS model_name,
    tupleElement(model, 'backend')   AS model_backend,
    IF(
        empty(tupleElement(runners[1], 'name')),
        IF(
            empty(tupleElement(benchmark, 'extra_info')['device']),
            'cpu',
            tupleElement(benchmark, 'extra_info')['device']
        ),
        tupleElement(runners[1], 'name')
    ) AS device,
    IF(
        empty(tupleElement(runners[1], 'type')),
        IF(
            empty(tupleElement(benchmark, 'extra_info')['arch']),
            tupleElement(runners[1], 'cpu_info'),
            tupleElement(benchmark, 'extra_info')['arch']
        ),
        tupleElement(runners[1], 'type')
    ) AS arch,
    tupleElement(metric, 'name') AS metric_name,
    head_branch AS head_branch,
    head_sha    AS head_sha,
    workflow_id AS workflow_id,
    timestamp   AS timestamp
FROM oss_ci_benchmark_v3
WHERE tupleElement(benchmark, 'name') != 'sccache_stats';
