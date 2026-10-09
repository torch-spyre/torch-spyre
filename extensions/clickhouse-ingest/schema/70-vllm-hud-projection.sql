-- vLLM benchmark results in upstream pytorch/test-infra's `oss_ci_benchmark_v3` shape, plus the
-- dropdown table upstream derives from it.
--
-- WHY THIS SHAPE. Matching upstream's record shape (an upstream contract we don't control) is
-- what lets the PyTorch HUD read our numbers with no query changes, only the DATABASE redirected.
--
-- THE TABLE NAMES ARE LOAD-BEARING. The HUD resolves `oss_ci_benchmark_v3` and
-- `oss_ci_benchmark_metadata` in TypeScript, and they must be real MergeTree tables: exposing
-- them as plain VIEWs fails with "Code 182: Storage View does not support PREWHERE" on
-- upstream's metadata query builder -- a partial failure where only the dropdowns break.
--
-- FED BY MATERIALIZED VIEW, NOT A SECOND INSERT: benchmark_runs is the one written perf fact,
-- and an MV's target being a real table satisfies the PREWHERE constraint above while keeping
-- one source of truth.
--
-- WHAT WE ADD. `run_id` is ours, not upstream's -- upstream has no artifact concept, so an
-- upstream-shaped table alone cannot join artifact_results.
--
-- MVs FIRE ON INSERT ONLY, so a definition change means re-inserting (upstream's own recipe: a
-- backfill INSERT beside the MV).

-- ── upstream's record table ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS oss_ci_benchmark_v3
(
    -- Our join key: uuid5(NS, "{source}|{external_run_id}|{arch}|{test_type}"), the same value
    -- artifact_results.run_id carries; on Jenkins, params.RUN_ID is already this uuid.
    run_id         UUID,

    -- SECONDS, not milliseconds: every upstream query reads this with toUnixTimestamp(), no intDiv.
    timestamp      Int64,
    schema_version LowCardinality(String) DEFAULT 'v3',

    -- A REGISTERED benchmark id, not the benchmark's own name -- the HUD routes
    -- /benchmark/v3/dashboard/<id> against this and refuses an unregistered one; the real name
    -- travels in benchmark.extra_info['benchmark_name'].
    name           String,

    repo           LowCardinality(String),
    head_branch    String,
    head_sha       String,
    workflow_id    Int64,
    run_attempt    UInt32 DEFAULT 0,
    job_id         Int64 DEFAULT 0,

    -- Upstream's full 11-field tuple; GPU fields are empty for a Spyre run, but trimming them
    -- breaks upstream's metadata MV, which reads runners[1] as (name=DEVICE, type=ARCH) for its
    -- arch fallback -- carrying unused fields is the cost of reading upstream's queries unmodified.
    runners        Array(Tuple(
                       name String, type String, cpu_info String, cpu_count UInt32,
                       mem_info String, avail_mem_in_gb UInt32, gpu_info String,
                       gpu_count UInt32, gpu_mem_info String, avail_gpu_mem_in_gb UInt32,
                       extra_info Map(String, String)
                   )),

    -- extra_info carries the keys the HUD reads by name: device, arch, hardware_type,
    -- use_compile, and `args` as a JSON STRING it JSONExtracts tensor_parallel_size/input_len/output_len from.
    benchmark      Tuple(name String, mode String, dtype String,
                         extra_info Map(String, String)),
    model          Tuple(name String, type String, backend String, origins Array(String),
                         extra_info Map(String, String)),
    inputs         Map(String, Tuple(dtype String, extra_info Map(String, String))),
    dependencies   Map(String, Tuple(repo String, branch String, sha String, version String,
                                     extra_info Map(String, String))),

    -- An array because a metric measured n times is n values: the HUD computes both an
    -- arithmetic and a geometric mean.
    metric         Tuple(name String, benchmark_values Array(Float32), target_value Float32,
                         extra_info Map(String, String))
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(toDateTime(timestamp))
-- Upstream's sort key, kept: this table serves upstream's time-window scan; reads by run go to benchmark_runs.
ORDER BY (timestamp, head_branch, head_sha, workflow_id, job_id);

-- One row per (benchmark run, metric): benchmark_runs holds a metric->samples Map, fanned out
-- here since the HUD wants one row per metric.
--
-- The props this reads (repo, head_branch, head_sha, workflow_id, arch, hardware_type) must be
-- written onto benchmark_runs by the ingest: this view cannot see the CI coordinates itself.
CREATE MATERIALIZED VIEW IF NOT EXISTS oss_ci_benchmark_v3_mv TO oss_ci_benchmark_v3 AS
SELECT
    r.run_id                               AS run_id,
    toInt64(toUnixTimestamp(r.ts))         AS timestamp,
    'v3'                                   AS schema_version,
    'spyre_e2e_benchmark'                  AS name,
    -- owner/repo, as upstream stores it: the HUD sends it as the repo filter and builds GitHub links on it.
    if(position(r.props['repo'], '/') > 0, r.props['repo'],
       concat('torch-spyre/', r.props['repo']))  AS repo,
    r.props['head_branch']                 AS head_branch,
    r.props['head_sha']                    AS head_sha,
    -- A Jenkins run has no workflow_id; 0 would file every such run as one HUD commit.
    if(toInt64OrZero(r.props['workflow_id']) != 0, toInt64OrZero(r.props['workflow_id']),
       toInt64(bitAnd(sipHash64(r.run_id), 0xFFFFFFFFFFFFF))) AS workflow_id,
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
-- Keeps oss_ci_benchmark_names/_branches fast, and is where the PREWHERE lands. Copied from
-- upstream's oss_ci_benchmark_v3_materialized_views/schema.sql, minus the replicated-engine args
-- a single-node server does not take.
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
    if(
        empty(tupleElement(runners[1], 'name')),
        if(
            empty(tupleElement(benchmark, 'extra_info')['device']),
            'cpu',
            tupleElement(benchmark, 'extra_info')['device']
        ),
        tupleElement(runners[1], 'name')
    ) AS device,
    if(
        empty(tupleElement(runners[1], 'type')),
        if(
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

-- ── the same records, keyed by artifact tag ─────────────────────────────────────────────────
-- The HUD compares and trends by (branch, commit). Here branch = tag family and commit = tag, so
-- a branch's trend runs across a family's tags and a commit pair compares two tags. Family
-- 'artifact' files every linked run under its artifact (tag = artifact_id12), so an untagged
-- artifact still compares with the one before it. Channel tags (tag = family: `nightly`,
-- `release`) are left out: they move to each new build, so they have no place in a trend.
--
-- Refreshed rather than insert-fed: a tag lands after the run that measured its artifact. Real
-- tables, not views, for the same PREWHERE reason as above.
CREATE TABLE IF NOT EXISTS oss_ci_benchmark_v3_by_tag
(
    run_id         UUID,
    timestamp      Int64,
    schema_version LowCardinality(String),
    name           String,
    repo           LowCardinality(String),
    head_branch    String,
    head_sha       String,
    -- 52-bit hash of (family, tag): the HUD keys a commit on it and parses it as a JS number.
    workflow_id    Int64,
    run_attempt    UInt32,
    job_id         Int64,
    runners        Array(Tuple(
                       name String, type String, cpu_info String, cpu_count UInt32,
                       mem_info String, avail_mem_in_gb UInt32, gpu_info String,
                       gpu_count UInt32, gpu_mem_info String, avail_gpu_mem_in_gb UInt32,
                       extra_info Map(String, String)
                   )),
    benchmark      Tuple(name String, mode String, dtype String,
                         extra_info Map(String, String)),
    model          Tuple(name String, type String, backend String, origins Array(String),
                         extra_info Map(String, String)),
    inputs         Map(String, Tuple(dtype String, extra_info Map(String, String))),
    dependencies   Map(String, Tuple(repo String, branch String, sha String, version String,
                                     extra_info Map(String, String))),
    metric         Tuple(name String, benchmark_values Array(Float32), target_value Float32,
                         extra_info Map(String, String)),
    -- The run's own coordinates, displaced from head_sha/workflow_id above.
    commit_sha         String,
    source_workflow_id Int64,
    artifact_id    UUID,
    artifact_id12  String,
    image          String,
    tag            String,
    tag_family     LowCardinality(String)
)
ENGINE = MergeTree()
ORDER BY (timestamp, head_branch, head_sha, workflow_id, job_id);

CREATE MATERIALIZED VIEW IF NOT EXISTS oss_ci_benchmark_v3_by_tag_mv
REFRESH EVERY 15 MINUTE TO oss_ci_benchmark_v3_by_tag AS
SELECT
    o.run_id, o.timestamp, o.schema_version, o.name, o.repo,
    t.tag_family                                                    AS head_branch,
    t.tag                                                           AS head_sha,
    toInt64(bitAnd(sipHash64(t.tag_family, t.tag), 0xFFFFFFFFFFFFF)) AS workflow_id,
    o.run_attempt, o.job_id, o.runners, o.benchmark, o.model, o.inputs, o.dependencies, o.metric,
    o.head_sha                                                      AS commit_sha,
    o.workflow_id                                                   AS source_workflow_id,
    t.artifact_id, t.artifact_id12, t.image, t.tag, t.tag_family
FROM oss_ci_benchmark_v3 AS o
INNER JOIN
(
    SELECT run_id, artifact_id, artifact_id12, image, tag, tag_family
    FROM v_benchmark_run_artifacts
    WHERE tag != '' AND tag != tag_family
    UNION DISTINCT
    SELECT run_id, artifact_id, artifact_id12, image, artifact_id12 AS tag, 'artifact' AS tag_family
    FROM v_benchmark_run_artifacts
) AS t ON t.run_id = o.run_id;

CREATE TABLE IF NOT EXISTS oss_ci_benchmark_metadata_by_tag
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

-- A refresh swaps its target whole, which fires no insert-triggered MV, so this one refreshes too.
CREATE MATERIALIZED VIEW IF NOT EXISTS oss_ci_benchmark_metadata_by_tag_mv
REFRESH EVERY 15 MINUTE DEPENDS ON oss_ci_benchmark_v3_by_tag_mv
TO oss_ci_benchmark_metadata_by_tag AS
SELECT DISTINCT
    repo,
    tupleElement(benchmark, 'name')                AS benchmark_name,
    tupleElement(benchmark, 'dtype')               AS benchmark_dtype,
    tupleElement(benchmark, 'mode')                AS benchmark_mode,
    tupleElement(model, 'name')                    AS model_name,
    tupleElement(model, 'backend')                 AS model_backend,
    tupleElement(benchmark, 'extra_info')['device'] AS device,
    tupleElement(benchmark, 'extra_info')['arch']   AS arch,
    tupleElement(metric, 'name')                   AS metric_name,
    head_branch,
    head_sha,
    toUInt64(workflow_id)                          AS workflow_id,
    toUInt64(timestamp)                            AS timestamp
FROM oss_ci_benchmark_v3_by_tag;
