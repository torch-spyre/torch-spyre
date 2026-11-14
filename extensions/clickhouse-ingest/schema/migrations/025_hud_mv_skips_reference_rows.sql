-- RERUNNABLE
-- oss_ci_benchmark_v3_mv skips comparison-target rows (run.ref_only), which share benchmark_runs
-- with measurements. Rows already projected are kept; no such rows existed before this migration.
DROP VIEW IF EXISTS oss_ci_benchmark_v3_mv;

CREATE MATERIALIZED VIEW oss_ci_benchmark_v3_mv TO oss_ci_benchmark_v3 AS
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
-- Measured model-level rows only: predicted backends and experiment subjects share these tables.
-- The run.ref_only / run.record_type test is v_benchmark_runs_measured's, which an MV cannot read.
WHERE r.component = 'spyre-inference' AND r.backend = 'spyre' AND b.props['record_type'] = 'model'
  AND NOT startsWith(r.props['source_file'], 'optbench-')
  AND r.props['run.ref_only'] != '1' AND r.props['run.record_type'] != 'reference';
