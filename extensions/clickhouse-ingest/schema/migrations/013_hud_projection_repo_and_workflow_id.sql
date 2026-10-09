-- oss_ci_benchmark_v3_mv now writes repo as owner/repo (the HUD filters on it, and the ingest's
-- bare name matched nothing) and gives a Jenkins run, which has no workflow_id, one hashed from
-- run_id (under 0 the HUD read every Jenkins run as one commit). Both tables are a pure
-- projection of benchmark_runs, so they are rebuilt whole; oss_ci_benchmark_metadata_mv refires
-- on the insert. As in 010, the MV's creation time is the cutoff, so no row lands twice.
DROP VIEW IF EXISTS oss_ci_benchmark_v3_mv;

TRUNCATE TABLE oss_ci_benchmark_metadata;

TRUNCATE TABLE oss_ci_benchmark_v3;

CREATE MATERIALIZED VIEW oss_ci_benchmark_v3_mv TO oss_ci_benchmark_v3 AS
SELECT
    r.run_id                               AS run_id,
    toInt64(toUnixTimestamp(r.ts))         AS timestamp,
    'v3'                                   AS schema_version,
    'spyre_e2e_benchmark'                  AS name,
    if(position(r.props['repo'], '/') > 0, r.props['repo'],
       concat('torch-spyre/', r.props['repo']))  AS repo,
    r.props['head_branch']                 AS head_branch,
    r.props['head_sha']                    AS head_sha,
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

INSERT INTO oss_ci_benchmark_v3
    (run_id, timestamp, schema_version, name, repo, head_branch, head_sha, workflow_id, run_attempt,
     job_id, runners, benchmark, model, inputs, dependencies, metric)
SELECT
    r.run_id                               AS run_id,
    toInt64(toUnixTimestamp(r.ts))         AS timestamp,
    'v3'                                   AS schema_version,
    'spyre_e2e_benchmark'                  AS name,
    if(position(r.props['repo'], '/') > 0, r.props['repo'],
       concat('torch-spyre/', r.props['repo']))  AS repo,
    r.props['head_branch']                 AS head_branch,
    r.props['head_sha']                    AS head_sha,
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
WHERE r.component = 'spyre-inference'
  AND r.ts < (
    SELECT metadata_modification_time FROM system.tables
    WHERE database = currentDatabase() AND name = 'oss_ci_benchmark_v3_mv'
  );
