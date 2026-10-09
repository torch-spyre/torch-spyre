-- A non-APPEND refresh swaps its target by EXCHANGE TABLES, which this server rejects
-- (renameat2), so the by_tag MVs are recreated in their APPEND / insert-triggered form.
DROP VIEW IF EXISTS oss_ci_benchmark_metadata_by_tag_mv;

DROP VIEW IF EXISTS oss_ci_benchmark_v3_by_tag_mv;

TRUNCATE TABLE oss_ci_benchmark_metadata_by_tag;

TRUNCATE TABLE oss_ci_benchmark_v3_by_tag;

CREATE MATERIALIZED VIEW oss_ci_benchmark_metadata_by_tag_mv
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

-- On an empty database the view this reads does not exist yet; the applier creates the MV later.
-- IF TABLE EXISTS: v_benchmark_run_artifacts
CREATE MATERIALIZED VIEW oss_ci_benchmark_v3_by_tag_mv
REFRESH EVERY 15 MINUTE APPEND TO oss_ci_benchmark_v3_by_tag AS
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
) AS t ON t.run_id = o.run_id
WHERE sipHash128(o.run_id, o.benchmark, o.model, tupleElement(o.metric, 'name'), t.tag_family, t.tag)
    NOT IN (SELECT sipHash128(run_id, benchmark, model, tupleElement(metric, 'name'), tag_family, tag)
            FROM oss_ci_benchmark_v3_by_tag);
