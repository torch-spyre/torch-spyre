-- Failure forensics: join a test case's result to the CI run, PR, commits and runner behind it.
-- Views only; every fact comes from rows already written. Apply after 10-, 20-, 47- and 53-.

-- One row per artifact_results run_id: the CI run it came from and that run's context. run_key is
-- the Jenkins run when artifact_results names one (props['orch_run_key']), else the GHA attempt
-- parsed from props['run_url']; gha_run_key is the GHA attempt whenever a run_url exists. A run
-- with no pipeline_runs row yet (the GHA poller lags) keeps its keys and has NULL context.
-- failed_job_agents / failed_job_images are the runners and `image_id_*` labels of the attempt's
-- failed jobs: the best available answer to "where did this fail", as the failing job is not
-- recorded per case. base_sha is the PR base the run was started against, when the poller saw it.
CREATE VIEW IF NOT EXISTS v_result_run_context AS
WITH runs AS
(
    SELECT
        run_id,
        anyIf(props['run_url'], props['run_url'] != '')           AS run_url,
        anyIf(props['orch_run_key'], props['orch_run_key'] != '') AS orch_run_key,
        anyIf(props['source'], props['source'] != '')             AS source,
        greatest(1, maxIf(toUInt16OrZero(props['run_attempt']), props['run_attempt'] != '')) AS run_attempt,
        any(artifact_id)                                          AS artifact_id,
        max(ts)                                                   AS result_ts
    FROM artifact_results
    GROUP BY run_id
),
keyed AS
(
    SELECT
        *,
        if(run_url LIKE '%/actions/runs/%',
           concat('gha:', extract(run_url, 'github\\.com/([^/]+/[^/]+)/actions/runs/'), '/',
                  extract(run_url, '/actions/runs/([0-9]+)'), '#', toString(run_attempt)),
           '') AS gha_run_key
    FROM runs
),
jobs AS
(
    SELECT
        parent_run_key,
        groupUniqArrayIf(agent, result IN ('failure', 'timed_out') AND agent != '') AS failed_job_agents,
        groupUniqArrayIf(extract(props['labels'], 'image_id_[0-9a-z_]+'),
                         result IN ('failure', 'timed_out') AND props['labels'] LIKE '%image_id_%') AS failed_job_images,
        max(queue_ms) AS max_job_queue_ms
    FROM v_pipeline_runs
    WHERE pipeline_type = 'gha-job' AND parent_run_key IN (SELECT gha_run_key FROM keyed WHERE gha_run_key != '')
    GROUP BY parent_run_key
)
SELECT
    k.run_id AS run_id,
    k.artifact_id AS artifact_id,
    k.result_ts AS result_ts,
    k.source AS source,
    if(k.orch_run_key != '', k.orch_run_key, k.gha_run_key) AS run_key,
    k.gha_run_key AS gha_run_key,
    k.run_url AS run_url,
    k.run_attempt AS run_attempt,
    if(k.run_attempt > 1 AND k.gha_run_key != '',
       replaceRegexpOne(k.gha_run_key, '#[0-9]+$', concat('#', toString(k.run_attempt - 1))), '') AS retry_of,
    if(p.run_key != '', p.repo, g.repo) AS repo,
    if(p.pr_number != 0, p.pr_number, g.pr_number) AS pr_number,
    if(p.sha != '', p.sha, g.sha) AS head_sha,
    if(g.props['base_sha'] != '', g.props['base_sha'], p.props['base_sha']) AS base_sha,
    if(g.props['base_ref'] != '', g.props['base_ref'], p.props['base_ref']) AS base_ref,
    if(p.run_key != '', p.trigger_source, g.trigger_source) AS trigger_source,
    if(p.run_key != '', p.started_at, g.started_at) AS run_started_at,
    if(p.run_key != '', p.queue_ms, g.queue_ms) AS run_queue_ms,
    if(p.run_key != '', p.result, g.result) AS run_result,
    j.failed_job_agents AS failed_job_agents,
    j.failed_job_images AS failed_job_images,
    j.max_job_queue_ms AS max_job_queue_ms,
    (p.run_key != '' OR g.run_key != '') AS has_run_row
FROM keyed AS k
LEFT JOIN (SELECT * FROM v_pipeline_runs WHERE pipeline_type IN ('orchestrator', 'component-build', 'product-test', 'gha-workflow')
           AND run_key IN (SELECT if(orch_run_key != '', orch_run_key, gha_run_key) FROM keyed)) AS p
       ON p.run_key = if(k.orch_run_key != '', k.orch_run_key, k.gha_run_key)
LEFT JOIN (SELECT * FROM v_pipeline_runs WHERE pipeline_type = 'gha-workflow'
           AND run_key IN (SELECT gha_run_key FROM keyed WHERE gha_run_key != '')) AS g
       ON g.run_key = k.gha_run_key AND k.gha_run_key != ''
LEFT JOIN jobs AS j ON j.parent_run_key = k.gha_run_key AND k.gha_run_key != '';

-- Each case result with its run context and a normalised failure signature (numbers and addresses
-- masked, first line only), so the same assertion with different values groups together.
-- ALWAYS filter by name (and ts): unfiltered it reads every case result.
CREATE VIEW IF NOT EXISTS v_case_run_history AS
SELECT
    cr.ts AS ts,
    cr.run_id AS run_id,
    cr.component AS component,
    c.classname AS classname,
    c.name AS name,
    cr.status AS status,
    cr.props['result.retried'] AS retried,
    lower(substring(replaceRegexpAll(splitByChar('\n', cr.fail_message)[1], '0x[0-9a-fA-F]+|[0-9]+(\\.[0-9]+)?', '#'), 1, 160)) AS signature,
    x.run_key AS run_key,
    x.gha_run_key AS gha_run_key,
    x.run_attempt AS run_attempt,
    x.repo AS repo,
    x.pr_number AS pr_number,
    x.head_sha AS head_sha,
    x.base_sha AS base_sha,
    x.artifact_id AS artifact_id,
    x.failed_job_agents AS failed_job_agents,
    x.failed_job_images AS failed_job_images,
    x.has_run_row AS has_run_row
FROM test_case_runs AS cr
INNER JOIN test_cases AS c ON c.test_case_id = cr.test_case_id AND c.component = cr.component
LEFT JOIN v_result_run_context AS x ON x.run_id = cr.run_id;

-- Per case and failure signature: how often, where and when it failed. passes_between counts
-- passes inside [first_failed, last_failed]; > 0 means the failure comes and goes.
-- failing_runner_* are the runners of every failed job in the failing runs (the failing job is not recorded per
-- case, so a run with several failed jobs over-attributes); empty until the GHA poller has the run.
-- ALWAYS filter by name.
CREATE VIEW IF NOT EXISTS v_failure_recurrence AS
SELECT
    f.component AS component, f.classname AS classname, f.name AS name, f.signature AS signature,
    f.failing_results AS failing_results, f.failing_runs AS failing_runs, f.failing_prs AS failing_prs,
    f.failing_head_shas AS failing_head_shas, f.failing_runner_count AS failing_runner_count,
    f.failing_runner_sample AS failing_runner_sample,
    f.first_failed AS first_failed, f.last_failed AS last_failed,
    f.first_failed_run_key AS first_failed_run_key, f.last_failed_run_key AS last_failed_run_key,
    f.sample_prs AS sample_prs,
    arrayCount(t -> t > f.first_failed AND t < f.last_failed, p.pass_ts) AS passes_between,
    length(p.pass_ts) AS passes_total
FROM
(
    SELECT
        component, classname, name, signature,
        count() AS failing_results,
        uniqExact(run_id) AS failing_runs,
        uniqExactIf(pr_number, pr_number != 0) AS failing_prs,
        uniqExactIf(head_sha, head_sha != '') AS failing_head_shas,
        length(arrayDistinct(arrayFlatten(groupArray(failed_job_agents)))) AS failing_runner_count,
        arraySlice(arrayDistinct(arrayFlatten(groupArray(failed_job_agents))), 1, 5) AS failing_runner_sample,
        min(ts) AS first_failed, max(ts) AS last_failed,
        argMin(run_key, ts) AS first_failed_run_key, argMax(run_key, ts) AS last_failed_run_key,
        groupUniqArrayIf(5)(pr_number, pr_number != 0) AS sample_prs
    FROM v_case_run_history
    WHERE status IN ('failed', 'error')
    GROUP BY component, classname, name, signature
) AS f
LEFT JOIN
(
    SELECT component, classname, name, groupArray(ts) AS pass_ts
    FROM v_case_run_history
    WHERE status = 'passed'
    GROUP BY component, classname, name
) AS p ON p.component = f.component AND p.classname = f.classname AND p.name = f.name;

-- The candidate window for "which change introduced it": the last pass before each failure
-- signature's first failure (any PR) and that first failure, with both commits and the artifact
-- inputs (artifacts.sources) that differ. Inspect the range with
-- `git log last_pass_head_sha..first_fail_head_sha`; PR heads are not ancestors of each other, so
-- same_pr says whether the two runs share a PR. changed_sources is empty when either side has no
-- recorded artifact. ALWAYS filter by name.
CREATE VIEW IF NOT EXISTS v_failure_first_seen AS
SELECT
    w.component AS component, w.classname AS classname, w.name AS name, w.signature AS signature,
    w.first_failed AS first_failed, w.first_fail_run_key AS first_fail_run_key,
    w.first_fail_pr AS first_fail_pr, w.first_fail_head_sha AS first_fail_head_sha,
    w.first_fail_base_sha AS first_fail_base_sha,
    w.last_pass_ts AS last_pass_ts, w.last_pass_run_key AS last_pass_run_key,
    w.last_pass_pr AS last_pass_pr, w.last_pass_head_sha AS last_pass_head_sha,
    (w.first_fail_pr != 0 AND w.first_fail_pr = w.last_pass_pr) AS same_pr,
    arrayFilter(s -> NOT has(pa.srcs, s), fa.srcs) AS changed_sources
FROM
(
    SELECT
        f.component AS component, f.classname AS classname, f.name AS name, f.signature AS signature,
        f.first_failed AS first_failed, f.first_fail_run_key AS first_fail_run_key,
        f.first_fail_pr AS first_fail_pr, f.first_fail_head_sha AS first_fail_head_sha,
        f.first_fail_base_sha AS first_fail_base_sha, f.first_fail_artifact AS first_fail_artifact,
        argMaxIf(h.ts, h.ts, h.status = 'passed' AND h.ts < f.first_failed) AS last_pass_ts,
        argMaxIf(h.run_key, h.ts, h.status = 'passed' AND h.ts < f.first_failed) AS last_pass_run_key,
        argMaxIf(h.pr_number, h.ts, h.status = 'passed' AND h.ts < f.first_failed) AS last_pass_pr,
        argMaxIf(h.head_sha, h.ts, h.status = 'passed' AND h.ts < f.first_failed) AS last_pass_head_sha,
        argMaxIf(h.artifact_id, h.ts, h.status = 'passed' AND h.ts < f.first_failed) AS last_pass_artifact
    FROM
    (
        SELECT
            component, classname, name, signature,
            min(ts) AS first_failed,
            argMin(run_key, ts) AS first_fail_run_key, argMin(pr_number, ts) AS first_fail_pr,
            argMin(head_sha, ts) AS first_fail_head_sha, argMin(base_sha, ts) AS first_fail_base_sha,
            argMin(artifact_id, ts) AS first_fail_artifact
        FROM v_case_run_history
        WHERE status IN ('failed', 'error')
        GROUP BY component, classname, name, signature
    ) AS f
    INNER JOIN v_case_run_history AS h
            ON h.component = f.component AND h.classname = f.classname AND h.name = f.name
    GROUP BY component, classname, name, signature, first_failed, first_fail_run_key, first_fail_pr,
             first_fail_head_sha, first_fail_base_sha, first_fail_artifact
) AS w
LEFT JOIN (SELECT artifact_id, any(arrayMap(t -> concat(t.repo, '@', t.git_sha), sources)) AS srcs
           FROM artifacts GROUP BY artifact_id) AS fa ON fa.artifact_id = w.first_fail_artifact
LEFT JOIN (SELECT artifact_id, any(arrayMap(t -> concat(t.repo, '@', t.git_sha), sources)) AS srcs
           FROM artifacts GROUP BY artifact_id) AS pa ON pa.artifact_id = w.last_pass_artifact;
