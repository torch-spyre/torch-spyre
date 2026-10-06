-- Run-level views over pipeline_runs: CI stability, gate outcomes and where the time goes.

-- One row per run, latest phase only, so readers never need FINAL.
CREATE VIEW IF NOT EXISTS v_pipeline_runs AS
SELECT *
FROM pipeline_runs FINAL;

-- Orchestrator grain: each run with the child builds that failed under it and one outcome class.
-- own_component / other_component split on whether the PR's own repo is among the failed
-- builds; other_component is usually a dependency's broken main, but can be a real
-- downstream break the PR caused, so it is not labelled either way here.
CREATE VIEW IF NOT EXISTS v_pipeline_run_outcomes AS
WITH children AS
(
    SELECT
        parent_run_key,
        count()                                                 AS child_builds,
        sum(duration_ms)                                        AS child_build_ms,
        groupUniqArrayIf(component, result = 'failure')         AS failed_components,
        groupArrayIf(tuple(component, arches, failed_stage, failure_reason), result = 'failure')
                                                                AS failed_builds,
        countIf(result = 'failure' AND failure_is_infra)        AS infra_failures
    FROM v_pipeline_runs
    WHERE pipeline_type = 'component-build' AND parent_run_key != ''
    GROUP BY parent_run_key
)
SELECT
    r.run_key, r.job_name, r.build_number, r.build_url, r.started_at, r.ended_at,
    r.duration_ms, r.build_ms, r.test_ms, r.queue_ms,
    r.trigger_source, r.preset, r.repo, r.pr_number, r.sha, r.arches,
    r.state, r.result, r.verdict, r.superseded,
    r.failure_reason, r.failure_is_infra, r.failed_stage,
    r.nodes_built, r.nodes_reused, r.nodes_dropped, r.tests_total, r.tests_failed,
    c.child_builds, c.child_build_ms, c.failed_components, c.failed_builds,
    multiIf(
        r.state = 'running',                                     'running',
        r.superseded OR r.result = 'aborted',                    'superseded',
        r.verdict = 'green',                                     'passed',
        r.failure_is_infra OR c.infra_failures > 0,              'infra',
        r.repo != '' AND has(c.failed_components, r.repo),       'own_component',
        notEmpty(c.failed_components),                           'other_component',
        r.tests_failed > 0 OR r.failure_reason = 'test_failure', 'test_failure',
        r.duration_ms < 300000,                                  'ci_crash',
                                                                 'unattributed'
    ) AS outcome
FROM v_pipeline_runs AS r
LEFT JOIN children AS c ON c.parent_run_key = r.run_key
WHERE r.pipeline_type = 'orchestrator';

-- Daily gate health per lane and repo: the stability, outcome mix and time split the CI report
-- draws. pass_rate leaves out superseded and running runs, which reached no verdict.
CREATE VIEW IF NOT EXISTS v_pipeline_gate_daily AS
SELECT
    toDate(started_at)                                   AS day,
    trigger_source,
    repo,
    count()                                              AS runs,
    countIf(outcome = 'passed')                          AS passed,
    countIf(outcome = 'own_component')                   AS own_component,
    countIf(outcome = 'other_component')                 AS other_component,
    countIf(outcome = 'test_failure')                    AS test_failure,
    countIf(outcome = 'infra')                           AS infra,
    countIf(outcome = 'ci_crash')                        AS ci_crash,
    countIf(outcome = 'unattributed')                    AS unattributed,
    countIf(outcome = 'superseded')                      AS superseded,
    passed / nullIf(runs - superseded - countIf(outcome = 'running'), 0) AS pass_rate,
    quantileIf(0.5)(duration_ms, outcome = 'passed') / 60000 AS passed_p50_min,
    quantileIf(0.9)(duration_ms, outcome = 'passed') / 60000 AS passed_p90_min,
    quantileIf(0.5)(build_ms, outcome = 'passed' AND build_ms > 0) / 60000 AS build_p50_min,
    quantileIf(0.5)(test_ms, outcome = 'passed' AND test_ms > 0) / 60000  AS test_p50_min
FROM v_pipeline_run_outcomes
GROUP BY day, trigger_source, repo;
