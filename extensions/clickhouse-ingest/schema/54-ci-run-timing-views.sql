-- Run-level views over ci_run_timings: one row per orchestrator run, then one view per lane that
-- launches runs (/spyre-test, the merge queue, main push), then a daily rollup with the queue-time
-- percentiles a CI report draws.
--
-- Plain views, not materialized. Every *_ms span is already a MATERIALIZED column, computed once
-- at insert and stored, so a read only folds a run's few dozen entry rows into one. An
-- insert-triggered MV would also be wrong here: it sees each insert block, not the merged
-- ReplacingMergeTree, so a re-written run would be counted twice and quantiles cannot be undone.
-- If reads over long windows get slow, the fix is a refreshable MV over v_ci_runs into a
-- ReplacingMergeTree keyed by run_key, as benchmark_metric_verdicts_mv does.

-- One row per entry, latest write only, so readers never need FINAL. The MATERIALIZED columns are
-- named because SELECT * leaves them out. FINAL alone skips partition pruning, so a filter on
-- run_started_at would still read every month; merging each partition on its own restores it,
-- and is exact here because a replaced row always lands in its original's partition.
CREATE VIEW IF NOT EXISTS v_ci_run_timings AS
SELECT
    *,
    leg,
    comment_to_pickup_ms, comment_to_queued_ms, queued_to_running_ms, comment_to_end_ms, run_ms,
    queue_ms, duration_ms, provision_ms, exec_queue_ms, exec_ms, teardown_ms
FROM ci_run_timings FINAL
SETTINGS do_not_merge_across_partitions_select_final = 1;

-- One row per run, any lane: the run-level columns once, and what its builds and test cells add
-- up to. run_started_at and trigger_source are grouping keys, not any(), so a filter on either
-- reaches the table and prunes partitions. jenkins_queue_ms is the orchestrator's own wait for an
-- executor (scheduled -> started), the one queue every lane has. *_phase_ms is the wall time from
-- the first entry queued to the last one ended; *_max_ms is the worst single entry.
-- outcome follows v_pipeline_run_outcomes: an aborted run is superseded, and a run with no verdict
-- falls back on its Jenkins result. failure_summary is one readable line per failed build or test
-- cell; failed_builds is (component, arch, attempt, failed_stage, failure_reason, url) and
-- failed_cells is (component, arch, leg, executor, state, failure_reason, failed_stage, url).
CREATE VIEW IF NOT EXISTS v_ci_runs AS
SELECT
    *,
    CAST(if(dateDiff('millisecond', picked_up_at, run_scheduled_at) < 0, 0,
            dateDiff('millisecond', picked_up_at, run_scheduled_at)) AS Nullable(UInt64)) AS pickup_to_scheduled_ms,
    CAST(if(dateDiff('millisecond', run_scheduled_at, run_started_at) < 0, 0,
            dateDiff('millisecond', run_scheduled_at, run_started_at)) AS Nullable(UInt64)) AS jenkins_queue_ms,
    CAST(if(dateDiff('millisecond', run_scheduled_at, run_ended_at) < 0, 0,
            dateDiff('millisecond', run_scheduled_at, run_ended_at)) AS Nullable(UInt64)) AS scheduled_to_end_ms,
    CAST(if(dateDiff('millisecond', build_first_queued_at, build_last_ended_at) < 0, 0,
            dateDiff('millisecond', build_first_queued_at, build_last_ended_at)) AS Nullable(UInt64)) AS build_phase_ms,
    CAST(if(dateDiff('millisecond', test_first_queued_at, test_last_ended_at) < 0, 0,
            dateDiff('millisecond', test_first_queued_at, test_last_ended_at)) AS Nullable(UInt64)) AS test_phase_ms,
    multiIf(
        superseded OR run_result IN ('aborted', 'not_built'),               'superseded',
        verdict = 'green' OR (verdict = '' AND run_result = 'success'),     'passed',
        verdict = 'yellow' OR (verdict = '' AND run_result = 'unstable'),   'warned',
                                                                            'failed'
    ) AS outcome,
    arrayStringConcat(arrayConcat(
        arrayMap(b -> concat(b.1, ' ', b.2, ' build failed',
                             if(b.4 != '', concat(' in ', b.4), ''),
                             if(b.5 != '', concat(': ', b.5), '')), failed_builds),
        arrayMap(c -> concat(c.1, ' ', c.2, if(c.3 != '', concat(' [', c.3, ']'), ''), ' test ', c.5,
                             if(c.7 != '', concat(' in ', c.7), ''),
                             if(c.6 != '', concat(': ', c.6), '')), failed_cells)), ' | ') AS failed_entries,
    if(outcome = 'failed' AND failed_entries = '',
       concat('run ', if(run_result != '', run_result, 'failed'), ' with no failed build or test'),
       failed_entries) AS failure_summary
FROM
(
    SELECT
        run_key,
        run_started_at,
        trigger_source,
        any(trigger_kind)                                         AS trigger_kind,
        any(preset)                                               AS preset,
        any(build_mode)                                           AS build_mode,
        any(trigger_pr)                                           AS trigger_pr,
        any(repo)                                                 AS repo,
        any(pr_number)                                            AS pr_number,
        any(sha)                                                  AS sha,
        any(base_ref)                                             AS base_ref,
        any(build_url)                                            AS build_url,
        any(verdict)                                              AS verdict,
        any(run_result)                                           AS run_result,
        any(superseded)                                           AS superseded,
        any(pickup_path)                                          AS pickup_path,
        any(comment_at)                                           AS comment_at,
        any(picked_up_at)                                         AS picked_up_at,
        any(run_scheduled_at)                                     AS run_scheduled_at,
        any(pr_queued_at)                                         AS pr_queued_at,
        any(pr_running_at)                                        AS pr_running_at,
        any(run_ended_at)                                         AS run_ended_at,
        any(comment_to_pickup_ms)                                 AS comment_to_pickup_ms,
        any(comment_to_queued_ms)                                 AS comment_to_queued_ms,
        any(queued_to_running_ms)                                 AS queued_to_running_ms,
        any(comment_to_end_ms)                                    AS comment_to_end_ms,
        any(run_ms)                                               AS run_ms,

        countIf(entry = 'build')                                  AS builds,
        countIf(entry = 'build' AND state = 'built')              AS builds_built,
        countIf(entry = 'build' AND state = 'reused')             AS builds_reused,
        countIf(entry = 'build' AND state = 'dropped')            AS builds_dropped,
        countIf(entry = 'build' AND state = 'failed')             AS builds_failed,
        groupUniqArrayIf(component, entry = 'build' AND state = 'built')  AS built_components,
        groupUniqArrayIf(component, entry = 'build' AND state = 'failed') AS failed_components,
        groupArrayIf(tuple(component, arch, attempt, failed_stage, failure_reason, url),
                     entry = 'build' AND state = 'failed')        AS failed_builds,
        maxIf(queue_ms, entry = 'build')                          AS build_queue_max_ms,
        maxIf(duration_ms, entry = 'build' AND state = 'built')   AS build_max_ms,
        minIf(queued_at, entry = 'build')                         AS build_first_queued_at,
        minIf(started_at, entry = 'build')                        AS build_first_started_at,
        maxIf(ended_at, entry = 'build')                          AS build_last_ended_at,

        countIf(entry = 'test')                                   AS test_cells,
        countIf(entry = 'test' AND state = 'passed')              AS tests_passed,
        countIf(entry = 'test' AND state = 'failed')              AS tests_failed,
        countIf(entry = 'test' AND state = 'error')               AS tests_error,
        countIf(entry = 'test' AND runner_died)                   AS runners_died,
        groupArrayIf(tuple(component, arch, leg, executor, state, failure_reason, failed_stage, url),
                     entry = 'test' AND state IN ('failed', 'error')) AS failed_cells,
        maxIf(queue_ms, entry = 'test')                           AS test_queue_max_ms,
        maxIf(provision_ms, entry = 'test')                       AS provision_max_ms,
        maxIf(exec_queue_ms, entry = 'test')                      AS exec_queue_max_ms,
        maxIf(exec_ms, entry = 'test')                            AS exec_max_ms,
        maxIf(teardown_ms, entry = 'test')                        AS teardown_max_ms,
        minIf(queued_at, entry = 'test')                          AS test_first_queued_at,
        minIf(started_at, entry = 'test')                         AS test_first_started_at,
        maxIf(ended_at, entry = 'test')                           AS test_last_ended_at
    FROM v_ci_run_timings
    GROUP BY run_key, run_started_at, trigger_source
);

-- /spyre-test runs as a timeline, milestones in the order they happen: the comment, the poller
-- that saw it (pickup_path: webhook or cron), the orchestrator queued and started, the bot's
-- "queued" and "running" comments, builds, tests, and the end (the verdict comment). Each span
-- covers the step between two milestones; comment_to_end_ms is what the developer waited.
-- superseded runs stay in, flagged: a newer /spyre-test on the same PR cancelled them, so leave
-- them out of any timing.
CREATE VIEW IF NOT EXISTS v_ci_runs_spyre_test AS
SELECT
    run_key, repo, pr_number, trigger_pr, sha, base_ref, preset, build_mode,
    outcome, verdict, run_result, superseded, pickup_path,
    comment_at, picked_up_at, run_scheduled_at, run_started_at, pr_queued_at, pr_running_at,
    build_first_started_at, build_last_ended_at, test_first_started_at, test_last_ended_at,
    run_ended_at,
    comment_to_pickup_ms, pickup_to_scheduled_ms, jenkins_queue_ms,
    comment_to_queued_ms, queued_to_running_ms,
    build_phase_ms, test_phase_ms, run_ms, comment_to_end_ms,
    builds, builds_built, builds_reused, builds_dropped, builds_failed,
    build_queue_max_ms, build_max_ms,
    test_cells, tests_passed, tests_failed, tests_error, runners_died,
    test_queue_max_ms, provision_max_ms, exec_queue_max_ms, exec_max_ms, teardown_max_ms,
    failure_summary, failed_builds, failed_cells,
    build_url
FROM v_ci_runs
WHERE trigger_source = 'spyre-test';

-- Merge-queue runs: a PR's gate before it lands on base_ref. No comment, so the run's latency
-- starts at the Jenkins queue; pr_queued_at/pr_running_at are the queue head's status updates.
CREATE VIEW IF NOT EXISTS v_ci_runs_merge_queue AS
SELECT
    run_key, repo, pr_number, trigger_pr, sha, base_ref, preset, build_mode,
    outcome, verdict, run_result, superseded,
    run_scheduled_at, run_started_at, pr_queued_at, pr_running_at,
    build_first_started_at, build_last_ended_at, test_first_started_at, test_last_ended_at,
    run_ended_at,
    jenkins_queue_ms, queued_to_running_ms, build_phase_ms, test_phase_ms, run_ms,
    scheduled_to_end_ms,
    builds, builds_built, builds_reused, builds_dropped, builds_failed,
    build_queue_max_ms, build_max_ms,
    test_cells, tests_passed, tests_failed, tests_error, runners_died,
    test_queue_max_ms, provision_max_ms, exec_queue_max_ms, exec_max_ms, teardown_max_ms,
    failure_summary, failed_builds, failed_cells,
    build_url
FROM v_ci_runs
WHERE trigger_source = 'merge-queue';

-- Main-push runs end to end: the PR merged (merged_at, the merge commit's time), the push
-- triggered main-push-build (triggered_at), the orchestrator queued, started, built, tested and
-- ended. main-push-build sends those first two as the orchestrator's TRIGGER_COMMENT_AT and
-- TRIGGER_PICKED_UP_MS, so they land in comment_at and picked_up_at; until it does they are
-- NULL, and scheduled_to_end_ms is the longest span known. merge_to_green_ms is set only for a
-- passed run; a failed one says what broke in failure_summary. In reuse mode unchanged artifacts
-- are dropped at plan time, so built_components is what the push actually rebuilt.
CREATE VIEW IF NOT EXISTS v_ci_runs_main_push AS
SELECT
    run_key, repo, pr_number, sha, preset, build_mode,
    outcome, verdict, run_result, superseded,
    comment_at AS merged_at, picked_up_at AS triggered_at, run_scheduled_at, run_started_at,
    build_first_started_at, build_last_ended_at, test_first_started_at, test_last_ended_at,
    run_ended_at,
    comment_to_pickup_ms AS merge_to_trigger_ms, pickup_to_scheduled_ms AS trigger_to_scheduled_ms,
    jenkins_queue_ms, build_phase_ms, test_phase_ms, run_ms, scheduled_to_end_ms,
    comment_to_end_ms AS merge_to_end_ms,
    if(outcome = 'passed', comment_to_end_ms, NULL) AS merge_to_green_ms,
    builds, builds_built, builds_reused, builds_dropped, builds_failed,
    built_components, failed_components,
    build_queue_max_ms, build_max_ms,
    test_cells, tests_passed, tests_failed, tests_error, runners_died,
    test_queue_max_ms, provision_max_ms, exec_queue_max_ms, exec_max_ms, teardown_max_ms,
    failure_summary, failed_builds, failed_cells,
    build_url
FROM v_ci_runs
WHERE trigger_source = 'main-push';

-- Daily health per lane and repo. Superseded runs are counted, then left out of every rate and
-- percentile. pass_rate is over runs that reached a verdict. Percentiles are minutes; a lane that
-- lacks a span gets NULL, not 0. The comment_* spans start at the run's trigger event: the
-- /spyre-test comment, or for main push the merge (comment_to_end is merge to end there).
CREATE VIEW IF NOT EXISTS v_ci_lane_daily AS
SELECT
    toDate(run_started_at)                                            AS day,
    trigger_source,
    repo,
    count()                                                           AS runs,
    countIf(superseded)                                               AS superseded_runs,
    countIf(NOT superseded AND verdict = 'green')                     AS green,
    countIf(NOT superseded AND verdict = 'yellow')                    AS yellow,
    countIf(NOT superseded AND verdict = 'red')                       AS red,
    green / nullIf(green + yellow + red, 0)                           AS pass_rate,
    sumIf(builds_built, NOT superseded)                               AS components_built,
    sumIf(builds_failed, NOT superseded)                              AS build_failures,
    sumIf(tests_failed + tests_error, NOT superseded)                 AS test_cells_failed,
    quantileIf(0.5)(comment_to_pickup_ms, NOT superseded) / 60000     AS comment_to_pickup_p50_min,
    quantileIf(0.5)(comment_to_queued_ms, NOT superseded) / 60000     AS comment_to_queued_p50_min,
    quantileIf(0.9)(comment_to_queued_ms, NOT superseded) / 60000     AS comment_to_queued_p90_min,
    quantileIf(0.5)(queued_to_running_ms, NOT superseded) / 60000     AS queued_to_running_p50_min,
    quantileIf(0.9)(queued_to_running_ms, NOT superseded) / 60000     AS queued_to_running_p90_min,
    quantileIf(0.5)(comment_to_end_ms, NOT superseded) / 60000        AS comment_to_end_p50_min,
    quantileIf(0.9)(comment_to_end_ms, NOT superseded) / 60000        AS comment_to_end_p90_min,
    quantileIf(0.5)(jenkins_queue_ms, NOT superseded) / 60000         AS jenkins_queue_p50_min,
    quantileIf(0.9)(jenkins_queue_ms, NOT superseded) / 60000         AS jenkins_queue_p90_min,
    quantileIf(0.5)(build_queue_max_ms, NOT superseded) / 60000       AS build_queue_p50_min,
    quantileIf(0.9)(build_queue_max_ms, NOT superseded) / 60000       AS build_queue_p90_min,
    quantileIf(0.5)(exec_queue_max_ms, NOT superseded) / 60000        AS exec_queue_p50_min,
    quantileIf(0.9)(exec_queue_max_ms, NOT superseded) / 60000        AS exec_queue_p90_min,
    quantileIf(0.5)(run_ms, NOT superseded) / 60000                   AS run_p50_min,
    quantileIf(0.9)(run_ms, NOT superseded) / 60000                   AS run_p90_min
FROM v_ci_runs
GROUP BY day, trigger_source, repo;
