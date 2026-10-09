-- Where an orchestrator run's time went, one row per component build and per test leg (entry);
-- run-level columns repeat on each row. Executor-neutral: a GHA workflow and a Jenkins-local
-- `make test` fill the same provision/exec columns. *_ms are derived here, NULL when an end is unknown.
CREATE TABLE IF NOT EXISTS ci_run_timings
(
    -- The orchestrator's "<JOB_NAME>#<BUILD_NUMBER>", the same string as pipeline_runs.run_key.
    run_key                String,                                                        -- e.g. 'Spyre/orchestrator#4242'
    updated_at             DateTime64(3, 'UTC'),                                          -- e.g. '2026-10-08 11:30:05.123'
    entry                  LowCardinality(String),                                        -- e.g. 'build' | 'test'
    component              LowCardinality(String),                                        -- e.g. 'torch-spyre'
    artifact_name          String,                                                        -- e.g. 'torch-spyre-dev'
    arch                   LowCardinality(String),                                        -- e.g. 'x86_64' (also 'ppc64le', 's390x')
    id12                   String DEFAULT '',                                             -- e.g. 'bbbbbbbbbbbb'
    kind                   LowCardinality(String) DEFAULT 'image',                        -- e.g. 'image' (also 'rpm', 'wheel')
    -- A test leg's modes, sorted by the writer; '' key for a build entry.
    test_modes             Array(LowCardinality(String)),                                 -- e.g. build [] | test ['integration', 'smoke']
    leg                    String MATERIALIZED arrayStringConcat(test_modes, ','),        -- e.g. build '' | test 'integration,smoke'
    -- A retried build or re-dispatched leg is its own row, so a flaky first attempt stays visible.
    attempt                UInt16 DEFAULT 1,                                              -- e.g. 1 (2 for a retry)

    -- The run, as on its pipeline_runs row; trigger_pr is '' for a non-PR run, base_ref is
    -- the branch the trigger PR targets.
    trigger_kind           LowCardinality(String) DEFAULT '',                             -- e.g. 'upstream' (also 'manual', 'timer')
    trigger_source         LowCardinality(String) DEFAULT '',                             -- e.g. 'spyre-test' (also 'main-push', 'merge-queue')
    preset                 LowCardinality(String) DEFAULT '',                             -- e.g. 'trigger-pr-validation'
    build_mode             LowCardinality(String) DEFAULT '',                             -- e.g. 'pr'
    trigger_pr             String DEFAULT '',                                             -- e.g. 'github.ibm.com/ai-chip-toolchain/deeptools#4500' ('' for a non-PR run)
    repo                   LowCardinality(String) DEFAULT '',                             -- e.g. 'deeptools'
    pr_number              UInt32 DEFAULT 0,                                              -- e.g. 4500
    sha                    String DEFAULT '',                                             -- e.g. '0123456789abcdef0123456789abcdef01234567'
    base_ref               LowCardinality(String) DEFAULT '',                             -- e.g. 'main' (also 'release/2.x'; '' for a non-PR run)
    -- The component is one of the run's PRs (trigger or Test-With companion), not a dependency.
    is_pr_component        Bool DEFAULT false,                                            -- e.g. true (deeptools, or a Test-With companion) | false (a dependency)
    build_url              String DEFAULT '',                                             -- e.g. 'https://jenkins/job/orchestrator/4242/'
    verdict                LowCardinality(String) DEFAULT '',                             -- e.g. 'green' (also 'yellow', 'red')
    run_result             LowCardinality(String) DEFAULT '',                             -- e.g. 'success' (also 'failure', 'unstable', 'aborted')
    superseded             Bool DEFAULT false,                                            -- e.g. false
    pickup_path            LowCardinality(String) DEFAULT '',                             -- e.g. 'webhook' (also 'poller')
    comment_at             Nullable(DateTime64(3, 'UTC')),                                -- e.g. '2026-10-08 10:00:00.000' (/spyre-test commented)
    picked_up_at           Nullable(DateTime64(3, 'UTC')),                                -- e.g. '2026-10-08 10:00:05.000'
    run_scheduled_at       Nullable(DateTime64(3, 'UTC')),                                -- e.g. '2026-10-08 10:00:06.000'
    -- Not Nullable: it is the partition key, and a replaced row must land in the same partition.
    run_started_at         DateTime64(3, 'UTC'),                                          -- e.g. '2026-10-08 10:00:10.000'
    pr_queued_at           Nullable(DateTime64(3, 'UTC')),                                -- e.g. '2026-10-08 10:00:30.000' (PR comment first shows queued)
    pr_running_at          Nullable(DateTime64(3, 'UTC')),                                -- e.g. '2026-10-08 10:02:00.000' (PR comment first shows running)
    run_ended_at           Nullable(DateTime64(3, 'UTC')),                                -- e.g. '2026-10-08 11:30:00.000'

    -- The entry. build: built/reused (concurrent build)/dropped (already published)/failed;
    -- test: passed/failed/error (no test signal). result is the Jenkins result, lowercased.
    state                  LowCardinality(String) DEFAULT '',                             -- e.g. build 'built' | test 'passed'
    result                 LowCardinality(String) DEFAULT '',                             -- e.g. 'success'
    gating                 LowCardinality(String) DEFAULT '',                             -- e.g. build '' | test 'true' (also 'unstable', 'false')
    url                    String DEFAULT '',                                             -- e.g. 'https://jenkins/job/component-build/3/'
    agent                  LowCardinality(String) DEFAULT '',                             -- e.g. 'build-x86-1'
    -- build: component-build start, agent and build lock held, test stage start or job end.
    -- test: dispatched, leg's test stage start, leg job end.
    queued_at              Nullable(DateTime64(3, 'UTC')),                                -- e.g. build '2026-10-08 10:02:10.000' | test '2026-10-08 10:41:05.000'
    started_at             Nullable(DateTime64(3, 'UTC')),                                -- e.g. build '2026-10-08 10:02:40.000' | test '2026-10-08 10:41:20.000'
    ended_at               Nullable(DateTime64(3, 'UTC')),                                -- e.g. build '2026-10-08 10:20:40.000' | test '2026-10-08 11:25:00.000'

    -- The test executor. provision: ARC runner-set deploy, or the card lock wait (jenkins-local, jenkins-job).
    -- exec: workflow dispatch -> first job start -> last job end, or `make test` start -> end.
    executor               LowCardinality(String) DEFAULT '',                             -- e.g. build '' | test 'gha-ephemeral' (s390x: 'jenkins-local')
    provision_started_at   Nullable(DateTime64(3, 'UTC')),                                -- e.g. '2026-10-08 10:41:30.000' (runner-set deploy, or card lock requested)
    provision_ended_at     Nullable(DateTime64(3, 'UTC')),                                -- e.g. '2026-10-08 10:42:30.000' (runners up, or card lock acquired)
    exec_dispatched_at     Nullable(DateTime64(3, 'UTC')),                                -- e.g. '2026-10-08 10:42:40.500' (GHA workflow dispatched; NULL for jenkins-local)
    exec_started_at        Nullable(DateTime64(3, 'UTC')),                                -- e.g. '2026-10-08 10:46:40.000' (first GHA job starts, or make test starts)
    exec_ended_at          Nullable(DateTime64(3, 'UTC')),                                -- e.g. '2026-10-08 11:20:00.000' (last GHA job ends, or make test returns)
    exec_runs              UInt16 DEFAULT 0,                                              -- e.g. 1
    exec_jobs              UInt32 DEFAULT 0,                                              -- e.g. 4
    exec_result            LowCardinality(String) DEFAULT '',                             -- e.g. 'success' (also 'failure', 'never_started')
    exec_urls              Array(String),                                                 -- e.g. ['https://github.com/torch-spyre/torch-spyre/actions/runs/123']
    -- GHA: the gha:<owner>/<repo>/<run_id>#<attempt> pipeline_runs keys, for per-job runner waits.
    exec_run_keys          Array(String),                                                 -- e.g. ['gha:torch-spyre/torch-spyre/123#1']
    cards                  Array(LowCardinality(String)),                                 -- e.g. gha [] | jenkins-local ['0', '1']
    runner_died            Bool DEFAULT false,                                            -- e.g. false
    failure_reason         LowCardinality(String) DEFAULT '',                             -- e.g. '' (failed leg: 'runner_lost')
    failed_stage           String DEFAULT '',                                             -- e.g. '' (failed leg: 'Test')

    -- Derived. A cross-clock span (Jenkins ms vs GitHub s) can come out negative; it is stored as 0.
    -- if(d < 0, 0, d), not greatest(0, d): greatest returns 0 for a NULL end, this keeps the NULL.
    comment_to_pickup_ms   Nullable(UInt64) MATERIALIZED CAST(if(dateDiff('millisecond', comment_at, picked_up_at) < 0, 0, dateDiff('millisecond', comment_at, picked_up_at)) AS Nullable(UInt64)),  -- e.g. 5000
    comment_to_queued_ms   Nullable(UInt64) MATERIALIZED CAST(if(dateDiff('millisecond', comment_at, pr_queued_at) < 0, 0, dateDiff('millisecond', comment_at, pr_queued_at)) AS Nullable(UInt64)),  -- e.g. 30000
    queued_to_running_ms   Nullable(UInt64) MATERIALIZED CAST(if(dateDiff('millisecond', pr_queued_at, pr_running_at) < 0, 0, dateDiff('millisecond', pr_queued_at, pr_running_at)) AS Nullable(UInt64)),  -- e.g. 90000
    comment_to_end_ms      Nullable(UInt64) MATERIALIZED CAST(if(dateDiff('millisecond', comment_at, run_ended_at) < 0, 0, dateDiff('millisecond', comment_at, run_ended_at)) AS Nullable(UInt64)),  -- e.g. 5400000 (90 min)
    run_ms                 Nullable(UInt64) MATERIALIZED CAST(if(dateDiff('millisecond', run_started_at, run_ended_at) < 0, 0, dateDiff('millisecond', run_started_at, run_ended_at)) AS Nullable(UInt64)),  -- e.g. 5390000
    queue_ms               Nullable(UInt64) MATERIALIZED CAST(if(dateDiff('millisecond', queued_at, started_at) < 0, 0, dateDiff('millisecond', queued_at, started_at)) AS Nullable(UInt64)),  -- e.g. build 30000 (agent and build lock) | test 15000
    duration_ms            Nullable(UInt64) MATERIALIZED CAST(if(dateDiff('millisecond', started_at, ended_at) < 0, 0, dateDiff('millisecond', started_at, ended_at)) AS Nullable(UInt64)),  -- e.g. build 1080000 (18 min) | test 2620000
    provision_ms           Nullable(UInt64) MATERIALIZED CAST(if(dateDiff('millisecond', provision_started_at, provision_ended_at) < 0, 0, dateDiff('millisecond', provision_started_at, provision_ended_at)) AS Nullable(UInt64)),  -- e.g. 60000 (runner-set deploy, or card lock wait)
    exec_queue_ms          Nullable(UInt64) MATERIALIZED CAST(if(dateDiff('millisecond', exec_dispatched_at, exec_started_at) < 0, 0, dateDiff('millisecond', exec_dispatched_at, exec_started_at)) AS Nullable(UInt64)),  -- e.g. 239500 (dispatch to first job: runner pod queue)
    exec_ms                Nullable(UInt64) MATERIALIZED CAST(if(dateDiff('millisecond', exec_started_at, exec_ended_at) < 0, 0, dateDiff('millisecond', exec_started_at, exec_ended_at)) AS Nullable(UInt64)),  -- e.g. 2000000 (first job start to last job end)
    teardown_ms            Nullable(UInt64) MATERIALIZED CAST(if(dateDiff('millisecond', exec_ended_at, ended_at) < 0, 0, dateDiff('millisecond', exec_ended_at, ended_at)) AS Nullable(UInt64)),  -- e.g. 300000 (undeploy and cleanup after the last job)

    props                  Map(LowCardinality(String), String),                           -- e.g. {'test_map': '...'}
    audit_uuid             UUID DEFAULT generateUUIDv7(),                                 -- e.g. '01926f3a-7b1c-7d2e-8f00-0123456789ab'
    audit_timestamp        DateTime64(3) DEFAULT now64(3),                                -- e.g. '2026-10-08 11:30:05.200'

    CONSTRAINT chk_timing_entry CHECK entry IN ('build', 'test'),
    CONSTRAINT chk_timing_state CHECK (entry = 'build' AND state IN ('built', 'reused', 'dropped', 'failed', ''))
                                   OR (entry = 'test' AND state IN ('passed', 'failed', 'error', '')),
    CONSTRAINT chk_timing_executor CHECK executor IN ('gha-ephemeral', 'gha-standing', 'jenkins-local', 'jenkins-job', '')
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(run_started_at)
ORDER BY (run_key, entry, component, artifact_name, arch, leg, attempt);
