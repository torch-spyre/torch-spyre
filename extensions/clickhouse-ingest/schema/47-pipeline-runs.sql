-- CI audit trails: one row per Jenkins build of the Spyre/ pipelines, whatever its outcome. The
-- artifact layer only sees a run once it publishes or reports a test, so a run that dies in build
-- is invisible there; these tables are where its verdict, timing and failure cause land.
-- Written by vars/pushToClickhouse.groovy as JSONEachRow (no python3 on the controller).

-- run_key is "<JOB_NAME>#<BUILD_NUMBER>": the same string as artifact_results.props['orch_run_key']
-- and RunId's external_run_id, so a run joins to its artifacts without a lookup.
-- Upserted: a 'running' row at start, replaced by the 'finished' row from post{always}. A run that
-- hangs or loses its controller keeps its 'running' row.
CREATE TABLE IF NOT EXISTS pipeline_runs
(
    run_key           String,
    updated_at        DateTime64(3, 'UTC'),
    pipeline_type     LowCardinality(String),
    state             LowCardinality(String),

    job_name          String,
    build_number      UInt32,
    build_url         String,
    agent             LowCardinality(String) DEFAULT '',
    -- The orchestrator that dispatched a component-build or product-test run; '' for a top-level run.
    parent_run_key    String DEFAULT '',

    started_at        DateTime64(3, 'UTC'),
    ended_at          Nullable(DateTime64(3, 'UTC')),
    queue_ms          UInt64 DEFAULT 0,
    duration_ms       UInt64 DEFAULT 0,
    -- Time before and during the tests (product-test is all test). An orchestrator's build_ms
    -- spans its component builds, whose own test stages are split out on their rows.
    build_ms          UInt64 DEFAULT 0,
    test_ms           UInt64 DEFAULT 0,

    -- How Jenkins started it (upstream, manual, timer, other) vs which CI lane asked for it
    -- (merge-queue, spyre-test, main-push, scheduled-nightly, ...).
    trigger_kind      LowCardinality(String) DEFAULT '',
    trigger_source    LowCardinality(String) DEFAULT '',
    preset            LowCardinality(String) DEFAULT '',
    build_mode        LowCardinality(String) DEFAULT '',
    -- The product repo and PR a PR-validation or merge-queue run is for; '' / 0 otherwise.
    repo              LowCardinality(String) DEFAULT '',
    pr_number         UInt32 DEFAULT 0,
    sha               String DEFAULT '',

    -- component-build: the component it built. arches: canonical names (x86_64, ppc64le, s390x),
    -- the targets of an orchestrator or the platforms of one build.
    component         LowCardinality(String) DEFAULT '',
    arches            Array(LowCardinality(String)),

    -- Jenkins currentResult, lowercased; '' while running.
    result            LowCardinality(String) DEFAULT '',
    -- Orchestrator gate outcome: green passes, yellow and red do not; '' for other pipelines.
    verdict           LowCardinality(String) DEFAULT '',
    superseded        Bool DEFAULT false,
    reached_normal_completion Bool DEFAULT false,
    -- Per-arch lane status, e.g. {'x86_64': 'PARTIAL'}.
    lane_results      Map(LowCardinality(String), LowCardinality(String)),

    nodes_built       UInt32 DEFAULT 0,
    nodes_reused      UInt32 DEFAULT 0,
    nodes_dropped     UInt32 DEFAULT 0,
    tests_total       UInt32 DEFAULT 0,
    tests_failed      UInt32 DEFAULT 0,
    ch_write_failures UInt32 DEFAULT 0,

    -- diagnose_failure.py's category, left unconstrained: a CHECK here would drop the whole row
    -- the day the classifier gains a category.
    failure_reason    LowCardinality(String) DEFAULT '',
    failure_is_infra  Bool DEFAULT false,
    failure_evidence  String DEFAULT '',
    failed_stage      String DEFAULT '',
    fail_log_tail     String DEFAULT '' CODEC(ZSTD(3)),

    -- Run inputs kept for replay (test_map, preset_json), not for querying.
    props             Map(LowCardinality(String), String),
    audit_uuid        UUID DEFAULT generateUUIDv7(),
    audit_timestamp   DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_pipeline_type CHECK pipeline_type IN ('orchestrator', 'component-build', 'product-test'),
    CONSTRAINT chk_run_state CHECK state IN ('running', 'finished')
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(started_at)
ORDER BY (pipeline_type, job_name, build_number);

-- One row per orchestrator test leg (arch x image x mode set). Unlike artifact_results it exists
-- for a leg that never reached its tests (runner died, image failed to pull), which is the case a
-- stability report most needs.
CREATE TABLE IF NOT EXISTS pipeline_run_legs
(
    run_key         String,
    updated_at      DateTime64(3, 'UTC'),
    arch            LowCardinality(String),
    component       LowCardinality(String),
    image           LowCardinality(String),
    kind            LowCardinality(String) DEFAULT 'image',
    test_modes      Array(LowCardinality(String)),

    id12            String DEFAULT '',
    -- The leg's artifact_results.run_id; nil when the leg produced no result.
    run_id          UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),
    result          LowCardinality(String) DEFAULT '',
    -- true blocks the gate, unstable marks the run UNSTABLE only, false is informational.
    gating          LowCardinality(String) DEFAULT '',

    url             String DEFAULT '',
    plan_url        String DEFAULT '',
    started_at      Nullable(DateTime64(3, 'UTC')),
    ended_at        Nullable(DateTime64(3, 'UTC')),
    duration_ms     UInt64 DEFAULT 0,
    runner_died     Bool DEFAULT false,
    failure_reason  LowCardinality(String) DEFAULT '',
    failed_stage    String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (run_key, arch, component, image, kind);
