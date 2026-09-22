-- Functional-test results, schema v2 -- a clean break from v1's test_runs/test_cases/run_properties
-- (and hf_/si_ mirrors). Rationale for every decision: docs/clickhouse_v2_functional_tests_schema.md
--
-- Bag-column convention, uniform with 20-artifacts.sql: `props` = Map, open-ended, never in a
-- key; `tags` = Array, a SET, and IN the identity hash (sort before hashing).

CREATE TABLE IF NOT EXISTS test_cases
(
    ts           DateTime DEFAULT now(),

    -- uuid5 over (component, classname, name, sorted(tags)) -- derived, so one test reconciles
    -- across runs; re-tagging mints a new id, so trend queries group on the plain triple, never test_case_id.
    test_case_id UUID,

    component    LowCardinality(String),
    classname    String,
    name         String,

    -- Replaces the run_properties EAV table. Array, not Map: a namespace (e.g. testtype) repeats.
    tags         Array(LowCardinality(String)),

    CONSTRAINT chk_component CHECK component != '',
    CONSTRAINT chk_name      CHECK name != ''
)
ENGINE = MergeTree()
ORDER BY (component, test_case_id);


CREATE TABLE IF NOT EXISTS test_case_runs
(
    ts           DateTime DEFAULT now(),

    -- run_id is uuid5 over (source, external_run_id, arch, test_type) -- derived, no cross-job threading.
    run_id      UUID,
    test_case_id UUID,

    -- Denormalized: a test_case_id hash input and the leading sort key for per-component pruning.
    component    LowCardinality(String),

    status       LowCardinality(String),
    duration_s   Float32,
    fail_message String DEFAULT '',

    -- Per-execution incidentals; run-scoped data belongs on artifact_results, test-scoped on test_cases.tags.
    props        Map(LowCardinality(String), String),

    CONSTRAINT chk_status CHECK status IN
        ('passed','failed','error','skipped','xfail','xpass')
)
ENGINE = MergeTree()
-- Monthly parts are for retention (cheap DROP PARTITION); the ORDER BY prefix already prunes.
PARTITION BY toYYYYMM(ts)
ORDER BY (component, run_id, test_case_id);


-- Per-run case counters, maintained on insert -- computing them inline scanned ~213M rows/day on
-- prod. SummingMergeTree because a sharded run arrives as many XMLs in separate inserts.
CREATE TABLE IF NOT EXISTS run_case_counters
(
    run_id      UUID,
    component   LowCardinality(String),
    total_tests UInt64,
    passed      UInt64,
    failed      UInt64,
    errors      UInt64,
    skipped     UInt64,
    -- Split from failed/passed: an xfail is an expected failure.
    xfail       UInt64,
    xpass       UInt64
)
ENGINE = SummingMergeTree()
ORDER BY (run_id, component);

CREATE MATERIALIZED VIEW IF NOT EXISTS run_case_counters_mv TO run_case_counters AS
SELECT
    run_id,
    component,
    count()                        AS total_tests,
    countIf(status = 'passed')     AS passed,
    countIf(status = 'failed')     AS failed,
    countIf(status = 'error')      AS errors,
    countIf(status = 'skipped')    AS skipped,
    countIf(status = 'xfail')      AS xfail,
    countIf(status = 'xpass')      AS xpass
FROM test_case_runs
GROUP BY run_id, component;

-- Backfill once after creating the MV -- it fires on INSERT only, so the table starts empty:
--   INSERT INTO run_case_counters
--   SELECT run_id, component, count(), countIf(status='passed'), countIf(status='failed'),
--          countIf(status='error'), countIf(status='skipped'), countIf(status='xfail'),
--          countIf(status='xpass')
--   FROM test_case_runs GROUP BY run_id, component;
