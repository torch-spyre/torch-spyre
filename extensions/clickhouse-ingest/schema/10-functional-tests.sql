-- Functional-test results, schema v2. A clean break from the v1
-- test_runs/test_cases/run_properties set (and their hf_/si_ mirrors), on fresh data.
-- Rationale for every decision below: docs/clickhouse_v2_functional_tests_schema.md
--
-- The organising invariant: a column lives on the fact only if it is an input to the fact's
-- own key hash (so it cannot disagree with its source of truth) AND a leading ORDER BY
-- column (so it earns pruning). `component` passes both. Everything else run-scoped
-- resolves through run_id and test-scoped through test_case_id -- one home per fact.
--
-- One table pair serves all products, `component` replacing the hf_/si_ copies.
-- artifact_results (20-artifacts.sql) is the run record: run_id, arch, test_type, state and
-- the run-level counters live there, not here.
--
-- Bag-column convention, uniform with 20-artifacts.sql:
--   `props` = Map, open-ended, NEVER in a key -- extend freely, no identity churn.
--   `tags`  = Array, a SET, and IN the identity hash -- sort before hashing.
-- The names differ because the guarantees differ; do not converge them.

CREATE TABLE IF NOT EXISTS test_cases
(
    ts           DateTime DEFAULT now(),

    -- Content hash: uuid5 over (component, classname, name, sorted(tags)) -- derived, not
    -- minted per ingest, which is what lets one test reconcile across runs. Sorting is
    -- required, else two writers emitting the same tags in a different order mint different
    -- identities. Because `tags` is in the hash, re-tagging mints a NEW id -- so trend
    -- queries group on (component, classname, name), never on test_case_id.
    test_case_id UUID,

    component    LowCardinality(String),
    classname    String,
    name         String,

    -- Replaces the run_properties EAV table. Array, NOT Map: these are pytest tags in
    -- `namespace__value` form and a namespace repeats (testtype carries several values on
    -- most cases), so a Map would keep one and drop the rest. Filter with has()/hasAny().
    -- Named `tags`, not `props`, because it feeds test_case_id: writing to it mints a new
    -- identity, where `props` everywhere else sits outside all keys.
    tags         Array(LowCardinality(String)),

    CONSTRAINT chk_component CHECK component != '',
    CONSTRAINT chk_name      CHECK name != ''
)
ENGINE = MergeTree()
ORDER BY (component, test_case_id);


CREATE TABLE IF NOT EXISTS test_case_runs
(
    ts           DateTime DEFAULT now(),

    -- The only two foreign keys. run_id is uuid5 over (source, external_run_id, arch,
    -- test_type), derived by every writer from values it already holds with no cross-job
    -- threading contract -- which is what retires v1's minted run_id and runner_run_id.
    -- Deliberately the same name as v1's column: v2 is a separate database, and <thing>_id
    -- is the rule everywhere else here.
    run_id      UUID,
    test_case_id UUID,

    -- Denormalized only because it is a test_case_id hash input (so it cannot disagree)
    -- and the leading sort key, which is what makes a per-component read prune.
    component    LowCardinality(String),

    status       LowCardinality(String),
    duration_s   Float32,
    fail_message String DEFAULT '',

    -- Per-execution incidentals only. Anything run-scoped belongs on
    -- artifact_results; anything test-scoped on test_cases.tags.
    props        Map(LowCardinality(String), String),

    CONSTRAINT chk_status CHECK status IN
        ('passed','failed','error','skipped','xfail','xpass')
)
ENGINE = MergeTree()
-- Monthly parts are for retention (cheap DROP PARTITION), not pruning: the ORDER BY prefix
-- already prunes, and PARTITION BY component measured slower.
PARTITION BY toYYYYMM(ts)
ORDER BY (component, run_id, test_case_id);


-- Per-run case counters, maintained on insert. Computing them inline cost a full scan per
-- call (measured ~213M rows/day on prod to serve 425 runs); 75% of those rows belong to runs
-- no artifact references, so no predicate prunes them.
--
-- Grain is run_id alone, and that is what makes it usable: test_case_runs has no arch,
-- result_kind or tag, so every dimension a caller segments by is joined ABOVE this table.
-- SummingMergeTree because a sharded run arrives as many XMLs in separate inserts.
CREATE TABLE IF NOT EXISTS run_case_counters
(
    run_id      UUID,
    component   LowCardinality(String),
    total_tests UInt64,
    passed      UInt64,
    failed      UInt64,
    errors      UInt64,
    skipped     UInt64,
    -- Split, not folded into failed/passed: an xfail is an expected failure, and
    -- v_run_tier_counters already reports them this way.
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

-- Backfill once after creating the MV: an MV fires on INSERT only, so without this the table
-- stays empty and every reader reports zero counters.
--   INSERT INTO run_case_counters
--   SELECT run_id, component, count(), countIf(status='passed'), countIf(status='failed'),
--          countIf(status='error'), countIf(status='skipped'), countIf(status='xfail'),
--          countIf(status='xpass')
--   FROM test_case_runs GROUP BY run_id, component;
