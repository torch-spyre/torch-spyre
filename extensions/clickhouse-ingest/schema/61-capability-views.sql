-- Views over capabilities / capability_runs. Two jobs: the enriched base join, and the
-- per-suite counters v1 stored as columns on model_ops_suites -- which v2 dropped because a
-- stored counter drifts from the rows it summarises. Apply after 46-capabilities.sql.
--
-- MEASURED, not asserted: of 298 v1 suites, 269 disagree with their own variants on
-- spyre_enabled_count AND not_implemented_count (a typical row stores 24 enabled where 29
-- distinct operations actually XPASSed, and 4 not-implemented where 9 XFAILed).
-- cpu_fallback_count was right on all 298 -- so this is not "counters are hard", it is that
-- two of them counted something other than what their name says, invisibly, for 298 runs.
-- Deriving them makes that class of divergence unrepresentable.
--
-- The v1 columns these replace, and their v2 definitions:
--   spyre_enabled_count   -> status='passed'          AND backend='spyre'
--   not_implemented_count -> status='not_implemented'
--   cpu_fallback_count    -> status='passed'          AND backend='cpu'
--   spyre_failed_count    -> status='failed'          AND backend='spyre'
-- v1's four counters were not disjoint (a FALLBACK is also a pass), which is exactly why v2
-- splits status from backend instead of folding them into one vocabulary.


-- Every capability verdict with its identity and run context resolved. The base join for
-- everything below, and the drill-down a dashboard row expands into.
-- ALWAYS filter by run_id or (component, test_type); unfiltered this joins both tables whole.
CREATE VIEW IF NOT EXISTS v_capability_results AS
SELECT
    cr.ts            AS ts,
    cr.run_id        AS run_id,
    cr.capability_id AS capability_id,
    cr.component     AS component,
    cr.test_type     AS test_type,
    cr.arch          AS arch,
    c.subject        AS subject,
    c.name           AS name,
    c.tags           AS tags,
    cr.status        AS status,
    cr.backend       AS backend,
    cr.fail_reason   AS fail_reason,
    -- The identity's discriminator (input shapes/dtypes for model_ops) is hashed INTO
    -- capability_id, so it is read from the identity row rather than the observation.
    c.props          AS capability_props,
    cr.props         AS run_props,
    -- The shard that wrote this row, the dedup scope for a fan-out analysis. Surfaced so a
    -- reader can tell a missing shard from a genuinely absent verdict.
    cr.props['shard'] AS shard
FROM capability_runs AS cr
INNER JOIN capabilities AS c ON c.capability_id = cr.capability_id;


-- The per-(run, subject) counters model_ops_suites stored. One row per subject per run, which
-- is v1's suite grain: its suite_name folded the model into the suite, so a v1 row and a row
-- here line up once subject is read as the model.
--
-- countIf over uniqExactIf deliberately: these count VERDICTS, one per (capability, backend),
-- not distinct capabilities. v1's stored numbers were neither consistently -- which is the
-- drift above. A caller wanting distinct operations has distinct_* below.
CREATE VIEW IF NOT EXISTS v_capability_run_counters AS
SELECT
    run_id,
    component,
    test_type,
    subject,
    any(arch)                                                        AS arch,
    min(ts)                                                          AS started_at,
    count()                                                          AS total_verdicts,
    countIf(status = 'passed' AND backend = 'spyre')                  AS spyre_enabled,
    countIf(status = 'not_implemented')                               AS not_implemented,
    countIf(status = 'passed' AND backend = 'cpu')                    AS cpu_fallback,
    countIf(status = 'failed' AND backend = 'spyre')                  AS spyre_failed,
    countIf(status = 'failed')                                        AS failed_any_backend,
    -- Distinct CAPABILITIES rather than verdicts: one operation measured on three backends is
    -- one capability. This is the number "how many ops does this model exercise" wants, and
    -- the one v1's counters were most often mistaken for.
    uniqExact(capability_id)                                         AS distinct_capabilities,
    uniqExactIf(capability_id, status = 'passed' AND backend = 'spyre') AS distinct_spyre_enabled,
    uniqExactIf(capability_id, status = 'not_implemented')            AS distinct_not_implemented,
    uniqExactIf(capability_id, status = 'passed' AND backend = 'cpu') AS distinct_cpu_fallback,
    -- Support rate over what was actually attempted on spyre: a not_implemented capability is
    -- not a failure to fix, so folding it into the denominator would make an unsupported
    -- operation look like a regression. Guarded against a zero denominator, which is a run
    -- that attempted nothing on spyre rather than a run that failed everything.
    if(countIf(backend = 'spyre') = 0, 0,
       round(100.0 * countIf(status = 'passed' AND backend = 'spyre')
             / countIf(backend = 'spyre'), 2))                       AS spyre_pass_rate
FROM v_capability_results
GROUP BY run_id, component, test_type, subject;


-- Which capabilities work on the CPU but not on Spyre -- the query v1 could not answer at all,
-- because it stored one status where a CPU fallback and a Spyre pass were the same value.
-- The reason this table pair exists: backend is a VALUE, so this is a self-join, not a flag.
CREATE VIEW IF NOT EXISTS v_capability_backend_gap AS
SELECT
    cpu.run_id        AS run_id,
    cpu.component     AS component,
    cpu.test_type     AS test_type,
    cpu.subject       AS subject,
    cpu.name          AS name,
    cpu.capability_id AS capability_id,
    spy.status        AS spyre_status,
    spy.fail_reason   AS spyre_fail_reason
FROM v_capability_results AS cpu
INNER JOIN v_capability_results AS spy
        ON  spy.run_id        = cpu.run_id
        AND spy.capability_id = cpu.capability_id
WHERE cpu.backend = 'cpu'   AND cpu.status = 'passed'
  AND spy.backend = 'spyre' AND spy.status != 'passed';


-- Per-capability history across runs: is an operation newly supported, or newly broken.
-- Grouped on the IDENTITY, which is what makes this answerable -- v1's variant_id was a
-- per-row surrogate (51,356 ids for 51,356 rows), so no two runs of one operation ever
-- reconciled and a trend query had nothing to group on.
CREATE VIEW IF NOT EXISTS v_capability_history AS
SELECT
    capability_id,
    component,
    test_type,
    subject,
    name,
    backend,
    count()                                    AS runs_observed,
    countIf(status = 'passed')                 AS runs_passed,
    countIf(status = 'not_implemented')         AS runs_not_implemented,
    countIf(status = 'failed')                 AS runs_failed,
    min(ts)                                    AS first_seen,
    max(ts)                                    AS last_seen,
    -- The most recent verdict, which is what a status badge shows. argMax over ts rather than
    -- any(): a plain any() returns an arbitrary row, so a badge could show a stale pass for an
    -- operation that has since regressed.
    argMax(status, ts)                         AS latest_status,
    argMax(fail_reason, ts)                    AS latest_fail_reason,
    argMax(run_id, ts)                         AS latest_run_id
FROM v_capability_results
GROUP BY capability_id, component, test_type, subject, name, backend;
