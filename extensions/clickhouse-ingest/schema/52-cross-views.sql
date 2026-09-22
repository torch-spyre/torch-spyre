-- Views spanning both families, hence neither 50- nor 51-: v_run_coverage joins
-- artifact_results to test_case_runs to answer "did the leg that produced this artifact
-- actually report cases". Apply last -- it needs every table above.

-- Coverage honesty: artifact_results is a sparse junction, so a chart drawn from it is a
-- sample, not a census, and this view lets the UI label that rather than hide it.
-- Grain is (day, arch) because coverage varies by platform and over time as writers roll
-- out, so one global percentage would misdescribe both. Both gaps are reported because they
-- differ: cases with no junction row are invisible to every artifact page, while a junction
-- row with no cases is a suite that never reported and would chart as a silent zero.
CREATE VIEW IF NOT EXISTS v_run_coverage AS
SELECT
    day,
    arch,
    uniqExact(run_id)                              AS runs,
    uniqExactIf(run_id, in_results AND in_cases)   AS runs_linked,
    uniqExactIf(run_id, NOT in_results)            AS runs_missing_result,
    uniqExactIf(run_id, NOT in_cases)              AS runs_missing_cases,
    if(uniqExact(run_id) > 0,
       uniqExactIf(run_id, in_results AND in_cases) / uniqExact(run_id),
       NULL)                                        AS coverage
FROM
(
    -- Every run known from either side: a run in only one table is the gap being measured,
    -- so neither table alone can enumerate the denominator.
    SELECT
        min(day)   AS day,
        argMax(arch, arch != '') AS arch,
        run_id,
        max(in_results) AS in_results,
        max(in_cases)   AS in_cases
    FROM
    (
        -- state='running' excluded: a crashed run's stale row would count as covered.
        SELECT toDate(ts) AS day, if(arch IN ('amd64', 'x86', 'x86-64'), 'x86_64', arch) AS arch,
               run_id, 1 AS in_results, 0 AS in_cases
        FROM artifact_results
        WHERE state != 'running'
        UNION ALL
        SELECT toDate(min(ts)) AS day, '' AS arch, run_id, 0 AS in_results, 1 AS in_cases
        FROM test_case_runs
        GROUP BY run_id
    )
    GROUP BY run_id
)
GROUP BY day, arch;

-- Indexes. One only, and it is measured; see docs/clickhouse_v2_views.md.

-- test_case_runs is ORDER BY (component, run_id, test_case_id), so a run_id-only lookup
-- prunes to parts but not to granules. Retained ONLY for a measured 5-6x reduction in rows
-- read at 0.007% of table size -- it buys no wall-time win and is worse at the tail under
-- concurrency, so do not cite it as a latency optimisation.
-- PRECONDITION: it prunes only if a run_id's rows are contiguous in the sort order, i.e. one
-- run belongs to one component. True by construction and verified on prod. If ever violated
-- the index still materializes to a plausible size and prunes nothing, so the failure is
-- invisible here and is worth an ingest-side assertion.
-- Do NOT add `component` to a run_id predicate as tuning: a run_id belongs to exactly one
-- component, so it carries no information the index has not already used.
-- No index on artifact_results or test_cases -- both scan fully in a few ms.
-- ADD INDEX covers only parts written after it, so the MATERIALIZE below must follow it --
-- reversed, it targets an index that does not exist yet. mutations_sync=2 waits for
-- completion, without which a benchmark measures a half-built index.
ALTER TABLE test_case_runs
    ADD INDEX IF NOT EXISTS idx_run_id run_id TYPE bloom_filter(0.01) GRANULARITY 1;
ALTER TABLE test_case_runs MATERIALIZE INDEX idx_run_id SETTINGS mutations_sync = 2;
