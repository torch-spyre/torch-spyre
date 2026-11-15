-- Verdict, gate and promotion-gate views. They read v_invalidated_runs (83), so they follow it in apply order.
--
-- One row per run x series x metric; a series is (component, benchmark_id, backend, arch,
-- source_job). The baseline is the series' previous 20 baseline-eligible runs by run time --
-- backward-only, so a verdict is stable until older data is backfilled or the policy changes.
-- A run `exceeds` when its median is worse than the baseline's by more than floor_pct. One run
-- over is `suspected` (reason single_run); `regressed` needs the series' previous eligible run
-- over as well. Persistence stands in for a significance test, which one sample per run cannot
-- pass: with n_current = 1 the one-sided U's p cannot fall below about 1/(n_baseline+1). `score`
-- is still that p, informational only.
-- Exclusions, applied per (run, benchmark, repeat) before anything is aggregated:
--   * state `invalidated` in v_invalidated_runs: the samples are dropped, from the verdict output and every window.
--   * state `suspect`: still judged and shown, never part of a baseline, verdict capped at `suspected`.
--   * run.purpose outside benchmark_baseline_purposes (an absent prop is eligible): same as suspect.
-- Ineligible rows take no window slot: `pos` counts the eligible runs before a row, and the
-- window is the eligible rows with pos in [pos-20, pos-1].
CREATE VIEW IF NOT EXISTS v_benchmark_metric_verdicts AS
WITH 10 AS min_baseline, 0.01 AS alpha
SELECT
    * EXCEPT (exceeds, prev_exceeds, cap, pos),
    multiIf(n_baseline < min_baseline,       'no_baseline',
            exceeds AND cap != '',           'suspected',
            exceeds AND prev_exceeds,        'regressed',
            exceeds,                         'suspected',
            worse_pct < -floor_pct,          'improved',
                                             'unchanged') AS status,
    multiIf(status = 'suspected' AND cap = 'purpose', 'purpose_not_baseline',
            status = 'suspected' AND cap = 'suspect', 'suspect_data',
            status = 'suspected',                     'single_run',
            status = 'unchanged' AND score < alpha,   'below_floor',
                                                      '') AS reason
FROM (
    SELECT * EXCEPT (eligible),
        n_baseline >= min_baseline AND worse_pct > floor_pct AS exceeds,
        anyIf(exceeds, eligible) OVER (
            PARTITION BY component, benchmark_id, backend, arch, source_job, metric
            ORDER BY pos RANGE BETWEEN 1 PRECEDING AND 1 PRECEDING) AS prev_exceeds
    FROM (
        SELECT
            run_id, benchmark_id, component, backend, arch, source_job, name, input_shapes,
            run_ts AS ts, metric, tier, higher_is_better, pos, eligible,
            multiIf(off_purpose, 'purpose', suspect, 'suspect', '') AS cap,
            'rolling' AS baseline_kind,
            base_runs AS baseline_run_ids,
            length(base_runs) AS n_baseline,
            length(cur) AS n_current,
            round(base_med, 4) AS baseline_value,
            round(cur_med, 4) AS current_value,
            round((cur_med - base_med) / abs(base_med) * 100, 1) AS delta_pct,
            -- delta_pct signed so that positive is always worse.
            round(if(higher_is_better, -1, 1) * delta_pct, 1) AS worse_pct,
            round(if(higher_is_better, p_greater, p_less), 4) AS score,
            -- Never tighter than 2 robust sigmas (1.4826 * MAD) of the baseline's own noise.
            round(greatest(policy_floor, 200 * 1.4826 * base_mad / abs(base_med)), 1) AS floor_pct
        FROM (
            SELECT r.*,
                if(p.has_policy, p.tier, 'informational') AS tier,
                if(p.has_policy, p.higher_is_better,
                   match(metric, 'throughput|_per_second$') OR metric = 'pt_util_percent') AS higher_is_better,
                if(p.has_policy, p.floor_pct, 5) AS policy_floor,
                arrayReduce('median', base) AS base_med,
                arrayReduce('median', cur) AS cur_med,
                arrayReduce('median', arrayMap(x -> abs(x - base_med), base)) AS base_mad,
                -- U throws on an empty sample, and if() evaluates both branches, so rows that are
                -- no_baseline anyway get a stand-in baseline.
                if(empty(base), [cur[1]], base) AS b,
                arrayConcat(arrayWithConstant(length(b), 0), arrayWithConstant(length(cur), 1)) AS idx,
                arrayReduce('mannWhitneyUTest(\'less\')',    arrayConcat(b, cur), idx).2 AS p_less,
                arrayReduce('mannWhitneyUTest(\'greater\')', arrayConcat(b, cur), idx).2 AS p_greater
            FROM (
                SELECT *,
                    arrayFlatten(groupArrayIf(cur, eligible) OVER w) AS base,
                    groupArrayIf(run_id, eligible) OVER w AS base_runs
                FROM (
                    SELECT *,
                        sum(eligible) OVER (
                            PARTITION BY component, benchmark_id, backend, arch, source_job, metric
                            ORDER BY run_ts, run_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS pos
                    FROM (
                        -- Every sample, not the per-run mean: U needs the distribution. Zeros are dropped,
                        -- since producers write 0 for a metric they did not measure.
                        SELECT e.run_id AS run_id, e.benchmark_id AS benchmark_id, e.component AS component,
                               e.backend AS backend, e.arch AS arch, e.source_job AS source_job,
                               any(e.name) AS name, any(e.input_shapes) AS input_shapes, min(e.run_ts) AS run_ts,
                               s.1 AS metric, groupArray(s.2) AS cur,
                               max(e.run_props['run.purpose'] != ''
                                   AND lower(e.run_props['run.purpose']) NOT IN (SELECT purpose FROM benchmark_baseline_purposes)) AS off_purpose,
                               max(arrayExists(h -> tupleElement(h, 3) = 'suspect'
                                    AND (tupleElement(h, 1) = toUUID('00000000-0000-0000-0000-000000000000') OR tupleElement(h, 1) = e.benchmark_id)
                                    AND (tupleElement(h, 2) = -1 OR tupleElement(h, 2) = toInt32(toUInt16OrZero(e.run_props['run.repeat']))),
                                    i.hits)) AS suspect,
                               NOT (off_purpose OR suspect) AS eligible
                        FROM v_benchmark_results_enriched AS e
                        ARRAY JOIN arrayFlatten(arrayMap(kv -> arrayMap(v -> (kv.1, v), kv.2),
                                   CAST(e.samples, 'Array(Tuple(String, Array(Float64)))'))) AS s
                        LEFT ANY JOIN
                        (
                            SELECT x.run_id AS run_id,
                                   groupArray((ifNull(x.benchmark_id, toUUID('00000000-0000-0000-0000-000000000000')),
                                               ifNull(toInt32(x.repeat), -1), x.state)) AS hits
                            FROM v_invalidated_runs AS x
                            WHERE x.run_id IS NOT NULL AND x.state IN ('invalidated', 'suspect')
                            GROUP BY x.run_id
                        ) AS i ON i.run_id = e.run_id
                        WHERE s.2 != 0
                          AND e.record_type NOT IN (SELECT record_type FROM benchmark_excluded_record_types)
                          AND NOT arrayExists(h -> tupleElement(h, 3) = 'invalidated'
                                AND (tupleElement(h, 1) = toUUID('00000000-0000-0000-0000-000000000000') OR tupleElement(h, 1) = e.benchmark_id)
                                AND (tupleElement(h, 2) = -1 OR tupleElement(h, 2) = toInt32(toUInt16OrZero(e.run_props['run.repeat']))),
                                i.hits)
                        GROUP BY run_id, benchmark_id, component, backend, arch, source_job, metric
                    )
                )
                WINDOW w AS (PARTITION BY component, benchmark_id, backend, arch, source_job, metric
                             ORDER BY pos RANGE BETWEEN 20 PRECEDING AND 1 PRECEDING)
            ) AS r
            LEFT JOIN (SELECT *, true AS has_policy FROM benchmark_metric_policy) AS p USING (metric)
        )
    )
);

-- One row per run x benchmark: fail on a blocking regression, warn on any other
-- blocking/advisory signal. Reads the live verdicts, so a gate gets its answer right after ingest.
CREATE VIEW IF NOT EXISTS v_benchmark_gate AS
SELECT
    run_id, benchmark_id, component, backend, arch,
    any(name) AS name, any(input_shapes) AS input_shapes, max(ts) AS ts,
    multiIf(countIf(status = 'regressed' AND tier = 'blocking') > 0, 'fail',
            countIf(status IN ('regressed', 'suspected') AND tier IN ('blocking', 'advisory')) > 0, 'warn',
            countIf(status = 'no_baseline') = count(), 'no_baseline',
            'pass') AS gate_status,
    groupArrayIf(metric, status = 'regressed' AND tier = 'blocking') AS regressed_metrics,
    groupArrayIf(metric, status IN ('regressed', 'suspected') AND tier IN ('blocking', 'advisory')
                         AND NOT (status = 'regressed' AND tier = 'blocking')) AS warn_metrics,
    argMaxIf(metric, worse_pct, status IN ('regressed', 'suspected')) AS primary_metric,
    maxIf(worse_pct, status IN ('regressed', 'suspected')) AS worst_delta_pct
FROM v_benchmark_metric_verdicts
GROUP BY run_id, benchmark_id, component, backend, arch;

-- One row per artifact x perf run, folded from v_benchmark_gate, so a promotion step reads the gate
-- for the artifacts it promotes. A view: verdicts are derived live, and their history is
-- benchmark_metric_verdicts.
CREATE VIEW IF NOT EXISTS v_perf_regression_gate AS
SELECT
    ar.artifact_id AS artifact_id, g.run_id AS run_id,
    any(g.component) AS component, any(g.arch) AS arch, max(g.ts) AS ts,
    multiIf(countIf(g.gate_status = 'fail') > 0, 'fail',
            countIf(g.gate_status = 'warn') > 0, 'warn',
            countIf(g.gate_status = 'no_baseline') = count(), 'no_baseline',
            'pass') AS gate_status,
    groupUniqArrayIf(g.benchmark_id, g.gate_status = 'fail') AS failing_benchmarks,
    groupUniqArrayIf(g.benchmark_id, g.gate_status = 'warn') AS warn_benchmarks,
    countIf(g.gate_status = 'no_baseline') AS n_no_baseline,
    count() AS n_benchmarks
FROM v_benchmark_gate AS g
LEFT JOIN (
    SELECT run_id, argMax(artifact_id, (ts, audit_timestamp)) AS artifact_id
    FROM artifact_results
    WHERE result_kind = 'performance'
    GROUP BY run_id
) AS ar USING (run_id)
GROUP BY ar.artifact_id, g.run_id;
