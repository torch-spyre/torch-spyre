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
