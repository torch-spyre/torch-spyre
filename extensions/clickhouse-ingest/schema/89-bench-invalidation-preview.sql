-- What an invalidation scope hits, for the dry-run preview and the incident history. Depends on 60, 82, 83.

-- One row per (run, benchmark, repeat) with the facts a scope is matched on: when and where the repeat ran.
-- is_series is false for comparison targets and trace records, which no verdict is computed for.
-- A repeat without a placement row has empty nodes/serials and a NULL started_at.
CREATE VIEW IF NOT EXISTS v_bench_repeat_scope AS
SELECT
    r.run_id                                  AS run_id,
    r.benchmark_id                            AS benchmark_id,
    r.repeat                                  AS repeat,
    any(r.component)                          AS component,
    any(r.backend)                            AS backend,
    any(r.arch)                               AS arch,
    any(r.source_job)                         AS source_job,
    any(r.name)                               AS name,
    min(r.run_ts)                             AS run_ts,
    max(r.is_series)                          AS is_series,
    any(p.started_at)                         AS started_at,
    any(p.nodes)                              AS nodes,
    any(p.serials)                            AS serials,
    any(p.placement_key)                      AS placement_key
FROM
(
    SELECT
        e.run_id AS run_id, e.benchmark_id AS benchmark_id, toUInt16OrZero(e.run_props['run.repeat']) AS repeat,
        e.component AS component, e.backend AS backend, e.arch AS arch, e.source_job AS source_job, e.name AS name,
        e.run_ts AS run_ts,
        e.record_type NOT IN (SELECT record_type FROM benchmark_excluded_record_types) AS is_series
    FROM v_benchmark_results_enriched AS e
) AS r
LEFT JOIN
(
    SELECT run_id, repeat, started_at, nodes,
           arrayMap(d -> tupleElement(d, 'serial'), devices) AS serials,
           lower(toString(placement_id)) AS placement_key
    FROM v_run_placements_effective
) AS p ON p.run_id = r.run_id AND p.repeat = r.repeat
GROUP BY r.run_id, r.benchmark_id, r.repeat;

-- Every (invalidation, run, benchmark, repeat) an event hits, whatever its state, so a restored event still shows what it
-- covered and a restore can be previewed. Matching is v_invalidated_runs' (windows apply to started_at, inclusive, a
-- repeat with an unknown started_at matches any window), expanded to benchmark grain; retracted uploads are not included.
CREATE VIEW IF NOT EXISTS v_bench_invalidation_impact AS
SELECT
    s.invalidation_id AS invalidation_id, s.state AS state, s.reason_code AS reason_code, s.scope_kind AS scope_kind,
    r.run_id AS run_id, r.benchmark_id AS benchmark_id, r.repeat AS repeat, r.component AS component,
    r.backend AS backend, r.arch AS arch, r.source_job AS source_job, r.name AS name, r.run_ts AS run_ts,
    r.is_series AS is_series, r.started_at AS started_at
FROM v_invalidation_state AS s
INNER JOIN v_bench_repeat_scope AS r ON r.run_id = s.run_id
WHERE s.scope_kind IN ('run', 'run_benchmark')
  AND (s.scope_kind = 'run'
       OR (r.benchmark_id = s.benchmark_id AND (s.repeat IS NULL OR r.repeat = s.repeat)))

UNION ALL

SELECT
    s.invalidation_id, s.state, s.reason_code, s.scope_kind,
    p.run_id, p.benchmark_id, p.repeat, p.component, p.backend, p.arch, p.source_job, p.name, p.run_ts,
    p.is_series, p.started_at
FROM
(
    SELECT *,
           arrayDistinct(arrayConcat(
               arrayMap(n -> ('node', n), nodes),
               arrayMap(d -> ('device', d), serials),
               [('placement', placement_key)])) AS keys
    FROM v_bench_repeat_scope
    WHERE notEmpty(nodes) OR notEmpty(serials) OR placement_key != ''
) AS p
ARRAY JOIN p.keys AS k
INNER JOIN
(
    SELECT invalidation_id, state, reason_code, scope_kind, scope_key, window_from, window_to
    FROM v_invalidation_state
    WHERE scope_kind IN ('node', 'device', 'placement')
) AS s ON s.scope_kind = tupleElement(k, 1) AND s.scope_key = tupleElement(k, 2)
WHERE p.started_at IS NULL
   OR ((s.window_from IS NULL OR p.started_at >= s.window_from) AND (s.window_to IS NULL OR p.started_at <= s.window_to));

-- One row per invalidation: its latest state and the size of what it covers right now.
CREATE VIEW IF NOT EXISTS v_bench_invalidation_summary AS
SELECT
    s.invalidation_id AS invalidation_id, s.state AS state, s.scope_kind AS scope_kind, s.scope_key AS scope_key,
    s.reason_code AS reason_code, s.reason AS reason, s.actor AS actor, s.created_at AS created_at,
    s.updated_at AS updated_at, s.window_from AS window_from, s.window_to AS window_to,
    uniqExactIf(i.run_id, i.run_id != toUUID('00000000-0000-0000-0000-000000000000')) AS runs,
    uniqExactIf((i.run_id, i.repeat), i.run_id != toUUID('00000000-0000-0000-0000-000000000000')) AS repeats,
    uniqExactIf(i.benchmark_id, i.run_id != toUUID('00000000-0000-0000-0000-000000000000')) AS benchmarks
FROM v_invalidation_state AS s
LEFT JOIN v_bench_invalidation_impact AS i ON i.invalidation_id = s.invalidation_id
GROUP BY s.invalidation_id, s.state, s.scope_kind, s.scope_key, s.reason_code, s.reason, s.actor, s.created_at,
         s.updated_at, s.window_from, s.window_to;

-- The repeats one scope would hit, for a scope that has no event yet (dry-run preview, writer threshold check).
-- Parameterized: SELECT * FROM v_bench_scope_hits(scope_kind = 'node', node = 'n1', run_id = '', benchmark_id = '',
-- repeat = -1, device_serial = '', placement_id = '', window_from = '', window_to = ''); give every parameter, '' / -1 for
-- unused. An empty window bound is open; a bound that does not parse as a timestamp is also treated as open, so callers
-- validate it first. Same matching as v_bench_invalidation_impact (a test pins them equal).
CREATE VIEW IF NOT EXISTS v_bench_scope_hits AS
SELECT
    r.run_id AS run_id, r.benchmark_id AS benchmark_id, r.repeat AS repeat, r.component AS component,
    r.backend AS backend, r.arch AS arch, r.source_job AS source_job, r.name AS name, r.run_ts AS run_ts,
    r.is_series AS is_series, r.started_at AS started_at
FROM v_bench_repeat_scope AS r
WHERE multiIf(
    {scope_kind:String} = 'run',
        r.run_id = toUUIDOrZero({run_id:String}),
    {scope_kind:String} = 'run_benchmark',
        r.run_id = toUUIDOrZero({run_id:String}) AND r.benchmark_id = toUUIDOrZero({benchmark_id:String})
        AND ({repeat:Int32} < 0 OR r.repeat = {repeat:Int32}),
    {scope_kind:String} IN ('node', 'device', 'placement'),
        multiIf({scope_kind:String} = 'node', has(r.nodes, {node:String}),
                {scope_kind:String} = 'device', has(r.serials, {device_serial:String}),
                r.placement_key = lower({placement_id:String}))
        AND (r.started_at IS NULL
             OR (ifNull(r.started_at >= parseDateTime64BestEffortOrNull({window_from:String}, 3, 'UTC'), true)
                 AND ifNull(r.started_at <= parseDateTime64BestEffortOrNull({window_to:String}, 3, 'UTC'), true))),
    false);
