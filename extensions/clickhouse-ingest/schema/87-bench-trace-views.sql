-- Read side of per-step and per-span trace values (record_type step_trace and span; both are in benchmark_excluded_record_types, so the
-- trend, compare and verdict views never see them). Values only: one benchmark_runs row per (rank, prompt, repeat) holds aligned
-- arrays, and these views unnest them. No signal name is built in: the signals playing total / wait / compute / busy are named by the
-- run props role.*, and a signal's cross-rank rule by agg.<signal> (max, min, sum, median, mean). An undeclared rule takes the role's
-- default (total max, others mean) and is flagged. Invalidated repeats are dropped; read them through the views, not the arrays.
-- Filter every view by run_id: the filter reaches the GROUP BY below it, so one run is read, not the history.
-- Depends on 30, 83.

-- One row per (run, benchmark, repeat, rank, prompt, step): the latest row of that key, its arrays unnested by position.
-- step_kind 0 is the first (prefill) step, 1 a later one; cold marks steps excluded from headlines.
CREATE VIEW IF NOT EXISTS v_bench_step_records AS
SELECT
    t.run_id AS run_id, t.benchmark_id AS benchmark_id, t.component AS component, t.backend AS backend, t.repeat AS repeat, t.rank AS rank, t.prompt AS prompt,
    pos,
    toUInt32(t.m['step_index'][pos])                 AS step_index,
    toUInt8(t.m['step_kind'][pos])                   AS step_kind,
    toUInt8(t.m['cold'][pos])                        AS cold,
    if(toUInt8(t.m['step_kind'][pos]) = 0, 'ttft', concat('itl_', leftPad(toString(toUInt32(t.m['step_index'][pos])), 4, '0'))) AS step,
    if(toUInt8(t.m['step_kind'][pos]) = 0, 'ttft', 'itl') AS kind,
    mapApply((k, v) -> (k, v[pos]),
             mapFilter((k, v) -> NOT startsWith(k, 'summary.') AND NOT startsWith(k, 'headline.')
                                 AND k NOT IN ('step_index', 'step_kind', 'cold') AND length(v) >= pos, t.m)) AS metrics,
    t.pr['role.total'] AS total_signal, t.pr['role.wait'] AS wait_signal, t.pr['role.compute'] AS compute_signal, t.pr['role.busy'] AS busy_signal,
    if(mapContains(metrics, total_signal), metrics[total_signal], NULL)     AS total,
    if(mapContains(metrics, wait_signal), metrics[wait_signal], NULL)       AS wait,
    if(mapContains(metrics, compute_signal), metrics[compute_signal], NULL) AS compute,
    if(mapContains(metrics, busy_signal), metrics[busy_signal], NULL)       AS busy,
    if(t.pr['agg.' || total_signal] != '', t.pr['agg.' || total_signal], 'max')       AS total_rule,
    if(t.pr['agg.' || wait_signal] != '', t.pr['agg.' || wait_signal], 'mean')        AS wait_rule,
    if(t.pr['agg.' || compute_signal] != '', t.pr['agg.' || compute_signal], 'mean')  AS compute_rule,
    if(t.pr['agg.' || busy_signal] != '', t.pr['agg.' || busy_signal], 'mean')        AS busy_rule,
    t.pr['agg.' || total_signal] != ''               AS total_rule_declared,
    t.pr['run.world_size']                           AS world_size,
    t.pr['run.reporter_version']                     AS reporter_version,
    t.pr['run.trace_uri']                            AS trace_uri,
    arrayExists(h -> tupleElement(h, 3) = 'suspect'
                 AND (tupleElement(h, 1) = toUUID('00000000-0000-0000-0000-000000000000') OR tupleElement(h, 1) = t.benchmark_id)
                 AND (tupleElement(h, 2) = -1 OR tupleElement(h, 2) = toInt32(t.repeat)), i.hits) AS is_suspect
FROM
(
    SELECT r.run_id AS run_id, r.benchmark_id AS benchmark_id, any(r.component) AS component, r.backend AS backend,
           toUInt16OrZero(r.props['run.repeat']) AS repeat, toUInt16OrZero(r.props['run.rank']) AS rank, r.props['run.prompt'] AS prompt,
           argMax(r.measurements, (r.ts, r.audit_timestamp, r.audit_uuid)) AS m,
           argMax(r.props, (r.ts, r.audit_timestamp, r.audit_uuid)) AS pr
    FROM benchmark_runs AS r
    WHERE mapContains(r.measurements, 'step_kind') AND mapContains(r.measurements, 'step_index')
    GROUP BY run_id, benchmark_id, backend, repeat, rank, prompt
    HAVING length(m['step_kind']) = length(m['step_index']) AND length(m['cold']) IN (0, length(m['step_kind']))
       AND arrayAll(x -> x IN (0, 1), m['step_kind']) AND arrayAll(x -> x IN (0, 1), m['cold'])
       AND arrayAll(x -> isFinite(x) AND x >= 0 AND x < 4294967295 AND x = floor(x), m['step_index'])
       AND length(arrayDistinct(arrayZip(m['step_kind'], m['step_index']))) = length(m['step_index'])
       AND arrayAll(v -> arrayAll(x -> isFinite(x), v), mapValues(m))
) AS t
LEFT ANY JOIN
(
    SELECT x.run_id AS run_id,
           groupArray((ifNull(x.benchmark_id, toUUID('00000000-0000-0000-0000-000000000000')), ifNull(toInt32(x.repeat), -1), x.state)) AS hits
    FROM v_invalidated_runs AS x
    WHERE x.run_id IS NOT NULL AND x.state IN ('invalidated', 'suspect')
    GROUP BY x.run_id
) AS i ON i.run_id = t.run_id
ARRAY JOIN range(1, length(t.m['step_kind']) + 1) AS pos
WHERE NOT arrayExists(h -> tupleElement(h, 3) = 'invalidated'
                       AND (tupleElement(h, 1) = toUUID('00000000-0000-0000-0000-000000000000') OR tupleElement(h, 1) = t.benchmark_id)
                       AND (tupleElement(h, 2) = -1 OR tupleElement(h, 2) = toInt32(t.repeat)), i.hits);

-- The cross-rank value of each role per step, by the role's rule; ranks are matched by (kind, step_index), not by array position. slowest/fastest_rank follow the total; cold is any rank's.
CREATE VIEW IF NOT EXISTS v_bench_step_rank_agg AS
SELECT
    run_id, benchmark_id, backend, repeat, prompt, s_kind AS kind, s_index AS step_index,
    min(pos) AS pos, any(s_step) AS step, max(s_cold) AS cold, count() AS n_ranks,
    any(r_total) AS total_rule, any(r_total_declared) AS total_rule_declared,
    multiIf(any(r_total) = 'max', max(v_total), any(r_total) = 'min', min(v_total), any(r_total) = 'sum', sum(v_total),
            any(r_total) = 'median', median(v_total), avg(v_total))        AS total,
    multiIf(any(r_wait) = 'max', max(v_wait), any(r_wait) = 'min', min(v_wait), any(r_wait) = 'sum', sum(v_wait),
            any(r_wait) = 'median', median(v_wait), avg(v_wait))           AS wait,
    multiIf(any(r_compute) = 'max', max(v_compute), any(r_compute) = 'min', min(v_compute), any(r_compute) = 'sum', sum(v_compute),
            any(r_compute) = 'median', median(v_compute), avg(v_compute))  AS compute,
    multiIf(any(r_busy) = 'max', max(v_busy), any(r_busy) = 'min', min(v_busy), any(r_busy) = 'sum', sum(v_busy),
            any(r_busy) = 'median', median(v_busy), avg(v_busy))           AS busy,
    min(v_total) AS min_total, max(v_total) AS max_total, min(v_wait) AS min_wait, max(v_wait) AS max_wait,
    argMaxIf(rank, (v_total, -toInt32(rank)), v_total IS NOT NULL)         AS slowest_rank,
    argMinIf(rank, (v_total, rank), v_total IS NOT NULL)                   AS fastest_rank,
    argMinIf(rank, (v_wait, rank), v_wait IS NOT NULL)                     AS awaited_rank
FROM
(
    SELECT run_id, benchmark_id, backend, repeat, prompt, rank, pos, step AS s_step, kind AS s_kind, step_index AS s_index, cold AS s_cold,
           total AS v_total, wait AS v_wait, compute AS v_compute, busy AS v_busy,
           total_rule AS r_total, total_rule_declared AS r_total_declared, wait_rule AS r_wait, compute_rule AS r_compute, busy_rule AS r_busy
    FROM v_bench_step_records
)
GROUP BY run_id, benchmark_id, backend, repeat, prompt, kind, step_index;

-- Card x step heatmap: the total minus the fastest rank's total on the same step (a duration difference, no clock alignment), and the wait
-- picture. The awaited rank waits least: the others waited for it. wait_spread_share is the wait spread over the total spread.
CREATE VIEW IF NOT EXISTS v_bench_step_lag AS
SELECT
    r.run_id AS run_id, r.benchmark_id AS benchmark_id, r.backend AS backend, r.repeat AS repeat, r.prompt AS prompt, r.rank AS rank, r.pos AS pos,
    r.step_index AS step_index,
    r.step AS step, r.kind AS kind, r.cold AS cold, r.total AS total, r.wait AS wait, r.compute AS compute, r.busy AS busy,
    r.total - a.min_total                             AS lag,
    r.rank = a.awaited_rank                           AS is_awaited_rank,
    a.max_wait - a.min_wait                           AS wait_spread,
    if(a.max_total > a.min_total, (a.max_wait - a.min_wait) / (a.max_total - a.min_total), NULL) AS wait_spread_share,
    r.is_suspect AS is_suspect
FROM v_bench_step_records AS r
INNER JOIN v_bench_step_rank_agg AS a ON a.run_id = r.run_id AND a.benchmark_id = r.benchmark_id AND a.backend = r.backend
                                      AND a.repeat = r.repeat AND a.prompt = r.prompt AND a.kind = r.kind AND a.step_index = r.step_index;

-- Headline per (run, benchmark, repeat, prompt, kind): the median over warm (not cold) steps of the cross-rank total, plus each rank's
-- own warm median and rank_skew (max - min of those). NULL when the kind has no warm step. The slowest card sets the user-visible
-- time, hence max for a total unless the producer declared another rule.
CREATE VIEW IF NOT EXISTS v_bench_step_headline AS
SELECT
    h.run_id AS run_id, h.benchmark_id AS benchmark_id, h.backend AS backend, h.repeat AS repeat, h.prompt AS prompt, h.kind AS kind,
    h.n_warm_steps AS n_warm_steps, h.value AS headline, h.rule AS rule, h.rule_declared AS rule_declared, h.n_ranks AS n_ranks,
    if(h.n_warm_steps = 0, CAST(map(), 'Map(UInt16, Float64)'), m.rank_medians) AS rank_medians,
    if(h.n_warm_steps = 0, NULL, m.rank_skew)                                   AS rank_skew,
    if(h.n_warm_steps = 0, NULL, m.slowest_rank)                                AS slowest_rank
FROM
(
    SELECT run_id, benchmark_id, backend, repeat, prompt, kind, countIf(total IS NOT NULL AND cold = 0) AS n_warm_steps,
           if(n_warm_steps = 0, NULL, medianIf(total, total IS NOT NULL AND cold = 0)) AS value,
           any(total_rule) AS rule, any(total_rule_declared) AS rule_declared, max(n_ranks) AS n_ranks
    FROM v_bench_step_rank_agg
    GROUP BY run_id, benchmark_id, backend, repeat, prompt, kind
) AS h
LEFT JOIN
(
    SELECT run_id, benchmark_id, backend, repeat, prompt, kind,
           CAST((groupArray(rank), groupArray(med)), 'Map(UInt16, Float64)') AS rank_medians,
           max(med) - min(med) AS rank_skew, argMax(rank, (med, -toInt32(rank))) AS slowest_rank
    FROM
    (
        SELECT run_id, benchmark_id, backend, repeat, prompt, kind, rank, median(total) AS med
        FROM v_bench_step_records
        WHERE total IS NOT NULL AND cold = 0
        GROUP BY run_id, benchmark_id, backend, repeat, prompt, kind, rank
    )
    GROUP BY run_id, benchmark_id, backend, repeat, prompt, kind
) AS m ON m.run_id = h.run_id AND m.benchmark_id = h.benchmark_id AND m.backend = h.backend AND m.repeat = h.repeat AND m.prompt = h.prompt AND m.kind = h.kind;

-- Slow warm steps per rank and kind: modified z = 0.6745 (x - median) / MAD above 3.5, with at least 5 warm steps and a non-zero MAD.
-- The excess over the median is split into wait, compute and the remainder (host).
CREATE VIEW IF NOT EXISTS v_bench_step_outliers AS
SELECT
    run_id, benchmark_id, backend, repeat, prompt, rank, kind, tupleElement(e, 1) AS pos, tupleElement(e, 2) AS step, tupleElement(e, 3) AS value,
    median_total AS median, 0.6745 * (value - median_total) / mad AS z, value - median_total AS excess,
    if(isNaN(tupleElement(e, 4)) OR isNaN(median_wait), NULL, tupleElement(e, 4) - median_wait)         AS excess_wait,
    if(isNaN(tupleElement(e, 5)) OR isNaN(median_compute), NULL, tupleElement(e, 5) - median_compute)   AS excess_compute,
    excess - ifNull(excess_wait, 0) - ifNull(excess_compute, 0) AS excess_host
FROM
(
    SELECT run_id, benchmark_id, backend, repeat, prompt, rank, kind,
           groupArray((pos, step, total, ifNull(wait, nan), ifNull(compute, nan))) AS es,
           arrayReduce('median', arrayMap(x -> tupleElement(x, 3), es))                                     AS median_total,
           arrayReduce('median', arrayMap(x -> abs(tupleElement(x, 3) - median_total), es))                 AS mad,
           arrayReduce('median', arrayFilter(v -> NOT isNaN(v), arrayMap(x -> tupleElement(x, 4), es)))      AS median_wait,
           arrayReduce('median', arrayFilter(v -> NOT isNaN(v), arrayMap(x -> tupleElement(x, 5), es)))      AS median_compute
    FROM v_bench_step_records
    WHERE total IS NOT NULL AND cold = 0
    GROUP BY run_id, benchmark_id, backend, repeat, prompt, rank, kind
    HAVING length(es) >= 5 AND mad > 0
)
ARRAY JOIN es AS e
WHERE 0.6745 * (tupleElement(e, 3) - median_total) / mad > 3.5;

-- Ranks present against run.world_size, per (run, benchmark, repeat, prompt): a partial set of ranks is not a complete measurement;
-- complete means ranks 0..world_size-1 all reported; steps_complete also needs every step seen on every rank (min_ranks_per_step = world_size).
CREATE VIEW IF NOT EXISTS v_bench_rank_coverage AS
SELECT c.run_id AS run_id, c.benchmark_id AS benchmark_id, c.backend AS backend, c.repeat AS repeat, c.prompt AS prompt,
       c.ranks_present AS ranks_present, c.world_size AS world_size, c.complete AS complete,
       a.min_ranks_per_step AS min_ranks_per_step, c.complete AND a.min_ranks_per_step = c.world_size AS steps_complete
FROM
(
    SELECT run_id, benchmark_id, backend, repeat, prompt, uniqExact(rank) AS ranks_present, max(rank) AS max_rank,
           toUInt16OrNull(any(world_size)) AS world_size,
           world_size IS NOT NULL AND ranks_present = world_size AND max_rank < world_size AS complete
    FROM v_bench_step_records
    GROUP BY run_id, benchmark_id, backend, repeat, prompt
) AS c
INNER JOIN
(
    SELECT run_id, benchmark_id, backend, repeat, prompt, min(n_ranks) AS min_ranks_per_step
    FROM v_bench_step_rank_agg
    GROUP BY run_id, benchmark_id, backend, repeat, prompt
) AS a ON a.run_id = c.run_id AND a.benchmark_id = c.benchmark_id AND a.backend = c.backend AND a.repeat = c.repeat AND a.prompt = c.prompt;

-- One row per span: the latest (pattern, rank, prompt, repeat) row unnested, with the pattern's lane and name from its benchmarks row
-- (record_type span; props lane, kernel_name, case_id). A row whose arrays differ in length or hold a non-finite or out-of-range value is skipped, not an error; a span with no benchmarks row (record_type span) is not listed. The raw name
-- and string args are not values and are not here; the trace asset stays the source for them.
CREATE VIEW IF NOT EXISTS v_bench_spans AS
SELECT
    g.run_id AS run_id, g.benchmark_id AS benchmark_id, g.backend AS backend, g.repeat AS repeat, g.rank AS rank, g.prompt AS prompt,
    toUInt32(step) AS step, toUInt32(seq) AS seq, toInt32(parent_seq) AS parent_seq, toUInt8(depth) AS depth, toInt16(layer) AS layer,
    toInt64(start_ns) AS start_ns, toInt64(dur_ns) AS dur_ns, b.lane AS lane, b.name AS name, b.case_id AS case_id
FROM
(
    SELECT r.run_id AS run_id, r.benchmark_id AS benchmark_id, r.backend AS backend, toUInt16OrZero(r.props['run.repeat']) AS repeat,
           toUInt16OrZero(r.props['run.rank']) AS rank, r.props['run.prompt'] AS prompt,
           argMax(r.measurements, (r.ts, r.audit_timestamp, r.audit_uuid)) AS m
    FROM benchmark_runs AS r
    WHERE mapContains(r.measurements, 'dur_ns') AND mapContains(r.measurements, 'start_ns')
    GROUP BY run_id, benchmark_id, backend, repeat, rank, prompt
    HAVING length(m['step']) = length(m['dur_ns']) AND length(m['seq']) = length(m['dur_ns']) AND length(m['parent_seq']) = length(m['dur_ns'])
       AND length(m['depth']) = length(m['dur_ns']) AND length(m['layer']) = length(m['dur_ns']) AND length(m['start_ns']) = length(m['dur_ns'])
       AND arrayAll(x -> isFinite(x) AND x >= 0 AND x <= 9.0e18, m['dur_ns']) AND arrayAll(x -> isFinite(x) AND abs(x) <= 9.0e18, m['start_ns'])
       AND arrayAll(x -> isFinite(x) AND x >= 0 AND x < 4294967295, m['step']) AND arrayAll(x -> isFinite(x) AND x >= 0 AND x < 4294967295, m['seq'])
       AND arrayAll(x -> isFinite(x) AND abs(x) < 2147483647, m['parent_seq']) AND arrayAll(x -> isFinite(x) AND x >= 0 AND x <= 255, m['depth'])
       AND arrayAll(x -> isFinite(x) AND abs(x) <= 32767, m['layer'])
) AS g
INNER JOIN
(
    SELECT benchmark_id, any(props['lane']) AS lane, any(props['kernel_name']) AS name, any(props['case_id']) AS case_id
    FROM benchmarks WHERE props['record_type'] = 'span' GROUP BY benchmark_id
) AS b ON b.benchmark_id = g.benchmark_id
LEFT ANY JOIN
(
    SELECT x.run_id AS run_id,
           groupArray((ifNull(x.benchmark_id, toUUID('00000000-0000-0000-0000-000000000000')), ifNull(toInt32(x.repeat), -1))) AS hits
    FROM v_invalidated_runs AS x
    WHERE x.run_id IS NOT NULL AND x.state = 'invalidated'
    GROUP BY x.run_id
) AS i ON i.run_id = g.run_id
ARRAY JOIN g.m['step'] AS step, g.m['seq'] AS seq, g.m['parent_seq'] AS parent_seq, g.m['depth'] AS depth,
           g.m['layer'] AS layer, g.m['start_ns'] AS start_ns, g.m['dur_ns'] AS dur_ns
WHERE NOT arrayExists(h -> (tupleElement(h, 1) = toUUID('00000000-0000-0000-0000-000000000000') OR tupleElement(h, 1) = g.benchmark_id)
                       AND (tupleElement(h, 2) = -1 OR tupleElement(h, 2) = toInt32(g.repeat)), i.hits);
