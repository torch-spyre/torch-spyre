-- Flattened view of a benchmark repeat for diff and "vary one dimension" queries, and its aggregate over repeats.
-- Built on v_benchmark_results_enriched (benchmark_runs + benchmarks + artifact_results, sample arrays reduced to scalars).
-- An attribute that was not recorded is ABSENT from attrs, never an empty string: a controlled set may only be formed when every
-- unheld key is declared relaxed. Engine-written run props already carry a namespace (case, system, input, subject, env, run); legacy
-- producer keys and unit.* are put under run.*. Grouping values (tier, suite) are run props as of the run, not part of the benchmark id.
-- attrs sources are filtered of empty values first, then merged in this order, a later source replacing an earlier one's key (each key appears once): benchmark props (case.*), legacy run
-- props renamed to run.<k>, namespaced run props, identity (backend, arch, component, name, subject.id), fingerprint, labels.
-- A row without run.repeat is repeat 0. Invalidations and suspicion are matched per (run, benchmark, repeat), computed once.
-- Depends on 20, 30, 49, 60, 81, 82, 83.
CREATE VIEW IF NOT EXISTS v_bench_attrs AS
SELECT
    ts, run_id, benchmark_id, subject_id, component, backend, name, measurements, iterations, repeat, env_id, is_suspect,
    mapFromArrays(
        arrayFilter((k, i) -> NOT has(arraySlice(mapKeys(merged), i + 1), k), mapKeys(merged), arrayEnumerate(mapKeys(merged))),
        arrayFilter((v, k, i) -> NOT has(arraySlice(mapKeys(merged), i + 1), k), mapValues(merged), mapKeys(merged), arrayEnumerate(mapKeys(merged)))
    ) AS attrs
FROM
(
    SELECT
        e.ts                                              AS ts,
        e.run_id                                          AS run_id,
        e.benchmark_id                                    AS benchmark_id,
        e.artifact_id                                     AS subject_id,
        e.component                                       AS component,
        e.backend                                         AS backend,
        e.name                                            AS name,
        e.measurements                                    AS measurements,
        e.iterations                                      AS iterations,
        toUInt16OrZero(e.run_props['run.repeat'])         AS repeat,
        nullIf(toUUIDOrNull(e.run_props['env.id']), toUUID('00000000-0000-0000-0000-000000000000'))               AS env_id,
        arrayExists(h -> tupleElement(h, 3) = 'suspect'
                     AND (tupleElement(h, 1) = toUUID('00000000-0000-0000-0000-000000000000') OR tupleElement(h, 1) = e.benchmark_id)
                     AND (tupleElement(h, 2) = -1 OR tupleElement(h, 2) = toInt32(toUInt16OrZero(e.run_props['run.repeat']))),
                    i.hits)                               AS is_suspect,
        mapConcat(
            mapFilter((k, v) -> v != '', mapFromArrays(arrayMap(k -> concat('case.', k), mapKeys(e.bench_props)), mapValues(e.bench_props))),
            mapApply((k, v) -> (concat('run.', k), v), mapFilter((k, v) -> v != '' AND NOT match(k, '^(case|system|input|subject|env|run)[.]'), e.run_props)),
            mapFilter((k, v) -> v != '' AND match(k, '^(case|system|input|subject|env|run)[.]') AND NOT (k = 'env.id' AND nullIf(toUUIDOrNull(e.run_props['env.id']), toUUID('00000000-0000-0000-0000-000000000000')) IS NULL), e.run_props),
            mapFilter((k, v) -> v != '', map('backend', e.backend, 'arch', e.arch, 'component', e.component, 'name', e.name)),
            if(e.artifact_id = toUUID('00000000-0000-0000-0000-000000000000'), map(), map('subject.id', toString(e.artifact_id))),
            mapFilter((k, v) -> v != '', mapFromArrays(arrayMap(k -> concat('env.fp.', k), mapKeys(f.fingerprint)), mapValues(f.fingerprint))),
            mapFilter((k, v) -> v != '', mapFromArrays(arrayMap(k -> concat('env.label.', k), mapKeys(f.labels)), mapValues(f.labels)))
        )                                                 AS merged
    FROM v_benchmark_results_enriched AS e
    LEFT ANY JOIN v_environment_fingerprints AS f ON f.env_id = nullIf(toUUIDOrNull(e.run_props['env.id']), toUUID('00000000-0000-0000-0000-000000000000'))
    LEFT ANY JOIN
    (
        SELECT x.run_id AS run_id,
               groupArray((ifNull(x.benchmark_id, toUUID('00000000-0000-0000-0000-000000000000')),
                           ifNull(toInt32(x.repeat), -1), x.state)) AS hits
        FROM v_invalidated_runs AS x
        WHERE x.run_id IS NOT NULL AND x.state IN ('invalidated', 'suspect')
        GROUP BY x.run_id
    ) AS i ON i.run_id = e.run_id
    WHERE NOT arrayExists(h -> tupleElement(h, 3) = 'invalidated'
                           AND (tupleElement(h, 1) = toUUID('00000000-0000-0000-0000-000000000000') OR tupleElement(h, 1) = e.benchmark_id)
                           AND (tupleElement(h, 2) = -1 OR tupleElement(h, 2) = toInt32(toUInt16OrZero(e.run_props['run.repeat']))),
                          i.hits)
);

-- One row per (run, benchmark, metric): the median over valid repeats and their spread. Rows are first collapsed to one per
-- (run, benchmark, repeat, metric), so a re-ingested leg does not count twice; of two rows in the same second the value and attrs
-- come from one row (benchmark_runs.ts has second resolution and v_benchmark_results_enriched does not expose audit_timestamp,
-- so a content hash breaks the tie). env_id is that of the lowest repeat that recorded one. attrs is that of the lowest valid repeat; per-repeat
-- differences (placement, run.repeat) are read from v_bench_attrs. is_suspect is true if any valid repeat is suspect.
CREATE VIEW IF NOT EXISTS v_bench_results AS
SELECT
    run_id,
    benchmark_id,
    subject_id,
    backend,
    metric,
    count()                                           AS n_repeats,
    median(v)                                         AS value,
    min(v)                                            AS value_min,
    max(v)                                            AS value_max,
    if(count() > 1 AND avg(v) != 0, stddevSamp(v) / abs(avg(v)) * 100, NULL) AS cv_pct,
    min(ts)                                           AS ts,
    argMinIf(env_id, repeat, env_id IS NOT NULL)      AS env_id,
    max(is_suspect)                                   AS is_suspect,
    argMin(attrs, repeat)                             AS attrs
FROM
(
    SELECT a.run_id AS run_id, a.benchmark_id AS benchmark_id, a.subject_id AS subject_id, a.backend AS backend,
           a.repeat AS repeat, k AS metric,
           tupleElement(argMax(tuple(a.measurements[k], a.attrs), (a.ts, sipHash64(a.measurements[k], a.attrs))), 1) AS v,
           min(a.ts) AS ts,
           argMinIf(a.env_id, a.ts, a.env_id IS NOT NULL) AS env_id,
           max(a.is_suspect) AS is_suspect,
           tupleElement(argMax(tuple(a.measurements[k], a.attrs), (a.ts, sipHash64(a.measurements[k], a.attrs))), 2) AS attrs
    FROM v_bench_attrs AS a
    ARRAY JOIN mapKeys(a.measurements) AS k
    GROUP BY a.run_id, a.benchmark_id, a.subject_id, a.backend, a.repeat, k
)
GROUP BY run_id, benchmark_id, subject_id, backend, metric;
