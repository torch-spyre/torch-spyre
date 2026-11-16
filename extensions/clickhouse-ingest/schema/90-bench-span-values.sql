-- OPTIONAL speed path for v_bench_spans (87): an insert-triggered materialized view keeps one row per span in bench_span_values, so a
-- cross-run span query reads sorted columns instead of unnesting whole benchmark_runs rows. Drop this file and nothing else changes.
-- An MV sees only rows inserted after it exists: create it before the first span row, or backfill the rows already present with
--   INSERT INTO bench_span_values SELECT <the SELECT of mv_bench_span_values>  (see docs/trace-views.md).
-- The MV never throws on a malformed span row (arrays of different lengths, NaN or inf), which would fail the benchmark_runs insert it rides on:
-- arrayZipUnaligned makes the ARRAY JOIN safe whatever order the optimizer runs it in, and the WHERE drops such a row before any cast.
-- Plain MergeTree: the read view picks the newest upload itself (by ts, then audit_timestamp), which a merge-time collapse on audit_timestamp alone could contradict.
-- Raw names and args are not kept; invalidation is applied at read, so the MV never deletes. Depends on 30, 83, 87.
CREATE TABLE IF NOT EXISTS bench_span_values
(
    run_id          UUID,
    benchmark_id    UUID,
    backend         LowCardinality(String),
    repeat          UInt16,
    rank            UInt16,
    prompt          LowCardinality(String),
    step            UInt32,
    seq             UInt32,
    parent_seq      Int32,
    depth           UInt8,
    layer           Int16,
    start_ns        Int64 CODEC(Delta(8), ZSTD(1)),
    dur_ns          Int64 CODEC(ZSTD(1)),
    ts              DateTime,
    audit_timestamp DateTime64(3)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(audit_timestamp)
ORDER BY (run_id, benchmark_id, backend, repeat, rank, prompt, step, seq);

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_bench_span_values TO bench_span_values AS
SELECT r.run_id AS run_id, r.benchmark_id AS benchmark_id, r.backend AS backend,
       toUInt16OrZero(r.props['run.repeat']) AS repeat, toUInt16OrZero(r.props['run.rank']) AS rank, r.props['run.prompt'] AS prompt,
       toUInt32(assumeNotNull(tupleElement(z, 1))) AS step, toUInt32(assumeNotNull(tupleElement(z, 2))) AS seq,
       toInt32(assumeNotNull(tupleElement(z, 3))) AS parent_seq, toUInt8(assumeNotNull(tupleElement(z, 4))) AS depth,
       toInt16(assumeNotNull(tupleElement(z, 5))) AS layer, toInt64(assumeNotNull(tupleElement(z, 6))) AS start_ns,
       toInt64(assumeNotNull(tupleElement(z, 7))) AS dur_ns, r.ts AS ts, r.audit_timestamp AS audit_timestamp
FROM benchmark_runs AS r
ARRAY JOIN arrayZipUnaligned(r.measurements['step'], r.measurements['seq'], r.measurements['parent_seq'], r.measurements['depth'],
                             r.measurements['layer'], r.measurements['start_ns'], r.measurements['dur_ns']) AS z
WHERE mapContains(r.measurements, 'dur_ns') AND mapContains(r.measurements, 'start_ns')
  AND length(r.measurements['step']) = length(r.measurements['dur_ns']) AND length(r.measurements['seq']) = length(r.measurements['dur_ns'])
  AND length(r.measurements['parent_seq']) = length(r.measurements['dur_ns']) AND length(r.measurements['depth']) = length(r.measurements['dur_ns'])
  AND length(r.measurements['layer']) = length(r.measurements['dur_ns']) AND length(r.measurements['start_ns']) = length(r.measurements['dur_ns'])
  AND arrayAll(x -> isFinite(x) AND x >= 0 AND x <= 9.0e18, r.measurements['dur_ns']) AND arrayAll(x -> isFinite(x) AND abs(x) <= 9.0e18, r.measurements['start_ns'])
  AND arrayAll(x -> isFinite(x) AND x >= 0 AND x < 4294967295, r.measurements['step']) AND arrayAll(x -> isFinite(x) AND x >= 0 AND x < 4294967295, r.measurements['seq'])
  AND arrayAll(x -> isFinite(x) AND abs(x) < 2147483647, r.measurements['parent_seq']) AND arrayAll(x -> isFinite(x) AND x >= 0 AND x <= 255, r.measurements['depth'])
  AND arrayAll(x -> isFinite(x) AND abs(x) <= 32767, r.measurements['layer']);

-- v_bench_spans' columns from bench_span_values: of each (run, benchmark, repeat, rank, prompt) only the newest insert counts (a re-upload
-- replaces, never mixes, rows duplicated by a retried insert count once), then the same lane / name join and invalidation filter. Filter by run_id.
CREATE VIEW IF NOT EXISTS v_bench_spans_fast AS
SELECT s.run_id AS run_id, s.benchmark_id AS benchmark_id, s.backend AS backend, s.repeat AS repeat, s.rank AS rank, s.prompt AS prompt, s.step AS step,
       s.seq AS seq, s.parent_seq AS parent_seq, s.depth AS depth, s.layer AS layer, s.start_ns AS start_ns, s.dur_ns AS dur_ns,
       b.lane AS lane, b.name AS name, b.case_id AS case_id
FROM bench_span_values AS s
INNER JOIN
(
    SELECT run_id, benchmark_id, backend, repeat, rank, prompt, max((ts, audit_timestamp)) AS newest
    FROM bench_span_values GROUP BY run_id, benchmark_id, backend, repeat, rank, prompt
) AS l ON l.run_id = s.run_id AND l.benchmark_id = s.benchmark_id AND l.backend = s.backend AND l.repeat = s.repeat AND l.rank = s.rank AND l.prompt = s.prompt
      AND l.newest = (s.ts, s.audit_timestamp)
INNER JOIN
(
    SELECT benchmark_id, any(props['lane']) AS lane, any(props['kernel_name']) AS name, any(props['case_id']) AS case_id
    FROM benchmarks WHERE props['record_type'] = 'span' GROUP BY benchmark_id
) AS b ON b.benchmark_id = s.benchmark_id
LEFT ANY JOIN
(
    SELECT x.run_id AS run_id,
           groupArray((ifNull(x.benchmark_id, toUUID('00000000-0000-0000-0000-000000000000')), ifNull(toInt32(x.repeat), -1))) AS hits
    FROM v_invalidated_runs AS x
    WHERE x.run_id IS NOT NULL AND x.state = 'invalidated'
    GROUP BY x.run_id
) AS i ON i.run_id = s.run_id
WHERE NOT arrayExists(h -> (tupleElement(h, 1) = toUUID('00000000-0000-0000-0000-000000000000') OR tupleElement(h, 1) = s.benchmark_id)
                       AND (tupleElement(h, 2) = -1 OR tupleElement(h, 2) = toInt32(s.repeat)), i.hits)
LIMIT 1 BY s.run_id, s.benchmark_id, s.backend, s.repeat, s.rank, s.prompt, s.step, s.seq;
