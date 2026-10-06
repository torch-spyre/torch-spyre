-- RERUNNABLE
-- Re-key benchmark_id so a compiled kernel keeps one id across compiles: its kernel_name hashes as
-- `<stem>@<rank>` (BenchmarkId.rank_kernels), not with the per-compile `_<16 x [a-z0-9]>` token.
-- The rank is per source file, slowest first, ties by raw name; the raw name moves to the run
-- row's props and the key to benchmarks.props['kernel_key']. Every other hash input is
-- unchanged, so only compiled-kernel rows move.
--
-- The uuid5 below is BenchmarkId.derive in SQL over ingest_xml's _V2_BENCH_ID_KEYS, which must
-- match `keys`. Pinned for review, matching test_identity_golden: component 'torch-spyre', name
-- 'softmax', no tags, kernel_name 'spyre_kernel_v1_fused_softmax#2@1', other keys empty
-- -> d1ebd21d-1a04-5c64-9e52-5dd2c542cb36.
--
-- Safe to repeat: rows the ingest already keyed this way hash to their own id and are skipped;
-- rows written after the snapshot wait for the next pass.

DROP TABLE IF EXISTS benchmark_id_rekey;

CREATE TABLE benchmark_id_rekey
(
    run_id      UUID,
    source_file String,
    old_id      UUID,
    new_id      UUID,
    kernel_name String,
    kernel_key  String,
    snap        DateTime64(3)
)
ENGINE = MergeTree ORDER BY (old_id, run_id);

INSERT INTO benchmark_id_rekey
WITH
    ' \t\n\r\x0B\x0C' AS ws,
    ['record_type', 'config_name', 'input_shapes', 'run_mode', 'kernel_name', 'is_total'] AS keys,
    '_[a-z0-9]{16}(#[0-9]+)?$' AS token
SELECT run_id, source_file, old_id,
       toUUID(lower(concat(
           substring(h, 1, 8), '-', substring(h, 9, 4), '-5', substring(h, 14, 3), '-',
           substring(hex(bitOr(bitAnd(reinterpretAsUInt8(unhex(concat('0', substring(h, 17, 1)))), 3), 8)), 2, 1),
           substring(h, 18, 3), '-', substring(h, 21, 12)))) AS new_id,
       kernel, concat(stem, '@', toString(rnk)), now64(3)
FROM
(
    SELECT *,
        hex(substring(SHA1(concat(unhex('cb0af9bf28585eab9211f51190531bf3'), prefix,
            arrayStringConcat(arrayMap(k -> concat(k, '=', if(k = 'kernel_name',
                lowerUTF8(trimBoth(concat(stem, '@', toString(rnk)), ws)),
                lowerUTF8(trimBoth(props[k], ws)))), keys), ','))), 1, 16)) AS h
    FROM
    (
        SELECT *,
            row_number() OVER (
                PARTITION BY run_id, source_file, prefix, lowerUTF8(trimBoth(stem, ws)),
                             arrayMap(k -> lowerUTF8(trimBoth(props[k], ws)),
                                      arrayFilter(k -> k != 'kernel_name', keys))
                ORDER BY dur DESC, kernel ASC) AS rnk
        FROM
        (
            -- One row per kernel of a file: its slowest backend's mean ranks it.
            SELECT run_id, source_file, old_id, kernel, any(stem) AS stem, any(prefix) AS prefix,
                   any(props) AS props, max(mean) AS dur
            FROM
            (
                SELECT r.run_id AS run_id, r.props['source_file'] AS source_file,
                    r.benchmark_id AS old_id, b.props AS props,
                    if(r.props['kernel_name'] != '', r.props['kernel_name'], b.props['kernel_name']) AS kernel,
                    replaceRegexpOne(kernel, token, '\\1') AS stem,
                    concat(lowerUTF8(trimBoth(b.component, ws)), '|', lowerUTF8(trimBoth(b.name, ws)), '|',
                        arrayStringConcat(arraySort(arrayDistinct(arrayFilter(t -> t != '',
                            arrayMap(t -> lowerUTF8(trimBoth(t, ws)), b.tags)))), ','), '|') AS prefix,
                    if(empty(r.measurements['duration_ms']), 0, arrayAvg(r.measurements['duration_ms'])) AS mean
                FROM benchmark_runs AS r
                INNER JOIN
                (
                    -- An id can hold two rows; every one carries the same hash inputs.
                    SELECT benchmark_id, any(component) AS component, any(name) AS name,
                           any(tags) AS tags, any(props) AS props
                    FROM benchmarks GROUP BY benchmark_id
                ) AS b ON b.benchmark_id = r.benchmark_id
                WHERE startsWith(kernel, 'spyre_kernel_') AND match(kernel, token)
            )
            GROUP BY run_id, source_file, old_id, kernel
        )
    )
)
WHERE new_id != old_id;

-- One identity row per new id, unless a benchmark already holds it; the earliest row's labels.
INSERT INTO benchmarks (ts, benchmark_id, component, name, tags, props)
SELECT min(b.ts), k.new_id, any(b.component), argMin(b.name, b.ts), argMin(b.tags, b.ts),
       mapUpdate(argMin(b.props, b.ts), map('kernel_key', any(k.kernel_key)))
FROM benchmark_id_rekey AS k
INNER JOIN benchmarks AS b ON b.benchmark_id = k.old_id
WHERE k.new_id NOT IN (SELECT benchmark_id FROM benchmarks)
GROUP BY k.new_id;

-- audit_uuid/audit_timestamp carried over: a moved row is the same observation, and its
-- audit_uuid is what makes a repeated pass skip it.
INSERT INTO benchmark_runs
    (ts, run_id, benchmark_id, component, backend, measurements, iterations, props,
     audit_uuid, audit_timestamp)
SELECT r.ts, r.run_id, k.new_id, r.component, r.backend, r.measurements, r.iterations,
       mapUpdate(r.props, map('kernel_name', k.kernel_name)), r.audit_uuid, r.audit_timestamp
FROM benchmark_runs AS r
INNER JOIN benchmark_id_rekey AS k
    ON k.old_id = r.benchmark_id AND k.run_id = r.run_id AND k.source_file = r.props['source_file']
WHERE r.audit_timestamp <= k.snap
  AND r.audit_uuid NOT IN (
      SELECT audit_uuid FROM benchmark_runs
      WHERE benchmark_id IN (SELECT new_id FROM benchmark_id_rekey));

-- Only rows whose copy exists under the new id, so a row written after the snapshot is not lost.
DELETE FROM benchmark_runs
WHERE benchmark_id IN (SELECT old_id FROM benchmark_id_rekey)
  AND audit_uuid IN (SELECT audit_uuid FROM benchmark_runs
                     WHERE benchmark_id IN (SELECT new_id FROM benchmark_id_rekey));

DELETE FROM benchmarks
WHERE benchmark_id IN (SELECT old_id FROM benchmark_id_rekey)
  AND benchmark_id NOT IN (SELECT benchmark_id FROM benchmark_runs);

DROP TABLE benchmark_id_rekey;
