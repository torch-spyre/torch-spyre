-- RERUNNABLE
-- Re-key test_case_id to the current recipe: identity tags only (as 006), and the test name
-- hashed with its case, so sibling tests that differ only by case (test_T / test_t) get two ids.
-- component and classname stay case-folded; an all-lowercase name keeps its id.
--
-- Supersedes 006 for reruns: 006 hashes the lowercased name, so rerunning it would undo this.
-- The uuid5 below is CaseId.derive in SQL. Pinned for review, matching test_identity_golden:
-- component 'Torch-Spyre', classname 'TestViewOps', name ' test_T_spyre ', no tags
-- -> uuid5(ns, 'torch-spyre|testviewops|test_T_spyre|').
-- The four arrays must equal RUN_CONTEXT_TAG_NAMESPACES, RESULT_TAG_NAMESPACES and
-- LEGACY_TAG_ALIASES (keys, values) in identity.py; a test pins that.
--
-- Runs store no name, so a pair that already shares an id moves under its stored name; only new
-- ingests split it. Safe to repeat (`apply_schema --rerun`): rows written after the snapshot are
-- left for the next pass, never deleted unmoved; case_id_rekey_runs survives a failed pass so its
-- counters get recounted.

DROP TABLE IF EXISTS case_id_rekey;

CREATE TABLE case_id_rekey
(
    old_id       UUID,
    new_id       UUID,
    ts           DateTime,
    component    LowCardinality(String),
    classname    String,
    name         String,
    id_tags      Array(LowCardinality(String)),
    run_tags     Array(LowCardinality(String)),
    result_props Map(LowCardinality(String), String),
    snap         DateTime64(3)
)
ENGINE = MergeTree ORDER BY old_id;

-- Grouped by old_id: test_cases is written check-then-insert, so an id can hold two rows.
INSERT INTO case_id_rekey
WITH
    ['platform', 'testtype', 'cadence'] AS ctx,
    ['refcoverage'] AS res,
    -- str.strip()'s ASCII whitespace; lowerUTF8 and str.lower() differ only on rare non-ASCII.
    ' \t\n\r\x0B\x0C' AS ws,
    ['nightly', 'weekly', 'fvt', 'svt', 'spyre-inference', 'spyre-backend', 'torch-spyre'] AS legacy,
    ['cadence__nightly', 'cadence__weekly', 'testtype__fvt', 'testtype__svt',
     'domain__spyre-inference', 'domain__spyre-backend', 'domain__torch-spyre'] AS aliased
SELECT old_id, any(new_id), min(ts), any(component), any(classname), any(name),
       any(id_tags), any(run_tags), any(result_props), now64(3)
FROM
(
    SELECT
        test_case_id AS old_id,
        toUUID(lower(concat(
            substring(h, 1, 8), '-', substring(h, 9, 4), '-5', substring(h, 14, 3), '-',
            substring(hex(bitOr(bitAnd(reinterpretAsUInt8(unhex(concat('0', substring(h, 17, 1)))), 3), 8)), 2, 1),
            substring(h, 18, 3), '-', substring(h, 21, 12)))) AS new_id,
        ts, component, classname, name,
        arraySort(arrayDistinct(id_tags)) AS id_tags, run_tags, result_props
    FROM
    (
        SELECT *,
            hex(substring(SHA1(concat(
                unhex('cb0af9bf28585eab9211f51190531bf3'),
                lowerUTF8(trimBoth(component, ws)), '|',
                lowerUTF8(trimBoth(classname, ws)), '|',
                trimBoth(name, ws), '|',
                arrayStringConcat(arraySort(arrayDistinct(
                    arrayMap(t -> lowerUTF8(trimBoth(t, ws)), id_tags))), ','))), 1, 16)) AS h
        FROM
        (
            SELECT *,
                arrayFilter(t -> NOT has(ctx, splitByString('__', lowerUTF8(trimBoth(t, ws)))[1])
                             AND NOT has(res, splitByString('__', lowerUTF8(trimBoth(t, ws)))[1]), ctags) AS id_tags,
                arraySort(arrayDistinct(arrayFilter(
                    t -> has(ctx, splitByString('__', lowerUTF8(trimBoth(t, ws)))[1]), ctags))) AS run_tags,
                arrayFilter(t -> has(res, splitByString('__', lowerUTF8(trimBoth(t, ws)))[1]), ctags) AS rtags,
                mapFromArrays(
                    arrayMap(t -> concat('result.', splitByString('__', lowerUTF8(trimBoth(t, ws)))[1]), rtags),
                    arrayMap(t -> if(position(t, '__') > 0, substring(t, position(t, '__') + 2), ''), rtags)
                ) AS result_props
            FROM
            (
                SELECT *,
                    arrayMap(t -> if(has(legacy, lowerUTF8(trimBoth(t, ws))),
                                     transform(lowerUTF8(trimBoth(t, ws)), legacy, aliased, ''), t),
                             arrayFilter(t -> trimBoth(t, ws) != '', tags)) AS ctags
                FROM test_cases
            )
        )
    )
)
WHERE new_id != old_id
GROUP BY old_id;

-- One identity row per new id, unless a case already holds it.
INSERT INTO test_cases (ts, test_case_id, component, classname, name, tags)
SELECT min(ts), new_id, any(component), argMin(classname, ts), argMin(name, ts), argMin(id_tags, ts)
FROM case_id_rekey
WHERE new_id NOT IN (SELECT test_case_id FROM test_cases)
GROUP BY new_id;

CREATE TABLE IF NOT EXISTS case_id_rekey_runs (run_id UUID) ENGINE = MergeTree ORDER BY run_id;

INSERT INTO case_id_rekey_runs
SELECT DISTINCT r.run_id
FROM test_case_runs AS r
INNER JOIN case_id_rekey AS k ON k.old_id = r.test_case_id
WHERE r.audit_timestamp <= k.snap;

-- audit_uuid/audit_timestamp carried over: a moved row is the same observation, and its
-- audit_uuid is what makes a repeated pass skip it. The row's own props win over a result tag's.
INSERT INTO test_case_runs
    (ts, run_id, test_case_id, component, status, duration_s, fail_message, props, tags,
     measurements, audit_uuid, audit_timestamp)
SELECT r.ts, r.run_id, k.new_id, r.component, r.status, r.duration_s, r.fail_message,
       mapUpdate(k.result_props, r.props), arraySort(arrayDistinct(arrayConcat(r.tags, k.run_tags))),
       r.measurements, r.audit_uuid, r.audit_timestamp
FROM test_case_runs AS r
INNER JOIN case_id_rekey AS k ON k.old_id = r.test_case_id
WHERE r.audit_timestamp <= k.snap
  AND r.audit_uuid NOT IN (
      SELECT audit_uuid FROM test_case_runs
      WHERE test_case_id IN (SELECT new_id FROM case_id_rekey));

-- Only rows whose copy exists under the new id: a row that committed after the move read the
-- table was never copied, and waits for the next pass instead of being lost.
DELETE FROM test_case_runs
WHERE test_case_id IN (SELECT old_id FROM case_id_rekey)
  AND audit_uuid IN (SELECT audit_uuid FROM test_case_runs
                     WHERE test_case_id IN (SELECT new_id FROM case_id_rekey));

-- An old id that a post-snapshot row still uses keeps its identity row, for the next pass.
DELETE FROM test_cases
WHERE test_case_id IN (SELECT old_id FROM case_id_rekey)
  AND test_case_id NOT IN (SELECT test_case_id FROM test_case_runs);

-- The move fired run_case_counters_mv again and the DELETE subtracted nothing: recount the
-- touched runs from scratch. A row inserted into a touched run between this DELETE and the
-- recount is counted twice, so run a pass while no ingest is writing.
DELETE FROM run_case_counters WHERE run_id IN (SELECT run_id FROM case_id_rekey_runs);

INSERT INTO run_case_counters (run_id, component, total_tests, passed, failed, errors, skipped, xfail, xpass)
SELECT
    run_id,
    component,
    count(),
    countIf(status = 'passed'),
    countIf(status = 'failed'),
    countIf(status = 'error'),
    countIf(status = 'skipped'),
    countIf(status = 'xfail'),
    countIf(status = 'xpass')
FROM test_case_runs
WHERE run_id IN (SELECT run_id FROM case_id_rekey_runs)
GROUP BY run_id, component;

DROP TABLE case_id_rekey_runs;

DROP TABLE case_id_rekey;
