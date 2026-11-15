-- Comparison targets (references) as a read model for the Perf Lab: one row per reference version with its provenance, comparability,
-- the verdict link (run_id of the value rows) and the state its key's slot is in. Values stay in benchmark_runs (run.ref_only = '1'),
-- read run-scoped through run_id; there is no time window, a reference is not a measurement of ours.
-- A slot is one ref.key; its newest 'reference'-family tag row decides: 'ref-current' makes that artifact current, 'ref-retired' retires
-- it, and every other version of the key is superseded. A published version of a key with no tag row yet is 'untagged'.
-- Only versions whose upload is published are listed (an appended round of an upload does not make a second row).
-- Depends on 20, 49, 50.
CREATE VIEW IF NOT EXISTS v_bench_references AS
SELECT
    a.artifact_id                                    AS artifact_id,
    a.props['ref.key']                               AS ref_key,
    a.props['ref.version']                           AS version,
    a.props['ref.class']                             AS ref_class,
    a.props['ref.origin']                            AS origin,
    a.props['ref.label']                             AS label,
    a.arch                                           AS arch,
    a.props['ref.as_of']                             AS as_of,
    a.props['ref.approved_by']                       AS approved_by,
    a.props['ref.approved_at']                       AS approved_at,
    u.uploaded_by                                    AS submitted_by,
    u.run_id                                         AS run_id,
    u.state                                          AS upload_state,
    mapFilter((k, v) -> startsWith(k, 'ref.comparability.'), a.props) AS comparability,
    multiIf(tupleElement(s.slot, 1) = '', 'untagged',
            tupleElement(s.slot, 2) = a.artifact_id AND tupleElement(s.slot, 1) = 'ref-current', 'current',
            tupleElement(s.slot, 2) = a.artifact_id AND tupleElement(s.slot, 1) = 'ref-retired', 'retired',
            'superseded')                            AS state,
    tupleElement(s.slot, 2)                                         AS slot_artifact_id,
    a.ts                                             AS registered_at
FROM v_artifacts AS a
INNER JOIN v_result_uploads AS u ON u.artifact_id = a.artifact_id AND u.state = 'published' AND NOT u.is_round
LEFT JOIN
(
    SELECT t.props['ref.key']                                           AS ref_key,
           argMax(tuple(t.tag, t.artifact_id), (t.ts, t.audit_timestamp)) AS slot
    FROM artifact_tags AS t
    WHERE t.tag_family = 'reference'
    GROUP BY ref_key
) AS s ON s.ref_key = a.props['ref.key']
WHERE a.component = 'reference';
