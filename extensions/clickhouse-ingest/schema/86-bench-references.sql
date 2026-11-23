-- Comparison targets (references) as a read model for the Perf Lab: one row per reference version with its provenance, comparability,
-- the verdict link (run_id of the value rows) and the state its key's slot is in. Values stay in benchmark_runs (run.ref_only = '1'),
-- read run-scoped through run_id; there is no time window, a reference is not a measurement of ours.
-- A slot is one ref.key; its newest 'reference'-family tag row decides: 'ref-current' makes that artifact current, 'ref-retired' retires
-- it, and every other version of the key is superseded. A published version of a key with no tag row yet is 'untagged'.
-- A ref-current row counts only while its artifact's upload is published, so a version that is not published yet (or
-- was retracted) never holds the slot and the previous current version stays current meanwhile, with no write needed
-- after a retract. A ref-retired row is the operator's statement about the key and counts whatever happens to that version.
-- Only versions whose upload is published are listed (an appended round of an upload does not make a second row).
-- ref.clean / ref.warnings record the cleanliness of the run a reference was made from; ref.approval_override
-- and ref.approval_self flag an admin approval (without the binding, or of their own upload).
-- Depends on 20, 30, 49, 50.
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
    a.ts                                             AS registered_at,
    u.upload_id                                      AS upload_id,
    a.props['ref.digest']                            AS digest,
    a.props['ref.clean']                             AS clean,
    a.props['ref.warnings']                          AS warnings,
    a.props['ref.pinning']                           AS pinning,
    a.props['ref.source_run_id']                     AS source_run_id,
    a.props['ref.approval_reason']                   AS approval_reason,
    a.props['ref.approval_override'] = '1'           AS approval_override,
    a.props['ref.approval_self'] = '1'               AS approval_self
FROM v_artifacts AS a
INNER JOIN v_result_uploads AS u ON u.artifact_id = a.artifact_id AND u.state = 'published' AND NOT u.is_round
LEFT JOIN
(
    SELECT t.props['ref.key']                                           AS ref_key,
           argMax(tuple(t.tag, t.artifact_id), (t.ts, t.audit_timestamp)) AS slot
    FROM artifact_tags AS t
    WHERE t.tag_family = 'reference'
      AND (t.tag = 'ref-retired' OR t.artifact_id IN (SELECT artifact_id FROM v_result_uploads WHERE state = 'published' AND NOT is_round))
    GROUP BY ref_key
) AS s ON s.ref_key = a.props['ref.key']
WHERE a.component = 'reference';

-- The value rows of each listed version, one per (case, backend): what an engine compares a run to.
-- State is carried so a reader can take the current version of a key or any named one.
CREATE VIEW IF NOT EXISTS v_bench_reference_values AS
SELECT
    v.ref_key        AS ref_key,
    v.version        AS version,
    v.state          AS state,
    v.artifact_id    AS artifact_id,
    v.origin         AS origin,
    v.label          AS label,
    v.comparability  AS comparability,
    v.clean          AS clean,
    v.warnings       AS warnings,
    r.component      AS component,
    r.benchmark_id   AS benchmark_id,
    r.backend        AS backend,
    r.measurements   AS measurements,
    r.iterations     AS iterations,
    r.props          AS row_props
FROM v_bench_references AS v
INNER JOIN benchmark_runs AS r ON r.run_id = v.run_id
WHERE r.props['run.ref_only'] = '1' AND r.props['ref.key'] = v.ref_key AND r.props['ref.version'] = v.version;
