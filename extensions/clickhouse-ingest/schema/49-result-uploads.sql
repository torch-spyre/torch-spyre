-- Results uploaded by people (the v2 dashboard), one row per state change of an upload: the audit
-- log and, since v2 is append-only, the retract marker. Each upload is its own run, written by
-- `results --from-bundle` like any bundle; a retracted upload's run drops out of every view.
CREATE TABLE IF NOT EXISTS result_uploads
(
    ts             DateTime64(3, 'UTC') DEFAULT now64(3),
    upload_id      String,                        -- uuid7 per upload, its run key manual:<uploaded_by>:<upload_id>
    state          LowCardinality(String),        -- e.g. 'published' (also 'rejected', 'failed', 'retracted')
    actor          String,                        -- who made this change: the uploader, or an admin
    uploaded_by    String,                        -- the verified cluster identity, e.g. 'alice@ibm.com'
    artifact_id    UUID,                          -- nil until ingested
    run_id         UUID,                          -- nil until ingested
    kind           LowCardinality(String),        -- e.g. 'junit' (also 'vllm')
    test_type      LowCardinality(String),        -- e.g. 'regression'
    arch           LowCardinality(String),        -- e.g. 'x86_64'
    filename       String,                        -- what was uploaded, e.g. 'bundle.tgz' or 'a.xml, b.xml'
    size_bytes     UInt64,
    content_sha256 String,                        -- the bundle as uploaded (offline.digest): finds identical uploads
    bundle_url     String,                        -- where the sealed bundle is kept (the verdict's bundle_url)
    reason         String DEFAULT '',             -- why it was rejected, failed or retracted
    props          Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_state CHECK state IN ('published', 'rejected', 'failed', 'retracted')
)
ENGINE = MergeTree()
ORDER BY (upload_id, ts);

-- One row per upload at its latest state. A retract carries only state, actor and reason, so the
-- descriptive columns come from the events that set them.
CREATE VIEW IF NOT EXISTS v_result_uploads AS
SELECT
    upload_id,
    argMax(state, ts)                                                AS state,
    argMax(actor, ts)                                                AS actor,
    any(uploaded_by)                                                 AS uploaded_by,
    argMaxIf(artifact_id, ts, artifact_id != toUUID('00000000-0000-0000-0000-000000000000')) AS artifact_id,
    argMaxIf(run_id, ts, run_id != toUUID('00000000-0000-0000-0000-000000000000'))           AS run_id,
    anyIf(kind, kind != '')                                          AS kind,
    anyIf(test_type, test_type != '')                                AS test_type,
    anyIf(arch, arch != '')                                          AS arch,
    anyIf(filename, filename != '')                                  AS filename,
    max(size_bytes)                                                  AS size_bytes,
    anyIf(content_sha256, content_sha256 != '')                      AS content_sha256,
    argMaxIf(bundle_url, ts, bundle_url != '')                       AS bundle_url,
    argMax(reason, ts)                                               AS reason,
    min(ts)                                                          AS uploaded_at,
    max(ts)                                                          AS updated_at
FROM result_uploads
GROUP BY upload_id;

CREATE VIEW IF NOT EXISTS v_retracted_upload_runs AS
SELECT run_id
FROM v_result_uploads
WHERE state = 'retracted' AND run_id != toUUID('00000000-0000-0000-0000-000000000000');

-- artifact_results as every view reads it: a retracted upload's run is gone, uploads count like CI.
CREATE VIEW IF NOT EXISTS v_artifact_results_published AS
SELECT *
FROM artifact_results
WHERE run_id NOT IN (SELECT run_id FROM v_retracted_upload_runs);

-- Opt-in: CI's own verdicts, without uploads.
CREATE VIEW IF NOT EXISTS v_artifact_results_ci AS
SELECT *
FROM v_artifact_results_published
WHERE props['channel'] != 'dashboard';
