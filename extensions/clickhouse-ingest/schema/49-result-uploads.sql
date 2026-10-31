-- Results uploaded by people (the v2 dashboard), one row per event on an upload: the audit log and,
-- since v2 is append-only, the retract marker. Each upload is its own run, written by
-- `results --from-bundle` like any bundle. An upload is linked before it is ingested: one whose
-- artifact does not resolve is refused and leaves no row.
CREATE TABLE IF NOT EXISTS result_uploads
(
    ts             DateTime64(3, 'UTC') DEFAULT now64(3),
    upload_id      UUID,                          -- uuid7; the run key is manual:<uploader>:<upload_id>
    -- 'pending' (stored, awaiting review; not ingested), 'rejected', 'expired' (pending too long),
    -- 'published', 'failed' (ingest error) or 'retracted'
    state          LowCardinality(String),
    actor          String,                        -- who made this event: the uploader, a reviewer, or an admin who retracted
    artifact_id    UUID,
    run_id         UUID,                          -- RunId of the run key, known before the ingest runs
    format         LowCardinality(String),        -- bundle.schema.json `kind`
    result_kind    LowCardinality(String),
    test_type      LowCardinality(String),
    arch           LowCardinality(String),
    filename       String,                        -- what the user sent, e.g. 'bundle.tgz' or 'a.xml, b.xml'
    size_bytes     UInt64,
    content_sha256 String,                        -- offline.digest of the sealed bundle: finds identical uploads
    bundle_url     String,                        -- the sealed bundle in Artifactory (the verdict's bundle_url)
    reason         String DEFAULT '',             -- why it failed or was retracted
    props          Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_state  CHECK state IN ('pending', 'rejected', 'expired', 'published', 'failed', 'retracted'),
    CONSTRAINT chk_linked CHECK artifact_id != toUUID('00000000-0000-0000-0000-000000000000')
                            AND run_id != toUUID('00000000-0000-0000-0000-000000000000'),
    CONSTRAINT chk_format CHECK format IN ('junit', 'vllm', 'sendnn', 'benchmarks'),
    -- Same lists as artifact_results.chk_result_kind / chk_test_type; they move together.
    CONSTRAINT chk_result_kind CHECK result_kind IN
        ('functional','performance','capability'),
    CONSTRAINT chk_test_type   CHECK test_type   IN
        ('smoke','unit','integration','regression','trunk','perf','quality',
         'fvt','fvt-static','fvt-dynamic','svt','svt-static','svt-dynamic',
         'model_ops','model_modules','model_support','model_serving')
)
ENGINE = MergeTree()
ORDER BY (upload_id, ts);

-- One row per upload at its latest state; the uploader is whoever made its first event.
CREATE VIEW IF NOT EXISTS v_result_uploads AS
SELECT
    upload_id,
    argMax(state, ts)          AS state,
    argMax(r.actor, ts)        AS actor,
    argMin(r.actor, ts)        AS uploaded_by,
    argMin(artifact_id, ts)    AS artifact_id,
    argMin(run_id, ts)         AS run_id,
    argMin(format, ts)         AS format,
    argMin(result_kind, ts)    AS result_kind,
    argMin(test_type, ts)      AS test_type,
    argMin(arch, ts)           AS arch,
    argMin(filename, ts)       AS filename,
    argMin(size_bytes, ts)     AS size_bytes,
    argMin(content_sha256, ts) AS content_sha256,
    argMin(bundle_url, ts)     AS bundle_url,
    argMax(reason, ts)         AS reason,
    min(ts)                    AS uploaded_at,
    max(ts)                    AS updated_at
FROM result_uploads AS r  -- qualified: the actor alias would shadow the column
GROUP BY upload_id;

-- artifact_results minus retracted uploads' runs.
CREATE VIEW IF NOT EXISTS v_artifact_results_published AS
SELECT *
FROM artifact_results
WHERE run_id NOT IN (SELECT run_id FROM v_result_uploads WHERE state = 'retracted');
