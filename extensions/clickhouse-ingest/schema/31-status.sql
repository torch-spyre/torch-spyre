-- Component health samples and the notes people and the agent layer over the computed state.
-- Infrastructure telemetry, not an artifact fact (no artifact_id). Ids, states and redacted text
-- only: a secret or raw response body never lands here.

-- One row per probe of one component from one vantage. Idempotent: sample_id repeats on a retry or
-- on the late upload of a buffered sample, and the newest upload replaces the older row. `ts` is
-- when the probe ran, never when it was uploaded, so a backfill after an outage lands in the right
-- bucket. probe = 'heartbeat' (component '*') is a vantage's liveness, not a component sample.
CREATE TABLE IF NOT EXISTS status_samples
(
    sample_id    String,                  -- vantage|component|probe|bucket start epoch; see status_sample_id()
    ts           DateTime64(3, 'UTC'),
    bucket       DateTime('UTC') MATERIALIZED toStartOfFiveMinutes(ts),
    vantage      LowCardinality(String),  -- gha-hosted | build-vm | jenkins-worker | cluster-pod | laptop
    component    LowCardinality(String),
    probe        LowCardinality(String),  -- http_ping | tls | feed | tick | heartbeat
    state        LowCardinality(String),  -- ok | degraded (answered, but slow) | failing | unknown
    http_status  UInt16 DEFAULT 0,
    latency_ms   UInt32 DEFAULT 0,
    error_class  LowCardinality(String) DEFAULT '',  -- timeout | tls | reset | dns | http | other
    detail       String DEFAULT '',       -- redacted, short
    ingested_at  DateTime64(3, 'UTC') DEFAULT now64(3),
    upload_path  LowCardinality(String) DEFAULT 'primary',  -- primary | buffered

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_sample_state CHECK state IN ('ok', 'degraded', 'failing', 'unknown'),
    CONSTRAINT chk_sample_upload_path CHECK upload_path IN ('primary', 'buffered'),
    CONSTRAINT chk_sample_detail CHECK length(detail) <= 500
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(ts)
ORDER BY (component, vantage, bucket, sample_id)
TTL toDateTime(ts) + INTERVAL 400 DAY;

-- What a person or the agent says about a component, layered over the computed state and never
-- changing it. Append-only: an edit is a new row with `supersedes` naming the row it replaces, and
-- a row with empty text withdraws the one it supersedes. component '' means the whole system.
CREATE TABLE IF NOT EXISTS status_annotations
(
    annotation_id String,
    ts            DateTime64(3, 'UTC'),
    component     LowCardinality(String),
    kind          LowCardinality(String),   -- human | agent | system
    author        String,                   -- login, bot or agent name
    text          String,                   -- redacted; shown as plain text, never as markup
    expires_at    Nullable(DateTime64(3, 'UTC')),
    supersedes    String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_annotation_kind CHECK kind IN ('human', 'agent', 'system'),
    CONSTRAINT chk_annotation_text CHECK length(text) <= 2000,
    CONSTRAINT chk_annotation_supersedes CHECK supersedes != annotation_id
)
ENGINE = MergeTree()
ORDER BY (annotation_id, ts);
