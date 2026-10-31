-- Append-only records of CI decisions and failures that were only ever console lines: why the
-- /spyre-test poller did or did not start a run, which v2 writes failed, and what the merge queue
-- did with a PR. Nothing writes them yet; the producers land with the schema.

-- One row per poller decision about a /spyre-test comment. No comment bodies, only ids.
CREATE TABLE IF NOT EXISTS poller_decisions
(
    ts          DateTime64(3, 'UTC') DEFAULT now64(3),
    repo        LowCardinality(String),
    pr_number   UInt32,
    comment_id  String DEFAULT '',
    delivery    String DEFAULT '',
    decision    LowCardinality(String),
    reason      String DEFAULT '',
    build_url   String DEFAULT '',
    props       Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_decision CHECK decision IN ('picked', 'skipped', 'collapsed', 'superseded', 'ignored')
)
ENGINE = MergeTree()
ORDER BY (repo, pr_number, ts);

-- One row per failed v2 write, so a lost result is visible when the run's own tables are not.
-- detail is truncated and scrubbed of credentials by the writer.
CREATE TABLE IF NOT EXISTS ci_write_failures
(
    ts          DateTime64(3, 'UTC') DEFAULT now64(3),
    run_key     String,
    table_name  LowCardinality(String),
    stage       LowCardinality(String) DEFAULT '',
    detail      String DEFAULT '',
    props       Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
ORDER BY (run_key, ts);

-- One row per merge-queue transition of a PR. reason is GitHub's removal reason when it gives one.
CREATE TABLE IF NOT EXISTS merge_queue_events
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    repo            LowCardinality(String),
    pr_number       UInt32,
    event           LowCardinality(String),
    reason          LowCardinality(String) DEFAULT '',
    queue_position  UInt16 DEFAULT 0,
    head_sha        String DEFAULT '',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_event CHECK event IN ('enqueued', 'dequeued', 'merged', 'removed')
)
ENGINE = MergeTree()
ORDER BY (repo, pr_number, ts);
