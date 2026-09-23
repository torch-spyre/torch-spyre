-- CI audit trails: Jenkins fleet health -- infrastructure telemetry, not an artifact fact
-- (carries no artifact_id), living here only because vars/pushToClickhouse.groovy writes it.
--
-- A v2 pr_check_events table was deliberately not carried forward: it duplicated the artifact
-- layer, added a second status vocabulary, and had no writer; the v1 table still backs /pr-builds.

-- Sampled from the controller API by Spyre/monitoring/collect-agents, which ships a JSONEachRow
-- file (no python3 on the controller to generate this DDL); a test pins these columns to it.
CREATE TABLE IF NOT EXISTS jenkins_agents
(
    ts                  DateTime('UTC') DEFAULT now(),
    node                LowCardinality(String),
    arch                LowCardinality(String),

    offline             UInt8,
    temporarily_offline UInt8,
    idle                UInt8,
    offline_reason      String DEFAULT '',

    executors_total     UInt16,
    executors_busy      UInt16,
    executors_one_off   UInt16 DEFAULT 0,
    executors_pct       Float32,

    mem_free_bytes      UInt64,
    mem_total_bytes     UInt64,
    mem_used_pct        Float32,
    swap_free_bytes     UInt64 DEFAULT 0,
    swap_total_bytes    UInt64 DEFAULT 0,

    disk_path           String DEFAULT '',
    disk_free_bytes     UInt64,
    disk_total_bytes    UInt64,
    disk_used_pct       Float32,
    temp_path           String DEFAULT '',
    temp_free_bytes     UInt64 DEFAULT 0,
    temp_total_bytes    UInt64 DEFAULT 0,
    temp_used_pct       Float32 DEFAULT 0,

    response_ms         UInt32 DEFAULT 0,
    clock_diff_ms       Int32 DEFAULT 0,

    labels              Array(LowCardinality(String)),

    -- Non-empty when the per-agent request failed; the row still lands as the health signal.
    sample_error        String DEFAULT '',

    props               Map(LowCardinality(String), String)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (node, ts)
-- A sample every few minutes per node; 180d matches v1.
TTL ts + INTERVAL 180 DAY;
