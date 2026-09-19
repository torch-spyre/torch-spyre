-- CI audit trails: Jenkins fleet health.
--
-- Infrastructure telemetry, not an artifact fact: it carries no artifact_id, and inventing
-- one would be the synthetic-identity mistake this schema exists to avoid. It lives here
-- because the same writer (vars/pushToClickhouse.groovy) produces it and it must land in
-- that writer's database.
--
-- A v2 pr_check_events table was deliberately not carried forward: every field it held is
-- already answerable from the artifact layer, it added a second status vocabulary every
-- reader had to normalise, and having no artifact_id it could not answer the one question a
-- per-PR page asks. It had no writer and no rows. The v1 table is untouched and still backs
-- the v1 /pr-builds route.

-- Sampled from the controller API by Spyre/monitoring/collect-agents. That collector runs on
-- the controller, which has no python3, so it cannot generate this DDL at run time -- it ships
-- a JSONEachRow file and this file is the schema. A test asserts these columns match the
-- emitted row keys so the two cannot drift.
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

    -- Non-empty when the per-agent request failed. The row still lands: "could not sample
    -- this node" is itself the health signal, and dropping it reads as a healthy gap.
    sample_error        String DEFAULT '',

    props               Map(LowCardinality(String), String)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (node, ts)
-- A sample every few minutes per node, for data whose value is recent. 180d matches v1.
TTL ts + INTERVAL 180 DAY;
