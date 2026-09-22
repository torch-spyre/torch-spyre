-- Hardware-failure diagnostics: one row per (run, suite, attempt).
--
-- The parse/ingest logic lives in spyre_clickhouse_ingest.hw_parse / .hw_diagnostics and is
-- shared by torch-spyre, hf-adapters and spyre-inference. This file is the definition those
-- writers assume -- the live v1 table grew by ALTER ... ADD COLUMN from each ingest instead,
-- which is how it reached 45 columns with an empty sorting key and no checked-in shape.
--
-- ONE run id. run_id is the uuid5 over (source, external_run_id, arch, test_type), derived by
-- the same two-case rule as every other writer: the THREADED uuid when the orchestrator
-- supplied one, else the coordinate hash. v1 also stored the raw producer coordinate (a GHA run
-- id, a Jenkins build key) in a String column of the same name; that is a hash INPUT, recoverable
-- from nothing else only because nothing else recorded it -- so it lives in props now, and the
-- column name means one thing.
--
-- DROPPED as run-level duplication, measured over 841,583 prod rows / 6,090 run_ids -- 0 run_ids
-- carried two values of any of them:
--   workflow    -- it IS the test_type (its 7 values are the tiers: regression, trunk, perf,
--                  integration...), already a run_id hash input under another name
--   run_link    -- {server}/{repo}/actions/runs/{run_id}, pure derivation; blank on 209,519 rows
--   branch      -- the triggering run's head
--   commit_sha  -- likewise
-- The last two are real facts with no per-row variation, so they belong on the run, not here.
--
-- attempt stays in the key because retries are the point: a flaky card shows up as attempt 2+ of
-- the same (run, suite), and retry_trigger/pod_level_retry only make sense read alongside the
-- attempt they belong to.
--
-- No audit_uuid/audit_timestamp: the live v1 table carries both, but no writer sets them, so they
-- are unwritten defaults rather than data. Same call as jenkins_agents.
--
-- component leads, as in every other table here, and the join to the artifact side is the same:
--   run_id -> artifact_results.run_id -> .artifact_id -> artifacts -> artifact_tags
-- artifact_id is nil-UUID on an un-updated writer, which reads as "not linked" rather than
-- mis-linked -- a nil UUID joins nothing, whereas a defaulted hash would join everything.
CREATE TABLE IF NOT EXISTS hw_failure_diagnostics
(
    `run_id` UUID,
    `artifact_id` UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),
    `component` LowCardinality(String) DEFAULT '',
    `arch` LowCardinality(String) DEFAULT '',
    `suite_name` String,
    `attempt` UInt8,
    `total_attempts` UInt8,
    -- True when the row came from a pod-level-retry job (a fresh-pod re-run), not the original.
    `pod_level_retry` Bool DEFAULT false,
    `ingested_at` DateTime64(6, 'UTC'),
    `outcome` LowCardinality(String),
    `exit_code` Nullable(Int32),
    `failure_reason` LowCardinality(String),
    `failure_phase` LowCardinality(String),
    `retry_trigger` String,
    `failure_reason_detail` String DEFAULT '{}',
    `ras_code` LowCardinality(String),
    `ras_name` String,
    `ras_description` String,
    `ras_action` LowCardinality(String),
    `ras_category` LowCardinality(String),
    `ras_severity` LowCardinality(String),
    `ras_message` String,
    `ras_events_json` String DEFAULT '[]',
    `node_name` LowCardinality(String),
    `pci_device` LowCardinality(String),
    `aiu_world_rank0` LowCardinality(String),
    `card_serial` String,
    `chip_ecid_raw` String,
    `chip_wafer_id` LowCardinality(String),
    `chip_mfg_x` String,
    `chip_mfg_y` String,
    `chip_chipy` String,
    `chip_chipx` String,
    `first_error_ts` Nullable(DateTime64(6, 'UTC')),
    `attempt_start_ts` Nullable(DateTime64(6, 'UTC')),
    `tests_collected` UInt32,
    `tests_passed` UInt32,
    `tests_failed` UInt32,
    `tests_error` UInt32,
    `stall_max_secs` UInt32,
    -- external_run_id (the raw producer coordinate run_id is hashed from) and run_url. Both are
    -- run-level, but unlike branch/commit_sha they are the INPUTS to this row's own identity, so
    -- they stay recoverable here rather than needing the run to be resolvable first.
    `props` Map(LowCardinality(String), String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ingested_at)
ORDER BY (component, run_id, suite_name, attempt)
