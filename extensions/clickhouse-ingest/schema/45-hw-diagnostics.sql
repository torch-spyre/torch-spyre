-- Hardware-failure diagnostics: one row per (run, suite, attempt).
--
-- Parse/ingest logic lives in spyre_clickhouse_ingest.hw_parse / .hw_diagnostics, shared by
-- torch-spyre, hf-adapters and spyre-inference; this file is the definition those writers
-- assume, replacing the live v1 table's ALTER-per-ingest growth to 45 undeclared columns.
--
-- ONE run id: run_id is the uuid5 over (source, external_run_id, arch, test_type) -- the
-- threaded uuid when supplied, else the coordinate hash, same as every other writer. The raw
-- producer coordinate (a hash input) now lives in props, not in a same-named String column.
--
-- workflow/run_link/branch/commit_sha DROPPED as run-level duplication (0 of 6,090 run_ids in
-- 841,583 prod rows carried two values of any): workflow IS test_type; run_link is pure
-- derivation; branch/commit_sha are real facts but belong on the run, not here.
--
-- attempt stays in the key: retries are the point, and retry_trigger/pod_level_retry only make
-- sense read alongside the attempt they belong to.
--
-- No audit_uuid/audit_timestamp: the v1 table carries both but no writer sets them (same call
-- as jenkins_agents).
--
-- component leads, as elsewhere; artifact_id is nil-UUID on an un-updated writer, which reads
-- as "not linked" (a nil UUID joins nothing) rather than mis-linked.
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
    -- external_run_id (this row's identity hash input) and run_url; kept here rather than
    -- requiring the run to be resolved first.
    `props` Map(LowCardinality(String), String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ingested_at)
ORDER BY (component, run_id, suite_name, attempt)
