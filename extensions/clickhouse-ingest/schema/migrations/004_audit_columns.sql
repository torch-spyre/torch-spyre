-- Adds audit_uuid / audit_timestamp to tables created before they were part of the DDL.
-- An added column's DEFAULT is evaluated at READ time for older parts; MATERIALIZE COLUMN
-- stores it only in parts that lack the column, so rows written after the ADD keep their values.
-- audit_timestamp is added defaulting to the row's own time, then switched to the insert time.

ALTER TABLE test_cases
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT toDateTime64(ts, 3);
ALTER TABLE test_cases MATERIALIZE COLUMN audit_uuid, MATERIALIZE COLUMN audit_timestamp
    SETTINGS mutations_sync = 2;
ALTER TABLE test_cases MODIFY COLUMN audit_timestamp DateTime64(3) DEFAULT now64(3);

ALTER TABLE test_case_runs
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT toDateTime64(ts, 3);
ALTER TABLE test_case_runs MATERIALIZE COLUMN audit_uuid, MATERIALIZE COLUMN audit_timestamp
    SETTINGS mutations_sync = 2;
ALTER TABLE test_case_runs MODIFY COLUMN audit_timestamp DateTime64(3) DEFAULT now64(3);

ALTER TABLE artifacts
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT toDateTime64(ts, 3);
ALTER TABLE artifacts MATERIALIZE COLUMN audit_uuid, MATERIALIZE COLUMN audit_timestamp
    SETTINGS mutations_sync = 2;
ALTER TABLE artifacts MODIFY COLUMN audit_timestamp DateTime64(3) DEFAULT now64(3);

ALTER TABLE artifact_refs
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT toDateTime64(ts, 3);
ALTER TABLE artifact_refs MATERIALIZE COLUMN audit_uuid, MATERIALIZE COLUMN audit_timestamp
    SETTINGS mutations_sync = 2;
ALTER TABLE artifact_refs MODIFY COLUMN audit_timestamp DateTime64(3) DEFAULT now64(3);

ALTER TABLE artifact_tags
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT toDateTime64(ts, 3);
ALTER TABLE artifact_tags MATERIALIZE COLUMN audit_uuid, MATERIALIZE COLUMN audit_timestamp
    SETTINGS mutations_sync = 2;
ALTER TABLE artifact_tags MODIFY COLUMN audit_timestamp DateTime64(3) DEFAULT now64(3);

ALTER TABLE artifact_results
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT toDateTime64(ts, 3);
ALTER TABLE artifact_results MATERIALIZE COLUMN audit_uuid, MATERIALIZE COLUMN audit_timestamp
    SETTINGS mutations_sync = 2;
ALTER TABLE artifact_results MODIFY COLUMN audit_timestamp DateTime64(3) DEFAULT now64(3);

ALTER TABLE benchmarks
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT toDateTime64(ts, 3);
ALTER TABLE benchmarks MATERIALIZE COLUMN audit_uuid, MATERIALIZE COLUMN audit_timestamp
    SETTINGS mutations_sync = 2;
ALTER TABLE benchmarks MODIFY COLUMN audit_timestamp DateTime64(3) DEFAULT now64(3);

ALTER TABLE benchmark_runs
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT toDateTime64(ts, 3);
ALTER TABLE benchmark_runs MATERIALIZE COLUMN audit_uuid, MATERIALIZE COLUMN audit_timestamp
    SETTINGS mutations_sync = 2;
ALTER TABLE benchmark_runs MODIFY COLUMN audit_timestamp DateTime64(3) DEFAULT now64(3);

ALTER TABLE jenkins_agents
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT toDateTime64(ts, 3);
ALTER TABLE jenkins_agents MATERIALIZE COLUMN audit_uuid, MATERIALIZE COLUMN audit_timestamp
    SETTINGS mutations_sync = 2;
ALTER TABLE jenkins_agents MODIFY COLUMN audit_timestamp DateTime64(3) DEFAULT now64(3);

ALTER TABLE hw_failure_diagnostics
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT toDateTime64(ingested_at, 3);
ALTER TABLE hw_failure_diagnostics MATERIALIZE COLUMN audit_uuid, MATERIALIZE COLUMN audit_timestamp
    SETTINGS mutations_sync = 2;
ALTER TABLE hw_failure_diagnostics MODIFY COLUMN audit_timestamp DateTime64(3) DEFAULT now64(3);

ALTER TABLE capabilities
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT toDateTime64(ts, 3);
ALTER TABLE capabilities MATERIALIZE COLUMN audit_uuid, MATERIALIZE COLUMN audit_timestamp
    SETTINGS mutations_sync = 2;
ALTER TABLE capabilities MODIFY COLUMN audit_timestamp DateTime64(3) DEFAULT now64(3);

ALTER TABLE capability_runs
    ADD COLUMN IF NOT EXISTS audit_uuid UUID DEFAULT generateUUIDv7(),
    ADD COLUMN IF NOT EXISTS audit_timestamp DateTime64(3) DEFAULT toDateTime64(ts, 3);
ALTER TABLE capability_runs MATERIALIZE COLUMN audit_uuid, MATERIALIZE COLUMN audit_timestamp
    SETTINGS mutations_sync = 2;
ALTER TABLE capability_runs MODIFY COLUMN audit_timestamp DateTime64(3) DEFAULT now64(3);
