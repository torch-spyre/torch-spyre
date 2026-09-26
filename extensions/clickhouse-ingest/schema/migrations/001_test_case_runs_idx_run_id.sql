-- For a test_case_runs created before idx_run_id was part of its DDL. ADD INDEX covers only
-- parts written after it, so MATERIALIZE follows; mutations_sync=2 waits for it.
ALTER TABLE test_case_runs
    ADD INDEX IF NOT EXISTS idx_run_id run_id TYPE bloom_filter(0.01) GRANULARITY 1;
ALTER TABLE test_case_runs MATERIALIZE INDEX idx_run_id SETTINGS mutations_sync = 2;
