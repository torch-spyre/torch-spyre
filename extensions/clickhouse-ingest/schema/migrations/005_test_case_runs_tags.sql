-- Run-context tags (tier, arch) on each execution. Added ahead of the writer that fills it, so
-- a writer inserting the column never meets a table without it.
ALTER TABLE test_case_runs ADD COLUMN IF NOT EXISTS tags Array(LowCardinality(String)) AFTER props;
