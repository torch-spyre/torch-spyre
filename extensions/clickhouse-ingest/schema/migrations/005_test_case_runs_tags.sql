-- Run-context tags (tier, arch) and recorded measurements on each execution, plus the fvt/svt/
-- capability test types. Added ahead of the writers that use them, so a writer never meets a
-- table without the column or a CHECK that rejects its test_type.
ALTER TABLE test_case_runs ADD COLUMN IF NOT EXISTS tags Array(LowCardinality(String)) AFTER props;

ALTER TABLE test_case_runs ADD COLUMN IF NOT EXISTS measurements Map(LowCardinality(String), Float64) AFTER tags;

ALTER TABLE artifact_results DROP CONSTRAINT IF EXISTS chk_test_type;

ALTER TABLE artifact_results ADD CONSTRAINT chk_test_type CHECK test_type IN ('smoke', 'unit', 'integration', 'regression', 'trunk', 'perf', 'fvt', 'svt', 'capability');
