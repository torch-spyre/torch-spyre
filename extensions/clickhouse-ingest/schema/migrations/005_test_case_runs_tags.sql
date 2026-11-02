-- Run-context tags (tier, arch) and recorded measurements on each execution, plus the verdict
-- vocabulary: spyre-test-framework's stages as test types, and capability as a result kind whose
-- test types are the analyses (model_ops, model_support). Added ahead of the writers that use
-- them, so a writer never meets a table without the column or a CHECK that rejects its value.
-- Each CHECK is re-added in 20-artifacts.sql's order, since ADD CONSTRAINT appends.
ALTER TABLE test_case_runs ADD COLUMN IF NOT EXISTS tags Array(LowCardinality(String)) AFTER props;

ALTER TABLE test_case_runs ADD COLUMN IF NOT EXISTS measurements Map(LowCardinality(String), Float64) AFTER tags;

ALTER TABLE artifact_results DROP CONSTRAINT IF EXISTS chk_result_kind;

ALTER TABLE artifact_results ADD CONSTRAINT chk_result_kind CHECK result_kind IN ('functional', 'performance', 'capability');

ALTER TABLE artifact_results DROP CONSTRAINT IF EXISTS chk_test_type;

ALTER TABLE artifact_results ADD CONSTRAINT chk_test_type CHECK test_type IN ('smoke', 'unit', 'integration', 'regression', 'trunk', 'perf', 'fvt', 'fvt-static', 'fvt-dynamic', 'svt', 'svt-static', 'svt-dynamic', 'model_ops', 'model_support');
