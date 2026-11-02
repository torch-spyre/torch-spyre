-- pending / rejected / expired: an upload awaits review before it is ingested.
-- result_uploads' constraints are re-added in their DDL order, so it stays in step.
ALTER TABLE result_uploads DROP CONSTRAINT IF EXISTS chk_state;

ALTER TABLE result_uploads DROP CONSTRAINT IF EXISTS chk_linked;

ALTER TABLE result_uploads DROP CONSTRAINT IF EXISTS chk_format;

ALTER TABLE result_uploads DROP CONSTRAINT IF EXISTS chk_result_kind;

ALTER TABLE result_uploads DROP CONSTRAINT IF EXISTS chk_test_type;

ALTER TABLE result_uploads ADD CONSTRAINT chk_state CHECK state IN ('pending', 'rejected', 'expired', 'published', 'failed', 'retracted');

ALTER TABLE result_uploads ADD CONSTRAINT chk_linked CHECK artifact_id != toUUID('00000000-0000-0000-0000-000000000000') AND run_id != toUUID('00000000-0000-0000-0000-000000000000');

ALTER TABLE result_uploads ADD CONSTRAINT chk_format CHECK format IN ('junit', 'vllm', 'sendnn', 'benchmarks');

ALTER TABLE result_uploads ADD CONSTRAINT chk_result_kind CHECK result_kind IN ('functional', 'performance', 'capability');

ALTER TABLE result_uploads ADD CONSTRAINT chk_test_type CHECK test_type IN ('smoke', 'unit', 'integration', 'regression', 'trunk', 'perf', 'quality', 'fvt', 'fvt-static', 'fvt-dynamic', 'svt', 'svt-static', 'svt-dynamic', 'model_ops', 'model_modules', 'model_support', 'model_serving');
