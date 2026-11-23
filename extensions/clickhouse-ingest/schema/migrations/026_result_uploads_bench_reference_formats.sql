-- bench and reference: the upload formats of a Bench result and of a comparison target (bench-reference
-- bundle). result_uploads' last three constraints are re-added in their DDL order.
ALTER TABLE result_uploads DROP CONSTRAINT IF EXISTS chk_test_type;

ALTER TABLE result_uploads DROP CONSTRAINT IF EXISTS chk_result_kind;

ALTER TABLE result_uploads DROP CONSTRAINT IF EXISTS chk_format;

ALTER TABLE result_uploads ADD CONSTRAINT chk_format CHECK format IN ('junit', 'vllm', 'sendnn', 'benchmarks', 'tables', 'reference', 'bench');

ALTER TABLE result_uploads ADD CONSTRAINT chk_result_kind CHECK result_kind IN ('functional', 'performance', 'capability');

ALTER TABLE result_uploads ADD CONSTRAINT chk_test_type CHECK test_type IN ('smoke', 'unit', 'integration', 'regression', 'trunk', 'perf', 'quality', 'fvt', 'fvt-static', 'fvt-dynamic', 'svt', 'svt-static', 'svt-dynamic', 'model_ops', 'model_modules', 'model_support', 'model_serving');
