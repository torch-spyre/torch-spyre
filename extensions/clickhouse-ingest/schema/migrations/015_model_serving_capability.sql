-- model_serving: which serving configs (tensor parallel, max_model_len, max_num_seqs) a model is
-- declared for and which a test served. Applied ahead of its writers; each constraint is its
-- table's last, so re-adding it keeps the DDL order.
ALTER TABLE artifact_results DROP CONSTRAINT IF EXISTS chk_test_type;

ALTER TABLE artifact_results ADD CONSTRAINT chk_test_type CHECK test_type IN ('smoke', 'unit', 'integration', 'regression', 'trunk', 'perf', 'fvt', 'fvt-static', 'fvt-dynamic', 'svt', 'svt-static', 'svt-dynamic', 'model_ops', 'model_modules', 'model_support', 'model_serving');

ALTER TABLE result_uploads DROP CONSTRAINT IF EXISTS chk_test_type;

ALTER TABLE result_uploads ADD CONSTRAINT chk_test_type CHECK test_type IN ('smoke', 'unit', 'integration', 'regression', 'trunk', 'perf', 'fvt', 'fvt-static', 'fvt-dynamic', 'svt', 'svt-static', 'svt-dynamic', 'model_ops', 'model_modules', 'model_support', 'model_serving');
