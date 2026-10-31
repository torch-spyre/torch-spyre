-- model_modules: per-nn.Module capability verdicts, between model_ops (per op) and model_support
-- (per model). undetermined: a test that broke in setup/teardown, so it gave no verdict. Applied
-- ahead of their writers. Each is its table's last constraint, so re-adding it keeps the DDL order.
ALTER TABLE artifact_results DROP CONSTRAINT IF EXISTS chk_test_type;

ALTER TABLE artifact_results ADD CONSTRAINT chk_test_type CHECK test_type IN ('smoke', 'unit', 'integration', 'regression', 'trunk', 'perf', 'fvt', 'fvt-static', 'fvt-dynamic', 'svt', 'svt-static', 'svt-dynamic', 'model_ops', 'model_modules', 'model_support');

ALTER TABLE capability_runs DROP CONSTRAINT IF EXISTS chk_status;

ALTER TABLE capability_runs ADD CONSTRAINT chk_status CHECK status IN ('passed', 'failed', 'not_implemented', 'undetermined');
