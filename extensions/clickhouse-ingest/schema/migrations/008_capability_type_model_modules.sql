-- model_modules: per-nn.Module capability verdicts, between model_ops (per op) and model_support
-- (per model). Applied ahead of its writers. chk_test_type is already the last constraint, so
-- re-adding it keeps 20-artifacts.sql's order.
ALTER TABLE artifact_results DROP CONSTRAINT IF EXISTS chk_test_type;

ALTER TABLE artifact_results ADD CONSTRAINT chk_test_type CHECK test_type IN ('smoke', 'unit', 'integration', 'regression', 'trunk', 'perf', 'fvt', 'fvt-static', 'fvt-dynamic', 'svt', 'svt-static', 'svt-dynamic', 'model_ops', 'model_modules', 'model_support');
