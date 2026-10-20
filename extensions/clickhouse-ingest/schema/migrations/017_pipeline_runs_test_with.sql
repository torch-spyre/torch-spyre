-- The Test-With companions a run honored, as a JSON array, so the dashboard can show what a
-- run was tested together with. Added ahead of the writer so it never meets a table without it.
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS test_with String DEFAULT '' AFTER sha;
