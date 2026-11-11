-- The launched build's status on ops_requests: queued at launch, later passed, failed or error.
ALTER TABLE ops_requests ADD COLUMN IF NOT EXISTS status LowCardinality(String) DEFAULT '' AFTER state;

ALTER TABLE ops_requests ADD CONSTRAINT IF NOT EXISTS chk_status CHECK status IN ('', 'queued', 'passed', 'failed', 'error');

-- Stores '' in existing parts first, so a request launched before this migration never reads as queued.
ALTER TABLE ops_requests MATERIALIZE COLUMN status SETTINGS mutations_sync = 2;

ALTER TABLE ops_requests MODIFY COLUMN status LowCardinality(String) DEFAULT if(state = 'launched', 'queued', '');
