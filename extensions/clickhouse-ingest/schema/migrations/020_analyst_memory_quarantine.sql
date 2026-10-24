-- A memory can be quarantined (held back pending review) or retracted (withdrawn after confirmation).
-- Each is its table's last constraint, so re-adding it keeps the DDL order.
ALTER TABLE analyst_memory DROP CONSTRAINT IF EXISTS chk_memory_status;

ALTER TABLE analyst_memory ADD CONSTRAINT chk_memory_status CHECK status IN ('proposed', 'confirmed', 'rejected', 'expired', 'quarantined', 'retracted');
