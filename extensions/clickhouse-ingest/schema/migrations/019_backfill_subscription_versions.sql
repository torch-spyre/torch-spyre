-- artifact_subscription_versions_mv sees only inserts made after it was created; copy the versions
-- written before it. As in 002, the MV's creation time is the cutoff, so no version lands twice.
INSERT INTO artifact_subscription_versions
SELECT * FROM artifact_subscriptions
WHERE audit_timestamp < (
    SELECT metadata_modification_time FROM system.tables
    WHERE database = currentDatabase() AND name = 'artifact_subscription_versions_mv'
);
