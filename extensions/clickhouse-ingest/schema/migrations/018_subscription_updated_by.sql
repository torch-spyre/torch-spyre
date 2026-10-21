-- The login that wrote each subscription version, recorded by the dashboard's editor.
ALTER TABLE artifact_subscriptions ADD COLUMN IF NOT EXISTS updated_by String DEFAULT '' AFTER owner;
