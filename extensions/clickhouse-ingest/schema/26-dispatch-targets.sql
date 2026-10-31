-- The artifact-dispatch job's allow-list (its DISPATCH_TARGETS), published on every run so the
-- subscriptions page offers only targets the job will send to. One snapshot per publish, all rows
-- sharing its published_at; an emptied list is a single sentinel row (target_type = '',
-- target_prefix = ''), so "no targets" differs from "never published". Credential ids only:
-- the secrets stay in the CI system.
CREATE TABLE IF NOT EXISTS dispatch_targets
(
    target_type     LowCardinality(String),  -- jenkins | gha_dispatch | webhook; '' = the sentinel
    target_prefix   String,                  -- as written in DISPATCH_TARGETS; a target matches by prefix
    credential_id   String DEFAULT '',       -- a CI credential id, never a secret; '' = none
    published_at    DateTime64(3, 'UTC'),
    job_run         String DEFAULT ''        -- the dispatch build that published it
)
ENGINE = MergeTree()
ORDER BY (published_at, target_type, target_prefix)
TTL toDateTime(published_at) + INTERVAL 30 DAY;
