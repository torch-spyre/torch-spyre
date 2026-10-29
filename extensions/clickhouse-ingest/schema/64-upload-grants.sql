-- Who may upload results to the dashboard without review, and who reviews the rest. Written by
-- dashboard admins; the dashboard decides each upload from the active rows (newest version wins).
-- A grant's scope is the AND of its non-empty arrays; an empty array means "any" for that field.
CREATE TABLE IF NOT EXISTS upload_grants
(
    grant_id        String,                  -- a readable slug, e.g. alice-perf-s390x
    updated_at      DateTime64(3, 'UTC') DEFAULT now64(3),
    grantee         String,                  -- the OpenShift user name, exact case
    -- auto: the grantee's uploads in scope publish at once; reviewer: may approve others' uploads in scope.
    role            LowCardinality(String),
    active          Bool DEFAULT true,
    formats         Array(LowCardinality(String)),   -- result_uploads.format
    arches          Array(LowCardinality(String)),
    components      Array(String),
    test_types      Array(LowCardinality(String)),
    tag_families    Array(String),
    tags            Array(String),
    artifact_ids    Array(UUID),
    models          Array(String),
    updated_by      String DEFAULT '',       -- the login that wrote this version
    notes           String DEFAULT '',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_role    CHECK role IN ('auto', 'reviewer'),
    CONSTRAINT chk_grantee CHECK grantee != ''
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY grant_id;

-- Every version ever written: upload_grants merges older ones away.
CREATE TABLE IF NOT EXISTS upload_grant_versions
(
    grant_id        String,
    updated_at      DateTime64(3, 'UTC'),
    grantee         String,
    role            LowCardinality(String),
    active          Bool,
    formats         Array(LowCardinality(String)),
    arches          Array(LowCardinality(String)),
    components      Array(String),
    test_types      Array(LowCardinality(String)),
    tag_families    Array(String),
    tags            Array(String),
    artifact_ids    Array(UUID),
    models          Array(String),
    updated_by      String,
    notes           String,
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID,
    audit_timestamp DateTime64(3)
)
ENGINE = MergeTree()
ORDER BY (grant_id, updated_at);

-- SELECT *, not a column list: the applier creates MVs before migrations add columns to the source.
CREATE MATERIALIZED VIEW IF NOT EXISTS upload_grant_versions_mv
TO upload_grant_versions AS
SELECT * FROM upload_grants;
