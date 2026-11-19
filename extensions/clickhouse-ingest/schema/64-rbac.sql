-- Role-based access for the dashboard and its tooling: a role is a set of '<resource>:<action>'
-- permissions, and a binding gives a role to a user or an OpenShift group, optionally limited to a
-- scope. Written by dashboard admins; every feature asks the dashboard's one authorize() check, so
-- uploads, job launches and the rest cannot disagree on who may do what.

-- One row per role version; the newest updated_at wins, so an edit is a new INSERT.
CREATE TABLE IF NOT EXISTS rbac_roles
(
    role_id         String,                  -- a readable slug, e.g. uploader
    updated_at      DateTime64(3, 'UTC') DEFAULT now64(3),
    active          Bool DEFAULT true,
    -- '<resource>:<action>', '<resource>:*' or '*'. Matching is case-insensitive.
    permissions     Array(String),
    description     String DEFAULT '',
    updated_by      String DEFAULT '',       -- the login that wrote this version
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_role_id     CHECK role_id != '',
    CONSTRAINT chk_permissions CHECK arrayAll(p -> p = '*' OR position(p, ':') > 1, permissions)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY role_id;

-- Every role version ever written: rbac_roles merges older ones away.
CREATE TABLE IF NOT EXISTS rbac_role_versions
(
    role_id         String,
    updated_at      DateTime64(3, 'UTC'),
    active          Bool,
    permissions     Array(String),
    description     String,
    updated_by      String,
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID,
    audit_timestamp DateTime64(3)
)
ENGINE = MergeTree()
ORDER BY (role_id, updated_at);

-- SELECT *, not a column list: the applier creates MVs before migrations add columns to the source.
CREATE MATERIALIZED VIEW IF NOT EXISTS rbac_role_versions_mv
TO rbac_role_versions AS
SELECT * FROM rbac_roles;

-- One row per binding version. A group subject is an OpenShift group the login token carries, so
-- membership is managed there, not here. Subjects and scope values match case-insensitively.
CREATE TABLE IF NOT EXISTS rbac_bindings
(
    binding_id      String,
    updated_at      DateTime64(3, 'UTC') DEFAULT now64(3),
    active          Bool DEFAULT true,
    subject_type    LowCardinality(String),
    subject         String,
    role_id         String,
    -- dimension -> allowed values, ANDed across dimensions; a missing or empty dimension means any.
    -- The dimensions are the context the checking feature supplies, e.g. uploads: format, arch,
    -- component, test_type, tag_family, tag, artifact_id, model; jobs: job, component, arch.
    scope           Map(LowCardinality(String), Array(String)),
    -- 1970-01-01 = never.
    expires_at      DateTime64(3, 'UTC') DEFAULT toDateTime64(0, 3, 'UTC'),
    updated_by      String DEFAULT '',
    notes           String DEFAULT '',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_subject_type CHECK subject_type IN ('user', 'group'),
    CONSTRAINT chk_subject      CHECK subject != '',
    CONSTRAINT chk_binding_role CHECK role_id != ''
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY binding_id;

CREATE TABLE IF NOT EXISTS rbac_binding_versions
(
    binding_id      String,
    updated_at      DateTime64(3, 'UTC'),
    active          Bool,
    subject_type    LowCardinality(String),
    subject         String,
    role_id         String,
    scope           Map(LowCardinality(String), Array(String)),
    expires_at      DateTime64(3, 'UTC'),
    updated_by      String,
    notes           String,
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID,
    audit_timestamp DateTime64(3)
)
ENGINE = MergeTree()
ORDER BY (binding_id, updated_at);

CREATE MATERIALIZED VIEW IF NOT EXISTS rbac_binding_versions_mv
TO rbac_binding_versions AS
SELECT * FROM rbac_bindings;
