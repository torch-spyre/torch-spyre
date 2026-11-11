-- One row per artifactory-sweep plan or apply attempt, including refused and skipped ones. Append-only:
-- a later attempt is a new row. 'started' is the claim an apply writes before deleting; its final row (applied,
-- partial or failed) follows, and a lone started row means the run died mid-way: treat the digest as possibly
-- partially applied. plan_digest is the sha256 of the plan; a plan row's props carry plan_build.
CREATE TABLE IF NOT EXISTS artifact_sweep_runs
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    run_id          String,                        -- <job>#<build>
    mode            LowCardinality(String),        -- plan | apply
    status          LowCardinality(String),        -- planned | started | applied | partial | refused | failed | skipped
    repos           Array(String),
    days            UInt32 DEFAULT 0,
    keep            UInt32 DEFAULT 0,
    plan_digest     String DEFAULT '',             -- empty only when no plan was produced
    selected_files  UInt64 DEFAULT 0,
    selected_bytes  UInt64 DEFAULT 0,
    deleted_files   UInt64 DEFAULT 0,              -- 0 for a plan
    failed_files    UInt64 DEFAULT 0,
    reason          String DEFAULT '',             -- why refused or skipped
    job_run         String DEFAULT '',             -- BUILD_URL
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_run_id  CHECK trim(run_id) != '',
    CONSTRAINT chk_mode    CHECK mode IN ('plan', 'apply'),
    CONSTRAINT chk_status  CHECK (status = 'planned' AND mode = 'plan') OR (status IN ('started', 'applied', 'partial') AND mode = 'apply') OR status IN ('refused', 'failed', 'skipped'),
    CONSTRAINT chk_digest  CHECK match(plan_digest, '^([0-9a-f]{64})?$'),
    CONSTRAINT chk_digest_needed CHECK status NOT IN ('planned', 'started', 'applied', 'partial') OR plan_digest != '',
    CONSTRAINT chk_days    CHECK days BETWEEN 0 AND 3650,
    CONSTRAINT chk_keep    CHECK keep BETWEEN 0 AND 1000,
    CONSTRAINT chk_plan_counts CHECK mode = 'apply' OR (deleted_files = 0 AND failed_files = 0),
    CONSTRAINT chk_started CHECK status != 'started' OR (deleted_files = 0 AND failed_files = 0),
    CONSTRAINT chk_applied CHECK status != 'applied' OR failed_files = 0,
    CONSTRAINT chk_partial CHECK status != 'partial' OR failed_files > 0,
    CONSTRAINT chk_reason  CHECK status NOT IN ('refused', 'skipped') OR reason != ''
)
ENGINE = MergeTree()
ORDER BY (ts, run_id);
