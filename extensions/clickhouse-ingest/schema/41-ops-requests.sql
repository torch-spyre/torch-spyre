-- The ops trigger's audit log: one row per `run ...` request handled by the spyre-frameworks job
-- <lineage>/ops/trigger, whatever became of it -- validated and launched, rejected by the grammar,
-- denied at the admin approval, or a dry run that launched nothing. Infrastructure telemetry like
-- `jenkins_agents` (no artifact_id), living here because the job writes it through
-- vars/pushToClickhouse.groovy; the DDL is pinned to `audit_row()` by that repo's
-- pipelines/ops/test_ops_trigger.py, so a column added on either side is reviewed beside the other.
--
-- Append-only and best-effort: the job archives ops-result.json whether or not the insert lands, and
-- inserts with input_format_skip_unknown_fields=1, so the two audit columns below are filled by
-- their DEFAULTs rather than by the writer.
CREATE TABLE IF NOT EXISTS ops_requests
(
    requested_at     DateTime64(3, 'UTC'),
    requested_by     String,                             -- REQUESTED_BY as asserted by the caller, else triggered_by
    triggered_by     String DEFAULT '',                  -- the Jenkins user that started the trigger build
    request          String,                             -- the raw request text
    canonical        String DEFAULT '',                  -- the request rebuilt from validated values
    verb             LowCardinality(String),
    admin            Bool DEFAULT false,
    dry_run          Bool DEFAULT false,
    target_kind      LowCardinality(String) DEFAULT '',  -- pr | main | ref | artifact | build; '' = rejected before a target was read
    target           String DEFAULT '',
    preset           LowCardinality(String) DEFAULT '',
    orchestrator_job String DEFAULT '',
    params           Map(String, String),                -- the exact orchestrator params; never secrets
    state            LowCardinality(String),             -- dry_run | launched | rejected | failed | denied
    error            String DEFAULT '',                  -- audit_row() truncates to 500 chars
    approved_by      String DEFAULT '',                  -- the admin-run approver, who is never the requester
    build_url        String DEFAULT '',                  -- the orchestrator build it launched
    job_run          String DEFAULT '',                  -- this trigger build
    channel          LowCardinality(String) DEFAULT '',  -- where the request came from: ui | chat | mcp | api; '' = Jenkins UI
    on_behalf_of     String DEFAULT '',                  -- the person an agent acted for, when requested_by is the agent
    plan_id          String DEFAULT '',                  -- the dashboard preview this launch was confirmed from (ci_requests.plan_id)
    audit_uuid       UUID DEFAULT generateUUIDv7(),
    audit_timestamp  DateTime64(3) DEFAULT now64(3),

    -- ops_trigger.AUDIT_STATES; the writer rejects any other state before it inserts.
    CONSTRAINT chk_state CHECK state IN ('dry_run', 'launched', 'rejected', 'failed', 'denied')
)
ENGINE = MergeTree()
ORDER BY (requested_at, requested_by)
-- Who asked for what stays queryable for a release cycle and change.
TTL toDateTime(requested_at) + INTERVAL 400 DAY;
