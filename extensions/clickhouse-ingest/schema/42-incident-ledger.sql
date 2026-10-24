-- CI/CD incidents: normalized signals from every alert source, and one incident record that
-- collapses signals sharing a root cause. Infrastructure telemetry, not an artifact fact (no
-- artifact_id). Evidence is a link and a redacted summary; a secret or raw log never lands here.

-- One row per signal from any source (Jenkins job, node monitor, GHA, ArgoCD, credential probe,
-- the dashboard, a person, Ask Spyre). `signature` names the known cause and is empty until one matches.
CREATE TABLE IF NOT EXISTS incident_events
(
    ts           DateTime64(3, 'UTC') DEFAULT now64(3),
    source       LowCardinality(String),   -- jenkins | gha | argocd | grafana | credential | dashboard | ask_spyre | human
    subject      String,                   -- what is affected: job, node, workflow, app, credential id
    severity     LowCardinality(String),   -- info | warning | critical
    signature    LowCardinality(String) DEFAULT '',
    cause_class  LowCardinality(String) DEFAULT 'unknown',
    summary      String DEFAULT '',        -- redacted
    evidence_url String DEFAULT '',
    dedup_key    String DEFAULT '',        -- same key = same underlying condition
    incident_id  String DEFAULT '',        -- set once the router attaches the signal
    job_run      String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_event_severity CHECK severity IN ('info', 'warning', 'critical'),
    CONSTRAINT chk_event_cause CHECK cause_class IN
        ('credentials', 'resources', 'network', 'hardware', 'upstream', 'pipeline', 'data', 'unknown')
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (source, subject, ts)
TTL toDateTime(ts) + INTERVAL 180 DAY;

-- Every state change, note and action on an incident is a new row, so the table is its own
-- timeline and nothing is overwritten; the newest row per incident is the current state.
CREATE TABLE IF NOT EXISTS incident_log
(
    incident_id  String,
    updated_at   DateTime64(3, 'UTC'),
    updated_by   String,                   -- login, bot or agent name
    actor_kind   LowCardinality(String),   -- system | agent | ask_spyre | human
    state        LowCardinality(String),   -- open | acknowledged | mitigated | resolved | dismissed
    severity     LowCardinality(String),
    title        String DEFAULT '',
    cause_class  LowCardinality(String) DEFAULT 'unknown',
    signature    LowCardinality(String) DEFAULT '',
    owner        String DEFAULT '',
    owner_email  String DEFAULT '',
    action       LowCardinality(String) DEFAULT '',  -- what was done, e.g. retrigger, prune, page, note
    note         String DEFAULT '',
    thread_url   String DEFAULT '',        -- the Slack thread or PR comment that mirrors this incident
    evidence_url String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_incident_state CHECK state IN ('open', 'acknowledged', 'mitigated', 'resolved', 'dismissed'),
    CONSTRAINT chk_incident_severity CHECK severity IN ('info', 'warning', 'critical'),
    CONSTRAINT chk_incident_actor CHECK actor_kind IN ('system', 'agent', 'ask_spyre', 'human'),
    CONSTRAINT chk_incident_cause CHECK cause_class IN
        ('credentials', 'resources', 'network', 'hardware', 'upstream', 'pipeline', 'data', 'unknown')
)
ENGINE = MergeTree()
ORDER BY (incident_id, updated_at);
