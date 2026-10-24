-- State of the tiered failure analyst: which model plays which role, its sessions and turns, and the
-- knowledge it proposes. Infrastructure telemetry, not an artifact fact. Ids and redacted text only:
-- a secret or raw log never lands here.

-- Which model alias plays a role for which incidents. Append-only: the newest effective row per
-- role and scope wins, so every change is also its own history. '' in a scope column means any.
CREATE TABLE IF NOT EXISTS analyst_assignments
(
    role               LowCardinality(String),   -- triage | summarise | investigate | validate | curate
    scope_cause_class  LowCardinality(String) DEFAULT '',
    scope_severity     LowCardinality(String) DEFAULT '',
    scope_source       LowCardinality(String) DEFAULT '',
    scope_repo         String DEFAULT '',
    model_alias        LowCardinality(String),
    mode               LowCardinality(String),   -- shadow | propose | live
    weight             UInt8 DEFAULT 100,        -- percent of matching incidents sent to this row's model
    effective_from     DateTime64(3, 'UTC'),
    set_by             String,
    second_approver    String DEFAULT '',        -- required by the runner for validate and act
    reason             String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_assignment_mode CHECK mode IN ('shadow', 'propose', 'live'),
    CONSTRAINT chk_assignment_weight CHECK weight <= 100
)
ENGINE = MergeTree()
ORDER BY (role, scope_cause_class, scope_severity, scope_source, scope_repo, effective_from);

-- One investigation of one incident by one role; a handed-over session points at its parent.
-- Append-only: opening and closing are separate rows, the newest row per session is its state.
CREATE TABLE IF NOT EXISTS analyst_sessions
(
    session_id         String,
    updated_at         DateTime64(3, 'UTC'),
    incident_id        String,
    role               LowCardinality(String),
    state              LowCardinality(String),   -- open | closed | abandoned
    parent_session_id  String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_session_state CHECK state IN ('open', 'closed', 'abandoned')
)
ENGINE = MergeTree()
ORDER BY (session_id, updated_at);

-- One model call. `input_ref` names the evidence and case file the call was assembled from, not
-- their text, so a turn can be replayed against another model without copying logs here.
CREATE TABLE IF NOT EXISTS analyst_turns
(
    session_id      String,
    step            UInt32,
    ts              DateTime64(3, 'UTC'),
    role            LowCardinality(String),
    model_alias     LowCardinality(String),
    served_model    LowCardinality(String) DEFAULT '',
    prompt_version  LowCardinality(String) DEFAULT '',
    input_ref       String DEFAULT '',
    output          String DEFAULT '',           -- the role's output schema as JSON, redacted
    verdict         LowCardinality(String) DEFAULT '',
    confidence      Float32 DEFAULT 0,
    tokens_in       UInt32 DEFAULT 0,
    tokens_out      UInt32 DEFAULT 0,
    cost_usd        Float64 DEFAULT 0,           -- 0 for Spyre-hosted models
    outcome         LowCardinality(String),      -- ok | rejected_schema | error | timeout | budget

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_turn_outcome CHECK outcome IN ('ok', 'rejected_schema', 'error', 'timeout', 'budget'),
    CONSTRAINT chk_turn_confidence CHECK confidence >= 0 AND confidence <= 1
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (session_id, step, ts)
TTL toDateTime(ts) + INTERVAL 180 DAY;

-- What the analyst has learned. A model may propose a row; only a person or an independent
-- validator confirms it. Append-only versions: the newest row per memory_id is the current one.
CREATE TABLE IF NOT EXISTS analyst_memory
(
    memory_id     String,
    updated_at    DateTime64(3, 'UTC'),
    kind          LowCardinality(String),        -- case_summary | lesson | signature_candidate
    subject       String,                        -- signature, job, node or credential id the memory is about
    body          String DEFAULT '',             -- redacted
    evidence_ids  Array(String),
    confidence    Float32 DEFAULT 0,
    status        LowCardinality(String),        -- proposed | confirmed | rejected | expired
    proposed_by   String,
    confirmed_by  String DEFAULT '',
    expires_at    Nullable(DateTime64(3, 'UTC')),  -- NULL until a review date is set

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_memory_kind CHECK kind IN ('case_summary', 'lesson', 'signature_candidate'),
    CONSTRAINT chk_memory_status CHECK status IN ('proposed', 'confirmed', 'rejected', 'expired'),
    CONSTRAINT chk_memory_confidence CHECK confidence >= 0 AND confidence <= 1
)
ENGINE = MergeTree()
ORDER BY (memory_id, updated_at);
