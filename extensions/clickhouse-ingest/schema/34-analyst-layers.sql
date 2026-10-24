-- The analyst's persistent layers: role profiles and memory, the orchestrator, agent calls, access decisions,
-- recovery, the optimizer, goals and personal assistants. Infrastructure telemetry, not an artifact fact. Ids,
-- hashes and redacted text only: a secret or raw log never lands here.

-- Per-role memory: private to a role, a team proposal or a shared proposal. Append-only versions; the newest row per
-- memory_id is current. A private write lands directly, anything wider is a proposal until confirmed.
CREATE TABLE IF NOT EXISTS analyst_role_memory
(
    memory_id      String,
    updated_at     DateTime64(3, 'UTC'),
    scope          LowCardinality(String),   -- private | team | shared
    namespace      LowCardinality(String),
    kind           LowCardinality(String),
    subject        String,
    body           String DEFAULT '',   -- redacted
    evidence_ids   Array(String),
    confidence     Float32 DEFAULT 0,
    status         LowCardinality(String),
    proposed_by    String,
    confirmed_by   String DEFAULT '',
    author_role    LowCardinality(String) DEFAULT '',
    author_family  LowCardinality(String) DEFAULT '',
    expires_at     Nullable(DateTime64(3, 'UTC')),

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_role_memory_scope CHECK scope IN ('private', 'team', 'shared'),
    CONSTRAINT chk_role_memory_kind CHECK kind IN ('case_summary', 'lesson', 'signature_candidate', 'prompt_note', 'failure_case'),
    CONSTRAINT chk_role_memory_confidence CHECK confidence >= 0 AND confidence <= 1,
    CONSTRAINT chk_role_memory_status CHECK status IN ('proposed', 'confirmed', 'rejected', 'expired', 'quarantined', 'retracted', 'active')
)
ENGINE = MergeTree()
ORDER BY (scope, namespace, memory_id, updated_at);

-- A role's bounded project space. Append-only: a delete is a newer row with deleted = 1.
CREATE TABLE IF NOT EXISTS analyst_project_files
(
    role        LowCardinality(String),
    project     String,
    name        String,
    kind        LowCardinality(String),   -- case_file | note | artefact
    text        String DEFAULT '',   -- redacted
    updated_at  DateTime64(3, 'UTC'),
    deleted     UInt8 DEFAULT 0,

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_project_file_kind CHECK kind IN ('case_file', 'note', 'artefact')
)
ENGINE = MergeTree()
ORDER BY (role, project, name, updated_at);

-- How a role's answer was judged, by a person or an independent validator, never by the judged role.
CREATE TABLE IF NOT EXISTS analyst_outcomes
(
    outcome_id     String,
    recorded_at    DateTime64(3, 'UTC'),
    role           LowCardinality(String),
    incident_id    String,
    turn_ref       String,
    outcome        LowCardinality(String),   -- accepted | rejected | overturned
    judged_by      String,
    judged_kind    LowCardinality(String),   -- human | validator
    judged_family  LowCardinality(String) DEFAULT '',
    lesson_id      String DEFAULT '',
    note           String DEFAULT '',   -- redacted

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_outcome_outcome CHECK outcome IN ('accepted', 'rejected', 'overturned'),
    CONSTRAINT chk_outcome_judge CHECK judged_kind IN ('human', 'validator')
)
ENGINE = MergeTree()
ORDER BY (role, recorded_at, outcome_id);

-- A proposed change to a role's profile, backed by recorded outcomes. Append-only; the newest row per change_id wins.
CREATE TABLE IF NOT EXISTS analyst_profile_changes
(
    change_id          String,
    created_at         DateTime64(3, 'UTC'),
    proposed_by        String,
    target_role        LowCardinality(String),
    target             LowCardinality(String),
    text               String DEFAULT '',   -- redacted
    evidence_outcomes  Array(String),
    status             LowCardinality(String),
    decided_by         String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
ORDER BY (change_id, created_at);

-- The orchestrator's durable work items; the newest row per item_id is the item.
CREATE TABLE IF NOT EXISTS analyst_work_items
(
    item_id              String,
    updated_at           DateTime64(3, 'UTC'),
    started_at           DateTime64(3, 'UTC'),
    status               LowCardinality(String),
    reason               String DEFAULT '',
    not_before           DateTime64(3, 'UTC'),   -- epoch when unset
    step_count           UInt32 DEFAULT 0,
    escalations          UInt8 DEFAULT 0,
    waits                UInt8 DEFAULT 0,
    actions_proposed     UInt8 DEFAULT 0,
    invalid_streak       UInt8 DEFAULT 0,
    error_streak         UInt8 DEFAULT 0,
    stale_cycles         Int32 DEFAULT 0,
    stall_escalated      Bool DEFAULT false,
    last_progress_cycle  UInt32 DEFAULT 0,
    digest               String DEFAULT '',
    observations         String DEFAULT '[]',   -- JSON
    routed               String DEFAULT '[]',   -- JSON
    last_rejection       String DEFAULT '',
    alias_override       String DEFAULT '',
    inc_digest           String DEFAULT '',
    base_digest          String DEFAULT '',
    base_cycle           UInt32 DEFAULT 0,
    mode_note            String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_work_item_status CHECK status IN ('runnable', 'waiting', 'waiting_human', 'parked', 'closed')
)
ENGINE = MergeTree()
ORDER BY (item_id, updated_at)
TTL toDateTime(updated_at) + INTERVAL 180 DAY;

-- One row per orchestrator step, carrying the item as it stands afterwards (the checkpoint a restart recovers from).
CREATE TABLE IF NOT EXISTS analyst_orch_steps
(
    item_id        String,
    step_no        UInt32,
    ts             DateTime64(3, 'UTC'),
    cycle_no       UInt32 DEFAULT 0,
    kind           LowCardinality(String),
    outcome        LowCardinality(String),
    detail         String DEFAULT '',
    proposal_hash  String DEFAULT '',
    model_alias    LowCardinality(String) DEFAULT '',
    tokens         UInt32 DEFAULT 0,
    item_after     String DEFAULT '{}',   -- JSON

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
ORDER BY (item_id, step_no, ts)
TTL toDateTime(ts) + INTERVAL 180 DAY;

-- Orchestrator events: cycles, stalls, dead-man alerts, role lifecycle.
CREATE TABLE IF NOT EXISTS analyst_orch_events
(
    ts        DateTime64(3, 'UTC'),
    kind      LowCardinality(String),
    cycle_no  UInt32 DEFAULT 0,
    reason    String DEFAULT '',
    detail    String DEFAULT '{}',   -- JSON
    actor     String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
ORDER BY (kind, ts)
TTL toDateTime(ts) + INTERVAL 180 DAY;

-- Created-role lifecycle: proposed, reviewed, drafted, promoted, retired. Append-only; the newest row per name is its state.
CREATE TABLE IF NOT EXISTS analyst_role_lifecycle
(
    name        String,
    updated_at  DateTime64(3, 'UTC'),
    state       LowCardinality(String),
    actor       String,
    action      LowCardinality(String),
    detail      String DEFAULT '',
    spec        String DEFAULT '{}',   -- JSON
    expires_at  DateTime64(3, 'UTC') DEFAULT toDateTime64(0, 3, 'UTC'),   -- epoch when unset

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
ORDER BY (name, updated_at);

-- Every agent-to-agent call, refused or not. Hashes only: never request or answer text.
CREATE TABLE IF NOT EXISTS analyst_agent_calls
(
    ts            DateTime64(3, 'UTC'),
    incident_id   String,
    kind          LowCardinality(String),   -- ask | cross_ref
    caller        LowCardinality(String),
    callee        LowCardinality(String),
    request_type  LowCardinality(String),
    request_hash  String,
    outcome       LowCardinality(String),   -- ok | refused | timeout | failed
    reason        String DEFAULT '',
    depth         UInt8 DEFAULT 0,
    tokens        UInt32 DEFAULT 0,
    latency_ms    UInt32 DEFAULT 0,
    stance        LowCardinality(String) DEFAULT '',
    confidence    Float32 DEFAULT 0,
    answer_hash   String DEFAULT '',
    cost_class    LowCardinality(String) DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_agent_call_kind CHECK kind IN ('ask', 'cross_ref'),
    CONSTRAINT chk_agent_call_outcome CHECK outcome IN ('ok', 'refused', 'timeout', 'failed'),
    CONSTRAINT chk_agent_call_stance CHECK stance IN ('', 'agree', 'disagree', 'unsure', 'answer')
)
ENGINE = MergeTree()
ORDER BY (incident_id, ts)
TTL toDateTime(ts) + INTERVAL 180 DAY;

-- The information-access gate's decision log. Labels and digests only: never the text that was gated.
CREATE TABLE IF NOT EXISTS access_decisions
(
    ts            DateTime64(3, 'UTC'),
    run_id        String,   -- process that wrote it; seq is only unique within one
    seq           UInt32,
    row_kind      LowCardinality(String),   -- decision | complaint
    sender        String,
    on_behalf_of  String DEFAULT '',
    dest_kind     LowCardinality(String),
    recipients    Array(String),
    label         String,
    outcome       LowCardinality(String),
    code          LowCardinality(String),
    digest        String DEFAULT '',
    rule          String DEFAULT '',
    approved_by   String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_access_row_kind CHECK row_kind IN ('decision', 'complaint'),
    CONSTRAINT chk_access_outcome CHECK outcome IN ('allow', 'redact', 'deny', 'route', 'summarise', 'declassify', 'complaint')
)
ENGINE = MergeTree()
ORDER BY (run_id, seq, ts)
TTL toDateTime(ts) + INTERVAL 180 DAY;

-- The recovery agent's audit: an action row per state change (newest per action_id wins), refusals, and signature demotions.
CREATE TABLE IF NOT EXISTS recovery_actions
(
    action_id     String,
    updated_at    DateTime64(3, 'UTC'),
    row_kind      LowCardinality(String),   -- action | refusal | demotion | reinstatement
    mode          LowCardinality(String),   -- dry_run | live | n/a
    actuator      LowCardinality(String),
    host          LowCardinality(String),
    repo          String,
    pr            UInt32 DEFAULT 0,
    head_sha      String,
    run_id        String,
    job           String,
    signature     LowCardinality(String),
    cause_class   LowCardinality(String),
    idem_key      String,
    attempt       UInt16 DEFAULT 0,
    status        LowCardinality(String),
    outcome       LowCardinality(String),
    outcome_ts    Nullable(DateTime64(3, 'UTC')),
    reason        String DEFAULT '',
    evidence_url  String DEFAULT '',
    incident_id   String DEFAULT '',
    detail        String DEFAULT '',   -- redacted

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_recovery_row_kind CHECK row_kind IN ('action', 'refusal', 'demotion', 'reinstatement'),
    CONSTRAINT chk_recovery_mode CHECK mode IN ('dry_run', 'live', 'n/a')
)
ENGINE = MergeTree()
ORDER BY (action_id, updated_at)
TTL toDateTime(updated_at) + INTERVAL 365 DAY;

-- What the optimizer role writes: runs, audit steps, suggestions, PR plans and the outcome of past suggestions.
CREATE TABLE IF NOT EXISTS analyst_optimizer_records
(
    ts           DateTime64(3, 'UTC') DEFAULT now64(3),
    kind         LowCardinality(String),   -- run | audit | suggestion | pr_plan | feedback
    fingerprint  String DEFAULT '',
    doc          String,   -- JSON

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_optimizer_kind CHECK kind IN ('run', 'audit', 'suggestion', 'pr_plan', 'feedback')
)
ENGINE = MergeTree()
ORDER BY (kind, ts)
TTL toDateTime(ts) + INTERVAL 180 DAY;

-- A scored review cycle: one row per key result and one per role. Append-only; a cycle scored twice reads as its newest rows.
CREATE TABLE IF NOT EXISTS goal_scores
(
    cycle          String,
    scored_at      DateTime64(3, 'UTC'),
    row_kind       LowCardinality(String),   -- kr | role
    subject        String,   -- kr id or role
    owner          LowCardinality(String) DEFAULT '',
    value          Nullable(Float64),
    n              UInt32 DEFAULT 0,
    raw            Nullable(Float64),
    score          Nullable(Float64),
    counter_value  Nullable(Float64),
    breached       UInt8 DEFAULT 0,
    coverage       Nullable(Float64),
    flags          Array(String),
    frozen         UInt8 DEFAULT 0,
    krs            Array(String),   -- a role's key results

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_goal_score_kind CHECK row_kind IN ('kr', 'role')
)
ENGINE = MergeTree()
ORDER BY (cycle, row_kind, subject, scored_at);

-- A reviewer model's review of one role for one cycle.
CREATE TABLE IF NOT EXISTS goal_reviews
(
    review_id        String,
    cycle            String,
    role             LowCardinality(String),
    reviewer_role    LowCardinality(String) DEFAULT '',
    reviewer_model   LowCardinality(String) DEFAULT '',
    reviewer_family  LowCardinality(String) DEFAULT '',
    created_at       DateTime64(3, 'UTC'),
    rating           LowCardinality(String),
    score            Nullable(Float64),
    strengths        Array(String),
    gaps             Array(String),
    status           LowCardinality(String),
    flags            Array(String),
    tokens           UInt32 DEFAULT 0,

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_goal_review_status CHECK status IN ('proposed', 'awaiting_reviewer', 'budget', 'rejected_schema')
)
ENGINE = MergeTree()
ORDER BY (cycle, role, created_at);

-- Proposals out of a review. Append-only; a decision is a newer row, and the newest row per proposal_id is current.
CREATE TABLE IF NOT EXISTS goal_proposals
(
    proposal_id     String,
    created_at      DateTime64(3, 'UTC'),
    cycle           String,
    proposed_by     String,
    proposer_kind   LowCardinality(String),
    target_role     LowCardinality(String),
    type            LowCardinality(String),
    target          String,
    text            String DEFAULT '',   -- redacted
    review_id       String DEFAULT '',
    status          LowCardinality(String),
    decided_by      String DEFAULT '',
    requires_human  UInt8 DEFAULT 1,
    detail          Array(String),

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_goal_proposal_status CHECK status IN ('proposed', 'approved', 'rejected', 'applied')
)
ENGINE = MergeTree()
ORDER BY (proposal_id, created_at);

-- A person's comment on a review.
CREATE TABLE IF NOT EXISTS goal_comments
(
    comment_id  String,
    created_at  DateTime64(3, 'UTC'),
    review_id   String,
    author      String,
    verdict     LowCardinality(String),
    text        String DEFAULT '',   -- redacted

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
ORDER BY (review_id, created_at);

-- Who did what in the goals layer.
CREATE TABLE IF NOT EXISTS goal_audit
(
    ts      DateTime64(3, 'UTC'),
    actor   String,
    action  LowCardinality(String),
    ref     String,
    detail  String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
ORDER BY (ts, ref);

-- A person's opted-in assistant. Wiped on request.
CREATE TABLE IF NOT EXISTS assistant_profiles
(
    person           String,
    display          String,
    state            LowCardinality(String),   -- active | paused
    card             String DEFAULT '{}',   -- JSON
    live_adapters    Array(String),
    backup_person    String DEFAULT '',
    allowed_targets  Array(String),
    created_at       DateTime64(3, 'UTC'),
    updated_at       DateTime64(3, 'UTC'),

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_assistant_state CHECK state IN ('active', 'paused')
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY person;

-- What an assistant knows about its person. Readable only by that person and their assistant.
CREATE TABLE IF NOT EXISTS assistant_private_memory
(
    person         String,
    preferences    String DEFAULT '{}',   -- JSON
    working_hours  String DEFAULT '{}',   -- JSON
    focus          String DEFAULT '',
    owned_areas    Array(String),
    updated_at     DateTime64(3, 'UTC'),

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY person;

-- Inbound items for a person's assistant; the item and its triage status in one row.
CREATE TABLE IF NOT EXISTS assistant_inbox
(
    item_id       String,
    recipient     String,
    source        LowCardinality(String),
    sender        String,
    subject       String DEFAULT '',
    body          String DEFAULT '',   -- redacted
    links         Array(String),
    urgency       LowCardinality(String),
    incident_ids  Array(String),
    respond_by    Nullable(DateTime64(3, 'UTC')),
    thread        String DEFAULT '',
    received_at   DateTime64(3, 'UTC'),
    status        LowCardinality(String),
    updated_at    DateTime64(3, 'UTC'),

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_assistant_item_status CHECK status IN ('new', 'needs_reply', 'ignored', 'deferred', 'delegated', 'escalated')
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (recipient, item_id)
TTL toDateTime(received_at) + INTERVAL 30 DAY;

-- Drafts awaiting a person's approval, and what became of them.
CREATE TABLE IF NOT EXISTS assistant_drafts
(
    draft_id    String,
    person      String,
    item_id     String DEFAULT '',
    kind        LowCardinality(String),
    adapter     LowCardinality(String),
    target      String,
    thread      String DEFAULT '',
    text        String DEFAULT '',   -- redacted
    status      LowCardinality(String),
    urgent      Bool DEFAULT false,
    payload     String DEFAULT '{}',   -- JSON
    created_at  DateTime64(3, 'UTC'),
    updated_at  DateTime64(3, 'UTC'),

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_assistant_draft_status CHECK status IN ('pending', 'sent', 'rejected', 'blocked', 'dry_run')
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (person, draft_id);

-- Commitments and maintenance windows an assistant reminds its person about.
CREATE TABLE IF NOT EXISTS assistant_commitments
(
    commitment_id  String,
    person         String,
    text           String DEFAULT '',   -- redacted
    due_at         DateTime64(3, 'UTC'),
    kind           LowCardinality(String),   -- commitment | maintenance_window
    lead_minutes   UInt32 DEFAULT 60,
    reminded       Bool DEFAULT false,
    done           Bool DEFAULT false,
    updated_at     DateTime64(3, 'UTC'),

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (person, commitment_id);

-- Messages held for a person's next digest. A drained row is a newer row with drained = 1.
CREATE TABLE IF NOT EXISTS assistant_queue
(
    queue_id    String,
    person      String,
    text        String DEFAULT '',   -- redacted
    queued_at   DateTime64(3, 'UTC'),
    drained     Bool DEFAULT false,
    updated_at  DateTime64(3, 'UTC'),

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (person, queue_id)
TTL toDateTime(queued_at) + INTERVAL 30 DAY;

-- Assistant lifecycle and send audit. Metadata only: never message text or private memory.
CREATE TABLE IF NOT EXISTS assistant_audit
(
    ts      DateTime64(3, 'UTC'),
    actor   String,
    person  String,
    event   LowCardinality(String),
    ref     String DEFAULT '',
    detail  String DEFAULT '{}',   -- JSON

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3)
)
ENGINE = MergeTree()
ORDER BY (person, ts)
TTL toDateTime(ts) + INTERVAL 365 DAY;
