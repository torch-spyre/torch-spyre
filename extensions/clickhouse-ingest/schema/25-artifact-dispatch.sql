-- Artifact dispatch: which downstream pipelines a published artifact triggers, and what each
-- trigger did. Subscriptions and requests are written by people (a reviewed INSERT, the
-- dashboard) and by pipelines (vars/v2Dispatch.groovy); only the spyre-frameworks
-- artifact-dispatch job writes artifact_dispatches. Matching lives in v_artifact_subscribers
-- (54-dispatch-views.sql), so the dispatcher and the dashboard cannot disagree on who is triggered.

-- One row per subscription version; the newest updated_at wins, so an edit is a new INSERT.
-- Every match field is '' (or an empty array) for "any". Secrets never live here: credential_id
-- names a Jenkins credential, and the dispatcher binds only the ids its own job configuration allows.
CREATE TABLE IF NOT EXISTS artifact_subscriptions
(
    subscription_id String,                  -- a readable slug, e.g. stf-torchspyre-s390x
    updated_at      DateTime64(3, 'UTC') DEFAULT now64(3),
    -- Gates automatic dispatch only: a request may still run a disabled subscription, which is
    -- how one is tried by hand before it is switched on.
    enabled         Bool DEFAULT false,
    -- artifact.tagged: a tag first points at the artifact (event key: the tag);
    -- artifact.recorded: its first row; artifact.promoted: its first origin='promoted' row;
    -- results.recorded: a verdict landed (event key: '<run_id>/<test_type>'). The tag_* filters
    -- and exclusions apply to artifact.tagged, the result_* ones to results.recorded.
    event_types     Array(LowCardinality(String)) DEFAULT ['artifact.tagged'],

    tag_family      LowCardinality(String) DEFAULT '',
    -- An RE2 pattern matched against the WHOLE tag (it is anchored by the view).
    tag_pattern     String DEFAULT '',
    -- Each key must equal the props value of the tag's first row for the artifact, e.g. to match
    -- only one writer's tags.
    tag_props       Map(String, String),
    exclude_families     Array(String),
    exclude_tag_patterns Array(String),      -- RE2, whole-tag
    result_test_types    Array(String),
    result_states        Array(String),      -- passed | failed | error; 'running' is not a verdict
    component       LowCardinality(String) DEFAULT '',
    -- artifacts.arch as stored: x86_64 | ppc64le | s390x | multi.
    arch            LowCardinality(String) DEFAULT '',
    kind            LowCardinality(String) DEFAULT '',
    -- Needed beside component: hf-adapters publishes both hf-adapters and hf-adapters-devel.
    artifact_name   String DEFAULT '',
    -- A tag first seen before this is never dispatched automatically, so enabling a
    -- subscription does not fire it at every artifact already tagged.
    not_before      DateTime64(3, 'UTC') DEFAULT now64(3),
    -- Conditions on the artifact's latest verdicts (any run arch), as '<test_type>' or
    -- '<test_type>=<state>': every require entry must hold, and any unless_exists entry blocks.
    -- An unmet one is re-evaluated every sweep while the event is within the lookback.
    require         Array(String),
    unless_exists   Array(String),
    -- 0 = no limit. Over a limit the event is deferred to a later sweep, not dropped.
    max_per_hour    UInt32 DEFAULT 0,
    max_per_day     UInt32 DEFAULT 0,
    -- Within this many minutes of an event only the newest one per (component, arch) fires; the
    -- older ones are recorded skipped. Firing therefore waits until the window closes.
    coalesce_minutes UInt32 DEFAULT 0,

    -- jenkins: job path on the dispatcher's controller ('Spyre-Test/testing/Jenkinsfile.torchspyre');
    -- gha_dispatch: '<owner>/<repo>#<event_type>' (repository_dispatch); webhook: an https URL.
    target_type     LowCardinality(String),
    target          String,
    -- Parameter name -> template of {placeholder}s (dispatch.Template.FIELDS) filled from the
    -- artifact and event. A placeholder it cannot fill skips the dispatch.
    params          Map(String, String),
    -- What a gha_dispatch / webhook body carries under "artifact" (dispatch.PAYLOAD_FIELDS);
    -- empty = artifact_id and tag.
    payload_fields  Array(String),
    credential_id   String DEFAULT '',
    owner           String DEFAULT '',
    notes           String DEFAULT '',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_target_type CHECK target_type IN ('jenkins','gha_dispatch','webhook'),
    CONSTRAINT chk_target      CHECK target != '',
    CONSTRAINT chk_event_types CHECK notEmpty(event_types) AND hasAll(
        ['artifact.tagged','artifact.recorded','artifact.promoted','results.recorded'], event_types)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY subscription_id;


-- Human or pipeline-initiated runs: "run this subscription against this artifact". Append-only;
-- a request is done once artifact_dispatches holds the dispatch derived from its request_id.
CREATE TABLE IF NOT EXISTS dispatch_requests
(
    request_id      UUID DEFAULT generateUUIDv4(),
    requested_at    DateTime64(3, 'UTC') DEFAULT now64(3),
    subscription_id String,
    artifact_id     UUID,
    tag             String DEFAULT '',       -- fills {tag}; '' when the run is not about a tag
    requested_by    LowCardinality(String),  -- user | pipeline
    requester       String DEFAULT '',       -- the user's login, or the requesting BUILD_URL
    -- Merged over the rendered template: a request may change a value, e.g. TEST_CADENCE.
    params          Map(String, String),
    reason          String DEFAULT '',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_requested_by CHECK requested_by IN ('user','pipeline')
)
ENGINE = MergeTree()
ORDER BY (requested_at, request_id);


-- One row per dispatch, upserted on every state change. dispatch_id is derived, so a re-run of
-- the dispatcher finds the row it wrote: uuid5('auto|<subscription>|<artifact_id>|<event_type>|
-- <event_key>') for an event -- a re-promotion of the same artifact to the same tag triggers
-- nothing new -- and uuid5('request|<request_id>') for a request.
CREATE TABLE IF NOT EXISTS artifact_dispatches
(
    dispatch_id     UUID,
    updated_at      DateTime64(3, 'UTC') DEFAULT now64(3),
    subscription_id String,
    artifact_id     UUID,
    event_type      LowCardinality(String),  -- '' for a request
    event_key       String,
    tag             String,
    -- When the event happened (a tag's first row for the artifact); the dispatcher's watermark.
    event_ts        DateTime64(3, 'UTC'),
    requested_by    LowCardinality(String),  -- auto | user | pipeline
    requester       String DEFAULT '',
    request_id      UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),
    requested_at    DateTime64(3, 'UTC'),
    -- queued: planned, trigger not yet confirmed; triggered: the target accepted it;
    -- failed: retried with backoff until attempts runs out; skipped: never retried (a template
    -- the artifact cannot fill, a target or credential the dispatcher does not allow, coalesced).
    state           LowCardinality(String),
    -- Snapshots, so a later subscription edit does not rewrite what was sent; '' for a request
    -- naming a subscription that does not exist (recorded skipped).
    target_type     LowCardinality(String),
    target          String,
    params          Map(String, String),
    payload         String DEFAULT '',       -- JSON "artifact" object sent to gha_dispatch / webhook
    target_url      String DEFAULT '',       -- Jenkins job / GHA actions page / webhook URL
    build_url       String DEFAULT '',       -- the run it started, when the target reports one
    error           String DEFAULT '',
    attempts        UInt16 DEFAULT 0,
    next_attempt_at DateTime64(3, 'UTC') DEFAULT now64(3),
    dispatcher_url  String DEFAULT '',       -- the dispatcher build that wrote this version
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_state        CHECK state IN ('queued','triggered','failed','skipped'),
    CONSTRAINT chk_requested_by CHECK requested_by IN ('auto','user','pipeline'),
    CONSTRAINT chk_target_type  CHECK target_type IN ('jenkins','gha_dispatch','webhook','')
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY dispatch_id;
-- Small by construction (one row per subscription x artifact x event), so every read uses FINAL.
