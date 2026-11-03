-- Triage state of a failure signature (an "effect"): who took it, what they decided, why. Append-only,
-- one row per event, like ops_audit_events; the dashboard writes with the upload writer's login and the
-- actor comes from the session. v_effect_status reads the latest event per signature.
CREATE TABLE IF NOT EXISTS effect_status
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    event_id        UUID DEFAULT generateUUIDv7(),
    signature       String,                        -- normalized failure signature, the group's stable key
    scope_key       String DEFAULT '',             -- scope the group was seen in (tag:..., pr:...)
    kind            LowCardinality(String) DEFAULT 'status',
    state           LowCardinality(String) DEFAULT 'untriaged',
    prior_state     LowCardinality(String) DEFAULT '',  -- state before this event, as the writer saw it
    actor           String DEFAULT '',             -- verified login, else anon:<id> (as in chat_turns)
    actor_kind      LowCardinality(String) DEFAULT 'human',
    source          LowCardinality(String) DEFAULT 'human',
    reason          String DEFAULT '',             -- required for invalid and downstream
    owner           String DEFAULT '',
    bug_url         String DEFAULT '',
    incident_id     String DEFAULT '',
    related_signature String DEFAULT '',           -- the upstream group of a downstream, the other group of a reassign
    dedup_key       String DEFAULT '',             -- a repeated submit carries the same key
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_kind       CHECK kind IN ('status', 'merge', 'split', 'note'),
    CONSTRAINT chk_state      CHECK state IN ('untriaged', 'acknowledged', 'invalid', 'downstream', 'reassigned', 'fixed', 'reopened'),
    CONSTRAINT chk_prior      CHECK prior_state IN ('', 'untriaged', 'acknowledged', 'invalid', 'downstream', 'reassigned', 'fixed', 'reopened'),
    CONSTRAINT chk_actor_kind CHECK actor_kind IN ('human', 'anonymous', 'agent', 'service'),
    CONSTRAINT chk_source     CHECK source IN ('human', 'rule'),
    CONSTRAINT chk_signature  CHECK signature != '',
    CONSTRAINT chk_event      CHECK event_id != toUUID('00000000-0000-0000-0000-000000000000')
)
ENGINE = MergeTree()
ORDER BY (signature, ts);

-- One row per signature at its latest state. Owner and bug link carry forward past events that do not
-- set them; taken_by is whoever first set an owner.
CREATE VIEW IF NOT EXISTS v_effect_status AS
SELECT
    e.signature                                             AS signature,
    argMax(e.scope_key, e.ts)                               AS scope_key,
    argMaxIf(e.state, e.ts, e.kind = 'status')              AS state,
    argMaxIf(e.owner, e.ts, e.owner != '')                  AS owner,
    argMaxIf(e.bug_url, e.ts, e.bug_url != '')              AS bug_url,
    argMaxIf(e.incident_id, e.ts, e.incident_id != '')      AS incident_id,
    argMaxIf(e.related_signature, e.ts, e.related_signature != '') AS related_signature,
    argMaxIf(e.reason, e.ts, e.kind = 'status')             AS reason,
    argMax(e.actor, e.ts)                                   AS last_actor,
    argMaxIf(e.ts, e.ts, e.kind = 'status')                 AS state_since,
    argMinIf(e.actor, e.ts, e.owner != '')                  AS taken_by,
    minIf(e.ts, e.state != 'untriaged' OR e.owner != '')    AS first_triaged_at,
    count()                                                 AS events
FROM effect_status AS e  -- qualified: the aliases above would shadow the columns
GROUP BY e.signature;
