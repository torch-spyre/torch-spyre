-- Environments for Bench: the registered request (what a case needs), the measured host state it ran on, and the
-- node history and qualification outcomes that decide whether a run is reference grade.
-- env_id is a uuid5 of the canonical fingerprint (no node names or serials); the exact hardware is run_placements.
CREATE TABLE IF NOT EXISTS environments
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    name            String,
    spec_digest     String,                        -- an edit to the authored spec is a new digest under the same name
    labels          Map(LowCardinality(String), String),
    devices         String DEFAULT '',             -- the resolved device requests and constraints, JSON
    host            String DEFAULT '',
    policy          Map(LowCardinality(String), String),
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_name   CHECK name != '',
    CONSTRAINT chk_digest CHECK spec_digest != ''
)
ENGINE = MergeTree()
ORDER BY (name, spec_digest);

CREATE TABLE IF NOT EXISTS node_classes
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    name            String,
    spec_digest     String,
    host_profile    Map(LowCardinality(String), String),
    qualification   Map(LowCardinality(String), String),   -- expiry_days, cv_limit_pct, probe
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_name   CHECK name != '',
    CONSTRAINT chk_digest CHECK spec_digest != ''
)
ENGINE = MergeTree()
ORDER BY (name, spec_digest);

CREATE TABLE IF NOT EXISTS environment_fingerprints
(
    ts                  DateTime64(3, 'UTC') DEFAULT now64(3),
    env_id              UUID,
    fingerprint_version UInt16,                    -- versions the hash definition, so a new fingerprint key never silently re-ids rows
    labels              Map(LowCardinality(String), String),
    fingerprint         Map(LowCardinality(String), String),
    members             Array(UUID),               -- member node fingerprints of a multi-node environment
    devices             String DEFAULT '',
    props               Map(LowCardinality(String), String),
    audit_uuid          UUID DEFAULT generateUUIDv7(),
    audit_timestamp     DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_env CHECK env_id != toUUID('00000000-0000-0000-0000-000000000000')
)
ENGINE = MergeTree()
ORDER BY (env_id, fingerprint_version);

-- One row per observation "node had this env_id from valid_from"; v_environment_nodes derives the intervals.
CREATE TABLE IF NOT EXISTS environment_nodes
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    node            String,
    env_id          UUID,
    node_class      LowCardinality(String),
    valid_from      DateTime64(3, 'UTC'),
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_node CHECK node != ''
)
ENGINE = MergeTree()
ORDER BY (node, valid_from);

-- A qualification binds to an env_id: any fingerprint change needs a new one.
CREATE TABLE IF NOT EXISTS environment_qualifications
(
    ts               DateTime64(3, 'UTC') DEFAULT now64(3),
    node             String,
    env_id           UUID,
    probe            LowCardinality(String),
    outcome          LowCardinality(String),
    cv_pct           Float64,                      -- coefficient of variation of the canary repeats
    limit_pct        Float64,
    repeats          UInt16,
    checks           Map(LowCardinality(String), String),
    evidence_run_id  UUID,                         -- the canary is an ordinary benchmark run
    qualified_at     DateTime64(3, 'UTC'),
    expires_at       DateTime64(3, 'UTC'),
    props            Map(LowCardinality(String), String),
    audit_uuid       UUID DEFAULT generateUUIDv7(),
    audit_timestamp  DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_node    CHECK node != '',
    CONSTRAINT chk_outcome CHECK outcome IN ('pass', 'fail')
)
ENGINE = MergeTree()
ORDER BY (env_id, node, qualified_at);

-- Latest row per key for the three dimension tables above (idempotent re-registration appends duplicates).
-- Newest wins by (ts, audit_timestamp, audit_uuid); within one millisecond the order is deterministic but arbitrary.
CREATE VIEW IF NOT EXISTS v_environments AS
SELECT
    e.name                                                          AS name,
    e.spec_digest                                                   AS spec_digest,
    argMax(e.labels, (e.ts, e.audit_timestamp, e.audit_uuid))       AS labels,
    argMax(e.devices, (e.ts, e.audit_timestamp, e.audit_uuid))      AS devices,
    argMax(e.host, (e.ts, e.audit_timestamp, e.audit_uuid))         AS host,
    argMax(e.policy, (e.ts, e.audit_timestamp, e.audit_uuid))       AS policy,
    min(e.ts)                                                       AS registered_at
FROM environments AS e
GROUP BY e.name, e.spec_digest;

CREATE VIEW IF NOT EXISTS v_node_classes AS
SELECT
    c.name                                                          AS name,
    c.spec_digest                                                   AS spec_digest,
    argMax(c.host_profile, (c.ts, c.audit_timestamp, c.audit_uuid)) AS host_profile,
    argMax(c.qualification, (c.ts, c.audit_timestamp, c.audit_uuid)) AS qualification,
    min(c.ts)                                                       AS registered_at
FROM node_classes AS c
GROUP BY c.name, c.spec_digest;

-- One row per env_id, from its highest fingerprint_version (then newest row), so a join never picks arbitrarily.
CREATE VIEW IF NOT EXISTS v_environment_fingerprints AS
SELECT
    f.env_id                                                                              AS env_id,
    max(f.fingerprint_version)                                                            AS fingerprint_version,
    argMax(f.labels, (f.fingerprint_version, f.ts, f.audit_timestamp, f.audit_uuid))      AS labels,
    argMax(f.fingerprint, (f.fingerprint_version, f.ts, f.audit_timestamp, f.audit_uuid)) AS fingerprint,
    argMax(f.members, (f.fingerprint_version, f.ts, f.audit_timestamp, f.audit_uuid))     AS members,
    argMax(f.devices, (f.fingerprint_version, f.ts, f.audit_timestamp, f.audit_uuid))     AS devices
FROM environment_fingerprints AS f
GROUP BY f.env_id;

-- A node's env_id intervals; the open one has valid_to = NULL. Observations are collapsed per (node, valid_from)
-- first (newest wins), so a re-sent heartbeat never makes a zero-length interval. Adjacent intervals with the same
-- env_id are not merged: a consumer that needs "since when on this env_id" takes the min over the run.
CREATE VIEW IF NOT EXISTS v_environment_nodes AS
SELECT
    node,
    env_id,
    node_class,
    valid_from,
    leadInFrame(toNullable(valid_from)) OVER (PARTITION BY node ORDER BY valid_from
        ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING) AS valid_to
FROM
(
    SELECT
        n.node                                                        AS node,
        n.valid_from                                                  AS valid_from,
        argMax(n.env_id, (n.ts, n.audit_timestamp, n.audit_uuid))     AS env_id,
        argMax(n.node_class, (n.ts, n.audit_timestamp, n.audit_uuid)) AS node_class
    FROM environment_nodes AS n
    GROUP BY n.node, n.valid_from
);

-- Latest qualification per (node, env_id) and whether it is still current (newest qualified_at, then insert order).
-- is_current reads now64(3), so it changes between evaluations. It does not check that env_id is the node's current
-- one: join the open v_environment_nodes interval for that.
CREATE VIEW IF NOT EXISTS v_environment_qualified AS
SELECT
    q.node                        AS node,
    q.env_id                      AS env_id,
    argMax(q.outcome, (q.qualified_at, q.audit_timestamp, q.audit_uuid))    AS outcome,
    argMax(q.cv_pct, (q.qualified_at, q.audit_timestamp, q.audit_uuid))     AS cv_pct,
    max(q.qualified_at)           AS qualified_at,
    argMax(q.expires_at, (q.qualified_at, q.audit_timestamp, q.audit_uuid)) AS expires_at,
    argMax(q.outcome, (q.qualified_at, q.audit_timestamp, q.audit_uuid)) = 'pass'
        AND argMax(q.expires_at, (q.qualified_at, q.audit_timestamp, q.audit_uuid)) > now64(3) AS is_current
FROM environment_qualifications AS q
GROUP BY q.node, q.env_id;
