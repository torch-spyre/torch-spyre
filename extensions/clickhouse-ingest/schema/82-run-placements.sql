-- Exact hardware of one execution, per repeat. Platform-level: functional runs record it too, so a faulty node or card
-- can be traced to every kind of result. placement_id is never part of a series key.
CREATE TABLE IF NOT EXISTS run_placements
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    run_id          UUID,
    repeat          UInt16 DEFAULT 0,
    started_at      Nullable(DateTime64(3, 'UTC')),   -- when the repeat began executing; NULL = unknown (a window then matches it)
    env_id          UUID,
    placement_id    UUID,                          -- uuid5 of the sorted nodes and device identities
    nodes           Array(String),
    devices         Array(Tuple(node String, serial String, pci String, firmware String, link String)),
    links           Array(String),
    clean           Map(LowCardinality(String), String),   -- exclusive, co_tenants, idle load before/after, throttled
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_run CHECK run_id != toUUID('00000000-0000-0000-0000-000000000000')
)
ENGINE = MergeTree()
ORDER BY (run_id, repeat);

-- The newest row per (run_id, repeat): a later resolver row supersedes a probe row, never adds to it. started_at is the
-- newest non-NULL value, so a later row can correct it and a row that omits it does not erase it; NULL if never recorded.
CREATE VIEW IF NOT EXISTS v_run_placements_effective AS
SELECT
    r.run_id                                                               AS run_id,
    r.repeat                                                               AS repeat,
    max(r.ts)                                                              AS ts,
    argMaxIf(r.started_at, (r.ts, r.audit_timestamp, r.audit_uuid), r.started_at IS NOT NULL) AS started_at,
    argMax(r.env_id, (r.ts, r.audit_timestamp, r.audit_uuid))              AS env_id,
    argMax(r.placement_id, (r.ts, r.audit_timestamp, r.audit_uuid))        AS placement_id,
    argMax(r.nodes, (r.ts, r.audit_timestamp, r.audit_uuid))               AS nodes,
    argMax(r.devices, (r.ts, r.audit_timestamp, r.audit_uuid))             AS devices,
    argMax(r.links, (r.ts, r.audit_timestamp, r.audit_uuid))               AS links,
    argMax(r.clean, (r.ts, r.audit_timestamp, r.audit_uuid))               AS clean
FROM run_placements AS r
GROUP BY r.run_id, r.repeat;

-- One row per device used by a repeat: "every run that touched card X".
CREATE VIEW IF NOT EXISTS v_run_devices AS
SELECT
    p.run_id        AS run_id,
    p.repeat        AS repeat,
    p.ts            AS ts,
    p.started_at    AS started_at,
    p.env_id        AS env_id,
    p.placement_id  AS placement_id,
    d.node          AS node,
    d.serial        AS serial,
    d.pci           AS pci,
    d.firmware      AS firmware,
    d.link          AS link
FROM v_run_placements_effective AS p
ARRAY JOIN p.devices AS d;
