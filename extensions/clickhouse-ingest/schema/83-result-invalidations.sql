-- Results that must not be trusted, one row per event (append-only, like result_uploads): a run, one benchmark or
-- repeat, or everything a node, card or placement touched in a window. v_invalidated_runs is the single read path.
-- scope_kind has no CHECK so `query` can be enabled later in the writer without a migration.
-- Writer contract: every event of an invalidation_id repeats the full scope fields (a CHECK needs them per row), and the
-- first event defines the scope. Order is (ts, audit_timestamp, audit_uuid): same-millisecond events resolve
-- deterministically but arbitrarily, so give events whose order matters distinct ts.
CREATE TABLE IF NOT EXISTS result_invalidations
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    invalidation_id UUID,                          -- uuid7; a later event with the same id changes its state
    scope_kind      LowCardinality(String),        -- run, run_benchmark, node, device, placement; query is reserved
    run_id          Nullable(UUID),
    benchmark_id    Nullable(UUID),
    repeat          Nullable(UInt16),
    node            String DEFAULT '',
    device_serial   String DEFAULT '',
    placement_id    String DEFAULT '',
    window_from     Nullable(DateTime64(3, 'UTC')),
    window_to       Nullable(DateTime64(3, 'UTC')),
    match           Map(LowCardinality(String), String),   -- query scope: attribute equalities, ANDed
    match_expr      String DEFAULT '',
    matched_runs    UInt32 DEFAULT 0,              -- preview count at creation
    state           LowCardinality(String),
    reason_code     LowCardinality(String),
    reason          String DEFAULT '',
    actor           String,
    evidence_url    String DEFAULT '',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_state CHECK state IN ('suspect', 'invalidated', 'restored'),
    CONSTRAINT chk_scope CHECK scope_kind != '',
    CONSTRAINT chk_id    CHECK invalidation_id != toUUID('00000000-0000-0000-0000-000000000000'),
    CONSTRAINT chk_run_scope CHECK scope_kind != 'run' OR (run_id IS NOT NULL AND benchmark_id IS NULL AND repeat IS NULL),
    CONSTRAINT chk_run_benchmark_scope CHECK scope_kind != 'run_benchmark' OR (run_id IS NOT NULL AND benchmark_id IS NOT NULL),
    CONSTRAINT chk_node_scope CHECK scope_kind != 'node' OR node != '',
    CONSTRAINT chk_device_scope CHECK scope_kind != 'device' OR device_serial != '',
    CONSTRAINT chk_placement_scope CHECK scope_kind != 'placement' OR placement_id != ''
)
ENGINE = MergeTree()
ORDER BY (invalidation_id, ts);

-- One row per invalidation at its latest state; the creation event (whole row, NULLs kept) defines the scope.
-- scope_key is the value a placement is matched on: node, device serial, or lower-cased placement uuid.
CREATE VIEW IF NOT EXISTS v_invalidation_state AS
SELECT
    invalidation_id,
    state,
    reason_code,
    reason,
    actor,
    tupleElement(c, 1) AS scope_kind,
    tupleElement(c, 2) AS run_id,
    tupleElement(c, 3) AS benchmark_id,
    tupleElement(c, 4) AS repeat,
    tupleElement(c, 5) AS node,
    tupleElement(c, 6) AS device_serial,
    tupleElement(c, 7) AS placement_id,
    tupleElement(c, 8) AS window_from,
    tupleElement(c, 9) AS window_to,
    multiIf(tupleElement(c, 1) = 'node', tupleElement(c, 5),
            tupleElement(c, 1) = 'device', tupleElement(c, 6),
            tupleElement(c, 1) = 'placement', lower(tupleElement(c, 7)), '') AS scope_key,
    created_at,
    updated_at
FROM
(
    SELECT
        r.invalidation_id                                                                   AS invalidation_id,
        argMax(r.state, (r.ts, r.audit_timestamp, r.audit_uuid))                            AS state,
        argMax(r.reason_code, (r.ts, r.audit_timestamp, r.audit_uuid))                      AS reason_code,
        argMax(r.reason, (r.ts, r.audit_timestamp, r.audit_uuid))                           AS reason,
        argMax(r.actor, (r.ts, r.audit_timestamp, r.audit_uuid))                            AS actor,
        argMin(tuple(r.scope_kind, r.run_id, r.benchmark_id, r.repeat, r.node, r.device_serial, r.placement_id,
                     r.window_from, r.window_to), (r.ts, r.audit_timestamp, r.audit_uuid))  AS c,
        min(r.ts)                                                                           AS created_at,
        max(r.ts)                                                                           AS updated_at
    FROM result_invalidations AS r
    GROUP BY r.invalidation_id
);

-- Runs hit by an active invalidation or suspicion; benchmark_id/repeat are NULL when it covers the whole run.
-- Consumers drop state = 'invalidated' and may show 'suspect'. Retracted uploads are included.
-- The placement branch reads only the repeats named by an active node/device/placement event, then matches them
-- in one pass; windows apply to started_at, inclusive, and a repeat with an unknown started_at matches any window.
CREATE VIEW IF NOT EXISTS v_invalidated_runs AS
SELECT s.invalidation_id AS invalidation_id, s.run_id AS run_id, s.benchmark_id AS benchmark_id, s.repeat AS repeat,
       s.state AS state, s.reason_code AS reason_code
FROM v_invalidation_state AS s
WHERE s.state != 'restored' AND s.scope_kind IN ('run', 'run_benchmark') AND s.run_id IS NOT NULL

UNION ALL

SELECT s.invalidation_id, p.run_id, CAST(NULL AS Nullable(UUID)), toNullable(p.repeat), s.state, s.reason_code
FROM
(
    SELECT e.run_id AS run_id, e.repeat AS repeat, e.started_at AS started_at,
           arrayDistinct(arrayConcat(
               arrayMap(n -> ('node', n), e.nodes),
               arrayMap(d -> ('device', tupleElement(d, 'serial')), e.devices),
               [('placement', lower(toString(e.placement_id)))])) AS keys
    FROM v_run_placements_effective AS e
    WHERE (e.run_id, e.repeat) IN
    (
        SELECT r.run_id, r.repeat
        FROM run_placements AS r
        WHERE hasAny(arrayConcat(
                  arrayMap(n -> ('node', n), r.nodes),
                  arrayMap(d -> ('device', tupleElement(d, 'serial')), r.devices),
                  [('placement', lower(toString(r.placement_id)))]),
              (SELECT groupUniqArray((x.scope_kind, x.scope_key)) FROM v_invalidation_state AS x
               WHERE x.state != 'restored' AND x.scope_kind IN ('node', 'device', 'placement')))
    )
) AS p
ARRAY JOIN p.keys AS k
INNER JOIN
(
    SELECT invalidation_id, state, reason_code, scope_kind, scope_key, window_from, window_to
    FROM v_invalidation_state
    WHERE state != 'restored' AND scope_kind IN ('node', 'device', 'placement')
) AS s ON s.scope_kind = tupleElement(k, 1) AND s.scope_key = tupleElement(k, 2)
WHERE p.started_at IS NULL
   OR ((s.window_from IS NULL OR p.started_at >= s.window_from) AND (s.window_to IS NULL OR p.started_at <= s.window_to))

UNION ALL

SELECT u.upload_id, u.run_id, CAST(NULL AS Nullable(UUID)), CAST(NULL AS Nullable(UInt16)), 'invalidated', 'retracted_upload'
FROM v_result_uploads AS u
WHERE u.state = 'retracted' AND NOT u.is_round;
