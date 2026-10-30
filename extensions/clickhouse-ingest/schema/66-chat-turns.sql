-- Ask Spyre / Insights turn log: one 'turn' event per answered question and 'feedback' events (thumbs) keyed by
-- turn_id, for signed-in and anonymous callers. Append-only like result_uploads; written by the dashboard with the
-- upload writer's login, and only when its V2_CHAT_TURN_LOG_ENABLED flag is on.
-- Privacy: an anonymous caller is a browser-held random id plus a salted hash of ip+user-agent, never the raw IP;
-- tool arguments are kept as a sha256, the answer as a sha256 and its length. Rows are kept forever by default (a decision of
-- 2026-10-30) with no table TTL; a person sets their own retention in chat_turn_retention below.
CREATE TABLE IF NOT EXISTS chat_turns
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    turn_id         UUID,                          -- minted by the dashboard; a feedback event repeats it
    event           LowCardinality(String),        -- turn | feedback
    conversation_id String DEFAULT '',             -- the page's id for one conversation, if it sent one
    surface         LowCardinality(String),        -- chat | insights | report_replay | slack | mcp
    actor           String,                        -- the verified login, or anon:<browser id> for anonymous callers
    actor_kind      LowCardinality(String),        -- user | anonymous | service
    anon_id         String DEFAULT '',             -- the random per-browser id; anonymous callers only
    client_hash     String DEFAULT '',             -- salted hash of ip+user-agent; anonymous callers only
    route           String DEFAULT '',             -- the page the question was asked from
    question        String DEFAULT '',             -- secrets redacted, cut to the dashboard's cap
    answer_sha256   String DEFAULT '',
    answer_chars    UInt32 DEFAULT 0,
    model           String DEFAULT '',
    tier            LowCardinality(String) DEFAULT '',   -- primary | fallback | optional
    tool_names      Array(LowCardinality(String)),
    tools           String DEFAULT '',             -- JSON [{name, ok, error, ms, rows, args_sha256}]
    tool_calls      UInt16 DEFAULT 0,
    tool_errors     UInt16 DEFAULT 0,
    rounds          UInt16 DEFAULT 0,
    budget_hit      UInt8 DEFAULT 0,               -- the tool-round budget ended the loop
    numbers_flagged UInt16 DEFAULT 0,              -- figures in the answer that no lookup returned
    charts_rejected UInt16 DEFAULT 0,
    gaps            UInt16 DEFAULT 0,              -- report_gap cards shown
    abstained       UInt8 DEFAULT 0,
    invalid_output  UInt8 DEFAULT 0,
    latency_ms      UInt32 DEFAULT 0,
    tokens_in       UInt32 DEFAULT 0,
    tokens_out      UInt32 DEFAULT 0,
    error_class     LowCardinality(String) DEFAULT '',
    thumbs          LowCardinality(String) DEFAULT 'none',  -- feedback events only
    reason          String DEFAULT '',                       -- feedback events only
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_event      CHECK event IN ('turn', 'feedback'),
    CONSTRAINT chk_surface    CHECK surface IN ('chat', 'insights', 'report_replay', 'slack', 'mcp'),
    CONSTRAINT chk_actor_kind CHECK actor_kind IN ('user', 'anonymous', 'service'),
    CONSTRAINT chk_thumbs     CHECK thumbs IN ('none', 'up', 'down'),
    CONSTRAINT chk_turn       CHECK turn_id != toUUID('00000000-0000-0000-0000-000000000000'),
    -- An anonymous caller is always anon:<id>, and nobody else can be.
    CONSTRAINT chk_anon_actor CHECK (actor_kind = 'anonymous' AND startsWith(actor, 'anon:'))
                                 OR (actor_kind != 'anonymous' AND NOT startsWith(actor, 'anon:'))
)
ENGINE = MergeTree()
ORDER BY (turn_id, ts);

-- Per-person retention settings and purge requests, append-only like chat_turns. A person sets how long their turns are kept
-- (retention_days, 0 = forever, the default for anyone with no row) or asks for their history to be deleted ('purge': every turn
-- of theirs up to the event's ts). actor is the login or anon:<browser id>. Nothing here deletes anything: the scheduled prune
-- (extensions/clickhouse-ingest/maintenance/prune-chat-turns.sql) is the only deleter. Reads hide purged turns at once.
CREATE TABLE IF NOT EXISTS chat_turn_retention
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    actor           String,
    kind            LowCardinality(String),        -- retention | purge
    retention_days  UInt32 DEFAULT 0,              -- retention events only; 0 = keep forever
    set_by          String,                        -- the verified login, or anon:<id> for a signed-out browser
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_kind      CHECK kind IN ('retention', 'purge'),
    CONSTRAINT chk_days      CHECK retention_days <= 36500,
    CONSTRAINT chk_actor     CHECK actor != '' AND set_by = actor
)
ENGINE = MergeTree()
ORDER BY (actor, ts);

-- One row per actor with a setting or purge: the latest retention (0 = forever) and the time of the latest purge request
-- (the epoch when there is none).
CREATE VIEW IF NOT EXISTS v_chat_turn_retention AS
SELECT
    r.actor                                              AS actor,
    argMaxIf(r.retention_days, r.ts, r.kind = 'retention') AS retention_days,
    maxIf(r.ts, r.kind = 'purge')                        AS purge_before,
    max(r.ts)                                            AS updated_at
FROM chat_turn_retention AS r  -- qualified: the aliases above would shadow the columns
GROUP BY r.actor;

-- One row per turn with its latest thumbs, without turns their owner purged. A feedback event whose turn is gone or never
-- logged is dropped.
CREATE VIEW IF NOT EXISTS v_chat_turns AS
SELECT
    t.turn_id                                              AS turn_id,
    minIf(t.ts, t.event = 'turn')                          AS ts,
    anyIf(t.conversation_id, t.event = 'turn')             AS conversation_id,
    anyIf(t.surface, t.event = 'turn')                     AS surface,
    anyIf(t.actor, t.event = 'turn')                       AS actor,
    anyIf(t.actor_kind, t.event = 'turn')                  AS actor_kind,
    anyIf(t.route, t.event = 'turn')                       AS route,
    anyIf(t.question, t.event = 'turn')                    AS question,
    anyIf(t.answer_chars, t.event = 'turn')                AS answer_chars,
    anyIf(t.model, t.event = 'turn')                       AS model,
    anyIf(t.tier, t.event = 'turn')                        AS tier,
    anyIf(t.tool_names, t.event = 'turn')                  AS tool_names,
    anyIf(t.tools, t.event = 'turn')                       AS tools,
    anyIf(t.tool_calls, t.event = 'turn')                  AS tool_calls,
    anyIf(t.tool_errors, t.event = 'turn')                 AS tool_errors,
    anyIf(t.rounds, t.event = 'turn')                      AS rounds,
    anyIf(t.budget_hit, t.event = 'turn')                  AS budget_hit,
    anyIf(t.numbers_flagged, t.event = 'turn')             AS numbers_flagged,
    anyIf(t.charts_rejected, t.event = 'turn')             AS charts_rejected,
    anyIf(t.gaps, t.event = 'turn')                        AS gaps,
    anyIf(t.abstained, t.event = 'turn')                   AS abstained,
    anyIf(t.invalid_output, t.event = 'turn')              AS invalid_output,
    anyIf(t.latency_ms, t.event = 'turn')                  AS latency_ms,
    anyIf(t.tokens_in, t.event = 'turn')                   AS tokens_in,
    anyIf(t.tokens_out, t.event = 'turn')                  AS tokens_out,
    anyIf(t.error_class, t.event = 'turn')                 AS error_class,
    anyIf(t.anon_id, t.event = 'turn')                     AS anon_id,
    anyIf(t.client_hash, t.event = 'turn')                 AS client_hash,
    if(countIf(t.event = 'feedback') = 0, 'none', argMaxIf(t.thumbs, t.ts, t.event = 'feedback')) AS thumbs,
    argMaxIf(t.reason, t.ts, t.event = 'feedback')         AS thumbs_reason
FROM chat_turns AS t  -- qualified: the aliases above would shadow the columns
LEFT JOIN (SELECT actor, maxIf(ts, kind = 'purge') AS purged FROM chat_turn_retention GROUP BY actor) AS p ON t.actor = p.actor
WHERE t.ts > ifNull(p.purged, toDateTime64(0, 3, 'UTC'))
GROUP BY t.turn_id
HAVING countIf(t.event = 'turn') > 0;

-- Per day, surface and tier: the counts and latency the SLO page can chart.
CREATE VIEW IF NOT EXISTS v_chat_turn_daily AS
SELECT
    toDate(v.ts)                                  AS day,
    v.surface                                     AS surface,
    v.tier                                        AS tier,
    count()                                       AS turns,
    sum(v.tool_calls)                             AS tool_calls,
    sum(v.tool_errors)                            AS tool_errors,
    countIf(v.numbers_flagged > 0)                AS turns_numbers_flagged,
    countIf(v.abstained = 1)                      AS abstained,
    countIf(v.invalid_output = 1)                 AS invalid_output,
    countIf(v.budget_hit = 1)                     AS budget_hit,
    countIf(v.error_class != '')                  AS errors,
    countIf(v.thumbs = 'up')                      AS thumbs_up,
    countIf(v.thumbs = 'down')                    AS thumbs_down,
    quantileExact(0.5)(v.latency_ms)              AS p50_latency_ms,
    quantileExact(0.95)(v.latency_ms)             AS p95_latency_ms,
    sum(v.tokens_in)                              AS tokens_in,
    sum(v.tokens_out)                             AS tokens_out
FROM v_chat_turns AS v
GROUP BY day, surface, tier;
