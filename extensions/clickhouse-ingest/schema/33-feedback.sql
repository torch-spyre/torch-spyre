-- What people (and, weakly, their behaviour) say about the system's alerts, matches, answers and
-- status posts. A signal is evidence, never a label by itself: it can open a review or demote a
-- signature, and promotion is a person's decision. Append-only; an actor's newest signal of a kind
-- on a target is their current one. `detail` is redacted and capped, and is untrusted text.

-- target_type signature_match: target_id is the run_key or incident the signature matched, up means
--   the cause was right and down means it was wrong or not a failure; signature_id is required.
-- target_type component: affected | not_affected are outage reports, one current per actor.
-- signal miss: "this failure had no explanation"; chip: detail names a reason (noise, too_late ...).
-- weight_hint is a voter weight earned from known-answer items; it defaults to 1.
CREATE TABLE IF NOT EXISTS feedback_signals
(
    created_at         DateTime64(3, 'UTC') DEFAULT now64(3),
    target_type        LowCardinality(String),
    target_id          String,
    signal             LowCardinality(String),
    detail             String DEFAULT '',
    source             LowCardinality(String),   -- explicit | implicit | reaction
    actor              String,                   -- login or system name
    weight_hint        Float32 DEFAULT 1,
    signature_id       String DEFAULT '',
    signature_version  String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_feedback_target CHECK target_type IN
        ('signature_match', 'incident', 'status_post', 'component', 'answer', 'failure'),
    CONSTRAINT chk_feedback_signal CHECK signal IN ('up', 'down', 'affected', 'not_affected', 'miss', 'chip'),
    CONSTRAINT chk_feedback_source CHECK source IN ('explicit', 'implicit', 'reaction'),
    CONSTRAINT chk_feedback_detail CHECK length(detail) <= 500,
    CONSTRAINT chk_feedback_weight CHECK weight_hint >= 0,
    CONSTRAINT chk_feedback_signature CHECK target_type != 'signature_match' OR signature_id != ''
)
ENGINE = MergeTree()
ORDER BY (target_type, target_id, actor, created_at);
