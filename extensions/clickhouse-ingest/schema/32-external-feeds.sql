-- What the services we depend on say about themselves: one row per fetch of a status feed, one row
-- per event a feed reports, and the markers that make a maintenance reminder go out once. Feed
-- text is untrusted: titles are sanitised and capped before they get here, and nothing in these
-- tables is executed or can set a component's state except through v_status_current's rules.

-- One row per fetch of one feed from one vantage. A fetch that failed is a row too: a feed that
-- cannot be read is 'unreachable', never 'no incidents'.
CREATE TABLE IF NOT EXISTS external_feed_fetches
(
    fetched_at      DateTime64(3, 'UTC') DEFAULT now64(3),
    vantage         LowCardinality(String),
    source          LowCardinality(String),   -- githubstatus | jfrog-cloud | ibmcloud-rss | ghe-spyre-clusters ...
    feed_url        String,
    outcome         LowCardinality(String),   -- ok | unreachable | http_error | bad_shape | not_modified
    failed_phase    LowCardinality(String) DEFAULT '',  -- dns | connect | tls | http | parse
    http_status     UInt16 DEFAULT 0,
    latency_ms      UInt32 DEFAULT 0,
    bytes           UInt32 DEFAULT 0,
    body_hash       String DEFAULT '',        -- sha256 of the normalised body (items sorted, no build date)
    page_indicator  LowCardinality(String) DEFAULT '',  -- none | minor | major | critical | maintenance | unknown
    job_run         String DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_fetch_outcome CHECK outcome IN ('ok', 'unreachable', 'http_error', 'bad_shape', 'not_modified'),
    CONSTRAINT chk_fetch_indicator CHECK page_indicator IN ('', 'none', 'minor', 'major', 'critical', 'maintenance', 'unknown')
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(fetched_at)
ORDER BY (source, vantage, fetched_at)
TTL toDateTime(fetched_at) + INTERVAL 90 DAY;

-- One row per (event, vantage) observation. event_id is stable across fetches and vantages, so
-- v_external_events_current can collapse three vantages seeing one notice into one event.
CREATE TABLE IF NOT EXISTS external_events
(
    observed_at         DateTime64(3, 'UTC') DEFAULT now64(3),
    event_id            String,               -- source || ':' || the upstream id
    vantage             LowCardinality(String),
    source              LowCardinality(String),
    component           LowCardinality(String),  -- OUR component this affects, '' when none of ours
    upstream_component  String DEFAULT '',    -- the feed's own component text, sanitised and capped
    kind                LowCardinality(String),  -- incident | maintenance | degradation | notice
    impact              LowCardinality(String),  -- none | minor | major | critical | maintenance | unknown
    status              LowCardinality(String),
    starts_at           Nullable(DateTime64(3, 'UTC')),
    ends_at             Nullable(DateTime64(3, 'UTC')),
    title               String,               -- sanitised, at most 200 characters
    url                 String DEFAULT '',
    content_hash        String DEFAULT '',    -- hash of this event's normalised content; a change is an edit
    upstream_updated_at Nullable(DateTime64(3, 'UTC')),

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_ext_kind CHECK kind IN ('incident', 'maintenance', 'degradation', 'notice'),
    CONSTRAINT chk_ext_impact CHECK impact IN ('none', 'minor', 'major', 'critical', 'maintenance', 'unknown'),
    CONSTRAINT chk_ext_status CHECK status IN
        ('scheduled', 'in_progress', 'investigating', 'identified', 'monitoring', 'resolved', 'completed', 'cancelled'),
    CONSTRAINT chk_ext_title CHECK length(title) <= 200,
    CONSTRAINT chk_ext_window CHECK starts_at IS NULL OR ends_at IS NULL OR ends_at >= starts_at
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(observed_at)
ORDER BY (event_id, vantage, observed_at)
TTL toDateTime(observed_at) + INTERVAL 365 DAY;

-- A reminder that went out. Append-only and written after the send succeeded, so a failed send is
-- retried and a sent one is not repeated. reminder_key is event_id|threshold|the window start
-- rounded to 5 minutes, so a moved window is announced again and an unchanged one is not.
CREATE TABLE IF NOT EXISTS external_reminders
(
    sent_at           DateTime64(3, 'UTC') DEFAULT now64(3),
    reminder_key      String,
    event_id          String,
    threshold         LowCardinality(String),   -- 7d | 1d | 1h | cancelled
    window_starts_at  DateTime64(3, 'UTC'),
    recipients        Array(String),
    channel           LowCardinality(String) DEFAULT '',
    message_ref       String DEFAULT '',        -- the Slack ts of the message
    sent_by_vantage   LowCardinality(String) DEFAULT '',

    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_reminder_threshold CHECK threshold IN ('7d', '1d', '1h', 'cancelled')
)
ENGINE = MergeTree()
ORDER BY (reminder_key, sent_at)
TTL toDateTime(sent_at) + INTERVAL 180 DAY;
