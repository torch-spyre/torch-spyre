-- Experiment ledger: what was tried (experiments, arms), what was measured (samples), how it was
-- decided (decisions) and what it showed (findings), for A/B and A/A runs, sweeps, cost-model
-- fits and parity/regression analyses alike. Append-only: 76- reads the latest row per id with
-- argMax, never FINAL. Writers derive every id from content, so a re-ingest adds nothing.
-- The bundle format writers emit is in README.md.
--
-- Not a second benchmark store: a sample taken by a CI run carries that run's run_id and
-- benchmark_id, and traces stay in otel_traces or Artifactory (experiment_links). Samples exist
-- because pod and offline runs have no artifact_results run, and A/B needs arm, pair and order.

-- One row per lifecycle event. A campaign is an experiment whose children name it in parent_id.
CREATE TABLE IF NOT EXISTS experiment_events
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    event_id        UUID DEFAULT generateUUIDv7(),  -- a file ingest derives it from content instead
    experiment_id   UUID,                          -- uuid5 of the experiment name
    parent_id       UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),
    name            String,
    kind            LowCardinality(String),        -- ab | noise | sweep | fit | parity | regression | feasibility | other
    state           LowCardinality(String),        -- proposed | approved | running | paused | done | stopped | failed | abandoned
    subject_level   LowCardinality(String) DEFAULT '',  -- workload | model | op | kernel | collective | host | compile | cost_model
    subject_key     String DEFAULT '',             -- model id, op name, kernel_key, cost-model file, ...
    workload        String DEFAULT '',
    hypothesis      String DEFAULT '',
    spec            String DEFAULT '',             -- JSON, the spec as launched
    spec_sha256     String DEFAULT '',
    budget          Map(LowCardinality(String), Float64),  -- card_min | wall_s | trials | usd
    spent           Map(LowCardinality(String), Float64),
    code_ref        String DEFAULT '',             -- repo@sha of the driver or harness
    reason          String DEFAULT '',
    actor           String,
    actor_kind      LowCardinality(String) DEFAULT 'human',
    on_behalf_of    String DEFAULT '',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_experiment CHECK experiment_id != toUUID('00000000-0000-0000-0000-000000000000'),
    CONSTRAINT chk_name       CHECK name != '',
    CONSTRAINT chk_kind       CHECK kind IN ('ab', 'noise', 'sweep', 'fit', 'parity', 'regression', 'feasibility', 'other'),
    CONSTRAINT chk_state      CHECK state IN ('proposed', 'approved', 'running', 'paused', 'done', 'stopped', 'failed', 'abandoned'),
    CONSTRAINT chk_actor      CHECK actor != '',
    CONSTRAINT chk_actor_kind CHECK actor_kind IN ('human', 'anonymous', 'agent', 'service')
)
ENGINE = MergeTree()
ORDER BY (experiment_id, ts, event_id);

-- One configuration under test. Two arms with one knob_sha make an A/A pair (76- floors).
-- change_class and hazard_class are separate axes: a runtime knob can still hang the card.
CREATE TABLE IF NOT EXISTS experiment_arms
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    event_id        UUID DEFAULT generateUUIDv7(),
    experiment_id   UUID,
    arm             String,
    role            LowCardinality(String),        -- baseline | candidate
    state           LowCardinality(String) DEFAULT 'active',  -- active | dropped | rejected
    hypothesis      String DEFAULT '',
    knobs           Map(String, String),           -- env vars, CLI args, params: the full vector, not only the delta
    knob_sha        String,                        -- sha256 of the sorted knobs
    change_class    LowCardinality(String) DEFAULT '',  -- analysis | runtime_knob | serving_config | codegen_param | model_param | source_patch
    hazard_class    LowCardinality(String) DEFAULT '',  -- safe | may_fail_compile | may_hang
    actor           String,
    actor_kind      LowCardinality(String) DEFAULT 'human',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_arm          CHECK arm != '',
    CONSTRAINT chk_role         CHECK role IN ('baseline', 'candidate'),
    CONSTRAINT chk_state        CHECK state IN ('active', 'dropped', 'rejected'),
    CONSTRAINT chk_knob_sha     CHECK knob_sha != '',
    CONSTRAINT chk_change_class CHECK change_class IN ('', 'analysis', 'runtime_knob', 'serving_config', 'codegen_param', 'model_param', 'source_patch'),
    CONSTRAINT chk_hazard_class CHECK hazard_class IN ('', 'safe', 'may_fail_compile', 'may_hang'),
    CONSTRAINT chk_actor        CHECK actor != '',
    CONSTRAINT chk_actor_kind   CHECK actor_kind IN ('human', 'anonymous', 'agent', 'service')
)
ENGINE = MergeTree()
ORDER BY (experiment_id, arm, ts);

-- One raw value per (sample, metric); a sample that produced nothing is one row with metric ''.
CREATE TABLE IF NOT EXISTS experiment_samples
(
    ts              DateTime64(3, 'UTC'),
    sample_id       UUID,                          -- uuid5 of (experiment, arm, trial, position, warmup, metric)
    experiment_id   UUID,
    arm             String,
    trial           UInt32,                        -- pair or repeat index
    position        UInt8 DEFAULT 0,               -- run order within the trial
    warmup          Bool DEFAULT false,
    split           LowCardinality(String) DEFAULT 'search',  -- search | hidden | confirm
    status          LowCardinality(String) DEFAULT 'ok',      -- ok | fail | hang | no_metric | void
    workload        String DEFAULT '',
    stratum         String DEFAULT '',             -- arch/card_class/cpu_model: floors transfer only within one
    metric          LowCardinality(String),
    value           Float64,
    unit            LowCardinality(String) DEFAULT '',
    kind            LowCardinality(String) DEFAULT 'measured',  -- measured | predicted
    fidelity        LowCardinality(String) DEFAULT 'hardware',  -- hardware | emulator | cost_model | analytic
    wall_s          Float64 DEFAULT 0,
    load1_before    Float32 DEFAULT 0,
    load1_after     Float32 DEFAULT 0,
    node            String DEFAULT '',
    env_id          UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),  -- run_env_reports
    run_id          UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),  -- set when a CI run took it
    benchmark_id    UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_arm    CHECK arm != '',
    CONSTRAINT chk_split  CHECK split IN ('search', 'hidden', 'confirm'),
    CONSTRAINT chk_status CHECK status IN ('ok', 'fail', 'hang', 'no_metric', 'void'),
    CONSTRAINT chk_metric CHECK status != 'ok' OR metric != '',
    CONSTRAINT chk_kind   CHECK kind IN ('measured', 'predicted'),
    CONSTRAINT chk_fidelity CHECK fidelity IN ('hardware', 'emulator', 'cost_model', 'analytic')
)
ENGINE = MergeTree()
ORDER BY (experiment_id, metric, arm, trial, sample_id);

-- A stopping-rule evaluation: interim rows as the run goes, final rows when it stops.
-- e_value and p_value are NULL when the test did not compute them (no floor yet, n < 2).
CREATE TABLE IF NOT EXISTS experiment_decisions
(
    ts              DateTime64(3, 'UTC'),
    decision_id     UUID,
    experiment_id   UUID,
    arm             String,                        -- the candidate
    baseline_arm    String,
    metric          LowCardinality(String),
    final           Bool DEFAULT false,
    n               UInt32,
    test            LowCardinality(String) DEFAULT '',  -- paired_log_t | msprt | welch | mwu | bootstrap
    effect_pct      Float64,                       -- positive = candidate better
    ci_lo_pct       Float64,
    ci_hi_pct       Float64,
    sd_log          Float64,
    e_value         Nullable(Float64),
    p_value         Nullable(Float64),
    floor_pct       Nullable(Float64),
    decision        LowCardinality(String),        -- continue | effect | null | max_n | stopped
    stop_reason     String DEFAULT '',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_arm      CHECK arm != '' AND baseline_arm != '',
    CONSTRAINT chk_metric   CHECK metric != '',
    CONSTRAINT chk_decision CHECK decision IN ('continue', 'effect', 'null', 'max_n', 'stopped')
)
ENGINE = MergeTree()
ORDER BY (experiment_id, arm, metric, ts, decision_id);

-- A claim an experiment supports. gain_evidence G0-G4 (predicted .. realized after merge) is
-- set by code from the evidence, never by hand, and is distinct from attribution evidence tiers.
-- `estimate` is in `unit`: effect % for effect, run sd % for noise_floor, error delta for calibration,
-- a rate for method (a test's false-positive rate or power).
CREATE TABLE IF NOT EXISTS experiment_findings
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    event_id        UUID DEFAULT generateUUIDv7(),
    finding_id      UUID,
    experiment_id   UUID,
    kind            LowCardinality(String),        -- effect | noise_floor | calibration | parity_gap | regression | feasibility | method
    arm             String DEFAULT '',
    baseline_arm    String DEFAULT '',
    subject_level   LowCardinality(String) DEFAULT '',
    subject_key     String DEFAULT '',
    workload        String DEFAULT '',
    stratum         String DEFAULT '',
    metric          LowCardinality(String),
    unit            LowCardinality(String) DEFAULT 'pct',
    estimate        Float64,
    ci_lo           Float64 DEFAULT nan,
    ci_hi           Float64 DEFAULT nan,
    n               UInt32 DEFAULT 0,
    floor_pct       Float64 DEFAULT nan,
    verdict         LowCardinality(String) DEFAULT '',  -- better | worse | equal | inconclusive | infeasible | void
    gain_evidence   LowCardinality(String) DEFAULT '',  -- G0 | G1 | G2 | G3 | G4
    demoted_by      Array(LowCardinality(String)), -- reasons the level is one lower: no_env_report, load, n_min, ...
    gates           Map(LowCardinality(String), String),  -- gate -> passed | failed | skipped
    policy_version  String DEFAULT '',
    state           LowCardinality(String) DEFAULT 'published',  -- draft | published | retracted
    summary         String DEFAULT '',
    actor           String,
    actor_kind      LowCardinality(String) DEFAULT 'human',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_finding  CHECK finding_id != toUUID('00000000-0000-0000-0000-000000000000'),
    CONSTRAINT chk_kind     CHECK kind IN ('effect', 'noise_floor', 'calibration', 'parity_gap', 'regression', 'feasibility', 'method'),
    CONSTRAINT chk_metric   CHECK metric != '',
    CONSTRAINT chk_verdict  CHECK verdict IN ('', 'better', 'worse', 'equal', 'inconclusive', 'infeasible', 'void'),
    CONSTRAINT chk_evidence CHECK gain_evidence IN ('', 'G0', 'G1', 'G2', 'G3', 'G4'),
    CONSTRAINT chk_effect   CHECK kind != 'effect' OR (verdict != '' AND gain_evidence != ''),
    CONSTRAINT chk_state    CHECK state IN ('draft', 'published', 'retracted'),
    CONSTRAINT chk_actor    CHECK actor != '',
    CONSTRAINT chk_actor_kind CHECK actor_kind IN ('human', 'anonymous', 'agent', 'service')
)
ENGINE = MergeTree()
ORDER BY (experiment_id, finding_id, ts);

-- Joins an experiment (or one finding) to what it used or produced elsewhere: spyre_v2 runs,
-- artifacts and benchmarks by id; traces, files, PRs and commits by reference.
CREATE TABLE IF NOT EXISTS experiment_links
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    link_id         UUID,                          -- uuid5 of (experiment, finding, kind, target)
    experiment_id   UUID,
    finding_id      UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),
    link_kind       LowCardinality(String),        -- run | artifact | benchmark | ci_request | trace | file | pr | issue | commit | experiment
    role            LowCardinality(String) DEFAULT 'evidence',  -- baseline | candidate | evidence | source | output | supersedes
    target_id       UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),
    target          String DEFAULT '',             -- url, path, repo@sha, id12, trace id
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_kind   CHECK link_kind IN ('run', 'artifact', 'benchmark', 'ci_request', 'trace', 'file', 'pr', 'issue', 'commit', 'experiment'),
    CONSTRAINT chk_role   CHECK role IN ('baseline', 'candidate', 'evidence', 'source', 'output', 'supersedes'),
    CONSTRAINT chk_target CHECK target_id != toUUID('00000000-0000-0000-0000-000000000000') OR target != ''
)
ENGINE = MergeTree()
ORDER BY (experiment_id, link_kind, target_id, target);

-- The host, card and software a measurement ran on, written once per distinct report. Any
-- producer may cite it (benchmark_runs.props['env_id'] as well as experiment_samples.env_id).
CREATE TABLE IF NOT EXISTS run_env_reports
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    env_id          UUID,                          -- uuid5 of env_sha
    env_sha         String,                        -- the producer's content hash of the report
    host            String DEFAULT '',
    node            String DEFAULT '',
    arch            LowCardinality(String) DEFAULT '',
    cpu_model       LowCardinality(String) DEFAULT '',
    ncpu            UInt32 DEFAULT 0,
    cpu_quota       String DEFAULT '',
    host_kernel     String DEFAULT '',
    card_class      LowCardinality(String) DEFAULT '',  -- PF | VF
    cards           Array(String),                 -- PCI addresses
    image           String DEFAULT '',
    image_digest    String DEFAULT '',
    artifact_id     UUID DEFAULT toUUID('00000000-0000-0000-0000-000000000000'),
    git_shas        Map(String, String),           -- repo -> sha
    packages        Map(String, String),           -- rpm or pip name -> version
    env             Map(String, String),
    profiler_build  Nullable(Bool),                -- libaiupti linked; NULL = not checked
    load1           Float32 DEFAULT 0,
    raw             String DEFAULT '',             -- the report verbatim
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_env_sha    CHECK env_sha != '',
    CONSTRAINT chk_card_class CHECK card_class IN ('', 'PF', 'VF')
)
ENGINE = MergeTree()
ORDER BY env_id;
