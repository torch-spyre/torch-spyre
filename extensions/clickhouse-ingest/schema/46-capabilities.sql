-- Capability analysis: for a subject, is a capability supported on a backend.
--
-- NOT test results, deliberately not in test_cases/test_case_runs: a model that cannot load is
-- an unsupported model, not a failing test, and folding these in would corrupt run_case_counters_mv's
-- pass_rate by counting every unimplemented operation as a functional failure.
--
-- EIGHT v1 TABLES COLLAPSE HERE: model_ops_{suites,variants} x {,_p,_z} (torch-spyre) and
-- {embedding,generative}_model_spyre_support (hf-adapters), each encoding a dimension in the
-- table NAME instead of a column. The v1 suites table (counters) is gone entirely: it disagreed
-- with the variants it summarised (24 stored spyre_enabled_count vs 27 distinct XPASS ops), so
-- 60-capability-views.sql aggregates instead.
--
-- SPLIT INTO IDENTITY + OBSERVATION, matching test_cases/benchmarks: 246,292 v1 rows carried
-- only 46,607 distinct identities, each re-observed 4-13 times.
--
-- FORWARD-ONLY: v1 rows cannot produce a run_id (the hash inputs were never recorded).
--
-- No branch/commit_sha: for model_ops those describe the ARTIFACT analysed, reached through
-- run_id -> artifact_results -> artifacts, so a per-row copy would only drift. model_support has
-- no such artifact at all -- it scans HuggingFace Hub checkpoints, not a build of ours, so a
-- reader must not assume every capability_run joins an artifact.
--
-- SHARDED ANALYSES share one run_id (the hf weekly scan fans out over up to 25 shards per tier),
-- so props['shard'] is the per-writer dedup discriminator, exactly as test_case_runs uses
-- props['source_file'] for a sharded XML run.


-- WHAT can be supported: the stable identity of one (subject, capability) pair.
CREATE TABLE IF NOT EXISTS capabilities
(
    ts            DateTime DEFAULT now(),

    -- uuid5 over (component, test_type, subject, name, disc) -- DERIVED, never minted. v1's
    -- variant_id was a per-row surrogate instead (51,356 distinct ids for 51,356 rows).
    capability_id UUID,

    component     LowCardinality(String),
    -- Which analysis: model_ops | model_support -- the axis v1 put in the table name. Named
    -- test_type (artifact_results' 'capability' tier covers the family; these are within it).
    -- Constrained by convention, not CHECK, so a new analysis can start writing unedited.
    test_type     LowCardinality(String),

    -- The thing analysed (a model), and the capability asked of it (a torch op for model_ops, an
    -- adapter for model_support). v1 spelled `subject` three ways for only 10 distinct triples.
    subject       String,
    name          String,

    tags          Array(LowCardinality(String)),

    -- The signature distinguishing two variants of one operation (input shapes/dtypes). Hashed
    -- into capability_id, so it cannot drift from the identity it defines.
    props         Map(LowCardinality(String), String),

    CONSTRAINT chk_component CHECK component != '',
    CONSTRAINT chk_test_type CHECK test_type != '',
    CONSTRAINT chk_name      CHECK name != ''
)
ENGINE = MergeTree()
ORDER BY (component, test_type, subject, capability_id);


-- WHETHER it was supported, per run: one row per (capability, run, backend).
CREATE TABLE IF NOT EXISTS capability_runs
(
    ts            DateTime DEFAULT now(),

    -- The only two foreign keys. artifact_id is NOT stored -- run_id determines it (0 of 1,441
    -- run_ids carry more than one artifact) via run_id -> artifact_results -> artifacts.
    run_id        UUID,
    capability_id UUID,

    -- Denormalized: leads the sort key and is a capability_id hash input, so it cannot disagree.
    component     LowCardinality(String),
    -- Which analysis, denormalized from capabilities so the dedup check can scope on it directly.
    test_type     LowCardinality(String),
    -- Replaces the _p / _z table suffixes. Canonical spelling (amd64 folds to x86_64).
    arch          LowCardinality(String),

    -- Two INDEPENDENT facts, deliberately two columns. v1 conflated XPASS/XFAIL/FALLBACK (a PASS
    -- on CPU) into one, silently undercounting operations that actually work.
    status        LowCardinality(String),
    -- Same axis and vocabulary as benchmark_runs.backend.
    backend       LowCardinality(String),

    -- Why, when status is not passed. hf-adapters emits a closed 13-value vocabulary (worth
    -- grouping by, hence LowCardinality); empty for model_ops, which emits no reason today.
    fail_reason   LowCardinality(String) DEFAULT '',

    props         Map(LowCardinality(String), String),

    CONSTRAINT chk_status CHECK status IN ('passed','failed','not_implemented')
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
-- run_id is second, not last: the same artifact re-analysed is a NEW run, kept and traceable
-- (an artifact-keyed sort would silently overwrite the earlier verdict).
ORDER BY (component, run_id, capability_id, backend);
