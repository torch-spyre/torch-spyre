-- Artifact registry, schema v2 -- a clean break from v1's artifacts/artifact_tags/artifact_results/
-- artifact_metadata set. Rationale for every decision: docs/clickhouse_v2_artifacts_schema.md
--
-- Organising invariant: `artifacts`, `artifact_refs` and `artifact_results` hold facts fixed at
-- production time, keyed by artifact_id; `artifact_tags` is the only mutable layer and points
-- down at the others, so `artifacts` never needs an ALTER per new channel or tier.
--
-- Bag-column convention, uniform with 10-functional-tests.sql: `props` = Map, open-ended, never
-- in a key; `tags` = Array, a SET, and IN an identity hash (sort before hashing).

CREATE TABLE IF NOT EXISTS artifacts
(
    ts            DateTime DEFAULT now(),
    -- uuid5 of (component, artifact_name, id12, arch) -- derived, so a consumer holding those
    -- four fields needs nothing threaded to it; the inputs stay in `props` since a uuid can't be read back.
    artifact_id   UUID,
    component     LowCardinality(String),
    -- amd64 | ppc64le | s390x, plus 'multi' for manifest-join refs; run_id's hash folds to x86_64.
    arch          LowCardinality(String),
    kind          LowCardinality(String),   -- image | rpm | wheel | generic
    artifact_name String,                   -- e.g. ibm-flex-devel; display only, NOT identity

    origin        LowCardinality(String),   -- how it came to exist, as history not plan:
                                            -- built | copied | promoted | upstream

    -- Split by identity participation: config.yaml's `identity: false` affects build order but
    -- must not feed the reuse hash. identity_deps hash INTO the id (e.g. ['flex@<id12>']);
    -- context_deps (chiefly the builder image) sit outside it, so a builder bump costs no rebuild.
    identity_deps Array(String),
    context_deps  Array(String),

    -- An array, not a table: multi-repo builds carry N sources with no natural ordering.
    sources       Array(Tuple(
                      repo    LowCardinality(String),
                      git_ref String,
                      git_sha String
                  )),

    -- id12/artifact_name are hash inputs kept readable here since artifact_id is opaque; run_url
    -- is the ONE url key across the schema (v1 spread this over five differently-named columns).
    props         Map(LowCardinality(String), String),  -- id12, content hash + alg, size,
                                                        -- labels, run_url, build_number

    CONSTRAINT chk_kind   CHECK kind   IN ('image','rpm','wheel','generic'),
    -- built=compiled here; copied=republished at a new arch (new id); promoted=same id gaining a
    -- channel tag (the tag itself lives in artifact_tags); upstream=pinned third-party, not built.
    -- No 'reused': reuse is an edge a job records against the existing id, not an artifact property.
    CONSTRAINT chk_origin CHECK origin IN ('built','copied','promoted','upstream')
)
ENGINE = MergeTree()
ORDER BY (component, arch, artifact_id);
-- MergeTree, not Replacing: a duplicate artifact_id is a producer bug that must stay visible.
-- No PARTITION BY -- the reuse/tier gate filters identity with no time predicate.


CREATE TABLE IF NOT EXISTS artifact_refs
(
    ts             DateTime DEFAULT now(),
    artifact_id    UUID,

    -- method is HOW a consumer obtains it; ref_kind is the shape `ref` takes (image: pullspec;
    -- rpm: dnf/glob; wheel: pip/url; generic: download/url).
    method         LowCardinality(String),  -- container-pull | dnf | pip | download
    ref_kind       LowCardinality(String),  -- pullspec | glob | url
    index_uri      String,                  -- registry host, yum repo base, PyPI index
    -- A glob for RPMs: the NEVRA's version/build segments are unpredictable.
    ref            String,
    content_digest String DEFAULT '',       -- '' rather than Nullable: no per-row null mask
    props          Map(LowCardinality(String), String),

    CONSTRAINT chk_method   CHECK method   IN ('container-pull','dnf','pip','download'),
    CONSTRAINT chk_ref_kind CHECK ref_kind IN ('pullspec','glob','url')
)
ENGINE = ReplacingMergeTree(ts)
ORDER BY (artifact_id, method, ref);
-- Holds only addresses that never move, so republishing to the same index/method dedupes
-- correctly; a ref pinned to one point in time (an id12 or date) belongs here, not on the tag.


CREATE TABLE IF NOT EXISTS artifact_tags
(
    ts             DateTime DEFAULT now(),

    tag            String,                  -- THE resolution key: 'nightly', 'nightly-2026-08-31'
    -- Reporting dimension only; the writer WARNs on a value outside nightly/weekly/main/pr rather
    -- than losing the row (there is no 'dev').
    tag_family     LowCardinality(String),  -- nightly | weekly | main | pr
    artifact_id    UUID,                    -- what the tag pointed to as of ts

    -- Inline, not a reference into artifact_refs: these are MOVING addresses (':nightly', no
    -- id12), and that table's ReplacingMergeTree would collapse successive tag holders into one row.
    refs           Array(Tuple(
                       method    LowCardinality(String),
                       ref_kind  LowCardinality(String),
                       index_uri String,
                       ref       String
                   )),
    -- The other direction: an immutable address a promotion also publishes lives in artifact_refs;
    -- only its key is recorded here.
    published_refs Array(String),
    props          Map(LowCardinality(String), String)     -- promoted_by, run_url, actor
)
ENGINE = MergeTree()
ORDER BY (tag, ts);
-- No skip index on artifact_id: a tag-led sort scatters an artifact's rows, so a bloom filter
-- would prune nothing, and the table is small by construction (one row per promotion). Both
-- arrays may be empty (a re-promotion publishing no new address). Plain MergeTree, since every
-- promotion is a new fact, not a new version -- rolling vs pinned is emergent (uniqExact(artifact_id) > 1).


CREATE TABLE IF NOT EXISTS artifact_results
(
    ts          DateTime DEFAULT now(),
    artifact_id UUID,                       -- WHAT was tested: immutable identity, never a tag
    run_id      UUID,                       -- joins the test/benchmark run for case detail

    result_kind LowCardinality(String),     -- functional | performance | image
    test_type   LowCardinality(String),     -- the tier ladder, constrained below
    state       LowCardinality(String),     -- passed | failed | error | running
    arch        LowCardinality(String),     -- where it RAN; may differ from artifacts.arch

    -- No stored total_tests/passed/failed/errors/skipped: exactly derivable from test_case_runs,
    -- and a stored copy drifts once a delta run copies a covering run's cases in.
    duration_s  Float32,                    -- suite wall clock; usually != sum(cases)

    -- run_url is THE link (Jenkins or GHA, one key); no separate run_key/job_key since the key
    -- form is recoverable from the url and is already a run_id hash input.
    props       Map(LowCardinality(String), String),  -- run_url, source, per-producer keys

    -- Closes a v1 defect where image names leaked into test_type. Not an Enum: an unknown value
    -- would throw on insert instead of needing an ALTER, and Enum declaration order would silently
    -- reorder tier_satisfies()'s ladder comparisons.
    CONSTRAINT chk_test_type   CHECK test_type   IN
        ('smoke','unit','integration','regression','trunk','perf'),
    CONSTRAINT chk_state       CHECK state       IN ('passed','failed','error','running'),
    CONSTRAINT chk_result_kind CHECK result_kind IN ('functional','performance','image'),

    -- run_id can't lead the sort key (reads want "this artifact's verdicts" first), so this index
    -- covers the reverse direction; built inline since ALTER...ADD INDEX registers but builds nothing.
    INDEX idx_run_id run_id TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (artifact_id, result_kind, test_type, ts)
TTL ts + INTERVAL 90 DAY DELETE WHERE state = 'running';
-- A sparse junction, not a spine: only a minority of runs land a row here, so run metadata must
-- not live on this table. state='running' is advisory only (ClickHouse has no locks); the TTL
-- reaps orphaned rows from crashed runs, and the tier gate accepts only 'passed'.
