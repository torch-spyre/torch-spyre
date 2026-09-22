-- Artifact registry, schema v2. A clean break from the v1
-- artifacts/artifact_tags/artifact_results/artifact_metadata set, on fresh data.
-- Rationale for every decision below: docs/clickhouse_v2_artifacts_schema.md
--
-- The organising invariant: `artifacts`, `artifact_refs` and `artifact_results` hold facts
-- fixed at production time, keyed by artifact_id. `artifact_tags` is the only mutable
-- layer and points down at the others -- nothing immutable references a tag. That is what
-- keeps `artifacts` from needing an ALTER per new channel, consume method or test tier.
--
-- Bag-column convention, uniform with 10-functional-tests.sql:
--   `props` = Map, open-ended, NEVER in a key -- extend freely, no identity churn.
--   `tags`  = Array, a SET, and IN an identity hash. Sort before hashing.
-- The names differ because the guarantees differ; do not converge them.

CREATE TABLE IF NOT EXISTS artifacts
(
    ts            DateTime DEFAULT now(),
    -- uuid5 derived from (component, artifact_name, id12, arch) -- artifactId() in
    -- pushToClickhouse.groovy and v2_artifact_id() in the ingest scripts must agree.
    -- Derived, so a consumer holding those four fields needs nothing threaded to it; a GHA
    -- leg has no id12 and hashes (base image + installed set) into that slot. arch is
    -- required: one id12 exists per arch plus a multi pointer, so dropping it collides.
    -- The four inputs stay in `props` -- a uuid cannot be read back.
    artifact_id   UUID,
    component     LowCardinality(String),
    -- amd64 | ppc64le | s390x, plus 'multi' for manifest-join refs. Prod carries two
    -- spellings by table family (here amd64; test_runs/images x86_64) and run_id's hash
    -- folds to x86_64 -- use the normalizing views cross-family, and never parse an arch
    -- out of an artifact_id or run_id.
    arch          LowCardinality(String),
    kind          LowCardinality(String),   -- image | rpm | wheel | generic
    artifact_name String,                   -- e.g. ibm-flex-devel; display only, NOT identity

    origin        LowCardinality(String),   -- how it came to exist, as history not plan:
                                            -- built | copied | promoted | upstream

    -- Split by identity participation: config.yaml's `identity: false` affects build order
    -- but must NOT feed the reuse hash.
    -- identity_deps: content that IS an input to this artifact's identity, e.g.
    --   ['flex@<id12>', 'deeptools@<id12>'].
    -- context_deps: present at build time, deliberately outside the hash. Chiefly the
    --   BUILDER image -- the artifact is a pure function of its own source+recipe, so
    --   folding the builder in would churn a cold rebuild per builder bump for identical
    --   output. See resolve_deps.py `identity`.
    identity_deps Array(String),
    context_deps  Array(String),

    -- An array, not a table: multi-repo builds carry N sources with no natural ordering,
    -- and this is immutable, written once with the artifact.
    sources       Array(Tuple(
                      repo    LowCardinality(String),
                      git_ref String,
                      git_sha String
                  )),

    -- id12 and artifact_name are required here, not optional metadata: they are hash
    -- inputs to artifact_id, so with the id opaque this is where they stay readable.
    --
    -- ONE url key across the schema: `run_url`, the CI run that produced or reported the
    -- row, Jenkins or GHA. v1 spread this over job_url / build_url / gha_url / run_urls /
    -- artifact_url plus orch_run_key / test_job_key / job_key, so a reader had to know the
    -- producer before knowing which field to read. Nothing ever joined on any of them.
    props         Map(LowCardinality(String), String),  -- id12, content hash + alg, size,
                                                        -- labels, run_url, build_number

    CONSTRAINT chk_kind   CHECK kind   IN ('image','rpm','wheel','generic'),
    -- Each value is a distinct way a row comes to exist, and the artifact_id differs in
    -- every case except 'promoted':
    --   built    -- this pipeline compiled it.
    --   copied   -- same content republished at a second address (an arch-specific image
    --               under a multi-arch manifest): new arch, so a new artifact_id.
    --   promoted -- the SAME artifact_id gaining a channel tag (nightly -> 2.0). The row is
    --               the promotion EVENT; the tag itself lives in artifact_tags.
    --   upstream -- not built here: a third-party wheel or base image we pin and test.
    --               Verdicts hang off it, so it needs a row.
    -- No 'reused' value: reuse is not a property of the artifact, which exists once. A
    -- reusing job records itself against the same artifact_id, so reuse is an edge.
    CONSTRAINT chk_origin CHECK origin IN ('built','copied','promoted','upstream')
)
ENGINE = MergeTree()
ORDER BY (component, arch, artifact_id);
-- MergeTree, not Replacing: an artifact is produced once and never restated, so a
-- duplicate artifact_id is a producer bug that must stay visible. No PARTITION BY -- the
-- hot query (the reuse/tier gate) filters identity with no time predicate, so a monthly
-- partition prunes nothing and only fragments parts at this size.


CREATE TABLE IF NOT EXISTS artifact_refs
(
    ts             DateTime DEFAULT now(),
    artifact_id    UUID,

    -- method is HOW a consumer obtains it; ref_kind is the shape `ref` therefore takes.
    -- They move together, by kind:
    --   image:   'container-pull' / 'pullspec' -- ref a full tagged pullspec
    --   rpm:     'dnf' / 'glob'   -- index_uri the yum repo base, ref an NEVRA glob
    --   wheel:   'pip' / 'url'    -- index_uri the team PyPI index
    --   generic: 'download' / 'url' -- ref the full Artifactory generic-local path
    method         LowCardinality(String),  -- container-pull | dnf | pip | download
    ref_kind       LowCardinality(String),  -- pullspec | glob | url
    index_uri      String,                  -- registry host, yum repo base, PyPI index
    -- A glob for RPMs: the NEVRA's version/build segments are unpredictable, and the
    -- gitversion sha7 in the filename is NOT the PR-head sha -- the producer embeds that
    -- separately as an `h<sha7>` token, which is what the glob matches.
    ref            String,
    content_digest String DEFAULT '',       -- '' rather than Nullable: absence is the
                                            -- empty string, no per-row null mask
    props          Map(LowCardinality(String), String),

    CONSTRAINT chk_method   CHECK method   IN ('container-pull','dnf','pip','download'),
    CONSTRAINT chk_ref_kind CHECK ref_kind IN ('pullspec','glob','url')
)
ENGINE = ReplacingMergeTree(ts)
ORDER BY (artifact_id, method, ref);
-- Holds only addresses that never move, so re-publishing the same artifact to the same
-- index by the same method is the same fact and dedupe is correct.
-- Routing rule: a ref carrying any discriminator pinning it to one artifact or one point
-- in time (an id12 or a date) is immutable and belongs here, dated promotion aliases
-- included. Only a bare channel address goes on the tag.


CREATE TABLE IF NOT EXISTS artifact_tags
(
    ts             DateTime DEFAULT now(),

    tag            String,                  -- THE resolution key: 'nightly', 'nightly-2026-08-31'
    -- Reporting dimension only, never the tag itself. The writer validates against these
    -- four and only WARNs otherwise, so a stray value stays visible rather than losing the
    -- row. nightly/weekly come from the orchestrator's dated-retag flow, main from a flat
    -- promote-to-2.0, pr from a per-PR tag. There is no 'dev'.
    tag_family     LowCardinality(String),  -- nightly | weekly | main | pr
    artifact_id    UUID,                    -- what the tag pointed to as of ts

    -- Inline, NOT a reference into artifact_refs, because these are a different KIND of
    -- address: moving ones that exist only because of this tag (':nightly', no id12, no
    -- date). Keying them by artifact_id would assert the artifact owns an address that
    -- outlives its claim on it, and that table's ReplacingMergeTree would collapse
    -- successive tag holders onto one row, destroying the history this table keeps.
    -- Frozen at ts, they answer "what did ':nightly' mean on 08-31".
    refs           Array(Tuple(
                       method    LowCardinality(String),
                       ref_kind  LowCardinality(String),
                       index_uri String,
                       ref       String
                   )),
    -- The other direction: a promotion that also publishes an immutable address writes it
    -- to artifact_refs and records only its key here, so the string is not owned twice.
    published_refs Array(String),
    props          Map(LowCardinality(String), String)     -- promoted_by, run_url, actor
)
ENGINE = MergeTree()
ORDER BY (tag, ts);
-- Deliberately no skip index on artifact_id for the reverse lookup. A bloom filter prunes
-- only on contiguous rows and a tag-led sort scatters them, since an artifact routinely
-- carries several tags (a rolling tag coexists with its dated aliases). The table is also
-- small by construction -- one row per promotion -- so the reverse scan is cheap.
-- The row's payload is the membership fact (tag, artifact_id, ts); both arrays are
-- optional annotations and BOTH may be empty (re-promoting an already-published artifact
-- publishes no address and moves nothing). Never constrain them to be non-empty.
-- Plain MergeTree: every promotion is a new fact, not a new version of one -- dedupe would
-- collapse the rolling tag's history. Rolling vs pinned is emergent, not stored:
-- uniqExact(artifact_id) > 1 per tag.
-- ORDER BY leads with `tag`, the resolution key; tag_family is only a grouping dimension.


CREATE TABLE IF NOT EXISTS artifact_results
(
    ts          DateTime DEFAULT now(),
    artifact_id UUID,                       -- WHAT was tested: immutable identity, never a
                                            -- tag (a moved tag would make an old verdict
                                            -- describe a new artifact)
    run_id      UUID,                       -- joins the test/benchmark run for case detail

    result_kind LowCardinality(String),     -- functional | performance | image
    test_type   LowCardinality(String),     -- the tier ladder, constrained below
    state       LowCardinality(String),     -- passed | failed | error | running
    arch        LowCardinality(String),     -- where it RAN; may differ from artifacts.arch
                                            -- (a 'multi' manifest tested on amd64)

    -- No stored total_tests/passed/failed/errors/skipped: they are exactly derivable by
    -- counting test_case_runs for this run_id, and a stored copy drifts the moment a delta
    -- run copies a covering run's cases in. `state` still distinguishes a suite that never
    -- ran ('error'/'running' with no case rows) from one that ran and regressed.
    duration_s  Float32,                    -- suite wall clock; usually != sum(cases)

    -- `run_url` is THE link: the CI run behind this verdict, Jenkins or GHA, under one key
    -- so a reader never needs to know which system ran it. Emitted even when it may have
    -- aged out under numToKeep -- a 404 saying "it ran here" beats no link. No separate
    -- run_key/job_key: the key form (JOB_NAME#BUILD_NUMBER) is recoverable from the url,
    -- nothing joined on it, and it is already a run_id hash input.
    props       Map(LowCardinality(String), String),  -- run_url, source, per-producer keys

    -- Closes a live v1 defect where image names were written into test_type
    -- (spyre-inference-dev, hf-adapters-dev). Deliberately NOT an Enum: an unknown value
    -- would throw on insert, so adding a tier would need an ALTER before the writer could
    -- emit it, and Enum ordering is by declaration -- a tier inserted in the middle would
    -- silently reorder the ladder comparisons tier_satisfies() depends on.
    CONSTRAINT chk_test_type   CHECK test_type   IN
        ('smoke','unit','integration','regression','trunk','perf'),
    CONSTRAINT chk_state       CHECK state       IN ('passed','failed','error','running'),
    CONSTRAINT chk_result_kind CHECK result_kind IN ('functional','performance','image'),

    -- run_id is the join key the rest of the schema hangs off, but it cannot lead the sort
    -- key: reads are overwhelmingly "this artifact's verdicts", which needs artifact_id
    -- first. This index covers the other direction, and pays only because a run's rows are
    -- contiguous here -- every run maps to exactly one artifact, so a granule either holds
    -- the run or does not.
    -- Inline, not ALTER ... ADD INDEX: an ADD registers the index but builds nothing until
    -- MATERIALIZE INDEX, and a registered-but-empty index prunes zero.
    INDEX idx_run_id run_id TYPE bloom_filter(0.01) GRANULARITY 1
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (artifact_id, result_kind, test_type, ts)
TTL ts + INTERVAL 90 DAY DELETE WHERE state = 'running';
-- A sparse junction, not a spine -- only a small minority of runs land a row here, so run
-- metadata must NOT live on this table.
-- state='running' is advisory display only. Mutual exclusion stays in Jenkins lock() and
-- liveness in Jenkins build state; ClickHouse has neither row updates nor locks, and a
-- crashed run's stale 'running' row must never gate anything. The TTL reaps orphans. The
-- tier gate accepts ONLY 'passed'.
