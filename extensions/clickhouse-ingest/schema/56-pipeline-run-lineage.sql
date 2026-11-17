-- Run-to-run edges recorded by the side that holds the fact (the launcher). Append-only: a dispatch
-- is first written as an intent (to_run_key = '', confidence 'unresolved'), then again once the child
-- run is known; a reconciler may add a lower-confidence resolution later. link_id ties the rows of
-- one dispatch together. Jenkins -> Jenkins edges are not written here: the child holds that fact
-- in pipeline_runs.parent_run_key, which v_pipeline_run_edges reads alongside.
CREATE TABLE IF NOT EXISTS pipeline_run_links
(
    link_id         String,
    observed_at     DateTime64(3, 'UTC') DEFAULT now64(3),
    from_run_key    String,
    to_run_key      String DEFAULT '',
    relation        LowCardinality(String) DEFAULT 'launched',
    -- recorded: written by the launcher; cited: the child names its parent; inferred: matched after the fact.
    evidence_kind   LowCardinality(String),
    -- exact: the dispatch returned the run id; timestamp: matched by creation time; unresolved: no single candidate.
    confidence      LowCardinality(String),
    -- "<owner>/<repo>/<workflow file>" the launcher dispatched; what the reconciler matches on.
    target          String DEFAULT '',
    dispatched_at   Nullable(DateTime64(3, 'UTC')),
    recorded_by     LowCardinality(String) DEFAULT '',
    props           Map(LowCardinality(String), String),
    audit_uuid      UUID DEFAULT generateUUIDv7(),
    audit_timestamp DateTime64(3) DEFAULT now64(3),

    CONSTRAINT chk_relation   CHECK relation IN ('launched', 'retry_of'),
    CONSTRAINT chk_evidence   CHECK evidence_kind IN ('recorded', 'cited', 'inferred'),
    CONSTRAINT chk_confidence CHECK confidence IN ('exact', 'timestamp', 'low', 'unresolved'),
    CONSTRAINT chk_resolved   CHECK to_run_key != '' OR confidence = 'unresolved'
)
ENGINE = MergeTree()
ORDER BY (from_run_key, link_id, observed_at);

-- One row per dispatch: the best resolution seen so far, else the intent.
CREATE VIEW IF NOT EXISTS v_pipeline_run_links AS
SELECT
    link_id, from_run_key, relation, target, cur_dispatched_at AS dispatched_at,
    cur_to AS to_run_key, cur_evidence AS evidence_kind, cur_confidence AS confidence,
    cur_observed_at AS observed_at
FROM
(
    SELECT
        link_id,
        any(from_run_key)  AS from_run_key,
        any(relation)      AS relation,
        any(target)        AS target,
        min(dispatched_at) AS cur_dispatched_at,
        argMax(to_run_key,     (to_run_key != '', observed_at)) AS cur_to,
        argMax(evidence_kind,  (to_run_key != '', observed_at)) AS cur_evidence,
        argMax(confidence,     (to_run_key != '', observed_at)) AS cur_confidence,
        max(observed_at)   AS cur_observed_at
    FROM pipeline_run_links
    GROUP BY link_id
);

-- Dispatches still waiting for a child run id: the reconciler's work list.
CREATE VIEW IF NOT EXISTS v_pipeline_run_links_pending AS
SELECT * FROM v_pipeline_run_links WHERE to_run_key = '';

-- The best single parent of each run, from a recorded link or the child's own parent_run_key.
-- rank: exact link 4, parent_run_key 3, timestamp-matched link 2, anything else 1.
CREATE VIEW IF NOT EXISTS v_pipeline_run_edges AS
SELECT
    child_run_key, cur_parent AS parent_run_key, cur_via AS via, cur_rank AS rank
FROM
(
    SELECT
        child_run_key,
        argMax(parent, rnk) AS cur_parent,
        argMax(via, rnk)    AS cur_via,
        max(rnk)            AS cur_rank
    FROM
    (
        SELECT to_run_key AS child_run_key, from_run_key AS parent, 'pipeline_run_links' AS via,
               multiIf(confidence = 'exact', 4, confidence = 'timestamp', 2, 1) AS rnk
        FROM v_pipeline_run_links
        WHERE to_run_key != ''
        UNION ALL
        SELECT run_key AS child_run_key, parent_run_key AS parent, 'parent_run_key' AS via, 3 AS rnk
        FROM v_pipeline_runs
        WHERE parent_run_key != ''
    )
    WHERE child_run_key != parent
    GROUP BY child_run_key
);

-- Every run with its ancestors, nearest first, up to six levels (PR -> orchestrator -> component-build
-- -> GHA workflow -> GHA job is four). Down-walk: filter on root_run_key. depth = ancestors found.
CREATE VIEW IF NOT EXISTS v_pipeline_run_lineage AS
SELECT
    r.run_key AS run_key, r.source AS source, r.pipeline_type AS pipeline_type,
    e1.parent_run_key AS parent_run_key, e1.via AS parent_via, e1.rank AS parent_rank,
    arrayFilter(x -> x != '', [e1.parent_run_key, e2.parent_run_key, e3.parent_run_key,
                               e4.parent_run_key, e5.parent_run_key, e6.parent_run_key]) AS ancestors,
    length(ancestors) AS depth,
    if(depth = 0, r.run_key, ancestors[depth]) AS root_run_key
FROM v_pipeline_runs AS r
LEFT JOIN v_pipeline_run_edges AS e1 ON e1.child_run_key = r.run_key
LEFT JOIN v_pipeline_run_edges AS e2 ON e2.child_run_key = e1.parent_run_key
LEFT JOIN v_pipeline_run_edges AS e3 ON e3.child_run_key = e2.parent_run_key
LEFT JOIN v_pipeline_run_edges AS e4 ON e4.child_run_key = e3.parent_run_key
LEFT JOIN v_pipeline_run_edges AS e5 ON e5.child_run_key = e4.parent_run_key
LEFT JOIN v_pipeline_run_edges AS e6 ON e6.child_run_key = e5.parent_run_key;

-- Runs with neither a parent edge nor a reason to be a root: the traceability gaps.
CREATE VIEW IF NOT EXISTS v_pipeline_run_lineage_gaps AS
SELECT
    r.run_key AS run_key, r.source AS source, r.pipeline_type AS pipeline_type,
    r.trigger_kind AS trigger_kind, r.repo AS repo, r.pr_number AS pr_number, r.started_at AS started_at
FROM v_pipeline_runs AS r
LEFT JOIN v_pipeline_run_edges AS e ON e.child_run_key = r.run_key
WHERE e.child_run_key = ''
  AND NOT (r.pipeline_type = 'orchestrator'
           OR (r.source = 'jenkins' AND r.trigger_kind IN ('manual', 'timer'))
           OR (r.pipeline_type = 'gha-workflow'
               AND r.trigger_kind IN ('pull_request', 'pull_request_target', 'push', 'schedule',
                                      'merge_group', 'issue_comment')));
