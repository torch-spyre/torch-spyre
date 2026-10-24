-- Current state of the append-only analyst layer tables.

-- Newest version of each role memory, every status kept: a reader needs the dead rows to know what to drop.
CREATE VIEW IF NOT EXISTS v_analyst_role_memory_current AS
WITH cur AS
(
    SELECT
        memory_id,
        argMax(scope, updated_at) AS cur_scope,
        argMax(namespace, updated_at) AS cur_namespace,
        argMax(kind, updated_at) AS cur_kind,
        argMax(subject, updated_at) AS cur_subject,
        argMax(body, updated_at) AS cur_body,
        argMax(evidence_ids, updated_at) AS cur_evidence_ids,
        argMax(confidence, updated_at) AS cur_confidence,
        argMax(status, updated_at) AS cur_status,
        argMax(proposed_by, updated_at) AS cur_proposed_by,
        argMax(confirmed_by, updated_at) AS cur_confirmed_by,
        argMax(author_role, updated_at) AS cur_author_role,
        argMax(author_family, updated_at) AS cur_author_family,
        argMax(expires_at, updated_at) AS cur_expires_at,
        argMax(props, updated_at) AS cur_props,
        max(updated_at) AS last_updated_at
    FROM analyst_role_memory
    GROUP BY memory_id
)
SELECT
    memory_id,
    cur_scope AS scope,
    cur_namespace AS namespace,
    cur_kind AS kind,
    cur_subject AS subject,
    cur_body AS body,
    cur_evidence_ids AS evidence_ids,
    cur_confidence AS confidence,
    cur_status AS status,
    cur_proposed_by AS proposed_by,
    cur_confirmed_by AS confirmed_by,
    cur_author_role AS author_role,
    cur_author_family AS author_family,
    cur_expires_at AS expires_at,
    cur_props AS props,
    last_updated_at AS updated_at
FROM cur;

-- Newest version of each project-space file; a deleted file stays as a tombstone row.
CREATE VIEW IF NOT EXISTS v_analyst_project_files_current AS
WITH cur AS
(
    SELECT
        role, project, name,
        argMax(kind, updated_at) AS cur_kind,
        argMax(text, updated_at) AS cur_text,
        argMax(deleted, updated_at) AS cur_deleted,
        argMax(props, updated_at) AS cur_props,
        max(updated_at) AS last_updated_at
    FROM analyst_project_files
    GROUP BY role, project, name
)
SELECT
    role, project, name,
    cur_kind AS kind,
    cur_text AS text,
    cur_deleted AS deleted,
    cur_props AS props,
    last_updated_at AS updated_at
FROM cur;

-- Newest row of each profile-change proposal.
CREATE VIEW IF NOT EXISTS v_analyst_profile_changes_current AS
WITH cur AS
(
    SELECT
        change_id,
        argMax(proposed_by, created_at) AS cur_proposed_by,
        argMax(target_role, created_at) AS cur_target_role,
        argMax(target, created_at) AS cur_target,
        argMax(text, created_at) AS cur_text,
        argMax(evidence_outcomes, created_at) AS cur_evidence_outcomes,
        argMax(status, created_at) AS cur_status,
        argMax(decided_by, created_at) AS cur_decided_by,
        argMax(props, created_at) AS cur_props,
        max(created_at) AS last_created_at
    FROM analyst_profile_changes
    GROUP BY change_id
)
SELECT
    change_id,
    cur_proposed_by AS proposed_by,
    cur_target_role AS target_role,
    cur_target AS target,
    cur_text AS text,
    cur_evidence_outcomes AS evidence_outcomes,
    cur_status AS status,
    cur_decided_by AS decided_by,
    cur_props AS props,
    last_created_at AS created_at
FROM cur;

-- The newest scoring of each key result and role in each cycle.
CREATE VIEW IF NOT EXISTS v_goal_scores_current AS
WITH cur AS
(
    SELECT
        cycle, row_kind, subject,
        argMax(owner, scored_at) AS cur_owner,
        argMax(value, scored_at) AS cur_value,
        argMax(n, scored_at) AS cur_n,
        argMax(raw, scored_at) AS cur_raw,
        argMax(score, scored_at) AS cur_score,
        argMax(counter_value, scored_at) AS cur_counter_value,
        argMax(breached, scored_at) AS cur_breached,
        argMax(coverage, scored_at) AS cur_coverage,
        argMax(flags, scored_at) AS cur_flags,
        argMax(frozen, scored_at) AS cur_frozen,
        argMax(krs, scored_at) AS cur_krs,
        argMax(props, scored_at) AS cur_props,
        max(scored_at) AS last_scored_at
    FROM goal_scores
    GROUP BY cycle, row_kind, subject
)
SELECT
    cycle, row_kind, subject,
    cur_owner AS owner,
    cur_value AS value,
    cur_n AS n,
    cur_raw AS raw,
    cur_score AS score,
    cur_counter_value AS counter_value,
    cur_breached AS breached,
    cur_coverage AS coverage,
    cur_flags AS flags,
    cur_frozen AS frozen,
    cur_krs AS krs,
    cur_props AS props,
    last_scored_at AS scored_at
FROM cur;

-- Newest row of each goal proposal: its current status and who decided it.
CREATE VIEW IF NOT EXISTS v_goal_proposals_current AS
WITH cur AS
(
    SELECT
        proposal_id,
        argMax(cycle, created_at) AS cur_cycle,
        argMax(proposed_by, created_at) AS cur_proposed_by,
        argMax(proposer_kind, created_at) AS cur_proposer_kind,
        argMax(target_role, created_at) AS cur_target_role,
        argMax(type, created_at) AS cur_type,
        argMax(target, created_at) AS cur_target,
        argMax(text, created_at) AS cur_text,
        argMax(review_id, created_at) AS cur_review_id,
        argMax(status, created_at) AS cur_status,
        argMax(decided_by, created_at) AS cur_decided_by,
        argMax(requires_human, created_at) AS cur_requires_human,
        argMax(detail, created_at) AS cur_detail,
        argMax(props, created_at) AS cur_props,
        max(created_at) AS last_created_at
    FROM goal_proposals
    GROUP BY proposal_id
)
SELECT
    proposal_id,
    cur_cycle AS cycle,
    cur_proposed_by AS proposed_by,
    cur_proposer_kind AS proposer_kind,
    cur_target_role AS target_role,
    cur_type AS type,
    cur_target AS target,
    cur_text AS text,
    cur_review_id AS review_id,
    cur_status AS status,
    cur_decided_by AS decided_by,
    cur_requires_human AS requires_human,
    cur_detail AS detail,
    cur_props AS props,
    last_created_at AS created_at
FROM cur;
