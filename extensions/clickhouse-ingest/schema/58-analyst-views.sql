-- Current analyst configuration and memory, and what the analyst costs.

-- The row in force for each role and scope: the newest row whose effective_from has passed, so a
-- row dated in the future is staged and not yet selected. Aggregate aliases are cur_*: an alias
-- equal to a source column is substituted into the other aggregates.
CREATE VIEW IF NOT EXISTS v_analyst_assignments_current AS
SELECT
    role, scope_cause_class, scope_severity, scope_source, scope_repo,
    argMax(model_alias, effective_from)     AS cur_model_alias,
    argMax(mode, effective_from)            AS cur_mode,
    argMax(weight, effective_from)          AS cur_weight,
    argMax(set_by, effective_from)          AS cur_set_by,
    argMax(second_approver, effective_from) AS cur_second_approver,
    argMax(reason, effective_from)          AS cur_reason,
    max(effective_from)                     AS since
FROM analyst_assignments
WHERE effective_from <= now64(3)
GROUP BY role, scope_cause_class, scope_severity, scope_source, scope_repo;

-- The newest version of each memory that is still in force. A row past expires_at reads as
-- expired whatever its stored status, and is left out.
CREATE VIEW IF NOT EXISTS v_analyst_memory_current AS
WITH cur AS
(
    SELECT
        memory_id,
        argMax(kind, updated_at)         AS cur_kind,
        argMax(subject, updated_at)      AS cur_subject,
        argMax(body, updated_at)         AS cur_body,
        argMax(evidence_ids, updated_at) AS cur_evidence_ids,
        argMax(confidence, updated_at)   AS cur_confidence,
        argMax(status, updated_at)       AS cur_status,
        argMax(proposed_by, updated_at)  AS cur_proposed_by,
        argMax(confirmed_by, updated_at) AS cur_confirmed_by,
        argMax(expires_at, updated_at)   AS cur_expires_at,
        max(updated_at)                  AS last_updated_at
    FROM analyst_memory
    GROUP BY memory_id
)
SELECT
    memory_id, cur_kind AS kind, cur_subject AS subject, cur_body AS body,
    cur_evidence_ids AS evidence_ids, cur_confidence AS confidence, cur_status AS status,
    cur_proposed_by AS proposed_by, cur_confirmed_by AS confirmed_by,
    cur_expires_at AS expires_at, last_updated_at AS updated_at
FROM cur
WHERE cur_status != 'expired' AND (cur_expires_at IS NULL OR cur_expires_at > now64(3));

-- Tokens and cost per day, role and model alias; failed calls are counted apart from good ones.
CREATE VIEW IF NOT EXISTS v_analyst_costs AS
SELECT
    toDate(ts)              AS day,
    role,
    model_alias,
    count()                 AS calls,
    countIf(outcome = 'ok') AS ok_calls,
    sum(tokens_in)          AS tokens_in_total,
    sum(tokens_out)         AS tokens_out_total,
    sum(cost_usd)           AS cost_usd_total
FROM analyst_turns
GROUP BY day, role, model_alias;
