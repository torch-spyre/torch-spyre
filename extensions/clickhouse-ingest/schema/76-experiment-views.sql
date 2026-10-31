-- Latest-state views over the 75- ledger, plus the reads its consumers make: arm verdicts, A/A
-- noise floors, campaign progress, and what was already tried for a subject.

CREATE VIEW IF NOT EXISTS v_experiments AS
SELECT
    e.experiment_id                               AS experiment_id,
    argMax(e.parent_id, (e.ts, e.event_id))       AS parent_id,
    argMax(e.name, (e.ts, e.event_id))            AS name,
    argMax(e.kind, (e.ts, e.event_id))            AS kind,
    argMax(e.state, (e.ts, e.event_id))           AS state,
    argMax(e.subject_level, (e.ts, e.event_id))   AS subject_level,
    argMax(e.subject_key, (e.ts, e.event_id))     AS subject_key,
    argMax(e.workload, (e.ts, e.event_id))        AS workload,
    argMax(e.hypothesis, (e.ts, e.event_id))      AS hypothesis,
    argMax(e.spec_sha256, (e.ts, e.event_id))     AS spec_sha256,
    argMax(e.budget, (e.ts, e.event_id))          AS budget,
    argMax(e.spent, (e.ts, e.event_id))           AS spent,
    argMax(e.code_ref, (e.ts, e.event_id))        AS code_ref,
    argMax(e.reason, (e.ts, e.event_id))          AS reason,
    argMin(e.actor, (e.ts, e.event_id))           AS created_by,
    argMax(e.props, (e.ts, e.event_id))           AS props,
    min(e.ts)                                     AS created_at,
    max(e.ts)                                     AS updated_at
FROM experiment_events AS e  -- qualified: the aliases above would shadow the columns
GROUP BY e.experiment_id;

CREATE VIEW IF NOT EXISTS v_experiment_arms AS
SELECT
    a.experiment_id                               AS experiment_id,
    a.arm                                         AS arm,
    argMax(a.role, (a.ts, a.event_id))            AS role,
    argMax(a.state, (a.ts, a.event_id))           AS state,
    argMax(a.hypothesis, (a.ts, a.event_id))      AS hypothesis,
    argMax(a.knobs, (a.ts, a.event_id))           AS knobs,
    argMax(a.knob_sha, (a.ts, a.event_id))        AS knob_sha,
    argMax(a.change_class, (a.ts, a.event_id))    AS change_class,
    argMax(a.hazard_class, (a.ts, a.event_id))    AS hazard_class,
    argMax(a.props, (a.ts, a.event_id))           AS props
FROM experiment_arms AS a
GROUP BY a.experiment_id, a.arm;

-- A later row with the same sample_id corrects one (status 'void' withdraws it).
CREATE VIEW IF NOT EXISTS v_experiment_samples AS
SELECT
    s.sample_id                                   AS sample_id,
    argMax(s.ts, s.audit_timestamp)               AS ts,
    argMax(s.experiment_id, s.audit_timestamp)    AS experiment_id,
    argMax(s.arm, s.audit_timestamp)              AS arm,
    argMax(s.trial, s.audit_timestamp)            AS trial,
    argMax(s.position, s.audit_timestamp)         AS position,
    argMax(s.warmup, s.audit_timestamp)           AS warmup,
    argMax(s.split, s.audit_timestamp)            AS split,
    argMax(s.status, s.audit_timestamp)           AS status,
    argMax(s.workload, s.audit_timestamp)         AS workload,
    argMax(s.stratum, s.audit_timestamp)          AS stratum,
    argMax(s.metric, s.audit_timestamp)           AS metric,
    argMax(s.value, s.audit_timestamp)            AS value,
    argMax(s.unit, s.audit_timestamp)             AS unit,
    argMax(s.kind, s.audit_timestamp)             AS kind,
    argMax(s.fidelity, s.audit_timestamp)         AS fidelity,
    argMax(s.wall_s, s.audit_timestamp)           AS wall_s,
    argMax(s.load1_before, s.audit_timestamp)     AS load1_before,
    argMax(s.node, s.audit_timestamp)             AS node,
    argMax(s.env_id, s.audit_timestamp)           AS env_id,
    argMax(s.run_id, s.audit_timestamp)           AS run_id,
    argMax(s.benchmark_id, s.audit_timestamp)     AS benchmark_id
FROM experiment_samples AS s
GROUP BY s.sample_id;

CREATE VIEW IF NOT EXISTS v_experiment_findings AS
SELECT
    f.finding_id                                  AS finding_id,
    argMax(f.experiment_id, (f.ts, f.event_id))   AS experiment_id,
    argMax(f.kind, (f.ts, f.event_id))            AS kind,
    argMax(f.arm, (f.ts, f.event_id))             AS arm,
    argMax(f.baseline_arm, (f.ts, f.event_id))    AS baseline_arm,
    argMax(f.subject_level, (f.ts, f.event_id))   AS subject_level,
    argMax(f.subject_key, (f.ts, f.event_id))     AS subject_key,
    argMax(f.workload, (f.ts, f.event_id))        AS workload,
    argMax(f.stratum, (f.ts, f.event_id))         AS stratum,
    argMax(f.metric, (f.ts, f.event_id))          AS metric,
    argMax(f.unit, (f.ts, f.event_id))            AS unit,
    argMax(f.estimate, (f.ts, f.event_id))        AS estimate,
    argMax(f.ci_lo, (f.ts, f.event_id))           AS ci_lo,
    argMax(f.ci_hi, (f.ts, f.event_id))           AS ci_hi,
    argMax(f.n, (f.ts, f.event_id))               AS n,
    argMax(f.floor_pct, (f.ts, f.event_id))       AS floor_pct,
    argMax(f.verdict, (f.ts, f.event_id))         AS verdict,
    argMax(f.gain_evidence, (f.ts, f.event_id))   AS gain_evidence,
    argMax(f.demoted_by, (f.ts, f.event_id))      AS demoted_by,
    argMax(f.gates, (f.ts, f.event_id))           AS gates,
    argMax(f.policy_version, (f.ts, f.event_id))  AS policy_version,
    argMax(f.state, (f.ts, f.event_id))           AS state,
    argMax(f.summary, (f.ts, f.event_id))         AS summary,
    argMax(f.props, (f.ts, f.event_id))           AS props,
    max(f.ts)                                     AS updated_at
FROM experiment_findings AS f
GROUP BY f.finding_id;

-- Per candidate arm and metric: its newest published effect finding, beside the newest
-- stopping-rule decision (which exists before any finding is published). Finding columns are
-- NULL when there is none, not 0.
CREATE VIEW IF NOT EXISTS v_arm_verdicts AS
SELECT
    d.experiment_id                               AS experiment_id,
    d.arm                                         AS arm,
    d.baseline_arm                                AS baseline_arm,
    d.metric                                      AS metric,
    d.n                                           AS n,
    d.effect_pct                                  AS last_effect_pct,
    d.ci_lo_pct                                   AS last_ci_lo_pct,
    d.ci_hi_pct                                   AS last_ci_hi_pct,
    d.e_value                                     AS last_e_value,
    d.decision                                    AS decision,
    d.final                                       AS final,
    d.ts                                          AS decided_at,
    f.finding_id                                  AS finding_id,
    f.verdict                                     AS verdict,
    f.gain_evidence                               AS gain_evidence,
    f.estimate                                    AS effect_pct,
    f.ci_lo                                       AS ci_lo,
    f.ci_hi                                       AS ci_hi,
    f.demoted_by                                  AS demoted_by
FROM
(
    SELECT x.experiment_id AS experiment_id, x.arm AS arm, x.metric AS metric,
           argMax(x.baseline_arm, (x.ts, x.decision_id)) AS baseline_arm,
           argMax(x.n, (x.ts, x.decision_id))            AS n,
           argMax(x.effect_pct, (x.ts, x.decision_id))   AS effect_pct,
           argMax(x.ci_lo_pct, (x.ts, x.decision_id))    AS ci_lo_pct,
           argMax(x.ci_hi_pct, (x.ts, x.decision_id))    AS ci_hi_pct,
           argMax(x.e_value, (x.ts, x.decision_id))      AS e_value,
           argMax(x.decision, (x.ts, x.decision_id))     AS decision,
           argMax(x.final, (x.ts, x.decision_id))        AS final,
           max(x.ts)                                     AS ts
    FROM experiment_decisions AS x
    GROUP BY x.experiment_id, x.arm, x.metric
) AS d
LEFT JOIN
(
    SELECT y.experiment_id AS experiment_id, y.arm AS arm, y.metric AS metric,
           argMax(y.finding_id, y.updated_at)    AS finding_id,
           argMax(y.verdict, y.updated_at)       AS verdict,
           argMax(y.gain_evidence, y.updated_at) AS gain_evidence,
           argMax(y.estimate, y.updated_at)      AS estimate,
           argMax(y.ci_lo, y.updated_at)         AS ci_lo,
           argMax(y.ci_hi, y.updated_at)         AS ci_hi,
           argMax(y.demoted_by, y.updated_at)    AS demoted_by
    FROM v_experiment_findings AS y
    WHERE y.kind = 'effect' AND y.state = 'published'
    GROUP BY y.experiment_id, y.arm, y.metric
) AS f USING (experiment_id, arm, metric)
SETTINGS join_use_nulls = 1;

-- A/A noise floor per (workload, metric, stratum). `measured` pools the paired log differences
-- of every experiment whose two arms share a knob_sha; `reported` is a noise_floor finding from
-- an offline study. run_sd_pct is the sd of ONE run: sd(paired diff) / sqrt(2).
CREATE VIEW IF NOT EXISTS v_noise_floors AS
SELECT
    'measured'                                    AS source,
    p.workload                                    AS workload,
    p.metric                                      AS metric,
    p.stratum                                     AS stratum,
    count()                                       AS n_pairs,
    uniqExact(p.experiment_id)                    AS experiments,
    stddevSamp(p.d)                               AS sd_log_diff,
    100 * (exp(stddevSamp(p.d) / sqrt(2)) - 1)    AS run_sd_pct,
    100 * (exp(avg(p.d)) - 1)                     AS bias_pct,
    max(p.ts)                                     AS last_ts
FROM
(
    SELECT b.experiment_id AS experiment_id, b.workload AS workload, b.metric AS metric,
           b.stratum AS stratum, log(c.value / b.value) AS d, greatest(b.ts, c.ts) AS ts
    FROM v_experiment_samples AS b
    INNER JOIN v_experiment_samples AS c
        ON c.experiment_id = b.experiment_id AND c.trial = b.trial AND c.metric = b.metric
       AND c.warmup = b.warmup
    INNER JOIN v_experiment_arms AS ab ON ab.experiment_id = b.experiment_id AND ab.arm = b.arm
    INNER JOIN v_experiment_arms AS ac ON ac.experiment_id = c.experiment_id AND ac.arm = c.arm
    WHERE ab.role = 'baseline' AND ac.role = 'candidate' AND ab.knob_sha = ac.knob_sha
      AND NOT b.warmup AND b.status = 'ok' AND c.status = 'ok' AND b.value > 0 AND c.value > 0
) AS p
GROUP BY p.workload, p.metric, p.stratum
UNION ALL
SELECT
    'reported', f.workload, f.metric, f.stratum, toUInt64(f.n), toUInt64(1),
    nan, f.estimate, nan, f.updated_at
FROM v_experiment_findings AS f
WHERE f.kind = 'noise_floor' AND f.state = 'published';

-- One row per experiment: state, budget against spend, and how far its samples got.
CREATE VIEW IF NOT EXISTS v_experiment_progress AS
SELECT
    e.experiment_id                               AS experiment_id,
    e.parent_id                                   AS parent_id,
    e.name                                        AS name,
    e.kind                                        AS kind,
    e.state                                       AS state,
    e.budget                                      AS budget,
    e.spent                                       AS spent,
    if(e.budget['card_min'] > 0, e.spent['card_min'] / e.budget['card_min'], nan) AS card_min_used,
    ifNull(r.runs, 0)                             AS runs,
    ifNull(r.runs_failed, 0)                      AS runs_failed,
    ifNull(r.trials, 0)                           AS trials,
    ifNull(r.wall_s, 0)                           AS wall_s,
    ifNull(r.last_sample_at, e.updated_at)        AS last_sample_at,
    ifNull(d.decision, '')                        AS last_decision,
    ifNull(fc.findings, 0)                        AS findings,
    e.updated_at                                  AS updated_at
FROM v_experiments AS e
LEFT JOIN
(
    SELECT u.experiment_id                             AS experiment_id,
           count()                                     AS runs,
           countIf(u.status NOT IN ('ok', 'void'))     AS runs_failed,
           uniqExactIf(u.trial, NOT u.warmup)          AS trials,
           sum(u.wall_s)                               AS wall_s,
           max(u.ts)                                   AS last_sample_at
    FROM
    (
        -- One row per run, not per metric: a run's wall_s repeats on each of its metrics.
        SELECT v.experiment_id AS experiment_id, v.arm AS arm, v.trial AS trial, v.position AS position,
               v.warmup AS warmup, any(v.status) AS status, max(v.wall_s) AS wall_s, max(v.ts) AS ts
        FROM v_experiment_samples AS v
        GROUP BY v.experiment_id, v.arm, v.trial, v.position, v.warmup
    ) AS u
    GROUP BY u.experiment_id
) AS r ON r.experiment_id = e.experiment_id
LEFT JOIN
(
    SELECT z.experiment_id AS experiment_id, argMax(z.decision, (z.ts, z.decision_id)) AS decision
    FROM experiment_decisions AS z
    GROUP BY z.experiment_id
) AS d ON d.experiment_id = e.experiment_id
LEFT JOIN
(
    SELECT w.experiment_id AS experiment_id, count() AS findings
    FROM v_experiment_findings AS w
    WHERE w.state = 'published'
    GROUP BY w.experiment_id
) AS fc ON fc.experiment_id = e.experiment_id;

-- What was already tried for a subject, one row per (experiment, arm): its knobs and best
-- published finding (NULL when none), so an opportunity is not re-run blind. Filter on subject_key.
CREATE VIEW IF NOT EXISTS v_subject_tried AS
SELECT
    e.subject_level                               AS subject_level,
    e.subject_key                                 AS subject_key,
    e.workload                                    AS workload,
    e.experiment_id                               AS experiment_id,
    e.name                                        AS experiment,
    e.kind                                        AS experiment_kind,
    e.state                                       AS experiment_state,
    a.arm                                         AS arm,
    a.role                                        AS role,
    a.knobs                                       AS knobs,
    a.knob_sha                                    AS knob_sha,
    a.change_class                                AS change_class,
    f.metric                                      AS metric,
    f.verdict                                     AS verdict,
    f.gain_evidence                               AS gain_evidence,
    f.estimate                                    AS effect_pct,
    f.ci_lo                                       AS ci_lo,
    f.ci_hi                                       AS ci_hi,
    e.updated_at                                  AS updated_at
FROM v_experiments AS e
INNER JOIN v_experiment_arms AS a ON a.experiment_id = e.experiment_id
LEFT JOIN
(
    -- The strongest evidence wins, then the newest.
    SELECT t.experiment_id AS experiment_id, t.arm AS arm,
           argMax(t.metric, (t.gain_evidence, t.updated_at))   AS metric,
           argMax(t.verdict, (t.gain_evidence, t.updated_at))  AS verdict,
           max(t.gain_evidence)                                AS gain_evidence,
           argMax(t.estimate, (t.gain_evidence, t.updated_at)) AS estimate,
           argMax(t.ci_lo, (t.gain_evidence, t.updated_at))    AS ci_lo,
           argMax(t.ci_hi, (t.gain_evidence, t.updated_at))    AS ci_hi
    FROM v_experiment_findings AS t
    WHERE t.state = 'published' AND t.kind IN ('effect', 'calibration', 'feasibility')
    GROUP BY t.experiment_id, t.arm
) AS f ON f.experiment_id = a.experiment_id AND f.arm = a.arm
SETTINGS join_use_nulls = 1;
