-- Scores built from feedback_signals. Votes are signals, not labels: a score ranks what to review and
-- can demote a signature, and a share of down-votes is not a precision. Constants, all proposals:
--   half-life 30 days (a signal of age t counts 0.5^(t/30d));
--   one actor's current signal of a kind on a target counts once, and an actor's weight is capped at 3;
--   up/affected add, down/not_affected subtract; miss and chip carry no sign and are counted apart.

-- The current signal per (target, signature, actor, axis). An axis is what a signal is about: up and
-- down are one axis (a vote), affected and not_affected are one (an outage report), a miss is its
-- own, and each chip reason is its own. An actor who votes down and later up has changed their
-- mind, so only the newest on the axis counts and the history stays in the table. signature_id is
-- in the key because one failure can match two signatures and be right about one and wrong about
-- the other; it is '' for every target that is not a signature match.
CREATE VIEW IF NOT EXISTS v_feedback_current AS
SELECT
    target_type, target_id, signature_id, actor,
    multiIf(signal IN ('up', 'down'), 'vote',
            signal IN ('affected', 'not_affected'), 'impact',
            signal = 'chip', concat('chip:', detail),
            signal)                       AS axis,
    argMax(signal, created_at)            AS cur_signal,
    argMax(source, created_at)            AS cur_source,
    argMax(detail, created_at)            AS cur_detail,
    argMax(weight_hint, created_at)       AS cur_weight_hint,
    argMax(signature_version, created_at) AS cur_signature_version,
    max(created_at)                       AS cur_at
FROM feedback_signals
GROUP BY target_type, target_id, signature_id, actor, axis;

CREATE VIEW IF NOT EXISTS v_feedback_scores AS
SELECT
    target_type, target_id, signature_id,
    uniqExact(actor)                                                  AS distinct_voters,
    uniqExactIf(actor, cur_signal IN ('up', 'affected'))                  AS ups,
    uniqExactIf(actor, cur_signal IN ('down', 'not_affected'))            AS downs,
    uniqExactIf(actor, cur_signal = 'miss')                               AS misses,
    round(sumIf(
        least(cur_weight_hint, 3) * exp2(-dateDiff('second', cur_at, now64(3)) / (30 * 86400.)),
        cur_signal IN ('up', 'affected')), 4)                             AS decayed_up,
    round(sumIf(
        least(cur_weight_hint, 3) * exp2(-dateDiff('second', cur_at, now64(3)) / (30 * 86400.)),
        cur_signal IN ('down', 'not_affected')), 4)                       AS decayed_down,
    round(decayed_up - decayed_down, 4)                               AS score,
    max(cur_at)                                                       AS last_signal_at
FROM v_feedback_current
GROUP BY target_type, target_id, signature_id;

-- Quality of each signature version from human verdicts on its matches (target_type signature_match).
-- Only explicit votes and reactions are verdicts; an implicit signal (an acknowledgement, a mute) is
-- counted apart and never enters the precision. A match two reviewers disagree on is `disputed` and
-- is left out of both counts. precision_lower is the Wilson 95% lower bound of the share of right
-- matches, and is what to trust; precision_raw sits beside it. A signature with no human verdict
-- is `unproven` and has no precision, however many matches it has.
CREATE VIEW IF NOT EXISTS v_signature_quality AS
WITH per_match AS
(
    SELECT
        signature_id, cur_signature_version AS sig_version, target_id,
        uniqExactIf(actor, cur_signal = 'up'   AND cur_source IN ('explicit', 'reaction')) AS human_up,
        uniqExactIf(actor, cur_signal = 'down' AND cur_source IN ('explicit', 'reaction')) AS human_down,
        groupUniqArrayIf(actor, cur_signal IN ('up', 'down') AND cur_source IN ('explicit', 'reaction')) AS who,
        countIf(cur_source = 'implicit' AND cur_signal IN ('up', 'down'))                  AS implicit_n,
        countIf(cur_signal = 'miss')                                                   AS miss_n,
        max(cur_at)                                                                    AS last_at
    FROM v_feedback_current
    WHERE target_type = 'signature_match' AND signature_id != ''
    GROUP BY signature_id, sig_version, target_id
)
SELECT
    signature_id,
    sig_version                                                      AS signature_version,
    countIf(human_up > 0 AND human_down = 0)                         AS tp_matches,
    countIf(human_down > 0 AND human_up = 0)                         AS fp_matches,
    countIf(human_up > 0 AND human_down > 0)                         AS disputed_matches,
    tp_matches + fp_matches                                          AS labelled_matches,
    length(arrayDistinct(arrayFlatten(groupArray(who))))             AS reviewers,
    sum(human_up + human_down)                                       AS human_verdicts,
    if(labelled_matches = 0, NULL, round(tp_matches / labelled_matches, 4)) AS precision_raw,
    if(labelled_matches = 0, NULL, round(
        (tp_matches / labelled_matches + 1.96 * 1.96 / (2 * labelled_matches)
         - 1.96 * sqrt((tp_matches / labelled_matches * (1 - tp_matches / labelled_matches)
                        + 1.96 * 1.96 / (4 * labelled_matches)) / labelled_matches))
        / (1 + 1.96 * 1.96 / labelled_matches), 4))                  AS precision_lower,
    multiIf(human_verdicts = 0, 'unproven', labelled_matches < 20, 'few_labels', 'measured') AS evidence,
    sum(implicit_n)                                                  AS implicit_signals,
    sum(miss_n)                                                      AS misses_reported,
    max(last_at)                                                     AS last_label_at
FROM per_match
GROUP BY signature_id, signature_version;
