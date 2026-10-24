-- Grants with their current state and use, and a daily usage rollup, for the grants page and Ask Spyre.

-- `active` is what a gateway may honour right now: not revoked, not past expires_at, not used up.
-- Uses count only events that succeeded. minIf over no rows is the epoch, so an unset milestone is
-- returned as NULL. Aggregate aliases are not source column names: such an alias is substituted
-- into the other aggregates (ILLEGAL_AGGREGATION).
CREATE VIEW IF NOT EXISTS v_delegation_grants_current AS
WITH ev AS
(
    SELECT
        grant_id,
        countIf(event = 'used' AND result = 'ok')   AS n_used,
        countIf(event = 'denied' OR result = 'denied') AS n_denied,
        minIf(ts, event = 'revoked')                AS first_revoked,
        maxIf(ts, event = 'used' AND result = 'ok') AS last_used,
        minIf(ts, event = 'expired')                AS first_expired
    FROM delegation_grant_events
    GROUP BY grant_id
)
SELECT
    g.grant_id AS grant_id, g.grantor AS grantor, g.grantee AS grantee, g.tools AS tools,
    g.scope_json AS scope_json, g.condition AS condition, g.max_uses AS max_uses,
    g.created_at AS created_at, g.created_by AS created_by, g.expires_at AS expires_at,
    g.consent_text_hash AS consent_text_hash, g.props AS props,
    e.n_used AS uses, e.n_denied AS denials,
    if(e.last_used = toDateTime64(0, 3, 'UTC'), NULL, e.last_used) AS last_used_at,
    if(e.first_revoked = toDateTime64(0, 3, 'UTC'), NULL, e.first_revoked) AS revoked_at,
    multiIf(
        e.first_revoked != toDateTime64(0, 3, 'UTC'),                      'revoked',
        g.expires_at <= now64(3) OR e.first_expired != toDateTime64(0, 3, 'UTC'), 'expired',
        g.max_uses > 0 AND e.n_used >= g.max_uses,                          'exhausted',
                                                                            'active'
    ) AS state,
    state = 'active' AS is_active
FROM delegation_grants AS g
LEFT JOIN ev AS e USING (grant_id);

-- What was done under grants, by day, grantee and tool; refusals are counted apart from good calls.
CREATE VIEW IF NOT EXISTS v_delegation_usage AS
SELECT
    toDate(ts)                      AS day,
    on_behalf_of                    AS grantor,
    actor                           AS agent,
    tool,
    countIf(event = 'used')         AS calls,
    countIf(event = 'used' AND result = 'ok') AS ok_calls,
    countIf(event = 'denied' OR result = 'denied') AS refused,
    countIf(result = 'error')       AS errors
FROM delegation_grant_events
WHERE event IN ('used', 'denied')
GROUP BY day, grantor, agent, tool;
