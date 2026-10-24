-- Credential health for the CI/CD maintenance page: each inventoried credential with its latest
-- probe, its latest human-entered owner details, and a single health class.

-- `expired` and `expiring` come from a probe's expires_at; `failing` wins over both, so a 401
-- reads red whatever the date says. A credential no probe has reached is `unprobed`.
CREATE VIEW IF NOT EXISTS v_credential_health AS
WITH probes AS
(
    SELECT
        system, scope, credential_id,
        argMax(status, ts)       AS probe_status,
        argMax(error_class, ts)  AS error_class,
        argMax(probe_site, ts)   AS probe_site,
        argMax(probe, ts)        AS probe,
        argMax(expires_at, ts)   AS expires_at,
        argMax(fingerprint, ts)  AS fingerprint,
        argMax(detail, ts)       AS detail,
        max(ts)                  AS last_probe_at,
        maxIf(ts, status = 'ok') AS last_ok_at
    FROM credential_probes
    WHERE ts >= now() - INTERVAL 90 DAY
    GROUP BY system, scope, credential_id
),
owners AS
(
    SELECT
        system, scope, credential_id,
        argMax(owner, updated_at)           AS owner,
        argMax(owner_email, updated_at)     AS owner_email,
        argMax(backup_owner, updated_at)    AS backup_owner,
        argMax(linked_identity, updated_at) AS linked_identity,
        argMax(linked_email, updated_at)    AS linked_email,
        argMax(issuer, updated_at)          AS issuer,
        argMax(rotation_days, updated_at)   AS rotation_days,
        argMax(last_rotated_at, updated_at) AS last_rotated_at,
        argMax(runbook_url, updated_at)     AS runbook_url,
        argMax(notes, updated_at)           AS notes,
        argMax(updated_by, updated_at)      AS owner_updated_by,
        max(updated_at)                     AS owner_updated_at
    FROM credential_owners
    GROUP BY system, scope, credential_id
)
SELECT
    i.system AS system, i.scope AS scope, i.credential_id AS credential_id,
    i.kind AS kind, i.description AS description,
    i.consumers AS consumers, i.state AS state,
    i.source_created_at AS source_created_at, i.source_updated_at AS source_updated_at,
    i.last_bound_at AS last_bound_at, i.observed_at AS observed_at,
    p.probe_site AS probe_site, p.probe AS probe,
    p.probe_status AS probe_status, p.error_class AS error_class,
    p.expires_at AS expires_at, p.fingerprint AS fingerprint, p.detail AS detail,
    p.last_probe_at AS last_probe_at, p.last_ok_at AS last_ok_at,
    o.owner AS owner, o.owner_email AS owner_email, o.backup_owner AS backup_owner,
    o.linked_identity AS linked_identity, o.linked_email AS linked_email, o.issuer AS issuer,
    o.rotation_days AS rotation_days, o.last_rotated_at AS last_rotated_at,
    o.runbook_url AS runbook_url, o.notes AS notes,
    o.owner_updated_by AS owner_updated_by, o.owner_updated_at AS owner_updated_at,
    dateDiff('day', now(), p.expires_at) AS days_to_expiry,
    multiIf(
        p.last_probe_at = toDateTime(0),                        'unprobed',
        p.probe_status = 'failing',                             'failing',
        p.expires_at IS NOT NULL AND p.expires_at <= now(),     'expired',
        p.expires_at IS NOT NULL AND p.expires_at <= now() + INTERVAL 30 DAY, 'expiring',
        p.probe_status = 'ok',                                  'ok',
                                                                'unknown'
    ) AS health
FROM credential_inventory AS i FINAL
LEFT JOIN probes AS p USING (system, scope, credential_id)
LEFT JOIN owners AS o USING (system, scope, credential_id)
WHERE i.state != 'gone';
