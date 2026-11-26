# ch_mgmt schema

DDL for the `ch_mgmt` database behind the dashboard's ClickHouse login self-service. It is a
separate database from `spyre_v2`, so it has its own directory and is never applied by a plain
`apply_schema` run.

| file | declares |
|---|---|
| `10-login-requests.sql` | `login_requests`: one row per login, latest `version` wins (read with `FINAL`) |
| `20-login-events.sql` | `login_events`: append-only lifecycle log |
| `roles/chlogins-roles.sql` | `dashboard_chlogins_writer` (SELECT, INSERT) and `dashboard_chlogins_reader` (SELECT); run by the admin, not by the applier |

No password or digest is stored in either table.

`version` has no default: the writer must supply a strictly increasing value per login (two rows
with the same `version` leave the winner undefined, and clock skew can make an older state win).

The `status` and `tier` CHECKs fire only on insert, so a new status or tier value needs a
migration (`DROP CONSTRAINT` + `ADD CONSTRAINT`) under `migrations/` before the writer sends it.

## Apply

The database must exist first (created by the ClickHouse admin). The schema login needs
`CREATE TABLE` on `ch_mgmt.*`, `CREATE`/`SELECT`/`INSERT` on `ch_mgmt.schema_migrations` (the
applier's ledger), and `SELECT` on `system.tables`.

```bash
python -m spyre_clickhouse_ingest.apply_schema --schema-dir schema-ch-mgmt --database ch_mgmt --check
python -m spyre_clickhouse_ingest.apply_schema --schema-dir schema-ch-mgmt --database ch_mgmt
```

With `CLICKHOUSE_V2_HOST` set, add `ch_mgmt` to `CH_V2_DATABASES`. Without it the schema-role
connection is a `RoutedClient` that sends the DDL to the legacy (v1) host, not the v2 host.

## Roles

Both roles can read `grants` and `reason` for every login, so grant them only to the dashboard
logins. A role is inactive until it is in the user's `DEFAULT ROLE` list.
