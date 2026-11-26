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

## Apply

The database must exist first (created by the ClickHouse admin), and the schema login needs
`CREATE TABLE` on it (the applier also creates `schema_migrations` there).

```bash
python -m spyre_clickhouse_ingest.apply_schema --schema-dir schema-ch-mgmt --database ch_mgmt --check
python -m spyre_clickhouse_ingest.apply_schema --schema-dir schema-ch-mgmt --database ch_mgmt
```

With `CLICKHOUSE_V2_HOST` set, add `ch_mgmt` to `CH_V2_DATABASES` so the applier reaches the same
host as `spyre_v2`.
