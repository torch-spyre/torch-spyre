-- Grants for the dashboard's login self-service, run once by the ClickHouse admin after
-- `apply_schema --schema-dir schema-ch-mgmt --database ch_mgmt`. Not applied by the applier.
-- The writer never gets ALTER or DELETE: a login's current state is a newer row, not an update.
-- Name every role the user already holds in DEFAULT ROLE (SHOW GRANTS FOR <user>); the clause
-- replaces the list.
CREATE ROLE IF NOT EXISTS dashboard_chlogins_writer;
CREATE ROLE IF NOT EXISTS dashboard_chlogins_reader;

GRANT SELECT, INSERT ON ch_mgmt.login_requests TO dashboard_chlogins_writer;
GRANT SELECT, INSERT ON ch_mgmt.login_events TO dashboard_chlogins_writer;
GRANT SELECT ON ch_mgmt.login_requests TO dashboard_chlogins_reader;
GRANT SELECT ON ch_mgmt.login_events TO dashboard_chlogins_reader;

-- GRANT dashboard_chlogins_writer TO <dashboard write login>;
-- GRANT dashboard_chlogins_reader TO <dashboard read login>;
