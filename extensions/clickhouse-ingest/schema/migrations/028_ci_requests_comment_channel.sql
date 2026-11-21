-- RERUNNABLE
-- A PR comment (/spyre-test bench=, /bench) is a request channel. chk_channel and chk_request are dropped and
-- re-added in DDL order (state, channel, request), so a migrated table formats exactly like a fresh one.
ALTER TABLE ci_requests DROP CONSTRAINT IF EXISTS chk_channel;

ALTER TABLE ci_requests DROP CONSTRAINT IF EXISTS chk_request;

ALTER TABLE ci_requests ADD CONSTRAINT chk_channel CHECK channel IN ('ui', 'chat', 'mcp', 'api', 'comment');

ALTER TABLE ci_requests ADD CONSTRAINT chk_request CHECK request_id != toUUID('00000000-0000-0000-0000-000000000000');
