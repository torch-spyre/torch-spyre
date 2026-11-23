-- The only deleter of chat_turns: run daily as a login with ALTER DELETE on it and SELECT on the retention objects; the dashboard never deletes.
-- Per actor in v_chat_turn_retention: turns older than retention_days (0 = forever) and every turn up to the latest purge request.
-- Scoped by (turn_id, actor): feedback carries a client-supplied turn_id, so a bare turn_id would let one person delete another's turn.
-- A mutation, not DELETE FROM: that needs ALTER UPDATE(_row_exists) under the default lightweight_delete_mode.
ALTER TABLE chat_turns DELETE WHERE (turn_id, actor) IN (
    SELECT t.turn_id, t.actor
    FROM chat_turns AS t
    INNER JOIN v_chat_turn_retention AS r ON t.actor = r.actor
    WHERE t.ts <= r.purge_before
       OR (r.retention_days > 0 AND t.ts < now64(3) - toIntervalDay(r.retention_days))
) SETTINGS mutations_sync = 2;
