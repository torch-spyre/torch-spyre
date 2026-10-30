-- The only deleter of chat_turns: run on a schedule (daily is plenty) as a login that may DELETE on it; the dashboard never does.
-- Removes, for each actor in v_chat_turn_retention, turns older than their retention_days (0 = forever deletes nothing) and
-- every turn up to their latest purge request. Feedback rows go with their turn. Actors with no row keep everything.
DELETE FROM chat_turns WHERE turn_id IN (
    SELECT t.turn_id
    FROM chat_turns AS t
    INNER JOIN v_chat_turn_retention AS r ON t.actor = r.actor
    WHERE t.ts <= r.purge_before
       OR (r.retention_days > 0 AND t.ts < now64(3) - toIntervalDay(r.retention_days))
);
