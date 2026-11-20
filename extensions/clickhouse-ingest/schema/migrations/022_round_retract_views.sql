-- v_result_uploads gains is_round and the row views gain the retracted-round filter; plain views
-- hold no data, so the two bases are dropped here and the applier recreates every view from schema/.
DROP VIEW IF EXISTS v_artifact_results_published;

DROP VIEW IF EXISTS v_result_uploads;
