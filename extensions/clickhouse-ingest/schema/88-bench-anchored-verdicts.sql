-- Anchored verdicts: a second, fixed comparison per series beside the rolling verdict of 85.
-- The rolling baseline (previous 20 eligible runs) absorbs a persistent shift after about 20 runs and the
-- verdict goes quiet. The anchor is an accepted baseline that does not move: a current reference version that
-- declares itself an anchor (artifact prop ref.anchor = '1'; a reference made from an accepted run also carries
-- ref.anchor.from_run_id and ref.anchor.as_of, the accepted run's time). Its value rows are ordinary
-- run.ref_only rows with the series' own component, benchmark_id and backend; run prop anchor.source_job
-- optionally narrows it to one producing job ('' = every job), and anchor.as_of overrides the artifact's
-- time for that row (a value carried over from an earlier version keeps its own; rows of one series in one version
-- are pooled and take the latest). The reference artifact's arch is folded like 60 folds a run's. "Current" is the reference tag slot of
-- v_bench_references, so accepting a new normal is a new reference version and a tag move, nothing else.
-- Only runs after the anchor's as_of (row, else artifact, else its registration time) are compared; earlier runs are `no_anchor`.
-- Depends on 20, 30, 49, 50, 60, 62, 83, 85, 86.

-- Per-run, per-series samples with the same exclusions as v_benchmark_metric_verdicts (invalidated samples dropped;
-- suspect or off-purpose runs ineligible). The eligibility rules are repeated from 85 so that view stays as it is.
CREATE VIEW IF NOT EXISTS v_benchmark_series_runs AS
SELECT e.run_id AS run_id, e.benchmark_id AS benchmark_id, e.component AS component,
       e.backend AS backend, e.arch AS arch, e.source_job AS source_job,
       any(e.name) AS name, any(e.input_shapes) AS input_shapes, min(e.run_ts) AS run_ts,
       s.1 AS metric, groupArray(s.2) AS cur,
       max(e.run_props['run.purpose'] != ''
           AND lower(e.run_props['run.purpose']) NOT IN (SELECT purpose FROM benchmark_baseline_purposes)) AS off_purpose,
       max(arrayExists(h -> tupleElement(h, 3) = 'suspect'
            AND (tupleElement(h, 1) = toUUID('00000000-0000-0000-0000-000000000000') OR tupleElement(h, 1) = e.benchmark_id)
            AND (tupleElement(h, 2) = -1 OR tupleElement(h, 2) = toInt32(toUInt16OrZero(e.run_props['run.repeat']))),
            i.hits)) AS suspect,
       NOT (off_purpose OR suspect) AS eligible
FROM v_benchmark_results_enriched AS e
ARRAY JOIN arrayFlatten(arrayMap(kv -> arrayMap(v -> (kv.1, v), kv.2),
           CAST(e.samples, 'Array(Tuple(String, Array(Float64)))'))) AS s
LEFT ANY JOIN
(
    SELECT x.run_id AS run_id,
           groupArray((ifNull(x.benchmark_id, toUUID('00000000-0000-0000-0000-000000000000')),
                       ifNull(toInt32(x.repeat), -1), x.state)) AS hits
    FROM v_invalidated_runs AS x
    WHERE x.run_id IS NOT NULL AND x.state IN ('invalidated', 'suspect')
    GROUP BY x.run_id
) AS i ON i.run_id = e.run_id
WHERE s.2 != 0
  AND e.record_type NOT IN (SELECT record_type FROM benchmark_excluded_record_types)
  AND NOT arrayExists(h -> tupleElement(h, 3) = 'invalidated'
        AND (tupleElement(h, 1) = toUUID('00000000-0000-0000-0000-000000000000') OR tupleElement(h, 1) = e.benchmark_id)
        AND (tupleElement(h, 2) = -1 OR tupleElement(h, 2) = toInt32(toUInt16OrZero(e.run_props['run.repeat']))),
        i.hits)
GROUP BY run_id, benchmark_id, component, backend, arch, source_job, metric;

-- The current accepted anchor per series (component, benchmark_id, backend, arch, source_job, metric); source_job
-- '' applies to every job. Two current anchors for one series: the newest registered wins.
CREATE VIEW IF NOT EXISTS v_benchmark_anchor AS
SELECT
    component, benchmark_id, backend, arch, source_job, metric,
    tupleElement(t, 1)                          AS anchor_artifact_id,
    tupleElement(t, 2)                          AS ref_key,
    tupleElement(t, 3)                          AS version,
    tupleElement(t, 4)                          AS anchor_from,
    tupleElement(t, 5)                          AS accepted_by,
    tupleElement(t, 6)                          AS accepted_at,
    tupleElement(t, 7)                          AS from_run_id,
    tupleElement(t, 8)                          AS registered_at,
    length(tupleElement(t, 9))                  AS n_samples,
    arrayReduce('median', tupleElement(t, 9))   AS anchor_value,
    tupleElement(t, 9)                          AS anchor_samples,
    arrayReduce('median', arrayMap(x -> abs(x - arrayReduce('median', tupleElement(t, 9))), tupleElement(t, 9))) AS anchor_mad
FROM
(
    SELECT component, benchmark_id, backend, arch, source_job, metric,
           argMax(tuple(artifact_id, ref_key, version, anchor_from, approved_by, approved_at, from_run_id, registered_at, samples),
                  (registered_at, artifact_id)) AS t
    FROM
    (
        SELECT br.component AS component, br.benchmark_id AS benchmark_id, br.backend AS backend,
               if(r.arch IN ('amd64', 'x86', 'x86-64'), 'x86_64', r.arch) AS arch,
               br.props['anchor.source_job'] AS source_job, m.1 AS metric,
               r.artifact_id AS artifact_id, r.ref_key AS ref_key, r.version AS version,
               max(coalesce(parseDateTimeBestEffortOrNull(br.props['anchor.as_of'], 'UTC'),
                            parseDateTimeBestEffortOrNull(a.props['ref.anchor.as_of'], 'UTC'), r.registered_at)) AS anchor_from,
               r.approved_by AS approved_by, r.approved_at AS approved_at,
               a.props['ref.anchor.from_run_id'] AS from_run_id, r.registered_at AS registered_at,
               groupArray(m.2) AS samples
        FROM v_bench_references AS r
        INNER JOIN v_artifacts AS a ON a.artifact_id = r.artifact_id
        INNER JOIN
        (
            SELECT * FROM benchmark_runs
            WHERE props['run.ref_only'] = '1'
              AND run_id IN (SELECT run_id FROM v_bench_references WHERE state = 'current')
        ) AS br ON br.run_id = r.run_id
        ARRAY JOIN arrayFlatten(arrayMap(kv -> arrayMap(v -> (kv.1, v), kv.2),
                   CAST(br.measurements, 'Array(Tuple(String, Array(Float64)))'))) AS m
        WHERE r.state = 'current' AND a.props['ref.anchor'] = '1' AND m.2 != 0
        GROUP BY component, benchmark_id, backend, arch, source_job, metric, artifact_id, ref_key, version,
                 approved_by, approved_at, from_run_id, registered_at
    )
    GROUP BY component, benchmark_id, backend, arch, source_job, metric
);

-- One row per run x series x metric, judged against the anchor with the rolling view's noise rule: floor =
-- max(policy floor, 2 x 1.4826 x MAD% of the anchor's samples). One run over is `suspected` (single_run);
-- `regressed` needs the previous eligible run over as well. Suspect or off-purpose runs are capped at `suspected`.
-- A run before the anchor, in a series with none, or against an anchor of fewer than 3 samples (no usable noise
-- estimate) is `no_anchor`; a job-specific anchor applies only to runs after its own time, earlier runs fall back to
-- the job-less one. `reason` is anchor_regressed, anchor_improved, anchor_unchanged or no_anchor, else the rolling
-- view's cap/single-run reasons. While a run is over the floor, since_ts and n_exceeding_runs describe its unbroken
-- streak of eligible runs over the floor (ineligible runs neither extend nor break it; NULL / 0 until the first
-- eligible one) and days_exceeding is the calendar days from since_ts. Columns of a `no_anchor` row are defaults.
CREATE VIEW IF NOT EXISTS v_benchmark_anchored_verdicts AS
SELECT
    * EXCEPT (exceeds, prev_exceeds, cap, pos, streak, raw_since),
    multiIf(NOT has_anchor,           'no_anchor',
            exceeds AND cap != '',    'suspected',
            exceeds AND prev_exceeds, 'regressed',
            exceeds,                  'suspected',
            worse_pct < -floor_pct,   'improved',
                                      'unchanged') AS status,
    multiIf(status = 'no_anchor',                      'no_anchor',
            status = 'suspected' AND cap = 'purpose',  'purpose_not_baseline',
            status = 'suspected' AND cap = 'suspect',  'suspect_data',
            status = 'suspected',                      'single_run',
            status = 'regressed',                      'anchor_regressed',
            status = 'improved',                       'anchor_improved',
                                                       'anchor_unchanged') AS reason,
    if(exceeds, raw_since, NULL) AS since_ts,
    if(exceeds, countIf(exceeds AND eligible) OVER (
           PARTITION BY component, benchmark_id, backend, arch, source_job, metric, streak
           ORDER BY run_ts, run_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0) AS n_exceeding_runs,
    if(since_ts IS NULL, 0, greatest(0, dateDiff('day', since_ts, run_ts))) AS days_exceeding
FROM (
    SELECT *,
        minIf(toNullable(run_ts), exceeds AND eligible) OVER (
            PARTITION BY component, benchmark_id, backend, arch, source_job, metric, streak
            ORDER BY run_ts, run_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS raw_since
    FROM (
    SELECT *,
        anyIf(exceeds, eligible) OVER (
            PARTITION BY component, benchmark_id, backend, arch, source_job, metric
            ORDER BY pos RANGE BETWEEN 1 PRECEDING AND 1 PRECEDING) AS prev_exceeds,
        sumIf(1, eligible AND NOT exceeds) OVER (
            PARTITION BY component, benchmark_id, backend, arch, source_job, metric
            ORDER BY run_ts, run_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS streak
    FROM (
        SELECT
            run_id, benchmark_id, component, backend, arch, source_job, name, input_shapes,
            run_ts, metric, tier, higher_is_better, eligible, pos, has_anchor,
            multiIf(off_purpose, 'purpose', suspect, 'suspect', '') AS cap,
            if(has_anchor, anchor_artifact_id, toUUID('00000000-0000-0000-0000-000000000000')) AS anchor_artifact_id,
            if(has_anchor, ref_key, '') AS ref_key,
            if(has_anchor, version, '') AS version,
            if(has_anchor, anchor_from, toDateTime(0, 'UTC')) AS anchor_from,
            if(has_anchor, accepted_by, '') AS accepted_by,
            if(has_anchor, accepted_at, '') AS accepted_at,
            if(has_anchor, from_run_id, '') AS from_run_id,
            if(has_anchor, round(anchor_med, 4), 0) AS anchor_value,
            length(cur) AS n_current,
            round(cur_med, 4) AS current_value,
            if(has_anchor, round((cur_med - anchor_med) / abs(anchor_med) * 100, 1), 0) AS delta_pct,
            if(has_anchor, round(if(higher_is_better, -1, 1) * delta_pct, 1), 0) AS worse_pct,
            if(has_anchor, round(greatest(policy_floor, 200 * 1.4826 * anchor_mad / abs(anchor_med)), 1), 0) AS floor_pct,
            has_anchor AND worse_pct > floor_pct AS exceeds
        FROM (
            SELECT q.*,
                if(p.has_policy, p.tier, 'informational') AS tier,
                if(p.has_policy, p.higher_is_better,
                   match(metric, 'throughput|_per_second$') OR metric = 'pt_util_percent') AS higher_is_better,
                if(p.has_policy, p.floor_pct, 5) AS policy_floor,
                arrayReduce('median', cur) AS cur_med
            FROM (
                SELECT
                    s.run_id AS run_id, s.benchmark_id AS benchmark_id, s.component AS component, s.backend AS backend,
                    s.arch AS arch, s.source_job AS source_job, s.name AS name, s.input_shapes AS input_shapes,
                    s.run_ts AS run_ts, s.metric AS metric, s.cur AS cur,
                    s.off_purpose AS off_purpose, s.suspect AS suspect, s.eligible AS eligible,
                    sum(s.eligible) OVER (
                        PARTITION BY s.component, s.benchmark_id, s.backend, s.arch, s.source_job, s.metric
                        ORDER BY s.run_ts, s.run_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS pos,
                    -- A job-specific anchor beats the job-less one for the runs after its own time.
                    ifNull(sp.n_samples, 0) >= 3 AND ifNull(sp.anchor_value, 0) != 0 AND s.run_ts > ifNull(sp.anchor_from, toDateTime('2100-01-01 00:00:00', 'UTC')) AS use_sp,
                    ifNull(ge.n_samples, 0) >= 3 AND ifNull(ge.anchor_value, 0) != 0 AND s.run_ts > ifNull(ge.anchor_from, toDateTime('2100-01-01 00:00:00', 'UTC')) AS use_ge,
                    use_sp OR use_ge AS has_anchor,
                    if(use_sp, sp.anchor_artifact_id, ge.anchor_artifact_id) AS anchor_artifact_id,
                    if(use_sp, sp.ref_key, ge.ref_key) AS ref_key,
                    if(use_sp, sp.version, ge.version) AS version,
                    if(use_sp, sp.anchor_from, ge.anchor_from) AS anchor_from,
                    if(use_sp, sp.accepted_by, ge.accepted_by) AS accepted_by,
                    if(use_sp, sp.accepted_at, ge.accepted_at) AS accepted_at,
                    if(use_sp, sp.from_run_id, ge.from_run_id) AS from_run_id,
                    if(use_sp, sp.anchor_value, ge.anchor_value) AS anchor_med,
                    if(use_sp, sp.anchor_mad, ge.anchor_mad) AS anchor_mad
                FROM v_benchmark_series_runs AS s
                LEFT ANY JOIN v_benchmark_anchor AS sp
                    ON sp.component = s.component AND sp.benchmark_id = s.benchmark_id AND sp.backend = s.backend
                   AND sp.arch = s.arch AND sp.source_job = s.source_job AND sp.metric = s.metric
                LEFT ANY JOIN v_benchmark_anchor AS ge
                    ON ge.component = s.component AND ge.benchmark_id = s.benchmark_id AND ge.backend = s.backend
                   AND ge.arch = s.arch AND ge.source_job = '' AND ge.metric = s.metric
            ) AS q
            LEFT JOIN (SELECT *, true AS has_policy FROM benchmark_metric_policy) AS p USING (metric)
        )
    )
    )
);
