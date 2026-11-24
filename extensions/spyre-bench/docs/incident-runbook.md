# Incident runbook: invalidate bad results, check the effect, restore

Use this when stored results cannot be trusted: a card ran hot, a node was misconfigured, a harness bug skewed a run. The
verdicts and baselines stop using the affected measurements; the rows stay in the store and the decision is logged, so it
can be reversed.

## What an invalidation is

An event on an `invalidation_id` with a state: `suspect`, `invalidated` or `restored`. Events are append-only; the latest
state of an id wins. Readers apply it at once:

| State | Verdict output | Baselines |
|---|---|---|
| `suspect` | still shown, status capped at `suspected` | never part of a baseline |
| `invalidated` | dropped | dropped |
| `restored` | as if it never happened | back in |

A reference or a retracted upload is a different mechanism; do not use this for those.

## Scopes

| Flags | Covers |
|---|---|
| `--run RUN` | every benchmark and repeat of one run |
| `--run RUN --benchmark BID [--repeat N]` | one benchmark of a run, optionally one repeat |
| `--node NODE [--from T --to T]` | every repeat that executed on the node |
| `--device SERIAL [--from T --to T]` | every repeat that used the card |
| `--placement ID [--from T --to T]` | every repeat with that exact hardware placement |

Windows bound the repeat start time (`--from` and `--to` both inclusive, ISO-8601, UTC if no zone). A repeat whose start
time was not recorded matches any window, so a window can only widen on missing data, never narrow. A node, device or
placement scope needs a placement record for the repeat (the run recorded where it ran); a run without one is not
matched, find those by run id.

## 1. Find what is affected

The preview is the finder. Run any scope without `--apply`:

```bash
spyre-bench invalidate --device SN0042 --from 2026-11-01T00:00:00Z \
    --reason-code faulty_hardware --reason "card SN0042 throttled"
```

It prints the runs, series (component, benchmark, backend, arch, source job), benchmarks and repeats the scope covers,
and the **verdict changes**: each affected series' verdict rows before and after (`regressed -> absent`, ...). Nothing is
written. Narrow with `--from/--to` until the list is the incident and nothing else. To look wider first, vary the scope:
the same device without a window, then the node.

Cross-check by hand if needed (read-only SQL on the v2 database):

```sql
SELECT run_id, repeat, started_at FROM v_bench_scope_hits(
    scope_kind = 'device', device_serial = 'SN0042', run_id = '', benchmark_id = '', repeat = -1,
    node = '', placement_id = '', window_from = '', window_to = '')
```

## 2. Prefer `suspect` first when unsure

`--state suspect` keeps the data visible but out of every baseline and caps verdicts at `suspected`. Escalate later with
`spyre-bench invalidate set ID --state invalidated ...` or clear it with `restore`.

## 3. Record it

```bash
spyre-bench invalidate --device SN0042 --from 2026-11-01T00:00:00Z \
    --reason-code faulty_hardware --reason "card SN0042 throttled" --evidence-url https://example.org/incident/17 --apply
```

- `--reason-code` is one of `faulty_hardware`, `measurement_bug`, `bad_config`, `under_investigation`, `other`;
  `--reason` is required; both are kept in the history.
- A scope that covers more than 50 runs (`BENCH_INVALIDATION_CONFIRM_ABOVE`) is refused until you add `--confirm`. The
  preview prints a token; `--confirm TOKEN` ties the confirmation to the runs you saw, and the write is refused if the
  set changed since.
- `--apply` writes a `bench-invalidation` bundle (under `$BENCH_OUT/invalidations/`) and ingests it. `--bundle-only`
  stops after the bundle, for a machine that cannot reach the store: hand the directory to
  `python -m spyre_clickhouse_ingest results --from-bundle DIR` where it can.
- Repeating the same command with the same `--invalidation-id` is a no-op, so a retry after a timeout is safe.
- The actor recorded is the login name (`--actor` / `BENCH_ACTOR`), never read from the bundle.

## 4. Verify

```bash
spyre-bench invalidate show ID          # events, who and why, the runs covered now
spyre-bench invalidate list --state invalidated
spyre-bench verdict --run-id NEXT_RUN   # the baseline no longer lists the affected runs
```

In the JSON verdict, `baseline_run_ids` of the next healthy run must not contain the invalidated runs, and `n_baseline`
refills from older eligible runs. If it does not, the scope missed them: check the placement records.

## 5. Restore

```bash
spyre-bench invalidate restore ID --reason-code other --reason "card replaced and re-qualified" --apply
```

The preview shows what comes back. Restore a wrong or too-broad invalidation, or narrow it: restore it, then create a
tighter one. A restored id can be re-opened with `set --state suspect|invalidated`.

## Things to know

- Events of one id are ordered by time; the writer moves a new event to at least 1 ms after the id's latest, so quick
  successive events keep their order. Do not write `result_invalidations` rows by other means.
- `plan` does not read invalidations from the store yet: after an incident, plan the affected cases with reuse `never` so
  an invalidated run is not picked as a cached result.
- Preview cost: the verdict diff re-evaluates the verdict view for the affected series; on a very large store it can take
  minutes. `--json` has the full lists.
- Try an unfamiliar scope on a personal sandbox database first (set `BENCH_CLICKHOUSE_DB`); the verbs only read and
  write through the configured store.
