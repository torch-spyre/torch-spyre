# References

A **reference** is a comparison target: values (or output files) a run of ours is compared to, such as a CPU output
set, a figure from a model card, or a number measured elsewhere. It is unrelated to a run's "reference grade", it is
never a baseline, and nothing compared to it gates a build or a check. `spyre-bench reference` builds, checks and
seals the file, submits it for approval and compares a run with it; the engine writes no rows, an approval service
writes them once, at approval.

## Flows

**F3, from a run.** Run the benchmark as usual, then:

```bash
spyre-bench reference create --from-run results/bench --key perf.example.add.refsys --class perf \
    --label ref-sys-a --meta ref-meta.yaml --out ref/          # reference.json + attachments/
spyre-bench reference validate ref/
spyre-bench reference seal ref/ --out ref.tgz                  # a bench-reference bundle
spyre-bench reference submit ref.tgz --store clickhouse         # pending; an approver decides in the UI
spyre-bench reference compare --run results/bench --reference perf.example.add.refsys --store clickhouse
```

`create` reads the run's `result.json` files. `--meta` is a YAML or JSON file of reference descriptive fields (`model`, `dataset`,
`tools`, `system`, `comparability`, `producer`, `source_url`, `terms`, `as_of`) so the command line stays short; `--tool NAME=VERSION` and
`--comparability KEY=VALUE` add single entries. A YAML `--meta` file reads only `true` and `false` as booleans, so an
unquoted `yes`, `no`, `on` or `off` stays text. `--meta` cannot set `clean`, `warnings`, `pinning`, `values` or
`source_run`: those come from the run. A reference that fails validation is not written, and the message lists
every missing field.

**From an accepted run.** `spyre-bench accept` writes a `bench.reference-request/1` file. `create --from-request FILE`
turns it into a reference directory: each request value becomes a value row (metrics of one case and backend share a
row; carried anchored series are included), and the request's placeholder comparability is settled from the values:
`precision` from the cases' `dtype` or `precision` discriminator, the `shape.*` keys `n/a` when the benchmark has none.
Add `--run RESULTS_DIR` to take cleanliness, warnings and pinning from the accepted run's own results; without it the
reference is `clean: null` with the warning `clean_unknown`. `model.revision` defaults to `run-<id>` of the accepted
run and `tools` to `spyre-bench: accept`; `--meta`, `--tool` and `--comparability` override them. Each request value
must carry a `case` block (`component`, `name`, `tags`, `disc`, `disc_keys`) that reproduces its stored `benchmark_id`,
otherwise `create` refuses with the reason.

**F5, from numbers someone else measured.** Write `reference.json` by hand (`origin: uploaded` or `published`, with
`producer`, `as_of` and `comparability.caveats`; `published` also needs `source_url` and non-empty caveats), put any
output files under `attachments/`, then `validate`, `seal` and `submit` as above. The `case.benchmark_id` of every
value is recomputed from its component, name, tags and discriminators and must match.

## The file

`results/reference.json` in a bundle, `reference.json` in a reference directory; `apiVersion: bench/v1alpha1`,
`kind: Reference`.

```json
{
  "apiVersion": "bench/v1alpha1", "kind": "Reference",
  "key": "perf.example.add.refsys", "origin": "generated", "class": "perf",
  "label": "ref-sys-a", "terms": "internal-generated", "as_of": "2026-11-15T03:00:00Z",
  "tools": {"torch": "2.9.1"}, "model": {"repo": "org/model", "revision": "abc123"},
  "system": {"backend": "cpu"}, "source_run": {"run_id": "<uuid>"},
  "clean": true, "warnings": [], "pinning": "none",
  "comparability": {"harness": "tool@1", "harness_match": "yes", "precision": "fp32", "dataset_subset": "full",
                    "dataset_match": "same", "shape.batch": 1, "shape.input_len": 128, "shape.output_len": 64},
  "values": [{"case": {"component": "example", "name": "add", "tags": [], "disc": {"shape": "64"},
                       "benchmark_id": "<recomputed and checked>"},
              "backend": "cpu", "measurements": {"kernel.time.mean": [1.0, 1.1]}, "iterations": 2}],
  "assets": []
}
```

- `origin`: `generated` (ours), `uploaded` or `published` (someone else's). `class`: `perf` or `quality`.
- `label` is a public-safe slug (`[a-z0-9-]`, 3 to 41 characters). It may not contain a whole word `node`, `host` or `worker` (`host-1` is refused, `ghost-x` is fine) or a
  dot, nor match a site deny-list: a file of one case-insensitive regex per line (`#` comments and blank lines
  ignored) named by `--label-denylist FILE` on `create`, `validate`, `seal` and `submit`, else by
  `SPYRE_BENCH_LABEL_DENYLIST`. An unreadable file or an invalid regex exits 2. `terms` is
  required (`internal-generated` is fine for `generated`).
- `comparability` always carries `harness`, `harness_match` (`yes|partial|no|unknown`), `precision`,
  `dataset_subset` and `dataset_match` (`same|subset|different|unknown`); a `perf` reference adds `shape.batch`,
  `shape.input_len`, `shape.output_len`, a `quality` one `fewshot` and `chat_template`. `unknown` is refused for
  `generated`.
- `values[].measurements` are the samples of the run's ok repeats, flattened; `iterations` is the longest list (1 to 4294967295, like `records`).
- `assets[]` name files under `attachments/` with `sha256`, a record count and the parity case they are for.
- The approval service sets `version`, `visibility`, `trust`, `ref_only`, `submitted_by`, `approved_by`, `approved_at`,
  `approval_reason`, `terms_ack_by`, `approvals` and `upload_id`; a file that names any of them is refused. The version
  is assigned at approval.
- The architecture is `noarch` for a data file and `multi` only when the values name several architectures (`disc.arch`).
- The content digest (the reference's identity) covers the key, class, origin, system, dataset, model, comparability,
  values and assets; terms, label, warnings and `clean` do not change it. The same content under another upload is
  refused at approval (HTTP 409, reported as "duplicate content"); retrying the same upload is idempotent.
- A case whose `disc.kernel_name` is a compiled-kernel name (`spyre_kernel_<name>_<16 hex>[#n]`) and is a key is
  refused: the writer re-keys those.
- A lone surrogate anywhere in the file, or a tag that is not a string, is refused. The digest reads measurement
  numbers as floats, so `11` and `11.0` are the same content.
- An asset and a value row may not share a case on the reference's backend.

Each case block carries `disc_keys`, the ordered key list its `benchmark_id` hashes (absent means the sorted keys of
`disc`), and optionally `id_profile` (`perf-kernel`, `vllm`, `vllm-sweep` or `generic`, informational). `create` copies
the keys the engine used when it derived the stored id, so a profile-ordered id is reproduced exactly and `validate`
recomputes it the same way. Discriminator values are written with `str()`, as the ingest hashes them; a value that is not a string (null, a list)
may give a different id from the engine's own, and `validate` is the arbiter.

## Cleanliness and warnings

A run that is not clean may still become a reference. `create` records `clean` (`true`, `false` or `null` for
unknown) and `warnings`, prints each warning, and exits 0:

| Warning | Meaning |
|---|---|
| `not_clean` | other load was present or exclusivity was not enforced |
| `clean_unknown` | cleanliness was not recorded |
| `exclusivity_not_enforced` | exclusive device access was observed, not enforced |
| `suspect` | the run's validity is suspect |
| `pinning_unverified` | CPU pinning was requested but not verified |
| `single_repeat` | one repeat, so variance is unknown |
| `not_baseline_purpose` | the run's purpose is not `nightly`, `weekly` or `merge` (a run with no purpose is eligible) |
| `unqualified_environment` | the environment was not qualified |

Blockers are a run with an active invalidation and a run with nothing measured: `create` exits 1 and writes
nothing. The invalidation lookup uses the run's stored run ids (`stored_run_ids` in `run.json`, else its run id) in the
store's `v_invalidated_runs` (run, benchmark, node, device and placement invalidations all count). Run ids that are not
UUIDs cannot be looked up and count as unchecked. `create --from-request --run` does the same lookup. When the store cannot answer (`--store none`, an unreachable server, or no id to look up), `create` still writes
the reference but records the warning `invalidation_unchecked` and prints it, so an unchecked run is never silently
clean. Warnings never block and blockers cannot be overridden. The pinning policy the run requested is carried in `pinning`.

`create` and `submit` look the reference's cases up in the store and print the ones that are not registered; the
approval page offers to register them. This never fails the command, and prints a note instead when the store cannot
answer.

## Approval is UI-only in v1

`submit` posts the sealed `.tgz` to the upload route (`POST /api/v2/uploads` as multipart with `bundle`,
`format=reference` and a `submit_token` derived from the bundle's hash; `--dry-run` adds `?dry_run=1`, `--inspect` uses
`/api/v2/uploads/inspect`) and prints the reply (`state`, `upload_id`, `report`). `submit`, `approve`, `reject`,
`retract`, `list` and `show` go through the dashboard API with a bearer token
(`BENCH_DASHBOARD_URL`, `BENCH_DASHBOARD_TOKEN`). In v1 there is no programmatic identity: without a token `approve`,
`reject` and `retract` stop with a message saying approval happens on the dashboard's approval page, and `submit`,
`list` and `show` say they need a token. `submit` checks the bundle locally first, by the same rules as the ingest's
offline check (layout, `files[]`, the keys of `bundle.json`, no artifact, arch, test type, component and schema
version), and only then looks for the token. The rules the service enforces:

- Every origin waits for approval, CI-generated references included; nothing is published automatically.
- Eligible approvers hold the approve permission in scope (class, origin, component, model family, dataset,
  architecture) and are never the submitter. A scope dimension with no fact in the reference fails closed.
- An administrator may approve any reference, including their own; the approval is flagged as an override or a
  self-approval in the audit record, and a reason is optional.
- The submitter (pending or published) or an administrator may retract. Rejecting or requesting changes is for an
  approver in scope or an administrator; a submitter withdraws with `retract`. A retract writes no tag: the current version is the newest
  of the retire rows and the current rows of published uploads, so retracting the current version makes the previous
  published version current, retracting any other version changes nothing, and a retire outlives the retraction of
  its version.
- A scope dimension is compared case-insensitively and a `key` dimension limits an approver to named references;
a binding's expiry may be naive or timezone-aware. `approve --no-promote` approves without making the version current; `--register-missing` registers unregistered
  cases at approval; `--terms-ack` is required, and `--ack-not-clean` when the reference's run was not known to be clean (the service
  answers 400 otherwise). `reject` needs `--reason`.

The client sends the token only over `https://` (`http://` only to localhost), never follows a redirect (the bearer token must not leave the configured host), turns a reply that is not JSON or is cut short into a service error (exit 5), quotes ids in paths, and
derives the `submit_token` from the bundle's hash so a retried submit is the same submit.

## Comparing

```bash
spyre-bench reference compare --run results/bench --reference ref/              # a directory or bundle
spyre-bench reference compare --run results/bench --reference <key> [--version current|N] --store clickhouse
```

With a zero reference value the percent is undefined, so any nonzero delta is outside the tolerance and is judged by
its sign (`worse` when a lower-is-better metric rises, and so on); without a direction it is `changed`. An existing local
path given to `--reference` wins over a key of the same name.

For each case of the run with a value row in the reference (same id, backend and metric) it prints the run value, the
reference value, the delta and percent, and a status: `better`, `worse` or `same` within the tolerance (direction
aware), or `changed` when the metric has no direction. The value on each side is the median of its samples. The
tolerance is `--tolerance-pct`, else the metric's `noise.floor_pct` from `--catalog`, else 5. `comparability` is one
word from the reference's block: `not_comparable` when `harness_match` is `no` or `dataset_match` is `different`;
`comparable` when `harness_match` is `yes`, `dataset_match` is `same` and every other key equals the run's
(`--run-context KEY=VALUE`, repeatable); otherwise `partial`. Output states that it is informational.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | done (`compare` also when everything is worse) |
| 1 | a blocker refused `create`, or the service refused the request |
| 2 | bad flags, an invalid or incomplete reference, a missing token |
| 4 | `list` found nothing, or `compare` matched no case |
| 5 | the store or the service could not answer |

`seal` copies only the files `assets[]` lists (a link, a path with `..` or `\\`, or a file in `attachments/` that no
asset lists is an error), refuses an existing target, except an empty directory, and writes `.tgz` files atomically. The `.tgz` is reproducible: members are sorted, with no owner, no user name and a zero modification time, so sealing
the same reference twice gives the same bytes. It writes `bundle.json` (`kind: bench-reference`, `files[]` with sha256), `results/reference.json` and
`attachments/`, as a directory or a `.tgz`.
