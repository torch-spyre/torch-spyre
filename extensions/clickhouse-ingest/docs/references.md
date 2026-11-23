# Reference bundles (`bench-reference`)

A **reference** is a comparison target: an output set or a set of values that a run of ours is
compared to (a CPU fp32 output set, a figure from a model card, a value measured elsewhere). It is
unrelated to a run's "reference grade", it is never a baseline, and nothing compared to it gates a
build or a check.

A reference lives on the existing tables, written once, at approval:

| Table | Row |
|---|---|
| `artifacts` | one version: `component='reference'`, `kind='generic'`, `artifact_name` = the key, `id12` = first 12 hex of the content digest, `arch` = `noarch` (`multi` only when the values name several architectures); every `ref.*` fact in `props` |
| `artifact_refs` | the sealed bundle's address |
| `benchmark_runs` | the values, one row per (case, backend), flagged `run.ref_only=1`, `run.trust=external`, `run.record_type=reference` |
| `artifact_results` | one write-once `passed` verdict linking the artifact to the value rows' run |
| `artifact_tags` | `ref-current` / `ref-retired` rows (family `reference`); the newest row of a key decides its state |
| `result_uploads` | the lifecycle: `format=reference`, `pending` -> `published` -> `retracted` (or `rejected`, `expired`, `failed`) |

Every reader of `benchmark_runs`, `artifacts` and the tag views skips these rows (schema 85 and the
reader exclusions); `v_bench_references` and `v_bench_reference_values` (schema 86) are the readers
that want them.

## The file

`results/reference.json`, `apiVersion: bench/v1alpha1`, `kind: Reference`:

```json
{
  "apiVersion": "bench/v1alpha1", "kind": "Reference",
  "key": "perf.example.decode.refsys.latency", "origin": "generated", "class": "perf",
  "label": "ref-sys-a", "terms": "internal-generated",
  "tools": {"torch": "2.9.1"}, "model": {"repo": "org/model", "revision": "abc123"},
  "system": {"backend": "cpu", "precision": "fp32"},
  "clean": false, "warnings": ["not_clean"], "pinning": "none",
  "comparability": {"harness": "tool@1", "harness_match": "yes", "precision": "fp32", "dataset_subset": "full",
                    "dataset_match": "same", "shape.batch": 1, "shape.input_len": 128, "shape.output_len": 64},
  "values": [{"case": {"component": "example", "name": "decode/latency", "tags": [], "disc": {"model": "m"},
                       "benchmark_id": "<recomputed and checked>"},
              "backend": "cpu", "measurements": {"llm.itl.median": [11.0, 11.5]}, "iterations": 2}]
}
```

- `origin`: `generated` (we produced it), `uploaded` (someone else's measurement), `published`
  (transcribed from a public source). `uploaded` and `published` also need `producer`, `as_of`
  (not in the future), `comparability.caveats`; `published` also `source_url` and non-empty caveats.
  `generated` also needs `tools` and `model.revision`.
- `comparability` is required by origin and class (`fewshot`, `chat_template` for quality;
  `shape.batch`, `shape.input_len`, `shape.output_len` for perf). `unknown` is allowed for
  `harness_match`, `chat_template`, `dataset_match` except on a `generated` reference.
- `label` is abstract (`[a-z0-9-]`; the built-in check refuses `node`, `host`, `worker` and dots). A
  deployment adds its own patterns, one regex per line (`#` comments), in the file named by
  `$SPYRE_BENCH_LABEL_DENYLIST`; the same file makes the engine's pre-check agree with the writer.
- A case's `benchmark_id` is recomputed from component/name/tags/disc and must match. `disc_keys`
  (the ordered key list the id hashes; absent means the sorted keys) and `id_profile`
  (informational) let a profile-ordered stored id be reproduced. The case must already be
  registered in `benchmarks`, or the approval must ask to register it now; a dry run reports the
  unregistered ones instead of failing.
- A file in `attachments/` that no asset lists, a link, or a path with `..` is refused.
- An asset-only reference (reference outputs as bytes) lists `assets[]` (`path` under `attachments/`,
  `sha256`, `records`, the `case` it is for); each becomes a `ref.records` row on that case.
- The writer sets, and a file naming them is refused: `version`, `visibility`, `trust`, `ref_only`,
  `submitted_by`, `approved_by`, `approved_at`, `approval_reason`, `terms_ack_by`, `approvals`,
  `upload_id`.
- `ref.digest` covers the meaning (key, class, origin, system, dataset, model, comparability, values,
  asset digests), not the wording (label, terms, warnings): any change of values, system or
  comparability is a new version. The same content under another upload is refused; a retry of the
  same upload resumes a failed write or reports `duplicate`.

## Writing one

```
results --offline --reference-dir <dir with reference.json and attachments/> --out <bundle>   # seal
results --from-bundle <bundle> --validate-only                                               # check
results --upload <bundle>                                                                    # held in references/
```

A reference bundle is held in a `references/` folder next to the relay inbox and is never
recorded on arrival. `results --from-bundle` writes it only at approval, with
`--bundle-prop approved_by=<login> --bundle-prop upload_id=<id> --bundle-prop submitted_by=<login>`
(optionally `approval_reason`, `approval_override=1`, `terms_ack_by`, `approved_at`, `promote=0`,
`register_missing=1`). This is an authorised-caller-only entry point: it verifies none of it (who may
approve is decided by the caller with `authorize_approval`), except that a self-approval must carry
`approval_override=1` and then always records both `ref.approval_override` and `ref.approval_self`.
The submitter is the stored upload's, never the bundle's. `--wait` is refused for a reference.

The write order is artifact, address, value rows, the `ref-current` tag, then the verdict, which is
the completion marker. Approvals of one key must be serialised by the caller (the version is a count of
the key's completed versions); a version held by two references is reported as an error.

## Approval rules (`reference.py`, pure)

- Eligible approvers hold an active `references:approve` binding whose scope (`class`, `origin`,
  `component`, `model_family`, `dataset`, `arch`, `key`) fits the reference; a limited dimension with no fact
  fails closed.
- A non-admin approver must be in scope and is never the submitter.
- An admin (`references:admin`) may approve any reference, including their own; the approval is
  flagged `ref.approval_override` (no binding in scope, or own upload) and `ref.approval_self`.
- Every origin, `generated` included, waits for approval; there is no auto-publish.
- A retraction is final for that content: v2 is append-only and a retracted upload stays hidden, so
  the same content cannot be approved again; a corrected reference has changed content (a corrected
  comparability block counts).
- Retract: the submitter or an admin. Nothing else is written: the view derives the state, so retracting
  the current version leaves the previous tagged published version current, and retracting a duplicate
  upload changes nothing. A `ref-retired` row is the operator's statement about the key and outlives the
  retraction of the version it names.
- Promoting a stored run: `promotion_check` blocks an invalidated or empty run and warns on a run
  that was not clean (`ref.clean`, `ref.warnings`); the warning is acknowledged, not a refusal.
  A purpose other than `nightly`, `weekly`, `merge` (the rows of `benchmark_baseline_purposes`)
  warns `not_baseline_purpose`; a run with no purpose does not.

## Lifecycle

`pending` (submitted) -> `published` (approved; the write happens here) -> `retracted`; a pending
upload can also be `rejected`, `expired` or `failed`. `result_uploads.chk_format` accepts
`reference` (and `bench`) from migration 026.

## Invariants (held by the writer and the view, pinned by a randomised interleaving test on chdb)

- Per key, at most one version is `current`, and only a version whose upload is published can be.
- A `ref-current` row counts only while its artifact's upload is published; a `ref-retired` row counts
  always; the newest counted row decides. So `current` is a function of the published uploads and the tag
  rows, with no write needed after a retract.
- A version number is never reused. An attempt that fails after its artifact keeps its number; its retry
  resumes it with the same run, address and approver.
- A resumed older write never takes `current` from a newer published version.
- Retracting an upload never changes the live version unless that upload held it.
- Every accepted file inserts: counts are checked against the destination columns (`iterations`,
  `records` are UInt32), and a compiled-kernel `disc.kernel_name` (which the writer re-keys) is refused.

Things an operator can be surprised by (all follow from the rules above):

- A version approved without promotion is `untagged`; if the current version is then retracted, the key has
  no current until an operator rolls back to a version explicitly.
- A rollback to an unpublished or retracted version is ignored by the view (the row is written, nothing
  changes).
- A write resumed after an operator retired the key can tag the version again; retire after the write is
  complete.
- `promote` is recorded with the first attempt (`ref.promote`) and a retry cannot change it.
- A collision between concurrent approvals of one key is resolved by renumbering the later claimant with a
  second row for its artifact (the merged view reads the newest props, the row is stamped later than the
  artifact's earlier rows); the caller should still serialise. An attempt that already wrote value rows is
  not renumbered: it fails with a clear message and the content must be re-sent changed. A renumbered
  resume never takes `current` from a newer published version.
