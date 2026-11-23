# Uploading results

Results reach the store through one writer, whichever way they arrive. How they arrive decides their **trust**; trust is
derived by the server from the channel, never written by the uploader.

| Channel | Who | Trust | Status |
|---|---|---|---|
| CI writes directly after a run | automation | `ci` | Available |
| Dashboard upload page | any signed-in person | `lab` or `external` (per account) | Available for the existing result kinds; the `bench` kind is Pending |
| Command line (`results --offline`, `--upload`, `--from-bundle`) | person with access | by channel | Available for existing kinds |
| Programmatic API with a token | a script | by channel | Planned |
| Results inbox relay | people without direct access | by channel | Available, with a fix Pending |
| Offline (air-gapped) bundle | a site with no connection | `external` | Pending, see below |

Only `ci` data can set baselines or references. A `lab` or `external` upload is stored, labelled, and appears pending
review where the deployment requires it.

## Dashboard

1. Sign in and open the upload page.
2. Pick the result kind and attach the file or bundle.
3. Submit; the upload shows as pending, then published, or rejected with a reason.
4. You can retract your own upload while it is pending or published.

## Command line

```bash
# build a sealed bundle from raw results (works offline)
python -m spyre_clickhouse_ingest results --offline --out bundle.tgz ...
# check it offline (Pending: --validate-only ships with the air-gapped tooling)
python -m spyre_clickhouse_ingest results --from-bundle bundle.tgz --validate-only
# upload it: the dashboard/inbox route, which keeps an upload record
python -m spyre_clickhouse_ingest results --upload bundle.tgz --wait
```

`--from-bundle` (with `--dry-run` or `--strict`) is the step the relay or the dashboard runs *on the server side*: it
writes straight to the database and needs database credentials. Use it only if you are the operator of that database;
everyone else uses `--upload` or the dashboard.

Exit codes: 0 ingested or duplicate, 1 failed (retry), 2 rejected, 3 incomplete. Re-sealing the same raw results
creates a new bundle (it carries a timestamp); retry the same bundle file instead.

## Air-gapped sites

A site with no connection collects results, seals them into a bundle, verifies it offline, transfers it, and a
connected person uploads it. The full procedure, checklist and trust rules are in the air-gapped results guide and
checklist (**Pending** cross-reference: `air-gapped-results.md` and `air-gapped-checklist.md` appear in this folder
when the air-gapped documentation merges).

## Before you upload

- Use the engine's own output rather than hand-made rows: the engine records environment, placement and validity.
- Redact: bundles must not contain tokens or internal hostnames; the sealing tool can redact (**Pending**: the `--redact-*` options ship with the air-gapped tooling).
- Use a development database for experiments.

## When it fails

| Symptom | Fix |
|---|---|
| rejected: malformed | run `--validate-only` and read the message |
| duplicate | the same run was already ingested; nothing to do |
| "trust" refused | the bundle claims a trust level; remove it, the server sets it |
| nothing appears | the upload is pending review; ask a reviewer |

## Who to ask

The maintainers of your deployment for access; the benchmark owner for the content.
