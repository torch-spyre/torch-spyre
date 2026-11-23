# Air-gapped results: one-page checklist

Full guide: [air-gapped-results.md](air-gapped-results.md). Print this page; tick as you go.
Site = no network. Connected side = can reach the dashboard or inbox.

Site: ............  Operator: ............  Date (UTC): ............  Kit version: ............

## Before leaving the connected side

- [ ] Kit built; `SHA256SUMS` written; version range recorded (never a commit)
- [ ] Subject named by **artifact id** or **image digest + arch** (never a tag): ............
- [ ] Images pulled by digest; datasets and models copied, with digests
- [ ] Plan or case list agreed with the uploader (if used)
- [ ] Throwaway bundle sealed with this kit; `results --check <bundle>` says `compatible` on the receiver

## At the site

- [ ] `sha256sum -c SHA256SUMS` on the kit passes
- [ ] Zone-less JUnit times are read as UTC; if tests ran elsewhere give `--triggered-at`/`--ended-at` with a zone
- [ ] Clock checked (`date -u`), offset noted: ............
- [ ] Loaded image digest equals the noted digest
- [ ] No credentials on this machine (none are needed)
- [ ] Benchmarks run; raw results gathered, one folder per run
- [ ] Logs and traces ship **beside** the bundle, listed in `TRANSFER.txt` (only the vLLM kind takes attachments inside;
      other files in `attachments/` fail validation unless listed in `files[]`)
- [ ] Environment facts written down; anything unknown marked **unknown**, never guessed
- [ ] Attachments reviewed for secrets, host names, user names, paths
- [ ] **Seal** (always pass `--arch`; never set `uploader`; add `--seal-trust lab` if it applies; default `external`).
      `SEAL` is `python -m spyre_clickhouse_ingest results` (wheel, 3.11+) or `python offline.py` (kit, 3.9+):
      `SEAL --offline --out <site>-<label>-<date>.tgz ...`
- [ ] No `[warn] not recorded in a bundle` line in the seal output (it means a flag had no effect)
- [ ] Bundle sha256 printed by seal copied here: ............
- [ ] Validate: `SEAL --from-bundle <bundle> --validate-only` prints `valid` and the same bundle sha256 as at seal
      (unless you edited the bundle by hand: then record the new one)
- [ ] Host/user must not leave: seal with `--redact-host` and/or `--redact-user`; read every `[warn]` it prints
- [ ] Redaction is partial (JUnit `<properties>`, `system-out`, attachments are untouched): review those files

## Transfer

- [ ] One bundle per file; no repacking or editing after sealing
- [ ] `TRANSFER.txt`: site, date, kit version, per bundle: file, file sha256, bundle sha256
- [ ] `SHA256SUMS` for every file on the media
- [ ] Media scanned per the receiving side's rules; custody noted (sealed by / carried by / received by)

## Upload (connected side)

- [ ] `sha256sum -c SHA256SUMS`; each bundle `--validate-only`; bundle sha256 matches `TRANSFER.txt`
- [ ] Preview read (dashboard preview or `--dry-run`); no `identical to upload` warning; missing fields filled
- [ ] "On behalf of" = the site identifier
- [ ] Uploaded (dashboard, `results --from-bundle --strict`, or inbox with `--wait`)
- [ ] Receipt copied for each bundle: upload id: ............  run id: ............  content sha256 equals bundle sha256 [ ]
- [ ] State is `published` (or `pending`: reviewer informed)

## After

- [ ] Run visible on the subject's page with the expected counts
- [ ] Corrected bundle? Seal once, upload, then retract the old upload with a reason
- [ ] Receipts sent back to the site; media wiped per policy

## Never

- Never use `--redact` alone (ambiguous): write `--redact-host` / `--redact-user`.
- To retry an upload, resend the same file rather than re-sealing.
- Never guess an environment fact, a time zone or an artifact id.
- Never put a tag where a digest goes.
- Never edit a bundle after sealing, except by the guide's recipes (then record the new digest).
- Never expect a site upload to be a baseline; a missing trust label means unknown, not trusted.
- Never treat the sha256 as tamper-proof: it is self-attested; it catches accidental corruption only.
- Upgrade the receiving side first: older receivers reject the kit's `trust` and `producer` keys.
