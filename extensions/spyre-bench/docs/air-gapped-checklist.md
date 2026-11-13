# Air-gapped results: one-page checklist

Full guide: [air-gapped-results.md](air-gapped-results.md). Print this page; tick as you go. Site = no network. Connected side = can reach the dashboard or inbox.

Site: ............  Operator: ............  Date (UTC): ............  Kit version: ............

## Before leaving the connected side

- [ ] Kit built; `SHA256SUMS` written; version range recorded (never a commit)
- [ ] Subject named by **artifact id** or **image digest + arch** (never a tag): ............
- [ ] Images pulled by digest; datasets and models copied, with digests
- [ ] Plan or case list agreed with the uploader (if used)
- [ ] Throwaway bundle sealed and validated with this kit on a connected machine

## At the site

- [ ] `sha256sum -c SHA256SUMS` on the kit passes
- [ ] Sealing on the test machine, or `--triggered-at <time with zone>` given (a zone-less JUnit time is read in the sealing machine's zone)
- [ ] Clock checked (`date -u`), offset noted: ............
- [ ] Loaded image digest equals the noted digest
- [ ] No credentials on this machine (none are needed)
- [ ] Benchmarks run; raw results gathered, one folder per run. Logs and traces ship **beside** the bundle and are listed in `TRANSFER.txt` (only the vLLM kind takes attachments inside; anything else in `attachments/` fails validation unless listed in `files[]`)
- [ ] Environment facts written down; anything unknown marked **unknown**, never guessed
- [ ] Attachments reviewed for secrets, host names, user names, paths
- [ ] **Seal once** (always pass `--arch`; never set `uploader`). `SEAL` is `python -m spyre_clickhouse_ingest results` (wheel, Python 3.11+) or `python offline.py` (two-file kit, Python 3.9+):
      `SEAL --offline --out <site>-<label>-<date>.tgz ...`
- [ ] No `[warn] not recorded in a bundle` line in the seal output (it means a flag had no effect)
- [ ] Bundle sha256 printed by seal copied here: ............
- [ ] Validate: `SEAL --from-bundle <bundle> --validate-only` prints `valid` and the same bundle sha256 as at seal (unless you edited the bundle: then record the new one)
- [ ] Host and user name removed if they must not leave: seal to a folder, delete `runner` from `bundle.json`, validate (record the NEW digest), then `(cd <folder> && tar --uid 0 --gid 0 --uname '' --gname '' -czf ../x.tgz *)` (GNU tar: `--owner=0 --group=0 --numeric-owner`)

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

- Never re-seal to retry; resend the same file.
- Never guess an environment fact, a time zone or an artifact id.
- Never put a tag where a digest goes.
- Never edit a bundle after sealing, except by the guide's recipes (then record the new digest).
- Never expect a site upload to be a baseline.
