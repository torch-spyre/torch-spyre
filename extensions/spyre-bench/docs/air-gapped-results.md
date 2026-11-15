# Collecting and uploading results from an air-gapped environment

This is a runbook for taking benchmark and test results on a system that has no network route to the
results database, the dashboard, the package index or the source host, and getting them recorded
on the connected side. A one-page printable version is in [air-gapped-checklist.md](air-gapped-checklist.md).

## Contents

- [Scope and vocabulary](#scope-and-vocabulary)
- [What exists today and what is planned](#what-exists-today-and-what-is-planned)
- [The flow in one picture](#the-flow-in-one-picture)
- [Prerequisites](#prerequisites)
- [Step 1: prepare the kit (connected side)](#step-1-prepare-the-kit-connected-side)
- [Step 2: prepare the inputs](#step-2-prepare-the-inputs)
- [Step 3: configure the site](#step-3-configure-the-site)
- [Step 4: run and collect](#step-4-run-and-collect)
- [Step 5: seal](#step-5-seal)
- [Step 6: verify offline](#step-6-verify-offline)
- [Step 7: transfer](#step-7-transfer)
- [Step 8: upload (connected side)](#step-8-upload-connected-side)
- [Step 9: verify and keep the receipt](#step-9-verify-and-keep-the-receipt)
- [Corrections, duplicates and retraction](#corrections-duplicates-and-retraction)
- [The bundle: layout and manifest](#the-bundle-layout-and-manifest)
- [Trust and what a site cannot claim](#trust-and-what-a-site-cannot-claim)
- [Time and clocks](#time-and-clocks)
- [Security notes](#security-notes)
- [Versioning and compatibility](#versioning-and-compatibility)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Scope and vocabulary

An environment is air-gapped for this guide when the machines that run the benchmarks have **no route** to the
results database, the dashboard, the artifact store, the source host or the engine's package index. Data
leaves on removable media or through a one-way transfer host. Typical cases are a partner or customer
site, a secured lab, or lab nodes behind a one-way firewall.

| Term | Meaning |
|---|---|
| **Site** | the air-gapped side: where benchmarks run and bundles are sealed |
| **Site operator** | the person who runs the benchmarks and seals the bundles |
| **Connected side** | the side that can reach the dashboard, the inbox store or the results database |
| **Uploader** | an internal operator, or the dashboard acting for a signed-in user, who records bundles on the connected side |
| **Bundle** | a folder or `.tgz` holding result files and a `bundle.json` manifest with a sha256 for every file |
| **Seal** | write the manifest and pack the bundle; needs no network and no credentials |
| **Subject** | the thing that was measured: an image, wheel or package, named by an artifact id or by an image digest |

The site never writes to the results database. It produces a sealed file; the uploader decides whether
and how it is recorded.

## What exists today and what is planned

Everything marked **available** ships in `extensions/clickhouse-ingest`. Anything marked **planned** or
**proposed** does not exist yet; the guide mentions it only so a site can plan, and gives the workaround that
works today.

> **Warning: an unknown flag is silently ignored.** The sealer prints
> `[warn] not recorded in a bundle: --attach foo` (the flag and its value), carries on with exit code 0 and
> produces a normal bundle. If you see this line, **that flag had no effect.** Check the output of every seal
> for it. A misspelt prefix of a real flag is different: `--redact` matches both `--redact-host` and
> `--redact-user`, so it is refused with `ambiguous option` and exit code 2.

| Capability | Command or place | Status |
|---|---|---|
| Seal a bundle from JUnit, vLLM bench, perf-suite report, generic benchmark rows or per-table rows | `SEAL --offline --out <bundle> ...` | available |
| Same results, same host, same kit: same bundle digest and same `.tgz` bytes | seal | available (see [Step 5](#step-5-seal)) |
| Leave the sealing host and user out of the bundle | `--redact-host`, `--redact-user` | available (partial, see Step 5) |
| Label where the results came from | `--seal-trust lab\|external` (`none` writes no label, for a CI job) | available |
| Seal an engine results directory (kind `bench`) | `SEAL --offline --bench-dir <dir> ...` | available |
| Run the sealer with the standard library only | `SEAL="python offline.py"` (one file plus `bundle.schema.json`) | available |
| Validate a bundle with no network | `results --from-bundle <bundle> --validate-only` | available |
| Ask what an ingest accepts, and whether it can read a bundle | `results --check [<bundle>]` | available |
| Put a bundle in the results inbox | `results --upload <bundle> [--wait]` | available |
| Record a bundle on the connected side | `results --from-bundle <bundle> --strict` (add `--dry-run` to preview) | available; prints a receipt id |
| Dashboard upload page: inspect, missing-field prompts, preview, publish or hold for review, retract | the dashboard's Uploads page | available (one bundle per upload) |
| Engine-level collection verbs, batch upload, a kit build script | various | planned |
| Size caps (`--max-bytes`), split bundles (`--split`), attaching files (`--attach`), a site field | seal flags, manifest | proposed (none exist) |
| Signing | seal | not implemented |

## The flow in one picture

```text
 CONNECTED SIDE                  TRANSFER                    SITE (air-gapped)
 --------------                  --------                    -----------------
 1 build the kit  ------------>  media  ------------------>  install kit, check sha256
 2 note subject ids / digests --> media  ------------------>  load images, datasets, models
                                                             3 configure (no secrets)
                                                             4 run, collect raw reports,
                                                               traces, environment notes
                                                             5 seal (no network)
                                                             6 validate (no network)
 8 upload  <-------------------  media  <------------------  7 transfer: bundle + checksums
 9 receipt (id + digest) ------>  media  ------------------>  site matches receipt to its digest
```

## Prerequisites

At the site:

- Python, in one of two forms (see [Command notation](#command-notation)):
  - **the two-file kit** (`offline.py` and `bundle.schema.json`): standard library only, **Python 3.9 or newer**;
  - **the wheel**: the package declares **Python 3.11 or newer** and will not install on 3.9 or 3.10.

  A site on 3.9 or 3.10 must use the two-file kit.
- The benchmark tools themselves, their images, datasets and model weights, loaded from media.
- Somewhere to write output (a few times the size of the raw results) and a way to compute sha256
  (`sha256sum` or `shasum -a 256`).
- No account, token or credential of any kind. A sealing machine that holds none cannot leak any.

On the connected side:

- A signed-in user with upload rights on the dashboard, or an operator who can run
  `spyre-clickhouse-ingest` against the results database, or write access to the results inbox.
- The subject must already be known to the results database, or you must be able to record it
  (see [Step 2](#step-2-prepare-the-inputs)). `--from-bundle` refuses an artifact that is not recorded.
- For recording from the command line: the full package (`pip install` of `extensions/clickhouse-ingest`, Python
  3.11 or newer), `CLICKHOUSE_DB_V2` naming the target database, the usual connection variables, and a database
  that is reachable from the machine.
- For the inbox route: `ARTIFACTORY_TOKEN` (or `~/.netrc`), `ARTIFACTORY_BASE_URL`, `ARTIFACTORY_V2_RESULTS_REPO` and
  `ARTIFACTORY_V2_RESULTS_ROOT` (or `--token`, `--base-url`, `--repo`, `--root`), supplied by the deployment.

## Command notation

The sealer runs in two forms with the same flags. This guide writes `SEAL` for whichever one your kit gives you:

```bash
# the wheel (Python 3.11+, installed with pip)
SEAL="python -m spyre_clickhouse_ingest results"
# the two-file kit (Python 3.9+), run from the folder that holds offline.py and bundle.schema.json
SEAL="python offline.py"
```

Site-side commands (`--offline`, `--validate-only`, `--upload`) work in both forms. Commands that **record** a bundle
(`--from-bundle ... --strict`, `--dry-run`) belong to the connected side and need the full wheel and a database
connection: they are always written as `python -m spyre_clickhouse_ingest results`.

## Step 1: prepare the kit (connected side)

The kit is whatever the site needs to seal and validate. Today that is two files:

```text
spyre_clickhouse_ingest/offline.py
spyre_clickhouse_ingest/bundle.schema.json
```

Carry the two files together (they must sit side by side); this works on Python 3.9 and newer. Or build a wheel
and carry it; **the wheel installs only on Python 3.11 or newer**, so it is not an option for a 3.9 or 3.10 site.

```bash
# two-file kit: copy offline.py and bundle.schema.json from the checkout into kit/
cp extensions/clickhouse-ingest/spyre_clickhouse_ingest/{offline.py,bundle.schema.json} kit/
# wheel (needs Python 3.11+ on the site): on a connected machine, from a checkout
pip wheel --no-deps extensions/clickhouse-ingest -w kit/
(cd kit && sha256sum * > SHA256SUMS)          # macOS: shasum -a 256 * > SHA256SUMS
```

On the site:

```bash
cd kit && sha256sum -c SHA256SUMS         # macOS: shasum -a 256 -c SHA256SUMS
pip install --no-deps --no-index spyre_clickhouse_ingest-*.whl     # Python 3.11+ only; else run python offline.py
```

Then seal and validate one throwaway bundle from sample results (steps 5 and 6) before the campaign starts.
The two-file kit has no installed package, so it records `producer.version` as `unknown`; the wheel records its version.

Version rule: **pin the kit by a version range, never by a commit.** Record the kit version in the
transfer notes. The upload side accepts bundles from an older kit as long as `schema_version` is not newer than
the ingest it runs, and the connected side must know every key the kit writes; see
[Versioning and compatibility](#versioning-and-compatibility). Before the campaign, ask the connected side to run
`results --check <your throwaway bundle>` (it prints `compatible` and names what to upgrade if not).

A single kit archive or wheelhouse with a build script is **planned**; none exists.

## Step 2: prepare the inputs

Before leaving the connected side, collect the identifiers the site must put in the bundle, and carry the inputs.

1. **Subject identity.** Every bundle names what was measured, in one of two ways:
   - the subject's **artifact id** (a UUID) if it is already recorded, or
   - the image **digest** (`<repo>@sha256:<64 hex>`) plus the architecture. Never a tag: a tag moves and says
     nothing about what ran.
2. **Pull by digest** on the connected side, then carry the image archive. On the site, load it and confirm
   the digest is the one you noted (`podman image inspect --format '{{index .RepoDigests 0}}' <image>`).
3. **Datasets and models**, with their digests, as media. Benchmarks that fetch data at run time will fail
   or silently use something else; pin them to local copies.

If the subject was built at the site and is not known to the results database, give the digest and
architecture and say which component owns it; the uploader records it as an externally built artifact
(the dashboard form asks for the owning component, or `bundle.json` carries `create_artifact`). Do not try to compute
an artifact id at the site: the id depends on inputs that only the connected side knows.

## Step 3: configure the site

Nothing here is a secret.

| Setting | Where | Notes |
|---|---|---|
| Architecture | `--arch` on every seal | pass it explicitly; the default is the sealing machine's own and a host that reports an unsupported name (for example `arm64`) is rejected |
| Test type (tier) | `--trigger-type` | `perf` or `quality` for benchmark rows, otherwise the tier of the tests (`smoke`, `regression`, ...) |
| Suite owner | `--component` | for `benchmarks` and `tables` kinds |
| Site identifier | see below | free text such as `site-a` |
| Trust label | `--seal-trust lab\|external` (default `external`) | a claim only; the connected side decides the level recorded, see [Trust](#trust-and-what-a-site-cannot-claim) |

The manifest has no dedicated site field, and a manifest with a key the schema does not list is rejected. Until one exists, put
the site name in the transfer notes and give it to the uploader, who records it as "measured by / on behalf of"
in the upload form. A `site` field is **proposed**; do not add it by hand.

Do not set `uploader` in `bundle.json`. The dashboard refuses a bundle whose `uploader` differs from the signed-in
user.

## Step 4: run and collect

Run the benchmarks with their own tools, then gather the raw output into one folder per run. What the bundle can hold
today:

| You have | Seal with | Bundle `kind` |
|---|---|---|
| JUnit XML (pytest or any producer) | `--xml-dir <dir>` or `--xml-file <file>` | `junit` (default) |
| `vllm bench` output: every `*.json` in the directory is taken (keep the `.pytorch.json` files); `.cmd` and `.log` files go to `attachments/` | `--vllm-results-dir <dir>` | `vllm` |
| a perf-suite text report | `--kind sendnn --sendnn-report <file> --triggered-at <ISO time>` | `sendnn` |
| rows of generic benchmark records (evals, harnesses) | `--benchmarks-dir <dir> --trigger-type perf\|quality --component <owner>` | `benchmarks` |
| your own rows, one JSON file per results table | `--tables-dir <dir> --trigger-type <tier> --component <owner>` | `tables` |

Collect, in addition to the results themselves:

- **The command lines and the log of each run.** Attachments are kept with the bundle and are not ingested.
  **Today only the `vllm` kind adds attachments** (the `.cmd` and `.log` files beside its results). For every other
  kind the sealer copies result files only, and a file you drop into `attachments/` of a sealed folder makes
  validation fail with `file(s) not in files[]`. You have two options:
  1. **Preferred: ship the logs beside the bundle**, not inside it, and list them in `TRANSFER.txt` with their
     sha256. The uploader keeps them with the campaign notes.
  2. **Put them inside by hand**, with the recipe [below](#recipe-add-attachments-by-hand).

  > **Warning: `--attach` does not exist.** It is proposed only; the sealer ignores it with a
  > `[warn] not recorded in a bundle: --attach` line and the file is not in the bundle.
- **Traces and profiles** are large (see [size](#step-5-seal)). Ship them beside the bundle by agreement with the
  uploader (option 1); do not put them in the bundle unless the size caps allow it.
- **Environment and placement.** Record what the machine was: operating system, kernel, driver and firmware
  versions, device model, device count, how the devices were connected, and whether anything else was running.
  A tool that writes this as a file is planned. Until it exists, write the same facts in a plain text file and ship it beside the bundle (or inside it, with the hand recipe above). **Never guess a value:** an unknown
  field is recorded as unknown, and a guessed one is worse than none because it can put a result in the wrong series.

A site that cannot supply an environment record still gets its results recorded, with the environment unknown.

### Recipe: add attachments by hand

Seal to a folder, add the files (names without spaces), list every file in `files[]`, validate. Copy and paste each
block as it is; the Python program is a separate file so nothing in it is indented.

```bash
SEAL --offline --out site-a-run1 --xml-dir out --artifact id:<artifact-id> --arch <arch> --trigger-type <tier>
mkdir -p site-a-run1/attachments
cp run.log site-a-run1/attachments/
```

Save this as `list_files.py`:

```python
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
meta = json.loads((root / "bundle.json").read_text())
meta["files"] = [
    {"path": p.relative_to(root).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    for p in sorted(root.rglob("*"))
    if p.is_file() and p.name != "bundle.json"
]
(root / "bundle.json").write_text(json.dumps(meta, indent=2) + "\n")
```

```bash
python3 list_files.py site-a-run1
SEAL --from-bundle site-a-run1 --validate-only      # must print "valid"; record its bundle_sha256
```

The digest now differs from the one printed when you sealed: record the one printed by this validation. This is an
edit after sealing, which is allowed only by the recipes in this guide. Then pack the folder; a `.tgz` of the same
folder has the same digest:

```bash
(cd site-a-run1 && tar -czf ../site-a-run1.tgz *)
SEAL --from-bundle site-a-run1.tgz --validate-only
```

## Step 5: seal

```bash
SEAL --offline --out site-a-run1.tgz \
    --xml-dir out --artifact id:<artifact-id> --arch <arch> --trigger-type <tier>

# image instead of an id
SEAL --offline --out site-a-run1.tgz \
    --vllm-results-dir out --artifact image:<repo>@sha256:<digest> --arch <arch>
```

The command prints the **bundle sha256** and an inbox path. Keep the digest; the inbox path is a placeholder with an unresolved variable at a site, so ignore it (validation prints it as `upload_path`). It makes no
network connection and reads no credential.

What sealing does: copies the result files into `results/`, writes `bundle.json` with the sha256 of every file,
checks the result, and packs the folder. `--out` can be a `.tgz` or a plain folder name (then nothing is packed). The bundle
sha256 covers every file's path and content, `bundle.json` included, and is **the same for a folder and for its `.tgz`**.

Things to know before sealing:

- **Sealing is repeatable.** No seal time is written, and the `.tgz` carries no times, owners or modes. Sealing the same
  results again gives the same bundle sha256 and the same `.tgz` bytes, provided these are also the same: the sealing
  host and user (unless redacted), the kit version (`producer.version`), the flags, and the zlib build in the
  `.tgz`. A re-seal after a lost upload is therefore a duplicate, not a second run. Still, **seal once and copy the
  file** when you can: a different host, kit version or zlib gives a different digest, which the connected side
  treats as a different run.
- **`ended_at` is the end of the last JUnit suite**, or `--ended-at <ISO time with zone>` if given, or absent. It is
  not the time of sealing. `--triggered-at` and `--ended-at` must carry a zone; an end before the start is clamped to
  the start with a warning.
- **The sealing host and user are recorded** in the `runner` block of `bundle.json`. To leave them out, add
  `--redact-host` and/or `--redact-user` to the seal command. Redaction changes the digest, and the seal prints
  the digest of the redacted bundle, so there is nothing to re-validate or repack.
  - `--redact-host` also blanks the `hostname` attribute of JUnit `<testsuite>` elements in `results/*.xml`
    (the rest of each file is kept byte for byte). The seal prints `[info] blanked N testsuite hostname
    attribute(s)`, and `[warn] the sealing host name is still in [...]` for any result file that still contains it.
  - **Redaction is partial.** `--redact-host` scans only `results/`, never `attachments/`. It does not touch JUnit
    `<properties>`, `system-out`, `run_url`, attachments or other kinds' result files, and the leftover-name check does not read UTF-16 XML or reliably tell a short host
    name from other text. Review the files yourself; act on every `[warn]`.
  - Test it on a throwaway bundle first: `tar -xzOf x.tgz bundle.json` shows what `runner` holds.
- **Size.** The connected side limits the size of an upload, the unpacked size and the file count; these are
  deployment settings shown on the upload page. The sealer does not check them. Keep traces out of the bundle
  unless needed, and send large attachments separately by agreement with the uploader. Size caps and splitting
  (`--max-bytes`, `--split`) are **proposed**; a result set that has to be added to in rounds can use the dashboard's
  "add to an existing run" option when the deployment enables it.

  > **Warning: `--max-bytes` and `--split` do not exist.** They are ignored with a
  > `[warn] not recorded in a bundle` line, and an oversize bundle is only refused at upload.
- **One test type per bundle.** Test cases and benchmark rows are separate bundles.

## Step 6: verify offline

```bash
SEAL --from-bundle site-a-run1.tgz --validate-only
```

A valid bundle prints a one-line JSON report with `"status": "valid"`, its `kind`, `test_type`, `bundle_sha256` and the
inbox path. Validation checks the manifest against the schema, that every file is listed and its sha256 matches, that no
archive member escapes the folder, that each result file parses for its kind, and that the bundle records something.
It cannot check that the subject exists, that the run key is free, or what the database would say; that
is the uploader's preview (below).

Exit codes: `0` valid, `2` rejected (fix the bundle), `3` incomplete (a listed file is missing), `1` failed.

Compare the printed `bundle_sha256` with the one printed when you sealed. If they differ, the bundle changed.
This check detects **accidental** corruption (a bad copy, a repacked archive). It is not tamper-proof: the digests
are written by the sealer into the bundle itself, so anyone who edits the files can recompute them and the
bundle will still validate. See [Security notes](#security-notes).

To ask whether the connected side's ingest can read the bundle at all, run `results --check site-a-run1.tgz` there:
it prints the schema versions, kinds and trust labels the ingest accepts and, for the bundle, `"compatible": true`, or
`false` with `problems` and exit `2`.

Record, for the transfer notes: the file name, `sha256sum` of the file, the bundle sha256, the kit version, and the
site identifier. `sha256sum` of the `.tgz` file and the bundle sha256 are different numbers: the first guards the
file in transit, the second is the identity the database records.

## Step 7: transfer

Put the following on the media or through the transfer host:

```text
site-a-run1.tgz            the sealed bundle (one file per bundle)
TRANSFER.txt               site id, date, kit version, one line per bundle: file, file sha256, bundle sha256, notes
SHA256SUMS                 sha256 of every file on the media, made with (cd media && sha256sum * > SHA256SUMS)
```

Rules:

- **Name files for what they are**: `<site>-<run label>-<date>.tgz`. The name is only a label; identity comes from the
  contents.
- **One bundle per file**, no repacking, no renaming of members, no editing after sealing, except by the recipes in this guide (then record the new digest from validation).
- **Recompute on arrival.** The uploader runs `sha256sum -c SHA256SUMS` (macOS: `shasum -a 256 -c SHA256SUMS`), then `--validate-only` on each bundle, and
  compares the bundle sha256 with `TRANSFER.txt`.
- **Chain of custody.** Write down who sealed it, who carried it, who uploaded it, and when. The upload record keeps the
  uploader and time; `TRANSFER.txt` keeps the rest.
- **Scan the media** according to the receiving side's rules before opening anything.
- Delete the media or the staging folder afterwards according to your retention policy. A bundle on the connected side
  is kept as the record.

## Step 8: upload (connected side)

Choose the route that your deployment offers. All three use the same bundle and the same writer.

### Dashboard upload page

1. Sign in, open the Uploads page, choose **Sealed bundle**, and select the `.tgz`.
2. Read the preview. The form shows the format it found, the case, measurement or row counts, the subject it resolved
   and what is still missing. Fill in the fields it marks required (subject if the bundle does not name one it can
   resolve, tier, architecture, and for a site's bundle the person or site the results are **on behalf of**).
3. If the preview reports a duplicate (`identical to upload ...`), stop: see [Corrections](#corrections-duplicates-and-retraction).
4. Choose **Upload**. Depending on the deployment, the result is published or waits as **pending** until a reviewer
   approves it. A result from outside the program should wait for review.
5. Copy the **upload id**, the **run id** and the **content sha256** from the response; they are the receipt.

The page accepts one sealed bundle per upload. Several bundles are several uploads; do them one at a time and keep one
receipt each. Uploading many at once is **proposed**.

### Command line

With the results database reachable from the connected machine:

```bash
python -m spyre_clickhouse_ingest results --from-bundle site-a-run1.tgz --dry-run    # preview, writes nothing
python -m spyre_clickhouse_ingest results --from-bundle site-a-run1.tgz --strict --uploader <your login>
```

Exit `0` means recorded or already recorded; `1` is a failure worth retrying; `2` means rejected; `3` means incomplete.
The JSON report lists `run_id`, `artifact_id`, `bundle_sha256`, `status` (`ingested`, `duplicate` or
`would-ingest`) and, when recorded, `trust` and `receipt`. `--channel-trust ci|lab|external` is for the operator
running the ingest; see [Trust](#trust-and-what-a-site-cannot-claim).

### Results inbox

If your deployment provides an inbox, upload the sealed file and let the relay record it:

```bash
python -m spyre_clickhouse_ingest results --upload site-a-run1.tgz --wait
```

`--wait` reports `ingested` with a link, or `rejected` with the reason. The inbox location and credentials come from the
deployment (`--base-url`, `--repo`, `--root`, or the matching environment variables); the site never needs them. A rejected
bundle is moved aside with a `REJECTED.json` that states the reason.

### Which bundle kinds the connected side accepts

| Kind | Who creates it | Notes |
|---|---|---|
| `junit`, `vllm`, `sendnn`, `benchmarks`, `tables` | a site or anyone | available |
| `bench` (results with repeats, placement and environment, from `spyre-bench`) | a site or anyone | available; [details](../../clickhouse-ingest/docs/bench-bundles.md). A site seals with `--bench-dir`, and the upload is stored as `external` unless the channel says otherwise |
| `bench-definitions`, `bench-qualification`, `bench-placement` | the program's CI | not for sites: only a CI-trusted channel ingests them, and the inbox refuses them |
| `bench-invalidation` | the program's operators | not for sites |
| comparison references | the program's operators | planned; not accepted yet |

## Step 9: verify and keep the receipt

A receipt is what lets the site prove that its results arrived and match.

| Receipt item | Where it comes from | Site checks against |
|---|---|---|
| Receipt id | `RCPT-XXXX-XXXX-XXXX-XXXX`, printed on stderr and in the CLI report's `receipt` by `--from-bundle` | recomputed by the site, below |
| Upload id | dashboard response and "My uploads" | recorded in `TRANSFER.txt` |
| Run id | dashboard response, CLI JSON report | the run page |
| Content sha256 | dashboard response; `bundle_sha256` in the CLI report | the bundle sha256 printed at seal and validate |
| State | `published`, `pending`, `rejected`, `failed`, `retracted`, `expired` | expected state |
| Uploader and time | upload record | custody notes |

If the content sha256 does not equal the bundle sha256 the site kept, the file changed in transit or was repacked.
Stop and re-transfer.

After a `published` upload, open the subject's page and confirm the run is listed with the expected test type and
counts. A `pending` upload is not recorded until approved; it expires if nobody decides (the deployment sets the time).
The receipt id is the first 16 hex digits of sha256(`<bundle sha256>`, a NUL byte, `<upload id>`), grouped in fours. The
upload id is whatever was passed as `--bundle-prop upload_id=<id>` (empty otherwise), so anyone holding the bundle
digest and the upload id can recompute it. A duplicate or `--dry-run` prints no receipt. The receipt is a
reference for asking about a run; it is not a signature.

## Corrections, duplicates and retraction

- **Sending the same file again** is safe on the inbox path and with `--from-bundle` (the bundle's digest is recognized
  as already recorded, exit `0`). The dashboard records every upload as a new run and only **warns** when the
  content is identical to an earlier upload; heed the warning and do not upload it.
- **A corrected bundle is a new bundle.** Seal the corrected results once, upload it, then retract the earlier
  upload so its run leaves the rollups. Retraction is by the uploader who sent it, or an administrator, and needs a reason.
  A retracted upload stays listed with its bundle; nothing is deleted.
- **Planned:** a way to tie a corrected bundle to the one it replaces, and to refuse an identical re-upload.
- **Rounds.** If the deployment enables it, extra files of the same run, format, test type and architecture can be added
  as a new round to an existing run instead of a new run.
- Retraction applies to uploads made through the dashboard. A bundle recorded by the relay or by the command line has no
  upload record to retract; ask an administrator.

## The bundle: layout and manifest

```text
bundle.json              the manifest (schema_version 1)
results/                 flat: the files that are ingested (extension depends on kind)
attachments/             optional: logs, command lines, environment notes; kept, never ingested; every file must be listed in files[]
```

A folder and a `.tgz` of the same folder are the same bundle. Members must be regular files inside the folder.

Minimal manifest:

```json
{"artifact_id": "<uuid>", "test_type": "regression"}
{"image": "<repo>@sha256:<64 hex>", "arch": "x86_64", "test_type": "perf", "kind": "vllm"}
```

The sealer writes the rest. The fields:

| Key | Meaning |
|---|---|
| `schema_version` | `1`; a newer version is refused |
| `kind` | `junit` (default), `vllm`, `sendnn`, `benchmarks`, `tables` |
| `artifact_id` or `image` plus `arch` | the subject; both must name the same artifact if both are given |
| `create_artifact` | `{component, name?}`: record an image the program did not build; needs `image` and `arch` |
| `component`, `arch`, `test_type` | suite owner, architecture, tier |
| `run_key` | normally left out; a Jenkins-style key is refused on the dashboard because every upload is its own run |
| `started_at`, `ended_at` | ISO-8601 with a zone; from the JUnit suites, `--triggered-at` / `--ended-at`, or absent |
| `runner`, `env`, `notes` | free text maps and notes; `runner` holds the sealing machine's host and user unless `--redact-host` / `--redact-user` leave them out |
| `trust` | `lab` or `external` (default `external`; `ci` is refused); a claim, see [Trust](#trust-and-what-a-site-cannot-claim) |
| `producer` | `{name, version}` of the kit that sealed it, read by `results --check <bundle>` |
| `uploader` | leave unset |
| `sealed_at`, `tags`, `tag_family`, `run_url`, `workflow`, `jenkins_run_key`, `artifact` | optional; a site leaves them out (`artifact` is an older spelling of `artifact_id`/`image`; tags are for CI run keys only) |
| `files` | list of `{path, sha256}` for every file except `bundle.json`; when present, a file that is missing, extra or changed rejects the bundle |
| `perf` | `vllm` kind only: model, parallelism and shape context |

The schema is closed: any key it does not list (for example a site identifier) is rejected, including when added by hand.

The bundle sha256 is `sha256` over lines of `path`, a NUL, the file's sha256 and a newline, for every file sorted by
path with `bundle.json` included. It is self-attested: the sealer writes it, so it detects accidental change only.

## Trust and what a site cannot claim

- **The site labels; the connected side decides.** `--seal-trust lab|external` (default `external`) is written
  as `trust` in `bundle.json`. It is a claim. On `--from-bundle` the recorded level is the lower of the bundle's label
  and the level the receiving channel assigns, stored as `run.trust` on the run's verdict. A bundle can lower its
  trust, never raise it; `ci` is refused in a bundle. Treat results from a site as outside results: stored and
  comparable, but not baselines. The other protection is review before publishing (see Step 8).
- **`--channel-trust ci|lab|external`** is the trusted-operator flag of `--from-bundle` and of the direct vllm, sendnn,
  benchmarks and tables ingests. Whoever can run the ingest can assert a level, as they can already write the database.
  It is an error when sealing, uploading, with `--validate-only`, with `--check`, and on the direct JUnit path.
  A labelled bundle ingested with no channel level is recorded as `external`.
- **Claims about exclusive use of the machine are claims.** A site may write "ran alone on the machine"; the connected
  side cannot verify it.
- **Do not set reserved keys.** Keys that look like server-set fields (anything under `run.` or `ref.`) are not accepted
  from a bundle. `run.trust` in benchmark or table rows (`props`, `run_props`), in `--bundle-prop` and in
  `--result-prop` is refused.
- **Comparison data** generated offline (for example outputs from a reference implementation, or published figures) is
  a separate, planned kind of upload. Do not send it as measurements.
- **Nothing gates.** Results from a site never block a build, a pull request or a promotion.

### Trust semantics

These describe what the code does today; some of it may change.

- **Upgrade receivers first.** Every bundle the sealer now writes carries `trust` (default `external`) and `producer`,
  and the bundle schema rejects keys it does not list. An ingest that predates this change refuses such a bundle
  (`results --check <bundle>` says `fields this ingest does not know` and exits `2`). Upgrade the connected side
  before any site seals with the new kit; there is no flag to omit the keys.
- **Appending to a run lowers trust, partially.** An `--append-to-run` round reads `run.trust` from the run's published
  earlier verdicts (retracted rounds do not count) and records the lowest of those and its own. Earlier rows that carry
  no `run.trust` are ignored, not treated as the lowest or highest level. If no earlier row has one, the round's own
  value is recorded as it is.
- **No `run.trust` means not labelled, not trusted.** A bundle with neither a label nor a channel level records no
  `run.trust`. Readers must treat a missing value as unknown, never as trusted.

## Time and clocks

- Timestamps in a manifest are UTC (`...Z`) or carry an explicit offset; one without a zone is rejected.
- Test frameworks often write a timestamp **without** a zone. The sealer reads it as **UTC**, whatever the sealing
  machine's zone, and prints `[warn] N JUnit timestamp(s) carry no time zone; read as UTC` whenever it takes a time from
  such a file. The dashboard reads a zone-less file timestamp the same way. If the tests ran in another zone, the
  recorded time is wrong by the offset: pass the real times with `--triggered-at` and `--ended-at`
  (ISO-8601 **with a zone**), or make the test run write zoned timestamps.
- **Check the site clock** before a run (`date -u`) against a trusted reference and note the offset in `TRANSFER.txt`.
  The connected side does not check for skew; a skew warning is **proposed**.
- `ended_at` is the end of the last JUnit suite (its start plus its `time`), or `--ended-at`. It is not the sealing time.

Worked example. A JUnit file says `<testsuite timestamp="2026-11-13T10:00:00" time="5">` with no zone. The tests ran
in a zone 5 hours 30 minutes ahead of UTC, so the real start was `2026-11-13T04:30:00Z`. Sealing that file, on any
machine, gives `started_at` `2026-11-13T10:00:00Z` (5.5 hours late) and `ended_at` `2026-11-13T10:00:05Z`. To
record the real times:

```bash
SEAL --offline --out site-a-run1.tgz ... --triggered-at 2026-11-13T10:00:00+05:30 --ended-at 2026-11-13T10:00:05+05:30
```

`--triggered-at 2026-11-13T10:00:00` (no zone) is refused:
`--triggered-at '2026-11-13T10:00:00' is not an ISO-8601 time with a zone`.

## Security notes

What a bundle **contains**:

- every result file you gave it, byte for byte;
- everything under `attachments/`, byte for byte;
- the sealing machine's host name and user name (`runner`), unless you passed `--redact-host` / `--redact-user` (the `.tgz` itself holds no owner names or times);
- whatever your results contain: JUnit files may carry a `hostname` attribute, logs may carry paths, user names,
  addresses, environment variables and command lines.

What a bundle **does not contain**: credentials (the sealer reads none and writes none), the subject itself, the datasets or
the model weights, or anything outside the files you passed. It makes no network connection.

Handling:

- Review attachments and result files for secrets and names before sealing; the sealer does not scrub them. `--redact-host` blanks only the JUnit `testsuite` `hostname` attribute.
- The bundle is **not secret and not tamper-proof.** The sha256 values (`files[]` and the bundle digest) are written by the
  sealer into the bundle, so they detect accidental corruption only: anyone who edits a file can recompute them. They do not
  authenticate who made it. The uploader's identity and your custody notes are the authenticity record. Signing is not
  implemented. The `trust` label and the receipt id are equally self-attested.
- Treat a bundle from a site as untrusted input on the connected side. The ingest refuses archive members with an
  absolute path, `..`, links or devices, but it does not limit the unpacked size or file count and does not guard
  XML parsing against hostile DOCTYPE or entity content; any such limits in the dashboard or relay are outside this
  repository. Do not unpack a bundle by hand outside a scratch directory.
- Do not send customer or partner names in free-text fields unless the uploader asks for them; use the site identifier.

## Versioning and compatibility

| Item | Rule |
|---|---|
| Bundle `schema_version` | `1` today. The ingest refuses a bundle with a larger number than it knows. Older bundles keep working. |
| Kit version | pin by range (for example `>=0.1,<0.2`), never by commit. A kit older than the connected side's ingest is fine; a newer kit that writes a newer `schema_version` is rejected until the connected side upgrades. |
| Unknown manifest keys | rejected. The kit now always writes `trust` and `producer`, so an older ingest refuses every new bundle: upgrade the connected side first. |
| Compatibility check | `results --check <bundle>` reports `schema_version`, unknown keys and whether the bundle validates; exit `2` when not compatible. Identity, plan and pack versions are not checked. |
| Order of upgrades | upgrade the **connected side first**, then ship a new kit to sites. Never the reverse. |
| Changing a benchmark's identity attributes | starts a new series; do not "fix" a case by editing its parameters at the site. |

Before every campaign: take the connected side's current kit and the version range it supports, ship that, and record the
version in `TRANSFER.txt`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `bundle.json: needs artifact_id or image` | no subject given | add `--artifact id:<uuid>` or `--artifact image:<repo>@sha256:<digest>` with `--arch` |
| `bundle.json.arch: 'arm64' is not one of [...]` | `--arch` omitted and the sealing host reports an unsupported name | pass `--arch` explicitly |
| `sha256 mismatch: [...]` | a file changed after sealing, or a transfer corrupted it | re-copy the file; if it still fails, re-seal from the original results (same host and kit give the same digest) |
| `listed file(s) missing: [...]` (exit 3) | a file was dropped when copying a folder | copy the whole folder, or send the `.tgz` |
| `file(s) not in files[]` | something (usually a log under `attachments/`) was added to the folder after sealing | remove it and ship it beside the bundle, or list it in `files[]` (see Step 4) |
| `[warn] not recorded in a bundle: --x` | you passed a flag the sealer does not have (for example `--attach`); it had **no effect** | remove the flag; see the proposed items in the capability table |
| `ambiguous option: --redact could match ...` (exit 2) | an abbreviated flag | write `--redact-host` and/or `--redact-user` |
| `--channel-trust applies to --from-bundle ...` | the flag was given while sealing, uploading or validating | drop it; it belongs to recording a bundle |
| `--check stands alone` / `--check wants a bundle path` | `--check` combined with another mode, or given an empty value | run `results --check [<bundle>]` by itself |
| `fields this ingest does not know` or `schema_version ... is newer` (from `--check`, exit 2) | the connected ingest is older than the kit | upgrade the connected side, or re-seal with an older kit |
| `pip` refuses the wheel (`requires a different Python`) | the wheel needs Python 3.11 or newer | use the two-file kit (`python offline.py`) |
| `unknown key 'site'` (or any key) | a field the schema does not have, or an ingest older than the kit (`trust`, `producer`) | remove it; for `trust`/`producer` upgrade the connected side |
| `<name>: unsafe member` | the archive holds a path outside the folder, a link or a device | re-pack from inside the folder with `tar -czf x.tgz *` |
| `a junit bundle holds results/*.xml only` | results of another kind in the folder | one kind per bundle |
| `no <testcase> in any results/*.xml` | empty or wrong XML | check the test run actually produced results |
| `schema_version 2 is newer than this ingest` | the kit is newer than the connected side | upgrade the connected side, or use an older kit |
| `artifact_id '...': not recorded` / `image '...': not recorded` | the subject is not known | check the id or digest; for an image the program did not build, supply the owning component so it can be recorded |
| `bundle.json names uploader 'x', not you` | `uploader` is set in the manifest | remove the key |
| `run key '...' names a CI run` | `run_key` set to a CI-style key | remove `run_key` |
| `identical to upload <id> ...` (warning) | the same content was uploaded before | do not upload it again; retract the earlier one only if it was wrong |
| `run ... is already recorded by another run or bundle` | a hand-set `run_key` was reused with different content (a default key includes the digest, so a bundle with different content is a new run instead) | remove the hand-set `run_key`, or decide which bundle is right and retract or ask an administrator |
| upload stays `pending` | the deployment holds outside uploads for review | ask a reviewer; it expires if nobody decides |
| `413 upload is over N bytes` | bundle or unpacked size over the deployment cap | remove traces, send them separately, or ask for a larger cap |
| `this form was already submitted` | double submit | open "My uploads"; it is probably there |
| `--wait` prints `waiting` | the relay has not processed it yet | wait; check again later; the bundle is safe in the inbox |
| `rejected` with a reason | the relay refused it | read `REJECTED.json`; fix and send a corrected bundle |
| exit `1` from `--from-bundle` | the database was unreachable or a write failed | retry; if it failed part-way, check the run page before retrying, since some rows may exist |

## FAQ

**Can the site upload directly?** No, by definition. The site seals; the connected side uploads.

**Do we need an account on the dashboard?** The site does not. The uploader does.

**Which command works today for results with repeats and placement?** Seal an engine results directory with
`SEAL --offline --bench-dir <results dir> --arch <arch> --out <bundle>` (kind `bench`), or run
`spyre-bench upload <results dir>` on the connected side; both drop the keys the receiving channel owns. Do not
hand-write the manifest. Raw output that has no engine Result can still use `benchmarks` or `tables`.

**Can we seal on a machine other than the one that ran the tests?** Yes, if the results are copied there intact. The sealing
machine's zone no longer matters (zone-less JUnit times are read as UTC); pass `--triggered-at` and `--ended-at` with
an explicit zone if the tests ran in another zone. The sealing host is recorded in `runner` unless redacted.

**We sealed twice by mistake.** On the same host with the same kit and flags the two bundles are identical; send one. If they differ (another host, kit version or flags), send only one; if both were uploaded, retract one.

**Can the same results be used by several programs?** A bundle is recorded once. Sending it to two different
databases is the connected side's business.

**Can we see our results afterwards?** Through the uploader. The site has no access.

**How do we know a result is not a baseline?** Results from a site are recorded as `external` or `lab`, never `ci`, but a missing `run.trust` means not labelled, not trusted. Ask the uploader how outside uploads are handled; they should be held for review before publishing.

**Can the digest prove the bundle was not altered on purpose?** No. It is self-attested and catches accidental corruption only.

**Is a bundle encrypted?** No. Encrypt the media if your rules require it; the connected side decrypts before upload.

**What goes in the free-text "on behalf of" field?** The site identifier you agreed with the uploader.

**What if the machine has no Python 3.9 at all, or only 3.9 or 3.10?** The two-file kit runs on 3.9 and newer; the wheel needs 3.11 or newer. If there is no suitable Python, seal on another machine after copying the results, and pass
the real run time with `--triggered-at` and an explicit zone.

**What is the smallest thing that can go wrong in transit?** A repacked archive. If the bundle sha256 differs from the
one you recorded, nothing else matters; re-send the original file.
