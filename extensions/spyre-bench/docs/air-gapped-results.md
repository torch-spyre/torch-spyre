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

The bundle format and the commands marked **available** ship in `extensions/clickhouse-ingest`. Anything marked
**planned** or **proposed** does not exist yet; the guide mentions it only so a site can plan, and gives the
workaround that works today.

"Available" means on the main branch today. Work in progress on a separate branch (deterministic sealing, redaction,
a trust label, UTC handling, a compatibility preflight, receipts) is **planned** until it is merged; do not rely on any of
it.

> **Warning: a flag that does not exist is silently ignored.** The sealer prints
> `[warn] not recorded in a bundle: --redact` (or whichever flag) and carries on, exit code 0, producing a normal
> bundle. If you see a `[warn] not recorded in a bundle` line, **that flag had no effect.** Check the output of every
> seal for this line.

| Capability | Command or place | Status |
|---|---|---|
| Seal a bundle from JUnit, vLLM bench, perf-suite report, generic benchmark rows or per-table rows | `SEAL --offline --out <bundle> ...` | available |
| Run the sealer with the standard library only | `SEAL="python offline.py"` (one file plus `bundle.schema.json`) | available |
| Validate a bundle with no network | `results --from-bundle <bundle> --validate-only` | available |
| Put a bundle in the results inbox | `results --upload <bundle> [--wait]` | available |
| Record a bundle on the connected side | `results --from-bundle <bundle> --strict` (add `--dry-run` to preview) | available |
| Dashboard upload page: inspect, missing-field prompts, preview, publish or hold for review, retract | the dashboard's Uploads page | available (one bundle per upload) |
| Engine-level collection verbs, a version and compatibility check, receipts, batch upload, a kit build script | various | planned |
| Metadata redaction (`--redact`), size caps (`--max-bytes`), split bundles (`--split`), attaching files (`--attach`) | seal flags | proposed (none exist) |
| Deterministic sealing (the same results always give the same bundle digest) | seal | proposed (not true today, see [Step 5](#step-5-seal)) |

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

Version rule: **pin the kit by a version range, never by a commit.** Record the kit version in the
transfer notes. The upload side accepts bundles from an older kit as long as `schema_version` is not newer than
the ingest it runs; see [Versioning and compatibility](#versioning-and-compatibility).

A kit for the planned engine-level tooling (a single archive or a wheelhouse, with a version and compatibility
statement) is **planned**; a build script for it does not exist.

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
| Trust level | not set by the site | derived by the connected side from how the bundle arrives |

The manifest has no dedicated site field today, and a manifest with an unknown key is rejected. Until one exists, put
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
edit after sealing, which is allowed only by the recipes in this guide.

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

- **Seal once, then copy the file.** Sealing stamps the sealing time and the sealing machine into
  `bundle.json`, so sealing the same results twice (in different seconds, which is the usual case) gives two different digests, and the connected side treats
  them as two runs. If an upload is lost, send the same `.tgz` again. Never re-seal to retry. Making sealing
  deterministic is **proposed**.
- **Metadata about the sealing machine is recorded.** `bundle.json` gets a `runner` block with the host name and
  user name. If you must not disclose them, seal to a **folder** (`--out site-a-run1`, no `.tgz`), delete the `runner`
  key from `bundle.json`, validate the folder, and only then pack it:

  ```bash
  SEAL --from-bundle site-a-run1 --validate-only     # note the NEW bundle_sha256: the edit changed it
  # bsdtar (macOS, tested): no user or group name in the archive
  (cd site-a-run1 && tar --uid 0 --gid 0 --uname '' --gname '' -czf ../site-a-run1.tgz *)
  # GNU tar (Linux; flags from its manual, not tested here)
  (cd site-a-run1 && tar --owner=0 --group=0 --numeric-owner -czf ../site-a-run1.tgz *)
  SEAL --from-bundle site-a-run1.tgz --validate-only # same digest as the folder
  ```

  Use one of the two `tar` lines. `*` packs `bundle.json`, `results` and `attachments` if present. **A `.tgz` written
  by the sealer itself carries your user name in its tar headers**, so for a redacted bundle always pack the folder
  yourself with the flags above.

  Removing `runner` changes the bundle digest, so **do not compare it with the digest printed at seal time**: record
  the one printed by this validation. JUnit files can carry a `hostname` attribute and logs can carry names and paths;
  the sealer does not touch result files or attachments.

  > **Warning: `--redact` does not exist.** It is proposed only. `SEAL --offline ... --redact` prints
  > `[warn] not recorded in a bundle: --redact`, exits 0 and writes a bundle that still contains the host and user name.
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
The JSON report lists `run_id`, `artifact_id`, `bundle_sha256` and `status` (`ingested`, `duplicate` or
`would-ingest`).

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
| other kinds (results with repeats and placement, comparison references, definitions, qualification) | the program's operators | planned; not accepted yet |

## Step 9: verify and keep the receipt

A receipt is what lets the site prove that its results arrived and match.

| Receipt item | Where it comes from | Site checks against |
|---|---|---|
| Upload id | dashboard response and "My uploads" | recorded in `TRANSFER.txt` |
| Run id | dashboard response, CLI JSON report | the run page |
| Content sha256 | dashboard response; `bundle_sha256` in the CLI report | the bundle sha256 printed at seal and validate |
| State | `published`, `pending`, `rejected`, `failed`, `retracted`, `expired` | expected state |
| Uploader and time | upload record | custody notes |

If the content sha256 does not equal the bundle sha256 the site kept, the file changed in transit or was repacked.
Stop and re-transfer.

After a `published` upload, open the subject's page and confirm the run is listed with the expected test type and
counts. A `pending` upload is not recorded until approved; it expires if nobody decides (the deployment sets the time).
A printable receipt is **proposed**.

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
| `started_at`, `ended_at` | ISO-8601 with a zone; `ended_at` is the time of sealing, not of the run |
| `runner`, `env`, `notes` | free text maps and notes; `runner` is filled with the sealing machine's host and user |
| `uploader` | leave unset |
| `sealed_at`, `tags`, `tag_family`, `run_url`, `workflow`, `jenkins_run_key`, `artifact` | optional; a site leaves them out (`artifact` is an older spelling of `artifact_id`/`image`; tags are for CI run keys only) |
| `files` | list of `{path, sha256}` for every file except `bundle.json`; when present, a file that is missing, extra or changed rejects the bundle |
| `perf` | `vllm` kind only: model, parallelism and shape context |

Fields a future version may add (a site identifier, a producer and compatibility block) are not in the schema and are
rejected if added by hand.

The bundle sha256 is `sha256` over lines of `path`, a NUL, the file's sha256 and a newline, for every file sorted by
path with `bundle.json` included.

## Trust and what a site cannot claim

- **Trust is decided by the connected side, from how the bundle arrives, never by what the bundle says.** Treat results
  from a site as outside results: stored and comparable, but not baselines. Today the dashboard has no separate trust
  label; the protection is review before publishing (see Step 8). A stronger label is planned.
- **Claims about exclusive use of the machine are claims.** A site may write "ran alone on the machine"; the connected
  side cannot verify it.
- **Do not set reserved keys.** Keys that look like server-set fields (anything under `run.` or `ref.`) are not accepted
  from a bundle.
- **Comparison data** generated offline (for example outputs from a reference implementation, or published figures) is
  a separate, planned kind of upload. Do not send it as measurements.
- **Nothing gates.** Results from a site never block a build, a pull request or a promotion.

## Time and clocks

- Timestamps in a manifest are UTC (`...Z`) or carry an explicit offset; one without a zone is rejected.
- Test frameworks often write a timestamp **without** a zone. The sealer reads it as the **sealing machine's local
  time**, so seal on the machine that ran the tests, with its clock and time zone correct. If you seal elsewhere, pass
  the run's real start time with `--triggered-at <ISO-8601 with zone>`.
- The dashboard reads a zone-less file timestamp as UTC and says so in the preview. Do not rely on that: use
  explicit zones.
- **Check the site clock** before a run (`date -u`) against a trusted reference and note the offset in `TRANSFER.txt`.
  The connected side does not check for skew today; a skew warning for timestamps far in the past or future is **proposed**.
- `ended_at` is the sealing time. Do not use it as the run's end.

Worked example. A JUnit file says `<testsuite timestamp="2026-11-13T10:00:00">` with no zone. The tests ran in a
zone that is 5 hours 30 minutes ahead of UTC. Sealing that file gives `started_at` depending on the **sealing
machine's** zone:

| Sealing machine's zone | `started_at` written | Right? |
|---|---|---|
| the test machine's zone (UTC+05:30) | `2026-11-13T04:30:00Z` | yes |
| UTC | `2026-11-13T10:00:00Z` | no: 5.5 hours late |
| UTC-05:00 | `2026-11-13T15:00:00Z` | no: 10.5 hours late |

If you must seal elsewhere, give the real start with an explicit zone:

```bash
SEAL --offline --out site-a-run1.tgz ... --triggered-at 2026-11-13T10:00:00+05:30
```

`--triggered-at 2026-11-13T10:00:00` (no zone) is refused:
`bundle.json.started_at: '2026-11-13T10:00:00' is not an ISO-8601 time with a zone`.

## Security notes

What a bundle **contains**:

- every result file you gave it, byte for byte;
- everything under `attachments/`, byte for byte;
- the sealing machine's host name and user name (`runner`), and, in a `.tgz` the sealer wrote, the user name again in the tar headers;
- whatever your results contain: JUnit files may carry a `hostname` attribute, logs may carry paths, user names,
  addresses, environment variables and command lines.

What a bundle **does not contain**: credentials (the sealer reads none and writes none), the subject itself, the datasets or
the model weights, or anything outside the files you passed. It makes no network connection.

Handling:

- Review attachments for secrets and names before sealing; the sealer does not scrub them.
- The bundle is integrity-protected, not secret: the sha256 detects change, it does not authenticate who made it. The
  uploader's identity and your custody notes are the authenticity record. Signing is not implemented.
- Treat a bundle from a site as untrusted input on the connected side: the dashboard and the ingest refuse path
  traversal, links and devices in archives, refuse unsafe XML, and cap size; do not unpack it by hand outside a scratch
  directory.
- Do not send customer or partner names in free-text fields unless the uploader asks for them; use the site identifier.

## Versioning and compatibility

| Item | Rule |
|---|---|
| Bundle `schema_version` | `1` today. The ingest refuses a bundle with a larger number than it knows. Older bundles keep working. |
| Kit version | pin by range (for example `>=0.1,<0.2`), never by commit. A kit older than the connected side's ingest is fine; a newer kit that writes a newer `schema_version` is rejected until the connected side upgrades. |
| Unknown manifest keys | rejected. Upgrade the connected side first when a kit adds keys. |
| Other compatibility information (identity, plan and pack versions) | a compatibility check is planned; today only `schema_version` is checked |
| Order of upgrades | upgrade the **connected side first**, then ship a new kit to sites. Never the reverse. |
| Changing a benchmark's identity attributes | starts a new series; do not "fix" a case by editing its parameters at the site. |

Before every campaign: take the connected side's current kit and the version range it supports, ship that, and record the
version in `TRANSFER.txt`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `bundle.json: needs artifact_id or image` | no subject given | add `--artifact id:<uuid>` or `--artifact image:<repo>@sha256:<digest>` with `--arch` |
| `bundle.json.arch: 'arm64' is not one of [...]` | `--arch` omitted and the sealing host reports an unsupported name | pass `--arch` explicitly |
| `sha256 mismatch: [...]` | a file changed after sealing, or a transfer corrupted it | re-copy the file; if it still fails, re-seal from the original results once |
| `listed file(s) missing: [...]` (exit 3) | a file was dropped when copying a folder | copy the whole folder, or send the `.tgz` |
| `file(s) not in files[]` | something (usually a log under `attachments/`) was added to the folder after sealing | remove it and ship it beside the bundle, or list it in `files[]` (see Step 4) |
| `[warn] not recorded in a bundle: --x` | you passed a flag the sealer does not have (for example `--redact`); it had **no effect** | remove the flag and use the documented workaround |
| `pip` refuses the wheel (`requires a different Python`) | the wheel needs Python 3.11 or newer | use the two-file kit (`python offline.py`) |
| `unknown key 'site'` (or any key) | a field the schema does not have | remove it; use the transfer notes |
| `<name>: unsafe member` | the archive holds a path outside the folder, a link or a device | re-pack from inside the folder with `tar -czf x.tgz *` |
| `a junit bundle holds results/*.xml only` | results of another kind in the folder | one kind per bundle |
| `no <testcase> in any results/*.xml` | empty or wrong XML | check the test run actually produced results |
| `schema_version 2 is newer than this ingest` | the kit is newer than the connected side | upgrade the connected side, or use an older kit |
| `artifact_id '...': not recorded` / `image '...': not recorded` | the subject is not known | check the id or digest; for an image the program did not build, supply the owning component so it can be recorded |
| `bundle.json names uploader 'x', not you` | `uploader` is set in the manifest | remove the key |
| `run key '...' names a CI run` | `run_key` set to a CI-style key | remove `run_key` |
| `identical to upload <id> ...` (warning) | the same content was uploaded before | do not upload it again; retract the earlier one only if it was wrong |
| `run ... is already recorded by another run or bundle` | a hand-set `run_key` was reused with different content (a default key includes the digest, so a re-seal is a new run instead) | remove the hand-set `run_key`, or decide which bundle is right and retract or ask an administrator |
| upload stays `pending` | the deployment holds outside uploads for review | ask a reviewer; it expires if nobody decides |
| `413 upload is over N bytes` | bundle or unpacked size over the deployment cap | remove traces, send them separately, or ask for a larger cap |
| `this form was already submitted` | double submit | open "My uploads"; it is probably there |
| `--wait` prints `waiting` | the relay has not processed it yet | wait; check again later; the bundle is safe in the inbox |
| `rejected` with a reason | the relay refused it | read `REJECTED.json`; fix and send a corrected bundle |
| exit `1` from `--from-bundle` | the database was unreachable or a write failed | retry; if it failed part-way, check the run page before retrying, since some rows may exist |

## FAQ

**Can the site upload directly?** No, by definition. The site seals; the connected side uploads.

**Do we need an account on the dashboard?** The site does not. The uploader does.

**Which command works today for results with repeats and placement?** Seal the raw output with one of the existing
kinds (`benchmarks` or `tables`). A dedicated kind is planned; do not hand-write its manifest.

**Can we seal on a machine other than the one that ran the tests?** Yes, if the results are copied there intact; pass
`--triggered-at` with an explicit time zone for JUnit files.

**We sealed twice by mistake.** Send only one. If both were uploaded, retract one.

**Can the same results be used by several programs?** A bundle is recorded once. Sending it to two different
databases is the connected side's business.

**Can we see our results afterwards?** Through the uploader. The site has no access.

**How do we know a result is not a baseline?** Ask the uploader how outside uploads are handled in the deployment; today they should be held for review before publishing.

**Is a bundle encrypted?** No. Encrypt the media if your rules require it; the connected side decrypts before upload.

**What goes in the free-text "on behalf of" field?** The site identifier you agreed with the uploader.

**What if the machine has no Python 3.9 at all, or only 3.9 or 3.10?** The two-file kit runs on 3.9 and newer; the wheel needs 3.11 or newer. If there is no suitable Python, seal on another machine after copying the results, and pass
the real run time with `--triggered-at` and an explicit zone.

**What is the smallest thing that can go wrong in transit?** A repacked archive. If the bundle sha256 differs from the
one you recorded, nothing else matters; re-send the original file.
