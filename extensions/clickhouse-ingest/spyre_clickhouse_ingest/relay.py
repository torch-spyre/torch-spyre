# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Record the results bundles uploaded to the Artifactory inbox in spyre_v2.

    python -m spyre_clickhouse_ingest relay [--apply] [--inbox inbox] [--processed processed] [--rejected rejected]

A bundle (this extension's README, "Offline results bundles") is a
.tgz, or a folder whose bundle.json has arrived, under
    <repo>/next/<arch>/v2-results/<inbox>/<artifact_id or sha256-<digest>>/<bundle-name>[.tgz]
Each is recorded with `results --from-bundle --strict` (exit 0 ok, 1 retry, 2 rejected,
3 incomplete), then moved to <processed>/ or <rejected>/ (+ REJECTED.json) under the same relative path. A bundle
that cannot be ingested yet stays in the inbox, so every run is safe to repeat: the ingest itself
is idempotent. Without --apply it ingests with --dry-run and moves nothing.
Each ingested bundle also gets one `published` row in result_uploads (UploadLog), written before
the move, so the dashboard's retract and the published views see a relay upload like any other.
`--backfill-processed` writes the rows that bundles already in processed/ lack.
Artifactory: ARTIFACTORY_TOKEN (bearer, or with ARTIFACTORY_USER basic), ARTIFACTORY_BASE_URL,
ARTIFACTORY_V2_RESULTS_REPO and ARTIFACTORY_V2_RESULTS_ROOT (a path template with {arch}); ClickHouse: CLICKHOUSE_*.
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ENV_BASE_URL, ENV_REPO, ENV_ROOT = (
    "ARTIFACTORY_BASE_URL",
    "ARTIFACTORY_V2_RESULTS_REPO",
    "ARTIFACTORY_V2_RESULTS_ROOT",
)
# The deployment's results folder, a path template with {arch}; set by main() from $ARTIFACTORY_V2_RESULTS_ROOT.
ROOT = "{arch}/v2-results"
OK, FAILED, REJECTED, INCOMPLETE = 0, 1, 2, 3
# Ingested, but its result_uploads row cannot be written; the bundle stays in the inbox.
UNRECORDABLE = "unrecordable"
# The fingerprint of a folder bundle's listing at the previous poll, on its bundle.json.
LISTING_PROP = "v2relay.listing"
CHANNEL = "relay"
# Accepts no CI-looking run key: none starts with '#'. The default when no --trusted-job-prefix is given, so an
# uploader cannot place a bundle under a CI run.
NO_JOB_KEYS = "#"


def upload_id_of(run_id: str) -> str:
    """A uuid7-shaped id fixed by the run: the dashboard's routes only accept uuid7, and a repeat
    (or a race) lands on the same upload in v_result_uploads."""
    b = bytearray(hashlib.sha256(f"v2-results-relay:{run_id}".encode()).digest()[:16])
    b[6] = 0x70 | (b[6] & 0x0F)
    b[8] = 0x80 | (b[8] & 0x3F)
    return str(uuid.UUID(bytes=bytes(b)))


def trust_of(uploader: str, ci_accounts) -> str:
    """`ci` only for an Artifactory account listed as a CI credential, else `external`. Derived from
    the credential, never from the bundle (its run key is the uploader's own claim); no account
    maps to `lab` yet."""
    return "ci" if uploader and uploader in ci_accounts else "external"


def min_trust(uploaders, ci_accounts) -> str:
    """`ci` only when every account that wrote the bundle is a CI account."""
    return (
        "ci" if all(trust_of(u, ci_accounts) == "ci" for u in uploaders) else "external"
    )


def is_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def is_key(value: str) -> bool:
    """An inbox folder key: an artifact_id, or sha256-<digest> for a bundle naming only an image."""
    digest = value.removeprefix("sha256-")
    return is_uuid(value) or (
        digest != value
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest)
    )


class Artifactory:
    """The few Artifactory REST calls the relay makes."""

    def __init__(self, base: str, repo: str, token: str, user: str = ""):
        self.base, self.repo = base.rstrip("/"), repo
        self.auth = (
            "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()
            if user
            else f"Bearer {token}"
        )

    def url(self, path: str) -> str:
        return f"{self.base}/{self.repo}/{urllib.parse.quote(path)}"

    def _call(self, method: str, url: str, data=None, ctype: str = "") -> bytes:
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", self.auth)
        if ctype:
            req.add_header("Content-Type", ctype)
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()

    def aql(self, query: str) -> list:
        out = self._call(
            "POST", f"{self.base}/api/search/aql", query.encode(), "text/plain"
        )
        return json.loads(out)["results"]

    def find(self, criteria: dict) -> list:
        # A non-admin token must include repo, path and name.
        fields = (
            '"repo","path","name","type","created","created_by","size","actual_sha1"'
        )
        return self.aql(
            f"items.find({json.dumps({'repo': self.repo, **criteria})}).include({fields})"
        )

    def exists(self, path: str) -> bool:
        try:
            self._call(
                "GET", f"{self.base}/api/storage/{self.repo}/{urllib.parse.quote(path)}"
            )
            return True
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return False
            raise

    def prop(self, path: str, name: str) -> str:
        """One property of the item at `path`; '' when it has none."""
        url = f"{self.base}/api/storage/{self.repo}/{urllib.parse.quote(path)}?properties={name}"
        try:
            props = json.loads(self._call("GET", url)).get("properties", {})
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return ""
            raise
        return (props.get(name) or [""])[0]

    def set_prop(self, path: str, name: str, value: str) -> None:
        q = urllib.parse.quote
        self._call(
            "PUT",
            f"{self.base}/api/storage/{self.repo}/{q(path)}?properties={q(name)}={q(value)}",
        )

    def download(self, path: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self._call("GET", self.url(path)))

    def put(self, path: str, data: bytes) -> None:
        self._call("PUT", self.url(path), data, "application/json")

    def move(self, src: str, dst: str) -> None:
        q = urllib.parse.quote
        self._call(
            "POST",
            f"{self.base}/api/move/{self.repo}/{q(src)}?to=/{self.repo}/{q(dst)}",
        )


class UploadLog:
    """result_uploads in spyre_v2, as the dashboard's UploadLog writes it: one row per event,
    read back through v_result_uploads. The connection opens on first use."""

    COLUMNS = ("upload_id", "state", "actor", "artifact_id", "run_id", "format", "result_kind",
               "test_type", "arch", "filename", "size_bytes", "content_sha256", "bundle_url",
               "reason", "props")  # fmt: skip

    def __init__(self, client=None, db: str = ""):
        self._client, self.db = client, db

    def _ready(self):
        if self._client is None:
            from .client import ClickHouse, ClickHouseEnv

            db = ClickHouseEnv.target_database()
            if not db:
                raise RuntimeError("no database: set CLICKHOUSE_DB_V2")
            self._client, self.db = ClickHouse.connect(database=db), db
        return self._client

    def has(self, run_id: str) -> bool:
        """Whether a published or retracted upload names the run. A dashboard `failed`, `rejected`, `pending` or
        `expired` event is not a live upload, so it does not stop the relay recording the run it just ingested."""
        n = (
            self._ready()
            .query(
                f"SELECT count() FROM {self.db}.result_uploads WHERE run_id = {{rid:UUID}} "
                "AND state IN ('published', 'retracted')",
                parameters={"rid": run_id},
            )
            .result_rows[0][0]
        )
        return n > 0

    def kinds(self, artifact_id: str, run_id: str, test_type: str) -> tuple:
        """(result_kind, format) the ingest recorded on the run's verdict."""
        rows = self._ready().query(
            "SELECT result_kind, props['bundle_kind'] "
            f"FROM {self.db}.artifact_results WHERE artifact_id = {{aid:UUID}} "
            "AND run_id = {rid:UUID} AND test_type = {t:String} AND state != 'running' "
            "ORDER BY ts DESC, audit_timestamp DESC LIMIT 1",
            parameters={"aid": artifact_id, "rid": run_id, "t": test_type},
        ).result_rows  # fmt: skip
        if not rows:
            raise RuntimeError(f"no {test_type} verdict for run {run_id} in {self.db}")
        return rows[0][0], rows[0][1] or "junit"

    TYPES = {
        "upload_id": "UUID",
        "artifact_id": "UUID",
        "run_id": "UUID",
        "size_bytes": "UInt64",
    }

    def append(self, **event) -> None:
        """Insert the event unless the run has an upload event by now: one statement, so a stale
        has() (a retract landing in between) cannot publish a retracted run again."""
        props = {k: v for k, v in event["props"].items() if v}
        # props go in as map() of scalar parameters: a dict parameter is sent as JSON, which the server rejects.
        params = {
            c: ("" if c == "reason" else event[c]) for c in self.COLUMNS if c != "props"
        }
        for n, (k, v) in enumerate(props.items()):
            params[f"pk{n}"], params[f"pv{n}"] = k, v
        pairs = ", ".join(
            f"{{pk{n}:String}}, {{pv{n}:String}}" for n in range(len(props))
        )
        values = ", ".join(
            f"map({pairs})"
            if c == "props"
            else f"{{{c}:{self.TYPES.get(c, 'String')}}}"
            for c in self.COLUMNS
        )
        self._ready().command(
            f"INSERT INTO {self.db}.result_uploads ({', '.join(self.COLUMNS)}) SELECT {values} "
            f"WHERE NOT EXISTS (SELECT 1 FROM {self.db}.result_uploads WHERE run_id = {{run_id:UUID}} "
            "AND state IN ('published', 'retracted'))",
            parameters=params,
            settings={
                "select_sequential_consistency": 1
            },  # a retract on another replica must be visible
        )


def bundles(items: list, inbox: str) -> list:
    """The inbox's bundles, oldest first: {path, arch, rel, key, kind, uploader, created}.

    `items` are the bundle.json and .tgz files found under ROOT, any arch.
    """
    inbox_parts, root = inbox.split("/"), ROOT.split("/")
    found, folders = [], set()
    for it in items:
        parts = it["path"].split("/")
        if len(parts) < len(root) or any(
            r != "{arch}" and p != r for p, r in zip(parts, root)
        ):
            continue
        arch, rest = parts[root.index("{arch}")], parts[len(root) :]
        if rest[: len(inbox_parts)] != inbox_parts:
            continue
        rel = rest[len(inbox_parts) :]
        if it["name"] == "bundle.json":
            if not rel or "results" in rel or "attachments" in rel:
                continue
            kind, path = "dir", it["path"]
            folders.add(path)
        else:
            kind, path, rel = "tgz", f"{it['path']}/{it['name']}", [*rel, it["name"]]
        found.append({"path": path, "arch": arch, "rel": "/".join(rel), "key": rel[0] if rel else "",
                      "kind": kind, "uploader": it.get("created_by", ""),
                      "created": it.get("created", "")})  # fmt: skip
    # A .tgz inside a folder bundle (an attachment) is part of that bundle, not one of its own.
    found = [
        b for b in found
        if b["kind"] == "dir" or not any(b["path"].startswith(f + "/") for f in folders)
    ]  # fmt: skip
    return sorted(found, key=lambda b: (b["created"], b["path"]))


def age_hours(created: str, now: datetime) -> float:
    try:
        then = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return (now - then).total_seconds() / 3600


class Relay:
    def __init__(self, art: Artifactory, args, ingest=None, now=None, uploads=None):
        self.art, self.args = art, args
        self.uploads = uploads
        self.ingest = ingest or run_ingest
        self.now = now or datetime.now(timezone.utc)

    def target(self, b: dict, where: str) -> str:
        """`where`/<rel> under the bundle's arch, suffixed when a bundle already holds that name."""
        dst = f"{ROOT.format(arch=b['arch'])}/{where}/{b['rel']}"
        if self.art.exists(dst):
            dst += self.now.strftime("-dup%Y%m%dT%H%M%SZ")
        return dst

    def fetch(self, b: dict, work: Path) -> Path:
        if b["kind"] == "tgz":
            dest = work / Path(b["path"]).name
            self.art.download(b["path"], dest)
            return dest
        for rel in b["files"]:
            self.art.download(f"{b['path']}/{rel}", work / rel)
        return work

    def folder_items(self, folder: str) -> list:
        return self.art.find({"$or": [{"path": folder}, {"path": {"$match": f"{folder}/*"}}],
                              "type": "file"})  # fmt: skip

    def settled(self, b: dict, work: Path) -> str:
        """'' when the folder may be ingested, else why it waits.

        bundle.json is the commit marker. With files[] the ingest checks completeness itself;
        without, the folder's listing must be unchanged since the previous poll.
        """
        folder = b["path"]
        items = self.folder_items(folder)
        listing = sorted(
            (f"{it['path']}/{it['name']}"[len(folder) + 1 :], it.get("size", 0), it.get("actual_sha1", ""))
            for it in items
        )  # fmt: skip
        b["files"] = [rel for rel, _, _ in listing]
        # Trust follows the least trusted account that wrote any file of the folder, not only bundle.json.
        b["uploaders"] = sorted({it.get("created_by", "") for it in items})
        self.art.download(f"{folder}/bundle.json", work / "bundle.json")
        try:
            listed = "files" in json.loads((work / "bundle.json").read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            listed = True  # the ingest rejects it with the reason
        if listed:
            return ""
        fingerprint = hashlib.sha256(json.dumps(listing).encode()).hexdigest()
        marker = f"{folder}/bundle.json"
        if self.art.prop(marker, LISTING_PROP) == fingerprint:
            return ""
        if self.args.apply:
            self.art.set_prop(marker, LISTING_PROP, fingerprint)
        return "no files[]: waiting for its listing to stay unchanged across two polls"

    def one(self, b: dict, work: Path) -> dict:
        entry = {"bundle": b["path"], "uploader": b["uploader"]}
        if not is_key(b["key"]):
            why = f"path names no artifact_id or sha256-<digest>: {b['rel']!r}"
            return {**entry, **self.reject(b, why)}
        processed = self.target(b, self.args.processed)
        if b["kind"] == "dir":
            unsettled = self.settled(b, work)
            if unsettled:
                if age_hours(b["created"], self.now) > self.args.stale_hours:
                    return {
                        **entry,
                        **self.reject(
                            b, f"still changing after {self.args.stale_hours} h"
                        ),
                    }
                return {**entry, "outcome": "waiting", "reason": unsettled}
        local = self.fetch(b, work)
        argv = ["--strict", "--uploader", b["uploader"] or "unknown", "--bundle-url", self.art.url(processed),
                "--expect-key", b["key"],
                "--channel-trust", min_trust(b.get("uploaders") or [b["uploader"]], self.args.ci_account)]  # fmt: skip
        for prefix in self.args.trusted_job_prefix:
            argv += ["--trusted-job-prefix", prefix]
        if not self.args.apply:
            argv.append("--dry-run")
        code, report = self.ingest(local, argv)
        entry.update(report=report, code=code)
        if code == OK:
            if self.args.apply:
                if self.uploads is not None:
                    # Before the move: a bundle whose row failed stays in the inbox and is retried.
                    try:
                        entry["upload"] = self.record_upload(
                            b, report, local, self.art.url(processed)
                        )
                    # SystemExit: ClickHouse.connect() exits on bad CLICKHOUSE_* settings.
                    except (Exception, SystemExit) as err:  # noqa: BLE001 -- any write error is a retry, not the bundle's fault
                        why = f"ingested, but its result_uploads row cannot be written: {err!r}"
                        return {**entry, "outcome": UNRECORDABLE, "reason": why}
                self.art.move(b["path"], processed)
            return {
                **entry,
                "outcome": report.get("status", "ingested"),
                "to": processed,
            }
        why = report.get("reason", f"results --from-bundle exited {code}")
        stale = age_hours(b["created"], self.now) > self.args.stale_hours
        if code == REJECTED or (code == INCOMPLETE and stale):
            return {**entry, **self.reject(b, why, report)}
        # FAILED (ClickHouse unreachable, a write error) never rejects: the bundle is not at fault.
        return {
            **entry,
            "outcome": "waiting" if code == INCOMPLETE else "failed",
            "reason": why,
        }

    def record_upload(
        self, b: dict, report: dict, local: Path, url: str, **props
    ) -> str:
        """Append the bundle's `published` event unless the run has one: 'recorded' or 'exists'.

        A duplicate ingest (the run is already in) still gets its row when it has none, which
        covers a crash between the ingest and this write.
        """
        rid = report["run_id"]
        if self.uploads.has(rid):
            return "exists"
        result_kind, fmt = self.uploads.kinds(
            report["artifact_id"], rid, report["test_type"]
        )
        size = local.stat().st_size if local.is_file() else sum(
            p.stat().st_size for p in local.rglob("*") if p.is_file()
        )  # fmt: skip
        self.uploads.append(
            upload_id=upload_id_of(rid), state="published", actor=b["uploader"] or "unknown",
            artifact_id=report["artifact_id"], run_id=rid, format=fmt, result_kind=result_kind,
            test_type=report["test_type"], arch=report["arch"], filename=Path(b["rel"]).name,
            size_bytes=size, content_sha256=report["bundle_sha256"], bundle_url=url,
            props={"channel": CHANNEL, "trust": report.get("trust") or min_trust(b.get("uploaders") or [b["uploader"]], self.args.ci_account),
                   "ingest_status": report.get("status", ""), "bundle_key": b["key"],
                   "relay_run": os.environ.get("BUILD_URL", ""), **props},
        )  # fmt: skip
        return "recorded"

    def backfill(self, work: Path) -> list:
        """Rows for bundles already in processed/ whose run has none (ingested before rows existed).

        The ingest answers `duplicate` for each (read-only); nothing is moved. Without --apply
        it only reports which rows it would write.
        """
        found = self.art.find(
            {"path": {"$match": ROOT.format(arch="*") + "/*"},
             "$or": [{"name": "bundle.json"}, {"name": {"$match": "*.tgz"}},
                     {"name": {"$match": "*.tar.gz"}}]}
        )  # fmt: skip
        out = []
        for i, b in enumerate(bundles(found, self.args.processed)):
            entry = {"bundle": b["path"], "uploader": b["uploader"]}
            try:
                entry.update(self.backfill_one(b, work / str(i)))
            except (Exception, SystemExit) as err:  # noqa: BLE001 -- one bundle's error must not stop the rest
                entry.update(outcome=UNRECORDABLE, reason=repr(err))
            print(
                f"[{entry['outcome']}] {b['path']}"
                + (f": {entry['reason']}" if entry.get("reason") else "")
            )
            out.append(entry)
        return out

    def backfill_one(self, b: dict, work: Path) -> dict:
        if not is_key(b["key"]):
            return {
                "outcome": "skipped",
                "reason": f"path names no artifact_id or sha256-<digest>: {b['rel']!r}",
            }
        if b["kind"] == "dir":
            items = self.folder_items(b["path"])
            b["files"] = [
                f"{it['path']}/{it['name']}"[len(b["path"]) + 1 :] for it in items
            ]
            b["uploaders"] = sorted({it.get("created_by", "") for it in items})
        local = self.fetch(b, work)
        url = self.art.url(b["path"])
        argv = [
            "--strict",
            "--dry-run",
            "--uploader",
            b["uploader"],
            "--bundle-url",
            url,
            "--expect-key",
            b["key"],
            "--channel-trust",
            min_trust(b.get("uploaders") or [b["uploader"]], self.args.ci_account),
        ]
        for prefix in self.args.trusted_job_prefix:
            argv += ["--trusted-job-prefix", prefix]
        code, report = self.ingest(local, argv)
        if code != OK or report.get("status") != "duplicate":
            why = (
                report.get("reason")
                or f"ingest says {report.get('status', code)!r}, not an ingested run"
            )
            return {"outcome": "skipped", "reason": why}
        if self.uploads.has(report["run_id"]):
            return {"outcome": "exists"}
        if not self.args.apply:
            return {"outcome": "would-backfill"}
        return {"outcome": self.record_upload(b, report, local, url, backfilled="1")}

    def reject(self, b: dict, reason: str, report=None) -> dict:
        dst = self.target(b, self.args.rejected)
        if self.args.apply:
            self.art.move(b["path"], dst)
            note = {"reason": reason, "rejected_at": self.now.isoformat(), "report": report or {},
                    "relay_run": os.environ.get("BUILD_URL", "")}  # fmt: skip
            self.art.put(dst + ("/REJECTED.json" if b["kind"] == "dir" else ".REJECTED.json"),
                         json.dumps(note, indent=2).encode())  # fmt: skip
        return {"outcome": "rejected", "reason": reason, "to": dst}

    def run(self, work: Path) -> list:
        found = self.art.find(
            {"path": {"$match": ROOT.format(arch="*") + "/*"},
             "$or": [{"name": "bundle.json"}, {"name": {"$match": "*.tgz"}},
                     {"name": {"$match": "*.tar.gz"}}]}
        )  # fmt: skip
        out = []
        for i, b in enumerate(bundles(found, self.args.inbox)):
            try:
                entry = self.one(b, work / str(i))
            except Exception as err:  # noqa: BLE001 -- one bundle's error must not stop the rest
                entry = {"bundle": b["path"], "outcome": "failed", "reason": repr(err)}
            print(
                f"[{entry['outcome']}] {b['path']}"
                + (f": {entry['reason']}" if entry.get("reason") else "")
            )
            out.append(entry)
        return out


def run_ingest(local: Path, argv: list) -> tuple:
    """`results --from-bundle` in this interpreter's environment: (exit code, its JSON report)."""
    cmd = [
        sys.executable,
        "-m",
        "spyre_clickhouse_ingest",
        "results",
        "--from-bundle",
        str(local),
        *argv,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    sys.stderr.write(p.stderr)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    try:
        report = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        report = {}
    if p.returncode and "status" not in report:
        # No verdict on the bundle (argparse's own exit 2 on an older library, a crash): retry.
        tail = (p.stderr.strip().splitlines() or [""])[-1]
        return FAILED, {
            "reason": f"results --from-bundle exited {p.returncode}: {tail}"
        }
    return p.returncode, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--apply", action="store_true", help="write and move; default: dry run"
    )
    parser.add_argument("--base-url", default="", help=f"default ${ENV_BASE_URL}")
    parser.add_argument("--repo", default="", help=f"generic repo; default ${ENV_REPO}")
    parser.add_argument(
        "--root",
        default="",
        help=f"results folder template with {{arch}}; default ${ENV_ROOT}",
    )
    parser.add_argument(
        "--inbox", default="inbox", help="under v2-results/; may be nested"
    )
    parser.add_argument("--processed", default="processed")
    parser.add_argument("--rejected", default="rejected")
    parser.add_argument("--trusted-job-prefix", action="append", default=None,
                        help="repeatable: accept a CI-looking run key only from these job prefixes; default none: no CI-looking run key is accepted")  # fmt: skip
    parser.add_argument("--stale-hours", type=float, default=24.0,
                        help="reject a bundle still incomplete after this long")  # fmt: skip
    parser.add_argument("--no-record-uploads", action="store_true",
                        help="skip the result_uploads row (the login lacks INSERT on it)")  # fmt: skip
    parser.add_argument("--ci-account", action="append", default=[],
                        help="repeatable: an Artifactory account whose uploads are recorded as trust=ci")  # fmt: skip
    parser.add_argument("--backfill-processed", action="store_true",
                        help="instead of the inbox: write the missing result_uploads rows of bundles in processed/")  # fmt: skip
    parser.add_argument("--work", default="v2-results-work")
    parser.add_argument("--summary", default="v2-results-relay.json")
    args = parser.parse_args(argv)
    args.trusted_job_prefix = args.trusted_job_prefix or [NO_JOB_KEYS]
    token = os.environ.get("ARTIFACTORY_TOKEN", "")
    if not token:
        sys.exit("[error] ARTIFACTORY_TOKEN is unset")
    global ROOT
    ROOT = (
        args.root
        or os.environ.get(ENV_ROOT, "")
        or sys.exit(f"[error] no --root: set ${ENV_ROOT}")
    )
    base = (
        args.base_url
        or os.environ.get(ENV_BASE_URL, "")
        or sys.exit(f"[error] no --base-url: set ${ENV_BASE_URL}")
    )
    repo = (
        args.repo
        or os.environ.get(ENV_REPO, "")
        or sys.exit(f"[error] no --repo: set ${ENV_REPO}")
    )
    art = Artifactory(base, repo, token, os.environ.get("ARTIFACTORY_USER", ""))
    started = time.time()
    if args.backfill_processed and args.no_record_uploads:
        sys.exit("[error] --backfill-processed contradicts --no-record-uploads")
    uploads = (
        UploadLog()
        if (args.apply or args.backfill_processed) and not args.no_record_uploads
        else None
    )
    relay = Relay(art, args, uploads=uploads)
    entries = (relay.backfill if args.backfill_processed else relay.run)(
        Path(args.work)
    )
    counts = {}
    for e in entries:
        counts[e["outcome"]] = counts.get(e["outcome"], 0) + 1
    summary = {"apply": args.apply, "inbox": args.inbox, "counts": counts, "bundles": entries,
               "seconds": round(time.time() - started, 1)}  # fmt: skip
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"[info] {len(entries)} bundle(s): {counts or 'inbox empty'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
