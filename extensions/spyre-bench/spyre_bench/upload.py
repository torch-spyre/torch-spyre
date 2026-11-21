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

"""spyre-bench upload: seal an engine results directory (or take a sealed bundle) and send it to the results inbox.

  spyre-bench upload RESULTS_DIR_OR_BUNDLE [--arch A] [--apply [--wait [MIN]]]

Dry run by default: the bundle is sealed and checked and the command prints what would be sent, the run it keys and
a receipt id, without a network call. --apply puts it in the Artifactory inbox, where the relay records it; --wait
polls for the relay's outcome. The engine never writes ClickHouse rows; sealing and the transfer are the ingest's
offline module (the adapter), so a lab host needs no database client.
"""

import argparse
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

from spyre_bench import cliutil, ingest_adapter
from spyre_bench.cliutil import UsageError

UPLOAD_SCHEMA = "bench.upload/1"
TEST_TYPES = ("perf", "quality")
SEAL_TRUST = ("lab", "external")


def is_bundle(path):
    p = Path(path)
    return (p.is_file() and p.name.endswith((".tgz", ".tar.gz"))) or (p / "bundle.json").is_file()


def run_arch(directory):
    """The arch a run.json records, else None: results measured elsewhere are never assumed to be the host's."""
    try:
        arch = json.loads((Path(directory) / "run.json").read_text()).get("arch")
    except (OSError, ValueError, AttributeError):
        return None
    return arch if isinstance(arch, str) and arch else None


def seal(off, a, source):
    """The bundle path: the source itself, or a run directory sealed into --out (default RESULTS_DIR/upload)."""
    if is_bundle(source):
        return Path(source)
    home = Path(source).parent if Path(source).is_file() else Path(source)
    out = Path(a.out) if a.out else home / "upload" / "bundle.tgz"
    arch = a.arch or run_arch(source)
    if not arch:
        raise UsageError("say where the results were measured: --arch x86_64|s390x|ppc64le (run.json records none)")
    argv = ["--offline", "--bench-dir", str(source), "--arch", arch, "--out", str(out),
            "--trigger-type", a.test_type or "perf", "--seal-trust", a.seal_trust or "external"]  # fmt: skip
    for flag, on in (("--redact-host", a.redact_host), ("--redact-user", a.redact_user)):
        if on:
            argv.append(flag)
    if a.ended_at:
        argv += ["--ended-at", a.ended_at]
    return off.write(off.parser().parse_args(argv))


def transfer_args(a):
    return SimpleNamespace(base_url=cliutil.pick(a.base_url, "ARTIFACTORY_BASE_URL", ""),
                           repo=cliutil.pick(a.repo, "ARTIFACTORY_V2_RESULTS_REPO", ""),
                           root=cliutil.pick(a.root, "ARTIFACTORY_V2_RESULTS_ROOT", ""),
                           token=cliutil.pick(a.token, "ARTIFACTORY_TOKEN", ""), inbox=a.inbox, as_folder=False,
                           wait=(a.wait if a.wait is not None else 0))  # fmt: skip


def describe(off, path):
    with off.opened(Path(path)) as root:
        meta, sha, content = off.check(root), off.digest(root), off.content_digest(root)
        cases = sorted((root / "results").glob("*.json")) if meta.get("kind") == "bench" else []
        repeats = sum(len(json.loads(c.read_text())["status"]["repeats"]) for c in cases)
    return {"kind": meta.get("kind", "junit"), "artifact_id": meta.get("artifact_id", ""), "arch": meta.get("arch", ""),
            "run_key": meta.get("run_key") or f"manual:<uploader>:{sha}", "bundle_sha256": sha,
            "content_sha256": content, "trust_label": meta.get("trust", ""), "cases": len(cases), "repeats": repeats,
            "receipt": off.receipt_id(sha)}  # fmt: skip


SEAL_ONLY = (("arch", None), ("test_type", None), ("seal_trust", None), ("redact_host", False),
             ("redact_user", False), ("ended_at", None), ("out", None))  # fmt: skip


def execute(a):
    if a.wait is not None and not a.apply:
        raise UsageError("--wait needs --apply")
    if a.out and not a.out.endswith((".tgz", ".tar.gz")):
        raise UsageError("--out is a .tgz file to write (a folder cannot be written twice)")
    if is_bundle(a.source):
        ignored = [f"--{k.replace('_', '-')}" for k, default in SEAL_ONLY if getattr(a, k) not in (default, "", None)]
        if ignored:
            raise UsageError(
                f"{', '.join(ignored)} apply when sealing a results directory; {a.source} is already sealed"
            )
    if a.apply:
        cfg = transfer_args(a)
        missing = [
            flag for flag, v in (("--base-url", cfg.base_url), ("--repo", cfg.repo), ("--root", cfg.root)) if not v
        ]
        if missing:
            raise UsageError(
                f"{', '.join(missing)} (or ARTIFACTORY_BASE_URL, ARTIFACTORY_V2_RESULTS_REPO, ARTIFACTORY_V2_RESULTS_ROOT) not set"
            )
    if a.via != "inbox":
        raise UsageError(
            "--via api: the dashboard API takes a browser login only, so there is no programmatic route yet; "
            "use the inbox (the default) or the dashboard upload page"
        )
    if not Path(a.source).exists():
        raise UsageError(f"{a.source}: no such file or directory")
    try:
        off = ingest_adapter.offline()
    except ingest_adapter.IngestUnavailable as e:
        raise UsageError(str(e)) from None
    try:
        path = seal(off, a, a.source)
        doc = {"upload_schema": UPLOAD_SCHEMA, "bundle": str(path), **describe(off, path)}
        if doc["kind"] in off.ADMIN_KINDS:
            raise off.BundleError(f"a {doc['kind']} bundle is ingested by its CI job, not uploaded to the inbox")
        if not a.apply:
            doc["status"] = "dry-run"
            doc["next"] = "re-run with --apply to send it to the inbox"
            emit(doc, a)
            return cliutil.EXIT_OK
    except off.BundleError as e:
        return fail(off, a, e)
    try:  # only the transfer: a local sealing error is an I/O failure, not an inbox failure
        sent = json.loads(off.upload(path, transfer_args(a)))
    except off.BundleError as e:
        return fail(off, a, e)
    except (urllib.error.URLError, OSError) as e:
        emit({"upload_schema": UPLOAD_SCHEMA, "status": "failed", "reason": f"inbox: {e}"}, a)
        return cliutil.EXIT_STORE
    doc.update(sent)
    emit(doc, a)
    if sent.get("status") == "rejected":
        return cliutil.EXIT_INVALID
    return cliutil.EXIT_STORE if sent.get("status") == "waiting" else cliutil.EXIT_OK


def fail(off, a, e):
    emit({"upload_schema": UPLOAD_SCHEMA, "status": off.STATUS[e.code], "reason": str(e)}, a)
    config = "no Artifactory token" in str(e)  # a missing credential is configuration, not a transfer failure
    return cliutil.EXIT_INVALID if config or e.code in (off.REJECTED, off.INCOMPLETE) else cliutil.EXIT_STORE


def emit(doc, a):
    if a.json:
        print(json.dumps(doc, indent=2, sort_keys=True))
        return
    rows = [(k, doc[k]) for k in ("status", "reason", "kind", "bundle", "artifact_id", "arch", "trust_label", "run_key", "cases", "repeats", "bundle_sha256",
                                  "content_sha256", "url", "at", "receipt", "next") if doc.get(k) not in (None, "")]  # fmt: skip
    print("\n".join(f"{k:<15} {v}" for k, v in rows))


def parser():
    ap = argparse.ArgumentParser(prog="spyre-bench upload", description=__doc__.split("\n")[0])
    ap.add_argument("source", help="an engine results directory (its result.json files), or a sealed bundle")
    ap.add_argument("--apply", action="store_true", help="send it (default: seal, check and print only)")
    ap.add_argument("--wait", nargs="?", type=int, const=40, default=None, metavar="MIN",
                    help="after --apply, wait (default 40 min) for the relay's outcome")  # fmt: skip
    ap.add_argument("--via", default="inbox", help="inbox (the default); api is not available yet")
    ap.add_argument("--arch", help="where it ran (default: run.json; required when it records none)")
    ap.add_argument("--test-type", choices=TEST_TYPES, help="the tier (default perf)")
    ap.add_argument("--seal-trust", choices=SEAL_TRUST,
                    help="where the results came from; the receiving channel decides the trust it records")  # fmt: skip
    ap.add_argument("--redact-host", action="store_true", help="leave the sealing host out of the bundle")
    ap.add_argument("--redact-user", action="store_true", help="leave the sealing user out of the bundle")
    ap.add_argument("--ended-at", help="ISO-8601 end time with a zone (UTC)")
    ap.add_argument(
        "--out",
        help="the sealed bundle to write, a .tgz (default RESULTS_DIR/upload/bundle.tgz, beside a single Result file)",
    )
    ap.add_argument("--inbox", default="inbox", help="under the results root")
    ap.add_argument("--base-url", help="ARTIFACTORY_BASE_URL")
    ap.add_argument("--repo", help="ARTIFACTORY_V2_RESULTS_REPO")
    ap.add_argument("--root", help="results folder template with {arch}: ARTIFACTORY_V2_RESULTS_ROOT")
    ap.add_argument("--token", help="ARTIFACTORY_TOKEN, else ~/.netrc")
    ap.add_argument("--json", action="store_true")
    return ap


def main(argv=None):
    return cliutil.run_main(execute, parser(), argv)


if __name__ == "__main__":
    sys.exit(main())
