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

# Python 3.9 hosts: no datetime.UTC, and fromisoformat() refuses a "Z".
# ruff: noqa: UP017, FURB162
"""`results` without ClickHouse: write a results bundle, validate it, upload it.

    results --offline --out <dir|file.tgz> [the usual results flags]   # write a bundle
    results --offline --upload [--wait] [the usual results flags]      # write it and upload it
    results --upload <dir|tgz> [--wait]                                # upload an existing one
    results --from-bundle <dir|tgz> --validate-only                    # check one

Standard library only (Python 3.9+), with no package import: it runs from a copied wheel
(`pip install --no-deps`) or as this file plus bundle.schema.json beside it
(`python offline.py --offline ...`). `results --from-bundle <x>` (bundle.py) is the online
half, which the v2-results relay runs. See the README, "Offline results bundles".
"""

import argparse
import base64
import contextlib
import getpass
import hashlib
import json
import netrc
import os
import platform
import shutil
import string
import socket
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as etree
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
BUNDLE_FILE = "bundle.json"
FAILED, REJECTED, INCOMPLETE = 1, 2, 3
STATUS = {FAILED: "failed", REJECTED: "rejected", INCOMPLETE: "incomplete"}
MANUAL_PREFIX = "manual:"
# kind -> the extension of its results/ files.
KINDS = {"junit": ".xml", "vllm": ".json"}
VLLM_COMPONENT = "spyre-inference"
RUN_MODES = ("latency", "throughput", "serve")
# A record carrying any of these is a vLLM bench result (native) or vLLM's pytorch format.
VLLM_SIGNATURES = (
    "avg_latency", "requests_per_second", "tokens_per_second", "request_throughput",
    "output_throughput",
)  # fmt: skip
ARTIFACT_KEYS = ("artifact_id", "image", "artifact")
BASE_URL = "https://na.artifactory.swg-devops.com/artifactory"
REPO = "sys-ai-sw-accel-team-cos-dev-generic-local"
ROOT = "zsp/next/{arch}/v2-results"
WAIT_MINUTES = 40
POLL_SECONDS = 30


class BundleError(Exception):
    """The bundle cannot be ingested; `code` says whether a retry can help."""

    def __init__(self, message: str, code: int = REJECTED):
        super().__init__(message)
        self.code = code


def schema() -> dict:
    return json.loads((Path(__file__).with_name("bundle.schema.json")).read_text())


def _date_time(value: str) -> bool:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


HEX = frozenset("0123456789abcdef")
DIGITS = frozenset("0123456789")
SLUG = frozenset(string.ascii_letters + string.digits + "._-")


def _hex(s: str, n: int) -> bool:
    return len(s) == n and set(s) <= HEX


def _uuid(s: str) -> bool:
    parts = s.split("-")
    return [len(x) for x in parts] == [8, 4, 4, 4, 12] and all(
        set(x) <= HEX for x in parts
    )


def _image(s: str) -> bool:
    ref, sep, digest = s.rpartition("@sha256:")
    return (
        bool(sep and ref)
        and "@" not in ref
        and not any(c.isspace() for c in ref)
        and _hex(digest, 64)
    )


def _jenkins_key(s: str) -> bool:
    job, sep, build = s.partition("#")
    return (
        bool(sep and job and build)
        and set(build) <= DIGITS
        and not any(c.isspace() for c in job)
    )


def _manual_key(s: str) -> bool:
    parts = s.split(":")
    who_ok = frozenset(string.ascii_letters + string.digits + "._@+-")
    token_ok = frozenset(string.ascii_letters + string.digits + "-")
    return (
        len(parts) == 3
        and parts[0] == "manual"
        and bool(parts[1] and parts[2])
        and (set(parts[1]) <= who_ok and set(parts[2]) <= token_ok)
    )


def _bundle_path(s: str) -> bool:
    if s.startswith("attachments/"):
        rest = s[len("attachments/") :]
        return bool(rest) and not any(c.isspace() for c in rest)
    rest = s[len("results/") :] if s.startswith("results/") else ""
    return "/" not in rest and any(
        rest.endswith(e) and len(rest) > len(e) for e in (".xml", ".json")
    )


# Every `pattern` bundle.schema.json uses, as plain string logic: no `re` on an air-gapped host.
# tests/test_bundle.py proves each agrees with the regex; `--from-bundle` re-checks with it.
PATTERNS = {
    "^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?$": lambda s: s
    == ""
    or _uuid(s),
    "^([^\\s@]+@sha256:[0-9a-f]{64})?$": lambda s: s == "" or _image(s),
    "^[a-z0-9][a-z0-9._-]*$": lambda s: bool(s)
    and s[0] in string.ascii_lowercase + string.digits
    and set(s) <= frozenset(string.ascii_lowercase + string.digits + "._-"),
    "^(manual:[A-Za-z0-9._@+-]+:[A-Za-z0-9-]+|[^#\\s]+#[0-9]+)$": lambda s: _manual_key(
        s
    )
    or _jenkins_key(s),
    "^[^#\\s]+#[0-9]+$": _jenkins_key,
    "^(results/[^/]+\\.(xml|json)|attachments/[^\\s]+)$": _bundle_path,
    "^[0-9a-f]{64}$": lambda s: _hex(s, 64),
    "^[0-9]+$": lambda s: bool(s) and set(s) <= DIGITS,
}


def pattern_match(pattern: str, value: str) -> bool:
    """`value` against one of the schema's patterns; True for a pattern not in PATTERNS, which
    only `--from-bundle` (with `regex`) checks."""
    check = PATTERNS.get(pattern)
    return check(value) if check else True


def slug(key: str) -> str:
    """Every run of characters outside [A-Za-z0-9._-] replaced by one "_"."""
    out, replacing = [], False
    for c in key:
        if c in SLUG:
            out.append(c)
        elif not replacing:
            out.append("_")
        replacing = c not in SLUG
    return "".join(out)


def schema_errors(
    value, node: dict, where: str = "bundle.json", match=pattern_match
) -> list:
    """`value` checked against the subset of JSON Schema bundle.schema.json uses."""
    errors = []
    if "const" in node and value != node["const"]:
        return [f"{where}: must be {node['const']!r}, got {value!r}"]
    if "enum" in node and value not in node["enum"]:
        return [f"{where}: {value!r} is not one of {node['enum']}"]
    kind = node.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int}
    if kind and not isinstance(value, types[kind]):
        return [f"{where}: must be a {kind}"]
    if isinstance(value, str):
        if len(value) < node.get("minLength", 0):
            errors.append(f"{where}: must not be empty")
        if "pattern" in node and not match(node["pattern"], value):
            errors.append(f"{where}: {value!r} does not match {node['pattern']}")
        if node.get("format") == "date-time" and not _date_time(value):
            errors.append(f"{where}: {value!r} is not an ISO-8601 time with a zone")
    if isinstance(value, list):
        if len(value) < node.get("minItems", 0):
            errors.append(f"{where}: needs at least {node['minItems']} item(s)")
        for i, item in enumerate(value):
            errors += schema_errors(item, node.get("items", {}), f"{where}[{i}]", match)
    if isinstance(value, dict):
        errors += [
            f"{where}: missing {k!r}"
            for k in node.get("required", ())
            if k not in value
        ]
        props, extra = (
            node.get("properties", {}),
            node.get("additionalProperties", True),
        )
        for k, v in value.items():
            if k in props:
                errors += schema_errors(v, props[k], f"{where}.{k}", match)
            elif extra is False:
                errors.append(f"{where}: unknown key {k!r}")
            elif isinstance(extra, dict):
                errors += schema_errors(v, extra, f"{where}.{k}", match)
    return errors


def normalized(meta: dict) -> dict:
    """bundle.json with blank artifact fields dropped, and an `artifact` id:/image: spec moved
    to artifact_id / image (refused when it disagrees with them)."""
    meta = {k: v for k, v in meta.items() if not (k in ARTIFACT_KEYS and v == "")}
    named = meta.get("artifact", "")
    for prefix, key in (("id:", "artifact_id"), ("image:", "image")):
        if isinstance(named, str) and named.startswith(prefix):
            value = named[len(prefix) :].strip()
            if meta.setdefault(key, value) != value:
                raise BundleError(f"artifact {named!r} and {key} {meta[key]!r} differ")
            del meta["artifact"]
    return meta


def specs(meta: dict) -> list:
    """(field, spec) for each way the bundle names its artifact, the id first."""
    out = (
        [("artifact_id", f"id:{meta['artifact_id']}")]
        if meta.get("artifact_id")
        else []
    )
    if meta.get("image"):
        out.append(("image", f"image:{meta['image']}"))
    if meta.get("artifact"):
        out.append(("artifact", meta["artifact"]))
    return out


def inbox_keys(meta: dict) -> set:
    """The folder names an inbox may file this bundle under: its id, or its image digest."""
    keys = {meta["artifact_id"]} if meta.get("artifact_id") else set()
    if meta.get("image"):
        keys.add("sha256-" + meta["image"].rpartition("@sha256:")[2])
    return keys


def run_key(meta: dict) -> str:
    """The run key bundle.json names; '' when ingest derives one (manual:<uploader>:<digest>)."""
    return (meta.get("run_key") or meta.get("jenkins_run_key") or "").strip()


def is_manual(key: str) -> bool:
    return not key or key.startswith(MANUAL_PREFIX)


def bundle_name(meta: dict, digest: str) -> str:
    """Its inbox folder (or .tgz) name: unique, since its run key or its content is."""
    key = run_key(meta) or f"manual_{digest[:16]}"
    return slug(key) + "-" + meta["test_type"]


def upload_path(meta: dict, digest: str, inbox: str = "inbox") -> str:
    """Its inbox folder under the generic repo; a .tgz goes at this path + .tgz."""
    key = meta.get("artifact_id") or min(inbox_keys(meta), default="<artifact_id>")
    root = ROOT.format(arch=meta.get("arch") or "any")
    return f"{root}/{inbox}/{key}/{bundle_name(meta, digest)}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_files(root: Path) -> list:
    """Every file under `root` but bundle.json and macOS tar litter, as relative posix paths."""
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and p.relative_to(root).as_posix() != BUNDLE_FILE
        and not (p.name.startswith("._") or p.name == ".DS_Store")
    )


def digest(root: Path) -> str:
    """The bundle's content identity: every file's path and sha256, bundle.json included."""
    lines = [
        f"{p}\0{sha256(root / p)}\n" for p in sorted([BUNDLE_FILE, *bundle_files(root)])
    ]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def count_cases(path: Path) -> int:
    """testcase elements in one XML; raises BundleError when it does not parse."""
    try:
        root = etree.parse(path).getroot()
    except etree.ParseError as err:
        raise BundleError(f"{path.name}: not XML ({err})") from None
    if root.tag not in ("testsuites", "testsuite"):
        raise BundleError(
            f"{path.name}: root is <{root.tag}>, not a JUnit <testsuite(s)>"
        )
    return sum(1 for _ in root.iter("testcase"))


def read_records(path: Path) -> list:
    """A JSON file's records: one object, a list, or JSON lines."""
    text = path.read_text()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        out = []
        for line in text.splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                data = json.loads(line)
                out += data if isinstance(data, list) else [data]
        return out


def shapes(name: str) -> dict:
    """tp1_in64_out64 -> {tensor_parallel, input_len, output_len}, as the ingest reads them."""
    out = {}
    for token in name.split("_"):
        for prefix, key in (
            ("tp", "tensor_parallel"),
            ("in", "input_len"),
            ("out", "output_len"),
        ):
            rest = token[len(prefix) :]
            if token.startswith(prefix) and rest.isdigit():
                out[key] = rest
    return out


def check_vllm(root: Path, meta: dict, results: list) -> None:
    """Each results/*.json is a vLLM bench result named as the runner names it, consistent
    with `perf`. The ingest re-reads them through the live leg's extraction."""
    perf = meta.get("perf", {})
    for p in results:
        name = Path(p).name.removesuffix(".json").removesuffix(".pytorch")
        if name.split("_")[0] not in RUN_MODES:
            raise BundleError(f"{p}: the name must start with one of {RUN_MODES}_")
        records = [r for r in read_records(root / p) if isinstance(r, dict)]
        if not any(
            any(k in r for k in VLLM_SIGNATURES) or ("benchmark" in r and "metric" in r)
            for r in records
        ):
            raise BundleError(f"{p}: no vLLM latency/throughput/serve metric in it")
        got = shapes(name)
        clash = [k for k in got if perf.get(k, got[k]) != got[k]]
        if clash:
            raise BundleError(f"{p}: its name's {clash} differ from perf {clash}")


def check(root: Path, match=pattern_match) -> dict:
    """Validate the bundle at `root` offline; returns bundle.json with test_type filled in."""
    path = root / BUNDLE_FILE
    if not path.is_file():
        raise BundleError(f"no {BUNDLE_FILE} in {root}", INCOMPLETE)
    try:
        meta = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise BundleError(f"{BUNDLE_FILE} is not JSON: {err}") from None
    if not isinstance(meta, dict):
        raise BundleError(f"{BUNDLE_FILE} is not a JSON object")
    version = meta.get("schema_version", SCHEMA_VERSION)
    if isinstance(version, int) and version > SCHEMA_VERSION:
        raise BundleError(
            f"schema_version {version} is newer than this ingest ({SCHEMA_VERSION})"
        )
    errors = schema_errors(meta, schema(), match=match)
    if errors:
        raise BundleError("; ".join(errors))
    meta = normalized(meta)
    if not specs(meta):
        raise BundleError(f"{BUNDLE_FILE}: needs artifact_id or image")
    if meta.get("image") and not meta.get("artifact_id") and not meta.get("arch"):
        errors.append("an image names its artifact only with arch")
    kind = meta.get("kind", "junit")
    meta["test_type"] = meta.get("test_type") or ("perf" if kind == "vllm" else "")
    if not meta["test_type"]:
        errors.append("test_type is required (the tier: unit, regression, fvt, ...)")
    if kind == "vllm" and (meta["test_type"], meta.get("component", VLLM_COMPONENT)) != (
        "perf", VLLM_COMPONENT,
    ):  # fmt: skip
        errors.append(f"a vllm bundle is test_type perf, component {VLLM_COMPONENT}")
    if kind != "vllm" and meta.get("perf"):
        errors.append("perf context is for a vllm bundle")
    key = run_key(meta)
    if meta.get("run_key") and meta.get("jenkins_run_key", key) != key:
        errors.append("run_key and jenkins_run_key differ")
    if is_manual(key) and (meta.get("tag_family") or meta.get("tags")):
        errors.append("a manual bundle records verdicts only; drop tag_family/tags")
    started, ended = meta.get("started_at"), meta.get("ended_at")
    if started and ended and _ts(ended) < _ts(started):
        errors.append("ended_at is before started_at")
    present = bundle_files(root)
    stray = [p for p in present if not p.startswith(("results/", "attachments/"))]
    if stray or any(
        "/" in p.removeprefix("results/") for p in present if p.startswith("results/")
    ):
        errors.append(
            f"files belong in results/ (flat) or attachments/: {stray or present}"
        )
    if "files" in meta:
        listed = [f["path"] for f in meta["files"]]
        if len(set(listed)) != len(listed) or any(".." in p.split("/") for p in listed):
            errors.append("files lists a path twice, or a '..' path")
        unlisted = sorted(set(present) - set(listed))
        if unlisted:
            errors.append(f"file(s) not in files[]: {unlisted}")
    if errors:
        raise BundleError("; ".join(errors))
    if "files" in meta:
        missing = [f["path"] for f in meta["files"] if not (root / f["path"]).is_file()]
        if missing:
            raise BundleError(f"listed file(s) missing: {missing}", INCOMPLETE)
        bad = [
            f["path"] for f in meta["files"] if sha256(root / f["path"]) != f["sha256"]
        ]
        if bad:
            raise BundleError(f"sha256 mismatch: {bad}")
    results = [p for p in present if p.startswith("results/")]
    ext = KINDS[kind]
    wrong = [p for p in results if not p.endswith(ext)]
    if wrong:
        raise BundleError(f"a {kind} bundle holds results/*{ext} only: {wrong}")
    if not results:
        raise BundleError(f"no results/*{ext}: the bundle records nothing")
    if kind == "vllm":
        check_vllm(root, meta, results)
    elif sum(count_cases(root / p) for p in results) == 0:
        raise BundleError(
            "no <testcase> in any results/*.xml: the bundle records nothing"
        )
    return meta


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextlib.contextmanager
def opened(path: Path):
    """The bundle directory at `path`; a .tgz/.tar.gz is unpacked to a temporary one."""
    if path.is_dir():
        yield path
        return
    if not path.is_file():
        raise BundleError(f"{path}: no such bundle", INCOMPLETE)
    with tempfile.TemporaryDirectory(prefix="bundle-") as tmp:
        try:
            with tarfile.open(path, "r:*") as tar:
                for m in tar.getmembers():
                    name = Path(m.name)
                    if (
                        name.is_absolute()
                        or ".." in name.parts
                        or not (m.isfile() or m.isdir())
                    ):
                        raise BundleError(f"{path.name}: unsafe member {m.name!r}")
                # Members are checked above; `filter` is absent before Python 3.9.17/3.11.4.
                try:
                    tar.extractall(tmp, filter="data")
                except TypeError:
                    tar.extractall(tmp)
        except tarfile.TarError as err:
            raise BundleError(f"{path.name}: not a tar archive ({err})") from None
        top = Path(tmp)
        # Either bundle.json at the top, or one directory holding it.
        entries = list(top.iterdir())
        if (
            not (top / BUNDLE_FILE).exists()
            and len(entries) == 1
            and entries[0].is_dir()
        ):
            top = entries[0]
        yield top


def pack(root: Path, tgz: Path) -> Path:
    with tarfile.open(tgz, "w:gz") as tar:
        for p in [BUNDLE_FILE, *bundle_files(root)]:
            tar.add(root / p, arcname=p)
    return tgz


def _earliest_suite(paths) -> str:
    """The earliest <testsuite timestamp>, as UTC; pytest writes it naive, in local time."""
    stamps = []
    for xml in paths:
        with contextlib.suppress(etree.ParseError, OSError):
            for suite in etree.parse(xml).getroot().iter("testsuite"):
                with contextlib.suppress(ValueError):
                    ts = datetime.fromisoformat(suite.get("timestamp", ""))
                    stamps.append(ts if ts.tzinfo else ts.astimezone())
    return (
        min(stamps).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if stamps
        else ""
    )


def _arch(value: str) -> str:
    a = (value or platform.machine()).strip().lower()
    return "x86_64" if a in ("amd64", "x86", "x86-64") else a


def write(args) -> Path:
    """The bundle these `results` flags describe, written to --out (a folder, or a .tgz)."""
    kind = "vllm" if (args.kind == "vllm" or args.vllm_results_dir) else "junit"
    if kind == "vllm":
        if not args.vllm_results_dir:
            raise BundleError("--kind vllm needs --vllm-results-dir")
        src = Path(args.vllm_results_dir)
        results = sorted(src.glob("*.json"))
        extra = sorted(
            p for p in src.iterdir() if p.is_file() and p.suffix in (".cmd", ".log")
        )
    else:
        results = (
            [Path(args.xml_file)]
            if args.xml_file
            else sorted(Path(args.xml_dir or ".").glob("*.xml"))
        )
        extra = []
    named = {
        "artifact_id": (args.artifact_id or "").strip(),
        "artifact": (args.artifact or "").strip(),
    }
    meta = {"schema_version": SCHEMA_VERSION, **({"kind": kind} if kind == "vllm" else {}),
            **normalized(named), "arch": _arch(args.arch)}  # fmt: skip
    for key, value in (
        ("component", args.component), ("test_type", args.trigger_type),
        ("run_key", args.jenkins_run_key), ("run_url", args.run_url), ("workflow", args.workflow),
        ("tag_family", args.tag_family),
    ):  # fmt: skip
        if value:
            meta[key] = value
    if args.tags:
        meta["tags"] = list(args.tags)
    if args.perf:
        meta["perf"] = dict(args.perf)
    started = args.triggered_at or (_earliest_suite(results) if kind == "junit" else "")
    if started:
        meta["started_at"] = started
    meta["ended_at"] = _now()
    meta["runner"] = {"host": socket.gethostname(), "user": getpass.getuser()}
    out = (
        Path(args.out)
        if args.out
        else Path(tempfile.mkdtemp(prefix="bundle-")) / "bundle"
    )
    tgz = out.name.endswith((".tgz", ".tar.gz"))
    root = Path(tempfile.mkdtemp(prefix="bundle-")) if tgz else out
    if root.exists() and any(root.iterdir()):
        raise BundleError(f"{root} is not empty")
    (root / "results").mkdir(parents=True, exist_ok=True)
    for p in results:
        shutil.copy2(p, root / "results" / p.name)
    if extra:
        (root / "attachments").mkdir(exist_ok=True)
        for p in extra:
            shutil.copy2(p, root / "attachments" / p.name)
    meta["files"] = [
        {"path": p, "sha256": sha256(root / p)} for p in bundle_files(root)
    ]
    (root / BUNDLE_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    checked = check(root)
    if tgz:
        out.parent.mkdir(parents=True, exist_ok=True)
        pack(root, out)
        shutil.rmtree(root)
    with opened(out) as top:
        dig = digest(top)
    print(
        f"[info] wrote {out}: {len(meta['files'])} file(s), bundle sha256 {dig}",
        file=sys.stderr,
    )
    print(
        f"[info] inbox path: {upload_path(checked, dig, args.inbox)}{'.tgz' if tgz else '/'}",
        file=sys.stderr,
    )
    return out


def report(path: Path) -> dict:
    """What `--validate-only` prints: the bundle's run identity and where it uploads."""
    with opened(path) as root:
        meta, dig = check(root), digest(root)
    key = run_key(meta)
    out = {"status": "valid", "kind": meta.get("kind", "junit"), "test_type": meta["test_type"],
           "run_key": key or f"{MANUAL_PREFIX}<uploader>:{dig}", "bundle_sha256": dig,
           "upload_path": upload_path(meta, dig)}  # fmt: skip
    if key and meta.get("arch"):
        with contextlib.suppress(ImportError):
            from spyre_clickhouse_ingest.identity import RunId

            source = "bundle" if is_manual(key) else "jenkins"
            out["run_id"] = RunId.derive(source, key, meta["arch"], meta["test_type"])
    return out


class Artifactory:
    """PUT and GET against the generic repo, with retries; auth from a token or ~/.netrc."""

    def __init__(self, base: str, repo: str, token: str = ""):
        self.base, self.repo = base.rstrip("/"), repo
        user = os.environ.get("ARTIFACTORY_USER", "")
        token = token or os.environ.get("ARTIFACTORY_TOKEN", "")
        if not token:
            host = urllib.parse.urlparse(self.base).hostname or ""
            with contextlib.suppress(FileNotFoundError, netrc.NetrcParseError):
                found = netrc.netrc().authenticators(host)
                if found:
                    user, _, token = found
        if not token:
            raise BundleError(
                "no Artifactory token: pass --token, set ARTIFACTORY_TOKEN, or add the host to ~/.netrc",
                FAILED,
            )
        self.auth = (
            "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()
            if user
            else f"Bearer {token}"
        )

    def url(self, path: str) -> str:
        return f"{self.base}/{self.repo}/{urllib.parse.quote(path)}"

    def call(self, method: str, url: str, data=None, tries: int = 3) -> bytes:
        for attempt in range(1, tries + 1):
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("Authorization", self.auth)
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return resp.read()
            except urllib.error.HTTPError as err:
                if err.code < 500 or attempt == tries:
                    raise
            except urllib.error.URLError:
                if attempt == tries:
                    raise
            time.sleep(5 * attempt)
        raise AssertionError("unreachable")

    def put(self, path: str, data: bytes) -> str:
        url = self.url(path)
        self.call("PUT", url, data)
        return url

    def children(self, path: str) -> list:
        try:
            out = self.call(
                "GET",
                f"{self.base}/api/storage/{self.repo}/{urllib.parse.quote(path)}",
                tries=2,
            )
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return []
            raise
        return [c["uri"].lstrip("/") for c in json.loads(out).get("children", [])]

    def get(self, path: str) -> bytes:
        return self.call("GET", self.url(path), tries=2)


def upload(path: Path, args) -> str:
    """PUT the bundle into its inbox (a .tgz, unless --as-folder); returns where it went."""
    art = Artifactory(args.base_url, args.repo, args.token)
    with opened(path) as root:
        meta, dig = check(root), digest(root)
        dest = upload_path(meta, dig, args.inbox)
        if args.as_folder:
            # bundle.json last: the relay takes a folder only once it is there.
            for p in [*bundle_files(root), BUNDLE_FILE]:
                art.put(f"{dest}/{p}", (root / p).read_bytes())
            url = art.url(dest) + "/"
        elif path.is_file():
            url = art.put(dest + ".tgz", path.read_bytes())
        else:
            with tempfile.TemporaryDirectory() as tmp:
                url = art.put(
                    dest + ".tgz", pack(root, Path(tmp) / "bundle.tgz").read_bytes()
                )
    print(f"[info] uploaded {url}", file=sys.stderr)
    if args.wait:
        return wait(art, meta, dig, args.wait, args.as_folder, args.inbox)
    return json.dumps({"status": "uploaded", "url": url, "bundle_sha256": dig})


def wait(
    art, meta: dict, dig: str, minutes: int, as_folder=False, inbox="inbox"
) -> str:
    """Poll until the relay has moved the bundle to processed/ or rejected/, beside the inbox."""
    key = meta.get("artifact_id") or min(inbox_keys(meta))
    parent = inbox.rpartition("/")[0]
    root = ROOT.format(arch=meta.get("arch") or "any") + (
        f"/{parent}" if parent else ""
    )
    stem = bundle_name(meta, dig)
    name = stem if as_folder else stem + ".tgz"
    deadline = time.time() + minutes * 60
    while True:
        for where in ("processed", "rejected"):
            folder = f"{root}/{where}/{key}"
            hits = [
                c for c in art.children(folder)
                if (c == name or c.startswith(stem + "-dup")) and not c.endswith(".REJECTED.json")
            ]  # fmt: skip
            if not hits:
                continue
            out = {"status": "ingested" if where == "processed" else "rejected",
                   "at": art.url(f"{folder}/{hits[0]}")}  # fmt: skip
            if where == "rejected":
                note = f"{folder}/{hits[0]}" + (
                    "/REJECTED.json" if as_folder else ".REJECTED.json"
                )
                with contextlib.suppress(urllib.error.HTTPError, json.JSONDecodeError):
                    out["reason"] = json.loads(art.get(note)).get("reason", "")
            return json.dumps(out)
        if time.time() > deadline:
            return json.dumps(
                {"status": "waiting", "reason": f"not relayed within {minutes} min"}
            )
        time.sleep(POLL_SECONDS)


def add_offline_options(parser) -> None:
    """The bundle flags of `results`, declared once for both its online and offline parsers."""
    g = parser.add_argument_group(
        "offline results bundles (README: Offline results bundles)"
    )
    g.add_argument(
        "--offline",
        action="store_true",
        help="write a bundle instead of connecting to ClickHouse",
    )
    g.add_argument(
        "--out", default="", help="--offline: the bundle to write, a folder or a .tgz"
    )
    g.add_argument("--upload", nargs="?", const=True, default=None, metavar="BUNDLE",
                   help="upload to the Artifactory inbox: the bundle --offline wrote, or BUNDLE")  # fmt: skip
    g.add_argument("--wait", nargs="?", type=int, const=WAIT_MINUTES, default=0, metavar="MIN",
                   help=f"after --upload, wait (default {WAIT_MINUTES} min) for the relay's outcome")  # fmt: skip
    g.add_argument(
        "--as-folder",
        action="store_true",
        help="upload a folder, bundle.json last, not a .tgz",
    )
    g.add_argument(
        "--token",
        default="",
        help="Artifactory token; default $ARTIFACTORY_TOKEN, else ~/.netrc",
    )
    g.add_argument("--base-url", default=BASE_URL)
    g.add_argument(
        "--inbox",
        default="inbox",
        help="under v2-results/; a trial uses inbox-test/inbox",
    )
    g.add_argument("--repo", default=REPO)
    g.add_argument("--kind", choices=tuple(KINDS), default="junit")
    g.add_argument(
        "--vllm-results-dir", default="", help="vLLM bench *.json (implies --kind vllm)"
    )
    g.add_argument("--perf", action="append", type=_pair, default=[],
                   help="vllm run context k=v: model, tensor_parallel, input_len, output_len, cards, "
                   "head_sha, head_branch, state")  # fmt: skip
    g.add_argument(
        "--from-bundle",
        default="",
        help="record this bundle (the relay); see --validate-only",
    )
    g.add_argument(
        "--validate-only",
        action="store_true",
        help="with --from-bundle: check it offline",
    )
    g.add_argument(
        "--uploader", default="", help="--from-bundle: recorded as props['uploader']"
    )
    g.add_argument(
        "--bundle-url",
        default="",
        help="--from-bundle: where it is kept; the run_url default",
    )
    g.add_argument("--trusted-job-prefix", action="append", default=[],
                   help="--from-bundle: accept a Jenkins run key only from these jobs")  # fmt: skip
    g.add_argument(
        "--expect-key",
        default="",
        help="--from-bundle: its inbox folder, <artifact_id> or sha256-<digest>",
    )


def _pair(value: str) -> tuple:
    key, sep, val = value.partition("=")
    if not (sep and key):
        raise argparse.ArgumentTypeError(f"wants key=value, got {value!r}")
    return key, val


# The `results` flags an offline bundle records; results.py declares the same names.
SHARED = (
    ("--xml-dir", {}), ("--xml-file", {}), ("--artifact", {}), ("--artifact-id", {}),
    ("--component", {}), ("--trigger-type", {}), ("--jenkins-run-key", {}), ("--run-url", {}),
    ("--workflow", {}), ("--triggered-at", {}), ("--tag-family", {}),
)  # fmt: skip


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spyre_clickhouse_ingest results", description=__doc__.splitlines()[0]
    )
    for flag, kw in SHARED:
        p.add_argument(flag, default="", **kw)
    p.add_argument("--arch", "--platform", dest="arch", default="")
    p.add_argument("--tag", dest="tags", action="append", default=[])
    add_offline_options(p)
    return p


def main(argv=None) -> int:
    args, ignored = parser().parse_known_args(argv)
    if ignored:
        print(f"[warn] not recorded in a bundle: {' '.join(ignored)}", file=sys.stderr)
    try:
        if args.validate_only:
            if not args.from_bundle:
                raise BundleError("--validate-only needs --from-bundle <dir|tgz>")
            print(json.dumps(report(Path(args.from_bundle)), sort_keys=True))
            return 0
        if args.offline:
            if not (args.out or args.upload):
                raise BundleError("--offline needs --out <dir|tgz> and/or --upload")
            path = write(args)
            if args.upload:
                print(upload(path, args))
            else:
                print(json.dumps(report(path), sort_keys=True))
            return 0
        if isinstance(args.upload, str):
            print(upload(Path(args.upload), args))
            return 0
        raise BundleError(
            "pass --offline, --upload <bundle> or --from-bundle <bundle> --validate-only"
        )
    except BundleError as err:
        print(json.dumps({"status": STATUS[err.code], "reason": str(err)}))
        return err.code
    except urllib.error.URLError as err:
        print(json.dumps({"status": "failed", "reason": f"Artifactory: {err}"}))
        return FAILED


if __name__ == "__main__":
    sys.exit(main())
