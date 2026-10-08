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

"""Results bundles: test results recorded where spyre_v2 is unreachable, ingested later.

    python -m spyre_clickhouse_ingest bundle init (--artifact id:<uuid> | --image <repo>@sha256:<d>) \
        --arch A [--kind vllm | --component C --test-type T] [--out DIR]   # offline skeleton
    python -m spyre_clickhouse_ingest bundle seal DIR   # offline: files[] checksums, validate
    python -m spyre_clickhouse_ingest bundle validate DIR|TGZ        # offline
    python -m spyre_clickhouse_ingest bundle ingest DIR|TGZ --strict # online

A bundle is DIR/bundle.json (bundle.schema.json) + DIR/results/ (+ DIR/attachments/): JUnit XML
(kind junit), or vLLM bench JSON (kind vllm). `ingest` records it through the path a connected
run takes -- `results` for JUnit, vllm.py for vLLM -- so its rows link exactly as that run's do,
under the run_id of its run_key. It writes no artifact: the artifact must already be recorded,
found by artifact_id, else by image digest.
Exit codes: 0 ingested or already ingested, 1 failed (retry), 2 rejected, 3 incomplete.
"""

import argparse
import contextlib
import getpass
import hashlib
import json
import os
import socket
import sys
import tarfile
import tempfile
import uuid
import xml.etree.ElementTree as etree
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

import regex as re

from . import vllm
from .identity import DerivedId, RunId
from .options import pair

SCHEMA_VERSION = 1
BUNDLE_FILE = "bundle.json"
FAILED, REJECTED, INCOMPLETE = 1, 2, 3
STATUS = {FAILED: "failed", REJECTED: "rejected", INCOMPLETE: "incomplete"}
# A Jenkins run key is hashed as source `jenkins` (equal to a connected run's id), a manual one
# as `bundle`.
MANUAL_PREFIX = "manual:"
SOURCE = "bundle"
# kind -> the extension of its results/ files.
KINDS = {"junit": ".xml", "vllm": ".json"}


class BundleError(Exception):
    """The bundle cannot be ingested; `code` says whether a retry can help."""

    def __init__(self, message: str, code: int = REJECTED):
        super().__init__(message)
        self.code = code


def schema() -> dict:
    return json.loads(
        resources.files(__package__).joinpath("bundle.schema.json").read_text()
    )


def _date_time(value: str) -> bool:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def schema_errors(value, node: dict, where: str = "bundle.json") -> list:
    """`value` checked against the subset of JSON Schema bundle.schema.json uses."""
    errors = []
    if "const" in node and value != node["const"]:
        return [f"{where}: must be {node['const']!r}, got {value!r}"]
    if "enum" in node and value not in node["enum"]:
        return [f"{where}: {value!r} is not one of {node['enum']}"]
    kind = node.get("type")
    types = {"object": dict, "array": list, "string": str}
    if kind and not isinstance(value, types[kind]):
        return [f"{where}: must be a {kind}"]
    if isinstance(value, str):
        if len(value) < node.get("minLength", 0):
            errors.append(f"{where}: must not be empty")
        if "pattern" in node and not re.search(node["pattern"], value):
            errors.append(f"{where}: {value!r} does not match {node['pattern']}")
        if node.get("format") == "date-time" and not _date_time(value):
            errors.append(f"{where}: {value!r} is not an ISO-8601 time with a zone")
    if isinstance(value, list):
        if len(value) < node.get("minItems", 0):
            errors.append(f"{where}: needs at least {node['minItems']} item(s)")
        for i, item in enumerate(value):
            errors += schema_errors(item, node.get("items", {}), f"{where}[{i}]")
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
                errors += schema_errors(v, props[k], f"{where}.{k}")
            elif extra is False:
                errors.append(f"{where}: unknown key {k!r}")
            elif isinstance(extra, dict):
                errors += schema_errors(v, extra, f"{where}.{k}")
    for branch in node.get("allOf", ()):
        errors += schema_errors(value, branch, where)
    if "anyOf" in node and all(schema_errors(value, b, where) for b in node["anyOf"]):
        need = " or ".join("/".join(b.get("required", ["?"])) for b in node["anyOf"])
        errors.append(f"{where}: needs {need}")
    return errors


def run_key(meta: dict) -> str:
    return (meta.get("run_key") or meta.get("jenkins_run_key") or "").strip()


def is_manual(key: str) -> bool:
    return key.startswith(MANUAL_PREFIX)


def run_id(meta: dict) -> str:
    """The run_id `results` writes this bundle's rows under."""
    key = run_key(meta)
    source = SOURCE if is_manual(key) else "jenkins"
    return RunId.derive(source, key, meta["arch"], meta["test_type"])


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


ARTIFACT_KEYS = ("artifact_id", "image", "artifact")


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


def bundle_name(meta: dict) -> str:
    """The bundle's folder name in the inbox: unique, since its run key is."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", run_key(meta)) + "-" + meta["test_type"]


def upload_path(meta: dict) -> str:
    """Its inbox folder under the generic repo (see README); a .tgz goes at this path + .tgz."""
    key = meta.get("artifact_id") or min(inbox_keys(meta), default="<artifact_id>")
    return f"zsp/next/{meta['arch']}/v2-results/inbox/{key}/{bundle_name(meta)}"


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


def check(root: Path) -> dict:
    """Validate the bundle at `root` offline; returns bundle.json, raises BundleError."""
    path = root / BUNDLE_FILE
    if not path.is_file():
        raise BundleError(f"no {BUNDLE_FILE} in {root}", INCOMPLETE)
    try:
        meta = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise BundleError(f"{BUNDLE_FILE} is not JSON: {err}") from None
    errors = schema_errors(meta, schema())
    if errors:
        raise BundleError("; ".join(errors))
    meta = normalized(meta)
    if not specs(meta):
        raise BundleError(f"{BUNDLE_FILE}: needs artifact_id or image")
    key = run_key(meta)
    if meta.get("run_key") and meta.get("jenkins_run_key", key) != key:
        errors.append("run_key and jenkins_run_key differ")
    if is_manual(key) and (meta.get("tag_family") or meta.get("tags")):
        errors.append("a manual bundle records verdicts only; drop tag_family/tags")
    kind = meta.get("kind", "junit")
    if kind == "vllm" and (meta["test_type"], meta["component"]) != (
        "perf",
        vllm.BENCH_COMPONENT,
    ):
        errors.append(
            f"a vllm bundle is test_type perf, component {vllm.BENCH_COMPONENT}"
        )
    if kind != "vllm" and meta.get("perf"):
        errors.append("perf context is for a vllm bundle")
    if meta.get("ended_at") and _ts(meta["ended_at"]) < _ts(meta["started_at"]):
        errors.append("ended_at is before started_at")
    listed = [f["path"] for f in meta["files"]]
    if len(set(listed)) != len(listed):
        errors.append("files lists a path twice")
    if any(".." in p.split("/") for p in listed):
        errors.append("files has a '..' path")
    unlisted = sorted(set(bundle_files(root)) - set(listed))
    if unlisted:
        errors.append(f"file(s) not in files[]: {unlisted}")
    if errors:
        raise BundleError("; ".join(errors))
    missing = [p for p in listed if not (root / p).is_file()]
    if missing:
        raise BundleError(f"listed file(s) missing: {missing}", INCOMPLETE)
    bad = [f["path"] for f in meta["files"] if sha256(root / f["path"]) != f["sha256"]]
    if bad:
        raise BundleError(f"sha256 mismatch: {bad}")
    results = [p for p in listed if p.startswith("results/")]
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


def check_vllm(root: Path, meta: dict, results: list) -> None:
    """Every results/*.json is a vLLM bench file the live ingest reads, consistent with `perf`."""
    perf = meta.get("perf", {})
    for p in results:
        name = vllm.bench_name(Path(p).name)
        if name.split("_")[0] not in vllm.RUN_MODES:
            raise BundleError(f"{p}: the name must start with one of {vllm.RUN_MODES}_")
        try:
            records = vllm.read_benchmark_results(str(root / p))
        except (OSError, UnicodeDecodeError) as err:
            raise BundleError(f"{p}: unreadable ({err})") from None
        if not any(
            isinstance(r, dict)
            and (vllm.extract_vllm_metrics(r) or vllm.extract_pytorch_metrics(r))
            for r in records
        ):
            raise BundleError(f"{p}: no vLLM latency/throughput/serve metric in it")
        shapes = vllm.parse_input_shapes(name)
        clash = [k for k in shapes if perf.get(k, shapes[k]) != shapes[k]]
        if clash:
            raise BundleError(f"{p}: its name's {clash} differ from perf {clash}")


def vllm_rows(root: Path, meta: dict) -> list:
    """The bundle's flat vLLM rows, exactly as the live ingest extracts them."""
    perf = meta.get("perf", {})
    with contextlib.redirect_stdout(sys.stderr):
        return vllm.extract_rows(
            str(root / "results"), perf.get("head_branch", ""), perf.get("head_sha", ""),
            "", "0", meta.get("workflow") or "vLLM Benchmark", 0, arch=meta["arch"],
            model=perf.get("model", ""),
        )  # fmt: skip


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def digest(root: Path) -> str:
    """The bundle's identity: bundle.json's sha256, which covers every file through files[]."""
    return sha256(root / BUNDLE_FILE)


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
                tar.extractall(tmp, filter="data")
        except tarfile.TarError as err:
            raise BundleError(f"{path.name}: not a tar archive ({err})") from None
        top = Path(tmp)
        # Either bundle.json at the top, or one directory holding it.
        entries = [p for p in top.iterdir()]
        if (
            not (top / BUNDLE_FILE).exists()
            and len(entries) == 1
            and entries[0].is_dir()
        ):
            top = entries[0]
        yield top


def init(args) -> int:
    out = Path(args.out)
    path = out / BUNDLE_FILE
    if path.exists() and not args.force:
        sys.exit(f"[error] {path} exists; pass --force to replace it")
    vllm_kind = args.kind == "vllm"
    try:
        named = normalized(
            {"artifact_id": args.artifact_id.strip(), "image": args.image.strip(),
             "artifact": args.artifact.strip()}
        )  # fmt: skip
    except BundleError as err:
        sys.exit(f"[error] {err}")
    meta = {
        "schema_version": SCHEMA_VERSION,
        **({"kind": args.kind} if vllm_kind else {}),
        **named,
        "component": args.component or (vllm.BENCH_COMPONENT if vllm_kind else ""),
        "arch": DerivedId.arch(args.arch),
        "test_type": args.test_type or ("perf" if vllm_kind else ""),
        "run_key": args.run_key or f"{MANUAL_PREFIX}{getpass.getuser()}:{uuid.uuid4()}",
        "started_at": args.started_at,
        "runner": {
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            **dict(args.runner),
        },
        "files": [],
    }
    if args.perf:
        meta["perf"] = dict(args.perf)
    for key in ("run_url", "workflow", "notes", "tag_family"):
        if getattr(args, key):
            meta[key] = getattr(args, key)
    if args.tags:
        meta["tags"] = list(args.tags)
    if args.env:
        meta["env"] = dict(args.env)
    (out / "results").mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n")
    what = (
        "the vLLM bench *.json (and *.pytorch.json); .cmd/.log files go in attachments/"
        if vllm_kind
        else "the JUnit XML"
    )
    print(
        f"[info] wrote {path}; put {what} in {out / 'results'}/, then: bundle seal {out}"
    )
    print(f"[info] upload path: {upload_path(meta)}")
    return 0


def seal(args) -> int:
    root = Path(args.dir)
    path = root / BUNDLE_FILE
    meta = json.loads(path.read_text())
    meta["files"] = [
        {"path": p, "sha256": sha256(root / p)} for p in bundle_files(root)
    ]
    if not meta.get("started_at"):
        meta["started_at"] = _earliest_suite(root) or _earliest_file(root) or _now()
    meta["ended_at"] = args.ended_at or meta.get("ended_at") or _now()
    meta["sealed_at"] = _now()
    path.write_text(json.dumps(meta, indent=2) + "\n")
    try:
        check(root)
    except BundleError as err:
        print(f"[error] sealed, but not valid: {err}", file=sys.stderr)
        return err.code
    print(f"[info] sealed {root}: {len(meta['files'])} file(s), sha256 {digest(root)}")
    print(f"[info] upload path: {upload_path(meta)}")
    return 0


def _earliest_suite(root: Path) -> str:
    """The earliest <testsuite timestamp> in results/, as UTC; '' when none carries one.

    pytest writes it naive, in the test host's local time: seal where the tests ran.
    """
    stamps = []
    for xml in sorted((root / "results").glob("*.xml")):
        with contextlib.suppress(etree.ParseError):
            for suite in etree.parse(xml).getroot().iter("testsuite"):
                with contextlib.suppress(ValueError):
                    ts = datetime.fromisoformat(suite.get("timestamp", ""))
                    stamps.append(ts if ts.tzinfo else ts.astimezone())
    return min(stamps).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if stamps else ""


def _earliest_file(root: Path) -> str:
    """The oldest results/ file's mtime: a vLLM result carries no start time of its own."""
    times = [p.stat().st_mtime for p in (root / "results").glob("*") if p.is_file()]
    return (
        datetime.fromtimestamp(min(times), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        if times
        else ""
    )


def validate(args) -> int:
    try:
        with opened(Path(args.bundle)) as root:
            meta = check(root)
            report = {"status": "valid", "kind": meta.get("kind", "junit"), "run_key": run_key(meta),
                      "run_id": run_id(meta), "bundle_sha256": digest(root),
                      "upload_path": upload_path(meta)}  # fmt: skip
    except BundleError as err:
        print(json.dumps({"status": "invalid", "reason": str(err)}))
        return err.code
    print(json.dumps(report, sort_keys=True))
    return 0


def existing_run(client, db: str, rid: str, component: str) -> tuple:
    """(bundle digests of the run's recorded verdicts, the source files of its cases)."""
    verdicts = client.query(
        f"SELECT props['bundle_sha256'] FROM {db}.artifact_results "
        "WHERE run_id = {rid:UUID} AND state != 'running'",
        parameters={"rid": rid},
    ).result_rows
    files = client.query(
        f"SELECT DISTINCT props['source_file'] FROM {db}.test_case_runs "
        "WHERE component = {c:String} AND run_id = {rid:UUID}",
        parameters={"rid": rid, "c": component},
    ).result_rows
    return {r[0] for r in verdicts}, {r[0] for r in files}


def verdict_recorded(client, db: str, aid: str, rid: str, test_type: str) -> bool:
    return (
        client.query(
            f"SELECT count() FROM {db}.artifact_results WHERE artifact_id = {{aid:UUID}} "
            "AND run_id = {rid:UUID} AND test_type = {t:String} AND state != 'running'",
            parameters={"aid": aid, "rid": rid, "t": test_type},
        ).result_rows[0][0]
        > 0
    )


def results_argv(meta: dict, root: Path, aid: str, rid: str, props: dict, strict: bool):
    """The `results` argv recording this bundle: the artifact by id, lookup only."""
    key = run_key(meta)
    argv = [
        "--schema", "v2", "--xml-dir", str(root / "results"),
        "--component", meta["component"], "--arch", meta["arch"],
        "--trigger-type", meta["test_type"], "--run-id", rid,
        "--artifact", f"id:{aid}", "--lookup", "only", "--registry", "off",
        "--run-url", meta.get("run_url") or props.get("bundle_url", ""),
        "--triggered-at", meta["started_at"],
    ]  # fmt: skip
    if not is_manual(key):
        argv += ["--jenkins-run-key", key]
    for flag, value in (
        ("--workflow", meta.get("workflow")),
        ("--tag-family", meta.get("tag_family")),
    ):
        if value:
            argv += [flag, value]
    for tag in meta.get("tags", []):
        argv += ["--tag", tag]
    for k, v in props.items():
        if v:
            argv += ["--result-prop", f"{k}={v}"]
    return argv + (["--strict"] if strict else [])


def ingest(args) -> int:
    try:
        with opened(Path(args.bundle)) as root:
            report = _ingest(root, args)
    except BundleError as err:
        print(json.dumps({"status": STATUS[err.code], "reason": str(err)}))
        return err.code
    print(json.dumps(report, sort_keys=True))
    return 0


def resolve_artifact(client, db: str, meta: dict) -> tuple:
    """(artifact_id, the field that named it): the id first, else the image, lookup only.

    Each named field must resolve to the same artifact; one that names none (an unknown id)
    is passed over when another resolves.
    """
    from .resolver import resolve

    found, missed = {}, []
    for field, sp in specs(meta):
        try:
            r = resolve(sp, meta["arch"], client=client, db=db, lookup="only")
        except ValueError as err:
            missed.append(f"{field} {sp!r}: {err}")
            continue
        if r is None:
            missed.append(f"{field} {sp!r}: not recorded in {db} on {meta['arch']}")
        else:
            found.setdefault(r.artifact_id, field)
    if len(found) > 1:
        raise BundleError(
            f"the bundle's artifact fields name different artifacts: {found}"
        )
    if not found:
        raise BundleError("artifact not recorded: " + "; ".join(missed))
    return next(iter(found.items()))


def _ingest(root: Path, args) -> dict:
    meta = check(root)
    key = run_key(meta)
    if not is_manual(key) and args.trusted_job_prefix:
        if not any(key.startswith(p) for p in args.trusted_job_prefix):
            raise BundleError(
                f"run key {key!r} is not from a trusted job ({args.trusted_job_prefix}); "
                f"a hand-made bundle uses {MANUAL_PREFIX}<who>:<uuid4>"
            )
    if args.expect_key and args.expect_key not in inbox_keys(meta):
        raise BundleError(
            f"its folder {args.expect_key!r} is not its artifact_id or sha256-<image digest> "
            f"({sorted(inbox_keys(meta))})"
        )
    from .client import ClickHouse, ClickHouseEnv

    db = args.database or ClickHouseEnv.target_database()
    if not db:
        raise BundleError(
            "no database: pass --database or set CLICKHOUSE_DB_V2", FAILED
        )
    try:
        client = ClickHouse.connect(database=db)
        if args.dry_run:
            client.set_client_setting("readonly", "2")
    except Exception as err:  # noqa: BLE001 -- unreachable is a retry, not a verdict on the bundle
        raise BundleError(f"spyre_v2 unreachable: {err}", FAILED) from None
    aid, via = resolve_artifact(client, db, meta)
    rid, sha = run_id(meta), digest(root)
    report = {"run_key": key, "run_id": rid, "artifact_id": aid, "artifact_from": via,
              "test_type": meta["test_type"], "bundle_sha256": sha}  # fmt: skip
    seen, files = existing_run(client, db, rid, meta["component"])
    if seen == {sha}:
        return {**report, "status": "duplicate"}
    if seen:
        raise BundleError(
            f"run {key!r} ({rid}) is already recorded by another run or bundle"
        )
    names = {
        Path(f["path"]).name for f in meta["files"] if f["path"].startswith("results/")
    }
    if files - names:
        raise BundleError(
            f"run {key!r} ({rid}) already has cases from {sorted(files - names)}"
        )
    if args.dry_run:
        return {**report, "status": "would-ingest"}
    props = {"source": SOURCE, "uploader": args.uploader or meta.get("uploader", ""),
             "bundle_sha256": sha, "bundle_url": args.bundle_url}  # fmt: skip
    if meta.get("kind") == "vllm":
        report["benchmark_runs"] = write_vllm(client, db, root, meta, aid, rid, props)
    else:
        write_junit(db, root, meta, aid, rid, props, args.strict)
    if not verdict_recorded(client, db, aid, rid, meta["test_type"]):
        raise BundleError(
            f"no {meta['test_type']} verdict landed for {aid} under {rid}", FAILED
        )
    return {**report, "status": "ingested"}


def write_junit(db, root, meta, aid, rid, props, strict) -> None:
    from .results import main as results

    os.environ["CLICKHOUSE_DB_V2"] = db
    # stdout carries only this command's one-line JSON report.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            results(results_argv(meta, root, aid, rid, props, strict))
    except SystemExit as exit_:
        if exit_.code not in (0, None):
            raise BundleError(f"results exited {exit_.code}", FAILED) from None


def write_vllm(client, db, root, meta, aid, rid, props) -> int:
    """The live vLLM leg's v2 rows: benchmarks + benchmark_runs, then its performance verdict."""
    from .client import tables_present
    from .schema import ARTIFACT_RESULTS, BENCHMARK_RUNS, BENCHMARKS
    from .writer import insert_artifact_result

    if not tables_present(
        client, db, tables=(BENCHMARKS, BENCHMARK_RUNS, ARTIFACT_RESULTS)
    ):
        raise BundleError(f"{db} lacks the benchmark tables", FAILED)
    rows = vllm_rows(root, meta)
    try:
        n = vllm.write_benchmarks(client, db, rows, rid)
        insert_artifact_result(
            client, db, artifact_id=aid, run_id=rid, test_type=meta["test_type"],
            state=meta.get("perf", {}).get("state", "passed"), arch=meta["arch"],
            result_kind="performance", duration_s=vllm.duration_s(rows),
            props={**{k: v for k, v in props.items() if v},
                   "run_url": meta.get("run_url") or props.get("bundle_url", "")},
        )  # fmt: skip
    except Exception as err:  # noqa: BLE001 -- a write error is a retry; the writes dedup
        raise BundleError(f"vLLM write failed: {err!r}", FAILED) from None
    return n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="spyre_clickhouse_ingest bundle", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ini = sub.add_parser("init", help="write a bundle.json skeleton (offline)")
    ini.add_argument(
        "--artifact", default="", help="id:<uuid> or image:<repo>@sha256:<digest>"
    )
    ini.add_argument("--artifact-id", default="")
    ini.add_argument("--image", default="", help="<repo>@sha256:<digest>: the fallback")
    ini.add_argument("--kind", choices=tuple(KINDS), default="junit")
    ini.add_argument("--component", default="", help="required for junit")
    ini.add_argument("--arch", required=True)
    ini.add_argument("--test-type", default="", help="required for junit; vllm: perf")
    ini.add_argument(
        "--perf",
        action="append",
        type=pair,
        default=[],
        help="vllm context k=v: model, head_sha, head_branch, state, tensor_parallel, ...",
    )
    ini.add_argument("--run-key", default="", help="default: manual:<user>:<uuid4>")
    ini.add_argument("--started-at", default="", help="default: set by seal")
    ini.add_argument("--run-url", default="")
    ini.add_argument("--workflow", default="")
    ini.add_argument("--notes", default="")
    ini.add_argument(
        "--runner",
        action="append",
        type=pair,
        default=[],
        help="k=v, e.g. image=<ref@digest>",
    )
    ini.add_argument(
        "--env", action="append", type=pair, default=[], help="k=v, repeatable"
    )
    ini.add_argument("--tag-family", default="")
    ini.add_argument("--tag", dest="tags", action="append", default=[])
    ini.add_argument("--out", default=".", help="the bundle directory (default: .)")
    ini.add_argument("--force", action="store_true")

    sea = sub.add_parser(
        "seal", help="checksum every file into files[] and validate (offline)"
    )
    sea.add_argument("dir")
    sea.add_argument("--ended-at", default="")

    val = sub.add_parser(
        "validate", help="validate a bundle directory or .tgz (offline)"
    )
    val.add_argument("bundle")

    ing = sub.add_parser("ingest", help="record a sealed bundle in spyre_v2 (online)")
    ing.add_argument("bundle")
    ing.add_argument("--database", default="", help="default: $CLICKHOUSE_DB_V2")
    ing.add_argument(
        "--strict", action="store_true", help="fail when a verdict is not recorded"
    )
    ing.add_argument("--uploader", default="", help="recorded as props['uploader']")
    ing.add_argument(
        "--bundle-url", default="", help="where the bundle is kept; the run_url default"
    )
    ing.add_argument("--trusted-job-prefix", action="append", default=[],
                     help="accept a Jenkins run key only from these jobs (repeatable)")  # fmt: skip
    ing.add_argument(
        "--expect-key",
        default="",
        help="its inbox folder: <artifact_id> or sha256-<digest>",
    )
    ing.add_argument(
        "--dry-run", action="store_true", help="check everything, write nothing"
    )

    args = parser.parse_args(argv)
    if args.cmd == "init":
        if not (args.artifact or args.artifact_id or args.image):
            parser.error("--artifact, --artifact-id or --image is required")
        if args.kind == "junit" and not (args.component and args.test_type):
            parser.error("a junit bundle needs --component and --test-type")
    return {"init": init, "seal": seal, "validate": validate, "ingest": ingest}[
        args.cmd
    ](args)


if __name__ == "__main__":
    sys.exit(main())
