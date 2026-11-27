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
half, which the v2-results relay runs. The sha256s in a bundle are self-attested: they detect
accidental corruption, not tampering. See the README, "Offline results bundles".
"""

import argparse
import base64
import contextlib
import getpass
import gzip
import hashlib
import json
import math
import netrc
import os
import platform
import re
import shutil
import socket
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as etree
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA_VERSION = 1
KIT = "spyre-clickhouse-ingest"
# What a sealer may claim about its own source; `ci` is assigned by the receiving channel only.
TRUST_LABELS = ("lab", "external")
CHANNEL_TRUST = ("external", "lab", "ci")  # least to most trusted
DEFAULT_TRUST = "external"
NO_LABEL = "none"
BUNDLE_FILE = "bundle.json"
FAILED, REJECTED, INCOMPLETE = 1, 2, 3
STATUS = {FAILED: "failed", REJECTED: "rejected", INCOMPLETE: "incomplete"}
MANUAL_PREFIX = "manual:"
# kind -> the extension of its results/ files.
KINDS = {
    "junit": ".xml",
    "vllm": ".json",
    "sendnn": ".txt",
    "benchmarks": ".json",
    "tables": ".json",
    "bench": ".json",
    "bench-definitions": ".json",
    "bench-qualification": ".json",
    "bench-invalidation": ".json",
    "bench-placement": ".json",
    "bench-reference": ".json",
}
# bundle kind -> result_uploads.format, where they differ.
UPLOAD_FORMAT = {"bench-reference": "reference"}
# kinds whose test_type is always perf
PERF_KINDS = ("vllm", "sendnn")
# Kinds that record no result of a tested artifact: they name no artifact, tier or arch, and only a
# CI-trusted channel (a lab one too, for bench-invalidation) may ingest them (bundle.py).
ADMIN_KINDS = (
    "bench-definitions",
    "bench-qualification",
    "bench-invalidation",
    "bench-placement",
)
# Keys the receiving channel sets itself; a bundle that carries one is refused, never overwritten.
RESERVED_RUN_KEYS = (
    "run.ref_only", "run.record_type",
    "run.trust", "run.actor", "run.channel", "run.upload_id", "run.writer", "run.bundle_sha256",
    "run.content_sha256", "run.row_sha256", "run.on_behalf_of", "run.lib_ref", "upload_id", "round",
)  # fmt: skip
RESERVED_PREFIXES = ("claim.",)
BENCH_API = "bench/"
MAX_REPEAT = 65535
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# A benchmarks bundle's tiers: perf numbers, or accuracy evals.
BENCHMARK_TEST_TYPES = ("perf", "quality")
VLLM_COMPONENT = "spyre-inference"
# serving: a concurrency sweep (serving_<model>_c<N>_n<N>_..._len<N>_seqs<N>).
RUN_MODES = ("latency", "throughput", "serve", "serving")
# Name token prefix -> identity key, as vllm.parse_input_shapes reads it; the last four only
# in sweep names, so they key a benchmark only there.
SHAPE_PREFIXES = (
    ("tp", "tensor_parallel"), ("in", "input_len"), ("out", "output_len"),
    ("c", "concurrency"), ("n", "num_prompts"), ("len", "max_model_len"), ("seqs", "max_num_seqs"),
)  # fmt: skip
# The host resource sampler's file: records of {test_name, metric, value[, model]}.
HOST_RESOURCES_SUFFIX = ".host_resources.json"
# A record carrying any of these is a vLLM bench result (native) or vLLM's pytorch format.
VLLM_SIGNATURES = (
    "avg_latency", "requests_per_second", "tokens_per_second", "request_throughput",
    "output_throughput",
)  # fmt: skip
ARTIFACT_KEYS = ("artifact_id", "image", "artifact")
ENV_BASE_URL = "ARTIFACTORY_BASE_URL"
ENV_REPO = "ARTIFACTORY_V2_RESULTS_REPO"
ENV_ROOT = "ARTIFACTORY_V2_RESULTS_ROOT"  # a path template with {arch}
WAIT_MINUTES = 40
POLL_SECONDS = 30


class BundleError(Exception):
    """The bundle cannot be ingested; `code` says whether a retry can help."""

    def __init__(self, message: str, code: int = REJECTED):
        super().__init__(message)
        self.code = code


def current_user() -> str:
    """The login name; a container UID with no passwd entry falls back to $USER/$LOGNAME, then the uid."""
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return (
            os.environ.get("USER")
            or os.environ.get("LOGNAME")
            or str(getattr(os, "getuid", lambda: "unknown")())
        )


def kit_version() -> str:
    try:
        from importlib.metadata import version

        return version(KIT)
    except Exception:  # noqa: BLE001 -- running from a bare offline.py
        return "unknown"


def effective_trust(label: str = "", channel: str = "") -> str:
    """The trust recorded: '' when neither the bundle nor the channel says. Otherwise the
    channel's level (the lowest, `external`, when a labelled bundle arrives without one),
    lowered by the bundle's label if that is lower. A bundle never raises its trust."""
    if not (label or channel):
        return ""
    levels = [t for t in (label, channel or DEFAULT_TRUST) if t]
    return min(levels, key=CHANNEL_TRUST.index)


def receipt_id(content_sha256: str, upload_id: str = "") -> str:
    """RCPT-XXXX-XXXX-XXXX-XXXX, from the bundle digest and the upload id; recomputable by
    whoever holds both."""
    h = (
        hashlib.sha256(f"{content_sha256}\0{upload_id}".encode())
        .hexdigest()[:16]
        .upper()
    )
    return "RCPT-" + "-".join(h[i : i + 4] for i in range(0, 16, 4))


def preflight(path: Path = None) -> dict:
    """What this ingest accepts, and when `path` names a bundle, whether it can read it."""
    out = {
        "kit": KIT, "kit_version": kit_version(), "schema_versions": [SCHEMA_VERSION],
        "kinds": list(KINDS), "trust_labels": list(TRUST_LABELS),
    }  # fmt: skip
    if path is None:
        return out
    with opened(path) as root:
        f = root / BUNDLE_FILE
        if not f.is_file():
            raise BundleError(f"no {BUNDLE_FILE} in {path}", INCOMPLETE)
        try:
            meta = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise BundleError(f"{BUNDLE_FILE} is not JSON: {err}") from None
        invalid = ""
        try:
            check(root)
        except BundleError as err:
            invalid = str(err)
    if not isinstance(meta, dict):
        raise BundleError(f"{BUNDLE_FILE} is not a JSON object")
    version = meta.get("schema_version", SCHEMA_VERSION)
    unknown = sorted(set(meta) - set(schema()["properties"]))
    producer = meta.get("producer") if isinstance(meta.get("producer"), dict) else {}
    out["bundle"] = {"schema_version": version, "producer": producer,
                     "kind": meta.get("kind", "junit"), "trust": meta.get("trust", "")}  # fmt: skip
    problems = []
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        problems.append(
            f"schema_version {version!r} is newer than this ingest ({SCHEMA_VERSION}): "
            "upgrade the connected ingest, or re-seal with an older kit"
        )
    if unknown:
        problems.append(
            f"fields this ingest does not know: {unknown}: upgrade the connected ingest, "
            "or re-seal with an older kit"
        )
    if invalid and not problems:
        problems.append(f"does not validate: {invalid}")
    out["compatible"] = not problems
    out["problems"] = problems
    return out


def schema() -> dict:
    return json.loads((Path(__file__).with_name("bundle.schema.json")).read_text())


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
    types = {"object": dict, "array": list, "string": str, "integer": int}
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
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key) + "-" + meta["test_type"]


def setting(value: str, flag: str, env: str) -> str:
    """`value`, else $env; a deployment-specific setting has no built-in default."""
    value = value or os.environ.get(env, "")
    if not value:
        raise BundleError(f"no {flag}: pass it or set ${env}", FAILED)
    return value


def results_root(meta: dict, root: str = "") -> str:
    """The v2-results folder for the bundle's arch; a placeholder when unconfigured, for display."""
    root = root or os.environ.get(ENV_ROOT, "") or f"<{ENV_ROOT}>"
    return root.format(arch=meta.get("arch") or "any")


def upload_path(meta: dict, digest: str, inbox: str = "inbox", root: str = "") -> str:
    """Its inbox folder under the generic repo; a .tgz goes at this path + .tgz. A reference goes to
    `references/`, which the relay does not read: it is held for approval, never ingested on arrival."""
    if meta.get("kind") == "bench-reference":
        return f"{results_root(meta, root)}/references/{bundle_name(meta, digest)}"
    key = meta.get("artifact_id") or min(inbox_keys(meta), default="<artifact_id>")
    return f"{results_root(meta, root)}/{inbox}/{key}/{bundle_name(meta, digest)}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# The twin of vllm.REPEAT_DIR_PREFIX, spelled again rather than imported: this module seals
# bundles with the standard library alone (test_the_offline_path_needs_only_the_standard_library),
# and importing vllm would pull in clickhouse_connect.
REPEAT_DIR_PREFIX = "repeat-"


def _repeat_prefix(path: Path) -> str:
    """`repeat-003` when the file sits in a repeat subdir, else '' -- the staged path keeps it."""
    parent = path.parent.name
    stem = parent.removeprefix(REPEAT_DIR_PREFIX)
    return parent if parent.startswith(REPEAT_DIR_PREFIX) and stem.isdigit() else ""


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


def content_digest(root: Path) -> str:
    """The results' identity without bundle.json, so sealing the same results again (a new
    sealed time, another host) keeps it; the idempotency key of a bench upload."""
    lines = [f"{p}\0{sha256(root / p)}\n" for p in bundle_files(root)]
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
    """tp1_in64_out64 -> {tensor_parallel, input_len, output_len} (and a sweep's c/n/len/seqs),
    as the ingest reads them."""
    out = {}
    for token in name.split("_"):
        for prefix, key in SHAPE_PREFIXES:
            rest = token[len(prefix) :]
            if token.startswith(prefix) and rest.isdigit():
                out[key] = rest
    return out


def check_vllm(root: Path, meta: dict, results: list) -> None:
    """Each results/*.json is a vLLM bench result named as the runner names it, consistent
    with `perf`. The ingest re-reads them through the live leg's extraction."""
    perf = meta.get("perf", {})
    for p in results:
        host = Path(p).name.endswith(HOST_RESOURCES_SUFFIX)
        name = (
            Path(p)
            .name.removesuffix(".json")
            .removesuffix(".pytorch")
            .removesuffix(".host_resources")
        )
        if name.split("_")[0] not in RUN_MODES:
            raise BundleError(f"{p}: the name must start with one of {RUN_MODES}_")
        records = [r for r in read_records(root / p) if isinstance(r, dict)]
        if host:
            # The host resource sampler's records: {test_name, metric, value[, model]}.
            if not any("metric" in r and "value" in r for r in records):
                raise BundleError(f"{p}: no host resource metric in it")
        elif not any(
            any(k in r for k in VLLM_SIGNATURES) or ("benchmark" in r and "metric" in r)
            for r in records
        ):
            raise BundleError(f"{p}: no vLLM latency/throughput/serve metric in it")
        got = shapes(name)
        clash = [k for k in got if perf.get(k, got[k]) != got[k]]
        if clash:
            raise BundleError(f"{p}: its name's {clash} differ from perf {clash}")


def _finite(x) -> bool:
    try:
        return (
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        )
    except OverflowError:  # an integer too large for a float
        return False


def benchmark_entries(path: Path) -> list:
    """A benchmarks results file's rows, each checked; raises BundleError on the first bad one."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise BundleError(f"{path.name}: not JSON: {err}") from None
    rows = data.get("benchmarks") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise BundleError(f'{path.name}: needs a non-empty "benchmarks" list')
    for i, b in enumerate(rows):
        where = f"{path.name} benchmarks[{i}]"
        if (
            not isinstance(b, dict)
            or not isinstance(b.get("name"), str)
            or not b["name"].strip()
        ):
            raise BundleError(f"{where}: needs a name")
        unknown = set(b) - {
            "name",
            "props",
            "backend",
            "measurements",
            "iterations",
            "run_props",
            "tags",
        }
        if unknown:
            raise BundleError(f"{where}: unknown key(s) {sorted(unknown)}")
        for key in ("props", "run_props"):
            m = b.get(key, {})
            if isinstance(m, dict) and "run.trust" in m:
                raise BundleError(
                    f"{where}: run.trust is assigned by the receiving channel"
                )
            if not isinstance(m, dict) or not all(
                isinstance(v, (str, int, float)) and not isinstance(v, bool)
                for v in m.values()
            ):
                raise BundleError(f"{where}: {key} maps names to strings or numbers")
        m = b.get("measurements")
        if not isinstance(m, dict) or not m:
            raise BundleError(f"{where}: needs measurements")
        for k, v in m.items():
            values = v if isinstance(v, list) else [v]
            if not values or not all(_finite(x) for x in values):
                raise BundleError(
                    f"{where}: measurements[{k!r}] must be finite numbers"
                )
        if not isinstance(b.get("backend", ""), str) or not isinstance(
            b.get("tags", []), list
        ):
            raise BundleError(f"{where}: backend is a string and tags a list")
        n = b.get("iterations", 0)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise BundleError(f"{where}: iterations is a count")
    return rows


def check_benchmarks(root: Path, meta: dict, results: list) -> None:
    """Each results/*.json is a list of generic benchmark rows, for a perf or quality tier."""
    if meta["test_type"] not in BENCHMARK_TEST_TYPES:
        raise BundleError(
            f"a benchmarks bundle is test_type {' or '.join(BENCHMARK_TEST_TYPES)}"
        )
    if not meta.get("component"):
        raise BundleError("a benchmarks bundle names its component (the suite's owner)")
    for p in results:
        benchmark_entries(root / p)


def check_tables(root: Path, meta: dict, results: list) -> None:
    """Each results/<table>.json holds that table's rows; one bundle is one leg, so one test_type."""
    from . import schema as _schema
    from .tables import RowError, load_rows

    test_type, loaded = meta["test_type"], {}
    for p in results:
        try:
            table, rows = load_rows(root / p)
        except RowError as err:
            raise BundleError(str(err)) from None
        if any(
            isinstance(r, dict)
            and any("run.trust" in (r.get(k) or {}) for k in ("props", "run_props"))
            for r in rows
        ):
            raise BundleError(f"{p}: run.trust is assigned by the receiving channel")
        if table in loaded:
            raise BundleError(f"{table} appears twice")
        loaded[table] = rows
    capability = test_type in _schema.CAPABILITY_TYPE_VALUES
    if "benchmark_runs" in loaded and test_type not in BENCHMARK_TEST_TYPES:
        raise BundleError(
            f"benchmark_runs is test_type {' or '.join(BENCHMARK_TEST_TYPES)}, not {test_type!r}"
        )
    if "capability_runs" in loaded and not capability:
        raise BundleError(
            f"capability_runs is a capability test_type, not {test_type!r}"
        )
    if "test_case_runs" in loaded and (capability or test_type in BENCHMARK_TEST_TYPES):
        raise BundleError(
            f"test_case_runs is a functional test_type, not {test_type!r}"
        )
    for row in loaded.get("capability_runs", []):
        if row["test_type"] != test_type:
            raise BundleError(
                f"capability_runs row has test_type {row['test_type']!r}; the bundle is {test_type!r}"
            )
    verdicts = loaded.get("artifact_results", [])
    if len(verdicts) > 1:
        raise BundleError("artifact_results holds the leg's one verdict")
    if verdicts and verdicts[0]["test_type"] != test_type:
        raise BundleError(
            f"artifact_results has test_type {verdicts[0]['test_type']!r}; the bundle is {test_type!r}"
        )


def check_sendnn(root: Path, meta: dict, results: list) -> None:
    """Each results/*.txt is a perf-suite report_sendnn.txt: sections with a sendnn column."""
    if meta["test_type"] != "perf":
        raise BundleError("a sendnn bundle is test_type perf")
    for p in results:
        lines = [
            line.split() for line in (root / p).read_text(errors="replace").splitlines()
        ]
        sections = sum(
            1 for t in lines if t[:1] == ["Op:"] or t[:2] == ["Granite", "Config:"]
        )
        if not sections or not any(
            t[:1] == ["Metric"] and "sendnn" in t for t in lines
        ):
            raise BundleError(
                f"{p}: not a report_sendnn.txt (no Op:/Granite Config: sections with a sendnn column)"
            )


def _reserved(mapping, where: str) -> None:
    for k in mapping:
        if k in RESERVED_RUN_KEYS or str(k).startswith(RESERVED_PREFIXES):
            raise BundleError(
                f"{where}: {k!r} is set by the receiving channel, not carried in a bundle"
            )


def _strmap(value, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(v, (str, int, float)) and not isinstance(v, bool)
        for v in value.values()
    ):
        raise BundleError(f"{where}: maps names to strings or numbers")
    return {str(k): str(v) for k, v in value.items()}


def bench_result(path: Path) -> dict:
    """A bench Result file, checked for shape (the receiving side recomputes every id)."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise BundleError(f"{path.name}: not JSON: {err}") from None
    where = path.name
    if not (
        isinstance(data, dict)
        and data.get("kind") == "Result"
        and str(data.get("apiVersion", "")).startswith(BENCH_API)
        and isinstance(data.get("status"), dict)
    ):
        raise BundleError(
            f"{where}: not a bench Result (apiVersion {BENCH_API}*, kind Result)"
        )
    st = data["status"]
    case = st.get("case")
    if not isinstance(case, dict):
        raise BundleError(f"{where}: status.case is missing")
    for key in ("component", "name"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise BundleError(f"{where}: status.case.{key} is required")
    if not isinstance(case.get("disc_keys", []), list) or not all(
        isinstance(k, str) for k in case.get("disc_keys", [])
    ):
        raise BundleError(f"{where}: status.case.disc_keys is a list of strings")
    if not isinstance(case.get("tags", []), list):
        raise BundleError(f"{where}: status.case.tags is a list")
    _strmap(case.get("disc"), f"{where} status.case.disc")
    _strmap(case.get("props"), f"{where} status.case.props")
    subject = st.get("subject")
    if not (
        isinstance(subject, dict) and UUID_RE.match(str(subject.get("artifact_id", "")))
    ):
        raise BundleError(f"{where}: status.subject.artifact_id is required")
    env = st.get("environment")
    if not (isinstance(env, dict) and UUID_RE.match(str(env.get("env_id", "")))):
        raise BundleError(f"{where}: status.environment.env_id is required")
    version = env.get("fingerprint_version", 1)
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or not 0 < version <= 65535
    ):
        raise BundleError(f"{where}: status.environment.fingerprint_version is a count")
    _strmap(env.get("fingerprint"), f"{where} status.environment.fingerprint")
    _strmap(env.get("labels"), f"{where} status.environment.labels")
    _reserved(st.get("run_props") or {}, f"{where} status.run_props")
    _strmap(st.get("run_props"), f"{where} status.run_props")
    reps = st.get("repeats")
    if not isinstance(reps, list) or not reps:
        raise BundleError(f"{where}: status.repeats needs at least one repeat")
    seen = set()
    for i, r in enumerate(reps):
        at = f"{where} repeats[{i}]"
        n = r.get("run.repeat") if isinstance(r, dict) else None
        if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= MAX_REPEAT:
            raise BundleError(f"{at}: run.repeat is a count")
        if n in seen:
            raise BundleError(f"{at}: run.repeat {n} appears twice")
        seen.add(n)
        _reserved(r.get("props") or {}, f"{at} props")
        _strmap(r.get("props"), f"{at} props")
        _strmap(r.get("clean"), f"{at} clean")
        if r.get("started_at") is not None:
            try:
                if not 1970 <= _ts(str(r["started_at"])).year <= 2299:
                    raise ValueError("out of range")
            except ValueError:
                raise BundleError(f"{at}: started_at is not an ISO-8601 time") from None
        n_it = r.get("iterations", 0)
        if not isinstance(n_it, int) or isinstance(n_it, bool) or not 0 <= n_it < 2**32:
            raise BundleError(f"{at}: iterations is a count")
        pl_ = r.get("placement") or {}
        links = pl_.get("links", []) if isinstance(pl_, dict) else None
        if not isinstance(links, list) or not all(isinstance(x, str) for x in links):
            raise BundleError(f"{at}: placement.links is a list of strings")
        samples = r.get("samples") or {}
        if not isinstance(samples, dict):
            raise BundleError(f"{at}: samples maps a metric to its values")
        if r.get("status", "ok") == "ok" and not samples:
            raise BundleError(f"{at}: an ok repeat needs samples")
        for k, v in samples.items():
            values = v if isinstance(v, list) else [v]
            if not values or not all(_finite(x) for x in values):
                raise BundleError(f"{at}: samples[{k!r}] must be finite numbers")
        pl = r.get("placement") or {}
        if not isinstance(pl, dict) or not isinstance(pl.get("nodes", []), list):
            raise BundleError(f"{at}: placement.nodes is a list")
        devs = pl.get("devices", [])
        if not isinstance(devs, list) or not all(isinstance(d, dict) for d in devs):
            raise BundleError(f"{at}: placement.devices is a list of objects")
    return data


def bench_case_key(result: dict) -> str:
    c = result["status"]["case"]
    return json.dumps(
        [c["component"].strip().lower(), c["name"].strip().lower(),
         sorted(str(t).strip().lower() for t in c.get("tags") or []),
         sorted((c.get("disc") or {}).items())], default=str,
    )  # fmt: skip


def check_bench(root: Path, meta: dict, results: list) -> None:
    """Each results/*.json is one case's Result (apiVersion bench/*), for a perf or quality tier."""
    if meta["test_type"] not in BENCHMARK_TEST_TYPES:
        raise BundleError(
            f"a bench bundle is test_type {' or '.join(BENCHMARK_TEST_TYPES)}"
        )
    cases, subjects = set(), set()
    for p in results:
        result = bench_result(root / p)
        key = bench_case_key(result)
        if key in cases:
            raise BundleError(f"{p}: the same case appears in two Results")
        cases.add(key)
        subjects.add(result["status"]["subject"]["artifact_id"])
    named = meta.get("artifact_id")
    if len(subjects) > 1 or (named and subjects != {named}):
        raise BundleError(
            f"the Results name subject artifact(s) {sorted(subjects)}; bundle artifact_id {named!r}"
        )


def _records(path: Path, key: str) -> list:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise BundleError(f"{path.name}: not JSON: {err}") from None
    rows = data.get(key) if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise BundleError(f'{path.name}: needs a non-empty "{key}" list')
    if not all(isinstance(r, dict) for r in rows):
        raise BundleError(f'{path.name}: "{key}" holds objects')
    return rows


DEFINITION_KINDS = ("Environment", "NodeClass")
INVALIDATION_SCOPES = ("run", "run_benchmark", "node", "device", "placement")
INVALIDATION_STATES = ("suspect", "invalidated", "restored")


def check_admin(root: Path, meta: dict, results: list) -> None:
    """One results/*.json holding the records of a definitions, qualification or invalidation bundle."""
    kind = meta["kind"]
    if len(results) != 1:
        raise BundleError(f"a {kind} bundle holds exactly one results/*.json")
    path = root / results[0]
    if kind == "bench-definitions":
        for i, r in enumerate(_records(path, "records")):
            at = f"{path.name} records[{i}]"
            if r.get("kind") not in DEFINITION_KINDS:
                raise BundleError(
                    f"{at}: kind {r.get('kind')!r}; only {', '.join(DEFINITION_KINDS)} have a table so far"
                )
            for key in ("name", "spec_digest"):
                if not isinstance(r.get(key), str) or not r[key].strip():
                    raise BundleError(f"{at}: {key} is required")
            spec = r.get("spec", {})
            if not isinstance(spec, dict) or not all(
                isinstance(spec.get(k, {}), dict)
                for k in ("labels", "policy", "host_profile", "qualification")
            ):
                raise BundleError(
                    f"{at}: spec is an object whose labels, policy, host_profile and qualification are objects"
                )
        source = json.loads(path.read_text()).get("source", {})
        if not isinstance(source, dict):
            raise BundleError(f"{path.name}: source is an object")
    elif kind == "bench-qualification":
        for i, r in enumerate(_records(path, "records")):
            at = f"{path.name} records[{i}]"
            if not (isinstance(r.get("node"), str) and r["node"]):
                raise BundleError(f"{at}: node is required")
            for key in ("env_id", "evidence_run_id"):
                if not UUID_RE.match(str(r.get(key, ""))):
                    raise BundleError(f"{at}: {key} is a UUID")
            if r.get("outcome") not in ("pass", "fail"):
                raise BundleError(f"{at}: outcome is pass or fail")
            for key in ("qualified_at", "expires_at"):
                if not _date_time(str(r.get(key, ""))):
                    raise BundleError(f"{at}: {key} is an ISO-8601 time with a zone")
            if r.get("metric") is not None and not (
                isinstance(r["metric"], str) and r["metric"]
            ):
                raise BundleError(
                    f"{at}: metric names the canary metric the CV is recomputed from"
                )
    elif kind == "bench-placement":
        for i, u in enumerate(_records(path, "updates")):
            at = f"{path.name} updates[{i}]"
            if not UUID_RE.match(str(u.get("run_id", ""))):
                raise BundleError(f"{at}: run_id is a UUID")
            reps = u.get("repeats")
            if reps is not None and not (
                isinstance(reps, list)
                and reps
                and all(
                    isinstance(n, int)
                    and not isinstance(n, bool)
                    and 0 <= n <= MAX_REPEAT
                    for n in reps
                )
            ):
                raise BundleError(
                    f"{at}: repeats is a non-empty list of counts (absent: every repeat of the run)"
                )
            _strmap(u.get("clean"), f"{at} clean")
            _reserved(u.get("clean") or {}, f"{at} clean")
            for key in ("nodes", "links"):
                if key in u and not (
                    isinstance(u[key], list) and all(isinstance(x, str) for x in u[key])
                ):
                    raise BundleError(f"{at}: {key} is a list of strings")
            if "devices" in u and not (
                isinstance(u["devices"], list)
                and all(isinstance(d, dict) for d in u["devices"])
            ):
                raise BundleError(f"{at}: devices is a list of objects")
            if not any(k in u for k in ("clean", "nodes", "devices", "links")):
                raise BundleError(
                    f"{at}: nothing to update (clean, nodes, devices or links)"
                )
    else:
        for i, e in enumerate(_records(path, "events")):
            at = f"{path.name} events[{i}]"
            if not UUID_RE.match(str(e.get("invalidation_id", ""))):
                raise BundleError(f"{at}: invalidation_id is a UUID")
            if e.get("scope_kind") not in INVALIDATION_SCOPES:
                raise BundleError(f"{at}: scope_kind is one of {INVALIDATION_SCOPES}")
            if e.get("state") not in INVALIDATION_STATES:
                raise BundleError(f"{at}: state is one of {INVALIDATION_STATES}")
            if not (isinstance(e.get("reason_code"), str) and e["reason_code"]):
                raise BundleError(f"{at}: reason_code is required")
            scope = e.get("scope") or {}
            if not isinstance(scope, dict):
                raise BundleError(f"{at}: scope is an object")
            need = {"run": ("run_id",), "run_benchmark": ("run_id", "benchmark_id"),
                    "node": ("node",), "device": ("device_serial",),
                    "placement": ("placement_id",)}[e["scope_kind"]]  # fmt: skip
            for key in need:
                if not scope.get(key):
                    raise BundleError(
                        f"{at}: scope_kind {e['scope_kind']} needs scope.{key}"
                    )
            for key in ("run_id", "benchmark_id"):
                if scope.get(key) is not None and not UUID_RE.match(str(scope[key])):
                    raise BundleError(f"{at}: scope.{key} is a UUID")
            if e["scope_kind"] == "run" and (
                scope.get("benchmark_id") or scope.get("repeat") is not None
            ):
                raise BundleError(
                    f"{at}: a run scope names neither benchmark_id nor repeat"
                )
            rep_n = scope.get("repeat")
            if rep_n is not None and (
                not isinstance(rep_n, int)
                or isinstance(rep_n, bool)
                or not 0 <= rep_n <= MAX_REPEAT
            ):
                raise BundleError(f"{at}: scope.repeat is a count")
            for key in ("window_from", "window_to"):
                if e.get(key) and not _date_time(str(e[key])):
                    raise BundleError(f"{at}: {key} is an ISO-8601 time with a zone")
            from .bench_invalidation import InvalidationError, normalize

            try:
                normalize(e)
            except InvalidationError as err:
                raise BundleError(f"{at}: {err}") from None


def check_reference(root: Path, meta: dict, results: list) -> None:
    """results/reference.json alone, valid; arch (noarch, or multi) and test_type come from it."""
    try:
        from . import reference
    except ImportError:
        raise BundleError(
            "a bench-reference bundle is checked by the package, not by offline.py copied alone"
        ) from None

    if results != [f"results/{reference.FILE}"]:
        raise BundleError(
            f"a bench-reference bundle holds results/{reference.FILE} only: {results}"
        )
    try:
        doc = reference.load(root / results[0])
    except (
        ValueError,
        RecursionError,
    ) as err:  # JSON syntax, bad encoding, a huge integer or a deep nest
        raise BundleError(
            f"{reference.FILE} is not usable JSON: {err!r}"[:300]
        ) from None
    try:
        errors = reference.validate(doc, root)
    except Exception as err:  # noqa: BLE001 -- validate is total; this is the last resort, never a traceback
        raise BundleError(f"{reference.FILE} is malformed: {err!r}") from None
    if errors:
        raise BundleError("; ".join(errors))
    meta["component"] = meta.get("component") or reference.COMPONENT
    if meta["component"] != reference.COMPONENT:
        raise BundleError(
            f"a bench-reference bundle is component {reference.COMPONENT}"
        )
    if meta["test_type"] not in ("", doc["class"]):
        raise BundleError(
            f"test_type {meta['test_type']!r} differs from the reference's class {doc['class']!r}"
        )
    if meta.get("arch", reference.arch_of(doc)) != reference.arch_of(doc):
        raise BundleError(f"arch must be {reference.arch_of(doc)} for these values")
    meta["test_type"], meta["arch"] = doc["class"], reference.arch_of(doc)


def check(root: Path) -> dict:
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
    errors = schema_errors(meta, schema())
    if errors:
        raise BundleError("; ".join(errors))
    meta = normalized(meta)
    kind = meta.get("kind", "junit")
    admin = kind in ADMIN_KINDS
    reference = kind == "bench-reference"
    if admin and (specs(meta) or meta.get("test_type") or meta.get("create_artifact")):
        raise BundleError(f"a {kind} bundle names no artifact or test_type")
    if reference and (specs(meta) or meta.get("create_artifact")):
        raise BundleError(
            "a bench-reference bundle names no artifact: it creates its own"
        )
    if not admin and not reference and not specs(meta):
        raise BundleError(f"{BUNDLE_FILE}: needs artifact_id or image")
    if not reference and meta.get("arch") in ("noarch", "multi"):
        raise BundleError("arch noarch and multi are for bench-reference bundles")
    if meta.get("image") and not meta.get("artifact_id") and not meta.get("arch"):
        errors.append("an image names its artifact only with arch")
    if meta.get("create_artifact") and not (meta.get("image") and meta.get("arch")):
        errors.append(
            "create_artifact records an image: it needs image (by digest) and arch"
        )
    meta["test_type"] = meta.get("test_type") or ("perf" if kind in PERF_KINDS else "")
    if admin:
        meta.pop("test_type")
    elif not meta["test_type"] and not reference:
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
    elif kind == "sendnn":
        check_sendnn(root, meta, results)
    elif kind == "benchmarks":
        check_benchmarks(root, meta, results)
    elif kind == "tables":
        check_tables(root, meta, results)
    elif kind == "bench":
        check_bench(root, meta, results)
    elif admin:
        check_admin(root, meta, results)
    elif kind == "bench-reference":
        check_reference(root, meta, results)
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
        except getattr(tarfile, "FilterError", ()) as err:
            raise BundleError(f"{path.name}: unsafe member ({err})") from None
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


def _plain(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = (
        0,
        0,
        0,
        "",
        "",
        0o644,
    )
    return info


def pack(root: Path, tgz: Path) -> Path:
    """A .tgz of the bundle whose bytes depend on its content alone (no times, owners or modes)."""
    with contextlib.ExitStack() as stack:
        raw = stack.enter_context(tgz.open("wb"))
        gz = stack.enter_context(
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
        )
        tar = stack.enter_context(tarfile.open(fileobj=gz, mode="w"))
        for p in [BUNDLE_FILE, *bundle_files(root)]:
            tar.add(root / p, arcname=p, filter=_plain)
    return tgz


def _suite_end(start: datetime, text: str) -> datetime:
    """`start` plus the suite's time; `start` itself when the time is unusable."""
    try:
        seconds = float(text or 0)
        if math.isfinite(seconds) and seconds >= 0:
            return start + timedelta(seconds=seconds)
    except (ValueError, OverflowError):
        pass
    return start


def _suite_window(paths) -> tuple:
    """(first start, last end, zone-less count) over the <testsuite>s' timestamps and times, as
    UTC. A zone-less timestamp is read as UTC, as the dashboard reads it."""
    starts, ends, naive, dropped = [], [], 0, 0
    for xml in paths:
        with contextlib.suppress(etree.ParseError, OSError):
            for suite in etree.parse(xml).getroot().iter("testsuite"):
                text = suite.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
                    naive += ts.tzinfo is None
                    ts = (
                        ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                    ).astimezone(timezone.utc)
                except (ValueError, OverflowError):
                    dropped += bool(text)
                    continue
                starts.append(ts)
                ends.append(_suite_end(ts, suite.get("time", "0")))
    if dropped:
        print(
            f"[warn] {dropped} JUnit timestamp(s) could not be read and were ignored "
            "(Python before 3.11 reads only some ISO-8601 forms)",
            file=sys.stderr,
        )
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (
        min(starts).strftime(fmt) if starts else "",
        max(ends).strftime(fmt) if ends else "",
        naive,
    )


_SUITE_TAG = re.compile(rb"<testsuite\b(?:\"[^\"]*\"|'[^']*'|[^>\"'])*>")
_HOSTNAME = re.compile(rb"(\shostname\s*=\s*)(?:\"[^\"]+\"|'[^']+')")


def _scrub_hostnames(root: Path, host: str) -> tuple:
    """Blank every non-empty hostname attribute of the <testsuite> tags in results/*.xml, byte
    for byte otherwise. Returns (attributes blanked, files that still mention `host`)."""
    blanked, mentions = 0, []
    for p in sorted((root / "results").iterdir()):
        data = p.read_bytes()
        new, n = data, 0
        if p.suffix.lower() == ".xml":

            def tag(m):
                nonlocal n
                out, k = _HOSTNAME.subn(rb'\1""', m.group(0))
                n += k
                return out

            new = _SUITE_TAG.sub(tag, data)
        if n:
            p.chmod(p.stat().st_mode | 0o200)
            p.write_bytes(new)
            blanked += n
        if any(n.encode() in new for n in {host, host.split(".")[0]} if n):
            mentions.append(p.name)
    return blanked, mentions


def _unreserved(bag, dropped: list) -> None:
    for k in [
        k for k in bag if k in RESERVED_RUN_KEYS or str(k).startswith(RESERVED_PREFIXES)
    ]:
        dropped.append(k)
        del bag[k]


def bench_inputs(src: Path, stage: Path) -> tuple:
    """(staged Result files, attachments, facts) from a run directory, a folder of Results or one
    Result. The engine writes channel-owned keys (run.trust) into its Results; a sealed copy drops
    them, since the receiving channel assigns them."""
    if src.is_file():
        found = [src]
    else:
        found = sorted(
            p
            for p in src.rglob("result.json")
            if "ingest" not in p.relative_to(src).parts
        )
        found = found or sorted(src.glob("*.json"))
    paths, subjects, runs, dropped = [], set(), set(), []
    for p in found:
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise BundleError(f"{p}: not JSON: {err}") from None
        if not (isinstance(data, dict) and data.get("kind") == "Result"):
            if src.is_file():
                raise BundleError(f"{p}: not a bench Result")
            continue
        st = data.get("status") if isinstance(data.get("status"), dict) else {}
        _unreserved(st.get("run_props") or {}, dropped)
        for r in st.get("repeats") or []:
            if isinstance(r, dict):
                _unreserved(r.get("props") or {}, dropped)
        text = json.dumps(data, indent=2, sort_keys=True) + "\n"
        meta_block = (
            data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        )
        name = str(meta_block.get("name") or p.parent.name)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80]
        out = stage / f"{slug}-{hashlib.sha256(text.encode()).hexdigest()[:8]}.json"
        out.write_text(text)
        paths.append(out)
        subjects.add(str((st.get("subject") or {}).get("artifact_id", "")))
        runs.add(name.split(".")[0])
    if not paths:
        raise BundleError(f"no bench Result (kind Result) found under {src}")
    if dropped:
        print(
            f"[warn] dropped channel-owned key(s) from the sealed Results: {sorted(set(dropped))}",
            file=sys.stderr,
        )
    if len(subjects) > 1:
        raise BundleError(
            f"the Results name different subject artifacts: {sorted(subjects)}"
        )
    one = next(iter(runs))
    key = (
        f"{MANUAL_PREFIX}spyre-bench:{one}"
        if len(runs) == 1 and UUID_RE.match(one)
        else ""
    )
    return paths, [], {"artifact_id": next(iter(subjects)), "run_key": key}


def _arch(value: str) -> str:
    a = (value or platform.machine()).strip().lower()
    return "x86_64" if a in ("amd64", "x86", "x86-64") else a


def write(args, validate_times: bool = True) -> Path:
    """The bundle these `results` flags describe, written to --out (a folder, or a .tgz)."""
    kind = "vllm" if (args.kind == "vllm" or args.vllm_results_dir) else args.kind
    if getattr(args, "benchmarks_dir", ""):
        kind = "benchmarks"
    if getattr(args, "tables_dir", ""):
        kind = "tables"
    if getattr(args, "bench_dir", ""):
        kind = "bench"
    if getattr(args, "reference_dir", ""):
        kind = "bench-reference"
    admin = kind in ADMIN_KINDS
    staged = None
    if kind == "bench":
        if not args.bench_dir:
            raise BundleError("--kind bench needs --bench-dir")
        staged = Path(tempfile.mkdtemp(prefix="bench-stage-"))
        results, extra, bench = bench_inputs(Path(args.bench_dir), staged)
    elif admin:
        if not getattr(args, "records_file", ""):
            raise BundleError(f"--kind {kind} needs --records-file")
        results, extra = [Path(args.records_file)], []
    elif kind == "bench-reference":
        if not args.reference_dir:
            raise BundleError("--kind bench-reference needs --reference-dir")
        src = Path(args.reference_dir)
        results = [src / "reference.json"]
        extra = sorted(p for p in (src / "attachments").glob("*") if p.is_file())
    elif kind == "tables":
        if not args.tables_dir:
            raise BundleError("--kind tables needs --tables-dir")
        results, extra = sorted(Path(args.tables_dir).glob("*.json")), []
    elif kind == "benchmarks":
        if not args.benchmarks_dir:
            raise BundleError("--kind benchmarks needs --benchmarks-dir")
        results, extra = sorted(Path(args.benchmarks_dir).glob("*.json")), []
    elif kind == "sendnn":
        if not args.sendnn_report:
            raise BundleError("--kind sendnn needs --sendnn-report <report_sendnn.txt>")
        results, extra = [Path(args.sendnn_report)], []
    elif kind == "vllm":
        if not args.vllm_results_dir:
            raise BundleError("--kind vllm needs --vllm-results-dir")
        src = Path(args.vllm_results_dir)
        # The repeat-NNN subdirs too: a repeated benchmark writes one set of files per repeat.
        results = sorted(p for g in ("*.json", "repeat-*/*.json") for p in src.glob(g))
        extra = sorted(
            p
            for g in ("*", "repeat-*/*")
            for p in src.glob(g)
            if p.is_file() and p.suffix in (".cmd", ".log")
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
    if kind == "bench" and not any(named.values()):
        named["artifact_id"] = bench["artifact_id"]
    meta = {
        "schema_version": SCHEMA_VERSION,
        **({"kind": kind} if kind != "junit" else {}),
    }
    if not admin:
        meta.update(normalized(named), arch=_arch(args.arch))
    if kind == "bench-reference":
        meta.pop("arch")  # derived from the values (noarch, or multi)
    for key, value in (
        ("component", args.component), ("test_type", args.trigger_type or ("perf" if kind == "bench" else "")),
        ("run_key", args.jenkins_run_key or (bench["run_key"] if kind == "bench" else "")),
        ("run_url", args.run_url), ("workflow", args.workflow), ("tag_family", args.tag_family),
    ):  # fmt: skip
        if value and not (admin and key != "run_url"):
            meta[key] = value
    if args.tags:
        meta["tags"] = list(args.tags)
    if args.perf:
        meta["perf"] = dict(args.perf)
    first, last, naive = _suite_window(results) if kind == "junit" else ("", "", 0)
    if naive and not (args.triggered_at and getattr(args, "ended_at", "")):
        print(
            f"[warn] {naive} JUnit timestamp(s) carry no time zone; read as UTC "
            "(--triggered-at / --ended-at override)",
            file=sys.stderr,
        )
    given_end = getattr(args, "ended_at", "")
    for flag, value in (
        ("--triggered-at", args.triggered_at),
        ("--ended-at", given_end),
    ):
        if validate_times and value and not _date_time(value):
            raise BundleError(f"{flag} {value!r} is not an ISO-8601 time with a zone")
    if args.triggered_at or first:
        meta["started_at"] = args.triggered_at or first
    if given_end or last:
        meta["ended_at"] = given_end or last
    if (
        _date_time(meta.get("ended_at", ""))
        and _date_time(meta.get("started_at", ""))
        and _ts(meta["ended_at"]) < _ts(meta["started_at"])
    ):
        if given_end:
            raise BundleError("--ended-at is before the start time")
        print(
            f"[warn] the last suite ends before the start time {meta['started_at']}; "
            "ended_at set to the start",
            file=sys.stderr,
        )
        meta["ended_at"] = meta["started_at"]
    trust = getattr(args, "seal_trust", DEFAULT_TRUST)
    if (
        trust and trust != NO_LABEL and not admin
    ):  # a CI-owned bundle carries no label that could lower its channel
        meta["trust"] = trust
    meta["producer"] = {"name": KIT, "version": kit_version()}
    runner = {}
    if not getattr(args, "redact_host", False):
        runner["host"] = socket.gethostname()
    if not getattr(args, "redact_user", False):
        runner["user"] = current_user()
    if runner:
        meta["runner"] = runner
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
        if kind == "bench-invalidation":
            dest = root / "results" / "invalidation.json"
        else:
            # Keep a repeat's own subdir: flattening would make its files collide with the next repeat's.
            dest = root / "results" / _repeat_prefix(p) / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    if staged:
        shutil.rmtree(staged, ignore_errors=True)
    if getattr(args, "redact_host", False):
        blanked, mentions = _scrub_hostnames(root, socket.gethostname())
        print(
            f"[info] blanked {blanked} testsuite hostname attribute(s)", file=sys.stderr
        )
        if mentions:
            print(
                f"[warn] the sealing host name is still in {mentions}; --redact-host "
                "only blanks the testsuite hostname attribute",
                file=sys.stderr,
            )
    if extra:
        (root / "attachments").mkdir(exist_ok=True)
        for p in extra:
            dest = root / "attachments" / _repeat_prefix(p) / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
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
    if not admin:
        print(
            f"[info] inbox path: {upload_path(checked, dig, args.inbox, args.root)}{'.tgz' if tgz else '/'}",
            file=sys.stderr,
        )
    return out


def report(path: Path) -> dict:
    """What `--validate-only` prints: the bundle's run identity and where it uploads."""
    with opened(path) as root:
        meta, dig, content = check(root), digest(root), content_digest(root)
    if meta.get("kind") in ADMIN_KINDS:
        return {
            "status": "valid",
            "kind": meta["kind"],
            "bundle_sha256": dig,
            "content_sha256": content,
        }
    key = run_key(meta)
    out = {"status": "valid", "kind": meta.get("kind", "junit"), "test_type": meta["test_type"],
           "run_key": key or f"{MANUAL_PREFIX}<uploader>:{dig}", "bundle_sha256": dig,
           "upload_path": upload_path(meta, dig)}  # fmt: skip
    if meta.get("kind") == "bench":
        out["content_sha256"] = content
    if key and meta.get("arch"):
        with contextlib.suppress(ImportError):
            from spyre_clickhouse_ingest.identity import RunId

            source = "bundle" if is_manual(key) else "jenkins"
            out["run_id"] = RunId.derive(source, key, meta["arch"], meta["test_type"])
    return out


class Artifactory:
    """PUT and GET against the generic repo, with retries; auth from a token or ~/.netrc."""

    def __init__(self, base: str, repo: str, token: str = ""):
        self.base = setting(base, "--base-url", ENV_BASE_URL).rstrip("/")
        self.repo = setting(repo, "--repo", ENV_REPO)
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
    results = setting(args.root, "--root", ENV_ROOT)
    with opened(path) as root:
        meta, dig = check(root), digest(root)
        if meta.get("kind") in ADMIN_KINDS:
            raise BundleError(
                f"a {meta['kind']} bundle is ingested by its CI job, not uploaded to the inbox"
            )
        if args.wait and meta.get("kind") == "bench-reference":
            raise BundleError(
                "--wait does not apply to a bench-reference bundle: it is held for approval, not relayed"
            )
        dest = upload_path(meta, dig, args.inbox, results)
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
        return wait(art, meta, dig, args.wait, args.as_folder, args.inbox, results)
    return json.dumps({"status": "uploaded", "url": url, "bundle_sha256": dig})


def wait(
    art, meta: dict, dig: str, minutes: int, as_folder=False, inbox="inbox", results=""
) -> str:
    """Poll until the relay has moved the bundle to processed/ or rejected/, beside the inbox."""
    key = meta.get("artifact_id") or min(inbox_keys(meta))
    parent = inbox.rpartition("/")[0]
    root = results_root(meta, results) + (f"/{parent}" if parent else "")
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
    g.add_argument(
        "--base-url", default="", help=f"Artifactory URL; default ${ENV_BASE_URL}"
    )
    g.add_argument(
        "--inbox",
        default="inbox",
        help="under the results root; a trial uses inbox-test/inbox",
    )
    g.add_argument("--repo", default="", help=f"generic repo; default ${ENV_REPO}")
    g.add_argument("--root", default="",
                   help=f"results folder template with {{arch}}; default ${ENV_ROOT}")  # fmt: skip
    g.add_argument("--kind", choices=tuple(KINDS), default="junit")
    g.add_argument("--ended-at", default="",
                   help="--offline: ISO-8601 end time; default the last JUnit suite's end, else none")  # fmt: skip
    g.add_argument("--seal-trust", choices=(*TRUST_LABELS, NO_LABEL), default=DEFAULT_TRUST,
                   help=f"--offline: where the results came from (default {DEFAULT_TRUST}); "
                   "the receiving channel decides the trust it records, and never accepts `ci` from a bundle; "
                   f"`{NO_LABEL}` writes no label, for a CI job whose channel is trusted")  # fmt: skip
    g.add_argument("--redact-host", action="store_true",
                   help="--offline: leave the sealing host out of bundle.json and blank the JUnit testsuite hostname")  # fmt: skip
    g.add_argument("--redact-user", action="store_true",
                   help="--offline: leave the sealing user out of bundle.json")  # fmt: skip
    g.add_argument("--channel-trust", choices=CHANNEL_TRUST, default="",
                   help="--from-bundle: the trust the receiving channel assigns (ci, lab, external); "
                   "the bundle's own label can only lower it")  # fmt: skip
    g.add_argument("--check", nargs="?", const=True, default=None, metavar="BUNDLE",
                   help="print what this ingest accepts; with BUNDLE, whether it can read that bundle")  # fmt: skip
    g.add_argument(
        "--vllm-results-dir", default="", help="vLLM bench *.json (implies --kind vllm)"
    )
    g.add_argument(
        "--sendnn-report",
        default="",
        help="--kind sendnn: the perf suite's report_sendnn.txt",
    )
    g.add_argument(
        "--benchmarks-dir",
        default="",
        help='--kind benchmarks: *.json of {"benchmarks": [...]} rows (implies --kind benchmarks)',
    )
    g.add_argument(
        "--bench-dir",
        default="",
        help="--kind bench: an engine results directory (its result.json files), a folder of "
        "Results or one Result file (implies --kind bench)",
    )
    g.add_argument(
        "--records-file",
        default="",
        help="--kind bench-definitions|bench-qualification|bench-invalidation: the records JSON",
    )
    g.add_argument(
        "--tables-dir",
        default="",
        help='--kind tables: <table>.json files of {"rows": [...]} (implies --kind tables)',
    )
    g.add_argument(
        "--reference-dir",
        default="",
        help="--kind bench-reference: a folder with reference.json and attachments/ (implies --kind bench-reference)",
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
    g.add_argument("--bundle-prop", action="append", type=_pair, default=[],
                   help="--from-bundle: k=v added to the verdict's props (the source keys win)")  # fmt: skip
    g.add_argument("--append-to-run", action="store_true",
                   help="--from-bundle: add this bundle's files to the manual run it names, "
                   "which already has results (needs --bundle-prop upload_id=<id>)")  # fmt: skip
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
        if args.channel_trust:
            raise BundleError(
                "--channel-trust applies to --from-bundle, not sealing or uploading"
            )
        if args.check is not None:
            if args.offline or args.from_bundle or args.upload or args.validate_only:
                raise BundleError(
                    "--check stands alone: drop --offline/--upload/--from-bundle"
                )
            if args.check == "":
                raise BundleError("--check wants a bundle path, or none at all")
            out = preflight(Path(args.check) if isinstance(args.check, str) else None)
            print(json.dumps(out, sort_keys=True))
            return 0 if out.get("compatible", True) else REJECTED
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
