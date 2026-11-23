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

"""`bench-reference` bundles: a comparison target on the existing tables.

A reference is a synthetic artifact (component `reference`, kind `generic`) whose values are
benchmark_runs rows flagged `run.ref_only=1` / `run.trust=external`, written once at approval.
Submission leaves nothing in those tables: the sealed bundle and the `pending` result_uploads
event are the record. This module holds the file format and its validation, the identity, the
lifecycle and the approval rules (all pure), and the writer that runs at approval.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .batching import BatchParked

COMPONENT = "reference"
TAG_FAMILY = "reference"
TAG_CURRENT = "ref-current"
TAG_RETIRED = "ref-retired"
ARCH_NOARCH = "noarch"
ARCH_MULTI = "multi"
FILE = "reference.json"
API_VERSION = "bench/v1alpha1"
KIND = "Reference"
ORIGINS = ("generated", "uploaded", "published")
CLASSES = ("quality", "perf")
MAX_VALUE_ROWS = 10_000
# artifacts.origin of a reference by its ref.origin: ours is built, anyone else's is pinned upstream content.
ARTIFACT_ORIGIN = {
    "generated": "built",
    "uploaded": "upstream",
    "published": "upstream",
}
LABEL = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")
# The minimum a label must not carry; deployments add their own patterns (one regex per line) in the
# file ENV_LABEL_DENYLIST names. A heuristic: the approver confirms.
LABEL_DENY = re.compile(r"(^|-)(node|host|worker)(-|$)|\.", re.I)
ENV_LABEL_DENYLIST = "SPYRE_BENCH_LABEL_DENYLIST"
# The run purposes whose results join baselines: the rows of benchmark_baseline_purposes (schema 62);
# a run with no purpose is eligible, any other purpose is not.
BASELINE_PURPOSES = ("nightly", "weekly", "merge")
KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{2,120}$")
BACKEND = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,60}$")
TOP_KEYS = {
    "apiVersion", "kind", "key", "origin", "class", "label", "terms", "as_of", "producer", "source_url",
    "model", "dataset", "tools", "system", "comparability", "source_run", "clean", "warnings", "pinning",
    "values", "assets",
}  # fmt: skip
# Written by the writer after the merge; a bundle that names one is refused.
RESERVED = (
    "version", "visibility", "trust", "ref_only", "submitted_by", "approved_by", "approved_at",
    "approval_reason", "terms_ack_by", "approvals", "upload_id",
)  # fmt: skip
COMPARABILITY = {
    "harness_match": ("yes", "partial", "no", "unknown"),
    "dataset_match": ("same", "subset", "different", "unknown"),
    "chat_template": ("applied", "not_applied", "unknown"),
}
COMPARABILITY_ALWAYS = (
    "harness",
    "harness_match",
    "precision",
    "dataset_subset",
    "dataset_match",
)
COMPARABILITY_CLASS = {
    "quality": ("fewshot", "chat_template"),
    "perf": ("shape.batch", "shape.input_len", "shape.output_len"),
}
COMPARABILITY_OPTIONAL = ("shape.max_model_len", "caveats")
UNKNOWN_OK = ("harness_match", "chat_template", "dataset_match")
CAVEATS_MAX = 500


# Destination column bounds, checked once at validation so an accepted file always inserts
# (schema/30: benchmark_runs.iterations is UInt32).
UINT32_MAX = 2**32 - 1


def norm_login(who) -> str:
    return str(who or "").strip().lower()


def _finite(x) -> bool:
    try:
        return (
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        )
    except OverflowError:
        return False


def _list(doc, key) -> list:
    """doc[key] as a list; a missing, null or non-list value is empty (validate reports it)."""
    value = doc.get(key) if isinstance(doc, dict) else None
    return value if isinstance(value, list) else []


def label_denylist(path=None) -> list:
    """Compiled patterns from the deny-list file (path, else $SPYRE_BENCH_LABEL_DENYLIST); none when unset."""
    path = path or os.environ.get(ENV_LABEL_DENYLIST, "")
    if not path:
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                out.append(re.compile(line, re.I))
            except re.error as err:
                raise ValueError(f"{path}: bad pattern {line!r}: {err}") from None
    return out


def _case_inputs(case: dict) -> tuple:
    """component, name, tags, disc and the ordered key list the id hashes; absent disc_keys = sorted(disc)."""
    props = {k: str(v) for k, v in (case.get("disc") or {}).items()}
    keys = case.get("disc_keys")
    keys = tuple(sorted(props)) if keys is None else tuple(keys)
    return (
        case["component"],
        case["name"].strip(),
        list(case.get("tags") or []),
        props,
        keys,
    )


def case_id(case: dict) -> str:
    """The benchmark_id a case block names, recomputed by the ingest rule."""
    from .identity import BenchmarkId

    return str(BenchmarkId.derive(*_case_inputs(case)))


def load(path) -> dict:
    return json.loads(Path(path).read_text())


def canonical(doc) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def archs_of(doc: dict) -> list:
    """The named architectures a reference's values were measured on (case disc `arch`, else none)."""
    return sorted(
        {
            str((v.get("case") or {}).get("disc", {}).get("arch", ""))
            for v in _list(doc, "values")
        }
        - {""}
    )


def arch_of(doc: dict) -> str:
    """noarch for a data file; multi only when the values name several architectures (D22.9)."""
    return ARCH_MULTI if len(archs_of(doc)) > 1 else ARCH_NOARCH


def content_digest(doc: dict) -> str:
    """sha256 over the reference's meaning: key, class, origin, system, dataset, model, comparability, values
    and assets. Terms, label, warnings and provenance text are excluded, so a re-send of the same content is
    the same artifact and a change of values, system or comparability is a new version."""
    meaning = {
        "key": doc.get("key"), "class": doc.get("class"), "origin": doc.get("origin"),
        "system": doc.get("system") or {}, "dataset": doc.get("dataset") or {}, "model": doc.get("model") or {},
        "comparability": doc.get("comparability") or {},
        "values": sorted(
            ({"id": v["case"].get("benchmark_id") or case_id(v["case"]), "backend": v.get("backend", ""),
              "m": {k: [float(x) for x in xs] for k, xs in (v.get("measurements") or {}).items()}, "n": v.get("iterations", 0)} for v in _list(doc, "values")),
            key=lambda r: (r["id"], r["backend"]),
        ),
        "assets": sorted(((a.get("sha256", ""), a.get("records", 0), (a.get("case") or {}).get("benchmark_id", "")) for a in _list(doc, "assets"))),
    }  # fmt: skip
    return hashlib.sha256(canonical(meaning).encode()).hexdigest()


def identity(doc: dict, ref: str = ""):
    digest = content_digest(doc)
    from .identity import ArtifactIdentity

    return ArtifactIdentity(
        component=COMPONENT, artifact_name=doc["key"], id12=digest[:12], arch=arch_of(doc), kind="generic",
        ref=ref, content_digest=f"sha256:{digest}",
    )  # fmt: skip


def test_type_of(doc: dict) -> str:
    return doc["class"]


def run_id_for(doc: dict, submitter: str, upload_id: str) -> str:
    """The run the value rows and the verdict share; known at submit time, before any write."""
    from .identity import RunId

    return RunId.derive(
        "bundle",
        f"manual:{norm_login(submitter)}:{upload_id}",
        arch_of(doc),
        test_type_of(doc),
    )


def _validate(doc, root, label_deny) -> list:
    """Every problem in a reference.json (empty = valid); `root` also checks asset bytes.
    `label_deny`: extra label patterns (compiled), else the deny-list file of the environment."""
    if not isinstance(doc, dict):
        return [f"{FILE} is not a JSON object"]
    shape = _shape_errors(doc)
    if shape:
        return shape
    try:
        json.dumps(doc, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        return [f"{FILE} holds a lone surrogate, which cannot be stored"]
    errors = []
    extra = sorted(set(doc) - TOP_KEYS)
    bad = [k for k in extra if k in RESERVED]
    if bad:
        errors.append(f"reserved key(s) set by the writer, not the bundle: {bad}")
    extra = [k for k in extra if k not in RESERVED]
    if extra:
        errors.append(f"unknown key(s): {extra}")
    if doc.get("apiVersion") != API_VERSION or doc.get("kind") != KIND:
        errors.append(f"apiVersion/kind must be {API_VERSION}/{KIND}")
    origin, klass = doc.get("origin"), doc.get("class")
    if origin not in ORIGINS:
        errors.append(f"origin must be one of {ORIGINS}")
    if klass not in CLASSES:
        errors.append(f"class must be one of {CLASSES}")
    if not KEY.fullmatch(str(doc.get("key", ""))):
        errors.append("key must be a slug: [a-z0-9][a-z0-9._-]{2,120}")
    label = str(doc.get("label", ""))
    deny = [LABEL_DENY, *(label_denylist() if label_deny is None else label_deny)]
    if not LABEL.fullmatch(label) or any(d.search(label) for d in deny):
        errors.append(
            "label must match [a-z0-9][a-z0-9-]{2,40} and carry no node, host or deny-listed name"
        )
    if not str(doc.get("terms", "")).strip():
        errors.append("terms is required (generated may say internal-generated)")
    errors += _provenance(doc, origin, klass)
    errors += _comparability(doc, origin, klass)
    values, assets = _list(doc, "values"), _list(doc, "assets")
    if not (values or assets):
        errors.append("a reference needs values or assets")
    if len(values) > MAX_VALUE_ROWS:
        errors.append(f"more than {MAX_VALUE_ROWS} value rows")
    seen = set()
    for i, v in enumerate(values):
        errors += _value(i, v, seen)
    taken = set()
    for i, a in enumerate(assets):
        errors += _asset(i, a, root)
        if isinstance(a, dict):
            for what, mark in (("path", a.get("path")), ("sha256", a.get("sha256")),
                               ("case", (a.get("case") or {}).get("benchmark_id") if isinstance(a.get("case"), dict) else None)):  # fmt: skip
                if mark is not None and (what, str(mark)) in taken:
                    errors.append(f"assets[{i}] repeats a {what}")
                taken.add((what, str(mark)))
    errors += _overlap(doc)
    if root is not None and (root / "attachments").is_dir():
        listed = {a.get("path") for a in assets if isinstance(a, dict)}
        loose = sorted(
            p.relative_to(root).as_posix()
            for p in (root / "attachments").rglob("*")
            if p.is_file() and p.relative_to(root).as_posix() not in listed
        )
        if loose:
            errors.append(f"attachments not listed in assets: {loose}")
    return errors


def validate(doc, root: Path | None = None, label_deny=None) -> list:
    """Every problem in a reference.json (empty = valid); `root` also checks asset bytes.
    `label_deny`: extra label patterns (compiled), else the deny-list file of the environment.
    Total: whatever the JSON holds, it returns errors and never raises."""
    try:
        return _validate(doc, root, label_deny)
    except (
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        OverflowError,
        OSError,
        RecursionError,
    ) as err:
        return [f"{FILE} is malformed: {err!r}"]


def _shape_errors(doc) -> list:
    """Wrong JSON types, found before any rule that reads inside a block."""
    errors = []
    for k in ("model", "dataset", "system", "tools", "source_run", "comparability"):
        if doc.get(k) is not None and not isinstance(doc[k], dict):
            errors.append(f"{k} must be an object")
    for k in (
        "key",
        "origin",
        "class",
        "label",
        "terms",
        "producer",
        "source_url",
        "as_of",
        "pinning",
    ):
        if doc.get(k) is not None and not isinstance(doc[k], str):
            errors.append(f"{k} must be a string")
    for k in ("values", "assets", "warnings"):
        if doc.get(k) is not None and not isinstance(doc[k], list):
            errors.append(f"{k} must be a list")
    if isinstance(doc.get("tools"), dict) and not all(
        isinstance(v, str) for v in doc["tools"].values()
    ):
        errors.append("tools maps names to version strings")
    if isinstance(doc.get("system"), dict) and not all(
        isinstance(v, (str, int, float)) and not isinstance(v, bool)
        for v in doc["system"].values()
    ):
        errors.append("system holds strings and numbers only")
    if isinstance(doc.get("system"), dict) and not isinstance(
        doc["system"].get("backend", ""), str
    ):
        errors.append("system.backend must be a string")
    for k, blocks in (("values", doc.get("values")), ("assets", doc.get("assets"))):
        for i, b in enumerate(blocks if isinstance(blocks, list) else []):
            case = b.get("case") if isinstance(b, dict) else None
            if not isinstance(b, dict) or not isinstance(case, dict):
                errors.append(f"{k}[{i}] needs a case object")
            elif not (
                isinstance(case.get("component"), str)
                and isinstance(case.get("name"), str)
                and isinstance(case.get("tags", []), list)
                and all(isinstance(t, str) for t in case.get("tags", []))
                and isinstance(case.get("disc", {}), dict)
                and isinstance(case.get("benchmark_id", ""), str)
                and isinstance(b.get("backend", ""), str)
            ):
                errors.append(
                    f"{k}[{i}].case needs string component/name, a tags list and a disc object"
                )
    return errors


def _provenance(doc, origin, klass) -> list:
    errors = []
    if origin in ("uploaded", "published"):
        if not str(doc.get("producer", "")).strip():
            errors.append(f"an {origin} reference needs producer")
        errors += _as_of(doc.get("as_of"), origin)
    if origin == "published" and not str(doc.get("source_url", "")).strip():
        errors.append("a published reference needs source_url")
    if origin == "generated":
        if not (doc.get("tools") or {}):
            errors.append("a generated reference needs tools (name -> version)")
        if not str((doc.get("model") or {}).get("revision", "")).strip():
            errors.append("a generated reference needs model.revision")
    if klass == "quality" and not (doc.get("dataset") or {}).get("id"):
        errors.append("a quality reference needs dataset.id")
    if not isinstance(doc.get("system") or {}, dict) or not (
        doc.get("system") or {}
    ).get("backend"):
        errors.append("system.backend is required")
    elif not BACKEND.fullmatch(str(doc["system"]["backend"])):
        errors.append("system.backend must be a closed backend value or ref:<label>")
    clean = doc.get("clean")
    if clean not in (True, False, None):
        errors.append("clean must be true, false or null")
    if not isinstance(doc.get("warnings", []), list) or any(
        not isinstance(w, str) for w in doc.get("warnings", [])
    ):
        errors.append("warnings must be a list of codes")
    return errors


def _as_of(value, origin) -> list:
    try:
        when = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return [f"an {origin} reference needs as_of (ISO date)"]
    when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return ["as_of is in the future"] if when > datetime.now(timezone.utc) else []


def _comparability(doc, origin, klass) -> list:
    block = doc.get("comparability")
    if not isinstance(block, dict):
        return ["comparability is required"]
    errors = []
    needed = COMPARABILITY_ALWAYS + COMPARABILITY_CLASS.get(klass, ())
    for k in needed:
        if k not in block or block[k] in ("", None):
            errors.append(f"comparability.{k} is required")
    for k, allowed in COMPARABILITY.items():
        if k in block and block[k] not in allowed:
            errors.append(f"comparability.{k} must be one of {allowed}")
        if block.get(k) == "unknown" and origin == "generated":
            errors.append(
                f"comparability.{k} cannot be unknown for a generated reference"
            )
    for k in block:
        if k not in needed and k not in COMPARABILITY_OPTIONAL:
            errors.append(f"unknown comparability key {k!r}")
    caveats = block.get("caveats")
    if origin in ("uploaded", "published") and not str(caveats or "").strip():
        errors.append(
            f"an {origin} reference needs comparability.caveats (may say none)"
        )
    if origin == "published" and str(caveats or "").strip().lower() == "none":
        errors.append("a published reference needs non-empty caveats")
    if caveats and len(str(caveats)) > CAVEATS_MAX:
        errors.append(f"comparability.caveats is over {CAVEATS_MAX} characters")
    return errors


def _overlap(doc) -> list:
    """An asset's record-count row and a value row on the same (case, backend) would merge into one."""
    backend = (doc.get("system") or {}).get("backend", "")
    taken = {
        (v["case"].get("benchmark_id"), v.get("backend", ""))
        for v in _list(doc, "values")
        if isinstance(v, dict) and isinstance(v.get("case"), dict)
    }
    return [
        f"assets[{i}] and a value row share case {a['case'].get('benchmark_id')} on backend {backend}"
        for i, a in enumerate(_list(doc, "assets"))
        if isinstance(a, dict)
        and isinstance(a.get("case"), dict)
        and (a["case"].get("benchmark_id"), backend) in taken
    ]


def _id_errors(where, case) -> list:
    """The case's id recomputed by the ingest rule; skipped where the identity module cannot load
    (a standard-library-only offline host): the writer recomputes it again at approval."""
    keys = case.get("disc_keys")
    if keys is not None and not (
        isinstance(keys, list) and all(isinstance(k, str) for k in keys)
    ):
        return [f"{where}.case.disc_keys must be a list of strings"]
    try:
        from .identity import BenchmarkId

        if "kernel_name" in (
            keys if keys is not None else (case.get("disc") or {})
        ) and BenchmarkId.kernel_stem((case.get("disc") or {}).get("kernel_name")):
            return [
                f"{where}.case.disc.kernel_name is a compiled-kernel name, which the writer re-keys"
            ]
        ok = case_id(case) == case["benchmark_id"]
    except ImportError:
        return []
    except Exception as err:  # noqa: BLE001 -- a malformed case block is a validation error
        return [f"{where}.case: {err}"]
    return (
        []
        if ok
        else [f"{where}.case.benchmark_id does not match its component/name/tags/disc"]
    )


def _value(i, v, seen) -> list:
    where = f"values[{i}]"
    if not isinstance(v, dict):
        return [f"{where} is not an object"]
    case = v.get("case")
    errors = []
    if (
        not isinstance(case, dict)
        or not case.get("component")
        or not str(case.get("name", "")).strip()
    ):
        return [f"{where}.case needs component and name"]
    if not case.get("benchmark_id"):
        errors.append(f"{where}.case.benchmark_id is required")
    else:
        errors += _id_errors(where, case)
    backend = v.get("backend", "")
    if not BACKEND.fullmatch(str(backend)):
        errors.append(f"{where}.backend must be a closed backend value or ref:<label>")
    if (case.get("benchmark_id"), backend) in seen:
        errors.append(f"{where} repeats (benchmark_id, backend)")
    seen.add((case.get("benchmark_id"), backend))
    m = v.get("measurements")
    if not isinstance(m, dict) or not m:
        errors.append(f"{where}.measurements needs at least one metric")
    else:
        for k, xs in m.items():
            if not isinstance(xs, list) or not xs or not all(_finite(x) for x in xs):
                errors.append(
                    f"{where}.measurements.{k} must be a non-empty list of finite numbers"
                )
    if (
        not isinstance(v.get("iterations"), int)
        or isinstance(v["iterations"], bool)
        or not 1 <= v["iterations"] <= UINT32_MAX
    ):
        errors.append(f"{where}.iterations (n) must be 1..4294967295")
    return errors


def _asset(i, a, root) -> list:
    where = f"assets[{i}]"
    errors = []
    path_text = str(a.get("path", ""))
    if (
        not isinstance(a, dict)
        or not re.fullmatch(r"attachments/[^\s/][^\s]*", path_text)
        or ".." in path_text.split("/")
        or "\\" in path_text
        or "\x00" in path_text
    ):
        return [f"{where}.path must be a plain path under attachments/"]
    if not re.fullmatch(r"[0-9a-f]{64}", str(a.get("sha256", ""))):
        errors.append(f"{where}.sha256 must be 64 hex")
    if (
        not isinstance(a.get("records"), int)
        or isinstance(a["records"], bool)
        or not 1 <= a["records"] <= UINT32_MAX
    ):
        errors.append(f"{where}.records must be 1..4294967295")
    case = a.get("case")
    if not isinstance(case, dict) or not case.get("benchmark_id"):
        errors.append(
            f"{where}.case with benchmark_id is required (the parity case the bytes are for)"
        )
    else:
        errors += _id_errors(where, case)
    if root is not None and not errors:
        path = root / a["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            errors.append(f"{where}: {a['path']} is a link or leaves the bundle")
        elif not path.is_file():
            errors.append(f"{where}: {a['path']} is not in the bundle")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != a["sha256"]:
            errors.append(f"{where}: sha256 does not match {a['path']}")
        elif a["path"].endswith((".jsonl", ".csv")):
            lines = sum(1 for ln in path.read_bytes().splitlines() if ln.strip())
            if lines != a["records"]:
                errors.append(
                    f"{where}: {a['path']} has {lines} record(s), not {a['records']}"
                )
    return errors


def value_entries(doc: dict) -> list:
    """Reference values as the benchmark writer's entries: every case hashes as the case itself does."""
    out = []
    for v in _list(doc, "values"):
        out.append(
            _entry(v["case"], v.get("backend", ""), v["measurements"], v["iterations"])
        )
    for a in _list(doc, "assets"):
        out.append(_entry(a["case"], doc["system"]["backend"], {"ref.records": [a["records"]]}, a["records"],
                          {"ref.asset_only": "1", "ref.asset_sha256": a["sha256"]}))  # fmt: skip
    return out


def _entry(case, backend, measurements, n, extra=None) -> dict:
    component, name, tags, props, keys = _case_inputs(case)
    return {
        "name": name, "tags": tags, "backend": backend, "props": props, "disc": props, "disc_keys": keys,
        "measurements": {k: [float(x) for x in xs] for k, xs in measurements.items()},
        "iterations": n, "component": component, "run_props": dict(extra or {}),
    }  # fmt: skip


def case_ids(doc: dict) -> list:
    """(component, benchmark_id) of every case the reference has values or assets for."""
    return sorted(
        {
            (v["case"]["component"], v["case"]["benchmark_id"])
            for v in [*_list(doc, "values"), *_list(doc, "assets")]
        }
    )


def missing_cases(client, db: str, doc: dict) -> list:
    """benchmark_ids the reference names that `benchmarks` does not hold yet (the register-first check)."""
    ids = [bid for _, bid in case_ids(doc)]
    if not ids:
        return []
    have = {
        str(r[0])
        for r in client.query(
            f"SELECT DISTINCT benchmark_id FROM {db}.benchmarks WHERE benchmark_id IN {{ids:Array(UUID)}}",
            parameters={"ids": ids},
        ).result_rows
    }
    return [i for i in ids if i not in have]


def next_version(client, db: str, key: str) -> int:
    """1 + the highest version any artifact of the key holds, including an attempt that never completed
    (it keeps its number, so a retry resumes it and a later upload cannot take it); gaps are fine."""
    n = client.query(
        "SELECT max(toUInt32OrZero(v)) FROM (SELECT argMax(props['ref.version'], audit_timestamp) AS v "
        f"FROM {db}.artifacts WHERE component = '{COMPONENT}' AND props['ref.key'] = {{k:String}} GROUP BY artifact_id)",
        parameters={"k": key},
    ).result_rows[0][0]
    return int(n or 0) + 1


def _stored(client, db: str, aid: str):
    """(version, upload_id, submitted_by, approval props, props) of an artifact already recorded, else None.
    Props are the merged rows of the artifact, so a renumbering row wins over the first write."""
    rows = client.query(
        f"SELECT props['ref.version'], props['ref.upload_id'], props['ref.submitted_by'], props FROM {db}.v_artifacts "
        "WHERE artifact_id = {a:UUID} LIMIT 1",
        parameters={"a": aid},
    ).result_rows
    if not rows:
        return None
    props = rows[0][3]
    approval = {k: v for k, v in props.items() if k in APPROVAL_PROPS}
    return int(rows[0][0] or 0), rows[0][1], rows[0][2], approval, dict(props)


def _holders_of_version(client, db: str, key: str, version: int) -> list:
    """artifact ids whose CURRENT version of the key is this one, in the order they took it."""
    rows = client.query(
        "SELECT toString(artifact_id) FROM (SELECT artifact_id, argMax(props['ref.version'], audit_timestamp) AS v, "
        "maxIf(audit_timestamp, props['ref.version'] = {v:String}) AS took "
        f"FROM {db}.artifacts WHERE component = '{COMPONENT}' AND props['ref.key'] = {{k:String}} GROUP BY artifact_id) "
        "WHERE v = {v:String} ORDER BY took, artifact_id",
        parameters={"k": key, "v": str(version)},
    ).result_rows
    return [_uuid(r[0]) for r in rows]


def _renumber(client, db: str, ident, doc: dict, version: int) -> None:
    """A later row for the same artifact that moves it to another number (v_artifacts merges rows, the
    newest props win). Its audit_timestamp is explicitly later than every earlier row of the artifact, so
    a same-millisecond tie cannot leave the merged and the raw reads disagreeing."""
    last = client.query(
        f"SELECT max(audit_timestamp) FROM {db}.artifacts WHERE artifact_id = {{a:UUID}}",
        parameters={"a": ident.artifact_id},
    ).result_rows[0][0]
    if isinstance(last, str):
        last = datetime.fromisoformat(last)
    at = datetime.now(timezone.utc).replace(tzinfo=None)
    if last is not None:
        at = max(at, _utc(last).replace(tzinfo=None) + timedelta(milliseconds=1))
    client.insert("artifacts", [[
        ident.artifact_id, COMPONENT, ident.arch, "generic", ident.artifact_name, ARTIFACT_ORIGIN[doc["origin"]],
        [], [], [], {"ref.key": doc["key"], "ref.version": str(version)}, at,
    ]], column_names=["artifact_id", "component", "arch", "kind", "artifact_name", "origin", "identity_deps",
                      "context_deps", "sources", "props", "audit_timestamp"], database=db)  # fmt: skip


def _guard_rows(client, db: str, doc: dict, rid: str, version: int) -> None:
    if _rows_written(client, db, doc, rid):
        raise ReferenceError(
            f"version {version} of {doc['key']} is held by another reference and this attempt already wrote rows: "
            "re-send with changed content, or retract the attempt and send it again"
        )


def _renumber_stored(client, db: str, ident, doc: dict, version: int, rid: str) -> None:
    """A resumed artifact whose stored number was taken: moved only while it has written no value rows."""
    _guard_rows(client, db, doc, rid, version)
    _renumber(
        client, db, ident, doc, version
    )  # the stored row still says the old number


def _rows_written(client, db: str, doc: dict, rid: str) -> bool:
    from .writer import BenchmarkWriter

    return any(
        BenchmarkWriter.already_ingested(client, db, rid, component, "reference", FILE)
        for component, _ in case_ids(doc)
    )


def _newer_current(client, db: str, key: str, version: int) -> bool:
    """Does the key's slot (the view's rule: newest ref-retired row, or ref-current row of a published artifact)
    now point at a newer version, or retire the key? A resumed older write then leaves the slot alone."""
    rows = client.query(
        "SELECT tupleElement(slot, 1), toString(tupleElement(slot, 2)) FROM (SELECT argMax(tuple(t.tag, t.artifact_id), "
        f"(t.ts, t.audit_timestamp)) AS slot FROM {db}.artifact_tags AS t WHERE t.tag_family = '{TAG_FAMILY}' "
        "AND t.props['ref.key'] = {k:String} AND (t.tag = 'ref-retired' OR t.artifact_id IN "
        f"(SELECT artifact_id FROM {db}.v_result_uploads WHERE state = 'published' AND NOT is_round)))",
        parameters={"k": key},
    ).result_rows
    if not rows or not rows[0][0]:
        return False
    if rows[0][0] == TAG_RETIRED:
        return True
    held = client.query(
        f"SELECT props['ref.version'] FROM {db}.v_artifacts WHERE artifact_id = {{a:UUID}} LIMIT 1",
        parameters={"a": rows[0][1]},
    ).result_rows
    return bool(held) and int(held[0][0] or 0) > version


def _as_stored(doc: dict, props: dict) -> dict:
    """The doc with the wording the first attempt recorded (label, terms, cleanliness, warnings, pinning), so a
    resume cannot split an artifact's props from its value rows."""
    clean = {"1": True, "0": False}.get(props.get("ref.clean", ""))
    return {
        **doc, "label": props.get("ref.label", doc["label"]), "terms": props.get("ref.terms", doc["terms"]),
        "clean": clean, "warnings": [w for w in props.get("ref.warnings", "").split(",") if w],
        **({"pinning": props["ref.pinning"]} if props.get("ref.pinning") else {}),
    }  # fmt: skip


def _stored_ref(client, db: str, aid: str) -> str:
    rows = client.query(
        f"SELECT ref FROM {db}.artifact_refs WHERE artifact_id = {{a:UUID}} LIMIT 1",
        parameters={"a": aid},
    ).result_rows
    if rows:
        return rows[0][0]
    rows = client.query(
        f"SELECT props['ref'] FROM {db}.v_artifacts WHERE artifact_id = {{a:UUID}} LIMIT 1",
        parameters={"a": aid},
    ).result_rows
    return rows[0][0] if rows else ""


# ---- lifecycle -----------------------------------------------------------------------------------------

# result_uploads states: submitted = pending; approved and published are one step (approval is the write);
# retract is terminal. Events never rewrite an earlier one.
TRANSITIONS = {
    "submit": (None, "pending"),
    "approve": ("pending", "published"),
    "reject": ("pending", "rejected"),
    "request_changes": ("pending", "rejected"),
    "expire": ("pending", "expired"),
    "fail": ("pending", "failed"),
    "retry": ("failed", "pending"),
    "retract": (("pending", "published"), "retracted"),
}


def next_state(state, action: str) -> str:
    """The state `action` moves an upload to; raises ValueError when it is not allowed from `state`."""
    if action not in TRANSITIONS:
        raise ValueError(f"unknown action {action!r}")
    allowed, target = TRANSITIONS[action]
    allowed = allowed if isinstance(allowed, tuple) else (allowed,)
    if state not in allowed:
        raise ValueError(f"cannot {action} an upload that is {state!r}")
    return target


# ---- approval rules (pure) -----------------------------------------------------------------------------

SCOPE_DIMENSIONS = (
    "class",
    "origin",
    "component",
    "model_family",
    "dataset",
    "arch",
    "key",
)
APPROVAL_PROPS = (
    "ref.approved_by", "ref.approval_reason", "ref.approval_override", "ref.approval_self", "ref.terms_ack_by",
)  # fmt: skip
PERMISSION_APPROVE = "references:approve"
PERMISSION_ADMIN = "references:admin"
PERMISSION_SUBMIT = "references:submit"


def facts_of(doc: dict) -> dict:
    """The scope facts of a reference, taken from the validated file and never from form text."""
    comps = sorted({c for c, _ in case_ids(doc)})
    return {
        "class": [doc["class"]], "origin": [doc["origin"]], "component": comps,
        "model_family": [str((doc.get("model") or {}).get("family", "")).lower()] if (doc.get("model") or {}).get("family") else [],
        "dataset": [str((doc.get("dataset") or {}).get("id", ""))] if (doc.get("dataset") or {}).get("id") else [],
        "arch": archs_of(doc) or [arch_of(doc)], "key": [doc["key"]],
    }  # fmt: skip


def in_scope(scope: dict, facts: dict) -> bool:
    """ANDed over dimensions; a missing or empty dimension means any, a limited one fails closed
    when the facts carry no value for it. Several fact values (cases of two components) must all fit."""
    for dim, allowed in (scope or {}).items():
        if not allowed:
            continue
        allowed = {str(a).strip().lower() for a in allowed}
        have = [str(h).strip().lower() for h in facts.get(dim) or []]
        if not have or not all(h in allowed for h in have):
            return False
    return True


def _uuid(value) -> str:
    """A UUID in its one canonical spelling, whatever case or braces the caller used."""
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        return str(value).strip().lower()


def _utc(when: datetime) -> datetime:
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _live(binding: dict, now: datetime) -> bool:
    exp = binding.get("expires_at")
    never = (
        exp is None or _utc(exp).year <= 1970
    )  # schema/64: the epoch means no expiry
    return bool(binding.get("active", True)) and (never or _utc(exp) > _utc(now))


def holders(
    bindings, permission: str, facts: dict, members: dict | None = None, now=None
) -> set:
    """Logins holding `permission` in scope for `facts`: users bound directly, plus members of bound groups
    (`members`: group -> logins). A `*` role permission or `<resource>:*` counts."""
    now = now or datetime.now(timezone.utc)
    resource = permission.split(":")[0]
    out = set()
    for b in bindings:
        if not _live(b, now) or not in_scope(b.get("scope") or {}, facts):
            continue
        perms = b.get("permissions") or []
        if not (permission in perms or "*" in perms or f"{resource}:*" in perms):
            continue
        if b.get("subject_type", "user") == "group":
            out |= {norm_login(u) for u in (members or {}).get(b["subject"], ())}
        else:
            out.add(norm_login(b["subject"]))
    return out


def eligible_approvers(
    bindings, facts: dict, submitter: str, members=None, now=None
) -> set:
    """Approvers in scope other than the submitter (admins are separate: see decide_approval)."""
    return holders(bindings, PERMISSION_APPROVE, facts, members, now) - {
        norm_login(submitter)
    }


@dataclass(frozen=True)
class Approval:
    allowed: bool
    error: str = ""
    override: bool = False  # an admin approved without holding the approve binding in scope, or their own upload
    self_approved: bool = False
    props: dict = field(
        default_factory=dict
    )  # recorded on the artifact, the tag row and the upload event


def requires_approval(origin: str) -> bool:
    """Every reference waits for a human in v1, CI-generated included; relaxed per class later."""
    return True


def decide_approval(
    *, submitter: str, approver: str, is_admin: bool, approvers, reason: str = ""
) -> Approval:
    """May `approver` approve? `approvers`: logins holding references:approve in scope (submitter included).

    A non-admin needs the binding in scope and is never the submitter. An admin may approve any reference,
    including their own; each such approval is flagged (`ref.approval_override`, `ref.approval_self`) so the
    audit log lists them without needing a reason (D21.1).
    """
    who, sub = norm_login(approver), norm_login(submitter)
    if not who:
        return Approval(False, "no approver identity")
    bound = who in {norm_login(a) for a in approvers}
    selfie = who == sub
    if not is_admin:
        if not bound:
            return Approval(
                False, f"{who} holds no references:approve binding for this reference"
            )
        if selfie:
            return Approval(False, "the submitter cannot approve their own reference")
        return Approval(True, props=_approval_props(who, reason, False, False))
    override = selfie or not bound
    return Approval(
        True,
        override=override,
        self_approved=selfie,
        props=_approval_props(who, reason, override, selfie),
    )


def _approval_props(who, reason, override, selfie) -> dict:
    return {
        "ref.approved_by": who, "ref.approval_reason": reason or "",
        "ref.approval_override": "1" if override else "", "ref.approval_self": "1" if selfie else "",
    }  # fmt: skip


def decide_retract(
    *, submitter: str, actor: str, is_admin: bool, state: str
) -> Approval:
    """The submitter may retract a pending or published upload of their own; an admin any."""
    if state not in ("pending", "published"):
        return Approval(False, f"an upload that is {state!r} cannot be retracted")
    if is_admin or norm_login(actor) == norm_login(submitter):
        return Approval(True)
    return Approval(False, "only the submitter or an admin can retract")


def decide_reject(*, submitter: str, actor: str, is_admin: bool, approvers) -> Approval:
    """Reject or request changes: an approver in scope or an admin; a submitter withdraws with retract."""
    who = norm_login(actor)
    if who == norm_login(submitter) and not is_admin:
        return Approval(
            False, "a submitter withdraws their own pending upload with retract"
        )
    if is_admin or who in {norm_login(a) for a in approvers}:
        return Approval(True)
    return Approval(
        False, f"{who} holds no references:approve binding for this reference"
    )


def authorize_approval(doc: dict, *, submitter: str, approver: str, is_admin: bool, bindings,
                       members=None, reason: str = "", now=None) -> Approval:  # fmt: skip
    """The approval decision for a validated reference: scope facts from the file, the approvers from the
    bindings, then decide_approval. The one call an approval route makes before the writer."""
    facts = facts_of(doc)
    return decide_approval(
        submitter=submitter, approver=approver, is_admin=is_admin, reason=reason,
        approvers=holders(bindings, PERMISSION_APPROVE, facts, members, now),
    )  # fmt: skip


# ---- promoting a run to a reference (D21.3) -------------------------------------------------------------

BLOCKERS = ("invalidated", "retracted", "no_measurements")
WARNING_TEXT = {
    "not_clean": "the run was not clean (other load was present or exclusivity was not enforced)",
    "clean_unknown": "the run's cleanliness is unknown",
    "exclusivity_not_enforced": "exclusive access to the devices was observed, not enforced",
    "suspect": "the run's validity is suspect",
    "pinning_unverified": "CPU pinning was not verified",
    "single_repeat": "the run has a single repeat; variance is unknown",
    "not_baseline_purpose": "the run's purpose is not a baseline purpose (for example a PR check)",
    "unqualified_environment": "the environment was not qualified",
}


@dataclass(frozen=True)
class Promotion:
    ok: bool
    clean: str  # "1" | "0" | "unknown": recorded as ref.clean
    warnings: tuple = ()
    blockers: tuple = ()

    @property
    def needs_ack(self) -> bool:
        return bool(self.warnings)


def promotion_check(run: dict) -> Promotion:
    """Judge a stored run offered as a reference. A non-clean run is allowed with warnings (D21.3);
    an invalidated or empty one is not. `run` carries: invalidated, retracted, measured, validity,
    clean (the run.clean.* map: exclusive, load_clean, devices_exclusive), repeats, pinning_verified, purpose, qualified."""
    blockers = tuple(
        b
        for b in BLOCKERS
        if (b == "no_measurements" and not run.get("measured", True))
        or (b != "no_measurements" and run.get(b))
    )
    clean = run.get("clean") or {}
    exclusive = clean.get("exclusive") or "unknown"
    warnings = []
    # The engine's cleanliness rule (engine.run_clean), restated so the ingest does not import the engine.
    if exclusive == "enforced" or (
        exclusive == "observed"
        and clean.get("load_clean") is True
        and clean.get("devices_exclusive") == "enforced"
    ):
        state = "1"
    elif exclusive == "unknown":
        state = "unknown"
        warnings.append("clean_unknown")
    else:
        state = "0"
        warnings.append("not_clean")
        if exclusive == "observed":
            warnings.append("exclusivity_not_enforced")
    if run.get("validity") in ("suspect", "suspected"):
        warnings.append("suspect")
    if run.get("pinning_verified") is False:
        warnings.append("pinning_unverified")
    if (run.get("repeats") if run.get("repeats") is not None else 2) < 2:
        warnings.append("single_repeat")
    purpose = str(run.get("purpose") or "").strip().lower()
    if purpose and purpose not in BASELINE_PURPOSES:
        warnings.append("not_baseline_purpose")
    if run.get("qualified") is False:
        warnings.append("unqualified_environment")
    return Promotion(not blockers, state, tuple(dict.fromkeys(warnings)), blockers)


# ---- rows ----------------------------------------------------------------------------------------------


def row_props(doc: dict, *, version: int, upload_id: str) -> dict:
    """ref.* props shared by the value rows (run level)."""
    return {
        "run.trust": "external", "run.ref_only": "1", "run.record_type": "reference",
        "ref.key": doc["key"], "ref.version": str(version), "ref.origin": doc["origin"], "ref.terms": doc["terms"],
        "ref.label": doc["label"], "ref.class": doc["class"], "upload_id": upload_id,
        "ref.clean": _clean(doc), "ref.warnings": ",".join(doc.get("warnings") or []),
        **({"ref.pinning": str(doc["pinning"])} if doc.get("pinning") else {}),
    }  # fmt: skip


def _clean(doc) -> str:
    return {True: "1", False: "0", None: "unknown"}[doc.get("clean")]


def artifact_props(
    doc: dict,
    *,
    version: int,
    upload_id: str,
    submitter: str,
    approval: dict,
    bundle_sha256: str,
    approved_at: str,
    promote: bool = True,
) -> dict:
    p = {
        "ref.key": doc["key"], "ref.version": str(version), "ref.origin": doc["origin"], "ref.class": doc["class"],
        "ref.label": doc["label"], "ref.visibility": "internal", "ref.digest": content_digest(doc),
        "ref.producer": doc.get("producer", ""), "ref.source_url": doc.get("source_url", ""),
        "ref.as_of": doc.get("as_of", ""), "ref.terms": doc["terms"],
        "ref.model.repo": (doc.get("model") or {}).get("repo", ""),
        "ref.model.revision": (doc.get("model") or {}).get("revision", ""),
        "ref.dataset.id": (doc.get("dataset") or {}).get("id", ""),
        "ref.dataset.subset": (doc.get("dataset") or {}).get("subset", ""),
        "ref.submitted_by": norm_login(submitter), "ref.approved_at": approved_at, "ref.upload_id": upload_id,
        "ref.bundle_sha256": bundle_sha256, "ref.clean": _clean(doc),
        "ref.warnings": ",".join(doc.get("warnings") or []), "ref.pinning": str(doc.get("pinning") or ""),
        "ref.source_run_id": str((doc.get("source_run") or {}).get("run_id", "")), "created_by": "reference-upload", "ref.promote": "1" if promote else "0",
        **{f"ref.tools.{k}": v for k, v in (doc.get("tools") or {}).items()},
        **{f"ref.system.{k}": v for k, v in (doc.get("system") or {}).items()},
        **{f"ref.comparability.{k}": v for k, v in (doc.get("comparability") or {}).items()},
        **approval,
    }  # fmt: skip
    return {k: str(v) for k, v in p.items() if v not in ("", None)}


def tag_row(artifact_id: str, tag: str, key: str, props: dict) -> dict:
    return {"tag": tag, "tag_family": TAG_FAMILY, "artifact_id": artifact_id, "refs": [], "published_refs": [],
            "props": {"ref.key": key, **{k: str(v) for k, v in props.items() if v not in ("", None)}}}  # fmt: skip


def upload_event(*, upload_id, state, actor, artifact_id, run_id, doc, filename="", size_bytes=0, content_sha256="",
                 bundle_url="", reason="", props=None) -> dict:  # fmt: skip
    """A result_uploads row for one lifecycle event of a reference."""
    return {
        "upload_id": upload_id, "state": state, "actor": norm_login(actor), "artifact_id": artifact_id, "run_id": run_id,
        "format": "reference", "result_kind": "performance", "test_type": doc["class"], "arch": arch_of(doc),
        "filename": filename, "size_bytes": size_bytes, "content_sha256": content_sha256, "bundle_url": bundle_url,
        "reason": reason, "props": {k: str(v) for k, v in (props or {}).items() if v not in ("", None)},
    }  # fmt: skip


# ---- writer ----------------------------------------------------------------------------------------------


WOULD = {
    "ingested": "would-ingest",
    "resumed": "would-resume",
    "duplicate": "would-duplicate",
}


class ReferenceError(Exception):
    """The reference cannot be written; `retry` says whether trying again can help."""

    def __init__(self, message, retry=False):
        super().__init__(message)
        self.retry = retry


@dataclass(frozen=True)
class WriteContext:
    submitter: str
    upload_id: str
    approval: dict  # decide_approval(...).props
    bundle_sha256: str
    bundle_url: str = ""
    promote: bool = True
    register_missing: bool = False
    approved_at: str = ""


def write(client, db: str, doc: dict, ctx: WriteContext, dry_run: bool = False) -> dict:
    """Write an approved reference once: artifact, address, value rows, the current tag, then the verdict.

    Everything is validated before the first insert. The verdict is written last and is the completion
    marker: a retry of the same upload resumes a failed write with the same run and version, and one that
    already completed is `duplicate` and changes nothing. The same content under another upload is refused.
    Approvals of one key must be serialised by the caller: the version claim is a read-then-write, so
    two PROCESSES approving the same key at once are not guaranteed to be told apart.
    """
    from .schema import ArtifactTags
    from .writer import ArtifactWriter, BenchmarkWriter, insert_artifact_result

    errors = validate(doc)
    if errors:
        raise ReferenceError("; ".join(errors))
    ident = identity(doc)
    aid = ident.artifact_id
    absent = missing_cases(client, db, doc)
    if absent and not ctx.register_missing and not dry_run:
        raise ReferenceError(
            f"{len(absent)} case(s) are not registered in benchmarks: {absent[:5]}; register them first "
            "(or approve with register_missing to register them now)"
        )
    stored = _stored(client, db, aid)
    approval, promote = ctx.approval, bool(ctx.promote)
    if stored:
        version, first_upload, first_submitter, first_approval, first_props = stored
        first_version = version
        if first_upload != ctx.upload_id:
            raise ReferenceError(
                f"identical content is already version {version} of {doc['key']} (upload {first_upload}), "
                "retracted or not: identical content cannot be approved again, a corrected reference needs "
                "changed content"
            )
        approval = first_approval or ctx.approval
        promote = first_props.get("ref.promote", "1") != "0"
        rid = run_id_for(doc, first_submitter, first_upload)
        done = ArtifactWriter.result_recorded(
            client, db, aid, rid, "performance", doc["class"]
        )
    else:
        version, done = next_version(client, db, doc["key"]), False
        rid = run_id_for(doc, ctx.submitter, ctx.upload_id)
    status = "duplicate" if done else "resumed" if stored else "ingested"
    report = {"status": WOULD[status] if dry_run else status, "artifact_id": aid, "run_id": rid,
              "key": doc["key"], "version": version, "arch": arch_of(doc),
              "registered_now": absent if ctx.register_missing else [],
              "unregistered": [] if ctx.register_missing else absent,
              "warnings": doc.get("warnings") or []}  # fmt: skip
    if dry_run or done:
        return report
    when = ctx.approved_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if stored:
        ident = identity(
            doc, _stored_ref(client, db, aid)
        )  # a retry keeps the first address
    else:
        ident = identity(doc, ctx.bundle_url)
    try:
        for _ in range(
            5
        ):  # claim a number: an earlier holder of ours moves us to the next free one
            holders = _holders_of_version(client, db, doc["key"], version)
            if not holders or holders[0] == aid:
                break
            version = next_version(client, db, doc["key"])
            report["version"] = version
            if stored:
                _renumber_stored(client, db, ident, doc, version, rid)
        else:
            raise ReferenceError(
                f"could not claim a version of {doc['key']}: approvals of one key must be serialised"
            )
        props = row_props(
            _as_stored(doc, first_props) if stored else doc,
            version=version,
            upload_id=ctx.upload_id,
        )
        ArtifactWriter.insert_artifact(  # each step is existence-checked, so a resume repairs a missing address
            client, db, ident, origin=ARTIFACT_ORIGIN[doc["origin"]],
            identity_deps=[f"dataset@{(doc.get('dataset') or {}).get('id', '')}"] if (doc.get("dataset") or {}).get("id") else [],
            props=artifact_props(doc, version=version, upload_id=ctx.upload_id, submitter=ctx.submitter,
                                 approval=approval, bundle_sha256=ctx.bundle_sha256, approved_at=when, promote=promote),
        )  # fmt: skip
        for _ in range(
            5
        ):  # lost a concurrent claim after inserting: the earliest keeps the number, we take the next
            holders = _holders_of_version(client, db, doc["key"], version)
            if holders and holders[0] != aid:
                version = next_version(client, db, doc["key"])
                report["version"] = version
                if stored:
                    _guard_rows(client, db, doc, rid, version)
                _renumber(client, db, ident, doc, version)
                props = row_props(
                    _as_stored(doc, first_props) if stored else doc,
                    version=version,
                    upload_id=ctx.upload_id,
                )
                continue
            break
        else:
            raise ReferenceError(
                f"could not claim a version of {doc['key']}: approvals of one key must be serialised",
                retry=True,
            )
        by_component = {}
        for e in value_entries(doc):
            by_component.setdefault(e.pop("component"), []).append(e)
        written = 0
        for component, entries in sorted(by_component.items()):
            if BenchmarkWriter.already_ingested(
                client, db, rid, component, "reference", FILE
            ):
                continue
            written += BenchmarkWriter.insert(
                client, db, component, rid,
                [{**e, "run_props": {**e["run_props"], **props}} for e in entries],
                report_kind="reference", source_file=FILE,
            )  # fmt: skip
        take_slot = promote
        suppressed = False
        if (
            take_slot
            and stored
            and _newer_current(client, db, doc["key"], first_version)
        ):
            take_slot, suppressed = (
                False,
                True,
            )  # a resumed older write never takes the slot back from a newer version
        if take_slot:
            ArtifactTags.insert(client, [tag_row(aid, TAG_CURRENT, doc["key"], {
                "promoted_by": approval.get("ref.approved_by", ""), "ref.version": version,
                "ref.approval_reason": approval.get("ref.approval_reason", ""), "ref.digest": content_digest(doc),
                "upload_id": ctx.upload_id, "ref.approval_override": approval.get("ref.approval_override", ""),
            })], db=db)  # fmt: skip
        insert_artifact_result(
            client, db, artifact_id=aid, run_id=rid, test_type=doc["class"], state="passed", arch=arch_of(doc),
            result_kind="performance", duration_s=0.0,
            props={"source": "bundle", "bundle_kind": "bench-reference", "bundle_sha256": ctx.bundle_sha256,
                   "bundle_url": ctx.bundle_url, "ref.key": doc["key"], "ref.version": str(version),
                   "ref.digest": content_digest(doc), "upload_id": ctx.upload_id},
        )  # fmt: skip
    except ReferenceError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, AssertionError):
        raise  # a bug, not a write error: never retried
    except BatchParked as err:  # the retry resumes the write; a replay of these batches would double them
        raise ReferenceError(
            f"reference rows parked, not stored ({err}): retry the upload to resume; do not replay them",
            retry=True,
        ) from err
    except Exception as err:  # noqa: BLE001 -- a driver or storage error: the same upload resumes where it stopped
        raise ReferenceError(f"reference write failed: {err!r}", retry=True) from err
    return {
        **report,
        "benchmark_runs": written,
        "current": bool(promote and not suppressed),
    }


def set_tag(
    client, db: str, tag: str, artifact_id: str, key: str, actor: str, reason: str = ""
) -> None:
    """Move a key's slot: ref-current on any version (approve, rollback) or ref-retired (retire).
    Always a new row, never an update, so the newest row decides and history is kept."""
    from .schema import ArtifactTags

    if tag not in (TAG_CURRENT, TAG_RETIRED):
        raise ValueError(f"not a reference tag: {tag!r}")
    ArtifactTags.insert(
        client,
        [
            tag_row(
                artifact_id,
                tag,
                key,
                {"promoted_by": norm_login(actor), "ref.approval_reason": reason},
            )
        ],
        db=db,
    )
