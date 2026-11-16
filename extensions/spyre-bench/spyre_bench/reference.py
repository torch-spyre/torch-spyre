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

"""reference: turn a run into a reference file, validate and seal it, submit it for approval, compare a run to it.

The file format, validation, promotion check and approval rules restate the ingest's so the engine never imports it
(tests/test_reference.py checks both copies agree). Comparison is informational: never a gate, never a baseline.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import statistics
import sys
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import regex as re
import yaml

from spyre_bench import cliutil, engine, store as stores
from spyre_bench.cliutil import UsageError

API_VERSION = "bench/v1alpha1"
KIND = "Reference"
FILE = "reference.json"
BUNDLE_FILE = "bundle.json"
ORIGINS = ("generated", "uploaded", "published")
CLASSES = ("quality", "perf")
MAX_VALUE_ROWS = 10_000
UINT32_MAX = 2**32 - 1
KERNEL_PREFIX, KERNEL_TOKEN = "spyre_kernel_", 16
LABEL = re.compile(r"^[a-z0-9][a-z0-9-]{2,40}$")
LABEL_DENY = re.compile(
    r"(^|-)(node|host|worker)(-|$)|\.", re.I
)  # the generic minimum; sites add more with a deny-list file
DENYLIST_ENV = "SPYRE_BENCH_LABEL_DENYLIST"
KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{2,120}$")
BACKEND = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,60}$")
TOP_KEYS = {
    "apiVersion", "kind", "key", "origin", "class", "label", "terms", "as_of", "producer", "source_url",
    "model", "dataset", "tools", "system", "comparability", "source_run", "clean", "warnings", "pinning",
    "values", "assets",
}  # fmt: skip
RESERVED = (
    "version", "visibility", "trust", "ref_only", "submitted_by", "approved_by", "approved_at",
    "approval_reason", "terms_ack_by", "approvals", "upload_id",
)  # fmt: skip
COMPARABILITY = {
    "harness_match": ("yes", "partial", "no", "unknown"),
    "dataset_match": ("same", "subset", "different", "unknown"),
    "chat_template": ("applied", "not_applied", "unknown"),
}
COMPARABILITY_ALWAYS = ("harness", "harness_match", "precision", "dataset_subset", "dataset_match")
COMPARABILITY_CLASS = {
    "quality": ("fewshot", "chat_template"),
    "perf": ("shape.batch", "shape.input_len", "shape.output_len"),
}
COMPARABILITY_OPTIONAL = ("shape.max_model_len", "caveats")
CAVEATS_MAX = 500
NOT_KEYS = ("harness", "harness_match", "dataset_match", "caveats")  # judged apart from the keys a run must equal
BASELINE_PURPOSES = ("nightly", "weekly", "merge")  # the view benchmark_baseline_purposes
DEFAULT_TOLERANCE_PCT = 5.0
META_KEYS = ("model", "dataset", "tools", "system", "comparability", "producer", "source_url", "terms", "as_of")
ENGINE_WARNING_TEXT = {"invalidation_unchecked": "invalidation of the run could not be checked (no readable store)"}
REQUEST_SCHEMA = "bench.reference-request/1"
PRECISION_KEYS = ("dtype", "precision")
INFORMATIONAL = "informational: a reference is never a gate and never a baseline"


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (ValueError, RecursionError) as e:
        raise UsageError(f"{path} is not usable JSON: {type(e).__name__}: {str(e)[:120]}") from None


def canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(x):
    try:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    except OverflowError:
        return False


def _list(doc, key):
    """doc[key] as a list; a missing, null or non-list value is empty (validate reports it)."""
    value = doc.get(key) if isinstance(doc, dict) else None
    return value if isinstance(value, list) else []


def case_id(case):
    """The benchmark_id a case block names: the ingest derivation over component, name, tags, disc and disc_keys."""
    props = {k: str(v) for k, v in (case.get("disc") or {}).items()}
    keys = case.get("disc_keys")
    keys = sorted(props) if keys is None else list(keys)
    return engine.benchmark_id(case["component"], case["name"].strip(), list(case.get("tags") or []), props, keys)


def archs_of(doc):
    return sorted({str((v.get("case") or {}).get("disc", {}).get("arch", "")) for v in _list(doc, "values")} - {""})


def arch_of(doc):
    """noarch for a data file; multi only when the values name several architectures."""
    return "multi" if len(archs_of(doc)) > 1 else "noarch"


def content_digest(doc):
    """sha256 over key, class, origin, system, dataset, model, comparability, values and assets (not terms, label, warnings)."""
    meaning = {
        "key": doc.get("key"), "class": doc.get("class"), "origin": doc.get("origin"),
        "system": doc.get("system") or {}, "dataset": doc.get("dataset") or {}, "model": doc.get("model") or {},
        "comparability": doc.get("comparability") or {},
        "values": sorted(
            ({"id": v["case"].get("benchmark_id") or case_id(v["case"]), "backend": v.get("backend", ""),
              "m": {k: [float(x) for x in xs] for k, xs in (v.get("measurements") or {}).items()},
              "n": v.get("iterations", 0)} for v in _list(doc, "values")),
            key=lambda r: (r["id"], r["backend"]),
        ),
        "assets": sorted(
            (a.get("sha256", ""), a.get("records", 0), (a.get("case") or {}).get("benchmark_id", ""))
            for a in _list(doc, "assets")
        ),
    }  # fmt: skip
    return hashlib.sha256(canonical(meaning).encode()).hexdigest()


def case_ids(doc):
    """(component, benchmark_id) of every case the reference has values or assets for."""
    return sorted(
        {(v["case"]["component"], v["case"]["benchmark_id"]) for v in [*_list(doc, "values"), *_list(doc, "assets")]}
    )


# ---- validation ----------------------------------------------------------------------------------------------


def load_denylist(path):
    """Compiled patterns of a deny-list file: one case-insensitive regex per line, `#` comments and blanks ignored."""
    try:
        lines = Path(path).read_text().splitlines()
    except (OSError, UnicodeDecodeError) as e:
        raise UsageError(f"cannot read the label deny-list {path}: {e}") from None
    out = []
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                out.append(re.compile(line, re.I))
            except re.error as e:
                raise UsageError(f"{path}:{n}: invalid pattern {line!r}: {e}") from None
    return out


def denylist_of(flag=None):
    """The configured deny-list: --label-denylist, else SPYRE_BENCH_LABEL_DENYLIST, else none."""
    path = flag or cliutil.env(DENYLIST_ENV)
    return load_denylist(path) if path else []


def _validate(doc, root, label_deny):
    """Every problem in a reference.json (empty = valid); `root` also checks asset bytes; `label_deny` adds patterns."""
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
    if not LABEL.fullmatch(label) or LABEL_DENY.search(label) or any(d.search(label) for d in label_deny):
        errors.append("label must match [a-z0-9][a-z0-9-]{2,40} and carry no node, host or deny-listed name")
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
            case = a.get("case")
            for what, mark in (
                ("path", a.get("path")),
                ("sha256", a.get("sha256")),
                ("case", case.get("benchmark_id") if isinstance(case, dict) else None),
            ):
                if mark is not None and (what, str(mark)) in taken:
                    errors.append(f"assets[{i}] repeats a {what}")
                taken.add((what, str(mark)))
    errors += _overlap(doc)
    if root is not None and (Path(root) / "attachments").is_dir():
        listed = {a.get("path") for a in assets if isinstance(a, dict)}
        loose = sorted(
            p.relative_to(root).as_posix()
            for p in (Path(root) / "attachments").rglob("*")
            if p.is_file() and p.relative_to(root).as_posix() not in listed
        )
        if loose:
            errors.append(f"attachments not listed in assets: {loose}")
    return errors


def validate(doc, root=None, label_deny=()):
    """Every problem in a reference.json (empty = valid); total: whatever the JSON holds, it never raises."""
    try:
        return _validate(doc, None if root is None else Path(root), label_deny)
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError, OSError, RecursionError) as err:
        return [f"{FILE} is malformed: {err!r}"]


def _shape_errors(doc):
    """Wrong JSON types, found before any rule that reads inside a block."""
    errors = []
    for k in ("model", "dataset", "system", "tools", "source_run", "comparability"):
        if doc.get(k) is not None and not isinstance(doc[k], dict):
            errors.append(f"{k} must be an object")
    for k in ("key", "origin", "class", "label", "terms", "producer", "source_url", "as_of", "pinning"):
        if doc.get(k) is not None and not isinstance(doc[k], str):
            errors.append(f"{k} must be a string")
    for k in ("values", "assets", "warnings"):
        if doc.get(k) is not None and not isinstance(doc[k], list):
            errors.append(f"{k} must be a list")
    if isinstance(doc.get("tools"), dict) and not all(isinstance(v, str) for v in doc["tools"].values()):
        errors.append("tools maps names to version strings")
    if isinstance(doc.get("system"), dict) and not all(
        isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in doc["system"].values()
    ):
        errors.append("system holds strings and numbers only")
    if isinstance(doc.get("system"), dict) and not isinstance(doc["system"].get("backend", ""), str):
        errors.append("system.backend must be a string")
    for k in ("values", "assets"):
        blocks = doc.get(k)
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
                errors.append(f"{k}[{i}].case needs string component/name, a tags list and a disc object")
    return errors


def _overlap(doc):
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
        if isinstance(a, dict) and isinstance(a.get("case"), dict) and (a["case"].get("benchmark_id"), backend) in taken
    ]


def _provenance(doc, origin, klass):
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
    if not isinstance(doc.get("system") or {}, dict) or not (doc.get("system") or {}).get("backend"):
        errors.append("system.backend is required")
    elif not BACKEND.fullmatch(str(doc["system"]["backend"])):
        errors.append("system.backend must be a closed backend value or ref:<label>")
    if doc.get("clean") not in (True, False, None):
        errors.append("clean must be true, false or null")
    if not isinstance(doc.get("warnings", []), list) or any(not isinstance(w, str) for w in doc.get("warnings", [])):
        errors.append("warnings must be a list of codes")
    return errors


def _as_of(value, origin):
    try:
        when = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return [f"an {origin} reference needs as_of (ISO date)"]
    when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return ["as_of is in the future"] if when > datetime.now(timezone.utc) else []


def _comparability(doc, origin, klass):
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
            errors.append(f"comparability.{k} cannot be unknown for a generated reference")
    for k in block:
        if k not in needed and k not in COMPARABILITY_OPTIONAL:
            errors.append(f"unknown comparability key {k!r}")
    caveats = block.get("caveats")
    if origin in ("uploaded", "published") and not str(caveats or "").strip():
        errors.append(f"an {origin} reference needs comparability.caveats (may say none)")
    if origin == "published" and str(caveats or "").strip().lower() == "none":
        errors.append("a published reference needs non-empty caveats")
    if caveats and len(str(caveats)) > CAVEATS_MAX:
        errors.append(f"comparability.caveats is over {CAVEATS_MAX} characters")
    return errors


def kernel_stem(kernel_name):
    """A compiled kernel's name without its per-compile token; '' for any other name."""
    name = "" if kernel_name is None else str(kernel_name)
    head, sep, n = name.rpartition("#")
    if not (sep and n and all(c in "0123456789" for c in n)):
        head, sep, n = name, "", ""
    token = head[-KERNEL_TOKEN - 1 :]
    if not (
        name.startswith(KERNEL_PREFIX)
        and len(token) == KERNEL_TOKEN + 1
        and token[0] == "_"
        and all(c in "abcdefghijklmnopqrstuvwxyz0123456789" for c in token[1:])
    ):
        return ""
    return head[: -len(token)] + sep + n


def _id_errors(where, case):
    """The case's id recomputed; one clean message for each way a case block can be wrong."""
    keys = case.get("disc_keys")
    if keys is not None and not (isinstance(keys, list) and all(isinstance(k, str) for k in keys)):
        return [f"{where}.case.disc_keys must be a list of strings"]
    try:
        if "kernel_name" in (keys if keys is not None else (case.get("disc") or {})) and kernel_stem(
            (case.get("disc") or {}).get("kernel_name")
        ):
            return [f"{where}.case.disc.kernel_name is a compiled-kernel name, which the writer re-keys"]
        ok = case_id(case) == case["benchmark_id"]
    except Exception as err:  # noqa: BLE001 - a malformed case block is a validation error
        return [f"{where}.case: {err}"]
    return [] if ok else [f"{where}.case.benchmark_id does not match its component/name/tags/disc"]


def _value(i, v, seen):
    where = f"values[{i}]"
    if not isinstance(v, dict):
        return [f"{where} is not an object"]
    case = v.get("case")
    errors = []
    if not isinstance(case, dict) or not case.get("component") or not str(case.get("name", "")).strip():
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
                errors.append(f"{where}.measurements.{k} must be a non-empty list of finite numbers")
    if (
        not isinstance(v.get("iterations"), int)
        or isinstance(v["iterations"], bool)
        or not 1 <= v["iterations"] <= UINT32_MAX
    ):
        errors.append(f"{where}.iterations (n) must be 1..{UINT32_MAX}")
    return errors


def plain_asset_path(text):
    return bool(
        re.fullmatch(r"attachments/[^\s/][^\s]*", text)
        and ".." not in text.split("/")
        and "\\" not in text
        and "\x00" not in text
    )


def asset_file(root, rel):
    """The regular file an asset path names inside `root`, or None when it is a link or leaves the bundle."""
    path = Path(root) / rel
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        return None
    return path


def _asset(i, a, root):
    where = f"assets[{i}]"
    errors = []
    path_text = str(a.get("path", ""))
    if not isinstance(a, dict) or not plain_asset_path(path_text):
        return [f"{where}.path must be a plain path under attachments/"]
    if not re.fullmatch(r"[0-9a-f]{64}", str(a.get("sha256", ""))):
        errors.append(f"{where}.sha256 must be 64 hex")
    if not isinstance(a.get("records"), int) or isinstance(a["records"], bool) or not 1 <= a["records"] <= UINT32_MAX:
        errors.append(f"{where}.records must be 1..{UINT32_MAX}")
    case = a.get("case")
    if not isinstance(case, dict) or not case.get("benchmark_id"):
        errors.append(f"{where}.case with benchmark_id is required (the parity case the bytes are for)")
    else:
        errors += _id_errors(where, case)
    if root is not None and not errors:
        path = asset_file(root, a["path"])
        if path is None:
            errors.append(f"{where}: {a['path']} is a link or leaves the bundle")
        elif not path.is_file():
            errors.append(f"{where}: {a['path']} is not in the bundle")
        elif sha256_file(path) != a["sha256"]:
            errors.append(f"{where}: sha256 does not match {a['path']}")
        elif a["path"].endswith((".jsonl", ".csv")):
            lines = sum(1 for ln in path.read_bytes().splitlines() if ln.strip())
            if lines != a["records"]:
                errors.append(f"{where}: {a['path']} has {lines} record(s), not {a['records']}")
    return errors


# ---- promoting a run (D21.3) ---------------------------------------------------------------------------------

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
    clean: str  # "1" | "0" | "unknown": recorded as clean true / false / null
    warnings: tuple = ()
    blockers: tuple = ()

    @property
    def needs_ack(self):
        return bool(self.warnings)


def promotion_check(run):
    """Judge facts of a run offered as a reference: blockers refuse it, warnings are recorded with it."""
    blockers = tuple(
        b
        for b in BLOCKERS
        if (b == "no_measurements" and not run.get("measured", True)) or (b != "no_measurements" and run.get(b))
    )
    clean = run.get("clean") or {}
    exclusive = clean.get("exclusive") or "unknown"
    warnings = []
    if exclusive == "enforced" or (
        exclusive == "observed" and clean.get("load_clean") is True and clean.get("devices_exclusive") == "enforced"
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
    if str(run.get("purpose") or "").strip().lower() not in ("", *BASELINE_PURPOSES):
        warnings.append("not_baseline_purpose")
    if run.get("qualified") is False:
        warnings.append("unqualified_environment")
    return Promotion(not blockers, state, tuple(dict.fromkeys(warnings)), blockers)


# ---- reading a run -------------------------------------------------------------------------------------------


def find_run(ref, runs_dir=None):
    """(directory, run.json or {}) of a results directory, or of the run id found under `runs_dir`."""
    p = Path(ref)
    if p.is_file() and p.name == "result.json":
        p = p.parent
    if p.is_dir():
        return p, _manifest(p)
    for base in [Path(runs_dir)] if runs_dir else [Path.cwd()]:
        for cand in [base, *sorted(base.glob("*"))]:
            if cand.is_dir() and _manifest(cand).get("run_id") == ref:
                return cand, _manifest(cand)
    raise UsageError(f"no results directory or run id {ref!r} (pass the directory, or --runs-dir holding it)")


def _manifest(d):
    try:
        doc = json.loads((Path(d) / "run.json").read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def run_results(d, manifest):
    """The Result documents a run wrote, in manifest order (else every result.json under the directory)."""
    base = Path(d).resolve()
    named = [c["result"] for c in manifest.get("cases", []) if isinstance(c, dict) and c.get("result")]
    paths = [base / n for n in named] if named else sorted(base.glob("**/result.json"))
    out = []
    for path in paths:
        if not path.resolve().is_relative_to(base):
            raise UsageError(f"{path} is outside the results directory")
        try:
            doc = load(path)
        except (OSError, ValueError) as e:
            raise UsageError(f"{path} is not a readable result: {e}") from None
        if isinstance(doc, dict) and doc.get("kind") == "Result":
            out.append(doc)
    return out


def ok_repeats(result):
    return [r for r in result["status"].get("repeats", []) if r.get("status", "ok") == "ok" and r.get("samples")]


def samples_of(result):
    """metric -> every finite sample of the ok repeats."""
    out = {}
    for r in ok_repeats(result):
        for m, xs in r["samples"].items():
            out.setdefault(m, []).extend(float(x) for x in xs if _finite(x))
    return {m: xs for m, xs in out.items() if xs}


def case_block(result):
    c = result["status"]["case"]
    return {
        "component": c["component"], "name": c["name"], "tags": list(c.get("tags") or []),
        "disc": {k: str(v) for k, v in (c.get("disc") or {}).items()},
        "disc_keys": list(c["disc_keys"]) if c.get("disc_keys") is not None else sorted(c.get("disc") or {}),
        **({"id_profile": c["id_profile"]} if c.get("id_profile") else {}),
    }  # fmt: skip


def run_facts(results, manifest, invalidated=False):
    """The promotion_check input for the weakest of a run's results."""
    reps = [r for res in results for r in ok_repeats(res)]
    cleans = [r.get("clean") or {} for r in reps]
    weak = next((c for c in cleans if not engine.run_clean(c)), cleans[0] if cleans else {})
    props = [r.get("props") or {} for r in reps]
    requested = any(p.get("run.exec.cpu_pinning", "none") != "none" for p in props)
    verified = all(p.get("run.exec.pinning_verified") == "true" for p in props)
    states = {(res["status"].get("validity") or {}).get("state") for res in results} - {None}
    return {
        "invalidated": invalidated,
        "measured": any(samples_of(res) for res in results),
        "validity": min(states, key=lambda v: engine.VALIDITY_ORDER.index(v) if v in engine.VALIDITY_ORDER else -1)
        if states
        else None,
        "clean": weak,
        "repeats": min((len(ok_repeats(res)) for res in results), default=0),
        "pinning_verified": False if requested and not verified else None,
        "purpose": manifest.get("purpose"),
        "qualified": False if "unqualified_env" in states else None,
    }


def pinning_of(results):
    """The pinning policies the run's repeats requested, joined (none when it asked for none)."""
    props = [r.get("props") or {} for res in results for r in ok_repeats(res)]
    return ",".join(sorted({p["run.exec.cpu_pinning"] for p in props if p.get("run.exec.cpu_pinning")})) or "none"


# ---- create --------------------------------------------------------------------------------------------------


def build_values(results, backend=""):
    """(value rows, errors) for the results with measurements, one per (case, backend)."""
    values, errors = [], []
    for res in results:
        samples = samples_of(res)
        if not samples:
            continue
        case = case_block(res)
        case["benchmark_id"] = case_id(case)
        be = (res["status"]["case"].get("props") or {}).get("system.backend") or backend
        if not be:
            errors.append(f"{case['name']}: no backend in the result; pass --backend")
            continue
        values.append(
            {
                "case": case,
                "backend": str(be),
                "measurements": samples,
                "iterations": max(len(x) for x in samples.values()),
            }
        )
    return values, errors


def build_doc(results, manifest, args_meta, promotion, pinning, values):
    backends = sorted({v["backend"] for v in values})
    doc = {
        "apiVersion": API_VERSION, "kind": KIND, "key": args_meta["key"], "origin": args_meta["origin"],
        "class": args_meta["class"], "label": args_meta["label"],
        "terms": args_meta.get("terms") or ("internal-generated" if args_meta["origin"] == "generated" else ""),
        "as_of": args_meta.get("as_of") or manifest.get("ended_at") or cliutil.utc_stamp(),
        "system": {"backend": backends[0] if len(backends) == 1 else "multi"},
        "comparability": {}, "source_run": {"run_id": manifest.get("run_id", "")},
        "clean": {"1": True, "0": False}.get(promotion.clean), "warnings": list(promotion.warnings),
        "pinning": pinning, "values": values, "assets": [],
    }  # fmt: skip
    for k, v in (args_meta.get("meta") or {}).items():
        if k not in META_KEYS:
            raise UsageError(f"--meta key {k!r} is not allowed (descriptive keys only: {', '.join(META_KEYS)})")
        doc[k] = {**doc[k], **v} if isinstance(v, dict) and isinstance(doc.get(k), dict) else v
    for k in ("producer", "source_url"):
        if args_meta.get(k):
            doc[k] = args_meta[k]
    return doc


def write_dir(doc, out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise UsageError(f"{out} exists and is not empty")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        (tmp / "attachments").mkdir()
        (tmp / FILE).write_text(json.dumps(doc, indent=2) + "\n")
        if out.exists():
            out.rmdir()
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


class _StrictLoader(yaml.SafeLoader):
    """YAML 1.2 booleans: only true/false, so an unquoted yes/no/on/off stays the string it looks like."""


_StrictLoader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictLoader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$"), list("tf"))


def read_meta(path):
    if not path:
        return {}
    try:
        doc = yaml.load(Path(path).read_text(), Loader=_StrictLoader)  # noqa: S506 - a SafeLoader subclass
    except RecursionError:
        raise UsageError(f"--meta {path} is nested too deeply") from None
    if not isinstance(doc, dict):
        raise UsageError(f"--meta {path} is not a mapping")
    return doc


def pairs(items, flag):
    out = {}
    for item in items or []:
        k, sep, v = item.partition("=")
        if not sep or not k:
            raise UsageError(f"{flag} wants KEY=VALUE, got {item!r}")
        out[k] = int(v) if v.isdigit() else v
    return out


def registered_report(st, doc):
    """(missing ids, note): the reference's cases the store does not register; never raises."""
    ids = sorted({bid for _, bid in case_ids(doc)})
    try:
        have = st.registered(ids)
    except stores.StoreError as e:
        return None, f"could not check case registration: {e}"
    if have is None:
        return None, "case registration not checked (no store)"
    return [i for i in ids if i not in have], ""


def print_registration(st, doc, err):
    missing, note = registered_report(st, doc)
    if note:
        print(f"note: {note}", file=err)
    elif missing:
        print(
            f"note: {len(missing)} case(s) are not registered; approval offers to register them now: {missing[:5]}",
            file=err,
        )


def slug_label(text):
    """A label slug from free text (the request's label is prose)."""
    out = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:41].strip("-")
    return out if len(out) >= 3 else f"ref-{out}".strip("-")


def request_values(req):
    """Reference value rows from a request's values: one row per (case, backend), a metric each."""
    backend = str((req.get("scope") or {}).get("backend") or "")
    groups = {}
    for i, v in enumerate(_list(req, "values")):
        case = v.get("case") if isinstance(v, dict) else None
        if not isinstance(case, dict):
            raise UsageError(
                f"request values[{i}] ({v.get('name') if isinstance(v, dict) else v}) has no case block "
                "(component, name, tags, disc, disc_keys): the accept side must carry it to rebuild the id"
            )
        block = {k: case[k] for k in ("component", "name", "tags", "disc", "disc_keys", "id_profile") if k in case}
        block["disc"] = {k: str(x) for k, x in (block.get("disc") or {}).items()}
        try:
            bid = case_id(block)
        except (KeyError, AttributeError, TypeError) as e:
            raise UsageError(f"request values[{i}].case is malformed: {e!r}") from None
        if v.get("benchmark_id") and v["benchmark_id"] != bid:
            raise UsageError(
                f"request values[{i}]: its case block derives {bid}, not the stored id {v['benchmark_id']}"
            )
        block["benchmark_id"] = bid
        samples = [float(x) for x in v.get("samples") or []]
        row = groups.setdefault((bid, backend), {"case": block, "backend": backend, "measurements": {}})
        row["measurements"][str(v.get("metric"))] = samples
    values = list(groups.values())
    for row in values:
        row["iterations"] = max((len(x) for x in row["measurements"].values()), default=0)
    return values


def request_comparability(req, values):
    """The request's placeholder block, settled from the values' own facts: precision from the case dtype, shape n/a."""
    block = dict(req.get("comparability") or {})
    precisions = {c["disc"][k] for v in values for c in [v["case"]] for k in PRECISION_KEYS if k in c["disc"]}
    if len(precisions) == 1:
        block["precision"] = precisions.pop()
    elif precisions:
        block["precision"] = "mixed"
    for k in COMPARABILITY_CLASS["perf"]:
        block.setdefault(k, "n/a")
    return block


def run_promotion(results, manifest, st):
    """promotion_check of a run with the store's invalidation lookup; an unreadable store adds invalidation_unchecked."""
    ids = manifest.get("stored_run_ids") or ([manifest["run_id"]] if manifest.get("run_id") else [])
    invalid, unchecked = False, ""
    try:
        invalid = bool(st.invalidated_runs(ids)) if ids else False
        unchecked = "" if ids else "the run names no stored run id"
    except stores.StoreError as e:
        unchecked = str(e)
    promotion = promotion_check(run_facts(results, manifest, invalid))
    if unchecked:
        promotion = Promotion(
            promotion.ok, promotion.clean, (*promotion.warnings, "invalidation_unchecked"), promotion.blockers
        )
        print(f"warning: invalidation_unchecked: {unchecked}", file=sys.stderr)
    return promotion


def cmd_create_from_request(a, st):
    req = load(a.from_request)
    if not isinstance(req, dict) or req.get("request_schema") != REQUEST_SCHEMA:
        raise UsageError(f"{a.from_request} is not a {REQUEST_SCHEMA} file")
    values = request_values(req)
    run_id = str((req.get("anchor") or {}).get("from_run_id", ""))
    facts, pinning, results = None, "none", []
    if a.run:
        d, manifest = find_run(a.run, a.runs_dir)
        results = run_results(d, manifest)
        facts = run_promotion(results, manifest, st)
        pinning = pinning_of(results)
    warnings = [*_list(req, "warnings"), *(facts.warnings if facts else ["clean_unknown"])]
    clean = {"1": True, "0": False}.get(facts.clean) if facts else None
    if facts and not facts.ok:
        print(f"refused: {', '.join(facts.blockers)}", file=sys.stderr)
        return cliutil.EXIT_FAILED
    doc = {
        "apiVersion": API_VERSION, "kind": KIND, "key": a.key or req["key"], "origin": req.get("origin", "generated"),
        "class": a.cls or req.get("class", "perf"), "label": a.label or slug_label(req.get("label", "")),
        "terms": a.terms or "internal-generated", "as_of": (req.get("anchor") or {}).get("as_of") or cliutil.utc_stamp(),
        "tools": {"spyre-bench": "accept"}, "model": {"revision": f"run-{run_id[:8]}"},
        "system": {"backend": values[0]["backend"] if len({v["backend"] for v in values}) == 1 else "multi"},
        "comparability": request_comparability(req, values),
        "source_run": {"run_id": run_id, "role": req.get("role", ""), "as_of": (req.get("anchor") or {}).get("as_of", "")},
        "clean": clean, "warnings": list(dict.fromkeys(warnings)), "pinning": pinning, "values": values, "assets": [],
    }  # fmt: skip
    for k, v in read_meta(a.meta).items():
        if k not in META_KEYS:
            raise UsageError(f"--meta key {k!r} is not allowed (descriptive keys only: {', '.join(META_KEYS)})")
        doc[k] = {**doc[k], **v} if isinstance(v, dict) and isinstance(doc.get(k), dict) else v
    if a.comparability:
        doc["comparability"].update(pairs(a.comparability, "--comparability"))
    if a.tool:
        doc["tools"] = {**doc["tools"], **pairs(a.tool, "--tool")}
    problems = validate(doc, label_deny=denylist_of(a.label_denylist))
    if problems:
        raise UsageError("the reference is incomplete:\n  " + "\n  ".join(problems) + "\n(add --meta FILE or flags)")
    out = write_dir(doc, a.out or f"reference-{doc['key']}")
    for w in doc["warnings"]:
        if w != "invalidation_unchecked":
            print(
                f"warning: {w}: {WARNING_TEXT.get(w) or ENGINE_WARNING_TEXT.get(w, '')}".rstrip(": "), file=sys.stderr
            )
    print_registration(st, doc, sys.stderr)
    print(f"wrote {out} ({len(values)} value row(s) from {a.from_request})")
    return cliutil.EXIT_OK


def cmd_create(a, st):
    if bool(a.from_run) == bool(a.from_request):
        raise UsageError("pass exactly one of --from-run and --from-request")
    if a.from_request:
        return cmd_create_from_request(a, st)
    if not (a.key and a.cls and a.label):
        raise UsageError("--key, --class and --label are required with --from-run")
    d, manifest = find_run(a.from_run, a.runs_dir)
    results = run_results(d, manifest)
    if not results:
        raise UsageError(f"no results in {d}")
    promotion = run_promotion(results, manifest, st)
    if not promotion.ok:
        print(f"refused: {', '.join(promotion.blockers)}", file=sys.stderr)
        return cliutil.EXIT_FAILED
    values, errors = build_values(results, a.backend or "")
    if errors:
        raise UsageError("; ".join(errors))
    meta = {
        "key": a.key, "origin": a.origin, "class": a.cls, "label": a.label, "terms": a.terms, "as_of": a.as_of,
        "producer": a.producer, "source_url": a.source_url, "meta": read_meta(a.meta),
    }  # fmt: skip
    doc = build_doc(results, manifest, meta, promotion, pinning_of(results), values)
    if a.comparability:
        doc["comparability"].update(pairs(a.comparability, "--comparability"))
    if a.tool:
        doc["tools"] = {**doc.get("tools", {}), **pairs(a.tool, "--tool")}
    problems = validate(doc, label_deny=denylist_of(a.label_denylist))
    if problems:
        raise UsageError("the reference is incomplete:\n  " + "\n  ".join(problems) + "\n(add --meta FILE or flags)")
    out = write_dir(doc, a.out or f"reference-{a.key}")
    for w in promotion.warnings:
        if w != "invalidation_unchecked":
            print(f"warning: {w}: {WARNING_TEXT[w]}", file=sys.stderr)
    print_registration(st, doc, sys.stderr)
    print(f"wrote {out} ({len(values)} value row(s), clean={promotion.clean}, {len(promotion.warnings)} warning(s))")
    return cliutil.EXIT_OK


# ---- validate and seal ---------------------------------------------------------------------------------------


def reference_root(path):
    """(root, reference.json) of a reference directory or an unpacked bundle (results/reference.json)."""
    p = Path(path)
    for sub in (FILE, f"results/{FILE}"):
        if (p / sub).is_file():
            return p, p / sub
    raise UsageError(f"no {FILE} (or results/{FILE}) in {p}")


def cmd_validate(a, st):
    root, path = reference_root(a.dir)
    problems = validate(load(path), root, denylist_of(a.label_denylist))
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    print("valid" if not problems else f"{len(problems)} problem(s)")
    return cliutil.EXIT_OK if not problems else cliutil.EXIT_INVALID


def write_tgz(out, root, names):
    """A .tgz of `names` (relative paths under root) that is byte-identical for the same content."""
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for name in sorted(names):
                info = tar.gettarinfo(str(Path(root) / name), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                with open(Path(root) / name, "rb") as fh:
                    tar.addfile(info, fh)
    return Path(out)


def listed_assets(doc):
    return [a["path"] for a in _list(doc, "assets") if isinstance(a, dict)]


def seal(src, out, label_deny=()):
    """Write the bench-reference bundle of a reference directory to `out` (a directory, or .tgz/.tar.gz).

    Only the files assets[] lists are copied; links and paths outside attachments/ are refused by validation.
    """
    src_root = Path(src)
    if (src_root / BUNDLE_FILE).is_file():
        raise UsageError(f"{src} is already a sealed bundle (it has {BUNDLE_FILE}); submit it as is")
    root, path = reference_root(src)
    if path != root / FILE:
        raise UsageError(f"{src} is an unpacked bundle without {BUNDLE_FILE}; seal the reference directory")
    doc = load(path)
    problems = validate(doc, root, label_deny)
    if problems:
        raise UsageError("not valid, not sealed:\n  " + "\n  ".join(problems))
    out = Path(out)
    tgz = out.name.endswith((".tgz", ".tar.gz"))
    if out.exists() and (tgz or any(out.iterdir())):
        raise UsageError(f"{out} already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bundle-", dir=out.parent) as tmp:
        dest = Path(tmp) / "bundle"
        (dest / "results").mkdir(parents=True)
        (dest / "results" / FILE).write_bytes(path.read_bytes())
        for rel in listed_assets(doc):
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / rel).read_bytes())
        files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
        meta = {
            "schema_version": 1,
            "kind": "bench-reference",
            "files": [{"path": p, "sha256": sha256_file(dest / p)} for p in files],
        }
        (dest / BUNDLE_FILE).write_text(json.dumps(meta, indent=2) + "\n")
        if tgz:
            part = write_tgz(Path(tmp) / "out.tgz", dest, [BUNDLE_FILE, *files])
            os.replace(part, out)
        else:
            shutil.copytree(dest, out, dirs_exist_ok=True)
    return out


def unpack(path, tmp):
    """The root directory of a sealed bundle .tgz unpacked under `tmp` (bundle.json at the top or one directory down)."""
    try:
        with tarfile.open(path) as tar:
            for m in tar.getmembers():
                name = Path(m.name)
                if name.is_absolute() or ".." in name.parts or not (m.isfile() or m.isdir()):
                    raise UsageError(f"{path}: unsafe member {m.name!r}")
            try:
                tar.extractall(tmp, filter="data")
            except TypeError:  # filter= is absent before Python 3.11.4; members were checked above
                tar.extractall(tmp)
    except tarfile.TarError as e:
        raise UsageError(f"{path} is not a tar archive: {e}") from None
    top = Path(tmp)
    entries = list(top.iterdir())
    if not (top / BUNDLE_FILE).exists() and len(entries) == 1 and entries[0].is_dir():
        top = entries[0]
    return top


BUNDLE_KEYS = (
    "schema_version", "kind", "artifact_id", "image", "artifact", "create_artifact", "component", "arch", "test_type",
    "run_key", "jenkins_run_key", "run_url", "workflow", "started_at", "ended_at", "tag_family", "tags", "runner", "env",
    "notes", "uploader", "sealed_at", "files", "perf",
)  # fmt: skip
BUNDLE_SCHEMA_VERSION = 1


def check_bundle(root, doc=None):
    """Problems with an unpacked bundle, by the ingest's offline check: bundle.json, its files[] and the layout."""
    root = Path(root)
    try:
        meta = json.loads((root / BUNDLE_FILE).read_text())
    except (OSError, ValueError, RecursionError) as e:
        return [f"{BUNDLE_FILE} is not readable: {e}"]
    if not isinstance(meta, dict):
        return [f"{BUNDLE_FILE} is not a JSON object"]
    version = meta.get("schema_version", BUNDLE_SCHEMA_VERSION)
    if isinstance(version, int) and not isinstance(version, bool) and version > BUNDLE_SCHEMA_VERSION:
        return [f"schema_version {version} is newer than this engine ({BUNDLE_SCHEMA_VERSION})"]
    errors = []
    if version != BUNDLE_SCHEMA_VERSION or isinstance(version, bool):
        errors.append(f"schema_version must be {BUNDLE_SCHEMA_VERSION}")
    errors += [f"{BUNDLE_FILE}: unknown key {k!r}" for k in sorted(set(meta) - set(BUNDLE_KEYS))]
    if meta.get("kind") != "bench-reference":
        errors.append(f"{BUNDLE_FILE} kind must be bench-reference")
    if any(meta.get(k) for k in ("artifact_id", "image", "artifact", "create_artifact")):
        errors.append("a bench-reference bundle names no artifact: it creates its own")
    if meta.get("component", "reference") != "reference":
        errors.append("a bench-reference bundle is component reference")
    arch = arch_of(doc) if isinstance(doc, dict) else None
    if meta.get("arch", arch or "noarch") not in ((arch,) if arch else ("noarch", "multi")):
        errors.append(f"arch must be {arch or 'noarch or multi'} for these values")
    if isinstance(doc, dict) and meta.get("test_type", doc.get("class")) not in ("", doc.get("class")):
        errors.append(f"test_type {meta.get('test_type')!r} differs from the reference's class")
    present = {
        p.relative_to(root).as_posix(): p
        for p in root.rglob("*")
        if (p.is_file() or p.is_symlink()) and not (p.name.startswith("._") or p.name == ".DS_Store")
    }
    present.pop(BUNDLE_FILE, None)
    errors += [f"{p} is a link" for p, f in present.items() if f.is_symlink()]
    stray = [p for p in present if not p.startswith(("results/", "attachments/"))]
    nested = [p for p in present if p.startswith("results/") and "/" in p.removeprefix("results/")]
    if stray or nested:
        errors.append(f"files belong in results/ (flat) or attachments/: {stray or nested}")
    results = sorted(p for p in present if p.startswith("results/"))
    if results != [f"results/{FILE}"]:
        errors.append(f"a bench-reference bundle holds results/{FILE} only: {results}")
    files = meta.get("files")
    if files is not None:
        if not isinstance(files, list) or not all(
            isinstance(f, dict) and isinstance(f.get("path"), str) for f in files
        ):
            errors.append("files must be a list of {path, sha256}")
        else:
            paths = [f["path"] for f in files]
            if len(set(paths)) != len(paths) or any(".." in p.split("/") for p in paths):
                errors.append("files lists a path twice, or a '..' path")
            listed = {f["path"]: f.get("sha256") for f in files}
            errors += [f"{p} is not listed in {BUNDLE_FILE}" for p in sorted(set(present) - set(listed))]
            errors += [f"{p} is listed but missing" for p in sorted(set(listed) - set(present))]
            errors += [
                f"{p}: sha256 mismatch"
                for p in sorted(set(listed) & set(present))
                if not present[p].is_symlink() and sha256_file(present[p]) != listed[p]
            ]
    return errors


def bundle_doc(path):
    """reference.json of a sealed bundle (directory or tgz), a reference directory, or a bare reference.json."""
    p = Path(path)
    if p.is_file() and p.name.endswith((".tgz", ".tar.gz")):
        with tempfile.TemporaryDirectory(prefix="unpack-") as tmp:
            top = unpack(p, tmp)
            if not (top / "results" / FILE).is_file():
                raise UsageError(f"{p} holds no results/{FILE}")
            return load(top / "results" / FILE)
    if p.is_file():
        return load(p)
    return load(reference_root(p)[1])


def cmd_seal(a, st):
    out = seal(a.dir, a.out, denylist_of(a.label_denylist))
    print(f"sealed {out}")
    return cliutil.EXIT_OK


# ---- submit and the approval verbs ---------------------------------------------------------------------------


def pack_dir(root, out):
    """A sealed bundle directory as a .tgz; links are refused."""
    files = sorted(p for p in Path(root).rglob("*") if p.is_file() or p.is_symlink())
    if any(p.is_symlink() for p in files):
        raise UsageError(f"{root} holds a link; a bundle holds regular files only")
    return write_tgz(out, root, [p.relative_to(root).as_posix() for p in files])


def cmd_submit(a, st, get_client):
    """Validate locally first; the client (and its credentials) is only built once the bundle is sound."""
    deny = denylist_of(a.label_denylist)
    src = Path(a.bundle)
    with tempfile.TemporaryDirectory(prefix="submit-") as tmp:
        if src.is_file() and src.name.endswith((".tgz", ".tar.gz")):
            bundle_root, path = unpack(src, tmp), src
        elif src.is_dir() and (src / BUNDLE_FILE).is_file():
            bundle_root, path = src, None
        elif src.is_dir() and (src / FILE).is_file():
            path = seal(src, Path(tmp) / "reference.tgz", deny)
            bundle_root = unpack(path, Path(tmp) / "x")
        else:
            raise UsageError(f"{src} is not a bundle (.tgz), a sealed bundle directory or a reference directory")
        if not (bundle_root / "results" / FILE).is_file():
            raise UsageError(f"{src} is not a bundle: it holds no results/{FILE}")
        doc = load(bundle_root / "results" / FILE)
        problems = check_bundle(bundle_root, doc) + validate(doc, bundle_root, deny)
        if problems:
            raise UsageError("not valid, not submitted:\n  " + "\n  ".join(problems))
        if path is None:
            path = pack_dir(src, Path(tmp) / "reference.tgz")
        print_registration(st, doc, sys.stderr)
        out = get_client().submit(path, doc, dry_run=a.dry_run, inspect=a.inspect)
    print(json.dumps(out, indent=2, sort_keys=True))
    return cliutil.EXIT_OK


def cmd_decide(a, client):
    verb = a.cmd
    if verb == "approve":
        if not a.terms_ack:
            raise UsageError("approve needs --terms-ack: the approver acknowledges the reference's terms")
        out = client.approve(a.upload_id, reason=a.reason or "", terms_ack=True, promote=not a.no_promote,
                             register_missing=a.register_missing, ack_not_clean=a.ack_not_clean)  # fmt: skip
    elif verb == "reject":
        if not (a.reason or "").strip():
            raise UsageError("reject needs --reason")
        out = client.reject(a.upload_id, reason=a.reason or "", request_changes=a.request_changes)
    else:
        out = client.retract(a.upload_id, reason=a.reason or "")
    print(json.dumps(out, indent=2, sort_keys=True))
    return cliutil.EXIT_OK


def cmd_list(a, client):
    rows = client.list(state=a.state, klass=a.cls, origin=a.origin, component=a.component, pending=a.pending,
                       mine_approvable=a.mine_approvable)  # fmt: skip
    if a.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    for r in rows:
        if not a.json:
            print(
                f"{r.get('upload_id', '-'):<10} {r.get('state', '-'):<10} v{r.get('version') or '-':<3} {r.get('key', '')}"
            )
    return cliutil.EXIT_OK if rows else cliutil.EXIT_NOTHING


def cmd_show(a, client):
    info = client.show(a.id)
    print(f"{info.get('key', '')} v{info.get('version') or '-'} {info.get('state', '-')}")
    print(json.dumps(info, indent=2, sort_keys=True))
    return cliutil.EXIT_OK


# ---- approval rules (pure; the server enforces them, FakeClient reproduces them) -----------------------------

PERMISSION_APPROVE = "references:approve"
PERMISSION_ADMIN = "references:admin"


def norm_login(who):
    return str(who or "").strip().lower()


def facts_of(doc):
    """The scope facts of a reference, taken from the validated file."""
    model, dataset = doc.get("model") or {}, doc.get("dataset") or {}
    return {
        "class": [doc["class"]], "origin": [doc["origin"]], "component": sorted({c for c, _ in case_ids(doc)}),
        "model_family": [str(model["family"]).lower()] if model.get("family") else [],
        "dataset": [str(dataset["id"])] if dataset.get("id") else [],
        "arch": archs_of(doc) or [arch_of(doc)],
        "key": [doc["key"]],
    }  # fmt: skip


def in_scope(scope, facts):
    """ANDed over dimensions; case-insensitive; an empty dimension means any, a limited one fails closed when
    the facts have no value for it."""
    for dim, allowed in (scope or {}).items():
        if not allowed:
            continue
        allowed = {str(a).strip().lower() for a in allowed}
        have = [str(h).strip().lower() for h in facts.get(dim) or []]
        if not have or not all(h in allowed for h in have):
            return False
    return True


def _utc(when):
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def holders(bindings, permission, facts, members=None, now=None):
    """Logins holding `permission` in scope: users bound directly plus members of bound groups."""
    now = now or datetime.now(timezone.utc)
    resource = permission.split(":")[0]
    out = set()
    for b in bindings:
        exp = b.get("expires_at")
        never = exp is None or _utc(exp).year <= 1970  # the epoch means no expiry
        if (
            not b.get("active", True)
            or not (never or _utc(exp) > _utc(now))
            or not in_scope(b.get("scope") or {}, facts)
        ):
            continue
        perms = b.get("permissions") or []
        if not (permission in perms or "*" in perms or f"{resource}:*" in perms):
            continue
        if b.get("subject_type", "user") == "group":
            out |= {norm_login(u) for u in (members or {}).get(b["subject"], ())}
        else:
            out.add(norm_login(b["subject"]))
    return out


@dataclass(frozen=True)
class Approval:
    allowed: bool
    error: str = ""
    override: bool = False
    self_approved: bool = False
    props: dict = field(default_factory=dict)


def decide_approval(*, submitter, approver, is_admin, approvers, reason=""):
    """A non-admin needs the binding in scope and is never the submitter; an admin may approve any, flagged."""
    who, sub = norm_login(approver), norm_login(submitter)
    if not who:
        return Approval(False, "no approver identity")
    bound = who in {norm_login(a) for a in approvers}
    selfie = who == sub

    def props(override, self_):
        return {
            "ref.approved_by": who, "ref.approval_reason": reason or "",
            "ref.approval_override": "1" if override else "", "ref.approval_self": "1" if self_ else "",
        }  # fmt: skip

    if not is_admin:
        if not bound:
            return Approval(False, f"{who} holds no references:approve binding for this reference")
        if selfie:
            return Approval(False, "the submitter cannot approve their own reference")
        return Approval(True, props=props(False, False))
    override = selfie or not bound
    return Approval(True, override=override, self_approved=selfie, props=props(override, selfie))


def decide_retract(*, submitter, actor, is_admin, state):
    """The submitter may retract a pending or published upload of their own; an admin any."""
    if state not in ("pending", "published"):
        return Approval(False, f"an upload that is {state!r} cannot be retracted")
    if is_admin or norm_login(actor) == norm_login(submitter):
        return Approval(True)
    return Approval(False, "only the submitter or an admin can retract")


def decide_reject(*, submitter, actor, is_admin, approvers):
    """Reject or request changes: an approver in scope or an admin; a submitter withdraws with retract."""
    who = norm_login(actor)
    if who == norm_login(submitter) and not is_admin:
        return Approval(False, "a submitter withdraws their own pending upload with retract")
    if is_admin or who in {norm_login(a) for a in approvers}:
        return Approval(True)
    return Approval(False, f"{who} holds no references:approve binding for this reference")


# ---- compare -------------------------------------------------------------------------------------------------


def comparability_word(block, context):
    """comparable | partial | not_comparable from a reference's comparability block and the run's own context."""
    block = {k.removeprefix("ref.comparability."): v for k, v in (block or {}).items()}
    if block.get("harness_match") == "no" or block.get("dataset_match") == "different":
        return "not_comparable"
    if block.get("harness_match") != "yes" or block.get("dataset_match") != "same":
        return "partial"
    keys = [k for k in block if k not in NOT_KEYS]
    ctx = context or {}
    return "comparable" if all(k in ctx and str(ctx[k]) == str(block[k]) for k in keys) else "partial"


def reference_rows(source):
    """(header, rows) of a reference document or of v_bench_reference_values rows."""
    if isinstance(source, dict):
        rows = [
            {"benchmark_id": v["case"]["benchmark_id"], "backend": v.get("backend", ""), "measurements": v["measurements"],
             "component": v["case"]["component"], "name": v["case"]["name"]}
            for v in source.get("values", [])
        ]  # fmt: skip
        head = {k: source.get(k) for k in ("key", "origin", "label", "clean", "warnings", "comparability")}
        return {**head, "version": None, "state": "local"}, rows
    if not source:
        return {}, []
    first = source[0]
    head = {
        "key": first.get("ref_key"), "version": first.get("version"), "state": first.get("state"),
        "origin": first.get("origin"), "label": first.get("label"), "clean": first.get("clean"),
        "warnings": first.get("warnings"), "comparability": dict(first.get("comparability") or {}),
    }  # fmt: skip
    return head, [dict(r, measurements=dict(r.get("measurements") or {})) for r in source]


def _delta(run_v, ref_v):
    delta = run_v - ref_v
    return delta, (delta / abs(ref_v) * 100 if ref_v else None)


def _status(delta, pct, tol, direction):
    if delta == 0 or (pct is not None and abs(pct) <= tol):
        return "same"
    if direction not in ("lower_is_better", "higher_is_better"):
        return (
            "changed"  # a zero reference has no percent: any nonzero delta is outside the tolerance, judged by its sign
        )
    return "better" if (delta < 0) == (direction == "lower_is_better") else "worse"


def compare_to_reference(results, source, tolerance_pct=None, metrics=None, context=None):
    """Informational comparison of a run's Results to a reference (a document, or store value rows).

    Joins on the reference-style benchmark_id of each Result's case, the backend (when the Result names one) and the
    metric; the value of each side is the median of its samples. Never gates and is never a baseline.
    """
    head, rows = reference_rows(source)
    by_id = {}
    for r in rows:
        by_id.setdefault(r["benchmark_id"], []).append(r)
    metrics = metrics or {}
    out, unmatched = [], []
    for res in results:
        samples = samples_of(res)
        case = case_block(res)
        rid = case_id(case)
        backend = (res["status"]["case"].get("props") or {}).get("system.backend", "")
        cands = [r for r in by_id.get(rid, []) if not backend or r.get("backend") in ("", backend)]
        shared = sorted(set(samples) & {m for r in cands for m in r["measurements"]})
        if not cands or not shared:
            unmatched.append(case["name"])
            continue
        for m in shared:
            ref_row = next(r for r in cands if m in r["measurements"])
            ref_v = statistics.median(ref_row["measurements"][m])
            run_v = statistics.median(samples[m])
            spec = metrics.get(m) or {}
            tol = tolerance_pct
            if tol is None:
                tol = float((spec.get("noise") or {}).get("floor_pct", DEFAULT_TOLERANCE_PCT))
            delta, pct = _delta(run_v, ref_v)
            out.append(
                {
                    "name": case["name"], "benchmark_id": rid, "backend": ref_row.get("backend", ""), "metric": m,
                    "run_value": run_v, "reference_value": ref_v, "delta": delta,
                    "percent": None if pct is None else round(pct, 4), "tolerance_pct": tol,
                    "direction": spec.get("direction"), "status": _status(delta, pct, tol, spec.get("direction")),
                }
            )  # fmt: skip
    return {
        "informational": INFORMATIONAL,
        "reference": {k: head.get(k) for k in ("key", "version", "state", "origin", "label", "clean", "warnings")},
        "comparability": comparability_word(head.get("comparability"), context),
        "rows": out, "unmatched_cases": unmatched,
    }  # fmt: skip


def local_reference(ref):
    p = Path(ref)
    if p.is_dir() or p.is_file():
        return bundle_doc(p)
    return None


def cmd_compare(a, st):
    d, manifest = find_run(a.run, a.runs_dir)
    results = run_results(d, manifest)
    source = local_reference(a.reference)
    if source is None:
        if a.version != "current" and not a.version.isdigit():
            raise UsageError("--version is current or a version number")
        version = a.version if a.version == "current" else int(a.version)
        source = st.reference_values(a.reference, version)
        if not source:
            raise UsageError(f"no stored values for reference {a.reference!r} version {a.version}")
    cat = cliutil.open_catalog(a.catalog, a.central, a.pack, required=False, implicit=False)
    report = compare_to_reference(
        results, source, a.tolerance_pct, cat.reg.metrics if cat else None, pairs(a.run_context, "--run-context")
    )
    if a.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        ref = report["reference"]
        print(
            f"reference {ref['key']} version {ref['version'] or 'local'} ({ref['origin']}): {report['comparability']}"
        )
        print(report["informational"])
        for r in report["rows"]:
            pct = "n/a" if r["percent"] is None else f"{r['percent']:+.2f}%"
            print(
                f"{r['status']:<8} {r['name']}  {r['metric']}  run {r['run_value']:g} vs ref {r['reference_value']:g}  {pct}"
            )
        for n in report["unmatched_cases"]:
            print(f"unmatched {n}")
    return cliutil.EXIT_OK if report["rows"] else cliutil.EXIT_NOTHING


# ---- command line --------------------------------------------------------------------------------------------


OFFLINE = {"create": cmd_create, "validate": cmd_validate, "seal": cmd_seal, "compare": cmd_compare}


def parser():
    ap = argparse.ArgumentParser(prog="spyre-bench reference", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="write a reference directory from a run's results")
    c.add_argument("--from-request", help="a bench.reference-request/1 file written by `accept`")
    c.add_argument(
        "--run", help="with --from-request: the accepted run's results directory, for its cleanliness and pinning"
    )
    c.add_argument("--from-run", help="results directory of a run, or its run id (with --runs-dir)")
    c.add_argument("--runs-dir", help="where to look for a run id")
    c.add_argument("--key")
    c.add_argument("--origin", choices=ORIGINS, default="generated")
    c.add_argument("--class", dest="cls", choices=CLASSES)
    c.add_argument("--label")
    c.add_argument("--terms")
    c.add_argument("--as-of")
    c.add_argument("--producer")
    c.add_argument("--source-url")
    c.add_argument("--backend", help="backend of results that name none")
    c.add_argument("--tool", action="append", help="NAME=VERSION of a tool used (repeatable)")
    c.add_argument("--comparability", action="append", help="KEY=VALUE of the comparability block (repeatable)")
    c.add_argument("--meta", help="YAML/JSON file of reference fields (model, dataset, tools, comparability, ...)")
    c.add_argument("--out", help="reference directory to create (default reference-<key>)")
    c.add_argument("--store", choices=("none", "clickhouse"))

    v = sub.add_parser("validate", help="validate a reference directory or unpacked bundle offline")
    v.add_argument("dir")

    s = sub.add_parser("seal", help="write the bench-reference bundle of a reference directory")
    s.add_argument("dir")
    s.add_argument("--out", required=True, help="bundle directory, or a .tgz path")

    sm = sub.add_parser("submit", help="submit a bundle (or reference directory) for approval")
    sm.add_argument("bundle")
    sm.add_argument("--store", choices=("none", "clickhouse"))
    sm.add_argument(
        "--dry-run", action="store_true", help="ask the service to validate without recording the submission"
    )
    sm.add_argument("--inspect", action="store_true", help="inspect the bundle only (POST /api/v2/uploads/inspect)")

    for sp in (c, v, s, sm):
        sp.add_argument("--label-denylist", help="file of regexes a label must not match (SPYRE_BENCH_LABEL_DENYLIST)")

    ap_ = sub.add_parser("approve", help="approve a pending reference (a token-holding service identity; the UI in v1)")
    ap_.add_argument("upload_id")
    ap_.add_argument("--reason")
    ap_.add_argument("--terms-ack", action="store_true")
    ap_.add_argument("--no-promote", action="store_true", help="approve without making it the current version")
    ap_.add_argument("--register-missing", action="store_true")
    ap_.add_argument(
        "--ack-not-clean", action="store_true", help="acknowledge a reference whose run was not known to be clean"
    )

    rj = sub.add_parser("reject", help="reject (or request changes to) a pending reference")
    rj.add_argument("upload_id")
    rj.add_argument("--reason")
    rj.add_argument("--request-changes", action="store_true")

    rt = sub.add_parser("retract", help="retract your pending or published reference")
    rt.add_argument("upload_id")
    rt.add_argument("--reason")

    ls = sub.add_parser("list", help="list references and pending submissions")
    ls.add_argument("--state")
    ls.add_argument("--class", dest="cls", choices=CLASSES)
    ls.add_argument("--origin", choices=ORIGINS)
    ls.add_argument("--component")
    ls.add_argument("--pending", action="store_true")
    ls.add_argument("--mine-approvable", action="store_true")
    ls.add_argument("--json", action="store_true")

    sh = sub.add_parser("show", help="show one reference or submission")
    sh.add_argument("id")

    cp = sub.add_parser("compare", help="compare a run with a reference (informational)")
    cp.add_argument("--run", required=True, help="results directory of a run, or its run id (with --runs-dir)")
    cp.add_argument("--runs-dir")
    cp.add_argument("--reference", required=True, help="a reference directory, bundle, or a key (with --store)")
    cp.add_argument("--version", default="current", help="current (default) or a version number; keys only")
    cp.add_argument(
        "--tolerance-pct", type=float, help="same-within percent (default: the metric's noise floor, else 5)"
    )
    cp.add_argument(
        "--run-context", action="append", help="KEY=VALUE the run measured under (precision=..., shape.batch=...)"
    )
    cp.add_argument("--store", choices=("none", "clickhouse"))
    cp.add_argument("--json", action="store_true")
    cliutil.add_catalog_flags(cp)
    return ap


def main(argv=None, st=None, client=None):
    from spyre_bench import reference_client

    ap = parser()

    def run(a):
        store = st or stores.open_store(cliutil.pick(getattr(a, "store", None), "BENCH_STORE", "none"))
        try:
            if a.cmd in OFFLINE:
                return OFFLINE[a.cmd](a, store)

            def get_client():
                return client or reference_client.HttpClient.from_env(write=a.cmd not in ("list", "show", "submit"))

            if a.cmd == "submit":
                return cmd_submit(a, store, get_client)
            cl = get_client()
            return {"list": cmd_list, "show": cmd_show}.get(a.cmd, cmd_decide)(a, cl)
        except reference_client.NoCredentials as e:
            raise UsageError(str(e)) from None
        except reference_client.Refused as e:
            print(f"refused: {e}", file=sys.stderr)
            return cliutil.EXIT_FAILED
        except (reference_client.ClientError, stores.StoreError) as e:
            print(f"error: {e}", file=sys.stderr)
            return cliutil.EXIT_STORE

    return cliutil.run_main(run, ap, argv)


if __name__ == "__main__":
    sys.exit(main())
