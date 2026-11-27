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

"""spyre-bench reference engine (package spyre_bench): pack registry, catalog validation, case expansion, benchmark identity, plan with reuse,
invalidation and qualification helpers, result shape, id recompute. PyYAML; jsonschema when importable,
else a stdlib schema-subset validator.

Library; the CLI verbs live in cli.py (validate, explain, gen-schemas).
"""

import copy
import difflib
import fnmatch
import itertools
import json
import math
import statistics
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import regex as re
import yaml

from spyre_bench import ingest_adapter, metric_context, pinning_plan, placement, quality
from spyre_bench.ingest_adapter import IngestUnavailable
from spyre_bench.placement import parse_cpu_list

try:
    from jsonschema import Draft202012Validator
except ImportError:  # minimal images: the stdlib validator below covers the schema subset
    Draft202012Validator = None

ROOT = Path(__file__).resolve().parent
CORE_DIR, PACKS_DIR = ROOT / "core", ROOT / "packs"
TOOL, PACKAGE, CLI_ALIAS = "spyre-bench", "spyre_bench", "bench"  # alias: optional documented symlink only
ENGINE, CORE_VERSION = "bench", "0.2.0"  # ENGINE is the apiVersion group
API = "bench/v1alpha1"
AUTHORED = (
    "Project",
    "Suite",
    "Selection",
    "Benchmark",
    "System",
    "Workload",
    "Asset",
    "Dataset",
    "Model",
    "Store",
    "Environment",
    "NodeClass",
    "ValidityPolicy",
    "Expectation",
    "Analysis",
    "Pack",
)
TOOL_WRITTEN = ("Run", "Result", "InvalidationEvent")
KINDS = AUTHORED
RESERVED = {
    "System": ("base", "patch", "type", "token", "backend", "extensions"),
    "Workload": ("base", "patch", "type", "scenario", "inputs", "load", "slo", "expand", "outputs"),
    "Store": ("type", "endpoint", "root", "auth", "visibility", "mirrors", "resolve", "offline"),
}
CORE_RELATIONS = (
    "part_of",
    "variant_of",
    "reference_for",
    "derived_from",
    "depends_on",
    "produces",
    "supersedes",
    "see_also",
)
DEFAULT_GROUPINGS = [
    {"name": "tier", "ordered": True, "cardinality": "one", "values": ["sanity", "core", "extended"]},
    {"name": "suite", "ordered": False, "cardinality": "many"},
]
CATEGORY_OK = {"perf": {"performance", "efficiency", "diagnostic"}, "quality": {"quality", "diagnostic"}}
PIN_LABEL = "env.label.cpu_pinning"  # derived from host.cpu_pinning; joins series keys and reuse
# Fingerprint attributes hashed only from the given fingerprint_version on; older versions record but ignore them.
FP_SINCE = {
    f"env.fp.{k}": 2
    for k in (
        "host.sockets",
        "host.numa_nodes",
        "host.cores_per_socket",
        "host.smt",
        "host.llc_domains",
        "host.sub_numa",
        "device.numa_node",
        "device.pcie_width",
        "device.pcie_gen",
    )
}
VALIDITY_ORDER = ["invalid", "suspect", "external", "unqualified_env", "shared_env", "valid"]
FLOORS = ("suspect", "unqualified_env", "shared_env", "valid")
VERDICTS = ("pass", "regressed", "improved", "suspected", "no_baseline")
SUBSET = {
    "$schema",
    "$id",
    "$ref",
    "$defs",
    "title",
    "description",
    "default",
    "type",
    "const",
    "enum",
    "required",
    "properties",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "format",
    "anyOf",
    "allOf",
    "if",
    "then",
}
YAML11_HINT = " (YAML 1.1 reads off/on/yes/no as booleans: quote the value)"
PY = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}
ATTR_ROOTS = {
    "identity": ("case.", "system.", "input."),
    "run": ("run.", "case.", "system.", "input.", "role."),
    "subject": ("subject.",),
    "label": ("env.label.",),
    "environment": ("env.",),
}


# ------------------------------------------------------------------ schema validation
def subset_keywords(node, where="#"):
    bad = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("properties", "$defs"):
                for n, sub in v.items():
                    bad += subset_keywords(sub, f"{where}/{k}/{n}")
            elif k not in SUBSET:
                bad.append(f"{where}: {k}")
            elif k in ("items", "additionalProperties", "if", "then") and isinstance(v, dict):
                bad += subset_keywords(v, f"{where}/{k}")
            elif k in ("anyOf", "allOf"):
                for i, sub in enumerate(v):
                    bad += subset_keywords(sub, f"{where}/{k}/{i}")
    return bad


def _subset(value, node, root, where):
    if "$ref" in node:
        node = root["$defs"][node["$ref"].split("/")[-1]]
    if "anyOf" in node and all(_subset(value, s, root, where) for s in node["anyOf"]):
        return [f"{where}: matches no allowed form"]
    if "const" in node and value != node["const"]:
        return [f"{where}: must be {node['const']!r}"]
    if "enum" in node and value not in node["enum"]:
        return [f"{where}: {value!r} is not one of {node['enum']}" + YAML11_HINT * isinstance(value, bool)]
    t = node.get("type")
    if t and (not isinstance(value, PY[t]) or (t in ("integer", "number") and isinstance(value, bool))):
        return [f"{where}: must be {t}"]
    out = []
    if isinstance(value, str):
        if len(value) < node.get("minLength", 0) or len(value) > node.get("maxLength", 1 << 30):
            out.append(f"{where}: length out of range")
        if "pattern" in node and not re.search(node["pattern"], value):
            out.append(f"{where}: {value!r} does not match {node['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < node.get("minimum", -math.inf) or value > node.get("maximum", math.inf):
            out.append(f"{where}: {value} out of range")
    if isinstance(value, list):
        if len(value) < node.get("minItems", 0):
            out.append(f"{where}: needs at least {node['minItems']} items")
        for i, x in enumerate(value):
            out += _subset(x, node.get("items", {}), root, f"{where}[{i}]")
    if isinstance(value, dict):
        out += [f"{where}: missing required key {k!r}" for k in node.get("required", ()) if k not in value]
        props, extra = node.get("properties", {}), node.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                out += _subset(v, props[k], root, f"{where}.{k}")
            elif extra is False:
                close = difflib.get_close_matches(k, list(props), 1)
                out.append(
                    f"{where}: unknown key {k!r}"
                    + (f" (did you mean {close[0]!r}?)" if close else f" (allowed: {', '.join(props)})")
                )
            elif isinstance(extra, dict):
                out += _subset(v, extra, root, f"{where}.{k}")
    return out


class Schema:
    """One compiled schema; jsonschema when available, the subset validator otherwise."""

    use_jsonschema = Draft202012Validator is not None

    def __init__(self, doc, where):
        self.doc, self.where = doc, where
        self.v = Draft202012Validator(doc) if (Draft202012Validator and self.use_jsonschema) else None

    def errors(self, value, at="$"):
        if self.v is None:
            return _subset(value, self.doc, self.doc, at)
        out = []
        for e in sorted(self.v.iter_errors(value), key=lambda e: [str(p) for p in e.absolute_path]):
            where = at + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in e.absolute_path)
            if e.validator == "additionalProperties" and isinstance(e.instance, dict):
                allowed = list((e.schema.get("properties") or {}).keys())
                for k in e.instance:
                    if k not in allowed:
                        close = difflib.get_close_matches(k, allowed, 1)
                        out.append(
                            f"{where}: unknown key {k!r}"
                            + (f" (did you mean {close[0]!r}?)" if close else f" (allowed: {', '.join(allowed)})")
                        )
            else:
                out.append(
                    f"{where}: {e.message[:240]}"
                    + YAML11_HINT * (e.validator == "enum" and isinstance(e.instance, bool))
                )
        return out


_CORE: dict[str, Schema] = {}


def core_schema(kind):
    if kind not in _CORE:
        _CORE[kind] = Schema(json.loads((CORE_DIR / "schemas" / f"{kind.lower()}.schema.json").read_text()), kind)
    return _CORE[kind]


def shape_schema(name):
    key = "shape:" + name
    if key not in _CORE:
        _CORE[key] = Schema(json.loads((CORE_DIR / "schemas" / "tool-output" / f"{name}.shape.json").read_text()), name)
    return _CORE[key]


# ------------------------------------------------------------------ documents
class Doc:
    def __init__(self, path, index, body):
        self.path, self.index, self.body = Path(path), index, body
        self.kind = body.get("kind") if isinstance(body, dict) else None
        self.name = ((body.get("metadata") or {}).get("name")) if isinstance(body, dict) else None

    @property
    def where(self):
        return f"{self.path}#{self.index}" + (f" {self.kind}/{self.name}" if self.name else "")

    @property
    def spec(self):
        return self.body.get("spec") or {}

    @property
    def meta(self):
        return self.body.get("metadata") or {}


def read_docs(paths):
    docs, errs, files = [], [], []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.yaml")) if p.is_dir() else [p]
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errs.append(f"{f}: not valid UTF-8")
            continue
        if "\t" in text:
            errs.append(f"{f}: tab character")
        try:
            for ev in yaml.parse(text):
                if getattr(ev, "anchor", None) or isinstance(ev, yaml.AliasEvent):
                    errs.append(
                        f"{f}: YAML anchors and aliases are refused (line {ev.start_mark.line + 1}); "
                        "use base: or a Workload/System reference"
                    )
                    break
            bodies = list(yaml.safe_load_all(text))
        except yaml.YAMLError as x:
            errs.append(f"{f}: YAML error: {x}")
            continue
        docs += [Doc(f, i, b) for i, b in enumerate(bodies) if b is not None]
    return docs, errs


def envelope_errors(d):
    b = d.body
    if not isinstance(b, dict):
        return ["document is not a mapping"]
    out = []
    if "status" in b:
        out.append("status: tool-written state is never authored; remove it from source")
        b = {k: v for k, v in b.items() if k != "status"}
    if b.get("apiVersion") != API:
        return out + [f"apiVersion: must be {API!r}"]
    if d.kind in TOOL_WRITTEN:
        return out + [
            f"kind: {d.kind} is tool output (rows are written by the tool), never authored; see core/schemas/tool-output"
        ]
    if d.kind not in KINDS:
        close = difflib.get_close_matches(str(d.kind), KINDS, 1)
        return out + [f"kind: {d.kind!r} is not a core kind" + (f" (did you mean {close[0]}?)" if close else "")]
    return out + core_schema(d.kind).errors(b)


# ------------------------------------------------------------------ ids (the repo's DerivedId / BenchmarkId rule)
# Opaque constant: must equal ID_NAMESPACE in clickhouse-ingest's spyre_clickhouse_ingest/identity.py (tested when present).
ID_NAMESPACE = uuid.UUID("cb0af9bf-2858-5eab-9211-f51190531bf3")
ID_SEP = "|"
RUN_CONTEXT_TAG_NAMESPACES = frozenset({"platform", "testtype", "cadence"})
RESULT_TAG_NAMESPACES = frozenset({"refcoverage"})
LEGACY_TAG_ALIASES = {
    "nightly": "cadence__nightly",
    "weekly": "cadence__weekly",
    "fvt": "testtype__fvt",
    "svt": "testtype__svt",
    "spyre-inference": "domain__spyre-inference",
    "spyre-backend": "domain__spyre-backend",
    "torch-spyre": "domain__torch-spyre",
}


# Discriminator key lists of the ingest producers, in hash order (compared with the ingest's constants by
# ingest_adapter.profile_keys). `generic` has no fixed list: the keys are the sorted prop names the block gives.
INGEST_DISC_KEYS = ("record_type", "config_name", "input_shapes", "run_mode", "kernel_name", "is_total")
_VLLM_KEYS = ("record_type", "run_mode", "tensor_parallel", "input_len", "output_len")
ID_PROFILES = {
    "perf-kernel": INGEST_DISC_KEYS,
    "vllm": _VLLM_KEYS,
    "vllm-sweep": _VLLM_KEYS + ("concurrency", "num_prompts", "max_model_len", "max_num_seqs"),
}
GENERIC_PROFILE = "generic"
DEFAULT_PROFILE = "perf-kernel"
ID_SOURCES = ("case.", "system.", "input.")
TAG_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,62}\Z"
IDENT_NAME = r"^[a-z0-9_][a-z0-9._@:/-]{0,199}\Z"
RUN_LEVEL_TAGS = ("tier", "suite", "run")


def check_profile(profile):
    """'match'; 'skipped (...)' when the ingest has no producer module (vendored identity); 'unverified (...)' when it
    has one that cannot be imported; IngestUnavailable on a difference."""
    fixed = ID_PROFILES.get(profile)
    if not fixed:
        return "skipped (no fixed key list)"
    kind, theirs = ingest_adapter.profile_status(profile)
    if kind == "absent":
        return f"skipped (ingest producer module absent: {theirs})"
    if kind == "unverified":
        return f"unverified ({theirs})"
    if tuple(theirs) != fixed:
        raise IngestUnavailable(
            f"the ingest hashes {list(theirs)} for profile {profile}, the engine expects {list(fixed)}: "
            "the ingest rule changed; update ID_PROFILES, the examples and the conformance table together"
        )
    return "match"


def under(n, names, delims):
    """True when `n` is one of `names` or starts with one followed by a delimiter character."""
    return any(n == g or (n.startswith(g) and n[len(g)] in delims) for g in names)


def norm(value):
    return ("" if value is None else str(value)).strip().lower()


def derive(*parts):
    return str(uuid.uuid5(ID_NAMESPACE, ID_SEP.join(parts)))


def split_tags(tags):
    """(identity tags, run-context tags): only identity tags are hashed."""
    ident, ctx = set(), set()
    for t in (LEGACY_TAG_ALIASES.get(norm(x), x) for x in (tags or []) if norm(x)):
        ns = norm(t).split("__", 1)[0]
        if ns in RESULT_TAG_NAMESPACES:
            continue
        (ctx if ns in RUN_CONTEXT_TAG_NAMESPACES else ident).add(t)
    return sorted(ident), sorted(ctx)


def tag_part(tags):
    return ",".join(sorted({t for t in (norm(x) for x in (tags or [])) if t}))


def disc_part(disc, disc_keys):
    disc = disc or {}
    return ",".join(f"{k}={norm(disc.get(k))}" for k in disc_keys or ())


def benchmark_id(component, name, tags, disc=None, disc_keys=()):
    """BenchmarkId.derive of the repo: component, name, identity tags, discriminators; `backend` is not hashed."""
    if not (norm(component) and norm(name)):
        return ""
    return derive(norm(component), norm(name), tag_part(tags), disc_part(disc, disc_keys))


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hid(tag, obj):
    return derive(tag, canonical(obj))


def gstr(v):
    return ("true" if v else "false") if isinstance(v, bool) else str(v)


def dstr(v):
    """A discriminator value as a stable string."""
    if isinstance(v, bool):
        return gstr(v)
    if isinstance(v, (dict, list)):
        return canonical(v)
    return "" if v is None else str(v)


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


# ------------------------------------------------------------------ small helpers
def merge_patch(target, patch):
    """RFC 7386: maps merge recursively, null deletes, lists and scalars replace."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    out = copy.deepcopy(target) if isinstance(target, dict) else {}
    for k, v in patch.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = merge_patch(out.get(k), v)
    return out


def drop_paths(d, paths):
    d = copy.deepcopy(d)
    for p in paths:
        node, parts = d, p.split(".")
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    return d


def walk(node, path="$"):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}")
    yield path, node


def nested(path, value):
    out = value
    for part in reversed(path.split(".")):
        out = {part: out}
    return out


def typed_fields(spec, for_):
    return {k: v for k, v in spec.items() if k not in RESERVED[for_]}


def dedupe(items):
    return list(dict.fromkeys(items))


DUR = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(s|m|h|d)$")
QTY = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(us|ms|s)$")


def seconds(text):
    m = DUR.match(str(text))
    return float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)] if m else None


def max_age_seconds(text):
    secs = seconds(text)
    if not secs:
        raise PlanError(f"reuse max_age {text!r}: use a positive duration such as 7d, 12h or 30m")
    return secs


def canon_workload(wl):
    """Durations to seconds and SLO bounds to base units, so 10m == 600s and 2000ms == 2s in identity."""
    wl = copy.deepcopy(wl)

    def dur(node):
        for k in ("duration", "warmup"):
            if isinstance(node.get(k), str) and seconds(node[k]) is not None:
                node[k] = seconds(node[k])

    if isinstance(wl.get("load"), dict):
        dur(wl["load"])
        for st in wl["load"].get("stages", []):
            dur(st)
    for bounds in (wl.get("slo") or {}).values():
        for op, v in list(bounds.items()):
            m = QTY.match(v) if isinstance(v, str) else None
            if m:
                bounds[op] = float(m.group(1)) * {"us": 1e-6, "ms": 1e-3, "s": 1}[m.group(2)]
    return wl


def key_regex(key):
    return re.compile("^" + re.escape(key).replace(re.escape("<name>"), ".+") + "$")


def key_sample(key):
    return key.replace("<name>", "x")


def glob_regex(pattern):
    return re.compile("^" + re.escape(pattern).replace(r"\*", ".*") + "$")


# ------------------------------------------------------------------ load / select / predicate checks
LOAD_NEEDS = {
    "closed": (("concurrency",), ("requests", "duration")),
    "open": (("rate",), ("requests", "duration")),
    "batch": (("requests",), ()),
    "iterations": (("iterations",), ()),
    "ramp": (("stages",), ()),
}
LOAD_FIELDS = {
    "closed": {"concurrency", "requests", "duration", "warmup", "warmup_requests"},
    "open": {"rate", "arrival", "requests", "duration", "warmup", "warmup_requests"},
    "batch": {"requests", "warmup_requests"},
    "iterations": {"iterations", "warmup_iterations"},
    "ramp": {"stages", "warmup"},
}


def check_load(load, allowed, where):
    out = []
    if allowed is not None and not allowed:
        return [f"{where}: this workload type takes no load block"] if load else []
    if load is None:
        return [f"{where}: load is required (executors: {', '.join(allowed)})"] if allowed else []
    ex = load.get("executor")
    if allowed and ex not in allowed:
        out.append(f"{where}: load.executor {ex!r} is not accepted by this workload type ({', '.join(allowed)})")
    req, one_of = LOAD_NEEDS.get(ex, ((), ()))
    out += [f"{where}: load.executor {ex} needs {k}" for k in req if k not in load]
    if one_of and not any(k in load for k in one_of):
        out.append(f"{where}: load.executor {ex} needs one of {' or '.join(one_of)}")
    out += [
        f"{where}: load.{k} does not apply to executor {ex}"
        for k in load
        if ex in LOAD_FIELDS and k != "executor" and k not in LOAD_FIELDS[ex]
    ]
    return out


PRED_OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "notin", "between", "exists", "matches")


def check_predicate(p, where):
    if "any" in p or "not" in p:
        if len(p) != 1:
            return [f"{where}: any/not stand alone"]
        subs = p["any"] if "any" in p else [p["not"]]
        return [e for s_ in subs for e in check_predicate(s_, where)]
    ops = [k for k in p if k in PRED_OPS]
    if "field" not in p or len(ops) != 1:
        return [f"{where}: a predicate is {{field, exactly one of {', '.join(PRED_OPS)}}}"]
    if "matches" in ops:
        try:
            re.compile(p["matches"])
        except re.error as x:
            return [f"{where}: matches: {x}"]
    return []


def check_select(sel, where, asset_names=None):
    out = []
    if "sample" in sel and ("limit" in sel or "offset" in sel):
        out.append(f"{where}: sample excludes limit/offset")
    if "sample" in sel and ("n" in sel["sample"]) == ("fraction" in sel["sample"]):
        out.append(f"{where}: sample needs exactly one of n, fraction")
    if sel.get("from") and asset_names is not None and sel["from"] not in asset_names:
        out.append(f"{where}: from {sel['from']!r} is not an asset of the dataset ({sorted(asset_names)})")
    for i, p in enumerate(sel.get("filter", [])):
        out += check_predicate(p, f"{where}.filter[{i}]")
    return out


# ------------------------------------------------------------------ packs
def compatible(have, want):
    """semver caret: same major, and for 0.x the same minor; have >= want."""
    h, w = [int(x) for x in have.split("-")[0].split(".")], [int(x) for x in (want + ".0.0").split(".")[:3]]
    if h[0] != w[0] or (h[0] == 0 and h[1] != w[1]):
        return False
    return h >= w


UCUM = re.compile(r"^(1|[A-Za-z]+|\{[a-z]+\})(/(s|[A-Za-z]+|\{[a-z]+\}))?$")


class Registry:
    def __init__(self):
        self.packs, self.types, self.ext, self.attrs, self.metrics = {}, {}, {}, {}, {}
        self.device_classes, self.relations, self.analyzers, self.resolvers = {}, set(), {}, {}
        self.builtin_metrics = set()
        self.vocab = {"test_types": set(), "levels": set(), "purposes": set()}
        self.runners, self.probes, self.gates, self.errors = {}, {}, {}, []
        self.load(CORE_DIR)

    def load(self, pack_dir):
        pack_dir = Path(pack_dir)
        docs, errs = read_docs([pack_dir / "pack.yaml"])
        self.errors += errs
        packs = [d for d in docs if d.kind == "Pack"]
        if len(packs) != 1 or len(docs) != 1:
            self.errors.append(f"{pack_dir}/pack.yaml: needs exactly one Pack document and nothing else")
            return None
        p = packs[0]
        e = envelope_errors(p)
        if e:
            self.errors += [f"{p.where}: {x}" for x in e]
            return None
        sp, prefix = p.spec, p.spec["prefix"]
        if prefix in self.packs:
            if self.packs[prefix]["dir"] != pack_dir:
                self.errors.append(f"{p.where}: prefix {prefix!r} already registered by {self.packs[prefix]['dir']}")
            return self.packs[prefix]
        if prefix != "core" and not compatible(CORE_VERSION, sp.get("requires", {}).get("core", "0.2")):
            self.errors.append(f"{p.where}: requires core {sp['requires']['core']}, engine core is {CORE_VERSION}")
        info = {"name": p.name, "prefix": prefix, "version": sp["version"], "dir": pack_dir}
        self.packs[prefix] = info
        self._types(p, sp, prefix, pack_dir)
        for k, vals in (sp.get("vocabularies") or {}).items():
            self.vocab[k] |= set(vals)
        self._attributes(p, sp, prefix)
        for dc in sp.get("device_classes", []):
            if dc["name"] in self.device_classes:
                self.errors.append(f"{p.where}: device class {dc['name']} already registered")
            self.device_classes[dc["name"]] = {
                **dc,
                "pack": prefix,
                "attrs": {a["name"]: a for a in dc.get("attributes", [])},
            }
        for m in sp.get("metrics", []):
            if m["name"].split(".", 1)[0] != prefix:
                self.errors.append(f"{p.where}: metric {m['name']} is outside the pack namespace {prefix}.*")
            if not UCUM.match(str(m["unit"])):
                self.errors.append(
                    f"{p.where}: metric {m['name']}: unit {m['unit']!r} is not a UCUM unit in the allowed subset"
                )
            self.metrics[m["name"]] = m
            if pack_dir.resolve() == CORE_DIR.resolve() or pack_dir.resolve().parent == PACKS_DIR.resolve():
                self.builtin_metrics.add(m["name"])
        vocab = metric_context.default_vocab()
        self.errors += [f"{p.where}: {e}" for e in metric_context.pack_errors(sp.get("metrics", []), vocab)]
        self.relations |= {f"{prefix}/{r}" for r in sp.get("relations", [])}
        for kind, reg in (("runners", self.runners), ("probes", self.probes)):
            for r in sp.get(kind, []):
                reg[r["name"]] = {**r, "pack": prefix}
        for kind, reg in (("analyzers", self.analyzers), ("resolvers", self.resolvers)):
            for r in sp.get(kind, []):
                if not r["name"].startswith(prefix + "/"):
                    self.errors.append(f"{p.where}: {kind[:-1]} {r['name']} must be named {prefix}/<name>")
                reg[r["name"]] = r
        for g in sp.get("gates", []):
            if not g["id"].startswith(prefix + "."):
                self.errors.append(f"{p.where}: gate {g['id']} is outside the pack namespace {prefix}.*")
            self.gates[g["id"]] = g
        for st in sp.get("storage", []):
            if st["attribute"] not in self.attrs:
                self.errors.append(f"{p.where}: storage attribute {st['attribute']} is not in the attribute registry")
        return info

    def _types(self, p, sp, prefix, pack_dir):
        for t in sp.get("types", []):
            sch = self._schema(pack_dir / t["schema"], p.where)
            if sch is None:
                continue
            props = set(sch.doc.get("properties", {}))
            clash = props & set(RESERVED[t["for"]])
            if clash:
                self.errors.append(f"{p.where}: type {t['name']} redefines core fields {sorted(clash)}")
            for f_ in t.get("refs", {}):
                if f_ not in props:
                    self.errors.append(f"{p.where}: type {t['name']}: refs.{f_} is not a field of its schema")
            for sc, load in (t.get("scenarios") or {}).items():
                self.errors += [
                    f"{p.where}: type {t['name']} scenario {sc}: {x}" for x in check_load(load, t.get("load"), "load")
                ]
            self.types[(t["for"], f"{prefix}/{t['name']}")] = {**t, "schema": sch, "pack": prefix}
        for x in sp.get("extensions", []):
            sch = self._schema(pack_dir / x["schema"], p.where)
            if sch:
                self.ext[(x["for"], prefix)] = {"schema": sch, "identity": x.get("identity", False)}

    def _attributes(self, p, sp, prefix):
        for a in sp.get("attributes", []):
            key, cls = a["key"], a["class"]
            if key in self.attrs:
                self.errors.append(f"{p.where}: attribute {key} already registered by pack {self.attrs[key]['pack']}")
            if not key.startswith(ATTR_ROOTS[cls]) or (cls == "environment" and key.startswith("env.label.")):
                self.errors.append(
                    f"{p.where}: attribute {key}: class {cls} keys start with {', '.join(ATTR_ROOTS[cls])}"
                )
            if (a["role"] == "derived") != bool(a.get("derived_from")):
                self.errors.append(f"{p.where}: attribute {key}: role derived needs derived_from, and only then")
            elif a.get("derived_from") and a["derived_from"] not in self.attrs and a["derived_from"] != key:
                self.errors.append(f"{p.where}: attribute {key}: derived_from {a['derived_from']} is not registered")
            if a.get("fingerprint_since", FP_SINCE.get(key, 1)) != FP_SINCE.get(key, 1):
                self.errors.append(f"{p.where}: attribute {key}: fingerprint_since must match the engine's table")
            if a.get("fingerprint") and cls != "environment":
                self.errors.append(f"{p.where}: attribute {key}: fingerprint applies to class environment only")
            if a["type"] == "enum" and not a.get("values"):
                self.errors.append(f"{p.where}: attribute {key}: type enum needs values")
            self.attrs[key] = {**a, "pack": prefix}

    def _schema(self, path, where):
        try:
            doc = json.loads(Path(path).read_text())
        except (OSError, ValueError) as x:
            self.errors.append(f"{where}: schema {path}: {x}")
            return None
        if Draft202012Validator:
            try:
                Draft202012Validator.check_schema(doc)
            except Exception as x:  # noqa: BLE001 - report any metaschema failure
                self.errors.append(f"{where}: schema {path} fails the 2020-12 metaschema: {str(x)[:200]}")
        bad = subset_keywords(doc)
        if bad:
            self.errors.append(f"{where}: schema {path} uses keywords outside the subset: {bad}")
        return Schema(doc, str(path))

    def find(self, name, extra_dirs):
        index = yaml.safe_load((PACKS_DIR / "index.yaml").read_text())["packs"]
        for d in list(extra_dirs) + [PACKS_DIR / n for n in index]:
            f = Path(d) / "pack.yaml"
            if f.exists() and (next(yaml.safe_load_all(f.read_text())).get("metadata") or {}).get("name") == name:
                return Path(d)
        return None

    def type(self, for_, ref, where, out):
        t = self.types.get((for_, ref))
        if t is None:
            known = sorted(r for (f, r) in self.types if f == for_)
            close = difflib.get_close_matches(str(ref), known, 1)
            prefix = str(ref).split("/")[0]
            hint = (f" (did you mean {close[0]!r}?)" if close else "") + (
                f"; no pack with prefix {prefix!r} is loaded: add it to Project.spec.packs or pass --pack DIR"
                if prefix not in self.packs
                else f"; registered: {', '.join(known)}"
            )
            out.append(f"{where}: {for_} type {ref!r} is not registered" + hint)
        return t

    def lookup(self, key):
        """The registry entry for a concrete attribute key: exact first, then <name> patterns."""
        if key in self.attrs:
            return self.attrs[key]
        return next((a for k, a in self.attrs.items() if "<name>" in k and key_regex(k).match(key)), None)

    def matches_registry(self, pattern):
        """True when a concrete key or a glob names at least one registered attribute."""
        if "*" not in pattern:
            return self.lookup(pattern) is not None
        rx = glob_regex(pattern)
        return any(rx.match(key_sample(k)) for k in self.attrs)

    def finish(self):
        """Cross-pack references, checked once every requested pack is loaded."""
        for (f, ref), t in self.types.items():
            self.errors += [
                f"type {ref}: default_metrics: {m!r} is not a registered metric"
                for m in t.get("default_metrics", [])
                if m not in self.metrics
            ]
            self.errors += [
                f"type {ref}: level {lv!r} is not a registered level"
                for lv in t.get("levels", [])
                if lv not in self.vocab["levels"]
            ]
        for name, a in self.analyzers.items():
            self.errors += [
                f"analyzer {name}: produces {m!r}: not a registered metric"
                for m in a.get("produces", [])
                if m not in self.metrics
            ]

    def fingerprint_keys(self, version=None):
        """Recorded fingerprint attributes; with a version, only those hashed into env_id under it."""
        return {
            k
            for k, a in self.attrs.items()
            if a.get("fingerprint") and (version is None or version >= FP_SINCE.get(k, 1))
        }


# ------------------------------------------------------------------ catalog
class Catalog:
    def __init__(self, paths, packs=(), central=()):
        self.docs, self.errors = read_docs(paths)
        self.errors += [
            f"{p}: expected a catalog directory, not a file" for p in [*paths, *central] if Path(p).is_file()
        ]
        cdocs, cerr = read_docs(central) if central else ([], [])
        self.errors += cerr
        self.errors += [
            f"--pack {p}: no pack.yaml in that directory" for p in packs if not (Path(p) / "pack.yaml").is_file()
        ]
        self.reg = Registry()
        self.cases, self.by = [], {}
        for d in self.docs + cdocs:
            self.errors += [f"{d.where}: {x}" for x in envelope_errors(d)]
        if self.errors:
            return
        projects = [d for d in self.docs if d.kind == "Project"]
        if len(projects) != 1:
            self.errors.append(f"{paths}: a catalog needs exactly one Project (found {len(projects)})")
            return
        self.project = projects[0]
        self.groupings = {g["name"]: g for g in self.project.spec.get("groupings", DEFAULT_GROUPINGS)}
        self.component = self.project.spec.get("component", self.project.name)
        self.policy = None
        self.quality_policy = None
        for d in [self.project] + [x for x in cdocs if x.kind == "Project"]:
            for want in d.spec["packs"]:
                found = self.reg.find(want["name"], packs)
                if found is None:
                    self.errors.append(
                        f"{d.where}: pack {want['name']!r} not found in packs/index.yaml or --pack "
                        f"dirs; register it at run time with --pack <dir>"
                    )
                    continue
                info = self.reg.load(found)
                if info and not compatible(info["version"], want["version"]):
                    self.errors.append(
                        f"{d.where}: pack {want['name']} {info['version']} does not satisfy {want['version']}"
                    )
        self.reg.finish()
        self.errors += self.reg.errors
        for d in self.docs + cdocs:
            if d.kind in ("Project", "Pack") or not d.name:
                continue
            key = (d.kind, d.name)
            if key in self.by:
                self.errors.append(f"{d.where}: duplicate {d.kind}/{d.name} (first: {self.by[key].where})")
            self.by[key] = d
        if self.errors:
            return
        self.local = set(map(id, self.docs))
        self._check_project()
        self._check_resources()
        self._expand()
        self._check_cases()
        self._check_policies()
        self._check_selections()
        self._check_relations()
        self._check_analyses()
        self.errors = dedupe(self.errors)

    # ------------------------------------------------------------- lookup
    def of(self, kind):
        return [d for (k, _), d in self.by.items() if k == kind]

    def get(self, kind, name, where, field):
        d = self.by.get((kind, name))
        if d is None:
            names = [n for (k, n) in self.by if k == kind]
            close = difflib.get_close_matches(str(name), names, 1)
            self.errors.append(
                f"{where}: {field} {name!r}: no {kind} of that name"
                + (f" (did you mean {close[0]!r}?)" if close else "")
            )
        return d

    def effective(self, kind, name, where="", field="", seen=()):
        """Resolve `base` + `patch` (RFC 7386); a sibling `token` stays on the child."""
        d = self.get(kind, name, where, field)
        if d is None:
            return None
        spec = {k: v for k, v in d.spec.items() if k not in ("base", "patch")}
        if "base" not in d.spec:
            if "patch" in d.spec:
                self.errors.append(f"{d.where}: patch needs a base")
            return spec
        loose = sorted(set(spec) - {"token"})
        if loose:
            self.errors.append(f"{d.where}: with base, changes go under patch (RFC 7386); found {loose}")
        if name in seen or len(seen) > 4:
            self.errors.append(f"{d.where}: base chain {' -> '.join(seen + (name,))} is cyclic or deeper than 4")
            return None
        parent = self.effective(kind, d.spec["base"], d.where, "spec.base", seen + (name,))
        if parent is None:
            return None
        out = merge_patch(parent, d.spec.get("patch") or {})
        if "token" in spec:
            out["token"] = spec["token"]
        return out

    def inline(self, kind, ref, where, field):
        """A name, or an inline spec: {base, patch?, overrides...} over a named resource, or a typed spec."""
        if isinstance(ref, str):
            return self.effective(kind, ref, where, field)
        if "base" not in ref:
            return ref
        parent = self.effective(kind, ref["base"], where, field + ".base")
        if parent is None:
            return None
        rest = {k: v for k, v in ref.items() if k not in ("base", "patch")}
        return merge_patch(merge_patch(parent, ref.get("patch") or {}), rest)

    # ------------------------------------------------------------- typed resources
    def check_typed(self, for_, spec, where):
        out = []
        t = self.reg.type(for_, spec.get("type"), where, out) if spec.get("type") else None
        if spec.get("type") is None:
            out.append(f"{where}: spec.type is required (after base)")
        if t:
            out += [f"{where}: {x}" for x in t["schema"].errors(typed_fields(spec, for_), "$.spec")]
        self.errors += out
        return t

    def check_extensions(self, for_, ext, where):
        for prefix, block in (ext or {}).items():
            x = self.reg.ext.get((for_, prefix))
            if x is None:
                known = sorted(p for (f, p) in self.reg.ext if f == for_)
                self.errors.append(
                    f"{where}: extensions.{prefix}: no loaded pack extends {for_} with prefix "
                    f"{prefix!r} (registered: {known or 'none'})"
                )
            else:
                self.errors += [f"{where}: {e}" for e in x["schema"].errors(block, f"$.extensions.{prefix}")]

    def _check_project(self):
        sp, w = self.project.spec, self.project.where
        for s in sp.get("default_suites", []):
            if ("Suite", s) not in self.by:
                self.errors.append(f"{w}: default_suites: {s!r} has no Suite")
        if sp.get("default_environment"):
            self.get("Environment", sp["default_environment"], w, "default_environment")
        for g in sp.get("groupings", []):
            if g["ordered"] and not g.get("values"):
                self.errors.append(f"{w}: grouping {g['name']}: an ordered grouping needs values")
        for k in sp.get("series_labels", []):
            a = self.reg.lookup(k)
            if a is None or a["class"] != "label":
                self.errors.append(f"{w}: series_labels: {k} is not a registered label attribute")

    def _check_resources(self):
        for d in list(self.by.values()):
            if d.kind == "System":
                self._check_system(d)
            elif d.kind == "Workload":
                eff = self.effective("Workload", d.name, d.where)
                if eff is not None:
                    self.check_workload(eff, d.where)
            elif d.kind == "Store":
                self.check_typed("Store", d.spec, d.where)
            elif d.kind == "Asset":
                self._check_asset(d)
            elif d.kind == "Dataset":
                self._check_dataset(d)
            elif d.kind == "Model":
                self._check_refs(
                    d, [a["asset"] for a in d.spec.get("assets", [])] + ([d.spec["asset"]] if "asset" in d.spec else [])
                )
            elif d.kind == "Environment":
                eff = self.effective("Environment", d.name, d.where)
                if eff is not None:
                    if "base" in d.spec:  # a patch can break a list it replaces
                        wrapped = {"apiVersion": API, "kind": "Environment", "metadata": {"name": d.name}, "spec": eff}
                        self.errors += [
                            f"{d.where}: {e}"
                            for e in core_schema("Environment").errors(wrapped)
                            if e.startswith("$.spec")
                        ]
                    self._check_environment(d, eff)
            elif d.kind == "NodeClass":
                self.check_extensions("NodeClass", d.spec.get("extensions"), d.where)
            elif d.kind == "Suite" and d.spec["type"] not in self.reg.vocab["test_types"]:
                self.errors.append(
                    f"{d.where}: type {d.spec['type']!r} is not a registered test type "
                    f"{sorted(self.reg.vocab['test_types'])}"
                )

    def _check_refs(self, d, assets):
        if ("asset" in d.spec) == ("assets" in d.spec):
            self.errors.append(f"{d.where}: needs exactly one of asset, assets")
        for a in assets:
            self.get("Asset", a, d.where, "asset")

    def _check_system(self, d):
        eff = self.effective("System", d.name, d.where)
        if eff is None:
            return
        t = self.check_typed("System", eff, d.where)
        self.check_extensions("System", eff.get("extensions"), d.where)
        for f_, kind in ((t or {}).get("refs") or {}).items():
            if eff.get(f_) is not None:
                self.get(kind, eff[f_], d.where, f_)

    def check_workload(self, wl, where):
        t = self.check_typed("Workload", wl, where)
        if wl.get("scenario") and (t is None or wl["scenario"] not in (t.get("scenarios") or {})):
            self.errors.append(
                f"{where}: scenario {wl['scenario']!r} is not a preset of this workload type "
                f"({sorted((t or {}).get('scenarios') or {})})"
            )
        load = self.resolved_load(wl, t)
        if t:
            self.errors += check_load(load, t.get("load"), where)
        elif load is not None:
            self.errors += check_load(load, None, where)
        for m in wl.get("slo") or {}:
            if m not in self.reg.metrics:
                self.errors.append(f"{where}: slo.{m}: metric is not registered")
        names = set()
        for i, inp in enumerate(wl.get("inputs", [])):
            w = f"{where} inputs[{i}]"
            if inp["name"] in names:
                self.errors.append(f"{w}: duplicate input name {inp['name']!r}")
            names.add(inp["name"])
            self._check_input(inp, w)
        for o in wl.get("outputs", []):
            a = self.get("Asset", o["asset"], where, "outputs.asset")
            if a and a.spec["type"] != "generated":
                self.errors.append(f"{where}: outputs {o['name']}: asset {o['asset']} must be type generated")

    @staticmethod
    def resolved_load(wl, t):
        sc = ((t or {}).get("scenarios") or {}).get(wl.get("scenario"))
        if sc is None:
            return wl.get("load")
        return merge_patch(sc, wl.get("load") or {})

    def _check_input(self, inp, w):
        refs = [k for k in ("dataset", "asset", "model") if k in inp]
        if len(refs) != 1:
            self.errors.append(f"{w}: needs exactly one of dataset, asset, model")
            return
        kind = {"dataset": "Dataset", "asset": "Asset", "model": "Model"}[refs[0]]
        d = self.get(kind, inp[refs[0]], w, refs[0])
        if inp.get("subset") and (refs[0] != "dataset" or d is None or inp["subset"] not in d.spec.get("subsets", {})):
            self.errors.append(f"{w}: subset {inp['subset']!r} is not a named subset of the dataset")
        names = self._dataset_assets(d) if d is not None and kind == "Dataset" else None
        if inp.get("select"):
            self.errors += check_select(inp["select"], w + ".select", names)

    def _dataset_assets(self, d):
        return {a["name"] for a in d.spec["assets"]} if "assets" in d.spec else {"data"}

    def _check_asset(self, d):
        sp, w = d.spec, d.where
        src = sp["source"]
        if src.get("path") and src.get("slug"):
            self.errors.append(f"{w}: source takes path or slug, not both")
        forms = [
            bool(src.get("store") and (src.get("path") or src.get("slug"))),
            bool(src.get("uri")),
            bool(src.get("generator")),
        ]
        if sum(forms) != 1:
            self.errors.append(f"{w}: source needs exactly one of store+path, store+slug, uri, generator")
        st = src.get("store")
        if isinstance(st, str):
            store = self.get("Store", st, w, "source.store")
            if store and sp.get("visibility") == "public" and store.spec["visibility"] == "internal":
                self.errors.append(f"{w}: a public Asset cannot be served from the internal Store {st!r}")
        elif isinstance(st, dict):
            self.check_typed("Store", {"visibility": "internal", **st}, w + " source.store")
        integ = sp.get("integrity") or {}
        if sp["type"] == "generator":
            if not src.get("generator"):
                self.errors.append(f"{w}: type generator needs source.generator")
        elif sp["type"] == "generated":
            pb = sp.get("produced_by")
            if not pb:
                self.errors.append(f"{w}: type generated needs produced_by")
            elif ("Benchmark", pb["benchmark"]) not in self.by:
                self.errors.append(f"{w}: produced_by.benchmark {pb['benchmark']!r}: no Benchmark of that name")
            if not (integ.get("manifest") or integ.get("files")) and not (pb or {}).get("inputs_digest"):
                self.errors.append(f"{w}: a generated asset needs integrity or produced_by.inputs_digest")
        elif not (integ.get("files") or integ.get("manifest") or integ.get("revision_pinned")):
            self.errors.append(f"{w}: integrity needs files, a manifest or revision_pinned")
        if integ.get("revision_pinned") and not re.match(r"^[0-9a-f]{40}$", str(src.get("revision", ""))):
            self.errors.append(f"{w}: revision_pinned needs source.revision as a 40-hex commit")

    def _check_dataset(self, d):
        self._check_refs(
            d, [a["asset"] for a in d.spec.get("assets", [])] + ([d.spec["asset"]] if "asset" in d.spec else [])
        )
        if d.spec.get("store"):
            self.get("Store", d.spec["store"], d.where, "store")
        names = self._dataset_assets(d)
        if "assets" in d.spec and len(names) != len(d.spec["assets"]):
            self.errors.append(f"{d.where}: duplicate asset names")
        for n, sel in d.spec.get("subsets", {}).items():
            self.errors += check_select(sel, f"{d.where} subsets.{n}", names)

    # ------------------------------------------------------------- environments
    def _check_environment(self, d, env):
        w = d.where
        self.check_extensions("Environment", env.get("extensions"), w)
        names, classes = {}, {}
        for block, kinds in (("devices", None), ("interconnect", {"network"})):
            for r in (env.get(block) or {}).get("requests", []):
                rw = f"{w} {block}.requests.{r['name']}"
                if r["name"] in names:
                    self.errors.append(f"{rw}: duplicate request name")
                if "device_class" not in r:
                    continue
                dc = self.reg.device_classes.get(r["device_class"])
                if dc is None:
                    known = sorted(self.reg.device_classes)
                    close = difflib.get_close_matches(r["device_class"], known, 1)
                    self.errors.append(
                        f"{rw}: device_class {r['device_class']!r} is not registered by a loaded pack"
                        + (f" (did you mean {close[0]!r}?)" if close else f" (registered: {', '.join(known)})")
                    )
                    continue
                if kinds and dc["kind"] not in kinds:
                    self.errors.append(f"{rw}: interconnect requests must be network device classes")
                if block == "devices" and dc["kind"] == "network":
                    self.errors.append(f"{rw}: network classes belong under interconnect")
                names[r["name"]] = dc
                classes[r["name"]] = r
                for sel in r.get("selectors", []):
                    self._check_selector(sel, dc, rw)
        for i, c in enumerate((env.get("devices") or {}).get("constraints", [])):
            cw = f"{w} devices.constraints[{i}]"
            for rn in c["requests"]:
                if rn not in classes or rn not in {x["name"] for x in (env.get("devices") or {}).get("requests", [])}:
                    self.errors.append(f"{cw}: request {rn!r} is not a devices request")
                elif c["match_attribute"] not in names[rn]["attrs"]:
                    self.errors.append(
                        f"{cw}: match_attribute {c['match_attribute']!r} is not an attribute of "
                        f"{classes[rn]['device_class']}"
                    )
        if env.get("nodes", 1) > 1 and not (env.get("interconnect") or {}).get("requests"):
            self.errors.append(
                f"{w}: nodes > 1 needs an interconnect request (the fabric is part of what the case needs)"
            )
        host = env.get("host")
        host = host if isinstance(host, dict) else {}
        cl = host.get("clean")
        if cl:
            if not host.get("exclusive"):
                self.errors.append(f"{w}: host.clean needs host.exclusive: true")
            if not seconds(cl["max_hold"]):
                self.errors.append(f"{w}: host.clean.max_hold must be positive")
        pin = pin_block(env)
        if pin:
            self.errors += [f"{w}: {x}" for x in pin_problems(pin, self.accelerator_count(env))]
            if "cpu_pinning" in (env.get("labels") or {}):
                self.errors.append(f"{w}: labels.cpu_pinning is derived from host.cpu_pinning; remove it")
        if host.get("node_class"):
            n = self.get("NodeClass", host["node_class"], w, "host.node_class")
            if n and n.spec.get("arch") and host.get("arch") and n.spec["arch"] != host["arch"]:
                self.errors.append(f"{w}: NodeClass/{host['node_class']} arch {n.spec['arch']} != {host['arch']}")
        for k, v in (env.get("labels") or {}).items():
            if not re.match(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$", k):
                self.errors.append(f"{w}: labels.{k}: keys are dotted lowercase names")
            a = self.reg.lookup("env.label." + k)
            if a is None:
                self.errors.append(f"{w}: labels.{k}: env.label.{k} is not a registered attribute")
            elif a.get("values") and gstr(v) not in [gstr(x) for x in a["values"]]:
                self.errors.append(f"{w}: labels.{k}: {v!r} is not one of {a['values']}")
        lab = (env.get("labels") or {}).get("device.count")
        if lab is not None and str(lab).isdigit() and int(lab) != self.accelerator_count(env):
            self.errors.append(
                f"{w}: labels.device.count {lab} != {self.accelerator_count(env)} accelerators requested"
            )

    def _check_selector(self, sel, dc, where):
        ops = [k for k in ("eq", "ne", "gt", "gte", "lt", "lte", "in") if k in sel]
        if "attribute" not in sel or len(ops) != 1:
            self.errors.append(f"{where}: a selector is {{attribute, exactly one of eq|ne|gt|gte|lt|lte|in}}")
            return
        a = dc["attrs"].get(sel["attribute"])
        if a is None:
            close = difflib.get_close_matches(sel["attribute"], list(dc["attrs"]), 1)
            self.errors.append(
                f"{where}: {sel['attribute']!r} is not an attribute of {dc['name']}"
                + (f" (did you mean {close[0]!r}?)" if close else f" (has: {', '.join(dc['attrs'])})")
            )
            return
        vals = sel[ops[0]] if ops[0] == "in" else [sel[ops[0]]]
        for v in vals:
            numeric = a["type"] in ("int", "float")
            if numeric != (isinstance(v, (int, float)) and not isinstance(v, bool)) and a["type"] != "bool":
                self.errors.append(f"{where}: {sel['attribute']} is {a['type']}, got {v!r}")
            elif a.get("values") and ops[0] in ("eq", "in", "ne") and v not in a["values"]:
                self.errors.append(f"{where}: {sel['attribute']}={v!r} is not one of {a['values']}")

    def device_totals(self, env):
        n = env.get("nodes", 1)
        out = {}
        for block in ("devices", "interconnect"):
            for r in (env.get(block) or {}).get("requests", []):
                if "device_class" not in r:
                    continue
                out[r["device_class"]] = out.get(r["device_class"], 0) + r.get("count", 1) * (
                    n if r.get("per_node") else 1
                )
        return out

    def accelerator_count(self, env):
        return sum(
            c
            for k, c in self.device_totals(env).items()
            if self.reg.device_classes.get(k, {}).get("kind") == "accelerator"
        )

    def env_labels(self, env):
        out = {f"env.label.{k}": gstr(v) for k, v in (env.get("labels") or {}).items()}
        pin = pin_block(env)
        if pin:
            out[PIN_LABEL] = pin["policy"]
        elif PIN_LABEL in self.project.spec.get("series_labels", []):
            out[PIN_LABEL] = "none"
        return out

    def clean_view(self, env):
        cl = (env.get("host") or {}).get("clean")
        if not cl:
            return {"requested": False, "expected_exclusive": "observed"}
        nodes = env.get("nodes", 1)
        ub = round(nodes * (seconds(cl["max_wait"]) + seconds(cl["max_hold"])) / 3600, 2)
        return {
            "requested": True,
            **cl,
            "expected_exclusive": "enforced",
            "held_node_hours": {
                "upper_bound": ub,
                "expected": None,
                "note": "estimate only; measured wait + hold are recorded per run",
            },
        }

    def env_view(self, name):
        env = self.effective("Environment", name, "explain", "environment")
        if env is None:
            return None
        return {
            "name": name,
            "nodes": env.get("nodes", 1),
            "device_totals": self.device_totals(env),
            "accelerators": self.accelerator_count(env),
            "labels": self.env_labels(env),
            "reference_node": bool((env.get("host") or {}).get("reference_node")),
            "exclusive": bool((env.get("host") or {}).get("exclusive")),
            "clean": self.clean_view(env),
            "cpu_pinning": pin_block(env) or {"policy": "none"},
            "validity_floor": (env.get("policy") or {}).get("validity_floor", "valid"),
            "spec_digest": hid("spec", normal_env(env)),
        }

    # ------------------------------------------------------------- expansion
    def suite_order(self):
        return [d.name for d in self.docs if d.kind == "Suite"]

    def _expand(self):
        order = {s: i for i, s in enumerate(self.suite_order())}
        benches = sorted(
            (d for d in self.docs if d.kind == "Benchmark"),
            key=lambda d: (order.get(d.path.parent.name, len(order)), str(d.path)),
        )
        later = []
        for b in benches:
            if not self._file_ok(b, order):
                continue
            self._check_identity(b)
            modes = [k for k in ("variants", "matrix", "derive") if k in b.spec]
            if len(modes) != 1:
                self.errors.append(
                    f"{b.where}: spec needs exactly one of variants, matrix, derive (has {modes or 'none'})"
                )
            elif modes[0] == "derive":
                later.append(b)
            else:
                for name, patch, extra in self._variants(b) if modes[0] == "variants" else self._matrix(b):
                    self._add(b, name, patch, extra)
        for b in later:
            self._derive(b)
        seen, ids = {}, {}
        for c in self.cases:
            if c["name"] in seen:
                self.errors.append(f"{c['where']}: case name {c['name']!r} also produced by {seen[c['name']]}")
            seen.setdefault(c["name"], c["where"])
            unresolved = []
            self.ingest_inputs(c, unresolved)
            if unresolved:
                self.errors += unresolved
                continue
            bid = self.id_token(c)
            if bid in ids and ids[bid] != c["name"]:
                self.errors.append(
                    f"{c['where']}: same benchmark_id as case {ids[bid]!r}: they differ only in run "
                    "attributes (environment, profile, tier), which are runs, not cases"
                )
            ids.setdefault(bid, c["name"])

    def _check_identity(self, b):
        """Static rules of a Benchmark's `identity` block; `from` references are resolved per case (ingest_inputs)."""
        w, bad = f"{b.where}", []
        hours = b.spec.get("est_hours")
        if isinstance(hours, float) and not math.isfinite(hours):
            bad.append(f"est_hours: {hours!r} must be finite (the schema bounds it below by 0)")
        blk = b.spec.get("identity")
        if blk is not None:
            bad += [f"identity: {x}" for x in self._identity_errors(blk)]
        self.errors += [f"{w}: {x}" for x in bad]

    def _identity_errors(self, blk):
        bad = []
        mode = blk.get("mode")
        extra = sorted(set(blk) - {"mode"})
        if mode == "engine" and extra:
            bad.append(f"{extra} need mode: ingest (mode: engine takes no other field)")
        if mode != "ingest":
            return bad
        for f_ in ("component", "name"):
            if f_ in blk and not norm(blk[f_]):
                bad.append(f"{f_} is empty: the ingest refuses an id without a component and a name")
            elif f_ in blk and not re.match(IDENT_NAME, blk[f_]):
                bad.append(f"{f_}: {blk[f_]!r} does not match {IDENT_NAME}")
        names = {norm(g) for g in self.groupings} | {"tier", "suite"}
        groups = names | {"run", "backend"}
        values = {norm(v) for g in self.groupings.values() for v in g.get("values", [])}
        values |= {norm(d.name) for d in self.docs if d.kind == "Suite"}
        seen_tags = set()
        for t in blk.get("tags", []):
            n = norm(t)
            ns = n.split("__", 1)[0]
            if n in seen_tags:
                bad.append(f"tags: {t!r} is listed twice")
            seen_tags.add(n)
            if n in LEGACY_TAG_ALIASES or ns in RUN_CONTEXT_TAG_NAMESPACES | RESULT_TAG_NAMESPACES:
                bad.append(f"tags: {t!r} is a run-context or result tag, not an identity tag")
            elif under(n, groups, "_-.:="):
                bad.append(
                    f"tags: {t!r} starts with a run-level grouping or `backend` ({sorted(groups)}) up to a "
                    "delimiter (_ - . : =); groupings are run props and backend is unhashed, never tags"
                )
            elif n in values:
                bad.append(f"tags: {t!r} is a grouping value (a tier or suite name): a run prop, never a tag")
            elif not re.match(TAG_PATTERN, t):
                bad.append(f"tags: {t!r} does not match {TAG_PATTERN}")
        profile = blk.get("profile", DEFAULT_PROFILE)
        discs = blk.get("discriminators", [])
        keys = [d.get("key") for d in discs]
        for k in dict.fromkeys(keys):
            nk = norm(k)
            if keys.count(k) > 1:
                bad.append(f"discriminators: key {k!r} is listed twice")
            if not nk:
                bad.append("discriminators: key is empty")
            elif "backend" in re.split(r"__|[.:=]", nk) or re.split(r"__|[_.:=-]", nk)[0] == "backend":
                bad.append(f"discriminators: key {k!r}: backend is the comparison axis and is never hashed")
            elif under(nk, names, "_-.:=") or under(nk, {"run"}, ".:=") or nk.startswith("run__"):
                bad.append(f"discriminators: key {k!r} is a run-level grouping or run prop, never a hashed key")
            elif profile != GENERIC_PROFILE and k not in ID_PROFILES.get(profile, ()):
                bad.append(f"discriminators: unknown key {k!r} for profile {profile} ({list(ID_PROFILES[profile])})")
        if not bad and profile == GENERIC_PROFILE and keys != sorted(keys):
            bad.append(f"discriminators: profile generic hashes the sorted prop names; list {sorted(keys)}, not {keys}")
        if not bad and profile in ID_PROFILES and keys != list(ID_PROFILES[profile]):
            bad.append(
                f"discriminators: profile {profile} hashes exactly {list(ID_PROFILES[profile])} in that order "
                f"(use value: '' for an empty one), not {keys}: any other list derives an id that producer never writes"
            )
        for n, d in enumerate(discs):
            at = f"discriminators[{n}] ({d.get('key')})"
            if ("value" in d) == ("from" in d):
                bad.append(f"{at}: needs exactly one of value, from")
            if "join" in d and "from" not in d:
                bad.append(f"{at}: join needs from")
            if "from" in d and not d["from"].startswith(ID_SOURCES):
                bad.append(f"{at}: from {d['from']!r} must start with one of {ID_SOURCES}")
        return bad

    def _file_ok(self, b, order):
        folder, ok = b.path.parent.name, True
        if b.path.name != f"{b.name}.yaml":
            self.errors.append(f"{b.where}: file must be named {b.name}.yaml (one Benchmark per file)")
            ok = False
        if b.spec.get("suite", folder) != folder or folder not in order:
            self.errors.append(f"{b.where}: folder {folder!r} must be a Suite of this Project (and equal spec.suite)")
            return False
        suite = self.by[("Suite", folder)]
        if suite.spec["type"] != b.spec["type"]:
            self.errors.append(f"{b.where}: type {b.spec['type']!r} != Suite/{folder} type {suite.spec['type']!r}")
            ok = False
        for x in self.docs:
            if (
                x.path == b.path
                and x is not b
                and x.kind not in ("System", "Workload", "Environment", "Asset", "Model", "Dataset", "Expectation")
            ):
                self.errors.append(f"{x.where}: a Benchmark file holds one Benchmark plus the resources it alone uses")
                ok = False
        return ok

    COMMON = ("tier", "groupings", "system", "workload", "environment", "metrics", "env", "run", "profile")

    def common(self, b):
        out = {k: copy.deepcopy(b.spec[k]) for k in self.COMMON if k in b.spec}
        if b.meta.get("tags"):
            out["tags"] = list(b.meta["tags"])
        return out

    def _variants(self, b):
        for v in b.spec["variants"]:
            yield (
                v["name"],
                {k: x for k, x in v.items() if k not in ("name", "description", "previous_names")},
                {"previous_names": v.get("previous_names", [])},
            )

    def _axis_values(self, vals, where, axis):
        if "range" in vals:
            r = vals["range"]
            out, v = [], r["from"]
            while v <= r["to"] and len(out) < 10000:
                out.append(v)
                v = v * float(r["step"][1:]) if isinstance(r["step"], str) else v + r["step"]
            out = [int(x) if float(x).is_integer() else x for x in out]
        else:
            out = vals.get("values")
            if out is None:
                self.errors.append(f"{where}: axis {axis}: needs values or range")
                return []
        return out

    def _matrix(self, b):
        m = b.spec["matrix"]
        axes = []
        for a, vals in m["axes"].items():
            if isinstance(vals, dict):  # sweep: {path, values|range, token}
                fmt = vals.get("token", a + "{}")
                vals = [
                    {"token": fmt.format(v), "patch": nested(vals["path"], v)}
                    for v in self._axis_values(vals, b.where, a)
                ]
            axes.append((a, vals))
        excl = m.get("exclude", [])
        tokens = {a: [v["token"] for v in vals] for a, vals in axes}
        for e in excl:
            for a, t in e.items():
                if t not in tokens.get(a, []):
                    self.errors.append(f"{b.where}: matrix.exclude {e}: unknown axis or token {a}={t}")
        for combo in itertools.product(*[vals for _, vals in axes]):
            tok = {a: v["token"] for (a, _), v in zip(axes, combo)}
            if any(all(tok.get(a) == t for a, t in e.items()) for e in excl):
                continue
            patch, owner = {}, {}
            for (a, _), v in zip(axes, combo):
                for path, leaf in walk(v["patch"]):
                    if not isinstance(leaf, dict):
                        if path in owner and owner[path] != a:
                            self.errors.append(f"{b.where}: matrix axes {owner[path]} and {a} both set {path}")
                        owner[path] = a
                patch = merge_patch(patch, v["patch"])
            template = m.get("name_template") or b.name + "".join(f"/{a}={{{a}}}" for a, _ in axes)
            name = self.render(b, template, tok, merge_patch(self.common(b), patch))
            if name:
                yield name, patch, {}

    def _derive(self, b):
        d = b.spec["derive"]
        systems = list(dict.fromkeys(c["system"] for c in self.select(d["from"])))
        if not systems:
            self.errors.append(f"{b.where}: derive.from selects no case")
        for s in systems:
            name = self.render(b, d["name_template"], {}, merge_patch(self.common(b), {"system": s}))
            if name:
                self._add(b, name, {"system": s}, {})

    def render(self, b, template, tokens, inst):
        sys_ = self.inline("System", inst.get("system", ""), b.where, "system") or {}
        wl = self.inline("Workload", inst.get("workload") or "", b.where, "workload") or {}
        ctx = {
            "benchmark": b.name,
            **tokens,
            **{f"system.{k}": v for k, v in flatten(sys_).items()},
            **{f"workload.{k}": v for k, v in flatten(wl).items()},
        }

        def sub(mt):
            if mt.group(1) not in ctx:
                self.errors.append(
                    f"{b.where}: name template {template!r}: {{{mt.group(1)}}} is undefined "
                    f"(axes {sorted(tokens)}, benchmark, system.*, workload.*)"
                )
                return "\0"
            v = ctx[mt.group(1)]
            return ("on" if v else "off") if isinstance(v, bool) else str(v).lower()

        name = re.sub(r"\{([A-Za-z0-9_.-]+)\}", sub, template)
        return None if "\0" in name else name

    def _build(self, b, name, patch):
        inst = merge_patch(self.common(b), patch)
        where = f"{b.where} case {name}"
        if not re.match(r"^[a-z0-9][a-z0-9._=/,@-]{0,199}$", name):
            self.errors.append(f"{where}: case name has characters outside [a-z0-9._=/,@-]")
            return None
        sys_ = self.inline("System", inst["system"], where, "system")
        wl = self.inline("Workload", inst["workload"], where, "workload")
        if sys_ is None or wl is None:
            return None
        suite = self.by[("Suite", b.path.parent.name)]
        groupings = self.case_groupings(inst, suite.name, where)
        t = self.reg.types.get(("Workload", wl.get("type")))
        env_name = inst.get("environment") or self.project.spec.get("default_environment")
        return {
            "name": name,
            "where": where,
            "benchmark": b.name,
            "suite": suite.name,
            "patch": patch,
            "entry_file": f"{suite.name}/{b.path.name}",
            "type": b.spec["type"],
            "level": b.spec["level"],
            "groupings": groupings,
            "owner": b.meta.get("owner", "UNOWNED"),
            "tags": inst.get("tags", []),
            "system": inst["system"] if isinstance(inst["system"], str) else "<inline>",
            "sys": sys_,
            "workload": wl,
            "load": self.resolved_load(wl, t),
            "environment": env_name,
            "metrics": inst.get("metrics"),
            "env": inst.get("env", {}),
            "run": inst.get("run", {}),
            "profile": inst.get("profile", {}),
            "relations": b.spec.get("relations", []),
            "analyze": b.spec.get("analyze", []),
            "derived_metrics": b.spec.get("derived_metrics", []),
            "requires_reference": bool(suite.spec.get("requires_reference")),
            "identity": b.spec.get("identity"),
            "est_hours": b.spec.get("est_hours"),
        }

    def _add(self, b, name, patch, extra):
        c = self._build(b, name, patch)
        if c:
            self.cases.append({**c, **extra})

    def case_groupings(self, inst, folder_suite, where):
        g = {k: v for k, v in (inst.get("groupings") or {}).items()}
        if "tier" in inst:
            g["tier"] = inst["tier"]
        g["suite"] = dedupe(
            [folder_suite] + (g["suite"] if isinstance(g.get("suite"), list) else [g["suite"]] if "suite" in g else [])
        )
        out = {}
        for name, v in g.items():
            spec = self.groupings.get(name)
            if spec is None:
                self.errors.append(
                    f"{where}: grouping {name!r} is not declared by the Project ({sorted(self.groupings)})"
                )
                continue
            vals = [gstr(x) for x in (v if isinstance(v, list) else [v])]
            if spec["cardinality"] == "one" and isinstance(v, list):
                self.errors.append(f"{where}: grouping {name} takes one value")
            for x in vals:
                if spec.get("values") and x not in [gstr(y) for y in spec["values"]]:
                    self.errors.append(f"{where}: grouping {name}: {x!r} is not one of {spec['values']}")
            out[name] = vals
        if "tier" in self.groupings and "tier" not in out:
            self.errors.append(f"{where}: no tier (set spec.tier or groupings.tier)")
        return out

    def run_props(self, c):
        """Grouping membership as run-level props: run.<g> for one, run.<g>.<value> for many (never tags)."""
        out = {}
        for name, vals in c["groupings"].items():
            if self.groupings[name]["cardinality"] == "one":
                out[f"run.{name}"] = vals[0]
            else:
                out.update({f"run.{name}.{v}": "true" for v in vals})
        return out

    def select(self, sel, cases=None):
        out = []
        for c in self.cases if cases is None else cases:
            if sel.get("suites") and not set(sel["suites"]) & set(c["groupings"].get("suite", [])):
                continue
            if sel.get("types") and c["type"] not in sel["types"]:
                continue
            if sel.get("levels") and c["level"] not in sel["levels"]:
                continue
            if sel.get("workload_types") and c["workload"]["type"] not in sel["workload_types"]:
                continue
            if sel.get("tags") and not set(sel["tags"]) <= set(c["tags"]):
                continue
            if all(self._grouping_ok(c, g, term) for g, term in (sel.get("groupings") or {}).items()):
                out.append(c)
        return out

    def _grouping_ok(self, c, name, term):
        spec, have = self.groupings.get(name), c["groupings"].get(name, [])
        if spec is None or not have:
            return False
        order = [gstr(v) for v in spec.get("values", [])]
        for op, want in term.items():
            if op in ("at_most", "at_least"):
                if not spec["ordered"] or gstr(want) not in order or have[0] not in order:
                    return False
                i, j = order.index(have[0]), order.index(gstr(want))
                if (i > j) if op == "at_most" else (i < j):
                    return False
            elif op == "eq" and gstr(want) not in have:
                return False
            elif op in ("in", "any") and not set(gstr(x) for x in want) & set(have):
                return False
        return True

    # ------------------------------------------------------------- checks
    def _check_cases(self):
        voc = self.reg.vocab
        series = self.project.spec.get("series_labels", [])
        for c in self.cases:
            w = c["where"]
            if c["type"] not in voc["test_types"]:
                self.errors.append(f"{w}: type {c['type']!r} is not a registered test type")
            t = self.reg.types.get(("Workload", c["workload"].get("type")))
            self.check_workload(c["workload"], w + " workload")
            if c["level"] not in voc["levels"]:
                self.errors.append(f"{w}: level {c['level']!r} is not registered (known: {sorted(voc['levels'])})")
            elif t and t.get("levels") and c["level"] not in t["levels"]:
                self.errors.append(
                    f"{w}: level {c['level']!r} is not a level of workload type {c['workload']['type']} ({t['levels']})"
                )
            sys_t = self.reg.types.get(("System", c["sys"].get("type")))
            self.check_typed("System", c["sys"], w + " system")
            self.check_extensions("System", c["sys"].get("extensions"), w + " system")
            for pre, blk in (c["sys"].get("extensions") or {}).items():
                for k in flatten(blk, f"system.ext.{pre}."):
                    if self.reg.lookup(k) is None:
                        self.errors.append(
                            f"{w}: unhashed field {k} is not in the attribute registry (it must stay queryable)"
                        )
            for f_, kind in ((sys_t or {}).get("refs") or {}).items():
                if c["sys"].get(f_) is not None:
                    self.get(kind, c["sys"][f_], w, f"system.{f_}")
            self._check_case_metrics(c, t, w)
            self.errors += [f"{w}: {e}" for e in quality.case_errors(c)]
            env = None
            if c["environment"]:
                env = self.effective("Environment", c["environment"], w, "environment")
            if env is not None:
                lab = self.env_labels(env)
                miss = [k for k in series if k not in lab]
                if miss:
                    self.errors.append(f"{w}: environment {c['environment']} lacks series labels {miss}")
            for path, leaf in walk(c["workload"]):
                if re.search(r"\.(threshold|target|baseline|expected_value)$", path):
                    self.errors.append(f"{w}: {path}: numeric targets do not belong in a Workload")
            for a in c["analyze"]:
                if a["analyzer"] not in self.reg.analyzers:
                    self.errors.append(f"{w}: analyze: analyzer {a['analyzer']!r} is not registered by a loaded pack")
            for m in c["derived_metrics"]:
                if m not in self.reg.metrics:
                    self.errors.append(f"{w}: derived_metrics: {m!r} is not a registered metric")
            prof = c["profile"]
            if prof.get("default") and prof.get("supported") and prof["default"] not in prof["supported"]:
                self.errors.append(f"{w}: profile.default {prof['default']!r} is not in supported {prof['supported']}")
            for a in (prof.get("on_regression"),):
                if a and prof.get("supported") and a not in prof["supported"]:
                    self.errors.append(f"{w}: profile.on_regression {a!r} is not in supported {prof['supported']}")

    def case_metrics(self, c, t=None):
        """(required, headline) after defaults: omitted metrics means the workload type's default set."""
        t = t or self.reg.types.get(("Workload", c["workload"].get("type"))) or {}
        m = c["metrics"]
        if m is None:
            d = list(t.get("default_metrics", []))
            return d, d
        if isinstance(m, list):
            return list(m), list(m)
        return list(m.get("required", [])), list(m.get("headline", m.get("required", [])))

    def _check_case_metrics(self, c, t, w):
        m = c["metrics"]
        required, headline = self.case_metrics(c, t)
        if isinstance(m, dict):
            for h in headline:
                if h not in required:
                    self.errors.append(f"{w}: headline metric {h} is not in required")
        for name in dedupe(required + headline):
            spec = self.reg.metrics.get(name)
            if spec is None:
                close = difflib.get_close_matches(name, sorted(self.reg.metrics), 1)
                self.errors.append(
                    f"{w}: metric {name!r} is not registered" + (f" (did you mean {close[0]!r}?)" if close else "")
                )
            elif spec["category"] not in CATEGORY_OK.get(c["type"], {spec["category"]}):
                self.errors.append(f"{w}: metric {name} is category {spec['category']}, not valid for type {c['type']}")

    def _check_selections(self):
        names = {c["name"] for c in self.cases}
        for d in self.docs:
            if d.kind != "Selection":
                continue
            has = [k for k in ("cases", "selector") if k in d.spec]
            if len(has) != 1:
                self.errors.append(f"{d.where}: spec needs exactly one of cases, selector")
            elif has[0] == "cases":
                for n in d.spec["cases"]:
                    if n not in names:
                        close = difflib.get_close_matches(n, sorted(names), 1)
                        self.errors.append(
                            f"{d.where}: cases: {n!r} is not a case"
                            + (f" (did you mean {close[0]!r}?)" if close else "")
                        )
            else:
                bad = [g for g in d.spec["selector"].get("groupings", {}) if g not in self.groupings]
                if bad:
                    self.errors.append(f"{d.where}: selector.groupings {bad} are not declared by the Project")
                elif not self.select(d.spec["selector"]):
                    self.errors.append(f"{d.where}: selector selects no case")
            rm = (d.spec.get("reuse") or {}).get("match", {}).get("environment")
            for pat in (rm or {}).get("labels", []) if isinstance(rm, dict) else []:
                if not pat.startswith("env.label.") or not self.reg.matches_registry(pat):
                    self.errors.append(
                        f"{d.where}: reuse.match.environment.labels: {pat} names no registered env.label attribute"
                    )
            for n in (d.spec.get("reuse") or {}).get("force", []):
                if n != "all" and n not in names:
                    self.errors.append(f"{d.where}: reuse.force: {n!r} is not a case")

    def _check_relations(self):
        benches = {d.name for d in self.of("Benchmark")}
        edges = {}
        for b in self.of("Benchmark"):
            if id(b) not in self.local:
                continue
            for i, r in enumerate(b.spec.get("relations", [])):
                w, t, target = f"{b.where} relations[{i}]", r["type"], r["target"]
                if t == "compares_to":
                    self.errors.append(f"{w}: compares_to is not a relation: use an Analysis that varies attributes")
                elif "/" in t:
                    if t not in self.reg.relations:
                        self.errors.append(f"{w}: relation {t!r} is not registered by a loaded pack")
                elif t not in CORE_RELATIONS:
                    close = difflib.get_close_matches(t, CORE_RELATIONS, 1)
                    self.errors.append(
                        f"{w}: relation type {t!r} is unknown" + (f" (did you mean {close[0]!r}?)" if close else "")
                    )
                if r.get("via", {}).get("decompose") and t != "part_of":
                    self.errors.append(f"{w}: via.decompose applies to part_of only")
                if "/" in target or t in ("see_also",):
                    continue
                kind = "Asset" if t == "produces" else None
                if kind:
                    self.get(kind, target, w, "target")
                elif t == "derived_from" and ("Asset", target) in self.by:
                    pass
                elif target not in benches:
                    self.errors.append(f"{w}: target {target!r}: no Benchmark of that name")
                if t == "depends_on":
                    edges.setdefault(b.name, set()).add(target)
        for start in edges:
            seen, stack = set(), [start]
            while stack:
                n = stack.pop()
                for m in edges.get(n, ()):
                    if m == start:
                        self.errors.append(f"depends_on cycle through Benchmark/{start}")
                        stack = []
                        break
                    if m not in seen:
                        seen.add(m)
                        stack.append(m)

    def _check_analyses(self):
        benches = {d.name for d in self.of("Benchmark")}
        for d in self.of("Analysis"):
            sp, w = d.spec, d.where
            for n in sp["select"]["benchmarks"]:
                if n not in benches:
                    self.errors.append(f"{w}: select.benchmarks: {n!r}: no Benchmark of that name")
            m = self.reg.metrics.get(sp["metric"])
            if m is None:
                self.errors.append(f"{w}: metric {sp['metric']!r} is not registered by a loaded pack")
            lists = {k: sp.get(k, []) for k in ("hold", "relax", "vary")}
            seen = {}
            for k, pats in lists.items():
                for p in pats:
                    if not self.reg.matches_registry(p):
                        close = difflib.get_close_matches(p, [key_sample(x) for x in self.reg.attrs], 1)
                        self.errors.append(
                            f"{w}: {k}: {p!r} matches no registered attribute"
                            + (f" (did you mean {close[0]!r}?)" if close else "")
                        )
                    if p in seen and seen[p] != k:
                        self.errors.append(f"{w}: {p!r} is in both {seen[p]} and {k}")
                    seen[p] = k
            self._check_derive_compare(d)
            self.errors += quality.reference_errors(sp, {c["name"] for c in self.cases}, self.reg.metrics, w)
            dec = sp.get("decompose")
            if dec:
                linked = {
                    (b.name, r["target"])
                    for b in self.of("Benchmark")
                    for r in b.spec.get("relations", [])
                    if r["type"] == "part_of"
                }
                for n in [dec["from"]] + dec["to"]:
                    if n not in benches:
                        self.errors.append(f"{w}: decompose: {n!r}: no Benchmark of that name")
                for n in dec["to"]:
                    if (n, dec["from"]) not in linked and (dec["from"], n) not in linked:
                        self.errors.append(
                            f"{w}: decompose: {n} and {dec['from']} are not linked by a part_of relation"
                        )

    def _check_derive_compare(self, d):
        sp, w = d.spec, d.where
        dv, cmp_ = sp.get("derive"), sp.get("compare")
        names = {c["name"] for c in self.cases}
        if dv:
            if dv["name"] in self.reg.metrics:
                self.errors.append(f"{w}: derive.name {dv['name']!r} shadows a registered metric")
            for k in ("first", "second"):
                if dv["cases"][k] not in names:
                    self.errors.append(f"{w}: derive.cases.{k}: {dv['cases'][k]!r}: no case of that name")
            if dv["cases"]["first"] == dv["cases"]["second"]:
                self.errors.append(f"{w}: derive.cases: first and second must be different cases")
            x = dv.get("x")
            if (dv["op"] == "slope") != (x is not None):
                self.errors.append(f"{w}: derive.x is required for, and only valid with, op slope")
            elif x and x["first"] == x["second"]:
                self.errors.append(f"{w}: derive.x: first and second must differ")
        if cmp_:
            allowed = set(self.reg.metrics) | ({dv["name"]} if dv else set())
            if cmp_["metric"] not in allowed:
                self.errors.append(f"{w}: compare.metric {cmp_['metric']!r} is neither registered nor derive.name")
            for k in [cmp_["by"], *cmp_.get("across", [])]:
                if self.reg.lookup(k) is None:
                    self.errors.append(f"{w}: compare: {k!r} is not a registered attribute")
            if cmp_["by"] in cmp_.get("across", []):
                self.errors.append(f"{w}: compare.by must not also be in across")

    # ------------------------------------------------------------- validity policy and expectations
    def _check_policies(self):
        allp = self.of("ValidityPolicy")
        pols = []
        quals = []
        for d in allp:
            kinds = set(d.spec.get("applies_to") or ["perf"])
            if kinds == {"quality"}:
                quals.append(d)
            elif kinds == {"perf"}:
                pols.append(d)
            else:
                self.errors.append(f"{d.where}: applies_to must be [perf] or [quality], not a mixed list")
        want = self.project.spec.get("validity_policy")
        if want:
            self.policy = self.get("ValidityPolicy", want, self.project.where, "validity_policy")
            if self.policy is not None and id(self.policy) in {id(q) for q in quals}:
                self.errors.append(
                    f"{self.project.where}: validity_policy names a quality policy; it must be a perf one"
                )
                self.policy = None
        elif len(pols) == 1:
            self.policy = pols[0]
        elif len(pols) > 1:
            local = [d for d in pols if id(d) in self.local]
            if len(local) == 1:  # a catalog's own policy wins over a central one
                self.policy = local[0]
            else:
                self.errors.append(
                    f"{self.project.where}: {len(pols)} ValidityPolicies loaded; set Project.spec.validity_policy"
                )
        if len(quals) == 1:
            self.quality_policy = quals[0]
        elif len(quals) > 1:
            local = [d for d in quals if id(d) in self.local]
            if len(local) == 1:
                self.quality_policy = local[0]
            else:
                self.errors.append(f"{self.project.where}: {len(quals)} quality ValidityPolicies loaded; keep one")
        for d in quals:
            self.errors += quality.policy_errors(d, self.reg)
        for d in pols:
            for name, row in d.spec["purposes"].items():
                w = f"{d.where} purposes.{name}"
                if name not in self.reg.vocab["purposes"]:
                    self.errors.append(f"{w}: not a registered purpose {sorted(self.reg.vocab['purposes'])}")
                req = row["requires"]
                if "gates" in req:
                    self.errors.append(f"{w}: requires.gates belongs to a quality policy (applies_to: [quality])")
                if name == "pr" and (row["baseline"] or row["reference"]):
                    self.errors.append(f"{w}: a PR run can never be a baseline or reference")
                if row["reference"]:
                    if (
                        row["floor"] != "valid"
                        or req.get("qualification") != "current"
                        or req.get("clean", "any") == "any"
                    ):
                        self.errors.append(
                            f"{w}: a reference purpose needs floor valid, qualification current and a clean requirement"
                        )
                    if not row["baseline"]:
                        self.errors.append(f"{w}: a reference purpose must also allow baselines")
                if row["baseline"] and req.get("qualification") != "current":
                    self.errors.append(f"{w}: baselines need qualification current")
        names = {c["name"] for c in self.cases}
        for d in self.of("Expectation"):
            sp, w = d.spec, d.where
            m = self.reg.metrics.get(sp["metric"])
            if m is None:
                self.errors.append(f"{w}: metric {sp['metric']!r} is not registered by a loaded pack")
            elif m["direction"] == "none":
                self.errors.append(f"{w}: metric {sp['metric']} has no direction, so it cannot be judged")
            sel = sp["selector"]
            for n in sel.get("cases", []):
                if n not in names:
                    self.errors.append(f"{w}: selector.cases: {n!r} is not a case")
            for n in sel.get("benchmarks", []):
                if ("Benchmark", n) not in self.by:
                    self.errors.append(f"{w}: selector.benchmarks: {n!r}: no Benchmark of that name")
            hit = self.expectation_cases(d)
            if not hit:
                self.errors.append(f"{w}: selector selects no case")
            for c in hit:
                if m and sp["metric"] not in dedupe(sum(self.case_metrics(c), [])):
                    self.errors.append(f"{w}: metric {sp['metric']} is not recorded by case {c['name']}")
            b = sp["baseline"]
            if b["kind"] == "fixed" and "value" not in b:
                self.errors.append(f"{w}: a fixed baseline needs value")
            if b["kind"] == "rolling":
                if "window" not in b or not b.get("purposes"):
                    self.errors.append(f"{w}: a rolling baseline needs window and purposes")
                if b.get("min_n", 1) > b.get("window", 1):
                    self.errors.append(f"{w}: min_n exceeds window")
                pols = {id(p): p for p in (self.policy_of(c) for c in hit) if p} or (
                    {id(self.policy): self.policy} if self.policy else {}
                )
                for pu in b.get("purposes", []):
                    for pol in pols.values():
                        row = pol.spec["purposes"].get(pu)
                        if row is None or not row["baseline"]:
                            self.errors.append(
                                f"{w}: baseline purpose {pu!r} is not baseline-eligible in ValidityPolicy/{pol.name}"
                            )
            elif b.get("window") or b.get("purposes"):
                self.errors.append(f"{w}: window/purposes apply to a rolling baseline only")

    def expectation_cases(self, d):
        sel = d.spec["selector"]
        pool = [
            c
            for c in self.cases
            if (not sel.get("benchmarks") or c["benchmark"] in sel["benchmarks"])
            and (not sel.get("cases") or c["name"] in sel["cases"])
        ]
        terms = {k: v for k, v in sel.items() if k not in ("benchmarks", "cases")}
        return self.select(terms, pool) if terms else pool

    def expectations_for(self, c):
        return [d for d in self.of("Expectation") if any(x["name"] == c["name"] for x in self.expectation_cases(d))]

    def digest(self, kind, name):
        d = self.by.get((kind, name))
        if d is None:
            return None
        eff = self.effective(kind, name) if kind in ("System", "Workload", "Environment") else d.spec
        return hid("spec", normal_env(eff) if kind == "Environment" else eff)

    def registered_rows(self):
        """Rows written at ingest for authored definitions, keyed by name + spec_digest (old results stay interpretable)."""
        return [
            {"kind": k, "name": n, "spec_digest": self.digest(k, n)}
            for (k, n) in self.by
            if k in ("Environment", "NodeClass", "ValidityPolicy", "Expectation")
        ]

    def policy_of(self, c):
        """The ValidityPolicy that judges a case: the quality one for a quality case, else the perf one."""
        return self.quality_policy if c["type"] == "quality" else self.policy

    def validity_view(self, c, purpose, env=None):
        """What a case needs to earn each validity under the policy, and which expectations judge it."""
        out = {
            "expectations": [
                {
                    "name": d.name,
                    "metric": d.spec["metric"],
                    "baseline": d.spec["baseline"]["kind"],
                    "severity": d.spec.get("severity", "advisory"),
                    "spec_digest": self.digest("Expectation", d.name),
                }
                for d in self.expectations_for(c)
            ]
        }
        pol = self.policy_of(c)
        if pol and purpose in pol.spec["purposes"]:
            row = pol.spec["purposes"][purpose]
            out["policy"] = {
                "name": pol.name,
                "spec_digest": self.digest("ValidityPolicy", pol.name),
                "purpose": purpose,
                "floor": effective_floor(pol.spec, purpose, env),
                "requires": row["requires"],
                "baseline": row["baseline"],
                "reference": row["reference"],
            }
        return out

    # ------------------------------------------------------------- identity
    def identity(self, c):
        """The engine-native BenchmarkId inputs: component, name, identity tags, discriminators (+ unhashed props)."""
        i = self._identity(c)
        return {**i, "disc": {k: dstr(v) for k, v in i["disc"].items()}}

    def _identity(self, c):
        """identity() with the discriminator values as authored (lists stay lists)."""
        wl = c["workload"]
        t = self.reg.types.get(("Workload", wl.get("type")), {})
        body = {k: v for k, v in wl.items() if k not in ("base", "patch", "scenario", "inputs", "type")}
        if c["load"] is not None:
            body["load"] = c["load"]
        body = drop_paths(canon_workload(body), t.get("non_identity", []))
        disc = {"case.type": wl.get("type", ""), **flatten(body, "case.")}
        sys_ = c["sys"]
        st = self.reg.types.get(("System", sys_.get("type")), {})
        fields = drop_paths(typed_fields(sys_, "System"), st.get("non_identity", []))
        props = {"system.name": c["system"], "system.backend": sys_.get("backend", "")}
        for f_, kind in (st.get("refs") or {}).items():
            if f_ in fields:
                props[f"system.{f_}"] = fields.pop(f_)
                fields[f"{f_}_id"] = self.ref_id(kind, props[f"system.{f_}"])
        disc.update({"system.type": sys_.get("type", ""), **flatten(fields, "system.")})
        for p, blk in (sys_.get("extensions") or {}).items():
            if self.reg.ext.get(("System", p), {}).get("identity"):
                disc.update(flatten(blk, f"system.ext.{p}."))
            props.update({k: v for k, v in flatten(blk, f"system.ext.{p}.").items()})
        for inp in wl.get("inputs", []):
            n = inp["name"]
            ids, sel, props_in = self.input_identity(inp)
            disc[f"input.{n}.asset"] = ",".join(ids)
            disc[f"input.{n}.select"] = canonical(sel)
            props.update({f"input.{n}.{k}": v for k, v in props_in.items()})
        ident, ctx = split_tags(c["tags"])
        return {
            "component": self.component,
            "name": c["benchmark"],
            "tags": ident,
            "disc": disc,
            "props": {k: dstr(v) for k, v in props.items()},
            "run_context_tags": ctx,
        }

    def ref_id(self, kind, name):
        d = self.by.get((kind, name))
        if d is None:
            return ""
        if kind == "Asset":
            return asset_id(d.spec)
        pairs = (
            [(a.get("role", a.get("name", "data")), a["asset"]) for a in d.spec["assets"]]
            if "assets" in d.spec
            else [("weights", d.spec["asset"])]
        )
        return hid("model", sorted((r, self.ref_id("Asset", a)) for r, a in pairs))

    def dataset_slug(self, name):
        """The opaque slug a Dataset is stored under; its name when it declares none."""
        d = self.by.get(("Dataset", name))
        return d.spec.get("slug", name) if d is not None else name

    def dataset_asset_specs(self, name):
        d = self.by.get(("Dataset", name))
        if d is None:
            return []
        names = [a["asset"] for a in d.spec.get("assets", [])] + ([d.spec["asset"]] if "asset" in d.spec else [])
        return [self.by[("Asset", a)].spec for a in names if ("Asset", a) in self.by]

    def dataset_store(self, name):
        """The Store spec that holds a Dataset's `<slug>/dataset.json`: its `store`, else its first Asset's store."""
        d = self.by.get(("Dataset", name))
        if d is None:
            return None
        ref = d.spec.get("store")
        if ref is None:
            assets = [a["asset"] for a in d.spec.get("assets", [])] + ([d.spec["asset"]] if "asset" in d.spec else [])
            for a in assets:
                st = ((self.by.get(("Asset", a)) or d).spec.get("source") or {}).get("store")
                if st:
                    ref = st
                    break
        if isinstance(ref, str):
            ref = self.by.get(("Store", ref))
            return ref.spec if ref is not None else None
        return ref

    def input_identity(self, inp):
        """(asset ids, effective select, unhashed props) of one Workload input."""
        if "dataset" in inp:
            d = self.by.get(("Dataset", inp["dataset"]))
            sel = merge_patch(
                (d.spec.get("subsets", {}).get(inp.get("subset")) if d else None) or {}, inp.get("select") or {}
            )
            pairs = (
                [(a["name"], a["asset"]) for a in d.spec["assets"]]
                if d and "assets" in d.spec
                else [("data", d.spec["asset"])]
                if d
                else []
            )
            pick = [p for p in pairs if not sel.get("from") or p[0] == sel["from"]]
            ids = [self.ref_id("Asset", a) for _, a in pick]
            return ids, sel, {"dataset": inp["dataset"], "subset": inp.get("subset", "")}
        if "model" in inp:
            return [self.ref_id("Model", inp["model"])], inp.get("select") or {}, {"model": inp["model"]}
        return [self.ref_id("Asset", inp["asset"])], inp.get("select") or {}, {"asset_name": inp["asset"]}

    def ingest_inputs(self, c, errs=None):
        """The ingest derivation inputs of a case whose Benchmark declares `identity: {mode: ingest}`, else None.

        Needs no ingest package. Unresolvable `from` references are appended to `errs`, or raise ValueError.
        """
        blk = c.get("identity")
        if not blk or blk.get("mode") != "ingest":
            return None
        raw = self._identity(c)["disc"]
        bad = [] if errs is None else errs
        disc, keys = {}, []
        for n, d in enumerate(blk.get("discriminators", [])):
            keys.append(d.get("key"))
            if "from" not in d:
                disc[d.get("key")] = dstr(d.get("value"))
                continue
            at = f"{c['where']}: identity.discriminators[{n}] ({d.get('key')})"
            if d["from"] not in raw:
                bad.append(
                    f"{at}: {d['from']!r} is not an identity value of this case; unhashed run properties such as "
                    f"system.backend cannot feed an id (available: {', '.join(sorted(raw))})"
                )
                disc[d.get("key")] = ""
            elif "join" in d and not isinstance(raw[d["from"]], list):
                bad.append(f"{at}: join needs a list value, {d['from']!r} is not one")
                disc[d.get("key")] = ""
            elif isinstance(raw[d["from"]], (list, dict)) and "join" not in d:
                bad.append(
                    f"{at}: {d['from']!r} is a list or mapping; the ingest writes scalars, so a list needs "
                    "`join` and a mapping cannot feed a discriminator"
                )
                disc[d.get("key")] = ""
            else:
                v = raw[d["from"]]
                disc[d.get("key")] = d["join"].join(dstr(x) for x in v) if "join" in d else dstr(v)
        if errs is None and bad:
            raise ValueError("\n".join(bad))
        return {
            "component": blk.get("component", self.component),
            "name": blk.get("name", c["benchmark"]),
            "tags": list(blk.get("tags", [])),
            "disc": disc,
            "disc_keys": keys,
            "profile": blk.get("profile", DEFAULT_PROFILE),
        }

    def engine_id(self, c):
        """The engine-native hash of the case; needs no ingest package. Never a join key in ingest mode."""
        i = self.identity(c)
        return benchmark_id(i["component"], i["name"], i["tags"], i["disc"], sorted(i["disc"]))

    def ids(self, c):
        """`benchmark_id` is the stored join key. With `identity: {mode: ingest}` it comes from the ingest derivation
        (IngestUnavailable without it) and `engine_id` is kept beside it."""
        eid = self.engine_id(c)
        a = self.ingest_inputs(c)
        if a is None:
            return {"benchmark_id": eid}
        check_profile(a["profile"])
        bid = ingest_adapter.benchmark_id(a["component"], a["name"], a["tags"], a["disc"], a["disc_keys"])
        return {"benchmark_id": bid, "engine_id": eid, "id_mode": "ingest"}

    def id_token(self, c):
        """Equality token for duplicate and change detection that needs no ingest package: `benchmark_id` in engine
        mode, a hash of the normalised ingest inputs otherwise. Never stored."""
        a = self.ingest_inputs(c)
        if a is None:
            return self.ids(c)["benchmark_id"]
        parts = [norm(a["component"]), norm(a["name"]), tag_part(a["tags"]), disc_part(a["disc"], a["disc_keys"])]
        return hid("ingest-inputs", ID_SEP.join(parts))  # the string the ingest hashes, so its collisions collide

    def try_id(self, c):
        try:
            return self.ids(c)["benchmark_id"]
        except IngestUnavailable:
            return None

    # ------------------------------------------------------------- ad hoc variants
    def variant(self, case_name, sets):
        """An ephemeral case from `--set path=value` overrides; returns (case, identity diff). Never baseline-eligible."""
        base = next((c for c in self.cases if c["name"] == case_name), None)
        if base is None:
            raise KeyError(case_name)
        patch = copy.deepcopy(base["patch"])
        for s_ in sets:
            path, _, raw = s_.partition("=")
            patch = merge_patch(patch, nested(path, yaml.safe_load(raw)))
        b = self.by[("Benchmark", base["benchmark"])]
        new = self._build(b, case_name + "@adhoc", patch)
        if new is None:
            raise ValueError("\n".join(self.errors))
        saved, self.cases, self.errors = self.cases, [new], []
        try:
            self._check_cases()
            found = self.errors
        finally:
            self.cases, self.errors = saved, []
        if found:
            raise ValueError("\n".join(found))
        diff = identity_diff(self.identity(base), self.identity(new), self.try_id(base), self.try_id(new))
        diff["identity_changed"] = self.id_token(base) != self.id_token(new)
        return new, diff


def identity_diff(old, new, old_id="", new_id=""):
    keys = sorted(set(old["disc"]) | set(new["disc"]))
    rows = [(k, old["disc"].get(k), new["disc"].get(k)) for k in keys if old["disc"].get(k) != new["disc"].get(k)]
    pk = sorted(set(old["props"]) | set(new["props"]))
    props = [(k, old["props"].get(k), new["props"].get(k)) for k in pk if old["props"].get(k) != new["props"].get(k)]
    return {
        "benchmark_id": (old_id, new_id),
        "changed": rows,
        "props_changed": props,
        "identity_changed": old_id != new_id,
    }


def asset_id(spec):
    """Content address of an Asset: file digests, a pinned manifest, a pinned revision, a generator, or produced_by."""
    integ, src = spec.get("integrity") or {}, spec["source"]
    if spec["type"] == "generator":
        body = {"generator": src.get("generator")}
    elif spec["type"] == "generated":
        body = {
            "manifest": integ.get("manifest"),
            "inputs_digest": (spec.get("produced_by") or {}).get("inputs_digest"),
        }
    elif integ.get("files"):
        body = {"files": sorted((f["path"], f["sha256"]) for f in integ["files"])}
    elif integ.get("manifest"):
        body = {"manifest": integ["manifest"]["sha256"]}
    else:
        body = {k: src.get(k) for k in ("uri", "revision", "config", "split")}
    return hid("asset", body)


def dump_yaml_docs(docs):
    return "---\n".join(dump_yaml(d) for d in docs)


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def dump_yaml(doc):
    return yaml.dump(doc, Dumper=NoAliasDumper, sort_keys=False, width=120)


# ------------------------------------------------------------------ plan, reuse, invalidation, qualification
class PlanError(Exception):
    pass


def validity_at_least(have, floor):
    """A validity outside the vocabulary (stored data we cannot read) never satisfies a floor."""
    return have in VALIDITY_ORDER and VALIDITY_ORDER.index(have) >= VALIDITY_ORDER.index(floor)


def parse_ts(v):
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(timezone.utc)


def active_invalidations(events):
    """Events in force: invalid or suspect, minus those a later `restored` event points at."""
    restored = {e.get("restores") for e in events if e.get("state") == "restored"}
    return [e for e in events if e.get("state") in ("invalid", "suspect") and e.get("id") not in restored]


def check_invalidation(ev):
    """Writer-side validation of an invalidation row; the query scope is reserved."""
    errs = shape_schema("invalidation").errors(ev)
    if errs:
        raise ValueError("; ".join(errs))
    if ev["scope_kind"] == "query":
        raise ValueError("scope_kind query is reserved: the writer rejects it until enabled")
    need = {
        "run": ("run_id",),
        "run_benchmark": ("run_id", "benchmark_id"),
        "node": ("node", "from"),
        "device": ("device", "from"),
        "placement": ("placement_id",),
    }[ev["scope_kind"]]
    miss = [k for k in need if k not in ev]
    if miss:
        raise ValueError(f"scope_kind {ev['scope_kind']} needs {', '.join(miss)}")
    if ev["state"] == "restored" and "restores" not in ev:
        raise ValueError("a restored event names the invalidation it restores")
    return ev


def excluded_repeats(row, events):
    """Repeat numbers of a stored result that an active invalidation removes from reuse, baselines and verdicts."""
    out = set()
    for ev in active_invalidations(events):
        k = ev["scope_kind"]
        if k in ("run", "run_benchmark") and ev.get("run_id") != row.get("run_id"):
            continue
        if k == "run_benchmark" and ev.get("benchmark_id") != row.get("benchmark_id"):
            continue
        for r in row.get("repeats", []):
            n = r["run.repeat"]
            if k == "run" or (k == "run_benchmark" and ev.get("repeat") in (None, n)):
                out.add(n)
            elif k == "placement" and ev["placement_id"] == r.get("placement_id"):
                out.add(n)
            elif k in ("node", "device"):
                hit = ev["node"] in r.get("nodes", []) if k == "node" else ev["device"] in r.get("devices", [])
                ts = parse_ts(r.get("ts") or row["ts"])
                if hit and parse_ts(ev["from"]) <= ts <= (parse_ts(ev["to"]) if ev.get("to") else ts):
                    out.add(n)
    return out


def qualification_current(rows, node_class, node, now):
    """The latest passing qualification row for a node within the NodeClass expiry; results never expire."""
    q = node_class.get("qualification") or {}
    limit = q.get("cv_limit_pct")
    ok = [
        r
        for r in rows
        if r["node"] == node
        and r["outcome"] == "pass"
        and (limit is None or r.get("cv_pct", 0) <= limit)
        and parse_ts(r["qualified_at"]) <= now
    ]
    if not ok:
        return None
    last = max(ok, key=lambda r: parse_ts(r["qualified_at"]))
    return last if now - parse_ts(last["qualified_at"]) <= timedelta(days=q.get("expiry_days", 14)) else None


def run_clean(clean):
    """Cleanliness rule: exclusive == enforced, or observed with a clean load result AND devices_exclusive == enforced.
    none and unknown never qualify. Cleanliness is a run property and never enters env_id."""
    ex = clean.get("exclusive")
    if ex == "enforced":
        return True
    return ex == "observed" and clean.get("load_clean") is True and clean.get("devices_exclusive") == "enforced"


reference_grade_clean = run_clean


def repeat_validity(env, qualified, clean, gates_pass=True):
    """Reference grade = qualified environment AND clean run AND the purpose's gates; shared hosts give shared_env."""
    if not gates_pass:
        return "suspect"
    host = env.get("host") or {}
    if not run_clean(clean):
        return "shared_env"
    return "valid" if qualified or not host.get("reference_node") else "unqualified_env"


def effective_floor(policy, purpose, env=None):
    """Stricter of the purpose floor and the Environment's: the Environment may only tighten."""
    floors = []
    if policy and purpose in policy["purposes"]:
        floors.append(policy["purposes"][purpose]["floor"])
    if env and (env.get("policy") or {}).get("validity_floor"):
        floors.append(env["policy"]["validity_floor"])
    return max(floors or ["valid"], key=VALIDITY_ORDER.index)


def validity_of(policy, purpose, facts, trust="internal"):
    """(state, reasons): the worst validity any failed gate caps the result at. External trust never earns more than external.

    facts: measured, gates_pass, qualified, clean (run.clean.* map), repeats, cv_pct, warmup_discarded, placement,
    pinning_required, pinning_verified (see placement.validity_facts).
    """
    row = policy["purposes"][purpose]
    req, caps = row["requires"], []
    if not facts.get("measured", True):
        caps.append(("invalid", "not_measured"))
    if facts.get("gates_pass", True) is False:
        caps.append(("suspect", "gate_failed"))
    if req.get("qualification") == "current" and not facts.get("qualified"):
        caps.append(("unqualified_env", "qualification_not_current"))
    need = req.get("clean", "any")
    clean = facts.get("clean") or {}
    if (need == "enforced" and clean.get("exclusive") != "enforced") or (need == "clean" and not run_clean(clean)):
        caps.append(("shared_env", f"clean_{need}_not_met"))
    if facts.get("repeats", 0) < req.get("min_repeats", 1):
        caps.append(("suspect", "repeats_below_min"))
    if "max_cv_pct" in req and (facts.get("cv_pct") is None or facts["cv_pct"] > req["max_cv_pct"]):
        caps.append(("suspect", "cv_above_max"))
    if facts.get("warmup_discarded", 0) < req.get("discard_warmup", 0):
        caps.append(("suspect", "warmup_not_discarded"))
    if req.get("placement") == "required" and not facts.get("placement"):
        caps.append(("suspect", "placement_missing"))
    if cap := pinning_cap(facts):
        caps.append(cap)
    if trust == "external":
        caps.append(("external", "trust_external"))
    if not caps:
        return "valid", []
    return min(caps, key=lambda c: VALIDITY_ORDER.index(c[0]))[0], [r for _, r in caps]


def reference_grade(policy, purpose, state, trust="internal"):
    """Reference grade needs a valid result of a reference purpose from internal trust: external never earns it."""
    return trust != "external" and state == "valid" and bool(policy["purposes"][purpose]["reference"])


def baseline_eligible(policy, row, events=()):
    """Only VALID, internal results of a baseline purpose count; invalid, suspect, external and PR runs never do."""
    pu = policy["purposes"].get(row.get("purpose"))
    return (
        bool(pu and pu["baseline"])
        and row.get("validity") == "valid"
        and row.get("trust", "internal") != "external"
        and not (row.get("repeats") and len(excluded_repeats(row, events)) >= len(row["repeats"]))
    )


def select_baseline(policy, exp, rows, current, events=()):
    """The prior runs a rolling Expectation compares against: same benchmark, environment and pinning policy, eligible,
    purpose-filtered, newest first."""
    b = exp["baseline"]
    env = b.get("match", {}).get("environment", "id")
    out = []
    for r in sorted(rows, key=lambda r: str(r["ts"]), reverse=True):
        if (
            r["benchmark_id"] != current["benchmark_id"]
            or r["run_id"] == current["run_id"]
            or str(r["ts"]) >= str(current["ts"])
        ):
            continue
        if r.get("purpose") not in b.get("purposes", []) or not baseline_eligible(policy, r, events):
            continue
        if env == "id":
            if not current.get("env_id") or r.get("env_id") != current["env_id"]:
                continue
            if pin_label(r.get("labels")) != pin_label(current.get("labels")):
                continue
        elif not labels_match(r.get("labels", {}), current.get("labels", {}), env["labels"]):
            continue
        out.append(r)
    return out[: b.get("window", 20)]


def judge(
    exp,
    expectation_digest,
    validity_digest,
    direction,
    current,
    baseline,
    validity="valid",
    floor="valid",
    noise_cv_pct=None,
    reasons=(),
):
    """Verdict for one metric. current: this run's per-repeat values; baseline: one value per prior valid run (rolling) or ignored (fixed).

    Maps onto benchmark_metric_verdicts (baseline_kind, n_baseline, n_current, baseline_value, current_value, delta_pct, worse_pct,
    floor_pct, status, reason, policy_version); pass is the table's `unchanged`. A single run is capped at suspected.
    reasons: the validity reasons; a placement reason (PINNING_REASONS) always gives suspected, whatever the floor.
    """
    b, sev = exp["baseline"], exp.get("severity", "advisory")
    cur = statistics.median(current)
    rec = {
        "baseline_kind": b["kind"],
        "n_baseline": 0,
        "n_current": len(current),
        "baseline_value": None,
        "current_value": cur,
        "delta_pct": None,
        "worse_pct": None,
        "floor_pct": None,
        "severity": sev,
        "expectation_digest": expectation_digest,
        "validity_digest": validity_digest,
        "policy_version": f"{expectation_digest[:12]}.{validity_digest[:12]}",
    }
    if not validity_at_least(validity, floor):
        return {**rec, "status": "suspected", "reason": f"validity_{validity}_below_{floor}"}
    placement_reason = next((r for r in reasons if r in PINNING_REASONS), None)
    if placement_reason:
        return {**rec, "status": "suspected", "reason": placement_reason}
    if b["kind"] == "fixed":
        base, n_base = b["value"], 1
    else:
        n_base = len(baseline)
        if n_base < b.get("min_n", 1):
            return {**rec, "n_baseline": n_base, "status": "no_baseline", "reason": "baseline_too_short"}
        base = statistics.median(baseline)
    noise = exp.get("noise") or {}
    sigma = (
        (1.4826 * statistics.median(abs(x - base) for x in baseline) / abs(base) * 100)
        if b["kind"] == "rolling" and base
        else 0.0
    )
    if noise.get("source") in ("series_cv", "qualification"):
        sigma = noise_cv_pct or 0.0
    floor_pct = max(exp["tolerance"]["pct"], noise.get("floor_pct", 0.0), noise.get("sigmas", 2.0) * sigma)
    delta = (cur - base) / abs(base) * 100
    worse = delta if direction == "lower_is_better" else -delta
    rec.update(
        n_baseline=n_base,
        baseline_value=base,
        delta_pct=round(delta, 4),
        worse_pct=round(worse, 4),
        floor_pct=round(floor_pct, 4),
    )
    if worse > floor_pct:
        return {
            **rec,
            **(
                {"status": "regressed", "reason": ""}
                if len(current) >= 2
                else {"status": "suspected", "reason": "single_run"}
            ),
        }
    if worse < -floor_pct:
        return {
            **rec,
            **(
                {"status": "improved", "reason": ""}
                if len(current) >= 2
                else {"status": "pass", "reason": "single_run"}
            ),
        }
    return {**rec, "status": "pass", "reason": ""}


def pin_block(env):
    """The well-formed host.cpu_pinning block of an Environment spec, or None (schema errors are reported elsewhere)."""
    host = (env or {}).get("host")
    pin = host.get("cpu_pinning") if isinstance(host, dict) else None
    return pin if isinstance(pin, dict) and isinstance(pin.get("policy"), str) else None


def pin_problems(pin, accelerators):
    """What is wrong with a host.cpu_pinning block, as messages (empty when it is consistent)."""
    out, policy = [], pin["policy"]
    if (policy == "explicit") != ("cpus" in pin):
        out.append("host.cpu_pinning: cpus is required for, and only valid with, policy explicit")
    if pin.get("require_verified") and policy == "none":
        out.append("host.cpu_pinning: require_verified needs a policy other than none")
    if pin.get("require_verified") and policy == "platform":
        out.append(
            "host.cpu_pinning: require_verified cannot be met by policy platform (the engine does not verify it)"
        )
    for key in ("cpus_by_device", "assign", "layout"):
        if key in pin and policy != "per_device":
            out.append(f"host.cpu_pinning: {key} is only valid with policy per_device")
    if "launcher" in pin and policy in ("none", "platform"):
        out.append(f"host.cpu_pinning: launcher is not valid with policy {policy}")
    if "cpus_by_device" in pin and "assign" in pin:
        out.append("host.cpu_pinning: give cpus_by_device or assign, not both")
    if pin.get("layout") == "ranks" and not ("cpus_by_device" in pin or "assign" in pin):
        out.append("host.cpu_pinning: layout ranks needs cpus_by_device or assign")
    by = pin.get("cpus_by_device")
    if by is not None and not (isinstance(by, list) and all(isinstance(x, str) for x in by)):
        out.append("host.cpu_pinning.cpus_by_device: a list of CPU list strings")
        by = None
    for key, lists in (("cpus", [pin["cpus"]] if "cpus" in pin else []), ("cpus_by_device", by)):
        for x in lists or []:
            try:
                parse_cpu_list(x)
            except ValueError as e:
                out.append(f"host.cpu_pinning.{key}: {e}")
    if "assign" in pin:
        try:
            pinning_plan.Policy.from_dict(pin["assign"])
        except (pinning_plan.PinningError, TypeError) as e:
            out.append(f"host.cpu_pinning.assign: {e}")
    if policy in ("device_local", "per_device") and not accelerators:
        out.append(f"host.cpu_pinning: policy {policy} needs an accelerator request")
    return out


def overlay_pinning(env, pin):
    """The Environment with host.cpu_pinning replaced by a request-level block (None leaves it as is)."""
    if not pin:
        return env
    return {**env, "host": {**(env.get("host") or {}), "cpu_pinning": pin}}


def request_digest(cat, name, pin):
    """spec_digest of an Environment as requested: the authored one with a request-level pinning block laid over it."""
    env = cat.effective("Environment", name)
    return hid("spec", normal_env(overlay_pinning(env, pin))) if env is not None else None


def normal_env(env):
    """An explicit cpu_pinning policy none (not required verified) means the same request as no block: both digest alike."""
    pin = pin_block(env)
    if (
        not pin
        or pin["policy"] != "none"
        or pin.get("require_verified")
        or not set(pin) <= {"policy", "require_verified"}
    ):
        return env
    rest = {k: v for k, v in env["host"].items() if k != "cpu_pinning"}
    return {**{k: v for k, v in env.items() if k != "host"}, **({"host": rest} if rest else {})}


def pin_label(labels):
    """The pinning policy of a label map; a map without the label is an unpinned one."""
    return (labels or {}).get(PIN_LABEL, "none")


def labels_match(have, want, patterns):
    """Same pinning policy and equal values for every label key matching a pattern (at least one key must match)."""
    if pin_label(have) != pin_label(want):
        return False
    keys = {k for k in set(have) | set(want) if any(fnmatch.fnmatch(k, p) for p in patterns)}
    if not keys:
        return False
    return all(
        (pin_label(have) if k == PIN_LABEL else have.get(k)) == (pin_label(want) if k == PIN_LABEL else want.get(k))
        for k in keys
    )


def reuse_lookup(rows, events, want, now=None):
    """(row, why): the stored result that satisfies a request, or None and the reason it was rejected.

    want: benchmark_id, subject_id, env_id, labels, repeats, floor, environment (id | {labels: [...]}), max_age, prefer.
    """
    why = "no_result_for_case"
    for row in sorted(rows, key=lambda r: str(r.get("ts")), reverse=True):
        if row["benchmark_id"] != want["benchmark_id"]:
            continue
        why = "different_subject"
        if row["subject_id"] != want["subject_id"]:
            continue
        why = "different_environment"
        env = want["environment"]
        if env == "id":
            if not want.get("env_id") or row.get("env_id") != want["env_id"]:
                continue
            if pin_label(row.get("labels")) != pin_label(want["labels"]):
                continue  # same hardware, different pinning policy: never a silent reuse
        elif not labels_match(row.get("labels", {}), want["labels"], env["labels"]):
            continue
        why = "validity_below_request"
        if not validity_at_least(row["validity"], want["floor"]):
            continue
        why = "repeats_below_request"
        n = len(row.get("repeats", [])) - len(excluded_repeats(row, events)) if row.get("repeats") else row.get("n", 1)
        if n < (1 if want.get("prefer") else want["repeats"]):
            continue
        why = "too_old"
        if want.get("max_age") and now and now - parse_ts(row["ts"]) > timedelta(seconds=want["max_age"]):
            continue
        return row, "same_subject+same_environment"
    return None, why


def plan(cat, request, leg, existing=(), invalidations=(), now=None):
    """Request + leg facts + stored results -> plan (cases with action run | reuse | skip | missing and a reason)."""
    if cat.errors:
        raise PlanError("\n".join(cat.errors))
    now = now or datetime.now(timezone.utc)
    pol = cat.policy.spec if cat.policy else None
    known = set((pol or {}).get("purposes", {})) | set(
        (cat.quality_policy.spec["purposes"] if cat.quality_policy else {})
    )
    if (pol or cat.quality_policy) and request.get("purpose") and request["purpose"] not in known:
        raise PlanError(
            f"purpose {request['purpose']!r} is not in ValidityPolicy/{(cat.policy or cat.quality_policy).name} ({sorted(known)})"
        )
    sel, ov = request.get("selection") or {}, request.get("overrides") or {}
    sels = {d.name: d for d in cat.docs if d.kind == "Selection"}
    chosen, sel_of = [], {}
    # Named by the request, or by a Selection stating its own `types`: the default type must not drop these.
    asked_for = set()
    if sel.get("cases"):
        for n in sel["cases"]:
            if n in sels:
                s = sels[n]
                found = (
                    [x for x in cat.cases if x["name"] in s.spec["cases"]]
                    if "cases" in s.spec
                    else cat.select(s.spec["selector"])
                )
                declares_type = bool((s.spec.get("selector") or {}).get("types"))
                for c in found:
                    chosen.append(c)
                    sel_of[c["name"]] = s
                    if declares_type:
                        asked_for.add(c["name"])
            else:
                hit = [c for c in cat.cases if c["name"] == n]
                if not hit:
                    raise PlanError(f"unknown case or selection {n!r}")
                chosen += hit
                asked_for.update(c["name"] for c in hit)
    else:
        groupings = dict(sel.get("groupings") or {})
        if "tier" in sel:
            groupings["tier"] = {"at_most": sel["tier"]}
        elif "tier" not in groupings and "tier" in cat.groupings:
            groupings["tier"] = {"at_most": "core"}
        t = (groupings.get("tier") or {}).get("at_most")
        if t is not None and t not in [gstr(v) for v in cat.groupings["tier"].get("values", [])]:
            raise PlanError(f"tier {t!r} not in {cat.groupings['tier'].get('values')}")
        chosen = cat.select(
            {"suites": sel.get("suites") or cat.project.spec.get("default_suites", []), "groupings": groupings}
        )
    # An explicit type narrows everything; the `perf` default only filters what nothing asked for.
    ttype, explicit = sel.get("type", "perf"), "type" in sel
    chosen = [
        c
        for c in dict((c["name"], c) for c in chosen).values()
        if c["type"] == ttype or (not explicit and c["name"] in asked_for)
    ]
    records = []
    for c in chosen:
        env_name = request.get("environment") or c["environment"]
        env = cat.effective("Environment", env_name) if env_name else None
        if env is None and request.get("cpu_pinning"):
            raise PlanError(f"{c['name']}: a cpu_pinning request needs the case to have an Environment")
        if env is not None and request.get("cpu_pinning"):
            env = overlay_pinning(env, request["cpu_pinning"])
            if problems := pin_problems(request["cpu_pinning"], cat.accelerator_count(env)):
                raise PlanError(f"{c['name']}: {'; '.join(problems)}")
        if env_name and env is not None and (env.get("host") or {}).get("arch") not in (None, leg["arch"]):
            continue
        records.append(
            _plan_case(cat, c, env_name, env, request, ov, sel_of.get(c["name"]), leg, existing, invalidations, now)
        )
    if not records:
        raise PlanError("selection is empty")
    est = sum(r["est_min"] * (2 if r["cache"] == "both" else 1) for r in records if r["action"] == "run")
    acc_h = round(
        sum(
            r["est_min"] * (2 if r["cache"] == "both" else 1) * r["accelerators"] / 60
            for r in records
            if r["action"] == "run"
        ),
        2,
    )
    budget = (request.get("budget") or {}).get("device_hours")
    if budget is not None and acc_h > budget:
        raise PlanError(f"plan needs {acc_h} accelerator-hours > budget {budget}")
    return {
        "purpose": request.get("purpose"),
        "leg": leg,
        "cases": records,
        "estimate": {"wall_min": est, "device_hours": acc_h},
    }


def _plan_case(cat, c, env_name, env, request, ov, s, leg, existing, invalidations, now):
    pol = cat.policy_of(c)
    if pol and request.get("purpose") and request["purpose"] not in pol.spec["purposes"]:
        raise PlanError(
            f"purpose {request['purpose']!r} is not in ValidityPolicy/{pol.name} ({sorted(pol.spec['purposes'])}), "
            f"which judges case {c['name']}"
        )
    ids = cat.ids(c)
    t = cat.reg.types.get(("Workload", c["workload"]["type"]), {})
    required, _ = cat.case_metrics(c, t)
    mins = [cat.reg.metrics[m].get("noise", {}).get("min_repeats", 1) for m in required if m in cat.reg.metrics]
    repeats = int(
        ov.get("repeats")
        or (s and s.spec.get("run", {}).get("repeats_min"))
        or c["run"].get("repeats")
        or max(mins or [1])
    )
    prof = c["profile"]
    level = ov.get("profile_level") or (s and s.spec.get("run", {}).get("profile_level")) or prof.get("default", "off")
    labels = cat.env_labels(env) if env else {}
    series = {k: labels[k] for k in cat.project.spec.get("series_labels", []) if k in labels}
    need = cat.device_totals(env) if env else {}
    reason = ""
    if env is None:
        reason = "no_environment"
    elif any(n > leg.get("devices", {}).get(k, 0) for k, n in need.items()) or env.get("nodes", 1) > leg.get(
        "nodes", 1
    ):
        reason = "capacity"
    elif ((env.get("host") or {}).get("reference_node") or c["requires_reference"]) and not leg.get("reference_node"):
        reason = "reference_node_required"
    elif prof.get("supported") and level not in prof["supported"]:
        reason = "profile_unsupported"
    clean = ((env or {}).get("host") or {}).get("clean")
    if clean and not reason:
        access = leg.get("clean")
        hold = repeats * (seconds(c["run"].get("timeout", "60m")) or 0)
        if not access:
            reason = "capacity"  # no exclusive access available: never run unclean while claiming clean
        elif access.get("expected_wait_s", 0) > seconds(clean["max_wait"]):
            reason = "capacity_timeout"
        elif hold > seconds(clean["max_hold"]):
            reason = "max_hold_too_short"
    rec = {
        "name": c["name"],
        **ids,
        "series_key": series_key(ids["benchmark_id"], series) if series else None,
        "run_props": {
            **cat.run_props(c),
            **({"run.purpose": request["purpose"]} if request.get("purpose") else {}),
            **(
                {"run.validity_policy": cat.digest("ValidityPolicy", cat.policy_of(c).name)} if cat.policy_of(c) else {}
            ),
            "run.profile_level": level,
        },
        "action": "skip" if reason else "run",
        "reason": reason,
        "reused_from": None,
        "type": c["type"],
        "level": c["level"],
        "workload_type": c["workload"]["type"],
        "environment": env_name,
        "runner": c["run"].get("runner") or t.get("runner"),
        "repeats": repeats,
        "cache": ov.get("cache") or c["run"].get("cache", "warm"),
        "profile_level": level,
        "accelerators": cat.accelerator_count(env) if env else 0,
        "timeout_min": (seconds(c["run"].get("timeout", "60m")) or 0) / 60,
        "est_min": (seconds(c["run"].get("est", "0m")) or 0) / 60,
        "case_props": cat.identity(c)["props"],
        "validity": cat.validity_view(c, request.get("purpose"), env),
        "clean": {"requested": bool(clean), "expected_exclusive": "enforced" if clean else "observed"},
        "inputs": [_plan_input(cat, i, request) for i in c["workload"].get("inputs", [])],
        **({"selection": s.name} if s else {}),
        **({"expand": c["workload"]["expand"]} if c["workload"].get("expand") else {}),
    }
    asked = request.get("reuse") or None
    cfg = (asked if asked and asked.get("override") else None) or (s.spec.get("reuse") if s else None) or asked
    cfg = cfg or {"mode": "never"}
    subject = request.get("subject") or {}
    forced = {"all", c["name"]} & set(cfg.get("force", []))
    if not reason and cfg["mode"] != "never":
        if forced:
            rec["reason"] = "forced"
        elif not subject.get("artifact_id"):
            rec["reason"] = "reuse_needs_subject"
        else:
            match = cfg.get("match", {})
            want = {
                "benchmark_id": ids["benchmark_id"],
                "subject_id": subject["artifact_id"],
                "env_id": leg.get("env_id"),
                "labels": labels,
                "repeats": repeats,
                "environment": match.get("environment", "id"),
                "floor": effective_floor(
                    cat.policy_of(c).spec if cat.policy_of(c) else None, request.get("purpose"), env
                ),
                "max_age": max_age_seconds(cfg["max_age"]) if cfg.get("max_age") is not None else None,
                "prefer": cfg["mode"] == "prefer",
            }
            row, why = reuse_lookup(existing, invalidations, want, now)
            rec["reuse_key"] = {k: want[k] for k in ("benchmark_id", "subject_id", "env_id", "environment")}
            if row:
                rec.update(action="reuse", reason=why, reused_from=row["run_id"])
            elif cfg["mode"] == "only":
                rec.update(action="missing", reason=why)
            else:
                rec["reason"] = why
    return rec


def _plan_input(cat, inp, request):
    ids, sel, _ = cat.input_identity(inp)
    policy = "always" if request.get("strict") else inp.get("digest", "on_expect")
    return {
        "name": inp["name"],
        "asset_ids": ids,
        "select": sel,
        "expect": inp.get("expect", {}),
        "selection_digest": "required"
        if policy == "always" or (policy == "on_expect" and inp.get("expect"))
        else "off",
    }


# ------------------------------------------------------------------ result shape, aggregate, recompute
def series_key(bid, labels):
    return hid("series", {"benchmark_id": bid, "labels": {k: gstr(v) for k, v in sorted(labels.items())}})


def env_id_of(fingerprint, version=1):
    """Only the attributes hashed under `version` count: later attributes are recorded but leave older ids unchanged."""
    return hid(
        "env",
        {"v": version, "fp": {k: dstr(v) for k, v in sorted(fingerprint.items()) if version >= FP_SINCE.get(k, 1)}},
    )


def placement_id(p):
    return hid(
        "placement",
        {
            "nodes": sorted(p.get("nodes", [])),
            "devices": sorted((d.get("serial", ""), d.get("slot", "")) for d in p.get("devices", [])),
            "links": sorted(p.get("links", [])),
        },
    )


def per_repeat_values(ok, statistic="median"):
    """{metric: one value per repeat}; a metric seen in warm repeats never mixes in its cold ones (cache: both)."""
    by = {}
    for r in ok:
        cold = (r.get("props") or {}).get("run.cache") == "cold"
        for m, samples in r["samples"].items():
            by.setdefault(m, {True: [], False: []})[cold].append(
                statistics.median(samples) if statistic == "median" else statistics.fmean(samples)
            )
    return {m: g[False] or g[True] for m, g in by.items()}


def aggregate(repeats, units, statistic="median", outliers="none"):
    """Derived from the valid repeats and recorded with its spec: one value per repeat, then the statistic across repeats."""
    ok = [r for r in repeats if r.get("status", "ok") == "ok"]
    out = {}
    for m, vals in per_repeat_values(ok, statistic).items():
        if outliers == "mad" and len(vals) > 2:
            med = statistics.median(vals)
            mad = statistics.median(abs(v - med) for v in vals) or 1e-12
            vals = [v for v in vals if abs(v - med) <= 3 * 1.4826 * mad]
        out[m] = {
            "value": statistics.median(vals) if statistic == "median" else statistics.fmean(vals),
            "unit": units.get(m, "1"),
        }
    return {"statistic": statistic, "outliers": outliers, "n": len(ok), "metrics": out}


def build_result(
    cat,
    case,
    name,
    subject,
    fingerprint,
    observations,
    labels,
    repeats,
    statistic="median",
    outliers="none",
    reused_from=None,
    request=None,
    fingerprint_version=None,
    validity=None,
    pinning=None,
):
    """TOOL-WRITTEN Result (shape only; rows are stored by the tool). Every id recomputes from it (recompute()).

    pinning: the host.cpu_pinning request that gates validity; taken from the case's Environment when omitted.
    """
    fingerprint_version = fingerprint_version or cat.project.spec.get("fingerprint_version", 1)
    if pinning is None and case.get("environment"):
        pinning = pin_block(cat.effective("Environment", case["environment"]))
    bad = sorted(set(fingerprint) - cat.reg.fingerprint_keys())
    if bad:
        raise ValueError(f"not fingerprint attributes in the registry: {bad}")
    ident = cat.identity(case)
    ids = cat.ids(case)
    bid = ids["benchmark_id"]
    ingest = cat.ingest_inputs(case)
    case_block = {
        "benchmark_id": bid,
        "component": ident["component"],
        "name": ident["name"],
        "tags": ident["tags"],
        "disc": ident["disc"],
        "props": ident["props"],
        **({"case": case["name"]} if case["type"] == "quality" else {}),
    }
    if ingest:
        case_block = {
            **case_block,
            **{k: ingest[k] for k in ("component", "name", "tags", "disc", "disc_keys")},
            "id_mode": "ingest",
            "engine_id": ids["engine_id"],
            "benchmark": case["benchmark"],
            "case": case["name"],
        }
    labels = {**({PIN_LABEL: "none"} if PIN_LABEL in cat.project.spec.get("series_labels", []) else {}), **labels}
    series = {k: labels[k] for k in cat.project.spec.get("series_labels", []) if k in labels}
    reps = []
    for r in repeats:
        pl = {**r["placement"], "id": placement_id(r["placement"])}
        reps.append(
            {
                "run.repeat": r["run.repeat"],
                "status": r.get("status", "ok"),
                "placement": pl,
                "clean": r.get("clean", {}),
                **({"props": r["props"]} if r.get("props") else {}),
                "samples": r["samples"],
            }
        )
    units = {m: cat.reg.metrics[m]["unit"] for r in reps for m in r["samples"] if m in cat.reg.metrics}
    return {
        "apiVersion": API,
        "kind": "Result",
        "metadata": {"name": name},
        "status": {
            "case": case_block,
            "subject": subject,
            "environment": {
                "env_id": env_id_of(fingerprint, fingerprint_version),
                "fingerprint_version": fingerprint_version,
                "fingerprint": fingerprint,
                "observations": observations,
                "labels": labels,
                **({"request": request} if request else {}),
            },
            "series": {"key": series_key(bid, series), "labels": series},
            "repeats": reps,
            "aggregate": aggregate(reps, units, statistic, outliers),
            "reused_from": reused_from,
            **_validity_block(
                cat,
                validity if case["type"] == "quality" else pinning_capped(validity, pinning, reps),
                cat.policy_of(case),
            ),
        },
    }


PINNING_REASONS = ("pinning_violated", "pinning_unverified")


def pinning_cap(facts):
    """(state, reason) a required-but-unverified placement caps the result at, or None."""
    if not facts.get("pinning_required"):
        return None
    seen = facts.get("pinning_verified", "unknown")
    if seen == "true":
        return None
    return "suspect", "pinning_violated" if seen == "false" else "pinning_unverified"


def pinning_capped(validity, pinning, repeats):
    """validity with the placement cap applied from the repeats' run.exec.* props; unchanged without a request."""
    cap = pinning_cap(placement.validity_facts(pinning, [r.get("props") for r in repeats]))
    if not validity or not cap:
        return validity
    state, reasons, purpose, trust = validity
    return (
        min(state, cap[0], key=VALIDITY_ORDER.index),
        [*reasons, *([cap[1]] if cap[1] not in reasons else [])],
        purpose,
        trust,
    )


def _validity_block(cat, validity, policy=None):
    """validity = (state, reasons, purpose, trust) from validity_of(); stored as run.validity / run.validity_policy."""
    policy = policy or cat.policy
    if not validity or not policy:
        return {}
    state, reasons, purpose, trust = validity
    dg = cat.digest("ValidityPolicy", policy.name)
    return {
        "validity": {"state": state, "reasons": reasons, "policy_digest": dg, "purpose": purpose},
        "run_props": {"run.validity": state, "run.validity_policy": dg, "run.purpose": purpose, "run.trust": trust},
    }


def recompute(result, reg=None):
    """Mismatches between stored and recomputed ids; uses nothing but the Result. With a registry it also checks the
    case props: unhashed system.* props must be registered, in range, and never repeated among the hashed disc."""
    st, bad = result["status"], []
    c = st["case"]
    for k, v in (c.get("props") or {}).items():
        if k in c["disc"] and c.get("id_mode") != "ingest":  # an ingest disc is the producer's namespace, not ours
            bad.append(f"props overlap {k}")
        elif reg is not None and k.startswith("system.ext."):
            a = reg.lookup(k)
            if a is None or (a.get("values") and v not in [dstr(x) for x in a["values"]]):
                bad.append(f"props {k}")
    if c.get("id_mode") == "ingest":  # IngestUnavailable propagates: an ingest id is never silently unchecked
        keys = c.get("disc_keys")
        again = (
            None if keys is None else ingest_adapter.benchmark_id(c["component"], c["name"], c["tags"], c["disc"], keys)
        )
    else:
        again = benchmark_id(c["component"], c["name"], c["tags"], c["disc"], sorted(c["disc"]))
    if again != c["benchmark_id"]:
        bad.append("benchmark_id")
    env = st["environment"]
    if env_id_of(env["fingerprint"], env.get("fingerprint_version", 1)) != env["env_id"]:
        bad.append("env_id")
    for r in st["repeats"]:
        if placement_id(r["placement"]) != r["placement"]["id"]:
            bad.append(f"placement repeat {r['run.repeat']}")
    if series_key(c["benchmark_id"], st["series"]["labels"]) != st["series"]["key"]:
        bad.append("series")
    return bad


def check_aggregate(result, units=None):
    """Recompute the aggregate from the repeats and report differing metrics (the aggregate is derived, never a fact)."""
    st = result["status"]
    a = st["aggregate"]
    units = units or {m: v.get("unit", "1") for m, v in a.get("metrics", {}).items()}
    fresh = aggregate(st["repeats"], units, a["statistic"], a.get("outliers", "none"))
    return [
        m
        for m in fresh["metrics"]
        if abs(fresh["metrics"][m]["value"] - a["metrics"].get(m, {}).get("value", math.nan)) > 1e-12
    ] + (["n"] if fresh["n"] != a["n"] else [])
