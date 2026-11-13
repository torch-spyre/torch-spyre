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

"""Generate the core JSON Schemas (Draft 2020-12) for bench/v1alpha1 from one source.

core/schemas/<kind>.schema.json: one self-contained schema per authored kind, inside the stdlib-validator keyword subset.
Typed specs (System, Workload, Store, Benchmark.workload) leave the type's own fields open here; the engine validates
them against the registered type schema. core/schemas/resources.schema.json dispatches on `kind` (editor use).
core/schemas/tool-output/*.shape.json describe rows the tool writes (Result, invalidation); never authored.
  spyre-bench gen-schemas [--check]
"""

import argparse
import json
import sys
from pathlib import Path

TOOL = "spyre-bench"  # CLI; the Python package is spyre_bench
ENGINE = "bench"  # apiVersion group, stays bench/v1alpha1
API = f"{ENGINE}/v1alpha1"
ID_BASE = f"https://schemas.{ENGINE}.invalid/v1alpha1"
OUT = Path(__file__).resolve().parent / "core" / "schemas"
SOURCE_CHECKOUT = (
    Path(__file__).resolve().parent.parent / "pyproject.toml"
).is_file()  # installed packages are not rewritten
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

NAME = r"^[a-z0-9]([a-z0-9._@-]{0,126}[a-z0-9])?$"
CASE_NAME = r"^[a-z0-9][a-z0-9._=/,@-]{0,199}$"
TYPE_REF = r"^[a-z][a-z0-9]*/[a-z][a-z0-9-]*$"
PREFIX = r"^[a-z][a-z0-9]{0,15}$"
ATTR = r"^[a-z][a-z0-9_]*(\.[a-z0-9_<>-]+)+$"
ATTR_GLOB = r"^[a-z][a-z0-9_]*(\.[a-z0-9_<>*-]+)*(\.\*)?$"
METRIC = r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$"
SEMVER = r"^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$"
SHA256 = r"^[0-9a-f]{64}$"
DUR = r"^[0-9]+(\.[0-9]+)?(s|m|h|d)$"
GROUPING = r"^[a-z][a-z0-9_]*$"


def s(**kw):
    return {"type": "string", **kw}


def obj(props, required=(), desc=None, extra=False):
    o = {"type": "object", "additionalProperties": extra, "properties": props}
    if required:
        o["required"] = list(required)
    if desc:
        o["description"] = desc
    return o


def arr(items, **kw):
    return {"type": "array", "items": items, **kw}


def ref(name):
    return {"$ref": f"#/$defs/{name}"}


def num(**kw):
    return {"type": "number", **kw}


def integer(**kw):
    return {"type": "integer", **kw}


def any_of(*forms):
    return {"anyOf": list(forms)}


SCALAR = any_of({"type": "string"}, {"type": "number"}, {"type": "boolean"})
BOOL = {"type": "boolean"}
OPS = {k: SCALAR for k in ("eq", "ne", "gt", "gte", "lt", "lte")}

DEFS = {
    "name": s(pattern=NAME),
    "case_name": s(pattern=CASE_NAME, description="label of a case; not identity (benchmark_id is derived)"),
    "type_ref": s(pattern=TYPE_REF, description="<pack prefix>/<registered name>, e.g. llm/serve"),
    "prefix": s(pattern=PREFIX),
    "attr": s(pattern=ATTR, description="registry key; <name> marks a variable segment"),
    "attr_glob": s(pattern=ATTR_GLOB, description="attribute key or glob over registry keys"),
    "metric": s(pattern=METRIC),
    "owner": s(pattern=r"^[A-Za-z0-9][A-Za-z0-9-]*$|^<[a-z-]+>$"),
    "labels": {"type": "object", "additionalProperties": s(maxLength=63), "description": "selector key/values"},
    "tags": arr(
        s(pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$"),
        description="hashed into benchmark_id unless in a run-context namespace (platform__, testtype__, cadence__)",
    ),
    "links": arr(s(minLength=1)),
    "annotations": {"type": "object", "additionalProperties": s()},
    "arch": {"enum": ["x86_64", "ppc64le", "s390x", "aarch64"]},
    "duration": s(pattern=DUR, description="e.g. 90s, 10m, 2h"),
    "env": {
        "type": "object",
        "additionalProperties": SCALAR,
        "description": "process environment knobs; run attributes",
    },
    "extensions": {
        "type": "object",
        "additionalProperties": {"type": "object"},
        "description": "pack prefix -> block validated by that pack's extension schema for this kind",
    },
    "valref": obj({"env": s(), "value": s()}, desc="a value or the name of an environment variable; never a secret"),
    "predicate": obj(
        {
            "field": s(),
            **OPS,
            "in": arr(SCALAR),
            "notin": arr(SCALAR),
            "between": arr(SCALAR, minItems=2, maxItems=2),
            "exists": BOOL,
            "matches": s(),
            "any": arr(ref("predicate"), minItems=1),
            "not": ref("predicate"),
        },
        desc="structured filter predicate: {field, one op}, or {any: [...]}, or {not: {...}}",
    ),
    "select": obj(
        {
            "from": s(description="asset name inside the Dataset"),
            "files": arr(s()),
            "split": s(),
            "filter": arr(ref("predicate")),
            "columns": arr(s()),
            "order": any_of(
                {"enum": ["sequential"]},
                obj({"shuffle": obj({"seed": integer()}, required=("seed",))}, required=("shuffle",)),
            ),
            "offset": integer(minimum=0),
            "limit": integer(minimum=1),
            "sample": obj(
                {"n": integer(minimum=1), "fraction": num(minimum=0, maximum=1), "seed": integer(), "stratify_by": s()},
                required=("seed",),
            ),
            "ids": obj({"file": s(), "sha256": s(pattern=SHA256)}, required=("file", "sha256")),
            "repeat": {"enum": ["none", "cycle"]},
        },
        desc="pure function of (asset contents, select); every random operation carries a seed",
    ),
    "input": obj(
        {
            "name": ref("name"),
            "dataset": ref("name"),
            "asset": ref("name"),
            "model": ref("name"),
            "subset": s(description="named subset of the Dataset; inline select fields override it"),
            "select": ref("select"),
            "expect": obj({"records": integer(minimum=0), "selection_digest": s(pattern=r"^sha256:[0-9a-f]{64}$")}),
            "digest": {
                "enum": ["off", "on_expect", "always"],
                "description": "selection_digest policy; default on_expect",
            },
            "access": {"enum": ["path", "reader"]},
        },
        required=("name",),
        desc="exactly one of dataset, asset, model",
    ),
    "load": obj(
        {
            "executor": {"enum": ["closed", "open", "batch", "iterations", "ramp"]},
            "concurrency": integer(minimum=1),
            "requests": integer(minimum=1),
            "duration": ref("duration"),
            "rate": any_of(num(minimum=0), {"const": "inf"}),
            "arrival": {"enum": ["poisson", "uniform"]},
            "warmup": ref("duration"),
            "warmup_requests": integer(minimum=0),
            "iterations": integer(minimum=1),
            "warmup_iterations": integer(minimum=0),
            "stages": arr(
                obj({"duration": ref("duration"), "target": num(minimum=0)}, required=("duration", "target")),
                minItems=1,
            ),
        },
        required=("executor",),
        desc="closed {concurrency, requests|duration}; open {rate, requests|duration}; "
        "batch {requests}; iterations {iterations}; ramp {stages}",
    ),
    "slo": {
        "type": "object",
        "additionalProperties": obj(
            {
                "lte": any_of(num(), s(pattern=r"^[0-9]+(\.[0-9]+)?(us|ms|s)$")),
                "gte": any_of(num(), s(pattern=r"^[0-9]+(\.[0-9]+)?(us|ms|s)$")),
            }
        ),
        "description": "metric name -> bound; hashed in base units",
    },
    "workload_inline": obj(
        {"base": ref("name"), "patch": {"type": "object"}, "type": ref("type_ref")},
        extra=True,
        desc="a named Workload extended with `patch`, or a typed inline spec",
    ),
    "system_inline": obj({"base": ref("name"), "patch": {"type": "object"}, "type": ref("type_ref")}, extra=True),
    "tier": s(pattern=GROUPING),
    "groupings_value": any_of(SCALAR, arr(SCALAR, minItems=1)),
    "run": obj(
        {
            "cache": {"enum": ["warm", "cold", "both"]},
            "repeats": integer(minimum=1),
            "isolation": {"enum": ["session", "process"]},
            "timeout": ref("duration"),
            "est": ref("duration"),
            "runner": s(pattern=NAME),
            "seed": integer(),
            "fresh_cache": BOOL,
        },
        desc="run attributes: never identity; overridable at run time",
    ),
    "profile": obj(
        {
            "supported": arr({"enum": ["off", "light", "full"]}, minItems=1),
            "default": {"enum": ["off", "light", "full"]},
            "on_regression": {"enum": ["off", "light", "full"]},
        },
        desc="profile level is a run attribute: same benchmark_id at every level",
    ),
    "metrics": any_of(
        arr(ref("metric")),
        obj({"required": arr(ref("metric")), "headline": arr(ref("metric"))}),
    ),
    "relation": obj(
        {
            "type": s(pattern=r"^([a-z_]+|[a-z][a-z0-9]*/[a-z][a-z0-9-]*)$"),
            "target": s(minLength=1),
            "via": obj({"decompose": {"enum": ["sum", "max", "mean"]}}),
        },
        required=("type", "target"),
    ),
    "patch": obj(
        {
            "tier": ref("tier"),
            "groupings": obj({}, extra=True),
            "system": any_of(ref("name"), ref("system_inline")),
            "workload": any_of(ref("name"), ref("workload_inline")),
            "environment": ref("name"),
            "metrics": ref("metrics"),
            "env": ref("env"),
            "run": ref("run"),
            "profile": ref("profile"),
        },
        desc="JSON Merge Patch (RFC 7386) over the Benchmark spec",
    ),
    "variant": obj(
        {
            "name": ref("case_name"),
            "description": s(),
            "previous_names": arr(ref("case_name")),
            "tier": ref("tier"),
            "groupings": obj({}, extra=True),
            "system": any_of(ref("name"), ref("system_inline")),
            "workload": any_of(ref("name"), ref("workload_inline")),
            "environment": ref("name"),
            "metrics": ref("metrics"),
            "env": ref("env"),
            "run": ref("run"),
            "profile": ref("profile"),
        },
        required=("name",),
    ),
    "axis_value": obj(
        {"token": s(pattern=r"^[a-z0-9][a-z0-9.-]*$"), "patch": ref("patch")}, required=("token", "patch")
    ),
    "axis_sweep": obj(
        {
            "path": s(pattern=r"^[a-z_]+(\.[A-Za-z0-9_-]+)+$", description="dotted path into the patch"),
            "values": arr(SCALAR, minItems=1),
            "range": obj(
                {"from": num(), "to": num(), "step": any_of(num(), s(pattern=r"^x[0-9.]+$"))},
                required=("from", "to", "step"),
                desc="numeric sweep; step N adds, step xK multiplies",
            ),
            "token": s(description="format, {} = value; default <axis>{}"),
        },
        required=("path",),
    ),
    "matrix": obj(
        {
            "name_template": s(minLength=1, description="default <benchmark>/<axis>=<token>/..."),
            "axes": {
                "type": "object",
                "additionalProperties": any_of(arr(ref("axis_value"), minItems=1), ref("axis_sweep")),
            },
            "exclude": arr({"type": "object", "additionalProperties": s()}),
        },
        required=("axes",),
    ),
    "selector": obj(
        {
            "suites": arr(ref("name")),
            "types": arr(s()),
            "levels": arr(s()),
            "workload_types": arr(ref("type_ref")),
            "tags": ref("tags"),
            "groupings": {
                "type": "object",
                "additionalProperties": obj(
                    {"at_most": SCALAR, "at_least": SCALAR, "eq": SCALAR, "in": arr(SCALAR), "any": arr(SCALAR)}
                ),
            },
        },
        desc="AND of the given terms; groupings select on any Project grouping (at_most/at_least need an ordered one)",
    ),
    "derive": obj(
        {"from": ref("selector"), "per": {"enum": ["system"]}, "name_template": s(minLength=1)},
        required=("from", "per", "name_template"),
    ),
    "selector_term": obj(
        {"attribute": s(), **OPS, "in": arr(SCALAR)},
        desc="structured device selector",
    ),
    "device_request": obj(
        {
            "name": ref("name"),
            "device_class": s(pattern=r"^[a-z0-9-]+/[a-z0-9-]+$"),
            "count": integer(minimum=1, description="default 1"),
            "per_node": BOOL,
            "selectors": arr(ref("selector_term")),
        },
        required=("name", "device_class"),
    ),
    "device_constraint": obj(
        {
            "match_attribute": s(),
            "requests": arr(ref("name"), minItems=1),
            "scope": {"enum": ["per_node", "all"]},
        },
        required=("match_attribute", "requests"),
    ),
    "host_profile": obj(
        {
            "governor": {"enum": ["performance", "powersave", "schedutil", "n/a"]},
            "turbo": {"enum": ["on", "off", "n/a"]},
            "smt": s(),
            "cstate_policy": {"enum": ["fixed", "c1-only", "default", "n/a"]},
            "thp": {"enum": ["always", "madvise", "never"]},
            "numa_balancing": {"enum": [0, 1]},
            "cpu_quota_min_cores": num(minimum=0),
        }
    ),
    "store_inline": obj({"type": ref("type_ref")}, required=("type",), extra=True),
    "asset_source": obj(
        {
            "store": any_of(ref("name"), ref("store_inline")),
            "path": s(),
            "uri": s(),
            "revision": s(),
            "config": s(),
            "split": s(),
            "generator": obj(
                {"tool": s(minLength=1), "seed": integer(), "params": {"type": "object"}}, required=("tool", "seed")
            ),
        },
        desc="exactly one of store+path, uri, generator",
    ),
}

SPECS = {
    "Project": obj(
        {
            "component": s(pattern=NAME, description="hashed into benchmark_id; default metadata.name"),
            "packs": arr(
                obj(
                    {"name": ref("name"), "version": s(description="compatible range, e.g. 0.1")},
                    required=("name", "version"),
                ),
                minItems=1,
            ),
            "default_suites": arr(ref("name"), minItems=1),
            "default_environment": ref("name"),
            "validity_policy": ref("name"),
            "groupings": arr(
                obj(
                    {
                        "name": s(pattern=GROUPING),
                        "ordered": BOOL,
                        "cardinality": {"enum": ["one", "many"]},
                        "values": arr(SCALAR, minItems=1),
                    },
                    required=("name", "ordered", "cardinality"),
                ),
                description="stored as run props run.<name> / run.<name>.<value>; never tags",
            ),
            "series_labels": arr(
                s(pattern=r"^env\.label\.[a-z0-9_.]+$"),
                description="baseline/series key = hash of these label values + benchmark_id",
            ),
            "visibility": {"enum": ["internal", "public"], "description": "default internal"},
            "runner": ref("name"),
        },
        required=("packs",),
        desc="a catalog root (one per repo or central config tree)",
    ),
    "Suite": obj(
        {
            "type": s(description="test type (registered vocabulary test_types)"),
            "pr_selectable": BOOL,
            "requires_reference": BOOL,
        },
        required=("type",),
        desc="a folder of Benchmarks and its policy; the folder is the primary run.suite.<name>",
    ),
    "Selection": obj(
        {
            "cases": arr(ref("case_name"), minItems=1),
            "selector": ref("selector"),
            "reuse": obj(
                {
                    "mode": {"enum": ["never", "if_valid", "prefer", "only"]},
                    "match": obj(
                        {
                            "case": {"enum": ["exact"]},
                            "subject": {"enum": ["exact"]},
                            "environment": any_of(
                                {"enum": ["id"]},
                                obj({"labels": arr(ref("attr_glob"), minItems=1)}, required=("labels",)),
                            ),
                            "validity": {"enum": ["at_least_request"]},
                            "repeats": {"enum": ["at_least_request"]},
                        }
                    ),
                    "max_age": ref("duration"),
                    "force": arr(s()),
                    "grain": arr({"enum": ["result", "asset", "point"]}, minItems=1),
                },
                required=("mode",),
            ),
            "run": obj(
                {
                    "repeats_min": integer(minimum=1),
                    "max_minutes": num(minimum=0),
                    "arm": {"enum": ["", "AA"]},
                    "subject": s(description="pin the subject, e.g. a pinned reference artifact"),
                    "profile_level": {"enum": ["off", "light", "full"]},
                }
            ),
        },
        desc="named set: explicit case names or a selector (exactly one), plus reuse and run defaults",
    ),
    "Benchmark": obj(
        {
            "type": s(description="test type: perf | quality (+ pack vocabularies)"),
            "level": s(description="stack level measured (registered vocabulary levels)"),
            "suite": ref("name"),
            "tier": ref("tier"),
            "groupings": obj({}, extra=True),
            "system": any_of(ref("name"), ref("system_inline")),
            "workload": any_of(ref("name"), ref("workload_inline")),
            "environment": ref("name"),
            "metrics": ref("metrics"),
            "profile": ref("profile"),
            "analyze": arr(obj({"analyzer": s(), "args": {"type": "object"}}, required=("analyzer",))),
            "derived_metrics": arr(ref("metric")),
            "env": ref("env"),
            "run": ref("run"),
            "previous_names": arr(ref("name"), description="names this Benchmark had; links history, ids still change"),
            "relations": arr(ref("relation")),
            "variants": arr(ref("variant"), minItems=1),
            "matrix": ref("matrix"),
            "derive": ref("derive"),
        },
        required=("type", "level", "system", "workload"),
        desc="a TEMPLATE: exactly one of variants, matrix, derive expands it into cases; a case id is its benchmark_id",
    ),
    "System": obj(
        {
            "base": ref("name"),
            "patch": {"type": "object"},
            "type": ref("type_ref"),
            "token": s(pattern=r"^[a-z0-9][a-z0-9.-]*$"),
            "backend": s(pattern=r"^[a-z0-9][a-z0-9._-]*$"),
            "extensions": ref("extensions"),
        },
        extra=True,
        desc="configuration under test; typed. `backend` and `extensions` are unhashed (the comparison axis)",
    ),
    "Workload": obj(
        {
            "base": ref("name"),
            "patch": {"type": "object"},
            "type": ref("type_ref"),
            "scenario": s(),
            "inputs": arr(ref("input")),
            "load": ref("load"),
            "slo": ref("slo"),
            "expand": {"enum": ["per_record"]},
            "outputs": arr(
                obj(
                    {"name": ref("name"), "asset": ref("name"), "format": s(), "capture": arr(s())},
                    required=("name", "asset"),
                )
            ),
        },
        extra=True,
        desc="load put on a System: inputs (Asset layer), load executor, SLO; typed. Names no hardware",
    ),
    "Asset": obj(
        {
            "type": {"enum": ["fileset", "table", "generator", "generated", "stream"]},
            "format": s(),
            "version": s(),
            "license": s(),
            "visibility": {"enum": ["internal", "public"]},
            "source": ref("asset_source"),
            "integrity": obj(
                {
                    "files": arr(
                        obj(
                            {
                                "path": s(pattern=r"^[A-Za-z0-9._/-]+$"),
                                "sha256": s(pattern=SHA256),
                                "size": integer(minimum=0),
                                "media_type": s(),
                            },
                            required=("path", "sha256"),
                        ),
                        minItems=1,
                    ),
                    "manifest": obj({"path": s(), "sha256": s(pattern=SHA256)}, required=("path", "sha256")),
                    "revision_pinned": BOOL,
                }
            ),
            "access": obj(
                {"mode": {"enum": ["pull", "stream"]}, "resolve": arr(s()), "offline": {"enum": ["use-cache", "fail"]}}
            ),
            "produced_by": obj(
                {
                    "benchmark": ref("name"),
                    "output": ref("name"),
                    "inputs_digest": s(),
                    "policy": obj({"regenerate": {"enum": ["never", "if_inputs_change", "always"]}}),
                },
                required=("benchmark", "output"),
            ),
        },
        required=("type", "source"),
        desc="physical data: shape, source, integrity, access; asset_id is content-addressed",
    ),
    "Dataset": obj(
        {
            "asset": ref("name"),
            "assets": arr(obj({"name": ref("name"), "asset": ref("name")}, required=("name", "asset")), minItems=1),
            "role": {"enum": ["trace", "eval", "calibration", "shapes", "fixture", "schema"]},
            "schema": obj({"record": {"type": "object", "additionalProperties": s()}, "id": obj({"field": s()})}),
            "validate": obj({"mode": {"enum": ["off", "sample", "full"]}, "records": integer(minimum=1)}),
            "subsets": {"type": "object", "additionalProperties": ref("select")},
        },
        required=("role",),
        desc="semantics over assets (role, schema, named subsets); `asset` is shorthand for one",
    ),
    "Model": obj(
        {
            "asset": ref("name"),
            "assets": arr(
                obj(
                    {"role": {"enum": ["weights", "tokenizer", "config", "adapter"]}, "asset": ref("name")},
                    required=("role", "asset"),
                ),
                minItems=1,
            ),
            "family": s(),
            "architecture": s(),
            "modality": arr(s()),
            "params": obj({"total": num(minimum=0), "active": num(minimum=0)}),
            "native_precision": s(),
        },
        desc="semantics over weight assets; model_id = hash of the (role, asset_id) set",
    ),
    "Store": obj(
        {
            "type": ref("type_ref"),
            "endpoint": ref("valref"),
            "root": ref("valref"),
            "auth": obj(
                {
                    "mode": {"enum": ["token", "basic", "none"]},
                    "optional": BOOL,
                    "refs": arr(
                        obj(
                            {
                                "env": s(),
                                "file": s(),
                                "secret": s(),
                            }
                        )
                    ),
                }
            ),
            "visibility": {"enum": ["internal", "public"]},
            "mirrors": arr(ref("name")),
            "resolve": arr(s()),
            "offline": obj({"allowed": BOOL}),
        },
        required=("type", "visibility"),
        extra=True,
        desc="how to reach data: references to credentials, never values",
    ),
    "Environment": obj(
        {
            "base": ref("name"),
            "patch": {"type": "object"},
            "nodes": integer(minimum=1),
            "devices": obj({"requests": arr(ref("device_request")), "constraints": arr(ref("device_constraint"))}),
            "interconnect": obj({"requests": arr(ref("device_request"))}),
            "host": obj(
                {
                    "node_class": ref("name"),
                    "arch": ref("arch"),
                    "reference_node": BOOL,
                    "exclusive": BOOL,
                    "clean": obj(
                        {
                            "mode": {"enum": ["wait"]},
                            "max_wait": ref("duration"),
                            "max_hold": ref("duration"),
                        },
                        required=("mode", "max_wait", "max_hold"),
                        desc="clean run on demand via exclusive access to the node; absent means ordinary CI",
                    ),
                    "resources": obj({"cpu": integer(minimum=1), "memory": s(pattern=r"^[0-9]+(Mi|Gi|Ti)$")}),
                }
            ),
            "labels": {
                "type": "object",
                "additionalProperties": SCALAR,
                "description": "declared coarse facts, stored as env.label.<key>",
            },
            "policy": obj(
                {"validity_floor": {"enum": ["valid", "shared_env", "unqualified_env", "suspect"]}},
                desc="may only tighten the purpose floor",
            ),
            "extensions": ref("extensions"),
        },
        desc="the REQUEST: what a case needs. Never identity, never says where or how to launch",
    ),
    "NodeClass": obj(
        {
            "arch": ref("arch"),
            "match": obj({"nodes": arr(s()), "node_regex": s(), "labels": ref("labels")}),
            "host_profile": ref("host_profile"),
            "qualification": obj(
                {"expiry_days": integer(minimum=1), "probe": s(), "cv_limit_pct": num(minimum=0), "checks": arr(s())},
                desc="reference grade needs a passing probe at most expiry_days old; results never expire",
            ),
            "extensions": ref("extensions"),
        },
        required=("match",),
        desc="a class of hosts, their expected profile and qualification policy",
    ),
    "ValidityPolicy": obj(
        {
            "purposes": {
                "type": "object",
                "additionalProperties": obj(
                    {
                        "floor": {
                            "enum": ["valid", "shared_env", "unqualified_env", "suspect"],
                            "description": "lowest validity a result may have to be used for this purpose; an Environment may only tighten it",
                        },
                        "requires": obj(
                            {
                                "qualification": {"enum": ["current", "any"]},
                                "clean": {
                                    "enum": ["enforced", "clean", "any"],
                                    "description": "enforced = run.clean.exclusive enforced; clean = enforced, or observed + load_clean + devices_exclusive enforced",
                                },
                                "min_repeats": integer(minimum=1),
                                "max_cv_pct": num(minimum=0),
                                "discard_warmup": integer(minimum=0),
                                "placement": {"enum": ["required", "optional"]},
                            },
                            desc="gates a result must pass to earn `valid`; failing one caps the validity (qualification: unqualified_env, "
                            "clean: shared_env, the rest: suspect)",
                        ),
                        "baseline": {
                            "type": "boolean",
                            "description": "valid results of this purpose may serve as baselines",
                        },
                        "reference": {
                            "type": "boolean",
                            "description": "valid results of this purpose earn reference grade (never external)",
                        },
                        "description": s(),
                    },
                    required=("floor", "requires", "baseline", "reference"),
                ),
            },
        },
        required=("purposes",),
        desc="central: per purpose, the gates for each validity and the floor; external uploads never earn reference grade",
    ),
    "Expectation": obj(
        {
            "selector": obj(
                {
                    "benchmarks": arr(ref("name")),
                    "cases": arr(ref("case_name")),
                    "suites": arr(ref("name")),
                    "types": arr(s()),
                    "levels": arr(s()),
                    "workload_types": arr(ref("type_ref")),
                    "groupings": {
                        "type": "object",
                        "additionalProperties": obj(
                            {"at_most": SCALAR, "at_least": SCALAR, "eq": SCALAR, "in": arr(SCALAR), "any": arr(SCALAR)}
                        ),
                    },
                },
                desc="which cases this judges; AND of the given terms",
            ),
            "metric": ref("metric"),
            "baseline": obj(
                {
                    "kind": {"enum": ["rolling", "fixed"]},
                    "window": integer(minimum=1),
                    "min_n": integer(minimum=1),
                    "purposes": arr(s(), minItems=1),
                    "match": obj(
                        {
                            "environment": any_of(
                                {"enum": ["id"]},
                                obj({"labels": arr(ref("attr_glob"), minItems=1)}, required=("labels",)),
                            )
                        }
                    ),
                    "value": num(),
                },
                required=("kind",),
                desc="rolling: the last `window` prior VALID runs of these purposes in the same series; fixed: a target in base units",
            ),
            "tolerance": obj(
                {"pct": num(minimum=0)},
                required=("pct",),
                desc="worse-than-baseline tolerance (floor_pct); direction comes from the Metric",
            ),
            "noise": obj(
                {
                    "sigmas": num(minimum=0),
                    "floor_pct": num(minimum=0),
                    "source": {"enum": ["baseline_mad", "series_cv", "qualification"]},
                },
                desc="the effective floor is never tighter than this noise estimate",
            ),
            "severity": {
                "enum": ["informational", "advisory", "gating"],
                "description": "default advisory; gating fails the verdict",
            },
        },
        required=("selector", "metric", "baseline", "tolerance"),
        desc="authored next to the benchmark: how to judge one metric. Verdicts: pass | regressed | improved | suspected | no_baseline",
    ),
    "Analysis": obj(
        {
            "select": obj(
                {
                    "benchmarks": arr(ref("name"), minItems=1),
                    "via": {"enum": ["relations"]},
                    "groupings": {"type": "object", "additionalProperties": SCALAR},
                },
                required=("benchmarks",),
            ),
            "metric": ref("metric"),
            "hold": arr(ref("attr_glob")),
            "relax": arr(ref("attr_glob")),
            "vary": arr(ref("attr_glob"), minItems=1),
            "output": obj(
                {
                    "controlled_sets": BOOL,
                    "rank_variables": BOOL,
                    "noise": {"enum": ["series_floor", "none"]},
                    "chart": s(),
                }
            ),
            "decompose": obj(
                {"from": ref("name"), "to": arr(ref("name"), minItems=1), "show": {"enum": ["residual"]}},
                required=("from", "to"),
            ),
        },
        required=("select", "metric", "vary"),
        desc="a query over stored results, not a run: unlisted keys are held; unknown attributes are never held",
    ),
    "Pack": obj(
        {
            "prefix": ref("prefix"),
            "version": s(pattern=SEMVER),
            "requires": obj({"core": s()}),
            "depends": arr(obj({"pack": ref("name"), "version": s()}, required=("pack", "version"))),
            "types": arr(
                obj(
                    {
                        "name": s(pattern=r"^[a-z][a-z0-9-]*$"),
                        "for": {"enum": ["System", "Workload", "Store"]},
                        "schema": s(),
                        "non_identity": arr(s()),
                        "levels": arr(s()),
                        "runner": ref("name"),
                        "default_metrics": arr(ref("metric")),
                        "load": arr(
                            {"enum": ["closed", "open", "batch", "iterations", "ramp"]},
                            description="executors the workload type accepts; empty = no load block",
                        ),
                        "scenarios": {"type": "object", "additionalProperties": ref("load")},
                        "refs": {
                            "type": "object",
                            "additionalProperties": {"enum": ["Model", "Asset"]},
                            "description": "typed field -> kind; identity uses the referenced id, not the name",
                        },
                        "description": s(),
                    },
                    required=("name", "for", "schema"),
                )
            ),
            "extensions": arr(
                obj(
                    {"for": {"enum": ["System", "Environment", "NodeClass"]}, "schema": s(), "identity": BOOL},
                    required=("for", "schema"),
                )
            ),
            "vocabularies": obj({"test_types": arr(s()), "levels": arr(s()), "purposes": arr(s())}),
            "attributes": arr(
                obj(
                    {
                        "key": ref("attr"),
                        "class": {"enum": ["identity", "subject", "label", "environment", "run"]},
                        "role": {"enum": ["root", "derived"]},
                        "derived_from": ref("attr"),
                        "type": {"enum": ["string", "int", "float", "bool", "enum", "uuid"]},
                        "values": arr(SCALAR),
                        "unit": s(),
                        "order": {"enum": ["numeric", "lexical", "declared"]},
                        "plot": BOOL,
                        "fingerprint": BOOL,
                        "stored_in": s(),
                        "description": s(),
                    },
                    required=("key", "class", "role", "type"),
                )
            ),
            "device_classes": arr(
                obj(
                    {
                        "name": s(pattern=r"^[a-z0-9-]+/[a-z0-9-]+$"),
                        "kind": {"enum": ["accelerator", "cpu", "storage", "network"]},
                        "attributes": arr(
                            obj(
                                {
                                    "name": ref("attr"),
                                    "type": {"enum": ["string", "int", "float", "bool"]},
                                    "values": arr(SCALAR),
                                },
                                required=("name", "type"),
                            )
                        ),
                    },
                    required=("name", "kind"),
                )
            ),
            "metrics": arr(
                obj(
                    {
                        "name": ref("metric"),
                        "unit": s(description="UCUM, e.g. s, By, {request}/s, {token}/s, W, 1"),
                        "direction": {"enum": ["lower_is_better", "higher_is_better", "none"]},
                        "category": {"enum": ["performance", "quality", "efficiency", "capability", "diagnostic"]},
                        "statistic": {
                            "enum": ["value", "mean", "median", "p90", "p99", "max", "min", "count", "ratio", "sum"]
                        },
                        "compare": {"enum": ["relative", "absolute", "exact"]},
                        "noise": obj({"min_repeats": integer(minimum=1), "cv_max": num(minimum=0)}),
                        "otel": s(),
                        "description": s(),
                    },
                    required=("name", "unit", "direction", "category"),
                )
            ),
            "relations": arr(
                s(pattern=r"^[a-z][a-z0-9-]*$"), description="pack relation names, used as <prefix>/<name>"
            ),
            "runners": arr(
                obj(
                    {
                        "name": ref("name"),
                        "protocol": {"enum": ["process/v1", "python/v1"]},
                        "command": arr(s(), minItems=1),
                        "entry_point": s(),
                    },
                    required=("name", "protocol"),
                )
            ),
            "probes": arr(
                obj(
                    {
                        "name": ref("name"),
                        "protocol": {"enum": ["process/v1", "python/v1"]},
                        "command": arr(s()),
                        "entry_point": s(),
                        "produces": arr(ref("attr")),
                    },
                    required=("name", "protocol", "produces"),
                )
            ),
            "analyzers": arr(
                obj(
                    {
                        "name": s(pattern=TYPE_REF),
                        "consumes": arr(s()),
                        "produces": arr(ref("attr")),
                        "provenance": {"enum": ["derived", "measured"]},
                        "gating": BOOL,
                    },
                    required=("name",),
                )
            ),
            "resolvers": arr(
                obj({"name": s(pattern=TYPE_REF), "produces": arr(ref("attr"))}, required=("name",)),
                description="turn a build reference into subject.id and subject.dep.* attributes",
            ),
            "gates": arr(
                obj(
                    {
                        "id": ref("attr"),
                        "phase": {"enum": ["pre", "in", "post", "ingest"]},
                        "check": s(),
                        "reason": s(pattern=r"^[A-Z][A-Z0-9_]*$"),
                    },
                    required=("id", "phase", "reason"),
                )
            ),
            "storage": arr(
                obj(
                    {"attribute": ref("attr"), "column": s(pattern=r"^[a-z][a-z0-9_]*$"), "type": s()},
                    required=("attribute", "column"),
                )
            ),
        },
        required=("prefix", "version"),
        desc="registers a pack: types, extensions, vocabularies, attribute registry, "
        "device classes, metrics, runners, probes, analyzers, resolvers",
    ),
}

PLACEMENT = obj(
    {
        "id": s(),
        "nodes": arr(s()),
        "devices": arr(obj({"serial": s(), "group": s(), "firmware": s()}, extra=True)),
        "links": arr(s()),
    },
    required=("id", "nodes"),
)
RESULT_SHAPE = obj(
    {
        "apiVersion": {"const": API},
        "kind": {"const": "Result"},
        "metadata": obj({"name": s(), "labels": {"type": "object", "additionalProperties": s()}}, required=("name",)),
        "status": obj(
            {
                "case": obj(
                    {
                        "benchmark_id": s(),
                        "component": s(),
                        "name": s(),
                        "tags": arr(s()),
                        "disc": {"type": "object", "additionalProperties": s()},
                        "props": {"type": "object", "additionalProperties": s()},
                    },
                    required=("benchmark_id",),
                ),
                "subject": obj(
                    {"artifact_id": s(), "resolver": s(), "attrs": {"type": "object", "additionalProperties": s()}},
                    required=("artifact_id",),
                ),
                "environment": obj(
                    {
                        "env_id": s(),
                        "fingerprint_version": integer(minimum=1),
                        "fingerprint": {"type": "object", "additionalProperties": SCALAR},
                        "observations": {"type": "object", "additionalProperties": SCALAR},
                        "labels": {"type": "object", "additionalProperties": SCALAR},
                        "request": obj({"name": s(), "spec_digest": s()}),
                    },
                    required=("env_id", "fingerprint"),
                ),
                "series": obj(
                    {"key": s(), "labels": {"type": "object", "additionalProperties": SCALAR}}, required=("key",)
                ),
                "repeats": arr(
                    obj(
                        {
                            "run.repeat": integer(minimum=1),
                            "status": {"enum": ["ok", "failed"]},
                            "placement": PLACEMENT,
                            "clean": {"type": "object", "additionalProperties": SCALAR},
                            "samples": {"type": "object", "additionalProperties": arr(num())},
                        },
                        required=("run.repeat", "placement"),
                    ),
                    minItems=1,
                ),
                "aggregate": obj(
                    {
                        "statistic": {"enum": ["median", "mean"]},
                        "outliers": {"enum": ["none", "mad"]},
                        "n": integer(minimum=0),
                        "metrics": {
                            "type": "object",
                            "additionalProperties": obj({"value": num(), "unit": s()}, required=("value",)),
                        },
                    },
                    required=("statistic", "n"),
                ),
                "reused_from": any_of({"type": "null"}, s()),
                "validity": obj(
                    {
                        "state": {"enum": ["valid", "shared_env", "unqualified_env", "suspect", "invalid", "external"]},
                        "reasons": arr(s()),
                        "policy_digest": s(),
                        "purpose": s(),
                    },
                    required=("state",),
                ),
                "run_props": {"type": "object", "additionalProperties": s()},
            },
            required=("case", "subject", "environment", "series", "repeats", "aggregate"),
        ),
    },
    required=("apiVersion", "kind", "metadata", "status"),
    desc="TOOL-WRITTEN. Rows are stored by the tool; never authored",
)
INVALIDATION_SHAPE = obj(
    {
        "id": s(description="invalidation_id; a restored event names it in `restores`"),
        "scope_kind": {
            "enum": ["run", "run_benchmark", "node", "device", "placement", "query"],
            "description": "query is reserved: the writer rejects it until enabled",
        },
        "state": {"enum": ["invalid", "suspect", "restored"]},
        "run_id": s(),
        "benchmark_id": s(),
        "repeat": integer(minimum=1),
        "node": s(),
        "device": s(),
        "placement_id": s(),
        "match": s(),
        "from": s(format="date-time"),
        "to": s(format="date-time"),
        "code": {"enum": ["faulty_hardware", "measurement_bug", "bad_config", "under_investigation", "other"]},
        "reason": s(minLength=1),
        "restores": s(),
    },
    required=("scope_kind", "state", "reason"),
    desc="TOOL-WRITTEN row written by the invalidation writer",
)


def metadata():
    return obj(
        {
            "name": ref("name"),
            "description": s(),
            "owner": ref("owner"),
            "labels": ref("labels"),
            "tags": ref("tags"),
            "links": ref("links"),
            "annotations": ref("annotations"),
        },
        required=("name",),
    )


def resource(kind):
    props = {"apiVersion": {"const": API}, "kind": {"const": kind}, "metadata": metadata(), "spec": SPECS[kind]}
    return obj(props, required=("apiVersion", "kind", "metadata", "spec"))


def per_kind(kind):
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{ID_BASE}/{kind.lower()}.schema.json",
        "title": f"{API} {kind}",
        **resource(kind),
        "$defs": DEFS,
    }


def combined():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{ID_BASE}/resources.schema.json",
        "title": f"{API}: any authored resource",
        "type": "object",
        "required": ["apiVersion", "kind"],
        "allOf": [{"if": {"properties": {"kind": {"const": k}}, "required": ["kind"]}, "then": ref(k)} for k in KINDS],
        "$defs": {**DEFS, **{k: resource(k) for k in KINDS}},
    }


def shape(name, body):
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{ID_BASE}/tool-output/{name}.shape.json",
        "title": f"{API} tool output: {name}",
        **body,
    }


def render():
    files = {f"{k.lower()}.schema.json": per_kind(k) for k in KINDS}
    files["resources.schema.json"] = combined()
    files["tool-output/result.shape.json"] = shape("result", RESULT_SHAPE)
    files["tool-output/invalidation.shape.json"] = shape("invalidation", INVALIDATION_SHAPE)
    return {n: json.dumps(d, indent=1) + "\n" for n, d in files.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spyre-bench gen-schemas")
    ap.add_argument("--check", action="store_true", help="exit 1 if the schemas on disk are stale")
    ap.add_argument("--out", type=Path, help="schema directory (required to write from an installed package)")
    a = ap.parse_args(argv)
    out = a.out or OUT
    if not a.check and a.out is None and not SOURCE_CHECKOUT:
        print("error: installed package: pass --out DIR to write the schemas", file=sys.stderr)
        return 2
    files = render()
    expected = set(files)
    present = (
        {str(p.relative_to(out)) for p in list(out.glob("*.schema.json")) + list((out / "tool-output").glob("*.json"))}
        if out.is_dir()
        else set()
    )
    stale = sorted(present - expected) + [
        n for n, t in files.items() if not (out / n).exists() or (out / n).read_text() != t
    ]
    if a.check:
        if stale:
            print("stale:", stale)
        else:
            print(f"checked {len(files)} core schemas")
        return 1 if stale else 0
    (out / "tool-output").mkdir(parents=True, exist_ok=True)
    for n in present - expected:
        (out / n).unlink()
    for n, t in files.items():
        (out / n).write_text(t)
    print(f"wrote {len(files)} core schemas to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
