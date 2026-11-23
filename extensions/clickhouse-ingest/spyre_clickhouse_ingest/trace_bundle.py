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

"""A neutral multi-rank trace description as a `benchmarks` bundle (rows read by schema 87).

`python -m spyre_clickhouse_ingest.trace_bundle trace.json --out DIR --artifact-id UUID --arch ARCH`
writes DIR/bundle.json and DIR/results/*.json; `results --from-bundle DIR` ingests it. One results file
per (rank, prompt, repeat): the writer folds rows of one benchmark within a file, so a file must hold
exactly one rank. Parsing a particular profiler's report into the description below is the caller's job;
this module names no signal.

trace.json: {component, backend?, case: {name, props?}, roles: {total, wait?, compute?, busy?},
rank_rules?: {signal: max|min|sum|median|mean}, group_id?, reporter_version?, reporter_version_source?,
trace_uri?, ranks: [{rank, prompt?, repeat?, steps: [{kind: ttft|itl, cold?, index?, values: {signal: x}}],
spans?: [{step, seq, parent_seq, depth, layer, lane, name, start_ns, dur_ns}]}]}
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from .identity import BenchmarkId

ROLES = ("total", "wait", "compute", "busy")
RULES = {
    "max": max,
    "min": min,
    "sum": sum,
    "median": statistics.median,
    "mean": statistics.mean,
}
ROLE_DEFAULT_RULE = {"total": "max"}
MAX_TRACE_BYTES = 512 * 1024 * 1024
RESERVED = ("step_index", "step_kind", "cold")
SPAN_KEYS = ("step", "seq", "parent_seq", "depth", "layer", "start_ns", "dur_ns")


class TraceError(ValueError):
    pass


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _rule(trace: dict, signal: str) -> str:
    declared = (trace.get("rank_rules") or {}).get(signal)
    if declared is not None and declared not in RULES:
        raise TraceError(f"rank rule for {signal!r} must be one of {sorted(RULES)}")
    role = next((r for r in ROLES if (trace["roles"].get(r) == signal)), None)
    return declared or ROLE_DEFAULT_RULE.get(role, "mean")


def _step_index(steps: list) -> list:
    return [s["index"] if "index" in s else i for i, s in enumerate(steps)]


def validate(trace: dict) -> None:
    roles = trace.get("roles") or {}
    if (
        "total" not in roles
        or not trace.get("component")
        or not (trace.get("case") or {}).get("name")
    ):
        raise TraceError("a trace needs component, case.name and roles.total")
    if set(roles) - set(ROLES):
        raise TraceError(f"roles are {ROLES}")
    ranks = trace.get("ranks") or []
    if not ranks:
        raise TraceError("a trace needs at least one rank")
    if "world_size" not in (trace["case"].get("props") or {}):
        raise TraceError(
            "case.props.world_size is required: it is part of the benchmark identity"
        )
    seen = set()
    lengths: dict = {}
    keysets: dict = {}
    for rk in ranks:
        for field in ("rank", "repeat"):
            v = rk.get(field, 0)
            if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 65535:
                raise TraceError(f"{field} must be an integer in 0..65535, got {v!r}")
        prompt = rk.get("prompt", "")
        if (
            not isinstance(prompt, str)
            or len(prompt) > 64
            or not all(c.isascii() and (c.isalnum() or c in "_-") for c in prompt)
        ):
            raise TraceError(
                f"prompt must be up to 64 characters of [A-Za-z0-9_-], got {prompt!r}"
            )
        key = (rk["rank"], prompt, rk.get("repeat", 0))
        if key in seen:
            raise TraceError(f"rank/prompt/repeat {key} appears twice")
        seen.add(key)
        steps = rk.get("steps") or []
        if not steps or any(s.get("kind") not in ("ttft", "itl") for s in steps):
            raise TraceError(f"rank {rk['rank']}: steps need a kind of ttft or itl")
        lengths.setdefault((prompt, rk.get("repeat", 0)), set()).add(len(steps))
        keysets.setdefault((prompt, rk.get("repeat", 0)), set()).add(
            frozenset(_step_keys(steps))
        )
        idx = _step_index(steps)
        if len(set(zip((s["kind"] for s in steps), idx))) != len(idx) or not all(
            isinstance(i, int) and not isinstance(i, bool) and 0 <= i < 2**32 - 1
            for i in idx
        ):
            raise TraceError(
                f"rank {rk['rank']}: step indexes must be unique non-negative integers within a kind"
            )
        if not all(s.get("cold", 0) in (0, 1) for s in steps):
            raise TraceError(f"rank {rk['rank']}: cold is 0 or 1")
        for s in steps:
            clash = [
                k
                for k in s["values"]
                if k in RESERVED or k.startswith(("summary.", "headline."))
            ]
            if clash:
                raise TraceError(
                    f"rank {rk['rank']}: signal name(s) {clash} are reserved"
                )
            for role, sig in roles.items():
                if sig not in s["values"]:
                    raise TraceError(
                        f"rank {rk['rank']}: a step lacks the {role} signal {sig!r}"
                    )
            if not all(_finite(v) for v in s["values"].values()):
                raise TraceError(
                    f"rank {rk['rank']}: step values must be finite numbers"
                )
        for sp in rk.get("spans") or []:
            if (
                not sp.get("lane")
                or not sp.get("name")
                or any(k not in sp for k in SPAN_KEYS)
            ):
                raise TraceError(
                    f"rank {rk['rank']}: a span needs lane, name and {SPAN_KEYS}"
                )
            if not all(_finite(sp[k]) for k in SPAN_KEYS):
                raise TraceError(
                    f"rank {rk['rank']}: span fields must be finite numbers"
                )
            ints = ("step", "seq", "depth", "layer", "parent_seq")
            if not (
                all(float(sp[k]).is_integer() for k in ints)
                and 0 <= sp["depth"] <= 255
                and 0 <= sp["step"] < len(steps)
                and 0 <= sp["seq"] < 2**32 - 1
                and abs(sp["layer"]) <= 32767
                and abs(sp["parent_seq"]) < 2**31 - 1
                and 0 <= sp["dur_ns"] <= 9.0e18
                and abs(sp["start_ns"]) <= 9.0e18
            ):
                raise TraceError(
                    f"rank {rk['rank']}: span step and seq must be >= 0 and depth within 0..255, "
                    "integer fields integral, layer within +-32767, dur_ns within 0..9e18"
                )
    seen_signals = {k for rk in ranks for s in rk["steps"] for k in s["values"]}
    unknown = set(trace.get("rank_rules") or {}) - seen_signals
    if unknown:
        raise TraceError(f"rank_rules name signals no step has: {sorted(unknown)}")
    for group, counts in lengths.items():
        if len(counts) > 1:
            raise TraceError(
                f"ranks of prompt/repeat {group} have different step counts {sorted(counts)}; the producer must align or drop them"
            )
    for group, sets in keysets.items():
        if len(sets) > 1:
            raise TraceError(
                f"ranks of prompt/repeat {group} report different (kind, index) steps; the producer must align or drop them"
            )


def _step_keys(steps: list) -> list:
    return list(zip((s["kind"] for s in steps), _step_index(steps)))


def _headline(trace: dict, group: list) -> dict:
    """`headline.<kind>.<signal>`: median over warm steps of the cross-rank value of each step."""
    out = {}
    signals = sorted({k for rk in group for s in rk["steps"] for k in s["values"]})
    for kind in ("ttft", "itl"):
        for sig in signals:
            per_step = []
            by_key = [dict(zip(_step_keys(rk["steps"]), rk["steps"])) for rk in group]
            for key in sorted(by_key[0]):
                if key[0] != kind:
                    continue
                cells = [b[key] for b in by_key]
                if any(c.get("cold") for c in cells):
                    continue
                vals = [c["values"][sig] for c in cells if sig in c["values"]]
                if vals:
                    per_step.append(RULES[_rule(trace, sig)](vals))
            if per_step:
                out[f"headline.{kind}.{sig}"] = [statistics.median(per_step)]
    return out


def _summary(rk: dict) -> dict:
    out = {}
    for kind in ("ttft", "itl"):
        by_sig = {}
        for s in rk["steps"]:
            if s["kind"] == kind:
                for sig, v in s["values"].items():
                    by_sig.setdefault(sig, []).append(v)
        out.update(
            {f"summary.{kind}.{sig}": [statistics.mean(v)] for sig, v in by_sig.items()}
        )
    return out


def _identity_id(component: str, name: str, props: dict) -> str:
    props = {k: str(v) for k, v in props.items()}
    return BenchmarkId.derive(component, name, [], props, tuple(sorted(props)))


def emit(trace: dict) -> dict:
    """{results file name: {"benchmarks": [...]}} for a trace description."""
    validate(trace)
    comp, case = trace["component"], trace["case"]
    ranks = trace["ranks"]
    world = case.get("props", {}).get("world_size", len({rk["rank"] for rk in ranks}))
    step_props = {
        **(case.get("props") or {}),
        "record_type": "step_trace",
        "world_size": world,
    }
    case_id = _identity_id(comp, case["name"], step_props)
    declared = {s: r for s, r in (trace.get("rank_rules") or {}).items()}
    common = {"run.world_size": world, "run.group_id": trace.get("group_id", ""),
              "run.reporter_version": trace.get("reporter_version", ""),
              "run.reporter_version_source": trace.get("reporter_version_source", ""),
              "run.trace_uri": trace.get("trace_uri", ""),
              **{f"role.{r}": s for r, s in trace["roles"].items()},
              **{f"agg.{s}": r for s, r in declared.items()}}  # fmt: skip
    common = {k: v for k, v in common.items() if v != ""}
    groups = {}
    for rk in ranks:
        groups.setdefault((rk.get("prompt", ""), rk.get("repeat", 0)), []).append(rk)
    files = {}
    for (prompt, repeat), group in groups.items():
        headline = _headline(trace, group)
        for rk in group:
            steps = rk["steps"]
            signals = sorted({k for s in steps for k in s["values"]})
            meas = {"step_index": [float(i) for i in _step_index(steps)],
                    "step_kind": [0.0 if s["kind"] == "ttft" else 1.0 for s in steps],
                    "cold": [1.0 if s.get("cold") else 0.0 for s in steps]}  # fmt: skip
            for sig in signals:
                if any(sig not in s["values"] for s in steps):
                    raise TraceError(
                        f"signal {sig!r} is missing in some step of rank {rk['rank']}"
                    )
                meas[sig] = [float(s["values"][sig]) for s in steps]
            meas.update(_summary(rk))
            meas.update(headline)
            run_props = {
                **common,
                "run.rank": rk["rank"],
                "run.prompt": prompt,
                "run.repeat": repeat,
            }
            run_props = {k: v for k, v in run_props.items() if v != ""}
            rows = [{"name": case["name"], "props": step_props, "backend": trace.get("backend", ""),
                     "measurements": meas, "iterations": len(steps), "run_props": run_props}]  # fmt: skip
            patterns = {}
            for sp in rk.get("spans") or []:
                patterns.setdefault((sp["lane"], sp["name"]), []).append(sp)
            for (lane, name), spans in sorted(patterns.items()):
                sprops = {
                    "record_type": "span",
                    "lane": lane,
                    "kernel_name": name,
                    "case_id": case_id,
                }
                rows.append({"name": f"{case['name']}/{lane}/{name}", "props": sprops, "backend": trace.get("backend", ""),
                             "measurements": {k: [float(s[k]) for s in spans] for k in SPAN_KEYS},
                             "iterations": len(spans),
                             "run_props": {k: v for k, v in run_props.items() if not k.startswith(("role.", "agg."))}})  # fmt: skip
            tag = f"-p{prompt}" if prompt != "" else ""
            files[f"rank{rk['rank']}{tag}-rep{repeat}.json"] = {"benchmarks": rows}
    return files


def write_bundle(
    trace: dict, out: Path, artifact_id: str, arch: str, test_type: str = "perf"
) -> list:
    files = emit(trace)
    if (out / "results").exists() and any((out / "results").iterdir()):
        raise TraceError(
            f"{out}/results is not empty: a bundle must hold exactly one trace, use a new directory"
        )
    (out / "results").mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (out / "results" / name).write_text(json.dumps(body))
    (out / "bundle.json").write_text(json.dumps({"artifact_id": artifact_id, "kind": "benchmarks", "test_type": test_type,
                                                 "component": trace["component"], "arch": arch}))  # fmt: skip
    return sorted(files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="trace_bundle", description=__doc__.split("\n")[0]
    )
    ap.add_argument("trace", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--artifact-id", required=True)
    ap.add_argument("--arch", required=True)
    args = ap.parse_args(argv)
    try:
        if args.trace.stat().st_size > MAX_TRACE_BYTES:
            raise TraceError(f"{args.trace} is larger than {MAX_TRACE_BYTES} bytes")
        names = write_bundle(
            json.loads(args.trace.read_text()), args.out, args.artifact_id, args.arch
        )
    except (
        TraceError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        OSError,
    ) as err:
        print(f"trace_bundle: {err!r}", file=sys.stderr)
        return 2
    print(f"{len(names)} results file(s) in {args.out}/results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
