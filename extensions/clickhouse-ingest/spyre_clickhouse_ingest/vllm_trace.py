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

"""vLLM trace bundle -> spyre_v2 benchmarks / benchmark_runs (report_kind 'vllm-trace').

`vllm-trace <run_dir> [--analyzed <dir>] [--label L]` reads the analyzers' JSON for one profiled
run (`<label>.res.json` resource_report, `.roof.json` kernel_roofline, `.host.json`
host_timeline; any subset) and, from the run dir, the artifact (`spyre_artifact.json`, else the
`artifact` block of `template.json` / `manifest.json`), the workload and engine limits (`e2e.json` args, `vllm.log`, or CLI
flags), the env fingerprint and an optional `kernel_cost*.jsonl`. Raw traces are not analyzed here.
Rows: record_type trace_phase | trace_step | trace_kernel | compile_pass, linked to the artifact
by a performance verdict, as vllm.py's rows are.
Counts outside UInt32 are stored as 0 (not stated); a row whose benchmark id repeats an earlier one's
(a phase or kernel differing only in case or spacing) is dropped.
`arch unresolved` (no arch in the artifact block) rejects a live run, and also a `--dry-run` when
CLICKHOUSE_HOST is set; a dry-run without a host reports it and carries on.
Re-ingesting under the same `<id12>:<label>` reports 'duplicate' even if the workload differs, and a
failure after the first 5000-row batch leaves a partial run that later reads as duplicate (the shared
writer's behaviour).
"""

import argparse
import glob
import json
import logging
import math
import regex as re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import benchmeta
from .identity import BenchmarkId, RunId

log = logging.getLogger(__name__)

REPORT_KIND = "vllm-trace"
# schema/62-benchmark-verdicts.sql excludes exactly these from the regression verdicts.
RECORD_TYPES = ("trace_phase", "trace_step", "trace_kernel", "compile_pass")
RT_PHASE, RT_STEP, RT_KERNEL, RT_COMPILE = RECORD_TYPES
MAX_LABEL = 128
MAX_COUNT = 2**32 - 1  # iterations is UInt32
MAX_F32 = 3.4e38  # artifact_results.duration_s is Float32
BACKEND = "spyre"
COMPONENT = "spyre-inference"
# Workload and engine limits are in every hash: they change the compiled programs. Unknown limits hash empty.
# Positional: reordering re-keys every trace benchmark.
ID_KEYS = (
    "record_type",
    "model",
    "tensor_parallel",
    "batch_size",
    "input_len",
    "output_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "max_model_len",
    "run_mode",
    "phase",
    "provenance_key",
    "pass",
)
PHASE_FIELDS = {
    "wall_s": "wall_s",
    "cpu_s": "cpu_s",
    "cpu_cores_mean": "cpu_cores_mean",
    "cpu_cores_peak": "cpu_cores_peak",
    "throttled_s": "throttled_s",
    "nr_throttled": "nr_throttled",
    "rss_peak_b": "rss_peak_b",
    "cg_mem_peak_b": "cgroup_peak_b",
    "dev_allocated_peak_b": "dev_allocated_peak_b",
}
STEP_FIELDS = {
    "wall_ms": "wall_ms",
    "busy_ms": "busy_ms",
    "host_exposed_ms": "host_exposed_ms",
    "launches_per_step": "launches",
    "fallback_ms_per_step": "fallback_ms",
}
KERNEL_FIELDS = {
    "calls": "calls",
    "median_us": "median_us",
    "total_ms": "total_ms",
    "floor_us": "floor_us",
    "ratio": "meas_over_floor",
    "gbps": "gbps",
    "hbm_bytes": "hbm_bytes",
    "macs": "macs",
    "cores": "cores",
}
TOP_FALLBACK_OPS = 10
TOP_THREAD_GROUPS = 5


class TraceError(ValueError):
    """A bundle that cannot be ingested; the message says what is missing."""


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _f(v: Any) -> float | None:
    """v as a finite float; None for a non-number, NaN, infinity or an int beyond float."""
    if not _num(v):
        return None
    try:
        x = float(v)
    except OverflowError:
        return None
    return x if math.isfinite(x) else None


def _n(v: Any) -> int:
    """A count for the UInt32 `iterations`; 0 (not stated) for anything outside 0..2**32-1."""
    x = _f(v)
    return int(x) if x is not None and 0 <= x <= MAX_COUNT else 0


def _dict(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def _list(v: Any) -> list:
    return v if isinstance(v, list) else []


def _json(path: Path) -> Any:
    """The parsed file; None when unreadable, since every file here describes a run and never decides one."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, RecursionError) as err:
        log.warning("Unreadable %s (%s) — skipped", path, err)
        return None


def _measure(src: dict, fields: dict[str, str]) -> dict[str, list[float]]:
    return {to: [x] for k, to in fields.items() if (x := _f(src.get(k))) is not None}


def family(aten: list[str]) -> str:
    """Kernel family from a bundle's fused ATen ops: a bundle holding a matmul is a matmul."""
    a = aten or []
    if any("all_reduce" in x for x in a):
        return "allreduce"
    if any("scaled_dot_product" in x for x in a):
        return "attention_encoder"
    if any("batched_matmul" in x for x in a):
        return "attention"
    if any(x.endswith(".mm.default") or "bmm" in x or "scaled_mm" in x for x in a):
        return "matmul"
    if any("index_copy" in x or "embedding" in x for x in a):
        return "kv_write_embed"
    return "elementwise"


def model_name(raw: str) -> str:
    """org/name from an HF cache snapshot path, else the string as given."""
    m = re.search(r"models--([^/]+?)--([^/]+)", raw)
    return f"{m[1]}/{m[2]}" if m else raw


def _first_json(run: Path, *names: str) -> Any:
    for d in (run, run / "rank0"):
        for n in names:
            if (d / n).is_file():
                return _json(d / n)
    return None


def _artifact(run: Path) -> tuple[dict, dict]:
    """(artifact block, env fingerprint) of the run dir."""
    art, env = {}, {}
    for name, key in (
        ("spyre_artifact.json", None),
        ("template.json", "artifact"),
        ("manifest.json", "artifact"),
    ):
        p = run / name
        if not p.is_file():
            continue
        d = _dict(_json(p))
        block = d if key is None else d.get(key)
        if (
            isinstance(block, dict)
            and not art
            and (block.get("artifact_id") or block.get("id12"))
        ):
            art = block
        if isinstance(d.get("env"), dict) and not env:
            env = d["env"]
    env_json = _first_json(run, "env.json")
    if isinstance(env_json, dict):
        env = env_json
        if isinstance(env_json.get("artifact"), dict) and not art:
            art = env_json["artifact"]
    return art, env


def _run_utc(run: Path) -> str:
    for d in (run, run / "rank0"):
        p = d / "phases.jsonl"
        if p.is_file():
            try:
                with p.open(encoding="utf-8-sig", errors="replace") as fh:
                    t = json.loads(fh.readline())["t"]
                return datetime.fromtimestamp(t, timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except (
                IndexError, KeyError, TypeError, ValueError, OverflowError, OSError, RecursionError
            ):  # fmt: skip
                continue
    return ""


def _predicted(run: Path) -> dict[str, float]:
    """provenance_key -> predicted_us from the kernel-cost sidecar, when it carries one."""
    out: dict[str, float] = {}
    paths = glob.glob(str(run / "kernel_cost*.jsonl")) + glob.glob(
        str(run / "*" / "kernel_cost*.jsonl")
    )
    for path in sorted(paths):
        try:
            with open(path, encoding="utf-8-sig", errors="replace") as fh:
                for line in fh:
                    try:
                        r = _dict(json.loads(line))
                    except (ValueError, RecursionError):
                        continue
                    cost = _dict(r.get("cost"))
                    key = r.get("provenance_key") or r.get("key")
                    x = _f(cost.get("predicted_us"))
                    if isinstance(key, str) and key and x is not None:
                        out[key] = x
        except OSError:
            continue
    return out


ENGINE_LIMITS = ("max_num_batched_tokens", "max_num_seqs", "max_model_len")


def _logged_args(run: Path) -> dict[str, str]:
    """Engine limits and dtype from vllm.log's `non-default args` line."""
    logged: dict[str, str] = {}
    for d in (run, run / "rank0"):
        log = next(
            (q for q in (d / "vllm.log", d / "vllm.log.txt") if q.is_file()), None
        )
        if log:
            with log.open(errors="replace") as f:
                for line in f:
                    if "non-default args:" in line:
                        for k in ENGINE_LIMITS:
                            m = re.search(rf"'{k}': (\d+)", line)
                            if m:
                                logged[k] = m[1]
                        m = re.search(r"'dtype': '([^']+)'", line)
                        if m:
                            logged["dtype"] = m[1]
                        break
            break
    return logged


def _engine_limits(logged: dict, args: dict, batch_size, given: dict) -> dict[str, str]:
    """Engine limits: flag, else vllm.log's `non-default args` line, else e2e.json (budget 0 = default)."""
    e2e = {
        "max_num_batched_tokens": args.get("budget") or "",
        "max_model_len": args.get("max_model_len") or "",
        "max_num_seqs": batch_size if args.get("max_model_len") else "",
    }
    return {
        k: str(given.get(k) or logged.get(k) or e2e[k] or "") for k in ENGINE_LIMITS
    }


def load(run_dir, analyzed=None, label="", **workload) -> dict:
    """Everything the rows need, from the run dir and its analyzed JSON."""
    run = Path(run_dir)
    ana = Path(analyzed) if analyzed else run
    manifest = (
        _dict(_json(run / "manifest.json")) if (run / "manifest.json").is_file() else {}
    )
    # Harness bundles all sit in a dir named `trace`: their manifest names the run.
    label = str(label or manifest.get("label") or run.name)
    if (
        not label
        or len(label) > MAX_LABEL
        or label == ".."
        or any(c in label for c in "/\\\0")
    ):
        raise TraceError(
            f"unsafe label {label[:40]!r}: 1-{MAX_LABEL} chars, no path separator or NUL, not '..'"
        )
    found = {}
    for kind in ("res", "roof", "host"):
        for p in (ana / f"{label}.{kind}.json", ana / f"{kind}.json"):
            if p.is_file():
                got = _json(p)
                if got is not None:
                    found[kind] = got
                    break
    if not found:
        raise TraceError(f"no {label}.res/roof/host.json in {ana}")
    host = found.get("host")
    n_ranks = len(host) if isinstance(host, list) else None
    if isinstance(host, list):
        host = host[0] if host else None
    host = _dict(host)
    art, env = _artifact(run)
    e2e_json = _dict(_first_json(run, "e2e.json"))
    args = _dict(e2e_json.get("args"))
    w = {
        "model": workload.get("model") or args.get("model"),
        "tp": workload.get("tp") or args.get("tp"),
        "batch_size": workload.get("batch_size") or args.get("batch_size"),
        "input_len": workload.get("input_len") or args.get("input_len"),
        "output_len": workload.get("output_len") or args.get("output_len"),
    }
    missing = [k for k, v in w.items() if not v]
    if missing:
        raise TraceError(
            f"workload not known ({', '.join(missing)}): no e2e.json, no flag"
        )
    w["model"] = model_name(str(w["model"]))
    logged = _logged_args(run)
    w.update(_engine_limits(logged, args, w["batch_size"], workload))
    rc = _dict(env.get("runtime_config"))
    for k in ("dtype", "task"):
        w[k] = str(rc.get(k) or args.get(k) or logged.get(k) or "")
    return {
        "label": label,
        "res": _dict(found.get("res")),
        "roof": _dict(found.get("roof")),
        "host": host,
        "artifact": art,
        "env": env,
        "workload": w,
        "predicted": _predicted(run),
        "run_utc": _run_utc(run),
        "compile_summary": _dict(_first_json(run, "compile_summary.json")),
        "series_files": _series_files(run, str(host.get("trace") or "")),
        "n_ranks": n_ranks,
        "profile_level": manifest.get("profile_level") or e2e_json.get("profile_level"),
    }


def _series_files(run: Path, trace: str) -> dict[str, str]:
    """Run-dir-relative paths of the host and device-memory samples and the step trace."""
    out = {}
    for key, pats in (
        ("host", ("samples.jsonl",)),
        ("devmem", ("devmem.jsonl", "markers/devmem.*.jsonl")),
        ("step_trace", ("traces/*.json", "traces/*.json.gz", "raw/*.pt.trace.json*")),
    ):
        found = sorted(
            q for d in (run, run / "rank0") for pat in pats for q in d.glob(pat)
        )
        named = [q for q in found if trace and trace.endswith(q.name)]
        if named or found:
            out[key] = (named or found)[0].relative_to(run).as_posix()
    return out


def _compact(d: dict, n: int, key=None) -> str:
    """Top n of d as one-line JSON, largest first; empty when d is."""
    top = sorted(d.items(), key=key or (lambda kv: -kv[1]))[:n]
    return json.dumps(dict(top), separators=(",", ":")) if top else ""


def run_props(ctx: dict, bundle_url: str = "") -> dict[str, str]:
    """Props on every row: workload, engine limits, env fingerprint, artifact_id12, compile totals, run time."""
    art = ctx["artifact"]
    w = ctx["workload"]
    out = benchmeta.flatten("env", {**ctx["env"], "artifact": art})
    res = ctx["res"]
    comp = _dict(res.get("compile")) or ctx["compile_summary"]
    cache = comp.get("kernel_cache") or comp.get("cache")
    out.update(
        benchmeta.flatten(
            "compile_summary", {"total_s": comp.get("total_s"), "cache": cache}
        )
    )
    for k, v in (
        ("model", w["model"]),
        ("tensor_parallel", w["tp"]),
        ("batch_size", w["batch_size"]),
        ("input_len", w["input_len"]),
        ("output_len", w["output_len"]),
        *((k, w.get(k)) for k in (*ENGINE_LIMITS, "dtype", "task")),
        ("compile_n_compiles", comp.get("n_compiles")),
        ("profile_level", ctx.get("profile_level")),
        ("n_ranks", ctx.get("n_ranks")),
        ("dev_mem_total_b", _dict(res.get("device_memory")).get("total_b")),
        (
            "series_files",
            json.dumps(ctx["series_files"], separators=(",", ":"))
            if ctx["series_files"]
            else "",
        ),
        ("arch", art.get("arch") or ctx["env"].get("arch")),
        ("hardware_type", "IBM_Spyre"),
        ("run_utc", ctx["run_utc"]),
        ("trace_label", ctx["label"]),
        ("trace_bundle_url", bundle_url),
        ("source_file", f"{REPORT_KIND}/{ctx['label']}"),
    ):
        if v:
            out[k] = str(v)
    return out


def _extra(**kv) -> dict[str, str] | None:
    return {k: str(v) for k, v in kv.items() if v} or None


def entries(ctx: dict, bundle_url: str = "") -> list[dict]:
    """insert_benchmarks entries: one per phase, step kind, kernel and compile pass."""
    w = ctx["workload"]
    config = (
        f"{w['model'].rsplit('/', 1)[-1]}_tp{w['tp']}_bs{w['batch_size']}"
        f"_in{w['input_len']}_out{w['output_len']}"
    )
    shared = {
        "config_name": config,
        "model": w["model"],
        "tensor_parallel": str(w["tp"]),
        "batch_size": str(w["batch_size"]),
        "input_len": str(w["input_len"]),
        "output_len": str(w["output_len"]),
        **{k: w[k] for k in ENGINE_LIMITS if w.get(k)},
    }
    rprops = run_props(ctx, bundle_url)
    out: list[dict] = []
    seen: set[str] = set()

    def add(suffix, props, measurements, iterations=0, extra=None):
        if not measurements:
            return
        props = {**shared, **props}
        # The writer merges rows of one benchmark_id and sums their iterations: keep ids unique.
        bid = BenchmarkId.derive(COMPONENT, f"{config}:{suffix}", [], props, ID_KEYS)
        if bid in seen:
            log.warning("Duplicate %s:%s — first row kept", config, suffix)
            return
        seen.add(bid)
        out.append(
            {
                "name": f"{config}:{suffix}",
                "tags": [],
                "backend": BACKEND,
                "props": props,
                "measurements": measurements,
                "iterations": _n(iterations),
                "run_props": {**rprops, "run_mode": props["run_mode"], **(extra or {})},
                "disc": props,
                "disc_keys": ID_KEYS,
            }
        )

    res = ctx["res"]
    for phase, p in _dict(res.get("phases")).items():
        if not isinstance(p, dict):
            continue
        add(
            f"phase:{phase}",
            {"record_type": RT_PHASE, "run_mode": "trace", "phase": phase},
            _measure(p, PHASE_FIELDS),
            _n(p.get("n_samples")),
            _extra(
                top_thread_groups=_compact(
                    {
                        g: x
                        for g, v in _dict(p.get("top_thread_groups")).items()
                        if (x := _f(v)) is not None
                    },
                    TOP_THREAD_GROUPS,
                )
            ),
        )

    host = ctx["host"]
    steps = _dict(res.get("steps"))
    for kind in ("prefill", "decode", "mixed"):
        s = _dict(host.get(f"{kind}_summary")) or _dict(steps.get(kind))
        m = _measure(s, STEP_FIELDS)
        waterfall = s.get("waterfall_ms_per_step")
        if isinstance(waterfall, dict):
            # The waterfall lists only nonzero classes: an absent cpu_fallback is a measured 0.
            for cls, ms in {"cpu_fallback": 0.0, **waterfall}.items():
                if (x := _f(ms)) is not None:
                    m[f"host_{cls}_ms"] = [x]
        fb = _dict(
            s.get("fallback_ops")
            or _dict(_dict(steps.get("cpu_fallback")).get(kind)).get("ops")
        )
        fb = {
            op: {
                a: round(x, 4)
                for a in ("calls_per_step", "ms_per_step")
                if (x := _f(v.get(a))) is not None
            }
            for op, v in fb.items()
            if isinstance(v, dict) and _f(v.get("ms_per_step")) is not None
        }
        add(
            f"step:{kind}",
            {"record_type": RT_STEP, "run_mode": kind, "phase": kind},
            m,
            _n(s.get("n")),
            _extra(
                fallback_ops=_compact(
                    fb, TOP_FALLBACK_OPS, key=lambda kv: -kv[1]["ms_per_step"]
                )
            ),
        )

    kernels = ctx["roof"].get("kernels")
    for k in kernels if isinstance(kernels, list) else []:
        if not isinstance(k, dict):
            continue
        key, phase = k.get("key"), k.get("phase")
        if not (isinstance(key, str) and isinstance(phase, str) and key and phase):
            continue
        m = _measure(k, KERNEL_FIELDS)
        floor, median = _f(k.get("floor_us")), _f(k.get("median_us"))
        if floor is not None and median is not None:
            m["floor_valid"] = [float(median >= floor)]
        if key in ctx["predicted"]:
            m["predicted_us"] = [ctx["predicted"][key]]
        aten = sorted(a for a in _list(k.get("aten")) if isinstance(a, str))
        add(
            f"{phase}:{key}",
            {
                "record_type": RT_KERNEL,
                "run_mode": phase,
                "phase": phase,
                "kernel_name": key,
                "provenance_key": key,
                "family": family(aten),
                "aten_ops": ",".join(aten),
            },
            m,
            _n(k.get("calls")),
            _extra(bound=k.get("bound")),
        )

    comp = _dict(res.get("compile"))
    # by_pass values are {self_s, n}, or bare self seconds in older bundles.
    passes = {
        n: (v.get("self_s"), v.get("n")) if isinstance(v, dict) else (v, 0)
        for n, v in _dict(comp.get("by_pass")).items()
    } or {n: (v, 0) for n, v in _dict(ctx["compile_summary"].get("passes")).items()}
    for name, (self_s, n) in passes.items():
        pname = str(name).removeprefix("pass:")
        x = _f(self_s)
        add(
            f"compile:{pname}",
            {"record_type": RT_COMPILE, "run_mode": "trace", "pass": pname},
            {"self_s": [x]} if x is not None else {},
            _n(n),
        )
    return out


def _duration(ctx: dict) -> float:
    total = sum(
        x
        for p in _dict(ctx["res"].get("phases")).values()
        if (x := _f(_dict(p).get("wall_s"))) is not None
    )
    return total if math.isfinite(total) and total <= MAX_F32 else 0.0


def write(client, db: str, component: str, rid: str, rows: list[dict]) -> int:
    """benchmarks + benchmark_runs for the run; 0 when it already landed."""
    from .writer import benchmarks_already_ingested, insert_benchmarks

    if benchmarks_already_ingested(client, db, rid, component, REPORT_KIND):
        return 0
    return insert_benchmarks(client, db, component, rid, rows, report_kind=REPORT_KIND)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vllm-trace", description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument(
        "--analyzed", help="dir of <label>.res/roof/host.json (default: run_dir)"
    )
    ap.add_argument("--label", default="", help="default: the run dir's name")
    ap.add_argument("--run-key", default="", help="default: <artifact id12>:<label>")
    ap.add_argument("--component", default="", help="default: the artifact's")
    ap.add_argument(
        "--artifact", default="", help="artifact spec, else the bundle's artifact_id"
    )
    ap.add_argument("--trace-bundle-url", default="")
    ap.add_argument("--model")
    ap.add_argument("--tp", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--input-len", type=int)
    ap.add_argument("--output-len", type=int)
    ap.add_argument("--max-num-batched-tokens", type=int)
    ap.add_argument("--max-num-seqs", type=int)
    ap.add_argument("--max-model-len", type=int)
    ap.add_argument(
        "--dry-run", action="store_true", help="read only; print what would be written"
    )
    args = ap.parse_args(argv)
    try:
        ctx = load(
            args.run_dir, args.analyzed, args.label, model=args.model, tp=args.tp,
            batch_size=args.batch_size, input_len=args.input_len, output_len=args.output_len,
            max_num_batched_tokens=args.max_num_batched_tokens, max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
        )  # fmt: skip
        art = ctx["artifact"]
        spec = args.artifact or (
            f"id:{art['artifact_id']}" if art.get("artifact_id") else ""
        )
        if not spec:
            raise TraceError(
                "no artifact: no spyre_artifact.json/template.json/manifest.json, no --artifact"
            )
    except (TraceError, OSError) as err:
        print(json.dumps({"status": "rejected", "reason": str(err)}))
        return 2

    from .client import ClickHouse, ClickHouseEnv, tables_present
    from .schema import ARTIFACT_RESULTS, BENCHMARK_RUNS, BENCHMARKS
    from .writer import DryRunClient, insert_artifact_result

    db = ClickHouseEnv.target_database()
    live = bool(db and ClickHouseEnv.host())
    if not live and not args.dry_run:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "set CLICKHOUSE_HOST and CLICKHOUSE_DB_V2",
                }
            )
        )
        return 1
    client = ClickHouse.connect(database=db) if live else None
    aid, arch, component = (
        art.get("artifact_id", ""),
        art.get("arch", ""),
        art.get("component", ""),
    )
    if client:
        from .resolver import resolve

        try:
            r = resolve(spec, arch, client=client, db=db, lookup="only")
        except ValueError as err:
            r = None
            print(f"[warn] {err}", file=sys.stderr)
        if r is None:
            print(
                json.dumps(
                    {
                        "status": "rejected",
                        "reason": f"artifact {spec!r} not recorded in {db}",
                    }
                )
            )
            return 2
        aid, arch, component = r.artifact_id, arch or r.arch, component or r.component
    component = args.component or component or COMPONENT
    rows = entries(ctx, args.trace_bundle_url)
    if not rows:
        print(
            json.dumps(
                {"status": "rejected", "reason": "no measurements in the bundle"}
            )
        )
        return 2
    key = args.run_key or f"{art.get('id12') or aid[:12]}:{ctx['label']}"
    rid = RunId.derive(REPORT_KIND, key, arch, "perf")
    if not rid and client:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "reason": "arch unresolved: no arch in the artifact block",
                }
            )
        )
        return 2
    report = {
        "run_id": rid, "artifact_id": aid, "artifact_checked": bool(client), "report_kind": REPORT_KIND,
        "rows": dict(sorted(Counter(r["props"]["record_type"] for r in rows).items())),
    }  # fmt: skip
    if not rid:
        report["arch"] = "unresolved"
    if client and not tables_present(
        client, db, tables=(BENCHMARKS, BENCHMARK_RUNS, ARTIFACT_RESULTS)
    ):
        print(
            json.dumps(
                {
                    **report,
                    "status": "failed",
                    "reason": f"{db} lacks the benchmark tables",
                }
            )
        )
        return 1
    out = DryRunClient(client) if args.dry_run else client
    n = write(out, db, component, rid, rows)
    insert_artifact_result(
        out, db, artifact_id=aid, run_id=rid, test_type="perf", state="passed", arch=arch,
        result_kind="performance", duration_s=_duration(ctx),
        props={"source": REPORT_KIND, "trace_bundle_url": args.trace_bundle_url},
    )  # fmt: skip
    if args.dry_run:
        print(out.report(), file=sys.stderr)
    print(
        json.dumps(
            {
                **report,
                "status": "would-ingest"
                if args.dry_run
                else "ingested"
                if n
                else "duplicate",
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
