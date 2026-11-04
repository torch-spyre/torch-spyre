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
"""

import argparse
import glob
import json
import regex as re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import benchmeta
from .identity import RunId

REPORT_KIND = "vllm-trace"
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
}


class TraceError(ValueError):
    """A bundle that cannot be ingested; the message says what is missing."""


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise TraceError(f"unreadable {path}: {err}") from None


def _measure(src: dict, fields: dict[str, str]) -> dict[str, list[float]]:
    return {to: [float(src[k])] for k, to in fields.items() if _num(src.get(k))}


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
        d = _json(p)
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
                t = json.loads(p.read_text().splitlines()[0])["t"]
            except (IndexError, KeyError, ValueError):
                continue
            return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return ""


def _predicted(run: Path) -> dict[str, float]:
    """provenance_key -> predicted_us from the kernel-cost sidecar, when it carries one."""
    out: dict[str, float] = {}
    paths = glob.glob(str(run / "kernel_cost*.jsonl")) + glob.glob(
        str(run / "*" / "kernel_cost*.jsonl")
    )
    for path in sorted(paths):
        for line in Path(path).read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            cost = r.get("cost")
            key = r.get("provenance_key") or r.get("key")
            if key and isinstance(cost, dict) and _num(cost.get("predicted_us")):
                out[key] = float(cost["predicted_us"])
    return out


ENGINE_LIMITS = ("max_num_batched_tokens", "max_num_seqs", "max_model_len")


def _engine_limits(run: Path, args: dict, batch_size, given: dict) -> dict[str, str]:
    """Engine limits: flag, else vllm.log's `non-default args` line, else e2e.json (budget 0 = default)."""
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
                        break
            break
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
    label = label or run.name
    found = {}
    for kind in ("res", "roof", "host"):
        for p in (ana / f"{label}.{kind}.json", ana / f"{kind}.json"):
            if p.is_file():
                found[kind] = _json(p)
                break
    if not found:
        raise TraceError(f"no {label}.res/roof/host.json in {ana}")
    host = found.get("host")
    if isinstance(host, list):
        host = host[0] if host else None
    art, env = _artifact(run)
    args = (_first_json(run, "e2e.json") or {}).get("args", {})
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
    w.update(_engine_limits(run, args, w["batch_size"], workload))
    return {
        "label": label,
        "res": found.get("res"),
        "roof": found.get("roof"),
        "host": host,
        "artifact": art,
        "env": env,
        "workload": w,
        "predicted": _predicted(run),
        "run_utc": _run_utc(run),
        "compile_summary": _first_json(run, "compile_summary.json"),
    }


def run_props(ctx: dict, bundle_url: str = "") -> dict[str, str]:
    """Props on every row: workload, engine limits, env fingerprint, artifact_id12, compile totals, run time."""
    art = ctx["artifact"]
    w = ctx["workload"]
    out = benchmeta.flatten("env", {**ctx["env"], "artifact": art})
    res = ctx["res"] or {}
    comp = res.get("compile") or ctx["compile_summary"] or {}
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
        *((k, w.get(k)) for k in ENGINE_LIMITS),
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

    def add(suffix, props, measurements, iterations=0, extra=None):
        if not measurements:
            return
        props = {**shared, **props}
        out.append(
            {
                "name": f"{config}:{suffix}",
                "tags": [],
                "backend": BACKEND,
                "props": props,
                "measurements": measurements,
                "iterations": int(iterations),
                "run_props": {**rprops, "run_mode": props["run_mode"], **(extra or {})},
                "disc": props,
                "disc_keys": ID_KEYS,
            }
        )

    res = ctx["res"] or {}
    for phase, p in (res.get("phases") or {}).items():
        add(
            f"phase:{phase}",
            {"record_type": "trace_phase", "run_mode": "trace", "phase": phase},
            _measure(p, PHASE_FIELDS),
            p.get("n_samples") or 0,
        )

    host = ctx["host"] or {}
    steps = res.get("steps") or {}
    for kind in ("prefill", "decode"):
        s = host.get(f"{kind}_summary") or steps.get(kind) or {}
        m = _measure(s, STEP_FIELDS)
        for cls, ms in (s.get("waterfall_ms_per_step") or {}).items():
            if _num(ms):
                m[f"host_{cls}_ms"] = [float(ms)]
        add(
            f"step:{kind}",
            {"record_type": "trace_step", "run_mode": kind, "phase": kind},
            m,
            s.get("n") or 0,
        )

    for k in (ctx["roof"] or {}).get("kernels") or []:
        key, phase = k.get("key"), k.get("phase")
        if not (key and phase):
            continue
        m = _measure(k, KERNEL_FIELDS)
        if _num(k.get("floor_us")) and _num(k.get("median_us")):
            m["floor_valid"] = [float(k["median_us"] >= k["floor_us"])]
        if key in ctx["predicted"]:
            m["predicted_us"] = [ctx["predicted"][key]]
        aten = sorted(k.get("aten") or [])
        add(
            f"{phase}:{key}",
            {
                "record_type": "trace_kernel",
                "run_mode": phase,
                "phase": phase,
                "kernel_name": key,
                "provenance_key": key,
                "family": family(aten),
                "aten_ops": ",".join(aten),
            },
            m,
            k.get("calls") or 0,
            {"bound": k["bound"]} if k.get("bound") else None,
        )

    comp = res.get("compile") or {}
    passes = {
        n: (v.get("self_s"), v.get("n") or 0)
        for n, v in (comp.get("by_pass") or {}).items()
    } or {
        n: (v, 0)
        for n, v in ((ctx["compile_summary"] or {}).get("passes") or {}).items()
    }
    for name, (self_s, n) in passes.items():
        pname = name.removeprefix("pass:")
        add(
            f"compile:{pname}",
            {"record_type": "compile_pass", "run_mode": "trace", "pass": pname},
            {"self_s": [float(self_s)]} if _num(self_s) else {},
            n,
        )
    return out


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
    except TraceError as err:
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
    report = {
        "run_id": rid, "artifact_id": aid, "artifact_checked": bool(client), "report_kind": REPORT_KIND,
        "rows": dict(sorted(Counter(r["props"]["record_type"] for r in rows).items())),
    }  # fmt: skip
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
        result_kind="performance", duration_s=sum(
            p["wall_s"] for p in ((ctx["res"] or {}).get("phases") or {}).values() if _num(p.get("wall_s"))
        ),
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
