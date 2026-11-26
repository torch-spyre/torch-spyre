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

"""spyre-bench run: execute a plan's cases, keep the raw outputs, build Results, hand off to the ingest.

  spyre-bench run --plan plan.json [--case NAME] [--out DIR] [--ingest] [--store clickhouse] [--dry-run]

The engine writes no database rows: ingestion is a subprocess (the ingest) and the rows-landed check only reads.
Results directory, tokens, hand-off and exit codes: docs/README.md ("Running a plan").
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import shlex
import selectors
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from spyre_bench import (
    cliutil,
    engine,
    ingest_adapter,
    pinning_plan,
    pinning_run,
    placement,
    planfile,
    quality,
    store as stores,
)
from spyre_bench.cliutil import UsageError

POLL_S = 0.5
DEFAULT_TIMEOUT_S = 3600
LANDED_POLL_S = 5
CASE_LABEL = "bench.case"
INGEST_BASE = ["{python}", "-I", "-m", "spyre_clickhouse_ingest", "results"]
# perf-kernel takes the engine's --run-id verbatim; the bundle profiles derive their own run_id from the run key.
BUNDLE_TAIL = [
    "--artifact-id",
    "{subject_id}",
    "--arch",
    "{arch}",
    "--trigger-type",
    "{test_type}",
    "--jenkins-run-key",
    "{run_key}",
    "--schema",
    "v2",
]
INGEST_DEFAULTS = {
    "perf-kernel": INGEST_BASE
    + ["--xml-dir", "{raw_dir}", "--run-id", "{run_id}", "--artifact-id", "{subject_id}", "--arch", "{arch}"]
    + ["--trigger-type", "perf", "--schema", "v2"],
    # a vLLM bundle's component is fixed by the ingest; a generic benchmarks bundle names its own
    "vllm": INGEST_BASE + ["--vllm-results-dir", "{raw_dir}"] + BUNDLE_TAIL,
    "vllm-sweep": INGEST_BASE + ["--vllm-results-dir", "{raw_dir}"] + BUNDLE_TAIL,
    "generic": INGEST_BASE + ["--benchmarks-dir", "{raw_dir}", "--component", "{component}"] + BUNDLE_TAIL,
}
THREADS_RUN_ID = ("perf-kernel",)
ACCEPTED = ("ingested", "duplicate", "would-ingest")  # report statuses that mean the rows are (or already were) in
MIB = 1 << 20
DEFAULT_LOG_BYTES = 16 * MIB
DEFAULT_METRICS_BYTES = 8 * MIB
DEFAULT_RAW_WARN_BYTES = 1024 * MIB
RUN_PHASE_TOKENS = planfile.RUN_TOKENS


def now():
    return datetime.now(timezone.utc)


class EngineSkew(UsageError):
    pass


def load_plan(path):
    try:
        with open(path) as f:
            plan = json.load(f)
    except (OSError, ValueError) as e:
        raise UsageError(f"plan {path}: {e}") from None
    if not isinstance(plan, dict) or not isinstance(plan.get("cases"), list):
        raise UsageError(f"plan {path}: not a plan document (no cases list)")
    try:
        planfile.check_skew(plan)
    except planfile.SkewError as e:
        raise EngineSkew(f"engine_skew: {e}") from None
    check_records(plan)
    return plan


def check_records(plan):
    """Every record must carry what run reads; a hand-edited or truncated plan fails here, before anything runs."""
    seen = set()
    for n, rec in enumerate(plan["cases"]):
        where = f"plan case #{n}"
        if not isinstance(rec, dict) or not all(
            isinstance(rec.get(k), str) and rec[k] for k in ("name", "benchmark_id")
        ):
            raise UsageError(f"{where}: needs a name and a benchmark_id")
        if rec["name"] in seen:
            raise UsageError(f"{where}: case name {rec['name']!r} appears twice")
        seen.add(rec["name"])
        if rec.get("action") not in ("run", "reuse", "skip"):
            raise UsageError(f"{where} ({rec['name']}): action must be run, reuse or skip")
        if rec["action"] != "run":
            continue
        repeats = rec.get("repeats")
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            raise UsageError(f"{where} ({rec['name']}): repeats must be an integer >= 1")
        if rec.get("cache") not in ("warm", "cold", "both"):
            raise UsageError(f"{where} ({rec['name']}): cache must be warm, cold or both")
        for key in ("command", "ingest"):
            cmd = rec.get(key)
            if cmd is not None and not (isinstance(cmd, list) and cmd and all(isinstance(x, str) for x in cmd)):
                raise UsageError(f"{where} ({rec['name']}): {key} must be a non-empty list of strings or null")
            if cmd and any("<redacted>" in x for x in cmd):
                raise UsageError(
                    f"{where} ({rec['name']}): {key} holds <redacted>: this is a redacted copy, not a runnable plan"
                )
        minutes = rec.get("timeout_min")
        if (
            not isinstance(minutes, (int, float))
            or isinstance(minutes, bool)
            or not math.isfinite(minutes)
            or minutes <= 0
        ):
            raise UsageError(f"{where} ({rec['name']}): timeout_min must be a finite number > 0")
        pin = rec.get("pinning")
        if pin is not None and not (isinstance(pin, dict) and isinstance(pin.get("policy", "none"), str)):
            raise UsageError(f"{where} ({rec['name']}): pinning must be a mapping with a string policy or null")
        if (pin or {}).get("layout") == "ranks" and not any(
            "{" + pinning_run.WRAPPER_TOKEN + "}" in str(w) for w in rec.get("command") or ()
        ):
            raise UsageError(
                f"{where} ({rec['name']}): cpu_pinning layout ranks needs the command to run behind {{pin_wrapper}}"
            )
        labels = rec.get("labels")
        if labels is not None and not (
            isinstance(labels, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in labels.items())
        ):
            raise UsageError(f"{where} ({rec['name']}): labels must map strings to strings")
        if not isinstance(rec.get("allow_empty", False), bool):
            raise UsageError(f"{where} ({rec['name']}): allow_empty must be true or false")


def select_cases(plan, wanted):
    """The records to execute: every `run` record, or those named with --case (which must be `run` records)."""
    by = {r["name"]: r for r in plan["cases"]}
    for name in wanted:
        if name not in by:
            raise UsageError(f"--case {name!r} is not in the plan")
        if by[name]["action"] != "run":
            raise UsageError(f"--case {name!r} is planned as {by[name]['action']}, not run")
    return [r for r in plan["cases"] if r["action"] == "run" and (not wanted or r["name"] in wanted)]


def repeat_subdir(repeat):
    """The one spelling of a repeat's directory name, shared by raw_repeat_dir and the repeat dir."""
    return f"repeat-{int(repeat):03d}"


def repeat_schedule(rec):
    """[(repeat, cache)]: engine-driven repeats, cold then warm for cache both; numbering is continuous from 1."""
    caches = ["cold", "warm"] if rec.get("cache") == "both" else [rec.get("cache") or "warm"]
    return [(i + 1, c) for i, c in enumerate(c for c in caches for _ in range(int(rec["repeats"])))]


def run_values(rec, ctx, repeat, cache, raw_dir, repeat_dir):
    return {
        "run_id": ctx["run_id"],
        "subject_id": ctx["subject_id"] or "",
        "arch": ctx["arch"],
        "repeat": repeat,
        "repeats": len(repeat_schedule(rec)),
        "cache": cache,
        "profile_level": rec.get("profile_level") or "off",
        "raw_dir": raw_dir,
        # Under raw_dir, so fixed output filenames stay per-repeat and still reach the ingest.
        "raw_repeat_dir": str(Path(raw_dir) / repeat_subdir(repeat)),
        "repeat_dir": repeat_dir,
        "out_dir": ctx["out"],
        "python": sys.executable,
        "pin_wrapper": str(Path(repeat_dir) / pinning_run.WRAPPER_NAME),
    }


def command_env(rec, vals, quiet, pass_env=(), pass_prefixes=()):
    return {
        **planfile.benchmark_env(os.environ, pass_env, pass_prefixes),
        "BENCH_RUN_ID": vals["run_id"],
        "BENCH_CASE": rec["name"],
        "BENCH_REPEAT": str(vals["repeat"]),
        "BENCH_REPEATS": str(vals["repeats"]),
        "BENCH_CACHE": vals["cache"],
        "BENCH_PROFILE_LEVEL": vals["profile_level"],
        "BENCH_RAW_DIR": vals["raw_dir"],
        "BENCH_RAW_REPEAT_DIR": vals["raw_repeat_dir"],
        "BENCH_REPEAT_DIR": vals["repeat_dir"],
        "BENCH_SUBJECT_ID": vals["subject_id"],
        **({"BENCH_QUIET": "1"} if quiet else {}),
    }


def kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()
    proc.wait()


def reap_group(proc):
    """Kill stragglers a finished command left in its session, so nothing outlives its repeat on the device."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


def worst_pinning(samples):
    """The sample a verdict should rest on: a violation, else an unobservable one, else a verified one."""
    for want in ("false", "unknown", "true"):
        for s in samples:
            if s.get("run.exec.pinning_verified") == want:
                return s
    return None


class Capture(threading.Thread):
    """Copies a command's pipe to a file, keeping at most `cap` bytes and a marker; always drains the pipe.

    A failing write (a full disk) is remembered in `error` and the pipe keeps draining, so the command never blocks.
    Once `stop` is set the thread exits as soon as no data is waiting, which also frees us from a detached process that
    still holds the pipe open. `last_json` keeps the last line that looks like a JSON object, whatever the cap.
    """

    def __init__(self, stream, path, cap, exclusive=False):
        super().__init__(daemon=True)
        self.stream, self.path, self.cap, self.exclusive = stream, path, cap, exclusive
        self.truncated, self.error, self.last_json = False, None, None
        self.stop = threading.Event()
        self._line = b""

    def _scan(self, chunk):
        lines = (self._line + chunk).split(b"\n")
        self._line = lines.pop()[: 1 << 20]
        for line in lines:
            self._remember(line)

    def _remember(self, line):
        """Keep the line when it parses as a JSON object, so a trailing non-JSON line cannot hide the report."""
        if line.lstrip().startswith(b"{"):
            text = line[: 1 << 20].decode("utf-8", "replace")
            try:
                if isinstance(json.loads(text), dict):
                    self.last_json = text
            except ValueError:
                pass

    def run(self):
        fd, written, f = self.stream.fileno(), 0, None
        extra = os.O_EXCL if self.exclusive else 0
        try:
            f = open(self.path, "wb", opener=lambda path, flags: os.open(path, flags | extra | os.O_NOFOLLOW, 0o600))
        except OSError as e:
            self.error = e
        selector = selectors.DefaultSelector()
        selector.register(fd, selectors.EVENT_READ)
        try:
            while True:
                if not selector.select(0.1):
                    if self.stop.is_set():
                        break
                    continue
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                self._scan(chunk)
                keep = chunk[: max(0, self.cap - written)]
                written += len(keep)
                self.truncated = self.truncated or len(keep) < len(chunk)
                if f is not None and keep and self.error is None:
                    try:
                        f.write(keep)
                    except OSError as e:
                        self.error = e
            self._remember(self._line)
            if f is not None and self.truncated and self.error is None:
                try:
                    f.write(f"\n[spyre-bench: output truncated at {self.cap} bytes]\n".encode())
                except OSError as e:
                    self.error = e
        finally:
            selector.close()
            if f is not None:
                f.close()


def finish_pumps(proc, pumps):
    """Let the pumps drain what is waiting and leave; close the pipes unless a detached process still holds one."""
    for pump in pumps:
        pump.stop.set()
    for pump in pumps:
        pump.join(timeout=2)
    if not any(pump.is_alive() for pump in pumps):
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


def execute_command(
    argv,
    env,
    cwd,
    out_path,
    err_path,
    timeout_s,
    pin,
    device_numa,
    popen,
    max_log=DEFAULT_LOG_BYTES,
    truncated=None,
    markers=None,
    sys_root="/sys",
):
    """(returncode | None, reason, pinning props | None). Logs are capped; the group is killed on timeout or interrupt.

    `pin` is the request the process tree is judged against; `markers` (wrapper paths, possibly empty) tells the observer
    the engine launched the tree under a pinning tool, so samples taken while it is still that tool are skipped.

    `truncated`, when given, receives "stdout" / "stderr" for each capped log. A log that could not be written turns a
    zero exit into reason `log_write_failed:<errno>`; output a detached process writes after the command exits is dropped.
    """
    samples = []
    try:
        proc = popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as e:
        return None, f"spawn_failed:{type(e).__name__}", None
    pumps = [Capture(proc.stdout, out_path, max_log), Capture(proc.stderr, err_path, max_log)]
    for pump in pumps:
        pump.start()
    started = placement.starttime(proc.pid) if pin else None
    deadline = time.monotonic() + timeout_s
    result = None
    try:
        while result is None:
            if pin:
                sample = placement.verify_pinning(
                    pin, proc.pid, device_numa, sys_root=sys_root, expected_starttime=started, markers=markers
                )
                if sample is not None:
                    samples.append(sample)
            try:
                rc = proc.wait(timeout=max(0.01, min(POLL_S, deadline - time.monotonic())))
                result = (rc, "" if rc == 0 else f"exit_{rc}")
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    kill_group(proc)
                    result = (None, "timeout")
    except BaseException:
        kill_group(proc)  # an interrupt or SIGTERM to spyre-bench must not leave the benchmark on the device
        raise
    finally:
        reap_group(proc)
        finish_pumps(proc, pumps)
    if truncated is not None:
        truncated += [name for name, pump in zip(("stdout", "stderr"), pumps) if pump.truncated]
    failed = next((pump.error for pump in pumps if pump.error), None)
    if failed and result[0] == 0:
        result = (0, f"log_write_failed:{failed.errno}")
    return result[0], result[1], worst_pinning(samples)


def finite_floats(vals):
    out = []
    for v in vals:
        try:
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                out.append(float(v))
        except OverflowError:  # an int too large for a float
            continue
    return out


def clean_samples(raw):
    """Metric -> finite floats from a metrics.json `samples` mapping; (samples, notes)."""
    out, notes = {}, []
    if not isinstance(raw, dict):
        return out, ["samples is not a mapping"]
    for metric, vals in raw.items():
        good = finite_floats(vals) if isinstance(vals, list) else []
        if good and len(good) == len(vals):
            out[str(metric)] = good
        else:
            notes.append(f"metric {metric}: dropped (not a list of finite numbers)")
    return out, notes


def read_sidecar(repeat_dir, max_bytes=DEFAULT_METRICS_BYTES):
    """(samples, props, warmup_discarded, notes) from the optional metrics.json the command wrote."""
    path = Path(repeat_dir) / "metrics.json"
    if not path.is_file():
        return {}, {}, 0, []
    if path.stat().st_size > max_bytes:
        return {}, {}, 0, [f"metrics.json ignored: {path.stat().st_size} bytes exceeds the {max_bytes} byte cap"]
    try:
        doc = json.loads(path.read_text())
        if not isinstance(doc, dict):
            raise ValueError("not a mapping")
    except (OSError, ValueError) as e:
        return {}, {}, 0, [f"metrics.json unreadable: {e}"]
    samples, notes = clean_samples(doc.get("samples") or {})
    props = {str(k): str(v) for k, v in (doc.get("props") or {}).items()} if isinstance(doc.get("props"), dict) else {}
    warm = doc.get("warmup_discarded")
    return samples, props, warm if isinstance(warm, int) and warm >= 0 else 0, notes


def run_case(rec, ctx, popen):
    """Execute every repeat of one case; returns the case outcome (see run.json)."""
    out = ctx["out"]
    case_dir = Path(out) / "cases" / planfile.slug(rec["name"])
    planfile.ensure_dirs(out, case_dir)  # links planted in cases/ or cases/<slug> are removed, never followed
    for stale in [case_dir / "raw", case_dir / "result.json", *case_dir.glob("repeat-*")]:
        if stale.is_symlink() or stale.is_file():
            stale.unlink()  # run owns these paths: output of an earlier run must never pass as this one's
        elif stale.is_dir():
            shutil.rmtree(stale)
    raw_dir = case_dir / "raw"
    planfile.ensure_dirs(out, raw_dir)
    case_record = planfile.dump(
        {**rec, **{k: planfile.redact_argv(rec[k]) for k in ("command", "ingest") if rec.get(k)}}
    )
    planfile.write_file(out, case_dir / "case.json", case_record)
    env_dropped = planfile.env_filter(os.environ, ctx["pass_env"], ctx["pass_prefixes"])[1]
    schedule = repeat_schedule(rec)
    outcome = {
        "name": rec["name"],
        "benchmark_id": rec["benchmark_id"],
        "id_mode": rec.get("id_mode", "engine"),
        "id_profile": rec.get("id_profile"),
        "dir": str(case_dir.relative_to(ctx["out"])),
        "repeats": {"planned": len(schedule), "ok": 0, "failed": 0},
        "status": "failed",
        "reason": "",
        "result": None,
        "_repeats": [],
        "_notes": [],
    }
    if not rec.get("command"):
        outcome["reason"] = "no_command"
        return outcome
    timeout_s = (float(rec.get("timeout_min") or 0) * 60) or DEFAULT_TIMEOUT_S
    if ctx["timeout"]:
        timeout_s = min(timeout_s, ctx["timeout"])
    device_numa = (ctx["env_file"].get("fingerprint") or {}).get("env.fp.device.numa_node", "")
    try:
        pinned = pinning_run.prepare(rec.get("pinning"), device_numa, ctx["sys_root"])
    except pinning_plan.PinningError as e:
        raise UsageError(f"case {rec['name']}: {e}") from None
    outcome["_notes"] += pinned.notes
    pin = pinned.observer_request() if pinned.observes else None
    uses_wrapper = any("{" + pinning_run.WRAPPER_TOKEN + "}" in str(w) for w in rec["command"])
    # Only for a command that asks for it: raw/ stays untouched for every other case.
    uses_raw_repeat = any("{raw_repeat_dir}" in str(w) for w in rec["command"])
    sidecar_seen = False
    for repeat, cache in schedule:
        rdir = case_dir / repeat_subdir(repeat)
        planfile.fresh_dir(out, rdir)
        vals = run_values(rec, ctx, repeat, cache, str(raw_dir), str(rdir))
        if uses_raw_repeat:
            planfile.ensure_dirs(out, Path(vals["raw_repeat_dir"]))
        try:
            argv = planfile.render_tokens(rec["command"], vals, RUN_PHASE_TOKENS, f"case {rec['name']}")
        except ValueError as e:
            raise UsageError(str(e)) from None
        argv = pinned.wrap(argv, rdir, uses_wrapper)
        started = now()
        capped = []
        rc, reason, pinning = execute_command(
            argv,
            command_env(rec, vals, ctx["quiet"], ctx["pass_env"], ctx["pass_prefixes"]),
            ctx["workdir"],
            rdir / "stdout.log",
            rdir / "stderr.log",
            timeout_s,
            pin,
            device_numa,
            popen,
            max_log=ctx["max_log"],
            truncated=capped,
            markers=pinned.markers(rdir),
            sys_root=ctx["sys_root"],
        )
        pinning = pinned.recorded(pinning)
        ended = now()
        samples, props, warm, notes = read_sidecar(rdir, ctx["max_metrics"])
        sidecar_seen = sidecar_seen or (rdir / "metrics.json").is_file()
        notes += [f"{name} log truncated at {ctx['max_log']} bytes" for name in capped]
        notes += [
            f"{name} withheld from the command ({why}); name it with --pass-env to pass it"
            for name, why in env_dropped.items()
        ]
        status = "ok" if rc == 0 and not reason else "failed"
        info = {
            "case": rec["name"],
            "repeat": repeat,
            "cache": cache,
            "argv": planfile.redact_argv(argv),
            "returncode": rc,
            "status": status,
            "reason": reason,
            "started_at": cliutil.utc_stamp(started),
            "ended_at": cliutil.utc_stamp(ended),
            "duration_s": round((ended - started).total_seconds(), 3),
            **({"pinning": pinning} if pinning else {}),
            **({"notes": notes} if notes else {}),
        }
        planfile.write_file(out, rdir / "status.json", planfile.dump(info))
        outcome["repeats"]["ok" if status == "ok" else "failed"] += 1
        outcome["_repeats"].append(
            {
                "run.repeat": repeat,
                "status": status,
                "samples": samples if status == "ok" else {},
                "props": {**props, "run.cache": cache, **(pinning or {})},
                "warmup_discarded": warm,
            }
        )
    failed = outcome["repeats"]["failed"]
    outcome["status"] = "failed" if failed else "ok"
    outcome["reason"] = "repeats_failed" if failed else ""
    planfile.write_file(out, case_dir / "case.json", case_record)  # a benchmark may have replaced it meanwhile
    planfile.ensure_dirs(out, raw_dir)
    # Files, not entries: the per-repeat subdirs exist before the command runs, so an empty tree has entries.
    empty = not any(f.is_file() for f in raw_dir.rglob("*"))
    if not failed and empty and (ctx["ingest"] or not (sidecar_seen or rec.get("allow_empty"))):
        outcome.update(status="failed", reason="no_output")  # exit 0 with nothing recorded is not a run
    size = sum(f.stat().st_size for f in raw_dir.rglob("*") if f.is_file())
    if size > ctx["max_raw"]:
        outcome["_notes"] += [f"raw output is {size} bytes, over the {ctx['max_raw']} byte warning threshold"]
    return outcome


def cv_pct(repeats):
    """Largest coefficient of variation (percent) of the per-repeat medians across metrics, or None below two repeats.

    Cold repeats are left out of any metric that also has warm ones, matching the aggregate.
    """
    per = engine.per_repeat_values([r for r in repeats if r["status"] == "ok"])
    cvs = [
        statistics.stdev(v) / abs(statistics.fmean(v)) * 100 for v in per.values() if len(v) > 1 and statistics.fmean(v)
    ]
    return max(cvs) if cvs else None


def build_case_result(cat, rec, outcome, ctx):
    """The engine Result of one case, or (None, note). Unknown facts (qualification, cleanliness) lower validity."""
    case = next((c for c in cat.cases if c["name"] == rec["name"]), None)
    if case is None:
        return None, "case not in the catalog"
    if not ctx["subject_id"]:
        return None, "no subject artifact in the plan"
    envf = ctx["env_file"]
    keys = cat.reg.fingerprint_keys()
    fingerprint = {k: str(v) for k, v in (envf.get("fingerprint") or {}).items() if k in keys}
    node = envf.get("node") or os.environ.get("NODE_NAME") or ""
    placement_ = {"nodes": [node] if node else []}
    clean = envf.get("clean") if isinstance(envf.get("clean"), dict) else {}
    reps = [
        {
            "run.repeat": r["run.repeat"],
            "status": r["status"],
            "placement": placement_,
            "clean": clean,
            "props": r["props"],
            "samples": r["samples"],
        }
        for r in outcome["_repeats"]
    ]
    purpose = ctx["plan"].get("purpose") or ""
    validity = None
    pol_doc = cat.policy_of(case)
    policy = pol_doc.spec if pol_doc else None
    if policy and purpose in policy["purposes"] and case["type"] == "quality":
        ok = [r for r in outcome["_repeats"] if r["status"] == "ok"]
        want = quality.expected(case)
        state, reasons = quality.validity_of(
            policy, purpose, [quality.repeat_facts(r, want) for r in ok], measured=any(r["samples"] for r in ok)
        )
        validity = (state, reasons, purpose, "internal")
    elif policy and purpose in policy["purposes"]:
        ok = [r for r in outcome["_repeats"] if r["status"] == "ok"]
        warm_ok = [r for r in ok if r["props"].get("run.cache") != "cold"]  # cold repeats discard nothing by design
        facts = {
            "measured": any(r["samples"] for r in ok),
            "gates_pass": True,
            "qualified": envf.get("qualified") is True,
            "clean": clean,
            "repeats": len(warm_ok or ok),
            "cv_pct": cv_pct(outcome["_repeats"]),
            "warmup_discarded": min([r["warmup_discarded"] for r in warm_ok or ok] or [0]),
            "placement": bool(placement_["nodes"]),
        }
        state, reasons = engine.validity_of(policy, purpose, facts)
        validity = (state, reasons, purpose, "internal")
    env_name = rec.get("environment")
    result = engine.build_result(
        cat,
        case,
        f"{ctx['run_id']}.{planfile.slug(rec['name'])}",
        {"artifact_id": ctx["subject_id"], **({"attrs": ctx["subject_attrs"]} if ctx["subject_attrs"] else {})},
        fingerprint,
        envf.get("observations") or {},
        rec.get("labels") or {},
        reps,
        request={"name": env_name, "spec_digest": engine.request_digest(cat, env_name, rec.get("pinning"))}
        if env_name
        else None,
        validity=validity,
        pinning=rec.get("pinning"),
    )
    result["metadata"]["labels"] = {CASE_LABEL: rec["name"]}  # the Result is named after its run; analyses name cases
    dropped = sorted(set(envf.get("fingerprint") or {}) - set(fingerprint))
    return result, (f"fingerprint keys not in the registry were dropped: {dropped}" if dropped else "")


def check_result(result):
    errs = engine.shape_schema("result").errors(result)
    try:
        return errs + engine.recompute(result)
    except ingest_adapter.IngestUnavailable:
        return errs  # the stored id cannot be re-derived here; the case's own id was checked against the plan


def resolve_ingest(rec, override, ctx=None):
    if override:
        return shlex.split(override)
    if rec.get("ingest"):
        return rec["ingest"]
    argv = list(INGEST_DEFAULTS.get(rec.get("id_profile") or "generic", INGEST_DEFAULTS["generic"]))
    if ctx and ctx.get("ingest_pythonpath") and "-I" in argv:  # -I would ignore the PYTHONPATH we are about to set
        i = argv.index("-I")
        argv[i : i + 1] = ["-s", "-P"]
    return argv


def ingest_commands(outcomes, recs, ctx):
    """Distinct ingest argvs (token-resolved) with the case names each covers; only for cases that are ok.

    Each case gets its own run key (the ingest rejects a second bundle under one key), so bundle profiles yield one
    invocation per case. "default" marks argv built from the profile default; "threads" says the engine's run id is
    the stored one.
    """
    seen = {}
    for o in outcomes:
        if o["status"] != "ok":
            continue
        rec = recs[o["name"]]
        profile = rec.get("id_profile") or "generic"
        vals = {
            "python": ctx.get("ingest_python") or sys.executable,
            "run_id": ctx["run_id"],
            "run_key": f"manual:spyre-bench:{ctx['run_id']}-{planfile.case_digest(rec['name'])}",
            "subject_id": ctx["subject_id"],
            "arch": ctx["arch"],
            "component": ctx["component"],
            "raw_dir": str(Path(ctx["out"]) / o["dir"] / "raw"),
            "out_dir": ctx["out"],
            "case": rec["name"],
            "profile": profile,
            "test_type": rec.get("test_type") or "perf",
        }
        custom = bool(ctx["ingest_cmd"] or rec.get("ingest"))
        try:
            argv = planfile.render_tokens(
                resolve_ingest(rec, ctx["ingest_cmd"], ctx), vals, planfile.INGEST_TOKENS, f"ingest {rec['name']}"
            )
        except ValueError as e:
            raise UsageError(str(e)) from None
        entry = seen.setdefault(
            tuple(argv), {"cases": [], "default": not custom, "threads": custom or profile in THREADS_RUN_ID}
        )
        entry["cases"].append(rec["name"])
    return [{"argv": list(k), **v} for k, v in seen.items()]


def report_of(text):
    """The ingest's report: the last line of its stdout that is a JSON object ({} when there is none)."""
    for line in reversed((text or "").splitlines()):
        if not line.lstrip().startswith("{"):
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if isinstance(doc, dict):
            return doc
    return {}


def accepted_report(doc):
    """True for a report that names the run its rows are stored under and says they are (or already were) in."""
    return isinstance(doc.get("run_id"), str) and doc["run_id"] != "" and doc.get("status") in ACCEPTED


def fresh_ingest_dirs(out):
    """(log dir, cwd, home): new, empty and mode 0700, made after the last benchmark so none of it can be planted.

    Whatever is already at <out>/ingest (a directory, a file, a link) is removed without following it.
    """
    root = Path(out) / "ingest"
    if root.is_symlink() or root.is_file():
        root.unlink()
    elif root.exists():
        shutil.rmtree(root)
    root.mkdir(mode=0o700, parents=True)
    return root, Path(tempfile.mkdtemp(prefix="cwd-", dir=root)), Path(tempfile.mkdtemp(prefix="home-", dir=root))


def run_ingest(cmds, ctx):
    """Run each ingest argv in a fresh empty directory with the ingest's own environment and a fresh HOME.

    Every entry records its return code, report and the run id its rows are stored under; a rejected report, or a
    default bundle ingest that exits 0 without one, is an error. A custom command is trusted code: it gets the same
    environment and directory, nothing more.
    """
    log_dir, cwd, home = fresh_ingest_dirs(ctx["out"])
    env = planfile.ingest_env(os.environ, ctx.get("ingest_pass_env", ()), str(home))
    if ctx.get("ingest_pythonpath"):
        env["PYTHONPATH"] = os.pathsep.join(ctx["ingest_pythonpath"])
    done = []
    for n, cmd in enumerate(cmds, 1):
        log, err_log = log_dir / f"ingest-{n}.log", log_dir / f"ingest-{n}.stderr.log"
        rc, last = None, None
        try:
            proc = subprocess.Popen(
                cmd["argv"],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError:
            proc = None
        if proc is not None:
            cap = ctx.get("max_log", DEFAULT_LOG_BYTES)
            pumps = [Capture(proc.stdout, log, cap, exclusive=True), Capture(proc.stderr, err_log, cap, exclusive=True)]
            for pump in pumps:
                pump.start()
            try:
                rc = proc.wait(timeout=ctx["ingest_timeout"])
            except subprocess.TimeoutExpired:
                kill_group(proc)
            except BaseException:
                kill_group(proc)
                raise
            finally:
                reap_group(proc)
                finish_pumps(proc, pumps)
            last = pumps[0].last_json
        report = report_of(last)
        good = rc == 0 and accepted_report(report)
        if good:
            stored = report["run_id"]
        else:
            stored = ctx["run_id"] if rc == 0 and cmd.get("threads", True) and not report else None
        entry = {
            "argv": planfile.redact_argv(cmd["argv"]),
            "cases": cmd["cases"],
            "returncode": rc,
            "log": str(log.relative_to(ctx["out"])),
            "stored_run_id": stored,
            **(
                {"bundle_sha256": report["bundle_sha256"]}
                if good and isinstance(report.get("bundle_sha256"), str)
                else {}
            ),
        }
        if rc == 0 and report and not good:
            entry["error"] = f"report_{report.get('status', 'unreadable')}"
        elif rc == 0 and stored is None:
            entry["error"] = "no_report"  # a bundle ingest must say which run its rows landed under
        if (rc != 0 or entry.get("error")) and report.get("reason"):
            entry["reason"] = str(report["reason"])[:300]
        done.append(entry)
    return done


def check_landed(st, groups, wait, sleep=time.sleep, clock=time.monotonic):
    """rows_landed block. groups maps a stored run id to the outcomes ingested under it.

    Ingest-mode cases: every benchmark_id must be stored. Engine-mode cases cannot know the ingest's ids, so the run
    must hold at least as many other benchmark ids as there are such cases (a partial ingest fails).
    """
    deadline = clock() + wait
    while True:
        expected, missing, short = set(), [], []
        for run_id, outcomes in groups.items():
            ok = [o for o in outcomes if o["status"] == "ok"]
            want = {o["benchmark_id"] for o in ok if o["id_mode"] == "ingest"}
            engine_mode = sum(o["id_mode"] != "ingest" for o in ok)
            got = st.landed(run_id)
            expected |= want
            missing += sorted(want - got)
            if len(got - want) < engine_mode:
                short.append(f"{run_id}: {len(got - want)} rows for {engine_mode} cases")
        if not missing and not short:
            return {"status": "ok", "reason": "", "expected": sorted(expected), "missing": []}
        if clock() >= deadline:
            reason = "; ".join(
                ([f"{len(missing)} expected benchmark ids absent"] if missing else [])
                + [f"too few: {s}" for s in short]
            )
            return {"status": "no_rows_landed", "reason": reason, "expected": sorted(expected), "missing": missing}
        sleep(LANDED_POLL_S)


def exit_of(manifest):
    """(code, reason): precedence invalid > ingest/store failure > failed cases or no rows > nothing ran > ok."""
    c = manifest["counts"]
    if manifest["ingest"]["status"] == "failed":
        return cliutil.EXIT_STORE, "ingest_failed"
    if manifest["rows_landed"]["status"] == "store_error":
        return cliutil.EXIT_STORE, "store_error"
    if c["failed"]:
        return cliutil.EXIT_FAILED, "cases_failed"
    if manifest["rows_landed"]["status"] == "no_rows_landed":
        return cliutil.EXIT_FAILED, "no_rows_landed"
    if any(manifest["result_checks"].values()):
        return cliutil.EXIT_FAILED, "result_invalid"
    if c["run"] == 0 and not (c["reuse"] and not c["skip"]):
        return cliutil.EXIT_NOTHING, "nothing_ran"
    return cliutil.EXIT_OK, "ok"


def raise_terminated(*_):
    raise cliutil.Terminated


@contextlib.contextmanager
def terminate_as_interrupt():
    """SIGTERM (a CI abort) unwinds like Ctrl-C, so the running command's process group is killed on the way out."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGTERM, raise_terminated)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)


def render_commands(plan, selected, ctx):
    lines = []
    for rec in selected:
        device_numa = (ctx["env_file"].get("fingerprint") or {}).get("env.fp.device.numa_node", "")
        try:
            pinned = pinning_run.prepare(rec.get("pinning"), device_numa, ctx["sys_root"])
        except pinning_plan.PinningError as e:
            raise UsageError(f"case {rec['name']}: {e}") from None
        for repeat, cache in repeat_schedule(rec):
            vals = run_values(rec, ctx, repeat, cache, f"<out>/cases/{planfile.slug(rec['name'])}/raw", "<repeat_dir>")
            try:
                argv = (
                    planfile.render_tokens(rec["command"], vals, RUN_PHASE_TOKENS, rec["name"])
                    if rec.get("command")
                    else None
                )
            except ValueError as e:
                raise UsageError(str(e)) from None
            if argv and pinned.launches:
                argv = pinned.plan.prefix(pinned.tool) + argv
            lines.append({"case": rec["name"], "repeat": repeat, "cache": cache, "argv": argv})
    return lines


def execute(a, st=None, popen=subprocess.Popen):
    plan_path = cliutil.pick(a.plan, "BENCH_PLAN")
    if not plan_path:
        raise UsageError("--plan (or BENCH_PLAN) is required")
    plan = load_plan(plan_path)
    selected = select_cases(plan, cliutil.csv(a.case))
    # Absolute: commands run in --workdir, so a relative out would resolve against it and not against the caller.
    out = os.path.realpath(cliutil.pick(a.out, "RESULTS_DIR", cliutil.env("BENCH_OUT", "results/bench")))
    envf = load_env_file(cliutil.pick(a.env_file, "BENCH_ENV_FILE"))
    run_id = cliutil.pick(a.run_id, "BENCH_RUN_ID") or cliutil.env("RUN_ID")
    try:
        run_id = str(uuid.UUID(str(run_id))) if run_id else str(uuid.uuid4())
    except ValueError:
        if a.run_id or cliutil.env("BENCH_RUN_ID"):
            raise UsageError(f"run id {run_id!r} is not a UUID") from None
        run_id = str(uuid.uuid4())
    subject = ((plan.get("request") or {}).get("subject")) or {}
    want_ingest = cliutil.flag_on(a.ingest, "BENCH_INGEST") and not a.no_ingest
    run_id = run_id if a.dry_run else fresh_run_id(out, run_id, a.force_new_run, want_ingest)
    if want_ingest and not subject.get("artifact_id"):
        raise UsageError("--ingest needs a subject artifact: plan with --subject-id")
    timeout = cliutil.pick(a.timeout, "BENCH_TIMEOUT")
    timeout = number(timeout, "--timeout", minimum=1) if timeout is not None else None
    landed_wait = number(cliutil.pick(a.landed_wait, "BENCH_LANDED_WAIT", 30), "--landed-wait", minimum=0)
    ingest_timeout = number(cliutil.pick(a.ingest_timeout, "BENCH_INGEST_TIMEOUT", 900), "--ingest-timeout", minimum=1)
    max_log = int(number(cliutil.pick(a.max_log_bytes, "BENCH_MAX_LOG_BYTES", DEFAULT_LOG_BYTES), "--max-log-bytes", 1))
    max_metrics = int(
        number(
            cliutil.pick(a.max_metrics_bytes, "BENCH_MAX_METRICS_BYTES", DEFAULT_METRICS_BYTES),
            "--max-metrics-bytes",
            1,
        )
    )
    max_raw = int(
        number(cliutil.pick(a.max_raw_bytes, "BENCH_MAX_RAW_BYTES", DEFAULT_RAW_WARN_BYTES), "--max-raw-bytes", 1)
    )
    workdir = os.path.abspath(cliutil.pick(a.workdir, "BENCH_WORKDIR", os.getcwd()))
    ingest_python = cliutil.pick(a.ingest_python, "BENCH_INGEST_PYTHON")
    if ingest_python and not (os.path.isfile(ingest_python) and os.access(ingest_python, os.X_OK)):
        raise UsageError(f"--ingest-python {ingest_python!r}: not an executable file")
    ingest_pythonpath = ingest_pythonpath_dirs(
        cliutil.pick(a.ingest_pythonpath, "BENCH_INGEST_PYTHONPATH"), [workdir, out]
    )
    store_name = cliutil.pick(a.store, "BENCH_STORE", "none")
    if store_name != "none" and st is None and want_ingest and not a.dry_run:  # only a store that will be used
        try:
            stores.open_store(store_name).check()  # fail before running benchmarks, not after
        except stores.StoreError as e:
            raise UsageError(f"store: {e}") from None
    ctx = {
        "max_log": max_log,
        "max_metrics": max_metrics,
        "max_raw": max_raw,
        "pass_env": cliutil.csv(cliutil.pick(a.pass_env, "BENCH_PASS_ENV", "")),
        "pass_prefixes": tuple(cliutil.csv(cliutil.pick(a.pass_env_prefix, "BENCH_PASS_ENV_PREFIXES", ""))),
        "ingest_pass_env": cliutil.csv(cliutil.pick(a.ingest_pass_env, "BENCH_INGEST_PASS_ENV", "")),
        "ingest_python": ingest_python,
        "ingest_pythonpath": ingest_pythonpath,
        "run_key": f"manual:spyre-bench:{run_id}",
        "component": (plan.get("catalog") or {}).get("component", ""),
        "ingest": want_ingest,
        "out": out,
        "run_id": run_id,
        "plan": plan,
        "subject_id": subject.get("artifact_id"),
        "subject_attrs": subject.get("attrs"),
        "arch": (plan.get("leg") or {}).get("arch", ""),
        "workdir": workdir,
        "timeout": timeout,
        "env_file": envf,
        "sys_root": cliutil.pick(a.sys_root, "BENCH_SYS_ROOT", "/sys"),
        "quiet": cliutil.flag_on(a.quiet, "BENCH_QUIET"),
        "ingest_cmd": cliutil.pick(a.ingest_cmd, "BENCH_INGEST_CMD"),
        "ingest_timeout": ingest_timeout,
    }
    # A catalog is used only when one is named: ./benchmarks of whatever directory we happen to be in is not.
    cat = cliutil.open_catalog(a.catalog, a.central, a.pack, required=False, implicit=False)
    if cat is not None:
        check_catalog(cat, selected)
    if want_ingest and not a.dry_run:
        check_ingest_installed(selected, ctx)
    lines = render_commands(plan, selected, ctx)  # renders every argv first: a bad token fails before any case ran
    if want_ingest:
        ingest_commands(
            [{"name": r["name"], "status": "ok", "dir": ""} for r in selected], {r["name"]: r for r in selected}, ctx
        )
    if a.dry_run:
        if a.json:
            shown = [{**ln, "argv": planfile.redact_argv(ln["argv"]) if ln["argv"] else None} for ln in lines]
            print(json.dumps({"dry_run": True, "run_id": run_id, "commands": shown}, indent=1))
        else:
            for ln in lines:
                shown = shlex.join(planfile.redact_argv(ln["argv"])) if ln["argv"] else "NO COMMAND"
                print(f"{ln['case']}\trepeat {ln['repeat']} ({ln['cache']})\t{shown}")
        if not selected:
            return cliutil.EXIT_OK if all_reuse(plan) else cliutil.EXIT_NOTHING
        return cliutil.EXIT_FAILED if any(ln["argv"] is None for ln in lines) else cliutil.EXIT_OK
    started = now()
    Path(out).mkdir(parents=True, exist_ok=True)
    prepare_out(out, plan, plan_path, ctx, started)
    recs = {r["name"]: r for r in plan["cases"]}
    with terminate_as_interrupt():
        outcomes = [run_case(r, ctx, popen) for r in selected]
        checks, notes = build_results(cat, outcomes, recs, ctx)
        for o in outcomes:
            if o["_notes"]:
                notes.setdefault(o["name"], []).extend(o["_notes"])
        ingest, landed = hand_off(outcomes, recs, ctx, store_name, landed_wait, st)
    counts = {
        "planned": len(plan["cases"]),
        "run": len(outcomes),
        "ok": sum(o["status"] == "ok" for o in outcomes),
        "failed": sum(o["status"] != "ok" for o in outcomes),
        "reuse": sum(r["action"] == "reuse" for r in plan["cases"]),
        "skip": sum(r["action"] == "skip" for r in plan["cases"]),
        "not_selected": sum(r["action"] == "run" for r in plan["cases"]) - len(outcomes),
    }
    manifest = {
        **base_manifest(plan, plan_path, ctx, started),
        "counts": counts,
        "cases": [{k: v for k, v in o.items() if not k.startswith("_")} for o in outcomes]
        + [
            {
                "name": r["name"],
                "benchmark_id": r["benchmark_id"],
                "status": {"reuse": "reused", "skip": "skipped"}.get(r["action"], "not_selected"),
                "reason": r.get("reason", ""),
            }
            for r in plan["cases"]
            if r["name"] not in {o["name"] for o in outcomes}
        ],
        "ingest": ingest,
        "stored_run_ids": ingest.get("stored_run_ids", []),
        "rows_landed": landed,
        "result_checks": checks,
        "result_notes": notes,
    }
    code, reason = exit_of(manifest)
    planfile.write_file(out, Path(out) / "plan.redacted.json", planfile.dump(planfile.redact_plan(plan)))
    manifest.update(exit_code=code, exit_reason=reason, ended_at=cliutil.utc_stamp(now()))
    planfile.write_file(out, Path(out) / "run.json", planfile.dump(manifest))
    if a.json:
        print(planfile.dump(manifest))
    else:
        for c in manifest["cases"]:
            print(f"{c['status']:<12} {c['name']}  {c.get('reason', '')}".rstrip())
        print(
            f"run {run_id}: {counts['ok']} ok, {counts['failed']} failed, {counts['reuse']} reused, {counts['skip']} skipped, "
            f"{counts['not_selected']} not selected; ingest {ingest['status']}, rows {landed['status']}; "
            f"exit {code} ({reason})"
        )
    if code == cliutil.EXIT_OK and want_ingest and landed["status"] == "unchecked":
        print(f"warning: rows were not verified ({landed['reason']})", file=sys.stderr)
    return code


def fresh_run_id(out, run_id, force_new, want_ingest=True):
    """The run id, refused (or replaced with --force-new-run) when an earlier run in `out` may have ingested under it.

    A run that does not ingest stores nothing, so it can repeat an id freely.
    """
    try:
        previous = json.loads((Path(out) / "run.json").read_text())
    except (OSError, ValueError):
        return run_id
    if not isinstance(previous, dict) or previous.get("engine_run_id") != run_id:
        return run_id
    ingested = (previous.get("ingest") or {}).get("requested", True)  # a provisional manifest may have ingested
    if not (want_ingest and ingested):
        return run_id
    if force_new:
        return str(uuid.uuid4())
    raise UsageError(
        f"run id {run_id} was already used in {out} with an ingest: a second bundle under it would be rejected. "
        "Pass a new --run-id, or --force-new-run (unset RUN_ID/BENCH_RUN_ID if two runs share them)"
    )


def ingest_pythonpath_dirs(value, forbidden):
    """Directories for the ingest's PYTHONPATH: real, existing, and not under the workdir or results directory."""
    items = value if isinstance(value, list) else [x for x in str(value or "").split(os.pathsep) if x]
    dirs = []
    for item in items:
        real = os.path.realpath(item)
        if not os.path.isdir(real):
            raise UsageError(f"--ingest-pythonpath {item!r}: not a directory")
        for bad in (os.path.realpath(f) for f in forbidden):
            if real == bad or real.startswith(bad + os.sep):
                raise UsageError(
                    f"--ingest-pythonpath {item!r} is under {bad}: a benchmark can write there, and the ingest would import it"
                )
        dirs.append(real)
    return dirs


def check_ingest_installed(selected, ctx):
    """A default ingest needs its package in its interpreter: probe it the way the real ingest will start (same
    interpreter flags, environment and an empty directory), and say so now, not after the benchmarks have run."""
    if ctx["ingest_cmd"]:
        return
    probe = [r for r in selected if not r.get("ingest") and "spyre_clickhouse_ingest" in resolve_ingest(r, None, ctx)]
    if not probe:
        return
    python = ctx.get("ingest_python") or sys.executable
    flags = ["-s", "-P"] if ctx["ingest_pythonpath"] else ["-I"]
    with tempfile.TemporaryDirectory(prefix="ingest-probe-") as scratch:
        env = planfile.ingest_env(os.environ, ctx.get("ingest_pass_env", ()), scratch)
        if ctx["ingest_pythonpath"]:
            env["PYTHONPATH"] = os.pathsep.join(ctx["ingest_pythonpath"])
        try:
            found = (
                subprocess.run(
                    [python, *flags, "-c", "import spyre_clickhouse_ingest"],
                    cwd=scratch,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=120,
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            found = False
    if not found:
        raise UsageError(
            f"the ingest package cannot be imported by `{python} {' '.join(flags)}`: install it there, or give "
            "--ingest-python PATH (an interpreter that has it) or --ingest-pythonpath DIR (outside --workdir and --out)"
        )


def load_env_file(path):
    """The probe's env.json, validated (types matter: a string "false" must not read as true); {} without a path."""
    if not path:
        return {}
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise UsageError(f"--env-file {path}: {e}") from None
    shapes = {"fingerprint": dict, "observations": dict, "clean": dict, "node": str, "qualified": bool}
    if not isinstance(doc, dict):
        raise UsageError(f"--env-file {path}: expected a JSON object")
    for key, kind in shapes.items():
        if key in doc and not isinstance(doc[key], kind):
            raise UsageError(f"--env-file {path}: {key} must be {kind.__name__}")
    return doc


def number(value, flag, minimum):
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise UsageError(f"{flag} {value!r}: a number") from None
    if not math.isfinite(x) or x < minimum:
        raise UsageError(f"{flag} {value!r}: a finite number >= {minimum}")
    return x


def all_reuse(plan):
    kinds = {r["action"] for r in plan["cases"]}
    return kinds == {"reuse"}


def base_manifest(plan, plan_path, ctx, started):
    return {
        "run_schema": planfile.RUN_SCHEMA,
        "run_id": ctx["run_id"],
        "engine_run_id": ctx["run_id"],
        "started_at": cliutil.utc_stamp(started),
        "ended_at": None,
        "plan": {
            "path": "plan.redacted.json",
            "sha256": planfile.sha256_file(plan_path),
            "redacted_sha256": hashlib.sha256(planfile.dump(planfile.redact_plan(plan)).encode()).hexdigest(),
            "plan_schema": plan["plan_schema"],
            "identity_version": plan["producer"]["engine"]["identity_version"],
        },
        "purpose": plan.get("purpose"),
        "subject_id": ctx["subject_id"],
        "arch": ctx["arch"],
    }


def prepare_out(out, plan, plan_path, ctx, started):
    """Copy the plan, drop the previous run's manifest and ingest logs, and leave a manifest saying "incomplete" that
    only a finished run replaces: an aborted run must not look like the previous one's success."""
    planfile.write_file(out, Path(out) / "plan.redacted.json", planfile.dump(planfile.redact_plan(plan)))
    shutil.rmtree(Path(out) / "ingest", ignore_errors=True)
    provisional = {
        **base_manifest(plan, plan_path, ctx, started),
        "exit_code": cliutil.EXIT_FAILED,
        "exit_reason": "incomplete",
    }
    planfile.write_file(out, Path(out) / "run.json", planfile.dump(provisional))


def build_results(cat, outcomes, recs, ctx):
    """(checks, notes): per case, the problems of the Result built for it and the things worth knowing about it."""
    checks, notes = {}, {}
    if cat is None:
        return checks, notes
    for o in outcomes:
        if not o["_repeats"]:
            continue
        try:
            result, note = build_case_result(cat, recs[o["name"]], o, ctx)
        except ingest_adapter.IngestUnavailable as e:
            result, note = None, f"result not built: {e}"
        except (ValueError, KeyError, TypeError) as e:
            result, note = None, None
            checks[o["name"]] = [f"result not built: {type(e).__name__}: {e}"]
        if result is not None:
            path = Path(ctx["out"]) / o["dir"] / "result.json"
            planfile.write_file(ctx["out"], path, planfile.dump(result))
            o["result"] = str(path.relative_to(ctx["out"]))
            checks[o["name"]] = check_result(result)
        if note:
            notes[o["name"]] = [note]
    return checks, notes


def hand_off(outcomes, recs, ctx, store_name, landed_wait, st):
    ingest = {"requested": ctx["ingest"], "status": "skipped", "invocations": []}
    landed = {"status": "unchecked", "reason": "ingest not requested", "expected": [], "missing": []}
    if not ctx["ingest"]:
        return ingest, landed
    if any(o["status"] == "ok" for o in outcomes):
        ingest["invocations"] = run_ingest(ingest_commands(outcomes, recs, ctx), ctx)
        bad = [i for i in ingest["invocations"] if i["returncode"] != 0 or i.get("error")]
        ingest["status"] = "failed" if bad else "ok"
        ingest["cases_ingested"] = sorted({c for i in ingest["invocations"] if i not in bad for c in i["cases"]})
        ingest["cases_failed"] = sorted({c for i in bad for c in i["cases"]})
    ingest["stored_run_ids"] = sorted({i["stored_run_id"] for i in ingest["invocations"] if i.get("stored_run_id")})
    if ingest["status"] != "ok":
        return ingest, {**landed, "reason": f"ingest {ingest['status']}"}
    if store_name == "none":
        return ingest, {**landed, "reason": "no store configured"}
    by_name = {o["name"]: o for o in outcomes}
    groups = {}
    for inv in ingest["invocations"]:
        groups.setdefault(inv["stored_run_id"], []).extend(by_name[c] for c in inv["cases"])
    try:
        return ingest, check_landed(st or stores.open_store(store_name), groups, landed_wait)
    except stores.StoreError as e:
        return ingest, {"status": "store_error", "reason": str(e), "expected": [], "missing": []}


def check_catalog(cat, selected):
    """Every selected case must exist in the catalog with the planned benchmark_id (skipped when the ingest is absent)."""
    by = {c["name"]: c for c in cat.cases}
    for rec in selected:
        c = by.get(rec["name"])
        if c is None:
            raise UsageError(f"case {rec['name']!r} is in the plan but not in the catalog: re-plan")
        try:
            again = cat.ids(c)["benchmark_id"]
        except ingest_adapter.IngestUnavailable:
            continue
        if again != rec["benchmark_id"]:
            raise UsageError(
                f"case {rec['name']!r}: benchmark_id differs from the plan ({again} != {rec['benchmark_id']}): re-plan"
            )


def parser():
    ap = argparse.ArgumentParser(prog="spyre-bench run", description=__doc__.split("\n")[0])
    cliutil.add_catalog_flags(ap)
    ap.add_argument("--plan", help="plan.json from `spyre-bench plan` (BENCH_PLAN)")
    ap.add_argument("--case", action="append", default=[], help="run only this planned case (repeatable)")
    ap.add_argument("--out", help="results directory (RESULTS_DIR, BENCH_OUT; default results/bench)")
    ap.add_argument("--workdir", help="cwd of every command (BENCH_WORKDIR; default the current directory)")
    ap.add_argument("--env-file", help="the probe's env.json (BENCH_ENV_FILE)")
    ap.add_argument("--run-id", help="a UUID (BENCH_RUN_ID, then RUN_ID when a UUID); default a new one")
    ap.add_argument("--ingest", action="store_true", help="hand the raw outputs to the ingest (BENCH_INGEST=1)")
    ap.add_argument("--no-ingest", action="store_true", help="do not ingest even if BENCH_INGEST=1")
    ap.add_argument("--ingest-cmd", help="ingest argv as one shell-split string; replaces the profile default")
    ap.add_argument("--ingest-timeout", help="seconds per ingest invocation (default 900)")
    ap.add_argument("--store", choices=("none", "clickhouse"), help="source of the rows-landed check (BENCH_STORE)")
    ap.add_argument("--landed-wait", help="seconds to wait for rows to land (default 30)")
    ap.add_argument("--timeout", help="cap, in seconds, on every command (BENCH_TIMEOUT)")
    ap.add_argument("--quiet", action="store_true", help="set BENCH_QUIET=1 for the commands")
    ap.add_argument(
        "--pass-env", action="append", help="environment variable NAME to hand to commands (BENCH_PASS_ENV)"
    )
    ap.add_argument(
        "--pass-env-prefix",
        action="append",
        help="hand every variable starting with PREFIX to commands (BENCH_PASS_ENV_PREFIXES)",
    )
    ap.add_argument(
        "--ingest-pass-env", action="append", help="extra variable NAME for the ingest (BENCH_INGEST_PASS_ENV)"
    )
    ap.add_argument("--ingest-python", help="interpreter that has the ingest installed (BENCH_INGEST_PYTHON)")
    ap.add_argument(
        "--ingest-pythonpath",
        action="append",
        help="directory holding the ingest package, outside --workdir and --out (BENCH_INGEST_PYTHONPATH)",
    )
    ap.add_argument(
        "--force-new-run", action="store_true", help="mint a new run id if --run-id was already used in --out"
    )
    ap.add_argument("--max-log-bytes", help="cap per stdout/stderr log (BENCH_MAX_LOG_BYTES, default 16 MiB)")
    ap.add_argument("--max-metrics-bytes", help="cap on a metrics.json (BENCH_MAX_METRICS_BYTES, default 8 MiB)")
    ap.add_argument(
        "--max-raw-bytes", help="raw output size that triggers a warning (BENCH_MAX_RAW_BYTES, default 1 GiB)"
    )
    ap.add_argument("--sys-root", help=argparse.SUPPRESS)  # test hook: a described host topology instead of /sys
    ap.add_argument("--dry-run", action="store_true", help="print the commands; create and run nothing")
    ap.add_argument("--json", action="store_true", help="print run.json (or the dry-run commands)")
    return ap


def main(argv=None, st=None, popen=subprocess.Popen):
    return cliutil.run_main(lambda a: execute(a, st, popen), parser(), argv)


if __name__ == "__main__":
    sys.exit(main())
