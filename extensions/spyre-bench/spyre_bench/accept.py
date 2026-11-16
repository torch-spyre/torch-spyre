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

"""spyre-bench accept: propose a stored run as the new accepted baseline (anchor) of its series.

  spyre-bench accept (--run-id UUID | --run-dir DIR) --reason TEXT [--benchmark UUID] [--metric M]
                     [--window N] [--out DIR] [--dry-run] [--json] [--store clickhouse]

Reads the run's samples from the store, refuses anything that is not baseline-eligible (suspect, invalidated,
off-purpose), and hands one reference-from-run request per series scope to a ReferenceSink. The request is the input
of the reference machinery (`reference create --from-request`, approval, tag move); each value carries the `case` block
(component, name, tags, disc, disc_keys[, id_profile]) that rebuilds its benchmark_id, and `anchor` is
{role, run_id, as_of} (`from_run_id` repeats run_id for older readers); the engine writes no rows and moves no
tag itself, so the anchor changes only when the reference is approved.
"""

import argparse
import getpass
import hashlib
import json
import math
import regex as re
import sys
from datetime import timezone
from pathlib import Path

from spyre_bench import cliutil, engine, store as stores, verdict
from spyre_bench.cliutil import UsageError

REQUEST_SCHEMA = "bench.reference-request/1"
MIN_SAMPLES = 3
MAX_WINDOW = 50


class CaseDataError(Exception):
    """A series whose case block cannot be rebuilt into its benchmark_id."""

    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = problems


class ReferenceSink:
    """Where a reference-from-run request goes. The reference machinery implements `propose`."""

    def preflight(self, requests):
        """Raise UsageError before anything is written if the sink would refuse one of `requests`."""

    def propose(self, request):
        raise NotImplementedError


class FileSink(ReferenceSink):
    """Writes each request as JSON under a directory, for the reference ingest to pick up."""

    def __init__(self, out):
        self.out = Path(out)

    def path_of(self, request):
        return self.out / f"reference-request-{request['key']}-{request['anchor']['from_run_id'][:8]}.json"

    def preflight(self, requests):
        taken = [str(self.path_of(r)) for r in requests if self.path_of(r).exists()]
        if taken:
            raise UsageError(f"already proposed, nothing written: {', '.join(taken)} (remove them or use --out)")

    def propose(self, request):
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.path_of(request)
        try:
            with open(path, "x") as fh:  # never replace a pending request
                fh.write(json.dumps(request, indent=1, sort_keys=True, allow_nan=False) + "\n")
        except FileExistsError:
            raise UsageError(f"{path} exists: already proposed (remove it or use --out)") from None
        return {"state": "proposed", "path": str(path)}


class MemorySink(ReferenceSink):
    """Records requests; for tests and --dry-run."""

    def __init__(self):
        self.requests = []

    def propose(self, request):
        self.requests.append(request)
        return {"state": "recorded"}


def slug(*parts):
    """A readable key plus a hash of the exact parts, so two scopes never share a key."""
    readable = re.sub(r"[^a-z0-9]+", "-", ".".join(p or "any" for p in parts).lower()).strip("-")[:80]
    return f"{readable}-{hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8]}"


def iso(value):
    if not hasattr(value, "strftime"):
        return str(value)
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")  # a naive time is UTC, as the store reads it


def select(rows, benchmarks, metrics):
    keep = [
        r
        for r in rows
        if (not benchmarks or r["benchmark_key"] in benchmarks) and (not metrics or r["metric"] in metrics)
    ]
    if not keep:
        raise UsageError("no series of the run matches --benchmark / --metric")
    return keep


def refusals(rows):
    """Why a row cannot become an anchor: a suspect or off-purpose run is not baseline-eligible."""
    out = []
    for r in rows:
        if not r["eligible"]:
            why = " and ".join(
                w for w, on in (("off-purpose run", r.get("off_purpose")), ("suspect run", r.get("suspect"))) if on
            )
            out.append(f"{r['name']} {r['metric']}: {why or 'not eligible'}")
        elif not r["samples"] or not all(math.isfinite(float(x)) for x in r["samples"]):
            out.append(f"{r['name']} {r['metric']}: no samples or a non-finite sample")
    return out


CASE_KEYS = ("component", "name", "tags", "disc", "disc_keys", "id_profile")


def rebuilds(block):
    """The benchmark_id a case block derives, by the rule `reference create --from-request` applies; '' if it cannot."""
    try:
        disc = {k: str(v) for k, v in (block.get("disc") or {}).items()}
        keys = block.get("disc_keys")
        return engine.benchmark_id(
            block["component"], block["name"].strip(), list(block.get("tags") or []), disc,
            sorted(disc) if keys is None else list(keys),
        )  # fmt: skip
    except (KeyError, AttributeError, TypeError):
        return ""


def result_cases(run_dir):
    """benchmark_id -> case block from the Result files of a results directory (exact disc_keys and id profile)."""
    manifest = verdict.manifest_of(run_dir) or {}
    base = Path(run_dir).resolve()
    out = {}
    for c in manifest.get("cases", []):
        name = c.get("result") if isinstance(c, dict) else None
        path = (base / name) if name else None
        try:
            case = json.loads(path.read_text())["status"]["case"]
        except (AttributeError, OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(case, dict) and case.get("benchmark_id") and path.resolve().is_relative_to(base):
            out[case["benchmark_id"]] = {k: case[k] for k in CASE_KEYS if k in case}
    return out


def case_blocks(st, ids, run_dir=None):
    """(benchmark_id -> case block, problems): from the run's Results first, else the registered case; every block must
    rebuild its id, because the reference refuses one that does not."""
    found = result_cases(run_dir) if run_dir else {}
    missing = [i for i in ids if i not in found]
    for r in st.case_rows(missing) if missing else []:
        props = {k: str(v) for k, v in (r.get("props") or {}).items()}
        found.setdefault(
            r["benchmark_key"],
            {"component": r["component"], "name": r["name"], "tags": list(r.get("tags") or []), "disc": props,
             "disc_keys": sorted(props)},
        )  # fmt: skip
    blocks, problems = {}, []
    for i in sorted(ids):
        block = found.get(i)
        if block is None:
            problems.append(f"{i}: no case data (not registered, and not in the --run-dir results)")
        elif rebuilds(block) != i:
            problems.append(f"{i}: its case data derives {rebuilds(block) or 'no id'}; pass the run's --run-dir")
        else:
            blocks[i] = block
    return blocks, problems


def build_request(scope, rows, carried, run_id, reason, actor, anchored, cases):
    """One reference-from-run request for a scope; `carried` are the scope's anchored series this run does not replace."""
    component, arch, backend, source_job = scope
    as_of = max(iso(r["run_ts"]) for r in rows)
    values = [
        {
            "benchmark_id": r["benchmark_key"], "name": r["name"], "case": cases[r["benchmark_key"]], "metric": r["metric"],
            "samples": [float(x) for x in r["samples"]], "as_of": iso(r["run_ts"]), "carried": False,
            "window_run_ids": r.get("window_run_ids") or [run_id],
            "replaces": anchored.get((r["benchmark_key"], r["metric"], backend, arch, source_job)),
        }
        for r in rows
    ]  # fmt: skip
    values += [
        {
            "benchmark_id": c["benchmark_key"], "name": None, "case": cases[c["benchmark_key"]], "metric": c["metric"],
            "samples": [float(x) for x in c["samples"]], "as_of": iso(c["anchor_from"]), "carried": True,
            "replaces": None,
        }
        for c in carried
    ]  # fmt: skip
    return {
        "request_schema": REQUEST_SCHEMA,
        "kind": "bench-reference",
        "origin": "generated",
        "class": "perf",
        "role": "anchor",
        "key": "anchor-" + slug(component, arch, backend, source_job),
        "label": f"accepted baseline {component} {backend}",
        "reason": reason,
        "accepted_by": actor,
        "scope": {"component": component, "arch": arch, "backend": backend, "source_job": source_job},
        "anchor": {"role": "anchor", "run_id": run_id, "from_run_id": run_id, "as_of": as_of},
        "comparability": {
            "harness": "spyre-bench",
            "harness_match": "yes",
            "dataset_subset": "full",
            "dataset_match": "same",
            "precision": "unknown",
            "caveats": "an accepted run of the same benchmark; see anchor.from_run_id",
        },  # fmt: skip
        "values": values,
        "warnings": [] if rows else ["no series"],
    }


def widen(st, rows, window):
    """With --window N, a row's samples are those of the series' newest N eligible runs up to the accepted run."""
    if window <= 1:
        return rows
    wide = []
    for r in rows:
        runs = st.series_window(r, window)
        keys = [w["run_key"] for w in runs]
        if r.get("run_key") and r["run_key"] not in keys:
            runs = [*runs, {"run_key": r["run_key"], "run_ts": r["run_ts"], "samples": r["samples"]}][-window:]
        wide.append(
            {**r, "samples": [x for w in runs for x in w["samples"]], "window_run_ids": [w["run_key"] for w in runs]}
        )
    return wide


def too_thin(rows):
    """Anchors need MIN_SAMPLES samples (what the SQL view requires); a single run often has fewer."""
    return [
        f"{r['name']} {r['metric']}: {len(r['samples'])} sample(s), the anchor needs {MIN_SAMPLES} (use --window N)"
        for r in rows
        if len(r["samples"]) < MIN_SAMPLES
    ]


def plan_requests(st, run_id, rows, reason, actor, run_dir=None):
    scopes = {}
    for r in rows:
        scopes.setdefault((r["component"], r["arch"], r["backend"], r["source_job"]), []).append(r)
    anchored = {
        (a.get("benchmark_key"), a["metric"], a.get("backend"), a.get("arch"), a.get("source_job")): a.get("version")
        for a in (st.anchored_rows(run_id) or [])
        if a.get("status") != "no_anchor" and a.get("version")
    }
    out = []
    for scope, srows in sorted(scopes.items()):
        taken = {(r["benchmark_key"], r["metric"]) for r in srows}
        carried = [c for c in st.anchor_rows(*scope) if (c["benchmark_key"], c["metric"]) not in taken]
        sound = [c for c in carried if c["samples"] and all(math.isfinite(float(x)) for x in c["samples"])]
        ids = {r["benchmark_key"] for r in [*srows, *sound]}
        cases, problems = case_blocks(st, ids, run_dir)
        if problems:
            raise CaseDataError(problems)
        request = build_request(scope, srows, sound, run_id, reason, actor, anchored, cases)
        if len(sound) < len(carried):
            request["warnings"].append(
                f"{len(carried) - len(sound)} carried series dropped: empty or non-finite samples"
            )
        out.append(request)
    return out


def text(doc):
    lines = [
        ("would propose" if doc["dry_run"] else "proposed")
        + f" {len(doc['requests'])} reference(s) from run {doc['run_id']}"
    ]
    for r in doc["requests"]:
        n = sum(not v["carried"] for v in r["values"])
        c = len(r["values"]) - n
        lines.append(
            f"  {r['key']}: {n} series accepted, {c} carried from the current anchor  [{r['result']['state']}]"
        )
    return "\n".join(lines)


def os_user():
    try:
        return getpass.getuser()
    except (KeyError, OSError):  # a container user with no passwd entry
        return "unknown"


def execute(a, st=None, sink=None):
    reason = (a.reason or "").strip()
    if not reason:
        raise UsageError("--reason is required: it is recorded with the new anchor")
    run_ids, note = verdict.resolve_ids(a)
    if len(run_ids) != 1:
        raise UsageError("give exactly one run: --run-id, or a --run-dir whose run.json lists one stored run id")
    store_name = cliutil.pick(a.store, "BENCH_STORE", "clickhouse")
    if store_name == "none":
        raise UsageError("accept needs a store (--store clickhouse)")
    actor = a.accepted_by or cliutil.env("BENCH_ACTOR") or os_user()
    try:
        st = st or stores.open_store(store_name)
        rows = select(st.series_runs(run_ids[0]), set(cliutil.csv(a.benchmark)), set(cliutil.csv(a.metric)))
        bad = refusals(rows)
        if not bad:
            rows = widen(st, rows, max(1, min(int(a.window or 1), MAX_WINDOW)))
            bad = refusals(rows) + too_thin(rows)
        if bad:
            print("error: not baseline-eligible, nothing proposed:\n  " + "\n  ".join(bad), file=sys.stderr)
            return cliutil.EXIT_FAILED
        requests = plan_requests(st, run_ids[0], rows, reason, actor, a.run_dir)
    except CaseDataError as e:
        print(
            "error: cannot rebuild the benchmark id, nothing proposed:\n  " + "\n  ".join(e.problems), file=sys.stderr
        )
        return cliutil.EXIT_FAILED
    except stores.StoreError as e:
        print(f"error: store: {e}", file=sys.stderr)
        return cliutil.EXIT_STORE
    if note:
        print(f"note: {note}", file=sys.stderr)
    if sink is None:
        out = a.out or Path(cliutil.env("BENCH_OUT", "results/bench")) / "accept"
        sink = MemorySink() if a.dry_run else FileSink(out)
    sink.preflight(requests)
    for r in requests:
        r["result"] = sink.propose(r)
    doc = {"accept_schema": "bench.accept/1", "run_id": run_ids[0], "dry_run": bool(a.dry_run), "requests": requests}
    print(json.dumps(doc, indent=1, allow_nan=False) if a.json else text(doc))
    return cliutil.EXIT_OK


def parser():
    ap = argparse.ArgumentParser(prog="spyre-bench accept", description=__doc__.split("\n")[0])
    ap.add_argument(
        "--run-dir",
        help="a results directory: its run.json names the stored run id and its Results give each case (BENCH_RUN_DIR)",
    )
    ap.add_argument("--run-id", help="the stored run UUID to accept (BENCH_RUN_ID, RUN_ID)")
    ap.add_argument("--reason", help="why this run is the new normal (required; recorded with the anchor)")
    ap.add_argument("--benchmark", action="append", default=[], help="accept only these benchmark ids")
    ap.add_argument("--metric", action="append", default=[], help="accept only these metrics")
    ap.add_argument(
        "--window", type=int, default=1, help="pool the series' newest N eligible runs up to the accepted run (max 50)"
    )
    ap.add_argument("--accepted-by", help="who accepts (BENCH_ACTOR, default the OS user)")
    ap.add_argument("--out", help="directory for the request files (default BENCH_OUT/accept)")
    ap.add_argument("--dry-run", action="store_true", help="print the requests, write nothing")
    ap.add_argument("--store", choices=("none", "clickhouse"), help="BENCH_STORE, default clickhouse")
    ap.add_argument("--json", action="store_true")
    ap.set_defaults(artifact=None)
    return ap


def main(argv=None, st=None, sink=None):
    return cliutil.run_main(lambda a: execute(a, st, sink), parser(), argv)


if __name__ == "__main__":
    sys.exit(main())
