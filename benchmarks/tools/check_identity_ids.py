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

"""Conformance check: the `identity` blocks of benchmarks/ derive the ids the ingest writes.

For every case the stored id from `spyre-bench ids` must equal the id `spyre_clickhouse_ingest` writes for
the matching row of a synthetic perf-suite report.xml (parsed by the ingest's own parser). Kernel and total
rows stay with the ingest parser; their ids are derived from each case's discriminators and pinned in the
golden table, so an edit that would start a new series fails here.

    python benchmarks/tools/check_identity_ids.py [--update]

Needs pyyaml, regex, lxml, clickhouse-connect and the spyre-bench package; exits 0 with a notice when a module
is missing unless BENCH_REQUIRED is set. Shapes and kernel names are synthetic fixtures.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "benchmarks"
GOLDEN = CATALOG / "ingest-ids.golden.tsv"
HEADER = "component\tname\trow\tinput_shapes\tkernel_name\tbenchmark_id"
KERNELS = (
    ("spyre_kernel_v1_fused_add_aaaaaaaaaaaaaaaa#2", 0.003),
    ("spyre_kernel_v1_fused_add_bbbbbbbbbbbbbbbb#2", 0.27),
)
METRIC = "wall_clock"


def engine(*args, catalog=CATALOG):
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "spyre_bench",
            *args,
            "--catalog",
            str(catalog),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode:
        raise RuntimeError(f"spyre-bench {args[0]} failed:\n{out.stdout}{out.stderr}")
    return json.loads(out.stdout)


def load_ingest():
    sys.path.insert(0, str(ROOT / "extensions" / "clickhouse-ingest"))
    from spyre_clickhouse_ingest import results
    from spyre_clickhouse_ingest.identity import BenchmarkId

    return results, BenchmarkId


def declared(catalog=CATALOG):
    """Per case: the identity inputs and the engine's stored id."""
    stored = {r["case"]: r["benchmark_id"] for r in engine("ids", catalog=catalog)}
    inputs = {
        r["case"]: r["inputs"] for r in engine("ids", "--inputs", catalog=catalog)
    }
    return [(case, inputs[case], stored[case]) for case in stored]


def report_xml(cases):
    """The report.xml the perf suite writes for these cases: one wall-clock testcase per op and shape."""
    seen, tcs = set(), []
    for _, inp, _ in cases:
        key = (inp["name"], inp["disc"]["input_shapes"])
        if key not in seen:
            seen.add(key)
            tcs.append(
                f'<testcase name="perf_{key[0]}_{METRIC}_ms_{key[1]}" time="1.0"/>'
            )
    return (
        '<?xml version="1.0"?><testsuites><testsuite name="spyre-perf-suite" '
        f'timestamp="2026-01-01T00:00:00Z">{"".join(tcs)}</testsuite></testsuites>'
    )


def ingest_report_ids(results, BenchmarkId, cases):
    """{(name, input_shapes): id} the ingest writes for the synthetic report."""
    with tempfile.TemporaryDirectory() as d:
        xml = Path(d) / "report.xml"
        xml.write_text(report_xml(cases))
        _, records = results.parse_benchmark_xml(xml, "bench-conformance", "", "x86_64")
    return {
        (e["name"], e["disc"]["input_shapes"]): BenchmarkId.derive(
            results.COMPONENT_DEFAULT, e["name"], e["tags"], e["disc"], e["disc_keys"]
        )
        for e in results._bench_entries(records)
    }


def lint(catalog=CATALOG):
    """Problems with the declared cost and identity shape: est_hours is the only cost hint."""
    import yaml

    problems = []
    for path in sorted(Path(catalog).rglob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if not doc:
                continue
            spec = doc.get("spec") or {}
            ann = (doc.get("metadata") or {}).get("annotations") or {}
            if "schedule" in spec or any(k.startswith("schedule.") for k in ann):
                problems.append(
                    f"{path.name}: cadence belongs to the invoker, not the definitions"
                )
            if doc.get("kind") == "Benchmark":
                ident = spec.get("identity") or {}
                name = doc["metadata"]["name"]
                if (
                    ident.get("mode") != "ingest"
                    or ident.get("profile") != "perf-kernel"
                ):
                    problems.append(
                        f"{path.name}: {name} needs identity mode ingest, profile perf-kernel"
                    )
                if "est_hours" not in spec:
                    problems.append(f"{path.name}: {name} has no est_hours")
    return problems


def table(results, BenchmarkId, cases, report_ids):
    """(rows, mismatches): golden rows and every case whose definition disagrees with the ingest parser."""
    keys = results._V2_BENCH_ID_KEYS
    comp = results.COMPONENT_DEFAULT
    rows, bad = [], []
    for case, inp, stored in cases:
        name, disc = inp["name"], inp["disc"]
        shapes = disc["input_shapes"]
        if inp["component"] != comp or list(inp["disc_keys"]) != list(keys):
            bad.append(
                f"{case}: component/keys {inp['component']}/{inp['disc_keys']} differ from the ingest's"
            )
        written = report_ids.get((name, shapes))
        if written != stored:
            bad.append(
                f"{case}: definition derives {stored}, the ingest writes {written}"
            )
        rows.append((comp, name, "report", shapes, "", written or ""))
        entries = [
            {
                "name": name,
                "tags": [],
                "backend": "spyre",
                "measurements": {"duration_ms": [ms]},
                "disc": {**disc, "kernel_name": raw, "is_total": 0},
                "disc_keys": keys,
            }
            for raw, ms in KERNELS
        ]
        for e in BenchmarkId.rank_kernels(comp, entries):
            kid = BenchmarkId.derive(comp, name, e["tags"], e["disc"], keys)
            rows.append((comp, name, "kernel", shapes, e["disc"]["kernel_name"], kid))
        total = {**disc, "kernel_name": "Total", "is_total": 1}
        rows.append(
            (
                comp,
                name,
                "total",
                shapes,
                "Total",
                BenchmarkId.derive(comp, name, [], total, keys),
            )
        )
    return rows, bad


def render(rows):
    return "\n".join([HEADER, *("\t".join(r) for r in rows)]) + "\n"


def check(catalog=CATALOG, golden=GOLDEN, update=False):
    """Return (exit_code, messages)."""
    results, BenchmarkId = load_ingest()
    try:
        cases = declared(catalog)
    except RuntimeError as e:
        return 1, [str(e)]
    msgs = lint(catalog)
    rows, bad = table(
        results, BenchmarkId, cases, ingest_report_ids(results, BenchmarkId, cases)
    )
    msgs += bad
    text = render(rows)
    if update and not msgs:
        Path(golden).write_text(text)
        return 0, [f"wrote {golden}: {len(rows)} ids"]
    old = {}
    if Path(golden).exists():
        old = {
            tuple(ln.split("\t")[:5]): ln.split("\t")[5]
            for ln in Path(golden).read_text().splitlines()[1:]
        }
    for r in rows:
        if old.get(r[:5]) != r[5]:
            msgs.append(
                f"{r[1]} {r[2]} {r[4]}: derived {r[5]}, golden {old.get(r[:5])}"
            )
    if len(old) != len(rows):
        msgs.append(f"golden has {len(old)} rows, the catalog derives {len(rows)}")
    if msgs:
        return 1, msgs
    return 0, [
        f"PASS identity conformance: {len(cases)} cases, {len(rows)} ids match {Path(golden).name}"
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--update",
        action="store_true",
        help="rewrite the golden table (an identity change)",
    )
    args = ap.parse_args(argv)
    try:
        code, msgs = check(update=args.update)
    except ImportError as e:
        print(
            f"notice: identity conformance skipped, missing module: {e.name}",
            file=sys.stderr,
        )
        return 2 if os.environ.get("BENCH_REQUIRED") else 0
    for m in msgs:
        print(m, file=sys.stderr if code else sys.stdout)
    if code:
        print(
            "identity or definition problems: review; an intended id change reruns with --update",
            file=sys.stderr,
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
