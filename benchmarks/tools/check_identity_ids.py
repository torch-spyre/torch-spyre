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

Needs pyyaml, regex, lxml, clickhouse-connect and the spyre-bench package; exits 2 when a module is
missing unless BENCH_OPTIONAL is set (and BENCH_REQUIRED is not). Kernel names are synthetic fixtures; shapes, models and configs are the perf suite's own.
"""

import argparse
import regex as re
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "benchmarks"
GOLDEN = CATALOG / "ingest-ids.golden.tsv"
HEADER = "component\tname\trow\tconfig_name\tinput_shapes\trun_mode\tkernel_name\tbenchmark_id"
KERNELS = (
    ("spyre_kernel_v1_fused_add_aaaaaaaaaaaaaaaa#2", 0.003),
    ("spyre_kernel_v1_fused_add_bbbbbbbbbbbbbbbb#2", 0.27),
)
METRIC = "wall_clock"
CLASSNAME = (
    "perf.benchmark"  # the ingest detects a perf report by "benchmark" in the classname
)


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


def row_key(name, disc):
    """What tells two rows of one name apart: the discriminators a case varies."""
    return (
        name,
        disc.get("record_type") or "",
        disc.get("config_name") or "",
        disc.get("input_shapes") or "",
        disc.get("run_mode") or "",
    )


def model_testcases(inp):
    """The testcases of a model record: no shape in the name, the model identity in the tags."""
    disc, name = inp["disc"], inp["name"]
    mode = disc["run_mode"]
    batch, prompt = (1, 512) if mode == "prefill_only" else (4, 128)
    op = f"{name}_bs{batch}_pl{prompt}_{mode}_{disc['config_name']}"
    tags = "".join(
        f'<property name="tag" value="{t}"/>'
        for t in (
            f"op__{op}",
            f"model__{name}",
            f"config__{disc['config_name']}",
            f"mode__{mode}",
            f"batch_size__{batch}",
            f"prompt_length__{prompt}",
        )
    )
    return f'<testcase classname="{CLASSNAME}" name="perf_{op}_{METRIC}_ms" time="1.0"><properties>{tags}</properties></testcase>'


def report_xml(cases):
    """The report.xml the perf suite writes for these cases: one wall-clock testcase per row."""
    seen, tcs = set(), []
    for _, inp, _ in cases:
        key = row_key(inp["name"], inp["disc"])
        if key in seen:
            continue
        seen.add(key)
        if inp["disc"]["record_type"] == "model":
            tcs.append(model_testcases(inp))
        else:
            tcs.append(
                f'<testcase classname="{CLASSNAME}" name="perf_{key[0]}_{METRIC}_ms_{key[3]}" time="1.0"/>'
            )
    return (
        '<?xml version="1.0"?><testsuites><testsuite name="spyre-perf-suite" '
        f'timestamp="2026-01-01T00:00:00Z">{"".join(tcs)}</testsuite></testsuites>'
    )


def ingest_report_ids(results, BenchmarkId, cases):
    """{row_key: id} the ingest writes for the synthetic report."""
    with tempfile.TemporaryDirectory() as d:
        xml = Path(d) / "report.xml"
        xml.write_text(report_xml(cases))
        _, records = results.parse_benchmark_xml(xml, "bench-conformance", "", "x86_64")
    return {
        row_key(e["name"], e["disc"]): BenchmarkId.derive(
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


def shape_collisions(cases):
    """Cases of one op whose shapes share their digits: kernel breakdown tags are matched on the digits."""
    seen, out = {}, []
    for case, inp, _ in cases:
        disc = inp["disc"]
        if disc["record_type"] == "model":
            continue
        key = (inp["name"], tuple(re.findall(r"\d+", disc["input_shapes"])))
        if key in seen:
            out.append(f"{case} and {seen[key]} share the digits of their shapes")
        seen[key] = case
    return out


def table(results, BenchmarkId, cases, report_ids):
    """(rows, mismatches): golden rows and every case whose definition disagrees with the ingest parser."""
    keys = results._V2_BENCH_ID_KEYS
    comp = results.COMPONENT_DEFAULT
    rows, bad = [], []
    for case, inp, stored in cases:
        name, disc = inp["name"], inp["disc"]
        key = (disc["config_name"], disc["input_shapes"], disc["run_mode"])
        if inp["component"] != comp or list(inp["disc_keys"]) != list(keys):
            bad.append(
                f"{case}: component/keys {inp['component']}/{inp['disc_keys']} differ from the ingest's"
            )
        written = report_ids.get(row_key(name, disc))
        if written != stored:
            bad.append(
                f"{case}: definition derives {stored}, the ingest writes {written}"
            )
        rows.append((comp, name, "report", *key, "", written or ""))
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
            rows.append((comp, name, "kernel", *key, e["disc"]["kernel_name"], kid))
        total = {**disc, "kernel_name": "Total", "is_total": 1}
        rows.append(
            (
                comp,
                name,
                "total",
                *key,
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
    msgs = lint(catalog) + shape_collisions(cases)
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
            tuple(ln.split("\t")[:7]): ln.split("\t")[7]
            for ln in Path(golden).read_text().splitlines()[1:]
        }
    for r in rows:
        if old.get(r[:7]) != r[7]:
            msgs.append(
                f"{r[1]} {r[2]} {r[6]}: derived {r[7]}, golden {old.get(r[:7])}"
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
            f"notice: identity conformance skipped, missing module: {e.name} "
            "(install: uv pip install ./extensions/spyre-bench ./extensions/clickhouse-ingest lxml)",
            file=sys.stderr,
        )
        soft = os.environ.get("BENCH_OPTIONAL") and not os.environ.get("BENCH_REQUIRED")
        return 0 if soft else 2
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
