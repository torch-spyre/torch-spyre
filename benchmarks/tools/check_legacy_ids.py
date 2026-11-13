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

"""Check that the catalog's declared legacy identity inputs still derive the ids the ingest writes.

Each Benchmark in ``benchmarks/ops`` carries ``legacy.*`` annotations: the discriminators the ingest hashes
(``record_type``, ``run_mode``, ...). This tool derives an id for synthetic report and kernel rows of every
Benchmark with ``spyre_clickhouse_ingest`` itself and compares it with ``legacy-ids.golden.tsv``. An edit that
would start a new series changes a golden id and fails here.

    python benchmarks/tools/check_legacy_ids.py [--update]

Needs ``pyyaml``, ``regex`` and ``clickhouse-connect`` (the ingest imports it); exits 0 with a notice when they
are missing unless BENCH_REQUIRED is set. The shapes and kernel names are synthetic fixtures, not the
enumeration of any real run.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "benchmarks"
GOLDEN = CATALOG / "legacy-ids.golden.tsv"
SHAPES = "1_512_4096__4096_4096"
KERNELS = (
    ("spyre_kernel_v1_fused_add_aaaaaaaaaaaaaaaa#2", 0.003),
    ("spyre_kernel_v1_fused_add_bbbbbbbbbbbbbbbb#2", 0.27),
)
HEADER = "component\tname\trow\tinput_shapes\tkernel_name\tbenchmark_id"


def load_ingest():
    sys.path.insert(0, str(ROOT / "extensions" / "clickhouse-ingest"))
    from spyre_clickhouse_ingest.identity import BenchmarkId
    from spyre_clickhouse_ingest.results import _V2_BENCH_ID_KEYS

    return BenchmarkId, _V2_BENCH_ID_KEYS


def declared():
    """(component, [(name, legacy discriminators)]) from the catalog."""
    import yaml

    docs = [
        d
        for p in sorted(CATALOG.rglob("*.yaml"))
        for d in yaml.safe_load_all(p.read_text())
        if d
    ]
    component = next(
        d["spec"].get("component") or d["metadata"]["name"]
        for d in docs
        if d["kind"] == "Project"
    )
    ops = []
    for d in docs:
        if d["kind"] != "Benchmark":
            continue
        ann = d["metadata"].get("annotations") or {}
        disc = {
            k.removeprefix("legacy."): v
            for k, v in ann.items()
            if k.startswith("legacy.")
        }
        ops.append((d["metadata"]["name"], disc))
    return component, ops


def rows(ingest, keys, component, ops):
    BenchmarkId = ingest
    out = []
    for name, base in ops:
        report = {**base, "input_shapes": SHAPES}
        out.append(
            (
                component,
                name,
                "report",
                SHAPES,
                "",
                BenchmarkId.derive(component, name, [], report, keys),
            )
        )
        entries = [
            {
                "name": name,
                "tags": [],
                "backend": "spyre",
                "measurements": {"duration_ms": [ms]},
                "disc": {
                    **base,
                    "input_shapes": SHAPES,
                    "kernel_name": raw,
                    "is_total": 0,
                },
                "disc_keys": keys,
            }
            for raw, ms in KERNELS
        ]
        for e in BenchmarkId.rank_kernels(component, entries):
            kid = BenchmarkId.derive(component, name, e["tags"], e["disc"], keys)
            out.append(
                (component, name, "kernel", SHAPES, e["disc"]["kernel_name"], kid)
            )
        total = {**base, "input_shapes": SHAPES, "kernel_name": "Total", "is_total": 1}
        out.append(
            (
                component,
                name,
                "total",
                SHAPES,
                "Total",
                BenchmarkId.derive(component, name, [], total, keys),
            )
        )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--update",
        action="store_true",
        help="rewrite the golden table (an identity change)",
    )
    args = ap.parse_args(argv)
    try:
        ingest, keys = load_ingest()
        component, ops = declared()
    except ImportError as e:
        print(
            f"notice: legacy id check skipped, missing module: {e.name}",
            file=sys.stderr,
        )
        return 2 if os.environ.get("BENCH_REQUIRED") else 0
    table = rows(ingest, keys, component, ops)
    text = "\n".join([HEADER, *("\t".join(r) for r in table)]) + "\n"
    if args.update:
        GOLDEN.write_text(text)
        print(f"wrote {GOLDEN.relative_to(ROOT)}: {len(table)} ids")
        return 0
    golden = GOLDEN.read_text() if GOLDEN.exists() else ""
    if golden == text:
        print(
            f"PASS legacy ids: {len(table)} ids across {len(ops)} benchmarks match {GOLDEN.name}"
        )
        return 0
    old = {
        tuple(ln.split("\t")[:5]): ln.split("\t")[5] for ln in golden.splitlines()[1:]
    }
    for r in table:
        if old.get(r[:5]) != r[5]:
            print(
                f"FAIL {r[1]} {r[2]} {r[4]}: derived {r[5]}, golden {old.get(r[:5])}",
                file=sys.stderr,
            )
    print("identity changed: review, then rerun with --update", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
