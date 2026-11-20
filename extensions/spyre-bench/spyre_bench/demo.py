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

"""Plan the PostgreSQL example, simulate its runner with synthetic samples and write the plan and the Results.

  python -m spyre_bench.demo [--check]    # --check: exit 1 if examples/results/ differs from a fresh build
Numbers are synthetic; the point is the shape, per-repeat placement, and that every id recomputes from a Result alone.
"""

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from spyre_bench import engine

ROOT = Path(__file__).resolve().parent.parent  # source checkout only: examples are not installed
OUT = ROOT / "examples" / "results"
REQUEST = ROOT / "examples" / "requests" / "pg-nightly.yaml"
FINGERPRINT = {
    "env.fp.kernel": "6.8.0-45",
    "env.fp.os.version": "os-a",
    "env.fp.governor": "performance",
    "env.fp.cpu.model": "cpu-model-a",
    "env.fp.cpu.base_ghz": 2.4,
    "env.fp.pg.server_version": "16.6",
}
NOW = datetime(2026, 11, 13, 6, 0, tzinfo=timezone.utc)
PLACEMENTS = [
    {"nodes": ["host-a"], "devices": [{"serial": "serial-a"}]},
    {"nodes": ["host-b"], "devices": [{"serial": "serial-b"}]},
    {"nodes": ["host-a"], "devices": [{"serial": "serial-a"}]},
]


def build():
    cat = engine.Catalog([ROOT / "examples" / "postgres-demo"], [ROOT / "examples" / "packs" / "postgres-example"])
    if cat.errors:
        raise SystemExit("\n".join(cat.errors))
    request = yaml.safe_load(REQUEST.read_text())
    view = cat.env_view(request["environment"])
    env_id = engine.env_id_of(FINGERPRINT)
    leg = {"arch": "x86_64", "devices": view["device_totals"], "nodes": 1, "reference_node": True, "env_id": env_id}
    plan = engine.plan(cat, request, leg, now=NOW)
    by = {c["name"]: c for c in cat.cases}
    results = []
    for i, p in enumerate(plan["cases"]):
        repeats = []
        for r in range(1, p["repeats"] + 1):
            tps = [100.0 + 3 * i + r + 0.1 * k for k in range(5)]
            repeats.append(
                {
                    "run.repeat": r,
                    "placement": PLACEMENTS[(r - 1) % len(PLACEMENTS)],
                    "clean": {
                        "exclusive": "observed",
                        "load_clean": True,
                        "devices_exclusive": "enforced",
                        "other_load": "unknown",
                    },
                    "samples": {"pg.tps": tps, "pg.latency.mean": [round(1 / t, 6) for t in tps]},
                }
            )
        vals = [statistics.median(r["samples"]["pg.tps"]) for r in repeats]
        facts = {
            "qualified": True,
            "clean": repeats[0]["clean"],
            "repeats": len(repeats),
            "warmup_discarded": 1,
            "placement": True,
            "cv_pct": statistics.pstdev(vals) / statistics.fmean(vals) * 100,
        }
        state, reasons = engine.validity_of(cat.policy.spec, "nightly", facts)
        results.append(
            engine.build_result(
                cat,
                by[p["name"]],
                p["name"],
                request["subject"],
                FINGERPRINT,
                {"env.node": "host-a", "env.uptime_s": 86400 + i},
                view["labels"],
                repeats,
                request=({"name": request["environment"], "spec_digest": view["spec_digest"]}),
                validity=(state, reasons, request["purpose"], "internal"),
            )
        )
    return {
        "pg-nightly.plan.yaml": engine.dump_yaml(plan),
        "pg-nightly.results.json": json.dumps(results, indent=1) + "\n",
    }


def main(argv=None):
    files = build()
    if "--check" in (argv if argv is not None else sys.argv[1:]):
        stale = [n for n, t in files.items() if not (OUT / n).exists() or (OUT / n).read_text() != t]
        print("stale:" if stale else "results current", *stale)
        return 1 if stale else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for n, t in files.items():
        (OUT / n).write_text(t)
    print("wrote", *files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
