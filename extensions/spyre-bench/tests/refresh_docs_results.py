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

"""Build examples/results/docs-analysis.results.json, the synthetic Results behind the Analysis guide.

    python tests/refresh_docs_results.py [--check]

Four Results (two cases x two pinning policies) with a per-element cost of 2.0e-6 s unpinned and 1.8e-6 s pinned.
Everything is synthetic.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "examples" / "results" / "docs-analysis.results.json"
METRIC = "kernel.time.mean"
PER_ELEMENT = {"two-cards": 2.0e-6, "two-cards-local": 1.8e-6}
SIZE = {"op_scale_n1024": 1024, "op_scale_n4096": 4096}


def build():
    from spyre_bench import engine

    cat = engine.Catalog([ROOT / "examples" / "docs-analysis"], [], [])
    if cat.errors:
        raise SystemExit("\n".join(cat.errors))
    by = {c["name"]: c for c in cat.cases}
    out = []
    for env, cost in PER_ELEMENT.items():
        view = cat.env_view(env)
        pin = view["cpu_pinning"]["policy"]
        for name, size in SIZE.items():
            reps = [
                {
                    "run.repeat": r,
                    "placement": {"nodes": ["host-a"]},
                    "samples": {METRIC: [size * cost * (1 + d)]},
                    "props": {
                        "run.exec.cpu_pinning": pin,
                        "run.exec.pinning_verified": "true" if pin != "none" else "unknown",
                    },
                }
                for r, d in enumerate((-0.01, 0.0, 0.01), 1)
            ]
            out.append(
                engine.build_result(
                    cat,
                    by[name],
                    name,
                    {"artifact_id": "9b8e17c5-d204-5f60-8a31-2c7f90e4b1d2"},
                    {"env.fp.kernel": "kernel-x"},
                    {"env.node": "host-a"},
                    view["labels"],
                    reps,
                )
            )
    return json.dumps(out, indent=1) + "\n"


if __name__ == "__main__":
    text = build()
    if "--check" in sys.argv:
        sys.exit(0 if OUT.is_file() and OUT.read_text() == text else 1)
    OUT.write_text(text)
