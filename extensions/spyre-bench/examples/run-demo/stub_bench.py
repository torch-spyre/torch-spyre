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

"""A stand-in benchmark command: writes synthetic raw output and a metrics.json sidecar. Not a measurement."""

import argparse
import json
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--op", required=True)
ap.add_argument("--shape", default="")
ap.add_argument("--raw", required=True)
ap.add_argument("--repeat", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if os.environ.get("STUB_FAIL") == a.op:
    sys.exit(3)
value = 1.0 + a.repeat / 100
raw = Path(a.raw)
raw.mkdir(parents=True, exist_ok=True)
(raw / f"{a.op}-{a.repeat}.json").write_text(
    json.dumps({"benchmarks": [{"name": a.op, "shape": a.shape, "samples": {"kernel.time.mean": [value]}}]})
)
(Path(a.out) / "metrics.json").write_text(json.dumps({"samples": {"kernel.time.mean": [value]}}))
