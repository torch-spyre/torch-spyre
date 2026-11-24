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

"""A serving-benchmark stand-in for the developer guides: reads the dataset path the launcher exports, writes metrics.json."""

import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repeat", type=int, required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

trace = os.environ.get("DEMO_TRACE_PATH", "unset")
scale = 1 + a.repeat / 100
raw = Path(a.raw)
raw.mkdir(parents=True, exist_ok=True)
(raw / f"serve-{a.repeat}.json").write_text(json.dumps({"trace": trace, "ttft_s": 0.2 * scale}))
(Path(a.out) / "metrics.json").write_text(
    json.dumps({"samples": {"llm.ttft.p99": [0.2 * scale], "llm.itl.median": [0.02 * scale]}})
)
