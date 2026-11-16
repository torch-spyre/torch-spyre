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

"""A queue-benchmark stand-in for the developer guides: writes synthetic throughput, latency and backlog samples."""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bytes", type=int, required=True)
ap.add_argument("--repeat", type=int, required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

scale = 1 + a.repeat / 100
raw = Path(a.raw)
raw.mkdir(parents=True, exist_ok=True)
(raw / f"produce-{a.repeat}.json").write_text(json.dumps({"bytes": a.bytes, "msgs_per_s": 50000 / scale}))
(Path(a.out) / "metrics.json").write_text(
    json.dumps(
        {
            "samples": {
                "mq.throughput": [50000 / scale],
                "mq.latency.p99": [0.004 * scale],
                "mq.errors.ratio": [0.0001 * a.repeat],
                "mq.depth.max": [120 * a.repeat],
            }
        }
    )
)
