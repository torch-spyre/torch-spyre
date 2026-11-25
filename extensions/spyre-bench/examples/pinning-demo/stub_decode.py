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

"""A stand-in decode benchmark: synthetic latencies, 25 % cheaper per token when its CPU affinity is narrower than
the machine's. It exists to exercise pinning end to end (launcher, observer, derive, compare); it measures nothing."""

import argparse
import json
import os
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out-len", type=int, required=True)
ap.add_argument("--repeat", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--hold", type=float, default=1.2, help="seconds to stay alive so the observer can sample the tree")
a = ap.parse_args()

affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
pinned = len(affinity) < (os.cpu_count() or 1)
helper = threading.Thread(target=time.sleep, args=(a.hold,))
helper.start()
time.sleep(a.hold)
helper.join()
per_token = 0.020 * (0.75 if pinned else 1.0) * (1 + 0.01 * ((a.repeat * 7) % 5 - 2))
latency = 0.2 + a.out_len * per_token
if os.environ.get("LOCAL_RANK", "0") == "0":
    Path(a.out, "metrics.json").write_text(json.dumps({"samples": {"llm.batch_latency.mean": [latency]}}))
