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

"""A stand-in quality command: writes a synthetic metrics.json sidecar. Not a measurement.

STUB_FLAW picks what is wrong with the run: digest, short, errors, truncated, unpinned or sampled.
"""

import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
flaw = os.environ.get("STUB_FLAW", "")
score = float(os.environ.get("STUB_SCORE", "0.625"))
n = 5 if flaw == "short" else 8
props = {
    "run.quality.harness": "demo-harness",
    "run.quality.harness_version": "" if flaw == "unpinned" else "1.0",
    "run.quality.protocol": "" if flaw == "unpinned" else "p-0001",
    "run.quality.model_revision": "8f2c4a6e1b3d5f708192a3b4c5d6e7f8091a2b3c",
    "run.quality.dataset_digest": "sha256:"
    + ("0" * 64 if flaw == "digest" else "4f3a1c0de2b64a5f8c7e91d0b2a36f5e4d8c7b6a5f4e3d2c1b0a99887766554a"),
    "run.quality.temperature": "0.7" if flaw == "sampled" else "0",
}
samples = {
    "llm.eval.exact_match": [score],
    "llm.eval.exact_match_stderr": [0.17],
    "llm.eval.n_samples": [n],
    "llm.eval.error_frac": [0.125 if flaw == "errors" else 0],
    "llm.eval.truncated_frac": [0.25 if flaw == "truncated" else 0],
}
(Path(a.out) / "metrics.json").write_text(json.dumps({"samples": samples, "props": props}))
