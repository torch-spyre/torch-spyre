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

"""Names of the run-context sidecar files beside a vLLM bench's results (stdlib only)."""

KINDS = ("env", "phases", "host_resources", "outputs", "compile_summary")
SIDECAR_FILES = frozenset(f"{k}.json" for k in KINDS)


def is_sidecar(filename: str) -> bool:
    """A `<kind>.json` or `<test_name>.<kind>.json` file; not a benchmark record file."""
    return filename in SIDECAR_FILES or any(
        filename.endswith(f".{k}.json") and k != "host_resources" for k in KINDS
    )
