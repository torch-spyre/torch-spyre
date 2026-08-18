# Copyright 2025 The Torch-Spyre Authors.
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

"""Shared output sink for the cost-model dumps.

Used by ``dump_cost_model`` (the per-op feature dump) and ``cost_model_pass`` (the
per-kernel report). Both are gated by ``config.cost_model``; this module only decides
WHERE the text goes -- stderr, or the file named by ``SPYRE_DUMP_COST_FILE``.
"""

import os
import sys


def emit(text: str) -> None:
    """Write one dump record to the configured sink (file or stderr)."""
    dest = os.environ.get("SPYRE_DUMP_COST_FILE")
    if dest:
        with open(dest, "a", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    else:
        sys.stderr.write(text)
        sys.stderr.write("\n")
        sys.stderr.flush()


def banner(title: str) -> str:
    """Return a boxed section header for a dump record."""
    bar = "=" * 78
    return f"{bar}\n==== {title}\n{bar}"
