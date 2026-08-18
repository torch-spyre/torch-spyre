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

"""Locate the measurement database the cost model is scored against.

**Generate your own.** The database is measured device time, so it belongs to the machine
and toolchain that produced it. The compiler is under active development and kernel
performance moves with it: a spot-check while preparing this feature measured 261 us for a
configuration the bundled reference has at 390 us, on the same shape and core count. Numbers
taken on someone else's build describe their build.

    python3 docs/source/user_guide/examples/run_cost_model_sweep.py

That re-measures every configuration and writes ``sweep_records.json`` beside this file. It
needs Spyre hardware and takes a few hours.

A REFERENCE COPY, for orientation rather than for scoring your build, is kept at
``REFERENCE_URL`` below. It was collected on PyTorch 2.11 and is the database the accuracy
figures quoted in ``docs/source/compiler/cost_model.md`` were computed from. Fetch it
deliberately if you want a starting point; nothing downloads it on your behalf.

Resolution order, first hit wins:

1. an explicit path passed by the caller (``--records``)
2. ``$SPYRE_COST_MODEL_RECORDS`` -- a path to a local copy
3. ``sweep_records.json`` beside this file

Nothing here needs hardware; the database holds measured times, and re-scoring only re-runs
the model over them.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_HERE, "sweep_records.json")

#: A reference database for orientation, NOT a default. It is one machine's measurements on
#: PyTorch 2.11 and will drift from any current build. Nothing fetches it automatically.
REFERENCE_URL = (
    "https://raw.githubusercontent.com/HieronZhang/torch-spyre/"
    "prepare_pr/tools/cost_model/sweep_records.json"
)

_HELP = f"""\
No cost-model measurement database found.

The database is measured device time and belongs to the build that produced it. Generate
one for this machine:

    python3 docs/source/user_guide/examples/run_cost_model_sweep.py

(needs Spyre hardware; a few hours). It writes {CACHE}

Or point at a copy you already have:

    export SPYRE_COST_MODEL_RECORDS=/path/to/sweep_records.json

A reference database collected on PyTorch 2.11 -- the one the accuracy figures in the cost
model report were computed from -- is available for orientation. Kernel performance changes
as the compiler develops, so treat it as a starting point, not as a measurement of your
build:

    {REFERENCE_URL}
"""


def find_records(explicit=None):
    """Absolute path to the database, or None if there is not one.

    Separate from ``records_path`` because not every caller should die when the database
    is absent: the sweep runner can be told which configurations to measure directly, and
    on a machine that has never swept, "no database" is its normal starting state.
    """
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"no such records file: {explicit}")
        return explicit

    env = os.environ.get("SPYRE_COST_MODEL_RECORDS")
    if env:
        if not os.path.exists(env):
            sys.exit(f"SPYRE_COST_MODEL_RECORDS points at a missing file: {env}")
        return env

    return CACHE if os.path.exists(CACHE) else None


def records_path(explicit=None):
    """Absolute path to the database.

    Raises SystemExit with instructions rather than a traceback: every caller is a
    command-line tool, and a missing database is a setup step, not a bug. Never downloads
    anything -- fetching someone else's measurements silently would make a stale reference
    look like a measurement of the machine it is running on.
    """
    path = find_records(explicit)
    if path is None:
        sys.exit(_HELP)
    return path
