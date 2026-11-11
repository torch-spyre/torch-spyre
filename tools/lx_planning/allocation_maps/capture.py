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

"""Run a model/compile command while recording every CP-SAT LX allocation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: python capture.py --output-dir RUN/capture -- "
            "python run_model.py --compile"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide the command to run after --")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hook_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["SPYRE_GRAPH_CAPTURE_DIR"] = str(output_dir)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(hook_dir), existing_pythonpath) if value
    )
    metadata = {
        "schema": "spyre-lx-allocation-capture-run-v1",
        "started_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "command_shell": shlex.join(command),
        "cwd": os.getcwd(),
        "python": sys.executable,
        "capture_dir": str(output_dir),
        "environment": {
            key: env.get(key)
            for key in (
                "DXP_LX_FRAC_AVAIL",
                "LAYOUT_SOLVER",
                "CO_OPTIMIZING_LX_PLANNING",
                "LX_PLANNING",
                "HBM_POOL_PLANNING",
                "SENCORES",
                "SPYRE_NUM_CPUS",
                "TORCHINDUCTOR_CACHE_DIR",
                "TORCH_COMPILE_DEBUG",
            )
        },
    }
    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    log_path = output_dir / "command.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    metadata["finished_utc"] = datetime.now(UTC).isoformat()
    metadata["returncode"] = result.returncode
    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Command log: {log_path}")
    print(f"Capture directory: {output_dir}")
    print(f"Command exit status: {result.returncode}")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
