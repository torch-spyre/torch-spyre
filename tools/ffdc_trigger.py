#!/usr/bin/env python3
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

"""
FFDC trigger — exercises kernel_runner.py exception paths on real hardware.

Drives SpyreSDSCKernelRunner and SpyreUnimplementedRunner through their
real exception paths so FFDC fires on a genuine traceback with real
hardware state and compiler artifacts.

Run from repo root with:
    USE_SPYRE_PROFILER=1 TORCH_COMPILE_DEBUG=1 python3 tools/ffdc_trigger.py
"""

import glob
import json
import os
import time
from typing import Any

import torch  # noqa: F401 — ensures torch_spyre._C loads via real extension
import torch_spyre  # noqa: F401

from torch_spyre.execution.kernel_runner import (
    SpyreSDSCKernelRunner,
    SpyreUnimplementedRunner,
)
from torch_spyre.profiler._ffdc import _default_output_dir

FFDC_OUT = _default_output_dir()


def _newest_since(pattern, since_ts):
    matches = [
        m for m in glob.glob(pattern, recursive=True) if os.path.getmtime(m) > since_ts
    ]
    return max(matches, key=os.path.getmtime) if matches else None


def _print_collector_stats(collector: dict[str, Any]) -> None:
    print(
        f"  completeness={collector['completeness_pct']}%  "
        f"latency={collector['capture_latency_ms']}ms  "
        f"missing={collector['missing_fields']}"
    )


def main():
    print("\n=== FFDC Real Trigger ===\n")
    reports = []
    os.environ.setdefault("USE_SPYRE_PROFILER", "1")

    # ── Scenario A: runtime_launch failure ──────────────────────────────────────
    # SpyreSDSCKernelRunner.__init__ calls prepare_kernel(); with a fake code_dir
    # that fails or run() fails in launch_jobplan, FFDC should capture
    # CATEGORY_RUNTIME_LAUNCH.
    os.environ.pop("DUMP_SPYRE_CODE", None)

    print("Scenario A: SpyreSDSCKernelRunner.run() → launch_kernel() raises")
    runner = SpyreSDSCKernelRunner(
        name="test_kernel_add",
        code_dir="/tmp/fake_spyre_code_dir",
    )
    t0 = time.time()
    try:
        runner.run()
    except RuntimeError as e:
        print(f"  Exception re-raised (expected): {e}")
    else:
        raise AssertionError(
            "Expected RuntimeError from runner.run() but none was raised"
        )

    report_path = _newest_since(str(FFDC_OUT / "ffdc_runtime_launch_*.json"), t0)
    if report_path:
        with open(report_path) as f:
            report = json.load(f)
        reports.append(("runtime_launch", report))
        print(f"  Report written: {report_path}")
        _print_collector_stats(report["collector"])
    else:
        print("  [WARN] No report found — check FFDC output_dir")

    # ── Scenario B: unimplemented op failure ────────────────────────────────────
    print(
        "\nScenario B: SpyreUnimplementedRunner.run() → unimplemented op → FFDC fires"
    )
    urunner = SpyreUnimplementedRunner(
        name="test_kernel_fft",
        op="aten::fft_fft",
    )
    t0 = time.time()
    try:
        urunner.run()
    except RuntimeError as e:
        print(f"  Exception re-raised (expected): {e}")
    else:
        raise AssertionError(
            "Expected RuntimeError from urunner.run() but none was raised"
        )

    report_path_u: Any | None = _newest_since(
        str(FFDC_OUT / "ffdc_unimplemented_*.json"), t0
    )
    if report_path_u:
        with open(report_path_u) as f:
            report_u = json.load(f)
        reports.append(("unimplemented", report_u))
        print(f"  Report written: {report_path_u}")
        _print_collector_stats(report_u["collector"])
    else:
        print("  [WARN] No report found — check FFDC output_dir")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n=== Captured Report Fields ===")
    for cat, r in reports:
        print(f"\n[{cat}]")
        print(f"  failure.exception_type : {r['failure']['exception_type']}")
        print(f"  failure.message        : {r['failure']['message'][:80]}")
        tb = r["failure"]["traceback"]
        tb_str = tb if isinstance(tb, str) else "".join(tb)
        print(f"  failure.traceback_lines: {len(tb_str.splitlines())}")
        print(f"  metadata.torch_version : {r['metadata'].get('torch_version', 'N/A')}")
        print(
            f"  metadata.torch_spyre_version : "
            f"{r['metadata'].get('torch_spyre_version', 'N/A')}"
        )
        print(f"  artifacts.found_count  : {r['artifacts']['found_count']}")
        print(f"  hardware_state         : {r['hardware_state']}")


if __name__ == "__main__":
    main()
