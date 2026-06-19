#!/usr/bin/env python3
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

"""
FFDC trigger — exercises kernel_runner.py exception paths on real hardware.

Drives SpyreSDSCKernelRunner and SpyreUnimplementedRunner through their
real exception paths so FFDC fires on a genuine traceback with real
hardware state and compiler artifacts.

Run from repo root with:
    TORCH_COMPILE_DEBUG=1 python3 tools/ffdc_trigger.py
"""

import glob
import json
import os
from typing import Any

import torch  # noqa: F401 — ensures torch_spyre._C loads via real extension
import torch_spyre  # noqa: F401

from torch_spyre.execution.kernel_runner import (
    SpyreSDSCKernelRunner,
    SpyreUnimplementedRunner,
)
from torch_spyre.profiler._ffdc import _default_output_dir

FFDC_OUT = _default_output_dir()


def _newest(pattern):
    matches = glob.glob(pattern, recursive=True)
    return max(matches, key=os.path.getmtime) if matches else None


def main():
    print("\n=== FFDC Real Trigger ===\n")
    reports = []

    # ── Scenario A: runtime_launch failure ──────────────────────────────────────
    print("Scenario A: SpyreSDSCKernelRunner.run() → launch_kernel() raises")
    runner = SpyreSDSCKernelRunner(
        name="test_kernel_add",
        code_dir="/tmp/fake_spyre_code_dir",
    )
    try:
        runner.run()
    except RuntimeError as e:
        print(f"  Exception re-raised (expected): {e}")
    else:
        raise AssertionError(
            "Expected RuntimeError from runner.run() but none was raised"
        )

    report_path = _newest(str(FFDC_OUT / "ffdc_runtime_launch_*.json"))
    if report_path:
        with open(report_path) as f:
            report = json.load(f)
        reports.append(("runtime_launch", report))
        print(f"  Report written: {report_path}")
        c = report["collector"]
        print(
            f"  completeness={c['completeness_pct']}%  latency={c['capture_latency_ms']}ms  missing={c['missing_fields']}"
        )
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
    try:
        urunner.run()
    except RuntimeError as e:
        print(f"  Exception re-raised (expected): {e}")
    else:
        raise AssertionError(
            "Expected RuntimeError from urunner.run() but none was raised"
        )

    report_path_u: Any | None = _newest(str(FFDC_OUT / "ffdc_unimplemented_*.json"))
    if report_path_u:
        with open(report_path_u) as f:
            report_u = json.load(f)
        reports.append(("unimplemented", report_u))
        print(f"  Report written: {report_path_u}")
        c = report_u["collector"]
        print(
            f"  completeness={c['completeness_pct']}%  latency={c['capture_latency_ms']}ms  missing={c['missing_fields']}"
        )
    else:
        print("  [WARN] No report found — check FFDC output_dir")

    # ── Summary ─────────────────────────────────────────────────────────────────
    print("\n=== Captured Report Fields ===")
    for cat, r in reports:
        print(f"\n[{cat}]")
        print(f"  failure.exception_type : {r['failure']['exception_type']}")
        print(f"  failure.message        : {r['failure']['message'][:80]}")
        print(
            f"  failure.traceback_lines: {len(r['failure']['traceback'].splitlines())}"
        )
        print(f"  metadata.torch_version : {r['metadata'].get('torch_version', 'N/A')}")
        print(
            f"  metadata.torch_spyre_version : {r['metadata'].get('torch_spyre_version', 'N/A')}"
        )
        print(f"  artifacts.found_count  : {r['artifacts']['found_count']}")
        print(f"  hardware_state         : {r['hardware_state']}")


if __name__ == "__main__":
    main()
