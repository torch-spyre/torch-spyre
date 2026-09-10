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
    TORCH_SPYRE_FFDC=1 TORCH_COMPILE_DEBUG=1 python3 tools/ffdc_trigger.py

When ``TORCHINDUCTOR_CACHE_DIR`` is unset, this tool creates an isolated
temp dir (``ffdc_inductor_cache_*``) so reports do not land in the shared
cache. That dir is left on disk so a later interpreter can export the
printed ``TORCHINDUCTOR_CACHE_DIR`` and call
``torch.spyre.get_diagnostic_report()``. It also prints a commented
``rm -rf <dir>`` to run only after that later retrieve.
"""

import glob
import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

# ``cache_dir()`` assigns ``TORCHINDUCTOR_CACHE_DIR`` when unset. Do that
# into an isolated tree *before* ``import torch`` (autoload → fallbacks).
_auto_cache_dir: str | None = None
if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
    _auto_cache_dir = tempfile.mkdtemp(prefix="ffdc_inductor_cache_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _auto_cache_dir

import torch  # noqa: E402

from torch_spyre.constants import DEVICE_NAME  # noqa: E402
from torch_spyre.execution import kernel_runner as kr  # noqa: E402
from torch_spyre.execution.kernel_runner import (  # noqa: E402
    SpyreSDSCKernelRunner,
    SpyreUnimplementedRunner,
)
from torch_spyre.profiler._ffdc import _default_output_dir, _report_sort_key  # noqa: E402


def _category_pattern(output_dir: Path, category: str) -> str:
    return str(output_dir / f"ffdc_{category}_*.json")


def _new_report_path(pattern: str, before: set[str]) -> str | None:
    """Newest report path in ``glob(pattern)`` that was not in ``before``."""
    new = [Path(path) for path in glob.glob(pattern) if path not in before]
    keyed = []
    for path in new:
        sort_key = _report_sort_key(path)
        if sort_key is not None:
            keyed.append((sort_key, path))
    if not keyed:
        return None
    return str(max(keyed, key=lambda item: item[0])[1].resolve())


def _print_collector_stats(collector: dict[str, Any]) -> None:
    print(
        f"  completeness={collector['completeness_pct']}%  "
        f"latency={collector['capture_latency_ms']}ms  "
        f"missing={collector['missing_fields']}"
    )


def _write_minimal_spyrecode(parent: str) -> str:
    """Write ``parent/spyreCodeDir`` for ``prepare_kernel``; return ``parent``."""
    spyrecode_dir = os.path.join(parent, "spyreCodeDir")
    os.makedirs(spyrecode_dir, exist_ok=True)
    spyrecode_json = {
        "JobPreparationPlan": [
            {"command": "Allocate", "properties": {"size": "1024"}},
            {
                "command": "InitTransfer",
                "properties": {
                    "init_bin_file": "init_binary.bin",
                    "dev_ptr": "120259084288",
                    "size": "1024",
                },
            },
        ],
        "JobExecPlan": [
            {
                "command": "ComputeOnDevice",
                "properties": {"job_bin_ptr": "120259084288"},
            }
        ],
    }
    with open(os.path.join(spyrecode_dir, "spyrecode.json"), "w") as f:
        json.dump(spyrecode_json, f)
    with open(os.path.join(spyrecode_dir, "init_binary.bin"), "wb") as f:
        f.write(b"\x00" * 1024)
    return parent


def _record_report(reports, category: str, before: set[str], output_dir: Path) -> None:
    report_path = _new_report_path(_category_pattern(output_dir, category), before)
    if report_path is None:
        print("  [WARN] No new report found — check FFDC output_dir")
        return
    print(f"  Report written: {report_path}")
    if not hasattr(torch, "spyre") or not hasattr(torch.spyre, "get_diagnostic_report"):
        raise SystemExit("torch.spyre.get_diagnostic_report is not bound")
    print("  using torch.spyre.get_diagnostic_report()")
    retrieved = torch.spyre.get_diagnostic_report()
    if retrieved is None:
        raise SystemExit("get_diagnostic_report returned None")
    print(f"  failure.category : {retrieved['failure']['category']}")
    print(f"  _report_path     : {retrieved['_report_path']}")
    if Path(retrieved["_report_path"]).resolve() != Path(report_path).resolve():
        raise SystemExit("get_diagnostic_report selected a different file")
    if retrieved["failure"]["category"] != category:
        raise SystemExit("retrieved category does not match this scenario")
    reports.append((category, retrieved))
    _print_collector_stats(retrieved["collector"])


def _print_isolated_cleanup() -> None:
    if _auto_cache_dir is None:
        return
    print("\n# After a new interpreter has retrieved the report:")
    print("# rm -rf " + shlex.quote(_auto_cache_dir))


def main():
    print("\n=== FFDC Real Trigger ===\n")
    reports = []
    os.environ.setdefault("TORCH_SPYRE_FFDC", "1")
    output_dir = _default_output_dir()
    print(
        "export TORCHINDUCTOR_CACHE_DIR="
        + shlex.quote(os.environ.get("TORCHINDUCTOR_CACHE_DIR") or "")
    )
    if _auto_cache_dir is not None:
        print("Isolated TORCHINDUCTOR_CACHE_DIR is left on disk.")
    print(f"FFDC output_dir: {output_dir}")

    try:
        _run_scenarios(reports, output_dir)
    finally:
        _print_isolated_cleanup()


def _run_scenarios(reports, output_dir: Path) -> None:
    # Survive prepare_kernel in __init__, then fail in run() / launch_jobplan.
    os.environ.pop("DUMP_SPYRE_CODE", None)

    print("Scenario A: SpyreSDSCKernelRunner.run() → launch_jobplan raises")
    with tempfile.TemporaryDirectory(prefix="ffdc_spyrecode_") as tmp:
        code_dir = _write_minimal_spyrecode(tmp)
        runner = None
        try:
            torch.zeros(1, device=DEVICE_NAME)
            runner = SpyreSDSCKernelRunner(
                name="test_kernel_add",
                code_dir=code_dir,
            )
        except Exception as e:
            print(f"  [SKIP] prepare_kernel failed in __init__: {e}")

        if runner is not None:

            def boom(*_args, **_kwargs):
                raise RuntimeError("ffdc launch boom")

            before = set(glob.glob(_category_pattern(output_dir, "runtime_launch")))
            with patch.object(kr, "launch_jobplan", boom):
                try:
                    runner.run()
                except RuntimeError as e:
                    print(f"  Exception re-raised (expected): {e}")
                else:
                    raise AssertionError(
                        "Expected RuntimeError from runner.run() but none was raised"
                    )
            _record_report(reports, "runtime_launch", before, output_dir)

    # ── Scenario B: unimplemented op failure ────────────────────────────────────
    print(
        "\nScenario B: SpyreUnimplementedRunner.run() → unimplemented op → FFDC fires"
    )
    urunner = SpyreUnimplementedRunner(
        name="test_kernel_fft",
        op="aten::fft_fft",
    )
    before = set(glob.glob(_category_pattern(output_dir, "unimplemented")))
    try:
        urunner.run()
    except RuntimeError as e:
        print(f"  Exception re-raised (expected): {e}")
    else:
        raise AssertionError(
            "Expected RuntimeError from urunner.run() but none was raised"
        )

    _record_report(reports, "unimplemented", before, output_dir)

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
