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
audit.py — Source-to-Kernel Provenance Audit for SimpleMLP (issue #2574).

Runs ONE cache-defeated torch.compile under the in-process capture layer
(pipeline/captures.py), writes the raw captured structure to
provenance_capture_raw.json, then renders the Markdown report.

Usage:
    python audit.py [--raw provenance_capture_raw.json]

Notes:
  * Caches are force-disabled and Dynamo is reset so codegen actually runs
    (a cache hit silently skips create_op_spec / define_kernel).
  * torch.export is intentionally NOT used: it is a separate front-end, not on
    the compile path. Every stage here reads the real compile-path object.
"""

import argparse
import json
import os
import pathlib

# Force-disable Inductor caches before torch is imported anywhere downstream.
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch
import torch_spyre  # noqa: F401 — registers the Spyre backend

from reference_mlp import SimpleMLP
from pipeline import captures, superdsc, report

BATCH_SIZE = 2
INPUT_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 128

DEFAULT_RAW = pathlib.Path(__file__).parent / "provenance_capture_raw.json"
DEFAULT_REPORT = pathlib.Path(__file__).parent / "provenance_audit.md"


def _stage_summary(results: dict) -> None:
    """Print a one-line fired/count summary per stage so mismatches are obvious."""
    s = results
    print("\nCapture summary (fired? / count):")

    def line(key: str, detail: str) -> None:
        print(f"  {key:<25}: {detail}")

    line(
        "stage2_pre_grad",
        f"{s['stage2_pre_grad']['fired']}  nodes={len(s['stage2_pre_grad']['nodes'])}",
    )
    line(
        "stage2_post_grad",
        f"{s['stage2_post_grad']['fired']}  "
        f"nodes={len(s['stage2_post_grad']['nodes'])}",
    )
    line(
        "stage3_looplevel_prepass",
        f"{s['stage3_looplevel_prepass']['fired']}  "
        f"ops={len(s['stage3_looplevel_prepass']['operations'])}",
    )
    line(
        "stage4_looplevel_postpass",
        f"{s['stage4_looplevel_postpass']['fired']}  "
        f"ops={len(s['stage4_looplevel_postpass']['operations'])}",
    )
    line(
        "stage5_opspec",
        f"{s['stage5_opspec']['fired']}  "
        f"ops={len(s['stage5_opspec']['ops'])}  "
        f"opspec_fields={s['stage5_opspec']['opspec_fields']}",
    )
    line(
        "stage6_kernels",
        f"{s['stage6_kernels']['fired']}  "
        f"kernels={len(s['stage6_kernels']['kernels'])}  "
        f"dirs={len(s['stage6_kernels']['output_dirs'])}",
    )
    print(f"\n  _hooks: {json.dumps(s['_hooks'], indent=2)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Provenance capture for SimpleMLP")
    parser.add_argument(
        "--raw",
        type=pathlib.Path,
        default=DEFAULT_RAW,
        help="Path for the raw captured JSON dump",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_REPORT,
        help="Path for the Markdown audit report",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  PROVENANCE AUDIT — SimpleMLP")
    print("  torch-spyre issue #2574")
    print("=" * 72)
    print(
        f"  Config: batch={BATCH_SIZE}, input={INPUT_DIM}, "
        f"hidden={HIDDEN_DIM}, output={OUTPUT_DIM}"
    )
    print("  Caches force-disabled; first device access triggers Spyre init.")

    # Defeat caches so scheduling + codegen actually run this process.
    try:
        torch._inductor.config.force_disable_caches = True
    except Exception as e:
        print(f"  (warn) could not set force_disable_caches: {e}")
    torch._dynamo.reset()

    model = SimpleMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).half().to("spyre")
    model.eval()
    x = torch.randn(BATCH_SIZE, INPUT_DIM, dtype=torch.float16, device="spyre")

    compiled = torch.compile(model)
    print("\nRunning torch.compile under capture (codegen runs this process)...")
    with captures.capture() as results, torch.no_grad():
        _ = compiled(x)

    _stage_summary(results)

    args.raw.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Raw capture written to: {args.raw}")

    # Stage 6: read the emitted bundles from the exact captured dirs.
    bundles = superdsc.run(results["stage6_kernels"]["output_dirs"])
    print(
        f"  SuperDSC: {bundles['total_kernels']} kernels, "
        f"{bundles['total_sdsc_files']} sdsc_*.json, "
        f"provenance_present={bundles['provenance_present']}"
    )

    md = report.render(capture=results, bundles=bundles, model_name="SimpleMLP")
    args.output.write_text(md)
    print(f"  Audit report written to: {args.output}\n")


if __name__ == "__main__":
    main()
