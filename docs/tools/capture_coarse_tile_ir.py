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
"""Capture real IR/OpSpec/bundle.mlir for the coarse-tiling loops doc's
small example (test_hint_nested_loop_with_scratchpad).

See docs/tools/README.md for usage. In short:

    rm -rf /tmp/torchinductor_$USER
    python3 docs/tools/capture_coarse_tile_ir.py > /tmp/coarse_tile_capture.txt 2>&1

Set SPYRE_LOG_PASSES=_maybe_coarse_tile_hints (or another pass name, or
"all") together with --debug to additionally dump per-pass IR snapshots,
e.g. immediately after coarse_tile() stamps loop_info and before later
passes (split_multi_ops, stickification, work division, scratchpad
planning) touch anything.
"""

import argparse
import glob
import logging
import os
import sys
from unittest.mock import patch as mock_patch

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "tests", "inductor")
)

from torch._inductor.runtime.runtime_utils import cache_dir  # noqa: E402
from torch._inductor.utils import run_and_get_code  # noqa: E402

import torch_spyre  # noqa: E402,F401
from torch_spyre._inductor import config  # noqa: E402
from torch_spyre._inductor.wsr import propagate_named_dims as _pnd  # noqa: E402

_declare_tensor_dim = _pnd.declare_tensor_dim
_name_tensor_dims = _pnd.name_tensor_dims
from torch_spyre._inductor import spyre_hint  # noqa: E402

_LAUNCH_JOBPLAN = "torch_spyre.execution.kernel_runner.launch_jobplan"
_PREPARE_KERNEL = "torch_spyre.execution.kernel_runner.prepare_kernel"

DEVICE = torch.device("spyre")


def _setup_logging(debug: bool) -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
    passes_logger = logging.getLogger("spyre.inductor.passes")
    passes_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    passes_logger.addHandler(handler)
    passes_logger.propagate = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Log at DEBUG instead of INFO. Combine with SPYRE_LOG_PASSES=<pass"
            " name|all> to also dump per-pass IR snapshots, not just the final"
            " AFTER PRE-SCHEDULING dump."
        ),
    )
    parser.add_argument(
        "--outer-tiles",
        type=int,
        default=2,
        help="num_tiles_per_dim for the outer spyre_hint (dim A). Default: 2.",
    )
    parser.add_argument(
        "--inner-tiles",
        type=int,
        default=4,
        help="num_tiles_per_dim for the inner spyre_hint (dim B). Default: 4.",
    )
    parser.add_argument(
        "--size-a", type=int, default=1024, help="Size of dim A. Default: 1024."
    )
    parser.add_argument(
        "--size-b", type=int, default=4096, help="Size of dim B. Default: 4096."
    )
    parser.add_argument(
        "--sencores", type=int, default=4, help="SENCORES value. Default: 4."
    )
    args = parser.parse_args()

    _setup_logging(args.debug)

    a = torch.randn(args.size_a, args.size_b, dtype=torch.float16)
    b = torch.randn(args.size_a, args.size_b, dtype=torch.float16)
    c = torch.randn(args.size_a, args.size_b, dtype=torch.float16)
    a_dev = a.to(DEVICE)
    b_dev = b.to(DEVICE)
    c_dev = c.to(DEVICE)

    _declare_tensor_dim("A", args.size_a)
    _declare_tensor_dim("B", args.size_b)
    for t in (a_dev, b_dev, c_dev):
        _name_tensor_dims(t, ["A", "B"])

    outer_tiles, inner_tiles = args.outer_tiles, args.inner_tiles

    def fn(a, b, c):
        with spyre_hint(num_tiles_per_dim={"A": outer_tiles}):
            with spyre_hint(num_tiles_per_dim={"B": inner_tiles}):
                y = a + b
                z = y * c
                return z

    with config.patch(
        {
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
            "sencores": args.sencores,
        }
    ):
        cfn = torch.compile(fn)

        spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
        before = set(glob.glob(f"{spyre_dir}/*"))

        print("=" * 80)
        print("BEGIN graph.operations IR dump (interleaved via logger)")
        print("=" * 80)

        with (
            mock_patch(_LAUNCH_JOBPLAN),
            mock_patch(_PREPARE_KERNEL),
            mock_patch("subprocess.run"),
        ):
            _, source_codes = run_and_get_code(cfn, a_dev, b_dev, c_dev)

        print("=" * 80)
        print("END graph.operations IR dump")
        print("=" * 80)

        print("=" * 80)
        print("BEGIN generated OpSpec/LoopSpec wrapper source")
        print("=" * 80)
        print(source_codes[0])
        print("=" * 80)
        print("END generated OpSpec/LoopSpec wrapper source")
        print("=" * 80)

        new_dirs = set(glob.glob(f"{spyre_dir}/*")) - before
        for d in sorted(new_dirs):
            mlir_path = os.path.join(d, "bundle.mlir")
            if os.path.exists(mlir_path):
                print("=" * 80)
                print(f"BEGIN bundle.mlir ({mlir_path})")
                print("=" * 80)
                with open(mlir_path) as f:
                    print(f.read())
                print("=" * 80)
                print("END bundle.mlir")
                print("=" * 80)


if __name__ == "__main__":
    main()
