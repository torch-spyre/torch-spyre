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

"""Unit tests for bundle.py symbol-id extraction and pool emission.

compile_op_spec is mocked throughout so no Spyre hardware is required.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from torch._inductor.test_case import TestCase as InductorTestCase

from torch_spyre._inductor import config
from torch_spyre._inductor.codegen.bundle import (
    _extract_symbol_ids,
    generate_bundle,
)
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.constants import MAX_POOL_SIZE_BYTES
from torch_spyre._inductor.op_spec import OpSpec


class TestFrontendPoolAllocationConfig(unittest.TestCase):
    """Standalone sanity checks for the frontend_pool_allocation config flag."""

    def test_default_is_false(self):
        # The flag is read from FRONTEND_POOL_ALLOCATION at import time, so check
        # the code default in a fresh interpreter with that variable removed --
        # otherwise a CI job that exports it would make this test fail.
        env = {k: v for k, v in os.environ.items() if k != "FRONTEND_POOL_ALLOCATION"}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; "
                "from torch_spyre._inductor import config; "
                "print(config.frontend_pool_allocation)",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "False")

    def test_patchable(self):
        original = config.frontend_pool_allocation
        with config.patch({"frontend_pool_allocation": not original}):
            self.assertEqual(config.frontend_pool_allocation, not original)
        self.assertEqual(config.frontend_pool_allocation, original)


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _make_sdsc_json(
    sdsc_idx: int = 0,
    *,
    hbm_sym_ids_per_core: dict[str, int] | None = None,
    num_cores: int = 1,
) -> dict:
    """Minimal SDSC JSON for testing _extract_symbol_ids and generate_bundle.

    Args:
        sdsc_idx: Used to form the top-level JSON key, e.g. "0_fused_test".
        hbm_sym_ids_per_core: Mapping of core key → symbol ID for
            startAddressCoreCorelet_.data_, e.g. {"[0, 0, 0]": -2}.
        num_cores: Value for numCoresUsed_.
    """
    schedule_tree: list = []
    if hbm_sym_ids_per_core:
        schedule_tree = [
            {
                "component_": "hbm",
                "startAddressCoreCorelet_": {"data_": hbm_sym_ids_per_core},
            }
        ]
    return {
        f"{sdsc_idx}_fused_test": {
            "numCoresUsed_": num_cores,
            "dscs_": [
                {
                    "op": {
                        "scheduleTree_": schedule_tree,
                    }
                }
            ],
        }
    }


def _minimal_op_spec() -> OpSpec:
    """A stub OpSpec whose content is irrelevant (compile_op_spec is mocked)."""
    return OpSpec(
        op="gelu",
        is_reduction=False,
        iteration_space={},
        args=[],
        op_info={},
    )


class TestExtractSymbolIds(InductorTestCase):
    """Unit tests for _extract_symbol_ids."""

    def test_hbm_only(self):
        """HBM address ID from scheduleTree_ is collected."""
        sdsc_json = _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": -2})
        self.assertEqual(_extract_symbol_ids(sdsc_json), [-2])

    def test_positive_values_excluded(self):
        """Positive (concrete) addresses are ignored."""
        sdsc_json = _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": 0x400000000})
        self.assertEqual(_extract_symbol_ids(sdsc_json), [])


class TestGenerateBundlePool(InductorTestCase):
    """Integration tests for generate_bundle's HBM pool emission."""

    def setUp(self):
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()
        super().tearDown()

    def _read_bundle(self) -> str:
        with open(os.path.join(self.output_dir, "bundle.mlir")) as f:
            return f.read()

    def _run_bundle(self, compiled_entries, op_specs=None, pool_size=0):
        """Run generate_bundle with mocked compile_op_spec, return bundle.mlir text."""
        if op_specs is None:
            op_specs = [_minimal_op_spec() for _ in compiled_entries]

        # compile_op_spec is called twice per OpSpec: once as a probe to derive
        # the cache key (result discarded except for JSON), once for real.
        side_effects = [e for entry in compiled_entries for e in (entry, entry)]
        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=side_effects,
        ):
            generate_bundle(
                "test",
                self.output_dir,
                op_specs,
                pool_size=pool_size,
            )
        return self._read_bundle()

    @config.patch({"frontend_pool_allocation": False})
    def test_pool_device_mem_allocate_emitted(self):
        """A pool symbol triggers device_mem_allocate with the right byte
        count, and no %pool_base_addr parameter."""
        pool_kind = SymbolKind.pool()
        entry = (
            _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": -1}),
            [0],
            [],
            [pool_kind],
        )

        bundle = self._run_bundle([entry], pool_size=65536)

        self.assertIn(
            "%pool = sdscbundle.device_mem_allocate 65536 bytes : index",
            bundle,
        )
        self.assertNotIn("%pool_base_addr", bundle)
        self.assertNotIn("input_arg_extract value from %pool_base_addr", bundle)

    def test_pool_frontend_allocation_emitted(self):
        """With frontend_pool_allocation=True, a pool symbol produces a
        %pool_base_addr input_arg parameter and an input_arg_extract
        statement instead of device_mem_allocate."""
        pool_kind = SymbolKind.pool()
        entry = (
            _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": -1}),
            [0],
            [],
            [pool_kind],
        )

        with patch(
            "torch_spyre._inductor.codegen.bundle._spyre_config.frontend_pool_allocation",
            True,
        ):
            bundle = self._run_bundle([entry], pool_size=65536)

        self.assertIn("%pool_base_addr: !sdscbundle.input_arg<index>", bundle)
        self.assertIn(
            "%pool = sdscbundle.input_arg_extract value from"
            " %pool_base_addr : !sdscbundle.input_arg<index> -> index",
            bundle,
        )
        self.assertNotIn("device_mem_allocate", bundle)

    def test_pool_frontend_allocation_param_ordering(self):
        """The pool param is emitted before kernel_arg_sym_indices params,
        matching the pre-PR-#3707 ordering (pool first)."""
        pool_kind = SymbolKind.pool()
        kernel_kind = SymbolKind.kernel(arg_index=0)
        entry = (
            _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": -1}),
            [0, 0],
            [],
            [kernel_kind, pool_kind],
        )

        with patch(
            "torch_spyre._inductor.codegen.bundle._spyre_config.frontend_pool_allocation",
            True,
        ):
            bundle = self._run_bundle([entry], pool_size=65536)

        sig_line = next(
            line for line in bundle.splitlines() if "func.func @sdsc_bundle(" in line
        )
        self.assertLess(
            sig_line.index("%pool_base_addr"),
            sig_line.index("%arg_0_base_addr"),
            "pool_base_addr param must come before arg_0_base_addr",
        )

    def test_pool_absent_when_no_pool_symbols(self):
        """A bundle with no pool symbols emits no device_mem_allocate at all,
        regardless of the pool_size argument passed in."""
        kernel_kind = SymbolKind.kernel(arg_index=0)
        entry = (
            _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": -1}),
            [0],
            [],
            [kernel_kind],
        )

        bundle = self._run_bundle([entry], pool_size=65536)

        self.assertNotIn("device_mem_allocate", bundle)
        self.assertNotIn("%pool", bundle)

    @config.patch({"frontend_pool_allocation": False})
    def test_pool_and_kernel_address_combination(self):
        """A pool symbol and a kernel-address symbol coexist: device_mem_allocate
        appears in the body, %arg_0_base_addr is still the only signature param."""
        pool_kind = SymbolKind.pool()
        kernel_kind = SymbolKind.kernel(arg_index=0)
        entry = (
            _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": -1}),
            [0, 0],
            [],
            [kernel_kind, pool_kind],
        )

        bundle = self._run_bundle([entry], pool_size=4096)

        self.assertIn(
            "func.func @sdsc_bundle(%arg_0_base_addr: !sdscbundle.input_arg<index>)",
            bundle,
        )
        self.assertIn(
            "%pool = sdscbundle.device_mem_allocate 4096 bytes : index",
            bundle,
        )

    def test_pool_size_zero_with_pool_symbol_raises(self):
        """A pool symbol with pool_size=0 must fail loudly rather than emit a
        useless zero-byte device_mem_allocate."""
        pool_kind = SymbolKind.pool()
        entry = (
            _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": -1}),
            [0],
            [],
            [pool_kind],
        )

        with self.assertRaises(AssertionError):
            self._run_bundle([entry], pool_size=0)

    def test_pool_size_above_max_raises(self):
        """A pool symbol with pool_size > MAX_POOL_SIZE_BYTES must fail loudly."""
        pool_kind = SymbolKind.pool()
        entry = (
            _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": -1}),
            [0],
            [],
            [pool_kind],
        )

        with self.assertRaises(AssertionError):
            self._run_bundle([entry], pool_size=MAX_POOL_SIZE_BYTES + 1)

    @config.patch({"frontend_pool_allocation": False})
    def test_pool_size_at_max_succeeds(self):
        """A pool symbol with pool_size == MAX_POOL_SIZE_BYTES (the exact budget
        boundary) must be accepted, not rejected -- MAX_POOL_SIZE_BYTES is itself
        an in-budget value, and Allocator can legitimately return an offset
        whose pool_end lands exactly there."""
        pool_kind = SymbolKind.pool()
        entry = (
            _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": -1}),
            [0],
            [],
            [pool_kind],
        )

        bundle = self._run_bundle([entry], pool_size=MAX_POOL_SIZE_BYTES)
        self.assertIn(
            f"%pool = sdscbundle.device_mem_allocate {MAX_POOL_SIZE_BYTES} bytes : index",
            bundle,
        )
