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

"""Unit tests for bundle.py dimension-symbol support (mark_dynamic path).

compile_op_spec is mocked throughout so no Spyre hardware is required.
"""

import os
import tempfile
from unittest.mock import patch

from torch._inductor.test_case import TestCase as InductorTestCase

from torch_spyre._inductor.codegen.bundle import (
    _extract_symbol_ids,
    generate_bundle,
)
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.op_spec import OpSpec


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _make_sdsc_json(
    sdsc_idx: int = 0,
    *,
    dim_sym_ids: dict[str, list[int]] | None = None,
    hbm_sym_ids_per_core: dict[str, int] | None = None,
    num_cores: int = 1,
) -> dict:
    """Minimal SDSC JSON for testing _extract_symbol_ids and generate_bundle.

    Args:
        sdsc_idx: Used to form the top-level JSON key, e.g. "0_fused_test".
        dim_sym_ids: Mapping of dim label → list of symbol IDs for
            dimToSymbolMapping_, e.g. {"mb": [-1]}.
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
                        "dimToSymbolMapping_": dim_sym_ids or {},
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

    def test_dim_only(self):
        """Dimension ID from dimToSymbolMapping_ is collected."""
        sdsc_json = _make_sdsc_json(dim_sym_ids={"mb": [-1]})
        self.assertEqual(_extract_symbol_ids(sdsc_json), [-1])

    def test_hbm_only(self):
        """HBM address ID from scheduleTree_ is collected."""
        sdsc_json = _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": -2})
        self.assertEqual(_extract_symbol_ids(sdsc_json), [-2])

    def test_dim_before_hbm_ordering(self):
        """Dimension IDs come before HBM address IDs."""
        sdsc_json = _make_sdsc_json(
            dim_sym_ids={"mb": [-1]},
            hbm_sym_ids_per_core={"[0, 0, 0]": -2},
        )
        self.assertEqual(_extract_symbol_ids(sdsc_json), [-1, -2])

    def test_positive_values_excluded(self):
        """Positive (concrete) addresses are ignored."""
        sdsc_json = _make_sdsc_json(hbm_sym_ids_per_core={"[0, 0, 0]": 0x400000000})
        self.assertEqual(_extract_symbol_ids(sdsc_json), [])

    def test_dim_deduplication_across_tensors(self):
        """Two tensors sharing the same mark_dynamic symbol yield one ID, not two."""
        sdsc_json = {
            "0_fused_test": {
                "numCoresUsed_": 1,
                "dscs_": [
                    {
                        "op": {
                            "dimToSymbolMapping_": {"dim_0": [-1]},
                            "scheduleTree_": [],
                        }
                    },
                    {
                        "op": {
                            "dimToSymbolMapping_": {"dim_0": [-1]},
                            "scheduleTree_": [],
                        }
                    },
                ],
            }
        }
        self.assertEqual(_extract_symbol_ids(sdsc_json), [-1])


class TestGenerateBundleDimensionSymbols(InductorTestCase):
    """Integration tests for generate_bundle with mark_dynamic dimension symbols."""

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

    def _run_bundle(self, compiled_entries, op_specs=None, use_symbols=False):
        """Run generate_bundle with mocked compile_op_spec, return bundle.mlir text."""
        if op_specs is None:
            op_specs = [_minimal_op_spec() for _ in compiled_entries]

        side_effects = list(compiled_entries)
        with patch(
            "torch_spyre._inductor.codegen.bundle.compile_op_spec",
            side_effect=side_effects,
        ):
            generate_bundle(
                "test",
                self.output_dir,
                op_specs,
                use_symbols=use_symbols,
            )
        return self._read_bundle()

    def test_single_dim_sym_function_signature(self):
        """Dimension symbol produces an input_arg param even when use_symbols=False."""
        dim_kind = SymbolKind.dimension(granularity=56, max_value=616, pytorch_sym="s0")
        entry = (_make_sdsc_json(dim_sym_ids={"mb": [-1]}), [0], [], [dim_kind])

        bundle = self._run_bundle([entry])

        self.assertIn(
            "%sym_0_1_base: !sdscbundle.input_arg<index, granularity=56, max_value=616>",
            bundle,
        )

    def test_single_dim_sym_extract_op(self):
        """input_arg_extract unpacks %sym_0_1_base into plain index %sym_0_1."""
        dim_kind = SymbolKind.dimension(granularity=56, max_value=616, pytorch_sym="s0")
        entry = (_make_sdsc_json(dim_sym_ids={"mb": [-1]}), [0], [], [dim_kind])

        bundle = self._run_bundle([entry])

        self.assertIn(
            "%sym_0_1 = sdscbundle.input_arg_extract value from %sym_0_1_base"
            " : !sdscbundle.input_arg<index, granularity=56, max_value=616> -> index",
            bundle,
        )

    def test_single_dim_sym_sdsc_execute_operand(self):
        """sdsc_execute passes %sym_0_1 as operand with symbol_id=-1."""
        dim_kind = SymbolKind.dimension(granularity=56, max_value=616, pytorch_sym="s0")
        entry = (_make_sdsc_json(dim_sym_ids={"mb": [-1]}), [0], [], [dim_kind])

        bundle = self._run_bundle([entry])

        self.assertIn("sdscbundle.sdsc_execute (%sym_0_1)", bundle)
        self.assertIn('"symbol_ids"=[-1]', bundle)

    def test_duplicate_pytorch_sym_single_param(self):
        """Two SDSCs with the same pytorch_sym share one param; both resolve to %sym_0_1."""
        dim_kind_0 = SymbolKind.dimension(
            granularity=56, max_value=616, pytorch_sym="s0"
        )
        dim_kind_1 = SymbolKind.dimension(
            granularity=56, max_value=616, pytorch_sym="s0"
        )
        entry_0 = (
            _make_sdsc_json(sdsc_idx=0, dim_sym_ids={"mb": [-1]}),
            [0],
            [],
            [dim_kind_0],
        )
        entry_1 = (
            _make_sdsc_json(sdsc_idx=1, dim_sym_ids={"mb": [-2]}),
            [0],
            [],
            [dim_kind_1],
        )

        bundle = self._run_bundle([entry_0, entry_1])

        # One param + one extract op (2 occurrences of the type string).
        self.assertEqual(
            bundle.count("!sdscbundle.input_arg<index, granularity=56, max_value=616>"),
            2,
        )
        self.assertEqual(bundle.count("sdscbundle.sdsc_execute (%sym_0_1)"), 2)

    def test_two_distinct_pytorch_syms_two_params(self):
        """Two distinct pytorch_syms produce two independent params and extract ops."""
        dim_s0 = SymbolKind.dimension(granularity=56, max_value=616, pytorch_sym="s0")
        dim_s1 = SymbolKind.dimension(granularity=32, max_value=256, pytorch_sym="s1")
        entry = (
            _make_sdsc_json(dim_sym_ids={"mb": [-1], "nb": [-2]}),
            [0, 0],
            [],
            [dim_s0, dim_s1],
        )

        bundle = self._run_bundle([entry])

        self.assertIn(
            "%sym_0_1_base: !sdscbundle.input_arg<index, granularity=56, max_value=616>",
            bundle,
        )
        self.assertIn(
            "%sym_0_2_base: !sdscbundle.input_arg<index, granularity=32, max_value=256>",
            bundle,
        )
        self.assertIn("%sym_0_1 = sdscbundle.input_arg_extract", bundle)
        self.assertIn("%sym_0_2 = sdscbundle.input_arg_extract", bundle)
        self.assertIn("sdscbundle.sdsc_execute (%sym_0_1, %sym_0_2)", bundle)

    def test_dimension_and_kernel_address_combination(self):
        """A dimension symbol and a kernel-address symbol coexist in one bundle.

        Both kinds must appear side by side, in the correct param/operand order,
        when use_symbols=True and a dimension symbol is also present.
        """
        dim_kind = SymbolKind.dimension(granularity=56, max_value=616, pytorch_sym="s0")
        kernel_kind = SymbolKind.kernel(arg_index=0)
        entry = (
            _make_sdsc_json(
                dim_sym_ids={"mb": [-1]},
                hbm_sym_ids_per_core={"[0, 0, 0]": -2},
            ),
            [0, 0],
            [],
            [dim_kind, kernel_kind],
        )

        bundle = self._run_bundle([entry], use_symbols=True)

        # Kernel-address param comes before the dimension param in the signature.
        self.assertIn(
            "func.func @sdsc_bundle("
            "%arg_0_base_addr: !sdscbundle.input_arg<index>, "
            "%sym_0_1_base: !sdscbundle.input_arg<index, granularity=56,"
            " max_value=616>)",
            bundle,
        )
        self.assertIn(
            "%arg_0 = sdscbundle.input_arg_extract value from"
            " %arg_0_base_addr : !sdscbundle.input_arg<index> -> index",
            bundle,
        )
        self.assertIn(
            "%sym_0_1 = sdscbundle.input_arg_extract value from"
            " %sym_0_1_base : !sdscbundle.input_arg<index, granularity=56,"
            " max_value=616> -> index",
            bundle,
        )
        # Dimension IDs precede HBM/kernel-address IDs (both in the operand
        # list and in the symbol_ids attribute).
        self.assertIn("sdscbundle.sdsc_execute (%sym_0_1, %arg_0)", bundle)
        self.assertIn('"symbol_ids"=[-1, -2]', bundle)
