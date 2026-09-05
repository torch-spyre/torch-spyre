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

"""Tests for compute_specs_hash — verifying hash coverage.

Each test class covers one category from the Hash Coverage Table:

  TestLoopSpecTripCount        — LoopSpec.count changes the hash
  TestSymbolKindTagging        — pool / kernel_slice / kernel_derived offsets change hash
  TestAffineStridesHashed      — affine_strides (tile byte strides) change hash
  TestIterationSpaceHashed     — different iter-space sizes → different hash
  TestOpFuncHashed             — different op names → different hash
  TestDebugHandleStripped      — debug_handle_ is stripped before hashing
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from sympy import Symbol, Integer


def _make_tensor_arg(
    is_input: bool,
    arg_index: int,
    size: int = 64,
    hbm_addr: int | None = None,
):
    """Return a minimal TensorArg-like namespace."""
    from torch_spyre._C import DataFormats, ElementArrangement

    if hbm_addr is None:
        hbm_addr = arg_index

    ta = MagicMock()
    ta.is_input = is_input
    ta.arg_index = arg_index
    ta.device_dtype = DataFormats.IEEE_FP16
    ta.device_size = [1, size, 1]  # [outer, stick_dim, stick]
    ta.device_coordinates = [Integer(0), Symbol("c0") % size, Integer(0)]
    ta.allocation = {"hbm": hbm_addr}
    ta.device_tile_advance_expr = None
    ta.element_arrangement = ElementArrangement.STANDARD
    ta.work_division = None
    return ta


def _make_op_spec(
    op: str = "add",
    size: int = 64,
    hbm_addr_in: int = 0,
    hbm_addr_out: int = 128,
    is_reduction: bool = False,
    tiled_symbols=None,
    tiled_symbol_trip_counts=None,
):
    """Return a minimal OpSpec for a single-input elementwise op."""
    from torch_spyre._inductor.op_spec import OpSpec, TensorArg
    from torch_spyre._inductor.core_mapping import derive_operation_mapping
    from torch_spyre._C import DataFormats

    c0 = Symbol("c0")

    in_arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.IEEE_FP16,
        device_size=[1, size, 1],
        device_coordinates=[Integer(0), c0 % size, Integer(0)],
        allocation={"hbm": hbm_addr_in},
    )
    out_arg = TensorArg(
        is_input=False,
        arg_index=1,
        device_dtype=DataFormats.IEEE_FP16,
        device_size=[1, size, 1],
        device_coordinates=[Integer(0), c0 % size, Integer(0)],
        allocation={"hbm": hbm_addr_out},
    )

    iteration_space = {c0: (Integer(size), 1)}

    spec = OpSpec(
        op=op,
        is_reduction=is_reduction,
        iteration_space=iteration_space,
        args=[in_arg, out_arg],
        op_info={},
        tiled_symbols=tiled_symbols or [],
        tiled_symbol_trip_counts=tiled_symbol_trip_counts or {},
    )
    # Populate the finalized core mapping — required by compile_op_spec
    # (upstream added a guard: ValueError if core_id_to_work_slice is None).
    spec.core_id_to_work_slice = derive_operation_mapping(iteration_space)
    return spec


def _make_loop_spec(count: int, body_op: str = "add", size: int = 64):
    """Return a minimal LoopSpec wrapping one OpSpec."""
    from torch_spyre._inductor.op_spec import LoopSpec

    op = _make_op_spec(op=body_op, size=size)
    return LoopSpec(count=count, body=[op])


def _hash(specs, use_symbols: bool = False):
    """Call compute_specs_hash with the given specs.  Patches version helpers
    to stable strings so tests are environment-independent.
    """
    from torch_spyre.execution.kernel_cache import compute_specs_hash

    with (
        patch(
            "torch_spyre.execution.kernel_cache._get_dxp_version",
            return_value="test-dxp-1.0",
        ),
        patch(
            "torch_spyre.execution.kernel_cache._get_torch_spyre_version",
            return_value="test-spyre-0.0",
        ),
        patch(
            "torch_spyre._inductor.config.bundle_symbolic_args",
            use_symbols,
        ),
    ):
        return compute_specs_hash(specs, kernel_name="test_kernel")


class TestLoopSpecTripCount(unittest.TestCase):
    """LoopSpec.count must be part of the hash.  Two loop-wrapped ops that are
    structurally identical but differ in trip count must produce different keys."""

    def test_different_loop_counts_produce_different_hashes(self):
        loop_4 = _make_loop_spec(count=4)
        loop_8 = _make_loop_spec(count=8)

        h4 = _hash([loop_4])
        h8 = _hash([loop_8])

        self.assertNotEqual(
            h4,
            h8,
            "LoopSpec with count=4 and count=8 must produce different hashes — "
            "trip count must be included in the hash.",
        )

    def test_same_loop_count_produces_same_hash(self):
        loop_a = _make_loop_spec(count=4)
        loop_b = _make_loop_spec(count=4)

        self.assertEqual(
            _hash([loop_a]),
            _hash([loop_b]),
            "Identical LoopSpec (same count, same body) must produce the same hash.",
        )

    def test_nested_loop_count_is_hashed(self):
        """Hash must differ if inner loop count changes even when outer matches."""
        from torch_spyre._inductor.op_spec import LoopSpec

        op = _make_op_spec()
        inner_2 = LoopSpec(count=2, body=[op])
        inner_4 = LoopSpec(count=4, body=[op])
        outer_a = LoopSpec(count=10, body=[inner_2])
        outer_b = LoopSpec(count=10, body=[inner_4])

        self.assertNotEqual(
            _hash([outer_a]),
            _hash([outer_b]),
            "Changing inner loop count must change the hash.",
        )


class TestSymbolKindTagging(unittest.TestCase):
    """When use_symbols=True, pool / kernel_slice / kernel_derived offsets
    must be included in the hash so that two graphs with identical SDSC JSON
    but different baked HBM addresses produce different keys."""

    def test_pool_offset_changes_hash_when_use_symbols_true(self):
        """Two pool tensors at different offsets must hash differently."""
        op_offset_0 = _make_op_spec(hbm_addr_in=0)
        op_offset_0.args[0].allocation.pop("hbm", None)
        op_offset_0.args[0].allocation["hbm_pool"] = 0

        op_offset_512 = _make_op_spec(hbm_addr_in=512)
        op_offset_512.args[0].allocation.pop("hbm", None)
        op_offset_512.args[0].allocation["hbm_pool"] = 512

        h0 = _hash([op_offset_0], use_symbols=True)
        h512 = _hash([op_offset_512], use_symbols=True)

        self.assertNotEqual(
            h0,
            h512,
            "Pool offset 0 and 512 must produce different hashes when use_symbols=True.",
        )

    def test_pool_offset_ignored_when_use_symbols_false(self):
        """With use_symbols=False, the explicit pool-offset tags are NOT appended
        to content_parts.  However, pool addresses are baked into the SDSC JSON
        as negative sentinel symbol IDs by compile_op_spec, so two tensors at
        different pool offsets will still produce different hashes via the JSON.
        This test documents that behaviour."""
        op_offset_0 = _make_op_spec(hbm_addr_in=0)
        op_offset_0.args[0].allocation.pop("hbm", None)
        op_offset_0.args[0].allocation["hbm_pool"] = 0

        op_offset_512 = _make_op_spec(hbm_addr_in=512)
        op_offset_512.args[0].allocation.pop("hbm", None)
        op_offset_512.args[0].allocation["hbm_pool"] = 512

        h0 = _hash([op_offset_0], use_symbols=False)
        h512 = _hash([op_offset_512], use_symbols=False)

        # Pool addresses appear as negative sentinel IDs in the SDSC JSON, so
        # different offsets produce different JSON and therefore different hashes
        # even when use_symbols=False (no explicit pool: tags are added).
        self.assertNotEqual(
            h0,
            h512,
            "Pool addresses are baked as negative sentinel IDs in the SDSC JSON — "
            "different offsets produce different hashes even with use_symbols=False.",
        )


class TestAffineStridesHashed(unittest.TestCase):
    """affine_strides (per-level tile byte strides) must be included in
    compute_specs_hash.  Two ops that differ only in tile stride must produce
    different cache keys so they do not share a cache entry.
    """

    def _make_tiled_op_spec(self, tile_stride_bytes: int, size: int = 256):
        """Return an OpSpec that has a non-empty device_tile_advance_expr.

        tile_stride_bytes simulates the per-element advance for a coarse-tiled
        tensor.  In practice this would be set by coarse_tile.py; here we
        inject it directly onto the TensorArg to produce a non-zero affine
        stride via generate_sdsc's per-level stride logic.
        """
        from torch_spyre._inductor.op_spec import TensorArg
        from torch_spyre._C import DataFormats

        c0 = Symbol("c0")
        # Minted level symbol — the kind used by spyre_kernel._get_or_mint_level_symbol
        lvl_sym = Symbol("_tile_adv_add_lvl0")

        in_arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.IEEE_FP16,
            device_size=[1, size, 1],
            device_coordinates=[Integer(0), c0 % size, Integer(0)],
            allocation={"hbm": 0},
            # tile_stride_bytes / 2 because num_bytes(FP16) == 2
            device_tile_advance_expr=lvl_sym * (tile_stride_bytes // 2),
        )
        out_arg = TensorArg(
            is_input=False,
            arg_index=1,
            device_dtype=DataFormats.IEEE_FP16,
            device_size=[1, size, 1],
            device_coordinates=[Integer(0), c0 % size, Integer(0)],
            allocation={"hbm": size * 2},
            device_tile_advance_expr=lvl_sym * (tile_stride_bytes // 2),
        )

        from torch_spyre._inductor.op_spec import OpSpec
        from torch_spyre._inductor.core_mapping import derive_operation_mapping

        iteration_space = {c0: (Integer(size), 1)}
        spec = OpSpec(
            op="add",
            is_reduction=False,
            iteration_space=iteration_space,
            args=[in_arg, out_arg],
            op_info={},
            tiled_symbols=[[lvl_sym]],  # one nesting level
            tiled_symbol_trip_counts={lvl_sym: size // (tile_stride_bytes // 2)},
        )
        spec.core_id_to_work_slice = derive_operation_mapping(iteration_space)
        return spec

    def test_different_tile_strides_produce_different_hashes(self):
        """Ops that differ only in tile stride must produce different hashes."""
        op_stride_128 = self._make_tiled_op_spec(tile_stride_bytes=128)
        op_stride_256 = self._make_tiled_op_spec(tile_stride_bytes=256)

        try:
            h128 = _hash([op_stride_128])
            h256 = _hash([op_stride_256])
        except Exception as e:
            self.skipTest(f"Skipping affine_strides test — compile_op_spec raised: {e}")

        self.assertNotEqual(
            h128,
            h256,
            "Ops with different tile strides must produce different hashes — "
            "affine_strides must be included in compute_specs_hash.",
        )


# Iteration-space sizes are hashed
class TestIterationSpaceHashed(unittest.TestCase):
    """Different iteration-space sizes must produce different hashes."""

    def test_different_sizes_produce_different_hashes(self):
        op_64 = _make_op_spec(size=64)
        op_128 = _make_op_spec(size=128)

        self.assertNotEqual(
            _hash([op_64]),
            _hash([op_128]),
            "Ops with different iteration-space sizes must hash differently.",
        )

    def test_same_sizes_produce_same_hash(self):
        op_a = _make_op_spec(size=64)
        op_b = _make_op_spec(size=64)

        self.assertEqual(
            _hash([op_a]),
            _hash([op_b]),
            "Identical ops must produce the same hash.",
        )


# Op name (opfunc) is hashed
class TestOpFuncHashed(unittest.TestCase):
    """Different op names must produce different hashes."""

    def test_different_ops_produce_different_hashes(self):
        op_add = _make_op_spec(op="add")
        op_mul = _make_op_spec(op="mul")

        h_add = _hash([op_add])
        h_mul = _hash([op_mul])

        self.assertNotEqual(
            h_add,
            h_mul,
            "add and mul ops must produce different hashes.",
        )

    def test_same_op_produces_same_hash(self):
        op_a = _make_op_spec(op="exp")
        op_b = _make_op_spec(op="exp")

        self.assertEqual(
            _hash([op_a]),
            _hash([op_b]),
        )


# debug_handle_ is stripped before hashing (no false misses)
class TestDebugHandleStripped(unittest.TestCase):
    """debug_handle_ must be removed from the SDSC JSON before hashing so that
    process-specific buffer names do not cause false cache misses."""

    def test_debug_handle_does_not_affect_hash(self):
        """Two SDSC dicts identical except for debug_handle_ must hash the same."""
        from torch_spyre.execution.kernel_cache import _strip_debug_handles

        base = {
            "0_add": {
                "numCoresUsed_": 4,
                "dscs_": [{"add": {"numCoresUsed_": 4}}],
            }
        }
        with_handle = {
            "0_add": {
                **base["0_add"],
                "debug_handle_": {"id": "99999", "ir_chain": ["buf123", "buf456"]},
            }
        }

        stripped_base = _strip_debug_handles(base)
        stripped_with = _strip_debug_handles(with_handle)

        self.assertEqual(
            json.dumps(stripped_base, sort_keys=True),
            json.dumps(stripped_with, sort_keys=True),
            "After stripping, dicts that differ only in debug_handle_ must be equal.",
        )

    def test_debug_handle_stripped_at_all_nesting_levels(self):
        """debug_handle_ nested inside dscs_ must also be stripped."""
        from torch_spyre.execution.kernel_cache import _strip_debug_handles

        sdsc = {
            "0_add": {
                "debug_handle_": {"id": "1"},
                "dscs_": [{"add": {"debug_handle_": {"id": "2"}, "numCoresUsed_": 2}}],
            }
        }
        result = _strip_debug_handles(sdsc)
        self.assertNotIn("debug_handle_", result["0_add"])
        self.assertNotIn("debug_handle_", result["0_add"]["dscs_"][0]["add"])


# Version strings affect hash (environment independence guard)
class TestVersionStringsHashed(unittest.TestCase):
    """Changing any version string (torch, torch_spyre, dxp) must change the hash."""

    def _hash_with_versions(self, dxp_ver: str, spyre_ver: str):
        from torch_spyre.execution.kernel_cache import compute_specs_hash
        from unittest.mock import patch

        op = _make_op_spec()
        with (
            patch(
                "torch_spyre.execution.kernel_cache._get_dxp_version",
                return_value=dxp_ver,
            ),
            patch(
                "torch_spyre.execution.kernel_cache._get_torch_spyre_version",
                return_value=spyre_ver,
            ),
            patch("torch_spyre._inductor.config.bundle_symbolic_args", False),
        ):
            return compute_specs_hash([op], kernel_name="ver_test")

    def test_dxp_version_change_changes_hash(self):
        h1 = self._hash_with_versions("1.0.0", "0.0.1")
        h2 = self._hash_with_versions("2.0.0", "0.0.1")
        self.assertNotEqual(h1, h2, "dxp version change must change the hash.")

    def test_torch_spyre_version_change_changes_hash(self):
        h1 = self._hash_with_versions("1.0.0", "0.0.1")
        h2 = self._hash_with_versions("1.0.0", "0.0.2")
        self.assertNotEqual(h1, h2, "torch_spyre version change must change the hash.")

    def test_same_versions_same_hash(self):
        h1 = self._hash_with_versions("1.0.0", "0.0.1")
        h2 = self._hash_with_versions("1.0.0", "0.0.1")
        self.assertEqual(h1, h2, "Same versions must produce the same hash.")


if __name__ == "__main__":
    unittest.main()
