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

"""Consolidated gather-style indirect-access tests (one file per op family).

Each scenario compiles once and makes all relevant assertions in one place
(classification, op-spec structure, detection, SDSC fields), instead of
spreading the same kernel across detection/op_spec/sdsc stage files.
Slice by stage with the optional check(...) parameters.

Every gather scenario now carries through to SDSC generation: check() (and the
capture-based tests via bundle_jsons_from_captured) validate the indirect-access
encoding of the produced bundle, not just that op specs were generated.

Every gather scenario is generated once per SENCORES value (1, 2, 4, 8, 16, 32)
by register_multicore_variants, so the same checks also exercise the multicore
work-division planner. See MULTICORE_SENCORES in indirect_access_common.

There is one test per scenario -- no separate e2e variants. Each scenario
validates the capture path and then runs the kernel on the real backend,
validating the result and warning that the device values diverge from the CPU
reference (the backend does not yet implement indirect gather). Scenarios route
their compile through _stage_and_e2e (stage check + e2e run); capture-based
tests call run_e2e directly after their own assertions. The three structural
tests (sdsc_fields, sdsc_handoff, python_bundle_generation) stay capture-only.

The scenario body lives in _GatherScenarios and moves its value/source tensor
via self.to_spyre (plain `.to("spyre")` layout, no explicit layout pin). The
relayout pass must rotate the indexed dim outermost for multi-stick sources,
proving the pass works independently.

"""

import os
import sys
from unittest.mock import patch

import pytest
import torch
from torch._inductor.utils import run_and_get_code

sys.path.insert(0, os.path.dirname(__file__))
from indirect_access_common import (  # noqa: E402
    DIRECT_OP_SPEC,
    GATHER_OP_SPEC,
    register_multicore_variants,
    bundle_jsons_from_captured,
    capture_op_specs,
    capture_sdsc_calls,
    flatten_op_specs,
    generate_sdsc_jsons,
    indirect_access_target_names,
    iter_schedule_tree_nodes,
    iter_sdsc_op_bodies,
    op_spec_has_indirect_access,
    op_spec_has_indirect_input,
    op_spec_has_indirect_output,
    run_e2e,
    plain_to_spyre,
)

from torch_spyre._C import (  # noqa: E402
    DataFormats,
    SpyreTensorLayout,
    get_device_dtype,
    get_elem_in_stick,
)
from torch_spyre._inductor.constants import IDENTITY_OP, RESTICKIFY_OP  # noqa: E402
from torch_spyre._inductor.op_spec import find_unimplemented  # noqa: E402


class _GatherScenarios:
    """torch gather-family ops: one compile + all-stage checks per scenario.

    A plain mixin (not a TestCase, so it is not collected on its own). The
    concrete, collectable classes ``TestGather_cores{1,2,4,8,16,32}`` are
    generated at the bottom of the module by ``register_multicore_variants``,
    each pinned to its SENCORES value via ``@config.patch``.
    """

    to_spyre = staticmethod(plain_to_spyre)

    def _xi(self, P=32, two_d=False, dtype=torch.int32, M=128, N=256, Q=192):
        """Named (x[M,N], idx) gather operands. two_d means idx[P,Q]."""
        x = self.to_spyre(torch.rand(M, N, dtype=torch.float16))
        shape = (P, Q) if two_d else (P,)
        i = torch.randint(0, M, shape, dtype=dtype).to("spyre")
        self.name_dims(x, {"M": M, "N": N})
        self.name_dims(i, {"P": P, "Q": Q} if two_d else {"P": P})
        return x, i

    def _xi3d(self, P=32, dtype=torch.int32, A=64, B=8, C=64):
        """Named (x[A,B,C], idx[P]) gather operands: 3-D value, 1-D index."""
        x = self.to_spyre(torch.rand(A, B, C, dtype=torch.float16))
        i = torch.randint(0, A, (P,), dtype=dtype).to("spyre")
        self.name_dims(x, {"A": A, "B": B, "C": C})
        self.name_dims(i, {"P": P})
        return x, i

    def _xi4d(self, P=32, dtype=torch.int32, A=8, B=4, C=8, D=64):
        """Named (x[A,B,C,D], idx[P]) gather operands: 4-D value, 1-D index."""
        x = self.to_spyre(torch.rand(A, B, C, D, dtype=torch.float16))
        i = torch.randint(0, A, (P,), dtype=dtype).to("spyre")
        self.name_dims(x, {"A": A, "B": B, "C": C, "D": D})
        self.name_dims(i, {"P": P})
        return x, i

    # -- core x[i].exp(): op-spec structure + detection (one compile) ------
    def test_gather_with_exp(self):
        """x[i].exp(): GATHER op spec with IndirectAccess on an input, a named
        index arg matching the IndirectAccess target, one output, a well-formed
        iteration space and TensorArg metadata, and the index is detected."""
        x, i = self._xi(P=3, two_d=True)
        # Stage check first (and all the detailed op-spec assertions below) so
        # they always run; the e2e at the end xfails on the expected
        # device-side divergence/abort, which would otherwise stop the test.
        r = self.check(
            lambda x, i: x[i].exp(),
            x,
            i,
            expect=GATHER_OP_SPEC,
            op="exp",
            detected=True,
        )
        op_specs = r.op_specs
        self.assertTrue(any(op_spec_has_indirect_input(s) for s in op_specs))
        self.assertFalse(any(op_spec_has_indirect_output(s) for s in op_specs))

        named = [
            a for s in op_specs for a in s.args if a.is_input and a.name is not None
        ]
        self.assertTrue(named, "no named index arg found")
        targets = indirect_access_target_names(op_specs)
        self.assertTrue(any(a.name in targets for a in named))
        for a in named:
            self.assertIsInstance(a.name, str)
            self.assertTrue(a.name)
        self.assertTrue(
            any(a.is_input and a.name is None for s in op_specs for a in s.args),
            "expected a value (non-index) input arg",
        )
        for s in op_specs:
            self.assertEqual(len([a for a in s.args if not a.is_input]), 1)
            self.assertGreaterEqual(len([a for a in s.args if a.is_input]), 1)
            self.assertTrue(s.iteration_space)
            for _rng, work in s.iteration_space.values():
                self.assertGreaterEqual(int(work), 1)
            for a in s.args:
                self.assertIsInstance(a.device_dtype, DataFormats)
                self.assertTrue(a.device_size, "device_size should be non-empty")
                self.assertEqual(len(a.device_coordinates), len(a.device_size))
        run_e2e(self, lambda x, i: x[i].exp(), x, i)

    def test_gather_bare_index(self):
        """x[i] (no unary) produces an identity/restickify copy op.
        Arg ordering is: index first, value, then output."""
        x, i = self._xi(P=3, two_d=True)
        with capture_op_specs() as captured:
            torch.compile(lambda x, i: x[i])(x, i)
        op_specs = self.assert_reaches_op_spec(captured)
        gather_specs = [
            s
            for s in op_specs
            if s.op in (IDENTITY_OP, RESTICKIFY_OP) and op_spec_has_indirect_input(s)
        ]
        self.assertTrue(
            gather_specs,
            f"expected an indirect copy op ({IDENTITY_OP}/{RESTICKIFY_OP}); "
            f"got ops={[s.op for s in op_specs]}",
        )
        spec = gather_specs[0]
        self.assertIsNotNone(spec.args[0].name, "first arg should be the named index")
        self.assertTrue(spec.args[0].is_input, "index arg should be an input")
        self.assertFalse(spec.args[-1].is_input, "last arg should be the output")
        self.assertEqual(len([a for a in spec.args if not a.is_input]), 1)
        self.assert_indirect_source_indexed_dim_outermost(op_specs)
        # Carry the same scenario through to SDSC and validate the indirect
        # encoding of the copy op (not just that op specs were produced).
        self.assert_indirect_sdsc_fields(bundle_jsons_from_captured(captured), "gather")
        run_e2e(self, lambda x, i: x[i], x, i)

    def test_gather_supported_unaries(self):
        """A gather fused with each supported unary keeps the gather signature.
        The op name may differ if a unary decomposes, so we check the signature.
        """
        M, N, P = 128, 256, 32
        unaries = {
            "abs": torch.abs,
            "neg": torch.neg,
            "exp": torch.exp,
            "tanh": torch.tanh,
            "sqrt": torch.sqrt,
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
        }
        # Validate every unary's stage encoding first (all subtests run), then
        # run one representative end-to-end. They share the same gather
        # structure, and run_e2e xfails on the expected device-side divergence,
        # which would otherwise stop the loop before later unaries are checked.
        for label, fn in unaries.items():
            with self.subTest(unary=label):
                torch._dynamo.reset()
                x = self.to_spyre(torch.rand(M, N, dtype=torch.float16))
                idx = torch.randint(0, M, (P,), dtype=torch.int32).to("spyre")
                self.name_dims(x, {"M": M, "N": N})
                self.name_dims(idx, {"P": P})
                with capture_op_specs() as captured:
                    torch.compile(lambda x, i, _fn=fn: _fn(x[i]))(x, idx)
                op_specs = self.assert_reaches_op_spec(captured)
                self.assertTrue(
                    any(op_spec_has_indirect_input(s) for s in op_specs),
                    f"{label}: gather signature lost",
                )
                self.assert_indirect_source_indexed_dim_outermost(op_specs)
                # Each supported unary must also lower to a well-formed
                # indirect-access SDSC bundle, not just an op spec.
                self.assert_indirect_sdsc_fields(
                    bundle_jsons_from_captured(captured), "gather"
                )
        torch._dynamo.reset()
        x = self.to_spyre(torch.rand(M, N, dtype=torch.float16))
        idx = torch.randint(0, M, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(x, {"M": M, "N": N})
        self.name_dims(idx, {"P": P})
        run_e2e(self, lambda x, i: torch.exp(x[i]), x, idx)

    def test_gather_chained_unaries(self):
        """x[i].exp().tanh(): both unaries stay on Spyre, gather keeps its indirect access."""
        x, i = self._xi(P=32)
        with capture_op_specs() as captured:
            torch.compile(lambda x, i: x[i].exp().tanh())(x, i)
        op_specs = self.assert_reaches_op_spec(captured)
        ops = [s.op for s in op_specs]
        self.assertIn("exp", ops)
        self.assertIn("tanh", ops)
        self.assertTrue(any(op_spec_has_indirect_input(s) for s in op_specs))
        self.assert_indirect_source_indexed_dim_outermost(op_specs)
        # The fused chain must still emit a valid indirect-access SDSC bundle.
        self.assert_indirect_sdsc_fields(bundle_jsons_from_captured(captured), "gather")
        run_e2e(self, lambda x, i: x[i].exp().tanh(), x, i)

    # -- classification of torch gather ops -------------------------------
    def test_advanced_indexing(self):
        x, i = self._xi(P=32)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_advanced_indexing_int64_index(self):
        x, i = self._xi(P=32, dtype=torch.int64)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_advanced_indexing_single_row(self):
        x, i = self._xi(P=1)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_embedding_lookup(self):
        """LLM embedding lookup: table[token_ids] with vocab-scale shapes and int64 ids."""
        V, D, B, S = 32000, 512, 2, 128
        table = self.to_spyre(torch.rand(V, D, dtype=torch.float16))
        token_ids = torch.randint(0, V, (B, S), dtype=torch.int64).to("spyre")
        self.name_dims(table, {"V": V, "D": D})
        self.name_dims(token_ids, {"B": B, "S": S})
        self._stage_and_e2e(lambda t, i: t[i], table, token_ids, expect=GATHER_OP_SPEC)

    def test_embedding_lookup_1d_index(self):
        """1-D-index counterpart of test_embedding_lookup (which uses a 2-D
        index): table[ids] with vocab-scale shapes and 1-D int64 ids."""
        V, D, P = 32000, 512, 32
        table = self.to_spyre(torch.rand(V, D, dtype=torch.float16))
        ids = torch.randint(0, V, (P,), dtype=torch.int64).to("spyre")
        self.name_dims(table, {"V": V, "D": D})
        self.name_dims(ids, {"P": P})
        self._stage_and_e2e(lambda t, i: t[i], table, ids, expect=GATHER_OP_SPEC)

    def test_embedding_2d_index_stick_size_1(self):
        V, E, B = 4096, 1024, 2
        weight = self.to_spyre(torch.rand(V, E, dtype=torch.float16))
        idx = torch.randint(0, V, (B, 1), dtype=torch.int32).to("spyre")
        self.name_dims(weight, {"V": V, "E": E})
        self.name_dims(idx, {"B": B, "one": 1})
        self._stage_and_e2e(
            lambda w, i: torch.embedding(w, i), weight, idx, expect=GATHER_OP_SPEC
        )

    def test_embedding_2d_index_dim_size_1(self):
        V, E, B = 4096, 1024, 1
        weight = self.to_spyre(torch.rand(V, E, dtype=torch.float16))
        idx = torch.randint(0, V, (B, 1), dtype=torch.int32).to("spyre")
        self.name_dims(weight, {"V": V, "E": E})
        self.name_dims(idx, {"B": B, "one": 1})
        self._stage_and_e2e(
            lambda w, i: torch.embedding(w, i), weight, idx, expect=GATHER_OP_SPEC
        )

    def test_advanced_indexing_with_tanh(self):
        x, i = self._xi(P=32)
        self._stage_and_e2e(
            lambda x, i: x[i].tanh(), x, i, expect=GATHER_OP_SPEC, op="tanh"
        )

    def test_advanced_indexing_with_abs(self):
        x, i = self._xi(P=32)
        self._stage_and_e2e(
            lambda x, i: x[i].abs(), x, i, expect=GATHER_OP_SPEC, op="abs"
        )

    def test_advanced_indexing_with_exp(self):
        """1-D-index counterpart of test_gather_with_exp (which uses a 2-D index)."""
        x, i = self._xi(P=32)
        self._stage_and_e2e(
            lambda x, i: x[i].exp(), x, i, expect=GATHER_OP_SPEC, op="exp"
        )

    def test_index_select(self):
        x, i = self._xi(P=32)  # source [128, 256] -> 4 sticks per row
        self._stage_and_e2e(
            lambda x, i: torch.index_select(x, 0, i), x, i, expect=GATHER_OP_SPEC
        )

    def test_index_select_with_exp(self):
        x, i = self._xi(P=32)
        self._stage_and_e2e(
            lambda x, i: torch.index_select(x, 0, i).exp(),
            x,
            i,
            expect=GATHER_OP_SPEC,
            op="exp",
        )

    def test_index_select_larger_index(self):
        x, i = self._xi(P=96)
        self._stage_and_e2e(
            lambda x, i: torch.index_select(x, 0, i), x, i, expect=GATHER_OP_SPEC
        )

    @pytest.mark.skip(
        reason=(
            "dxp_standalone SIGABRT: std::fmod(dsDim, size) == 0 in "
            "GatherIndexConversion.cpp — torch.gather with a 2-D index "
            "tensor is not yet supported by the backend"
        )
    )
    def test_torch_gather(self):
        """torch.gather(x, 0, index) with a same-rank [M,K] index tensor.

        Indices address dim 0, so they must be < M (not < N) -- otherwise the
        CPU reference itself raises out-of-bounds once the kernel actually runs.
        """
        M, N, K = 128, 256, 64
        x = self.to_spyre(torch.rand(M, N, dtype=torch.float16))
        index = torch.randint(0, M, (M, K), dtype=torch.int64).to("spyre")
        self.name_dims(x, {"M": M, "N": N})
        self.name_dims(index, {"M": M, "K": K})
        self._stage_and_e2e(
            lambda x, index: torch.gather(x, 0, index), x, index, expect=GATHER_OP_SPEC
        )

    def test_embedding(self):
        """torch.embedding(weight, idx) currently falls back to CPU."""
        V, E, P = 256, 128, 32
        weight = self.to_spyre(torch.rand(V, E, dtype=torch.float16))
        idx = torch.randint(0, V, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(weight, {"V": V, "E": E})
        self.name_dims(idx, {"P": P})
        self._stage_and_e2e(
            lambda w, i: torch.embedding(w, i), weight, idx, expect=GATHER_OP_SPEC
        )

    # -- additional real gather scenarios ---------------------------------
    def test_gather_3d_data(self):
        """x[i] where x is 3-D [A,B,C] -- gather rows of higher-rank data."""
        A, B, C, P = 64, 8, 64, 16
        x = self.to_spyre(torch.rand(A, B, C, dtype=torch.float16))
        idx = torch.randint(0, A, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(x, {"A": A, "B": B, "C": C})
        self.name_dims(idx, {"P": P})
        self._stage_and_e2e(lambda x, i: x[i], x, idx, expect=GATHER_OP_SPEC)

    # -- 3-D value tensor + 1-D index: mirror the 2-D scenarios above ------
    def test_advanced_indexing_3d_int64_index(self):
        x, i = self._xi3d(P=32, dtype=torch.int64)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_advanced_indexing_3d_single_row(self):
        x, i = self._xi3d(P=1)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_advanced_indexing_3d_with_tanh(self):
        x, i = self._xi3d(P=32)
        self._stage_and_e2e(
            lambda x, i: x[i].tanh(), x, i, expect=GATHER_OP_SPEC, op="tanh"
        )

    def test_advanced_indexing_3d_with_abs(self):
        x, i = self._xi3d(P=32)
        self._stage_and_e2e(
            lambda x, i: x[i].abs(), x, i, expect=GATHER_OP_SPEC, op="abs"
        )

    def test_gather_3d_with_exp(self):
        x, i = self._xi3d(P=32)
        self._stage_and_e2e(
            lambda x, i: x[i].exp(), x, i, expect=GATHER_OP_SPEC, op="exp"
        )

    def test_index_select_3d(self):
        x, i = self._xi3d(P=32)  # source [64, 8, 64] -> row B*C = 512 = 8 sticks
        self._stage_and_e2e(
            lambda x, i: torch.index_select(x, 0, i), x, i, expect=GATHER_OP_SPEC
        )

    def test_index_select_3d_with_exp(self):
        x, i = self._xi3d(P=32)
        self._stage_and_e2e(
            lambda x, i: torch.index_select(x, 0, i).exp(),
            x,
            i,
            expect=GATHER_OP_SPEC,
            op="exp",
        )

    def test_index_select_3d_larger_index(self):
        x, i = self._xi3d(P=96)
        self._stage_and_e2e(
            lambda x, i: torch.index_select(x, 0, i), x, i, expect=GATHER_OP_SPEC
        )

    def test_gather_3d_then_scalar_mul(self):
        x, i = self._xi3d(P=32)
        self._stage_and_e2e(lambda x, i: x[i] * 2.0, x, i, expect=GATHER_OP_SPEC)

    def test_gather_3d_then_scalar_add(self):
        x, i = self._xi3d(P=32)
        self._stage_and_e2e(lambda x, i: x[i] + 1.0, x, i, expect=GATHER_OP_SPEC)

    def test_gather_3d_chained_unaries(self):
        x, i = self._xi3d(P=32)
        self._stage_and_e2e(
            lambda x, i: x[i].exp().tanh(), x, i, expect=GATHER_OP_SPEC, op="tanh"
        )

    def test_gather_3d_then_sum_reduction(self):
        """x[i].sum(dim=1) -- reduction over a gathered 3-D tensor's B dim."""
        x, i = self._xi3d(P=32)
        self._stage_and_e2e(lambda x, i: x[i].sum(dim=1), x, i, expect=GATHER_OP_SPEC)

    def test_gather_3d_two_indices_added(self):
        """x[i] + x[j] -- two independent 3-D gathers feeding a binary op."""
        A = 64
        x, i = self._xi3d(P=32, A=A)
        j = torch.randint(0, A, (32,), dtype=torch.int32).to("spyre")
        self.name_dims(j, {"P": 32})
        self._stage_and_e2e(lambda x, i, j: x[i] + x[j], x, i, j, expect=GATHER_OP_SPEC)

    def test_paged_kv(self):
        """Paged KV-cache gather: keys[slot_idxs] with 3D cache and 2D int64 slot index."""
        cache, H, Dh, B, Lk = 32768, 8, 128, 2, 256
        keys = self.to_spyre(torch.rand(cache, H, Dh, dtype=torch.float16))
        slot_idxs = torch.randint(0, cache, (B, Lk), dtype=torch.int64).to("spyre")
        self.name_dims(keys, {"cache": cache, "H": H, "Dh": Dh})
        self.name_dims(slot_idxs, {"B": B, "Lk": Lk})
        # Source row is H*Dh = 1024 elems = 16 sticks, so the indexed dim must be
        # relaid out to outermost or the gather reads only the first stick --
        # check() asserts that (assert_indirect_source_indexed_dim_outermost).
        self._stage_and_e2e(lambda k, i: k[i], keys, slot_idxs, expect=GATHER_OP_SPEC)

    def test_gather_multistick_source_indexed_dim_outermost(self):
        """Gather whose source rows span several sticks: the source relayout must
        put the indexed dim outermost in the OpSpec device coordinates.

        cols=256 is 4 sticks per row, so this exercises the multi-stick striding
        that a 1-stick source cannot. Without the relayout the indexed dim sits
        behind the stick-count dim and the gather reads only the first stick of
        each row. check() asserts the indexed dim landed outermost.
        """
        x = self.to_spyre(torch.rand(20, 256, dtype=torch.float16))
        i = torch.tensor([0, 12, 3, 15, 7], dtype=torch.int32).to("spyre")
        self.name_dims(x, {"rows": 20, "cols": 256})
        self.name_dims(i, {"P": 5})
        self.check(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_gather_singlestick_source_indexed_dim_outermost(self):
        """Control for the multi-stick case: a source whose row is exactly one
        stick (cols=64). The indexed dim is already outermost, so the relayout is
        a no-op -- this asserts the pass does not disturb the common single-stick
        gather (the shape test_gather_1d covers) while still landing the indexed
        dim outermost (check() asserts it).
        """
        x = self.to_spyre(torch.rand(20, 64, dtype=torch.float16))
        i = torch.tensor([0, 12, 3, 15, 7], dtype=torch.int32).to("spyre")
        self.name_dims(x, {"rows": 20, "cols": 64})
        self.name_dims(i, {"P": 5})
        self.check(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_gather_after_reshape(self):
        """x.reshape(12, 256)[i] -- gather on a reshaped tensor."""
        x = self.to_spyre(torch.rand(3, 1024, dtype=torch.float16))
        i = torch.tensor((1, 2), dtype=torch.int32).to("spyre")
        self.name_dims(x, {"rows": 3, "cols": 1024})
        self.name_dims(i, {"P": 2})
        self._stage_and_e2e(
            lambda x, i: x.reshape(12, 256)[i], x, i, expect=GATHER_OP_SPEC
        )

    def test_gather_after_reshape_1d(self):
        """t.reshape(16, 256)[idx] -- gather on a 1D tensor reshaped to 2D (page table)."""
        t = self.to_spyre(torch.arange(4096, dtype=torch.float16))
        idx = torch.tensor((7, 3), dtype=torch.int32).to("spyre")
        self.name_dims(t, {"N": 4096})
        self.name_dims(idx, {"P": 2})
        self._stage_and_e2e(
            lambda t, i: t.reshape(16, 256)[i], t, idx, expect=GATHER_OP_SPEC
        )

    def test_gather_then_scalar_mul(self):
        """x[i] * 2.0 -- gather fused with a scalar binary op."""
        x, i = self._xi(P=32)
        self._stage_and_e2e(lambda x, i: x[i] * 2.0, x, i, expect=GATHER_OP_SPEC)

    def test_gather_then_scalar_add(self):
        """x[i] + 1.0 -- gather fused with a scalar add."""
        x, i = self._xi(P=32)
        self._stage_and_e2e(lambda x, i: x[i] + 1.0, x, i, expect=GATHER_OP_SPEC)

    def test_gather_two_indices_added(self):
        """x[i] + x[j] -- two independent gathers feeding a binary op."""
        M, N, P = 128, 256, 32
        x = self.to_spyre(torch.rand(M, N, dtype=torch.float16))
        i = torch.randint(0, M, (P,), dtype=torch.int32).to("spyre")
        j = torch.randint(0, M, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(x, {"M": M, "N": N})
        self.name_dims(i, {"P": P})
        self.name_dims(j, {"P": P})
        self._stage_and_e2e(lambda x, i, j: x[i] + x[j], x, i, j, expect=GATHER_OP_SPEC)

    def test_gather_then_sum_reduction(self):
        """x[i].sum(dim=1) -- a reduction over gathered rows."""
        x, i = self._xi(P=32)
        self._stage_and_e2e(lambda x, i: x[i].sum(dim=1), x, i, expect=GATHER_OP_SPEC)

    def test_gather_after_producer_exp_2d(self):
        """x.exp()[i] -- 2D gather with producer layout rewrite."""
        M, N, P = 128, 256, 32
        x = self.to_spyre(torch.rand(M, N, dtype=torch.float16))
        i = torch.randint(0, M, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(x, {"M": M, "N": N})
        self.name_dims(i, {"P": P})
        self._stage_and_e2e(lambda x, i: x.exp()[i], x, i, expect=GATHER_OP_SPEC)

    def test_gather_after_producer_exp_3d(self):
        """x.exp()[i] -- 3D gather with producer layout rewrite."""
        M, N, K, P = 64, 128, 64, 16
        x = self.to_spyre(torch.rand(M, N, K, dtype=torch.float16))
        i = torch.randint(0, M, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(x, {"M": M, "N": N, "K": K})
        self.name_dims(i, {"P": P})
        self._stage_and_e2e(lambda x, i: x.exp()[i], x, i, expect=GATHER_OP_SPEC)

    def test_gather_after_producer_exp_4d(self):
        """x.exp()[i] -- 4D gather with producer layout rewrite."""
        M, N, K, L, P = 128, 4, 4, 64, 8
        x = self.to_spyre(torch.rand(M, N, K, L, dtype=torch.float16))
        i = torch.randint(0, M, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(x, {"M": M, "N": N, "K": K, "L": L})
        self.name_dims(i, {"P": P})
        self._stage_and_e2e(lambda x, i: x.exp()[i], x, i, expect=GATHER_OP_SPEC)

    def test_gather_after_producer_exp_2d_index(self):
        """x.exp()[i] -- 2D value tensor with 2D index tensor."""
        M, N, P, Q = 128, 256, 16, 16
        x = self.to_spyre(torch.rand(M, N, dtype=torch.float16))
        i = torch.randint(0, M, (P, Q), dtype=torch.int32).to("spyre")
        self.name_dims(x, {"M": M, "N": N})
        self.name_dims(i, {"P": P, "Q": Q})
        self._stage_and_e2e(lambda x, i: x.exp()[i], x, i, expect=GATHER_OP_SPEC)

    def test_gather_after_complex_producer_ops(self):
        """x.abs().neg().exp()[i] -- gather with multiple producer ops fused."""
        M, N, P = 128, 256, 32
        x = self.to_spyre(torch.rand(M, N, dtype=torch.float16))
        i = torch.randint(0, M, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(x, {"M": M, "N": N})
        self.name_dims(i, {"P": P})
        self._stage_and_e2e(
            lambda x, i: x.abs().neg().exp()[i], x, i, expect=GATHER_OP_SPEC
        )

    def test_functional_embedding(self):
        """torch.nn.functional.embedding(idx, weight) -- like torch.embedding."""
        V, E, P = 256, 128, 32
        weight = self.to_spyre(torch.rand(V, E, dtype=torch.float16))
        idx = torch.randint(0, V, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(weight, {"V": V, "E": E})
        self.name_dims(idx, {"P": P})
        self._stage_and_e2e(
            lambda i, w: torch.nn.functional.embedding(i, w),
            idx,
            weight,
            expect=GATHER_OP_SPEC,
        )

    # -- P=1 gather: comprehensive shape and rank coverage ----------------
    # Verify that a single-element index (P=1) compiles and produces correct
    # results across 2-D, 3-D, and 4-D value tensors with stick-aligned,
    # non-stick-aligned, small, and large shapes.

    # 2-D value tensor, P=1
    def test_single_idx_2d_stick_aligned(self):
        """P=1, 2-D value [64, 64] — both dims stick-aligned (1 stick per row)."""
        x, i = self._xi(P=1, M=64, N=64)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_2d_non_stick_aligned(self):
        """P=1, 2-D value [67, 100] — non-stick-aligned row width."""
        x, i = self._xi(P=1, M=67, N=100)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_2d_multistick_row(self):
        """P=1, 2-D value [128, 512] — 8 sticks per row."""
        x, i = self._xi(P=1, M=128, N=512)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_2d_small(self):
        """P=1, 2-D value [8, 64] — small table, 1 stick per row."""
        x, i = self._xi(P=1, M=8, N=64)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_2d_int64_index(self):
        """P=1, 2-D value, int64 index."""
        x, i = self._xi(P=1, dtype=torch.int64)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_2d_with_exp(self):
        """P=1, 2-D value, fused exp."""
        x, i = self._xi(P=1)
        self._stage_and_e2e(
            lambda x, i: x[i].exp(), x, i, expect=GATHER_OP_SPEC, op="exp"
        )

    def test_single_idx_2d_with_tanh(self):
        """P=1, 2-D value, fused tanh."""
        x, i = self._xi(P=1)
        self._stage_and_e2e(
            lambda x, i: x[i].tanh(), x, i, expect=GATHER_OP_SPEC, op="tanh"
        )

    def test_single_idx_2d_embedding(self):
        """P=1, torch.embedding — single token lookup."""
        V, E = 256, 128
        weight = self.to_spyre(torch.rand(V, E, dtype=torch.float16))
        idx = torch.randint(0, V, (1,), dtype=torch.int32).to("spyre")
        self.name_dims(weight, {"V": V, "E": E})
        self.name_dims(idx, {"P": 1})
        self._stage_and_e2e(
            lambda w, i: torch.embedding(w, i), weight, idx, expect=GATHER_OP_SPEC
        )

    def test_single_idx_2d_index_select(self):
        """P=1, torch.index_select, 2-D value."""
        x, i = self._xi(P=1)
        self._stage_and_e2e(
            lambda x, i: torch.index_select(x, 0, i), x, i, expect=GATHER_OP_SPEC
        )

    # 3-D value tensor, P=1
    def test_single_idx_3d_stick_aligned(self):
        """P=1, 3-D value [32, 4, 128] — C stick-aligned (2 sticks per row)."""
        x, i = self._xi3d(P=1, A=32, B=4, C=128)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_3d_non_stick_aligned(self):
        """P=1, 3-D value [32, 4, 100] — C not stick-aligned."""
        x, i = self._xi3d(P=1, A=32, B=4, C=100)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_3d_small(self):
        """P=1, 3-D value [8, 2, 64] — small A and B, single-stick C."""
        x, i = self._xi3d(P=1, A=8, B=2, C=64)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_3d_large(self):
        """P=1, 3-D value [64, 16, 256] — large, 4 sticks per C."""
        x, i = self._xi3d(P=1, A=64, B=16, C=256)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_3d_non_power_of_two(self):
        """P=1, 3-D value [48, 6, 96] — non-power-of-two dims."""
        x, i = self._xi3d(P=1, A=48, B=6, C=96)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_3d_int64_index(self):
        """P=1, 3-D value, int64 index."""
        x, i = self._xi3d(P=1, dtype=torch.int64)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_3d_with_exp(self):
        """P=1, 3-D value, fused exp."""
        x, i = self._xi3d(P=1)
        self._stage_and_e2e(
            lambda x, i: x[i].exp(), x, i, expect=GATHER_OP_SPEC, op="exp"
        )

    def test_single_idx_3d_with_tanh(self):
        """P=1, 3-D value, fused tanh."""
        x, i = self._xi3d(P=1)
        self._stage_and_e2e(
            lambda x, i: x[i].tanh(), x, i, expect=GATHER_OP_SPEC, op="tanh"
        )

    def test_single_idx_3d_index_select(self):
        """P=1, torch.index_select, 3-D value."""
        x, i = self._xi3d(P=1)
        self._stage_and_e2e(
            lambda x, i: torch.index_select(x, 0, i), x, i, expect=GATHER_OP_SPEC
        )

    def test_single_idx_3d_chained_unary(self):
        """P=1, 3-D value, chained exp+tanh fused after gather."""
        x, i = self._xi3d(P=1)
        self._stage_and_e2e(lambda x, i: x[i].exp().tanh(), x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_3d_scalar_add(self):
        """P=1, 3-D value, scalar add fused after gather."""
        x, i = self._xi3d(P=1)
        self._stage_and_e2e(lambda x, i: x[i] + 1.0, x, i, expect=GATHER_OP_SPEC)

    # 4-D value tensor, P=1
    def test_single_idx_4d_stick_aligned(self):
        """P=1, 4-D value [8, 4, 8, 64] — D stick-aligned (1 stick per row)."""
        x, i = self._xi4d(P=1, A=8, B=4, C=8, D=64)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_4d_non_stick_aligned(self):
        """P=1, 4-D value [8, 4, 8, 100] — D not stick-aligned."""
        x, i = self._xi4d(P=1, A=8, B=4, C=8, D=100)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_4d_multistick_row(self):
        """P=1, 4-D value [8, 4, 4, 128] — D spans 2 sticks per row."""
        x, i = self._xi4d(P=1, A=8, B=4, C=4, D=128)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_4d_small(self):
        """P=1, 4-D value [4, 2, 2, 64] — small inner dims."""
        x, i = self._xi4d(P=1, A=4, B=2, C=2, D=64)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_4d_int64_index(self):
        """P=1, 4-D value, int64 index."""
        x, i = self._xi4d(P=1, dtype=torch.int64)
        self._stage_and_e2e(lambda x, i: x[i], x, i, expect=GATHER_OP_SPEC)

    def test_single_idx_4d_with_exp(self):
        """P=1, 4-D value, fused exp."""
        x, i = self._xi4d(P=1)
        self._stage_and_e2e(
            lambda x, i: x[i].exp(), x, i, expect=GATHER_OP_SPEC, op="exp"
        )

    def test_single_idx_4d_with_tanh(self):
        """P=1, 4-D value, fused tanh."""
        x, i = self._xi4d(P=1)
        self._stage_and_e2e(
            lambda x, i: x[i].tanh(), x, i, expect=GATHER_OP_SPEC, op="tanh"
        )

    def test_single_idx_4d_index_select(self):
        """P=1, torch.index_select, 4-D value."""
        x, i = self._xi4d(P=1)
        self._stage_and_e2e(
            lambda x, i: torch.index_select(x, 0, i), x, i, expect=GATHER_OP_SPEC
        )

    def test_single_idx_4d_scalar_mul(self):
        """P=1, 4-D value, scalar multiply fused after gather."""
        x, i = self._xi4d(P=1)
        self._stage_and_e2e(lambda x, i: x[i] * 2.0, x, i, expect=GATHER_OP_SPEC)

    # P=1 embedding API variants
    def test_single_idx_embedding_large_vocab(self):
        """P=1, torch.embedding, large vocab (V=4096) with a multi-stick embedding dim."""
        V, E = 4096, 512
        weight = self.to_spyre(torch.rand(V, E, dtype=torch.float16))
        idx = torch.randint(0, V, (1,), dtype=torch.int32).to("spyre")
        self.name_dims(weight, {"V": V, "E": E})
        self.name_dims(idx, {"P": 1})
        self._stage_and_e2e(
            lambda w, i: torch.embedding(w, i), weight, idx, expect=GATHER_OP_SPEC
        )

    def test_single_idx_embedding_non_aligned_dim(self):
        """P=1, torch.embedding, embedding dim not stick-aligned (E=100)."""
        V, E = 256, 100
        weight = self.to_spyre(torch.rand(V, E, dtype=torch.float16))
        idx = torch.randint(0, V, (1,), dtype=torch.int32).to("spyre")
        self.name_dims(weight, {"V": V, "E": E})
        self.name_dims(idx, {"P": 1})
        self._stage_and_e2e(
            lambda w, i: torch.embedding(w, i), weight, idx, expect=GATHER_OP_SPEC
        )

    def test_single_idx_embedding_int64(self):
        """P=1, torch.embedding, int64 index — typical token id dtype."""
        V, E = 256, 128
        weight = self.to_spyre(torch.rand(V, E, dtype=torch.float16))
        idx = torch.randint(0, V, (1,), dtype=torch.int64).to("spyre")
        self.name_dims(weight, {"V": V, "E": E})
        self.name_dims(idx, {"P": 1})
        self._stage_and_e2e(
            lambda w, i: torch.embedding(w, i), weight, idx, expect=GATHER_OP_SPEC
        )

    def test_single_idx_functional_embedding(self):
        """P=1, torch.nn.functional.embedding — single token."""
        V, E = 256, 128
        weight = self.to_spyre(torch.rand(V, E, dtype=torch.float16))
        idx = torch.randint(0, V, (1,), dtype=torch.int32).to("spyre")
        self.name_dims(weight, {"V": V, "E": E})
        self.name_dims(idx, {"P": 1})
        self._stage_and_e2e(
            lambda i, w: torch.nn.functional.embedding(i, w),
            idx,
            weight,
            expect=GATHER_OP_SPEC,
        )

    # -- negative / control scenarios -------------------------------------
    def test_direct_access_control(self):
        """x.exp() (no indexing): direct op spec, no IndirectAccess, no named
        index arg, and not flagged by detection."""
        M, N = 128, 256
        x = self.to_spyre(torch.rand(M, N, dtype=torch.float16))
        self.name_dims(x, {"M": M, "N": N})
        # x.exp() is a supported direct op, so its e2e result must match the CPU
        # reference (expect_close=True) -- unlike the indirect gathers.
        r = self._stage_and_e2e(
            lambda x: x.exp(),
            x,
            expect=DIRECT_OP_SPEC,
            op="exp",
            detected=False,
            expect_close=True,
        )
        self.assertFalse(any(op_spec_has_indirect_access(s) for s in r.op_specs))
        self.assertFalse(
            any(a.name is not None for s in r.op_specs for a in s.args),
            "direct access should have no named index arg",
        )

    def test_unsupported_unary_after_gather_falls_back_to_cpu(self):
        """x[i].sin(): sin has no Spyre lowering so it falls back to CPU.
        The gather still compiles (GATHER_OP_SPEC) and 'sin' is not a Spyre op.
        The Spyre-side gather must still lower to a valid indirect SDSC bundle."""
        x, i = self._xi(P=32)
        with capture_op_specs() as captured:
            torch.compile(lambda x, i: x[i].sin())(x, i)
        op_specs = flatten_op_specs(captured)
        self.assertTrue(any(op_spec_has_indirect_input(s) for s in op_specs))
        self.assertNotIn("sin", [s.op for s in op_specs])
        self.assert_indirect_source_indexed_dim_outermost(op_specs)
        self.assert_indirect_sdsc_fields(bundle_jsons_from_captured(captured), "gather")
        run_e2e(self, lambda x, i: x[i].sin(), x, i)

    # -- SDSC generation field checks (one compile, all fields) -----------
    def _gen_sdsc(self):
        x, i = self._xi(P=3, two_d=True)
        return generate_sdsc_jsons(lambda x, i: x[i].exp(), x, i)

    def test_gather_sdsc_fields(self):
        """Real SDSC generation for x[i].exp(): checks structure, indirect encoding,
        index-tensor HBM-only, wordLength, dsName, node fields, and cross-links.
        """
        jsons = self._gen_sdsc()
        self.assertTrue(jsons, "no SDSC JSON files generated")
        for fname, top in jsons.items():
            self.assertTrue(fname.endswith(".json"))
            self.assertEqual(len(top), 1, "expected one top-level operation key")
            self.assertRegex(next(iter(top)), r"^\d+_.+")

        index_nodes, value_nodes = [], []
        for opfunc, body in iter_sdsc_op_bodies(jsons):
            for field in (
                "N_",
                "scheduleTree_",
                "labeledDs_",
                "computeOp_",
                "numCoresUsed_",
            ):
                self.assertIn(field, body, f"{opfunc} missing field: {field}")
            self.assertEqual(len(body["labeledDs_"]), len(body["scheduleTree_"]))
            self.assertGreaterEqual(body["numCoresUsed_"], 1)
            self.assertLessEqual(body["numCoresUsed_"], 32)
            cop = body["computeOp_"][0]
            self.assertTrue(cop["opFuncName"])
            self.assertIsInstance(cop["inputLabeledDs"], list)
            self.assertTrue(cop["outputLabeledDs"], "output label missing")

            idx_ldsidx = {
                node["ldsIdx_"]
                for node in body["scheduleTree_"]
                if node.get("indirectAllocType_") == "index_tensor"
            }
            for lds in body["labeledDs_"]:
                for key in (
                    "ldsIdx_",
                    "dsName_",
                    "dsType_",
                    "dataFormat_",
                    "wordLength",
                    "memOrg_",
                ):
                    self.assertIn(key, lds)
                self.assertIsInstance(lds["dsName_"], str)
                self.assertIsInstance(lds["dataFormat_"], str)
                self.assertEqual(lds["dsName_"], f"Tensor{lds['ldsIdx_']}")
                if lds["ldsIdx_"] in idx_ldsidx:
                    self.assertIn("hbm", lds["memOrg_"])
                    self.assertNotIn("lx", lds["memOrg_"], "index must be HBM only")
                    self.assertEqual(
                        lds["wordLength"], 4, "index int32 -> wordLength 4"
                    )
                else:
                    self.assertEqual(lds["wordLength"], 2, "value fp16 -> wordLength 2")

            index_nodes += [
                n
                for n in body["scheduleTree_"]
                if n.get("indirectAllocType_") == "index_tensor"
            ]
            value_nodes += [
                n
                for n in body["scheduleTree_"]
                if n.get("indirectAllocType_") == "value_tensor"
            ]

        for node in iter_schedule_tree_nodes(jsons):
            self.assertEqual(node["nodeType_"], "allocate")
            self.assertTrue(node["name_"].startswith("allocate-Tensor"))
            self.assertIn(node["component_"], ("hbm", "lx"))
            self.assertIn(
                node["indirectAllocType_"],
                ("index_tensor", "value_tensor", "no_indirection"),
            )

        self.assertTrue(index_nodes, "no index_tensor nodes found")
        self.assertTrue(value_nodes, "no value_tensor nodes found")
        for n in index_nodes:
            self.assertEqual(n.get("indexTensorType_"), "index")
            self.assertIn("relatedIndirectAccessAlloc_", n)
            related = n.get("relatedIndirectAccessAlloc_")
            if related is not None:
                self.assertRegex(related, r"^allocate-Tensor\d+_hbm$")

    # -- SDSC handoff -----------------------------------------------------
    def test_gather_sdsc_handoff(self):
        """sdsc is invoked with an 'sdsc'-named kernel and a non-empty, fully
        implemented spec list carrying a real OpSpec."""
        x, i = self._xi(P=3, two_d=True)
        with capture_sdsc_calls() as calls:
            torch.compile(lambda x, i: x[i].exp())(x, i)

        self.assertTrue(calls, "sdsc was never called")
        for kernel_name, specs in calls:
            self.assertIsInstance(kernel_name, str)
            self.assertTrue(kernel_name, "kernel name is empty")
            self.assertTrue(specs, "spec list is empty")
            self.assertIsNone(
                find_unimplemented(specs),
                "supported gather should not contain an UnimplementedOp",
            )
        self.assertTrue(
            any("sdsc" in name.lower() for name, _ in calls),
            f"expected an 'sdsc'-named kernel; got {[n for n, _ in calls]}",
        )
        all_specs = flatten_op_specs([[s for _, specs in calls for s in specs]])
        self.assertTrue(all_specs)

    def test_python_bundle_generation_succeeds(self):
        """Real generate_bundle runs end-to-end with the backend mocked.

        All hardware/toolchain touchpoints are stubbed so no device is needed:
        subprocess.run (the dxp_standalone --bundle step that produces
        spyreCodeDir), launch_jobplan (device execution), and
        prepare_kernel (reads spyreCodeDir; only invoked when DUMP_SPYRE_CODE is
        set). Stubbing prepare_kernel + launch_jobplan exercises the
        DUMP_SPYRE_CODE=1 code path harmlessly, so the test passes whether or not
        a developer has DUMP_SPYRE_CODE exported.
        """
        # Mock targets to disable actual kernel execution on device.
        # launch_jobplan and prepare_kernel are only used on the DUMP_SPYRE_CODE
        # path; stubbing them keeps this test hermetic whether or not a developer
        # has DUMP_SPYRE_CODE exported.
        kr = "torch_spyre.execution.kernel_runner"
        x, i = self._xi(P=3, two_d=True)
        with (
            patch("subprocess.run"),
            patch(f"{kr}.launch_jobplan"),
            patch(f"{kr}.prepare_kernel"),
        ):
            result = torch.compile(lambda x, i: x[i].exp())(x, i)
        self.assertIsNotNone(result, "gather compile returned nothing")

    @pytest.mark.skip(
        reason=(
            "dxp_standalone crash: std::out_of_range (map::at) — "
            "multiple gathers sharing the same index tensor in one kernel "
            "are not yet supported"
        )
    )
    def test_gather_multiple_indices_same_kernel(self):
        """Test x[i] + y[i] - multiple gathers with same index in one kernel.

        Verifies that when multiple gather operations use the same index tensor
        in the same kernel, each operation correctly gets only the index tensors
        it needs, not all indirect tensors in the kernel (regression test for
        kernel-level indirect_vars carryover bug).
        """
        x, i = self._xi(P=3, two_d=True, dtype=torch.int64)
        y = torch.randn_like(x)

        def kernel(x, y, i):
            return x[i] + y[i]

        self._stage_and_e2e(kernel, x, y, i, expect=GATHER_OP_SPEC)

    @pytest.mark.skip(
        reason=(
            "RuntimeError: StreamInErrorState — cannot launch D2H; "
            "multiple gathers with different index tensors in one kernel "
            "are not yet supported"
        )
    )
    def test_gather_different_indices_same_kernel(self):
        """Test x[i] + y[j] - multiple gathers with different indices in one kernel.

        Verifies that when multiple gather operations use different index tensors
        in the same kernel, each operation gets the correct index tensor without
        cross-contamination from other gathers.
        """
        x, i = self._xi(P=3, two_d=True, dtype=torch.int64, M=128, N=256)
        y, j = self._xi(P=3, two_d=True, dtype=torch.int64, M=100, N=256)

        def kernel(x, y, i, j):
            return x[i] + y[j]

        self._stage_and_e2e(kernel, x, y, i, j, expect=GATHER_OP_SPEC)


# Op-behaviour scenarios run once at the default 32 cores. They classify / lower
# / run each op and do not depend on the core count, so sweeping all ~90 of them
# across every SENCORES value added little coverage for a 7x test-count blowup.
register_multicore_variants(_GatherScenarios, "TestGather", globals(), counts=(32,))


class _GatherMulticoreScenarios:
    """Gather scenarios whose BEHAVIOUR depends on the core count -- swept across
    SENCORES, unlike the op-behaviour scenarios above (which run once at 32). The
    work-division split-map tests assert core_split at each count; the cross-core
    test checks shared-table correctness under a real multi-core split; and
    test_moe changes its shape by core count. See MULTICORE_SENCORES."""

    to_spyre = staticmethod(plain_to_spyre)

    def test_moe(self):
        """MoE expert selection: expert_w[expert_ids] with 3D weights and 2D int64 index."""
        from torch_spyre._inductor import config

        E, D, B, S = 8, 512, 2, 128
        # 512 MB unsplit output overflows the 256 MB single-core span; shrink it
        # only for SENCORES=1 and 2, where the output cannot be split across cores.
        # TODO : Set one large size for all cores when span reduction is handled.
        F = 2048 if config.sencores >= 3 else 512
        expert_w = self.to_spyre(torch.rand(E, D, F, dtype=torch.float16))
        expert_ids = torch.randint(0, E, (B, S), dtype=torch.int64).to("spyre")
        self.name_dims(expert_w, {"E": E, "D": D, "F": F})
        self.name_dims(expert_ids, {"B": B, "S": S})
        self._stage_and_e2e(
            lambda w, i: w[i], expert_w, expert_ids, expect=GATHER_OP_SPEC
        )

    def test_embedding_large_vocab_table_no_span_overflow(self):
        """Vocab-scale table (V=262144, E=2816, ~1.4 GB): would overflow the
        256 MB span limit if the indirectly-accessed vocab dim was used in full.
        full, but that dim is never actually resident per-core."""
        V, E = 262144, 2816
        eps = get_elem_in_stick(torch.float16)
        stick_count = (E + eps - 1) // eps
        stl = SpyreTensorLayout(
            device_size=[V, stick_count, eps],
            stride_map=[E, eps, 1],
            device_dtype=get_device_dtype(torch.float16),
        )
        table = torch.rand(V, E, dtype=torch.float16).to("spyre", device_layout=stl)
        token_ids = torch.randint(0, V, (2, 128), dtype=torch.int64).to("spyre")
        self.name_dims(table, {"V": V, "E": E})
        self.name_dims(token_ids, {"B": 2, "S": 128})
        self._stage_and_e2e(lambda t, i: t[i], table, token_ids, expect=GATHER_OP_SPEC)

    # -- Work-division scenarios -----------------------------------------
    # Swept across SENCORES, so each TestGatherMulticore_cores{N} variant
    # checks the split map that N produces. The invariant for out = x[i]: the
    # planner must split the index dim (c0) and never the value-table data dim
    # (c1 = K) -- splitting K makes every core read address 0 of the shared
    # table, silently returning wrong results. Shapes are chosen so the planner
    # *would* prefer K (its largest output-coordinate dim) if the guard were
    # absent. assert_indexed_dim_split() reads the current SENCORES and expects
    # c0 to split by min(SENCORES, index_size // 32) with c1 pinned at 1.
    #
    # After the split-map check each scenario runs through the shared
    # _stage_and_e2e path (fresh inputs, since run_and_get_code has already
    # executed the split-map compile) -- the same capture-path stage checks +
    # e2e leg every other gather scenario uses -- so the multicore split is also
    # exercised end-to-end. (Skipped at sencores=1 by assert_indexed_dim_split
    # -- nothing to divide.)

    @staticmethod
    def _gather_fn(table, idx):
        return table[idx]

    def test_work_division_index_split_full(self):
        """Index dim has 32 sticks (Q=1024): it would split a full 32 ways, but
        the indirect uint32 address cap (INDIRECT_ACCESS_MAX_CORES) holds it to
        16-way at SENCORES=32 while K=64 stays unsplit -- verifying the split map
        and that the cap keeps a full-scale index off the 32-way path the backend
        rejects (a per-core address past 4 GB overflows uint32)."""

        def make():
            x = torch.rand(128, 64, 1024, dtype=torch.float16).to("spyre")
            i = (torch.arange(1024) % 128).int().to("spyre")
            return x, i

        fn = self._gather_fn
        _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), *make())
        self.assert_indexed_dim_split(source_codes[0], index_size=1024, data_size=64)
        self._stage_and_e2e(fn, *make(), expect=GATHER_OP_SPEC)

    def test_work_division_index_split_capped(self):
        """Index dim has only 8 sticks (Q=256): when SENCORES exceeds 8 the split
        caps at 8 and must never spill onto the forbidden K dim."""

        def make():
            x = torch.rand(128, 64, 256, dtype=torch.float16).to("spyre")
            i = (torch.arange(256) % 128).int().to("spyre")
            return x, i

        fn = self._gather_fn
        _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), *make())
        self.assert_indexed_dim_split(source_codes[0], index_size=256, data_size=64)
        self._stage_and_e2e(fn, *make(), expect=GATHER_OP_SPEC)

    def test_work_division_unaligned_data_dim(self):
        """An unaligned value-table data dim (K=48, not divisible by any core
        count) stays unsplit at every SENCORES."""

        def make():
            x = torch.rand(128, 48, 256, dtype=torch.float16).to("spyre")
            i = (torch.arange(256) % 128).int().to("spyre")
            return x, i

        fn = self._gather_fn
        _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), *make())
        self.assert_indexed_dim_split(source_codes[0], index_size=256, data_size=48)
        self._stage_and_e2e(fn, *make(), expect=GATHER_OP_SPEC)

    # -- stick-aligned vs partial-last-stick index-entry counts -----------
    # The index-entry dim is the index tensor's stick dim, so work division
    # splits it in whole 32-entry sticks. A STICK-ALIGNED count (a multiple of
    # 32) has always split cleanly. A NON-aligned count (a partial last stick)
    # used to be forced onto one core, because an even per-core slice straddles
    # the index stick boundary and the backend cannot step a sticked dim across it;
    # the fix pads the gather output's entry dim up to the next whole stick
    # (enforce_indirect_access_layout) so the split lands stick-aligned. Both
    # kinds are swept across every SENCORES; the expected split for either is
    # min(SENCORES, ceil(count / 32)) with the K data dim always pinned at 1.

    def test_work_division_index_split_aligned_six_sticks(self):
        """Stick-aligned index with exactly 6 sticks (P=192): a non-power-of-two
        stick count, so the split is the largest divisor of 6 that fits the core
        count -- 6 at SENCORES>=6, 3 at 4 cores, 2 at 2 -- exercising the
        SENCORES=6 case. Complements the 8- and 32-stick aligned cases.
        """

        def make():
            x = torch.rand(128, 64, 256, dtype=torch.float16).to("spyre")
            i = (torch.arange(192) % 128).int().to("spyre")
            return x, i

        fn = self._gather_fn
        _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), *make())
        self.assert_indexed_dim_split(source_codes[0], index_size=192, data_size=64)
        self._stage_and_e2e(fn, *make(), expect=GATHER_OP_SPEC)

    def test_work_division_index_split_partial_stick(self):
        """Non-stick-aligned index (P=40 = one full stick + a partial second):
        the entry dim is padded up to ceil(40/32)=2 whole sticks so the split
        lands stick-aligned (2 across any SENCORES>=2) instead of an even slice
        that would straddle the index stick boundary and miscompile."""

        def make():
            x = torch.rand(128, 64, 256, dtype=torch.float16).to("spyre")
            i = (torch.arange(40) % 128).int().to("spyre")
            return x, i

        fn = self._gather_fn
        _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), *make())
        self.assert_indexed_dim_split(source_codes[0], index_size=40, data_size=64)
        self._stage_and_e2e(fn, *make(), expect=GATHER_OP_SPEC)

    def test_work_division_index_split_partial_stick_mid(self):
        """Non-stick-aligned index spanning several sticks (P=250, ceil=8 sticks):
        the padded split scales past a single partial stick -- the non-aligned
        counterpart of the aligned 8-stick test_work_division_index_split_capped,
        splitting by the same largest divisor of 8 that fits the core count."""

        def make():
            x = torch.rand(128, 64, 256, dtype=torch.float16).to("spyre")
            i = (torch.arange(250) % 128).int().to("spyre")
            return x, i

        fn = self._gather_fn
        _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), *make())
        self.assert_indexed_dim_split(source_codes[0], index_size=250, data_size=64)
        self._stage_and_e2e(fn, *make(), expect=GATHER_OP_SPEC)

    def test_work_division_index_split_partial_stick_full(self):
        """Non-stick-aligned index at 32-stick scale (P=1000, ceil=32): the padded
        split would reach a full 32-way division, but the indirect uint32 address
        cap holds it to 16-way at SENCORES=32 -- the partial-stick counterpart of
        test_work_division_index_split_full."""

        def make():
            x = torch.rand(128, 64, 256, dtype=torch.float16).to("spyre")
            i = (torch.arange(1000) % 128).int().to("spyre")
            return x, i

        fn = self._gather_fn
        _, source_codes = run_and_get_code(torch.compile(fn, dynamic=False), *make())
        self.assert_indexed_dim_split(source_codes[0], index_size=1000, data_size=64)
        self._stage_and_e2e(fn, *make(), expect=GATHER_OP_SPEC)

    # -- shared value table: cross-core read correctness ------------------
    def test_gather_cross_core_shared_value(self):
        """Every output row gathers a value row in a DIFFERENT core's work slice,
        and must still be correct -- the value table is shared, not partitioned.

        The table is row-identifying (weight[r, :] == r, exact in fp16 for
        r < 2048), and the index maps output row i -> value row (i + V/2) % V:
        always a half-table hop, so under any multi-core split the fetched row
        belongs to another core's slice. The gather must match the CPU reference
        exactly (expect_close=True) -- if the shared value tensor's per-core base
        ever drifted with the work-division slice, a core would read a shifted
        row and diverge. Runs at 32 cores, where the indirect uint32 address cap
        holds the split to 16-way (still cross-core: a V/2 hop lands 8 cores
        away), exercising the shared-table property at a real multicore split.
        """
        V, E = 1024, 128
        weight = self.to_spyre(
            torch.arange(V, dtype=torch.float16).unsqueeze(1).repeat(1, E)
        )
        idx = ((torch.arange(V) + V // 2) % V).to(torch.int32).to("spyre")
        self.name_dims(weight, {"V": V, "E": E})
        self.name_dims(idx, {"P": V})
        self._stage_and_e2e(
            lambda w, i: torch.embedding(w, i),
            weight,
            idx,
            expect=GATHER_OP_SPEC,
            expect_close=True,
        )


# Scenarios whose BEHAVIOUR varies with the core count -- the work-division
# split-map tests, the cross-core correctness test, and test_moe (core-dependent
# shape) -- are swept across all SENCORES values.
register_multicore_variants(_GatherMulticoreScenarios, "TestGatherMulticore", globals())


# ---------------------------------------------------------------------------
# Registered through register_multicore_variants with a single count (32,) — so
# it is a runtime-generated TestCase like the TestGather_cores* classes rather
# than a literal class def. Pinned to sencores=32 because that is the only count
# that exercises all 32 partial-stick buckets
# ---------------------------------------------------------------------------
class _EmbeddingAllIndexBucketsScenario:
    """torch.embedding over every partial-stick index-stick count at sencores=32.

    The index is int32 (32 entries/stick), so ceil(N/32) is the number of index
    sticks -- and at SENCORES=32 the entry dim splits core_split(sticks, 32) ==
    sticks ways, i.e. the stick count is also the core count.
    For each count s:
      N = s*32 - offset  (offset ∈ [1,31] seeded from s)
      → N is non-stick-aligned, _pad_output_for_stick_aligned_split fires
      → entry dim and hidden dim are co-split to keep per-core span ≤ 256 MB
      → classified as GATHER_OP_SPEC, then validated end-to-end on the backend.
    """

    _A = 30000  # TODO : Increase the size when span reduction is handled.
    _STICK = 4096

    @staticmethod
    def _embedding_fn(weight, input_ids):
        return torch.embedding(weight, input_ids)

    def _make(self, N):
        weight = torch.rand(self._A, self._STICK, dtype=torch.float16).to("spyre")
        input_ids = (torch.arange(N) % self._A).int().to("spyre")
        return weight, input_ids

    @staticmethod
    def _n_for_index_sticks(sticks: int) -> int:
        """Non-stick-aligned N spanning `sticks` index sticks: N = sticks*32 - offset."""
        rng = torch.Generator()
        rng.manual_seed(sticks)
        offset = int(torch.randint(1, 32, (1,), generator=rng).item())
        return sticks * 32 - offset

    def _run_for_index_sticks(self, sticks: int):
        N = self._n_for_index_sticks(sticks)
        torch._dynamo.reset()
        self._stage_and_e2e(self._embedding_fn, *self._make(N), expect=GATHER_OP_SPEC)


def _make_index_sticks_test(sticks: int):
    """One test method for a single partial-stick index-stick (== core) count."""

    def test(self):
        self._run_for_index_sticks(sticks)

    # `sticks` index sticks == `sticks` cores at SENCORES=32 (core_split == sticks).
    test.__name__ = test.__qualname__ = f"test_gather_{sticks:02d}_index_sticks"
    test.__doc__ = (
        f"Classify + e2e torch.embedding across {sticks} partial index stick(s) "
        f"({sticks} cores at SENCORES=32)."
    )
    return test


# Generate one collectible test method per index-stick count on the scenario
# mixin BEFORE register_multicore_variants mixes it into the concrete TestCase,
# so each stick count is an independent test rather than one serial 32-iteration
# loop.
for _sticks in range(1, 33):
    _sticks_test = _make_index_sticks_test(_sticks)
    setattr(_EmbeddingAllIndexBucketsScenario, _sticks_test.__name__, _sticks_test)
del _sticks, _sticks_test


register_multicore_variants(
    _EmbeddingAllIndexBucketsScenario,
    "TestEmbeddingAllIndexBuckets",
    globals(),
    counts=(32,),
)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
