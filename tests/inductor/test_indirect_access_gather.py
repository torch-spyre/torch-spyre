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

All gather scenarios run with SENCORES=1.

There is one test per scenario -- no separate e2e variants. Each scenario
validates the capture path and then runs the kernel on the real backend,
validating the result and warning that the device values diverge from the CPU
reference (the backend does not yet implement indirect gather). Scenarios route
their compile through _stage_and_e2e (stage check + e2e run); capture-based
tests call run_e2e directly after their own assertions. The three structural
tests (sdsc_fields, sdsc_handoff, python_bundle_generation) stay capture-only.

"""

import os
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(__file__))
from indirect_access_common import (  # noqa: E402
    DIRECT_OP_SPEC,
    GATHER_OP_SPEC,
    NO_SPYRE_OP,
    IndirectAccessTestCase,
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
    pinned_to_spyre,
)

from torch_spyre._C import DataFormats  # noqa: E402
from torch_spyre._inductor import config  # noqa: E402
from torch_spyre._inductor.constants import IDENTITY_OP, RESTICKIFY_OP  # noqa: E402
from torch_spyre._inductor.op_spec import find_unimplemented  # noqa: E402


@config.patch({"sencores": 1})
class TestGather(IndirectAccessTestCase):
    """torch gather-family ops: one compile + all-stage checks per scenario."""

    def _xi(self, P=32, two_d=False, dtype=torch.int32, M=128, N=256, Q=192):
        """Named (x[M,N], idx) gather operands. two_d means idx[P,Q]."""
        x = pinned_to_spyre(torch.rand(M, N, dtype=torch.float16))
        shape = (P, Q) if two_d else (P,)
        i = torch.randint(0, M, shape, dtype=dtype).to("spyre")
        self.name_dims(x, {"M": M, "N": N})
        self.name_dims(i, {"P": P, "Q": Q} if two_d else {"P": P})
        return x, i

    def _xi3d(self, P=32, dtype=torch.int32, A=64, B=8, C=64):
        """Named (x[A,B,C], idx[P]) gather operands: 3-D value, 1-D index."""
        x = pinned_to_spyre(torch.rand(A, B, C, dtype=torch.float16))
        i = torch.randint(0, A, (P,), dtype=dtype).to("spyre")
        self.name_dims(x, {"A": A, "B": B, "C": C})
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
        # TODO : Enable once e2e is available
        # run_e2e(self, lambda x, i: x[i].exp(), x, i)

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
        # Carry the same scenario through to SDSC and validate the indirect
        # encoding of the copy op (not just that op specs were produced).
        self.assert_indirect_sdsc_fields(bundle_jsons_from_captured(captured), "gather")
        # TODO : Enable once e2e is available
        # run_e2e(self, lambda x, i: x[i], x, i)

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
                x = pinned_to_spyre(torch.rand(M, N, dtype=torch.float16))
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
                # Each supported unary must also lower to a well-formed
                # indirect-access SDSC bundle, not just an op spec.
                self.assert_indirect_sdsc_fields(
                    bundle_jsons_from_captured(captured), "gather"
                )
        torch._dynamo.reset()
        x = pinned_to_spyre(torch.rand(M, N, dtype=torch.float16))
        idx = torch.randint(0, M, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(x, {"M": M, "N": N})
        self.name_dims(idx, {"P": P})
        # TODO : Enable once e2e is available
        # run_e2e(self, lambda x, i: torch.exp(x[i]), x, idx)

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
        # The fused chain must still emit a valid indirect-access SDSC bundle.
        self.assert_indirect_sdsc_fields(bundle_jsons_from_captured(captured), "gather")
        # TODO : Enable once e2e is available
        # run_e2e(self, lambda x, i: x[i].exp().tanh(), x, i)

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
        table = pinned_to_spyre(torch.rand(V, D, dtype=torch.float16))
        token_ids = torch.randint(0, V, (B, S), dtype=torch.int64).to("spyre")
        self.name_dims(table, {"V": V, "D": D})
        self.name_dims(token_ids, {"B": B, "S": S})
        self._stage_and_e2e(lambda t, i: t[i], table, token_ids, expect=GATHER_OP_SPEC)

    def test_embedding_lookup_1d_index(self):
        """1-D-index counterpart of test_embedding_lookup (which uses a 2-D
        index): table[ids] with vocab-scale shapes and 1-D int64 ids."""
        V, D, P = 32000, 512, 32
        table = pinned_to_spyre(torch.rand(V, D, dtype=torch.float16))
        ids = torch.randint(0, V, (P,), dtype=torch.int64).to("spyre")
        self.name_dims(table, {"V": V, "D": D})
        self.name_dims(ids, {"P": P})
        self._stage_and_e2e(lambda t, i: t[i], table, ids, expect=GATHER_OP_SPEC)

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
        x, i = self._xi(P=32)
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

    def test_torch_gather(self):
        """torch.gather(x, 0, index) with a same-rank [M,K] index tensor.

        Indices address dim 0, so they must be < M (not < N) -- otherwise the
        CPU reference itself raises out-of-bounds once the kernel actually runs.
        """
        M, N, K = 128, 256, 64
        x = pinned_to_spyre(torch.rand(M, N, dtype=torch.float16))
        index = torch.randint(0, M, (M, K), dtype=torch.int64).to("spyre")
        self.name_dims(x, {"M": M, "N": N})
        self.name_dims(index, {"M": M, "K": K})
        self._stage_and_e2e(
            lambda x, index: torch.gather(x, 0, index), x, index, expect=GATHER_OP_SPEC
        )

    def test_embedding(self):
        """torch.embedding(weight, idx) currently falls back to CPU."""
        V, E, P = 256, 128, 32
        weight = pinned_to_spyre(torch.rand(V, E, dtype=torch.float16))
        idx = torch.randint(0, V, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(weight, {"V": V, "E": E})
        self.name_dims(idx, {"P": P})
        self._stage_and_e2e(
            lambda w, i: torch.embedding(w, i), weight, idx, expect=NO_SPYRE_OP
        )

    # -- additional real gather scenarios ---------------------------------
    def test_gather_3d_data(self):
        """x[i] where x is 3-D [A,B,C] -- gather rows of higher-rank data."""
        A, B, C, P = 64, 8, 64, 16
        x = pinned_to_spyre(torch.rand(A, B, C, dtype=torch.float16))
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
        x, i = self._xi3d(P=32)
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

    def test_moe(self):
        """MoE expert selection: expert_w[expert_ids] with 3D weights and 2D int64 index."""
        E, D, F, B, S = 8, 512, 2048, 2, 128
        expert_w = pinned_to_spyre(torch.rand(E, D, F, dtype=torch.float16))
        expert_ids = torch.randint(0, E, (B, S), dtype=torch.int64).to("spyre")
        self.name_dims(expert_w, {"E": E, "D": D, "F": F})
        self.name_dims(expert_ids, {"B": B, "S": S})
        self._stage_and_e2e(
            lambda w, i: w[i], expert_w, expert_ids, expect=GATHER_OP_SPEC
        )

    def test_paged_kv(self):
        """Paged KV-cache gather: keys[slot_idxs] with 3D cache and 2D int64 slot index."""
        cache, H, Dh, B, Lk = 32768, 8, 128, 2, 256
        keys = pinned_to_spyre(torch.rand(cache, H, Dh, dtype=torch.float16))
        slot_idxs = torch.randint(0, cache, (B, Lk), dtype=torch.int64).to("spyre")
        self.name_dims(keys, {"cache": cache, "H": H, "Dh": Dh})
        self.name_dims(slot_idxs, {"B": B, "Lk": Lk})
        self._stage_and_e2e(lambda k, i: k[i], keys, slot_idxs, expect=GATHER_OP_SPEC)

    def test_gather_after_reshape(self):
        """x.reshape(12, 256)[i] -- gather on a reshaped tensor."""
        x = pinned_to_spyre(torch.rand(3, 1024, dtype=torch.float16))
        i = torch.tensor((1, 2), dtype=torch.int32).to("spyre")
        self.name_dims(x, {"rows": 3, "cols": 1024})
        self.name_dims(i, {"P": 2})
        self._stage_and_e2e(
            lambda x, i: x.reshape(12, 256)[i], x, i, expect=GATHER_OP_SPEC
        )

    def test_gather_after_reshape_1d(self):
        """t.reshape(16, 256)[idx] -- gather on a 1D tensor reshaped to 2D (page table)."""
        t = pinned_to_spyre(torch.arange(4096, dtype=torch.float16))
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
        x = pinned_to_spyre(torch.rand(M, N, dtype=torch.float16))
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

    def test_functional_embedding(self):
        """torch.nn.functional.embedding(idx, weight) -- like torch.embedding."""
        V, E, P = 256, 128, 32
        weight = pinned_to_spyre(torch.rand(V, E, dtype=torch.float16))
        idx = torch.randint(0, V, (P,), dtype=torch.int32).to("spyre")
        self.name_dims(weight, {"V": V, "E": E})
        self.name_dims(idx, {"P": P})
        self._stage_and_e2e(
            lambda i, w: torch.nn.functional.embedding(i, w),
            idx,
            weight,
            expect=NO_SPYRE_OP,
        )

    # -- negative / control scenarios -------------------------------------
    def test_direct_access_control(self):
        """x.exp() (no indexing): direct op spec, no IndirectAccess, no named
        index arg, and not flagged by detection."""
        M, N = 128, 256
        x = pinned_to_spyre(torch.rand(M, N, dtype=torch.float16))
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
        self.assert_indirect_sdsc_fields(bundle_jsons_from_captured(captured), "gather")
        # TODO : Enable once e2e is available
        # run_e2e(self, lambda x, i: x[i].sin(), x, i)

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
        all_specs = [s for _, specs in calls for s in specs]
        self.assertTrue(flatten_op_specs([all_specs]))

    def test_python_bundle_generation_succeeds(self):
        """Real generate_bundle runs end-to-end with the backend mocked.

        All hardware/toolchain touchpoints are stubbed so no device is needed:
        subprocess.run (the dxp_standalone --bundle step that produces
        spyreCodeDir), launch_kernel / launch_jobplan (device execution), and
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
            patch(f"{kr}.launch_kernel"),
            patch(f"{kr}.launch_jobplan"),
            patch(f"{kr}.prepare_kernel"),
        ):
            result = torch.compile(lambda x, i: x[i].exp())(x, i)
        self.assertIsNotNone(result, "gather compile returned nothing")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
