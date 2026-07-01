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

"""Device-free unit tests for the indirect-access harness and helpers.

These need no compilation, no Spyre device -- they exercise the primitives the
op-family tests (test_indirect_access_gather.py / test_indirect_access_scatter.py) are built on:

  * the IndirectAccess sympy atom,
  * the detection helpers on synthetic OpSpecs,
  * the capture-harness patch/restore behavior,
  * the spec-tree flattening / sdsc-json navigation utilities.
"""

import os
import sys

import sympy

sys.path.insert(0, os.path.dirname(__file__))
from indirect_access_common import (  # noqa: E402
    CRASHED,
    DIRECT_OP_SPEC,
    GATHER_OP_SPEC,
    NO_SPYRE_OP,
    SCATTER_OP_SPEC,
    UNIMPLEMENTED,
    IndirectAccessTestCase,
    _label_for,
    alloc_node_index,
    arg_has_indirect_access,
    capture_op_specs,
    capture_sdsc_calls,
    coord_atoms,
    flatten_entries,
    flatten_op_specs,
    indirect_access_target_names,
    iter_schedule_tree_nodes,
    iter_sdsc_op_bodies,
    labeled_ds_index,
    op_spec_has_indirect_access,
    op_spec_has_indirect_input,
    op_spec_has_indirect_output,
)

from torch_spyre._C import DataFormats  # noqa: E402
from torch_spyre._inductor.op_spec import (  # noqa: E402
    IndirectAccess,
    LoopSpec,
    OpSpec,
    TensorArg,
    UnimplementedOp,
)
from torch_spyre.execution.async_compile import SpyreAsyncCompile  # noqa: E402


def _tensor_arg(is_input, coords, name=None):
    return TensorArg(
        is_input=is_input,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[1, 128, 64],
        device_coordinates=coords,
        allocation={},
        name=name,
    )


def _op(name):
    return OpSpec(op=name, is_reduction=False, iteration_space={}, args=[], op_info={})


# ===========================================================================
# Layer 1: IndirectAccess atom tests (pure sympy, no device)
# ===========================================================================
class TestIndirectAccessAtom(IndirectAccessTestCase):
    def test_atom_is_opaque_and_carries_name(self):
        """Test that IndirectAccess atom stores and returns the buffer name."""
        ia = IndirectAccess(sympy.Symbol("arg1_1"))
        self.assertIsInstance(ia, IndirectAccess)
        self.assertEqual(str(ia.args[0]), "arg1_1")

    def test_atom_found_inside_compound_expression(self):
        """IndirectAccess can be found within complex sympy expressions."""
        c1 = sympy.Symbol("c1")
        ia = IndirectAccess(sympy.Symbol("arg1_1"))
        expr = sympy.floor(c1 / 64) + ia * 128
        found = expr.atoms(IndirectAccess)
        self.assertEqual(len(found), 1)
        self.assertIn(ia, found)

    def test_direct_expression_has_no_indirect_atom(self):
        """Regular expressions without IndirectAccess return an empty set."""
        c0, c1 = sympy.symbols("c0 c1")
        expr = sympy.floor(c1 / 64) + c0
        self.assertEqual(expr.atoms(IndirectAccess), set())

    def test_xreplace_substitutes_indirect_symbol(self):
        """Replace a symbol with IndirectAccess (mimics compute_coordinates)."""
        c1, tmp0 = sympy.symbols("c1 tmp0")
        coord = sympy.floor(c1 / 64) + tmp0 * 128
        out = coord.xreplace({tmp0: IndirectAccess(sympy.Symbol("arg1_1"))})
        self.assertNotIn(tmp0, out.free_symbols)
        self.assertEqual(len(out.atoms(IndirectAccess)), 1)

    def test_distinct_names_are_distinct_atoms(self):
        """IndirectAccess atoms with different names are distinct atoms."""
        a = IndirectAccess(sympy.Symbol("buf_a"))
        b = IndirectAccess(sympy.Symbol("buf_b"))
        self.assertNotEqual(a, b)
        self.assertEqual(len((a + b).atoms(IndirectAccess)), 2)

    def test_indirect_access_with_constant_offset(self):
        ia = IndirectAccess(sympy.Symbol("arg1_1"))
        self.assertEqual(len((ia + 5).atoms(IndirectAccess)), 1)

    def test_indirect_access_inside_mod(self):
        c1 = sympy.Symbol("c1")
        ia = IndirectAccess(sympy.Symbol("arg1_1"))
        expr = sympy.Mod(ia + c1, 64)
        self.assertEqual(len(expr.atoms(IndirectAccess)), 1)

    def test_indirect_access_equality_same_name(self):
        self.assertEqual(
            IndirectAccess(sympy.Symbol("b")), IndirectAccess(sympy.Symbol("b"))
        )

    def test_indirect_access_usable_in_set(self):
        s = {IndirectAccess(sympy.Symbol("b")), IndirectAccess(sympy.Symbol("b"))}
        self.assertEqual(len(s), 1)

    def test_two_indirect_symbols_substituted(self):
        t0, t1, c0 = sympy.symbols("t0 t1 c0")
        expr = t0 * 256 + t1 + c0
        out = expr.xreplace(
            {
                t0: IndirectAccess(sympy.Symbol("a")),
                t1: IndirectAccess(sympy.Symbol("b")),
            }
        )
        self.assertEqual(len(out.atoms(IndirectAccess)), 2)
        self.assertNotIn(t0, out.free_symbols)
        self.assertNotIn(t1, out.free_symbols)


# ===========================================================================
# Layer 2: Detection helpers on synthetic OpSpecs (no compile)
# ===========================================================================
class TestDetectionHelpers(IndirectAccessTestCase):
    def _op(self, args):
        """Helper to create a synthetic OpSpec for testing."""
        return OpSpec(
            op="identity",
            is_reduction=False,
            iteration_space={},
            args=args,
            op_info={},
        )

    def test_gather_signature_detected(self):
        """Gather pattern (IndirectAccess on an input) is identified."""
        ia = IndirectAccess(sympy.Symbol("idx_buf"))
        gathered_in = _tensor_arg(True, [ia, sympy.Symbol("c0"), sympy.Symbol("c1")])
        out = _tensor_arg(False, list(sympy.symbols("c0 c1 c2")))
        op = self._op([gathered_in, out])

        self.assertTrue(arg_has_indirect_access(gathered_in))
        self.assertFalse(arg_has_indirect_access(out))
        self.assertTrue(op_spec_has_indirect_access(op))
        self.assertTrue(op_spec_has_indirect_input(op))
        self.assertFalse(op_spec_has_indirect_output(op))

    def test_scatter_signature_detected(self):
        """Scatter pattern (IndirectAccess on an output) is identified."""
        ia = IndirectAccess(sympy.Symbol("idx_buf"))
        src_in = _tensor_arg(True, list(sympy.symbols("c0 c1 c2")))
        scattered_out = _tensor_arg(False, [ia, sympy.Symbol("c0"), sympy.Symbol("c1")])
        op = self._op([src_in, scattered_out])

        self.assertTrue(op_spec_has_indirect_access(op))
        self.assertFalse(op_spec_has_indirect_input(op))
        self.assertTrue(op_spec_has_indirect_output(op))

    def test_direct_op_no_false_positive(self):
        """Direct operations don't falsely trigger indirect detection."""
        a = _tensor_arg(True, list(sympy.symbols("c0 c1 c2")))
        b = _tensor_arg(False, list(sympy.symbols("c0 c1 c2")))
        op = self._op([a, b])
        self.assertFalse(op_spec_has_indirect_access(op))
        self.assertFalse(op_spec_has_indirect_input(op))
        self.assertFalse(op_spec_has_indirect_output(op))

    def test_coord_atoms_extracts_target_name(self):
        """coord_atoms extracts buffer names from IndirectAccess."""
        ia = IndirectAccess(sympy.Symbol("idx_buf"))
        arg = _tensor_arg(True, [ia, sympy.Symbol("c0")])
        atoms = coord_atoms(arg, IndirectAccess)
        self.assertEqual({str(a.args[0]) for a in atoms}, {"idx_buf"})

    def test_coord_atoms_tolerates_non_sympy_coordinate(self):
        """A bare `0` coord has no .atoms() but must not error."""
        ia = IndirectAccess(sympy.Symbol("idx_buf"))
        arg = _tensor_arg(True, [0, ia, sympy.Symbol("c0")])
        self.assertTrue(arg_has_indirect_access(arg))

    def test_multiple_index_tensors_all_detected(self):
        """A 2-D advanced index uses two index buffers - both are found."""
        ia0 = IndirectAccess(sympy.Symbol("idx0"))
        ia1 = IndirectAccess(sympy.Symbol("idx1"))
        arg = _tensor_arg(True, [ia0, ia1, sympy.Symbol("c0")])
        names = {str(a.args[0]) for a in coord_atoms(arg, IndirectAccess)}
        self.assertEqual(names, {"idx0", "idx1"})

    def test_empty_args_has_no_indirect(self):
        op = self._op([])
        self.assertFalse(op_spec_has_indirect_access(op))
        self.assertFalse(op_spec_has_indirect_input(op))
        self.assertFalse(op_spec_has_indirect_output(op))

    def test_multiple_inputs_single_indirect(self):
        ia = IndirectAccess(sympy.Symbol("idx"))
        plain = _tensor_arg(True, list(sympy.symbols("c0 c1 c2")))
        gathered = _tensor_arg(True, [ia, sympy.Symbol("c0"), sympy.Symbol("c1")])
        out = _tensor_arg(False, list(sympy.symbols("c0 c1 c2")))
        op = self._op([plain, gathered, out])
        self.assertTrue(op_spec_has_indirect_input(op))
        self.assertFalse(op_spec_has_indirect_output(op))

    def test_indirect_access_target_names_extracted(self):
        ia = IndirectAccess(sympy.Symbol("idx_buf"))
        gathered = _tensor_arg(True, [ia, sympy.Symbol("c0")])
        out = _tensor_arg(False, list(sympy.symbols("c0 c1")))
        names = indirect_access_target_names([self._op([gathered, out])])
        self.assertEqual(names, {"idx_buf"})

    def test_zero_coordinate_arg_no_false_positive(self):
        arg = _tensor_arg(True, [0, 0, 0])
        self.assertFalse(arg_has_indirect_access(arg))

    def test_gather_and_scatter_signatures_mutually_exclusive(self):
        ia = IndirectAccess(sympy.Symbol("idx"))
        gather = self._op(
            [
                _tensor_arg(True, [ia, sympy.Symbol("c0")]),
                _tensor_arg(False, list(sympy.symbols("c0 c1"))),
            ]
        )
        scatter = self._op(
            [
                _tensor_arg(True, list(sympy.symbols("c0 c1"))),
                _tensor_arg(False, [ia, sympy.Symbol("c0")]),
            ]
        )
        self.assertTrue(op_spec_has_indirect_input(gather))
        self.assertFalse(op_spec_has_indirect_output(gather))
        self.assertTrue(op_spec_has_indirect_output(scatter))
        self.assertFalse(op_spec_has_indirect_input(scatter))

    def test_op_spec_indirect_on_both_input_and_output(self):
        """An op with IndirectAccess on a distinct input and output trips both
        the input and output helpers."""
        ia = IndirectAccess(sympy.Symbol("idx"))
        op = self._op(
            [
                _tensor_arg(True, [ia, sympy.Symbol("c0")]),
                _tensor_arg(False, [ia, sympy.Symbol("c0")]),
            ]
        )
        self.assertTrue(op_spec_has_indirect_input(op))
        self.assertTrue(op_spec_has_indirect_output(op))
        self.assertTrue(op_spec_has_indirect_access(op))

    def test_coord_atoms_empty_coords(self):
        """An arg with no coordinates yields no atoms and no indirect access."""
        arg = _tensor_arg(True, [])
        self.assertEqual(coord_atoms(arg, IndirectAccess), set())
        self.assertFalse(arg_has_indirect_access(arg))

    def test_target_names_across_multiple_op_specs(self):
        """indirect_access_target_names unions index names across op specs."""
        a = self._op([_tensor_arg(True, [IndirectAccess(sympy.Symbol("ia"))])])
        b = self._op([_tensor_arg(True, [IndirectAccess(sympy.Symbol("ib"))])])
        self.assertEqual(indirect_access_target_names([a, b]), {"ia", "ib"})


# ===========================================================================
# Capture harness self-checks (patch/restore)
# ===========================================================================
class TestCaptureHarness(IndirectAccessTestCase):
    def test_capture_restores_sdsc(self):
        """capture_op_specs patches and restores sdsc."""
        original = SpyreAsyncCompile.sdsc
        with capture_op_specs() as captured:
            self.assertIsNot(SpyreAsyncCompile.sdsc, original, "should be patched")
            self.assertEqual(captured, [], "nothing compiled yet")
        self.assertIs(SpyreAsyncCompile.sdsc, original, "must be restored")

    def test_capture_sdsc_calls_restores_on_exception(self):
        """sdsc is restored even when the body raises."""
        original = SpyreAsyncCompile.sdsc
        with self.assertRaises(ValueError):
            with capture_sdsc_calls():
                raise ValueError("boom")
        self.assertIs(SpyreAsyncCompile.sdsc, original, "must be restored after error")


# ===========================================================================
# Spec-tree flattening / sdsc-json navigation utilities (no device)
# ===========================================================================
class TestSpecTreeUtils(IndirectAccessTestCase):
    def test_flatten_descends_into_loopspec(self):
        """flatten_op_specs extracts ops from nested LoopSpecs."""
        inner = LoopSpec(count=2, body=[_op("exp")])
        loop = LoopSpec(count=4, body=[_op("identity"), inner])
        specs = flatten_op_specs([[loop, _op("tanh")]])
        self.assertEqual([s.op for s in specs], ["identity", "exp", "tanh"])

    def test_flatten_op_specs_drops_unimplemented(self):
        """flatten_op_specs ignores UnimplementedOps."""
        specs = flatten_op_specs([[_op("exp"), UnimplementedOp("sin")]])
        self.assertEqual([s.op for s in specs], ["exp"])

    def test_flatten_entries_keeps_unimplemented(self):
        """flatten_entries includes UnimplementedOps."""
        entries = flatten_entries([[_op("exp"), UnimplementedOp("sin")]])
        kinds = [type(e).__name__ for e in entries]
        self.assertIn("UnimplementedOp", kinds)
        self.assertIn("OpSpec", kinds)

    def test_flatten_multiple_sdsc_calls(self):
        """Multiple sdsc calls (multiple kernels) flatten correctly."""
        specs = flatten_op_specs([[_op("identity")], [_op("exp")]])
        self.assertEqual([s.op for s in specs], ["identity", "exp"])

    def test_classification_labels_are_distinct(self):
        """All classification outcome labels are unique."""
        labels = {
            CRASHED,
            GATHER_OP_SPEC,
            SCATTER_OP_SPEC,
            DIRECT_OP_SPEC,
            UNIMPLEMENTED,
            NO_SPYRE_OP,
        }
        self.assertEqual(len(labels), 6, "classification labels must be unique")

    def test_flatten_empty(self):
        self.assertEqual(flatten_op_specs([]), [])
        self.assertEqual(flatten_op_specs([[]]), [])

    def test_flatten_entries_empty(self):
        self.assertEqual(flatten_entries([]), [])

    def test_flatten_deeply_nested_loopspec(self):
        deep = LoopSpec(count=2, body=[LoopSpec(count=2, body=[_op("exp")])])
        outer = LoopSpec(count=4, body=[_op("identity"), deep])
        specs = flatten_op_specs([[outer]])
        self.assertEqual([s.op for s in specs], ["identity", "exp"])

    def test_flatten_entries_mixed(self):
        entries = flatten_entries(
            [[_op("a"), LoopSpec(count=2, body=[UnimplementedOp("sin"), _op("b")])]]
        )
        self.assertEqual([getattr(e, "op", None) for e in entries], ["a", "sin", "b"])

    def test_iter_sdsc_op_bodies_synthetic(self):
        fake = {
            "sdsc_0.json": {
                "0_exp": {"dscs_": [{"exp": {"scheduleTree_": [{"ldsIdx_": 0}]}}]}
            }
        }
        bodies = list(iter_sdsc_op_bodies(fake))
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0][0], "exp")

    def test_iter_schedule_tree_nodes_synthetic(self):
        fake = {
            "sdsc_0.json": {
                "0_exp": {
                    "dscs_": [
                        {"exp": {"scheduleTree_": [{"ldsIdx_": 0}, {"ldsIdx_": 1}]}}
                    ]
                }
            }
        }
        self.assertEqual(len(list(iter_schedule_tree_nodes(fake))), 2)


# ===========================================================================
# Classification logic (_label_for) -- the core of classify_compile /
# run_scenario, tested directly on synthetic outcomes (no compile)
# ===========================================================================
class TestClassificationLogic(IndirectAccessTestCase):
    def _arg(self, is_input, indirect=False):
        coords = (
            [IndirectAccess(sympy.Symbol("idx")), sympy.Symbol("c0")]
            if indirect
            else list(sympy.symbols("c0 c1"))
        )
        return _tensor_arg(is_input, coords)

    def _spec(self, *args):
        return OpSpec(
            op="identity",
            is_reduction=False,
            iteration_space={},
            args=list(args),
            op_info={},
        )

    def test_label_crashed_on_exception(self):
        self.assertEqual(_label_for(RuntimeError("boom"), [], []), CRASHED)

    def test_label_no_spyre_when_empty(self):
        self.assertEqual(_label_for(None, [], []), NO_SPYRE_OP)

    def test_label_direct_op_spec(self):
        spec = self._spec(self._arg(True), self._arg(False))
        self.assertEqual(_label_for(None, [spec], [spec]), DIRECT_OP_SPEC)

    def test_label_gather_on_indirect_input(self):
        spec = self._spec(self._arg(True, indirect=True), self._arg(False))
        self.assertEqual(_label_for(None, [spec], [spec]), GATHER_OP_SPEC)

    def test_label_scatter_on_indirect_output(self):
        spec = self._spec(self._arg(True), self._arg(False, indirect=True))
        self.assertEqual(_label_for(None, [spec], [spec]), SCATTER_OP_SPEC)

    def test_label_unimplemented(self):
        u = UnimplementedOp("sin")
        self.assertEqual(_label_for(None, [], [u]), UNIMPLEMENTED)

    def test_label_scatter_takes_priority_over_gather(self):
        """An op with IndirectAccess on both an input and an output classifies
        as scatter (output is checked first)."""
        spec = self._spec(
            self._arg(True, indirect=True), self._arg(False, indirect=True)
        )
        self.assertEqual(_label_for(None, [spec], [spec]), SCATTER_OP_SPEC)

    def test_label_unimplemented_takes_priority_over_direct(self):
        """A direct op spec alongside an UnimplementedOp classifies as
        unimplemented."""
        spec = self._spec(self._arg(True), self._arg(False))
        entries = [spec, UnimplementedOp("sin")]
        self.assertEqual(_label_for(None, [spec], entries), UNIMPLEMENTED)

    def test_label_crashed_wins_even_with_op_specs(self):
        """If an exception is present, the outcome is CRASHED regardless of any
        op specs captured before the failure."""
        spec = self._spec(self._arg(True, indirect=True), self._arg(False))
        self.assertEqual(_label_for(ValueError("x"), [spec], [spec]), CRASHED)


# ===========================================================================
# SDSC label / allocation-name parsing
# ===========================================================================
class TestLabelParsing(IndirectAccessTestCase):
    def test_labeled_ds_index(self):
        self.assertEqual(labeled_ds_index("Tensor0-idx0"), 0)
        self.assertEqual(labeled_ds_index("Tensor12-idx12"), 12)

    def test_labeled_ds_index_rejects_garbage(self):
        for bad in ("Tensor3", "tensor3-idx3", "Tensor3-idx", "allocate-Tensor3_hbm"):
            with self.assertRaises(ValueError):
                labeled_ds_index(bad)

    def test_alloc_node_index(self):
        self.assertEqual(alloc_node_index("allocate-Tensor0_hbm"), 0)
        self.assertEqual(alloc_node_index("allocate-Tensor7_lx"), 7)

    def test_alloc_node_index_rejects_garbage(self):
        for bad in ("Tensor3-idx3", "allocate-Tensor_hbm", "allocateTensor3_hbm"):
            with self.assertRaises(ValueError):
                alloc_node_index(bad)


# ===========================================================================
# assert_indirect_sdsc_fields on synthetic SDSC bundles (no device, no compile)
# ===========================================================================
def _alloc(idx, *, indirect="no_indirection", related=None, component="hbm"):
    """A synthetic scheduleTree_ allocate node."""
    node = {
        "nodeType_": "allocate",
        "name_": f"allocate-Tensor{idx}_{component}",
        "ldsIdx_": idx,
        "component_": component,
        "indirectAllocType_": indirect,
    }
    if related is not None:
        node["relatedIndirectAccessAlloc_"] = related
    if indirect == "index_tensor":
        node["indexTensorType_"] = "index"
    return node


def _lds(idx, *, hbm_only=False, word=2):
    """A synthetic labeledDs_ entry."""
    mem = (
        {"hbm": {"isPresent": 1}}
        if hbm_only
        else {"hbm": {"isPresent": 1}, "lx": {"isPresent": 1}}
    )
    return {
        "ldsIdx_": idx,
        "dsName_": f"Tensor{idx}",
        "memOrg_": mem,
        "wordLength": word,
        "dataFormat_": "SEN169_FP16",
    }


def _body(sched, labeled, *, inputs, output, indices):
    """Assemble a synthetic op body (the inner dsc value)."""
    compute = {
        "opFuncName": "op",
        "inputLabeledDs": [f"Tensor{i}-idx{i}" for i in inputs],
        "outputLabeledDs": [f"Tensor{output}-idx{output}"],
    }
    if indices:
        compute["indirectAccessIndexLabeledDs"] = [f"Tensor{i}-idx{i}" for i in indices]
    return {
        "N_": {"name_": "n"},
        "scheduleTree_": sched,
        "labeledDs_": labeled,
        "computeOp_": [compute],
    }


def _bundle(body, opfunc="op", fname="sdsc_0.json"):
    """Wrap an op body in the full sdsc-json envelope iter_sdsc_op_bodies expects."""
    return {fname: {f"0_{opfunc}": {"dscs_": [{opfunc: body}]}}}


def _gather_bundle():
    """index=Tensor0, value-input=Tensor1, output=Tensor2 (value is an input)."""
    sched = [
        _alloc(0, indirect="index_tensor", related="allocate-Tensor1_hbm"),
        _alloc(1, indirect="value_tensor", related="allocate-Tensor0_hbm"),
        _alloc(2),
    ]
    labeled = [_lds(0, hbm_only=True, word=4), _lds(1), _lds(2)]
    return _bundle(_body(sched, labeled, inputs=[1], output=2, indices=[0]))


def _scatter_bundle():
    """index=Tensor0, src-input=Tensor1, output=Tensor2 (output is the value)."""
    sched = [
        _alloc(0, indirect="index_tensor", related="allocate-Tensor2_hbm"),
        _alloc(1),
        _alloc(2, indirect="value_tensor", related="allocate-Tensor0_hbm"),
    ]
    labeled = [_lds(0, hbm_only=True, word=4), _lds(1), _lds(2)]
    return _bundle(_body(sched, labeled, inputs=[1], output=2, indices=[0]))


class TestIndirectSdscValidator(IndirectAccessTestCase):
    """Drive assert_indirect_sdsc_fields with hand-built bundles, so the
    validator's own logic is covered without a device or compiler."""

    def test_valid_gather_passes(self):
        self.assert_indirect_sdsc_fields(_gather_bundle(), "gather")

    def test_valid_scatter_passes(self):
        self.assert_indirect_sdsc_fields(_scatter_bundle(), "scatter")

    def test_gather_validated_as_scatter_fails(self):
        """A gather (value tensor is an input) must not pass scatter validation."""
        with self.assertRaises(AssertionError):
            self.assert_indirect_sdsc_fields(_gather_bundle(), "scatter")

    def test_scatter_validated_as_gather_fails(self):
        with self.assertRaises(AssertionError):
            self.assert_indirect_sdsc_fields(_scatter_bundle(), "gather")

    def test_empty_bundle_fails(self):
        with self.assertRaises(AssertionError):
            self.assert_indirect_sdsc_fields({}, "gather")

    def test_unknown_kind_fails(self):
        with self.assertRaises(AssertionError):
            self.assert_indirect_sdsc_fields(_gather_bundle(), "sideways")

    def test_index_also_listed_as_input_fails(self):
        sched = [
            _alloc(0, indirect="index_tensor", related="allocate-Tensor1_hbm"),
            _alloc(1, indirect="value_tensor", related="allocate-Tensor0_hbm"),
            _alloc(2),
        ]
        labeled = [_lds(0, hbm_only=True, word=4), _lds(1), _lds(2)]
        # Index Tensor0 wrongly appears in inputLabeledDs.
        bundle = _bundle(_body(sched, labeled, inputs=[0, 1], output=2, indices=[0]))
        with self.assertRaises(AssertionError):
            self.assert_indirect_sdsc_fields(bundle, "gather")

    def test_missing_indirect_index_labeled_ds_fails(self):
        sched = [
            _alloc(0, indirect="index_tensor", related="allocate-Tensor1_hbm"),
            _alloc(1, indirect="value_tensor", related="allocate-Tensor0_hbm"),
            _alloc(2),
        ]
        labeled = [_lds(0, hbm_only=True, word=4), _lds(1), _lds(2)]
        bundle = _bundle(_body(sched, labeled, inputs=[1], output=2, indices=[]))
        with self.assertRaises(AssertionError):
            self.assert_indirect_sdsc_fields(bundle, "gather")

    def test_dangling_cross_link_fails(self):
        sched = [
            # Index points at a value tensor that does not exist.
            _alloc(0, indirect="index_tensor", related="allocate-Tensor9_hbm"),
            _alloc(1, indirect="value_tensor", related="allocate-Tensor0_hbm"),
            _alloc(2),
        ]
        labeled = [_lds(0, hbm_only=True, word=4), _lds(1), _lds(2)]
        bundle = _bundle(_body(sched, labeled, inputs=[1], output=2, indices=[0]))
        with self.assertRaises(AssertionError):
            self.assert_indirect_sdsc_fields(bundle, "gather")

    def test_index_in_lx_fails(self):
        sched = [
            _alloc(0, indirect="index_tensor", related="allocate-Tensor1_hbm"),
            _alloc(1, indirect="value_tensor", related="allocate-Tensor0_hbm"),
            _alloc(2),
        ]
        # Index labeledDs wrongly allows LX.
        labeled = [_lds(0, hbm_only=False, word=4), _lds(1), _lds(2)]
        bundle = _bundle(_body(sched, labeled, inputs=[1], output=2, indices=[0]))
        with self.assertRaises(AssertionError):
            self.assert_indirect_sdsc_fields(bundle, "gather")

    def test_non_integer_index_word_length_fails(self):
        sched = [
            _alloc(0, indirect="index_tensor", related="allocate-Tensor1_hbm"),
            _alloc(1, indirect="value_tensor", related="allocate-Tensor0_hbm"),
            _alloc(2),
        ]
        # Index with fp16 word length (2) is invalid for an index tensor.
        labeled = [_lds(0, hbm_only=True, word=2), _lds(1), _lds(2)]
        bundle = _bundle(_body(sched, labeled, inputs=[1], output=2, indices=[0]))
        with self.assertRaises(AssertionError):
            self.assert_indirect_sdsc_fields(bundle, "gather")

    def test_value_without_index_fails(self):
        sched = [
            _alloc(0),
            _alloc(1, indirect="value_tensor", related="allocate-Tensor0_hbm"),
            _alloc(2),
        ]
        labeled = [_lds(0), _lds(1), _lds(2)]
        bundle = _bundle(_body(sched, labeled, inputs=[1], output=2, indices=[]))
        with self.assertRaises(AssertionError):
            self.assert_indirect_sdsc_fields(bundle, "gather")

    def test_auxiliary_direct_op_is_tolerated(self):
        """A direct op sharing the bundle is skipped, not failed, as long as the
        bundle still contains a valid indirect op somewhere."""
        direct = _bundle(
            _body(
                [_alloc(0), _alloc(1)],
                [_lds(0), _lds(1)],
                inputs=[0],
                output=1,
                indices=[],
            ),
            opfunc="exp",
            fname="sdsc_1.json",
        )
        # Merge a direct op file with the gather op file (distinct filenames).
        merged = {**direct, **_gather_bundle()}
        self.assert_indirect_sdsc_fields(merged, "gather")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
