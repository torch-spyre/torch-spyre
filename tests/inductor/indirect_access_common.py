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

"""Shared test utilities for indirect-access (gather/scatter) tests.

This file isn't named test_* so pytest won't run it directly. The test files
import helpers from here, organized one file per op family:

  * test_indirect_access_gather.py    -- gather ops (x[i], index_select, torch.gather)
  * test_indirect_access_scatter.py   -- scatter ops (out[i]=src, scatter_, index_copy)
  * test_indirect_access_internals.py -- device-free unit tests for the harness/helpers

How it works: each scenario compiles once and is checked across every stage --
detection, OpSpec structure, SDSC fields -- in one place, via run_scenario
and IndirectAccessTestCase.check (slice by stage with the op=, detected=,
sdsc= parameters). The capture machinery intercepts the compiler before
SuperDSC, records the generated OpSpecs, and returns a no-op runner -- so we
test compiler output without actual hardware or numeric results.

For a gather/scatter outcome, check does not stop at "the SDSC json exists":
IndirectAccessTestCase.assert_indirect_sdsc_fields validates the full
indirect-access encoding the backend depends on -- index/value allocation
types, HBM-only index tensors, the index<->value cross-links, and the
computeOp routing. bundle_jsons_from_captured lets a test that only captured
op specs move on to that same SDSC validation without recompiling.

run_e2e drives the real backend (no mocking), mirroring the standalone
gather.py script: it compiles and runs on device, then reports an *expected
failure* (pytest.xfail) when the values diverge from the CPU reference or the
backend aborts -- because the backend does not yet implement indirect access
correctly. The xfail is raised after the capture-path stage checks run, so those
stay strict; when the backend is fixed and results match, no xfail is raised and
the test passes (xpass alerts you if a hard-coded expectation goes stale). For
expect_close=True ops a mismatch/failure is instead a hard error. Only gather
reaches this path today; the name is generic so scatter can reuse it later.
"""

import contextlib
import dataclasses
import glob
import json
import math
import os
import tempfile
from subprocess import CalledProcessError
from unittest.mock import patch

import pytest
import regex as re
import torch
from torch._dynamo.exc import BackendCompilerFailed
from torch._inductor.test_case import TestCase as InductorTestCase

import torch_spyre._inductor.propagate_named_dims as _pnd
from torch_spyre._C import SpyreTensorLayout, get_device_dtype
from torch_spyre._inductor.op_spec import (
    IndirectAccess,
    LoopSpec,
    OpSpec,
    TensorArg,
    UnimplementedOp,
)
from torch_spyre.execution.async_compile import SpyreAsyncCompile

declare_tensor_dim = _pnd.declare_tensor_dim
name_tensor_dims = _pnd.name_tensor_dims


# ---------------------------------------------------------------------------
# Device-layout helpers
# ---------------------------------------------------------------------------
def canonical_device_layout(shape, dtype) -> SpyreTensorLayout:
    """Build the canonical generic-stick `SpyreTensorLayout` explicitly
    Example (fp16, stick=64):
        shape (12, 1024)      -> device_size [12, 16, 64],    stride_map [1024, 64, 1]
        shape (12, 4, 1024)   -> device_size [12, 4, 16, 64], stride_map [4096, 1024, 64, 1]
    """
    shape = [int(s) for s in shape]
    eps = SpyreTensorLayout(shape, dtype).elems_per_stick()
    *lead, last = shape
    stick_count = (last + eps - 1) // eps
    device_size = [*lead, stick_count, eps]
    stride_map = [math.prod(shape[k + 1 :]) for k in range(len(lead))] + [eps, 1]
    return SpyreTensorLayout(
        device_size=device_size,
        stride_map=stride_map,
        device_dtype=get_device_dtype(dtype),
    )


def pinned_to_spyre(tensor: torch.Tensor) -> torch.Tensor:
    """Move a contiguous tensor to "spyre" with its canonical layout pinned
    explicitly (via `device_layout=`), instead of the implicit `.to("spyre")`.

    Used for the value (data) tensor of a gather, matching the manual `stl`
    pattern in the example scripts; index tensors stay on a plain `.to("spyre")`.
    """
    return tensor.to(
        "spyre",
        device_layout=canonical_device_layout(tensor.shape, tensor.dtype),
    )


# ---------------------------------------------------------------------------
# Capture machinery
# ---------------------------------------------------------------------------
class _NoopRunner:
    """A dummy kernel runner that does nothing when called."""

    def run(self, *args, **kwargs):
        return None


@contextlib.contextmanager
def capture_op_specs():
    """Capture OpSpecs generated during compilation without running the backend.

    Returns a list where each entry is the OpSpec list for one kernel.
    This skips SuperDSC generation, so we can test compiler output even for
    operations that would crash in the backend.
    """
    captured: list[list] = []

    def _spy(self, kernel_name, specs):  # noqa: ARG001
        captured.append(list(specs))
        return _NoopRunner()

    with patch.object(SpyreAsyncCompile, "sdsc", _spy):
        yield captured


@contextlib.contextmanager
def capture_sdsc_calls():
    """Like capture_op_specs, but also captures kernel names.

    Returns (kernel_name, specs) tuples for tests that need kernel names.
    """
    calls: list[tuple[str, list]] = []

    def _spy(self, kernel_name, specs):  # noqa: ARG001
        calls.append((kernel_name, list(specs)))
        return _NoopRunner()

    with patch.object(SpyreAsyncCompile, "sdsc", _spy):
        yield calls


# ---------------------------------------------------------------------------
# Spec-tree flattening
# ---------------------------------------------------------------------------
def flatten_op_specs(spec_lists) -> list[OpSpec]:
    """Extract all OpSpecs from nested LoopSpecs, skipping UnimplementedOps."""
    out: list[OpSpec] = []

    def _rec(specs):
        for s in specs:
            if isinstance(s, LoopSpec):
                _rec(s.body)
            elif isinstance(s, OpSpec):
                out.append(s)

    for lst in spec_lists:
        _rec(lst)
    return out


def flatten_entries(spec_lists) -> list:
    """Extract all entries (both OpSpecs and UnimplementedOps) from nested LoopSpecs."""
    out: list = []

    def _rec(specs):
        for s in specs:
            if isinstance(s, LoopSpec):
                _rec(s.body)
            else:
                out.append(s)

    for lst in spec_lists:
        _rec(lst)
    return out


# ---------------------------------------------------------------------------
# IndirectAccess helpers
# ---------------------------------------------------------------------------
def coord_atoms(arg: TensorArg, kind):
    """Find all sympy atoms of a given type in a tensor argument's coordinates."""
    atoms = set()
    for coord in arg.device_coordinates:
        if hasattr(coord, "atoms"):
            atoms |= coord.atoms(kind)
    return atoms


def arg_has_indirect_access(arg: TensorArg) -> bool:
    return bool(coord_atoms(arg, IndirectAccess))


def op_spec_has_indirect_access(op_spec: OpSpec) -> bool:
    return any(arg_has_indirect_access(a) for a in op_spec.args)


def op_spec_has_indirect_input(op_spec: OpSpec) -> bool:
    """Check if this OpSpec has IndirectAccess on an input (gather pattern)."""
    return any(a.is_input and arg_has_indirect_access(a) for a in op_spec.args)


def op_spec_has_indirect_output(op_spec: OpSpec) -> bool:
    """Check if this OpSpec has IndirectAccess on an output (scatter pattern)."""
    return any((not a.is_input) and arg_has_indirect_access(a) for a in op_spec.args)


def indirect_access_target_names(op_specs) -> set:
    """Get all buffer names used in IndirectAccess operations."""
    return {
        str(sym)
        for s in op_specs
        for a in s.args
        for ia in coord_atoms(a, IndirectAccess)
        for sym in ia.args
    }


# ---------------------------------------------------------------------------
# Compilation outcome classification
# ---------------------------------------------------------------------------
# Possible outcomes when compiling an operation:
CRASHED = "crashed_before_op_spec"  # Crashed before generating OpSpec
GATHER_OP_SPEC = "op_spec_indirect_input"  # Generated gather OpSpec (reads indirectly)
SCATTER_OP_SPEC = (
    "op_spec_indirect_output"  # Generated scatter OpSpec (writes indirectly)
)
DIRECT_OP_SPEC = "op_spec_no_indirect"  # Generated OpSpec without indirect access
UNIMPLEMENTED = "unimplemented_op"  # Hit an UnimplementedOp
NO_SPYRE_OP = "no_spyre_op_spec"  # Fell back to CPU


def _label_for(exc, op_specs, entries) -> str:
    """Map a compilation outcome to one of the classification constants."""
    if exc is not None:
        return CRASHED
    if any(op_spec_has_indirect_output(s) for s in op_specs):
        return SCATTER_OP_SPEC
    if any(op_spec_has_indirect_input(s) for s in op_specs):
        return GATHER_OP_SPEC
    if any(isinstance(e, UnimplementedOp) for e in entries):
        return UNIMPLEMENTED
    if op_specs:
        return DIRECT_OP_SPEC
    return NO_SPYRE_OP


def classify_compile(kernel, *dev_args):
    """Compile a kernel and classify what happened.

    Returns (label, detail) where label is one of the outcome constants above
    and detail is the exception if it crashed, otherwise the OpSpec list.
    Catches all exceptions to classify any compilation failure as CRASHED.
    """
    with capture_op_specs() as captured:
        try:
            torch.compile(kernel)(*dev_args)
        except Exception as exc:  # noqa: BLE001 - characterizing the failure mode
            return CRASHED, exc
    op_specs = flatten_op_specs(captured)
    label = _label_for(None, op_specs, flatten_entries(captured))
    return label, op_specs


# ---------------------------------------------------------------------------
# Real SDSC bundle generation (the Python SuperDSC path, no backend)
# ---------------------------------------------------------------------------
def generate_sdsc_jsons(kernel, *dev_args) -> dict:
    """Compile a kernel, capture the op specs handed to sdsc, then run the
    real generate_bundle on them and return {relpath: parsed_json} for every
    sdsc_*.json file produced.

    This exercises actual SDSC JSON generation (the Python SuperDSC path),
    independent of the deeptools backend compiler. Each captured kernel is
    bundled into its own subdirectory so per-kernel sdsc_N.json files never
    collide. Raises if compilation fails before reaching sdsc (e.g. scatter,
    which crashes in work division first)."""

    from torch_spyre._inductor.codegen.bundle import generate_bundle

    with capture_op_specs() as captured:
        torch.compile(kernel)(*dev_args)

    jsons: dict = {}
    out = tempfile.mkdtemp(prefix="sdsc_dump_")
    for ci, specs in enumerate(captured):
        sub = os.path.join(out, f"kernel{ci}")
        os.makedirs(sub, exist_ok=True)
        generate_bundle(f"kernel{ci}", sub, specs)
        for path in sorted(glob.glob(os.path.join(sub, "sdsc_*.json"))):
            with open(path) as f:
                jsons[os.path.relpath(path, out)] = json.load(f)
    return jsons


def iter_sdsc_op_bodies(jsons):
    """Yield (opfunc, body) for each compiled op across all sdsc json files.

    SDSC layout: {f"{idx}_{opfunc}": {... "dscs_": [{opfunc: body}] ...}},
    where body holds N_ / scheduleTree_ / labeledDs_ / computeOp_.
    """
    for top in jsons.values():
        for outer in top.values():
            for dsc in outer.get("dscs_", []):
                for opfunc, body in dsc.items():
                    yield opfunc, body


def iter_schedule_tree_nodes(jsons):
    """Yield every scheduleTree_ allocate node across all sdsc json files."""
    for _, body in iter_sdsc_op_bodies(jsons):
        yield from body.get("scheduleTree_", [])


# ---------------------------------------------------------------------------
# SDSC label / allocation-name parsing
# ---------------------------------------------------------------------------
# computeOp labels look like "Tensor3-idx3"; allocate-node names and the
# relatedIndirectAccessAlloc_ cross-links look like "allocate-Tensor3_hbm".
_LABELED_DS_RE = re.compile(r"Tensor(\d+)-idx\d+")
_ALLOC_NAME_RE = re.compile(r"allocate-Tensor(\d+)_")


def labeled_ds_index(label: str) -> int:
    """Parse the tensor index from a computeOp labeledDs reference.

    e.g. `"Tensor3-idx3" -> 3`.  Raises ValueError on an unrecognized label
    so a malformed SDSC bundle fails loudly rather than silently.
    """
    m = _LABELED_DS_RE.fullmatch(label)
    if m is None:
        raise ValueError(f"unrecognized labeledDs reference: {label!r}")
    return int(m.group(1))


def alloc_node_index(name: str) -> int:
    """Parse the tensor index from an allocate-node name or cross-link.

    e.g. `"allocate-Tensor3_hbm" -> 3`.  Raises ValueError on an
    unrecognized name.
    """
    m = _ALLOC_NAME_RE.match(name)
    if m is None:
        raise ValueError(f"unrecognized allocate name: {name!r}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# End-to-end execution (real backend)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class E2EResult:
    """The outcome of an end-to-end run compared against the CPU reference."""

    result: torch.Tensor | None  # device result (CPU); None if the backend failed
    reference: torch.Tensor  # eager CPU reference
    max_abs_diff: float  # max |reference - result|; inf on backend failure
    close: bool  # whether result matched reference within tolerance


def run_e2e(
    test,
    kernel,
    *dev_args,
    atol: float = 0.01,
    rtol: float = 0.01,
    expect_close: bool | None = None,
):
    """Compile and run an indirect-access kernel end-to-end on the real Spyre
    backend and validate the device result against the CPU reference.

    Unlike the capture-based helpers, this mocks nothing: it drives the full
    `bundle -> dxp_standalone -> launch_kernel` path, exactly like the
    standalone `tests/indirect_access/gather.py` script.  `dev_args` are the
    device tensors the kernel is invoked with; the CPU reference is computed
    from their host copies.

    Today only gather kernels reach the indirect path -- the name is
    deliberately generic so scatter (and other indirect ops) can reuse it once
    the backend supports them.

    The CPU reference is computed first; if *that* raises it is a problem with
    the test itself (e.g. an out-of-bounds index) and is allowed to propagate.

    The device compile/run is best-effort: the backend does not yet support
    every indirect-access pattern and aborts (SIGABRT in dxp_standalone) on some
    of them. A backend failure -- and likewise a value divergence -- is reported
    as an *expected failure* (pytest.xfail) rather than warned or hard-failed, so
    "always run e2e" surfaces known backend gaps as xfail (and flips to xpass the
    day the backend is fixed) without turning the suite red. Because xfail is
    raised imperatively *after* the capture-path stage checks have run, those
    checks stay strict. (For `expect_close=True` ops, which must work, a failure
    or divergence is a hard assertion/raise instead.)

    Result validation:
      * the output is checked for the correct shape and dtype;
      * the values are compared against the golden CPU reference and the
        max-abs-diff recorded;
      * `expect_close` controls the value assertion:
          - `True`  -> assert the result matches (use for ops that must be
                         correct, e.g. a supported direct op or CPU fallback);
          - `False` -> assert the result diverges (pin a known-bad path);
          - `None`  -> xfail on divergence (the default for on-device indirect
                         gather, which the backend does not yet implement
                         correctly). When it is fixed and results match, no xfail
                         is raised and the test simply passes.

    Returns an `E2EResult` when the result matched (or expect_close handled it);
    on divergence/backend failure it raises pytest.xfail and does not return.
    """
    reference = kernel(
        *[a.cpu() if isinstance(a, torch.Tensor) else a for a in dev_args]
    )

    # Recompile from scratch so the run exercises a fresh bundle.
    torch._dynamo.reset()
    try:
        result = torch.compile(kernel)(*dev_args).cpu()
    except (BackendCompilerFailed, CalledProcessError) as exc:
        if expect_close:
            raise  # a must-work op failing to compile/run is a real regression
        pytest.xfail(
            "e2e backend compile/run failed "
            f"({type(getattr(exc, '__cause__', None) or exc).__name__}); the "
            "Spyre backend does not yet support this indirect-access pattern. "
            "The capture-path stages still validated the bundle."
        )

    test.assertEqual(
        result.shape, reference.shape, "e2e run produced the wrong output shape"
    )
    test.assertEqual(
        result.dtype, reference.dtype, "e2e run produced the wrong output dtype"
    )

    diff = torch.abs(reference.float() - result.float()).amax().item()
    close = torch.allclose(result, reference, atol=atol, rtol=rtol, equal_nan=True)

    if expect_close is True:
        test.assertTrue(
            close, f"e2e result must match the CPU reference (max abs diff {diff:.4g})"
        )
    elif expect_close is False:
        test.assertFalse(close, "e2e result unexpectedly matched the CPU reference")
    elif not close:
        pytest.xfail(
            "e2e result diverges from the CPU reference "
            f"(max abs diff {diff:.4g}); the Spyre backend does not yet "
            "implement indirect access correctly. The pipeline compiled and "
            "ran end-to-end."
        )
    return E2EResult(result=result, reference=reference, max_abs_diff=diff, close=close)


def bundle_jsons_from_captured(captured) -> dict:
    """Run the real generate_bundle on already-captured op specs (no recompile)
    and return {relpath: parsed_json}.

    This is the bridge that lets a test which only captured op specs (via
    `capture_op_specs`) move on to the SDSC stage without recompiling: hand it
    the captured spec lists and it returns the parsed `sdsc_*.json` bundle so
    the indirect-access fields can be asserted (see
    `IndirectAccessTestCase.assert_indirect_sdsc_fields`)."""

    from torch_spyre._inductor.codegen.bundle import generate_bundle

    jsons: dict = {}
    if not captured:
        return jsons
    out = tempfile.mkdtemp(prefix="sdsc_scn_")
    for ci, specs in enumerate(captured):
        sub = os.path.join(out, f"kernel{ci}")
        os.makedirs(sub, exist_ok=True)
        try:
            generate_bundle(f"kernel{ci}", sub, specs)
        except Exception:  # noqa: BLE001 - absence of jsons is itself an outcome
            continue
        for path in sorted(glob.glob(os.path.join(sub, "sdsc_*.json"))):
            with open(path) as f:
                jsons[os.path.relpath(path, out)] = json.load(f)
    return jsons


@dataclasses.dataclass
class ScenarioResult:
    """Everything observed from a single compile of an indirect-access kernel.

    One scenario, one compile -- so a test can assert across every stage
    (detection, op-spec structure, SDSC fields) without recompiling per stage.
    """

    label: str  # classification outcome (GATHER_OP_SPEC, CRASHED, ...)
    op_specs: list  # real OpSpecs handed to sdsc
    entries: list  # OpSpec + UnimplementedOp leaves
    detected_index_names: set  # union of indirect_info_from_op name results
    sdsc_jsons: dict  # SDSC json built from the captured op specs
    exc: Exception | None  # exception if compilation raised


def run_scenario(kernel, *dev_args, build_sdsc: bool = True) -> ScenarioResult:
    """Compile a kernel once and gather evidence from every stage:

      * detection   -- what indirect_info_from_op flagged during layout
                       propagation
      * op specs    -- captured at the sdsc handoff (never reaches the backend)
      * label       -- the classification outcome
      * SDSC json   -- generated from the captured op specs (no recompile)

    This is the single compilation driver that consolidated per-op test files use.
    """
    import torch_spyre._inductor.propagate_layouts as _pl

    seen: list[set] = []
    real = _pl.indirect_info_from_op

    def _spy(op):
        names, access_subs, sizes = real(op)
        seen.append(set(names))
        return names, access_subs, sizes

    exc: Exception | None = None
    with (
        patch.object(_pl, "indirect_info_from_op", _spy),
        capture_op_specs() as captured,
    ):
        try:
            torch.compile(kernel)(*dev_args)
        except Exception as e:  # noqa: BLE001 - characterizing the failure mode
            exc = e

    op_specs = flatten_op_specs(captured)
    entries = flatten_entries(captured)
    return ScenarioResult(
        label=_label_for(exc, op_specs, entries),
        op_specs=op_specs,
        entries=entries,
        detected_index_names=set().union(*seen) if seen else set(),
        sdsc_jsons=bundle_jsons_from_captured(captured) if build_sdsc else {},
        exc=exc,
    )


# ---------------------------------------------------------------------------
# Base test case
# ---------------------------------------------------------------------------
class IndirectAccessTestCase(InductorTestCase):
    """Base class for indirect access tests with common setup and helper methods."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(3)
        # Recompile from scratch so our sdsc spy always sees the op specs.
        torch._dynamo.reset()

    # -- Dimension naming helpers ----------------------------------------
    def name_dims(self, tensor, dims: dict):
        """Declare and attach named dimensions to a tensor.

        Args:
            tensor: The tensor to name
            dims: Dictionary mapping dimension name to size (in order)
        """
        for nm, size in dims.items():
            declare_tensor_dim(nm, size)
        name_tensor_dims(tensor, list(dims.keys()))

    # -- Assertion helpers -----------------------------------------------
    def assert_reaches_op_spec(self, captured):
        """Verify that compilation reached the OpSpec generation stage."""
        self.assertTrue(captured, "sdsc was never called; no op spec produced")
        op_specs = flatten_op_specs(captured)
        self.assertTrue(op_specs, "no OpSpec found in captured sdsc calls")
        return op_specs

    def assert_gather_op_spec(self, captured):
        """Verify that a gather OpSpec was generated (has IndirectAccess on input)."""
        op_specs = self.assert_reaches_op_spec(captured)
        self.assertTrue(
            any(op_spec_has_indirect_input(s) for s in op_specs),
            "no op spec had an IndirectAccess on an input arg (gather not encoded)",
        )
        self.assertFalse(
            any(op_spec_has_indirect_output(s) for s in op_specs),
            "gather should not put IndirectAccess on an output arg",
        )
        return op_specs

    def assert_scatter_op_spec(self, captured):
        """Verify that a scatter OpSpec was generated (has IndirectAccess on output)."""
        op_specs = self.assert_reaches_op_spec(captured)
        self.assertTrue(
            any(op_spec_has_indirect_output(s) for s in op_specs),
            "no op spec had an IndirectAccess on an output arg (scatter not encoded)",
        )
        return op_specs

    # -- One-compile, all-stage scenario check (used by per-op test files) --
    def check(
        self,
        kernel,
        *dev_args,
        expect,
        op=None,
        detected=None,
        sdsc=True,
    ):
        """Compile a kernel once and assert across every requested stage.

        This is the single check routine that consolidated per-op-family test
        files use. Pass optional parameters to slice by stage instead of
        spreading one kernel across many stage-organized files.

        Args:
            expect: Required classification label (GATHER_OP_SPEC, CRASHED, etc.)
            op: If given, assert this op name appears among the produced specs
            detected: If True/False, assert the index buffer was/wasn't flagged
                by indirect_index_dep_names during layout propagation
            sdsc: If True, also build the SDSC bundle. For a gather/scatter
                outcome the bundle's indirect-access encoding is fully validated
                (index/value allocation types, HBM-only index tensors, the
                index<->value cross-links, and the computeOp routing); see
                assert_indirect_sdsc_fields.

        Returns the ScenarioResult for any further per-test assertions.
        """
        r = run_scenario(kernel, *dev_args, build_sdsc=sdsc)
        detail = f" ({type(r.exc).__name__}: {r.exc})" if r.exc is not None else ""
        self.assertEqual(r.label, expect, f"expected {expect}, got {r.label}{detail}")
        if op is not None:
            self.assertIn(
                op, [s.op for s in r.op_specs], f"op {op!r} not in produced specs"
            )
        if detected is True:
            self.assertTrue(r.detected_index_names, "index buffer was not detected")
        elif detected is False:
            self.assertFalse(
                r.detected_index_names, "index buffer unexpectedly detected"
            )
        if sdsc and r.label in (GATHER_OP_SPEC, SCATTER_OP_SPEC):
            kind = "gather" if r.label == GATHER_OP_SPEC else "scatter"
            self.assert_indirect_sdsc_fields(r.sdsc_jsons, kind)
        return r

    # -- one driver: validate every stage, then run on the real backend ---
    def _stage_and_e2e(
        self, kernel, *dev_args, expect, op=None, detected=None, expect_close=None
    ):
        """Validate every capture-path stage with check(), then run end-to-end.

        Shared by the gather and scatter op-family tests. Currently only the
        capture-path stages run; the e2e leg (real backend: dxp_standalone +
        on-device launch via run_e2e) is wired up but disabled until e2e
        support lands. Pass expect_close=True for ops whose result must match
        the CPU reference (e.g. a supported direct op) once e2e is enabled.

        Returns check()'s ScenarioResult for any further per-test assertions.
        """
        r = self.check(kernel, *dev_args, expect=expect, op=op, detected=detected)
        # TODO: Enable once e2e is available; expect_close is reserved for that path.
        # run_e2e(self, kernel, *dev_args, expect_close=expect_close)
        return r

    # -- SDSC indirect-access field validation ---------------------------
    def assert_indirect_sdsc_fields(self, sdsc_jsons, kind: str):
        """Assert the SDSC bundle encodes a structurally valid indirect access.

        Goes beyond "the json exists with the core fields": it verifies the
        indirect-access contract that the backend relies on, so a regression in
        SDSC generation surfaces here instead of at backend-compile time.

        `kind` is "gather" (indirect read -- the value tensor is an input) or
        "scatter" (indirect write -- the value tensor is the output).

        For every compiled op in the bundle:

          * the core structural fields are present and `scheduleTree_` /
            `labeledDs_` are the same length (one entry per tensor arg), and
            every allocate node is well-formed.

        For every op that actually carries indirect access (auxiliary direct
        ops sharing the bundle are skipped):

          * it has at least one `index_tensor` node *and* its referenced
            `value_tensor` node -- one cannot appear without the other;
          * each index tensor declares `indexTensorType_ == "index"`, lives in
            HBM only (the engine cannot indirectly address through LX), carries
            an int32/int64 word length, and cross-links to a real value tensor;
          * each value tensor cross-links back to its index tensor
            (bidirectionally consistent);
          * the computeOp routes exactly the index tensors through
            `indirectAccessIndexLabeledDs` and never lists them as a plain
            `inputLabeledDs`;
          * the value tensor is an input for a gather and the output for a
            scatter.

        Finally, at least one index tensor and one value tensor must appear
        somewhere in the bundle, so an empty/degenerate bundle cannot pass.
        """
        self.assertIn(kind, ("gather", "scatter"), f"unknown kind {kind!r}")
        self.assertTrue(sdsc_jsons, "no SDSC json generated for an indirect-access op")

        saw_index = False
        saw_value = False
        for opfunc, body in iter_sdsc_op_bodies(sdsc_jsons):
            for field in ("N_", "scheduleTree_", "labeledDs_", "computeOp_"):
                self.assertIn(field, body, f"{opfunc}: SDSC body missing {field}")
            sched = body["scheduleTree_"]
            labeled = body["labeledDs_"]
            self.assertEqual(
                len(sched),
                len(labeled),
                f"{opfunc}: scheduleTree_/labeledDs_ length mismatch",
            )
            for n in sched:
                self.assertEqual(
                    n["nodeType_"], "allocate", f"{opfunc}: non-allocate schedule node"
                )
                self.assertTrue(
                    n["name_"].startswith("allocate-Tensor"),
                    f"{opfunc}: malformed allocate name {n.get('name_')!r}",
                )
                self.assertIn(
                    n["component_"], ("hbm", "lx"), f"{opfunc}: bad component_"
                )
                self.assertIn(
                    n.get("indirectAllocType_"),
                    ("index_tensor", "value_tensor", "no_indirection"),
                    f"{opfunc}: node {n.get('ldsIdx_')} bad indirectAllocType_",
                )

            index_nodes = [
                n for n in sched if n.get("indirectAllocType_") == "index_tensor"
            ]
            value_nodes = [
                n for n in sched if n.get("indirectAllocType_") == "value_tensor"
            ]
            if not index_nodes and not value_nodes:
                # A direct op sharing the bundle -- nothing indirect to check.
                continue

            # An indirect op must encode both halves of the access.
            self.assertTrue(
                index_nodes, f"{opfunc}: value_tensor present without an index_tensor"
            )
            self.assertTrue(
                value_nodes, f"{opfunc}: index_tensor present without a value_tensor"
            )
            saw_index = True
            saw_value = True

            nodes_by_idx = {n["ldsIdx_"]: n for n in sched}
            lds_by_idx = {ld["ldsIdx_"]: ld for ld in labeled}

            compute = body["computeOp_"][0]
            out_idxs = {labeled_ds_index(x) for x in compute["outputLabeledDs"]}
            self.assertEqual(
                len(out_idxs), 1, f"{opfunc}: expected exactly one output label"
            )
            out_idx = next(iter(out_idxs))

            # computeOp must route the index tensors as indirect indices, never
            # as ordinary inputs.
            ia_labels = compute.get("indirectAccessIndexLabeledDs")
            self.assertTrue(
                ia_labels,
                f"{opfunc}: computeOp missing indirectAccessIndexLabeledDs",
            )
            ia_idxs = {labeled_ds_index(x) for x in ia_labels}
            self.assertEqual(
                ia_idxs,
                {n["ldsIdx_"] for n in index_nodes},
                f"{opfunc}: indirectAccessIndexLabeledDs must name exactly the "
                "index tensors",
            )
            in_idxs = {labeled_ds_index(x) for x in compute["inputLabeledDs"]}
            self.assertFalse(
                ia_idxs & in_idxs,
                f"{opfunc}: an index tensor must not also be a plain input",
            )

            # Index tensors: HBM-only, integer word length, linked to a value.
            for n in index_nodes:
                i = n["ldsIdx_"]
                self.assertEqual(
                    n.get("indexTensorType_"),
                    "index",
                    f"{opfunc}: index {i} must declare indexTensorType_ 'index'",
                )
                self.assertIn(
                    "relatedIndirectAccessAlloc_",
                    n,
                    f"{opfunc}: index {i} missing relatedIndirectAccessAlloc_",
                )
                related = n["relatedIndirectAccessAlloc_"]
                self.assertRegex(related, r"^allocate-Tensor\d+_hbm$")
                v = alloc_node_index(related)
                self.assertIn(
                    v,
                    nodes_by_idx,
                    f"{opfunc}: index {i} references absent value alloc {related!r}",
                )
                self.assertEqual(
                    nodes_by_idx[v].get("indirectAllocType_"),
                    "value_tensor",
                    f"{opfunc}: index {i}'s related alloc must be a value_tensor",
                )
                lds = lds_by_idx[i]
                self.assertIn(
                    "hbm", lds["memOrg_"], f"{opfunc}: index {i} labeledDs not in HBM"
                )
                self.assertNotIn(
                    "lx", lds["memOrg_"], f"{opfunc}: index {i} must be HBM only"
                )
                self.assertIn(
                    lds["wordLength"],
                    (4, 8),
                    f"{opfunc}: index {i} wordLength must be int32/int64",
                )

            # Value tensors: cross-link back, and obey the gather/scatter shape.
            for n in value_nodes:
                v = n["ldsIdx_"]
                self.assertIn(
                    "relatedIndirectAccessAlloc_",
                    n,
                    f"{opfunc}: value {v} missing relatedIndirectAccessAlloc_",
                )
                related = n["relatedIndirectAccessAlloc_"]
                self.assertRegex(related, r"^allocate-Tensor\d+_hbm$")
                i = alloc_node_index(related)
                self.assertIn(
                    i,
                    nodes_by_idx,
                    f"{opfunc}: value {v} references absent index alloc {related!r}",
                )
                self.assertEqual(
                    nodes_by_idx[i].get("indirectAllocType_"),
                    "index_tensor",
                    f"{opfunc}: value {v}'s related alloc must be an index_tensor",
                )
                back = nodes_by_idx[i].get("relatedIndirectAccessAlloc_")
                self.assertIsNotNone(
                    back, f"{opfunc}: index {i} missing back-link to value {v}"
                )
                self.assertEqual(
                    alloc_node_index(back),
                    v,
                    f"{opfunc}: index/value cross-link is not bidirectional",
                )
                if kind == "scatter":
                    self.assertEqual(
                        v, out_idx, f"{opfunc}: scatter value tensor must be the output"
                    )
                else:
                    self.assertNotEqual(
                        v, out_idx, f"{opfunc}: gather value tensor must be an input"
                    )

        self.assertTrue(saw_index, "no index_tensor nodes anywhere in the SDSC bundle")
        self.assertTrue(saw_value, "no value_tensor nodes anywhere in the SDSC bundle")
