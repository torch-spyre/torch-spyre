# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Smoke tests for the OpSpec lab.

A tripwire, not a specification: the lab is a debugging tool, so what matters is
that it still works when someone reaches for it.  Everything here is pure Python
-- the device paths need hardware and are not covered.
"""

import ast
import builtins
import inspect
import os
import sys
import types
from unittest.mock import patch

import pytest
from sympy import sympify

import torch

from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.core_mapping import derive_operation_mapping
from torch_spyre._inductor.op_spec import (
    DebugHandle,
    LoopSpec,
    OpSpec,
    SourceLoc,
    TensorArg,
    UnimplementedOp,
)
from torch_spyre.execution.async_compile import SpyreAsyncCompile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import capture  # noqa: E402
import explain  # noqa: E402
import runner  # noqa: E402


def _add_op(debug_handle=None):
    """A 128x256 fp16 elementwise add -- the shape verified on hardware."""
    iteration_space = {
        sympify("c0"): (sympify("128"), 32),
        sympify("c1"): (sympify("256"), 1),
    }
    return OpSpec(
        op="add",
        is_reduction=False,
        iteration_space=iteration_space,
        core_id_to_work_slice=derive_operation_mapping(iteration_space),
        op_info={},
        debug_handle=debug_handle,
        args=[
            TensorArg(
                is_input=(i < 2),
                arg_index=i,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[4, 128, 64],
                device_coordinates=[
                    sympify(c) for c in ("floor(c1/64)", "c0", "Mod(c1, 64)")
                ],
                allocation={"hbm": i},
            )
            for i in range(3)
        ],
    )


def _pool_op():
    """An add whose output spilled to the HBM pool, leaving two kernel args.

    ``arg_index=-1`` is what makes this a *real* pool spec rather than one that
    only looks like it: ``generate_bundle`` keys its ``device_mem_allocate`` on
    the pooled tensor being an intermediate, so with a non-negative arg_index it
    emits no pool at all and accepts ``pool_size=0`` without complaint.
    """
    op = _add_op()
    op.args[2].allocation = {"hbm_pool": 0}
    op.args[2].arg_index = -1
    return op


class _FakeLayout:
    """The subset of SpyreTensorLayout that _record_args reads."""

    def __init__(self):
        self.device_size = [4, 128, 64]
        self.stride_map = [64, 256, 1]
        self.device_dtype = DataFormats.SEN169_FP16
        self.element_arrangement = ElementArrangement.STANDARD


class _FakeTensor:
    """A stand-in for a device tensor, counting the .cpu() calls it receives."""

    def __init__(self, shape=(128, 256), dtype=torch.float16, numel=32768):
        self.shape = shape
        self.dtype = dtype
        self._numel = numel
        self.cpu_calls = 0

    def numel(self):
        return self._numel

    def device_tensor_layout(self):
        return _FakeLayout()

    def cpu(self):
        self.cpu_calls += 1
        return self


class _FakeRunner:
    """A kernel runner that records its launches instead of performing them."""

    def __init__(self):
        self.code_dir = "/tmp/fake_code_dir"
        self.launches = []

    def run(self, *args, **kwargs):
        self.launches.append(args)


def _fake_sdsc(self, kernel_name, specs, pool_size=0):
    """Stand in for sdsc() with its real signature, so the spy is called as it is."""
    return _FakeRunner()


def _arg_record(dtype=torch.float16):
    return capture.ArgRecord(
        shape=(128, 256),
        dtype=dtype,
        device_size=[4, 128, 64],
        stride_map=[64, 256, 1],
        device_dtype_name="SEN169_FP16",
        element_arrangement_name="STANDARD",
    )


def _record(name="sdsc_add_0", pool_size=0):
    return capture.KernelRecord(
        name=name,
        specs=[_add_op()],
        index=0,
        sencores=32,
        bundle_symbolic_args=True,
        pool_size=pool_size,
        # A distinct record per arg: sharing one object made a test that retyped
        # arg0 silently retype all three.
        args=[_arg_record() for _ in range(3)],
        observed_run=True,
    )


def test_render_decodes_every_section():
    """The whole explain view, in one assertion set.

    Sections are individually isolated, so one that raises degrades to a ``!!``
    line rather than killing the render -- which is how #3567 deleting
    ``SDSCArgs.per_tile_fixed`` left the args table dead with the suite green.
    Asserting no section reports failure turns that resilience into a tripwire;
    asserting a table *row* is what catches the table vanishing, since the
    title's "3 kernel args" already contains the word "args".
    """
    out = explain.render([_add_op()], kernel_name="sdsc_add_0")
    assert "section failed" not in out
    assert "parse_op_spec failed" not in out
    assert "None" not in out  # a bare None here is always a formatting bug

    row = next(ln for ln in out.splitlines() if ln.strip().startswith("0 input"))
    assert "hbm @ 0" in row
    assert "L0" in row  # layout class, from the resolved view
    assert "mb=" in row  # scales and strides, in renamed dim labels

    # Sections that return [] on an internal miss rather than raising, so the
    # "section failed" assertion above cannot see them go quiet.
    assert "elems/stick at SEN169_FP16" in out
    assert "stick dim c1 = 256 -> ceil(256/64)" in out
    assert "c0 -> mb" in out


def test_render_discloses_that_a_layout_label_is_not_a_role():
    """Trap 2 from explain.py's docstring, and the reason the file exists.

    All three args of a flat add share one layout, so its raw label is
    LAYOUT_LABELS[0] -- the string "OUTPUT" -- on two inputs. Leaving that
    unexplained is what sends readers hunting for a nonexistent bug.
    """
    out = explain.render([_add_op()], kernel_name="k")
    assert 'raw SDSC label "OUTPUT"' in out
    assert "NOT an input/output role" in out


def test_render_reports_the_pool_it_was_given():
    out = explain.render([_add_op()], kernel_name="k", pool_size=32768)
    assert "32768-byte pool" in out  # title
    # The bundle emits its own device_mem_allocate, so the pool is not an arg.
    assert "32768 bytes -- allocated by the bundle, not an arg" in out
    assert "no pool" in explain.render([_add_op()], kernel_name="k")


def test_render_names_an_unimplemented_op_instead_of_skipping_it():
    """An UnimplementedOp reaches sdsc(), so the view has to account for it."""
    out = explain.render([UnimplementedOp(op="sin"), _add_op()], kernel_name="k")
    assert "UNIMPLEMENTED  sin" in out
    assert "OP 1/1  add" in out  # the count covers OpSpecs only


def test_scale_decode_explains_a_stick_dim_reduction():
    """Where a reduction's meaning lives, per example 02 of the guide.

    Driven off a stub resolved view rather than a real reduction OpSpec: the
    mapping from negative scale to prose is the whole content of this section.
    """
    args = [
        types.SimpleNamespace(scales={sympify("out"): -2, sympify("mb"): 1}),
        types.SimpleNamespace(scales={sympify("kj"): -1}),
    ]
    lines = explain._scale_decode(types.SimpleNamespace(args=args))

    assert any("out=-2 -> reduced along the stick dim" in ln for ln in lines)
    assert any("kj=-1 -> reduced dimension" in ln for ln in lines)
    assert not any("mb=" in ln for ln in lines)  # positive scales say nothing


def test_table_header_does_not_repeat_a_spilled_column_name():
    """A spilled cell names its column, except on the header row where it is one.

    Every cell is padded to the shared widths, so the header line is exactly as
    wide as the widest row and spills whenever any row does.
    """
    rows = [["0", "input", "hbm_pool @ 1048576", "L0", "mb=1 in=1 out=1", "x" * 30]]
    headers = ["#", "role", "alloc", "layout", "scales", "strides"]
    lines = explain._table(rows, headers, "    ")

    assert not any("strides strides" in ln for ln in lines)
    assert any("strides " + "x" * 30 in ln for ln in lines)  # rows still self-label


def test_comment_block_is_valid_python():
    """capture.py splices this into every emitted file, so it must parse."""
    block = explain.render_comment_block([_add_op()], kernel_name="k")
    assert all(ln.startswith("#") for ln in block.splitlines())
    assert block.encode("ascii")
    ast.parse(block + "\nx = 1\n")


def test_hostile_names_stay_within_width():
    """The 78-column guarantee, on the inputs that have actually broken it.

    A short flat add exercises none of it.  Three overflows shipped unnoticed
    before: a tiled kernel's ``dim labels`` reached 246 columns, a custom op's
    qualified name overflowed ``origin``, and a fused kernel name overflowed the
    title rule.  Nesting was the multiplier -- sections did not account for the
    columns the enclosing loops had already spent.

    A token longer than the budget cannot be wrapped and must not be truncated:
    an identifier is either printed whole or it is a lie.  So those are the one
    accepted exception rather than a blanket pass.
    """
    long_kernel = (
        "sdsc_fused__scaled_mm_quantize_fp8_with_scale_quantize_weight_fp8_with_scale_0"
    )
    handle = DebugHandle(
        id=1,
        source=SourceLoc(file="07_fp8_scaled_mm.py", start_line=43),
        aten_op="spyre.quantize_fp8_with_scale.default",
        ir_chain=(),
    )
    op = _add_op(debug_handle=handle)
    op.tiled_symbols = [
        [sympify("_tile_adv_coarse_tile_read_copy_buf0_arg0_1_lvl0")],
        [sympify("_tile_adv_coarse_tile_read_copy_buf0_arg0_1_lvl1")],
    ]
    inner = LoopSpec(count=sympify("4"), body=[op])
    nested = LoopSpec(count=sympify("2"), body=[inner])

    out = explain.render([nested], kernel_name=long_kernel)
    overflows = [
        ln
        for ln in out.splitlines()
        if len(ln) > explain.WIDTH
        and max((len(w) for w in ln.split()), default=0) <= explain.WIDTH - 14
    ]
    assert not overflows, overflows
    assert long_kernel in out  # printed whole, never truncated
    # The tiled section wraps rather than truncating, and an identifier split
    # across lines no longer names anything.
    assert "_tile_adv_coarse_tile_read_copy_buf0_arg0_1_lvl1" in out


def test_emitted_script_executes_its_definitions():
    """The load-bearing test for capture.py.

    Module-level code includes the OpSpec literal and the LAYOUTS list, so this
    proves the spec printer round-trips and that the emitted imports cover every
    name the file uses -- an annotation needing a new import would otherwise
    break every future capture at *its* import time.  ``__main__`` does not fire
    under exec, so nothing is compiled or launched.
    """
    src = capture.emit_script(_record(pool_size=32768), "prog.py", 1, False)
    namespace: dict = {"__file__": "/tmp/captured/sdsc_add_0.py"}
    exec(compile(src, "<emitted>", "exec"), namespace)  # noqa: S102
    assert namespace["KERNEL_NAME"] == "sdsc_add_0"
    assert namespace["POOL_SIZE"] == 32768
    assert len(namespace["ops"]) == 1
    assert len(namespace["LAYOUTS"]) == 3
    # Keyed on the script, not the kernel: two graphs can share a fused name, and
    # a kernel-keyed .pt would be clobbered by the second capture.
    assert namespace["INPUTS_PT"] == "/tmp/captured/sdsc_add_0.inputs.pt"


def test_emitted_script_forwards_the_pool_size():
    """A pool kernel bundles only if pool_size reaches sdsc().

    ``generate_bundle`` asserts ``0 < pool_size`` whenever a pool symbol is
    present, so an emitted script that drops it fails in codegen -- no device
    needed to get it wrong.
    """
    src = capture.emit_script(_record(pool_size=32768), "prog.py", 1, False)
    assert "POOL_SIZE = 32768" in src
    assert "pool_size=POOL_SIZE" in src


def test_bundling_a_pool_spec_emits_the_pool_allocation(tmp_path):
    """Bundle a pool spec for real -- no device, no mock, no dxp_standalone.

    The unmocked call is the point: ``generate_bundle`` is where a dropped
    pool_size actually bites, and it asserts ``0 < pool_size`` only when it sees
    a pool symbol. So this fails on a lab that forgets to forward the size, where
    a mocked ``generate_bundle`` would happily record the call and pass.
    """
    with spyre_config.patch(bundle_symbolic_args=True):  # type: ignore[attr-defined]
        listing = runner.bundle_op_specs(
            "k", [_pool_op()], str(tmp_path), pool_size=32768
        )
        assert "bundle.mlir" in listing
        mlir = (tmp_path / "bundle.mlir").read_text()
        assert "sdscbundle.device_mem_allocate 32768 bytes" in mlir

        with pytest.raises(AssertionError, match="pool_size=0 out of range"):
            runner.bundle_op_specs("k", [_pool_op()], str(tmp_path / "dropped"))


def test_run_op_specs_forwards_the_pool_size():
    """The replay path, which is where a dropped pool_size would go unnoticed."""
    tensors = [torch.zeros(4, dtype=torch.float16) for _ in range(2)]
    fake = _FakeRunner()
    with (
        patch.object(runner, "to_device", lambda ts, layouts=None: list(ts)),
        patch.object(SpyreAsyncCompile, "sdsc", return_value=fake) as sdsc,
    ):
        runner.run_op_specs("k", [_pool_op()], tensors, pool_size=32768)

    assert sdsc.call_args.kwargs["pool_size"] == 32768
    # The pool is not an argument any more, so only the kernel args reach .run().
    assert len(fake.launches[0]) == 2


def test_runner_template_is_self_contained():
    """Every name the inlined run loop loads must resolve in a bare script.

    capture.py reads these helpers off the live functions, so the loop can never
    go stale -- but a helper that starts calling something left out of
    _TEMPLATE_FUNCS is a NameError in every captured script, not here.  Executing
    only proves the ``def`` lines are valid, so walk the bodies too.
    """
    src = runner.TEMPLATE_IMPORTS + "\n\n" + runner.runner_template()
    namespace: dict = {}
    exec(compile(src, "<template>", "exec"), namespace)  # noqa: S102
    for fn in runner._TEMPLATE_FUNCS:
        assert fn.__name__ in namespace

    tree = ast.parse(src)
    bound = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            bound |= {a.arg for a in node.args.args}
        elif isinstance(node, ast.alias):
            # Function-local imports bind too: main pulls in explain.render that
            # way so --explain can degrade without --stage run caring.
            bound.add((node.asname or node.name).split(".")[0])
    used = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    assert used - (set(namespace) | set(dir(builtins)) | bound) == set()


def test_write_kernel_disambiguates_script_and_pt_together(tmp_path):
    """Two graphs in one program can compile to the same fused kernel name."""
    first = capture.write_kernel(
        _record(name="dup"), str(tmp_path), "prog.py", 2, False, True
    )
    second = capture.write_kernel(
        _record(name="dup"), str(tmp_path), "prog.py", 2, False, True
    )
    assert os.path.basename(first) == "dup.py"
    assert os.path.basename(second) == "dup_1.py"
    for path in (first, second):
        assert os.path.exists(os.path.splitext(path)[0] + ".inputs.pt")


def test_spy_accepts_everything_sdsc_accepts():
    """The spy stands in for sdsc(), so it must take sdsc()'s parameters.

    #3707 added ``pool_size`` and the spy did not follow, which left every
    pool-using program uncapturable -- a TypeError raised through the Inductor
    backend -- while this suite stayed green, because every other test here
    patches sdsc itself and never exercises the real signature.
    """
    real = inspect.signature(SpyreAsyncCompile.sdsc)
    with capture.capture_kernels():
        spy = inspect.signature(SpyreAsyncCompile.sdsc)

    assert list(spy.parameters) == list(real.parameters)


def test_record_args_keeps_every_arg_for_a_pool_kernel():
    """Nothing is skipped even when the kernel spills to the pool.

    Since #3707 the bundle allocates its own pool via ``device_mem_allocate``, so
    no pool tensor is passed at launch and ``arg_index`` is the ``.run()``
    position for every arg. Popping one here would silently drop arg0 from the
    capture and record its ``numel()`` as a byte count.
    """
    rec = capture.KernelRecord(
        name="k", specs=[_pool_op()], index=0, sencores=32, bundle_symbolic_args=True
    )
    capture._record_args(rec, tuple(_FakeTensor() for _ in range(2)), save_inputs=True)

    assert rec.pool_size == 0  # only the sdsc() keyword sets this
    assert len(rec.args) == 2
    assert all(a.values is not None for a in rec.args)


def test_record_args_records_the_layout_of_every_arg():
    rec = capture.KernelRecord(
        name="k", specs=[_add_op()], index=0, sencores=32, bundle_symbolic_args=True
    )
    capture._record_args(rec, tuple(_FakeTensor() for _ in range(3)), save_inputs=False)

    assert len(rec.args) == 3
    assert rec.args[0].device_dtype_name == "SEN169_FP16"
    assert rec.args[0].element_arrangement_name == "STANDARD"
    assert all(a.values is None for a in rec.args)


def test_run_recorder_records_the_first_launch_only():
    """A graph invoked repeatedly must not have its recorded inputs overwritten."""
    rec = capture.KernelRecord(
        name="k", specs=[_add_op()], index=0, sencores=32, bundle_symbolic_args=True
    )
    inner = _FakeRunner()
    proxy = capture._RunRecorder(inner, rec, save_inputs=False)

    proxy.run(*[_FakeTensor() for _ in range(3)])
    proxy.run(*[_FakeTensor() for _ in range(3)])

    assert rec.observed_run
    assert len(rec.args) == 3  # not 6
    assert len(inner.launches) == 2  # both still reached the real runner
    assert proxy.code_dir == inner.code_dir  # unknown attributes forward


def _stub_sdsc_program(tmp_path, body):
    """Write a target program that compiles ``n`` kernels, then runs ``body``."""
    program = tmp_path / "prog.py"
    program.write_text(
        "from torch_spyre.execution.async_compile import SpyreAsyncCompile\n" + body
    )
    return program


def test_capture_writes_what_it_recorded_before_the_target_crashed(tmp_path, capsys):
    """A program that raises is this tool's normal input, not a reason to abort.

    The kernels compiled before the failure are already complete reproducers --
    and for a kernel holding an UnimplementedOp the launch raises every time, so
    aborting would make those uncapturable.
    """
    program = _stub_sdsc_program(
        tmp_path,
        "SpyreAsyncCompile().sdsc('sdsc_stub_0', [])\nraise RuntimeError('boom')\n",
    )
    out_dir = tmp_path / "captured"

    with patch.object(SpyreAsyncCompile, "sdsc", _fake_sdsc):
        status = capture.main([str(program), "--out", str(out_dir)])

    # Not 0: a caller has to be able to tell a partial capture from a whole one
    # without parsing stdout, since the scripts written look no different.
    assert status == capture.EXIT_PARTIAL
    assert (out_dir / "sdsc_stub_0.py").exists()
    # The traceback itself goes to stderr; the note that says why is on stdout.
    assert "raised RuntimeError" in capsys.readouterr().out


def test_capture_of_a_clean_program_exits_zero(tmp_path):
    """The other half of the exit-code contract, so EXIT_PARTIAL means something."""
    body = "SpyreAsyncCompile().sdsc('sdsc_ok_0', [])\n"
    program = _stub_sdsc_program(tmp_path, body)
    out_dir = tmp_path / "captured"

    with patch.object(SpyreAsyncCompile, "sdsc", _fake_sdsc):
        assert capture.main([str(program), "--out", str(out_dir)]) == 0

    assert (out_dir / "sdsc_ok_0.py").exists()


def test_capture_blames_the_crash_when_nothing_compiled(tmp_path):
    """Otherwise the "check the script calls torch.compile" hint misleads."""
    program = _stub_sdsc_program(tmp_path, "raise RuntimeError('boom')\n")

    with pytest.raises(SystemExit) as exc:
        capture.main([str(program), "--out", str(tmp_path / "captured")])

    assert "RuntimeError" in str(exc.value)
    assert "torch.compile" not in str(exc.value)


def test_emitted_script_pins_bundle_symbolic_args():
    """The spec's hbm allocations were baked under this flag.

    ``generate_bundle`` re-reads it at call time, so leaving the ambient
    BUNDLE_SYMBOLIC_ARGS to differ produces a bundle whose addresses do not match
    the spec -- silently, which is the one failure a bisection tool must not have.
    """
    rec = _record()
    rec.bundle_symbolic_args = False
    namespace: dict = {"__file__": "/tmp/captured/sdsc_add_0.py"}
    src = capture.emit_script(rec, "prog.py", 1, False)
    exec(compile(src, "<emitted>", "exec"), namespace)  # noqa: S102

    assert namespace["BUNDLE_SYMBOLIC_ARGS"] is False
    assert "bundle_symbolic_args=BUNDLE_SYMBOLIC_ARGS" in src


def test_build_tensors_synthesizes_fp8_args():
    """fp8 reports is_floating_point but torch.rand has no kernel for it.

    So the obvious ``torch.rand(shape, dtype=dtype)`` raises NotImplementedError
    on the fp8 dtypes and every captured fp8 kernel -- a scaled_mm, say -- is
    unreplayable. Exercised through the emitted script because that is where
    build_tensors lives.
    """
    rec = _record()
    rec.args[0].dtype = torch.float8_e4m3fn
    namespace: dict = {"__file__": "/tmp/captured/sdsc_fp8_0.py"}
    src = capture.emit_script(rec, "prog.py", 1, False)
    exec(compile(src, "<emitted>", "exec"), namespace)  # noqa: S102

    tensors = namespace["build_tensors"]()

    assert tensors[0].dtype == torch.float8_e4m3fn
    assert tuple(tensors[0].shape) == (128, 256)
    assert tensors[0].to(torch.float32).abs().sum() > 0  # not all zeros
    assert tensors[1].dtype == torch.float16  # the wider dtypes are untouched


def test_pin_bundle_symbolic_args_overrides_the_ambient_flag():
    with spyre_config.patch(bundle_symbolic_args=True):  # type: ignore[attr-defined]
        runner.pin_bundle_symbolic_args(False)
        assert spyre_config.bundle_symbolic_args is False


def test_pin_bundle_symbolic_args_ignores_an_unrecorded_value():
    """None means "not recorded" -- a hand-written PROGRAM dict -- not False."""
    with spyre_config.patch(bundle_symbolic_args=True):  # type: ignore[attr-defined]
        runner.pin_bundle_symbolic_args(None)
        assert spyre_config.bundle_symbolic_args is True


def test_dedup_ignores_provenance_only_differences():
    """An eager Spyre op is itself torch.compile'd, so running one both ways
    yields two identically-named kernels differing only in debug_handle."""
    handle = DebugHandle(
        id=7,
        source=SourceLoc(file="/x/prog.py", start_line=3),
        aten_op="aten.add.Tensor",
        ir_chain=(),
    )
    assert capture.dedup_key([_add_op()]) == capture.dedup_key(
        [_add_op(debug_handle=handle)]
    )
    other = _add_op()
    other.op = "mul"
    assert capture.dedup_key([_add_op()]) != capture.dedup_key([other])
