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
import os
import sys

from sympy import sympify

import torch

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import (
    DebugHandle,
    LoopSpec,
    OpSpec,
    SourceLoc,
    TensorArg,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import capture  # noqa: E402
import explain  # noqa: E402
import runner  # noqa: E402


def _add_op(debug_handle=None):
    """A 128x256 fp16 elementwise add -- the shape verified on hardware."""
    return OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={
            sympify("c0"): (sympify("128"), 32),
            sympify("c1"): (sympify("256"), 1),
        },
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


def _record(name="sdsc_add_0", pool_bytes=0):
    arg = capture.ArgRecord(
        shape=(128, 256),
        dtype=torch.float16,
        device_size=[4, 128, 64],
        stride_map=[64, 256, 1],
        device_dtype_name="SEN169_FP16",
        element_arrangement_name="STANDARD",
    )
    return capture.KernelRecord(
        name=name,
        specs=[_add_op()],
        index=0,
        sencores=32,
        bundle_symbolic_args=True,
        args=[arg for _ in range(3)],
        pool_bytes=pool_bytes,
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


def test_emitted_script_executes_its_definitions():
    """The load-bearing test for capture.py.

    Module-level code includes the OpSpec literal and the LAYOUTS list, so this
    proves the spec printer round-trips and that the emitted imports cover every
    name the file uses -- an annotation needing a new import would otherwise
    break every future capture at *its* import time.  ``__main__`` does not fire
    under exec, so nothing is compiled or launched.
    """
    src = capture.emit_script(_record(pool_bytes=32768), "prog.py", 1, False)
    namespace: dict = {"__file__": "/tmp/captured/sdsc_add_0.py"}
    exec(compile(src, "<emitted>", "exec"), namespace)  # noqa: S102
    assert namespace["KERNEL_NAME"] == "sdsc_add_0"
    assert namespace["POOL_BYTES"] == 32768
    assert len(namespace["ops"]) == 1
    assert len(namespace["LAYOUTS"]) == 3
    # Keyed on the script, not the kernel: two graphs can share a fused name, and
    # a kernel-keyed .pt would be clobbered by the second capture.
    assert namespace["INPUTS_PT"] == "/tmp/captured/sdsc_add_0.inputs.pt"


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
