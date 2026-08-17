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

"""Turn a PyTorch program into standalone, runnable OpSpec scripts.

Point it at a script you already have -- it runs unmodified -- and it writes one
self-contained Python file per compiled kernel::

    python tests/op_specs/capture.py my_repro.py --out captured/
    python captured/sdsc_fused_add_mul_0.py

Each emitted file carries that kernel's whole OpSpec list plus tensors matching
the real shapes, dtypes and device layouts, so it reproduces the backend half of
the compile with the Inductor frontend removed.

A *kernel* here is one ``SpyreAsyncCompile.sdsc()`` call, which generally holds
several OpSpecs -- so one emitted file is one kernel, not one operation.  The
target script need not call ``torch.compile``: many eager Spyre ops are
themselves implemented by compiling, which is also why a script running an op
both ways compiles it twice (see :func:`dedup_key`).

Options:
    --out DIR         where to write the scripts (default: ./captured)
    --kernel NAME     only emit kernels whose name contains NAME
    --save-inputs     also dump recorded input values (and pool bytes) to a .pt
                      beside each script, for byte-exact replay
    --no-execute      capture without a device or dxp_standalone (see below)
    --no-explain-header
                      omit the decoded OpSpec explanation from each script

Exit status:
    0                 the target ran to completion and every kernel was written
    3                 scripts were written, but the target raised partway through
                      so later kernels are missing (see ``EXIT_PARTIAL``)
    1                 nothing was written
"""

import argparse
import contextlib
import dataclasses
import os
import runpy
import sys
import traceback
from unittest.mock import patch

import torch

import torch._inductor.config as inductor_config
from torch._inductor.utils import IndentedBuffer

import torch_spyre  # noqa: F401  -- registers the "spyre" device
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.op_spec import IndirectAccess, TensorArg
from torch_spyre._inductor.spyre_kernel import (
    _codegen_op_spec_list,
    _iter_op_specs,
    uses_hbm_pool,
)
from torch_spyre.execution.async_compile import SpyreAsyncCompile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explain import _enum_name, render_comment_block  # noqa: E402
from runner import TEMPLATE_IMPORTS, runner_template  # noqa: E402

# Mock targets for --no-execute, matching docs/tools/capture_coarse_tile_ir.py.
_PREPARE_KERNEL = "torch_spyre.execution.kernel_runner.prepare_kernel"
_LAUNCH_JOBPLAN = "torch_spyre.execution.kernel_runner.launch_jobplan"

# Scripts were written, but the target raised before it finished, so the capture
# covers only the kernels compiled up to that point.  Distinct from 1 ("nothing
# written") and from argparse's 2, so a caller can tell a partial capture from a
# complete one without parsing stdout.
EXIT_PARTIAL = 3

LICENSE_HEADER = "".join(
    f"# {line}\n" if line else "#\n"
    for line in (
        "Copyright 2026 The Torch-Spyre Authors.",
        "",
        'Licensed under the Apache License, Version 2.0 (the "License"); you may not',
        "use this file except in compliance with the License. You may obtain a copy of",
        "the License at",
        "",
        "    http://www.apache.org/licenses/LICENSE-2.0",
        "",
        "Unless required by applicable law or agreed to in writing, software",
        'distributed under the License is distributed on an "AS IS" BASIS, WITHOUT',
        "WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the",
        "License for the specific language governing permissions and limitations under",
        "the License.",
    )
)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ArgRecord:
    """One tensor observed being passed to a kernel's .run()."""

    shape: tuple
    dtype: torch.dtype
    device_size: list
    stride_map: list
    device_dtype_name: str
    element_arrangement_name: str
    values: object = None


@dataclasses.dataclass
class KernelRecord:
    """One sdsc() call: its name, its OpSpec list, and its observed inputs.

    ``sencores`` and ``bundle_symbolic_args`` are snapshotted at spec-build time,
    not at write time: a program may build its specs inside a ``config.patch``
    that has already exited.  ``sencores`` is provenance only -- work division is
    already baked into ``iteration_space``.  ``bundle_symbolic_args`` is not, and
    the emitted script pins it: see ``pin_bundle_symbolic_args`` in runner.py.
    """

    name: str
    specs: list
    index: int
    sencores: int
    bundle_symbolic_args: bool
    args: list = dataclasses.field(default_factory=list)
    pool_bytes: int = 0
    observed_run: bool = False


def _record_args(rec: KernelRecord, args: tuple, save_inputs: bool) -> None:
    """Record shape/dtype/device-layout for each .run() argument.

    The pool's *size* is recorded but never its contents, even under
    ``save_inputs``: ``pool.cpu()`` on the flat SENINT8 pool tensor corrupts the
    heap and aborts the process, which no try/except can catch.  Replays get an
    uninitialized pool, as production does for a graph's first kernel.

    So the pool must be identified by the same predicate ``call_kernel`` used to
    prepend it -- ``uses_hbm_pool``, shared rather than reimplemented, because a
    copy that drifted would mean calling ``.cpu()`` on the pool.
    """
    remaining = list(args)
    if uses_hbm_pool(rec.specs) and remaining:
        pool = remaining.pop(0)
        rec.pool_bytes = int(pool.numel())
    for tensor in remaining:
        layout = tensor.device_tensor_layout()
        rec.args.append(
            ArgRecord(
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                device_size=list(layout.device_size),
                stride_map=list(layout.stride_map),
                device_dtype_name=_enum_name(layout.device_dtype),
                element_arrangement_name=_enum_name(layout.element_arrangement),
                values=tensor.cpu() if save_inputs else None,
            )
        )


class _RunRecorder:
    """Proxy around a kernel runner that records the first .run() it sees.

    Only the first call: a repeatedly-invoked graph would otherwise keep
    overwriting the inputs.
    """

    def __init__(self, inner, rec: KernelRecord, save_inputs: bool):
        self._inner = inner
        self._rec = rec
        self._save_inputs = save_inputs

    def __getattr__(self, name):
        # Only reached for names the proxy itself does not define.
        return getattr(self._inner, name)

    def run(self, *args, **kwargs):
        if not self._rec.observed_run:
            _record_args(self._rec, args, self._save_inputs)
            self._rec.observed_run = True
        return self._inner.run(*args, **kwargs)


@contextlib.contextmanager
def capture_kernels(save_inputs: bool = False, no_execute: bool = False):
    """Record every sdsc() call made inside the block.

    Yields the list of :class:`KernelRecord`, appended to as compilation happens.
    ``force_disable_caches`` is patched on for the duration: a warm fxgraph cache
    skips recompilation and the capture would silently produce nothing.
    """
    records: list = []
    real_sdsc = SpyreAsyncCompile.sdsc

    def spy(self, kernel_name, specs):
        rec = KernelRecord(
            name=kernel_name,
            specs=list(specs),
            index=len(records),
            sencores=spyre_config.sencores,
            bundle_symbolic_args=spyre_config.bundle_symbolic_args,
        )
        records.append(rec)
        return _RunRecorder(real_sdsc(self, kernel_name, specs), rec, save_inputs)

    with contextlib.ExitStack() as stack:
        stack.enter_context(inductor_config.patch({"force_disable_caches": True}))
        stack.enter_context(patch.object(SpyreAsyncCompile, "sdsc", spy))
        if no_execute:
            # Bundle generation still runs -- it is what validates the spec. Note
            # patching subprocess.run also stubs any subprocess the target spawns.
            stack.enter_context(patch(_PREPARE_KERNEL))
            stack.enter_context(patch(_LAUNCH_JOBPLAN))
            stack.enter_context(patch("subprocess.run"))
        yield records


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def spec_source(specs: list) -> str:
    """Render an OpSpec list as eval-able Python source.

    Uses the compiler's own printer, so the literal is what the compiler would
    have written rather than something re-derived here.
    """

    def sympy_str(expr) -> str:
        # Mirrors the local closure in SpyreKernel.codegen_kernel, which cannot
        # be imported; keep the two in sync.
        if isinstance(expr, IndirectAccess):
            return f"IndirectAccess('{expr.args[0]}')"
        return "sympify('" + str(expr) + "')"

    buf = IndentedBuffer()
    buf.writeline("[")
    with buf.indent():
        _codegen_op_spec_list(list(specs), buf, sympy_str)
    buf.writeline("]")
    return buf.getvalue()


def dedup_key(specs: list) -> str:
    """Rendered spec with provenance stripped, for identity comparison.

    ``debug_handle`` must be excluded or dedup is dead code: its id and source
    vary across compilation *paths* without affecting execution.  The case that
    shows why is an eager Spyre op, itself implemented by ``torch.compile``, run
    both eagerly and compiled -- two identically-named kernels whose specs match
    apart from these fields.
    """
    return "\n".join(
        line
        for line in spec_source(specs).splitlines()
        if not line.lstrip().startswith("debug_handle=")
    )


def _layout_source(arg: ArgRecord) -> str:
    """A SpyreTensorLayout constructor call reproducing a recorded layout.

    Round-trips the observed layout without inferring anything from the host
    shape.
    """
    return (
        "    SpyreTensorLayout(\n"
        f"        device_size={arg.device_size},\n"
        f"        stride_map={arg.stride_map},\n"
        f"        device_dtype=DataFormats.{arg.device_dtype_name},\n"
        "        element_arrangement=ElementArrangement."
        f"{arg.element_arrangement_name},\n"
        "    ),"
    )


def _provenance(rec: KernelRecord, source: str, total: int, no_execute: bool) -> str:
    """Header comment recording how and from what this script was captured."""
    n_spec_args = len(
        {
            arg.arg_index
            for op in _iter_op_specs(rec.specs)
            for arg in op.args
            if isinstance(arg, TensorArg) and arg.arg_index >= 0
        }
    )
    # A fused kernel name can run past 80 columns by itself, so give it its own
    # line rather than overflow the header.
    title = f"Captured OpSpec kernel: {rec.name}"
    split = ["Captured OpSpec kernel:", f"    {rec.name}"]
    lines = [title] if len(title) <= 76 else split
    lines += [
        "",
        f"Source program:  {source}",
        f"Kernel:          {rec.index + 1} of {total} compiled from that program",
        f"Environment:     torch {torch.__version__}, "
        f"SENCORES={rec.sencores}, "
        f"bundle_symbolic_args={rec.bundle_symbolic_args}",
        f"Kernel args:     {n_spec_args} in the spec, "
        f"{len(rec.args)} observed at .run()",
    ]
    if rec.pool_bytes:
        lines.append(
            f"Pool:            {rec.pool_bytes} bytes, passed to .run() "
            "ahead of the args"
        )
    lines += [
        "",
        "Run it:",
        "    python <this file>                 # compile and launch on the device",
        "    python <this file> --stage bundle  # sdsc_N.json + bundle.mlir only",
    ]
    if not rec.observed_run:
        lines += [
            "",
            "WARNING: no .run() was observed for this kernel, so input shapes and",
            "layouts below are empty. --stage bundle still works; fill in SHAPES",
            "and LAYOUTS by hand to run it.",
        ]
    if no_execute:
        lines += [
            "",
            "WARNING: captured with --no-execute, so nothing actually ran. Shapes",
            "and layouts are still exact, but any recorded *values* for a kernel",
            "after the first are meaningless -- their inputs are intermediates",
            "that no earlier kernel computed. Whatever the source program printed",
            "during the capture is meaningless for the same reason: it read back",
            "tensors no launch ever wrote.",
        ]
    return "".join(f"# {line}\n" if line else "#\n" for line in lines)


def _explain_header(rec: KernelRecord) -> str:
    """The decoded OpSpec explanation, as a comment block for the script header.

    Embedded at capture time so the emitted file self-documents with no import
    of this repo.  Never allowed to fail the capture.
    """
    try:
        block = render_comment_block(
            rec.specs,
            kernel_name=rec.name,
            args=rec.args or None,
            pool_bytes=rec.pool_bytes,
        )
    except Exception as exc:
        block = f"# (explain header unavailable: {type(exc).__name__}: {exc})"
    # Trailing blank line: the template splices this directly against the
    # import block, which would otherwise start on the next line.
    return block + "\n\n"


def emit_script(
    rec: KernelRecord,
    source: str,
    total: int,
    no_execute: bool,
    explain_header: bool = True,
) -> str:
    """Build the full text of the standalone script for one kernel."""
    shapes = ",\n".join(
        f"    ({list(a.shape)}, torch.{str(a.dtype).removeprefix('torch.')})"
        for a in rec.args
    )
    layouts = "\n".join(_layout_source(a) for a in rec.args)
    explain = _explain_header(rec) if explain_header else ""

    return f'''{LICENSE_HEADER}
{_provenance(rec, source, total, no_execute)}
{explain}{TEMPLATE_IMPORTS}
from sympy import sympify

from torch_spyre._C import ElementArrangement
from torch_spyre._inductor.op_spec import (
    DebugHandle,
    IndirectAccess,
    LoopSpec,
    OpSpec,
    ProvenanceTransform,
    SourceLoc,
    TensorArg,
    UnimplementedOp,
    spyre_constant_tensor,
)

KERNEL_NAME = "{rec.name}"
POOL_BYTES = {rec.pool_bytes}
BUNDLE_SYMBOLIC_ARGS = {rec.bundle_symbolic_args}
# Named after this file, not the kernel: two graphs can share a fused kernel
# name, and write_kernel disambiguates the scripts (name_1.py) -- so a
# kernel-keyed .pt would be shared and the second capture would clobber the first.
INPUTS_PT = os.path.splitext(os.path.abspath(__file__))[0] + ".inputs.pt"

# Host (shape, dtype) per kernel arg, in arg_index order.
SHAPES = [
{shapes}
]

# Exact device layout each arg had when the real graph ran.
LAYOUTS = [
{layouts}
]

ops = {spec_source(rec.specs)}

def build_tensors():
    """Return (host tensors, pool bytes or None) for this kernel's args.

    Replays a .inputs.pt sitting beside this script, otherwise synthesizes.
    Integer args get zeros, not random values: they are usually indirect-access
    indices, where random data would be out of bounds.
    """
    if os.path.exists(INPUTS_PT):
        saved = torch.load(INPUTS_PT)
        print(f"inputs: replayed from {{INPUTS_PT}}")
        return saved["tensors"], saved["pool"]
    torch.manual_seed(0xAFFE)
    tensors = []
    for shape, dtype in SHAPES:
        if dtype.is_floating_point:
            # The fp8 dtypes report is_floating_point but have no uniform
            # kernel, so draw in fp32 and cast. Keyed on itemsize rather than a
            # list of dtypes, which would go stale as fp8 variants are added.
            src = dtype if dtype.itemsize > 1 else torch.float32
            tensors.append(torch.rand(shape, dtype=src).to(dtype))
        else:
            tensors.append(torch.zeros(shape, dtype=dtype))
    return tensors, None


{runner_template()}


if __name__ == "__main__":
    host_tensors, pool_contents = build_tensors()
    main(
        KERNEL_NAME,
        ops,
        host_tensors,
        layouts=LAYOUTS,
        pool_bytes=POOL_BYTES,
        pool_contents=pool_contents,
        bundle_symbolic_args=BUNDLE_SYMBOLIC_ARGS,
    )
'''


def write_kernel(
    rec, out_dir, source, total, no_execute, save_inputs, explain_header=True
) -> str:
    """Write one kernel's script (and .pt) into out_dir; return the script path."""
    path = os.path.join(out_dir, f"{rec.name}.py")
    suffix = 1
    while os.path.exists(path):
        # Two graphs in one program can produce the same fused kernel name.
        path = os.path.join(out_dir, f"{rec.name}_{suffix}.py")
        suffix += 1
    with open(path, "w") as f:
        f.write(emit_script(rec, source, total, no_execute, explain_header))
    if save_inputs and rec.observed_run:
        # Keyed on the script path, so a disambiguated script gets its own .pt.
        # "pool" is always None -- see _record_args -- but stays in the dict so
        # the schema does not depend on whether the kernel used one.
        torch.save(
            {"tensors": [a.values for a in rec.args], "pool": None},
            os.path.splitext(path)[0] + ".inputs.pt",
        )
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("script", help="PyTorch program to capture; runs unmodified.")
    parser.add_argument("--out", default="captured", help="Output dir.")
    parser.add_argument("--kernel", default=None, help="Only kernels matching this.")
    parser.add_argument(
        "--save-inputs",
        action="store_true",
        help="Dump recorded input values and pool bytes to a .pt for exact replay.",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Stub the backend compile and launch, so no device is needed.",
    )
    parser.add_argument(
        "--no-explain-header",
        action="store_true",
        help="Omit the decoded OpSpec explanation from each emitted script.",
    )
    args = parser.parse_args(argv)

    script = os.path.abspath(args.script)
    if not os.path.exists(script):
        raise SystemExit(f"no such script: {script}")

    failure = None
    with capture_kernels(args.save_inputs, args.no_execute) as records:
        # Run as if invoked directly, so the target's __main__ block still fires.
        sys.argv = [script]
        try:
            runpy.run_path(script, run_name="__main__")
        except Exception as exc:  # noqa: BLE001 -- a crash is a capture, not a stop
            # The programs worth capturing are the ones that misbehave, and a
            # kernel holding an UnimplementedOp raises on every launch. Whatever
            # was recorded before the failure is still a valid reproducer, so
            # write it out and report the exception at the end.
            failure = exc
            traceback.print_exc()

    if args.no_execute:
        # Said here as well as in the emitted header: by the time anyone reads
        # the header they have already seen the meaningless number.
        print(
            "\nnote: --no-execute stubbed the launch, so any numbers the target"
            "\n      program printed above came from unwritten tensors."
        )

    if not records:
        if failure is not None:
            raise SystemExit(
                f"no kernels captured: {script} raised "
                f"{type(failure).__name__} before compiling anything (traceback above)."
            )
        raise SystemExit(
            f"no kernels captured from {script}.\n"
            "Nothing was compiled for Spyre: check the script calls torch.compile "
            "on tensors that are on the 'spyre' device."
        )

    selected = [r for r in records if args.kernel is None or args.kernel in r.name]
    if not selected:
        names = ", ".join(r.name for r in records)
        raise SystemExit(f"--kernel {args.kernel!r} matched none of: {names}")

    os.makedirs(args.out, exist_ok=True)
    print(f"\ncaptured {len(records)} kernel(s); writing {len(selected)}")

    # Dedup on the rendered spec rather than the name alone: different graphs can
    # share a fused name. Provenance excluded -- see dedup_key.
    seen: dict = {}
    written = 0
    for rec in selected:
        key = (rec.name, dedup_key(rec.specs))
        if key in seen:
            other = os.path.basename(seen[key])
            print(f"  (skipped {rec.name}: same spec as {other}, differing only")
            print("   in debug provenance -- e.g. an eager vs compiled call)")
            continue
        path = write_kernel(
            rec,
            args.out,
            script,
            len(records),
            args.no_execute,
            args.save_inputs,
            explain_header=not args.no_explain_header,
        )
        seen[key] = path
        written += 1
        n_ops = sum(1 for _ in _iter_op_specs(rec.specs))
        detail = f"{n_ops} OpSpec(s), {len(rec.args)} arg(s)"
        if rec.pool_bytes:
            detail += f", {rec.pool_bytes}-byte pool"
        note = "" if rec.observed_run else "  [no .run() seen -- shapes unfilled]"
        print(f"  {path}  ({detail}){note}")
        if args.save_inputs and rec.pool_bytes:
            print("    note: arg values saved; pool contents are not captured")

    if written == 0:
        raise SystemExit("nothing written")
    if failure is not None:
        print(
            f"\nnote: {os.path.basename(script)} raised"
            f" {type(failure).__name__} partway through (traceback above)."
            "\n      The scripts above cover the kernels compiled before that"
            f"\n      point; later kernels never reached sdsc(). Exit {EXIT_PARTIAL}."
        )
        return EXIT_PARTIAL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
