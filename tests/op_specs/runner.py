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

"""Run a list of OpSpecs on Spyre directly, with no torch.compile involved.

``capture.py`` inlines the helpers below into every standalone script it emits,
which is what lets a captured kernel run with nothing from this repo importable.
Keep them free of module-level state and of annotations needing imports beyond
``torch``, so the inlined copy stays valid on its own.
"""

import argparse
import importlib.util
import inspect
import os
import sys

import torch

from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre._inductor.codegen.ktir import dialect_available, generate_ktir
from torch_spyre.execution.async_compile import SpyreAsyncCompile


# ---------------------------------------------------------------------------
# Inlined into captured scripts by capture.py -- see runner_template().
# ---------------------------------------------------------------------------


def ensure_runtime() -> None:
    """Bring the Spyre runtime up before any layout-aware allocation.

    ``t.to(device_layout=...)`` reaches the C++ allocator directly, bypassing the
    dispatcher that triggers ``startRuntime()``.  Without this it fails on a fresh
    process with ContextNotCreated.
    """
    torch.empty(1, dtype=torch.float16).to(torch.device("spyre"))


def pin_bundle_symbolic_args(value) -> None:
    """Force ``config.bundle_symbolic_args`` back to its capture-time value.

    A spec's ``allocation["hbm"]`` is baked when the spec is built -- the
    arg_index itself under True, an absolute address under False -- so a replay
    has to bundle under the value the capture used.  ``generate_bundle`` requires
    the flag on and raises otherwise, so a mismatch fails loudly rather than
    producing a bundle that disagrees with the spec; pinning keeps a captured
    script off the ambient ``BUNDLE_SYMBOLIC_ARGS`` altogether.  ``None`` means
    "no recorded value", as for a hand-written PROGRAM dict.
    """
    if value is None:
        return
    if spyre_config.bundle_symbolic_args != value:
        print(
            f"note: pinning bundle_symbolic_args={value} from the capture;"
            f" ambient config said {spyre_config.bundle_symbolic_args}"
        )
    spyre_config.bundle_symbolic_args = value


def to_device(tensors: list, layouts=None) -> list:
    """Move host tensors to Spyre, honouring an explicit layout when given.

    ``layouts[i]`` may be None, in which case the compiler picks the default
    tiled layout for that shape and dtype.
    """
    if layouts is not None and any(layout is not None for layout in layouts):
        # Only the explicit-layout branch bypasses the dispatcher.
        ensure_runtime()
    out = []
    for i, t in enumerate(tensors):
        layout = layouts[i] if layouts is not None else None
        if layout is None:
            out.append(t.to(torch.device("spyre")))
        else:
            out.append(t.to("spyre", device_layout=layout))
    return out


def bundle_op_specs(name: str, ops: list, out_dir: str, pool_size=0) -> list:
    """Write sdsc_N.json + bundle.mlir for ``ops`` into ``out_dir``.

    Needs no device and no backend compiler.  Returns the directory listing.
    ``pool_size`` must be non-zero whenever any TensorArg is ``hbm_pool``-
    allocated: the bundle emits its own ``device_mem_allocate`` of that size.
    """
    os.makedirs(out_dir, exist_ok=True)
    generate_bundle(name, out_dir, list(ops), pool_size=pool_size)
    return sorted(os.listdir(out_dir))


def ktir_op_specs(name: str, ops: list, out_dir: str) -> list:
    """Write ``<name>.ktir`` for ``ops`` into ``out_dir``.

    The KTIR counterpart of ``bundle_op_specs``: needs no device and no
    ``dbo-opt``, but does need the ``mlir_ktdp`` bindings.

    Whether to bake addresses is not a choice here.  The baked form reads a byte
    address out of ``allocation["hbm"]``, which only holds one when the spec was
    built with ``bundle_symbolic_args=False``, so it follows the pinned flag
    exactly as ``async_compile.ktir`` does.
    """
    if not dialect_available():
        raise SystemExit(
            "--stage ktir needs the mlir_ktdp bindings, which are not importable"
            " here. Install them with `uv sync --group ktir`; nothing else in"
            " this script needs them."
        )
    os.makedirs(out_dir, exist_ok=True)
    text = generate_ktir(
        name, list(ops), bake_addresses=not spyre_config.bundle_symbolic_args
    )
    with open(os.path.join(out_dir, f"{name}.ktir"), "w") as fh:
        fh.write(text)
    return sorted(os.listdir(out_dir))


def run_op_specs(
    name: str,
    ops: list,
    tensors: list,
    layouts=None,
    pool_size=0,
    ktir_emitter=False,
):
    """Compile ``ops`` and run them on the device against ``tensors``.

    ``tensors[i]`` is the host tensor for the TensorArg with ``arg_index=i``,
    and results are written back into it in place -- compute any reference
    value first.  ``copy_`` rather than ``t[:] =``: a 0-dim scalar arg has no
    valid slice.  Every tensor is a kernel arg: the HBM pool, when there is one,
    is allocated inside the bundle rather than passed at launch, so ``pool_size``
    only has to reach ``sdsc``.  Returns the artifact directory.

    ``ktir_emitter`` picks the backend, and has to match the capture rather than
    the ambient config: a spec built under the KTIR emitter carries baked
    addresses, which ``generate_bundle`` refuses outright, so compiling it
    through ``sdsc`` cannot work whatever the environment says.  It takes no
    ``pool_size``: that emitter refuses an ``hbm_pool`` buffer outright, so a
    kernel with a pool never reaches it.
    """
    dev_tensors = to_device(tensors, layouts)

    compiler = SpyreAsyncCompile()
    if ktir_emitter:
        runner = compiler.ktir(name, list(ops))
    else:
        runner = compiler.sdsc(name, list(ops), pool_size=pool_size)
    code_dir = getattr(runner, "code_dir", None)
    print(f"artifacts: {code_dir}")

    runner.run(*dev_tensors)

    for t, dt in zip(tensors, dev_tensors):
        t.copy_(dt.cpu())
    return code_dir


def main(
    name: str,
    ops: list,
    tensors: list,
    layouts=None,
    pool_size=0,
    bundle_symbolic_args=None,
    ktir_emitter=False,
    argv=None,
):
    """Tiny CLI shared by this module and every captured script.

    ``--stage bundle`` and ``--stage ktir`` stop after writing one emitter's
    artifacts, which needs no hardware; ``--stage run`` (the default) goes all
    the way to a launch, through whichever emitter the capture used.
    """
    pin_bundle_symbolic_args(bundle_symbolic_args)
    parser = argparse.ArgumentParser(description=f"Run OpSpec kernel {name!r}.")
    parser.add_argument(
        "--stage",
        choices=("bundle", "ktir", "run"),
        default="run",
        help=(
            "bundle: write sdsc_N.json + bundle.mlir and stop (no device or"
            " dxp_standalone needed). ktir: write <name>.ktir and stop (no device"
            " or dbo-opt needed, but mlir_ktdp must be installed). run: compile"
            " and launch. Default: run."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for --stage bundle/ktir. Default: ./op_spec_out/<name>.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print a decoded explanation of the OpSpec list before running.",
    )
    parser.add_argument(
        "--explain-verbose",
        action="store_true",
        help="With --explain, also dump the raw SDSCSpec for each op.",
    )
    args = parser.parse_args(argv)

    if args.explain or args.explain_verbose:
        # Lazy and guarded: this function is inlined into captured scripts, so
        # --stage run can never depend on a repo import. --explain may degrade.
        try:
            from explain import render
        except ImportError:
            print(
                "(--explain needs explain.py importable: re-run with"
                " PYTHONPATH=<repo>/tests/op_specs. Nothing else here needs it.)"
            )
        else:
            print(
                render(
                    ops,
                    kernel_name=name,
                    pool_size=pool_size,
                    verbose=args.explain_verbose,
                )
            )

    if args.stage in ("bundle", "ktir"):
        out_dir = args.out_dir or os.path.join("op_spec_out", name)
        if args.stage == "bundle":
            listing = bundle_op_specs(name, ops, out_dir, pool_size=pool_size)
        else:
            listing = ktir_op_specs(name, ops, out_dir)
        print(f"artifacts: {os.path.abspath(out_dir)}")
        print(f"contents: {listing}")
        return out_dir

    code_dir = run_op_specs(
        name,
        ops,
        tensors,
        layouts=layouts,
        pool_size=pool_size,
        ktir_emitter=ktir_emitter,
    )
    # Only after a launch: before one these are still the inputs.
    for i, t in enumerate(tensors):
        print(f"arg{i} {tuple(t.shape)} {t.dtype}: {t.flatten()[:6].tolist()}")
    return code_dir


# ---------------------------------------------------------------------------
# Template extraction
# ---------------------------------------------------------------------------

# Import block a captured script needs for the inlined helpers above.  capture.py
# extends this with the imports the OpSpec literal itself requires.
TEMPLATE_IMPORTS = """\
import argparse
import os

import torch

from torch_spyre._C import DataFormats, SpyreTensorLayout
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre._inductor.codegen.ktir import dialect_available, generate_ktir
from torch_spyre.execution.async_compile import SpyreAsyncCompile
"""

_TEMPLATE_FUNCS = (
    ensure_runtime,
    pin_bundle_symbolic_args,
    to_device,
    bundle_op_specs,
    ktir_op_specs,
    run_op_specs,
    main,
)


def runner_template() -> str:
    """Return the helpers above as source text, for inlining into a script.

    Read straight off the live functions with ``inspect.getsource`` rather than
    kept as a parallel string literal, so a captured script can never embed a
    stale copy of the run loop.
    """
    return "\n\n".join(inspect.getsource(fn).rstrip() for fn in _TEMPLATE_FUNCS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_program(path: str) -> dict:
    """Import ``path`` and return its ``PROGRAM`` dict of main() kwargs."""
    spec = importlib.util.spec_from_file_location("_op_spec_program", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    program = getattr(module, "PROGRAM", None)
    if program is None:
        raise SystemExit(
            f"{path} defines no PROGRAM dict. Expected e.g.\n"
            "    PROGRAM = {'name': 'my_kernel', 'ops': ops, 'tensors': [x, y, z]}"
        )
    return dict(program)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        raise SystemExit(
            "usage: python tests/op_specs/runner.py <program.py> [--stage ...]\n"
            "  <program.py> must define a PROGRAM dict of main() keyword args."
        )
    main(argv=sys.argv[2:], **_load_program(sys.argv[1]))
