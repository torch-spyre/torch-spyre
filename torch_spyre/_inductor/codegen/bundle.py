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

import json
import os
from typing import IO

import sympy

from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec
from torch_spyre._inductor.logging_utils import get_inductor_logger


logger = get_inductor_logger("sdsc_compile")


def generate_bundle(kernel_name: str, output_dir: str, specs: list[OpSpec | LoopSpec]):
    """Output the SDSC Bundle for the OpSpecs in the given output_dir.

    ``specs`` may contain ``OpSpec`` entries (compiled to SDSC JSON and
    executed directly) or ``LoopSpec`` entries (whose body ops are executed
    inside an ``scf.for`` loop in ``bundle.mlir``).  ``LoopSpec`` bodies may
    themselves contain nested ``LoopSpec`` entries.

    SDSC JSON files are numbered in depth-first traversal order across all
    nesting levels.
    """
    # Collect all OpSpecs in depth-first order, compile them, write JSON.
    op_specs_ordered: list[OpSpec] = []
    _collect_op_specs(specs, op_specs_ordered)

    files: list[str] = []
    for idx, op_spec in enumerate(op_specs_ordered):
        sdsc_json = compile_op_spec(idx, op_spec)
        sdsc_name = next(iter(sdsc_json))
        file_name = f"sdsc_{sdsc_name}.json"
        files.append(file_name)
        with open(os.path.join(output_dir, file_name), "w") as f:
            logger.info(f"Generating {f.name}")
            json.dump(sdsc_json, f, indent=2)

    # Build a name→file index for the mlir emitter.
    # op_specs_ordered[i] corresponds to files[i]; the emitter walks the
    # same depth-first order and pops from a shared counter.
    file_iter = iter(files)

    # Generate bundle.mlir
    with open(os.path.join(output_dir, "bundle.mlir"), "w") as mlir_file:
        logger.info(f"Generating {mlir_file.name}")
        mlir_file.write("module {\n")
        mlir_file.write("\tfunc.func @sdsc_bundle() {\n")

        # Emit arith.constant declarations needed for loop bounds.
        loop_counts = _collect_loop_counts(specs)
        if loop_counts:
            mlir_file.write("\t\t%c0 = arith.constant 0 : index\n")
            mlir_file.write("\t\t%c1 = arith.constant 1 : index\n")
            for i, count in enumerate(loop_counts):
                mlir_file.write(f"\t\t%loop_bound_{i} = {_mlir_count_value(count)}\n")

        count_idx = [0]  # mutable counter for loop_counts list
        _emit_mlir_body(
            specs, mlir_file, indent=2, file_iter=file_iter, count_idx=count_idx
        )

        mlir_file.write("\t\treturn\n")
        mlir_file.write("\t}\n")
        mlir_file.write("}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_op_specs(specs: list[OpSpec | LoopSpec], result: list[OpSpec]) -> None:
    """Depth-first traversal: append every OpSpec in order."""
    for item in specs:
        if isinstance(item, LoopSpec):
            _collect_op_specs(item.body, result)
        elif isinstance(item, OpSpec):
            result.append(item)


def _collect_loop_counts(
    specs: list[OpSpec | LoopSpec],
) -> list[sympy.Expr]:
    """Depth-first traversal: collect loop counts in encounter order."""
    result: list[sympy.Expr] = []
    for item in specs:
        if isinstance(item, LoopSpec):
            result.append(item.count)
            result.extend(_collect_loop_counts(item.body))
    return result


def _mlir_count_value(count: sympy.Expr) -> str:
    """Return the MLIR RHS expression for a loop count.

    Concrete integers become ``arith.constant N : index``.
    Symbolic counts are not yet supported in the prototype; raise so the
    caller sees a clear error rather than silently wrong codegen.
    """
    if isinstance(count, sympy.Integer) or (
        not count.free_symbols and count.is_integer
    ):
        n = int(count)
        return f"arith.constant {n} : index"
    raise NotImplementedError(
        f"Symbolic loop count {count!r} in bundle.mlir is not yet supported. "
        "Runtime shape wiring for scf.for bounds is a TODO."
    )


def _emit_mlir_body(
    specs: list[OpSpec | LoopSpec],
    f: IO[str],
    indent: int,
    file_iter,
    count_idx: list[int],
) -> None:
    """Recursively emit sdsc_execute calls and scf.for blocks."""
    tab = "\t" * indent
    for item in specs:
        if isinstance(item, LoopSpec):
            loop_var = f"%i_{count_idx[0]}"
            count_var = f"%loop_bound_{count_idx[0]}"
            count_idx[0] += 1
            f.write(f"{tab}scf.for {loop_var} = %c0 to {count_var} step %c1 {{\n")
            _emit_mlir_body(item.body, f, indent + 1, file_iter, count_idx)
            f.write(f"{tab}}}\n")
        elif isinstance(item, OpSpec):
            file_name = next(file_iter)
            f.write(
                f'{tab}sdscbundle.sdsc_execute () {{sdsc_filename="{file_name}"}}\n'
            )
