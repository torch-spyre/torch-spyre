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
from collections.abc import Sequence
from typing import Any

import sympy

from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.codegen.unroll import unroll_loop_specs
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec
from torch_spyre._inductor.logging_utils import get_inductor_logger


logger = get_inductor_logger("sdsc_compile")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Compiled SDSC entry: (json_dict, base_symbol_values, affine_strides)
#   base_symbol_values: list[int] of base HBM byte offsets for this SDSC,
#                       one per registered symbol ID
#   affine_strides:     list[dict] parallel to SDSCSpec.args —
#                       {tiled_sym: stride_bytes} for tiled HBM tensors,
#                       empty dict for non-tiled / lx tensors
_CompiledEntry = tuple[Any, list[int], list[dict]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_bundle(
    kernel_name: str,
    output_dir: str,
    specs: Sequence,
    use_symbols: bool | None = None,
    unroll_loops: bool | None = None,
):
    """Output the SDSC Bundle for the OpSpecs in output_dir.

    ``specs`` is a list of ``OpSpec | LoopSpec`` entries (nested ``LoopSpec``
    entries are supported).

    ``use_symbols`` controls whether HBM tensor addresses are emitted as
    runtime symbols (``%sym_N`` constants) in ``bundle.mlir`` with
    ``affine.apply`` indirection.  When ``None`` (the default) the value is
    read from ``config.bundle_hbm_symbols``.

    ``unroll_loops`` controls whether ``LoopSpec`` nodes are fully unrolled
    into flat ``OpSpec`` nodes before bundle generation.  When ``None`` (the
    default) the value is read from ``config.unroll_loops``.  Pass an explicit
    ``True`` or ``False`` to override the config — useful in unit tests that
    call ``generate_bundle`` directly.

    When ``unroll_loops=True``, each ``LoopSpec`` iteration becomes an
    independent ``OpSpec`` with concrete per-iteration HBM addresses baked in.
    When ``unroll_loops=False``, ``LoopSpec`` entries are passed through intact
    and produce ``scf.for`` loops in the generated ``bundle.mlir``.
    """
    if use_symbols is None:
        use_symbols = _spyre_config.bundle_hbm_symbols
    if unroll_loops is None:
        unroll_loops = _spyre_config.unroll_loops

    specs_list: list = unroll_loop_specs(list(specs)) if unroll_loops else list(specs)

    # -----------------------------------------------------------------------
    # Pass 1: compile all OpSpecs depth-first.
    # ``symbols`` is indexed by abs(symbol_id)-1: one entry per symbol ID in
    # registration order, values may repeat across SDSCs.  Writes one
    # ``sdsc_N.json`` file per OpSpec.
    # -----------------------------------------------------------------------
    symbols: list[int] = []
    compiled: list[_CompiledEntry] = []
    sdsc_counter = [0]
    symbol_id_offset_counter = [0]

    _compile_specs(
        specs_list,
        symbols,
        compiled,
        sdsc_counter,
        symbol_id_offset_counter,
        output_dir,
        use_symbols=use_symbols,
    )

    # -----------------------------------------------------------------------
    # Pass 2: emit bundle.mlir.
    # -----------------------------------------------------------------------

    # Collect loop bounds and affine maps needed across the whole tree.
    loop_bounds: list[sympy.Expr] = []
    _collect_loop_bounds(specs_list, loop_bounds)

    # Affine map deduplication: stride_key -> map index (0-based).
    # A stride_key is a tuple of (stride,) values — one per loop variable at
    # the nesting depth where the op lives.  For a single-level loop with one
    # tiled sym the key is (stride_bytes,).
    affine_map_index: dict[tuple, int] = {}
    _collect_affine_maps(specs_list, iter(compiled), [], affine_map_index)

    compiled_iter = iter(compiled)
    addr_counter = [0]

    with open(os.path.join(output_dir, "bundle.mlir"), "w") as f:
        logger.info(f"Generating {f.name}")

        # Module-level affine map definitions (deduped).
        for stride_key, map_idx in sorted(affine_map_index.items(), key=lambda x: x[1]):
            dims = len(stride_key)
            dim_args = ", ".join(f"d{i}" for i in range(dims))
            terms = " + ".join(f"{stride_key[i]}*d{i}" for i in range(dims))
            f.write(
                f"#map_{map_idx} = affine_map<({dim_args})[s0] -> (s0 + {terms})>\n"
            )

        f.write("module {\n")
        f.write("\tfunc.func @sdsc_bundle() {\n")

        # Standard loop constants (only emitted when there are loops).
        if loop_bounds:
            f.write("\t\t%c0 = arith.constant 0 : index\n")
            f.write("\t\t%c1 = arith.constant 1 : index\n")
            for lb_idx, lb in enumerate(loop_bounds):
                f.write(f"\t\t%loop_bound_{lb_idx} = {_mlir_count_value(lb)}\n")

        # One arith.constant per symbol ID (symbols[N] → %sym_{N+1}).
        # Skipped when use_symbols=False (symbols list is empty in that case).
        for sym_idx, value in enumerate(symbols):
            f.write(f"\t\t%sym_{sym_idx + 1} = arith.constant {value} : index\n")

        # Recursive body emission.
        loop_bound_idx = [0]
        _emit_specs(
            specs_list,
            compiled_iter,
            loop_bounds,
            loop_bound_idx,
            affine_map_index,
            addr_counter,
            [],
            f,
            indent=2,
            use_symbols=use_symbols,
        )

        f.write("\t\treturn\n")
        f.write("\t}\n")
        f.write("}\n")


# ---------------------------------------------------------------------------
# Pass 1 helpers
# ---------------------------------------------------------------------------


def _compile_specs(
    specs: list,
    symbols: list[int],
    compiled: list,
    sdsc_counter: list,
    symbol_id_offset_counter: list,
    output_dir: str,
    use_symbols: bool = False,
) -> None:
    """Recursively compile all OpSpecs in specs depth-first."""
    for entry in specs:
        if isinstance(entry, LoopSpec):
            _compile_specs(
                entry.body,
                symbols,
                compiled,
                sdsc_counter,
                symbol_id_offset_counter,
                output_dir,
                use_symbols=use_symbols,
            )
        elif isinstance(entry, OpSpec):
            idx = sdsc_counter[0]
            sdsc_counter[0] += 1
            sdsc_json, local_sym_values, affine_strides = compile_op_spec(
                idx,
                entry,
                symbols,
                symbol_id_offset_counter[0],
                use_symbols=use_symbols,
            )
            symbol_id_offset_counter[0] += len(local_sym_values)
            compiled.append((sdsc_json, local_sym_values, affine_strides))
            file_name = f"sdsc_{idx}.json"
            with open(os.path.join(output_dir, file_name), "w") as f:
                logger.info(f"Generating {f.name}")
                json.dump(sdsc_json, f, indent=2)
        # UnimplementedOp and other types are silently skipped.


# ---------------------------------------------------------------------------
# Loop-bound collection
# ---------------------------------------------------------------------------


def _collect_loop_bounds(specs: list, bounds: list) -> None:
    """Collect loop trip counts depth-first (same order as loop var naming)."""
    for entry in specs:
        if isinstance(entry, LoopSpec):
            bounds.append(entry.count)
            _collect_loop_bounds(entry.body, bounds)


# ---------------------------------------------------------------------------
# Affine map deduplication
# ---------------------------------------------------------------------------


def _collect_affine_maps(
    specs: list,
    compiled_iter,
    loop_var_depth: list,
    affine_map_index: dict,
) -> None:
    """Walk the spec tree and register unique affine stride keys."""
    for entry in specs:
        if isinstance(entry, LoopSpec):
            _collect_affine_maps(
                entry.body,
                compiled_iter,
                loop_var_depth + [len(loop_var_depth)],
                affine_map_index,
            )
        elif isinstance(entry, OpSpec):
            _, _, affine_strides = next(compiled_iter)
            for tensor_strides in affine_strides:
                if not tensor_strides:
                    continue
                # Build stride key from the tiled symbols present in this tensor,
                # in the order they appear in affine_strides dict.
                stride_key = tuple(tensor_strides.values())
                if stride_key not in affine_map_index:
                    affine_map_index[stride_key] = len(affine_map_index)


# ---------------------------------------------------------------------------
# Pass 2 helpers
# ---------------------------------------------------------------------------


def _mlir_count_value(count: sympy.Expr) -> str:
    """Return an MLIR value expression for a loop trip count."""
    if isinstance(count, (sympy.Integer, int)):
        return f"arith.constant {int(count)} : index"
    raise NotImplementedError(
        f"Symbolic loop counts are not yet supported in bundle.mlir generation: {count}"
    )


def _emit_specs(
    specs: list,
    compiled_iter,
    loop_bounds: list,
    loop_bound_idx: list,
    affine_map_index: dict,
    addr_counter: list,
    loop_vars: list,
    f,
    indent: int,
    use_symbols: bool = False,
) -> None:
    """Recursively emit MLIR ops for specs into file f."""
    tab = "\t" * indent
    for entry in specs:
        if isinstance(entry, LoopSpec):
            lb_idx = loop_bound_idx[0]
            loop_bound_idx[0] += 1
            loop_var = f"%i_{lb_idx}"
            f.write(
                f"{tab}scf.for {loop_var} = %c0 to %loop_bound_{lb_idx} step %c1 {{\n"
            )
            _emit_specs(
                entry.body,
                compiled_iter,
                loop_bounds,
                loop_bound_idx,
                affine_map_index,
                addr_counter,
                loop_vars + [loop_var],
                f,
                indent + 1,
                use_symbols=use_symbols,
            )
            f.write(f"{tab}}}\n")

        elif isinstance(entry, OpSpec):
            sdsc_json, local_sym_values, affine_strides = next(compiled_iter)
            # Determine the JSON filename from the sdsc_json key.
            sdsc_name = next(iter(sdsc_json))
            sdsc_idx = sdsc_name.split("_")[0]
            sdsc_filename = f"sdsc_{sdsc_idx}.json"

            # Extract symbol_ids from the negative IDs stored in the JSON
            # (unique, in registration order).
            symbol_ids = _extract_symbol_ids(sdsc_json)

            # Build affine.apply ops for tiled tensors, tracking which
            # symbol IDs have been upgraded to per-iteration %addr_N names.
            sym_id_to_operand: dict[int, str] = {}
            for tensor_idx, tensor_strides in enumerate(affine_strides):
                if not tensor_strides:
                    continue
                num_cores = _sdsc_num_cores(sdsc_json)
                for c in range(num_cores):
                    base_sym_id = _get_tensor_core_sym_id(sdsc_json, tensor_idx, c)
                    if base_sym_id is None or base_sym_id in sym_id_to_operand:
                        continue
                    stride_key = tuple(tensor_strides.values())
                    map_idx = affine_map_index[stride_key]
                    addr_name = f"%addr_{addr_counter[0]}"
                    addr_counter[0] += 1
                    base_addr_name = _sym_id_to_mlir_name(base_sym_id)
                    loop_var_str = ", ".join(loop_vars)
                    f.write(
                        f"{tab}{addr_name} = affine.apply #map_{map_idx}"
                        f"({loop_var_str})[{base_addr_name}]\n"
                    )
                    sym_id_to_operand[base_sym_id] = addr_name

            # Each operand position matches one symbol_id entry.
            # Tiled sym_ids use the %addr_N computed above; others use %sym_N.
            operands = [
                sym_id_to_operand.get(sid, _sym_id_to_mlir_name(sid))
                for sid in symbol_ids
            ]

            operand_str = ", ".join(operands)
            if use_symbols:
                symbol_ids_str = ", ".join(str(i) for i in symbol_ids)
                f.write(
                    f"{tab}sdscbundle.sdsc_execute ({operand_str}) "
                    f'{{sdsc_filename="{sdsc_filename}", '
                    f'"symbol_ids"=[{symbol_ids_str}]}}\n'
                )
            else:
                f.write(
                    f"{tab}sdscbundle.sdsc_execute () "
                    f'{{sdsc_filename="{sdsc_filename}"}}\n'
                )


def _extract_symbol_ids(sdsc_json: dict) -> list[int]:
    """Extract all negative symbol IDs from the SDSC JSON startAddressCoreCorelet_ data."""
    ids: list[int] = []
    seen: set[int] = set()
    for top_val in sdsc_json.values():
        for dsc_entry in top_val.get("dscs_", []):
            for op_val in dsc_entry.values():
                for node in op_val.get("scheduleTree_", []):
                    if node.get("component_") == "hbm":
                        data = node.get("startAddressCoreCorelet_", {}).get("data_", {})
                        for v in data.values():
                            sym_id = int(v)
                            if sym_id < 0 and sym_id not in seen:
                                ids.append(sym_id)
                                seen.add(sym_id)
    return ids


def _sdsc_num_cores(sdsc_json: dict) -> int:
    """Extract num_cores from the SDSC JSON."""
    for top_val in sdsc_json.values():
        return top_val.get("numCoresUsed_", 1)
    return 1


def _get_tensor_core_sym_id(sdsc_json: dict, tensor_idx: int, core: int) -> int | None:
    """Return the symbol ID (negative int) for (tensor_idx, core), or None if lx."""
    for top_val in sdsc_json.values():
        for dsc_entry in top_val.get("dscs_", []):
            for op_val in dsc_entry.values():
                nodes = op_val.get("scheduleTree_", [])
                if tensor_idx < len(nodes):
                    node = nodes[tensor_idx]
                    if node.get("component_") != "hbm":
                        return None
                    data = node.get("startAddressCoreCorelet_", {}).get("data_", {})
                    key = f"[{core}, 0, 0]"
                    if key in data:
                        return int(data[key])
    return None


def _sym_id_to_mlir_name(sym_id: int) -> str:
    """Map a negative symbol ID to a %sym_N MLIR name.

    Symbol IDs are assigned sequentially across the whole bundle starting at
    -1, and symbols[abs(id)-1] holds the corresponding value.  So %sym_N where
    N = abs(sym_id) is always correct.
    """
    return f"%sym_{abs(sym_id)}"


# ---------------------------------------------------------------------------
# Helpers re-exported for tests
# ---------------------------------------------------------------------------


def _collect_op_specs(specs: list, result: list) -> None:
    """Collect all OpSpec leaves depth-first (for tests / async_compile)."""
    for entry in specs:
        if isinstance(entry, LoopSpec):
            _collect_op_specs(entry.body, result)
        elif isinstance(entry, OpSpec):
            result.append(entry)


def _collect_loop_counts(specs: list) -> list:
    """Return loop counts in depth-first order (for tests)."""
    counts: list = []
    for entry in specs:
        if isinstance(entry, LoopSpec):
            counts.append(entry.count)
            counts.extend(_collect_loop_counts(entry.body))
    return counts
