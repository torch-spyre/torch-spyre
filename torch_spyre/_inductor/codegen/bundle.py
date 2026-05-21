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
from typing import Any

import sympy

from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec
from torch_spyre._inductor.logging_utils import get_inductor_logger


logger = get_inductor_logger("sdsc_compile")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Compiled SDSC entry: (json_dict, base_symbol_values, affine_strides)
#   base_symbol_values: list[int] of base HBM byte offsets registered in the
#                       global symbols table for this SDSC
#   affine_strides:     list[dict] parallel to SDSCSpec.args —
#                       {tiled_sym: stride_bytes} for tiled HBM tensors,
#                       empty dict for non-tiled / lx tensors
_CompiledEntry = tuple[Any, list[int], list[dict]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_bundle(kernel_name: str, output_dir: str, specs: list):
    """Output the SDSC Bundle for the OpSpecs in output_dir.

    ``specs`` is a list of ``OpSpec | LoopSpec`` entries (nested ``LoopSpec``
    entries are supported).  ``LoopSpec`` entries produce ``scf.for`` loops in
    the generated ``bundle.mlir``, with ``affine.apply`` expressions computing
    per-iteration tensor start addresses for tiled dimensions.
    """
    # -----------------------------------------------------------------------
    # Pass 1: compile all OpSpecs depth-first.
    # Populates the global deduplicated ``symbols`` list of base HBM addresses
    # and writes one ``sdsc_N.json`` file per OpSpec.
    # -----------------------------------------------------------------------
    symbols: list[int] = []
    compiled: list[_CompiledEntry] = []
    sdsc_counter = [0]
    symbol_id_offset_counter = [0]

    _compile_specs(
        specs, symbols, compiled, sdsc_counter, symbol_id_offset_counter, output_dir
    )

    # -----------------------------------------------------------------------
    # Pass 2: emit bundle.mlir.
    # -----------------------------------------------------------------------

    # Collect loop bounds and affine maps needed across the whole tree.
    loop_bounds: list[sympy.Expr] = []
    _collect_loop_bounds(specs, loop_bounds)

    # Affine map deduplication: stride_key -> map index (0-based).
    # A stride_key is a tuple of (stride,) values — one per loop variable at
    # the nesting depth where the op lives.  For a single-level loop with one
    # tiled sym the key is (stride_bytes,).
    affine_map_index: dict[tuple, int] = {}
    _collect_affine_maps(specs, compiled, iter(compiled), [], affine_map_index)

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

        # One arith.constant per unique base address.
        for sym_idx, value in enumerate(symbols):
            f.write(f"\t\t%sym_{sym_idx + 1} = arith.constant {value} : index\n")

        # Recursive body emission.
        loop_bound_idx = [0]
        _emit_specs(
            specs,
            compiled_iter,
            loop_bounds,
            loop_bound_idx,
            symbols,
            affine_map_index,
            addr_counter,
            [],
            f,
            indent=2,
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
            )
        elif isinstance(entry, OpSpec):
            idx = sdsc_counter[0]
            sdsc_counter[0] += 1
            sdsc_json, local_sym_values, affine_strides = compile_op_spec(
                idx, entry, symbols, symbol_id_offset_counter[0]
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
    compiled: list,
    compiled_iter,
    loop_var_depth: list,
    affine_map_index: dict,
) -> None:
    """Walk the spec tree and register unique affine stride keys."""
    for entry in specs:
        if isinstance(entry, LoopSpec):
            _collect_affine_maps(
                entry.body,
                compiled,
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
    symbols: list[int],
    affine_map_index: dict,
    addr_counter: list,
    loop_vars: list,
    f,
    indent: int,
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
                symbols,
                affine_map_index,
                addr_counter,
                loop_vars + [loop_var],
                f,
                indent + 1,
            )
            f.write(f"{tab}}}\n")

        elif isinstance(entry, OpSpec):
            sdsc_json, local_sym_values, affine_strides = next(compiled_iter)
            # Determine the JSON filename from the sdsc_json key.
            sdsc_name = next(iter(sdsc_json))
            sdsc_idx = sdsc_name.split("_")[0]
            sdsc_filename = f"sdsc_{sdsc_idx}.json"

            # Extract symbol_ids from the negative IDs stored in the JSON.
            symbol_ids = _extract_symbol_ids(sdsc_json)

            # Build operand list: one %sym_N or %addr_N per (tensor, core).
            operands: list[str] = []
            for tensor_idx, tensor_strides in enumerate(affine_strides):
                num_cores = _sdsc_num_cores(sdsc_json)
                for c in range(num_cores):
                    addr_str = _emit_tensor_core_addr(
                        tensor_idx,
                        c,
                        tensor_strides,
                        sdsc_json,
                        symbols,
                        affine_map_index,
                        addr_counter,
                        loop_vars,
                        f,
                        tab,
                    )
                    if addr_str is not None:
                        operands.append(addr_str)

            operand_str = ", ".join(operands)
            symbol_ids_str = ", ".join(str(i) for i in symbol_ids)
            f.write(
                f"{tab}sdscbundle.sdsc_execute ({operand_str}) "
                f'{{sdsc_filename="{sdsc_filename}", '
                f'"symbol_ids"=[{symbol_ids_str}]}}\n'
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


def _emit_tensor_core_addr(
    tensor_idx: int,
    core: int,
    tensor_strides: dict,
    sdsc_json: dict,
    symbols: list[int],
    affine_map_index: dict,
    addr_counter: list,
    loop_vars: list,
    f,
    tab: str,
) -> str | None:
    """Emit an affine.apply (if tiled) or return %sym_N (if not), or None (if lx).

    Returns the MLIR SSA name to use as an operand to sdsc_execute, or None
    if the tensor is lx (address baked into JSON, not an operand).
    """
    # Look up this tensor's base symbol ID from the JSON.
    base_sym_id = _get_tensor_core_sym_id(sdsc_json, tensor_idx, core)
    if base_sym_id is None:
        # lx tensor — address is baked into JSON, not an operand.
        return None

    # Map the negative symbol ID back to a %sym_N name.
    # symbols list index = (-base_sym_id - 1) for globally allocated symbols,
    # but the actual global index is stored in the symbols list by value.
    # We need to find which %sym_N corresponds to this symbol ID.
    base_addr_name = _sym_id_to_mlir_name(base_sym_id, sdsc_json, symbols)

    if not tensor_strides or not loop_vars:
        # Non-tiled or outside a loop: use the base constant directly.
        return base_addr_name

    # Tiled tensor inside a loop: emit affine.apply.
    stride_key = tuple(tensor_strides.values())
    map_idx = affine_map_index[stride_key]
    addr_name = f"%addr_{addr_counter[0]}"
    addr_counter[0] += 1
    loop_var_str = ", ".join(loop_vars)
    f.write(
        f"{tab}{addr_name} = affine.apply #map_{map_idx}"
        f"({loop_var_str})[{base_addr_name}]\n"
    )
    return addr_name


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


def _sym_id_to_mlir_name(sym_id: int, sdsc_json: dict, symbols: list[int]) -> str:
    """Map a negative symbol ID back to a %sym_N MLIR name.

    The mapping is: the N-th unique symbol ID registered for this SDSC (in
    registration order) maps to %sym_{global_index+1} in the symbols list.
    We recover the base address value from the compiled entry and look it up
    in the global symbols list.
    """
    # Build the local -> global mapping by collecting all unique symbol IDs
    # from the JSON in registration order, paired with their position in symbols.
    # The global %sym_N index is: symbols.index(base_addr_value) + 1.
    # We have the sym_id but need the base_addr_value.  Since the JSON stores
    # str(sym_id) as the data value, we collect (sym_id -> base_addr_value) by
    # cross-referencing with the local_sym_values we stored... but we don't have
    # them here.  Instead, we rebuild the mapping: the K-th unique negative ID
    # in the JSON (in ascending absolute value) corresponds to the K-th entry
    # in local_sym_values.  We get the %sym_N name from the position of that
    # value in the global symbols list.
    #
    # However, we don't have local_sym_values here.  To avoid threading it,
    # we use the fact that the sdsc_json data values ARE the symbol IDs, and
    # the global symbols list contains the base addresses in registration order.
    # The (abs(sym_id) - 1 - offset) within the local group gives the local
    # index, which maps to the same position in local_sym_values.
    #
    # Simpler approach: collect unique sym_ids in order of absolute value,
    # that order is the same as local registration order, which is also the
    # order entries were appended to the global symbols list.  Find the first
    # sym_id in global symbols to get the starting offset.
    all_sym_ids = _extract_symbol_ids(sdsc_json)
    if not all_sym_ids:
        raise RuntimeError(f"No symbol IDs found in SDSC JSON for sym_id={sym_id}")
    # Sort by absolute value (registration order).
    ordered = sorted(all_sym_ids, key=lambda x: abs(x))
    local_idx = ordered.index(sym_id)
    # The global index of the first symbol for this SDSC.
    # We locate it by finding the minimum abs value sym_id's global position.
    # Since local_sym_values are appended to symbols in order, and the first
    # sym_id has abs value = symbol_id_offset + 1 for this SDSC, the global
    # position is len(symbols) - len(ordered) + local_idx at registration time.
    # We can recover this because all symbols are in the global list.
    # The absolute value of the first registered sym_id for this SDSC minus 1
    # equals symbol_id_offset at the time of compilation.
    first_abs = abs(ordered[0])  # = symbol_id_offset + 1
    global_start = first_abs - 1  # = symbol_id_offset = index into symbols list
    global_idx = global_start + local_idx
    return f"%sym_{global_idx + 1}"


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
