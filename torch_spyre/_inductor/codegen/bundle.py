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
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.codegen.unroll import unroll_loop_specs
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec
from torch_spyre._inductor.logging_utils import get_inductor_logger


logger = get_inductor_logger("sdsc_compile")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Compiled SDSC entry: (json_dict, base_symbol_values, affine_strides, symbol_kinds)
#   base_symbol_values: list[int] of base HBM byte offsets for this SDSC,
#                       one per registered symbol ID
#   affine_strides:     list[dict] parallel to SDSCSpec.args —
#                       {tiled_sym: stride_bytes} for tiled HBM tensors,
#                       empty dict for non-tiled / lx tensors
#   symbol_kinds:       list[SymbolKind] parallel to base_symbol_values
_CompiledEntry = tuple[Any, list[int], list[dict], list[SymbolKind]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_bundle(
    kernel_name: str,
    output_dir: str,
    specs: Sequence,
    use_symbols: bool | None = None,
    unroll_loops: bool | None = None,
    symbolic_args: bool | None = None,
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
    if symbolic_args is None:
        symbolic_args = _spyre_config.bundle_symbolic_args

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

    # Build a per-symbol kind list from compiled entries (symbolic_args path only).
    symbol_kinds: list[SymbolKind] = []
    if symbolic_args and use_symbols:
        for _, _, _, local_kinds in compiled:
            symbol_kinds.extend(local_kinds)

    # Determine whether a pool parameter is needed (any pool symbol present).
    has_pool = symbolic_args and use_symbols and any(sk.is_pool for sk in symbol_kinds)
    # Indices of kernel-base symbols that become input_arg parameters.
    # Deduplicated by address value: multiple SDSCs may register the same kernel arg
    # address independently (no cross-SDSC dedup in generate_sdsc), so we keep only
    # the first sym_idx for each unique address and map subsequent duplicates to it.
    # kernel_arg_sym_indices: list of sym_idx values, one per unique kernel arg address.
    # kernel_dup_canonical: maps duplicate kernel sym_idx → canonical sym_idx.
    kernel_arg_sym_indices: list[int] = []
    kernel_dup_canonical: dict[int, int] = {}  # duplicate sym_idx → canonical sym_idx
    if symbolic_args and use_symbols:
        seen_kernel_addr: dict[int, int] = {}  # address → canonical sym_idx
        for i, kind_i in enumerate(symbol_kinds):
            if kind_i.kind == "kernel":
                addr = symbols[i]
                if addr not in seen_kernel_addr:
                    seen_kernel_addr[addr] = i
                    kernel_arg_sym_indices.append(i)
                else:
                    kernel_dup_canonical[i] = seen_kernel_addr[addr]

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

        # Function signature when symbolic_args is active:
        #   - optional leading %pool_base_addr param for pool-allocated tensors
        #   - one !sdscbundle.input_arg<index> param per kernel tensor arg, with a
        #     descriptive formal name %arg_{arg_index}_base_addr; the short form
        #     %arg_{arg_index} is used in the body after input_arg_extract
        if symbolic_args and use_symbols and (has_pool or kernel_arg_sym_indices):
            params = []
            if has_pool:
                params.append("%pool_base_addr: !sdscbundle.input_arg<index>")
            for sym_idx in kernel_arg_sym_indices:
                ai = symbol_kinds[sym_idx].arg_index
                params.append(f"%arg_{ai}_base_addr: !sdscbundle.input_arg<index>")
            f.write(f"\tfunc.func @sdsc_bundle({', '.join(params)}) {{\n")
            if has_pool:
                f.write(
                    "\t\t%pool = sdscbundle.input_arg_extract value from"
                    " %pool_base_addr : !sdscbundle.input_arg<index> -> index\n"
                )
            for sym_idx in kernel_arg_sym_indices:
                ai = symbol_kinds[sym_idx].arg_index
                f.write(
                    f"\t\t%arg_{ai} = sdscbundle.input_arg_extract value from"
                    f" %arg_{ai}_base_addr : !sdscbundle.input_arg<index> -> index\n"
                )
        else:
            f.write("\tfunc.func @sdsc_bundle() {\n")

        # Standard loop constants (only emitted when there are loops).
        if loop_bounds:
            f.write("\t\t%c0 = arith.constant 0 : index\n")
            f.write("\t\t%c1 = arith.constant 1 : index\n")
            for lb_idx, lb in enumerate(loop_bounds):
                f.write(f"\t\t%loop_bound_{lb_idx} = {_mlir_count_value(lb)}\n")

        # Emit one declaration per symbol (symbolic_args path):
        #   - "kernel"          → skipped; already a function param + extract op above
        #   - "kernel_derived"  → arith.addi %arg_{arg_index}, <per_core_offset>
        #                         deduped by (arg_index, offset) pair
        #   - "pool"            → arith.addi %pool, <pool_offset>
        #                         deduped by pool offset value
        #   - anything else     → arith.constant (use_symbols=False or non-symbolic path)
        # All kernel sym indices to skip during emission (canonical + duplicates).
        kernel_arg_sym_set = set(kernel_arg_sym_indices) | set(kernel_dup_canonical)
        # Map kernel sym_idx → arg_index for SSA name generation.
        # Duplicate kernel sym indices inherit the arg_index of their canonical.
        kernel_sym_to_arg_idx: dict[int, int] = {
            sym_idx: symbol_kinds[sym_idx].arg_index
            for sym_idx in kernel_arg_sym_indices
        }
        for dup_idx, canon_idx in kernel_dup_canonical.items():
            if canon_idx in kernel_sym_to_arg_idx:
                kernel_sym_to_arg_idx[dup_idx] = kernel_sym_to_arg_idx[canon_idx]
        # sym_canonical[sym_idx] → canonical SSA name for derived/pool symbols (deduped).
        # Also pre-populate duplicate kernel sym_idx entries with their canonical extracted name.
        sym_canonical: dict[int, str] = {
            dup_idx: f"%arg_{kernel_sym_to_arg_idx[dup_idx]}"
            for dup_idx in kernel_dup_canonical
            if dup_idx in kernel_sym_to_arg_idx
        }
        # derived_addi_emitted[(arg_index, offset)] → SSA name already emitted
        derived_addi_emitted: dict[tuple[int, int], str] = {}
        # pool_addi_emitted[pool_offset_value] → SSA name already emitted
        pool_addi_emitted: dict[int, str] = {}

        for sym_idx, value in enumerate(symbols):
            if sym_idx in kernel_arg_sym_set:
                continue  # replaced by function parameter + extract op (or duplicate)
            sk: SymbolKind | None = symbol_kinds[sym_idx] if symbol_kinds else None
            if sk is not None and sk.is_derived:
                base_ai = kernel_sym_to_arg_idx.get(sk.base_sym_idx)
                if base_ai is not None:
                    key = (base_ai, sk.offset)
                    if key not in derived_addi_emitted:
                        offset_ssa = f"%arg_{base_ai}_core_offset_{sk.offset}"
                        addi_ssa = f"%arg_{base_ai}_core_{sk.offset}"
                        f.write(
                            f"\t\t{offset_ssa} = arith.constant {sk.offset} : index\n"
                        )
                        f.write(
                            f"\t\t{addi_ssa} = arith.addi"
                            f" %arg_{base_ai}, {offset_ssa} : index\n"
                        )
                        derived_addi_emitted[key] = addi_ssa
                    sym_canonical[sym_idx] = derived_addi_emitted[key]
                else:
                    f.write(
                        f"\t\t%sym_{sym_idx + 1} = arith.constant {value} : index\n"
                    )
            elif sk is not None and sk.is_pool:
                if value not in pool_addi_emitted:
                    offset_ssa = f"%pool_offset_{value}"
                    addi_ssa = f"%pool_addr_{value}"
                    f.write(f"\t\t{offset_ssa} = arith.constant {value} : index\n")
                    f.write(
                        f"\t\t{addi_ssa} = arith.addi %pool, {offset_ssa} : index\n"
                    )
                    pool_addi_emitted[value] = addi_ssa
                sym_canonical[sym_idx] = pool_addi_emitted[value]
            else:
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
            symbolic_args=symbolic_args,
            kernel_sym_to_arg_idx=kernel_sym_to_arg_idx,
            sym_canonical=sym_canonical,
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
            sdsc_json, local_sym_values, affine_strides, local_symbol_kinds = (
                compile_op_spec(
                    idx,
                    entry,
                    symbols,
                    symbol_id_offset_counter[0],
                    use_symbols=use_symbols,
                )
            )
            symbol_id_offset_counter[0] += len(local_sym_values)
            compiled.append(
                (sdsc_json, local_sym_values, affine_strides, local_symbol_kinds)
            )
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
            _, _, affine_strides, _ = next(compiled_iter)
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
    symbolic_args: bool = False,
    kernel_sym_to_arg_idx: dict | None = None,
    sym_canonical: dict | None = None,
) -> None:
    """Recursively emit MLIR ops for specs into file f."""
    if kernel_sym_to_arg_idx is None:
        kernel_sym_to_arg_idx = {}
    if sym_canonical is None:
        sym_canonical = {}

    # Map from 0-based symbol index to the short SSA name for kernel-arg symbols.
    # sym_idx → %arg_{arg_index}  (the result of input_arg_extract in the function body)
    kernel_arg_sym_to_name: dict[int, str] = {
        sym_idx: f"%arg_{ai}" for sym_idx, ai in kernel_sym_to_arg_idx.items()
    }

    def _resolve_sym(sid: int) -> str:
        # sid is a negative symbol ID; abs(sid)-1 is the 0-based index into symbols[].
        sym_idx = abs(sid) - 1
        if symbolic_args:
            if sym_idx in kernel_arg_sym_to_name:
                return kernel_arg_sym_to_name[sym_idx]
            if sym_idx in sym_canonical:
                return sym_canonical[sym_idx]
        return f"%sym_{abs(sid)}"

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
                symbolic_args=symbolic_args,
                kernel_sym_to_arg_idx=kernel_sym_to_arg_idx,
                sym_canonical=sym_canonical,
            )
            f.write(f"{tab}}}\n")

        elif isinstance(entry, OpSpec):
            sdsc_json, local_sym_values, affine_strides, _ = next(compiled_iter)
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
                    base_addr_name = _resolve_sym(base_sym_id)
                    loop_var_str = ", ".join(loop_vars)
                    f.write(
                        f"{tab}{addr_name} = affine.apply #map_{map_idx}"
                        f"({loop_var_str})[{base_addr_name}]\n"
                    )
                    sym_id_to_operand[base_sym_id] = addr_name

            # Each operand position matches one symbol_id entry.
            # Tiled sym_ids use the %addr_N computed above; others use %sym_N.
            operands = [
                sym_id_to_operand.get(sid, _resolve_sym(sid)) for sid in symbol_ids
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
