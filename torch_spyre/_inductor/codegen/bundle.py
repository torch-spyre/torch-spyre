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
import logging
import os
from collections.abc import Sequence
from typing import Any

import sympy

from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, format_op_spec_list
from torch_spyre._inductor.logging_utils import get_inductor_logger


logger = get_inductor_logger("sdsc_compile")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Compiled SDSC entry: (json_dict, symbol_values, affine_strides, symbol_kinds)
#   symbol_values:  list[int] of registered symbol values for this SDSC,
#                   one per symbol ID.  Values are HBM byte addresses for
#                   derived/pool symbols; arg_index sentinels for kernel
#                   symbols on the symbolic-args path.  Only len() is used
#                   by bundle.py; individual values are resolved via symbols[].
#   affine_strides: list[list[dict]] — per tensor, per loop-nesting level
#                   (outermost first).  Each inner dict maps
#                   tiled_sym -> stride_bytes for that level's symbols.
#                   [{} for _ in tiled_symbols] for non-tiled / lx tensors
#                   (one empty dict per level, preserving the level count).
#   symbol_kinds:   list[SymbolKind] parallel to symbol_values
_CompiledEntry = tuple[Any, list[int], list[list[dict]], list[SymbolKind]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_bundle(
    kernel_name: str,
    output_dir: str,
    specs: Sequence,
    use_symbols: bool | None = None,
):
    """Output the SDSC Bundle for the OpSpecs in output_dir.

    ``specs`` is a list of ``OpSpec | LoopSpec`` entries (nested ``LoopSpec``
    entries are supported).

    ``use_symbols`` controls whether HBM tensor addresses are emitted as
    runtime symbols (``%sym_N`` constants) in ``bundle.mlir``.
    When ``None`` (the default) the value is
    read from ``config.bundle_symbolic_args``.

     Dimension symbols (from ``mark_dynamic``) always produce
    ``!sdscbundle.input_arg<index, granularity=G, max_value=M>`` parameters
    independent of ``use_symbols``.
    """
    if use_symbols is None:
        use_symbols = _spyre_config.bundle_symbolic_args

    specs_list: list = list(specs)

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "OP SPECS FOR BUNDLE GENERATION\n%s",
            format_op_spec_list(specs_list),
        )

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
    # A stride_key is a tuple of stride values in outermost-first level order.
    # Strides from each level are appended in level order; within a level, in
    # symbol dict insertion order.  The corresponding loop-var indices are built
    # from the explicit level index, so each stride is mapped to the correct
    # loop variable regardless of nesting depth.
    #
    # affine_map_loop_var_indices: parallel to compiled, per op a list of
    # per-tensor loop-var index lists.  Each inner list records which positions
    # in the enclosing loop_vars list correspond to the strides in stride_key,
    # one entry per non-zero stride in outermost-first level order.
    # _emit_specs uses this to pass only the relevant loop vars to affine.apply.
    affine_map_index: dict[tuple, int] = {}
    affine_map_loop_var_indices: list[list[list[int]]] = []
    _collect_affine_maps(
        specs_list, iter(compiled), [], affine_map_index, affine_map_loop_var_indices
    )

    compiled_iter = iter(compiled)
    addr_counter = [0]

    # Flatten symbol kinds from all SDSCs. sym_idx_to_dim_origin records
    # (sdsc_idx, ordinal) for each dimension symbol to generate its MLIR name.
    symbol_kinds: list[SymbolKind] = []
    sym_idx_to_dim_origin: dict[int, tuple[int, int]] = {}
    for sdsc_idx, (_, _, _, local_kinds) in enumerate(compiled):
        local_dim_ordinal = 0
        for lk in local_kinds:
            if lk.is_dimension:
                local_dim_ordinal += 1
                sym_idx_to_dim_origin[len(symbol_kinds)] = (
                    sdsc_idx,
                    local_dim_ordinal,
                )
            elif not use_symbols:
                raise AssertionError(
                    "use_symbols=False but compute_op_spec registered a "
                    f"non-dimension SymbolKind ({lk.kind!r}); address-symbol."
                )
            symbol_kinds.append(lk)

    # Determine whether a pool parameter is needed (any pool symbol present).
    has_pool = use_symbols and any(sk.is_pool for sk in symbol_kinds)
    # Indices of kernel-base symbols that become input_arg parameters.
    # Deduplicated by arg_index: multiple SDSCs operating on different slices of
    # the same logical tensor arg share one function parameter (the first-seen
    # sym_idx, which corresponds to core-0 / the lowest address).  Dedup by
    # address alone is insufficient — different slices have different addresses
    # but the same arg_index and must map to one %arg_{ai}_base_addr param.
    # kernel_arg_sym_indices: list of sym_idx values, one per unique arg_index.
    # kernel_dup_canonical: maps duplicate kernel sym_idx → canonical sym_idx.
    kernel_arg_sym_indices: list[int] = []
    kernel_dup_canonical: dict[int, int] = {}  # duplicate sym_idx → canonical sym_idx
    if use_symbols:
        seen_kernel_arg_index: dict[int, int] = {}  # arg_index → canonical sym_idx
        for i, kind_i in enumerate(symbol_kinds):
            if kind_i.kind == "kernel":
                ai = kind_i.arg_index
                if ai not in seen_kernel_arg_index:
                    seen_kernel_arg_index[ai] = i
                    kernel_arg_sym_indices.append(i)
                else:
                    kernel_dup_canonical[i] = seen_kernel_arg_index[ai]
        # Sort by arg_index so the function signature matches the positional order
        # that call_kernel passes tensors to .run().
        kernel_arg_sym_indices.sort(key=lambda idx: symbol_kinds[idx].arg_index)

    # Deduplicate dimension symbols by pytorch_sym (same shape var may appear
    # in every SDSC with a different local ID). Runs regardless of use_symbols.
    dimension_sym_indices: list[int] = []
    dimension_dup_canonical: dict[int, int] = {}  # dup sym_idx → canonical sym_idx
    seen_dim_sym: dict[str, int] = {}  # pytorch_sym → canonical sym_idx
    for i, kind_i in enumerate(symbol_kinds):
        if kind_i.is_dimension:
            dim_sym_key = kind_i.pytorch_sym
            if dim_sym_key not in seen_dim_sym:
                seen_dim_sym[dim_sym_key] = i
                dimension_sym_indices.append(i)
            else:
                dimension_dup_canonical[i] = seen_dim_sym[dim_sym_key]
    # MLIR name for each canonical dimension symbol, e.g. "%sym_0_1".
    dim_param_names: dict[int, str] = {
        sym_idx: (
            f"%sym_{sym_idx_to_dim_origin[sym_idx][0]}"
            f"_{sym_idx_to_dim_origin[sym_idx][1]}"
        )
        for sym_idx in dimension_sym_indices
    }

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

        # Function signature:
        #   - optional leading %pool_base_addr param for pool-allocated tensors
        #     (only when use_symbols=True)
        #   - one !sdscbundle.input_arg<index> param per kernel tensor arg
        #     (only when use_symbols=True)
        #   - one !sdscbundle.input_arg<index, granularity=G, max_value=M> param
        #     per unique dynamic-shape (mark_dynamic) symbol; emitted whenever
        #     present, independent of use_symbols (which only governs the
        #     pool/kernel-address params above).
        if has_pool or kernel_arg_sym_indices or dimension_sym_indices:
            params = []
            if has_pool:
                params.append("%pool_base_addr: !sdscbundle.input_arg<index>")
            for sym_idx in kernel_arg_sym_indices:
                ai = symbol_kinds[sym_idx].arg_index
                params.append(f"%arg_{ai}_base_addr: !sdscbundle.input_arg<index>")
            for sym_idx in dimension_sym_indices:
                dim_sk = symbol_kinds[sym_idx]
                params.append(
                    f"{dim_param_names[sym_idx]}_base: {_dim_input_arg_type(dim_sk)}"
                )
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
            for sym_idx in dimension_sym_indices:
                dim_sk = symbol_kinds[sym_idx]
                name = dim_param_names[sym_idx]
                f.write(
                    f"\t\t{name} = sdscbundle.input_arg_extract value from"
                    f" {name}_base : {_dim_input_arg_type(dim_sk)} -> index\n"
                )
        else:
            f.write("\tfunc.func @sdsc_bundle() {\n")

        # Standard loop constants (only emitted when there are loops).
        if loop_bounds:
            f.write("\t\t%c0 = arith.constant 0 : index\n")
            f.write("\t\t%c1 = arith.constant 1 : index\n")
            for lb_idx, lb in enumerate(loop_bounds):
                f.write(f"\t\t%loop_bound_{lb_idx} = {_mlir_count_value(lb)}\n")

        # Emit one declaration per symbol (use_symbols path):
        #   - "kernel"          → skipped; already a function param + extract op above
        #   - "kernel_slice"    → arith.addi %arg_{arg_index}, <slice_offset_bytes>
        #                         deduped by (arg_index, slice_offset) pair;
        #                         produces the SSA "sliced base" that per-core offsets
        #                         and sdsc_execute args reference for sliced tensors
        #   - "kernel_derived"  → arith.addi <sliced_base_ssa>, <per_core_offset>
        #                         deduped by (sliced_base_ssa, per_core_offset)
        #   - "pool"            → arith.addi %pool, <pool_offset>
        #                         deduped by pool offset value
        #   - "dimension"       → skipped; replaced by function parameter above,
        #                         resolved at use-sites via sym_canonical
        #   - anything else     → arith.constant (non-symbolic path)
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
        # sym_canonical[sym_idx] → canonical SSA name for derived/pool/slice symbols.
        # Pre-populate duplicate kernel sym_idx entries with their canonical extracted name.
        sym_canonical: dict[int, str] = {
            dup_idx: f"%arg_{kernel_sym_to_arg_idx[dup_idx]}"
            for dup_idx in kernel_dup_canonical
            if dup_idx in kernel_sym_to_arg_idx
        }
        # Dimension symbols resolve to their input_arg_extract result.
        sym_canonical.update(
            (sym_idx, dim_param_names[sym_idx]) for sym_idx in dimension_sym_indices
        )
        sym_canonical.update(
            (dup_idx, dim_param_names[canon_idx])
            for dup_idx, canon_idx in dimension_dup_canonical.items()
        )
        # slice_addi_emitted[(arg_index, slice_offset)] → SSA name for sliced base
        slice_addi_emitted: dict[tuple[int, int], str] = {}
        # derived_addi_emitted[(sliced_base_ssa, per_core_offset)] → SSA name
        derived_addi_emitted: dict[tuple[str, int], str] = {}
        # pool_addi_emitted[pool_offset_value] → SSA name already emitted
        pool_addi_emitted: dict[int, str] = {}

        for sym_idx, value in enumerate(symbols):
            if sym_idx in kernel_arg_sym_set:
                continue  # replaced by function parameter + extract op (or duplicate)
            sk: SymbolKind | None = symbol_kinds[sym_idx] if symbol_kinds else None
            if sk is not None and sk.kind == "kernel_slice":
                ai = sk.arg_index
                sl = sk.offset  # slice offset in bytes
                key = (ai, sl)
                if key not in slice_addi_emitted:
                    slice_offset_ssa = f"%arg_{ai}_slice_offset_{sl}"
                    sliced_base_ssa = f"%arg_{ai}_slice_{sl}"
                    f.write(f"\t\t{slice_offset_ssa} = arith.constant {sl} : index\n")
                    f.write(
                        f"\t\t{sliced_base_ssa} = arith.addi"
                        f" %arg_{ai}, {slice_offset_ssa} : index\n"
                    )
                    slice_addi_emitted[key] = sliced_base_ssa
                sym_canonical[sym_idx] = slice_addi_emitted[key]
            elif sk is not None and sk.is_derived:
                # Resolve the SSA name of the sliced base that this core offset builds on.
                base_sym_idx = sk.base_sym_idx
                if base_sym_idx in sym_canonical:
                    sliced_base_ssa = sym_canonical[base_sym_idx]
                elif base_sym_idx in kernel_arg_sym_indices:
                    # slice_offset == 0: sliced base == raw arg extract (%arg_N)
                    ai = symbol_kinds[base_sym_idx].arg_index
                    sliced_base_ssa = f"%arg_{ai}"
                elif base_sym_idx in kernel_dup_canonical:
                    canon = kernel_dup_canonical[base_sym_idx]
                    ai = kernel_sym_to_arg_idx.get(
                        canon, symbol_kinds[base_sym_idx].arg_index
                    )
                    sliced_base_ssa = f"%arg_{ai}"
                else:
                    sliced_base_ssa = None
                if sliced_base_ssa is not None:
                    key_d = (sliced_base_ssa, sk.offset)
                    if key_d not in derived_addi_emitted:
                        offset_ssa = f"%{sliced_base_ssa[1:]}_core_offset_{sk.offset}"
                        addi_ssa = f"%{sliced_base_ssa[1:]}_core_{sk.offset}"
                        f.write(
                            f"\t\t{offset_ssa} = arith.constant {sk.offset} : index\n"
                        )
                        f.write(
                            f"\t\t{addi_ssa} = arith.addi"
                            f" {sliced_base_ssa}, {offset_ssa} : index\n"
                        )
                        derived_addi_emitted[key_d] = addi_ssa
                    sym_canonical[sym_idx] = derived_addi_emitted[key_d]
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
            elif sk is not None and sk.is_dimension:
                continue  # replaced by function parameter; resolved via sym_canonical
            else:
                f.write(f"\t\t%sym_{sym_idx + 1} = arith.constant {value} : index\n")

        # Recursive body emission.
        # affine_map_lv_iter spans the entire spec tree (one entry per OpSpec,
        # in the same depth-first order as compiled_iter) and is consumed by
        # _emit_specs across all recursive calls — not reset per loop level.
        loop_bound_idx = [0]
        affine_map_lv_iter = iter(affine_map_loop_var_indices)
        _emit_specs(
            specs_list,
            compiled_iter,
            loop_bounds,
            loop_bound_idx,
            affine_map_index,
            affine_map_lv_iter,
            addr_counter,
            [],
            f,
            indent=2,
            use_symbols=use_symbols,
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
    loop_var_indices_out: list,
) -> None:
    """Walk the spec tree and register unique affine stride keys.

    Populates ``affine_map_index`` (stride_key -> map_idx) and appends one
    entry per OpSpec to ``loop_var_indices_out``.  Each entry is a list of
    per-tensor index lists: ``loop_var_indices_out[op_idx][tensor_idx]`` is
    the list of loop-var positions (into the enclosing ``loop_vars`` list at
    emit time) that correspond to the strides in the tensor's stride_key,
    in outermost-first level order.

    ``affine_strides[tensor_idx]`` is a list of dicts, one per loop-nesting
    level (outermost first).  We iterate over levels explicitly and use
    ``loop_var_depth[level_idx]`` to find the correct loop variable for each
    level's strides — no counting from the end.
    """
    for entry in specs:
        if isinstance(entry, LoopSpec):
            _collect_affine_maps(
                entry.body,
                compiled_iter,
                loop_var_depth + [len(loop_var_depth)],
                affine_map_index,
                loop_var_indices_out,
            )
        elif isinstance(entry, OpSpec):
            _, _, affine_strides, _ = next(compiled_iter)
            per_tensor_lv_indices: list[list[int]] = []
            for per_level_strides in affine_strides:
                # per_level_strides is list[dict], one dict per level (outermost first).
                # Build stride_key and lv_indices by iterating levels explicitly.
                stride_vals: list[int] = []
                lv_idxs: list[int] = []
                for level_idx, level_strides in enumerate(per_level_strides):
                    if not level_strides:
                        continue
                    assert level_idx < len(loop_var_depth), (
                        f"affine_strides has {len(per_level_strides)} levels but "
                        f"only {len(loop_var_depth)} enclosing loop(s); "
                        "create_op_spec built more tiled_syms levels than LoopSpec ancestors"
                    )
                    lv = loop_var_depth[level_idx]
                    for stride in level_strides.values():
                        stride_vals.append(stride)
                        lv_idxs.append(lv)
                if not stride_vals:
                    per_tensor_lv_indices.append([])
                    continue
                stride_key = tuple(stride_vals)
                if stride_key not in affine_map_index:
                    affine_map_index[stride_key] = len(affine_map_index)
                per_tensor_lv_indices.append(lv_idxs)
            loop_var_indices_out.append(per_tensor_lv_indices)


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


def _dim_input_arg_type(dim_sk: SymbolKind) -> str:
    """MLIR input_arg type string for a dimension symbol.

    Shared by the function-parameter declaration and the corresponding
    input_arg_extract op so the two can't drift out of sync.
    """
    return (
        f"!sdscbundle.input_arg<index, granularity={dim_sk.granularity}, "
        f"max_value={dim_sk.max_value}>"
    )


def _emit_specs(
    specs: list,
    compiled_iter,
    loop_bounds: list,
    loop_bound_idx: list,
    affine_map_index: dict,
    affine_map_lv_iter,
    addr_counter: list,
    loop_vars: list,
    f,
    indent: int,
    use_symbols: bool = False,
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
        # Both dicts are safe to check unconditionally — empty when their feature is off.
        sym_idx = abs(sid) - 1
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
                affine_map_lv_iter,
                addr_counter,
                loop_vars + [loop_var],
                f,
                indent + 1,
                use_symbols=use_symbols,
                kernel_sym_to_arg_idx=kernel_sym_to_arg_idx,
                sym_canonical=sym_canonical,
            )
            f.write(f"{tab}}}\n")

        elif isinstance(entry, OpSpec):
            sdsc_json, local_sym_values, affine_strides, _ = next(compiled_iter)
            # Per-tensor loop-var index lists: which positions in the enclosing
            # loop_vars list correspond to the strides for each tensor.
            per_tensor_lv_indices: list[list[int]] = next(affine_map_lv_iter)

            # Determine the JSON filename from the sdsc_json key.
            sdsc_name = next(iter(sdsc_json))
            sdsc_idx = sdsc_name.split("_")[0]
            sdsc_filename = f"sdsc_{sdsc_idx}.json"

            # Extract symbol_ids from the negative IDs stored in the JSON
            # (unique, in registration order).
            symbol_ids = _extract_symbol_ids(sdsc_json)

            # Build affine.apply ops for tiled tensors, tracking which
            # symbol IDs have been upgraded to per-iteration %addr_N names.
            # affine_strides[tensor_idx] is list[dict] (per level, outermost first).
            sym_id_to_operand: dict[int, str] = {}
            for tensor_idx, per_level_strides in enumerate(affine_strides):
                # Flatten per-level strides to build the stride_key in the same
                # outermost-first order used by _collect_affine_maps.
                flat_strides: list[int] = [
                    stride
                    for level_strides in per_level_strides
                    for stride in level_strides.values()
                ]
                if not flat_strides:
                    continue
                num_cores = _sdsc_num_cores(sdsc_json)
                for c in range(num_cores):
                    base_sym_id = _get_tensor_core_sym_id(sdsc_json, tensor_idx, c)
                    if base_sym_id is None or base_sym_id in sym_id_to_operand:
                        continue
                    stride_key = tuple(flat_strides)
                    map_idx = affine_map_index[stride_key]
                    addr_name = f"%addr_{addr_counter[0]}"
                    addr_counter[0] += 1
                    base_addr_name = _resolve_sym(base_sym_id)
                    # lv_indices[tensor_idx] was built by _collect_affine_maps using
                    # explicit level indexing — each entry is the loop_vars position
                    # for the corresponding stride in stride_key.
                    lv_indices = per_tensor_lv_indices[tensor_idx]
                    apply_loop_vars = [loop_vars[i] for i in lv_indices]
                    loop_var_str = ", ".join(apply_loop_vars)
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
            symbol_ids_str = ", ".join(str(i) for i in symbol_ids)
            f.write(
                f"{tab}sdscbundle.sdsc_execute ({operand_str}) "
                f'{{sdsc_filename="{sdsc_filename}", '
                f'"symbol_ids"=[{symbol_ids_str}]}}\n'
            )


def _extract_symbol_ids(sdsc_json: dict) -> list[int]:
    """Extract all negative symbol IDs from an SDSC JSON, dimension IDs first.

    Dimension IDs (``dimToSymbolMapping_``) have lower-magnitude negatives than
    HBM address IDs, so scanning them first keeps ``ids`` sorted naturally.
    """
    ids: list[int] = []
    seen: set[int] = set()
    for top_val in sdsc_json.values():
        for dsc_entry in top_val.get("dscs_", []):
            for op_val in dsc_entry.values():
                for dim_syms in op_val.get("dimToSymbolMapping_", {}).values():
                    for v in dim_syms:
                        sym_id = int(v)
                        if sym_id < 0 and sym_id not in seen:
                            ids.append(sym_id)
                            seen.add(sym_id)
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
