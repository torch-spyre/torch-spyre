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

import dataclasses

from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.pass_utils import decompose_tiled_count
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.constants import MAX_POOL_SIZE_BYTES
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.op_spec import (
    LoopSpec,
    OpSpec,
    format_op_spec_list,
    walk_loop_specs,
)
from torch_spyre._inductor.op_spec_validation import validate_op_specs


logger = get_inductor_logger("sdsc_compile")
sdsc_log = get_inductor_logger("sdsc")

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
#   cached_json:    the JSON from the first (canonical) compilation of this SDSC,
#                   used for the sdsc_filename and printed symbol_ids in
#                   sdsc_execute.  Equals sdsc_json on a cache miss; on a hit it
#                   carries the original symbol IDs while sdsc_json carries the
#                   fresh ones used for operand resolution.
_CompiledEntry = tuple[Any, list[int], list[list[dict]], list[SymbolKind], Any]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _merge_loop_symbol_maps(specs, attr: str) -> dict:
    """Union one per-symbol map across every LoopSpec in the tree.

    Both ``count_symbol_bounds`` and ``count_symbol_sources`` are per-loop, but a
    bundle declares ONE parameter per dimension, so the maps have to agree.

    Conflicting values raise rather than silently picking one. For bounds that
    means two loops disagree about a dimension's max or tile size, and the
    parameter would be wrong for one of them. For sources it means two loops read
    the same dimension off different launch arguments, and the parameter would
    bind from the wrong tensor.

    The bounds conflict becomes reachable the moment tile sizes are chosen per
    kernel, and the right answer then is the lcm of the steps together with the
    rule that every tile size divides the declared granularity. Until that
    exists, refusing is correct and deliberately stricter than necessary: it also
    rejects the legal case where both steps divide the granularity.

    Values are normalised to tuples, because the reload path reconstructs them
    from generated source and a sequence literal there is not required to come
    back as the same type it went out as.
    """
    merged: dict = {}
    for loop in walk_loop_specs(specs):
        for sym_name, raw in (getattr(loop, attr, None) or {}).items():
            value = tuple(raw)
            existing = merged.get(sym_name)
            if existing is not None and existing != value:
                raise NotImplementedError(
                    f"two loops in one kernel disagree about {sym_name}: "
                    f"{attr} is {existing} on one and {value} on another. A "
                    f"bundle declares one parameter per dimension, so it cannot "
                    f"serve both. See _merge_loop_symbol_maps."
                )
            merged[sym_name] = value
    return merged


def generate_bundle(
    kernel_name: str,
    output_dir: str,
    specs: Sequence,
    pool_size: int = 0,
) -> list[SymbolKind]:
    """Output the SDSC Bundle for the OpSpecs in output_dir.

    ``specs`` is a list of ``OpSpec | LoopSpec`` entries (nested ``LoopSpec``
    entries are supported).

    HBM tensor addresses are emitted as runtime symbols (``%sym_N``
    constants) in ``bundle.mlir``. Dimension symbols (from ``mark_dynamic``)
    always produce
    ``!sdscbundle.input_arg<index, granularity=G, max_value=M>`` parameters.

    Requires ``config.bundle_symbolic_args`` to be True: the SDSC path
    (this function) unconditionally emits symbolic addresses, but
    ``spyre_kernel.py``/``hbm_pool_planning.py`` still bake absolute
    addresses into tensor allocations when the flag is False (reserved for
    the KTIR emitter path, which is gated separately). Running this
    function with the flag off would silently miscompile addresses rather
    than error, so it's rejected up front instead.

    ``pool_size`` is the byte count for this bundle's HBM pool (from
    ``hbm_pool_planning.py`` via ``SpyreKernel.pool_size``). Ignored unless
    a pool symbol is present in ``specs``. When a pool symbol is present,
    how ``%pool`` is produced depends on
    ``config.frontend_pool_allocation``:

    - ``False`` (default): emitted as ``%pool = sdscbundle.device_mem_allocate
      <pool_size> bytes : index`` as the first statement of the bundle
      body — there is no ``%pool_base_addr`` function parameter.
    - ``True``: emitted as a ``%pool_base_addr: !sdscbundle.input_arg<index>``
      function parameter (ordered first, before any
      ``kernel_arg_sym_indices``/``dimension_sym_indices`` params) plus an
      ``sdscbundle.input_arg_extract value from %pool_base_addr`` statement
      producing ``%pool`` as the first statement of the bundle body. The
      caller (``SpyreKernel.call_kernel()``) is then responsible for
      allocating a real pool tensor and passing its address as this
      argument.

    Either way, ``%pool`` is an ordinary ``index``-typed SSA value; every
    per-buffer ``arith.addi %pool, <offset>`` emission downstream is
    identical in both modes.
    """
    if not _spyre_config.bundle_symbolic_args:
        raise AssertionError(
            "generate_bundle() requires config.bundle_symbolic_args=True "
            "(BUNDLE_SYMBOLIC_ARGS=1). The SDSC bundle path always emits "
            "symbolic HBM addresses; baked absolute addresses "
            "(bundle_symbolic_args=False) are only supported on the KTIR "
            "emitter path (config.ktir_emitter=True)."
        )

    specs_list: list = list(specs)

    if _spyre_config.validate_op_specs:
        validate_op_specs(specs_list, stage="before_bundle_generation")
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

    sdsc_cache_counts: list[int] | None = None
    if _spyre_config.sdsc_cache:
        sdsc_cache_counts = [0, 0]  # [hits, misses]
    _compile_specs(
        specs_list,
        symbols,
        compiled,
        sdsc_counter,
        symbol_id_offset_counter,
        output_dir,
        sdsc_cache={} if _spyre_config.sdsc_cache else None,
        _sdsc_cache_counts=sdsc_cache_counts,
    )
    if sdsc_cache_counts is not None:
        hits, misses = sdsc_cache_counts
        logger.info(
            "sdsc_cache: %d/%d ops reused an existing sdsc file (%d unique)",
            hits,
            hits + misses,
            misses,
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
    for sdsc_idx, (_, _, _, local_kinds, _) in enumerate(compiled):
        local_dim_ordinal = 0
        for lk in local_kinds:
            if lk.is_dimension:
                local_dim_ordinal += 1
                sym_idx_to_dim_origin[len(symbol_kinds)] = (
                    sdsc_idx,
                    local_dim_ordinal,
                )
            symbol_kinds.append(lk)

    # Determine whether a pool parameter is needed (any pool symbol present).
    has_pool = any(sk.is_pool for sk in symbol_kinds)
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
    # in every SDSC with a different local ID).
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

    # Loop-bound dimensions come from the SPEC TREE, not from symbol_kinds. That
    # distinction is the whole reason this route exists: symbol_kinds is
    # flattened from the compiled SDSCs above, and a loop dimension never enters
    # an SDSC, so scanning it for one finds nothing, always. The dimension lives
    # on the LoopSpec that needs it, as the bounds the scheduler resolved and the
    # source it placed.
    #
    # Deduplicated across loops by symbol name, because the same shape variable
    # reaches every loop that tiles on it and one bundle parameter serves them
    # all. The name is the key for the reason it is everywhere else on this path:
    # it is what survives the reload.
    loop_dim_bounds = _merge_loop_symbol_maps(specs_list, "count_symbol_bounds")
    loop_dim_sources = _merge_loop_symbol_maps(specs_list, "count_symbol_sources")
    loop_dim_ssa: dict[str, str] = {
        sym_name: f"%dim_{sym_name}" for sym_name in loop_dim_bounds
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
        #   - when config.frontend_pool_allocation and a pool symbol is
        #     present: one %pool_base_addr !sdscbundle.input_arg<index>
        #     param, emitted first
        #   - one !sdscbundle.input_arg<index> param per kernel tensor arg
        #   - one !sdscbundle.input_arg<index, granularity=G, max_value=M> param
        #     per unique dynamic-shape (mark_dynamic) symbol; emitted whenever
        #     present.
        # Otherwise (default), pool allocation is emitted in the body as
        # device_mem_allocate, not as a function parameter.
        emit_pool_param = has_pool and _spyre_config.frontend_pool_allocation
        # Built in lock-step with the params list so the two can never diverge.
        # Order: pool (when frontend_pool_allocation), kernel addresses, dimensions.
        param_symbol_kinds: list[SymbolKind] = []
        if (
            emit_pool_param
            or kernel_arg_sym_indices
            or dimension_sym_indices
            or loop_dim_bounds
        ):
            params = []
            if emit_pool_param:
                params.append("%pool_base_addr: !sdscbundle.input_arg<index>")
                param_symbol_kinds.append(SymbolKind.pool())
            for sym_idx in kernel_arg_sym_indices:
                ai = symbol_kinds[sym_idx].arg_index
                params.append(f"%arg_{ai}_base_addr: !sdscbundle.input_arg<index>")
                param_symbol_kinds.append(symbol_kinds[sym_idx])
            for sym_idx in dimension_sym_indices:
                dim_sk = symbol_kinds[sym_idx]
                params.append(
                    f"{dim_param_names[sym_idx]}_base: {_dim_input_arg_type(dim_sk)}"
                )
                param_symbol_kinds.append(symbol_kinds[sym_idx])
            # Last, so adding a loop dimension never shifts an existing
            # parameter's position: the runtime fills these slots by order.
            #
            # The SymbolKind is CONSTRUCTED here rather than looked up, because
            # this is the first point where everything it needs is in one place:
            # the bounds from the scheduler, the source it placed, and the
            # parameter position. param_symbol_kinds is what generate_bundle
            # returns and what the launch path reads to build its argument
            # payload, so a loop dimension has to appear in it or the runtime has
            # nothing telling it to read a size rather than an address.
            for sym_name, (max_value, granularity) in loop_dim_bounds.items():
                source = loop_dim_sources.get(sym_name)
                if source is None:
                    raise NotImplementedError(
                        f"symbolic loop dimension {sym_name} has bounds "
                        f"{(max_value, granularity)} but no source, so nothing "
                        f"at launch knows which tensor dimension to bind into "
                        f"its parameter. SpyreKernel._resolve_loop_dimension_"
                        f"sources is where that is worked out, and it logs the "
                        f"launch arguments it looked at. Refusing to declare a "
                        f"parameter nothing can fill, rather than binding a "
                        f"wrong number on every launch."
                    )
                arg_index, dim_index = source
                dim_kind = SymbolKind.loop_dimension(
                    granularity=granularity,
                    max_value=max_value,
                    pytorch_sym=sym_name,
                    arg_index=arg_index,
                    dim_index=dim_index,
                )
                params.append(
                    f"{loop_dim_ssa[sym_name]}_base: {_dim_input_arg_type(dim_kind)}"
                )
                param_symbol_kinds.append(dim_kind)
            f.write(f"\tfunc.func @sdsc_bundle({', '.join(params)}) {{\n")
        else:
            f.write("\tfunc.func @sdsc_bundle() {\n")

        assert not has_pool or 0 < pool_size <= MAX_POOL_SIZE_BYTES, (
            f"generate_bundle: pool_size={pool_size} out of range "
            f"(0, {MAX_POOL_SIZE_BYTES}] for a bundle with a pool symbol present"
        )
        if has_pool:
            if _spyre_config.frontend_pool_allocation:
                f.write(
                    "\t\t%pool = sdscbundle.input_arg_extract value from"
                    " %pool_base_addr : !sdscbundle.input_arg<index> -> index\n"
                )
            else:
                f.write(
                    f"\t\t%pool = sdscbundle.device_mem_allocate {pool_size} bytes"
                    " : index\n"
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

        # Before the loop constants on purpose: a symbolic bound IS one of these
        # SSA values, so it has to be in scope by the time the loop setup below
        # refers to it.
        for sym_name, (max_value, granularity) in loop_dim_bounds.items():
            name = loop_dim_ssa[sym_name]
            arg_type = (
                f"!sdscbundle.input_arg<index, granularity={granularity}, "
                f"max_value={max_value}>"
            )
            f.write(
                f"\t\t{name} = sdscbundle.input_arg_extract value from"
                f" {name}_base : {arg_type} -> index\n"
            )

        # One LoopLevel per loop, in _collect_loop_bounds order. A concrete count
        # keeps the constant-bound, step-1 form this emitter has always produced;
        # a symbolic one becomes "to <dim> step <G>". See _loop_level.
        loop_levels = [
            _loop_level(lb, lb_idx, loop_dim_ssa)
            for lb_idx, lb in enumerate(loop_bounds)
        ]

        # Standard loop constants (only emitted when there are loops).
        if loop_bounds:
            f.write("\t\t%c0 = arith.constant 0 : index\n")
            f.write("\t\t%c1 = arith.constant 1 : index\n")
            for level in loop_levels:
                for line in level.setup:
                    f.write(f"\t\t{line}\n")

        # Emit one declaration per symbol:
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
            kernel_sym_to_arg_idx=kernel_sym_to_arg_idx,
            sym_canonical=sym_canonical,
            loop_levels=loop_levels,
        )

        f.write("\t\treturn\n")
        f.write("\t}\n")
        f.write("}\n")

    if sdsc_log.isEnabledFor(logging.DEBUG):
        bundle_path = os.path.join(output_dir, "bundle.mlir")
        with open(bundle_path, "r") as bf:
            sdsc_log.debug("BUNDLE MLIR [bundle.mlir]\n%s", bf.read())

    return param_symbol_kinds


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
    sdsc_cache: dict | None = None,
    _sdsc_cache_counts: list | None = None,
) -> None:
    """Recursively compile all OpSpecs in specs depth-first.

    Identical op specs (same canonical SDSC at counter 0) reuse the previously
    compiled entry — same sdsc file and same symbol registrations.
    Pass sdsc_cache={} to enable caching; None disables it.
    """
    for entry in specs:
        if isinstance(entry, LoopSpec):
            _compile_specs(
                entry.body,
                symbols,
                compiled,
                sdsc_counter,
                symbol_id_offset_counter,
                output_dir,
                sdsc_cache,
                _sdsc_cache_counts,
            )
        elif isinstance(entry, OpSpec):
            cached = None
            if sdsc_cache is not None:
                # Generate a canonical (counter-0) version as cache key,
                # ignoring debug_handle_ which varies per op but is irrelevant
                # to structural identity.
                canonical_json, _, _, _ = compile_op_spec(0, entry, [], 0)
                top_val = next(iter(canonical_json.values()))
                top_val.pop("debug_handle_", None)
                # arg_indices must be part of the key: the canonical json only
                # records sequential placeholder IDs (-1,-2,-3), not which
                # kernel tensor argument each slot belongs to. Two structurally
                # identical ops on different tensors would otherwise collide.
                arg_indices = tuple(a.arg_index for a in entry.args)
                cache_key = json.dumps(canonical_json, sort_keys=True) + str(
                    arg_indices
                )
                cached = sdsc_cache.get(cache_key)
            if cached is None:
                idx = sdsc_counter[0]
                sdsc_counter[0] += 1
                if _sdsc_cache_counts is not None:
                    _sdsc_cache_counts[1] += 1
            else:
                idx, cached_json = cached
                if _sdsc_cache_counts is not None:
                    _sdsc_cache_counts[0] += 1
            sdsc_json, local_sym_values, affine_strides, local_symbol_kinds = (
                compile_op_spec(
                    idx,
                    entry,
                    symbols,
                    symbol_id_offset_counter[0],
                )
            )
            symbol_id_offset_counter[0] += len(local_sym_values)
            file_name = f"sdsc_{idx}.json"
            if cached is None:
                cached_json = sdsc_json
                if sdsc_cache is not None:
                    sdsc_cache[cache_key] = (idx, cached_json)
                with open(os.path.join(output_dir, file_name), "w") as f:
                    logger.info(f"Generating {f.name}")
                    json.dump(sdsc_json, f, indent=2)
            compiled.append(
                (
                    sdsc_json,
                    local_sym_values,
                    affine_strides,
                    local_symbol_kinds,
                    cached_json,
                )
            )
            if sdsc_log.isEnabledFor(logging.DEBUG):
                sdsc_log.debug(
                    "SDSC JSON [%s]\n%s",
                    file_name,
                    json.dumps(sdsc_json, indent=2),
                )
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
    scale_stack: "list[int] | None" = None,
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

    ``scale_stack`` carries each enclosing level's stride scale, so the stride
    KEY registered here matches the stride the emitter will actually write. A
    symbolic level's strides are divided by its tile size (see
    ``_scaled_strides``); registering the unscaled value would allocate a map
    nothing uses and miss a share with an identical scaled key.
    """
    if scale_stack is None:
        scale_stack = []
    for entry in specs:
        if isinstance(entry, LoopSpec):
            _collect_affine_maps(
                entry.body,
                compiled_iter,
                loop_var_depth + [len(loop_var_depth)],
                affine_map_index,
                loop_var_indices_out,
                scale_stack + [_count_scale(entry.count)],
            )
        elif isinstance(entry, OpSpec):
            _, _, affine_strides, _, _ = next(compiled_iter)
            per_tensor_lv_indices: list[list[int]] = []
            for per_level_strides in affine_strides:
                # per_level_strides is list[dict], one dict per level (outermost first).
                # Build stride_key and lv_indices by iterating levels explicitly.
                stride_vals: list[int] = []
                lv_idxs: list[int] = []
                for level_idx, stride in _scaled_strides(
                    per_level_strides, scale_stack
                ):
                    assert level_idx < len(loop_var_depth), (
                        f"affine_strides has {len(per_level_strides)} levels but "
                        f"only {len(loop_var_depth)} enclosing loop(s); "
                        "create_op_spec built more tiled_syms levels than LoopSpec ancestors"
                    )
                    stride_vals.append(stride)
                    lv_idxs.append(loop_var_depth[level_idx])
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


@dataclasses.dataclass(frozen=True)
class LoopLevel:
    """How one ``scf.for`` level is emitted, and what that does to its strides.

    ``stride_scale`` is the divisor every affine stride multiplying this level's
    loop variable must be divided by. It is exact by construction: a level's
    stride is its per-element step times the tile size, so dividing by the tile
    size leaves the per-element step.
    """

    setup: "tuple[str, ...]"
    bound: str
    step: str
    stride_scale: int


def _count_scale(count: sympy.Expr) -> int:
    """The stride divisor a level with this trip count imposes.

    1 for a concrete count or a tile size of 1, because the loop variable counts
    single steps either way. Otherwise the tile size, because the loop variable
    counts elements and steps by a whole tile each trip.

    Kept separate from ``_loop_level`` because the affine-map collection pass
    runs before any SSA names exist and only needs the scale.
    """
    if isinstance(count, (sympy.Integer, int)):
        return 1
    decomposed = decompose_tiled_count(count)
    return decomposed[1] if decomposed else 1


def _loop_level(
    count: sympy.Expr,
    lb_idx: int,
    loop_dim_ssa: "dict[str, str]",
) -> LoopLevel:
    """Pick the ``scf.for`` form for one loop level.

    A concrete count keeps the form this emitter has always produced: a constant
    upper bound with step 1, loop variable counting TILES.

    A symbolic count becomes ``to <dim> step <G>``, so the loop variable counts
    ELEMENTS along the varying dimension and the device works the trip count out
    itself as ``(ub - lb) / step``. We deliberately do not author the division:
    the instruction that builds the device loop counter switches on the bound's
    defining op, and only a constant, a query-map result or a symbol-creation
    result becomes a true symbolic loop count, so an authored divide would land
    on a generic dynamic-loop path instead. Keeping the dimension itself as the
    bound is also what lets one dimension parameter drive two loops with
    different tile sizes, since each derives its own count.

    Floor versus ceiling does not arise: G divides the size exactly, which the
    declared contract guarantees and the host checks before dispatch.

    Args:
        count: This level's trip count, concrete or symbolic.
        lb_idx: Index used to name this level's SSA values.
        loop_dim_ssa: Symbol name to the SSA value its ``input_arg`` extracted
            to, so a symbolic bound can be wired to its parameter.

    Returns:
        The level's setup lines, bound, step and stride scale.

    Raises:
        NotImplementedError: The count is symbolic but not a shape this emitter
            recognises, or it names a symbol with no ``input_arg`` parameter.
    """
    if isinstance(count, (sympy.Integer, int)):
        return LoopLevel(
            setup=(f"%loop_bound_{lb_idx} = arith.constant {int(count)} : index",),
            bound=f"%loop_bound_{lb_idx}",
            step="%c1",
            stride_scale=1,
        )

    decomposed = decompose_tiled_count(count)
    if decomposed is None:
        raise NotImplementedError(
            f"symbolic loop count {count!r} (type {type(count).__name__}, "
            f"free_symbols="
            f"{sorted(map(str, getattr(count, 'free_symbols', [])))}) is not a "
            "recognized trip-count shape. Expected a symbol divided by an "
            "integer, or a bare symbol. Extend "
            "pass_utils.decompose_tiled_count if this shape is legitimate."
        )

    sym, tile = decomposed
    sym_name = str(sym)
    if sym_name not in loop_dim_ssa:
        raise NotImplementedError(
            f"symbolic loop count {count} references {sym_name}, which has no "
            f"input_arg parameter to read its value from. Known dimension "
            f"params: {sorted(loop_dim_ssa)}. A symbol reaches that list via "
            f"LoopSpec.count_symbol_bounds"
        )

    dim_ssa = loop_dim_ssa[sym_name]
    if tile == 1:
        # The loop variable already counts single elements, so the dimension is
        # the bound as it stands and no stride needs rescaling.
        return LoopLevel(setup=(), bound=dim_ssa, step="%c1", stride_scale=1)
    return LoopLevel(
        setup=(f"%step_{lb_idx} = arith.constant {tile} : index",),
        bound=dim_ssa,
        step=f"%step_{lb_idx}",
        stride_scale=tile,
    )


def _scaled_strides(per_level_strides: list, scale_stack: list):
    """Yield (level_idx, stride) with each level's stride scale applied.

    A symbolic level's loop variable steps by its tile size rather than by 1, so
    every stride multiplying it shrinks by that same factor. Writing the
    per-tile stride against an element-stepping loop would advance the address a
    whole tile too far on every trip, so the two always move together.
    """
    for level_idx, level_strides in enumerate(per_level_strides):
        if not level_strides:
            continue
        scale = scale_stack[level_idx] if level_idx < len(scale_stack) else 1
        for stride in level_strides.values():
            if scale != 1:
                if stride % scale:
                    raise AssertionError(
                        f"affine stride {stride} is not divisible by the loop "
                        f"step {scale} at level {level_idx}. A symbolic level "
                        f"steps by its tile size, so its strides must be a "
                        f"multiple of it, or the loop variable and the stride "
                        f"disagree about what one step means."
                    )
                stride = stride // scale
            yield level_idx, stride


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
    kernel_sym_to_arg_idx: dict | None = None,
    sym_canonical: dict | None = None,
    loop_levels: "list[LoopLevel] | None" = None,
    scale_stack: "list[int] | None" = None,
) -> None:
    """Recursively emit MLIR ops for specs into file f.

    ``loop_levels`` carries each loop's emitted form, indexed the same way as
    ``loop_bounds``. Absent, every level falls back to the constant-bound,
    step-1 form, which is what a tree of concrete counts produces anyway.

    ``scale_stack`` carries each enclosing level's stride scale and must be
    maintained exactly as ``_collect_affine_maps`` maintains its own. The two
    are the write and read halves of one contract: that pass registers an affine
    map under its SCALED stride key and this one looks the map up by the same
    key, so scaling in one and not the other is a KeyError on every symbolic
    kernel.
    """
    if scale_stack is None:
        scale_stack = []
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
            level = (
                loop_levels[lb_idx]
                if loop_levels is not None and lb_idx < len(loop_levels)
                else None
            )
            bound = level.bound if level is not None else f"%loop_bound_{lb_idx}"
            step = level.step if level is not None else "%c1"
            f.write(f"{tab}scf.for {loop_var} = %c0 to {bound} step {step} {{\n")
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
                kernel_sym_to_arg_idx=kernel_sym_to_arg_idx,
                sym_canonical=sym_canonical,
                loop_levels=loop_levels,
                scale_stack=scale_stack + [_count_scale(entry.count)],
            )
            f.write(f"{tab}}}\n")

        elif isinstance(entry, OpSpec):
            sdsc_json, local_sym_values, affine_strides, _, cached_json = next(
                compiled_iter
            )
            # Per-tensor loop-var index lists: which positions in the enclosing
            # loop_vars list correspond to the strides for each tensor.
            per_tensor_lv_indices: list[list[int]] = next(affine_map_lv_iter)

            # Filename and printed symbol_ids come from the cached (first) JSON so
            # that deduplicated executions reference the same sdsc file and IDs.
            sdsc_name = next(iter(cached_json))
            sdsc_idx = sdsc_name.split("_")[0]
            sdsc_filename = f"sdsc_{sdsc_idx}.json"
            cached_symbol_ids = _extract_symbol_ids(cached_json)

            # Fresh symbol_ids (from sdsc_json) are used only for resolving operands.
            symbol_ids = _extract_symbol_ids(sdsc_json)

            # Build affine.apply ops for tiled tensors, tracking which
            # symbol IDs have been upgraded to per-iteration %addr_N names.
            # affine_strides[tensor_idx] is list[dict] (per level, outermost first).
            sym_id_to_operand: dict[int, str] = {}
            for tensor_idx, per_level_strides in enumerate(affine_strides):
                # Built through the same helper _collect_affine_maps used, so
                # the key looked up here is the key that pass registered. A
                # symbolic level's strides are divided by its step there, and
                # flattening the raw values instead misses every one of them.
                flat_strides: list[int] = [
                    stride
                    for _level_idx, stride in _scaled_strides(
                        per_level_strides, scale_stack
                    )
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
            symbol_ids_str = ", ".join(str(i) for i in cached_symbol_ids)
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
                    # NOTE: "hbm" is an sdsc component field and is
                    # distinct from and NOT to be confused with the internal
                    # layout.allocation dict keys ("hbm"/"lx"/"hbm_pool").
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
                    # NOTE: "hbm" is an sdsc component field and is
                    # distinct from and NOT to be confused with the internal
                    # layout.allocation dict keys ("hbm"/"lx"/"hbm_pool").
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
