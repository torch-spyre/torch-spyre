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


import dataclasses

from torch_spyre._C import encode_constant, DataFormats
from sympy import Symbol


@dataclasses.dataclass(frozen=True)
class SymbolKind:
    """Classifies a symbol registered in the bundle symbol table.

    Four variants (constructed via class methods):
      - ``kernel(arg_index)``:               raw HBM base address of a kernel tensor arg;
                                             emitted as a ``!sdscbundle.input_arg`` param
                                             named ``%arg_{arg_index}``.  Value =
                                             ``tensor.start_address``.
      - ``kernel_slice(arg_i, slice_off)``:  sliced base = raw base + compile-time slice
                                             offset (from device_coordinates like ``z0+3``).
                                             Emitted as ``arith.addi %arg_{arg_i},
                                             {slice_off}``.  ``slice_off`` is in bytes.
                                             Only present when ``slice_off > 0``;
                                             when ``slice_off == 0`` the ``kernel`` symbol
                                             itself serves as the sliced base.
      - ``kernel_derived(idx, off, arg_i)``: per-core derived address = sliced_base + offset;
                                             emitted as ``arith.addi <sliced_base_ssa>, off``.
                                             ``base_sym_idx`` is the 0-based index into the
                                             global ``symbols`` list of the sliced-base symbol
                                             (either a ``kernel`` or ``kernel_slice`` entry).
      - ``pool()``:                          pool-allocated tensor address;
                                             emitted as ``arith.addi %pool, value``.
      - ``dimension(gran, max, sym)``:       dynamic iteration-space dim size from
                                             mark_dynamic; carried in SDSC JSON as a
                                             ``dimToSymbolMapping_`` entry.  Registered
                                             before address symbols so their negative IDs
                                             never collide with address symbol IDs.
    """

    kind: str
    base_sym_idx: int = -1
    offset: int = 0
    arg_index: int = -1
    granularity: int = 0
    max_value: int = 0
    pytorch_sym: str = ""

    @classmethod
    def kernel(cls, arg_index: int) -> "SymbolKind":
        return cls(kind="kernel", arg_index=arg_index)

    @classmethod
    def kernel_slice(cls, arg_index: int, offset: int) -> "SymbolKind":
        return cls(kind="kernel_slice", arg_index=arg_index, offset=offset)

    @classmethod
    def kernel_derived(
        cls, base_sym_idx: int, offset: int, arg_index: int
    ) -> "SymbolKind":
        return cls(
            kind="kernel_derived",
            base_sym_idx=base_sym_idx,
            offset=offset,
            arg_index=arg_index,
        )

    @classmethod
    def pool(cls) -> "SymbolKind":
        return cls(kind="pool")

    @classmethod
    def dimension(
        cls, granularity: int, max_value: int, pytorch_sym: str
    ) -> "SymbolKind":
        return cls(
            kind="dimension",
            granularity=granularity,
            max_value=max_value,
            pytorch_sym=pytorch_sym,
        )

    @property
    def is_derived(self) -> bool:
        return self.kind == "kernel_derived"

    @property
    def is_pool(self) -> bool:
        return self.kind == "pool"

    @property
    def is_dimension(self) -> bool:
        return self.kind == "dimension"


def core_idx_to_slice_offset(
    arg,
    wk_slice: dict,
    work_slices: dict,
) -> int:
    offset = sum(arg.offsets.values())
    for dim, stride in arg.strides.items():
        if str(dim) in wk_slice and arg.scales[dim] > 0:
            offset += wk_slice[str(dim)] * stride // work_slices[dim]
    return offset


def num_bytes(df: DataFormats) -> int:
    """Try to avoid using this method; it is a bad API due to sub-byte datatypes"""
    num_elems = df.elems_per_stick()
    if num_elems > 128:
        raise RuntimeError(f"sub-byte dataformat {df}")
    return 128 // num_elems


def generate_constant_info(data_format, constants, num_cores):
    if len(constants.keys()) == 0:
        return "{}"
    constant_info = {}
    for name, value in constants.items():
        ci = {
            "dataFormat_": data_format.name,
            "name_": name,
            "data_": {
                "dim_prop_func": [{"Const": {}}, {"Const": {}}, {"Map": {}}],
                "dim_prop_attr": [
                    {"factor_": num_cores, "label_": "core"},
                    {"factor_": 1, "label_": "corelet"},
                    {"factor_": 1, "label_": "time"},
                ],
                "data_": {"[0, 0, 0]": [encode_constant(value, data_format)]},
            },
        }
        constant_info[f"{len(constant_info)}"] = ci
    return constant_info


def add_constant(kwargs, name, value) -> int:
    """
    Add a constant to kwargs['op_info']['constants'] and return its index.
    Returns:
        int: The index of the newly added constant (0-based)
    """
    # Ensure structure exists
    if "op_info" not in kwargs:
        kwargs["op_info"] = {}
    if "constants" not in kwargs["op_info"]:
        kwargs["op_info"]["constants"] = {}

    index = len(kwargs["op_info"]["constants"])
    kwargs["op_info"]["constants"][name] = value

    return index


def gen_coord_info_value(
    size: int,
    nsplits: int,
    elems_per_stick: int,
    is_stick_dim: bool,
    is_stick_reduction: bool = False,
):
    return (
        {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 1,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {
                        "Affine": {
                            "alpha_": size,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 1,
                            "beta_": 0,
                        }
                    },
                ],
                "dim_prop_attr": [
                    {
                        "factor_": nsplits,
                        "label_": "core_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "corelet_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "row_fold",
                    },
                    {
                        "factor_": size,
                        "label_": "elem_arr_0",
                    },
                ],
            },
        }
        if not is_stick_dim
        else {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 2,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {
                        "Affine": {
                            "alpha_": elems_per_stick if is_stick_reduction else size,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": elems_per_stick,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0 if is_stick_reduction else 1,
                            "beta_": 0,
                        }
                    },
                ],
                "dim_prop_attr": [
                    {
                        "factor_": nsplits,
                        "label_": "core_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "corelet_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "row_fold",
                    },
                    {
                        "factor_": 1
                        if is_stick_reduction
                        else (size // elems_per_stick),
                        "label_": "elem_arr_1",
                    },
                    {
                        "factor_": elems_per_stick,
                        "label_": "elem_arr_0",
                    },
                ],
            },
        }
    )


def _per_core_symbolic_dim_info(symbolic_dims: dict, work_slices: dict) -> dict:
    """Per-core ``symbolicDimInfo_`` block: granularity_/maxSize_ divided by
    each dim's work_slices.

    Shared by the ``ss_`` and ``el_`` sub-dicts of ``dataStageParam_``, which
    must stay byte-for-byte identical -- factored out so the two never drift.
    """
    info = {}
    for dim_name, (_, granularity, max_val) in symbolic_dims.items():
        wk_slices = work_slices[Symbol(dim_name)]
        info[dim_name] = {
            "maxSize_": max_val // wk_slices,
            "granularity_": max(1, granularity // wk_slices),
        }
    return info


def _tiled_byte_stride(tensor, tiled_sym) -> int:
    """Byte stride per loop iteration for a single tiled dimension.

    ``tensor.strides[tiled_sym]`` is already the per-tile element stride
    (``_create_sdsc_tensors`` receives the per-tile iteration space, since
    ``coarse_tile.py`` has already divided the op ranges by ``loop_count``
    before ``create_op_spec`` runs).  Multiplying by bytes-per-element
    gives the correct per-iteration byte advance for the ``affine.apply``
    in the ``scf.for`` loop body.
    """
    return int(tensor.strides[tiled_sym] * num_bytes(tensor.data_format))


def _find_index_tensor_for_value(sdsc_spec, value_tensor_idx: int) -> int:
    """Find the index of the index tensor that references the given value tensor.

    Returns -1 if no index tensor references this value tensor.
    """
    for j, t in enumerate(sdsc_spec.args):
        if t.is_index_tensor and t.related_value_tensor_idx == value_tensor_idx:
            return j
    return -1


def _get_indirect_access_info(
    sdsc_spec, tensor, tensor_idx: int
) -> tuple[str, str | None]:
    """Get indirect access allocation type and related allocation name for a tensor.

    Returns:
        A tuple of (alloc_type, related_alloc_or_none) where:
        - alloc_type: "index_tensor", "value_tensor", or "no_indirection"
        - related_alloc_or_none: allocation name of related tensor, or None
    """
    # Index tensors and value tensors involved in indirect access must reside in HBM;
    # the Spyre engine does not support indirect addressing through LX scratchpad.
    if tensor.is_index_tensor:
        alloc_type = "index_tensor"
        related_alloc = (
            f"allocate-Tensor{tensor.related_value_tensor_idx}_hbm"
            if tensor.related_value_tensor_idx >= 0
            else None
        )
        return alloc_type, related_alloc

    # Check if this is a value tensor referenced by an index tensor
    value_tensor_indices = [
        t.related_value_tensor_idx for t in sdsc_spec.args if t.is_index_tensor
    ]
    if tensor_idx in value_tensor_indices:
        alloc_type = "value_tensor"
        index_tensor_idx = _find_index_tensor_for_value(sdsc_spec, tensor_idx)
        if index_tensor_idx < 0:
            raise ValueError(
                f"Tensor {tensor_idx} is listed as a value tensor but no index "
                "tensor claims it — sdsc_spec is malformed"
            )
        related_alloc = f"allocate-Tensor{index_tensor_idx}_hbm"
        return alloc_type, related_alloc

    return "no_indirection", None


def _build_indirect_access_fields(sdsc_spec, tensor, tensor_idx: int) -> dict:
    """Build the indirect access fields for a tensor allocation.

    Returns a dictionary containing:
    - indirectAllocType_: The allocation type ("index_tensor", "value_tensor",
      or "no_indirection")
    - relatedIndirectAccessAlloc_: The related allocation name (only if applicable)
    - indexTensorType_: The index tensor type - only for index tensors; the
      backend supports "address" and "index" but we only generate "index"
    """
    alloc_type, related_alloc = _get_indirect_access_info(sdsc_spec, tensor, tensor_idx)

    fields = {"indirectAllocType_": alloc_type}
    if related_alloc is not None:
        fields["relatedIndirectAccessAlloc_"] = related_alloc

    if tensor.is_index_tensor:
        fields["indexTensorType_"] = "index"

    return fields


def generate_sdsc(
    idx,
    sdsc_spec,
    symbols: list[int],
    symbol_id_offset: int = 0,
    tiled_symbols=None,
    use_symbols: bool = False,
):
    """Generate SDSC JSON for one OpSpec.

    Returns a 4-tuple ``(sdsc_json, base_symbol_values, affine_strides, symbol_kinds)``:
    - ``sdsc_json``: the JSON dict to write to ``sdsc_N.json``
    - ``base_symbol_values``: list of HBM byte offsets registered in ``symbols``;
      empty when ``use_symbols=False``
    - ``affine_strides``: list (parallel to ``sdsc_spec.args``) of per-level
      stride lists.  Each element is a list of dicts, one per loop-nesting level
      (outermost first), where each dict maps ``tiled_sym -> stride_bytes`` for
      that level's tiled symbols.  Always ``[[]] * len(sdsc_spec.args)`` when
      ``use_symbols=False``.  Used by ``bundle.py`` to emit ``affine.apply`` ops
      inside ``scf.for`` loops, with one stride per level mapped to the correct
      loop variable.
    - ``symbol_kinds``: list of ``SymbolKind`` parallel to ``base_symbol_values``;
      empty when ``use_symbols=False``.  Classifies each symbol as a kernel base
      address, per-core derived address, or pool-allocated address.

    When ``use_symbols=False``, HBM tensor addresses are baked directly as
    concrete integers into the SDSC JSON.  No symbol IDs are registered and
    ``symbols`` is not modified.

    When ``use_symbols=True``, HBM addresses are registered as negative symbol
    IDs in the JSON and their values appended to ``symbols``, enabling
    ``affine.apply`` address computation in ``bundle.mlir`` for tiled loops.
    """
    # tiled_symbols is list[list[Symbol]], outermost-first per nesting level.
    if tiled_symbols is None:
        tiled_symbols = []

    out_idx = len(sdsc_spec.args) - 1
    core_id_to_wk_slice = {
        str(c): {
            str(dim): int(expr.subs({Symbol("core_id"): c}))
            for dim, expr in sdsc_spec.core_id_to_work_slice.items()
        }
        for c in range(sdsc_spec.num_cores)
    }
    symbolic_dims = sdsc_spec.symbolic_dims or {}

    # Register dimension symbols BEFORE address symbols so their IDs never collide.
    # IDs are laid out as: -(offset+1)..-(offset+n_dim) for dim symbols, then
    # -(offset+n_dim+1)..-(offset+n_dim+k) for address symbols.
    # Dim symbols carry no HBM byte value; 0 is appended to `symbols` as a placeholder.
    dim_local_symbols: dict[str, int] = {}  # pytorch_sym_name -> negative symbol ID
    dim_symbol_kinds: list[SymbolKind] = []
    for sdsc_dim, (pytorch_sym, granularity, max_value) in symbolic_dims.items():
        if pytorch_sym not in dim_local_symbols:
            sym_id = -(symbol_id_offset + len(dim_symbol_kinds) + 1)
            dim_local_symbols[pytorch_sym] = sym_id
            dim_symbol_kinds.append(
                SymbolKind.dimension(granularity, max_value, pytorch_sym)
            )
            symbols.append(0)  # placeholder: dim symbols have no HBM byte value
    n_dim_syms = len(dim_symbol_kinds)

    # local_symbols maps address key -> globally-unique negative symbol id.
    # symbol_id_offset ensures ids are unique across all SDSCs in the bundle.
    # For tiled tensors the base is the iteration-0 address (tiled dims contribute 0);
    # for non-tiled tensors it is the full per-core address (as before).
    #
    # Keys use explicit namespacing to prevent any possibility of collision:
    #   ("kernel", arg_index)       — raw HBM base for kernel tensor arg_index
    #   ("kernel_slice", arg_index) — sliced base (raw + compile-time offset)
    #   int addr                    — per-core derived address (c>0 kernel tensors,
    #                                 always large HBM byte addresses)
    #   ("pool", int offset)        — pool-allocated tensor compile-time offset
    #
    # On the symbolic path, kernel sentinels are arg_index integers (0, 1, 2...).
    # Keying by ("kernel", arg_index) rather than the sentinel value itself ensures
    # no collision with pool offset 0 or any future sentinel scheme.
    #
    # NOTE: no cross-SDSC deduplication — each call to offset_as_symbol within
    # this SDSC gets its own sequential ID and appends to symbols.  Two SDSCs
    # that happen to share a base address will emit two separate arith.constant
    # declarations in bundle.mlir.  This keeps symbol IDs contiguous with the
    # symbols list indices: symbols[abs(id)-1] is always the value for id.
    local_symbols: dict[tuple | int, int] = {}
    # Parallel to local_symbols (insertion order): one SymbolKind per registered symbol.
    local_symbol_kind: list[SymbolKind] = []

    def _derived_kind(
        arg_index: int,
        core0_addr: int,
        addr: int,
        sliced_base_sym_idx: int,
    ) -> SymbolKind:
        """Return the SymbolKind for a per-core (c>0) HBM address.

        Core 0 is handled by the caller (either ``kernel`` or ``kernel_slice``).
        ``sliced_base_sym_idx`` is the 0-based index in ``symbols`` of the
        sliced-base symbol (``kernel`` or ``kernel_slice``) for this tensor.
        """
        return SymbolKind.kernel_derived(
            base_sym_idx=sliced_base_sym_idx,
            offset=addr - core0_addr,
            arg_index=arg_index,
        )

    if use_symbols:

        def offset_as_symbol(s, kind: SymbolKind):
            key: tuple | int
            if kind.is_pool:
                key = ("pool", s)
            elif kind.kind == "kernel":
                key = ("kernel", kind.arg_index)
            elif kind.kind == "kernel_slice":
                key = ("kernel_slice", kind.arg_index, kind.offset)
            else:
                # kernel_derived: s is a large per-core HBM byte address,
                # distinct from pool offsets and sentinel values.
                key = s
            if key not in local_symbols:
                # Address symbols start after dim symbols in the ID counter.
                local_symbols[key] = -(
                    symbol_id_offset + n_dim_syms + len(local_symbols) + 1
                )
                symbols.append(s)
                local_symbol_kind.append(kind)
            return local_symbols[key]

        # Compute per-tensor, per-level affine strides and register base addresses.
        # affine_strides[i] is a list of dicts, one per loop-nesting level
        # (outermost first), where each dict maps tiled_sym -> stride_bytes for
        # the symbols at that level that advance tensor i.  Empty list of dicts
        # (i.e. [{}] * n_levels or []) for non-tiled / lx tensors.
        affine_strides: list[list[dict]] = []
        for tensor in sdsc_spec.args:
            if "lx" in tensor.allocation:
                affine_strides.append([{} for _ in tiled_symbols])
                continue
            nb = num_bytes(tensor.data_format)
            slice_offset_bytes = sum(tensor.offsets.values()) * nb
            # core0_addr: compile-time address for core 0 including the tensor's
            # slice offset (device_coordinate constant terms, e.g. z0+3 → 3 rows).
            core0_addr = (
                tensor.start_address
                + core_idx_to_slice_offset(
                    tensor, core_id_to_wk_slice["0"], sdsc_spec.work_slices
                )
                * nb
            )
            if tensor.arg_index >= 0:
                # Kernel tensors: register the raw base address first so bundle.py
                # can emit the input_arg function parameter.
                #
                # On the symbolic path, tensor.start_address = arg_index + tile_offset_bytes,
                # where tile_offset_bytes is the per-tile byte advance added by the loop
                # unroller.  We always register the raw kernel symbol keyed by arg_index so
                # that bundle.py emits exactly one !sdscbundle.input_arg parameter per logical
                # tensor, regardless of how many tiles reference it.
                raw_base = tensor.arg_index  # sentinel value for this arg
                offset_as_symbol(
                    raw_base, SymbolKind.kernel(arg_index=tensor.arg_index)
                )
                # Derive the 0-based symbols[] index of the kernel symbol from its
                # registered ID.  Must be looked up (not inferred from current
                # len(local_symbols)) because the same arg_index may have been
                # registered already by an earlier tensor in this SDSC, in which case
                # the offset_as_symbol call above was a no-op.
                kernel_sym_idx = abs(local_symbols[("kernel", tensor.arg_index)]) - 1
                # tile_offset_bytes: the loop unroller advances arg.allocation['hbm']
                # by i*stride for tile i, so start_address = arg_index + tile_offset.
                # tile_offset_bytes == 0 for tile 0, positive for later tiles.
                tile_offset_bytes = tensor.start_address - tensor.arg_index
                # total_slice_offset: combine the loop-unroll tile offset with any
                # device-coordinate compile-time slice offset (e.g. from z0+3 expressions).
                # This is the total compile-time offset above the raw %arg_N base that the
                # sliced-base SSA value represents in bundle.mlir.
                total_slice_offset = tile_offset_bytes + slice_offset_bytes
                # sliced_base_sym_idx: the symbols[] index that per-core derived symbols
                # reference.  When total_slice_offset == 0 the kernel sym IS the sliced
                # base; otherwise a kernel_slice sym is registered for the combined offset.
                if total_slice_offset > 0:
                    offset_as_symbol(
                        core0_addr,
                        SymbolKind.kernel_slice(
                            arg_index=tensor.arg_index, offset=total_slice_offset
                        ),
                    )
                    slice_key = ("kernel_slice", tensor.arg_index, total_slice_offset)
                    sliced_base_sym_idx = abs(local_symbols[slice_key]) - 1
                else:
                    sliced_base_sym_idx = kernel_sym_idx
            else:
                # Pool tensor: no raw-base or slice symbol needed.
                sliced_base_sym_idx = -1
            # Build per-level strides: for each level, collect the symbols at that
            # level that tile this tensor (i.e. appear in tensor.strides).
            # Exclude symbols whose scale is negative: those are reduced dimensions
            # whose stride describes element layout within one tile, not the advance
            # between tiles.  Tiling by a reduction-dim symbol would incorrectly
            # advance the base address of a pool output past its single allocated slot.
            per_level_strides: list[dict] = []
            any_tiled = False
            for level_syms in tiled_symbols:
                tensor_tiled_at_level = [
                    s
                    for s in level_syms
                    if s in tensor.strides and tensor.scales.get(s, 1) > 0
                ]
                strides_for_level: dict = {}
                for s in tensor_tiled_at_level:
                    strides_for_level[s] = _tiled_byte_stride(tensor, s)
                    any_tiled = True
                per_level_strides.append(strides_for_level)
            if not any_tiled:
                # Non-tiled HBM: register per-core addresses.
                for c in range(sdsc_spec.num_cores):
                    addr = (
                        tensor.start_address
                        + core_idx_to_slice_offset(
                            tensor, core_id_to_wk_slice[str(c)], sdsc_spec.work_slices
                        )
                        * nb
                    )
                    if c == 0:
                        if tensor.arg_index < 0:
                            offset_as_symbol(addr, SymbolKind.pool())
                        # kernel / kernel_slice already registered above; skip c==0
                    else:
                        if tensor.arg_index < 0:
                            offset_as_symbol(addr, SymbolKind.pool())
                        elif addr != core0_addr:
                            # Only register a derived symbol when the core has a
                            # distinct address from core 0.  When addr == core0_addr
                            # (e.g. a non-split tensor where all cores share one
                            # address) the sliced-base symbol already covers it and
                            # we must not create a duplicate registration.
                            offset_as_symbol(
                                addr,
                                _derived_kind(
                                    tensor.arg_index,
                                    core0_addr,
                                    addr,
                                    sliced_base_sym_idx,
                                ),
                            )
                affine_strides.append([{} for _ in tiled_symbols])
            else:
                # Tiled HBM: symbol value = per-core iter-0 base address.
                # The affine map adds loop_var * tile_stride on top at runtime.
                for c in range(sdsc_spec.num_cores):
                    addr = (
                        tensor.start_address
                        + core_idx_to_slice_offset(
                            tensor, core_id_to_wk_slice[str(c)], sdsc_spec.work_slices
                        )
                        * nb
                    )
                    if c == 0:
                        if tensor.arg_index < 0:
                            offset_as_symbol(addr, SymbolKind.pool())
                        # kernel / kernel_slice already registered above; skip c==0
                    else:
                        if tensor.arg_index < 0:
                            offset_as_symbol(addr, SymbolKind.pool())
                        elif addr != core0_addr:
                            offset_as_symbol(
                                addr,
                                _derived_kind(
                                    tensor.arg_index,
                                    core0_addr,
                                    addr,
                                    sliced_base_sym_idx,
                                ),
                            )
                affine_strides.append(per_level_strides)

        def _start_addr_data(tensor):
            # All per-core addresses were already registered by the per-tensor loop
            # above. Look them up using the same key scheme as offset_as_symbol.
            if "lx" in tensor.allocation:
                return {
                    f"[{c}, 0, 0]": str(tensor.start_address)
                    for c in range(sdsc_spec.num_cores)
                }
            nb = num_bytes(tensor.data_format)
            is_pool_tensor = tensor.arg_index < 0 and "pool" in tensor.allocation
            # Hoist kernel-tensor compile-time offsets so they are not
            # duplicated across the c==0 and c>0 branches.
            if not is_pool_tensor:
                slice_offset_bytes = sum(tensor.offsets.values()) * nb
                tile_offset_bytes = tensor.start_address - tensor.arg_index
                total_slice_offset = tile_offset_bytes + slice_offset_bytes
                c0_slice_key: tuple | int = (
                    ("kernel_slice", tensor.arg_index, total_slice_offset)
                    if total_slice_offset > 0
                    else ("kernel", tensor.arg_index)
                )
                core0_addr_lookup = (
                    tensor.start_address
                    + core_idx_to_slice_offset(
                        tensor, core_id_to_wk_slice["0"], sdsc_spec.work_slices
                    )
                    * nb
                )
            result = {}
            for c in range(sdsc_spec.num_cores):
                addr = (
                    tensor.start_address
                    + core_idx_to_slice_offset(
                        tensor, core_id_to_wk_slice[str(c)], sdsc_spec.work_slices
                    )
                    * nb
                )
                if is_pool_tensor:
                    key: tuple | int = ("pool", addr)
                elif c == 0:
                    key = c0_slice_key
                else:
                    # c>0: per-core derived address.  When addr == core0_addr
                    # (non-split tensor, all cores share one address) no derived
                    # symbol was registered — reuse the c==0 sliced-base key.
                    key = c0_slice_key if addr == core0_addr_lookup else addr
                result[f"[{c}, 0, 0]"] = str(local_symbols[key])
            return result

    else:
        # use_symbols=False: bake concrete HBM addresses directly into the JSON.
        # symbols and local_symbols are not modified.
        affine_strides = [[{} for _ in tiled_symbols] for _ in sdsc_spec.args]

        def _start_addr_data(tensor):
            if "lx" in tensor.allocation:
                return {
                    f"[{c}, 0, 0]": str(tensor.start_address)
                    for c in range(sdsc_spec.num_cores)
                }
            return {
                f"[{c}, 0, 0]": str(
                    tensor.start_address
                    + core_idx_to_slice_offset(
                        tensor, core_id_to_wk_slice[str(c)], sdsc_spec.work_slices
                    )
                    * num_bytes(tensor.data_format)
                )
                for c in range(sdsc_spec.num_cores)
            }

    return (
        {
            f"{idx}_{sdsc_spec.opfunc}": {
                "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
                "sdscFolds_": {
                    "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                    "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                    "data_": {"[0]": "0"},
                },
                "coreFoldProp_": {"factor_": sdsc_spec.num_cores, "label_": "core"},
                "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
                "numCoresUsed_": sdsc_spec.num_cores,
                "coreIdToDsc_": {str(c): 0 for c in range(sdsc_spec.num_cores)},
                "numWkSlicesPerDim_": {
                    str(dim): num_wk_slices
                    for dim, num_wk_slices in sdsc_spec.work_slices.items()
                },
                "coreIdToWkSlice_": core_id_to_wk_slice,
                "coreIdToDscSchedule": {
                    str(c): [[-1, 0, 0, 0]] for c in range(sdsc_spec.num_cores)
                },
                "dscs_": [
                    {
                        sdsc_spec.opfunc: {
                            "numCoresUsed_": sdsc_spec.num_cores,
                            "numCoreletsUsed_": 1,
                            "coreIdsUsed_": [c for c in range(sdsc_spec.num_cores)],
                            "N_": {
                                "name_": "n",
                                **{
                                    str(dim) + "_": size
                                    for dim, size in sdsc_spec.iteration_space.items()
                                },
                            },
                            "coordinateMasking_": {
                                str(dim): mask_range
                                for dim, mask_range in sdsc_spec.coordinate_masking.items()
                            },
                            "maskingConstId_": 0
                            if sdsc_spec.coordinate_masking
                            else -1,
                            # Emit dimToSymbolMapping_ only when there are symbolic dims;
                            # the runtime uses it to bind runtime shape values to symbols.
                            **(
                                {
                                    "dimToSymbolMapping_": {
                                        sdsc_dim: [dim_local_symbols[pytorch_sym]]
                                        for sdsc_dim, (
                                            pytorch_sym,
                                            granularity,
                                            max_value,
                                        ) in symbolic_dims.items()
                                        if pytorch_sym in dim_local_symbols
                                    },
                                }
                                if symbolic_dims
                                else {}
                            ),
                            "dataStageParam_": {
                                "0": {
                                    "ss_": {
                                        "name_": "core",
                                        **{
                                            str(dim) + "_": size
                                            // sdsc_spec.work_slices[dim]
                                            for dim, size in sdsc_spec.iteration_space.items()
                                        },
                                        # Per-dim symbolic bounds (per-core slice).
                                        # min_val / work_slices is the granularity that
                                        # the runtime must respect when choosing a batch size.
                                        "symbolicDimInfo_": _per_core_symbolic_dim_info(
                                            symbolic_dims, sdsc_spec.work_slices
                                        ),
                                        "maxSymbolicVolume_": {},
                                        "coreletSplit_": {},
                                        "rowSplit_": {},
                                        "peSfpSplit_": {},
                                        "paddingSizes_": {},
                                    },
                                    "el_": {
                                        "name_": "core",
                                        **{
                                            str(dim) + "_": size
                                            // sdsc_spec.work_slices[dim]
                                            for dim, size in sdsc_spec.iteration_space.items()
                                        },
                                        "symbolicDimInfo_": _per_core_symbolic_dim_info(
                                            symbolic_dims, sdsc_spec.work_slices
                                        ),
                                        "maxSymbolicVolume_": {},
                                        "coreletSplit_": {},
                                        "rowSplit_": {},
                                        "peSfpSplit_": {},
                                        "paddingSizes_": {},
                                    },
                                }
                            },
                            "primaryDsInfo_": {
                                label: {
                                    "layoutDimOrder_": [
                                        str(dim) for dim in layout_info["dim_order"]
                                    ],
                                    "stickDimOrder_": [
                                        str(layout_info["stick_dim_order"])
                                    ],
                                    "stickSize_": [layout_info["stick_size"]],
                                }
                                for label, layout_info in sdsc_spec.layouts.items()
                            },
                            "scheduleTree_": [
                                {
                                    "nodeType_": "allocate",
                                    "name_": f"allocate-Tensor{i}_{'lx' if 'lx' in tensor.allocation else 'hbm'}",
                                    "prev_": "",
                                    "ldsIdx_": i,
                                    "component_": "lx"
                                    if "lx" in tensor.allocation
                                    else "hbm",
                                    **(
                                        {"isStartAddrSymbolic_": 1}
                                        if use_symbols and "lx" not in tensor.allocation
                                        else {}
                                    ),
                                    "layoutDimOrder_": [
                                        str(dim) for dim in tensor.dim_order
                                    ],
                                    "maxDimSizes_": [
                                        tensor.max_dim_sizes[dim]
                                        for dim in sdsc_spec.layouts[tensor.layout][
                                            "dim_order"
                                        ]
                                    ],
                                    **_build_indirect_access_fields(
                                        sdsc_spec, tensor, i
                                    ),
                                    "startAddressCoreCorelet_": {
                                        "dim_prop_func": [
                                            {"Map": {}},
                                            {"Const": {}},
                                            {"Const": {}},
                                        ],
                                        "dim_prop_attr": [
                                            {
                                                "factor_": sdsc_spec.num_cores,
                                                "label_": "core",
                                            },
                                            {"factor_": 1, "label_": "corelet"},
                                            {"factor_": 1, "label_": "time"},
                                        ],
                                        "data_": _start_addr_data(tensor),
                                    },
                                    **(
                                        {
                                            "backGapCore_": {
                                                str(dim): (
                                                    # LX: per-core keys 0..num_cores-1
                                                    {
                                                        str(c): str(gap)
                                                        for c in range(
                                                            sdsc_spec.num_cores
                                                        )
                                                    }
                                                    if "lx" in tensor.allocation
                                                    # HBM: -1 sentinel covers all cores
                                                    else {"-1": str(gap)}
                                                )
                                                for dim, gap in tensor.backGap.items()
                                            }
                                        }
                                        if tensor.backGap
                                        else {}
                                    ),
                                    "coordinates_": {
                                        "coordInfo": {
                                            str(dim): gen_coord_info_value(
                                                size=sdsc_spec.iteration_space[dim]
                                                // sdsc_spec.work_slices[dim]
                                                if (tensor.scales[dim] == 1)
                                                else 1,
                                                nsplits=sdsc_spec.work_slices[dim]
                                                if (tensor.scales[dim] == 1)
                                                else 1,
                                                elems_per_stick=tensor.data_format.elems_per_stick(),
                                                is_stick_dim=(
                                                    sdsc_spec.layouts[tensor.layout][
                                                        "stick_dim_order"
                                                    ].has(dim)
                                                ),
                                                is_stick_reduction=(
                                                    tensor.scales[dim] == -2
                                                ),
                                            )
                                            for dim in sdsc_spec.layouts[tensor.layout][
                                                "dim_order"
                                            ]
                                        },
                                        "coreIdToWkSlice_": {},
                                    },
                                }
                                for i, tensor in enumerate(sdsc_spec.args)
                            ],
                            "labeledDs_": [
                                {
                                    "ldsIdx_": i,
                                    "dsName_": f"Tensor{i}",
                                    "dsType_": tensor.layout,
                                    "scale_": [
                                        tensor.scales[dim]
                                        for dim in sdsc_spec.layouts[tensor.layout][
                                            "dim_order"
                                        ]
                                    ],
                                    "wordLength": num_bytes(tensor.data_format),
                                    "dataFormat_": tensor.data_format.name,
                                    # Index tensors must reside in HBM; the Spyre
                                    # engine does not support indirect addressing
                                    # through LX scratchpad.
                                    "memOrg_": {"hbm": {"isPresent": 1}}
                                    if tensor.is_index_tensor
                                    else {
                                        "hbm": {"isPresent": 1},
                                        "lx": {"isPresent": 1},
                                    }
                                    if "lx" not in tensor.allocation
                                    else {"lx": {"isPresent": 1}},
                                }
                                for i, tensor in enumerate(sdsc_spec.args)
                            ],
                            "constantInfo_": generate_constant_info(
                                sdsc_spec.data_format,
                                sdsc_spec.constants,
                                sdsc_spec.num_cores,
                            ),
                            "computeOp_": [
                                {
                                    "exUnit": sdsc_spec.execution_unit,
                                    "opFuncName": sdsc_spec.opfunc,
                                    "attributes_": {
                                        "dataFormat_": sdsc_spec.data_format.name,
                                        "fidelity_": "regular",
                                    },
                                    "location": "Inner",
                                    "inputLabeledDs": [
                                        f"Tensor{i}-idx{i}"
                                        for i in range(sdsc_spec.num_inputs)
                                        if i not in sdsc_spec.indirect_access_indices
                                    ],
                                    "outputLabeledDs": [
                                        f"Tensor{out_idx}-idx{out_idx}"
                                    ],
                                    **(
                                        {
                                            "indirectAccessIndexLabeledDs": [
                                                f"Tensor{i}-idx{i}"
                                                for i in sdsc_spec.indirect_access_indices
                                            ]
                                        }
                                        if sdsc_spec.indirect_access_indices
                                        else {}
                                    ),
                                }
                            ],
                        }
                    }
                ],
                # Emit top-level symbolic metadata only when symbolic dims are present.
                # inputSymbolsAndTags_ maps symbol ID -> pytorch symbol name for the runtime.
                **(
                    {
                        "datadscs_": [],
                        "dimToSymbolMappingOpcodeCorrection_": {},
                        "inputSymbolsAndTags_": {
                            str(sym_id): pytorch_sym
                            for pytorch_sym, sym_id in dim_local_symbols.items()
                        },
                        "symbolDefinitions_": {},
                    }
                    if symbolic_dims
                    else {}
                ),
            }
        },
        # Dim symbols occupy the first n_dim_syms slots (value 0); address symbols follow.
        [0] * n_dim_syms + list(local_symbols.keys()),
        affine_strides,
        dim_symbol_kinds + local_symbol_kind,
    )
