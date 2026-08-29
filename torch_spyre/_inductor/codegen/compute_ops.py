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

from sympy import Symbol

from torch_spyre._C import DataFormats, encode_constant
from torch_spyre._inductor.constants import CONV2D_DIM_LABELS, DEPTHWISE_CONV2D_OP
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.op_spec import TensorWorkDivision
from torch_spyre._inductor.pass_utils import coeff_through_floor


@dataclasses.dataclass(frozen=True)
class SymbolKind:
    """Classifies a symbol registered in the bundle symbol table.

    Five variants (constructed via class methods):
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
      - ``kernel_derived_symbolic(...)``:    per-core derived address when the
                                             tensor is split across cores on a
                                             *symbolic* dim.  The real byte offset
                                             is ``core_idx *
                                             ceildiv(S, split_count) *
                                             per_element_stride`` where ``S`` is the
                                             runtime size of the symbolic dim, so it
                                             cannot be baked at compile time.  This
                                             variant only TAGS the per-core address
                                             as symbolic, carrying ``core_idx``,
                                             ``split_count`` and the ``pytorch_sym``
                                             it depends on.  Emitting the actual
                                             arith formula, and computing the
                                             per-element stride it needs, is the
                                             bundle-arm follow-up (which depends on
                                             the dim ``input_arg`` SSA and is out of
                                             scope here).  ``is_derived`` stays False
                                             so the existing bundle
                                             ``kernel_derived`` addi branch does not
                                             match this variant.  At the SDSC-JSON
                                             layer this is identical to
                                             ``kernel_derived``: the per-core entry is
                                             a negative symbol id under
                                             ``isStartAddrSymbolic_``.
      - ``pool()``:                          MLIR-symbol-table mirror of a
                                             ``TensorArg.allocation["hbm_pool"]``-tagged
                                             tensor (see ``hbm_pool_planning.py`` and
                                             ``TensorArg.allocation``). This is the
                                             *same* underlying concept re-expressed at
                                             the symbol-table layer for MLIR emission,
                                             not a separate allocation kind ``SymbolKind``
                                             has no ``"hbm"``/``"lx"`` analog because
                                             those don't need symbolic-address emission
                                             the same way (kernel args are
                                             ``input_arg`` params directly; LX addresses
                                             are baked constants, never symbols).
                                             Emitted as ``arith.addi %pool, value``.
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
    # Only meaningful for the kernel_derived_symbolic variant.
    core_idx: int = -1
    split_count: int = 0

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
    def kernel_derived_symbolic(
        cls,
        arg_index: int,
        core_idx: int,
        split_count: int,
        base_sym_idx: int,
        pytorch_sym: str,
    ) -> "SymbolKind":
        """Per-core derived address for a symbolic-dim core split.

        Tags the per-core address as symbolic. The runtime formula
        ``core_idx * ceildiv(S, split_count) * per_element_stride`` and the
        per-element stride it needs are emitted by the bundle-arm follow-up,
        not here, so no stride is stored on this marker.
        """
        return cls(
            kind="kernel_derived_symbolic",
            arg_index=arg_index,
            core_idx=core_idx,
            split_count=split_count,
            base_sym_idx=base_sym_idx,
            pytorch_sym=pytorch_sym,
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
    def is_derived_symbolic(self) -> bool:
        return self.kind == "kernel_derived_symbolic"

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


def _compute_fp8_coord_params(tensor, dim, sdsc_spec):
    """Compute FP8 2D stick coordinate parameters for a dimension.

    Returns tuple: (is_fp8_stick, other_stick_size, stick_idx)
    """
    stick_size_list = sdsc_spec.layouts[tensor.layout]["stick_size"]
    stick_dim_order = sdsc_spec.layouts[tensor.layout]["stick_dim_order"]

    is_fp8_stick = (
        tensor.data_format == DataFormats.SEN143_FP8 and len(stick_size_list) > 1
    )

    if dim in stick_dim_order and len(stick_size_list) > 1:
        stick_idx = stick_dim_order.index(dim)
        other_idx = 1 - stick_idx
        other_stick_size = stick_size_list[other_idx]
    else:
        stick_idx = -1
        other_stick_size = 1

    return is_fp8_stick, other_stick_size, stick_idx


def gen_coord_info_value(
    size: int,
    nsplits: int,
    elems_per_stick: int,
    is_stick_dim: bool,
    is_stick_reduction: bool = False,
    conv_params=None,
    padding: str = "nopad",
    is_fp8_stick: bool = False,
    stick_idx: int = -1,
    tensor_idx: int = -1,
    opfunc: str = "",
    core_stride: int | None = None,
):
    """
    Args:
        conv_params: Dict with padding info for convolution ops; contains 'conv_padding' (pad type) and 'total_size' (per-core slice size for padding dims).
        If conv_params is not specified, pad type should default to "nopad" and total_size to size.
    """
    if conv_params is None:
        conv_params = {"conv_padding": padding, "stride_len": 1, "total_size": size}

    # How far the coordinate advances per core (the core_fold Affine alpha).
    # Two equivalent encodings feed this, one per caller:
    #   - Forward conv2d (#3284) passes core_stride (= out_per_core * stride) for
    #     an overlapping reduction window (stride < kernel), where the per-core
    #     window span (`size`) is larger than the per-core output stride.
    #   - Depthwise conv2d (#3510) and every other op advance by
    #     size * stride_len (stride_len==1, total_size==size for the non-conv
    #     default), i.e. contiguous non-overlapping tiling.
    # See _coord_per_core_size / _coord_core_stride.
    core_alpha = (
        core_stride if core_stride is not None else size * conv_params["stride_len"]
    )
    # The conv_params-derived padding string and per-core total_size are the
    # depthwise-conv (#3510) encoding. Forward conv2d (#3284) and every other op
    # keep their own `padding` arg and the windowed per-core `size` (which
    # already equals total_size for the non-conv default), so gate on the
    # depthwise opfunc to avoid overwriting forward conv's fullspan padding /
    # windowed elem_arr_0 factor.
    is_depthwise = opfunc == DEPTHWISE_CONV2D_OP
    stick_padding = str(conv_params["conv_padding"]) if is_depthwise else padding
    elem_arr_factor = conv_params["total_size"] if is_depthwise else size
    if not is_stick_dim:
        return {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 1,
            "padding": stick_padding,
            "folds": {
                "dim_prop_func": [
                    {"Affine": {"alpha_": core_alpha, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": 1, "beta_": 0}},
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
                        "factor_": elem_arr_factor,
                        "label_": "elem_arr_0",
                    },
                ],
            },
        }
    elif is_stick_dim and is_fp8_stick and not (stick_idx == 0):
        return {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 3,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {"Affine": {"alpha_": size, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": (size // 8), "beta_": 0}},
                    {"Affine": {"alpha_": 8, "beta_": 0}},
                    {"Affine": {"alpha_": 1, "beta_": 0}},
                ],
                "dim_prop_attr": [
                    {"factor_": nsplits, "label_": "core_fold"},
                    {"factor_": 1, "label_": "corelet_fold"},
                    {"factor_": 1, "label_": "row_fold"},
                    {"factor_": 64, "label_": "elem_arr_2"},
                    {"factor_": 2, "label_": "elem_arr_1"},
                    {"factor_": 1, "label_": "elem_arr_0"},
                ],
            },
        }
    elif is_stick_dim and tensor_idx == 0 and opfunc == "batchmatmulfp8":
        return {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 4,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {"Affine": {"alpha_": size, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": 128, "beta_": 0}},
                    {"Affine": {"alpha_": 8, "beta_": 0}},
                    {"Affine": {"alpha_": 64, "beta_": 0}},
                    {"Affine": {"alpha_": 1, "beta_": 0}},
                ],
                "dim_prop_attr": [
                    {"factor_": nsplits, "label_": "core_fold"},
                    {"factor_": 1, "label_": "corelet_fold"},
                    {"factor_": 1, "label_": "row_fold"},
                    {"factor_": (size // 128), "label_": "elem_arr_3"},
                    {"factor_": 8, "label_": "elem_arr_2"},
                    {"factor_": 2, "label_": "elem_arr_1"},
                    {"factor_": 8, "label_": "elem_arr_0"},
                ],
            },
        }
    else:
        return {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 2,
            "padding": padding,
            "folds": {
                "dim_prop_func": [
                    {
                        "Affine": {
                            "alpha_": elems_per_stick if is_stick_reduction else size,
                            "beta_": 0,
                        }
                    },
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": elems_per_stick, "beta_": 0}},
                    {"Affine": {"alpha_": 0 if is_stick_reduction else 1, "beta_": 0}},
                ],
                "dim_prop_attr": [
                    {"factor_": nsplits, "label_": "core_fold"},
                    {"factor_": 1, "label_": "corelet_fold"},
                    {"factor_": 1, "label_": "row_fold"},
                    {
                        "factor_": 1
                        if is_stick_reduction
                        else (size // elems_per_stick),
                        "label_": "elem_arr_1",
                    },
                    {"factor_": elems_per_stick, "label_": "elem_arr_0"},
                ],
            },
        }


# SDSC dim labels for the conv2d padding (output-spatial) axes.

_CONV2D_PAD_DIM_I = CONV2D_DIM_LABELS[2]
_CONV2D_PAD_DIM_J = CONV2D_DIM_LABELS[3]


def _build_padding_for_tensor(conv_params):
    """Build padding_ for tensor allocations, only when conv_params is non-empty."""
    if not conv_params:
        return {}
    return {
        "padding_": {
            str(_CONV2D_PAD_DIM_I): conv_params["pad_type"],
            str(_CONV2D_PAD_DIM_J): conv_params["pad_type"],
        }
    }


def get_conv_params(tensor_num, dim, opfunc, conv_params, size, splits):
    conv_padding = "nopad"
    total_size = size // splits
    padding_len = 0
    stride_len = 1
    if tensor_num == 0 and opfunc == DEPTHWISE_CONV2D_OP:
        required_keys = [
            "stride_i",
            "stride_j",
            "kernel_h",
            "kernel_w",
        ]
        missing = [k for k in required_keys if k not in conv_params]
        if missing:
            raise ValueError(f"Missing conv_params keys: {missing}")
        if "pad_type" in conv_params and str(dim) in (
            str(_CONV2D_PAD_DIM_I),
            str(_CONV2D_PAD_DIM_J),
        ):
            conv_padding = conv_params["pad_type"]
            padding_len = conv_params["pad_i"]
            stride_len = conv_params["stride_i"]
        if str(dim) == str(_CONV2D_PAD_DIM_I):
            total_size = (
                (size // splits) * conv_params["stride_i"] + conv_params["kernel_h"] - 1
            )
            padding_len = conv_params["pad_i"]
            stride_len = conv_params["stride_i"]
        elif str(dim) == str(_CONV2D_PAD_DIM_J):
            total_size = (
                (size // splits) * conv_params["stride_j"] + conv_params["kernel_w"] - 1
            )
            padding_len = conv_params["pad_j"]
            stride_len = conv_params["stride_j"]
    return {
        "conv_padding": conv_padding,
        "padding_len": padding_len,
        "stride_len": stride_len,
        "total_size": total_size,
    }


def _symbolic_split_info(
    tensor,
    work_slices: dict,
    symbolic_dims: dict,
) -> tuple[str, int, str] | None:
    """Return ``(sdsc_dim_name, split_count, pytorch_sym)`` if ``tensor`` is
    split across more than one core along a symbolic dim, else ``None``.

    A tensor qualifies iff all of the following hold:
      - it is a kernel tensor arg (``arg_index >= 0``); pool tensors are
        skipped because they have no kernel base to derive from.
      - one of the ``symbolic_dims`` has ``work_slices > 1`` (i.e. it is
        actually split across cores).
      - the tensor uses that dim (``tensor.scales[dim] > 0`` so the dim is
        neither reduced nor broadcast away for this tensor).

    Only the first qualifying dim is reported; the planner rejects plans with
    multiple symbolic splits per tensor upstream.
    """
    if tensor.arg_index < 0:
        return None
    for sdsc_dim, (pytorch_sym, _granularity, _max_val) in symbolic_dims.items():
        dim_sym = Symbol(sdsc_dim)
        split = work_slices.get(dim_sym)
        if split is None or split <= 1:
            continue
        if dim_sym not in tensor.scales or tensor.scales[dim_sym] <= 0:
            continue
        return sdsc_dim, int(split), pytorch_sym
    return None


def _tensor_has_symbolic_split(
    tensor,
    work_slices: dict,
    symbolic_dims: dict,
) -> bool:
    """True iff ``tensor`` is split across more than one core on a symbolic
    dim.  Thin boolean wrapper over ``_symbolic_split_info`` for sites that
    only need the yes/no answer.
    """
    return _symbolic_split_info(tensor, work_slices, symbolic_dims) is not None


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


def _tensor_tiled_by_symbol(tensor, sym) -> bool:
    """True iff `sym` contributes a nonzero term to this tensor's own
    tile advance.

    Real dimension symbols additionally require a positive scale (exclude
    reduction dims, whose stride describes intra-tile layout, not the
    inter-tile advance). Minted level symbols (Task 5) carry no
    dimension/scale identity of their own, so that half of the check is
    skipped for them; tensor.device_tile_advance_expr already only
    contains a minted symbol's term when this tensor genuinely advances
    at that level, so the coefficient check alone is both necessary and
    sufficient for minted symbols.

    This is the sole test for whether a reference advances -- callers do
    not pre-filter by any other per-tensor flag.
    """
    if sym in tensor.strides and tensor.scales.get(sym, 1) <= 0:
        return False
    if tensor.device_tile_advance_expr is None:
        return False
    return bool(coeff_through_floor(tensor.device_tile_advance_expr, sym))


def generate_sdsc(
    idx,
    sdsc_spec,
    symbols: list[int],
    symbol_id_offset: int = 0,
    tiled_symbols=None,
):
    """Generate SDSC JSON for one OpSpec.

    Returns a 4-tuple ``(sdsc_json, base_symbol_values, affine_strides, symbol_kinds)``:
    - ``sdsc_json``: the JSON dict to write to ``sdsc_N.json``
    - ``base_symbol_values``: list of HBM byte offsets registered in ``symbols``
    - ``affine_strides``: list (parallel to ``sdsc_spec.args``) of per-level
      stride lists.  Each element is a list of dicts, one per loop-nesting level
      (outermost first), where each dict maps ``tiled_sym -> stride_bytes`` for
      that level's tiled symbols.  Used by ``bundle.py`` to emit ``affine.apply``
      ops inside ``scf.for`` loops, with one stride per level mapped to the
      correct loop variable.
    - ``symbol_kinds``: list of ``SymbolKind`` parallel to ``base_symbol_values``.
      Classifies each symbol as a kernel base address, per-core derived
      address, or pool-allocated address.

    HBM addresses are registered as negative symbol IDs in the JSON and their
    values appended to ``symbols``, enabling ``affine.apply`` address
    computation in ``bundle.mlir`` for tiled loops.

    ``tensor.device_tile_advance_expr``: each tensor's own device-element-
    offset ``sympy.Expr | None``, symbolic in the real Inductor iteration
    symbols. For a symbol tiled at exactly one nesting level (the only case
    this function handles correctly -- a symbol tiled at more than one
    level has no single coefficient ``expr.coeff(sym)`` could return),
    ``expr.coeff(sym)`` is that level's byte stride once multiplied by
    ``num_bytes(tensor.data_format)``.
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
    operation_work_division = TensorWorkDivision(
        {dim: int(split) for dim, split in sdsc_spec.work_slices.items()},
        {dim: sdsc_spec.core_id_to_work_slice[dim] for dim in sdsc_spec.work_slices},
        num_cores=sdsc_spec.num_cores,
    )
    for tensor in sdsc_spec.args:
        if tensor.work_division is None:
            tensor.work_division = operation_work_division
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
    #   ("kernel_derived_symbolic", arg_index, core_idx)
    #                                 per-core address for a symbolic-dim split
    #                                 (c>0), keyed by (tensor, core) so it never
    #                                 collides with the bare-int kernel_derived key
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

    def offset_as_symbol(s, kind: SymbolKind):
        key: tuple | int
        if kind.is_pool:
            key = ("pool", s)
        elif kind.kind == "kernel":
            key = ("kernel", kind.arg_index)
        elif kind.kind == "kernel_slice":
            key = ("kernel_slice", kind.arg_index, kind.offset)
        elif kind.is_derived_symbolic:
            # Per-core symbolic address: key by (tensor, core) so every
            # (arg_index, core_idx) is a distinct registration and never
            # collides with a concrete kernel_derived key (a bare int addr).
            key = ("kernel_derived_symbolic", kind.arg_index, kind.core_idx)
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

    def _register_per_core_derived(
        tensor,
        c: int,
        addr: int,
        core0_addr: int,
        sliced_base_sym_idx: int,
        symbolic_split: tuple[str, int, str] | None,
    ) -> None:
        """Register the c>0 derived address for a kernel tensor.

        Routes to ``kernel_derived_symbolic`` when the tensor is split on a
        symbolic dim (the byte offset depends on the runtime dim size and is
        resolved by a later bundle arm), otherwise the concrete
        ``kernel_derived`` path.  ``addr`` is the compile-time (max-shape)
        address, used for the concrete path and as the symbols[] placeholder
        value for the symbolic path.
        """
        if symbolic_split is not None:
            _sdsc_dim_name, split_count, pytorch_sym = symbolic_split
            # TODO:  only TAG the per-core address as symbolic. The
            # runtime arith (core * ceildiv(S, split) * per_element_stride)
            # and the per-element stride it needs are the bundle-arm
            # follow-up, so nothing stride-related is computed here.
            offset_as_symbol(
                addr,
                SymbolKind.kernel_derived_symbolic(
                    arg_index=tensor.arg_index,
                    core_idx=c,
                    split_count=split_count,
                    base_sym_idx=sliced_base_sym_idx,
                    pytorch_sym=pytorch_sym,
                ),
            )
        else:
            offset_as_symbol(
                addr,
                _derived_kind(tensor.arg_index, core0_addr, addr, sliced_base_sym_idx),
            )

    # Compute per-tensor, per-level affine strides and register base addresses.
    # affine_strides[i] is a list of dicts, one per loop-nesting level
    # (outermost first), where each dict maps tiled_sym -> stride_bytes for
    # the symbols at that level that advance tensor i.  Empty list of dicts
    # (i.e. [{}] * n_levels or []) for non-tiled tensors.
    affine_strides: list[list[dict]] = []
    for tensor in sdsc_spec.args:
        if "lx" in tensor.allocation:
            # LX addresses are never registered as symbols in the SDSC JSON
            # (isStartAddrSymbolic_ is always unset for lx, and bundle.py's
            # _get_tensor_core_sym_id returns None for non-hbm components), so
            # affine.apply can never target an LX address today. A tiled
            # (advancing) lx tensor therefore has no way to express its
            # per-iteration address change in this preserved-loop path.
            # A non-advancing reference has no term in
            # device_tile_advance_expr, which _tensor_tiled_by_symbol
            # already detects directly.
            is_tiled_lx = any(
                _tensor_tiled_by_symbol(tensor, s)
                for level_syms in tiled_symbols
                for s in level_syms
            )
            if is_tiled_lx:
                raise NotImplementedError(
                    "Tiled (advancing) lx-allocated tensors are not yet supported."
                )
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
        # Per-core symbolic split: cores 1..n-1 of a kernel tensor split on a
        # symbolic dim get kernel_derived_symbolic addresses (byte offset
        # depends on the runtime dim size).  A symbolic-split dim that is ALSO
        # tiled for this tensor is out of scope: the per-core address would
        # need both a symbolic term and an affine.apply term.
        symbolic_split = _symbolic_split_info(
            tensor, sdsc_spec.work_slices, symbolic_dims
        )
        if symbolic_split is not None:
            sym_dim_name = symbolic_split[0]
            sym_dim = Symbol(sym_dim_name)
            # Real-symbol fast path: s IS the dim symbol (already renamed
            # to its SDSC dim label by symbol_mapping), so name equality
            # against sym_dim_name is a correct, direct test.
            #
            # Minted-symbol path (spyre_kernel._get_or_mint_level_symbol):
            # a minted symbol names a loop-nesting *level*, not a
            # dimension -- _general_tile_advance (spyre_kernel.py) sums
            # every host dim tiled at a level into ONE combined
            # coefficient on that level's minted symbol before this
            # tensor's device_tile_advance_expr is ever built, so by the
            # time we get here there is no way to recover, from a
            # nonzero coeff(minted_sym) alone, *which* of this tensor's
            # active dims that coefficient came from (see fix-loop
            # round-1 review: a tensor with two active dims, tiled only
            # on one of them, previously false-positived on the other
            # merely because it was also active and the tensor advanced
            # via *some* dim).
            #
            # Absent that per-dimension provenance, the only sound test
            # (no false positives) is: flag `sym_dim_name` only when it
            # is this tensor's *sole* active (non-reduced) dim -- then a
            # nonzero combined coefficient cannot be attributed to any
            # other dim, because there is no other dim. This is a
            # deliberate narrowing versus "any tiling at all, on any
            # dim" -- it can under-detect (miss a real conflict on a
            # tensor with 2+ active dims where sym_dim_name genuinely is
            # the tiled one) but never over-detects, which is the
            # correctness-critical direction for a False positive to
            # avoid: it would otherwise reject support for supported
            # ops using this check.
            active_dims = [d for d in tensor.strides if tensor.scales.get(d, 1) > 0]
            tensor_advances_at_some_level = (
                tensor.device_tile_advance_expr is not None
                and any(
                    coeff_through_floor(tensor.device_tile_advance_expr, s)
                    for level_syms in tiled_symbols
                    for s in level_syms
                )
            )
            tiled_on_split_dim = any(
                str(s) == sym_dim_name
                for level_syms in tiled_symbols
                for s in level_syms
                if s in tensor.strides
            ) or (
                sym_dim in tensor.strides
                and tensor.scales.get(sym_dim, 1) > 0
                and active_dims == [sym_dim]
                and tensor_advances_at_some_level
            )
            if tiled_on_split_dim:
                raise Unsupported(
                    f"Symbolic dim '{sym_dim_name}' is both split across "
                    f"cores and tiled on the tensor at arg_index="
                    f"{tensor.arg_index}; per-core symbolic addresses inside "
                    "tiled loops are not supported."
                )
        if tensor.arg_index >= 0:
            # Kernel tensors: register the raw base address first so bundle.py
            # can emit the input_arg function parameter.
            #
            # On the symbolic path, tensor.start_address = arg_index + tile_offset_bytes,
            # where tile_offset_bytes is the per-tile byte advance computed for the
            # affine-stride path.  We always register the raw kernel symbol keyed by
            # arg_index so that bundle.py emits exactly one !sdscbundle.input_arg
            # parameter per logical tensor, regardless of how many tiles reference it.
            raw_base = tensor.arg_index  # sentinel value for this arg
            offset_as_symbol(raw_base, SymbolKind.kernel(arg_index=tensor.arg_index))
            # Derive the 0-based symbols[] index of the kernel symbol from its
            # registered ID.  Must be looked up (not inferred from current
            # len(local_symbols)) because the same arg_index may have been
            # registered already by an earlier tensor in this SDSC, in which case
            # the offset_as_symbol call above was a no-op.
            kernel_sym_idx = abs(local_symbols[("kernel", tensor.arg_index)]) - 1
            # tile_offset_bytes: arg.allocation['hbm'] advances by i*stride for
            # tile i, so start_address = arg_index + tile_offset. tile_offset_bytes
            # == 0 for tile 0, positive for later tiles.
            tile_offset_bytes = tensor.start_address - tensor.arg_index
            # total_slice_offset: combine the per-tile byte offset with any
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
        # level that tile this tensor (see _tensor_tiled_by_symbol -- a nonzero
        # coeff on tensor.device_tile_advance_expr, with real dimension symbols
        # additionally required to have a positive scale).
        # Exclude symbols whose scale is negative: those are reduced dimensions
        # whose stride describes element layout within one tile, not the advance
        # between tiles.  Tiling by a reduction-dim symbol would incorrectly
        # advance the base address of a pool output past its single allocated slot.
        # A tile-local scratch tensor reused every iteration (see unroll.py)
        # never advances either -- it simply has no term in
        # device_tile_advance_expr, which _tensor_tiled_by_symbol already
        # detects directly.
        per_level_strides: list[dict] = []
        any_tiled = False
        for level_syms in tiled_symbols:
            tensor_tiled_at_level = [
                s for s in level_syms if _tensor_tiled_by_symbol(tensor, s)
            ]
            strides_for_level: dict = {}
            for s in tensor_tiled_at_level:
                coeff = (
                    coeff_through_floor(tensor.device_tile_advance_expr, s)
                    if tensor.device_tile_advance_expr is not None
                    else 0
                )
                strides_for_level[s] = int(coeff) * nb
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
                        _register_per_core_derived(
                            tensor,
                            c,
                            addr,
                            core0_addr,
                            sliced_base_sym_idx,
                            symbolic_split,
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
                        # Symbolic split on a non-tiled dim combined with
                        # tiling on another dim still routes through the
                        # helper; the split==tiled case is rejected above.
                        _register_per_core_derived(
                            tensor,
                            c,
                            addr,
                            core0_addr,
                            sliced_base_sym_idx,
                            symbolic_split,
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
        is_pool_tensor = tensor.arg_index < 0 and "hbm_pool" in tensor.allocation
        # Recompute the symbolic-split status so c>0 cores resolve to the
        # ("kernel_derived_symbolic", arg_index, core_idx) key the per-tensor
        # loop registered.  Pure function of the tensor + work_slices, so this
        # matches the registration decision exactly.
        symbolic_split = _symbolic_split_info(
            tensor, sdsc_spec.work_slices, symbolic_dims
        )
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
            elif addr == core0_addr_lookup:
                # Non-split tensor: all cores share core 0's address, so no
                # derived symbol was registered, reuse the sliced-base key.
                key = c0_slice_key
            elif symbolic_split is not None:
                # Symbolic-dim split: c>0 registered a kernel_derived_symbolic
                # symbol keyed by (tensor, core).
                key = ("kernel_derived_symbolic", tensor.arg_index, c)
            else:
                # c>0 concrete per-core derived address (bare int key).
                key = addr
            result[f"[{c}, 0, 0]"] = str(local_symbols[key])
        return result

    def _build_coord_info(op, tensor, tensor_idx: int) -> dict:
        """Builds the coordinate information for all dimensions of a tensor.

        Computes layout, slicing, and FP8 parameter metadata required for
        generating coordinate info across each dimension specified in the tensor's
        layout order.  For windowed conv2d input spatial dims the per-core size and
        core-fold stride come from _coord_per_core_size / _coord_core_stride so
        overlapping windows (stride < kernel) distribute correctly; non-conv ops
        keep the plain per-core size and a None core_stride (unchanged behavior).

        Args:
            tensor: The tensor object containing layout, scale, and data format specs.
            tensor_idx: The index of the tensor within the parent operation.

        Returns:
            dict[str, Any]: A dictionary mapping string dimension names to their
                corresponding generated coordinate information value structure.
        """
        layout = sdsc_spec.layouts[tensor.layout]
        dim_order = _filter_window_dims(layout["dim_order"], tensor.layout)
        stick_dim_order = layout["stick_dim_order"]
        is_input = tensor_idx < sdsc_spec.num_inputs
        result = {}
        for dim in dim_order:
            dim_str = str(dim)
            scale = tensor.scales[dim]
            is_tiled = scale == 1
            nsplits = tensor.work_division.work_slices[dim] if is_tiled else 1
            # dim_size feeds get_conv_params (#3510). size uses #3284's
            # _coord_per_core_size, a safe superset: it returns the windowed
            # span only for forward conv2d (opfunc=="conv2d") and otherwise
            # falls back to dim_size // nsplits -- identical to #3510's path for
            # depthwise and every other op.
            dim_size = _coord_size(dim_str, sdsc_spec.iteration_space[dim], is_input)
            size = _coord_per_core_size(dim, is_input, nsplits) if is_tiled else 1
            is_fp8, _, st_idx = _compute_fp8_coord_params(tensor, dim, sdsc_spec)
            conv_params = (
                get_conv_params(
                    tensor_idx,
                    dim,
                    sdsc_spec.opfunc,
                    sdsc_spec.conv_params,
                    dim_size,
                    nsplits,
                )
                if op == DEPTHWISE_CONV2D_OP
                else None
            )

            result[dim_str] = gen_coord_info_value(
                size=size,
                nsplits=nsplits,
                elems_per_stick=tensor.data_format.elems_per_stick(),
                is_stick_dim=(dim in stick_dim_order),
                is_stick_reduction=(scale == -2),
                is_fp8_stick=is_fp8,
                stick_idx=st_idx,
                tensor_idx=tensor_idx,
                opfunc=sdsc_spec.opfunc,
                padding=_coord_padding(dim_str, is_input),
                # conv_params (#3510) drives depthwise padding; core_stride
                # (#3284) is non-None only for forward-conv windowed dims. Each
                # is inert for the other op, so both are always passed.
                conv_params=conv_params,
                core_stride=(
                    _coord_core_stride(dim, is_input, nsplits) if is_tiled else None
                ),
            )
        return result

    def _filter_window_dims(dims: list, layout_key: str | None = None) -> list:
        """Drop the op's reduction-window dims (e.g. pool/conv ki/kj) from a dim order.

        sdsc_spec.window_dims is empty for ops without a reduction window, so
        this is a no-op for them.

        Exception: the KERNEL (weight) layout of conv2d carries ki/kj as explicit
        physical axes (weight is [kj, ki, in, out]), so they must NOT be stripped
        there.  For the activation (INPUT) and OUTPUT layouts ki/kj appear only as
        folded window offsets and are correctly removed.
        """
        if layout_key == "KERNEL":
            return list(dims)
        return [d for d in dims if str(d) not in sdsc_spec.window_dims]

    def _tensor_sched_layout_dims(
        dim_order: list, layout_key: str | None = None
    ) -> list:
        """Return a tensor's own dim_order for scheduleTree_, minus window dims.

        scheduleTree_ layoutDimOrder_ must use the per-tensor dim_order, NOT the
        layout-canonical order.  Multiple tensors may share a layout label (same
        symbol Counter, different ordering), so sdsc_spec.layouts[label]["dim_order"]
        is only correct for the tensor that created that label.
        """
        return _filter_window_dims(dim_order, layout_key)

    def _tensor_input_padding(tensor) -> dict:
        """input_coord_padding restricted to the dims this tensor actually has.

        For multi-input windowed ops (conv2d) only the activation carries the
        padded spatial dims (i/j); the weight (KERNEL) has none, so it gets no
        padding_ entry.  For single-input pools this is a no-op (the sole input
        has i/j).
        """
        tdims = {
            str(d) for d in _tensor_sched_layout_dims(tensor.dim_order, tensor.layout)
        }
        return {
            dim: pad
            for dim, pad in sdsc_spec.input_coord_padding.items()
            if dim in tdims
        }

    def _coord_size(dim, default: int, is_input: bool) -> int:
        """Per-dim coordinate size, overridable for input tensors (pool pads H/W)."""
        if is_input:
            return sdsc_spec.input_coord_sizes.get(str(dim), default)
        return default

    def _coord_per_core_size(dim, is_input: bool, nsplits: int) -> int:
        """Per-core coordinate size for a tensor dim.

        For a windowed conv input spatial dim (i/j, present in padding_sizes) the
        per-core input footprint is the window span of the per-core output slice:
        ``(out_per_core - 1)*stride + dilation*(k - 1) + 1``.  This is the number
        of input elements the reduction window (ki/kj) is distributed over on
        each core.  Dividing the *total* padded input span by the core count
        instead (the plain path) undercounts for an overlapping window
        (stride < kernel): e.g. in=8, cores=6 gives 1 < kernel, so the ki loop
        has nothing to distribute.  Conv-only: avgpool keeps the plain path (for
        its non-overlapping stride==kernel windows the two coincide anyway).
        """
        ps = sdsc_spec.padding_sizes.get(str(dim)) if is_input else None
        if sdsc_spec.opfunc == "conv2d" and ps is not None and "windowDim_" in ps:
            out_per_core = sdsc_spec.iteration_space[dim] // nsplits
            stride = int(ps.get("stride_", 1))
            dilation = int(ps.get("dilation_", 1))
            k = int(sdsc_spec.iteration_space.get(Symbol(ps["windowDim_"]), 1))
            return (out_per_core - 1) * stride + dilation * (k - 1) + 1
        return (
            _coord_size(str(dim), sdsc_spec.iteration_space[dim], is_input) // nsplits
        )

    def _coord_core_stride(dim, is_input: bool, nsplits: int) -> int | None:
        """Core-fold stride (advance per core) for a windowed input spatial dim.

        Returns ``out_per_core * stride`` -- how far the input window origin moves
        between adjacent cores -- so overlapping windows (conv stride < kernel)
        step correctly while each core still reads the full window span
        (_coord_per_core_size).  Returns None for non-windowed dims (and for
        avgpool), where gen_coord_info_value defaults the core stride to the
        per-core size -- so avgpool's original behavior is unchanged.
        """
        ps = sdsc_spec.padding_sizes.get(str(dim)) if is_input else None
        if sdsc_spec.opfunc == "conv2d" and ps is not None and "windowDim_" in ps:
            out_per_core = sdsc_spec.iteration_space[dim] // nsplits
            return out_per_core * int(ps.get("stride_", 1))
        return None

    def _coord_padding(dim, is_input: bool) -> str:
        """Per-dim coordinate padding tag, overridable for input tensors."""
        if is_input:
            return sdsc_spec.input_coord_padding.get(str(dim), "nopad")
        return "nopad"

    def _memorg_extra(is_input: bool, alloc_node: str) -> dict:
        """Extra memOrg_ padding fields, emitted only when the op needs them."""
        if not sdsc_spec.emit_memorg_padding:
            return {}
        extra: dict = {
            "isPadded": 1 if is_input else 0,
            "isZeroPadded": 0,
        }
        # dsOffset / allocateNode_ are part of the forward-conv (#3284) and
        # general memOrg contract; depthwise conv (#3510) intentionally omits
        # them. Restore them for every non-depthwise op so forward conv2d
        # matches its golden SDSC.
        if sdsc_spec.opfunc != DEPTHWISE_CONV2D_OP:
            extra["dsOffset"] = 0
            extra["allocateNode_"] = alloc_node
        return extra

    return (
        {
            f"{idx}_{sdsc_spec.opfunc}": {
                # Source-to-kernel provenance. JSON key uses the SDSC
                # trailing-underscore convention; the Python field stays
                # `debug_handle`. dxp_standalone ignores unknown keys.
                "debug_handle_": (
                    sdsc_spec.debug_handle.to_dict()
                    if sdsc_spec.debug_handle is not None
                    else None
                ),
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
                # The top-level map is the operation schedule. Each shuffle
                # tensor's physical owners are carried separately on its
                # allocation coordinates below.
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
                                **(
                                    {"paddingSizes_": sdsc_spec.padding_sizes}
                                    if sdsc_spec.padding_sizes
                                    else {}
                                ),
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
                                        "paddingSizes_": sdsc_spec.padding_sizes_per_core
                                        if sdsc_spec.padding_sizes_per_core
                                        else sdsc_spec.padding_sizes,
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
                                        "paddingSizes_": sdsc_spec.padding_sizes_per_core
                                        if sdsc_spec.padding_sizes_per_core
                                        else sdsc_spec.padding_sizes,
                                    },
                                }
                            },
                            "primaryDsInfo_": {
                                label: {
                                    "layoutDimOrder_": [
                                        str(dim)
                                        for dim in _filter_window_dims(
                                            layout_info["dim_order"], label
                                        )
                                    ],
                                    "stickDimOrder_": [
                                        str(dim)
                                        for dim in layout_info["stick_dim_order"]
                                    ],
                                    "stickSize_": layout_info["stick_size"],
                                    **(
                                        {"stickRepl_": [1]}
                                        if sdsc_spec.stick_replication
                                        else {}
                                    ),
                                }
                                for label, layout_info in sdsc_spec.layouts.items()
                            },
                            **(
                                {"pdsRelation_": {"isPdsReuse": 1}}
                                if sdsc_spec.pds_reuse
                                else {}
                            ),
                            "scheduleTree_": [
                                {
                                    "nodeType_": "allocate",
                                    "name_": f"allocate-Tensor{i}_{'lx' if 'lx' in tensor.allocation else 'hbm'}",
                                    "prev_": "",
                                    "ldsIdx_": i,
                                    # NOTE: "hbm"/"lx" here are sdsc fields and are
                                    # not to be confused with the internal
                                    # layout.allocation dict keys ("hbm"/"lx"/
                                    # "hbm_pool").
                                    "component_": "lx"
                                    if "lx" in tensor.allocation
                                    else "hbm",
                                    **(
                                        _build_padding_for_tensor(sdsc_spec.conv_params)
                                        if sdsc_spec.opfunc == DEPTHWISE_CONV2D_OP
                                        and i == 0
                                        else {}
                                    ),
                                    **(
                                        {"isStartAddrSymbolic_": 1}
                                        if "lx" not in tensor.allocation
                                        else {}
                                    ),
                                    "layoutDimOrder_": [
                                        str(dim)
                                        for dim in _tensor_sched_layout_dims(
                                            tensor.dim_order, tensor.layout
                                        )
                                    ],
                                    "maxDimSizes_": [
                                        tensor.max_dim_sizes[dim]
                                        for dim in _tensor_sched_layout_dims(
                                            tensor.dim_order, tensor.layout
                                        )
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
                                        {"padding_": _tensor_input_padding(tensor)}
                                        if (
                                            i < sdsc_spec.num_inputs
                                            and _tensor_input_padding(tensor)
                                        )
                                        else {}
                                    ),
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
                                                if str(dim)
                                                in {
                                                    str(d)
                                                    for d in _tensor_sched_layout_dims(
                                                        tensor.dim_order, tensor.layout
                                                    )
                                                }
                                            }
                                        }
                                        if tensor.backGap
                                        else {}
                                    ),
                                    "coordinates_": {
                                        "coordInfo": _build_coord_info(
                                            sdsc_spec.opfunc, tensor, i
                                        ),
                                        "coreIdToWkSlice_": (
                                            tensor.work_division.to_core_slices(
                                                tensor.work_division.num_cores
                                                or sdsc_spec.num_cores
                                            )
                                            if sdsc_spec.opfunc == "shuffle"
                                            and tensor.work_division is not None
                                            else {}
                                        ),
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
                                        for dim in _filter_window_dims(
                                            sdsc_spec.layouts[tensor.layout][
                                                "dim_order"
                                            ],
                                            tensor.layout,
                                        )
                                    ],
                                    "wordLength": num_bytes(tensor.data_format),
                                    "dataFormat_": tensor.data_format.name,
                                    # Index tensors must reside in HBM; the Spyre
                                    # engine does not support indirect addressing
                                    # through LX scratchpad.
                                    # NOTE: "hbm"/"lx" here are sdsc fields and are
                                    # not to be confused with the internal
                                    # layout.allocation dict keys ("hbm"/"lx"/
                                    # "hbm_pool").
                                    "memOrg_": {"hbm": {"isPresent": 1}}
                                    if tensor.is_index_tensor
                                    else {
                                        "hbm": {
                                            "isPresent": 1,
                                            **(
                                                _memorg_extra(
                                                    i < sdsc_spec.num_inputs,
                                                    f"allocate-Tensor{i}_hbm",
                                                )
                                                if sdsc_spec.opfunc
                                                != DEPTHWISE_CONV2D_OP
                                                or i == 0
                                                else {}
                                            ),
                                        },
                                        "lx": {
                                            "isPresent": 1,
                                            **(
                                                _memorg_extra(
                                                    i < sdsc_spec.num_inputs,
                                                    "",
                                                )
                                                if sdsc_spec.opfunc
                                                != DEPTHWISE_CONV2D_OP
                                                or i == 0
                                                else {}
                                            ),
                                        },
                                    }
                                    if "lx" not in tensor.allocation
                                    else (
                                        {
                                            "lx": {
                                                "isPresent": 1,
                                                "isPadded": 1,
                                            }
                                        }
                                        if (
                                            i == 0
                                            and sdsc_spec.opfunc == DEPTHWISE_CONV2D_OP
                                        )
                                        else {"lx": {"isPresent": 1}}
                                    ),
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
