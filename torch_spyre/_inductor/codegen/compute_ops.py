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


from torch_spyre._C import encode_constant, DataFormats
from sympy import Symbol


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


def _tiled_byte_stride(tensor, tiled_sym, iteration_space) -> int:
    """Byte stride per loop iteration for a single tiled dimension.

    The coarse_tile pass already divided iteration_space[tiled_sym].range by
    the loop count, so that range is the per-iteration element count.  The full
    per-iteration byte advancement is:
        per_iter_elems * device_stride_for_dim * bytes_per_element
    """
    per_iter_range = iteration_space[tiled_sym]
    return int(
        per_iter_range * tensor.strides[tiled_sym] * num_bytes(tensor.data_format)
    )


def generate_sdsc(
    idx,
    sdsc_spec,
    symbols: list[int],
    symbol_id_offset: int = 0,
    tiled_symbols=None,
    use_symbols: bool = False,
):
    """Generate SDSC JSON for one OpSpec.

    Returns a 3-tuple ``(sdsc_json, base_symbol_values, affine_strides)``:
    - ``sdsc_json``: the JSON dict to write to ``sdsc_N.json``
    - ``base_symbol_values``: list of base HBM byte offsets registered in
      ``symbols``; empty when ``use_symbols=False``
    - ``affine_strides``: list (parallel to ``sdsc_spec.args``) of dicts
      ``{tiled_sym: stride_bytes}`` for tiled HBM tensors; always empty when
      ``use_symbols=False``.  Used by ``bundle.py`` to emit ``affine.apply``
      ops inside ``scf.for`` loops.

    When ``use_symbols=False``, HBM tensor addresses are baked directly as
    concrete integers into the SDSC JSON.  No symbol IDs are registered and ``symbols``
    is not modified.  This is the default until backend symbol-table support
    is complete.

    When ``use_symbols=True``, HBM addresses are registered as negative symbol
    IDs in the JSON and their values appended to ``symbols``, enabling
    ``affine.apply`` address computation in ``bundle.mlir`` for tiled loops.
    """
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

    # local_symbols maps base HBM byte offset -> globally-unique negative symbol id.
    # symbol_id_offset ensures ids are unique across all SDSCs in the bundle.
    # For tiled tensors the base is the iteration-0 address (tiled dims contribute 0);
    # for non-tiled tensors it is the full per-core address (as before).
    #
    # NOTE: no cross-SDSC deduplication — each call to offset_as_symbol within
    # this SDSC gets its own sequential ID and appends to symbols.  Two SDSCs
    # that happen to share a base address will emit two separate arith.constant
    # declarations in bundle.mlir.  This keeps symbol IDs contiguous with the
    # symbols list indices: symbols[abs(id)-1] is always the value for id.
    #
    # When use_symbols=False this dict stays empty (symbols is not modified).
    local_symbols: dict[int, int] = {}

    if use_symbols:

        def offset_as_symbol(s):
            if s not in local_symbols:
                local_symbols[s] = -(symbol_id_offset + len(local_symbols) + 1)
                symbols.append(s)
            return local_symbols[s]

        # Compute per-tensor affine strides and register base addresses in symbols.
        # affine_strides[i] is {tiled_sym: stride_bytes} for tensor i (empty if
        # non-tiled/lx).
        affine_strides: list[dict] = []
        for tensor in sdsc_spec.args:
            if "lx" in tensor.allocation:
                affine_strides.append({})
                continue
            tensor_tiled = [s for s in tiled_symbols if s in tensor.strides]
            if not tensor_tiled:
                # Non-tiled HBM: register full per-core addresses.
                for c in range(sdsc_spec.num_cores):
                    full_addr = tensor.start_address + core_idx_to_slice_offset(
                        tensor, core_id_to_wk_slice[str(c)], sdsc_spec.work_slices
                    ) * num_bytes(tensor.data_format)
                    offset_as_symbol(full_addr)
                affine_strides.append({})
            else:
                # Tiled HBM: symbol value = per-core iter-0 base address.
                # The affine map adds loop_var * tile_stride on top at runtime.
                strides_for_tensor = {}
                for s in tensor_tiled:
                    strides_for_tensor[s] = _tiled_byte_stride(
                        tensor, s, sdsc_spec.iteration_space
                    )
                for c in range(sdsc_spec.num_cores):
                    base_addr = tensor.start_address + core_idx_to_slice_offset(
                        tensor, core_id_to_wk_slice[str(c)], sdsc_spec.work_slices
                    ) * num_bytes(tensor.data_format)
                    offset_as_symbol(base_addr)
                affine_strides.append(strides_for_tensor)

        def _start_addr_data(tensor):
            if "lx" in tensor.allocation:
                return {
                    f"[{c}, 0, 0]": str(tensor.start_address)
                    for c in range(sdsc_spec.num_cores)
                }
            result = {}
            for c in range(sdsc_spec.num_cores):
                addr = tensor.start_address + core_idx_to_slice_offset(
                    tensor, core_id_to_wk_slice[str(c)], sdsc_spec.work_slices
                ) * num_bytes(tensor.data_format)
                result[f"[{c}, 0, 0]"] = str(offset_as_symbol(addr))
            return result

    else:
        # use_symbols=False: bake concrete HBM addresses directly into the JSON,
        # mirroring the LX tensor path.  symbols and local_symbols are not modified.
        affine_strides = [{} for _ in sdsc_spec.args]

        def _start_addr_data(tensor):
            if "lx" in tensor.allocation:
                # Mirrors the use_symbols=True branch above.
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
                            "dataStageParam_": {
                                "0": {
                                    "ss_": {
                                        "name_": "core",
                                        **{
                                            str(dim) + "_": size
                                            // sdsc_spec.work_slices[dim]
                                            for dim, size in sdsc_spec.iteration_space.items()
                                        },
                                    },
                                    "el_": {
                                        "name_": "core",
                                        **{
                                            str(dim) + "_": size
                                            // sdsc_spec.work_slices[dim]
                                            for dim, size in sdsc_spec.iteration_space.items()
                                        },
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
                                                str(dim): {
                                                    "-1": str(gap)  # HBM is -1
                                                }
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
                                    "memOrg_": {
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
                                    ],
                                    "outputLabeledDs": [
                                        f"Tensor{out_idx}-idx{out_idx}"
                                    ],
                                }
                            ],
                        }
                    }
                ],
            }
        },
        list(local_symbols.keys()),
        affine_strides,
    )
