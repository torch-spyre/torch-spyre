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

import math
from dataclasses import dataclass
from torch_spyre._C import encode_constant, DataFormats


@dataclass
class DimInfo:
    label: str
    index: int
    nsplits: int
    size: int
    split_size: int


def num_bytes(df: DataFormats) -> int:
    """Try to avoid using this method; it is a bad API due to sub-byte datatypes"""
    num_elems = df.elems_per_stick()
    if num_elems > 128:
        raise RuntimeError(f"sub-byte dataformat {df}")
    return 128 // num_elems


def generate_constant_info(data_format, **kwargs):
    if "op_info" not in kwargs or "constants" not in kwargs["op_info"]:
        return "{}"
    constant_info = {}
    for name, value in kwargs["op_info"]["constants"].items():
        ci = {
            "dataFormat_": data_format.name,
            "name_": name,
            "data_": {
                "dim_prop_func": [{"Const": {}}, {"Const": {}}, {"Map": {}}],
                "dim_prop_attr": [
                    {"factor_": 1, "label_": "core"},
                    {"factor_": 1, "label_": "corelet"},
                    {"factor_": 1, "label_": "time"},
                ],
                "data_": {"[0, 0, 0]": [encode_constant(value, data_format)]},
            },
        }
        constant_info[f"{len(constant_info)}"] = ci
    return constant_info


def gen_coord_info_value(
    size: int,
    nsplits: int,
    elems_per_stick: int,
    is_stick_dim: bool,
    is_stick_reduction: bool,
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


def get_ordered_dim_info_list(
    dim_labels: list[str],
    dim_indices: list[int],
    dim_sizes: list[int],
    dim_splits: list[int],
):
    return [
        DimInfo(
            label=label,
            index=index,
            nsplits=nsplits,
            size=size,
            split_size=(size // nsplits),
        )
        for label, index, size, nsplits in zip(
            dim_labels, dim_indices, dim_sizes, dim_splits
        )
    ]


def generate_sfp_op(pointers, *, op, dimensions, inputs, outputs, reduction, **kwargs):
    tensors = inputs + outputs

    data_format = inputs[0]["ddtype"]

    d3 = len(dimensions) >= 3

    ndim = len(dimensions)
    assert ndim <= 3

    # implement core division on stick dimension
    cores = 1

    if "op_info" in kwargs and "core_division" in kwargs["op_info"]:
        # enable work division for non-reduction only for now
        if not reduction:
            split_idx = -3 if d3 else 0  # split along stick dim
            cores = kwargs["op_info"]["core_division"][-1][split_idx]
            # FIXME: cores should be the product of list of splits

    # TODO: fix constant generation with multiple cores
    if "op_info" in kwargs and "constants" in kwargs["op_info"]:
        cores = 1

    if reduction and tensors[-1]["scale"][-1] == 1:
        op += "nonstick"

    # FIXME: use core_division instead of cores to fill the list of splits
    if ndim == 1:
        dim_labels = ["out"]
        dim_indices = [0]
        dim_splits = [cores]
        core_id_to_wk_slice = {str(i): {"out": i} for i in range(cores)}
    elif ndim == 2:
        dim_labels = ["mb", "out"]
        dim_indices = [0, 1]
        dim_splits = [1, cores]
        core_id_to_wk_slice = {str(i): {"mb": 0, "out": i} for i in range(cores)}
    else:  # ndim == 3
        # NOTE: Pytorch host tensor shape is [mb, x, out] from the most to the
        #       least significant dimension. Here when filling in the
        #       layoutDimOrder, we use 3d generic stick layout on device
        #       [mb, out, x] from the least to the most significant
        #       dimension.
        dim_labels = ["mb", "out", "x"]
        dim_indices = [0, 2, 1]
        dim_splits = [1, cores, 1]
        core_id_to_wk_slice = {
            str(i): {"mb": 0, "x": 0, "out": i} for i in range(cores)
        }

    # reorder sizes according to layoutDimOrder
    dim_sizes = [dimensions[i] for i in dim_indices]
    dim_info = get_ordered_dim_info_list(
        dim_labels,
        dim_indices,
        dim_sizes,
        dim_splits,
    )

    return {
        op: {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": cores,
            "coreIdToDsc_": {str(c): 0 for c in range(cores)},
            "numWkSlicesPerDim_": {di.label: di.nsplits for di in dim_info},
            "coreIdToWkSlice_": core_id_to_wk_slice,
            "coreIdToDscSchedule": {str(c): [[-1, 0, 0, 0]] for c in range(cores)},
            "dscs_": [
                {
                    op: {
                        "numCoresUsed_": cores,
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": [c for c in range(cores)],
                        "N_": {
                            "name_": "n",
                            **{
                                di.label + "_": di.size for di in dim_info
                            },  # dim sizes before split
                        },
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size for di in dim_info
                                    },
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size for di in dim_info
                                    },
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            "OUTPUT": {
                                "layoutDimOrder_": dim_labels,
                                "stickDimOrder_": ["out"],
                                "stickSize_": [inputs[0]["ddtype"].elems_per_stick()],
                            }
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": f"allocate-Tensor{i}_hbm",
                                "prev_": "",
                                "ldsIdx_": i,
                                "component_": "hbm",
                                "layoutDimOrder_": dim_labels,
                                "maxDimSizes_": [-1] * len(dim_labels),
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {"factor_": cores, "label_": "core"},
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        f"[{c}, 0, 0]": str(
                                            pointers[tensor["name"]]
                                            + c
                                            # calculate the prod of dim sizes
                                            # less significant than chosen split dim i.e. the stick
                                            * math.prod(dim_sizes[:2])
                                            * num_bytes(tensor["ddtype"])
                                            // cores
                                        )
                                        for c in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        di.label: gen_coord_info_value(
                                            size=di.split_size
                                            if (tensor["scale"][di.index] == 1)
                                            else 1,
                                            nsplits=di.nsplits,
                                            elems_per_stick=tensor[
                                                "ddtype"
                                            ].elems_per_stick(),
                                            is_stick_dim=(di.label == "out"),
                                            is_stick_reduction=(
                                                di.label == "out"
                                                and tensor["scale"][di.index] == -1
                                            ),
                                        )
                                        for di in dim_info
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            }
                            for i, tensor in enumerate(tensors)
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": i,
                                "dsName_": f"Tensor{i}",
                                "dsType_": "OUTPUT",
                                "scale_": [
                                    (
                                        tensor["scale"][di.index]
                                        # TODO: revisit whether this special case can be removed
                                        #       pending change in deeptools
                                        if not (
                                            di.label == "out"
                                            and tensor["scale"][di.index] == -1
                                        )
                                        else -2
                                    )
                                    for di in dim_info
                                ],
                                "wordLength": num_bytes(tensor["ddtype"]),
                                "dataFormat_": tensor["ddtype"].name,
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                },
                            }
                            for i, tensor in enumerate(tensors)
                        ],
                        "constantInfo_": generate_constant_info(data_format, **kwargs),
                        "computeOp_": [
                            {
                                "exUnit": "sfp",
                                "opFuncName": op,
                                "attributes_": {
                                    "dataFormat_": data_format.name,
                                    "fidelity_": "regular",
                                },
                                "location": "Inner",
                                "inputLabeledDs": [
                                    f"Tensor{i}-idx{i}"
                                    for i in range(
                                        len(tensors if reduction else inputs)
                                    )
                                ],
                                "outputLabeledDs": [
                                    f"Tensor{i}-idx{i}"
                                    for i in range(len(inputs), len(tensors))
                                ],
                            }
                        ],
                    }
                }
            ],
        }
    }


def generate_matmul(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    # [mb=dim0, in=dim1] @ [in=dim1, out=dim2]

    # implement core division on stick dimension
    cores = 1
    if "op_info" in kwargs and "core_division" in kwargs["op_info"]:
        cores = kwargs["op_info"]["core_division"][-1][0]

    dim_labels = ["mb", "in", "out"]
    dim_indices = [0, 1, 2]
    dim_sizes = dimensions
    dim_splits = [1, 1, cores]
    dim_split_sizes = [size // nsplits for size, nsplits in zip(dim_sizes, dim_splits)]

    # map label to dim_info
    dim_info_dict = {
        label: DimInfo(
            label=label,
            index=index,
            nsplits=nsplits,
            size=size,
            split_size=split_size,
        )
        for label, index, nsplits, size, split_size in zip(
            dim_labels, dim_indices, dim_splits, dim_sizes, dim_split_sizes
        )
    }

    input_layoutDimOrder = ["mb", "in"]
    kernel_layoutDimOrder = ["in", "out"]
    output_layoutDimOrder = ["mb", "out"]

    return {
        op: {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": cores,
            "coreIdToDsc_": {str(i): 0 for i in range(cores)},
            "numWkSlicesPerDim_": {"mb": 1, "in": 1, "out": cores},
            "coreIdToWkSlice_": {
                str(i): {"in": 0, "out": i, "mb": 0} for i in range(cores)
            },
            "coreIdToDscSchedule": {str(i): [[-1, 0, 0, 0]] for i in range(cores)},
            "dscs_": [
                {
                    op: {
                        "numCoresUsed_": cores,
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": [i for i in range(cores)],
                        "N_": {
                            "name_": "n",
                            **{
                                label + "_": di.size
                                for label, di in dim_info_dict.items()
                            },
                        },
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    **{
                                        label + "_": di.split_size
                                        for label, di in dim_info_dict.items()
                                    },
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        label + "_": di.split_size
                                        for label, di in dim_info_dict.items()
                                    },
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            "INPUT": {
                                "layoutDimOrder_": input_layoutDimOrder,
                                "stickDimOrder_": ["in"],
                                "stickSize_": [inputs[0]["ddtype"].elems_per_stick()],
                            },
                            "OUTPUT": {
                                "layoutDimOrder_": output_layoutDimOrder,
                                "stickDimOrder_": ["out"],
                                "stickSize_": [outputs[0]["ddtype"].elems_per_stick()],
                            },
                            "KERNEL": {
                                "layoutDimOrder_": kernel_layoutDimOrder,
                                "stickDimOrder_": ["out"],
                                "stickSize_": [inputs[1]["ddtype"].elems_per_stick()],
                            },
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": node_name,
                                "prev_": "",
                                "ldsIdx_": idx,
                                "component_": "hbm",
                                "layoutDimOrder_": layout_dim_order,
                                "maxDimSizes_": [-1] * len(layout_dim_order),
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {"factor_": cores, "label_": "core"},
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        # TODO: generalize this to avoid special case handling
                                        f"[{c}, 0, 0]": str(
                                            pointers[tensor["name"]]
                                            + c
                                            * math.prod(
                                                [
                                                    dim_info_dict[label].split_size
                                                    for label in layout_dim_order
                                                ]
                                            )
                                            * num_bytes(tensor["ddtype"])
                                        )
                                        if idx != 0  # duplicated tensor
                                        else str(pointers[tensor["name"]])
                                        for c in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        label: gen_coord_info_value(
                                            size=di.split_size
                                            if (tensor["scale"][di.index] == 1)
                                            else 1,
                                            nsplits=di.nsplits,
                                            elems_per_stick=tensor[
                                                "ddtype"
                                            ].elems_per_stick(),
                                            is_stick_dim=(di.label == stick_label),
                                            is_stick_reduction=False,
                                        )
                                        for label in layout_dim_order
                                        if (di := dim_info_dict[label])
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            }
                            for idx, (
                                node_name,
                                tensor,
                                layout_dim_order,
                                stick_label,
                            ) in enumerate(
                                zip(
                                    [
                                        "allocate_bmm-Input0_hbm",
                                        "allocate_bmm-Input1_hbm",
                                        "allocate_bmm_out_hbm",
                                    ],
                                    inputs + outputs,
                                    [
                                        input_layoutDimOrder,
                                        kernel_layoutDimOrder,
                                        output_layoutDimOrder,
                                    ],
                                    ["in", "out", "out"],
                                )
                            )
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": idx,
                                "dsName_": f"Tensor{idx}",
                                "dsType_": ds_type,
                                # permute scale values according to layoutDimOrder
                                "scale_": [
                                    tensor["scale"][di.index]
                                    for label in layout_dim_order
                                    if (di := dim_info_dict[label])
                                ],
                                "wordLength": num_bytes(tensor["ddtype"]),
                                "dataFormat_": tensor["ddtype"].name,
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                },
                            }
                            for idx, (ds_type, tensor, layout_dim_order) in enumerate(
                                zip(
                                    ["INPUT", "KERNEL", "OUTPUT"],
                                    inputs + outputs,
                                    [
                                        input_layoutDimOrder,
                                        kernel_layoutDimOrder,
                                        output_layoutDimOrder,
                                    ],
                                )
                            )
                        ],
                        "computeOp_": [
                            {
                                "exUnit": "pt",
                                "opFuncName": op,
                                "attributes_": {
                                    "dataFormat_": inputs[0]["ddtype"].name,
                                    "fidelity_": "regular",
                                },
                                "location": "Inner",
                                "inputLabeledDs": [
                                    "Tensor0-idx0",
                                    "Tensor1-idx1",
                                ],
                                "outputLabeledDs": ["Tensor2-idx2"],
                            }
                        ],
                    }
                }
            ],
        }
    }


def generate_bmm(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    # [x=dim0, mb=dim1, in=dim2] @ [x=dim0, in=dim2, out=dim3]

    # implement core division on stick dimension
    cores = 1
    if "op_info" in kwargs and "core_division" in kwargs["op_info"]:
        cores = kwargs["op_info"]["core_division"][-1][0]  # mb_nsplit of the output

    dim_labels = ["x", "mb", "in", "out"]
    dim_indices = [0, 1, 2, 3]
    dim_sizes = dimensions
    dim_splits = [1, cores, 1, 1]
    dim_split_sizes = [size // nsplits for size, nsplits in zip(dim_sizes, dim_splits)]

    # map label to dim_info
    dim_info_dict = {
        label: DimInfo(
            label=label,
            index=index,
            nsplits=nsplits,
            size=size,
            split_size=split_size,
        )
        for label, index, nsplits, size, split_size in zip(
            dim_labels, dim_indices, dim_splits, dim_sizes, dim_split_sizes
        )
    }

    input_layoutDimOrder = ["x", "in", "mb"]
    kernel_layoutDimOrder = ["x", "out", "in"]
    output_layoutDimOrder = ["x", "out", "mb"]

    return {
        op: {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": cores,
            "coreIdToDsc_": {str(i): 0 for i in range(cores)},
            "numWkSlicesPerDim_": {"in": 1, "out": 1, "mb": cores, "x": 1},
            "coreIdToWkSlice_": {
                str(i): {"x": 0, "mb": i, "in": 0, "out": 0} for i in range(cores)
            },
            "coreIdToDscSchedule": {str(i): [[-1, 0, 0, 0]] for i in range(cores)},
            "dscs_": [
                {
                    op: {
                        "numCoresUsed_": cores,
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": list(range(cores)),
                        "N_": {
                            "name_": "n",
                            **{
                                label + "_": di.size
                                for label, di in dim_info_dict.items()
                            },
                        },
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    **{
                                        label + "_": di.split_size
                                        for label, di in dim_info_dict.items()
                                    },
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        label + "_": di.split_size
                                        for label, di in dim_info_dict.items()
                                    },
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            "INPUT": {
                                "layoutDimOrder_": input_layoutDimOrder,
                                "stickDimOrder_": ["in"],
                                "stickSize_": [inputs[0]["ddtype"].elems_per_stick()],
                            },
                            "OUTPUT": {
                                "layoutDimOrder_": output_layoutDimOrder,
                                "stickDimOrder_": ["out"],
                                "stickSize_": [outputs[0]["ddtype"].elems_per_stick()],
                            },
                            "KERNEL": {
                                "layoutDimOrder_": kernel_layoutDimOrder,
                                "stickDimOrder_": ["out"],
                                "stickSize_": [inputs[1]["ddtype"].elems_per_stick()],
                            },
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": node_name,
                                "prev_": "",
                                "ldsIdx_": idx,
                                "component_": "hbm",
                                "layoutDimOrder_": layout_dim_order,
                                "maxDimSizes_": [-1] * len(layout_dim_order),
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {"factor_": cores, "label_": "core"},
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        # TODO: generalize this to avoid special case handling
                                        f"[{c}, 0, 0]": str(
                                            pointers[tensor["name"]]
                                            + c
                                            * math.prod(
                                                [
                                                    dim_info_dict[label].split_size
                                                    for label in layout_dim_order
                                                ]
                                            )
                                            * num_bytes(tensor["ddtype"])
                                        )
                                        if idx != 1  # duplicated tensor
                                        else str(pointers[tensor["name"]])
                                        for c in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        label: gen_coord_info_value(
                                            size=di.split_size
                                            if (tensor["scale"][di.index] == 1)
                                            else 1,
                                            nsplits=di.nsplits,
                                            elems_per_stick=tensor[
                                                "ddtype"
                                            ].elems_per_stick(),
                                            is_stick_dim=(di.label == stick_label),
                                            is_stick_reduction=False,
                                        )
                                        for label in layout_dim_order
                                        if (di := dim_info_dict[label])
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            }
                            for idx, (
                                node_name,
                                tensor,
                                layout_dim_order,
                                stick_label,
                            ) in enumerate(
                                zip(
                                    [
                                        "allocate_bmm-Input0_hbm",
                                        "allocate_bmm-Input1_hbm",
                                        "allocate_bmm_out_hbm",
                                    ],
                                    inputs + outputs,
                                    [
                                        input_layoutDimOrder,
                                        kernel_layoutDimOrder,
                                        output_layoutDimOrder,
                                    ],
                                    ["in", "out", "out"],
                                )
                            )
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": idx,
                                "dsName_": f"Tensor{idx}",
                                "dsType_": ds_type,
                                # permute scale values according to layoutDimOrder
                                "scale_": [
                                    tensor["scale"][di.index]
                                    for label in layout_dim_order
                                    if (di := dim_info_dict[label])
                                ],
                                "wordLength": num_bytes(tensor["ddtype"]),
                                "dataFormat_": tensor["ddtype"].name,
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                },
                            }
                            for idx, (ds_type, tensor, layout_dim_order) in enumerate(
                                zip(
                                    ["INPUT", "KERNEL", "OUTPUT"],
                                    inputs + outputs,
                                    [
                                        input_layoutDimOrder,
                                        kernel_layoutDimOrder,
                                        output_layoutDimOrder,
                                    ],
                                )
                            )
                        ],
                        "computeOp_": [
                            {
                                "exUnit": "pt",
                                "opFuncName": op,
                                "attributes_": {
                                    "dataFormat_": inputs[0]["ddtype"].name,
                                    "fidelity_": "regular",
                                },
                                "location": "Inner",
                                "inputLabeledDs": ["Tensor0-idx0", "Tensor1-idx1"],
                                "outputLabeledDs": ["Tensor2-idx2"],
                            }
                        ],
                    }
                }
            ],
        }
    }
