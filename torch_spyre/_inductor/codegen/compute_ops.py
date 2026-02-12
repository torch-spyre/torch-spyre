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
from torch_spyre._inductor.constants import (
    LAYOUT_LABELS,
    INPUT_DIM_LABELS,
    OUTPUT_DIM_LABELS,
)


@dataclass
class DimInfo:
    def __init__(
        self,
        label: str,
        index: int,
        unpadded_size: int,
        padded_size: int,
        nsplits: int,
        scale: int,
    ):
        self.label = label
        self.index = index
        self.unpadded_size = unpadded_size
        self.padded_size = padded_size
        self.nsplits = nsplits
        self.scale = scale
        self.split_size = self.padded_size // nsplits
        self.padding = self.padded_size - self.unpadded_size

        assert self.padding >= 0


def reorder_dims(cur_list, dim_map):
    return [cur_list[i] for i in dim_map]


@dataclass
class DimInfos:
    """
    Class to help iterate over dimension information in various formats
    Input lists are in host order, but are immediately reordered according to the dim_indices position map
    """

    def __init__(
        self,
        labels: list[str],
        dim_indices: list[int],
        unpadded_sizes: list[int],
        padded_sizes: list[int],
        nsplits: list[int],
        scales: list[int] = [],
    ):
        self.dim_infos_list = []
        self.dim_infos_dict = {}

        self.labels = labels
        self.dim_indices = dim_indices
        self.unpadded_sizes = unpadded_sizes
        self.padded_sizes = padded_sizes
        self.nsplits = nsplits

        # SDSC needs non-negative scale values to be 1
        self.scales = [1 if s >= 0 else s for s in scales]

        self.do_reordering()

        for i in range(len(labels)):
            dim_info = DimInfo(
                self.labels[i],
                self.dim_indices[i],
                self.unpadded_sizes[i],
                self.padded_sizes[i],
                self.nsplits[i],
                self.scales[i] if self.scales else -1,
            )
            self.dim_infos_list.append(dim_info)
            self.dim_infos_dict[labels[i]] = dim_info

    def as_list(self) -> list:
        return self.dim_infos_list

    def as_dict(self) -> dict[str, DimInfo]:
        return self.dim_infos_dict

    # Reorder lists to align with dim_indices position map.
    def do_reordering(self):
        self.labels = reorder_dims(self.labels, self.dim_indices)
        self.unpadded_sizes = reorder_dims(self.unpadded_sizes, self.dim_indices)
        self.padded_sizes = reorder_dims(self.padded_sizes, self.dim_indices)
        self.nsplits = reorder_dims(self.nsplits, self.dim_indices)
        if self.scales:
            self.scales = reorder_dims(self.scales, self.dim_indices)


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


def create_padding_mask_info(dim_infos: DimInfos, kwargs) -> tuple[dict, int]:
    coordinateMasking = {}
    maskingConstId = -1

    for di in dim_infos.as_list():
        if di.padding > 0:
            coordinateMasking[di.label] = [[di.unpadded_size, di.padding]]
    if coordinateMasking:
        maskingConstId = add_constant(kwargs, "samv-maskvalue", 0)

    return coordinateMasking, maskingConstId


def generate_sfp_op(pointers, *, op, dimensions, inputs, outputs, reduction, **kwargs):
    tensors = inputs + outputs

    data_format = inputs[0]["device_layout"].device_dtype

    d3 = len(dimensions) >= 3

    ndim = len(dimensions)

    # implement core division on stick dimension
    cores = 1

    if "op_info" in kwargs and "core_division" in kwargs["op_info"]:
        # enable work division for non-reduction only for now
        if not reduction:
            split_idx = len(dimensions) * -1 if d3 else 0  # split along stick dim
            cores = kwargs["op_info"]["core_division"][-1][split_idx]
            # FIXME: cores should be the product of list of splits

    # TODO: fix constant generation with multiple cores
    if "op_info" in kwargs and "constants" in kwargs["op_info"]:
        cores = 1

    if reduction and tensors[-1]["scale"][-1] >= 0:
        op += "nonstick"

    # Get operation dim map from input or output tensor
    op_dims_tensor = inputs[0] if reduction else outputs[0]
    dim_indices = op_dims_tensor["device_layout"].dim_map[::-1][1:]

    dim_labels = INPUT_DIM_LABELS[: ndim - 1] + OUTPUT_DIM_LABELS[:1]
    dim_splits = [1] * (ndim - 1) + [cores]

    core_id_to_wk_slice = {}
    for i in range(cores):
        core_id_to_wk_slice[str(i)] = {
            str(s): i if s == "out" else 0 for s in dim_labels
        }

    # Obtain (padded) dimensions of the op from a spyre tensor layout
    padded_op_dimensions = [1] * len(dimensions)
    dl = op_dims_tensor["device_layout"]

    # Un-tile and put in host order
    dim_map = dl.dim_map[::-1][1:]
    sizes = dl.device_size[::-1][1:]

    for dim in range(ndim):
        si = op_dims_tensor["scale"][dim]
        assert si >= 0, "Scale value should be non-negative for op_dims_tensor"
        size = sizes[dim_map.index(si)]
        padded_op_dimensions[dim] = (
            size * dl.elems_per_stick() if (dim == dl.host_stick_dim()) else size
        )

    op_dim_infos = DimInfos(
        dim_labels,
        dim_indices,
        dimensions,
        padded_op_dimensions,
        dim_splits,
    )

    coordinateMasking, maskingConstId = create_padding_mask_info(op_dim_infos, kwargs)
    layouts = {}
    # Compute tensor-specific dimension info
    for i, tensor in enumerate(tensors):
        # Adjust for output tensors that have leading dimensions of size 1
        # These dimensions do not exist on the device, and the tiling is different
        # Compute the number of leading missing dims (-1)
        dev_dim_order = tensor["device_layout"].dim_map[::-1][1:]
        missing_dims = list(set(dim_indices) - set(dev_dim_order))
        if len(missing_dims) > 0 and ndim >= 3 and tensor["scale"][0] == -1:
            if missing_dims[0] == 0:
                # Add missing dimensions to end of device dimension order
                # Compute the number of leading missing dims (-1)
                tensor_dim_indices = dev_dim_order + list(
                    set(dim_indices) - set(dev_dim_order)
                )
            else:  # keepdim=0 case
                tensor_dim_indices = [idx + 1 for idx in dev_dim_order] + [0]

                print(tensor_dim_indices)
        else:
            # Indices and order unchanged
            tensor_dim_indices = dim_indices

        # Create dim infos specific to this tensor, reordered if necessary
        tensor["dim_infos"] = DimInfos(
            dim_labels,
            tensor_dim_indices,
            dimensions,
            padded_op_dimensions,
            dim_splits,
            scales=tensor["scale"],
        )

        # primaryDsInfo_ requires each unique layout order to have a name.
        # Reuse the same label for tensors with the same layout, for compactness
        tensor["ds_type"] = None
        for label, dim_order in layouts.items():
            if tensor["dim_infos"].labels == dim_order:
                tensor["ds_type"] = label
                break
        if tensor["ds_type"] is None:
            tensor["ds_type"] = LAYOUT_LABELS[len(layouts.keys())]
            layouts[LAYOUT_LABELS[len(layouts.keys())]] = tensor["dim_infos"].labels

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
            "numWkSlicesPerDim_": {
                di.label: di.nsplits for di in op_dim_infos.as_list()
            },
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
                                di.label + "_": di.padded_size
                                for di in op_dim_infos.as_list()
                            },  # dim sizes before split
                        },
                        "coordinateMasking_": coordinateMasking,
                        "maskingConstId_": maskingConstId,
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size
                                        for di in op_dim_infos.as_list()
                                    },
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size
                                        for di in op_dim_infos.as_list()
                                    },
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            name: {
                                "layoutDimOrder_": dim_order,
                                "stickDimOrder_": ["out"],
                                "stickSize_": [data_format.elems_per_stick()],
                            }
                            for name, dim_order in layouts.items()
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": f"allocate-Tensor{i}_hbm",
                                "prev_": "",
                                "ldsIdx_": i,
                                "component_": "hbm",
                                "layoutDimOrder_": tensor["dim_infos"].labels,
                                "maxDimSizes_": [-1] * len(tensor["dim_infos"].labels),
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
                                            * math.prod(op_dim_infos.padded_sizes[:2])
                                            * num_bytes(
                                                tensor["device_layout"].device_dtype
                                            )
                                            // cores
                                        )
                                        for c in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        di.label: gen_coord_info_value(
                                            size=di.split_size
                                            if (di.scale == 1)
                                            else 1,
                                            nsplits=di.nsplits,
                                            elems_per_stick=tensor[
                                                "device_layout"
                                            ].device_dtype.elems_per_stick(),
                                            is_stick_dim=(di.label == "out"),
                                            is_stick_reduction=(
                                                di.label == "out" and di.scale == -1
                                            ),
                                        )
                                        for di in tensor["dim_infos"].as_list()
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
                                "dsType_": tensor["ds_type"],
                                "scale_": [
                                    (
                                        di.scale
                                        # TODO: revisit whether this special case can be removed
                                        #       pending change in deeptools
                                        if not (di.label == "out" and di.scale == -1)
                                        else -2
                                    )
                                    for di in tensor["dim_infos"].as_list()
                                ],
                                "wordLength": num_bytes(
                                    tensor["device_layout"].device_dtype
                                ),
                                "dataFormat_": tensor[
                                    "device_layout"
                                ].device_dtype.name,
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


# TODO: temp manual padding for matmu / bmm
def pad_up(size, stick_size):
    return ((size + stick_size - 1) // stick_size) * stick_size


def generate_matmul(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    # [mb=dim0, in=dim1] @ [in=dim1, out=dim2]

    # TODO: This is temporary; will move scales to dims_info
    for tensor in inputs + outputs:
        tensor["scale"] = [1 if s >= 0 else s for s in tensor["scale"]]

    # implement core division on stick dimension
    cores = 1
    if "op_info" in kwargs and "core_division" in kwargs["op_info"]:
        cores = kwargs["op_info"]["core_division"][-1][0]

    dim_labels = ["mb", "in", "out"]
    dim_indices = [0, 1, 2]
    dim_splits = [1, 1, cores]

    # TODO: Temp manual padding
    elems_per_stick = inputs[0]["device_layout"].elems_per_stick()
    padded_dimensions = dimensions[:-1] + [pad_up(dimensions[-1], elems_per_stick)]

    op_dim_infos = DimInfos(
        dim_labels,
        dim_indices,
        dimensions,
        padded_dimensions,
        dim_splits,
    )
    dim_info_dict = op_dim_infos.as_dict()

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
                                label + "_": di.padded_size
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
                                "stickSize_": [
                                    inputs[0][
                                        "device_layout"
                                    ].device_dtype.elems_per_stick()
                                ],
                            },
                            "OUTPUT": {
                                "layoutDimOrder_": output_layoutDimOrder,
                                "stickDimOrder_": ["out"],
                                "stickSize_": [
                                    outputs[0][
                                        "device_layout"
                                    ].device_dtype.elems_per_stick()
                                ],
                            },
                            "KERNEL": {
                                "layoutDimOrder_": kernel_layoutDimOrder,
                                "stickDimOrder_": ["out"],
                                "stickSize_": [
                                    inputs[1][
                                        "device_layout"
                                    ].device_dtype.elems_per_stick()
                                ],
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
                                            * num_bytes(
                                                tensor["device_layout"].device_dtype
                                            )
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
                                                "device_layout"
                                            ].device_dtype.elems_per_stick(),
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
                                "wordLength": num_bytes(
                                    tensor["device_layout"].device_dtype
                                ),
                                "dataFormat_": tensor[
                                    "device_layout"
                                ].device_dtype.name,
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
                                    "dataFormat_": inputs[0][
                                        "device_layout"
                                    ].device_dtype.name,
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

    # TODO: This is temporary; will move scales to dims_info
    for tensor in inputs + outputs:
        tensor["scale"] = [1 if s >= 0 else s for s in tensor["scale"]]

    data_format = inputs[0]["device_layout"].device_dtype
    elems_per_stick = data_format.elems_per_stick()

    # implement core division on stick dimension
    cores = 1
    if "op_info" in kwargs and "core_division" in kwargs["op_info"]:
        cores = kwargs["op_info"]["core_division"][-1][0]  # mb_nsplit of the output

    dim_labels = ["x", "mb", "in", "out"]
    dim_indices = [0, 1, 2, 3]
    dim_splits = [1, cores, 1, 1]

    # TODO: Temp manual padding
    elems_per_stick = inputs[0]["device_layout"].elems_per_stick()
    padded_dimensions = dimensions[:-1] + [pad_up(dimensions[-1], elems_per_stick)]

    op_dim_infos = DimInfos(
        dim_labels,
        dim_indices,
        dimensions,
        padded_dimensions,
        dim_splits,
    )
    dim_info_dict = op_dim_infos.as_dict()

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
                                label + "_": di.padded_size
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
                                "stickSize_": [
                                    inputs[0][
                                        "device_layout"
                                    ].device_dtype.elems_per_stick()
                                ],
                            },
                            "OUTPUT": {
                                "layoutDimOrder_": output_layoutDimOrder,
                                "stickDimOrder_": ["out"],
                                "stickSize_": [
                                    outputs[0][
                                        "device_layout"
                                    ].device_dtype.elems_per_stick()
                                ],
                            },
                            "KERNEL": {
                                "layoutDimOrder_": kernel_layoutDimOrder,
                                "stickDimOrder_": ["out"],
                                "stickSize_": [
                                    inputs[1][
                                        "device_layout"
                                    ].device_dtype.elems_per_stick()
                                ],
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
                                            * num_bytes(
                                                tensor["device_layout"].device_dtype
                                            )
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
                                                "device_layout"
                                            ].device_dtype.elems_per_stick(),
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
                                "wordLength": num_bytes(
                                    tensor["device_layout"].device_dtype
                                ),
                                "dataFormat_": tensor[
                                    "device_layout"
                                ].device_dtype.name,
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
                                    "dataFormat_": inputs[0][
                                        "device_layout"
                                    ].device_dtype.name,
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
