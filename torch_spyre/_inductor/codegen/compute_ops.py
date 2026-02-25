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
    LAYOUT_INPUT_LABELS,
    LAYOUT_OUTPUT_LABELS,
    INPUT_DIM_LABELS,
    OUTPUT_DIM_LABELS,
)


class DimInfo:
    def __init__(self, index: int, fields: dict = {}):
        self.index = index
        for name, value in fields.items():
            self.add_data(name, value)

    def add_data(self, name, value):
        setattr(self, name, value)


def get_scales_sdsc_format(tensor, op):
    # SDSC needs non-negative scale values to be 1
    if op == "layernormscale" and tensor["name"] == "arg0":
        return [1] * (len(tensor["scale"]) - 1) + [-1]
    elif op == "layernormnorm" and tensor["name"] == "arg1":
        return [1] * (len(tensor["scale"]) - 1) + [-1]
    else:
        return [1 if s >= 0 else s for s in tensor["scale"]]


@dataclass
class DimInfos:
    """
    Class to provide various views of dimension information during sdsc generation.

    All lists for the rows are provided in and stored in host order.
    Access functions reorder when queried.

    There are 3 main categories of orderings requested during sdsc generation
      1. Op dimensions, ordered by op dim_indices (rank == op dimensions)
      2. Op dimensions, reorderd by a specific tensor.  (rank == op dimensions)
         This one is non-obvious, because a tensor may have fewer dimensions than the op,
         but all op dimension are returned.  See detailed comment on get_tensor_op_index_order()
      3. Tensor dimensions (rank <= operation dimensions)
    """

    def __init__(
        self,
        dim_indices: list[int],
        labels: list[str],
        unpadded_sizes: list[int],
        padded_sizes: list[int],
        nsplits: list[int],
    ):
        # Non-consecutive dim_indices can occur because dims of size 1 are deleted on device.
        # Current code expects dim_indices to be consecutive, so reindex them.
        # If this creates problems in the future, update Diminfos to support non-consecutive indices
        reindex_map = {v: k for k, v in enumerate(sorted(dim_indices))}
        self.dim_indices = [reindex_map[x] for x in dim_indices]
        self.ndim = len(dim_indices)
        self.rows: dict[str, list] = {}
        self.add_row("label", labels)
        self.add_row("unpadded_size", unpadded_sizes)
        self.add_row("padded_size", padded_sizes)
        self.add_row("nsplits", nsplits)
        self.add_row(
            "split_size",
            [size // nsplits for size, nsplits in zip(padded_sizes, nsplits)],
        )
        self.add_row(
            "padding",
            [
                padded_size - unpadded_size
                for padded_size, unpadded_size in zip(padded_sizes, unpadded_sizes)
            ],
        )
        assert all(p >= 0 for p in self.rows["padding"]), (
            "Negative padding found, check padded and unpadded sizes"
        )

    # Internal implementation functions.
    def add_row(self, name, info_list):
        self.rows[name] = info_list

    def make_dim_infos(self, fields=[], index_order=None, additional_rows={}):
        rows = self.rows | additional_rows
        if not index_order:
            index_order = self.dim_indices
        if not fields:
            fields = rows.keys()
        return [self.make_dim_info(i, rows, fields) for i in index_order]

    def make_dim_info(self, i, rows, fields):
        di_dict = {field: rows[field][i] for field in fields}
        return DimInfo(i, di_dict)

    def ordered_row(self, field_name, index_order=None):
        if not index_order:
            index_order = self.dim_indices
        return [self.rows[field_name][i] for i in index_order]

    # Dimensions of length 1 do not exist on the device layout
    # Therefore some sdsc sections require all op dimensions, but
    # with the order of a subset of the indices adjusted to match
    # the tile layout of the tensor on the device.  This
    # function computes that reordering
    def get_tensor_op_index_order(self, tensor):
        dim_indices = self.dim_indices  # op dimension order
        dev_dim_order = tensor["device_layout"].dim_map[::-1][1:]
        missing_dims = list(set(dim_indices) - set(dev_dim_order))
        if len(missing_dims) > 0 and len(dim_indices) >= 3 and tensor["scale"][0] == -1:
            if missing_dims[0] == 0:
                # Add missing dimensions to end of device dimension order
                # Compute the number of leading missing dims (-1)
                tensor_dim_indices = dev_dim_order + list(
                    set(dim_indices) - set(dev_dim_order)
                )
            else:  # keepdim=0 case
                tensor_dim_indices = [idx + 1 for idx in dev_dim_order] + [0]
        else:
            # Indices and order unchanged
            tensor_dim_indices = dim_indices
        return tensor_dim_indices

    def get_labels_host_order(self):
        return self.rows["label"]

    # Accessor functions used by SDSC generation
    def get_op_infos(self) -> list:
        return self.make_dim_infos()

    def get_op_layout_order(self, index_order=None):
        return self.ordered_row("label", index_order)

    def get_padded_sizes(self):
        return self.ordered_row("padded_size")

    # Get infos for the operation dimensions, with order influenced
    # by tensor layout. Rank of returned list == op dimensions
    # See get_tensor_op_index_order
    def get_tensor_op_layout_order(self, tensor, op):
        return [di.label for di in self.get_tensor_op_infos(tensor, op)]

    def get_tensor_op_infos(self, tensor, op):
        result = self.make_dim_infos(
            additional_rows={"scale": get_scales_sdsc_format(tensor, op)},
            index_order=self.get_tensor_op_index_order(tensor),
        )
        return result

    # Get labels corresponding to tensor layout
    # Rank of returned list == num tensor dimensions
    def get_tensor_layout_order(self, tensor):
        # TODO: Understsand why matmul needed this layout order to be different
        # than the order obtained by get_tensor_infos().  Is it possible
        # get_tensor_infos() should be using this same order as well?
        dl = tensor["device_layout"]
        scale = tensor["scale"]
        dev_dim_order = dl.dim_map[::-1][1:]
        return [self.rows["label"][scale.index(dmv)] for dmv in dev_dim_order]

    def get_tensor_infos(self, tensor, op):
        tensor_op_infos = self.get_tensor_op_infos(tensor, op)
        return [di for di in tensor_op_infos if di.scale >= 0]

    def get_tensor_stick_dim_labels(self, tensor):
        dl = tensor["device_layout"]
        idx = tensor["scale"].index(dl.host_stick_dim())
        return [self.rows["label"][idx]]


# Extract the device size for a give host dim
# Assumption is that the passed tensor operate in host dimension space
def get_device_size(host_dim, tensor):
    dl = tensor["device_layout"]
    device_dim = tensor["scale"][host_dim]
    if device_dim == -3:  # special case to skip elided dim
        return 1
    assert device_dim >= 0, "Scale value should be non-negative for tensor provided"
    size = dl.device_size[dl.dim_map.index(device_dim)]
    if device_dim == dl.host_stick_dim():
        size *= dl.elems_per_stick()
    return size


def calculate_core_to_slice_mapping(
    dim_labels: list[str], dim_splits: list[int]
) -> dict[str, dict[str, int]]:
    """
    Calculate mapping from core ID to slice indices for each dimension.

    Iterates dimensions right-to-left (innermost varies fastest), similar to
    row-major ordering in multi-dimensional arrays.

    Args:
        dim_labels: List of dimension labels (e.g., ["mb", "out", "x"])
        dim_splits: Number of splits per dimension (e.g., [2, 4, 1])

    Returns:
        Dictionary mapping core ID (as string) to dimension slice indices
    """
    total_cores = 1
    for splits in dim_splits:
        total_cores *= splits

    core_to_slice = {}

    for core_id in range(total_cores):
        # Calculate multi-dimensional index from flat core_id
        # Iterate right-to-left (innermost dimension varies fastest)
        indices = {}
        remaining = core_id

        for i in range(len(dim_labels) - 1, -1, -1):
            indices[dim_labels[i]] = remaining % dim_splits[i]
            remaining //= dim_splits[i]

        core_to_slice[str(core_id)] = indices

    return core_to_slice


def core_idx_to_slice_offset(
    dim_info_list: list[DimInfo],
    wk_slice: dict[str, int],
    device_size: list[int],
) -> int:
    # compute tensor specific strides from its device layout
    strides = {}
    for i, di in enumerate(dim_info_list):
        strides[di.label] = math.prod(device_size[-i - 2 :])

    # Calculate offset by accumulating contribution from each dimension
    offset = 0
    for di in dim_info_list:
        label = di.label
        slice_idx = wk_slice[label]
        offset += slice_idx * strides[label] // di.nsplits

    return offset


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


def create_padding_mask_info(dim_infos: DimInfos, kwargs) -> tuple[dict, int]:
    coordinateMasking = {}
    maskingConstId = -1

    for di in dim_infos.get_op_infos():
        if di.padding > 0:
            coordinateMasking[di.label] = [[di.unpadded_size, di.padding]]
    if coordinateMasking:
        maskingConstId = add_constant(kwargs, "samv-maskvalue", 0)

    return coordinateMasking, maskingConstId


def create_tensor_specific_layouts(tensors, dim_infos, op, is_matmul=False):
    layouts = {}
    # Compute tensor-specific dimension info
    for i, tensor in enumerate(tensors):
        # primaryDsInfo_ requires each unique layout order to have a name.
        # Reuse the same label for tensors with the same layout, for compactness
        tensor["ds_type"] = None

        # For sfp, the sdsc layout expected is determined by the operation layout
        # For matmul, the layout order is determined by the tensor's layout
        layout_order = (
            dim_infos.get_tensor_layout_order(tensor)
            if is_matmul
            else dim_infos.get_tensor_op_layout_order(tensor, op)
        )
        for label, layout_infos in layouts.items():
            if layout_order == layout_infos["layout_order"]:
                tensor["ds_type"] = label
                break
        if tensor["ds_type"] is None:
            tensor["ds_type"] = (
                LAYOUT_INPUT_LABELS[len(layouts.keys())]
                if len(layouts.keys()) < len(LAYOUT_INPUT_LABELS)
                else LAYOUT_OUTPUT_LABELS[
                    len(layouts.keys()) - len(LAYOUT_INPUT_LABELS)
                ]
            )
            layouts[LAYOUT_INPUT_LABELS[len(layouts.keys())]] = {
                "layout_order": layout_order,
                "stick_dim_order": dim_infos.get_tensor_stick_dim_labels(tensor),
            }

    # Now adjust the label of the final tensor (and all that share the same layout) to be "OUTPUT".
    # This is not strictly required, but conformes the standard naming pattern
    final_label = tensors[-1]["ds_type"]
    output_label = LAYOUT_OUTPUT_LABELS[0]
    for tensor in tensors:
        if tensor["ds_type"] == final_label:
            tensor["ds_type"] = output_label
    layouts[output_label] = layouts.pop(final_label)

    return layouts


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

    # Get operation dim map from the tensor that represents the operation space
    op_dims_tensor = inputs[0] if reduction else outputs[0]
    dl = op_dims_tensor["device_layout"]
    dim_map = dl.dim_map[::-1][1:]
    dim_labels = INPUT_DIM_LABELS[: ndim - 1] + OUTPUT_DIM_LABELS[:1]
    dim_splits = [1] * (ndim - 1) + [cores]

    # Obtain (padded) dimensions of the op from a spyre tensor layout
    padded_op_dimensions = [
        get_device_size(host_dim, op_dims_tensor) for host_dim in range(ndim)
    ]

    dim_infos = DimInfos(
        dim_map,
        dim_labels,
        dimensions,
        padded_op_dimensions,
        dim_splits,
    )

    coordinateMasking, maskingConstId = create_padding_mask_info(dim_infos, kwargs)
    layouts = create_tensor_specific_layouts(tensors, dim_infos, op)

    # Compute the stick label from the op tensor.
    # For now we expect stick dim to always be "out", so check.
    # Remove the assertion when this invariant changes
    op_stick_labels = dim_infos.get_tensor_stick_dim_labels(op_dims_tensor)
    assert op_stick_labels == ["out"]

    core_id_to_wk_slice = {}
    for i in range(cores):
        core_id_to_wk_slice[str(i)] = {
            str(s): i if s in op_stick_labels else 0 for s in dim_labels
        }

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
                di.label: di.nsplits for di in dim_infos.get_op_infos()
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
                                for di in dim_infos.get_op_infos()
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
                                        for di in dim_infos.get_op_infos()
                                    },
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size
                                        for di in dim_infos.get_op_infos()
                                    },
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            name: {
                                "layoutDimOrder_": layout_info["layout_order"],
                                "stickDimOrder_": layout_info["stick_dim_order"],
                                "stickSize_": [data_format.elems_per_stick()],
                            }
                            for name, layout_info in layouts.items()
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": f"allocate-Tensor{i}_{'hbm' if tensor['lx_addr'] is None else 'lx'}",
                                "prev_": "",
                                "ldsIdx_": i,
                                "component_": "hbm"
                                if tensor["lx_addr"] is None
                                else "lx",
                                "layoutDimOrder_": dim_infos.get_tensor_op_layout_order(
                                    tensor, op
                                ),
                                "maxDimSizes_": [-1]
                                * len(dim_infos.get_tensor_op_layout_order(tensor, op)),
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
                                            * math.prod(
                                                dim_infos.get_padded_sizes()[:2]
                                            )
                                            * num_bytes(
                                                tensor["device_layout"].device_dtype
                                            )
                                            // cores
                                        )
                                        if tensor["lx_addr"] is None
                                        else tensor["lx_addr"]
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
                                            is_stick_dim=(di.label in op_stick_labels),
                                            is_stick_reduction=(
                                                di.label in op_stick_labels
                                                and di.scale == -1
                                            ),
                                        )
                                        for di in dim_infos.get_tensor_op_infos(
                                            tensor, op
                                        )
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
                                        if not (
                                            di.label in op_stick_labels
                                            and di.scale == -1
                                        )
                                        else -2
                                    )
                                    for di in dim_infos.get_tensor_op_infos(tensor, op)
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
                                }
                                if tensor["lx_addr"] is None
                                else {"lx": {"isPresent": 1}},
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


# Extract the padded sizes from the tensors for mm and bmm.
# The pattern of how op dimensions map to tensors seems to be:
#  - The first N-2 dims come from tensor 0
#  - Last 2 dims come from tensor 1
def get_padded_dimensions_matmul(ndim, inputs):
    padded_dimensions = [0] * ndim
    for host_dim in range(ndim):
        tensor_idx = 0 if host_dim < ndim - 2 else 1
        padded_dimensions[host_dim] = get_device_size(host_dim, inputs[tensor_idx])
    return padded_dimensions


def _generate_matmul_common(
    pointers,
    *,
    op,
    dimensions,
    inputs,
    outputs,
    dim_labels,
    dim_indices,
    dim_splits,
    coreid_to_wk_slice,
    cores,
):
    """
    Common implementation for matmul and bmm operations.

    This function contains the shared logic between generate_matmul and generate_bmm,
    which differ primarily in their dimension configurations.

    Args:
        pointers: Memory pointers for tensors
        op: Operation name
        dimensions: Tensor host dimensions
        inputs: Input tensor specifications
        outputs: Output tensor specifications
        dim_labels: Dimension labels (e.g., ["mb", "in", "out"] for matmul)
        dim_indices: Dimension indices
        dim_splits: Number of splits per dimension
        coreid_to_wk_slice: Mapping from core ID to work slice
        cores: Number of cores used

    Returns:
        Dictionary containing the SDSC structure for the operation
    """
    tensors = inputs + outputs
    data_format = inputs[0]["device_layout"].device_dtype

    padded_dimensions = get_padded_dimensions_matmul(len(dim_indices), inputs)

    dim_infos = DimInfos(
        dim_indices,
        dim_labels,
        dimensions,
        padded_dimensions,
        dim_splits,
    )
    dim_info_dict = {di.label: di for di in dim_infos.get_op_infos()}

    layouts = create_tensor_specific_layouts(tensors, dim_infos, op, is_matmul=True)

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
            "numWkSlicesPerDim_": {k: v for k, v in zip(dim_labels, dim_splits)},
            "coreIdToWkSlice_": coreid_to_wk_slice,
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
                            name: {
                                "layoutDimOrder_": layout_info["layout_order"],
                                "stickDimOrder_": layout_info["stick_dim_order"],
                                "stickSize_": [data_format.elems_per_stick()],
                            }
                            for name, layout_info in layouts.items()
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                # "name_": node_name,
                                "name_": f"allocate_Input{idx}_hbm"
                                if idx < len(tensors) - 1
                                else "allocate_out_hbm",
                                "prev_": "",
                                "ldsIdx_": idx,
                                "component_": "hbm",
                                "layoutDimOrder_": dim_infos.get_tensor_layout_order(
                                    tensor
                                ),
                                "maxDimSizes_": [-1]
                                * len(dim_infos.get_tensor_layout_order(tensor)),
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
                                            + core_idx_to_slice_offset(
                                                [
                                                    dim_info_dict[label]
                                                    for label in dim_infos.get_tensor_layout_order(
                                                        tensor
                                                    )
                                                ],
                                                coreid_to_wk_slice[str(c)],
                                                tensor["device_layout"].device_size,
                                            )
                                            * num_bytes(
                                                tensor["device_layout"].device_dtype
                                            )
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
                                            is_stick_dim=(
                                                di.label
                                                in dim_infos.get_tensor_stick_dim_labels(
                                                    tensor
                                                )
                                            ),
                                        )
                                        for di in dim_infos.get_tensor_infos(tensor, op)
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            }
                            for idx, tensor in enumerate(tensors)
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": idx,
                                "dsName_": f"Tensor{idx}",
                                "dsType_": tensor["ds_type"],
                                "scale_": [
                                    di.scale
                                    for di in dim_infos.get_tensor_infos(tensor, op)
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
                            for idx, tensor in enumerate(tensors)
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


def generate_matmul(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    """
    Generate SDSC structure for matrix multiplication operation.

    Matmul operation: [mb=dim0, in=dim1] @ [in=dim1, out=dim2]

    This is a thin wrapper around _generate_matmul_common that provides
    matmul-specific configuration (3D dimensions, specific layouts).
    """
    dim_labels = ["mb", "in", "out"]
    dim_indices = [0, 1, 2]

    # work division logic
    cores = 1
    dim_splits = [1, 1, 1]
    if "op_info" in kwargs:
        if "n_cores_used" in kwargs["op_info"]:
            cores = kwargs["op_info"]["n_cores_used"]

        if "core_division" in kwargs["op_info"]:
            core_div = kwargs["op_info"]["core_division"][-1]  # output core division
            for dev_dim_idx, nsplit in enumerate(
                core_div[:-1]
            ):  # exclude the last device dim
                if nsplit > 1:
                    # dev_dim_idx -> host_dim_idx
                    host_dim_idx = outputs[0]["device_layout"].dim_map[dev_dim_idx]
                    # host_dim_idx -> op_dim_idx for nsplit assignment
                    dim_splits[outputs[0]["scale"].index(host_dim_idx)] = nsplit

    coreid_to_wk_slice = calculate_core_to_slice_mapping(dim_labels, dim_splits)

    return _generate_matmul_common(
        pointers,
        op=op,
        dimensions=dimensions,
        inputs=inputs,
        outputs=outputs,
        dim_labels=dim_labels,
        dim_indices=dim_indices,
        dim_splits=dim_splits,
        coreid_to_wk_slice=coreid_to_wk_slice,
        cores=cores,
    )


def generate_bmm(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    """
    Generate SDSC structure for batched matrix multiplication operation.

    BMM operation: [x=dim0, mb=dim1, in=dim2] @ [x=dim0, in=dim2, out=dim3]

    This is a thin wrapper around _generate_matmul_common that provides
    bmm-specific configuration (4D dimensions with batch, specific layouts).
    """
    if len(dimensions) == 4:  # 3d bmm
        dim_labels = ["x", "mb", "in", "out"]
    else:  # 4d bmm
        dim_labels = ["x", "y", "mb", "in", "out"]

    dim_indices = list(range(len(dim_labels)))

    cores = 1
    dim_splits = [1] * len(dim_labels)
    if "op_info" in kwargs:
        if "n_cores_used" in kwargs["op_info"]:
            cores = kwargs["op_info"]["n_cores_used"]

        if "core_division" in kwargs["op_info"]:
            core_div = kwargs["op_info"]["core_division"][-1]  # output core division
            for dev_dim_idx, nsplit in enumerate(
                core_div[:-1]
            ):  # exclude the last device dim
                if nsplit > 1:
                    # dev_dim_idx -> host_dim_idx
                    host_dim_idx = outputs[0]["device_layout"].dim_map[dev_dim_idx]
                    # host_dim_idx -> op_dim_idx for nsplit assignment
                    dim_splits[outputs[0]["scale"].index(host_dim_idx)] = nsplit

    coreid_to_wk_slice = calculate_core_to_slice_mapping(dim_labels, dim_splits)

    return _generate_matmul_common(
        pointers,
        op=op,
        dimensions=dimensions,
        inputs=inputs,
        outputs=outputs,
        dim_labels=dim_labels,
        dim_indices=dim_indices,
        dim_splits=dim_splits,
        coreid_to_wk_slice=coreid_to_wk_slice,
        cores=cores,
    )
