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
import os
from torch_spyre._C import encode_constant, get_sen_data_format
from torch_spyre._inductor.constants import BYTES_PER_STICK


def generate_constant_info(data_format, **kwargs):
    if "op_info" not in kwargs or "constants" not in kwargs["op_info"]:
        return "{}"
    constant_info = {}
    for name, value in kwargs["op_info"]["constants"].items():
        ci = {
            "dataFormat_": data_format,
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


def core_split(size):
    size = (size + 63) // 64 * 64
    max_cores = int(os.getenv("SENCORES", "1"))
    for i in range(max_cores, 0, -1):
        if size // 64 % i == 0:
            return i


def generate_sfp_op(pointers, *, op, dimensions, inputs, outputs, reduction, **kwargs):
    tensors = inputs + outputs

    data_format = get_sen_data_format(inputs[0]["dtype"])

    # implement core division for non-broadcasting 1-d pointwise ops with large enough inputs
    if len(dimensions) > 1:
        cores = 1
    else:
        cores = core_split(dimensions[-1])
        for t in tensors:
            for s in t["scale"]:
                if s != 1:
                    cores = 1

    d2 = len(dimensions) >= 2
    d3 = len(dimensions) >= 3

    if reduction and tensors[-1]["scale"][-1] == 1:
        op += "nonstick"
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
            "numWkSlicesPerDim_": {
                "mb": 1 if d2 else 0,
                "x": 1 if d3 else 0,
                "out": cores,
            },
            "coreIdToWkSlice_": {
                str(i): {"mb": 0, "x": 0, "out": i} for i in range(cores)
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
                            "mb_": dimensions[0] if d2 else 0,
                            "x_": dimensions[1] if d3 else 0,
                            "out_": dimensions[-1],
                        },
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    "mb_": dimensions[0] if d2 else 0,
                                    "x_": dimensions[1] if d3 else 0,
                                    "out_": dimensions[-1] // cores,
                                },
                                "el_": {
                                    "name_": "core",
                                    "mb_": dimensions[0] if d2 else 0,
                                    "x_": dimensions[1] if d3 else 0,
                                    "out_": dimensions[-1] // cores,
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            "OUTPUT": {
                                "layoutDimOrder_": (["mb"] if d2 else [])
                                + ["out"]
                                + (["x"] if d3 else []),
                                "stickDimOrder_": ["out"],
                                "stickSize_": [
                                    BYTES_PER_STICK // inputs[0]["dtype"].itemsize
                                ],
                            }
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": f"allocate-Tensor{i}_hbm",
                                "prev_": "",
                                "ldsIdx_": i,
                                "component_": "hbm",
                                "layoutDimOrder_": (["mb"] if d2 else [])
                                + ["out"]
                                + (["x"] if d3 else []),
                                "maxDimSizes_": [-1] * len(dimensions),
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
                                        f"[{i}, 0, 0]": str(
                                            pointers[tensor["name"]]
                                            + i * math.prod(dimensions) * 2 // cores
                                        )
                                        for i in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        "out": {
                                            "spatial": 3,
                                            "temporal": 0,
                                            "elemArr": 2,
                                            "padding": "nopad",
                                            "folds": {
                                                "dim_prop_func": [
                                                    {
                                                        "Affine": {
                                                            "alpha_": dimensions[-1]
                                                            // cores,
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
                                                            "alpha_": BYTES_PER_STICK
                                                            // tensor["dtype"].itemsize,
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
                                                        "factor_": cores,
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
                                                        "factor_": dimensions[-1]
                                                        // (
                                                            BYTES_PER_STICK
                                                            // tensor["dtype"].itemsize
                                                        )
                                                        // cores,
                                                        "label_": "elem_arr_1",
                                                    },
                                                    {
                                                        "factor_": BYTES_PER_STICK
                                                        // tensor["dtype"].itemsize,
                                                        "label_": "elem_arr_0",
                                                    },
                                                ],
                                            },
                                        },
                                        "mb": {
                                            "spatial": 3,
                                            "temporal": 0,
                                            "elemArr": 1,
                                            "padding": "nopad",
                                            "folds": {
                                                "dim_prop_func": [
                                                    {
                                                        "Affine": {
                                                            "alpha_": dimensions[0],
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
                                                        "factor_": 1,
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
                                                        "factor_": dimensions[0],
                                                        "label_": "elem_arr_0",
                                                    },
                                                ],
                                            },
                                        },
                                        "x": {
                                            "spatial": 3,
                                            "temporal": 0,
                                            "elemArr": 1,
                                            "padding": "nopad",
                                            "folds": {
                                                "dim_prop_func": [
                                                    {
                                                        "Affine": {
                                                            "alpha_": dimensions[1]
                                                            if d3
                                                            else 1,
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
                                                        "factor_": 1,
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
                                                        "factor_": dimensions[1]
                                                        if d3
                                                        else 1,
                                                        "label_": "elem_arr_0",
                                                    },
                                                ],
                                            },
                                        },
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
                                "scale_": [tensor["scale"][0]]
                                + (
                                    [-2 if tensor["scale"][-1] == -1 else 1]
                                    if d2
                                    else []
                                )
                                + tensor["scale"][1:-1],
                                "wordLength": tensor["dtype"].itemsize,
                                "dataFormat_": get_sen_data_format(tensor["dtype"]),
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
                                    "dataFormat_": data_format,
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

    cores = core_split(dimensions[-1])

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
                            "mb_": dimensions[0],
                            "in_": dimensions[1],
                            "out_": dimensions[2],
                        },
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    "mb_": dimensions[0],
                                    "in_": dimensions[1],
                                    "out_": dimensions[2] // cores,
                                },
                                "el_": {
                                    "name_": "core",
                                    "mb_": dimensions[0],
                                    "in_": dimensions[1],
                                    "out_": dimensions[2] // cores,
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            "INPUT": {
                                "layoutDimOrder_": ["mb", "in"],
                                "stickDimOrder_": ["in"],
                                "stickSize_": [
                                    BYTES_PER_STICK // inputs[0]["dtype"].itemsize
                                ],
                            },
                            "OUTPUT": {
                                "layoutDimOrder_": ["mb", "out"],
                                "stickDimOrder_": ["out"],
                                "stickSize_": [
                                    BYTES_PER_STICK // outputs[0]["dtype"].itemsize
                                ],
                            },
                            "KERNEL": {
                                "layoutDimOrder_": ["in", "out"],
                                "stickDimOrder_": ["out"],
                                "stickSize_": [
                                    BYTES_PER_STICK // inputs[1]["dtype"].itemsize
                                ],
                            },
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": "allocate-Tensor0_hbm",
                                "prev_": "",
                                "ldsIdx_": 0,
                                "component_": "hbm",
                                "layoutDimOrder_": ["mb", "in"],
                                "maxDimSizes_": [-1, -1],
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
                                        f"[{i}, 0, 0]": str(pointers[inputs[0]["name"]])
                                        for i in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        "mb": {
                                            "spatial": 3,
                                            "temporal": 0,
                                            "elemArr": 1,
                                            "padding": "nopad",
                                            "folds": {
                                                "dim_prop_func": [
                                                    {
                                                        "Affine": {
                                                            "alpha_": dimensions[0],
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
                                                        "factor_": 1,
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
                                                        "factor_": dimensions[0],
                                                        "label_": "elem_arr_0",
                                                    },
                                                ],
                                            },
                                        },
                                        "in": {
                                            "spatial": 3,
                                            "temporal": 0,
                                            "elemArr": 2,
                                            "padding": "nopad",
                                            "folds": {
                                                "dim_prop_func": [
                                                    {
                                                        "Affine": {
                                                            "alpha_": dimensions[1],
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
                                                            "alpha_": 64,
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
                                                        "factor_": 1,
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
                                                        "factor_": dimensions[1] // 64,
                                                        "label_": "elem_arr_1",
                                                    },
                                                    {
                                                        "factor_": 64,
                                                        "label_": "elem_arr_0",
                                                    },
                                                ],
                                            },
                                        },
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            },
                            {
                                "nodeType_": "allocate",
                                "name_": "allocate-Tensor1_hbm",
                                "prev_": "",
                                "ldsIdx_": 1,
                                "component_": "hbm",
                                "layoutDimOrder_": ["in", "out"],
                                "maxDimSizes_": [-1, -1],
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
                                        f"[{i}, 0, 0]": str(
                                            pointers[inputs[1]["name"]]
                                            + i
                                            * dimensions[1]
                                            * dimensions[2]
                                            * inputs[1]["dtype"].itemsize
                                            // cores
                                        )
                                        for i in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        "in": {
                                            "spatial": 3,
                                            "temporal": 0,
                                            "elemArr": 1,
                                            "padding": "nopad",
                                            "folds": {
                                                "dim_prop_func": [
                                                    {
                                                        "Affine": {
                                                            "alpha_": dimensions[1],
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
                                                        "factor_": 1,
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
                                                        "factor_": dimensions[1],
                                                        "label_": "elem_arr_0",
                                                    },
                                                ],
                                            },
                                        },
                                        "out": {
                                            "spatial": 3,
                                            "temporal": 0,
                                            "elemArr": 2,
                                            "padding": "nopad",
                                            "folds": {
                                                "dim_prop_func": [
                                                    {
                                                        "Affine": {
                                                            "alpha_": dimensions[2]
                                                            // cores,
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
                                                            "alpha_": 64,
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
                                                        "factor_": cores,
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
                                                        "factor_": dimensions[2]
                                                        // cores
                                                        // 64,
                                                        "label_": "elem_arr_1",
                                                    },
                                                    {
                                                        "factor_": 64,
                                                        "label_": "elem_arr_0",
                                                    },
                                                ],
                                            },
                                        },
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            },
                            {
                                "nodeType_": "allocate",
                                "name_": "allocate-Tensor2_hbm",
                                "prev_": "",
                                "ldsIdx_": 2,
                                "component_": "hbm",
                                "layoutDimOrder_": ["mb", "out"],
                                "maxDimSizes_": [-1, -1],
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
                                        f"[{i}, 0, 0]": str(
                                            pointers[outputs[0]["name"]]
                                            + i
                                            * dimensions[0]
                                            * dimensions[2]
                                            * outputs[0]["dtype"].itemsize
                                            // cores
                                        )
                                        for i in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        "mb": {
                                            "spatial": 3,
                                            "temporal": 0,
                                            "elemArr": 1,
                                            "padding": "nopad",
                                            "folds": {
                                                "dim_prop_func": [
                                                    {
                                                        "Affine": {
                                                            "alpha_": dimensions[0],
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
                                                        "factor_": 1,
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
                                                        "factor_": dimensions[0],
                                                        "label_": "elem_arr_0",
                                                    },
                                                ],
                                            },
                                        },
                                        "out": {
                                            "spatial": 3,
                                            "temporal": 0,
                                            "elemArr": 2,
                                            "padding": "nopad",
                                            "folds": {
                                                "dim_prop_func": [
                                                    {
                                                        "Affine": {
                                                            "alpha_": dimensions[2]
                                                            // cores,
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
                                                            "alpha_": 64,
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
                                                        "factor_": cores,
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
                                                        "factor_": dimensions[2]
                                                        // cores
                                                        // 64,
                                                        "label_": "elem_arr_1",
                                                    },
                                                    {
                                                        "factor_": 64,
                                                        "label_": "elem_arr_0",
                                                    },
                                                ],
                                            },
                                        },
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            },
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": 0,
                                "dsName_": "Tensor0",
                                "dsType_": "INPUT",
                                "scale_": inputs[0]["scale"][0:2],
                                "wordLength": inputs[0]["dtype"].itemsize,
                                "dataFormat_": get_sen_data_format(inputs[0]["dtype"]),
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                },
                            },
                            {
                                "ldsIdx_": 1,
                                "dsName_": "Tensor1",
                                "dsType_": "KERNEL",
                                "scale_": inputs[1]["scale"][1:3],
                                "wordLength": inputs[1]["dtype"].itemsize,
                                "dataFormat_": get_sen_data_format(inputs[1]["dtype"]),
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                },
                            },
                            {
                                "ldsIdx_": 2,
                                "dsName_": "Tensor2",
                                "dsType_": "OUTPUT",
                                "scale_": [
                                    outputs[0]["scale"][0],
                                    outputs[0]["scale"][2],
                                ],
                                "wordLength": outputs[0]["dtype"].itemsize,
                                "dataFormat_": get_sen_data_format(outputs[0]["dtype"]),
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                },
                            },
                        ],
                        "computeOp_": [
                            {
                                "exUnit": "pt",
                                "opFuncName": op,
                                "attributes_": {
                                    "dataFormat_": get_sen_data_format(
                                        inputs[0]["dtype"]
                                    ),
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
    return {
        op: {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": 1, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": 1,
            "coreIdToDsc_": {"0": 0},
            "numWkSlicesPerDim_": {"in": 1, "out": 1, "mb": 1, "x": 1},
            "coreIdToWkSlice_": {"0": {"in": 0, "out": 0, "mb": 0, "x": 0}},
            "coreIdToDscSchedule": {"0": [[-1, 0, 0, 0]]},
            "dscs_": [
                {
                    op: {
                        "numCoresUsed_": 1,
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": [0],
                        "N_": {
                            "name_": "n",
                            "in_": dimensions[2],
                            "out_": dimensions[3],
                            "mb_": dimensions[1],
                            "x_": dimensions[0],
                        },
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    "in_": dimensions[2],
                                    "out_": dimensions[3],
                                    "mb_": dimensions[1],
                                    "x_": dimensions[0],
                                },
                                "el_": {
                                    "name_": "core",
                                    "in_": dimensions[2],
                                    "out_": dimensions[3],
                                    "mb_": dimensions[1],
                                    "x_": dimensions[0],
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            "INPUT": {
                                "layoutDimOrder_": ["x", "in", "mb"],
                                "stickDimOrder_": ["in"],
                                "stickSize_": [64],
                            },
                            "OUTPUT": {
                                "layoutDimOrder_": ["x", "out", "mb"],
                                "stickDimOrder_": ["out"],
                                "stickSize_": [64],
                            },
                            "KERNEL": {
                                "layoutDimOrder_": ["x", "out", "in"],
                                "stickDimOrder_": ["out"],
                                "stickSize_": [64],
                            },
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": "allocate_bmm-Input0_hbm",  # input0
                                "prev_": "",
                                "ldsIdx_": 0,
                                "component_": "hbm",
                                "layoutDimOrder_": ["x", "in", "mb"],
                                "maxDimSizes_": [-1, -1, -1],
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {"factor_": 1, "label_": "core"},
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        "[0, 0, 0]": str(pointers[inputs[0]["name"]])
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        name: {
                                            "spatial": 3,
                                            "temporal": 0,
                                            "elemArr": 2,
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
                                                            "alpha_": 64
                                                            if size % 64 == 0
                                                            else 1,
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
                                                        "factor_": 1,
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
                                                        "factor_": size // 64
                                                        if size % 64 == 0
                                                        else size,
                                                        "label_": "elem_arr_1",
                                                    },
                                                    {
                                                        "factor_": 64
                                                        if size % 64 == 0
                                                        else 1,
                                                        "label_": "elem_arr_0",
                                                    },
                                                ],
                                            },
                                        }
                                        for name, size in zip(
                                            ["mb", "in", "x"],
                                            [
                                                dimensions[1],
                                                dimensions[2],
                                                dimensions[0],
                                            ],
                                        )
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            },
                            {
                                "nodeType_": "allocate",
                                "name_": "allocate_bmm-Input1_hbm",  # input1
                                "prev_": "",
                                "ldsIdx_": 1,
                                "component_": "hbm",
                                "layoutDimOrder_": ["x", "out", "in"],
                                "maxDimSizes_": [-1, -1, -1],
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {"factor_": 1, "label_": "core"},
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        "[0, 0, 0]": str(pointers[inputs[1]["name"]])
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        name: {
                                            "spatial": 3,
                                            "temporal": 0,
                                            "elemArr": 2,
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
                                                            "alpha_": 64
                                                            if size % 64 == 0
                                                            else 1,
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
                                                        "factor_": 1,
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
                                                        "factor_": size // 64
                                                        if size % 64 == 0
                                                        else size,
                                                        "label_": "elem_arr_1",
                                                    },
                                                    {
                                                        "factor_": 64
                                                        if size % 64 == 0
                                                        else 1,
                                                        "label_": "elem_arr_0",
                                                    },
                                                ],
                                            },
                                        }
                                        for name, size in zip(
                                            ["in", "out", "x"],
                                            [
                                                dimensions[2],
                                                dimensions[3],
                                                dimensions[0],
                                            ],
                                        )
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            },
                            {
                                "nodeType_": "allocate",
                                "name_": "allocate_bmm_out_hbm",  # output
                                "prev_": "",
                                "ldsIdx_": 2,
                                "component_": "hbm",
                                "layoutDimOrder_": ["x", "out", "mb"],
                                "maxDimSizes_": [-1, -1, -1],
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {"factor_": 1, "label_": "core"},
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        "[0, 0, 0]": str(pointers[outputs[0]["name"]])
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        name: {
                                            "spatial": 3,
                                            "temporal": 0,
                                            "elemArr": 2,
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
                                                            "alpha_": 64
                                                            if size % 64 == 0
                                                            else 1,
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
                                                        "factor_": 1,
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
                                                        "factor_": size // 64
                                                        if size % 64 == 0
                                                        else size,
                                                        "label_": "elem_arr_1",
                                                    },
                                                    {
                                                        "factor_": 64
                                                        if size % 64 == 0
                                                        else 1,
                                                        "label_": "elem_arr_0",
                                                    },
                                                ],
                                            },
                                        }
                                        for name, size in zip(
                                            ["mb", "out", "x"],
                                            [
                                                dimensions[1],
                                                dimensions[3],
                                                dimensions[0],
                                            ],
                                        )
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            },
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": 0,
                                "dsName_": "Tensor0",
                                "dsType_": "INPUT",
                                "scale_": [
                                    inputs[0]["scale"][0],
                                    inputs[0]["scale"][1],
                                    inputs[0]["scale"][2],
                                ],
                                "wordLength": 2,
                                "dataFormat_": "SEN169_FP16",
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                },
                            },
                            {
                                "ldsIdx_": 1,
                                "dsName_": "Tensor1",
                                "dsType_": "KERNEL",
                                "scale_": [
                                    inputs[1]["scale"][0],
                                    inputs[1]["scale"][2],
                                    inputs[1]["scale"][3],
                                ],
                                "wordLength": 2,
                                "dataFormat_": "SEN169_FP16",
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                },
                            },
                            {
                                "ldsIdx_": 2,
                                "dsName_": "Tensor2",
                                "dsType_": "OUTPUT",
                                "scale_": [
                                    outputs[0]["scale"][0],
                                    outputs[0]["scale"][1],
                                    outputs[0]["scale"][3],
                                ],
                                "wordLength": 2,
                                "dataFormat_": "SEN169_FP16",
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                },
                            },
                        ],
                        "computeOp_": [
                            {
                                "exUnit": "pt",
                                "opFuncName": op,
                                "attributes_": {
                                    "dataFormat_": "SEN169_FP16",
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
