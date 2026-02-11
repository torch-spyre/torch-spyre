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

from torch_spyre._inductor.codegen.compute_ops import num_bytes


def generate_transpose(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    return {
        "reshape": {
            "numCoresUsed_": 1,
            "dscs_": [],
            "datadscs_": [
                {
                    "reshape": {
                        "coreIdsUsed_": [0],
                        "dimPool_": ["mb", "out"],
                        "primaryDs_": [{"name_": "pds0", "dimNames": ["mb", "out"]}],
                        "labeledDs_": [
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["mb", "out"],
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": {
                                    "mb": dimensions[0],
                                    "out": dimensions[1],
                                },
                                "dimToStickSize_": {"out": 64},
                                "validGap_": {
                                    "mb": [[64, 0]],
                                    "out": [[64, 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": {"mb": 64, "out": 64},
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[inputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [0],
                                            },
                                        ],
                                    },
                                    {
                                        "key_": "p1",
                                        "dimToSize_": {"mb": 64, "out": 64},
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [0],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [8192],
                                            },
                                        ],
                                    },
                                ],
                                "hbmStartAddress_": pointers[inputs[0]["name"]] // 128,
                            },
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["out", "mb"],
                                "stickDimOrder_": ["mb"],
                                "dimToLayoutSize_": {
                                    "mb": dimensions[0],
                                    "out": dimensions[1],
                                },
                                "dimToStickSize_": {"mb": 64},
                                "validGap_": {
                                    "mb": [[64, 0]],
                                    "out": [[64, 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": {"mb": 64, "out": 64},
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [[64, 0]],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[outputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [16384],
                                            },
                                        ],
                                    },
                                    {
                                        "key_": "p1",
                                        "dimToSize_": {"mb": 64, "out": 64},
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [0],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [24576],
                                            },
                                        ],
                                    },
                                ],
                                "hbmStartAddress_": pointers[outputs[0]["name"]] // 128,
                            },
                        ],
                        "op": {
                            "name": "ReStickifyOpWithPTHBM",
                            "coreIDtoANInfo": {
                                "0": {
                                    "loopCount": {
                                        "out": dimensions[0] // 64,
                                        "mb": dimensions[1] // 64,
                                    },
                                    "loopCountL3SU": {
                                        "out": dimensions[0] // 64,
                                        "mb": dimensions[1] // 64,
                                    },
                                    "addr_info_": {
                                        "l3lu": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": dimensions[0],
                                                "out": 64,
                                            },
                                        },
                                        "l3su": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": 64,
                                                "out": dimensions[1],
                                            },
                                        },
                                    },
                                    "inpPieceOrder": [
                                        f"p{i % 2}"
                                        for i in range(
                                            dimensions[0] * dimensions[1] // 4096
                                        )
                                    ],
                                    "outPieceOrder": [
                                        f"p{i % 2}"
                                        for i in range(
                                            dimensions[0] * dimensions[1] // 4096
                                        )
                                    ],
                                }
                            },
                            "numClToUse": 1,
                            "cl0ToLxOffsetLU": 0,
                            "cl0ToLxOffsetSU": 0,
                        },
                    }
                }
            ],
        }
    }


# 3D transpose restickify
def generate_transpose_3d_stick(
    pointers, *, op, dimensions, inputs, outputs, transposed_dims, **kwargs
):
    transpose_0_2 = 0 in transposed_dims and 2 in transposed_dims
    return {
        "reshape": {
            "numCoresUsed_": 1,
            "dscs_": [],
            "datadscs_": [
                {
                    "reshape": {
                        "coreIdsUsed_": [0],
                        "dimPool_": ["mb", "out", "x"],
                        "primaryDs_": [
                            {"name_": "pds0", "dimNames": ["mb", "out", "x"]}
                        ],
                        "labeledDs_": [
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["mb", "out", "x"],
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": {
                                    "mb": dimensions[0],
                                    "out": dimensions[-1],
                                    "x": dimensions[1],
                                },
                                "dimToStickSize_": {"out": 64},
                                "validGap_": {
                                    "mb": [[64, 0]],
                                    "out": [[64, 0]],
                                    "x": [[64, 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": {
                                            "mb": 64 if transpose_0_2 else 1,
                                            "out": 64,
                                            "x": 1 if transpose_0_2 else 64,
                                        },
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [
                                                [
                                                    64 if transpose_0_2 else 1,
                                                    0,
                                                ]
                                            ],
                                            "x": [
                                                [
                                                    1 if transpose_0_2 else 64,
                                                    0,
                                                ]
                                            ],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[inputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [0],
                                            },
                                        ],
                                    },
                                    {
                                        "key_": "p1",
                                        "dimToSize_": {
                                            "mb": 64 if transpose_0_2 else 1,
                                            "out": 64,
                                            "x": 1 if transpose_0_2 else 64,
                                        },
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [
                                                [
                                                    64 if transpose_0_2 else 1,
                                                    0,
                                                ]
                                            ],
                                            "x": [
                                                [
                                                    1 if transpose_0_2 else 64,
                                                    0,
                                                ]
                                            ],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [0],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [8192],
                                            },
                                        ],
                                    },
                                ],
                                "hbmStartAddress_": pointers[inputs[0]["name"]] // 128,
                            },
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["out", "mb", "x"]
                                if transpose_0_2
                                else ["mb", "x", "out"],
                                "stickDimOrder_": ["mb" if transpose_0_2 else "x"],
                                "dimToLayoutSize_": {
                                    "mb": dimensions[0],
                                    "out": dimensions[-1],
                                    "x": dimensions[1],
                                },
                                "dimToStickSize_": {"mb": 64}
                                if transpose_0_2
                                else {"x": 64},
                                "validGap_": {
                                    "mb": [[64, 0]],
                                    "out": [[64, 0]],
                                    "x": [[64, 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": {
                                            "mb": 64 if transpose_0_2 else 1,
                                            "out": 64,
                                            "x": 1 if transpose_0_2 else 64,
                                        },
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [
                                                [
                                                    64 if transpose_0_2 else 1,
                                                    0,
                                                ]
                                            ],
                                            "x": [
                                                [
                                                    1 if transpose_0_2 else 64,
                                                    0,
                                                ]
                                            ],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[outputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [16384],
                                            },
                                        ],
                                    },
                                    {
                                        "key_": "p1",
                                        "dimToSize_": {
                                            "mb": 64 if transpose_0_2 else 1,
                                            "out": 64,
                                            "x": 1 if transpose_0_2 else 64,
                                        },
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [
                                                [
                                                    64 if transpose_0_2 else 1,
                                                    0,
                                                ]
                                            ],
                                            "x": [
                                                [
                                                    1 if transpose_0_2 else 64,
                                                    0,
                                                ]
                                            ],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [0],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [24576],
                                            },
                                        ],
                                    },
                                ],
                                "hbmStartAddress_": pointers[outputs[0]["name"]] // 128,
                            },
                        ],
                        "op": {
                            "name": "ReStickifyOpWithPTHBM",
                            "coreIDtoANInfo": {
                                "0": {
                                    "loopCount": {
                                        "out": dimensions[0] // 64
                                        if transpose_0_2
                                        else dimensions[-1] // 64,
                                        "mb": dimensions[-1] // 64
                                        if transpose_0_2
                                        else dimensions[0],
                                        "x": dimensions[1]
                                        if transpose_0_2
                                        else dimensions[1] // 64,
                                    },
                                    "loopCountL3SU": {
                                        "out": dimensions[0] // 64
                                        if transpose_0_2
                                        else dimensions[-1] // 64,
                                        "mb": dimensions[-1] // 64
                                        if transpose_0_2
                                        else dimensions[0],
                                        "x": dimensions[1]
                                        if transpose_0_2
                                        else dimensions[1] // 64,
                                    },
                                    "addr_info_": {
                                        "l3lu": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": dimensions[0]
                                                if transpose_0_2
                                                else 1,
                                                "out": 64
                                                if transpose_0_2
                                                else dimensions[0],
                                                "x": dimensions[-1]
                                                * dimensions[0]
                                                // 64
                                                if transpose_0_2
                                                else dimensions[0] * dimensions[-1],
                                            },
                                        },
                                        "l3su": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": 64 if transpose_0_2 else 1,
                                                "out": dimensions[-1]
                                                if transpose_0_2
                                                else dimensions[0] * dimensions[1],
                                                "x": dimensions[-1]
                                                * dimensions[0]
                                                // 64
                                                if transpose_0_2
                                                else dimensions[0],
                                            },
                                        },
                                    },
                                    "inpPieceOrder": [
                                        f"p{i % 2}"
                                        for i in range(
                                            dimensions[0]
                                            * dimensions[1]
                                            * dimensions[-1]
                                            // (64 * 64)
                                        )
                                    ],
                                    "outPieceOrder": [
                                        f"p{i % 2}"
                                        for i in range(
                                            dimensions[0]
                                            * dimensions[1]
                                            * dimensions[-1]
                                            // (64 * 64)
                                        )
                                    ],
                                }
                            },
                            "numClToUse": 1,
                            "cl0ToLxOffsetLU": 0,
                            "cl0ToLxOffsetSU": 0,
                        },
                    }
                }
            ],
        }
    }


def generate_slice(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    return {
        "reshape": {
            "numCoresUsed_": 1,
            "dscs_": [],
            "coreIdToDscSchedule": {"0": [[0, -1, 0, 0]]},
            "datadscs_": [
                {
                    "reshape": {
                        "coreIdsUsed_": [0],
                        "dimPool_": ["mb", "out"],
                        "primaryDs_": [{"name_": "pds0", "dimNames": ["mb", "out"]}],
                        "labeledDs_": [
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["mb", "out"],
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": {
                                    "mb": 64,
                                    "out": dimensions[0],
                                },
                                "dimToStickSize_": {"out": 64},
                                "validGap_": {
                                    "mb": [[64, 0]],
                                    "out": [[dimensions[0], 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": f"p{i}",
                                        "dimToSize_": {"mb": 1, "out": 64},
                                        "validGap_": {
                                            "mb": [[1, 0]],
                                            "out": [[64, 0]],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[inputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [0],
                                            },
                                        ],
                                    }
                                    for i in range(dimensions[0] // 64)
                                ],
                                "hbmStartAddress_": pointers[inputs[0]["name"]] // 128,
                            },
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["mb", "out"],
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": {
                                    "mb": 1,
                                    "out": dimensions[0],
                                },
                                "dimToStickSize_": {"out": 64},
                                "validGap_": {
                                    "mb": [[1, 0]],
                                    "out": [[dimensions[0], 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": f"p{i}",
                                        "dimToSize_": {"mb": 1, "out": 64},
                                        "validGap_": {
                                            "mb": [[1, 0]],
                                            "out": [[64, 0]],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[outputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [16384],
                                            },
                                        ],
                                    }
                                    for i in range(dimensions[0] // 64)
                                ],
                                "hbmStartAddress_": pointers[outputs[0]["name"]] // 128,
                            },
                        ],
                        "op": {
                            "name": "STCDPOpHBM",
                            "gtrIdsUsed": [],
                            "coreIDtoANInfo": {
                                "0": {
                                    "loopCount": {
                                        "out": dimensions[0] // 64,
                                        "mb": 1,
                                    },
                                    "loopCountL3SU": {},
                                    "addr_info_": {
                                        "l3lu": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": 1,
                                                "out": 64,
                                            },
                                        },
                                        "l3su": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": 1,
                                                "out": 1,
                                            },
                                        },
                                    },
                                    "inpPieceOrder": [
                                        f"p{i}" for i in range(dimensions[0] // 64)
                                    ],
                                    "outPieceOrder": [
                                        f"p{i}" for i in range(dimensions[0] // 64)
                                    ],
                                }
                            },
                            "numClToUse": 1,
                            "cl0ToLxOffsetLU": 0,
                            "cl0ToLxOffsetSU": 0,
                        },
                    }
                }
            ],
        }
    }


def generate_transpose_4d_stick(
    pointers, *, op, dimensions, inputs, outputs, transposed_dims, **kwargs
):
    transpose_0_3 = 0 in transposed_dims
    transpose_2_3 = 2 in transposed_dims
    input_dtype = inputs[0]["device_layout"].device_dtype
    word_length = num_bytes(input_dtype)
    data_format = input_dtype.name
    elems_per_stick = input_dtype.elems_per_stick()
    piece_count = (
        dimensions[0]
        * dimensions[1]
        * dimensions[-1]
        * dimensions[2]
        // (elems_per_stick * elems_per_stick)
    )
    valid_gaps = {
        "mb": [[dimensions[0], 0]],
        "out": [[dimensions[-1], 0]],
        "x": [[dimensions[1], 0]],
        "y": [[dimensions[2], 0]],
    }
    if transpose_0_3:
        input_layout = ["mb", "out", "x", "y"]
        output_layout = ["out", "mb", "x", "y"]
        output_stick = {"mb": elems_per_stick}
        dim_map = {
            "mb": dimensions[0],
            "out": dimensions[-1],
            "x": dimensions[1],
            "y": dimensions[2],
        }
        l3su_offsets = {
            "mb": elems_per_stick,
            "out": dimensions[-1],
            "y": dimensions[0] * dimensions[1] * dimensions[-1] // elems_per_stick,
            "x": dimensions[-1] * dimensions[0] // elems_per_stick,
        }
        l3lu_offsets = {
            "mb": dimensions[0],
            "out": elems_per_stick,
            "y": dimensions[0] * dimensions[1] * dimensions[-1] // elems_per_stick,
            "x": dimensions[-1] * dimensions[0] // elems_per_stick,
        }
        loop_counts = {
            "mb": dimensions[-1] // elems_per_stick,
            "out": dimensions[0] // elems_per_stick,
            "x": dimensions[1],
            "y": dimensions[2],
        }
        piece_sizes = {"mb": elems_per_stick, "out": elems_per_stick, "x": 1, "y": 1.0}
        piece_valid_gaps = {
            "mb": [[piece_sizes["mb"], 0]],
            "out": [[piece_sizes["out"], 0]],
            "x": [[piece_sizes["x"], 0]],
            "y": [[piece_sizes["y"], 0]],
        }
    elif transpose_2_3:
        input_layout = ["mb", "out", "y", "x"]
        output_layout = ["mb", "y", "out", "x"]
        output_stick = {"y": elems_per_stick}
        dim_map = {
            "mb": dimensions[0],
            "out": dimensions[-1],
            "x": dimensions[1],
            "y": dimensions[2],
        }
        l3su_offsets = {
            "mb": 1,
            "out": dimensions[0],
            "y": dimensions[2] * dimensions[0],
            "x": dimensions[0] * dimensions[-1] * dimensions[2] // elems_per_stick,
        }
        l3lu_offsets = {
            "mb": 1,
            "out": dimensions[-1] * dimensions[0],
            "y": dimensions[0],
            "x": dimensions[0] * dimensions[-1] * dimensions[2] // elems_per_stick,
        }
        loop_counts = {
            "mb": dimensions[0],
            "out": dimensions[2] // elems_per_stick,
            "x": dimensions[1],
            "y": dimensions[-1] // elems_per_stick,
        }
        piece_sizes = {
            "mb": 1,
            "out": elems_per_stick,
            "x": 1,
            "y": elems_per_stick,
        }
        piece_valid_gaps = {
            "mb": [[piece_sizes["mb"], 0]],
            "out": [[piece_sizes["out"], 0]],
            "x": [[piece_sizes["x"], 0]],
            "y": [[piece_sizes["y"], 0]],
        }
    else:  # transpose_1_3
        input_layout = ["mb", "out", "y", "x"]
        output_layout = ["mb", "x", "y", "out"]
        output_stick = {"x": elems_per_stick}
        dim_map = {
            "mb": dimensions[0],
            "out": dimensions[-1],
            "x": dimensions[1],
            "y": dimensions[2],
        }
        l3su_offsets = {
            "mb": 1,
            "out": dimensions[0],
            "x": dimensions[1] * dimensions[0] * dimensions[2],
            "y": dimensions[0] * dimensions[1] // elems_per_stick,
        }
        l3lu_offsets = {
            "mb": 1,
            "x": dimensions[0],
            "out": dimensions[-1] * dimensions[0] * dimensions[2],
            "y": dimensions[0] * dimensions[-1] // elems_per_stick,
        }
        loop_counts = {
            "mb": dimensions[0],
            "out": dimensions[1] // elems_per_stick,
            "y": dimensions[2],
            "x": dimensions[-1] // elems_per_stick,
        }
        piece_sizes = {
            "mb": 1,
            "out": elems_per_stick,
            "y": 1,
            "x": elems_per_stick,
        }
        piece_valid_gaps = {
            "mb": [[piece_sizes["mb"], 0]],
            "out": [[piece_sizes["out"], 0]],
            "x": [[piece_sizes["x"], 0]],
            "y": [[piece_sizes["y"], 0]],
        }

    return {
        "reshape": {
            "numCoresUsed_": 1,
            "dscs_": [],
            "datadscs_": [
                {
                    "reshape": {
                        "coreIdsUsed_": [0],
                        "dimPool_": ["mb", "out", "x", "y"],
                        "primaryDs_": [
                            {
                                "name_": "pds0",
                                "dimNames": ["mb", "out", "y", "x"],
                            }
                        ],
                        "labeledDs_": [
                            {
                                "pdsName_": "pds0",
                                "wordLength": word_length,
                                "dataformat": data_format,
                                "layoutDimOrder_": input_layout,
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": dim_map,
                                "dimToStickSize_": {"out": 64},
                                "validGap_": valid_gaps,
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": piece_sizes,
                                        "validGap_": piece_valid_gaps,
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[inputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [0],
                                            },
                                        ],
                                    },
                                    {
                                        "key_": "p1",
                                        "dimToSize_": piece_sizes,
                                        "validGap_": piece_valid_gaps,
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [0],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [8192],
                                            },
                                        ],
                                    },
                                ],
                                "hbmStartAddress_": pointers[inputs[0]["name"]] // 128,
                            },
                            {
                                "pdsName_": "pds0",
                                "wordLength": word_length,
                                "dataformat": data_format,
                                "layoutDimOrder_": output_layout,
                                "stickDimOrder_": list(output_stick.keys()),
                                "dimToLayoutSize_": dim_map,
                                "dimToStickSize_": output_stick,
                                "validGap_": valid_gaps,
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": piece_sizes,
                                        "validGap_": piece_valid_gaps,
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[outputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [16384],
                                            },
                                        ],
                                    },
                                    {
                                        "key_": "p1",
                                        "dimToSize_": piece_sizes,
                                        "validGap_": piece_valid_gaps,
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [0],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [24576],
                                            },
                                        ],
                                    },
                                ],
                                "hbmStartAddress_": pointers[outputs[0]["name"]] // 128,
                            },
                        ],
                        "op": {
                            "name": "ReStickifyOpWithPTHBM",
                            "coreIDtoANInfo": {
                                "0": {
                                    "loopCount": loop_counts,
                                    "loopCountL3SU": loop_counts,
                                    "addr_info_": {
                                        "l3lu": {
                                            "type_": "stride",
                                            "offset_": l3lu_offsets,
                                        },
                                        "l3su": {
                                            "type_": "stride",
                                            "offset_": l3su_offsets,
                                        },
                                    },
                                    "inpPieceOrder": [
                                        f"p{i % 2}" for i in range(piece_count)
                                    ],
                                    "outPieceOrder": [
                                        f"p{i % 2}" for i in range(piece_count)
                                    ],
                                }
                            },
                            "numClToUse": 1,
                            "cl0ToLxOffsetLU": 0,
                            "cl0ToLxOffsetSU": 0,
                        },
                    }
                }
            ],
        }
    }


def generate_clone(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    ndims = len(dimensions)
    # Get data type information from inputs
    input_dtype = inputs[0]["device_layout"].device_dtype
    word_length = num_bytes(input_dtype)
    data_format = input_dtype.name
    elems_per_stick = input_dtype.elems_per_stick()
    if ndims == 1:
        layout = ["out"]
        dim_map = {"out": dimensions[0]}
        offsets = {"out": 1}
        loop_counts = {"out": dimensions[0] // elems_per_stick}
        piece_valid_gaps = {"out": [[elems_per_stick, 0]]}
        piece_sizes = {"out": elems_per_stick}
        valid_gaps = {"out": [[dimensions[0], 0]]}
        piece_count = dimensions[0] // elems_per_stick
    elif ndims == 2:
        layout = ["mb", "out"]
        dim_map = {"mb": dimensions[0], "out": dimensions[-1]}
        offsets = {
            "mb": elems_per_stick if dimensions[0] % elems_per_stick == 0 else 1,
            "out": dimensions[0],
        }
        loop_counts = {
            "mb": dimensions[0] // elems_per_stick
            if dimensions[0] % elems_per_stick == 0
            else dimensions[0],
            "out": dimensions[-1] // elems_per_stick,
        }
        piece_sizes = {
            "mb": elems_per_stick if dimensions[0] % elems_per_stick == 0 else 1,
            "out": elems_per_stick,
        }
        piece_valid_gaps = {
            "mb": [[piece_sizes["mb"], 0]],
            "out": [[piece_sizes["out"], 0]],
        }
        valid_gaps = {"mb": [[dimensions[0], 0]], "out": [[dimensions[-1], 0]]}
        piece_count = (
            dimensions[0]
            * dimensions[-1]
            // (
                elems_per_stick * elems_per_stick
                if dimensions[0] % elems_per_stick == 0
                else elems_per_stick
            )
        )
    elif ndims == 3:
        layout = ["mb", "out", "x"]
        dim_map = {"mb": dimensions[0], "out": dimensions[-1], "x": dimensions[1]}
        offsets = {
            "mb": elems_per_stick if dimensions[0] % elems_per_stick == 0 else 1,
            "out": dimensions[0],
            "x": dimensions[-1] * dimensions[0] // elems_per_stick,
        }
        loop_counts = {
            "mb": dimensions[0] // elems_per_stick
            if dimensions[0] % elems_per_stick == 0
            else dimensions[0],
            "out": dimensions[-1] // elems_per_stick,
            "x": dimensions[1],
        }
        piece_sizes = {
            "mb": elems_per_stick if dimensions[0] % elems_per_stick == 0 else 1,
            "out": elems_per_stick,
            "x": 1,
        }
        piece_valid_gaps = {
            "mb": [[piece_sizes["mb"], 0]],
            "out": [[piece_sizes["out"], 0]],
            "x": [[piece_sizes["x"], 0]],
        }
        valid_gaps = {
            "mb": [[dimensions[0], 0]],
            "out": [[dimensions[-1], 0]],
            "x": [[dimensions[1], 0]],
        }
        piece_count = (
            dimensions[0]
            * dimensions[1]
            * dimensions[-1]
            // (
                elems_per_stick * elems_per_stick
                if dimensions[0] % elems_per_stick == 0
                else elems_per_stick
            )
        )
    else:  # 4d
        layout = ["mb", "out", "x", "y"]
        dim_map = {
            "mb": dimensions[0],
            "out": dimensions[-1],
            "x": dimensions[1],
            "y": dimensions[2],
        }
        offsets = {
            "mb": elems_per_stick if dimensions[0] % elems_per_stick == 0 else 1,
            "out": dimensions[0],
            "x": dimensions[-1] * dimensions[0] // elems_per_stick,
            "y": dimensions[-1] * dimensions[0] * dimensions[1] // elems_per_stick,
        }
        loop_counts = {
            "mb": dimensions[0] // elems_per_stick
            if dimensions[0] % elems_per_stick == 0
            else dimensions[0],
            "out": dimensions[-1] // elems_per_stick,
            "x": dimensions[1],
            "y": dimensions[2],
        }
        piece_sizes = {
            "mb": elems_per_stick if dimensions[0] % elems_per_stick == 0 else 1,
            "out": elems_per_stick,
            "x": 1,
            "y": 1,
        }
        piece_valid_gaps = {
            "mb": [[piece_sizes["mb"], 0]],
            "out": [[piece_sizes["out"], 0]],
            "x": [[piece_sizes["x"], 0]],
            "y": [[piece_sizes["y"], 0]],
        }
        valid_gaps = {
            "mb": [[dimensions[0], 0]],
            "out": [[dimensions[-1], 0]],
            "x": [[dimensions[1], 0]],
            "y": [[dimensions[2], 0]],
        }
        piece_count = (
            dimensions[0]
            * dimensions[1]
            * dimensions[-1]
            * dimensions[2]
            // (
                elems_per_stick * elems_per_stick
                if dimensions[0] % elems_per_stick == 0
                else elems_per_stick
            )
        )

    return {
        "clone": {
            "numCoresUsed_": 1,
            "dscs_": [],
            "coreIdToDscSchedule": {"0": [[0, -1, 0, 0]]},
            "datadscs_": [
                {
                    "clone": {
                        "coreIdsUsed_": [0],
                        "dimPool_": layout,
                        "primaryDs_": [{"name_": "pds0", "dimNames": layout}],
                        "labeledDs_": [
                            {
                                "pdsName_": "pds0",
                                "wordLength": word_length,
                                "dataformat": data_format,
                                "layoutDimOrder_": layout,
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": dim_map,
                                "dimToStickSize_": {"out": elems_per_stick},
                                "validGap_": valid_gaps,
                                "PieceInfo": [
                                    {
                                        "key_": f"p{i}",
                                        "dimToSize_": piece_sizes,
                                        "validGap_": piece_valid_gaps,
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[inputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [0],
                                            },
                                        ],
                                    }
                                    for i in range(piece_count)
                                ],
                                "hbmStartAddress_": pointers[inputs[0]["name"]] // 128,
                            },
                            {
                                "pdsName_": "pds0",
                                "wordLength": word_length,
                                "dataformat": data_format,
                                "layoutDimOrder_": layout,
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": dim_map,
                                "dimToStickSize_": {"out": elems_per_stick},
                                "validGap_": valid_gaps,
                                "PieceInfo": [
                                    {
                                        "key_": f"p{i}",
                                        "dimToSize_": piece_sizes,
                                        "validGap_": piece_valid_gaps,
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[outputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [16384],
                                            },
                                        ],
                                    }
                                    for i in range(piece_count)
                                ],
                                "hbmStartAddress_": pointers[outputs[0]["name"]] // 128,
                            },
                        ],
                        "op": {
                            "name": "STCDPOpHBM",
                            "gtrIdsUsed": [],
                            "coreIDtoANInfo": {
                                "0": {
                                    "loopCount": loop_counts,
                                    "loopCountL3SU": {},
                                    "addr_info_": {
                                        "l3lu": {
                                            "type_": "stride",
                                            "offset_": offsets,
                                        },
                                        "l3su": {"type_": "stride", "offset_": offsets},
                                    },
                                    "inpPieceOrder": [
                                        f"p{i}" for i in range(piece_count)
                                    ],
                                    "outPieceOrder": [
                                        f"p{i}" for i in range(piece_count)
                                    ],
                                }
                            },
                            "numClToUse": 1,
                            "cl0ToLxOffsetLU": 0,
                            "cl0ToLxOffsetSU": 0,
                        },
                    }
                }
            ],
        }
    }
