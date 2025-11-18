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
    transpose_1_3 = 1 in transposed_dims
    transpose_2_3 = 2 in transposed_dims
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
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["mb", "x", "out", "y"],
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": {
                                    "mb": dimensions[2],
                                    "out": dimensions[-1],
                                    "x": dimensions[1],
                                    "y": dimensions[0],
                                },
                                "dimToStickSize_": {"out": 64},
                                "validGap_": {
                                    "mb": [[dimensions[2], 0]],
                                    "out": [[dimensions[-1], 0]],
                                    "x": [[dimensions[1], 0]],
                                    "y": [[dimensions[0], 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": {
                                            "mb": 64 if transpose_2_3 else 1,
                                            "out": 64,
                                            "x": 64 if transpose_1_3 else 1,
                                            "y": 64 if transpose_0_3 else 1,
                                        },
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [[64 if transpose_2_3 else 1, 0]],
                                            "x": [[64 if transpose_1_3 else 1, 0]],
                                            "y": [[64 if transpose_0_3 else 1, 0]],
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
                                            "mb": 64 if transpose_2_3 else 1,
                                            "out": 64,
                                            "x": 64 if transpose_1_3 else 1,
                                            "y": 64 if transpose_0_3 else 1,
                                        },
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [[64 if transpose_2_3 else 1, 0]],
                                            "x": [[64 if transpose_1_3 else 1, 0]],
                                            "y": [[64 if transpose_0_3 else 1, 0]],
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
                                "layoutDimOrder_": ["mb", "x", "y", "out"],
                                "stickDimOrder_": ["y"],
                                "dimToLayoutSize_": {
                                    "mb": dimensions[2],
                                    "out": dimensions[-1],
                                    "x": dimensions[1],
                                    "y": dimensions[0],
                                },
                                "dimToStickSize_": {"y": 64},
                                "validGap_": {
                                    "mb": [[dimensions[2], 0]],
                                    "out": [[dimensions[-1], 0]],
                                    "x": [[dimensions[1], 0]],
                                    "y": [[dimensions[0], 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": "p0",
                                        "dimToSize_": {
                                            "mb": 64 if transpose_2_3 else 1,
                                            "out": 64,
                                            "x": 64 if transpose_1_3 else 1,
                                            "y": 64 if transpose_0_3 else 1,
                                        },
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [[64 if transpose_2_3 else 1, 0]],
                                            "x": [[64 if transpose_1_3 else 1, 0]],
                                            "y": [[64 if transpose_0_3 else 1, 0]],
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
                                            "mb": 64 if transpose_2_3 else 1,
                                            "out": 64,
                                            "x": 64 if transpose_1_3 else 1,
                                            "y": 64 if transpose_0_3 else 1,
                                        },
                                        "validGap_": {
                                            "out": [[64, 0]],
                                            "mb": [[64 if transpose_2_3 else 1, 0]],
                                            "x": [[64 if transpose_1_3 else 1, 0]],
                                            "y": [[64 if transpose_0_3 else 1, 0]],
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
                                        "out": dimensions[-1] // 64,
                                        "mb": dimensions[2],
                                        "x": dimensions[1],
                                        "y": dimensions[0] // 64
                                        if transpose_0_3
                                        else dimensions[0],
                                    },
                                    "loopCountL3SU": {
                                        "out": dimensions[-1] // 64,
                                        "mb": dimensions[2],
                                        "x": dimensions[1],
                                        "y": dimensions[0] // 64
                                        if transpose_0_3
                                        else dimensions[0],
                                    },
                                    "addr_info_": {
                                        "l3lu": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": 1,
                                                "out": dimensions[2] * dimensions[1],
                                                "y": dimensions[2]
                                                * dimensions[1]
                                                * dimensions[-1],
                                                "x": dimensions[2],
                                            },
                                        },
                                        "l3su": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": 1,
                                                "out": dimensions[0]
                                                * dimensions[1]
                                                * dimensions[2],
                                                "y": dimensions[2] * dimensions[1],
                                                "x": dimensions[2],
                                            },
                                        },
                                    },
                                    "inpPieceOrder": [
                                        f"p{i % 2}"
                                        for i in range(
                                            dimensions[0]
                                            * dimensions[1]
                                            * dimensions[2]
                                            * dimensions[-1]
                                            // (64 * 64)
                                        )
                                    ],
                                    "outPieceOrder": [
                                        f"p{i % 2}"
                                        for i in range(
                                            dimensions[0]
                                            * dimensions[1]
                                            * dimensions[2]
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
