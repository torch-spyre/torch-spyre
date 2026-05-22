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
import os

from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.op_spec import OpSpec
from torch_spyre._inductor.logging_utils import get_inductor_logger


logger = get_inductor_logger("sdsc_compile")


def update_core_id_to_wk_slice(sdsc_json: dict, ks: OpSpec, argIdx_to_split: dict):
    """
    Update tensor's split info under "scheduleTree_" (ONLY for LX tensors.)
    Currently this field is always {}. If a LX tensor and op have different splits,
    backend compiler will try to reconcile, e.g. collect the entire tensor to HBM.
    NOTE
    1. Tensor split is the "op split" when this tensor was generated.
    2. sdsc_json will be updated inplace, no need to return.
    """
    idx_opfunc = list(sdsc_json.keys())[0]
    opfunc = idx_opfunc.split("_")[1]
    op_split = sdsc_json[idx_opfunc]["coreIdToWkSlice_"]
    op_root = sdsc_json[idx_opfunc]["dscs_"][0][opfunc]
    full_ten_size = op_root["N_"]
    slice_shape = [s//op_split["0"][sym] for sym, s in full_ten_size.items()]
    sym_order = op_root["primaryDsInfo_"]["OUTPUT"]["layoutDimOrder_"]
    # both symbols and order could vary from op to op

    # make sure the order in scheduleTree follows ks.args (which controls ldsIdx)
    sch_tree = op_root["scheduleTree_"]
    assert [t["ldsIdx_"] for t in sch_tree] == list(range(len(sch_tree)))

    for ks_arg_idx, ks_arg in enumerate(ks.args):

        if "lx" not in ks_arg.allocation:
            continue
        # when ks_arg.arg_index=-1, alloc could be "lx" or "pool", only handle lx cases

        global_arg_idx = f"lx_{ks_arg.allocation["lx"]}"
        if not ks_arg.is_input:
            argIdx_to_split[global_arg_idx] = (op_split, sym_order, slice_shape)
        elif global_arg_idx in argIdx_to_split:
            tensor_split, sym_order_prev, slice_shape_prev = argIdx_to_split[global_arg_idx]
            if slice_shape != slice_shape_prev:
                # what if order changed
                sch_tree[ks_arg_idx]["coordinates_"]["coreIdToWkSlice_"] = tensor_split


def generate_bundle(kernel_name: str, output_dir: str, specs: list[OpSpec]):
    """Output the SDSC Bundle for the OpSpecs in the given output_dir for the OpSpecs"""

    # 1. Generate SDSC.json for each OpSpec
    sdscs_json = []
    argIdx_to_split = {}
    for idx, ks in enumerate(specs):
        sdsc_json = compile_op_spec(idx, ks)
        # update_core_id_to_wk_slice(sdsc_json, ks, argIdx_to_split)
        sdscs_json.append(sdsc_json)

    # Write JSON SDSCs to file system
    files = []
    for sdsc_json in sdscs_json:
        sdsc_name = next(iter(sdsc_json))
        file_name = f"sdsc_{sdsc_name}.json"
        files.append(file_name)
        with open(os.path.join(output_dir, file_name), "w") as file:
            logger.info(f"Generating {file.name}")
            json.dump(sdsc_json, file, indent=2)

    # Generate bundle.mlir
    with open(os.path.join(output_dir, "bundle.mlir"), "w") as file:
        logger.info(f"Generating {file.name}")
        file.write("module {\n")
        file.write("\tfunc.func @sdsc_bundle() {\n")
        for f in files:
            file.write('\t\tsdscbundle.sdsc_execute () {sdsc_filename="' + f + '"}\n')
        file.write("\t\treturn\n")
        file.write("\t}\n")
        file.write("}\n")
