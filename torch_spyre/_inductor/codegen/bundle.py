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


def generate_bundle(kernel_name: str, output_dir: str, specs: list[OpSpec]):
    """Output the SDSC Bundle for the OpSpecs in the given output_dir for the OpSpecs"""

    # 1. Generate SDSC.json for each OpSpec.  symbols is the global deduplicated
    #    list of offset values for arith.constant declarations.  symbol_id_offset
    #    ensures each SDSC's negative symbol ids are unique across the bundle.
    symbols: list[int] = []
    sdscs_json = []
    symbol_id_offset = 0
    for idx, ks in enumerate(specs):
        sdsc_json, local_symbol_values = compile_op_spec(
            idx, ks, symbols, symbol_id_offset
        )
        symbol_id_offset += len(local_symbol_values)
        sdscs_json.append((sdsc_json, local_symbol_values))

    # Write JSON SDSCs to file system
    for idx, (sdsc_json, _) in enumerate(sdscs_json):
        with open(os.path.join(output_dir, f"sdsc_{idx}.json"), "w") as file:
            logger.info(f"Generating {file.name}")
            json.dump(sdsc_json, file, indent=2)

    # Generate bundle.mlir.  One arith.constant per unique offset value.
    # symbol_ids are globally unique negative integers across all sdsc_execute
    # ops; the JSON for each SDSC uses the same ids via symbol_id_offset.
    with open(os.path.join(output_dir, "bundle.mlir"), "w") as file:
        logger.info(f"Generating {file.name}")
        file.write("module {\n")
        file.write("\tfunc.func @sdsc_bundle() {\n")
        for sym_idx, value in enumerate(symbols):
            file.write(f"\t\t%sym_{sym_idx + 1} = arith.constant {value} : index\n")
        id_offset = 0
        for idx, (_, local_symbol_values) in enumerate(sdscs_json):
            sym_names = [f"%sym_{symbols.index(v) + 1}" for v in local_symbol_values]
            symbol_ids = [-(id_offset + i + 1) for i in range(len(local_symbol_values))]
            id_offset += len(local_symbol_values)
            file.write(
                "\t\tsdscbundle.sdsc_execute ("
                + ", ".join(sym_names)
                + ') {sdsc_filename="sdsc_'
                + f"{idx}"
                + '.json", '
                + '"symbol_ids"=['
                + ", ".join([str(i) for i in symbol_ids])
                + "]}\n"
            )
        file.write("\t\treturn\n")
        file.write("\t}\n")
        file.write("}\n")
