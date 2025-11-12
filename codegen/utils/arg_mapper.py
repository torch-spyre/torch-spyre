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

import regex as re

from sendnn import GraphBuilder


def parse_sendnn_schema(schema):
    if "Overloaded" in schema:
        locs = [loc.start() for loc in re.finditer(schema.split("(")[0], schema)]
        locs += [len(schema)]
        schema_list = [schema[locs[i] : locs[i + 1]] for i in range(1, len(locs) - 1)]
    else:
        schema_list = [schema]

    params_list = []
    for schema in schema_list:
        parts = schema.split("(")
        params_part = parts[1].split(")")[0]
        params = [p.strip() for p in params_part.split(",")] if params_part else []
        params_list.append(params)

    return params_list


def map_types(dec_arg_type, sendnn_arg_type):
    out_type = ""

    success = 0
    if "Node" in sendnn_arg_type:
        if "TensorList" in dec_arg_type:
            out_type = "PrimaryInputList"
            success = 1
        elif "Tensor" in dec_arg_type:
            out_type = "PrimaryInput"
            success = 1
        elif (
            any([t in dec_arg_type for t in ["float", "double", "Scalar"]])
            and "Type" not in dec_arg_type
        ):
            out_type = "PrimaryInputScalar"  # TODO: Check?
            success = 1
        elif "int" in dec_arg_type and "Type" not in dec_arg_type:
            out_type = "ConstInput"  # TODO: Check?
            success = 1
    elif sendnn_arg_type == "int" or sendnn_arg_type == "SupportsInt":
        if "int" in dec_arg_type:
            out_type = "ByPass"
            success = 1
        elif "Scalar" in dec_arg_type and "Type" not in dec_arg_type:
            out_type = "int"
            success = 1
    elif sendnn_arg_type == "float":
        if (
            any([t in dec_arg_type for t in ["float", "double", "Scalar"]])
            and "Type" not in dec_arg_type
        ):
            out_type = "float"
            success = 1
    elif sendnn_arg_type == "bool":
        if "bool" in dec_arg_type:
            out_type = "ByPass"
            success = 1
    elif "TensorShape" in sendnn_arg_type:
        if "int" in dec_arg_type or "IntArrayRef" in dec_arg_type:
            out_type = "TensorShape"
            success = 1
    else:
        success = -1

    return out_type, success


def mapping_helper(pytorch_args: list, sendnn_args: list, extra_args):
    found = False
    order_list = []

    pos = len(pytorch_args)
    if pytorch_args[-1]["name"] in ["out", "Output"]:
        pos -= 1
    for i, extra_arg in enumerate(extra_args):
        insert_flag = True
        for pt_arg in pytorch_args:
            if extra_arg["name"] == pt_arg["name"]:
                if extra_arg.get("overwrite", False):
                    pt_arg["type"] = extra_arg["type"]
                    pt_arg["default"] = extra_arg["default"]
                    pt_arg["sendnn_type"] = "Default"
                insert_flag = False
                break
        if insert_flag:
            pytorch_args.insert(
                pos,
                extra_arg | {"in_signature": False, "sendnn_type": "Default"},
            )
            pos += 1

    # Arguments in the torch declaration that could not mapped into sendnn will be ignored
    map_list = ["Ignore" for _ in range(len(pytorch_args))]

    # Decide how to map each arg in sendnn function
    for sendnn_arg in sendnn_args:
        sendnn_arg_type = sendnn_arg.split(" ")[1].split(".")[-1]
        found = False

        for j, pt_arg in enumerate(pytorch_args):
            if j in order_list or found or pt_arg["name"] == "out":
                continue
            out_type, success = map_types(pt_arg["type"], sendnn_arg_type)
            if success == 1:
                map_list[j] = out_type
                order_list.append(j)
                found = True
            elif success == -1:
                print(
                    f"Unresolved argument type {sendnn_arg_type} for arg {sendnn_arg}."
                )
                break
            else:
                continue

        if found:  # continue mapping next sendnn argument
            continue
        else:
            break  # try next sendnn overload

    return found, order_list, map_list


def map_arguments(pt_declaration, op_metadata):
    sendnn_schema = getattr(
        GraphBuilder, pt_declaration["template_data"]["sendnn_func_name"]
    ).__doc__
    sendnn_args_list = parse_sendnn_schema(sendnn_schema)
    sendnn_args_list = [
        s[3:] for s in sendnn_args_list
    ]  # filter self, key, tensor_info
    pt_args_list = pt_declaration["arguments"]
    found = False
    order_list = []
    map_list = []

    # TODO: Hard filter based on contained argument types in declaration - will be supported in the future
    if any(
        [any(t in dec_arg["type"] for t in ["Dimname"]) for dec_arg in pt_args_list]
    ):
        print(f"There is an unsupported data type in {pt_declaration['name']}.")
        return False

    for sendnn_args in sendnn_args_list:
        found, order_list, map_list = mapping_helper(
            pt_args_list, sendnn_args, op_metadata.get("extra_arguments", {})
        )
        if found:  # use the mapping for this overload
            break
        else:  # try next sendnn overload
            continue

    if not found:
        print(
            f"There are additional arguments in sendnn function for operation {pt_declaration['operator_name']}.{pt_declaration['overload_name']}."
        )
        return False

    for j, dec_arg in enumerate(pt_args_list):
        if "sendnn_type" not in dec_arg:
            dec_arg["sendnn_type"] = map_list[j]
        if (
            dec_arg["sendnn_type"] in ["Ignore", "Default"]
            and "out" not in dec_arg["name"]
        ):
            print(
                f"Warning: {dec_arg['name']} will be ignored or defaulted in operation {pt_declaration['operator_name']}.{pt_declaration['overload_name']}."
            )

    pt_declaration["sendnn_arg_order_list"] = order_list
    return True
