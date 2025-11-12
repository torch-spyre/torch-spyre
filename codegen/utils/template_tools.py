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

from jinja2 import Environment, FileSystemLoader

from utils.arg_mapper import map_arguments
from utils.shape_extractor import infer_output_shape_stride


def generate_signature_dict(replacement_dict):
    signatures = {}

    if len(replacement_dict["returns"]) == 0:
        signatures["signature_out"] = "void"
    elif len(replacement_dict["returns"]) == 1:
        signatures["signature_out"] = replacement_dict["returns"][0]["type"]
    else:
        signatures["signature_out"] = (
            f"::std::tuple<{','.join([o['type'] for o in replacement_dict['returns']])}>"
        )

    signatures["signature_in"] = ", ".join(
        [
            f"{i['type']} {i['name']}"
            for i in replacement_dict["arguments"]
            if i.get("in_signature", True)
        ]
    )

    return signatures


def generate_from_template(
    template_dir: str, template_name: str, replacement_data: dict
):
    """
    Generates a snippet from a template file by replacing keywords.

    Args:
        replacement_data (dict): A dict containing replacement data.
    """

    template_path = f"{template_name}.jinja2"  # Path of the body template file

    env = Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path)

    output = template.render(**replacement_data) + "\n\n"

    return output


def generate_replacements(
    all_declarations, all_schemas, metadata, action="skip", only_req=False
):
    """
    Generates replacement data for pytorch ops (specified in declaration and schema files)

    Args:
        all_declarations (list): list of dicts parsed from pytorch Declarations.yaml
        all_schemas (list): list of dicts parsed from pytorch RegistrationDeclarations.yaml (indices match with declarations)
        metadata (dict): dict of metadata for each operator (contains sendnn_func_name, template_name and arg_mapping) parsed from Metadata.yaml
        action (str): what to do if the operator is not supported, options: 'skip', 'fallback', 'native_call'
        only_req (bool): set true to enable filtering with (dispatch=True, default=False)
    """
    replacements = []

    num_total_decs = len(all_declarations)
    num_supported_decs = 0

    for i, declaration in enumerate(all_declarations):
        schema = all_schemas[i]
        if only_req:  # only generate for required operations according to docs
            if not (schema["dispatch"] == "True" and schema["default"] == "False"):
                print(
                    f"Warning: {declaration['operator_name']}.{declaration['overload_name']} - Not required, skipping..."
                )
                continue

        if declaration["operator_name"] in metadata:
            declaration["template_name"] = metadata[declaration["operator_name"]][
                "template_name"
            ]
            cur_metadata = metadata[declaration["operator_name"]]
            num_supported_decs += 1
        else:
            cur_metadata = {
                "operator_name": declaration["operator_name"].capitalize(),
                "out_shape_stride_expr": "infer",
            }

            if action == "skip":  # skip
                # print(f"Warning: {dec['operator_name']}.{dec['overload_name']} - No metadata found, skipping...")
                continue
            else:
                if action == "fallback":  # use cpu fallback template
                    declaration["template_name"] = "fallback"
                elif action == "native":  # call aten::native
                    declaration["template_name"] = "native_call"
                else:
                    raise NotImplementedError(
                        f"{action} is not implemented, options: 'skip', 'fallback', 'native', 'auto'"
                    )

        # Use ordered arguments in template
        declaration["arguments"] = declaration["schema_order_arguments"]
        del declaration["schema_order_arguments"]

        # TODO: If first argument is not a Tensor (e.g. arange), skip.
        if len(declaration["arguments"]) > 0 and any(
            [
                t in declaration["arguments"][0]["type"]
                for t in ["int", "double", "float", "Scalar"]
            ]
        ):
            continue

        declaration["template_data"] = {
            "op_name": declaration["operator_name"]
            + "_"
            + (
                declaration["overload_name"]
                if declaration["overload_name"]
                else "default"
            ),
            "op_label": f'"{declaration["operator_name"].capitalize()}"',
            "sendnn_func_name": cur_metadata["sendnn_func_name"],
            "reg_name": f'"{declaration["operator_name"]}.{declaration["overload_name"]}"'
            if declaration["overload_name"]
            else f'"{declaration["operator_name"]}"',
        }

        signatures = generate_signature_dict(declaration)
        declaration |= signatures

        # if the template is base or list_inp, will try mapping args between torch and sendnn
        if declaration["template_name"] in ["base", "list_inp"]:
            maparg_success_flag = map_arguments(declaration, cur_metadata)
            if not maparg_success_flag:
                # Argument mapping has failed, so this operation is skipped
                print(
                    f"Warning: {declaration['operator_name']}.{declaration['overload_name']} - Argument mapping failed, skipping..."
                )
                continue

        # For view ops, skip dtype overload
        if (
            declaration["template_name"] in ["view", "view_copy"]
            and declaration["overload_name"] == "dtype"
        ):
            print(
                f"Warning: {declaration['operator_name']}.{declaration['overload_name']} - View op with dtype overload, skipping..."
            )
            continue

        for dec_arg in declaration["arguments"]:
            if "default" in dec_arg and isinstance(dec_arg["default"], bool):
                dec_arg["default"] = str(dec_arg["default"]).lower()

        # unless there is a provided out_shape_stride_expr method, we will skip output shape and stride inference (first input will be used directly)
        declaration["out_shape_stride_expr"] = cur_metadata.get(
            "out_shape_stride_expr", "bypass"
        )

        # if the template is base and out_shape_stride_expr is infer, we can try auto shape inference
        if (
            declaration["template_name"] == "base"
            and declaration["out_shape_stride_expr"] == "infer"
        ):
            output_shape_stride_list, bypass_flag = infer_output_shape_stride(
                declaration
            )
            if output_shape_stride_list is None:
                # Output shape/stride inference has failed, so this operation is skipped
                # print(f"Warning: {dec['operator_name']}.{dec['overload_name']} - Output shape/stride inference failed, skipping...")
                continue
            else:
                if bypass_flag:
                    declaration["out_shape_stride_expr"] = "bypass"
                    # Output shape inference is not necessary
                    pass
                else:
                    # inferred symbolic representation that will be used in the template
                    for i, output_shape_stride in enumerate(output_shape_stride_list):
                        if output_shape_stride:
                            declaration["returns"][i]["shape"] = output_shape_stride[
                                "shape"
                            ]
                            declaration["returns"][i]["stride"] = output_shape_stride[
                                "stride"
                            ]

        replacements.append(declaration)

    print(f"{num_supported_decs} of {num_total_decs} declarations are supported.")

    return replacements
