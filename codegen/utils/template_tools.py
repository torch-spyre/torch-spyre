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

from utils.shape_extractor import infer_output_shape_stride

import regex as re
from typing import List


def extract_scalar_arg_names(schema_string: str) -> List[str]:
    """
    Extract scalar argument names from PyTorch operator schema.

    Examples:
        extract_scalar_arg_names("aten::add.Tensor(Tensor self, *, Scalar alpha=1) -> Tensor")
        ['alpha']
    """
    # Extract arguments from parentheses
    match = re.search(r"\((.*?)\)\s*->", schema_string)
    if not match:
        return []
    args_str = match.group(1)
    # Find all Scalar arguments with their names
    # Pattern: Scalar (optionally ?) followed by whitespace and name
    pattern = r"Scalar\??[\s]+([a-zA-Z_][a-zA-Z0-9_]*)"
    all_scalar_names = re.findall(pattern, args_str)
    # Filter out alpha and beta
    return [name for name in all_scalar_names if name not in ["alpha", "beta"]]


def get_args_with_default_vals(schema_string):
    """
    Extract keyword-only argument names (those after '*')
    from a PyTorch operator schema string.

    In PyTorch schemas, arguments after '*' are keyword-only and typically have
    default values. This function identifies and returns the names of those arguments.

    Args:
        schema_string (str): PyTorch operator schema string in the format:
                "op_name(arg1, arg2, *, kwarg1=default1, kwarg2=default2) -> return_type"

    Returns:
        list[str]: List of keyword-only argument names (without defaults or types)

    Examples:
        schema = "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
        ['alpha']

        schema = "aten::clamp(Tensor self, *, Scalar? min=None, Scalar? max=None) -> Tensor"
        ['min', 'max']

        schema = "aten::mm(Tensor self, Tensor mat2) -> Tensor"
        []

        schema = "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *,
                            Scalar beta=1, Scalar alpha=1) -> Tensor"
        ['beta', 'alpha']

    """
    # Extract everything inside parentheses
    inside = re.search(r"\((.*?)\)\s*->", schema_string).group(1)
    # Split by comma and clean spacing
    parts = [p.strip() for p in inside.split(",")]
    # Find the index of "*"
    if "*" in parts:
        parts = parts[parts.index("*") + 1 :]
    else:
        parts = []
    args_with_def_vals = []
    for p in parts:
        name = p.split()[-1]  # last token: alpha=1
        name = name.split("=")[0]  # remove default: alpha
        args_with_def_vals.append(name)
    return args_with_def_vals


def format_python_signature(arguments):
    """
    Convert argument list to Python function signature string.

    Example:
        [{'name': 'self', 'type': 'Tensor'}, {'name': 'mat2', 'type': 'Tensor'}]
        -> "self: torch.Tensor, mat2: torch.Tensor"
    """
    sig_parts = []
    for arg in arguments:
        type_str = convert_cpp_type_to_python(arg["type"])
        # Handle default values if present
        if "default" in arg and arg["default"] is not None and arg["default"] != "":
            default_val = format_default_value(arg["default"])
            sig_parts.append(f"{arg['name']}: {type_str} = {default_val}")
        elif "out" in arg["name"]:
            sig_parts.append(f"{arg['name']}: {type_str} = None")
        else:
            sig_parts.append(f"{arg['name']}: {type_str}")
    return ", ".join(sig_parts)


def format_default_value(default_val):
    """
    Format default value for Python.

    Examples:
        c10::nullopt -> None
        true -> True
        false -> False
        1.0 -> 1.0
    """
    if default_val in ["c10::nullopt", "nullptr", "::std::nullopt"]:
        return "None"
    elif default_val == "true":
        return "True"
    elif default_val == "false":
        return "False"
    else:
        return str(default_val)


def format_python_return_type(returns):
    """
    Convert return type to Python type annotation.

    Example:
        [{'type': 'Tensor'}] -> "torch.Tensor"
        [{'type': 'Tensor'}, {'type': 'Tensor'}] -> "tuple[torch.Tensor, torch.Tensor]"
    """
    if not returns:
        return "None"

    if len(returns) == 1:
        return convert_cpp_type_to_python(returns[0]["type"])

    # Multiple returns - use tuple
    types = [convert_cpp_type_to_python(r["type"]) for r in returns]
    return f"tuple[{', '.join(types)}]"


def convert_cpp_type_to_python(cpp_type):
    """Convert C++ type to Python type annotation."""
    # Remove const, &, * modifiers
    clean_type = (
        cpp_type.replace("at::", "")
        .replace("const", "")
        .replace("&", "")
        .replace("*", "")
        .strip()
    )
    type_mapping = {
        "ITensorListRef": "list[Tensor]",
        "TensorList": "list[Tensor]",
        "Tensor": "torch.Tensor",
        "int64_t": "int",
        "double": "float",
        "bool": "bool",
        "Scalar": "int | float | bool | complex",
        "IntArrayRef": "list[int]",
        "c10::string_view": "str",
        "DimnameList": "list[str]",
        "Dimname": "str",
    }

    for cpp, py in type_mapping.items():
        if cpp in clean_type:
            clean_type = clean_type.replace(cpp, py)

    # Handle optionals
    if "optional" in cpp_type.lower() or "Optional" in cpp_type:
        clean_type = "None"

    return clean_type


def get_argument_names(arguments, schema_string):
    """
    Get comma-separated list of argument names.

    Args:
        arguments: List of argument dicts
        exclude_out: If True, exclude the 'out' parameter

    Returns:
        str: "self, mat2" or "self, mat2, alpha=alpha"
    """
    names = []
    args_with_def_vals = get_args_with_default_vals(schema_string)
    for arg in arguments:
        if arg["name"] == "out":
            continue
        if arg["name"] in args_with_def_vals:
            names.append(f"{arg['name']}={arg['name']}")
        else:
            names.append(arg["name"])
    return ", ".join(names)


def append_scalar_suffix(arg_names: str, scalar_arg_names: List[str]) -> str:
    """
    Append '_scaTensor' suffix to scalar argument names in the arg_names string.

    Args:
        arg_names: Comma-separated string of argument names
        scalar_arg_names: List of scalar argument names

    Returns:
        Modified arg_names string with scalar args suffixed

    Examples:
        append_scalar_suffix("self, other, alpha", ["other"])
        'self, other_scaTensor, alpha'
    """
    # Split arg_names into individual arguments
    args = [arg.strip() for arg in arg_names.split(",")]
    # Modify arguments that are scalars
    modified_args = []
    for arg in args:
        if arg in scalar_arg_names:
            modified_args.append(f"{arg}_scaTensor")
        else:
            modified_args.append(arg)
    # Rejoin with comma-space
    return ", ".join(modified_args)


def enhance_replacement_data(rep_data):
    """
    Add Python-specific fields to replacement data.
    This should be called in generate_replacements() for each operation.
    """
    arguments = rep_data.get("arguments", [])
    returns = rep_data.get("returns", [])
    schema_string = rep_data.get("schema_string", [])

    # Generate Python signature
    rep_data["signature_in"] = format_python_signature(arguments)
    rep_data["signature_out"] = format_python_return_type(returns)

    # Generate argument name lists
    rep_data["scalar_arg_names"] = extract_scalar_arg_names(schema_string)
    rep_data["arg_names"] = get_argument_names(arguments, schema_string)
    rep_data["arg_names"] = append_scalar_suffix(
        rep_data["arg_names"], rep_data["scalar_arg_names"]
    )

    return rep_data


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
    Generates replacement data for PyTorch ops (specified in declaration and schema files)

    Args:
        all_declarations (list): list of dicts parsed from pytorch Declarations.yaml
        all_schemas (list): list of dicts parsed from pytorch RegistrationDeclarations.yaml (indices match with declarations)
        metadata (dict): dict of metadata for each operator (contains template_name and arg_mapping) parsed from Metadata.yaml
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
            "reg_name": f'"{declaration["operator_name"]}.{declaration["overload_name"]}"'
            if declaration["overload_name"]
            else f'"{declaration["operator_name"]}"',
            "torch_prefix": cur_metadata.get("torch_prefix", "torch"),
            "torch_func_name": cur_metadata.get(
                "torch_func_name", declaration["operator_name"]
            ),
        }

        signatures = generate_signature_dict(declaration)
        declaration |= signatures

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
        declaration = enhance_replacement_data(declaration)
        replacements.append(declaration)

    print(f"{num_supported_decs} of {num_total_decs} declarations are supported.")

    return replacements
