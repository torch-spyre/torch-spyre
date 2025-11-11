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
import torch
from torch.testing._internal.opinfo.core import (
    UnaryUfuncInfo,
    ShapeFuncInfo,
    ReductionOpInfo,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.fx.experimental.symbolic_shapes import (
    ShapeEnv,
    StatelessSymbolicContext,
    DimDynamic,
)
from torch._subclasses.fake_tensor import FakeTensorMode
import logging

logger = logging.getLogger(__name__)


def create_fake_tensor_with_dynamic_size(x, fake_mode):
    # Creates a fake tensor from given tensor x
    with fake_mode:
        return fake_mode.from_tensor(
            x,
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC for _ in range(x.dim())],
                dynamic_strides=[DimDynamic.DYNAMIC for _ in range(x.dim())],
            ),
        )


def make_sym_data_str(x):
    # Utility for symbolic expressions for easier parsing
    # Adds 'c' before constant dims and adds '_' after each symbolic dim, e.g.: [s10+1, s1*1] -> [s10_+c1_, s1_*c1_]
    out_x = []
    for s in x:
        expr = str(s)
        expr = re.sub(r"(?<![a-zA-Z0-9_.-])(\d+)(?!\.\d\d[a-zA-Z0-9_])", r"c\1", expr)
        expr = re.sub(r"\b(s\d+|c\d+)\b", r"\1_", expr)
        out_x.append(expr)
    return out_x


# TODO: Can ShapeEnv be used instead of manual memo to track shapes?
def fakeify_inputs_and_fill_memo(fake_mode, declaration, sample_inputs, memo):
    # Fakeifies the tensors provided in sample_inputs
    # and fills the memo dict that holds the relation between symbolic dim and initial owner
    # (e.g.: 's0': 'self.sizes()[0]')
    for i in range(len(sample_inputs)):
        if isinstance(sample_inputs[i], torch.Tensor):
            sample_inputs[i] = create_fake_tensor_with_dynamic_size(
                sample_inputs[i], fake_mode
            )
            shapes = make_sym_data_str(sample_inputs[i].shape)
            strides = make_sym_data_str(sample_inputs[i].stride())

            for k in range(len(shapes)):
                if shapes[k] not in memo:
                    memo[shapes[k]] = (
                        f"{declaration['arguments'][i]['name']}.sizes()[{-len(shapes) + k}]"
                    )
                if strides[k] not in memo:
                    memo[strides[k]] = (
                        f"{declaration['arguments'][i]['name']}.strides()[{-len(shapes) + k}]"
                    )

        elif isinstance(sample_inputs[i], list):
            for j in range(len(sample_inputs[i])):
                if isinstance(sample_inputs[i][j], torch.Tensor):
                    sample_inputs[i][j] = create_fake_tensor_with_dynamic_size(
                        sample_inputs[i][j], fake_mode
                    )
                    shapes = make_sym_data_str(sample_inputs[i][j].shape)
                    strides = make_sym_data_str(sample_inputs[i][j].stride())

                    for k in range(len(shapes)):
                        if shapes[k] not in memo:
                            memo[shapes[k]] = (
                                f"{declaration['arguments'][i]['name']}[{j}].sizes()[{-len(shapes) + k}]"
                            )
                        if strides[k] not in memo:
                            memo[strides[k]] = (
                                f"{declaration['arguments'][i]['name']}[{j}].strides()[{-len(shapes) + k}]"
                            )


def generate_sample_inputs(declaration):
    # Generates sample inputs for a given declaration - note: very naive implementation used as fallback for OpInfo.sample_inputs()
    sample_inputs = []
    for i, arg in enumerate(declaration["arguments"]):
        if "default" in arg or arg["sendnn_type"] == "Ignore":
            continue
        if "ArrayRef" in arg["type"] and i > 0:
            inp = [1, 1]
        elif "TensorList" in arg["type"]:
            inp = [torch.empty(5, 5, 5), torch.empty(5, 5, 5)]
        elif "Tensor" in arg["type"]:
            inp = torch.empty(5, 5, 5)
        elif (
            any(t in arg["type"] for t in ["int", "double", "float", "Scalar"])
            and i > 0
        ):
            inp = 1
        elif "bool" in arg["type"] and i > 0:
            inp = True
        else:
            print(
                f"Sample input generation for argument type {arg['type']} in declaration {declaration['name']} is not implemented."
            )
            return None
        sample_inputs.append(inp)
    return sample_inputs


def generate_outputs_and_memo(declaration, auto=True):
    # For a given declaration dict, returns list of outputs storing fake tensors, memo dict to map symbol dims
    # and bypass_flag (if true, we will directly use the first Tensor input to infer output shape)

    op_handle = getattr(torch.ops.aten, declaration["operator_name"])
    if declaration["overload_name"]:
        try:
            op_handle = getattr(op_handle, declaration["overload_name"])
        except AttributeError:
            logger.debug(
                f"Attribute '{declaration['overload_name']}' not found on {op_handle}"
            )

    op_info = [op for op in op_db if op.name == declaration["name"]]
    memo = {}
    bypass_flag = True

    if op_info and auto:
        op_info = op_info[0]
        if isinstance(op_info, UnaryUfuncInfo):
            return [], memo, bypass_flag, False
        if isinstance(op_info, ShapeFuncInfo):
            return None, memo, bypass_flag, False
        if isinstance(op_info, ReductionOpInfo):
            return None, memo, bypass_flag, False

        # TODO: testing utils can be used to generate sample inputs but does not work for every overload because
        # I can't control which overload it should sample inputs for.
        sample_inputs = list(op_info.sample_inputs(device="cpu", dtype=torch.float32))
        sample_input = None
        for s in sample_inputs:
            if isinstance(s.input, torch.Tensor) and s.input.numel() > 1:
                sample_input = s
                break
        if sample_input is None:
            return None, memo, bypass_flag, True
        else:
            sample_input = [
                sample_input.input,
                *sample_input.args[: len(declaration["arguments"]) - 1],
            ]
    else:
        print(f"Generating sample input manually for {declaration['name']}.")
        sample_input = generate_sample_inputs(declaration)
        if sample_input is None:
            return None, memo, bypass_flag, False

    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env)

    fakeify_inputs_and_fill_memo(fake_mode, declaration, sample_input, memo)

    with fake_mode:
        try:
            outputs = op_handle(*sample_input)
        except Exception as e:
            logger.debug(
                f"Warning at {declaration['operator_name']}.{declaration['overload_name']}: {e}"
            )
            return None, memo, bypass_flag, True

    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    if isinstance(outputs[0], torch.Tensor):
        if isinstance(sample_input[0], torch.Tensor):
            # if output and first input has the same shape, the rest of the code will bypass
            # inferred shapes for this declaration
            bypass_flag = (sample_input[0].shape == outputs[0].shape) and (
                sample_input[0].stride() == outputs[0].stride()
            )
        else:
            bypass_flag = False
    else:
        return (
            None,
            memo,
            bypass_flag,
            False,
        )  # TODO: handle non-tensor first input arguments and outputs
    return outputs, memo, bypass_flag, False


def infer_output_shape_stride(declaration):
    # For a given declaration dict, returns the list of expressions for the output shapes and
    # and bypass_flag (if true, we will directly use the first argument to infer output shape)

    # If out is provided or declaration is inplace, no need to do output shape inference
    if (
        "out" in declaration["overload_name"] or declaration["inplace"]
    ) and "Tensor" in declaration["arguments"][0]["type"]:
        return [], True

    # Try auto sample generation (using testing utils) first, if it does not work try with manually generated sample inputs
    outputs, memo, bypass_flag, force_flag = generate_outputs_and_memo(
        declaration, auto=True
    )
    if outputs is None and force_flag:
        outputs, memo, bypass_flag, _ = generate_outputs_and_memo(
            declaration, auto=False
        )

    # If we have exited here, it means that no shape inference logic will be used in the template either because
    # we failed to infer the output shape (code won't generate)
    if outputs is None:
        logger.debug(
            f"Sample input generation for shape inference of {declaration['operator_name']}.{declaration['overload_name']} has failed."
        )
        return None, True
    # or, if the op is unary or output and first input are identical in shape, we can directly use input shape
    if bypass_flag:
        return [], bypass_flag

    output_expr_list = []

    # TODO: Can ShapeEnv or FakeMode utils be used?
    for output in outputs:
        expr = {}

        if isinstance(output, torch.Tensor):
            output_shape = []
            output_stride = []
            shapes = make_sym_data_str(output.shape)
            strides = make_sym_data_str(output.stride())
            for i in range(len(shapes)):
                shape = shapes[i]
                stride = strides[i]
                for k in memo:
                    shape = shape.replace(k, memo[k])
                    stride = stride.replace(k, memo[k])
                output_shape.append(shape)
                output_stride.append(shape)

            expr["shape"] = output_shape
            expr["stride"] = output_stride
        else:
            return None, True  # TODO: handle non-tensor outputs

        output_expr_list.append(expr)

    return output_expr_list, False
