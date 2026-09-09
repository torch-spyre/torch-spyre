# Copyright 2026 The Torch-Spyre Authors.
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

from pathlib import Path


def _launch(path, tensors):
    # delayed import, for easy testing
    from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

    # this is the magic line
    # should have already compiled at this point
    runner = SpyreSDSCKernelRunner("spyre-cli", str(path))
    runner.run(*tensors)


dtype_mapping = {
    "fp16": "float16",
    "fp32": "float32",
    "bf16": "bfloat16",
}


def create_tensor_info(tinfo):
    """
    Parses strings of type: "10x1024@fp16".
    """

    # defaults
    dtype = "float16"
    dims = ""

    parts = tinfo.split("@")

    if len(parts) == 1:
        print(f"{tinfo}: No dtype found. Assuming fp16")
        dims = parts[0]
    elif len(parts) > 2:
        raise ValueError(f"Unexpected tensor info: {tinfo}. Expected a single @")
    else:
        dims = parts[0]
        dtype = parts[1]
        if dtype not in list(dtype_mapping.keys()):
            raise ValueError(
                f"Unexpected dtype: {dtype}. Wanted one of: {list(dtype_mapping.keys())}"
            )
        dtype = dtype_mapping[dtype]

    dims = dims.split("x")
    dims = list(filter(lambda x: x != "", dims))

    if len(dims) == 0:
        raise ValueError(f"Found no dimensions in: {tinfo}")

    try:
        dims = list(map(lambda x: int(x), dims))
    except ValueError:
        raise ValueError(f"Found non integer dimension in: {tinfo}")

    return (dims, dtype)


def launch_from_cli(path, inputs, outputs):
    path = Path(path)
    import torch

    tensors = []
    for iarg in inputs:
        shape, dtype = create_tensor_info(iarg)
        tensor = torch.ones(
            shape,
            dtype=getattr(torch, dtype),
            device="spyre",
        )
        tensors.append(tensor)

    for oarg in outputs:
        shape, dtype = create_tensor_info(oarg)
        tensor = torch.empty(
            shape,
            dtype=getattr(torch, dtype),
            device="spyre",
        )
        tensors.append(tensor)

    _launch(path, tensors)

    # TODO: deal with the cases when we have multiple outputs
    # which ops would those be?
    if len(outputs) > 1:
        print("WARNING: multiple outputs found, but only last one printed!")

    print(tensors[-1])


def launch(*tensors, path="."):
    path = Path(path)
    _launch(path, tensors)
