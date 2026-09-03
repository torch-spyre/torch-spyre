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
from spyre_cli.iospec.iospec import parse_json_file


def _launch(path, tensors):
    # delayed import, for easy testing
    import torch

    spyrecode_dir = path / "spyreCodeDir"
    if not spyrecode_dir.exists():
        raise FileNotFoundError(
            f"spyreCodeDir not found at '{spyrecode_dir}'. "
            "Ensure the model has been compiled before launching."
        )

    # this is the magic line
    # should have already compiled at this point
    jobplan = torch.spyre.prepare_kernel(str(spyrecode_dir))

    print(jobplan.expected_input_shapes())
    torch.spyre.launch_jobplan(jobplan, tensors)


def launch_from_iofile(path=".", iofile="io.json"):
    path = Path(path)
    try:
        spec = parse_json_file(iofile)
    except Exception as e:
        raise RuntimeError(f"Failed to parse IO Spec file '{iofile}': {e}") from e

    import torch

    tensors = []
    for tensor_spec in spec.inputs:
        dtype = getattr(torch, tensor_spec.dtype.value)
        tensor = torch.ones(
            tensor_spec.dimensions,
            dtype=dtype,
            device="spyre",
        )
        tensors.append(tensor)

    # Use `torch.empty` for output tensors
    for tensor_spec in spec.outputs:
        dtype = getattr(torch, tensor_spec.dtype.value)
        tensor = torch.empty(
            tensor_spec.dimensions,
            dtype=dtype,
            device="spyre",
        )
        tensors.append(tensor)

    _launch(path, tensors)

    # TODO: deal with the cases when we have multiple outputs
    # which ops would those be?
    if len(spec.outputs) > 1:
        print("WARNING: multiple outputs found, but only last one printed!")

    print(tensors[-1])


def launch(*args, **kwargs):
    path = Path(kwargs.get("path", "."))

    _launch(path, args)
