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
    """Launch the bundle and return the runner.

    The returned runner MUST be kept alive until the outputs have been read
    back. Dropping it frees the JobPlan, and with it the device allocation and
    pinned buffers that the still in-flight launch is reading.
    """
    # delayed import, for easy testing
    from torch_spyre.execution.kernel_runner import SpyreSDSCKernelRunner

    # this is the magic line
    # should have already compiled at this point
    runner = SpyreSDSCKernelRunner("spyre-cli", str(path))
    runner.run(*tensors)
    return runner


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


def _load_spec(path):
    """``path``'s launch spec as a ``LaunchSpec``, or None when it has none.

    Imported lazily, like ``_launch``: this package does not list torch-spyre
    as an install dependency, though a launch needs it present.
    """
    from torch_spyre._inductor.codegen.kernel_launchspec import (
        LaunchSpec,
        load_launch_spec,
    )

    raw = load_launch_spec(str(path))
    return None if raw is None else LaunchSpec.from_dict(raw)


def _tensors_from_spec(spec, bindings):
    """Build every tensor ``spec`` says the kernel expects.

    Inputs are filled with ones and outputs left uninitialised, matching the
    explicit path. The pool tensor, when the kernel takes one, is prepended
    exactly as inductor's ``call_kernel`` does -- it is not one of ``args``.

    Each tensor is allocated with the recorded device layout. Shape and dtype
    alone are not the contract: an op can be compiled against a particular
    packing along the sticks -- a depthwise conv2d puts its channels in one
    stick -- and a tensor in the default arrangement then has the right shape
    and dtype while being laid out wrongly, which launches and returns wrong
    data.
    """
    import torch
    from torch_spyre._inductor.codegen.kernel_launchspec import (
        UnboundSymbol,
        spyre_layout,
    )

    tensors = []
    if spec.caller_passes_pool:
        tensors.append(torch.empty(spec.pool_size, dtype=torch.uint8, device="spyre"))

    for arg in spec.args:
        try:
            shape = arg.resolved_shape(bindings)
        except UnboundSymbol as sym:
            raise ValueError(
                f"arg {arg.arg_index}: dimension '{sym}' is symbolic. "
                f"Bind it with --bind {sym}=N"
            ) from None
        tensors.append(
            _alloc(torch, shape, getattr(torch, arg.dtype), spyre_layout(arg.layout))
        )
        if arg.role == "input":
            tensors[-1].fill_(1.0)
    return tensors


def _alloc(torch, shape, dtype, layout):
    """An uninitialised spyre tensor of ``shape``, in ``layout`` when given."""
    if layout is None:
        return torch.empty(shape, dtype=dtype, device="spyre")

    from torch_spyre._C import spyre_empty_with_layout

    # Unlike the ordinary factories, this goes straight to the allocator and
    # does not bring the runtime up, and it can be our first allocation.
    torch.spyre._impl._lazy_init()
    strides = torch.empty(shape, dtype=dtype, device="meta").stride()
    return spyre_empty_with_layout(
        tuple(shape), tuple(strides), dtype, layout, torch.device("spyre")
    )


def launch_from_cli(path, inputs, outputs, bindings=None):
    path = Path(path)
    import torch

    bindings = bindings or {}
    spec = _load_spec(path)

    if spec is None and not inputs and not outputs:
        # Nothing to launch with: no spec to read the arguments from, and none
        # given. Caught here rather than in the CLI, which cannot see whether
        # the folder has a spec. Launching anyway would pass zero tensors and
        # fail deep in the runtime with an argument-count error.
        raise ValueError(
            f"{path} has no launch spec, so the tensors must be described: "
            "pass -i/-o, or recompile to get a folder that describes itself."
        )

    if spec is not None and not inputs and not outputs:
        # The folder describes itself, so there is nothing to retype.
        tensors = _tensors_from_spec(spec, bindings)
        # By role, not by position: an output need not be a trailing argument,
        # and an in-place arg is both. ``arg_offset`` skips the pool, which sits
        # at 0 and is not one of ``args``.
        out_positions = [
            arg.arg_index + spec.arg_offset for arg in spec.args if arg.role != "input"
        ]
        print(f"using {spec.kernel_name} launch spec from {path}")
    else:
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
        # Explicit form appends outputs after inputs, so they are the tail.
        out_positions = list(range(len(inputs), len(tensors)))

        if spec is not None:
            # Explicit arguments AND a spec: check them rather than trusting
            # them. This is the case that used to launch and return wrong data.
            from torch_spyre._inductor.codegen.kernel_launchspec import (
                check_launch_spec,
            )

            problems = check_launch_spec(spec, tensors, bindings)
            if problems:
                raise ValueError(
                    "the tensors given do not match what this kernel expects:\n"
                    + "\n".join(f"  - {p}" for p in problems)
                    + "\n\nOmit -i/-o to build them from the spec instead."
                )

    runner = _launch(path, tensors)

    # Every output, not just the last: reading them all back is also what makes
    # a failed launch visible, since .cpu() is where a dead kernel surfaces.
    if not out_positions:
        out_positions = [len(tensors) - 1]
    for pos in out_positions:
        if len(out_positions) > 1:
            print(f"output (arg {pos}):")
        print(tensors[pos].cpu())
    del runner


def launch(*tensors, path="."):
    """Launch a bundle; returns the runner, which the caller MUST keep alive.

    Bind the result (``runner = launch(a, b, c)``) and keep it in scope until
    the outputs have been read back. Calling this as a bare statement frees the
    JobPlan while the launch is still in flight, and the device then faults
    with what looks like a hardware error.
    """
    path = Path(path)
    return _launch(path, tensors)
