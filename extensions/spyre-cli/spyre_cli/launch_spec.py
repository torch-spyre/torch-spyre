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


"""Everything the launcher does with a kernel's launch spec.

The spec's schema -- the dataclasses, the JSON read and write -- lives on the
compiler side (``torch_spyre._inductor.codegen.kernel_launchspec``), which
writes the file and so must own its shape. What is here is the consumer half:
reading a spec into tensors, and checking tensors against one. Both need a
*live* tensor and so need ``_C``, which is why they sit on this side; the
producer stays stdlib-only.

TODO: prevent drift risk on LaunchArgLayout
      its fields are serialized by from_device_layout over there and compared here,
      so a new field is silently unchecked.

``torch_spyre`` is imported at module level here. ``pyproject.toml`` does not
list it as an install dependency, but every function in this module needs it,
so deferring the import only moves the failure later without avoiding it.
``core.py`` keeps its imports function-local so ``spyre --help`` and argument
errors still work with no torch-spyre present.
"""

import warnings
from typing import Optional, Sequence

import torch
from torch_spyre._C import (
    DataFormats,
    ElementArrangement,
    SpyreTensorLayout,
    get_spyre_tensor_layout,
    spyre_empty_with_layout,
)
from torch_spyre._inductor.codegen.kernel_launchspec import (
    LaunchSpec,
    UnboundSymbol,
    load_launch_spec,
)


# ---------------------------------------------------------------------------
# Two ways to get tensors a kernel will accept, for the two things a caller
# might already have.
#
# With only the folder: ``load_kernelspec`` reads what it requires and
# ``tensors_from_spec`` allocates to match. Nothing needs checking afterwards,
# since the spec is what they were built from.
#
# With tensors of its own -- loaded from a file, carried over from an eager
# run, or allocated before the spec was consulted -- ``check_tensors_against_spec``
# says whether the kernel will accept them. That is the case worth checking:
# shape and dtype can look right while the packing along the sticks is wrong,
# which launches and returns wrong data.
#
# Problems come back as a list rather than an exception, so a script can print
# them, assert on them, or ignore the ones it knows are harmless.
# ---------------------------------------------------------------------------


def load_kernelspec(path):
    """``path``'s launch spec as a ``LaunchSpec``, or None when it has none.

    The artifact's requirements, for a caller who wants to inspect or satisfy
    them before launching: ``spec.args`` gives each argument's host shape and
    dtype plus the device layout it must have, ``spec.symbols`` the dimensions
    that are only known at launch, and ``spec.caller_passes_pool`` whether a
    pool tensor must be prepended.

    Imported lazily, like ``_launch``: this package does not list torch-spyre
    as an install dependency, though a launch needs it present.
    """
    raw = load_launch_spec(str(path))
    return None if raw is None else LaunchSpec.from_dict(raw)


def tensors_from_spec(spec, bindings: Optional[dict] = None) -> list:
    """Allocate every tensor ``spec`` says the kernel expects, uninitialised.

    Allocation only: the values are the caller's business. An SDK caller fills
    the inputs however it likes and launches, and the CLI is one such caller --
    :func:`default_tensors` is what gives it torch.ones.

    Each tensor gets the recorded device layout, which is the part that cannot
    be reconstructed from shape and dtype.

       The pool tensor, when the kernel takes one, is prepended exactly as
    inductor's ``call_kernel`` does; it is not one of ``spec.args``, so the
    returned list is longer than ``spec.args`` by ``spec.arg_offset``.
    """
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
            _alloc(shape, getattr(torch, arg.dtype), spyre_layout(arg.layout))
        )
    return tensors


def default_tensors(spec, tensors: Sequence) -> None:
    """Fill ``spec``'s input tensors with ones, in place.

    Torch.ones, always: this is the smoke-test fill, and it says whether a folder
    loads and runs and nothing about whether it computes.
    A caller that cares about numerics allocates with
    :func:`tensors_from_spec` and writes its own values instead.

    Outputs are left alone, and so are ``input_output`` arguments, whose
    starting value is part of what is being tested.
    """
    for arg in spec.args:
        if arg.role == "input":
            tensors[arg.arg_index + spec.arg_offset].fill_(1.0)


def _alloc(shape, dtype, layout):
    """An uninitialised spyre tensor of ``shape``, in ``layout`` when given."""
    if layout is None:
        return torch.empty(shape, dtype=dtype, device="spyre")

    # Unlike the ordinary factories, this goes straight to the allocator and
    # does not bring the runtime up, and it can be our first allocation.
    torch.spyre._impl._lazy_init()
    strides = torch.empty(shape, dtype=dtype, device="meta").stride()
    return spyre_empty_with_layout(
        tuple(shape), tuple(strides), dtype, layout, torch.device("spyre")
    )


def spyre_layout(layout):
    """Build a ``SpyreTensorLayout`` from a spec's ``layout`` block.

    Returns None when the block cannot be turned into a layout (an older spec
    missing a field, or an enum spelling this build does not know), so a caller
    can fall back to the default arrangement rather than fail.
    """
    if layout is None:
        return None
    try:
        return SpyreTensorLayout(
            [int(d) for d in layout.device_size],
            [int(x) for x in layout.stride_map],
            getattr(DataFormats, layout.device_dtype),
            getattr(ElementArrangement, layout.element_arrangement or "STANDARD"),
        )
    except (AttributeError, TypeError, ValueError):
        # An enum spelling this build does not know, or a field it predates:
        # fall back to the default arrangement rather than fail the launch.
        return None


def device_layout_of(tensor):
    """``tensor``'s recorded ``SpyreTensorLayout``, or None if it has no device.

    ``get_spyre_tensor_layout`` raises for a CPU tensor rather than returning
    anything, so the None here means "nothing device-level to compare", which
    every caller treats as "fall back" rather than as a mismatch.
    """
    try:
        return get_spyre_tensor_layout(tensor)
    except Exception:  # noqa: BLE001 - a CPU tensor has no device layout to read
        return None


def _enum_name(value) -> str:
    # Peforms operation like DataFormats.SEN169_FP16 -> "SEN169_FP16
    return getattr(value, "name", None) or str(value).rsplit(".", 1)[-1]


def _layout_mismatch(arg, tensor) -> Optional[str]:
    """How ``tensor``'s device layout differs from what ``arg`` records.

    None when they agree, or when the comparison cannot be made. A tensor whose
    layout differs is packed differently along the sticks: the launch will
    succeed and return wrong data, which is exactly what the spec exists to
    prevent, so this is reported like any other mismatch.
    """
    want = spyre_layout(arg.layout)
    if want is None:
        return None
    got = device_layout_of(tensor)
    if got is None:
        return None
    if list(got.device_size) == list(want.device_size) and list(got.stride_map) == list(
        want.stride_map
    ):
        return None

    def describe_layout(layout) -> str:
        return (
            f"device_size={list(layout.device_size)} "
            f"stride_map={list(layout.stride_map)}"
        )

    return (
        f"expected layout {describe_layout(want)}, "
        f"got {describe_layout(got)} -- the tensor is packed differently "
        "along the sticks, so the launch would return wrong data"
    )


def check_tensors_against_spec(
    spec, tensors: Sequence, bindings: Optional[dict] = None
) -> list[str]:
    """Return the ways ``tensors`` disagree with ``spec``; empty means well-formed.

    ``spec`` may be a ``LaunchSpec`` or the dict read off disk. Reports every
    problem rather than the first, so one run tells the caller everything to
    fix. Well-formed is not correct: this checks count, shape, dtype and layout,
    and says nothing about the values in the tensors.
    """
    if not isinstance(spec, LaunchSpec):
        spec = LaunchSpec.from_dict(spec)

    problems: list[str] = []
    offset = spec.arg_offset
    expected_n = len(spec.args) + offset

    if len(tensors) != expected_n:
        note = ""
        if offset:
            note = (
                f" ({len(spec.args)} kernel args + 1 caller-supplied pool "
                f"tensor of {spec.pool_size} bytes)"
            )
        # Positions no longer line up, so per-arg checks would be noise.
        return [f"expected {expected_n} tensors{note}, got {len(tensors)}"]

    if offset:
        pool = tensors[0]
        got_bytes = pool.numel() * pool.element_size()
        if got_bytes != spec.pool_size:
            problems.append(
                f"pool tensor (position 0): expected {spec.pool_size} bytes, "
                f"got {got_bytes}"
            )

    for arg in spec.args:
        tensor = tensors[arg.arg_index + offset]
        where = f"arg {arg.arg_index} ({arg.role})"

        try:
            want_shape = arg.resolved_shape(bindings)
        except UnboundSymbol as e:
            problems.append(
                f"{where}: dimension '{e}' is symbolic and unbound "
                f"(known symbols: {sorted(spec.symbols) or 'none recorded'})"
            )
            continue

        got_shape = list(tensor.shape)
        if got_shape != want_shape:
            hint = ""
            if sorted(got_shape) == sorted(want_shape):
                hint = " -- same extents in a different order (transposed?)"
            problems.append(
                f"{where}: expected shape {want_shape}, got {got_shape}{hint}"
            )

        # The device format is the contract, not the host dtype: float16,
        # bfloat16 and bool are all SEN169_FP16, and one compiled kernel
        # genuinely serves all three -- verified on hardware, where a bfloat16
        # launch reusing a float16-compiled artifact returned exact results.
        # Rejecting a host-dtype difference would refuse tensors the kernel
        # runs correctly, so it warns instead: that is where numerics quietly
        # diverge from what the caller expects.
        got_dtype = str(tensor.dtype).removeprefix("torch.")
        want_device = arg.layout.device_dtype if arg.layout else None
        got_layout = device_layout_of(tensor)
        got_device = _enum_name(got_layout.device_dtype) if got_layout else None

        if want_device is None or got_device is None:
            # A CPU tensor has no device layout, and a spec from a newer build
            # may name a format this one cannot resolve. The host dtype is then
            # the only thing both sides have.
            if got_dtype != arg.dtype:
                problems.append(f"{where}: expected dtype {arg.dtype}, got {got_dtype}")
        elif got_device != want_device:
            problems.append(
                f"{where}: expected device dtype {want_device} "
                f"(compiled for host dtype {arg.dtype}), got {got_device} "
                f"(host dtype {got_dtype})"
            )
        elif got_dtype != arg.dtype:
            warnings.warn(
                f"{where}: host dtype {got_dtype} differs from the recorded "
                f"{arg.dtype}; both are {want_device} on device so the kernel "
                "applies, but check the numerics are what you intend",
                stacklevel=2,
            )

        mismatch = _layout_mismatch(arg, tensor)
        if mismatch:
            problems.append(f"{where}: {mismatch}")

    return problems
