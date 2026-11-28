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

import dataclasses
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
from torch_spyre._inductor.codegen.compute_ops import SymbolKind
from torch_spyre._inductor.codegen.kernel_launchspec import (
    LaunchSpec,
    UnboundSymbol,
    UnresolvedExpression,
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


def _symbol_kinds_from_spec(spec) -> list:
    """The bundle's runtime-slot descriptors, in the form the runner takes.

    They are what ``generate_bundle`` returned for this kernel, so a launch from
    the folder builds the same per-slot payload inductor builds in-process.
    Without them the runtime fills one address per tensor, in order, which only
    fits a kernel whose slots are exactly that: a dimension slot, a pool or a
    per-core address makes the count or the meaning disagree.

    Fields this build does not know are dropped, so a spec written by a newer
    build still loads.
    """
    known = {f.name for f in dataclasses.fields(SymbolKind)}
    return [
        sk
        if isinstance(sk, SymbolKind)
        else SymbolKind(**{k: v for k, v in sk.items() if k in known})
        for sk in spec.symbol_kinds
    ]


def tensors_from_spec(spec, bindings: Optional[dict] = None) -> list:
    """Allocate every tensor ``spec`` says the kernel expects, uninitialised.

    Allocation only: the values are the caller's business. An SDK caller fills
    the inputs however it likes and launches, and the CLI is one such caller --
    :func:`default_tensors` is what gives it torch.ones.

    Each tensor gets the recorded device layout, which is the part that cannot
    be reconstructed from shape and dtype, and the recorded host strides, since
    a graph input or output can be non-contiguous. A layout this build cannot
    rebuild is refused rather than replaced by the default arrangement: a
    tensor of the right shape and dtype in the wrong arrangement launches and
    returns wrong data.

    To fill a tensor whose recorded strides are not contiguous, copy from a
    host tensor with the same strides, ``torch.empty_strided(t.shape,
    t.stride(), dtype=t.dtype)``: the host to device copy does not write a
    non-contiguous destination from a contiguous source.

       The pool tensor, when the kernel takes one, is prepended exactly as
    inductor's ``call_kernel`` does; it is not one of ``spec.args``, so the
    returned list is longer than ``spec.args`` by ``spec.arg_offset``.
    """
    tensors = []
    if spec.caller_passes_pool:
        tensors.append(torch.empty(spec.pool_size, dtype=torch.uint8, device="spyre"))

    for arg in spec.args:
        where = f"arg {arg.arg_index}"
        try:
            shape = arg.resolved_shape(bindings)
            stride = arg.resolved_stride(bindings)
            offset = arg.resolved_storage_offset(bindings)
        except UnboundSymbol as sym:
            raise ValueError(
                f"{where}: dimension '{sym}' is symbolic. Bind it with --bind {sym}=N"
            ) from None
        except UnresolvedExpression as expr:
            raise ValueError(
                f"{where}: dimension '{expr}' was recorded as an expression, so "
                "the spec cannot size this tensor. Allocate it yourself and "
                "launch through the SDK."
            ) from None
        if offset != 0:
            raise ValueError(
                f"{where}: recorded at storage offset {offset}; a tensor rebuilt "
                "from the spec cannot be placed at an offset into a larger "
                "storage. Allocate it yourself and launch through the SDK."
            )
        try:
            layout = spyre_layout(arg.layout)
        except ValueError as e:
            raise ValueError(
                f"{where}: {e}, so the tensor cannot be allocated as the kernel expects"
            ) from None
        tensors.append(_alloc(shape, getattr(torch, arg.dtype), layout, stride))
    return tensors


def default_tensors(spec, tensors: Sequence) -> None:
    """Fill ``spec``'s input tensors with ones, in place.

    Torch.ones, always: this is the smoke-test fill, and it says whether a folder
    loads and runs and nothing about whether it computes.
    A caller that cares about numerics allocates with
    :func:`tensors_from_spec` and writes its own values instead.

    Outputs are left alone. ``input_output`` arguments (an in-place op's
    target) get ones as well: the kernel reads them, so without a value the
    smoke test would compute on uninitialised memory.

    The ones are written by a copy from the host rather than ``fill_``. A
    declared dynamic tensor bound below its maximum is allocated at the
    reservation, so it is a partial view of its storage, and the device
    ``fill_`` refuses partial views because it writes whole sticks. A host copy
    writes the logical extent only, which is also how a caller's own data
    arrives.
    """
    for arg in spec.args:
        if arg.role in ("input", "input_output"):
            tensor = tensors[arg.arg_index + spec.arg_offset]
            # From a host tensor with the destination's strides: the host to
            # device copy does not write a non-contiguous (e.g. transposed)
            # destination from a contiguous source.
            ones = torch.empty_strided(
                tuple(tensor.shape), tuple(tensor.stride()), dtype=tensor.dtype
            )
            ones.fill_(1.0)
            tensor.copy_(ones)


def _alloc(shape, dtype, layout, stride=None):
    """An uninitialised spyre tensor of ``shape`` in ``layout``.

    ``stride`` is the recorded host stride; None (a spec that predates it)
    means contiguous.
    """
    # Unlike the ordinary factories, this goes straight to the allocator and
    # does not bring the runtime up, and it can be our first allocation.
    torch.spyre._impl._lazy_init()
    if stride is None:
        stride = torch.empty(shape, dtype=dtype, device="meta").stride()
    return spyre_empty_with_layout(
        tuple(shape), tuple(stride), dtype, layout, torch.device("spyre")
    )


def spyre_layout(layout):
    """Build a ``SpyreTensorLayout`` from a spec's ``layout`` block.

    Raises ``ValueError`` when the block cannot be turned into a layout: no
    layout recorded, an enum spelling this build does not know, or values the
    constructor rejects. There is no fallback to the default arrangement,
    because a kernel compiled against a particular packing returns wrong data
    for a tensor in any other one.
    """
    if layout is None:
        raise ValueError("no device layout is recorded for it")
    device_dtype = getattr(DataFormats, str(layout.device_dtype), None)
    if not isinstance(device_dtype, DataFormats):
        raise ValueError(
            f"its device dtype {layout.device_dtype!r} is not one this build knows"
        )
    arrangement_name = layout.element_arrangement or "STANDARD"
    arrangement = getattr(ElementArrangement, str(arrangement_name), None)
    if not isinstance(arrangement, ElementArrangement):
        raise ValueError(
            f"its element arrangement {arrangement_name!r} is not one this build knows"
        )
    # valid_elements is a required argument of the only constructor that takes
    # a device_size and stride_map. A spec that predates the field gives {},
    # which the constructor documents as "unknown".
    valid_elements = {
        tuple(int(d) for d in dims): int(count)
        for dims, count in (layout.valid_elements or [])
    }
    try:
        return SpyreTensorLayout(
            [int(d) for d in layout.device_size],
            [int(x) for x in layout.stride_map],
            device_dtype,
            valid_elements,
            arrangement,
        )
    except (TypeError, ValueError, RuntimeError) as e:
        raise ValueError(f"its recorded layout is not valid here ({e})") from None


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


def _layout_mismatch(want, got) -> Optional[str]:
    """How device layout ``got`` differs from the recorded ``want``, or None.

    A tensor whose layout differs is packed differently along the sticks: the
    launch will succeed and return wrong data, which is exactly what the spec
    exists to prevent, so this is reported like any other mismatch. The element
    arrangement counts too: an FP8 tensor in ``STANDARD`` and one in
    ``QFP8CH`` can share device_size and stride_map and still hold different
    bytes.

    Size-1 device dimensions are left out of the comparison. Such a dimension is
    always indexed at 0, so its stride never contributes to an address, and two
    layouts that differ only in them place every element in the same place: a
    keepdim reduction's output is recorded as ``[1, 1, 512, 64]`` while
    ``torch.empty((512, 1))`` gets ``[1, 512, 64]``, and either launches
    correctly.
    """

    def placement(layout) -> list:
        return [
            (int(size), int(stride))
            for size, stride in zip(layout.device_size, layout.stride_map)
            if int(size) != 1
        ]

    if placement(got) != placement(want):

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
    if got.element_arrangement != want.element_arrangement:
        return (
            f"expected element arrangement {_enum_name(want.element_arrangement)}, "
            f"got {_enum_name(got.element_arrangement)} -- the same placement "
            "holds different bytes, so the launch would return wrong data"
        )
    return None


def _stride_mismatch(want_shape, want_stride, want_offset, tensor) -> Optional[str]:
    """How ``tensor``'s host strides or offset differ from the recorded ones.

    Size-1 dimensions are skipped: torch leaves their strides arbitrary, and
    they never contribute to an address.
    """
    if want_stride is not None:

        def significant(stride) -> list:
            return [s for s, n in zip(stride, want_shape) if n != 1]

        got_stride = list(tensor.stride())
        if significant(got_stride) != significant(want_stride):
            return (
                f"expected host strides {want_stride}, got {got_stride} -- "
                "the compiled graph reads and writes this tensor with the "
                "recorded strides"
            )
    if tensor.storage_offset() != want_offset:
        return f"expected storage offset {want_offset}, got {tensor.storage_offset()}"
    return None


def check_tensors_against_spec(
    spec, tensors: Sequence, bindings: Optional[dict] = None
) -> list[str]:
    """Return the ways ``tensors`` disagree with ``spec``; empty means well-formed.

    ``spec`` may be a ``LaunchSpec`` or the dict read off disk. Reports every
    problem rather than the first, so one run tells the caller everything to
    fix. Well-formed is not correct: this checks count, shape, host strides,
    dtype and device layout, and says nothing about the values in the tensors.
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
            want_stride = arg.resolved_stride(bindings)
            want_offset = arg.resolved_storage_offset(bindings)
        except UnboundSymbol as e:
            problems.append(
                f"{where}: dimension '{e}' is symbolic and unbound "
                f"(known symbols: {sorted(spec.symbols) or 'none recorded'})"
            )
            continue
        except UnresolvedExpression as e:
            problems.append(
                f"{where}: dimension '{e}' was recorded as an expression, so "
                "its extent cannot be checked"
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
        else:
            mismatch = _stride_mismatch(want_shape, want_stride, want_offset, tensor)
            if mismatch:
                problems.append(f"{where}: {mismatch}")

        got_layout = device_layout_of(tensor)
        if got_layout is None:
            problems.append(
                f"{where}: not a Spyre tensor, so it has no device layout to "
                "launch with"
            )
            continue

        # The device format is the contract, not the host dtype: float16,
        # bfloat16 and bool are all SEN169_FP16, and one compiled kernel
        # genuinely serves all three -- verified on hardware, where a bfloat16
        # launch reusing a float16-compiled artifact returned exact results.
        # Rejecting a host-dtype difference would refuse tensors the kernel
        # runs correctly, so it warns instead: that is where numerics quietly
        # diverge from what the caller expects.
        got_dtype = str(tensor.dtype).removeprefix("torch.")
        want_device = arg.layout.device_dtype if arg.layout else None
        got_device = _enum_name(got_layout.device_dtype)

        if want_device is None:
            # No layout recorded: the host dtype is the only thing to compare,
            # and the missing layout is reported below.
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

        try:
            want_layout = spyre_layout(arg.layout)
        except ValueError as e:
            problems.append(f"{where}: {e}, so its layout cannot be checked")
            continue
        mismatch = _layout_mismatch(want_layout, got_layout)
        if mismatch:
            problems.append(f"{where}: {mismatch}")

    return problems
