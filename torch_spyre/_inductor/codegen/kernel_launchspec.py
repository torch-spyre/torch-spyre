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

"""Producer for the launch spec that `spyre launch` consumes.

The compiler writes `launch_spec.json` into a compiled kernel's
`spyreCodeDir/`, so that folder -- the one the runtime loads -- also states
what it expects to be called with. The consumer is the spyre-cli extension
(`extensions/spyre-cli`), which reads it to build or validate the tensors a
launch needs. Schema and rationale: RFC 4755 (torch-spyre/RFCs#43).
"""

import dataclasses
import json
import os
from collections.abc import Sequence
from typing import Any, Optional

LAUNCH_SPEC_FILE = "launch_spec.json"
LAUNCH_SPEC_VERSION = 1

# this is not defined anywhere rather just used by name which has a chance
# of diverging from launchspec implementation.
# TODO: define a common SPYRECODE_DIR somewhere which compiler uses
SPYRECODE_DIR = "spyreCodeDir"


# ---------------------------------------------------------------------------
# What the producer builds. These exist so the compile side is typed rather
# than assembling dicts by hand; the wire format is still plain JSON, and the
# reader stays dict-based (see the module docstring).
#
# They reach the compile step through generated wrapper source via repr(), the
# same way OpSpec and TensorArg do, so ``wrapper.py`` imports them into that
# namespace alongside those. Keep both reprs eval-able: no computed defaults,
# no types that do not round-trip through repr().
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LaunchArgLayout:
    """The device arrangement one argument's tensor must have.

    Part of the execution contract, not a diagnostic: an op can be compiled
    against a particular packing along the sticks, and a tensor in the default
    arrangement then has the right shape and dtype while being laid out wrongly.
    """

    device_size: list[int]
    stride_map: list[int]
    device_dtype: str
    element_arrangement: str

    @classmethod
    def from_device_layout(cls, device_layout) -> "LaunchArgLayout":
        """Build from a ``SpyreTensorLayout``.

        The two enums are stored by name: a spec is read by a process that may
        not have our types, so it carries the spelling rather than the value.
        """
        return cls(
            device_size=[int(d) for d in device_layout.device_size],
            stride_map=[int(s) for s in device_layout.stride_map],
            device_dtype=_enum_name(device_layout.device_dtype),
            element_arrangement=_enum_name(device_layout.element_arrangement),
        )


@dataclasses.dataclass
class LaunchArg:
    """One positional argument of a compiled kernel.

    ``arg_index`` *is* the contract: binding is positional, which is why there
    is deliberately no per-argument name. ``shape`` holds ints, or a symbol's
    name for a dimension only known at launch.
    """

    arg_index: int
    role: str
    shape: list[Any]
    dtype: str
    layout: LaunchArgLayout


def _enum_name(value) -> str:
    """``DataFormats.SEN169_FP16`` -> ``"SEN169_FP16"``."""
    return getattr(value, "name", None) or str(value).rsplit(".", 1)[-1]


@dataclasses.dataclass
class LaunchSpec:
    """One compiled folder's whole contract.

    ``pool_size`` is the size of the pool tensor the *caller* must pass, 0 when
    it passes none -- not the kernel's own pool extent, since a bundle can
    allocate scratch internally and so have a pool with no parameter for one.
    ``for_kernel`` takes the resolved value.

    Unlike ``LaunchArg`` this never crosses the generated-wrapper boundary, so
    nothing here has to stay repr-eval-able.
    """

    kernel_name: str
    args: list[LaunchArg]
    pool_size: int = 0
    bundle_symbolic_args: bool = True
    emitter: Optional[str] = None
    symbol_kinds: list = dataclasses.field(default_factory=list)
    symbols: dict = dataclasses.field(default_factory=dict)
    version: int = LAUNCH_SPEC_VERSION

    @classmethod
    def for_kernel(
        cls,
        kernel_name: str,
        args: Sequence[LaunchArg],
        *,
        caller_pool_size: int,
        bundle_symbolic_args: bool,
        emitter: Optional[str] = None,
        symbol_kinds: Optional[Sequence] = None,
    ) -> "LaunchSpec":
        """Build a spec, deriving the symbol table from ``args``.

        A dimension recorded as a name rather than an extent is symbolic and
        must be bound at launch, so each one gets an entry -- empty until the
        producer can supply a range.
        """
        args = list(args)
        return cls(
            kernel_name=kernel_name,
            args=args,
            pool_size=caller_pool_size,
            bundle_symbolic_args=bool(bundle_symbolic_args),
            emitter=emitter,
            symbol_kinds=list(symbol_kinds or []),
            symbols={
                dim: {} for arg in args for dim in arg.shape if not isinstance(dim, int)
            },
        )

    def to_dict(self) -> dict:
        """The JSON object written to disk.

        A field carrying nothing is left out rather than written empty, so an
        older reader sees an absent key instead of a meaningless one.
        """
        out: dict[str, Any] = {
            "version": self.version,
            "kernel_name": self.kernel_name,
            "pool_size": self.pool_size,
            "bundle_symbolic_args": self.bundle_symbolic_args,
            "args": launch_args_to_dicts(self.args),
        }
        if self.emitter is not None:
            out["emitter"] = self.emitter
        if self.symbol_kinds:
            out["symbol_kinds"] = [dataclasses.asdict(sk) for sk in self.symbol_kinds]
        if self.symbols:
            out["symbols"] = self.symbols
        return out


def launch_args_to_dicts(args: Sequence[LaunchArg]) -> list[dict]:
    """``LaunchArg`` records as the plain dicts the spec is written from.

    ``json`` cannot serialize a dataclass, and the reader does not import our
    types, so the conversion happens here -- the same boundary
    ``save_symbol_kinds`` crosses with ``dataclasses.asdict``.
    """
    return [dataclasses.asdict(a) for a in args]


def spec_path(code_dir: str) -> str:
    return os.path.join(code_dir, SPYRECODE_DIR, LAUNCH_SPEC_FILE)


def save_launch_spec(compile_dir: str, spec: dict) -> None:
    """Write the launch spec into ``compile_dir``'s ``spyreCodeDir/``.

    Creates that directory when it is absent. On the SDSC path this runs right
    after ``generate_bundle``, before the backend compiler has made it; on the
    KTIR path the compiler has already run and it exists. Either way the spec
    ends up in the same place, so a launcher looks in one spot whichever emitter
    produced the folder.

    Pre-creating the directory is safe: the backend compiler writes
    ``spyrecode.json`` and ``init_binary.bin`` into it and leaves anything else
    alone, and the caller's success check tests for ``spyrecode.json`` by name
    rather than for the directory, so it still catches a compiler that produced
    nothing.
    """
    path = spec_path(compile_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)


def load_launch_spec(code_dir: str) -> Optional[dict]:
    """Read ``code_dir``'s launch spec, or None when there is none.

    ``code_dir`` may be either the folder holding ``spyreCodeDir/`` -- what a
    runner is given -- or that directory itself, since the folder is meant to
    stand alone. Both are tried.

    Absent is not an error: folders compiled before this existed have no spec,
    and callers fall back to their old behaviour. A spec that is present but
    unreadable, or newer than this build, *is* an error -- ignoring it would put
    the caller back to guessing, which is what the spec exists to stop.
    """
    for path in (spec_path(code_dir), os.path.join(code_dir, LAUNCH_SPEC_FILE)):
        if os.path.isfile(path):
            break
    else:
        return None
    try:
        with open(path) as f:
            spec = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"could not read launch spec at {path}: {e}") from e

    version = spec.get("version")
    if version is None:
        raise RuntimeError(f"launch spec at {path} has no 'version'")
    if version > LAUNCH_SPEC_VERSION:
        raise RuntimeError(
            f"launch spec at {path} is version {version}, newer than this build "
            f"understands ({LAUNCH_SPEC_VERSION}). Upgrade torch-spyre to launch "
            "this folder."
        )
    return spec


def spyre_layout_from_spec(layout: dict):
    """Build a ``SpyreTensorLayout`` from a spec's ``layout`` block.

    Returns None when the block cannot be turned into a layout (an older spec
    missing a field, or an enum spelling this build does not know), so a caller
    can fall back to the default arrangement rather than fail.
    """
    from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout

    try:
        device_size = [int(d) for d in layout["device_size"]]
        stride_map = [int(x) for x in layout["stride_map"]]
        device_dtype = getattr(DataFormats, layout["device_dtype"])
        arrangement = getattr(
            ElementArrangement, layout.get("element_arrangement", "STANDARD")
        )
    except (KeyError, TypeError, AttributeError, ValueError):
        return None
    return SpyreTensorLayout(device_size, stride_map, device_dtype, arrangement)


def _layout_mismatch(arg: dict, tensor) -> Optional[str]:
    """How ``tensor``'s device layout differs from what ``arg`` records.

    None when they agree, or when the comparison cannot be made. A tensor whose
    layout differs is packed differently along the sticks: the launch will
    succeed and return wrong data, which is exactly what the spec exists to
    prevent, so this is reported like any other mismatch.
    """
    spec_layout = arg.get("layout")
    if not spec_layout:
        return None
    want = spyre_layout_from_spec(spec_layout)
    if want is None:
        return None
    try:
        from torch_spyre._C import get_spyre_tensor_layout

        got = get_spyre_tensor_layout(tensor)
    except Exception:  # noqa: BLE001 - a CPU tensor has no device layout to read
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


def check_launch_spec(
    spec: dict, tensors: Sequence, bindings: Optional[dict] = None
) -> list[str]:
    """Return the ways ``tensors`` disagree with ``spec``; empty means well-formed.

    Reports every problem rather than the first, so one run tells the caller
    everything to fix. Well-formed is not correct: this checks count, shape and
    dtype, and says nothing about the values in the tensors.
    """
    problems: list[str] = []
    args = sorted(spec["args"], key=lambda a: a["arg_index"])
    pool_size = spec.get("pool_size", 0)
    offset = 1 if pool_size > 0 else 0

    expected_n = len(args) + offset
    if len(tensors) != expected_n:
        note = ""
        if offset:
            note = (
                f" ({len(args)} kernel args + 1 caller-supplied pool tensor of "
                f"{pool_size} bytes)"
            )
        # Positions no longer line up, so per-arg checks would be noise.
        return [f"expected {expected_n} tensors{note}, got {len(tensors)}"]

    if offset:
        pool = tensors[0]
        got_bytes = pool.numel() * pool.element_size()
        if got_bytes != pool_size:
            problems.append(
                f"pool tensor (position 0): expected {pool_size} bytes, "
                f"got {got_bytes}"
            )

    symbols = spec.get("symbols", {})
    bindings = bindings or {}
    for arg in args:
        tensor = tensors[arg["arg_index"] + offset]
        where = f"arg {arg['arg_index']} ({arg['role']})"

        want_shape = []
        unbound = False
        for dim in arg["shape"]:
            if isinstance(dim, int):
                want_shape.append(dim)
            elif dim in bindings:
                want_shape.append(int(bindings[dim]))
            else:
                problems.append(
                    f"{where}: dimension '{dim}' is symbolic and unbound "
                    f"(known symbols: {sorted(symbols) or 'none recorded'})"
                )
                unbound = True
        if unbound:
            continue

        got_shape = list(tensor.shape)
        if got_shape != want_shape:
            hint = ""
            if sorted(got_shape) == sorted(want_shape):
                hint = " -- same extents in a different order (transposed?)"
            problems.append(
                f"{where}: expected shape {want_shape}, got {got_shape}{hint}"
            )

        got_dtype = str(tensor.dtype).removeprefix("torch.")
        if got_dtype != arg["dtype"]:
            problems.append(f"{where}: expected dtype {arg['dtype']}, got {got_dtype}")

        mismatch = _layout_mismatch(arg, tensor)
        if mismatch:
            problems.append(f"{where}: {mismatch}")

    return problems
