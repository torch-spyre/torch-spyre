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


from __future__ import annotations

from collections import OrderedDict
import dataclasses
import threading
from typing import Any, Literal, Sequence

from sympy import Symbol, Expr, Function, sympify
from torch_spyre._C import DataFormats, ElementArrangement
import torch
from torch_spyre import _C

from .constants import IDENTITY_OP


class IndirectAccess(Function):
    """Sympy function: IndirectAccess(tensor_name) — runtime index read from that tensor at the current iteration point.

    Used in TensorArg.device_coordinates to encode indirect access: as a coordinate of an
    input arg (gather) or an output arg (scatter).
    IndexedBase was not used because sympify('arg1_1[i]') fails: the parser reconstructs
    arg1_1 as a Symbol, and Symbol.__getitem__ raises TypeError.
    """

    @classmethod
    def eval(cls, name):  # noqa: ARG003
        return None  # keep unevaluated


# --- Source-to-kernel provenance schema -------------------------------------
# These dataclasses live here with the other IR-op schema types; the logic that
# builds them from Inductor IR lives in ``provenance.py``. They serialize into
# the current OpSpec/SuperDSC JSON path, with field names lined up to MLIR
# location attributes so the future KTIR (MLIR) migration is a low-friction
# serializer change rather than a redesign.


@dataclasses.dataclass(frozen=True)
class SourceLoc:
    """Structured source location attached to provenance handles.

    Serialized into the current OpSpec/SuperDSC JSON path; the field names
    mirror MLIR ``FileLineColRange`` (start/end line and column) so a future
    KTIR (MLIR) migration maps 1:1 rather than requiring a reshape.
    """

    file: str
    start_line: int
    start_col: int = 0
    end_line: int | None = None
    end_col: int | None = None

    def to_str(self) -> str:
        return f"{self.file}:{self.start_line}:{self.start_col}"

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


ProvenanceTransformKind = Literal[
    "rewrite", "fusion", "decomposition", "clone", "remap"
]


@dataclasses.dataclass(frozen=True)
class ProvenanceTransform:
    """One structured lower-IR transformation in a provenance history.

    ``kind`` identifies the rewrite shape (for example ``fusion`` or
    ``decomposition``); ``pass_name`` and optional ``reason`` are deliberately
    separate so the record maps cleanly to MLIR location metadata.

    Histories are immutable tuples so reconstructed buffers cannot accidentally
    share and mutate provenance state.
    """

    kind: ProvenanceTransformKind
    pass_name: str
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "pass_name": self.pass_name,
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True)
class DebugHandle:
    """Source-to-kernel provenance handle.

    Nestable to map onto MLIR locations: ``NameLoc(aten_op) -> SourceLoc``,
    ``fused_from -> FusedLoc``, and ``ir_chain -> CallSiteLoc`` lineage.
    ``transform_history`` retains structured lower-IR rewrite metadata rather
    than overloading one scalar fusion label.

    A ``None`` ``source`` or ``aten_op`` is a *normal, expected* value, not a
    missing-data error: when an op fuses origins from several distinct source
    lines there is no single honest headline, so both are set to ``None`` and the
    full set is preserved in ``fused_from``. Consumers should fall back to
    ``fused_from`` rather than treating a null headline as an error.
    """

    id: int
    source: SourceLoc | None
    aten_op: str | None
    ir_chain: tuple[str, ...]
    fused_from: tuple["DebugHandle", ...] = ()
    transform_history: tuple[ProvenanceTransform, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            # id is serialized as a string: a 63-bit value exceeds JS
            # Number.MAX_SAFE_INTEGER (2**53-1), and JSON.parse would round it to
            # float64 before a consumer could act. The dataclass field stays int
            # for the MLIR/protobuf mapping (a separate serializer).
            "id": str(self.id),
            "source": self.source.to_dict() if self.source is not None else None,
            "aten_op": self.aten_op,
            "ir_chain": list(self.ir_chain),
            "fused_from": [h.to_dict() for h in self.fused_from],
            "transform_history": [t.to_dict() for t in self.transform_history],
        }


@dataclasses.dataclass(frozen=True)
class TensorWorkDivision:
    """Tensor ownership expressed in operation-loop symbols.

    Both mappings have the same symbol keys. ``work_slices`` gives each loop's
    split count and ``core_id_to_work_slice`` maps the free symbol ``core_id``
    to that loop's owned slice.
    """

    work_slices: dict[Symbol, int]
    core_id_to_work_slice: dict[Symbol, Expr]

    def remap_symbols(self, symbol_mapping: dict[Symbol, Symbol]) -> TensorWorkDivision:
        """Return this ownership expressed in remapped loop symbols."""

        return TensorWorkDivision(
            {
                symbol_mapping.get(dim, dim): split
                for dim, split in self.work_slices.items()
            },
            {
                symbol_mapping.get(dim, dim): sympify(slot).xreplace(symbol_mapping)
                for dim, slot in self.core_id_to_work_slice.items()
            },
        )

    def to_core_slices(self, num_cores: int) -> dict[str, dict[str, int]]:
        """Expand symbolic ownership into DeepTools' per-core slice map."""

        core_id = Symbol("core_id")
        result = {}
        for core in range(num_cores):
            slots = {}
            for dim, expression in self.core_id_to_work_slice.items():
                slot = int(sympify(expression).subs(core_id, core))
                split = self.work_slices[dim]
                if not 0 <= slot < split:
                    raise ValueError(
                        f"core {core} owns invalid {dim} slot {slot} for split {split}"
                    )
                slots[str(dim)] = slot
            result[str(core)] = slots
        return result


@dataclasses.dataclass
class TensorArg:
    """
    A class representing a Tensor argument to an OpSpec

    Attributes:
        is_input: Is the Tensor used as an input to the operation?
        arg_index: The index of the Tensor in the argument array of the Kernel.
        device_dtype: The device dtype of the tensor elements.
        device_size: The device size (as per SpyreTensorLayout) of the Tensor
        device_coordinates: The sympy Exprs that describe how elements in the Tensor are accessed.
                Free variables in device_coordinates refer to entries in the OpSpec's iteration_space.
        allocation: dict tagging where this Tensor's data lives. Mirrors
                layout.allocation and carries exactly one of three
                mutually-exclusive keys:
                - "hbm": graph input/output or fallback-kernel input/output,
                  addressed directly in HBM.
                - "lx": placed in on-chip LX scratchpad by LX planning
                  (scratchpad/allocator.py).
                - "hbm_pool": intermediate that didn't fit in LX, bump-
                  allocated into the off-chip HBM intermediates segment by
                  hbm_pool_planning.py. See
                  docs/source/compiler/hbm_pool_planning.md.
        device_tile_advance_expr: This arg's own device-*element*-offset sympy.Expr for one
            unit step of each tiled Inductor iteration symbol (d0, d1, ...), built by
            SpyreKernel._general_tile_advance from CoarseTileInfo.tiled_dims_per_read /
            output_tiled_dims's per-level (dim, extent) decisions: one term per nesting
            level, substituted with that level's own minted symbol, reprojected to
            device-element space via views.tiling_expr_to_device_expr, and summed into a
            single combined Expr. This is the sole tile-advance mechanism. ``None`` for
            ops without loop_info/coarse tiling.
        work_division: Optional tensor-specific ownership used when it differs
            from the operation's work division.
    """

    is_input: bool
    arg_index: int
    device_dtype: DataFormats
    device_size: list[int]
    device_coordinates: list[Expr]
    allocation: dict[str, Any]
    name: str | None = None
    device_tile_advance_expr: Expr | None = None
    element_arrangement: ElementArrangement = dataclasses.field(
        default_factory=lambda: ElementArrangement.STANDARD
    )
    work_division: TensorWorkDivision | None = None


def is_lx_relayout_identity(op: str, args: Sequence[TensorArg]) -> bool:
    """An LX identity whose input and output have different owners."""

    if op != IDENTITY_OP or len(args) != 2:
        return False
    source, destination = args
    return (
        "lx" in source.allocation
        and "lx" in destination.allocation
        and source.work_division is not None
        and destination.work_division is not None
        and source.work_division != destination.work_division
    )


@dataclasses.dataclass
class OpSpec:
    """
    A class representing a single operation to perform on the device

    Attributes:
        op: The name of the operation.
        is_reduction: Is the operation a reduction?
        iteration_space: The iteration space of the operation. The values are tuples of (range, work_division).
        args: The input and output arguments to the operation.
        op_info: A dictionary of auxiliary information whose content is operation-specific.
        tiled_symbols: Per-loop-level iteration-space symbols, innermost first.
            ``tiled_symbols[0]`` lists the symbols tiled by the innermost enclosing
            loop; ``tiled_symbols[1]`` lists those tiled by the next-outer loop; etc.
            Empty for ops not inside any loop.
            **Invariant:** every enclosing loop level must have an entry, even if
            empty (``[]``).  An empty entry means the op is loop-invariant at that
            level.  This keeps level indices aligned with nesting depth so that
            ``compile_op_spec``'s reversal maps each level to the correct
            ``loop_var_depth`` index in ``_collect_affine_maps``.
            The unroller reads ``tiled_symbols[0]`` at each level and removes it
            from the list after processing, leaving outer-level entries intact.
            The bundle path (compile_op_spec / generate_sdsc) reverses this list to
            outermost-first and builds per-level affine.apply stride maps, mapping
            each level's strides to the correct loop variable by explicit index.
        tiled_symbol_trip_counts: Maps each symbol appearing in tiled_symbols
            to its own nesting level's trip count (CoarseTileInfo.loop_count
            for that level). Used by SDSC codegen to compute each tiled
            tensor's full pre-tiling extent as
            (per-unit-step device element advance) * trip_count, without
            needing a separately tracked full-extent field on TensorArg.
            Only correct when a symbol belongs to exactly one nesting level
            -- a symbol tiled at more than one level has no single trip
            count this field could hold, so this is scoped to the common
            one-level-per-symbol case -- empty for non-tiled ops.
    """

    op: str
    is_reduction: bool
    iteration_space: dict[Symbol, tuple[Expr, int]]
    args: Sequence[TensorArg]
    op_info: dict[str, Any]
    tiled_symbols: list[list[Symbol]] = dataclasses.field(default_factory=list)
    tiled_symbol_trip_counts: dict[Symbol, int] = dataclasses.field(
        default_factory=dict
    )
    # Maps PyTorch symbol name (e.g. 's97') -> (max, granularity) bounds.
    # Populated by compute_symbolic_bounds during
    # create_op_spec; empty for concrete dims.
    symbolic_dim_bounds: dict[str, tuple[int, int]] = dataclasses.field(
        default_factory=dict
    )
    # Full logical output ranges of the write/reduction node (NCHW for pools),
    # including unit dims.  Distinct from the squeezed, permuted iteration_space:
    # pool codegen derives which dim roles survived (and the channel count) from
    # these live ranges rather than a lowering-time size snapshot.  None when the
    # node exposes no data.ranges.
    node_output_ranges: tuple[Expr, ...] | None = None
    debug_handle: DebugHandle | None = None


# --- Module-level constant tensor cache --------------------------------------
#
# LRU cache for Spyre constant tensors to avoid redundant tensor creation across
# multiple forward passes. Uses OrderedDict for least-recently-used eviction
# with a maximum size limit to bound memory usage.
#
# Thread-safe for concurrent inference workloads.
#
_CACHE_MAX_SIZE = 1000  # Maximum number of cached constants (~128KB)
_CONSTANT_TENSOR_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def clear_constant_tensor_cache():
    """Clear the constant tensor cache.

    Should be called when:
    - Running torch.compiler.reset() to ensure test isolation
    - Manually reclaiming device memory
    - Starting a new inference session with different constants

    Example:
        >>> torch.compiler.reset()
        >>> clear_constant_tensor_cache()  # Clear Spyre constants
    """
    _CONSTANT_TENSOR_CACHE.clear()


@dataclasses.dataclass
class UnimplementedOp:
    op: str


@dataclasses.dataclass
class LoopSpec:
    """A counted loop whose body is a sequence of ops, possibly nested.

    Attributes:
        count: Trip count of the loop. May be a symbolic shape expression.
        body: The operations to execute each iteration. Each element may be
            an OpSpec, UnimplementedOp, or a nested LoopSpec.

    Each OpSpec in the body carries its own ``tiled_symbols`` list identifying
    which of its iteration-space symbols are tiled by the loop that directly
    contains it.  The unroller reads these per-op symbols rather than a shared
    list, so ops with different iteration-space layouts in the same loop are
    each advanced by the correct stride.
    """

    count: Expr
    # list[OpSpec | UnimplementedOp | LoopSpec], typed as Any to accommodate
    # the two distinct UnimplementedOp types (op_spec vs spyre_kernel).
    body: list[Any]


def spyre_constant_tensor(const_val, device, dtype=torch.float16):
    """Create or retrieve a cached constant tensor for Spyre device.

    Uses module-level LRU cache that persists across forward passes with a
    maximum size limit (default: 1000 entries, ~128KB). Least-recently-used
    entries are automatically evicted when the cache exceeds the limit.

    For test isolation, call clear_constant_tensor_cache() after
    torch.compiler.reset().

    WARNING: Returns a shared tensor that MUST NEVER be mutated in-place.

    The returned tensor is reused across multiple calls with matching
    (value, device, dtype). In-place mutation would corrupt all callers.

    Inductor enforces immutability via:
      - SpyreConstantFallback.should_allocate() == False
      - SpyreConstantFallback.get_mutation_names() == []

    Any pass that modifies this contract MUST update this function.

    Args:
        const_val: Scalar constant value
        device: Target device (e.g., "spyre", "cpu", or torch.device(...))
        dtype: Data type (default: torch.float16)

    Returns:
        Immutable constant tensor (DO NOT MUTATE)

    Note:
        For non-Spyre devices (e.g., CPU), uses standard torch.tensor path
        without caching as these tensors are cheap to create.

        float(const_val) makes NaN values always miss the cache because
        float('nan') != float('nan'). This is intentional - it falls back to
        creating a fresh tensor for each NaN request rather than attempting
        cache matching.
    """
    # Normalize device to torch.device object if string is passed
    if isinstance(device, str):
        device = torch.device(device)

    # For non-Spyre devices (e.g., CPU), use standard PyTorch path without caching
    if device.type != "spyre":
        return torch.tensor(const_val, dtype=dtype).to(device)

    # Create cache key with normalized device (device.type, device.index)
    device_key = (device.type, device.index)
    cache_key = (float(const_val), device_key, dtype)

    # Thread-safe cache lookup and insertion with LRU eviction
    with _CACHE_LOCK:
        if cache_key in _CONSTANT_TENSOR_CACHE:
            # Cache hit: move to end (most recently used)
            _CONSTANT_TENSOR_CACHE.move_to_end(cache_key)
            return _CONSTANT_TENSOR_CACHE[cache_key]

        # Cache miss: create new tensor with device-side fill
        t = torch.empty((), dtype=dtype, device=device)
        _C.fill_tensor(t, float(const_val))

        _CONSTANT_TENSOR_CACHE[cache_key] = t

        # Evict least-recently-used entries if over limit
        while len(_CONSTANT_TENSOR_CACHE) > _CACHE_MAX_SIZE:
            _CONSTANT_TENSOR_CACHE.popitem(last=False)  # Remove oldest (LRU)

        return t


def find_unimplemented(specs: list) -> UnimplementedOp | None:
    """Return the first UnimplementedOp in specs (recursing into LoopSpec), or None."""
    for entry in specs:
        if isinstance(entry, UnimplementedOp):
            return entry
        if isinstance(entry, LoopSpec):
            found = find_unimplemented(entry.body)
            if found is not None:
                return found
    return None


def format_op_spec_list(specs: list, indent: int = 0) -> str:
    """Format an op spec list for structured logging output.

    Uses an explicit stack to avoid recursion-depth issues with deeply
    nested LoopSpecs.
    """
    lines: list[str] = []
    stack: list[tuple[list, int, int]] = [(specs, indent, 0)]
    while stack:
        current_specs, cur_indent, idx = stack.pop()
        if idx >= len(current_specs):
            continue
        # Push remainder back for later processing.
        stack.append((current_specs, cur_indent, idx + 1))
        item = current_specs[idx]
        prefix = "  " * cur_indent
        if isinstance(item, LoopSpec):
            lines.append(f"{prefix}LoopSpec(count={item.count})")
            lines.append(f"{prefix}  body=[")
            # Push a sentinel to close the body bracket after children.
            stack.append(([_LoopClose(prefix)], cur_indent, 0))
            # Push the body for processing at deeper indent.
            stack.append((item.body, cur_indent + 2, 0))
        elif isinstance(item, OpSpec):
            it_space_str = ", ".join(
                f"{k}: ({v[0]}, {v[1]})" for k, v in item.iteration_space.items()
            )
            lines.append(
                f"{prefix}OpSpec(op={item.op!r}, "
                f"is_reduction={item.is_reduction}, "
                f"iteration_space={{{it_space_str}}})"
            )
            for arg in item.args:
                lines.append(
                    f"{prefix}  TensorArg("
                    f"{'input' if arg.is_input else 'output'}, "
                    f"arg_index={arg.arg_index}, "
                    f"device_size={arg.device_size}, "
                    f"device_coordinates={arg.device_coordinates}, "
                    f"device_tile_advance_expr={arg.device_tile_advance_expr}, "
                    f"allocation={arg.allocation})"
                )
            if item.tiled_symbols:
                lines.append(f"{prefix}  tiled_symbols={item.tiled_symbols}")
            if item.symbolic_dim_bounds:
                lines.append(
                    f"{prefix}  symbolic_dim_bounds={item.symbolic_dim_bounds}"
                )
        elif isinstance(item, UnimplementedOp):
            lines.append(f"{prefix}UnimplementedOp(op={item.op!r})")
        elif isinstance(item, _LoopClose):
            lines.append(f"{item.prefix}  ]")
        else:
            lines.append(f"{prefix}{item!r}")
    return "\n".join(lines)


class _LoopClose:
    """Sentinel used by format_op_spec_list to emit closing brackets."""

    __slots__ = ("prefix",)

    def __init__(self, prefix: str):
        self.prefix = prefix
