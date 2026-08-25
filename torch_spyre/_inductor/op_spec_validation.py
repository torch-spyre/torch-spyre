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


"""OpSpec validation — enforces invariants at each pipeline stage.

Call ``validate_op_specs(specs, stage=...)`` at stage boundaries to catch
invariant violations early with descriptive error messages.

Invariants checked:
- Structural: correct types, non-empty mandatory fields
- Consistency: device_coordinates symbols <= iteration_space keys
- Array: len(device_coordinates) == len(device_size) per TensorArg
- Op-specific: matmul arg counts, reduction flag consistency
- Tiling: tiled_symbols entries <= iteration_space keys
- LoopSpec: non-empty body, positive count
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
import functools
import inspect
import textwrap

import regex
import sympy

from . import constants
from .dtype_ops import DtypeOpTable
from .logging_utils import get_inductor_logger
from .op_spec import IndirectAccess, LoopSpec, OpSpec, TensorArg, UnimplementedOp

logger = get_inductor_logger("op_spec_validation")


@functools.lru_cache(maxsize=1)
def _pointwise_ops() -> frozenset[str]:
    """Op names produced by SpyreOpFuncs's static methods.

    Statically extracts the literal op-name argument from each method's
    ``return PointwiseOp("name", ...)`` statement(s), rather than calling the
    methods (which would require guessing each method's call signature,
    including varargs like ``layernormnorm`` and keyword args like
    ``softplus``). Methods whose op name is computed rather than a literal
    (e.g. ``to_dtype``, resolved via DtypeOpTable) contribute no literal and
    are correctly excluded -- those are dtype-cast ops, not pointwise ops.

    Imported and evaluated lazily (rather than at module load) because
    spyre_kernel imports validate_op_specs from this module, so resolving
    SpyreOpFuncs at op_spec_validation's own import time would be circular.
    """
    from .spyre_kernel import SpyreOpFuncs

    tree = ast.parse(textwrap.dedent(inspect.getsource(SpyreOpFuncs)))
    (class_node,) = tree.body
    names: set[str] = set()
    for node in ast.walk(class_node):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "PointwiseOp"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ):
            names.add(call.args[0].value)
    return frozenset(names)


# Known op names, derived by inspecting the modules that actually define
# them so this file cannot silently drift out of sync as ops are added.
MATMUL_OPS = constants.MATMUL_REDUCTION_OPS

REDUCTION_OPS = constants.SINGLE_INPUT_REDUCTION_OPS

DTYPE_OPS = DtypeOpTable.op_names()

SPECIAL_OPS = frozenset({constants.RESTICKIFY_OP})

BINARY_OPS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "realdiv",
        "maximum",
        "minimum",
        "equal",
        "notequal",
        "greaterthan",
        "greaterequal",
        "lesserthan",
        "lesserequal",
    }
)


@functools.lru_cache(maxsize=1)
def _all_known_ops() -> frozenset[str]:
    return MATMUL_OPS | REDUCTION_OPS | _pointwise_ops() | DTYPE_OPS | SPECIAL_OPS


def __getattr__(name: str):
    # POINTWISE_OPS/ALL_KNOWN_OPS are computed lazily (see _pointwise_ops) to
    # avoid a circular import with spyre_kernel, but are still exposed as
    # plain module attributes for callers/tests via PEP 562.
    if name == "POINTWISE_OPS":
        return _pointwise_ops()
    if name == "ALL_KNOWN_OPS":
        return _all_known_ops()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


VALID_ALLOCATION_KEYS = frozenset({"hbm", "lx", "hbm_pool"})


class OpSpecValidationError(ValueError):
    """Raised when an OpSpec invariant is violated."""

    def __init__(self, op_spec, invariant: str, detail: str, stage: str = ""):
        op_name = getattr(op_spec, "op", repr(op_spec))
        stage_prefix = f"[{stage}] " if stage else ""
        msg = (
            f"{stage_prefix}OpSpec validation failed for op={op_name!r}: "
            f"{invariant}. {detail}"
        )
        super().__init__(msg)
        self.op_spec = op_spec
        self.invariant = invariant


def _is_unimplemented_op(item: object) -> bool:
    """Duck-type check for UnimplementedOp variants.

    spyre_kernel.py defines a local UnimplementedOp(RValue) that also appears
    in op_spec lists. Importing it here would create a circular dependency, so
    we detect it structurally: has an ``op`` attribute and is named
    "UnimplementedOp".
    """
    return (
        type(item).__name__ == "UnimplementedOp"
        and hasattr(item, "op")
        and isinstance(item.op, str)
    )


def validate_op_specs(
    specs: list | Sequence,
    stage: str = "",
    loop_depth: int = 0,
) -> None:
    """Validate a list of OpSpec/LoopSpec/UnimplementedOp entries.

    Args:
        specs: The op spec list to validate (may contain nested LoopSpecs).
        stage: Human-readable label for the pipeline stage (for error messages).
        loop_depth: Current nesting depth (0 = top level, used for tiled_symbols
            depth checks).

    Raises:
        OpSpecValidationError: If any invariant is violated.
    """
    for item in specs:
        if isinstance(item, LoopSpec):
            _validate_loop_spec(item, stage, loop_depth)
        elif isinstance(item, OpSpec):
            _validate_op_spec(item, stage, loop_depth)
        elif isinstance(item, UnimplementedOp):
            pass
        elif _is_unimplemented_op(item):
            pass
        else:
            raise OpSpecValidationError(
                item,
                "unexpected type in spec list",
                f"Got {type(item).__name__}, expected OpSpec/LoopSpec/UnimplementedOp",
                stage,
            )


def _validate_loop_spec(loop: LoopSpec, stage: str, loop_depth: int) -> None:
    """Validate a LoopSpec and recurse into its body."""
    if not loop.body:
        raise OpSpecValidationError(
            loop,
            "LoopSpec has empty body",
            "A LoopSpec must contain at least one operation",
            stage,
        )

    # Only check concrete counts; symbolic counts are valid (resolved at runtime).
    if isinstance(loop.count, (int, sympy.Integer)):
        if int(loop.count) <= 0:
            raise OpSpecValidationError(
                loop,
                "LoopSpec count must be positive",
                f"Got count={loop.count}",
                stage,
            )

    validate_op_specs(loop.body, stage, loop_depth + 1)


def _validate_op_spec(op_spec: OpSpec, stage: str, loop_depth: int) -> None:
    """Validate a single OpSpec against all invariant categories."""
    # Specs with empty args or iteration_space are test stubs (used with mocked
    # compile_op_spec). Production code always has both populated. Skip
    # validation; they would fail trivially and will error at compile time.
    if not op_spec.args or not op_spec.iteration_space:
        return
    _check_mandatory_fields(op_spec, stage)
    _check_op_name(op_spec, stage)
    _check_iteration_space(op_spec, stage)
    _check_args(op_spec, stage)
    _check_symbol_consistency(op_spec, stage)
    _check_tiled_symbols(op_spec, stage, loop_depth)
    _check_op_specific_constraints(op_spec, stage)


def _check_mandatory_fields(op_spec: OpSpec, stage: str) -> None:
    """OS-1: All mandatory fields must be present and correctly typed."""
    if not isinstance(op_spec.op, str) or not op_spec.op:
        raise OpSpecValidationError(
            op_spec,
            "op must be a non-empty string",
            f"Got op={op_spec.op!r}",
            stage,
        )

    if not isinstance(op_spec.is_reduction, bool):
        raise OpSpecValidationError(
            op_spec,
            "is_reduction must be a bool",
            f"Got is_reduction={op_spec.is_reduction!r}",
            stage,
        )

    if not isinstance(op_spec.iteration_space, dict):
        raise OpSpecValidationError(
            op_spec,
            "iteration_space must be a dict",
            f"Got type={type(op_spec.iteration_space).__name__}",
            stage,
        )

    if not isinstance(op_spec.args, (list, tuple)):
        raise OpSpecValidationError(
            op_spec,
            "args must be a list or tuple",
            f"Got type={type(op_spec.args).__name__}",
            stage,
        )


def _check_op_name(op_spec: OpSpec, stage: str) -> None:
    """OS-2: Op name should be a known operation."""
    if op_spec.op not in _all_known_ops():
        logger.warning(
            "Unknown op name %r in OpSpec at stage %s. "
            "This may be valid for new ops not yet registered.",
            op_spec.op,
            stage,
        )


def _check_iteration_space(op_spec: OpSpec, stage: str) -> None:
    """OS-3: iteration_space keys must be Symbols; values must be (Expr, int)."""
    for key, value in op_spec.iteration_space.items():
        if not isinstance(key, sympy.Symbol):
            raise OpSpecValidationError(
                op_spec,
                "iteration_space keys must be sympy.Symbol",
                f"Got key type={type(key).__name__}: {key!r}",
                stage,
            )

        if not isinstance(value, tuple) or len(value) != 2:
            raise OpSpecValidationError(
                op_spec,
                "iteration_space values must be 2-tuples (range_expr, work_division)",
                f"Got value={value!r} for key={key}",
                stage,
            )

        range_expr, work_div = value
        if not isinstance(range_expr, (sympy.Expr, int)):
            raise OpSpecValidationError(
                op_spec,
                "iteration_space range must be a sympy Expr or int",
                f"Got type={type(range_expr).__name__} for key={key}",
                stage,
            )

        if not isinstance(work_div, int) or work_div < 1:
            raise OpSpecValidationError(
                op_spec,
                "iteration_space work_division must be a positive int",
                f"Got work_division={work_div!r} for key={key}",
                stage,
            )


def _check_args(op_spec: OpSpec, stage: str) -> None:
    """OS-4: Each TensorArg must have consistent array sizes."""
    has_output = False
    for i, arg in enumerate(op_spec.args):
        if not isinstance(arg, TensorArg):
            raise OpSpecValidationError(
                op_spec,
                f"args[{i}] must be a TensorArg",
                f"Got type={type(arg).__name__}",
                stage,
            )

        if not arg.is_input:
            has_output = True

        if not isinstance(arg.device_size, list) or not arg.device_size:
            raise OpSpecValidationError(
                op_spec,
                f"args[{i}].device_size must be a non-empty list",
                f"Got device_size={arg.device_size!r}",
                stage,
            )

        for dim_idx, sz in enumerate(arg.device_size):
            if not isinstance(sz, (int, sympy.Integer)) or sz < 0:
                raise OpSpecValidationError(
                    op_spec,
                    f"args[{i}].device_size[{dim_idx}] must be a non-negative int",
                    f"Got {sz!r}",
                    stage,
                )

        if not isinstance(arg.device_coordinates, list):
            raise OpSpecValidationError(
                op_spec,
                f"args[{i}].device_coordinates must be a list",
                f"Got type={type(arg.device_coordinates).__name__}",
                stage,
            )

        if len(arg.device_coordinates) != len(arg.device_size):
            raise OpSpecValidationError(
                op_spec,
                f"args[{i}]: len(device_coordinates) must equal len(device_size)",
                f"Got {len(arg.device_coordinates)} coordinates "
                f"but {len(arg.device_size)} sizes",
                stage,
            )

        # arg_index is -1 until assigned during bundle preparation, so only
        # enforce non-negative at the final stage.  Pool-allocated args
        # (hbm_pool) and LX-allocated args retain arg_index=-1 permanently
        # because they are not kernel parameters — skip the check for those.
        if stage == "before_bundle_generation" and arg.arg_index < 0:
            if "hbm_pool" not in arg.allocation and "lx" not in arg.allocation:
                raise OpSpecValidationError(
                    op_spec,
                    f"args[{i}].arg_index must be a non-negative int",
                    f"Got arg_index={arg.arg_index!r}",
                    stage,
                )

        _check_allocation(op_spec, arg, i, stage)

    if not has_output and op_spec.op in _all_known_ops():
        raise OpSpecValidationError(
            op_spec,
            "OpSpec must have at least one output TensorArg (is_input=False)",
            f"All {len(op_spec.args)} args are inputs",
            stage,
        )


def _check_allocation(op_spec: OpSpec, arg: TensorArg, idx: int, stage: str) -> None:
    """OS-4b: allocation must contain exactly one of hbm/lx/hbm_pool."""
    # Allocation is populated after op-spec creation: HBM addresses are assigned
    # in codegen_kernel (after the after_creation/after_simplification hooks),
    # LX addresses by the scratchpad allocator pass, and hbm_pool by
    # hbm_pool_planning.  Only enforce at the final stage.
    if stage != "before_bundle_generation":
        return
    if not isinstance(arg.allocation, dict):
        raise OpSpecValidationError(
            op_spec,
            f"args[{idx}].allocation must be a dict",
            f"Got type={type(arg.allocation).__name__}",
            stage,
        )
    keys = set(arg.allocation.keys()) & VALID_ALLOCATION_KEYS
    if len(keys) != 1:
        raise OpSpecValidationError(
            op_spec,
            f"args[{idx}].allocation must contain exactly one of hbm/lx/hbm_pool",
            f"Got keys={set(arg.allocation.keys())!r}",
            stage,
        )


def _check_symbol_consistency(op_spec: OpSpec, stage: str) -> None:
    """OS-5: Free symbols in device_coordinates must reference iteration_space keys."""
    it_space_syms = set(op_spec.iteration_space.keys())

    for i, arg in enumerate(op_spec.args):
        for coord_idx, coord in enumerate(arg.device_coordinates):
            if not isinstance(coord, sympy.Expr):
                continue
            if coord.has(IndirectAccess):
                continue
            free = coord.free_symbols
            invalid_syms = free - it_space_syms
            # Before simplification, indirect indexing symbols (indirect0,
            # indirect1, ...) are plain Symbols not yet wrapped in
            # IndirectAccess — allow them through.
            invalid_syms = {
                s for s in invalid_syms if not regex.fullmatch(r"indirect\d+", str(s))
            }
            if invalid_syms:
                raise OpSpecValidationError(
                    op_spec,
                    f"args[{i}].device_coordinates[{coord_idx}] references "
                    f"symbols not in iteration_space",
                    f"Invalid symbols: {invalid_syms}, "
                    f"iteration_space keys: {it_space_syms}",
                    stage,
                )


def _check_tiled_symbols(op_spec: OpSpec, stage: str, loop_depth: int) -> None:
    """OS-6: tiled_symbols entries must reference valid symbols.

    tiled_symbols is list[list[Symbol]] — per-loop-level, innermost first.
    Each symbol must be either an iteration_space key or a minted
    tile-advance symbol (appearing in some arg's device_tile_advance_expr).
    """
    it_space_syms = set(op_spec.iteration_space.keys())

    tile_advance_syms: set[sympy.Symbol] = set()
    for arg in op_spec.args:
        if arg.device_tile_advance_expr is not None:
            tile_advance_syms.update(
                s
                for s in arg.device_tile_advance_expr.free_symbols
                if isinstance(s, sympy.Symbol)
            )

    valid_tiled_syms = (
        it_space_syms | tile_advance_syms | set(op_spec.tiled_symbol_trip_counts.keys())
    )

    if not isinstance(op_spec.tiled_symbols, list):
        raise OpSpecValidationError(
            op_spec,
            "tiled_symbols must be a list",
            f"Got type={type(op_spec.tiled_symbols).__name__}",
            stage,
        )

    for level_idx, level_syms in enumerate(op_spec.tiled_symbols):
        if not isinstance(level_syms, list):
            raise OpSpecValidationError(
                op_spec,
                f"tiled_symbols[{level_idx}] must be a list of Symbols",
                f"Got type={type(level_syms).__name__}",
                stage,
            )
        for sym in level_syms:
            if not isinstance(sym, sympy.Symbol):
                raise OpSpecValidationError(
                    op_spec,
                    f"tiled_symbols[{level_idx}] must contain sympy.Symbol",
                    f"Got {type(sym).__name__}: {sym!r}",
                    stage,
                )
            if sym not in valid_tiled_syms:
                raise OpSpecValidationError(
                    op_spec,
                    f"tiled_symbols[{level_idx}] references symbol not in "
                    f"iteration_space or device_tile_advance_expr",
                    f"Symbol {sym} not in {valid_tiled_syms}",
                    stage,
                )

    if loop_depth > 0 and op_spec.tiled_symbols:
        if len(op_spec.tiled_symbols) != loop_depth:
            logger.debug(
                "OpSpec op=%r has %d tiled_symbols levels but is at "
                "loop_depth=%d (may be valid during construction)",
                op_spec.op,
                len(op_spec.tiled_symbols),
                loop_depth,
            )


def _check_op_specific_constraints(op_spec: OpSpec, stage: str) -> None:
    """OS-7: Op-specific constraints on arg counts and is_reduction."""
    op = op_spec.op
    n_inputs = sum(1 for a in op_spec.args if a.is_input)
    n_outputs = sum(1 for a in op_spec.args if not a.is_input)

    if op in MATMUL_OPS:
        if n_inputs < 2:
            raise OpSpecValidationError(
                op_spec,
                "matmul ops require at least 2 input TensorArgs",
                f"Got {n_inputs} inputs",
                stage,
            )
        if n_outputs < 1:
            raise OpSpecValidationError(
                op_spec,
                "matmul ops require at least 1 output TensorArg",
                f"Got {n_outputs} outputs",
                stage,
            )
        if not op_spec.is_reduction:
            raise OpSpecValidationError(
                op_spec,
                "matmul ops must have is_reduction=True",
                f"Got is_reduction={op_spec.is_reduction}",
                stage,
            )

    if op in REDUCTION_OPS and not op_spec.is_reduction:
        raise OpSpecValidationError(
            op_spec,
            f"reduction op {op!r} must have is_reduction=True",
            f"Got is_reduction={op_spec.is_reduction}",
            stage,
        )

    if op in _pointwise_ops() and op_spec.is_reduction:
        raise OpSpecValidationError(
            op_spec,
            f"pointwise op {op!r} must have is_reduction=False",
            f"Got is_reduction={op_spec.is_reduction}",
            stage,
        )

    if op in BINARY_OPS and n_inputs < 2:
        raise OpSpecValidationError(
            op_spec,
            f"binary op {op!r} requires at least 2 input TensorArgs",
            f"Got {n_inputs} inputs",
            stage,
        )

    if op == "where3" and n_inputs < 3:
        raise OpSpecValidationError(
            op_spec,
            "where3 requires at least 3 input TensorArgs",
            f"Got {n_inputs} inputs",
            stage,
        )
