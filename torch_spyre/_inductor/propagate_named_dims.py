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


import dataclasses
import logging
import sympy
import torch
from .logging_utils import get_inductor_logger
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
    Reduction,
    StorageBox,
    TensorBox,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from .errors import Unsupported
from .pass_utils import host_coordinates, device_coordinates, op_out_coords
from .ir import SpyreConstantFallback
from .propagate_hints import DimHint, get_op_hints
from .views import matching_dim, compute_coordinates
from torch_spyre._C import SpyreTensorLayout
from torch.utils.weak import WeakTensorKeyDictionary

logger = get_inductor_logger("propagate_named_dims")
hints_logger = get_inductor_logger("assign_dim_hints")


# Used for propagation of named dims if this pass runs.
# This pass does not run unless the driver program called name_tensor_dims.
_named_dims: dict[str, int] = {}
_named_tensor_dims = WeakTensorKeyDictionary()
_enabled = False


def reset():
    global _enabled
    _named_dims.clear()
    _named_tensor_dims.clear()
    _enabled = False


def declare_tensor_dim(name: str, size: int) -> None:
    """Declare a named tensor dimension and its size."""
    _named_dims[name] = size


def name_tensor_dims(tensor: torch.Tensor, named_dims: list[str]) -> torch.Tensor:
    """Annotate a tensor with its named dimensions: [name, ...]"""
    global _enabled
    _enabled = True
    _named_tensor_dims[tensor] = named_dims
    return tensor


def _get_buffer(dep):
    return V.graph.get_buffer(dep.name)


def _get_dim_prop_info(dep):
    buf = _get_buffer(dep)
    if buf is not None:
        dpi = getattr(buf, "_dim_prop_info", None)
        if dpi is not None:
            return dpi
    tb = V.graph.graph_inputs.get(dep.name)
    return getattr(tb, "_dim_prop_info", None) if tb is not None else None


def _lone_sym(coord: sympy.Expr) -> sympy.Symbol:
    return next(iter(coord.free_symbols))


def _untracked_name(context: str, sym, size: int) -> str:
    name = f"_untracked_{size}"
    _named_dims.setdefault(name, size)
    logger.warning(
        f"{context}: loop var {sym} has no named dim mapping -- using {name}"
    )
    return name


def _compute_named_layout(named_dims):
    """Compute size and stride from declared named dim sizes."""
    size = []
    stride = [1]
    for s in reversed(named_dims):
        if s not in _named_dims:
            raise KeyError(
                f"Named dim '{s}' used in name_tensor_dims but not declared -- "
                f"call declare_tensor_dim('{s}', size) before compiling"
            )
        stride.append(stride[-1] * _named_dims[s])
        size.append(_named_dims[s])
    return list(reversed(size)), list(reversed(stride[:-1]))


def compute_input_named_dims(dep: MemoryDep, op=None) -> dict:
    """Map loop vars to named dim names for a single input dep, using named-space coords."""
    dpi = _get_dim_prop_info(dep)
    buf_named_dims = dpi.named_dims if dpi is not None else None
    if not buf_named_dims:
        # Scalar broadcast: constant index, contributes nothing to loop_var_dims
        if not dep.index.free_symbols:
            return {}
        # Unannotated tensor: synthesize _untracked_ names from dep ranges
        context = f"{op.get_name()}/{dep.name}" if op is not None else dep.name
        return {
            sym: [_untracked_name(context, sym, int(size))]
            for sym, size in dep.ranges.items()
        }
    named_size, named_stride = _compute_named_layout(buf_named_dims)
    coords = compute_coordinates(named_size, named_stride, dep.ranges, dep.index)
    result: dict[sympy.Symbol, list[str]] = {}
    for i, coord in enumerate(coords):
        if coord.free_symbols:
            sym = _lone_sym(coord)
            if sym in dep.ranges:
                result.setdefault(sym, []).append(buf_named_dims[i])
    for sym, names in result.items():
        actual_range = int(dep.ranges[sym])
        product = 1
        for n in names:
            product *= _named_dims.get(n, actual_range)
        if actual_range != product:
            logger.warning(
                f"{dep.name}: loop var {sym} has range {actual_range} "
                f"but maps to {names} with product {product} -- partial/sliced dim, "
                f"continuing using range {actual_range}"
            )
    return result


def coords_to_named_dims(coords: list, loop_var_dims: dict) -> list:
    """Map coordinate expressions to named dim names via their loop variable."""
    result = []
    for c in coords:
        if c.free_symbols:
            sym = _lone_sym(c)
            assert sym in loop_var_dims, (
                f"coords_to_named_dims: no mapping for loop var {sym} -- "
                f"this is a bug in _compute_named_dims synthesis"
            )
            result.extend(loop_var_dims[sym])
    return result


def named_dims_for_sym(op: ComputedBuffer, sym: sympy.Symbol) -> list[tuple[str, int]]:
    """Return [(name, size), ...] for the named dims covered by a loop variable."""
    dp = getattr(op, "_dim_prop_info", None)
    names = dp.loop_var_dims.get(sym, []) if dp is not None else []
    return [(n, _named_dims[n]) for n in names if n in _named_dims]


def named_dims_for_coord(
    op: ComputedBuffer, coord: sympy.Expr
) -> list[tuple[str, int]] | None:
    """Return [(name, size), ...] for the named dims covered by a host coord expression."""
    if not coord.free_symbols:
        return None
    return named_dims_for_sym(op, _lone_sym(coord))


def get_input_named_dims(inputs: list, op=None) -> dict:
    """
    Merge named dim mappings from all inputs into a single loop-var → names dict.
    Real names win over _untracked_ placeholders when both inputs cover the same sym.
    """
    loop_var_dims: dict[sympy.Symbol, list[str]] = {}
    for inp in inputs:
        new = compute_input_named_dims(inp, op)
        for sym, names in new.items():
            if sym not in loop_var_dims or all(
                n.startswith("_untracked_") for n in loop_var_dims[sym]
            ):
                loop_var_dims[sym] = names
    return loop_var_dims


def get_reduction_dim(dep: MemoryDep, out_coords: list) -> sympy.Symbol:
    """Return the reduction loop variable: the input coord absent from the output."""
    in_coords = host_coordinates(_get_buffer(dep).get_layout(), dep)
    reduction_coord = next(
        c for c in in_coords if c.free_symbols and matching_dim(out_coords, c) is None
    )
    return _lone_sym(reduction_coord)


@dataclasses.dataclass
class _DimPropInfo:
    named_dims: list = dataclasses.field(default_factory=list)
    reduction_named_dims: list | None = None
    loop_var_dims: dict = dataclasses.field(default_factory=dict)


def _set_no_named_dims(op):
    op._dim_prop_info = _DimPropInfo()  # type: ignore[attr-defined]


def _compute_named_dims(op, inputs):
    loop_var_dims = get_input_named_dims(inputs, op)
    out_coords = op_out_coords(op)
    # Synthesize names for loop vars not covered by any input.
    # This handles full/zeros_like/constant ops whose iteration space defines
    # named dims but whose constant value contributes nothing to loop_var_dims.
    output_dep = next(iter(op.get_read_writes().writes))
    for coord in out_coords:
        if coord.free_symbols:
            sym = _lone_sym(coord)
            if sym not in loop_var_dims:
                size = int(output_dep.ranges[sym])
                loop_var_dims[sym] = [_untracked_name(op.get_name(), sym, size)]
    reduction_named_dims = None
    if isinstance(op.data, Reduction):
        reduction_sym = get_reduction_dim(inputs[0], out_coords)
        if reduction_sym not in loop_var_dims:
            size = int(inputs[0].ranges[reduction_sym])
            loop_var_dims[reduction_sym] = [
                _untracked_name(op.get_name(), reduction_sym, size)
            ]
        reduction_named_dims = loop_var_dims[reduction_sym]
    named_dims = coords_to_named_dims(out_coords, loop_var_dims)
    op._dim_prop_info = _DimPropInfo(  # type: ignore[attr-defined]
        named_dims=named_dims,
        loop_var_dims=loop_var_dims,
        reduction_named_dims=reduction_named_dims,
    )


def _log_dep_debug(
    label: str, dep: MemoryDep
) -> (
    None
):  # TODO: thread indirect_load_subs to show IndirectAccess in device_coordinates
    buf = _get_buffer(dep)
    layout = (
        buf.get_layout() if buf is not None and hasattr(buf, "get_layout") else None
    )
    dpi = _get_dim_prop_info(dep)
    named_dims = dpi.named_dims if dpi is not None else []
    logger.debug(f"  {label} {dep.name}: named_dims={named_dims}")
    if layout is not None:
        logger.debug(
            f"    host_size={list(layout.size)}  host_stride={list(layout.stride)}"
        )
        logger.debug(f"    host_coordinates={host_coordinates(layout, dep)}")
    stl = getattr(buf, "layout", None) if buf is not None else None
    if isinstance(stl, SpyreTensorLayout):
        logger.debug(f"    device_size={stl.device_size}  stride_map={stl.stride_map}")
        logger.debug(f"    device_coordinates={device_coordinates(stl, dep)}")
    logger.debug(f"    index={dep.index}  ranges={dict(dep.ranges)}")


def _log_op_inputs(op: ComputedBuffer) -> None:
    for dep in op.get_read_writes().reads:
        if isinstance(dep, MemoryDep):
            dpi = _get_dim_prop_info(dep)
            named_dims = dpi.named_dims if dpi is not None else "?"
            buf = _get_buffer(dep)
            host_size = (
                list(buf.get_layout().size)
                if buf is not None and hasattr(buf, "get_layout")
                else "?"
            )
            logger.info(
                f"    input {dep.name}: named_dims={named_dims}  host_size={host_size}"
                f"  index={dep.index}  ranges={dict(dep.ranges)}"
            )


def _log_op(op: Operation) -> None:
    origins: set = getattr(getattr(op, "data", op), "origins", set())
    aten_ops = [str(n.target) for n in origins if hasattr(n, "target")]
    dp = getattr(op, "_dim_prop_info", None)
    if dp is None or not dp.loop_var_dims:
        logger.info(
            f"  {op.get_operation_name()}: skipped"
            f" ({type(op).__name__} / {type(getattr(op, 'data', op)).__name__})"
            f"  aten={aten_ops}"
        )
        if isinstance(op, ComputedBuffer):
            _log_op_inputs(op)
            logger.info(
                f"    output: ({op.get_name()}) named_dims={dp.named_dims if dp else []}"
            )
        return
    is_reduction = isinstance(op.data, Reduction)
    reduction_type = getattr(op.data, "reduction_type", None)
    logger.info(
        f"  {op.get_operation_name()}"
        f" ({'reduction' if is_reduction else 'pointwise'})"
        f"  aten={aten_ops}  reduction_type={reduction_type}"
    )
    _log_op_inputs(op)
    logger.info("    loop vars:")
    rw = op.get_read_writes()
    ranges = {}
    for dep in list(rw.reads) + list(rw.writes):
        if isinstance(dep, MemoryDep):
            ranges.update({str(s): int(v) for s, v in dep.ranges.items()})
    for sym, names in dp.loop_var_dims.items():
        sym_range: int | str = ranges.get(str(sym), "?")
        declared = [f"{n}={_named_dims[n] if n in _named_dims else '?'}" for n in names]
        logger.info(
            f"      {sym}: range={sym_range}  named_dim(s)={names}  declared={declared}"
        )
    if is_reduction:
        logger.info(f"    reduction over: {dp.reduction_named_dims}")
    logger.info(f"    output: ({op.get_name()}) named_dims={dp.named_dims}")
    logger.info("")


def _propagate_named_dims_impl(graph: GraphLowering) -> None:
    operations = graph.operations
    if graph.graph_input_names:
        for name, real_input in zip(graph.graph_input_names, V.get_real_inputs()):
            if isinstance(real_input, torch.Tensor):
                tb = graph.graph_inputs[name]
                if (
                    not isinstance(tb, TensorBox)
                    or not isinstance(tb.data, StorageBox)
                    or not isinstance(tb.data.data, InputBuffer)
                ):
                    raise Unsupported(
                        f"graph input {name} is not a TensorBox(StorageBox(InputBuffer))"
                    )
                layout = tb.data.data.layout
                if not isinstance(layout, FixedLayout):
                    raise Unsupported(f"graph input {name} does not have a FixedLayout")
                tb._dim_prop_info = _DimPropInfo(  # type: ignore[attr-defined]
                    named_dims=_named_tensor_dims.get(real_input) or []
                )

    for op in operations:
        if op.is_no_op():
            _set_no_named_dims(op)
        elif isinstance(op, ComputedBuffer):
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                continue
            hint = False
            for hint_dict in get_op_hints(op).values():
                if "named_dims" in hint_dict:
                    hint = True
                    named_dims = hint_dict["named_dims"]
                    break
            if hint:
                coords = op_out_coords(op)
                loop_var_dims = {
                    _lone_sym(coord): [dim_name]
                    for coord, dim_name in zip(coords, named_dims)
                    if len(coord.free_symbols) == 1
                }
                op._dim_prop_info = _DimPropInfo(  # type: ignore[attr-defined]
                    named_dims=named_dims,
                    loop_var_dims=loop_var_dims,
                )
                continue
            origins: set = getattr(op.data, "origins", set())
            aten_ops = [str(n.target) for n in origins if hasattr(n, "target")]
            reduction_type = getattr(op.data, "reduction_type", None)
            logger.debug(
                f"\n--- {op.get_operation_name()} ({type(op.data).__name__})"
                f" aten={aten_ops} reduction_type={reduction_type}"
            )
            rw = op.get_read_writes()
            inputs = [d for d in rw.reads if isinstance(d, MemoryDep)]
            if logger.isEnabledFor(logging.DEBUG):
                for dep in inputs:
                    _log_dep_debug("input", dep)
                for dep in rw.writes:
                    if isinstance(dep, MemoryDep):
                        _log_dep_debug("output", dep)
            if isinstance(op.data, (Pointwise, Reduction)):
                _compute_named_dims(op, inputs)
            else:
                logger.warning(f"unhandled node type {type(op.data)}")
                _set_no_named_dims(op)
        elif isinstance(op, SpyreConstantFallback):
            _set_no_named_dims(op)
        else:
            logger.warning(f"unhandled operation type {type(op)}")
            _set_no_named_dims(op)

    if logger.isEnabledFor(logging.INFO):
        logger.info("DECLARED DIMS")
        for name, size in _named_dims.items():
            logger.info(f"  {name} = {size}")

        logger.info("INPUT TENSORS")
        for name in graph.graph_input_names:
            tb = graph.graph_inputs[name]
            if isinstance(tb, TensorBox):
                dp = getattr(tb, "_dim_prop_info", None)
                logger.info(f"  {name}: named_dims={dp.named_dims if dp else []}")

        logger.info("OPS")
        for op in operations:
            _log_op(op)


def propagate_named_dims(
    graph: GraphLowering,
) -> None:
    """Propagate named dims from annotated inputs through the op graph."""
    global _enabled
    if not _enabled:
        return
    try:
        _propagate_named_dims_impl(graph)
    finally:
        _named_tensor_dims.clear()
        _enabled = False


def _assign_dim_hints_impl(operations: list[Operation]) -> None:
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        # Reconstructed buffers can copy optional metadata; recompute it here.
        if hasattr(op, "work_div_loop_info"):
            del op.work_div_loop_info  # type: ignore[attr-defined]
        dp = getattr(op, "_dim_prop_info", None)
        op_hints = get_op_hints(op) if dp and dp.loop_var_dims else {}
        if not op_hints:
            op.dim_hints = []  # type: ignore[attr-defined]
            if dp is not None:
                del op._dim_prop_info  # type: ignore[attr-defined]
            continue

        assert dp is not None  # guaranteed by op_hints check above
        if any(hint_dict.get("work_div") for hint_dict in op_hints.values()):
            op.work_div_loop_info = {  # type: ignore[attr-defined]
                sym: list(names) for sym, names in dp.loop_var_dims.items()
            }

        reduction_dims = set(dp.reduction_named_dims or [])

        coord_for_name: dict[str, sympy.Symbol] = {}
        for coord in op_out_coords(op):
            if not coord.free_symbols:
                continue
            sym = _lone_sym(coord)
            for name, _ in named_dims_for_sym(op, sym):
                coord_for_name[name] = sym
        # Also map reduction dim names to their loop variable.  Reduction dims
        # don't appear in output coordinates, so they would never be found by
        # the output-coord loop above.  dp.loop_var_dims covers all loop vars
        # (including the reduction dim), so we invert it for reduction names.
        for sym, names in dp.loop_var_dims.items():
            for name in names:
                if name in reduction_dims:
                    coord_for_name[name] = sym

        dim_hints = []
        for hint_id, hint_dict in sorted(op_hints.items()):
            # A hint scope uses exactly one of tiles/slices/num_tiles_per_dim.
            dims: dict[str, int] = next(
                (
                    v
                    for k in ("tiles", "slices", "num_tiles_per_dim")
                    if (v := hint_dict.get(k))
                ),
                {},
            )
            # TODO: support multiple dimensions per spyre_hint() call.
            # hint_id_to_ranges_pos in _stamp_group would need to become
            # dict[int, list[int]] and _hints_levels would need to deduplicate
            # by hint_id.
            if len(dims) > 1:
                raise NotImplementedError(
                    f"spyre_hint() argument {list(hint_dict.items())} specifies "
                    f"{len(dims)} dimensions; only one is currently allowed per "
                    f"spyre_hint() call (not yet implemented)"
                )
            for name, count in dims.items():
                sym = coord_for_name.get(name)
                dim_hints.append(
                    DimHint(
                        dim_names=[name],
                        split_count=count,
                        loop_var=sym,
                        is_reduction=name in reduction_dims,
                        hint_id=hint_id,
                    )
                )
        op.dim_hints = dim_hints  # type: ignore[attr-defined]

        # Clean up temp intermediates — only dim_hints persists.
        del op._dim_prop_info  # type: ignore[attr-defined]

    if hints_logger.isEnabledFor(logging.INFO):
        ops = [
            op
            for op in operations
            if isinstance(op, ComputedBuffer) and getattr(op, "dim_hints", None)
        ]
        if ops:
            hints_logger.info("=== assign_dim_hints ===")
            for op in ops:
                rw = op.get_read_writes()
                all_ranges = {
                    s: int(v)
                    for dep in [*rw.reads, *rw.writes]
                    for s, v in dep.ranges.items()
                }
                hints_logger.info(f"{op.get_operation_name()}:")
                for h in op.dim_hints:
                    r = all_ranges.get(h.loop_var, 0) if h.loop_var else 0
                    per_tile = r // h.split_count if r else "?"
                    reduction_tag = "  [reduction]" if h.is_reduction else ""
                    hints_logger.info(
                        f"  {h.dim_names}  range={r}"
                        f"  split_count={h.split_count}  -> {per_tile} per tile"
                        f"  loop_var={h.loop_var}{reduction_tag}"
                    )


def assign_dim_hints(graph: GraphLowering) -> None:
    """Combine spyre_hint scope annotations with propagated named dimensions.

    Reads the hint scopes (from spyre_hint() context managers in user code,
    attached to FX nodes via meta["custom"]) and matches hinted dimension names
    against the named loop variables on each op.  The named loop variables come
    from propagate_named_dims(), which propagates name_tensor_dims() annotations
    through the op graph — that pass must run before this one.

    Produces op.dim_hints: a flat list of DimHint, one per hinted dimension,
    ordered outermost hint scope first.  Consumed by hints_to_coarse_tile_groups
    to form coarse tiling groups.

    Deletes op._dim_prop_info when done — those fields are only needed here.
    """
    try:
        _assign_dim_hints_impl(graph.operations)
    finally:
        reset()
