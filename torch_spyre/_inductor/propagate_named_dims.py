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
from torch._inductor.virtualized import V
from .errors import Unsupported
from .pass_utils import host_coordinates, device_coordinates
from .views import matching_dim, compute_coordinates
from torch_spyre._C import SpyreTensorLayout
from torch.utils.weak import WeakTensorKeyDictionary

logger = get_inductor_logger("propagate_named_dims")


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
    buf = _get_buffer(dep)
    if not hasattr(buf, "named_dims") or buf.named_dims is None:
        # Scalar broadcast: constant index, contributes nothing to loop_var_dims
        if not dep.index.free_symbols:
            return {}
        # Unannotated tensor: synthesize _untracked_ names from dep ranges
        context = f"{op.get_name()}/{dep.name}" if op is not None else dep.name
        return {
            sym: [_untracked_name(context, sym, int(size))]
            for sym, size in dep.ranges.items()
        }
    named_size, named_stride = _compute_named_layout(buf.named_dims)
    coords = compute_coordinates(named_size, named_stride, dep.ranges, dep.index)
    result: dict[sympy.Symbol, list[str]] = {}
    for i, coord in enumerate(coords):
        if coord.free_symbols:
            sym = _lone_sym(coord)
            result.setdefault(sym, []).append(buf.named_dims[i])
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


def op_out_coords(op: ComputedBuffer) -> list:
    output_dep = next(iter(op.get_read_writes().writes))
    return host_coordinates(op.get_layout(), output_dep)


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
    names = op.loop_var_dims.get(sym, [])
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


def _set_no_named_dims(op):
    op.named_dims = []
    op.reduction_named_dims = None
    op.loop_var_dims = {}


def _compute_named_dims(op, inputs):
    loop_var_dims = get_input_named_dims(inputs, op)
    out_coords = op_out_coords(op)
    if not isinstance(op.data, Reduction):
        # For pointwise ops, synthesize names for loop vars not covered by any input.
        # This handles full/zeros_like: their iteration space defines named dims but
        # their constant value contributes nothing to loop_var_dims.
        output_dep = next(iter(op.get_read_writes().writes))
        for coord in out_coords:
            if coord.free_symbols:
                sym = _lone_sym(coord)
                if sym not in loop_var_dims:
                    size = int(output_dep.ranges[sym])
                    loop_var_dims[sym] = [_untracked_name(op.get_name(), sym, size)]
    named_dims = coords_to_named_dims(out_coords, loop_var_dims)
    op.named_dims = named_dims
    op.loop_var_dims = loop_var_dims
    if isinstance(op.data, Reduction):
        op.reduction_named_dims = loop_var_dims[
            get_reduction_dim(inputs[0], out_coords)
        ]
    else:
        op.reduction_named_dims = None


def _log_dep_debug(label: str, dep: MemoryDep) -> None:
    buf = V.graph.get_buffer(dep.name)
    layout = buf.get_layout() if hasattr(buf, "get_layout") else None
    named_dims = getattr(buf, "named_dims", "?")
    logger.debug(f"  {label} {dep.name}: named_dims={named_dims}")
    if layout is not None:
        logger.debug(
            f"    host_size={list(layout.size)}  host_stride={list(layout.stride)}"
        )
        logger.debug(f"    host_coordinates={host_coordinates(layout, dep)}")
    stl = getattr(buf, "layout", None)
    if isinstance(stl, SpyreTensorLayout):
        logger.debug(f"    device_size={stl.device_size}  stride_map={stl.stride_map}")
        logger.debug(f"    device_coordinates={device_coordinates(stl, dep)}")
    logger.debug(f"    index={dep.index}  ranges={dict(dep.ranges)}")


def _log_op_inputs(op: ComputedBuffer) -> None:
    for dep in op.get_read_writes().reads:
        if isinstance(dep, MemoryDep):
            buf = _get_buffer(dep)
            named_dims = getattr(buf, "named_dims", "?")
            host_size = (
                list(buf.get_layout().size) if hasattr(buf, "get_layout") else "?"
            )
            logger.info(
                f"    input {dep.name}: named_dims={named_dims}  host_size={host_size}"
                f"  index={dep.index}  ranges={dict(dep.ranges)}"
            )


def _log_op(op: Operation) -> None:
    origins: set = getattr(getattr(op, "data", op), "origins", set())
    aten_ops = [str(n.target) for n in origins if hasattr(n, "target")]
    if not hasattr(op, "loop_var_dims") or not op.loop_var_dims:
        logger.info(
            f"  {op.get_operation_name()}: skipped"
            f" ({type(op).__name__} / {type(getattr(op, 'data', op)).__name__})"
            f"  aten={aten_ops}"
        )
        if isinstance(op, ComputedBuffer):
            _log_op_inputs(op)
            logger.info(
                f"    output: ({op.get_name()})"
                f" named_dims={getattr(op, 'named_dims', '?')}"
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
    for sym, names in op.loop_var_dims.items():
        sym_range: int | str = ranges.get(str(sym), "?")
        declared = [f"{n}={_named_dims[n] if n in _named_dims else '?'}" for n in names]
        logger.info(
            f"      {sym}: range={sym_range}  named_dim(s)={names}  declared={declared}"
        )
    if is_reduction:
        logger.info(f"    reduction over: {op.reduction_named_dims}")
    logger.info(f"    output: ({op.get_name()}) named_dims={op.named_dims}")
    logger.info("")


def propagate_named_dims(
    operations: list[Operation],
) -> None:
    """Propagate named dims from annotated inputs through the op graph."""
    if not _enabled:
        return
    if len(V.graph.graph_input_names) > 0:
        for name, real_input in zip(V.graph.graph_input_names, V.get_real_inputs()):
            if isinstance(real_input, torch.Tensor):
                tb = V.graph.graph_inputs[name]
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
                tb.named_dims = _named_tensor_dims.get(real_input)

    for op in operations:
        if op.is_no_op():
            op.named_dims = []
        elif isinstance(op, ComputedBuffer):
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
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
            for dep in inputs:
                _log_dep_debug("input", dep)
            for dep in rw.writes:
                if isinstance(dep, MemoryDep):
                    _log_dep_debug("output", dep)
            if isinstance(op.data, (Pointwise, Reduction)):
                _compute_named_dims(op, inputs)
            else:
                logger.warning(f"Warning: unhandled node type {type(op.data)}")
                _set_no_named_dims(op)
        else:
            logger.warning(f"unhandled operation type {type(op)}")
            _set_no_named_dims(op)

    # LOG THE RESULTS
    logger.info("DECLARED DIMS")
    for name, size in _named_dims.items():
        logger.info(f"  {name} = {size}")

    logger.info("INPUT TENSORS")
    for name in V.graph.graph_input_names:
        tb = V.graph.graph_inputs[name]
        if isinstance(tb, TensorBox):
            logger.info(f"  {name}: named_dims={tb.named_dims}")

    logger.info("OPS")
    for op in operations:
        _log_op(op)
