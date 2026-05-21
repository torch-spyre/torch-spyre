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
from .constants import BATCH_MATMUL_OP
from .pass_utils import host_coordinates, device_coordinates
from .views import matching_dim, compute_coordinates
from torch_spyre._C import SpyreTensorLayout
from torch.utils.weak import WeakTensorKeyDictionary

logger = get_inductor_logger("propagate_real_dims")


_real_dims = {}

_real_tensor_dims = WeakTensorKeyDictionary()


def declare_real_dim(name, size):
    """
    Declare a real dimension
    """
    _real_dims[name] = size


def annotate_real_dims(tensor, real_dims):
    """
    Annotate tensor with real dimensions: [(name, size), ...]
    """
    _real_tensor_dims[tensor] = real_dims
    return tensor


def _get_buffer(dep):
    return V.graph.get_buffer(dep.name)



def _lone_sym(coord: sympy.Expr) -> sympy.Symbol:
    return next(iter(coord.free_symbols))


def _compute_real_layout(real_dims):
    """Compute real size and stride from declared real dim sizes."""
    size = []
    stride = [1]
    for s in reversed(real_dims):
        stride.append(stride[-1] * _real_dims[s])
        size.append(_real_dims[s])
    return list(reversed(size)), list(reversed(stride[:-1]))


def compute_input_real_dims(dep: MemoryDep) -> dict:
    """Map loop vars to real dim names for a single input dep, using real-space coords."""
    buf = _get_buffer(dep)
    if not hasattr(buf, 'real_dims') or buf.real_dims is None:
        # Scalar broadcast: constant index, contributes nothing to rdims
        if not dep.index.free_symbols:
            return {}
        return None
    real_size, real_stride = _compute_real_layout(buf.real_dims)
    coords = compute_coordinates(real_size, real_stride, dep.ranges, dep.index)
    result = {}
    for i, coord in enumerate(coords):
        if coord.free_symbols:
            sym = _lone_sym(coord)
            name = buf.real_dims[i]
            actual_range = int(dep.ranges[sym])
            declared_size = _real_dims.get(name)
            if declared_size is not None and actual_range != declared_size:
                logger.warning(
                    f"{dep.name}: loop var {sym} has range {actual_range} "
                    f"but maps to '{name}' declared as {declared_size} -- partial/sliced dim"
                )
            result.setdefault(sym, []).append(name)
    return result


def op_out_coords(op: ComputedBuffer) -> list:
    output_dep = next(iter(op.get_read_writes().writes))
    return host_coordinates(op.get_layout(), output_dep)


def coords_to_real_dims(coords: list, rdims: dict) -> list:
    """Map coordinate expressions to real dim names via their loop variable."""
    result = []
    for c in coords:
        if c.free_symbols:
            sym = _lone_sym(c)
            if sym not in rdims:
                logger.warning(f"coords_to_real_dims: no mapping for {sym} -- returning None")
                return None
            result.extend(rdims[sym])
    return result


def get_input_real_dims(inputs: list) -> dict:
    """
    Manage real_dims for all inputs, mapping dims from upstream node to this node.
    Returns None if any input has no real_dims.
    """
    rdims = {}
    for inp in inputs:
        new = compute_input_real_dims(inp)
        if new is None:
            return None
        rdims.update(new)
    return rdims


def get_reduction_dim(dep: MemoryDep, out_coords: list) -> sympy.Symbol:
    """Return the reduction loop variable: the input coord absent from the output."""
    in_coords = host_coordinates(_get_buffer(dep).get_layout(), dep)
    reduction_coord = next(
        c for c in in_coords
        if c.free_symbols and matching_dim(out_coords, c) is None
    )
    return _lone_sym(reduction_coord)


def _set_no_real_dims(op):
    op.real_dims = None
    op.reduction_dims = None
    op.rdims = {}


def _compute_real_dims(op, inputs):
    rdims = get_input_real_dims(inputs)
    if rdims is None:
        _set_no_real_dims(op)
        return
    out_coords = op_out_coords(op)
    real_dims = coords_to_real_dims(out_coords, rdims)
    if real_dims is None:
        _set_no_real_dims(op)
        return
    op.real_dims = real_dims
    op.rdims = rdims
    if isinstance(op.data, Reduction):
        op.reduction_dims = rdims[get_reduction_dim(inputs[0], out_coords)]
    else:
        op.reduction_dims = None


def propagate_real_dims(
    operations: list[Operation],
) -> None:
    """
    Propagate real dims from inputs though graph
    """
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
                ptl = tb.data.data.layout
                if not isinstance(ptl, FixedLayout):
                    raise Unsupported(f"graph input {name} does not have a FixedLayout")
                tb.real_dims = _real_tensor_dims.get(real_input)

    def _dump_dep(label, dep):
        buf = V.graph.get_buffer(dep.name)
        layout = buf.get_layout() if hasattr(buf, 'get_layout') else None
        real_dims = getattr(buf, 'real_dims', '?')
        logger.debug(f"  {label} {dep.name}: real_dims={real_dims}")
        if layout is not None:
            logger.debug(f"    host_size={list(layout.size)}  host_stride={list(layout.stride)}")
            logger.debug(f"    host_coordinates={host_coordinates(layout, dep)}")
        stl = getattr(buf, 'layout', None)
        if isinstance(stl, SpyreTensorLayout):
            logger.debug(f"    device_size={stl.device_size}  stride_map={stl.stride_map}")
            logger.debug(f"    device_coordinates={device_coordinates(stl, dep)}")
        logger.debug(f"    index={dep.index}  ranges={dict(dep.ranges)}")

    it = iter(operations)
    for op in it:
        if op.is_no_op():
            op.real_dims = []
            op.iterations = []
        elif isinstance(op, ComputedBuffer):
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
                continue
            origins = getattr(op.data, 'origins', set())
            aten_ops = [str(n.target) for n in origins if hasattr(n, 'target')]
            reduction_type = getattr(op.data, 'reduction_type', None)
            logger.debug(f"\n--- {op.get_operation_name()} ({type(op.data).__name__}) aten={aten_ops} reduction_type={reduction_type}")
            rw = op.get_read_writes()
            inputs = []
            for input in rw.reads:
                if isinstance(input, MemoryDep):
                    inputs.append(input)
                    _dump_dep("input", input)
            for write in rw.writes:
                if isinstance(write, MemoryDep):
                    _dump_dep("output", write)
            if isinstance(op.data, (Pointwise, Reduction)):
                _compute_real_dims(op, inputs)
            else:
                logger.warning(f"Warning: unhandled node type {type(op.data)}")
                _set_no_real_dims(op)
        else:
            logger.warning(f"unhandled operation type {type(op)}")
            _set_no_real_dims(op)

    # debug

    print("DECLARED DIMS")
    for name, size in _real_dims.items():
        print(f"  {name} = {size}")

    print("INPUT TENSORS")
    for name in V.graph.graph_input_names:
        tb = V.graph.graph_inputs[name]
        if isinstance(tb, TensorBox):
            print(f"  {name}: real_dims={tb.real_dims}")

    print("OPS")
    for op in iter(operations):
        if not hasattr(op, 'rdims') or op.real_dims is None:
            origins = getattr(getattr(op, 'data', op), 'origins', set())
            aten_ops = [str(n.target) for n in origins if hasattr(n, 'target')]
            print(f"  {op.get_operation_name()}: skipped ({type(op).__name__} / {type(getattr(op, 'data', op)).__name__})  aten={aten_ops}")
            continue
        is_reduction = isinstance(op.data, Reduction)
        origins = getattr(op.data, 'origins', set())
        aten_ops = [str(n.target) for n in origins if hasattr(n, 'target')]
        reduction_type = getattr(op.data, 'reduction_type', None)
        print(f"  {op.get_operation_name()} ({'reduction' if is_reduction else 'pointwise'})  aten={aten_ops}  reduction_type={reduction_type}")
        rw = op.get_read_writes()
        all_deps = list(rw.reads) + list(rw.writes)
        for dep in rw.reads:
            if isinstance(dep, MemoryDep):
                buf = _get_buffer(dep)
                real_dims = getattr(buf, 'real_dims', '?')
                host_size = list(buf.get_layout().size) if hasattr(buf, 'get_layout') else '?'
                print(f"    input {dep.name}: real_dims={real_dims}  host_size={host_size}  index={dep.index}  ranges={dict(dep.ranges)}")
        print(f"    loop vars:")
        ranges = {}
        for dep in all_deps:
            if isinstance(dep, MemoryDep):
                ranges.update({str(s): int(v) for s, v in dep.ranges.items()})
        for sym, names in op.rdims.items():
            size = ranges.get(str(sym), "?")
            declared = [f"{n}={_real_dims.get(n,'?')}" for n in names]
            print(f"      {sym}: range={size}  real_dim(s)={names}  declared={declared}")
        if is_reduction:
            print(f"    reduction over: {op.reduction_dims}")
        print(f"    output: ({op.get_name()}) real_dims={op.real_dims}")
        print()
