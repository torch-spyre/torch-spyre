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
import sympy
import torch
import torch.fx as fx
from torch._inductor.ir import ComputedBuffer, FixedLayout, Pointwise
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from .logging_utils import get_inductor_logger
from .errors import Unsupported
from .pass_utils import replace_computed_buffer_body
from torch_spyre._C import SpyreTensorLayout, ElementArrangement
from torch_spyre.constants import DEVICE_NAME

logger = get_inductor_logger("split_multi_ops")

# Operations that don't perform computation but manage data flow
_STRUCTURAL_OPS = frozenset({"load", "store", "get_index"})

# Operations that involve dtype conversion or constant creation
_DTYPE_OPS = frozenset({"to_dtype", "convert_element_type", "constant"})

# These operations are a special case for SDSC codegen: their scalar parameters must be
# passed as compile-time constants via an `op_info['constants']` dictionary rather than
# as standard input buffers.
_OPS_WITH_CONSTANT_ARGS = frozenset({"clamp", "layernormscale", "softplus"})

# Mapping of operation names to their FX graph targets when there is no 1-1 mapping.
_OP_TARGET_TABLE = {
    "to_dtype": torch.ops.prims.convert_element_type.default,
    "convert_element_type": torch.ops.prims.convert_element_type.default,
}


class _Val:
    __slots__ = ("handler", "vid")

    def __init__(self, handler, vid):
        self.handler, self.vid = handler, vid


class _TracingHandler:
    """Context manager that traces operations into a list of operation records.

    This handler intercepts all V.ops calls during tracing and records them
    as tuples of (op_name, vid, input_vids, kwargs). It's used to analyze
    the structure of fused operations before splitting them.

    Attributes:
        ops: List of traced operations as (op_name, vid, input_vids, kwargs)
        _next_vid: Counter for generating unique value IDs
    """

    def __init__(self, prev_handler):
        self._prev = prev_handler
        self.ops = []
        self._next_vid = 0

    def __enter__(self):
        """Install this handler as the active V.ops handler."""
        self._saved = V.ops
        V.set_ops_handler(self)
        return self

    def __exit__(self, *args):
        """Restore the previous V.ops handler."""
        V.set_ops_handler(self._saved)

    def __getattr__(self, name):
        """Dynamically create recording functions for any operation name.

        Args:
            name: Operation name to record

        Returns:
            A function that records the operation and returns a _Val
        """
        if name.startswith("_"):
            raise AttributeError(name)

        def _record(*args, **kwargs):
            # Separate _Val arguments from other arguments
            vids = []
            extra = []
            for a in args:
                if isinstance(a, _Val):
                    vids.append(a.vid)
                elif isinstance(a, (bool, int, float)):
                    # Special case: Wrap literals as constants
                    vids.append(self._wrap_literal(a).vid)
                else:
                    extra.append(a)
            # Store non _Val args with _p prefix to avoid name conflicts
            merged = dict(kwargs)
            for i, v in enumerate(extra):
                merged[f"_p{i}"] = v
            vid = self._alloc_vid()
            self.ops.append((name, vid, tuple(vids), merged))
            return _Val(self, vid)

        return _record

    def load(self, name, index):
        """Record a load operation from a named buffer.

        Args:
            name: Buffer name to load from
            index: Index expression for the load

        Returns:
            _Val representing the loaded value
        """
        vid = self._alloc_vid()
        self.ops.append(("load", vid, (), {"_name": name, "_index": index}))
        return _Val(self, vid)

    def store(self, name, index, value, mode=None):
        """Record a store operation to a named buffer.

        Args:
            name: Buffer name to store to
            index: Index expression for the store
            value: _Val to store
            mode: Optional store mode (e.g., 'atomic_add')

        Returns:
            _Val representing the store operation
        """
        vid = self._alloc_vid()
        self.ops.append(
            (
                "store",
                vid,
                (value.vid,),
                {"_name": name, "_index": index, "_mode": mode},
            )
        )
        return _Val(self, vid)

    def constant(self, fill_value, dtype):
        """Record a constant value creation."""
        vid = self._alloc_vid()
        self.ops.append(
            ("constant", vid, (), {"fill_value": fill_value, "dtype": dtype})
        )
        return _Val(self, vid)

    def _alloc_vid(self):
        """Allocate a new unique value ID."""
        v = self._next_vid
        self._next_vid += 1
        return v

    def _wrap_literal(self, value):
        """Wrap a Python literal as a constant operation."""
        dtype = (
            torch.bool
            if isinstance(value, bool)
            else (torch.int64 if isinstance(value, int) else torch.float32)
        )
        return self.constant(value, dtype)


def _infer_output_dtype(input_dtypes, kwargs, fallback):
    if "dtype" in kwargs:
        return kwargs["dtype"]
    if not input_dtypes:
        return fallback
    result = input_dtypes[0]
    for dt in input_dtypes[1:]:
        result = torch.result_type(
            torch.empty(0, dtype=result), torch.empty(0, dtype=dt)
        )
    return result


def _resolve_fx_target(op_name):
    if op_name in _OP_TARGET_TABLE:
        return _OP_TARGET_TABLE[op_name]
    for ns in (torch.ops.aten, torch.ops.prims, getattr(torch.ops, DEVICE_NAME, None)):
        if ns is None:
            continue
        target = getattr(ns, op_name, None)
        if target is not None:
            default = getattr(target, "default", None)
            return default or target
    return None


def _normalize_op_args(op_name, input_fx_nodes, kwargs, out_dtype, device=None):
    """Normalize operation arguments for FX graph node creation.

    Special cases:
    1. 'constant' ops: Return (fill_value, dtype, device) tuple
    2. _DTYPE_OPS: Ensure dtype is first positional arg after input
    3. Positional args stored with '_p' prefix are extracted and appended
    4. Internal kwargs (starting with '_') are filtered out

    Args:
        op_name: Name of the operation
        input_fx_nodes: List of input FX nodes
        kwargs: Operation keyword arguments (may include _p0, _p1, etc.)
        out_dtype: Default output dtype
        device: Optional device for constant operations

    Returns:
        Tuple of (args, clean_kwargs, final_dtype)
    """
    if op_name == "constant":
        fill = kwargs["fill_value"]
        dtype = kwargs.get("dtype", out_dtype)
        dev = device if device is not None else torch.device(DEVICE_NAME)
        return (fill, dtype, dev), {}, dtype

    pos_keys = sorted(k for k in kwargs if k.startswith("_p"))
    pos_vals = [kwargs[k] for k in pos_keys]
    clean_kw = {
        k: v
        for k, v in kwargs.items()
        if not k.startswith("_") and k not in ("dtype", "src_dtype")
    }
    args = list(input_fx_nodes)

    if op_name in _DTYPE_OPS:
        target_dtype = pos_vals[0] if pos_vals else kwargs.get("dtype", out_dtype)
        extra_pos = pos_vals[1:] if pos_vals else []
        args = [input_fx_nodes[0], target_dtype] if input_fx_nodes else [target_dtype]
        args.extend(extra_pos)
        out_dtype = target_dtype
    else:
        args.extend(pos_vals)
        if "dtype" in kwargs:
            clean_kw["dtype"] = kwargs["dtype"]
            out_dtype = kwargs["dtype"]
    return tuple(args), clean_kw, out_dtype


def _build_inner_fn(op_name, value_vids, kwargs, vid_to_bufname, vid_to_constant):
    """Build an inner_fn that loads from intermediate buffers.

    Special cases:
    1. Constants in _OPS_WITH_CONSTANT_ARGS: Use scalar value directly
    2. Other constants: Load from buffer at index 0
    3. Regular values: Compute strided index and load

    Args:
        op_name: Name of the operation to perform
        value_vids: List of value IDs for operation inputs
        kwargs: Operation keyword arguments
        vid_to_bufname: Mapping from value ID to buffer name
        vid_to_constant: Mapping from value ID to (fill_value, dtype)

    Returns:
        Function that takes an index tuple and returns the operation result
    """
    pos_keys = sorted(k for k in kwargs if k.startswith("_p"))
    extra = tuple(kwargs[k] for k in pos_keys)
    clean_kw = {k: v for k, v in kwargs.items() if not k.startswith("_p")}

    # Pre-compute strides for non-constant inputs
    vid_to_stride = {}
    for v in value_vids:
        if v not in vid_to_constant:
            buf_name = vid_to_bufname[v]
            vid_to_stride[v] = V.graph.get_buffer(buf_name).layout.stride

    def inner_fn(index):
        inputs = []
        for v in value_vids:
            if v in vid_to_constant:
                # Handle constant arguments that need to be passed as scalars
                if op_name in _OPS_WITH_CONSTANT_ARGS:
                    fill, _ = vid_to_constant[v]
                    inputs.append(fill)
                else:
                    inputs.append(V.ops.load(vid_to_bufname[v], sympy.Integer(0)))
            else:
                buf_stride = vid_to_stride[v]
                if len(index) != len(buf_stride):
                    raise ValueError(
                        f"Mismatch between index & stride dimensions: "
                        f"{len(index)} vs {len(buf_stride)}"
                    )
                idx = sum(i * s for i, s in zip(index, buf_stride))
                inputs.append(V.ops.load(vid_to_bufname[v], idx))
        return getattr(V.ops, op_name)(*inputs, *extra, **clean_kw)

    return inner_fn


def _trace_inner_fn(op):
    """Trace the inner_fn of an operation to extract its structure.

    Returns None if tracing fails (e.g., unsupported ops)

    Args:
        op: ComputedBuffer operation to trace

    Returns:
        List of traced operations or None if tracing failed
    """
    ranges = op.data.ranges
    syms = tuple(sympy.Symbol(f"_i{k}") for k in range(len(ranges)))
    tracer = _TracingHandler(V.ops)
    try:
        with tracer:
            op.data.inner_fn(syms)
    except Exception:
        return None
    return tracer.ops


def _get_compute_ops(trace):
    """Compute ops from a trace other than structural ops."""
    return [e for e in trace if e[0] not in _STRUCTURAL_OPS]


def _propagate_dtypes(trace, fallback):
    """Propagate dtypes through a trace of operations.

    Special cases:
    1. 'load' ops: Get dtype from buffer layout
    2. 'store' ops: No dtype propagation needed
    3. Other ops: Infer dtype from inputs

    Args:
        trace: List of traced operations
        fallback: Default dtype if inference fails

    Returns:
        Dictionary mapping value IDs to their dtypes
    """
    dtype_map = {}
    for op, vid, inputs, kwargs in trace:
        if op == "load":
            buf_name = kwargs["_name"]
            dtype_map[vid] = V.graph.get_buffer(buf_name).get_layout().dtype
        elif op == "store":
            pass
        else:
            in_dtypes = [dtype_map[v] for v in inputs if v in dtype_map]
            dtype_map[vid] = _infer_output_dtype(in_dtypes, kwargs, fallback)
    return dtype_map


def _init_vid_to_bufname(trace):
    """Extract buffer names from load operations in a trace.

    Args:
        trace: List of traced operations

    Returns:
        Dictionary mapping value IDs to buffer names
    """
    return {vid: kwargs["_name"] for op, vid, _, kwargs in trace if op == "load"}


def _init_vid_to_constant(trace):
    """Extract constant values from constant operations in a trace.

    Args:
        trace: List of traced operations

    Returns:
        Dictionary mapping value IDs to (fill_value, dtype) tuples
    """
    return {
        vid: (kwargs["fill_value"], kwargs["dtype"])
        for op, vid, _, kwargs in trace
        if op == "constant"
    }


def _find_fx_node(name, gl):
    """Find an FX node by buffer name in the graph.

    Searches both gl.env and placeholder nodes

    Args:
        name: Buffer name to search for
        gl: GraphLowering instance

    Returns:
        FX Node corresponding to the buffer name

    Raises:
        KeyError: If no FX node is found for the given name
    """
    for n, tb in gl.env.items():
        if isinstance(n, fx.Node) and tb is not None:
            if tb.get_name() == name:
                return n
    for n in gl.graph.nodes:
        if n.op == "placeholder" and n.name == name:
            return n
    raise KeyError(f"No FX node for {name}")


def _lower_fx_node(node, gl, ops, idx):
    """Lower an FX node to a buffer and insert it into operations list.

    Args:
        node: FX node to lower
        gl: GraphLowering instance
        ops: Operations list to insert into
        idx: Index position for insertion

    Returns:
        The created buffer
    """
    tb = gl.run_node(node)
    buf = tb.data.data
    gl.operations.remove(buf)
    ops.insert(idx, buf)
    gl.name_to_buffer[buf.get_name()] = buf
    return buf


def _make_intermediate_bufs(
    intermediate_ops,
    vid_to_dtype,
    vid_to_bufname,
    layout,
    operations,
    insert_idx,
    gl,
    orig_node,
) -> None:
    """Create intermediate buffers for operations.

    For each intermediate operation in a fused computation, this creates:
    1. An FX graph node with the appropriate target and arguments
    2. A lowered buffer from that node
    3. Updates vid_to_bufname mapping for subsequent operations

    Special cases:
    1. Metadata propagation: Copies 'val' metadata from inputs or orig_node
    2. Node insertion: Uses inserting_before context to maintain graph order
    3. Origins tracking: Sets OrderedSet with single new_node as origin

    Args:
        intermediate_ops: List of (op_name, vid, inputs, kwargs) tuples
        vid_to_dtype: Mapping from value ID to dtype
        vid_to_bufname: Mapping from value ID to buffer name (updated)
        layout: Layout of the original fused operation
        operations: List of operations to insert into
        insert_idx: Starting index for insertion
        gl: GraphLowering instance
        orig_node: Original FX node being split
    """
    bufs = []
    for op_name, vid, inputs, kwargs in intermediate_ops:
        out_dtype = vid_to_dtype.get(vid, layout.dtype)
        input_nodes = [_find_fx_node(vid_to_bufname[v], gl) for v in inputs]
        target = _resolve_fx_target(op_name)
        if target is None:
            raise RuntimeError(f"Cannot resolve target for '{op_name}'")

        args, clean_kw, out_dtype = _normalize_op_args(
            op_name, input_nodes, kwargs, out_dtype, layout.device
        )

        with gl.graph.inserting_before(orig_node):
            new_node = gl.graph.create_node("call_function", target, args, clean_kw)

        # Propagate metadata for shape inference
        if input_nodes and "val" in input_nodes[0].meta:
            new_node.meta["val"] = input_nodes[0].meta["val"].to(out_dtype)
        elif "val" in orig_node.meta:
            new_node.meta["val"] = orig_node.meta["val"].to(out_dtype)

        new_buf = _lower_fx_node(new_node, gl, operations, insert_idx)
        new_buf.origins = OrderedSet([new_node])
        vid_to_bufname[vid] = new_buf.get_name()
        bufs.append(new_buf)
        insert_idx += 1


def _update_original_buf(
    op, final_entry, vid_to_bufname, vid_to_constant, operations
) -> None:
    """Update the original buffer to use intermediate buffers.

    Replaces the original fused operation's inner_fn with one that loads
    from the newly created intermediate buffers. Preserves all metadata
    and attributes from the original operation.

    Args:
        op: Original ComputedBuffer to update
        final_entry: Final operation tuple (op_name, vid, inputs, kwargs)
        vid_to_bufname: Mapping from value ID to buffer name
        vid_to_constant: Mapping from value ID to constant values
        operations: List of operations containing op
    """
    op_name, _, vids, kwargs = final_entry

    # New data for the new op
    new_data = dataclasses.replace(
        op.data,
        inner_fn=_build_inner_fn(
            op_name, vids, kwargs, vid_to_bufname, vid_to_constant
        ),
    )

    # Preserve metadata from original's ops data in new_data
    metadata_attrs = (
        "origins",
        "traceback",
        "origin_node",
        "annotations",
        "stream_idx",
    )
    for attr in metadata_attrs:
        if hasattr(op.data, attr):
            object.__setattr__(new_data, attr, getattr(op.data, attr))

    new_op = replace_computed_buffer_body(op, new_data, operations)
    V.graph.name_to_buffer[new_op.get_name()] = new_op


def _get_op_name(op) -> str:
    """Extract the operation name from a node for validation."""
    if not hasattr(op.data, "origins") or not op.data.origins:
        return ""

    # Get the first origin node
    origin_node = next(iter(op.data.origins))

    # Try to get the target and its name
    target = getattr(origin_node, "target", None)
    if target and hasattr(target, "name"):
        full_name = target.name()
        if "::" in full_name:
            # Extract just the operation name (e.g., "add" from "aten::add.Tensor")
            return full_name.split("::")[1].split(".")[0]
        return full_name

    # If we got here, just return the target as string
    return str(target) if target else str(origin_node)


def validate_ops(graph: GraphLowering) -> None:
    """Validate inputs to ops have same ElementArrangement.

    This pass need to be run after propagate_spyre_tensor_layouts so that it
    has all the required SpyreTensorLayouts.
    """
    for op in graph.operations:
        if not hasattr(op, "data"):
            continue
        if not isinstance(op.data, Pointwise):
            continue

        read_writes = op.get_read_writes()
        inputs = [r for r in read_writes.reads if isinstance(r, MemoryDep)]

        # Exclude single input ops
        if len(inputs) <= 1:
            continue

        layouts = []
        input_names = []
        for inp in inputs:
            buf = V.graph.get_buffer(inp.name)
            # Skip buffers without layouts
            if not hasattr(buf, "layouts"):
                continue
            for layout in buf.layouts:
                if isinstance(layout, SpyreTensorLayout):
                    layouts.append(layout)
                    input_names.append(inp.name)

        if len(layouts) <= 1:
            continue

        op_name = _get_op_name(op)

        # Check all layouts have the same element_arrangement
        stl_eas = [layout.element_arrangement for layout in layouts]

        # Skip ops with special ElementArrangement e.g. layernormnorm/scale with ElementArrangement.EXX2
        skip_ops = {"layernormnorm", "layernormscale"}
        skip_eas = {ElementArrangement.EXX2}
        if op_name in skip_ops and any(ea in skip_eas for ea in stl_eas):
            continue

        if len(set(stl_eas)) != 1:
            args_str = ", ".join(
                f'"{name}": {ea}' for name, ea in zip(input_names, stl_eas)
            )
            raise Unsupported(
                f"All inputs to an op must have same element arrangement, "
                f"op: {op_name}, args: {args_str}"
            )


def split_multi_ops(graph: GraphLowering):
    """Split multi-ops in a single loop body into separate buffers.

    1. Multi-op fusion: Splits multi-op loop bodies (e.g., type conversion + arithmetic)
       into individual ops by creating FX graph nodes and lowering.
    2. Constant arguments: creates SpyreConstantFallback IRNode.

    More pytorch programming patterns that result in multi-op in a single loop body can
    be supported by adding required handling in this pass.

    Special cases:
    1. Only processes ComputedBuffer with Pointwise data and FixedLayout
    2. Skips operations if only 1 compute ops (nothing to split)
    3. Skips if tracing fails or operation is not in operations list
    4. Requires valid FX node origins for graph manipulation
    5. Builds environment mapping from name_to_users for FX node lookup

    Algorithm:
    1. Build FX node environment from name_to_users
    2. For each eligible operation in graph.operations:
       a. Trace inner_fn to extract operation structure
       b. Filter to compute operations (exclude load/store/get_index)
       c. Propagate dtypes through the trace
       d. Create intermediate buffers for all but last operation
       e. Update final buffer to load from intermediates

    Args:
        graph: GraphLowering instance containing operations to process
    """
    gl = V.graph
    # Skip if in graph lowering context
    if not (hasattr(gl, "graph") and hasattr(gl, "run_node")):
        return

    # Build environment mapping FX nodes to TensorBox for node lookup
    env = {}
    for tbs in gl.name_to_users.values():
        for tb in tbs:
            if tb.data.origins:
                fx_node = next(iter(tb.data.origins))
                env[fx_node] = tb
    gl.env.update(env)

    operations = graph.operations
    for op in list(operations):
        if not (
            isinstance(op, ComputedBuffer)
            and isinstance(op.data, Pointwise)
            and isinstance(op.layout, FixedLayout)
        ):
            continue

        trace = _trace_inner_fn(op)
        if not trace:
            continue

        compute_ops = _get_compute_ops(trace)
        # Skip if nothing to split
        if len(compute_ops) <= 1:
            continue

        layout = op.layout
        dtype_map = _propagate_dtypes(trace, layout.dtype)
        bufname_map = _init_vid_to_bufname(trace)
        const_map = _init_vid_to_constant(trace)

        try:
            insert_idx = operations.index(op)
        except ValueError:
            continue

        # Skip if no FX graph origin node
        if not op.origins:
            continue

        orig_node = next(iter(op.origins))
        # Skip if orgin node is not FX graph node
        if not isinstance(orig_node, fx.Node):
            continue

        intermediate_ops, final_op = compute_ops[:-1], compute_ops[-1]
        _make_intermediate_bufs(
            intermediate_ops,
            dtype_map,
            bufname_map,
            layout,
            operations,
            insert_idx,
            gl,
            orig_node,
        )
        _update_original_buf(op, final_op, bufname_map, const_map, operations)
        logger.info(
            "split_multi_op: '%s' -> %d intermediate buffers",
            op.get_name(),
            len(intermediate_ops),
        )
