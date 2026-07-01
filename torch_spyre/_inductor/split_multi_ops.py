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

import collections
import dataclasses
import sympy
import torch
import torch.fx as fx
from torch._inductor.ir import ComputedBuffer, FixedLayout, Pointwise
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ops_handler import WrapperHandler
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

# Operations that involve dtype conversion
_DTYPE_OPS = frozenset({"to_dtype", "convert_element_type"})

# Operations whose scalar constant parameters must be passed as compile-time constants
# via op_info["constants"] (not as 1-d tensor buffers) in SDSC codegen.
# When the final op is one of these, intermediate constant ops must NOT be materialized
# as SpyreConstantFallback buffers — they must remain as V.ops.constant() scalars so
# SpyreOpFuncs.clamp / layernormscale / softplus can read them out of the RValue tree.
_OPS_WITH_CONSTANT_ARGS = frozenset({"clamp", "layernormscale", "softplus"})

# Mapping of operation names to their FX graph targets when there is no 1-1 mapping.
_OP_TARGET_TABLE = {
    "to_dtype": torch.ops.prims.convert_element_type.default,
    "convert_element_type": torch.ops.prims.convert_element_type.default,
}


class _SplitOpsHandler(WrapperHandler):
    """Redirect loads and materialized constants to intermediate buffers.

    After split_multi_ops materializes intermediates, this handler intercepts load() and
    constant() calls to reroute them. Index expressions are never touched — the original
    inner_fn's index computations are preserved exactly, avoiding stale-index bugs.
    """

    def __init__(
        self,
        inner,
        name_map: dict[str, str],
        constant_map: dict[tuple, str] | None = None,
    ):
        super().__init__(inner)
        self._name_map = name_map
        self._constant_map = constant_map or {}

    def load(self, name, index):
        return super().load(self._name_map.get(name, name), index)

    def constant(self, fill_value, dtype):
        key = (fill_value, dtype)
        if key in self._constant_map:
            # Load from the materialized constant buffer at index 0 (0-dim tensor)
            return super().load(self._constant_map[key], sympy.Integer(0))
        return super().constant(fill_value, dtype)


class _IntermediateOpHandler(WrapperHandler):
    """Replace materialized compute ops with buffer loads.

    When a compute op like to_dtype is materialized into a buffer, this handler intercepts
    the original op call and returns a load from that buffer instead. Tracks _last_index
    across loads so the returned load uses the correct (current) iteration index.

    For multiple ops of the same name, op_queues uses deques to consume materialized
    buffers in trace order (deterministic replay of original inner_fn).
    """

    def __init__(self, inner, op_queues: dict[str, collections.deque]):
        super().__init__(inner)
        self._op_queues = op_queues
        self._last_index = sympy.Integer(0)

    def load(self, name, index):
        self._last_index = index
        return super().load(name, index)

    def _default(self, name, args, kwargs):
        queue = self._op_queues.get(name)
        if queue:
            buf_name = queue.popleft()
            return super().load(buf_name, self._last_index)
        return super()._default(name, args, kwargs)


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
    1. _DTYPE_OPS: Ensure dtype is first positional arg after input
    2. Positional args stored with '_p' prefix are extracted and appended
    3. Internal kwargs (starting with '_') are filtered out

    Args:
        op_name: Name of the operation
        input_fx_nodes: List of input FX nodes
        kwargs: Operation keyword arguments (may include _p0, _p1, etc.)
        out_dtype: Default output dtype
        device: Optional device

    Returns:
        Tuple of (args, clean_kwargs, final_dtype)
    """
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
    final_op_name="",
) -> tuple[dict[tuple, str], dict[str, collections.deque]]:
    """Create intermediate buffers for operations.

    For each intermediate operation in a fused computation, this creates:
    1. An FX graph node with the appropriate target and arguments
    2. A lowered buffer from that node
    3. Updates vid_to_bufname mapping for subsequent operations

    Constants that are direct scalar inputs to _OPS_WITH_CONSTANT_ARGS (clamp, softplus,
    layernormscale) are skipped — they must remain as V.ops.constant() scalars so that
    SpyreOpFuncs can place them in op_info["constants"] for SDSC codegen.  All other
    constants are materialized as SpyreConstantFallback buffers via
    torch.ops.spyre.constant.default.

    Args:
        intermediate_ops: List of (op_name, vid, inputs, kwargs) tuples
        vid_to_dtype: Mapping from value ID to dtype
        vid_to_bufname: Mapping from value ID to buffer name (updated)
        layout: Layout of the original fused operation
        operations: List of operations to insert into
        insert_idx: Starting index for insertion
        gl: GraphLowering instance
        orig_node: Original FX node being split
        final_op_name: Name of the final (consumer) op, used to skip constant
            materialization for _OPS_WITH_CONSTANT_ARGS.

    Returns:
        constant_map: Mapping from (fill_value, dtype) to materialized constant buffer name
        op_queues: Mapping from op_name to deque of materialized buffer names in trace order
    """
    constant_map: dict[tuple[float, torch.dtype], str] = {}
    op_queues: dict[str, collections.deque] = {}
    for op_name, vid, inputs, kwargs in intermediate_ops:
        out_dtype = vid_to_dtype.get(vid, layout.dtype)

        if op_name == "constant":
            # Constants that feed _OPS_WITH_CONSTANT_ARGS ops (clamp, softplus,
            # layernormscale) must remain as Python scalars — do not materialize.
            if final_op_name in _OPS_WITH_CONSTANT_ARGS:
                continue
            # All other constants: lower as SpyreConstantFallback via the registered
            # torch.ops.spyre.constant lowering so they can be loaded from a buffer.
            fill_value = float(kwargs["fill_value"])
            out_dtype = kwargs.get("dtype", layout.dtype)
            # Reuse an already-materialized buffer for duplicate (value, dtype) pairs
            # rather than creating a second identical SpyreConstantFallback.
            if (fill_value, out_dtype) in constant_map:
                vid_to_bufname[vid] = constant_map[(fill_value, out_dtype)]
                continue
            with gl.graph.inserting_before(orig_node):
                new_node = gl.graph.create_node(
                    "call_function",
                    torch.ops.spyre.constant.default,
                    (fill_value, out_dtype, layout.device),
                    {},
                )
            new_node.meta["val"] = torch.tensor(
                fill_value, dtype=out_dtype, device="meta"
            )
            # Lower the FX node; run_node registers in gl.operations.
            # For ExternKernel like SpyreConstantFallback, extract the raw buffer
            # only for positioning — don't replace it with raw buffer in operations.
            tb = gl.run_node(new_node)
            # tb.data.data is the SpyreConstantFallback, but keep it wrapped as TensorBox
            # in operations to preserve proper attribute initialization.
            new_buf = tb.data.data
            # Set origins using object.__setattr__ to work around dataclass frozen fields.
            object.__setattr__(new_buf, "origins", OrderedSet([new_node]))
            # Move the buffer to our desired position, then continue with it wrapped.
            gl.operations.remove(new_buf)
            operations.insert(insert_idx, new_buf)
            buf_name = new_buf.get_name()
            vid_to_bufname[vid] = buf_name
            # Track this materialized constant for the handler's constant_map.
            constant_map[(fill_value, out_dtype)] = buf_name
            insert_idx += 1
            continue

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

        # Lower the FX node; _lower_fx_node extracts, removes, and reinserts.
        new_buf = _lower_fx_node(new_node, gl, operations, insert_idx)
        # Set origins using object.__setattr__ to work around dataclass frozen fields.
        object.__setattr__(new_buf, "origins", OrderedSet([new_node]))
        buf_name = new_buf.get_name()
        vid_to_bufname[vid] = buf_name
        # Track materialized compute intermediates so _IntermediateOpHandler can
        # intercept inline calls to this op and load from the buffer instead.
        # Use a deque so multiple ops of the same name are consumed in trace order.
        if op_name not in op_queues:
            op_queues[op_name] = collections.deque()
        op_queues[op_name].append(buf_name)
        insert_idx += 1

    return constant_map, op_queues


def _patch_original_buf(
    op: ComputedBuffer,
    orig_load_names: dict[int, str],
    vid_to_bufname: dict[int, str],
    operations: list,
    constant_map: dict[tuple, str] | None = None,
    op_queues: dict[str, collections.deque] | None = None,
    load_redirects: dict[int, str] | None = None,
) -> None:
    """Wrap original inner_fn with handlers to use intermediate buffers.

    Builds a handler stack that preserves all original index logic while redirecting
    buffer loads and ops to materialized intermediates:
    - _SplitOpsHandler: redirects load names and materialized constants to buffers
    - _IntermediateOpHandler: intercepts materialized ops and returns buffer loads

    Args:
        op: ComputedBuffer to patch
        orig_load_names: vid → buffer name (from original trace loads)
        vid_to_bufname: vid → current buffer name (updated by _make_intermediate_bufs)
        operations: list of buffers in graph
        constant_map: (fill_value, dtype) → buf_name for materialized constants
        op_queues: op_name → deque[buf_name] to intercept materialized ops
        load_redirects: input_vid → output_buf_name for compute intermediates.
            Redirects original input buffers to intermediate output buffers so
            validate_ops sees uniform ElementArrangement across inputs.
    """
    # Build name_map from updated buffer names. Load redirects for compute intermediates
    # ensure the original input buffers are no longer read dependencies of the final op,
    # which fixes validate_ops ElementArrangement checks.
    name_map = {
        orig_load_names[v]: vid_to_bufname[v]
        for v in orig_load_names
        if vid_to_bufname.get(v) != orig_load_names[v]
    }
    for v_in, new_buf in (load_redirects or {}).items():
        name_map[orig_load_names[v_in]] = new_buf

    orig_inner = op.data.inner_fn
    _cm = constant_map or {}
    _oq_snapshot = {k: collections.deque(v) for k, v in (op_queues or {}).items()}

    def new_inner_fn(*args, _nm=name_map, _orig=orig_inner):
        # inner_fn is called multiple times (e.g., get_default_sizes_body). Each call
        # needs a fresh copy of the deques so popleft() resets for each replay of the trace.
        fresh_queues = {k: collections.deque(v) for k, v in _oq_snapshot.items()}
        inner = _SplitOpsHandler(V.ops, _nm, _cm)
        with V.set_ops_handler(_IntermediateOpHandler(inner, fresh_queues)):
            return _orig(*args)

    new_data = dataclasses.replace(op.data, inner_fn=new_inner_fn)
    # Preserve origins from the original data object (dataclasses.replace creates
    # a fresh object with empty origins; we must restore it).
    object.__setattr__(new_data, "origins", op.data.origins)
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

    Splits fused multi-op loop bodies (e.g., type conversion + clamp) into individual
    ComputedBuffer ops by creating FX graph nodes, lowering them, and patching the
    original buffer's inner_fn to load from the new intermediates.

    Scalar constants are materialized as SpyreConstantFallback buffers via
    torch.ops.spyre.constant.default.

    The final (original) buffer's inner_fn is wrapped with _SplitOpsHandler rather
    than rebuilt from scratch.  This preserves all index computations exactly — only
    buffer names are redirected — avoiding stale-index substitution bugs.

    Algorithm:
    1. Build FX node environment from name_to_users.
    2. For each eligible ComputedBuffer (Pointwise, FixedLayout):
       a. Trace inner_fn to detect multi-op structure.
       b. Skip if only 1 compute op.
       c. Create intermediate buffers (including SpyreConstantFallback for constants).
       d. Wrap the original inner_fn with _SplitOpsHandler to redirect buffer names.

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

        intermediate_ops = compute_ops[:-1]
        final_op_name = compute_ops[-1][0]

        logger.debug(
            "split_multi_ops: '%s' has %d compute ops: [%s] -> final: %s",
            op.get_name(),
            len(compute_ops),
            ", ".join(op_name for op_name, *_ in intermediate_ops),
            final_op_name,
        )

        # Snapshot the original load-buffer names before _make_intermediate_bufs
        # updates bufname_map.  This lets _patch_original_buf build the redirect
        # map: orig_name → new_intermediate_name.
        orig_load_names = dict(bufname_map)

        constant_map, op_queues = _make_intermediate_bufs(
            intermediate_ops,
            dtype_map,
            bufname_map,
            layout,
            operations,
            insert_idx,
            gl,
            orig_node,
            final_op_name=final_op_name,
        )

        # For compute intermediates, build redirects mapping input load vids to output
        # buffers. This removes original inputs from read dependencies, fixing
        # validate_ops when ops change dtype (e.g. add(x_fp32, to_dtype(y_fp16→fp32))).
        load_redirects: dict[int, str] = {}
        for op_name, op_vid, op_inputs, _ in intermediate_ops:
            if op_name == "constant":
                continue
            for v_in in op_inputs:
                if v_in in orig_load_names:
                    load_redirects[v_in] = bufname_map[op_vid]

        _patch_original_buf(
            op,
            orig_load_names,
            bufname_map,
            operations,
            constant_map=constant_map,
            op_queues=op_queues,
            load_redirects=load_redirects,
        )
        logger.info(
            "split_multi_op: '%s' -> %d intermediate buffers",
            op.get_name(),
            len(intermediate_ops),
        )
