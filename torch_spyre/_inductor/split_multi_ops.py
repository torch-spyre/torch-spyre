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
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    Reduction,
    Scatter,
)
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch.fx.traceback import NodeSource, NodeSourceAction
from torch._inductor.ops_handler import WrapperHandler
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from .logging_utils import get_inductor_logger
from .loop_info import ReadCopyElisionRecord
from .errors import Unsupported
from .pass_utils import (
    find_fx_node,
    indirect_info_from_op,
    origin_in_graph,
    patch_env,
    redirect_computed_buffer_reads,
    replace_computed_buffer_body,
)
from .constants import is_ea_compatible, STAGGERED_EAS, SUPPORTED_EAS
from .dtype_ops import DtypeOpTable
from .optimize_restickify import AllSameNode, AnyInNode
from .propagate_layouts import _get_prop_args
from .provenance import decompose_provenance
from torch_spyre._C import (
    SpyreTensorLayout,
    ElementArrangement,
    get_device_size_in_bytes,
)
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


def _is_invalid_compute_op(op) -> bool:
    if not isinstance(op, ComputedBuffer):
        return True
    if not isinstance(op.data, (Pointwise, Reduction)):
        return True
    if not isinstance(op.layout, (FixedLayout, MutationLayoutSHOULDREMOVE)):
        return True

    return False


# In case of reduction op with single to_dtype e.g. torch.sum(t, dtype=dst_dtype)
def _is_reduction_with_to_dtype(op, compute_ops) -> bool:
    is_reduction = isinstance(op.data, Reduction)
    has_type_conversion = any(record[0] in _DTYPE_OPS for record in compute_ops)
    return is_reduction and has_type_conversion


def _skip_splitting(op, compute_ops) -> bool:
    if _is_reduction_with_to_dtype(op, compute_ops):
        return False

    if len(compute_ops) > 1:
        return False

    return True


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


def _set_split_child_provenance(
    orig_node: fx.Node,
    new_node: fx.Node,
    target: torch.fx.node.Target,
) -> None:
    """Attach parent source lineage and the child's semantic op identity."""
    if "stack_trace" in orig_node.meta:
        new_node.meta["stack_trace"] = orig_node.meta["stack_trace"]

    # This lower-IR pass creates FX nodes after upstream's graph-pass observer
    # has finished, so record the exact derivation explicitly. NodeSource nests
    # the parent's prior from_node chain rather than flattening or duplicating it.
    new_node.meta["from_node"] = [
        NodeSource(
            orig_node,
            "split_multi_ops",
            NodeSourceAction.CREATE,
        )
    ]

    # The source line belongs to the parent, while the operation identity belongs
    # to this distinct decomposed child.
    new_node.meta["original_aten"] = target


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


def _index_syms(n: int) -> tuple[sympy.Symbol, ...]:
    """The index symbols _trace_inner_fn passes to a traced inner_fn."""
    return tuple(sympy.Symbol(f"_i{k}") for k in range(n))


def _trace_inner_fn(op):
    """Trace the inner_fn of an operation to extract its structure.

    Returns None if tracing fails (e.g., unsupported ops)

    Args:
        op: ComputedBuffer operation to trace

    Returns:
        List of traced operations or None if tracing failed
    """
    ranges = op.data.ranges
    syms = _index_syms(len(ranges))
    tracer = _TracingHandler(V.ops)
    try:
        with tracer:
            if isinstance(op.data, Reduction):
                r_ranges = op.data.reduction_ranges
                r_syms = tuple(sympy.Symbol(f"_r{k}") for k in range(len(r_ranges)))
                # Reduction inner_fn expects (index, rindex)
                op.data.inner_fn(syms, r_syms)
            else:
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
    gl.env[node] = tb  # run_node does not record it in the env
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
    orig_buffer,
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
        orig_buffer: Original ComputedBuffer being split
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
            _set_split_child_provenance(orig_node, new_node, new_node.target)
            new_node.meta["val"] = torch.tensor(
                fill_value, dtype=out_dtype, device="meta"
            )
            # Lower the FX node; run_node registers in gl.operations.
            # For ExternKernel like SpyreConstantFallback, extract the raw buffer
            # only for positioning — don't replace it with raw buffer in operations.
            tb = gl.run_node(new_node)
            gl.env[new_node] = tb  # run_node does not record it in the env
            # tb.data.data is the SpyreConstantFallback, but keep it wrapped as TensorBox
            # in operations to preserve proper attribute initialization.
            new_buf = tb.data.data
            # This child node is the sole semantic origin of the new buffer.
            object.__setattr__(new_buf, "origins", OrderedSet([new_node]))
            decompose_provenance(
                orig_buffer,
                (new_buf,),
                pass_name="split_multi_ops",
                reason=f"materialize {op_name}",
                inherit_origins=False,
            )
            # Move the buffer to our desired position, then continue with it wrapped.
            gl.operations.remove(new_buf)
            operations.insert(insert_idx, new_buf)
            buf_name = new_buf.get_name()
            vid_to_bufname[vid] = buf_name
            # Track this materialized constant for the handler's constant_map.
            constant_map[(fill_value, out_dtype)] = buf_name
            insert_idx += 1
            continue

        def assert_fx_node(v):
            if (_node := find_fx_node(vid_to_bufname[v], gl)) is not None:
                return _node
            raise KeyError(f"No FX node for {vid_to_bufname[v]}")

        input_nodes = [assert_fx_node(v) for v in inputs]
        target = _resolve_fx_target(op_name)
        if target is None:
            raise RuntimeError(f"Cannot resolve target for '{op_name}'")

        args, clean_kw, out_dtype = _normalize_op_args(
            op_name, input_nodes, kwargs, out_dtype, layout.device
        )

        with gl.graph.inserting_before(orig_node):
            new_node = gl.graph.create_node("call_function", target, args, clean_kw)
        _set_split_child_provenance(orig_node, new_node, target)

        # Propagate metadata for shape inference
        if input_nodes and "val" in input_nodes[0].meta:
            new_node.meta["val"] = input_nodes[0].meta["val"].to(out_dtype)
        elif "val" in orig_node.meta:
            new_node.meta["val"] = orig_node.meta["val"].to(out_dtype)

        # Lower the FX node; _lower_fx_node extracts, removes, and reinserts.
        new_buf = _lower_fx_node(new_node, gl, operations, insert_idx)
        # This child node is the sole semantic origin of the new buffer.
        object.__setattr__(new_buf, "origins", OrderedSet([new_node]))
        decompose_provenance(
            orig_buffer,
            (new_buf,),
            pass_name="split_multi_ops",
            reason=f"materialize {op_name}",
            inherit_origins=False,
        )
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
    read_copy_record = getattr(op, "_read_copy_elision_record", None)
    _cm = constant_map or {}
    _oq_snapshot = {k: collections.deque(v) for k, v in (op_queues or {}).items()}

    def new_inner_fn(*args, _nm=name_map, _orig=orig_inner):
        # inner_fn is called multiple times (e.g., get_default_sizes_body). Each call
        # needs a fresh copy of the deques so popleft() resets for each replay of the trace.
        fresh_queues = {k: collections.deque(v) for k, v in _oq_snapshot.items()}
        inner = _SplitOpsHandler(V.ops, _nm, _cm)
        with V.set_ops_handler(_IntermediateOpHandler(inner, fresh_queues)):
            return _orig(*args)

    direct_inner_fn = None
    if isinstance(read_copy_record, ReadCopyElisionRecord):

        def direct_inner_fn(
            *args,
            _nm=name_map,
            _orig=read_copy_record.direct_inner_fn,
        ):
            fresh_queues = {k: collections.deque(v) for k, v in _oq_snapshot.items()}
            inner = _SplitOpsHandler(V.ops, _nm, _cm)
            with V.set_ops_handler(_IntermediateOpHandler(inner, fresh_queues)):
                return _orig(*args)

    new_data = dataclasses.replace(op.data, inner_fn=new_inner_fn)
    # Preserve origins from the original data object (dataclasses.replace creates
    # a fresh object with empty origins; we must restore it).
    object.__setattr__(new_data, "origins", op.data.origins)
    new_op = replace_computed_buffer_body(
        op,
        new_data,
        operations,
        pass_name="split_multi_ops",
        reason="rewrite original buffer body",
    )
    if isinstance(read_copy_record, ReadCopyElisionRecord):
        new_op._read_copy_elision_record = dataclasses.replace(  # type: ignore[attr-defined]
            read_copy_record,
            direct_inner_fn=direct_inner_fn,
        )
    V.graph.name_to_buffer[new_op.get_name()] = new_op


def _get_op_name(op) -> str:
    """Extract the operation name from a node for validation."""
    if not hasattr(op.data, "origins") or not op.data.origins:
        return ""

    # Prefer the origin in the current lowering graph (subgraph-safe); fall back
    # to an arbitrary origin when no graph context is available. That fallback
    # may yield a parent-graph node for a subgraph buffer, which is safe *here*
    # because this function only extracts a name string and never inserts into
    # or mutates a graph. Do not gate graph-mutating logic on the result.
    gl = V.graph
    origin_node = None
    if hasattr(gl, "graph"):
        origin_node = origin_in_graph(op.data.origins, gl.graph)
    if origin_node is None:
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


def _find_fx_node(graph: GraphLowering, buf) -> fx.Node | None:
    """Return the FX node (by name match on ``buf.origins``) that produced ``buf``."""
    for origin in getattr(buf, "origins", ()):
        for node in graph.graph.nodes:
            if node.name == origin.name:
                return node
    return None


def _insert_shuffle_op(
    graph: GraphLowering, op, stg_name: str, stg_buf, target_ea: ElementArrangement
) -> ComputedBuffer | None:
    """Insert an identity copy of ``stg_buf`` that re-arranges it to ``target_ea``.

    The copy is lowered as a plain ``aten.clone``; its output layout (the same
    device geometry with ``target_ea``) is set explicitly below, and codegen
    emits a ``shuffle`` because the input and output element arrangements differ.
    The new buffer is tagged ``ea_shuffle`` so ``optimize_shuffling`` can find
    it later, and ``op`` (rebuilt and returned) is redirected to read it.
    """

    # Find the origin FX node from the operation
    orig_node = origin_in_graph(op.data.origins, graph.graph)
    if orig_node is None:
        logger.warning(f"Cannot find origin node for op {op.get_name()}")
        return None

    try:
        insert_idx = graph.operations.index(op)
    except ValueError:
        insert_idx = 0

    # Find the FX node that produces the staggered buffer
    stg_fx_node = _find_fx_node(graph, stg_buf)

    if stg_fx_node is None:
        logger.warning(
            f"Cannot find FX node for buffer {stg_name} among origins {[o.name for o in stg_buf.origins]}"
        )
        return None

    shuffle_op = torch.ops.aten.clone.default
    with graph.graph.inserting_before(orig_node):
        shuffle_node = graph.graph.create_node(
            "call_function",
            shuffle_op,
            (stg_fx_node,),
            {},
        )

    # Copy metadata
    if "val" in stg_fx_node.meta:
        shuffle_node.meta["val"] = stg_fx_node.meta["val"]
    if "tensor_meta" in stg_fx_node.meta:
        shuffle_node.meta["tensor_meta"] = stg_fx_node.meta["tensor_meta"]

    # Set provenance
    _set_split_child_provenance(orig_node, shuffle_node, shuffle_op)

    # Lower the shuffle FX node into a buffer
    new_buf = _lower_fx_node(shuffle_node, graph, graph.operations, insert_idx)

    if new_buf is None:
        logger.warning(f"Failed to lower shuffle node for {stg_name}")
        return None

    object.__setattr__(new_buf, "origins", OrderedSet([shuffle_node]))

    # Set layouts on new_buf
    stg_stl = next(iter(stg_buf.layouts))
    out_stl = SpyreTensorLayout(
        stg_stl.device_size,
        stg_stl.stride_map,
        stg_stl.device_dtype,
        target_ea,
    )
    new_buf.layouts = [out_stl]
    new_buf.restick_cost_fn = AnyInNode.from_args()
    new_buf.ea_shuffle = True

    # Update provenance
    decompose_provenance(
        stg_buf,
        (new_buf,),
        pass_name="split_multi_ops",
        reason="insert EA shuffle",
        inherit_origins=False,
    )

    # Redirect the staggered buffer load to the new shuffled buffer
    name_map = {stg_name: new_buf.get_name()}
    orig_inner = op.data.inner_fn

    def new_inner_fn(*args, _nm=name_map, _orig=orig_inner):
        inner = _SplitOpsHandler(V.ops, _nm)
        with V.set_ops_handler(inner):
            return _orig(*args)

    new_data = dataclasses.replace(op.data, inner_fn=new_inner_fn)
    object.__setattr__(new_data, "origins", op.data.origins)

    new_op = replace_computed_buffer_body(
        op,
        new_data,
        graph.operations,
        pass_name="split_multi_ops",
        reason="insert EA shuffle redirect",
    )
    graph.name_to_buffer[new_op.get_name()] = new_op

    if hasattr(op, "layouts"):
        new_op.layouts = list(op.layouts)
        rw = new_op.get_read_writes()
        output_dep = next(iter(rw.writes))
        args = _get_prop_args(rw.reads)
        new_op.restick_cost_fn = AllSameNode.from_args(
            args, new_op.layouts, output_dep, new_op
        )

    logger.info(f"Successfully inserted shuffle: {stg_name} -> {new_buf.get_name()}")
    return new_op


def _has_indirect_access(op, read_writes, indirect_names) -> bool:
    """True if ``op`` is an indirect scatter/gather (index read or index write)."""
    if isinstance(op.data, Scatter) or indirect_names:
        return True
    return any(
        isinstance(d, MemoryDep)
        and any(str(s).startswith(("indirect", "tmp")) for s in d.index.free_symbols)
        for d in (*read_writes.reads, *read_writes.writes)
    )


def validate_ops(graph: GraphLowering) -> None:
    """Validate inputs to ops have same ElementArrangement.

    This pass needs to be run after propagate_spyre_tensor_layouts so that it
    has all the required SpyreTensorLayouts.
    """
    for op in list(graph.operations):
        if not hasattr(op, "data"):
            continue
        if not isinstance(op.data, Pointwise):
            continue

        read_writes = op.get_read_writes()
        indirect_names, _, _ = indirect_info_from_op(op)

        # Indirect scatter/gather never gets an EA shuffle: the index operand has
        # no arrangement to match, and shuffling the data operand would un-stagger
        # it while the output keeps its staggered tag.
        if _has_indirect_access(op, read_writes, indirect_names):
            continue

        inputs = [
            r
            for r in read_writes.reads
            if isinstance(r, MemoryDep) and r.name not in indirect_names
        ]

        # Exclude single input ops
        if len(inputs) <= 1:
            continue

        # Collect ONE (layout) per unique input buffer
        buf_stls = {}  # buffer_name -> SpyreTensorLayout
        for inp in inputs:
            buf = V.graph.get_buffer(inp.name)
            if not hasattr(buf, "layouts"):
                continue

            # Get the first SpyreTensorLayout for this buffer
            for layout in buf.layouts:
                if isinstance(layout, SpyreTensorLayout):
                    if inp.name not in buf_stls:
                        buf_stls[inp.name] = layout
                    break

        if len(buf_stls) <= 1:
            continue

        input_names = list(buf_stls.keys())
        stl_eas = [stl.element_arrangement for stl in buf_stls.values()]

        op_name = _get_op_name(op)

        # Skip ops with special ElementArrangement e.g. layernormnorm/scale with ElementArrangement.EXX2
        skip_ops = {"layernormnorm", "layernormscale"}
        skip_eas = {ElementArrangement.EXX2}
        if op_name in skip_ops and any(ea in skip_eas for ea in stl_eas):
            continue

        # Valid EA patterns (see is_ea_compatible):
        # 1. All operands share one EA.
        # 2. Exactly one distinct non-STANDARD EA (not EXX2) mixed with STANDARD
        #    operands (the broadcast pattern).
        if not is_ea_compatible(stl_eas):
            args_str = ", ".join(
                f'"{name}": {ea}' for name, ea in zip(input_names, stl_eas)
            )
            raise Unsupported(
                f"Spyre backend does not support: Incompatible ElementArrangement in multi-arg op. "
                f"Valid patterns: all inputs share one EA, or one non-STANDARD EA broadcast against "
                f"STANDARD inputs. op: {op_name}, args: {args_str}"
            )

        # Mixed STANDARD + supported staggered EA where neither side is a
        # stick broadcast (case 3.3 in _multi_arg_pointwise_layouts): shuffle
        # the staggered operand to STANDARD so the op sees one arrangement.
        # When either side broadcasts at the stick dim the stagger is
        # unobservable and no shuffle is needed.
        #
        # buf_stls holds each input's first *candidate* layout; the candidates
        # are not resolved until finalize_layouts, and the broadcast pruning done
        # in _multi_arg_pointwise_layouts (case 3.1) only touches the consumer's
        # copy. So an input can look non-broadcast here although the op was
        # laid out with it broadcasting. The op's own output EA records the
        # decision: it is staggered when the STANDARD operands broadcast (case
        # 3.1), and shuffling then would compute in STANDARD order while the
        # output is still tagged staggered. Only shuffle when the op produces
        # STANDARD.
        out_staggered = any(
            isinstance(layout, SpyreTensorLayout)
            and layout.element_arrangement in SUPPORTED_EAS
            for layout in getattr(op, "layouts", ())
        )
        if ElementArrangement.STANDARD in stl_eas and not out_staggered:
            std_names = [
                n
                for n, s in buf_stls.items()
                if s.element_arrangement == ElementArrangement.STANDARD
            ]
            std_has_real_stick = any(
                buf_stls[n].stride_map[-1] != -1 for n in std_names
            )
            for stg_name, stg_stl in list(buf_stls.items()):
                if (
                    stg_stl.element_arrangement in SUPPORTED_EAS
                    and stg_stl.stride_map[-1] != -1
                    and std_has_real_stick
                ):
                    new_op = _insert_shuffle_op(
                        graph,
                        op,
                        stg_name,
                        V.graph.get_buffer(stg_name),
                        ElementArrangement.STANDARD,
                    )
                    if new_op is None:
                        raise Unsupported(
                            f"Failed to insert EA shuffle for buffer {stg_name} "
                            f"in op {op_name}"
                        )
                    op = new_op


def _reader_ops(graph: GraphLowering, name: str) -> list:
    """Operations that read buffer ``name`` through a ``MemoryDep``."""
    readers = []
    for op in graph.operations:
        if op.get_name() == name:
            continue
        try:
            reads = op.get_read_writes().reads
        except Exception:  # ops without read/write metadata cannot read ``name``
            continue
        if any(isinstance(d, MemoryDep) and d.name == name for d in reads):
            readers.append(op)
    return readers


def _can_redirect_readers(readers: list) -> bool:
    """True if every reader can be re-pointed to another buffer in place.

    A reader is redirectable when it is a plain ``ComputedBuffer`` whose layout
    cost function can be rebuilt from its (new) inputs: ``AllSameNode`` (rebuilt
    from the new input layouts) or ``AnyInNode`` (input independent). Matmul,
    conv and mutation consumers carry layout state tied to the exact producer,
    so they are left alone and the shuffle stays.
    """
    for op in readers:
        if not isinstance(op, ComputedBuffer):
            return False
        if isinstance(op.layout, MutationLayoutSHOULDREMOVE):
            return False
        if isinstance(op.data, Scatter):
            return False
        cost_fn = getattr(op, "restick_cost_fn", None)
        if cost_fn is not None and not isinstance(cost_fn, (AllSameNode, AnyInNode)):
            return False
    return True


def _redirect_readers(graph: GraphLowering, readers: list, old: str, new: str) -> None:
    """Make every op in ``readers`` read ``new`` instead of ``old``.

    ``new`` must hold the same value with the same candidate layouts as ``old``
    (callers check this), so the readers' own ``layouts`` are unchanged. Only the
    cost function is rebuilt, because its edges are keyed by input buffer name.
    The input is rewired by wrapping ``inner_fn`` (never by rebuilding indices).
    """
    for op in readers:
        layouts = getattr(op, "layouts", None)
        cost_fn = getattr(op, "restick_cost_fn", None)
        new_op = redirect_computed_buffer_reads(
            op,
            {old: new},
            graph.operations,
            pass_name="optimize_shuffling",
            reason="redirect reads past a removed EA shuffle",
        )
        if layouts is not None:
            new_op.layouts = list(layouts)
        if isinstance(cost_fn, AllSameNode):
            rw = new_op.get_read_writes()
            new_op.restick_cost_fn = AllSameNode.from_args(
                _get_prop_args(rw.reads),
                new_op.layouts,
                next(iter(rw.writes)),
                new_op,
            )
        elif cost_fn is not None:
            new_op.restick_cost_fn = cost_fn


def _remove_op(graph: GraphLowering, op) -> None:
    """Drop ``op`` from the graph the way ``deadcode_elimination`` does."""
    graph.removed_buffers.add(op.get_name())
    graph.operations.remove(op)


def _single_layout(buf) -> SpyreTensorLayout | None:
    """The one committed-candidate layout of ``buf``, or None if there are several."""
    layouts = getattr(buf, "layouts", None)
    if layouts is None or len(layouts) != 1:
        return None
    return layouts[0]


def _shuffle_source(op) -> str | None:
    """Name of the buffer a shuffle copy reads, or None if it is not a plain copy."""
    reads = [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]
    return reads[0].name if len(reads) == 1 else None


def _try_drop_shuffle(graph: GraphLowering, op, outputs: set[str]) -> bool:
    """Techniques 1-3: remove a dead, no-op or collapsible shuffle."""
    name = op.get_name()
    src_name = _shuffle_source(op)
    out_stl = _single_layout(op)
    if src_name is None or out_stl is None:
        return False
    src = graph.get_buffer(src_name)
    src_stl = _single_layout(src)
    readers = _reader_ops(graph, name)

    # 1. Dead shuffle: nothing reads it and it is not a graph output.
    if not readers and name not in outputs:
        _remove_op(graph, op)
        return True

    # 2. No-op shuffle: the source already has the target arrangement and the same
    #    device geometry, so the copy moves nothing.
    if (
        name not in outputs
        and src_stl is not None
        and src_stl == out_stl
        and _can_redirect_readers(readers)
    ):
        _redirect_readers(graph, readers, name, src_name)
        _remove_op(graph, op)
        return True

    # 3. Shuffle of a shuffle: ``A -S1-> B -S2-> C`` becomes ``A -S2-> C``. If A
    #    already has C's arrangement the pair cancels and C's readers read A.
    if (
        getattr(src, "ea_shuffle", False)
        and src_name not in outputs
        and _can_redirect_readers([op])
    ):
        a_name = _shuffle_source(src)
        a_stl = _single_layout(graph.get_buffer(a_name)) if a_name else None
        if a_name is not None and a_stl is not None:
            if a_stl == out_stl and name not in outputs:
                if _can_redirect_readers(readers):
                    _redirect_readers(graph, readers, name, a_name)
                    _remove_op(graph, op)
                    return True
            elif (
                a_stl.device_size == out_stl.device_size
                and a_stl.stride_map == out_stl.stride_map
                and a_stl.device_dtype == out_stl.device_dtype
            ):
                _redirect_readers(graph, [op], src_name, a_name)
                return True
    return False


def _try_merge_duplicates(graph: GraphLowering, shuffles: list, outputs: set[str]):
    """Technique 4: keep one shuffle per (source, target layout)."""
    seen: dict = {}
    for op in shuffles:
        src_name, out_stl = _shuffle_source(op), _single_layout(op)
        if src_name is None or out_stl is None:
            continue
        key = (
            src_name,
            out_stl.element_arrangement,
            tuple(out_stl.device_size),
            tuple(out_stl.stride_map),
        )
        keep = seen.setdefault(key, op)
        if keep is op or op.get_name() in outputs:
            continue
        readers = _reader_ops(graph, op.get_name())
        if _can_redirect_readers(readers):
            _redirect_readers(graph, readers, op.get_name(), keep.get_name())
            _remove_op(graph, op)
            return True
    return False


def _try_sink_shuffle_through_cast(graph: GraphLowering, op, outputs: set[str]) -> bool:
    """Technique 5: shuffle the narrow side of a cast instead of the wide one.

    Rewrites ``x -upcast-> s (staggered) -shuffle-> t (STANDARD)`` into
    ``x -shuffle-> x' -upcast-> t``. The upcast reads ``x'`` in the arrangement
    for which the hardware conversion map already yields STANDARD (see
    ``DtypeOpTable.ea_map``), so the staggering is undone by the cast that is
    executed anyway. The shuffle now moves fp16 (2 B/element) instead of fp32
    (4 B/element), halving its traffic; the cast and its output layout are
    unchanged, so ``t``'s readers need no layout update.
    """
    name = op.get_name()
    s_name = _shuffle_source(op)
    t_stl = _single_layout(op)
    if s_name is None or t_stl is None or name in outputs:
        return False
    if t_stl.element_arrangement != ElementArrangement.STANDARD:
        return False
    cast = graph.get_buffer(s_name)
    if (
        not isinstance(cast, ComputedBuffer)
        or not isinstance(cast.data, Pointwise)
        or s_name in outputs
    ):
        return False
    s_stl = _single_layout(cast)
    if s_stl is None or s_stl.element_arrangement not in STAGGERED_EAS:
        return False

    # ``cast`` must be a whole-tensor, single-input conversion with no other user.
    rw = cast.get_read_writes()
    reads = [d for d in rw.reads if isinstance(d, MemoryDep)]
    if len(reads) != 1 or reads[0].index != next(iter(rw.writes)).index:
        return False
    if [r.get_name() for r in _reader_ops(graph, s_name)] != [name]:
        return False
    x_name = reads[0].name
    x = graph.get_buffer(x_name)
    x_stl = _single_layout(x)
    if x_stl is None or x_stl.element_arrangement != ElementArrangement.STANDARD:
        return False
    if x.get_dtype() == cast.get_dtype():
        return False

    # Only the fp16 -> fp32 direction narrows the shuffled data.
    if get_device_size_in_bytes(x_stl) >= get_device_size_in_bytes(s_stl):
        return False

    # Find the input arrangement for which the cast itself produces STANDARD.
    e_in = None
    for candidate in STAGGERED_EAS:
        try:
            produced = DtypeOpTable.ea_map(x.get_dtype(), cast.get_dtype(), candidate)
        except Unsupported:
            continue
        if produced == ElementArrangement.STANDARD:
            e_in = candidate
            break
    if e_in is None:
        return False

    readers = _reader_ops(graph, name)
    if not _can_redirect_readers(readers):
        return False

    # Insert ``x' = shuffle(x, e_in)`` in front of the cast and re-point the cast.
    new_cast = _insert_shuffle_op(graph, cast, x_name, x, e_in)
    if new_cast is None:
        return False
    # The cast now yields STANDARD, which is exactly what ``op`` used to hold.
    new_cast.layouts = list(op.layouts)
    rw = new_cast.get_read_writes()
    new_cast.restick_cost_fn = AllSameNode.from_args(
        _get_prop_args(rw.reads),
        new_cast.layouts,
        next(iter(rw.writes)),
        new_cast,
    )
    _redirect_readers(graph, readers, name, new_cast.get_name())
    _remove_op(graph, op)
    return True


def optimize_shuffling(graph: GraphLowering) -> None:
    """Minimize the element-arrangement (EA) shuffles inserted by ``validate_ops``.

    A shuffle is a full read-and-write copy of a tensor through the device, so it
    is among the most expensive things the pre-scheduling passes can add, and
    ``validate_ops`` inserts one greedily for every mixed STANDARD/staggered
    operand pair. This pass looks at all of them together (``_insert_shuffle_op``
    tags each with ``ea_shuffle``) and removes or cheapens what it can. It runs
    after layout propagation and ``validate_ops`` and before
    ``optimize_restickify_locations``, so the layouts it keeps are the ones the
    restickify optimizer sees. It repeats until nothing changes.

    Implemented techniques, in the order they are tried:

    1. **Dead shuffle.** A shuffle nobody reads and that is not a graph output
       is deleted. Earlier rewrites below can orphan shuffles.
    2. **No-op shuffle.** If the source already has the target arrangement and
       the same device geometry, the copy moves nothing; its readers read the
       source directly.
    3. **Shuffle chains.** ``A -> B -> C`` through two shuffles collapses to one
       shuffle ``A -> C``; if ``A`` already has ``C``'s layout the pair cancels
       and ``C``'s readers read ``A``. This is how a shuffle in front of and
       behind the same value (``STANDARD -> staggered -> STANDARD``) disappears.
    4. **Duplicate shuffles.** Several ops that need the same value in the same
       arrangement (e.g. a weight shuffled once per consumer) share one shuffle.
    5. **Cast sinking (the "bracket" conversion).** Spyre converts between fp16
       and fp32 by staggering the elements, and the conversion map
       (``DtypeOpTable.ea_map``) is a bracket: an fp16 value stored with
       ``FP32_TO_DL16`` up-casts to a STANDARD fp32 value, and a staggered fp32
       down-casts to STANDARD fp16. A cast therefore already undoes a stagger for
       free. ``upcast`` followed by a shuffle back to STANDARD is rewritten as
       "shuffle the fp16 input, then up-cast", which moves half the bytes and
       removes the shuffle from the wide (fp32) tensor. This is the case where a
       shuffle is not needed at all if the surrounding conversions are chosen
       as a bracket.

    A shuffle is only rewritten when all of its readers are plain ``ComputedBuffer``
    ops whose cost function can be rebuilt (see ``_can_redirect_readers``), when
    the buffers involved have a single candidate layout (a shuffle's output
    layout is fixed when it is inserted, so a multi-candidate source could later
    be committed to a different layout than the shuffle assumed), and when the
    value is not a graph output. Anything else is left exactly as
    ``validate_ops`` produced it.

    Not implemented yet (the rewrites need layout re-propagation or codegen
    support that this pass cannot provide on its own):

    * **Operand flipping for a bracket.** In ``(a_staggered, b_STANDARD) -> op ->
      downcast`` the shuffle can be placed on ``b`` (to the staggered EA) instead
      of ``a`` (to STANDARD). The op's output then stays staggered and the
      closing conversion restores STANDARD, so the bracket is kept and only one
      narrow operand is shuffled. It also changes the EA of every value up to the
      closing cast, whose layouts would have to be recomputed.
    * **Restickify instead of cast plus shuffle.** When an edge needs both a
      stick change and an EA change, a single ``ReStickifyOpHBM`` that also
      writes the target arrangement replaces a restickify followed by a shuffle,
      saving a full pass over the tensor. It needs the restickify codegen to
      encode the EA on its output tensor.
    """
    outputs = set(graph.get_output_names())
    # Every successful rewrite removes at least one operation or one shuffle
    # candidate, so this loop is bounded.
    for _ in range(len(graph.operations) + 1):
        shuffles = [op for op in graph.operations if getattr(op, "ea_shuffle", False)]
        if not shuffles:
            return
        changed = False
        for op in shuffles:
            if _try_drop_shuffle(graph, op, outputs) or _try_sink_shuffle_through_cast(
                graph, op, outputs
            ):
                changed = True
                break
        if not changed:
            changed = _try_merge_duplicates(graph, shuffles, outputs)
        if not changed:
            return


def _validate_split_supported(op, trace, intermediate_ops) -> None:
    """Refuse to split ``op`` when an intermediate would combine view loads.

    An intermediate is materialized as an FX op over the base buffers, which
    reproduces a load only at its buffer's canonical position: the buffer's
    strides on the trailing output index symbols, aligned as for broadcasting.
    A load at an offset or through a view (e.g. one select() row) is elsewhere.
    A single such operand is still reloaded at its own index on replay, but two
    tensor operands are combined over the base buffers and pair the wrong
    elements.
    """
    # A reduction's loads also index the reduction symbols, which the canonical
    # position does not model, so only pointwise bodies are checked.
    if not isinstance(op.data, Pointwise):
        return
    syms = _index_syms(len(op.data.ranges))
    tensors: set[int] = set()  # values derived from any load
    from_view: set[int] = set()  # values derived from a view load
    for name, vid, inputs, kwargs in trace:
        if name == "load":
            tensors.add(vid)
            layout = V.graph.get_buffer(kwargs["_name"]).get_layout()
            lead = len(syms) - len(layout.size)
            # A buffer of higher rank than the output is read through a view.
            is_view = lead < 0
            if not is_view:
                canonical = layout.offset + sum(
                    st * syms[lead + k]
                    for k, (sz, st) in enumerate(zip(layout.size, layout.stride))
                    if sz != 1
                )
                is_view = sympy.expand(kwargs["_index"] - canonical) != 0
            if is_view:
                from_view.add(vid)
        elif any(v in tensors for v in inputs):
            tensors.add(vid)
            if any(v in from_view for v in inputs):
                from_view.add(vid)
    for name, _, inputs, _ in intermediate_ops:
        operands = [v for v in inputs if v in tensors]
        if len(operands) > 1 and any(v in from_view for v in operands):
            raise Unsupported(
                f"split_multi_ops: {op.get_name()}: intermediate '{name}' combines "
                "loads through a view, which split_multi_ops does not support"
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
       b. Skip if only 1 compute op and not Reduction with type conversion.
       c. Create intermediate buffers (including SpyreConstantFallback for constants).
       d. Wrap the original inner_fn with _SplitOpsHandler to redirect buffer names.

    Args:
        graph: GraphLowering instance containing operations to process
    """
    gl = V.graph
    # Skip if in graph lowering context
    if not (hasattr(gl, "graph") and hasattr(gl, "run_node")):
        return

    patch_env(gl)

    operations = graph.operations
    for op in list(operations):
        if _is_invalid_compute_op(op):
            continue

        trace = _trace_inner_fn(op)
        if not trace:
            continue

        compute_ops = _get_compute_ops(trace)
        if _skip_splitting(op, compute_ops):
            continue

        # Use real_layout() for MutationLayoutSHOULDREMOVE so _make_intermediate_bufs
        # gets a valid device and dtype from the underlying FixedLayout.
        layout = (
            op.layout.real_layout()
            if isinstance(op.layout, MutationLayoutSHOULDREMOVE)
            else op.layout
        )
        dtype_map = _propagate_dtypes(trace, layout.dtype)
        bufname_map = _init_vid_to_bufname(trace)

        try:
            insert_idx = operations.index(op)
        except ValueError:
            continue

        # Skip if no FX graph origin node in the current (sub)graph. A buffer
        # lowered inside an invoke_subgraph HOP carries origins from both the
        # parent graph and the subgraph; only the subgraph-local origin is a
        # valid inserting_before anchor here. origin_in_graph already enforces
        # node-ness and current-graph membership.
        orig_node = origin_in_graph(op.origins, gl.graph)
        if orig_node is None:
            continue

        intermediate_ops = compute_ops[:-1]
        final_op_name = compute_ops[-1][0]

        # An intermediate in a multi-op body that combines a view load with
        # another tensor operand is refused; replaying it over the base buffers
        # would pair the wrong elements.
        # TODO: support views. A view load needs a buffer of its own, which can
        # come from two places:
        # - here, by rebuilding the intermediate from traced index expressions.
        #   That reconstructs inner_fn, against the rule that a pass wraps inner_fn
        #   and never reconstructs it (#2797), so it means revisiting that rule.
        # - in the lowering that builds the body, by realizing the viewed operand
        #   before it is inlined into another op and giving it an FX origin with
        #   _ensure_synthetic_origin. It costs a kernel per operand and covers
        #   only operands a lowering builds itself.
        _validate_split_supported(op, trace, intermediate_ops)

        if _is_reduction_with_to_dtype(op, compute_ops):
            intermediate_ops = compute_ops
            final_op_name = "reduction"

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
            op,
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
