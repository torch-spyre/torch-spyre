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
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from .logging_utils import get_inductor_logger

logger = get_inductor_logger("split_multi_ops")

_STRUCTURAL_OPS = frozenset({"load", "store", "get_index"})
_DTYPE_OPS = frozenset({"to_dtype", "convert_element_type", "constant"})
_OPS_WITH_CONSTANT_ARGS = frozenset({"clamp", "layernormscale", "softplus"})

_OP_TARGET_TABLE = {
    "to_dtype": torch.ops.prims.convert_element_type.default,
    "convert_element_type": torch.ops.prims.convert_element_type.default,
}


class _Val:
    __slots__ = ("handler", "vid")

    def __init__(self, handler, vid):
        self.handler, self.vid = handler, vid


class _TracingHandler:
    def __init__(self, prev_handler):
        self._prev = prev_handler
        self.ops = []
        self._next_vid = 0

    def __enter__(self):
        self._saved = V.ops
        V.set_ops_handler(self)
        return self

    def __exit__(self, *args):
        V.set_ops_handler(self._saved)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def _record(*args, **kwargs):
            vids = []
            extra = []
            for a in args:
                if isinstance(a, _Val):
                    vids.append(a.vid)
                elif isinstance(a, (bool, int, float)):
                    vids.append(self._wrap_literal(a).vid)
                else:
                    extra.append(a)
            merged = dict(kwargs)
            for i, v in enumerate(extra):
                merged[f"_p{i}"] = v
            vid = self._alloc_vid()
            self.ops.append((name, vid, tuple(vids), merged))
            return _Val(self, vid)

        return _record

    def load(self, name, index):
        vid = self._alloc_vid()
        self.ops.append(("load", vid, (), {"_name": name, "_index": index}))
        return _Val(self, vid)

    def store(self, name, index, value, mode=None):
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
        vid = self._alloc_vid()
        self.ops.append(
            ("constant", vid, (), {"fill_value": fill_value, "dtype": dtype})
        )
        return _Val(self, vid)

    def _alloc_vid(self):
        v = self._next_vid
        self._next_vid += 1
        return v

    def _wrap_literal(self, value):
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
    for ns in (torch.ops.aten, torch.ops.prims, getattr(torch.ops, "spyre", None)):
        if ns is None:
            continue
        target = getattr(ns, op_name, None)
        if target is not None:
            default = getattr(target, "default", None)
            return default or target
    return None


def _normalize_op_args(op_name, input_fx_nodes, kwargs, out_dtype, device=None):
    if op_name == "constant":
        fill = kwargs["fill_value"]
        dtype = kwargs.get("dtype", out_dtype)
        dev = device if device is not None else torch.device("spyre")
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
    pos_keys = sorted(k for k in kwargs if k.startswith("_p"))
    extra = tuple(kwargs[k] for k in pos_keys)
    clean_kw = {k: v for k, v in kwargs.items() if not k.startswith("_p")}

    vid_to_stride = {}
    for v in value_vids:
        if v not in vid_to_constant:
            buf_name = vid_to_bufname[v]
            vid_to_stride[v] = V.graph.get_buffer(buf_name).layout.stride

    def inner_fn(index):
        inputs = []
        for v in value_vids:
            if v in vid_to_constant:
                if op_name in _OPS_WITH_CONSTANT_ARGS:
                    fill, _ = vid_to_constant[v]
                    inputs.append(fill)
                else:
                    inputs.append(V.ops.load(vid_to_bufname[v], sympy.Integer(0)))
            else:
                buf_stride = vid_to_stride[v]
                idx = sum(i * s for i, s in zip(index, buf_stride))
                inputs.append(V.ops.load(vid_to_bufname[v], idx))
        return getattr(V.ops, op_name)(*inputs, *extra, **clean_kw)

    return inner_fn


def _trace_inner_fn(op):
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
    return [e for e in trace if e[0] not in _STRUCTURAL_OPS]


def _propagate_dtypes(trace, fallback):
    dtype_map = {}
    for op, vid, inputs, kwargs in trace:
        if op == "load":
            dtype_map[vid] = V.graph.get_buffer(kwargs["_name"]).get_layout().dtype
        elif op == "store":
            pass
        else:
            in_dtypes = [dtype_map[v] for v in inputs if v in dtype_map]
            dtype_map[vid] = _infer_output_dtype(in_dtypes, kwargs, fallback)
    return dtype_map


def _init_vid_to_bufname(trace):
    return {vid: kwargs["_name"] for op, vid, _, kwargs in trace if op == "load"}


def _init_vid_to_constant(trace):
    return {
        vid: (kwargs["fill_value"], kwargs["dtype"])
        for op, vid, _, kwargs in trace
        if op == "constant"
    }


def _find_fx_node(name, gl):
    for n, tb in gl.env.items():
        if isinstance(n, fx.Node) and tb is not None and tb.get_name() == name:
            return n
    for n in gl.graph.nodes:
        if n.op == "placeholder" and n.name == name:
            return n
    raise KeyError(f"No FX node for {name}")


def _lower_fx_node(node, gl, ops, idx):
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
):
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
        if input_nodes and "val" in input_nodes[0].meta:
            new_node.meta["val"] = input_nodes[0].meta["val"].to(out_dtype)
        elif "val" in orig_node.meta:
            new_node.meta["val"] = orig_node.meta["val"].to(out_dtype)
        new_buf = _lower_fx_node(new_node, gl, operations, insert_idx)
        new_buf.origins = OrderedSet([new_node])
        vid_to_bufname[vid] = new_buf.get_name()
        bufs.append(new_buf)
        insert_idx += 1
    return bufs, insert_idx


def _update_original_buf(op, final_entry, vid_to_bufname, vid_to_constant, operations):
    op_name, _, vids, kwargs = final_entry
    new_data = dataclasses.replace(
        op.data,
        inner_fn=_build_inner_fn(
            op_name, vids, kwargs, vid_to_bufname, vid_to_constant
        ),
    )
    for attr in ("origins", "traceback", "origin_node", "annotations", "stream_idx"):
        if hasattr(op.data, attr):
            object.__setattr__(new_data, attr, getattr(op.data, attr))
    new_op = ComputedBuffer(
        name=op.get_name(),
        layout=op.layout,
        data=new_data,
    )
    new_op.operation_name = op.operation_name
    new_op.origins = op.origins
    for attr in (
        "_split_size",
        "_original_inner_fn",
        "_original_ranges",
        "_original_reduction_ranges",
    ):
        if hasattr(op, attr):
            object.__setattr__(new_op, attr, getattr(op, attr))
    if hasattr(new_op, "_cached_read_writes"):
        del new_op._cached_read_writes
    _ = new_op.get_read_writes()
    idx = operations.index(op)
    operations[idx] = new_op
    V.graph.name_to_buffer[new_op.get_name()] = new_op
    ComputedBuffer.get_default_sizes_body.clear_cache(new_op)
    return new_op


def split_multi_ops(graph: GraphLowering):
    operations = graph.operations
    gl = V.graph
    if not (hasattr(gl, "graph") and hasattr(gl, "run_node")):
        return
    env = {}
    for tbs in gl.name_to_users.values():
        for tb in tbs:
            if tb.data.origins:
                fx_node = next(iter(tb.data.origins))
                env[fx_node] = tb
    gl.env.update(env)

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

        intermediate_ops, final_op = compute_ops[:-1], compute_ops[-1]
        if not op.origins:
            continue
        orig_node = next(iter(op.origins))
        if not isinstance(orig_node, fx.Node):
            continue

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
            "split_multi_op: '%s' → %d intermediate buffers",
            op.get_name(),
            len(intermediate_ops),
        )
