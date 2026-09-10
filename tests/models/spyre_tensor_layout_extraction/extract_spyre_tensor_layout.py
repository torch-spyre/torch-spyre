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

import logging
import torch
import torch.fx
import torch.fx.node
import torch._inductor.compile_fx
from collections import OrderedDict, defaultdict

from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch._inductor import ir
from torch._inductor.ir import FixedLayout

from torch_spyre._inductor.propagate_layouts import propagate_spyre_tensor_layouts
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.ir import FixedTiledLayout
from typing import Any
import torch._inductor.debug as dbg
from torch._inductor.dependencies import MemoryDep

log = logging.getLogger(__name__)

ISINSTANCE_TO_OP = [
    (torch.nn.Embedding, "torch.nn.functional.embedding"),
    (torch.nn.Linear, "torch.nn.functional.linear"),
    (torch.nn.LayerNorm, "torch.nn.functional.layer_norm"),
    (torch.nn.SiLU, "torch.nn.functional.silu"),
    (torch.nn.GELU, "torch.nn.functional.gelu"),
    (torch.nn.Softmax, "torch.nn.functional.softmax"),
    (torch.nn.Conv1d, "torch.nn.functional.conv1d"),
    (torch.nn.Conv2d, "torch.nn.functional.conv2d"),
]
MODULE_CLASSNAME_FALLBACK = {
    "RMSNorm": "torch.nn.functional.rms_norm",
}


def resolve_module_op(module) -> str | None:
    for cls, op in ISINSTANCE_TO_OP:
        if isinstance(module, cls):
            return op
    return MODULE_CLASSNAME_FALLBACK.get(type(module).__name__)


def get_op_name(node, graph_module=None) -> str | None:
    if node.op == "call_function":
        mod = getattr(node.target, "__module__", "") or ""
        if mod == "_operator":
            name = node.target.__name__
            if name in torch.fx.graph.magic_methods:
                return f"torch.{name}"
            if name in torch.fx.graph.inplace_methods and name[0] == "i":
                return f"torch.{name[1:]}_"
        op = torch.fx.node._get_qualified_name(node.target)
        if op.startswith("torch._C._nn."):
            op = "torch.nn.functional." + op[len("torch._C._nn."):]
        return op
    elif node.op == "call_method":
        method_name = str(node.target)
        return (
            f"torch.Tensor.{method_name}"
            if hasattr(torch.Tensor, method_name)
            else f"torch.{method_name}"
        )
    elif node.op == "call_module":
        if graph_module is not None:
            try:
                module = graph_module.get_submodule(node.target)
                op = resolve_module_op(module)
                return (
                    op if op else f"{type(module).__module__}.{type(module).__name__}"
                )
            except Exception as exc:
                log.warning(
                    "get_op_name: failed to resolve module for node %r (target=%r): %s",
                    getattr(node, "name", "?"),
                    node.target,
                    exc,
                )
    return None


def get_description(node) -> str | None:
    st = getattr(node, "stack_trace", None) or node.meta.get("stack_trace", None)
    if st:
        for line in reversed(st.strip().split("\n")):
            if ", code:" in line:
                return line.strip()
    return None


def get_nn_module(node) -> dict:
    stack = node.meta.get("nn_module_stack", {})
    if not stack:
        return {}
    _, (qname, cls) = list(stack.items())[-1]
    return {"module_path": qname, "module_class": cls.__name__}


def walk_from_node_to_root(node, max_depth: int = 20):
    current, visited, depth = node, set(), 0
    while depth < max_depth:
        if id(current) in visited:
            break
        visited.add(id(current))
        if not hasattr(current, "meta"):
            break
        fn_src = current.meta.get("from_node", None)
        if fn_src is None:
            break
        if isinstance(fn_src, list):
            if not fn_src:
                break
            fn_src = fn_src[0]
        if hasattr(fn_src, "node"):
            parent = fn_src.node
            if isinstance(parent, list):
                if not parent:
                    break
                parent = parent[0]
            if hasattr(parent, "node") and not hasattr(parent, "meta"):
                parent = parent.node
        else:
            parent = fn_src
        if parent is current or parent is None:
            break
        current = parent
        depth += 1
    if depth == max_depth:
        log.warning(
            "walk_from_node_to_root: depth cap from %r",
            getattr(node, "name", "<unknown>"),
        )
    return current


def pre_node_key(node) -> tuple:
    return (node.name, id(node.graph))


def get_origins(op) -> set[Any]:
    data = getattr(op, "data", None)
    origins: set[Any] = getattr(data, "origins", set()) if data else set()
    if not origins:
        origins = getattr(op, "origins", set()) or set()
    if not origins:
        node = getattr(op, "origin_node", None)
        if node is not None:
            origins = {node}
    return origins


def pick_deterministic_origin(origins):
    if not origins:
        return None
    return min(origins, key=lambda n: getattr(n, "name", ""))


def to_serializable(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return str(val)


def make_stub(ptl):
    """Fabricate a synthetic seed FixedTiledLayout
    for graph inputs and ExternKernelOut nodes.

    The SpyreTensorLayout constructed here uses an identity dim_order
    (list(range(len(size)))) and row-major stride — NOT the layout the Spyre
    compiler would actually assign. It exists purely to give
    propagate_spyre_tensor_layouts a concrete starting point from which to
    propagate real layouts through the rest of the graph.
    """
    size, stride, dtype = list(ptl.size), list(ptl.stride), ptl.dtype
    stl = SpyreTensorLayout(size, stride, dtype, list(range(len(size))))
    ftl = FixedTiledLayout(ptl.device, dtype, ptl.size, ptl.stride, stl)
    return ftl, stl


def _arg_tensor(arg):
    """Return shape/stride/dtype dict for an FX node arg, or None."""
    if not isinstance(arg, torch.fx.Node):
        return None
    mv = arg.meta.get(
        "val",
        arg.meta.get("tensor_meta", arg.meta.get("example_value", None)),
    )
    return (
        {
            "shape": list(mv.shape),
            "stride": list(mv.stride()),
            "dtype": str(mv.dtype),
        }
        if isinstance(mv, torch.Tensor)
        else None
    )


def _sanitize_args(args):
    out = []
    for a in args:
        if isinstance(a, torch.fx.Node):
            out.append(None)
        elif isinstance(a, (list, tuple)):
            out.append(_sanitize_args(a))
        else:
            out.append(a)
    return out


def _stub_input_buffers(graph, operations):
    for name, tb in graph.graph_inputs.items():
        if not isinstance(tb, ir.TensorBox):
            continue
        if not isinstance(tb.data, ir.StorageBox):
            continue
        ib = tb.data.data
        if not isinstance(ib, ir.InputBuffer):
            continue
        ptl = ib.layout
        if isinstance(ptl, FixedTiledLayout):
            continue
        ftl, stl = make_stub(ptl)
        ib.layout = ftl
        tb.layouts = [stl]
        ib.committed_stl = stl
        tb.committed_stl = stl
    for op in operations:
        if isinstance(op, (ir.ExternKernelOut, ir.MultiOutput)):
            if not isinstance(op.layout, FixedTiledLayout):
                ftl, stl = make_stub(op.layout)
                op.layout = ftl
                op.layouts = [stl]
                op.committed_stl = stl


def _propagate_layouts(graph, operations):
    try:
        propagate_spyre_tensor_layouts(graph)
    except Exception as exc:
        log.warning(
            "_propagate_layouts: propagate_spyre_tensor_layouts stopped early — "
            "%s: %s. Ops processed before this point will still be extracted.",
            type(exc).__name__,
            exc,
        )
    for op in operations:
        if (
            hasattr(op, "layouts")
            and op.layouts
            and not isinstance(op.layout, FixedTiledLayout)
        ):
            ptl = op.layout
            op.layout = FixedTiledLayout(
                ptl.device, ptl.dtype, ptl.size, ptl.stride, op.layouts[0]
            )


def _group_ops_by_pre_node(
    operations, local_postToPre, local_pre_graph_id, pre_node_info, pre_node_order
):
    via_mapping = via_walk = 0
    groups = {}

    for op in operations:
        layout = getattr(op, "layout", None)
        if not isinstance(layout, (FixedTiledLayout, FixedLayout)):
            continue
        origins = get_origins(op)
        if not origins:
            continue

        post_fx_node = pick_deterministic_origin(origins)
        root_key = None

        post_name = getattr(post_fx_node, "name", None)
        if post_name and post_name in local_postToPre:
            pre_names = local_postToPre[post_name]
            pre_name = pre_names[0] if pre_names else None
            if pre_name:
                mk = (pre_name, local_pre_graph_id)
                if mk in pre_node_info:
                    root_key = mk
                    via_mapping += 1

        if root_key is None:
            root = walk_from_node_to_root(post_fx_node)
            if hasattr(root, "node") and not hasattr(root, "meta"):
                root = root.node
            root_name = getattr(root, "name", None)
            if root_name:
                gid = (
                    root.graph_id
                    if hasattr(root, "graph_id")
                    else id(root.graph)
                    if hasattr(root, "graph")
                    else local_pre_graph_id
                )
                wk = (root_name, gid)
                if wk in pre_node_info:
                    root_key = wk
                    via_walk += 1

        if root_key is None:
            log.debug(
                "Attribution failed for IR op %r (%s): post_fx_node=%r "
                "not resolved via postToPre or from_node walk.",
                getattr(op, "name", "?"),
                type(op).__name__,
                getattr(post_fx_node, "name", "?"),
            )
            continue

        if root_key in pre_node_info:
            groups.setdefault(root_key, []).append(op)
        else:
            log.debug(
                "Attribution resolved to key %r for IR op %r but key is "
                "absent from pre_node_info — op dropped.",
                root_key,
                getattr(op, "name", "?"),
            )

    groups = OrderedDict(
        sorted(groups.items(), key=lambda x: pre_node_order.get(x[0], 99999))
    )
    return groups, via_mapping, via_walk


def _collect_pre_input_tensors(pre_node, buffer_map=None) -> list:
    tensors = []

    def _ir_name_for_arg(arg: torch.fx.Node):
        if buffer_map is None:
            return None

        if arg.name in buffer_map:
            return arg.name

        for ir_name, buf in buffer_map.items():
            for origin in get_origins(buf):
                if getattr(origin, "name", None) == arg.name:
                    return ir_name
        return None

    for arg in pre_node.args:
        if isinstance(arg, torch.fx.Node):
            entry = _arg_tensor(arg)
            if entry is not None:
                entry["ir_name"] = _ir_name_for_arg(arg)
            tensors.append(entry)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, torch.fx.Node):
                    entry = _arg_tensor(item)
                    if entry is not None:
                        entry["ir_name"] = _ir_name_for_arg(item)
                    tensors.append(entry)
    return tensors


def _extract_layouts(groups, buffer_map, pre_node_info, local_postToPre):
    extracted = []

    for pre_key, group_ops in groups.items():
        try:
            last_op = group_ops[-1]
            out_layout = getattr(last_op, "layout", None)
            if not isinstance(out_layout, (FixedTiledLayout, FixedLayout)):
                continue

            pinfo = pre_node_info[pre_key]
            pre_node = pinfo["node"]

            pre_input_tensors = _collect_pre_input_tensors(pre_node, buffer_map)
            pre_kwarg_tensors = {}
            for kn, a in pre_node.kwargs.items():
                t = _arg_tensor(a)
                pre_kwarg_tensors[kn] = (
                    t
                    if t is not None
                    else (
                        a if isinstance(a, (int, float, bool, type(None))) else str(a)
                    )
                )

            internal = {getattr(op, "name", None) for op in group_ops}

            pre_arg_name_to_idx: dict[str, int] = {
                arg.name: i
                for i, arg in enumerate(pre_node.args)
                if isinstance(arg, torch.fx.Node)
            }

            seen, input_layouts = set(), []
            for op in group_ops:
                try:
                    for dep in op.get_read_writes().reads:
                        if not isinstance(dep, MemoryDep):
                            continue
                        dn = dep.name
                        if dn in internal or dn in seen:
                            continue
                        seen.add(dn)
                        buf = buffer_map.get(dn)
                        if buf is None:
                            continue
                        if isinstance(buf, ir.TensorBox):
                            try:
                                il = buf.get_layout()
                            except Exception as exc:
                                log.warning(
                                    "get_layout failed for input %r of op %r (%s): %s "
                                    "— this input layout will be missing from the YAML.",
                                    dn,
                                    pinfo["op"],
                                    pre_key[0],
                                    exc,
                                )
                                continue
                        else:
                            il = getattr(buf, "layout", None)
                        if not isinstance(il, (FixedTiledLayout, FixedLayout)):
                            continue
                        dl = (
                            il.device_layout
                            if isinstance(il, FixedTiledLayout)
                            else type(
                                "DL",
                                (),
                                {
                                    "device_size": list(il.size),
                                    "stride_map": list(il.stride),
                                    "device_dtype": str(il.dtype),
                                },
                            )()
                        )

                        pre_arg_name = None
                        if pre_arg_name_to_idx:
                            for origin in get_origins(buf):
                                post_name = getattr(origin, "name", None)
                                for pname in local_postToPre.get(post_name or "", []):
                                    if pname in pre_arg_name_to_idx:
                                        pre_arg_name = pname
                                        break
                                if pre_arg_name is not None:
                                    break

                        input_layouts.append(
                            {
                                "name": dn,
                                "pre_arg_name": pre_arg_name,
                                "dtype": str(il.dtype),
                                "size": [to_serializable(s) for s in il.size],
                                "stride": [to_serializable(s) for s in il.stride],
                                "device_size": [
                                    to_serializable(s) for s in dl.device_size
                                ],
                                "stride_map": [
                                    to_serializable(s) for s in dl.stride_map
                                ],
                                "device_dtype": str(dl.device_dtype),
                            }
                        )
                except (AttributeError, KeyError, NotImplementedError):
                    pass

            extracted.append(
                {
                    "op": pinfo["op"],
                    "pre_node": pre_key[0],
                    # in input_layouts.append({...}):
                    "pre_node_graph_id": pre_key[1],
                    "num_subops": len(group_ops),
                    "sub_ops": [getattr(o, "name", "?") for o in group_ops],
                    "nn_module": pinfo["nn_module"],
                    "description": pinfo["description"],
                    "dtype": str(out_layout.dtype),
                    "pre_input_tensors": pre_input_tensors,
                    "pre_kwarg_tensors": pre_kwarg_tensors,
                    "pre_node_args": _sanitize_args(pre_node.args),
                    "inputs": input_layouts,
                }
            )
        except (AttributeError, KeyError, RuntimeError) as exc:
            log.warning("Layout extraction failed for %r: %s", pre_key, exc)

    return extracted


def make_capture_fns(
    pre_node_info: dict,
    pre_node_order: dict,
    order_counter: list,
    ir_graphs: list,
    cur_pre_graph_id: list,
):
    _orig_compile_fx = torch._inductor.compile_fx.compile_fx
    _orig_gl_run = GraphLowering.run

    def device_layout_capture_backend(gm, example_inputs):
        cur_pre_graph_id[0] = id(gm.graph)
        try:
            for node in gm.graph.nodes:
                if node.op not in ("call_function", "call_method", "call_module"):
                    continue
                op_name = get_op_name(node, graph_module=gm)
                if op_name is None:
                    continue
                key = pre_node_key(node)
                if key not in pre_node_info:
                    pre_node_info[key] = {
                        "op": op_name,
                        "description": get_description(node),
                        "nn_module": get_nn_module(node),
                        "node": node,
                    }
                    pre_node_order[key] = order_counter[0]
                    order_counter[0] += 1
        except (AttributeError, KeyError, RuntimeError) as exc:
            log.exception("device_layout_capture_backend error: %s", exc)
        return _orig_compile_fx(gm, example_inputs)

    def patched_gl_run(self, *example_inputs):
        result = _orig_gl_run(self, *example_inputs)
        try:
            snap = dict(dbg._inductor_post_to_pre_grad_nodes.get("postToPre", {}))

            ops_by_name = {op.name: op for op in self.buffers if hasattr(op, "name")}
            for op in getattr(self, "operations", []):
                if hasattr(op, "name") and op.name not in ops_by_name:
                    ops_by_name[op.name] = op
            ir_graphs.append(
                {
                    "graph": self,
                    "operations": list(ops_by_name.values()),
                    "pre_graph_id": cur_pre_graph_id[0],
                    "postToPre": snap,
                }
            )
        except (AttributeError, KeyError, ImportError) as exc:
            log.exception("patched_gl_run error: %s", exc)
        return result

    def install_patches():
        GraphLowering.run = patched_gl_run

    def remove_patches():
        GraphLowering.run = _orig_gl_run

    return device_layout_capture_backend, install_patches, remove_patches


def process_ir_graphs(
    ir_graphs: list, pre_node_info: dict, pre_node_order: dict
) -> tuple:
    all_layouts = []
    via_map_tot = via_walk_tot = 0

    for graph_idx, gd in enumerate(ir_graphs):
        graph, operations = gd["graph"], gd["operations"]
        local_pre_graph_id, local_postToPre = gd["pre_graph_id"], gd["postToPre"]
        try:
            with V.set_graph_handler(graph):
                _stub_input_buffers(graph, operations)

                buffer_map = {op.name: op for op in operations if hasattr(op, "name")}
                for name, tb in graph.graph_inputs.items():
                    buffer_map[name] = tb

                orig_get_buf = graph.get_buffer
                orig_names = graph.graph_input_names
                try:
                    graph.get_buffer = (
                        lambda n: buffer_map[n] if n in buffer_map else orig_get_buf(n)
                    )
                    graph.graph_input_names = []
                    _propagate_layouts(graph, operations)
                finally:
                    graph.graph_input_names = orig_names
                    graph.get_buffer = orig_get_buf

                groups, via_map, via_walk = _group_ops_by_pre_node(
                    operations,
                    local_postToPre,
                    local_pre_graph_id,
                    pre_node_info,
                    pre_node_order,
                )
                via_map_tot += via_map
                via_walk_tot += via_walk

                extracted = _extract_layouts(
                    groups, buffer_map, pre_node_info, local_postToPre
                )
                all_layouts.extend(extracted)
                log.info(
                    "IR graph %d: %d ops → %d groups → %d extracted",
                    graph_idx,
                    len(operations),
                    len(groups),
                    len(extracted),
                )
        except (RuntimeError, AttributeError) as exc:
            log.exception("Failed IR graph %d: %s", graph_idx, exc)

    return all_layouts, via_map_tot, via_walk_tot


def collect_primitive_ops(
    captured_pre_keys: set, pre_node_info: dict, pre_node_order: dict
) -> list:
    primitive_ops = []
    for pre_key in sorted(pre_node_order, key=lambda k: pre_node_order[k]):
        if pre_key in captured_pre_keys or pre_key not in pre_node_info:
            continue
        pinfo = pre_node_info[pre_key]
        pre_node = pinfo["node"]

        input_tensors = _collect_pre_input_tensors(pre_node)
        first_tensor = next((t for t in input_tensors if t is not None), None)
        dtype = first_tensor["dtype"] if first_tensor else "torch.float32"

        primitive_ops.append(
            {
                "op": pinfo["op"],
                "pre_node": pre_key[0],
                "pre_node_graph_id": pre_key[1],
                "num_subops": 0,
                "sub_ops": [],
                "nn_module": pinfo["nn_module"],
                "description": pinfo["description"],
                "dtype": dtype,
                "pre_input_tensors": input_tensors,
                "pre_kwarg_tensors": {},
                "pre_node_args": _sanitize_args(pre_node.args),
                "inputs": [],
                "primitive": True,
            }
        )
    return primitive_ops


def get_operation_signature(op_entry: dict) -> tuple:
    input_shapes = [
        tuple(i["size"]) for i in op_entry.get("inputs", []) if i and "size" in i
    ]
    if not input_shapes:
        input_shapes = [
            tuple(t["shape"])
            for t in op_entry.get("pre_input_tensors", [])
            if t and "shape" in t
        ]

    dtype = next(
        (t["dtype"] for t in op_entry.get("pre_input_tensors", []) if t),
        op_entry.get("dtype", "?"),
    )
    return (
        op_entry.get("op", "?"),
        tuple(input_shapes),
        dtype,
    )


def deduplicate_ops(all_ops: list) -> list:
    sig_map, seen = defaultdict(list), []
    for op in all_ops:
        sig = get_operation_signature(op)
        if sig not in sig_map:
            seen.append(sig)
        sig_map[sig].append(op)
    return [sig_map[s][0] for s in seen]
