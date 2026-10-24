#!/usr/bin/env python3
# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""Batch summarize all sdsc.json files from Inductor debug artifacts."""

import collections
import json
import os
import regex as re
import sys
from collections import defaultdict
from pathlib import Path

from tabulate import SEPARATING_LINE, tabulate

_SDSC_IDX_RE = re.compile(r"sdsc_(\d+)_")

# Match both sdsc_N.json and sdsc_N_<detail>.json for the file-index lookup.
_SDSC_FILE_IDX_RE = re.compile(r"sdsc_(\d+)(?:_|\.json)")

# The Inductor <hash>.py provides each fused kernel's LoopSpec/OpSpec tree
# structure; buffer names come from the DEBUG log (see _parse_debug_log).
_SDSC_CALL_RE = re.compile(r"async_compile\.sdsc\(\s*'([^']+)'")


def _sdsc_file_index(op: dict) -> int | None:
    """Return the numeric N from an op's `sdsc_N_...json` filename, else None."""
    name = Path(op["file"]).name
    m = _SDSC_FILE_IDX_RE.search(name)
    return int(m.group(1)) if m else None


def _flatten_opspec_tree(arg_text: str) -> list[int] | None:
    """Flatten a `sdsc(...)` argument list into per-op (leaf_idx, loop_coords).

    The argument is a tree of `LoopSpec(count=sympify('c'), body=[...])` and
    `OpSpec(...)` leaves. Each leaf OpSpec is assigned a definition index in
    source order (0, 1, 2, ...). A LoopSpec repeats its expanded body `count`
    times. Returns a flat list (in sdsc_N order, length == number of sdsc_N.json
    the kernel expands to) of tuples:
        (leaf_idx, loop_coords)
    where loop_coords is a tuple of the enclosing loops' iteration indices,
    outermost first (empty when the op is inside no loop). Returns None if the
    structure can't be parsed.

    Purely lexical - no import/eval of the (large, sympify-laden) source.
    """
    n = len(arg_text)

    # Cursor-based recursive descent over the bracketed structure. `pos` walks
    # the text; `leaf_counter` hands out definition indices in source order.
    leaf_counter = [0]

    def parse_list(start: int) -> tuple[list[tuple[int, tuple]], int]:
        """Parse a `[...]` list beginning at arg_text[start] == '['.

        Returns (flat [(leaf_idx, loop_coords), ...], index just past ']'). The
        coords here are relative to loops *inside* this list; enclosing loops
        are prepended by parse_loopspec.
        """
        assert arg_text[start] == "["
        i = start + 1
        result: list[tuple[int, tuple]] = []
        while i < n:
            c = arg_text[i]
            if c == "]":
                return result, i + 1
            if arg_text.startswith("LoopSpec(", i):
                seq, i = parse_loopspec(i)
                result.extend(seq)
            elif arg_text.startswith("OpSpec(", i):
                seq, i = parse_opspec(i)
                result.extend(seq)
            else:
                i += 1
        raise ValueError("unterminated list")

    def parse_loopspec(start: int) -> tuple[list[tuple[int, tuple]], int]:
        """Parse `LoopSpec(count=sympify('c'), ..., body=[...])`.

        Repeats the body `count` times, prepending this loop's iteration index
        (0..count-1) to each contained op's loop_coords.
        """
        # Find count=sympify('<n>') before the body= list.
        m = re.compile(r"count=sympify\('(\d+)'\)").search(arg_text, start)
        count = int(m.group(1)) if m else 1
        body_kw = arg_text.find("body=", start)
        if body_kw < 0:
            raise ValueError("LoopSpec without body")
        lb = arg_text.find("[", body_kw)
        if lb < 0:
            raise ValueError("LoopSpec body has no list")
        body_seq, after = parse_list(lb)
        # Advance past the closing ')' of this LoopSpec.
        end = _match_paren(arg_text, arg_text.find("(", start))
        expanded: list[tuple[int, tuple]] = []
        for it in range(count):
            for leaf_idx, coords in body_seq:
                expanded.append((leaf_idx, (it,) + coords))
        return expanded, max(after, end)

    def parse_opspec(start: int) -> tuple[list[tuple[int, tuple]], int]:
        """Parse a leaf `OpSpec(...)` - one op, one fresh definition index."""
        idx = leaf_counter[0]
        leaf_counter[0] += 1
        end = _match_paren(arg_text, arg_text.find("(", start))
        return [(idx, ())], end

    lb = arg_text.find("[")
    if lb < 0:
        return None
    try:
        flat, _ = parse_list(lb)
    except (ValueError, AssertionError, RecursionError):
        return None
    return flat


def _match_paren(text: str, open_idx: int) -> int:
    """Return the index just past the ')' matching the '(' at open_idx."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _match_bracket(text: str, open_idx: int) -> int:
    """Return the index just past the ']' matching the '[' at open_idx."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _parse_py_kernels(
    base_dir: Path,
) -> dict[str, list[tuple[Path, tuple[list[int], int, list[tuple]]]]]:
    """Map fused-kernel name -> (leaf_seq, n_leaves, coords_seq) from the .py.

    `leaf_seq[k]` is the leaf-OpSpec definition index that sdsc_k.json expands
    from (the LoopSpec/OpSpec tree flattened); its length equals the number of
    sdsc_N.json the kernel expands to. `n_leaves` is the count of distinct leaf
    OpSpecs. `coords_seq[k]` is the tuple of enclosing loop iteration indices
    (outermost first, empty when the op is inside no loop) - used to render a
    per-op loop counter. The .py provides the loop *structure* only; per-op
    buffer names come from the DEBUG log (see _parse_debug_log). Additive and
    never fatal: parse errors skip that file/kernel.

    Search scope: the Inductor <hash>.py that carries the OpSpec tree is often a
    *sibling* of the sdsc_*/ dirs, not a descendant. E.g. with
    base_dir=.../torchinductor_x/inductor-spyre/ the tree lives in
    .../torchinductor_x/47/c47poj….py - both under the shared cache root, but
    the .py is outside the subdir. glob() only searches downward, so a subdir
    argument misses it. Fix: scan base_dir first; if that finds no kernels, walk
    up parent dirs (bounded) and scan there, so a subdir still resolves the tree.
    """

    def _scan(root: Path) -> dict[str, tuple[list[int], int, list[tuple]]]:
        found: dict[str, list[tuple[Path, tuple]]] = {}
        for py_file in root.glob("**/*.py"):
            try:
                text = py_file.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            for m in _SDSC_CALL_RE.finditer(text):
                try:
                    name = m.group(1)
                    # The argument list: the balanced [...] after the kernel name.
                    lb = text.find("[", m.end())
                    if lb < 0:
                        continue
                    rb = _match_bracket(text, lb)
                    arg_text = text[lb:rb]
                    flat = _flatten_opspec_tree(arg_text)
                    if flat is None:
                        continue
                    leaf_seq = [leaf_idx for leaf_idx, _ in flat]
                    coords_seq = [coords for _, coords in flat]
                    n_leaves = (max(leaf_seq) + 1) if leaf_seq else 0
                    # Several compiles of one kernel (different shapes) share
                    # the name; keep each with the cache root its .py sits in
                    # (<root>/<xx>/<hash>.py beside <root>/inductor-spyre/).
                    found.setdefault(name, []).append(
                        (py_file.parent.parent, (leaf_seq, n_leaves, coords_seq))
                    )
                except (ValueError, IndexError):
                    continue
        return found

    kernels = _scan(base_dir)
    if kernels:
        return kernels

    # Nothing under base_dir: the .py tree is likely a sibling under a shared
    # cache root. Walk up (at most a few levels) and scan each parent, stopping
    # as soon as a scan yields kernels or we hit the filesystem root.
    root = base_dir.resolve()
    for _ in range(4):
        parent = root.parent
        if parent == root:  # reached filesystem root
            break
        root = parent
        kernels = _scan(root)
        if kernels:
            break
    return kernels


# spyre_kernel DEBUG channel: per-op output buffer stores and the op_spec tree.
_KSTORE_RE = re.compile(r"kernel_store(?:_reduction)?:\s*(buf\d+)\b")
# kernel_load names an op input; may be an intermediate (bufN), a graph input
# (argN_1), or a scalar const - accept any \w+ operand name.
_KLOAD_RE = re.compile(r"kernel_load:\s*(\w+)\b")
_OPSPEC_LOOP_RE = re.compile(r"op_spec:\s*LoopSpec\(count=(\d+)\)")
_OPSPEC_LEAF_RE = re.compile(r"op_spec:\s*(\w+),")
_GENERATING_RE = re.compile(r"Generating\s+(\S+?/sdsc_0\.json)\b")


def _parse_debug_log(log_path: Path) -> dict[str, tuple[list[str], list[list[str]]]]:
    """Map kernel-dir basename -> (per-leaf output bufN, per-leaf input names).

    Parses the SPYRE_INDUCTOR_LOG DEBUG log. The `spyre_kernel` phase logs, per
    fused kernel and per op: a run of `kernel_load: <name>` lines (that op's
    inputs, in order) immediately followed by its `kernel_store[_reduction]:
    bufN` line (its output). A run of `op_spec:` lines (the LoopSpec/OpSpec tree)
    closes the kernel. The `sdsc_compile` phase then logs `Generating
    .../sdsc_0.json` once per kernel dir, in the same kernel order - used to
    attach each parsed block to its directory.

    Returns {dir_basename: (stores, inputs)} where `stores[k]` is leaf op k's
    output bufN and `inputs[k]` is the ordered list of leaf op k's input names
    (both in source order, parallel). The caller combines this with the .py tree
    (leaf index per sdsc_k) to name every op. Additive and never fatal: any
    error yields an empty / partial map.
    """
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return {}

    # Phase 1: split the spyre_kernel stream into per-kernel blocks. Within a
    # kernel, each op is a run of kernel_load lines (accumulated in
    # `pending_loads`) terminated by a kernel_store line: on the store we emit
    # the store buf (`cur_stores`) and flush the pending loads as that op's
    # inputs (`cur_inputs`). A run of op_spec lines follows; the block closes
    # when a store/load line reappears after an op_spec run (next kernel).
    blocks: list[tuple[list[str], list[list[str]]]] = []
    cur_stores: list[str] = []
    cur_inputs: list[list[str]] = []
    pending_loads: list[str] = []
    in_opspec = False
    for line in lines:
        if "spyre_kernel]" not in line:
            continue
        is_opspec = "op_spec:" in line
        is_kernel_io = ("kernel_store" in line) or ("kernel_load" in line)
        if is_opspec:
            in_opspec = True
            continue
        if is_kernel_io:
            # A store/load after an op_spec run means a new kernel started.
            if in_opspec:
                blocks.append((cur_stores, cur_inputs))
                cur_stores = []
                cur_inputs = []
                pending_loads = []
                in_opspec = False
            lm = _KLOAD_RE.search(line)
            if lm:
                pending_loads.append(lm.group(1))
                continue
            sm = _KSTORE_RE.search(line)
            if sm:
                cur_stores.append(sm.group(1))
                cur_inputs.append(pending_loads)
                pending_loads = []
    if in_opspec and cur_stores:
        blocks.append((cur_stores, cur_inputs))

    # Phase 2: ordered kernel dirs from the Generating sdsc_0.json lines.
    kernel_dirs: list[str] = []
    for line in lines:
        gm = _GENERATING_RE.search(line)
        if gm:
            kernel_dirs.append(Path(gm.group(1)).parent.name)

    result: dict[str, tuple[list[str], list[list[str]]]] = {}
    for dir_name, block in zip(kernel_dirs, blocks):
        result[dir_name] = block
    return result


def _compress_core_id_to_wk_slice(wk_slice: dict) -> str:
    """Render coreIdToWkSlice as a compact per-dim range expression.

    For each dim, computes the unique sorted values across all cores and
    renders them as `dim=lo:hi` (or `dim=val` when constant). Falls back
    to `dim=core_id` when values are exactly 0..N-1.
    """
    if not wk_slice:
        return ""
    cores = sorted(wk_slice, key=lambda k: int(k))
    if not cores:
        return ""
    dims = list(wk_slice[cores[0]].keys())
    parts = []
    for d in dims:
        values = [wk_slice[c].get(d, 0) for c in cores]
        unique = sorted(set(values))
        if len(unique) == 1:
            parts.append(f"{d}={unique[0]}")
        elif values == list(range(len(cores))):
            parts.append(f"{d}=core_id")
        else:
            parts.append(f"{{{d}={unique[0]}:{unique[-1]}}}")
    return " ".join(parts)


def _extract_tensor_idx_to_role(op_data: dict) -> dict:
    """Extract tensor ldsIdx to role mapping from computeOp_ and primaryDsInfo_.

    Returns a dict mapping ldsIdx to role (INPUT/OUTPUT).
    """
    lds_idx_to_role = {}

    # computeOp_ is the authoritative source for data-flow roles.
    if "computeOp_" in op_data:
        compute_ops = op_data["computeOp_"]
        if isinstance(compute_ops, list):
            compute_op = compute_ops[0] if compute_ops else {}
        else:
            compute_op = compute_ops

        input_indices = set()
        for labeled_ds in compute_op.get("inputLabeledDs", []):
            if "-idx" in labeled_ds:
                idx_str = labeled_ds.split("-idx")[1]
                input_indices.add(int(idx_str))

        output_indices = set()
        for labeled_ds in compute_op.get("outputLabeledDs", []):
            if "-idx" in labeled_ds:
                idx_str = labeled_ds.split("-idx")[1]
                output_indices.add(int(idx_str))

        for lds_idx in range(10):
            if lds_idx in output_indices:
                lds_idx_to_role[lds_idx] = "OUTPUT"
            elif lds_idx in input_indices:
                lds_idx_to_role[lds_idx] = "INPUT"

    # Fall back to primaryDsInfo roles if still not mapped.
    if "primaryDsInfo_" in op_data:
        for role_idx, (role, _) in enumerate(op_data["primaryDsInfo_"].items()):
            if role_idx not in lds_idx_to_role:
                lds_idx_to_role[role_idx] = role

    return lds_idx_to_role


def _extract_lds_idx_to_ds_type(op_data: dict) -> dict:
    """Return a dict mapping ldsIdx to dsType_ from labeledDs_.

    dsType_ is a key into primaryDsInfo_ for layout lookup - it is NOT a
    data-flow role. Use _extract_tensor_idx_to_role for INPUT/OUTPUT labels.
    """
    result = {}
    if "labeledDs_" in op_data and isinstance(op_data["labeledDs_"], list):
        for labeled_ds in op_data["labeledDs_"]:
            lds_idx = labeled_ds.get("ldsIdx_", -1)
            ds_type = labeled_ds.get("dsType_", "")
            if lds_idx >= 0 and ds_type:
                result[lds_idx] = ds_type
    return result


def _extract_operation_data(op_data: dict, outer: dict | None = None) -> dict:
    """Extract common operation data from op_data dict.

    `outer` is the container that holds this op's `dscs_` list; op-level
    fields like `numWkSlicesPerDim_` live there, not on op_data itself.
    """
    # Extract components and their details
    components = []
    component_details = []
    if "scheduleTree_" in op_data:
        for node in op_data["scheduleTree_"]:
            if isinstance(node, dict):
                comp = node.get("component_", "")
                if comp:
                    components.append(comp)
                    node_name = node.get("name_", "")
                    node_type = node.get("nodeType_", "")
                    component_details.append(
                        {
                            "name": node_name,
                            "type": node_type,
                            "component": comp,
                        }
                    )

    # Extract labeledDs_ to map ldsIdx to memOrg and scale.
    # `scale_` is a per-dim list indexed by the tensor's own
    # `layoutDimOrder_`. A non-positive entry (e.g. -1, -2) marks that dim
    # as broadcast/reduced - its physical extent is 1, not N_[dim]. This is
    # how reduction outputs (max/sum) and broadcast operands (the
    # subtrahend in sub, the divisor in realdiv) collapse to 1 along the
    # reduction axis.
    lds_idx_to_mem_org = {}
    lds_idx_to_scale: dict[int, list] = {}
    # Element format and width per tensor (`dataFormat_`, e.g. SEN169_FP16,
    # and `wordLength` in bytes) for the Tile Size and Format columns.
    lds_idx_to_format: dict[int, tuple[str, int | None]] = {}
    if "labeledDs_" in op_data and isinstance(op_data["labeledDs_"], list):
        for labeled_ds in op_data["labeledDs_"]:
            lds_idx = labeled_ds.get("ldsIdx_", -1)
            word_length = labeled_ds.get("wordLength")
            lds_idx_to_format[lds_idx] = (
                str(labeled_ds.get("dataFormat_", "") or ""),
                word_length if isinstance(word_length, int) else None,
            )
            mem_org = labeled_ds.get("memOrg_", {})
            if mem_org:
                mem_org_str = "/".join(sorted(mem_org.keys()))
                lds_idx_to_mem_org[lds_idx] = mem_org_str
            scale = labeled_ds.get("scale_")
            if isinstance(scale, list):
                lds_idx_to_scale[lds_idx] = scale

    # coreIdToWkSlice_ lives on the outer container alongside
    # numWkSlicesPerDim_ (the per-allocate copy in
    # coordinates_.coreIdToWkSlice_ is empty in current artifacts).
    op_core_id_to_wk_slice = {}
    if isinstance(outer, dict):
        op_core_id_to_wk_slice = outer.get("coreIdToWkSlice_", {}) or {}
    if not op_core_id_to_wk_slice:
        op_core_id_to_wk_slice = op_data.get("coreIdToWkSlice_", {}) or {}

    # Extract allocate nodes indexed by ldsIdx
    allocate_nodes = {}
    if "scheduleTree_" in op_data:
        for node in op_data["scheduleTree_"]:
            if isinstance(node, dict) and node.get("nodeType_") == "allocate":
                lds_idx = node.get("ldsIdx_", -1)
                node_wk_slice = (
                    node.get("coordinates_", {}).get("coreIdToWkSlice_", {}) or {}
                )
                # Per-tensor per-dim extent lives at
                # coordinates_.coordInfo.<dim>.folds.dim_prop_func[0].Affine.alpha_.
                # This is the authoritative source - N_ holds op-level
                # extents, but a broadcast/reduced tensor will have a
                # smaller extent here (e.g. 1 instead of N_["mb"]).
                tensor_extents: dict[str, int] = {}
                coord_info = node.get("coordinates_", {}).get("coordInfo", {}) or {}
                for dim_name, dim_info in coord_info.items():
                    folds = dim_info.get("folds", {}) or {}
                    func_list = folds.get("dim_prop_func", []) or []
                    if func_list and isinstance(func_list[0], dict):
                        affine = func_list[0].get("Affine", {})
                        alpha = affine.get("alpha_")
                        if isinstance(alpha, int):
                            tensor_extents[dim_name] = alpha
                allocate_nodes[lds_idx] = {
                    "name": node.get("name_", "").lower().replace("-", "_"),
                    "component": node.get("component_", ""),
                    "address": "",
                    "layoutDimOrder": node.get("layoutDimOrder_", []),
                    "maxDimSizes": node.get("maxDimSizes_", []),
                    "coreIdToWkSlice": node_wk_slice or op_core_id_to_wk_slice,
                    "tensor_extents": tensor_extents,
                }
                if "startAddressCoreCorelet_" in node:
                    addr_data = node["startAddressCoreCorelet_"].get("data_", {})
                    if addr_data:
                        allocate_nodes[lds_idx]["address"] = str(
                            next(iter(addr_data.values()))
                        )

    # Get tensor index to role mapping from computeOp_
    lds_idx_to_role = _extract_tensor_idx_to_role(op_data)
    # dsType_ is a key into primaryDsInfo_ for layout lookup - distinct from
    # the data-flow role (INPUT/OUTPUT) assigned by computeOp_.
    lds_idx_to_ds_type = _extract_lds_idx_to_ds_type(op_data)

    # Extract tensors - one entry per ldsIdx (each allocation)
    tensors = []

    for lds_idx, alloc_node in sorted(allocate_nodes.items()):
        # Get the role for this ldsIdx
        role = lds_idx_to_role.get(lds_idx, "UNKNOWN")

        # Get layout and stick info from primaryDsInfo_ using dsType_ as the
        # lookup key (not role). dsType_ directly names the primaryDsInfo_
        # entry that describes this tensor's memory layout. Fall back to
        # layoutDimOrder_ on the allocate node itself when unavailable.
        layout_dims = ""
        stick_dims = ""
        stick_size = None
        if "primaryDsInfo_" in op_data:
            ds_type = lds_idx_to_ds_type.get(lds_idx, role)
            tensor_data = op_data["primaryDsInfo_"].get(ds_type, {})
            layout_dims = ", ".join(tensor_data.get("layoutDimOrder_", []))
            stick_dims = ", ".join(tensor_data.get("stickDimOrder_", []))
            sizes = tensor_data.get("stickSize_") or []
            if sizes and isinstance(sizes[0], int):
                stick_size = sizes[0]
        if not layout_dims:
            layout_dims = ", ".join(alloc_node.get("layoutDimOrder", []))
        alloc_layout = alloc_node.get("layoutDimOrder", []) or []

        # One allocation per ldsIdx
        tensors.append(
            {
                "name": alloc_node["name"],
                "lds_idx": lds_idx,
                "role": role,
                "layout": layout_dims,
                "sticks": stick_dims,
                "stick_size": stick_size,
                "component": alloc_node["component"],
                "address": alloc_node["address"],
                "mem_org": lds_idx_to_mem_org.get(lds_idx, ""),
                "max_dim_sizes": alloc_node.get("maxDimSizes", []),
                "layout_dim_order": alloc_layout,
                "core_id_to_wk_slice": alloc_node.get("coreIdToWkSlice", {}),
                # Per-tensor scale_ list, parallel to layout_dim_order.
                # Kept around because the renderer still uses it to suppress
                # the wkSlices `/N` annotation on a reduced/broadcast dim.
                "scale": lds_idx_to_scale.get(lds_idx, []),
                # Per-tensor per-dim extent (e.g. {"mb": 1, "out": 1024} for
                # a max output that's been reduced over mb). Collected from
                # coordInfo.<dim>.folds.dim_prop_func[0].Affine.alpha_.
                "tensor_extents": alloc_node.get("tensor_extents", {}),
                "data_format": lds_idx_to_format.get(lds_idx, ("", None))[0],
                "word_length": lds_idx_to_format.get(lds_idx, ("", None))[1],
            }
        )

    # Extract address
    addr = ""
    if op_data.get("scheduleTree_"):
        node = op_data["scheduleTree_"][0]
        if "startAddressCoreCorelet_" in node:
            addr_data = node["startAddressCoreCorelet_"].get("data_", {})
            if addr_data:
                addr = str(next(iter(addr_data.values())))

    # Extract dimensions
    dims = ""
    if "N_" in op_data:
        n_info = op_data["N_"]
        dim_parts = []
        for k in ["mb_", "out_", "in_", "k_", "h_", "w_"]:
            if k in n_info:
                dim_parts.append(f"{k[:-1]}:{n_info[k]}")
        dims = ", ".join(dim_parts)

    # Op-level work slices per dim lives on the outer container (and
    # occasionally on op_data itself in older artifacts).
    wk_slices = {}
    if isinstance(outer, dict):
        wk_slices = outer.get("numWkSlicesPerDim_", {}) or {}
    if not wk_slices:
        wk_slices = op_data.get("numWkSlicesPerDim_", {}) or {}

    # Per-dim host extent from N_ (e.g. {mb: 1024, out: 512}). Used to
    # render layouts with extents instead of op-local dim names, so the
    # same physical axis labelled "mb" in one op and "out" in another
    # shows up as the same number in the layout column.
    dim_extents: dict[str, int] = {}
    if "N_" in op_data:
        for k, v in op_data["N_"].items():
            if k.endswith("_") and isinstance(v, int):
                dim_extents[k[:-1]] = v

    return {
        "components": components,
        "components_str": ", ".join(components),
        "component_details": component_details,
        "tensors": tensors,
        "tensors_str": ", ".join([t["role"] for t in tensors]),
        "address": addr,
        "dims": dims,
        "wk_slices": wk_slices,
        "dim_extents": dim_extents,
    }


def extract_ops_from_sdsc(data: dict, file_name: str) -> list:
    """Extract operations from sdsc.json files (both standard and alternative formats)."""
    operations = []

    # Format 1: Standard format with dscs_ or identity/dscs_
    identity = data.get("identity", {})
    dscs_list = data.get("dscs_", identity.get("dscs_", []))
    outer = data if "dscs_" in data else identity if "dscs_" in identity else data

    if dscs_list:
        # Process standard format
        for dsc_idx, dsc_dict in enumerate(dscs_list):
            for op_name, op_data in dsc_dict.items():
                if not isinstance(op_data, dict):
                    continue

                op_info = _extract_operation_data(op_data, outer)
                operations.append(
                    {
                        "file": file_name,
                        "op_name": op_name,
                        "dsc_idx": dsc_idx,
                        "cores": op_data.get("numCoresUsed_", 0),
                        "corelets": op_data.get("numCoreletsUsed_", 0),
                        **op_info,
                    }
                )
    else:
        # Format 2: Alternative format with operation key at top level.
        # Any top-level key whose value contains "dscs_" is treated as an op.
        for op_key, op_container in data.items():
            if not isinstance(op_container, dict):
                continue
            if op_container.get("dscs_"):
                nested_dscs = op_container["dscs_"]
                for dsc_idx, dsc_dict in enumerate(nested_dscs):
                    for nested_op_name, nested_op_data in dsc_dict.items():
                        if not isinstance(nested_op_data, dict):
                            continue

                        op_info = _extract_operation_data(nested_op_data, op_container)
                        operations.append(
                            {
                                "file": file_name,
                                "op_name": nested_op_name,
                                "dsc_idx": dsc_idx,
                                "cores": nested_op_data.get("numCoresUsed_", 0),
                                "corelets": nested_op_data.get("numCoreletsUsed_", 0),
                                **op_info,
                            }
                        )
            elif op_container.get("datadscs_"):
                # Data-only op (no compute dscs_): a cross-core / copy-out
                # data movement such as an Stcdp relayout (STCDPOpLx, LX->LX)
                # or output copy-out (STCDPOpHBM, LX->HBM). These carry no
                # compute descriptor; surface them so the summary shows the
                # full op sequence, not just compute ops.
                op_info = _extract_datadsc_op(op_container)
                if op_info is not None:
                    operations.append(
                        {
                            "file": file_name,
                            "op_name": _short_datadsc_name(op_key, op_info),
                            "dsc_idx": 0,
                            "cores": op_container.get("numCoresUsed_", 0),
                            "corelets": op_container.get("numCoreletsUsed_", 0),
                            **op_info,
                        }
                    )

    return operations


def _short_datadsc_name(op_key: str, op_info: dict) -> str:
    """Shorten a verbose relayout op key for display.

    deeptools names a relayout like
    `BatchMatMulV2_QC_0_inpLds_1_MNI_LeafQC-unfold0_0_0_-LxRelayout`, which is
    far too wide for the table. Keep only the relayed tensor name plus
    `Relayout`. Other data-only op keys (e.g. `permute_1_reducedStcdpNode`) are
    left as-is.
    """
    if "Relayout" not in op_key:
        return op_key
    tensors = op_info.get("tensors", [])
    tensor = tensors[0]["name"] if tensors else ""
    return " ".join(p for p in (tensor, "Relayout") if p)


def _distinct_slots(wk_slice: dict) -> int:
    """Number of distinct work-slice slots across cores (1 = every core holds
    the same piece, i.e. the tensor is whole/replicated on each core)."""
    if not wk_slice:
        return 1
    return len({tuple(sorted(v.items())) for v in wk_slice.values()}) or 1


def _loc(tensor: dict) -> str:
    return (tensor.get("component") or "?").upper()


# Ops that move data without reducing it (see the Layout column's
# reduced-dim rule): a per-tensor extent below the op-level share is a
# per-core partition for these, never a reduction.
_DATA_MOVE_OPS = frozenset({"shuffle", "identity", "restickifyophbm", "restickifyoplx"})


def _tile_shape(tensor: dict, op: dict) -> list[int] | None:
    """Per-core tile extents in the allocation's own dimension order.

    Each dim is the smaller of the tensor's own extent (coordInfo alpha, which
    a relayout already records per core and a reduction records as 1) and the
    op-level share ceil(N_[dim] / numWkSlicesPerDim_[dim]) that a compute op
    hands each core. ``None`` when a dim's extent is unknown.
    """
    order = tensor.get("layout_dim_order") or []
    if not order:
        return None
    extents = tensor.get("tensor_extents") or {}
    full_extents = op.get("dim_extents") or {}
    wk_slices = op.get("wk_slices") or {}
    shape: list[int] = []
    for dim in order:
        alpha = extents.get(dim)
        full = full_extents.get(dim)
        n = wk_slices.get(dim, 1) or 1
        per_core = -(-full // n) if isinstance(full, int) and n > 0 else None
        candidates = [v for v in (alpha, per_core) if isinstance(v, int) and v > 0]
        if not candidates:
            return None
        shape.append(min(candidates))
    return shape


def _tile_size_label(shape: list[int] | None, word_length: int | None) -> str:
    if not shape or not isinstance(word_length, int):
        return ""
    elems = 1
    for v in shape:
        elems *= v
    return f"{elems * word_length / 1024:.2f} KB"


def _format_label(tensor: dict) -> str:
    fmt = tensor.get("data_format") or ""
    wl = tensor.get("word_length")
    if fmt and isinstance(wl, int):
        return f"{fmt} ({wl}B)"
    return fmt or (f"{wl}B" if isinstance(wl, int) else "")


# The planner's per-core LX budget: Deeptools' allocatable capacity (2 MiB minus
# the 64 KiB program/debug reservation) times the frontend fraction
# (1 - DXP_LX_FRAC_AVAIL, 0.8 by default), rounded up to the 128 B allocation
# granularity. Mirrors allocator._lx_planning_size() without importing torch.
_LX_CAPACITY_DEFAULT = 1_625_344


def _bytes_label(n: int | None) -> str:
    if not isinstance(n, int):
        return ""
    if n < 1024:
        return f"{n} B"
    return f"{n / 1024:.2f} KB"


def _padded_tile(tensor: dict, tile: list[int] | None) -> list[int] | None:
    """The tile as LX stores it: every stick dimension rounded up to a whole
    number of sticks (the allocator budgets stick-padded ``device_size``)."""
    if not tile:
        return None
    order = tensor.get("layout_dim_order") or []
    sticks = tensor.get("sticks") or ""
    stick_dims = set(sticks.split(", ")) if sticks else set()
    size = tensor.get("stick_size")
    if not stick_dims or not isinstance(size, int) or size <= 0:
        return list(tile)
    padded = []
    for dim, extent in zip(order, tile):
        if dim in stick_dims:
            padded.append(-(-extent // size) * size)
        else:
            padded.append(extent)
    return padded


def _prod(values: list[int] | None) -> int | None:
    if not values:
        return None
    out = 1
    for v in values:
        out *= v
    return out


def _sdsc_stem(op: dict) -> str:
    stem = Path(op["file"]).stem
    return "...Relayout" if "Relayout" in stem else stem


def _kernel_id(op: dict) -> str:
    """Short id of the kernel directory an sdsc file belongs to: the hash
    prefix of ``<hash>_sdsc_fused_...``, or the directory name itself."""
    name = Path(op["file"]).parent.name
    head = name.split("_", 1)[0]
    return head if 6 <= len(head) <= 12 else name[:12]


def enrich_ops(all_ops: list, lx_capacity: int = _LX_CAPACITY_DEFAULT) -> None:
    """Derive the per-tensor and per-op fields the renderers and the TUI show,
    from what the SDSC parse already holds.

    Per tensor: ``tile`` (per-core tile, allocation dim order), ``tile_bytes``,
    ``footprint_bytes`` (stick-padded, what LX actually holds), ``addr_int``,
    ``addr_label`` (an LX range ``0xS-0xE``, ``hbm#N`` for a placeholder, hex
    otherwise), ``producer`` (``<- op sdsc_N`` for an LX input, by address) and
    ``consumer`` (``-> op sdsc_N`` for an LX output).
    Per op: ``kernel`` (short kernel id), ``lx_live_bytes`` (LX bytes live
    while this op runs: every LX range first touched at or before it and last
    touched at or after it, within the same kernel), ``lx_live_pct``.
    """
    for op in all_ops:
        op["kernel"] = _kernel_id(op)
        for tensor in op.get("tensors") or []:
            tile = _tile_shape(tensor, op)
            tensor["tile"] = tile
            wl = tensor.get("word_length")
            elems = _prod(tile)
            tensor["tile_bytes"] = elems * wl if elems and isinstance(wl, int) else None
            padded = _prod(_padded_tile(tensor, tile))
            tensor["footprint_bytes"] = (
                padded * wl if padded and isinstance(wl, int) else None
            )
            addr = tensor.get("address")
            try:
                addr_int = int(addr) if addr not in ("", None) else None
            except (TypeError, ValueError):
                addr_int = None
            tensor["addr_int"] = addr_int
            if addr_int is None:
                tensor["addr_label"] = str(addr) if addr not in (None,) else ""
            elif addr_int < 0:
                tensor["addr_label"] = f"hbm#{-addr_int}"
            elif tensor.get("component") == "lx" and tensor["footprint_bytes"]:
                end = addr_int + tensor["footprint_bytes"]
                tensor["addr_label"] = f"0x{addr_int:x}-0x{end:x}"
            else:
                tensor["addr_label"] = f"0x{addr_int:x}"
            tensor["producer"] = ""
            tensor["consumer"] = ""

    def _lx_key(t: dict):
        if t.get("component") != "lx" or not isinstance(t.get("addr_int"), int):
            return None
        return (t["addr_int"], t.get("footprint_bytes") or 0)

    # Producer / consumer by LX address, within a kernel.
    for op_idx, op in enumerate(all_ops):
        kernel = op["kernel"]
        for tensor in op.get("tensors") or []:
            key = _lx_key(tensor)
            if key is None:
                continue
            if tensor["role"] == "INPUT":
                for earlier_idx in range(op_idx - 1, -1, -1):
                    earlier = all_ops[earlier_idx]
                    if earlier["kernel"] != kernel:
                        continue
                    if any(
                        t["role"] == "OUTPUT"
                        and _lx_key(t) is not None
                        and _lx_key(t)[0] == key[0]
                        for t in earlier.get("tensors") or []
                    ):
                        tensor["producer"] = (
                            f"<- {earlier['op_name']} {_sdsc_stem(earlier)}"
                        )
                        break
            elif tensor["role"] == "OUTPUT":
                for later in all_ops[op_idx + 1 :]:
                    if later["kernel"] != kernel:
                        continue
                    if any(
                        t["role"] == "INPUT"
                        and _lx_key(t) is not None
                        and _lx_key(t)[0] == key[0]
                        for t in later.get("tensors") or []
                    ):
                        tensor["consumer"] = (
                            f"-> {later['op_name']} {_sdsc_stem(later)}"
                        )
                        break

    # LX occupancy per op: lifetimes of every (kernel, address, footprint).
    first_last: dict[tuple, list[int]] = {}
    for op_idx, op in enumerate(all_ops):
        for tensor in op.get("tensors") or []:
            key = _lx_key(tensor)
            if key is None:
                continue
            k = (op["kernel"],) + key
            span = first_last.setdefault(k, [op_idx, op_idx])
            span[1] = op_idx
    for op_idx, op in enumerate(all_ops):
        live = [
            (k[1], k[1] + k[2])
            for k, (lo, hi) in first_last.items()
            if k[0] == op["kernel"] and lo <= op_idx <= hi and k[2] > 0
        ]
        total = 0
        cur_end: int | None = None
        for start, end in sorted(live):
            if cur_end is not None and start < cur_end:
                if end > cur_end:
                    total += end - cur_end
                    cur_end = end
            else:
                total += end - start
                cur_end = end
        op["lx_live_bytes"] = total
        op["lx_live_pct"] = (100.0 * total / lx_capacity) if lx_capacity else None
        op["lx_capacity"] = lx_capacity


# --------------------------------------------------------------- .py buffer names
_IR_CHAIN_RE = re.compile(r"ir_chain=\(([^)]*)\)")
_TARG_FIELD_RE = {
    "is_input": re.compile(r"is_input=(True|False)"),
    "name": re.compile(r"\bname='(\w+)'"),
    "allocation": re.compile(r"allocation=(\{[^}]*\})"),
    "device_size": re.compile(r"device_size=(\[[^\]]*\])"),
}


def _parse_py_opspecs(
    base_dir: Path,
) -> dict[str, list[tuple[Path, tuple]]]:
    """Per-kernel buffer names straight from the Inductor ``.py``: for each leaf
    OpSpec in definition order, its output buffer (the last entry of
    ``debug_handle.ir_chain``) and its input names (an explicit ``name=`` for a
    graph input; for an intermediate, the output buffer of the earlier OpSpec
    whose output has the same ``allocation`` and ``device_size``). The DEBUG
    log's names take precedence when a log is given; this fills in without one.
    """

    def _scan(root: Path):
        found: dict[str, list[tuple[Path, tuple]]] = {}
        for py_file in root.glob("**/*.py"):
            try:
                text = py_file.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            for m in _SDSC_CALL_RE.finditer(text):
                name = m.group(1)
                lb = text.find("[", m.end())
                if lb < 0:
                    continue
                try:
                    rb = _match_bracket(text, lb)
                except (ValueError, IndexError):
                    continue
                arg_text = text[lb:rb]
                stores: list[str] = []
                inputs: list[list[str]] = []
                ir_names: list[str] = []
                kinds: list[str] = []
                by_alloc: dict = {}
                for om in re.finditer(r"\bOpSpec\(", arg_text):
                    try:
                        end = _match_paren(arg_text, om.end() - 1)
                    except (ValueError, IndexError):
                        break
                    block = arg_text[om.start() : end]
                    kind_m = re.search(r"\bop='([^']+)'", block)
                    kinds.append(kind_m.group(1) if kind_m else "")
                    chain = _IR_CHAIN_RE.search(block)
                    out_buf = ""
                    ir_name = ""
                    if chain:
                        names = re.findall(r"'(\w+)'", chain.group(1))
                        # ('mul', 'buf5'): the Inductor node name first, the
                        # buffer it stores last. A chain of views has no buf.
                        ir_name = names[0] if names else ""
                        out_buf = (
                            names[-1] if names and names[-1].startswith("buf") else ""
                        )
                    op_inputs: list[str] = []
                    for am in re.finditer(r"TensorArg\(", block):
                        try:
                            aend = _match_paren(block, am.end() - 1)
                        except (ValueError, IndexError):
                            break
                        arg = block[am.start() : aend]
                        fields = {
                            k: (r.search(arg).group(1) if r.search(arg) else "")
                            for k, r in _TARG_FIELD_RE.items()
                        }
                        akey = (fields["allocation"], fields["device_size"])
                        if fields["is_input"] == "True":
                            # Exact (allocation, size) first; a view of the
                            # producer's output keeps the allocation but not
                            # the size, so fall back to the most recent writer
                            # of that allocation.
                            op_inputs.append(
                                fields["name"]
                                or by_alloc.get(akey)
                                or by_alloc.get(fields["allocation"], "")
                            )
                        elif out_buf and fields["allocation"]:
                            by_alloc[akey] = out_buf
                            by_alloc[fields["allocation"]] = out_buf
                    stores.append(out_buf)
                    inputs.append(op_inputs)
                    ir_names.append(ir_name)
                if stores:
                    found.setdefault(name, []).append(
                        (py_file.parent.parent, (stores, inputs, ir_names, kinds))
                    )
        return found

    kernels = _scan(base_dir)
    root = base_dir.resolve()
    for _ in range(4):
        if kernels:
            break
        parent = root.parent
        if parent == root:
            break
        root = parent
        kernels = _scan(root)
    return kernels


def _kind_equal(sdsc_kind: str, opspec_kind: str) -> bool:
    a, b = sdsc_kind.lower(), opspec_kind.lower()
    # An LX relayout materializes an identity OpSpec as a shuffle SDSC.
    return a == b or (a == "shuffle" and b == "identity")


def _align_leaves(sdsc_kinds: list[str], leaf_kinds: list[str]) -> list[int | None]:
    """sdsc index -> expanded-leaf index when the counts differ (relayout
    materialization inserts shuffles and drops ops after the .py is written).
    Longest-common-subsequence alignment on op kinds; an unaligned sdsc op
    maps to ``None``."""
    n, m = len(sdsc_kinds), len(leaf_kinds)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if _kind_equal(sdsc_kinds[i], leaf_kinds[j]):
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    out: list[int | None] = [None] * n
    i = j = 0
    while i < n and j < m:
        if _kind_equal(sdsc_kinds[i], leaf_kinds[j]):
            out[i] = j
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


# ------------------------------------------------------------------ cost dump
_COST_OP_RE = re.compile(r"^\s{2}(\S+)\s+read=(\d+)B write=(\d+)B lx=(\d+)B")
_COST_ARG_RE = re.compile(r"^\s{6}(input|output|INPUT|OUTPUT)\s+(\S+)")
_COST_T_RE = re.compile(r"T_model = ([0-9.]+) us")
_COST_TOTAL_RE = re.compile(
    r"predicted total:\s+([0-9.]+) us over (\d+) kernel\(s\), (\d+) op\(s\)"
)
_COST_ATTR_RE = re.compile(
    r"^\s+([0-9.]+)\s+(\S+)\s+HBM\s+(-|[0-9.]+ [KMG]?B)(?:\s+LX\s+([0-9.]+ [KMG]?B))?"
)
_FEATURES_BANNER = "Cost model features + prediction"
_RUNTIME_BANNER = "Cost model: predicted runtime"
_UNIT = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9}


def _size_to_bytes(label: str) -> float | None:
    if label == "-":
        return 0.0
    try:
        num, unit = label.split()
        return float(num) * _UNIT[unit]
    except (ValueError, KeyError):
        return None


def _parse_cost_dump(path: Path) -> list[dict]:
    """Blocks of a captured ``SPYRE_DUMP_COST=1`` stream.

    Per kernel the dump writes a *runtime* block (``predicted total`` and one
    attributed line per op: ``us  name  HBM bytes  LX bytes``) followed by a
    *features* block (per op ``name read=..B write=..B lx=..B`` with its
    ``output opN`` / ``input bufN|argN`` lines and the bundle's ``T_model``).
    Returns one dict per kernel: {"ops": [{name, read, write, lx, inputs,
    output, attributed_us}], "t_us", "n_ops"}. Attributed times join to ops
    by name, and by HBM bytes when a name repeats (``Pointwise``).
    """
    blocks: list[dict] = []
    runtime: dict | None = None
    cur: dict | None = None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return blocks
    for line in lines:
        if _RUNTIME_BANNER in line:
            runtime = {"attr": [], "t_us": None}
            cur = None
            continue
        if _FEATURES_BANNER in line:
            cur = {"ops": [], "t_us": None, "runtime": runtime}
            blocks.append(cur)
            runtime = None
            continue
        if cur is None and runtime is not None:
            m = _COST_TOTAL_RE.search(line)
            if m:
                runtime["t_us"] = float(m.group(1))
                continue
            m = _COST_ATTR_RE.match(line)
            if m:
                runtime["attr"].append(
                    (float(m.group(1)), m.group(2), _size_to_bytes(m.group(3)))
                )
            continue
        if cur is None:
            continue
        m = _COST_OP_RE.match(line)
        if m:
            cur["ops"].append(
                {
                    "name": m.group(1),
                    "read": int(m.group(2)),
                    "write": int(m.group(3)),
                    "lx": int(m.group(4)),
                    "inputs": [],
                    "output": "",
                    "attributed_us": None,
                }
            )
            continue
        m = _COST_ARG_RE.match(line)
        if m and cur["ops"]:
            role, name = m.group(1).lower(), m.group(2)
            if role == "output":
                cur["ops"][-1]["output"] = name
            else:
                cur["ops"][-1]["inputs"].append(name)
            continue
        m = _COST_T_RE.search(line)
        if m:
            cur["t_us"] = float(m.group(1))
    for block in blocks:
        rt = block.pop("runtime", None) or {}
        if block["t_us"] is None:
            block["t_us"] = rt.get("t_us")
        block["n_ops"] = len(block["ops"])
        attr = list(rt.get("attr") or [])
        # Unique names first, then the rest by closest HBM byte count.
        by_name: dict[str, list] = {}
        for entry in attr:
            by_name.setdefault(entry[1], []).append(entry)
        for cop in block["ops"]:
            cands = by_name.get(cop["name"]) or []
            if not cands:
                continue
            if len(cands) == 1:
                cop["attributed_us"] = cands[0][0]
                continue
            hbm = cop["read"] + cop["write"]
            best = min(
                cands,
                key=lambda e: abs((e[2] if e[2] is not None else -1) - hbm),
            )
            cop["attributed_us"] = best[0]
            cands.remove(best)
    return blocks


def _join_cost(all_ops: list, blocks: list[dict]) -> int:
    """Attach ``op["cost"]`` to every SDSC op that matches a dumped op.

    The dump names an op by its Inductor node (``index_1``, ``Pointwise``) and
    its inputs by buffer (``buf16``, ``arg3_1``); the SDSC side has the same
    node name from the ``.py``'s ``ir_chain`` and the same input buffers. Score
    a candidate by name match plus shared input buffers, best score wins, ties
    to the earlier dump block. Returns the number of ops matched."""
    matched = 0
    used_single_blocks: set[int] = set()
    for op in all_ops:
        ir_name = op.get("ir_name") or ""
        in_bufs = {b for b in (op.get("in_bufs") or []) if b}
        best = None
        best_score = 0
        for block in blocks:
            for cop in block["ops"]:
                score = 0
                if ir_name and cop["name"] == ir_name:
                    score += 2
                overlap = len(in_bufs & set(cop["inputs"]))
                score += overlap
                if in_bufs and set(cop["inputs"]) & in_bufs == in_bufs:
                    score += 1
                if score > best_score:
                    best_score = score
                    best = (cop, block)
        if best is None and int(op.get("cores", 0) or 0) <= 1:
            # A single-core kernel (D2D copy) has no buffer names to match
            # on; its HBM tiles are whole tensors, so the byte count is. Equal
            # copies are told apart by order: kernels compile in the order the
            # dump wrote their blocks, and a block serves one kernel.
            hbm_bytes = sum(
                t.get("tile_bytes") or 0
                for t in op.get("tensors") or []
                if t.get("component") == "hbm" and t["role"] in ("INPUT", "OUTPUT")
            )
            for allow_reuse in (False, True):
                for block in blocks:
                    if len(block["ops"]) != 1:
                        continue
                    if not allow_reuse and id(block) in used_single_blocks:
                        continue
                    cop = block["ops"][0]
                    if hbm_bytes and cop["read"] + cop["write"] == hbm_bytes:
                        best = (cop, block)
                        used_single_blocks.add(id(block))
                        break
                if best is not None:
                    break
        if best is None:
            continue
        cop, block = best
        op["cost"] = {
            "read": cop["read"],
            "write": cop["write"],
            "lx": cop["lx"],
            "t_us": block["t_us"],
            "attributed_us": cop["attributed_us"],
            "bundle_ops": block["n_ops"],
            "dump_name": cop["name"],
        }
        matched += 1
    return matched


# --------------------------------------------------------- cost-expression dump
try:
    import sympy as _sympy

    class RelayoutCharge(_sympy.Function):
        """Stand-in for ``plan_solver.RelayoutCharge`` so a dumped objective
        parses and evaluates here without importing torch_spyre."""

        @classmethod
        def eval(cls, is_lx, division, *prices):
            if is_lx.is_Number:
                if is_lx.is_zero:
                    return _sympy.S.Zero
                if division.is_Integer:
                    i = int(division)
                    return is_lx * (prices[i] if 0 <= i < len(prices) else 0)
            return None

except ImportError:  # pragma: no cover - sympy ships with torch
    _sympy = None


def _parse_expr(text: str):
    if _sympy is None or not text:
        return None
    try:
        return _sympy.parse_expr(text, local_dict={"RelayoutCharge": RelayoutCharge})
    except Exception:  # noqa: BLE001 - a dump from another version must not kill the report
        return None


def _eval_expr(expr, bindings: dict) -> float | None:
    if expr is None:
        return None
    try:
        return float(expr.xreplace(bindings).evalf())
    except (TypeError, ValueError, AttributeError):
        return None


def _parse_cost_expr_dump(path: Path) -> list[dict]:
    """Records of ``SPYRE_DUMP_COST_EXPR_FILE`` (one JSON line per co-optimized
    graph): the objective's per-bundle terms and relayout charges as sympy
    ``srepr`` strings, the solved symbol bindings, and the evaluated prices."""
    records: list[dict] = []
    try:
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return records


def _division_lines(out_buf: str, rec: dict, max_pairs: int = 6) -> list[str]:
    """The core division the solver gave this op's output buffer, the candidates
    it was chosen from, and per producer edge the division pairs the residency
    gate admitted. ``matches`` pairs are ``(parent index, consumer index)`` into
    the two buffers' candidate lists, so a pair that is not the chosen one is an
    LX residency the plan gave up by dividing the way it did. Producers that are
    not themselves solved buffers (graph inputs) are left out: they are never an
    LX residency source, so their missing edge says nothing."""

    divs = rec.get("divisions") or {}
    me = divs.get(out_buf)
    if not me:
        return []

    def _cand(d: dict, i) -> str:
        cores, labels = d.get("cores") or [], d.get("labels") or []
        if i is None or not 0 <= i < len(cores):
            return "?"
        lab = labels[i] if i < len(labels) else ""
        n = cores[i]
        shape = f" ({lab})" if lab and lab != "whole" else ""
        return f"{n} core{'' if n == 1 else 's'}{shape}"

    def _pairs(d: dict, me_d: dict, pairs: list) -> str:
        shown = ", ".join(
            f"{_cand(d, a)} -> {_cand(me_d, b)}" for a, b in pairs[:max_pairs]
        )
        rest = len(pairs) - max_pairs
        return shown + (f" (+{rest} more)" if rest > 0 else "")

    chosen = me.get("chosen")
    cores = me.get("cores") or []
    # The full candidate list runs to dozens of entries on a matmul; the useful
    # summary is how many there were and what core counts they spanned.
    span = ", ".join(str(n) for n in sorted(set(cores)))
    plural = "" if len(cores) == 1 else "s"
    core_word = "core" if span == "1" else "cores"
    lines = [
        f"{_cand(me, chosen)}, chosen from {len(cores)} candidate{plural} "
        + f"at {span} {core_word}"
    ]
    matches = me.get("matches") or {}
    for parent in sorted(set(me.get("parents") or []) | set(matches)):
        pd = divs.get(parent)
        if pd is None:
            continue
        taken = pd.get("chosen")
        if parent not in matches:
            # No edge was built at all: the producer was refused as a source
            # before any division pair was considered -- a residency reason on
            # it, or no comparable per-core view.
            lines.append(
                f"from {parent}: no edge, the gate refused this source "
                f"outright before any division pair was weighed"
            )
            continue
        pairs = matches[parent]
        if not pairs:
            lines.append(
                f"from {parent}: no compatible pair, so the gate forbids LX "
                f"residency across this edge whatever either side picks"
            )
            continue
        if any(a == taken and b == chosen for a, b in pairs):
            of_n = (
                "the only pair the gate admits"
                if len(pairs) == 1
                else f"one of {len(pairs)} admitted pairs"
            )
            lines.append(
                f"from {parent}: the plan took {_cand(pd, taken)} -> "
                f"{_cand(me, chosen)}, {of_n}"
            )
            continue
        lines.append(
            f"from {parent}: the plan took {_cand(pd, taken)} -> "
            f"{_cand(me, chosen)}, which is NOT among the "
            f"{len(pairs)} the gate admits: {_pairs(pd, me, pairs)}"
        )
    return lines


def _parse_probe_json(path: Path) -> list[dict]:
    """The per-graph records of an LX residency probe JSON.

    The probe is the older source of a buffer's residency REASON -- the
    allocator's own words for why a buffer is not in LX ("op not allowed",
    "spilled by solver (no residency benefit / no room)"). The cost-expression
    dump records the same reason since PR #4738, so this is only needed for runs
    captured before that. Nothing in the SDSC output records it: by then the
    decision is a fact, not a reason."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    graphs = data.get("graphs") if isinstance(data, dict) else None
    return [g for g in (graphs or []) if isinstance(g, dict) and g.get("buffers")]


def _reason_short(reason: str) -> str:
    """The residency reason as a table cell: the head of the phrase, which is
    the distinguishing part ("spilled by solver (no residency benefit / no
    room)" -> "spilled by solver"). The detail pane keeps the whole thing."""
    head = reason.split(" (")[0].strip()
    return head if len(head) <= 28 else head[:27] + "\u2026"


def _join_probe(all_ops: list, graphs: list[dict]) -> int:
    """Attach ``op["residency"]`` -- the probe's ``lx``/``reason``/``address``
    for the op's output buffer -- to every op whose kernel matched a graph.

    A graph is matched to a kernel by how much of the kernel's output-buffer
    set it covers, then by buffer-count proximity: a probe graph is a superset
    of the kernel's stores (it also carries the solver's relayout copies), so
    coverage alone would let the largest graph claim every kernel. Returns the
    number of ops annotated."""
    if not graphs:
        return 0
    by_kernel: dict[str, set[str]] = {}
    for op in all_ops:
        if op.get("out_buf"):
            by_kernel.setdefault(op.get("kernel", ""), set()).add(op["out_buf"])
    parsed = [
        (
            {b["name"] for b in g["buffers"] if b.get("name")},
            {b["name"]: b for b in g["buffers"] if b.get("name")},
        )
        for g in graphs
    ]
    scored = []
    for kernel, names in by_kernel.items():
        for idx, (g_names, _) in enumerate(parsed):
            hit = len(names & g_names)
            if not hit:
                continue
            scored.append(
                ((hit / len(names), -abs(len(g_names) - len(names)), hit), kernel, idx)
            )
    kernel_graph: dict[str, int] = {}
    used: set[int] = set()
    for _score, kernel, idx in sorted(scored, key=lambda x: x[0], reverse=True):
        if kernel in kernel_graph or idx in used:
            continue
        kernel_graph[kernel] = idx
        used.add(idx)
    annotated = 0
    for op in all_ops:
        idx = kernel_graph.get(op.get("kernel", ""))
        out_buf = op.get("out_buf")
        if idx is None or not out_buf:
            continue
        buf = parsed[idx][1].get(out_buf)
        if buf is None:
            continue
        op["residency"] = {
            "lx": bool(buf.get("lx")),
            "reason": buf.get("reason"),
            "address": buf.get("address"),
            "copy_of": buf.get("copy_of") or "",
        }
        annotated += 1
    return annotated


def _join_cost_expr(all_ops: list, records: list[dict]) -> int:
    """Attach ``op["objective"]`` for every op whose output buffer is in a
    dumped bundle: the bundle's symbolic term, its value under the solved
    bindings, the same term with this op's residency flipped, the symbols it
    depends on with their solved values, and the relayout charges whose source
    is this op. A record is matched to a kernel by output-buffer overlap.

    Returns ``(ops annotated, relayout summary)``. The summary counts the
    relayout copies of every MATCHED record -- the copies the solver priced,
    how many it kept (a copy is "fired" when it was given an LX address) and
    what they cost -- so the header can say what the plan paid for shuffling
    without a reader opening the dump."""
    if _sympy is None or not records:
        return 0, {}

    def _bind(expr, values: dict) -> dict:
        # Bind by NAME against the expression's own symbols: a symbol rebuilt
        # with different assumptions is a different symbol to sympy.
        return {
            sym: values[str(sym)] for sym in expr.free_symbols if str(sym) in values
        }

    parsed = []
    for rec in records:
        values = rec.get("bindings") or {}
        bundles = []
        for b in rec.get("bundles") or []:
            expr = _parse_expr(b.get("expr", ""))
            bundles.append((set(b.get("ops") or []), expr, b))
        parsed.append((set(rec.get("buffers") or []), values, bundles, rec))

    # Kernel -> record. Every kernel numbers its buffers from buf0, so names
    # alone collide; score a record by (name, byte size) agreement on the
    # kernel's output buffers (the SDSC output's full extent x element width),
    # then by name overlap.
    def _out_bytes(op: dict) -> int | None:
        for t in op.get("tensors") or []:
            if t["role"] != "OUTPUT":
                continue
            elems = _prod(list((t.get("tensor_extents") or {}).values()))
            wl = t.get("word_length")
            return elems * wl if elems and isinstance(wl, int) else None
        return None

    by_kernel: dict[str, dict[str, int | None]] = {}
    for op in all_ops:
        if op.get("out_buf"):
            by_kernel.setdefault(op.get("kernel", ""), {})[op["out_buf"]] = _out_bytes(
                op
            )

    def _record_score(pr, sizes: dict, names: set, t_us) -> tuple:
        rec_sizes = pr[3].get("buffer_sizes") or {}
        known = {n: b for n, b in sizes.items() if b is not None}
        exact = sum(
            1
            for n, b in known.items()
            if rec_sizes.get(n) not in (None, -1) and rec_sizes[n] == b
        )
        # The record's objective against the cost dump's prediction for the same
        # kernel (same model, same plan): identical for a kernel with no solver
        # choice, and the only key an unsized (-1) single-buffer kernel has.
        obj = pr[3].get("objective_ns")
        t_match = int(
            t_us is not None
            and obj is not None
            and abs(obj - t_us * 1000.0) <= 0.005 * max(1.0, abs(t_us * 1000.0))
        )
        # Prediction match first, then buffer-count proximity (a one-buffer
        # kernel must not take a 76-buffer record because buf0 happens to be
        # the same size), then size agreement, then name overlap.
        return (
            t_match,
            -abs(len(pr[0]) - len(names)),
            exact / max(1, len(known)),
            len(pr[0] & names),
        )

    # Global assignment: score every (kernel, record) pair, take the best pairs
    # first, one record per kernel and one kernel per record, so a small kernel
    # cannot claim a large kernel's record merely by compiling earlier.
    kernel_t: dict[str, float | None] = {}
    for op in all_ops:
        cost = op.get("cost") or {}
        if cost.get("t_us") is not None:
            kernel_t.setdefault(op.get("kernel", ""), cost["t_us"])
    pairs = []
    for kernel, sizes in by_kernel.items():
        names = set(sizes)
        for pr in parsed:
            score = _record_score(pr, sizes, names, kernel_t.get(kernel))
            t_match, _, exact_frac, overlap = score
            if overlap and (t_match or exact_frac > 0):
                pairs.append((score, kernel, pr))
    kernel_rec: dict[str, tuple] = {}
    used: set[int] = set()
    for score, kernel, pr in sorted(pairs, key=lambda x: x[0], reverse=True):
        if kernel in kernel_rec or id(pr) in used:
            continue
        kernel_rec[kernel] = pr
        used.add(id(pr))
    # Elimination for what is left: a single unused record with a kernel's
    # buffer count and overlapping names (an unsized copy kernel whose cost
    # block was ambiguous).
    for kernel, sizes in by_kernel.items():
        if kernel in kernel_rec:
            continue
        names = set(sizes)
        left = [
            pr
            for pr in parsed
            if id(pr) not in used and len(pr[0]) == len(names) and pr[0] & names
        ]
        if len(left) == 1:
            kernel_rec[kernel] = left[0]
            used.add(id(left[0]))
    annotated = 0
    for op in all_ops:
        out_buf = op.get("out_buf")
        rec_t = kernel_rec.get(op.get("kernel", ""))
        if not out_buf or rec_t is None:
            continue
        _buffers, values, bundles, rec = rec_t
        # Bundle membership by output buffer; by Inductor node name for a dump
        # written before bundles carried buffer names; a kernel priced as one
        # bundle contains every op.
        hit = next((b for b in bundles if out_buf in b[0]), None)
        if hit is None and op.get("ir_name"):
            hit = next((b for b in bundles if op["ir_name"] in b[0]), None)
        if hit is None and len(bundles) == 1:
            hit = bundles[0]
        if hit is None:
            continue
        ops_in_bundle, expr, raw = hit
        value = raw.get("value_ns")
        flipped = None
        resident = None
        is_lx_name = f"is_lx_{out_buf}"
        if expr is not None and is_lx_name in values:
            bindings = _bind(expr, values)
            is_lx_sym = next((k for k in bindings if str(k) == is_lx_name), None)
            if is_lx_sym is not None:
                resident = int(values[is_lx_name])
                if value is None:
                    value = _eval_expr(expr, bindings)
                alt = dict(bindings)
                alt[is_lx_sym] = 1 - resident
                flipped = _eval_expr(expr, alt)
        symbols = {}
        own_expr = None
        own_value = own_flipped = None
        n_terms = n_own = 0
        term_lines: list[dict] = []
        if expr is not None:
            # This op's own symbols only; a whole-kernel bundle has hundreds.
            own_syms = {
                sym
                for sym in expr.free_symbols
                if str(sym) in (f"is_lx_{out_buf}", f"division_{out_buf}")
                or str(sym).startswith(f"split_{out_buf}_")
            }
            for sym in sorted(own_syms, key=str):
                symbols[str(sym)] = values.get(str(sym))
            # The additive terms that mention this op: what its decisions can
            # move. The rest of the bundle term is constant with respect to it.
            all_terms = _sympy.Add.make_args(expr)
            own_terms = [t for t in all_terms if t.free_symbols & own_syms]
            n_terms, n_own = len(all_terms), len(own_terms)
            if own_terms:
                own_expr = _sympy.Add(*own_terms)
                bindings = _bind(expr, values)
                own_value = _eval_expr(own_expr, bindings)
                is_lx_sym = next((k for k in bindings if str(k) == is_lx_name), None)
                alt = None
                if is_lx_sym is not None:
                    alt = dict(bindings)
                    alt[is_lx_sym] = 1 - int(values[is_lx_name])
                    own_flipped = _eval_expr(own_expr, alt)
                # Each term reduced to THIS op's symbols: every other buffer's
                # symbol replaced by its solved value, constants folded. A term
                # such as the bundle turnaround Min(R, W) mentions every buffer;
                # reduced, it reads as a function of this op alone. Listed
                # with the term's value and its change under the flip, largest
                # change first.
                others = {k: v for k, v in bindings.items() if k not in own_syms}
                for term in own_terms:
                    # evalf folds the numeric leftovers (log(16), ...) so the
                    # line reads as numbers around this op's symbols.
                    reduced = term.xreplace(others)
                    try:
                        reduced = reduced.evalf(6)
                    except (TypeError, ValueError):
                        pass
                    v_now = _eval_expr(term, bindings)
                    v_alt = _eval_expr(term, alt) if alt is not None else None
                    term_lines.append(
                        {
                            "term": str(reduced),
                            "value_ns": v_now,
                            "flipped_ns": v_alt,
                        }
                    )
                term_lines.sort(
                    key=lambda t: (
                        -abs((t["flipped_ns"] or 0) - (t["value_ns"] or 0))
                        if t["flipped_ns"] is not None and t["value_ns"] is not None
                        else 0
                    )
                )
        relayouts = [
            rt for rt in rec.get("relayout_terms") or [] if rt.get("source") == out_buf
        ]
        # A relayout op (shuffle) materializes a copy: its price is the copy's
        # RelayoutCharge, keyed by the SOURCE buffer it reads, not a term of the
        # bundle expression. Show that charge as the op's objective, and the
        # other copies of the same source the solver did not fire.
        is_relayout = op["op_name"].lower() in ("shuffle",) or bool(op.get("move_op"))
        relayout_op = None
        if is_relayout:
            src = next((x for x in (op.get("in_bufs") or []) if x), None)
            src_terms = [
                rt for rt in rec.get("relayout_terms") or [] if rt.get("source") == src
            ]
            if src_terms:
                fired = [rt for rt in src_terms if rt.get("resident")]
                relayout_op = {
                    "source": src,
                    "fired": fired,
                    "n_copies": len(src_terms),
                    "charge_ns": sum(rt.get("value_ns") or 0 for rt in fired),
                }

                # What an unfired copy WOULD have cost under the source's
                # solved division: the charge with its residency set to 1.
                def _if_fired(rt, solved=values) -> float | None:
                    e = _parse_expr(rt.get("expr", ""))
                    if e is None:
                        return None
                    bd = _bind(e, solved)
                    for k in list(bd):
                        if str(k).startswith("is_lx_"):
                            bd[k] = 1
                    return _eval_expr(e, bd)

                term_lines = []
                for rt in sorted(src_terms, key=lambda rt: not rt.get("resident")):
                    if rt.get("resident"):
                        note, val = "[fired]", rt.get("value_ns")
                    else:
                        val = _if_fired(rt)
                        note = "[not fired; would cost this]"
                    term_lines.append(
                        {
                            "term": f"RelayoutCharge {rt['copy']}  {note}",
                            "value_ns": val,
                            "flipped_ns": None,
                        }
                    )
                n_own, n_terms = len(fired), len(src_terms)
                value, flipped, resident = relayout_op["charge_ns"], None, None
        op["objective"] = {
            "relayout_op": relayout_op,
            # The expression shown is the op's own terms, not the whole bundle.
            "expr_str": str(own_expr) if own_expr is not None else "",
            "n_terms": n_terms,
            "n_own_terms": n_own,
            "terms": term_lines,
            "own_value_ns": own_value,
            "own_flipped_ns": own_flipped,
            "value_ns": value,
            "flipped_ns": flipped,
            "resident": resident,
            "bundle_ops": sorted(ops_in_bundle),
            "symbols": symbols,
            "relayout_terms": relayouts,
            "objective_ns": rec.get("objective_ns"),
            "divisions": _division_lines(out_buf, rec),
        }
        # The plan's provenance, which the artifacts cannot show. `sencores` in
        # particular is a per-COMPILE cap -- spyre-inference caps the head-major
        # attention graph while the rest of the model stays at 32 -- so without
        # it a reader sees "8 cores" and assumes 24 were left unused.
        op["env"] = rec.get("env") or {}
        op["solve"] = rec.get("solve") or {}
        # Residency straight from the plan, so the separate probe JSON is only
        # needed for dumps written before the record carried this. Three
        # outcomes, and the reason distinguishes only the first: excluded before
        # the solver saw it, weighed and declined, or resident.
        div = (rec.get("divisions") or {}).get(out_buf) or {}
        is_lx = values.get(f"is_lx_{out_buf}")
        if "reason" in div or is_lx is not None:
            op.setdefault(
                "residency",
                {
                    "lx": bool(is_lx),
                    "reason": div.get("reason")
                    or (None if is_lx else "spilled by solver"),
                    "address": None,
                    "copy_of": "",
                },
            )
        annotated += 1
    # One entry per matched record, so two kernels sharing a record (they
    # cannot, but the assignment is best-effort) cannot count its copies twice.
    seen: set[int] = set()
    n_copies = n_fired = 0
    charged_ns = 0.0
    sources: set[str] = set()
    for pr in kernel_rec.values():
        rec = pr[3]
        if id(rec) in seen:
            continue
        seen.add(id(rec))
        for rt in rec.get("relayout_terms") or []:
            n_copies += 1
            if not rt.get("resident"):
                continue
            n_fired += 1
            charged_ns += rt.get("value_ns") or 0.0
            if rt.get("source"):
                sources.add(rt["source"])
    summary = {
        "copies": n_copies,
        "fired": n_fired,
        "charged_ns": charged_ns,
        "sources": sorted(sources),
    }
    return annotated, summary


def _objective_label(op: dict) -> str:
    """``12.3 us`` for the op's bundle term under the solved plan, then what the
    same term costs with this op's residency flipped."""
    obj = op.get("objective")
    if not obj:
        return ""
    lines = []
    ro = obj.get("relayout_op")
    if ro:
        lines.append(f"{ro['charge_ns'] / 1000:.2f} us relayout charge")
        lines.append(
            f"({len(ro['fired'])} of {ro['n_copies']} copies of {ro['source']} fired)"
        )
        return "\n".join(lines)
    n_ops = len(obj.get("bundle_ops") or [])
    if obj.get("value_ns") is not None:
        scope = f" ({n_ops}-op bundle)" if n_ops > 1 else ""
        lines.append(f"{obj['value_ns'] / 1000:.1f} us{scope}")
    if (
        obj.get("flipped_ns") is not None
        and obj.get("value_ns") is not None
        and obj.get("resident") is not None
    ):
        other = "HBM" if obj["resident"] else "LX"
        delta = (obj["flipped_ns"] - obj["value_ns"]) / 1000
        lines.append(f"if {other}: {delta:+.1f} us")
    return "\n".join(lines)


def find_companion(base_dir: Path, kind: str) -> Path | None:
    """The dump file that belongs to an artifact directory, by convention:
    ``cost_dump*.log`` (SPYRE_DUMP_COST_FILE), ``cost_expr*.jsonl``
    (SPYRE_DUMP_COST_EXPR_FILE) or ``probe*.json``/``lx_probe*.json``
    (the older residency probe) inside the directory, else in its parent (newest wins).
    ``sdsc capture`` writes them there; ``--cost``/``--cost-expr``/``--probe``
    override."""
    patterns = {
        "cost": ("cost_dump*.log",),
        "cost_expr": ("cost_expr*.jsonl",),
        "probe": ("probe*.json", "lx_probe*.json"),
    }[kind]
    for root in (base_dir, base_dir.parent):
        hits = sorted(
            (f for pat in patterns for f in root.glob(pat)),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if hits:
            return hits[0]
    return None


def _cost_label(op: dict) -> str:
    """``12.3 us of 1311.4 us bundle`` (this op's HBM-share attribution of the
    kernel's one prediction), then its read/write/LX bytes."""
    cost = op.get("cost")
    if not cost:
        return ""
    lines = []
    if cost.get("attributed_us") is not None and cost.get("t_us") is not None:
        lines.append(f"{cost['attributed_us']:.1f} us of {cost['t_us']:.1f} us")
    elif cost.get("t_us") is not None:
        lines.append(f"bundle T {cost['t_us']:.1f} us")
    dump_name = cost.get("dump_name") or ""
    if dump_name and dump_name.lower() != (op.get("op_name") or "").lower():
        lines.append(f"as {dump_name}")
    parts = f"R {_bytes_label(cost['read'])} W {_bytes_label(cost['write'])}"
    if cost.get("lx"):
        parts += f" LX {_bytes_label(cost['lx'])}"
    lines.append(parts)
    return "\n".join(lines)


def _describe_op(op: dict, op_idx: int, all_ops: list) -> str:
    """One-line description of what an op is, derived from the SDSC itself.

    Line 1 classifies the op from its name, its tensor roles and the LX/HBM
    placement of its inputs and outputs. A KERNEL_IDX input marks an indexed
    gather (index_select of KV pages); a shuffle is an LX relayout whose kind
    is read off the work-slice maps (one slot in -> many out = broadcast,
    many -> one = gather, many -> many = permutation). Line 2 names the
    next op that reads this op's LX output address, so the data flow can be
    followed down the table without matching addresses by hand.
    """
    name = op["op_name"]
    lname = name.lower()
    tensors = op.get("tensors") or []
    roles = {t["role"] for t in tensors}
    ins = [t for t in tensors if t["role"] == "INPUT"]
    outs = [t for t in tensors if t["role"] == "OUTPUT"]
    in_locs = ",".join(dict.fromkeys(_loc(t) for t in ins)) or "?"
    out_loc = _loc(outs[0]) if outs else "?"
    if "KERNEL_IDX" in roles:
        line = f"index gather {in_locs}->{out_loc}"
    elif (
        "shuffle" in lname
        or "relayout" in lname
        or "Relayout" in str(op.get("file", ""))
    ):
        n_in = _distinct_slots(ins[0].get("core_id_to_wk_slice") or {}) if ins else 1
        n_out = _distinct_slots(outs[0].get("core_id_to_wk_slice") or {}) if outs else 1
        if n_in == 1 and n_out > 1:
            kind = "broadcast"
        elif n_out < n_in:
            kind = "gather"
        else:
            kind = "permutation"
        line = f"{in_locs}->{out_loc} relayout: {kind} ({n_in}->{n_out} slices)"
    elif "restickify" in lname:
        line = f"restickify {in_locs}->{out_loc}"
    elif lname == "identity":
        d2d = (
            " (D2D)"
            if int(op.get("cores", 0) or 0) <= 1 and in_locs == out_loc == "HBM"
            else ""
        )
        line = f"copy {in_locs}->{out_loc}{d2d}"
    elif "matmul" in lname:
        line = f"matmul {in_locs}->{out_loc}"
    else:
        line = f"compute {in_locs}->{out_loc}"
    # Consumer: the next op whose INPUT sits at this op's LX output address.
    consumer = ""
    if (
        outs
        and (outs[0].get("component") == "lx")
        and outs[0].get("address") not in ("", None)
    ):
        addr = str(outs[0]["address"])
        for later in all_ops[op_idx + 1 :]:
            hit = any(
                t["role"] == "INPUT"
                and t.get("component") == "lx"
                and str(t.get("address")) == addr
                for t in later.get("tensors") or []
            )
            if hit:
                stem = Path(later["file"]).stem
                if "Relayout" in stem:
                    stem = "...Relayout"
                consumer = f"-> {later['op_name']} {stem}"
                break
    elif (
        outs
        and outs[0].get("component") == "hbm"
        and "KERNEL_IDX" not in roles
        and lname != "identity"
    ):
        consumer = "-> HBM output"
    return f"{line}\n{consumer}" if consumer else line


def _extract_datadsc_op(op_container: dict) -> dict | None:
    """Build an op record from a data-only `datadscs_` entry (no compute).

    Models an Stcdp-style data movement: the first labeledDs is the source
    (INPUT), the last is the destination (OUTPUT). Placement (lx/hbm) is read
    from each labeledDs's per-piece PlacementInfo. The data-movement op name
    (e.g. STCDPOpLx / STCDPOpHBM) is recorded so the summary can show it.
    """
    datadscs = op_container.get("datadscs_") or []
    if not datadscs:
        return None
    body = next(iter(datadscs[0].values()), None)
    if not isinstance(body, dict):
        return None
    labeled = body.get("labeledDs_") or []
    if not labeled:
        return None

    move_op = (body.get("op", {}) or {}).get("name", "data-move")

    def _placement(ld: dict) -> str:
        # Prefer the "real" placement: hbm if any piece lands in hbm, else lx.
        places = {
            pl.get("type")
            for pi in ld.get("PieceInfo", []) or []
            for pl in pi.get("PlacementInfo", []) or []
        }
        if "hbm" in places:
            return "hbm"
        if "lx" in places:
            return "lx"
        return next(iter(places), "")

    def _start_addr(ld: dict, want_type: str) -> str:
        # First piece's start address for the placement matching `want_type`.
        # startAddr.data_ maps a coord key to a per-core address list; the
        # first entry is the (uniform) base. Empty list -> no address.
        for pi in ld.get("PieceInfo", []) or []:
            for pl in pi.get("PlacementInfo", []) or []:
                if pl.get("type") != want_type:
                    continue
                data = (pl.get("startAddr", {}) or {}).get("data_", {}) or {}
                vals = next(iter(data.values()), None)
                if isinstance(vals, list) and vals:
                    return str(vals[0])
                if isinstance(vals, str) and vals:
                    return vals
        return ""

    # Union of per-dim sizes across labeledDs entries, used so the shared
    # layout decorator can render bare dim names (e.g. "out") with extents
    # ("4096*") the same way it does for compute ops.
    dim_extents: dict[str, int] = {}
    for ld in labeled:
        for k, v in (ld.get("dimToLayoutSize_", {}) or {}).items():
            if isinstance(v, int):
                dim_extents[k] = v

    tensors = []
    for i, ld in enumerate(labeled):
        role = "OUTPUT" if i == len(labeled) - 1 and len(labeled) > 1 else "INPUT"
        # layoutDimOrder_ is the bare dim-name list; fall back to the keys of
        # dimToLayoutSize_ if absent.
        layout_order = ld.get("layoutDimOrder_") or list(
            (ld.get("dimToLayoutSize_", {}) or {}).keys()
        )
        component = _placement(ld)
        tensors.append(
            {
                "name": ld.get("ldsName_", ""),
                "role": role,
                "layout": ", ".join(layout_order),
                "sticks": ", ".join(ld.get("stickDimOrder_", []) or []),
                "component": component,
                "address": _start_addr(ld, component),
                "mem_org": "",
                "max_dim_sizes": [],
                "layout_dim_order": layout_order,
                "core_id_to_wk_slice": {},
                "scale": [],
                "tensor_extents": {
                    k: v
                    for k, v in (ld.get("dimToLayoutSize_", {}) or {}).items()
                    if isinstance(v, int)
                },
                "data_format": str(ld.get("dataFormat_", "") or ""),
                "word_length": ld.get("wordLength")
                if isinstance(ld.get("wordLength"), int)
                else None,
            }
        )

    components = [t["component"] for t in tensors if t["component"]]
    return {
        "components": components,
        "components_str": ", ".join(components),
        "component_details": [],
        "tensors": tensors,
        "tensors_str": ", ".join(t["role"] for t in tensors),
        "address": "",
        "dims": "",
        "wk_slices": op_container.get("numWkSlicesPerDim_", {}) or {},
        "dim_extents": dim_extents,
        "move_op": move_op,
    }


def _topological_sort_ops(all_ops: list) -> list:
    """Sort operations by data dependency (topological order).

    Tensor identity is keyed on (component, address), since tensor names like
    `allocate_tensor0_hbm` are per-op placeholders reused across ops. Ties are
    broken by input order - callers pass ops in file-mtime order so mtime wins
    among independent ops.
    """
    import heapq

    # (component, address) -> op index producing it
    output_to_op_idx = {}
    for idx, op in enumerate(all_ops):
        for tensor in op["tensors"]:
            if tensor["role"] == "OUTPUT":
                key = (tensor["component"], tensor["address"])
                if key[1]:
                    output_to_op_idx[key] = idx

    n = len(all_ops)
    adj = defaultdict(list)
    in_degree = [0] * n

    for idx, op in enumerate(all_ops):
        for tensor in op["tensors"]:
            if tensor["role"] != "INPUT":
                continue
            key = (tensor["component"], tensor["address"])
            producer = output_to_op_idx.get(key)
            if producer is not None and producer != idx:
                adj[producer].append(idx)
                in_degree[idx] += 1

    # Min-heap on index → ties follow input (mtime) order
    heap = [i for i in range(n) if in_degree[i] == 0]
    heapq.heapify(heap)
    sorted_indices = []
    while heap:
        node = heapq.heappop(heap)
        sorted_indices.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, neighbor)

    if len(sorted_indices) < n:
        remaining = [i for i in range(n) if i not in set(sorted_indices)]
        sorted_indices.extend(remaining)

    return [all_ops[i] for i in sorted_indices]


def build_report(
    base_dir_str: str,
    index_range: tuple[int, int] | None = None,
    meta: dict | None = None,
    log_path: Path | None = None,
    cost_path: Path | None = None,
    lx_capacity: int = _LX_CAPACITY_DEFAULT,
    cost_expr_path: Path | None = None,
    probe_path: Path | None = None,
) -> dict:
    """Parse every sdsc.json under a directory into enriched op records.

    Returns {"base_dir", "n_files", "index_range", "ops", "stats",
    "cost_matched"}; ``ops`` is in execution order with the fields
    :func:`enrich_ops` adds. The renderers (:func:`render_report`) and the
    Textual browser (``sdsc_tui.py``) both consume this.

    cost_path: optional captured ``SPYRE_DUMP_COST=1`` output; per-op
    read/write/lx bytes and the bundle prediction are joined by output buffer
    name (``op["cost"]``).
    lx_capacity: per-core LX budget the occupancy percentage is measured
    against (default: the planner's 1,625,344 B).

    index_range: optional (lo, hi) inclusive filter on the sdsc_N index within
    each kernel subdirectory. Files whose N falls outside [lo, hi] are skipped.
    log_path: optional SPYRE_INDUCTOR_LOG DEBUG log. When provided, each op's
    OUTPUT tensor row is annotated with its real buffer name (bufN), parsed
    from the log and mapped through the .py's LoopSpec/OpSpec tree structure.
    meta: optional dict loaded from a --meta JSON file. Supported keys:
      - "column"       : str, header for the injected annotation column
      - "op_sequence"  : list of {"op_name_substr": str, "label": str}. Labels
                         are assigned sequentially in display order - from the
                         first op, through any prologue, and into the FIRST
                         inner-loop iteration only (boundary from the .py's
                         LoopSpec tree). Later iterations and any op_sequence
                         entries past the loop end get no label.
      - "name_fallback": list of {"op_name_substr": str, "label": str}, matched
                         by op name only when the sequence entry doesn't match.
      - "loop_length"  : deprecated / ignored (the loop boundary now comes from
                         the .py LoopSpec structure, not a fixed op count).
    When meta is None (default) no extra column is added.
    """
    base_dir = Path(base_dir_str)

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir_str}", file=sys.stderr)
        sys.exit(1)

    # Find only sdsc_*.json files (not .out.json variants), sorted by mtime
    sdsc_files = [
        f for f in base_dir.glob("**/sdsc_*.json") if not f.name.endswith(".out.json")
    ]
    sdsc_files.sort(key=lambda f: f.stat().st_mtime)

    # Fallback: if no sdsc_*.json found, try all *.json files in the directory
    if not sdsc_files:
        sdsc_files = [
            f for f in base_dir.glob("**/*.json") if not f.name.endswith(".out.json")
        ]
        sdsc_files.sort(key=lambda f: f.stat().st_mtime)

    if not sdsc_files:
        print(f"No sdsc.json files found in {base_dir_str}")
        return

    # Per-kernel-dir count of compute sdsc_N.json - the number the kernel
    # expands to. Captured from the FULL file list before the range filter, so
    # buffer-name resolution still works under --range (the guard compares this
    # against the .py tree-expansion length, not the filtered subset).
    full_dir_sdsc_count: dict[str, int] = defaultdict(int)
    for f in sdsc_files:
        if _SDSC_FILE_IDX_RE.search(f.name):
            full_dir_sdsc_count[f.parent.name] += 1

    # Apply index range filter when requested.
    # _SDSC_FILE_IDX_RE (module scope) matches sdsc_N.json filenames (no
    # trailing underscore) as well as the sdsc_N_... form.
    if index_range is not None:
        lo, hi = index_range

        def _in_range(f: Path) -> bool:
            m = _SDSC_FILE_IDX_RE.search(f.name)
            if m is None:
                return True  # non-indexed files pass through
            return lo <= int(m.group(1)) <= hi

        sdsc_files = [f for f in sdsc_files if _in_range(f)]

    n_files = len(sdsc_files)

    # Collect all operations data
    all_ops = []
    stats = {
        "total_files": len(sdsc_files),
        "files_with_ops": 0,
        "files_skipped": 0,
        "operation_types": defaultdict(int),
        "component_types": defaultdict(int),
    }

    for file_path in sdsc_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            stats["files_skipped"] += 1
            continue

        file_rel_path = file_path.relative_to(base_dir)
        file_name = str(file_rel_path)

        ops = extract_ops_from_sdsc(data, file_name)

        if ops:
            stats["files_with_ops"] += 1
            mtime = file_path.stat().st_mtime
            for op in ops:
                op["_mtime"] = mtime
                stats["operation_types"][op["op_name"]] += 1
                for comp in op["components"]:
                    stats["component_types"][comp] += 1
            all_ops.extend(ops)
        else:
            stats["files_skipped"] += 1

    # Sort priority: (1) file mtime, (2) sdsc_N_ numeric index, (3) data-
    # dependency topological order for groups that tie on both.
    if all_ops:
        # Bucket ops by mtime, preserving mtime order.
        # Kernels in compile order (the earliest file mtime of each kernel
        # directory), and within a kernel by sdsc_N index: the index is the
        # emission order and survives a copy of the artifacts, which mtimes do
        # not. Files without an index (relayout datadscs) follow the indexed
        # ones of their kernel in topological order.
        dir_start: dict[str, float] = {}
        for op in all_ops:
            d = Path(op["file"]).parent.name
            dir_start[d] = min(dir_start.get(d, op["_mtime"]), op["_mtime"])
        by_dir: dict[str, list] = {}
        for op in all_ops:
            by_dir.setdefault(Path(op["file"]).parent.name, []).append(op)
        sorted_all = []
        for d in sorted(by_dir, key=lambda d: (dir_start[d], d)):
            group = by_dir[d]
            indexed = [op for op in group if _sdsc_file_index(op) is not None]
            rest = [op for op in group if _sdsc_file_index(op) is None]
            indexed.sort(key=_sdsc_file_index)
            sorted_all.extend(indexed)
            if rest:
                sorted_all.extend(_topological_sort_ops(rest))
        all_ops = sorted_all

    # Resolve per-op annotations from the Inductor <hash>.py LoopSpec/OpSpec
    # tree (always, when a .py is present) and optionally the DEBUG log:
    #   - op["loop_coords"]: enclosing loop iteration indices (outermost first),
    #     rendered under the op name so loop boundaries are visible.
    #   - op["out_buf"]: the op's output buffer name (bufN), only when --log is
    #     given (the log's kernel_store lines carry the real per-op bufN).
    #   - op["in_bufs"]: the op's ordered input names (bufN/argN_1), only when
    #     --log is given (the log's kernel_load lines carry the per-op inputs).
    # Parsed once; guarded against the per-dir sdsc file count (pre-range-filter).
    if all_ops:
        py_kernels = _parse_py_kernels(
            base_dir
        )  # name -> (leaf_seq, n_leaves, coords_seq)
        log_bufs = _parse_debug_log(log_path) if log_path is not None else {}
        py_bufs = _parse_py_opspecs(base_dir)

        dir_sdsc_count = full_dir_sdsc_count

        def _dir_matches(dirname: str, kname: str) -> bool:
            # Artifact dirs are ``<hash>_<kernel name>_<rand8>`` (the hash
            # prefix keeps long fused names under NAME_MAX); older ones are
            # ``<kernel name>_<rand8>``. Match either.
            return dirname.startswith(kname) or dirname.split("_", 1)[-1].startswith(
                kname
            )

        def _pick(table: dict, kernel_dir: Path, want_len) -> tuple | None:
            """The candidate for ``kernel_dir``: same kernel name, preferring
            the .py under the same cache root (``<root>/inductor-spyre/<dir>``),
            then one whose leaf count ``want_len`` accepts."""
            root = kernel_dir.resolve().parent.parent
            cands = [
                c
                for kname, lst in table.items()
                if _dir_matches(kernel_dir.name, kname)
                for c in lst
            ]
            same_root = [c for c in cands if c[0].resolve() == root]
            for pool in (same_root, cands):
                for _, val in pool:
                    if want_len(val):
                        return val
            return None

        def _py_kernel_for_dir(kernel_dir: Path):
            n = dir_sdsc_count.get(kernel_dir.name, -1)
            return _pick(py_kernels, kernel_dir, lambda v: len(v[0]) == n)

        def _py_bufs_for_dir(kernel_dir: Path, n_leaves: int):
            return _pick(py_bufs, kernel_dir, lambda v: len(v[0]) == n_leaves)

        # Per-dir: flat loop-coords list, and (when --log) flat buffer lists.
        # flat_bufs[k] = store buf of the leaf that sdsc_k expands from;
        # flat_inputs[k] = that leaf's ordered input names (both keyed by leaf,
        # so a repeated loop-body leaf reuses the same names across iterations).
        dir_coords: dict[str, list[tuple]] = {}
        dir_flat_bufs: dict[str, list[str]] = {}
        dir_flat_inputs: dict[str, list[list[str]]] = {}
        dir_flat_ir: dict[str, list[str]] = {}
        kernel_dirs = {
            Path(op["file"]).parent.name: (base_dir / op["file"]).parent
            for op in all_ops
        }
        dir_align: dict[str, list[int | None]] = {}
        for dirname, kernel_dir in kernel_dirs.items():
            n_dir_files = dir_sdsc_count.get(dirname, -1)
            py_val = _py_kernel_for_dir(kernel_dir)
            if py_val is None:
                # No compile with exactly n_dir_files leaves (relayouts turned
                # into shuffles, ops dropped): take the compile whose leaf
                # count is closest from above, and align by op kind below.
                root = kernel_dir.resolve().parent.parent
                cands = [
                    c
                    for kname, lst in py_kernels.items()
                    if _dir_matches(kernel_dir.name, kname)
                    for c in lst
                ]
                same_root = [c for c in cands if c[0].resolve() == root] or cands
                py_val = min(
                    (v for _, v in same_root),
                    key=lambda v: (
                        len(v[0]) < n_dir_files,
                        abs(len(v[0]) - n_dir_files),
                    ),
                    default=None,
                )
                if py_val is None:
                    continue
            leaf_seq, n_leaves, coords_seq = py_val
            if len(leaf_seq) != n_dir_files:
                py_kinds = _py_bufs_for_dir(kernel_dir, n_leaves)
                if py_kinds is None:
                    continue
                dir_ops = sorted(
                    (op for op in all_ops if Path(op["file"]).parent.name == dirname),
                    key=lambda op: _sdsc_file_index(op) or 0,
                )
                sdsc_kinds = [op["op_name"] for op in dir_ops]
                leaf_kinds = [py_kinds[3][i] for i in leaf_seq]
                dir_align[dirname] = _align_leaves(sdsc_kinds, leaf_kinds)
            dir_coords[dirname] = coords_seq
            block = log_bufs.get(dirname)
            # Guard: the log must give one store buf per distinct leaf op.
            if block is not None:
                stores, inputs = block
                if len(stores) == n_leaves:
                    dir_flat_bufs[dirname] = [stores[i] for i in leaf_seq]
                if len(inputs) == n_leaves:
                    dir_flat_inputs[dirname] = [inputs[i] for i in leaf_seq]
            # No log (or a log that does not cover this kernel): the .py's
            # OpSpec tree carries the same names.
            py_block = _py_bufs_for_dir(kernel_dir, n_leaves)
            if py_block is not None:
                stores, inputs, ir_names, _kinds = py_block
                if dirname not in dir_flat_bufs and len(stores) == n_leaves:
                    dir_flat_bufs[dirname] = [stores[i] for i in leaf_seq]
                if dirname not in dir_flat_inputs and len(inputs) == n_leaves:
                    dir_flat_inputs[dirname] = [inputs[i] for i in leaf_seq]
                if len(ir_names) == n_leaves:
                    dir_flat_ir[dirname] = [ir_names[i] for i in leaf_seq]

        for op in all_ops:
            dirname = Path(op["file"]).parent.name
            m = _SDSC_FILE_IDX_RE.search(Path(op["file"]).name)
            idx = int(m.group(1)) if m else None
            if idx is None:
                continue
            # Position in the expanded leaf list: the file index itself when
            # the counts match, else the kind-aligned position (None = no
            # OpSpec corresponds, e.g. an inserted relayout).
            align = dir_align.get(dirname)
            if align is not None:
                pos = align[idx] if 0 <= idx < len(align) else None
            else:
                pos = idx
            if pos is None:
                continue
            coords_seq = dir_coords.get(dirname)
            if coords_seq is not None and 0 <= pos < len(coords_seq):
                op["loop_coords"] = coords_seq[pos]
            flat_bufs = dir_flat_bufs.get(dirname)
            if flat_bufs is not None and 0 <= pos < len(flat_bufs):
                op["out_buf"] = flat_bufs[pos]
            flat_inputs = dir_flat_inputs.get(dirname)
            if flat_inputs is not None and 0 <= pos < len(flat_inputs):
                op["in_bufs"] = flat_inputs[pos]
            flat_ir = dir_flat_ir.get(dirname)
            if flat_ir is not None and 0 <= pos < len(flat_ir):
                op["ir_name"] = flat_ir[pos]

    # meta annotation position (op["meta_pos"]): a sequential index into the
    # meta op_sequence, assigned in display order from the first op (0, 1, 2,
    # ...) through the prologue and the FIRST inner-loop iteration only. Once
    # that first iteration ends (loop_coords changes after the loop begins),
    # labeling stops - later iterations and leftover op_sequence entries get no
    # label. This replaces the old op_idx % loop_length scheme, which drifted
    # when the meta's loop_length disagreed with the real .py loop body.
    if meta is not None and all_ops:
        first_loop_coords = None  # the loop_coords of the first looped op seen
        for pos, op in enumerate(all_ops):
            coords = op.get("loop_coords") or ()
            if first_loop_coords is None:
                if coords:
                    first_loop_coords = coords  # loop just started
            elif coords != first_loop_coords:
                # First inner-loop iteration finished - stop labeling.
                break
            op["meta_pos"] = pos

    # Print operation summaries first
    enrich_ops(all_ops, lx_capacity)
    cost_matched = 0
    if cost_path is not None and all_ops:
        cost_matched = _join_cost(all_ops, _parse_cost_dump(cost_path))
    objective_matched = 0
    relayouts: dict = {}
    if cost_expr_path is not None and all_ops:
        objective_matched, relayouts = _join_cost_expr(
            all_ops, _parse_cost_expr_dump(cost_expr_path)
        )
    residency_matched = 0
    if probe_path is not None and all_ops:
        residency_matched = _join_probe(all_ops, _parse_probe_json(probe_path))
    residency_matched = residency_matched or sum(
        1 for op in all_ops if op.get("residency")
    )
    # Occupancy is a fraction of a budget, so it is only as right as the budget.
    # The dump carries the one the planner used; the CLI default is a guess that
    # happens to match the shipped value.
    planner_cap = next(
        (
            c
            for op in all_ops
            if isinstance(c := (op.get("env") or {}).get("lx_capacity"), int) and c > 0
        ),
        None,
    )
    if planner_cap and planner_cap != lx_capacity:
        lx_capacity = planner_cap
        enrich_ops(all_ops, lx_capacity)
    return {
        "objective_matched": objective_matched,
        "residency_matched": residency_matched,
        "relayouts": relayouts,
        "base_dir": str(base_dir),
        "n_files": n_files,
        "index_range": index_range,
        "ops": all_ops,
        "stats": stats,
        "meta": meta,
        "cost_matched": cost_matched,
        "lx_capacity": lx_capacity,
    }


# Column keys in display order. The renderers pick from these; ``basic`` is
# the compact set (the original column set plus What).
_ALL_COLUMNS = [
    "what",
    "annotation",
    "op",
    "cores",
    "tensor",
    "role",
    "layout",
    "tile",
    "tile_size",
    "footprint",
    "address",
    "core_map",
    "format",
    "lx_live",
    "cost",
    "objective",
    "kernel",
    "json",
]
_BASIC_COLUMNS = [
    "what",
    "annotation",
    "op",
    "cores",
    "tensor",
    "role",
    "layout",
    "tile",
    "tile_size",
    "address",
    "core_map",
    "format",
    "json",
]
_HEADERS = {
    "what": "What\n(-> consumer)",
    "annotation": "annotation",
    "op": "Op\n(outer, inner)\nloop iter",
    "cores": "cores",
    "tensor": "alloc_tensor\n{i}_{loc}\n(except relayout)",
    "role": "Role",
    "layout": "Layout*\nextent/wkSlices",
    "tile": "Tile Shape",
    "tile_size": "Tile Size",
    "footprint": "LX footprint\n(stick padded)",
    "address": "Address\n(LX range)",
    "core_map": "coreIdToWkSlice",
    "format": "Format",
    "lx_live": "LX live\n/ capacity",
    "cost": "Cost\n(SPYRE_DUMP_COST)",
    "objective": "Objective term\n(solver, this plan)",
    "kernel": "kernel",
    "json": "json\nfiles",
}


def _layout_label(tensor: dict, op: dict) -> str:
    """The Layout column: host extents in primaryDsInfo order, ``*`` on stick
    dims, ``/N`` for a dim split N ways (op-level, or inferred for a relayout
    source whose per-core extent the op-level slices do not record), ``1``
    for a reduced or broadcast dim."""
    op_name = op["op_name"]
    wk_slices = op.get("wk_slices", {}) or {}
    dim_extents = op.get("dim_extents", {}) or {}
    layout = tensor["layout"]
    sticks = tensor["sticks"]
    if not layout:
        return layout
    tensor_extents = tensor.get("tensor_extents") or {}
    stick_dim_set = set(sticks.split(", ")) if sticks else set()
    stick_size = tensor.get("stick_size")
    data_move = op_name.lower() in _DATA_MOVE_OPS or bool(op.get("move_op"))
    reduced_dims: set = set()
    inferred_slices: dict[str, int] = {}
    for dim, alpha in tensor_extents.items():
        full_extent = dim_extents.get(dim)
        n_slices = wk_slices.get(dim, 1) or 1
        if not (
            isinstance(full_extent, int)
            and isinstance(alpha, int)
            and isinstance(n_slices, int)
            and n_slices > 0
            and alpha < full_extent // n_slices
        ):
            continue
        stick_reduced = (
            not data_move
            and dim in stick_dim_set
            and isinstance(stick_size, int)
            and alpha == stick_size
        )
        if data_move and alpha == 1 and int(op.get("cores", 0) or 0) > 1:
            inferred_slices[dim] = full_extent
        elif alpha == 1 or stick_reduced:
            reduced_dims.add(dim)
        elif full_extent % alpha == 0:
            inferred_slices[dim] = full_extent // alpha
        else:
            reduced_dims.add(dim)
    decorated = []
    for dim in layout.split(", "):
        if dim in reduced_dims:
            base = "1"
        else:
            extent = dim_extents.get(dim)
            base = str(extent) if extent is not None else dim
        label = f"{base}*" if dim in stick_dim_set else base
        n = inferred_slices.get(dim, wk_slices.get(dim))
        if isinstance(n, int) and n > 1 and dim not in reduced_dims:
            label = f"{label}/{n}"
        decorated.append(label)
    return ", ".join(decorated)


def _tensor_label(tensor: dict, op: dict, in_slot: int) -> str:
    """``{lds}_{loc}`` plus the Inductor buffer name when known."""
    lds_idx = tensor.get("lds_idx")
    if isinstance(lds_idx, int) and lds_idx >= 0:
        short = f"{lds_idx}_{tensor['component']}"
    else:
        short = tensor["name"]
        if len(short) > 12 and short.endswith("_out"):
            short = "..._out"
        if tensor.get("component"):
            short = f"{short}_{tensor['component']}"
    if tensor["role"] == "OUTPUT" and op.get("out_buf"):
        short = f"{short} ({op['out_buf']})"
    elif tensor["role"] == "INPUT":
        in_bufs = op.get("in_bufs")
        if in_bufs is not None and in_slot < len(in_bufs) and in_bufs[in_slot]:
            short = f"{short} ({in_bufs[in_slot]})"
    return short


def _annotation(op: dict, meta: dict | None) -> str:
    if meta is None:
        return ""
    seq = meta.get("op_sequence", [])
    fallback = meta.get("name_fallback", [])
    pos = op.get("meta_pos")
    op_name = op["op_name"]
    if pos is not None and pos < len(seq):
        entry = seq[pos]
        if entry.get("op_name_substr", "").lower() in op_name.lower():
            return entry.get("label", "")
    if pos is not None:
        for entry in fallback:
            if entry.get("op_name_substr", "").lower() in op_name.lower():
                return entry.get("label", "")
    return ""


def build_records(report: dict) -> list[dict]:
    """One record per tensor row, keyed by ``_ALL_COLUMNS`` plus ``op_idx``
    and ``first`` (True on an op's first row). Renderers and the TUI select
    the columns they show."""
    all_ops = report["ops"]
    meta = report.get("meta")
    records: list[dict] = []
    for op_idx, op in enumerate(all_ops):
        if not op["tensors"]:
            continue
        op_name = op["op_name"]
        what = _describe_op(op, op_idx, all_ops)
        multi_core = int(op.get("cores", 0) or 0) > 1
        lx_live = ""
        if op.get("lx_live_bytes") is not None:
            lx_live = _bytes_label(op["lx_live_bytes"])
            if op.get("lx_live_pct") is not None:
                lx_live += f"\n{op['lx_live_pct']:.0f}%"
        in_slot = 0
        for alloc_idx, tensor in enumerate(op["tensors"]):
            first = alloc_idx == 0
            op_label = op_name if first else ""
            loop_coords = op.get("loop_coords") or ()
            if loop_coords and tensor["role"] == "OUTPUT":
                counter = f"({', '.join(str(c) for c in loop_coords)})"
                op_label = f"{op_label}\n{counter}" if first else counter
            what_cell = what if first else ""
            if tensor["role"] == "INPUT" and tensor.get("producer"):
                what_cell = (
                    f"{what_cell}\n{tensor['producer']}"
                    if what_cell
                    else tensor["producer"]
                )
            tensor_label = _tensor_label(tensor, op, in_slot)
            if tensor["role"] == "INPUT":
                in_slot += 1
            tile = tensor.get("tile")
            records.append(
                {
                    "op_idx": op_idx,
                    "first": first,
                    "what": what_cell,
                    "annotation": _annotation(op, meta) if first else "",
                    "op": op_label,
                    "cores": _cores_cell(op) if first else "",
                    "tensor": tensor_label,
                    "role": tensor["role"],
                    "layout": _layout_label(tensor, op),
                    "tile": " x ".join(str(v) for v in tile) if tile else "",
                    "tile_size": _bytes_label(tensor.get("tile_bytes")),
                    "footprint": _bytes_label(tensor.get("footprint_bytes"))
                    if tensor.get("component") == "lx"
                    else "",
                    "address": _address_cell(tensor, op),
                    "core_map": _compress_core_id_to_wk_slice(
                        tensor.get("core_id_to_wk_slice", {}) or {}
                    )
                    if multi_core
                    else "",
                    "format": _format_label(tensor),
                    "lx_live": lx_live if first else "",
                    "cost": _cost_label(op) if first else "",
                    "objective": _objective_label(op) if first else "",
                    "kernel": op.get("kernel", "") if first else "",
                    "json": _sdsc_stem(op) if first else "",
                }
            )
    return records


def _cores_cell(op: dict) -> str:
    """The Cores column, against the cap this compile actually had.

    ``sencores`` is per COMPILE, not per run: spyre-inference caps the
    head-major attention graph to 8 while the rest of the model stays at 32. A
    bare "8" then reads as 24 cores left unused, when it is full occupancy.
    """
    cores = op.get("cores")
    if cores in (None, ""):
        return ""
    cap = (op.get("env") or {}).get("sencores")
    if not isinstance(cap, int) or not isinstance(cores, int) or cap <= 0:
        return str(cores)
    if cores >= cap:
        return f"{cores} of {cap}\n(full)"
    return f"{cores} of {cap}"


def _address_cell(tensor: dict, op: dict) -> str:
    """The Address column: the LX range or ``hbm#N``, and on the op's OUTPUT row
    the probe's reason when that output is NOT in LX. The reason belongs on the
    row that shows the placement, which is where a reader asks for it."""
    label = tensor.get("addr_label", "")
    res = op.get("residency")
    if tensor.get("role") != "OUTPUT" or not res or res.get("lx"):
        return label
    reason = res.get("reason")
    if not reason:
        return label
    return f"{label}\n{_reason_short(reason)}" if label else _reason_short(reason)


def _residency_summary_line(report: dict) -> str:
    """One line on why the plan's HBM buffers are in HBM, counted by reason.
    A run dominated by "spilled by solver" was a cost decision; one dominated
    by a gate reason ("op not allowed", "partial/offset read") never reached
    the solver at all, which is a different thing to go and fix."""
    if not report.get("residency_matched"):
        return ""
    counts: dict[str, int] = {}
    lx = 0
    for op in report["ops"]:
        res = op.get("residency")
        if not res:
            continue
        if res.get("lx"):
            lx += 1
            continue
        counts[_reason_short(res.get("reason") or "unrecorded")] = (
            counts.get(_reason_short(res.get("reason") or "unrecorded"), 0) + 1
        )
    if not counts and not lx:
        return ""
    worst = ", ".join(
        f"{reason} x{n}"
        for reason, n in sorted(counts.items(), key=lambda kv: -kv[1])[:4]
    )
    return (
        f"Residency: {lx} of {report['residency_matched']} joined op outputs in LX"
        + (f"; HBM by reason: {worst}" if worst else "")
    )


def _select_columns(report: dict, basic: bool, show_what: bool) -> list[str]:
    cols = list(_BASIC_COLUMNS if basic else _ALL_COLUMNS)
    if not show_what:
        cols.remove("what")
    if report.get("meta") is None:
        cols.remove("annotation")
    all_ops = report["ops"]
    if not any(int(op.get("cores", 0) or 0) > 1 for op in all_ops):
        cols.remove("core_map")
    if "cost" in cols and not report.get("cost_matched"):
        cols.remove("cost")
    if "objective" in cols and not report.get("objective_matched"):
        cols.remove("objective")
    if "kernel" in cols and len({op.get("kernel") for op in all_ops}) <= 1:
        cols.remove("kernel")
    return cols


def _headers_for(cols: list[str], report: dict, table_format: str) -> list[str]:
    headers = []
    for c in cols:
        h = _HEADERS[c]
        if c == "annotation" and report.get("meta") is not None:
            h = report["meta"].get("column", "annotation")
        headers.append(h if table_format == "text" else h.replace("\n", " "))
    return headers


def _terminal_width(requested: int | None = None) -> int | None:
    """The width a text table must fit in, or ``None`` for no limit.

    An explicit ``requested`` wins (0 means no limit), then ``$COLUMNS``, then
    the terminal stdout is attached to. Output going to a file or a pipe has no
    width to honour, so it keeps its natural, greppable one-row-per-line form."""
    if requested is not None:
        return requested or None
    env = os.environ.get("COLUMNS", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    if sys.stdout.isatty():
        try:
            return os.get_terminal_size(sys.stdout.fileno()).columns
        except OSError:
            return None
    return None


# Narrowest a wrapped column is squeezed to: below this a cell stops reading as
# words and starts reading as a column of fragments.
_MIN_WRAP_WIDTH = 8


def _fit_column_widths(
    rows: list, headers: list[str], width: int
) -> list[int | None] | None:
    """Per-column wrap widths that make a ``mixed_outline`` table fit ``width``
    terminal columns, or ``None`` when it already fits.

    Water-filling: every column keeps its natural width up to a common cap, and
    the cap is the largest one that fits, so only the widest columns (the long
    free-text ones) wrap and the narrow key columns never do. Columns never go
    below ``_MIN_WRAP_WIDTH`` (or their natural width, if narrower); a table
    that cannot fit even then overflows rather than turning unreadable."""

    def cell_width(value) -> int:
        return max((len(line) for line in str(value).split("\n")), default=0)

    natural = [cell_width(h) for h in headers]
    for row in rows:
        if row is SEPARATING_LINE:
            continue
        for i, value in enumerate(row):
            natural[i] = max(natural[i], cell_width(value))
    # "│ " before each column, " " after it, and one closing "│".
    budget = width - 3 * len(natural) - 1
    if sum(natural) <= budget:
        return None
    floors = [min(w, _MIN_WRAP_WIDTH) for w in natural]
    lo, hi = _MIN_WRAP_WIDTH, max(natural)
    while lo < hi:
        cap = (lo + hi + 1) // 2
        if sum(max(f, min(w, cap)) for f, w in zip(floors, natural)) <= budget:
            lo = cap
        else:
            hi = cap - 1
    caps = [max(f, min(w, lo)) for f, w in zip(floors, natural)]
    # Hand the rounding leftover to the capped columns, one column each.
    spare = budget - sum(caps)
    for i, w in enumerate(natural):
        if spare <= 0:
            break
        if caps[i] < w:
            caps[i] += 1
            spare -= 1
    return [c if c < w else None for c, w in zip(caps, natural)]


def _tabulate_records(
    records: list[dict],
    cols: list[str],
    headers: list[str],
    table_format: str,
    width: int | None = None,
) -> str:
    """Render records as a table. For ``text``, ``width`` is the terminal width
    to fit: the widest columns wrap inside their cells until the table fits."""
    if table_format == "text":
        rows: list = []
        for i, r in enumerate(records):
            if r["first"] and i > 0:
                rows.append(SEPARATING_LINE)
            rows.append([r[c] for c in cols])
        maxcolwidths = _fit_column_widths(rows, headers, width) if width else None
        return tabulate(
            rows,
            headers=headers,
            tablefmt="mixed_outline",
            maxcolwidths=maxcolwidths,
        )
    rows = [[str(r[c]).replace("\n", "<br>") for c in cols] for r in records]
    return tabulate(
        rows,
        headers=headers,
        tablefmt="github" if table_format == "github" else "html",
        disable_numparse=True,
    )


def _render_collapsible(report: dict, records: list[dict], cols: list[str]) -> None:
    """Markdown with one ``<details>`` block per op, grouped under a heading
    per kernel: the summary line carries the op, cores, what it is, its json
    file and LX occupancy; the body is the op's tensor table with every
    column. GitHub renders the blocks collapsed."""
    all_ops = report["ops"]
    per_op_cols = [c for c in cols if c not in ("op", "cores", "json", "kernel")]
    headers = _headers_for(per_op_cols, report, "github")
    by_kernel: dict[str, list[int]] = {}
    for op_idx, op in enumerate(all_ops):
        by_kernel.setdefault(op.get("kernel", ""), []).append(op_idx)
    for kernel, op_idxs in by_kernel.items():
        dirname = Path(all_ops[op_idxs[0]]["file"]).parent.name
        print(f"### Kernel `{kernel}` ({len(op_idxs)} ops)\n")
        print(f"`{dirname}`\n")
        for op_idx in op_idxs:
            op = all_ops[op_idx]
            op_records = [r for r in records if r["op_idx"] == op_idx]
            if not op_records:
                continue
            what = _describe_op(op, op_idx, all_ops).split("\n")[0]
            summary = (
                f"<b>{op['op_name']}</b> · {op['cores']} cores · {what}"
                f" · {_sdsc_stem(op)}"
            )
            if op.get("lx_live_pct") is not None:
                summary += f" · LX live {op['lx_live_pct']:.0f}%"
            cost = op.get("cost")
            if cost and cost.get("t_us") is not None:
                summary += f" · T {cost['t_us']:.1f} us"
            print(f"<details><summary>{summary}</summary>\n")
            print(_tabulate_records(op_records, per_op_cols, headers, "github"))
            print("\n</details>\n")


def _solve_quality_lines(report: dict) -> list[str]:
    """What to know before reading any "the solver chose X" below.

    A FEASIBLE plan is the best found before a time limit, not the best there
    is; a plan from the fallback was not chosen by this objective at all. Both
    read identically in the table, so they are said once, at the top.
    """
    seen: dict = {}
    for op in report["ops"]:
        solve = op.get("solve") or {}
        status = solve.get("status")
        if status:
            seen.setdefault(status, solve)
    lines = []
    for status, solve in seen.items():
        if status == "OPTIMAL":
            continue
        if status == "NOT_LINEARIZABLE":
            lines.append(
                "Solver: the objective could not be lowered, so this plan came "
                "from the fallback -- the Objective column describes a cost "
                "nothing minimized"
            )
            continue
        limit, took = solve.get("limit_s"), solve.get("solve_s")
        at_limit = (
            isinstance(limit, (int, float))
            and isinstance(took, (int, float))
            and took >= 0.95 * limit
        )
        lines.append(
            f"Solver: {status}, not OPTIMAL"
            + (f" -- the solve hit its {limit:g}s limit" if at_limit else "")
            + "; the plan below is the best found, not the best there is"
        )
    caps = {
        (op.get("env") or {}).get("sencores")
        for op in report["ops"]
        if (op.get("env") or {}).get("sencores")
    }
    if len(caps) > 1:
        lines.append(
            f"Core cap varies by compile: {', '.join(str(c) for c in sorted(caps))} "
            "-- the Cores column is read against each graph's own cap"
        )
    return lines


def _relayout_summary_line(report: dict) -> str:
    """One line on what the plan paid to relayout: how many of the priced copies
    it kept and what they cost. A copy the solver priced but did not place is an
    alternative it declined, so the two counts together say whether relayout was
    on the table at all and whether it was taken."""
    r = report.get("relayouts") or {}
    total = r.get("copies") or 0
    if not total:
        return ""
    fired = r.get("fired") or 0
    if not fired:
        return f"Relayouts: none of {total} priced copies fired"
    srcs = r.get("sources") or []
    shown = ", ".join(srcs[:4]) + (f" +{len(srcs) - 4} more" if len(srcs) > 4 else "")
    return (
        f"Relayouts: {fired} of {total} priced copies fired, "
        f"{(r.get('charged_ns') or 0.0) / 1000:.2f} us charged"
        + (f" (sources: {shown})" if shown else "")
    )


def render_report(
    report: dict,
    table_format: str = "text",
    show_what: bool = True,
    collapsible: bool = False,
    basic: bool = False,
) -> None:
    """Print a report from :func:`build_report`.

    table_format: "text" (box table, a rule between ops), "github" (Markdown
    pipe table; with ``collapsible`` one <details> block per op under a
    heading per kernel), "html".
    basic: the compact column set (no footprint, occupancy, cost or kernel).
    """
    heading = "## " if table_format != "text" else ""
    index_range = report.get("index_range")
    print(f"\n{heading}SDSC Operations Summary - Batch Report")
    print(f"Directory: {report['base_dir']}")
    range_note = (
        f" (sdsc index {index_range[0]}-{index_range[1]})" if index_range else ""
    )
    print(f"Total sdsc.json files found: {report['n_files']}{range_note}\n")
    all_ops = report["ops"]
    if not all_ops:
        print("No operations found in any sdsc.json files.\n")
        return
    print(f"{heading}Operations Summary:\n")
    seen_ops = set()
    for op in all_ops:
        op_name = op["op_name"]
        if op_name in seen_ops:
            continue
        seen_ops.add(op_name)
        tensors_desc = (
            ", ".join(f"{t['role']} ({t['component']})" for t in op["tensors"])
            or "no tensors"
        )
        move_op = op.get("move_op")
        suffix = f"  [{move_op}]" if move_op else ""
        if table_format == "text":
            print(f"{op_name:15} - {tensors_desc}{suffix}")
        else:
            print(f"- `{op_name}` - {tensors_desc}{suffix}")
    if not basic:
        peak = max((op.get("lx_live_bytes") or 0) for op in all_ops)
        cap = report.get("lx_capacity") or 0
        pct = (
            f" ({100.0 * peak / cap:.0f}% of {_bytes_label(cap)} per core)"
            if cap
            else ""
        )
        print(f"\nPeak LX live: {_bytes_label(peak)}{pct}")
        if report.get("cost_matched"):
            print(
                f"Cost dump joined for {report['cost_matched']} of {len(all_ops)} ops"
            )
        if report.get("objective_matched"):
            print(
                f"Solver objective joined for {report['objective_matched']} of "
                f"{len(all_ops)} ops"
            )
        for line in (
            *_solve_quality_lines(report),
            _residency_summary_line(report),
            _relayout_summary_line(report),
        ):
            if line:
                print(line)
    print(f"\n{heading}Tensor Summary Table:\n")
    records = build_records(report)
    cols = _select_columns(report, basic, show_what)
    if collapsible and table_format == "github":
        _render_collapsible(report, records, cols)
        return
    headers = _headers_for(cols, report, table_format)
    print(_tabulate_records(records, cols, headers, table_format))


def _kernel_groups(report: dict) -> list[tuple[str, list[dict]]]:
    """Ops grouped by kernel, in the order the kernels first appear."""
    groups: dict[str, list[dict]] = {}
    for op in report["ops"]:
        groups.setdefault(op.get("kernel", ""), []).append(op)
    return list(groups.items())


def _align_kernels(a: list[tuple], b: list[tuple]) -> list[tuple]:
    """Pair the two runs' kernels. The kernel directory is named by a content
    hash, so the SAME kernel compiled from a changed planner gets a different
    name; what survives is the op-name sequence. Exact sequences pair first,
    then what is left pairs by op-name overlap, so a kernel that gained or lost
    an op (a relayout shuffle appearing) still finds its counterpart.

    Returns ``(kernel_a | None, ops_a, kernel_b | None, ops_b)`` tuples, with
    ``None`` on the side that has no counterpart."""
    pairs: list[tuple] = []
    left = list(a)
    right = list(b)
    for ka, ops_a in list(left):
        sig = tuple(op["op_name"] for op in ops_a)
        hit = next(
            (item for item in right if tuple(op["op_name"] for op in item[1]) == sig),
            None,
        )
        if hit is None:
            continue
        pairs.append((ka, ops_a, hit[0], hit[1]))
        left.remove((ka, ops_a))
        right.remove(hit)
    # Best remaining overlap, greedily, so one kernel cannot claim two.
    scored = []
    for ka, ops_a in left:
        names_a = collections.Counter(op["op_name"] for op in ops_a)
        for kb, ops_b in right:
            names_b = collections.Counter(op["op_name"] for op in ops_b)
            shared = sum((names_a & names_b).values())
            if shared:
                scored.append((shared, ka, ops_a, kb, ops_b))
    taken_a: set = set()
    taken_b: set = set()
    for _n, ka, ops_a, kb, ops_b in sorted(scored, key=lambda x: -x[0]):
        if ka in taken_a or kb in taken_b:
            continue
        pairs.append((ka, ops_a, kb, ops_b))
        taken_a.add(ka)
        taken_b.add(kb)
    pairs.extend((ka, ops_a, None, []) for ka, ops_a in left if ka not in taken_a)
    pairs.extend((None, [], kb, ops_b) for kb, ops_b in right if kb not in taken_b)
    return pairs


def _align_ops(ops_a: list[dict], ops_b: list[dict]) -> list[tuple]:
    """Pair two kernels' ops. Equal lengths pair by position; otherwise a
    longest-common-subsequence on the op names, so an inserted or removed op
    shifts nothing after it."""
    if len(ops_a) == len(ops_b):
        return list(zip(ops_a, ops_b))
    na = [op["op_name"] for op in ops_a]
    nb = [op["op_name"] for op in ops_b]
    lcs = [[0] * (len(nb) + 1) for _ in range(len(na) + 1)]
    for i in range(len(na) - 1, -1, -1):
        for j in range(len(nb) - 1, -1, -1):
            lcs[i][j] = (
                lcs[i + 1][j + 1] + 1
                if na[i] == nb[j]
                else max(lcs[i + 1][j], lcs[i][j + 1])
            )
    out: list[tuple] = []
    i = j = 0
    while i < len(na) and j < len(nb):
        if na[i] == nb[j]:
            out.append((ops_a[i], ops_b[j]))
            i += 1
            j += 1
        elif lcs[i + 1][j] >= lcs[i][j + 1]:
            out.append((ops_a[i], None))
            i += 1
        else:
            out.append((None, ops_b[j]))
            j += 1
    out.extend((ops_a[k], None) for k in range(i, len(na)))
    out.extend((None, ops_b[k]) for k in range(j, len(nb)))
    return out


def _op_facts(op: dict | None) -> dict[str, str]:
    """The comparable facts of one op, as strings: what a reader would look at
    to decide whether two runs planned this op the same way. Absent values are
    left out rather than rendered empty, so a fact one run did not record does
    not read as a change."""
    if op is None:
        return {}
    facts: dict[str, str] = {
        "placement": op.get("components_str", ""),
        "cores": str(op.get("cores", "")),
    }
    if op.get("lx_live_bytes") is not None:
        facts["LX live"] = _bytes_label(op["lx_live_bytes"])
    cost = op.get("cost") or {}
    if cost.get("attributed_us") is not None:
        facts["cost"] = f"{cost['attributed_us']:.1f} us"
    res = op.get("residency") or {}
    if res:
        facts["residency"] = "LX" if res.get("lx") else (res.get("reason") or "HBM")
    obj = op.get("objective") or {}
    if obj.get("own_value_ns") is not None:
        facts["own terms"] = f"{obj['own_value_ns'] / 1000:.2f} us"
    divisions = obj.get("divisions") or []
    if divisions:
        facts["division"] = divisions[0]
    return facts


def _run_totals(report: dict) -> dict[str, str]:
    """The whole-run numbers a diff leads with."""
    ops = report["ops"]
    comps = (report.get("stats") or {}).get("component_types") or {}
    totals = {
        "ops": str(len(ops)),
        "LX tensors": f"{comps.get('lx', 0)} of {sum(comps.values())}",
        "peak LX live": _bytes_label(max((o.get("lx_live_bytes") or 0) for o in ops))
        if ops
        else "",
    }
    # One prediction per kernel, so sum the distinct kernels, not the ops.
    per_kernel: dict[str, float] = {}
    for op in ops:
        t = (op.get("cost") or {}).get("t_us")
        if t is not None:
            per_kernel.setdefault(op.get("kernel", ""), t)
    if per_kernel:
        totals["predicted"] = f"{sum(per_kernel.values()):.1f} us"
    r = report.get("relayouts") or {}
    if r.get("copies"):
        totals["relayouts fired"] = (
            f"{r.get('fired', 0)} of {r['copies']}, "
            f"{(r.get('charged_ns') or 0.0) / 1000:.2f} us"
        )
    if report.get("residency_matched"):
        lx = sum(1 for o in ops if (o.get("residency") or {}).get("lx"))
        totals["outputs in LX"] = f"{lx} of {report['residency_matched']}"
    return totals


def render_diff(
    rep_a: dict, rep_b: dict, table_format: str = "text", width: int | None = None
) -> None:
    """Print what changed between two captured runs.

    Same model, two planners: the question is never "what does this run look
    like" but "what did the change move". Kernels are paired by their op-name
    sequence (the directory hash changes with the code), ops within a kernel by
    position or by a longest-common-subsequence, and only the facts that differ
    are printed -- an op planned identically is silence.

    ``width`` is the terminal width the text output fits (``None``: no limit);
    tables wrap their widest columns and prose wraps at word boundaries."""
    import textwrap

    heading = "## " if table_format != "text" else ""
    if table_format != "text":
        width = None

    def prose(text: str) -> str:
        return textwrap.fill(text, width) if width else text

    print(f"\n{heading}SDSC Run Diff")
    print(f"A: {rep_a['base_dir']}")
    print(f"B: {rep_b['base_dir']}\n")
    ta, tb = _run_totals(rep_a), _run_totals(rep_b)
    rows = [
        [k, ta.get(k, ""), tb.get(k, ""), "" if ta.get(k) == tb.get(k) else "changed"]
        for k in dict.fromkeys(list(ta) + list(tb))
    ]
    print(f"{heading}Totals\n")
    print(
        _tabulate_records(
            [
                {"first": i == 0, "what": r[0], "a": r[1], "b": r[2], "note": r[3]}
                for i, r in enumerate(rows)
            ],
            ["what", "a", "b", "note"],
            ["", "A", "B", ""],
            table_format,
            width,
        )
    )
    # A fact only one run recorded is a missing artifact, not a plan change:
    # diff a run whose records carry residency reasons against one without
    # it and every op would "change" residency. Compare only the kinds of fact
    # BOTH runs carry, and say which kinds were dropped.
    keys_a = {k for op in rep_a["ops"] for k in _op_facts(op)}
    keys_b = {k for op in rep_b["ops"] for k in _op_facts(op)}
    comparable = keys_a & keys_b
    dropped = sorted((keys_a | keys_b) - comparable)
    changes: list[dict] = []
    # The kernel directory is a content hash and differs between the runs, so
    # number the pairs instead: "k2/sdsc_11" is stable across both sides and
    # tells two identically named ops in different kernels apart.
    for ki, (_ka, ops_a, _kb, ops_b) in enumerate(
        _align_kernels(_kernel_groups(rep_a), _kernel_groups(rep_b)), start=1
    ):
        for op_a, op_b in _align_ops(ops_a, ops_b):
            fa, fb = _op_facts(op_a), _op_facts(op_b)
            label = (op_a or op_b)["op_name"]
            where = f"k{ki}/{_sdsc_stem(op_a or op_b)}"
            if op_a is None or op_b is None:
                changes.append(
                    {
                        "op": f"{label} ({where})",
                        "field": "only in A" if op_b is None else "only in B",
                        "a": ", ".join(f"{k}={v}" for k, v in fa.items()),
                        "b": ", ".join(f"{k}={v}" for k, v in fb.items()),
                    }
                )
                continue
            first = True
            for key in dict.fromkeys(list(fa) + list(fb)):
                if key not in comparable or fa.get(key, "") == fb.get(key, ""):
                    continue
                changes.append(
                    {
                        "op": f"{label} ({where})" if first else "",
                        "field": key,
                        "a": fa.get(key, ""),
                        "b": fb.get(key, ""),
                    }
                )
                first = False
    print(f"\n{heading}Changed ops: {len({c['op'] for c in changes if c['op']})}\n")
    if not changes:
        print("Every paired op was planned identically.\n")
        return
    print(
        prose(
            "Ops are labelled by paired-kernel ordinal and sdsc index on the A "
            "side (B's where the op exists only in B)."
        )
        + "\n"
    )
    if dropped:
        print(
            prose(f"Not compared (only one run recorded it): {', '.join(dropped)}.")
            + "\n"
        )
    print(
        _tabulate_records(
            [dict(c, first=bool(c["op"])) for c in changes],
            ["op", "field", "a", "b"],
            ["Op", "", "A", "B"],
            table_format,
            width,
        )
    )


def batch_summarize_directory(
    base_dir_str: str,
    index_range: tuple[int, int] | None = None,
    meta: dict | None = None,
    log_path: Path | None = None,
    table_format: str = "text",
    show_what: bool = True,
    cost_path: Path | None = None,
    lx_capacity: int = _LX_CAPACITY_DEFAULT,
    collapsible: bool = False,
    basic: bool = False,
    cost_expr_path: Path | None = None,
    probe_path: Path | None = None,
) -> None:
    """Parse and print in one call (the CLI entry point)."""
    report = build_report(
        base_dir_str,
        index_range=index_range,
        meta=meta,
        log_path=log_path,
        cost_path=cost_path,
        lx_capacity=lx_capacity,
        cost_expr_path=cost_expr_path,
        probe_path=probe_path,
    )
    render_report(
        report,
        table_format=table_format,
        show_what=show_what,
        collapsible=collapsible,
        basic=basic,
    )


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Summarize sdsc.json files.")
    parser.add_argument("directory", nargs="?", help="Base directory to search")
    parser.add_argument(
        "--range",
        dest="range",
        metavar="LO-HI",
        help="Only show sdsc files whose index N satisfies LO <= N <= HI (e.g. --range 0-18)",
    )
    parser.add_argument(
        "--meta",
        metavar="FILE",
        help="JSON meta file that adds an annotation column (see skills/summarize-sdsc/)",
    )
    parser.add_argument(
        "--log",
        metavar="FILE",
        help="SPYRE_INDUCTOR_LOG DEBUG log; when given, annotate each op's "
        "OUTPUT row with its buffer name (bufN) parsed from the log.",
    )
    parser.add_argument(
        "--format",
        dest="table_format",
        choices=("text", "github", "html"),
        default="text",
        help="Table renderer: text (box table, default), github (Markdown pipe "
        "table for PR/issue comments and .md files), html.",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        help="Write the report to FILE instead of stdout (only the path is "
        "printed). Pair with --format github for a .md file.",
    )
    parser.add_argument(
        "--no-what",
        dest="show_what",
        action="store_false",
        help="Omit the leading 'What (-> consumer)' column.",
    )
    parser.add_argument(
        "--cost",
        metavar="FILE",
        help="Captured SPYRE_DUMP_COST=1 output; joins per-op read/write/lx "
        "bytes and the bundle prediction by output buffer name.",
    )
    parser.add_argument(
        "--cost-expr",
        metavar="FILE",
        help="SPYRE_DUMP_COST_EXPR_FILE output (JSON lines): the solver's symbolic "
        "objective per bundle, its solved bindings and evaluated terms.",
    )
    parser.add_argument(
        "--diff",
        metavar="DIR",
        help="Compare this run with another captured run: print the whole-run "
        "totals side by side and only the ops whose plan differs.",
    )
    parser.add_argument(
        "--width",
        type=int,
        metavar="COLS",
        help="With --diff: fit the text tables to COLS columns, wrapping the "
        "widest cells (0: never wrap). Default: $COLUMNS, else the terminal's "
        "width; output to a pipe or --out is not wrapped.",
    )
    parser.add_argument(
        "--probe",
        metavar="FILE",
        help="LX residency probe JSON: the allocator's own reason for every "
        "buffer it did not keep in LX. Only for runs captured before the "
        "cost-expression dump recorded reasons (PR #4738).",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Do not pick up cost_dump*.log / cost_expr*.jsonl / probe*.json "
        "found next to the "
        "artifacts.",
    )
    parser.add_argument(
        "--lx-capacity",
        type=int,
        default=_LX_CAPACITY_DEFAULT,
        metavar="BYTES",
        help="Per-core LX budget for the occupancy percentage "
        f"(default {_LX_CAPACITY_DEFAULT}, the planner's).",
    )
    parser.add_argument(
        "--collapsible",
        action="store_true",
        help="With --format github: one collapsed <details> block per op under "
        "a heading per kernel, instead of one flat table.",
    )
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Compact column set: no footprint, LX occupancy, cost or kernel.",
    )
    args = parser.parse_args()

    if args.directory:
        base_dir = args.directory
    else:
        pattern = "/tmp/torchinductor_*"
        matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if matches:
            base_dir = matches[0]
        else:
            print(
                f"Error: No torchinductor directories found matching {pattern}",
                file=sys.stderr,
            )
            sys.exit(1)

    index_range = None
    if args.range:
        lo_str, _, hi_str = args.range.partition("-")
        index_range = (int(lo_str), int(hi_str))

    meta = None
    if args.meta:
        with open(args.meta) as f:
            meta = json.load(f)

    log_path = Path(args.log).expanduser() if args.log else None
    if log_path is not None and not log_path.exists():
        print(f"Warning: --log file not found: {log_path}", file=sys.stderr)
        log_path = None

    cost_path = Path(args.cost).expanduser() if args.cost else None
    if cost_path is not None and not cost_path.exists():
        print(f"Warning: --cost file not found: {cost_path}", file=sys.stderr)
        cost_path = None
    cost_expr_path = Path(args.cost_expr).expanduser() if args.cost_expr else None
    if cost_expr_path is not None and not cost_expr_path.exists():
        print(f"Warning: --cost-expr file not found: {cost_expr_path}", file=sys.stderr)
        cost_expr_path = None
    # Companion dumps next to the artifacts, unless given or disabled.
    if cost_path is None and not args.no_auto:
        cost_path = find_companion(Path(base_dir), "cost")
        if cost_path:
            print(f"Using cost dump: {cost_path}", file=sys.stderr)
    if cost_expr_path is None and not args.no_auto:
        cost_expr_path = find_companion(Path(base_dir), "cost_expr")
        if cost_expr_path:
            print(f"Using cost-expression dump: {cost_expr_path}", file=sys.stderr)
    probe_path = Path(args.probe).expanduser() if args.probe else None
    if probe_path is not None and not probe_path.exists():
        print(f"Warning: --probe file not found: {probe_path}", file=sys.stderr)
        probe_path = None
    if probe_path is None and not args.no_auto:
        probe_path = find_companion(Path(base_dir), "probe")
        if probe_path:
            print(f"Using residency probe: {probe_path}", file=sys.stderr)

    print(f"Using directory: {base_dir}\n", file=sys.stderr)
    kwargs = {
        "index_range": index_range,
        "meta": meta,
        "log_path": log_path,
        "table_format": args.table_format,
        "show_what": args.show_what,
        "cost_path": cost_path,
        "lx_capacity": args.lx_capacity,
        "collapsible": args.collapsible,
        "basic": args.basic,
        "cost_expr_path": cost_expr_path,
        "probe_path": probe_path,
    }

    def _emit() -> None:
        if args.diff:
            # Both runs are parsed the same way, companion dumps and all, so a
            # fact present in one and missing in the other is a real difference
            # and not an artifact of how each side was read.
            other = str(Path(args.diff).expanduser())
            build = {
                k: v
                for k, v in kwargs.items()
                if k
                in (
                    "index_range",
                    "meta",
                    "log_path",
                    "cost_path",
                    "lx_capacity",
                    "cost_expr_path",
                    "probe_path",
                )
            }
            other_build = dict(build)
            for key, kind in (
                ("cost_path", "cost"),
                ("cost_expr_path", "cost_expr"),
                ("probe_path", "probe"),
            ):
                other_build[key] = (
                    find_companion(Path(other), kind) if not args.no_auto else None
                )
            render_diff(
                build_report(base_dir, **build),
                build_report(other, **other_build),
                table_format=args.table_format,
                # A file written with --out has no terminal: wrap it only when
                # --width asks for it.
                width=(
                    _terminal_width(args.width)
                    if args.width is not None or not args.out
                    else None
                ),
            )
            return
        batch_summarize_directory(base_dir, **kwargs)

    if args.out:
        import contextlib

        out_path = Path(args.out).expanduser()
        with open(out_path, "w") as out_file, contextlib.redirect_stdout(out_file):
            _emit()
        print(f"Wrote {out_path}")
    else:
        _emit()
