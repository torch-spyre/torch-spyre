# Copyright 2025 The Torch-Spyre Authors. All Rights Reserved.
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
"""Extract a knowledge graph from torch-spyre source via AST parsing.

Produces a JSON graph covering:
- Operations: decompositions, lowerings, custom ops, fallbacks, eager kernels
- Compiler passes: pass groups and their constituent functions
- Architecture: class hierarchies, dataclasses, module relationships
- Configuration: environment variables and their controlling modules
- Codegen: IR data structures and their relationships
- Runtime: device registration, streams, execution classes

No imports of torch or torch_spyre are needed — extraction is purely syntactic.
"""

import ast
import json
import subprocess
from pathlib import Path
import importlib.util

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_attr(node):
    """Recursively resolve an ast.Attribute chain to a dotted string."""
    if isinstance(node, ast.Attribute):
        parent = _resolve_attr(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _normalize_op_name(raw):
    """Strip common prefixes to produce a short op identifier."""
    if not raw:
        return None
    for prefix in ("torch.ops.", "torch._ops.ops."):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break
    return raw


def _extract_list_ops(list_node):
    """Extract op names from an ast.List of Attribute references."""
    ops = []
    if not isinstance(list_node, ast.List):
        return ops
    for elt in list_node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            ops.append(elt.value)
        else:
            name = _resolve_attr(elt)
            normalized = _normalize_op_name(name)
            if normalized:
                ops.append(normalized)
    return ops


def _get_decorator_name(decorator):
    """Get the function name from a decorator Call or Name node."""
    if isinstance(decorator, ast.Call):
        return _resolve_attr(decorator.func)
    return _resolve_attr(decorator)


def _module_from_path(filepath, repo_root):
    """Convert a file path to a Python module name."""
    rel = Path(filepath).relative_to(repo_root)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Op-level extractors
# ---------------------------------------------------------------------------


def extract_decompositions(filepath):
    """Parse @register_spyre_decomposition decorators."""
    nodes = []
    edges = []
    module = importlib.util.spec_from_file_location("module.name", filepath)
    if module:
        tree = ast.parse(module.loader.get_source(module.name))
    else:
        tree = ast.parse(Path(filepath).read_text())
    rel_path = str(filepath)

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            dec_name = _get_decorator_name(dec)
            if dec_name and (
                "register_spyre_decomposition" in dec_name
                or "register_decomposition" in dec_name
            ):
                if isinstance(dec, ast.Call) and dec.args:
                    ops = _extract_list_ops(dec.args[0])
                    decomp_id = f"decomp::{node.name}"
                    nodes.append(
                        {
                            "id": decomp_id,
                            "label": node.name,
                            "type": "decomposition",
                            "source_file": rel_path,
                            "line": node.lineno,
                        }
                    )
                    for op in ops:
                        op_id = f"op::{op}"
                        nodes.append(
                            {
                                "id": op_id,
                                "label": op,
                                "type": "op",
                            }
                        )
                        edges.append(
                            {
                                "source": op_id,
                                "target": decomp_id,
                                "relationship": "decomposed_by",
                            }
                        )
    return nodes, edges


def extract_lowerings(filepath):
    """Parse @register_spyre_lowering decorators."""
    nodes = []
    edges = []
    module = importlib.util.spec_from_file_location("module.name", filepath)
    if module:
        tree = ast.parse(module.loader.get_source(module.name))
    else:
        tree = ast.parse(Path(filepath).read_text())
    rel_path = str(filepath)

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            dec_name = _get_decorator_name(dec)
            if dec_name and "register_spyre_lowering" in dec_name:
                if isinstance(dec, ast.Call) and dec.args:
                    raw = _resolve_attr(dec.args[0])
                    op = _normalize_op_name(raw)
                    if op:
                        lowering_id = f"lowering::{node.name}"
                        op_id = f"op::{op}"
                        nodes.append(
                            {
                                "id": lowering_id,
                                "label": node.name,
                                "type": "lowering",
                                "source_file": rel_path,
                                "line": node.lineno,
                            }
                        )
                        nodes.append(
                            {
                                "id": op_id,
                                "label": op,
                                "type": "op",
                            }
                        )
                        edges.append(
                            {
                                "source": op_id,
                                "target": lowering_id,
                                "relationship": "lowered_by",
                            }
                        )
    return nodes, edges


def extract_custom_ops(filepath):
    """Parse @torch.library.custom_op decorators."""
    nodes = []
    edges = []
    module = importlib.util.spec_from_file_location("module.name", filepath)
    if module:
        tree = ast.parse(module.loader.get_source(module.name))
    else:
        tree = ast.parse(Path(filepath).read_text())
    rel_path = str(filepath)

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            dec_name = _get_decorator_name(dec)
            if dec_name and "custom_op" in dec_name:
                if isinstance(dec, ast.Call) and dec.args:
                    first_arg = dec.args[0]
                    if isinstance(first_arg, ast.Constant) and isinstance(
                        first_arg.value, str
                    ):
                        op_name = first_arg.value
                        custom_id = f"customop::{op_name}"
                        nodes.append(
                            {
                                "id": custom_id,
                                "label": op_name,
                                "type": "custom_op",
                                "source_file": rel_path,
                                "line": node.lineno,
                            }
                        )
    return nodes, edges


def extract_fallbacks(filepath):
    """Parse register_fallback_default() calls, @register_fallback, and appends."""
    nodes = []
    edges = []
    module = importlib.util.spec_from_file_location("module.name", filepath)
    if module:
        tree = ast.parse(module.loader.get_source(module.name))
    else:
        tree = ast.parse(Path(filepath).read_text())
    rel_path = str(filepath)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _resolve_attr(node.func)
            if func_name == "register_fallback_default" and node.args:
                ops = _extract_list_ops(node.args[0])
                for op in ops:
                    fb_id = f"fallback::{op}"
                    op_id = f"op::{op}"
                    nodes.append(
                        {
                            "id": fb_id,
                            "label": f"{op} (CPU fallback)",
                            "type": "fallback",
                            "source_file": rel_path,
                            "line": node.lineno,
                        }
                    )
                    nodes.append({"id": op_id, "label": op, "type": "op"})
                    edges.append(
                        {
                            "source": op_id,
                            "target": fb_id,
                            "relationship": "falls_back_to",
                        }
                    )

        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                dec_name = _get_decorator_name(dec)
                if dec_name == "register_fallback":
                    if isinstance(dec, ast.Call) and dec.args:
                        ops = _extract_list_ops(dec.args[0])
                        for op in ops:
                            fb_id = f"fallback::{op}"
                            op_id = f"op::{op}"
                            nodes.append(
                                {
                                    "id": fb_id,
                                    "label": f"{op} (CPU fallback)",
                                    "type": "fallback",
