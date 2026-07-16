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
    source = Path(filepath).read_text()
    tree = ast.parse(source)
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
    source = Path(filepath).read_text()
    tree = ast.parse(source)
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
    source = Path(filepath).read_text()
    tree = ast.parse(source)
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
    source = Path(filepath).read_text()
    tree = ast.parse(source)
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

        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            func = _resolve_attr(call.func)
            if func == "fallback_ops.append" and call.args:
                raw = _resolve_attr(call.args[0])
                op = _normalize_op_name(raw)
                if op:
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

    return nodes, edges


def extract_eager_kernels(filepath):
    """Parse register_torch_compile_kernel() and @torch.library.register_kernel."""
    nodes = []
    edges = []
    source = Path(filepath).read_text()
    tree = ast.parse(source)
    rel_path = str(filepath)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _resolve_attr(node.func)
            if func_name == "register_torch_compile_kernel" and node.args:
                ops = _extract_list_ops(node.args[0])
                for op in ops:
                    eager_id = f"eager::{op}"
                    op_id = f"op::{op}"
                    nodes.append(
                        {
                            "id": eager_id,
                            "label": f"{op} (eager)",
                            "type": "eager_kernel",
                            "source_file": rel_path,
                            "line": node.lineno,
                        }
                    )
                    nodes.append({"id": op_id, "label": op, "type": "op"})
                    edges.append(
                        {
                            "source": op_id,
                            "target": eager_id,
                            "relationship": "eager_via",
                        }
                    )

        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                dec_name = _get_decorator_name(dec)
                if dec_name and "register_kernel" in dec_name:
                    if isinstance(dec, ast.Call) and dec.args:
                        first_arg = dec.args[0]
                        if isinstance(first_arg, ast.Constant) and isinstance(
                            first_arg.value, str
                        ):
                            op_name = first_arg.value
                            eager_id = f"eager::{op_name}"
                            op_id = f"op::{op_name}"
                            nodes.append(
                                {
                                    "id": eager_id,
                                    "label": f"{op_name} (eager)",
                                    "type": "eager_kernel",
                                    "source_file": rel_path,
                                    "line": node.lineno,
                                }
                            )
                            nodes.append(
                                {
                                    "id": op_id,
                                    "label": op_name,
                                    "type": "op",
                                }
                            )
                            edges.append(
                                {
                                    "source": op_id,
                                    "target": eager_id,
                                    "relationship": "eager_via",
                                }
                            )

    return nodes, edges


def extract_passes(filepath):
    """Parse Custom*Passes and _Spyre*Pass classes with their pass lists."""
    nodes = []
    edges = []
    source = Path(filepath).read_text()
    tree = ast.parse(source)
    rel_path = str(filepath)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not node.name.startswith("Custom") and not node.name.startswith("_Spyre"):
            continue
        if "Pass" not in node.name:
            continue

        group_id = f"pass::{node.name}"
        nodes.append(
            {
                "id": group_id,
                "label": node.name,
                "type": "pass_group",
                "source_file": rel_path,
                "line": node.lineno,
            }
        )

        for base in node.bases:
            base_name = _resolve_attr(base)
            if base_name:
                base_id = f"class::{base_name}"
                nodes.append(
                    {
                        "id": base_id,
                        "label": base_name,
                        "type": "class",
                    }
                )
                edges.append(
                    {
                        "source": group_id,
                        "target": base_id,
                        "relationship": "inherits_from",
                    }
                )

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func_name = _resolve_attr(child.func)
                if func_name and "__init__" in func_name and child.args:
                    for arg in child.args:
                        if isinstance(arg, ast.List):
                            for elt in arg.elts:
                                name = _resolve_attr(elt)
                                if name:
                                    fn_id = f"passfn::{name}"
                                    nodes.append(
                                        {
                                            "id": fn_id,
                                            "label": name,
                                            "type": "pass_function",
                                            "source_file": rel_path,
                                            "line": getattr(elt, "lineno", node.lineno),
                                        }
                                    )
                                    edges.append(
                                        {
                                            "source": group_id,
                                            "target": fn_id,
                                            "relationship": "contains_pass",
                                        }
                                    )
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == "passes"
                        and isinstance(child.value, ast.List)
                    ):
                        for elt in child.value.elts:
                            name = _resolve_attr(elt)
                            if name:
                                fn_id = f"passfn::{name}"
                                nodes.append(
                                    {
                                        "id": fn_id,
                                        "label": name,
                                        "type": "pass_function",
                                        "source_file": rel_path,
                                        "line": getattr(elt, "lineno", node.lineno),
                                    }
                                )
                                edges.append(
                                    {
                                        "source": group_id,
                                        "target": fn_id,
                                        "relationship": "contains_pass",
                                    }
                                )

    return nodes, edges


# ---------------------------------------------------------------------------
# Architecture extractors
# ---------------------------------------------------------------------------


def extract_classes(filepath):
    """Extract class definitions and inheritance hierarchies."""
    nodes = []
    edges = []
    source = Path(filepath).read_text()
    tree = ast.parse(source)
    rel_path = str(filepath)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name.startswith("_") and "Spyre" not in node.name:
            continue

        is_dataclass = any(
            _get_decorator_name(d) in ("dataclass", "dataclasses.dataclass")
            for d in node.decorator_list
        )

        class_id = f"class::{node.name}"
        node_type = "dataclass" if is_dataclass else "class"
        nodes.append(
            {
                "id": class_id,
                "label": node.name,
                "type": node_type,
                "source_file": rel_path,
                "line": node.lineno,
            }
        )

        for base in node.bases:
            base_name = _resolve_attr(base)
            if base_name and base_name not in ("object", "ABC", "abc.ABC"):
                base_id = f"class::{base_name}"
                nodes.append(
                    {
                        "id": base_id,
                        "label": base_name,
                        "type": "class",
                    }
                )
                edges.append(
                    {
                        "source": class_id,
                        "target": base_id,
                        "relationship": "inherits_from",
                    }
                )

    return nodes, edges


def extract_config(filepath):
    """Extract environment variable declarations from config module."""
    nodes = []
    edges = []
    source = Path(filepath).read_text()
    tree = ast.parse(source)
    rel_path = str(filepath)

    module_id = f"module::{rel_path}"
    nodes.append(
        {
            "id": module_id,
            "label": "config",
            "type": "module",
            "source_file": rel_path,
            "line": 1,
        }
    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = _resolve_attr(node.func)
        if func_name not in (
            "os.environ.get",
            "os.getenv",
            "os.environ.setdefault",
        ):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not (
            isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str)
        ):
            continue

        env_name = first_arg.value
        default_val = None
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            default_val = node.args[1].value

        env_id = f"envvar::{env_name}"
        label = env_name
        if default_val is not None:
            label = f"{env_name}={default_val}"

        nodes.append(
            {
                "id": env_id,
                "label": label,
                "type": "env_var",
                "source_file": rel_path,
                "line": getattr(node, "lineno", 0),
            }
        )
        edges.append(
            {
                "source": module_id,
                "target": env_id,
                "relationship": "reads_env",
            }
        )

    return nodes, edges


def extract_modules(torch_spyre_root, repo_root):
    """Build a module dependency graph from intra-package imports."""
    nodes = []
    edges = []
    root = Path(torch_spyre_root)

    py_files = sorted(root.rglob("*.py"))

    for filepath in py_files:
        if "__pycache__" in str(filepath):
            continue
        rel_path = str(filepath.relative_to(repo_root))
        mod_name = _module_from_path(filepath, repo_root)

        if mod_name == "torch_spyre.version":
            continue

        mod_id = f"module::{mod_name}"
        short_label = mod_name.replace("torch_spyre.", "")
        nodes.append(
            {
                "id": mod_id,
                "label": short_label,
                "type": "module",
                "source_file": rel_path,
                "line": 1,
            }
        )

        try:
            source = filepath.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            target_mod = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("torch_spyre"):
                        target_mod = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("torch_spyre"):
                    target_mod = node.module

            if target_mod and target_mod != mod_name:
                target_id = f"module::{target_mod}"
                edges.append(
                    {
                        "source": mod_id,
                        "target": target_id,
                        "relationship": "imports",
                    }
                )

    return nodes, edges


def extract_codegen_structures(filepath):
    """Extract codegen IR dataclasses and their field relationships."""
    nodes = []
    edges = []
    source = Path(filepath).read_text()
    tree = ast.parse(source)
    rel_path = str(filepath)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        is_dataclass = any(
            _get_decorator_name(d) in ("dataclass", "dataclasses.dataclass")
            for d in node.decorator_list
        )
        if not is_dataclass:
            continue

        class_id = f"class::{node.name}"
        nodes.append(
            {
                "id": class_id,
                "label": node.name,
                "type": "dataclass",
                "source_file": rel_path,
                "line": node.lineno,
            }
        )

        for base in node.bases:
            base_name = _resolve_attr(base)
            if base_name and base_name not in ("object",):
                base_id = f"class::{base_name}"
                nodes.append(
                    {
                        "id": base_id,
                        "label": base_name,
                        "type": "class",
                    }
                )
                edges.append(
                    {
                        "source": class_id,
                        "target": base_id,
                        "relationship": "inherits_from",
                    }
                )

        known_types = {
            "SDSCArgs",
            "SDSCSpec",
            "OpSpec",
            "LoopSpec",
            "TensorArg",
            "FixedTiledLayout",
            "SpyreTensorLayout",
            "RValue",
            "TensorAccess",
            "PointwiseOp",
            "ReductionOp",
            "SymbolKind",
            "UnimplementedOp",
        }
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and item.annotation:
                ann = _resolve_attr(item.annotation)
                if ann and ann in known_types:
                    ref_id = f"class::{ann}"
                    nodes.append(
                        {
                            "id": ref_id,
                            "label": ann,
                            "type": "dataclass",
                        }
                    )
                    edges.append(
                        {
                            "source": class_id,
                            "target": ref_id,
                            "relationship": "contains_field",
                        }
                    )

    return nodes, edges


# ---------------------------------------------------------------------------
# Build orchestrator
# ---------------------------------------------------------------------------


def _get_git_sha(repo_root):
    """Get the current HEAD SHA for metadata."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _run_file_extractor(extractor, filepath, repo_root):
    """Run a single-file extractor and normalize the path it was handed.

    Extractors read their input via the path they are handed and echo it
    back as each node's ``source_file``; ``extract_config`` also bakes that
    path into the config module node id (and the endpoints of its
    ``reads_env`` edges). We pass the absolute path so the read resolves
    regardless of the current working directory (the docs build runs from
    docs/source on ReadTheDocs, not the repo root), then rewrite the absolute
    path back to one relative to ``repo_root`` wherever it appears so node
    ids and ``source_file`` stay stable across build environments.
    """
    abs_path = str(filepath)
    rel_path = str(Path(filepath).relative_to(repo_root))

    nodes, edges = extractor(abs_path)

    id_remap = {}
    for node in nodes:
        if node.get("source_file"):
            node["source_file"] = node["source_file"].replace(abs_path, rel_path)
        if abs_path in node["id"]:
            new_id = node["id"].replace(abs_path, rel_path)
            id_remap[node["id"]] = new_id
            node["id"] = new_id

    for edge in edges:
        for endpoint in ("source", "target"):
            if edge.get(endpoint) in id_remap:
                edge[endpoint] = id_remap[edge[endpoint]]

    return nodes, edges


def build_graph(torch_spyre_root):
    """Run all extractors and assemble the deduplicated graph."""
    root = Path(torch_spyre_root)
    repo_root = root.parent

    all_nodes = []
    all_edges = []

    # --- Op-level extractors ---
    op_extractors = [
        (extract_decompositions, root / "_inductor" / "decompositions.py"),
        (extract_lowerings, root / "_inductor" / "lowering.py"),
        (extract_custom_ops, root / "_inductor" / "customops.py"),
        (extract_fallbacks, root / "ops" / "fallbacks.py"),
        (extract_eager_kernels, root / "ops" / "eager.py"),
        (extract_passes, root / "_inductor" / "passes.py"),
    ]

    for extractor, filepath in op_extractors:
        if filepath.exists():
            n, e = _run_file_extractor(extractor, filepath, repo_root)
            all_nodes.extend(n)
            all_edges.extend(e)

    # --- Architecture: classes from key files ---
    class_files = [
        root / "_inductor" / "spyre_kernel.py",
        root / "_inductor" / "ir.py",
        root / "_inductor" / "scheduler.py",
        root / "_inductor" / "optimize_restickify.py",
        root / "_inductor" / "propagate_hints.py",
        root / "_inductor" / "work_division.py",
        root / "_inductor" / "memory_planning.py",
        root / "device" / "interface.py",
        root / "execution" / "async_compile.py",
        root / "execution" / "kernel_runner.py",
        root / "streams.py",
        root / "__init__.py",
    ]

    for filepath in class_files:
        if filepath.exists():
            n, e = _run_file_extractor(extract_classes, filepath, repo_root)
            all_nodes.extend(n)
            all_edges.extend(e)

    # --- Codegen IR structures ---
    codegen_files = [
        root / "_inductor" / "codegen" / "superdsc.py",
        root / "_inductor" / "codegen" / "compute_ops.py",
        root / "_inductor" / "codegen" / "bundle.py",
        root / "_inductor" / "op_spec.py",
    ]

    for filepath in codegen_files:
        if filepath.exists():
            n, e = _run_file_extractor(extract_codegen_structures, filepath, repo_root)
            all_nodes.extend(n)
            all_edges.extend(e)

    # --- Configuration / environment variables ---
    config_path = root / "_inductor" / "config.py"
    if config_path.exists():
        n, e = _run_file_extractor(extract_config, config_path, repo_root)
        all_nodes.extend(n)
        all_edges.extend(e)
    for extra_config in [root / "__init__.py", root / "logging_config.py"]:
        if extra_config.exists():
            n, e = _run_file_extractor(extract_config, extra_config, repo_root)
            all_nodes.extend(n)
            all_edges.extend(e)

    # --- Module dependency graph ---
    n, e = extract_modules(str(root), repo_root)
    all_nodes.extend(n)
    all_edges.extend(e)

    # --- Deduplicate nodes by id ---
    seen = {}
    for node in all_nodes:
        nid = node["id"]
        if nid not in seen:
            seen[nid] = node
        else:
            if "source_file" in node and "source_file" not in seen[nid]:
                seen[nid].update(node)

    # --- Deduplicate edges ---
    edge_keys = set()
    unique_edges = []
    for edge in all_edges:
        key = (edge["source"], edge["target"], edge["relationship"])
        if key not in edge_keys:
            edge_keys.add(key)
            unique_edges.append(edge)

    # --- Remove edges referencing non-existent nodes ---
    node_ids = set(seen.keys())
    valid_edges = [
        e for e in unique_edges if e["source"] in node_ids and e["target"] in node_ids
    ]

    graph = {
        "metadata": {
            "source_commit": _get_git_sha(repo_root),
            "torch_spyre_root": str(root.relative_to(repo_root)),
            # Canonical repo used to build "view source" links in the explorer.
            # The JS pins links to source_commit when available and falls back
            # to the default branch otherwise.
            "repo_url": "https://github.com/torch-spyre/torch-spyre",
            "default_branch": "main",
        },
        "nodes": list(seen.values()),
        "edges": valid_edges,
    }

    return graph


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = str(Path(__file__).resolve().parents[2].parent / "torch_spyre")

    graph = build_graph(root)
    print(f"Nodes: {len(graph['nodes'])}")
    print(f"Edges: {len(graph['edges'])}")

    type_counts = {}
    for n in graph["nodes"]:
        t = n["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    out = Path(__file__).resolve().parent.parent / "_static" / "js" / "graph.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"\nWritten to {out}")
