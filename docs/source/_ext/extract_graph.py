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
import functools
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
    # Schema strings (``"aten::cumsum.out"``) name the same op as attribute
    # chains (``torch.ops.aten.cumsum.out``); use one id for both.
    return raw.replace("::", ".", 1)


def _module_op_aliases(tree):
    """Map module-level ``NAME = torch.ops.<ns>`` assignments to the namespace."""
    aliases = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            value = _resolve_attr(stmt.value)
            if value and value.startswith(("torch.ops.", "torch._ops.ops.")):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = _normalize_op_name(value)
    return aliases


def _apply_alias(raw, aliases):
    """Replace a leading alias name in a dotted op reference."""
    if not raw:
        return raw
    head, sep, rest = raw.partition(".")
    if head in aliases:
        return aliases[head] + sep + rest
    return raw


def _factory_packets(func):
    """Map ``V = getattr(torch.ops.<ns>, PARAM)`` in ``func`` to (ns, PARAM index)."""
    params = [a.arg for a in func.args.args]
    packets = {}
    for stmt in ast.walk(func):
        if not (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and _resolve_attr(stmt.value.func) == "getattr"
            and len(stmt.value.args) == 2
            and isinstance(stmt.value.args[1], ast.Name)
            and stmt.value.args[1].id in params
        ):
            continue
        ns = _normalize_op_name(_resolve_attr(stmt.value.args[0]))
        if not ns:
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                packets[target.id] = (ns, params.index(stmt.value.args[1].id))
    return packets


def _factory_call_literals(tree, func_name, index):
    """String literals passed at ``index`` in module-level calls to a function."""
    literals = []
    for stmt in tree.body:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Call)
                and _resolve_attr(node.func) == func_name
                and len(node.args) > index
                and isinstance(node.args[index], ast.Constant)
                and isinstance(node.args[index].value, str)
            ):
                literals.append(node.args[index].value)
    return literals


# Module-level calls that register eager kernels for a list of ops.
_EAGER_REGISTRARS = ("register_torch_compile_kernel", "register_inplace_kernels")

# In-place names the naming rule produces but ``_functional_sibling`` rejects,
# so no kernel is registered: ``pow.Scalar(Scalar, Tensor)`` has operands
# swapped relative to ``pow_.Scalar(Tensor, Scalar)``.
_NO_INPLACE_SIBLING = frozenset({"pow"})


def _inplace_name(op):
    """``aten.mul`` -> ``aten.mul_``, or ``None`` if there is no in-place form.

    The trailing ``_`` attaches to the op name, which is the last dotted
    component before any overload: ``aten.ne.Tensor`` -> ``aten.ne_.Tensor``.

    Mirrors the naming rule ``register_inplace_kernels`` relies on, then drops
    the shapes that never yield a registration: ``.out`` variants (they write
    through an explicit out-arg), already-in-place names, the operand-swapped
    ``pow`` pair, and -- when torch is importable -- names ATen does not define
    at all. Without this the ~37 derived in-place kernels are missing from the
    graph entirely.
    """
    parts = op.split(".")
    if len(parts) < 2:
        return None
    # Split name from overload: with 3+ components the last is the overload
    # (``aten.ne.Tensor``); with 2 only a known lowercase overload is one
    # (``aten.silu.out``), otherwise the last component is the name itself.
    if len(parts) > 2 or parts[-1] in ("out", "default"):
        name_idx = len(parts) - 2
    else:
        name_idx = len(parts) - 1
    name = parts[name_idx]
    overloads = parts[name_idx + 1 :]
    # ``.out``/``.Tensor_out`` write through an explicit out-arg; there is no
    # in-place sibling to register.
    if (
        name.endswith("_")
        or name in _NO_INPLACE_SIBLING
        or any(o == "out" or o.endswith("_out") for o in overloads)
    ):
        return None
    parts[name_idx] = name + "_"
    candidate = ".".join(parts)
    return candidate if _inplace_op_exists(candidate) else None


@functools.lru_cache(maxsize=1)
def _real_aten():
    """``torch.ops.aten`` if it can answer existence questions, else ``None``.

    Extraction must keep working with no torch *and* under the docs build's
    ``autodoc_mock_imports`` (see module docstring), where a mock would claim
    every name exists and silently prune nothing -- or raise on ``in``. The
    negative probe rejects such a stand-in: a real ATen has no ``definitely_``
    ``not_an_op``, a mock happily returns one.
    """
    try:
        import torch

        aten = torch.ops.aten
        if getattr(aten, "definitely_not_an_op_zzz", None) is not None:
            return None
        # A real packet answers ``overloads()`` with a list of str.
        if not isinstance(aten.mul_.overloads(), (list, tuple)):
            return None
        return aten
    except Exception:
        return None


def _inplace_op_exists(op):
    """Whether ``op`` is a real ATen overload, when that can be checked cheaply.

    An unusable torch (absent or mocked) means "assume it exists", keeping the
    syntactic guess. When a real one is importable this prunes names like
    ``aten.mm_``/``aten.cat_`` that the naming rule produces but ATen does not
    define, so the graph matches what actually gets registered.
    """
    aten = _real_aten()
    if aten is None:
        return True

    packet, _, overload = op.partition(".")[2].partition(".")
    try:
        aten_packet = getattr(aten, packet, None)
        if aten_packet is None:
            return False
        if not overload:
            # No overload named: the packet existing is the whole question. Do
            # not require a ``default`` -- ``mul_`` only has
            # ``Scalar``/``Tensor``.
            return True
        return overload in aten_packet.overloads()
    except Exception:
        return False


def _module_list_bindings(tree):
    """Map module-level ``NAME = [...]`` assignments to their ast.List node."""
    bindings = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.List):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = stmt.value
    return bindings


def _extract_list_ops(list_node, list_bindings=None):
    """Extract op names from an ast.List of Attribute references.

    ``list_bindings`` (from :func:`_module_list_bindings`) lets a bare name be
    resolved, so ``register_torch_compile_kernel(COMPILED_OPS)`` is parsed the
    same as an inline list literal.
    """
    ops = []
    if isinstance(list_node, ast.Name) and list_bindings:
        list_node = list_bindings.get(list_node.id, list_node)
    if not isinstance(list_node, ast.List):
        return ops
    for elt in list_node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            ops.append(_normalize_op_name(elt.value))
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
                    # Capture the ops this decomposition expands into by
                    # walking its body for explicit op calls. Best-effort:
                    # torch.ops.spyre.* / torch.ops.aten.* and bare torch.*
                    # functional calls are named here; Python operators
                    # (*, +, @) and tensor methods (.transpose) cannot be
                    # resolved statically and are omitted.
                    seen_targets = set()
                    for call in ast.walk(node):
                        if not isinstance(call, ast.Call):
                            continue
                        target = _normalize_op_name(_resolve_attr(call.func))
                        if not target:
                            continue
                        if target.startswith("torch.") and not target.startswith(
                            ("torch.ops.", "torch._ops.")
                        ):
                            target = "aten." + target[len("torch.") :]
                        if not target.startswith(("spyre.", "aten.")):
                            continue
                        if target in seen_targets:
                            continue
                        seen_targets.add(target)
                        target_id = f"op::{target}"
                        nodes.append({"id": target_id, "label": target, "type": "op"})
                        edges.append(
                            {
                                "source": decomp_id,
                                "target": target_id,
                                "relationship": "decomposes_into",
                            }
                        )
    return nodes, edges


def _lowering_registrations(tree):
    """(lowering FunctionDef, op name, lowering id suffix) for every
    @register_spyre_lowering, with factory-registered lowerings expanded per
    call site."""
    aliases = _module_op_aliases(tree)
    registrations = []
    factory_inner = set()
    for stmt in tree.body:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        packets = _factory_packets(stmt)
        if not packets:
            continue
        for inner in ast.walk(stmt):
            if inner is stmt or not isinstance(inner, ast.FunctionDef):
                continue
            for dec in inner.decorator_list:
                dec_name = _get_decorator_name(dec)
                if not (
                    dec_name
                    and "register_spyre_lowering" in dec_name
                    and isinstance(dec, ast.Call)
                    and dec.args
                ):
                    continue
                raw = _resolve_attr(dec.args[0])
                head, _, overload = (raw or "").partition(".")
                if head not in packets:
                    continue
                factory_inner.add(id(inner))
                ns, index = packets[head]
                for literal in _factory_call_literals(tree, stmt.name, index):
                    op = ".".join(p for p in (ns, literal, overload) if p)
                    registrations.append((inner, op, f"{inner.name}[{literal}]"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or id(node) in factory_inner:
            continue
        for dec in node.decorator_list:
            dec_name = _get_decorator_name(dec)
            if dec_name and "register_spyre_lowering" in dec_name:
                if isinstance(dec, ast.Call) and dec.args:
                    raw = _apply_alias(_resolve_attr(dec.args[0]), aliases)
                    op = _normalize_op_name(raw)
                    if op:
                        registrations.append((node, op, node.name))
    return registrations


def extract_lowerings(filepath):
    """Parse @register_spyre_lowering decorators, expanding factory-registered ones
    (``_register_cmp_lowerings("eq", ...)``) once per call site."""
    nodes = []
    edges = []
    source = Path(filepath).read_text()
    tree = ast.parse(source)
    rel_path = str(filepath)

    for node, op, lowering_name in _lowering_registrations(tree):
        lowering_id = f"lowering::{lowering_name}"
        op_id = f"op::{op}"
        nodes.append(
            {
                "id": lowering_id,
                "label": lowering_name,
                "type": "lowering",
                "source_file": rel_path,
                "line": node.lineno,
            }
        )
        nodes.append({"id": op_id, "label": op, "type": "op"})
        edges.append(
            {
                "source": op_id,
                "target": lowering_id,
                "relationship": "lowered_by",
            }
        )
    return nodes, edges


def _string_constants(filepath):
    """Module-level ``NAME = "string"`` assignments of a file, or {} if absent."""
    path = Path(filepath)
    if not path.exists():
        return {}
    constants = {}
    for stmt in ast.parse(path.read_text()).body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            constants[stmt.targets[0].id] = stmt.value.value
    return constants


def _string_value(node, constants):
    """A string literal, or a name bound to one in ``constants``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _opfunc_node(name, rel_path, line):
    return {
        "id": f"opfunc::{name}",
        "label": name,
        "type": "opfunc",
        "source_file": rel_path,
        "line": line,
    }


def extract_inductor_ops(filepath):
    """Map each ``SpyreOpFuncs`` method (an Inductor op) to the SDSC opfunc
    strings it builds with ``PointwiseOp(...)`` or ``ReductionOp(...)``."""
    nodes = []
    edges = []
    tree = ast.parse(Path(filepath).read_text())
    rel_path = str(filepath)
    constants = _string_constants(Path(filepath).with_name("constants.py"))
    constants.update(_string_constants(filepath))

    for cls in tree.body:
        if not (isinstance(cls, ast.ClassDef) and cls.name == "SpyreOpFuncs"):
            continue
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            iop_id = f"iop::{fn.name}"
            nodes.append(
                {
                    "id": iop_id,
                    "label": f"ops.{fn.name}",
                    "type": "inductor_op",
                    "source_file": rel_path,
                    "line": fn.lineno,
                }
            )
            for call in ast.walk(fn):
                if not (isinstance(call, ast.Call) and call.args):
                    continue
                func = _resolve_attr(call.func) or ""
                if func.split(".")[-1] not in ("PointwiseOp", "ReductionOp"):
                    continue
                opfunc = _string_value(call.args[0], constants)
                if not opfunc:
                    continue
                nodes.append(_opfunc_node(opfunc, rel_path, call.lineno))
                edges.append(
                    {
                        "source": iop_id,
                        "target": f"opfunc::{opfunc}",
                        "relationship": "maps_to",
                    }
                )
    return nodes, edges


def _emits_of(fn, helpers, constants, depth=0, seen=None):
    """(exact Inductor op or opfunc ids, name-inferred Inductor op ids) that
    ``fn`` emits, following calls to module-level helpers up to three deep.

    Exact: ``ops_wrapper(...)`` names, ``ops.<name>(...)`` calls, and any
    ``*_OP`` constant (an SDSC opfunc name) the body uses. Inferred: a
    reference to ``lowering.<name>``, Inductor's lowering of that name, which
    emits the Inductor op of the same name."""
    seen = seen if seen is not None else set()
    seen.add(fn.name)
    exact, inferred = set(), set()
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Attribute):
            ref = _resolve_attr(sub) or ""
            if ref.startswith("lowering.") and ref.count(".") == 1:
                inferred.add(f"iop::{ref[len('lowering.') :]}")
        if isinstance(sub, ast.Name) and sub.id.endswith("_OP") and sub.id in constants:
            exact.add(f"opfunc::{constants[sub.id]}")
        if not isinstance(sub, ast.Call):
            continue
        func = _resolve_attr(sub.func) or ""
        if func.split(".")[-1] == "ops_wrapper" and sub.args:
            arg = sub.args[0]
            name = _string_value(arg, constants)
            if (
                name is None
                and isinstance(arg, ast.Attribute)
                and arg.attr == "__name__"
            ):
                name = (_resolve_attr(arg.value) or "").split(".")[-1]
            if name:
                exact.add(f"iop::{name}")
        elif func.startswith("ops.") and func.count(".") == 1:
            exact.add(f"iop::{func[len('ops.') :]}")
        for kw in sub.keywords:
            if kw.arg == "reduction_type":
                opfunc = _string_value(kw.value, constants)
                if opfunc:
                    exact.add(f"opfunc::{opfunc}")
        if (
            depth < 3
            and isinstance(sub.func, ast.Name)
            and sub.func.id in helpers
            and sub.func.id not in seen
        ):
            e, i = _emits_of(helpers[sub.func.id], helpers, constants, depth + 1, seen)
            exact |= e
            inferred |= i
    return exact, inferred


def extract_lowering_emits(filepath):
    """What each Spyre lowering emits (see ``_emits_of``), and which other Spyre
    lowerings it calls. An edge to an Inductor op that ``SpyreOpFuncs`` does not
    define is dropped when the graph is assembled."""
    nodes = []
    edges = []
    tree = ast.parse(Path(filepath).read_text())
    rel_path = str(filepath)
    constants = _string_constants(Path(filepath).with_name("constants.py"))
    constants.update(_string_constants(filepath))

    registrations = _lowering_registrations(tree)
    ids_by_function = {}
    for fn, _op, lowering_name in registrations:
        ids_by_function.setdefault(fn.name, set()).add(f"lowering::{lowering_name}")
    # Module-level functions that are not themselves registered lowerings.
    helpers = {
        f.name: f
        for f in tree.body
        if isinstance(f, ast.FunctionDef) and f.name not in ids_by_function
    }

    for fn, _op, lowering_name in registrations:
        lowering_id = f"lowering::{lowering_name}"
        exact, inferred = _emits_of(fn, helpers, constants)
        # A factory-registered lowering (``_tensor[eq]``) names its op only as the
        # literal passed to the factory: the Inductor op of that name (inferred).
        if lowering_name.endswith("]") and "[" in lowering_name:
            inferred.add(f"iop::{lowering_name[lowering_name.rindex('[') + 1 : -1]}")
        delegates = set()
        for sub in ast.walk(fn):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in ids_by_function
                and sub.func.id != fn.name
            ):
                delegates |= ids_by_function[sub.func.id]
        for target in sorted(exact):
            if target.startswith("opfunc::"):
                nodes.append(
                    _opfunc_node(target[len("opfunc::") :], rel_path, fn.lineno)
                )
            edges.append(
                {"source": lowering_id, "target": target, "relationship": "emits"}
            )
        for target in sorted(inferred - exact):
            edges.append(
                {
                    "source": lowering_id,
                    "target": target,
                    "relationship": "emits",
                    "inferred": True,
                }
            )
        for target in sorted(delegates - {lowering_id}):
            edges.append(
                {
                    "source": lowering_id,
                    "target": target,
                    "relationship": "delegates_to",
                }
            )
    return nodes, edges


def _aten_refs(node, aliases):
    """Normalized ``aten.*`` op references inside ``node``, longest form only
    (``aten.mm.default`` rather than both it and ``aten.mm``)."""
    refs = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Attribute):
            continue
        name = _normalize_op_name(_apply_alias(_resolve_attr(sub), aliases))
        if name and name.startswith("aten.") and name.count(".") in (1, 2):
            refs.add(name)
    return {r for r in refs if not any(o != r and o.startswith(r + ".") for o in refs)}


def extract_fx_rewrites(filepath):
    """Aten ops each FX graph rewrite matches or creates.

    Pattern passes are ``@register_graph_pattern(..., pass_dict=P)`` functions;
    their ops are attributed to ``P.apply``. Every other top-level function is
    attributed to itself. Edges whose pass is not an FX pass function are
    dropped when the graph is assembled."""
    nodes = []
    edges = []
    tree = ast.parse(Path(filepath).read_text())
    aliases = _module_op_aliases(tree)
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        pass_name = fn.name
        for dec in fn.decorator_list:
            if isinstance(dec, ast.Call) and (_get_decorator_name(dec) or "").endswith(
                "register_graph_pattern"
            ):
                for kw in dec.keywords:
                    if kw.arg == "pass_dict" and isinstance(kw.value, ast.Name):
                        pass_name = f"{kw.value.id}.apply"
        for op in sorted(_aten_refs(fn, aliases)):
            nodes.append(
                {"id": f"op::{op}", "label": op, "type": "op", "_rewrite_only": True}
            )
            edges.append(
                {
                    "source": f"op::{op}",
                    "target": f"passfn::{pass_name}",
                    "relationship": "rewritten_by",
                }
            )
    return nodes, edges


def _link_lowering_paths(nodes_by_id, edges):
    """Edges that join the op-level chains: an op packet to its overloads, and
    an aten op with no Spyre path to the Inductor op of the same name, which
    Inductor's default lowering emits (inferred by name)."""
    op_ids = {i for i in nodes_by_id if i.startswith("op::")}
    added = []
    for oid in sorted(op_ids):
        parts = oid[len("op::") :].split(".")
        if len(parts) == 3 and f"op::{parts[0]}.{parts[1]}" in op_ids:
            added.append(
                {
                    "source": f"op::{parts[0]}.{parts[1]}",
                    "target": oid,
                    "relationship": "has_overload",
                }
            )

    handled = {
        e["source"]
        for e in edges
        if e["relationship"] in ("lowered_by", "decomposed_by", "falls_back_to")
    }

    def _family(oid):
        parts = oid[len("op::") :].split(".")
        packet = f"op::{parts[0]}.{parts[1]}" if len(parts) >= 2 else oid
        return {i for i in op_ids if i == packet or i.startswith(packet + ".")}

    done = set()
    for oid in sorted(op_ids):
        parts = oid[len("op::") :].split(".")
        if parts[0] != "aten" or len(parts) < 2:
            continue
        family = _family(oid)
        key = min(family)
        if family & handled or key in done:
            continue
        iop = f"iop::{parts[1]}"
        if iop not in nodes_by_id:
            continue
        done.add(key)
        # Attach once per family, to the packet when it exists.
        source = f"op::aten.{parts[1]}"
        if source not in op_ids:
            source = min(family)
        added.append(
            {
                "source": source,
                "target": iop,
                "relationship": "inductor_default",
                "inferred": True,
            }
        )
    return added


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
            # A bare name (``fallback_ops.append(op)`` inside
            # ``register_fallback``) is a loop variable, not an op reference.
            if (
                func == "fallback_ops.append"
                and call.args
                and not isinstance(call.args[0], ast.Name)
            ):
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
    list_bindings = _module_list_bindings(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _resolve_attr(node.func)
            if func_name in _EAGER_REGISTRARS and node.args:
                inplace = func_name == "register_inplace_kernels"
                ops = _extract_list_ops(node.args[0], list_bindings)
                for op in ops:
                    # ``register_inplace_kernels`` consumes the functional list
                    # and registers the ``foo_`` sibling of each entry, so the
                    # graph node is the in-place name, not the one in the list.
                    if inplace:
                        op = _inplace_name(op)
                        if op is None:
                            continue
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

        # ``torch.library.register_kernel("ns::op", ["spyre"])``, used either
        # as a decorator or called directly. Only Spyre-device registrations
        # are eager kernels; ``["cpu"]`` registrations (the CPU reference
        # kernels of custom ops) are not.
        if isinstance(node, ast.Call):
            func_name = _resolve_attr(node.func)
            if (
                func_name
                and func_name.split(".")[-1] == "register_kernel"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and _names_spyre_device(node.args[1])
            ):
                op_name = _normalize_op_name(node.args[0].value)
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
                nodes.append({"id": op_id, "label": op_name, "type": "op"})
                edges.append(
                    {
                        "source": op_id,
                        "target": eager_id,
                        "relationship": "eager_via",
                    }
                )

    return nodes, edges


def _names_spyre_device(arg):
    """Whether a ``register_kernel`` device argument includes ``"spyre"``."""
    values = arg.elts if isinstance(arg, (ast.List, ast.Tuple)) else [arg]
    return any(
        isinstance(v, ast.Constant)
        and v.value == "spyre"
        or isinstance(v, ast.Name)
        and v.id == "DEVICE_NAME"
        for v in values
    )


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


def _dotted_module_name(filepath):
    """Dotted module name for a file under the ``torch_spyre`` package."""
    parts = list(Path(filepath).with_suffix("").parts)
    start = len(parts) - 1 - parts[::-1].index("torch_spyre")
    parts = parts[start:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def extract_config(filepath):
    """Extract the environment variables a module reads, as ``reads_env`` edges."""
    nodes = []
    edges = []
    source = Path(filepath).read_text()
    tree = ast.parse(source)
    rel_path = str(filepath)

    module_id = f"module::{_dotted_module_name(filepath)}"

    str_constants = {}
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            str_constants[stmt.targets[0].id] = stmt.value.value

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
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            env_name = first_arg.value
        elif isinstance(first_arg, ast.Name) and first_arg.id in str_constants:
            env_name = str_constants[first_arg.id]
        else:
            continue
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

        if mod_name == "torch_spyre._version":
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

    def _is_dataclass(cls):
        return any(
            _get_decorator_name(d) in ("dataclass", "dataclasses.dataclass")
            for d in cls.decorator_list
        )

    local_dataclasses = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and _is_dataclass(n)
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or not _is_dataclass(node):
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
            if not (isinstance(item, ast.AnnAssign) and item.annotation):
                continue
            # Collect every type named in the annotation, so container and
            # optional fields (``list[SDSCArgs]``, ``DebugHandle | None``)
            # count as well as bare ones.
            referenced = {
                _resolve_attr(sub).split(".")[-1]
                for sub in ast.walk(item.annotation)
                if isinstance(sub, (ast.Name, ast.Attribute)) and _resolve_attr(sub)
            }
            for ann in sorted(referenced & (known_types | local_dataclasses)):
                if ann != node.name:
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
        (extract_eager_kernels, root / "_inductor" / "customops.py"),
        (extract_passes, root / "_inductor" / "passes.py"),
        (extract_inductor_ops, root / "_inductor" / "spyre_kernel.py"),
        (extract_lowering_emits, root / "_inductor" / "lowering.py"),
        (extract_fx_rewrites, root / "_inductor" / "temp_passes.py"),
        (extract_fx_rewrites, root / "_inductor" / "passes.py"),
        (extract_fx_rewrites, root / "_inductor" / "materialize_hinted_views.py"),
        (extract_fx_rewrites, root / "_inductor" / "propagate_hints.py"),
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
        root / "_inductor" / "hbm_pool_planning.py",
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

    # config.py first, so its line wins the dedup for a variable read elsewhere too.
    config_path = root / "_inductor" / "config.py"
    env_files = [config_path] + sorted(
        p
        for p in root.rglob("*.py")
        if p != config_path and "__pycache__" not in p.parts
    )
    for env_file in env_files:
        if env_file.exists():
            n, e = _run_file_extractor(extract_config, env_file, repo_root)
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

    # An FX rewrite edge only counts for a pass in an FX graph pipeline.
    fx_passes = {
        e["target"]
        for e in valid_edges
        if e["relationship"] == "contains_pass"
        and e["source"] in ("pass::CustomPrePasses", "pass::CustomPostPasses")
    }
    valid_edges = [
        e
        for e in valid_edges
        if e["relationship"] != "rewritten_by" or e["target"] in fx_passes
    ]
    # Drop op nodes that only a discarded rewrite edge introduced.
    connected = {e["source"] for e in valid_edges} | {e["target"] for e in valid_edges}
    for nid in [i for i, n in seen.items() if n.get("_rewrite_only")]:
        if nid not in connected:
            del seen[nid]
    for n in seen.values():
        n.pop("_rewrite_only", None)
    valid_edges.extend(_link_lowering_paths(seen, valid_edges))

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
