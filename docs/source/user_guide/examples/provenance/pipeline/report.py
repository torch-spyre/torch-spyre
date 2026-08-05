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
"""
pipeline/report.py  —  renders provenance audit report.

The report is **measurement-only**: every table is computed from the captured
data (captures.capture()) and the Stage-6 bundle read (superdsc.run()). It
contains no hand-written analysis, field semantics, or recommendations — those
are intentionally left out so the report cannot drift from what a given run
actually observed.
"""

from __future__ import annotations

import datetime
from typing import Any

# Field names come from the single source of truth in fields.py (shared with
# captures.py) so the two layers can never drift.
from .fields import FX_FIELDS, IR_FIELDS, source_loc_str

# Matrix glyphs in plain text-class unicode.
TICK = "✓"  # present & non-empty on all relevant nodes/ops (U+2713)
PARTIAL = "◐"  # present on some (shown as n/total) (U+25D0)
CROSS = "✗"  # measured empty/absent, i.e. dropped (U+2717)
DASH = "–"  # n/a: not created yet, or carried only indirectly (en dash U+2013)

# One-line description of each provenance field, used in the per-stage intros.
FIELD_DESC = {
    "stack_trace": "user source `file:line`",
    "nn_module_stack": "owning `nn.Module` path",
    "source_fn_stack": "source fn/op that produced it",
    "original_aten": "ATen op it lowered from",
    "from_node": "producing pass/transform chain",
    "origins": "FX nodes that lowered into this buffer",
    "origin_node": "single representative FX node (nullable)",
    "traceback": "IR-node creation-site traceback",
    "get_stack_traces": "source lines derived from `origins`",
    "debug_handle": "source-to-kernel record (id, source, aten_op, ir_chain, fused_from)",
}


def _fields_line(names: list[str]) -> str:
    """A concise 'Tracked: `f` (desc), …' line for a stage's intro."""
    parts = [f"`{n}` ({FIELD_DESC[n]})" for n in names if n in FIELD_DESC]
    return "Tracked fields: " + ", ".join(parts) + "."


# --- Provenance carry model -------------------------------------------------
# The matrix distinguishes four states per (field, stage): the field is
# populated in its OWN slot here (✓/◐), it is CARRIED here inside another
# field (–), it does not EXIST yet at this stage (–), or it is genuinely
# DROPPED (✗). "Carried" is not hardcoded per cell -- it is derived from what
# each *carrier* field actually retains, combined with whether that carrier is
# measured present at the stage. Extending to a future stage (the #2577
# profiler-event column, the #2578 sidecar column -- both keyed by the
# debug_handle id) or a future carrier is a one-line change to STAGES / CARRIES
# below; the cell logic in `_matrix_cell` is then automatic.
#
# `origins` holds the actual fx.Nodes, so every FX-meta field is recoverable
# from it at the IR stages.
_ORIGINS_CARRIES = frozenset(FX_FIELDS)
# `debug_handle` keeps only what build_debug_handle copies: `source`
# (<- stack_trace / get_stack_traces), `aten_op` (<- original_aten) and
# `ir_chain` (<- origins / origin_node). It does NOT retain nn_module_stack,
# source_fn_stack, from_node or traceback -- those are genuinely dropped at the
# OpSpec boundary, which is exactly what the audit must show.
#
# This set is derived BY HAND from build_debug_handle in #2945
# (torch_spyre/_inductor/provenance.py); it is not measured from the run. If that
# function changes which fields it copies into the handle, update this set to
# match -- otherwise the carried/dropped (– / ✗) cells in the OpSpec and SuperDSC
# columns will be wrong while still looking authoritative.
_DEBUG_HANDLE_CARRIES = frozenset(
    {"stack_trace", "get_stack_traces", "original_aten", "origins", "origin_node"}
)
# carrier field -> the set of fields whose provenance it subsumes downstream.
CARRIES: dict[str, frozenset[str]] = {
    "origins": _ORIGINS_CARRIES,
    "debug_handle": _DEBUG_HANDLE_CARRIES,
}

# Column index at which each field first exists. Columns *before* it read –
# (not created yet); columns *at or after* it with no own slot and no carrier
# read ✗ (dropped). Indices match the STAGES order built in `render`.
CREATED_AT: dict[str, int] = {
    "stack_trace": 0,
    "nn_module_stack": 0,
    "source_fn_stack": 0,
    "original_aten": 1,  # generated in the grad/lowering transition
    "from_node": 1,
    "origins": 2,  # created when fx nodes lower to a ComputedBuffer
    "origin_node": 2,
    "traceback": 2,
    "get_stack_traces": 2,
    "debug_handle": 4,  # introduced on OpSpec by Phase 2a (#2575)
}


def _plural(n: int, word: str) -> str:
    """Naive pluralizer for count headers: ``1 kernel`` / ``2 kernels``.

    The report runs on arbitrary models, so every "N <thing>" header must read
    correctly at ``n == 1`` (e.g. a single-kernel model), not just for the
    SimpleMLP snapshot. Regular ``+s`` is sufficient for the nouns used here.
    """
    return word if n == 1 else f"{word}s"


# --------------------------------------------------------------------------
# Presence computation (purely from captured data)
# --------------------------------------------------------------------------
def _fx_present(node: dict, field: str) -> bool:
    rec = node.get("fields", {}).get(field)
    return isinstance(rec, dict) and rec.get("nonempty", False)


def _fx_symbol(nodes: list[dict], field: str) -> str:
    total = len(nodes)
    if total == 0:
        return CROSS
    present = sum(1 for n in nodes if _fx_present(n, field))
    if present == 0:
        return CROSS
    return TICK if present == total else f"{PARTIAL} {present}/{total}"


def _ir_value(op: dict, field: str):
    return op.get("fields", {}).get(field)


def _ir_present(op: dict, field: str) -> bool:
    v = _ir_value(op, field)
    if field == "origins":
        return bool(v)
    if field == "get_stack_traces":
        return isinstance(v, dict) and v.get("nonempty", False)
    # origin_node / traceback: populated == truthy (treat "" / None as absent).
    return bool(v)


def _ir_symbol(ops: list[dict], field: str) -> str:
    total = len(ops)
    if total == 0:
        return CROSS
    present = sum(1 for o in ops if _ir_present(o, field))
    if present == 0:
        return CROSS
    return TICK if present == total else f"{PARTIAL} {present}/{total}"


def _ir_attr_exists(ops: list[dict], field: str) -> bool:
    """Does the provenance attribute even exist on these IR objects? Uses the
    capture's `attr_exists` (hasattr) record."""
    return any(o.get("attr_exists", {}).get(field, True) for o in ops)


def _ir_cell(ops: list[dict], field: str) -> str:
    """Matrix cell for an IR field at one stage: – if the attribute isn't on
    the object here at all; otherwise ✓/◐/✗ from the populated values."""
    if not ops:
        return CROSS
    if not _ir_attr_exists(ops, field):
        return DASH
    return _ir_symbol(ops, field)


def _sdsc_symbol(bundles: dict, field: str) -> str:
    """SuperDSC matrix cell for one provenance field, counted across all
    scanned ``sdsc_*.json`` files (mirrors ``_ir_symbol``: ✓ / ◐ x/n / ✗).
    Parse-error files have no ``provenance_fields`` and are excluded."""
    files = [
        fobj
        for k in bundles.get("kernels", [])
        for fobj in k.get("sdsc_files", [])
        if "provenance_fields" in fobj
    ]
    total = len(files)
    if total == 0:
        return CROSS
    present = sum(1 for fobj in files if field in fobj["provenance_fields"])
    if present == 0:
        return CROSS
    return TICK if present == total else f"{PARTIAL} {present}/{total}"


def _opspec_symbol(ops: list[dict], field: str, opspec_fields: list[str]) -> str:
    """OpSpec matrix cell. ✗ if the dataclass declares no such field (the
    genuine drop); otherwise ✓ / ◐ x,total / ✗ from per-instance population,
    so a field declared but not populated on every op shows partial."""
    if field not in opspec_fields:
        return CROSS
    total = len(ops)
    if total == 0:
        return CROSS
    present = sum(1 for o in ops if o.get("opspec_present", {}).get(field))
    if present == 0:
        return CROSS
    return TICK if present == total else f"{PARTIAL} {present}/{total}"


def _dh_fused_from_str(dh: Any) -> str:
    """Render a handle's ``fused_from`` lineage as its constituent ATen ops
    (MLIR ``FusedLoc``-style). ``—`` when the op is not a fusion."""
    if not isinstance(dh, dict):
        return "—"
    aten = [
        c.get("aten_op")
        for c in (dh.get("fused_from") or [])
        if isinstance(c, dict) and c.get("aten_op")
    ]
    return ", ".join(f"`{a}`" for a in aten) if aten else "—"


def _sdsc_debug_handle_symbol(bundles: dict) -> str:
    """SuperDSC matrix cell for ``debug_handle``: counted across all scanned
    ``sdsc_*.json`` files by non-null handle (✓ / ◐ x/n / ✗)."""
    files = [
        f
        for k in bundles.get("kernels", [])
        for f in k.get("sdsc_files", [])
        if "debug_handle_present" in f
    ]
    total = len(files)
    if total == 0:
        return CROSS
    present = sum(1 for f in files if f.get("debug_handle_nonnull"))
    if present == 0:
        return CROSS
    return TICK if present == total else f"{PARTIAL} {present}/{total}"


def _is_present(symbol: str | None) -> bool:
    """A cell counts as populated (its own slot carries content) when it is ✓
    or a partial ◐ n/total -- not ✗, –, or a missing (None) slot."""
    return symbol == TICK or (isinstance(symbol, str) and symbol.startswith(PARTIAL))


def _matrix_cell(
    field: str, col_idx: int, own: str | None, active_carriers: list[str]
) -> str:
    """Resolve one Stage × Field cell from measured data + the carry model.

    ``own`` is the field's measured OWN-slot symbol at this stage (or None when
    the stage's object has no such slot); ``active_carriers`` are the carrier
    fields measured present at this stage. Precedence: own slot populated →
    carried in a present carrier → not created yet → dropped.
    """
    if _is_present(own):
        return own  # populated in its own slot here
    for carrier in active_carriers:
        if field in CARRIES.get(carrier, ()):
            return DASH  # not a slot here, but rides in a present carrier
    if col_idx < CREATED_AT.get(field, 0):
        return DASH  # this stage is upstream of where the field is created
    return CROSS  # reachable at/after creation, not carried -> genuinely dropped


def _fx_type(node: dict, field: str) -> str:
    rec = node.get("fields", {}).get(field)
    if not (isinstance(rec, dict) and rec.get("nonempty", False)):
        return CROSS
    return f"`{rec.get('type', '?')}`"


def _mermaid_id(prefix: str, name: Any) -> str:
    """A Mermaid-safe node id (alphanumerics/underscore only), namespaced by
    stage so the same name in two stages (e.g. ``op0``) never collides."""
    safe = "".join(c if c.isalnum() else "_" for c in str(name))
    return f"{prefix}_{safe}"


def _mermaid_label(text: Any) -> str:
    """Sanitize a Mermaid node label: drop the ``<...>`` and quotes Mermaid
    would choke on; keep it short."""
    s = str(text).replace('"', "'").replace("<", "").replace(">", "")
    return s[:48]


def _short_target(target: Any) -> str:
    """A compact op kind from an FX target repr (best-effort, generic)."""
    t = str(target)
    if "built-in function " in t:
        return t.split("built-in function ", 1)[1].rstrip(">")
    if "built-in method " in t:
        return t.split("built-in method ", 1)[1].split(" ", 1)[0]
    return t.rsplit(".", 1)[-1] if "." in t else t


def _op_sources(
    opspec_ops: list[dict], post_nodes: list[dict]
) -> tuple[dict[int, set[str]], dict[str, int]]:
    """For each OpSpec op, the set of source lines it belongs to.

    An op's source is the code-text ``stack_trace`` source(s) carried by its own
    origin FX nodes when present; otherwise it is inherited from the ops whose
    buffers it directly consumes (walked over the captured FX ``input_nodes``
    edges, crossing op boundaries). *All* source-bearing producers contribute --
    a genuinely multi-source op (e.g. a residual add of two layers) lists every
    contributing source, not just one. Uses the FX code-text source throughout
    so it keys identically to the pre-/post-grad layers (the ``debug_handle``
    ``file:line`` is a different string and would not unify).
    Returns (op_index -> sources, fx_name -> owning op_index).
    """
    fx_src: dict[str, str | None] = {}
    fx_inputs: dict[str, list[str]] = {}
    for n in post_nodes:
        st = n.get("fields", {}).get("stack_trace")
        fx_src[n["name"]] = st.get("source_line") if isinstance(st, dict) else None
        fx_inputs[n["name"]] = n.get("input_nodes", [])

    owner: dict[str, int] = {}
    for i, o in enumerate(opspec_ops):
        for x in o.get("fields", {}).get("origins") or []:
            owner.setdefault(x["name"], i)

    producers: dict[int, set[int]] = {}
    own: list[set[str]] = []
    for i, o in enumerate(opspec_ops):
        deps: set[int] = set()
        srcs: set[str] = set()
        for x in o.get("fields", {}).get("origins") or []:
            name = x["name"]
            if fx_src.get(name):
                srcs.add(fx_src[name])
            for inp in fx_inputs.get(name, []):
                j = owner.get(inp)
                if j is not None and j != i:
                    deps.add(j)
        producers[i] = deps
        own.append(srcs)

    memo: dict[int, set[str]] = {}

    def resolve(i: int, stack: frozenset[int]) -> set[str]:
        if own[i]:
            return own[i]
        if i in memo:
            return memo[i]
        if i in stack:  # cycle guard (FX is a DAG, but be safe)
            return set()
        out: set[str] = set()
        for j in producers[i]:
            out |= resolve(j, stack | {i})
        memo[i] = out
        return out

    return {i: resolve(i, frozenset()) for i in range(len(opspec_ops))}, owner


def _lineage_section(
    opspec_ops: list[dict],
    bundles: dict[str, Any],
    pre_nodes: list[dict],
    post_nodes: list[dict],
    prepass_ops: list[dict],
    postpass_ops: list[dict],
) -> list[str]:
    """Source→kernel lineage as a Mermaid graph, built entirely from captured
    data. Six stage layers left-to-right (**Source → FX pre-grad → FX post-grad
    → LoopLevelIR pre-pass → LoopLevelIR post-pass → OpSpec → SuperDSC**); a
    fan-out is a decomposition, a fan-in is a fusion. Edges are real measured
    transitions: source lines from ``stack_trace``; FX decomposition attributed
    by source; ``origins`` for FX→LoopLevelIR; op name for the pre/post-pass and
    OpSpec hand-offs; the ``debug_handle`` id for OpSpec→kernel. A source-less op
    attaches to every source-bearing producer it consumes (multi-source). Node/
    IR transformation only -- field survival is the Stage x Field matrix's job.
    """

    def _src_of(n: dict) -> str | None:
        st = n.get("fields", {}).get("stack_trace")
        return st.get("source_line") if isinstance(st, dict) else None

    op_src, owner = _op_sources(opspec_ops, post_nodes)

    def _node_sources(name: str) -> set[str]:
        own = next(
            (_src_of(n) for n in post_nodes if n["name"] == name and _src_of(n)), None
        )
        if own:
            return {own}
        i = owner.get(name)
        return op_src.get(i, set()) if i is not None else set()

    # Nodes per stage (id -> label, insertion-ordered) and a de-duped edge list.
    nodes: dict[str, dict[str, str]] = {
        k: {} for k in ("src", "pg", "post", "lpre", "lpost", "ops", "sd")
    }
    edges: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()

    def add(stage: str, prefix: str, key: Any, label: str) -> str:
        nid = _mermaid_id(prefix, key)
        nodes[stage].setdefault(nid, _mermaid_label(label))
        return nid

    def link(a: str, b: str) -> None:
        if (a, b) not in seen_edges:
            seen_edges.add((a, b))
            edges.append((a, b))

    # Source + FX pre-grad (grouped by source line).
    all_sources: set[str] = {s for n in pre_nodes if (s := _src_of(n))}
    for i in range(len(opspec_ops)):
        all_sources |= op_src.get(i, set())
    for s in sorted(all_sources):
        add("src", "src", s, s)
    pg_by_source: dict[str, str] = {}
    for n in pre_nodes:
        s = _src_of(n)
        if not s:
            continue
        kind = _short_target(n.get("target", ""))
        nid = add("pg", "pg", n["name"], f"{n['name']} · {kind}" if kind else n["name"])
        link(_mermaid_id("src", s), nid)
        # TODO(#<issue>): one pre-grad node per source line only. When several
        # pre-grad nodes share a source line, the post-grad decomposition edges
        # below attach to just this first anchor and the rest are dropped (fine
        # for SimpleMLP's 1-node-per-line graph, wrong for larger models). Fix =
        # store a list of ids per source line and link post-grad nodes to all.
        pg_by_source.setdefault(s, nid)

    # FX post-grad, linked up to its source(s)' pre-grad node (decomposition).
    for n in post_nodes:
        nid = add("post", "post", n["name"], n["name"])
        for s in _node_sources(n["name"]):
            pg = pg_by_source.get(s)
            if pg:
                link(pg, nid)

    # FX post-grad → LoopLevelIR (pre-pass), via origins.
    for op in prepass_ops:
        nid = add("lpre", "lpre", op["name"], op["name"])
        for x in op.get("fields", {}).get("origins") or []:
            link(_mermaid_id("post", x["name"]), nid)

    # LoopLevelIR pre-pass → post-pass (by op name; a post-only node = inserted
    # by a pass, e.g. restickify, and simply has no incoming pre-pass edge).
    pre_names = {op["name"] for op in prepass_ops}
    for op in postpass_ops:
        nid = add("lpost", "lpost", op["name"], op["name"])
        if op["name"] in pre_names:
            link(_mermaid_id("lpre", op["name"]), nid)

    # LoopLevelIR post-pass → OpSpec (by buffer name).
    post_names = {op["name"] for op in postpass_ops}
    for o in opspec_ops:
        buf = o.get("name", "?")
        nid = add("ops", "ops", buf, f"{buf} · {o.get('op', '?')}")
        if buf in post_names:
            link(_mermaid_id("lpost", buf), nid)

    # OpSpec → SuperDSC kernel (fan-in = fusion), via the handle id.
    id_to_kernel: dict[Any, str] = {}
    for b in bundles.get("kernels", []):
        for f in b.get("sdsc_files", []):
            dh = f.get("debug_handle")
            if isinstance(dh, dict) and dh.get("id") is not None:
                id_to_kernel[dh["id"]] = b["kernel_name"]
    for o in opspec_ops:
        kern = id_to_kernel.get((o.get("debug_handle") or {}).get("id"))
        if kern:
            link(_mermaid_id("ops", o.get("name", "?")), add("sd", "sd", kern, kern))

    # Emit the Mermaid graph: one subgraph per non-empty stage, then edges.
    titles = [
        ("src", "Source"),
        ("pg", "FX pre-grad"),
        ("post", "FX post-grad"),
        ("lpre", "LoopLevelIR pre-pass"),
        ("lpost", "LoopLevelIR post-pass"),
        ("ops", "OpSpec"),
        ("sd", "SuperDSC"),
    ]
    m = ["```mermaid", "flowchart LR"]
    for stage, title in titles:
        if not nodes[stage]:
            continue
        m.append(f'  subgraph {stage}_sg["{title}"]')
        for nid, label in nodes[stage].items():
            m.append(f'    {nid}["{label}"]')
        m.append("  end")
    m += [f"  {a} --> {b}" for a, b in edges]
    m.append("```")

    return [
        "## Source → Kernel Lineage",
        "",
        "How each source line flows through the pipeline — **Source → FX "
        "pre-grad → FX post-grad → LoopLevelIR pre-pass → LoopLevelIR post-pass "
        "→ OpSpec → SuperDSC**. A **fan-out** is a decomposition (e.g. `linear` "
        "→ `permute` + `mm` + `add`); a **fan-in** is a fusion (several OpSpec "
        "ops → one kernel). An op with no source of its own attaches to every "
        "source-bearing producer whose buffer it consumes (multi-source). "
        "Node/IR transformation only — field survival is the matrix below.",
        "",
        *m,
        "",
    ]


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------
def render(
    capture: dict[str, Any], bundles: dict[str, Any], model_name: str = "SimpleMLP"
) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pre = capture["stage2_pre_grad"]["nodes"]
    post = capture["stage2_post_grad"]["nodes"]
    prepass_ops = capture["stage3_looplevel_prepass"]["operations"]
    postpass_ops = capture["stage4_looplevel_postpass"]["operations"]
    opspec_ops = capture["stage5_opspec"]["ops"]
    opspec_fields = capture["stage5_opspec"]["opspec_fields"] or []
    kernels = capture["stage6_kernels"]["kernels"]

    L: list[str] = []

    # Header --------------------------------------------------------------
    L += [
        f"# Example Audit: `{model_name}` — Source-to-Kernel Provenance",
        "",
        f"> Generated: {now} &nbsp;|&nbsp; Issue: "
        "[torch-spyre#2574](https://github.com/torch-spyre/torch-spyre/issues/2574)",
        "",
        "One worked example of the provenance audit (see the README for how to "
        f"run it): `{model_name}` traced in-process through a single "
        "cache-defeated `torch.compile`. Measurement-only — every table and the "
        "lineage graph below are computed from the captured compile-path objects.",
        "",
        "This is a committed **example snapshot**: the `Generated` timestamp and "
        "the `debug_handle` ids are specific to this run and machine (the id is a "
        "content hash of the source location, which includes an absolute path), "
        "so they change when regenerated — run `audit.py` to reproduce it for "
        "your own setup.",
        "",
    ]

    # Source → Kernel lineage graph first: the end-to-end transformation story.
    L += _lineage_section(opspec_ops, bundles, pre, post, prepass_ops, postpass_ops)

    # Stage x field matrix ------------------------------------------------
    # Each column is measured on the object that stage produces. The Layer column
    # groups fields by the object they live on: FX = FX-node meta, IR =
    # ComputedBuffer/LoopLevelIR attributes. The two IR columns are the same
    # LoopLevelIR before ("pre-pass", the lowered IR) and after ("post-pass") the
    # Spyre pre-scheduling passes, which may insert restickify buffers and null
    # origin_node on them. (Issue #2574: "Inductor passes" -> "LoopLevelIR".)
    L += [
        "## Stage × Field Matrix",
        "",
        f"{TICK} present & non-empty on **all** instances &nbsp; "
        f"{PARTIAL} on some (n/total) &nbsp; "
        f"{CROSS} reachable here but measured **empty/absent** (dropped) &nbsp; "
        f"{DASH} not created yet here, **or** carried indirectly inside another "
        "field (see that field's row).",
        "",
        "Every column tests **population** (the field exists *and* carries "
        'non-empty content; `0` counts as content, `None`/`[]`/`{}`/`""` do '
        "not). Own-slot cells are measured directly; whether an absent field "
        "reads *carried indirectly* (`–`) or *dropped* (`✗`) at the OpSpec / "
        "SuperDSC columns is **derived** from what `debug_handle` retains, not "
        "measured.",
        "",
        "The **Layer** column marks whether a field lives on the FX node (`FX`), "
        "the IR `ComputedBuffer` (`IR`), or the `debug_handle` (`Spyre`). The two "
        "IR columns are the same "
        "LoopLevelIR before and after the Spyre pre-scheduling passes: "
        "**LoopLevelIR (pre-pass)** is the lowered IR entering them, "
        "**LoopLevelIR (post-pass)** is after they mutate it in place. These "
        "map to issue #2574's "
        '"Inductor passes" → "LoopLevelIR".',
        "",
    ]

    # Ordered matrix columns. Each stage's own_fn(field) returns the field's
    # measured OWN-slot symbol on the object that stage produces, or None when
    # that object has no such slot. Appending a stage here (e.g. a "Profiler
    # event" column for #2577, or a "Sidecar" column for #2578 -- both keyed by
    # the debug_handle id) makes every cell (own / carried / not-yet / dropped)
    # resolve automatically via `_matrix_cell`; no per-cell edits.
    def _fx_own(nodes):
        return lambda f: _fx_symbol(nodes, f) if f in FX_FIELDS else None

    def _ir_own(ops):
        return lambda f: _ir_cell(ops, f) if f in IR_FIELDS else None

    def _opspec_own(f: str) -> str | None:
        if f in IR_FIELDS or f == "debug_handle":
            return _opspec_symbol(opspec_ops, f, opspec_fields)
        return None

    def _sdsc_own(f: str) -> str | None:
        if f == "debug_handle":
            return _sdsc_debug_handle_symbol(bundles)
        return _sdsc_symbol(bundles, f) if f in IR_FIELDS else None

    stages: list[tuple[str, Any]] = [
        ("FX Graph (pre-grad)", _fx_own(pre)),
        ("FX Graph (post-grad)", _fx_own(post)),
        ("LoopLevelIR (pre-pass)", _ir_own(prepass_ops)),
        ("LoopLevelIR (post-pass)", _ir_own(postpass_ops)),
        ("OpSpec", _opspec_own),
        ("SuperDSC JSON", _sdsc_own),
    ]
    rows = (
        [("FX", f) for f in FX_FIELDS]
        + [("IR", f) for f in IR_FIELDS]
        + [("Spyre", "debug_handle")]
    )
    # A carrier field is "active" at a column where its own slot is populated;
    # only then can it carry other fields' provenance there.
    active_by_col = [
        [c for c in CARRIES if _is_present(own_fn(c))] for _, own_fn in stages
    ]

    col_names = [name for name, _ in stages]
    L += [
        "| Layer | Field | " + " | ".join(col_names) + " |",
        "| " + " | ".join(["---"] * (len(col_names) + 2)) + " |",
    ]
    for layer, field in rows:
        cells = [
            _matrix_cell(field, i, own_fn(field), active_by_col[i])
            for i, (_, own_fn) in enumerate(stages)
        ]
        L.append(f"| {layer} | `{field}` | " + " | ".join(cells) + " |")
    L.append("")

    # Stage 2 detail (2a pre-grad, 2b post-grad) --------------------------
    for slabel, plabel, transition, nodes in [
        (
            "2a",
            "pre-grad",
            "Dynamo traces the model into an FX graph; each node still carries "
            "its Python-source metadata.",
            pre,
        ),
        (
            "2b",
            "post-grad",
            "AOTAutograd/Inductor lower and decompose the graph (e.g. `linear` "
            "→ `permute` + `mm` + `add`); synthesized nodes keep only part of "
            "the metadata.",
            post,
        ),
    ]:
        meta_keys = sorted({k for n in nodes for k in n.get("all_meta_keys", [])})
        L += [
            f"## Stage {slabel} — FX Graph ({plabel}): {len(nodes)} "
            f"compute {_plural(len(nodes), 'node')}",
            "",
            transition,
            "",
            _fields_line(FX_FIELDS),
            "",
            f"All observed `node.meta` keys: `{meta_keys}`.",
            "",
            "| Node | target | "
            + " | ".join(f"`{f}`" for f in FX_FIELDS)
            + " | source line |",
            "| --- | --- | " + " | ".join("---" for _ in FX_FIELDS) + " | --- |",
        ]
        for n in nodes:
            st = n.get("fields", {}).get("stack_trace")
            src = st.get("source_line") if isinstance(st, dict) else None
            cells = " | ".join(_fx_type(n, f) for f in FX_FIELDS)
            L.append(
                f"| `{n['name']}` | `{n.get('target', '')}` | {cells} "
                f"| {('`' + src + '`') if src else '—'} |"
            )
        L.append("")

    # Stages 3 & 4 detail (pre-pass / post-pass LoopLevelIR) --------------
    # Two tables because the pre-scheduling passes can change the op set.
    for stage_label, transition, ops_list in [
        (
            "Stage 3 — LoopLevelIR (pre-pass)",
            "FX nodes lower into LoopLevelIR `ComputedBuffer`s entering the "
            "Spyre pre-scheduling passes.",
            prepass_ops,
        ),
        (
            "Stage 4 — LoopLevelIR (post-pass)",
            "The same IR after the pre-scheduling passes mutate it in place.",
            postpass_ops,
        ),
    ]:
        ir_attrs = sorted({a for o in ops_list for a in o.get("all_attrs", [])})
        L += [
            f"## {stage_label}: {len(ops_list)} {_plural(len(ops_list), 'operation')}",
            "",
            transition,
            "",
            _fields_line(IR_FIELDS),
            "",
            f"All observed `ComputedBuffer` attributes: `{ir_attrs}`.",
            "",
            "| Op | `origins` | `origin_node` | `traceback` | `get_stack_traces` |",
            "| --- | --- | --- | --- | --- |",
        ]
        for o in ops_list:
            fields = o.get("fields", {})
            origins = (
                ", ".join(f"`{x['name']}`" for x in (fields.get("origins") or []))
                or "—"
            )
            onode = fields.get("origin_node") or CROSS
            tb = fields.get("traceback") or CROSS
            gst = fields.get("get_stack_traces") or {}
            gst_cell = TICK if gst.get("nonempty") else CROSS
            L.append(
                f"| `{o['name']}` | {origins} "
                f"| {('`' + onode + '`') if onode != CROSS else CROSS} "
                f"| {tb if tb == CROSS else '`' + str(tb)[:40] + '`'} | {gst_cell} |"
            )
        L.append("")

    # Stage 5 detail ------------------------------------------------------
    total_ops = len(opspec_ops)
    L += [
        f"## Stage 5 — OpSpec: {total_ops} {_plural(total_ops, 'op')}",
        "",
        "Each scheduled `ComputedBuffer` becomes an `OpSpec` (the device op); "
        "`debug_handle` carries source provenance onto it, while the `origins` / "
        "`origin_node` shown here are read from the input buffer (not stored on "
        "`OpSpec`).",
        "",
        _fields_line(["origins", "origin_node", "debug_handle"]),
        "",
        f"`OpSpec` declared fields: `{opspec_fields}`.",
        "",
        "| Spyre op | buffer | `origins` | `origin_node` | `debug_handle` id "
        "| source line |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for o in opspec_ops:
        fields = o.get("fields", {})
        origins = (
            ", ".join(f"`{x['name']}`" for x in (fields.get("origins") or [])) or "—"
        )
        onode = fields.get("origin_node") or CROSS
        dh = o.get("debug_handle") or {}
        dh_id = dh.get("id")
        dh_src = dh.get("source_line")
        L.append(
            f"| `{o.get('op', '?')}` | `{o.get('name', '?')}` | {origins} "
            f"| {('`' + onode + '`') if onode != CROSS else CROSS} "
            f"| {('`' + str(dh_id) + '`') if dh_id is not None else CROSS} "
            f"| {('`' + dh_src + '`') if dh_src else '—'} |"
        )
    L.append("")

    # Stage 6 detail ------------------------------------------------------
    total_files = bundles.get("total_sdsc_files", 0)
    dh_present = bundles.get("debug_handle_files", 0)
    dh_nonnull = bundles.get("debug_handle_nonnull_files", 0)
    dh_keys = sorted(
        {
            k
            for b in bundles.get("kernels", [])
            for f in b.get("sdsc_files", [])
            if isinstance(f.get("debug_handle"), dict)
            for k in f["debug_handle"]
        }
    )
    # The debug_handle counts only tally successfully-parsed files, so the
    # coverage denominator must match: an unparsed sdsc_*.json is "unknown", not
    # "missing a handle". Parse errors never occur for valid compiler output;
    # this just keeps the ratio honest (and flags the drop) if one ever does.
    parsed_files = sum(
        1
        for b in bundles.get("kernels", [])
        for f in b.get("sdsc_files", [])
        if "parse_error" not in f
    )
    unparsed = total_files - parsed_files
    coverage_note = (
        f"`debug_handle_` present in `{dh_present}/{parsed_files}` files, "
        f"non-null in `{dh_nonnull}/{parsed_files}`"
        + (f" ({unparsed} unparsed, excluded)." if unparsed else ".")
    )
    L += [
        f"## Stage 6 — SuperDSC: {total_files} "
        f"`sdsc_*.json` {_plural(total_files, 'file')} "
        f"({len(kernels)} {_plural(len(kernels), 'kernel')})",
        "",
        "Each `OpSpec` is serialized to a `sdsc_*.json` kernel spec; the "
        "`debug_handle` travels with it (JSON key `debug_handle_`), resolving "
        "each kernel back to source.",
        "",
        "Tracked (per `debug_handle_`): `id` (stable content hash), `source` "
        "(`file:line`), `aten_op`, `fused_from` (constituent handles when fused).",
        "",
        coverage_note,
        "",
        f"All observed `debug_handle` keys: `{dh_keys}`.",
        "",
    ]

    # Payoff: kernel → source line, straight from the serialized debug_handle_.
    L += [
        "| Kernel | `sdsc_*.json` | Spyre op | `aten_op` | source line "
        "| `fused_from` | `debug_handle` id |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b in bundles.get("kernels", []):
        kname = b["kernel_name"]
        for f in b.get("sdsc_files", []):
            dh = f.get("debug_handle")
            if not isinstance(dh, dict):
                continue
            op_keys = ", ".join(f"`{k}`" for k in f.get("op_keys", [])) or "—"
            aten = dh.get("aten_op")
            src = source_loc_str(dh.get("source"))
            dh_id = dh.get("id")
            L.append(
                f"| `{kname}` | `{f['file']}` | {op_keys} "
                f"| {('`' + aten + '`') if aten else '—'} "
                f"| {('`' + src + '`') if src else '—'} "
                f"| {_dh_fused_from_str(dh)} "
                f"| {('`' + str(dh_id) + '`') if dh_id is not None else '—'} |"
            )
    L += [
        "",
        "`fused_from` lists the constituent handles' ATen ops (MLIR `FusedLoc`"
        "-style lineage). A `—` source line means the handle resolves only to an "
        "ATen op (its FX node carried no `stack_trace`), not that the handle is "
        "absent.",
        "",
    ]
    bundle_by_name = {b["kernel_name"]: b for b in bundles.get("kernels", [])}
    for k in kernels:
        name = k["kernel_name"]
        ops = [no["name"] for no in k.get("node_origins", [])]
        origins = sorted(
            {
                o["name"]
                for no in k.get("node_origins", [])
                for o in (no.get("fields", {}).get("origins") or [])
            }
        )
        md = k.get("kernel_metadata", {}).get("metadata", "")
        b = bundle_by_name.get(name, {})
        nfiles = len(b.get("sdsc_files", []))
        dh_nn = b.get("debug_handle_nonnull_files", 0)
        L += [
            f"### `{name}`",
            "",
            f"- buffers ({len(ops)}): {', '.join(f'`{o}`' for o in ops) or '—'}",
            f"- fx origins: {', '.join(f'`{o}`' for o in origins) or '—'}",
            f"- kernel metadata: `{md}`",
            f"- `sdsc_*.json` files: {nfiles} &nbsp; `debug_handle_` non-null: "
            f"{dh_nn}/{nfiles}",
            "",
        ]

    return "\n".join(L)
