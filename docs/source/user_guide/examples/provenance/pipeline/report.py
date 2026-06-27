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
pipeline/report.py  —  renders the issue #2574 provenance audit report.

The report is **measurement-only**: every table is computed from the captured
data (captures.capture()) and the Stage-6 bundle read (superdsc.run()). It
contains no hand-written analysis, field semantics, or recommendations — those
are intentionally left out so the report cannot drift from what a given run
actually observed. Interpretation is a separate, later deliverable.
"""

from __future__ import annotations

import datetime
from typing import Any

TICK = "✅"  # present on all relevant nodes/ops
PARTIAL = "◐"  # present on some (shown as n/total)
CROSS = "❌"  # measured absent
DASH = "➖"  # n/a: not generated yet, or carried only indirectly (other fields)

# Field sets, in the order issue #2574 lists them.
FX_FIELDS = [
    "stack_trace",
    "nn_module_stack",
    "source_fn_stack",
    "original_aten",
    "from_node",
]
IR_FIELDS = ["origins", "origin_node", "traceback", "get_stack_traces"]


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
    """Matrix cell for an IR field at one stage: ➖ if the attribute isn't on
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


def _fx_type(node: dict, field: str) -> str:
    rec = node.get("fields", {}).get(field)
    if not (isinstance(rec, dict) and rec.get("nonempty", False)):
        return CROSS
    return f"`{rec.get('type', '?')}`"


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------
def render(
    capture: dict[str, Any], bundles: dict[str, Any], model_name: str = "SimpleMLP"
) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pre = capture["stage2_pre_grad"]["nodes"]
    post = capture["stage2_post_grad"]["nodes"]
    passes_before = capture["stage3_passes"].get("before", [])
    looplevel = capture["stage4_looplevel"]["operations"]
    opspec_ops = capture["stage5_opspec"]["ops"]
    opspec_fields = capture["stage5_opspec"]["opspec_fields"] or []
    kernels = capture["stage6_kernels"]["kernels"]
    sdsc_present = bundles.get("provenance_present", False)
    sdsc_cell = TICK if sdsc_present else CROSS

    L: list[str] = []

    # Header --------------------------------------------------------------
    L += [
        f"# Provenance Audit: `{model_name}` — Metadata Across "
        "the Compilation Pipeline",
        "",
        f"> Generated: {now} &nbsp;|&nbsp; Issue: "
        "[torch-spyre#2574](https://github.com/torch-spyre/torch-spyre/issues/2574)",
        "",
        "Measured in-process during one cache-defeated `torch.compile` (compile-path "
        "objects only). This report is **measurement-only**; interpretation is a "
        "separate deliverable.",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| FX pre-grad compute nodes | {len(pre)} |",
        f"| FX post-grad compute nodes | {len(post)} |",
        f"| LoopLevelIR operations | {len(looplevel)} |",
        f"| OpSpec ops created | {len(opspec_ops)} |",
        f"| SuperDSC kernels | {len(kernels)} |",
        f"| `sdsc_*.json` files | {bundles.get('total_sdsc_files', 0)} |",
        f"| `OpSpec` declared fields | `{opspec_fields}` |",
        "",
    ]

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
        f"{CROSS} reachable here but measured **empty/absent** &nbsp; "
        f"{DASH} not applicable here (no such slot, or carried indirectly via "
        "other fields).",
        "",
        "Every column tests **population** (the field exists *and* carries "
        'non-empty content; `0` counts as content, `None`/`[]`/`{}`/`""` do '
        "not). These cells are measurements only; interpreting each absence is "
        "the separate analysis deliverable (`provenance_analysis.md`).",
        "",
        "The **Layer** column marks whether a field lives on the FX node (`FX`) "
        "or the IR `ComputedBuffer` (`IR`). The two IR columns are the same "
        "LoopLevelIR before and after the Spyre pre-scheduling passes: "
        "**LoopLevelIR (pre-pass)** is the lowered IR entering them, "
        "**LoopLevelIR (post-pass)** is after they mutate it in place (e.g. "
        "inserting `restickify` buffers). These map to issue #2574's "
        '"Inductor passes" → "LoopLevelIR".',
        "",
        "| Layer | Field | FX Graph (pre-grad) | FX Graph (post-grad) "
        "| LoopLevelIR (pre-pass) | LoopLevelIR (post-pass) | OpSpec "
        "| SuperDSC JSON |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in FX_FIELDS:
        pre_sym = _fx_symbol(pre, f)
        post_sym = _fx_symbol(post, f)
        # Absent pre-grad but present post-grad => generated in the grad/lowering
        # transition (original_aten, from_node) -> pre-grad is "not yet created".
        if pre_sym == CROSS and post_sym != CROSS:
            pre_sym = DASH
        # FX-meta is not a direct attribute of the IR objects, but it is not lost
        # there either — it rides in `origins` until the OpSpec drop -> ➖.
        L.append(
            f"| FX | `{f}` | {pre_sym} | {post_sym} | {DASH} | {DASH} | {DASH} "
            f"| {DASH} |"
        )
    for f in IR_FIELDS:
        passes_c = _ir_cell(passes_before, f)
        llir_c = _ir_cell(looplevel, f)
        # OpSpec/JSON have no provenance field, so an IR field present upstream is
        # genuinely not carried there -> ❌ (unless a Phase-2 field is declared).
        opspec_c = _opspec_symbol(opspec_ops, f, opspec_fields)
        sdsc_c = _sdsc_symbol(bundles, f)
        L.append(
            f"| IR | `{f}` | {DASH} | {DASH} | {passes_c} | {llir_c} "
            f"| {opspec_c} | {sdsc_c} |"
        )
    L.append("")

    # Stage 2 detail ------------------------------------------------------
    for label, nodes in [("pre-grad", pre), ("post-grad", post)]:
        L += [
            f"## Stage 2 — FX Graph ({label}): {len(nodes)} compute nodes",
            "",
            "Cell = observed `type` of the field, or ❌ if absent.",
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
    # Shown as two tables because the pre-scheduling passes can change the op
    # count (e.g. insert restickify buffers), so pre-pass and post-pass may differ.
    for stage_label, desc, ops_list in [
        (
            "Stage 3 — LoopLevelIR (pre-pass)",
            "The lowered IR entering the Spyre pre-scheduling passes.",
            passes_before,
        ),
        (
            "Stage 4 — LoopLevelIR (post-pass)",
            "The same IR after the pre-scheduling passes mutate it in place.",
            looplevel,
        ),
    ]:
        L += [
            f"## {stage_label}: {len(ops_list)} operations",
            "",
            desc,
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
    L += [
        f"## Stage 5 — OpSpec: {len(opspec_ops)} ops",
        "",
        f"`OpSpec` declared fields: `{opspec_fields}` — no provenance field. The "
        "`origins` below are what is *available on the input `ComputedBuffer`* at "
        "`create_op_spec`; the `OpSpec` object itself declares no field to hold "
        "them.",
        "",
        "| Spyre op | buffer | `origins` | `origin_node` |",
        "| --- | --- | --- | --- |",
    ]
    for o in opspec_ops:
        fields = o.get("fields", {})
        origins = (
            ", ".join(f"`{x['name']}`" for x in (fields.get("origins") or [])) or "—"
        )
        onode = fields.get("origin_node") or CROSS
        L.append(
            f"| `{o.get('op', '?')}` | `{o.get('name', '?')}` | {origins} "
            f"| {('`' + onode + '`') if onode != CROSS else CROSS} |"
        )
    L.append("")

    # Stage 6 detail ------------------------------------------------------
    L += [
        f"## Stage 6 — SuperDSC: {bundles.get('total_sdsc_files', 0)} "
        f"`sdsc_*.json` files ({len(kernels)} kernels)",
        "",
        f"Provenance field present in any emitted `sdsc_*.json`: {sdsc_cell}",
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
        prov = TICK if b.get("provenance_present") else CROSS
        L += [
            f"### `{name}`",
            "",
            f"- buffers ({len(ops)}): {', '.join(f'`{o}`' for o in ops) or '—'}",
            f"- fx origins: {', '.join(f'`{o}`' for o in origins) or '—'}",
            f"- kernel metadata: `{md}`",
            f"- `sdsc_*.json` files: {nfiles} &nbsp; provenance in JSON: {prov}",
            "",
        ]

    return "\n".join(L)
