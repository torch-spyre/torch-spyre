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

"""Interactive browser for an SDSC summary: kernels > ops > tensors.

Built on :func:`batch_summarize_sdsc.build_report`, so it shows exactly what
the static report shows, collapsed by default. Three panels stacked top to
bottom: a tree of kernels and their ops (with what the op is and its LX
occupancy); the selected op's tensor rows as a table; and a detail pane with
every field of the selected row, including the producer and consumer pointers
you can jump along.

Keys: ``d`` toggle detail columns, ``e``/``c`` expand/collapse all,
``/`` filter ops by name, ``n``/``p`` next/previous op, ``g`` go to the
selected tensor's consumer, ``b`` back to its producer, ``]``/``[`` grow or
shrink the bottom detail pane, ``z`` zoom the focused panel to the full screen
(again to restore), ``Tab`` moves focus between panels, ``s`` save an SVG
screenshot, ``q`` quit.

Usage:
  sdsc_tui.py [DIR] [--range LO-HI] [--log FILE] [--cost FILE]
              [--lx-capacity BYTES] [--screenshot FILE.svg]
``--screenshot`` renders headless (no terminal needed) and exits; the SVG is
what to attach to a PR or Slack.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
import regex as re
import sys
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_summarize_sdsc as core

try:
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical, VerticalScroll
    from textual.widgets import DataTable, Footer, Header, Input, Static, Tree
except ImportError as exc:  # pragma: no cover - dependency hint
    print(
        f"sdsc_tui needs textual ({exc}). Install with: "
        "uv pip install --python $(which python3) textual",
        file=sys.stderr,
    )
    sys.exit(1)

_DETAIL_COLS = [
    "what",
    "tensor",
    "role",
    "layout",
    "tile",
    "tile_size",
    "footprint",
    "address",
    "core_map",
    "format",
    "kernel",
]
_BASIC_COLS = ["tensor", "role", "layout", "tile", "tile_size", "address", "format"]
_POINTER_RE = re.compile(r"^(<-|->)\s+(\S+)\s+(sdsc_\d+|\.\.\.Relayout)")


def _one_line(s: str) -> str:
    return " ".join(s.split("\n"))


def _one_term_per_line(expr: str) -> str:
    """Break a sympy sum at top-level ``+``/``-`` so each additive term of a
    long objective sits on its own line (signs kept)."""
    out: list[str] = []
    depth = 0
    cur = ""
    for i, ch in enumerate(expr):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if depth == 0 and ch in "+-" and i > 0 and expr[i - 1] == " " and cur.strip():
            out.append(cur.rstrip())
            cur = ch
            continue
        cur += ch
    if cur.strip():
        out.append(cur.rstrip())
    return "\n".join(out)


class DetailPane(VerticalScroll):
    """The scrollable detail pane; containers refuse to maximize by default."""

    ALLOW_MAXIMIZE = True


class SdscBrowser(App):
    TITLE = "SDSC summary"
    CSS = """
    #body { height: 1fr; }
    #tree { height: 2fr; min-height: 6; border-bottom: solid $secondary; }
    #table { height: 2fr; min-height: 5; border-bottom: solid $secondary; }
    #detail { height: 12; min-height: 4; padding: 0 1; }
    #detail:focus { border-left: thick $accent; }
    #detail_text { width: 1fr; }
    #filter { dock: bottom; display: none; }
    #filter.shown { display: block; }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("d", "toggle_detail", "Detail cols"),
        Binding("e", "expand_all", "Expand"),
        Binding("c", "collapse_all", "Collapse"),
        Binding("/", "filter", "Filter"),
        Binding("n", "next_op", "Next op"),
        Binding("p", "prev_op", "Prev op"),
        Binding("g", "goto_consumer", "-> consumer"),
        Binding("b", "goto_producer", "<- producer"),
        Binding("s", "screenshot", "Screenshot"),
        Binding("]", "grow_detail", "Detail +"),
        Binding("[", "shrink_detail", "Detail -"),
        Binding("z", "zoom", "Zoom panel"),
    ]

    def __init__(self, report: dict, detail_rows: int = 12) -> None:
        super().__init__()
        self.detail_rows = max(4, detail_rows)
        self.report = report
        self.ops = report["ops"]
        self.records = core.build_records(report)
        self.by_op: dict[int, list[dict]] = {}
        for r in self.records:
            self.by_op.setdefault(r["op_idx"], []).append(r)
        self.detail = False
        self.current_op: int | None = None
        self.op_nodes: dict[int, object] = {}
        self.filter_text = ""

    # ------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Tree("kernels", id="tree")
            yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
            # A scrollable detail pane: Tab or click to focus, arrows / PageUp /
            # PageDown to scroll, z to zoom it to the full screen.
            with DetailPane(id="detail", can_focus=True):
                yield Static("Select an op in the tree.", id="detail_text")
        box = Input(
            placeholder="filter ops by name (Enter to keep, Esc to clear)", id="filter"
        )
        box.can_focus = False  # focusable only while shown (see action_filter)
        yield box
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"{self.report['base_dir']}  ({len(self.ops)} ops)"
        self._apply_detail_height()
        self._build_tree()
        self._configure_table()
        self.query_one("#tree", Tree).focus()

    def _op_label(self, op_idx: int) -> Text:
        op = self.ops[op_idx]
        what = _one_line(core._describe_op(op, op_idx, self.ops).split("\n")[0])
        label = Text()
        label.append(f"{core._sdsc_stem(op)} ", style="dim")
        label.append(f"{op['op_name']}", style="bold")
        label.append(f" ({op['cores']}c) ", style="dim")
        label.append(what)
        if op.get("lx_live_pct") is not None:
            pct = op["lx_live_pct"]
            style = "red" if pct > 90 else "yellow" if pct > 60 else "green"
            label.append(f"  LX {pct:.0f}%", style=style)
        obj = op.get("objective")
        if obj and obj.get("value_ns") is not None:
            label.append(f"  obj {obj['value_ns'] / 1000:.1f}us", style="magenta")
        cost = op.get("cost")
        if cost:
            # The op's HBM-share attribution of the kernel's one prediction;
            # the bundle total when the dump had no attribution table.
            us = cost.get("attributed_us")
            if us is None:
                us = cost.get("t_us")
            if us is not None:
                label.append(f"  {us:.1f}us", style="cyan")
        return label

    def _build_tree(self) -> None:
        tree = self.query_one("#tree", Tree)
        tree.clear()
        tree.root.expand()
        self.op_nodes = {}
        by_kernel: dict[str, list[int]] = {}
        for op_idx, op in enumerate(self.ops):
            if not op["tensors"]:
                continue
            if self.filter_text and self.filter_text not in op["op_name"].lower():
                continue
            by_kernel.setdefault(op.get("kernel", ""), []).append(op_idx)
        for kernel, idxs in by_kernel.items():
            dirname = Path(self.ops[idxs[0]]["file"]).parent.name
            peak = max(self.ops[i].get("lx_live_bytes") or 0 for i in idxs)
            knode = tree.root.add(
                Text.assemble(
                    (f"kernel {kernel}", "bold"),
                    (f"  {len(idxs)} ops, peak LX {core._bytes_label(peak)}", "dim"),
                    (f"\n{dirname[:60]}", "dim"),
                ),
                data=("kernel", kernel),
            )
            for op_idx in idxs:
                node = knode.add_leaf(self._op_label(op_idx), data=("op", op_idx))
                self.op_nodes[op_idx] = node
            knode.expand()

    def _cols(self) -> list[str]:
        cols = list(_DETAIL_COLS if self.detail else _BASIC_COLS)
        if len({op.get("kernel") for op in self.ops}) <= 1 and "kernel" in cols:
            cols.remove("kernel")
        return cols

    def _configure_table(self) -> None:
        table = self.query_one("#table", DataTable)
        table.clear(columns=True)
        for c in self._cols():
            table.add_column(_one_line(core._HEADERS[c]), key=c)
        if self.current_op is not None:
            self._fill_table(self.current_op)

    def _fill_table(self, op_idx: int) -> None:
        table = self.query_one("#table", DataTable)
        table.clear()
        cols = self._cols()
        for i, r in enumerate(self.by_op.get(op_idx, [])):
            cells = [Text(str(r[c])) for c in cols]
            table.add_row(*cells, key=f"{op_idx}:{i}", height=None)
        if table.row_count:
            table.move_cursor(row=0)
            self._show_detail(op_idx, 0)

    def _show_detail(self, op_idx: int, row: int) -> None:
        op = self.ops[op_idx]
        recs = self.by_op.get(op_idx, [])
        detail = self.query_one("#detail_text", Static)
        if not recs or row >= len(recs):
            detail.update("")
            return
        tensor = op["tensors"][row]
        r = recs[row]
        lines = Text()
        lines.append(f"{op['op_name']}  ", style="bold")
        lines.append(
            f"{core._sdsc_stem(op)}  {op['cores']} cores  kernel {op.get('kernel', '')}\n",
            style="dim",
        )
        lines.append(_one_line(core._describe_op(op, op_idx, self.ops)) + "\n")
        fields = [
            ("tensor", r["tensor"]),
            ("role", r["role"]),
            ("layout", r["layout"]),
            ("tile", f"{r['tile']}  ({r['tile_size']})"),
            ("LX footprint", r["footprint"] or "n/a (HBM)"),
            ("address", r["address"]),
            ("coreIdToWkSlice", r["core_map"] or "(single core)"),
            ("format", r["format"]),
            ("producer", tensor.get("producer") or "-"),
            ("consumer", tensor.get("consumer") or "-"),
        ]
        if op.get("lx_live_bytes") is not None:
            pct = op.get("lx_live_pct")
            fields.append(
                (
                    "LX live at this op",
                    f"{core._bytes_label(op['lx_live_bytes'])}"
                    + (
                        f"  ({pct:.0f}% of {core._bytes_label(op.get('lx_capacity'))})"
                        if pct is not None
                        else ""
                    ),
                )
            )
        if op.get("cost"):
            fields.append(("cost", _one_line(core._cost_label(op))))
        res = op.get("residency")
        if res:
            # The allocator's own words, unabbreviated: the table cell shows
            # only the head of the phrase, and the parenthetical is where the
            # reason usually says WHICH condition failed.
            where = "LX" if res.get("lx") else "HBM"
            if res.get("address") is not None:
                where += f" @ {res['address']}"
            if res.get("copy_of"):
                where += f", a copy of {res['copy_of']}"
            if not res.get("lx") and res.get("reason"):
                where += f" -- {res['reason']}"
            fields.append(("residency", where))
        obj = op.get("objective")
        if obj and obj.get("relayout_op"):
            ro = obj["relayout_op"]
            charge_text = (
                f"{ro['charge_ns'] / 1000:.2f} us for the copy this shuffle "
                f"materializes ({len(ro['fired'])} of {ro['n_copies']} copies of "
                f"{ro['source']} fired)"
            )
            fields.append(("relayout charge", charge_text))
            if obj.get("objective_ns") is not None:
                fields.append(
                    ("  kernel objective", f"{obj['objective_ns'] / 1000:.2f} us")
                )
            lines_out = [
                (
                    f"{t['value_ns'] / 1000:9.2f} us"
                    if t.get("value_ns") is not None
                    else f"{'?':>12}"
                )
                + f"   {t['term']}"
                for t in obj.get("terms") or []
            ]
            fields.append(
                (f"  copies of {ro['source']}", "\n".join(lines_out) or "(none)")
            )
        elif obj:
            if obj.get("value_ns") is not None:
                fields.append(
                    (
                        "objective term",
                        f"{obj['value_ns'] / 1000:.2f} us under the solved plan",
                    )
                )
            if (
                obj.get("flipped_ns") is not None
                and obj.get("value_ns") is not None
                and obj.get("resident") is not None
            ):
                other = "HBM" if obj["resident"] else "LX"
                delta = (obj["flipped_ns"] - obj["value_ns"]) / 1000
                fields.append(
                    (
                        f"  if {other} instead",
                        f"{obj['flipped_ns'] / 1000:.2f} us ({delta:+.2f} us)",
                    )
                )
            if obj.get("objective_ns") is not None:
                fields.append(
                    ("  kernel objective", f"{obj['objective_ns'] / 1000:.2f} us")
                )
            if obj.get("own_value_ns") is not None:
                own = f"{obj['own_value_ns'] / 1000:.2f} us"
                if (
                    obj.get("own_flipped_ns") is not None
                    and obj.get("resident") is not None
                ):
                    other = "HBM" if obj["resident"] else "LX"
                    own += f"  (if {other}, others fixed: {obj['own_flipped_ns'] / 1000:.2f} us)"
                fields.append(("  this op's terms", own))
            if obj.get("symbols"):
                fields.append(
                    (
                        "  symbols",
                        ", ".join(f"{k}={v}" for k, v in obj["symbols"].items()),
                    )
                )
            # What the division decision was made over. The first line is this
            # buffer's own candidates, the rest are the pairs each producer edge
            # could have stayed LX across -- the "why not more cores" answer the
            # objective's numbers alone cannot give.
            if obj.get("divisions"):
                head, *rest = obj["divisions"]
                fields.append(("  core division", head))
                for line in rest:
                    fields.append(("    gate", line))
            for rt in obj.get("relayout_terms") or []:
                val = rt.get("value_ns")
                fields.append(
                    (
                        "  relayout charge",
                        f"{rt.get('copy', '')}: {'resident' if rt.get('resident') else 'not fired'}"
                        + (f", {val / 1000:.2f} us" if val is not None else ""),
                    )
                )
            # Each of the op's terms, reduced to the op's own symbols (other
            # buffers' solved values substituted), with the term's value under
            # the plan and its change under the flip. Largest change first.
            other = (
                ("HBM" if obj.get("resident") else "LX")
                if obj.get("resident") is not None
                else None
            )
            lines_out = []
            for t in obj.get("terms") or []:
                v = t.get("value_ns")
                f = t.get("flipped_ns")
                head = f"{v / 1000:9.2f} us" if v is not None else f"{'?':>12}"
                if v is not None and f is not None and other is not None:
                    head += f"  ({f - v:+8.2f} us if {other})"
                lines_out.append(f"{head}   {t['term']}")
            label = (
                f"  terms mentioning this op "
                f"({obj.get('n_own_terms', 0)} of {obj.get('n_terms', 0)}; "
                f"other buffers at their solved values)"
            )
            fields.append((label, "\n".join(lines_out) or "(none)"))
        for k, v in fields:
            lines.append(f"  {k:<20}", style="cyan")
            if "\n" in str(v) or len(str(v)) > 100:
                lines.append("\n")
                for line in str(v).split("\n"):
                    lines.append(f"      {line}\n")
            else:
                lines.append(f"{v}\n")
        detail.update(lines)
        self.query_one("#detail", VerticalScroll).scroll_home(animate=False)

    # ------------------------------------------------------------ events
    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        data = event.node.data
        if data and data[0] == "op":
            self.current_op = data[1]
            self._fill_table(data[1])

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self.current_op is not None and event.cursor_row is not None:
            self._show_detail(self.current_op, event.cursor_row)

    def on_input_changed(self, event: Input.Changed) -> None:
        self.filter_text = event.value.strip().lower()
        self._build_tree()

    def _hide_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.remove_class("shown")
        box.can_focus = False
        self.query_one("#tree", Tree).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._hide_filter()

    def on_key(self, event) -> None:
        # Esc in the filter box clears the filter and returns to the tree.
        if event.key == "escape" and self.query_one("#filter", Input).has_focus:
            box = self.query_one("#filter", Input)
            box.value = ""
            self.filter_text = ""
            self._build_tree()
            self._hide_filter()
            event.stop()

    # ------------------------------------------------------------ actions
    def action_toggle_detail(self) -> None:
        self.detail = not self.detail
        self._configure_table()

    def action_expand_all(self) -> None:
        self.query_one("#tree", Tree).root.expand_all()

    def action_collapse_all(self) -> None:
        for child in self.query_one("#tree", Tree).root.children:
            child.collapse()

    def action_filter(self) -> None:
        box = self.query_one("#filter", Input)
        box.add_class("shown")
        box.can_focus = True
        box.focus()

    def _select_op(self, op_idx: int) -> None:
        node = self.op_nodes.get(op_idx)
        if node is None:
            return
        tree = self.query_one("#tree", Tree)
        node.parent.expand()
        tree.select_node(node)
        tree.scroll_to_node(node)
        self.current_op = op_idx
        self._fill_table(op_idx)

    def action_next_op(self) -> None:
        idxs = sorted(self.op_nodes)
        if not idxs:
            return
        later = [i for i in idxs if self.current_op is None or i > self.current_op]
        self._select_op(later[0] if later else idxs[0])

    def action_prev_op(self) -> None:
        idxs = sorted(self.op_nodes)
        if not idxs:
            return
        earlier = [
            i for i in idxs if self.current_op is not None and i < self.current_op
        ]
        self._select_op(earlier[-1] if earlier else idxs[-1])

    def _follow(self, pointer: str) -> None:
        m = _POINTER_RE.match(pointer or "")
        if not m:
            return
        _, op_name, stem = m.groups()
        start = self.current_op or 0
        order = (
            list(range(start + 1, len(self.ops)))
            if pointer.startswith("->")
            else list(range(start - 1, -1, -1))
        )
        for i in order:
            op = self.ops[i]
            if op["op_name"] == op_name and core._sdsc_stem(op) == stem:
                self._select_op(i)
                return

    def _selected_tensor(self) -> dict | None:
        if self.current_op is None:
            return None
        table = self.query_one("#table", DataTable)
        row = table.cursor_row
        tensors = self.ops[self.current_op]["tensors"]
        return tensors[row] if 0 <= row < len(tensors) else None

    def action_goto_consumer(self) -> None:
        t = self._selected_tensor()
        if t:
            self._follow(t.get("consumer", ""))

    def action_goto_producer(self) -> None:
        t = self._selected_tensor()
        if t:
            self._follow(t.get("producer", ""))

    def _apply_detail_height(self) -> None:
        self.query_one("#detail", VerticalScroll).styles.height = self.detail_rows

    def action_grow_detail(self) -> None:
        """Give the detail pane three more rows (the table and tree share the
        rest). Up to 60 rows."""
        self.detail_rows = min(60, self.detail_rows + 3)
        self._apply_detail_height()

    def action_shrink_detail(self) -> None:
        self.detail_rows = max(4, self.detail_rows - 3)
        self._apply_detail_height()

    def action_zoom(self) -> None:
        """Toggle the focused panel (tree, table or detail) to the full
        screen; press again to restore the three-panel layout."""
        if self.screen.maximized is not None:
            self.screen.minimize()
            return
        target = self.focused
        if target is None:
            return
        for wid in ("#tree", "#table", "#detail"):
            widget = self.query_one(wid)
            if target is widget or widget in target.ancestors_with_self:
                self.screen.maximize(widget)
                return

    def action_screenshot(self) -> None:
        path = self.save_screenshot()
        self.notify(f"saved {path}")


def _screenshot(app: SdscBrowser, out: Path, size: tuple[int, int]) -> None:
    async def run() -> None:
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            app.action_expand_all()
            app.detail = True
            app._configure_table()
            # A screenshot has no scrolling: give the detail pane room for
            # every field, including the objective block.
            app.detail_rows = max(app.detail_rows, 26)
            app._apply_detail_height()
            idxs = sorted(app.op_nodes)
            if idxs:
                app._select_op(idxs[min(1, len(idxs) - 1)])
            await pilot.pause()
            app.save_screenshot(filename=out.name, path=str(out.parent))

    asyncio.run(run())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "directory",
        nargs="?",
        help="artifact directory (default: newest /tmp/torchinductor_*)",
    )
    parser.add_argument("--range", metavar="LO-HI")
    parser.add_argument("--log", metavar="FILE")
    parser.add_argument("--cost", metavar="FILE")
    parser.add_argument(
        "--cost-expr", metavar="FILE", help="SPYRE_DUMP_COST_EXPR_FILE output"
    )
    parser.add_argument("--lx-capacity", type=int, default=core._LX_CAPACITY_DEFAULT)
    parser.add_argument(
        "--probe",
        metavar="FILE",
        help="LX residency probe JSON: the allocator's reason for every buffer "
        "it did not keep in LX (only for runs captured before PR #4738)",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="do not pick up cost_dump*.log / cost_expr*.jsonl / probe*.json "
        "next to the artifacts",
    )
    parser.add_argument(
        "--screenshot", metavar="FILE.svg", help="render headless to an SVG and exit"
    )
    parser.add_argument(
        "--size", default="200x80", help="terminal size for --screenshot, COLSxROWS"
    )
    parser.add_argument(
        "--detail-rows",
        type=int,
        default=12,
        help="starting height of the bottom detail pane in rows (default 12; "
        "resize with ] and [ while running)",
    )
    args = parser.parse_args()
    base = args.directory
    if not base:
        matches = sorted(
            glob.glob("/tmp/torchinductor_*"), key=os.path.getmtime, reverse=True
        )
        if not matches:
            sys.exit("no /tmp/torchinductor_* directory found")
        base = matches[0]
    index_range = None
    if args.range:
        lo, _, hi = args.range.partition("-")
        index_range = (int(lo), int(hi))
    cost_path = Path(args.cost).expanduser() if args.cost else None
    cost_expr_path = Path(args.cost_expr).expanduser() if args.cost_expr else None
    probe_path = Path(args.probe).expanduser() if args.probe else None
    if not args.no_auto:
        # Companion dumps by name next to the artifacts (sdsc capture puts
        # them there), unless given explicitly.
        cost_path = cost_path or core.find_companion(Path(base), "cost")
        cost_expr_path = cost_expr_path or core.find_companion(Path(base), "cost_expr")
        probe_path = probe_path or core.find_companion(Path(base), "probe")
    report = core.build_report(
        base,
        index_range=index_range,
        log_path=Path(args.log).expanduser() if args.log else None,
        cost_path=cost_path,
        lx_capacity=args.lx_capacity,
        cost_expr_path=cost_expr_path,
        probe_path=probe_path,
    )
    app = SdscBrowser(report, detail_rows=args.detail_rows)
    if args.screenshot:
        cols, _, rows = args.size.partition("x")
        _screenshot(app, Path(args.screenshot).expanduser(), (int(cols), int(rows)))
        print(f"Wrote {args.screenshot}")
        return
    app.run()


if __name__ == "__main__":
    main()
