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

"""Generates the counted-loop diagrams in docs/source/_static/images.

Each box is sized from its text with a generous per-character width, so the
text fits whatever font the viewer substitutes.  The output is a static SVG:
nothing is measured or laid out in the reader's browser.

Usage: python3 docs/tools/make_loop_diagrams.py
"""

import pathlib
from xml.sax.saxutils import escape

FONT = 14
CHAR_W = 8.6  # wider than any common sans-serif at 14px (monospace is 8.4)
PAD_X = 22
BOX_H = 40
OUT = pathlib.Path(__file__).resolve().parents[1] / "source" / "_static" / "images"

STYLE = (
    "text{font-family:'DejaVu Sans',Verdana,Helvetica,Arial,sans-serif;"
    f"font-size:{FONT}px;fill:#0b0b0b;}}"
    ".e{fill:none;stroke:#333;stroke-width:1.6;}"
    ".l{font-size:13px;fill:#333;}"
    ".box{fill:#e8eef7;stroke:#2c5aa0;stroke-width:1.5;}"
    ".dec{fill:#fdf2d0;stroke:#b7950b;stroke-width:1.5;}"
    ".out{fill:#d5f4e6;stroke:#1e8449;stroke-width:1.5;}"
    ".alt{fill:#fde2e4;stroke:#c0392b;stroke-width:1.5;}"
)


class Node:
    def __init__(self, text, cx, cy, kind="box"):
        self.text, self.cx, self.cy, self.kind = text, cx, cy, kind
        tw = len(text) * CHAR_W
        if kind == "dec":
            self.w, self.h = tw * 1.7 + 2 * PAD_X, BOX_H * 1.9
        else:
            self.w, self.h = tw + 2 * PAD_X, BOX_H

    def a(self, side):
        return {
            "t": (self.cx, self.cy - self.h / 2),
            "b": (self.cx, self.cy + self.h / 2),
            "l": (self.cx - self.w / 2, self.cy),
            "r": (self.cx + self.w / 2, self.cy),
        }[side]

    def svg(self):
        x, y, w, h = self.cx - self.w / 2, self.cy - self.h / 2, self.w, self.h
        if self.kind == "dec":
            pts = f"{self.cx},{y} {x + w},{self.cy} {self.cx},{y + h} {x},{self.cy}"
            shape = f'<polygon class="dec" points="{pts}"/>'
        else:
            shape = f'<rect class="{self.kind}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>'
        text = (
            f'<text x="{self.cx}" y="{self.cy + 5}" text-anchor="middle">'
            f"{escape(self.text)}</text>"
        )
        return shape + text


def edge(points, label=None, label_at=None):
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    out = f'<path class="e" d="{d}" marker-end="url(#arrow)"/>'
    if label:
        lx, ly = label_at
        out += (
            f'<text class="l" x="{lx}" y="{ly}" text-anchor="middle">'
            f"{escape(label)}</text>"
        )
    return out


def write(name, width, height, nodes, edges, title):
    body = "".join(n.svg() for n in nodes) + "".join(edges)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title><style>{STYLE}</style>"
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>{body}</svg>\n'
    )
    (OUT / name).write_text(svg)


def loop_overview():
    cx = 330
    a = Node("for_each_tile call", 150, 30)
    b = Node("span-overflow or allocator tiling", 500, 30)
    c = Node("ops with a LoopMember", cx, 120)
    d = Node("LoopNode in graph.operations", cx, 200)
    e = Node("pre-scheduling passes", cx, 280)
    f = Node("LoopSchedulerNode", cx, 360)
    g = Node("LoopSpec", cx, 440)
    h = Node("scf.for in bundle.mlir", cx, 520, "out")
    nodes = [a, b, c, d, e, f, g, h]
    ax, ay = a.a("b")
    bx, by = b.a("b")
    cx_, cy_ = c.a("t")
    edges = [
        edge(
            [(ax, ay), (ax, 75), (cx - 40, 75), (cx - 40, cy_)],
            "splice_while_loops",
            (150, 66),
        ),
        edge(
            [(bx, by), (bx, 75), (cx + 40, 75), (cx + 40, cy_)],
            "tile_groups",
            (500, 66),
        ),
        edge([c.a("b"), d.a("t")], "build nodes", (cx + 62, 163)),
        edge([d.a("b"), e.a("t")]),
        edge([e.a("b"), f.a("t")], "hand off", (cx + 42, 323)),
        edge([f.a("b"), g.a("t")], "SpyreKernel", (cx + 58, 403)),
        edge([g.a("b"), h.a("t")], "codegen", (cx + 42, 483)),
    ]
    write(
        "wsr/loop-node-pipeline.svg",
        680,
        560,
        nodes,
        edges,
        "From a lowering to a LoopNode, a LoopSchedulerNode, a LoopSpec and an scf.for",
    )


def for_each_tile_flow():
    cx = 260
    a = Node("for_each_tile WhileLoop", cx, 30)
    b = Node("bounded tile scan?", cx, 125, "dec")
    c = Node("splice the body", cx, 235)
    d = Node("synthesize dim hints", cx, 305)
    e = Node("build a LoopNode", cx, 375)
    f = Node("WhileLoop left?", cx, 470, "dec")
    g = Node("continue pre-scheduling", cx, 580, "out")
    h = Node("leave for upstream path", 640, 125, "alt")
    nodes = [a, b, c, d, e, f, g, h]
    left = min(b.a("l")[0], f.a("l")[0]) - 55
    edges = [
        edge([a.a("b"), b.a("t")]),
        edge([b.a("b"), c.a("t")], "yes", (cx + 20, 205)),
        edge([c.a("b"), d.a("t")]),
        edge([d.a("b"), e.a("t")]),
        edge([e.a("b"), f.a("t")]),
        edge([f.a("b"), g.a("t")], "no", (cx + 18, 550)),
        edge(
            [f.a("l"), (left, f.a("l")[1]), (left, b.a("l")[1]), b.a("l")],
            "yes (nested)",
            (left + 45, 300),
        ),
        edge([b.a("r"), h.a("l")], "no", ((b.a("r")[0] + h.a("l")[0]) / 2, 118)),
    ]
    write(
        "wsr/for-each-tile-lowering.svg",
        820,
        620,
        nodes,
        edges,
        "How splice_while_loops proves, splices and wraps a for_each_tile WhileLoop",
    )


if __name__ == "__main__":
    loop_overview()
    for_each_tile_flow()
