# Knowledge Graph Explorer

The tabs below show how concepts in the Torch-Spyre codebase connect.
Each view filters the graph to a specific domain so you can explore
without noise from unrelated subsystems.

```{raw} html
<div id="kg-tabs"></div>
<p id="kg-view-desc" style="margin: 0.4em 0 0.8em; font-size: 0.9em; color: #555;"></p>
<div style="display: flex; align-items: center; gap: 1em; margin-bottom: 0.5em; flex-wrap: wrap;">
  <input id="kg-search" type="text" placeholder="Search nodes..." style="padding: 5px 10px; border: 1px solid #ccc; border-radius: 4px; width: 220px; font-size: 0.9em;">
  <div id="kg-controls" style="display: flex; gap: 0.4em;">
    <button id="kg-fit" class="kg-ctrl-btn" type="button" title="Fit the graph to the viewport">Fit</button>
    <button id="kg-focus" class="kg-ctrl-btn" type="button" title="Dim everything except the selected node and its neighbors">Focus</button>
    <button id="kg-png" class="kg-ctrl-btn" type="button" title="Download the current view as a PNG">PNG</button>
    <button id="kg-reset" class="kg-ctrl-btn" type="button" title="Clear selection, search, and focus">Reset</button>
  </div>
  <div id="kg-legend" style="display: flex; flex-wrap: wrap; gap: 0.8em; font-size: 0.8em;"></div>
</div>
<div id="cy"></div>
<div id="kg-info">
  <em>Click a node to see its source location and connections. Double-click a node to jump straight to its code.</em>
</div>
<p id="kg-stats" style="font-size: 0.8em; color: #888; margin-top: 0.5em;"></p>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/dist/cytoscape.min.js" integrity="sha384-H3uzGzTfGHUAumB8+s4GEdfFwzAceN9wCCndN8AXubWKFIPuBSWKKtWDx7RhSf/z" crossorigin="anonymous"></script>
<script src="../_static/js/knowledge_graph.js"></script>
```

## Views

**Operations** — Each PyTorch op and its Spyre implementation path:
decomposition, lowering, custom op, CPU fallback, or direct eager
kernel. Use this view to check whether a specific op is supported and
how the backend handles it.

**Compiler Passes** — Pass groups and their constituent transformation
functions, laid out top-to-bottom in pipeline order.

**Architecture** — Module dependencies, class inheritance, and
dataclass definitions across the `torch_spyre` package.

**Configuration** — Environment variables and the modules that read
them, showing which runtime knobs control which subsystems.

## Navigation

- Switch views with the tab bar.
- Pan by dragging the background; zoom with the scroll wheel.
- **Click a node** to highlight its connections and see its source
  file, line number, and neighbors in the panel below.
- **Follow the source link** — the file:line shown in the panel is a
  clickable link straight to the defining code on GitHub, pinned to the
  commit the graph was built from.
- **Double-click a node** to jump directly to that code in a new tab.
- **Click a neighbor name** in the "Connected to" list to hop to that
  node without leaving the graph.
- **Focus** dims everything except the selected node and its immediate
  neighbors, so a dense view collapses to one concept and its edges.
- **Fit** re-frames the graph; **PNG** downloads the current view;
  **Reset** clears selection, search, and focus. Press **Esc** to clear
  the current selection.
- Type in the search box to filter nodes by name.
- **Deep links:** selecting a node updates the page URL (for example
  `…/explorer/index.html#ops/op::mm`). Copy that URL to link a
  teammate straight to a specific node and view.

For a deeper walkthrough of *why* this is useful and how each persona
gets value from it, see {doc}`using_the_explorer`.

## How the graph is built

A Sphinx extension runs `docs/source/_ext/extract_graph.py` at build
time. The script parses the torch-spyre source tree with Python's
`ast` module and writes a `graph.json` into `_static/js/`. Because
extraction is purely syntactic, no imports of `torch` or `torch_spyre`
are required.

The extractors cover:

- Op registration decorators (`@register_spyre_decomposition`,
  `@register_spyre_lowering`, `@torch.library.custom_op`,
  `register_fallback_default`, `register_torch_compile_kernel`)
- `Custom*Passes` class definitions and their pass function lists
- Class definitions with base classes
- `@dataclass`-decorated structs and their typed fields
- Intra-package import statements
- `os.environ` and `os.getenv` call sites
