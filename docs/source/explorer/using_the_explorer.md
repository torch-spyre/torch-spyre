# Using the knowledge graph explorer

The {doc}`index` is a generated view of the Torch-Spyre codebase. Sphinx
rebuilds it from the source tree on every documentation build, so it
matches the code at the commit it was built from. Every node records the
file and line it came from, and the panel below the graph links to that
location.

This page describes what you can do with it. The first half is for people
running models on Spyre. The second is for people working on the backend.

## What it is for

Torch-Spyre is a few hundred Python modules plus a C++ runtime. Most of
the useful facts are relationships: a PyTorch op is handled by a lowering,
that lowering is part of a pass group, and the pass group reads an
environment variable. These relationships are spread across decorators,
class definitions, and import statements, so reading them one file at a
time is slow. A diagram maintained by hand goes out of date as soon as
someone merges a PR.

The explorer avoids that. It reads the structure directly from the code
with Python's `ast` module at build time, so the graph shows the current
code rather than a description of it.

## For users

If you run models through `torch.compile`, the two most useful views are
Operations and Configuration.

### Checking whether an op is supported

Open the Operations view and search for the op, for example `mm` or
`softmax`. The node color and its outgoing edge show how the backend
handles it:

- Decomposition: the op is rewritten into smaller ops before it runs on
  the hardware.
- Lowering: the op maps to a Spyre kernel during compilation.
- Custom op: a hand-written Spyre operator implements it.
- CPU fallback (dashed red edge): the op runs on the host instead of the
  accelerator. Fallbacks cause graph breaks and slowdowns, so check this
  view when a model is slower than you expect.
- Eager kernel: the op has a direct implementation that also works outside
  `torch.compile`.

If an op is not in the graph, the backend has no registration for it. That
is a concrete answer, and a useful detail to include when you file an
issue.

### Finding the variable that controls a behavior

The Configuration view shows every environment variable the code reads
(`SENCORES`, `LX_PLANNING`, `TORCH_SPYRE_DEBUG`, and the rest) and the
module that reads it. To change runtime or compilation behavior, use this
view to find the variable, then follow the source link to the code that
reads it, where you can see the accepted values and the default.

### Sharing a node in a bug report

Selecting a node writes a link into the page URL. You can paste that link,
for example `.../explorer/index.html#ops/op::embedding`, into an issue,
and it opens on that node and view.

## For developers

For contributors, the explorer helps with two things: learning an
unfamiliar area, and checking what a change affects.

### Exploring an unfamiliar subsystem

The Architecture view shows module dependencies and class inheritance for
the `torch_spyre` package. When you start working in a subsystem you have
not seen before, find the class you were pointed to, turn on Focus to hide
everything else, and read its base classes and the modules that import it.
That is usually enough to decide where to set a breakpoint.

### Opening the source

Every node with a single definition site is a link. Click it to see the
file and line in the panel, or double-click to open that definition on
GitHub in a new tab. The link uses the commit the graph was built from, so
the line number stays correct even after the code moves.

### Checking what a change affects

Two relationships are worth tracing before you refactor:

- `imports` edges (Architecture view) show which modules depend on the one
  you are changing. Focus the module node to see the inbound edges.
- `inherits_from` edges show the subclasses a base-class change affects.

This does not replace running the tests, but it shows you where to look.

### Learning the op registration patterns

When you add an operation, the Operations view shows the ops already
implemented with each pattern. Filter to the op family you are working on
and open the existing decomposition, lowering, or custom-op definitions.
Use this with the {doc}`../compiler/adding_operations` guide: the guide
explains the three patterns, and the graph shows every op that uses each
one, with a link to its code.

### Viewing the compiler pipeline

The Compiler Passes view shows each `Custom*Passes` group and the pass
functions it runs, in pipeline order from top to bottom. Use it to see
where a transformation happens and what runs before it, before you add or
reorder a pass.

## Accuracy and limits

Three properties are worth relying on:

- The graph is rebuilt on every documentation build. A Sphinx extension
  runs `docs/source/_ext/extract_graph.py` and writes a fresh
  `graph.json`, so there is no stored copy that can go out of date.
- Extraction is syntactic. It parses the source with `ast` and imports
  nothing, so it runs without Spyre hardware or the C++ extensions and is
  not affected by import-time side effects.
- Source links are commit-pinned. They use the commit the graph was built
  from, and use the default branch only when the build runs outside a git
  checkout.

The graph covers registrations, class hierarchies, imports, dataclass
fields, and environment-variable reads. It does not trace runtime call
graphs or data flow, and it only parses the files listed in
`build_graph()`. If a new registration pattern or source file is not
showing up, the extractor needs updating. The {doc}`../contributing/index`
notes and the `check-docs` checklist explain how.

## See also

- {doc}`index`: the explorer, with the navigation reference.
- {doc}`../compiler/adding_operations`: the op registration patterns the
  Operations view lists.
- {doc}`../compiler/architecture`: the compilation pipeline the Compiler
  Passes view shows.
