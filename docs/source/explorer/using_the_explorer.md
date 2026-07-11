# Why the Knowledge Graph Explorer Is Useful

The {doc}`index` is a live map of the Torch-Spyre codebase. It is
regenerated from the source tree on every documentation build, so it
never drifts from the code the way a hand-drawn diagram would. Every
node knows the file and line it came from, and the panel below the
graph links straight there.

This page explains what you can actually *do* with it — first for
people running models on Spyre, then for people building the backend.

## The problem it solves

Torch-Spyre spans a few hundred Python modules plus a C++ runtime, and
the interesting facts are relationships: *this* PyTorch op is handled by
*that* lowering, which lives in *this* pass group, which reads *that*
environment variable. Those relationships are spread across decorators,
class definitions, and import statements. Reading them one file at a
time is slow, and any diagram that tries to summarize them goes stale
the moment someone lands a PR.

The explorer takes the opposite approach. Instead of describing the
structure in prose, it extracts it directly from the code with Python's
`ast` module at build time. What you see is what the code says right
now — down to the commit the docs were built from.

## For users

If you run models on Spyre through `torch.compile`, the two views that
pay off fastest are **Operations** and **Configuration**.

### "Is this op supported, and how?"

Open the **Operations** view and search for the op — say `mm` or
`softmax`. The color of the node and its outgoing edge tell you the
path the backend takes:

- **Decomposition** — the op is rewritten into smaller ops before it
  ever reaches the hardware.
- **Lowering** — the op maps to a Spyre kernel during compilation.
- **Custom op** — a hand-written Spyre operator backs it.
- **CPU fallback** (dashed red edge) — the op runs on the host, not on
  the accelerator. Fallbacks are where graph breaks and slowdowns come
  from, so this is the view to check when a model is slower than you
  expect.
- **Eager kernel** — the op has a direct implementation that also works
  outside `torch.compile`.

If an op isn't in the graph at all, the backend has no registration for
it yet. That is a concrete answer you can act on, and a good thing to
mention when you file an issue.

### "Which knob controls this behavior?"

The **Configuration** view shows every environment variable the code
reads (`SENCORES`, `LX_PLANNING`, `TORCH_SPYRE_DEBUG`, and the rest)
and the module that reads each one. When you want to change runtime or
compilation behavior, this tells you the exact variable and — via the
source link — the code that consumes it, so you can see what values it
accepts and what the default is.

### Reporting something upstream

Because selecting a node writes a shareable link into the URL, you can
paste "the `embedding` op falls back to CPU — see
`…/explorer/index.html#ops/op::embedding`" into an issue and the
maintainer lands on exactly the node you mean.

## For developers

If you contribute to Torch-Spyre, the explorer is most useful as an
*orientation and impact-analysis* tool.

### Getting oriented in an unfamiliar area

The **Architecture** view is a module dependency and class-inheritance
map of the whole `torch_spyre` package. When you are dropped into a
subsystem you have never touched, start here: find the class you were
told about, turn on **Focus** to hide everything else, and read just
its base classes and the modules that import it. That is usually enough
to know where to put a breakpoint.

### Jumping to the code

Every node that has a single definition site is a link. Click it and
read the panel for the file and line; **double-click** to open that
definition on GitHub in a new tab. The link is pinned to the commit the
graph was built from, so it points at the code as it actually was, not
at a line number that has since moved.

### Impact analysis before a change

Two relationships are worth tracing before you refactor:

- **`imports` edges** (Architecture view) — who depends on the module
  you are about to change. Focus the module node and the inbound edges
  are your blast radius.
- **`inherits_from` edges** — subclasses that a base-class change would
  ripple into.

It will not replace running the tests, but it tells you where to look.

### Learning the op registration patterns

When you add an operation, the **Operations** view is a catalog of
prior art. Filter to the op family you are implementing and click
through to the decomposition, lowering, or custom-op definitions that
already exist. Pair this with the {doc}`../compiler/adding_operations`
guide: the guide explains the three patterns, and the graph shows you
every op that currently uses each one, with a link to its code.

### Seeing the compiler pipeline

The **Compiler Passes** view lays out each `Custom*Passes` group and
the pass functions it runs, top to bottom in pipeline order. It is the
fastest way to answer "where in the pipeline does this transformation
happen, and what runs before it?" before you add or reorder a pass.

## How it stays honest

A few properties are worth trusting the explorer for — and a few limits
worth knowing.

- **Regenerated every build.** A Sphinx extension runs
  `docs/source/_ext/extract_graph.py` at build time and writes a fresh
  `graph.json`. There is no committed snapshot to fall out of date.
- **Purely syntactic.** Extraction parses the source with `ast` and
  imports nothing, so it works without Spyre hardware or the C++
  extensions, and it cannot be thrown off by import-time side effects.
- **Commit-pinned links.** Source links use the commit the graph was
  built from, falling back to the default branch only when the build
  runs outside a git checkout.

What it does **not** show: it captures registrations, class hierarchies,
imports, dataclass fields, and env-var reads — the structural surface.
It does not trace runtime call graphs or data flow, and it only parses
the file list wired into `build_graph()`. If a new registration pattern
or source file appears and the graph misses it, that is a signal the
extractor needs updating; the {doc}`../contributing/index` notes and
the `check-docs` maintenance checklist cover how.

## See also

- {doc}`index` — the explorer itself, with the navigation reference.
- {doc}`../compiler/adding_operations` — the op registration patterns
  the Operations view catalogs.
- {doc}`../compiler/architecture` — the compilation pipeline the
  Compiler Passes view lays out.
